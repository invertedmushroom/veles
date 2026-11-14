/**
 * @module SYS-KERNEL/idb-storage
 * Concrete IndexedDB adapter implementing `JournalStorageBackend`.
 * Enforces atomic multi-store `readwrite` transactions across `workspaces`,
 * `journal`, and `tree_manifests`, resolving strictly on `transaction.oncomplete`.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type {
  RunJournalRecord,
  Sha256Hex,
  WorkspaceId,
} from "../protocol/types.ts";
import { StorageQuotaError } from "./blobs.ts";
import type {
  JournalStorageBackend,
  WorkspaceMetadata,
} from "./journal.ts";

export const VELES_IDB_NAME = "veles_kernel_v1";
export const VELES_IDB_SCHEMA_VERSION = 1;

export const STORE_WORKSPACES = "workspaces" as const;
export const STORE_JOURNAL = "journal" as const;
export const STORE_TREE_MANIFESTS = "tree_manifests" as const;

export interface StoredTreeManifest {
  readonly workspaceId: WorkspaceId;
  readonly schemaVersion: 1;
  readonly fileTree: Record<string, Sha256Hex>;
}

export interface IDBRequestLike<T = unknown> {
  readonly result: T;
  readonly error: DOMException | Error | null;
  onsuccess: ((event: unknown) => void) | null;
  onerror: ((event: unknown) => void) | null;
}

export interface IDBObjectStoreLike {
  get(key: unknown): IDBRequestLike<unknown>;
  getAll(): IDBRequestLike<unknown[]>;
  put(value: unknown, key?: unknown): IDBRequestLike<unknown>;
  delete(key: unknown): IDBRequestLike<unknown>;
}

export interface IDBTransactionLike {
  readonly error: DOMException | Error | null;
  oncomplete: ((event: unknown) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onabort: ((event: unknown) => void) | null;
  objectStore(name: string): IDBObjectStoreLike;
  abort(): void;
}

export interface IDBDatabaseLike {
  readonly version: number;
  readonly objectStoreNames: { contains(name: string): boolean };
  createObjectStore(
    name: string,
    options?: { keyPath?: string | string[] },
  ): IDBObjectStoreLike;
  transaction(
    storeNames: readonly string[],
    mode: "readonly" | "readwrite",
  ): IDBTransactionLike;
  close(): void;
}

export interface IDBOpenDBRequestLike extends IDBRequestLike<IDBDatabaseLike> {
  onupgradeneeded: ((event: { oldVersion: number; newVersion: number | null }) => void) | null;
  onblocked: ((event: unknown) => void) | null;
}

export interface IDBFactoryLike {
  open(name: string, version?: number): IDBOpenDBRequestLike;
}

function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: string | number };
  return (
    e.name === "QuotaExceededError" ||
    e.code === "ERR_STORAGE_QUOTA_EXCEEDED" ||
    e.code === 22
  );
}

function normalizeIdbError(err: unknown, fallbackMessage: string): Error {
  if (err instanceof StorageQuotaError) return err;
  if (isQuotaError(err)) {
    return new StorageQuotaError();
  }
  if (err instanceof Error) return err;
  return new Error(fallbackMessage);
}

function awaitRequest<T>(req: IDBRequestLike<T>, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () =>
      reject(normalizeIdbError(req.error, `ERR_IDB_REQUEST_FAILED: ${label}`));
  });
}

function awaitTransactionComplete(
  tx: IDBTransactionLike,
  label: string,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () =>
      reject(normalizeIdbError(tx.error, `ERR_IDB_TX_ERROR: ${label}`));
    tx.onabort = () =>
      reject(normalizeIdbError(tx.error, `ERR_IDB_TX_ABORTED: ${label}`));
  });
}

function ensureSchemaStores(db: IDBDatabaseLike): void {
  if (!db.objectStoreNames.contains(STORE_WORKSPACES)) {
    db.createObjectStore(STORE_WORKSPACES, { keyPath: "workspaceId" });
  }
  if (!db.objectStoreNames.contains(STORE_JOURNAL)) {
    db.createObjectStore(STORE_JOURNAL, { keyPath: ["workspaceId", "seq"] });
  }
  if (!db.objectStoreNames.contains(STORE_TREE_MANIFESTS)) {
    db.createObjectStore(STORE_TREE_MANIFESTS, { keyPath: "workspaceId" });
  }
}

function isValidMetadataShape(raw: unknown): raw is WorkspaceMetadata {
  if (typeof raw !== "object" || raw === null) return false;
  const m = raw as Record<string, unknown>;
  return (
    typeof m.workspaceId === "string" &&
    typeof m.title === "string" &&
    typeof m.headSeq === "number" &&
    typeof m.updatedAtMs === "number"
  );
}

function filterWorkspaceJournal(
  allRecords: unknown[],
  workspaceId: WorkspaceId,
): RunJournalRecord[] {
  const filtered: RunJournalRecord[] = [];
  for (const item of allRecords) {
    if (typeof item !== "object" || item === null) continue;
    const rec = item as RunJournalRecord;
    if (rec.workspaceId !== workspaceId) continue;
    if (rec.schemaVersion !== 1) {
      throw new Error(
        `ERR_IDB_SCHEMA_VERSION_INCOMPATIBLE: Unsupported journal schemaVersion ${String(rec.schemaVersion)}`,
      );
    }
    filtered.push(rec);
  }
  filtered.sort((a, b) => a.seq - b.seq);
  return filtered;
}

export class IndexedDbJournalBackend implements JournalStorageBackend {
  private readonly factory: IDBFactoryLike;
  private readonly dbName: string;
  private readonly targetVersion: number;
  private dbPromise: Promise<IDBDatabaseLike> | null = null;

  constructor(
    factory?: IDBFactoryLike,
    dbName = VELES_IDB_NAME,
    targetVersion = VELES_IDB_SCHEMA_VERSION,
  ) {
    const resolvedFactory =
      factory ??
      (globalThis as unknown as { indexedDB?: IDBFactoryLike }).indexedDB;
    if (!resolvedFactory) {
      throw new Error(
        "ERR_IDB_UNAVAILABLE: indexedDB is not available in this runtime",
      );
    }
    this.factory = resolvedFactory;
    this.dbName = dbName;
    this.targetVersion = targetVersion;
  }

  private openDatabase(): Promise<IDBDatabaseLike> {
    if (this.dbPromise !== null) return this.dbPromise;
    this.dbPromise = new Promise<IDBDatabaseLike>((resolve, reject) => {
      let req: IDBOpenDBRequestLike;
      try {
        req = this.factory.open(this.dbName, this.targetVersion);
      } catch (err) {
        this.dbPromise = null;
        reject(normalizeIdbError(err, "ERR_IDB_OPEN_FAILED"));
        return;
      }
      req.onupgradeneeded = () => {
        ensureSchemaStores(req.result);
      };
      req.onblocked = () => {
        this.dbPromise = null;
        reject(new Error("ERR_IDB_UPGRADE_BLOCKED: Close older tabs to upgrade"));
      };
      req.onsuccess = () => {
        const db = req.result;
        if (db.version > this.targetVersion) {
          db.close();
          this.dbPromise = null;
          reject(
            new Error(
              `ERR_IDB_NEWER_VERSION: Stored version ${db.version} > supported ${this.targetVersion}`,
            ),
          );
          return;
        }
        resolve(db);
      };
      req.onerror = () => {
        this.dbPromise = null;
        reject(
          normalizeIdbError(
            req.error,
            "ERR_IDB_OPEN_FAILED: Could not open IndexedDB",
          ),
        );
      };
    });
    return this.dbPromise;
  }

  public async getMetadata(
    workspaceId: WorkspaceId,
  ): Promise<WorkspaceMetadata | null> {
    const db = await this.openDatabase();
    const tx = db.transaction([STORE_WORKSPACES], "readonly");
    const store = tx.objectStore(STORE_WORKSPACES);
    const raw = await awaitRequest(store.get(workspaceId), "getMetadata");
    await awaitTransactionComplete(tx, "getMetadata");
    if (raw === undefined || raw === null) return null;
    if (!isValidMetadataShape(raw)) {
      throw new Error("ERR_IDB_CORRUPT_METADATA: Invalid workspace metadata shape");
    }
    return { ...raw };
  }

  public async listWorkspaces(): Promise<WorkspaceMetadata[]> {
    const db = await this.openDatabase();
    const tx = db.transaction([STORE_WORKSPACES], "readonly");
    const store = tx.objectStore(STORE_WORKSPACES);
    const all = await awaitRequest(store.getAll(), "listWorkspaces");
    await awaitTransactionComplete(tx, "listWorkspaces");
    const result: WorkspaceMetadata[] = [];
    for (const item of all) {
      if (isValidMetadataShape(item)) {
        result.push({ ...item });
      }
    }
    result.sort((a, b) => a.workspaceId.localeCompare(b.workspaceId));
    return result;
  }

  public async deleteWorkspace(workspaceId: WorkspaceId): Promise<void> {
    const db = await this.openDatabase();
    const tx = db.transaction(
      [STORE_WORKSPACES, STORE_JOURNAL, STORE_TREE_MANIFESTS],
      "readwrite",
    );
    const txDone = awaitTransactionComplete(tx, "deleteWorkspace");
    try {
      const journalStore = tx.objectStore(STORE_JOURNAL);
      const all = await awaitRequest(journalStore.getAll(), "readJournalForDelete");
      const existing = filterWorkspaceJournal(all, workspaceId);
      for (const oldRec of existing) {
        journalStore.delete([workspaceId, oldRec.seq]);
      }
      tx.objectStore(STORE_WORKSPACES).delete(workspaceId);
      tx.objectStore(STORE_TREE_MANIFESTS).delete(workspaceId);
    } catch (err) {
      tx.abort();
      await txDone.catch(() => {});
      throw normalizeIdbError(err, "ERR_IDB_DELETE_WORKSPACE_FAILED");
    }
    await txDone;
  }

  public async getJournalRecords(
    workspaceId: WorkspaceId,
  ): Promise<RunJournalRecord[]> {
    const db = await this.openDatabase();
    const tx = db.transaction([STORE_JOURNAL], "readonly");
    const store = tx.objectStore(STORE_JOURNAL);
    const all = await awaitRequest(store.getAll(), "getJournalRecords");
    await awaitTransactionComplete(tx, "getJournalRecords");
    return filterWorkspaceJournal(all, workspaceId);
  }

  public async getFileTree(
    workspaceId: WorkspaceId,
  ): Promise<Record<string, Sha256Hex>> {
    const db = await this.openDatabase();
    const tx = db.transaction([STORE_TREE_MANIFESTS], "readonly");
    const store = tx.objectStore(STORE_TREE_MANIFESTS);
    const raw = (await awaitRequest(
      store.get(workspaceId),
      "getFileTree",
    )) as StoredTreeManifest | undefined;
    await awaitTransactionComplete(tx, "getFileTree");
    if (!raw) return {};
    if (raw.schemaVersion !== 1 || typeof raw.fileTree !== "object" || raw.fileTree === null) {
      throw new Error("ERR_IDB_CORRUPT_MANIFEST: Invalid tree manifest");
    }
    return { ...raw.fileTree };
  }

  public async commitAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    newRecord: RunJournalRecord | null,
    updatedFileTree: Record<string, Sha256Hex>,
  ): Promise<void> {
    const db = await this.openDatabase();
    const tx = db.transaction(
      [STORE_WORKSPACES, STORE_JOURNAL, STORE_TREE_MANIFESTS],
      "readwrite",
    );
    const txDone = awaitTransactionComplete(tx, "commitAtomic");

    try {
      const journalStore = tx.objectStore(STORE_JOURNAL);
      const all = await awaitRequest(journalStore.getAll(), "readJournalForCommit");
      const existing = filterWorkspaceJournal(all, workspaceId);

      if (newRecord !== null) {
        if (newRecord.seq !== existing.length + 1) {
          tx.abort();
          throw new Error(
            `ERR_NON_MONOTONIC_SEQ: Expected seq ${existing.length + 1}, got ${newRecord.seq}`,
          );
        }
        journalStore.put({ ...newRecord });
      }

      tx.objectStore(STORE_WORKSPACES).put({ ...metadata });
      const manifest: StoredTreeManifest = {
        workspaceId,
        schemaVersion: 1,
        fileTree: { ...updatedFileTree },
      };
      tx.objectStore(STORE_TREE_MANIFESTS).put(manifest);
    } catch (err) {
      tx.abort();
      await txDone.catch(() => {});
      throw normalizeIdbError(err, "ERR_IDB_COMMIT_FAILED");
    }

    await txDone;
  }

  public async replaceWorkspaceAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    journal: readonly RunJournalRecord[],
    fileTree: Record<string, Sha256Hex>,
  ): Promise<void> {
    const db = await this.openDatabase();
    const tx = db.transaction(
      [STORE_WORKSPACES, STORE_JOURNAL, STORE_TREE_MANIFESTS],
      "readwrite",
    );
    const txDone = awaitTransactionComplete(tx, "replaceWorkspaceAtomic");

    try {
      const journalStore = tx.objectStore(STORE_JOURNAL);
      const all = await awaitRequest(journalStore.getAll(), "readJournalForReplace");
      const existing = filterWorkspaceJournal(all, workspaceId);
      for (const oldRec of existing) {
        journalStore.delete([workspaceId, oldRec.seq]);
      }
      for (const rec of journal) {
        journalStore.put({ ...rec, workspaceId });
      }

      tx.objectStore(STORE_WORKSPACES).put({ ...metadata, workspaceId });
      const manifest: StoredTreeManifest = {
        workspaceId,
        schemaVersion: 1,
        fileTree: { ...fileTree },
      };
      tx.objectStore(STORE_TREE_MANIFESTS).put(manifest);
    } catch (err) {
      tx.abort();
      await txDone.catch(() => {});
      throw normalizeIdbError(err, "ERR_IDB_REPLACE_FAILED");
    }

    await txDone;
  }

  public async close(): Promise<void> {
    if (this.dbPromise === null) return;
    const db = await this.dbPromise;
    db.close();
    this.dbPromise = null;
  }
}
