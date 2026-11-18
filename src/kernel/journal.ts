/**
 * @module SYS-KERNEL/journal
 * Deep module owning durable workspace metadata, monotonic RunJournal,
 * content-addressed file manifests, and atomic workspace export/import.
 * Enforces !INVARIANT [KERN-INV-01], [KERN-INV-04], and Scenarios 5, 6, 7.
 */
import {
  type EventId,
  type JournalEventType,
  type OperationId,
  type RunId,
  type RunJournalRecord,
  type RunState,
  type Sha256Hex,
  type WorkspaceId,
} from "../protocol/types.ts";
import { isValidStateTransition } from "../protocol/validation.ts";
import {
  ContentAddressedBlobStore,
  StorageQuotaError,
  base64ToBytes,
  bytesToBase64,
  computeSha256Hex,
} from "./blobs.ts";
import type { LockProvider, WorkspaceLease } from "./lock.ts";

const BINARY_MIME_BY_EXT: Readonly<Record<string, string>> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
};

function inferBinaryMimeType(relPath: string): string | null {
  const lower = relPath.toLowerCase();
  const dotIdx = lower.lastIndexOf(".");
  if (dotIdx < 0) return null;
  return BINARY_MIME_BY_EXT[lower.slice(dotIdx)] ?? null;
}

function normalizeSafeWorkspacePath(rawPath: string): string {
  const trimmed = rawPath.trim();
  if (!trimmed || trimmed.startsWith("/") || trimmed.includes("\\")) {
    throw new Error(`ERR_UNSAFE_FILE_PATH: ${rawPath}`);
  }
  const segments = trimmed.split("/");
  for (const seg of segments) {
    if (!seg || seg === "." || seg === "..") {
      throw new Error(`ERR_UNSAFE_FILE_PATH: ${rawPath}`);
    }
  }
  return segments.join("/");
}

export interface WorkspaceMetadata {
  readonly workspaceId: WorkspaceId;
  readonly title: string;
  readonly headSeq: number;
  readonly activeRunId: RunId | null;
  readonly runState: RunState | null;
  readonly updatedAtMs: number;
  readonly historyBlobHashes?: readonly Sha256Hex[];
}

export interface WorkspaceArchiveBundle {
  readonly archiveVersion: 1;
  readonly exportedAtMs: number;
  readonly metadata: WorkspaceMetadata;
  readonly journal: readonly RunJournalRecord[];
  readonly fileTree: Readonly<Record<string, Sha256Hex>>;
  readonly blobsBase64: Readonly<Record<Sha256Hex, string>>;
}

export interface AppendJournalInput<TPayload = Record<string, unknown>> {
  readonly eventId: EventId;
  readonly runId: RunId;
  readonly type: JournalEventType;
  readonly nextState: RunState;
  readonly relatedOperationId?: OperationId | null;
  readonly payload: TPayload;
  readonly fileUpdates?: Readonly<Record<string, Uint8Array | string | null>>;
}

export interface JournalStorageBackend {
  getMetadata(workspaceId: WorkspaceId): Promise<WorkspaceMetadata | null>;
  listWorkspaces?(): Promise<WorkspaceMetadata[]>;
  deleteWorkspace?(workspaceId: WorkspaceId): Promise<void>;
  getJournalRecords(workspaceId: WorkspaceId): Promise<RunJournalRecord[]>;
  getFileTree(workspaceId: WorkspaceId): Promise<Record<string, Sha256Hex>>;
  commitAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    newRecord: RunJournalRecord | null,
    updatedFileTree: Record<string, Sha256Hex>,
  ): Promise<void>;
  replaceWorkspaceAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    journal: readonly RunJournalRecord[],
    fileTree: Record<string, Sha256Hex>,
  ): Promise<void>;
}

/**
 * In-memory transactional backend (local-substitutable stand-in for IndexedDB).
 */
export class InMemoryJournalBackend implements JournalStorageBackend {
  private readonly metaMap = new Map<WorkspaceId, WorkspaceMetadata>();
  private readonly journalMap = new Map<WorkspaceId, RunJournalRecord[]>();
  private readonly treeMap = new Map<WorkspaceId, Record<string, Sha256Hex>>();
  private failNextCommitWithQuota = false;

  public simulateQuotaFailureOnNextCommit(fail = true): void {
    this.failNextCommitWithQuota = fail;
  }

  public async getMetadata(workspaceId: WorkspaceId): Promise<WorkspaceMetadata | null> {
    const m = this.metaMap.get(workspaceId);
    return m ? { ...m } : null;
  }

  public async listWorkspaces(): Promise<WorkspaceMetadata[]> {
    return Array.from(this.metaMap.values())
      .map((m) => ({ ...m }))
      .sort((a, b) => a.workspaceId.localeCompare(b.workspaceId));
  }

  public async deleteWorkspace(workspaceId: WorkspaceId): Promise<void> {
    this.metaMap.delete(workspaceId);
    this.journalMap.delete(workspaceId);
    this.treeMap.delete(workspaceId);
  }

  public async getJournalRecords(workspaceId: WorkspaceId): Promise<RunJournalRecord[]> {
    return [...(this.journalMap.get(workspaceId) ?? [])];
  }

  public async getFileTree(workspaceId: WorkspaceId): Promise<Record<string, Sha256Hex>> {
    return { ...(this.treeMap.get(workspaceId) ?? {}) };
  }

  public async commitAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    newRecord: RunJournalRecord | null,
    updatedFileTree: Record<string, Sha256Hex>,
  ): Promise<void> {
    if (this.failNextCommitWithQuota) {
      this.failNextCommitWithQuota = false;
      throw new StorageQuotaError();
    }
    const existing = this.journalMap.get(workspaceId) ?? [];
    if (newRecord !== null) {
      if (newRecord.seq !== existing.length + 1) {
        throw new Error(
          `ERR_NON_MONOTONIC_SEQ: Expected seq ${existing.length + 1}, got ${newRecord.seq}`,
        );
      }
      this.journalMap.set(workspaceId, [...existing, newRecord]);
    } else if (!this.journalMap.has(workspaceId)) {
      this.journalMap.set(workspaceId, []);
    }
    this.metaMap.set(workspaceId, { ...metadata });
    this.treeMap.set(workspaceId, { ...updatedFileTree });
  }

  public async replaceWorkspaceAtomic(
    workspaceId: WorkspaceId,
    metadata: WorkspaceMetadata,
    journal: readonly RunJournalRecord[],
    fileTree: Record<string, Sha256Hex>,
  ): Promise<void> {
    if (this.failNextCommitWithQuota) {
      this.failNextCommitWithQuota = false;
      throw new StorageQuotaError();
    }
    this.metaMap.set(workspaceId, { ...metadata });
    this.journalMap.set(workspaceId, [...journal]);
    this.treeMap.set(workspaceId, { ...fileTree });
  }
}

function extractIntentTargetPath(payload: Record<string, unknown>): string | null {
  const args =
    typeof payload.args === "object" && payload.args !== null
      ? (payload.args as Record<string, unknown>)
      : null;
  if (!args) return null;
  for (const key of ["path", "outputPath", "saveToPath", "saveResultToPath"]) {
    if (typeof args[key] === "string" && args[key].trim()) {
      return normalizeSafeWorkspacePath(args[key]);
    }
  }
  return null;
}

const READ_ONLY_FILE_TOOLS = new Set([
  "workspace.read_file",
  "workspace.inspect_image",
  "workspace.list_files",
]);

function isReadOnlyFileTool(toolName: unknown): boolean {
  return typeof toolName === "string" && READ_ONLY_FILE_TOOLS.has(toolName);
}

function addObservedPathsToSet(
  observed: unknown,
  targetSet: Set<string>,
): void {
  if (typeof observed !== "object" || observed === null) return;
  for (const [p, h] of Object.entries(observed as Record<string, unknown>)) {
    if (typeof h === "string") targetSet.add(p);
  }
}

function collectRunTouchedPaths(
  journal: readonly RunJournalRecord[],
  targetRunId: RunId,
): Set<string> {
  const targetRuns = new Set<RunId>([targetRunId]);
  for (const rec of journal) {
    if (
      rec.type === "RunCreated" &&
      typeof rec.payload.parentRunId === "string" &&
      targetRuns.has(rec.payload.parentRunId)
    ) {
      targetRuns.add(rec.runId);
    }
  }
  const touched = new Set<string>();
  for (const rec of journal) {
    if (!targetRuns.has(rec.runId) || isReadOnlyFileTool(rec.payload.toolName)) {
      continue;
    }
    addObservedPathsToSet(rec.payload.observedFileHashes, touched);
    const intentPath = extractIntentTargetPath(rec.payload);
    if (intentPath) touched.add(intentPath);
  }
  return touched;
}

function applyRecordToPriorHashes(
  rec: RunJournalRecord,
  prior: Record<string, Sha256Hex>,
): void {
  if (
    rec.type === "CheckpointCommitted" &&
    typeof rec.payload.uploadedPath === "string" &&
    typeof rec.payload.sha256 === "string"
  ) {
    prior[rec.payload.uploadedPath] = rec.payload.sha256;
  }
  if (rec.type === "ToolResultCommitted") {
    const obs = rec.payload.observedFileHashes;
    if (typeof obs === "object" && obs !== null) {
      for (const [p, h] of Object.entries(obs as Record<string, unknown>)) {
        if (typeof h === "string") prior[p] = h;
      }
    }
  }
  if (
    rec.type === "ToolIntentLogged" &&
    rec.payload.toolName === "workspace.delete_file"
  ) {
    const delPath = extractIntentTargetPath(rec.payload);
    if (delPath) delete prior[delPath];
  }
}

function copyTreeHashes(
  rawTree: unknown,
  target: Record<string, Sha256Hex>,
): void {
  if (typeof rawTree !== "object" || rawTree === null) return;
  for (const [p, h] of Object.entries(rawTree as Record<string, unknown>)) {
    if (typeof h === "string") target[p] = h;
  }
}

function buildPriorFileHashes(
  journal: readonly RunJournalRecord[],
  createdIdx: number,
): Record<string, Sha256Hex> {
  const revertedRuns = new Set<string>();
  for (const rec of journal) {
    if (
      typeof rec.payload.deletedRunId === "string" &&
      Array.isArray(rec.payload.revertedFiles) && rec.payload.revertedFiles.length > 0
    ) {
      revertedRuns.add(rec.payload.deletedRunId);
    }
  }
  const prior: Record<string, Sha256Hex> = {};
  const preTree = journal[createdIdx]?.payload.preRunFileTree;
  if (revertedRuns.size === 0 && typeof preTree === "object" && preTree !== null) {
    copyTreeHashes(preTree, prior);
    return prior;
  } else {
    const firstCreated = journal.find((r) => r.type === "RunCreated");
    copyTreeHashes(firstCreated?.payload.preRunFileTree, prior);
  }
  for (let i = 0; i < createdIdx; i++) {
    const rec = journal[i]!;
    if (!revertedRuns.has(rec.runId)) {
      applyRecordToPriorHashes(rec, prior);
    }
  }
  return prior;
}

function computeRevertedFileTreeForRun(
  currentTree: Record<string, Sha256Hex>,
  journal: readonly RunJournalRecord[],
  targetRunId: RunId,
): { readonly tree: Record<string, Sha256Hex>; readonly revertedPaths: string[] } {
  const createdIdx = journal.findIndex(
    (r) => r.runId === targetRunId && r.type === "RunCreated",
  );
  if (createdIdx === -1) {
    return { tree: currentTree, revertedPaths: [] };
  }
  const touched = collectRunTouchedPaths(journal, targetRunId);
  const prior = buildPriorFileHashes(journal, createdIdx);
  const nextTree: Record<string, Sha256Hex> = { ...currentTree };
  for (const path of touched) {
    if (normalizeSafeWorkspacePath(path) !== path) throw new Error(`ERR_UNSAFE_FILE_PATH: ${path}`);
    if (prior[path] !== undefined) {
      nextTree[path] = prior[path]!;
    } else {
      delete nextTree[path];
    }
  }
  return { tree: nextTree, revertedPaths: Array.from(touched).sort() };
}

export class WorkspaceKernelStore {
  private readonly commitListeners = new Set<(record: RunJournalRecord) => void>();

  public onCommit(listener: (record: RunJournalRecord) => void): () => void {
    this.commitListeners.add(listener);
    return () => { this.commitListeners.delete(listener); };
  }

  private readonly journalBackend: JournalStorageBackend;
  private readonly blobStore: ContentAddressedBlobStore;
  private readonly lockProvider?: LockProvider;
  private lease: WorkspaceLease;
  private activeWorkspaceId: WorkspaceId;
  private leaseReleased = false;
  private leaseAcquisition: Promise<WorkspaceLease> | undefined;
  private commitQueue: Promise<void> = Promise.resolve();

  constructor(
    journalBackend: JournalStorageBackend,
    blobStore: ContentAddressedBlobStore,
    lease: WorkspaceLease,
    lockProvider?: LockProvider,
  ) {
    this.journalBackend = journalBackend;
    this.blobStore = blobStore;
    this.lease = lease;
    this.lockProvider = lockProvider;
    this.activeWorkspaceId = lease.workspaceId;
  }

  private enqueueCommit<T>(task: () => Promise<T>): Promise<T> {
    const next = this.commitQueue.then(task, task);
    this.commitQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  public getActiveWorkspaceId(): WorkspaceId {
    return this.activeWorkspaceId;
  }

  private assertLeaderWriteAuthority(): void {
    if (this.getLeaseMode() !== "ReadWrite_Leader") {
      throw new Error(
        "ERR_WORKSPACE_LOCKED_BY_OTHER_TAB: Secondary observer tab cannot mutate workspace state",
      );
    }
  }

  private async reacquireWorkspaceLease(targetWorkspaceId: WorkspaceId): Promise<void> {
    if (targetWorkspaceId === this.lease.workspaceId && !this.leaseReleased) {
      this.activeWorkspaceId = targetWorkspaceId;
      return;
    }
    if (!this.lockProvider) throw new Error("ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED");
    const previousLease = this.lease;
    const wasReleased = this.leaseReleased;
    const nextLease = await this.lockProvider.acquireWorkspaceLease(
      targetWorkspaceId,
      previousLease.holderId,
    );
    if (!wasReleased) {
      await previousLease.release();
    }
    this.lease = nextLease;
    this.leaseReleased = false;
    this.activeWorkspaceId = targetWorkspaceId;
  }

  public async releaseWorkspaceLease(): Promise<void> {
    if (this.leaseReleased) return;
    this.leaseReleased = true;
    await this.lease.release();
  }

  private async reacquireReleasedLease(): Promise<void> {
    if (!this.leaseReleased) return;
    if (!this.lockProvider) throw new Error("ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED");
    const acquisition = this.leaseAcquisition ??= this.lockProvider.acquireWorkspaceLease(
      this.activeWorkspaceId, this.lease.holderId,
    );
    try {
      this.lease = await acquisition;
      this.leaseReleased = false;
    } finally {
      if (this.leaseAcquisition === acquisition) this.leaseAcquisition = undefined;
    }
  }

  public async ensureWorkspaceLease(title?: string): Promise<WorkspaceMetadata> {
    await this.reacquireReleasedLease();
    const resolvedTitle = title?.trim() || this.activeWorkspaceId;
    if (this.lease.mode === "ReadOnly_Observer") {
      const existing = await this.journalBackend.getMetadata(this.activeWorkspaceId);
      if (existing) return existing;
      return {
        workspaceId: this.activeWorkspaceId,
        title: resolvedTitle,
        headSeq: 0,
        activeRunId: null,
        runState: null,
        updatedAtMs: Date.now(),
      };
    }
    return this.initWorkspace(resolvedTitle);
  }

  public async forkForWorkspace(
    workspaceId: WorkspaceId,
    title?: string,
  ): Promise<WorkspaceKernelStore> {
    const cleanId = workspaceId.trim();
    if (!cleanId) {
      throw new Error("ERR_INVALID_WORKSPACE_ID: Workspace ID cannot be empty");
    }
    if (cleanId === this.activeWorkspaceId) {
      await this.ensureWorkspaceLease(title);
      return this;
    }
    if (!this.lockProvider) throw new Error("ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED");
    const nextLease = await this.lockProvider.acquireWorkspaceLease(cleanId, this.lease.holderId);
    const forked = new WorkspaceKernelStore(
      this.journalBackend,
      this.blobStore,
      nextLease,
      this.lockProvider,
    );
    await forked.ensureWorkspaceLease(title);
    return forked;
  }

  public async deleteWorkspaceStorage(workspaceId: WorkspaceId): Promise<void> {
    this.assertLeaderWriteAuthority();
    if (workspaceId !== this.activeWorkspaceId) {
      throw new Error("ERR_WORKSPACE_AUTHORITY_MISMATCH: Deletion requires the target workspace's own lease");
    }
    if (this.journalBackend.deleteWorkspace) {
      await this.journalBackend.deleteWorkspace(workspaceId);
    }
  }

  public async initWorkspace(title: string): Promise<WorkspaceMetadata> {
    this.assertLeaderWriteAuthority();
    const existing = await this.journalBackend.getMetadata(this.activeWorkspaceId);
    if (existing !== null) return existing;

    const initialMeta: WorkspaceMetadata = {
      workspaceId: this.activeWorkspaceId,
      title,
      headSeq: 0,
      activeRunId: null,
      runState: null,
      updatedAtMs: Date.now(),
      historyBlobHashes: [],
    };
    await this.journalBackend.commitAtomic(this.activeWorkspaceId, initialMeta, null, {});
    return initialMeta;
  }

  public async listWorkspaces(): Promise<WorkspaceMetadata[]> {
    if (this.journalBackend.listWorkspaces) {
      const list = await this.journalBackend.listWorkspaces();
      if (list.length > 0) return list;
    }
    const current = await this.getMetadata();
    return current ? [current] : [];
  }

  public async switchWorkspace(
    workspaceId: WorkspaceId,
    title?: string,
  ): Promise<WorkspaceMetadata> {
    this.assertLeaderWriteAuthority();
    await this.assertNoActiveInFlightRun();
    const cleanId = workspaceId.trim();
    if (!cleanId) {
      throw new Error("ERR_INVALID_WORKSPACE_ID: Workspace ID cannot be empty");
    }
    await this.reacquireWorkspaceLease(cleanId);
    if (this.lease.mode === "ReadOnly_Observer") {
      const existing = await this.journalBackend.getMetadata(cleanId);
      if (existing) return existing;
      return {
        workspaceId: cleanId,
        title: title?.trim() || cleanId,
        headSeq: 0,
        activeRunId: null,
        runState: null,
        updatedAtMs: Date.now(),
      };
    }
    return this.initWorkspace(title?.trim() || cleanId);
  }

  public async deleteWorkspace(workspaceId: WorkspaceId): Promise<WorkspaceMetadata> {
    this.assertLeaderWriteAuthority();
    if (workspaceId !== this.activeWorkspaceId) throw new Error("ERR_WORKSPACE_AUTHORITY_MISMATCH");
    if (!this.lockProvider) throw new Error("ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED");
    await this.assertNoActiveInFlightRun();
    await this.deleteWorkspaceStorage(workspaceId);
    const remaining = await this.listWorkspaces();
    const nextWs = remaining.find((w) => w.workspaceId !== workspaceId);
    if (nextWs) {
      await this.reacquireWorkspaceLease(nextWs.workspaceId);
      return nextWs;
    }
    const fallbackId = workspaceId === "ws-alpha" ? "ws-default" : "ws-alpha";
    await this.reacquireWorkspaceLease(fallbackId);
    return this.initWorkspace("Alpha");
  }

  public async writeWorkspaceFile(
    relPath: string,
    content: string | Uint8Array,
  ): Promise<void> {
    this.assertLeaderWriteAuthority();
    const normalizedPath = normalizeSafeWorkspacePath(relPath);
    return this.enqueueCommit(async () => {
      const wsId = this.activeWorkspaceId;
      const meta = await this.journalBackend.getMetadata(wsId);
      if (!meta) {
        throw new Error(`ERR_WORKSPACE_NOT_FOUND: ${wsId}`);
      }
      const { tree, historyBlobHashes } = await this.stageFileUpdates(
        wsId, { [normalizedPath]: content }, meta.historyBlobHashes,
      );
      const nextSeq = meta.headSeq + 1;
      const now = Date.now();
      const record: RunJournalRecord = {
        eventId: `upload-${wsId}-${nextSeq}`,
        workspaceId: wsId,
        runId: meta.activeRunId ?? "workspace-upload",
        seq: nextSeq,
        type: "CheckpointCommitted",
        schemaVersion: 1,
        timestampMs: now,
        relatedOperationId: null,
        committedState: meta.runState ?? "completed",
        payload: {
          uploadedPath: normalizedPath,
          sha256: tree[normalizedPath],
          byteLength:
            typeof content === "string"
              ? new TextEncoder().encode(content).byteLength
              : content.byteLength,
        },
      };
      const nextMeta: WorkspaceMetadata = {
        ...meta,
        headSeq: nextSeq,
        updatedAtMs: now,
        historyBlobHashes,
      };
      await this.journalBackend.commitAtomic(wsId, nextMeta, record, tree);
    });
  }

  public async recordSessionConfiguration(
    sessionId: string,
    configPayload: Record<string, unknown>,
  ): Promise<void> {
    if (this.lease.mode !== "ReadWrite_Leader") return;
    return this.enqueueCommit(async () => {
      const wsId = this.activeWorkspaceId;
      const meta = await this.journalBackend.getMetadata(wsId);
      if (!meta) return;
      const tree = await this.journalBackend.getFileTree(wsId);
      const nextSeq = meta.headSeq + 1;
      const now = Date.now();
      const record: RunJournalRecord = {
        eventId: `session-cfg-${wsId}-${nextSeq}`,
        workspaceId: wsId,
        runId: `session:${sessionId}`,
        seq: nextSeq,
        type: "CheckpointCommitted",
        schemaVersion: 1,
        timestampMs: now,
        relatedOperationId: null,
        committedState: meta.runState ?? "completed",
        payload: {
          ...configPayload,
          sessionConfigured: true,
          sessionId,
        },
      };
      const nextMeta: WorkspaceMetadata = {
        ...meta,
        headSeq: nextSeq,
        updatedAtMs: now,
      };
      await this.journalBackend.commitAtomic(wsId, nextMeta, record, tree);
    });
  }

  /**
   * Appends a tombstone journal record deleting a chat turn (`runId`) from
   * `ContextBuilder` message reconstruction, and optionally reverts any workspace
   * files created or modified by `runId` back to their pre-turn SHA-256 blobs.
   */
  public async deleteRun(
    runId: RunId,
    options?: { readonly revertFiles?: boolean },
  ): Promise<void> {
    this.assertLeaderWriteAuthority();
    return this.enqueueCommit(async () => {
      const wsId = this.activeWorkspaceId;
      const meta = await this.journalBackend.getMetadata(wsId);
      if (!meta) return;
      if (meta.activeRunId === runId) {
        await this.assertNoActiveInFlightRun();
      }
      const currentTree = await this.journalBackend.getFileTree(wsId);
      const journal = options?.revertFiles
        ? await this.journalBackend.getJournalRecords(wsId)
        : [];
      const { tree, revertedPaths } = options?.revertFiles
        ? computeRevertedFileTreeForRun(currentTree, journal, runId)
        : { tree: currentTree, revertedPaths: [] as string[] };
      await this.verifyRevertHashes(
        revertedPaths.flatMap((path) => tree[path] ? [tree[path]!] : []),
        meta.historyBlobHashes ?? Object.values(currentTree),
      );
      const nextSeq = meta.headSeq + 1;
      const now = Date.now();
      const isClearingActive = meta.activeRunId === runId;
      const record: RunJournalRecord = {
        eventId: `delete-run-${wsId}-${nextSeq}`,
        workspaceId: wsId,
        runId: `delete:${runId}`,
        seq: nextSeq,
        type: "CheckpointCommitted",
        schemaVersion: 1,
        timestampMs: now,
        relatedOperationId: null,
        committedState: isClearingActive
          ? "completed"
          : (meta.runState ?? "completed"),
        payload: {
          deletedRunId: runId,
          ...(revertedPaths.length > 0 ? { revertedFiles: revertedPaths } : {}),
        },
      };
      const nextMeta: WorkspaceMetadata = {
        ...meta,
        headSeq: nextSeq,
        activeRunId: isClearingActive ? null : meta.activeRunId,
        runState: isClearingActive ? null : meta.runState,
        updatedAtMs: now,
      };
      await this.journalBackend.commitAtomic(wsId, nextMeta, record, tree);
    });
  }

  private async verifyRevertHashes(
    requestedHashes: readonly Sha256Hex[],
    ownedHashes: readonly Sha256Hex[],
  ): Promise<void> {
    const owned = new Set(ownedHashes);
    for (const hash of new Set(requestedHashes)) {
      if (!owned.has(hash)) {
        throw new Error("ERR_REVERT_BLOB_UNAVAILABLE: This workspace archive does not contain the requested file history");
      }
      try {
        await this.blobStore.getBytesVerified(hash);
      } catch (err) {
        throw new Error("ERR_REVERT_BLOB_UNAVAILABLE: Historical file bytes are missing or corrupted", { cause: err });
      }
    }
  }

  private async assertNoActiveInFlightRun(): Promise<void> {
    const current = await this.getMetadata();
    const state = current?.runState;
    if (
      state &&
      state !== "completed" &&
      state !== "failed" &&
      state !== "cancelled" &&
      state !== "interrupted"
    ) {
      throw new Error(
        `ERR_WORKSPACE_RUN_ACTIVE: Cannot switch or delete workspace while run '${current.activeRunId ?? ""}' is in state '${state}'`,
      );
    }
  }

  private async resolvePriorRunState(
    wsId: WorkspaceId,
    meta: WorkspaceMetadata,
    runId: RunId,
  ): Promise<RunState | null> {
    if (meta.activeRunId === runId && meta.runState !== null) {
      return meta.runState;
    }
    const records = await this.journalBackend.getJournalRecords(wsId);
    for (let i = records.length - 1; i >= 0; i--) {
      const rec = records[i]!;
      if (rec.runId === runId && !rec.eventId.startsWith("upload-")) {
        return rec.committedState;
      }
    }
    return null;
  }

  /**
   * Atomically writes any updated files to content-addressed OPFS blobs first,
   * validates state transition legality, and commits the monotonic journal event.
   */
  public async appendEvent(input: AppendJournalInput): Promise<RunJournalRecord> {
    this.assertLeaderWriteAuthority();
    return this.enqueueCommit(() => this.appendEventInternal(input));
  }

  private async appendEventInternal(input: AppendJournalInput): Promise<RunJournalRecord> {
    const wsId = this.activeWorkspaceId;
    const meta = await this.journalBackend.getMetadata(wsId);
    if (meta === null) {
      throw new Error(`ERR_WORKSPACE_NOT_FOUND: ${wsId}`);
    }

    const priorState = await this.resolvePriorRunState(wsId, meta, input.runId);
    if (priorState === null) {
      if (input.nextState !== "created" && input.nextState !== "preparing") {
        throw new Error(`ERR_INVALID_INITIAL_RUN_STATE: ${input.nextState}`);
      }
    } else if (
      priorState !== input.nextState &&
      !isValidStateTransition(priorState, input.nextState)
    ) {
      throw new Error(
        `ERR_ILLEGAL_STATE_TRANSITION: Cannot transition from ${priorState} to ${input.nextState}`,
      );
    }

    // Step 1: Write content-addressed blobs BEFORE committing the journal transaction
    const { tree, historyBlobHashes } = await this.stageFileUpdates(
      wsId, input.fileUpdates, meta.historyBlobHashes,
    );

    const nextSeq = meta.headSeq + 1;
    const now = Date.now();
    const record: RunJournalRecord = {
      eventId: input.eventId,
      workspaceId: wsId,
      runId: input.runId,
      seq: nextSeq,
      type: input.type,
      schemaVersion: 1,
      timestampMs: now,
      relatedOperationId: input.relatedOperationId ?? null,
      committedState: input.nextState,
      payload: input.payload,
    };

    const nextMeta: WorkspaceMetadata = {
      ...meta,
      headSeq: nextSeq,
      activeRunId: input.runId,
      runState: input.nextState,
      updatedAtMs: now,
      historyBlobHashes,
    };

    // Step 2: Commit metadata + journal record + updated file tree atomically
    await this.journalBackend.commitAtomic(wsId, nextMeta, record, tree);
    for (const listener of this.commitListeners) {
      // A detached view must never turn a successful durable commit into failure.
      try { listener(record); } catch { /* The subscriber owns delivery failures. */ }
    }
    return record;
  }

  private async stageFileUpdates(
    wsId: WorkspaceId,
    fileUpdates?: Readonly<Record<string, Uint8Array | string | null>>,
    ownedHashes?: readonly Sha256Hex[],
  ): Promise<{ tree: Record<string, Sha256Hex>; historyBlobHashes: readonly Sha256Hex[] }> {
    const tree = await this.journalBackend.getFileTree(wsId);
    if (!fileUpdates) {
      return { tree, historyBlobHashes: ownedHashes ?? Object.values(tree) };
    }
    const history = new Set([...(ownedHashes ?? []), ...Object.values(tree)]);
    for (const [relPath, content] of Object.entries(fileUpdates)) {
      const safePath = normalizeSafeWorkspacePath(relPath);
      if (content === null) {
        delete tree[safePath];
        continue;
      }
      const sha256 =
        typeof content === "string"
          ? await this.blobStore.putText(content)
          : await this.blobStore.putBytes(content);
      tree[safePath] = sha256;
      history.add(sha256);
    }
    return { tree, historyBlobHashes: [...history] };
  }

  public async readFileBytes(relPath: string): Promise<Uint8Array> {
    const tree = await this.journalBackend.getFileTree(this.activeWorkspaceId);
    const hash = tree[relPath];
    if (!hash) {
      throw new Error(`ERR_FILE_NOT_FOUND: ${relPath}`);
    }
    return this.blobStore.getBytesVerified(hash);
  }

  public async readFileText(
    relPath: string,
    expectedSha256?: Sha256Hex,
  ): Promise<string> {
    const tree = await this.journalBackend.getFileTree(this.activeWorkspaceId);
    const hash = tree[relPath];
    if (!hash) {
      throw new Error(`ERR_FILE_NOT_FOUND: ${relPath}`);
    }
    if (expectedSha256 !== undefined && hash !== expectedSha256) {
      throw new Error(`ERR_FILE_CHANGED: ${relPath} no longer matches the requested manifest`);
    }
    return this.blobStore.getTextVerified(hash);
  }

  public async readFileTextOrDataUrl(relPath: string): Promise<string> {
    const mime = inferBinaryMimeType(relPath);
    if (!mime) {
      return this.readFileText(relPath);
    }
    const bytes = await this.readFileBytes(relPath);
    const prefix = new TextDecoder().decode(bytes.subarray(0, 16)).trimStart();
    if (prefix.startsWith("data:")) {
      return new TextDecoder().decode(bytes);
    }
    if (prefix.startsWith("<svg")) {
      const svgText = new TextDecoder().decode(bytes);
      return `data:image/svg+xml;utf8,${encodeURIComponent(svgText)}`;
    }
    return `data:${mime};base64,${bytesToBase64(bytes)}`;
  }

  public async listFiles(): Promise<string[]> {
    const tree = await this.journalBackend.getFileTree(this.activeWorkspaceId);
    return Object.keys(tree).sort();
  }

  public async getFileTree(): Promise<Record<string, Sha256Hex>> {
    return this.journalBackend.getFileTree(this.activeWorkspaceId);
  }

  public async computeBlobHashForPreview(
    content: string | Uint8Array,
  ): Promise<Sha256Hex> {
    const bytes =
      typeof content === "string" ? new TextEncoder().encode(content) : content;
    return computeSha256Hex(bytes);
  }

  public getLeaseMode(): "ReadWrite_Leader" | "ReadOnly_Observer" {
    return this.leaseReleased || this.lease.workspaceId !== this.activeWorkspaceId
      ? "ReadOnly_Observer"
      : this.lease.mode;
  }

  public async getMetadata(): Promise<WorkspaceMetadata | null> {
    return this.journalBackend.getMetadata(this.activeWorkspaceId);
  }

  public async getJournal(): Promise<RunJournalRecord[]> {
    return this.journalBackend.getJournalRecords(this.activeWorkspaceId);
  }

  public async exportWorkspaceBundle(): Promise<WorkspaceArchiveBundle> {
    const wsId = this.activeWorkspaceId;
    const meta = await this.journalBackend.getMetadata(wsId);
    if (!meta) throw new Error(`ERR_WORKSPACE_NOT_FOUND: ${wsId}`);
    const journal = await this.journalBackend.getJournalRecords(wsId);
    const fileTree = await this.journalBackend.getFileTree(wsId);
    const blobsBase64 = await this.blobStore.exportBlobs(
      new Set([...Object.values(fileTree), ...(meta.historyBlobHashes ?? [])]),
    );

    return {
      archiveVersion: 1,
      exportedAtMs: Date.now(),
      metadata: meta,
      journal,
      fileTree,
      blobsBase64,
    };
  }

  /**
   * Validates and atomically imports a workspace archive bundle (Scenario 7).
   * Rejects tampered blob hashes, non-monotonic sequences, or schema mismatches
   * before touching committed workspace state.
   */
  public async importWorkspaceBundle(rawBundle: unknown): Promise<WorkspaceMetadata> {
    this.assertLeaderWriteAuthority();
    if (typeof rawBundle !== "object" || rawBundle === null) {
      throw new Error("ERR_IMPORT_INVALID_ARCHIVE: Bundle must be an object");
    }
    const bundle = rawBundle as Partial<WorkspaceArchiveBundle>;
    if (bundle.archiveVersion !== 1 || !bundle.metadata || !Array.isArray(bundle.journal)) {
      throw new Error("ERR_IMPORT_SCHEMA_MISMATCH: Invalid archiveVersion or missing fields");
    }

    // Verify monotonic sequence numbers in the journal
    for (let i = 0; i < bundle.journal.length; i++) {
      const rec = bundle.journal[i]!;
      if (rec.seq !== i + 1 || rec.schemaVersion !== 1) {
        throw new Error(`ERR_IMPORT_CORRUPT_JOURNAL: Invalid sequence at index ${i}`);
      }
    }
    if (bundle.metadata.headSeq !== bundle.journal.length) {
      throw new Error("ERR_IMPORT_HEAD_SEQ_MISMATCH: headSeq does not match journal length");
    }

    // Verify every blob's cryptographic SHA-256 digest BEFORE writing anything
    const blobsMap = bundle.blobsBase64 ?? {};
    const verifiedBlobs = new Map<Sha256Hex, Uint8Array>();
    for (const [expectedHash, b64] of Object.entries(blobsMap)) {
      if (typeof b64 !== "string") {
        throw new Error(`ERR_IMPORT_INVALID_BLOB_ENCODING: ${expectedHash}`);
      }
      const bytes = base64ToBytes(b64);
      const actualHash = await computeSha256Hex(bytes);
      if (actualHash !== expectedHash) {
        throw new Error(
          `ERR_IMPORT_HASH_MISMATCH: Blob ${expectedHash} Tampered (computed ${actualHash})`,
        );
      }
      verifiedBlobs.set(expectedHash, bytes);
    }

    // Verify every file in fileTree points to a verified blob
    const fileTree = bundle.fileTree ?? {};
    for (const [relPath, hash] of Object.entries(fileTree)) {
      if (relPath.startsWith("/") || relPath.includes("..")) {
        throw new Error(`ERR_IMPORT_UNSAFE_PATH: ${relPath}`);
      }
      if (!verifiedBlobs.has(hash)) {
        throw new Error(`ERR_IMPORT_DANGLING_FILE_HASH: ${relPath} -> ${hash}`);
      }
    }

    // All checks passed: write blobs and atomically replace workspace metadata/journal
    for (const [, bytes] of verifiedBlobs) {
      await this.blobStore.putBytes(bytes);
    }

    const importedMeta: WorkspaceMetadata = {
      ...bundle.metadata,
      workspaceId: this.activeWorkspaceId,
      historyBlobHashes: [...verifiedBlobs.keys()],
    };
    await this.journalBackend.replaceWorkspaceAtomic(
      this.activeWorkspaceId,
      importedMeta,
      bundle.journal,
      { ...fileTree },
    );
    return importedMeta;
  }
}
