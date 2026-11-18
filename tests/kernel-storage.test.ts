import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  ContentAddressedBlobStore,
  InMemoryBlobBackend,
  StorageQuotaError,
} from "../src/kernel/blobs.ts";
import {
  IndexedDbJournalBackend,
  STORE_JOURNAL,
  STORE_TREE_MANIFESTS,
  STORE_WORKSPACES,
  type IDBDatabaseLike,
  type IDBFactoryLike,
  type IDBObjectStoreLike,
  type IDBOpenDBRequestLike,
  type IDBRequestLike,
  type IDBTransactionLike,
} from "../src/kernel/idb-storage.ts";
import {
  InMemoryJournalBackend,
  WorkspaceKernelStore,
} from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import {
  OpfsBlobBackend,
  type OpfsDirectoryHandleLike,
  type OpfsFileHandleLike,
} from "../src/kernel/opfs-blobs.ts";

describe("SYS-KERNEL Workspace Storage (IndexedDB, OPFS & Scenarios 5, 6, 7)", () => {
  it("rejects released leases and requires the target workspace's own authority", async () => {
    const locks = new InMemoryLockProvider();
    const backend = new InMemoryJournalBackend();
    const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const lease = await locks.acquireWorkspaceLease("released-a", "first-tab");
    const store = new WorkspaceKernelStore(backend, blobs, lease, locks);
    await store.initWorkspace("A");
    const other = await store.forkForWorkspace("released-b", "B");
    const withoutProvider = new WorkspaceKernelStore(backend, blobs, lease);
    await assert.rejects(withoutProvider.forkForWorkspace("released-b"), /ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED/);
    await assert.rejects(withoutProvider.switchWorkspace("released-b"), /ERR_WORKSPACE_LOCK_PROVIDER_REQUIRED/);
    assert.equal(withoutProvider.getActiveWorkspaceId(), "released-a");
    await assert.rejects(store.deleteWorkspaceStorage("released-b"), /ERR_WORKSPACE_AUTHORITY_MISMATCH/);
    await assert.rejects(store.deleteWorkspace("released-b"), /ERR_WORKSPACE_AUTHORITY_MISMATCH/);
    await store.releaseWorkspaceLease();
    assert.equal(store.getLeaseMode(), "ReadOnly_Observer");
    const secondLease = await locks.acquireWorkspaceLease("released-a", "second-tab");
    try {
      await assert.rejects(store.writeWorkspaceFile("leak.txt", "stale owner"), /ERR_WORKSPACE_LOCKED_BY_OTHER_TAB/);
      const third = await store.forkForWorkspace("released-c", "C");
      await third.writeWorkspaceFile("ok.txt", "independent lease");
      await third.releaseWorkspaceLease();
    } finally {
      await secondLease.release();
      await other.releaseWorkspaceLease();
    }
  });

  it("persists workspace files and monotonic journal across simulated browser reload", async () => {
    const locks = new InMemoryLockProvider();
    const journalBackend = new InMemoryJournalBackend();
    const blobBackend = new InMemoryBlobBackend();
    const blobs = new ContentAddressedBlobStore(blobBackend);

    // Session 1: Create workspace, start run, commit file via tool result
    const lease1 = await locks.acquireWorkspaceLease("ws-alpha", "tab-1");
    assert.equal(lease1.mode, "ReadWrite_Leader");
    const store1 = new WorkspaceKernelStore(journalBackend, blobs, lease1);

    await store1.initWorkspace("Alpha Project");
    await store1.appendEvent({
      eventId: "ev-1",
      runId: "run-100",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "Build a counter app" },
    });
    await store1.appendEvent({
      eventId: "ev-2",
      runId: "run-100",
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await store1.appendEvent({
      eventId: "ev-3",
      runId: "run-100",
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: {},
      fileUpdates: { "src/app.ts": "export const count = 1;\n" },
    });

    await lease1.release();

    // Session 2 (Browser Reload): Re-open against same durable backends
    const lease2 = await locks.acquireWorkspaceLease("ws-alpha", "tab-2");
    assert.equal(lease2.mode, "ReadWrite_Leader");
    const store2 = new WorkspaceKernelStore(journalBackend, blobs, lease2);

    const meta = await store2.getMetadata();
    assert.equal(meta?.headSeq, 3);
    assert.equal(meta?.runState, "inferring");
    assert.equal(await store2.readFileText("src/app.ts"), "export const count = 1;\n");
    await lease2.release();
  });

  it("Scenario 5: prevents two tabs from concurrently mutating the same workspace", async () => {
    const locks = new InMemoryLockProvider();
    const journalBackend = new InMemoryJournalBackend();
    const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());

    const tab1Lease = await locks.acquireWorkspaceLease("ws-shared", "tab-1");
    const tab2Lease = await locks.acquireWorkspaceLease("ws-shared", "tab-2");

    assert.equal(tab1Lease.mode, "ReadWrite_Leader");
    assert.equal(tab2Lease.mode, "ReadOnly_Observer");

    const tab1Store = new WorkspaceKernelStore(journalBackend, blobs, tab1Lease);
    const tab2Store = new WorkspaceKernelStore(journalBackend, blobs, tab2Lease);

    await tab1Store.initWorkspace("Shared Workspace");

    // Tab 2 (Observer) can read metadata, but any write attempt is deterministically blocked
    const observerMeta = await tab2Store.getMetadata();
    assert.equal(observerMeta?.title, "Shared Workspace");

    await assert.rejects(
      async () => {
        await tab2Store.appendEvent({
          eventId: "ev-tab2-illegal",
          runId: "run-tab2",
          type: "RunCreated",
          nextState: "created",
          payload: {},
        });
      },
      /ERR_WORKSPACE_LOCKED_BY_OTHER_TAB/,
    );

    await tab1Lease.release();
  });

  it("Scenario 6: surfaces storage quota errors without corrupting committed state or headSeq", async () => {
    const locks = new InMemoryLockProvider();
    const journalBackend = new InMemoryJournalBackend();
    // Allow only 64 bytes of OPFS blob storage
    const blobBackend = new InMemoryBlobBackend(64);
    const blobs = new ContentAddressedBlobStore(blobBackend);

    const lease = await locks.acquireWorkspaceLease("ws-quota", "tab-1");
    const store = new WorkspaceKernelStore(journalBackend, blobs, lease);
    await store.initWorkspace("Quota Test");

    await store.appendEvent({
      eventId: "ev-1",
      runId: "run-1",
      type: "RunCreated",
      nextState: "created",
      payload: {},
      fileUpdates: { "small.txt": "ok" },
    });

    // Attempt to write a file larger than the 64-byte OPFS quota
    await assert.rejects(
      async () => {
        await store.appendEvent({
          eventId: "ev-2",
          runId: "run-1",
          type: "RunPreparing",
          nextState: "preparing",
          payload: {},
          fileUpdates: { "huge.txt": "x".repeat(256) },
        });
      },
      (err: unknown) => err instanceof StorageQuotaError,
    );

    // Verify headSeq and runState did NOT advance and small.txt is untouched
    const metaAfterBlobFail = await store.getMetadata();
    assert.equal(metaAfterBlobFail?.headSeq, 1);
    assert.equal(metaAfterBlobFail?.runState, "created");
    assert.equal(await store.readFileText("small.txt"), "ok");

    // Also test IndexedDB journal quota failure mid-commit
    journalBackend.simulateQuotaFailureOnNextCommit(true);
    await assert.rejects(
      async () => {
        await store.appendEvent({
          eventId: "ev-2-retry",
          runId: "run-1",
          type: "RunPreparing",
          nextState: "preparing",
          payload: {},
        });
      },
      (err: unknown) => err instanceof StorageQuotaError,
    );
    assert.equal((await store.getMetadata())?.headSeq, 1);
  });

  it("Scenario 7: exports a complete workspace and rejects corrupted/tampered imports atomically", async () => {
    const locks = new InMemoryLockProvider();
    const journalBackend1 = new InMemoryJournalBackend();
    const blobs1 = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const lease1 = await locks.acquireWorkspaceLease("ws-source", "tab-1");
    const store1 = new WorkspaceKernelStore(journalBackend1, blobs1, lease1);

    await store1.initWorkspace("Exportable App");
    await store1.appendEvent({
      eventId: "ev-1",
      runId: "run-1",
      type: "RunCreated",
      nextState: "created",
      payload: {},
      fileUpdates: { "index.html": "<h1>Hello</h1>" },
    });

    const bundle = await store1.exportWorkspaceBundle();

    // Tamper with the base64 blob payload in a cloned bundle
    const tamperedBundle = structuredClone(bundle);
    const [firstHash] = Object.keys(tamperedBundle.blobsBase64);
    assert.ok(firstHash);
    (tamperedBundle.blobsBase64 as Record<string, string>)[firstHash] =
      Buffer.from("<h1>Tampered!</h1>").toString("base64");

    // Target workspace should reject tampered import and preserve its existing state
    const journalBackend2 = new InMemoryJournalBackend();
    const blobs2 = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const lease2 = await locks.acquireWorkspaceLease("ws-dest", "tab-1");
    const store2 = new WorkspaceKernelStore(journalBackend2, blobs2, lease2);
    await store2.initWorkspace("Clean Destination");

    await assert.rejects(
      async () => {
        await store2.importWorkspaceBundle(tamperedBundle);
      },
      /ERR_IMPORT_HASH_MISMATCH/,
    );
    assert.equal((await store2.getMetadata())?.title, "Clean Destination");

    // Valid bundle imports cleanly and reconstructs all files and journal history
    await store2.importWorkspaceBundle(bundle);
    assert.equal((await store2.getMetadata())?.title, "Exportable App");
    assert.equal((await store2.getMetadata())?.headSeq, 1);
    assert.equal(await store2.readFileText("index.html"), "<h1>Hello</h1>");
  });

  it("IndexedDbJournalBackend: awaits transaction.oncomplete, rolls back on abort/quota, and guards newer versions", async () => {
    const rawStores = new Map<string, Map<string, unknown>>([
      [STORE_WORKSPACES, new Map()],
      [STORE_JOURNAL, new Map()],
      [STORE_TREE_MANIFESTS, new Map()],
    ]);
    let dbVersion = 1;
    let simulateQuotaOnNextTx = false;
    let txCompletedCount = 0;

    const createFakeFactory = (): IDBFactoryLike => ({
      open(_name: string, requestedVersion = 1): IDBOpenDBRequestLike {
        const req: IDBOpenDBRequestLike = {
          result: undefined as unknown as IDBDatabaseLike,
          error: null,
          onsuccess: null,
          onerror: null,
          onupgradeneeded: null,
          onblocked: null,
        };
        queueMicrotask(() => {
          const db: IDBDatabaseLike = {
            version: Math.max(dbVersion, requestedVersion),
            objectStoreNames: { contains: (n) => rawStores.has(n) },
            createObjectStore: (n) => {
              if (!rawStores.has(n)) rawStores.set(n, new Map());
              return {} as IDBObjectStoreLike;
            },
            close: () => {},
            transaction: (storeNames, _mode) => {
              const staged = new Map<string, Map<string, unknown>>();
              for (const s of storeNames) {
                staged.set(s, new Map(rawStores.get(s) ?? []));
              }
              let aborted = false;
              const tx: IDBTransactionLike = {
                error: null,
                oncomplete: null,
                onerror: null,
                onabort: null,
                abort: () => {
                  aborted = true;
                  queueMicrotask(() => tx.onabort?.({}));
                },
                objectStore: (name) => {
                  const map = staged.get(name)!;
                  const makeReq = <T>(val: T): IDBRequestLike<T> => {
                    const r: IDBRequestLike<T> = {
                      result: val,
                      error: null,
                      onsuccess: null,
                      onerror: null,
                    };
                    queueMicrotask(() => r.onsuccess?.({}));
                    return r;
                  };
                  return {
                    get: (k) => makeReq(map.get(JSON.stringify(k))),
                    getAll: () => makeReq(Array.from(map.values())),
                    put: (v: unknown) => {
                      const rec = v as Record<string, unknown>;
                      const key =
                        name === STORE_JOURNAL
                          ? [rec.workspaceId, rec.seq]
                          : rec.workspaceId;
                      map.set(JSON.stringify(key), structuredClone(v));
                      return makeReq(key);
                    },
                    delete: (k) => {
                      map.delete(JSON.stringify(k));
                      return makeReq(undefined);
                    },
                  };
                },
              };
              setTimeout(() => {
                if (aborted) return;
                if (simulateQuotaOnNextTx) {
                  simulateQuotaOnNextTx = false;
                  (tx as { error: Error }).error = Object.assign(
                    new Error("Quota exceeded"),
                    { name: "QuotaExceededError" },
                  );
                  tx.onabort?.({});
                  return;
                }
                for (const [s, m] of staged) {
                  rawStores.set(s, m);
                }
                txCompletedCount += 1;
                tx.oncomplete?.({});
              }, 5);
              return tx;
            },
          };
          (req as { result: IDBDatabaseLike }).result = db;
          req.onupgradeneeded?.({ oldVersion: 0, newVersion: db.version });
          req.onsuccess?.({});
        });
        return req;
      },
    });

    const idbBackend = new IndexedDbJournalBackend(createFakeFactory());
    const opfsFiles = new Map<string, Uint8Array>();
    let failOpfsWriteWithQuota = false;

    const fakeOpfsDir: OpfsDirectoryHandleLike = {
      getDirectoryHandle: async () => fakeOpfsDir,
      getFileHandle: async (name, opts) => {
        if (!opts?.create && !opfsFiles.has(name)) {
          throw Object.assign(new Error("Missing"), { name: "NotFoundError" });
        }
        const handle: OpfsFileHandleLike = {
          getFile: async () => ({
            arrayBuffer: async () =>
              (opfsFiles.get(name) ?? new Uint8Array()).slice(0)
                .buffer as ArrayBuffer,
          }),
          createWritable: async () => {
            let buf = new Uint8Array();
            return {
              write: async (d) => {
                if (failOpfsWriteWithQuota) {
                  throw Object.assign(new Error("OPFS Quota"), {
                    name: "QuotaExceededError",
                  });
                }
                buf = new Uint8Array(d);
              },
              close: async () => {
                opfsFiles.set(name, buf);
              },
              abort: async () => {},
            };
          },
        };
        return handle;
      },
      removeEntry: async (name) => {
        opfsFiles.delete(name);
      },
      entries: async function* () {
        for (const k of opfsFiles.keys()) {
          yield [k, { kind: "file" as const }];
        }
      },
    };

    const opfsBackend = new OpfsBlobBackend(async () => fakeOpfsDir);
    const blobs = new ContentAddressedBlobStore(opfsBackend);
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease("ws-idb", "tab-1");
    const store = new WorkspaceKernelStore(idbBackend, blobs, lease);

    await store.initWorkspace("IDB Workspace");
    assert.ok(txCompletedCount >= 1);

    await store.appendEvent({
      eventId: "ev-idb-1",
      runId: "run-idb",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "Persist me" },
      fileUpdates: { "app.ts": "console.log('durable');" },
    });

    assert.equal(await store.readFileText("app.ts"), "console.log('durable');");
    assert.equal((await opfsBackend.listHashes()).length, 1);

    // Simulate OPFS quota error: staging file must be cleaned up and headSeq must stay 1
    failOpfsWriteWithQuota = true;
    await assert.rejects(
      async () => {
        await store.appendEvent({
          eventId: "ev-idb-2",
          runId: "run-idb",
          type: "RunPreparing",
          nextState: "preparing",
          payload: {},
          fileUpdates: { "fail.ts": "too big" },
        });
      },
      (err: unknown) => err instanceof StorageQuotaError,
    );
    failOpfsWriteWithQuota = false;
    assert.equal((await store.getMetadata())?.headSeq, 1);
    assert.equal(
      Array.from(opfsFiles.keys()).some((k) => k.endsWith(".tmp")),
      false,
    );

    // Simulate IDB transaction abort on quota: metadata and journal must roll back
    simulateQuotaOnNextTx = true;
    await assert.rejects(
      async () => {
        await store.appendEvent({
          eventId: "ev-idb-2",
          runId: "run-idb",
          type: "RunPreparing",
          nextState: "preparing",
          payload: {},
        });
      },
      (err: unknown) => err instanceof StorageQuotaError,
    );
    assert.equal((await store.getMetadata())?.headSeq, 1);

    // Newer database version guard: opening with stored version 99 > targetVersion 1 rejects
    await idbBackend.close();
    dbVersion = 99;
    const incompatibleBackend = new IndexedDbJournalBackend(createFakeFactory());
    await assert.rejects(
      async () => {
        await incompatibleBackend.getMetadata("ws-idb");
      },
      /ERR_IDB_NEWER_VERSION/,
    );
  });
});
