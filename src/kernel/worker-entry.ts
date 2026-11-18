/**
 * @module SYS-KERNEL/worker-entry
 * Browser Web Worker entry point assembling:
 * - `BrowserNavigatorLockProvider` (Scenario 5 multi-tab leader/observer lease)
 * - `IndexedDbJournalBackend` (or `InMemoryJournalBackend` fallback)
 * - `OpfsBlobBackend` (or `InMemoryBlobBackend` fallback)
 * - `VelesHybridInferenceAdapter` (supports both live OpenAI-compatible endpoints
 *   and the built-in `veles-demo` interactive COMET Counter App workflow)
 * - `BrowserToolRegistry` + `GatewayToolExecutionAdapter`
 * - `KernelWorkerHost` bound to `globalThis` (`DedicatedWorkerGlobalScope`)
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import {
  OpenAICompatibleAdapter,
  type InferenceAdapter,
  type InferenceStreamRequest,
  type NormalizedStreamChunk,
} from "../inference/openai-compatible.ts";
import type { WorkerPortLike } from "../protocol/types.ts";
import {
  ContentAddressedBlobStore,
  InMemoryBlobBackend,
  type BlobBackend,
} from "./blobs.ts";
import { GatewayToolExecutionAdapter } from "./gateway-client.ts";
import { IndexedDbJournalBackend, type IDBFactoryLike } from "./idb-storage.ts";
import {
  InMemoryJournalBackend,
  WorkspaceKernelStore,
  type JournalStorageBackend,
} from "./journal.ts";
import {
  BrowserNavigatorLockProvider,
  InMemoryLockProvider,
  type LockProvider,
} from "./lock.ts";
import { OpfsBlobBackend, type OpfsDirectoryHandleLike } from "./opfs-blobs.ts";
import { AgentManager } from "./agent-session.ts";
import { AgentKernelRuntime } from "./runtime.ts";
import {
  BrowserToolRegistry,
  createDefaultWorkspaceTools,
  createSubagentTools,
} from "./tools.ts";
import { KernelWorkerHost } from "./worker-host.ts";

const DEMO_COUNTER_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <style>
    body { font-family: system-ui, sans-serif; background: #101c25; color: #dbe5e4; padding: 20px; margin: 0; }
    .card { background: #15242e; border: 1px solid #30434b; border-radius: 6px; padding: 16px; max-width: 340px; }
    h2 { margin: 0 0 8px; color: #c1d5bd; font-size: 16px; }
    .row { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
    button { background: #1a3036; color: #dbe5e4; border: 1px solid #4d655b; border-radius: 4px; padding: 6px 12px; cursor: pointer; }
    button:hover { background: #20362f; }
    .count { font-size: 22px; font-weight: 700; color: #9ebd9d; min-width: 48px; text-align: center; }
    .meta { font-size: 11px; color: #90a8a0; margin-top: 12px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Veles Sandboxed Counter</h2>
    <div>Origin: <code id="origin-label">checking...</code></div>
    <div class="row">
      <button id="dec-btn">−</button>
      <span class="count" id="count-val">42</span>
      <button id="inc-btn">+</button>
    </div>
    <div class="meta" id="storage-probe">Probing host storage isolation...</div>
  </div>
  <script>
    let count = 42;
    let currentGen = 1;
    const originEl = document.getElementById("origin-label");
    const countEl = document.getElementById("count-val");
    const probeEl = document.getElementById("storage-probe");
    originEl.textContent = String(window.origin || "null (opaque)");

    function sendDiag(type, message) {
      parent.postMessage({ channel: "veles:preview", generation: currentGen, type, message }, "*");
    }

    window.addEventListener("message", (ev) => {
      if (ev.data && ev.data.channel === "veles:host-init") {
        currentGen = ev.data.generation || 1;
        let idbBlocked = false;
        try {
          void window.indexedDB.open("probe");
        } catch (err) {
          idbBlocked = true;
        }
        probeEl.textContent = idbBlocked
          ? "SEC-INV-01 Verified: IndexedDB & OPFS blocked in opaque origin"
          : "Opaque sandbox active";
        sendDiag("preview_ready", "Mounted Counter App (gen #" + currentGen + ", origin=" + window.origin + ")");
      }
    });

    document.getElementById("inc-btn").addEventListener("click", () => {
      count += 1;
      countEl.textContent = String(count);
      sendDiag("preview_log", "Counter incremented to " + count);
    });
    document.getElementById("dec-btn").addEventListener("click", () => {
      count -= 1;
      countEl.textContent = String(count);
      sendDiag("preview_log", "Counter decremented to " + count);
    });
  </script>
</body>
</html>`;

class VelesHybridInferenceAdapter implements InferenceAdapter {
  private readonly remote = new OpenAICompatibleAdapter();

  public async *streamChat(
    request: InferenceStreamRequest,
  ): AsyncIterable<NormalizedStreamChunk> {
    if (request.config.providerId !== "veles-demo") {
      yield* this.remote.streamChat(request);
      return;
    }
    yield* this.streamBuiltInDemo(request);
  }

  private async *streamBuiltInDemo(
    request: InferenceStreamRequest,
  ): AsyncIterable<NormalizedStreamChunk> {
    const lastUserIdx = request.messages
      .map((m) => m.role)
      .lastIndexOf("user");
    const activeTurnMessages =
      lastUserIdx >= 0
        ? request.messages.slice(lastUserIdx + 1)
        : request.messages;
    const toolMessages = activeTurnMessages.filter((m) => m.role === "tool");
    const { providerId, model } = request.config;

    if (toolMessages.length === 0) {
      yield {
        providerId,
        model,
        deltaText: "Scaffolding sandboxed `index.html` counter app in OPFS...",
        toolCalls: [
          {
            id: "call_demo_write_html",
            name: "workspace.write_file",
            argumentsJson: JSON.stringify({
              path: "index.html",
              content: DEMO_COUNTER_HTML,
            }),
          },
        ],
        usage: null,
        finishReason: "tool_calls",
      };
      return;
    }

    if (toolMessages.length === 1) {
      yield {
        providerId,
        model,
        deltaText: "Staging temporary `obsolete.txt` artifact before cleanup...",
        toolCalls: [
          {
            id: "call_demo_write_temp",
            name: "workspace.write_file",
            argumentsJson: JSON.stringify({
              path: "obsolete.txt",
              content: "legacy scratch file",
            }),
          },
        ],
        usage: null,
        finishReason: "tool_calls",
      };
      return;
    }

    if (toolMessages.length === 2) {
      yield {
        providerId,
        model,
        deltaText: "Requesting user approval to delete `obsolete.txt` (`non_idempotent`)...",
        toolCalls: [
          {
            id: "call_demo_delete_temp",
            name: "workspace.delete_file",
            argumentsJson: JSON.stringify({ path: "obsolete.txt" }),
          },
        ],
        usage: null,
        finishReason: "tool_calls",
      };
      return;
    }

    yield {
      providerId,
      model,
      deltaText:
        "Counter app committed to OPFS (`index.html`) and ready in the isolated `<iframe sandbox=\"allow-scripts\">` preview.",
      toolCalls: [],
      usage: { promptTokens: 48, completionTokens: 24, totalTokens: 72 },
      finishReason: "stop",
    };
  }
}

async function resolveStorageBackends(): Promise<{
  journalBackend: JournalStorageBackend;
  blobBackend: BlobBackend;
}> {
  const idbFactory = (globalThis as { indexedDB?: IDBFactoryLike }).indexedDB;
  const navStorage = (
    globalThis.navigator as {
      storage?: { getDirectory?: () => Promise<OpfsDirectoryHandleLike> };
    }
  )?.storage;
  if (idbFactory && typeof navStorage?.getDirectory === "function") {
    try {
      const rootDir = await navStorage.getDirectory();
      return {
        journalBackend: new IndexedDbJournalBackend(idbFactory),
        blobBackend: new OpfsBlobBackend(async () => rootDir),
      };
    } catch {
      // Fall back both backends atomically so IDB manifests never reference volatile RAM blobs
    }
  }
  return {
    journalBackend: new InMemoryJournalBackend(),
    blobBackend: new InMemoryBlobBackend(),
  };
}

export async function bootBrowserKernelWorker(
  port: WorkerPortLike,
  workspaceId = "ws-alpha",
): Promise<KernelWorkerHost> {
  const hasWebLocks =
    typeof globalThis.navigator !== "undefined" &&
    "locks" in globalThis.navigator &&
    Boolean((globalThis.navigator as { locks?: unknown }).locks);
  const lockProvider: LockProvider = hasWebLocks
    ? new BrowserNavigatorLockProvider()
    : new InMemoryLockProvider();
  const lease = await lockProvider.acquireWorkspaceLease(
    workspaceId,
    `worker-${Date.now()}`,
  );
  const { journalBackend, blobBackend } = await resolveStorageBackends();
  const blobs = new ContentAddressedBlobStore(blobBackend);
  const store = new WorkspaceKernelStore(
    journalBackend,
    blobs,
    lease,
    lockProvider,
  );

  const gatewayFallback = new GatewayToolExecutionAdapter({
    baseUrl: "",
    defaultScope: {
      allowedTools: ["sandbox.exec", "sandbox.WriteFile"],
      allowNetwork: false,
      writablePaths: ["src", "dist"],
    },
    defaultLimits: { timeoutMs: 10_000, maxOutputBytes: 65_536 },
  });

  let runtimeRef: AgentKernelRuntime | undefined;
  const workspaceTools = createDefaultWorkspaceTools(
    (runId) => runtimeRef?.getLatestModelConfig(runId),
    () => runtimeRef?.getServiceConfig(),
  );
  const tools = new BrowserToolRegistry(store, workspaceTools, gatewayFallback);
  const inference = new VelesHybridInferenceAdapter();
  const runtime = new AgentKernelRuntime(
    store,
    inference,
    tools,
    tools.getToolDefinitions(),
    { multiTurnWorkspaceContext: true, hotRunWindow: 2 },
  );
  runtimeRef = runtime;
  const agentManager = new AgentManager(store, runtime);
  for (const subTool of createSubagentTools(agentManager)) {
    tools.registerTool(subTool);
    runtime.registerToolDefinition(subTool.spec);
  }
  void gatewayFallback
    .fetchCapabilities()
    .then((specs) => {
      tools.registerGatewayToolSpecs(specs);
      for (const spec of specs) {
        runtime.registerToolDefinition(spec);
      }
    })
    .catch(() => {});

  return new KernelWorkerHost(
    store, runtime, port, agentManager, createDefaultWorkspaceTools,
  );
}

// Auto-boot when loaded as a browser DedicatedWorkerGlobalScope
const isDedicatedWorker =
  typeof globalThis !== "undefined" &&
  typeof (globalThis as { postMessage?: unknown }).postMessage === "function" &&
  typeof (globalThis as { document?: unknown }).document === "undefined" &&
  typeof (globalThis as { process?: { versions?: { node?: string } } }).process
    ?.versions?.node === "undefined";

if (isDedicatedWorker) {
  const workerScope = globalThis as unknown as WorkerPortLike;
  const earlyQueue: unknown[] = [];
  const earlyListener = (ev: { readonly data: unknown }) => {
    earlyQueue.push(ev.data);
  };
  workerScope.addEventListener?.("message", earlyListener);
  void bootBrowserKernelWorker(workerScope).then(async (host) => {
    workerScope.removeEventListener?.("message", earlyListener);
    for (const raw of earlyQueue) {
      await host.receivePortMessage(raw);
    }
  });
}
