import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { EphemeralExecutionGateway } from "../services/gateway/server.ts";
import type { InferenceAdapter } from "../src/inference/openai-compatible.ts";
import { AgentManager } from "../src/kernel/agent-session.ts";
import {
  ContentAddressedBlobStore,
  InMemoryBlobBackend,
} from "../src/kernel/blobs.ts";
import { GatewayToolExecutionAdapter } from "../src/kernel/gateway-client.ts";
import {
  InMemoryJournalBackend,
  type WorkspaceArchiveBundle,
  WorkspaceKernelStore,
} from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import { AgentKernelRuntime } from "../src/kernel/runtime.ts";
import { BrowserToolRegistry } from "../src/kernel/tools.ts";
import { bootBrowserKernelWorker } from "../src/kernel/worker-entry.ts";
import { KernelWorkerHost } from "../src/kernel/worker-host.ts";
import { VelesAppClient } from "../src/ui/app-client.ts";
import {
  PreviewSandboxHost,
  type MessageEventLike,
  type PreviewRuntimeDiagnostic,
  type SandboxEnvironmentLike,
  type SandboxIframeLike,
} from "../src/ui/preview-sandbox.ts";

function createTestSandboxEnv() {
  let listener: ((ev: MessageEventLike) => void) | null = null;
  const posted: unknown[] = [];
  let lastSandboxAttr = "";
  let lastSrcdoc = "";
  let removed = 0;

  const contentWindow = {
    postMessage(msg: unknown) {
      posted.push(msg);
    },
  };

  const env: SandboxEnvironmentLike = {
    createIframe(): SandboxIframeLike {
      return {
        get sandbox() {
          return lastSandboxAttr;
        },
        set sandbox(v: string) {
          lastSandboxAttr = v;
        },
        get srcdoc() {
          return lastSrcdoc;
        },
        set srcdoc(v: string) {
          lastSrcdoc = v;
        },
        contentWindow,
        remove() {
          removed += 1;
        },
      };
    },
    addMessageListener(h) {
      listener = h;
      return () => {
        if (listener === h) listener = null;
      };
    },
  };

  return {
    env,
    contentWindow,
    getPosted: () => posted,
    getSandboxAttr: () => lastSandboxAttr,
    getSrcdoc: () => lastSrcdoc,
    getRemoved: () => removed,
    emitFromIframe: (data: unknown, origin = "null") =>
      listener?.({ source: contentWindow, origin, data }),
  };
}

function createGatewayFetch(gateway: EphemeralExecutionGateway) {
  return async (
    url: string,
    init: {
      method: string;
      headers: Record<string, string>;
      body?: string;
      signal?: AbortSignal;
    },
  ) => {
    if (!gateway.checkAuthorized(init.headers.Authorization)) {
      return { ok: false, status: 401, body: null };
    }
    const u = new URL(url);
    if (init.method === "GET" && u.pathname.startsWith("/v1/operations/")) {
      const opId = decodeURIComponent(u.pathname.slice("/v1/operations/".length));
      return {
        ok: true,
        status: 200,
        json: async () => gateway.getOperationStatus(opId),
      };
    }
    if (init.method === "POST" && u.pathname === "/v1/tools/execute") {
      const reqBody: unknown = JSON.parse(init.body ?? "{}");
      const encoder = new TextEncoder();
      const iter = gateway.executeToolStream(reqBody)[Symbol.asyncIterator]();
      const body = new ReadableStream<Uint8Array>({
        async pull(ctrl) {
          try {
            const nxt = await iter.next();
            if (nxt.done) {
              ctrl.enqueue(encoder.encode("data: [DONE]\n\n"));
              ctrl.close();
              return;
            }
            ctrl.enqueue(encoder.encode(`data: ${JSON.stringify(nxt.value)}\n\n`));
          } catch (e) {
            ctrl.error(e);
          }
        },
      });
      return { ok: true, status: 200, body };
    }
    return { ok: false, status: 404, body: null };
  };
}

describe("SYS-UI & SYS-KERNEL Worker MessagePort Bridge, Release Journey, & All 10 Scenarios", () => {
  for (const credentialScope of ["workspace", "session"] as const) {
  it(`keeps background image and search calls bound to their originating run credentials across ${credentialScope}s`, async (t) => {
    let enter!: () => void;
    let release!: () => void;
    const entered = new Promise<void>((done) => { enter = done; });
    const held = new Promise<void>((done) => { release = done; });
    const requests: Array<{ url: string; authorization: string | null }> = [];
    const streamed = (chunk: unknown) => new Response(
      `data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`,
      { headers: { "Content-Type": "text/event-stream" } },
    );
    t.mock.method(globalThis, "fetch", async (input: unknown, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/v1/capabilities")) return Response.json({ toolDefinitions: [] });
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        model?: string; messages?: Array<{ role: string }>;
      };
      if (url.endsWith("/chat/completions")) {
        if (body.model === "model-a" && !body.messages?.some((m) => m.role === "tool")) {
          enter();
          await held;
          return streamed({ choices: [{ delta: { tool_calls: [
            { index: 0, id: "image-a", function: { name: "image__generate", arguments: JSON.stringify({ outputPath: "assets/a.png", prompt: "an A image" }) } },
            { index: 1, id: "search-a", function: { name: "web__search", arguments: JSON.stringify({ query: "A query" }) } },
          ] }, finish_reason: "tool_calls" }] });
        }
        return streamed({ choices: [{ delta: { content: "Done" }, finish_reason: "stop" }] });
      }
      requests.push({ url, authorization: new Headers(init?.headers).get("Authorization") });
      if (url.endsWith("/images/generations")) return Response.json({ data: [{ b64_json: "AA==" }] });
      return Response.json({ output_text: "A search result" });
    });
    const cfgA = { providerId: "openai", baseUrl: "https://provider-a.test/v1", model: "model-a", apiKey: "key-a" };
    const ch = new MessageChannel();
    const host = await bootBrowserKernelWorker(ch.port1, "ws-provider-a");
    const client = new VelesAppClient(ch.port2, cfgA);
    try {
      await client.initWorkspace("A");
      const services = client.getStateView().serviceConfig;
      await client.updateServiceConfig({ ...services, decision: { ...services.decision, mode: "tier0-only" } });
      const cfgB = { ...cfgA, baseUrl: "https://provider-b.test/v1", model: "model-b", apiKey: "key-b" };
      if (credentialScope === "session") {
        await client.createSession({ id: "provider-a", workspaceId: "ws-provider-a", model: cfgA, allowedTools: ["image.generate", "web.search"], status: "idle" });
        await client.createSession({ id: "provider-b", workspaceId: "ws-provider-a", model: cfgB, allowedTools: [], status: "idle" });
      }
      const backgroundRun = credentialScope === "workspace"
        ? client.startRun("provider-a-run", "Generate and search", cfgA)
        : host.agentManager.runWithId("provider-a", "Generate and search", "provider-a-run");
      await entered;
      if (credentialScope === "workspace") {
        await client.switchWorkspace("ws-provider-b", "B");
        await client.startRun("provider-b-run", "Finish", cfgB);
      } else {
        await host.agentManager.runWithId("provider-b", "Finish", "provider-b-run");
      }
      release();
      await backgroundRun;
      assert.deepEqual(requests, [
        { url: "https://provider-a.test/v1/images/generations", authorization: "Bearer key-a" },
        { url: "https://provider-a.test/v1/responses", authorization: "Bearer key-a" },
      ]);
    } finally {
      release(); client.dispose(); host.dispose(); ch.port1.close(); ch.port2.close();
    }
  });
  }

  it("refuses to delete a workspace whose lease is held by another tab", async () => {
    const locks = new InMemoryLockProvider();
    const backend = new InMemoryJournalBackend();
    const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const lease = await locks.acquireWorkspaceLease("ws-delete-a", "first-tab");
    const store = new WorkspaceKernelStore(backend, blobs, lease, locks);
    await store.initWorkspace("A");
    const registry = new BrowserToolRegistry(store);
    const inference: InferenceAdapter = { async *streamChat() {
      yield { providerId: "test", model: "test", deltaText: "Done", toolCalls: [], usage: null, finishReason: "stop" };
    } };
    const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions());
    const ch = new MessageChannel();
    const host = new KernelWorkerHost(store, runtime, ch.port1);
    const client = new VelesAppClient(ch.port2);
    let secondLease: Awaited<ReturnType<InMemoryLockProvider["acquireWorkspaceLease"]>> | undefined;
    try {
      await client.initWorkspace("A");
      await client.switchWorkspace("ws-delete-b", "B");
      secondLease = await locks.acquireWorkspaceLease("ws-delete-a", "second-tab");
      assert.equal(secondLease.mode, "ReadWrite_Leader");
      const owner = new WorkspaceKernelStore(backend, blobs, secondLease, locks);
      await owner.writeWorkspaceFile("notes/owner.txt", "second-tab content");
      await assert.rejects(client.deleteWorkspace("ws-delete-a"), /ERR_WORKSPACE_LOCKED_BY_OTHER_TAB/);
      assert.equal(await owner.readFileText("notes/owner.txt"), "second-tab content");
      assert.equal(client.getStateView().snapshot?.workspaceId, "ws-delete-b");
      await secondLease.release();
      const removed = await client.deleteWorkspace("ws-delete-a");
      assert.equal(removed.workspaceId, "ws-delete-b");
      assert.equal(await owner.getMetadata(), null);
    } finally {
      await secondLease?.release(); client.dispose(); host.dispose(); ch.port1.close(); ch.port2.close();
    }
  });

  it("preserves explicit restricted and unrestricted policies through the port and journal recovery", async () => {
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease("ws-explicit-policy", "policy-test");
    const store = new WorkspaceKernelStore(
      new InMemoryJournalBackend(),
      new ContentAddressedBlobStore(new InMemoryBlobBackend()),
      lease,
    );
    const requests: string[][] = [];
    const inference: InferenceAdapter = {
      async *streamChat(req) {
        requests.push((req.tools ?? []).map((tool) => tool.name));
        const toolCalls = req.messages.some((message) => message.role === "tool")
          ? []
          : [{
              id: "write-policy-marker",
              name: "workspace.write_file",
              argumentsJson: JSON.stringify({ path: "permitted.txt", content: "authorized" }),
            }];
        yield {
          providerId: req.config.providerId, model: req.config.model,
          deltaText: "done", toolCalls, usage: null,
          finishReason: toolCalls.length ? "tool_calls" : "stop",
        };
      },
    };
    const tools = new BrowserToolRegistry(store);
    const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions());
    const manager = new AgentManager(store, runtime);
    const channel = new MessageChannel();
    const host = new KernelWorkerHost(store, runtime, channel.port1, manager);
    const client = new VelesAppClient(channel.port2);
    const model = { providerId: "dummy", baseUrl: "https://dummy.invalid", model: "policy" };
    try {
      await client.initWorkspace("Policy Regression");
      await client.createSession({
        id: "session-main", workspaceId: "ws-explicit-policy", model,
        allowedTools: [], unrestrictedTools: false, status: "idle",
      });
      await manager.recoverSessionsFromJournal();
      assert.equal(manager.getSession("session-main")?.unrestrictedTools, false);
      await assert.rejects(client.createSession({
        id: "session-main", workspaceId: "ws-explicit-policy", model,
        allowedTools: [], unrestrictedTools: "false" as unknown as boolean, status: "idle",
      }), /ERR_INVALID_SESSION_POLICY/);
      assert.equal(manager.getSession("session-main")?.unrestrictedTools, false);
      await client.startRun("restricted-run", "Attempt a write", model);
      assert.deepEqual(requests[0], []);
      assert.deepEqual(await store.listFiles(), []);
      assert.equal(manager.getSession("session-main")?.unrestrictedTools, false);
      await manager.recoverSessionsFromJournal();
      const restricted = manager.getSession("session-main")!;
      assert.equal(restricted.unrestrictedTools, false);
      // A routine update that omits authority must retain the saved denial.
      manager.upsertSessionInMemory({ ...restricted, unrestrictedTools: undefined });
      assert.equal(manager.getSession("session-main")?.unrestrictedTools, false);
      const restrictedRun = (await store.getJournal()).find(
        (record) => record.type === "RunCreated" && record.runId === "restricted-run",
      );
      assert.equal(restrictedRun?.payload.unrestrictedTools, false);

      await client.createSession({
        id: "coordinator", workspaceId: "ws-explicit-policy", model,
        allowedTools: [], unrestrictedTools: true, status: "idle",
      });
      const allowed = await client.startRun("authorized-run", "Write a marker", model, "coordinator");
      assert.equal(allowed.files["permitted.txt"], "authorized");
      assert.equal(allowed.sessions?.find((session) => session.id === "coordinator")?.unrestrictedTools, true);
      await manager.recoverSessionsFromJournal();
      assert.equal(manager.getSession("coordinator")?.unrestrictedTools, true);
      const authorizedRun = (await store.getJournal()).find(
        (record) => record.type === "RunCreated" && record.runId === "authorized-run",
      );
      assert.equal(authorizedRun?.payload.unrestrictedTools, true);
    } finally {
      client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close();
    }
  });

  it("executes the First Useful Release Flow over MessagePort: init -> select model -> prompt -> approve -> preview -> export/import", async () => {
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease("ws-m5-journey", "tab-leader");
    const journalBackend = new InMemoryJournalBackend();
    const blobBackend = new InMemoryBlobBackend();
    const store = new WorkspaceKernelStore(
      journalBackend,
      new ContentAddressedBlobStore(blobBackend),
      lease,
    );

    let turn = 0;
    const inference: InferenceAdapter = {
      async *streamChat(req) {
        turn += 1;
        if (turn === 1) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Writing index.html...",
            toolCalls: [
              {
                id: "tc-write-html",
                name: "workspace.write_file",
                argumentsJson: JSON.stringify({
                  path: "index.html",
                  content: "<!doctype html><h1>Hello Veles M5</h1>",
                }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        if (turn === 2) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Write legacy.js...",
            toolCalls: [
              {
                id: "tc-write-legacy",
                name: "workspace.write_file",
                argumentsJson: JSON.stringify({
                  path: "legacy.js",
                  content: "// old",
                }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        if (turn === 3) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Requesting approval to delete legacy.js...",
            toolCalls: [
              {
                id: "tc-del-legacy",
                name: "workspace.delete_file",
                argumentsJson: JSON.stringify({ path: "legacy.js" }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Workspace ready for preview!",
          toolCalls: [],
          usage: { promptTokens: 30, completionTokens: 10, totalTokens: 40 },
          finishReason: "stop",
        };
      },
    };

    const tools = new BrowserToolRegistry(store);
    const runtime = new AgentKernelRuntime(
      store,
      inference,
      tools,
      tools.getToolDefinitions(),
    );

    const channel = new MessageChannel();
    const host = new KernelWorkerHost(store, runtime, channel.port1);
    const client = new VelesAppClient(channel.port2);

    try {
      // 1. Create workspace
      const initialSnap = await client.initWorkspace("Veles M5 Release App");
      assert.equal(initialSnap.title, "Veles M5 Release App");
      assert.equal(initialSnap.headSeq, 0);

      // 2. Select provider/model (Scenario 9)
      client.selectProviderModel({
        providerId: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
        model: "anthropic/claude-3.7-sonnet",
      });

      // 3. Start run -> writes index.html & legacy.js, then pauses at awaiting_approval for delete_file
      const pausedSnap = await client.startRun("run-m5-1", "Build app and clean legacy.js");
      assert.equal(pausedSnap.runState, "awaiting_approval");
      assert.equal(pausedSnap.pendingApproval?.toolName, "workspace.delete_file");
      assert.equal(pausedSnap.pendingApproval?.toolCallId, "tc-del-legacy");
      assert.deepEqual(Object.keys(pausedSnap.files).sort(), ["index.html", "legacy.js"]);

      // 4. Approve deletion -> completes run and removes legacy.js
      const completedSnap = await client.resolveApproval("run-m5-1", true);
      assert.equal(completedSnap.runState, "completed");
      assert.equal(completedSnap.pendingApproval, null);
      assert.deepEqual(Object.keys(completedSnap.files), ["index.html"]);
      assert.equal(
        completedSnap.files["index.html"],
        "<!doctype html><h1>Hello Veles M5</h1>",
      );

      // 5. Mount generated workspace into isolated <iframe sandbox="allow-scripts"> preview (Scenario 8)
      const sandboxMock = createTestSandboxEnv();
      const diagnostics: PreviewRuntimeDiagnostic[] = [];
      const previewHost = new PreviewSandboxHost(sandboxMock.env, (d) =>
        diagnostics.push(d),
      );
      const gen = client.mountWorkspacePreview(previewHost, "index.html");
      assert.equal(gen, 1);
      assert.equal(sandboxMock.getSandboxAttr(), "allow-scripts");
      assert.equal(
        sandboxMock.getSrcdoc(),
        "<!doctype html><h1>Hello Veles M5</h1>",
      );
      sandboxMock.emitFromIframe({
        channel: "veles:preview",
        generation: 1,
        type: "preview_ready",
        message: "Rendered h1",
      });
      assert.equal(diagnostics.length, 1);
      previewHost.dispose();
      assert.equal(sandboxMock.getRemoved(), 1);

      // 6. Export workspace archive and import into a fresh workspace (Scenario 7)
      const exportedBundle = (await client.exportWorkspace()) as WorkspaceArchiveBundle;
      assert.equal(exportedBundle.archiveVersion, 1);

      // Tampered bundle is rejected without corrupting existing workspace
      const tamperedBundle: WorkspaceArchiveBundle = {
        ...exportedBundle,
        metadata: { ...exportedBundle.metadata, headSeq: 999 },
      };
      await assert.rejects(
        () => client.importWorkspace(tamperedBundle),
        /ERR_IMPORT_HEAD_SEQ_MISMATCH/,
      );
      assert.equal(client.getStateView().snapshot?.headSeq, completedSnap.headSeq);

      // Valid bundle imports cleanly
      const restoredSnap = await client.importWorkspace(exportedBundle);
      assert.equal(restoredSnap.headSeq, completedSnap.headSeq);
      assert.equal(
        restoredSnap.files["index.html"],
        "<!doctype html><h1>Hello Veles M5</h1>",
      );
    } finally {
      client.dispose();
      host.dispose();
      channel.port1.close();
      channel.port2.close();
    }
  });

  it("verifies Scenarios 1, 2, 3, 4, 5, 6, and 10 over the Worker MessagePort seam", async () => {
    const locks = new InMemoryLockProvider();
    const leaderLease = await locks.acquireWorkspaceLease("ws-m5-scenarios", "tab-1");
    const observerLease = await locks.acquireWorkspaceLease("ws-m5-scenarios", "tab-2");
    assert.equal(leaderLease.mode, "ReadWrite_Leader");
    assert.equal(observerLease.mode, "ReadOnly_Observer");

    const journalBackend = new InMemoryJournalBackend();
    const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const leaderStore = new WorkspaceKernelStore(journalBackend, blobs, leaderLease);
    const observerStore = new WorkspaceKernelStore(journalBackend, blobs, observerLease);

    // Scenario 1 & 3: Stream disconnect during inference discards uncommitted draft and recovers cleanly
    const disconnectInference: InferenceAdapter = {
      async *streamChat() {
        yield {
          providerId: "openai",
          model: "gpt-4o",
          deltaText: "Uncommitted partial token...",
          toolCalls: [],
          usage: null,
          finishReason: null,
        };
        throw new Error("ERR_NETWORK_LOST");
      },
    };

    const gateway = new EphemeralExecutionGateway({ authToken: "m5-tok" });
    const gwAdapter = new GatewayToolExecutionAdapter(
      {
        baseUrl: "https://gw.example.test",
        authToken: "m5-tok",
        defaultScope: {
          allowedTools: ["sandbox.WriteFile", "sandbox.exec"],
          allowNetwork: false,
          writablePaths: ["dist"],
        },
        defaultLimits: { timeoutMs: 5000, maxOutputBytes: 16_384 },
      },
      createGatewayFetch(gateway),
      gateway.gatewayInstanceId,
    );
    const tools = new BrowserToolRegistry(leaderStore, undefined, gwAdapter);
    const leaderRuntime = new AgentKernelRuntime(
      leaderStore,
      disconnectInference,
      tools,
      tools.getToolDefinitions(),
    );

    const ch1 = new MessageChannel();
    const leaderHost = new KernelWorkerHost(leaderStore, leaderRuntime, ch1.port1);
    const leaderClient = new VelesAppClient(ch1.port2);

    // Observer tab setup (Scenario 5)
    const chObs = new MessageChannel();
    const observerRuntime = new AgentKernelRuntime(
      observerStore,
      disconnectInference,
      tools,
    );
    const observerHost = new KernelWorkerHost(observerStore, observerRuntime, chObs.port1);
    const observerClient = new VelesAppClient(chObs.port2);

    try {
      await leaderClient.initWorkspace("Scenarios Workspace");

      // Scenario 5: Secondary observer tab cannot mutate workspace over MessagePort
      await assert.rejects(
        () => observerClient.startRun("run-obs-illegal", "Mutate from tab 2"),
        /ERR_WORKSPACE_LOCKED_BY_OTHER_TAB/,
      );

      // Scenario 1 & 3: Inference disconnect transitions to interrupted; uncommitted draft is cleared
      const interruptedSnap = await leaderClient.startRun("run-s1", "Stream disconnect");
      assert.equal(interruptedSnap.runState, "interrupted");
      assert.equal(leaderClient.getStateView().streamingDraftText, "");
      assert.equal(interruptedSnap.messages.length, 1);

      // Scenario 6: Storage quota failure preserves committed headSeq
      const seqBeforeQuota = interruptedSnap.headSeq;
      journalBackend.simulateQuotaFailureOnNextCommit(true);
      await assert.rejects(
        () => leaderClient.startRun("run-quota", "Trigger quota error"),
        /ERR_STORAGE_QUOTA_EXCEEDED/,
      );
      const snapAfterQuota = await leaderClient.refreshSnapshot();
      assert.equal(snapAfterQuota.headSeq, seqBeforeQuota);

      // Scenario 2, 4, & 10: Gateway tool execution + reconnect reconciliation over MessagePort
      for await (const _ of gateway.executeToolStream({
        protocolVersion: "1.0",
        operationId: "op-m5-gw-1",
        runId: "run-gw-recover",
        operation: "tool.execute",
        capabilityScope: {
          allowedTools: ["sandbox.WriteFile"],
          allowNetwork: false,
          writablePaths: ["dist"],
        },
        limits: { timeoutMs: 5000, maxOutputBytes: 16_384 },
        payload: {
          toolName: "sandbox.WriteFile",
          effectClass: "non_idempotent",
          host: "gateway",
          args: { path: "dist/bundle.js", content: "console.log('bundled');" },
        },
      })) {
        // Stage 3 completed on gateway; simulate reload before Stage 4 browser commit
      }
      assert.equal(gateway.activeSandboxCount, 0);

      await leaderStore.appendEvent({
        eventId: "ev-gw-1",
        runId: "run-gw-recover",
        type: "RunCreated",
        nextState: "created",
        payload: { prompt: "Bundle app" },
      });
      await leaderStore.appendEvent({
        eventId: "ev-gw-2",
        runId: "run-gw-recover",
        type: "RunPreparing",
        nextState: "preparing",
        payload: {},
      });
      await leaderStore.appendEvent({
        eventId: "ev-gw-3",
        runId: "run-gw-recover",
        type: "CheckpointCommitted",
        nextState: "inferring",
        payload: {},
      });
      await leaderStore.appendEvent({
        eventId: "ev-gw-4",
        runId: "run-gw-recover",
        type: "ToolIntentLogged",
        nextState: "awaiting_tool",
        relatedOperationId: "op-m5-gw-1",
        payload: {
          toolCallId: "tc-gw-1",
          toolName: "sandbox.WriteFile",
          effectClass: "non_idempotent",
        },
      });
      await leaderStore.appendEvent({
        eventId: "ev-gw-5",
        runId: "run-gw-recover",
        type: "ToolDispatched",
        nextState: "executing_tool",
        relatedOperationId: "op-m5-gw-1",
        payload: {
          toolCallId: "tc-gw-1",
          toolName: "sandbox.WriteFile",
          effectClass: "non_idempotent",
          gatewayInstanceId: gateway.gatewayInstanceId,
        },
      });

      // Reconnect via VelesAppClient.recoverBoot against the live gateway instance
      const recovery = await leaderClient.recoverBoot(gateway.gatewayInstanceId);
      assert.equal(recovery.decision?.nextState, "inferring");
      assert.equal(
        recovery.snapshot.files["dist/bundle.js"],
        "console.log('bundled');",
      );

      // Verify Chaos Simulator & Manual Uncertain Tool Resolution (Workbench Telemetry)
      const chaosUncertain = await leaderClient.simulateChaos("uncertain_tool_outcome");
      assert.equal(chaosUncertain.runState, "interrupted");
      assert.equal(chaosUncertain.interruptReason, "uncertain_tool_outcome");
      assert.equal(chaosUncertain.leaseMode, "ReadWrite_Leader");
      assert.ok(chaosUncertain.fileHashes["dist/bundle.js"]);

      const cancelledSnap = await leaderClient.resolveUncertainTool(
        chaosUncertain.activeRunId!,
        "cancel_run",
      );
      assert.equal(cancelledSnap.runState, "cancelled");
    } finally {
      leaderClient.dispose();
      leaderHost.dispose();
      observerClient.dispose();
      observerHost.dispose();
      ch1.port1.close();
      ch1.port2.close();
      chObs.port1.close();
      chObs.port2.close();
    }
  });

  it("boots worker-entry.ts with the built-in veles-demo Counter App flow over MessagePort", async () => {
    const ch = new MessageChannel();
    const host = await bootBrowserKernelWorker(ch.port1, "ws-demo-workbench");
    const client = new VelesAppClient(ch.port2, {
      providerId: "veles-demo",
      baseUrl: "local://demo-comet",
      model: "comet-counter-v1",
    });

    try {
      await client.initWorkspace("Alpha Demo");
      const paused = await client.startRun("run-demo-1", "Build counter app");
      assert.equal(paused.runState, "awaiting_approval");
      assert.equal(paused.pendingApproval?.toolName, "workspace.delete_file");
      assert.ok(paused.files["index.html"]?.includes("Veles Sandboxed Counter"));
      assert.ok(paused.fileHashes["index.html"]);

      const finished = await client.resolveApproval("run-demo-1", true);
      assert.equal(finished.runState, "completed");
      assert.deepEqual(Object.keys(finished.files), ["index.html"]);
      assert.ok(
        finished.sessions?.some(
          (s) => s.id === "session-main" && s.status === "completed",
        ),
        "KernelWorkerHost must expose AgentManager sessions in snapshot",
      );

      // Verify binary Uint8Array upload preserves raw bytes and exposes data:image/png;base64,... in snapshot
      const pngMagicBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
      const withPng = await client.uploadWorkspaceFile("assets/pixel.png", pngMagicBytes);
      assert.ok(withPng.files["assets/pixel.png"]?.startsWith("data:image/png;base64,"));
      assert.ok(withPng.fileHashes["assets/pixel.png"]);
    } finally {
      client.dispose();
      host.dispose();
      ch.port1.close();
      ch.port2.close();
    }
  });

  it("re-acquires per-workspace lock lease on switchWorkspace and transitions to ReadOnly_Observer if target workspace is locked", async () => {
    const locks = new InMemoryLockProvider();
    const tab1Lease = await locks.acquireWorkspaceLease("ws-alpha", "tab-1");
    const tab2LeaseBeta = await locks.acquireWorkspaceLease("ws-beta", "tab-2");
    assert.equal(tab1Lease.mode, "ReadWrite_Leader");
    assert.equal(tab2LeaseBeta.mode, "ReadWrite_Leader");

    const journalBackend = new InMemoryJournalBackend();
    const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
    const tab1Store = new WorkspaceKernelStore(
      journalBackend,
      blobs,
      tab1Lease,
      locks,
    );
    await tab1Store.initWorkspace("Alpha");

    // Tab 1 switches to ws-gamma (unlocked) -> releases ws-alpha and acquires ReadWrite_Leader on ws-gamma
    const gammaMeta = await tab1Store.switchWorkspace("ws-gamma", "Gamma");
    assert.equal(gammaMeta.workspaceId, "ws-gamma");
    assert.equal(tab1Store.getLeaseMode(), "ReadWrite_Leader");

    // Because Tab 1 released ws-alpha, Tab 3 can now acquire ReadWrite_Leader on ws-alpha
    const tab3LeaseAlpha = await locks.acquireWorkspaceLease("ws-alpha", "tab-3");
    assert.equal(tab3LeaseAlpha.mode, "ReadWrite_Leader");

    // Tab 1 switches to ws-beta (held by Tab 2) -> enters ReadOnly_Observer on ws-beta
    await tab1Store.switchWorkspace("ws-beta", "Beta");
    assert.equal(tab1Store.getLeaseMode(), "ReadOnly_Observer");
    await assert.rejects(
      () => tab1Store.writeWorkspaceFile("notes/blocked.txt", "denied"),
      /ERR_WORKSPACE_LOCKED_BY_OTHER_TAB/,
    );
  });

  it("supports multiple chats per workspace (switchChatSession) and deleting user+LLM turns (deleteChatRun) without touching workspace files", async () => {
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease("ws-multichat", "tab-leader");
    const store = new WorkspaceKernelStore(
      new InMemoryJournalBackend(),
      new ContentAddressedBlobStore(new InMemoryBlobBackend()),
      lease,
      locks,
    );

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        const lastUser = [...req.messages].reverse().find((m) => m.role === "user");
        const promptText = lastUser?.content ?? "";
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: `Echo: ${promptText}`,
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const tools = new BrowserToolRegistry(store);
    const runtime = new AgentKernelRuntime(
      store,
      inference,
      tools,
      tools.getToolDefinitions(),
    );

    const ch = new MessageChannel();
    const host = new KernelWorkerHost(store, runtime, ch.port1);
    const client = new VelesAppClient(ch.port2);

    try {
      await client.initWorkspace("Multi-Chat Workspace");
      await client.uploadWorkspaceFile("preview/index.html", "<h1>Shared File</h1>");

      // 1. Run a turn in Main Chat ("session-main")
      const mainSnap1 = await client.startRun("run-main-1", "Hello from Main Chat");
      assert.equal(mainSnap1.activeSessionId, "session-main");
      assert.equal(mainSnap1.messages.length, 2);
      assert.equal(mainSnap1.messages[0]?.runId, "run-main-1");
      assert.equal(mainSnap1.messages[1]?.content, "Echo: Hello from Main Chat");

      // 2. Switch to a second chat in the same workspace
      const chat2Init = await client.switchChatSession("chat-2", "Design Chat");
      assert.equal(chat2Init.activeSessionId, "chat-2");
      assert.equal(chat2Init.messages.length, 0);
      assert.equal(chat2Init.files["preview/index.html"], "<h1>Shared File</h1>");
      assert.ok(
        chat2Init.sessions?.some((s) => s.id === "chat-2" && s.title === "Design Chat"),
      );

      // 3. Run two turns in chat-2
      await client.startRun("run-chat2-1", "First turn in Chat 2");
      const chat2After2 = await client.startRun("run-chat2-2", "Second turn in Chat 2");
      assert.equal(chat2After2.messages.length, 4);

      // 4. Switch back to Main Chat -> only sees Main Chat messages
      const backToMain = await client.switchChatSession("session-main");
      assert.equal(backToMain.activeSessionId, "session-main");
      assert.equal(backToMain.messages.length, 2);
      assert.equal(backToMain.messages[0]?.content, "Hello from Main Chat");

      // 5. Switch back to Chat 2 and delete the first turn ("run-chat2-1")
      await client.switchChatSession("chat-2");
      const afterDelete = await client.deleteChatRun("run-chat2-1");
      assert.equal(afterDelete.messages.length, 2);
      assert.equal(afterDelete.messages[0]?.runId, "run-chat2-2");
      assert.equal(afterDelete.messages[0]?.content, "Second turn in Chat 2");
      // Workspace files are never touched when deleting a chat turn without revertFiles
      assert.equal(afterDelete.files["preview/index.html"], "<h1>Shared File</h1>");

      // Verify deleteChatRun(..., { revertFiles: true }) restores files modified by a turn back to pre-turn SHA-256
      await store.appendEvent({
        eventId: "ev-revert-1",
        runId: "run-file-edit",
        type: "RunCreated",
        nextState: "created",
        payload: {
          prompt: "Overwrite index.html and add temp.txt",
          sessionId: "chat-2",
          preRunFileTree: await store.getFileTree(),
        },
      });
      await store.appendEvent({
        eventId: "ev-revert-2",
        runId: "run-file-edit",
        type: "RunPreparing",
        nextState: "preparing",
        payload: {},
      });
      await store.appendEvent({
        eventId: "ev-revert-3",
        runId: "run-file-edit",
        type: "CheckpointCommitted",
        nextState: "inferring",
        payload: {},
      });
      await store.appendEvent({
        eventId: "ev-revert-4",
        runId: "run-file-edit",
        type: "ToolIntentLogged",
        nextState: "awaiting_tool",
        relatedOperationId: "op-rev-1",
        payload: {
          toolCallId: "tc-rev-1",
          toolName: "workspace.write_file",
          args: { path: "preview/index.html", content: "<h1>Modified in turn</h1>" },
        },
      });
      await store.appendEvent({
        eventId: "ev-revert-5",
        runId: "run-file-edit",
        type: "ToolDispatched",
        nextState: "executing_tool",
        relatedOperationId: "op-rev-1",
        payload: { toolCallId: "tc-rev-1", toolName: "workspace.write_file" },
      });
      await store.appendEvent({
        eventId: "ev-revert-6",
        runId: "run-file-edit",
        type: "ToolResultCommitted",
        nextState: "inferring",
        relatedOperationId: "op-rev-1",
        payload: {
          toolCallId: "tc-rev-1",
          toolName: "workspace.write_file",
          outputText: "Updated",
          observedFileHashes: {
            "preview/index.html": await store.computeBlobHashForPreview("<h1>Modified in turn</h1>"),
            "preview/new-temp.txt": await store.computeBlobHashForPreview("temp"),
          },
        },
        fileUpdates: {
          "preview/index.html": "<h1>Modified in turn</h1>",
          "preview/new-temp.txt": "temp",
        },
      });
      await store.appendEvent({
        eventId: "ev-revert-7",
        runId: "run-file-edit",
        type: "RunCompleted",
        nextState: "completed",
        payload: { assistantText: "Modified files", sessionId: "chat-2" },
      });
      const beforeRevert = await client.refreshSnapshot();
      assert.equal(beforeRevert.files["preview/index.html"], "<h1>Modified in turn</h1>");
      assert.equal(beforeRevert.files["preview/new-temp.txt"], "temp");

      const afterRevert = await client.deleteChatRun("run-file-edit", { revertFiles: true });
      assert.equal(afterRevert.files["preview/index.html"], "<h1>Shared File</h1>");
      assert.equal(afterRevert.files["preview/new-temp.txt"], undefined);

      // 6. Export and re-import bundle to verify tombstone record preserves monotonic seq
      const bundle = await client.exportWorkspace();
      const reimported = await client.importWorkspace(bundle);
      assert.equal(reimported.files["preview/index.html"], "<h1>Shared File</h1>");
    } finally {
      client.dispose();
      host.dispose();
      ch.port1.close();
      ch.port2.close();
    }
  });
});
