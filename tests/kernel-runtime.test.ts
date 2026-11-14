import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  OpenAICompatibleAdapter,
  type FetchLike,
  type InferenceAdapter,
  type ProviderModelConfig,
} from "../src/inference/openai-compatible.ts";
import {
  ContentAddressedBlobStore,
  InMemoryBlobBackend,
} from "../src/kernel/blobs.ts";
import {
  InMemoryJournalBackend,
  WorkspaceKernelStore,
} from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import {
  AgentKernelRuntime,
  ContextBuilder,
  type ToolExecutionAdapter,
  validateWorkerCommand,
} from "../src/kernel/runtime.ts";
import {
  AgentManager,
  type AgentSession,
} from "../src/kernel/agent-session.ts";

function makeSseReadableStream(chunks: readonly string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let idx = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (idx >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(chunks[idx]!));
      idx += 1;
    },
  });
}

async function createTestStore(workspaceId = "ws-runtime") {
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease(workspaceId, "worker-1");
  const journalBackend = new InMemoryJournalBackend();
  const blobBackend = new InMemoryBlobBackend();
  const blobs = new ContentAddressedBlobStore(blobBackend);
  const store = new WorkspaceKernelStore(journalBackend, blobs, lease);
  await store.initWorkspace("Runtime Test Workspace");
  return { store, journalBackend, blobBackend, lease };
}

describe("SYS-KERNEL & SYS-INFERENCE Worker Runtime (SSE Adapter, COMET Loop & Scenarios 1, 2, 3, 4, 9)", () => {
  it("OpenAICompatibleAdapter normalizes split SSE chunks, tool call deltas, usage, and AbortSignal", async () => {
    const rawSseChunks = [
      'data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null}]}\n\n',
      'data: {"choices":[{"delta":{"content":"world!","tool_calls":[{"index":0,"id":"call_1","function":{"name":"workspace__","arguments":"{\\"path\\":"}}]}}]}\n',
      '\ndata: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"\\"a.txt\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
      'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
      "data: [DONE]\n\n",
    ];

    let capturedBody: Record<string, unknown> | null = null;
    const fakeFetch: FetchLike = async (_url, init) => {
      if (init.signal?.aborted) {
        throw new Error("Aborted");
      }
      capturedBody = JSON.parse(init.body) as Record<string, unknown>;
      return {
        ok: true,
        status: 200,
        body: makeSseReadableStream(rawSseChunks),
      };
    };

    const adapter = new OpenAICompatibleAdapter(fakeFetch);
    const config: ProviderModelConfig = {
      providerId: "openai",
      baseUrl: "https://api.example.test/v1/",
      model: "gpt-4o-mini",
      apiKey: "test-token",
    };

    const collected = [];
    for await (const chunk of adapter.streamChat({
      config,
      tools: [
        {
          name: "workspace.write_file",
          description: "Write a file",
          effectClass: "idempotent_write",
        },
      ],
      messages: [
        { role: "user", content: "Write a.txt" },
        {
          role: "assistant",
          content: "Calling workspace.write_file",
          toolCalls: [
            { id: "call_0", name: "workspace.write_file", argumentsJson: '{"path":"a.txt"}' },
          ],
        },
        { role: "tool", content: "ok", toolCallId: "call_0" },
      ],
    })) {
      collected.push(chunk);
    }

    const wireMessages = (capturedBody as { messages?: Array<Record<string, unknown>> } | null)
      ?.messages;
    assert.ok(wireMessages);
    assert.deepEqual(wireMessages[1]?.tool_calls, [
      {
        id: "call_0",
        type: "function",
        function: { name: "workspace__write_file", arguments: '{"path":"a.txt"}' },
      },
    ]);
    assert.equal(wireMessages[2]?.tool_call_id, "call_0");

    assert.deepEqual((capturedBody as Record<string, unknown> | null)?.stream_options, { include_usage: true });
    assert.equal(collected.length, 4);
    assert.equal(collected[0]!.deltaText, "Hello ");
    assert.equal(collected[1]!.deltaText, "world!");
    const finalChunk = collected[2]!;
    assert.equal(finalChunk.finishReason, "tool_calls");
    assert.deepEqual(finalChunk.toolCalls, [
      { id: "call_1", name: "workspace.write_file", argumentsJson: '{"path":"a.txt"}' },
    ]);
    assert.equal(finalChunk.usage, null);
    assert.deepEqual(collected[3]!.usage, {
      promptTokens: 10,
      completionTokens: 5,
      totalTokens: 15,
    });

    // AbortSignal cancellation check
    const abortCtrl = new AbortController();
    abortCtrl.abort();
    await assert.rejects(
      async () => {
        for await (const _ of adapter.streamChat({
          config,
          messages: [{ role: "user", content: "Cancel me" }],
          signal: abortCtrl.signal,
        })) {
          // should not yield
        }
      },
      /ERR_INFERENCE_ABORTED/,
    );
  });

  it("executes full COMET tool loop (ToolIntentLogged -> ToolDispatched -> ToolResultCommitted -> RunCompleted)", async () => {
    const { store } = await createTestStore("ws-comet");
    let round = 0;

    const fakeInference: InferenceAdapter = {
      async *streamChat(req) {
        round += 1;
        if (round === 1) {
          assert.equal(req.messages.length, 1);
          assert.equal(req.messages[0]!.content, "Create greeting.txt");
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Writing file...",
            toolCalls: [
              {
                id: "tc-1",
                name: "fs.write",
                argumentsJson: JSON.stringify({ path: "greeting.txt" }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        // Round 2: ContextBuilder must include assistant toolCalls followed by committed ToolResultCommitted
        assert.equal(req.messages.length, 3);
        assert.deepEqual(req.messages[1], {
          role: "assistant",
          content: "Writing file...",
          toolCalls: [
            {
              id: "tc-1",
              name: "fs.write",
              argumentsJson: JSON.stringify({ path: "greeting.txt" }),
            },
          ],
        });
        assert.deepEqual(req.messages[2], {
          role: "tool",
          content: "Wrote greeting.txt",
          toolCallId: "tc-1",
        });
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Done creating greeting.txt!",
          toolCalls: [],
          usage: { promptTokens: 20, completionTokens: 6, totalTokens: 26 },
          finishReason: "stop",
        };
      },
    };

    const toolAdapter: ToolExecutionAdapter = {
      currentGatewayInstanceId: "gw-live-1",
      async executeTool(req) {
        // Verify KERN-INV-01: ToolDispatched is ALREADY committed in journal when executeTool runs
        const journalNow = await store.getJournal();
        assert.equal(journalNow[journalNow.length - 1]?.type, "ToolDispatched");
        assert.equal(req.toolName, "fs.write");
        return {
          gatewayInstanceId: "gw-live-1",
          outputText: "Wrote greeting.txt",
          fileUpdates: { "greeting.txt": "Hello from Veles!" },
        };
      },
    };

    const runtime = new AgentKernelRuntime(store, fakeInference, toolAdapter, [
      {
        name: "fs.write",
        description: "Write file",
        effectClass: "idempotent_write",
      },
    ]);

    const finalState = await runtime.startRun("run-1", "Create greeting.txt", {
      providerId: "openai",
      baseUrl: "https://api.example.test/v1",
      model: "gpt-4o",
    });

    assert.equal(finalState, "completed");
    assert.equal(await store.readFileText("greeting.txt"), "Hello from Veles!");

    const journal = await store.getJournal();
    assert.deepEqual(
      journal.map((r) => r.type),
      [
        "RunCreated",
        "RunPreparing",
        "CheckpointCommitted",
        "ToolIntentLogged",
        "ToolDispatched",
        "ToolResultCommitted",
        "RunCompleted",
      ],
    );

    // Verify ApprovalRequested records toolCallId in payload for interactive approval resumption
    const approvalInference: InferenceAdapter = {
      async *streamChat(req) {
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Requesting delete...",
          toolCalls: [
            {
              id: "tc-del-99",
              name: "fs.delete",
              argumentsJson: JSON.stringify({ path: "greeting.txt" }),
            },
          ],
          usage: null,
          finishReason: "tool_calls",
        };
      },
    };
    const approvalRuntime = new AgentKernelRuntime(
      store,
      approvalInference,
      toolAdapter,
      [
        {
          name: "fs.delete",
          description: "Delete file",
          effectClass: "non_idempotent",
          requiresApproval: true,
        },
      ],
    );
    const approvalState = await approvalRuntime.startRun(
      "run-approval-1",
      "Delete greeting.txt",
      {
        providerId: "openai",
        baseUrl: "https://api.example.test/v1",
        model: "gpt-4o",
      },
    );
    assert.equal(approvalState, "awaiting_approval");
    const afterApprovalJournal = await store.getJournal();
    const approvalRec = afterApprovalJournal[afterApprovalJournal.length - 1]!;
    assert.equal(approvalRec.type, "ApprovalRequested");
    assert.equal(approvalRec.payload.toolCallId, "tc-del-99");
  });

  it("Scenarios 1, 2, 3, 4: recovers deterministically from inference loss, uncertain tool outcomes, and reload", async () => {
    const { store } = await createTestStore("ws-recovery");

    // Scenario 1: Gateway/stream loss during inference transitions to interrupted (safe to retry)
    const failingInference: InferenceAdapter = {
      async *streamChat() {
        yield {
          providerId: "openai",
          model: "gpt-4o",
          deltaText: "Uncommitted partial stream delta...",
          toolCalls: [],
          usage: null,
          finishReason: null,
        };
        throw new Error("ERR_STREAM_DISCONNECTED");
      },
    };
    const noopTools: ToolExecutionAdapter = {
      currentGatewayInstanceId: "gw-1",
      async executeTool() {
        throw new Error("unreachable");
      },
    };

    const runtime1 = new AgentKernelRuntime(store, failingInference, noopTools);
    const state1 = await runtime1.startRun("run-inf-loss", "Explain Veles", {
      providerId: "openai",
      baseUrl: "https://api.example.test/v1",
      model: "gpt-4o",
    });
    assert.equal(state1, "interrupted");
    const journal1 = await store.getJournal();
    const tail1 = journal1[journal1.length - 1]!;
    assert.equal(tail1.type, "RunInterrupted");
    assert.equal(tail1.payload.interruptReason, "gateway_disconnected_safe_to_retry");

    // Scenario 3: Uncommitted streaming deltas never pollute ContextBuilder
    const rebuiltMessages = ContextBuilder.buildMessages(journal1, "run-inf-loss");
    assert.equal(rebuiltMessages.length, 1);
    assert.equal(rebuiltMessages[0]!.content, "Explain Veles");

    // Scenario 2 & 4: Simulate browser crash/reload right after non-idempotent ToolDispatched
    await store.appendEvent({
      eventId: "ev-manual-create",
      runId: "run-crash-tool",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "Send webhook" },
    });
    await store.appendEvent({
      eventId: "ev-manual-prep",
      runId: "run-crash-tool",
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await store.appendEvent({
      eventId: "ev-manual-inf",
      runId: "run-crash-tool",
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: {},
    });
    await store.appendEvent({
      eventId: "ev-manual-intent",
      runId: "run-crash-tool",
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: "op-webhook-1",
      payload: { toolName: "webhook.send", effectClass: "non_idempotent" },
    });
    await store.appendEvent({
      eventId: "ev-manual-dispatched",
      runId: "run-crash-tool",
      type: "ToolDispatched",
      nextState: "executing_tool",
      relatedOperationId: "op-webhook-1",
      payload: {
        toolName: "webhook.send",
        effectClass: "non_idempotent",
        gatewayInstanceId: "gw-old",
      },
    });

    // Boot recovery against a restarted gateway ("gw-new" !== "gw-old")
    const decision = await runtime1.recoverUnfinishedRun("gw-new");
    assert.ok(decision);
    assert.equal(decision.nextState, "interrupted");
    assert.equal(decision.interruptReason, "uncertain_tool_outcome");
    assert.equal(decision.canAutoRetry, false);
    assert.equal((await store.getMetadata())?.runState, "interrupted");
  });

  it("Scenario 9: provider/model switching preserves run isolation and validates Worker commands", async () => {
    const { store } = await createTestStore("ws-isolation");
    const observedConfigs: ProviderModelConfig[] = [];

    const mutableConfig: ProviderModelConfig = {
      providerId: "anthropic-compat",
      baseUrl: "https://provider-a.example.test/v1",
      model: "claude-3-7-sonnet",
    };

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        // Mutate external config mid-run to verify run snapshot isolation
        (mutableConfig as { model: string }).model = "mutated-mid-stream";
        observedConfigs.push(req.config);
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: `Reply from ${req.config.providerId}:${req.config.model}`,
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const noopTools: ToolExecutionAdapter = {
      currentGatewayInstanceId: null,
      async executeTool() {
        throw new Error("unused");
      },
    };

    const runtime = new AgentKernelRuntime(store, inference, noopTools);

    await runtime.handleWorkerCommand({
      type: "START_RUN",
      runId: "run-a",
      prompt: "First run",
      providerConfig: {
        providerId: "provider-a",
        baseUrl: "https://a.example.test/v1",
        model: "model-a",
      },
    });

    await runtime.handleWorkerCommand({
      type: "START_RUN",
      runId: "run-b",
      prompt: "Second run",
      providerConfig: {
        providerId: "provider-b",
        baseUrl: "https://b.example.test/v1",
        model: "model-b",
      },
    });

    assert.equal(observedConfigs.length, 2);
    assert.equal(observedConfigs[0]!.providerId, "provider-a");
    assert.equal(observedConfigs[0]!.model, "model-a");
    assert.equal(observedConfigs[1]!.providerId, "provider-b");
    assert.equal(observedConfigs[1]!.model, "model-b");

    const journal = await store.getJournal();
    const runACompleted = journal.find(
      (r) => r.runId === "run-a" && r.type === "RunCompleted",
    );
    const runBCompleted = journal.find(
      (r) => r.runId === "run-b" && r.type === "RunCompleted",
    );
    assert.equal(runACompleted?.payload.providerId, "provider-a");
    assert.equal(runACompleted?.payload.model, "model-a");
    assert.equal(runBCompleted?.payload.providerId, "provider-b");
    assert.equal(runBCompleted?.payload.model, "model-b");

    // Boundary validation rejects malformed worker commands
    assert.throws(() => validateWorkerCommand(null), /ERR_INVALID_WORKER_COMMAND/);
    assert.throws(
      () => validateWorkerCommand({ type: "START_RUN", runId: "" }),
      /ERR_INVALID_WORKER_COMMAND/,
    );
    assert.throws(
      () => validateWorkerCommand({ type: "UNKNOWN_CMD" }),
      /ERR_INVALID_WORKER_COMMAND/,
    );

    // 4-Layer Tiered ContextBuilder: pins OPFS file tree and folds runs older than hotRunWindow
    const longCode = "console.log('mario');\n".repeat(20);
    const tieredMsgs = ContextBuilder.buildMessages(
      [
        {
          seq: 1,
          eventId: "e1",
          workspaceId: "ws",
          runId: "run-old",
          type: "RunCreated",
          schemaVersion: 1,
          relatedOperationId: null,
          committedState: "created",
          timestampMs: 1,
          payload: { prompt: "Build Mario" },
        },
        {
          seq: 2,
          eventId: "e2",
          workspaceId: "ws",
          runId: "run-old",
          type: "ToolIntentLogged",
          schemaVersion: 1,
          relatedOperationId: "op-1",
          committedState: "awaiting_tool",
          timestampMs: 2,
          payload: {
            toolCallId: "c1",
            toolName: "workspace.write_file",
            args: { path: "preview/script.js", content: longCode },
          },
        },
        {
          seq: 3,
          eventId: "e3",
          workspaceId: "ws",
          runId: "run-hot",
          type: "RunCreated",
          schemaVersion: 1,
          relatedOperationId: null,
          committedState: "created",
          timestampMs: 3,
          payload: { prompt: "Make Mario jump higher" },
        },
      ],
      {
        hotRunWindow: 1,
        fileTree: { "preview/script.js": "abcdef0123456789" },
      },
    );
    assert.equal(tieredMsgs[0]?.role, "system");
    assert.ok(tieredMsgs[0]?.content.includes("preview/script.js [sha256:abcdef01]"));
    assert.ok(
      tieredMsgs[2]?.toolCalls?.[0]?.argumentsJson.includes(
        "read from OPFS if needed",
      ),
    );
  });

  it("AgentSession & AgentManager: isolates interleaved sessions, enforces allowedTools, keeps tool pairs intact on compaction, and recovers across restart", async () => {
    const { store } = await createTestStore("ws-multi-agent");

    const observedBySession = new Map<
      string,
      { messages: unknown[]; tools: string[] }
    >();

    const multiAgentInference: InferenceAdapter = {
      async *streamChat(req) {
        const sessionKey = req.config.model;
        const toolNames = (req.tools ?? []).map((t) => t.name);
        observedBySession.set(sessionKey, {
          messages: [...req.messages],
          tools: toolNames,
        });

        const lastMsg = req.messages[req.messages.length - 1];
        if (lastMsg?.role === "user" && lastMsg.content.includes("use-tool")) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: `Calling tool from ${sessionKey}`,
            toolCalls: [
              {
                id: `call-${sessionKey}`,
                name: "workspace.read_file",
                argumentsJson: JSON.stringify({ path: "shared.txt" }),
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
          deltaText: `Response from ${sessionKey} (${req.messages.length} msgs in context)`,
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const toolAdapter: ToolExecutionAdapter = {
      currentGatewayInstanceId: "gw-multi",
      async executeTool(req) {
        return {
          gatewayInstanceId: "gw-multi",
          outputText: `Content of ${String(req.args.path)}`,
        };
      },
    };

    const toolSpecs = [
      {
        name: "workspace.read_file",
        description: "Read file",
        effectClass: "pure_read" as const,
      },
      {
        name: "workspace.delete_file",
        description: "Delete file",
        effectClass: "non_idempotent" as const,
      },
    ];

    const runtime = new AgentKernelRuntime(
      store,
      multiAgentInference,
      toolAdapter,
      toolSpecs,
    );
    const manager = new AgentManager(store, runtime);

    const sessionA: AgentSession = {
      id: "agent-alpha",
      workspaceId: "ws-multi-agent",
      model: {
        providerId: "openai",
        baseUrl: "https://api.example.test/v1",
        model: "model-alpha",
      },
      allowedTools: ["workspace.read_file"],
      status: "idle",
    };

    const sessionB: AgentSession = {
      id: "agent-beta",
      parentSessionId: "agent-alpha",
      workspaceId: "ws-multi-agent",
      model: {
        providerId: "anthropic",
        baseUrl: "https://api.example.test/v1",
        model: "model-beta",
      },
      allowedTools: [],
      status: "idle",
    };

    await manager.createSession(sessionA);
    await manager.createSession(sessionB);

    // 1. Interleave runs between Agent A and Agent B on the same WorkspaceKernelStore
    const outA1 = await manager.run("agent-alpha", "Alpha turn 1: use-tool");
    assert.ok(outA1.includes("Response from model-alpha"));

    const outB1 = await manager.run("agent-beta", "Beta secret prompt: build shader");
    assert.ok(outB1.includes("Response from model-beta"));

    const outA2 = await manager.run("agent-alpha", "Alpha turn 2: summarize");
    assert.ok(outA2.includes("Response from model-alpha"));

    // 2. Verify neither agent sees the other's private conversation history or disallowed tools
    const alphaObserved = observedBySession.get("model-alpha");
    const betaObserved = observedBySession.get("model-beta");
    assert.ok(alphaObserved);
    assert.ok(betaObserved);

    assert.deepEqual(alphaObserved.tools, ["workspace.read_file"]);
    assert.deepEqual(betaObserved.tools, []);

    const alphaMsgs = await manager.getSessionMessages("agent-alpha");
    const betaMsgs = await manager.getSessionMessages("agent-beta");
    assert.ok(
      alphaMsgs.every((m) => !m.content.includes("Beta secret prompt")),
      "Agent Alpha must never see Agent Beta's messages",
    );
    assert.ok(
      betaMsgs.every((m) => !m.content.includes("Alpha turn")),
      "Agent Beta must never see Agent Alpha's messages",
    );

    // 3. Verify ContextBuilder.enforceCharBudget keeps tool-call/result pairs intact and protects active turn
    const compactedAlpha = await manager.getSessionMessages("agent-alpha", {
      maxCharBudget: 60,
    });
    for (let i = 0; i < compactedAlpha.length; i++) {
      if (compactedAlpha[i]?.role === "tool") {
        assert.equal(
          compactedAlpha[i - 1]?.role,
          "assistant",
          "Orphan tool message without preceding assistant tool_call is forbidden",
        );
      }
    }
    assert.equal(
      compactedAlpha[compactedAlpha.length - 2]?.content,
      "Alpha turn 2: summarize",
    );

    // 4. Create an idle session (never run) and verify default ContextBuilder excludes child sessions
    await manager.createSession({
      id: "agent-gamma-idle",
      parentSessionId: "agent-alpha",
      workspaceId: "ws-multi-agent",
      model: {
        providerId: "openai",
        baseUrl: "https://api.example.test/v1",
        model: "model-gamma",
      },
      allowedTools: ["workspace.read_file"],
      status: "idle",
    });

    // Verify concurrent manager.run() calls are serialized cleanly without state corruption
    const [concurrentA, concurrentB] = await Promise.all([
      manager.run("agent-alpha", "Alpha concurrent turn"),
      manager.run("agent-beta", "Beta secret prompt: concurrent turn"),
    ]);
    assert.ok(concurrentA.includes("Response from model-alpha"));
    assert.ok(concurrentB.includes("Response from model-beta"));

    // Verify default top-level ContextBuilder.buildMessages(journal) excludes child sessions (parentSessionId)
    const fullJournal = await store.getJournal();
    const topLevelMsgs = ContextBuilder.buildMessages(fullJournal);
    assert.ok(
      topLevelMsgs.every((m) => !m.content.includes("Beta secret prompt")),
      "Default top-level ContextBuilder must exclude child session runs (parentSessionId)",
    );

    // 5. Simulate Worker Restart: create a fresh AgentManager on the same store and recover sessions (including idle)
    const restartedRuntime = new AgentKernelRuntime(
      store,
      multiAgentInference,
      toolAdapter,
      toolSpecs,
    );
    const restartedManager = new AgentManager(store, restartedRuntime);
    await restartedManager.recoverSessionsFromJournal();

    const recoveredA = restartedManager.getSession("agent-alpha");
    const recoveredB = restartedManager.getSession("agent-beta");
    const recoveredIdle = restartedManager.getSession("agent-gamma-idle");
    assert.ok(recoveredA);
    assert.ok(recoveredB);
    assert.ok(recoveredIdle, "Idle session must survive worker restart");
    assert.equal(recoveredA.status, "completed");
    assert.deepEqual(recoveredA.allowedTools, ["workspace.read_file"]);
    assert.equal(recoveredB.status, "completed");
    assert.equal(recoveredB.parentSessionId, "agent-alpha");
    assert.deepEqual(recoveredB.allowedTools, []);
    assert.equal(recoveredIdle.status, "idle");
    assert.equal(recoveredIdle.parentSessionId, "agent-alpha");
    assert.deepEqual(recoveredIdle.allowedTools, ["workspace.read_file"]);

    // 6. Verify writeWorkspaceFile advances headSeq, records CheckpointCommitted, and rejects unsafe paths
    const beforeMeta = await store.getMetadata();
    assert.ok(beforeMeta);
    await store.writeWorkspaceFile("assets/logo.svg", "<svg></svg>");
    const afterMeta = await store.getMetadata();
    assert.equal(afterMeta?.headSeq, beforeMeta.headSeq + 1);
    await assert.rejects(
      () => store.writeWorkspaceFile("./unsafe.txt", "bad"),
      /ERR_UNSAFE_FILE_PATH/,
    );
  });
});

