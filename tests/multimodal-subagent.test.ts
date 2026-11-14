import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  OpenAICompatibleAdapter,
  type FetchLike,
  type InferenceAdapter,
  type ProviderModelConfig,
} from "../src/inference/openai-compatible.ts";
import { AgentManager } from "../src/kernel/agent-session.ts";
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
} from "../src/kernel/runtime.ts";
import {
  BrowserToolRegistry,
  createSubagentDelegationTool,
} from "../src/kernel/tools.ts";

const TEST_MODEL_CONFIG: ProviderModelConfig = {
  providerId: "openai",
  baseUrl: "https://api.example.test/v1",
  model: "gpt-4o",
};

async function createTestWorkspace(id = "ws-mm-sub") {
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease(id, "worker-mm");
  const journalBackend = new InMemoryJournalBackend();
  const blobBackend = new InMemoryBlobBackend();
  const store = new WorkspaceKernelStore(
    journalBackend,
    new ContentAddressedBlobStore(blobBackend),
    lease,
    locks,
  );
  await store.initWorkspace("Multimodal & Subagent Workspace");
  return { store, journalBackend, blobBackend };
}

describe("Multimodal Vision (workspace.inspect_image), Subagent Delegation (agent.delegate), & Gateway Tool Catalog", () => {
  it("workspace.inspect_image loads OPFS PNG/SVG into multimodal image_url content parts for vision LLMs", async () => {
    const { store } = await createTestWorkspace("ws-vision");
    const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    await store.writeWorkspaceFile("assets/horse.png", pngBytes);

    const tools = new BrowserToolRegistry(store);
    const capturedWireBodies: Array<Record<string, unknown>> = [];
    let callCount = 0;

    const fakeFetch: FetchLike = async (_url, init) => {
      callCount += 1;
      capturedWireBodies.push(JSON.parse(init.body) as Record<string, unknown>);
      const encoder = new TextEncoder();
      const ssePayload =
        callCount === 1
          ? 'data: {"choices":[{"delta":{"content":"Inspecting horse.png...","tool_calls":[{"index":0,"id":"call_img_1","function":{"name":"workspace__inspect_image","arguments":"{\\"path\\":\\"assets/horse.png\\"}"}}]},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
          : 'data: {"choices":[{"delta":{"content":"The image shows a horse sprite."},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
      return {
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(ctrl) {
            ctrl.enqueue(encoder.encode(ssePayload));
            ctrl.close();
          },
        }),
      };
    };

    const adapter = new OpenAICompatibleAdapter(fakeFetch);
    const runtime = new AgentKernelRuntime(
      store,
      adapter,
      tools,
      tools.getToolDefinitions(),
      { multiTurnWorkspaceContext: true },
    );

    const state = await runtime.startRun(
      "run-vision-1",
      "Take assets/horse.png in our workspace and describe it",
      TEST_MODEL_CONFIG,
    );
    assert.equal(state, "completed");
    assert.equal(capturedWireBodies.length, 2);

    // Verify Turn 2 wire request contains a multimodal user message with type: "image_url"
    const turn2Messages = capturedWireBodies[1]!.messages as Array<
      Record<string, unknown>
    >;
    const visionMsg = turn2Messages.find((m) => Array.isArray(m.content));
    assert.ok(visionMsg, "Expected multimodal content array in Turn 2 wire payload");
    assert.equal(visionMsg.role, "user");
    const parts = visionMsg.content as Array<Record<string, unknown>>;
    const imagePart = parts.find((p) => p.type === "image_url") as
      | { image_url?: { url?: string } }
      | undefined;
    assert.ok(imagePart?.image_url?.url?.startsWith("data:image/png;base64,"));
  });

  it("agent.delegate spawns an isolated child AgentSession with custom model & scoped tools without deadlocking parent", async () => {
    const { store } = await createTestWorkspace("ws-delegate");
    const tools = new BrowserToolRegistry(store);

    const observedModels: string[] = [];
    const inference: InferenceAdapter = {
      async *streamChat(req) {
        observedModels.push(`${req.config.providerId}:${req.config.model}`);
        const lastMsg = req.messages[req.messages.length - 1];

        // 1. Parent turn 1: delegates physics implementation to child subagent
        if (
          req.config.model === "gpt-4o" &&
          lastMsg?.role === "user" &&
          lastMsg.content.includes("Build Mario game")
        ) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Delegating physics script to specialist...",
            toolCalls: [
              {
                id: "call_del_sub_1",
                name: "agent.delegate",
                argumentsJson: JSON.stringify({
                  role: "physics-coder",
                  task: "Write preview/script.js with jump=42",
                  model: "qwen2.5-coder:7b",
                  providerId: "ollama",
                  allowedTools: ["workspace.write_file", "agent.delegate"],
                }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }

        // 2. Child session turn 1: verifies agent.delegate was stripped from its tools, writes file
        if (req.config.model === "qwen2.5-coder:7b") {
          const childToolNames = (req.tools ?? []).map((t) => t.name);
          assert.equal(
            childToolNames.includes("agent.delegate"),
            false,
            "Child session must not receive agent.delegate (recursion guard)",
          );
          if (lastMsg?.role === "user") {
            yield {
              providerId: req.config.providerId,
              model: req.config.model,
              deltaText: "Child writing preview/script.js...",
              toolCalls: [
                {
                  id: "call_child_write",
                  name: "workspace.write_file",
                  argumentsJson: JSON.stringify({
                    path: "preview/script.js",
                    content: "window.jumpVelocity = 42;",
                  }),
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
            deltaText: "Physics module written to preview/script.js.",
            toolCalls: [],
            usage: null,
            finishReason: "stop",
          };
          return;
        }

        // 3. Parent turn 2: receives child summary and finishes
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Parent verified child completion!",
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      tools,
      tools.getToolDefinitions(),
      { multiTurnWorkspaceContext: true },
    );
    const manager = new AgentManager(store, runtime);
    const delegateTool = createSubagentDelegationTool(manager);
    tools.registerTool(delegateTool);
    runtime.registerToolDefinition(delegateTool.spec);

    await manager.createSession({
      id: "session-main",
      workspaceId: "ws-delegate",
      model: TEST_MODEL_CONFIG,
      allowedTools: [],
      status: "idle",
    });

    const finalReply = await manager.run("session-main", "Build Mario game");
    assert.equal(finalReply, "Parent verified child completion!");
    assert.equal(
      await store.readFileText("preview/script.js"),
      "window.jumpVelocity = 42;",
    );
    assert.deepEqual(observedModels, [
      "openai:gpt-4o",
      "ollama:qwen2.5-coder:7b",
      "ollama:qwen2.5-coder:7b",
      "openai:gpt-4o",
    ]);

    // Verify child session was recorded with parentSessionId = "session-main"
    const allSessions = manager.listSessions();
    const childSession = allSessions.find((s) => s.id.startsWith("sub-physics-coder-"));
    assert.ok(childSession);
    assert.equal(childSession.parentSessionId, "session-main");
    assert.equal(childSession.status, "completed");

    // Verify BrowserToolRegistry merges discovered gateway tool specs without overwriting local tools
    tools.registerGatewayToolSpecs([
      {
        name: "image.transform_flux",
        description: "Transform workspace image using Flux 4B LoRA on gateway",
        effectClass: "idempotent_write",
      },
    ]);
    assert.equal(
      tools.getToolDefinitions().some((t) => t.name === "image.transform_flux"),
      true,
    );
  });

  it("hardens parent error recovery after child run updates meta.activeRunId and budgets imageDataUrls weight", async () => {
    const { store } = await createTestWorkspace("ws-edge-cases");
    const tools = new BrowserToolRegistry(store);

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        const lastMsg = req.messages[req.messages.length - 1];
        if (
          req.config.model === "gpt-4o" &&
          lastMsg?.role === "user" &&
          lastMsg.content.includes("Delegate then fail")
        ) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Delegating to child first...",
            toolCalls: [
              {
                id: "call_del_fail_parent",
                name: "agent.delegate",
                argumentsJson: JSON.stringify({
                  role: "helper",
                  task: "Complete child work cleanly",
                  model: "qwen2.5-coder:7b",
                  providerId: "ollama",
                }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        if (req.config.model === "qwen2.5-coder:7b") {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Child completed ok.",
            toolCalls: [],
            usage: null,
            finishReason: "stop",
          };
          return;
        }
        // Parent Turn 2 throws network error after child updated meta.activeRunId
        throw new Error("Simulated parent turn 2 inference drop");
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      tools,
      tools.getToolDefinitions(),
      { multiTurnWorkspaceContext: true },
    );
    const manager = new AgentManager(store, runtime);
    const delegateTool = createSubagentDelegationTool(manager);
    tools.registerTool(delegateTool);
    runtime.registerToolDefinition(delegateTool.spec);

    const parentState = await runtime.startRun(
      "run-parent-fail-test",
      "Delegate then fail",
      TEST_MODEL_CONFIG,
    );
    // Parent run must transition cleanly to "interrupted" instead of throwing unhandled error
    assert.equal(parentState, "interrupted");

    // Verify ContextBuilder imageDataUrls char weight budgeting
    const bigDataUrl = "data:image/png;base64," + "A".repeat(6000);
    await store.appendEvent({
      eventId: "run-img-budget:created:1",
      runId: "run-img-budget",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "Inspect big image", sessionId: "s-budget" },
    });
    await store.appendEvent({
      eventId: "run-img-budget:prep:2",
      runId: "run-img-budget",
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await store.appendEvent({
      eventId: "run-img-budget:ckpt:3",
      runId: "run-img-budget",
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: {},
    });
    await store.appendEvent({
      eventId: "run-img-budget:intent:4",
      runId: "run-img-budget",
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      payload: {
        callId: "c1",
        toolName: "workspace.inspect_image",
        argumentsJson: '{"path":"big.png"}',
        effectClass: "pure_read",
      },
    });
    await store.appendEvent({
      eventId: "run-img-budget:tool:5",
      runId: "run-img-budget",
      type: "ToolDispatched",
      nextState: "executing_tool",
      payload: {
        callId: "c1",
        toolName: "workspace.inspect_image",
        argumentsJson: '{"path":"big.png"}',
        effectClass: "pure_read",
      },
    });
    await store.appendEvent({
      eventId: "run-img-budget:res:6",
      runId: "run-img-budget",
      type: "ToolResultCommitted",
      nextState: "inferring",
      payload: {
        toolCallId: "c1",
        toolName: "workspace.inspect_image",
        outputText: "Loaded big.png",
        imageDataUrls: [bigDataUrl],
      },
    });

    const journal = await store.getJournal();
    const built = ContextBuilder.buildMessages(journal, {
      sessionId: "s-budget",
      maxCharBudget: 1200,
    });
    // Vision attachment in active turn is preserved while earlier turns are trimmed under tight budget
    assert.equal(
      built.some((m) => (m.imageDataUrls?.length ?? 0) > 0),
      true,
    );
  });

  it("image.generate and image.transform commit durable OPFS image artifacts and attach vision dataUrls", async () => {
    const { store } = await createTestWorkspace("ws-img-gen");
    const tools = new BrowserToolRegistry(store);

    // 1. Generate a base sprite in OPFS
    const genRes = await tools.executeTool({
      runId: "r-img-1",
      operationId: "op-img-1",
      toolName: "image.generate",
      effectClass: "idempotent_write",
      args: {
        outputPath: "assets/horse-base.svg",
        prompt: "Pixel art galloping horse",
        style: "pixel-sprite",
      },
    });
    assert.ok(genRes.fileUpdates?.["assets/horse-base.svg"]);
    assert.ok(genRes.imageDataUrls?.[0]?.startsWith("data:image/svg+xml;utf8,"));
    await store.writeWorkspaceFile(
      "assets/horse-base.svg",
      genRes.fileUpdates["assets/horse-base.svg"]!,
    );

    // 2. Transform existing OPFS image using image.transform (Flux/LoRA style workflow)
    const txRes = await tools.executeTool({
      runId: "r-img-1",
      operationId: "op-img-2",
      toolName: "image.transform",
      effectClass: "idempotent_write",
      args: {
        sourcePath: "assets/horse-base.svg",
        outputPath: "assets/horse-cyberpunk.svg",
        prompt: "Neon cyberpunk armored horse",
        style: "flux-4b-lora",
      },
    });
    assert.ok(txRes.fileUpdates?.["assets/horse-cyberpunk.svg"]);
    assert.ok(txRes.imageDataUrls?.[0]?.startsWith("data:image/svg+xml;utf8,"));

    // 3. Verify OpenAICompatibleAdapter serializes SVG dataUrls as text markup (not image_url)
    const wireBodies: Array<Record<string, unknown>> = [];
    const svgAdapter = new OpenAICompatibleAdapter(async (_url, init) => {
      wireBodies.push(JSON.parse(init.body) as Record<string, unknown>);
      const enc = new TextEncoder();
      return {
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(c) {
            c.enqueue(enc.encode('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'));
            c.close();
          },
        }),
      };
    });
    for await (const _ of svgAdapter.streamChat({
      config: TEST_MODEL_CONFIG,
      messages: [
        {
          role: "user",
          content: "[Attached workspace image from image.generate]",
          imageDataUrls: genRes.imageDataUrls,
        },
      ],
    })) {
      // consume stream
    }
    const sentMsg = (wireBodies[0]?.messages as Array<{ content: Array<{ type: string; text?: string }> }>)[0]!;
    assert.equal(sentMsg.content.some((p) => p.type === "image_url"), false);
    assert.match(sentMsg.content[1]?.text ?? "", /\[Attached SVG Image Markup\]/);
  });

  it("Repeated-Error Loop Guard intercepts duplicate tool calls so the agent recovers instead of exhausting maxToolRounds", async () => {
    const { store } = await createTestWorkspace("ws-loop-guard");
    const tools = new BrowserToolRegistry(store);
    let turn = 0;

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        turn += 1;
        const lastMsg = req.messages[req.messages.length - 1];
        // Turn 1 & Turn 2: model stubbornly calls the exact same tool with identical arguments
        if (turn <= 2) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: `Attempt ${turn}`,
            toolCalls: [
              {
                id: `call_dup_${turn}`,
                name: "workspace.list_files",
                argumentsJson: "{}",
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        // Turn 3: model sees LOOP_GUARD_TRIGGERED in the tool result and completes cleanly
        assert.equal(lastMsg?.role, "tool");
        assert.match(lastMsg?.content ?? "", /LOOP_GUARD_TRIGGERED/);
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Adjusted strategy after loop guard feedback.",
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      tools,
      tools.getToolDefinitions(),
      { multiTurnWorkspaceContext: true },
    );

    const state = await runtime.startRun(
      "run-loop-guard-1",
      "Test loop guard",
      TEST_MODEL_CONFIG,
    );
    assert.equal(state, "completed");
    assert.equal(turn, 3);
  });
});

