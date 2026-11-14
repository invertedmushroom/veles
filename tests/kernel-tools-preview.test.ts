import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type {
  InferenceAdapter,
  ProviderModelConfig,
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
import { AgentManager } from "../src/kernel/agent-session.ts";
import { AgentKernelRuntime } from "../src/kernel/runtime.ts";
import {
  BrowserToolRegistry,
  createAgentDelegateTool,
} from "../src/kernel/tools.ts";
import {
  PreviewSandboxHost,
  assertIsolatedSandboxAttribute,
  compileMultiFileSrcdoc,
  type MessageEventLike,
  type PreviewRuntimeDiagnostic,
  type SandboxEnvironmentLike,
  type SandboxIframeLike,
} from "../src/ui/preview-sandbox.ts";

const DEFAULT_CONFIG: ProviderModelConfig = {
  providerId: "openai",
  baseUrl: "https://api.example.test/v1",
  model: "gpt-4o",
};

async function setupToolsStore(workspaceId = "ws-tools") {
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease(workspaceId, "worker-tools");
  const journalBackend = new InMemoryJournalBackend();
  const blobBackend = new InMemoryBlobBackend();
  const blobs = new ContentAddressedBlobStore(blobBackend);
  const store = new WorkspaceKernelStore(journalBackend, blobs, lease);
  await store.initWorkspace("Tools Workspace");
  return { store, journalBackend, blobBackend, lease };
}

describe("SYS-KERNEL & SYS-UI Browser Tools, Approval Flow, & Preview Sandbox Isolation", () => {
  it("executes browser workspace tools (write, list, read) and blocks path traversal", async () => {
    const { store } = await setupToolsStore("ws-tools-basic");
    const registry = new BrowserToolRegistry(store);
    let step = 0;

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        step += 1;
        if (step === 1) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Creating src/main.ts",
            toolCalls: [
              {
                id: "call_write",
                name: "workspace.write_file",
                argumentsJson: JSON.stringify({
                  path: "src/main.ts",
                  content: "export const v = 42;",
                }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        if (step === 2) {
          // Verify ContextBuilder reconstructed the assistant tool_calls + tool result
          assert.equal(
            req.messages.some(
              (m) => m.role === "assistant" && m.toolCalls?.[0]?.name === "workspace.write_file",
            ),
            true,
          );
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Reading src/main.ts",
            toolCalls: [
              {
                id: "call_read",
                name: "workspace.read_file",
                argumentsJson: JSON.stringify({ path: "src/main.ts" }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        assert.equal(
          req.messages.some(
            (m) => m.role === "tool" && m.content === "export const v = 42;",
          ),
          true,
        );
        yield {
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Verified file content!",
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      registry,
      registry.getToolDefinitions(),
    );

    const state = await runtime.startRun("run-tools-1", "Write and read file", DEFAULT_CONFIG);
    assert.equal(state, "completed");
    assert.deepEqual(await store.listFiles(), ["src/main.ts"]);

    // Verify path traversal is deterministically rejected without corrupting workspace
    await assert.rejects(
      async () => {
        await registry.executeTool({
          runId: "run-tools-1",
          operationId: "op-bad",
          toolName: "workspace.write_file",
          effectClass: "idempotent_write",
          args: { path: "../escape.txt", content: "hack" },
        });
      },
      /ERR_TOOL_UNSAFE_PATH/,
    );
    assert.deepEqual(await store.listFiles(), ["src/main.ts"]);
  });

  it("pauses on requiresApproval tools (workspace.delete_file) and resumes on approve or deny", async () => {
    const { store } = await setupToolsStore("ws-approval");
    const registry = new BrowserToolRegistry(store);

    // Seed an existing file to delete
    await store.appendEvent({
      eventId: "seed-1",
      runId: "run-seed",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "seed" },
      fileUpdates: { "obsolete.txt": "remove me", "keep.txt": "stay" },
    });

    let turn = 0;
    const inference: InferenceAdapter = {
      async *streamChat(req) {
        turn += 1;
        if (turn === 1 || turn === 2) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Requesting deletion of obsolete.txt",
            toolCalls: [
              {
                id: `call_del_${turn}`,
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
          providerId: req.config.providerId,
          model: req.config.model,
          deltaText: "Deletion flow finished.",
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      registry,
      registry.getToolDefinitions(),
    );

    // Turn 1: Agent requests deletion -> pauses in awaiting_approval; file still exists
    const pausedState = await runtime.startRun(
      "run-del",
      "Delete obsolete.txt",
      DEFAULT_CONFIG,
    );
    assert.equal(pausedState, "awaiting_approval");
    assert.deepEqual(await store.listFiles(), ["keep.txt", "obsolete.txt"]);

    // User denies first request -> transitions to inferring (Turn 2), which asks again
    const afterDenyState = await runtime.handleWorkerCommand({
      type: "RESOLVE_APPROVAL",
      runId: "run-del",
      approved: false,
    });
    assert.equal(afterDenyState, "awaiting_approval");
    assert.deepEqual(await store.listFiles(), ["keep.txt", "obsolete.txt"]);

    // User approves second request -> executes deletion and completes in Turn 3
    const finalState = await runtime.handleWorkerCommand({
      type: "RESOLVE_APPROVAL",
      runId: "run-del",
      approved: true,
    });
    assert.equal(finalState, "completed");
    assert.deepEqual(await store.listFiles(), ["keep.txt"]);
  });

  it("Scenario 8 (Local): PreviewSandboxHost enforces opaque origin isolation and cleans up on dispose()", () => {
    // Reject any sandbox string that includes allow-same-origin or omits allow-scripts
    assert.throws(
      () => assertIsolatedSandboxAttribute("allow-scripts allow-same-origin"),
      /ERR_SANDBOX_ORIGIN_VIOLATION/,
    );
    assert.throws(
      () => assertIsolatedSandboxAttribute(""),
      /ERR_SANDBOX_MISSING_ALLOW_SCRIPTS/,
    );

    let messageListener = null as ((ev: MessageEventLike) => void) | null;
    const postedMessages: unknown[] = [];
    let removedCount = 0;

    const fakeContentWindow = {
      postMessage(msg: unknown) {
        postedMessages.push(msg);
      },
    };

    const fakeEnv: SandboxEnvironmentLike = {
      createIframe(): SandboxIframeLike {
        return {
          sandbox: "",
          srcdoc: "",
          contentWindow: fakeContentWindow,
          remove() {
            removedCount += 1;
          },
        };
      },
      addMessageListener(handler) {
        messageListener = handler;
        return () => {
          if (messageListener === handler) {
            messageListener = null;
          }
        };
      },
    };

    const diagnostics: PreviewRuntimeDiagnostic[] = [];
    const host = new PreviewSandboxHost(fakeEnv, (d) => diagnostics.push(d));

    const gen1 = host.mountPreview({
      entryHtml: "<!doctype html><script>console.log('hi')</script>",
      files: { "index.js": "console.log('hi')" },
    });
    assert.equal(gen1, 1);
    assert.equal(postedMessages.length, 1);

    // Valid diagnostic from opaque ("null") origin and matching generation is accepted
    messageListener?.({
      source: fakeContentWindow,
      origin: "null",
      data: {
        channel: "veles:preview",
        generation: 1,
        type: "preview_ready",
        message: "Mounted",
      },
    });
    assert.equal(diagnostics.length, 1);
    assert.equal(diagnostics[0]!.message, "Mounted");

    // Spoofed non-null origin or stale generation is ignored
    messageListener?.({
      source: fakeContentWindow,
      origin: "https://evil.example.com",
      data: {
        channel: "veles:preview",
        generation: 1,
        type: "preview_error",
        message: "Spoofed",
      },
    });
    assert.equal(diagnostics.length, 1);

    // Disposing the host removes the iframe, detaches listeners, and is idempotent
    host.dispose();
    host.dispose();
    assert.equal(removedCount, 1);
    assert.equal(messageListener, null);
    assert.throws(
      () => host.mountPreview({ entryHtml: "", files: {} }),
      /ERR_PREVIEW_DISPOSED/,
    );

    // Multi-file preview bundler (compileMultiFileSrcdoc) resolves relative CSS, CSS url(...), JS, and SVG assets
    const bundled = compileMultiFileSrcdoc(
      {
        entryPath: "preview/index.html",
        entryHtml:
          '<!doctype html><html><head><link rel="stylesheet" href="style.css"></head><body><img src="../assets/mario.svg"><script src="script.js"></script></body></html>',
        files: {
          "preview/style.css":
            "body { background: #101c25; background-image: url('../assets/mario.svg'); }",
          "preview/script.js": "window.marioJump = 42;",
          "assets/mario.svg": '<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>',
        },
      },
      2,
    );
    assert.ok(bundled.includes('data-veles-path="preview/style.css"'));
    assert.ok(bundled.includes('url("data:image/svg+xml;utf8,'));
    assert.ok(bundled.includes('data-veles-path="preview/script.js"'));
    assert.ok(bundled.includes("window.marioJump = 42;"));
    assert.ok(bundled.includes("data:image/svg+xml;utf8,"));
    assert.ok(bundled.includes('data-veles-error-bridge="true"'));
  });

  it("delegates synchronous parent-to-child subtask via agent.delegate without deadlock and inspects binary images via workspace.inspect_image", async () => {
    const { store } = await setupToolsStore("ws-delegate");
    const pngBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    await store.writeWorkspaceFile("assets/hero.png", pngBytes);

    const registry = new BrowserToolRegistry(store);
    let parentRound = 0;
    let childRound = 0;

    const inference: InferenceAdapter = {
      async *streamChat(req) {
        if (req.config.model === "child-repair-specialist") {
          childRound += 1;
          if (childRound === 1) {
            yield {
              providerId: req.config.providerId,
              model: req.config.model,
              deltaText: "Repairing preview/script.js...",
              toolCalls: [
                {
                  id: "call_child_write",
                  name: "workspace.write_file",
                  argumentsJson: JSON.stringify({
                    path: "preview/script.js",
                    content: "window.repaired = true;",
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
            deltaText: "Child repair completed: fixed preview/script.js.",
            toolCalls: [],
            usage: null,
            finishReason: "stop",
          };
          return;
        }

        parentRound += 1;
        if (parentRound === 1) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Inspecting hero.png before delegating repair...",
            toolCalls: [
              {
                id: "call_inspect_img",
                name: "workspace.inspect_image",
                argumentsJson: JSON.stringify({ path: "assets/hero.png" }),
              },
            ],
            usage: null,
            finishReason: "tool_calls",
          };
          return;
        }
        if (parentRound === 2) {
          yield {
            providerId: req.config.providerId,
            model: req.config.model,
            deltaText: "Delegating script repair to child session...",
            toolCalls: [
              {
                id: "call_delegate_repair",
                name: "agent.delegate",
                argumentsJson: JSON.stringify({
                  task: "Fix ReferenceError in preview/script.js",
                  sessionId: "session-main:repair-1",
                  parentSessionId: "session-main",
                  model: "child-repair-specialist",
                  allowedTools: ["workspace.write_file"],
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
          deltaText: "Parent verified child repair and image inspection.",
          toolCalls: [],
          usage: null,
          finishReason: "stop",
        };
      },
    };

    const runtime = new AgentKernelRuntime(
      store,
      inference,
      registry,
      registry.getToolDefinitions(),
    );
    const manager = new AgentManager(store, runtime);
    const delegateTool = createAgentDelegateTool(manager);
    registry.registerTool(delegateTool);
    runtime.registerToolDefinition(delegateTool.spec);

    await manager.createSession({
      id: "session-main",
      workspaceId: "ws-delegate",
      model: DEFAULT_CONFIG,
      allowedTools: [
        "workspace.inspect_image",
        "agent.delegate",
        "workspace.read_file",
      ],
      delegatableTools: ["workspace.write_file"],
      status: "idle",
    });

    const parentOutput = await manager.run(
      "session-main",
      "Inspect assets/hero.png and delegate preview/script.js repair",
    );
    assert.equal(
      parentOutput,
      "Parent verified child repair and image inspection.",
    );
    assert.equal(
      await store.readFileText("preview/script.js"),
      "window.repaired = true;",
    );
    assert.equal(manager.getSession("session-main:repair-1")?.status, "completed");
    assert.equal(
      manager.getSession("session-main:repair-1")?.parentSessionId,
      "session-main",
    );
  });
});