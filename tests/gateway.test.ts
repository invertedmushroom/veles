import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve, sep } from "node:path";
import { createSandboxedToolRunner, EphemeralExecutionGateway } from "../services/gateway/server.ts";
import type { InferenceAdapter } from "../src/inference/openai-compatible.ts";
import {
  ContentAddressedBlobStore,
  InMemoryBlobBackend,
} from "../src/kernel/blobs.ts";
import { GatewayToolExecutionAdapter } from "../src/kernel/gateway-client.ts";
import {
  InMemoryJournalBackend,
  WorkspaceKernelStore,
} from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import { AgentKernelRuntime } from "../src/kernel/runtime.ts";
import { PROTOCOL_VERSION } from "../src/protocol/types.ts";

async function setupGatewayWorkspace(id = "ws-gateway") {
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease(id, "worker-gw");
  const journalBackend = new InMemoryJournalBackend();
  const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
  const store = new WorkspaceKernelStore(journalBackend, blobs, lease);
  await store.initWorkspace("Gateway Workspace");
  return { store, journalBackend, blobs, lease };
}

it("blocks host-loading Node flags in restricted mode and retains explicitly trusted execution", async () => {
  const root = await mkdtemp(join(tmpdir(), "veles-gateway-flags-"));
  try {
    const sandbox = join(root, "sandbox");
    await mkdir(sandbox);
    const outside = join(root, "outside-config.txt");
    await writeFile(outside, "VELES_REVIEW_MARKER=HOST_FILE_MARKER\n");
    const restricted = createSandboxedToolRunner();
    const scope = { allowedTools: ["sandbox.exec"], writablePaths: ["out"], allowNetwork: false };
    const limits = { timeoutMs: 3000, maxOutputBytes: 6000 };
    const signal = new AbortController().signal;
    const code = "process.stdout.write(process.env.VELES_REVIEW_MARKER || 'missing')";
    for (const option of [
      `--env-file=${outside}`, `--env-file-if-exists=${outside}`,
      `--openssl-config=${outside}`, `--require=${outside}`, `-r${outside}`,
      `--trace-event-file-pattern=${outside}`,
    ]) {
      await assert.rejects(
        restricted(sandbox, {
          toolName: "sandbox.exec", effectClass: "non_idempotent", host: "gateway",
          args: { argv: [process.execPath, option, "-e", code] },
        }, scope, limits, signal),
        /ERR_GATEWAY_RUNTIME_FLAG_DENIED/,
      );
    }
    const safe = await restricted(sandbox, {
      toolName: "sandbox.exec", effectClass: "non_idempotent", host: "gateway",
      args: { argv: [process.execPath, "--input-type", "module", "--no-warnings", "-e", "console.log(42)"] },
    }, scope, limits, signal);
    assert.equal(safe.exitCode, 0);
    assert.equal(safe.outputText.trim(), "42");
    const scriptArgument = await restricted(sandbox, {
      toolName: "sandbox.exec", effectClass: "non_idempotent", host: "gateway",
      args: { argv: [process.execPath, "-e", code, "app-argument", `--env-file=${outside}`] },
    }, scope, limits, signal);
    assert.equal(scriptArgument.exitCode, 0);
    assert.equal(scriptArgument.outputText, "missing");
    const trusted = await createSandboxedToolRunner({ allowUnrestrictedExec: true })(sandbox, {
      toolName: "sandbox.exec", effectClass: "non_idempotent", host: "gateway",
      args: { argv: [process.execPath, `--env-file=${outside}`, "-e", code] },
    }, scope, limits, signal);
    assert.equal(trusted.exitCode, 0);
    assert.equal(trusted.outputText, "HOST_FILE_MARKER");
  } finally {
    const target = resolve(root);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep));
    assert.ok(basename(target).startsWith("veles-gateway-flags-"));
    await rm(target, { recursive: true, force: true });
  }
});

function createInProcessGatewayFetch(gateway: EphemeralExecutionGateway, _token = "secret-gw-token") {
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
    const parsedUrl = new URL(url);
    if (init.method === "GET" && parsedUrl.pathname.startsWith("/v1/operations/")) {
      const opId = decodeURIComponent(parsedUrl.pathname.slice("/v1/operations/".length));
      const statusObj = gateway.getOperationStatus(opId);
      return {
        ok: true,
        status: 200,
        json: async () => statusObj,
      };
    }
    if (init.method === "POST" && parsedUrl.pathname === "/v1/tools/execute") {
      const reqBody: unknown = JSON.parse(init.body ?? "{}");
      const encoder = new TextEncoder();
      const iterator = gateway.executeToolStream(reqBody)[Symbol.asyncIterator]();
      const stream = new ReadableStream<Uint8Array>({
        async pull(controller) {
          try {
            const next = await iterator.next();
            if (next.done) {
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              controller.close();
              return;
            }
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(next.value)}\n\n`),
            );
          } catch (err) {
            controller.error(err);
          }
        },
      });
      return { ok: true, status: 200, body: stream };
    }
    return { ok: false, status: 404, body: null };
  };
}

describe("SYS-GATEWAY Ephemeral Gateway, Sandboxed Execution & Scenarios 2, 4, 8, 10", () => {
  it("streams 3-stage tool execution, cleans up temp sandboxes, and redacts leaked secrets (Scenario 10)", async () => {
    const gateway = new EphemeralExecutionGateway({ authToken: "tok-1" });
    const { store } = await setupGatewayWorkspace("ws-gw-exec");

    const client = new GatewayToolExecutionAdapter(
      {
        baseUrl: "https://gw.example.test",
        authToken: "tok-1",
        defaultScope: {
          allowedTools: ["sandbox.exec", "sandbox.WriteFile"],
          allowNetwork: false,
          writablePaths: ["out"],
        },
        defaultLimits: { timeoutMs: 5000, maxOutputBytes: 32_768 },
      },
      createInProcessGatewayFetch(gateway, "tok-1"),
      gateway.gatewayInstanceId,
    );

    // Execute a real node subprocess inside the ephemeral sandbox that prints a secret token
    const leakedKey = "sk-" + "a".repeat(28);
    const outcome = await client.executeTool({
      runId: "run-1",
      operationId: "op-exec-1",
      toolName: "sandbox.exec",
      effectClass: "pure_read",
      args: {
        argv: [
          process.execPath,
          "-e",
          `console.log("Build ok with token ${leakedKey}")`,
        ],
      },
    });

    assert.equal(outcome.gatewayInstanceId, gateway.gatewayInstanceId);
    assert.match(outcome.outputText, /Build ok with token <REDACTED>/);
    assert.equal(outcome.outputText.includes(leakedKey), false);

    // Scenario 10: Temp sandbox directory is wiped immediately after operation completes
    assert.equal(gateway.activeSandboxCount, 0);

    // Expiry reaper removes buffered operation metadata once TTL elapses
    assert.equal(gateway.retainedOperationCount, 1);
    gateway.reapExpiredOperations(Date.now() + 600_000);
    assert.equal(gateway.retainedOperationCount, 0);
    assert.equal((await store.getMetadata())?.headSeq, 0);
  });

  it("Scenario 8 (Gateway): enforces auth, capabilityScope.allowedTools, writablePaths, and timeouts", async () => {
    const gateway = new EphemeralExecutionGateway({ authToken: "valid-token" });

    // 1. Unauthorized token is rejected
    assert.equal(gateway.checkAuthorized("Bearer wrong-token"), false);

    // 2. Disallowed tool in capabilityScope is rejected by gateway
    await assert.rejects(
      async () => {
        for await (const _ of gateway.executeToolStream({
          protocolVersion: PROTOCOL_VERSION,
          operationId: "op-unauth-tool",
          runId: "run-1",
          operation: "tool.execute",
          capabilityScope: {
            allowedTools: ["sandbox.readOnly"],
            allowNetwork: false,
            writablePaths: ["src"],
          },
          limits: { timeoutMs: 2000, maxOutputBytes: 4096 },
          payload: {
            toolName: "sandbox.exec",
            effectClass: "non_idempotent",
            host: "gateway",
            args: { argv: [process.execPath, "-e", "process.exit(0)"] },
          },
        })) {
          // consume
        }
      },
      /ERR_GATEWAY_TOOL_UNAUTHORIZED/,
    );

    // 3. Writing outside writablePaths is rejected and temp sandbox is still cleaned up
    await assert.rejects(
      async () => {
        for await (const _ of gateway.executeToolStream({
          protocolVersion: PROTOCOL_VERSION,
          operationId: "op-unauth-path",
          runId: "run-1",
          operation: "tool.execute",
          capabilityScope: {
            allowedTools: ["sandbox.WriteFile"],
            allowNetwork: false,
            writablePaths: ["allowed_dir"],
          },
          limits: { timeoutMs: 2000, maxOutputBytes: 4096 },
          payload: {
            toolName: "sandbox.WriteFile",
            effectClass: "idempotent_write",
            host: "gateway",
            args: { path: "forbidden_dir/file.txt", content: "blocked" },
          },
        })) {
          // consume
        }
      },
      /ERR_GATEWAY_WRITABLE_SCOPE_DENIED/,
    );
    assert.equal(gateway.activeSandboxCount, 0);

    // 4. Runtime-constructed destination path outside sandbox in node -e is blocked by Node permission model
    await assert.rejects(
      async () => {
        for await (const _ of gateway.executeToolStream({
          protocolVersion: PROTOCOL_VERSION,
          operationId: "op-runtime-escape",
          runId: "run-1",
          operation: "tool.execute",
          capabilityScope: {
            allowedTools: ["sandbox.exec"],
            allowNetwork: false,
            writablePaths: ["out"],
          },
          limits: { timeoutMs: 5000, maxOutputBytes: 4096 },
          payload: {
            toolName: "sandbox.exec",
            effectClass: "non_idempotent",
            host: "gateway",
            args: {
              argv: [
                process.execPath,
                "-e",
                "const p = ['..', 'escaped-' + Date.now() + '.txt'].join('/'); require('node:fs').writeFileSync(p, 'escaped');",
              ],
            },
          },
        })) {
          // consume
        }
      },
      /ERR_GATEWAY_SANDBOX_ESCAPE/,
    );
    assert.equal(gateway.activeSandboxCount, 0);

    // 2d. Verify allowNetwork: false blocks network requests while allowNetwork: true permits them
    const { createServer } = await import("node:http");
    const testServer = createServer((_req, res) => {
      res.statusCode = 200;
      res.end("GATEWAY_NET_OK");
    });
    await new Promise<void>((resolve) => testServer.listen(0, "127.0.0.1", () => resolve()));
    const addr = testServer.address() as { port: number };
    const netUrl = `http://127.0.0.1:${addr.port}/ping`;
    try {
      await assert.rejects(
        async () => {
          for await (const _ of gateway.executeToolStream({
            protocolVersion: PROTOCOL_VERSION,
            operationId: "op-net-denied",
            runId: "run-net-denied",
            operation: "tool.execute",
            capabilityScope: {
              allowedTools: ["sandbox.exec"],
              allowNetwork: false,
              writablePaths: [],
            },
            limits: { timeoutMs: 5000, maxOutputBytes: 4096 },
            payload: {
              toolName: "sandbox.exec",
              effectClass: "pure_read",
              host: "gateway",
              args: {
                argv: [
                  process.execPath,
                  "-e",
                  `fetch(${JSON.stringify(netUrl)}).then(r => r.text()).then(t => console.log(t));`,
                ],
              },
            },
          })) {
            // consume
          }
        },
        /ERR_GATEWAY_SANDBOX_ESCAPE/,
      );

      const allowedNetEvents = [];
      for await (const ev of gateway.executeToolStream({
        protocolVersion: PROTOCOL_VERSION,
        operationId: "op-net-allowed",
        runId: "run-net-allowed",
        operation: "tool.execute",
        capabilityScope: {
          allowedTools: ["sandbox.exec"],
          allowNetwork: true,
          writablePaths: [],
        },
        limits: { timeoutMs: 5000, maxOutputBytes: 4096 },
        payload: {
          toolName: "sandbox.exec",
          effectClass: "pure_read",
          host: "gateway",
          args: {
            argv: [
              process.execPath,
              "-e",
              `fetch(${JSON.stringify(netUrl)}).then(r => r.text()).then(t => console.log(t));`,
            ],
          },
        },
      })) {
        allowedNetEvents.push(ev);
      }
      const completedNet = allowedNetEvents.find(
        (e) => e.event.stage === "COMPLETED",
      );
      assert.equal(completedNet?.outcome?.exitCode, 0);
      assert.match(String(completedNet?.outcome?.outputText), /GATEWAY_NET_OK/);
    } finally {
      await new Promise<void>((resolve) => testServer.close(() => resolve()));
    }
  });

  it("Scenario 4 (Gateway Reconnect vs. Gateway Restart): reconciles buffered result if alive, or interrupts if restarted", async () => {
    const gateway1 = new EphemeralExecutionGateway({ authToken: "tok-reconnect" });
    const { store } = await setupGatewayWorkspace("ws-reconnect");

    // 1. Run tool on gateway1 so result is buffered under op-buffered-99
    for await (const _ of gateway1.executeToolStream({
      protocolVersion: PROTOCOL_VERSION,
      operationId: "op-buffered-99",
      runId: "run-reconnect",
      operation: "tool.execute",
      capabilityScope: {
        allowedTools: ["sandbox.WriteFile"],
        allowNetwork: false,
        writablePaths: ["src"],
      },
      limits: { timeoutMs: 5000, maxOutputBytes: 16_384 },
      payload: {
        toolName: "sandbox.WriteFile",
        effectClass: "non_idempotent",
        host: "gateway",
        args: { path: "src/generated.ts", content: "export const ok = true;" },
      },
    })) {
      // Gateway finished Stage 3, but simulate browser crash before Stage 4 commit!
    }

    // Seed browser journal up to ToolDispatched (Stage 4 not yet committed)
    await store.appendEvent({
      eventId: "ev-1",
      runId: "run-reconnect",
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: "Generate file" },
    });
    await store.appendEvent({
      eventId: "ev-2",
      runId: "run-reconnect",
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await store.appendEvent({
      eventId: "ev-3",
      runId: "run-reconnect",
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: {},
    });
    await store.appendEvent({
      eventId: "ev-4",
      runId: "run-reconnect",
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: "op-buffered-99",
      payload: {
        toolCallId: "call_99",
        toolName: "sandbox.WriteFile",
        effectClass: "non_idempotent",
      },
    });
    await store.appendEvent({
      eventId: "ev-5",
      runId: "run-reconnect",
      type: "ToolDispatched",
      nextState: "executing_tool",
      relatedOperationId: "op-buffered-99",
      payload: {
        toolCallId: "call_99",
        toolName: "sandbox.WriteFile",
        effectClass: "non_idempotent",
        gatewayInstanceId: gateway1.gatewayInstanceId,
      },
    });

    const client1 = new GatewayToolExecutionAdapter(
      {
        baseUrl: "https://gw.example.test",
        authToken: "tok-reconnect",
        defaultScope: {
          allowedTools: ["sandbox.WriteFile"],
          allowNetwork: false,
          writablePaths: ["src"],
        },
        defaultLimits: { timeoutMs: 5000, maxOutputBytes: 16_384 },
      },
      createInProcessGatewayFetch(gateway1, "tok-reconnect"),
      gateway1.gatewayInstanceId,
    );

    const noopInference: InferenceAdapter = {
      async *streamChat() {},
    };
    const runtime = new AgentKernelRuntime(store, noopInference, client1);

    // Case A: Reconnect while gateway1 is still alive -> reconciles buffered outcome into Stage 4!
    const reconciled = await runtime.recoverUnfinishedRun(gateway1.gatewayInstanceId);
    assert.equal(reconciled?.nextState, "inferring");
    assert.equal(
      await store.readFileText("src/generated.ts"),
      "export const ok = true;",
    );

    // Case B: Simulate another non-idempotent tool dispatched before gateway restart (gateway2 !== gateway1)
    await store.appendEvent({
      eventId: "ev-6",
      runId: "run-reconnect",
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: "op-lost-100",
      payload: {
        toolCallId: "call_100",
        toolName: "sandbox.WriteFile",
        effectClass: "non_idempotent",
      },
    });
    await store.appendEvent({
      eventId: "ev-7",
      runId: "run-reconnect",
      type: "ToolDispatched",
      nextState: "executing_tool",
      relatedOperationId: "op-lost-100",
      payload: {
        toolCallId: "call_100",
        toolName: "sandbox.WriteFile",
        effectClass: "non_idempotent",
        gatewayInstanceId: gateway1.gatewayInstanceId,
      },
    });
    const gateway2 = new EphemeralExecutionGateway({ authToken: "tok-reconnect" });
    const afterRestart = await runtime.recoverUnfinishedRun(gateway2.gatewayInstanceId);
    assert.equal(afterRestart?.nextState, "interrupted");
    assert.equal(afterRestart?.interruptReason, "uncertain_tool_outcome");
    assert.equal(afterRestart?.canAutoRetry, false);
  });
});
