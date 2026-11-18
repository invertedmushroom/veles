import assert from "node:assert/strict";
import { it } from "node:test";
import { ContentAddressedBlobStore, InMemoryBlobBackend } from "../src/kernel/blobs.ts";
import { InMemoryJournalBackend, WorkspaceKernelStore } from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import { AgentKernelRuntime, ContextBuilder } from "../src/kernel/runtime.ts";
import { BrowserToolRegistry, createSubagentDelegationTool } from "../src/kernel/tools.ts";
import { AgentManager } from "../src/kernel/agent-session.ts";
import { KernelWorkerHost } from "../src/kernel/worker-host.ts";
import { VelesAppClient } from "../src/ui/app-client.ts";
import {
  formatToolTargetHint,
  projectChatRuns,
  projectRunTelemetry,
} from "../src/ui/chat-timeline.ts";
import type { InferenceAdapter, ProviderModelConfig } from "../src/inference/openai-compatible.ts";

const config: ProviderModelConfig = {
  providerId: "test",
  baseUrl: "https://example.test/v1",
  model: "fixture",
};

function deferred<T = void>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function fixture() {
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease("ws-stop", "test");
  const store = new WorkspaceKernelStore(
    new InMemoryJournalBackend(),
    new ContentAddressedBlobStore(new InMemoryBlobBackend()),
    lease,
    locks,
  );
  await store.initWorkspace("Stop fixture");
  return store;
}

async function bounded<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("Stop did not settle the run")), 150);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

it("projects committed turns by session, associates child activity, and honors deletion", async () => {
  const store = await fixture();
  const append = async (runId: string, payload: Record<string, unknown>) => store.appendEvent({
    eventId: runId + "-created", runId, type: "RunCreated", nextState: "created", payload,
  });
  await append("main-first", { sessionId: "session-main", prompt: "First" });
  await append("child-first", { sessionId: "child", parentSessionId: "session-main", prompt: "Child" });
  await append("other", { sessionId: "another-chat", prompt: "Private" });
  await append("main-second", { sessionId: "session-main", prompt: "Second" });
  await append("child-second", { sessionId: "child", parentSessionId: "session-main", prompt: "New child" });
  const journal = await store.getJournal();
  const runs = projectChatRuns(journal, "session-main");
  assert.deepEqual(runs.map((run) => run.prompt), ["First", "Second"]);
  assert.deepEqual(runs[0]!.records.map((record) => record.runId), ["main-first", "child-first"]);
  assert.deepEqual(runs[1]!.records.map((record) => record.runId), ["main-second", "child-second"]);
  const tombstone = { ...journal.at(-1)!, type: "CheckpointCommitted" as const, payload: { deletedRunId: "main-first" } };
  assert.deepEqual(projectChatRuns([...journal, tombstone], "session-main").map((run) => run.runId), ["main-second"]);

  const base = journal.at(-1)!;
  const enrichedRuns = projectChatRuns(
    [
      ...journal,
      {
        ...base,
        seq: base.seq + 1,
        eventId: "main-second-routing",
        runId: "main-second",
        type: "CheckpointCommitted",
        committedState: "inferring",
        payload: {
          harnessRouting: {
            source: "tier1_classifier",
            configuredMode: "openai-decisions",
            specialistIds: ["data-script-compute", "diagram-design"],
          },
          contextEconomics: {
            systemContractChars: 420,
            specialistCapsuleChars: 640,
            workspaceTreeChars: 280,
            conversationChars: 180,
            toolIOChars: 520,
            imageWeightChars: 0,
            totalChars: 2040,
            estimatedTokens: 510,
            cacheablePrefixTokens: 265,
            foldedSavedChars: 1200,
            hotRunCount: 2,
            coldFoldedRunCount: 1,
          },
        },
      },
      {
        ...base,
        seq: base.seq + 2,
        eventId: "main-second-usage",
        runId: "main-second",
        type: "CheckpointCommitted",
        committedState: "inferring",
        payload: {
          inferenceUsage: { promptTokens: 120, completionTokens: 45, totalTokens: 165 },
        },
      },
      {
        ...base,
        seq: base.seq + 3,
        eventId: "child-second-done",
        runId: "child-second",
        type: "RunCompleted",
        committedState: "completed",
        payload: {
          assistantText: "Subagent computed stats",
          usage: { promptTokens: 50, completionTokens: 20, totalTokens: 70 },
        },
      },
    ],
    "session-main",
  );
  const telemetry = projectRunTelemetry(enrichedRuns[1]!);
  assert.equal(telemetry.isLive, true);
  assert.equal(telemetry.latestRouting?.routerLabel, "⚡ Luna System-1");
  assert.deepEqual(telemetry.latestRouting?.specialistIds, [
    "data-script-compute",
    "diagram-design",
  ]);
  assert.equal(telemetry.totalTokens, 165);
  assert.equal(telemetry.contextEconomics?.estimatedTokens, 510);
  assert.equal(telemetry.contextEconomics?.cacheablePrefixTokens, 265);
  assert.equal(telemetry.contextEconomics?.foldedSavedChars, 1200);
  assert.equal(telemetry.subagentRuns.length, 1);
  assert.equal(telemetry.subagentRuns[0]?.stateLabel, "Completed");
  assert.equal(telemetry.subagentRuns[0]?.totalTokens, 70);
  assert.equal(formatToolTargetHint({ path: "assets/OldFaithful.csv" }), "assets/OldFaithful.csv");
});

it("publishes journal events only after durable success and detaches subscribers", async () => {
  const backend = new InMemoryJournalBackend();
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease("ws-commit", "test");
  const store = new WorkspaceKernelStore(backend, new ContentAddressedBlobStore(new InMemoryBlobBackend()), lease);
  await store.initWorkspace("Commits");
  const seen: string[] = [];
  const detach = store.onCommit((record) => seen.push(record.eventId));
  backend.simulateQuotaFailureOnNextCommit();
  const input = { eventId: "created", runId: "commit", type: "RunCreated", nextState: "created", payload: {} } as const;
  await assert.rejects(store.appendEvent(input), /QUOTA/);
  assert.deepEqual(seen, []);
  await store.appendEvent(input);
  assert.deepEqual(seen, ["created"]);
  detach();
  await store.appendEvent({ ...input, eventId: "preparing", type: "RunPreparing", nextState: "preparing" });
  assert.deepEqual(seen, ["created"]);
});

it("publishes an optimistic message then live commits over MessagePort before completion", async () => {
  const store = await fixture();
  const entered = deferred();
  const release = deferred();
  const inference: InferenceAdapter = { async *streamChat() {
    entered.resolve(); await release.promise;
    yield { providerId: "test", model: "fixture", deltaText: "Late", usage: null, finishReason: "stop", toolCalls: [] };
  } };
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Client fixture");
    const committed = deferred();
    client.onStateChange((view) => {
      if (view.snapshot?.journal.some((record) => record.type === "RunCreated")) committed.resolve();
    });
    const run = client.startRun("port-stop", "My immediate message", config);
    assert.equal(client.getStateView().pendingUserTurn?.prompt, "My immediate message");
    await bounded(committed.promise);
    assert.equal(client.getStateView().pendingUserTurn, null);
    assert.equal(projectChatRuns(client.getStateView().snapshot!.journal, "session-main")[0]?.prompt, "My immediate message");
    await entered.promise;
    await assert.rejects(client.startRun("duplicate", "No duplicate", config), /ALREADY_ACTIVE/);
    await client.cancelRun("port-stop");
    assert.equal((await bounded(run)).runState, "cancelled");
    release.resolve();
    await new Promise((done) => setTimeout(done, 10));
    assert.equal(client.getStateView().streamingDraftText, "");
    assert.equal(client.getStateView().inFlightRunId, null);
  } finally {
    release.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close();
  }
});

it("keeps a failed submission separate from durable chat history", async () => {
  const store = await fixture();
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() { throw new Error("unused"); } }, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Failed submission");
    client.dispose();
    await assert.rejects(client.startRun("failed", "Keep this text", config), /DISPOSED/);
    assert.equal(client.getStateView().pendingUserTurn?.status, "failed");
    assert.equal(client.getStateView().pendingUserTurn?.prompt, "Keep this text");
    assert.equal(client.getStateView().snapshot?.journal.length, 0);
    assert.equal(client.getStateView().snapshot?.runState, null);
    assert.equal(client.getStateView().inFlightRunId, null);
  } finally { host.dispose(); channel.port1.close(); channel.port2.close(); }
});

for (const effectClass of ["idempotent_write", "non_idempotent"] as const) {
  it(`stops a delegated ${effectClass} child and surfaces its uncertainty in the parent`, async () => {
    const store = await fixture();
    const entered = deferred();
    const release = deferred();
    const registry = new BrowserToolRegistry(store, [{
      spec: { name: "test.child_wait", description: "Delayed child tool", effectClass },
      async handler() { entered.resolve(); await release.promise; return { outputText: "late", fileUpdates: { "child-late.txt": "bad" } }; },
    }]);
    let calls = 0;
    const inference: InferenceAdapter = { async *streamChat() {
      calls += 1;
      yield { providerId: "test", model: "fixture", deltaText: "", usage: null, finishReason: "tool_calls",
        toolCalls: calls === 1
          ? [{ id: "delegate", name: "agent.delegate", argumentsJson: JSON.stringify({ task: "Wait in child", sessionId: "stop-child", allowedTools: ["test.child_wait"] }) }]
          : [{ id: "child-wait", name: "test.child_wait", argumentsJson: "{}" }],
      };
    } };
    const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions(), { multiTurnWorkspaceContext: true });
    const manager = new AgentManager(store, runtime);
    const delegate = createSubagentDelegationTool(manager);
    registry.registerTool(delegate); runtime.registerToolDefinition(delegate.spec);
    await manager.createSession({ id: "session-main", workspaceId: "ws-stop", model: config, allowedTools: ["agent.delegate", "test.child_wait"], status: "idle" });
    const channel = new MessageChannel();
    const host = new KernelWorkerHost(store, runtime, channel.port1, manager);
    const client = new VelesAppClient(channel.port2);
    await client.initWorkspace("Delegated stop");
    const run = client.startRun("parent-stop", "Delegate", config);
    void run.catch(() => {});
    await entered.promise;
    try {
      const during = await client.refreshSnapshot();
      assert.equal(during.activeRunId, "parent-stop", "Stop must target the selected parent, not the last child writer");
      await client.cancelRun("parent-stop");
      const stopped = await bounded(run);
      assert.equal(stopped.activeRunId, "parent-stop");
      assert.equal(stopped.runState, effectClass === "non_idempotent" ? "interrupted" : "cancelled");
      await new Promise((done) => setTimeout(done, 10));
      const journal = await store.getJournal();
      const childId = journal.find((record) => record.type === "RunCreated" && record.payload.parentSessionId === "session-main")!.runId;
      assert.equal(journal.filter((record) => record.runId === childId).at(-1)?.committedState, effectClass === "non_idempotent" ? "interrupted" : "cancelled");
      if (effectClass === "non_idempotent") {
        const parentTail = journal.filter((record) => record.runId === "parent-stop").at(-1)!;
        assert.equal(parentTail.payload.interruptReason, "uncertain_tool_outcome");
        assert.ok((parentTail.payload.uncertainRunIds as string[]).includes(childId));
        assert.equal((await client.resolveUncertainTool("parent-stop", "cancel_run")).runState, "cancelled");
      }
      release.resolve();
      await new Promise((done) => setTimeout(done, 10));
      assert.deepEqual(await store.listFiles(), []);
      assert.ok(!(await store.getJournal()).some((record) => record.type === "ToolResultCommitted"));
    } finally { release.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
  });
}


it("cleans up an approval request in its originating workspace after switching away", async () => {
  const store = await fixture();
  const entered = deferred();
  const release = deferred();
  const registry = new BrowserToolRegistry(store, [{
    spec: { name: "test.approval", description: "Held approval", effectClass: "idempotent_write", requiresApproval: true },
    async handler() { entered.resolve(); await release.promise; return { outputText: "done" }; },
  }]);
  const inference: InferenceAdapter = { async *streamChat(req) {
    const done = req.messages.some((message) => message.role === "tool");
    yield { providerId: "test", model: "fixture", deltaText: "", usage: null, finishReason: done ? "stop" : "tool_calls",
      toolCalls: done ? [] : [{ id: "approval", name: "test.approval", argumentsJson: "{}" }] };
  } };
  const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Approval origin");
    assert.equal((await client.startRun("origin-approval", "Approve", config)).runState, "awaiting_approval");
    const approval = client.resolveApproval("origin-approval", true);
    await entered.promise;
    await client.switchWorkspace("approval-other", "Other");
    release.resolve();
    await approval;
    await client.switchWorkspace("ws-stop");
    assert.equal(client.getStateView().snapshot?.runState, "completed");
    assert.equal(client.getStateView().inFlightRunId, null);
    assert.equal((await client.startRun("origin-next", "Next", config)).activeRunId, "origin-next");
  } finally { release.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});

it("cleans up Stop in its originating workspace after switching away", async (t) => {
  const store = await fixture();
  const entered = deferred();
  const stopEntered = deferred();
  const releaseStop = deferred();
  const originalAppend = store.appendEvent.bind(store);
  t.mock.method(store, "appendEvent", async (input: Parameters<typeof store.appendEvent>[0]) => {
    if (input.type === "RunCancelled") { stopEntered.resolve(); await releaseStop.promise; }
    return originalAppend(input);
  });
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() {
    entered.resolve(); await new Promise<void>(() => {});
  } }, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Stop origin");
    const run = client.startRun("origin-stop", "Wait", config);
    await entered.promise;
    const stop = client.cancelRun("origin-stop");
    await stopEntered.promise;
    await client.switchWorkspace("stop-other", "Other");
    releaseStop.resolve();
    await Promise.all([run, stop]);
    await client.switchWorkspace("ws-stop");
    assert.equal(client.getStateView().stoppingRunId, null);
  } finally { releaseStop.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});

for (const pendingAction of ["archive import", "session switch"] as const) {
it(`pins ${pendingAction === "archive import" ? "an archive import" : "a session switch"} to its source workspace while cancellation is pending`, async (t) => {
  const store = await fixture();
  const running = deferred();
  const cancelEntered = deferred();
  const releaseCancel = deferred();
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() {
    running.resolve(); await new Promise<void>(() => {});
  } }, registry, registry.getToolDefinitions());
  const originalCancel = runtime.cancelActiveRun.bind(runtime);
  t.mock.method(runtime, "cancelActiveRun", async (runId: string) => {
    cancelEntered.resolve(); await releaseCancel.promise;
    return originalCancel(runId);
  });
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Import origin");
    await client.uploadWorkspaceFile("notes/original.txt", "original");
    const archive = await client.exportWorkspace();
    await client.uploadWorkspaceFile("notes/later.txt", "later");
    const run = client.startRun("import-old-run", "Wait", config);
    await running.promise;
    const importing = pendingAction === "archive import"
      ? client.importWorkspace(archive)
      : client.switchChatSession("origin-session");
    await cancelEntered.promise;
    await client.switchWorkspace("import-other", "Other");
    await client.uploadWorkspaceFile("notes/other.txt", "other workspace");
    releaseCancel.resolve();
    await Promise.all([run, importing]);
    assert.equal(client.getStateView().snapshot?.workspaceId, "import-other");
    assert.deepEqual(client.getStateView().snapshot?.files, { "notes/other.txt": "other workspace" });
    const original = await client.switchWorkspace("ws-stop");
    assert.deepEqual(original.files, pendingAction === "archive import"
      ? { "notes/original.txt": "original" }
      : { "notes/original.txt": "original", "notes/later.txt": "later" });
    if (pendingAction === "session switch") assert.equal(original.activeSessionId, "origin-session");
  } finally { releaseCancel.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});
}

it("retains the workspace lease until an in-flight upload settles during disposal", async (t) => {
  const locks = new InMemoryLockProvider();
  const backend = new InMemoryJournalBackend();
  const blobs = new ContentAddressedBlobStore(new InMemoryBlobBackend());
  const lease = await locks.acquireWorkspaceLease("dispose-upload", "owner");
  const store = new WorkspaceKernelStore(backend, blobs, lease, locks);
  await store.initWorkspace("Upload disposal");
  const entered = deferred();
  const release = deferred();
  const committed = deferred();
  const originalCommit = backend.commitAtomic.bind(backend);
  t.mock.method(backend, "commitAtomic", async (...args: Parameters<typeof backend.commitAtomic>) => {
    entered.resolve(); await release.promise;
    await originalCommit(...args);
    committed.resolve();
  });
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() {} }, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    const uploading = client.uploadWorkspaceFile("notes/upload.txt", "in progress");
    void uploading.catch(() => {});
    await entered.promise;
    host.dispose();
    const observer = await locks.acquireWorkspaceLease("dispose-upload", "contender");
    assert.equal(observer.mode, "ReadOnly_Observer", "Pending durable writes must keep their lease");
    await observer.release();
    release.resolve();
    await committed.promise;
    await new Promise<void>((done) => setImmediate(done));
    const nextOwner = await locks.acquireWorkspaceLease("dispose-upload", "next-owner");
    assert.equal(nextOwner.mode, "ReadWrite_Leader", "The lease must release once the write settles");
    await nextOwner.release();
  } finally { release.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});

it("restores an older archive after live commits without retaining newer files or journal rows", async () => {
  const store = await fixture();
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() {} }, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Archive restore");
    const archive = await client.exportWorkspace();
    await client.uploadWorkspaceFile("later.txt", "Remove by explicit restore");
    const restored = await client.importWorkspace(archive);
    assert.equal(restored.headSeq, 0);
    assert.deepEqual(restored.files, {});
    assert.deepEqual(client.getStateView().snapshot?.journal, []);
  } finally { client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});


it("keeps a new workspace selected while an old workspace run continues concurrently in the background", async () => {
  const store = await fixture();
  const entered = deferred();
  const release = deferred();
  const registry = new BrowserToolRegistry(store);
  const inference: InferenceAdapter = { async *streamChat(req) {
    const userPrompt = [...req.messages].reverse().find((m) => m.role === "user")?.content ?? "";
    const hasToolOutput = req.messages.some((m) => m.role === "tool");
    if (userPrompt === "Wait" && !hasToolOutput) {
      entered.resolve(); await release.promise;
      yield { providerId: "test", model: "fixture", deltaText: "", usage: null, finishReason: "tool_calls",
        toolCalls: [{ id: "late-write", name: "workspace.write_file", argumentsJson: JSON.stringify({ path: "old-workspace-only.txt", content: "isolated" }) }],
      };
      return;
    }
    yield { providerId: "test", model: "fixture", deltaText: `Done:${userPrompt}`, usage: null, finishReason: "stop", toolCalls: [] };
  } };
  const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions());
  const channel = new MessageChannel();
  const host = new KernelWorkerHost(store, runtime, channel.port1);
  const client = new VelesAppClient(channel.port2);
  try {
    await client.initWorkspace("Old workspace");
    const run = client.startRun("old-workspace-run", "Wait", config);
    await entered.promise;
    const next = await bounded(client.switchWorkspace("ws-next", "Next"));
    assert.equal(next.workspaceId, "ws-next");
    assert.equal(client.getStateView().snapshot?.workspaceId, "ws-next");
    assert.equal(client.getStateView().inFlightRunId, null);
    // Start and complete a concurrent run in ws-next while old-workspace-run is still in flight in ws-stop
    const nextDone = await bounded(client.startRun("next-workspace-run", "Concurrent", config));
    assert.equal(nextDone.workspaceId, "ws-next");
    assert.equal(nextDone.runState, "completed");
    assert.deepEqual(nextDone.files, {});
    // Now release old-workspace-run in the background and verify it writes only to ws-stop, keeping ws-next selected
    release.resolve();
    const oldSettled = await bounded(run);
    assert.equal(oldSettled.workspaceId, "ws-stop");
    assert.equal(oldSettled.runState, "completed");
    assert.equal(client.getStateView().snapshot?.workspaceId, "ws-next");
    assert.deepEqual(client.getStateView().snapshot?.files, {});
    const old = await client.switchWorkspace("ws-stop");
    assert.equal(old.runState, "completed");
    assert.deepEqual(old.files, { "old-workspace-only.txt": "isolated" });
  } finally { release.resolve(); client.dispose(); host.dispose(); channel.port1.close(); channel.port2.close(); }
});

it("accepts a durable completion that was already committing when Stop arrived", async () => {
  const entered = deferred();
  const release = deferred();
  class DelayedCompletionBackend extends InMemoryJournalBackend {
    override async commitAtomic(...args: Parameters<InMemoryJournalBackend["commitAtomic"]>) {
      if (args[2]?.type === "RunCompleted") { entered.resolve(); await release.promise; }
      return super.commitAtomic(...args);
    }
  }
  const locks = new InMemoryLockProvider();
  const lease = await locks.acquireWorkspaceLease("ws-finish-race", "test");
  const store = new WorkspaceKernelStore(new DelayedCompletionBackend(), new ContentAddressedBlobStore(new InMemoryBlobBackend()), lease);
  await store.initWorkspace("Completion race");
  const registry = new BrowserToolRegistry(store);
  const inference: InferenceAdapter = { async *streamChat() {
    yield { providerId: "test", model: "fixture", deltaText: "Done", usage: null, finishReason: "stop", toolCalls: [] };
  } };
  const runtime = new AgentKernelRuntime(store, inference, registry, registry.getToolDefinitions());
  const run = runtime.startRun("finish-race", "Finish", config);
  await entered.promise;
  const stop = runtime.cancelActiveRun("finish-race");
  release.resolve();
  await bounded(stop);
  assert.equal(await run, "completed");
  assert.equal((await store.getJournal()).at(-1)?.type, "RunCompleted");
});


it("Stop aborts classifier transport and prevents late routing checkpoints", async () => {
  const store = await fixture();
  const entered = deferred<AbortSignal>();
  const release = deferred();
  let inferenceCalls = 0;
  const registry = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, { async *streamChat() { inferenceCalls += 1; } }, registry, registry.getToolDefinitions(), {
    multiTurnWorkspaceContext: true,
    classifier: async (_prompt, _paths, _profiles, signal) => {
      entered.resolve(signal); await release.promise;
      return { status: "undecided", reason: "late" };
    },
  });
  const run = runtime.startRun("classifier-stop", "Classify this", config);
  const signal = await entered.promise;
  await runtime.cancelActiveRun("classifier-stop");
  assert.equal(signal.aborted, true);
  assert.equal(await bounded(run), "cancelled");
  release.resolve();
  await new Promise((done) => setTimeout(done, 10));
  assert.equal(inferenceCalls, 0);
  assert.ok(!(await store.getJournal()).some((record) => record.payload.harnessRouting));
});


it("stops an uncooperative inference iterator and ignores its late tool call", async () => {
  const store = await fixture();
  const started = deferred();
  const release = deferred();
  const inference: InferenceAdapter = { async *streamChat() {
    started.resolve();
    await release.promise;
    yield { providerId: "test", model: "fixture", deltaText: "late", usage: null, finishReason: "tool_calls", toolCalls: [{ id: "late", name: "workspace.write_file", argumentsJson: JSON.stringify({ path: "late.txt", content: "unexpected" }) }] };
  } };
  const tools = new BrowserToolRegistry(store);
  const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions());
  const run = runtime.startRun("stop-inference", "Wait", config);
  await started.promise;
  await runtime.cancelActiveRun("stop-inference");
  try { assert.equal(await bounded(run), "cancelled"); }
  finally { release.resolve(); }
  await new Promise((done) => setTimeout(done, 10));
  assert.deepEqual(await store.listFiles(), []);
  assert.equal((await store.getJournal()).at(-1)?.type, "RunCancelled");
});

for (const effectClass of ["idempotent_write", "non_idempotent"] as const) {
  it(`stops a delayed ${effectClass} tool without committing its late result`, async () => {
    const store = await fixture();
    const entered = deferred();
    const release = deferred();
    const tools = new BrowserToolRegistry(store, [{
      spec: { name: "test.delay", description: "Delayed fixture", effectClass },
      async handler() { entered.resolve(); await release.promise; return { outputText: "late", fileUpdates: { "late.txt": "unexpected" } }; },
    }]);
    const inference: InferenceAdapter = { async *streamChat() {
      yield { providerId: "test", model: "fixture", deltaText: "", usage: null, finishReason: "tool_calls", toolCalls: [{ id: "delay", name: "test.delay", argumentsJson: "{}" }] };
    } };
    const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions());
    const run = runtime.startRun("stop-tool", "Wait", config);
    await entered.promise;
    assert.equal(await runtime.handleWorkerCommand({ type: "CANCEL_RUN", runId: "stop-tool" }), effectClass === "non_idempotent" ? "interrupted" : "cancelled");
    try { assert.equal(await bounded(run), effectClass === "non_idempotent" ? "interrupted" : "cancelled"); }
    finally { release.resolve(); }
    await new Promise((done) => setTimeout(done, 10));
    const journal = await store.getJournal();
    assert.deepEqual(await store.listFiles(), []);
    assert.ok(!journal.some((r) => r.type === "ToolResultCommitted"));
    if (effectClass === "non_idempotent") assert.equal(journal.at(-1)?.payload.interruptReason, "uncertain_tool_outcome");
    const rebuiltMessages = ContextBuilder.buildMessages(journal);
    const toolReply = rebuiltMessages.find(
      (m) => m.role === "tool" && m.toolCallId === "delay",
    );
    assert.ok(
      toolReply?.content.includes("TOOL_CALL_INTERRUPTED"),
      "Cancelled/interrupted mid-tool run must emit synthetic tool response so follow-up runs never fail with HTTP 400",
    );
  });
}
