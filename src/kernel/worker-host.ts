/**
 * @module SYS-KERNEL/worker-host
 * Web Worker host endpoint bridging `WorkspaceKernelStore` and `AgentKernelRuntime`
 * to the main-thread UI exclusively over a `MessagePort`-compatible boundary.
 * Enforces !INVARIANT [Coupling_Cohesion_Law] (Data Coupling) and Resource Lifetimes.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import {
  type InterruptReason,
  type PendingApprovalSummary,
  type RunJournalRecord,
  type ToolEffectClass,
  type UiToWorkerMessage,
  type UiToWorkerRequestPayload,
  type WorkerPortLike,
  type WorkerToUiMessage,
  type WorkspaceStateSnapshot,
} from "../protocol/types.ts";
import { AgentManager } from "./agent-session.ts";
import type { WorkspaceKernelStore } from "./journal.ts";
import { isLifecycleJournalRecord } from "./run-cancellation.ts";
import {
  type AgentKernelRuntime,
  ContextBuilder,
  type WorkerRuntimeEvent,
} from "./runtime.ts";

function attachPortListener(
  port: WorkerPortLike,
  handler: (ev: { readonly data: unknown }) => void,
): () => void {
  if (typeof port.addEventListener === "function") {
    port.addEventListener("message", handler);
    port.start?.();
    return () => {
      port.removeEventListener?.("message", handler);
    };
  }
  const fallbackPort = port as {
    onmessage?: ((ev: { readonly data: unknown }) => void) | null;
  };
  fallbackPort.onmessage = handler;
  port.start?.();
  return () => {
    if (fallbackPort.onmessage === handler) {
      fallbackPort.onmessage = null;
    }
  };
}

function isUiToWorkerMessage(raw: unknown): raw is UiToWorkerMessage {
  if (typeof raw !== "object" || raw === null) return false;
  const msg = raw as Record<string, unknown>;
  if (msg.channel !== "veles:worker-port" || typeof msg.requestId !== "string") {
    return false;
  }
  return typeof msg.request === "object" && msg.request !== null;
}

function extractPendingApproval(
  journal: readonly RunJournalRecord[],
  activeRunId: string | null,
  runState: string | null,
): PendingApprovalSummary | null {
  if (!activeRunId || runState !== "awaiting_approval") {
    return null;
  }
  const runRecords = journal.filter(
    (r) => r.runId === activeRunId && isLifecycleJournalRecord(r),
  );
  const tail = runRecords[runRecords.length - 1];
  if (!tail || tail.type !== "ApprovalRequested") {
    return null;
  }
  const p = tail.payload;
  return {
    runId: activeRunId,
    operationId: tail.relatedOperationId,
    toolCallId: typeof p.toolCallId === "string" ? p.toolCallId : "call_0",
    toolName: String(p.toolName ?? ""),
    effectClass:
      (p.effectClass as ToolEffectClass | undefined) ?? "non_idempotent",
    args:
      typeof p.args === "object" && p.args !== null
        ? (p.args as Record<string, unknown>)
        : {},
    assistantText: typeof p.assistantText === "string" ? p.assistantText : "",
  };
}

function extractInterruptReason(
  journal: readonly RunJournalRecord[],
  activeRunId: string | null,
  runState: string | null,
): InterruptReason | null {
  if (!activeRunId || runState !== "interrupted") return null;
  const runRecords = journal.filter(
    (r) => r.runId === activeRunId && isLifecycleJournalRecord(r),
  );
  const tail = runRecords[runRecords.length - 1];
  if (!tail || tail.type !== "RunInterrupted") return null;
  return (tail.payload.interruptReason as InterruptReason | undefined) ?? null;
}

function visibleSessionRun(
  journal: readonly RunJournalRecord[],
  sessionId: string,
): RunJournalRecord | undefined {
  const created = [...journal].reverse().find(
    (record) =>
      record.type === "RunCreated" &&
      (record.payload.sessionId ?? "session-main") === sessionId,
  );
  if (!created) return undefined;
  if (journal.some((record) => record.payload.deletedRunId === created.runId)) {
    return undefined;
  }
  return [...journal].reverse().find(
    (record) => record.runId === created.runId && isLifecycleJournalRecord(record),
  );
}

export class KernelWorkerHost {
  private readonly store: WorkspaceKernelStore;
  private readonly runtime: AgentKernelRuntime;
  public readonly agentManager: AgentManager;
  private readonly port: WorkerPortLike;
  private activeSessionId = "session-main";
  private readonly activeRequests = new Map<string, Promise<unknown>>();
  private detachPort: (() => void) | null = null;
  private unsubscribeRuntime: (() => void) | null = null;
  private unsubscribeJournal: (() => void) | null = null;
  private disposed = false;

  constructor(
    store: WorkspaceKernelStore,
    runtime: AgentKernelRuntime,
    port: WorkerPortLike,
    agentManager?: AgentManager,
  ) {
    this.store = store;
    this.runtime = runtime;
    this.agentManager = agentManager ?? new AgentManager(store, runtime);
    this.port = port;
    this.unsubscribeJournal = this.store.onCommit((record) => {
      if (!this.disposed) this.port.postMessage({
        channel: "veles:worker-port", kind: "EVENT", eventType: "JOURNAL_COMMITTED",
        runId: record.runId, state: record.committedState, record,
      } satisfies WorkerToUiMessage);
    });

    this.unsubscribeRuntime = this.runtime.onEvent((ev) => {
      this.forwardRuntimeEvent(ev);
    });
    this.detachPort = attachPortListener(this.port, (ev) => {
      void this.receivePortMessage(ev.data);
    });
  }

  public get isDisposed(): boolean {
    return this.disposed;
  }

  public async receivePortMessage(raw: unknown): Promise<void> {
    await this.handleIncomingMessage(raw);
  }

  private forwardRuntimeEvent(ev: WorkerRuntimeEvent): void {
    if (this.disposed) return;
    const msg: WorkerToUiMessage = {
      channel: "veles:worker-port",
      kind: "EVENT",
      eventType: ev.type,
      runId: ev.runId,
      state: ev.state,
      detail: ev.detail,
    };
    this.port.postMessage(msg);
  }

  private async handleIncomingMessage(raw: unknown): Promise<void> {
    if (this.disposed || !isUiToWorkerMessage(raw)) return;
    const { requestId, request } = raw;
    try {
      const response = await this.dispatchRequest(requestId, request);
      if (!this.disposed) {
        this.port.postMessage(response);
      }
    } catch (err) {
      if (!this.disposed) {
        const errorMsg: WorkerToUiMessage = {
          channel: "veles:worker-port",
          kind: "RESPONSE",
          requestId,
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        };
        this.port.postMessage(errorMsg);
      }
    }
  }

  private async dispatchRequest(
    requestId: string,
    req: UiToWorkerRequestPayload,
  ): Promise<WorkerToUiMessage> {
    if (req.type === "INIT_WORKSPACE") {
      if (this.store.getLeaseMode() === "ReadOnly_Observer") {
        return this.okObserverInitResponse(requestId, req.title);
      }
      await this.store.initWorkspace(req.title);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "GET_SNAPSHOT") {
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "CREATE_SESSION") {
      await this.agentManager.createSession({
        id: req.session.id,
        title: req.session.title,
        parentSessionId: req.session.parentSessionId,
        workspaceId: req.session.workspaceId || this.store.getActiveWorkspaceId(),
        model: req.session.model,
        allowedTools:
          req.session.allowedTools ?? this.runtime.getRegisteredToolNames(),
        delegatableTools: req.session.delegatableTools,
        unrestrictedTools: req.session.unrestrictedTools,
        status: "idle",
      });
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "START_RUN") {
      await this.trackRunRequest(req.runId, () => this.handleStartRunRequest(req));
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "CANCEL_RUN") {
      await this.runtime.cancelActiveRun(req.runId);
      await this.activeRequests.get(req.runId);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "RESOLVE_APPROVAL") {
      await this.trackRunRequest(req.runId, () => this.runtime.resolveApproval(req.runId, req.approved));
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    return this.dispatchRecoveryOrArchiveRequest(requestId, req);
  }

  private async dispatchRecoveryOrArchiveRequest(
    requestId: string,
    req: UiToWorkerRequestPayload,
  ): Promise<WorkerToUiMessage> {
    if (req.type === "RECOVER_BOOT") {
      const decision = await this.runtime.recoverUnfinishedRun(
        req.currentGatewayInstanceId,
      );
      await this.agentManager.recoverSessionsFromJournal();
      const snapshot = await this.buildWorkspaceSnapshot();
      return {
        channel: "veles:worker-port",
        kind: "RESPONSE",
        requestId,
        ok: true,
        snapshot,
        recoveryDecision: decision,
      };
    }
    if (req.type === "EXPORT_BUNDLE") {
      const bundle = await this.store.exportWorkspaceBundle();
      return {
        channel: "veles:worker-port",
        kind: "RESPONSE",
        requestId,
        ok: true,
        bundle,
      };
    }
    if (req.type === "RESOLVE_UNCERTAIN_TOOL") {
      await this.trackRunRequest(req.runId, () => this.runtime.resolveUncertainTool(req.runId, req.resolution));
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "SIMULATE_CHAOS") {
      await this.executeChaosScenario(req.scenario);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "IMPORT_BUNDLE") {
      await this.cancelInFlightRunBeforeWorkspaceChange();
      await this.store.importWorkspaceBundle(req.bundle);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    return this.dispatchWorkspaceRequest(requestId, req);
  }

  private async dispatchWorkspaceRequest(
    requestId: string,
    req: UiToWorkerRequestPayload,
  ): Promise<WorkerToUiMessage> {
    if (req.type === "SWITCH_WORKSPACE") {
      await this.cancelInFlightRunBeforeWorkspaceChange();
      this.activeSessionId = "session-main";
      await this.store.switchWorkspace(req.workspaceId, req.title);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "DELETE_WORKSPACE") {
      await this.cancelInFlightRunBeforeWorkspaceChange();
      this.activeSessionId = "session-main";
      await this.store.deleteWorkspace(req.workspaceId);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "WRITE_WORKSPACE_FILE") {
      await this.store.writeWorkspaceFile(req.path, req.content);
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "SWITCH_SESSION") {
      await this.handleSwitchSessionRequest(req.sessionId, req.title);
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "DELETE_RUN") {
      await this.store.deleteRun(req.runId);
      await this.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId);
    }
    if (req.type === "UPDATE_SERVICE_CONFIG") {
      this.runtime.setServiceConfig(req.serviceConfig);
      return this.okSnapshotResponse(requestId);
    }
    throw new Error("ERR_UNKNOWN_UI_REQUEST");
  }

  private async handleStartRunRequest(
    req: Extract<UiToWorkerRequestPayload, { readonly type: "START_RUN" }>,
  ): Promise<void> {
    if (req.serviceConfig) {
      this.runtime.setServiceConfig(req.serviceConfig);
    }
    const sessionId = req.sessionId?.trim() || "session-main";
    this.activeSessionId = sessionId;
    const existing = this.agentManager.getSession(sessionId);
    this.agentManager.upsertSessionInMemory({
      id: sessionId,
      title: existing?.title,
      parentSessionId: existing?.parentSessionId,
      workspaceId: this.store.getActiveWorkspaceId(),
      model: req.providerConfig,
      allowedTools:
        existing?.allowedTools ?? this.runtime.getRegisteredToolNames(),
      delegatableTools: existing?.delegatableTools,
      unrestrictedTools: existing?.unrestrictedTools,
      status: "idle",
    });
    await this.agentManager.runWithId(sessionId, req.prompt, req.runId);
  }

  private async handleSwitchSessionRequest(
    rawSessionId: string,
    title?: string,
  ): Promise<void> {
    await this.cancelInFlightRunBeforeWorkspaceChange();
    const targetId = rawSessionId.trim() || "session-main";
    this.activeSessionId = targetId;
    if (targetId !== "session-main" && !this.agentManager.getSession(targetId)) {
      await this.agentManager.createSession({
        id: targetId,
        title: title ?? targetId,
        workspaceId: this.store.getActiveWorkspaceId(),
        model: {
          providerId: "openai",
          baseUrl: "https://api.openai.com/v1",
          model: "gpt-4o-mini",
        },
        allowedTools: this.runtime.getRegisteredToolNames(),
        status: "idle",
      });
    }
  }

  private async trackRunRequest(runId: string, operation: () => Promise<unknown>): Promise<void> {
    if (this.activeRequests.has(runId)) throw new Error("ERR_RUN_ALREADY_ACTIVE");
    const pending = Promise.resolve().then(operation);
    this.activeRequests.set(runId, pending);
    try { await pending; }
    finally { this.activeRequests.delete(runId); }
  }

  private async cancelInFlightRunBeforeWorkspaceChange(): Promise<void> {
    const tails = new Map<string, RunJournalRecord>();
    for (const record of await this.store.getJournal()) {
      if (isLifecycleJournalRecord(record)) {
        tails.set(record.runId, record);
      }
    }
    const runIds = new Set(this.activeRequests.keys());
    for (const record of tails.values()) {
      if (["created", "preparing", "inferring", "awaiting_approval", "awaiting_tool", "executing_tool"].includes(record.committedState)) runIds.add(record.runId);
    }
    await Promise.all([...runIds].map((runId) => this.runtime.cancelActiveRun(runId)));
    await Promise.allSettled([...this.activeRequests.values()]);
  }

  private async okSnapshotResponse(requestId: string): Promise<WorkerToUiMessage> {
    const snapshot = await this.buildWorkspaceSnapshot();
    return {
      channel: "veles:worker-port",
      kind: "RESPONSE",
      requestId,
      ok: true,
      snapshot,
    };
  }

  private async okObserverInitResponse(
    requestId: string,
    fallbackTitle: string,
  ): Promise<WorkerToUiMessage> {
    const meta = await this.store.getMetadata();
    if (meta !== null) {
      return this.okSnapshotResponse(requestId);
    }
    const emptySnapshot: WorkspaceStateSnapshot = {
      workspaceId: "ws-alpha",
      title: fallbackTitle,
      headSeq: 0,
      activeRunId: null,
      runState: null,
      updatedAtMs: Date.now(),
      leaseMode: "ReadOnly_Observer",
      gatewayInstanceId: this.runtime.currentGatewayInstanceId,
      interruptReason: null,
      files: {},
      fileHashes: {},
      messages: [],
      pendingApproval: null,
      journal: [],
    };
    return {
      channel: "veles:worker-port",
      kind: "RESPONSE",
      requestId,
      ok: true,
      snapshot: emptySnapshot,
    };
  }

  private async executeChaosScenario(
    scenario:
      | "quota_exceeded"
      | "drop_stream_mid_inference"
      | "uncertain_tool_outcome",
  ): Promise<void> {
    if (scenario === "quota_exceeded") {
      throw new Error(
        "ERR_STORAGE_QUOTA_EXCEEDED: Simulated OPFS/IndexedDB storage quota exhaustion (Scenario 6)",
      );
    }
    const runId = `run-chaos-${Date.now()}`;
    await this.store.appendEvent({
      eventId: `${runId}-1`,
      runId,
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: `[Chaos Fault Injection: ${scenario}]` },
    });
    await this.store.appendEvent({
      eventId: `${runId}-2`,
      runId,
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await this.store.appendEvent({
      eventId: `${runId}-3`,
      runId,
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: {},
    });
    if (scenario === "drop_stream_mid_inference") {
      this.forwardRuntimeEvent({
        type: "STREAM_DELTA",
        runId,
        state: "inferring",
        detail: { deltaText: "Uncommitted partial stream before network drop..." },
      });
      await this.store.appendEvent({
        eventId: `${runId}-4`,
        runId,
        type: "RunInterrupted",
        nextState: "interrupted",
        payload: {
          interruptReason: "gateway_disconnected_safe_to_retry",
          canAutoRetry: true,
        },
      });
      return;
    }
    await this.seedUncertainToolChaos(runId);
  }

  private async seedUncertainToolChaos(runId: string): Promise<void> {
    await this.store.appendEvent({
      eventId: `${runId}-4`,
      runId,
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: `${runId}-op-webhook`,
      payload: {
        toolCallId: "call_chaos_webhook",
        toolName: "webhook.send",
        effectClass: "non_idempotent",
        args: { endpoint: "https://hooks.example.test/deploy" },
        assistantText: "Dispatching production webhook...",
      },
    });
    await this.store.appendEvent({
      eventId: `${runId}-5`,
      runId,
      type: "ToolDispatched",
      nextState: "executing_tool",
      relatedOperationId: `${runId}-op-webhook`,
      payload: {
        toolCallId: "call_chaos_webhook",
        toolName: "webhook.send",
        effectClass: "non_idempotent",
        gatewayInstanceId: "gw-crashed-old-instance",
      },
    });
    await this.runtime.recoverUnfinishedRun("gw-restarted-new-instance");
  }

  public async buildWorkspaceSnapshot(): Promise<WorkspaceStateSnapshot> {
    const meta = await this.store.getMetadata();
    if (!meta) {
      throw new Error("ERR_WORKSPACE_NOT_INITIALIZED");
    }
    const journal = await this.store.getJournal();
    const visible = visibleSessionRun(journal, this.activeSessionId);
    const fileHashes = await this.store.getFileTree();
    const filePaths = Object.keys(fileHashes).sort();
    const files: Record<string, string> = {};
    for (const relPath of filePaths) {
      files[relPath] = await this.store.readFileTextOrDataUrl(relPath);
    }
    const messages = ContextBuilder.buildMessages(journal, {
      sessionId: this.activeSessionId,
      hotRunWindow: 2,
    });
    const pendingApproval = extractPendingApproval(
      journal,
      visible?.runId ?? null,
      visible?.committedState ?? null,
    );
    const interruptReason = extractInterruptReason(
      journal,
      visible?.runId ?? null,
      visible?.committedState ?? null,
    );
    const allWorkspaces = await this.store.listWorkspaces();
    const workspaces = allWorkspaces.map((w) => ({
      workspaceId: w.workspaceId,
      title: w.title,
      headSeq: w.headSeq,
      updatedAtMs: w.updatedAtMs,
    }));
    return {
      workspaceId: meta.workspaceId,
      title: meta.title,
      headSeq: meta.headSeq,
      activeRunId: visible?.runId ?? null,
      runState: visible?.committedState ?? null,
      updatedAtMs: meta.updatedAtMs,
      leaseMode: this.store.getLeaseMode(),
      gatewayInstanceId: this.runtime.currentGatewayInstanceId,
      interruptReason,
      files,
      fileHashes,
      messages,
      pendingApproval,
      journal,
      workspaces,
      sessions: this.agentManager.listSessions(),
      activeSessionId: this.activeSessionId,
      serviceConfig: this.runtime.getServiceConfig(),
    };
  }

  public dispose(): void {
    this.unsubscribeJournal?.();
    this.unsubscribeJournal = null;
    if (this.disposed) return;
    this.disposed = true;
    this.detachPort?.();
    this.detachPort = null;
    this.unsubscribeRuntime?.();
    this.unsubscribeRuntime = null;
  }
}
