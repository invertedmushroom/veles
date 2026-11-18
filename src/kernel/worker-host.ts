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
import {
  type createDefaultWorkspaceTools,
  createSubagentTools,
  type RegisteredBrowserTool,
} from "./tools.ts";

interface WorkspaceRuntimeSlot {
  readonly workspaceId: string;
  readonly store: WorkspaceKernelStore;
  readonly runtime: AgentKernelRuntime;
  readonly agentManager: AgentManager;
  activeSessionId: string;
  readonly activeRequests: Map<string, Promise<unknown>>;
  activeOperations: number;
  readonly disposeListeners: () => void;
}

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
  private readonly rootStore: WorkspaceKernelStore;
  private readonly rootRuntime: AgentKernelRuntime;
  private readonly slots = new Map<string, WorkspaceRuntimeSlot>();
  private currentSlot: WorkspaceRuntimeSlot;
  private readonly port: WorkerPortLike;
  private detachPort: (() => void) | null = null;
  private disposed = false;
  private readonly workspaceToolsFactory?: typeof createDefaultWorkspaceTools;

  constructor(
    store: WorkspaceKernelStore,
    runtime: AgentKernelRuntime,
    port: WorkerPortLike,
    agentManager?: AgentManager,
    workspaceToolsFactory?: typeof createDefaultWorkspaceTools,
  ) {
    this.rootStore = store;
    this.rootRuntime = runtime;
    this.port = port;
    this.workspaceToolsFactory = workspaceToolsFactory;
    this.currentSlot = this.bindSlot(
      store,
      runtime,
      agentManager ?? new AgentManager(store, runtime),
    );
    this.detachPort = attachPortListener(this.port, (ev) => {
      void this.receivePortMessage(ev.data);
    });
  }

  public get agentManager(): AgentManager {
    return this.currentSlot.agentManager;
  }

  public getLatestModelConfig(): ReturnType<AgentKernelRuntime["getLatestModelConfig"]> {
    const active = this.currentSlot.runtime.getLatestModelConfig();
    if (active) return active;
    for (const slot of this.slots.values()) {
      const cfg = slot.runtime.getLatestModelConfig();
      if (cfg) return cfg;
    }
    return undefined;
  }

  public getServiceConfig(): ReturnType<AgentKernelRuntime["getServiceConfig"]> {
    return this.currentSlot.runtime.getServiceConfig();
  }

  private get runtime(): AgentKernelRuntime {
    return this.currentSlot.runtime;
  }

  private bindSlot(
    store: WorkspaceKernelStore,
    runtime: AgentKernelRuntime,
    agentManager: AgentManager,
  ): WorkspaceRuntimeSlot {
    const workspaceId = store.getActiveWorkspaceId();
    const unsubJournal = store.onCommit((record) => {
      if (!this.disposed) {
        this.port.postMessage({
          channel: "veles:worker-port",
          kind: "EVENT",
          eventType: "JOURNAL_COMMITTED",
          runId: record.runId,
          state: record.committedState,
          record,
        } satisfies WorkerToUiMessage);
      }
    });
    const unsubRuntime = runtime.onEvent((ev) => {
      if (this.currentSlot?.workspaceId === workspaceId) {
        this.forwardRuntimeEvent(ev);
      }
    });
    const slot: WorkspaceRuntimeSlot = {
      workspaceId,
      store,
      runtime,
      agentManager,
      activeSessionId: "session-main",
      activeRequests: new Map<string, Promise<unknown>>(),
      activeOperations: 0,
      disposeListeners: () => {
        unsubJournal();
        unsubRuntime();
      },
    };
    this.slots.set(workspaceId, slot);
    return slot;
  }

  private createForkedSlot(targetStore: WorkspaceKernelStore): WorkspaceRuntimeSlot {
    let forkedAdapter: { registerTool?: (t: RegisteredBrowserTool) => void } | null = null;
    let forkedRuntime: AgentKernelRuntime | undefined;
    const runtime = this.rootRuntime.forkForStore(targetStore, (adapter, store) => {
      const workspaceTools = this.workspaceToolsFactory?.(
        (runId) => forkedRuntime?.getLatestModelConfig(runId),
        () => forkedRuntime?.getServiceConfig(),
      );
      const next =
        (
          adapter as {
            forkForStore?: (s: WorkspaceKernelStore, tools?: readonly RegisteredBrowserTool[]) => typeof adapter;
          }
        ).forkForStore?.(store, workspaceTools) ?? adapter;
      forkedAdapter = next as { registerTool?: (t: RegisteredBrowserTool) => void };
      return next;
    });
    forkedRuntime = runtime;
    const agentManager = new AgentManager(targetStore, runtime);
    if (this.rootRuntime.getRegisteredToolNames().includes("agent.delegate")) {
      for (const subTool of createSubagentTools(agentManager)) {
        (forkedAdapter as { registerTool?: (t: RegisteredBrowserTool) => void } | null)?.registerTool?.(subTool);
        runtime.registerToolDefinition(subTool.spec);
      }
    }
    return this.bindSlot(targetStore, runtime, agentManager);
  }

  private async acquireWorkspaceSlot(
    workspaceId: string,
    title?: string,
  ): Promise<WorkspaceRuntimeSlot> {
    const cleanId = workspaceId.trim();
    if (!cleanId) {
      throw new Error("ERR_INVALID_WORKSPACE_ID: Workspace ID cannot be empty");
    }
    const existing = this.slots.get(cleanId);
    if (existing) {
      await existing.store.ensureWorkspaceLease(title);
      if (this.isSlotIdle(existing)) {
        await existing.agentManager.recoverSessionsFromJournal();
      }
      return existing;
    }
    const forkedStore = await this.rootStore.forkForWorkspace(cleanId, title);
    const slot = this.createForkedSlot(forkedStore);
    await slot.agentManager.recoverSessionsFromJournal();
    return slot;
  }

  private async resolveSlotForRun(runId: string): Promise<WorkspaceRuntimeSlot> {
    for (const slot of this.slots.values()) {
      if (slot.activeRequests.has(runId)) return slot;
    }
    const currentJournal = await this.currentSlot.store.getJournal();
    if (currentJournal.some((r) => r.runId === runId)) {
      return this.currentSlot;
    }
    for (const slot of this.slots.values()) {
      const journal = await slot.store.getJournal();
      if (journal.some((r) => r.runId === runId)) return slot;
    }
    return this.currentSlot;
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
    const slot = this.currentSlot;
    if (req.type === "INIT_WORKSPACE") {
      if (slot.store.getLeaseMode() === "ReadOnly_Observer") {
        return this.okObserverInitResponse(requestId, req.title, slot);
      }
      await this.withSlotLease(slot, async () => {
        await slot.store.initWorkspace(req.title);
        await slot.agentManager.recoverSessionsFromJournal();
      }, req.title);
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "GET_SNAPSHOT") {
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "CREATE_SESSION") {
      await this.withSlotLease(slot, () => slot.agentManager.createSession({
        id: req.session.id,
        title: req.session.title,
        parentSessionId: req.session.parentSessionId,
        workspaceId: slot.workspaceId,
        model: req.session.model,
        allowedTools:
          req.session.allowedTools ?? slot.runtime.getRegisteredToolNames(),
        delegatableTools: req.session.delegatableTools,
        unrestrictedTools: req.session.unrestrictedTools,
        status: "idle",
      }));
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "START_RUN") {
      const slot = this.currentSlot;
      await this.trackSlotRunRequest(slot, req.runId, () =>
        this.handleStartRunRequest(slot, req),
      );
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "CANCEL_RUN") {
      const slot = await this.resolveSlotForRun(req.runId);
      await this.withSlotLease(slot, async () => {
        await slot.runtime.cancelActiveRun(req.runId);
        await slot.activeRequests.get(req.runId);
        await slot.agentManager.recoverSessionsFromJournal();
      });
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "RESOLVE_APPROVAL") {
      const slot = await this.resolveSlotForRun(req.runId);
      await this.trackSlotRunRequest(slot, req.runId, () =>
        slot.runtime.resolveApproval(req.runId, req.approved),
      );
      await slot.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId, slot);
    }
    return this.dispatchRecoveryOrArchiveRequest(requestId, req);
  }

  private async dispatchRecoveryOrArchiveRequest(
    requestId: string,
    req: UiToWorkerRequestPayload,
  ): Promise<WorkerToUiMessage> {
    const slot = this.currentSlot;
    if (req.type === "RECOVER_BOOT") {
      const decision = await this.withSlotLease(slot, () => slot.runtime.recoverUnfinishedRun(req.currentGatewayInstanceId));
      await slot.agentManager.recoverSessionsFromJournal();
      const snapshot = await this.buildWorkspaceSnapshot(slot);
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
      const bundle = await slot.store.exportWorkspaceBundle();
      return {
        channel: "veles:worker-port",
        kind: "RESPONSE",
        requestId,
        ok: true,
        bundle,
      };
    }
    if (req.type === "RESOLVE_UNCERTAIN_TOOL") {
      const slot = await this.resolveSlotForRun(req.runId);
      await this.trackSlotRunRequest(slot, req.runId, () =>
        slot.runtime.resolveUncertainTool(req.runId, req.resolution),
      );
      await slot.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "SIMULATE_CHAOS") {
      await this.withSlotLease(slot, () => this.executeChaosScenario(req.scenario, slot));
      await slot.agentManager.recoverSessionsFromJournal();
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "IMPORT_BUNDLE") {
      const slot = this.currentSlot;
      return this.withSlotLease(slot, async () => {
        await this.cancelInFlightRunsInSlot(slot);
        await slot.store.importWorkspaceBundle(req.bundle);
        await slot.agentManager.recoverSessionsFromJournal();
        return this.okSnapshotResponse(requestId, slot);
      });
    }
    return this.dispatchWorkspaceRequest(requestId, req);
  }

  private async dispatchWorkspaceRequest(
    requestId: string,
    req: UiToWorkerRequestPayload,
  ): Promise<WorkerToUiMessage> {
    const slot = this.currentSlot;
    if (req.type === "SWITCH_WORKSPACE") {
      const nextSlot = await this.handleSwitchWorkspaceRequest(
        req.workspaceId,
        req.title,
      );
      return this.okSnapshotResponse(requestId, nextSlot);
    }
    if (req.type === "DELETE_WORKSPACE") {
      const nextSlot = await this.handleDeleteWorkspaceRequest(req.workspaceId);
      return this.okSnapshotResponse(requestId, nextSlot);
    }
    if (req.type === "WRITE_WORKSPACE_FILE") {
      await this.withSlotLease(slot, () => slot.store.writeWorkspaceFile(req.path, req.content));
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "SWITCH_SESSION") {
      await this.withSlotLease(slot, () => this.handleSwitchSessionRequest(slot, req.sessionId, req.title));
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "DELETE_RUN") {
      await this.withSlotLease(slot, async () => {
        await slot.store.deleteRun(req.runId, { revertFiles: req.revertFiles === true });
        await slot.agentManager.recoverSessionsFromJournal();
      });
      return this.okSnapshotResponse(requestId, slot);
    }
    if (req.type === "UPDATE_SERVICE_CONFIG") {
      for (const slot of this.slots.values()) {
        slot.runtime.setServiceConfig(req.serviceConfig);
      }
      return this.okSnapshotResponse(requestId);
    }
    throw new Error("ERR_UNKNOWN_UI_REQUEST");
  }

  private async handleSwitchWorkspaceRequest(
    workspaceId: string,
    title?: string,
  ): Promise<WorkspaceRuntimeSlot> {
    const prevSlot = this.currentSlot;
    const cleanId = workspaceId.trim();
    if (cleanId === prevSlot.workspaceId) {
      await prevSlot.store.ensureWorkspaceLease(title);
      return prevSlot;
    }
    const nextSlot = await this.acquireWorkspaceSlot(cleanId, title);
    this.currentSlot = nextSlot;
    if (this.isSlotIdle(prevSlot)) {
      await prevSlot.store.releaseWorkspaceLease();
    }
    return nextSlot;
  }

  private async handleDeleteWorkspaceRequest(
    workspaceId: string,
  ): Promise<WorkspaceRuntimeSlot> {
    const cleanId = workspaceId.trim();
    const foregroundSlot = this.currentSlot;
    if (!cleanId) throw new Error("ERR_INVALID_WORKSPACE_ID: Workspace ID cannot be empty");
    const workspaces = await this.rootStore.listWorkspaces();
    if (!this.slots.has(cleanId) && !workspaces.some((w) => w.workspaceId === cleanId)) {
      return foregroundSlot;
    }
    const targetSlot = await this.acquireWorkspaceSlot(cleanId);
    if (targetSlot.store.getLeaseMode() !== "ReadWrite_Leader") {
      await targetSlot.store.releaseWorkspaceLease();
      throw new Error("ERR_WORKSPACE_LOCKED_BY_OTHER_TAB: Cannot delete a workspace owned by another tab");
    }
    await this.withSlotLease(targetSlot, async () => {
      await this.cancelInFlightRunsInSlot(targetSlot);
      await targetSlot.store.deleteWorkspaceStorage(cleanId);
    });
    await targetSlot.store.releaseWorkspaceLease();
    targetSlot.disposeListeners();
    this.slots.delete(cleanId);
    if (foregroundSlot.workspaceId !== cleanId) return foregroundSlot;
    const remaining = await this.rootStore.listWorkspaces();
    const nextWs = remaining.find((w) => w.workspaceId !== cleanId);
    const nextId = nextWs
      ? nextWs.workspaceId
      : cleanId === "ws-alpha"
        ? "ws-default"
        : "ws-alpha";
    const nextTitle = nextWs ? nextWs.title : "Alpha";
    const nextSlot = await this.acquireWorkspaceSlot(nextId, nextTitle);
    this.currentSlot = nextSlot;
    return nextSlot;
  }

  private async handleStartRunRequest(
    slot: WorkspaceRuntimeSlot,
    req: Extract<UiToWorkerRequestPayload, { readonly type: "START_RUN" }>,
  ): Promise<void> {
    if (req.serviceConfig) {
      for (const s of this.slots.values()) {
        s.runtime.setServiceConfig(req.serviceConfig);
      }
    }
    const sessionId = req.sessionId?.trim() || "session-main";
    slot.activeSessionId = sessionId;
    const existing = slot.agentManager.getSession(sessionId);
    slot.agentManager.upsertSessionInMemory({
      id: sessionId,
      title: existing?.title,
      parentSessionId: existing?.parentSessionId,
      workspaceId: slot.store.getActiveWorkspaceId(),
      model: req.providerConfig,
      allowedTools:
        existing?.allowedTools ?? slot.runtime.getRegisteredToolNames(),
      delegatableTools: existing?.delegatableTools,
      unrestrictedTools: existing?.unrestrictedTools,
      status: "idle",
    });
    await slot.agentManager.runWithId(sessionId, req.prompt, req.runId);
  }

  private async handleSwitchSessionRequest(
    slot: WorkspaceRuntimeSlot,
    rawSessionId: string,
    title?: string,
  ): Promise<void> {
    await this.cancelInFlightRunsInSlot(slot);
    const targetId = rawSessionId.trim() || "session-main";
    slot.activeSessionId = targetId;
    if (targetId !== "session-main" && !slot.agentManager.getSession(targetId)) {
      await slot.agentManager.createSession({
        id: targetId,
        title: title ?? targetId,
        workspaceId: slot.store.getActiveWorkspaceId(),
        model: {
          providerId: "openai",
          baseUrl: "https://api.openai.com/v1",
          model: "gpt-4o-mini",
        },
        allowedTools: slot.runtime.getRegisteredToolNames(),
        status: "idle",
      });
    }
  }

  private async trackSlotRunRequest(
    slot: WorkspaceRuntimeSlot,
    runId: string,
    operation: () => Promise<unknown>,
  ): Promise<void> {
    if (slot.activeRequests.has(runId)) throw new Error("ERR_RUN_ALREADY_ACTIVE");
    const pending = Promise.resolve().then(() => this.withSlotLease(slot, operation));
    slot.activeRequests.set(runId, pending);
    try {
      await pending;
    } finally {
      slot.activeRequests.delete(runId);
      await this.releaseIdleBackgroundSlot(slot);
    }
  }

  private isSlotIdle(slot: WorkspaceRuntimeSlot): boolean {
    return slot.activeRequests.size === 0 && slot.activeOperations === 0;
  }

  private async releaseIdleBackgroundSlot(slot: WorkspaceRuntimeSlot): Promise<void> {
    if (this.isSlotIdle(slot) && (this.disposed || this.currentSlot.workspaceId !== slot.workspaceId)) {
      await slot.store.releaseWorkspaceLease();
    }
  }

  private async withSlotLease<T>(
    slot: WorkspaceRuntimeSlot,
    operation: () => Promise<T>,
    title?: string,
  ): Promise<T> {
    slot.activeOperations += 1;
    try {
      await slot.store.ensureWorkspaceLease(title);
      if (slot.store.getLeaseMode() !== "ReadWrite_Leader") {
        throw new Error("ERR_WORKSPACE_LOCKED_BY_OTHER_TAB: This operation requires the workspace lease");
      }
      return await operation();
    } finally {
      slot.activeOperations -= 1;
      await this.releaseIdleBackgroundSlot(slot);
    }
  }

  private async cancelInFlightRunsInSlot(
    slot: WorkspaceRuntimeSlot,
  ): Promise<void> {
    const tails = new Map<string, RunJournalRecord>();
    for (const record of await slot.store.getJournal()) {
      if (isLifecycleJournalRecord(record)) {
        tails.set(record.runId, record);
      }
    }
    const runIds = new Set(slot.activeRequests.keys());
    for (const record of tails.values()) {
      if (
        [
          "created",
          "preparing",
          "inferring",
          "awaiting_approval",
          "awaiting_tool",
          "executing_tool",
        ].includes(record.committedState)
      ) {
        runIds.add(record.runId);
      }
    }
    await Promise.all([...runIds].map((runId) => slot.runtime.cancelActiveRun(runId)));
    await Promise.allSettled([...slot.activeRequests.values()]);
  }

  private async okSnapshotResponse(
    requestId: string,
    slot: WorkspaceRuntimeSlot = this.currentSlot,
  ): Promise<WorkerToUiMessage> {
    const snapshot = await this.buildWorkspaceSnapshot(slot);
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
    slot: WorkspaceRuntimeSlot,
  ): Promise<WorkerToUiMessage> {
    const meta = await slot.store.getMetadata();
    if (meta !== null) {
      return this.okSnapshotResponse(requestId, slot);
    }
    const emptySnapshot: WorkspaceStateSnapshot = {
      workspaceId: slot.workspaceId,
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
    slot: WorkspaceRuntimeSlot,
  ): Promise<void> {
    if (scenario === "quota_exceeded") {
      throw new Error(
        "ERR_STORAGE_QUOTA_EXCEEDED: Simulated OPFS/IndexedDB storage quota exhaustion (Scenario 6)",
      );
    }
    const runId = `run-chaos-${Date.now()}`;
    await slot.store.appendEvent({
      eventId: `${runId}-1`,
      runId,
      type: "RunCreated",
      nextState: "created",
      payload: { prompt: `[Chaos Fault Injection: ${scenario}]` },
    });
    await slot.store.appendEvent({
      eventId: `${runId}-2`,
      runId,
      type: "RunPreparing",
      nextState: "preparing",
      payload: {},
    });
    await slot.store.appendEvent({
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
      await slot.store.appendEvent({
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
    await this.seedUncertainToolChaos(runId, slot);
  }

  private async seedUncertainToolChaos(runId: string, slot: WorkspaceRuntimeSlot): Promise<void> {
    await slot.store.appendEvent({
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
    await slot.store.appendEvent({
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
    await slot.runtime.recoverUnfinishedRun("gw-restarted-new-instance");
  }

  public async buildWorkspaceSnapshot(
    slot: WorkspaceRuntimeSlot = this.currentSlot,
  ): Promise<WorkspaceStateSnapshot> {
    const meta = await slot.store.getMetadata();
    if (!meta) {
      throw new Error("ERR_WORKSPACE_NOT_INITIALIZED");
    }
    const journal = await slot.store.getJournal();
    const visible = visibleSessionRun(journal, slot.activeSessionId);
    const fileHashes = await slot.store.getFileTree();
    const filePaths = Object.keys(fileHashes).sort();
    const files: Record<string, string> = {};
    for (const relPath of filePaths) {
      files[relPath] = await slot.store.readFileTextOrDataUrl(relPath);
    }
    const messages = ContextBuilder.buildMessages(journal, {
      sessionId: slot.activeSessionId,
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
    const allWorkspaces = await slot.store.listWorkspaces();
    const workspaces = allWorkspaces.map((w) => ({
      workspaceId: w.workspaceId,
      title: w.title,
      headSeq: w.headSeq,
      updatedAtMs: w.updatedAtMs,
      runState: w.runState,
    }));
    return {
      workspaceId: meta.workspaceId,
      title: meta.title,
      headSeq: meta.headSeq,
      activeRunId: visible?.runId ?? null,
      runState: visible?.committedState ?? null,
      updatedAtMs: meta.updatedAtMs,
      leaseMode: slot.store.getLeaseMode(),
      gatewayInstanceId: slot.runtime.currentGatewayInstanceId,
      interruptReason,
      files,
      fileHashes,
      messages,
      pendingApproval,
      journal,
      workspaces,
      sessions: slot.agentManager.listSessions(),
      activeSessionId: slot.activeSessionId,
      serviceConfig: slot.runtime.getServiceConfig(),
    };
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots.values()) {
      slot.disposeListeners();
      if (this.isSlotIdle(slot)) {
        void slot.store.releaseWorkspaceLease();
      } else {
        void this.cancelInFlightRunsInSlot(slot)
          .catch(() => {});
      }
    }
    this.slots.clear();
    this.detachPort?.();
    this.detachPort = null;
  }
}
