/**
 * @module SYS-UI/app-client
 * Framework-agnostic browser UI client communicating with the Web Worker Kernel
 * strictly over `WorkerPortLike` (`MessagePort` / `Worker`).
 *
 * Enforces !INVARIANT [Coupling_Cohesion_Law] (Data Coupling):
 * - Zero imports from `src/kernel/**`, `IndexedDB`, or `OPFS`.
 * - Coordinates the First Useful Release Flow: workspace init/load, provider/model
 *   selection, prompt execution, interactive tool approval (`RESOLVE_APPROVAL`),
 *   isolated `<iframe sandbox="allow-scripts">` preview mounting, and archive export/import.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import {
  DEFAULT_SERVICE_CAPABILITIES_CONFIG,
  type AgentSessionSpec,
  type GatewayInstanceId,
  type InterruptReason,
  type ProviderModelConfigSpec,
  type RunId,
  type RunState,
  type RunJournalRecord,
  type ServiceCapabilitiesConfigSpec,
  type UiToWorkerMessage,
  type UiToWorkerRequestPayload,
  type WorkerPortLike,
  type WorkerToUiMessage,
  type WorkspaceStateSnapshot,
} from "../protocol/types.ts";
import type { PreviewSandboxHost } from "./preview-sandbox.ts";

export interface AppClientStateView {
  readonly snapshot: WorkspaceStateSnapshot | null;
  readonly selectedProvider: ProviderModelConfigSpec;
  readonly serviceConfig: ServiceCapabilitiesConfigSpec;
  readonly streamingDraftText: string;
  readonly lastError: string | null;
  readonly pendingUserTurn: PendingUserTurn | null;
  readonly inFlightRunId: RunId | null;
  readonly stoppingRunId: RunId | null;
}

export interface PendingUserTurn {
  readonly runId: RunId;
  readonly sessionId: string;
  readonly prompt: string;
  readonly status: "sending" | "failed";
}

export function isActiveRun(state: RunState | null | undefined): boolean {
  return state != null && ["created", "preparing", "inferring", "awaiting_approval", "awaiting_tool", "executing_tool"].includes(state);
}

interface PendingRpc {
  readonly resolve: (msg: Extract<WorkerToUiMessage, { kind: "RESPONSE" }>) => void;
  readonly reject: (err: Error) => void;
}

function bindClientPortListener(
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

function isWorkerToUiMessage(raw: unknown): raw is WorkerToUiMessage {
  if (typeof raw !== "object" || raw === null) return false;
  const obj = raw as Record<string, unknown>;
  return (
    obj.channel === "veles:worker-port" &&
    (obj.kind === "RESPONSE" || obj.kind === "EVENT")
  );
}

export class VelesAppClient {
  private readonly port: WorkerPortLike;
  private detachPort: (() => void) | null = null;
  private readonly pendingRpcs = new Map<string, PendingRpc>();
  private readonly listeners = new Set<(view: AppClientStateView) => void>();
  private snapshot: WorkspaceStateSnapshot | null = null;
  private selectedProvider: ProviderModelConfigSpec;
  private serviceConfig: ServiceCapabilitiesConfigSpec;
  private readonly draftByWorkspace = new Map<string, string>();
  private readonly inFlightByWorkspace = new Map<string, RunId>();
  private readonly stoppingByWorkspace = new Map<string, RunId>();
  private readonly pendingTurnByWorkspace = new Map<string, PendingUserTurn>();
  private lastError: string | null = null;
  private viewRevision = 0;
  private materializedHeadSeq = -1;
  private rpcCounter = 0;
  private disposed = false;

  private get wsKey(): string {
    return this.snapshot?.workspaceId ?? "__default__";
  }

  private get streamingDraftText(): string {
    return this.draftByWorkspace.get(this.wsKey) ?? "";
  }

  private set streamingDraftText(val: string) {
    if (val) this.draftByWorkspace.set(this.wsKey, val);
    else this.draftByWorkspace.delete(this.wsKey);
  }

  private get inFlightRunId(): RunId | null {
    return this.inFlightByWorkspace.get(this.wsKey) ?? null;
  }

  private set inFlightRunId(val: RunId | null) {
    if (val) this.inFlightByWorkspace.set(this.wsKey, val);
    else this.inFlightByWorkspace.delete(this.wsKey);
  }

  private get stoppingRunId(): RunId | null {
    return this.stoppingByWorkspace.get(this.wsKey) ?? null;
  }

  private set stoppingRunId(val: RunId | null) {
    if (val) this.stoppingByWorkspace.set(this.wsKey, val);
    else this.stoppingByWorkspace.delete(this.wsKey);
  }

  private get pendingUserTurn(): PendingUserTurn | null {
    return this.pendingTurnByWorkspace.get(this.wsKey) ?? null;
  }

  private set pendingUserTurn(val: PendingUserTurn | null) {
    if (val) this.pendingTurnByWorkspace.set(this.wsKey, val);
    else this.pendingTurnByWorkspace.delete(this.wsKey);
  }

  constructor(
    port: WorkerPortLike,
    initialProvider: ProviderModelConfigSpec = {
      providerId: "openai",
      baseUrl: "https://api.openai.com/v1",
      model: "gpt-4o-mini",
    },
    initialServiceConfig: ServiceCapabilitiesConfigSpec = DEFAULT_SERVICE_CAPABILITIES_CONFIG,
  ) {
    this.port = port;
    this.selectedProvider = { ...initialProvider };
    this.serviceConfig = {
      ...initialServiceConfig,
      decision: { ...initialServiceConfig.decision },
    };
    this.detachPort = bindClientPortListener(this.port, (ev) => {
      this.handleWorkerMessage(ev.data);
    });
  }

  public get isDisposed(): boolean {
    return this.disposed;
  }

  public getStateView(): AppClientStateView {
    return {
      snapshot: this.snapshot,
      selectedProvider: { ...this.selectedProvider },
      serviceConfig: {
        ...this.serviceConfig,
        decision: { ...this.serviceConfig.decision },
      },
      streamingDraftText: this.streamingDraftText,
      lastError: this.lastError,
      pendingUserTurn: this.pendingUserTurn,
      inFlightRunId: this.inFlightRunId,
      stoppingRunId: this.stoppingRunId,
    };
  }

  public onStateChange(listener: (view: AppClientStateView) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notifyListeners(): void {
    if (this.disposed) return;
    const view = this.getStateView();
    for (const listener of this.listeners) {
      listener(view);
    }
  }

  public selectProviderModel(config: ProviderModelConfigSpec): void {
    this.assertNotDisposed();
    this.selectedProvider = { ...config };
    this.notifyListeners();
  }

  public selectServiceConfig(config: ServiceCapabilitiesConfigSpec): void {
    this.assertNotDisposed();
    this.serviceConfig = {
      ...config,
      decision: { ...config.decision },
    };
    this.notifyListeners();
    void this.sendRequest({
      type: "UPDATE_SERVICE_CONFIG",
      serviceConfig: this.serviceConfig,
    }).catch(() => {});
  }

  public async updateServiceConfig(
    config: ServiceCapabilitiesConfigSpec,
  ): Promise<WorkspaceStateSnapshot> {
    this.assertNotDisposed();
    this.serviceConfig = {
      ...config,
      decision: { ...config.decision },
    };
    return this.requestWorkspaceSnapshot({
      type: "UPDATE_SERVICE_CONFIG",
      serviceConfig: this.serviceConfig,
    });
  }

  public async initWorkspace(title: string): Promise<WorkspaceStateSnapshot> {
    return this.requestWorkspaceSnapshot({ type: "INIT_WORKSPACE", title });
  }

  public async refreshSnapshot(): Promise<WorkspaceStateSnapshot> {
    return this.requestWorkspaceSnapshot({ type: "GET_SNAPSHOT" });
  }

  public async createSession(
    session: AgentSessionSpec,
  ): Promise<WorkspaceStateSnapshot> {
    return this.requestWorkspaceSnapshot({
      type: "CREATE_SESSION",
      session,
    });
  }

  public async startRun(
    runId: RunId,
    prompt: string,
    overrideConfig?: ProviderModelConfigSpec,
    sessionId?: string,
  ): Promise<WorkspaceStateSnapshot> {
    if (this.inFlightRunId || isActiveRun(this.snapshot?.runState)) throw new Error("ERR_RUN_ALREADY_ACTIVE: Stop the current run before sending another message");
    const originWsKey = this.wsKey;
    const previousSnapshot = this.snapshot;
    const effectiveSessionId = sessionId ?? this.snapshot?.activeSessionId ?? "session-main";
    this.pendingTurnByWorkspace.set(originWsKey, { runId, sessionId: effectiveSessionId, prompt, status: "sending" });
    this.draftByWorkspace.delete(originWsKey);
    this.lastError = null;
    this.inFlightByWorkspace.set(originWsKey, runId);
    if (this.snapshot) {
      this.snapshot = {
        ...this.snapshot,
        activeRunId: runId,
        runState: "preparing",
        pendingApproval: null,
      };
      this.notifyListeners();
    }
    const providerConfig = overrideConfig
      ? { ...overrideConfig }
      : { ...this.selectedProvider };
    try {
      const res = await this.sendRequest({
        type: "START_RUN",
        runId,
        prompt,
        providerConfig,
        sessionId: effectiveSessionId,
        serviceConfig: {
          ...this.serviceConfig,
          decision: { ...this.serviceConfig.decision },
        },
      });
      this.draftByWorkspace.delete(originWsKey);
      this.pendingTurnByWorkspace.delete(originWsKey);
      if (this.isBackgroundWorkspaceResponse(res)) {
        return res.snapshot!;
      }
      return this.applySnapshotFromResponse(res);
    } catch (error) {
      const pending = this.pendingTurnByWorkspace.get(originWsKey);
      if (pending?.runId === runId) {
        this.pendingTurnByWorkspace.set(originWsKey, { ...pending, status: "failed" });
        if (this.wsKey === originWsKey) {
          this.snapshot = previousSnapshot;
        }
      }
      throw error;
    } finally {
      if (this.inFlightByWorkspace.get(originWsKey) === runId) {
        this.inFlightByWorkspace.delete(originWsKey);
      }
      this.notifyListeners();
    }
  }

  private isBackgroundWorkspaceResponse(
    res: Extract<WorkerToUiMessage, { kind: "RESPONSE" }>,
  ): boolean {
    if (!this.snapshot || !res.snapshot) return false;
    if (this.snapshot.workspaceId === res.snapshot.workspaceId) return false;
    if (res.snapshot.workspaces) {
      this.snapshot = { ...this.snapshot, workspaces: res.snapshot.workspaces };
    }
    return true;
  }

  public async switchChatSession(
    sessionId: string,
    title?: string,
  ): Promise<WorkspaceStateSnapshot> {
    this.viewRevision += 1;
    this.pendingUserTurn = null;
    this.streamingDraftText = "";
    this.lastError = null;
    return this.requestWorkspaceSnapshot({
      type: "SWITCH_SESSION",
      sessionId,
      title,
    });
  }

  public async deleteChatRun(
    runId: RunId,
    options?: { readonly revertFiles?: boolean },
  ): Promise<WorkspaceStateSnapshot> {
    this.streamingDraftText = "";
    this.lastError = null;
    return this.requestWorkspaceSnapshot({
      type: "DELETE_RUN",
      runId,
      ...(options?.revertFiles ? { revertFiles: true } : {}),
    });
  }

  public async cancelRun(runId: RunId): Promise<WorkspaceStateSnapshot> {
    const revision = this.viewRevision;
    const originWsKey = this.wsKey;
    this.stoppingRunId = runId;
    this.streamingDraftText = "";
    this.notifyListeners();
    try {
      const res = await this.sendRequest({ type: "CANCEL_RUN", runId });
      if (revision !== this.viewRevision) return this.snapshot!;
      return this.applySnapshotFromResponse(res);
    } finally {
      if (this.stoppingByWorkspace.get(originWsKey) === runId) {
        this.stoppingByWorkspace.delete(originWsKey);
      }
      this.notifyListeners();
    }
  }

  public async resolveApproval(
    runId: RunId,
    approved: boolean,
  ): Promise<WorkspaceStateSnapshot> {
    const revision = this.viewRevision;
    const originWsKey = this.wsKey;
    this.lastError = null;
    this.inFlightRunId = runId;
    if (this.snapshot) {
      this.snapshot = {
        ...this.snapshot,
        pendingApproval: null,
        runState: approved ? "executing_tool" : "inferring",
      };
      this.notifyListeners();
    }
    try {
      const res = await this.sendRequest({
        type: "RESOLVE_APPROVAL",
        runId,
        approved,
      });
      this.draftByWorkspace.delete(originWsKey);
      if (revision !== this.viewRevision) return this.snapshot!;
      return this.applySnapshotFromResponse(res);
    } finally {
      if (this.inFlightByWorkspace.get(originWsKey) === runId) {
        this.inFlightByWorkspace.delete(originWsKey);
      }
      this.notifyListeners();
    }
  }

  public async resolveUncertainTool(
    runId: RunId,
    resolution: "mark_succeeded" | "retry_anyway" | "cancel_run",
  ): Promise<WorkspaceStateSnapshot> {
    const revision = this.viewRevision;
    const originWsKey = this.wsKey;
    this.lastError = null;
    this.inFlightRunId = runId;
    if (this.snapshot) {
      this.snapshot = {
        ...this.snapshot,
        interruptReason: null,
        runState: resolution === "cancel_run" ? "cancelled" : "inferring",
      };
      this.notifyListeners();
    }
    try {
      const res = await this.sendRequest({
        type: "RESOLVE_UNCERTAIN_TOOL",
        runId,
        resolution,
      });
      this.draftByWorkspace.delete(originWsKey);
      if (revision !== this.viewRevision) return this.snapshot!;
      return this.applySnapshotFromResponse(res);
    } finally {
      if (this.inFlightByWorkspace.get(originWsKey) === runId) {
        this.inFlightByWorkspace.delete(originWsKey);
      }
      this.notifyListeners();
    }
  }

  public async simulateChaos(
    scenario:
      | "quota_exceeded"
      | "drop_stream_mid_inference"
      | "uncertain_tool_outcome",
  ): Promise<WorkspaceStateSnapshot> {
    const res = await this.sendRequest({
      type: "SIMULATE_CHAOS",
      scenario,
    });
    this.streamingDraftText = "";
    return this.applySnapshotFromResponse(res);
  }

  public async recoverBoot(
    currentGatewayInstanceId: GatewayInstanceId | null,
  ): Promise<{
    readonly decision: {
      readonly nextState: RunState;
      readonly interruptReason?: InterruptReason;
      readonly canAutoRetry: boolean;
      readonly queryGatewayStatus: boolean;
    } | null;
    readonly snapshot: WorkspaceStateSnapshot;
  }> {
    const res = await this.sendRequest({
      type: "RECOVER_BOOT",
      currentGatewayInstanceId,
    });
    this.streamingDraftText = "";
    const snapshot = this.applySnapshotFromResponse(res);
    return {
      decision: res.recoveryDecision ?? null,
      snapshot,
    };
  }

  public async exportWorkspace(): Promise<unknown> {
    const res = await this.sendRequest({ type: "EXPORT_BUNDLE" });
    return res.bundle;
  }

  public async importWorkspace(bundle: unknown): Promise<WorkspaceStateSnapshot> {
    this.viewRevision += 1;
    const revision = this.viewRevision;
    const originWsKey = this.wsKey;
    this.pendingUserTurn = null;
    this.streamingDraftText = "";
    const res = await this.sendRequest({ type: "IMPORT_BUNDLE", bundle });
    if (revision !== this.viewRevision || originWsKey !== this.wsKey) return res.snapshot!;
    // An explicit restore can intentionally replace this workspace with an older journal.
    this.materializedHeadSeq = -1;
    this.snapshot = null;
    return this.applySnapshotFromResponse(res);
  }

  public async switchWorkspace(
    workspaceId: string,
    title?: string,
  ): Promise<WorkspaceStateSnapshot> {
    this.viewRevision += 1;
    this.lastError = null;
    const res = await this.sendRequest({
      type: "SWITCH_WORKSPACE",
      workspaceId,
      title,
    });
    return this.applySnapshotFromResponse(res);
  }

  public async deleteWorkspace(
    workspaceId: string,
  ): Promise<WorkspaceStateSnapshot> {
    this.viewRevision += 1;
    this.draftByWorkspace.delete(workspaceId);
    this.inFlightByWorkspace.delete(workspaceId);
    this.stoppingByWorkspace.delete(workspaceId);
    this.pendingTurnByWorkspace.delete(workspaceId);
    this.lastError = null;
    const res = await this.sendRequest({
      type: "DELETE_WORKSPACE",
      workspaceId,
    });
    return this.applySnapshotFromResponse(res);
  }

  public async uploadWorkspaceFile(
    path: string,
    content: string | Uint8Array,
  ): Promise<WorkspaceStateSnapshot> {
    return this.requestWorkspaceSnapshot({
      type: "WRITE_WORKSPACE_FILE",
      path,
      content,
    });
  }

  /**
   * Mounts the committed workspace files into the isolated `<iframe sandbox="allow-scripts">`
   * preview host (`!INVARIANT [SEC-INV-01]`).
   */
  public mountWorkspacePreview(
    sandboxHost: PreviewSandboxHost,
    entryHtmlPath = "index.html",
  ): number {
    this.assertNotDisposed();
    if (!this.snapshot) {
      throw new Error("ERR_NO_WORKSPACE_SNAPSHOT: Initialize or refresh workspace first");
    }
    const files = this.snapshot.files;
    const isPreviewablePath = (p: string) =>
      /\.(html|svg|png|jpe?g|webp|gif)$/i.test(p);
    const resolvedPath =
      files[entryHtmlPath] !== undefined
        ? entryHtmlPath
        : files["preview/index.html"] !== undefined
          ? "preview/index.html"
          : files["index.html"] !== undefined
            ? "index.html"
            : Object.keys(files).find(isPreviewablePath) ?? entryHtmlPath;
    const entryHtml =
      files[resolvedPath] ??
      "<!doctype html><html><body><div id=\"app\">Veles Preview</div></body></html>";
    return sandboxHost.mountPreview({
      workspaceId: this.snapshot.workspaceId,
      entryHtml,
      entryPath: resolvedPath,
      workspaceSeq: this.snapshot.headSeq,
      files,
    });
  }

  private async requestWorkspaceSnapshot(request: UiToWorkerRequestPayload): Promise<WorkspaceStateSnapshot> {
    const originWsKey = this.wsKey;
    const revision = this.viewRevision;
    const response = await this.sendRequest(request);
    if (!response.snapshot) throw new Error("ERR_MISSING_SNAPSHOT_IN_WORKER_RESPONSE");
    if (originWsKey !== this.wsKey || revision !== this.viewRevision) return response.snapshot;
    return this.applySnapshotFromResponse(response);
  }

  private applySnapshotFromResponse(
    res: Extract<WorkerToUiMessage, { kind: "RESPONSE" }>,
  ): WorkspaceStateSnapshot {
    if (!res.snapshot) {
      throw new Error("ERR_MISSING_SNAPSHOT_IN_WORKER_RESPONSE");
    }
    const current = this.snapshot;
    const next = res.snapshot;
    const sameWorkspace = current?.workspaceId === next.workspaceId;
    if (sameWorkspace && next.headSeq < this.materializedHeadSeq) return current;
    this.materializedHeadSeq = next.headSeq;
    const tail = sameWorkspace ? current.journal.filter((record) => record.seq > next.headSeq) : [];
    this.snapshot = tail.length ? {
      ...next, journal: [...next.journal, ...tail], headSeq: tail.at(-1)!.seq,
      ...(next.activeRunId === current?.activeRunId ? {
        runState: current.runState, interruptReason: current.interruptReason,
      } : {}),
    } : next;
    this.notifyListeners();
    return this.snapshot;
  }

  private sendRequest(
    request: UiToWorkerRequestPayload,
  ): Promise<Extract<WorkerToUiMessage, { kind: "RESPONSE" }>> {
    this.assertNotDisposed();
    this.rpcCounter += 1;
    const requestId = `rpc-${this.rpcCounter}`;
    const envelope: UiToWorkerMessage = {
      channel: "veles:worker-port",
      requestId,
      request,
    };

    return new Promise((resolve, reject) => {
      this.pendingRpcs.set(requestId, { resolve, reject });
      this.port.postMessage(envelope);
    });
  }

  private handleWorkerMessage(raw: unknown): void {
    if (this.disposed || !isWorkerToUiMessage(raw)) return;
    if (raw.kind === "RESPONSE") {
      const pending = this.pendingRpcs.get(raw.requestId);
      if (!pending) return;
      this.pendingRpcs.delete(raw.requestId);
      if (!raw.ok) {
        const errText = raw.error ?? "ERR_WORKER_RPC_FAILED";
        this.lastError = errText;
        this.notifyListeners();
        pending.reject(new Error(errText));
        return;
      }
      pending.resolve(raw);
      return;
    }

    this.applyWorkerEvent(raw);
    this.notifyListeners();
  }

  private applyWorkerEvent(
    raw: Extract<WorkerToUiMessage, { kind: "EVENT" }>,
  ): void {
    if (raw.eventType === "JOURNAL_COMMITTED" && raw.record) {
      this.applyCommittedRecord(raw.record);
      return;
    }
    if (raw.eventType === "STREAM_DELTA") {
      if (raw.runId !== this.inFlightRunId || this.stoppingRunId === raw.runId || !isActiveRun(this.snapshot?.runState)) return;
      const delta =
        typeof raw.detail?.deltaText === "string" ? raw.detail.deltaText : "";
      this.streamingDraftText += delta;
      return;
    }
    if (raw.eventType === "STATE_CHANGED" && this.snapshot && raw.state) {
      const targetRunId = this.inFlightRunId ?? this.snapshot.activeRunId;
      if (
        (raw.runId && targetRunId && raw.runId !== targetRunId) ||
        (this.inFlightRunId !== null && raw.state === "completed") ||
        (!isActiveRun(this.snapshot.runState) && isActiveRun(raw.state))
      ) {
        return;
      }
      this.snapshot = {
        ...this.snapshot,
        runState: raw.state,
        ...(raw.state !== "awaiting_approval" ? { pendingApproval: null } : {}),
      };
    }
  }

  private applyCommittedRecord(record: RunJournalRecord): void {
    if (
      record.type === "RunCreated" &&
      this.pendingTurnByWorkspace.get(record.workspaceId)?.runId === record.runId
    ) {
      this.pendingTurnByWorkspace.delete(record.workspaceId);
    }
    if (
      [
        "ToolIntentLogged",
        "ApprovalRequested",
        "RunCompleted",
        "RunCancelled",
        "RunInterrupted",
      ].includes(record.type)
    ) {
      this.draftByWorkspace.delete(record.workspaceId);
    }
    const snap = this.snapshot;
    if (!snap) return;
    if (record.workspaceId !== snap.workspaceId) {
      this.updateBackgroundWorkspaceSummary(snap, record);
      return;
    }
    if (record.seq <= snap.headSeq) return;
    const targetRunId = this.inFlightRunId ?? snap.activeRunId;
    const isCurrent = record.runId === targetRunId;
    const reason = record.payload.interruptReason;
    const interruptReason = reason === "uncertain_tool_outcome" || reason === "user_reload_or_tab_close" || reason === "gateway_disconnected_safe_to_retry" ? reason : null;
    this.snapshot = {
      ...snap, journal: [...snap.journal, record], headSeq: record.seq, updatedAtMs: record.timestampMs,
      ...(isCurrent ? {
        runState: record.committedState, interruptReason,
        ...(record.committedState !== "awaiting_approval" ? { pendingApproval: null } : {}),
      } : {}),
    };
  }

  private updateBackgroundWorkspaceSummary(
    snap: WorkspaceStateSnapshot,
    record: RunJournalRecord,
  ): void {
    if (!snap.workspaces?.some((w) => w.workspaceId === record.workspaceId)) {
      return;
    }
    this.snapshot = {
      ...snap,
      workspaces: snap.workspaces.map((w) =>
        w.workspaceId === record.workspaceId
          ? {
              ...w,
              headSeq: Math.max(w.headSeq, record.seq),
              updatedAtMs: record.timestampMs,
              runState: record.committedState,
            }
          : w,
      ),
    };
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error("ERR_CLIENT_DISPOSED: VelesAppClient has been disposed");
    }
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.detachPort?.();
    this.detachPort = null;
    for (const [, pending] of this.pendingRpcs) {
      pending.reject(new Error("ERR_CLIENT_DISPOSED"));
    }
    this.pendingRpcs.clear();
    this.listeners.clear();
  }
}
