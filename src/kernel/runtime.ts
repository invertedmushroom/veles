/**
 * @module SYS-KERNEL/runtime
 * Web Worker Agent Runtime owning the COMET state loop, deterministic crash recovery
 * (`resolveUnfinishedRunTail`), interactive tool approval resolution (`resolveApproval`),
 * and `ContextBuilder` (reconstructing LLM context strictly from committed `RunJournalRecord[]`).
 * Enforces !INVARIANT [KERN-INV-01], [KERN-INV-02], and Scenarios 1, 2, 3, 4, 9.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { awaitWithAbort } from "./abort.ts";
import { isLifecycleJournalRecord, uncertainRunIds } from "./run-cancellation.ts";
import {
  type ChatMessage,
  type InferenceAdapter,
  type ProviderModelConfig,
  type TokenUsage,
  type ToolCallSpec,
  type ToolDefinitionSpec,
} from "../inference/openai-compatible.ts";
import {
  DEFAULT_SERVICE_CAPABILITIES_CONFIG,
  type GatewayInstanceId,
  type OperationId,
  type RunId,
  type RunJournalRecord,
  type RunState,
  type ServiceCapabilitiesConfigSpec,
  type ToolEffectClass,
} from "../protocol/types.ts";
import {
  type CrashRecoveryDecision,
  resolveUnfinishedRunTail,
} from "../protocol/validation.ts";
import {
  createDecisionClassifierAdapter,
  VelesHarnessRouter,
  type InBrowserSpecialistProfile,
  type OptionalClassifierAdapter,
} from "./harness-router.ts";
import type { WorkspaceKernelStore } from "./journal.ts";

export interface ToolExecutionRequest {
  readonly runId: RunId;
  readonly operationId: OperationId;
  readonly toolName: string;
  readonly effectClass: ToolEffectClass;
  readonly args: Record<string, unknown>;
  readonly signal?: AbortSignal;
}

export interface ToolExecutionOutcome {
  readonly gatewayInstanceId: GatewayInstanceId;
  readonly outputText: string;
  readonly fileUpdates?: Readonly<Record<string, Uint8Array | string | null>>;
  readonly imageDataUrls?: readonly string[];
}

export interface ToolExecutionAdapter {
  readonly currentGatewayInstanceId: GatewayInstanceId | null;
  executeTool(request: ToolExecutionRequest): Promise<ToolExecutionOutcome>;
  queryOperationStatus?(
    operationId: OperationId,
    gatewayInstanceId: GatewayInstanceId,
  ): Promise<{ readonly completed: boolean; readonly outcome?: ToolExecutionOutcome }>;
}

export type WorkerCommand =
  | {
      readonly type: "START_RUN";
      readonly runId: RunId;
      readonly prompt: string;
      readonly providerConfig: ProviderModelConfig;
    }
  | {
      readonly type: "CANCEL_RUN";
      readonly runId: RunId;
    }
  | {
      readonly type: "RESOLVE_APPROVAL";
      readonly runId: RunId;
      readonly approved: boolean;
    }
  | {
      readonly type: "RECOVER_BOOT";
      readonly currentGatewayInstanceId: GatewayInstanceId | null;
    };

export interface WorkerRuntimeEvent {
  readonly type:
    | "STATE_CHANGED"
    | "STREAM_DELTA"
    | "RECOVERY_RESOLVED"
    | "ERROR";
  readonly runId: RunId | null;
  readonly state: RunState | null;
  readonly detail?: Record<string, unknown>;
}

export interface ContextBuildOptions {
  /** If set, isolates context strictly to a single runId (e.g. child subagent run). */
  readonly isolateToRunId?: RunId;
  /** If set, reconstructs context strictly from runs belonging to this AgentSession ID. */
  readonly sessionId?: string;
  /** Number of most-recent runs to keep with full uncompressed tool outputs (default: 2). */
  readonly hotRunWindow?: number;
  /** Current workspace file tree (path -> sha256) injected as Layer 1 system context. */
  readonly fileTree?: Readonly<Record<string, string>>;
  /** Matched Layer 2 in-browser specialist capsules injected into the system prompt. */
  readonly specialistCapsules?: readonly InBrowserSpecialistProfile[];
  /** Optional character budget (~4 chars per token; default 48_000 chars ~ 12k tokens). */
  readonly maxCharBudget?: number;
}

export interface StartRunOptions {
  readonly sessionId?: string;
  readonly parentSessionId?: string;
  readonly allowedTools?: readonly string[];
  readonly delegatableTools?: readonly string[];
  readonly unrestrictedTools?: boolean;
}

export interface AgentKernelRuntimeOptions {
  readonly multiTurnWorkspaceContext?: boolean;
  readonly hotRunWindow?: number;
  readonly router?: VelesHarnessRouter;
  readonly classifier?: OptionalClassifierAdapter;
}

/**
 * 4-Layer Tiered Context Compiler reconstructing LLM conversation messages strictly from
 * committed `RunJournalRecord[]` and OPFS file tree (`!INVARIANT [KERN-INV-01]`).
 * - Layer 1: Live OPFS Workspace Manifest (`preview/`, `assets/`, `scripts/`, `notes/`)
 * - Layer 2: In-Browser Specialist Capsules (`VelesHarnessRouter`)
 * - Layer 3: Folded Prior Runs (Chain-of-Draft compression + `sha256` drift annotations)
 * - Layer 4: Active Run & Last K=2 Hot Runs (uncompressed tool arguments and outputs)
 */
export class ContextBuilder {
  private static normalizeOptions(
    runIdOrOptions?: RunId | ContextBuildOptions,
  ): ContextBuildOptions {
    return typeof runIdOrOptions === "string"
      ? { isolateToRunId: runIdOrOptions }
      : (runIdOrOptions ?? {});
  }

  public static buildMessages(
    records: readonly RunJournalRecord[],
    runIdOrOptions?: RunId | ContextBuildOptions,
  ): ChatMessage[] {
    const opts = ContextBuilder.normalizeOptions(runIdOrOptions);
    const { runOrder, byRun } = ContextBuilder.groupRecordsByRun(records, opts);
    const hotRunIds = ContextBuilder.selectHotRunIds(
      runOrder,
      byRun,
      opts.hotRunWindow,
    );
    const messages: ChatMessage[] = [];

    const sysHeader = ContextBuilder.formatWorkspaceHeader(
      opts.fileTree,
      opts.specialistCapsules,
    );
    if (sysHeader !== null) {
      messages.push({ role: "system", content: sysHeader });
    }

    for (let i = 0; i < runOrder.length; i++) {
      const rId = runOrder[i]!;
      const recs = byRun.get(rId) ?? [];
      if (i < runOrder.length - 1 && ContextBuilder.isEmptyInterruptedRun(recs)) {
        continue;
      }
      const isHot = hotRunIds.has(rId);
      const committedCallIds = ContextBuilder.collectCommittedToolCallIds(recs);
      for (const rec of recs) {
        ContextBuilder.appendRecordMessage(
          messages,
          rec,
          isHot,
          opts.fileTree,
          committedCallIds,
        );
      }
    }

    if (opts.maxCharBudget !== undefined) {
      return ContextBuilder.enforceCharBudget(messages, opts.maxCharBudget);
    }
    return messages;
  }

  private static selectHotRunIds(
    runOrder: readonly RunId[],
    byRun: ReadonlyMap<RunId, readonly RunJournalRecord[]>,
    explicitHotWindow?: number,
  ): Set<RunId> {
    const windowSize = explicitHotWindow ?? 2;
    const hot = new Set<RunId>(runOrder.slice(-windowSize));
    if (windowSize <= 0) {
      return hot;
    }
    let substantiveRunsKept = 0;
    for (
      let i = runOrder.length - 1;
      i >= 0 && substantiveRunsKept < windowSize;
      i--
    ) {
      const rId = runOrder[i]!;
      const recs = byRun.get(rId) ?? [];
      const isLatestRun = i === runOrder.length - 1;
      const hasToolActivity = recs.some(
        (r) =>
          r.type === "ToolIntentLogged" ||
          r.type === "ToolResultCommitted" ||
          r.type === "ApprovalRequested" ||
          r.type === "ToolDenied",
      );
      if (isLatestRun || hasToolActivity) {
        hot.add(rId);
        substantiveRunsKept += 1;
      }
    }
    return hot;
  }

  private static groupRecordsByRun(
    records: readonly RunJournalRecord[],
    opts: ContextBuildOptions,
  ): { runOrder: RunId[]; byRun: Map<RunId, RunJournalRecord[]> } {
    const allowedRuns = ContextBuilder.resolveAllowedSessionRuns(records, opts);
    const runOrder: RunId[] = [];
    const byRun = new Map<RunId, RunJournalRecord[]>();
    for (const rec of records) {
      if (!allowedRuns.has(rec.runId)) continue;
      let list = byRun.get(rec.runId);
      if (!list) {
        list = [];
        byRun.set(rec.runId, list);
        runOrder.push(rec.runId);
      }
      list.push(rec);
    }
    return { runOrder, byRun };
  }

  private static collectDeletedRunIds(
    records: readonly RunJournalRecord[],
  ): Set<RunId> {
    const deleted = new Set<RunId>();
    for (const rec of records) {
      if (typeof rec.payload.deletedRunId === "string") {
        deleted.add(rec.payload.deletedRunId);
      }
    }
    return deleted;
  }

  private static isChildSubagentRun(rec: RunJournalRecord): boolean {
    return (
      typeof rec.payload.parentRunId === "string" ||
      typeof rec.payload.parentSessionId === "string"
    );
  }

  private static doesRunMatchSession(
    rec: RunJournalRecord,
    sessionId: string,
  ): boolean {
    if (rec.type !== "RunCreated") return false;
    if (sessionId === "session-main") {
      return (
        (rec.payload.sessionId === "session-main" ||
          rec.payload.sessionId === undefined) &&
        !ContextBuilder.isChildSubagentRun(rec)
      );
    }
    return rec.payload.sessionId === sessionId;
  }

  private static resolveAllowedSessionRuns(
    records: readonly RunJournalRecord[],
    opts: ContextBuildOptions,
  ): Set<RunId> {
    const deletedRuns = ContextBuilder.collectDeletedRunIds(records);
    if (opts.isolateToRunId !== undefined) {
      return deletedRuns.has(opts.isolateToRunId)
        ? new Set<RunId>()
        : new Set<RunId>([opts.isolateToRunId]);
    }
    const excludedRuns = new Set<RunId>(deletedRuns);
    const allowed = new Set<RunId>();
    for (const rec of records) {
      if (opts.sessionId !== undefined) {
        if (
          !excludedRuns.has(rec.runId) &&
          ContextBuilder.doesRunMatchSession(rec, opts.sessionId)
        ) {
          allowed.add(rec.runId);
        }
        continue;
      }
      if (rec.type === "RunCreated" && ContextBuilder.isChildSubagentRun(rec)) {
        excludedRuns.add(rec.runId);
        allowed.delete(rec.runId);
      } else if (!excludedRuns.has(rec.runId)) {
        allowed.add(rec.runId);
      }
    }
    return allowed;
  }

  private static formatWorkspaceHeader(
    fileTree?: Readonly<Record<string, string>>,
    capsules?: readonly InBrowserSpecialistProfile[],
  ): string | null {
    const hasCapsules = Boolean(capsules && capsules.length > 0);
    if (!fileTree && !hasCapsules) {
      return null;
    }
    const resolvedTree = fileTree ?? {};
    const paths = Object.keys(resolvedTree).sort();
    const treeLines =
      paths.length === 0
        ? "  (empty workspace)"
        : paths
            .map((p) => `  - ${p} [sha256:${resolvedTree[p]!.slice(0, 8)}]`)
            .join("\n");
    const lines = [
      "You are the Veles Browser-Native Agent.",
      "Workspace folders: `preview/` (live HTML/CSS/JS/SVG rendered in sandbox), `assets/` (images/media), `scripts/` (CLI/gateway scripts), `notes/` (plans/memory).",
      "Preview sandbox (`origin === \"null\"`): `localStorage`/`sessionStorage` are blocked; `window.veles` provides `kv.get(k)`, `kv.set(k, v)`, `kv.remove(k)`, `log(msg, detail?)`, and `sensors.latest` / `sensors.onUpdate(cb)`.",
      "kv.get(k) returns a value or null synchronously (do not use top-level await in classic <script> tags or call .then() on kv.get). Keep the real host bridge intact.",
      "Current Committed OPFS Workspace Tree:",
      treeLines,
    ];
    if (capsules && capsules.length > 0) {
      lines.push("", "--- Active In-Browser Specialist Guidance ---");
      for (const cap of capsules) {
        lines.push(cap.capsulePrompt);
      }
    }
    return lines.join("\n");
  }

  private static messageCharWeight(m: ChatMessage): number {
    const imgWeight =
      m.imageDataUrls?.reduce((s, u) => s + Math.min(u.length, 4000), 0) ?? 0;
    return m.content.length + imgWeight;
  }

  private static enforceCharBudget(
    messages: ChatMessage[],
    maxChars: number,
  ): ChatMessage[] {
    let total = messages.reduce(
      (sum, m) => sum + ContextBuilder.messageCharWeight(m),
      0,
    );
    if (total <= maxChars || messages.length <= 2) return messages;
    const startIdx = messages[0]?.role === "system" ? 1 : 0;
    const activeTurnStart = ContextBuilder.findLastUserTurnIndex(messages, startIdx);
    const result = [...messages];
    let limitIdx = activeTurnStart > startIdx ? activeTurnStart : result.length - 1;
    while (total > maxChars && startIdx < limitIdx) {
      const removed = result.splice(startIdx, 1)[0]!;
      total -= ContextBuilder.messageCharWeight(removed);
      limitIdx -= 1;
      while (startIdx < limitIdx && result[startIdx]?.role === "tool") {
        const orphanTool = result.splice(startIdx, 1)[0]!;
        total -= ContextBuilder.messageCharWeight(orphanTool);
        limitIdx -= 1;
      }
    }
    return result;
  }

  private static findLastUserTurnIndex(
    messages: readonly ChatMessage[],
    minIndex: number,
  ): number {
    for (let i = messages.length - 1; i >= minIndex; i--) {
      if (messages[i]?.role === "user" && !messages[i]?.imageDataUrls?.length) {
        return i;
      }
    }
    return messages.length - 1;
  }

  private static collectCommittedToolCallIds(
    recs: readonly RunJournalRecord[],
  ): Set<string> {
    const ids = new Set<string>();
    for (const rec of recs) {
      if (
        rec.type === "ToolResultCommitted" &&
        typeof rec.payload.toolCallId === "string"
      ) {
        ids.add(rec.payload.toolCallId);
      }
    }
    return ids;
  }

  private static isEmptyInterruptedRun(
    recs: readonly RunJournalRecord[],
  ): boolean {
    const tail = recs.at(-1);
    if (!tail || tail.type !== "RunInterrupted") return false;
    return !recs.some(
      (r) =>
        r.type === "ToolIntentLogged" ||
        r.type === "ToolDenied" ||
        r.type === "ToolResultCommitted" ||
        r.type === "RunCompleted" ||
        (r.type === "CheckpointCommitted" &&
          typeof r.payload.assistantText === "string" &&
          r.payload.assistantText.length > 0),
    );
  }

  private static appendRecordMessage(
    messages: ChatMessage[],
    rec: RunJournalRecord,
    isHotRun = true,
    liveFileTree?: Readonly<Record<string, string>>,
    committedCallIds?: ReadonlySet<string>,
  ): void {
    const p = rec.payload;
    if (rec.type === "RunCreated" && typeof p.prompt === "string") {
      messages.push({ role: "user", content: p.prompt, runId: rec.runId });
      return;
    }
    if (rec.type === "CheckpointCommitted" && typeof p.assistantText === "string") {
      messages.push({ role: "assistant", content: p.assistantText });
      return;
    }
    if (
      (rec.type === "ToolIntentLogged" || rec.type === "ToolDenied") &&
      typeof p.toolName === "string"
    ) {
      const rawArgs =
        typeof p.args === "object" && p.args !== null
          ? (p.args as Record<string, unknown>)
          : {};
      const compactArgs =
        !isHotRun &&
        typeof rawArgs.content === "string" &&
        rawArgs.content.length > 120
          ? {
              ...rawArgs,
              content: `<folded ${rawArgs.content.length} chars; read from OPFS if needed>`,
            }
          : rawArgs;
      ContextBuilder.appendToolIntentOrDenied(
        messages,
        rec.type,
        p,
        p.toolName,
        compactArgs,
        committedCallIds,
      );
      return;
    }
    if (rec.type === "ToolResultCommitted" && typeof p.outputText === "string") {
      ContextBuilder.appendToolResultRecord(messages, p, isHotRun, liveFileTree);
      return;
    }
    if (rec.type === "RunCompleted" && typeof p.assistantText === "string") {
      messages.push({ role: "assistant", content: p.assistantText });
    }
  }

  private static buildDriftAnnotation(
    observedHashes: unknown,
    liveFileTree?: Readonly<Record<string, string>>,
  ): string {
    if (
      !liveFileTree ||
      typeof observedHashes !== "object" ||
      observedHashes === null
    ) {
      return "";
    }
    const notes: string[] = [];
    for (const [path, oldHash] of Object.entries(
      observedHashes as Record<string, unknown>,
    )) {
      if (typeof oldHash !== "string") continue;
      const currentHash = liveFileTree[path];
      if (currentHash === undefined) {
        notes.push(
          `[DRIFT NOTE: '${path}' has been deleted from OPFS since this turn]`,
        );
      } else if (currentHash !== oldHash) {
        notes.push(
          `[DRIFT NOTE: '${path}' has changed in OPFS since this turn (current sha256:${currentHash.slice(0, 8)}); call workspace.read_file before editing]`,
        );
      }
    }
    return notes.length > 0 ? `\n${notes.join("\n")}` : "";
  }

  private static appendToolResultRecord(
    messages: ChatMessage[],
    p: Record<string, unknown>,
    isHotRun: boolean,
    liveFileTree?: Readonly<Record<string, string>>,
  ): void {
    const rawText = String(p.outputText ?? "");
    const text =
      !isHotRun && rawText.length > 200
        ? `${rawText.slice(0, 120)}... [folded ${rawText.length} chars]`
        : rawText;
    const driftNote = ContextBuilder.buildDriftAnnotation(
      p.observedFileHashes,
      liveFileTree,
    );
    messages.push({
      role: "tool",
      content: `${text}${driftNote}`,
      toolCallId: typeof p.toolCallId === "string" ? p.toolCallId : undefined,
    });
    if (!isHotRun || !Array.isArray(p.imageDataUrls) || p.imageDataUrls.length === 0) {
      return;
    }
    const validUrls = p.imageDataUrls.filter(
      (u): u is string => typeof u === "string" && u.startsWith("data:image/"),
    );
    if (validUrls.length > 0) {
      messages.push({
        role: "user",
        content: `[Attached workspace image for tool call ${String(p.toolCallId ?? "")}]`,
        imageDataUrls: validUrls,
      });
    }
  }

  private static appendToolIntentOrDenied(
    messages: ChatMessage[],
    recordType: "ToolIntentLogged" | "ToolDenied",
    p: Record<string, unknown>,
    toolName: string,
    argsObj?: Record<string, unknown>,
    committedCallIds?: ReadonlySet<string>,
  ): void {
    const callId = typeof p.toolCallId === "string" ? p.toolCallId : "call_0";
    messages.push({
      role: "assistant",
      content: typeof p.assistantText === "string" ? p.assistantText : "",
      toolCalls: [
        {
          id: callId,
          name: toolName,
          argumentsJson: JSON.stringify(argsObj ?? p.args ?? {}),
        },
      ],
    });
    if (recordType === "ToolDenied") {
      messages.push({
        role: "tool",
        toolCallId: callId,
        content: "USER_APPROVAL_DENIED: Action was rejected by the user.",
      });
    } else if (committedCallIds && !committedCallIds.has(callId)) {
      messages.push({
        role: "tool",
        toolCallId: callId,
        content:
          "TOOL_CALL_INTERRUPTED: Tool execution was stopped or interrupted before completion.",
      });
    }
  }
}

function isNonTerminalState(state: RunState | null): boolean {
  return (
    state !== null &&
    state !== "completed" &&
    state !== "cancelled" &&
    state !== "failed" &&
    state !== "interrupted"
  );
}

function parseToolArgs(rawJson: string): Record<string, unknown> {
  if (!rawJson.trim()) return {};
  try {
    const parsed: unknown = JSON.parse(rawJson);
    return typeof parsed === "object" && parsed !== null
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

export function validateWorkerCommand(input: unknown): WorkerCommand {
  if (typeof input !== "object" || input === null) {
    throw new Error("ERR_INVALID_WORKER_COMMAND: Command must be an object");
  }
  const cmd = input as Record<string, unknown>;
  if (cmd.type === "START_RUN") {
    if (
      typeof cmd.runId !== "string" ||
      !cmd.runId ||
      typeof cmd.prompt !== "string" ||
      typeof cmd.providerConfig !== "object" ||
      cmd.providerConfig === null
    ) {
      throw new Error("ERR_INVALID_WORKER_COMMAND: Invalid START_RUN payload");
    }
    return cmd as unknown as WorkerCommand;
  }
  if (cmd.type === "CANCEL_RUN" && typeof cmd.runId === "string" && cmd.runId) {
    return cmd as unknown as WorkerCommand;
  }
  if (
    cmd.type === "RESOLVE_APPROVAL" &&
    typeof cmd.runId === "string" &&
    !cmd.runId.startsWith(" ") &&
    cmd.runId.length > 0 &&
    typeof cmd.approved === "boolean"
  ) {
    return cmd as unknown as WorkerCommand;
  }
  if (cmd.type === "RECOVER_BOOT") {
    return cmd as unknown as WorkerCommand;
  }
  throw new Error(`ERR_INVALID_WORKER_COMMAND: Unknown type ${String(cmd.type)}`);
}

function parseRecordedUsage(raw: unknown): TokenUsage | null {
  if (typeof raw !== "object" || raw === null) return null;
  const value = raw as Record<string, unknown>;
  const counts = [value.promptTokens, value.completionTokens, value.totalTokens];
  if (!counts.every((count) => typeof count === "number" && Number.isFinite(count) && count >= 0)) {
    return null;
  }
  return {
    promptTokens: value.promptTokens as number,
    completionTokens: value.completionTokens as number,
    totalTokens: value.totalTokens as number,
  };
}

function findUnfinishedRunTail(
  activeRunId: RunId | null,
  journal: readonly RunJournalRecord[],
): RunJournalRecord | null {
  const tailsByRun = new Map<RunId, RunJournalRecord>();
  for (const r of journal) {
    if (isLifecycleJournalRecord(r)) {
      tailsByRun.set(r.runId, r);
    }
  }
  const activeTail = activeRunId ? tailsByRun.get(activeRunId) : undefined;
  if (activeTail && isNonTerminalState(activeTail.committedState)) {
    return activeTail;
  }
  let latestUnfinished: RunJournalRecord | null = null;
  for (const tail of tailsByRun.values()) {
    if (isNonTerminalState(tail.committedState)) {
      latestUnfinished = tail;
    }
  }
  return latestUnfinished;
}

export class AgentKernelRuntime {
  private readonly store: WorkspaceKernelStore;
  private readonly inference: InferenceAdapter;
  private readonly toolAdapter: ToolExecutionAdapter;
  private readonly tools = new Map<string, ToolDefinitionSpec>();
  private readonly runConfigs = new Map<RunId, ProviderModelConfig>();
  private readonly runOptions = new Map<RunId, StartRunOptions>();
  private readonly listeners = new Set<(ev: WorkerRuntimeEvent) => void>();
  private readonly options: AgentKernelRuntimeOptions;
  private readonly router: VelesHarnessRouter;
  private readonly activeAborts = new Map<RunId, AbortController>();
  private readonly stopCommits = new Map<RunId, Promise<void>>();
  private serviceConfig: ServiceCapabilitiesConfigSpec =
    DEFAULT_SERVICE_CAPABILITIES_CONFIG;
  private eventCounter = 0;

  constructor(
    store: WorkspaceKernelStore,
    inference: InferenceAdapter,
    toolAdapter: ToolExecutionAdapter,
    toolDefinitions: readonly ToolDefinitionSpec[] = [],
    options: AgentKernelRuntimeOptions = {},
  ) {
    this.store = store;
    this.inference = inference;
    this.toolAdapter = toolAdapter;
    this.options = options;
    const classifier =
      options.classifier ??
      createDecisionClassifierAdapter(() => ({
        decision: this.serviceConfig.decision,
        chatApiKey: this.getLatestModelConfig()?.apiKey,
      }));
    this.router =
      options.router ??
      new VelesHarnessRouter(
        undefined,
        classifier,
        () => this.serviceConfig.decision.timeoutMs ?? 800,
      );
    for (const t of toolDefinitions) {
      this.tools.set(t.name, t);
    }
  }

  public getServiceConfig(): ServiceCapabilitiesConfigSpec {
    return {
      ...this.serviceConfig,
      decision: { ...this.serviceConfig.decision },
    };
  }

  public setServiceConfig(config: ServiceCapabilitiesConfigSpec): void {
    this.serviceConfig = {
      ...config,
      decision: { ...config.decision },
    };
  }

  public onEvent(listener: (ev: WorkerRuntimeEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public get currentGatewayInstanceId(): GatewayInstanceId | null {
    return this.toolAdapter.currentGatewayInstanceId;
  }

  public getRegisteredToolNames(): string[] {
    return Array.from(this.tools.keys());
  }

  public registerToolDefinition(spec: ToolDefinitionSpec): void {
    this.tools.set(spec.name, spec);
  }

  public getLatestModelConfig(): ProviderModelConfig | undefined {
    let latest: ProviderModelConfig | undefined;
    for (const cfg of this.runConfigs.values()) {
      latest = cfg;
    }
    return latest ? { ...latest } : undefined;
  }

  private emit(ev: WorkerRuntimeEvent): void {
    for (const listener of this.listeners) {
      listener(ev);
    }
  }

  private nextEventId(runId: RunId, suffix: string): string {
    this.eventCounter += 1;
    return `${runId}-ev-${this.eventCounter}-${suffix}`;
  }

  /**
   * Deterministic Boot Recovery (`!INVARIANT [KERN-INV-02]`, Scenarios 1, 2, 3, 4).
   */
  public async recoverUnfinishedRun(
    currentGatewayInstanceId: GatewayInstanceId | null,
  ): Promise<CrashRecoveryDecision | null> {
    const meta = await this.store.getMetadata();
    if (!meta) return null;

    const journal = await this.store.getJournal();
    const tail = findUnfinishedRunTail(meta.activeRunId, journal);
    if (!tail) return null;
    const targetRunId = tail.runId;

    const toolName =
      typeof tail.payload.toolName === "string" ? tail.payload.toolName : "";
    const effectClass: ToolEffectClass =
      (tail.payload.effectClass as ToolEffectClass | undefined) ??
      this.tools.get(toolName)?.effectClass ??
      "non_idempotent";
    const dispatchedGw =
      typeof tail.payload.gatewayInstanceId === "string"
        ? tail.payload.gatewayInstanceId
        : null;

    const decision = resolveUnfinishedRunTail(
      tail,
      effectClass,
      dispatchedGw,
      currentGatewayInstanceId,
    );
    if (decision.queryGatewayStatus) {
      const reconciled = await this.tryReconcileGatewayStatus(
        targetRunId,
        tail,
        toolName,
        currentGatewayInstanceId,
      );
      if (reconciled) return reconciled;
    }

    if (decision.nextState === "interrupted") {
      await this.store.appendEvent({
        eventId: this.nextEventId(targetRunId, "recover"),
        runId: targetRunId,
        type: "RunInterrupted",
        nextState: "interrupted",
        relatedOperationId: tail.relatedOperationId,
        payload: {
          interruptReason: decision.interruptReason,
          canAutoRetry: decision.canAutoRetry,
        },
      });
    }

    this.emit({
      type: "RECOVERY_RESOLVED",
      runId: targetRunId,
      state: decision.nextState,
      detail: { ...decision },
    });
    return decision;
  }

  private async tryReconcileGatewayStatus(
    runId: RunId,
    tail: RunJournalRecord,
    toolName: string,
    currentGatewayInstanceId: GatewayInstanceId | null,
  ): Promise<CrashRecoveryDecision | null> {
    if (
      !tail.relatedOperationId ||
      !currentGatewayInstanceId ||
      !this.toolAdapter.queryOperationStatus
    ) {
      return null;
    }
    const status = await this.toolAdapter.queryOperationStatus(
      tail.relatedOperationId,
      currentGatewayInstanceId,
    );
    if (!status.completed || !status.outcome) {
      return null;
    }
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "reconciled"),
      runId,
      type: "ToolResultCommitted",
      nextState: "inferring",
      relatedOperationId: tail.relatedOperationId,
      payload: {
        toolCallId: tail.payload.toolCallId,
        toolName,
        outputText: status.outcome.outputText,
        gatewayInstanceId: status.outcome.gatewayInstanceId,
      },
      fileUpdates: status.outcome.fileUpdates,
    });
    return {
      nextState: "inferring",
      canAutoRetry: false,
      queryGatewayStatus: true,
    };
  }

  public async cancelActiveRun(runId: RunId): Promise<void> {
    const existing = this.stopCommits.get(runId);
    if (existing) return existing;
    const commit = this.commitStoppedRun(runId);
    this.stopCommits.set(runId, commit);
    this.activeAborts.get(runId)?.abort();
    try { await commit; }
    finally { this.stopCommits.delete(runId); }
  }

  private async commitStoppedRun(runId: RunId): Promise<void> {
    const journal = await this.store.getJournal();
    const lastForRun = [...journal]
      .reverse()
      .find((r) => r.runId === runId && isLifecycleJournalRecord(r));
    if (lastForRun && isNonTerminalState(lastForRun.committedState)) {
      const uncertainRuns = uncertainRunIds(journal, runId);
      const uncertain = uncertainRuns.length > 0;
      try {
        await this.store.appendEvent({
          eventId: this.nextEventId(runId, "cancel"),
          runId,
          type: uncertain ? "RunInterrupted" : "RunCancelled",
          nextState: uncertain ? "interrupted" : "cancelled",
          payload: {
            reason: "user_cancelled",
            ...(uncertain
              ? {
                  interruptReason: "uncertain_tool_outcome",
                  uncertainRunIds: uncertainRuns,
                }
              : {}),
          },
        });
      } catch (error) {
        const tail = (await this.store.getJournal())
          .filter((record) => record.runId === runId && isLifecycleJournalRecord(record))
          .at(-1);
        // A final commit already queued before Stop may win. Its durable state stands.
        if (tail && !isNonTerminalState(tail.committedState)) return;
        throw error;
      }
      this.emit({
        type: "STATE_CHANGED",
        runId,
        state: uncertain ? "interrupted" : "cancelled",
      });
    }
  }

  public async startRun(
    runId: RunId,
    prompt: string,
    providerConfig: ProviderModelConfig,
    runOpts: StartRunOptions = {},
    parentSignal?: AbortSignal,
  ): Promise<RunState> {
    parentSignal?.throwIfAborted();
    const isolatedConfig: ProviderModelConfig = { ...providerConfig };
    this.runConfigs.set(runId, isolatedConfig);
    this.runOptions.set(runId, { ...runOpts });
    const abortController = new AbortController();
    const abortFromParent = () => abortController.abort(parentSignal?.reason);
    parentSignal?.addEventListener("abort", abortFromParent, { once: true });
    this.activeAborts.set(runId, abortController);

    try {
      await this.transitionToInferring(runId, prompt, isolatedConfig, runOpts);
      return await this.executeInferenceCycle(
        runId,
        isolatedConfig,
        abortController.signal,
        runOpts,
      );
    } catch (err) {
      return await this.handleRunCycleFailure(runId, abortController.signal, err);
    } finally {
      parentSignal?.removeEventListener("abort", abortFromParent);
      this.activeAborts.delete(runId);
    }
  }

  /**
   * Resolves a pending `awaiting_approval` state (`RESOLVE_APPROVAL` command).
   * If approved: logs `ToolIntentLogged` -> `ToolDispatched` -> executes tool -> resumes inference.
   * If denied: logs `ToolDenied` -> resumes inference so the model sees the rejection.
   */
  public async resolveApproval(
    runId: RunId,
    approved: boolean,
  ): Promise<RunState> {
    const journal = await this.store.getJournal();
    const runRecords = journal.filter(
      (r) => r.runId === runId && isLifecycleJournalRecord(r),
    );
    const approvalRec = runRecords.at(-1);
    if (
      !approvalRec ||
      approvalRec.committedState !== "awaiting_approval" ||
      approvalRec.type !== "ApprovalRequested"
    ) {
      throw new Error(
        `ERR_NO_PENDING_APPROVAL: Run ${runId} is not in awaiting_approval state`,
      );
    }

    const config = this.reconstructProviderConfig(runId, runRecords);
    const runOpts = this.reconstructRunOptions(runId, runRecords);
    const abortController = new AbortController();
    this.activeAborts.set(runId, abortController);

    try {
      const batchState = await this.applyApprovalDecision(
        runId,
        approved,
        approvalRec,
        abortController.signal,
        runOpts,
      );
      if (batchState !== "inferring") return batchState;

      return await this.executeInferenceCycle(
        runId,
        config,
        abortController.signal,
        runOpts,
      );
    } catch (err) {
      return await this.handleRunCycleFailure(runId, abortController.signal, err);
    } finally {
      this.activeAborts.delete(runId);
    }
  }

  private async applyApprovalDecision(
    runId: RunId,
    approved: boolean,
    approvalRec: RunJournalRecord,
    signal: AbortSignal,
    runOpts: StartRunOptions,
  ): Promise<RunState> {
    const {
      operationId,
      callId,
      toolName,
      effectClass,
      args,
      assistantText,
      remainingToolCalls,
    } = this.extractApprovalPayload(runId, approvalRec);

    if (!approved) {
      await this.store.appendEvent({
        eventId: this.nextEventId(runId, "denied"),
        runId,
        type: "ToolDenied",
        nextState: "inferring",
        relatedOperationId: operationId,
        payload: { toolCallId: callId, toolName, effectClass, args, assistantText },
      });
      this.emit({ type: "STATE_CHANGED", runId, state: "inferring" });
    } else {
      await this.commitAndExecuteTool(
        runId,
        operationId,
        callId,
        toolName,
        effectClass,
        args,
        assistantText,
        signal,
      );
    }

    if (remainingToolCalls.length > 0) {
      return this.executeToolCallBatch(
        runId,
        remainingToolCalls,
        "",
        signal,
        runOpts.allowedTools,
        new Map<string, number>(),
      );
    }
    return "inferring";
  }

  public async resolveUncertainTool(
    runId: RunId,
    resolution: "mark_succeeded" | "retry_anyway" | "cancel_run",
  ): Promise<RunState> {
    const journal = await this.store.getJournal();
    const runRecords = journal.filter(
      (record) => record.runId === runId && isLifecycleJournalRecord(record),
    );
    if (runRecords.at(-1)?.committedState !== "interrupted") {
      throw new Error(`ERR_NOT_INTERRUPTED: Run ${runId} is not in interrupted state`);
    }
    if (resolution === "cancel_run") {
      await this.store.appendEvent({
        eventId: this.nextEventId(runId, "cancel-uncertain"),
        runId,
        type: "RunCancelled",
        nextState: "cancelled",
        payload: { reason: "user_cancelled_uncertain_tool" },
      });
      this.emit({ type: "STATE_CHANGED", runId, state: "cancelled" });
      return "cancelled";
    }

    const config = this.reconstructProviderConfig(runId, runRecords);
    const runOpts = this.reconstructRunOptions(runId, runRecords);
    const abortController = new AbortController();
    this.activeAborts.set(runId, abortController);

    try {
      await this.applyUncertainToolResolutionEvent(runId, resolution, runRecords);
      return await this.executeInferenceCycle(
        runId,
        config,
        abortController.signal,
        runOpts,
      );
    } catch (err) {
      return await this.handleRunCycleFailure(runId, abortController.signal, err);
    } finally {
      this.activeAborts.delete(runId);
    }
  }

  private async applyUncertainToolResolutionEvent(
    runId: RunId,
    resolution: "mark_succeeded" | "retry_anyway",
    runRecords: readonly RunJournalRecord[],
  ): Promise<void> {
    const dispatched = [...runRecords].reverse().find((r) => r.type === "ToolDispatched");
    const callId =
      typeof dispatched?.payload.toolCallId === "string"
        ? dispatched.payload.toolCallId
        : "call_0";
    const toolName = String(dispatched?.payload.toolName ?? "unknown_tool");
    const outputText =
      resolution === "mark_succeeded"
        ? "USER_CONFIRMED_SUCCEEDED: External side effect verified manually by user."
        : "USER_REQUESTED_RETRY: Previous gateway attempt was interrupted; retry permitted.";
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, resolution),
      runId,
      type: "ToolResultCommitted",
      nextState: "inferring",
      relatedOperationId: dispatched?.relatedOperationId ?? null,
      payload: {
        toolCallId: callId,
        toolName,
        outputText,
      },
    });
    this.emit({ type: "STATE_CHANGED", runId, state: "inferring" });
  }

  private extractApprovalPayload(
    runId: RunId,
    approvalRec: RunJournalRecord,
  ): {
    operationId: OperationId;
    callId: string;
    toolName: string;
    effectClass: ToolEffectClass;
    args: Record<string, unknown>;
    assistantText: string;
    remainingToolCalls: readonly ToolCallSpec[];
  } {
    const p = approvalRec.payload;
    const rawRemaining = Array.isArray(p.remainingToolCalls)
      ? (p.remainingToolCalls as ToolCallSpec[])
      : [];
    return {
      operationId:
        approvalRec.relatedOperationId ?? `${runId}-op-${this.eventCounter + 1}`,
      callId: typeof p.toolCallId === "string" ? p.toolCallId : "call_0",
      toolName: String(p.toolName ?? ""),
      effectClass:
        (p.effectClass as ToolEffectClass | undefined) ?? "non_idempotent",
      args:
        typeof p.args === "object" && p.args !== null
          ? (p.args as Record<string, unknown>)
          : {},
      assistantText:
        typeof p.assistantText === "string" ? p.assistantText : "",
      remainingToolCalls: rawRemaining,
    };
  }

  private reconstructProviderConfig(
    runId: RunId,
    runRecords: readonly RunJournalRecord[],
  ): ProviderModelConfig {
    const cached = this.runConfigs.get(runId);
    if (cached) return cached;
    const created = runRecords.find((r) => r.type === "RunCreated");
    return {
      providerId: String(created?.payload.providerId ?? "openai"),
      baseUrl: String(created?.payload.baseUrl ?? "https://api.openai.com/v1"),
      model: String(created?.payload.model ?? "gpt-4o-mini"),
    };
  }

  private reconstructRunOptions(
    runId: RunId,
    runRecords: readonly RunJournalRecord[],
  ): StartRunOptions {
    const cached = this.runOptions.get(runId);
    if (cached) return cached;
    const created = runRecords.find((r) => r.type === "RunCreated");
    const p = created?.payload;
    if (!p) return {};
    const opts: StartRunOptions = {
      ...(typeof p.sessionId === "string" ? { sessionId: p.sessionId } : {}),
      ...(typeof p.parentSessionId === "string"
        ? { parentSessionId: p.parentSessionId }
        : {}),
      ...(Array.isArray(p.allowedTools)
        ? {
            allowedTools: p.allowedTools.filter(
              (t): t is string => typeof t === "string",
            ),
          }
        : {}),
      ...(Array.isArray(p.delegatableTools)
        ? {
            delegatableTools: p.delegatableTools.filter(
              (t): t is string => typeof t === "string",
            ),
          }
        : {}),
      ...(typeof p.unrestrictedTools === "boolean"
        ? { unrestrictedTools: p.unrestrictedTools }
        : {}),
    };
    this.runOptions.set(runId, opts);
    return opts;
  }

  private async transitionToInferring(
    runId: RunId,
    prompt: string,
    config: ProviderModelConfig,
    runOpts: StartRunOptions = {},
  ): Promise<void> {
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "created"),
      runId,
      type: "RunCreated",
      nextState: "created",
      payload: {
        prompt,
        providerId: config.providerId,
        baseUrl: config.baseUrl,
        model: config.model,
        ...(runOpts.sessionId ? { sessionId: runOpts.sessionId } : {}),
        ...(runOpts.parentSessionId ? { parentSessionId: runOpts.parentSessionId } : {}),
        ...(runOpts.allowedTools ? { allowedTools: [...runOpts.allowedTools] } : {}),
        ...(runOpts.delegatableTools !== undefined
          ? { delegatableTools: [...runOpts.delegatableTools] }
          : {}),
        ...(runOpts.unrestrictedTools !== undefined
          ? { unrestrictedTools: runOpts.unrestrictedTools }
          : {}),
      },
    });
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "preparing"),
      runId,
      type: "RunPreparing",
      nextState: "preparing",
      payload: { providerId: config.providerId, model: config.model },
    });
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "inferring"),
      runId,
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: { providerId: config.providerId, model: config.model },
    });
    this.emit({ type: "STATE_CHANGED", runId, state: "inferring" });
  }

  private async buildRoundMessages(
    runId: RunId,
    resolvedOpts: StartRunOptions,
    signal: AbortSignal,
  ): Promise<ChatMessage[]> {
    const journal = await this.store.getJournal();
    if (resolvedOpts.sessionId !== undefined || this.options.multiTurnWorkspaceContext) {
      const fileTree = await this.store.getFileTree();
      const latestPrompt =
        [...journal]
          .reverse()
          .find((r) => r.runId === runId && r.type === "RunCreated")?.payload
          .prompt ?? "";
      const configuredMode = this.serviceConfig.decision.mode;
      const routed = await this.router.routeWithOptionalClassifier(
        String(latestPrompt),
        Object.keys(fileTree),
        signal,
      );
      signal.throwIfAborted();
      await this.store.appendEvent({
        eventId: this.nextEventId(runId, "harness-routing"),
        runId,
        type: "CheckpointCommitted",
        nextState: "inferring",
        payload: {
          harnessRouting: {
            source: routed.source,
            specialistIds: routed.profiles.map((p) => p.id),
            configuredMode,
            ...(routed.fallbackReason ? { fallbackReason: routed.fallbackReason } : {}),
          },
        },
      });
      return ContextBuilder.buildMessages(journal, {
        sessionId: resolvedOpts.sessionId,
        fileTree,
        specialistCapsules: routed.profiles,
        hotRunWindow: this.options.hotRunWindow ?? 2,
        maxCharBudget: 48_000,
      });
    }
    return ContextBuilder.buildMessages(journal, runId);
  }

  private async commitInferenceUsage(runId: RunId, usage: TokenUsage | null): Promise<void> {
    if (!usage) return;
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "inference-usage"),
      runId,
      type: "CheckpointCommitted",
      nextState: "inferring",
      payload: { inferenceUsage: { ...usage } },
    });
  }

  private async buildRunTelemetry(
    runId: RunId,
    finalUsage: TokenUsage | null,
  ): Promise<{ durationMs: number; usage?: TokenUsage }> {
    const records = (await this.store.getJournal()).filter((r) => r.runId === runId);
    const startedAt = records.find((r) => r.type === "RunCreated")?.timestampMs ?? Date.now();
    const durationMs = Math.max(0, Date.now() - startedAt);
    const reports = records
      .filter((r) => r.type === "CheckpointCommitted")
      .map((r) => parseRecordedUsage(r.payload.inferenceUsage))
      .filter((usage): usage is TokenUsage => usage !== null);
    if (finalUsage) reports.push(finalUsage);
    if (reports.length === 0) return { durationMs };
    const usage = reports.reduce((sum, report) => ({
      promptTokens: sum.promptTokens + report.promptTokens,
      completionTokens: sum.completionTokens + report.completionTokens,
      totalTokens: sum.totalTokens + report.totalTokens,
    }), { promptTokens: 0, completionTokens: 0, totalTokens: 0 });
    return { durationMs, usage };
  }

  private async executeInferenceCycle(
    runId: RunId,
    config: ProviderModelConfig,
    signal: AbortSignal,
    runOpts?: StartRunOptions,
  ): Promise<RunState> {
    const resolvedOpts = runOpts ?? this.runOptions.get(runId) ?? {};
    const maxToolRounds = 10;
    const seenCallSignatures = new Map<string, number>();

    for (let round = 0; round < maxToolRounds; round++) {
      signal.throwIfAborted();
      const messages = await awaitWithAbort(this.buildRoundMessages(runId, resolvedOpts, signal), signal);
      signal.throwIfAborted();
      const { assistantText, toolCalls, usage } = await this.consumeInferenceStream(
        runId,
        config,
        messages,
        signal,
        resolvedOpts.allowedTools,
      );

      if (toolCalls.length === 0) {
        const telemetry = await this.buildRunTelemetry(runId, usage);
        await this.store.appendEvent({
          eventId: this.nextEventId(runId, "completed"),
          runId,
          type: "RunCompleted",
          nextState: "completed",
          payload: {
            assistantText,
            providerId: config.providerId,
            model: config.model,
            ...telemetry,
            ...(resolvedOpts.sessionId ? { sessionId: resolvedOpts.sessionId } : {}),
          },
        });
        this.emit({ type: "STATE_CHANGED", runId, state: "completed" });
        return "completed";
      }

      await this.commitInferenceUsage(runId, usage);
      const nextState = await this.executeToolCallBatch(
        runId,
        toolCalls,
        assistantText,
        signal,
        resolvedOpts.allowedTools,
        seenCallSignatures,
      );
      if (nextState !== "inferring") {
        return nextState;
      }
    }
    throw new Error("ERR_MAX_TOOL_ROUNDS_EXCEEDED");
  }

  /**
   * Executes all tool calls returned in an assistant turn sequentially (b.md #3).
   * Attaches `assistantText` only to the first call in the batch and preserves
   * any remaining calls if a tool pauses in `awaiting_approval`.
   */
  private async executeToolCallBatch(
    runId: RunId,
    toolCalls: readonly ToolCallSpec[],
    assistantText: string,
    signal: AbortSignal,
    allowedTools: readonly string[] | undefined,
    seenCallSignatures: Map<string, number>,
  ): Promise<RunState> {
    for (let i = 0; i < toolCalls.length; i++) {
      signal.throwIfAborted();
      const call = toolCalls[i]!;
      const textForCall = i === 0 ? assistantText : "";
      const callSig = `${call.name}:${call.argumentsJson.trim()}`;
      const seenCount = (seenCallSignatures.get(callSig) ?? 0) + 1;
      seenCallSignatures.set(callSig, seenCount);

      if (seenCount >= 2) {
        await this.recordRepeatedCallLoopGuard(runId, call, textForCall);
        continue;
      }

      const remaining = toolCalls.slice(i + 1);
      const nextState = await this.processToolCall(
        runId,
        call,
        textForCall,
        signal,
        allowedTools,
        remaining,
      );
      if (nextState !== "inferring") {
        return nextState;
      }
    }
    return "inferring";
  }

  private async recordRepeatedCallLoopGuard(
    runId: RunId,
    call: ToolCallSpec,
    assistantText: string,
  ): Promise<void> {
    const operationId = `${runId}-op-${this.eventCounter + 1}`;
    const args = parseToolArgs(call.argumentsJson);
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "intent-loop-guard"),
      runId,
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: operationId,
      payload: {
        toolCallId: call.id,
        toolName: call.name,
        effectClass: "pure_read",
        args,
        assistantText,
      },
    });
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "result-loop-guard"),
      runId,
      type: "ToolResultCommitted",
      nextState: "inferring",
      relatedOperationId: operationId,
      payload: {
        toolCallId: call.id,
        toolName: call.name,
        outputText:
          `LOOP_GUARD_TRIGGERED: Duplicate tool call '${call.name}' with identical arguments blocked. ` +
          `Inspect the existing workspace state or adjust your strategy instead of repeating the same call.`,
        gatewayInstanceId: this.toolAdapter.currentGatewayInstanceId,
      },
    });
  }

  private async consumeInferenceStream(
    runId: RunId,
    config: ProviderModelConfig,
    messages: readonly ChatMessage[],
    signal: AbortSignal,
    allowedTools?: readonly string[],
  ): Promise<{
    assistantText: string;
    toolCalls: ToolCallSpec[];
    usage: TokenUsage | null;
  }> {
    let assistantText = "";
    let latestToolCalls: ToolCallSpec[] = [];
    let latestUsage: TokenUsage | null = null;
    const availableTools = allowedTools
      ? Array.from(this.tools.values()).filter((t) => allowedTools.includes(t.name))
      : Array.from(this.tools.values());
    const stream = this.inference.streamChat({
      config,
      messages,
      tools: availableTools,
      signal,
    });

    const iterator = stream[Symbol.asyncIterator]();
    try {
      while (true) {
        signal.throwIfAborted();
        const next = await awaitWithAbort(iterator.next(), signal);
        signal.throwIfAborted();
        if (next.done) break;
        const chunk = next.value;
        if (chunk.deltaText) {
          assistantText += chunk.deltaText;
          this.emit({
            type: "STREAM_DELTA",
            runId,
            state: "inferring",
            detail: { deltaText: chunk.deltaText },
          });
        }
        if (chunk.toolCalls.length > 0) {
          latestToolCalls = [...chunk.toolCalls];
        }
        if (chunk.usage) {
          latestUsage = chunk.usage;
        }
      }
    } finally {
      if (signal.aborted) void iterator.return?.().catch(() => {});
    }
    return { assistantText, toolCalls: latestToolCalls, usage: latestUsage };
  }

  private async processToolCall(
    runId: RunId,
    call: ToolCallSpec,
    assistantText: string,
    signal: AbortSignal,
    allowedTools?: readonly string[],
    remainingToolCalls: readonly ToolCallSpec[] = [],
  ): Promise<RunState> {
    if (allowedTools && !allowedTools.includes(call.name)) {
      throw new Error(
        `ERR_TOOL_PERMISSION_DENIED: Tool '${call.name}' is not in session allowedTools`,
      );
    }
    const spec = this.tools.get(call.name);
    const effectClass: ToolEffectClass = spec?.effectClass ?? "non_idempotent";
    const args = parseToolArgs(call.argumentsJson);
    const operationId = `${runId}-op-${this.eventCounter + 1}`;

    if (spec?.requiresApproval) {
      await this.store.appendEvent({
        eventId: this.nextEventId(runId, "approval"),
        runId,
        type: "ApprovalRequested",
        nextState: "awaiting_approval",
        relatedOperationId: operationId,
        payload: {
          toolCallId: call.id,
          toolName: call.name,
          effectClass,
          args,
          assistantText,
          ...(remainingToolCalls.length > 0
            ? { remainingToolCalls: [...remainingToolCalls] }
            : {}),
        },
      });
      this.emit({ type: "STATE_CHANGED", runId, state: "awaiting_approval" });
      return "awaiting_approval";
    }

    return this.commitAndExecuteTool(
      runId,
      operationId,
      call.id,
      call.name,
      effectClass,
      args,
      assistantText,
      signal,
    );
  }

  private async commitAndExecuteTool(
    runId: RunId,
    operationId: OperationId,
    toolCallId: string,
    toolName: string,
    effectClass: ToolEffectClass,
    args: Record<string, unknown>,
    assistantText: string,
    signal: AbortSignal,
  ): Promise<RunState> {
    // KERN-INV-01: Commit ToolIntentLogged BEFORE dispatching effect
    signal.throwIfAborted();
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "intent"),
      runId,
      type: "ToolIntentLogged",
      nextState: "awaiting_tool",
      relatedOperationId: operationId,
      payload: {
        toolCallId,
        toolName,
        effectClass,
        args,
        assistantText,
      },
    });

    // KERN-INV-01: Commit ToolDispatched BEFORE invoking external/OPFS adapter
    signal.throwIfAborted();
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "dispatched"),
      runId,
      type: "ToolDispatched",
      nextState: "executing_tool",
      relatedOperationId: operationId,
      payload: {
        toolCallId,
        toolName,
        effectClass,
        gatewayInstanceId: this.toolAdapter.currentGatewayInstanceId,
      },
    });
    this.emit({ type: "STATE_CHANGED", runId, state: "executing_tool" });

    signal.throwIfAborted();
    const outcome = await awaitWithAbort(this.executeToolWithSafeFallback({
      runId,
      operationId,
      toolName,
      effectClass,
      args,
      signal,
    }), signal);

    const observedFileHashes = await this.captureObservedFileHashes(args, outcome);
    signal.throwIfAborted();
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "result"),
      runId,
      type: "ToolResultCommitted",
      nextState: "inferring",
      relatedOperationId: operationId,
      payload: {
        toolCallId,
        toolName,
        outputText: outcome.outputText,
        gatewayInstanceId: outcome.gatewayInstanceId,
        ...(Object.keys(observedFileHashes).length > 0
          ? { observedFileHashes }
          : {}),
        ...(outcome.imageDataUrls && outcome.imageDataUrls.length > 0
          ? { imageDataUrls: [...outcome.imageDataUrls] }
          : {}),
      },
      fileUpdates: outcome.fileUpdates,
    });
    this.emit({ type: "STATE_CHANGED", runId, state: "inferring" });
    return "inferring";
  }

  private async executeToolWithSafeFallback(
    req: ToolExecutionRequest,
  ): Promise<ToolExecutionOutcome> {
    try {
      return await this.toolAdapter.executeTool(req);
    } catch (err) {
      if (req.signal?.aborted || req.effectClass === "non_idempotent") {
        throw err;
      }
      const msg = err instanceof Error ? err.message : String(err);
      return {
        gatewayInstanceId:
          this.toolAdapter.currentGatewayInstanceId ?? "browser-local-v1",
        outputText: `ToolError (${req.toolName}): ${msg}`,
      };
    }
  }

  private async captureObservedFileHashes(
    args: Record<string, unknown>,
    outcome: ToolExecutionOutcome,
  ): Promise<Record<string, string>> {
    const tree = await this.store.getFileTree();
    const hashes: Record<string, string> = {};
    if (typeof args.path === "string" && tree[args.path]) {
      hashes[args.path] = tree[args.path]!;
    }
    if (outcome.fileUpdates) {
      for (const [relPath, content] of Object.entries(outcome.fileUpdates)) {
        if (content !== null) {
          hashes[relPath] = await this.store.computeBlobHashForPreview(content);
        }
      }
    }
    return hashes;
  }

  private async handleRunCycleFailure(
    runId: RunId,
    signal: AbortSignal,
    err: unknown,
  ): Promise<RunState> {
    await this.stopCommits.get(runId);
    const journal = await this.store.getJournal();
    const lastForRun = [...journal]
      .reverse()
      .find((r) => r.runId === runId && isLifecycleJournalRecord(r));
    if (!lastForRun) {
      throw err;
    }
    if (!isNonTerminalState(lastForRun.committedState)) {
      return lastForRun.committedState;
    }
    if (signal.aborted) {
      await this.cancelActiveRun(runId);
      const stopped = (await this.store.getJournal())
        .filter((r) => r.runId === runId && isLifecycleJournalRecord(r))
        .at(-1);
      return stopped?.committedState ?? "cancelled";
    }
    const reason =
      lastForRun.committedState === "executing_tool"
        ? "uncertain_tool_outcome"
        : "gateway_disconnected_safe_to_retry";
    await this.store.appendEvent({
      eventId: this.nextEventId(runId, "interrupted"),
      runId,
      type: "RunInterrupted",
      nextState: "interrupted",
      payload: {
        interruptReason: reason,
        error: err instanceof Error ? err.message : String(err),
      },
    });
    this.emit({ type: "STATE_CHANGED", runId, state: "interrupted" });
    return "interrupted";
  }

  public async handleWorkerCommand(rawCommand: unknown): Promise<RunState | null> {
    const cmd = validateWorkerCommand(rawCommand);
    if (cmd.type === "START_RUN") {
      return this.startRun(cmd.runId, cmd.prompt, cmd.providerConfig);
    }
    if (cmd.type === "CANCEL_RUN") {
      await this.cancelActiveRun(cmd.runId);
      const tail = (await this.store.getJournal())
        .filter((record) => record.runId === cmd.runId && isLifecycleJournalRecord(record))
        .at(-1);
      return tail?.committedState ?? null;
    }
    if (cmd.type === "RESOLVE_APPROVAL") {
      return this.resolveApproval(cmd.runId, cmd.approved);
    }
    if (cmd.type === "RECOVER_BOOT") {
      const decision = await this.recoverUnfinishedRun(
        cmd.currentGatewayInstanceId,
      );
      return decision?.nextState ?? null;
    }
    return null;
  }
}
