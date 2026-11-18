import type {
  ContextEconomicsSummary,
  RunJournalRecord,
  RunState,
} from "../protocol/types.ts";

export interface ChatRun {
  readonly runId: string;
  readonly prompt: string;
  readonly records: RunJournalRecord[];
}

export interface HarnessRoutingProjection {
  readonly record: RunJournalRecord;
  readonly round: number;
  readonly source: "tier1_classifier" | "tier0_deterministic";
  readonly configuredMode: string;
  readonly specialistIds: readonly string[];
  readonly fallbackReason?: string;
  readonly routerLabel: string;
}

export interface SubagentRunProjection {
  readonly runId: string;
  readonly sessionId: string;
  readonly prompt: string;
  readonly model: string;
  readonly allowedTools: readonly string[];
  readonly stateLabel: "Completed" | "Stopped" | "Running";
  readonly assistantText: string;
  readonly totalTokens: number | null;
  readonly records: readonly RunJournalRecord[];
  readonly createdRecord: RunJournalRecord;
}

export interface RunTelemetryProjection {
  readonly providerId: string;
  readonly model: string;
  readonly liveState: RunState;
  readonly isLive: boolean;
  readonly routingRounds: readonly HarnessRoutingProjection[];
  readonly latestRouting: HarnessRoutingProjection | null;
  readonly totalTokens: number | null;
  readonly promptTokens: number | null;
  readonly completionTokens: number | null;
  readonly durationMs: number | null;
  readonly toolCount: number;
  readonly subagentRuns: readonly SubagentRunProjection[];
  readonly contextEconomics: ContextEconomicsSummary | null;
  readonly telemetryAnchorRecord: RunJournalRecord | null;
}

const ROUTER_MODE_LABELS: Readonly<Record<string, string>> = {
  "openai-decisions": "⚡ Luna System-1",
  typesafe: "⚡ TypeSafe System-1",
  "local-laya": "⚡ Local Laya",
};

const TERMINAL_STATES: ReadonlySet<RunState> = new Set([
  "completed",
  "cancelled",
  "failed",
  "interrupted",
]);

/** A presentation projection only: the Worker journal remains authoritative. */
export function projectChatRuns(
  journal: readonly RunJournalRecord[],
  sessionId: string,
): ChatRun[] {
  const runs = new Map<string, ChatRun>();
  const ownersByRun = new Map<string, string>();
  const ownersBySession = new Map<string, string>();
  const deleted = new Set(
    journal
      .map((r) => r.payload.deletedRunId)
      .filter((id): id is string => typeof id === "string"),
  );
  for (const record of journal) {
    if (record.type === "RunCreated") {
      const session = String(record.payload.sessionId ?? "session-main");
      const parent = record.payload.parentSessionId;
      const owner =
        typeof parent === "string"
          ? ownersBySession.get(parent)
          : session === sessionId
            ? record.runId
            : undefined;
      if (owner) {
        ownersByRun.set(record.runId, owner);
        ownersBySession.set(session, owner);
        if (owner === record.runId) {
          runs.set(owner, {
            runId: owner,
            prompt: String(record.payload.prompt ?? ""),
            records: [],
          });
        }
      }
    }
    const owner = ownersByRun.get(record.runId);
    if (owner) runs.get(owner)?.records.push(record);
  }
  return [...runs.values()].filter((run) => !deleted.has(run.runId));
}

export function firstToolRecord(
  records: readonly RunJournalRecord[],
  record: RunJournalRecord,
): boolean {
  return (
    records.find(
      (r) =>
        r.runId === record.runId &&
        r.payload.toolCallId === record.payload.toolCallId &&
        (r.type === "ToolIntentLogged" || r.type === "ApprovalRequested"),
    ) === record
  );
}

export function formatRouterLabel(
  source: string,
  configuredMode: string,
): string {
  if (source !== "tier1_classifier") return "🛡️ Tier-0 Gate";
  return ROUTER_MODE_LABELS[configuredMode] ?? "⚡ System-1 Router";
}

function parseRoutingCheckpoint(
  record: RunJournalRecord,
  round: number,
): HarnessRoutingProjection | null {
  const raw = record.payload.harnessRouting;
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  const source =
    obj.source === "tier1_classifier"
      ? "tier1_classifier"
      : "tier0_deterministic";
  const configuredMode = String(obj.configuredMode ?? "tier0-only");
  const specialistIds = Array.isArray(obj.specialistIds)
    ? obj.specialistIds.filter((id): id is string => typeof id === "string")
    : [];
  const fallbackReason =
    typeof obj.fallbackReason === "string" ? obj.fallbackReason : undefined;
  return {
    record,
    round,
    source,
    configuredMode,
    specialistIds,
    ...(fallbackReason ? { fallbackReason } : {}),
    routerLabel: formatRouterLabel(source, configuredMode),
  };
}

export function extractHarnessRouting(
  records: readonly RunJournalRecord[],
  runId: string,
): HarnessRoutingProjection[] {
  const result: HarnessRoutingProjection[] = [];
  for (const record of records) {
    if (record.runId !== runId || record.type !== "CheckpointCommitted") {
      continue;
    }
    const parsed = parseRoutingCheckpoint(record, result.length + 1);
    if (parsed) result.push(parsed);
  }
  return result;
}

function readUsageTuple(
  raw: unknown,
): { promptTokens: number; completionTokens: number; totalTokens: number } | null {
  if (typeof raw !== "object" || raw === null) return null;
  const u = raw as Record<string, unknown>;
  if (typeof u.totalTokens !== "number" || !Number.isFinite(u.totalTokens)) {
    return null;
  }
  return {
    promptTokens: typeof u.promptTokens === "number" ? u.promptTokens : 0,
    completionTokens:
      typeof u.completionTokens === "number" ? u.completionTokens : 0,
    totalTokens: u.totalTokens,
  };
}

function extractRunTokenTotals(
  records: readonly RunJournalRecord[],
  runId: string,
): { promptTokens: number; completionTokens: number; totalTokens: number } | null {
  const completed = records.find(
    (r) => r.runId === runId && r.type === "RunCompleted",
  );
  const finalUsage = readUsageTuple(completed?.payload.usage);
  if (finalUsage) return finalUsage;

  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;
  let found = false;
  for (const r of records) {
    if (r.runId !== runId || r.type !== "CheckpointCommitted") continue;
    const step = readUsageTuple(r.payload.inferenceUsage);
    if (step) {
      found = true;
      promptTokens += step.promptTokens;
      completionTokens += step.completionTokens;
      totalTokens += step.totalTokens;
    }
  }
  return found ? { promptTokens, completionTokens, totalTokens } : null;
}

function resolveSubagentStateLabel(
  childRecords: readonly RunJournalRecord[],
): "Completed" | "Stopped" | "Running" {
  if (childRecords.some((r) => r.type === "RunCompleted")) return "Completed";
  if (
    childRecords.some((r) =>
      ["RunCancelled", "RunInterrupted", "RunFailed"].includes(r.type),
    )
  ) {
    return "Stopped";
  }
  return "Running";
}

export function projectSubagentRuns(
  run: ChatRun,
): SubagentRunProjection[] {
  const childCreated = run.records.filter(
    (r) => r.runId !== run.runId && r.type === "RunCreated",
  );
  return childCreated.map((createdRecord) => {
    const childRunId = createdRecord.runId;
    const childRecords = run.records.filter((r) => r.runId === childRunId);
    const completed = childRecords.find((r) => r.type === "RunCompleted");
    const usage = extractRunTokenTotals(childRecords, childRunId);
    const rawTools = createdRecord.payload.allowedTools;
    return {
      runId: childRunId,
      sessionId: String(createdRecord.payload.sessionId ?? childRunId),
      prompt: String(createdRecord.payload.prompt ?? ""),
      model: String(createdRecord.payload.model ?? "default"),
      allowedTools: Array.isArray(rawTools)
        ? rawTools.filter((t): t is string => typeof t === "string")
        : [],
      stateLabel: resolveSubagentStateLabel(childRecords),
      assistantText: String(completed?.payload.assistantText ?? ""),
      totalTokens: usage?.totalTokens ?? null,
      records: childRecords,
      createdRecord,
    };
  });
}

function isRunLifecycleEvent(record: RunJournalRecord): boolean {
  const p = record.payload;
  return !p.sessionConfig && !p.uploadedFile && !p.deletedRunId;
}

function resolveRunLiveState(
  records: readonly RunJournalRecord[],
  runId: string,
): RunState {
  let state: RunState = "created";
  for (const record of records) {
    if (record.runId === runId && isRunLifecycleEvent(record)) {
      state = record.committedState;
    }
  }
  return state;
}

function numField(obj: Record<string, unknown>, key: string): number {
  const val = obj[key];
  return typeof val === "number" && Number.isFinite(val) ? val : 0;
}

export function extractContextEconomics(
  records: readonly RunJournalRecord[],
  runId: string,
): ContextEconomicsSummary | null {
  for (let i = records.length - 1; i >= 0; i--) {
    const rec = records[i]!;
    if (rec.runId !== runId || rec.type !== "CheckpointCommitted") continue;
    const raw = rec.payload.contextEconomics;
    if (typeof raw !== "object" || raw === null) continue;
    const obj = raw as Record<string, unknown>;
    return {
      systemContractChars: numField(obj, "systemContractChars"),
      specialistCapsuleChars: numField(obj, "specialistCapsuleChars"),
      workspaceTreeChars: numField(obj, "workspaceTreeChars"),
      conversationChars: numField(obj, "conversationChars"),
      toolIOChars: numField(obj, "toolIOChars"),
      imageWeightChars: numField(obj, "imageWeightChars"),
      totalChars: numField(obj, "totalChars"),
      estimatedTokens: numField(obj, "estimatedTokens"),
      cacheablePrefixTokens: numField(obj, "cacheablePrefixTokens"),
      foldedSavedChars: numField(obj, "foldedSavedChars"),
      hotRunCount: numField(obj, "hotRunCount"),
      coldFoldedRunCount: numField(obj, "coldFoldedRunCount"),
    };
  }
  return null;
}

export function projectRunTelemetry(run: ChatRun): RunTelemetryProjection {
  const ownRecords = run.records.filter((r) => r.runId === run.runId);
  const created = ownRecords.find((r) => r.type === "RunCreated");
  const completed = ownRecords.find((r) => r.type === "RunCompleted");
  const providerId = String(
    completed?.payload.providerId ?? created?.payload.providerId ?? "openai",
  );
  const model = String(
    completed?.payload.model ?? created?.payload.model ?? "gpt-4o",
  );
  const liveState = resolveRunLiveState(ownRecords, run.runId);
  const routingRounds = extractHarnessRouting(ownRecords, run.runId);
  const latestRouting = routingRounds.at(-1) ?? null;
  const usage = extractRunTokenTotals(ownRecords, run.runId);
  const rawDuration = completed?.payload.durationMs;
  const durationMs =
    typeof rawDuration === "number" && Number.isFinite(rawDuration)
      ? rawDuration
      : null;
  const toolCount = ownRecords.filter((r) =>
    (r.type === "ToolIntentLogged" || r.type === "ApprovalRequested") &&
    firstToolRecord(ownRecords, r),
  ).length;
  const subagentRuns = projectSubagentRuns(run);
  const contextEconomics = extractContextEconomics(ownRecords, run.runId);
  return {
    providerId,
    model,
    liveState,
    isLive: !TERMINAL_STATES.has(liveState),
    routingRounds,
    latestRouting,
    totalTokens: usage?.totalTokens ?? null,
    promptTokens: usage?.promptTokens ?? null,
    completionTokens: usage?.completionTokens ?? null,
    durationMs,
    toolCount,
    subagentRuns,
    contextEconomics,
    telemetryAnchorRecord:
      latestRouting?.record ?? completed ?? created ?? null,
  };
}

export function formatToolTargetHint(args: unknown): string {
  if (typeof args !== "object" || args === null) return "";
  const record = args as Record<string, unknown>;
  const candidate =
    record.path ?? record.query ?? record.url ?? record.sessionId ?? "";
  if (typeof candidate !== "string" || !candidate.trim()) return "";
  const clean = candidate.trim();
  return clean.length > 38 ? `${clean.slice(0, 35)}…` : clean;
}

