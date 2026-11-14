/**
 * @module SYS-PROTO/validation
 * Runtime boundary validation, state-machine transition guards, and deterministic
 * crash-window recovery resolution. Cyclomatic complexity V(G) <= 10 per function.
 */
import {
  PROTOCOL_VERSION,
  type ExecutionStreamEvent,
  type GatewayInstanceId,
  type InterruptReason,
  type ProtocolCommandRequest,
  type ProtocolOperationType,
  type RunJournalRecord,
  type RunState,
  type ToolEffectClass,
} from "./types.ts";

export interface ValidationResult<T> {
  readonly ok: boolean;
  readonly value?: T;
  readonly error?: string;
}

const VALID_OPERATIONS: ReadonlySet<ProtocolOperationType> = new Set([
  "inference.start",
  "inference.cancel",
  "tool.execute",
  "tool.cancel",
  "execution.status",
  "capabilities.list",
]);

const VALID_GATEWAY_STAGES: ReadonlySet<string> = new Set([
  "ACCEPTED",
  "STARTED",
  "COMPLETED",
]);

/**
 * Explicit adjacency map for the COMET Run State Machine.
 */
const ALLOWED_TRANSITIONS: Readonly<Record<RunState, ReadonlySet<RunState>>> = {
  created: new Set(["preparing", "cancelled", "failed"]),
  preparing: new Set(["inferring", "cancelled", "failed", "interrupted"]),
  inferring: new Set([
    "awaiting_approval",
    "awaiting_tool",
    "completed",
    "cancelled",
    "failed",
    "interrupted",
  ]),
  awaiting_approval: new Set(["awaiting_tool", "inferring", "cancelled", "interrupted"]),
  awaiting_tool: new Set(["executing_tool", "inferring", "cancelled", "failed", "interrupted"]),
  executing_tool: new Set(["inferring", "cancelled", "failed", "interrupted"]),
  interrupted: new Set(["preparing", "inferring", "awaiting_tool", "cancelled", "failed"]),
  completed: new Set([]),
  cancelled: new Set([]),
  failed: new Set([]),
};

export function isValidStateTransition(from: RunState, to: RunState): boolean {
  return ALLOWED_TRANSITIONS[from].has(to);
}

function isNonEmptyString(val: unknown): val is string {
  return typeof val === "string" && val.trim().length > 0;
}

function hasValidResourceLimits(limits: unknown): boolean {
  if (typeof limits !== "object" || limits === null) return false;
  const l = limits as Record<string, unknown>;
  return (
    typeof l.timeoutMs === "number" &&
    l.timeoutMs > 0 &&
    l.timeoutMs <= 300_000 &&
    typeof l.maxOutputBytes === "number" &&
    l.maxOutputBytes > 0
  );
}

function hasValidPathConfinement(paths: unknown): boolean {
  if (!Array.isArray(paths)) return false;
  for (const p of paths) {
    if (typeof p !== "string") return false;
    if (p.startsWith("/") || p.includes("..") || p.includes("\\")) return false;
  }
  return true;
}

export function validateCommandRequest(
  input: unknown,
): ValidationResult<ProtocolCommandRequest> {
  if (typeof input !== "object" || input === null) {
    return { ok: false, error: "ERR_INVALID_ENVELOPE: Request must be an object" };
  }
  const req = input as Record<string, unknown>;
  if (req.protocolVersion !== PROTOCOL_VERSION) {
    return { ok: false, error: `ERR_PROTOCOL_VERSION: Expected ${PROTOCOL_VERSION}` };
  }
  if (!isNonEmptyString(req.operationId) || !isNonEmptyString(req.runId)) {
    return { ok: false, error: "ERR_MISSING_IDS: operationId and runId are required" };
  }
  if (!VALID_OPERATIONS.has(req.operation as ProtocolOperationType)) {
    return { ok: false, error: `ERR_UNKNOWN_OPERATION: ${String(req.operation)}` };
  }
  if (!hasValidResourceLimits(req.limits)) {
    return { ok: false, error: "ERR_INVALID_LIMITS: timeoutMs (1..300000) and maxOutputBytes required" };
  }
  const scope = req.capabilityScope as Record<string, unknown> | undefined;
  if (!scope || !hasValidPathConfinement(scope.writablePaths)) {
    return { ok: false, error: "ERR_UNSAFE_SCOPE: writablePaths must be relative without '..'" };
  }
  return { ok: true, value: req as unknown as ProtocolCommandRequest };
}

export function validateStreamEvent(
  input: unknown,
): ValidationResult<ExecutionStreamEvent> {
  if (typeof input !== "object" || input === null) {
    return { ok: false, error: "ERR_INVALID_EVENT: Event must be an object" };
  }
  const ev = input as Record<string, unknown>;
  if (ev.protocolVersion !== PROTOCOL_VERSION) {
    return { ok: false, error: "ERR_EVENT_VERSION_MISMATCH" };
  }
  if (!isNonEmptyString(ev.operationId) || !isNonEmptyString(ev.gatewayInstanceId)) {
    return { ok: false, error: "ERR_MISSING_EVENT_IDS" };
  }
  if (ev.stage === "COMMITTED_BY_BROWSER") {
    return {
      ok: false,
      error: "ERR_ILLEGAL_GATEWAY_STAGE: Gateway must never claim COMMITTED_BY_BROWSER",
    };
  }
  if (!VALID_GATEWAY_STAGES.has(String(ev.stage))) {
    return { ok: false, error: `ERR_UNKNOWN_STAGE: ${String(ev.stage)}` };
  }
  return { ok: true, value: ev as unknown as ExecutionStreamEvent };
}

export interface CrashRecoveryDecision {
  readonly nextState: RunState;
  readonly interruptReason?: InterruptReason;
  readonly canAutoRetry: boolean;
  readonly queryGatewayStatus: boolean;
}

/**
 * Deterministic Crash-Window Resolver (!INVARIANT [KERN-INV-02]).
 * Evaluates the tail record of an unfinished run on browser reload or reconnect.
 */
export function resolveUnfinishedRunTail(
  tailRecord: RunJournalRecord,
  effectClass: ToolEffectClass,
  dispatchedGatewayInstanceId: GatewayInstanceId | null,
  currentGatewayInstanceId: GatewayInstanceId | null,
): CrashRecoveryDecision {
  // Case 1: Intent was logged locally, but never dispatched to Gateway/OPFS -> safe to resume/retry
  if (tailRecord.type === "ToolIntentLogged") {
    return {
      nextState: "interrupted",
      interruptReason: "user_reload_or_tab_close",
      canAutoRetry: true,
      queryGatewayStatus: false,
    };
  }

  // Case 2: Tool was dispatched; check if the same ephemeral Gateway process is still alive
  if (tailRecord.type === "ToolDispatched") {
    const sameGatewayAlive =
      dispatchedGatewayInstanceId !== null &&
      dispatchedGatewayInstanceId === currentGatewayInstanceId;

    if (sameGatewayAlive) {
      return {
        nextState: "executing_tool",
        canAutoRetry: false,
        queryGatewayStatus: true,
      };
    }

    // Gateway restarted or unreachable: non-idempotent tools MUST NOT auto-retry
    if (effectClass === "non_idempotent") {
      return {
        nextState: "interrupted",
        interruptReason: "uncertain_tool_outcome",
        canAutoRetry: false,
        queryGatewayStatus: false,
      };
    }

    return {
      nextState: "interrupted",
      interruptReason: "gateway_disconnected_safe_to_retry",
      canAutoRetry: true,
      queryGatewayStatus: false,
    };
  }

  return {
    nextState: "interrupted",
    interruptReason: "user_reload_or_tab_close",
    canAutoRetry: true,
    queryGatewayStatus: false,
  };
}