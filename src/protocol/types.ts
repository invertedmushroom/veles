/**
 * @module SYS-PROTO/types
 * Pure, zero-dependency wire protocol, state machine, and journal types
 * shared between the Browser Web Worker Kernel and the Ephemeral Execution Gateway.
 */

export const PROTOCOL_VERSION = "1.0" as const;
export type ProtocolVersion = typeof PROTOCOL_VERSION;

export type WorkspaceId = string;
export type RunId = string;
export type EventId = string;
export type OperationId = string;
export type GatewayInstanceId = string;
export type Sha256Hex = string;

/**
 * Side-effect classification for every tool.
 * Governs deterministic crash recovery when the browser or gateway disconnects mid-execution.
 */
export type ToolEffectClass = "pure_read" | "idempotent_write" | "non_idempotent";

export type ToolExecutionHost = "browser" | "gateway";

/**
 * COMET Run Lifecycle States (Section 5 of Architecture Plan + Crash-Window Guard).
 */
export type RunState =
  | "created"
  | "preparing"
  | "inferring"
  | "awaiting_approval"
  | "awaiting_tool"
  | "executing_tool"
  | "completed"
  | "cancelled"
  | "failed"
  | "interrupted";

export type InterruptReason =
  | "user_reload_or_tab_close"
  | "gateway_disconnected_safe_to_retry"
  | "uncertain_tool_outcome";

/**
 * 4-Stage Acknowledgement Contract (Section 7 of Architecture Plan).
 * Stage 4 (COMMITTED_BY_BROWSER) is recorded only by the Browser Kernel, never claimed by the Gateway.
 */
export type AckStage =
  | "ACCEPTED"
  | "STARTED"
  | "COMPLETED"
  | "COMMITTED_BY_BROWSER";

export type ProtocolOperationType =
  | "inference.start"
  | "inference.cancel"
  | "tool.execute"
  | "tool.cancel"
  | "execution.status"
  | "capabilities.list";

export interface ResourceLimits {
  readonly timeoutMs: number;
  readonly maxOutputBytes: number;
  readonly maxMemoryMb?: number;
}

export interface CapabilityScope {
  readonly allowedTools: readonly string[];
  readonly allowNetwork: boolean;
  readonly writablePaths: readonly string[];
}

export interface ProtocolCommandRequest<TPayload = unknown> {
  readonly protocolVersion: ProtocolVersion;
  readonly operationId: OperationId;
  readonly runId: RunId;
  readonly operation: ProtocolOperationType;
  readonly capabilityScope: CapabilityScope;
  readonly limits: ResourceLimits;
  readonly payload: TPayload;
}

export interface ToolExecutePayload {
  readonly toolName: string;
  readonly effectClass: ToolEffectClass;
  readonly host: ToolExecutionHost;
  readonly args: Record<string, unknown>;
}

export interface ExecutionStreamEvent {
  readonly protocolVersion: ProtocolVersion;
  readonly operationId: OperationId;
  readonly gatewayInstanceId: GatewayInstanceId;
  readonly seq: number;
  readonly stage: Exclude<AckStage, "COMMITTED_BY_BROWSER">;
  readonly timestampMs: number;
  readonly deltaText?: string;
  readonly exitCode?: number;
  readonly outputSha256?: Sha256Hex;
  readonly errorCode?: string;
}

/**
 * Durable Browser Run Journal Event Types (Section 6 of Architecture Plan).
 */
export type JournalEventType =
  | "RunCreated"
  | "RunPreparing"
  | "CheckpointCommitted"
  | "ApprovalRequested"
  | "ToolDenied"
  | "ToolIntentLogged"
  | "ToolDispatched"
  | "ToolResultCommitted"
  | "RunInterrupted"
  | "RunCompleted"
  | "RunCancelled"
  | "RunFailed";

export interface RunJournalRecord<TPayload = Record<string, unknown>> {
  readonly eventId: EventId;
  readonly workspaceId: WorkspaceId;
  readonly runId: RunId;
  readonly seq: number;
  readonly type: JournalEventType;
  readonly schemaVersion: 1;
  readonly timestampMs: number;
  readonly relatedOperationId: OperationId | null;
  readonly committedState: RunState;
  readonly payload: TPayload;
}

/**
 * Pure Data-Coupled Worker <-> UI MessagePort wire contracts (Milestone 5a).
 * Ensures `src/ui/**` communicates with the Web Worker Kernel strictly over MessagePort
 * without importing storage or runtime internals (`!INVARIANT [Coupling_Cohesion_Law]`).
 */
export interface ProviderModelConfigSpec {
  readonly providerId: string;
  readonly baseUrl: string;
  readonly model: string;
  readonly apiKey?: string;
}

export type DecisionServiceMode =
  | "tier0-only"
  | "typesafe"
  | "openai-decisions"
  | "local-laya";

export type ImageServiceMode =
  | "openai-auto"
  | "openai-dalle3"
  | "openai-gpt-svg"
  | "local-deterministic-svg";

export type WebSearchServiceMode =
  | "openai-responses"
  | "searxng"
  | "offline-disabled";

export interface DecisionServiceConfigSpec {
  readonly mode: DecisionServiceMode;
  readonly endpointUrl: string;
  readonly model: string;
  readonly apiKey?: string;
  readonly threshold?: number;
  readonly timeoutMs?: number;
}

export interface ServiceCapabilitiesConfigSpec {
  readonly decision: DecisionServiceConfigSpec;
  readonly imageMode: ImageServiceMode;
  readonly webSearchMode: WebSearchServiceMode;
  readonly searxngUrl?: string;
}

export const TYPESAFE_SYSTEM_ONE_ENDPOINT = "https://api.typesafe.ai/v1/systemone";

export const DEFAULT_SERVICE_CAPABILITIES_CONFIG: ServiceCapabilitiesConfigSpec = {
  decision: {
    mode: "tier0-only",
    endpointUrl: TYPESAFE_SYSTEM_ONE_ENDPOINT,
    model: "jev-1.14.0",
    threshold: 0.75,
    timeoutMs: 800,
  },
  imageMode: "openai-auto",
  webSearchMode: "openai-responses",
  searxngUrl: "http://127.0.0.1:8080/search",
};

export interface ChatTurnMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content: string;
  readonly runId?: RunId;
  readonly imageDataUrls?: readonly string[];
  readonly toolCallId?: string;
  readonly toolCalls?: readonly {
    readonly id: string;
    readonly name: string;
    readonly argumentsJson: string;
  }[];
}

export interface PendingApprovalSummary {
  readonly runId: RunId;
  readonly operationId: OperationId | null;
  readonly toolCallId: string;
  readonly toolName: string;
  readonly effectClass: ToolEffectClass;
  readonly args: Record<string, unknown>;
  readonly assistantText: string;
}

export interface WorkspaceSummary {
  readonly workspaceId: WorkspaceId;
  readonly title: string;
  readonly headSeq: number;
  readonly updatedAtMs: number;
  readonly runState?: RunState | null;
}

export type AgentSessionStatus =
  | "idle"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface AgentSessionSpec {
  readonly id: string;
  readonly title?: string;
  readonly parentSessionId?: string;
  readonly workspaceId: WorkspaceId;
  readonly model: ProviderModelConfigSpec;
  readonly systemPrompt?: string;
  readonly allowedTools?: readonly string[];
  readonly delegatableTools?: readonly string[];
  readonly unrestrictedTools?: boolean;
  readonly status: AgentSessionStatus;
  readonly activeRunId?: RunId | null;
}

export interface WorkspaceStateSnapshot {
  readonly workspaceId: WorkspaceId;
  readonly title: string;
  readonly headSeq: number;
  readonly activeRunId: RunId | null;
  readonly runState: RunState | null;
  readonly updatedAtMs: number;
  readonly leaseMode: "ReadWrite_Leader" | "ReadOnly_Observer";
  readonly gatewayInstanceId: GatewayInstanceId | null;
  readonly interruptReason: InterruptReason | null;
  readonly files: Readonly<Record<string, string>>;
  readonly fileHashes: Readonly<Record<string, Sha256Hex>>;
  readonly messages: readonly ChatTurnMessage[];
  readonly pendingApproval: PendingApprovalSummary | null;
  readonly journal: readonly RunJournalRecord[];
  readonly workspaces?: readonly WorkspaceSummary[];
  readonly sessions?: readonly AgentSessionSpec[];
  readonly activeSessionId?: string;
  readonly serviceConfig?: ServiceCapabilitiesConfigSpec;
}

export interface WorkerPortLike {
  postMessage(message: unknown): void;
  addEventListener?(
    type: "message",
    listener: (ev: { readonly data: unknown }) => void,
  ): void;
  removeEventListener?(
    type: "message",
    listener: (ev: { readonly data: unknown }) => void,
  ): void;
  start?(): void;
  close?(): void;
}

export type UiToWorkerRequestPayload =
  | { readonly type: "INIT_WORKSPACE"; readonly title: string }
  | { readonly type: "GET_SNAPSHOT" }
  | {
      readonly type: "START_RUN";
      readonly runId: RunId;
      readonly prompt: string;
      readonly providerConfig: ProviderModelConfigSpec;
      readonly sessionId?: string;
      readonly serviceConfig?: ServiceCapabilitiesConfigSpec;
    }
  | {
      readonly type: "UPDATE_SERVICE_CONFIG";
      readonly serviceConfig: ServiceCapabilitiesConfigSpec;
    }
  | {
      readonly type: "CREATE_SESSION";
      readonly session: AgentSessionSpec;
    }
  | { readonly type: "CANCEL_RUN"; readonly runId: RunId }
  | {
      readonly type: "RESOLVE_APPROVAL";
      readonly runId: RunId;
      readonly approved: boolean;
    }
  | {
      readonly type: "RECOVER_BOOT";
      readonly currentGatewayInstanceId: GatewayInstanceId | null;
    }
  | {
      readonly type: "RESOLVE_UNCERTAIN_TOOL";
      readonly runId: RunId;
      readonly resolution: "mark_succeeded" | "retry_anyway" | "cancel_run";
    }
  | {
      readonly type: "SIMULATE_CHAOS";
      readonly scenario:
        | "quota_exceeded"
        | "drop_stream_mid_inference"
        | "uncertain_tool_outcome";
    }
  | { readonly type: "EXPORT_BUNDLE" }
  | { readonly type: "IMPORT_BUNDLE"; readonly bundle: unknown }
  | {
      readonly type: "SWITCH_WORKSPACE";
      readonly workspaceId: WorkspaceId;
      readonly title?: string;
    }
  | {
      readonly type: "DELETE_WORKSPACE";
      readonly workspaceId: WorkspaceId;
    }
  | {
      readonly type: "WRITE_WORKSPACE_FILE";
      readonly path: string;
      readonly content: string | Uint8Array;
    }
  | {
      readonly type: "SWITCH_SESSION";
      readonly sessionId: string;
      readonly title?: string;
    }
  | {
      readonly type: "DELETE_RUN";
      readonly runId: RunId;
      readonly revertFiles?: boolean;
    };

export interface ContextEconomicsSummary {
  readonly systemContractChars: number;
  readonly specialistCapsuleChars: number;
  readonly workspaceTreeChars: number;
  readonly conversationChars: number;
  readonly toolIOChars: number;
  readonly imageWeightChars: number;
  readonly totalChars: number;
  readonly estimatedTokens: number;
  readonly cacheablePrefixTokens: number;
  readonly foldedSavedChars: number;
  readonly hotRunCount: number;
  readonly coldFoldedRunCount: number;
}

export interface UiToWorkerMessage {
  readonly channel: "veles:worker-port";
  readonly requestId: string;
  readonly request: UiToWorkerRequestPayload;
}

export type WorkerToUiMessage =
  | {
      readonly channel: "veles:worker-port";
      readonly kind: "RESPONSE";
      readonly requestId: string;
      readonly ok: boolean;
      readonly snapshot?: WorkspaceStateSnapshot;
      readonly bundle?: unknown;
      readonly recoveryDecision?: {
        readonly nextState: RunState;
        readonly interruptReason?: InterruptReason;
        readonly canAutoRetry: boolean;
        readonly queryGatewayStatus: boolean;
      } | null;
      readonly error?: string;
    }
  | {
      readonly channel: "veles:worker-port";
      readonly kind: "EVENT";
      readonly eventType: "STATE_CHANGED" | "STREAM_DELTA" | "RECOVERY_RESOLVED" | "ERROR" | "JOURNAL_COMMITTED";
      readonly runId: RunId | null;
      readonly state: RunState | null;
      readonly detail?: Record<string, unknown>;
      readonly snapshot?: WorkspaceStateSnapshot;
      readonly record?: RunJournalRecord;
    };
