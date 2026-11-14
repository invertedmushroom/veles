import type { RunJournalRecord, RunState } from "../protocol/types.ts";
import type { PendingUserTurn } from "./app-client.ts";
import {
  firstToolRecord,
  formatToolTargetHint,
  projectChatRuns,
  projectRunTelemetry,
  type ChatRun,
  type RunTelemetryProjection,
  type SubagentRunProjection,
} from "./chat-timeline.ts";

function formatDurationLabel(durationMs: number): string {
  if (durationMs < 1000) return `${Math.round(durationMs)}ms`;
  return `${(durationMs / 1000).toFixed(1)}s`;
}

function resolveToolStateLabel(
  record: RunJournalRecord,
  dispatched: boolean,
  hasResult: boolean,
  denied: boolean,
  hasTerminalRun: boolean,
): string {
  if (hasResult) return "Saved";
  if (denied) return "Denied";
  if (hasTerminalRun) return "Stopped";
  if (dispatched) return "Running";
  return record.type === "ApprovalRequested" ? "Needs approval" : "Accepted";
}

function ToolActivity({
  record,
  records,
  onInspect,
}: {
  readonly record: RunJournalRecord;
  readonly records: readonly RunJournalRecord[];
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  const related = records.filter(
    (r) =>
      r.runId === record.runId &&
      r.payload.toolCallId === record.payload.toolCallId,
  );
  const result = related.find((r) => r.type === "ToolResultCommitted");
  const dispatched = related.some((r) => r.type === "ToolDispatched");
  const terminal = records.find(
    (r) =>
      r.runId === record.runId &&
      ["RunCancelled", "RunInterrupted", "RunFailed"].includes(r.type),
  );
  const denied = related.some((r) => r.type === "ToolDenied");
  const state = resolveToolStateLabel(
    record,
    dispatched,
    Boolean(result),
    denied,
    Boolean(terminal),
  );
  const targetHint = formatToolTargetHint(record.payload.args);
  return (
    <div className="tool-activity">
      {record.payload.assistantText ? (
        <div className="chat-content">{String(record.payload.assistantText)}</div>
      ) : null}
      <details className="activity-details">
        <summary>
          🔧 {String(record.payload.toolName)}
          {targetHint ? (
            <code className="tool-target-hint">{targetHint}</code>
          ) : null}{" "}
          <span className="hash-badge">{state}</span>
        </summary>
        <div className="activity-body">
          <div className="stage-stepper" aria-label="Tool progress">
            <span className="stage-pip">Accepted</span>
            <span className={dispatched ? "stage-pip" : "stage-pip pending"}>
              Started
            </span>
            <span className={result ? "stage-pip" : "stage-pip pending"}>
              Completed
            </span>
            <span className={result ? "stage-pip" : "stage-pip pending"}>
              Saved by browser
            </span>
          </div>
          <pre className="code-pre">
            {JSON.stringify(record.payload.args ?? {}, null, 2)}
          </pre>
          {result && (
            <pre className="code-pre">
              {String(result.payload.outputText ?? "")}
            </pre>
          )}
          {Array.isArray(result?.payload.imageDataUrls) && (
            <div className="chat-images">
              {result.payload.imageDataUrls.map((url, index) =>
                typeof url === "string" ? (
                  <img
                    key={index}
                    src={url}
                    alt={`Tool result ${index + 1}`}
                    loading="lazy"
                  />
                ) : null,
              )}
            </div>
          )}
          <button
            className="wb-btn"
            type="button"
            onClick={() => onInspect(result ?? record)}
          >
            Inspect journal record
          </button>
        </div>
      </details>
    </div>
  );
}

function resolvePipelineStageClass(
  stage: "route" | "reason" | "tool",
  state: RunState,
  hasRouting: boolean,
  toolCount: number,
): string {
  if (stage === "route") {
    return hasRouting || state !== "created"
      ? "pipeline-step done"
      : "pipeline-step active";
  }
  if (stage === "reason") {
    if (state === "inferring") return "pipeline-step active";
    return toolCount > 0 ? "pipeline-step done" : "pipeline-step";
  }
  const inTool =
    state === "awaiting_tool" ||
    state === "awaiting_approval" ||
    state === "executing_tool";
  if (inTool) return "pipeline-step active";
  return toolCount > 0 ? "pipeline-step done" : "pipeline-step";
}

function LiveHarnessPipelineBar({
  telemetry,
}: {
  readonly telemetry: RunTelemetryProjection;
}) {
  const hasRouting = telemetry.latestRouting !== null;
  return (
    <div
      className="harness-live-pipeline"
      data-testid="harness-live-pipeline"
      aria-label="Active harness pipeline"
    >
      <span
        className={resolvePipelineStageClass(
          "route",
          telemetry.liveState,
          hasRouting,
          telemetry.toolCount,
        )}
      >
        ⚡ 1. Route
      </span>
      <span
        className={resolvePipelineStageClass(
          "reason",
          telemetry.liveState,
          hasRouting,
          telemetry.toolCount,
        )}
      >
        🧠 2. Reason
      </span>
      <span
        className={resolvePipelineStageClass(
          "tool",
          telemetry.liveState,
          hasRouting,
          telemetry.toolCount,
        )}
      >
        🔧 3. Tool / Sandbox
      </span>
      <span className="pipeline-step">💾 4. Commit</span>
    </div>
  );
}

function HarnessTelemetryStrip({
  telemetry,
  onInspect,
}: {
  readonly telemetry: RunTelemetryProjection;
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  const routing = telemetry.latestRouting;
  const anchor = telemetry.telemetryAnchorRecord;
  return (
    <div
      className="harness-telemetry-strip"
      data-testid="harness-telemetry-strip"
      aria-label="Harness routing and run telemetry"
    >
      {routing && (
        <button
          type="button"
          className={`harness-pill router ${routing.source === "tier1_classifier" ? "tier1" : "tier0"}`}
          data-testid="harness-router-pill"
          title={`Configured mode: ${routing.configuredMode}${routing.fallbackReason ? ` (fallback: ${routing.fallbackReason})` : ""}`}
          onClick={() => onInspect(routing.record)}
        >
          <span>{routing.routerLabel}</span>
          {routing.fallbackReason && (
            <span className="harness-fallback-tag">
              {routing.fallbackReason}
            </span>
          )}
        </button>
      )}
      {routing &&
        (routing.specialistIds.length > 0 ? (
          routing.specialistIds.map((specId) => (
            <button
              key={specId}
              type="button"
              className="harness-pill specialist"
              data-testid="harness-specialist-pill"
              onClick={() => onInspect(routing.record)}
            >
              🎯 {specId}
            </button>
          ))
        ) : (
          <span
            className="harness-pill specialist muted"
            data-testid="harness-specialist-pill"
          >
            🎯 base-workspace
          </span>
        ))}
      <button
        type="button"
        className="harness-pill model"
        disabled={!anchor}
        onClick={() => {
          if (anchor) onInspect(anchor);
        }}
      >
        🧠 {telemetry.model}
      </button>
      {telemetry.totalTokens !== null && (
        <span
          className="harness-pill metrics"
          data-testid="harness-usage-pill"
          title={`Prompt: ${telemetry.promptTokens ?? 0} · Completion: ${telemetry.completionTokens ?? 0}`}
        >
          🪙 {telemetry.totalTokens.toLocaleString()} tok
        </span>
      )}
      {telemetry.durationMs !== null && (
        <span
          className="harness-pill metrics"
          data-testid="harness-duration-pill"
        >
          ⏱️ {formatDurationLabel(telemetry.durationMs)}
        </span>
      )}
    </div>
  );
}

function SubagentActivityCard({
  subagent,
  onInspect,
}: {
  readonly subagent: SubagentRunProjection;
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  const toolRecords = subagent.records.filter(
    (r) =>
      (r.type === "ToolIntentLogged" || r.type === "ApprovalRequested") &&
      firstToolRecord(subagent.records, r),
  );
  return (
    <details
      className="activity-details subagent-card"
      data-testid="subagent-card"
    >
      <summary>
        🤖 Subagent <code>{subagent.sessionId}</code>{" "}
        <span className="hash-badge">
          {subagent.model} · {subagent.stateLabel}
          {subagent.totalTokens !== null
            ? ` · ${subagent.totalTokens} tok`
            : ""}
        </span>
      </summary>
      <div className="activity-body">
        <div className="subagent-meta-row">
          <span className="hash-badge">
            Scoped tools:{" "}
            {subagent.allowedTools.length > 0
              ? subagent.allowedTools.join(", ")
              : "inherited"}
          </span>
        </div>
        <div className="chat-content subagent-prompt">{subagent.prompt}</div>
        {toolRecords.map((toolRec) => (
          <ToolActivity
            key={toolRec.eventId}
            record={toolRec}
            records={subagent.records}
            onInspect={onInspect}
          />
        ))}
        {subagent.assistantText && (
          <div className="msg-card assistant">
            <div className="msg-meta">SUBAGENT RESPONSE</div>
            <div className="chat-content">{subagent.assistantText}</div>
          </div>
        )}
        <button
          className="wb-btn"
          type="button"
          onClick={() => onInspect(subagent.createdRecord)}
        >
          Inspect subagent journal
        </button>
      </div>
    </details>
  );
}

function RunEntry({
  record,
  run,
  onInspect,
}: {
  readonly record: RunJournalRecord;
  readonly run: ChatRun;
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  if (record.runId !== run.runId) return null;
  if (
    (record.type === "ToolIntentLogged" ||
      record.type === "ApprovalRequested") &&
    firstToolRecord(run.records, record)
  ) {
    return (
      <ToolActivity record={record} records={run.records} onInspect={onInspect} />
    );
  }
  if (record.type === "RunCompleted") {
    return (
      <div className="msg-card assistant">
        <div className="msg-meta">VELES</div>
        <div className="chat-content">
          {String(record.payload.assistantText ?? "")}
        </div>
      </div>
    );
  }
  if (record.type === "RunCancelled") {
    return (
      <div className="run-outcome">
        Stopped · Previously saved work is kept.
      </div>
    );
  }
  if (record.type === "RunInterrupted") {
    return (
      <div className="run-outcome">
        {record.payload.interruptReason === "uncertain_tool_outcome"
          ? "Stopped with an unresolved tool outcome. Review it before retrying."
          : "Run interrupted. Saved work is kept."}
      </div>
    );
  }
  if (record.type === "RunFailed") {
    return (
      <div className="run-outcome">
        Run failed · Open run details for the recorded error.
      </div>
    );
  }
  return null;
}

function RunConversation({
  run,
  isBusy,
  onDelete,
  onInspect,
}: {
  readonly run: ChatRun;
  readonly isBusy: boolean;
  readonly onDelete: (runId: string) => void;
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  const telemetry = projectRunTelemetry(run);
  return (
    <article className="chat-run" data-run-id={run.runId}>
      <div className="msg-card user">
        <div className="msg-meta">
          <span>YOU</span>
          <button
            className="wb-btn turn-delete"
            disabled={isBusy}
            type="button"
            title="Delete this conversation turn; keep workspace files"
            onClick={() => onDelete(run.runId)}
          >
            ✕ Delete Turn
          </button>
        </div>
        <div className="chat-content">{run.prompt}</div>
      </div>
      <HarnessTelemetryStrip telemetry={telemetry} onInspect={onInspect} />
      {telemetry.isLive && <LiveHarnessPipelineBar telemetry={telemetry} />}
      {run.records.map((record) => (
        <RunEntry
          key={record.eventId}
          record={record}
          run={run}
          onInspect={onInspect}
        />
      ))}
      {telemetry.subagentRuns.map((sub) => (
        <SubagentActivityCard
          key={sub.runId}
          subagent={sub}
          onInspect={onInspect}
        />
      ))}
      <details className="activity-details run-details">
        <summary>
          Run details <span>{run.records.length} events</span>
        </summary>
        <div className="activity-body">
          {run.records.map((record) => (
            <button
              type="button"
              className="journal-link"
              key={record.eventId}
              onClick={() => onInspect(record)}
            >
              <span>
                #{record.seq} {record.type}
              </span>
              <span className="hash-badge">
                {record.runId === run.runId
                  ? record.committedState
                  : "delegated"}
              </span>
            </button>
          ))}
        </div>
      </details>
    </article>
  );
}

export function ChatTimeline({
  journal,
  sessionId,
  pending,
  isBusy,
  onDelete,
  onInspect,
}: {
  readonly journal: readonly RunJournalRecord[];
  readonly sessionId: string;
  readonly pending: PendingUserTurn | null;
  readonly isBusy: boolean;
  readonly onDelete: (runId: string) => void;
  readonly onInspect: (record: RunJournalRecord) => void;
}) {
  const runs = projectChatRuns(journal, sessionId);
  return (
    <>
      {runs.map((run) => (
        <RunConversation
          key={run.runId}
          run={run}
          isBusy={isBusy}
          onDelete={onDelete}
          onInspect={onInspect}
        />
      ))}
      {pending &&
        pending.sessionId === sessionId &&
        !runs.some((run) => run.runId === pending.runId) && (
          <div className="msg-card user pending-submission">
            <div className="msg-meta">
              YOU ·{" "}
              {pending.status === "failed"
                ? "Could not send — your message is kept here"
                : "Sending…"}
            </div>
            <div className="chat-content">{pending.prompt}</div>
          </div>
        )}
      {runs.length === 0 && !pending && (
        <div className="chat-empty">
          <strong>What shall we build?</strong>
          <p>
            Chat with Veles. Your files and run details are a click away in the
            header.
          </p>
        </div>
      )}
    </>
  );
}

