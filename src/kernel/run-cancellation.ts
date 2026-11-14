import type { RunJournalRecord } from "../protocol/types.ts";

/** True when a journal record belongs to a run's execution lifecycle rather than workspace/session metadata. */
export function isLifecycleJournalRecord(record: RunJournalRecord): boolean {
  return (
    !record.eventId.startsWith("upload-") &&
    !record.eventId.startsWith("session-") &&
    !record.eventId.startsWith("delete-")
  );
}

/** Follow chronological session ownership from this turn, excluding earlier turns. */
export function uncertainRunIds(
  journal: readonly RunJournalRecord[],
  rootRunId: string,
): string[] {
  const start = journal.findIndex(
    (record) => record.runId === rootRunId && record.type === "RunCreated",
  );
  const sessions = new Set<string>();
  const runs = new Set([rootRunId]);
  const tails = new Map<string, RunJournalRecord>();
  for (const record of journal.slice(Math.max(0, start))) {
    if (record.type === "RunCreated") {
      const parent = record.payload.parentSessionId;
      if (typeof parent === "string" && sessions.has(parent)) {
        runs.add(record.runId);
      }
      if (runs.has(record.runId)) {
        sessions.add(String(record.payload.sessionId ?? "session-main"));
      }
    }
    if (runs.has(record.runId) && isLifecycleJournalRecord(record)) {
      tails.set(record.runId, record);
    }
  }
  return [...tails.values()]
    .filter(hasUncertainEffect)
    .map((record) => record.runId);
}

function hasUncertainEffect(record: RunJournalRecord): boolean {
  return (
    (record.committedState === "executing_tool" &&
      record.payload.effectClass === "non_idempotent") ||
    (record.committedState === "interrupted" &&
      record.payload.interruptReason === "uncertain_tool_outcome")
  );
}


