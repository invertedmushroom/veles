import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { PROTOCOL_VERSION, type RunJournalRecord } from "../src/protocol/types.ts";
import {
  isValidStateTransition,
  resolveUnfinishedRunTail,
  validateCommandRequest,
  validateStreamEvent,
} from "../src/protocol/validation.ts";

describe("SYS-PROTO & SYS-KERNEL Protocol Invariants", () => {
  it("enforces valid COMET run state transitions and rejects illegal jumps", () => {
    assert.equal(isValidStateTransition("created", "preparing"), true);
    assert.equal(isValidStateTransition("preparing", "inferring"), true);
    assert.equal(isValidStateTransition("inferring", "awaiting_approval"), true);
    assert.equal(isValidStateTransition("awaiting_approval", "awaiting_tool"), true);
    assert.equal(isValidStateTransition("awaiting_tool", "executing_tool"), true);
    assert.equal(isValidStateTransition("executing_tool", "inferring"), true);
    assert.equal(isValidStateTransition("inferring", "completed"), true);

    // Illegal transitions: cannot jump straight from created -> executing_tool or reopen completed
    assert.equal(isValidStateTransition("created", "executing_tool"), false);
    assert.equal(isValidStateTransition("completed", "inferring"), false);
  });

  it("rejects gateway events that claim browser durability (Stage 4)", () => {
    const fakeStage4 = {
      protocolVersion: PROTOCOL_VERSION,
      operationId: "op-101",
      gatewayInstanceId: "gw-uuid-1",
      seq: 3,
      stage: "COMMITTED_BY_BROWSER",
      timestampMs: Date.now(),
    };
    const res = validateStreamEvent(fakeStage4);
    assert.equal(res.ok, false);
    assert.match(res.error ?? "", /ERR_ILLEGAL_GATEWAY_STAGE/);
  });

  it("blocks path traversal and absolute paths in capabilityScope.writablePaths (Scenario 8)", () => {
    const badReq = {
      protocolVersion: PROTOCOL_VERSION,
      operationId: "op-102",
      runId: "run-1",
      operation: "tool.execute",
      capabilityScope: {
        allowedTools: ["fs.write"],
        allowNetwork: false,
        writablePaths: ["../secrets.txt"],
      },
      limits: { timeoutMs: 5000, maxOutputBytes: 1024 * 64 },
      payload: {},
    };
    const res = validateCommandRequest(badReq);
    assert.equal(res.ok, false);
    assert.match(res.error ?? "", /ERR_UNSAFE_SCOPE/);
  });

  it("prevents blind replay of non-idempotent tools after gateway restart (Scenario 4)", () => {
    const tailRecord: RunJournalRecord = {
      eventId: "ev-9",
      workspaceId: "ws-1",
      runId: "run-1",
      seq: 9,
      type: "ToolDispatched",
      schemaVersion: 1,
      timestampMs: Date.now(),
      relatedOperationId: "op-500",
      committedState: "executing_tool",
      payload: { toolName: "webhook.post" },
    };

    // Gateway restarted (gw-old !== gw-new) on a non-idempotent tool
    const decision = resolveUnfinishedRunTail(
      tailRecord,
      "non_idempotent",
      "gw-instance-old",
      "gw-instance-new",
    );
    assert.equal(decision.nextState, "interrupted");
    assert.equal(decision.interruptReason, "uncertain_tool_outcome");
    assert.equal(decision.canAutoRetry, false);
    assert.equal(decision.queryGatewayStatus, false);
  });

  it("re-queries status when the same gateway instance is still alive after browser reload", () => {
    const tailRecord: RunJournalRecord = {
      eventId: "ev-10",
      workspaceId: "ws-1",
      runId: "run-1",
      seq: 10,
      type: "ToolDispatched",
      schemaVersion: 1,
      timestampMs: Date.now(),
      relatedOperationId: "op-501",
      committedState: "executing_tool",
      payload: { toolName: "sandbox.test" },
    };

    const decision = resolveUnfinishedRunTail(
      tailRecord,
      "non_idempotent",
      "gw-instance-same",
      "gw-instance-same",
    );
    assert.equal(decision.nextState, "executing_tool");
    assert.equal(decision.queryGatewayStatus, true);
    assert.equal(decision.canAutoRetry, false);
  });
});