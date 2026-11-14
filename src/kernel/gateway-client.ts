/**
 * @module SYS-KERNEL/gateway-client
 * Browser-side client adapter for the Ephemeral Execution Gateway.
 * Streams SSE events from `/v1/tools/execute`, validates every event with
 * `validateStreamEvent` (blocking Stage-4 spoofing), and queries `/v1/operations/:id`
 * on reconnect to reconcile crash windows (Scenario 4).
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type { ToolDefinitionSpec } from "../inference/openai-compatible.ts";
import {
  PROTOCOL_VERSION,
  type CapabilityScope,
  type GatewayInstanceId,
  type OperationId,
  type ProtocolCommandRequest,
  type ResourceLimits,
  type ToolExecutePayload,
} from "../protocol/types.ts";
import { validateStreamEvent } from "../protocol/validation.ts";
import type {
  ToolExecutionAdapter,
  ToolExecutionOutcome,
  ToolExecutionRequest,
} from "./runtime.ts";

export interface GatewayClientConfig {
  readonly baseUrl: string;
  readonly authToken?: string;
  readonly defaultScope: CapabilityScope;
  readonly defaultLimits: ResourceLimits;
}

export type GatewayFetchLike = (
  url: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
  },
) => Promise<{
  readonly ok: boolean;
  readonly status: number;
  readonly json?: () => Promise<unknown>;
  readonly body?: ReadableStream<Uint8Array> | null;
}>;

export class GatewayToolExecutionAdapter implements ToolExecutionAdapter {
  public currentGatewayInstanceId: GatewayInstanceId | null = null;
  private readonly config: GatewayClientConfig;
  private readonly fetchImpl: GatewayFetchLike;

  constructor(
    config: GatewayClientConfig,
    fetchImpl?: GatewayFetchLike,
    initialGatewayInstanceId: GatewayInstanceId | null = null,
  ) {
    const defaultFetch =
      typeof globalThis.fetch === "function"
        ? ((url: string, init: Parameters<GatewayFetchLike>[1]) =>
            globalThis.fetch(url, init)) as unknown as GatewayFetchLike
        : undefined;
    const resolved = fetchImpl ?? defaultFetch;
    if (!resolved) {
      throw new Error("ERR_FETCH_UNAVAILABLE: globalThis.fetch is required");
    }
    this.config = config;
    this.fetchImpl = resolved;
    this.currentGatewayInstanceId = initialGatewayInstanceId;
  }

  private normalizedBaseUrl(): string {
    return this.config.baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
  }

  private buildHeaders(acceptSse = false): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: acceptSse ? "text/event-stream" : "application/json",
    };
    if (this.config.authToken) {
      headers.Authorization = `Bearer ${this.config.authToken}`;
    }
    return headers;
  }

  public async executeTool(
    request: ToolExecutionRequest,
  ): Promise<ToolExecutionOutcome> {
    const cmd: ProtocolCommandRequest<ToolExecutePayload> = {
      protocolVersion: PROTOCOL_VERSION,
      operationId: request.operationId,
      runId: request.runId,
      operation: "tool.execute",
      capabilityScope: this.config.defaultScope,
      limits: this.config.defaultLimits,
      payload: {
        toolName: request.toolName,
        effectClass: request.effectClass,
        host: "gateway",
        args: request.args,
      },
    };

    const endpoint = `${this.normalizedBaseUrl()}/v1/tools/execute`;
    const resp = await this.fetchImpl(endpoint, {
      method: "POST",
      headers: this.buildHeaders(true),
      body: JSON.stringify(cmd),
      signal: request.signal,
    });

    if (!resp.ok || !resp.body) {
      throw new Error(`ERR_GATEWAY_HTTP_${resp.status}: Tool execution failed`);
    }

    return this.consumeToolSseStream(resp.body, request.operationId);
  }

  private async consumeToolSseStream(
    body: ReadableStream<Uint8Array>,
    expectedOpId: OperationId,
  ): Promise<ToolExecutionOutcome> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalOutcome: ToolExecutionOutcome | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const rawData = line.slice(5).trim();
          if (!rawData || rawData === "[DONE]") continue;
          const completed = this.processToolSseDataLine(rawData, expectedOpId);
          if (completed) {
            finalOutcome = completed;
          }
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
    }

    if (!finalOutcome) {
      throw new Error("ERR_GATEWAY_STREAM_INCOMPLETE: Missing COMPLETED stage");
    }
    return finalOutcome;
  }

  private processToolSseDataLine(
    rawData: string,
    expectedOpId: OperationId,
  ): ToolExecutionOutcome | null {
    const parsed = JSON.parse(rawData) as {
      event?: unknown;
      outcome?: {
        outputText?: string;
        fileUpdates?: Record<string, string>;
      };
    };
    const check = validateStreamEvent(parsed.event);
    if (!check.ok || !check.value) {
      throw new Error(check.error ?? "ERR_INVALID_GATEWAY_STREAM_EVENT");
    }
    if (check.value.operationId !== expectedOpId) {
      throw new Error("ERR_OPERATION_ID_MISMATCH");
    }
    this.currentGatewayInstanceId = check.value.gatewayInstanceId;

    if (check.value.errorCode) {
      throw new Error(check.value.errorCode);
    }
    if (check.value.stage !== "COMPLETED") {
      return null;
    }
    return {
      gatewayInstanceId: check.value.gatewayInstanceId,
      outputText: parsed.outcome?.outputText ?? check.value.deltaText ?? "",
      fileUpdates: parsed.outcome?.fileUpdates,
    };
  }

  public async queryOperationStatus(
    operationId: OperationId,
    expectedGatewayInstanceId: GatewayInstanceId,
  ): Promise<{ readonly completed: boolean; readonly outcome?: ToolExecutionOutcome }> {
    const endpoint = `${this.normalizedBaseUrl()}/v1/operations/${encodeURIComponent(operationId)}`;
    const resp = await this.fetchImpl(endpoint, {
      method: "GET",
      headers: this.buildHeaders(false),
    });
    if (!resp.ok || !resp.json) {
      return { completed: false };
    }
    const data = (await resp.json()) as {
      found?: boolean;
      gatewayInstanceId?: string;
      completed?: boolean;
      outcome?: {
        outputText?: string;
        fileUpdates?: Record<string, string>;
      };
    };
    if (
      !data.found ||
      data.gatewayInstanceId !== expectedGatewayInstanceId ||
      !data.completed ||
      !data.outcome
    ) {
      return { completed: false };
    }
    return {
      completed: true,
      outcome: {
        gatewayInstanceId: data.gatewayInstanceId,
        outputText: data.outcome.outputText ?? "",
        fileUpdates: data.outcome.fileUpdates,
      },
    };
  }

  public async fetchCapabilities(): Promise<ToolDefinitionSpec[]> {
    const endpoint = `${this.normalizedBaseUrl()}/v1/capabilities`;
    try {
      const resp = await this.fetchImpl(endpoint, {
        method: "GET",
        headers: this.buildHeaders(false),
      });
      if (!resp.ok || !resp.json) return [];
      const data = (await resp.json()) as {
        gatewayInstanceId?: string;
        toolDefinitions?: ToolDefinitionSpec[];
      };
      if (typeof data.gatewayInstanceId === "string") {
        this.currentGatewayInstanceId = data.gatewayInstanceId;
      }
      return Array.isArray(data.toolDefinitions) ? data.toolDefinitions : [];
    } catch {
      return [];
    }
  }
}