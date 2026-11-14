/**
 * @module SYS-INFERENCE/openai-compatible
 * Pure streaming adapter normalizing OpenAI-compatible SSE chunks into
 * `{ deltaText, toolCalls, usage, finishReason }` with `AbortSignal` cancellation
 * and per-run provider/model isolation (Scenario 9).
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type { ToolEffectClass } from "../protocol/types.ts";

export type ChatRole = "system" | "user" | "assistant" | "tool";

export interface ToolCallSpec {
  readonly id: string;
  readonly name: string;
  readonly argumentsJson: string;
}

export type ChatContentPart =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "image_url"; readonly image_url: { readonly url: string } };

export interface ChatMessage {
  readonly role: ChatRole;
  readonly content: string;
  readonly runId?: string;
  readonly imageDataUrls?: readonly string[];
  readonly toolCallId?: string;
  readonly toolCalls?: readonly ToolCallSpec[];
}

export interface ToolDefinitionSpec {
  readonly name: string;
  readonly description: string;
  readonly effectClass: ToolEffectClass;
  readonly requiresApproval?: boolean;
  readonly parametersSchema?: Record<string, unknown>;
}

export interface ProviderModelConfig {
  readonly providerId: string;
  readonly baseUrl: string;
  readonly model: string;
  readonly apiKey?: string;
}

export interface TokenUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

export type NormalizedFinishReason =
  | "stop"
  | "tool_calls"
  | "length"
  | "content_filter"
  | null;

export interface NormalizedStreamChunk {
  readonly providerId: string;
  readonly model: string;
  readonly deltaText: string;
  readonly toolCalls: readonly ToolCallSpec[];
  readonly usage: TokenUsage | null;
  readonly finishReason: NormalizedFinishReason;
}

export interface InferenceStreamRequest {
  readonly config: ProviderModelConfig;
  readonly messages: readonly ChatMessage[];
  readonly tools?: readonly ToolDefinitionSpec[];
  readonly signal?: AbortSignal;
}

export interface InferenceAdapter {
  streamChat(
    request: InferenceStreamRequest,
  ): AsyncIterable<NormalizedStreamChunk>;
}

export type FetchLike = (
  url: string,
  init: {
    method: string;
    headers: Record<string, string>;
    body: string;
    signal?: AbortSignal;
  },
) => Promise<{
  readonly ok: boolean;
  readonly status: number;
  readonly body: ReadableStream<Uint8Array> | null;
  readonly text?: () => Promise<string>;
}>;

interface RawToolCallDelta {
  readonly index?: number;
  readonly id?: string;
  readonly function?: {
    readonly name?: string;
    readonly arguments?: string;
  };
}

interface MutableToolCallAccumulator {
  id: string;
  name: string;
  argumentsJson: string;
}

function normalizeFinishReason(raw: unknown): NormalizedFinishReason {
  if (
    raw === "stop" ||
    raw === "tool_calls" ||
    raw === "length" ||
    raw === "content_filter"
  ) {
    return raw;
  }
  return null;
}

function normalizeUsage(raw: unknown): TokenUsage | null {
  if (typeof raw !== "object" || raw === null) return null;
  const u = raw as Record<string, unknown>;
  const promptTokens =
    typeof u.prompt_tokens === "number" ? u.prompt_tokens : 0;
  const completionTokens =
    typeof u.completion_tokens === "number" ? u.completion_tokens : 0;
  const totalTokens =
    typeof u.total_tokens === "number"
      ? u.total_tokens
      : promptTokens + completionTokens;
  return { promptTokens, completionTokens, totalTokens };
}

function toWireToolName(name: string): string {
  return name.replaceAll(".", "__");
}

function fromWireToolName(
  wireName: string,
  tools?: readonly ToolDefinitionSpec[],
): string {
  if (!tools) return wireName;
  const match = tools.find((t) => toWireToolName(t.name) === wireName);
  return match ? match.name : wireName;
}

function mergeToolCallDeltas(
  accumulators: Map<number, MutableToolCallAccumulator>,
  deltas: unknown,
  tools?: readonly ToolDefinitionSpec[],
): ToolCallSpec[] {
  if (!Array.isArray(deltas) || deltas.length === 0) return [];
  for (let i = 0; i < deltas.length; i++) {
    const d = deltas[i] as RawToolCallDelta | undefined;
    if (!d || typeof d !== "object") continue;
    const idx = typeof d.index === "number" ? d.index : i;
    const acc = accumulators.get(idx) ?? {
      id: d.id ?? `call_${idx}`,
      name: "",
      argumentsJson: "",
    };
    if (typeof d.id === "string" && d.id.length > 0) acc.id = d.id;
    if (typeof d.function?.name === "string") acc.name += d.function.name;
    if (typeof d.function?.arguments === "string") {
      acc.argumentsJson += d.function.arguments;
    }
    accumulators.set(idx, acc);
  }
  return Array.from(accumulators.values()).map((acc) => ({
    id: acc.id,
    name: fromWireToolName(acc.name, tools),
    argumentsJson: acc.argumentsJson,
  }));
}

function parseSseDataPayload(
  dataLine: string,
  config: ProviderModelConfig,
  toolAccumulators: Map<number, MutableToolCallAccumulator>,
  tools?: readonly ToolDefinitionSpec[],
): NormalizedStreamChunk | "DONE" | null {
  const trimmed = dataLine.trim();
  if (!trimmed || trimmed === "[DONE]") {
    return trimmed === "[DONE]" ? "DONE" : null;
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(trimmed) as Record<string, unknown>;
  } catch {
    throw new Error("ERR_INFERENCE_MALFORMED_SSE: Invalid JSON in SSE data line");
  }

  const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
  const firstChoice = (choices[0] ?? {}) as Record<string, unknown>;
  const delta = (firstChoice.delta ?? {}) as Record<string, unknown>;
  const deltaText = typeof delta.content === "string" ? delta.content : "";
  const toolCalls = mergeToolCallDeltas(
    toolAccumulators,
    delta.tool_calls,
    tools,
  );
  const finishReason = normalizeFinishReason(firstChoice.finish_reason);
  const usage = normalizeUsage(parsed.usage);

  return {
    providerId: config.providerId,
    model: config.model,
    deltaText,
    toolCalls,
    usage,
    finishReason,
  };
}

function decodeSvgDataUrl(url: string): string | null {
  if (url.startsWith("data:image/svg+xml;utf8,")) {
    try {
      return decodeURIComponent(url.slice("data:image/svg+xml;utf8,".length));
    } catch {
      return url.slice("data:image/svg+xml;utf8,".length);
    }
  }
  if (url.startsWith("data:image/svg+xml;base64,")) {
    try {
      return globalThis.atob(url.slice("data:image/svg+xml;base64,".length));
    } catch {
      return null;
    }
  }
  return null;
}

function isRasterVisionImageUrl(url: string): boolean {
  if (url.startsWith("https://") || url.startsWith("http://")) return true;
  if (!/^data:image\/(png|jpeg|jpg|webp|gif);base64,/i.test(url)) return false;
  const b64Payload = url.slice(url.indexOf(",") + 1);
  return b64Payload.length > 0 && !b64Payload.startsWith("PHN2Zy");
}

function serializeMessageContent(m: ChatMessage): string | readonly ChatContentPart[] {
  if (!m.imageDataUrls || m.imageDataUrls.length === 0) {
    return m.content;
  }
  const parts: ChatContentPart[] = [{ type: "text", text: m.content }];
  for (const url of m.imageDataUrls) {
    if (typeof url !== "string") continue;
    const svgSource = decodeSvgDataUrl(url);
    if (svgSource) {
      parts.push({
        type: "text",
        text: `[Attached SVG Image Markup]\n${svgSource}`,
      });
    } else if (isRasterVisionImageUrl(url)) {
      parts.push({ type: "image_url", image_url: { url } });
    }
  }
  return parts;
}

function buildChatCompletionsBody(request: InferenceStreamRequest): string {
  const payload: Record<string, unknown> = {
    model: request.config.model,
    stream: true,
    messages: request.messages.map((m) => ({
      role: m.role,
      content: serializeMessageContent(m),
      ...(m.toolCallId ? { tool_call_id: m.toolCallId } : {}),
      ...(m.toolCalls && m.toolCalls.length > 0
        ? {
            tool_calls: m.toolCalls.map((tc) => ({
              id: tc.id,
              type: "function",
              function: {
                name: toWireToolName(tc.name),
                arguments: tc.argumentsJson,
              },
            })),
          }
        : {}),
    })),
  };
  if (request.config.providerId === "openai") {
    payload.stream_options = { include_usage: true };
  }
  if (request.tools && request.tools.length > 0) {
    payload.tools = request.tools.map((t) => ({
      type: "function",
      function: {
        name: toWireToolName(t.name),
        description: `${t.description} (Veles tool alias: ${t.name})`,
        parameters: t.parametersSchema ?? { type: "object", properties: {} },
      },
    }));
  }
  return JSON.stringify(payload);
}

function buildHeaders(config: ProviderModelConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  if (config.apiKey) {
    headers.Authorization = `Bearer ${config.apiKey}`;
  }
  return headers;
}

export class OpenAICompatibleAdapter implements InferenceAdapter {
  private readonly fetchImpl: FetchLike;

  constructor(fetchImpl?: FetchLike) {
    const defaultFetch =
      typeof globalThis.fetch === "function"
        ? ((url: string, init: Parameters<FetchLike>[1]) =>
            globalThis.fetch(url, init)) as unknown as FetchLike
        : undefined;
    const resolved = fetchImpl ?? defaultFetch;
    if (!resolved) {
      throw new Error("ERR_FETCH_UNAVAILABLE: globalThis.fetch is required");
    }
    this.fetchImpl = resolved;
  }

  public async *streamChat(
    request: InferenceStreamRequest,
  ): AsyncIterable<NormalizedStreamChunk> {
    // Snapshot config per invocation so concurrent or subsequent provider switches
    // cannot mutate an in-flight stream (Scenario 9).
    const frozenConfig: ProviderModelConfig = { ...request.config };
    if (request.signal?.aborted) {
      throw new Error("ERR_INFERENCE_ABORTED: Request cancelled before start");
    }

    const endpoint = `${frozenConfig.baseUrl.replace(/\/+$/, "")}/chat/completions`;
    const response = await this.fetchImpl(endpoint, {
      method: "POST",
      headers: buildHeaders(frozenConfig),
      body: buildChatCompletionsBody({ ...request, config: frozenConfig }),
      signal: request.signal,
    });

    if (!response.ok || !response.body) {
      let detail = "Stream failed";
      if (typeof response.text === "function") {
        try {
          const errText = (await response.text()).trim();
          if (errText) detail = errText.slice(0, 240);
        } catch {
          // ignore body read error
        }
      }
      throw new Error(`ERR_INFERENCE_HTTP_${response.status}: ${detail}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const toolAccumulators = new Map<number, MutableToolCallAccumulator>();
    let buffer = "";

    try {
      while (true) {
        if (request.signal?.aborted) {
          throw new Error("ERR_INFERENCE_ABORTED: Stream cancelled by AbortSignal");
        }
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const item = parseSseDataPayload(
            line.slice(5),
            frozenConfig,
            toolAccumulators,
            request.tools,
          );
          if (item === "DONE") return;
          if (item !== null) yield item;
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
    }
  }
}
