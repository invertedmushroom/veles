/**
 * @module SYS-KERNEL/tools
 * Effect-classified browser tool registry, local OPFS workspace tools,
 * multimodal image inspection (`workspace.inspect_image`), and parent-to-child
 * subagent session delegation (`agent.delegate`).
 * Enforces runtime input validation, relative path confinement (Scenario 8),
 * and atomic staging so failed tools never corrupt committed workspace state.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type {
  ProviderModelConfig,
  ToolDefinitionSpec,
} from "../inference/openai-compatible.ts";
import type {
  GatewayInstanceId,
  ImageServiceMode,
  OperationId,
  RunId,
  ServiceCapabilitiesConfigSpec,
  ToolEffectClass,
  WebSearchServiceMode,
} from "../protocol/types.ts";
import type { AgentManager, AgentSession } from "./agent-session.ts";
import { base64ToBytes, bytesToBase64 } from "./blobs.ts";
import type { WorkspaceKernelStore } from "./journal.ts";
import type {
  ToolExecutionAdapter,
  ToolExecutionOutcome,
  ToolExecutionRequest,
} from "./runtime.ts";
import {
  formatPreflightDiagnosticSuffix,
  verifyWorkspaceFilePreflight,
} from "./workspace-intelligence.ts";

export const LOCAL_BROWSER_INSTANCE_ID: GatewayInstanceId = "browser-local-v1";

export interface LocalToolExecutionContext {
  readonly runId: RunId;
}

export type LocalToolHandler = (
  args: Record<string, unknown>,
  store: WorkspaceKernelStore,
  signal?: AbortSignal,
  context?: LocalToolExecutionContext,
) => Promise<{
  readonly outputText: string;
  readonly fileUpdates?: Readonly<Record<string, Uint8Array | string | null>>;
  readonly imageDataUrls?: readonly string[];
}>;

export interface RegisteredBrowserTool {
  readonly spec: ToolDefinitionSpec;
  readonly handler: LocalToolHandler;
}

export function assertSafeWorkspacePath(rawPath: unknown): string {
  if (typeof rawPath !== "string" || !rawPath.trim()) {
    throw new Error("ERR_TOOL_INVALID_PATH: Path must be a non-empty string");
  }
  const normalized = rawPath.trim();
  if (
    normalized.startsWith("/") ||
    normalized.includes("\\") ||
    normalized.split("/").some((part) => part === ".." || part === "." || part.length === 0)
  ) {
    throw new Error(
      `ERR_TOOL_UNSAFE_PATH: Path '${rawPath}' must be a clean relative workspace path`,
    );
  }
  return normalized;
}

const readWorkspaceFileTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.read_file",
    description: "Read a UTF-8 text file from the committed browser workspace.",
    effectClass: "pure_read",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  async handler(args, store) {
    const safePath = assertSafeWorkspacePath(args.path);
    const content = await store.readFileText(safePath);
    return { outputText: content };
  },
};

const inspectWorkspaceImageTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.inspect_image",
    description:
      "Load an image (.png, .jpg, .webp, .gif, .svg) from the browser workspace into multimodal vision context so you can see and describe or analyze it.",
    effectClass: "pure_read",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  async handler(args, store) {
    const safePath = assertSafeWorkspacePath(args.path);
    const rawOrDataUrl = await store.readFileTextOrDataUrl(safePath);
    const isSvg =
      safePath.toLowerCase().endsWith(".svg") ||
      rawOrDataUrl.trim().startsWith("<svg");
    const dataUrl = rawOrDataUrl.startsWith("data:image/")
      ? rawOrDataUrl
      : isSvg
        ? `data:image/svg+xml;utf8,${encodeURIComponent(rawOrDataUrl)}`
        : null;
    if (!dataUrl) {
      throw new Error(
        `ERR_NOT_AN_IMAGE: '${safePath}' is not a supported image artifact`,
      );
    }
    return {
      outputText: `Loaded image '${safePath}' into multimodal vision context (${dataUrl.length} chars).`,
      imageDataUrls: [dataUrl],
    };
  },
};

const listWorkspaceFilesTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.list_files",
    description: "List all file paths in the committed browser workspace.",
    effectClass: "pure_read",
    requiresApproval: false,
    parametersSchema: { type: "object", properties: {} },
  },
  async handler(_args, store) {
    const paths = await store.listFiles();
    return { outputText: JSON.stringify(paths) };
  },
};

const writeWorkspaceFileTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.write_file",
    description:
      "Create a new UTF-8 file or overwrite an entire file in the browser workspace. Prefer workspace.edit_file when making targeted changes to an existing file.",
    effectClass: "idempotent_write",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      required: ["path", "content"],
    },
  },
  async handler(args) {
    const safePath = assertSafeWorkspacePath(args.path);
    if (typeof args.content !== "string") {
      throw new Error("ERR_TOOL_INVALID_ARGS: 'content' must be a string");
    }
    const preflightSuffix = formatPreflightDiagnosticSuffix(
      verifyWorkspaceFilePreflight(safePath, args.content),
    );
    return {
      outputText: `Updated ${safePath} (${args.content.length} chars)${preflightSuffix}`,
      fileUpdates: { [safePath]: args.content },
    };
  },
};

export interface WorkspaceFileEditOperation {
  readonly target: string;
  readonly replacement: string;
  readonly replaceAll?: boolean;
}

function normalizeLineEndings(text: string): string {
  return text.replaceAll("\r\n", "\n");
}

function parseSingleEditEntry(
  entry: unknown,
  index: number,
): WorkspaceFileEditOperation {
  if (typeof entry !== "object" || entry === null) {
    throw new Error(
      `ERR_EDIT_INVALID_ARGS: edit #${index + 1} must be an object with 'target' and 'replacement'`,
    );
  }
  const record = entry as Record<string, unknown>;
  if (typeof record.target !== "string" || record.target.length === 0) {
    throw new Error(
      `ERR_EDIT_INVALID_ARGS: edit #${index + 1} 'target' must be a non-empty string`,
    );
  }
  if (typeof record.replacement !== "string") {
    throw new Error(
      `ERR_EDIT_INVALID_ARGS: edit #${index + 1} 'replacement' must be a string`,
    );
  }
  return {
    target: record.target,
    replacement: record.replacement,
    replaceAll: record.replaceAll === true,
  };
}

function parseFileEditOperations(
  args: Record<string, unknown>,
): WorkspaceFileEditOperation[] {
  const ops: WorkspaceFileEditOperation[] = [];
  if (typeof args.target === "string" || args.replacement !== undefined) {
    ops.push(parseSingleEditEntry(args, 0));
  }
  if (Array.isArray(args.edits)) {
    for (let i = 0; i < args.edits.length; i++) {
      ops.push(parseSingleEditEntry(args.edits[i], ops.length));
    }
  }
  if (ops.length === 0) {
    throw new Error(
      "ERR_EDIT_INVALID_ARGS: Provide either ('target' and 'replacement') or a non-empty 'edits' array",
    );
  }
  return ops;
}

function countSubstringOccurrences(haystack: string, needle: string): number {
  let count = 0;
  let pos = 0;
  while (pos <= haystack.length - needle.length) {
    const found = haystack.indexOf(needle, pos);
    if (found === -1) break;
    count += 1;
    pos = found + needle.length;
  }
  return count;
}

function linesMatchTrimmed(
  contentLines: readonly string[],
  startLine: number,
  targetLines: readonly string[],
): boolean {
  for (let i = 0; i < targetLines.length; i++) {
    if (contentLines[startLine + i]!.trim() !== targetLines[i]!.trim()) {
      return false;
    }
  }
  return true;
}

function findLineTrimmedMatchRange(
  content: string,
  target: string,
): { readonly start: number; readonly end: number; readonly matchCount: number } | null {
  const targetLines = target.split("\n");
  if (!targetLines.some((line) => line.trim().length > 0)) {
    return null;
  }
  const contentLines = content.split("\n");
  if (targetLines.length > contentLines.length) {
    return null;
  }
  const lineOffsets: number[] = [];
  let offset = 0;
  for (const line of contentLines) {
    lineOffsets.push(offset);
    offset += line.length + 1;
  }
  let firstStart = -1;
  let firstEnd = -1;
  let matchCount = 0;
  const maxStart = contentLines.length - targetLines.length;
  for (let lineIdx = 0; lineIdx <= maxStart; lineIdx++) {
    if (linesMatchTrimmed(contentLines, lineIdx, targetLines)) {
      matchCount += 1;
      if (firstStart === -1) {
        firstStart = lineOffsets[lineIdx]!;
        const endLineIdx = lineIdx + targetLines.length - 1;
        firstEnd = lineOffsets[endLineIdx]! + contentLines[endLineIdx]!.length;
      }
    }
  }
  return matchCount > 0
    ? { start: firstStart, end: firstEnd, matchCount }
    : null;
}

function scanCollapsedLineSpan(
  contentLines: readonly string[],
  startLine: number,
  collapsedTarget: string,
): number {
  const firstNorm = contentLines[startLine]!.replace(/\s+/g, " ").trim();
  if (!firstNorm || !collapsedTarget.startsWith(firstNorm)) return -1;
  let acc = "";
  const maxEnd = Math.min(contentLines.length - 1, startLine + 40);
  for (let endLine = startLine; endLine <= maxEnd; endLine++) {
    const seg = contentLines[endLine]!.replace(/\s+/g, " ").trim();
    if (seg) acc = acc ? `${acc} ${seg}` : seg;
    if (acc === collapsedTarget) return endLine;
    if (acc.length >= collapsedTarget.length || !collapsedTarget.startsWith(acc)) {
      break;
    }
  }
  return -1;
}

function findCollapsedWhitespaceMatchRange(
  content: string,
  target: string,
): { readonly start: number; readonly end: number; readonly matchCount: number } | null {
  // Collapsing opaque literals can alter their meaning; require exact or line-trimmed matches.
  if (/['"`\/]/.test(target)) return null;
  const collapsedTarget = target.replace(/\s+/g, " ").trim();
  if (!collapsedTarget) return null;
  const contentLines = content.split("\n");
  const lineOffsets: number[] = [];
  let offset = 0;
  for (const line of contentLines) {
    lineOffsets.push(offset);
    offset += line.length + 1;
  }
  let firstStart = -1;
  let firstEnd = -1;
  let matchCount = 0;
  for (let startLine = 0; startLine < contentLines.length; startLine++) {
    const endLine = scanCollapsedLineSpan(contentLines, startLine, collapsedTarget);
    if (endLine === -1) continue;
    matchCount += 1;
    if (firstStart === -1) {
      firstStart = lineOffsets[startLine]!;
      firstEnd = lineOffsets[endLine]! + contentLines[endLine]!.length;
    }
  }
  return matchCount > 0 ? { start: firstStart, end: firstEnd, matchCount } : null;
}

function applySingleFileEdit(
  content: string,
  edit: WorkspaceFileEditOperation,
  safePath: string,
  index: number,
): string {
  const normTarget = normalizeLineEndings(edit.target);
  const normReplacement = normalizeLineEndings(edit.replacement);
  const exactCount = countSubstringOccurrences(content, normTarget);
  if (exactCount === 1) {
    const idx = content.indexOf(normTarget);
    return (
      content.slice(0, idx) +
      normReplacement +
      content.slice(idx + normTarget.length)
    );
  }
  if (exactCount > 1) {
    if (!edit.replaceAll) {
      throw new Error(
        `ERR_EDIT_AMBIGUOUS_TARGET: edit #${index + 1} target matched ${exactCount} locations in '${safePath}'; include more surrounding context lines or set replaceAll: true`,
      );
    }
    return content.split(normTarget).join(normReplacement);
  }
  const fuzzyMatch =
    findLineTrimmedMatchRange(content, normTarget) ??
    findCollapsedWhitespaceMatchRange(content, normTarget);
  if (!fuzzyMatch) {
    throw new Error(
      `ERR_EDIT_TARGET_NOT_FOUND: edit #${index + 1} target block not found in '${safePath}'; call workspace.read_file to verify exact contents`,
    );
  }
  if (fuzzyMatch.matchCount > 1) {
    throw new Error(
      `ERR_EDIT_AMBIGUOUS_TARGET: edit #${index + 1} target matched ${fuzzyMatch.matchCount} line-trimmed locations in '${safePath}'; include more surrounding context lines`,
    );
  }
  return (
    content.slice(0, fuzzyMatch.start) +
    normReplacement +
    content.slice(fuzzyMatch.end)
  );
}

const editWorkspaceFileTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.edit_file",
    description:
      "Apply targeted search-and-replace edits to an existing UTF-8 file in the browser workspace without rewriting the entire file. If the file's exact current text is not visible in recent messages (or was `<folded ...>`), call `workspace.read_file` first instead of guessing. Pass either (`path`, `target`, `replacement`) for a single edit or (`path`, `edits: [{ target, replacement }]`) for multiple edits. Each `target` must uniquely match the existing text.",
    effectClass: "idempotent_write",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        target: { type: "string" },
        replacement: { type: "string" },
        replaceAll: { type: "boolean" },
        edits: {
          type: "array",
          items: {
            type: "object",
            properties: {
              target: { type: "string" },
              replacement: { type: "string" },
              replaceAll: { type: "boolean" },
            },
            required: ["target", "replacement"],
          },
        },
      },
      required: ["path"],
    },
  },
  async handler(args, store) {
    const safePath = assertSafeWorkspacePath(args.path);
    const edits = parseFileEditOperations(args);
    const originalContent = await store.readFileText(safePath);
    let updatedContent = normalizeLineEndings(originalContent);
    for (let i = 0; i < edits.length; i++) {
      updatedContent = applySingleFileEdit(
        updatedContent,
        edits[i]!,
        safePath,
        i,
      );
    }
    const preflightSuffix = formatPreflightDiagnosticSuffix(
      verifyWorkspaceFilePreflight(safePath, updatedContent),
    );
    return {
      outputText: `Edited ${safePath} (${edits.length} edit${edits.length === 1 ? "" : "s"} applied, ${updatedContent.length} chars)${preflightSuffix}`,
      fileUpdates: { [safePath]: updatedContent },
    };
  },
};

const deleteWorkspaceFileTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.delete_file",
    description: "Delete a file from the browser workspace (requires user approval).",
    effectClass: "non_idempotent",
    requiresApproval: true,
    parametersSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  },
  async handler(args, store) {
    const safePath = assertSafeWorkspacePath(args.path);
    const existing = await store.listFiles();
    if (!existing.includes(safePath)) {
      return {
        outputText: `File ${safePath} not found (already absent)`,
      };
    }
    return {
      outputText: `Deleted ${safePath}`,
      fileUpdates: { [safePath]: null },
    };
  },
};

export interface SubagentDelegationOptions {
  readonly parentSessionId?: string;
  readonly defaultModelProvider?: () => ProviderModelConfig;
}

const DEFAULT_CHILD_TOOLS: readonly string[] = [
  "workspace.read_file",
  "workspace.list_files",
  "workspace.write_file",
  "workspace.edit_file",
  "workspace.run_script",
  "workspace.inspect_image",
  "image.generate",
  "image.transform",
  "web.search",
  "web.fetch_url",
];

function nonEmptyString(val: unknown): string | null {
  return typeof val === "string" && val.trim().length > 0 ? val.trim() : null;
}

function isSameOriginBaseUrl(baseUrl: string, candidateUrl: string): boolean {
  if (baseUrl === candidateUrl) return true;
  try {
    return new URL(baseUrl).origin === new URL(candidateUrl).origin;
  } catch {
    return baseUrl.startsWith("/") && candidateUrl.startsWith("/");
  }
}

function resolveChildModelConfig(
  args: Record<string, unknown>,
  baseModel: ProviderModelConfig,
): ProviderModelConfig {
  const nextBaseUrl = nonEmptyString(args.baseUrl) ?? baseModel.baseUrl;
  const keepApiKey = isSameOriginBaseUrl(baseModel.baseUrl, nextBaseUrl);
  return {
    providerId: nonEmptyString(args.providerId) ?? baseModel.providerId,
    baseUrl: nextBaseUrl,
    model: nonEmptyString(args.model) ?? baseModel.model,
    apiKey: keepApiKey ? baseModel.apiKey : undefined,
  };
}

function normalizeRequestedToolName(rawName: string): string {
  return rawName.trim().replace(/^functions\./, "").replaceAll("__", ".");
}

function getSessionToolCeiling(session?: AgentSession): readonly string[] | null {
  if (!session) return null;
  if (session.delegatableTools !== undefined) {
    return session.delegatableTools;
  }
  if (session.unrestrictedTools === true && !session.parentSessionId) {
    return null;
  }
  return session.allowedTools;
}

function resolveSafeChildTools(
  args: Record<string, unknown>,
  parentSession?: AgentSession,
  coordinatorSession?: AgentSession,
): string[] {
  const requested = Array.isArray(args.allowedTools)
    ? args.allowedTools
        .filter((t): t is string => typeof t === "string")
        .map(normalizeRequestedToolName)
    : DEFAULT_CHILD_TOOLS;
  const withoutDelegation = requested.filter(
    (t) => t !== "agent.delegate" && t !== "agent.spawn",
  );
  const parentCeiling = getSessionToolCeiling(parentSession);
  const coordCeiling = getSessionToolCeiling(coordinatorSession);
  return withoutDelegation.filter(
    (t) =>
      (parentCeiling === null || parentCeiling.includes(t)) &&
      (coordCeiling === null || coordCeiling.includes(t)),
  );
}

export interface ImageGenerationProvider {
  generateOrTransform(params: {
    readonly runId?: string;
    readonly prompt: string;
    readonly sourceDataUrl?: string;
    readonly style?: string;
    readonly signal?: AbortSignal;
  }): Promise<{
    readonly content: Uint8Array | string;
    readonly dataUrl: string;
    readonly diagnostics?: string;
  }>;
}

function buildDeterministicSvgArtwork(prompt: string, style = "sprite"): string {
  const cleanPrompt = prompt
    .replace(/[<>&"']/g, "")
    .slice(0, 72);
  let hash = 2166136261;
  for (let i = 0; i < prompt.length; i++) {
    hash = Math.imul(hash ^ prompt.charCodeAt(i), 16777619) >>> 0;
  }
  const hueA = hash % 360;
  const hueB = (hash >> 8) % 360;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="hsl(${hueA}, 65%, 24%)" />
      <stop offset="100%" stop-color="hsl(${hueB}, 70%, 14%)" />
    </linearGradient>
  </defs>
  <rect width="256" height="256" rx="18" fill="url(#bg)" />
  <circle cx="128" cy="112" r="54" fill="hsl(${(hueA + 140) % 360}, 75%, 60%)" opacity="0.9" />
  <text x="128" y="212" text-anchor="middle" fill="#eef6f5" font-family="system-ui,sans-serif" font-size="12" font-weight="600">${style}: ${cleanPrompt}</text>
</svg>`;
}

export const defaultLocalImageProvider: ImageGenerationProvider = {
  async generateOrTransform({ prompt, style }) {
    const svg = buildDeterministicSvgArtwork(prompt, style ?? "flux-lora");
    return {
      content: svg,
      dataUrl: `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`,
    };
  },
};

async function extractOpenAIErrorDetail(
  resp: { readonly status: number; readonly text?: () => Promise<string> },
): Promise<string> {
  try {
    const raw = resp.text ? await resp.text() : "";
    const parsed = JSON.parse(raw) as {
      error?: { message?: string; code?: string };
    };
    const msg = parsed.error?.message ?? raw.slice(0, 140);
    return `HTTP ${resp.status} (${parsed.error?.code ?? "err"}): ${msg}`;
  } catch {
    return `HTTP ${resp.status}`;
  }
}

async function extractImageB64FromGenerationResponse(
  resp: Response,
  signal?: AbortSignal,
): Promise<string | null> {
  const data = (await resp.json()) as {
    data?: Array<{ b64_json?: string; url?: string }>;
  };
  const first = data.data?.[0];
  if (typeof first?.b64_json === "string" && first.b64_json.length > 0) {
    return first.b64_json;
  }
  if (typeof first?.url === "string" && first.url.startsWith("http")) {
    const imgResp = await globalThis.fetch(first.url, { signal });
    if (imgResp.ok && typeof imgResp.arrayBuffer === "function") {
      return bytesToBase64(new Uint8Array(await imgResp.arrayBuffer()));
    }
  }
  return null;
}

async function requestOpenAIImageB64(
  endpoint: string,
  apiKey: string,
  body: Record<string, unknown>,
  diagnostics: string[],
  label: string,
  signal?: AbortSignal,
): Promise<string | null> {
  let resp = await globalThis.fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok && resp.status === 400 && "response_format" in body) {
    const { response_format: _omitted, ...bodyWithoutFormat } = body;
    resp = await globalThis.fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(bodyWithoutFormat),
      signal,
    });
  }
  if (!resp.ok) {
    diagnostics.push(`${label}=${await extractOpenAIErrorDetail(resp)}`);
    return null;
  }
  return extractImageB64FromGenerationResponse(resp, signal);
}

async function requestResponsesImageGenerationB64(
  baseUrl: string,
  apiKey: string,
  model: string,
  prompt: string,
  sourceDataUrl: string | undefined,
  diagnostics: string[],
  signal?: AbortSignal,
): Promise<string | null> {
  const isRasterSource =
    typeof sourceDataUrl === "string" &&
    /^data:image\/(png|jpeg|jpg|webp|gif);base64,/i.test(sourceDataUrl);
  if (sourceDataUrl && !isRasterSource) return null;
  const editInstruction =
    `Edit the attached source image in-place: ${prompt}. ` +
    `Preserve the exact original background, composition, lighting, camera angle, and color/monochrome palette of everything outside the requested change.`;
  const input = isRasterSource
    ? [
        {
          role: "user",
          content: [
            { type: "input_text", text: editInstruction },
            { type: "input_image", image_url: sourceDataUrl },
          ],
        },
      ]
    : prompt;
  try {
    const resp = await globalThis.fetch(`${baseUrl}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: model || "gpt-4o",
      tools: [{ type: "image_generation" }],
      input,
    }),
    signal,
  });
  if (!resp.ok) {
    diagnostics.push(`responses.image_generation=${await extractOpenAIErrorDetail(resp)}`);
    return null;
  }
  const data = (await resp.json()) as {
    output?: Array<{ type?: string; result?: string }>;
  };
  const call = (data.output ?? []).find(
    (item) => item.type === "image_generation_call" && typeof item.result === "string",
  );
    return call?.result ?? null;
  } catch (err) {
    signal?.throwIfAborted();
    diagnostics.push(`responses.image_generation=${err instanceof Error ? err.message : String(err)}`);
    return null;
  }
}

function buildSourceVisionContent(
  instruction: string,
  sourceDataUrl?: string,
): unknown {
  if (!sourceDataUrl) return instruction;
  if (sourceDataUrl.startsWith("data:image/svg+xml;utf8,")) {
    const svg = decodeURIComponent(
      sourceDataUrl.slice("data:image/svg+xml;utf8,".length),
    );
    return `${instruction}\n\n[Source SVG]\n${svg}`;
  }
  if (/^data:image\/(png|jpeg|jpg|webp|gif);base64,/i.test(sourceDataUrl)) {
    return [
      { type: "text", text: instruction },
      { type: "image_url", image_url: { url: sourceDataUrl } },
    ];
  }
  return `${instruction}\n\n[Source]\n${sourceDataUrl.slice(0, 2000)}`;
}

async function synthesizeTransformPromptViaVision(
  baseUrl: string,
  apiKey: string,
  model: string,
  editPrompt: string,
  sourceDataUrl: string,
  signal?: AbortSignal,
): Promise<string> {
  try {
    const instruction =
      `Inspect the source image carefully. Write a vivid, detailed DALL-E 3 image generation prompt ` +
      `that preserves the exact subject, pose, composition, and background of the source image ` +
      `while applying this transformation: "${editPrompt}". Output ONLY the final prompt text.`;
    const resp = await globalThis.fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: model || "gpt-4o",
        messages: [
          {
            role: "user",
            content: buildSourceVisionContent(instruction, sourceDataUrl),
          },
        ],
      }),
      signal,
    });
    if (!resp.ok) return editPrompt;
    const data = (await resp.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const text = data.choices?.[0]?.message?.content?.trim();
    return text && text.length > 10 ? text : editPrompt;
  } catch {
    return editPrompt;
  }
}

async function requestChatSvgIllustration(
  baseUrl: string,
  apiKey: string,
  model: string,
  prompt: string,
  sourceDataUrl?: string,
  signal?: AbortSignal,
): Promise<string | null> {
  try {
    const instruction =
      `Create a detailed, rich, visually appealing standalone <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512"> ` +
      `illustration for: "${prompt}". If a source image or SVG is attached, preserve its subject/scene and apply the requested transformation ` +
      `(such as changing colors, lighting, or details). Output ONLY raw <svg>...</svg> XML without markdown fences.`;
    const resp = await globalThis.fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: model || "gpt-4o",
        messages: [
          {
            role: "user",
            content: buildSourceVisionContent(instruction, sourceDataUrl),
          },
        ],
      }),
      signal,
    });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const raw = data.choices?.[0]?.message?.content ?? "";
    const match = /<svg[\s\S]*<\/svg>/i.exec(raw);
    return match ? match[0] : null;
  } catch {
    return null;
  }
}

function formatCleanOpenAIImagePrompt(rawPrompt: string, style?: string): string {
  const stripped = rawPrompt.replace(/\s*\[source:[^\]]+\]\s*$/i, "").trim();
  const customStyle =
    style && style !== "game-asset" && style !== "flux-4b-lora"
      ? style.trim()
      : "";
  return customStyle ? `${stripped} (${customStyle})` : stripped;
}

async function tryOpenAIRasterEndpoints(
  baseUrl: string,
  apiKey: string,
  model: string,
  cleanPrompt: string,
  sourceDataUrl: string | undefined,
  diagnostics: string[],
  imageMode: ImageServiceMode,
  signal?: AbortSignal,
): Promise<{ readonly b64: string; readonly via: string } | null> {
  if (imageMode === "openai-gpt-svg" || imageMode === "local-deterministic-svg") {
    return null;
  }
  const genUrl = `${baseUrl}/images/generations`;
  if (sourceDataUrl) {
    const edited = await requestResponsesImageGenerationB64(
      baseUrl,
      apiKey,
      model,
      cleanPrompt,
      sourceDataUrl,
      diagnostics,
      signal,
    );
    if (edited) return { b64: edited, via: "openai:responses.image_generation(edit)" };
  }
  const effectivePrompt = sourceDataUrl
    ? await synthesizeTransformPromptViaVision(
        baseUrl,
        apiKey,
        model,
        cleanPrompt,
        sourceDataUrl,
        signal,
      )
    : cleanPrompt;
  const dalle3 = await requestOpenAIImageB64(
    genUrl,
    apiKey,
    {
      model: "dall-e-3",
      prompt: effectivePrompt,
      n: 1,
      size: "1024x1024",
      response_format: "b64_json",
    },
    diagnostics,
    "dall-e-3",
    signal,
  );
  if (dalle3) {
    return { b64: dalle3, via: "openai:dall-e-3" };
  }

  return tryOpenAIGptAndResponsesImage(
    baseUrl,
    genUrl,
    apiKey,
    model,
    effectivePrompt,
    sourceDataUrl,
    diagnostics,
    signal,
  );
}

async function tryOpenAIGptAndResponsesImage(
  baseUrl: string,
  genUrl: string,
  apiKey: string,
  model: string,
  effectivePrompt: string,
  sourceDataUrl: string | undefined,
  diagnostics: string[],
  signal?: AbortSignal,
): Promise<{ readonly b64: string; readonly via: string } | null> {
  const gptImg = await requestOpenAIImageB64(
    genUrl,
    apiKey,
    { model: "gpt-image-1", prompt: effectivePrompt, size: "1024x1024" },
    diagnostics,
    "gpt-image-1",
    signal,
  );
  if (gptImg) return { b64: gptImg, via: "openai:gpt-image-1" };

  if (!sourceDataUrl) {
    const respImg = await requestResponsesImageGenerationB64(
      baseUrl,
      apiKey,
      model,
      effectivePrompt,
      undefined,
      diagnostics,
      signal,
    );
    if (respImg) return { b64: respImg, via: "openai:responses.image_generation" };
  }
  return null;
}

function canUseRemoteImageProvider(
  cfg: ProviderModelConfig | undefined,
  imageMode: ImageServiceMode,
): cfg is ProviderModelConfig & { readonly apiKey: string } {
  return (
    imageMode !== "local-deterministic-svg" &&
    Boolean(cfg?.apiKey) &&
    cfg?.providerId !== "veles-demo" &&
    typeof globalThis.fetch === "function"
  );
}

export function createOpenAIImageProvider(
  modelConfigProvider?: (runId?: string) => ProviderModelConfig | undefined,
  fallback: ImageGenerationProvider = defaultLocalImageProvider,
  imageModeProvider?: () => ImageServiceMode,
): ImageGenerationProvider {
  return {
    async generateOrTransform(params) {
      params.signal?.throwIfAborted();
      const cfg = modelConfigProvider?.(params.runId);
      const imageMode = imageModeProvider?.() ?? "openai-auto";
      if (!canUseRemoteImageProvider(cfg, imageMode)) {
        return fallback.generateOrTransform(params);
      }
      const baseUrl = cfg.baseUrl.replace(/\/+$/, "");
      const diagnostics: string[] = [];
      try {
        const cleanPrompt = formatCleanOpenAIImagePrompt(
          params.prompt,
          params.style,
        );

        const raster = await tryOpenAIRasterEndpoints(
          baseUrl,
          cfg.apiKey,
          cfg.model,
          cleanPrompt,
          params.sourceDataUrl,
          diagnostics,
          imageMode,
          params.signal,
        );
        if (raster) {
          return {
            content: base64ToBytes(raster.b64),
            dataUrl: `data:image/png;base64,${raster.b64}`,
            diagnostics: `Generated raster PNG via ${raster.via}`,
          };
        }

        const llmSvg = await requestChatSvgIllustration(
          baseUrl,
          cfg.apiKey,
          cfg.model,
          cleanPrompt,
          params.sourceDataUrl,
          params.signal,
        );
        if (llmSvg) {
          return {
            content: llmSvg,
            dataUrl: `data:image/svg+xml;utf8,${encodeURIComponent(llmSvg)}`,
            diagnostics: `Fallback to GPT SVG illustration (${diagnostics.join(" | ")})`,
          };
        }
      } catch (err) {
        diagnostics.push(err instanceof Error ? err.message : String(err));
      }
      params.signal?.throwIfAborted();
      const local = await fallback.generateOrTransform(params);
      return {
        ...local,
        diagnostics:
          diagnostics.length > 0
            ? `Local SVG fallback (${diagnostics.join(" | ")})`
            : undefined,
      };
    },
  };
}

function extractResponsesOutputText(raw: unknown): string | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as {
    output_text?: unknown;
    output?: Array<{ content?: Array<{ text?: string }> }>;
  };
  if (typeof obj.output_text === "string" && obj.output_text.trim()) {
    return obj.output_text.trim();
  }
  const chunks: string[] = [];
  for (const item of obj.output ?? []) {
    for (const part of item.content ?? []) {
      if (typeof part.text === "string") chunks.push(part.text);
    }
  }
  return chunks.length > 0 ? chunks.join("\n").trim() : null;
}

function canUseRemoteWebSearch(
  cfg: ProviderModelConfig | undefined,
  mode: WebSearchServiceMode,
): cfg is ProviderModelConfig & { readonly apiKey: string } {
  return (
    mode === "openai-responses" &&
    Boolean(cfg?.apiKey) &&
    cfg?.providerId !== "veles-demo" &&
    typeof globalThis.fetch === "function"
  );
}

async function querySearxngSearch(
  searxngUrl: string,
  query: string,
  signal?: AbortSignal,
): Promise<string | null> {
  if (typeof globalThis.fetch !== "function" || !searxngUrl.trim()) {
    return null;
  }
  try {
    const base = searxngUrl.trim().replace(/\?.*$/, "");
    const url = `${base}?q=${encodeURIComponent(query)}&format=json`;
    const resp = await globalThis.fetch(url, { method: "GET", signal });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      results?: Array<{ title?: string; url?: string; content?: string }>;
    };
    const items = (data.results ?? []).slice(0, 5);
    if (items.length === 0) return null;
    return items
      .map(
        (r, i) =>
          `${i + 1}. ${r.title ?? "Result"} (${r.url ?? ""})\n   ${r.content ?? ""}`,
      )
      .join("\n\n");
  } catch {
    return null;
  }
}

export function createWebSearchTool(
  modelConfigProvider?: (runId?: string) => ProviderModelConfig | undefined,
  webSearchModeProvider?: () => WebSearchServiceMode,
  searxngUrlProvider?: () => string | undefined,
): RegisteredBrowserTool {
  return {
    spec: {
      name: "web.search",
      description:
        "Search the live web using OpenAI web_search_preview or SearXNG (uses the configured search service or returns an offline fallback summary).",
      effectClass: "pure_read",
      requiresApproval: false,
      parametersSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
        },
        required: ["query"],
      },
    },
    async handler(args, _store, signal, context) {
      const query = nonEmptyString(args.query);
      if (!query) {
        throw new Error("ERR_WEB_SEARCH_EMPTY_QUERY: 'query' must be non-empty");
      }
      const cfg = modelConfigProvider?.(context?.runId);
      const mode = webSearchModeProvider?.() ?? "openai-responses";
      if (mode === "searxng") {
        const searxText = await querySearxngSearch(
          searxngUrlProvider?.() ?? "http://127.0.0.1:8080/search",
          query,
          signal,
        );
        if (searxText) return { outputText: searxText };
      }
      if (canUseRemoteWebSearch(cfg, mode)) {
        try {
          const endpoint = `${cfg.baseUrl.replace(/\/+$/, "")}/responses`;
          const resp = await globalThis.fetch(endpoint, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${cfg.apiKey}`,
            },
            body: JSON.stringify({
              model: cfg.model || "gpt-4o-mini",
              tools: [{ type: "web_search_preview" }],
              input: query,
            }),
            signal,
          });
          if (resp.ok) {
            const text = extractResponsesOutputText(await resp.json());
            if (text) return { outputText: text };
          }
        } catch {
          // Fall back to offline summary below
        }
      }
      return {
        outputText: `[Offline Web Search Fallback] Query: "${query}". Configure OpenAI or SearXNG in ⚙️ Services to run live web search queries.`,
      };
    },
  };
}

function parseIpv4Octets(host: string): [number, number, number, number] | null {
  const parts = host.split(".");
  if (parts.length !== 4) return null;
  const octets: number[] = [];
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const n = Number(part);
    if (n < 0 || n > 255) return null;
    octets.push(n);
  }
  return octets as [number, number, number, number];
}

function isPrivateIpv4Octets(
  [a, b, c]: readonly [number, number, number, number],
): boolean {
  if (a === 0 || a === 10 || a === 127 || a >= 224) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && (b === 168 || (b === 0 && (c === 0 || c === 2)))) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  return a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100));
}

function expandIpv6Words(rawIpv6: string): number[] | null {
  let normalized = rawIpv6;
  const lastColon = normalized.lastIndexOf(":");
  if (lastColon !== -1 && normalized.slice(lastColon + 1).includes(".")) {
    const v4 = parseIpv4Octets(normalized.slice(lastColon + 1));
    if (!v4) return null;
    const hi = ((v4[0] << 8) | v4[1]).toString(16);
    const lo = ((v4[2] << 8) | v4[3]).toString(16);
    normalized = `${normalized.slice(0, lastColon)}:${hi}:${lo}`;
  }
  const halves = normalized.split("::");
  if (halves.length > 2) return null;
  const left = halves[0] ? halves[0].split(":") : [];
  const right = halves.length === 2 && halves[1] ? halves[1].split(":") : [];
  const fill = halves.length === 2 ? 8 - (left.length + right.length) : 0;
  if ((halves.length === 1 && left.length !== 8) || fill < 0) return null;
  const groups = [...left, ...Array.from({ length: fill }, () => "0"), ...right];
  if (groups.length !== 8) return null;
  const words = groups.map((g) => (/^[0-9a-f]{1,4}$/i.test(g) ? parseInt(g, 16) : -1));
  return words.some((w) => w < 0) ? null : words;
}

function extractEmbeddedIpv4FromIpv6(
  w: readonly number[],
): [number, number, number, number] | null {
  const isV4MappedOrCompat =
    w[0] === 0 &&
    w[1] === 0 &&
    w[2] === 0 &&
    w[3] === 0 &&
    ((w[4] === 0 && (w[5] === 0xffff || w[5] === 0)) ||
      (w[4] === 0xffff && w[5] === 0));
  const isNat64 = w[0] === 0x0064 && w[1] === 0xff9b;
  if (isV4MappedOrCompat || isNat64) {
    return [(w[6]! >> 8) & 0xff, w[6]! & 0xff, (w[7]! >> 8) & 0xff, w[7]! & 0xff];
  }
  if (w[0] === 0x2002) {
    return [(w[1]! >> 8) & 0xff, w[1]! & 0xff, (w[2]! >> 8) & 0xff, w[2]! & 0xff];
  }
  return null;
}

function isPrivateIpv6Words(w: readonly number[]): boolean {
  const allZeroPrefix = w.slice(0, 7).every((x) => x === 0);
  if (allZeroPrefix && (w[7] === 0 || w[7] === 1)) return true;
  const embeddedV4 = extractEmbeddedIpv4FromIpv6(w);
  if (embeddedV4) return isPrivateIpv4Octets(embeddedV4);
  const w0 = w[0]!;
  if ((w0 & 0xfe00) === 0xfc00 || (w0 & 0xffc0) === 0xfe80) return true;
  if ((w0 & 0xffc0) === 0xfec0 || (w0 & 0xff00) === 0xff00) return true;
  return w0 === 0x2001 && w[1] === 0x0db8;
}

export function isBlockedPrivateHost(rawHost: string): boolean {
  const host = rawHost
    .trim()
    .replace(/^\[|\]$/g, "")
    .replace(/%.*$/, "")
    .replace(/\.+$/, "")
    .toLowerCase();
  if (
    !host ||
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host.endsWith(".local") ||
    host.endsWith(".internal") ||
    /^(0x[0-9a-f]+|\d+)$/i.test(host)
  ) {
    return true;
  }
  if (host.includes(":")) {
    const words = expandIpv6Words(host);
    return words === null ? true : isPrivateIpv6Words(words);
  }
  const v4 = parseIpv4Octets(host);
  return v4 ? isPrivateIpv4Octets(v4) : false;
}

const INITIAL_GLOBAL_FETCH = globalThis.fetch;

function assertPublicFetchUrl(rawUrl: unknown): string {
  const str = nonEmptyString(rawUrl);
  if (!str) {
    throw new Error("ERR_FETCH_URL_REQUIRED: 'url' must be a non-empty string");
  }
  const parsed = new URL(str);
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error("ERR_FETCH_UNSAFE_PROTOCOL: Only http/https URLs are allowed");
  }
  if (isBlockedPrivateHost(parsed.hostname)) {
    throw new Error(
      `ERR_FETCH_PRIVATE_HOST_BLOCKED: Host '${parsed.hostname}' is not permitted`,
    );
  }
  return parsed.toString();
}

async function assertPublicDestinationWithDns(rawUrl: unknown): Promise<string> {
  const safe = assertPublicFetchUrl(rawUrl);
  const isNodeEnv =
    typeof process !== "undefined" && Boolean(process.versions?.node);
  if (!isNodeEnv) {
    return safe;
  }
  const lookupHost = new URL(safe).hostname.replace(/^\[|\]$/g, "");
  let resolved: Array<{ address: string }>;
  try {
    const dns = await import("node:dns/promises");
    resolved = await dns.lookup(lookupHost, { all: true, verbatim: true });
  } catch {
    if (
      globalThis.fetch !== INITIAL_GLOBAL_FETCH &&
      /\.(test|example)$/i.test(lookupHost)
    ) {
      return safe;
    }
    throw new Error(
      `ERR_FETCH_DNS_RESOLUTION_FAILED: Could not resolve '${lookupHost}'`,
    );
  }
  if (
    resolved.length === 0 ||
    resolved.some((entry) => isBlockedPrivateHost(entry.address))
  ) {
    throw new Error(
      `ERR_FETCH_PRIVATE_HOST_BLOCKED: Host '${lookupHost}' resolved to a private address`,
    );
  }
  return safe;
}

export function normalizePublicFetchUrl(rawUrl: unknown): string {
  const safe = assertPublicFetchUrl(rawUrl);
  const ghBlobMatch =
    /^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/i.exec(safe);
  if (ghBlobMatch) {
    return `https://raw.githubusercontent.com/${ghBlobMatch[1]}/${ghBlobMatch[2]}/${ghBlobMatch[3]}`;
  }
  return safe;
}

async function fetchDirectFollowingSafeRedirects(
  initialUrl: string,
  signal?: AbortSignal,
): Promise<Response> {
  let currentUrl = await assertPublicDestinationWithDns(initialUrl);
  for (let hop = 0; hop <= 5; hop++) {
    const resp = await globalThis.fetch(currentUrl, {
      method: "GET",
      redirect: "manual",
      signal,
    });
    if (resp.url) {
      await assertPublicDestinationWithDns(resp.url);
    }
    if (resp.status >= 300 && resp.status < 400) {
      const location = resp.headers?.get?.("location");
      if (!location) return resp;
      currentUrl = await assertPublicDestinationWithDns(
        new URL(location, currentUrl).toString(),
      );
      continue;
    }
    return resp;
  }
  throw new Error("ERR_FETCH_TOO_MANY_REDIRECTS");
}

export const PUBLIC_FETCH_PROXY_HEADER = "X-Veles-Fetch-Proxy";
export const PUBLIC_FETCH_PROXY_VERSION = "1";

async function fetchFromPublicProxy(
  safeUrl: string,
  browserOrigin: string,
  signal?: AbortSignal,
): Promise<Response> {
  const proxyUrl = `${browserOrigin}/api/proxy/fetch?url=${encodeURIComponent(safeUrl)}`;
  let response: Response;
  try {
    response = await globalThis.fetch(proxyUrl, {
      method: "GET",
      redirect: "error",
      signal,
    });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new Error(
      "ERR_FETCH_PROXY_UNAVAILABLE: Public fetching requires the Veles /api/proxy/fetch service (Vite dev or preview).",
    );
  }
  if (response.headers.get(PUBLIC_FETCH_PROXY_HEADER) !== PUBLIC_FETCH_PROXY_VERSION) {
    throw new Error(
      "ERR_FETCH_PROXY_UNAVAILABLE: Public fetching requires the Veles /api/proxy/fetch service (Vite dev or preview).",
    );
  }
  if (response.status === 403 || response.status === 502) {
    const message = (await response.text()).trim();
    if (message.startsWith("ERR_FETCH_")) throw new Error(message);
  }
  return response;
}

async function fetchPublicResource(
  safeUrl: string,
  signal?: AbortSignal,
): Promise<Response> {
  const isNodeEnv =
    typeof process !== "undefined" && Boolean(process.versions?.node);
  if (isNodeEnv) return fetchDirectFollowingSafeRedirects(safeUrl, signal);
  const browserOrigin = (globalThis as { location?: { origin?: string } })
    .location?.origin;
  if (!browserOrigin || !/^https?:\/\//i.test(browserOrigin)) {
    throw new Error("ERR_FETCH_PROXY_UNAVAILABLE: Public fetching requires a Veles HTTP origin.");
  }
  return fetchFromPublicProxy(safeUrl, browserOrigin, signal);
}

const BINARY_EXT_RE =
  /\.(png|jpe?g|webp|gif|bmp|ico|pdf|zip|wasm|mp3|wav|ogg|mp4)(\?.*)?$/i;
const BINARY_MIME_RE =
  /^(image\/(?!svg\+xml)|audio\/|video\/|application\/(octet-stream|pdf|zip|wasm))/i;

function isBinaryFetchResource(
  resp: Response,
  url: string,
  saveToPath: string | null,
): boolean {
  const mime = resp.headers?.get?.("content-type")?.trim() ?? "";
  if (BINARY_MIME_RE.test(mime)) return true;
  return BINARY_EXT_RE.test(saveToPath ?? url);
}

async function handleBinaryFetchResponse(
  resp: Response,
  safeUrl: string,
  targetPath: string | null,
): Promise<{
  readonly outputText: string;
  readonly fileUpdates?: Readonly<Record<string, Uint8Array>>;
}> {
  const buf =
    typeof resp.arrayBuffer === "function"
      ? await resp.arrayBuffer()
      : new TextEncoder().encode(await resp.text()).buffer;
  const bytes = new Uint8Array(buf);
  if (!targetPath) {
    return {
      outputText: `Fetched binary resource from ${safeUrl} (${bytes.byteLength} bytes). Pass 'saveToPath' (e.g. 'assets/image.jpg') to save it into the workspace.`,
    };
  }
  return {
    outputText: `Fetched binary file from ${safeUrl} (${bytes.byteLength} bytes) and saved to '${targetPath}'.`,
    fileUpdates: { [targetPath]: bytes },
  };
}

function formatSavedTextFetchOutput(
  safeUrl: string,
  targetPath: string,
  fullText: string,
): string {
  if (fullText.length <= 1_800) {
    return fullText;
  }
  const lines = fullText.split(/\r?\n/);
  const preview = lines.slice(0, 20).join("\n").slice(0, 1_500);
  return (
    `Saved ${fullText.length} chars (${lines.length} lines) from ${safeUrl} to '${targetPath}'.\n` +
    `Preview (first ${Math.min(20, lines.length)} lines):\n${preview}`
  );
}

const fetchUrlWorkspaceTool: RegisteredBrowserTool = {
  spec: {
    name: "web.fetch_url",
    description:
      "Fetch a dataset (.csv, .json), text/documentation, or binary image (.png, .jpg, .webp, .gif, .svg) from a public HTTP/HTTPS URL (auto-normalizes GitHub /blob/ links to raw content) and optionally save it to `saveToPath` in OPFS.",
    effectClass: "pure_read",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: {
        url: { type: "string" },
        saveToPath: { type: "string" },
      },
      required: ["url"],
    },
  },
  async handler(args, _store, signal) {
    const safeUrl = normalizePublicFetchUrl(args.url);
    const saveToRaw = nonEmptyString(args.saveToPath);
    const targetPath = saveToRaw ? assertSafeWorkspacePath(saveToRaw) : null;
    const resp = await fetchPublicResource(safeUrl, signal);
    if (!resp.ok) {
      throw new Error(`ERR_FETCH_HTTP_${resp.status}: Failed to fetch ${safeUrl}`);
    }
    if (isBinaryFetchResource(resp, safeUrl, targetPath)) {
      return handleBinaryFetchResponse(resp, safeUrl, targetPath);
    }
    const rawText = await resp.text();
    const cleanedFull = rawText
      .replace(/<script[\s\S]*?<\/script>/gi, "")
      .replace(/<style[\s\S]*?<\/style>/gi, "")
      .slice(0, 500_000);
    if (targetPath) {
      return {
        outputText: formatSavedTextFetchOutput(safeUrl, targetPath, cleanedFull),
        fileUpdates: { [targetPath]: cleanedFull },
      };
    }
    return {
      outputText: cleanedFull.slice(0, 12_000),
    };
  },
};

async function loadWorkspaceTextMap(
  store: WorkspaceKernelStore,
): Promise<Record<string, string>> {
  const paths = await store.listFiles();
  const files: Record<string, string> = {};
  for (const p of paths) {
    try {
      files[p] = await store.readFileText(p);
    } catch {
      // Ignore binary blobs that are not UTF-8 text
    }
  }
  return files;
}

function formatWrittenFilePreview(path: string, text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const snippet = flat.length > 240 ? `${flat.slice(0, 240)}...` : flat;
  return `Wrote '${path}' (${text.length} chars). Preview: ${snippet}`;
}

function formatScriptOutcomeText(
  logs: readonly string[],
  result: unknown,
  stagedUpdates: Readonly<Record<string, Uint8Array | string | null>> = {},
): string {
  const parts: string[] = [];
  if (logs.length > 0) {
    parts.push(logs.join("\n"));
  }
  if (result !== undefined) {
    const serialized =
      typeof result === "string" ? result : JSON.stringify(result, null, 2);
    parts.push(serialized);
  }
  for (const [path, val] of Object.entries(stagedUpdates)) {
    if (typeof val === "string") {
      parts.push(formatWrittenFilePreview(path, val));
    }
  }
  const combined = parts.join("\n").trim();
  return (combined || "Script completed (no output).").slice(0, 8_000);
}

interface IsolatedScriptExecutionResult {
  readonly logs: string[];
  readonly hasResult: boolean;
  readonly result: unknown;
  readonly stagedUpdates: Record<string, string>;
}

async function executeHostValidatedTextFetch(
  rawUrl: unknown,
  signal?: AbortSignal,
): Promise<string> {
  const safeUrl = normalizePublicFetchUrl(rawUrl);
  const resp = await fetchPublicResource(safeUrl, signal);
  if (!resp.ok) {
    throw new Error(`ERR_FETCH_HTTP_${resp.status}: Failed to fetch ${safeUrl}`);
  }
  const rawText = await resp.text();
  return rawText
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .slice(0, 500_000);
}

const SANDBOX_SCRIPT_RUNNER_BODY = `
const files = JSON.parse(__initialFilesJson);
const initialSnapshot = Object.assign(Object.create(null), files);
const stagedUpdates = Object.create(null);
const logs = [];
function formatConsoleValue(value) {
  const tag = Object.prototype.toString.call(value);
  if (value instanceof Error || tag === "[object Error]" || tag === "[object DOMException]") {
    return String(value.name || "Error") + ": " + String(value.message || "");
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}
function fmt(args) {
  return args.map(formatConsoleValue).join(" ");
}
function normPath(raw) {
  if (typeof raw !== "string" || !raw.trim()) {
    throw new Error("ERR_TOOL_INVALID_PATH: Path must be a non-empty string");
  }
  const clean = raw.trim();
  if (clean.startsWith("/") || clean.includes("\\\\") || clean.split("/").some((p) => p === ".." || p === "." || p.length === 0)) {
    throw new Error("ERR_TOOL_UNSAFE_PATH: Path '" + raw + "' must be a clean relative workspace path");
  }
  return clean;
}
const readFile = (p) => {
  const safe = normPath(p);
  const val = files[safe];
  if (typeof val !== "string") throw new Error("ERR_FILE_NOT_FOUND: '" + safe + "'");
  const unix = val.replace(/\\r\\n/g, "\\n");
  return /\\.(csv|tsv)$/i.test(safe) ? unix.trimEnd() : unix;
};
const writeFile = (p, content) => {
  const safe = normPath(p);
  const text = typeof content === "string" ? content : JSON.stringify(content, null, 2);
  files[safe] = text;
  stagedUpdates[safe] = text;
};
const listFiles = (prefix) => {
  const all = Object.keys(files).filter((k) => typeof files[k] === "string").sort();
  if (typeof prefix !== "string" || !prefix.trim()) return all;
  const clean = prefix.trim();
  return all.filter((k) => k.startsWith(clean) || k.includes(clean));
};
const searchFiles = (query, pathFilter) => {
  const matches = [];
  const isRe = query instanceof RegExp;
  const needle = isRe ? null : String(query ?? "");
  if (!isRe && !needle) return matches;
  for (const p of listFiles(pathFilter)) {
    const lines = String(files[p] ?? "").replace(/\\r\\n/g, "\\n").split("\\n");
    for (let i = 0; i < lines.length; i++) {
      const lineText = lines[i];
      if (isRe ? query.test(lineText) : lineText.includes(needle)) {
        matches.push({ path: p, line: i + 1, text: lineText.slice(0, 240) });
        if (matches.length >= 100) return matches;
      }
    }
  }
  return matches;
};
const editFile = (p, target, replacement, replaceAll = false) => {
  const current = readFile(p);
  const t = String(target ?? "").replace(/\\r\\n/g, "\\n");
  const r = String(replacement ?? "").replace(/\\r\\n/g, "\\n");
  if (!t || !current.includes(t)) {
    if (replaceAll && r && current.includes(r)) return current;
    throw new Error("ERR_EDIT_TARGET_NOT_FOUND: Target not found in '" + normPath(p) + "'");
  }
  const updated = replaceAll ? current.split(t).join(r) : current.replace(t, r);
  writeFile(p, updated);
  return updated;
};
const fetchUrl = async (url, saveToPath) => {
  const text = await __rpcFetchUrl(String(url ?? ""));
  if (typeof saveToPath === "string" && saveToPath.trim()) {
    writeFile(saveToPath, text);
  }
  return text;
};
const tools = Object.freeze({
  workspace: Object.freeze({
    readFile,
    writeFile,
    listFiles,
    searchFiles,
    search: searchFiles,
    editFile,
  }),
  web: Object.freeze({
    fetchUrl,
    fetch_url: fetchUrl,
  }),
});
const scriptConsole = {
  log: (...a) => logs.push(fmt(a)),
  info: (...a) => logs.push(fmt(a)),
  warn: (...a) => logs.push(fmt(a)),
  error: (...a) => logs.push(fmt(a)),
};
const result = await (async function(files, readFile, writeFile, console, tools, listFiles, searchFiles, editFile, fetchUrl) {
  "use strict";
  return await (__runUserScript(files, readFile, writeFile, console, tools, listFiles, searchFiles, editFile, fetchUrl));
})(files, readFile, writeFile, scriptConsole, tools, listFiles, searchFiles, editFile, fetchUrl);
for (const [k, v] of Object.entries(files)) {
  if (typeof v === "string" && initialSnapshot[k] !== v) {
    stagedUpdates[normPath(k)] = v;
  }
}
`;

const WORKER_SCRUBBED_GLOBALS = [
  "indexedDB", "IDBFactory", "IDBDatabase", "IDBTransaction", "IDBObjectStore",
  "IDBIndex", "IDBCursor", "IDBCursorWithValue", "IDBKeyRange", "IDBRequest",
  "IDBOpenDBRequest", "IDBVersionChangeEvent", "navigator", "WorkerNavigator",
  "StorageManager", "FileSystemHandle", "FileSystemFileHandle",
  "FileSystemDirectoryHandle", "FileSystemWritableFileStream",
  "FileSystemSyncAccessHandle", "caches", "CacheStorage", "Cache", "fetch",
  "Request", "Response", "Headers", "XMLHttpRequest", "WebSocket", "EventSource",
  "WebTransport", "BroadcastChannel", "SharedWorker", "Worker", "importScripts",
  "cookieStore", "CookieStore", "localStorage", "sessionStorage", "origin",
  "location", "postMessage", "close", "WebAssembly",
];

function stripJsCommentsAndStrings(code: string): string {
  return code.replace(
    /("(?:\\[\s\S]|[^"\\])*")|('(?:\\[\s\S]|[^'\\])*')|\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g,
    (_match, dblQuote, sglQuote) => (dblQuote || sglQuote ? "''" : " "),
  );
}

function assertNoDynamicImportSyntax(code: string): void {
  const stripped = stripJsCommentsAndStrings(code);
  if (/\bimport\b/.test(stripped)) {
    throw new Error("ERR_SCRIPT_SANDBOX_VIOLATION: Dynamic import() is not permitted");
  }
}

export function buildIsolatedWorkerBootstrapSource(): string {
  return `
const __postToHost = self.postMessage.bind(self);
const __origSetTimeout = self.setTimeout.bind(self);
const __origSetInterval = self.setInterval.bind(self);
const __pendingRpcs = new Map();
let __rpcSeq = 0;
let __started = false;
const __rpcFetchUrl = (url) => new Promise((resolve, reject) => {
  const rpcId = ++__rpcSeq;
  __pendingRpcs.set(rpcId, { resolve, reject });
  __postToHost({ rpc: "fetch_url", rpcId, url: String(url ?? "") });
});
const __BLOCKED = ${JSON.stringify(WORKER_SCRUBBED_GLOBALS)};
let __proto = self;
while (__proto && __proto !== Object.prototype) {
  for (const k of __BLOCKED) {
    try { delete __proto[k]; } catch {}
  }
  __proto = Object.getPrototypeOf(__proto);
}
for (const k of __BLOCKED) {
  try {
    Object.defineProperty(self, k, { value: undefined, writable: false, configurable: false });
  } catch {}
}
self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg && typeof msg.rpcId === "number") {
    const pending = __pendingRpcs.get(msg.rpcId);
    if (pending) {
      __pendingRpcs.delete(msg.rpcId);
      if (msg.ok) pending.resolve(String(msg.text ?? ""));
      else pending.reject(new Error(String(msg.error ?? "ERR_FETCH_FAILED")));
    }
    return;
  }
  if (__started) return;
  __started = true;
  try {
    const { code, initialFilesJson } = msg;
    const AsyncFn = Object.getPrototypeOf(async function(){}).constructor;
    const GenFn = Object.getPrototypeOf(function*(){}).constructor;
    const AsyncGenFn = Object.getPrototypeOf(async function*(){}).constructor;
    const __runUserScript = new AsyncFn(
      "files", "readFile", "writeFile", "console", "tools", "listFiles", "searchFiles", "editFile", "fetchUrl",
      "'use strict';\\n" + code
    );
    const blockedCtor = () => { throw new Error("ERR_SCRIPT_SANDBOX_VIOLATION: Dynamic code generation is disabled"); };
    for (const proto of [Function.prototype, AsyncFn.prototype, GenFn.prototype, AsyncGenFn.prototype]) {
      try { Object.defineProperty(proto, "constructor", { value: blockedCtor, writable: false, configurable: false }); } catch {}
    }
    try { Object.defineProperty(self, "eval", { value: blockedCtor, writable: false, configurable: false }); } catch {}
    try { Object.defineProperty(self, "Function", { value: blockedCtor, writable: false, configurable: false }); } catch {}
    self.setTimeout = (cb, ms, ...rest) => {
      if (typeof cb !== "function") blockedCtor();
      return __origSetTimeout(cb, ms, ...rest);
    };
    self.setInterval = (cb, ms, ...rest) => {
      if (typeof cb !== "function") blockedCtor();
      return __origSetInterval(cb, ms, ...rest);
    };
    const __initialFilesJson = initialFilesJson;
    ${SANDBOX_SCRIPT_RUNNER_BODY}
    __postToHost({
      ok: true,
      logs,
      hasResult: result !== undefined,
      resultJson: result === undefined ? null : JSON.stringify(result),
      resultString: typeof result === "string" ? result : null,
      stagedUpdates,
    });
  } catch (err) {
    __postToHost({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
__postToHost({ ready: true });
`;
}

function decodeScriptWorkerResult(raw: unknown): IsolatedScriptExecutionResult {
  const data = raw as Record<string, unknown> | null;
  if (!data?.ok) {
    throw new Error(String(data?.error ?? "ERR_SCRIPT_FAILED"));
  }
  const hasResult = data.hasResult === true;
  const result = !hasResult
    ? undefined
    : typeof data.resultString === "string"
      ? data.resultString
      : typeof data.resultJson === "string"
        ? JSON.parse(data.resultJson)
        : undefined;
  return {
    logs: Array.isArray(data.logs) ? data.logs.map(String) : [],
    hasResult,
    result,
    stagedUpdates: (data.stagedUpdates as Record<string, string>) ?? {},
  };
}

function createProtectedScriptWorker(): Worker {
  const browserOrigin = (globalThis as { location?: { origin?: string } })
    .location?.origin;
  if (!browserOrigin || !/^https?:\/\//i.test(browserOrigin)) {
    throw new Error("ERR_SCRIPT_SANDBOX_UNAVAILABLE: Scripts require a Veles HTTP origin.");
  }
  return new Worker(`${browserOrigin}/api/sandbox/worker.js`);
}

async function handleWorkerFetchRpc(
  worker: Worker,
  rpcId: number,
  rawUrl: unknown,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const text = await executeHostValidatedTextFetch(rawUrl, signal);
    worker.postMessage({ rpcId, ok: true, text });
  } catch (err) {
    worker.postMessage({
      rpcId,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }
}

async function runScriptInBrowserWorker(
  code: string,
  initialFiles: Record<string, string>,
  signal?: AbortSignal,
): Promise<IsolatedScriptExecutionResult> {
  if (signal?.aborted) throw new Error("ERR_TOOL_ABORTED: Script execution cancelled");
  const worker = createProtectedScriptWorker();
  return new Promise<IsolatedScriptExecutionResult>((resolve, reject) => {
    let workerReady = false;
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
    };
    const onAbort = () => {
      cleanup();
      reject(new Error("ERR_TOOL_ABORTED: Script execution cancelled"));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("ERR_SCRIPT_TIMEOUT: Script execution timed out"));
    }, 5_000);
    signal?.addEventListener("abort", onAbort, { once: true });
    worker.onmessage = (ev: MessageEvent) => {
      const data = ev.data as Record<string, unknown>;
      if (data?.ready === true) {
        workerReady = true;
        return;
      }
      if (data?.rpc === "fetch_url" && typeof data.rpcId === "number") {
        void handleWorkerFetchRpc(worker, data.rpcId, data.url, signal);
        return;
      }
      cleanup();
      try {
        resolve(decodeScriptWorkerResult(data));
      } catch (err) {
        reject(err);
      }
    };
    worker.onerror = (err) => {
      cleanup();
      reject(new Error(workerReady
        ? err.message || "ERR_SCRIPT_WORKER_ERROR"
        : "ERR_SCRIPT_SANDBOX_UNAVAILABLE: The protected /api/sandbox/worker.js service is required (Vite dev or preview)."));
    };
    worker.postMessage({
      code,
      initialFilesJson: JSON.stringify(initialFiles),
    });
  });
}

async function runScriptInNodeVm(
  code: string,
  initialFiles: Record<string, string>,
  signal?: AbortSignal,
): Promise<IsolatedScriptExecutionResult> {
  const vm = await import("node:vm");
  const contextBridge = Object.create(null) as Record<string, unknown>;
  contextBridge.__hostFetchUrl = (rawUrl: unknown) =>
    executeHostValidatedTextFetch(rawUrl, signal);
  const context = vm.createContext(contextBridge, {
    codeGeneration: { strings: false, wasm: false },
  });
  const escapedFilesJson = JSON.stringify(JSON.stringify(initialFiles));
  const fullVmSource = `(async () => {
    "use strict";
    const __rpcFetchUrl = __hostFetchUrl;
    delete globalThis.__hostFetchUrl;
    const __initialFilesJson = ${escapedFilesJson};
    const __runUserScript = async function(files, readFile, writeFile, console, tools, listFiles, searchFiles, editFile, fetchUrl) {
      "use strict";
      ${code}
    };
    ${SANDBOX_SCRIPT_RUNNER_BODY}
    return JSON.stringify({
      logs,
      hasResult: result !== undefined,
      resultJson: result === undefined ? null : JSON.stringify(result),
      resultString: typeof result === "string" ? result : null,
      stagedUpdates,
    });
  })()`;
  const script = new vm.Script(fullVmSource, {
    filename: "workspace-script.vm.js",
    importModuleDynamically: () => {
      throw new Error("ERR_SCRIPT_SANDBOX_VIOLATION: Dynamic import() is not permitted");
    },
  });
  const rawPromise = script.runInContext(context, { timeout: 5_000 }) as Promise<string>;
  const rawJson = await rawPromise;
  const parsed = JSON.parse(rawJson) as {
    logs: string[];
    hasResult: boolean;
    resultJson: string | null;
    resultString: string | null;
    stagedUpdates: Record<string, string>;
  };
  const result = !parsed.hasResult
    ? undefined
    : typeof parsed.resultString === "string"
      ? parsed.resultString
      : typeof parsed.resultJson === "string"
        ? JSON.parse(parsed.resultJson)
        : undefined;
  return {
    logs: parsed.logs,
    hasResult: parsed.hasResult,
    result,
    stagedUpdates: parsed.stagedUpdates ?? {},
  };
}

async function executeIsolatedWorkspaceScript(
  code: string,
  initialFiles: Record<string, string>,
  signal?: AbortSignal,
): Promise<IsolatedScriptExecutionResult> {
  assertNoDynamicImportSyntax(code);
  const isNodeEnv =
    typeof process !== "undefined" && Boolean(process.versions?.node);
  if (isNodeEnv) return runScriptInNodeVm(code, initialFiles, signal);
  if (typeof Worker === "function") return runScriptInBrowserWorker(code, initialFiles, signal);
  throw new Error("ERR_SCRIPT_SANDBOX_UNAVAILABLE: Browser Workers are required.");
}

const runWorkspaceScriptTool: RegisteredBrowserTool = {
  spec: {
    name: "workspace.run_script",
    description:
      "Execute a JavaScript calculation or multi-step Code Mode (Programmatic Tool Calling) script in an isolated JS sandbox. Provides `files`, `readFile(path)`, `writeFile(path, data)`, `listFiles(prefix?)`, `searchFiles(query, pathFilter?)`, `editFile(path, target, replacement, replaceAll?)`, `await fetchUrl(url, saveToPath?)` (also available under `tools.workspace.*` and `tools.web.fetchUrl`), and `console.log(...)`.",
    effectClass: "idempotent_write",
    requiresApproval: false,
    parametersSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        saveResultToPath: { type: "string" },
      },
      required: ["code"],
    },
  },
  async handler(args, store, signal) {
    const code = nonEmptyString(args.code);
    if (!code) {
      throw new Error("ERR_SCRIPT_CODE_REQUIRED: 'code' must be a non-empty string");
    }
    const initialFiles = await loadWorkspaceTextMap(store);
    const { logs, hasResult, result, stagedUpdates: rawStaged } =
      await executeIsolatedWorkspaceScript(code, initialFiles, signal);
    const stagedUpdates: Record<string, Uint8Array | string | null> = {};
    for (const [rawPath, content] of Object.entries(rawStaged)) {
      const safePath = assertSafeWorkspacePath(rawPath);
      stagedUpdates[safePath] = String(content);
    }
    const saveResultRaw = nonEmptyString(args.saveResultToPath);
    if (saveResultRaw && hasResult) {
      const safeTarget = assertSafeWorkspacePath(saveResultRaw);
      stagedUpdates[safeTarget] =
        typeof result === "string" ? result : JSON.stringify(result, null, 2);
    }
    const meta = await store.getMetadata();
    if (meta?.runState !== "executing_tool") {
      for (const [safePath, content] of Object.entries(stagedUpdates)) {
        if (typeof content === "string" || content instanceof Uint8Array) {
          await store.writeWorkspaceFile(safePath, content);
        }
      }
    }
    return {
      outputText: formatScriptOutcomeText(
        logs,
        hasResult ? result : undefined,
        stagedUpdates,
      ),
      fileUpdates:
        Object.keys(stagedUpdates).length > 0 ? stagedUpdates : undefined,
    };
  },
};

export function createImageGenerationTools(
  provider: ImageGenerationProvider = defaultLocalImageProvider,
): RegisteredBrowserTool[] {
  const generateTool: RegisteredBrowserTool = {
    spec: {
      name: "image.generate",
      description:
        "Generate an image or game sprite from a text prompt and save it directly into the OPFS workspace (e.g. 'assets/sprite.svg' or 'assets/hero.png').",
      effectClass: "idempotent_write",
      requiresApproval: false,
      parametersSchema: {
        type: "object",
        properties: {
          outputPath: { type: "string" },
          prompt: { type: "string" },
          style: { type: "string" },
        },
        required: ["outputPath", "prompt"],
      },
    },
    async handler(args, _store, signal, context) {
      const outputPath = assertSafeWorkspacePath(args.outputPath ?? args.path);
      const prompt = nonEmptyString(args.prompt);
      if (!prompt) {
        throw new Error("ERR_IMAGE_PROMPT_REQUIRED: 'prompt' must be non-empty");
      }
      const style = nonEmptyString(args.style) ?? "game-asset";
      const generated = await provider.generateOrTransform({
        runId: context?.runId,
        prompt,
        style,
        signal,
      });
      const diagNote = generated.diagnostics
        ? ` [${generated.diagnostics}]`
        : "";
      return {
        outputText: `Generated image '${outputPath}' from prompt "${prompt}" (${style}).${diagNote}`,
        fileUpdates: { [outputPath]: generated.content },
        imageDataUrls: [generated.dataUrl],
      };
    },
  };

  const transformTool: RegisteredBrowserTool = {
    spec: {
      name: "image.transform",
      description:
        "Transform an existing workspace image (e.g. 'assets/horse.png') using a prompt/style (such as changing colors, lighting, or details) and commit the new image to `outputPath` in OPFS.",
      effectClass: "idempotent_write",
      requiresApproval: false,
      parametersSchema: {
        type: "object",
        properties: {
          sourcePath: { type: "string" },
          outputPath: { type: "string" },
          prompt: { type: "string" },
          style: { type: "string" },
        },
        required: ["sourcePath", "outputPath", "prompt"],
      },
    },
    async handler(args, store, signal, context) {
      const sourcePath = assertSafeWorkspacePath(args.sourcePath);
      const outputPath = assertSafeWorkspacePath(args.outputPath);
      const prompt = nonEmptyString(args.prompt);
      if (!prompt) {
        throw new Error("ERR_IMAGE_PROMPT_REQUIRED: 'prompt' must be non-empty");
      }
      const sourceRaw = await store.readFileTextOrDataUrl(sourcePath);
      const style = nonEmptyString(args.style) ?? "flux-4b-lora";
      const transformed = await provider.generateOrTransform({
        runId: context?.runId,
        prompt: `${prompt} [source:${sourcePath}]`,
        sourceDataUrl: sourceRaw,
        style,
        signal,
      });
      const diagNote = transformed.diagnostics
        ? ` [${transformed.diagnostics}]`
        : "";
      return {
        outputText: `Transformed '${sourcePath}' -> '${outputPath}' with style '${style}'.${diagNote}`,
        fileUpdates: { [outputPath]: transformed.content },
        imageDataUrls: [transformed.dataUrl],
      };
    },
  };

  return [generateTool, transformTool];
}

function assertValidChildSessionId(
  explicitId: string | null,
  parentId: string,
  coordinatorId: string | undefined,
  agentManager: AgentManager,
): void {
  if (!explicitId) return;
  if (
    explicitId === "session-main" ||
    explicitId === parentId ||
    explicitId === coordinatorId
  ) {
    throw new Error(
      `ERR_DELEGATE_RESERVED_SESSION_ID: Child sessionId '${explicitId}' is reserved`,
    );
  }
  const existing = agentManager.getSession(explicitId);
  if (existing && existing.parentSessionId !== parentId) {
    throw new Error(
      `ERR_DELEGATE_RESERVED_SESSION_ID: Session '${explicitId}' already belongs to another session`,
    );
  }
}

/**
 * Creates the `agent.delegate` tool allowing a parent agent session to spawn
 * an isolated child `AgentSession` with scoped `allowedTools` and optional model override.
 */
export function createSubagentDelegationTool(
  agentManager: AgentManager,
  options: SubagentDelegationOptions = {},
): RegisteredBrowserTool {
  let childCounter = 0;
  return {
    spec: {
      name: "agent.delegate",
      description:
        "Spawn an isolated child agent session to solve a focused sub-task with scoped tools and return its summary.",
      effectClass: "idempotent_write",
      requiresApproval: false,
      parametersSchema: {
        type: "object",
        properties: {
          role: { type: "string" },
          task: { type: "string" },
          sessionId: { type: "string" },
          parentSessionId: { type: "string" },
          allowedTools: { type: "array", items: { type: "string" } },
          model: { type: "string" },
          providerId: { type: "string" },
          baseUrl: { type: "string" },
        },
        required: ["task"],
      },
    },
    async handler(args, store, signal, context) {
      signal?.throwIfAborted();
      const task = nonEmptyString(args.task);
      if (!task) {
        throw new Error("ERR_DELEGATE_EMPTY_TASK: 'task' must be a non-empty string");
      }
      const rawRole = nonEmptyString(args.role) ?? "subagent";
      const role = rawRole.replace(/[^a-zA-Z0-9_-]/g, "-");
      childCounter += 1;
      const explicitId = nonEmptyString(args.sessionId);
      const childSessionId =
        explicitId ?? `sub-${role}-${childCounter}-${Date.now().toString(36).slice(-4)}`;
      const callerSession = context?.runId
        ? agentManager.resolveSessionForRun(context.runId, await store.getJournal())
        : undefined;
      const coordinatorSession = options.parentSessionId
        ? agentManager.getSession(options.parentSessionId)
        : undefined;
      const parentId =
        callerSession?.id ??
        options.parentSessionId ??
        nonEmptyString(args.parentSessionId) ??
        "session-main";
      assertValidChildSessionId(
        explicitId,
        parentId,
        options.parentSessionId,
        agentManager,
      );
      const parentSession = callerSession ?? agentManager.getSession(parentId);
      const baseModel: ProviderModelConfig =
        parentSession?.model ??
        coordinatorSession?.model ??
        options.defaultModelProvider?.() ?? {
          providerId: "openai",
          baseUrl: "https://api.openai.com/v1",
          model: "gpt-4o-mini",
        };

      signal?.throwIfAborted();
      await agentManager.createSession({
        id: childSessionId,
        parentSessionId: parentId,
        workspaceId: store.getActiveWorkspaceId(),
        model: resolveChildModelConfig(args, baseModel),
        allowedTools: resolveSafeChildTools(args, parentSession, coordinatorSession),
        unrestrictedTools: false,
        status: "idle",
      });

      const framedPrompt = `[Subagent Role: ${role}]\nTask: ${task}`;
      signal?.throwIfAborted();
      const outcome = await agentManager.runWithId(childSessionId, framedPrompt, undefined, signal);
      return {
        outputText: JSON.stringify({
          childSessionId,
          role,
          state: outcome.state,
          summary: outcome.assistantText,
        }),
      };
    },
  };
}

export const createAgentDelegateTool = createSubagentDelegationTool;

/**
 * Returns both `agent.delegate` and its `agent.spawn` alias so prompts or models
 * using either tool name route cleanly to the isolated child `AgentSession` spawner.
 */
export function createSubagentTools(
  agentManager: AgentManager,
  options: SubagentDelegationOptions = {},
): RegisteredBrowserTool[] {
  const delegate = createSubagentDelegationTool(agentManager, options);
  const spawnAlias: RegisteredBrowserTool = {
    spec: {
      ...delegate.spec,
      name: "agent.spawn",
      description:
        "Alias for agent.delegate: spawn an isolated child AgentSession with scoped tools and optional model override.",
    },
    handler: delegate.handler,
  };
  return [delegate, spawnAlias];
}

export function createDefaultWorkspaceTools(
  modelConfigProvider?: (runId?: string) => ProviderModelConfig | undefined,
  serviceConfigProvider?: () => ServiceCapabilitiesConfigSpec | undefined,
): RegisteredBrowserTool[] {
  const imgProvider = modelConfigProvider
    ? createOpenAIImageProvider(
        modelConfigProvider,
        defaultLocalImageProvider,
        () => serviceConfigProvider?.()?.imageMode ?? "openai-auto",
      )
    : defaultLocalImageProvider;
  return [
    readWorkspaceFileTool,
    inspectWorkspaceImageTool,
    listWorkspaceFilesTool,
    writeWorkspaceFileTool,
    editWorkspaceFileTool,
    runWorkspaceScriptTool,
    deleteWorkspaceFileTool,
    fetchUrlWorkspaceTool,
    createWebSearchTool(
      modelConfigProvider,
      () => serviceConfigProvider?.()?.webSearchMode ?? "openai-responses",
      () => serviceConfigProvider?.()?.searxngUrl,
    ),
    ...createImageGenerationTools(imgProvider),
  ];
}

export class BrowserToolRegistry implements ToolExecutionAdapter {
  public readonly currentGatewayInstanceId: GatewayInstanceId =
    LOCAL_BROWSER_INSTANCE_ID;
  private readonly store: WorkspaceKernelStore;
  private readonly tools = new Map<string, RegisteredBrowserTool>();
  private readonly remoteToolSpecs = new Map<string, ToolDefinitionSpec>();
  private readonly fallbackGateway: ToolExecutionAdapter | null;

  constructor(
    store: WorkspaceKernelStore,
    customTools: readonly RegisteredBrowserTool[] = createDefaultWorkspaceTools(),
    fallbackGateway: ToolExecutionAdapter | null = null,
  ) {
    this.store = store;
    this.fallbackGateway = fallbackGateway;
    for (const tool of customTools) {
      this.registerTool(tool);
    }
  }

  public registerTool(tool: RegisteredBrowserTool): void {
    if (!tool.spec.name.trim()) {
      throw new Error("ERR_TOOL_INVALID_NAME: Tool name cannot be empty");
    }
    this.tools.set(tool.spec.name, tool);
  }

  public forkForStore(
    targetStore: WorkspaceKernelStore,
    workspaceTools?: readonly RegisteredBrowserTool[],
  ): BrowserToolRegistry {
    const nonSubagentTools = workspaceTools ?? Array.from(this.tools.values()).filter(
      (t) => t.spec.name !== "agent.delegate" && t.spec.name !== "agent.spawn",
    );
    const forked = new BrowserToolRegistry(
      targetStore,
      nonSubagentTools,
      this.fallbackGateway,
    );
    forked.registerGatewayToolSpecs(Array.from(this.remoteToolSpecs.values()));
    return forked;
  }

  /**
   * Registers tool schemas discovered from the remote Ephemeral Execution Gateway
   * (`GET /v1/capabilities`) so the LLM can invoke both browser and gateway tools seamlessly.
   */
  public registerGatewayToolSpecs(specs: readonly ToolDefinitionSpec[]): void {
    for (const spec of specs) {
      if (spec.name.trim() && !this.tools.has(spec.name)) {
        this.remoteToolSpecs.set(spec.name, spec);
      }
    }
  }

  public getToolDefinitions(): ToolDefinitionSpec[] {
    const localSpecs = Array.from(this.tools.values()).map((t) => t.spec);
    const gatewaySpecs = Array.from(this.remoteToolSpecs.values());
    return [...localSpecs, ...gatewaySpecs];
  }

  public getEffectClass(toolName: string): ToolEffectClass {
    return (
      this.tools.get(toolName)?.spec.effectClass ??
      this.remoteToolSpecs.get(toolName)?.effectClass ??
      "non_idempotent"
    );
  }

  public async executeTool(
    request: ToolExecutionRequest,
  ): Promise<ToolExecutionOutcome> {
    if (request.signal?.aborted) {
      throw new Error("ERR_TOOL_ABORTED: Tool execution cancelled before start");
    }
    const localTool = this.tools.get(request.toolName);
    if (!localTool) {
      if (this.fallbackGateway) {
        return this.fallbackGateway.executeTool(request);
      }
      throw new Error(`ERR_TOOL_NOT_REGISTERED: ${request.toolName}`);
    }

    const result = await localTool.handler(
      request.args,
      this.store,
      request.signal,
      { runId: request.runId },
    );
    return {
      gatewayInstanceId: this.currentGatewayInstanceId,
      outputText: result.outputText,
      fileUpdates: result.fileUpdates,
      imageDataUrls: result.imageDataUrls,
    };
  }

  public async queryOperationStatus(
    operationId: OperationId,
    gatewayInstanceId: GatewayInstanceId,
  ): Promise<{ readonly completed: boolean; readonly outcome?: ToolExecutionOutcome }> {
    if (!this.fallbackGateway?.queryOperationStatus) {
      return { completed: false };
    }
    return this.fallbackGateway.queryOperationStatus(
      operationId,
      gatewayInstanceId,
    );
  }
}
