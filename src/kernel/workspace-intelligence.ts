/**
 * @module SYS-KERNEL/workspace-intelligence
 * Mechanical pre-commit verification ("Bend Law Checks") for workspace writes
 * and zero-round structural file signature extraction for `ContextBuilder`.
 *
 * Enforces !INVARIANT [Coupling_Cohesion_Law] (Data Coupling) and
 * !INVARIANT [McCabe_Cyclomatic_Bound] (V(G) <= 10 per function).
 */
import type { WorkspaceKernelStore } from "./journal.ts";

export const MAX_WORKSPACE_SIGNATURE_CHARS = 8_000;
const MAX_FILE_SIGNATURE_CHARS = 256;
const MAX_SIGNATURE_FIELD_CHARS = 64;
const MAX_SIGNATURE_CACHE_ENTRIES = 512;

function compactSignatureText(value: string, maxChars: number): string {
  const line = value.replace(/[\u0000-\u0020\u007f-\u009f\u2028\u2029]+/g, " ").trim();
  return line.length <= maxChars ? line : `${line.slice(0, maxChars - 1)}…`;
}

export function boundWorkspaceFileSignature(signature: string): string {
  return compactSignatureText(signature, MAX_FILE_SIGNATURE_CHARS);
}

function formatSignatureField(value: string): string {
  return compactSignatureText(value, MAX_SIGNATURE_FIELD_CHARS);
}

function workspaceSignatureExtension(path: string): string | undefined {
  return /\.(csv|tsv|json|html?|m?js|tsx?|svg)$/i.exec(path)?.[1]?.toLowerCase();
}

function stripCommentsPreserveStrings(code: string): string {
  return code.replace(
    /("(?:\\[\s\S]|[^"\\])*")|('(?:\\[\s\S]|[^'\\])*')|(`(?:\\[\s\S]|[^`\\])*`)|\/\*[\s\S]*?\*\/|\/\/[^\r\n]*/g,
    (_m, dq, sq, tq) => (dq || sq || tq ? "''" : " "),
  );
}

function checkClassicScriptSyntax(
  scriptBody: string,
  label: string,
): string | null {
  const trimmed = scriptBody.trim();
  if (!trimmed) return null;
  try {
    new Function(trimmed);
    return null;
  } catch (err) {
    if (!(err instanceof SyntaxError)) return null;
    const hint = /\bawait\b/.test(err.message)
      ? " (avoid top-level await in classic <script> tags; wrap in '(async () => { ... })()' or use a synchronous helper)"
      : "";
    return `JS SyntaxError in ${label}: ${err.message}${hint}`;
  }
}

function checkSandboxBridgeViolations(
  code: string,
  label: string,
): string[] {
  const warnings: string[] = [];
  const stripped = stripCommentsPreserveStrings(code);
  if (
    /\b(localStorage|sessionStorage)\s*\./.test(stripped) &&
    !/\btry\s*\{/.test(stripped)
  ) {
    warnings.push(
      `Direct localStorage/sessionStorage access in ${label} throws SecurityError in sandbox="allow-scripts"; use window.veles.kv.get(k) / kv.set(k, v).`,
    );
  }
  if (/\.kv\.(get|getItem)\s*\([^)]*\)\s*\.then\s*\(/.test(stripped)) {
    warnings.push(
      `window.veles.kv.get(k) in ${label} returns synchronously, not a Promise; remove .then().`,
    );
  }
  if (/\b(?:window\.)?(alert|confirm|prompt)\s*\(/.test(stripped)) {
    warnings.push(
      `Modal alert()/confirm()/prompt() calls in ${label} are blocked by sandbox="allow-scripts"; render an in-page DOM/canvas overlay instead.`,
    );
  }
  return warnings;
}

function checkGameplayInputPitfalls(
  rawCode: string,
  label: string,
): string[] {
  const warnings: string[] = [];
  const hasDotKey =
    /\.key\s*===\s*['"]Space['"]/.test(rawCode) ||
    (/\[\s*[a-zA-Z_$][a-zA-Z0-9_$]*\.key\s*\]/.test(rawCode) &&
      /\[\s*['"]Space['"]\s*\]/.test(rawCode) &&
      !/\[\s*['"] ['"]\s*\]/.test(rawCode));
  if (hasDotKey) {
    warnings.push(
      `KeyboardEvent.key for Spacebar in ${label} is ' ' (single space), not 'Space' (which is event.code); check keys[' '] || keys['Space'] or event.code === 'Space'.`,
    );
  }
  if (
    /\bsnake\.unshift\s*\(/.test(rawCode) &&
    /\bsnake\.some\s*\(/.test(rawCode) &&
    !/\bsnake\.slice\s*\(\s*1\s*\)/.test(rawCode) &&
    /\blet\s+head\s*=\s*snake\[\s*0\s*\]/.test(rawCode)
  ) {
    warnings.push(
      `${label} checks snake.some(...) with snake[0] after snake.unshift(head), causing immediate self-collision on every tick; check snake.slice(1).some(...) instead.`,
    );
  }
  return warnings;
}

function checkPlaceholderStubComments(
  rawCode: string,
  label: string,
): string | null {
  const placeholderRe =
    /\/\/[^\r\n]*\b(chart|plot|graph|canvas|game|render\w*)\s+(rendering\s+|drawing\s+)?logic\b[^\r\n]*\bgoes\s+here\b/i;
  if (placeholderRe.test(rawCode)) {
    return `${label} contains a placeholder comment instead of working rendering code; implement complete <canvas> or <svg> drawing logic.`;
  }
  return null;
}

function verifyHtmlScriptBlock(
  label: string,
  attrs: string,
  body: string,
  hasChartJsTag: boolean,
): string[] {
  const warnings: string[] = [];
  const isModule = /\btype\s*=\s*["']module["']/i.test(attrs);
  if (!isModule) {
    const syntaxErr = checkClassicScriptSyntax(body, label);
    if (syntaxErr) warnings.push(syntaxErr);
  }
  warnings.push(...checkSandboxBridgeViolations(body, label));
  warnings.push(...checkGameplayInputPitfalls(body, label));
  const stubWarn = checkPlaceholderStubComments(body, label);
  if (stubWarn) warnings.push(stubWarn);
  const stripped = stripCommentsPreserveStrings(body);
  if (/\bnew\s+Chart\s*\(/.test(stripped) && !hasChartJsTag) {
    warnings.push(
      `${label} calls new Chart(...) without loading Chart.js in <script src="...">; draw directly on <canvas> 2D context or include the library script tag.`,
    );
  }
  return warnings;
}

function verifyHtmlPreflight(path: string, html: string): string[] {
  const warnings: string[] = [];
  const hasChartJsTag = /<script\b[^>]*src=["'][^"']*chart(\.umd|\.min)?\.js/i.test(
    html,
  );
  const scriptRe = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let match: RegExpExecArray | null;
  let idx = 0;
  while ((match = scriptRe.exec(html)) !== null) {
    const attrs = match[1] ?? "";
    if (
      /\bsrc\s*=/.test(attrs) ||
      /\btype\s*=\s*["'](application\/(ld\+)?json|importmap)["']/i.test(attrs)
    ) {
      continue;
    }
    idx += 1;
    warnings.push(
      ...verifyHtmlScriptBlock(
        `${path} inline <script #${idx}>`,
        attrs,
        match[2] ?? "",
        hasChartJsTag,
      ),
    );
  }
  return warnings;
}

function verifyJsPreflight(path: string, code: string): string[] {
  const warnings: string[] = [];
  const stripped = stripCommentsPreserveStrings(code);
  const isEsModule =
    /\.mjs$/i.test(path) || /\b(import|export)\b/.test(stripped);
  if (!isEsModule) {
    const syntaxErr = checkClassicScriptSyntax(code, path);
    if (syntaxErr) warnings.push(syntaxErr);
  }
  warnings.push(...checkSandboxBridgeViolations(code, path));
  warnings.push(...checkGameplayInputPitfalls(code, path));
  const stubWarn = checkPlaceholderStubComments(code, path);
  if (stubWarn) warnings.push(stubWarn);
  return warnings;
}

function verifyJsonPreflight(path: string, content: string): string[] {
  try {
    JSON.parse(content);
    return [];
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return [`Invalid JSON in ${path}: ${msg}`];
  }
}

/**
 * Runs sub-millisecond mechanical verification ("Bend Law Checks") on workspace
 * files written via `workspace.write_file` or `workspace.edit_file`.
 */
export function verifyWorkspaceFilePreflight(
  path: string,
  content: string,
): string[] {
  if (/\.html?$/i.test(path)) {
    return verifyHtmlPreflight(path, content);
  }
  if (/\.(m?js)$/i.test(path)) {
    return verifyJsPreflight(path, content);
  }
  if (/\.json$/i.test(path)) {
    return verifyJsonPreflight(path, content);
  }
  return [];
}

export function formatPreflightDiagnosticSuffix(
  warnings: readonly string[],
): string {
  if (warnings.length === 0) return "";
  return (
    "\n" +
    warnings.map((w) => `[PREFLIGHT WARNING: ${w}]`).join("\n")
  );
}

function summarizeCsvSignature(
  content: string,
  delimiter: string,
  kind: "csv" | "tsv",
): string | null {
  const lines = content
    .trim()
    .split(/\r?\n/)
    .filter((l) => l.trim().length > 0);
  if (lines.length === 0) return `${kind}: empty`;
  const headers = lines[0]!
    .split(delimiter)
    .map((h) => h.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean)
    .slice(0, 8)
    .map(formatSignatureField);
  const rowCount = Math.max(0, lines.length - 1);
  return `${kind}: ${rowCount} rows, cols: [${headers.join(", ")}]`;
}

function summarizeJsonSignature(content: string): string | null {
  try {
    const parsed: unknown = JSON.parse(content);
    if (Array.isArray(parsed)) {
      const first = parsed[0];
      const itemKeys =
        typeof first === "object" && first !== null && !Array.isArray(first)
          ? Object.keys(first as Record<string, unknown>).slice(0, 8).map(formatSignatureField)
          : [];
      return itemKeys.length > 0
        ? `json: array[${parsed.length}], itemKeys: [${itemKeys.join(", ")}]`
        : `json: array[${parsed.length}]`;
    }
    if (typeof parsed === "object" && parsed !== null) {
      const keys = Object.keys(parsed as Record<string, unknown>)
        .slice(0, 8).map(formatSignatureField);
      return `json: object, keys: [${keys.join(", ")}]`;
    }
    return `json: ${typeof parsed}`;
  } catch {
    return "json: invalid";
  }
}

function extractRegexMatches(
  text: string,
  re: RegExp,
  limit: number,
  prefix = "",
): string[] {
  const found: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null && found.length < limit) {
    const val = match[1]?.trim();
    if (val) {
      const field = formatSignatureField(`${prefix}${val}`);
      if (!found.includes(field)) found.push(field);
    }
  }
  return found;
}

function summarizeHtmlSignature(html: string): string | null {
  const parts: string[] = [];
  if (/<canvas\b/i.test(html)) parts.push("<canvas>");
  if (/<svg\b/i.test(html)) parts.push("<svg>");
  const refs = [
    ...extractRegexMatches(html, /<link\b[^>]*href=["']([^"']+)["']/gi, 4),
    ...extractRegexMatches(html, /<script\b[^>]*src=["']([^"']+)["']/gi, 4),
  ].slice(0, 5);
  if (refs.length > 0) parts.push(`refs: [${refs.join(", ")}]`);
  const ids = extractRegexMatches(html, /\bid=["']([a-zA-Z0-9_-]+)["']/gi, 6, "#");
  if (ids.length > 0) parts.push(`ids: [${ids.join(", ")}]`);
  return parts.length > 0 ? `html: ${parts.join(", ")}` : "html";
}

function summarizeJsSignature(code: string): string | null {
  const fns = [
    ...extractRegexMatches(
      code,
      /\b(?:async\s+)?function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(/g,
      6,
    ),
    ...extractRegexMatches(
      code,
      /\b(?:const|let)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>/g,
      4,
    ),
  ].slice(0, 6);
  return fns.length > 0 ? `js: fns: [${fns.join(", ")}]` : null;
}

/**
 * Extracts a compact, single-line structural signature for a workspace file
 * so `ContextBuilder` can expose dataset columns, JSON keys, HTML IDs, and JS functions
 * directly in the OPFS workspace tree without extra tool round-trips.
 */
function extractWorkspaceFileSignature(
  path: string,
  content: string,
): string | null {
  if (/\.csv$/i.test(path)) return summarizeCsvSignature(content, ",", "csv");
  if (/\.tsv$/i.test(path)) return summarizeCsvSignature(content, "\t", "tsv");
  if (/\.json$/i.test(path)) return summarizeJsonSignature(content);
  if (/\.html?$/i.test(path)) return summarizeHtmlSignature(content);
  if (/\.(m?js|tsx?)$/i.test(path)) return summarizeJsSignature(content);
  if (/\.svg$/i.test(path)) {
    const vb = /\bviewBox=["']([^"']+)["']/i.exec(content)?.[1];
    return vb ? `svg: viewBox="${formatSignatureField(vb)}"` : "svg";
  }
  return null;
}

export function summarizeWorkspaceFileSignature(
  path: string,
  content: string,
): string | null {
  if (!content || content.startsWith("data:")) return null;
  const signature = extractWorkspaceFileSignature(path, content);
  return signature === null ? null : boundWorkspaceFileSignature(signature);
}

/** Runtime-owned, bounded cache; failures remain retryable and never poison a hash. */
export class WorkspaceFileSignatureCollector {
  private readonly cache = new Map<string, string | null>();

  private remember(key: string, signature: string | null): void {
    this.cache.delete(key);
    this.cache.set(key, signature);
    if (this.cache.size > MAX_SIGNATURE_CACHE_ENTRIES) {
      this.cache.delete(this.cache.keys().next().value!);
    }
  }

  private async readSignature(
    store: WorkspaceKernelStore,
    path: string,
    hash: string,
  ): Promise<string | null> {
    const text = await store.readFileText(path, hash);
    return text.length <= 250_000 ? summarizeWorkspaceFileSignature(path, text) : null;
  }

  public async collect(
    store: WorkspaceKernelStore,
    fileTree: Readonly<Record<string, string>>,
    signal?: AbortSignal,
  ): Promise<Record<string, string>> {
    const signatures: Record<string, string> = {};
    for (const [path, hash] of Object.entries(fileTree)) {
      signal?.throwIfAborted();
      const extension = workspaceSignatureExtension(path);
      if (!extension) continue;
      const key = `${extension}:${hash}`;
      try {
        let signature = this.cache.get(key);
        if (signature === undefined) {
          signature = await this.readSignature(store, path, hash);
        }
        signal?.throwIfAborted();
        this.remember(key, signature);
        if (signature) signatures[path] = signature;
      } catch {
        signal?.throwIfAborted();
        // Missing, changed, or unreadable files may succeed on the next round.
      }
    }
    return signatures;
  }
}
