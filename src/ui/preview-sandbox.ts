/**
 * @module SYS-UI/preview-sandbox
 * Origin-isolated preview host enforcing `!INVARIANT [SEC-INV-01]` (Scenario 8).
 * Renders generated HTML/JS artifacts inside `<iframe sandbox="allow-scripts">`
 * strictly WITHOUT `allow-same-origin` so `indexedDB`, `OPFS`, and host cookies/storage
 * remain unreachable from generated code.
 * Cyclomatic complexity V(G) <= 10 per function.
 */

export const REQUIRED_SANDBOX_FLAGS = "allow-scripts" as const;
const FORBIDDEN_SANDBOX_TOKENS: ReadonlySet<string> = new Set([
  "allow-same-origin",
  "allow-top-navigation",
  "allow-popups-to-escape-sandbox",
]);

export interface SandboxIframeLike {
  sandbox: string;
  srcdoc: string;
  onLoad?: (() => void) | null;
  readonly contentWindow: {
    postMessage(message: unknown, targetOrigin: string): void;
  } | null;
  remove(): void;
}

export interface PreviewSensorSample {
  readonly tiltX: number;
  readonly tiltY: number;
  readonly accelX?: number;
  readonly accelY?: number;
  readonly accelZ?: number;
  readonly pointerX: number;
  readonly pointerY: number;
  readonly timestampMs: number;
}

export interface SandboxEnvironmentLike {
  createIframe(): SandboxIframeLike;
  addMessageListener(handler: (event: MessageEventLike) => void): () => void;
}

export interface MessageEventLike {
  readonly source: unknown;
  readonly origin: string;
  readonly data: unknown;
}

export interface PreviewArtifactBundle {
  readonly workspaceId?: string;
  readonly entryHtml: string;
  readonly entryPath?: string;
  readonly workspaceSeq?: number;
  readonly files: Readonly<Record<string, string>>;
}

export function pickDefaultWorkspaceFile(
  files: Readonly<Record<string, string>>,
): string | null {
  if (files["preview/index.html"] !== undefined) return "preview/index.html";
  if (files["index.html"] !== undefined) return "index.html";
  const paths = Object.keys(files);
  return paths.find((p) => /\.html$/i.test(p)) ?? paths[0] ?? null;
}

export function resolvePreviewEntryPath(
  files: Readonly<Record<string, string>>,
  selectedFile: string | null,
): string {
  if (
    selectedFile &&
    files[selectedFile] !== undefined &&
    /\.html$/i.test(selectedFile)
  ) {
    return selectedFile;
  }
  const defaultFile = pickDefaultWorkspaceFile(files);
  if (defaultFile && /\.html$/i.test(defaultFile)) {
    return defaultFile;
  }
  if (
    selectedFile &&
    files[selectedFile] !== undefined &&
    /\.(svg|png|jpe?g|webp|gif)$/i.test(selectedFile)
  ) {
    return selectedFile;
  }
  return defaultFile ?? "index.html";
}

function normalizeAssetPath(rawRef: string, baseDir: string): string {
  const clean = rawRef.split(/[?#]/)[0]!.trim().replace(/^\.\/+/, "").replace(/^\/+/, "");
  const combined = baseDir ? `${baseDir}/${clean}` : clean;
  const parts: string[] = [];
  for (const seg of combined.split("/")) {
    if (!seg || seg === ".") continue;
    if (seg === "..") {
      parts.pop();
    } else {
      parts.push(seg);
    }
  }
  return parts.join("/");
}

function findAssetByBasename(
  cleanRef: string,
  files: Readonly<Record<string, string>>,
): { readonly path: string; readonly content: string } | null {
  const baseName = cleanRef.split("/").pop();
  if (!baseName) return null;
  for (const [filePath, content] of Object.entries(files)) {
    if (filePath.split("/").pop() === baseName) {
      return { path: filePath, content };
    }
  }
  return null;
}

function lookupWorkspaceAsset(
  rawRef: string,
  files: Readonly<Record<string, string>>,
  baseDir: string,
): { readonly path: string; readonly content: string } | null {
  if (/^(https?:|data:|blob:|\/\/)/i.test(rawRef.trim())) {
    return null;
  }
  const clean = rawRef.split(/[?#]/)[0]!.trim().replace(/^\.\/+/, "").replace(/^\/+/, "");
  const candidates = [
    normalizeAssetPath(rawRef, baseDir),
    clean,
    `preview/${clean}`,
    `scripts/${clean}`,
    `assets/${clean}`,
  ];
  for (const cand of candidates) {
    if (cand && Object.prototype.hasOwnProperty.call(files, cand)) {
      return { path: cand, content: files[cand]! };
    }
  }
  return findAssetByBasename(clean, files);
}

function toImageDataUrl(path: string, content: string): string | null {
  const trimmed = content.trim();
  if (trimmed.startsWith("data:")) return trimmed;
  if (path.endsWith(".svg") || trimmed.startsWith("<svg")) {
    return `data:image/svg+xml;utf8,${encodeURIComponent(content).replace(/'/g, "%27")}`;
  }
  return null;
}

function inlineScriptImagePaths(
  scriptContent: string,
  files: Readonly<Record<string, string>>,
  baseDir: string,
): string {
  return scriptContent.replace(
    /(['"])([^'"\r\n]+\.(?:png|jpe?g|webp|gif|svg))\1/gi,
    (fullMatch, _quote, rawPath) => {
      const asset = lookupWorkspaceAsset(String(rawPath), files, baseDir);
      if (!asset) return fullMatch;
      const dataUrl = toImageDataUrl(asset.path, asset.content);
      return dataUrl
        ? JSON.stringify(dataUrl).replace(/<\/script/gi, "<\\/script")
        : fullMatch;
    },
  );
}

function safeJsonForInlineScript(value: unknown): string {
  return JSON.stringify(value).replace(/<\/script/gi, "<\\/script");
}

function extractTextWorkspaceFiles(
  files?: Readonly<Record<string, string>>,
): Record<string, string> {
  const out: Record<string, string> = {};
  if (!files) return out;
  for (const [k, v] of Object.entries(files)) {
    if (typeof v === "string" && !v.startsWith("data:")) {
      out[k] = v;
    }
  }
  return out;
}

function buildErrorBridgeScript(
  generation: number,
  workspaceSeq = 0,
  bundle?: PreviewArtifactBundle,
  initialKv?: Readonly<Record<string, unknown>>,
  baseDir = "",
): string {
  const wsKey = bundle?.workspaceId?.trim() || "default";
  const kvJson = safeJsonForInlineScript(initialKv ?? {});
  const filesJson = safeJsonForInlineScript(
    extractTextWorkspaceFiles(bundle?.files),
  );
  const baseDirJson = safeJsonForInlineScript(baseDir);
  const wsPrefixJson = safeJsonForInlineScript(`__veles_kv__:${wsKey}:`);
  return `<script data-veles-error-bridge="true">
(function() {
  var _origW = Object.getOwnPropertyDescriptor(window, "innerWidth");
  var _origH = Object.getOwnPropertyDescriptor(window, "innerHeight");
  if (_origW && typeof _origW.get === "function") {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      get: function() {
        var w = _origW.get.call(window);
        return w > 0 ? w : (document.documentElement && document.documentElement.clientWidth) || 800;
      }
    });
  }
  if (_origH && typeof _origH.get === "function") {
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      get: function() {
        var h = _origH.get.call(window);
        return h > 0 ? h : (document.documentElement && document.documentElement.clientHeight) || 600;
      }
    });
  }
  var _wnPrefix = ${wsPrefixJson};
  var _kv = Object.assign(Object.create(null), ${kvJson});
  var _hasPersistedWn = false;
  var _kvTouched = Object.create(null);
  try {
    if (typeof window.name === "string" && window.name.indexOf(_wnPrefix) === 0) {
      _kv = Object.assign(Object.create(null), JSON.parse(window.name.slice(_wnPrefix.length)));
      _hasPersistedWn = true;
    }
  } catch (_e) {}
  var _files = Object.assign(Object.create(null), ${filesJson});
  var _baseDir = ${baseDirJson};
  var _origFetch = typeof window.fetch === "function" ? window.fetch.bind(window) : null;
  function _resolveLocalFile(rawUrl) {
    var clean = String(rawUrl || "").split(/[?#]/)[0].trim().replace(/^\\.?\\/+/, "");
    if (!clean) return null;
    var cands = [_baseDir ? _baseDir + "/" + clean : clean, clean, "preview/" + clean, "assets/" + clean];
    for (var i = 0; i < cands.length; i++) {
      if (Object.prototype.hasOwnProperty.call(_files, cands[i])) return { path: cands[i], body: _files[cands[i]] };
    }
    var base = clean.split("/").pop();
    var keys = Object.keys(_files);
    for (var k = 0; k < keys.length; k++) {
      if (keys[k].split("/").pop() === base) return { path: keys[k], body: _files[keys[k]] };
    }
    return null;
  }
  if (_origFetch && typeof Response === "function") {
    window.fetch = function(input, init) {
      if (typeof input === "string" && !/^(https?:|data:|blob:)/i.test(input)) {
        var hit = _resolveLocalFile(input);
        if (hit) {
          var mime = /\\.json$/i.test(hit.path) ? "application/json" : /\\.csv$/i.test(hit.path) ? "text/csv" : "text/plain";
          return Promise.resolve(new Response(hit.body, { status: 200, headers: { "Content-Type": mime } }));
        }
      }
      return _origFetch(input, init);
    };
  }
  var _sensorListeners = [];
  var _latestSensor = { tiltX: 0, tiltY: 0, accelX: 0, accelY: 0, accelZ: 0, pointerX: 0, pointerY: 0, timestampMs: 0 };
  function _kvGet(k) { return _kv[String(k)] !== undefined ? _kv[String(k)] : null; }
  function _kvSet(k, v) {
    _kvTouched[String(k)] = true;
    _kv[String(k)] = v;
    try { window.name = _wnPrefix + JSON.stringify(_kv); } catch (_e) {}
    parent.postMessage({ channel: "veles:preview", generation: ${generation}, workspaceSeq: ${workspaceSeq}, type: "kv_set", key: String(k), value: v }, "*");
    return Promise.resolve();
  }
  function _kvRemove(k) {
    _kvTouched[String(k)] = true;
    delete _kv[String(k)];
    try { window.name = _wnPrefix + JSON.stringify(_kv); } catch (_e) {}
    parent.postMessage({ channel: "veles:preview", generation: ${generation}, workspaceSeq: ${workspaceSeq}, type: "kv_set", key: String(k), value: null, removed: true }, "*");
    return Promise.resolve();
  }
  window.veles = {
    files: _files,
    kv: {
      get: _kvGet,
      getItem: _kvGet,
      set: _kvSet,
      setItem: _kvSet,
      remove: _kvRemove,
      removeItem: _kvRemove,
      delete: _kvRemove,
      all: function() { return Object.assign({}, _kv); },
      keys: function() { return Object.keys(_kv); }
    },
    log: function(msg, detail) {
      var extra = detail !== undefined ? " " + (typeof detail === "string" ? detail : JSON.stringify(detail)) : "";
      parent.postMessage({ channel: "veles:preview", generation: ${generation}, workspaceSeq: ${workspaceSeq}, type: "preview_log", message: String(msg) + extra }, "*");
    },
    sensors: {
      get latest() { return _latestSensor; },
      onUpdate: function(cb) {
        if (typeof cb === "function") _sensorListeners.push(cb);
        return function() { _sensorListeners = _sensorListeners.filter(function(f) { return f !== cb; }); };
      }
    }
  };
  window.addEventListener("message", function(ev) {
    var d = ev.data;
    if (!d || typeof d !== "object" || d.generation !== ${generation}) return;
    if (d.channel === "veles:host-init") {
      if (!_hasPersistedWn && d.kv && typeof d.kv === "object") {
        var hydratedKv = Object.assign(Object.create(null), d.kv);
        Object.keys(_kvTouched).forEach(function(k) {
          if (Object.prototype.hasOwnProperty.call(_kv, k)) hydratedKv[k] = _kv[k];
          else delete hydratedKv[k];
        });
        _kv = hydratedKv;
      }
      if (d.files && typeof d.files === "object") Object.assign(_files, d.files);
    }
    if (d.channel === "veles:host-sensor" && d.sample) {
      _latestSensor = d.sample;
      for (var j = 0; j < _sensorListeners.length; j++) {
        try { _sensorListeners[j](d.sample); } catch (_e) {}
      }
    }
  });
})();
window.addEventListener("error", function(ev) {
  parent.postMessage({ channel: "veles:preview", generation: ${generation}, workspaceSeq: ${workspaceSeq}, type: "preview_error", message: String(ev.message || ev.error || "Runtime error") }, "*");
});
window.addEventListener("unhandledrejection", function(ev) {
  parent.postMessage({ channel: "veles:preview", generation: ${generation}, workspaceSeq: ${workspaceSeq}, type: "preview_error", message: "Unhandled rejection: " + String(ev.reason) }, "*");
});
</script>`;
}

/**
 * Resolves relative workspace assets (`style.css`, `script.js`, `.svg`/data-URL images)
 * from `bundle.files` and inlines them into `srcdoc` so multi-file apps and games
 * execute inside `<iframe sandbox="allow-scripts">` without a virtual HTTP server.
 */
export function compileMultiFileSrcdoc(
  bundle: PreviewArtifactBundle,
  generation = 1,
  initialKv?: Readonly<Record<string, unknown>>,
): string {
  const rawHtml = bundle.entryHtml;
  if (rawHtml.trim().startsWith("<svg")) {
    return `<!doctype html><html><head><meta charset="utf-8"/><style>html,body{margin:0;height:100%;display:flex;align-items:center;justify-content:center;background:#101c25;color:#dbe5e4;}</style></head><body>${rawHtml}</body></html>`;
  }
  if (rawHtml.trim().startsWith("data:image/")) {
    return `<!doctype html><html><head><meta charset="utf-8"/><style>html,body{margin:0;height:100%;display:flex;align-items:center;justify-content:center;background:#101c25;color:#dbe5e4;}img{max-width:94%;max-height:94%;object-fit:contain;border-radius:8px;}</style></head><body><img src="${rawHtml.trim()}" alt="Preview"/></body></html>`;
  }

  const entryPath = bundle.entryPath ?? "index.html";
  const slashIdx = entryPath.lastIndexOf("/");
  const baseDir = slashIdx >= 0 ? entryPath.slice(0, slashIdx) : "";
  let inlinedAny = false;

  // 1. Inline relative <link rel="stylesheet" href="...">
  let compiled = rawHtml.replace(/<link\b[^>]*>/gi, (tag) => {
    const hrefMatch = /\bhref=["']([^"']+)["']/i.exec(tag);
    if (!hrefMatch) return tag;
    const isStylesheet =
      /\brel=["']stylesheet["']/i.test(tag) || hrefMatch[1]!.endsWith(".css");
    if (!isStylesheet) return tag;
    const asset = lookupWorkspaceAsset(hrefMatch[1]!, bundle.files, baseDir);
    if (!asset) return tag;
    inlinedAny = true;
    const cssDirIdx = asset.path.lastIndexOf("/");
    const cssBaseDir = cssDirIdx >= 0 ? asset.path.slice(0, cssDirIdx) : baseDir;
    const resolvedCss = asset.content.replace(
      /url\(\s*['"]?([^'")]+)['"]?\s*\)/gi,
      (fullUrl, refPath) => {
        const imgAsset = lookupWorkspaceAsset(String(refPath), bundle.files, cssBaseDir);
        if (!imgAsset) return fullUrl;
        if (imgAsset.content.startsWith("data:")) return `url("${imgAsset.content}")`;
        if (imgAsset.path.endsWith(".svg")) {
          return `url("data:image/svg+xml;utf8,${encodeURIComponent(imgAsset.content)}")`;
        }
        return fullUrl;
      },
    );
    return `<style data-veles-path="${asset.path}">\n${resolvedCss}\n</style>`;
  });

  // 2. Inline relative <script src="..."></script>
  compiled = compiled.replace(
    /<script\b([^>]*)\bsrc=["']([^"']+)["']([^>]*)>\s*<\/script>/gi,
    (full, pre, src, post) => {
      const asset = lookupWorkspaceAsset(String(src), bundle.files, baseDir);
      if (!asset) return full;
      inlinedAny = true;
      const rawAttrs = `${String(pre)} ${String(post)}`.trim();
      const hasDefer = /\bdefer\b/i.test(rawAttrs);
      const resolvedJs = inlineScriptImagePaths(asset.content, bundle.files, baseDir);
      const body = hasDefer
        ? `if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", function() {\n${resolvedJs}\n}); } else {\n${resolvedJs}\n}`
        : resolvedJs;
      const extraAttrs = rawAttrs ? ` ${rawAttrs}` : "";
      return `<script data-veles-path="${asset.path}"${extraAttrs}>\n${body}\n</script>`;
    },
  );

  // 3. Inline relative <img src="..."> when target is a workspace SVG or data URL
  compiled = compiled.replace(
    /<img\b([^>]*)\bsrc=["']([^"']+)["']([^>]*)>/gi,
    (full, pre, src, post) => {
      const asset = lookupWorkspaceAsset(String(src), bundle.files, baseDir);
      if (!asset) return full;
      const dataUrl = toImageDataUrl(asset.path, asset.content);
      if (!dataUrl) return full;
      inlinedAny = true;
      return `<img${String(pre)}src="${dataUrl}" data-veles-path="${asset.path}"${String(post)}>`;
    },
  );

  const usesVelesBridge =
    /\b(?:window\.)?veles\.(?:sensors|kv|log|files)\b|\b(?:localStorage|sessionStorage|fetch)\b/.test(
      compiled,
    ) || Object.keys(initialKv ?? {}).length > 0;
  if (inlinedAny || usesVelesBridge) {
    const bridge = buildErrorBridgeScript(
      generation,
      bundle.workspaceSeq ?? 0,
      bundle,
      initialKv,
      baseDir,
    );
    if (/<head\b[^>]*>/i.test(compiled)) {
      compiled = compiled.replace(/<head\b[^>]*>/i, (m) => `${m}\n${bridge}`);
    } else {
      compiled = `${bridge}\n${compiled}`;
    }
  }

  return compiled;
}

export interface PreviewRuntimeDiagnostic {
  readonly generation: number;
  readonly workspaceSeq?: number;
  readonly type: "preview_ready" | "preview_error" | "preview_log";
  readonly message: string;
}

export function assertIsolatedSandboxAttribute(sandboxAttr: string): void {
  const tokens = sandboxAttr
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0);
  if (!tokens.includes("allow-scripts")) {
    throw new Error("ERR_SANDBOX_MISSING_ALLOW_SCRIPTS");
  }
  for (const token of tokens) {
    if (FORBIDDEN_SANDBOX_TOKENS.has(token.toLowerCase())) {
      throw new Error(
        `ERR_SANDBOX_ORIGIN_VIOLATION: Forbidden sandbox token '${token}' violates SEC-INV-01`,
      );
    }
  }
}

function isMatchingDiagnosticSeq(
  rawSeq: unknown,
  expectedWorkspaceSeq?: number,
): boolean {
  if (expectedWorkspaceSeq === undefined || typeof rawSeq !== "number") {
    return true;
  }
  return rawSeq === expectedWorkspaceSeq;
}

function validatePreviewDiagnosticPayload(
  raw: unknown,
  expectedGeneration: number,
  expectedWorkspaceSeq?: number,
): PreviewRuntimeDiagnostic | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (
    obj.channel !== "veles:preview" ||
    obj.generation !== expectedGeneration ||
    !isMatchingDiagnosticSeq(obj.workspaceSeq, expectedWorkspaceSeq)
  ) {
    return null;
  }
  if (
    obj.type !== "preview_ready" &&
    obj.type !== "preview_error" &&
    obj.type !== "preview_log"
  ) {
    return null;
  }
  if (typeof obj.message !== "string") return null;
  return {
    generation: expectedGeneration,
    workspaceSeq:
      typeof obj.workspaceSeq === "number"
        ? obj.workspaceSeq
        : expectedWorkspaceSeq,
    type: obj.type,
    message: obj.message.slice(0, 2000),
  };
}

/**
 * Manages the lifecycle, generation fencing, opt-in sensor forwarding (`window.veles.sensors`),
 * and teardown of an isolated preview iframe.
 * Follows `.context/references/typescript-browser-apps/browser-runtime.md#resource-lifetimes`.
 */
export class PreviewSandboxHost {
  private readonly env: SandboxEnvironmentLike;
  private readonly onDiagnostic: (diag: PreviewRuntimeDiagnostic) => void;
  private readonly kvStoresByWorkspace = new Map<string, Map<string, unknown>>();
  private activeWorkspaceId = "default";
  private iframe: SandboxIframeLike | null = null;
  private removeListener: (() => void) | null = null;
  private sensorsEnabled = false;
  private generation = 0;
  private disposed = false;

  constructor(
    env: SandboxEnvironmentLike,
    onDiagnostic: (diag: PreviewRuntimeDiagnostic) => void = () => {},
  ) {
    this.env = env;
    this.onDiagnostic = onDiagnostic;
  }

  public get currentGeneration(): number {
    return this.generation;
  }

  public get isDisposed(): boolean {
    return this.disposed;
  }

  public get isSensorsEnabled(): boolean {
    return this.sensorsEnabled;
  }

  private getWorkspaceKvStore(
    workspaceId = this.activeWorkspaceId,
  ): Map<string, unknown> {
    const cleanId = workspaceId.trim() || "default";
    let store = this.kvStoresByWorkspace.get(cleanId);
    if (!store) {
      store = new Map<string, unknown>();
      this.kvStoresByWorkspace.set(cleanId, store);
    }
    return store;
  }

  public getKvSnapshot(
    workspaceId = this.activeWorkspaceId,
  ): Readonly<Record<string, unknown>> {
    return Object.fromEntries(this.getWorkspaceKvStore(workspaceId));
  }

  public setSensorsEnabled(enabled: boolean): void {
    this.sensorsEnabled = Boolean(enabled) && !this.disposed;
  }

  public pushSensorSample(sample: PreviewSensorSample): void {
    if (this.disposed || !this.sensorsEnabled || !this.iframe?.contentWindow) {
      return;
    }
    this.iframe.contentWindow.postMessage(
      {
        channel: "veles:host-sensor",
        generation: this.generation,
        sample,
      },
      "*",
    );
  }

  private recordKvUpdate(
    raw: unknown,
    expectedGen: number,
    workspaceKv: Map<string, unknown>,
  ): void {
    if (typeof raw !== "object" || raw === null) return;
    const obj = raw as Record<string, unknown>;
    if (
      obj.channel === "veles:preview" &&
      obj.generation === expectedGen &&
      obj.type === "kv_set" &&
      typeof obj.key === "string"
    ) {
      if (obj.removed === true) {
        workspaceKv.delete(obj.key);
      } else {
        workspaceKv.set(obj.key, obj.value);
      }
    }
  }

  public mountPreview(bundle: PreviewArtifactBundle): number {
    if (this.disposed) {
      throw new Error("ERR_PREVIEW_DISPOSED: Cannot mount on a disposed PreviewSandboxHost");
    }
    this.teardownCurrentFrame();
    this.generation += 1;
    const activeGen = this.generation;
    const workspaceKey = bundle.workspaceId?.trim() || "default";
    this.activeWorkspaceId = workspaceKey;
    const workspaceKv = this.getWorkspaceKvStore(workspaceKey);

    const frame = this.env.createIframe();
    frame.sandbox = REQUIRED_SANDBOX_FLAGS;
    assertIsolatedSandboxAttribute(frame.sandbox);

    // Register message listener guarded by frame identity, null origin, and generation
    this.removeListener = this.env.addMessageListener((ev) => {
      if (this.disposed || this.iframe !== frame) return;
      if (ev.source !== frame.contentWindow) return;
      // Sandboxed iframes without allow-same-origin have a serialized opaque origin ("null")
      if (ev.origin !== "null" && ev.origin !== "") {
        return;
      }
      this.recordKvUpdate(ev.data, activeGen, workspaceKv);
      const diag = validatePreviewDiagnosticPayload(
        ev.data,
        activeGen,
        bundle.workspaceSeq,
      );
      if (diag !== null) {
        this.onDiagnostic(diag);
      }
    });

    const sendHostInit = () => {
      if (this.disposed || this.iframe !== frame) return;
      frame.contentWindow?.postMessage(
        {
          channel: "veles:host-init",
          generation: activeGen,
          workspaceSeq: bundle.workspaceSeq ?? 0,
          files: { ...bundle.files },
          kv: Object.fromEntries(workspaceKv),
        },
        "*",
      );
    };

    frame.onLoad = sendHostInit;
    frame.srcdoc = compileMultiFileSrcdoc(
      bundle,
      activeGen,
      Object.fromEntries(workspaceKv),
    );
    this.iframe = frame;

    // Push virtual file map immediately (for sync test doubles) and on iframe load (real browser srcdoc)
    sendHostInit();

    return activeGen;
  }

  private teardownCurrentFrame(): void {
    if (this.removeListener !== null) {
      this.removeListener();
      this.removeListener = null;
    }
    if (this.iframe !== null) {
      this.iframe.srcdoc = "";
      this.iframe.remove();
      this.iframe = null;
    }
  }

  public dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.generation += 1;
    this.sensorsEnabled = false;
    this.teardownCurrentFrame();
  }
}
