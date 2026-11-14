/**
 * @module SYS-GATEWAY/server
 * Ephemeral Execution Gateway enforcing !INVARIANT [SEC-INV-02] and [PROTO-INV-01..03].
 * - Owns zero durable user state; generates a fresh `gatewayInstanceId` per boot.
 * - Independently enforces authentication, `capabilityScope`, `timeoutMs`, and `maxOutputBytes`.
 * - Executes sandboxed tools in ephemeral temp directories wiped immediately on completion/expiry.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { spawn, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve as resolvePath } from "node:path";
import {
  PROTOCOL_VERSION,
  type CapabilityScope,
  type ExecutionStreamEvent,
  type GatewayInstanceId,
  type OperationId,
  type ProtocolCommandRequest,
  type ResourceLimits,
  type ToolExecutePayload,
} from "../../src/protocol/types.ts";
import { computeSha256Hex } from "../../src/kernel/blobs.ts";
import { validateCommandRequest } from "../../src/protocol/validation.ts";

export const MAX_OPERATION_TTL_MS = 300_000 as const;

const SECRET_REDACT_RE =
  /(sk-[A-Za-z0-9_\-]{20,}|gsk_[A-Za-z0-9_\-]{20,}|apikey_[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9_\-\.~+/]+=*)/gi;

export function redactGatewayOutput(text: string): string {
  return text.replace(SECRET_REDACT_RE, "<REDACTED>");
}

export interface GatewayToolOutcome {
  readonly exitCode: number;
  readonly outputText: string;
  readonly outputSha256: string;
  readonly fileUpdates?: Readonly<Record<string, string>>;
}

export type GatewayToolRunner = (
  sandboxDir: string,
  payload: ToolExecutePayload,
  scope: CapabilityScope,
  limits: ResourceLimits,
  signal: AbortSignal,
) => Promise<{
  readonly exitCode: number;
  readonly outputText: string;
  readonly modifiedFiles?: Readonly<Record<string, string>>;
}>;

interface EphemeralOperationEntry {
  readonly operationId: OperationId;
  readonly runId: string;
  stage: "ACCEPTED" | "STARTED" | "COMPLETED";
  readonly events: ExecutionStreamEvent[];
  outcome?: GatewayToolOutcome;
  errorCode?: string;
  readonly abortController: AbortController;
  readonly expiresAtMs: number;
}

export interface EphemeralGatewayOptions {
  readonly authToken?: string;
  readonly maxTtlMs?: number;
  readonly registeredTools?: readonly string[];
  readonly toolRunner?: GatewayToolRunner;
  readonly allowUnrestrictedExec?: boolean;
}

function isCleanRelativePath(relPath: string): boolean {
  if (!relPath || relPath.startsWith("/") || relPath.includes("\\")) {
    return false;
  }
  return !relPath.split("/").some((p) => p === ".." || p === "." || p.length === 0);
}

function assertConfinementInSandbox(
  sandboxDir: string,
  relPath: string,
  writablePaths: readonly string[],
): string {
  if (!isCleanRelativePath(relPath)) {
    throw new Error(`ERR_GATEWAY_PATH_TRAVERSAL: ${relPath}`);
  }
  const fullTarget = resolvePath(sandboxDir, relPath);
  const rootWithSep = resolvePath(sandboxDir) + (process.platform === "win32" ? "\\" : "/");
  if (!fullTarget.startsWith(rootWithSep)) {
    throw new Error(`ERR_GATEWAY_SANDBOX_ESCAPE: ${relPath}`);
  }
  const allowed = writablePaths.some(
    (w) => relPath === w || relPath.startsWith(w.replace(/\/+$/, "") + "/"),
  );
  if (!allowed) {
    throw new Error(`ERR_GATEWAY_WRITABLE_SCOPE_DENIED: ${relPath}`);
  }
  return fullTarget;
}

function assertSafeExecArgv(sandboxDir: string, argv: readonly string[]): void {
  const rootWithSep =
    resolvePath(sandboxDir) + (process.platform === "win32" ? "\\" : "/");
  for (const arg of argv.slice(1)) {
    if (/^--(?:allow-|permission|experimental-permission|insecure-)/i.test(arg)) {
      throw new Error(`ERR_GATEWAY_SANDBOX_ESCAPE: Forbidden runtime flag ${arg}`);
    }
    if (/(^|[\\/])\.\.([\\/]|$)/.test(arg)) {
      throw new Error(`ERR_GATEWAY_PATH_TRAVERSAL: ${arg}`);
    }
    const isAbsPosix = arg.startsWith("/");
    const isAbsWin = /^[a-zA-Z]:[\\/]/.test(arg);
    if ((isAbsPosix || isAbsWin) && !resolvePath(arg).startsWith(rootWithSep)) {
      throw new Error(`ERR_GATEWAY_SANDBOX_ESCAPE: ${arg}`);
    }
  }
}

function isNodeExecutable(bin: string): boolean {
  if (bin === process.execPath) return true;
  const base = bin.split(/[\\/]/).pop() ?? "";
  return /^(node|nodejs)(\.exe)?$/i.test(base);
}

const SAFE_NODE_SWITCHES: ReadonlySet<string> = new Set([
  "--experimental-strip-types", "--experimental-transform-types", "--no-warnings",
]);
const NODE_CODE_OPTIONS: ReadonlySet<string> = new Set(["-e", "--eval", "-p", "--print"]);

function isSafeNodeRuntimeOption(arg: string): boolean {
  return SAFE_NODE_SWITCHES.has(arg) ||
    /^--(?:eval|print)=/.test(arg) ||
    /^--input-type=(?:module|commonjs)$/.test(arg);
}

function assertConfinedNodeOptions(argv: readonly string[]): void {
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i]!;
    // Node stops parsing runtime options at the entrypoint or explicit separator.
    if (arg === "--" || !arg.startsWith("-")) return;
    if (NODE_CODE_OPTIONS.has(arg)) {
      i += 1;
      continue;
    }
    if (arg === "--input-type" && /^(module|commonjs)$/.test(argv[i + 1] ?? "")) {
      i += 1;
      continue;
    }
    if (!isSafeNodeRuntimeOption(arg)) {
      throw new Error(
        `ERR_GATEWAY_RUNTIME_FLAG_DENIED: '${arg}' requires explicit allowUnrestrictedExec mode`,
      );
    }
  }
}

let cachedSupportsAllowNet: boolean | null = null;

function supportsNodeAllowNetFlag(): boolean {
  if (cachedSupportsAllowNet !== null) {
    return cachedSupportsAllowNet;
  }
  const probe = spawnSync(
    process.execPath,
    ["--permission", "--allow-net", "-e", ""],
    { timeout: 3000 },
  );
  cachedSupportsAllowNet = probe.status === 0;
  return cachedSupportsAllowNet;
}

const NETWORK_DENY_PRELOAD_DATA_URL = `data:text/javascript,${encodeURIComponent(
  [
    'import net from "node:net";',
    'import tls from "node:tls";',
    'import dgram from "node:dgram";',
    'import http from "node:http";',
    'import https from "node:https";',
    'const deny = () => { const err = new Error("Access to this API has been restricted"); err.code = "ERR_ACCESS_DENIED"; throw err; };',
    "net.Socket.prototype.connect = deny;",
    "net.connect = deny; net.createConnection = deny;",
    "tls.connect = deny;",
    "dgram.createSocket = deny;",
    "http.request = deny; http.get = deny;",
    "https.request = deny; https.get = deny;",
    'Object.defineProperty(globalThis, "fetch", { value: async () => deny(), configurable: false, writable: false });',
    'Object.defineProperty(globalThis, "WebSocket", { value: function() { deny(); }, configurable: false, writable: false });',
  ].join("\n"),
)}`;

async function buildConfinedExecArgv(
  sandboxDir: string,
  cleanArgv: readonly string[],
  scope: CapabilityScope,
  allowUnrestrictedExec: boolean,
): Promise<string[]> {
  assertSafeExecArgv(sandboxDir, cleanArgv);
  if (allowUnrestrictedExec) {
    return [...cleanArgv];
  }
  if (!isNodeExecutable(cleanArgv[0]!)) {
    throw new Error(
      `ERR_GATEWAY_SANDBOX_ESCAPE: Binary '${cleanArgv[0]}' requires explicit allowUnrestrictedExec mode`,
    );
  }
  assertConfinedNodeOptions(cleanArgv);
  const absSandbox = resolvePath(sandboxDir);
  const permFlags: string[] = [
    "--permission",
    `--allow-fs-read=${absSandbox}`,
    `--allow-fs-read=${absSandbox}/*`,
    `--allow-fs-read=${absSandbox}\\*`,
  ];
  if (scope.allowNetwork) {
    if (supportsNodeAllowNetFlag()) {
      permFlags.push("--allow-net");
    }
  } else if (!supportsNodeAllowNetFlag()) {
    permFlags.push("--import", NETWORK_DENY_PRELOAD_DATA_URL);
  }
  for (const relWritable of scope.writablePaths) {
    if (!isCleanRelativePath(relWritable)) continue;
    const absWritable = resolvePath(sandboxDir, relWritable);
    await mkdir(absWritable, { recursive: true });
    permFlags.push(
      `--allow-fs-write=${absWritable}`,
      `--allow-fs-write=${absWritable}/*`,
      `--allow-fs-write=${absWritable}\\*`,
    );
  }
  return [process.execPath, ...permFlags, ...cleanArgv.slice(1)];
}

export function createSandboxedToolRunner(
  opts: { readonly allowUnrestrictedExec?: boolean } = {},
): GatewayToolRunner {
  const allowUnrestricted = Boolean(opts.allowUnrestrictedExec);
  return async (sandboxDir, payload, scope, limits, signal) => {
    if (payload.toolName === "sandbox.WriteFile") {
      const relPath = String(payload.args.path ?? "");
      const content = String(payload.args.content ?? "");
      const target = assertConfinementInSandbox(
        sandboxDir,
        relPath,
        scope.writablePaths,
      );
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, content, "utf-8");
      const verified = await readFile(target, "utf-8");
      return {
        exitCode: 0,
        outputText: `Sandboxed write ok: ${relPath}`,
        modifiedFiles: { [relPath]: verified },
      };
    }

    if (payload.toolName === "sandbox.exec") {
      const cmdArgv = payload.args.argv;
      if (
        !Array.isArray(cmdArgv) ||
        cmdArgv.length === 0 ||
        typeof cmdArgv[0] !== "string"
      ) {
        throw new Error(
          "ERR_GATEWAY_INVALID_ARGV: sandbox.exec requires non-empty argv[]",
        );
      }
      const confinedArgv = await buildConfinedExecArgv(
        sandboxDir,
        cmdArgv.map(String),
        scope,
        allowUnrestricted,
      );
      return runBoundedSubprocess(confinedArgv, sandboxDir, limits, signal);
    }

    throw new Error(`ERR_GATEWAY_UNSUPPORTED_TOOL: ${payload.toolName}`);
  };
}

/**
 * Default sandboxed tool runner executing inside a temporary directory that is
 * wiped immediately upon completion, cancellation, or timeout.
 */
export const defaultSandboxedToolRunner: GatewayToolRunner =
  createSandboxedToolRunner({ allowUnrestrictedExec: false });

function runBoundedSubprocess(
  argv: string[],
  cwd: string,
  limits: ResourceLimits,
  signal: AbortSignal,
): Promise<{ exitCode: number; outputText: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(argv[0]!, argv.slice(1), {
      cwd,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
      env: { PATH: process.env.PATH ?? "", HOME: cwd, TMPDIR: cwd },
    });

    let collectedBytes = 0;
    const chunks: Buffer[] = [];
    let settled = false;

    const finish = (err: Error | null, exitCode = 1) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      if (err) {
        child.kill("SIGKILL");
        reject(err);
        return;
      }
      const rawText = Buffer.concat(chunks).toString("utf-8");
      if (rawText.includes("ERR_ACCESS_DENIED")) {
        reject(
          new Error(
            "ERR_GATEWAY_SANDBOX_ESCAPE: Subprocess attempted unauthorized access outside sandbox",
          ),
        );
        return;
      }
      resolve({ exitCode, outputText: redactGatewayOutput(rawText) });
    };

    const onData = (chunk: Buffer) => {
      collectedBytes += chunk.byteLength;
      if (collectedBytes > limits.maxOutputBytes) {
        finish(new Error("ERR_GATEWAY_OUTPUT_LIMIT_EXCEEDED"));
        return;
      }
      chunks.push(chunk);
    };

    const onAbort = () => finish(new Error("ERR_GATEWAY_OPERATION_CANCELLED"));
    signal.addEventListener("abort", onAbort, { once: true });

    const timer = setTimeout(() => {
      finish(new Error("ERR_GATEWAY_TIMEOUT"));
    }, limits.timeoutMs);

    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("error", (err) => finish(err));
    child.on("close", (code) => finish(null, code ?? 0));
  });
}

export class EphemeralExecutionGateway {
  public readonly gatewayInstanceId: GatewayInstanceId = `gw-${randomUUID()}`;
  private readonly authToken: string | null;
  private readonly maxTtlMs: number;
  private readonly registeredTools: ReadonlySet<string>;
  private readonly toolRunner: GatewayToolRunner;
  private readonly operations = new Map<OperationId, EphemeralOperationEntry>();
  private readonly activeSandboxes = new Set<string>();

  constructor(options: EphemeralGatewayOptions = {}) {
    this.authToken = options.authToken ?? null;
    this.maxTtlMs = Math.min(options.maxTtlMs ?? MAX_OPERATION_TTL_MS, MAX_OPERATION_TTL_MS);
    this.registeredTools = new Set(
      options.registeredTools ?? ["sandbox.exec", "sandbox.WriteFile", "inference.stream"],
    );
    this.toolRunner =
      options.toolRunner ??
      createSandboxedToolRunner({
        allowUnrestrictedExec: options.allowUnrestrictedExec,
      });
  }

  public get activeSandboxCount(): number {
    return this.activeSandboxes.size;
  }

  public get retainedOperationCount(): number {
    this.reapExpiredOperations();
    return this.operations.size;
  }

  public reapExpiredOperations(nowMs = Date.now()): number {
    let reaped = 0;
    for (const [opId, entry] of this.operations.entries()) {
      if (nowMs >= entry.expiresAtMs) {
        entry.abortController.abort();
        this.operations.delete(opId);
        reaped += 1;
      }
    }
    return reaped;
  }

  public checkAuthorized(authHeader: string | undefined): boolean {
    if (!this.authToken) return true;
    return authHeader === `Bearer ${this.authToken}`;
  }

  public listCapabilities() {
    const supported = Array.from(this.registeredTools);
    return {
      protocolVersion: PROTOCOL_VERSION,
      gatewayInstanceId: this.gatewayInstanceId,
      supportedTools: supported,
      toolDefinitions: supported.map((name) => ({
        name,
        description: `Ephemeral Gateway tool '${name}'`,
        effectClass: name.startsWith("read.") ? "pure_read" : "idempotent_write",
        requiresApproval: false,
      })),
      maxTtlMs: this.maxTtlMs,
    };
  }

  public getOperationStatus(operationId: OperationId) {
    this.reapExpiredOperations();
    const entry = this.operations.get(operationId);
    if (!entry) {
      return {
        found: false,
        gatewayInstanceId: this.gatewayInstanceId,
      };
    }
    return {
      found: true,
      gatewayInstanceId: this.gatewayInstanceId,
      operationId: entry.operationId,
      stage: entry.stage,
      completed: entry.stage === "COMPLETED",
      outcome: entry.outcome,
      errorCode: entry.errorCode,
      events: [...entry.events],
    };
  }

  public cancelOperation(operationId: OperationId): boolean {
    const entry = this.operations.get(operationId);
    if (!entry) return false;
    entry.abortController.abort();
    this.operations.delete(operationId);
    return true;
  }

  /**
   * Executes a validated `tool.execute` command inside an ephemeral temp directory
   * that is unconditionally deleted in `finally`, streaming Stages 1 -> 2 -> 3.
   */
  public async *executeToolStream(
    rawRequest: unknown,
  ): AsyncIterable<{
    readonly event: ExecutionStreamEvent;
    readonly outcome?: GatewayToolOutcome;
  }> {
    this.reapExpiredOperations();
    const validated = validateCommandRequest(rawRequest);
    if (!validated.ok || !validated.value) {
      throw new Error(validated.error ?? "ERR_INVALID_COMMAND");
    }
    const req = validated.value as ProtocolCommandRequest<ToolExecutePayload>;
    const payload = req.payload;
    if (
      !payload ||
      typeof payload.toolName !== "string" ||
      !req.capabilityScope.allowedTools.includes(payload.toolName) ||
      !this.registeredTools.has(payload.toolName)
    ) {
      throw new Error(
        `ERR_GATEWAY_TOOL_UNAUTHORIZED: Tool '${String(payload?.toolName)}' is not permitted by capabilityScope`,
      );
    }

    const abortController = new AbortController();
    const entry: EphemeralOperationEntry = {
      operationId: req.operationId,
      runId: req.runId,
      stage: "ACCEPTED",
      events: [],
      abortController,
      expiresAtMs: Date.now() + this.maxTtlMs,
    };
    this.operations.set(req.operationId, entry);

    const acceptedEvent = this.recordStageEvent(entry, 1, "ACCEPTED");
    yield { event: acceptedEvent };

    const startedEvent = this.recordStageEvent(entry, 2, "STARTED");
    yield { event: startedEvent };

    const sandboxDir = await mkdtemp(join(tmpdir(), "veles-gw-sandbox-"));
    this.activeSandboxes.add(sandboxDir);

    try {
      const execResult = await this.toolRunner(
        sandboxDir,
        payload,
        req.capabilityScope,
        req.limits,
        abortController.signal,
      );
      const cleanOutput = redactGatewayOutput(execResult.outputText);
      const outputSha256 = await computeSha256Hex(
        new TextEncoder().encode(cleanOutput),
      );
      const outcome: GatewayToolOutcome = {
        exitCode: execResult.exitCode,
        outputText: cleanOutput,
        outputSha256,
        fileUpdates: execResult.modifiedFiles,
      };
      entry.outcome = outcome;
      const completedEvent = this.recordStageEvent(entry, 3, "COMPLETED", {
        exitCode: outcome.exitCode,
        outputSha256: outcome.outputSha256,
        deltaText: outcome.outputText,
      });
      yield { event: completedEvent, outcome };
    } catch (err) {
      const errorCode = err instanceof Error ? err.message : "ERR_GATEWAY_EXEC_FAILED";
      entry.errorCode = errorCode;
      const failedEvent = this.recordStageEvent(entry, 3, "COMPLETED", {
        exitCode: 1,
        errorCode,
      });
      yield { event: failedEvent };
      throw err;
    } finally {
      // PROTO-INV-03 & Scenario 10: Always wipe sandbox directory immediately
      await rm(sandboxDir, { recursive: true, force: true }).catch(() => {});
      this.activeSandboxes.delete(sandboxDir);
    }
  }

  private recordStageEvent(
    entry: EphemeralOperationEntry,
    seq: number,
    stage: "ACCEPTED" | "STARTED" | "COMPLETED",
    extra: Partial<ExecutionStreamEvent> = {},
  ): ExecutionStreamEvent {
    entry.stage = stage;
    const ev: ExecutionStreamEvent = {
      protocolVersion: PROTOCOL_VERSION,
      operationId: entry.operationId,
      gatewayInstanceId: this.gatewayInstanceId,
      seq,
      stage,
      timestampMs: Date.now(),
      ...extra,
    };
    entry.events.push(ev);
    return ev;
  }

  public createHttpServer(): Server {
    return createServer(async (req, res) => {
      try {
        await this.handleHttpRequest(req, res);
      } catch (err) {
        if (!res.headersSent) {
          res.writeHead(400, { "Content-Type": "application/json" });
        }
        res.end(
          JSON.stringify({
            error: redactGatewayOutput(err instanceof Error ? err.message : String(err)),
          }),
        );
      }
    });
  }

  private async handleHttpRequest(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    if (!this.checkAuthorized(req.headers.authorization)) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "ERR_GATEWAY_UNAUTHORIZED" }));
      return;
    }

    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/v1/capabilities") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(this.listCapabilities()));
      return;
    }

    if (req.method === "GET" && url.pathname.startsWith("/v1/operations/")) {
      const opId = url.pathname.slice("/v1/operations/".length);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(this.getOperationStatus(opId)));
      return;
    }

    if (req.method === "POST" && url.pathname === "/v1/tools/execute") {
      const bodyText = await readRequestBody(req);
      const bodyJson: unknown = JSON.parse(bodyText);
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      try {
        for await (const item of this.executeToolStream(bodyJson)) {
          res.write(`data: ${JSON.stringify(item)}\n\n`);
        }
        res.write("data: [DONE]\n\n");
      } finally {
        res.end();
      }
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "ERR_NOT_FOUND" }));
  }
}

function readRequestBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}
