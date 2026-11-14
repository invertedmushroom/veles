/**
 * @module SYS-KERNEL/agent-session
 * Agent Session Isolation & Lifecycle Manager (`AgentSession` + `AgentManager`).
 * Ensures multiple independent agents can interleave journal events on shared browser storage,
 * enforce per-session `allowedTools` & `ProviderModelConfig`, reconstruct isolated session
 * contexts via `ContextBuilder`, and recover session state across worker restarts.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import type {
  ChatMessage,
  ProviderModelConfig,
} from "../inference/openai-compatible.ts";
import type {
  RunId,
  RunJournalRecord,
  RunState,
  WorkspaceId,
} from "../protocol/types.ts";
import type { WorkspaceKernelStore } from "./journal.ts";
import { type AgentKernelRuntime, ContextBuilder } from "./runtime.ts";

export type AgentSessionStatus = "idle" | "running" | "completed" | "failed";

export interface AgentSession {
  readonly id: string;
  readonly title?: string;
  readonly parentSessionId?: string;
  readonly workspaceId: WorkspaceId;
  readonly model: ProviderModelConfig;
  readonly allowedTools: readonly string[];
  readonly delegatableTools?: readonly string[];
  readonly unrestrictedTools?: boolean;
  readonly status: AgentSessionStatus;
}

export interface AgentManagerContract {
  createSession(config: AgentSession): Promise<void>;
  run(sessionId: string, prompt: string): Promise<string>;
  cancel(sessionId: string): Promise<void>;
  getSession(sessionId: string): AgentSession | undefined;
}

function mapRunStateToSessionStatus(state: RunState): AgentSessionStatus {
  if (state === "completed") return "completed";
  if (state === "failed" || state === "cancelled" || state === "interrupted") {
    return "failed";
  }
  return "running";
}

function assertDeclaredToolAuthority(value: unknown): void {
  if (value !== undefined && typeof value !== "boolean") {
    throw new Error("ERR_INVALID_SESSION_POLICY: unrestrictedTools must be a boolean");
  }
}

export class AgentManager implements AgentManagerContract {
  private readonly store: WorkspaceKernelStore;
  private readonly runtime: AgentKernelRuntime;
  private readonly sessions = new Map<string, AgentSession>();
  private readonly activeRunBySession = new Map<string, RunId>();
  private readonly sessionMutexes = new Map<string, Promise<void>>();
  private runSeq = 0;

  constructor(store: WorkspaceKernelStore, runtime: AgentKernelRuntime) {
    this.store = store;
    this.runtime = runtime;
  }

  private enqueueSessionExclusive<T>(
    sessionId: string,
    task: () => Promise<T>,
  ): Promise<T> {
    const prev = this.sessionMutexes.get(sessionId) ?? Promise.resolve();
    const next = prev.then(task, task);
    this.sessionMutexes.set(
      sessionId,
      next.then(
        () => undefined,
        () => undefined,
      ),
    );
    return next;
  }

  public upsertSessionInMemory(config: AgentSession): AgentSession {
    assertDeclaredToolAuthority(config.unrestrictedTools);
    const cleanId = config.id.trim();
    if (!cleanId) {
      throw new Error("ERR_INVALID_SESSION_ID: Session ID must be non-empty");
    }
    const declaredAuthority =
      config.unrestrictedTools ?? this.sessions.get(cleanId)?.unrestrictedTools;
    const unrestrictedTools = Boolean(
      !config.parentSessionId &&
        (declaredAuthority ??
          (cleanId === "session-main" &&
            config.allowedTools.length === 0 &&
            config.delegatableTools === undefined)),
    );
    const normalized: AgentSession = {
      id: cleanId,
      title: config.title,
      parentSessionId: config.parentSessionId,
      workspaceId: config.workspaceId,
      model: { ...config.model },
      allowedTools: [...config.allowedTools],
      delegatableTools:
        config.delegatableTools !== undefined
          ? [...config.delegatableTools]
          : undefined,
      unrestrictedTools,
      status: config.status ?? "idle",
    };
    this.sessions.set(cleanId, normalized);
    return normalized;
  }

  public async createSession(config: AgentSession): Promise<void> {
    const normalized = this.upsertSessionInMemory(config);
    await this.store.recordSessionConfiguration(normalized.id, {
      title: normalized.title,
      parentSessionId: normalized.parentSessionId,
      workspaceId: normalized.workspaceId,
      providerId: normalized.model.providerId,
      baseUrl: normalized.model.baseUrl,
      model: normalized.model.model,
      allowedTools: [...normalized.allowedTools],
      ...(normalized.delegatableTools !== undefined
        ? { delegatableTools: [...normalized.delegatableTools] }
        : {}),
      unrestrictedTools: normalized.unrestrictedTools,
      status: normalized.status,
    });
  }

  public getSession(sessionId: string): AgentSession | undefined {
    const s = this.sessions.get(sessionId);
    if (!s) return undefined;
    return {
      ...s,
      model: { ...s.model },
      allowedTools: [...s.allowedTools],
      ...(s.delegatableTools !== undefined
        ? { delegatableTools: [...s.delegatableTools] }
        : {}),
    };
  }

  public listSessions(): AgentSession[] {
    return Array.from(this.sessions.values()).map((s) => ({
      ...s,
      model: { ...s.model },
      allowedTools: [...s.allowedTools],
      ...(s.delegatableTools !== undefined
        ? { delegatableTools: [...s.delegatableTools] }
        : {}),
    }));
  }

  public resolveSessionForRun(
    runId: RunId,
    journal?: readonly RunJournalRecord[],
  ): AgentSession | undefined {
    for (const [sessionId, activeRunId] of this.activeRunBySession.entries()) {
      if (activeRunId === runId) {
        return this.getSession(sessionId);
      }
    }
    if (!journal) return undefined;
    for (let i = journal.length - 1; i >= 0; i--) {
      const rec = journal[i]!;
      if (
        rec.runId === runId &&
        rec.type === "RunCreated" &&
        typeof rec.payload.sessionId === "string"
      ) {
        return this.getSession(rec.payload.sessionId);
      }
    }
    return undefined;
  }

  public async run(sessionId: string, prompt: string): Promise<string> {
    const res = await this.runWithId(sessionId, prompt);
    return res.assistantText;
  }

  public async runWithId(
    sessionId: string,
    prompt: string,
    explicitRunId?: RunId,
    signal?: AbortSignal,
  ): Promise<{ readonly runId: RunId; readonly state: RunState; readonly assistantText: string }> {
    if (!this.sessions.has(sessionId)) {
      throw new Error(`ERR_SESSION_NOT_FOUND: ${sessionId}`);
    }
    return this.enqueueSessionExclusive(sessionId, () =>
      this.executeSessionRun(sessionId, prompt, explicitRunId, signal),
    );
  }

  private async executeSessionRun(
    sessionId: string,
    prompt: string,
    explicitRunId?: RunId,
    signal?: AbortSignal,
  ): Promise<{ readonly runId: RunId; readonly state: RunState; readonly assistantText: string }> {
    signal?.throwIfAborted();
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`ERR_SESSION_NOT_FOUND: ${sessionId}`);
    }
    this.runSeq += 1;
    const runId = explicitRunId ?? `${sessionId}-run-${this.runSeq}-${Date.now()}`;
    this.activeRunBySession.set(sessionId, runId);
    this.sessions.set(sessionId, { ...session, status: "running" });

    try {
      const effectiveAllowedTools =
        session.unrestrictedTools === true && !session.parentSessionId
          ? undefined
          : session.allowedTools;
      const finalState = await this.runtime.startRun(
        runId,
        prompt,
        session.model,
        {
          sessionId: session.id,
          parentSessionId: session.parentSessionId,
          allowedTools: effectiveAllowedTools,
          delegatableTools: session.delegatableTools,
          unrestrictedTools: session.unrestrictedTools,
        },
        signal,
      );

      const nextStatus = mapRunStateToSessionStatus(finalState);
      this.sessions.set(sessionId, { ...session, status: nextStatus });

      const journal = await this.store.getJournal();
      const completedRec = [...journal]
        .reverse()
        .find((r) => r.runId === runId && r.type === "RunCompleted");
      const assistantText =
        typeof completedRec?.payload.assistantText === "string"
          ? completedRec.payload.assistantText
          : "";
      return { runId, state: finalState, assistantText };
    } finally {
      this.activeRunBySession.delete(sessionId);
    }
  }

  public async cancel(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) return;
    const activeRunId = this.activeRunBySession.get(sessionId);
    if (activeRunId) {
      await this.runtime.cancelActiveRun(activeRunId);
      this.activeRunBySession.delete(sessionId);
    }
    this.sessions.set(sessionId, { ...session, status: "failed" });
  }

  public async getSessionMessages(
    sessionId: string,
    opts: { readonly hotRunWindow?: number; readonly maxCharBudget?: number } = {},
  ): Promise<ChatMessage[]> {
    const journal = await this.store.getJournal();
    return ContextBuilder.buildMessages(journal, {
      sessionId,
      hotRunWindow: opts.hotRunWindow ?? 2,
      maxCharBudget: opts.maxCharBudget,
    });
  }

  /**
   * Reconstructs all persisted `AgentSession` states and relationships from the
   * committed IndexedDB journal after a Web Worker restart or workspace switch.
   */
  public async recoverSessionsFromJournal(): Promise<void> {
    this.sessions.clear();
    this.activeRunBySession.clear();
    const journal = await this.store.getJournal();
    const runToSession = new Map<RunId, string>();
    for (const rec of journal) {
      this.applyJournalRecordToSessions(rec, runToSession);
    }
  }

  private applyJournalRecordToSessions(
    rec: RunJournalRecord,
    runToSession: Map<RunId, string>,
  ): void {
    const p = rec.payload;
    if (this.isSessionInitRecord(rec) && typeof p.sessionId === "string") {
      const sessionId = p.sessionId;
      if (rec.type === "RunCreated") {
        runToSession.set(rec.runId, sessionId);
      }
      const existing = this.sessions.get(sessionId);
      this.sessions.set(
        sessionId,
        this.buildRecoveredSession(rec, sessionId, existing),
      );
      return;
    }

    const ownerSessionId = runToSession.get(rec.runId);
    if (!ownerSessionId) return;
    const existing = this.sessions.get(ownerSessionId);
    if (!existing) return;
    this.sessions.set(ownerSessionId, {
      ...existing,
      status: mapRunStateToSessionStatus(rec.committedState),
    });
  }

  private isSessionInitRecord(rec: RunJournalRecord): boolean {
    return (
      rec.type === "RunCreated" ||
      (rec.type === "CheckpointCommitted" && rec.payload.sessionConfigured === true)
    );
  }

  private buildRecoveredSession(
    rec: RunJournalRecord,
    sessionId: string,
    existing?: AgentSession,
  ): AgentSession {
    const p = rec.payload;
    const allowedTools = Array.isArray(p.allowedTools)
      ? p.allowedTools.filter((t): t is string => typeof t === "string")
      : (existing?.allowedTools ?? []);
    const delegatableTools = Array.isArray(p.delegatableTools)
      ? p.delegatableTools.filter((t): t is string => typeof t === "string")
      : existing?.delegatableTools !== undefined
        ? [...existing.delegatableTools]
        : undefined;
    const parentSessionId =
      typeof p.parentSessionId === "string"
        ? p.parentSessionId
        : existing?.parentSessionId;
    const unrestrictedTools =
      typeof p.unrestrictedTools === "boolean"
        ? Boolean(p.unrestrictedTools && !parentSessionId)
        : existing?.unrestrictedTools !== undefined
          ? Boolean(existing.unrestrictedTools && !parentSessionId)
          : Boolean(
              !parentSessionId &&
                sessionId === "session-main" &&
                allowedTools.length === 0 &&
                delegatableTools === undefined,
            );
    const status: AgentSessionStatus =
      rec.type === "CheckpointCommitted"
        ? "idle"
        : mapRunStateToSessionStatus(rec.committedState);
    return {
      id: sessionId,
      title: typeof p.title === "string" ? p.title : existing?.title,
      parentSessionId,
      workspaceId: rec.workspaceId,
      model: {
        providerId: String(p.providerId ?? "openai"),
        baseUrl: String(p.baseUrl ?? "https://api.openai.com/v1"),
        model: String(p.model ?? "gpt-4o-mini"),
      },
      allowedTools,
      ...(delegatableTools !== undefined ? { delegatableTools } : {}),
      unrestrictedTools,
      status,
    };
  }
}

