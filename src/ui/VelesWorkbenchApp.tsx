/**
 * @module SYS-UI/VelesWorkbenchApp
 * Chat workbench with floating workspace browsers and isolated preview.
 * Communicates with the Web Worker Kernel strictly via `VelesAppClient` (`MessagePort`)
 * and renders generated HTML/JS inside `PreviewSandboxHost` (`<iframe sandbox="allow-scripts">`).
 * Zero imports from `src/kernel/**`, `IndexedDB`, or `OPFS` (`!INVARIANT [Coupling_Cohesion_Law]`).
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { DECISION_MODE_PRESETS, PROVIDER_PRESETS } from "./service-presets.ts";
import type {
  DecisionServiceConfigSpec,
  DecisionServiceMode,
  ImageServiceMode,
  ProviderModelConfigSpec,
  RunJournalRecord,
  ServiceCapabilitiesConfigSpec,
  WebSearchServiceMode,
  WorkspaceStateSnapshot,
} from "../protocol/types.ts";
import {
  isActiveRun,
  type AppClientStateView,
  type PendingUserTurn,
  type VelesAppClient,
} from "./app-client.ts";
import {
  PreviewSandboxHost,
  pickDefaultWorkspaceFile,
  resolvePreviewEntryPath,
  type MessageEventLike,
  type PreviewRuntimeDiagnostic,
  type SandboxEnvironmentLike,
  type SandboxIframeLike,
} from "./preview-sandbox.ts";
import { WorkbenchLayout, type ExpandedPane } from "./WorkbenchLayout.tsx";
import { ChatTimeline } from "./ChatTimeline.tsx";
import { projectChatRuns } from "./chat-timeline.ts";
import { WorkspaceBrowser, type BrowserPanel } from "./WorkspaceBrowsers.tsx";
import {
  usePwaDeviceSession,
  type PwaDeviceSessionView,
} from "./pwa-device-session.ts";
import "./workbench.css";

function createDomSandboxEnvironment(
  getContainer: () => HTMLElement | null,
): SandboxEnvironmentLike {
  return {
    createIframe(): SandboxIframeLike {
      const iframe = document.createElement("iframe");
      const container = getContainer();
      if (container) {
        container.innerHTML = "";
        container.appendChild(iframe);
      }
      return {
        get sandbox() {
          return iframe.getAttribute("sandbox") ?? "";
        },
        set sandbox(val: string) {
          iframe.setAttribute("sandbox", val);
        },
        get srcdoc() {
          return iframe.srcdoc;
        },
        set srcdoc(val: string) {
          iframe.srcdoc = val;
        },
        get onLoad() {
          return null;
        },
        set onLoad(fn: (() => void) | null | undefined) {
          iframe.onload = fn ? () => fn() : null;
        },
        get contentWindow() {
          return iframe.contentWindow;
        },
        remove() {
          iframe.remove();
        },
      };
    },
    addMessageListener(handler: (ev: MessageEventLike) => void) {
      const listener = (ev: MessageEvent) => {
        handler({ source: ev.source, origin: ev.origin, data: ev.data });
      };
      window.addEventListener("message", listener);
      return () => window.removeEventListener("message", listener);
    },
  };
}

function ChatServiceCard({
  provider,
  isObserver,
  onChangeProvider,
}: {
  readonly provider: ProviderModelConfigSpec;
  readonly isObserver: boolean;
  readonly onChangeProvider: (next: ProviderModelConfigSpec) => void;
}) {
  return (
    <div className="service-card" aria-label="Chat Reasoning LLM Service">
      <div className="service-card-title">
        <span>🧠 1. Chat / Reasoning LLM</span>
        <span className="hash-badge">{provider.providerId}</span>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-chat-provider">Provider</label>
        <select id="svc-chat-provider" className="service-input" disabled={isObserver} value={provider.providerId}
          onChange={(event) => {
            const preset = PROVIDER_PRESETS.find((entry) => entry.providerId === event.target.value);
            if (preset) onChangeProvider({ ...preset });
          }}>
          {PROVIDER_PRESETS.map((preset) => <option key={preset.providerId} value={preset.providerId}>
            {preset.providerId === "veles-demo" ? "Demo (offline)" : preset.providerId}
          </option>)}
        </select>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-chat-base-url">
          Base URL
        </label>
        <input
          id="svc-chat-base-url"
          className="service-input"
          type="text"
          disabled={isObserver}
          value={provider.baseUrl}
          onChange={(ev) =>
            onChangeProvider({ ...provider, baseUrl: ev.target.value })
          }
        />
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-chat-model">
          Model ID
        </label>
        <input
          id="svc-chat-model"
          className="service-input"
          type="text"
          disabled={isObserver}
          value={provider.model}
          onChange={(ev) =>
            onChangeProvider({ ...provider, model: ev.target.value })
          }
        />
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-chat-key">
          API Key
        </label>
        <input
          id="svc-chat-key"
          className="service-input"
          type="password"
          disabled={isObserver}
          placeholder="sk-..."
          value={provider.apiKey ?? ""}
          onChange={(ev) =>
            onChangeProvider({ ...provider, apiKey: ev.target.value })
          }
        />
      </div>
    </div>
  );
}

function DecisionServiceCard({
  decision,
  isObserver,
  onChangeDecision,
}: {
  readonly decision: DecisionServiceConfigSpec;
  readonly isObserver: boolean;
  readonly onChangeDecision: (next: DecisionServiceConfigSpec) => void;
}) {
  const handleModeChange = (mode: DecisionServiceMode) => {
    const preset = DECISION_MODE_PRESETS[mode];
    const nextKey = preset.apiKey;
    onChangeDecision({
      ...decision,
      mode,
      endpointUrl: preset.endpointUrl,
      model: preset.model,
      apiKey: nextKey,
    });
  };

  return (
    <div className="service-card" aria-label="Decision Classifier API Service">
      <div className="service-card-title">
        <span>⚡ 2. System-1 Decision API (Router)</span>
        <span className="hash-badge">{decision.mode}</span>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-decision-mode">
          Engine Mode
        </label>
        <select
          id="svc-decision-mode"
          aria-label="Decision API Mode"
          className="service-input"
          disabled={isObserver}
          value={decision.mode}
          onChange={(ev) => handleModeChange(ev.target.value as DecisionServiceMode)}
        >
          {(Object.keys(DECISION_MODE_PRESETS) as DecisionServiceMode[]).map(
            (mode) => (
              <option key={mode} value={mode}>
                {DECISION_MODE_PRESETS[mode].label}
              </option>
            ),
          )}
        </select>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-decision-url">
          Endpoint URL
        </label>
        <input
          id="svc-decision-url"
          className="service-input"
          type="text"
          disabled={isObserver || decision.mode === "tier0-only"}
          value={decision.endpointUrl}
          onChange={(ev) =>
            onChangeDecision({ ...decision, endpointUrl: ev.target.value })
          }
        />
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-decision-model">
          Model / Policy
        </label>
        <input
          id="svc-decision-model"
          className="service-input"
          type="text"
          disabled={isObserver || decision.mode === "tier0-only"}
          value={decision.model}
          onChange={(ev) =>
            onChangeDecision({ ...decision, model: ev.target.value })
          }
        />
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-decision-key">
          Decision Key
        </label>
        <input
          id="svc-decision-key"
          className="service-input"
          type="password"
          disabled={isObserver || decision.mode === "tier0-only"}
          placeholder={
            decision.mode === "openai-decisions"
              ? "Uses Chat OpenAI Key if blank"
              : "TYPESAFE_API_KEY / JEV_API_KEY"
          }
          value={decision.apiKey ?? ""}
          onChange={(ev) =>
            onChangeDecision({ ...decision, apiKey: ev.target.value })
          }
        />
      </div>
    </div>
  );
}

function MediaAndSearchServiceCard({
  serviceConfig,
  isObserver,
  onChangeServiceConfig,
}: {
  readonly serviceConfig: ServiceCapabilitiesConfigSpec;
  readonly isObserver: boolean;
  readonly onChangeServiceConfig: (next: ServiceCapabilitiesConfigSpec) => void;
}) {
  return (
    <div className="service-card" aria-label="Image and Web Search Services">
      <div className="service-card-title">
        <span>🎨 3. Image Generation &amp; Web Search</span>
        <span className="hash-badge">{serviceConfig.imageMode}</span>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-image-mode">
          Image Engine
        </label>
        <select
          id="svc-image-mode"
          aria-label="Image Service Mode"
          className="service-input"
          disabled={isObserver}
          value={serviceConfig.imageMode}
          onChange={(ev) =>
            onChangeServiceConfig({
              ...serviceConfig,
              imageMode: ev.target.value as ImageServiceMode,
            })
          }
        >
          <option value="openai-auto">
            OpenAI Auto (Responses + DALL-E 3 + GPT-Image-1)
          </option>
          <option value="openai-dalle3">
            OpenAI DALL-E 3 + Vision Edit Synthesis
          </option>
          <option value="openai-gpt-svg">
            OpenAI GPT-4o Vector &lt;svg&gt; Illustration
          </option>
          <option value="local-deterministic-svg">
            Offline Deterministic SVG (0-cost)
          </option>
        </select>
      </div>
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-search-mode">
          Web Search
        </label>
        <select
          id="svc-search-mode"
          aria-label="Web Search Mode"
          className="service-input"
          disabled={isObserver}
          value={serviceConfig.webSearchMode}
          onChange={(ev) =>
            onChangeServiceConfig({
              ...serviceConfig,
              webSearchMode: ev.target.value as WebSearchServiceMode,
            })
          }
        >
          <option value="openai-responses">
            OpenAI web_search_preview (Live Web)
          </option>
          <option value="searxng">SearXNG (Self-Hosted JSON API)</option>
          <option value="offline-disabled">Offline / Disabled</option>
        </select>
      </div>
      {serviceConfig.webSearchMode === "searxng" && (
        <div className="service-field-row">
          <label className="service-label" htmlFor="svc-searxng-url">
            SearXNG URL
          </label>
          <input
            id="svc-searxng-url"
            className="service-input"
            type="text"
            disabled={isObserver}
            value={serviceConfig.searxngUrl ?? "http://127.0.0.1:8080/search"}
            onChange={(ev) =>
              onChangeServiceConfig({
                ...serviceConfig,
                searxngUrl: ev.target.value,
              })
            }
          />
        </div>
      )}
      <div className="service-field-row">
        <label className="service-label" htmlFor="svc-decision-timeout">
          Router Timeout
        </label>
        <input
          id="svc-decision-timeout"
          className="service-input"
          type="number"
          min={100}
          max={5000}
          step={100}
          disabled={isObserver}
          value={serviceConfig.decision.timeoutMs ?? 800}
          onChange={(ev) =>
            onChangeServiceConfig({
              ...serviceConfig,
              decision: {
                ...serviceConfig.decision,
                timeoutMs: Number(ev.target.value) || 800,
              },
            })
          }
        />
      </div>
    </div>
  );
}

function WorkspaceAndDeviceCard({
  snap,
  pwa,
}: {
  readonly snap: WorkspaceStateSnapshot | null;
  readonly pwa: PwaDeviceSessionView;
}) {
  const isLeader = snap?.leaseMode !== "ReadOnly_Observer";
  return (
    <div className="service-card" aria-label="Workspace status">
      <div className="service-card-title">
        <span>Workspace status · {snap?.title ?? "Initializing…"}</span>
        <span data-testid="pwa-display-mode" className="hash-badge">
          {pwa.displayMode}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        <span className={`telemetry-pill ${isLeader ? "leader" : "observer"}`}>
          {isLeader
            ? "ReadWrite_Leader"
            : "ReadOnly_Observer (Locked by other tab)"}
        </span>
        <span className="telemetry-pill">
          Head Seq: <strong>#{snap?.headSeq ?? 0}</strong> ·{" "}
          {pwa.persisted ? "Persisted" : "Best-Effort"}
        </span>
        <span className="telemetry-pill gateway">
          Gateway:{" "}
          <code>{(snap?.gatewayInstanceId ?? "browser-local-v1").slice(0, 18)}</code>
        </span>
        <span data-testid="pwa-network-status" className="telemetry-pill">
          {pwa.networkOnline ? "online" : "offline"}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
        <button
          type="button"
          data-testid="pwa-install"
          className="wb-btn"
          disabled={!pwa.canInstall}
          onClick={() => void pwa.triggerInstall()}
        >
          {pwa.installButtonLabel}
        </button>
        <button
          type="button"
          data-testid="wake-lock-toggle"
          className="wb-btn"
          disabled={!pwa.wakeLockSupported || pwa.wakeLockPending}
          aria-pressed={pwa.wakeLockActive}
          onClick={() => void pwa.toggleWakeLock()}
        >
          {pwa.wakeLockButtonLabel}
        </button>
        <button
          type="button"
          data-testid="app-fullscreen-toggle"
          className="wb-btn"
          aria-pressed={pwa.fullscreenActive}
          onClick={() => void pwa.toggleAppFullscreen()}
        >
          {pwa.fullscreenButtonLabel}
        </button>
        {pwa.updateAvailable && (
          <button
            type="button"
            data-testid="pwa-update"
            className="wb-btn"
            disabled={pwa.applyingUpdate}
            onClick={pwa.triggerUpdate}
          >
            {pwa.applyingUpdate ? "Applying update…" : "Apply update"}
          </button>
        )}
      </div>
      <div
        data-testid="pwa-status"
        style={{ fontSize: 11, color: "#90a8a0", marginTop: 4 }}
      >
        {pwa.pwaStatus}
      </div>
      <div
        data-testid="wake-lock-status"
        style={{ fontSize: 11, color: "#90a8a0" }}
      >
        {pwa.wakeLockStatus}
      </div>
    </div>
  );
}

function ServiceConfigPanel({
  provider,
  snap,
  pwa,
  serviceConfig,
  isObserver,
  onChangeProvider,
  onChangeServiceConfig,
  onClose,
}: {
  readonly provider: ProviderModelConfigSpec;
  readonly snap: WorkspaceStateSnapshot | null;
  readonly pwa: PwaDeviceSessionView;
  readonly serviceConfig: ServiceCapabilitiesConfigSpec;
  readonly isObserver: boolean;
  readonly onChangeProvider: (next: ProviderModelConfigSpec) => void;
  readonly onChangeServiceConfig: (next: ServiceCapabilitiesConfigSpec) => void;
  readonly onClose: () => void;
}) {
  const panel = useRef<HTMLElement>(null);
  const previousFocus = useRef(document.activeElement);
  useEffect(() => {
    panel.current?.querySelector("select")?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); onClose(); }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      if (previousFocus.current instanceof HTMLElement && previousFocus.current.isConnected) previousFocus.current.focus();
    };
  }, [onClose]);
  return (
    <section
      id="service-config-panel"
      ref={panel}
      className="service-config-drawer"
      aria-label="Veles Service Configuration Panel"
    >
      <div className="service-config-header">
        <div>
          <strong>⚙️ Veles Capability &amp; Service Configuration</strong>{" "}
          <span style={{ color: "#90a8a0", marginLeft: 8 }}>
            Configure Chat LLM, System-1 Decision API (OpenAI Decisions / TypeSafe Jev / Local Laya),
            Image Generation, and Live Web Search.
          </span>
        </div>
        <button type="button" className="wb-btn" onClick={onClose}>
          ✕ Close Services
        </button>
      </div>
      <div className="service-config-grid">
        <ChatServiceCard
          provider={provider}
          isObserver={isObserver}
          onChangeProvider={onChangeProvider}
        />
        <DecisionServiceCard
          decision={serviceConfig.decision}
          isObserver={isObserver}
          onChangeDecision={(decision) =>
            onChangeServiceConfig({ ...serviceConfig, decision })
          }
        />
        <WorkspaceAndDeviceCard snap={snap} pwa={pwa} />
        <MediaAndSearchServiceCard
          serviceConfig={serviceConfig}
          isObserver={isObserver}
          onChangeServiceConfig={onChangeServiceConfig}
        />
      </div>
    </section>
  );
}

function TopAuthorityBar({
  snap, newWorkspaceName, showServicesPanel, browserPanel,
  onToggleBrowser, onChangeNewWorkspaceName, onToggleServicesPanel,
  onSwitchWorkspace, onCreateWorkspace, onDeleteWorkspace,
}: {
  readonly snap: WorkspaceStateSnapshot | null;
  readonly newWorkspaceName: string;
  readonly showServicesPanel: boolean;
  readonly browserPanel: BrowserPanel | null;
  readonly onToggleBrowser: () => void;
  readonly onChangeNewWorkspaceName: (val: string) => void;
  readonly onToggleServicesPanel: () => void;
  readonly onSwitchWorkspace: (workspaceId: string, title?: string) => void;
  readonly onCreateWorkspace: () => void;
  readonly onDeleteWorkspace: () => void;
}) {
  const isLeader = snap?.leaseMode !== "ReadOnly_Observer";
  const workspaces = snap?.workspaces ?? [];
  const menu = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (menu.current && event.target instanceof Node && !menu.current.contains(event.target)) menu.current.open = false;
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && menu.current?.open) { menu.current.open = false; menu.current.querySelector("summary")?.focus(); }
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => { document.removeEventListener("pointerdown", closeOutside); document.removeEventListener("keydown", closeOnEscape); };
  }, []);

  return <header className="workspace-toolbar" aria-label="Workspace toolbar">
    <span className="workspace-brand">VELES</span>
    <div className="workspace-controls">
    <select aria-label="Switch Workspace" value={snap?.workspaceId ?? "ws-alpha"} disabled={!isLeader}
      onChange={(event) => {
        const found = workspaces.find((workspace) => workspace.workspaceId === event.target.value);
        onSwitchWorkspace(event.target.value, found?.title);
      }}>
      {workspaces.length === 0 && <option value={snap?.workspaceId ?? "ws-alpha"}>{snap?.title ?? "Alpha"}</option>}
      {workspaces.map((workspace) => <option key={workspace.workspaceId} value={workspace.workspaceId}>{workspace.title}</option>)}
    </select>
      <details ref={menu} className="workspace-create" onToggle={(event) => {
        if (event.currentTarget.open) event.currentTarget.querySelector("input")?.focus();
      }}>
        <summary className="wb-btn workspace-action" role="button" aria-label="New workspace" title="New workspace"
          aria-disabled={!isLeader} onClick={(event) => { if (!isLeader) event.preventDefault(); }}>
          <span aria-hidden="true">＋</span><span className="action-label">New</span>
        </summary>
        <form className="workspace-create-form" onSubmit={(event) => {
          event.preventDefault(); onCreateWorkspace(); if (menu.current) menu.current.open = false;
        }}>
          <label htmlFor="new-workspace-name">Workspace name</label>
          <input id="new-workspace-name" aria-label="New Workspace Name" placeholder="New workspace name…" value={newWorkspaceName} disabled={!isLeader}
            onChange={(event) => onChangeNewWorkspaceName(event.target.value)} />
          <button type="submit" disabled={!isLeader}>Create workspace</button>
        </form>
      </details>
      <button type="button" className="workspace-action" aria-label="Delete workspace" title="Delete workspace" disabled={!isLeader} onClick={onDeleteWorkspace}><span aria-hidden="true">🗑</span><span className="action-label">Delete</span></button>
      <button type="button" data-browser-toggle aria-label="Browse files" title="Files, journal and project"
        aria-expanded={browserPanel !== null} aria-controls="workspace-browser" onClick={onToggleBrowser}>📁</button>
    </div>
    <button type="button" className="services-toggle" aria-label="Services" title="Services and workspace status"
      aria-expanded={showServicesPanel} aria-controls="service-config-panel" onClick={onToggleServicesPanel}>⚙️</button>
  </header>;
}

function ChatSessionSelectorBar({
  snap,
  isObserver,
  onSwitchChat,
  onCreateChat,
}: {
  readonly snap: WorkspaceStateSnapshot | null;
  readonly isObserver: boolean;
  readonly onSwitchChat: (sessionId: string, title?: string) => void;
  readonly onCreateChat: () => void;
}) {
  const activeSessionId = snap?.activeSessionId ?? "session-main";
  const extraChats = (snap?.sessions ?? []).filter(
    (s) => !s.parentSessionId && s.id !== "session-main",
  );

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <select
        aria-label="Switch Chat Session"
        value={activeSessionId}
        disabled={isObserver}
        onChange={(ev) => {
          const found = extraChats.find((c) => c.id === ev.target.value);
          onSwitchChat(ev.target.value, found?.title);
        }}
        style={{
          background: "#101c25",
          color: "#dbe5e4",
          border: "1px solid #30434b",
          borderRadius: "4px",
          padding: "2px 6px",
          fontSize: "11px",
        }}
      >
        <option value="session-main">Main Chat</option>
        {extraChats.map((c) => (
          <option key={c.id} value={c.id}>
            {c.title ?? c.id}
          </option>
        ))}
      </select>
      <button
        type="button"
        className="wb-btn"
        disabled={isObserver}
        onClick={onCreateChat}
        style={{ padding: "2px 6px", fontSize: "11px" }}
        title="Start a new chat in this workspace (shares workspace files)"
      >
        + New Chat
      </button>
      <span className="hash-badge">State: {snap?.runState ?? "idle"}</span>
    </div>
  );
}

function CenterRunStreamPane({
  snap,
  draftText,
  promptInput,
  isObserver,
  onChangePrompt,
  onSubmitPrompt,
  onSwitchChat,
  onCreateChat,
  onDeleteRun,
  onApprove,
  onResolveUncertain, pending, isBusy, isStopping, expanded, onExpand, onStop, onInspect, onRetry,
}: {
  readonly snap: WorkspaceStateSnapshot | null;
  readonly pending: PendingUserTurn | null;
  readonly isBusy: boolean;
  readonly isStopping: boolean;
  readonly expanded: boolean;
  readonly onExpand: () => void;
  readonly onStop: () => void;
  readonly onInspect: (record: RunJournalRecord) => void;
  readonly onRetry: () => void;
  readonly draftText: string;
  readonly promptInput: string;
  readonly isObserver: boolean;
  readonly onChangePrompt: (val: string) => void;
  readonly onSubmitPrompt: () => void;
  readonly onSwitchChat: (sessionId: string, title?: string) => void;
  readonly onCreateChat: () => void;
  readonly onDeleteRun: (runId: string) => void;
  readonly onApprove: (approved: boolean) => void;
  readonly onResolveUncertain: (
    res: "mark_succeeded" | "retry_anyway" | "cancel_run",
  ) => void;
}) {
  const scroll = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  useEffect(() => {
    if (follow.current && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [snap?.headSeq, draftText, pending, snap?.pendingApproval]);
  useEffect(() => { follow.current = true; }, [snap?.activeSessionId]);
  const pendingApproval = snap?.pendingApproval ?? null;
  const isUncertain =
    snap?.runState === "interrupted" &&
    snap.interruptReason === "uncertain_tool_outcome";
  const isSafeInterrupt =
    snap?.runState === "interrupted" && !isUncertain;

  return (
    <>
      <div className="pane-header">
        <span>Chat</span>
        <ChatSessionSelectorBar
          snap={snap}
          isObserver={isObserver || isBusy}
          onSwitchChat={onSwitchChat}
          onCreateChat={onCreateChat}
        />
        <button type="button" className="wb-btn pane-expand" aria-label={expanded ? "Exit chat fullscreen" : "Fullscreen chat"} aria-pressed={expanded} onClick={onExpand}>{expanded ? "⤡" : "⤢"}</button>
      </div>
      <div className="pane-body chat-scroll" ref={scroll} onScroll={() => {
        const element = scroll.current;
        if (element) follow.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
      }}>
        <ChatTimeline journal={snap?.journal ?? []} sessionId={snap?.activeSessionId ?? "session-main"}
          pending={pending} isBusy={isObserver || isBusy} onDelete={onDeleteRun} onInspect={onInspect} />

        {draftText && (
          <div className="msg-card uncommitted">
            <div className="msg-meta">
              <span>VELES · Writing…</span>
            </div>
            <div>{draftText}</div>
          </div>
        )}

        {pendingApproval && (
          <div className="approval-gate-card" role="region" aria-label="Tool Approval Gate">
            <div>
              ⚠️ <strong>[awaiting_approval]</strong> Tool{" "}
              <code>{pendingApproval.toolName}</code> (
              <span className="hash-badge">{pendingApproval.effectClass}</span>)
            </div>
            <pre className="code-pre" style={{ marginTop: 6, marginBottom: 8 }}>
              {JSON.stringify(pendingApproval.args, null, 2)}
            </pre>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onApprove(true)}
              >
                ✓ Approve Execution
              </button>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onApprove(false)}
              >
                ✕ Deny Tool
              </button>
            </div>
          </div>
        )}

        {isUncertain && snap?.activeRunId && (
          <div className="interrupt-banner" role="alert">
            <strong>
              ⚠️ Tool outcome needs review
            </strong>
            <p style={{ margin: "6px 0" }}>
              The run stopped while a tool with external effects was in flight.
              It may have finished outside the browser. Check the outcome before retrying.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onResolveUncertain("mark_succeeded")}
              >
                Mark Succeeded
              </button>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onResolveUncertain("retry_anyway")}
              >
                Retry Anyway
              </button>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onResolveUncertain("cancel_run")}
              >
                Cancel Run
              </button>
            </div>
          </div>
        )}

        {isSafeInterrupt && (
          <div className="interrupt-banner" role="alert">
            <strong>
              ⚡ Interrupted: <code>{snap?.interruptReason ?? "gateway_disconnected_safe_to_retry"}</code>
            </strong>
            {typeof snap?.journal[snap.journal.length - 1]?.payload.error ===
              "string" && (
              <pre className="code-pre" style={{ marginTop: 6, marginBottom: 4 }}>
                {String(snap.journal[snap.journal.length - 1]?.payload.error)}
              </pre>
            )}
            <p style={{ margin: "6px 0" }}>
              Uncommitted stream tokens were discarded cleanly. Safe to resume or start a new run.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={onRetry}
              >
                Resume / Retry Run
              </button>
              <button
                type="button"
                className="wb-btn"
                disabled={isObserver}
                onClick={() => onResolveUncertain("cancel_run")}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}
      </div>

      <form
        className="prompt-bar"
        onSubmit={(ev) => {
          ev.preventDefault();
          if (!isBusy) onSubmitPrompt();
        }}
      >
        <input
          type="text"
          aria-label="Message Veles"
          value={promptInput}
          disabled={isObserver}
          onChange={(ev) => onChangePrompt(ev.target.value)}
          placeholder={
            isObserver
              ? "Read-only observer tab (locked by leader)"
              : "Ask Veles…"
          }
        />
        {isBusy ? <button type="button" className="wb-btn stop-button" disabled={isObserver || isStopping} onClick={onStop}>
          {isStopping ? "Stopping…" : "Stop"}
        </button> : <button type="submit" className="wb-btn" disabled={isObserver || !promptInput.trim()}>Run Prompt</button>}
      </form>
    </>
  );
}

function PreviewBridgeToolbar({
  sensorsEnabled,
  onToggleSensors,
  onMountPreviewNow,
}: {
  readonly sensorsEnabled: boolean;
  readonly onToggleSensors: (enabled: boolean) => void;
  readonly onMountPreviewNow: () => void;
}) {
  return (
    <div className="preview-toolbar">
      <span className="preview-isolation" title="Sandboxed iframe with an isolated origin">🔒 Isolated preview</span>
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11 }}>
          <input
            type="checkbox"
            checked={sensorsEnabled}
            onChange={(ev) => onToggleSensors(ev.target.checked)}
          />
          📡 Sensors
        </label>
        <button type="button" className="wb-btn" onClick={onMountPreviewNow}>
          Remount Preview
        </button>
      </div>
    </div>
  );
}

function usePreviewSensorBridge(
  sandboxHostRef: React.RefObject<PreviewSandboxHost | null>,
  sensorsEnabled: boolean,
): void {
  useEffect(() => {
    sandboxHostRef.current?.setSensorsEnabled(sensorsEnabled);
    if (!sensorsEnabled || typeof window === "undefined") {
      return;
    }
    const lastState = { tiltX: 0, tiltY: 0, pointerX: 0, pointerY: 0 };
    const onOrientation = (ev: DeviceOrientationEvent) => {
      lastState.tiltX = typeof ev.gamma === "number" ? ev.gamma : 0;
      lastState.tiltY = typeof ev.beta === "number" ? ev.beta : 0;
      sandboxHostRef.current?.pushSensorSample({
        ...lastState,
        timestampMs: Date.now(),
      });
    };
    const onPointerMove = (ev: PointerEvent) => {
      const w = window.innerWidth || 1;
      const h = window.innerHeight || 1;
      lastState.pointerX = Number(((ev.clientX / w) * 2 - 1).toFixed(3));
      lastState.pointerY = Number(((ev.clientY / h) * 2 - 1).toFixed(3));
      sandboxHostRef.current?.pushSensorSample({
        ...lastState,
        timestampMs: Date.now(),
      });
    };
    window.addEventListener("deviceorientation", onOrientation);
    window.addEventListener("pointermove", onPointerMove);
    return () => {
      window.removeEventListener("deviceorientation", onOrientation);
      window.removeEventListener("pointermove", onPointerMove);
    };
  }, [sandboxHostRef, sensorsEnabled]);
}

function RightArtifactAndPreviewPane({
  rightTab, expanded, onExpand,
  previewGen,
  activeFilePath,
  activeFileHash,
  activeFileContent,
  selectedRecord,
  isObserver,
  autoRepairEnabled,
  sensorsEnabled,
  latestPreviewError,
  diagnostics,
  iframeHostContainerRef,
  onSelectRightTab,
  onMountPreviewNow,
  onToggleSensors,
  onToggleAutoRepair,
  onFixPreviewError, runBusy,
}: {
  readonly expanded: boolean;
  readonly onExpand: () => void;
  readonly rightTab: "code" | "preview";
  readonly previewGen: number;
  readonly activeFilePath: string | null;
  readonly activeFileHash: string | null | undefined;
  readonly activeFileContent: string | null | undefined;
  readonly selectedRecord: RunJournalRecord | null;
  readonly isObserver: boolean;
  readonly runBusy: boolean;
  readonly autoRepairEnabled: boolean;
  readonly sensorsEnabled: boolean;
  readonly latestPreviewError: PreviewRuntimeDiagnostic | undefined;
  readonly diagnostics: readonly PreviewRuntimeDiagnostic[];
  readonly iframeHostContainerRef: React.RefObject<HTMLDivElement | null>;
  readonly onSelectRightTab: (tab: "code" | "preview") => void;
  readonly onMountPreviewNow: () => void;
  readonly onToggleSensors: (enabled: boolean) => void;
  readonly onToggleAutoRepair: (enabled: boolean) => void;
  readonly onFixPreviewError: () => void;
}) {
  const [consoleOpen, setConsoleOpen] = useState(false);
  return (
    <>
      <div className="pane-header">
        <span>Inspector</span>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            className="wb-btn"
            aria-pressed={rightTab === "code"}
            onClick={() => onSelectRightTab("code")}
          >
            {selectedRecord ? "Journal" : "Code"}
          </button>
          <button
            type="button"
            className="wb-btn"
            aria-pressed={rightTab === "preview"}
            onClick={() => onSelectRightTab("preview")}
          >
            Live Preview
          </button>
          <button type="button" className="wb-btn pane-expand" aria-label={expanded ? "Exit inspector fullscreen" : "Fullscreen inspector"} aria-pressed={expanded} onClick={onExpand}>{expanded ? "⤡" : "⤢"}</button>
        </div>
      </div>
      <div className="pane-body inspector-body">
        <div className="inspector-code" style={{ display: rightTab === "code" ? "flex" : "none" }}>
          {!selectedRecord && <div className="section-box">
            <div className="section-title">
              <span>File: {activeFilePath ?? "(none)"}</span>
              <span className="hash-badge">
                SHA-256: {activeFileHash ?? "n/a"}
              </span>
            </div>
            {activeFileContent &&
              (activeFileContent.startsWith("data:image/") ||
                activeFilePath?.endsWith(".svg")) && (
                <div
                  style={{
                    padding: 12,
                    background: "#0d171f",
                    borderBottom: "1px solid #24353d",
                    display: "flex",
                    justifyContent: "center",
                  }}
                >
                  <img
                    src={
                      activeFileContent.startsWith("data:")
                        ? activeFileContent
                        : `data:image/svg+xml;utf8,${encodeURIComponent(activeFileContent)}`
                    }
                    alt={activeFilePath ?? "Artifact preview"}
                    style={{ maxWidth: "100%", maxHeight: 240, objectFit: "contain" }}
                  />
                </div>
              )}
            <pre className="code-pre">
              {activeFileContent ?? "// Run prompt to generate OPFS files"}
            </pre>
          </div>}
          {selectedRecord && (
            <div className="section-box">
              <div className="section-title">
                <span>
                  Journal Record #{selectedRecord.seq} ({selectedRecord.type})
                </span>
                <span className="hash-badge">{selectedRecord.committedState}</span>
              </div>
              <pre className="code-pre">
                {JSON.stringify(selectedRecord, null, 2)}
              </pre>
            </div>
          )}
        </div>
        <div className="preview-pane" style={{ display: rightTab === "preview" ? "flex" : "none" }}>
          <PreviewBridgeToolbar
            sensorsEnabled={sensorsEnabled}
            onToggleSensors={onToggleSensors}
            onMountPreviewNow={onMountPreviewNow}
          />
          <div ref={iframeHostContainerRef} className="preview-frame-container" />
          <div className="preview-footer">
            <div className="preview-controls">
              <button type="button" className="wb-btn" aria-expanded={consoleOpen} aria-controls="preview-console" onClick={() => setConsoleOpen((open) => !open)}>
                Console ({diagnostics.length})
              </button>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 10 }}>
                  <input
                    type="checkbox"
                    checked={autoRepairEnabled}
                    disabled={isObserver}
                    onChange={(ev) => onToggleAutoRepair(ev.target.checked)}
                  />
                  Auto-Repair (1x)
                </label>
                {latestPreviewError && (
                  <button
                    type="button"
                    className="wb-btn"
                    disabled={isObserver || runBusy}
                    onClick={onFixPreviewError}
                    style={{ padding: "2px 6px", fontSize: "11px" }}
                  >
                    🔧 Fix Error with Agent
                  </button>
                )}
                <span>Generation #{previewGen}</span>
              </div>
            </div>
            <div id="preview-console" className="console-box" hidden={!consoleOpen}>
              {diagnostics.length === 0 && (
                <div>Waiting for opaque-origin postMessage diagnostics...</div>
              )}
              {diagnostics.map((d, i) => (
                <div key={i}>
                  [{d.type} · gen #{d.generation}] {d.message}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

const LAST_WORKSPACE_STORAGE_KEY = "veles:activeWorkspaceId";

function saveLastActiveWorkspaceId(workspaceId: string): void {
  try {
    window.localStorage?.setItem(LAST_WORKSPACE_STORAGE_KEY, workspaceId);
  } catch {
    // Ignore storage access errors
  }
}

function readLastActiveWorkspaceId(): string | null {
  try {
    return window.localStorage?.getItem(LAST_WORKSPACE_STORAGE_KEY) ?? null;
  } catch {
    return null;
  }
}

function syncInitialWorkspaceSelection(
  client: VelesAppClient,
  snap: WorkspaceStateSnapshot,
  onResolved: (s: WorkspaceStateSnapshot) => void,
): void {
  const savedId = readLastActiveWorkspaceId();
  const target =
    savedId && snap.leaseMode !== "ReadOnly_Observer"
      ? snap.workspaces?.find((w) => w.workspaceId === savedId)
      : undefined;
  if (target && target.workspaceId !== snap.workspaceId) {
    void client
      .switchWorkspace(target.workspaceId, target.title)
      .then(onResolved)
      .catch(() => onResolved(snap));
    return;
  }
  onResolved(snap);
}

export function VelesWorkbenchApp({ client }: { readonly client: VelesAppClient }) {
  const [view, setView] = useState<AppClientStateView>(() => client.getStateView());
  const [selectedFile, setSelectedFile] = useState<string | null>("index.html");
  const [selectedRecord, setSelectedRecord] = useState<RunJournalRecord | null>(null);
  const [browserPanel, setBrowserPanel] = useState<BrowserPanel | null>(null);
  const [showServicesPanel, setShowServicesPanel] = useState(false);
  const closeServices = useCallback(() => setShowServicesPanel(false), []);

  const [expandedPane, setExpandedPane] = useState<ExpandedPane>(null);
  const closeBrowser = useCallback(() => setBrowserPanel(null), []);
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !browserPanel && !showServicesPanel) setExpandedPane(null);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [browserPanel, showServicesPanel]);
  const [rightTab, setRightTab] = useState<"code" | "preview">("preview");
  const [promptInput, setPromptInput] = useState(
    "Build a sandboxed Counter App in index.html and clean up obsolete.txt",
  );
  const [diagnostics, setDiagnostics] = useState<PreviewRuntimeDiagnostic[]>([]);
  const [previewGen, setPreviewGen] = useState(0);
  const pwa = usePwaDeviceSession();
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [autoRepairEnabled, setAutoRepairEnabled] = useState(false);
  const [sensorsEnabled, setSensorsEnabled] = useState(false);
  const mountedPreviewRef = useRef<{
    readonly generation: number;
    readonly assetsSignature: string;
    readonly entryPath: string;
  } | null>(null);

  const iframeHostContainerRef = useRef<HTMLDivElement | null>(null);
  const sandboxHostRef = useRef<PreviewSandboxHost | null>(null);

  usePreviewSensorBridge(sandboxHostRef, sensorsEnabled);

  const resetLocalWorkspaceView = (nextSnap?: WorkspaceStateSnapshot) => {
    if (nextSnap) {
      saveLastActiveWorkspaceId(nextSnap.workspaceId);
    }
    const firstFile = nextSnap ? pickDefaultWorkspaceFile(nextSnap.files) : null;
    setSelectedFile(firstFile);
    setSelectedRecord(null);
    setDiagnostics([]);
    if (!firstFile) {
      setPreviewGen(0);
      mountedPreviewRef.current = null;
      if (iframeHostContainerRef.current) {
        iframeHostContainerRef.current.innerHTML = "";
      }
    }
  };

  useEffect(() => {
    const unsub = client.onStateChange((nextView) => {
      setView({ ...nextView });
    });
    void client.initWorkspace("Alpha").then((snap) => {
      syncInitialWorkspaceSelection(client, snap, (resolvedSnap) => {
        saveLastActiveWorkspaceId(resolvedSnap.workspaceId);
        const preferred = pickDefaultWorkspaceFile(resolvedSnap.files);
        if (preferred) {
          setSelectedFile(preferred);
        }
      });
    });
    return () => {
      unsub();
      sandboxHostRef.current?.dispose();
    };
  }, [client]);

  const mountPreviewNow = () => {
    const container = iframeHostContainerRef.current;
    if (!container || !view.snapshot) return;
    if (!sandboxHostRef.current || sandboxHostRef.current.isDisposed) {
      sandboxHostRef.current = new PreviewSandboxHost(
        createDomSandboxEnvironment(() => iframeHostContainerRef.current),
        (diag) => setDiagnostics((prev) => [...prev, diag]),
      );
      sandboxHostRef.current.setSensorsEnabled(sensorsEnabled);
    }
    const candidateEntry = resolvePreviewEntryPath(
      view.snapshot.files,
      selectedFile,
    );
    setDiagnostics([]);
    const gen = client.mountWorkspacePreview(
      sandboxHostRef.current,
      candidateEntry,
    );
    mountedPreviewRef.current = {
      generation: gen,
      assetsSignature: workspaceAssetsSignature,
      entryPath: candidateEntry,
    };
    setPreviewGen(gen);
  };

  const hasRenderableEntry = Object.keys(view.snapshot?.files ?? {}).some(
    (p) => /\.(html|svg|png|jpe?g|webp|gif)$/i.test(p),
  );
  const workspaceAssetsSignature = `${view.snapshot?.workspaceId ?? ""}|${Object.entries(
    view.snapshot?.fileHashes ?? {},
  )
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}:${v}`)
    .join("|")}`;
  useEffect(() => {
    if (!hasRenderableEntry || !workspaceAssetsSignature) {
      if (iframeHostContainerRef.current) {
        mountedPreviewRef.current = null;
        iframeHostContainerRef.current.innerHTML = "";
      }
      return;
    }
    if (rightTab !== "preview") {
      return;
    }
    const candidateEntry = resolvePreviewEntryPath(
      view.snapshot?.files ?? {},
      selectedFile,
    );
    const mounted = mountedPreviewRef.current;
    if (
      mounted &&
      mounted.assetsSignature === workspaceAssetsSignature &&
      mounted.entryPath === candidateEntry
    ) {
      return;
    }
    mountPreviewNow();
  }, [hasRenderableEntry, workspaceAssetsSignature, selectedFile, rightTab]);

  const snap = view.snapshot;
  const isObserver = snap?.leaseMode === "ReadOnly_Observer";
  const activeFilePath =
    (selectedFile && snap?.files[selectedFile] !== undefined
      ? selectedFile
      : null) ?? (snap ? pickDefaultWorkspaceFile(snap.files) : null);
  const activeFileContent = activeFilePath ? snap?.files[activeFilePath] : null;
  const activeFileHash = activeFilePath ? snap?.fileHashes[activeFilePath] : null;
  const latestPreviewError = [...diagnostics]
    .reverse()
    .find((d) => d.type === "preview_error");

  useEffect(() => {
    if (
      !autoRepairEnabled ||
      Boolean(view.inFlightRunId) ||
      isObserver ||
      !snap ||
      snap.runState !== "completed" ||
      !latestPreviewError
    ) {
      return;
    }
    const mounted = mountedPreviewRef.current;
    if (
      latestPreviewError.generation !== mounted?.generation ||
      mounted.assetsSignature !== workspaceAssetsSignature
    ) {
      return;
    }
    // Consume the activation before dispatch; failed repairs require a new opt-in.
    setAutoRepairEnabled(false);
    void client.startRun(
      `run-autorepair-${Date.now()}`,
      `Delegate a preview-repair child agent via agent.delegate to inspect and fix this runtime error in the preview files: ${latestPreviewError.message}`,
    ).catch(() => {});
  }, [view.inFlightRunId, autoRepairEnabled, isObserver, snap, latestPreviewError, workspaceAssetsSignature, client]);

  const isBusy = Boolean(view.inFlightRunId || view.stoppingRunId) || isActiveRun(snap?.runState);
  const submitPrompt = (prompt: string) => {
    if (isObserver || isBusy || !prompt.trim()) return;
    setPromptInput("");
    void client.startRun(`run-${crypto.randomUUID()}`, prompt.trim()).catch(() => setPromptInput(prompt));
  };
  const handleRunPrompt = () => submitPrompt(promptInput);
  const inspectRecord = (record: RunJournalRecord) => {
    setSelectedRecord(record);
    setRightTab("code");
    setBrowserPanel(null);
    setExpandedPane((pane) => pane === "chat" || window.matchMedia("(max-width: 760px)").matches ? "inspector" : pane);
  };
  const inspectFile = (path: string) => {
    setSelectedFile(path);
    setSelectedRecord(null);
    setRightTab("code");
    setBrowserPanel(null);
    setExpandedPane((pane) => pane === "chat" || window.matchMedia("(max-width: 760px)").matches ? "inspector" : pane);
  };

  const handleCreateWorkspace = () => {
    if (isObserver) return;
    const existingCount = snap?.workspaces?.length ?? 1;
    const cleanTitle =
      newWorkspaceName.trim() || `Workspace ${existingCount + 1}`;
    const slug =
      cleanTitle
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "") || "workspace";
    const id = `ws-${slug}-${Date.now().toString(36).slice(-4)}`;
    setNewWorkspaceName("");
    setPromptInput("");
    void client
      .switchWorkspace(id, cleanTitle)
      .then((nextSnap) => {
        resetLocalWorkspaceView(nextSnap);
      })
      .catch(() => {});
  };

  const handleSwitchWorkspace = (id: string, title?: string) => {
    if (isObserver) return;
    void client
      .switchWorkspace(id, title)
      .then((nextSnap) => {
        resetLocalWorkspaceView(nextSnap);
      })
      .catch(() => {});
  };

  const handleDeleteWorkspace = () => {
    if (!snap || isObserver) return;
    void client
      .deleteWorkspace(snap.workspaceId)
      .then((nextSnap) => {
        resetLocalWorkspaceView(nextSnap);
      })
      .catch(() => {});
  };

  const handleExport = () => {
    void client
      .exportWorkspace()
      .then((bundle) => {
        const json = JSON.stringify(bundle, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `veles-${snap?.workspaceId ?? "workspace"}-seq-${snap?.headSeq ?? 0}.json`;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch(() => {});
  };

  const handleImportPrompt = () => {
    const raw = window.prompt("Paste your exported project JSON:");
    if (!raw) return;
    try {
      const parsed: unknown = JSON.parse(raw);
      void client.importWorkspace(parsed).then(resetLocalWorkspaceView).catch(() => {});
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <WorkbenchLayout
      expandedPane={expandedPane}
      topBar={
        <TopAuthorityBar
          snap={snap}
          browserPanel={browserPanel}
          onToggleBrowser={() => { setShowServicesPanel(false); setBrowserPanel((current) => current === null ? "files" : null); }}
          newWorkspaceName={newWorkspaceName}
          showServicesPanel={showServicesPanel}
          onChangeNewWorkspaceName={setNewWorkspaceName}
          onToggleServicesPanel={() => { setBrowserPanel(null); setShowServicesPanel((prev) => !prev); }}
          onSwitchWorkspace={handleSwitchWorkspace}
          onCreateWorkspace={handleCreateWorkspace}
          onDeleteWorkspace={handleDeleteWorkspace}
        />
      }
      banner={
        <>
          {showServicesPanel && (
            <ServiceConfigPanel
              snap={snap}
              pwa={pwa}
              provider={view.selectedProvider}
              serviceConfig={view.serviceConfig}
              isObserver={Boolean(isObserver)}
              onChangeProvider={(p) => client.selectProviderModel(p)}
              onChangeServiceConfig={(cfg) => client.selectServiceConfig(cfg)}
              onClose={closeServices}
            />
          )}
          {view.lastError && (
            <div className="quota-error-banner" role="alert">
              <span>⚠️ Kernel / Storage Alert: {view.lastError}</span>
            </div>
          )}
        </>
      }
      overlay={browserPanel && <WorkspaceBrowser panel={browserPanel} snapshot={snap} selectedFile={activeFilePath}
        selectedRecord={selectedRecord} onClose={closeBrowser} onChangePanel={setBrowserPanel}
        onSelectFile={inspectFile} onSelectRecord={inspectRecord}
        onImport={handleImportPrompt} onExport={handleExport}
        onUpload={(path, content) => { void client.uploadWorkspaceFile(path, content).catch(() => {}); }} />}
      chatPane={
        <CenterRunStreamPane
          snap={snap}
          pending={view.pendingUserTurn}
          isBusy={isBusy}
          isStopping={Boolean(view.stoppingRunId)}
          expanded={expandedPane === "chat"}
          onExpand={() => setExpandedPane((pane) => pane === "chat" ? null : "chat")}
          onInspect={inspectRecord}
          onStop={() => {
            if (snap?.activeRunId) {
              void client.cancelRun(snap.activeRunId).catch(() => {});
            }
          }}
          onRetry={() => {
            const lastTurn = projectChatRuns(
              snap?.journal ?? [],
              snap?.activeSessionId ?? "session-main",
            ).at(-1);
            submitPrompt(lastTurn?.prompt ?? "");
          }}
          draftText={view.streamingDraftText}
          promptInput={promptInput}
          isObserver={isObserver}
          onChangePrompt={setPromptInput}
          onSubmitPrompt={handleRunPrompt}
          onSwitchChat={(sessionId, title) => {
            void client.switchChatSession(sessionId, title).catch(() => {});
          }}
          onCreateChat={() => {
            const topChatCount =
              (snap?.sessions ?? []).filter(
                (s) => !s.parentSessionId && s.id !== "session-main",
              ).length + 2;
            const chatId = `chat-${Date.now().toString(36).slice(-4)}`;
            void client
              .switchChatSession(chatId, `Chat ${topChatCount}`)
              .catch(() => {});
          }}
          onDeleteRun={(runId) => {
            void client.deleteChatRun(runId).catch(() => {});
          }}
          onApprove={(approved) => {
            if (snap?.activeRunId) {
              void client.resolveApproval(snap.activeRunId, approved).catch(() => {});
            }
          }}
          onResolveUncertain={(res) => {
            if (snap?.activeRunId) {
              void client.resolveUncertainTool(snap.activeRunId, res).catch(() => {});
            }
          }}
        />
      }
      inspectorPane={
        <RightArtifactAndPreviewPane
          expanded={expandedPane === "inspector"}
          onExpand={() => setExpandedPane((pane) => pane === "inspector" ? null : "inspector")}
          rightTab={rightTab}
          previewGen={previewGen}
          activeFilePath={activeFilePath}
          activeFileHash={activeFileHash}
          activeFileContent={activeFileContent}
          selectedRecord={selectedRecord}
          isObserver={Boolean(isObserver)}
          runBusy={isBusy}
          autoRepairEnabled={autoRepairEnabled}
          sensorsEnabled={sensorsEnabled}
          latestPreviewError={latestPreviewError}
          diagnostics={diagnostics}
          iframeHostContainerRef={iframeHostContainerRef}
          onSelectRightTab={setRightTab}
          onMountPreviewNow={mountPreviewNow}
          onToggleSensors={(enabled) => {
            setSensorsEnabled(enabled);
            sandboxHostRef.current?.setSensorsEnabled(enabled);
          }}
          onToggleAutoRepair={setAutoRepairEnabled}
          onFixPreviewError={() => {
            if (latestPreviewError) {
              void client.startRun(
                `run-fix-${Date.now()}`,
                `Delegate a preview-repair child agent via agent.delegate to inspect and fix this runtime error in the preview files: ${latestPreviewError.message}`,
              ).catch(() => {});
            }
          }}
        />
      }
    />
  );
}
