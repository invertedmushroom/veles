import { useEffect, useRef, useState } from "react";
import type { RunJournalRecord, WorkspaceStateSnapshot } from "../protocol/types.ts";

export type BrowserPanel = "files" | "journal" | "project";

const PANEL_LABELS: Record<BrowserPanel, string> = {
  files: "Workspace files",
  journal: "Run journal",
  project: "Project archive",
};

function FileList({ snapshot, selectedFile, onSelect, onUpload }: {
  readonly snapshot: WorkspaceStateSnapshot | null;
  readonly selectedFile: string | null;
  readonly onSelect: (path: string) => void;
  readonly onUpload: (path: string, content: string | Uint8Array) => void;
}) {
  const [query, setQuery] = useState("");
  const paths = Object.keys(snapshot?.files ?? {}).sort().filter((path) => path.toLowerCase().includes(query.toLowerCase()));
  const input = useRef<HTMLInputElement>(null);
  return <>
    <div className="browser-actions"><input autoFocus aria-label="Find file" placeholder="Find a file…" value={query} onChange={(event) => setQuery(event.target.value)} />
      <button type="button" className="wb-btn" disabled={snapshot?.leaseMode === "ReadOnly_Observer"} onClick={() => input.current?.click()}>+ Upload File</button>
      <input ref={input} type="file" hidden onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void file.arrayBuffer().then((buffer) => onUpload(`assets/${file.name}`, new Uint8Array(buffer)));
        event.target.value = "";
      }} />
    </div>
    <ul className="item-list browser-list">{paths.map((path) => <li key={path}>
      <button className={`item-row${selectedFile === path ? " active" : ""}`} type="button" onClick={() => onSelect(path)}>
        <span>{path}</span><span className="hash-badge">{snapshot?.fileHashes[path]?.slice(0, 8)}</span>
      </button>
    </li>)}</ul>
    {paths.length === 0 && <p className="browser-empty">No files match.</p>}
  </>;
}

function JournalList({ journal, selectedId, onSelect }: {
  readonly journal: readonly RunJournalRecord[];
  readonly selectedId: string | undefined;
  readonly onSelect: (record: RunJournalRecord) => void;
}) {
  const [query, setQuery] = useState("");
  const records = [...journal].reverse().filter((record) => `${record.seq} ${record.type} ${record.runId}`.toLowerCase().includes(query.toLowerCase()));
  return <>
    <div className="browser-actions"><input autoFocus aria-label="Find journal record" placeholder="Find event, run, or sequence…" value={query} onChange={(event) => setQuery(event.target.value)} /></div>
    <ul className="item-list browser-list">{records.map((record) => <li key={record.eventId}>
      <button type="button" className={`item-row${selectedId === record.eventId ? " active" : ""}`} onClick={() => onSelect(record)}>
        <span><strong>#{record.seq}</strong> {record.type}</span><span className="hash-badge">{record.committedState}</span>
      </button>
    </li>)}</ul>
    {records.length === 0 && <p className="browser-empty">No journal records match.</p>}
  </>;
}

function ProjectActions({ snapshot, onImport, onExport }: {
  readonly snapshot: WorkspaceStateSnapshot | null;
  readonly onImport: () => void;
  readonly onExport: () => void;
}) {
  return <section className="browser-project" aria-label="Project actions">
    <strong>{snapshot?.title ?? "Project"}</strong>
    <p>Export this project as a JSON archive, or import a saved archive.</p>
    <div className="project-actions">
      <button type="button" className="wb-btn" disabled={!snapshot || snapshot.leaseMode === "ReadOnly_Observer"} onClick={onImport}>↧ Import project</button>
      <button type="button" className="wb-btn" disabled={!snapshot} onClick={onExport}>↥ Export project</button>
    </div>
  </section>;
}

export function WorkspaceBrowser({ panel, snapshot, selectedFile, selectedRecord, onClose, onChangePanel, onSelectFile, onSelectRecord, onUpload, onImport, onExport }: {
  readonly panel: BrowserPanel;
  readonly snapshot: WorkspaceStateSnapshot | null;
  readonly selectedFile: string | null;
  readonly selectedRecord: RunJournalRecord | null;
  readonly onClose: () => void;
  readonly onChangePanel: (panel: BrowserPanel) => void;
  readonly onSelectFile: (path: string) => void;
  readonly onSelectRecord: (record: RunJournalRecord) => void;
  readonly onUpload: (path: string, content: string | Uint8Array) => void;
  readonly onImport: () => void;
  readonly onExport: () => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const previousFocus = useRef(document.activeElement);
  useEffect(() => {
    const previous = previousFocus.current;
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); onClose(); } };
    const outside = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Element && !container.current?.contains(target) && !target.closest("[data-browser-toggle]")) onClose();
    };
    document.addEventListener("keydown", escape);
    document.addEventListener("pointerdown", outside);
    return () => {
      document.removeEventListener("keydown", escape);
      document.removeEventListener("pointerdown", outside);
      if (previous instanceof HTMLElement && previous.isConnected) previous.focus();
    };
  }, [onClose]);
  return <div id="workspace-browser" className="workspace-browser" ref={container} role="dialog" aria-label={PANEL_LABELS[panel]}>
    <div className="pane-header"><div className="browser-tabs">
      <button className="wb-btn" type="button" aria-pressed={panel === "files"} onClick={() => onChangePanel("files")}>📁 Files ({Object.keys(snapshot?.files ?? {}).length})</button>
      <button className="wb-btn" type="button" aria-pressed={panel === "journal"} onClick={() => onChangePanel("journal")}>📜 Journal ({snapshot?.journal.length ?? 0})</button>
      <button className="wb-btn" type="button" aria-pressed={panel === "project"} onClick={() => onChangePanel("project")}>Project</button>
    </div><button className="wb-btn" type="button" aria-label="Close browser" onClick={onClose}>✕</button></div>
    {panel === "files" && <FileList snapshot={snapshot} selectedFile={selectedFile} onSelect={onSelectFile} onUpload={onUpload} />}
    {panel === "journal" && <JournalList journal={snapshot?.journal ?? []} selectedId={selectedRecord?.eventId} onSelect={onSelectRecord} />}
    {panel === "project" && <ProjectActions snapshot={snapshot} onImport={onImport} onExport={onExport} />}
  </div>;
}
