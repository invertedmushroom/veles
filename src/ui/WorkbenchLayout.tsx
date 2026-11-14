/**
 * @module SYS-UI/WorkbenchLayout
 * Resizable chat and inspector workbench adapted from `local-ai`'s `WorkspaceLayout.tsx`.
 * Releases pointer capture on `blur`, `pagehide`, and `visibilitychange` per
 * `.context/references/typescript-browser-apps/browser-runtime.md#suspend-and-capabilities`.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

interface ResizableDividerProps {
  readonly label: string;
  readonly ratio: number;
  readonly onDeltaPx: (deltaPx: number) => void;
}

export function ResizablePaneDivider({
  label,
  ratio,
  onDeltaPx,
}: ResizableDividerProps) {
  const dragStartX = useRef<number | null>(null);
  const activePointerId = useRef<number | null>(null);
  const dividerRef = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const releaseDrag = () => {
    const el = dividerRef.current;
    const pid = activePointerId.current;
    dragStartX.current = null;
    activePointerId.current = null;
    if (el && pid !== null && el.hasPointerCapture(pid)) {
      el.releasePointerCapture(pid);
    }
    setDragging(false);
  };

  useEffect(() => {
    if (!dragging) return;
    const onVisibility = () => {
      if (document.hidden) releaseDrag();
    };
    window.addEventListener("blur", releaseDrag);
    window.addEventListener("pagehide", releaseDrag);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("blur", releaseDrag);
      window.removeEventListener("pagehide", releaseDrag);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [dragging]);

  return (
    <div
      ref={dividerRef}
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={30}
      aria-valuemax={70}
      aria-valuenow={Math.round(ratio * 100)}
      className={`pane-divider${dragging ? " is-interacting" : ""}`}
      onKeyDown={(event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        onDeltaPx(event.key === "ArrowLeft" ? -24 : 24);
      }}
      onPointerDown={(ev) => {
        if (ev.button !== 0 || !ev.isPrimary) return;
        ev.preventDefault();
        ev.currentTarget.setPointerCapture(ev.pointerId);
        dragStartX.current = ev.clientX;
        activePointerId.current = ev.pointerId;
        setDragging(true);
      }}
      onPointerMove={(ev) => {
        if (dragStartX.current === null || activePointerId.current !== ev.pointerId) {
          return;
        }
        const delta = ev.clientX - dragStartX.current;
        dragStartX.current = ev.clientX;
        onDeltaPx(delta);
      }}
      onPointerUp={releaseDrag}
      onPointerCancel={releaseDrag}
      onLostPointerCapture={releaseDrag}
    />
  );
}

export type ExpandedPane = "chat" | "inspector" | null;

export function WorkbenchLayout({ topBar, banner, overlay, chatPane, inspectorPane, expandedPane }: {
  readonly topBar: ReactNode;
  readonly banner?: ReactNode;
  readonly overlay?: ReactNode;
  readonly chatPane: ReactNode;
  readonly inspectorPane: ReactNode;
  readonly expandedPane: ExpandedPane;
}) {
  const [chatPct, setChatPct] = useState(46);
  const grid = useRef<HTMLDivElement>(null);
  const adjustChat = (deltaPx: number) => {
    const width = grid.current?.clientWidth ?? window.innerWidth;
    setChatPct((previous) => Math.min(70, Math.max(30, previous + deltaPx / width * 100)));
  };
  const gridStyle = {
    "--chat-size": `${chatPct}fr`,
    "--inspector-size": `${100 - chatPct}fr`,
  } as CSSProperties;
  return <div className="workspace-ui">
    {topBar}
    {banner}
    {overlay}
    <main className="workspace-stage">
      <div ref={grid} className={`workspace-grid${expandedPane ? ` expanded-${expandedPane}` : ""}`} style={gridStyle}>
        <section className="workspace-pane chat-pane" aria-label="Agent Run Stream" hidden={expandedPane === "inspector"}>
          {chatPane}
        </section>
        <ResizablePaneDivider label="Resize Chat and Inspector panes" ratio={chatPct / 100} onDeltaPx={adjustChat} />
        <section className="workspace-pane inspector-pane" aria-label="Artifact Inspector and Sandboxed Preview" hidden={expandedPane === "chat"}>
          {inspectorPane}
        </section>
      </div>
    </main>
  </div>;
}
