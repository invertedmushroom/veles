/**
 * @module SYS-UI/main
 * Browser entrypoint spawning the Dedicated Web Worker (`worker-entry.ts`)
 * and mounting `VelesWorkbenchApp` strictly over the `VelesAppClient` `MessagePort` seam.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { WorkerPortLike } from "../protocol/types.ts";
import { DEFAULT_BROWSER_PROVIDER, DEFAULT_BROWSER_SERVICES } from "./service-presets.ts";
import { VelesAppClient } from "./app-client.ts";
import { VelesWorkbenchApp } from "./VelesWorkbenchApp.tsx";

const worker = new Worker(
  new URL("../kernel/worker-entry.ts", import.meta.url),
  { type: "module" },
);

const client = new VelesAppClient(
  worker as unknown as WorkerPortLike,
  DEFAULT_BROWSER_PROVIDER,
  DEFAULT_BROWSER_SERVICES,
);

const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <VelesWorkbenchApp client={client} />
    </StrictMode>,
  );
}
