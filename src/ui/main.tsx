/**
 * @module SYS-UI/main
 * Browser entrypoint spawning the Dedicated Web Worker (`worker-entry.ts`)
 * and mounting `VelesWorkbenchApp` strictly over the `VelesAppClient` `MessagePort` seam.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { WorkerPortLike } from "../protocol/types.ts";
import { DEFAULT_BROWSER_PROVIDER, DEFAULT_BROWSER_SERVICES, PROVIDER_PRESETS, DECISION_MODE_PRESETS } from "./service-presets.ts";
import { VelesAppClient } from "./app-client.ts";
import { VelesWorkbenchApp } from "./VelesWorkbenchApp.tsx";
import { loadServiceSettings } from "./service-settings.ts";

const savedSettings = (() => {
  try { return loadServiceSettings(window.localStorage); } catch { return null; }
})();
const provider = savedSettings?.provider ?? DEFAULT_BROWSER_PROVIDER;
const preset = PROVIDER_PRESETS.find((entry) => entry.providerId === provider.providerId && entry.baseUrl === provider.baseUrl);
const services = savedSettings?.services ?? DEFAULT_BROWSER_SERVICES;
const decisionPreset = DECISION_MODE_PRESETS[services.decision.mode];

const worker = new Worker(
  new URL("../kernel/worker-entry.ts", import.meta.url),
  { type: "module" },
);

const client = new VelesAppClient(
  worker as unknown as WorkerPortLike,
  { ...provider, apiKey: provider.apiKey ?? preset?.apiKey },
  { ...services, decision: { ...services.decision, apiKey: services.decision.apiKey ??
    (decisionPreset.endpointUrl === services.decision.endpointUrl ? decisionPreset.apiKey : undefined) } },
);

const rootEl = document.getElementById("root");
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <VelesWorkbenchApp client={client} />
    </StrictMode>,
  );
}
