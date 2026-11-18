import { useEffect, useRef, useState } from "react";
import type { ProviderModelConfigSpec } from "../protocol/types.ts";
import { discoverModels } from "./service-settings.ts";

export function ModelCatalog({ provider, onSelect }: {
  readonly provider: ProviderModelConfigSpec;
  readonly onSelect: (id: string) => void;
}) {
  const [models, setModels] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => { request.current?.abort(); request.current = null; }, []);
  const probe = async () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const timer = setTimeout(() => controller.abort(), 10_000);
    setStatus("Loading models…");
    try {
      const found = await discoverModels(provider, controller.signal);
      if (controller.signal.aborted) return;
      setModels(found);
      setStatus(`${found.length} models available`);
    } catch (error) {
      if (request.current === controller) setStatus(controller.signal.aborted
        ? "Model request timed out or was cancelled."
        : `${error instanceof Error ? error.message : error}. Check the endpoint, key, and server CORS settings.`);
    } finally { clearTimeout(timer); }
  };
  const matches = models.filter((id) => id.toLowerCase().includes(query.toLowerCase()));
  return <details className="service-card">
    <summary>Browse provider models</summary>
    <button type="button" onClick={() => void probe()}>Fetch models</button>
    <p role="status">{status}</p>
    {models.length > 0 && <>
      <input className="service-input" aria-label="Filter models" value={query} onChange={(event) => setQuery(event.target.value)} />
      <select className="service-input" aria-label="Available models" size={8} value={provider.model}
        onChange={(event) => onSelect(event.target.value)}>
        {matches.slice(0, 100).map((id) => <option key={id} value={id}>{id}</option>)}
      </select>
      <p>{matches.length} matches; showing up to 100. Refine the filter or enter a model ID manually.</p>
    </>}
  </details>;
}
