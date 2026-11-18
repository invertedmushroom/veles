import type { ProviderModelConfigSpec, ServiceCapabilitiesConfigSpec } from "../protocol/types.ts";

const KEY = "veles.services.v1";
export interface ServiceSettings {
  readonly provider: ProviderModelConfigSpec;
  readonly services: ServiceCapabilitiesConfigSpec;
  readonly rememberKeys: boolean;
}

export function loadServiceSettings(storage: Pick<Storage, "getItem">): ServiceSettings | null {
  try {
    const raw = JSON.parse(storage.getItem(KEY) ?? "null");
    if (raw?.version !== 1 || !validProvider(raw.provider) || !validServices(raw.services)) return null;
    return { provider: raw.provider, services: raw.services, rememberKeys: raw.rememberKeys === true };
  } catch { return null; }
}

function validProvider(raw: unknown): raw is ProviderModelConfigSpec {
  if (!raw || typeof raw !== "object") return false;
  const p = raw as Record<string, unknown>;
  return [p.providerId, p.baseUrl, p.model].every((v) => typeof v === "string") &&
    (p.apiKey === undefined || typeof p.apiKey === "string");
}

function validServices(raw: unknown): raw is ServiceCapabilitiesConfigSpec {
  if (!raw || typeof raw !== "object") return false;
  const s = raw as Record<string, unknown>;
  if (!s.decision || typeof s.decision !== "object") return false;
  const d = s.decision as Record<string, unknown>;
  return validProvider({ ...d, providerId: d.mode, baseUrl: d.endpointUrl }) &&
    validServiceModes(s, d) && validDecisionNumbers(d) &&
    (s.searxngUrl === undefined || typeof s.searxngUrl === "string");
}

function validServiceModes(s: Record<string, unknown>, d: Record<string, unknown>): boolean {
  return ["tier0-only", "typesafe", "openai-decisions", "local-laya"].includes(String(d.mode)) &&
    ["openai-auto", "openai-dalle3", "openai-gpt-svg", "local-deterministic-svg"].includes(String(s.imageMode)) &&
    ["openai-responses", "offline-disabled", "searxng"].includes(String(s.webSearchMode));
}

function validDecisionNumbers(d: Record<string, unknown>): boolean {
  return (d.timeoutMs === undefined || typeof d.timeoutMs === "number" && Number.isFinite(d.timeoutMs) && d.timeoutMs > 0) &&
    (d.threshold === undefined || typeof d.threshold === "number" && d.threshold >= 0 && d.threshold <= 1);
}

export function saveServiceSettings(storage: Pick<Storage, "setItem">, settings: ServiceSettings): void {
  const { provider, services, rememberKeys } = settings;
  storage.setItem(KEY, JSON.stringify({ version: 1, rememberKeys,
    provider: { ...provider, apiKey: rememberKeys ? provider.apiKey : undefined },
    services: { ...services, decision: { ...services.decision, apiKey: rememberKeys ? services.decision.apiKey : undefined } },
  }));
}

export async function discoverModels(provider: ProviderModelConfigSpec, signal: AbortSignal): Promise<string[]> {
  const url = new URL(`${provider.baseUrl.replace(/\/+$/, "")}/models`);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("This provider has no HTTP model catalog.");
  const response = await fetch(url, { signal, redirect: "error", credentials: "omit",
    headers: provider.apiKey ? { Authorization: `Bearer ${provider.apiKey}` } : {},
  });
  if (!response.ok) throw new Error(`Model catalog returned HTTP ${response.status}`);
  const raw: unknown = await response.json();
  const data = (raw as { data?: unknown })?.data;
  if (!Array.isArray(data)) throw new Error("Provider returned an unsupported model catalog.");
  return [...new Set(data.flatMap((entry) => typeof entry?.id === "string" ? [entry.id] : []))].sort();
}
