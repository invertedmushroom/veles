import { DEFAULT_SERVICE_CAPABILITIES_CONFIG, TYPESAFE_SYSTEM_ONE_ENDPOINT } from "../protocol/types.ts";
import type { DecisionServiceMode, ProviderModelConfigSpec, ServiceCapabilitiesConfigSpec } from "../protocol/types.ts";

declare const __VELES_OPENAI_KEY__: string | undefined;
declare const __VELES_OPENROUTER_KEY__: string | undefined;
declare const __VELES_TYPESAFE_KEY__: string | undefined;
declare const __VELES_LAYA_URL__: string | undefined;
declare const __VELES_DEFAULT_PROVIDER__: string | undefined;
declare const __VELES_DEFAULT_DECISION_MODE__: string | undefined;

const ENV_OPENAI_KEY =
  typeof __VELES_OPENAI_KEY__ === "string" && __VELES_OPENAI_KEY__.length > 0
    ? __VELES_OPENAI_KEY__
    : undefined;

const ENV_OPENROUTER_KEY =
  typeof __VELES_OPENROUTER_KEY__ === "string" &&
  __VELES_OPENROUTER_KEY__.length > 0
    ? __VELES_OPENROUTER_KEY__
    : undefined;

const ENV_TYPESAFE_KEY =
  typeof __VELES_TYPESAFE_KEY__ === "string" &&
  __VELES_TYPESAFE_KEY__.length > 0
    ? __VELES_TYPESAFE_KEY__
    : undefined;

const ENV_LAYA_URL =
  typeof __VELES_LAYA_URL__ === "string" && __VELES_LAYA_URL__.length > 0
    ? __VELES_LAYA_URL__
    : "http://127.0.0.1:11434/v1/decisions";

export const OPENAI_DECISIONS_ENDPOINT = "https://api.openai.com/v1/decisions";

export const DECISION_MODE_PRESETS: Record<
  DecisionServiceMode,
  { readonly label: string; readonly endpointUrl: string; readonly model: string; readonly apiKey?: string }
> = {
  "tier0-only": {
    label: "Tier-0 Deterministic Rules (0ms, $0)",
    endpointUrl: "local://tier0-deterministic",
    model: "deterministic-rules",
  },
  typesafe: {
    label: "TypeSafe / Jev Decision API (jev-1.14.0)",
    apiKey: ENV_TYPESAFE_KEY,
    endpointUrl: TYPESAFE_SYSTEM_ONE_ENDPOINT,
    model: "jev-1.14.0",
  },
  "openai-decisions": {
    label: "OpenAI Decisions API (gpt-6-luna)",
    apiKey: ENV_OPENAI_KEY,
    endpointUrl: OPENAI_DECISIONS_ENDPOINT,
    model: "gpt-6-luna",
  },
  "local-laya": {
    label: "Local Laya Decision Server (laya-latest)",
    endpointUrl: ENV_LAYA_URL,
    model: "laya-latest",
  },
};

const DEMO_PROVIDER_PRESET: ProviderModelConfigSpec = {
  providerId: "veles-demo",
  baseUrl: "local://demo-comet",
  model: "comet-counter-v1",
};

const OPENAI_PROVIDER_PRESET: ProviderModelConfigSpec = {
  providerId: "openai",
  baseUrl: "https://api.openai.com/v1",
  model: "gpt-4o",
  apiKey: ENV_OPENAI_KEY,
};

export const PROVIDER_PRESETS: readonly ProviderModelConfigSpec[] = [
  DEMO_PROVIDER_PRESET,
  OPENAI_PROVIDER_PRESET,
  {
    providerId: "openrouter",
    baseUrl: "https://openrouter.ai/api/v1",
    model: "anthropic/claude-3.7-sonnet",
    apiKey: ENV_OPENROUTER_KEY,
  },
  {
    providerId: "ollama",
    baseUrl: "http://localhost:11434/v1",
    model: "qwen2.5-coder:7b",
  },
];

const ENV_DEFAULT_PROVIDER_ID =
  typeof __VELES_DEFAULT_PROVIDER__ === "string" &&
  __VELES_DEFAULT_PROVIDER__.length > 0
    ? __VELES_DEFAULT_PROVIDER__
    : undefined;

const ENV_DEFAULT_DECISION_MODE =
  typeof __VELES_DEFAULT_DECISION_MODE__ === "string" &&
  __VELES_DEFAULT_DECISION_MODE__ in DECISION_MODE_PRESETS
    ? (__VELES_DEFAULT_DECISION_MODE__ as DecisionServiceMode)
    : undefined;

export const DEFAULT_BROWSER_PROVIDER: ProviderModelConfigSpec =
  PROVIDER_PRESETS.find((p) => p.providerId === ENV_DEFAULT_PROVIDER_ID) ??
  OPENAI_PROVIDER_PRESET;

const DEFAULT_DECISION_MODE: DecisionServiceMode =
  ENV_DEFAULT_DECISION_MODE ?? "openai-decisions";

const IS_OFFLINE_DEMO_DEFAULT =
  DEFAULT_BROWSER_PROVIDER.providerId === "veles-demo";

export const DEFAULT_BROWSER_SERVICES: ServiceCapabilitiesConfigSpec = {
  ...DEFAULT_SERVICE_CAPABILITIES_CONFIG,
  imageMode: IS_OFFLINE_DEMO_DEFAULT
    ? "local-deterministic-svg"
    : DEFAULT_SERVICE_CAPABILITIES_CONFIG.imageMode,
  webSearchMode: IS_OFFLINE_DEMO_DEFAULT
    ? "offline-disabled"
    : DEFAULT_SERVICE_CAPABILITIES_CONFIG.webSearchMode,
  decision: {
    ...DEFAULT_SERVICE_CAPABILITIES_CONFIG.decision,
    mode: DEFAULT_DECISION_MODE,
    endpointUrl: DECISION_MODE_PRESETS[DEFAULT_DECISION_MODE].endpointUrl,
    model: DECISION_MODE_PRESETS[DEFAULT_DECISION_MODE].model,
    apiKey: DECISION_MODE_PRESETS[DEFAULT_DECISION_MODE].apiKey,
  },
};
