import { DEFAULT_SERVICE_CAPABILITIES_CONFIG, TYPESAFE_SYSTEM_ONE_ENDPOINT } from "../protocol/types.ts";
import type { DecisionServiceMode, ProviderModelConfigSpec, ServiceCapabilitiesConfigSpec } from "../protocol/types.ts";

declare const __VELES_OPENAI_KEY__: string | undefined;
declare const __VELES_OPENROUTER_KEY__: string | undefined;
declare const __VELES_TYPESAFE_KEY__: string | undefined;
declare const __VELES_LAYA_URL__: string | undefined;

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

export const DEFAULT_BROWSER_PROVIDER: ProviderModelConfigSpec = {
  providerId: "openai", baseUrl: "https://api.openai.com/v1", model: "gpt-4o", apiKey: ENV_OPENAI_KEY,
};

export const PROVIDER_PRESETS: readonly ProviderModelConfigSpec[] = [
  {
    providerId: "veles-demo",
    baseUrl: "local://demo-comet",
    model: "comet-counter-v1",
  },
  DEFAULT_BROWSER_PROVIDER,
  { providerId: "lm-studio", baseUrl: "http://127.0.0.1:1234/v1", model: "local-model" },
  { providerId: "llama.cpp", baseUrl: "http://127.0.0.1:8080/v1", model: "local-model" },
  { providerId: "custom-local", baseUrl: "http://127.0.0.1:8000/v1", model: "local-model" },
  { providerId: "google", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", model: "gemini-4-hydrogen" },
  { providerId: "claude", baseUrl: "https://api.anthropic.com/v1", model: "claude-opus-5-5" },
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

export const DEFAULT_BROWSER_SERVICES: ServiceCapabilitiesConfigSpec = {
  ...DEFAULT_SERVICE_CAPABILITIES_CONFIG,
  decision: {
    ...DEFAULT_SERVICE_CAPABILITIES_CONFIG.decision,
    mode: "openai-decisions",
    endpointUrl: OPENAI_DECISIONS_ENDPOINT,
    model: "gpt-6-luna",
    apiKey: ENV_OPENAI_KEY,
  },
};
