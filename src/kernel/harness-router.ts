/**
 * @module SYS-KERNEL/harness-router
 * In-Browser Runtime Harness Router (Layer 2 of ContextBuilder).
 * - Tier 0: Deterministic keyword & workspace path/extension matching (0ms, offline).
 * - Tier 1: Optional bounded classifier adapter (TypeSafe Jev, OpenAI Decisions API,
 *   or Local Laya) with explicit `"undecided"` fallback on timeout or network error
 *   so Veles never depends on a second service to function.
 * Cyclomatic complexity V(G) <= 10 per function.
 */
import { awaitWithAbort } from "./abort.ts";
import type { DecisionServiceConfigSpec } from "../protocol/types.ts";
import { TYPESAFE_SYSTEM_ONE_ENDPOINT } from "../protocol/types.ts";

export interface InBrowserSpecialistProfile {
    readonly id: string;
    readonly title: string;
    readonly priority: number;
    readonly keywords: readonly string[];
    readonly pathPatterns: readonly RegExp[];
    readonly capsulePrompt: string;
    readonly routingHint?: string;
}

export type ClassifierDecision =
    | { readonly status: "matched"; readonly specialistIds: readonly string[] }
    | { readonly status: "undecided"; readonly reason: string };

export interface SpecialistRoutingResult {
    readonly source: "tier1_classifier" | "tier0_deterministic";
    readonly profiles: readonly InBrowserSpecialistProfile[];
    readonly fallbackReason?: string;
}

export type OptionalClassifierAdapter = (
    prompt: string,
    filePaths: readonly string[],
    profiles: readonly InBrowserSpecialistProfile[],
    signal: AbortSignal,
) => Promise<ClassifierDecision>;

export const DEFAULT_BROWSER_SPECIALISTS: readonly InBrowserSpecialistProfile[] = [
    {
        id: "canvas-game-builder",
        title: "HTML5 Canvas & Interactive Game Specialist",
        priority: 100,
        keywords: [
            "game",
            "mario",
            "platformer",
            "canvas game",
            "game physics",
            "collision",
            "arcade",
            "jump",
        ],
        pathPatterns: [],
        routingHint: "Playable games and game loops; ordinary apps, articles, charts, or standalone sprites do not need game mechanics.",
        capsulePrompt: [
            "[Specialist: HTML5 Canvas & Game Builder]",
            "- Organize multi-file apps/games under `preview/index.html`, `preview/style.css`, and `preview/script.js` (ensure `<link href=\"style.css\">` and `<script src=\"script.js\">` match your saved file paths).",
            "- Always reference workspace images by their exact workspace path (e.g. `<img src=\"assets/horse.png\">`) in both HTML and JS.",
            "- Deliver working gameplay for the requested genre, with visible state and usable controls; implement requested mechanics rather than placeholder gameLoop/render stubs.",
            "- Use `requestAnimationFrame` with delta-time clamping for games, attach DOM listeners safely, and persist high scores or settings via `window.veles.kv.get(k)` / `window.veles.kv.set(k, v)` (never `localStorage`).",
        ].join("\n"),
    },
    {
        id: "preview-app-builder",
        title: "Browser Preview App Specialist",
        priority: 70,
        keywords: ["app", "page", "website", "layout", "dashboard", "form", "tinder"],
        pathPatterns: [/^preview\/.*\.(html|js|css)$/i],
        routingHint: "General browser apps and interfaces; games, diagrams, and articles have more specific specialists.",
        capsulePrompt: [
            "[Specialist: Browser Preview App Builder]",
            "- Build the requested interface under preview/index.html, with matching workspace CSS/JS paths and exact assets/ paths for images.",
            "- Implement the requested interactions with clear empty/end states, keyboard-accessible controls, and responsive layout; keep unrelated game mechanics out of ordinary apps.",
            "- window.veles.kv.get(k) returns a value or null synchronously (do not use top-level await in classic <script> tags or call .then() on kv.get). Use kv.set/remove for preferences and update visible state in-place.",
        ].join("\n"),
    },
    {
        id: "sensor-bridge",
        title: "Device Sensor & Preview Capability Bridge Specialist",
        priority: 95,
        keywords: [
            "sensor",
            "sensors",
            "tilt",
            "gyro",
            "gyroscope",
            "accelerometer",
            "orientation",
        ],
        pathPatterns: [],
        capsulePrompt: [
            "[Specialist: Device Sensor & Capability Bridge]",
            "- For device tilt or pointer sensors in preview, read `window.veles.sensors.latest` (`{ tiltX, tiltY, pointerX, pointerY }`) or subscribe via `window.veles.sensors.onUpdate(cb)`.",
            "- Always include keyboard or pointer fallback controls when device orientation values are zero or inactive.",
        ].join("\n"),
    },
    {
        id: "data-script-compute",
        title: "Dataset & Script Computation Specialist",
        priority: 92,
        keywords: [
            "csv",
            "tsv",
            "dataset",
            "average",
            "mean",
            "median",
            "calculate",
            "statistics",
            "histogram",
            "histograms",
            "scatter",
            "plot",
            "bins",
            "chart",
        ],
        pathPatterns: [/^assets\/.*\.(csv|tsv)$/i],
        capsulePrompt: [
            "[Specialist: Dataset & Script Computation]",
            "- `web.fetch_url` (optional `saveToPath`) and `workspace.run_script` (`readFile(path)`, `writeFile(path, data)`, `return result`) are independent tools: use `web.fetch_url` when downloading remote content, and `workspace.run_script` when running JS data transformations or calculations in an isolated worker (no DOM/`window`).",
            "- For interactive visualizations in `preview/`, render complete working `<canvas>` 2D or `<svg>` charts directly (never leave placeholder comments or call external chart libraries without loading their `<script>` tag), avoid top-level `await` in classic `<script>` tags, and update UI/chart state in-place on control events rather than calling `location.reload()`.",
        ].join("\n"),
    },
    {
        id: "preview-debugger",
        title: "Sandboxed Preview Runtime Debugger",
        priority: 110,
        keywords: [
            "error",
            "bug",
            "fix",
            "broken",
            "uncaught",
            "typeerror",
            "referenceerror",
            "crash",
            "blank",
        ],
        pathPatterns: [],
        capsulePrompt: [
            "[Specialist: Preview Runtime Debugger]",
            "- Inspect the current OPFS file via `workspace.read_file` and apply the fix directly using `workspace.edit_file` (for targeted changes) or `workspace.write_file` in the same turn without stopping to ask for confirmation.",
            "- Verify DOM element IDs, data structures, and script load order inside `preview/`.",
            "- Remember the preview runs in an opaque origin (`origin === \"null\"` without `allow-same-origin`): never call `localStorage`, `sessionStorage`, or `indexedDB` directly, and preserve built-in `window.veles.*` host bridge calls (`kv`, `log`, `sensors`, `files`).",
        ].join("\n"),
    },
    {
        id: "svg-multimodal-designer",
        title: "Editable SVG & Vector Asset Specialist",
        priority: 90,
        keywords: ["svg", "vector", "icon", "logo"],
        pathPatterns: [/\.svg$/i],
        routingHint: "Editable vector source, SVG artwork, icons, and logos. Raster photography and image recoloring use the raster specialist.",
        capsulePrompt: [
            "[Specialist: Editable SVG & Vector Designer]",
            "- Author editable SVG XML directly with `workspace.write_file` or `workspace.edit_file` under `assets/` (or inline in `preview/index.html`); read existing SVG source before editing, and reserve `image.generate` / `image.transform` for raster requests.",
            "- Include `xmlns` and `viewBox`, clean semantic groups (`<g>`), a cohesive palette, and planned geometry/text padding so strokes and labels never clip at small sizes.",
            "- Provide `<title>`/`<desc>` for informative SVGs (or `aria-hidden=\"true\"` for decorative inline icons), keep IDs unique across inline SVGs, and verify all `href`/`url(#id)` references resolve.",
        ].join("\n"),
    },
    {
        id: "raster-image-designer",
        title: "Raster Image & Multimodal Asset Specialist",
        priority: 88,
        keywords: [
            "image",
            "images",
            "picture",
            "photo",
            "photograph",
            "sprite",
            "lora",
            "flux",
            "transform",
            "describe",
        ],
        pathPatterns: [/^assets\/.*\.(png|jpe?g|webp|gif)$/i],
        routingHint: "Generated raster illustrations, photos, sprites, and edits of existing images; editable SVG and technical diagrams use their own specialists.",
        capsulePrompt: [
            "[Specialist: Raster Image & Multimodal Asset Designer]",
            "- Store generated or transformed media under `assets/` using `image.generate` or `image.transform`. For an edit, preserve the source subject/composition and change only the requested attributes.",
            "- Use `workspace.inspect_image` to load existing workspace images into vision context before describing or transforming them.",
            "- Read tool diagnostics: raster success, GPT SVG fallback, and deterministic placeholder output are distinct outcomes—report any fallback honestly.",
        ].join("\n"),
    },
    {
        id: "diagram-design",
        title: "Technical Topology & Diagram Specialist",
        priority: 104,
        keywords: ["diagram-design", "diagram", "diagrams", "flowchart", "state machine", "sequence diagram", "architecture diagram", "data layout", "wire frame", "wire-frame"],
        pathPatterns: [],
        routingHint: "Technical topology, protocol, state, and data-layout diagrams. Ordinary SVG icons and scientific plot curves do not need orthogonal connector rules.",
        capsulePrompt: [
            "[Specialist: Technical Diagram Design]",
            "- Ground nodes, states, and edges in supplied evidence (distinguishing observed vs. proposed behavior). Use a clear left-to-right or top-to-bottom topology and split dense flows into overview + detail (at most ~9 nodes per figure).",
            "- Plan node boxes on a 4px grid with text padding before writing SVG. Keep foreground boxes disjoint and route topology connectors orthogonally (horizontal/vertical with rounded elbows and >=12px spacing).",
            "- Layer background zones -> connectors -> label masks -> foreground nodes/labels. Use `viewBox`, `<title>`/`<desc>`, system fonts, line styles/labels alongside color, and save via `workspace.write_file` or `workspace.edit_file`.",
        ].join("\n"),
    },
    {
        id: "concept-article",
        title: "Concept Article & Interactive Explanation Specialist",
        priority: 106,
        keywords: ["concept-article", "article", "explainer", "tutorial", "essay", "technical article", "concept article", "scrolling article", "html article", "visual essay", "interactive explanation", "interactive explainer", "explorable explanation", "long-form explanation"],
        pathPatterns: [],
        routingHint: "A scrolling technical HTML article or visual essay combining prose and explanatory figures; ordinary app pages, slides, and short prose edits do not need this capsule.",
        capsulePrompt: [
            "[Specialist: Concept Article & Interactive Explanation]",
            "- Build a self-contained scrolling semantic HTML article in `preview/index.html` that opens with a concrete question/prediction, interleaves prose with equations/figures/interactive experiments, reuses consistent notation, and covers a meaningful failure or boundary case.",
            "- Ground claims in supplied material or verified `web.*` / `workspace.run_script` outputs; distinguish measured, simulated, and theoretical results, and never invent citations or unrun experiments.",
            "- Keep essential prose, caveats, and default figures visible in static HTML using system fonts, inline SVG (with labeled axes/units), and native MathML.",
            "- Make interactive controls update both the figure and nearby explanation in-place while preserving keyboard access, reduced-motion support, and narrow-screen/print readability.",
        ].join("\n"),
    },
];

function matchDeterministicTier0(
    prompt: string,
    filePaths: readonly string[],
    profiles: readonly InBrowserSpecialistProfile[],
): InBrowserSpecialistProfile[] {
    const lowerPrompt = prompt.toLowerCase();
    const matched: InBrowserSpecialistProfile[] = [];

    for (const profile of profiles) {
        const keywordHit = profile.keywords.some((kw) => {
            const escaped = kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
            return new RegExp(`(?<!\\w)${escaped}(?!\\w)`, "i").test(lowerPrompt);
        });
        const pathHit =
            !keywordHit &&
            profile.pathPatterns.some((re) =>
                filePaths.some((p) => re.test(p) && lowerPrompt.includes(p.toLowerCase())),
            );
        if (keywordHit || pathHit) {
            matched.push(profile);
        }
    }

    return matched.sort((a, b) => b.priority - a.priority).slice(0, 2);
}

const SECRET_TOKEN_RE =
    /(sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{20,}|apikey_[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|Bearer\s+[A-Za-z0-9_.~+/-]+=*)/gi;

export function redactClassifierSecrets(text: string): string {
    return (text || "").replace(SECRET_TOKEN_RE, "<REDACTED>").slice(-3000);
}

function buildDecisionRequestPayload(
    decision: DecisionServiceConfigSpec,
    contextText: string,
    profiles: readonly InBrowserSpecialistProfile[],
): Record<string, unknown> {
    if (decision.mode === "typesafe") {
        const questions: Record<string, { type: "noul"; instructions: string }> = {};
        for (const p of profiles) {
            questions[p.id] = {
                type: "noul",
                instructions: `Does this task need ${p.title} expertise? ${p.routingHint ?? ""} Relevant topics: ${p.keywords.join(", ")}.`,
            };
        }
        return {
            model: decision.model || "jev-1.14.0",
            state: contextText,
            questions,
        };
    }
    return {
        model:
            decision.model ||
            (decision.mode === "local-laya" ? "laya-latest" : "gpt-6-luna"),
        input: contextText,
        questions: profiles.map((p) => ({
            type: "predicate",
            name: p.id,
            instructions: `Does this task need ${p.title} expertise? ${p.routingHint ?? ""} Relevant topics: ${p.keywords.join(", ")}.`,
        })),
    };
}

function decisionRecord(raw: unknown): Record<string, unknown> | null {
    return typeof raw === "object" && raw !== null && !Array.isArray(raw)
        ? raw as Record<string, unknown>
        : null;
}

function addDecisionProbability(scores: Record<string, number>, name: string, value: unknown): void {
    const probability = typeof value === "string" && value.trim() ? Number(value) : value;
    if (typeof probability === "number" && Number.isFinite(probability) && probability >= 0 && probability <= 1) {
        scores[name] = probability;
    }
}

function parseDecisionScoresMap(raw: unknown): Record<string, number> {
    const obj = decisionRecord(raw);
    if (!obj) return {};
    const scores: Record<string, number> = {};
    for (const [name, value] of Object.entries(decisionRecord(obj.probabilities) ?? {})) {
        addDecisionProbability(scores, name, value);
    }
    if (Array.isArray(obj.answers)) {
        for (const answer of obj.answers) {
            const entry = decisionRecord(answer);
            if (typeof entry?.name === "string") {
                addDecisionProbability(scores, entry.name, entry.probability);
            }
        }
        return scores;
    }
    for (const [name, answer] of Object.entries(decisionRecord(obj.answers) ?? {})) {
        const entry = decisionRecord(answer);
        addDecisionProbability(scores, name, entry?.noul);
    }
    return scores;
}

function extractMatchedSpecialistIds(
    raw: unknown,
    profiles: readonly InBrowserSpecialistProfile[],
    threshold: number,
): string[] {
    const scores = parseDecisionScoresMap(raw);
    return profiles
        .filter((p) => (scores[p.id] ?? 0) >= threshold)
        .sort((a, b) => (scores[b.id] ?? 0) - (scores[a.id] ?? 0))
        .slice(0, 2)
        .map((p) => p.id);
}

function resolveDecisionRequestCredentials(
    decision: DecisionServiceConfigSpec | undefined,
    chatApiKey: string | undefined,
): { readonly ok: true; readonly apiKey: string } | { readonly ok: false; readonly reason: string } {
    if (!decision || decision.mode === "tier0-only") {
        return { ok: false, reason: "tier0_only" };
    }
    const apiKey =
        decision.apiKey?.trim() ||
        (decision.mode === "openai-decisions" ? chatApiKey?.trim() ?? "" : "");
    if (decision.mode !== "local-laya" && !apiKey) {
        return { ok: false, reason: "missing_api_key" };
    }
    return { ok: true, apiKey };
}

function resolveDecisionEndpointUrl(rawUrl: string): string {
    const raw = rawUrl.trim();
    const trimmed = raw === "https://api.typesafe.ai/v1/jev/decide"
        ? TYPESAFE_SYSTEM_ONE_ENDPOINT : raw;
    const origin = (globalThis as { location?: { origin?: string } }).location?.origin;
    if (
        typeof origin === "string" &&
        origin.startsWith("http") &&
        trimmed === TYPESAFE_SYSTEM_ONE_ENDPOINT
    ) {
        return `${origin}/api/proxy/typesafe`;
    }
    return trimmed;
}

/**
 * Creates a live Tier-1 System-1 Decision API adapter supporting TypeSafe (`jev-1.14.0`),
 * OpenAI Decisions API (`gpt-6-luna`), and Local Laya (`laya-latest`), with automatic
 * `"undecided"` fallback to Tier-0 deterministic rules when `mode === "tier0-only"`,
 * missing credentials, low confidence, or network/timeout errors.
 */
export function createDecisionClassifierAdapter(
    getConfig: () =>
        | {
              readonly decision?: DecisionServiceConfigSpec;
              readonly chatApiKey?: string;
          }
        | undefined,
): OptionalClassifierAdapter {
    const unreachableEndpoints = new Set<string>();
    return async (prompt, filePaths, profiles, signal) => {
        const cfg = getConfig();
        const creds = resolveDecisionRequestCredentials(cfg?.decision, cfg?.chatApiKey);
        if (!creds.ok || !cfg?.decision) {
            return { status: "undecided", reason: creds.ok ? "tier0_only" : creds.reason };
        }
        const decision = cfg.decision;
        const targetUrl = resolveDecisionEndpointUrl(decision.endpointUrl);
        if (
            typeof globalThis.fetch !== "function" ||
            !targetUrl ||
            unreachableEndpoints.has(targetUrl)
        ) {
            return { status: "undecided", reason: "endpoint_unavailable" };
        }

        const contextText = redactClassifierSecrets(
            `${prompt}\n[Workspace files: ${filePaths.slice(0, 20).join(", ")}]`,
        );
        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            ...(creds.apiKey ? { Authorization: `Bearer ${creds.apiKey}` } : {}),
        };
        try {
            const resp = await globalThis.fetch(targetUrl, {
                method: "POST",
                headers,
                body: JSON.stringify(
                    buildDecisionRequestPayload(decision, contextText, profiles),
                ),
                signal,
            });
            if (!resp.ok) {
                unreachableEndpoints.add(targetUrl);
                return { status: "undecided", reason: `http_${resp.status}` };
            }
            const matchedIds = extractMatchedSpecialistIds(
                await resp.json(),
                profiles,
                decision.threshold ?? 0.75,
            );
            return matchedIds.length > 0
                ? { status: "matched", specialistIds: matchedIds }
                : { status: "undecided", reason: "below_threshold" };
        } catch {
            if (!signal.aborted) unreachableEndpoints.add(targetUrl);
            return { status: "undecided", reason: "network_error" };
        }
    };
}

export class VelesHarnessRouter {
    private readonly profiles: readonly InBrowserSpecialistProfile[];
    private readonly classifier?: OptionalClassifierAdapter;
    private readonly timeoutMsOrGetter: number | (() => number);

    constructor(
        profiles: readonly InBrowserSpecialistProfile[] = DEFAULT_BROWSER_SPECIALISTS,
        classifier?: OptionalClassifierAdapter,
        timeoutMs: number | (() => number) = 800,
    ) {
        this.profiles = profiles;
        this.classifier = classifier;
        this.timeoutMsOrGetter = timeoutMs;
    }

    private resolveTimeoutMs(): number {
        const raw =
            typeof this.timeoutMsOrGetter === "function"
                ? this.timeoutMsOrGetter()
                : this.timeoutMsOrGetter;
        return Number.isFinite(raw) && raw > 0 ? raw : 800;
    }

    public selectTier0Capsules(
        prompt: string,
        filePaths: readonly string[],
    ): InBrowserSpecialistProfile[] {
        return matchDeterministicTier0(prompt, filePaths, this.profiles);
    }

    private async classifyWithinDeadline(
        prompt: string,
        filePaths: readonly string[],
        signal?: AbortSignal,
    ): Promise<ClassifierDecision> {
        signal?.throwIfAborted();
        const classifier = this.classifier;
        if (!classifier) return { status: "undecided", reason: "not_configured" };
        const ctrl = new AbortController();
        const abort = () => ctrl.abort(signal?.reason);
        signal?.addEventListener("abort", abort, { once: true });
        let timer: ReturnType<typeof setTimeout> | undefined;
        const deadline = new Promise<ClassifierDecision>((resolve) => {
            timer = setTimeout(() => {
                resolve({ status: "undecided", reason: "timeout" });
                ctrl.abort();
            }, this.resolveTimeoutMs());
        });
        try {
            return await Promise.race([
                awaitWithAbort(classifier(prompt, filePaths, this.profiles, ctrl.signal), ctrl.signal),
                deadline,
            ]);
        } catch {
            signal?.throwIfAborted();
            return { status: "undecided", reason: "classifier_error" };
        } finally {
            clearTimeout(timer);
            signal?.removeEventListener("abort", abort);
        }
    }

    public async routeWithOptionalClassifier(
        prompt: string,
        filePaths: readonly string[],
        signal?: AbortSignal,
    ): Promise<SpecialistRoutingResult> {
        const res = await this.classifyWithinDeadline(prompt, filePaths, signal);
        if (res.status === "matched") {
            const picked = this.profiles.filter((p) => res.specialistIds.includes(p.id));
            if (picked.length > 0) {
                return { source: "tier1_classifier", profiles: picked.slice(0, 2) };
            }
        }
        return {
            source: "tier0_deterministic",
            profiles: this.selectTier0Capsules(prompt, filePaths),
            fallbackReason: res.status === "undecided" ? res.reason : "unknown_specialist",
        };
    }
}
