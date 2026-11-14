# Veles

[![open in your browser](docs/images/open.svg)](https://invertedmushroom.github.io/veles/)

A browser-native agent runtime. A Web Worker owns durable workspace state —
files, conversations, approvals, and the run journal — in IndexedDB and OPFS;
the UI consumes typed projections over a `MessagePort` bridge; and remote
inference and tool execution stay ephemeral and never become the authority for
committed state.

Read the standalone [architecture walkthrough](docs/architecture.html) plus
deep-dives on the [in-browser agent harness](docs/harness.html),
[turn routing](docs/routing.html), [multimodal image flow](docs/multimodal.html),
the [preview sandbox](docs/sandbox.html), the [run journal](docs/journal.html),
the [execution gateway](docs/gateway.html), and the
[3-pane workbench](docs/workbench.html).

## Quickstart

```sh
npm install        # React runtime + Vite/TypeScript/Playwright toolchain
npm run dev        # 3-Pane Workbench UI on http://127.0.0.1:5173 (Vite dev server)
npm test           # all nine Node suites, zero-dependency type-stripped run
npm run check      # strict TypeScript typecheck (tsc -b)
npm run build      # typecheck + production bundle + PWA service worker in dist/
```

`npm test` requires a Node release supporting `--experimental-strip-types` and
runs offline without credentials or network access. Everything in the workbench
also runs offline out of the box using the built-in `veles-demo` provider and
deterministic Tier-0 routing; live providers and Decision APIs can be configured
either in the **Services** (`⚙️`) panel or via environment variables.

## Subsystems & Verification

| Area | Present | Verified by |
| :--- | :--- | :--- |
| Protocol | Types, validators, state transitions, recovery decisions, `MessagePort` wire contracts ([`src/protocol/types.ts`](src/protocol/types.ts), [`src/protocol/validation.ts`](src/protocol/validation.ts)) | [`tests/protocol.test.ts`](tests/protocol.test.ts) |
| Kernel & storage | Append-only journal, content-addressed blobs, Web Locks leases, archives, `IndexedDbJournalBackend`, `OpfsBlobBackend` ([`src/kernel/journal.ts`](src/kernel/journal.ts), [`src/kernel/blobs.ts`](src/kernel/blobs.ts), [`src/kernel/lock.ts`](src/kernel/lock.ts)) | [`tests/kernel-storage.test.ts`](tests/kernel-storage.test.ts) |
| Worker loop, sessions & inference | `AgentKernelRuntime`, 4-Layer `ContextBuilder`, `AgentManager`, `OpenAICompatibleAdapter` ([`src/kernel/runtime.ts`](src/kernel/runtime.ts), [`src/kernel/agent-session.ts`](src/kernel/agent-session.ts), [`src/inference/openai-compatible.ts`](src/inference/openai-compatible.ts)) | [`tests/kernel-runtime.test.ts`](tests/kernel-runtime.test.ts) |
| Browser tools & preview | `BrowserToolRegistry`, `RESOLVE_APPROVAL` resumption, `PreviewSandboxHost` & `compileMultiFileSrcdoc` (`SEC-INV-01`, [`src/kernel/tools.ts`](src/kernel/tools.ts), [`src/ui/preview-sandbox.ts`](src/ui/preview-sandbox.ts)) | [`tests/kernel-tools-preview.test.ts`](tests/kernel-tools-preview.test.ts) |
| Multimodal & subagents | `workspace.inspect_image` vision, `image.generate` / `image.transform` cascade, `agent.delegate` child sessions, gateway tool catalog ([`src/kernel/tools.ts`](src/kernel/tools.ts)) | [`tests/multimodal-subagent.test.ts`](tests/multimodal-subagent.test.ts) |
| In-browser agent harness | Multi-tool batching, OPFS `sha256` drift detection, `VelesHarnessRouter` specialist capsules, fenced preview KV & sensor bridge ([`src/kernel/harness-router.ts`](src/kernel/harness-router.ts), [`src/kernel/runtime.ts`](src/kernel/runtime.ts)) | [`tests/harness-slices.test.ts`](tests/harness-slices.test.ts) |
| Run control & projections | Durable journal projections, optimistic live commits, archive restore, Stop/cancellation, interrupted tool-call recovery ([`src/kernel/run-cancellation.ts`](src/kernel/run-cancellation.ts), [`src/ui/chat-timeline.ts`](src/ui/chat-timeline.ts)) | [`tests/run-control.test.ts`](tests/run-control.test.ts) |
| Ephemeral gateway | `EphemeralExecutionGateway` ([`services/gateway/server.ts`](services/gateway/server.ts)), `GatewayToolExecutionAdapter` ([`src/kernel/gateway-client.ts`](src/kernel/gateway-client.ts)) | [`tests/gateway.test.ts`](tests/gateway.test.ts) |
| Worker-to-UI bridge & 3-Pane Workbench | `KernelWorkerHost` ([`src/kernel/worker-host.ts`](src/kernel/worker-host.ts)), `bootBrowserKernelWorker` ([`src/kernel/worker-entry.ts`](src/kernel/worker-entry.ts)), `VelesAppClient` ([`src/ui/app-client.ts`](src/ui/app-client.ts)), `VelesWorkbenchApp` ([`src/ui/VelesWorkbenchApp.tsx`](src/ui/VelesWorkbenchApp.tsx)), `ChatTimeline` ([`src/ui/ChatTimeline.tsx`](src/ui/ChatTimeline.tsx)), `WorkbenchLayout` ([`src/ui/WorkbenchLayout.tsx`](src/ui/WorkbenchLayout.tsx)) | [`tests/worker-bridge-e2e.test.ts`](tests/worker-bridge-e2e.test.ts) plus [`tests/browser/workbench.spec.ts`](tests/browser/workbench.spec.ts), [`tests/browser/workbench-ui.spec.ts`](tests/browser/workbench-ui.spec.ts), [`tests/browser/boundaries.spec.ts`](tests/browser/boundaries.spec.ts) |

## Configuration & Operational Model

All environment variables are optional; Verified readers in parentheses:

- `VITE_OPENAI_API_KEY` (fallback `OPENAI_API_KEY`): pre-populates the browser
  `openai` (`gpt-4o`) chat preset and the `openai-decisions` (`gpt-6-luna`)
  Decision API preset ([`vite.config.ts`](vite.config.ts),
  [`src/ui/service-presets.ts`](src/ui/service-presets.ts)).
- `VITE_OPENROUTER_API_KEY` (fallback `OPENROUTER_API_KEY`): pre-populates the
  `openrouter` chat preset ([`vite.config.ts`](vite.config.ts),
  [`src/ui/service-presets.ts`](src/ui/service-presets.ts)).
- `VITE_TYPESAFE_API_KEY` (fallback `TYPESAFE_API_KEY` or `JEV_API_KEY`):
  pre-populates the `typesafe` (`jev-1.14.0`) System One Decision API preset,
  proxied in browsers via `/api/proxy/typesafe`
  ([`vite.config.ts`](vite.config.ts),
  [`src/ui/service-presets.ts`](src/ui/service-presets.ts)).
- `VITE_LAYA_API_URL` (fallback `LAYA_API_URL`): overrides the local Laya
  Decision API URL (default `http://127.0.0.1:11434/v1/decisions`, proxied in
  browsers via `/api/proxy/laya`, [`vite.config.ts`](vite.config.ts)).
- `VELES_GATEWAY_URL`: upstream URL for the optional `/v1` Ephemeral Execution
  Gateway proxy (default `http://127.0.0.1:4319`,
  [`vite.config.ts`](vite.config.ts)).
- `VELES_HTTPS_KEY_PATH`, `VELES_HTTPS_CERT_PATH` (fallback
  `PERUN_HTTPS_KEY_PATH`, `PERUN_HTTPS_CERT_PATH`): TLS key and certificate file
  paths required by `npm run dev:https` and `npm run preview:https` for LAN /
  mobile PWA testing ([`vite.config.ts`](vite.config.ts)).

### Runtime Service Boundaries

During `npm run dev` and `npm run preview` (`vite.config.ts`), Vite serves two
same-origin isolation endpoints alongside `/manifest.webmanifest`:

1. **`/api/sandbox/worker.js`:** serves the capability-scrubbed worker bootstrap
   for `workspace.run_script` with
   `Content-Security-Policy: default-src 'none'; script-src 'unsafe-eval'; connect-src 'none'`.
2. **`/api/proxy/fetch`:** performs DNS-validated public HTTP/HTTPS fetches for
   `web.fetch_url` (blocking loopback, RFC1918, link-local, and IPv4-mapped IPv6
   private addresses on every redirect hop) and stamps `X-Veles-Fetch-Proxy: 1`.

If static assets in `dist/` are hosted without those endpoints,
`workspace.run_script` and `web.fetch_url` fail closed with
`ERR_SCRIPT_SANDBOX_UNAVAILABLE` and `ERR_FETCH_PROXY_UNAVAILABLE` rather than
falling back to unisolated execution or direct browser fetches (verified in
[`tests/browser/boundaries.spec.ts`](tests/browser/boundaries.spec.ts)).

## Verification Commands

```sh
npm test               # 9 Node suites (63 tests: protocol, storage, runtime, tools, gateway, harness, run control)
npm run check          # strict TypeScript compiler check (tsc -b --pretty false)
npm run build          # production Vite + Workbox PWA build
npm run test:browser   # Playwright Chromium E2E suites (workbench, workbench-ui, boundaries)
```

In `tests/browser/workbench.spec.ts`, the live TypeSafe classification case skips
automatically when `VITE_TYPESAFE_API_KEY` is unconfigured; the live OpenAI SSE
case requires `VITE_OPENAI_API_KEY` (or `OPENAI_API_KEY`). All other browser
cases in `workbench.spec.ts`, `workbench-ui.spec.ts`, and `boundaries.spec.ts`
run offline against deterministic fixtures.

## Deploy to GitHub Pages

The [Pages workflow](.github/workflows/pages.yml) builds the static browser PWA
and standalone `docs/*.html` walkthroughs on pushes to `main`. In GitHub, select
**Settings → Pages → Source → GitHub Actions**.

The Pages build uses relative asset paths (`/veles/`) and defaults the browser
workbench to the offline `veles-demo` provider and `tier0-only` deterministic
routing (`VITE_DEFAULT_PROVIDER=veles-demo`, `VITE_DEFAULT_DECISION_MODE=tier0-only`)
so visitors can run the full Web Worker + IndexedDB/OPFS + sandboxed preview
loop without backend services or API keys, or configure their own OpenAI /
OpenRouter credentials in **Services (`⚙️`)**.

