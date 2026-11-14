import { existsSync, readFileSync } from "node:fs";
import { lookup } from "node:dns/promises";
import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import {
  buildIsolatedWorkerBootstrapSource,
  isBlockedPrivateHost,
  PUBLIC_FETCH_PROXY_HEADER,
  PUBLIC_FETCH_PROXY_VERSION,
} from "./src/kernel/tools.ts";

const VELES_PWA_MANIFEST = {
  id: ".",
  name: "Veles — Browser-Owned Agent Workbench",
  short_name: "Veles",
  description:
    "Browser-owned durable agent runtime and isolated preview workbench",
  theme_color: "#101c25",
  background_color: "#101c25",
  display: "standalone" as const,
  orientation: "any" as const,
  scope: "./",
  start_url: "./",
  icons: [
    {
      src: "pwa-192x192.png",
      sizes: "192x192",
      type: "image/png",
      purpose: "any",
    },
    {
      src: "pwa-512x512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "any",
    },
    {
      src: "pwa-maskable-512x512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "maskable",
    },
  ],
};

function readDeviceHttpsConfig(
  allEnv: Record<string, string>,
): { key: Buffer; cert: Buffer } {
  const keyPath =
    process.env.VELES_HTTPS_KEY_PATH ||
    allEnv.VELES_HTTPS_KEY_PATH ||
    process.env.PERUN_HTTPS_KEY_PATH ||
    allEnv.PERUN_HTTPS_KEY_PATH;
  const certificatePath =
    process.env.VELES_HTTPS_CERT_PATH ||
    allEnv.VELES_HTTPS_CERT_PATH ||
    process.env.PERUN_HTTPS_CERT_PATH ||
    allEnv.PERUN_HTTPS_CERT_PATH;
  if (!keyPath || !certificatePath || !existsSync(keyPath) || !existsSync(certificatePath)) {
    throw new Error(
      "device HTTPS mode requires VELES_HTTPS_KEY_PATH and VELES_HTTPS_CERT_PATH (or PERUN_HTTPS_KEY_PATH / PERUN_HTTPS_CERT_PATH) to point to trusted TLS files",
    );
  }
  return {
    key: readFileSync(keyPath),
    cert: readFileSync(certificatePath),
  };
}

async function assertPublicProxyDestination(parsed: URL): Promise<void> {
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error("ERR_FETCH_PRIVATE_HOST_BLOCKED");
  }
  const rawHost = parsed.hostname;
  if (isBlockedPrivateHost(rawHost)) {
    throw new Error("ERR_FETCH_PRIVATE_HOST_BLOCKED");
  }
  const lookupHost = rawHost.replace(/^\[|\]$/g, "");
  let resolved: Array<{ address: string }>;
  try {
    resolved = await lookup(lookupHost, { all: true, verbatim: true });
  } catch {
    throw new Error("ERR_FETCH_DNS_RESOLUTION_FAILED");
  }
  if (
    resolved.length === 0 ||
    resolved.some((entry) => isBlockedPrivateHost(entry.address))
  ) {
    throw new Error("ERR_FETCH_PRIVATE_HOST_BLOCKED");
  }
}

async function fetchPublicUpstreamWithManualRedirects(
  initialTarget: string,
): Promise<Response> {
  let currentUrl = new URL(initialTarget);
  for (let hop = 0; hop <= 5; hop++) {
    await assertPublicProxyDestination(currentUrl);
    const upstream = await fetch(currentUrl.toString(), {
      method: "GET",
      redirect: "manual",
    });
    if (upstream.status >= 300 && upstream.status < 400) {
      const location = upstream.headers.get("location");
      if (!location) return upstream;
      currentUrl = new URL(location, currentUrl);
      continue;
    }
    return upstream;
  }
  throw new Error("ERR_FETCH_TOO_MANY_REDIRECTS");
}

function registerVelesMiddlewares(middlewares: {
  use: (
    route: string,
    fn: (
      req: { url?: string },
      res: {
        statusCode: number;
        setHeader(k: string, v: string): void;
        end(body?: string | Buffer): void;
      },
    ) => void | Promise<void>,
  ) => void;
}): void {
  middlewares.use("/manifest.webmanifest", (_req, res) => {
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/manifest+json; charset=utf-8");
    res.end(JSON.stringify(VELES_PWA_MANIFEST, null, 2));
  });
  middlewares.use("/api/sandbox/worker.js", (_req, res) => {
    res.statusCode = 200;
    res.setHeader("Content-Type", "application/javascript; charset=utf-8");
    res.setHeader(
      "Content-Security-Policy",
      "default-src 'none'; script-src 'unsafe-eval'; connect-src 'none'",
    );
    res.end(buildIsolatedWorkerBootstrapSource());
  });
  middlewares.use("/api/proxy/fetch", async (req, res) => {
    res.setHeader(PUBLIC_FETCH_PROXY_HEADER, PUBLIC_FETCH_PROXY_VERSION);
    try {
      const reqUrl = new URL(req.url ?? "", "http://127.0.0.1");
      const targetRaw = reqUrl.searchParams.get("url") ?? "";
      const upstream = await fetchPublicUpstreamWithManualRedirects(targetRaw);
      res.statusCode = upstream.status;
      const ct = upstream.headers.get("content-type");
      if (ct) res.setHeader("Content-Type", ct);
      const buf = Buffer.from(await upstream.arrayBuffer());
      res.end(buf);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Bad Gateway";
      res.statusCode = msg === "ERR_FETCH_PRIVATE_HOST_BLOCKED" ? 403 : 502;
      res.end(msg);
    }
  });
}

function velesCorsFetchProxyPlugin(): Plugin {
  return {
    name: "veles-cors-fetch-proxy",
    configureServer(server) {
      registerVelesMiddlewares(server.middlewares);
    },
    configurePreviewServer(server) {
      registerVelesMiddlewares(server.middlewares);
    },
  };
}

export default defineConfig(({ command, mode }) => {
  const allEnv = loadEnv(mode, ".", "");
  const https =
    command === "serve" && mode === "device-https"
      ? readDeviceHttpsConfig(allEnv)
      : undefined;
  const openAiKey =
    allEnv.VITE_OPENAI_API_KEY ||
    allEnv.OPENAI_API_KEY ||
    Object.entries(allEnv).find(
      ([k, v]) =>
        (k.toUpperCase().includes("OPENAI") ||
          k.toUpperCase().includes("API_KEY")) &&
        typeof v === "string" &&
        v.trim().length > 10,
    )?.[1] ||
    "";
  const serverProxy = {
    "/v1": {
      target: allEnv.VELES_GATEWAY_URL || "http://127.0.0.1:4319",
    },
    "/api/proxy/typesafe": {
      target: "https://api.typesafe.ai",
      changeOrigin: true,
      rewrite: () => "/v1/systemone",
    },
    "/api/proxy/laya": {
      target: "http://127.0.0.1:11434",
      changeOrigin: true,
      rewrite: () => "/v1/decisions",
    },
  };
  return {
    base: "./",
    envPrefix: ["VITE_", "OPENAI_", "OPENROUTER_", "VELES_", "TYPESAFE_", "JEV_", "LAYA_"],
    define: {
      __VELES_OPENAI_KEY__: JSON.stringify(openAiKey.trim()),
      __VELES_OPENROUTER_KEY__: JSON.stringify(
        (allEnv.VITE_OPENROUTER_API_KEY || allEnv.OPENROUTER_API_KEY || "").trim(),
      ),
      __VELES_TYPESAFE_KEY__: JSON.stringify(
        (
          allEnv.VITE_TYPESAFE_API_KEY ||
          allEnv.TYPESAFE_API_KEY ||
          allEnv.JEV_API_KEY ||
          ""
        ).trim(),
      ),
      __VELES_LAYA_URL__: JSON.stringify(
        (
          allEnv.VITE_LAYA_API_URL ||
          allEnv.LAYA_API_URL ||
          "http://127.0.0.1:11434/v1/decisions"
        ).trim(),
      ),
    },
    plugins: [
      react(),
      velesCorsFetchProxyPlugin(),
      VitePWA({
        injectRegister: false,
        registerType: "prompt",
        includeAssets: [
          "pwa-icon.svg",
          "pwa-192x192.png",
          "pwa-512x512.png",
          "pwa-maskable-512x512.png",
        ],
        manifest: VELES_PWA_MANIFEST,
        workbox: {
          cleanupOutdatedCaches: true,
          clientsClaim: true,
          globPatterns: ["**/*.{css,html,js,png,svg,webmanifest}"],
          navigateFallback: "index.html",
          navigateFallbackDenylist: [/^\/api\//, /^\/v1\//],
        },
      }),
    ],
    server: {
      port: 5173,
      strictPort: true,
      https,
      proxy: serverProxy,
    },
    preview: {
      port: 4173,
      strictPort: true,
      https,
      proxy: serverProxy,
    },
  };
});
