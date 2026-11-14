import { expect, test, type Page } from "@playwright/test";
import type { BrowserToolRegistry } from "../../src/kernel/tools.ts";
import type { PreviewSandboxHost, SandboxEnvironmentLike } from "../../src/ui/preview-sandbox.ts";

type ReviewWindow = Window & {
  reviewTools: BrowserToolRegistry;
  reviewHost: PreviewSandboxHost;
};

async function openToolFixture(page: Page): Promise<void> {
  await page.route("http://127.0.0.1:4173/", (route) => route.fulfill({
    contentType: "text/html", body: '<!doctype html><body><div id="preview"></div></body>',
  }));
  await page.goto("/");
  await page.evaluate(async () => {
    const { ContentAddressedBlobStore, InMemoryBlobBackend } = await import(location.origin + "/src/kernel/blobs.ts");
    const { InMemoryJournalBackend, WorkspaceKernelStore } = await import(location.origin + "/src/kernel/journal.ts");
    const { InMemoryLockProvider } = await import(location.origin + "/src/kernel/lock.ts");
    const { BrowserToolRegistry } = await import(location.origin + "/src/kernel/tools.ts");
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease("boundary-fixture", "test");
    const store = new WorkspaceKernelStore(new InMemoryJournalBackend(),
      new ContentAddressedBlobStore(new InMemoryBlobBackend()), lease, locks);
    await store.initWorkspace("Boundary Fixture");
    await store.writeWorkspaceFile("notes/marker.txt", "WORKSPACE_MARKER");
    (window as unknown as ReviewWindow).reviewTools = new BrowserToolRegistry(store);
  });
}

async function executeBrowserTool(page: Page, name: string, args: Record<string, unknown>) {
  return page.evaluate(async ({ name, args }) => {
    try {
      const result = await (window as unknown as ReviewWindow).reviewTools.executeTool({
        runId: "boundary-run", operationId: "boundary-op", toolName: name,
        effectClass: name === "web.fetch_url" ? "pure_read" : "idempotent_write", args,
      });
      return { outputText: result.outputText, error: null };
    } catch (error) {
      return { outputText: null, error: String(error) };
    }
  }, { name, args });
}

const maskedImport = 'const regex = /"/; return (await import("http://127.0.0.1:4173/review-module?payload="+encodeURIComponent(readFile("notes/marker.txt")))).proof; //"';

test("executes local calculations while CSP blocks a scanner-obfuscated module request", async ({ page }) => {
  let requests = 0;
  await page.context().route("**/review-module?**", (route) => {
    requests += 1;
    return route.fulfill({ contentType: "application/javascript", body: 'export const proof="leaked";' });
  });
  await openToolFixture(page);
  const calculation = await executeBrowserTool(page, "workspace.run_script", {
    code: 'writeFile("notes/result.txt", "42"); return { sum: 20 + 22, fetch: typeof fetch, indexedDB: typeof indexedDB };',
  });
  expect(calculation.error).toBeNull();
  expect(JSON.parse(calculation.outputText!)).toEqual({ sum: 42, fetch: "undefined", indexedDB: "undefined" });
  const blocked = await executeBrowserTool(page, "workspace.run_script", { code: maskedImport });
  expect(blocked.error).toMatch(/Failed to fetch dynamically imported module|ERR_SCRIPT_SANDBOX/);
  expect(requests).toBe(0);
});

test("rejects an unavailable protected worker without creating a blob fallback", async ({ page }) => {
  let requests = 0;
  const workers: string[] = [];
  page.on("worker", (worker) => workers.push(worker.url()));
  await page.context().route("**/api/sandbox/worker.js", (route) => route.fulfill({ status: 404, body: "missing" }));
  await page.context().route("**/review-module?**", (route) => {
    requests += 1;
    return route.fulfill({ contentType: "application/javascript", body: 'export const proof="leaked";' });
  });
  await openToolFixture(page);
  const result = await executeBrowserTool(page, "workspace.run_script", { code: maskedImport });
  expect(result.error).toContain("ERR_SCRIPT_SANDBOX_UNAVAILABLE");
  expect(requests).toBe(0);
  expect(workers.some((url) => url.startsWith("blob:"))).toBe(false);
});

test("requires a validating fetch proxy and distinguishes missing service from upstream errors", async ({ page }) => {
  let directRequests = 0;
  await page.context().route("http://public-review.test/**", (route) => {
    directRequests += 1;
    return route.fulfill({ body: "PRIVATE_MARKER", headers: { "Access-Control-Allow-Origin": "*" } });
  });
  await openToolFixture(page);
  const proxyRoute = "**/api/proxy/fetch?**";
  for (const status of [404, 200]) {
    await page.context().route(proxyRoute, (route) => route.fulfill({ status, body: "unvalidated response" }));
    const result = await executeBrowserTool(page, "web.fetch_url", { url: "http://public-review.test/private" });
    expect(result.error).toContain("ERR_FETCH_PROXY_UNAVAILABLE");
    await page.context().unroute(proxyRoute);
  }
  await page.context().route(proxyRoute, (route) => route.fulfill({
    status: 404, body: "upstream missing", headers: { "X-Veles-Fetch-Proxy": "1" },
  }));
  const missing = await executeBrowserTool(page, "web.fetch_url", { url: "http://public-review.test/missing" });
  expect(missing.error).toContain("ERR_FETCH_HTTP_404");
  await page.context().unroute(proxyRoute);
  await page.context().route(proxyRoute, (route) => route.fulfill({
    status: 200, body: "x,y\n1,2", headers: { "X-Veles-Fetch-Proxy": "1", "Content-Type": "text/csv" },
  }));
  const valid = await executeBrowserTool(page, "web.fetch_url", { url: "http://public-review.test/data.csv" });
  expect(valid).toEqual({ outputText: "x,y\n1,2", error: null });
  expect(directRequests).toBe(0);
  await page.context().unroute(proxyRoute);
  const denied = await page.request.get("/api/proxy/fetch?url=http%3A%2F%2F127.0.0.1%2Fprivate");
  expect(denied.status()).toBe(403);
  expect(denied.headers()["x-veles-fetch-proxy"]).toBe("1");
});

async function mountKvFixture(page: Page, script: string, revision: string): Promise<void> {
  await page.evaluate(async ({ script, revision }) => {
    const { PreviewSandboxHost } = await import(location.origin + "/src/ui/preview-sandbox.ts");
    const view = window as unknown as ReviewWindow;
    if (!view.reviewHost) {
      const container = document.getElementById("preview")!;
      const environment: SandboxEnvironmentLike = {
        createIframe() {
          const frame = document.createElement("iframe");
          container.replaceChildren(frame);
          return {
            get sandbox() { return frame.getAttribute("sandbox") ?? ""; },
            set sandbox(value: string) { frame.setAttribute("sandbox", value); },
            get srcdoc() { return frame.srcdoc; },
            set srcdoc(value: string) { frame.srcdoc = value; },
            get contentWindow() { return frame.contentWindow; },
            set onLoad(handler: (() => void) | null | undefined) { frame.onload = () => handler?.(); },
            remove() { frame.remove(); },
          };
        },
        addMessageListener(handler) {
          const listener = (event: MessageEvent) => handler(event);
          window.addEventListener("message", listener);
          return () => window.removeEventListener("message", listener);
        },
      };
      view.reviewHost = new PreviewSandboxHost(environment);
    }
    view.reviewHost.mountPreview({
      workspaceId: "kv-fixture", entryPath: "preview/index.html", workspaceSeq: 1, files: {},
      entryHtml: `<html><head><script>${script}; window.addEventListener('message', function(e) { if(e.data.channel === 'veles:host-init') window.reviewInitReceived = true; });</script></head><body>${revision}</body></html>`,
    });
  }, { script, revision });
  await expect(page.frameLocator("iframe").locator("body")).toHaveText(revision);
  const frame = page.frames().find((candidate) => candidate.parentFrame() === page.mainFrame())!;
  await frame.waitForFunction(() => (window as Window & { reviewInitReceived?: boolean }).reviewInitReceived);
}

test("retains startup KV writes and deletions through delayed host initialization and reload", async ({ page }) => {
  await openToolFixture(page);
  await mountKvFixture(page, 'window.veles.kv.set("deleted", "OLD_VALUE")', "seed");
  await expect.poll(() => page.evaluate(() => (window as unknown as ReviewWindow).reviewHost.getKvSnapshot())).toEqual({ deleted: "OLD_VALUE" });
  await mountKvFixture(page, 'window.veles.kv.set("startup", "BOOT_VALUE"); window.veles.kv.remove("deleted")', "updated");
  const frame = page.frames().find((candidate) => candidate.parentFrame() === page.mainFrame())!;
  const values = () => frame.evaluate(() => {
    const kv = (window as unknown as { veles: { kv: { get(key: string): unknown } } }).veles.kv;
    return { startup: kv.get("startup"), deleted: kv.get("deleted") };
  });
  expect(await values()).toEqual({ startup: "BOOT_VALUE", deleted: null });
  await expect.poll(() => page.evaluate(() => (window as unknown as ReviewWindow).reviewHost.getKvSnapshot())).toEqual({ startup: "BOOT_VALUE" });
  await Promise.all([
    page.waitForEvent("framenavigated", (candidate) => candidate === frame),
    frame.evaluate(() => location.reload()),
  ]);
  await frame.waitForFunction(() => (window as Window & { reviewInitReceived?: boolean }).reviewInitReceived);
  expect(await values()).toEqual({ startup: "BOOT_VALUE", deleted: null });
});
