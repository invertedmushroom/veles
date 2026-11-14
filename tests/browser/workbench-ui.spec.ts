import { expect, test } from "@playwright/test";
import { configureOfflineRun, openWorkspacePanel } from "./workbench-helpers.ts";

test("shows submitted text immediately, stops a waiting provider, and permits the next run", async ({ page, context }) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  await context.route("https://api.openai.com/v1/chat/completions", async (route) => {
    calls += 1;
    const first = calls === 1;
    if (first) await held;
    await route.fulfill({
      contentType: "text/event-stream",
      body: `data: ${JSON.stringify({ choices: [{ delta: { content: first ? "Late response" : "Second finished" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`,
    }).catch(() => {});
  });
  await page.goto("/");
  await configureOfflineRun(page, "openai");
  await page.getByLabel("Message Veles").fill("Wait for my stop");
  await page.getByRole("button", { name: "Run Prompt" }).click();
  await expect(page.locator(".msg-card.user")).toContainText("Wait for my stop", { timeout: 1000 });
  await expect.poll(() => calls).toBe(1);
  await expect(page.getByTestId("harness-live-pipeline")).toBeVisible();
  await expect(page.getByTestId("harness-router-pill")).toContainText("🛡️ Tier-0 Gate");
  await expect(page.getByRole("button", { name: "Stop", exact: true })).toBeVisible();
  await page.getByLabel("Message Veles").fill("Second message");
  await page.getByLabel("Message Veles").press("Enter");
  expect(calls).toBe(1);
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page.locator(".chat-pane .pane-header")).toContainText("State: cancelled");
  await expect(page.locator(".run-outcome")).toContainText("Stopped");
  await page.getByRole("button", { name: "Run Prompt" }).click();
  await expect(page.locator(".msg-card.assistant")).toContainText("Second finished");
  release();
  await expect(page.locator(".msg-card.user")).toHaveCount(2);
  await expect(page.locator(".chat-scroll")).not.toContainText("Late response");
  await expect(page.getByRole("button", { name: "Run Prompt" })).toBeDisabled();
});

test("Stop at approval keeps committed files and prevents the approved effect", async ({ page }) => {
  await page.goto("/");
  await configureOfflineRun(page);
  await page.getByRole("button", { name: "Run Prompt" }).click();
  await expect(page.getByRole("region", { name: "Tool Approval Gate" })).toBeVisible();
  await expect(page.getByTestId("harness-specialist-pill")).toContainText("preview-app-builder");
  await expect(page.locator(".tool-target-hint").last()).toHaveText("obsolete.txt");
  await page.getByRole("button", { name: "Stop", exact: true }).click();
  await expect(page.getByRole("region", { name: "Tool Approval Gate" })).toBeHidden();
  await expect(page.locator(".chat-pane .pane-header")).toContainText("State: cancelled");
  await expect(page.locator(".tool-activity summary").last()).toContainText("Stopped");
  await page.getByRole("button", { name: "Browse files" }).click();
  await expect(page.getByRole("dialog", { name: "Workspace files" })).toBeVisible();
  await expect(page.locator(".item-row", { hasText: "obsolete.txt" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Browse files" })).toBeFocused();
});

test("mobile browsers, exclusive inspector, and fullscreen preserve the preview", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await configureOfflineRun(page);
  await openWorkspacePanel(page, "project");
  await expect(page.getByRole("dialog", { name: "Project archive" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Browse files", exact: true })).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByRole("banner").getByRole("button", { name: /Browse journal|Import|Export/ })).toHaveCount(0);
  await page.getByRole("button", { name: "Browse files", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "Run Prompt" }).click();
  await page.getByRole("button", { name: /Approve Execution/ }).click();
  await expect(page.locator(".chat-pane .pane-header")).toContainText("State: completed");
  const frame = page.frameLocator(".preview-frame-container iframe");
  await page.getByRole("button", { name: "Fullscreen inspector", exact: true }).click();
  await frame.locator("#inc-btn").click();
  await expect(frame.locator("#count-val")).toHaveText("43");
  const iframe = page.locator(".preview-frame-container iframe");
  const expanded = await iframe.boundingBox();
  expect(expanded!.height).toBeGreaterThan(600);
  await page.getByRole("button", { name: /^Console/ }).click();
  await expect(page.locator(".console-box")).toBeVisible();
  expect((await iframe.boundingBox())!.height).toBeLessThan(expanded!.height);
  await page.getByRole("button", { name: /^Console/ }).click();
  await expect(frame.locator("#count-val")).toHaveText("43");
  await page.getByRole("button", { name: "Browse files" }).click();
  await page.locator(".item-row", { hasText: "index.html" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.locator(".inspector-pane")).toContainText("File: index.html");
  await openWorkspacePanel(page, "journal");
  await page.locator(".item-row", { hasText: "RunCreated" }).click();
  await expect(page.locator(".inspector-pane")).toContainText("Journal Record #");
  await expect(page.locator(".inspector-pane")).not.toContainText("File: index.html");
  await page.getByRole("button", { name: /Live Preview/ }).click();
  await expect(frame.locator("#count-val")).toHaveText("43");
  await openWorkspacePanel(page, "journal");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Exit inspector fullscreen" })).toBeVisible();
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "Fullscreen chat", exact: true }).click();
  await expect(page.locator(".inspector-pane")).toBeHidden();
  await expect(page.getByLabel("Message Veles")).toBeInViewport();
  const width = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(width.page).toBeLessThanOrEqual(width.viewport);
  await page.getByRole("button", { name: "Exit chat fullscreen" }).click();
  await expect(page.locator(".inspector-pane")).toBeVisible();
});

test("exposes PWA manifest, install prompt, screen wake lock, document fullscreen, and update controls", async ({
  page,
  request,
}) => {
  await page.addInitScript(() => {
    const wakeState = {
      requests: 0,
      currentSentinel: null as { release: () => Promise<void> } | null,
    };

    class FakeWakeLockSentinel extends EventTarget {
      released = false;

      async release(): Promise<void> {
        if (this.released) return;
        this.released = true;
        this.dispatchEvent(new Event("release"));
      }
    }

    Object.defineProperty(window, "__VELES_WAKE_TEST__", {
      configurable: true,
      value: wakeState,
    });

    Object.defineProperty(navigator, "wakeLock", {
      configurable: true,
      value: {
        request: async (type: string) => {
          if (type !== "screen") throw new Error(`Unsupported wake lock type: ${type}`);
          wakeState.requests += 1;
          const sentinel = new FakeWakeLockSentinel();
          wakeState.currentSentinel = sentinel;
          return sentinel;
        },
      },
    });
  });

  const manifestResponse = await request.get("/manifest.webmanifest");
  expect(manifestResponse.ok()).toBeTruthy();
  const manifest = (await manifestResponse.json()) as {
    name?: string;
    short_name?: string;
    display?: string;
    icons?: Array<{ src?: string; sizes?: string; purpose?: string }>;
  };
  expect(manifest.short_name).toBe("Veles");
  expect(manifest.display).toBe("standalone");
  expect(manifest.icons?.length ?? 0).toBeGreaterThanOrEqual(3);

  await page.goto("/");
  await page.getByRole("button", { name: "Services", exact: true }).click();
  await expect(page.getByTestId("pwa-display-mode")).toHaveText("browser tab");
  await expect(page.getByTestId("pwa-network-status")).toHaveText("online");
  await expect(page.getByTestId("pwa-install")).toBeDisabled();

  await page.evaluate(() => {
    const promptEvent = new Event("beforeinstallprompt", { cancelable: true }) as Event & {
      prompt: () => Promise<void>;
      userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
    };
    promptEvent.prompt = async () => undefined;
    promptEvent.userChoice = Promise.resolve({ outcome: "accepted", platform: "web" });
    window.dispatchEvent(promptEvent);
  });

  await expect(page.getByTestId("pwa-install")).toBeEnabled();
  await expect(page.getByTestId("pwa-install")).toHaveText("Install app");
  await page.getByTestId("pwa-install").click();
  await expect(page.getByTestId("pwa-status")).toContainText("Install accepted");

  const wakeToggle = page.getByTestId("wake-lock-toggle");
  await expect(wakeToggle).toBeEnabled();
  await expect(wakeToggle).toHaveText("Keep screen awake");
  await wakeToggle.click();

  await expect(wakeToggle).toHaveAttribute("aria-pressed", "true");
  await expect(wakeToggle).toHaveText("Allow screen sleep");
  await expect(page.getByTestId("wake-lock-status")).toContainText("Screen will stay awake");
  expect(
    await page.evaluate(
      () => (window as unknown as { __VELES_WAKE_TEST__: { requests: number } }).__VELES_WAKE_TEST__.requests,
    ),
  ).toBe(1);

  await page.evaluate(async () => {
    const wakeTest = (
      window as unknown as {
        __VELES_WAKE_TEST__: { currentSentinel: { release: () => Promise<void> } | null };
      }
    ).__VELES_WAKE_TEST__;
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    document.dispatchEvent(new Event("visibilitychange"));
    await wakeTest.currentSentinel?.release();
  });

  await expect(page.getByTestId("wake-lock-status")).toContainText(
    "Wake lock paused while hidden; it will restore on return.",
  );

  await page.evaluate(() => {
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await expect(wakeToggle).toHaveAttribute("aria-pressed", "true");
  expect(
    await page.evaluate(
      () => (window as unknown as { __VELES_WAKE_TEST__: { requests: number } }).__VELES_WAKE_TEST__.requests,
    ),
  ).toBe(2);

  await wakeToggle.click();
  await expect(wakeToggle).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId("wake-lock-status")).toContainText("Screen wake lock is off.");

  await page.evaluate(() => {
    window.dispatchEvent(new Event("veles:sw-update-ready"));
  });
  await expect(page.getByTestId("pwa-update")).toBeVisible();
  await expect(page.getByTestId("pwa-status")).toContainText("An app update is ready");
  await page.getByTestId("pwa-update").click();
  await expect(page.getByTestId("pwa-status")).toContainText("Applying update");
});


