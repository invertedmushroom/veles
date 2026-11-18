import { readFile } from "node:fs/promises";
import { expect, test, type BrowserContext, type Page } from "@playwright/test";
import { configureOfflineRun, expectWorkspaceStatus, openWorkspacePanel } from "./workbench-helpers.ts";

test("executes live browser-only TypeSafe specialist classification through the frontend proxy", async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto("/");
  await page.getByRole("button", { name: "Services", exact: true }).click();
  await expect(page.locator("#svc-decision-mode")).toHaveValue("openai-decisions");
  await expect(page.locator("#svc-decision-model")).toHaveValue("gpt-6-luna");
  await expect(page.locator("#svc-chat-provider")).toHaveValue("openai");
  await expect(page.locator("#svc-chat-model")).toHaveValue("gpt-4o");
  await page.locator("#svc-decision-mode").selectOption("typesafe");
  await expect(page.locator("#svc-decision-model")).toHaveValue("jev-1.14.0");
  const configured = await page.locator("#svc-decision-key").evaluate(
    (input) => (input as HTMLInputElement).value.length > 10,
  );
  test.skip(!configured, "VITE_TYPESAFE_API_KEY is not configured");
  const responsePromise = page.waitForResponse((response) =>
    response.url().includes("/api/proxy/typesafe") && response.request().method() === "POST",
  );
  const result = await page.evaluate(async () => {
    const read = (id: string) => (document.getElementById(id) as HTMLInputElement).value;
    const { createDecisionClassifierAdapter, DEFAULT_BROWSER_SPECIALISTS } =
      await import(location.origin + "/src/kernel/harness-router.ts");
    const classifier = createDecisionClassifierAdapter(() => ({ decision: {
      mode: "typesafe", endpointUrl: read("svc-decision-url"), model: read("svc-decision-model"),
      apiKey: read("svc-decision-key"), threshold: 0.75,
    } }));
    const cases = [
      { prompt: "Compute the mean and median from assets/data.csv and render a histogram scatter plot.", paths: ["assets/data.csv"] },
      { prompt: "Draw an editable SVG icon for a compass with simple vector paths.", paths: [] },
      { prompt: "Write a self-contained technical HTML article with architecture diagrams explaining database transactions.", paths: [] },
      { prompt: "Change the horse in this photograph to rainbow colors, preserving its pose and the original scene.", paths: ["assets/horse.png"] },
    ];
    const results = [];
    for (const entry of cases) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 15_000);
      try {
        results.push(await classifier(entry.prompt, entry.paths, DEFAULT_BROWSER_SPECIALISTS, controller.signal));
      } finally {
        clearTimeout(timer);
      }
    }
    return results;
  });
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const expectedIds = [
    ["data-script-compute"], ["svg-multimodal-designer"],
    ["concept-article", "diagram-design"], ["raster-image-designer"],
  ];
  for (const [index, ids] of expectedIds.entries()) {
    expect(result[index].status).toBe("matched");
    expect(result[index].specialistIds).toEqual(expect.arrayContaining(ids));
    expect(result[index].specialistIds).not.toContain("canvas-game-builder");
  }
  const body = response.request().postDataJSON();
  expect(body.state).toContain("Compute the mean");
  expect(body.questions["data-script-compute"].type).toBe("noul");
});

async function installBrokenPreviewProvider(context: BrowserContext, approvalPath?: string) {
  let repairAttempts = 0;
  await context.route("https://api.openai.com/v1/chat/completions", async (route) => {
    const body = route.request().postDataJSON();
    const messages = body.messages as { role: string; content: string }[];
    const userIndex = messages.map((m) => m.role).lastIndexOf("user");
    const isRepair = messages[userIndex]!.content.includes("preview-repair child agent");
    const hasResults = messages.slice(userIndex + 1).some((m) => m.role === "tool");
    const calls: { id: string; type: string; function: { name: string; arguments: string } }[] = [];
    const writeFile = (path: string, content: string) => calls.push({
      id: `write-${calls.length}`, type: "function",
      function: { name: "workspace__write_file", arguments: JSON.stringify({ path, content }) },
    });
    if (!hasResults) {
      if (isRepair) repairAttempts += 1;
      if (!isRepair) {
        writeFile("index.html", '<html><head></head><body><script src="script.js"></script></body></html>');
      }
      // Bound the fixture even when the application regresses into a repair loop.
      if (repairAttempts < 3) writeFile("script.js", `throw new Error("still broken ${repairAttempts}");`);
      if (!isRepair && approvalPath) calls.push({
        id: "delete-unused", type: "function",
        function: { name: "workspace__delete_file", arguments: JSON.stringify({ path: approvalPath }) },
      });
    }
    await route.fulfill({
      contentType: "text/event-stream",
      body: `data: ${JSON.stringify({ choices: [{ delta: { content: "Done.", tool_calls: calls }, finish_reason: calls.length ? "tool_calls" : "stop" }] })}\n\ndata: [DONE]\n\n`,
    });
  });
  return () => repairAttempts;
}

async function startBrokenPreview(page: Page) {
  await page.goto("/");
  await configureOfflineRun(page, "openai");
  await page.locator(".prompt-bar input[type='text']").fill("Build a broken preview for the regression fixture");
  await page.getByRole("button", { name: "Run Prompt" }).click();
  await expect(page.locator(".console-box")).toContainText("still broken 0");
}

test.describe("Veles Browser Workbench E2E (Real Chromium Web Locks, IDB, OPFS, & Sandboxed Iframe)", () => {
  test("executes Counter App run, approval gate, sandboxed iframe postMessage, and persists across page.reload() (Scenarios 3 & 8)", async ({
    page,
  }) => {
    await page.goto("/");

    // Workspace controls are always accessible; telemetry is grouped in Services.
    await configureOfflineRun(page);
    await expect(page.getByLabel("Switch Workspace")).toContainText("Alpha");
    await expectWorkspaceStatus(page, "ReadWrite_Leader");
    await expectWorkspaceStatus(page, "Head Seq: #0");

    // 2. Run built-in veles-demo Counter App flow; pauses at awaiting_approval for workspace.delete_file
    const promptBox = page.locator(".prompt-bar input[type='text']");
    await page.getByRole("button", { name: "Run Prompt" }).click();
    await expect(promptBox).toHaveValue("");

    const approvalGate = page.getByRole("region", { name: "Tool Approval Gate" });
    await expect(approvalGate).toBeVisible();
    await expect(approvalGate).toContainText("workspace.delete_file");
    await expect(approvalGate).toContainText("obsolete.txt");

    // Both index.html and obsolete.txt are staged in OPFS before approval
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
    await expect(page.locator(".item-row", { hasText: "index.html" })).toBeVisible();
    await expect(page.locator(".item-row", { hasText: "obsolete.txt" })).toBeVisible();
    await page.getByRole("button", { name: "Close browser" }).click();

    // 3. Approve deletion -> completes run, removes obsolete.txt, and mounts isolated preview
    await approvalGate.getByRole("button", { name: /Approve Execution/i }).click();
    await expect(approvalGate).toBeHidden();
    await expect(page.locator(".pane-header", { hasText: "State: completed" })).toBeVisible();
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
    await expect(page.locator(".item-row", { hasText: "obsolete.txt" })).toHaveCount(0);
    await page.getByRole("button", { name: "Close browser" }).click();

    // 4. Verify <iframe sandbox="allow-scripts"> isolation (SEC-INV-01) and opaque-origin postMessage
    const previewIframe = page.locator(".preview-frame-container iframe");
    await expect(previewIframe).toHaveAttribute("sandbox", "allow-scripts");

    const frame = page.frameLocator(".preview-frame-container iframe");
    await expect(frame.locator("#count-val")).toHaveText("42");
    await expect(frame.locator("#origin-label")).toHaveText("null");

    // Click '+' inside the sandboxed iframe and verify postMessage diagnostic reaches host console
    await frame.locator("#inc-btn").click();
    await expect(frame.locator("#count-val")).toHaveText("43");
    await expect(page.locator(".console-box")).toContainText(
      "Counter incremented to 43",
    );

    // 4b. Switch between Code / Journal and Live Preview -> verify live iframe stays visible & Remount Preview works
    await page.getByRole("button", { name: "Code", exact: true }).click();
    await page.getByRole("button", { name: /Live Preview/i }).click();
    await expect(frame.locator("#count-val")).toHaveText("43");
    await page.getByRole("button", { name: "Remount Preview" }).click();
    await expect(frame.locator("#count-val")).toHaveText("42");

    // Project archives are accessible from the same browser as files and the journal.
    const originalWorkspaceId = await page.getByLabel("Switch Workspace").inputValue();
    await openWorkspacePanel(page, "project");
    const downloaded = page.waitForEvent("download");
    await page.getByRole("button", { name: "↥ Export project", exact: true }).click();
    const downloadPath = await (await downloaded).path();
    expect(downloadPath).not.toBeNull();
    const archive = await readFile(downloadPath!, "utf8");
    await page.getByRole("button", { name: "Close browser" }).click();

    // Create an isolated target, import the archive there, then delete the copy.
    await page.getByRole("button", { name: "New workspace", exact: true }).click();
    await page.getByLabel("New Workspace Name").fill("Sandbox Beta");
    await page.getByRole("button", { name: "Create workspace", exact: true }).click();
    await expect(page.getByLabel("Switch Workspace")).toContainText("Sandbox Beta");
    await expectWorkspaceStatus(page, "Head Seq: #0");
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
    await expect(page.getByRole("button", { name: "📁 Files (0)" })).toBeVisible();
    await page.getByRole("button", { name: "Close browser" }).click();
    await openWorkspacePanel(page, "project");
    const picker = page.waitForEvent("filechooser");
    await page.getByRole("button", { name: "↧ Import project", exact: true }).click();
    await (await picker).setFiles({ name: "project.json", mimeType: "application/json", buffer: Buffer.from(archive) });
    await expect(page.getByLabel("Switch Workspace").locator("option:checked")).toHaveText("Alpha");
    await page.getByRole("button", { name: /^📁 Files/ }).click();
    await expect(page.locator(".item-row", { hasText: "index.html" })).toBeVisible();
    await page.getByRole("button", { name: "Close browser" }).click();
    page.once("dialog", (dialog) => { void dialog.dismiss(); });
    await page.getByRole("button", { name: "Delete workspace", exact: true }).click();
    await expect(page.getByLabel("Switch Workspace").locator("option:checked")).toHaveText("Alpha");
    page.once("dialog", (dialog) => { void dialog.accept(); });
    await page.getByRole("button", { name: "Delete workspace", exact: true }).click();
    await expect(page.getByLabel("Switch Workspace")).toHaveValue(originalWorkspaceId);

    // 5. Scenario 3: Reload page and verify IndexedDB journal + OPFS index.html persist
    await page.reload();
    await expectWorkspaceStatus(page, "ReadWrite_Leader");
    await page.getByRole("button", { name: "Services", exact: true }).click();
    await expect(page.getByLabel("Workspace status")).not.toContainText("Head Seq: #0");
    await page.getByRole("button", { name: "✕ Close Services" }).click();
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
    await expect(page.locator(".item-row", { hasText: "index.html" })).toBeVisible();
    await openWorkspacePanel(page, "journal");
    await expect(page.locator(".item-row", { hasText: "RunCompleted" })).toBeVisible();
  });

  test("enforces Multi-Tab Web Locks lease authority without exposing chaos controls", async ({
    context,
  }) => {
    const leaderPage = await context.newPage();
    await leaderPage.goto("/");

    await expectWorkspaceStatus(leaderPage, "ReadWrite_Leader");

    // Open a second tab in the same BrowserContext -> navigator.locks demotes Tab 2 to ReadOnly_Observer
    const observerPage = await context.newPage();
    await observerPage.goto("/");

    await expectWorkspaceStatus(observerPage, "ReadOnly_Observer");
    await expect(
      observerPage.getByRole("button", { name: "Run Prompt" }),
    ).toBeDisabled();
    await openWorkspacePanel(observerPage, "project");
    await expect(observerPage.getByRole("button", { name: "↧ Import project", exact: true })).toBeDisabled();
    await expect(observerPage.getByRole("button", { name: "↥ Export project", exact: true })).toBeEnabled();
    await observerPage.close();

    await expect(leaderPage.getByRole("button", { name: /Simulate Quota|Gateway Restart|Drop Stream/i })).toHaveCount(0);
  });

  test("allows one automatic repair per activation and permits an explicit retry", async ({ page, context }) => {
    const attempts = await installBrokenPreviewProvider(context);
    await startBrokenPreview(page);
    const toggle = page.getByRole("checkbox", { name: "Auto-Repair (1x)" });
    await toggle.click();
    await expect(toggle).not.toBeChecked();
    await expect(page.locator(".console-box")).toContainText("still broken 1");
    expect(attempts()).toBe(1);
    await toggle.click();
    await expect(toggle).not.toBeChecked();
    await expect(page.locator(".console-box")).toContainText("still broken 2");
    expect(attempts()).toBe(2);
  });

  test("repairs the mounted preview after approval advances only the journal", async ({ page, context }) => {
    const attempts = await installBrokenPreviewProvider(context, "already-absent.txt");
    await startBrokenPreview(page);
    const approval = page.getByRole("region", { name: "Tool Approval Gate" });
    await expect(approval).toBeVisible();
    await page.getByRole("checkbox", { name: "Auto-Repair (1x)" }).check();
    await approval.getByRole("button", { name: /Deny/i }).click();
    await expect.poll(attempts).toBe(1);
    await expect(page.locator(".console-box")).toContainText("still broken 1");
  });

  test("executes live browser-only OpenAI SSE tool loop (workspace.write_file -> OPFS -> sandboxed iframe preview)", async ({
    page,
  }) => {
    test.setTimeout(60_000);
    await page.goto("/");

    // Verify the production defaults and environment credentials in the one service editor.
    await page.getByRole("button", { name: "Services", exact: true }).click();
    await expect(page.locator("#svc-chat-provider")).toHaveValue("openai");
    await expect(page.locator("#svc-chat-model")).toHaveValue("gpt-4o");
    await expect(page.locator("#svc-decision-mode")).toHaveValue("openai-decisions");
    await expect(page.locator("#svc-decision-model")).toHaveValue("gpt-6-luna");
    const apiKeyInput = page.locator("#svc-chat-key");
    expect((await apiKeyInput.inputValue()).length).toBeGreaterThan(10);
    await page.getByRole("button", { name: "✕ Close Services" }).click();

    // Send a deterministic prompt to OpenAI instructing it to write index.html in the workspace
    const promptInput = page.locator(".prompt-bar input[type='text']");
    await promptInput.fill(
      "Create index.html in the workspace with this exact HTML content: <!doctype html><html><body><h1 id='live-openai'>OpenAI OPFS Verified</h1></body></html>",
    );
    await page.getByRole("button", { name: "Run Prompt" }).click();

    // Wait for OpenAI tool call + OPFS commit + final completion turn
    await expect(
      page.locator(".pane-header", { hasText: "State: completed" }),
    ).toBeVisible({ timeout: 45_000 });

    // Verify OPFS file tree and IndexedDB journal records
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
    await expect(
      page.locator(".item-row", { hasText: "index.html" }),
    ).toBeVisible();
    await openWorkspacePanel(page, "journal");
    await expect(
      page.locator(".item-row", { hasText: "ToolResultCommitted" }),
    ).toBeVisible();
    await expect(
      page.locator(".item-row", { hasText: "RunCompleted" }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Close browser" }).click();

    // Verify live sandboxed iframe rendered the HTML written by OpenAI
    const frame = page.frameLocator(".preview-frame-container iframe");
    await expect(frame.locator("#live-openai")).toHaveText(
      "OpenAI OPFS Verified",
    );

    // Inspect the durable records through the workbench's existing journal view.
    await page.getByRole("button", { name: "Code", exact: true }).click();
    const journalRecord = page.locator(".section-box").filter({ hasText: "Journal Record #" }).locator("pre");
    await openWorkspacePanel(page, "journal");
    const checkpoints = page.locator(".item-row", { hasText: "CheckpointCommitted" });
    const checkpointCount = await checkpoints.count();
    let requestTokens = 0;
    let routingSeen = false;
    for (let index = 0; index < checkpointCount; index += 1) {
      if (index > 0) await openWorkspacePanel(page, "journal");
      await checkpoints.nth(index).click();
      const checkpoint = JSON.parse(await journalRecord.innerText());
      if (checkpoint.payload.inferenceUsage) requestTokens = checkpoint.payload.inferenceUsage.totalTokens;
      if (checkpoint.payload.harnessRouting) {
        const routing = checkpoint.payload.harnessRouting;
        expect(routing.configuredMode).toBe("openai-decisions");
        expect(["tier0_deterministic", "tier1_classifier"]).toContain(routing.source);
        if (routing.source === "tier1_classifier") expect(routing.specialistIds.length).toBeGreaterThan(0);
        else expect(routing.fallbackReason).toBeTruthy();
        routingSeen = true;
      }
    }
    expect(requestTokens).toBeGreaterThan(0);
    expect(routingSeen).toBe(true);
    await openWorkspacePanel(page, "journal");
    await page.locator(".item-row", { hasText: "RunCompleted" }).click();
    const completed = JSON.parse(await journalRecord.innerText());
    expect(completed.payload.durationMs).toBeGreaterThan(0);
    expect(completed.payload.usage.totalTokens).toBeGreaterThan(requestTokens);
  });
});
