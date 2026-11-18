import { expect, type Page } from "@playwright/test";

/** Offline scenarios opt out of the real provider defaults explicitly. */
export async function configureOfflineRun(page: Page, provider = "veles-demo") {
  await page.getByRole("button", { name: "Services", exact: true }).click();
  await page.locator("#svc-chat-provider").selectOption(provider);
  await page.locator("#svc-decision-mode").selectOption("tier0-only");
  await page.getByRole("button", { name: "✕ Close Services", exact: true }).click();
  await page.locator(".prompt-bar input[type='text']").fill("Build a sandboxed Counter App in index.html and clean up obsolete.txt");
}

export async function expectWorkspaceStatus(page: Page, text: string) {
  await page.getByRole("button", { name: "Services", exact: true }).click();
  await expect(page.getByLabel("Workspace status")).toContainText(text);
  await page.getByRole("button", { name: "✕ Close Services", exact: true }).click();
}

export async function openWorkspacePanel(page: Page, panel: "journal" | "project") {
  if (!await page.locator("#workspace-browser").isVisible()) {
    await page.getByRole("button", { name: "Browse files", exact: true }).click();
  }
  const name = panel === "journal" ? /^📜 Journal/ : "Project";
  await page.locator(".browser-tabs").getByRole("button", { name }).click();
}
