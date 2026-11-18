import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type {
    InferenceAdapter,
    NormalizedStreamChunk,
    ProviderModelConfig,
} from "../src/inference/openai-compatible.ts";
import { AgentManager } from "../src/kernel/agent-session.ts";
import {
    ContentAddressedBlobStore,
    InMemoryBlobBackend,
} from "../src/kernel/blobs.ts";
import { GatewayToolExecutionAdapter } from "../src/kernel/gateway-client.ts";
import {
    createDecisionClassifierAdapter,
    DEFAULT_BROWSER_SPECIALISTS,
    redactClassifierSecrets,
    VelesHarnessRouter,
} from "../src/kernel/harness-router.ts";
import {
    InMemoryJournalBackend,
    WorkspaceKernelStore,
} from "../src/kernel/journal.ts";
import { InMemoryLockProvider } from "../src/kernel/lock.ts";
import { AgentKernelRuntime, ContextBuilder } from "../src/kernel/runtime.ts";
import {
    BrowserToolRegistry,
    createDefaultWorkspaceTools,
    createOpenAIImageProvider,
    createSubagentTools,
    isBlockedPrivateHost,
    normalizePublicFetchUrl,
} from "../src/kernel/tools.ts";
import { KernelWorkerHost } from "../src/kernel/worker-host.ts";
import {
    summarizeWorkspaceFileSignature,
    verifyWorkspaceFilePreflight,
} from "../src/kernel/workspace-intelligence.ts";
import type {
    RunJournalRecord,
    ServiceCapabilitiesConfigSpec,
    WorkerPortLike,
} from "../src/protocol/types.ts";
import { VelesAppClient } from "../src/ui/app-client.ts";
import {
    compileMultiFileSrcdoc,
    pickDefaultWorkspaceFile,
    PreviewSandboxHost,
    resolvePreviewEntryPath,
    type MessageEventLike,
    type PreviewRuntimeDiagnostic,
    type SandboxEnvironmentLike,
    type SandboxIframeLike,
} from "../src/ui/preview-sandbox.ts";

const TEST_CONFIG: ProviderModelConfig = {
    providerId: "openai",
    baseUrl: "https://api.example.test/v1",
    model: "gpt-4o",
};

async function createSliceWorkspace(id = "ws-slice") {
    const locks = new InMemoryLockProvider();
    const lease = await locks.acquireWorkspaceLease(id, "worker-slice");
    const journalBackend = new InMemoryJournalBackend();
    const blobBackend = new InMemoryBlobBackend();
    const store = new WorkspaceKernelStore(
        journalBackend,
        new ContentAddressedBlobStore(blobBackend),
        lease,
        locks,
    );
    await store.initWorkspace("Slice Test Workspace");
    return { store, journalBackend, blobBackend };
}

async function commitWorkspaceToolRun(
    store: WorkspaceKernelStore,
    runId: string,
    toolName: string,
    args: Record<string, unknown>,
) {
    const tools = new BrowserToolRegistry(store);
    let round = 0;
    const runtime = new AgentKernelRuntime(store, {
        async *streamChat(req) {
            round += 1;
            yield {
                providerId: req.config.providerId, model: req.config.model,
                deltaText: "", usage: null, finishReason: round === 1 ? "tool_calls" : "stop",
                toolCalls: round === 1 ? [{ id: `${runId}-call`, name: toolName, argumentsJson: JSON.stringify(args) }] : [],
            };
        },
    }, tools, tools.getToolDefinitions());
    await runtime.startRun(runId, toolName, TEST_CONFIG);
}

describe("Veles In-Browser Agent Harness (Multi-Tool Batching, sha256 Drift, Specialist Router, & Fenced Preview)", () => {
    it("allows repair rereads after the target changes while guarding unchanged duplicate reads", async () => {
        const { store } = await createSliceWorkspace("ws-repair-reread");
        await store.writeWorkspaceFile("preview/a.txt", "A");
        const tools = new BrowserToolRegistry(store);
        const read = { name: "workspace.read_file", argumentsJson: JSON.stringify({ path: "preview/a.txt" }) };
        const write = (path: string, content: string) => ({ name: "workspace.write_file", argumentsJson: JSON.stringify({ path, content }) });
        const calls = [read, write("notes/unrelated.txt", "irrelevant"), read, write("preview/a.txt", "B"), read, write("preview/a.txt", "A"), read, read];
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? calls.map((call, i) => ({ ...call, id: `repair-${i}` })) : [],
                };
            },
        }, tools, tools.getToolDefinitions());
        assert.equal(await runtime.startRun("repair-read", "Read, edit, and verify", TEST_CONFIG), "completed");
        const results = (await store.getJournal()).filter((record) => record.type === "ToolResultCommitted" && record.payload.toolName === "workspace.read_file");
        assert.equal(results[0]!.payload.outputText, "A");
        assert.match(String(results[1]!.payload.outputText), /LOOP_GUARD_TRIGGERED/);
        assert.equal(results[2]!.payload.outputText, "B", "A changed target must be readable for repair verification");
        assert.equal(results[3]!.payload.outputText, "A", "Returning to a prior version must reset the consecutive-read guard");
        assert.match(String(results[4]!.payload.outputText), /LOOP_GUARD_TRIGGERED/);
    });

    it("allows listing a changed file tree while guarding an unchanged duplicate listing", async () => {
        const { store } = await createSliceWorkspace("ws-repair-list");
        const tools = new BrowserToolRegistry(store);
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? [
                        { id: "list-before", name: "workspace.list_files", argumentsJson: "{}" },
                        { id: "new-file", name: "workspace.write_file", argumentsJson: JSON.stringify({ path: "notes/new.txt", content: "new" }) },
                        { id: "list-after", name: "workspace.list_files", argumentsJson: "{}" },
                        { id: "list-duplicate", name: "workspace.list_files", argumentsJson: "{}" },
                    ] : [],
                };
            },
        }, tools, tools.getToolDefinitions());
        await runtime.startRun("repair-list", "Verify the new file", TEST_CONFIG);
        const results = (await store.getJournal()).filter((record) => record.type === "ToolResultCommitted" && record.payload.toolName === "workspace.list_files");
        assert.match(String(results[1]!.payload.outputText), /notes\/new\.txt/);
        assert.doesNotMatch(String(results[1]!.payload.outputText), /LOOP_GUARD_TRIGGERED/);
        assert.match(String(results[2]!.payload.outputText), /LOOP_GUARD_TRIGGERED/);
    });

    it("preserves caught sandbox error names and messages in console output", async () => {
        const { store } = await createSliceWorkspace("ws-script-error-output");
        const tools = new BrowserToolRegistry(store);
        const outcome = await tools.executeTool({
            runId: "script-error", operationId: "script-error-log", toolName: "workspace.run_script", effectClass: "idempotent_write",
            args: { code: "try { tools.workspace.read_file({ path: 'preview/script.js' }); } catch (error) { console.error('Error reading file:', error); } console.log({ ordinary: true });" },
        });
        assert.match(outcome.outputText, /Error reading file: TypeError: .*read_file.*not a function/);
        assert.match(outcome.outputText, /\{"ordinary":true\}/);
    });

    it("keeps workspace-read dispatch available when loop-guard metadata is unavailable", async (t) => {
        const { store } = await createSliceWorkspace("ws-read-guard-metadata");
        const originalTree = store.getFileTree.bind(store);
        let toolRound = false;
        let dispatched = false;
        t.mock.method(store, "getFileTree", async () => {
            if (toolRound && !dispatched) throw new Error("temporary guard metadata failure");
            return originalTree();
        });
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                toolRound = true;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? [{ id: "read", name: "workspace.read_file", argumentsJson: '{"path":"notes/a.txt"}' }] : [],
                };
            },
        }, {
            currentGatewayInstanceId: "test",
            async executeTool() { dispatched = true; return { outputText: "read result", gatewayInstanceId: "test" }; },
        }, [{ name: "workspace.read_file", description: "Fixture read", effectClass: "pure_read" }]);
        assert.equal(await runtime.startRun("metadata-read", "Read", TEST_CONFIG), "completed");
        assert.equal(dispatched, true);
    });

    it("scopes curated tool call IDs to their originating run", () => {
        const record = (seq: number, runId: string, type: RunJournalRecord["type"], payload: Record<string, unknown>): RunJournalRecord => ({
            eventId: `curation-${seq}`, workspaceId: "ws-curation", runId, seq, type, payload,
            schemaVersion: 1, timestampMs: seq, relatedOperationId: null, committedState: "inferring",
        });
        const records = [
            record(1, "old", "RunCreated", { prompt: "Inspect old" }),
            record(2, "old", "ToolIntentLogged", { toolCallId: "call_1", toolName: "workspace.read_file", args: { path: "preview/a.js" } }),
            record(3, "old", "ToolResultCommitted", { toolCallId: "call_1", outputText: `OLD:${"x".repeat(1200)}` }),
            record(4, "new", "RunCreated", { prompt: "Inspect current" }),
            record(5, "new", "ToolIntentLogged", { toolCallId: "call_1", toolName: "workspace.read_file", args: { path: "preview/a.js" } }),
            record(6, "new", "ToolResultCommitted", { toolCallId: "call_1", outputText: `NEW:${"y".repeat(1200)}` }),
        ];
        const tools = ContextBuilder.buildMessages(records, { hotRunWindow: 2 }).filter((m) => m.role === "tool");
        assert.match(tools[0]!.content, /\[folded/);
        assert.equal(tools[1]!.content, records[5]!.payload.outputText);
        const sameRun = records.filter((record) => record.seq !== 4).map((record) => ({
            ...record,
            runId: "single-run",
            relatedOperationId: record.type === "RunCreated" ? null : record.seq < 4 ? "first-op" : "second-op",
        }));
        const sameRunTools = ContextBuilder.buildMessages(sameRun).filter((message) => message.role === "tool");
        assert.match(sameRunTools[0]!.content, /\[folded/);
        assert.equal(sameRunTools[1]!.content, records[5]!.payload.outputText);
    });

    it("does not resolve a tool error with an uncommitted retry", () => {
        const record = (seq: number, runId: string, type: RunJournalRecord["type"], payload: Record<string, unknown>): RunJournalRecord => ({
            eventId: `retry-${seq}`, workspaceId: "ws-curation-retry", runId, seq, type, payload,
            schemaVersion: 1, timestampMs: seq, relatedOperationId: null, committedState: "inferring",
        });
        const error = `ToolError (workspace.edit_file): ${"target missing ".repeat(30)}`;
        const records = [
            record(1, "old", "RunCreated", { prompt: "Fix" }),
            record(2, "old", "ToolIntentLogged", { toolCallId: "error", toolName: "workspace.edit_file", args: { path: "preview/a.js" } }),
            record(3, "old", "ToolResultCommitted", { toolCallId: "error", outputText: error }),
            record(4, "retry", "RunCreated", { prompt: "Retry" }),
            record(5, "retry", "ToolIntentLogged", { toolCallId: "pending", toolName: "workspace.edit_file", args: { path: "preview/a.js" } }),
            record(6, "retry", "RunInterrupted", { interruptReason: "user_reload_or_tab_close" }),
            record(7, "current", "RunCreated", { prompt: "Continue" }),
        ];
        const result = ContextBuilder.buildMessages(records, { hotRunWindow: 3 }).find((m) => m.role === "tool" && m.toolCallId === "error");
        assert.equal(result?.content, error);
    });

    it("preserves significant literal whitespace when matching collapsed file edits", async () => {
        const { store } = await createSliceWorkspace("ws-literal-edits");
        const tools = new BrowserToolRegistry(store);
        for (const literal of ['"red  blue"', "'red  blue'", "`red  blue`", "/red  blue/"]) {
            const original = `const label = ${literal};\nconst count = 1;\n`;
            await store.writeWorkspaceFile("preview/a.js", original);
            await assert.rejects(tools.executeTool({
                runId: "literal", operationId: "literal-edit", toolName: "workspace.edit_file", effectClass: "idempotent_write",
                args: { path: "preview/a.js", target: `const label = ${literal.replace("  ", " ")}; const count = 1;`, replacement: "const count = 2;" },
            }), /ERR_EDIT_TARGET_NOT_FOUND/);
            assert.equal(await store.readFileText("preview/a.js"), original);
        }
    });

    it("retains file history from conversation-only deleted turns when reverting a later edit", async () => {
        const { store } = await createSliceWorkspace("ws-revert-deleted-conversation");
        const path = "preview/app.js";
        await store.writeWorkspaceFile(path, "baseline");
        await commitWorkspaceToolRun(store, "first-edit", "workspace.write_file", { path, content: "first edit" });
        await commitWorkspaceToolRun(store, "second-edit", "workspace.write_file", { path, content: "second edit" });
        await store.deleteRun("first-edit");
        assert.equal(await store.readFileText(path), "second edit");
        await store.deleteRun("second-edit", { revertFiles: true });
        assert.equal(await store.readFileText(path), "first edit");
    });

    it("rejects unsafe historical paths before publishing a reverted manifest", async () => {
        const { store } = await createSliceWorkspace("ws-revert-paths");
        const path = "notes/safe.txt";
        await store.writeWorkspaceFile(path, "owned baseline");
        const baselineHash = (await store.getFileTree())[path]!;
        await commitWorkspaceToolRun(store, "path-edit", "workspace.write_file", { path, content: "changed" });
        const archive = await store.exportWorkspaceBundle();
        const journal = archive.journal.map((record) => {
            if (record.runId !== "path-edit") return record;
            if (record.type === "RunCreated") return { ...record, payload: {
                ...record.payload,
                preRunFileTree: { ...(record.payload.preRunFileTree as Record<string, string>), "../escape.txt": baselineHash },
            } };
            if (record.type === "ToolResultCommitted") return { ...record, payload: {
                ...record.payload, observedFileHashes: { "../escape.txt": baselineHash },
            } };
            return record;
        });
        await store.importWorkspaceBundle({ ...archive, journal });
        const before = await store.exportWorkspaceBundle();
        await assert.rejects(store.deleteRun("path-edit", { revertFiles: true }), /ERR_UNSAFE_FILE_PATH/);
        assert.deepEqual(await store.getFileTree(), before.fileTree);
        assert.equal((await store.getMetadata())!.headSeq, before.metadata.headSeq);
    });

    it("uses read observations to restore legacy archives without pre-turn file snapshots", async () => {
        const { store } = await createSliceWorkspace("ws-revert-legacy-read");
        const path = "preview/app.js";
        await store.writeWorkspaceFile(path, "legacy baseline");
        await commitWorkspaceToolRun(store, "legacy-read", "workspace.read_file", { path });
        await commitWorkspaceToolRun(store, "legacy-write", "workspace.write_file", { path, content: "legacy edit" });
        const bundle = await store.exportWorkspaceBundle();
        const journal = bundle.journal
            .filter((record) => typeof record.payload.uploadedPath !== "string")
            .map((record, index) => {
                const { preRunFileTree: _snapshot, ...payload } = record.payload;
                return { ...record, seq: index + 1, payload };
            });
        await store.importWorkspaceBundle({ ...bundle, journal, metadata: { ...bundle.metadata, headSeq: journal.length } });
        await store.deleteRun("legacy-write", { revertFiles: true });
        assert.equal(await store.readFileText(path), "legacy baseline");
    });

    it("exports the owned historical blobs required to revert after a fresh archive import", async () => {
        const { store } = await createSliceWorkspace("ws-revert-fresh-archive");
        const path = "preview/app.js";
        await store.writeWorkspaceFile(path, "archived baseline");
        await commitWorkspaceToolRun(store, "archived-edit", "workspace.write_file", { path, content: "archived edit" });
        const bundle = await store.exportWorkspaceBundle();
        const { store: freshStore } = await createSliceWorkspace("ws-revert-fresh-archive");
        await freshStore.importWorkspaceBundle(bundle);
        await freshStore.deleteRun("archived-edit", { revertFiles: true });
        assert.equal(await freshStore.readFileText(path), "archived baseline");
    });

    it("rejects imported revert hashes belonging to another workspace without changing committed state", async () => {
        const { store } = await createSliceWorkspace("ws-revert-public");
        const privateStore = await store.forkForWorkspace("ws-revert-private", "Private");
        await privateStore.writeWorkspaceFile("notes/private.txt", "private workspace content");
        const privateHash = (await privateStore.getFileTree())["notes/private.txt"]!;
        const path = "preview/app.js";
        await store.writeWorkspaceFile(path, "public baseline");
        await commitWorkspaceToolRun(store, "public-edit", "workspace.write_file", { path, content: "public edit" });
        const bundle = await store.exportWorkspaceBundle();
        const journal = bundle.journal.map((record) => record.runId === "public-edit" && record.type === "RunCreated"
            ? { ...record, payload: { ...record.payload, preRunFileTree: { [path]: privateHash } } }
            : record);
        await store.importWorkspaceBundle({
            ...bundle, journal,
            metadata: { ...bundle.metadata, historyBlobHashes: [privateHash] },
        });
        const before = await store.getMetadata();
        await assert.rejects(store.deleteRun("public-edit", { revertFiles: true }), /ERR_REVERT_BLOB_UNAVAILABLE/);
        assert.deepEqual(await store.getMetadata(), before);
        assert.equal(await store.readFileText(path), "public edit");
        assert.equal((await store.getJournal()).length, bundle.journal.length);
        const exported = await store.exportWorkspaceBundle();
        assert.equal(exported.blobsBase64[privateHash], undefined);
    });

    it("includes GIF source images in direct Responses edit requests", async (t) => {
        const calls: Array<Record<string, unknown>> = [];
        t.mock.method(globalThis, "fetch", async (_url: unknown, init?: RequestInit) => {
            calls.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
            return Response.json({ output: [{ type: "image_generation_call", result: "AA==" }] });
        });
        const provider = createOpenAIImageProvider(() => ({ ...TEST_CONFIG, apiKey: "test-key" }));
        const sourceDataUrl = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
        await provider.generateOrTransform({ prompt: "make the subject blue", sourceDataUrl });
        assert.ok(Array.isArray(calls[0]!.input), "An edit request must include its source, not just a text prompt");
        assert.ok(JSON.stringify(calls[0]!.input).includes(sourceDataUrl));
    });

    it("keeps SVG source context when raster edit input cannot attach the original", async (t) => {
        const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
        t.mock.method(globalThis, "fetch", async (url: unknown, init?: RequestInit) => {
            const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
            requests.push({ url: String(url), body });
            if (String(url).endsWith("/chat/completions")) {
                return Response.json({ choices: [{ message: { content: "A detailed description preserving the source circle." } }] });
            }
            return String(url).endsWith("/responses")
                ? Response.json({ output: [{ type: "image_generation_call", result: "AA==" }] })
                : Response.json({ data: [{ b64_json: "AA==" }] });
        });
        const provider = createOpenAIImageProvider(() => ({ ...TEST_CONFIG, apiKey: "test-key" }));
        const source = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="42"/></svg>';
        await provider.generateOrTransform({ prompt: "make the circle blue", sourceDataUrl: `data:image/svg+xml;utf8,${encodeURIComponent(source)}` });
        assert.ok(requests.some((request) => JSON.stringify(request.body).includes("[Source SVG]") && JSON.stringify(request.body).includes("circle")), "A source-aware fallback must inspect the original SVG");
        assert.ok(requests.every((request) => !request.url.endsWith("/responses")), "Do not label a source-free generation as an edit");
    });

    it("continues to raster fallback when the direct edit endpoint is unavailable", async (t) => {
        t.mock.method(globalThis, "fetch", async (url: unknown) => {
            if (String(url).endsWith("/responses")) throw new TypeError("Failed to fetch");
            if (String(url).endsWith("/chat/completions")) {
                return Response.json({ choices: [{ message: { content: "Preserve the source subject and make it blue." } }] });
            }
            return Response.json({ data: [{ b64_json: "AA==" }] });
        });
        const provider = createOpenAIImageProvider(
            () => ({ ...TEST_CONFIG, apiKey: "test-key" }), undefined, () => "openai-dalle3",
        );
        const result = await provider.generateOrTransform({ prompt: "make it blue", sourceDataUrl: "data:image/png;base64,AA==" });
        assert.match(result.diagnostics ?? "", /openai:dall-e-3/);
    });

    it("does not enter image fallback after cancellation", async (t) => {
        const controller = new AbortController();
        const fetch = t.mock.method(globalThis, "fetch", async () => {
            controller.abort();
            throw controller.signal.reason;
        });
        const provider = createOpenAIImageProvider(() => ({ ...TEST_CONFIG, apiKey: "test-key" }));
        await assert.rejects(provider.generateOrTransform({
            prompt: "make it blue", sourceDataUrl: "data:image/png;base64,AA==", signal: controller.signal,
        }), { name: "AbortError" });
        assert.equal(fetch.mock.callCount(), 1);
    });

    it("executes multiple toolCalls in a single assistant turn and pauses/resumes cleanly if a mid-batch call requires approval", async () => {
        const { store } = await createSliceWorkspace("ws-batch-tools");
        await store.writeWorkspaceFile("obsolete.txt", "delete me");
        const tools = new BrowserToolRegistry(store);
        let llmRounds = 0;

        const inference: InferenceAdapter = {
            async *streamChat(req) {
                llmRounds += 1;
                if (llmRounds === 1) {
                    // Single assistant turn emits 3 tool calls: write HTML, delete obsolete.txt (requiresApproval!), write JS
                    yield {
                        providerId: req.config.providerId,
                        model: req.config.model,
                        deltaText: "Scaffolding multi-file Mario game in one turn...",
                        toolCalls: [
                            {
                                id: "call_1_html",
                                name: "workspace.write_file",
                                argumentsJson: JSON.stringify({
                                    path: "preview/index.html",
                                    content: "<!doctype html><canvas></canvas>",
                                }),
                            },
                            {
                                id: "call_2_del",
                                name: "workspace.delete_file",
                                argumentsJson: JSON.stringify({ path: "obsolete.txt" }),
                            },
                            {
                                id: "call_3_js",
                                name: "workspace.write_file",
                                argumentsJson: JSON.stringify({
                                    path: "preview/script.js",
                                    content: "console.log('mario');",
                                }),
                            },
                        ],
                        usage: null,
                        finishReason: "tool_calls",
                    };
                    return;
                }
                yield {
                    providerId: req.config.providerId,
                    model: req.config.model,
                    deltaText: "All 3 files processed!",
                    toolCalls: [],
                    usage: { promptTokens: 50, completionTokens: 12, totalTokens: 62 },
                    finishReason: "stop",
                };
            },
        };

        const runtime = new AgentKernelRuntime(
            store,
            inference,
            tools,
            tools.getToolDefinitions(),
            { multiTurnWorkspaceContext: true },
        );

        // Start run: Call 1 executes, Call 2 pauses at awaiting_approval, Call 3 is queued in remainingToolCalls
        const paused = await runtime.startRun(
            "run-batch-1",
            "Create Mario game and remove obsolete.txt",
            TEST_CONFIG,
        );
        assert.equal(paused, "awaiting_approval");
        assert.equal(llmRounds, 1);
        assert.deepEqual(await store.listFiles(), ["obsolete.txt", "preview/index.html"]);

        // Approve Call 2 -> deletes obsolete.txt AND automatically executes queued Call 3 (preview/script.js) before Round 2!
        const finalState = await runtime.resolveApproval("run-batch-1", true);
        assert.equal(finalState, "completed");
        assert.equal(llmRounds, 2);
        assert.deepEqual(await store.listFiles(), [
            "preview/index.html",
            "preview/script.js",
        ]);

        // Verify RunCompleted recorded usage and durationMs telemetry
        const journal = await store.getJournal();
        const completedRec = journal[journal.length - 1]!;
        assert.equal(completedRec.type, "RunCompleted");
        assert.equal(typeof completedRec.payload.durationMs, "number");
        assert.deepEqual(completedRec.payload.usage, {
            promptTokens: 50,
            completionTokens: 12,
            totalTokens: 62,
        });
    });

    it("annotates prior tool results with [DRIFT NOTE] when a file is modified externally in OPFS and injects Specialist capsules", async () => {
        const { store } = await createSliceWorkspace("ws-drift");
        const tools = new BrowserToolRegistry(store);
        const observedSystemPrompts: string[] = [];
        const observedToolOutputs: string[] = [];

        let turn = 0;
        const inference: InferenceAdapter = {
            async *streamChat(req) {
                turn += 1;
                const sysMsg = req.messages.find((m) => m.role === "system");
                if (sysMsg) observedSystemPrompts.push(sysMsg.content);
                for (const m of req.messages) {
                    if (m.role === "tool") observedToolOutputs.push(m.content);
                }

                if (turn === 1) {
                    yield {
                        providerId: req.config.providerId,
                        model: req.config.model,
                        deltaText: "Writing initial script...",
                        toolCalls: [
                            {
                                id: "call_init_js",
                                name: "workspace.write_file",
                                argumentsJson: JSON.stringify({
                                    path: "preview/script.js",
                                    content: "const jump = 10;",
                                }),
                            },
                        ],
                        usage: null,
                        finishReason: "tool_calls",
                    };
                    return;
                }
                yield {
                    providerId: req.config.providerId,
                    model: req.config.model,
                    deltaText: "Done.",
                    toolCalls: [],
                    usage: null,
                    finishReason: "stop",
                };
            },
        };

        const runtime = new AgentKernelRuntime(
            store,
            inference,
            tools,
            tools.getToolDefinitions(),
            { multiTurnWorkspaceContext: true },
        );

        // Run 1: Agent writes preview/script.js ("const jump = 10;")
        await runtime.startRun(
            "run-drift-1",
            "Build a Mario canvas platformer game in preview/script.js",
            TEST_CONFIG,
        );
        assert.ok(
            observedSystemPrompts[0]?.includes("[Specialist: HTML5 Canvas & Game Builder]"),
            "Expected Layer 2 Canvas Game specialist capsule in system prompt",
        );

        // External modification in OPFS (e.g. user upload or another chat session modifies preview/script.js)
        await store.writeWorkspaceFile("preview/script.js", "const jump = 999;");

        // Run 2: ContextBuilder detects that preview/script.js sha256 changed since Run 1's ToolResultCommitted
        const journal = await store.getJournal();
        const liveTree = await store.getFileTree();
        const messagesForRun2 = ContextBuilder.buildMessages(journal, {
            fileTree: liveTree,
            hotRunWindow: 2,
        });
        const toolMsg = messagesForRun2.find((m) => m.role === "tool");
        assert.ok(toolMsg);
        assert.match(
            toolMsg.content,
            /\[DRIFT NOTE: 'preview\/script\.js' has changed in OPFS since this turn/,
        );
    });

    it("preserves elapsed time and cumulative usage across approval pauses and runtime restarts", async (t) => {
        t.mock.timers.enable({ apis: ["Date"], now: 1_000 });
        const { store } = await createSliceWorkspace("ws-approval-telemetry");
        const tools = new BrowserToolRegistry(store);
        let round = 0;
        const inference: InferenceAdapter = {
            async *streamChat(req) {
                round += 1;
                t.mock.timers.tick(100);
                const chunk: NormalizedStreamChunk = {
                    providerId: req.config.providerId,
                    model: req.config.model,
                    deltaText: "",
                    toolCalls: round < 3 ? [{
                        id: `delete-${round}`,
                        name: "workspace.delete_file",
                        argumentsJson: JSON.stringify({ path: `unused-${round}.txt` }),
                    }] : [],
                    usage: { promptTokens: round * 10, completionTokens: round, totalTokens: round * 11 },
                    finishReason: round < 3 ? "tool_calls" : "stop",
                };
                // Repeated cumulative usage chunks count once per request.
                yield chunk;
                yield { ...chunk, deltaText: "Done." };
            },
        };
        const createRuntime = () => new AgentKernelRuntime(
            store, inference, tools, tools.getToolDefinitions(),
        );
        assert.equal(await createRuntime().startRun("run-metrics", "Clean unused files", TEST_CONFIG), "awaiting_approval");
        t.mock.timers.tick(300);
        assert.equal(await createRuntime().resolveApproval("run-metrics", true), "awaiting_approval");
        t.mock.timers.tick(500);
        assert.equal(await createRuntime().resolveApproval("run-metrics", false), "completed");
        const completed = (await store.getJournal()).at(-1)!;
        assert.equal(completed.payload.durationMs, 1_100);
        assert.deepEqual(completed.payload.usage, { promptTokens: 60, completionTokens: 6, totalTokens: 66 });
    });

    it("enforces the classifier deadline even when cancellation is ignored", async () => {
        let classifierSignal: AbortSignal | undefined;
        const router = new VelesHarnessRouter(undefined, async (_prompt, _paths, _profiles, signal) => {
            classifierSignal = signal;
            return new Promise(() => {});
        }, 10);
        let watchdog: ReturnType<typeof setTimeout> | undefined;
        try {
            const result = await Promise.race([
                router.routeWithOptionalClassifier("Fix TypeError", []),
                new Promise<never>((_resolve, reject) => {
                    watchdog = setTimeout(() => reject(new Error("Classifier deadline was not enforced")), 200);
                }),
            ]);
            assert.equal(result.source, "tier0_deterministic");
            assert.equal(result.fallbackReason, "timeout");
            assert.equal(result.profiles[0]?.id, "preview-debugger");
            assert.equal(classifierSignal?.aborted, true);
        } finally {
            clearTimeout(watchdog);
        }
    });

    it("retries an initial routing timeout once before caching the recovered route", async () => {
        const { store } = await createSliceWorkspace("ws-route-recover");
        let classified = 0;
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round < 3 ? "tool_calls" : "stop",
                    toolCalls: round < 3 ? [{ id: `route-${round}`, name: "noop", argumentsJson: JSON.stringify({ round }) }] : [],
                };
            },
        }, { currentGatewayInstanceId: "test", async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; } },
        [{ name: "noop", description: "Keep paths unchanged", effectClass: "pure_read" }], {
            multiTurnWorkspaceContext: true,
            classifier: async () => {
                classified += 1;
                if (classified === 1) return new Promise(() => {});
                return { status: "matched", specialistIds: ["svg-multimodal-designer"] };
            },
        });
        const config = runtime.getServiceConfig();
        runtime.setServiceConfig({ ...config, decision: { ...config.decision, timeoutMs: 10 } });
        await runtime.startRun("route-recover", "Create an editable SVG", TEST_CONFIG);
        assert.equal(classified, 2);
        const routes = (await store.getJournal()).filter((r) => r.payload.harnessRouting).map((r) => r.payload.harnessRouting as Record<string, unknown>);
        assert.equal(routes[0]!.fallbackReason, "timeout");
        assert.deepEqual(routes.map((route) => route.source), ["tier0_deterministic", "tier1_classifier", "tier1_classifier"]);
        assert.deepEqual(routes.map((route) => route.cacheHit), [false, false, true]);
        assert.deepEqual(routes.map((route) => route.attemptCount), [1, 2, 2]);
    });

    it("bounds repeated routing timeouts to two attempts for an unchanged context", async () => {
        const { store } = await createSliceWorkspace("ws-route-timeout-bound");
        let classified = 0;
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round < 4 ? "tool_calls" : "stop",
                    toolCalls: round < 4 ? [{ id: `bound-${round}`, name: "noop", argumentsJson: JSON.stringify({ round }) }] : [],
                };
            },
        }, { currentGatewayInstanceId: "test", async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; } },
        [{ name: "noop", description: "Keep paths unchanged", effectClass: "pure_read" }], {
            multiTurnWorkspaceContext: true,
            classifier: async () => { classified += 1; return new Promise(() => {}); },
        });
        const config = runtime.getServiceConfig();
        runtime.setServiceConfig({ ...config, decision: { ...config.decision, timeoutMs: 10 } });
        await runtime.startRun("route-timeout-bound", "Create an SVG", TEST_CONFIG);
        assert.equal(classified, 2, "The fallback must not trigger paid classification on every tool round");
        const routes = (await store.getJournal()).filter((r) => r.payload.harnessRouting).map((r) => r.payload.harnessRouting as Record<string, unknown>);
        assert.deepEqual(routes.map((route) => route.cacheHit), [false, false, true, true]);
        assert.deepEqual(routes.map((route) => route.attemptCount), [1, 2, 2, 2]);
    });

    it("invalidates routing when decision settings change and preserves unrelated setting cache hits", async () => {
        const changes: readonly Partial<ServiceCapabilitiesConfigSpec["decision"]>[] = [
            { timeoutMs: 900 },
            { endpointUrl: "https://other.example.test/decisions" },
            { model: "other-classifier" },
            { apiKey: "replacement-test-key" },
            { threshold: 0.9 },
            { mode: "openai-decisions" },
        ];
        for (const change of changes) {
            const { store } = await createSliceWorkspace("ws-route-settings");
            let classified = 0;
            let round = 0;
            const runtime = new AgentKernelRuntime(store, {
                async *streamChat(req) {
                    round += 1;
                    const config = runtime.getServiceConfig();
                    if (round === 1) runtime.setServiceConfig({ ...config, imageMode: "local-deterministic-svg" });
                    if (round === 2) runtime.setServiceConfig({ ...config, decision: { ...config.decision, ...change } });
                    yield {
                        providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                        finishReason: round < 3 ? "tool_calls" : "stop",
                        toolCalls: round < 3 ? [{ id: `settings-${round}`, name: "noop", argumentsJson: JSON.stringify({ round }) }] : [],
                    };
                },
            }, { currentGatewayInstanceId: "test", async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; } },
            [{ name: "noop", description: "Keep paths unchanged", effectClass: "pure_read" }], {
                multiTurnWorkspaceContext: true,
                classifier: async () => {
                    classified += 1;
                    return { status: "matched", specialistIds: [classified === 1 ? "svg-multimodal-designer" : "diagram-design"] };
                },
            });
            await runtime.startRun("route-settings", "Draw a diagram", TEST_CONFIG);
            assert.equal(classified, 2, `Routing must refresh after changing ${Object.keys(change)[0]}`);
            const routes = (await store.getJournal()).filter((r) => r.payload.harnessRouting).map((r) => r.payload.harnessRouting as Record<string, unknown>);
            assert.deepEqual(routes.map((route) => route.cacheHit), [false, true, false]);
            assert.deepEqual(routes.map((route) => route.attemptCount), [1, 1, 1]);
            assert.deepEqual(routes[2]!.specialistIds, ["diagram-design"]);
        }
    });

    it("does not refill the routing cache with an obsolete in-flight settings result", async () => {
        const { store } = await createSliceWorkspace("ws-route-settings-race");
        let enter!: () => void;
        let release!: () => void;
        const entered = new Promise<void>((done) => { enter = done; });
        const held = new Promise<void>((done) => { release = done; });
        let classified = 0;
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? [{ id: "settings-race", name: "noop", argumentsJson: "{}" }] : [],
                };
            },
        }, { currentGatewayInstanceId: "test", async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; } },
        [{ name: "noop", description: "Keep paths unchanged", effectClass: "pure_read" }], {
            multiTurnWorkspaceContext: true,
            classifier: async () => {
                classified += 1;
                if (classified === 1) { enter(); await held; }
                return { status: "matched", specialistIds: [classified === 1 ? "svg-multimodal-designer" : "diagram-design"] };
            },
        });
        const run = runtime.startRun("route-settings-race", "Draw a diagram", TEST_CONFIG);
        try {
            await entered;
            const config = runtime.getServiceConfig();
            runtime.setServiceConfig({ ...config, decision: { ...config.decision, threshold: 0.9 } });
        } finally { release(); }
        await run;
        assert.equal(classified, 2, "An old in-flight response must not repopulate the new-settings cache");
    });

    it("routes vector, raster, app, diagram, and article tasks without adding unrelated game guidance", () => {
        const router = new VelesHarnessRouter();
        const cases: readonly [string, readonly string[], readonly string[]][] = [
            ["create 4 images of a horse and then create simple horse tinder app", [], ["raster-image-designer", "preview-app-builder"]],
            ['change the color of the subject on this picture "https://github.com/yavuzceliker/sample-images/blob/main/docs/image-100.jpg"', [], ["raster-image-designer"]],
            ["Edit assets/mark.svg to simplify its paths", ["assets/mark.svg"], ["svg-multimodal-designer"]],
            ["Draw an SVG architecture diagram", [], ["diagram-design", "svg-multimodal-designer"]],
            ["Write a technical article with architecture diagrams explaining transactions", [], ["concept-article", "diagram-design"]],
            ["Use concept-article for an explanation of browser storage", [], ["concept-article"]],
            ["Create an interactive explanation with a scatter plot", [], ["concept-article", "data-script-compute"]],
            ["Fix the form in preview/index.html", ["preview/index.html"], ["preview-debugger", "preview-app-builder"]],
            ["Build a playable Mario game with keyboard controls", [], ["canvas-game-builder"]],
            ["Correct this paragraph", [], []],
        ];
        for (const [prompt, paths, expected] of cases) {
            assert.deepEqual(router.selectTier0Capsules(prompt, paths).map((p) => p.id), expected, prompt);
        }
    });

    it("exports each inference round's classifier source and fallback without service credentials", async () => {
        const { store } = await createSliceWorkspace("ws-routing-evidence");
        const tools = new BrowserToolRegistry(store);
        let rounds = 0;
        let classified = 0;
        const inference: InferenceAdapter = {
            async *streamChat(req) {
                rounds += 1;
                assert.match(String(req.messages[0]?.content), /Editable SVG & Vector Designer/);
                assert.doesNotMatch(String(req.messages[0]?.content), /HTML5 Canvas & Game Builder/);
                yield {
                    providerId: req.config.providerId, model: req.config.model,
                    deltaText: "Done", usage: null, finishReason: rounds === 1 ? "tool_calls" : "stop",
                    toolCalls: rounds === 1 ? [{
                        id: "save-vector", name: "workspace.write_file",
                        argumentsJson: JSON.stringify({ path: "assets/mark.svg", content: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"/>' }),
                    }] : [],
                };
            },
        };
        const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions(), {
            classifier: async () => ++classified === 1
                ? { status: "matched", specialistIds: ["svg-multimodal-designer"] }
                : { status: "undecided", reason: "below_threshold" },
        });
        runtime.setServiceConfig({
            ...runtime.getServiceConfig(),
            decision: { mode: "typesafe", endpointUrl: "https://api.typesafe.ai/v1/systemone", model: "jev-1.14.0", apiKey: "private-test-credential" },
        });
        assert.equal(await runtime.startRun("run-vector", "Create an editable SVG logo", TEST_CONFIG, {
            sessionId: "session-main", allowedTools: ["workspace.write_file"], unrestrictedTools: false,
        }), "completed");
        const exported = await store.exportWorkspaceBundle();
        const routing = exported.journal.filter((r) => r.payload.harnessRouting).map((r) => r.payload.harnessRouting);
        assert.deepEqual(routing, [
            { source: "tier1_classifier", specialistIds: ["svg-multimodal-designer"], configuredMode: "typesafe", cacheHit: false, attemptCount: 1 },
            { source: "tier0_deterministic", specialistIds: ["svg-multimodal-designer"], configuredMode: "typesafe", fallbackReason: "below_threshold", cacheHit: false, attemptCount: 1 },
        ]);
        assert.equal(rounds, 2);
        assert.ok(!JSON.stringify(exported).includes("private-test-credential"));
        assert.ok((await store.readFileText("assets/mark.svg")).includes("<svg"));
    });

    it("VelesHarnessRouter falls back deterministically to Tier-0 when optional classifier returns undecided or times out", async () => {
        const timeoutRouter = new VelesHarnessRouter(
            undefined,
            async (_prompt, _paths, _profiles, signal) => {
                await new Promise((resolve) => setTimeout(resolve, 100));
                if (signal.aborted) throw new Error("Classifier timed out");
                return { status: "undecided", reason: "low_confidence" };
            },
            20,
        );

        const res = await timeoutRouter.routeWithOptionalClassifier(
            "Fix TypeError in preview/index.html",
            ["preview/index.html"],
        );
        assert.equal(res.source, "tier0_deterministic");
        assert.equal(res.fallbackReason, "timeout");
        assert.equal(res.profiles[0]?.id, "preview-debugger");

        const undecidedRouter = new VelesHarnessRouter(undefined, async () => ({
            status: "undecided", reason: "low_confidence",
        }));
        assert.equal((await undecidedRouter.routeWithOptionalClassifier("Fix error", [])).fallbackReason, "low_confidence");
        const matchedRouter = new VelesHarnessRouter(undefined, async () => ({
            status: "matched", specialistIds: ["svg-multimodal-designer"],
        }));
        assert.equal((await matchedRouter.routeWithOptionalClassifier("Fix error", [])).profiles[0]?.id, "svg-multimodal-designer");
        const rejectedRouter = new VelesHarnessRouter(undefined, async () => {
            throw new Error("network unavailable");
        });
        assert.equal((await rejectedRouter.routeWithOptionalClassifier("Fix error", [])).fallbackReason, "classifier_error");
        assert.equal((await new VelesHarnessRouter().routeWithOptionalClassifier("Fix error", [])).fallbackReason, "not_configured");
        const unknownRouter = new VelesHarnessRouter(undefined, async () => ({ status: "matched", specialistIds: ["missing-profile"] }));
        assert.equal((await unknownRouter.routeWithOptionalClassifier("Fix error", [])).fallbackReason, "unknown_specialist");
    });

    it("PreviewSandboxHost fences diagnostics by both (generation, workspaceSeq) so stale errors are ignored", () => {
        let listener = null as ((ev: MessageEventLike) => void) | null;
        const contentWindow = { postMessage() { } };
        const env: SandboxEnvironmentLike = {
            createIframe(): SandboxIframeLike {
                return {
                    sandbox: "",
                    srcdoc: "",
                    contentWindow,
                    remove() { },
                };
            },
            addMessageListener(h) {
                listener = h;
                return () => {
                    listener = null;
                };
            },
        };

        const received: PreviewRuntimeDiagnostic[] = [];
        const host = new PreviewSandboxHost(env, (d) => received.push(d));
        const gen = host.mountPreview({
            entryHtml: "<!doctype html><script src='script.js'></script>",
            entryPath: "preview/index.html",
            workspaceSeq: 12,
            files: { "preview/script.js": "throw new Error('boom');" },
        });
        assert.equal(gen, 1);

        // Diagnostic from stale workspaceSeq (#11 !== #12) is rejected
        listener?.({
            source: contentWindow,
            origin: "null",
            data: {
                channel: "veles:preview",
                generation: 1,
                workspaceSeq: 11,
                type: "preview_error",
                message: "Stale error from seq 11",
            },
        });
        assert.equal(received.length, 0);

        // Diagnostic from current (generation=1, workspaceSeq=12) is accepted
        listener?.({
            source: contentWindow,
            origin: "null",
            data: {
                channel: "veles:preview",
                generation: 1,
                workspaceSeq: 12,
                type: "preview_error",
                message: "Active error at seq 12",
            },
        });
        assert.equal(received.length, 1);
        assert.equal(received[0]?.workspaceSeq, 12);
        host.dispose();
    });

    it("enforces origin-scoped apiKey forwarding and parent allowedTools ceiling in agent.delegate", async () => {
        const { store } = await createSliceWorkspace("ws-sec-delegate");
        const tools = new BrowserToolRegistry(store);
        const capturedConfigs: ProviderModelConfig[] = [];
        const inference: InferenceAdapter = {
            async *streamChat(req) {
                capturedConfigs.push(req.config);
                yield {
                    providerId: req.config.providerId,
                    model: req.config.model,
                    deltaText: "Child finished.",
                    toolCalls: [],
                    usage: null,
                    finishReason: "stop",
                };
            },
        };
        const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions());
        const manager = new AgentManager(store, runtime);
        for (const t of createSubagentTools(manager)) {
            tools.registerTool(t);
            runtime.registerToolDefinition(t.spec);
        }
        await manager.createSession({
            id: "session-restricted",
            workspaceId: "ws-sec-delegate",
            model: { ...TEST_CONFIG, apiKey: "sk-parent-secret" },
            allowedTools: ["workspace.read_file", "agent.delegate"],
            status: "idle",
        });

        await tools.executeTool({
            operationId: "op-del-1",
            runId: "run-del-1",
            toolName: "agent.delegate",
            effectClass: "idempotent_write",
            args: {
                task: "Inspect only",
                sessionId: "child-untrusted",
                parentSessionId: "session-restricted",
                baseUrl: "https://untrusted.attacker.example/v1",
                allowedTools: ["workspace.read_file", "workspace.write_file"],
            },
        });
        const child = manager.getSession("child-untrusted");
        assert.equal(child?.model.apiKey, undefined);
        assert.equal(capturedConfigs[0]?.apiKey, undefined);
        assert.deepEqual(child?.allowedTools, ["workspace.read_file"]);
    });

    it("scopes exportWorkspaceBundle blobs, recovers unfinished parent runs across child completion, and normalizes Gateway /v1 prefix", async () => {
        const { store } = await createSliceWorkspace("ws-alpha-exp");
        await store.writeWorkspaceFile("alpha.txt", "only-in-alpha");
        await store.switchWorkspace("ws-beta-exp", "Beta");
        await store.writeWorkspaceFile("beta.txt", "only-in-beta");
        const betaBundle = await store.exportWorkspaceBundle();
        assert.equal(Object.keys(betaBundle.blobsBase64).length, 1);

        // Parent run in 'inferring' while child run completes last
        await store.appendEvent({
            eventId: "p-1", runId: "run-parent", type: "RunCreated", nextState: "created", payload: { prompt: "parent" },
        });
        await store.appendEvent({
            eventId: "p-2", runId: "run-parent", type: "RunPreparing", nextState: "preparing", payload: {},
        });
        await store.appendEvent({
            eventId: "p-3", runId: "run-parent", type: "CheckpointCommitted", nextState: "inferring", payload: {},
        });
        await store.appendEvent({
            eventId: "c-1", runId: "run-child", type: "RunCreated", nextState: "created", payload: { prompt: "child" },
        });
        await store.appendEvent({
            eventId: "c-2", runId: "run-child", type: "RunPreparing", nextState: "preparing", payload: {},
        });
        await store.appendEvent({
            eventId: "c-3", runId: "run-child", type: "CheckpointCommitted", nextState: "inferring", payload: {},
        });
        await store.appendEvent({
            eventId: "c-4", runId: "run-child", type: "RunCompleted", nextState: "completed", payload: { assistantText: "ok" },
        });

        const tools = new BrowserToolRegistry(store);
        const runtime = new AgentKernelRuntime(store, { async *streamChat() {} }, tools, tools.getToolDefinitions());
        const decision = await runtime.recoverUnfinishedRun("gw-1");
        assert.equal(decision?.nextState, "interrupted");

        const requestedUrls: string[] = [];
        const gw = new GatewayToolExecutionAdapter(
            {
                baseUrl: "/v1",
                defaultScope: { allowedTools: [], allowNetwork: false, writablePaths: [] },
                defaultLimits: { timeoutMs: 1000, maxOutputBytes: 1024 },
            },
            async (url) => {
                requestedUrls.push(url);
                return { ok: true, status: 200, json: async () => ({ toolDefinitions: [] }) };
            },
        );
        await gw.fetchCapabilities();
        assert.equal(requestedUrls[0], "/v1/capabilities");
    });

    it("routes Decision API modes (typesafe, openai-decisions, local-laya, tier0-only) and honors imageMode + webSearchMode in serviceConfig", async () => {
        const redacted = redactClassifierSecrets(
            "Use key sk-1234567890abcdefghijklmnop for test",
        );
        assert.ok(redacted.includes("<REDACTED>"));
        assert.ok(!redacted.includes("sk-1234567890"));

        let activeServiceCfg: ServiceCapabilitiesConfigSpec = {
            decision: {
                mode: "tier0-only",
                endpointUrl: "https://api.typesafe.ai/v1/jev/decide",
                model: "jev-1.14.0",
                threshold: 0.75,
                timeoutMs: 800,
            },
            imageMode: "local-deterministic-svg",
            webSearchMode: "offline-disabled",
        };

        const originalFetch = globalThis.fetch;
        const capturedRequests: Array<{ url: string; body: Record<string, unknown> }> = [];
        globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
            const url = String(input);
            const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {};
            capturedRequests.push({ url, body });
            if (url.includes("/systemone")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        answers: {
                            "svg-multimodal-designer": { type: "noul", noul: 0.94 },
                            "preview-debugger": { type: "noul", noul: 0.1 },
                            "data-script-compute": { type: "noul", noul: 99 },
                        },
                    }),
                } as Response;
            }
            if (url.includes("/decisions")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        answers: [
                            { type: "predicate", name: "preview-debugger", probability: "0.91" },
                        ],
                    }),
                } as Response;
            }
            if (url.endsWith("/responses") && Array.isArray(body.input)) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        output: [{ type: "image_generation_call", result: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" }],
                    }),
                } as Response;
            }
            if (url.endsWith("/images/generations")) {
                if ("response_format" in body) {
                    return {
                        ok: false,
                        status: 400,
                        text: async () => JSON.stringify({ error: { code: "unknown_parameter", message: "Unknown parameter: 'response_format'." } }),
                    } as Response;
                }
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        data: [{ b64_json: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" }],
                    }),
                } as Response;
            }
            return { ok: false, status: 500, json: async () => ({}) } as Response;
        }) as typeof globalThis.fetch;

        try {
            const adapter = createDecisionClassifierAdapter(() => ({
                decision: activeServiceCfg.decision,
                chatApiKey: "sk-chat-fallback-key-1234567890123456",
            }));

            // 1. tier0-only returns undecided immediately without calling fetch
            const resTier0 = await adapter(
                "draw an icon",
                [],
                DEFAULT_BROWSER_SPECIALISTS,
                new AbortController().signal,
            );
            assert.equal(resTier0.status, "undecided");
            assert.equal(capturedRequests.length, 0);

            // 2. TypeSafe System One; migrate saved legacy endpoint configurations.
            activeServiceCfg = {
                ...activeServiceCfg,
                decision: {
                    mode: "typesafe",
                    endpointUrl: "https://api.typesafe.ai/v1/jev/decide",
                    model: "jev-1.14.0",
                    apiKey: "jev-key-123",
                    threshold: 0.75,
                },
            };
            const resTypesafe = await adapter(
                "create an editable SVG icon",
                ["assets/mark.svg"],
                DEFAULT_BROWSER_SPECIALISTS,
                new AbortController().signal,
            );
            assert.deepEqual(resTypesafe, {
                status: "matched",
                specialistIds: ["svg-multimodal-designer"],
            });
            assert.equal(capturedRequests[0]?.url, "https://api.typesafe.ai/v1/systemone");
            assert.match(String(capturedRequests[0]?.body.state), /editable SVG icon/);
            const questions = capturedRequests[0]?.body.questions as Record<string, { type: string; instructions: string }>;
            assert.equal(questions["svg-multimodal-designer"]?.type, "noul");
            assert.ok(questions["svg-multimodal-designer"]?.instructions.includes("SVG"));
            assert.equal(capturedRequests[0]?.body.predicates, undefined);

            // 3. openai-decisions using fallback chatApiKey
            activeServiceCfg = {
                ...activeServiceCfg,
                decision: {
                    mode: "openai-decisions",
                    endpointUrl: "https://api.openai.com/v1/decisions",
                    model: "gpt-6-luna",
                    threshold: 0.75,
                },
            };
            const resOpenAI = await adapter(
                "fix uncaught referenceerror",
                ["preview/index.html"],
                DEFAULT_BROWSER_SPECIALISTS,
                new AbortController().signal,
            );
            assert.deepEqual(resOpenAI, {
                status: "matched",
                specialistIds: ["preview-debugger"],
            });

            // 4. local-laya works without requiring an API key
            activeServiceCfg = {
                ...activeServiceCfg,
                decision: {
                    mode: "local-laya",
                    endpointUrl: "http://127.0.0.1:11434/v1/decisions",
                    model: "laya-latest",
                    threshold: 0.75,
                },
            };
            const resLaya = await adapter(
                "fix crash",
                [],
                DEFAULT_BROWSER_SPECIALISTS,
                new AbortController().signal,
            );
            assert.deepEqual(resLaya, {
                status: "matched",
                specialistIds: ["preview-debugger"],
            });

            // 5. Verify imageMode="local-deterministic-svg" and webSearchMode="offline-disabled" skip network calls
            const { store } = await createSliceWorkspace("ws-svc-modes");
            const countBeforeTools = capturedRequests.length;
            const tools = new BrowserToolRegistry(
                store,
                createDefaultWorkspaceTools(
                    () => ({
                        providerId: "openai",
                        baseUrl: "https://api.openai.com/v1",
                        model: "gpt-4o",
                        apiKey: "sk-live-key",
                    }),
                    () => activeServiceCfg,
                ),
            );
            const imgRes = await tools.executeTool({
                runId: "r-svc-1",
                operationId: "op-img",
                toolName: "image.generate",
                effectClass: "idempotent_write",
                args: { outputPath: "assets/local.svg", prompt: "castle" },
            });
            assert.ok(String(imgRes.fileUpdates?.["assets/local.svg"]).startsWith("<svg"));
            const searchRes = await tools.executeTool({
                runId: "r-svc-1",
                operationId: "op-web",
                toolName: "web.search",
                effectClass: "pure_read",
                args: { query: "slovenia weather" },
            });
            assert.ok(searchRes.outputText.includes("[Offline Web Search Fallback]"));
            assert.equal(capturedRequests.length, countBeforeTools);

            // 6. Verify image.transform in openai-dalle3 mode passes sourceDataUrl + clean in-place edit instruction to /responses, and image.generate retries without response_format on HTTP 400
            activeServiceCfg = { ...activeServiceCfg, imageMode: "openai-dalle3" };
            await store.writeWorkspaceFile("assets/horse.png", new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
            const txRes = await tools.executeTool({
                runId: "r-svc-2",
                operationId: "op-tx-dalle",
                toolName: "image.transform",
                effectClass: "idempotent_write",
                args: {
                    sourcePath: "assets/horse.png",
                    outputPath: "assets/horse-rainbow.png",
                    prompt: "change the color of the subject to rainbow",
                },
            });
            assert.match(txRes.outputText, /openai:responses\.image_generation\(edit\)/);
            const editReq = capturedRequests.at(-1)!;
            assert.equal(editReq.url, "https://api.openai.com/v1/responses");
            const editInputText = JSON.stringify(editReq.body.input);
            assert.match(editInputText, /Edit the attached source image in-place: change the color of the subject to rainbow/);
            assert.doesNotMatch(editInputText, /flux-4b-lora/);
            assert.doesNotMatch(editInputText, /\[source:/);

            const genRetryRes = await tools.executeTool({
                runId: "r-svc-2",
                operationId: "op-gen-retry",
                toolName: "image.generate",
                effectClass: "idempotent_write",
                args: { outputPath: "assets/gen.png", prompt: "a brown horse" },
            });
            assert.match(genRetryRes.outputText, /openai:dall-e-3/);
        } finally {
            globalThis.fetch = originalFetch;
        }
    });

    it("bridges opt-in sensors into <iframe sandbox='allow-scripts'>, resolves scripts/script.js in srcdoc, and supports SearXNG + SSRF-guarded web.fetch_url", async () => {
        const inbox: unknown[] = [];
        const contentWindow = {
            postMessage(msg: unknown) {
                inbox.push(msg);
            },
        };
        const env: SandboxEnvironmentLike = {
            createIframe(): SandboxIframeLike {
                return {
                    sandbox: "",
                    srcdoc: "",
                    contentWindow,
                    remove() {},
                };
            },
            addMessageListener() {
                return () => {};
            },
        };
        const host = new PreviewSandboxHost(env);
        host.mountPreview({
            entryHtml: "<!doctype html><script>window.veles.sensors.onUpdate(() => {});</script>",
            entryPath: "preview/index.html",
            workspaceSeq: 1,
            files: {},
        });
        inbox.length = 0; // Clear initial veles:host-init message

        // Opt-in sensor forwarding
        host.pushSensorSample({ tiltX: 12, tiltY: -4, pointerX: 0.5, pointerY: -0.5, timestampMs: 100 });
        assert.equal(inbox.length, 0, "Sensors must not forward when disabled");
        host.setSensorsEnabled(true);
        host.pushSensorSample({ tiltX: 12, tiltY: -4, pointerX: 0.5, pointerY: -0.5, timestampMs: 200 });
        assert.equal(inbox.length, 1);
        assert.equal((inbox[0] as Record<string, unknown>).channel, "veles:host-sensor");
        host.dispose();

        // Verify <script src="script.js"> inside preview/index.html resolves scripts/script.js
        const pongSrcdoc = compileMultiFileSrcdoc({
            entryHtml: '<!doctype html><html><head><link rel="stylesheet" href="style.css"></head><body><script src="script.js"></script></body></html>',
            entryPath: "preview/index.html",
            workspaceSeq: 15,
            files: {
                "preview/style.css": "canvas { border: 1px solid #fff; }",
                "scripts/script.js": "window.veles.sensors.onUpdate(() => {});",
            },
        });
        assert.ok(pongSrcdoc.includes('data-veles-path="scripts/script.js"'));
        assert.ok(!pongSrcdoc.includes('<script src="script.js">'));

        // 2. SearXNG web.search and SSRF-guarded web.fetch_url
        const { store } = await createSliceWorkspace("ws-searx-fetch");
        const originalFetch = globalThis.fetch;
        globalThis.fetch = (async (input: string | URL | Request) => {
            const url = String(input);
            if (url.includes("searx.example.test/search")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        results: [
                            {
                                title: "Perun WebRTC Docs",
                                url: "https://perun.example.test/docs",
                                content: "DataChannel 16 KiB limit and room signaling.",
                            },
                        ],
                    }),
                } as Response;
            }
            return {
                ok: true,
                status: 200,
                headers: new Headers({ "content-type": "text/html; charset=utf-8" }),
                text: async () => "<html><body><h1>Hello Veles</h1><script>evil()</script><p>Clean body text.</p></body></html>",
            } as Response;
        }) as typeof globalThis.fetch;

        try {
            const tools = new BrowserToolRegistry(
                store,
                createDefaultWorkspaceTools(
                    () => TEST_CONFIG,
                    () => ({
                        decision: {
                            mode: "tier0-only",
                            endpointUrl: "",
                            model: "jev-1.14.0",
                            threshold: 0.75,
                        },
                        imageMode: "local-deterministic-svg",
                        webSearchMode: "searxng",
                        searxngUrl: "https://searx.example.test/search",
                    }),
                ),
            );
            const searxRes = await tools.executeTool({
                runId: "r-sx-1",
                operationId: "op-sx",
                toolName: "web.search",
                effectClass: "pure_read",
                args: { query: "perun webrtc" },
            });
            assert.ok(searxRes.outputText.includes("Perun WebRTC Docs"));

            const fetchRes = await tools.executeTool({
                runId: "r-sx-1",
                operationId: "op-fetch-ok",
                toolName: "web.fetch_url",
                effectClass: "pure_read",
                args: { url: "https://perun.example.test/docs", saveToPath: "notes/perun.txt" },
            });
            assert.ok(fetchRes.outputText.includes("Hello Veles"));
            assert.ok(!fetchRes.outputText.includes("evil()"));
            assert.ok(String(fetchRes.fileUpdates?.["notes/perun.txt"]).includes("Clean body text."));

            await assert.rejects(
                () =>
                    tools.executeTool({
                        runId: "r-sx-1",
                        operationId: "op-fetch-ssrf",
                        toolName: "web.fetch_url",
                        effectClass: "pure_read",
                        args: { url: "http://127.0.0.1:11434/api/tags" },
                    }),
                /ERR_FETCH_PRIVATE_HOST_BLOCKED/,
            );
        } finally {
            globalThis.fetch = originalFetch;
        }
    });

    it("normalizes GitHub /blob/ URLs, saves binary images without token bloat, runs workspace.run_script for OldFaithful.csv averages, and syncs window.veles.kv", async () => {
        assert.equal(
            normalizePublicFetchUrl("https://github.com/gchoi/Dataset/blob/master/OldFaithful.csv"),
            "https://raw.githubusercontent.com/gchoi/Dataset/master/OldFaithful.csv",
        );

        const { store } = await createSliceWorkspace("ws-compute-kv");
        const jpegBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46]);
        const csvSample = [
            "rownames,TimeEruption,TimeWaiting",
            "1,3.6,79",
            "2,1.8,54",
            "3,3.333,74",
            "4,2.283,62",
        ].join("\n");

        const originalFetch = globalThis.fetch;
        globalThis.fetch = (async (input: string | URL | Request) => {
            const url = String(input);
            if (url === "https://raw.githubusercontent.com/yavuzceliker/sample-images/main/docs/image-100.jpg") {
                return {
                    ok: true,
                    status: 200,
                    headers: new Headers({ "content-type": "image/jpeg" }),
                    arrayBuffer: async () => jpegBytes.buffer,
                } as Response;
            }
            if (url === "https://raw.githubusercontent.com/gchoi/Dataset/master/OldFaithful.csv") {
                return {
                    ok: true,
                    status: 200,
                    headers: new Headers({ "content-type": "text/csv; charset=utf-8" }),
                    text: async () => csvSample,
                } as Response;
            }
            throw new TypeError("Failed to fetch");
        }) as typeof globalThis.fetch;

        try {
            const tools = new BrowserToolRegistry(store);

            // 1. Binary JPEG via GitHub /blob/ URL saves exact Uint8Array and returns compact summary
            const imgOut = await tools.executeTool({
                runId: "r-comp-1",
                operationId: "op-jpg",
                toolName: "web.fetch_url",
                effectClass: "pure_read",
                args: {
                    url: "https://github.com/yavuzceliker/sample-images/blob/main/docs/image-100.jpg",
                    saveToPath: "assets/image-100.jpg",
                },
            });
            assert.ok(imgOut.outputText.includes("Fetched binary file"));
            assert.ok(imgOut.outputText.length < 200);
            assert.deepEqual(imgOut.fileUpdates?.["assets/image-100.jpg"], jpegBytes);

            // 2. OldFaithful.csv via GitHub /blob/ URL + workspace.run_script to calculate averages
            const csvOut = await tools.executeTool({
                runId: "r-comp-1",
                operationId: "op-csv",
                toolName: "web.fetch_url",
                effectClass: "pure_read",
                args: {
                    url: "https://github.com/gchoi/Dataset/blob/master/OldFaithful.csv",
                    saveToPath: "assets/OldFaithful.csv",
                },
            });
            await store.writeWorkspaceFile(
                "assets/OldFaithful.csv",
                String(csvOut.fileUpdates?.["assets/OldFaithful.csv"] ?? ""),
            );

            const scriptOut = await tools.executeTool({
                runId: "r-comp-1",
                operationId: "op-script",
                toolName: "workspace.run_script",
                effectClass: "idempotent_write",
                args: {
                    code: `
                        const rows = readFile("assets/OldFaithful.csv").trim().split(/\\r?\\n/).slice(1);
                        let sumE = 0, sumW = 0;
                        for (const line of rows) {
                            const [, e, w] = line.split(",");
                            sumE += Number(e);
                            sumW += Number(w);
                        }
                        const stats = {
                            rows: rows.length,
                            avgTimeEruption: +(sumE / rows.length).toFixed(4),
                            avgTimeWaiting: +(sumW / rows.length).toFixed(4),
                        };
                        console.log("Computed OldFaithful stats:", stats);
                        return stats;
                    `,
                    saveResultToPath: "assets/stats.json",
                },
            });
            assert.ok(scriptOut.outputText.includes('"avgTimeEruption": 2.754'));
            assert.ok(scriptOut.outputText.includes('"avgTimeWaiting": 67.25'));
            assert.ok(scriptOut.outputText.includes("Wrote 'assets/stats.json'"));
            assert.ok(scriptOut.fileUpdates?.["assets/stats.json"]);
            assert.equal(scriptOut.fileUpdates?.["notes/veles_kv.json"], undefined);

            const writeOnlyOut = await tools.executeTool({
                runId: "r-comp-1",
                operationId: "op-script-write-only",
                toolName: "workspace.run_script",
                effectClass: "idempotent_write",
                args: {
                    code: `writeFile("preview/stats.json", [{ eruption: 3.6, waiting: 79 }]);`,
                },
            });
            assert.ok(writeOnlyOut.outputText.includes("Wrote 'preview/stats.json'"));
            assert.ok(writeOnlyOut.outputText.includes('"eruption": 3.6'));

            // 3. Verify PreviewSandboxHost persists preview-scoped window.veles.kv entries across mounts independently
            let listener = null as ((ev: MessageEventLike) => void) | null;
            const contentWindow = { postMessage() {} };
            const host = new PreviewSandboxHost({
                createIframe: () => ({ sandbox: "", srcdoc: "", contentWindow, remove() {} }),
                addMessageListener(h) {
                    listener = h;
                    return () => {
                        listener = null;
                    };
                },
            });
            const gen = host.mountPreview({
                entryHtml: "<!doctype html><script>window.veles.kv.set('theme', 'dark');</script>",
                entryPath: "preview/index.html",
                workspaceSeq: 5,
                files: {},
            });
            listener?.({
                source: contentWindow,
                origin: "null",
                data: {
                    channel: "veles:preview",
                    generation: gen,
                    workspaceSeq: 5,
                    type: "kv_set",
                    key: "theme",
                    value: "dark",
                },
            });
            assert.deepEqual(host.getKvSnapshot(), { theme: "dark" });
            host.dispose();
        } finally {
            globalThis.fetch = originalFetch;
        }
    });

    it("optimistically clears pendingApproval during slow multi-tool batches and guards preview entry/viewport sizing", async () => {
        // 1. Verify pickDefaultWorkspaceFile and resolvePreviewEntryPath prioritize preview/index.html over assets/*
        const ws5Files = {
            "assets/block.png": "data:image/png;base64,AAA",
            "assets/mario.png": "data:image/png;base64,BBB",
            "preview/index.html": "<!doctype html><canvas id='gameCanvas'></canvas><script src='script.js'></script>",
            "preview/script.js": "canvas.width = window.innerWidth; canvas.height = window.innerHeight * 0.9;",
        };
        assert.equal(pickDefaultWorkspaceFile(ws5Files), "preview/index.html");
        assert.equal(resolvePreviewEntryPath(ws5Files, "assets/block.png"), "preview/index.html");
        assert.equal(
            resolvePreviewEntryPath({ "assets/image-100.jpg": "data:image/jpeg;base64,CCC" }, "assets/image-100.jpg"),
            "assets/image-100.jpg",
        );

        // 2. Verify compileMultiFileSrcdoc injects non-zero window.innerWidth/innerHeight fallback guard
        const ws4Srcdoc = compileMultiFileSrcdoc({
            entryHtml: ws5Files["preview/index.html"],
            entryPath: "preview/index.html",
            workspaceSeq: 31,
            files: ws5Files,
        });
        assert.ok(ws4Srcdoc.includes('Object.defineProperty(window, "innerWidth"'));
        assert.ok(ws4Srcdoc.includes('Object.defineProperty(window, "innerHeight"'));

        // 3. Verify VelesAppClient.resolveApproval optimistically clears pendingApproval before slow follow-up tools finish
        const { store } = await createSliceWorkspace("ws-approval-race");
        await store.writeWorkspaceFile("assets/image-100-color.jpg", "old-image");

        let resolveSlowTransform = null as (() => void) | null;
        const slowTransformPromise = new Promise<void>((res) => {
            resolveSlowTransform = res;
        });

        let round = 0;
        const inf: InferenceAdapter = {
            async *streamChat(): AsyncIterable<NormalizedStreamChunk> {
                round += 1;
                if (round === 1) {
                    yield {
                        providerId: "openai",
                        model: "gpt-4o",
                        deltaText: "Deleting old image and transforming...",
                        toolCalls: [
                            {
                                id: "call_del_1",
                                name: "workspace.delete_file",
                                argumentsJson: JSON.stringify({ path: "assets/image-100-color.jpg" }),
                            },
                            {
                                id: "call_slow_2",
                                name: "workspace.write_file",
                                argumentsJson: JSON.stringify({ path: "assets/image-100-color.jpg", content: "new-image" }),
                            },
                        ],
                        usage: null,
                        finishReason: "tool_calls",
                    };
                    return;
                }
                yield {
                    providerId: "openai",
                    model: "gpt-4o",
                    deltaText: "Done!",
                    toolCalls: [],
                    usage: null,
                    finishReason: "stop",
                };
            },
        };

        const baseTools = new BrowserToolRegistry(store);
        const slowAdapter = {
            currentGatewayInstanceId: "gw-race-1",
            async executeTool(req: Parameters<typeof baseTools.executeTool>[0]) {
                if (req.toolName === "workspace.write_file") {
                    await slowTransformPromise;
                }
                return baseTools.executeTool(req);
            },
            async queryOperationStatus() {
                return { completed: false };
            },
        };
        const runtime = new AgentKernelRuntime(store, inf, slowAdapter, baseTools.getToolDefinitions());

        const uiListeners = new Set<(ev: { readonly data: unknown }) => void>();
        const workerListeners = new Set<(ev: { readonly data: unknown }) => void>();
        const uiPort: WorkerPortLike = {
            postMessage(msg) {
                for (const l of workerListeners) l({ data: msg });
            },
            addEventListener(_t, h) {
                uiListeners.add(h);
            },
            removeEventListener(_t, h) {
                uiListeners.delete(h);
            },
        };
        const workerPort: WorkerPortLike = {
            postMessage(msg) {
                for (const l of uiListeners) l({ data: msg });
            },
            addEventListener(_t, h) {
                workerListeners.add(h);
            },
            removeEventListener(_t, h) {
                workerListeners.delete(h);
            },
        };

        const workerHost = new KernelWorkerHost(store, runtime, workerPort);
        const client = new VelesAppClient(uiPort, TEST_CONFIG);
        await client.initWorkspace("Workspace 2");

        const snapPaused = await client.startRun("run-race-1", "delete and regenerate");
        assert.equal(snapPaused.runState, "awaiting_approval");
        assert.ok(snapPaused.pendingApproval !== null);

        // Start resolveApproval without awaiting yet (simulating the 22s DALL-E wait)
        const approvalPromise = client.resolveApproval("run-race-1", true);

        // Immediately after clicking Approve, pendingApproval MUST be null and runState MUST be executing_tool
        const immediateView = client.getStateView();
        assert.equal(immediateView.snapshot?.pendingApproval, null);
        assert.equal(immediateView.snapshot?.runState, "executing_tool");

        // Unblock the slow follow-up tool and verify clean completion
        resolveSlowTransform?.();
        const finalSnap = await approvalPromise;
        assert.equal(finalSnap.runState, "completed");
        assert.equal(finalSnap.pendingApproval, null);
        assert.equal(finalSnap.files["assets/image-100-color.jpg"], "data:image/jpeg;base64,bmV3LWltYWdl");

        client.dispose();
        workerHost.dispose();
    });

    it("enforces run_script isolation, IPv4-mapped IPv6 SSRF guards, caller delegation authority, delegatableTools recovery, workspace-scoped KV, and valid JS SVG string inlining", async () => {
        const { store } = await createSliceWorkspace("ws-sec-audit-all");
        const tools = new BrowserToolRegistry(store);

        // 1. [P1] workspace.run_script isolates execution from kernel globals and commits returned changes via store
        const isoOut = await tools.executeTool({
            runId: "r-iso-1",
            operationId: "op-iso-1",
            toolName: "workspace.run_script",
            effectClass: "idempotent_write",
            args: {
                code: `
                    const hasIdb = typeof indexedDB !== "undefined";
                    const hasNav = typeof navigator !== "undefined";
                    const hasProc = typeof process !== "undefined";
                    let escaped = false;
                    try {
                        const fn = readFile.constructor("return typeof process !== 'undefined'");
                        escaped = Boolean(fn());
                    } catch {
                        escaped = false;
                    }
                    writeFile("notes/isolated.txt", JSON.stringify({ hasIdb, hasNav, hasProc, escaped }));
                    return { hasIdb, hasNav, hasProc, escaped };
                `,
            },
        });
        assert.match(isoOut.outputText, /"hasIdb": false/);
        assert.match(isoOut.outputText, /"hasNav": false/);
        assert.match(isoOut.outputText, /"hasProc": false/);
        assert.match(isoOut.outputText, /"escaped": false/);
        assert.ok((await store.readFileText("notes/isolated.txt")).includes('"escaped":false'));

        // 2. [P1] Private-network & IPv4-mapped IPv6 loopback addresses are blocked
        assert.equal(isBlockedPrivateHost("[::ffff:127.0.0.1]"), true);
        assert.equal(isBlockedPrivateHost("[::ffff:7f00:1]"), true);
        assert.equal(isBlockedPrivateHost("[0:0:0:0:0:ffff:127.0.0.1]"), true);
        assert.equal(isBlockedPrivateHost("172.20.0.1"), true);
        assert.equal(isBlockedPrivateHost("[fc00::1]"), true);
        assert.equal(isBlockedPrivateHost("[fe80::1]"), true);
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-ssrf-v6",
                    operationId: "op-ssrf-v6",
                    toolName: "web.fetch_url",
                    effectClass: "pure_read",
                    args: { url: "http://[::ffff:127.0.0.1]:18080/secret" },
                }),
            /ERR_FETCH_PRIVATE_HOST_BLOCKED/,
        );

        // 3. [P1] Delegation caller authority cannot borrow session-main & [P2] delegatableTools survives recovery
        let childTurn = 0;
        const inference: InferenceAdapter = {
            async *streamChat(req) {
                if (req.config.model === "caller-model") {
                    childTurn += 1;
                    if (childTurn === 1 || childTurn === 3) {
                        yield {
                            providerId: "openai",
                            model: "caller-model",
                            deltaText: "Attempting to borrow session-main...",
                            toolCalls: [
                                {
                                    id: `call_borrow_${childTurn}`,
                                    name: "agent.delegate",
                                    argumentsJson: JSON.stringify({
                                        task: "Write secret",
                                        sessionId: `child-borrowed-${childTurn}`,
                                        parentSessionId: "session-main",
                                        model: "child-model",
                                        allowedTools: ["workspace.write_file"],
                                    }),
                                },
                            ],
                            usage: null,
                            finishReason: "tool_calls",
                        };
                        return;
                    }
                    yield {
                        providerId: "openai",
                        model: "caller-model",
                        deltaText: "Caller done.",
                        toolCalls: [],
                        usage: null,
                        finishReason: "stop",
                    };
                    return;
                }
                yield {
                    providerId: "openai",
                    model: "child-model",
                    deltaText: "Child has no tools.",
                    toolCalls: [],
                    usage: null,
                    finishReason: "stop",
                };
            },
        };

        const runtime = new AgentKernelRuntime(store, inference, tools, tools.getToolDefinitions());
        const manager = new AgentManager(store, runtime);
        for (const t of createSubagentTools(manager)) {
            tools.registerTool(t);
            runtime.registerToolDefinition(t.spec);
        }

        await manager.createSession({
            id: "session-main",
            workspaceId: "ws-sec-audit-all",
            model: TEST_CONFIG,
            allowedTools: ["workspace.write_file", "agent.delegate"],
            delegatableTools: ["workspace.write_file"],
            status: "idle",
        });
        await manager.createSession({
            id: "session-restricted-caller",
            workspaceId: "ws-sec-audit-all",
            model: { ...TEST_CONFIG, model: "caller-model" },
            allowedTools: ["agent.delegate"],
            delegatableTools: [],
            status: "idle",
        });

        await manager.run("session-restricted-caller", "Try delegating write via session-main");
        const child1 = manager.getSession("child-borrowed-1");
        assert.equal(child1?.parentSessionId, "session-restricted-caller");
        assert.deepEqual(child1?.allowedTools, []);

        // Recover sessions from journal and verify delegatableTools: [] is preserved
        await manager.recoverSessionsFromJournal();
        const recoveredCaller = manager.getSession("session-restricted-caller");
        assert.deepEqual(recoveredCaller?.delegatableTools, []);
        await manager.run("session-restricted-caller", "Try delegating again after recovery");
        const child2 = manager.getSession("child-borrowed-3");
        assert.equal(child2?.parentSessionId, "session-restricted-caller");
        assert.deepEqual(child2?.allowedTools, []);

        // 4. [P2] Preview KV storage is namespaced per workspace
        let listener = null as ((ev: MessageEventLike) => void) | null;
        const sentInitPayloads: Array<Record<string, unknown>> = [];
        const contentWindow = {
            postMessage(msg: unknown) {
                const obj = msg as Record<string, unknown>;
                if (obj.channel === "veles:host-init") sentInitPayloads.push(obj);
            },
        };
        const host = new PreviewSandboxHost({
            createIframe: () => ({ sandbox: "", srcdoc: "", contentWindow, remove() {} }),
            addMessageListener(h) {
                listener = h;
                return () => {
                    listener = null;
                };
            },
        });
        const genA = host.mountPreview({
            workspaceId: "ws-one",
            entryHtml: "<!doctype html>",
            workspaceSeq: 1,
            files: {},
        });
        listener?.({
            source: contentWindow,
            origin: "null",
            data: {
                channel: "veles:preview",
                generation: genA,
                workspaceSeq: 1,
                type: "kv_set",
                key: "marker",
                value: "ws-one-only",
            },
        });
        assert.deepEqual(host.getKvSnapshot("ws-one"), { marker: "ws-one-only" });
        host.mountPreview({
            workspaceId: "ws-two",
            entryHtml: "<!doctype html>",
            workspaceSeq: 1,
            files: {},
        });
        assert.deepEqual(host.getKvSnapshot("ws-two"), {});
        assert.deepEqual(sentInitPayloads.at(-1)?.kv, {});
        host.dispose();

        // 5. [P2] Single-quoted SVG image path inlining produces valid JS string literal
        const svgSrcdoc = compileMultiFileSrcdoc({
            entryHtml: "<!doctype html><html><body><script src='app.js'></script></body></html>",
            entryPath: "preview/index.html",
            workspaceSeq: 1,
            files: {
                "assets/sprite.svg": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><circle cx='5' cy='5' r='4'/></svg>",
                "preview/app.js": "const img = 'assets/sprite.svg'; console.log(img);",
            },
        });
        const scriptMatch = /<script data-veles-path="preview\/app\.js">([\s\S]*?)<\/script>/.exec(svgSrcdoc);
        assert.ok(scriptMatch?.[1]);
        assert.doesNotThrow(() => new Function(scriptMatch![1]!));

        // 6. Verify trailing CRLF in CSV does not poison split('\\n') with NaN and srcdoc pre-populates kv & local fetch
        await store.writeWorkspaceFile("assets/crlf.csv", "E,W\r\n3.6,79\r\n1.8,54\r\n");
        const crlfRes = await tools.executeTool({
            runId: "r-crlf",
            operationId: "op-crlf",
            toolName: "workspace.run_script",
            effectClass: "idempotent_write",
            args: {
                code: "const rows = readFile('assets/crlf.csv').split('\\n').slice(1).map(l => l.split(',').map(Number)); const waitings = rows.map(([,w]) => w); return { count: rows.length, minW: Math.min(...waitings) };",
            },
        });
        assert.match(crlfRes.outputText, /"count": 2/);
        assert.match(crlfRes.outputText, /"minW": 54/);

        const bridgeSrcdoc = compileMultiFileSrcdoc(
            {
                workspaceId: "ws-alpha",
                entryHtml: "<!doctype html><html><head><script>fetch('./stats.json'); window.veles.kv.get('bins');</script></head><body></body></html>",
                entryPath: "preview/index.html",
                workspaceSeq: 99,
                files: { "preview/stats.json": '{"bins":[1,2,3]}' },
            },
            5,
            { bins: 11 },
        );
        assert.ok(bridgeSrcdoc.includes('"bins":11'));
        assert.ok(bridgeSrcdoc.includes("_resolveLocalFile"));
        assert.ok(bridgeSrcdoc.includes("getItem: _kvGet"));
        assert.ok(bridgeSrcdoc.includes("setItem: _kvSet"));

        const kvRouter = new VelesHarnessRouter();
        const kvRouted = await kvRouter.routeWithOptionalClassifier(
            "compute scatter plot into preview/stats.json and remember filter using window.veles.kv",
            ["assets/OldFaithful.csv"],
        );
        assert.deepEqual(
            kvRouted.profiles.map((p) => p.id),
            ["data-script-compute"],
        );
    });

    it("applies targeted search-and-replace edits via workspace.edit_file with uniqueness & indentation tolerance", async () => {
        const { store } = await createSliceWorkspace("ws-edit-file");
        const tools = new BrowserToolRegistry(store, createDefaultWorkspaceTools());
        await store.writeWorkspaceFile(
            "preview/app.js",
            [
                "const bins = 10;",
                "function renderChart(count) {",
                "    const label = `Bins: ${count}`;",
                "    console.log(label);",
                "}",
                "renderChart(bins);",
            ].join("\r\n"),
        );

        // 1. Single edit with CRLF source & literal '$&' in replacement
        const singleEdit = await tools.executeTool({
            runId: "r-edit-1",
            operationId: "op-edit-1",
            toolName: "workspace.edit_file",
            effectClass: "idempotent_write",
            args: {
                path: "preview/app.js",
                target: "const bins = 10;",
                replacement: "const bins = Number(window.veles.kv.get('bins', 12)); // $& literal",
            },
        });
        assert.match(singleEdit.outputText, /Edited preview\/app\.js \(1 edit applied/);
        const afterSingle = String(singleEdit.fileUpdates?.["preview/app.js"]);
        assert.ok(afterSingle.includes("const bins = Number(window.veles.kv.get('bins', 12)); // $& literal"));
        await store.writeWorkspaceFile("preview/app.js", afterSingle);

        // 2. Indentation-tolerant line match, single-line collapsed multi-line match (snake_e.json), & multi-edit batch
        await store.writeWorkspaceFile(
            "preview/snake.js",
            [
                "if (this.x >= canvas.width) {",
                "    this.x = 0;",
                "}",
                "",
                "if (this.y >= canvas.height) {",
                "    this.y = 0;",
                "}",
            ].join("\n"),
        );
        const collapsedSnakeEdit = await tools.executeTool({
            runId: "r-edit-snake",
            operationId: "op-edit-snake",
            toolName: "workspace.edit_file",
            effectClass: "idempotent_write",
            args: {
                path: "preview/snake.js",
                target: "if (this.x >= canvas.width) { this.x = 0; } if (this.y >= canvas.height) { this.y = 0; }",
                replacement: "if (this.x >= canvas.width || this.y >= canvas.height) {\n    this.endGame();\n}",
            },
        });
        assert.ok(
            String(collapsedSnakeEdit.fileUpdates?.["preview/snake.js"]).includes(
                "this.endGame();",
            ),
            "Single-line whitespace-collapsed target must match multi-line span in workspace.edit_file",
        );

        const batchEdit = await tools.executeTool({
            runId: "r-edit-2",
            operationId: "op-edit-2",
            toolName: "workspace.edit_file",
            effectClass: "idempotent_write",
            args: {
                path: "preview/app.js",
                edits: [
                    {
                        // Indentation uses 2 spaces instead of 4 spaces in the file
                        target: "function renderChart(count) {\n  const label = `Bins: ${count}`;\n  console.log(label);\n}",
                        replacement: "function renderChart(count) {\n    window.veles.log('renderChart', count);\n}",
                    },
                    {
                        target: "renderChart(bins);",
                        replacement: "renderChart(bins);\nwindow.veles.kv.set('bins', bins);",
                    },
                ],
            },
        });
        assert.match(batchEdit.outputText, /Edited preview\/app\.js \(2 edits applied/);
        const afterBatch = String(batchEdit.fileUpdates?.["preview/app.js"]);
        assert.ok(afterBatch.includes("window.veles.log('renderChart', count);"));
        assert.ok(afterBatch.includes("window.veles.kv.set('bins', bins);"));
        await store.writeWorkspaceFile("preview/app.js", afterBatch);

        // 3. Reject missing target with concise ERR_EDIT_TARGET_NOT_FOUND (no duplicate file dump bloat)
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-edit-missing",
                    operationId: "op-edit-missing",
                    toolName: "workspace.edit_file",
                    effectClass: "idempotent_write",
                    args: {
                        path: "preview/app.js",
                        target: "nonExistentFunctionCall();",
                        replacement: "noop();",
                    },
                }),
            /ERR_EDIT_TARGET_NOT_FOUND/,
        );

        // 4. Reject ambiguous target unless replaceAll: true
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-edit-ambiguous",
                    operationId: "op-edit-ambiguous",
                    toolName: "workspace.edit_file",
                    effectClass: "idempotent_write",
                    args: {
                        path: "preview/app.js",
                        target: "bins",
                        replacement: "numBins",
                    },
                }),
            /ERR_EDIT_AMBIGUOUS_TARGET/,
        );

        const replaceAllRes = await tools.executeTool({
            runId: "r-edit-all",
            operationId: "op-edit-all",
            toolName: "workspace.edit_file",
            effectClass: "idempotent_write",
            args: {
                path: "preview/app.js",
                target: "renderChart",
                replacement: "drawHistogram",
                replaceAll: true,
            },
        });
        assert.ok(
            String(replaceAllRes.fileUpdates?.["preview/app.js"]).includes(
                "function drawHistogram(count)",
            ),
        );
        assert.ok(
            String(replaceAllRes.fileUpdates?.["preview/app.js"]).includes(
                "drawHistogram(bins);",
            ),
        );

        // 5. Verify ContextBuilder keeps the last 2 tool-bearing runs hot even when a chat-only run occurs in between (with explicit hotRunWindow: 2)
        const longHtml = `<!DOCTYPE html><html><head><style>body { font-family: Arial, sans-serif; }</style></head><body>${"x".repeat(200)}</body></html>`;
        const msgsAfterChatTurn = ContextBuilder.buildMessages(
            [
                {
                    seq: 1,
                    eventId: "e1",
                    workspaceId: "ws-edit-file",
                    runId: "run-1-tools",
                    type: "RunCreated",
                    schemaVersion: 1,
                    relatedOperationId: null,
                    committedState: "created",
                    timestampMs: 1,
                    payload: { prompt: "Build chart" },
                },
                {
                    seq: 2,
                    eventId: "e2",
                    workspaceId: "ws-edit-file",
                    runId: "run-1-tools",
                    type: "ToolIntentLogged",
                    schemaVersion: 1,
                    relatedOperationId: "op-1",
                    committedState: "awaiting_tool",
                    timestampMs: 2,
                    payload: {
                        toolCallId: "c1",
                        toolName: "workspace.write_file",
                        args: { path: "preview/index.html", content: longHtml },
                    },
                },
                {
                    seq: 3,
                    eventId: "e3",
                    workspaceId: "ws-edit-file",
                    runId: "run-1-tools",
                    type: "ToolResultCommitted",
                    schemaVersion: 1,
                    relatedOperationId: "op-1",
                    committedState: "inferring",
                    timestampMs: 3,
                    payload: {
                        toolCallId: "c1",
                        toolName: "workspace.write_file",
                        outputText: "Updated preview/index.html",
                    },
                },
                {
                    seq: 4,
                    eventId: "e4",
                    workspaceId: "ws-edit-file",
                    runId: "run-2-chat-only",
                    type: "RunCreated",
                    schemaVersion: 1,
                    relatedOperationId: null,
                    committedState: "created",
                    timestampMs: 4,
                    payload: { prompt: "let's test edit file" },
                },
                {
                    seq: 5,
                    eventId: "e5",
                    workspaceId: "ws-edit-file",
                    runId: "run-2-chat-only",
                    type: "RunCompleted",
                    schemaVersion: 1,
                    relatedOperationId: null,
                    committedState: "completed",
                    timestampMs: 5,
                    payload: { assistantText: "Sure, what would you like to edit?" },
                },
                {
                    seq: 6,
                    eventId: "e6",
                    workspaceId: "ws-edit-file",
                    runId: "run-3-edit",
                    type: "RunCreated",
                    schemaVersion: 1,
                    relatedOperationId: null,
                    committedState: "created",
                    timestampMs: 6,
                    payload: { prompt: "change background color" },
                },
            ],
            { hotRunWindow: 2 },
        );
        const writeCallMsg = msgsAfterChatTurn.find(
            (m) => m.role === "assistant" && m.toolCalls?.[0]?.name === "workspace.write_file",
        );
        assert.ok(
            writeCallMsg?.toolCalls?.[0]?.argumentsJson.includes(
                "body { font-family: Arial, sans-serif; }",
            ),
            "Tool-bearing run-1 must remain un-folded across chat-only run-2 even when hotRunWindow: 2 is explicitly passed",
        );

        // 6. Verify comment-obfuscated import/**/(url) and eval/GeneratorFunction dynamic code generation are blocked in workspace.run_script
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-imp-comment",
                    operationId: "op-imp-comment",
                    toolName: "workspace.run_script",
                    effectClass: "idempotent_write",
                    args: { code: "await import/**/('http://127.0.0.1:8080/leak');" },
                }),
            /ERR_SCRIPT_SANDBOX_VIOLATION/,
        );
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-imp-eval",
                    operationId: "op-imp-eval",
                    toolName: "workspace.run_script",
                    effectClass: "idempotent_write",
                    args: { code: "return eval('1 + 1');" },
                }),
            /EvalError|ERR_SCRIPT_SANDBOX_VIOLATION/,
        );

        // 7. Verify agent.delegate rejects reserved child sessionId 'session-main' and enforces explicit unrestrictedTools
        const runtime = new AgentKernelRuntime(
            store,
            {
                async *streamChat() {
                    yield {
                        providerId: "openai",
                        model: "gpt-4o",
                        deltaText: "ok",
                        toolCalls: [],
                        usage: null,
                        finishReason: "stop",
                    };
                },
            },
            tools,
            tools.getToolDefinitions(),
        );
        const manager = new AgentManager(store, runtime);
        for (const t of createSubagentTools(manager)) {
            tools.registerTool(t);
            runtime.registerToolDefinition(t.spec);
        }
        await manager.createSession({
            id: "session-restricted",
            workspaceId: "ws-edit-file",
            model: TEST_CONFIG,
            allowedTools: ["agent.delegate"],
            delegatableTools: [],
            status: "idle",
        });
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-del-reserved",
                    operationId: "op-del-reserved",
                    toolName: "agent.delegate",
                    effectClass: "idempotent_write",
                    args: {
                        task: "Attempt to hijack session-main",
                        sessionId: "session-main",
                        parentSessionId: "session-restricted",
                        allowedTools: ["workspace.write_file"],
                    },
                }),
            /ERR_DELEGATE_RESERVED_SESSION_ID/,
        );

        // 8. Verify deleted preview KV values stay deleted across iframe reload (window.name replaces initialKv instead of merging)
        const reloadSrcdoc = compileMultiFileSrcdoc(
            {
                workspaceId: "ws-kv-reload",
                entryHtml: "<!doctype html><html><body></body></html>",
                entryPath: "preview/index.html",
                workspaceSeq: 1,
                files: {},
            },
            1,
            { marker: "OLD_VALUE" },
        );
        const bridgeMatch = /<script data-veles-error-bridge="true">([\s\S]*?)<\/script>/.exec(
            reloadSrcdoc,
        );
        assert.ok(bridgeMatch?.[1]);
        const mockWin: Record<string, unknown> = {
            name: "",
            innerWidth: 800,
            innerHeight: 600,
            addEventListener() {},
            parent: { postMessage() {} },
        };
        const runBridge = new Function("window", "document", "parent", bridgeMatch![1]!);
        runBridge(mockWin, { documentElement: { clientWidth: 800, clientHeight: 600 } }, mockWin.parent);
        const veles1 = mockWin.veles as {
            kv: { get(k: string): unknown; remove(k: string): void };
        };
        assert.equal(veles1.kv.get("marker"), "OLD_VALUE");
        veles1.kv.remove("marker");
        assert.equal(veles1.kv.get("marker"), null);
        // Simulate iframe reload with same srcdoc (initialKv still has OLD_VALUE, window.name has __veles_kv__:ws-kv-reload:{})
        runBridge(mockWin, { documentElement: { clientWidth: 800, clientHeight: 600 } }, mockWin.parent);
        const velesAfterReload = mockWin.veles as {
            kv: { get(k: string): unknown };
        };
        assert.equal(
            velesAfterReload.kv.get("marker"),
            null,
            "Deleted KV entry must not reappear after reload",
        );

        // 9. Verify VelesAppClient.resolveApproval fences child completion and premature completed events until RPC response arrives
        let clientListener = null as ((ev: { readonly data: unknown }) => void) | null;
        let lastSentReqId = "";
        const mockPort: WorkerPortLike = {
            postMessage(msg: unknown) {
                const m = msg as { requestId: string; request: { type: string } };
                lastSentReqId = m.requestId;
                if (m.request.type === "INIT_WORKSPACE") {
                    clientListener?.({
                        data: {
                            channel: "veles:worker-port",
                            kind: "RESPONSE",
                            requestId: m.requestId,
                            ok: true,
                            snapshot: {
                                workspaceId: "ws-fence",
                                title: "Fence",
                                leaseMode: "ReadWrite_Leader",
                                headSeq: 2,
                                activeRunId: "run-parent-approval",
                                runState: "awaiting_approval",
                                activeSessionId: "session-main",
                                sessions: [],
                                files: {},
                                fileHashes: {},
                                journal: [],
                                pendingApproval: {
                                    runId: "run-parent-approval",
                                    operationId: "op-1",
                                    toolCallId: "c-1",
                                    toolName: "workspace.delete_file",
                                    effectClass: "non_idempotent",
                                    args: { path: "a.txt" },
                                    assistantText: "",
                                },
                                interruptReason: null,
                            },
                        },
                    });
                }
            },
            addEventListener(_type, h) {
                clientListener = h;
            },
            removeEventListener() {
                clientListener = null;
            },
        };
        const appClient = new VelesAppClient(mockPort);
        await appClient.initWorkspace("Fence");
        const approvalPromise = appClient.resolveApproval("run-parent-approval", true);
        // Emit premature child and parent STATE_CHANGED: completed while RESOLVE_APPROVAL RPC is still pending
        clientListener?.({
            data: {
                channel: "veles:worker-port",
                kind: "EVENT",
                eventType: "STATE_CHANGED",
                runId: "sub-child-1",
                state: "completed",
            },
        });
        clientListener?.({
            data: {
                channel: "veles:worker-port",
                kind: "EVENT",
                eventType: "STATE_CHANGED",
                runId: "run-parent-approval",
                state: "completed",
            },
        });
        assert.equal(
            appClient.getStateView().snapshot?.runState,
            "executing_tool",
            "Snapshot runState must remain executing_tool while RESOLVE_APPROVAL RPC is pending",
        );
        clientListener?.({
            data: {
                channel: "veles:worker-port",
                kind: "RESPONSE",
                requestId: lastSentReqId,
                ok: true,
                snapshot: {
                    workspaceId: "ws-fence",
                    title: "Fence",
                    leaseMode: "ReadWrite_Leader",
                    headSeq: 4,
                    activeRunId: "run-parent-approval",
                    runState: "completed",
                    activeSessionId: "session-main",
                    sessions: [],
                    files: { "preview/done.txt": "ok" },
                    fileHashes: { "preview/done.txt": "12345678" },
                    journal: [],
                    pendingApproval: null,
                    interruptReason: null,
                },
            },
        });
        const finalSnap = await approvalPromise;
        assert.equal(finalSnap.runState, "completed");
        assert.equal(finalSnap.files["preview/done.txt"], "ok");
        appClient.dispose();

        // 10. Verify direct web.fetch_url performs DNS destination validation before issuing HTTP requests
        await assert.rejects(
            () =>
                tools.executeTool({
                    runId: "r-dns-unresolvable",
                    operationId: "op-dns-unresolvable",
                    toolName: "web.fetch_url",
                    effectClass: "pure_read",
                    args: { url: "http://unresolvable-host-veles-probe.invalid/secret" },
                }),
            /ERR_FETCH_DNS_RESOLUTION_FAILED|ERR_FETCH_PRIVATE_HOST_BLOCKED/,
        );
    });

    it("bounds structural signature fields and their aggregate system-header contribution", () => {
        const longField = "x".repeat(100_000);
        const cases = [
            ["assets/object.json", JSON.stringify({ [longField + "\nextra"]: 1 })],
            ["assets/array.json", JSON.stringify([{ [longField]: 1 }])],
            ["assets/data.csv", `${longField},value\n1,2`],
            ["assets/data.tsv", `${longField}\tvalue\n1\t2`],
            ["preview/index.html", `<script src="${longField}\nscript.js"></script>`],
            ["preview/script.js", `function ${longField}() {}`],
            ["assets/shape.svg", `<svg viewBox="${longField}\n0 10 10"></svg>`],
        ];
        for (const [path, content] of cases) {
            const signature = summarizeWorkspaceFileSignature(path!, content!);
            assert.ok(signature && signature.length <= 256, `${path}: signature must be bounded`);
            assert.doesNotMatch(signature, /[\r\n\t]/, `${path}: signature must stay on one line`);
        }

        const fileTree = Object.fromEntries(Array.from({ length: 200 }, (_, i) => [`assets/data-${i}.json`, "a".repeat(64)]));
        const signature = summarizeWorkspaceFileSignature("assets/object.json", JSON.stringify({ [longField]: 1 }))!;
        const fileSignatures = Object.fromEntries(Object.keys(fileTree).map((path) => [path, signature]));
        const baseMessages = ContextBuilder.buildMessages([], { fileTree });
        const messages = ContextBuilder.buildMessages([], { fileTree, fileSignatures, maxCharBudget: 48_000 });
        assert.ok(messages[0]!.content.length - baseMessages[0]!.content.length <= 8_000);
        assert.ok(messages[0]!.content.length <= 48_000);
        assert.match(messages[0]!.content, /assets\/data-199\.json/);
        const injected = ContextBuilder.buildMessages([], {
            fileTree: { "assets/data.json": "a".repeat(64) },
            fileSignatures: { "assets/data.json": `${longField}\nignored` },
            maxCharBudget: 1_000,
        });
        assert.ok(injected[0]!.content.length <= 1_000, "Direct signature inputs must obey the header budget too");
        assert.doesNotMatch(injected[0]!.content, /\nignored/);
    });

    it("skips unsupported assets and reuses file signatures until their content hash changes", async (t) => {
        const { store, blobBackend } = await createSliceWorkspace("ws-signature-cache");
        await store.writeWorkspaceFile("assets/video.mp4", new Uint8Array(1024 * 1024).fill(65));
        await store.writeWorkspaceFile("assets/data.json", '{"before":1}');
        const tree = await store.getFileTree();
        const reads = t.mock.method(blobBackend, "readBytes");
        const observedPrompts: string[] = [];
        let round = 0;
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                observedPrompts.push(req.messages[0]!.content);
                round += 1;
                yield {
                    providerId: req.config.providerId, model: req.config.model,
                    deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? [{ id: "noop-1", name: "noop", argumentsJson: "{}" }] : [],
                };
            },
        }, {
            currentGatewayInstanceId: "test",
            async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; },
        }, [{ name: "noop", description: "No workspace changes", effectClass: "pure_read" }], { multiTurnWorkspaceContext: true });
        await runtime.startRun("signature-cache-1", "Continue", TEST_CONFIG);
        const countReads = (hash: string) => reads.mock.calls.filter((call) => call.arguments[0] === hash).length;
        assert.equal(countReads(tree["assets/video.mp4"]!), 0, "Unsupported assets must never be read for signatures");
        assert.equal(countReads(tree["assets/data.json"]!), 1, "An unchanged signature should be reused across inference rounds");
        assert.ok(observedPrompts.every((prompt) => prompt.includes("keys: [before]")));

        await store.writeWorkspaceFile("assets/data.json", '{"after":2}');
        reads.mock.resetCalls();
        await runtime.startRun("signature-cache-2", "Continue", TEST_CONFIG);
        assert.match(observedPrompts.at(-1)!, /keys: \[after\]/);
        assert.equal(reads.mock.callCount(), 1, "Only the changed supported blob should be read");
    });

    it("refuses a file signature read when its requested manifest hash is stale", async () => {
        const { store } = await createSliceWorkspace("ws-signature-version");
        await store.writeWorkspaceFile("assets/data.json", '{"old":1}');
        const oldHash = (await store.getFileTree())["assets/data.json"]!;
        await store.writeWorkspaceFile("assets/data.json", '{"new":2}');
        await assert.rejects(store.readFileText("assets/data.json", oldHash), /ERR_FILE_CHANGED/);
    });

    it("caches empty and oversized signature results, separates formats, and retries transient reads", async (t) => {
        const { store } = await createSliceWorkspace("ws-signature-misses");
        const source = "function renderPlot() {}";
        await store.writeWorkspaceFile("preview/shared.js", source);
        await store.writeWorkspaceFile("preview/shared.json", source);
        await store.writeWorkspaceFile("preview/empty.js", "const n = 1;");
        await store.writeWorkspaceFile("assets/large.json", " ".repeat(250_001));
        await store.writeWorkspaceFile("assets/retry.json", '{"retried":true}');
        const originalRead = store.readFileText.bind(store);
        let transientFailure = true;
        const reads = t.mock.method(store, "readFileText", async (path: string, expectedHash?: string) => {
            if (path === "assets/retry.json" && transientFailure) {
                transientFailure = false;
                throw new Error("temporary read failure");
            }
            return originalRead(path, expectedHash);
        });
        let round = 0;
        const prompts: string[] = [];
        const runtime = new AgentKernelRuntime(store, {
            async *streamChat(req) {
                round += 1;
                prompts.push(req.messages[0]!.content);
                yield {
                    providerId: req.config.providerId, model: req.config.model, deltaText: "", usage: null,
                    finishReason: round === 1 ? "tool_calls" : "stop",
                    toolCalls: round === 1 ? [{ id: "noop-2", name: "noop", argumentsJson: "{}" }] : [],
                };
            },
        }, {
            currentGatewayInstanceId: "test",
            async executeTool() { return { outputText: "ok", gatewayInstanceId: "test" }; },
        }, [{ name: "noop", description: "No workspace changes", effectClass: "pure_read" }], { multiTurnWorkspaceContext: true });
        await runtime.startRun("signature-misses", "Continue", TEST_CONFIG);
        for (const path of ["preview/shared.js", "preview/shared.json", "preview/empty.js", "assets/large.json"]) {
            assert.equal(reads.mock.calls.filter((call) => call.arguments[0] === path).length, 1, `${path}: read once`);
        }
        assert.equal(reads.mock.calls.filter((call) => call.arguments[0] === "assets/retry.json").length, 2);
        assert.match(prompts[0]!, /shared\.js .*\(js: fns: \[renderPlot\]\)/);
        assert.match(prompts[0]!, /shared\.json .*\(json: invalid\)/);
        assert.doesNotMatch(prompts[0]!, /keys: \[retried\]/);
        assert.match(prompts[1]!, /keys: \[retried\]/);
    });

    it("measures folding savings from retained serialized tool messages", () => {
        const record = (seq: number, runId: string, type: RunJournalRecord["type"], payload: Record<string, unknown>): RunJournalRecord => ({
            eventId: `econ-${seq}`, workspaceId: "ws-economics", runId, seq, type,
            schemaVersion: 1, timestampMs: seq, relatedOperationId: null, committedState: "inferring", payload,
        });
        const content = '\n"\\'.repeat(500);
        const args = { path: "preview/a.txt", content };
        const records = [
            record(1, "old", "RunCreated", { prompt: "Write source" }),
            record(2, "old", "ToolIntentLogged", { toolCallId: "write-1", toolName: "workspace.write_file", args }),
            record(3, "old", "ToolResultCommitted", { toolCallId: "write-1", outputText: "ok" }),
            record(4, "new", "RunCreated", { prompt: "Current turn" }),
        ];
        const foldedArgs = { ...args, content: `<folded ${content.length} chars; read from OPFS if needed>` };
        const economy = ContextBuilder.measureContextEconomics(records, { hotRunWindow: 1 });
        assert.equal(economy.foldedSavedChars, JSON.stringify(args).length - JSON.stringify(foldedArgs).length);
        assert.equal(economy.coldFoldedRunCount, 1);
        const dropped = ContextBuilder.measureContextEconomics(records, { hotRunWindow: 1, maxCharBudget: 20 });
        assert.equal(dropped.foldedSavedChars, 0, "Budget-discarded records cannot claim folding savings");
        assert.equal(dropped.coldFoldedRunCount, 0);
        assert.equal(dropped.hotRunCount, 1);
        const partialBudget = ContextBuilder.measureContextEconomics(records, { hotRunWindow: 1, maxCharBudget: economy.totalChars - "Write source".length });
        assert.equal(partialBudget.foldedSavedChars, economy.foldedSavedChars, "Trimming only the old user message must preserve its retained tool savings");
    });

    it("counts folded image attachments and reports only the signatures fitted beside the active turn", () => {
        const record = (seq: number, runId: string, type: RunJournalRecord["type"], payload: Record<string, unknown>): RunJournalRecord => ({
            eventId: `image-econ-${seq}`, workspaceId: "ws-image-economics", runId, seq, type,
            schemaVersion: 1, timestampMs: seq, relatedOperationId: null, committedState: "inferring", payload,
        });
        const records = [
            record(1, "old", "RunCreated", { prompt: "Inspect image" }),
            record(2, "old", "ToolIntentLogged", { toolCallId: "image-1", toolName: "workspace.inspect_image", args: {} }),
            record(3, "old", "ToolResultCommitted", { toolCallId: "image-1", outputText: "image", imageDataUrls: [`data:image/png;base64,${"A".repeat(5000)}`] }),
            record(4, "new", "RunCreated", { prompt: "Current turn" }),
        ];
        const hot = ContextBuilder.measureContextEconomics(records, { hotRunWindow: 2 });
        const cold = ContextBuilder.measureContextEconomics(records, { hotRunWindow: 1 });
        assert.equal(cold.foldedSavedChars, hot.totalChars - cold.totalChars);
        assert.ok(cold.foldedSavedChars > 4_000, "Folding also removes the image attachment's user-message label");

        const fileTree = { "assets/data.json": "a".repeat(64) };
        const fileSignatures = { "assets/data.json": "json: object, keys: [largeDataset]" };
        const onlyActive = [record(5, "active", "RunCreated", { prompt: "P".repeat(400) })];
        const bare = ContextBuilder.buildContext(onlyActive, { fileTree });
        const fitted = ContextBuilder.buildContext(onlyActive, { fileTree, fileSignatures, maxCharBudget: bare.economics.totalChars });
        assert.deepEqual(fitted.messages, bare.messages, "Optional signatures must yield to the active turn");
        assert.equal(fitted.economics.workspaceTreeChars, bare.economics.workspaceTreeChars);
        assert.equal(fitted.economics.totalChars, bare.economics.totalChars);
    });

    it("runs mechanical preflight checks on workspace writes and injects structural file signatures into ContextBuilder", async () => {
        const { store } = await createSliceWorkspace("ws-preflight-sig");
        const tools = new BrowserToolRegistry(store);

        // 1. Mechanical Preflight ("Bend Law Check"): catches top-level await in classic <script>, unloaded Chart.js, placeholder stubs, and localStorage/kv.then
        const badHtml = [
            "<!doctype html><html><body><canvas id='scatterCanvas'></canvas><script>",
            "const data = await fetch('stats.json').then(r => r.json());",
            "// Chart rendering logic (e.g., using a basic HTML canvas or a library like Chart.js) goes here.",
            "new Chart(document.getElementById('scatterCanvas'), {});",
            "localStorage.setItem('filter', 'all');",
            "window.veles.kv.get('filter').then(v => console.log(v));",
            "</script></body></html>",
        ].join("\n");

        const writeOut = await tools.executeTool({
            runId: "r-pf-1",
            operationId: "op-pf-html",
            toolName: "workspace.write_file",
            effectClass: "idempotent_write",
            args: { path: "preview/index.html", content: badHtml },
        });
        assert.match(writeOut.outputText, /\[PREFLIGHT WARNING: JS SyntaxError in preview\/index\.html inline <script #1>/);
        assert.match(writeOut.outputText, /\[PREFLIGHT WARNING: .*placeholder comment/i);
        assert.match(writeOut.outputText, /\[PREFLIGHT WARNING: .*new Chart\(\.\.\.\) without loading Chart\.js/);
        assert.match(writeOut.outputText, /\[PREFLIGHT WARNING: Direct localStorage\/sessionStorage access/);

        // Valid <script> with async IIFE and 2D canvas drawing produces zero preflight warnings
        const cleanHtml = [
            "<!doctype html><html><head><link rel='stylesheet' href='style.css'></head>",
            "<body><canvas id='scatterCanvas'></canvas><select id='filterSelect'></select>",
            "<script src='script.js'></script>",
            "<script>(async () => { const f = window.veles.kv.get('filter'); })();</script></body></html>",
        ].join("\n");
        const cleanOut = await tools.executeTool({
            runId: "r-pf-1",
            operationId: "op-pf-clean",
            toolName: "workspace.write_file",
            effectClass: "idempotent_write",
            args: { path: "preview/index.html", content: cleanHtml },
        });
        assert.ok(!cleanOut.outputText.includes("[PREFLIGHT WARNING"), "Clean HTML must not emit preflight warnings");
        assert.equal(verifyWorkspaceFilePreflight("preview/stats.json", "{bad json").length, 1);

        // 2. Structural File Signatures ("Zero-Round Project Intelligence")
        await store.writeWorkspaceFile(
            "assets/OldFaithful.csv",
            "rownames,TimeEruption,TimeWaiting\n1,3.6,79\n2,1.8,54\n",
        );
        await store.writeWorkspaceFile(
            "preview/stats.json",
            JSON.stringify([{ eruption: 3.6, waiting: 79 }, { eruption: 1.8, waiting: 54 }]),
        );
        await store.writeWorkspaceFile("preview/index.html", cleanHtml);
        await store.writeWorkspaceFile(
            "preview/script.js",
            "function renderScatter(pts) {}\nconst applyFilter = (mode) => {};",
        );

        assert.equal(
            summarizeWorkspaceFileSignature(
                "assets/OldFaithful.csv",
                "rownames,TimeEruption,TimeWaiting\n1,3.6,79\n2,1.8,54\n",
            ),
            "csv: 2 rows, cols: [rownames, TimeEruption, TimeWaiting]",
        );
        assert.equal(
            summarizeWorkspaceFileSignature(
                "preview/stats.json",
                JSON.stringify([{ eruption: 3.6, waiting: 79 }]),
            ),
            "json: array[1], itemKeys: [eruption, waiting]",
        );

        let capturedSystemPrompt = "";
        const runtime = new AgentKernelRuntime(
            store,
            {
                async *streamChat(req) {
                    capturedSystemPrompt = req.messages.find((m) => m.role === "system")?.content ?? "";
                    yield {
                        providerId: req.config.providerId,
                        model: req.config.model,
                        deltaText: "Done",
                        toolCalls: [],
                        usage: null,
                        finishReason: "stop",
                    };
                },
            },
            tools,
            tools.getToolDefinitions(),
            { multiTurnWorkspaceContext: true },
        );
        await runtime.startRun("run-sig-check", "Plot OldFaithful scatter chart", TEST_CONFIG);
        assert.match(
            capturedSystemPrompt,
            /assets\/OldFaithful\.csv \[sha256:[0-9a-f]{8}\] \(csv: 2 rows, cols: \[rownames, TimeEruption, TimeWaiting\]\)/,
        );
        assert.match(
            capturedSystemPrompt,
            /preview\/stats\.json \[sha256:[0-9a-f]{8}\] \(json: array\[2\], itemKeys: \[eruption, waiting\]\)/,
        );
        assert.match(
            capturedSystemPrompt,
            /preview\/index\.html \[sha256:[0-9a-f]{8}\] \(html: <canvas>, refs: \[style\.css, script\.js\], ids: \[#scatterCanvas, #filterSelect\]\)/,
        );
        assert.match(
            capturedSystemPrompt,
            /preview\/script\.js \[sha256:[0-9a-f]{8}\] \(js: fns: \[renderScatter, applyFilter\]\)/,
        );

        // 3. Verify Prompt-Cache-Friendly ordering (Specialist Guidance precedes volatile OPFS Workspace Tree)
        const specialistIdx = capturedSystemPrompt.indexOf("--- Active In-Browser Specialist Guidance ---");
        const treeIdx = capturedSystemPrompt.indexOf("Current Committed OPFS Workspace Tree:");
        assert.ok(specialistIdx > 0 && specialistIdx < treeIdx, "Specialist capsules must precede volatile OPFS tree for prefix caching");

        // 4. Verify ContextEconomicsSummary checkpointing and cold-run folding savings
        const journalAfterSig = await store.getJournal();
        const routingCheckpoint = journalAfterSig.find((r) => r.runId === "run-sig-check" && r.payload.contextEconomics);
        assert.ok(routingCheckpoint, "Expected contextEconomics on harness-routing CheckpointCommitted record");
        const econ1 = routingCheckpoint!.payload.contextEconomics as Record<string, number>;
        assert.ok(econ1.systemContractChars! > 100);
        assert.ok(econ1.specialistCapsuleChars! > 100);
        assert.ok(econ1.workspaceTreeChars! > 100);
        assert.ok(econ1.cacheablePrefixTokens! >= Math.ceil((econ1.systemContractChars! + econ1.specialistCapsuleChars!) / 4));

        // Simulate a cold run with large tool content and measure folded savings
        const synthJournal = [
            ...journalAfterSig,
            {
                ...journalAfterSig[0]!,
                seq: 100,
                runId: "run-cold-big",
                type: "RunCreated" as const,
                committedState: "created" as const,
                payload: { prompt: "Write large file" },
            },
            {
                ...journalAfterSig[0]!,
                seq: 101,
                runId: "run-cold-big",
                type: "ToolIntentLogged" as const,
                committedState: "awaiting_tool" as const,
                payload: {
                    toolCallId: "call_big",
                    toolName: "workspace.write_file",
                    args: { path: "preview/big.js", content: "x".repeat(600) },
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 102,
                runId: "run-cold-big",
                type: "ToolResultCommitted" as const,
                committedState: "inferring" as const,
                payload: {
                    toolCallId: "call_big",
                    toolName: "workspace.write_file",
                    outputText: "y".repeat(500),
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 103,
                runId: "run-hot-latest",
                type: "RunCreated" as const,
                committedState: "created" as const,
                payload: { prompt: "Latest turn" },
            },
        ];
        const coldEcon = ContextBuilder.measureContextEconomics(synthJournal, {
            fileTree: await store.getFileTree(),
            hotRunWindow: 1,
        });
        assert.equal(coldEcon.hotRunCount, 1);
        assert.ok(coldEcon.coldFoldedRunCount >= 1);
        assert.ok(coldEcon.foldedSavedChars > 800, `Expected >800 folded saved chars, got ${coldEcon.foldedSavedChars}`);

        // 5. Verify Tier-0 Active Context Curation: intra-hot-run superseded file & resolved ToolError folding + outcome-aware hot window
        const snakeCode = "const snake = 1;\n".repeat(250);
        const curatedJournal: RunJournalRecord[] = [
            {
                ...journalAfterSig[0]!,
                seq: 201,
                runId: "run-good-1",
                type: "RunCreated",
                committedState: "created",
                payload: { prompt: "Create snake game" },
            },
            {
                ...journalAfterSig[0]!,
                seq: 202,
                runId: "run-good-1",
                type: "ToolIntentLogged",
                committedState: "awaiting_tool",
                payload: {
                    toolCallId: "c-write-v1",
                    toolName: "workspace.write_file",
                    args: { path: "preview/script.js", content: snakeCode },
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 203,
                runId: "run-good-1",
                type: "ToolResultCommitted",
                committedState: "inferring",
                payload: {
                    toolCallId: "c-write-v1",
                    toolName: "workspace.write_file",
                    outputText: "Updated preview/script.js",
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 204,
                runId: "run-err-2",
                type: "RunCreated",
                committedState: "created",
                payload: { prompt: "Failed edit attempt" },
            },
            {
                ...journalAfterSig[0]!,
                seq: 205,
                runId: "run-err-2",
                type: "ToolIntentLogged",
                committedState: "awaiting_tool",
                payload: {
                    toolCallId: "c-err-edit",
                    toolName: "workspace.edit_file",
                    args: {
                        path: "preview/script.js",
                        edits: [{ target: "missing".repeat(30), replacement: "noop" }],
                    },
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 206,
                runId: "run-err-2",
                type: "ToolResultCommitted",
                committedState: "inferring",
                payload: {
                    toolCallId: "c-err-edit",
                    toolName: "workspace.edit_file",
                    outputText: "ToolError (workspace.edit_file): ERR_EDIT_TARGET_NOT_FOUND: edit #1 target block not found in 'preview/script.js'",
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 207,
                runId: "run-active-3",
                type: "RunCreated",
                committedState: "created",
                payload: { prompt: "Read and edit snake" },
            },
            {
                ...journalAfterSig[0]!,
                seq: 208,
                runId: "run-active-3",
                type: "ToolIntentLogged",
                committedState: "awaiting_tool",
                payload: {
                    toolCallId: "c-read-v1",
                    toolName: "workspace.read_file",
                    args: { path: "preview/script.js" },
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 209,
                runId: "run-active-3",
                type: "ToolResultCommitted",
                committedState: "inferring",
                payload: {
                    toolCallId: "c-read-v1",
                    toolName: "workspace.read_file",
                    outputText: snakeCode,
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 210,
                runId: "run-active-3",
                type: "ToolIntentLogged",
                committedState: "awaiting_tool",
                payload: {
                    toolCallId: "c-edit-ok",
                    toolName: "workspace.edit_file",
                    args: {
                        path: "preview/script.js",
                        target: "const snake = 1;",
                        replacement: "const snake = 2;",
                    },
                },
            },
            {
                ...journalAfterSig[0]!,
                seq: 211,
                runId: "run-active-3",
                type: "ToolResultCommitted",
                committedState: "inferring",
                payload: {
                    toolCallId: "c-edit-ok",
                    toolName: "workspace.edit_file",
                    outputText: "Edited preview/script.js (1 edit applied)",
                },
            },
        ];
        const curated = ContextBuilder.buildContext(curatedJournal, {
            hotRunWindow: 2,
        });
        assert.ok(
            curated.economics.foldedSavedChars > 7_000,
            `Expected superseded write_file + read_file + resolved ToolError to save >7000 chars, got ${curated.economics.foldedSavedChars}`,
        );
        const resolvedErrMsg = curated.messages.find(
            (m) => m.role === "tool" && m.toolCallId === "c-err-edit",
        );
        assert.ok(resolvedErrMsg?.content.includes("[resolved in later step]"));
    });

    it("supports Code Mode (Programmatic Tool Calling) in workspace.run_script with listFiles, searchFiles, editFile, and SSRF-guarded fetchUrl", async () => {
        const { store } = await createSliceWorkspace("ws-code-mode-ptc");
        await store.writeWorkspaceFile(
            "preview/app.js",
            "const a = oldApi('one');\nconst b = keepMe();\nconst c = oldApi('two');\n",
        );
        await store.writeWorkspaceFile("notes/readme.txt", "unrelated oldApi mention");

        const csvSample = "rownames,TimeEruption,TimeWaiting\n1,3.6,79\n2,1.8,54\n";
        const originalFetch = globalThis.fetch;
        globalThis.fetch = (async (input: string | URL | Request) => {
            const url = String(input);
            if (url === "https://raw.githubusercontent.com/gchoi/Dataset/master/OldFaithful.csv") {
                return {
                    ok: true,
                    status: 200,
                    headers: new Headers({ "content-type": "text/csv; charset=utf-8" }),
                    text: async () => csvSample,
                } as Response;
            }
            throw new TypeError(`Unexpected fetch URL: ${url}`);
        }) as typeof globalThis.fetch;

        try {
            const tools = new BrowserToolRegistry(store);

            // 1. Single-round Code Mode orchestration: fetch remote CSV -> parse -> compute -> search & edit workspace files -> stage all updates
            const ptcOut = await tools.executeTool({
                runId: "r-ptc-1",
                operationId: "op-ptc-1",
                toolName: "workspace.run_script",
                effectClass: "idempotent_write",
                args: {
                    code: `
                        const rawCsv = await tools.web.fetchUrl(
                            "https://github.com/gchoi/Dataset/blob/master/OldFaithful.csv",
                            "assets/OldFaithful.csv"
                        );
                        const points = rawCsv.trim().split("\\n").slice(1).map((line) => {
                            const [, e, w] = line.split(",");
                            return { eruption: Number(e), waiting: Number(w) };
                        });
                        writeFile("preview/stats.json", points);

                        const hits = tools.workspace.search("oldApi(", "preview/");
                        for (const hit of hits) {
                            tools.workspace.editFile(hit.path, "oldApi(", "newApi(", true);
                        }
                        return {
                            allFiles: listFiles(),
                            hitCount: hits.length,
                            pointCount: points.length,
                            rawFetchGlobal: typeof fetch,
                        };
                    `,
                },
            });

            assert.ok(ptcOut.outputText.includes('"hitCount": 2'));
            assert.ok(ptcOut.outputText.includes('"pointCount": 2'));
            assert.ok(ptcOut.outputText.includes('"rawFetchGlobal": "undefined"'));
            assert.ok(String(ptcOut.fileUpdates?.["assets/OldFaithful.csv"]).includes("TimeEruption"));
            assert.ok(String(ptcOut.fileUpdates?.["preview/stats.json"]).includes('"eruption": 3.6'));
            assert.equal(
                ptcOut.fileUpdates?.["preview/app.js"],
                "const a = newApi('one');\nconst b = keepMe();\nconst c = newApi('two');\n",
            );

            // 2. Verify SSRF private-host guard still blocks private URLs inside Code Mode fetchUrl()
            await assert.rejects(
                () =>
                    tools.executeTool({
                        runId: "r-ptc-ssrf",
                        operationId: "op-ptc-ssrf",
                        toolName: "workspace.run_script",
                        effectClass: "idempotent_write",
                        args: {
                            code: `return await fetchUrl("http://127.0.0.1:11434/api/tags");`,
                        },
                    }),
                /ERR_FETCH_PRIVATE_HOST_BLOCKED/,
            );
        } finally {
            globalThis.fetch = originalFetch;
        }
    });
});
