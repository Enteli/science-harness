import { loadCopilotSdk } from "./sdk.mjs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { hash, visibleResponse } from "../physics_bridge_state.mjs";

const runtime = path.resolve(process.argv[2]);
const { CopilotClient, RuntimeConnection, sdkResolution } = loadCopilotSdk();
await mkdir(runtime, { recursive: true });
const client = new CopilotClient({
  connection: RuntimeConnection.forStdio({
    path: process.env.COPILOT_CLI_PATH,
    args: ["--prefer-version", process.env.COPILOT_CLI_VERSION, "--no-auto-update"],
  }),
  mode: "empty", baseDirectory: runtime, workingDirectory: runtime,
  ...(process.env.GH_TOKEN
    ? { gitHubToken: process.env.GH_TOKEN, useLoggedInUser: false }
    : { useLoggedInUser: true }),
  logLevel: "error",
});
let prepared;

function config(model, system) {
  return {
    model, allowedModels: [model], reasoningSummary: "none", contextTier: "default",
    streaming: true,
    modelCapabilities: { limits: {
      max_prompt_tokens: 32768, max_output_tokens: 32000, max_context_window_tokens: 64768,
    } },
    availableTools: [], excludedTools: ["builtin:*", "mcp:*", "custom:*"],
    workingDirectory: runtime, enableSkills: false, enableSessionStore: false,
    enableHostGitOperations: false, enableConfigDiscovery: false, enableFileHooks: false,
    enableOnDemandInstructionDiscovery: false, skipEmbeddingRetrieval: true,
    embeddingCacheStorage: "in-memory",
    infiniteSessions: { enabled: false, backgroundCompactionThreshold: 1, bufferExhaustionThreshold: 1 },
    onPermissionRequest: () => ({ kind: "denied-by-rules" }),
    hooks: { onErrorOccurred: () => ({ errorHandling: "abort", retryCount: 0 }) },
    systemMessage: { mode: "replace", content: system },
  };
}

async function prepare(request) {
  if (prepared) throw new Error("Unconsumed prepared decision; no replacement");
  const { model, system, user } = request;
  if (![model, system, user].every((v) => typeof v === "string" && v.length)) {
    throw new Error("A decision needs nonempty model/system/user strings");
  }
  const session = await client.createSession(config(model, system));
  try {
    await session.rpc.tools.initializeAndValidate();
    const { tools } = await session.rpc.tools.getCurrentMetadata();
    const current = await session.rpc.model.getCurrent();
    const context = await session.rpc.metadata.recomputeContextTokens({ modelId: model });
    const { contextInfo } = await session.rpc.metadata.contextInfo({ promptTokenLimit: 0, outputTokenLimit: 0 });
    const before = await session.rpc.usage.getMetrics();
    if (!Array.isArray(tools) || tools.length || current.modelId !== model
        || before.totalUserRequests !== 0 || !contextInfo
        || contextInfo.promptTokenLimit !== 32768 || contextInfo.customInstructionsTokens !== null) {
      throw new Error("Fresh isolated decision preflight failed");
    }
    if (context.totalTokens + Buffer.byteLength(user) + 1024 > 32768) {
      throw new Error("Entire original decision does not fit; no truncation");
    }
    const packet = { model, system, user };
    const request_sha256 = hash(packet);
    prepared = { session, packet, request_sha256 };
    return { request_sha256, session_id: session.sessionId, available_tools: 0,
      current, context, contextInfo, metrics: before };
  } catch (error) {
    await session.disconnect();
    throw error;
  }
}

async function dispatch(request) {
  if (!prepared || request.request_sha256 !== prepared.request_sha256) {
    throw new Error("Dispatch does not match the reserved preparation");
  }
  const { session, packet, request_sha256 } = prepared;
  prepared = undefined;
  const messages = [], usage = [];
  const eventCounts = {};
  let started = false, ended = false, timer, unsubscribe, result;
  try {
    await new Promise((resolve, reject) => {
      const fail = (error) => {
        reject(error);
        session.abort().catch((abortError) => console.error(`Abort failed: ${abortError.message}`));
      };
      unsubscribe = session.on((event) => {
        if (ended || event.agentId) return;
        eventCounts[event.type] = (eventCounts[event.type] ?? 0) + 1;
        if (event.type === "user.message") started = true;
        if (event.type === "assistant.turn_start" && eventCounts[event.type] > 1) {
          fail(new Error("Unexpected second model turn"));
        }
        if (event.type === "assistant.usage") {
          usage.push(Object.fromEntries([
            "model", "inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens",
            "reasoningTokens", "availableToolCount", "numToolCalls", "finishReason",
          ].map((k) => [k, event.data[k]])));
        }
        if (event.type === "assistant.message" && started) {
          const { messageId, content, phase } = event.data;
          messages.push({ messageId, content, phase });
        }
        if (event.type === "tool.execution_start" || event.type === "subagent.started"
            || event.type.startsWith("session.compaction") || event.type === "model.call_failure"
            || event.type === "session.error") fail(new Error(`Forbidden/failed runtime event: ${event.type}`));
        if (event.type === "session.idle" && started) {
          ended = true;
          event.data.aborted ? reject(new Error("Aborted decision")) : resolve();
        }
      });
      timer = setTimeout(() => fail(new Error("Decision timeout; no retry")), 300000);
      session.rpc.sendMessages({ messages: [{ prompt: packet.user }], agentMode: "interactive" }).catch(fail);
    });
    const metrics = await session.rpc.usage.getMetrics();
    result = { ...visibleResponse(messages), metrics };
  } catch (error) {
    await session.abort();
    result = { error: error.message, metrics: await session.rpc.usage.getMetrics() };
  } finally {
    clearTimeout(timer);
    unsubscribe?.();
    await session.disconnect();
  }
  const eventsPath = path.join(runtime, "session-state", session.sessionId, "events.jsonl");
  const journal = await readFile(eventsPath, "utf8");
  const events = journal.trim().split("\n").map((line) => JSON.parse(line));
  const shutdown = events.filter((event) => event.type === "session.shutdown").at(-1);
  const turns = events.filter((event) => event.type === "assistant.turn_start" && !event.agentId);
  const shutdownVerified = Boolean(shutdown) && turns.length === 1;
  const receipt = { ...result, request_sha256, session_id: session.sessionId, usage,
    event_counts: eventCounts, shutdown_verified: shutdownVerified, journal_sha256: hash(journal) };
  await writeFile(path.join(runtime, `${request_sha256}.json`), JSON.stringify(receipt) + "\n", { flag: "wx" });
  if (result.error) throw Object.assign(new Error(result.error), { evidence: receipt });
  return receipt;
}

try {
  await client.start();
  const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    try {
      const request = JSON.parse(line);
      let result;
      if (request.operation === "catalog") {
        const models = await client.listModels();
        result = { models, ...(await client.rpc.models.listTokenPriceUnits()), sdkResolution };
      } else if (request.operation === "prepare") result = await prepare(request);
      else if (request.operation === "dispatch") result = await dispatch(request);
      else if (request.operation === "discard") {
        if (!prepared) throw new Error("No unpaid preparation to discard");
        const { session } = prepared;
        prepared = undefined;
        const metrics = await session.rpc.usage.getMetrics();
        await session.disconnect();
        if (metrics.totalUserRequests !== 0 || metrics.totalNanoAiu !== 0) {
          throw new Error("Unpaid SDK lifecycle qualification failed");
        }
        result = { paid_requests: 0, credits_nano: 0, context_disconnected: true,
          unused_context_has_no_transcript: true, session_id: session.sessionId };
      }
      else throw new Error("Unknown decision operation");
      console.log(JSON.stringify({ ok: true, result }));
    } catch (error) {
      console.log(JSON.stringify({ ok: false, error: error.message, evidence: error.evidence ?? null }));
    }
  }
} finally {
  if (prepared) await prepared.session.disconnect();
  await client.stop();
}
