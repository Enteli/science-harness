import { loadCopilotSdk } from "./providers/sdk.mjs";
import { mkdir, readFile, writeFile, rename } from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { hash, historyDelta, visibleResponse } from "./physics_bridge_state.mjs";

const runtime = path.resolve(process.argv[2]);
const { CopilotClient, RuntimeConnection, sdkResolution } = loadCopilotSdk();
if (!process.env.COPILOT_CLI_PATH || !process.env.COPILOT_CLI_VERSION) {
  throw new Error("Pinned Copilot executable and version are required");
}
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
const sessions = new Map();
const permitted = new Set(["gpt-6.1-sol", "claude-haiku-5.5"]);

async function catalog() {
  const models = (await client.listModels()).filter((m) => permitted.has(m.id));
  const { units } = await client.rpc.models.listTokenPriceUnits();
  return { models, units: units.filter((u) => permitted.has(u.modelId)), sdkResolution };
}

function sessionConfig(model, system) {
  return {
    model, allowedModels: [model], reasoningEffort: "high",
    contextTier: "long_context", streaming: true,
    modelCapabilities: { limits: {
      max_prompt_tokens: 131072, max_output_tokens: 32768, max_context_window_tokens: 163840,
    } },
    availableTools: [], excludedTools: ["builtin:*", "mcp:*", "custom:*"],
    workingDirectory: runtime, enableSkills: false, enableSessionStore: false,
    enableHostGitOperations: false, enableConfigDiscovery: false,
    enableFileHooks: false, enableOnDemandInstructionDiscovery: false,
    skipEmbeddingRetrieval: true, embeddingCacheStorage: "in-memory",
    infiniteSessions: {
      enabled: false, backgroundCompactionThreshold: 1, bufferExhaustionThreshold: 1,
    }, sessionLimits: { maxAiCredits: 1000 },
    onPermissionRequest: () => ({ kind: "denied-by-rules" }),
    hooks: { onErrorOccurred: () => ({ errorHandling: "abort", retryCount: 0 }) },
    systemMessage: { mode: "replace", content: system },
  };
}

async function storedSession(request) {
  if (!permitted.has(request.model)) throw new Error("Unregistered model");
  if (sessions.has(request.session_key)) return sessions.get(request.session_key);
  const statePath = path.join(runtime, `${hash(request.session_key)}.json`);
  let state;
  try { state = JSON.parse(await readFile(statePath, "utf8")); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const fingerprint = hash({ model: request.model, system: request.system });
  if (state && (state.pending || state.fingerprint !== fingerprint)) {
    throw new Error("Ambiguous dispatch or changed session contract; cannot resume");
  }
  const config = sessionConfig(request.model, request.system);
  const session = state
    ? await client.resumeSession(state.sessionId, { ...config, allowTranscriptRecovery: false })
    : await client.createSession(config);
  state ??= { fingerprint, sessionId: session.sessionId, logical: [], pending: false };
  const value = { session, state, statePath };
  sessions.set(request.session_key, value);
  return value;
}

async function persist(value) {
  const temporary = value.statePath + ".tmp";
  await writeFile(temporary, JSON.stringify(value.state) + "\n", { flag: "wx" });
  await rename(temporary, value.statePath);
}

async function prepare(request) {
  const value = await storedSession(request);
  const { state, session } = value;
  await session.rpc.tools.initializeAndValidate();
  const { tools } = await session.rpc.tools.getCurrentMetadata();
  if (!Array.isArray(tools) || tools.length !== 0) {
    throw new Error("Cannot verify an initialized zero-tool session before dispatch");
  }
  const current = await session.rpc.model.getCurrent();
  if (current.modelId !== request.model || current.reasoningEffort !== "high") {
    throw new Error("Runtime model or reasoning configuration differs from the protocol");
  }
  const delta = historyDelta(request.messages, state.logical);
  const runtimeContext = await session.rpc.metadata.recomputeContextTokens({ modelId: request.model });
  const { contextInfo } = await session.rpc.metadata.contextInfo({ promptTokenLimit: 0, outputTokenLimit: 0 });
  // Tokenize pending text with the runtime's model-specific counter, without sending it.
  const pendingText = delta.map((message) => message.content).join("\n\n") +
    (request.context_suffix ? "\n\n" + request.context_suffix : "");
  const meter = await client.createSession(sessionConfig(request.model, pendingText));
  let pendingTokens;
  try {
    await meter.rpc.tools.initializeAndValidate();
    pendingTokens = await meter.rpc.metadata.recomputeContextTokens({ modelId: request.model });
    const metrics = await meter.rpc.usage.getMetrics();
    if (metrics.totalUserRequests !== 0) throw new Error("Token measurement unexpectedly ran inference");
  } finally {
    await meter.disconnect();
    await client.deleteSession(meter.sessionId);
  }
  value.prepared = { request, delta };
  return {
    prepared_hash: hash(request), catalog: await catalog(),
    current, available_tool_count: tools.length,
    logical_prefix_length: state.logical.length, runtime_context: runtimeContext,
    pending_text_runtime_tokens: pendingTokens.totalTokens,
    context_info: contextInfo ? {
      modelName: contextInfo.modelName, totalTokens: contextInfo.totalTokens,
      promptTokenLimit: contextInfo.promptTokenLimit, compactionThreshold: contextInfo.compactionThreshold,
    } : null,
    metrics: await session.rpc.usage.getMetrics(),
    limits_requested: { input: 131072, output: 32768 },
    dispatch_policy: "one-interactive-turn-zero-tools-no-compaction",
  };
}

async function dispatch(request) {
  const value = sessions.get(request.session_key);
  if (!value?.prepared || hash(value.prepared.request) !== request.prepared_hash) {
    throw new Error("No matching pre-dispatch reservation preparation");
  }
  const { session, state } = value;
  const prepared = value.prepared;
  delete value.prepared;
  state.pending = true;
  await persist(value);
  const usage = [];
  const messages = [];
  let started = false, ended = false, response, timer, unsubscribe;
  try {
    await new Promise((resolve, reject) => {
      unsubscribe = session.on((event) => {
        if (event.type === "assistant.usage") {
          const d = event.data;
          usage.push(Object.fromEntries([
            "model", "inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens",
            "reasoningTokens", "maxOutputTokens", "availableToolCount", "numToolCalls",
          ].map((key) => [key, d[key]])));
        }
        if (event.agentId || ended) return;
        if (event.type === "user.message") started = true;
        if (event.type === "assistant.message" && started) {
          const { messageId, content, phase } = event.data;
          if (content && ![undefined, "commentary", "final_answer"].includes(phase)) {
            reject(new Error("Unsupported message phase; private reasoning is not scientific output"));
            return;
          }
          messages.push({ messageId, content, phase });
        }
        if (event.type === "tool.execution_start" || event.type.startsWith("session.compaction")) {
          reject(new Error("Unexpected tool or compaction activity; pilot must stop"));
        }
        if (event.type === "session.error") reject(new Error(event.data.message));
        if (event.type === "session.idle" && started) {
          ended = true;
          event.data.aborted ? reject(new Error("Model turn aborted")) : resolve();
        }
      });
      timer = setTimeout(() => reject(new Error("Model response timeout")), 300000);
      const inputs = prepared.delta.map((m) => ({ prompt: m.content }));
      if (prepared.request.context_suffix) {
        inputs[inputs.length - 1].prompt += "\n\n" + prepared.request.context_suffix;
      }
      session.rpc.sendMessages({ messages: inputs, agentMode: "interactive" }).catch(reject);
    });
    const metrics = await session.rpc.usage.getMetrics();
    const assembled = visibleResponse(messages);
    response = assembled.content;
    state.logical = [...prepared.request.messages, { role: "assistant", content: response }];
    state.pending = false;
    await persist(value);
    return { ...assembled, response_collection: "ordered-public-content-v1", usage, metrics };
  } catch (error) {
    await session.abort();
    error.evidence = { content: response ?? null, visible_messages: messages, usage,
      metrics: await session.rpc.usage.getMetrics() };
    throw error;
  } finally {
    clearTimeout(timer);
    unsubscribe?.();
  }
}

try {
  await client.start();
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    let request;
    try {
      request = JSON.parse(line);
      let result;
      if (request.operation === "catalog") result = await catalog();
      else if (request.operation === "prepare") result = await prepare(request);
      else if (request.operation === "dispatch") result = await dispatch(request);
      else throw new Error("Unknown bridge operation");
      process.stdout.write(JSON.stringify({ ok: true, result }) + "\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({
        ok: false, error: String(error.message), evidence: error.evidence ?? null,
      }) + "\n");
    }
  }
} finally {
  for (const { session, state } of sessions.values()) {
    await session.disconnect();
    if (!state.pending && !state.logical.length) await client.deleteSession(session.sessionId);
  }
  await client.stop();
}
