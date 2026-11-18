import { createHash } from "node:crypto";

export function canonicalJson(value) {
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (value !== null && typeof value === "object") {
    return "{" + Object.keys(value).sort()
      .map((key) => JSON.stringify(key) + ":" + canonicalJson(value[key])).join(",") + "}";
  }
  return JSON.stringify(value);
}

export const hash = (value) => createHash("sha256").update(canonicalJson(value)).digest("hex");

export function visibleResponse(messages) {
  const parts = new Map();
  for (const message of messages) {
    if (typeof message.content !== "string" || !message.content.trim()) continue;
    if (![undefined, "commentary", "final_answer"].includes(message.phase)) {
      throw new Error("Unsupported message phase; reasoning is not scientific output");
    }
    if (typeof message.messageId !== "string") throw new Error("Visible message lacks an identity");
    const previous = parts.get(message.messageId);
    if (previous && previous.content !== message.content) {
      throw new Error("Conflicting visible message revisions");
    }
    parts.set(message.messageId, {
      messageId: message.messageId, content: message.content, phase: message.phase,
    });
  }
  const visible = [...parts.values()];
  if (!visible.length) throw new Error("No nonempty user-visible scientific response");
  return { content: visible.map((message) => message.content).join(""), visible_messages: visible };
}

export function historyDelta(messages, logical) {
  if (hash(messages.slice(0, logical.length)) !== hash(logical)) {
    throw new Error("Native role/history prefix changed");
  }
  const delta = messages.slice(logical.length);
  if (!delta.length || delta.some((m) => m.role !== "user" || typeof m.content !== "string")) {
    throw new Error("Cannot seed or silently flatten assistant-role history");
  }
  return delta;
}
