"""Conservative admission without silently dropping mandatory history."""
import os
import math
from pathlib import Path

PROMPT_LIMIT = 131072
FRAMING_RESERVE = 2048


class ContextLimit(RuntimeError):
    pass


def encoding():
    import tiktoken
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path.cwd() / ".cache/tiktoken"))
    return tiktoken.get_encoding("o200k_base")


def text_tokens(text):
    return len(encoding().encode(text, disallowed_special=()))


def admit_context(packet, prepared=None):
    messages = packet["messages"]
    # The SDK's Claude estimate undercounted numeric-history payloads in validation.
    # This empirical safety margin is not an exact provider tokenizer.
    multiplier = 2 if packet.get("model") == "claude-haiku-5.5" else 1
    visible = text_tokens(packet["system"]) + sum(text_tokens(m["content"]) for m in messages)
    visible += text_tokens(packet.get("context_suffix", ""))
    estimated = multiplier * visible + FRAMING_RESERVE + 8 * len(messages)
    effective_limit = PROMPT_LIMIT
    if prepared is not None:
        runtime = prepared["runtime_context"]
        if (type(runtime.get("totalTokens")) is not int or runtime["totalTokens"] < 0
                or type(prepared.get("logical_prefix_length")) is not int):
            raise RuntimeError("Runtime context accounting is unavailable")
        prefix = prepared["logical_prefix_length"]
        if not 0 <= prefix <= len(messages):
            raise RuntimeError("Invalid native-history prefix length")
        pending = sum(text_tokens(m["content"]) for m in messages[prefix:])
        pending += text_tokens(packet.get("context_suffix", ""))
        runtime_pending = prepared.get("pending_text_runtime_tokens")
        if type(runtime_pending) is not int or runtime_pending < 0:
            raise RuntimeError("Runtime pending-text accounting is unavailable")
        pending = max(multiplier * pending, runtime_pending)
        metrics = prepared.get("metrics", {})
        prior = [metrics.get(k) for k in ("lastCallInputTokens", "lastCallOutputTokens")]
        if metrics.get("totalUserRequests", 0) == 0:
            previous = 0
        elif any(type(v) is not int or v < 0 for v in prior):
            raise RuntimeError("Prior actual token accounting is unavailable")
        else:
            previous = sum(prior)
        existing = max(runtime["totalTokens"], previous)
        estimated = max(estimated, existing + pending + FRAMING_RESERVE + 8 * (len(messages) - prefix))
        info = prepared.get("context_info")
        if not isinstance(info, dict):
            raise RuntimeError("Runtime input limit is unavailable")
        limit = info.get("promptTokenLimit")
        if type(limit) is not int or limit <= 0:
            raise RuntimeError("Runtime input limit is unavailable")
        effective_limit = min(effective_limit, limit)
        threshold = info.get("compactionThreshold")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
                raise RuntimeError("Runtime compaction threshold is invalid")
            effective_limit = min(effective_limit, math.floor(threshold))
    if estimated > effective_limit:
        raise ContextLimit(
            f"Complete history exceeds input allowance: estimate+reserve={estimated}, limit={effective_limit}; "
            "no truncation or paid dispatch")
    return {"visible_text_tokens": visible, "admitted_estimate": estimated,
            "framing_reserve": FRAMING_RESERVE, "effective_limit": effective_limit,
            "visible_token_safety_multiplier": multiplier,
            "exact_provider_tokenizer": False}
