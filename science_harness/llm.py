"""Model interface: complete(system, user) -> str. No provider SDK required."""
import json
import os
import urllib.request
import hashlib
import time
from typing import List, Optional, Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAICompatLLM:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model or os.environ.get("SCIENCE_HARNESS_MODEL", "gpt-4o")
        self.base_url = (base_url or os.environ.get(
            "SCIENCE_HARNESS_BASE_URL", "https://api.openai.com/v1"
        )).rstrip("/")
        self.api_key = api_key or os.environ.get("SCIENCE_HARNESS_API_KEY") or os.environ.get(
            "OPENAI_API_KEY", ""
        )
        self.last_usage = {}

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + "/chat/completions", body, headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.load(response)
        usage = payload.get("usage", {})
        self.last_usage = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Provider response has no assistant message") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Provider returned empty or non-text content")
        return content


class ScriptedLLM:
    """Finite replay fixture. Exhaustion is a failure, never an invented completion."""
    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise RuntimeError("ScriptedLLM responses exhausted")
        return self.responses.pop(0)


def create_model(provider, model=None, effort="medium"):
    if provider == "copilot":
        from .copilot_model import CopilotLLM

        return CopilotLLM(model or "gpt-5.4-mini", effort=effort)
    if provider == "llm":
        if effort != "medium":
            raise ValueError("Explicit reasoning-effort selection is supported by the Copilot transport only")
        return OpenAICompatLLM(model=model)
    raise ValueError("Scientific reasoning requires the copilot or llm provider")


class MeteredModel:
    """Capture framework requests, public responses and reported usage, not provider internals."""
    def __init__(self, delegate, *, calls=None, role=None):
        self.delegate = delegate
        self.calls = [] if calls is None else calls
        self.role = role
        self.last_usage = {}

    def complete(self, system, user):
        started = time.monotonic()
        if hasattr(self.delegate, "last_usage"):
            self.delegate.last_usage = {}
        call = {"started_at": time.time(), "system": system,
                "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                "request_sha256": hashlib.sha256(user.encode()).hexdigest(),
                "prompt_bytes": len(system.encode()) + len(user.encode()),
                "request": json.loads(user)}
        if self.role is not None:
            call.update(role=self.role, model=getattr(self.delegate, "model", None),
                        reasoning_effort=getattr(self.delegate, "effort", None))
        self.calls.append(call)
        try:
            response = self.delegate.complete(system, user)
            call["response"] = response
            return response
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.last_usage = dict(getattr(self.delegate, "last_usage", {}))
            call.update(usage=self.last_usage, seconds=time.monotonic()-started, ended_at=time.time())
