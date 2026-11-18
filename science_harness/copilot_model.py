"""Optional text-only Copilot transport, using an existing operator login."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time


class CopilotLLM:
    def __init__(self, model="gpt-5.4-mini", effort="medium", timeout=180, cli_version=None):
        self.executable = shutil.which("copilot")
        if not self.executable:
            raise RuntimeError("Copilot CLI is not installed; use the OpenAI-compatible client")
        self.model, self.effort, self.timeout = model, effort, timeout
        self.last_usage = {}
        allowed = {
            "PATH", "HOME", "TMPDIR", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR",
            "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "GH_HOST",
        }
        # Use the stored CLI login, never repository or child-session token variables.
        self.env = {key: value for key, value in os.environ.items() if key in allowed}
        selected = cli_version if cli_version is not None else os.environ.get("SCIENCE_HARNESS_COPILOT_CLI_VERSION")
        with tempfile.TemporaryDirectory(prefix="science-harness-cli-check-") as work:
            def inspect(arguments):
                return subprocess.check_output(arguments, cwd=work, env=self.env,
                                               stdin=subprocess.DEVNULL, text=True, timeout=30).strip()

            if selected is None:
                selected = self._version_number(inspect([self.executable, "--version"]))
            if not isinstance(selected, str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", selected):
                raise ValueError("cli_version must be an exact cached semantic version")
            self.command_prefix = [self.executable, "--prefer-version", selected, "--no-auto-update"]
            self.env["COPILOT_AUTO_UPDATE"] = "false"
            self.version = inspect(self.command_prefix + ["--version"])
            if self._version_number(self.version) != selected:
                raise RuntimeError("Copilot did not load the pinned CLI version; no inference was attempted")
            help_text = inspect(self.command_prefix + ["--help"])
        required = ("--model", "--reasoning-effort", "--context", "--available-tools",
                    "--disable-builtin-mcps", "--disable-mcp-server", "--no-custom-instructions", "--output-format")
        flags = set(re.findall(r"(?m)^\s*(?:-\w,\s+)?(--[\w-]+)\b", help_text))
        missing = [flag for flag in required if flag not in flags]
        if missing:
            raise RuntimeError("Copilot lacks required text-only transport flags: " + ", ".join(missing))
        self.cli_version = selected
        self.optional_flags = []
        if "--no-remote" in flags:
            self.optional_flags.append("--no-remote")
        if "--sync-session" in flags:
            self.optional_flags.append("--sync-session=off")

    @staticmethod
    def _version_number(output):
        first = output.splitlines()[0] if output else ""
        prefix = "GitHub Copilot CLI "
        if not first.startswith(prefix):
            raise RuntimeError("Unrecognized Copilot CLI version; cannot pin the transport")
        return first[len(prefix):].rstrip(".")

    def complete(self, system, user):
        self.last_usage = {}
        prompt = (
            "Act only as the scientific decision function below. The provided instrument "
            "actions are JSON output, not host tool calls. Reply with ONE JSON object, "
            "without Markdown or additional commentary.\n\n"
            + system + "\n\nINVESTIGATION INPUT:\n" + user
        )
        if len(prompt.encode()) > 180000:
            raise ValueError("Copilot transport prompt exceeds its explicit 180000-byte limit")
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="science-harness-inference-") as work:
            command = self.command_prefix + [
                "--model", self.model, "--reasoning-effort", self.effort,
                "--context", "default", "--available-tools=science_harness_no_tools",
                "--disable-builtin-mcps", "--disable-mcp-server", "computer-use",
                "--no-custom-instructions", *self.optional_flags, "--output-format", "json",
                "-p", prompt,
            ]
            result = subprocess.run(
                command, cwd=work, env=self.env, stdin=subprocess.DEVNULL,
                text=True, capture_output=True, timeout=self.timeout,
            )
        self.last_usage = {"elapsed_seconds": time.monotonic() - start, "cli_version": self.cli_version}
        if result.returncode:
            raise RuntimeError(f"Copilot inference failed ({result.returncode}): {result.stderr[-1000:]}")
        answer, usage, tool_counts = None, None, []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind, data = event.get("type"), event.get("data", {})
            if kind == "tool.execution_start":
                raise RuntimeError("Isolation violation: Copilot attempted a host tool")
            if kind == "assistant.message" and data.get("content"):
                answer = data["content"]
            if kind == "session.usage_checkpoint":
                snapshot = data.get("accountingSnapshot", {})
                model_metrics = snapshot.get("modelMetrics", {})
                if model_metrics and set(model_metrics) != {self.model}:
                    raise RuntimeError("Model changed during the experiment")
                usage = model_metrics.get(self.model, {}).get("usage")
                for conversation in data.get("promptCacheBreakState", []):
                    for state in conversation.get("models", {}).values():
                        tool_counts.append(state.get("tool_count"))
        if not tool_counts or any(count != 0 for count in tool_counts):
            raise RuntimeError("Could not verify a zero-host-tool inference boundary")
        if not answer or usage is None:
            raise RuntimeError("Copilot returned no assistant answer or usage accounting")
        # Do not retain encrypted reasoning blocks, provider IDs, or CLI-internal prompts.
        self.last_usage.update({
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "reasoning_tokens": usage.get("reasoningTokens"),
            "cache_read_tokens": usage.get("cacheReadTokens"),
            "host_tool_count": 0,
        })
        return answer
