"""Fresh, tool-free decision contexts with study-local, fail-closed accounting."""
from decimal import Decimal
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess

from ..physics_budget import (
    CreditLimit, PilotBudget, accounted_cost, number, price_envelope,
    reconcile_actual, upper_cost,
)
from ..physics_reference import digest
from ..harness.io import encode
from ..harness.runner import write_json


class StudyBudget(PilotBudget):
    def __init__(self, path, credits, requests, approval):
        if number(credits) <= 0 or type(requests) is not int or requests < 1:
            raise ValueError("A paid study needs positive credit and request caps")
        self.path, self.credits, self.requests = Path(path), number(credits), requests
        self.approval = digest(encode(approval))
        if self.path.exists():
            raise CreditLimit("Use a NEW study ledger; automatic resumption is disabled")
        write_json(self.path, self.initial_state())

    def initial_state(self):
        return {"credit_cap": str(self.credits), "request_cap": self.requests,
                "approval": self.approval, "requests": [], "status": "ready"}

    def validate_approval(self, state):
        if state.get("approval") != self.approval:
            raise CreditLimit("Study approval/configuration changed")

    def settle(self, index, evidence):
        with self.locked() as state:
            row = state["requests"][index]
            usage, metrics = evidence.get("usage"), evidence.get("metrics", {})
            model = row["quote"]["model"]
            summary = metrics.get("modelMetrics", {}).get(model, {})
            valid = (
                row["status"] == "reserved"
                and isinstance(usage, list) and len(usage) == 1
                and usage[0].get("model") == model
                and usage[0].get("availableToolCount") == 0
                and usage[0].get("numToolCalls") == 0
                and metrics.get("totalUserRequests") == 1
                and summary.get("requests", {}).get("count") == 1
                and metrics.get("totalNanoAiu") == summary.get("totalNanoAiu")
                and evidence.get("shutdown_verified") is True
            )
            if not valid:
                state["status"] = "blocked"
                write_json(self.path, state)
                raise CreditLimit("Missing owned, single-request, zero-tool billing/shutdown evidence")
            for key, bound in (("inputTokens", "input_limit"), ("outputTokens", "output_limit")):
                value = usage[0].get(key)
                if type(value) is not int or value < 0 or number(value) > number(row["quote"][bound]):
                    raise CreditLimit("Usage exceeds the full provider reservation")
            row.update(status="settled", evidence=evidence)
            reconcile_actual(state)
            write_json(self.path, state)
            return number(state["actual_settlements"][str(index)]["credits"])


class GuardedCopilotModel:
    """Implements complete(system, user); it never executes an agent action."""
    def __init__(self, directory, model, budget, cli_version):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.model, self.budget, self.last_usage = model, budget, {}
        self.index = 0
        cli = shutil.which("copilot")
        if not cli:
            raise RuntimeError("Copilot CLI unavailable; no paid fallback")
        version = subprocess.check_output(
            [cli, "--prefer-version", cli_version, "--no-auto-update", "--version"],
            text=True, timeout=30).splitlines()[0]
        if version != f"GitHub Copilot CLI {cli_version}.":
            raise RuntimeError("The exact declared Copilot CLI is unavailable")
        env = {k: v for k, v in os.environ.items() if k in {
            "PATH", "HOME", "TMPDIR", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR",
            "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "GH_HOST", "GH_TOKEN",
            "SCIENCE_HARNESS_SDK_ROOT"}}
        env.update(COPILOT_CLI_PATH=cli, COPILOT_CLI_VERSION=cli_version,
                   COPILOT_AUTO_UPDATE="false")
        self.stderr = (self.directory / "stderr.log").open("x")
        self.process = subprocess.Popen(
            ["node", str(Path(__file__).with_name("guarded_bridge.mjs")), str(self.directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, env=env, start_new_session=True)
        try:
            catalog = self.rpc("catalog")
            models = {m["id"]: m for m in catalog["models"]}
            units = {u["modelId"]: u["unit"] for u in catalog["units"]}
            if model not in models:
                raise CreditLimit("Configured model is unavailable; no fallback")
            quote = price_envelope(models[model], units.get(model))
            write_json(self.directory / "quote.json", {"model": models[model], "quote": quote,
                       "sdk_resolution": catalog["sdkResolution"],
                       "full_provider_reservation": str(upper_cost(quote))})
            if upper_cost(quote) > budget.credits:
                raise CreditLimit("The approved cap cannot cover one full provider reservation")
            self.quote = quote
        except BaseException:
            self.close()
            raise

    def rpc(self, operation, **payload):
        self.process.stdin.write(encode({"operation": operation, **payload}) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 360)[0]:
            raise RuntimeError("Decision bridge timeout; keep the complete reservation")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Decision bridge exited; inspect stderr; no retry")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def complete(self, system, user):
        self.last_usage = {}
        packet = {"model": self.model, "system": system, "user": user}
        request_hash = digest(json.dumps(packet, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False))
        prefix = self.directory / f"call-{self.index:03d}"
        write_json(prefix.with_suffix(".request.json"), packet)
        ready = self.rpc("prepare", **packet)
        write_json(prefix.with_suffix(".prepare.json"), ready)
        if ready["request_sha256"] != request_hash:
            raise RuntimeError("Prepared decision packet changed")
        reservation = self.budget.reserve(self.quote, request_hash)
        try:
            result = self.rpc("dispatch", request_sha256=request_hash)
        except Exception as exc:
            write_json(prefix.with_suffix(".failure.json"),
                       {"error": f"{type(exc).__name__}: {exc}", "reservation": reservation})
            raise
        write_json(prefix.with_suffix(".receipt.json"), result)
        if result.get("request_sha256") != request_hash:
            raise RuntimeError("Dispatched decision packet changed; keep reservation")
        credits = self.budget.settle(reservation, result)
        usage = result["usage"][0]
        self.last_usage = {
            "input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
            "reasoning_tokens": usage.get("reasoningTokens"), "host_tool_count": 0,
            "credits": str(credits), "request_sha256": request_hash,
            "session_id": result["session_id"], "shutdown_verified": True,
        }
        self.index += 1
        return result["content"]

    def accounting(self):
        state = self.budget.snapshot()
        unresolved = sum(r["status"] != "settled" for r in state["requests"])
        actual = sum((number(v["credits"]) for v in state.get("actual_settlements", {}).values()),
                     Decimal(0))
        return {"accounted_credits": str(accounted_cost(state)), "confirmed_credits": str(actual),
                "unresolved_requests": unresolved}

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.process.stdout.close()
        self.stderr.close()
