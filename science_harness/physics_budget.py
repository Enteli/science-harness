"""Fail-closed, serial reservations for the approved development pilot."""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import fcntl
import json
import math
from pathlib import Path

from .physics_reference import digest
from .harness.io import atomic_json, encode


class CreditLimit(RuntimeError):
    pass


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("Invalid credit quantity")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Credit quantities must be finite and nonnegative")
    return result


def price_envelope(model, unit):
    if unit != "ai_credits":
        raise CreditLimit("Cannot reserve against legacy or unknown billing units")
    prices = model.get("billing", {}).get("tokenPrices")
    limits = model.get("capabilities", {}).get("limits")
    if not prices or not limits or model.get("policy", {}).get("state") != "enabled":
        raise CreditLimit("Missing model price, limit, or admission metadata")
    tiers = [prices]
    if prices.get("longContext"):
        tiers.append(prices["longContext"])
    inputs, outputs = [], []
    for tier in tiers:
        for field in ("inputPrice", "outputPrice", "cacheReadPrice", "cacheWritePrice"):
            if field not in tier:
                raise CreditLimit("Incomplete token price classes")
        inputs += [number(tier[k]) for k in
                   ("inputPrice", "cacheReadPrice", "cacheWritePrice", "cacheWrite1hPrice")
                   if k in tier]
        outputs.append(number(tier["outputPrice"]))
    batch = number(prices["batchSize"])
    input_limit = number(limits["max_prompt_tokens"])
    output_limit = number(limits["max_output_tokens"])
    if not batch or not input_limit or not output_limit:
        raise CreditLimit("Invalid provider token limits")
    # Reserve using full provider limits, not the benchmark's post-response allowances.
    return {"model": model["id"], "input_price": str(max(inputs)),
            "output_price": str(max(outputs)), "batch": str(batch),
            "input_limit": str(input_limit), "output_limit": str(output_limit)}


def upper_cost(quote, input_tokens=None, output_tokens=None):
    inputs = number(quote["input_limit"] if input_tokens is None else input_tokens)
    outputs = number(quote["output_limit"] if output_tokens is None else output_tokens)
    value = (inputs * number(quote["input_price"]) +
             outputs * number(quote["output_price"])) / number(quote["batch"])
    return value.quantize(Decimal(".000001"), rounding=ROUND_CEILING)


def accounted_cost(state):
    total = Decimal(0)
    for row in state["requests"]:
        actual = state.get("actual_settlements", {}).get(str(row["id"]))
        if actual is None:
            total += number(row["debited_upper"])
            continue
        charge = number(actual["credits"])
        if (row["status"] != "settled" or actual["request_sha256"] != row["request_sha256"]
                or charge > number(row["debited_upper"])):
            raise CreditLimit("Invalid authoritative settlement")
        total += charge
    return total


def reconcile_actual(state):
    """Append accounting evidence; never alter original request records or unknown holds."""
    previous, settlements = {}, {}
    for row in state["requests"]:
        if row["status"] != "settled":
            continue
        evidence = row["evidence"]
        metrics = evidence.get("metrics")
        if metrics is None:
            continue
        model = row["quote"]["model"]
        summary = metrics.get("modelMetrics", {}).get(model, {})
        started = metrics.get("sessionStartTime")
        nano = summary.get("totalNanoAiu")
        count = summary.get("requests", {}).get("count")
        if (metrics.get("aiCreditsStatus") != "complete"
                or summary.get("aiCreditsStatus") != "complete"
                or not isinstance(started, str) or not started
                or type(nano) is not int or nano < 0 or type(count) is not int):
            raise CreditLimit("Incomplete authoritative credit evidence")
        key = (model, started)
        old_nano, old_count = previous.get(key, (0, 0))
        if nano < old_nano or count != old_count + 1:
            raise CreditLimit("Noncontiguous authoritative credit evidence")
        credits = Decimal(nano - old_nano) / Decimal(10**9)
        if credits > number(row["debited_upper"]):
            raise CreditLimit("Actual charge exceeds the settled conservative bound")
        settlements[str(row["id"])] = {
            "credits": str(credits), "request_sha256": row["request_sha256"],
            "evidence_sha256": digest(encode(evidence)),
            "source": "complete-sdk-totalNanoAiu-delta",
        }
        previous[key] = (nano, count)
    for key, value in state.get("actual_settlements", {}).items():
        if settlements.get(key) != value:
            raise CreditLimit("Previously reconciled billing evidence changed")
    state["actual_settlements"] = settlements


class PilotBudget:
    def __init__(self, path, credits=1000, requests=16):
        if (not (0 < number(credits) <= 1000 or number(credits) in (2000, 3000))
                or type(requests) is not int or not (1 <= requests <= 16 or requests in (48, 8048))
                or ((number(credits) in (2000, 3000)) != (requests == 8048))):
            raise ValueError("Only the initial pilot and explicitly approved extensions are supported")
        self.path = Path(path)
        self.credits, self.requests = number(credits), requests
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def validate_approval(self, state):
        if self.requests > 16 and not any(
                row.get("previous_request_cap") == 16 and row.get("request_cap") == 48
                and row.get("additional_requests") == 32 and row.get("approval")
                for row in state.get("approval_extensions", [])):
            raise CreditLimit("Extended request cap lacks its explicit approval record")
        if self.requests == 8048:
            approved = state.get("scientific_approval", {})
            if (approved.get("previous_credit_cap") != "1000"
                    or approved.get("credit_cap") != "2000"
                    or approved.get("previous_request_cap") != 48
                    or approved.get("request_cap") != 8048
                    or not approved.get("approval")):
                raise CreditLimit("Full benchmark lacks its explicit approval record")
        if self.credits == 3000 and not any(
                row.get("previous_credit_cap") == "2000"
                and row.get("credit_cap") == "3000"
                and row.get("request_cap_unchanged") == 8048 and row.get("approval")
                for row in state.get("credit_extensions", [])):
            raise CreditLimit("The3000-credit cap lacks its explicit approval record")

    def initial_state(self):
        if self.requests > 16:
            raise CreditLimit("An extension must preserve the existing pilot ledger")
        return {"credit_cap": str(self.credits), "request_cap": self.requests,
                "requests": [], "status": "ready"}

    @contextmanager
    def locked(self):
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if self.path.exists():
                    state = json.loads(self.path.read_text())
                    if number(state["credit_cap"]) != self.credits or state["request_cap"] != self.requests:
                        raise CreditLimit("Cannot change an existing pilot's approved caps")
                    self.validate_approval(state)
                else:
                    state = self.initial_state()
                yield state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def extend_approved_requests(self, approval):
        if self.requests != 16 or not isinstance(approval, str) or not approval.strip():
            raise ValueError("The explicit 16-to-48 request approval is required")
        with self.locked() as state:
            if (not self.path.exists() or state["status"] != "ready"
                    or len(state["requests"]) != 16
                    or any(row["status"] != "settled" for row in state["requests"])):
                raise CreditLimit("The original sixteen requests must be settled before extending")
            state.setdefault("approval_extensions", []).append({
                "previous_request_cap": 16, "request_cap": 48, "additional_requests": 32,
                "credit_cap_unchanged": state["credit_cap"], "requests_at_approval": 16,
                "approval": approval, "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            state["request_cap"] = 48
            atomic_json(self.path, state)
        self.requests = 48

    def reserve(self, quote, request_hash):
        with self.locked() as state:
            held = {record["request_id"] for record in state.get("held_continuations", [])}
            if state["status"] != "ready" or any(
                    row["status"] == "reserved" and row["id"] not in held
                    for row in state["requests"]):
                raise CreditLimit("Unreconciled dispatch; no additional paid requests allowed")
            if len(state["requests"]) >= self.requests:
                raise CreditLimit("Pilot request cap reached")
            cost = upper_cost(quote)
            used = accounted_cost(state)
            if used + cost > self.credits:
                raise CreditLimit("Cannot reserve the next call without exceeding the credit cap")
            index = len(state["requests"])
            state["requests"].append({"id": index, "request_sha256": request_hash, "quote": quote,
                                      "reserved": str(cost), "debited_upper": str(cost),
                                      "status": "reserved"})
            atomic_json(self.path, state)
            return index

    def extend_scientific_approval(self, approval):
        if self.credits != 1000 or self.requests != 48 or not isinstance(approval, str) or not approval.strip():
            raise ValueError("Explicit 2000-credit/full-panel approval is required")
        with self.locked() as state:
            if not self.path.exists() or state["status"] != "ready" or state.get("scientific_approval"):
                raise CreditLimit("Cannot extend this ledger")
            state["scientific_approval"] = {
                "previous_credit_cap": "1000", "credit_cap": "2000",
                "previous_request_cap": 48, "request_cap": 8048,
                "development_request_cap": 48, "scientific_assignments": 80,
                "requests_per_assignment": 100, "approval": approval,
                "requests_at_approval": len(state["requests"]),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            state["credit_cap"], state["request_cap"] = "2000", 8048
            atomic_json(self.path, state)
        self.credits, self.requests = Decimal(2000), 8048

    def extend_credit_approval(self, approval):
        if (self.credits != 2000 or self.requests != 8048
                or not isinstance(approval, str) or not approval.strip()):
            raise ValueError("Explicit2000-to3000-credit continuation approval is required")
        with self.locked() as state:
            held = {r["request_id"] for r in state.get("held_continuations", [])}
            if state["status"] != "ready" or any(
                    r["status"] == "reserved" and r["id"] not in held for r in state["requests"]):
                raise CreditLimit("Do not change caps underneath a live or unreviewed dispatch")
            state.setdefault("credit_extensions", []).append({
                "previous_credit_cap": "2000", "credit_cap": "3000",
                "request_cap_unchanged": 8048, "assignments_unchanged": 80,
                "approval": approval, "requests_at_approval": len(state["requests"]),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            state["credit_cap"] = "3000"
            atomic_json(self.path, state)
        self.credits = Decimal(3000)

    def authorize_held_continuation(self, index, approval, terminal_evidence_sha256):
        if (type(index) is not int or index < 0 or not isinstance(approval, str)
                or not approval.strip() or not isinstance(terminal_evidence_sha256, str)
                or len(terminal_evidence_sha256) != 64
                or any(c not in "0123456789abcdef" for c in terminal_evidence_sha256)):
            raise ValueError("Explicit continuation approval and terminal evidence hash are required")
        with self.locked() as state:
            if state["status"] != "ready" or index >= len(state["requests"]):
                raise CreditLimit("Cannot authorize this held dispatch")
            row = state["requests"][index]
            if (row["status"] != "reserved" or row["id"] != index
                    or number(row["debited_upper"]) != number(row["reserved"])
                    or number(row["reserved"]) != upper_cost(row["quote"])):
                raise CreditLimit("The unresolved dispatch must retain its full reservation")
            if any(r["request_id"] == index for r in state.get("held_continuations", [])):
                raise CreditLimit("Held continuation already recorded")
            state.setdefault("held_continuations", []).append({
                "request_id": index, "approval": approval,
                "terminal_evidence_sha256": terminal_evidence_sha256,
                "held_credits_unchanged": row["reserved"],
                "credit_cap_unchanged": state["credit_cap"],
                "request_cap_unchanged": state["request_cap"],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            atomic_json(self.path, state)

    def settle(self, index, evidence):
        with self.locked() as state:
            row = state["requests"][index]
            if row["status"] != "reserved":
                raise CreditLimit("A dispatch may be reconciled only once")
            usage = evidence.get("usage")
            if (not isinstance(usage, list) or len(usage) != 1
                    or usage[0].get("model") != row["quote"]["model"]
                    or usage[0].get("availableToolCount") != 0
                    or usage[0].get("numToolCalls") != 0):
                state["status"] = "blocked"
                atomic_json(self.path, state)
                raise CreditLimit("Cannot verify a single, zero-tool billed API call")
            event = usage[0]
            for key, bound in (("inputTokens", "input_limit"), ("outputTokens", "output_limit")):
                if (type(event.get(key)) is not int or event[key] < 0
                        or number(event[key]) > number(row["quote"][bound])):
                    state["status"] = "blocked"
                    atomic_json(self.path, state)
                    raise CreditLimit("Usage is missing or exceeds the reserved provider limit")
            charged = upper_cost(row["quote"], event["inputTokens"], event["outputTokens"])
            row.update(status="settled", debited_upper=str(charged), evidence=evidence)
            reconcile_actual(state)
            atomic_json(self.path, state)
            return charged

    def reconcile_authoritative_charges(self):
        with self.locked() as state:
            reconcile_actual(state)
            atomic_json(self.path, state)
            return accounted_cost(state)

    def snapshot(self):
        with self.locked() as state:
            return state


class ForecastBudget(PilotBudget):
    """Separately approved20-episode study; no access to the completed study's balance."""
    SCOPE, EPISODES, REQUESTS = "forecast-advisory-gpt20-v1", 20, 2000

    def __init__(self, path):
        self.path, self.credits, self.requests = Path(path), Decimal(3000), self.REQUESTS
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def validate_approval(self, state):
        if (state.get("scope") != self.SCOPE
                or not state.get("approval") or state.get("assigned_episodes") != self.EPISODES):
            raise CreditLimit("Comparison lacks explicit study approval")

    def initial_state(self):
        raise CreditLimit("Record the explicit forecast-study approval before any dispatch")

    def authorize(self, approval):
        if not isinstance(approval, str) or not approval.strip():
            raise ValueError("Explicit approval is required")
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if self.path.exists():
                    raise CreditLimit("Cannot replace an existing forecast ledger")
                atomic_json(self.path, {
                    "scope": self.SCOPE, "credit_cap": "3000",
                    "request_cap": self.REQUESTS, "assigned_episodes": self.EPISODES, "approval": approval,
                    "requests": [], "status": "ready",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class ExtraDimensionsBudget(ForecastBudget):
    SCOPE, EPISODES, REQUESTS = "extra-dimensions-eight-arms-v1", 40, 4000
