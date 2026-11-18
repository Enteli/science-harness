import copy
from decimal import Decimal

import pytest

pytest.importorskip("numpy")

from science_harness.physics_budget import CreditLimit, ForecastBudget, PilotBudget, accounted_cost, price_envelope, upper_cost
from science_harness.physics_runtime import atomic_json


def quote():
    return {"model": "gpt-6.1-sol", "input_price": "200", "output_price": "1000",
            "batch": "1000000", "input_limit": "1000", "output_limit": "1000"}


def evidence(inputs=100, outputs=100):
    return {"usage": [{"model": "gpt-6.1-sol", "inputTokens": inputs, "outputTokens": outputs,
                       "cacheReadTokens": 80, "cacheWriteTokens": 10,
                       "availableToolCount": 0, "numToolCalls": 0}]}


def test_separate_forecast_budget_requires_explicit_approval(tmp_path):
    budget = ForecastBudget(tmp_path / "new-study.json")
    with pytest.raises(CreditLimit, match="approval"):
        budget.reserve(quote(), "not-approved")
    budget.authorize("User selected GPTonly20episodes,3000additionalcredits")
    budget.settle(budget.reserve(quote(), "one"), evidence())
    state = budget.snapshot()
    assert state["credit_cap"] == "3000" and state["request_cap"] == 2000
    assert state["assigned_episodes"] == 20
    with pytest.raises(CreditLimit, match="replace"):
        budget.authorize("No reset")


def test_unknown_dispatch_holds_reservation_across_restart(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    budget.reserve(quote(), "first")
    resumed = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    with pytest.raises(CreditLimit, match="Unreconciled"):
        resumed.reserve(quote(), "second")
    assert resumed.snapshot()["requests"][0]["debited_upper"] == "1.200000"


def test_explicit_held_continuation_preserves_cost_and_serial_dispatch(tmp_path):
    path = tmp_path / "credits.json"
    budget = PilotBudget(path, credits=2, requests=3)
    held = budget.reserve(quote(), "timed-out")
    original = copy.deepcopy(budget.snapshot()["requests"][held])
    budget.authorize_held_continuation(held, "Proceed while keeping the full hold", "a" * 64)
    resumed = PilotBudget(path, credits=2, requests=3)
    cheaper = {**quote(), "input_limit": "100", "output_limit": "100"}
    next_call = resumed.reserve(cheaper, "independent")
    with pytest.raises(CreditLimit, match="Unreconciled"):
        resumed.reserve(cheaper, "concurrent")
    resumed.settle(next_call, evidence())
    with pytest.raises(CreditLimit, match="exceeding the credit cap"):
        resumed.reserve(quote(), "too-expensive")
    assert resumed.snapshot()["requests"][held] == original
    with pytest.raises(CreditLimit, match="already recorded"):
        resumed.authorize_held_continuation(held, "Repeated approval", "a" * 64)


def test_continuation_cannot_waive_unknown_cost_or_failed_billing_evidence(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    index = budget.reserve(quote(), "first")
    with pytest.raises(ValueError, match="approval"):
        budget.authorize_held_continuation(index, "", "a" * 64)
    with pytest.raises(ValueError, match="evidence"):
        budget.authorize_held_continuation(index, "Proceed", "")
    with pytest.raises(CreditLimit):
        budget.settle(index, {"usage": []})
    with pytest.raises(CreditLimit):
        budget.authorize_held_continuation(index, "Proceed", "a" * 64)


def test_spend_is_reserved_before_a_call_and_failed_evidence_blocks(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    index = budget.reserve(quote(), "first")
    with pytest.raises(CreditLimit, match="single"):
        budget.settle(index, {"usage": []})
    assert budget.snapshot()["status"] == "blocked"
    with pytest.raises(CreditLimit):
        budget.reserve(quote(), "second")


def test_settled_upper_bound_does_not_double_count_cached_input(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    index = budget.reserve(quote(), "first")
    assert budget.settle(index, evidence()) == Decimal(".120000")
    with pytest.raises(CreditLimit, match="once"):
        budget.settle(index, evidence())
    second = budget.reserve(quote(), "second")
    budget.settle(second, evidence())
    with pytest.raises(CreditLimit, match="request cap"):
        budget.reserve(quote(), "third")


def test_insufficient_credit_prevents_dispatch_even_after_small_charge(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits="1.25", requests=2)
    index = budget.reserve(quote(), "first")
    budget.settle(index, evidence())
    with pytest.raises(CreditLimit, match="Cannot reserve"):
        budget.reserve(quote(), "second")
    assert len(budget.snapshot()["requests"]) == 1


def test_catalog_reservation_includes_long_context_and_one_hour_cache_writes():
    model = {
        "id": "claude-haiku-5.5", "policy": {"state": "enabled"},
        "capabilities": {"limits": {"max_prompt_tokens": 936000, "max_output_tokens": 128000}},
        "billing": {"tokenPrices": {
            "batchSize": 1000000, "inputPrice": 10, "outputPrice": 50,
            "cacheReadPrice": 1, "cacheWritePrice": 12.5, "cacheWrite1hPrice": 20,
            "longContext": {"inputPrice": 50, "outputPrice": 250,
                            "cacheReadPrice": 5, "cacheWritePrice": 62.5, "cacheWrite1hPrice": 100},
        }},
    }
    bound = price_envelope(model, "ai_credits")
    assert upper_cost(bound) == Decimal("125.600000")
    with pytest.raises(CreditLimit, match="unknown"):
        price_envelope(model, "premium_requests")
    broken = copy.deepcopy(model)
    del broken["billing"]["tokenPrices"]["longContext"]["cacheWritePrice"]
    with pytest.raises(CreditLimit, match="Incomplete"):
        price_envelope(broken, "ai_credits")


def test_bad_token_evidence_cannot_refund_reservation(tmp_path):
    budget = PilotBudget(tmp_path / "credits.json", credits=2, requests=2)
    index = budget.reserve(quote(), "first")
    with pytest.raises(CreditLimit, match="exceeds"):
        budget.settle(index, evidence(inputs=1001))
    assert budget.snapshot()["requests"][0]["debited_upper"] == "1.200000"


@pytest.mark.parametrize("credits", [True, float("nan"), -1, 1001])
def test_cannot_expand_or_invalidate_the_approved_credit_cap(tmp_path, credits):
    with pytest.raises(ValueError):
        PilotBudget(tmp_path / "credits.json", credits=credits)


def test_approved_extension_preserves_all_prior_charges_and_cannot_repeat(tmp_path):
    path = tmp_path / "credits.json"
    budget = PilotBudget(path, credits=10, requests=16)
    for index in range(16):
        budget.settle(budget.reserve(quote(), str(index)), evidence())
    old = copy.deepcopy(budget.snapshot()["requests"])
    budget.extend_approved_requests("User approved 32 additional requests within the same credit cap")
    state = PilotBudget(path, credits=10, requests=48).snapshot()
    assert state["credit_cap"] == "10" and state["request_cap"] == 48
    assert state["requests"] == old
    assert state["approval_extensions"][0]["requests_at_approval"] == 16
    with pytest.raises(ValueError):
        budget.extend_approved_requests("Do not apply twice")
    with pytest.raises(CreditLimit):
        PilotBudget(path, credits=10, requests=16).reserve(quote(), "stale caller")


def test_new_ledger_cannot_silently_grant_an_extended_approval(tmp_path):
    with pytest.raises(CreditLimit, match="existing"):
        PilotBudget(tmp_path / "new.json", requests=48).reserve(quote(), "unapproved")


def test_full_approval_preserves_pending_hold_and_cannot_create_new_ledger(tmp_path):
    path = tmp_path / "credits.json"
    budget = PilotBudget(path)
    for index in range(16):
        budget.settle(budget.reserve(quote(), str(index)), evidence())
    budget.extend_approved_requests("Original development extension")
    budget.reserve(quote(), "unresolved")
    original = copy.deepcopy(budget.snapshot()["requests"])
    budget.extend_scientific_approval("User authorized validation and full80 within2000credits")
    state = PilotBudget(path, credits=2000, requests=8048).snapshot()
    assert state["credit_cap"] == "2000" and state["request_cap"] == 8048
    assert state["requests"] == original
    with pytest.raises(CreditLimit, match="Unreconciled"):
        budget.reserve(quote(), "must-still-explicitly-authorize-hold")
    with pytest.raises(ValueError):
        budget.extend_scientific_approval("No second doubling")
    with pytest.raises(CreditLimit, match="existing"):
        PilotBudget(tmp_path / "new-full.json", credits=2000, requests=8048).reserve(quote(), "unapproved")


def billed_evidence(nano=100000000, count=1):
    return {**evidence(), "metrics": {
        "aiCreditsStatus": "complete", "sessionStartTime": "owned-test-session",
        "modelMetrics": {"gpt-6.1-sol": {
            "aiCreditsStatus": "complete", "totalNanoAiu": nano, "requests": {"count": count}}}}}


def test_authoritative_reconciliation_preserves_rows_and_holds(tmp_path):
    path = tmp_path / "ledger.json"
    budget = PilotBudget(path, credits="1.31", requests=2)
    budget.settle(budget.reserve(quote(), "first"), billed_evidence())
    state = budget.snapshot()
    original = copy.deepcopy(state["requests"])
    del state["actual_settlements"]
    atomic_json(path, state)
    with pytest.raises(CreditLimit, match="credit cap"):
        budget.reserve(quote(), "before-reconciliation")
    assert budget.reconcile_authoritative_charges() == Decimal(".1")
    assert budget.snapshot()["requests"] == original
    budget.reserve(quote(), "second")
    assert accounted_cost(budget.snapshot()) == Decimal("1.3")
    assert budget.snapshot()["requests"][1]["debited_upper"] == "1.200000"


@pytest.mark.parametrize("bad", [billed_evidence(nano=200000000), billed_evidence(count=2)])
def test_unproven_credit_reconciliation_cannot_refund_reservation(tmp_path, bad):
    budget = PilotBudget(tmp_path / "ledger.json", credits=2, requests=2)
    index = budget.reserve(quote(), "first")
    with pytest.raises(CreditLimit):
        budget.settle(index, bad)
    state = budget.snapshot()
    assert state["requests"][0]["status"] == "reserved"
    assert accounted_cost(state) == Decimal("1.2")


def test_3000_credit_extension_requires_approval_and_a_safe_dispatch_boundary(tmp_path):
    path = tmp_path / "ledger.json"
    budget = PilotBudget(path)
    for index in range(16):
        budget.settle(budget.reserve(quote(), str(index)), evidence())
    budget.extend_approved_requests("Development approval")
    budget.extend_scientific_approval("Full80 within2000 approval")
    pending = budget.reserve(quote(), "interrupted")
    with pytest.raises(CreditLimit, match="live or unreviewed"):
        budget.extend_credit_approval("Raise to3000")
    budget.authorize_held_continuation(pending, "Retain the full terminated hold", "a" * 64)
    old = copy.deepcopy(budget.snapshot()["requests"])
    budget.extend_credit_approval("User approved3000 for the same80 assignments")
    current = PilotBudget(path, credits=3000, requests=8048).snapshot()
    assert current["requests"] == old
    assert current["credit_cap"] == "3000" and current["request_cap"] == 8048
    with pytest.raises(ValueError):
        budget.extend_credit_approval("No repeated increase")
    with pytest.raises(CreditLimit, match="existing"):
        PilotBudget(tmp_path / "fresh.json", credits=3000, requests=8048).reserve(quote(), "unapproved")
