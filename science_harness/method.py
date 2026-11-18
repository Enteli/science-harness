"""Optional scientific-method sequencing; artifacts are reasoning, not acquired evidence."""
import copy

from .books import encode, strings, text
from .tools import completed_acquisitions


VERSION = 1
ARTIFACT_BYTES = 1800
DESIGN_POLICIES = ("direct", "compare")
METHOD_SYSTEM = """
A skill-sequenced scientific method is enabled. Follow scientific_method.active_skill.
Its method actions supplement the ordinary action catalogue. Plans and hypotheses are
not observations. Read-only lookup/analysis remains available and costs decisions.
Use method_route with a reason for an explicit departure, not an unrecorded stage skip.
Finish is always available for a scoped result or limitation; it does not certify
unfinished stages. Execution and transitions do not make extra model calls.
"""
SKILLS = {
    "hypothesize": {
        "name": "generate-hypothesis",
        "instruction": "Frame the goal using the apparatus contract, current evidence and scoped priors. "
                       "Emit one or two concise, credible hypotheses with different testable consequences when "
                       "justified, not cosmetic variants or invented measurements. At most three hypotheses. "
                       "An empty set with a reason requests exploration rather than manufactured alternatives. "
                       "Use existing fits as optional model_id references. This action advances to design.",
        "output": {"action": "hypothesize", "reason": "string", "hypotheses": [
            {"claim": "string", "scope": "string", "prediction": "string",
             "would_change": "string", "evidence_ids": "list of visible evidence IDs; may be empty",
             "assumptions": "optional list of concise strings", "contradiction_ids": "optional evidence IDs",
             "model_id": "optional existing frozen candidate/fit ID"}]},
    },
    "design": {
        "name": "design-experiment",
        "instruction": "Choose one legal instrument call that could change a conclusion or resolve material "
                       "uncertainty. Use the existing tool/args/why/discriminates/assumptions format and add design. "
                       "Link current hypothesis IDs, record the prediction before execution, and distinguish "
                       "support, contradiction and inconclusive/noisy outcomes. Account for the apparatus's "
                       "controls, replication unit and cost. No invented uncertainty or automatic value score. "
                       "The harness saves this plan before executing the ordinary tool action, then requests review.",
        "output": {"action": "tool", "design": {
            "purpose": "exploration|discrimination|precision|calibration",
            "hypothesis_ids": "list of current h... IDs; empty only for exploration",
            "prediction": "expected observation, quantitative if justified; uncertainty may be unresolved",
            "if_unexpected": "how an unexpected or inconclusive outcome changes the investigation",
            "model_id": "optional existing frozen candidate/fit ID"}},
    },
    "review": {
        "name": "review-hypothesis-and-experiment",
        "instruction": "Compare the saved design/prediction with its actual observations, errors and calculations. "
                       "Support is conditional, not proof. A missed point forecast need not reject every parameter "
                       "setting; assess measurement/analysis uncertainty and shared assumptions. Do not rewrite "
                       "a prior forecast after seeing data. Review every linked hypothesis and choose the next "
                       "phase. Read-only calculations/lookups are allowed first. To conclude in this same decision, "
                       "use ordinary finish with a method_review object containing these review fields.",
        "output": {"action": "review", "design_id": "current d... ID",
                   "result": "usable|uncertain|failed", "evidence_ids": "visible evidence supporting this review",
                   "assessments": [{"hypothesis_id": "linked h... ID",
                                    "status": "supported|challenged|unresolved", "reason": "string"}],
                   "reason": "overall interpretation and limitations", "next": "hypothesize|design|finish"},
    },
}


def initialize(run):
    run["scientific_method"] = {"version": VERSION, "phase": "hypothesize", "next_hypothesis": 1,
                                "hypotheses": [], "design": None, "last_review": None}
    run["method_artifacts"] = []


def design_skill(policy="direct"):
    if policy not in DESIGN_POLICIES:
        raise ValueError("Unknown design policy")
    skill = copy.deepcopy(SKILLS["design"])
    if policy == "compare":
        skill["instruction"] += (
            " Compare the chosen question with one serious feasible alternative before selecting the call. "
            "In design.alternative state that alternative's controls or action, expected outcomes and what "
            "they would change. Use why to explain why the chosen question is a better use of the remaining "
            "budget for the current uncertainty. Calibration, independent replication, analysis or stopping "
            "may be better than novelty. If no meaningful alternative is feasible, explain rather than invent "
            "one. Keep the comparison short. Only the chosen call executes; alternatives are not observations.")
        skill["output"]["design"]["alternative"] = (
            "one feasible alternative, its expected outcomes and their implications; or why none is meaningful")
    return skill


def context(run):
    state = run.get("scientific_method")
    if state is None:
        return None
    phase = state["phase"]
    skill = copy.deepcopy(SKILLS["review" if phase in ("conclude", "finished") else phase])
    if phase == "design":
        skill = design_skill(run.get("design_policy", "direct"))
    if phase == "conclude":
        skill["instruction"] = ("Return ordinary finish with a scoped result, limitations and optional pending "
                                "book proposals. Synthesis does not certify that every design was reviewed. "
                                "If a new investigation step is necessary instead, use method_route with a reason.")
        skill["output"] = {"action": "finish", "status": "concluded|inconclusive|blocked",
                           "summary": "scoped conclusion", "evidence_ids": "supporting evidence IDs",
                           "submission": "apparatus-specific object", "updates": "optional pending proposals"}
    return {
        "version": VERSION, "phase": phase, "active_skill": skill,
        "hypotheses": copy.deepcopy(state["hypotheses"]),
        "current_design": copy.deepcopy(state["design"]) if phase in ("review", "conclude") else None,
        "last_review": copy.deepcopy(state["last_review"]),
        "artifact_byte_limit": ARTIFACT_BYTES,
        "references": "h... and d... IDs are scoped to this run and its continuation; originals persist in event:<step>.",
        "branch": {"action": "method_route", "next": "hypothesize|design",
                   "reason": "why the default sequence is inappropriate; skipped reviews remain recorded"},
        "closing": phase == "conclude",
    }


def bounded(value):
    if len(encode(value).encode()) > ARTIFACT_BYTES:
        raise ValueError(f"Method artifact exceeds {ARTIFACT_BYTES} bytes; use concise claims and source references")


def citations(store, run, ids):
    from .context import visible_source

    strings(ids, "evidence_ids")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate method citations")
    for entry_id in ids:
        record = visible_source(store, run, entry_id)
        if record.get("book") != "evidence":
            raise ValueError("Method citations must refer to observed or derived evidence")


def frozen_model(store, run, entry_id):
    from .context import visible_source

    if entry_id is None:
        return
    record = visible_source(store, run, text(entry_id, "model_id"))
    if (record.get("kind") != "derived"
            or record.get("data", {}).get("operation") not in ("candidate", "fit_basis", "fit_model")):
        raise ValueError("model_id must identify an existing frozen candidate or fit")


def append(run, event, artifact):
    artifact = dict(copy.deepcopy(artifact), step=event["step"], run_id=run["id"])
    run["method_artifacts"].append(artifact)
    event.setdefault("method_artifacts", []).append(copy.deepcopy(artifact))
    return artifact


def validate_action(run, action):
    state = run["scientific_method"]
    kind, phase = action.get("action"), state["phase"]
    if kind == "tool" and phase != "design":
        raise ValueError(f"Scientific phase is {phase}; acquire only after design, or use method_route with a reason")
    if kind == "hypothesize" and phase != "hypothesize":
        raise ValueError(f"Scientific phase is {phase}; review or explicitly route before replacing hypotheses")
    if kind == "review" and phase != "review":
        raise ValueError(f"Scientific phase is {phase}; no pending evidence review at this phase")
    if kind == "tool" and run.get("phase") == "analysis_only":
        raise ValueError("Analysis-only continuation cannot execute a design")


def hypothesize(store, run, event, action):
    hypotheses = action.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > 3:
        raise ValueError("hypotheses must be an array with zero to three concise entries")
    reason = text(action.get("reason"), "hypothesis reason")
    bounded({"hypotheses": hypotheses, "reason": reason})
    allowed = {"claim", "scope", "prediction", "would_change", "evidence_ids", "model_id",
               "assumptions", "contradiction_ids"}
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or set(hypothesis) - allowed:
            raise ValueError("Use the declared hypothesis fields; assumptions, contradiction_ids and model_id are optional")
        for field in ("claim", "scope", "prediction", "would_change"):
            text(hypothesis.get(field), field)
        citations(store, run, hypothesis.get("evidence_ids", []))
        citations(store, run, hypothesis.get("contradiction_ids", []))
        strings(hypothesis.get("assumptions", []), "hypothesis assumptions")
        frozen_model(store, run, hypothesis.get("model_id"))
    state = run["scientific_method"]
    created = [dict(copy.deepcopy(hypothesis), id=f"h{state['next_hypothesis']+index}",
                    created_step=event["step"]) for index, hypothesis in enumerate(hypotheses)]
    append(run, event, {"kind": "hypotheses", "hypotheses": created, "reason": reason})
    state["next_hypothesis"] += len(created)
    state.update(hypotheses=created, phase="conclude" if run.get("phase") == "analysis_only" else "design")


def design(store, run, event, action):
    plan = action.get("design")
    allowed = {"purpose", "hypothesis_ids", "prediction", "if_unexpected", "model_id"}
    compare = run.get("design_policy", "direct") == "compare"
    if compare:
        allowed.add("alternative")
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError("A tool action in design phase requires the declared design object")
    if compare:
        text(plan.get("alternative"), "design.alternative")
    if plan.get("purpose") not in ("exploration", "discrimination", "precision", "calibration"):
        raise ValueError("Invalid design purpose")
    ids = strings(plan.get("hypothesis_ids"), "hypothesis_ids")
    active = {hypothesis["id"] for hypothesis in run["scientific_method"]["hypotheses"]}
    if len(ids) != len(set(ids)) or not set(ids) <= active:
        raise ValueError("Design must cite distinct current hypothesis IDs")
    if not ids and plan["purpose"] != "exploration":
        raise ValueError("Only an exploratory design may omit hypothesis links")
    for field in ("prediction", "if_unexpected"):
        text(plan.get(field), field)
    frozen_model(store, run, plan.get("model_id"))
    artifact = dict(copy.deepcopy(plan), kind="design", id=f"d{event['step']}",
                    tool=action.get("tool"), args=copy.deepcopy(action.get("args")),
                    acquisitions=action.get("acquisitions", 1))
    bounded(artifact)
    artifact = append(run, event, artifact)
    run["scientific_method"]["design"] = copy.deepcopy(artifact)


def record_execution(run, event):
    state = run.get("scientific_method")
    if state is None or not any(a["kind"] == "design" for a in event.get("method_artifacts", [])):
        return
    observations = [item["evidence_id"] for item in completed_acquisitions([event])]
    execution = {"kind": "execution", "design_id": state["design"]["id"],
                 "status": event.get("execution", "not_started"),
                 "preflight_rejected": event.get("preflight_rejected", False),
                 "evidence_ids": observations}
    if event.get("error"):
        execution["error"] = event["error"]
    append(run, event, execution)
    state["design"]["execution"] = copy.deepcopy(execution)
    state["phase"] = "design" if execution["preflight_rejected"] or execution["status"] == "not_started" else "review"


def review_record(store, run, review):
    if not isinstance(review, dict):
        raise ValueError("Review must be an object")
    bounded(review)
    plan = run["scientific_method"]["design"]
    if plan is None or review.get("design_id") != plan["id"] or "execution" not in plan:
        raise ValueError("Review must identify the current executed design")
    if plan["execution"]["preflight_rejected"]:
        raise ValueError("A rejected preflight is not an executed experiment")
    if review.get("result") not in ("usable", "uncertain", "failed"):
        raise ValueError("Review result must be usable, uncertain or failed")
    evidence = review.get("evidence_ids", [])
    citations(store, run, evidence)
    observations = set(plan["execution"]["evidence_ids"])
    covered = set(evidence)
    for entry_id in evidence:
        covered.update(store.get(entry_id).get("evidence_ids", []))
    if observations and not observations & covered:
        raise ValueError("Review must cite this design's observations or a calculation using them")
    if not observations and review["result"] != "failed":
        raise ValueError("A design with no acquired observation cannot have a usable or uncertain result")
    assessments = review.get("assessments")
    if not isinstance(assessments, list) or any(not isinstance(item, dict) for item in assessments):
        raise ValueError("Review assessments must be an array of hypothesis assessments")
    ids = [item.get("hypothesis_id") for item in assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["hypothesis_ids"]):
        raise ValueError("Review must assess each linked hypothesis exactly once")
    for item in assessments:
        if item.get("status") not in ("supported", "challenged", "unresolved"):
            raise ValueError("Hypotheses may be supported, challenged or unresolved, not proved")
        text(item.get("reason"), "assessment reason")
    text(review.get("reason"), "review reason")
    if review.get("next") not in ("hypothesize", "design", "finish"):
        raise ValueError("Review next must be hypothesize, design or finish")
    return dict(copy.deepcopy(review), kind="review")


def commit_review(run, event, review):
    append(run, event, review)
    run["scientific_method"].update(last_review=copy.deepcopy(review),
                                   phase="conclude" if review["next"] == "finish" else review["next"])


def route(run, event, action):
    target = action.get("next")
    if target not in ("hypothesize", "design"):
        raise ValueError("method_route next must be hypothesize or design")
    if run.get("phase") == "analysis_only" and target == "design":
        raise ValueError("Analysis-only continuation cannot route to new experiments")
    reason = text(action.get("reason"), "route reason")
    bounded({"reason": reason})
    state = run["scientific_method"]
    append(run, event, {"kind": "route", "from": state["phase"], "next": target, "reason": reason,
                        "unreviewed_design": state["design"]["id"] if state["phase"] == "review" else None})
    state["phase"] = target


def finish(run, event, action, review=None):
    state = run["scientific_method"]
    prior_phase = state["phase"]
    if review is not None:
        commit_review(run, event, review)
    executed = {item["design_id"] for item in run["method_artifacts"] if item["kind"] == "execution"
                and not item["preflight_rejected"] and item["status"] != "not_started"}
    reviewed = {item["design_id"] for item in run["method_artifacts"] if item["kind"] == "review"}
    append(run, event, {"kind": "finish", "from": prior_phase,
                        "review_included": review is not None,
                        "unfinished_review": bool(executed - reviewed),
                        "reason": action["summary"]})
    state["phase"] = "finished"


def resume(run):
    state = run.get("scientific_method")
    if state is not None:
        execution = (state.get("design") or {}).get("execution")
        state["phase"] = "review" if execution and not execution["preflight_rejected"] else "hypothesize"
