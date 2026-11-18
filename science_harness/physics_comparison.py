"""DiscoveryAgent plus/minus a notebook, with selector-only random controls."""
import copy
import json
import math
from pathlib import Path
import re
import threading
from unittest.mock import patch

import numpy as np

from .physics_reference import digest, load_upstream, provenance
from .physics_context import ContextLimit
from .physics_notebook import (
    CONTRACT as NOTEBOOK_MEMORY_CONTRACT,
    PROMPT as NOTEBOOK_MEMORY_PROMPT,
    ExternalNotebook,
    parse_lookup,
)
from .physics_runtime import Journal, NOISE_STD, TIMES, encode, noise_seed, proposal, validate_design

MODELS = ("gpt-6.1-sol", "claude-haiku-5.5")
CONTROLLERS = ("discovery", "science")
SELECTORS = ("active", "random")
_NATIVE_LOCK = threading.Lock()


class RequestLimit(RuntimeError):
    pass


class NoteError(ValueError):
    pass


class ExecutionFailure(RuntimeError):
    pass


COMMON_CONTRACT = f"""
COMMON BENCHMARK WRAPPER (overrides conflicting task instructions):
- At most ONE experiment per response. No batches. Times exactly [1,2,3,4,5],
  duration 5, stationary source at [0,0], optional start_time not available.
- p1,p2 in [0.1,10]; pos2 coordinates in [-10,10]; velocity2 in [-5,5].
- Two startup experiments are already supplied and count toward 50 observations.
  At most 100 total model requests, INCLUDING fit-only turns and auxiliary replies.
- Optional run_mse_fit uses original position-MSE fitting, NOT velocity fitting.
  When paired with an experiment it fits AFTER that experiment. Fit-only turns
  acquire no measurements. A law is NOT required every turn.
- Native noise affects both particles' positions and velocities. Position sigma
  is {NOISE_STD}; velocity sigma is position sigma divided by measurement cadence
  (numerically equal here). Initial launch settings are known. Source readings
  are noisy measurements of the stationary source, not a moving object to fit.
- No supervisor or explanation judge is used. Retain the native explanation
  requirement. No hidden scores or test-validity flags will be supplied.
- Submitted Python runs in an isolated worker with no project-file or network
  access. NumPy, math and SciPy are available. The law owns its trajectory
  prediction; no special trusted integrator is supplied only to one controller.
- Each output includes the actual executed initial conditions. Those, rather
  than a proposed but unexecuted configuration, identify the measured evidence.
"""

NOTE_CONTRACT = """
SCIENTIFIC NOTEBOOK ADDITION (overrides the earlier final-response formatting rule):
EVERY scientific response MUST include exactly one <notebook_entry>JSON</notebook_entry>,
INCLUDING a response that submits <final_law>. A final submission MUST contain
final_law, explanation, AND notebook_entry; the earlier two-tag-only rule does
not apply to this notebook-augmented controller.
JSON has exactly insight, hypothesis, next_test. insight has summary, confidence
(low/medium/high), and evidence_ids (distinct existing experiment-NNNN IDs).
hypothesis and summary are nonempty strings, at most 2000 characters each.
next_test is a nonempty string at most 2000 characters, or null on final submission.
Cite only observations already available BEFORE this response. Do not invent
evidence or execution results. No law is required for an experiment-only turn.
The notebook preserves original sources and observed fits; they are not hidden
accuracy, and fits on different evidence prefixes are not directly comparable.
Only explanation-only follow-ups and note-only repairs are exempt from the
three-part scientific response. Do not add a notebook tag to those auxiliary replies.
"""

FINAL_VALIDATION_CONTRACT = """
SHARED OBSERVED-DATA FINAL VALIDATION:
- pos1, pos2 and velocity2 are Python lists of length two. Convert with np.asarray
  before vector arithmetic. p1, p2 and duration are scalars; duration is one time,
  not the measurement-time array. Return final position and velocity, two
  length-two vectors, not coordinate histories.
- run_mse_fit and final_law contain discovered_law predictor source, not a fitting
  script. Optional fit_parameters declares at most five bounded scalar values,
  never vectors or names that collide with the six known law inputs.
- Final proposals are checked with the original observed-only final fit before
  acceptance. An unevaluable program can be corrected in another charged response
  within the SAME total request cap. High observed error alone does not reject
  an executable program. Final repair does not acquire further measurements.
"""

ACTION_RECOVERY_CONTRACT = """
OPERATIONAL ACTION RECOVERY (single-design-and-domain-v1):
- A single schema-valid experiment object is accepted as its one-element array.
  Raw replies, law source, notebook content and actual controls remain unchanged.
- Probe and source initial positions must differ; exact coincidence is outside
  the supported native launch domain. No position or force is silently changed.
- A rejected experiment acquires no data and performs no optional fit. One
  charged scientist correction is allowed before accepting the draft, within
  the SAME total request budget. Use existing evidence; no new notebook reads
  during correction. A second invalid action stops explicitly.
- Review still has at most one commitment. An invalid commitment stops before
  acquisition; operational correction does not introduce another review round.
"""

RANDOM_SELECTION_CONTRACT = """
RANDOM EXPERIMENT-SELECTION CONTROL (overrides instructions to select the executed launch):
- You remain the scientist: analyze evidence, maintain the same notebook, and
  propose/refine/finalize laws normally. Hypotheses and law code are NOT random.
- Each run_experiment requests one acquisition, but the harness independently
  draws the actual controls from its registered random stream. Your proposed
  controls and intent are retained as unexecuted proposals, not measured evidence.
- Ground subsequent analysis in executed_initial_conditions and the returned
  evidence IDs. Do not describe a proposed transformation as actually executed.
- Notebook reads and fit-only turns acquire no experiments and do not advance
  the random experiment stream. Fitting and final validation are unchanged.
"""


def parse_note(text, observed, final=False):
    try:
        note = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid notebook JSON; scientific response retained") from exc
    if not isinstance(note, dict) or set(note) != {"insight", "hypothesis", "next_test"}:
        raise ValueError("Unexpected notebook fields")
    insight = note["insight"]
    if not isinstance(insight, dict) or set(insight) != {"summary", "confidence", "evidence_ids"}:
        raise ValueError("Unexpected insight fields")
    ids = insight["evidence_ids"]
    allowed = {f"experiment-{index:04d}" for index in range(1, observed + 1)}
    if (insight["confidence"] not in ("low", "medium", "high") or not isinstance(ids, list)
            or not ids or any(not isinstance(i, str) or i not in allowed for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid confidence or evidence citations; no repair allowed")
    for key, value in (("summary", insight["summary"]), ("hypothesis", note["hypothesis"]),
                       ("next_test", note["next_test"])):
        if final and key == "next_test" and value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise NoteError("Notebook text must contain 1..2000 characters")
    if final and note["next_test"] is not None:
        raise NoteError("Final next_test must be null")
    return note


class Experiments:
    def __init__(self, world, seed, selector, journal, observations_limit=50,
                 validate_initial_conditions=None):
        if selector not in SELECTORS or not 2 <= observations_limit <= 50:
            raise ValueError("Unregistered selector or observation budget")
        self.world, self.seed, self.selector, self.journal = world, seed, selector, journal
        self.limit, self.observations, self.events = observations_limit, [], []
        self.validate_initial_conditions = (validate_initial_conditions
                                            or getattr(world, "validate_initial_conditions", None))
        self.fatal = None
        for index in range(2):
            self._acquire(proposal(seed + 10000, index), startup=True)

    def _acquire(self, design, startup=False, intent=None):
        index = len(self.observations) + 1
        seed = noise_seed(self.seed, index)
        try:
            raw = self.journal.call(
                "measurement", {"design": design, "noise_seed": seed},
                lambda: self.world.measure(copy.deepcopy(design), seed))
        except Exception as exc:
            self.fatal = ExecutionFailure(f"Accepted launch failed: {type(exc).__name__}: {exc}")
            raise self.fatal from exc
        row = {"id": f"experiment-{index:04d}", "input": copy.deepcopy(design), "output": raw}
        self.observations.append(row)
        if not startup:
            self.events.append({"observation_id": row["id"], "intent": copy.deepcopy(intent),
                                "executed": design, "noise_seed": seed})
        return {**copy.deepcopy(raw), "evidence_id": row["id"],
                "executed_initial_conditions": copy.deepcopy(design)}

    def run(self, experiments):
        if not isinstance(experiments, list) or len(experiments) != 1:
            raise ValueError("Exactly one experiment per request; batch not executed")
        if len(self.observations) >= self.limit:
            raise ValueError("Observation budget exhausted; only fitting/final submission remain")
        intent = experiments[0]
        encode(intent)
        # Only the selected, executable launch receives numerical/range validation.
        chosen = intent if self.selector == "active" else proposal(self.seed + 50000, len(self.observations))
        chosen = validate_design(chosen)
        if self.validate_initial_conditions is not None:
            self.validate_initial_conditions(chosen)
        return [self._acquire(chosen, intent=intent)]


def run_case(checkout, world, numerics, provider, directory, *, model=MODELS[0],
             controller="discovery", selector="active", seed=990001, request_limit=100,
             observations_limit=50, validate_final=False, advisor=None, trace_directory=None,
             notebook_memory="conversation", design_advisor=None,
             total_request_limit=None, require_experiment_intent=False, action_recovery=False):
    if (model not in MODELS or controller not in CONTROLLERS or type(request_limit) is not int
            or not 1 <= request_limit <= 500):
        raise ValueError("Unregistered model/controller/request budget")
    if notebook_memory not in ("conversation", "external"):
        raise ValueError("Unknown notebook memory policy")
    if notebook_memory == "external" and (controller != "science" or advisor is not None):
        raise ValueError("External notebook requires ScienceAgent without forecast advice")
    if total_request_limit is not None and (
            type(total_request_limit) is not int or not 1 <= total_request_limit <= 500):
        raise ValueError("Combined request limit must be an integer in 1..500")
    if type(require_experiment_intent) is not bool:
        raise ValueError("require_experiment_intent must be boolean")
    if type(action_recovery) is not bool:
        raise ValueError("action_recovery must be boolean")
    native = load_upstream(checkout)
    journal = Journal(Path(directory) / "operations",
                      Path(trace_directory) / "operations" if trace_directory is not None else None)
    manifest = {**provenance(checkout), "model": model, "controller": controller,
                "selector": selector, "seed": seed, "request_limit": request_limit,
                "observations_limit": observations_limit}
    if validate_final:
        manifest["final_validation"] = "observed-only-charged-repair-v1"
    if advisor is not None:
        from .physics_planning import CONTRACT
        if controller != "science" or selector != "active":
            raise ValueError("Forecast advice is registered only for Science/active")
        manifest["advisory_forecasts"] = CONTRACT
    if notebook_memory == "external":
        manifest["notebook_memory"] = NOTEBOOK_MEMORY_CONTRACT
    if design_advisor is not None:
        if controller != "science" or selector != "active":
            raise ValueError("Design review is supported only for Science/active")
        if design_advisor.requests or design_advisor.records:
            raise ValueError("Each investigation requires a fresh design advisor")
        manifest["design_advisor"] = design_advisor.manifest
        manifest["total_request_limit"] = request_limit + design_advisor.request_limit
    if total_request_limit is not None:
        manifest["total_request_limit"] = total_request_limit
        manifest["request_budget_policy"] = "shared-scientist-lookup-review-revision-repair-v1"
    if require_experiment_intent:
        manifest["experiment_intent"] = "shared-reference-transformation-discriminator-v1"
    if require_experiment_intent or design_advisor is not None or action_recovery:
        manifest["public_response_validation"] = "exact-complete-proposal-phase-copies-v1"
    if action_recovery:
        manifest["action_recovery"] = {
            "contract": "single-design-and-domain-v1", "max_corrections_per_draft": 1,
            "extra_reads": 0, "coincident_launch": "reject-before-acquisition",
            "raw_history_preserved": True,
        }
        if selector == "random":
            manifest["experiment_selection"] = {
                "contract": "selector-only-random-v1",
                "proposed_controls_executed": False, "actual_controls_returned": True,
            }
    journal.call("manifest", manifest, lambda: manifest)
    memory = ExternalNotebook(
        Path(directory) / "external-notebook",
        Path(trace_directory) / "external-notebook" if trace_directory is not None else None,
    ) if notebook_memory == "external" else None
    def validate_launch(design):
        if action_recovery and design["pos2"] == [0, 0]:
            raise ValueError("Probe and source initial positions cannot coincide; native launch is unsupported")
        if hasattr(world, "validate_initial_conditions"):
            world.validate_initial_conditions(design)

    experiments = Experiments(world, seed, selector, journal, observations_limit,
                              validate_launch if action_recovery else None)
    proposed_launch_validator = experiments.validate_initial_conditions if selector == "active" else None
    candidates, notes, sources, responses, emissions, rejected_finals = [], [], {}, [], [], []
    requests, scientist_requests = 0, 0
    request_roles, forecast_advice = [], []
    session_key = str(Path(directory).resolve())
    delivered = {"notes": 0, "candidates": 0, "executions": 0, "sources": set(),
                 "design_reviews": 0}
    notebook_lookups = []
    external_context = None

    def sync_notebook():
        return memory.sync(observations=experiments.observations, notes=notes,
                           candidates=candidates, sources=sources, executions=experiments.events,
                           conversation=agent.conversation_log, forecast_advice=forecast_advice,
                           reviews=design_advisor.records if design_advisor is not None else [])

    def scientist_remaining():
        remaining = request_limit - scientist_requests
        if total_request_limit is not None:
            remaining = min(remaining, total_request_limit - requests)
        return remaining

    def fit(source, final=False):
        if experiments.fatal:
            raise experiments.fatal
        source_hash = digest(source)
        sources[source_hash] = source
        payload = {"operation": "final_fit" if final else "fit", "source": source,
                   "observations": copy.deepcopy(experiments.observations)}
        result = journal.call("numerics", payload, lambda: numerics.request(**payload))
        candidates.append({"source_sha256": source_hash,
                           "observed_count": len(experiments.observations),
                           "request_count": requests, "fit": result, "final": final})
        return result

    def cached_final(source):
        return (candidates and candidates[-1]["final"]
                and candidates[-1]["source_sha256"] == digest(source)
                and candidates[-1]["observed_count"] == len(experiments.observations))

    def request(system, messages, *, suffix="", auxiliary=False, assistance_role=None,
                assistance_provider=None, assistance_key=None, output_limit=32768):
        nonlocal requests, scientist_requests
        if experiments.fatal:
            raise experiments.fatal
        if assistance_role is None and scientist_requests >= request_limit:
            raise RequestLimit("Total model-request budget exhausted")
        if total_request_limit is not None and requests >= total_request_limit:
            raise RequestLimit("Combined model-request budget exhausted")
        if memory is not None and assistance_role == "revision":
            if external_context is None:
                raise RuntimeError("External scientist commitment lost its selected evidence context")
            context = copy.deepcopy(external_context)
            context["requests_remaining"] = scientist_remaining()
            context["commitment"] = {
                "draft": messages[-2]["content"], "instruction": messages[-1]["content"],
            }
            messages = [{"role": "user", "content": "<external_notebook_context>" +
                         encode(context) + "</external_notebook_context>"}]
        requests += 1
        if assistance_role is None:
            scientist_requests += 1
        request_roles.append(assistance_role or ("auxiliary" if auxiliary else "scientist"))
        key = assistance_key or (
            session_key if not auxiliary else f"{session_key}:aux:{requests}")
        if memory is not None and assistance_role != "review" and not auxiliary:
            key = f"{session_key}:notebook:{requests}"
        packet = {"model": model, "system": system, "messages": copy.deepcopy(messages),
                  "max_tokens": output_limit, "context_suffix": suffix,
                  "session_key": key}
        def dispatch():
            if assistance_role is not None:
                from .physics_context import admit_context
                admit_context(packet)
            return (assistance_provider if assistance_provider is not None else provider)(**packet)
        reply = journal.call("model", packet, dispatch)
        if not isinstance(reply, dict) or not isinstance(reply.get("content"), str):
            raise RuntimeError("Transport must return content and usage evidence")
        responses.append(copy.deepcopy(reply))
        return reply["content"]

    def external_response(system, messages):
        nonlocal external_context
        feedback = []
        for message in reversed(messages):
            if message["role"] == "assistant":
                break
            if message["role"] == "user":
                feedback.append(message["content"])
        packet = {"notebook": sync_notebook(), "latest_feedback": list(reversed(feedback)),
                  "observations_remaining": observations_limit - len(experiments.observations),
                  "reads": []}
        while True:
            packet["requests_remaining"] = scientist_remaining()
            packet["reads_remaining"] = max(
                0, NOTEBOOK_MEMORY_CONTRACT["lookups_per_decision"] - len(packet["reads"]))
            reply = request(system, [{"role": "user", "content": "<external_notebook_context>" +
                                     encode(packet) + "</external_notebook_context>"}])
            error = None
            try:
                action = parse_lookup(reply)
            except ValueError as exc:
                action = {"tool": "invalid", "reply": reply}
                error = str(exc)
            if action is None:
                external_context = copy.deepcopy(packet)
                return reply
            request_roles[-1] = "lookup"
            exhausted = packet["reads_remaining"] == 0

            def lookup():
                if exhausted:
                    return {
                        "ok": False, "code": "lookup_budget_exhausted", "reads_remaining": 0,
                        "error": "External notebook lookup budget exhausted; no read or scientific action executed",
                        "next_response": "Choose a scientific action using the supplied evidence; no more reads this decision",
                    }
                if error is not None:
                    return {"ok": False, "error": error}
                try:
                    return memory.lookup(action["tool"], action["arguments"])
                except ValueError as exc:
                    return {"ok": False, "error": str(exc)}

            payload = {**action, "revision": memory.info["revision"]}
            result = journal.call("notebook", payload, lookup)
            record = {"request": action, "result": result}
            notebook_lookups.append(record)
            if exhausted:
                packet["read_budget_feedback"] = result
            else:
                packet["reads"].append({"tool": action["tool"], "result": result})

    proposal_views, action_payloads = {}, {}
    native_extract = native._extract_tag

    def recovery_packet(text):
        from .physics_design_advisor import _tag, canonical_experiment, parse_proposal
        checked = canonical_experiment(
            text, proposed_launch_validator)
        if require_experiment_intent or design_advisor is not None:
            parse_proposal(checked, experiments.observations, parse_note,
                           proposed_launch_validator)
        elif controller == "science":
            parse_note(_tag(checked, "notebook_entry", required=True), len(experiments.observations))

    def validation_view(text):
        from .physics_design_advisor import canonical_experiment, proposal_validation_view
        index = next((i for i in reversed(range(len(responses)))
                      if responses[i]["content"] == text), None)
        if index is None:
            if action_recovery:
                raise RuntimeError("Action validation lost its exact model response")
            return text
        if index not in proposal_views:
            normalized = proposal_validation_view(
                text, responses[index].get("visible_messages"), experiments.observations,
                parse_note, proposed_launch_validator,
                validate_packet=recovery_packet if action_recovery else None)
            if normalized != text:
                journal.call("public_response_normalization", {
                    "contract": manifest["public_response_validation"],
                    "response_number": index + 1, "raw_response_sha256": digest(text),
                    "public_message_ids": [row["messageId"]
                                           for row in responses[index]["visible_messages"]],
                }, lambda: {"validation_view": normalized, "sha256": digest(normalized),
                            "raw_response_and_native_history_unchanged": True})
            if action_recovery and "<run_experiment>" in normalized:
                canonical = canonical_experiment(
                    normalized, proposed_launch_validator)
                action_payloads[text] = native_extract(canonical, "run_experiment")
                if canonical != normalized:
                    journal.call("action_normalization", {
                        "contract": manifest["action_recovery"]["contract"],
                        "response_number": index + 1, "raw_response_sha256": digest(text),
                    }, lambda: {"canonical_experiment": json.loads(action_payloads[text]),
                                "raw_response_and_native_history_unchanged": True,
                                "code_and_notebook_unchanged": True})
                normalized = canonical
            proposal_views[index] = normalized
        return proposal_views[index]

    def execution_tag(text, tag):
        if action_recovery and tag == "run_experiment" and text in action_payloads:
            return action_payloads[text]
        return native_extract(text, tag)

    class Controller(native.DiscoveryAgent):
        def _run_mse_fit(self, round_num, law_source, round_entry):
            round_entry["mse_fit_input"] = law_source.strip()
            result = fit(law_source.strip())
            round_entry["mse_fit_output"] = result
            return "<mse_fit_output>\n" + encode(result) + "\n</mse_fit_output>"

        def _log_trajectories(self, *args, **kwargs):
            # The immutable operation journal replaces CSV; fitting uses actual launches.
            return None

    initial = encode({"startup_observations": experiments.observations})
    agent = Controller(
        model=model, executor=experiments, mission=initial, verbose=False,
        max_tokens=32768, max_rounds=request_limit, min_rounds=2,
        system_prompt_path=str(Path(checkout).resolve() / "PhysicsSchool/prompts/_template_interactive.md"),
        instructions_path=str(Path(checkout).resolve() / "PhysicsSchool/prompts/2particle_instructions.md"),
        trajectory_logger=object(), critic=None, random_experiments=False)
    agent._system += COMMON_CONTRACT.replace(
        "100 total model requests", f"{request_limit} total model requests").replace(
        "50 observations", f"{observations_limit} observations")
    if controller == "science":
        agent._system += NOTE_CONTRACT
    if validate_final:
        agent._system += FINAL_VALIDATION_CONTRACT
    if action_recovery:
        agent._system += ACTION_RECOVERY_CONTRACT
        if selector == "random":
            agent._system += RANDOM_SELECTION_CONTRACT
    if getattr(world, "forbid_coincident_launch", False):
        agent._system += "\nProbe and source cannot have exactly coincident initial positions; that native input is undefined.\n"
    if advisor is not None:
        from .physics_planning import PROMPT
        agent._system += PROMPT
    if memory is not None:
        agent._system += NOTEBOOK_MEMORY_PROMPT
    if require_experiment_intent:
        from .physics_design_advisor import INTENT_PROMPT
        agent._system += INTENT_PROMPT
    if design_advisor is not None:
        from .physics_design_advisor import PROMPT
        agent._system = agent._system.replace(
            f"At most {request_limit} total model requests, INCLUDING fit-only turns and auxiliary replies.",
            f"At most {request_limit} scientist requests, INCLUDING fit-only turns and auxiliary replies; "
            f"plus at most {design_advisor.request_limit} separately metered design-review/revision requests.")
        agent._system += PROMPT
    if total_request_limit is not None:
        agent._system += (
            f"\nSHARED INFERENCE BUDGET: at most {total_request_limit} TOTAL model requests, "
            "including scientific decisions, notebook lookups, advisor reviews, scientist "
            "commitments and note/code repairs. Role allowances are ceilings, not extra "
            "requests beyond this combined cap.\n")

    def complete(*, model, messages, system, max_tokens, final_repair=False):
        is_explanation = bool(not final_repair and messages and "ONLY a single" in messages[-1]["content"]
                              and "<explanation>" in messages[-1]["content"])
        if controller == "science" and not is_explanation and memory is None:
            update = {
                "version": "native-actions-v1", "entries": notes[delivered["notes"]:],
                "sources": {key: value for key, value in sources.items() if key not in delivered["sources"]},
                "observed_fits": candidates[delivered["candidates"]:],
                "executions": [
                    {key: event[key] for key in ("observation_id", "intent", "executed")}
                    for event in experiments.events[delivered["executions"]:]
                ],
            }
            # Keep each update in the actual native transcript once, rather than
            # silently accumulating repeated full-book snapshots inside the SDK.
            messages.append({"role": "user", "content": "<scientific_notebook_update>" +
                             encode(update) + "</scientific_notebook_update>"})
            delivered.update(notes=len(notes), candidates=len(candidates),
                             executions=len(experiments.events), sources=set(sources))
        if advisor is not None and not is_explanation and not final_repair:
            advice = advisor.advise(experiments.observations, candidates, sources)
            if advice is not None:
                forecast_advice.append(copy.deepcopy(advice))
                messages.append({"role": "user", "content": "<advisory_forecasts>" +
                                 encode(advice) + "</advisory_forecasts>"})
        if design_advisor is not None:
            updates = [{k: row[k] for k in ("review_id", "status", "observed_count")}
                       for row in design_advisor.records[delivered["design_reviews"]:]]
            if updates:
                messages.append({"role": "user", "content": "<design_review_status>" +
                                 encode(updates) + "</design_review_status>"})
                delivered["design_reviews"] = len(design_advisor.records)
        if final_repair:
            remaining = scientist_remaining()
            remaining_message = (
                f"There are {remaining} charged scientist requests remaining, including this one."
                if design_advisor is not None else
                f"There are {remaining} charged requests remaining, including this one.")
            messages[-1]["content"] += (
                "\n\nTHIS TURN IS FINAL-PROGRAM CODE REPAIR, not another experiment. "
                "Correct the observed error reported above. Return a complete "
                "<final_law> program, <explanation>, and (for Science) <notebook_entry>. "
                "Its next_test must be literal JSON null. Do not request experiments "
                "or run_mse_fit here; the observed final fit will run automatically. "
                + remaining_message
            )
        before = len(experiments.observations)
        reply = external_response(system, messages) if memory is not None else request(system, messages)
        if action_recovery and not is_explanation and not final_repair:
            from .physics_design_advisor import ActionValidationError
            for attempt in range(2):
                try:
                    if attempt and ("<notebook_open>" in reply or "<notebook_grep>" in reply):
                        raise ActionValidationError("No notebook reads are allowed during action correction")
                    if "<run_experiment>" in reply:
                        validation_view(reply)
                    break
                except ActionValidationError as exc:
                    feedback = {
                        "ok": False, "error": str(exc), "observed_count": before,
                        "requests_remaining": scientist_remaining(),
                        "corrections_remaining": 1 - attempt,
                        "measurement_executed": False, "fit_executed": False,
                        "instruction": "Correct the rejected action or choose another permitted scientific "
                        "action. Return the complete scientific packet. Use only evidence already supplied; "
                        "no additional notebook reads. No measurement or fit ran for this proposal.",
                    }
                    journal.call("action_validation", {
                        "raw_response_sha256": digest(reply), "attempt": attempt,
                    }, lambda: feedback)
                    if attempt or scientist_remaining() <= 0:
                        raise
                    messages.extend([
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": "<action_validation_output>" +
                         encode(feedback) + "</action_validation_output>"},
                    ])
                    repair_messages = messages
                    if memory is not None:
                        context = copy.deepcopy(external_context)
                        context.update(
                            reads_remaining=0, requests_remaining=scientist_remaining(),
                            action_validation={"rejected_response": reply, "feedback": feedback})
                        repair_messages = [{"role": "user", "content": "<external_notebook_context>" +
                                            encode(context) + "</external_notebook_context>"}]
                    reply = request(system, repair_messages)
                    request_roles[-1] = "action_repair"
        if (require_experiment_intent and not is_explanation and not final_repair
                and "<run_experiment>" in reply):
            from .physics_design_advisor import parse_experiment
            parse_experiment(validation_view(reply), experiments.observations,
                             proposed_launch_validator)
        if (design_advisor is not None and not is_explanation and not final_repair
                and "<run_experiment>" in reply):
            reply = design_advisor.review(
                reply, system=system, messages=messages, session_key=session_key,
                journal=journal, request=request, parse_note=parse_note,
                observations=experiments.observations, notes=notes, candidates=candidates,
                sources=sources, executions=experiments.events, conversation=agent.conversation_log,
                forecast_advice=forecast_advice,
                available_requests=(total_request_limit - requests
                                    if total_request_limit is not None else None),
                validate_initial_conditions=experiments.validate_initial_conditions,
                validation_view=validation_view,
                validation_description=(
                    "Raw output is retained. The validation view applies only the journaled "
                    "complete-phase-copy and single-design-array envelope normalization; "
                    "Python source, notebook content and scientific controls are unchanged."
                    if action_recovery else None))
        if not is_explanation:
            for action in ("run_mse_fit", "final_law"):
                emitted = native._extract_tag(reply, action)
                if emitted is not None:
                    emitted = emitted.strip()
                    sources[digest(emitted)] = emitted
                    emissions.append({"action": action, "source_sha256": digest(emitted),
                                      "observed_count": before, "request_count": requests})
        if final_repair and native._extract_tag(reply, "final_law") is None:
            if scientist_remaining() <= 0:
                raise ValueError("Final repair did not submit final_law; no experiment was executed")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "The repair request requires final_law. "
                             "The proposed action was not executed. Fix the previously rejected program."})
            return complete(model=model, messages=messages, system=system,
                            max_tokens=max_tokens, final_repair=True)
        if controller == "science" and not is_explanation:
            raw_note = native._extract_tag(reply, "notebook_entry")
            final = native._extract_tag(reply, "final_law") is not None
            try:
                original = json.loads(raw_note) if raw_note is not None else None
            except (ValueError, TypeError):
                if not validate_final:
                    raise
                original = None
            preserved = original.get("insight", {}) if isinstance(original, dict) else {}
            if not isinstance(preserved, dict):
                preserved = {}
            for attempt in range(3):
                try:
                    note = parse_note(raw_note, before, final)
                    if any(note["insight"][k] != preserved[k]
                           for k in ("confidence", "evidence_ids") if k in preserved):
                        raise ValueError("Note repair changed confidence or evidence")
                    break
                except ValueError as error:
                    if attempt == 2 or (not validate_final and not isinstance(error, NoteError)):
                        raise
                    repair = {"original_note": original, "final": final,
                              "format_error": str(error),
                              "next_test_requirement": "JSON null, not the string \"null\"" if final
                              else "A nonempty string of at most 2000 characters",
                              "instruction": "Repair only the reported note-format violation. Return "
                              "JSON with exactly insight, hypothesis, next_test. "
                              "Keep claims, confidence and exact citations. Code/actions are frozen."}
                    if validate_final:
                        repair.update(
                            original_scientific_response=reply,
                            available_observation_ids=[o["id"] for o in experiments.observations],
                            note_schema={
                                "insight": {"summary": "Nonempty text, at most2000characters",
                                            "confidence": ["low", "medium", "high"],
                                            "evidence_ids": "Distinct already-observed experiment IDs"},
                                "hypothesis": "Nonempty text, at most2000characters",
                                "next_test": "JSON null" if final else "Nonempty text, at most2000characters"},
                            missing_or_unparseable_note=original is None)
                    raw_note = request(
                        "Return only the repaired scientific note; no law or experiment.",
                        [{"role": "user", "content": encode(repair)}], auxiliary=True)
                    raw_note = re.sub(r"^```(?:json)?\s*\n(.*)\n```$", r"\1",
                                      raw_note.strip(), flags=re.DOTALL)
                    if validate_final:
                        wrapped = re.fullmatch(r"<notebook_entry>\s*(.*?)\s*</notebook_entry>",
                                               raw_note, flags=re.DOTALL)
                        if wrapped:
                            raw_note = wrapped.group(1)
            notes.append({"decision_observed_count": before, "note": note,
                          "original_response_sha256": digest(reply),
                          "note_sha256": digest(encode(note))})
        final_source = native._extract_tag(reply, "final_law")
        if (validate_final and final_source is not None
                and len(agent.conversation_log) + 1 >= agent.min_rounds):
            checked = fit(final_source.strip(), final=True)
            if not checked["ok"] and scientist_remaining() > 0:
                candidates[-1]["rejected_final"] = True
                rejected_finals.append({"request_count": requests, "observed_count": before,
                                        "source_sha256": digest(final_source.strip()), "fit": checked})
                messages.append({"role": "assistant", "content": reply})
                feedback = {
                    "accepted": False, "observed_fit": checked,
                    "requests_remaining": scientist_remaining(),
                    "instruction": "The final program could not be evaluated on acquired data. "
                    "No final submission is accepted yet. Correct the program in another charged "
                    "scientific response. No hidden score is available. Inputs pos1, pos2 and "
                    "velocity2 are Python lists of length two: use np.asarray before arithmetic. "
                    "Return two length-two vectors, not their last coordinates. Both run_mse_fit "
                    "and final_law require discovered_law predictor code, not an optimizer script. "
                    "NumPy/math/SciPy are available; plotting and filesystem writes are not supported."
                }
                messages.append({"role": "user", "content": "<final_validation_output>" +
                                 encode(feedback) + "</final_validation_output>"})
                return complete(model=model, messages=messages, system=system,
                                max_tokens=max_tokens, final_repair=True)
        return reply

    final_source, error, error_kind, status = None, None, None, "running"
    with _NATIVE_LOCK:
        with patch.object(native.llm_client, "complete", complete), \
                patch.object(native, "_compact_json", lambda value: encode(value)), \
                patch.object(native, "_extract_tag", execution_tag):
            try:
                final_source = agent.run()
                if final_source is not None:
                    if not (validate_final and cached_final(final_source)):
                        fit(final_source, final=True)
                    status = ("invalid_final" if validate_final and not candidates[-1]["fit"]["ok"]
                              else "completed")
                else:
                    status = "no_final_submission"
            except (ValueError, RuntimeError) as exc:
                status = ("request_limit" if isinstance(exc, RequestLimit) else
                          "context_limit" if isinstance(exc, ContextLimit) else
                          "execution_failure" if isinstance(exc, ExecutionFailure) else "failed")
                error = f"{type(exc).__name__}: {exc}"
                error_kind = type(exc).__name__
    result = {"manifest": manifest, "status": status, "error": error,
              "error_kind": error_kind, "requests": requests,
              "observations": experiments.observations, "executions": experiments.events,
              "candidates": candidates, "sources": sources, "notebook": notes,
              "emissions": emissions,
              "rejected_finals": rejected_finals,
              "responses": responses, "conversation": agent.conversation_log,
              "final_source_sha256": digest(final_source) if final_source is not None else None}
    if advisor is not None:
        result["planning"] = advisor.records
    if memory is not None:
        result["notebook_memory"] = {
            "contract": NOTEBOOK_MEMORY_CONTRACT, "snapshot": sync_notebook(),
            "path": "external-notebook/notebook.md", "lookups": len(notebook_lookups),
            "denied_lookups": sum(
                row["result"].get("code") == "lookup_budget_exhausted" for row in notebook_lookups),
        }
    if design_advisor is not None:
        result.update(design_reviews=design_advisor.records, scientist_requests=scientist_requests,
                      assistance_requests=design_advisor.requests, request_roles=request_roles)
    elif memory is not None or total_request_limit is not None or require_experiment_intent or action_recovery:
        result.update(scientist_requests=scientist_requests, assistance_requests=0,
                      request_roles=request_roles)
    journal.call("result", result, lambda: {"status": status})
    return result


def grade(result, numerics, designs, truth, position_scale, velocity_scale):
    if not all(math.isfinite(s) and s > 0 for s in (position_scale, velocity_scale)):
        raise ValueError("Development normalization scales must be positive and finite")
    targets = np.asarray([[row["pos2"], row["velocity2"]] for row in truth], dtype=float)
    if targets.shape != (len(designs), 2, 5, 2) or not np.isfinite(targets).all():
        raise ValueError("Invalid private evaluation truth")
    scores = []
    for candidate in result["candidates"]:
        scored = {"observed_count": candidate["observed_count"], "valid": False,
                  "request_count": candidate["request_count"],
                  "source_sha256": candidate["source_sha256"],
                  "final": candidate["final"] and not candidate.get("rejected_final", False)}
        if candidate["fit"].get("ok"):
            prediction = numerics.request(
                "predict", result["sources"][candidate["source_sha256"]],
                parameters=candidate["fit"]["fitted_params"], queries=designs)
            if prediction["ok"]:
                states = np.asarray([[[row["position"], row["velocity"]] for row in case]
                                     for case in prediction["predictions"]])
                with np.errstate(over="ignore", invalid="ignore"):
                    pos = float(np.mean(np.sum((states[:, :, 0] - targets[:, 0]) ** 2, axis=-1)))
                    vel = float(np.mean(np.sum((states[:, :, 1] - targets[:, 1]) ** 2, axis=-1)))
                if all(math.isfinite(v) for v in (pos, vel, pos / position_scale ** 2, vel / velocity_scale ** 2)):
                    scored.update(valid=True, position_mse=pos, velocity_mse=vel,
                                  position_error=pos / position_scale ** 2,
                                  velocity_error=vel / velocity_scale ** 2)
                else:
                    scored["failure"] = {"error_kind": "NonfinitePredictionError",
                                         "error": "Frozen prediction error overflowed"}
            else:
                scored["failure"] = prediction
        scores.append(scored)
    hits = [s["observed_count"] for s in scores if s["valid"] and s["position_error"] <= .01]
    qualifying = [i for i, s in enumerate(scores) if s["valid"] and s["position_error"] <= .01]
    later = scores[qualifying[0]:] if qualifying else []
    return {"first_hit": min(hits) if hits else None,
            "first_hit_requests": scores[qualifying[0]]["request_count"] if qualifying else None,
            "capped_observations": min(hits) if hits else result["manifest"]["observations_limit"] + 1,
            "attained": bool(hits), "candidates": scores, "assigned": 1,
            "status": result["status"], "requests": result["requests"],
            "post_hit_retention": (sum(s["valid"] and s["position_error"] <= .01 for s in later)
                                   / len(later)) if later else None,
            "final": next((row for row in reversed(scores) if row["final"]), None)}
