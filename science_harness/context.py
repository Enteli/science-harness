"""Question-focused prompt projection; original records are never compacted away."""
import copy
import json
import re
from os.path import commonprefix

from .books import encode, text
from .method import context as method_context


class ContextBudgetExceeded(ValueError):
    pass


def size(value):
    return len(encode(value).encode("utf-8"))


def encode_prompt(request):
    return json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":"))


def words(value):
    return set(re.findall(r"[a-z]{3,}", value.lower())) - {
        "the", "and", "with", "from", "that", "this", "for", "are", "was", "its",
        "what", "which", "have", "into", "when", "then", "current", "underlying",
    }


def ranked(rows, query):
    tokens = words(query)
    scored = []
    for row in rows:
        # Do not search whole measurements or boilerplate JSON field names.
        description = " ".join(str(row.get(key, "")) for key in (
            "title", "claim", "guidance", "when", "scope", "tags", "source",
        ))
        intent = row.get("requested_intent")
        if isinstance(intent, dict):
            description += " " + " ".join(str(intent.get(key, "")) for key in ("purpose", "why", "discriminates"))
            replication = intent.get("replication")
            if isinstance(replication, dict):
                description += " " + " ".join(str(replication.get(key, "")) for key in ("unit", "question"))
        score = len(tokens & words(description))
        if row.get("id") == query:
            score += 100
        if score:
            scored.append((score, row))
    return [r for _, r in sorted(scored, key=lambda pair: -pair[0])]


def event_view(event):
    result = {key: event[key] for key in (
        "step", "action", "execution", "error", "error_detail", "evidence_id", "finding_id",
        "lookup", "accepted", "analysis_id",
        "monitor_update", "batch", "resource_plan", "resource_charge", "preflight_rejected", "interrupted",
        "inherited",
        "method_phase", "method_skill", "method_artifacts",
        "notebook_entry",
        "model_role", "review_proposal", "scientific_review", "review_response",
        "applied_proposal_step", "unperformed_review",
    ) if key in event}
    if "acquisitions" in event:
        result["acquisitions"] = [{key: item[key] for key in (
            "index", "execution", "evidence_id", "resource_charge", "error", "monitor_update",
        ) if key in item} for item in event["acquisitions"]]
    return result


def packet(record, limit=1800):
    if size(record) <= limit:
        return record
    raw = encode(record)
    result = {
        "id": record.get("id"), "revision": record.get("revision"),
        "book": record.get("book"), "status": record.get("status"),
        "preview": raw.encode("utf-8")[:max(0, limit - 400)].decode("utf-8", errors="ignore"),
        "truncated": True, "total_bytes": len(raw.encode()),
        "fetch": {"id": record.get("id"), "revision": record.get("revision"), "offset": 0},
    }
    for key in ("title", "scope", "when"):
        if key in record and size(record[key]) <= 400:
            result[key] = record[key]
    if record.get("book") in ("playbook", "learnings"):
        field = "guidance" if record["book"] == "playbook" else "claim"
        result["fetch_content"] = {"id": record.get("id"), "revision": record.get("revision"), "field": field}
    return result


def observation_view(record, tools):
    from .analysis import summarize

    observation = {key: record[key] for key in (
        "source", "sample_id", "conditions", "data", "uncertainties",
    ) if key in record}
    instrument = tools.tools.get(record["tool"])
    if instrument and instrument.readout_fields:
        data = dict(record["data"])
        summaries = {}
        for field in instrument.readout_fields:
            if field not in data or not isinstance(data[field], list):
                raise ValueError(f"Instrument-declared readout field is not an array: data.{field}")
            summaries[field] = summarize(data.pop(field))
        observation["data"] = data
        observation["readout_summaries"] = summaries
    projected = (observation if size(observation) <= 1800 else
                 packet(dict(observation, id=record["id"], revision=record["revision"])))
    return {"evidence_id": record["id"], "revision": record["revision"],
            "step": record["step"], "tool": record["tool"],
            "observation": projected}


def factor_observation_metadata(rows, tools):
    if not rows or any(row["observation"].get("truncated") for row in rows):
        return rows, {}
    observations = [row["observation"] for row in rows]
    protected = {view["group_by"].removeprefix("conditions.").split(".")[0]
                 for view in tools.analysis_views.values() if isinstance(view.get("group_by"), str)}
    conditions = {
        key: value for key, value in observations[0].get("conditions", {}).items()
        if not key.endswith("_id") and key not in protected
        and all(key in observation.get("conditions", {})
                and encode(observation["conditions"][key]) == encode(value) for observation in observations)
    }
    defaults = {"conditions": conditions} if conditions else {}
    if "source" in observations[0] and all(o.get("source") == observations[0]["source"] for o in observations):
        defaults["source"] = observations[0]["source"]
    compact = []
    for row, observation in zip(rows, observations):
        value = dict(observation)
        if conditions:
            value["conditions"] = {k: v for k, v in observation["conditions"].items() if k not in conditions}
        if "source" in defaults:
            value.pop("source")
        compact.append(dict(row, observation=value))
    if size(compact) + size(defaults) + 80 < size(rows):
        return compact, defaults
    return rows, {}


def expand_observations(request):
    rows = unpack_observations(request["observation_table"]) if "observation_table" in request else request["observations"]
    defaults = request.get("observation_defaults", {})
    if not defaults:
        return rows
    result = []
    for row in rows:
        observation = {**defaults, **row["observation"]}
        if "conditions" in defaults or "conditions" in row["observation"]:
            observation["conditions"] = {**defaults.get("conditions", {}), **row["observation"].get("conditions", {})}
        result.append(dict(row, observation=observation))
    return result


def pack_observations(rows):
    """Lossless row table; dictionaries compress repeated strings, never numeric readings."""
    def flatten(value, path=()):
        if isinstance(value, dict) and value:
            return {nested: item for key, child in value.items()
                    for nested, item in flatten(child, path + (key,)).items()}
        return {path: value}

    flat = [flatten(row) for row in rows]
    paths = list(dict.fromkeys(path for row in flat for path in row))
    columns, constants, data, missing = [], [], [[] for _ in rows], []
    for path in paths:
        present = [path in row for row in flat]
        values = [row.get(path) for row in flat]
        if all(present) and all(encode(value) == encode(values[0]) for value in values):
            constants.append({"path": list(path), "value": values[0]})
            continue
        column = {"path": list(path)}
        if all(isinstance(value, str) for value, exists in zip(values, present) if exists):
            prefix = commonprefix([value for value, exists in zip(values, present) if exists])
            if len(prefix) >= 16 and len(prefix) * (sum(present)-1) > 30:
                column["prefix"] = prefix
                values = [value[len(prefix):] if exists else None for value, exists in zip(values, present)]
            dictionary = list(dict.fromkeys(value for value, exists in zip(values, present) if exists))
            coded = [dictionary.index(value) if exists else None for value, exists in zip(values, present)]
            if size(dictionary) + size(coded) + 20 < size(values):
                column["dictionary"] = dictionary
                values = coded
        index = len(columns)
        columns.append(column)
        for row_number, (value, exists) in enumerate(zip(values, present)):
            data[row_number].append(value)
            if not exists:
                missing.append([row_number, index])
    return {"format": "lossless-row-table-v1", "columns": columns, "constants": constants,
            "rows": data, "missing": missing}


def unpack_observations(table):
    if table.get("format") != "lossless-row-table-v1":
        raise ValueError("Unknown observation table format")
    missing = {tuple(position) for position in table["missing"]}

    def assign(row, path, value):
        for key in path[:-1]:
            row = row.setdefault(key, {})
        row[path[-1]] = copy.deepcopy(value)

    result = []
    for index, values in enumerate(table["rows"]):
        if len(values) != len(table["columns"]):
            raise ValueError("Observation table row does not match its column definitions")
        row = {}
        for constant in table["constants"]:
            assign(row, constant["path"], constant["value"])
        for col, (column, value) in enumerate(zip(table["columns"], values)):
            if (index, col) in missing:
                continue
            if "dictionary" in column:
                if type(value) is not int or not 0 <= value < len(column["dictionary"]):
                    raise ValueError("Invalid observation table dictionary index")
                value = column["dictionary"][value]
            if "prefix" in column:
                value = column["prefix"] + value
            assign(row, column["path"], value)
        result.append(row)
    return result


def analysis_view(record):
    data = dict(record["data"])
    data.pop("evidence_ids", None)
    rows = data.pop("rows", None)
    if data.get("operation") == "summary" and rows is not None:
        data["rows"] = [{k: v for k, v in row.items() if k not in ("conditions", "sample_id")}
                        for row in rows]
    recipe = {key: value for key, value in record.get("recipe", {}).items()
              if key not in ("evidence_ids", "why", "discriminates", "assumptions", "uncertainties")}
    return {**essential_analysis_result(record), **record, "data": data, "recipe": recipe,
            "projection": "Repeated fit inputs, duplicate citation lists and action prose omitted; original unchanged.",
            "fetch": {"id": record["id"], "revision": record["revision"], "field": "data"}}


def analysis_summary(record):
    data = record["data"]
    essential = essential_analysis_result(record)
    summary = dict(essential["data"])
    if "precision_budget" in data:
        summary["precision_budget"] = data["precision_budget"]
    if "groups" in data:
        summary["groups"] = {
            name: {key: value for key, value in group.items() if key != "strata"}
            for name, group in data["groups"].items()
        }
    return {
        **essential,
        "id": record["id"], "revision": record["revision"], "book": "evidence", "kind": "derived",
        "source": record["source"], "step": record.get("step"), "data": summary,
        "projection": "Critical results only; full recipe, assumptions, rows and citations remain fetchable.",
        "source_count": len(record.get("evidence_ids", [])),
        "fetch": {"id": record["id"], "revision": record["revision"], "field": "data"},
        "fetch_sources": {"id": record["id"], "revision": record["revision"], "field": "evidence_ids"},
    }


def essential_analysis_result(record):
    data = record["data"]
    fields = (
        "operation", "version", "expression", "result", "parameters", "coefficients",
        "estimate", "scale", "confidence_interval", "confidence", "interval_status", "interval_reason",
        "interval_conditions", "method", "replication_unit", "degrees_of_freedom", "transform", "reducer", "stratify",
        "group_by", "group_labels", "selector_resolution", "observed_x_range", "rmse", "relative_rmse", "residual_degrees_of_freedom",
        "model_id", "prospective", "overlaps_training", "scope_warning", "validation_warning",
        "fit_warning", "optimizer_warning", "rank", "condition_number", "local_jacobian_rank", "parameter_count",
        "note", "units", "target",
    )
    result = {
        "id": record["id"], "revision": record["revision"], "step": record.get("step"),
        "data": {key: data[key] for key in fields if key in data},
        "projection": "Minimum result; full recipe, source IDs and diagnostics remain fetchable by this ID.",
    }
    if "evidence_ids" in record:
        result["source_count"] = len(record["evidence_ids"])
    if "acquisition_ids" in data:
        result["acquisition_count"] = len(data["acquisition_ids"])
    if data.get("operation") in ("candidate", "fit_basis", "fit_model"):
        result["frozen_model"] = {
            "model_id": record["id"],
            "use": "Already frozen. Use analyze/check with this model_id and explicit evidence_ids of "
                   "later acquisitions; selectors and transform are inherited. No extra candidate, view "
                   "or copied expression is needed.",
        }
    if "groups" in data:
        result["data"]["groups"] = {name: {"n": group["n"]} for name, group in data["groups"].items()}
    if data.get("outside_three_readout_stddev") is not None:
        result["data"]["outside_readout_band_count"] = sum(data["outside_three_readout_stddev"])
        result["data"]["readout_diagnostic"] = data.get("readout_diagnostic")
    recipe = record.get("recipe", {})
    selectors = ("view", "columns", "field", "where", "reducer", "weighting")
    if data.get("operation") == "calculate":
        selectors = ("expression", "values")
    selected = {key: recipe[key] for key in selectors if key in recipe}
    if selected:
        result["recipe"] = selected
    if data.get("operation") == "summary":
        rows = data["rows"]
        result["data"].update(
            rows=[{key: value for key, value in row.items() if key not in ("conditions", "sample_id")}
                  for row in rows[:1]],
            row_count=len(rows), next_start=1 if len(rows) > 1 else None,
        )
        result["fetch_rows"] = {"id": record["id"], "revision": record["revision"], "field": "data.rows"}
    return result


def essential_event_result(event):
    result = {"id": f"event:{event['step']}", "revision": 1, "step": event["step"],
              "action_kind": event.get("action", {}).get("action")}
    if "lookup" in event:
        original = event["lookup"]
        reply = {key: original[key] for key in ("id", "revision") if key in original}
        reply.update(delivered=False, error="Source reply saved but not delivered within the context budget. "
                     "Narrow the field, page or query; do not repeat the experiment.")
        result.update(lookup=reply, lookup_delivery="deferred_for_budget")
        delivered = {key: value for key, value in result.items() if key != "lookup_delivery"}
        delivered["lookup"] = original
        if len(encode_prompt(delivered)) <= len(encode_prompt(result)):
            result = delivered
    if "error" in event:
        error = event["error"]
        result.update(error=error[:1200], error_truncated=len(error) > 1200)
        if "error_detail" in event:
            result["error_detail"] = event["error_detail"]
    return result


def visible_source(store, run, entry_id, revision=None):
    if entry_id.startswith("notebook:"):
        from .notebook import snapshot

        suffix = entry_id.removeprefix("notebook:")
        if not re.fullmatch(r"0|[1-9][0-9]*", suffix):
            raise ValueError("Invalid notebook snapshot handle")
        if revision not in (None, 1):
            raise ValueError("Notebook snapshot revisions are immutable")
        return snapshot(store, run, int(suffix))
    if entry_id.startswith("event:"):
        try:
            step = int(entry_id.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("Invalid event handle") from exc
        for event in run["events"]:
            if event["step"] == step and "action" in event:
                return dict(event_view(event), id=entry_id, revision=1)
        raise ValueError("Unknown completed event")
    pinned = run["book_snapshot"].get(entry_id)
    if entry_id in run["evidence"] or entry_id in run["findings"] or entry_id in run.get("analyses", []):
        pinned = 1
    if pinned is None:
        raise ValueError("Source is outside this investigation's frozen knowledge snapshot")
    if revision is not None and revision != pinned:
        raise ValueError("Only the run's pinned revision is available")
    return store.get(entry_id, pinned)


def lookup(store, run, action):
    if action["action"] == "fetch":
        entry_id = text(action.get("id"), "id")
        source = visible_source(store, run, entry_id, action.get("revision"))
        if "field" in action:
            from .analysis import select

            if "offset" in action or "limit" in action:
                raise ValueError("Structured field fetch uses start/count, not character offset/limit")
            field = text(action["field"], "field")
            try:
                value = select(source, field)
            except KeyError as exc:
                raise ValueError(f"Unknown source field: {field}") from exc
            result = {"id": entry_id, "revision": source["revision"],
                      "format": "structured field", "field": field}
            if isinstance(value, list):
                start, count = action.get("start", 0), action.get("count", 10)
                if (type(start) is not int or not 0 <= start <= len(value)
                        or type(count) is not int or not 1 <= count <= 50):
                    raise ValueError("Structured fetch requires start in [0, length] and count in [1, 50]")
                result.update(value=value[start:start + count], start=start, total_items=len(value),
                              next_start=start + count if start + count < len(value) else None)
            else:
                if "start" in action or "count" in action:
                    raise ValueError("start/count apply only to array fields")
                result["value"] = value
            if size(result) > 4000:
                raise ValueError("Selected field page exceeds 4000 bytes; reduce count or select a narrower field")
            return result
        offset, limit = action.get("offset", 0), action.get("limit", 1600)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError("fetch offset >= 0 and limit in [1, 2000] must be integers")
        raw = encode(source)
        if offset > len(raw):
            raise ValueError("fetch offset exceeds source length")
        return {"id": entry_id, "revision": source["revision"],
                "format": "canonical JSON character page", "offset": offset,
                "text": raw[offset:offset + limit], "total_characters": len(raw),
                "next_offset": offset + limit if offset + limit < len(raw) else None}
    query = text(action.get("query"), "query")
    offset = action.get("offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("search offset must be a nonnegative integer")
    rows = store.frozen_entries(run["book_snapshot"])
    rows.extend(store.get(i) for i in run["evidence"] + run["findings"])
    rows.extend(store.get(i) for i in run.get("analyses", []))
    if type(action.get("current_only", False)) is not bool:
        raise ValueError("current_only must be a boolean")
    if action.get("current_only", False):
        rows = [r for r in rows if r.get("run_id") == run["id"]]
    selected = ranked(rows, query)
    return {"query": query, "total_matches": len(selected), "offset": offset,
            "sources": [packet(r, 600) for r in selected[offset:offset + 5]],
            "next_offset": offset + 5 if offset + 5 < len(selected) else None}


def update_state(store, run, action):
    from .books import strings

    state = {key: list(value) if isinstance(value, list) else value
             for key, value in run["working_state"].items()}
    if "question" in action:
        state["question"] = text(action["question"], "question")
    for field in ("hypotheses", "assumptions", "uncertainties"):
        for value in strings(action.get(field, []), field):
            text(value, field)
            if value not in state[field]:
                state[field].append(value)
    resolutions = action.get("resolve", [])
    if not isinstance(resolutions, list):
        raise ValueError("resolve must be a list")
    for resolution in resolutions:
        if not isinstance(resolution, dict):
            raise ValueError("Each resolution must be an object")
        field = resolution.get("field")
        if field not in ("hypotheses", "assumptions", "uncertainties"):
            raise ValueError("Invalid resolution field")
        value = resolution.get("text")
        if value not in state[field]:
            raise ValueError("Cannot resolve an unrecorded state item")
        text(resolution.get("reason"), "resolution reason")
        store.validate_citations(resolution, required=True)
        for evidence_id in resolution["evidence_ids"]:
            visible_source(store, run, evidence_id)
        state[field].remove(value)
    # Detailed resolutions remain in the immutable action history, not a growing summary.
    run["working_state"] = state


def analysis_capabilities(run, tools):
    from .analysis_contracts import OPERATIONS, describe_analysis

    return {
        "action": "analyze", "operations": list(OPERATIONS), **describe_analysis(),
        "available_now": True, "requires_host_tools": False,
        "conclusive_finish_requires_predictive_check": run.get("require_predictive_check", False),
        "rolling_ratio_monitor_enabled": run.get("monitor_enabled", bool(run.get("analysis_monitor"))),
        "views": tools.analysis_views,
    }


def monitor_view(run):
    monitor = run["analysis_monitor"]
    view = {"status": monitor["status"], "last_update_step": monitor["last_update_step"],
            "automatic_calculations": run.get("automatic_analysis_calls", 0),
            "note": "Fixed recipe; derived updates, not new observations or anytime-valid intervals."}
    if monitor.get("error"):
        view.update(error=monitor["error"], last_successful_result_is_stale=True)
    return view


def laboratory_context(run):
    budget = run.get("resource_budget")
    if budget is None:
        return {}
    return {
        "max_batch_size": run.get("max_batch_size", 1),
        "laboratory_budget": {
            "limits": budget["limits"], "charged": budget["charged"],
            "remaining": {key: limit - budget["charged"][key] for key, limit in budget["limits"].items()},
            "note": "Quoted upper-bound charges include failed/interrupted started calls. Unstarted work is uncharged.",
        },
    }


def build_context(store, run, tools, system, policy="focused", max_bytes=24000, k=8):
    if policy in ("legacy", "full"):
        books = store.search(f"{run['goal']} {run['context']} {encode(run['events'][-2:])}", k)
        request = {"goal": run["goal"], "apparatus": run["context"], "tools": tools.describe(),
                   "books": books, "events": run["events"], "step": run["steps"],
                   "max_steps": run["max_steps"]}
        request.update(laboratory_context(run))
        if policy == "full":
            request["working_state"] = run["working_state"]
        if run.get("analyses"):
            request["analyses"] = [store.get(i) for i in run["analyses"]]
        if run.get("analysis_enabled"):
            request["analysis_capabilities"] = analysis_capabilities(run, tools)
        if run.get("analysis_monitor") is not None:
            request["analysis_monitor"] = monitor_view(run)
        if run.get("scientific_method"):
            request["scientific_method"] = method_context(run)
        return request, {"policy": policy, "input_bytes": len(system.encode()) + size(request),
                         "retrieved": {r["id"]: r["revision"] for rs in books.values() for r in rs}}
    if policy != "focused":
        raise ValueError(f"Unknown context policy: {policy}")
    request = {
        "goal": run["goal"], "apparatus": run["context"], "tools": tools.describe(),
        "working_state": run["working_state"], "step": run["steps"],
        "max_steps": run["max_steps"], "events": [], "observations": [],
        "books": {name: [] for name in store.BOOKS},
        "book_context": {"current_run_id": run["id"],
                         "meaning": "Books are prior knowledge. Their old specimen measurements are not "
                                    "current acquisitions; use them to guide, not replace, this investigation."},
        "history": {"total_events": len(run["events"]), "observation_count": len(run["evidence"]),
                    "note": "Originals persist. Fetch source IDs or event:<step>; search older sources. "
                            "A missing item from this view is not a refutation or resolution."},
        "context": {"policy": policy, "max_input_bytes": max_bytes, "omitted_packets": 0},
        "decision_budget": {
            "remaining_including_this_call": run["max_steps"] - run["steps"] + 1,
            "phase": "final_synthesis" if run["steps"] == run["max_steps"] else run.get("phase", "investigation"),
            "instruction": (
                "This is the last decision. Return finish with the best-supported scoped result or justified "
                "inconclusiveness. No later decision can use another tool/lookup/analysis result."
                if run["steps"] == run["max_steps"] else
                "Leave the last decision for final synthesis; all actions consume the same decision budget."
            ),
        },
    }
    request.update(laboratory_context(run))
    if run.get("memory_policy", "inline") == "notebook":
        from .notebook import working_context

        method_view, index = working_context(run, method_context(run))
        request["lab_notebook"] = index
        if method_view is not None:
            request["scientific_method"] = method_view
    elif run.get("scientific_method"):
        request["scientific_method"] = method_context(run)
    if run.get("scientific_review"):
        from .scientific_review import working_context as review_context

        request["scientific_review"] = review_context(run)
    if run.get("continuation"):
        request["tools"] = {}
        request["continuation"] = {
            **run["continuation"],
            "note": "All listed acquisitions are reused from the parent apparatus, not new measurements. "
                    "Instruments and automatic replay are disabled; inherited events remain fetchable.",
        }
    if run.get("analysis_enabled"):
        request["analysis_capabilities"] = analysis_capabilities(run, tools)

    if run.get("analysis_monitor") is not None:
        request["analysis_monitor"] = monitor_view(run)

    critical_analyses = []
    if run.get("analyses"):
        critical_analyses.append(run["analyses"][-1])
        uncertainty = next((entry_id for entry_id in reversed(run["analyses"])
                            if "confidence_interval" in store.get(entry_id)["data"]), None)
        if uncertainty and uncertainty not in critical_analyses:
            critical_analyses.append(uncertainty)
        uncertainty_step = store.get(uncertainty).get("step", 0) if uncertainty is not None else None
        newer_observations = (uncertainty_step is not None and any(
            store.get(entry_id).get("step", 0) > uncertainty_step for entry_id in run["evidence"]))
        if len(critical_analyses) > 1 or newer_observations:
            request["context"]["uncertainty_scope_note"] = (
                "The retained interval belongs to its own original calculation and sources, "
                "not automatically to the latest model or all later observations."
            )

    recent = run["events"][-3:]
    urgent = recent[-1] if recent and ("lookup" in recent[-1] or "error" in recent[-1]) else None
    if run.get("analyses"):
        request["analyses"] = [essential_analysis_result(store.get(entry_id)) for entry_id in critical_analyses]
    if urgent is not None:
        request["events"] = [essential_event_result(urgent)]

    system_bytes = len(system.encode("utf-8"))
    max_omissions = len(recent) + 2 * len(run.get("analyses", [])) + len(run["book_snapshot"]) + 2

    def input_bytes():
        return system_bytes + len(encode_prompt(request).encode("utf-8"))

    def fits():
        # Only the omission counter can grow after an optional packet is rejected.
        counter_reserve = len(str(max_omissions)) - len(str(request["context"]["omitted_packets"]))
        return input_bytes() + max(0, counter_reserve) <= max_bytes

    def add(sequence, value):
        sequence.append(value)
        if fits():
            return True
        sequence.pop()
        request["context"]["omitted_packets"] += 1
        return False

    def replace(sequence, index, value):
        previous = sequence[index]
        sequence[index] = value
        if fits():
            return True
        sequence[index] = previous
        request["context"]["omitted_packets"] += 1
        return False

    if not fits():
        raise ContextBudgetExceeded("Goal, apparatus, tools, unresolved state and required results exceed "
                                    "the context budget; narrow the investigation or raise the cap")
    observed = []
    for entry_id in run["evidence"]:
        record = store.get(entry_id)
        item = observation_view(record, tools)
        cautions = item["observation"].pop("uncertainties", None)
        if cautions:
            groups = request.setdefault("measurement_cautions", [])
            group = next((g for g in groups if g["uncertainties"] == cautions), None)
            if group is None:
                group = {"uncertainties": cautions, "evidence_ids": []}
                groups.append(group)
            group["evidence_ids"].append(entry_id)
        observed.append(item)
    observed, defaults = factor_observation_metadata(observed, tools)
    if defaults:
        request["observation_defaults"] = defaults
    table = pack_observations(observed) if observed and run.get("max_batch_size", 1) > 1 else None
    if table is not None and size({"observation_table": table}) < size({"observations": observed}):
        request["observation_table"] = table
        del request["observations"]
        cautions = request.get("measurement_cautions", [])
        if len(cautions) == 1 and cautions[0]["evidence_ids"] == run["evidence"]:
            request["measurement_cautions"] = [{"uncertainties": cautions[0]["uncertainties"],
                                               "applies_to": "all current acquisitions"}]
        if not fits():
            compact = copy.deepcopy(observed)
            for item in compact:
                summaries = item["observation"].get("readout_summaries", {})
                for field, summary in summaries.items():
                    summaries[field] = {k: v for k, v in summary.items() if k in ("n", "mean", "median", "stddev")}
            smaller = pack_observations(compact)
            previous_bytes = input_bytes()
            request["observation_table"] = smaller
            request["context"]["readout_summary_projection"] = (
                "Only n, mean, median and stddev are inline; tails/MAD are not shown. "
                "Fetch original data arrays or use analyze summary for complete readings/statistics."
            )
            if input_bytes() >= previous_bytes:
                request["observation_table"] = table
                del request["context"]["readout_summary_projection"]
    else:
        request["observations"] = observed
    if not fits():
        raise ContextBudgetExceeded(
            "Mandatory current acquisitions, usable calculation results and the latest action response "
            f"require {input_bytes()} input bytes (limit {max_bytes}); narrow the investigation or raise "
            "the cap. No acquisition or required result was silently replaced by a handle.")
    required_input_bytes = input_bytes()

    # The complete minimum working set is reserved before any richer source or result is admitted.
    if urgent is not None:
        projected = dict(event_view(urgent), id=f"event:{urgent['step']}", revision=1)
        if "lookup" not in urgent:
            projected = dict(packet(projected), step=urgent["step"], error=urgent["error"],
                             action_kind=urgent.get("action", {}).get("action"))
            if "error_detail" in urgent:
                projected["error_detail"] = urgent["error_detail"]
        if (not replace(request["events"], 0, projected)
                and request["events"][0].get("lookup_delivery") == "deferred_for_budget"):
            original = urgent["lookup"]
            if isinstance(original.get("value"), dict) and original.get("field"):
                reply = dict(request["events"][0]["lookup"], available_fields=[
                    original["field"] + "." + key for key in list(original["value"])[:12]
                    if len(original["field"] + "." + key) <= 200
                ])
                replace(request["events"], 0, dict(request["events"][0], lookup=reply))

    def add_event(event):
        projected = dict(event_view(event), id=f"event:{event['step']}", revision=1)
        value = dict(packet(projected), step=event["step"])
        if "error" in event:
            value["error"] = event["error"]
            value["action_kind"] = event.get("action", {}).get("action")
            if "error_detail" in event:
                value["error_detail"] = event["error_detail"]
        add(request["events"], value)

    def add_analysis(entry_id):
        record = store.get(entry_id)
        if not add(request["analyses"], analysis_view(record)):
            add(request["analyses"], analysis_summary(record))

    for index, entry_id in enumerate(critical_analyses):
        record = store.get(entry_id)
        for projection in (analysis_view(record), analysis_summary(record)):
            if replace(request["analyses"], index, projection):
                break
    for event in reversed(recent):
        if event is not urgent:
            add_event(event)
    request["events"].sort(key=lambda event: event["step"])
    for entry_id in reversed(run.get("analyses", [])[-3:-1]):
        if entry_id not in critical_analyses and not store.get(entry_id).get("automatic"):
            add_analysis(entry_id)
    query = " ".join([run["working_state"]["question"]] + run["working_state"]["hypotheses"]
                     + run["working_state"]["uncertainties"])
    candidates = sorted(ranked(store.frozen_entries(run["book_snapshot"]), query),
                        key=lambda record: record["book"] == "evidence")
    counts = {book: 0 for book in store.BOOKS}
    retrieved = {}
    loaded_ids = set(run["evidence"])
    for record in candidates:
        book = record["book"]
        if counts[book] >= k or record["id"] in loaded_ids:
            continue
        selected = dict(packet(record), source_relationship="pinned_prior")
        # Keep support AND counterevidence discoverable, even when raw arrays do not fit.
        links = []
        for key in ("evidence_ids", "contradiction_ids"):
            for entry_id in record.get(key, []):
                source = visible_source(store, run, entry_id)
                links.append({"id": entry_id, "revision": source["revision"], "role": key,
                              "fetch": {"id": entry_id, "revision": source["revision"]}})
        if links:
            selected = dict(selected, source_links=links)
        if add(request["books"][book], selected):
            counts[book] += 1
            retrieved[record["id"]] = record["revision"]
    if not fits():
        raise ContextBudgetExceeded("Context accounting exceeded the configured byte limit")
    return request, {"policy": policy, "input_bytes": input_bytes(),
                     "required_input_bytes": required_input_bytes,
                     "retrieved": retrieved, "omitted_packets": request["context"]["omitted_packets"]}
