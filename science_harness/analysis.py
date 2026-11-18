"""Trusted calculations over cited observations, never simulator state or arbitrary code."""
import ast
import copy
import math
import statistics

from .books import encode, text
from .analysis_contracts import AnalysisInputError, DEFAULTS, LIMITS, validate_operation
from .context import visible_source
from .testbeds.expression import Expression, FUNCTIONS

ANALYSIS_VERSION = 8

class Formula:
    """Bind scalar names into the existing restricted arithmetic interpreter."""
    def __init__(self, expression, names):
        text(expression, "expression")
        if len(expression) > LIMITS["expression_characters"]:
            raise AnalysisInputError(f"Expression exceeds {LIMITS['expression_characters']} characters", "expression",
                                     {"max_characters": LIMITS["expression_characters"]},
                                     "Simplify the expression; splitting data does not change this syntax limit.")
        self.tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(self.tree))) > LIMITS["expression_nodes"]:
            raise AnalysisInputError("Expression too complex", "expression",
                                     {"max_ast_nodes": LIMITS["expression_nodes"]},
                                     "Use a simpler equivalent expression or a smaller candidate family.")
        self.names = set(names)
        if self.names & (set(FUNCTIONS) | {"pi", "e"}):
            raise ValueError("Variable names cannot shadow math functions or constants")
        for name in self.names:
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError("Invalid scalar variable name")
        allowed = self.names | set(FUNCTIONS) | {"pi", "e"}
        unknown = {node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name) and node.id not in allowed}
        if unknown:
            raise ValueError(f"Expression references undeclared names: {', '.join(sorted(unknown))}")
        self._bound({name: 0.1 for name in self.names})

    def substitute(self, values):
        names = set(values)
        if not names <= self.names:
            raise ValueError("Cannot substitute undeclared variables")

        class Bind(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in names:
                    value = values[node.id]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("Bindings must be finite numbers")
                    number = float(value)
                    # Python represents a negative literal as unary minus; a signed Constant
                    # unparses without the parentheses needed when it is a power's base.
                    bound = (ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-number))
                             if math.copysign(1., number) < 0 else ast.Constant(number))
                    return ast.copy_location(bound, node)
                return node

        tree = Bind().visit(copy.deepcopy(self.tree))
        return ast.unparse(ast.fix_missing_locations(tree))

    def _bound(self, values):
        # No eval/exec: even the bound expression passes the same restricted AST checks.
        return Expression(self.substitute(values))

    def __call__(self, **values):
        if set(values) != self.names:
            raise ValueError("Missing or unexpected scalar bindings")
        return self._bound(values)(0)


def select(record, path):
    if isinstance(path, dict):
        import numpy as np

        if set(path) != {"derivative"} or not isinstance(path["derivative"], dict):
            raise ValueError("Only an explicit derivative selector is supported")
        spec = path["derivative"]
        time_values = select(record, spec.get("time"))
        values = select(record, spec.get("values"))
        if (not isinstance(time_values, list) or not isinstance(values, list)
                or not 1 <= len(time_values) <= LIMITS["trajectory_samples"] or len(values) != len(time_values)
                or any(not isinstance(row, list) or len(row) != 2 for row in values)):
            raise ValueError("Derivative requires matching time and two-coordinate position arrays")
        times = np.array([numeric(v) for v in time_values])
        axis, order, degree = (spec.get("axis", DEFAULTS["derivative_axis"]),
                               spec.get("order", DEFAULTS["derivative_order"]),
                               spec.get("degree", DEFAULTS["derivative_degree"]))
        if type(axis) is not int or axis not in (0, 1):
            raise ValueError("Derivative axis must be 0 or 1")
        if (type(order) is not int or type(degree) is not int
                or not 1 <= order <= degree <= LIMITS["derivative_degree"]):
            raise AnalysisInputError(f"Derivative requires 1 <= order <= degree <= {LIMITS['derivative_degree']}", "derivative",
                                     {"order_min": 1, "degree_max": LIMITS["derivative_degree"],
                                      "relation": "order <= degree"},
                                     "Order zero is not supported; use a scalar array path for a recorded position.")
        ys = np.array([numeric(v[axis]) for v in values])
        if (len(times) != len(ys) or not degree < len(times) <= LIMITS["trajectory_samples"]
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Need ordered distinct trajectory times and more samples than degree")
        at = numeric(spec.get("at", float(times[0])))
        if not times[0] <= at <= times[-1]:
            raise ValueError("Derivative point must be inside the measured time interval")
        fit = np.polynomial.Polynomial.fit(times, ys, degree).convert()
        return numeric(float(fit.deriv(order)(at)))
    if not isinstance(path, str) or len(path) > 200:
        raise ValueError("Selector must be a short dot-separated data path")
    value = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            if int(part) >= len(value):
                raise ValueError(f"Selector index is outside the available data: {path}")
            value = value[int(part)]
        else:
            raise ValueError(f"Invalid selector: {path}")
    return value


def numeric(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Selected data must be finite numeric values")
    return float(value)


def summarize(value):
    values = value if isinstance(value, list) else [value]
    if not 1 <= len(values) <= LIMITS["summary_readouts"]:
        raise ValueError(f"A summary accepts 1..{LIMITS['summary_readouts']} readings")
    values = [numeric(v) for v in values]
    median = statistics.median(values)
    return {"n": len(values), "mean": statistics.mean(values), "median": median,
            "stddev": statistics.stdev(values) if len(values) > 1 else None,
            "mad": statistics.median(abs(v - median) for v in values),
            "min": min(values), "max": max(values)}


def sources(store, run, ids):
    if ids == "current":
        ids = list(run["evidence"])
    if not isinstance(ids, list) or not 1 <= len(ids) <= LIMITS["source_acquisitions"]:
        raise ValueError(f"evidence_ids must be a list of 1..{LIMITS['source_acquisitions']} IDs, or 'current'")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evidence is not independent replication")
    records = [visible_source(store, run, text(i, "evidence id")) for i in ids]
    if any(r.get("book") != "evidence" for r in records):
        raise ValueError("Analysis accepts observations, not claims or evaluator records")
    if any(r.get("kind") == "derived" for r in records):
        raise ValueError("A derived calculation is not a new acquired measurement")
    return records


def rows_for(records, action):
    columns = action.get("columns")
    if not isinstance(columns, dict) or not {"x", "y"} <= columns.keys() or len(columns) > LIMITS["columns"]:
        raise ValueError("columns must map x, y and optional scalar names to evidence selectors")
    reducer = action.get("reducer", DEFAULTS["reducer"])
    if reducer not in ("median", "mean"):
        raise ValueError("reducer must be median or mean")
    transform = Formula(action.get("transform", "y"), columns.keys())
    rows, summaries = [], []
    for record in records:
        row, summary = {}, {}
        for name, path in columns.items():
            value = select(record, path)
            if isinstance(value, list):
                summary[name] = summarize(value)
                value = summary[name][reducer]
            row[name] = numeric(value)
        try:
            row["y"] = transform(**row)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(f"Transform is undefined for source {record['id']}: {exc}. "
                             "Select compatible observations or change the transform; no row was dropped.") from exc
        rows.append(row)
        summaries.append({"evidence_id": record["id"], "values": row, "readouts": summary})
    return rows, summaries


def residuals(observed, predicted):
    residual = [p - y for p, y in zip(predicted, observed)]
    if not all(math.isfinite(v) for v in residual):
        raise ValueError("Nonfinite residual")
    return {"rmse": math.sqrt(statistics.mean(v*v for v in residual)),
            "relative_rmse": math.sqrt(statistics.mean(
                (d / max(abs(y), 1e-12))**2 for d, y in zip(residual, observed))),
            "residuals": residual}


def acquisition_mean(values, strata=None):
    values = [numeric(v) for v in values]
    if not 1 <= len(values) <= LIMITS["source_acquisitions"]:
        raise ValueError(f"Each ratio group needs 1..{LIMITS['source_acquisitions']} acquisitions, not repeated readouts")
    labels = [None] * len(values) if strata is None else list(strata)
    if len(labels) != len(values):
        raise ValueError("A stratum label is required for every acquisition")
    cells = {}
    for label, value in zip(labels, values):
        key = encode(label)
        cells.setdefault(key, {"label": label, "values": []})["values"].append(value)
    summaries, variance_parts = [], []
    for cell in cells.values():
        data = cell["values"]
        n = len(data)
        spread = statistics.stdev(data) if n > 1 else None
        summaries.append({"label": cell["label"], "n": n, "mean": statistics.mean(data), "stddev": spread})
        if spread is not None:
            variance_parts.append(((n / len(values))**2 * spread**2 / n, n - 1))
    result = {"n": len(values), "mean": statistics.mean(values), "strata": summaries,
              "stddev": statistics.stdev(values) if len(values) > 1 else None,
              "standard_error": None, "effective_degrees_of_freedom": None}
    if len(variance_parts) != len(summaries):
        return result
    variance = sum(part for part, _ in variance_parts)
    result["standard_error"] = math.sqrt(variance)
    result["effective_degrees_of_freedom"] = (
        1 / sum((part / variance)**2 / df for part, df in variance_parts)
        if variance else sum(df for _, df in variance_parts)
    )
    return result


def ratio_statistics(numerator, denominator, scale=1, confidence=0.95,
                     numerator_strata=None, denominator_strata=None):
    """Conditional Fieller interval using a conservative small-sample t critical value."""
    from scipy.stats import t

    scale, confidence = numeric(scale), numeric(confidence)
    if not LIMITS["confidence"][0] <= confidence <= LIMITS["confidence"][1]:
        raise ValueError(f"confidence must be between {LIMITS['confidence'][0]} and {LIMITS['confidence'][1]}")
    groups = {"numerator": acquisition_mean(numerator, numerator_strata),
              "denominator": acquisition_mean(denominator, denominator_strata)}
    a, b = groups["numerator"]["mean"], groups["denominator"]["mean"]
    sa, sb = groups["numerator"]["standard_error"], groups["denominator"]["standard_error"]
    estimate = numeric(scale * (a / b)) if b != 0 else None
    result = {
        "method": "fieller_t_min_effective_df_v1", "estimate": estimate, "scale": scale,
        "confidence": confidence, "confidence_interval": None, "groups": groups,
        "degrees_of_freedom": None, "critical_t": None,
        "interval_status": "not_finitely_bounded",
        "replication_unit": "acquisition",
        "interval_conditions": {
            "requires_independent_acquisitions": True, "uses_normal_mean_approximation": True,
            "includes_common_bias": False, "includes_uncertainty_in_supplied_constants": False,
            "adaptive_selection_adjusted": False,
        },
        "caveats": [
            "Conditional approximation: independent, approximately Gaussian acquisition-level "
            "responses within each group/design stratum; new IDs alone do not establish independence.",
            "Fixed stratum weights equal observed allocation fractions. The smaller group's "
            "Welch-Satterthwaite effective degrees of freedom sets the t critical value; "
            "this is not a distribution-free coverage guarantee.",
            "Shared bias, drift, model error and uncertainty in supplied transform constants "
            "are not included.",
            "Data-dependent choices of grouping, transforms, subsets or stopping are not adjusted "
            "for; fixed-design coverage does not establish coverage for an adaptive investigator.",
            "A wide or unbounded interval is a precision limitation, not proof of structural "
            "non-identifiability.",
        ],
    }
    if sa is not None and sb is not None and b != 0:
        components = [abs(scale) * (sa / abs(b)), abs(estimate) * (sb / abs(b))]
        total = math.hypot(*components)
        if math.isfinite(total):
            result["precision_budget"] = {
                "delta_method_standard_error": total,
                "variance_fractions": {
                    name: (value / total)**2 if total else 0.
                    for name, value in zip(("numerator", "denominator"), components)
                },
                "interpretation": "Local sampling-variance diagnostic, not a confidence guarantee or "
                                  "automatic allocation rule. Preserve controls and design balance.",
            }
        else:
            result["precision_budget"] = {"status": "unavailable", "reason": "Numerical range exceeded"}
    if sa is None or sb is None:
        result.update(interval_status="insufficient_replication",
                      interval_reason="Each group/design stratum needs at least two acquisitions for variance estimation.")
        return result
    df = min(groups["numerator"]["effective_degrees_of_freedom"],
             groups["denominator"]["effective_degrees_of_freedom"])
    critical = float(t.ppf((1 + confidence) / 2, df))
    result.update(degrees_of_freedom=df, critical_t=critical)
    if b == 0 or abs(b) <= critical * sb:
        result["interval_reason"] = "The denominator mean is not resolved away from zero at this confidence."
        return result
    denominator_factor = 1 - (critical * sb / b) ** 2
    center = (a / b) / denominator_factor
    radius = critical * math.hypot((a / b) * (sb / b), math.sqrt(denominator_factor) * (sa / b))
    radius /= denominator_factor
    result["confidence_interval"] = sorted([
        numeric(scale * (center - radius)), numeric(scale * (center + radius)),
    ])
    result["interval_status"] = "bounded" if sa or sb else "degenerate"
    if not sa and not sb:
        result["interval_reason"] = "Zero observed between-acquisition variation does not rule out unobserved uncertainty."
    return result


def filtered_records(records, where):
    if not isinstance(where, dict) or len(where) > LIMITS["where_fields"]:
        raise ValueError("where must be a small map of public condition fields to values")
    for name in where:
        if any(name not in r.get("conditions", {}) for r in records):
            allowed = sorted(set.intersection(*(set(r.get("conditions", {})) for r in records))) if records else []
            raise AnalysisInputError(f"Unknown condition filter: {name}", "where",
                                     {"allowed_keys": allowed[:16], "allowed_key_count": len(allowed)},
                                     "Use direct condition keys, not conditions.* or array paths. "
                                     "Select explicit evidence IDs for a more specific subset.")
    return [r for r in records if all(r["conditions"][key] == value for key, value in where.items())]


def estimate_ratio(records, action, base):
    group_by = text(action.get("group_by"), "group_by")
    selector = group_by if group_by.startswith("conditions.") else "conditions." + group_by
    numerator, denominator = action.get("numerator"), action.get("denominator")
    if any(value is None or isinstance(value, (list, dict)) for value in (numerator, denominator)):
        raise ValueError("numerator and denominator must be scalar condition labels, not filter objects")
    if numerator == denominator:
        raise ValueError("Numerator and denominator groups must be disjoint")
    try:
        labels = [select(record, selector) for record in records]
    except KeyError as exc:
        raise ValueError(f"Unknown grouping condition: {selector}; use a condition key or conditions.<key>") from exc
    a_records = [record for record, label in zip(records, labels) if label == numerator]
    b_records = [record for record, label in zip(records, labels) if label == denominator]
    selected = a_records + b_records
    setups = [r.get("conditions", {}).get("setup_id") for r in selected
              if r.get("conditions", {}).get("setup_id") is not None]
    if len({encode(value) for value in setups}) != len(setups):
        raise ValueError("Repeated setup_id values need a paired/clustered method, not an independent-group interval")
    effective = dict(action, reducer=action.get("reducer", DEFAULTS["ratio_reducer"]))
    a_rows, a_summary = rows_for(a_records, effective)
    b_rows, b_summary = rows_for(b_records, effective)
    strata = None
    stratify = action.get("stratify")
    resolution = None
    if "stratify" in action:
        columns = effective["columns"]
        if isinstance(stratify, str) and stratify.startswith(("conditions.", "data.")):
            aliases = [name for name, selector in columns.items() if name != "y" and selector == stratify]
            if len(aliases) != 1:
                raise AnalysisInputError("stratify selector must identify one declared predictor", "stratify",
                                         {"predictor_aliases": sorted(set(columns) - {"y"})},
                                         "Use a predictor alias or its exact declared source selector, not the response.")
            resolution = {"requested": stratify, "resolved_expression": aliases[0]}
            stratify = aliases[0]
        stratifier = Formula(stratify, set(columns) - {"y"})
        strata = [[stratifier(**{k: v for k, v in row.items() if k != "y"}) for row in group]
                  for group in (a_rows, b_rows)]
    result = ratio_statistics([row["y"] for row in a_rows], [row["y"] for row in b_rows],
                              action.get("scale", 1), action.get("confidence", 0.95),
                              *(strata or (None, None)))
    used = [r["id"] for r in selected]
    output = dict(base, **result, evidence_ids=used, group_by=selector,
                group_labels={"numerator": numerator, "denominator": denominator},
                reducer=effective["reducer"], transform=effective.get("transform", "y"),
                stratify=stratify,
                rows=a_summary + b_summary,
                excluded_evidence_ids=[r["id"] for r in records if r["id"] not in used],
                assumptions=[
                    "The estimand is a scaled ratio of group population means after the selected "
                    "per-acquisition reduction and transform.",
                    "Acquisitions are independent within and between groups. Pooling must justify a "
                    "common mean, or declared strata and their fixed allocation weights must define the estimand.",
                    "Nominal controls are not silently replaced by hidden actual settings.",
                ])
    if resolution is not None:
        output["selector_resolution"] = {"stratify": resolution}
    return output


def analyze(store, run, action):
    validate_operation(action)
    operation = action.get("operation")
    fitted = None
    if operation == "check" and action.get("model_id"):
        text(action["model_id"], "model_id")
        if action["model_id"] == "latest":
            models = [store.get(i) for i in run.get("analyses", [])
                      if store.get(i)["data"]["operation"] in ("candidate", "fit_basis", "fit_model")]
            if not models:
                raise ValueError("No frozen model exists; create a candidate or fit first")
            action = dict(action, model_id=models[-1]["id"])
        fitted = visible_source(store, run, action["model_id"])
        if (fitted.get("kind") != "derived"
                or fitted["data"]["operation"] not in ("candidate", "fit_basis", "fit_model")):
            raise ValueError("model_id must name a frozen candidate or fit artifact")
        recipe = fitted["recipe"]
        action = dict(action)
        for key, default in (("columns", None), ("reducer", "median"), ("transform", "y"),
                             ("where", {}), ("source_tool", None)):
            expected = recipe.get(key, default)
            if key in action and action[key] != expected:
                raise ValueError("Predictive checks must retain the frozen model's columns and transform")
            action[key] = expected
        expected = fitted["data"]["expression"]
        if "expression" in action and action["expression"] != expected:
            raise ValueError("Cannot change a frozen prediction during its check")
        action["expression"] = expected
        action["training_evidence_ids"] = fitted["evidence_ids"]
    if operation == "calculate":
        values = action.get("values", {})
        if not isinstance(values, dict) or len(values) > LIMITS["scalar_bindings"]:
            raise ValueError(f"values must be an object with at most {LIMITS['scalar_bindings']} scalar bindings")
        return {"operation": operation, "version": ANALYSIS_VERSION, "evidence_ids": [],
                "result": Formula(action.get("expression"), values)(**values),
                "note": "Arithmetic on model-supplied constants, not an observation."}
    records = sources(store, run, action.get("evidence_ids", "current"))
    if action.get("source_tool") and any(r.get("tool") != action["source_tool"] for r in records):
        raise ValueError("The selected view does not describe this observation's instrument")
    records = filtered_records(records, action.get("where", {}))
    if not records:
        raise ValueError("No acquisitions match the requested conditions")
    base = {"operation": operation, "version": ANALYSIS_VERSION, "evidence_ids": [r["id"] for r in records],
            "assumptions": [
                "Calculations use recorded/nominal controls, not hidden actual settings.",
                "Repeated readings in a record are not independent setup or specimen replicates.",
                "Fit errors are descriptive, not calibrated uncertainty or proof of the law.",
                "Trajectory derivatives use a finite-window polynomial approximation and amplify noise.",
            ]}
    if operation == "summary":
        field = action.get("field")
        return dict(base, rows=[{
            "evidence_id": r["id"], "conditions": r.get("conditions", {}),
            "sample_id": r.get("sample_id"), **summarize(select(r, field)),
        } for r in records])
    if operation == "ratio_estimate":
        return estimate_ratio(records, action, base)
    rows, summaries = rows_for(records, action)
    import numpy as np

    x = np.asarray([r["x"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    base.update(rows=summaries, transform=action.get("transform", "y"),
                observed_x_range=[float(min(x)), float(max(x))],
                scope_warning="No support for extrapolation beyond the sampled controls. "
                              "Use a new acquisition to check a frozen prediction.")
    if operation in ("check", "candidate"):
        formula = Formula(action.get("expression"), rows[0].keys())
        if any(isinstance(node, ast.Name) and node.id == "y" for node in ast.walk(formula.tree)):
            raise ValueError("A prediction cannot use its observed target y")
        predicted = [formula(**row) for row in rows]
        training = action.get("training_evidence_ids", [])
        if not isinstance(training, list):
            raise ValueError("training_evidence_ids must be a list")
        for entry_id in training:
            visible_source(store, run, text(entry_id, "training evidence id"))
        prospective = (fitted is not None and fitted.get("run_id") == run["id"]
                       and all(r.get("run_id") == run["id"] and r.get("step", 0) > fitted["step"]
                               for r in records))
        result = dict(base, **residuals(y.tolist(), predicted), predictions=predicted,
                    expression=action["expression"], model_id=action.get("model_id"),
                    overlaps_training=bool(set(training) & set(base["evidence_ids"])),
                    prospective=prospective,
                    validation_warning="New IDs establish new acquisition records, not "
                                       "automatically independent specimens or configurations.")
        result["acquisition_ids"] = list(base["evidence_ids"])
        if fitted is not None:
            result["evidence_ids"] = result["evidence_ids"] + [fitted["id"]]
        if action.get("transform", "y") == "y":
            spreads = [r["readouts"].get("y", {}).get("stddev") for r in summaries]
            if all(s is not None for s in spreads):
                result["outside_three_readout_stddev"] = [
                    abs(pred - target) > 3 * spread
                    for pred, target, spread in zip(predicted, y.tolist(), spreads)
                ]
                result["readout_diagnostic"] = (
                    "A discrepancy beyond three within-acquisition standard deviations is a "
                    "diagnostic, not a hypothesis test. Check model, transforms, calibration "
                    "or shared setup error; readout variability does not cover those effects."
                )
        return result
    weighting = action.get("weighting", DEFAULTS["weighting"])
    if weighting not in ("absolute", "relative"):
        raise ValueError("weighting must be absolute or relative")
    scale = np.maximum(np.abs(y), 1e-12) if weighting == "relative" else np.ones_like(y)
    base["weighting"] = weighting
    if operation == "fit_basis":
        basis = action.get("basis")
        if (not isinstance(basis, list) or not 1 <= len(basis) <= LIMITS["basis_terms"]
                or len(rows) < len(basis)):
            raise ValueError(f"Use 1..{LIMITS['basis_terms']} basis terms and at least as many acquired rows as terms")
        predictors = [{name: value for name, value in row.items() if name != "y"} for row in rows]
        functions = [Formula(term, predictors[0].keys()) for term in basis]
        matrix = np.array([[formula(**row) for formula in functions] for row in predictors])
        weighted = matrix / scale[:, None]
        norms = np.linalg.norm(weighted, axis=0)
        if np.any(norms == 0):
            raise ValueError("Degenerate basis")
        normalized = weighted / norms
        coefficients, _, rank, singular = np.linalg.lstsq(normalized, y/scale, rcond=None)
        if rank < len(basis):
            raise ValueError("Rank-deficient design; collect discriminating observations")
        coefficients = coefficients / norms
        predicted = (matrix @ coefficients).tolist()
        expression = " + ".join(f"({c:.16g})*({term})" for c, term in zip(coefficients, basis))
        return dict(base, coefficients=coefficients.tolist(), basis=basis,
                    expression=expression, rank=int(rank),
                    residual_degrees_of_freedom=len(rows)-len(basis),
                    fit_warning="An exactly determined fit has no residual degrees of freedom; "
                                "a perfect training fit is not independent validation.",
                    condition_number=float(singular[0]/singular[-1]),
                    **residuals(y.tolist(), predicted))
    if operation == "fit_model":
        from scipy.optimize import least_squares

        initial = action.get("initial")
        bounds = action.get("bounds")
        if not isinstance(initial, dict) or not 1 <= len(initial) <= LIMITS["parameters"]:
            raise ValueError(f"initial must contain 1..{LIMITS['parameters']} parameters")
        if not isinstance(bounds, dict) or set(bounds) != set(initial):
            raise ValueError("bounds must supply [low, high] for every parameter")
        if set(initial) & set(rows[0]):
            raise ValueError("Parameters cannot shadow data columns")
        if len(rows) < len(initial):
            raise ValueError("Need at least as many acquired rows as fitted parameters")
        # Canonical book JSON must not reorder the optimizer's coordinates on replay.
        names = sorted(initial)
        lo, hi, start = [], [], []
        for name in names:
            if not isinstance(bounds[name], list) or len(bounds[name]) != 2:
                raise ValueError("Each parameter needs finite bounds")
            low, high = map(numeric, bounds[name])
            value = numeric(initial[name])
            if not low < high or not low <= value <= high:
                raise ValueError("Initial parameter outside valid bounds")
            lo.append(low); hi.append(high); start.append(value)
        expression = text(action.get("expression"), "expression")
        formula = Formula(expression, set(rows[0]) | set(names))
        if any(isinstance(node, ast.Name) and node.id == "y" for node in ast.walk(formula.tree)):
            raise ValueError("A fitted model cannot use its observed target y")

        def predict(parameters):
            values = dict(zip(names, map(float, parameters)))
            return np.array([formula(**dict(row, **values)) for row in rows])

        fit = least_squares(lambda p: (predict(p)-y)/scale, start, bounds=(lo, hi),
                            max_nfev=LIMITS["optimizer_evaluations"], x_scale="jac")
        if not fit.success:
            raise ValueError(f"Optimizer did not converge: {fit.message}")
        values = dict(zip(names, map(float, fit.x)))
        # Substitute parameter names only, leaving observable column names in the expression.
        fitted_expression = formula.substitute(values)
        return dict(base, parameters=values, expression=fitted_expression,
                    optimizer_evaluations=int(fit.nfev),
                    local_jacobian_rank=int(np.linalg.matrix_rank(fit.jac)),
                    parameter_count=len(names),
                    residual_degrees_of_freedom=len(rows)-len(names),
                    **residuals(y.tolist(), predict(fit.x).tolist()),
                    optimizer_warning="Local optimization depends on model and initialization; "
                                      "compare alternatives and independently check predictions. "
                                      "A full-rank local Jacobian is not proof of global identifiability.")
    raise ValueError(f"Unknown analysis operation: {operation}")


def resolve_view(action, views):
    """Expand an instrument's public data schema, not a domain-law hypothesis."""
    if "view" not in action:
        return action
    name = action["view"]
    if not isinstance(name, str) or name not in views:
        raise ValueError(f"Unknown analysis view; available: {list(views)}")
    view = views[name]
    result = dict(action)
    if action.get("operation") == "summary":
        result.setdefault("field", view["columns"]["y"])
    else:
        result.setdefault("columns", view["columns"])
    if action.get("operation") == "ratio_estimate" and "group_by" in view:
        result.setdefault("group_by", view["group_by"])
    result["source_tool"] = view["source_tool"]
    return result
