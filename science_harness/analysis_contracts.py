"""Small public analysis contracts shared with validation; no model or simulator access."""
import copy


LIMITS = {
    "expression_characters": 1000, "expression_nodes": 128, "source_acquisitions": 128,
    "summary_readouts": 2000, "columns": 10, "basis_terms": 6, "parameters": 6,
    "scalar_bindings": 12, "where_fields": 8, "derivative_degree": 4, "trajectory_samples": 100,
    "optimizer_evaluations": 250, "confidence": [0.5, 0.99],
}
DEFAULTS = {"reducer": "median", "ratio_reducer": "mean", "weighting": "absolute",
            "derivative_axis": 0, "derivative_order": 2, "derivative_degree": 3}
OPERATIONS = {
    "summary": {"required": {"field": "selector"}},
    "calculate": {"required": {"expression": "string"}},
    "candidate": {"required": {"expression": "string"}},
    "fit_basis": {"required": {"basis": "array"}},
    "fit_model": {"required": {"expression": "string", "initial": "object", "bounds": "object"}},
    "check": {"required": {}, "one_of": ["model_id", "expression"]},
    "ratio_estimate": {"required": {"group_by": "string", "numerator": "scalar", "denominator": "scalar"}},
}


class AnalysisInputError(ValueError):
    def __init__(self, message, field, constraints, hint):
        super().__init__(message)
        self.detail = {"field": field, "constraints": copy.deepcopy(constraints), "hint": hint}


def validate_operation(action):
    operation = action.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise AnalysisInputError("Unknown analysis operation", "operation", {"allowed": list(OPERATIONS)},
                                 "Choose an advertised operation.")
    contract = OPERATIONS[operation]
    matches = {
        "string": lambda v: isinstance(v, str) and bool(v.strip()),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "scalar": lambda v: v is not None and isinstance(v, (str, bool, int, float)),
        "selector": lambda v: isinstance(v, dict) or (isinstance(v, str) and bool(v.strip())),
    }
    for field, kind in contract["required"].items():
        if not matches[kind](action.get(field)):
            raise AnalysisInputError(f"{operation}.{field} must be a {kind}", field, {"type": kind},
                                     "Use the operation contract; a named view supplies selectors and grouping.")
    if contract.get("one_of") and not any(action.get(key) for key in contract["one_of"]):
        raise AnalysisInputError("check needs model_id or expression", "model_id",
                                 {"one_of": contract["one_of"]}, "Use model_id='latest' for a frozen fit or candidate.")


def describe_analysis():
    return {"contract_version": 2, "contracts": copy.deepcopy(OPERATIONS), "limits": copy.deepcopy(LIMITS),
            "defaults": copy.deepcopy(DEFAULTS),
            "selectors": {
                "path": "public dot paths; numeric array indexes",
                "where": "direct condition keys; exact equality, not OR or nested paths",
                "stratify": "predictor expression or exact declared data./conditions. predictor selector; not y",
                "derivative": {"axis": [0, 1], "order_min": 1, "degree_max": LIMITS["derivative_degree"],
                               "relation": "order <= degree < sample_count; at lies in measured interval"},
            }}
