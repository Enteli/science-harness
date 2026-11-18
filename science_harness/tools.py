"""Bounded instruments. No model-controlled shell or Python execution."""
import inspect
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .books import encode, text


MAX_BATCH_SIZE = 16


def resource_counts(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a map of named integer resource units")
    for key, count in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", key):
            raise ValueError(f"Invalid {name} resource name: {key}")
        if type(count) is not int or not 0 <= count <= 2**63 - 1:
            raise ValueError(f"{name}.{key} must be a nonnegative integer no larger than 2**63-1")
    return dict(value)


@dataclass
class Observation:
    data: object
    source: str
    conditions: Dict = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    sample_id: str = ""

    def record(self):
        record = asdict(self)
        encode(record)
        return record


@dataclass(frozen=True)
class AcquisitionContract:
    unit: str
    description: str
    resources: Tuple[str, ...]
    estimate_cost: Callable[..., Dict[str, int]]


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Observation]
    readout_fields: Tuple[str, ...] = ()
    acquisition: Optional[AcquisitionContract] = None


class ToolRegistry:
    def __init__(self, tools=(), analysis_views=None):
        self.tools = {}
        self.analysis_views = analysis_views or {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"Duplicate tool: {tool.name}")
            if any(not isinstance(name, str) or not name for name in tool.readout_fields):
                raise ValueError("Readout fields must name top-level data arrays of repeated sensor readings")
            if tool.acquisition is not None:
                contract = tool.acquisition
                if not isinstance(contract, AcquisitionContract):
                    raise ValueError("acquisition must be an AcquisitionContract")
                text(contract.unit, "acquisition unit")
                text(contract.description, "acquisition description")
                if (not isinstance(contract.resources, (tuple, list)) or not contract.resources
                        or any(not isinstance(name, str) for name in contract.resources)
                        or len(set(contract.resources)) != len(contract.resources)):
                    raise ValueError("An acquisition contract needs distinct resource names")
                resource_counts(dict.fromkeys(contract.resources, 0), "acquisition resources")
                if not callable(contract.estimate_cost):
                    raise ValueError("Acquisition cost estimation must be a trusted, side-effect-free callable")
            self.tools[tool.name] = tool

    @classmethod
    def read_only(cls, contracts, analysis_views=None):
        def unavailable(**kwargs):
            raise ValueError("Acquisition is disabled in an analysis-only continuation")

        return cls([Tool(name, contract["description"], unavailable,
                         readout_fields=tuple(contract.get("repeated_readout_fields", [])))
                    for name, contract in contracts.items()], analysis_views=analysis_views)

    def describe(self):
        result = {}
        for name, tool in self.tools.items():
            result[name] = dict(description=tool.description, arguments=str(inspect.signature(tool.fn)),
                                repeated_readout_fields=list(tool.readout_fields))
            if tool.acquisition is not None:
                result[name]["acquisition"] = {
                    "unit": tool.acquisition.unit, "description": tool.acquisition.description,
                    "resources": list(tool.acquisition.resources),
                    "cost_accounting": "Trusted upper-bound charge per started invocation; failed attempts are not refunded.",
                }
        return result

    def _tool(self, name, args):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise ValueError("Tool args must be an object")
        tool = self.tools[name]
        inspect.signature(tool.fn).bind(**args)
        return tool

    def quote(self, name, args):
        """Validate inputs and quote costs without executing an instrument."""
        tool = self._tool(name, args)
        if tool.acquisition is None:
            raise ValueError(f"{name} has no trusted acquisition/cost contract; cannot use it with laboratory limits")
        costs = resource_counts(tool.acquisition.estimate_cost(**args), "acquisition cost")
        if set(costs) != set(tool.acquisition.resources) or not any(costs.values()):
            raise ValueError("Cost quote must cover exactly the declared resources and charge at least one unit")
        return costs

    def run(self, name, args):
        tool = self._tool(name, args)
        result = tool.fn(**args)
        if not isinstance(result, Observation):
            raise TypeError("Instruments must return an Observation")
        result.record()
        return result


def completed_acquisitions(events):
    """Successful original acquisitions, including a completed prefix of a failed batch."""
    for event in events:
        if event.get("inherited"):
            continue
        if "acquisitions" in event:
            for item in event["acquisitions"]:
                if item.get("execution") == "ok":
                    yield dict({k: event[k] for k in ("step", "action") if k in event}, **item)
        elif event.get("execution") == "ok":
            yield event
