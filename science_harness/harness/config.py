"""Strict, side-effect-free public study configuration."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..benchmarks import get
from ..policies import Arm


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


@dataclass(frozen=True)
class StudyConfig:
    value: dict
    arms: tuple

    @classmethod
    def parse(cls, value):
        required = {"schema_version", "benchmark", "protocol", "worlds", "seeds",
                    "model", "limits", "arms"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"parameters"}:
            raise ValueError("Study config has missing or unknown top-level fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported study schema_version")
        spec, loader = get(value["benchmark"])
        if value["protocol"] != spec.protocol:
            raise ValueError("Protocol must match the benchmark's declared scientific interface")
        worlds, seeds = value["worlds"], value["seeds"]
        if (not isinstance(worlds, list) or not worlds or
                any(w not in spec.worlds for w in worlds) or len(set(worlds)) != len(worlds)):
            raise ValueError("Choose distinct worlds supported by this benchmark")
        if (not isinstance(seeds, list) or not seeds or
                any(type(s) is not int or not 0 <= s < 2**31 for s in seeds) or
                len(set(seeds)) != len(seeds)):
            raise ValueError("Seeds must be distinct nonnegative 31-bit integers")
        model = value["model"]
        if not isinstance(model, dict) or set(model) != {"provider", "id"}:
            raise ValueError("model has exactly provider and id; credentials belong in the environment")
        if model["provider"] not in spec.providers or not isinstance(model["id"], str) or not model["id"]:
            raise ValueError("Provider/model is unsupported by this benchmark profile")
        limits = value["limits"]
        if not isinstance(limits, dict) or set(limits) != {"requests", "observations", "credits"}:
            raise ValueError("limits has exactly requests, observations and credits")
        for field in ("requests", "observations"):
            maximum = 500 if field == "requests" and spec.protocol == "native-trajectory-v1" else 100
            if type(limits[field]) is not int or not 1 <= limits[field] <= maximum:
                raise ValueError(f"limits.{field} must be an integer in 1..{maximum}")
        credits = limits["credits"]
        if type(credits) not in (int, float) or not math.isfinite(credits) or credits < 0:
            raise ValueError("limits.credits must be finite and nonnegative")
        if not isinstance(value["arms"], list) or not value["arms"]:
            raise ValueError("Declare the arms explicitly; no implicit factorial sweep")
        arms = tuple(Arm.parse(a) for a in value["arms"])
        if len({a.name for a in arms}) != len(arms) or any(a.profile not in spec.profiles for a in arms):
            raise ValueError("Arm names must be unique and profiles supported by the benchmark")
        if not isinstance(value.get("parameters", {}), dict):
            raise ValueError("parameters must be an object")
        config = cls(json.loads(json.dumps(value, allow_nan=False)), arms)
        loader().validate(config)
        return config

    def cells(self):
        cells = []
        for index, seed in enumerate(self.value["seeds"]):
            for world in self.value["worlds"]:
                order = self.arms[index % len(self.arms):] + self.arms[:index % len(self.arms)]
                for arm in order:
                    cells.append({"id": f"{world}-{seed}-{arm.name}", "world": world,
                                  "seed": seed, "arm": arm.record()})
        return cells

    def describe(self):
        spec, _ = get(self.value["benchmark"])
        return {"config": self.value, "episodes": len(self.cells()),
                "maximum_model_requests": len(self.cells()) * self.value["limits"]["requests"],
                "credit_cap": self.value["limits"]["credits"], "runtime": spec.runtime,
                "dispatches": 0, "cells": self.cells()}


def load_config(path):
    return StudyConfig.parse(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object))
