"""Small force/extension world with distinct observation failure modes."""
import math
import random
import uuid

from ..tools import AcquisitionContract, Observation, Tool, ToolRegistry


NOISE_SCENARIOS = ("noise_control", "measurement_noise", "configuration_noise", "combined_noise")
SCENARIOS = ("clean", "noise", "offset", "outliers", "gain", "unidentifiable", "mixed") + NOISE_SCENARIOS


def finite(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


class Spring:
    def __init__(self, scenario="noise", seed=0):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown spring scenario: {scenario}")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = random.Random(seed)
        self._stiffness = rng.uniform(60, 140)
        self._offset = rng.uniform(0.5, 1.5) if scenario in ("offset", "mixed") else 0.0
        self._gain = rng.uniform(0.65, 0.85) if scenario in ("gain", "unidentifiable", "mixed") else 1.0
        self._sigma = 0.0 if scenario == "clean" else 0.02
        if scenario in NOISE_SCENARIOS:
            self._sigma = 0.2 if scenario in ("measurement_noise", "combined_noise") else 0.0
        self._setup_sigma = 0.06 if scenario in ("configuration_noise", "combined_noise") else 0.0
        self._setup_rng = random.Random(seed + 20000)
        self._rng = random.Random(seed + 10000)
        self._scenario, self._seed = scenario, seed
        self._apparatus_id = f"spring-{uuid.uuid4().hex}"
        self.measurements = 0
        self.setups = 0
        self._reference_available = scenario != "unidentifiable"
        self.context = (
            f"Current apparatus ID: {self._apparatus_id}. Prior observations from other "
            "apparatus IDs do not establish this specimen's parameters. "
            "Identify the specimen's intrinsic linear spring stiffness (N/m), not merely "
            "the slope of the instrument readout. The instrument commands nominal extension "
            "in metres and reads force in newtons. Actual extension may differ from nominal "
            "extension by an unknown fixed positive scale. Force readout may have a fixed "
            "offset, random error, or occasional outliers. Do not assume these effects exist, "
            "but check when relevant. The specimen and instrument remain fixed within a run. "
            "Controls: specimen='blank' has zero physical force; specimen='spring' is the "
            "unknown specimen. "
            + ("A reference spring of known stiffness 100 N/m is available using "
               "specimen='reference', sharing the same instrument scale and force offset. "
               if self._reference_available else
               "No reference spring or independent extension calibration is available. ")
            + "measure accepts extension_m [-0.05, 0.05], repeats [1, 21]. "
            "Submit {\"stiffness_n_per_m\": number, \"identifiable\": true}, plus optional "
            "offset_n and extension_scale. If intrinsic stiffness cannot be distinguished "
            "from instrument scale, finish inconclusive with {\"identifiable\": false} "
            "and explain what additional measurement is needed."
        )
        if scenario in NOISE_SCENARIOS:
            self.context += (
                " PREPARATION UNCERTAINTY: A new measure call independently recreates the "
                "commanded setting. Realized extension may vary between calls even when the "
                "nominal command is identical. All repeats WITHIN one call share the same "
                "realized extension. Sensor noise may vary between individual readouts. "
                "Reference and specimen use the same stationary preparation process, with "
                "independent relative-setting errors drawn from the same distribution across "
                "preparations. The distribution's size is unknown. This shared-process "
                "assumption is part of this apparatus contract, not a universal laboratory fact. "
                "Repeated readouts cannot remove a shared setup error; use independent "
                "preparations when needed. Neither actual extension nor the error magnitude "
                "is directly observable. A new call is a new setup, not a new physical specimen."
            )
        self.tools = ToolRegistry([Tool(
            "measure", "Measure force; args: extension_m, specimen (spring/blank/reference), repeats.",
            self.measure, readout_fields=("force_n",),
            acquisition=AcquisitionContract(
                unit="preparation",
                description="One call recreates the nominal setting on the same physical specimen. "
                            "repeats are sensor readouts within that preparation; independence follows "
                            "the apparatus contract, not the number of calls.",
                resources=("preparations", "readouts"),
                estimate_cost=self.measure_cost,
            ),
        )], analysis_views={"response": {
            "source_tool": "measure",
            "description": "Force versus nominal extension; filter specimen roles rather than pooling them.",
            "columns": {"x": "conditions.nominal_extension_m", "y": "data.force_n"},
            "group_by": "conditions.specimen",
        }})

    @property
    def manifest(self):
        return {"testbed": "spring", "version": 1, "observation_contract": 3, "scenario": self._scenario,
                "seed": self._seed, "apparatus_id": self._apparatus_id}

    def _measurement_setting(self, extension_m, specimen, repeats):
        x = finite(extension_m, "extension_m", -0.05, 0.05)
        if type(repeats) is not int or not 1 <= repeats <= 21:
            raise ValueError("repeats must be an integer between 1 and 21")
        if specimen not in ("spring", "blank", "reference"):
            raise ValueError("Unknown specimen")
        if specimen == "reference" and not self._reference_available:
            raise ValueError("Reference specimen unavailable; scale is not independently calibrated")
        return x

    def measure_cost(self, extension_m: float, specimen: str = "spring", repeats: int = 1):
        self._measurement_setting(extension_m, specimen, repeats)
        return {"preparations": 1, "readouts": repeats}

    def measure(self, extension_m: float, specimen: str = "spring", repeats: int = 1):
        x = self._measurement_setting(extension_m, specimen, repeats)
        stiffness = {"spring": self._stiffness, "blank": 0.0, "reference": 100.0}[specimen]
        realized_x = self._gain * x
        if self._setup_sigma:
            realized_x *= 1 + self._setup_rng.gauss(0, self._setup_sigma)
        self.setups += 1
        readings = []
        for _ in range(repeats):
            y = stiffness * realized_x + self._offset + self._rng.gauss(0, self._sigma)
            if self._scenario in ("outliers", "mixed") and self._rng.random() < 0.15:
                y += self._rng.choice((-1, 1)) * self._rng.uniform(1, 3)
            readings.append(y)
        self.measurements += repeats
        conditions = {"nominal_extension_m": x, "actual_extension_m": None,
                      "specimen": specimen, "repeats": repeats,
                      "reference_stiffness_n_per_m": 100.0 if specimen == "reference" else None}
        conditions["setup_id"] = f"{self._apparatus_id}/setup-{self.setups}"
        conditions["replication_unit"] = "One preparation with repeated sensor readings"
        return Observation(
            data={"force_n": readings},
            source="spring.force_sensor",
            conditions=conditions,
            uncertainties=["Actual extension scale and force offset are not independently "
                           "verified by this measurement.",
                           "Repeated readings share the same specimen and instrument state."],
            sample_id=f"{self._apparatus_id}/{specimen}",
        )

    @staticmethod
    def validate_submission(submission):
        if submission.get("identifiable") is not True:
            raise ValueError("A concluded stiffness estimate must have identifiable=true")
        finite(submission.get("stiffness_n_per_m"), "stiffness", 0.001, 10000)

    def grade(self, conclusion):
        submission = conclusion.get("submission", {})
        status = conclusion.get("status")
        identifiable = submission.get("identifiable")
        if not self._reference_available:
            correct = status == "inconclusive" and identifiable is False
            return {"metric": "spring-v1", "identifiability_correct": correct,
                    "passed": correct, "measurements": self.measurements,
                    "note": "Only the product of stiffness and extension scale is observable."}
        if status != "concluded" or identifiable is not True:
            return {"metric": "spring-v1", "passed": False, "measurements": self.measurements,
                    "error": "No identifiable stiffness submitted"}
        try:
            estimate = finite(submission.get("stiffness_n_per_m"), "stiffness", 0.001, 10000)
        except ValueError as exc:
            return {"metric": "spring-v1", "passed": False, "error": str(exc),
                    "measurements": self.measurements}
        # Independent held-out extensions; no additional noisy acquisition or agent feedback.
        rng = random.Random(self._seed + 90000)
        xs = [rng.uniform(-0.05, 0.05) for _ in range(100)]
        truth = [self._stiffness * x for x in xs]
        errors = [(estimate * x - y) ** 2 for x, y in zip(xs, truth)]
        relative_rmse = math.sqrt(sum(errors) / sum(y * y for y in truth))
        return {"metric": "spring-v1", "relative_rmse": relative_rmse,
                "stiffness_relative_error": abs(estimate / self._stiffness - 1),
                "passed": relative_rmse < 0.05, "tolerance": 0.05,
                "held_out_points": len(xs), "measurements": self.measurements}

    def close(self):
        pass
