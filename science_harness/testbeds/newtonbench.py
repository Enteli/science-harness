"""Pinned NewtonBench Hooke-law adapter, isolated from the agent process's imports."""
import uuid

from ..tools import Observation, Tool, ToolRegistry
from .spring import finite
from .worker import PinnedWorker


NEWTONBENCH_REVISION = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
SYSTEMS = ("vanilla_equation", "simple_system", "complex_system")


class NewtonBench:
    def __init__(self, checkout, difficulty="easy", system="vanilla_equation",
                 law_version="v0", noise=0.01, seed=0):
        if difficulty not in ("easy", "medium", "hard") or system not in SYSTEMS:
            raise ValueError("Invalid NewtonBench difficulty or system")
        if law_version not in ("v0", "v1", "v2"):
            raise ValueError("Choose a fixed law version: v0, v1 or v2")
        noise = finite(noise, "noise", 0, 1)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a 32-bit nonnegative integer")
        self.manifest = {
            "testbed": "newtonbench", "module": "m9_hooke_law", "revision": NEWTONBENCH_REVISION,
            "difficulty": difficulty, "system": system, "law_version": law_version,
            "noise": noise, "seed": seed,
            "apparatus_id": f"newton-{uuid.uuid4().hex}",
        }
        self.measurements = 0
        self.worker = PinnedWorker(checkout, NEWTONBENCH_REVISION,
                                   "science_harness.testbeds.newton_worker", self.manifest)
        try:
            self._request({"command": "describe"})
        except BaseException:
            self.close()
            raise
        readout = {
            "vanilla_equation": (
                "The sensor measures U(x) directly in joules. Mass m is accepted for "
                "interface consistency but is unused in this direct-energy configuration."
            ),
            "simple_system": (
                "The sensor reports U(x)*(1 + 0.4*x/m) in joules. This is the benchmark's "
                "declared apparatus transformation, not the unknown material law."
            ),
            "complex_system": (
                "The sensor reports sqrt(2*U(x)/m)*exp(-x/ell) in metres/second. "
                "ell is an unknown positive, fixed apparatus length scale. Estimate "
                "its effect rather than treating the velocity readout as stored energy."
            ),
        }[system]
        self.context = (
            f"Current apparatus ID: {self.manifest['apparatus_id']}. Prior observations "
            "from other apparatus IDs do not establish this apparatus's law.\n"
            "Discover the unknown stored-energy function U(x). It may differ from familiar "
            "physics; do not assume a quadratic or a single power law. The underlying "
            "function depends on x; any mass effect is in the stated observation model. "
            + readout +
            " Readings may contain random measurement error. The world and instrument "
            "parameters are fixed throughout the run. Use the run_experiment tool with "
            "ONE object containing scalar x and m and integer repeats, not an experiment "
            "array or a list of x values. "
            "Bounds: 0.001 <= x <= 1; 0.1 <= m <= 10; 1 <= repeats <= 21. "
            "Finish submission: {\"expression\": \"an arithmetic expression in x\"} describing "
            "the underlying potential energy, not the apparatus output. Allowed: numeric "
            "constants, x, pi, e, + - * / ** and sqrt, exp, log, sin, cos, abs. "
            "No XML, imports, assignments, functions or other Python. Final expression must "
            "describe U(x), undoing the known readout transformation and any inferred "
            "apparatus loss. Experimental error need not be a property of the material law."
        )
        self.tools = ToolRegistry([Tool(
            "run_experiment", "Query the fixed Hooke-law apparatus; args: x, m=1.0, repeats=1.",
            self.measure, readout_fields=("readings",),
        )], analysis_views={"response": {
            "source_tool": "run_experiment",
            "description": "Measured apparatus response versus displacement and mass, not necessarily energy.",
            "columns": {"x": "conditions.x_m", "m": "conditions.m_kg", "y": "data.readings"},
        }})

    def _request(self, payload):
        return self.worker.request(payload)

    def measure(self, x: float, m: float = 1.0, repeats: int = 1):
        x = finite(x, "x", 0.001, 1)
        m = finite(m, "m", 0.1, 10)
        if type(repeats) is not int or not 1 <= repeats <= 21:
            raise ValueError("repeats must be an integer between 1 and 21")
        result = self._request({"command": "measure", "x": x, "m": m, "repeats": repeats})
        self.measurements += repeats
        units = {"vanilla_equation": "J", "simple_system": "J", "complex_system": "m/s"}
        return Observation(
            data={"readings": result, "unit": units[self.manifest["system"]]},
            source="NewtonBench.m9_hooke_law",
            conditions={"x_m": x, "m_kg": m, "repeats": repeats,
                        "system": self.manifest["system"],
                        "upstream_revision": NEWTONBENCH_REVISION},
            uncertainties=["Noisy readings and system-specific transformations are distinct "
                           "from the unknown underlying law."],
            sample_id=self.manifest["apparatus_id"],
        )

    def grade(self, conclusion):
        identifiability = ("structurally_confounded_unrestricted_energy"
                           if self.manifest["system"] == "complex_system" else "not_assessed")
        if conclusion.get("status") != "concluded":
            return {"metric": "custom-newton-predictive-v1", "passed": False,
                    "error": "No conclusive submission", "measurements": self.measurements,
                    "identifiability_target": identifiability}
        expression = conclusion.get("submission", {}).get("expression")
        result = self._request({"command": "grade", "expression": expression})
        return dict(result, measurements=self.measurements, identifiability_target=identifiability)

    @staticmethod
    def validate_submission(submission):
        from .expression import Expression

        Expression(submission.get("expression"))

    def close(self):
        self.worker.close()
