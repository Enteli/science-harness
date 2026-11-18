"""Controlled CPU cell optimization using MADE's actual analytic oracle."""
from pathlib import Path
import hashlib
import uuid

from ..tools import Observation, Tool, ToolRegistry
from .spring import finite
from .worker import PinnedWorker


MADE_REVISION = "bffda0c9a1fa1904b35c0ef67b455881c975df86"
PROTOTYPES = ("sc", "bcc", "fcc")


def candidate(prototype, a):
    if prototype not in PROTOTYPES:
        raise ValueError("Choose sc, bcc or fcc")
    return {"prototype": prototype, "a": finite(a, "a", 1.2, 3.0)}


class Made:
    def __init__(self, checkout, python, seed=0, budget=12):
        # Resolving a venv's interpreter symlink loses its site-packages.
        executable = Path(python).absolute()
        if not executable.is_file():
            raise ValueError("Prepare the isolated Python 3.12 MADE environment; see docs/made.md")
        self.manifest = {
            "testbed": "made", "revision": MADE_REVISION, "seed": seed,
            "task": "controlled-lj-crystal-v1", "query_budget": budget,
            "apparatus_id": f"made-{uuid.uuid4().hex}",
            "official_paper_benchmark": False, "oracle": "upstream-analytic-lj",
        }
        self.worker = PinnedWorker(checkout, MADE_REVISION, "science_harness.testbeds.made_worker",
                                   self.manifest, python=executable, network=False)
        try:
            ready = self.worker.request({"command": "describe"})
            expected = hashlib.sha256(Path(__file__).with_name("made_worker.py").read_bytes()).hexdigest()
            if ready.get("worker_source_sha256") != expected:
                raise RuntimeError("The isolated MADE environment has a different harness worker version")
        except BaseException:
            self.close()
            raise
        self.goal = (
            "Find a low-potential-energy periodic crystal cell using your chosen oracle queries. "
            "Compare candidate structures and cell sizes, refine a promising region, and submit "
            "the best candidate you have actually measured. Use the external lab notebook to "
            "retain hypotheses, observations and tradeoffs. This is controlled simulated "
            "optimization, not discovery of a new real material.")
        self.context = (
            "The instrument is a fixed CPU atomistic energy oracle in a controlled MADE environment. "
            "One species is labelled Ar, but the potential is not calibrated to real argon. "
            "Choose prototype sc (simple cubic), bcc (body-centred cubic), or fcc "
            "(face-centred cubic), and a cubic lattice constant a in [1.2,3.0]. "
            "The oracle reports potential energy per atom; lower is better. Atomic coordinates "
            "are fixed by each prototype; no automatic geometry relaxation occurs. "
            "Use run_experiment with one object {\"prototype\":\"fcc\",\"a\":2.0}. "
            "Each call is one measurement, including repeated candidates. "
            "The final submission must be {\"prototype\":\"...\",\"a\":number} for a candidate "
            "you actually measured. No formula, hidden optimum or scoring feedback is available. "
            "Compare energy per atom, not total energy across cells with different atom counts.")
        self.tools = ToolRegistry([
            Tool("run_experiment", "Evaluate one bounded periodic crystal cell using the MADE oracle.",
                 self.measure, readout_fields=("readings",)),
        ], analysis_views={"response": {
            "source_tool": "run_experiment", "description": "Energy per atom versus cell length.",
            "columns": {"x": "conditions.a", "y": "data.readings"},
        }})

    def measure(self, prototype: str, a: float):
        query = candidate(prototype, a)
        result = self.worker.request({"command": "measure", **query})
        return Observation(data={"readings": [result["energy_per_atom"]],
                                 "natoms": result["natoms"], "unit": "eV/atom"},
                           source="MADE.AnalyticOracle", conditions=query,
                           sample_id=self.manifest["apparatus_id"],
                           uncertainties=["Deterministic analytic potential; not a real-material oracle."])

    def grade(self, conclusion):
        if conclusion.get("status") != "concluded":
            return {"metric": "controlled-made-lj-regret-v1", "passed": False,
                    "official_paper_benchmark": False, "error": "No conclusive submission"}
        submission = conclusion.get("submission", {})
        self.validate_submission(submission)
        return self.worker.request({"command": "grade", **submission})

    @staticmethod
    def validate_submission(submission):
        if not isinstance(submission, dict) or set(submission) != {"prototype", "a"}:
            raise ValueError("Submit exactly prototype and a")
        candidate(**submission)

    def close(self):
        self.worker.close()
