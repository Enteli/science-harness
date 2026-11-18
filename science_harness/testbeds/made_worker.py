"""Trusted MADE worker. Private numerical reference search runs only after finish."""
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys


def load_upstream(checkout):
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    # The pinned package initializers eagerly import unrelated generators and GPU
    # stacks. Load only these fixed source namespaces; oracle modules stay unmodified.
    for name in ("made", "made.data", "made.envs", "made.oracles",
                 "made.oracles.classic", "made.utils"):
        directory = Path(checkout) / "src" / Path(*name.split("."))
        if not directory.is_dir() or name in sys.modules:
            raise RuntimeError("The fixed MADE source namespace is missing or already loaded")
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, package)


def main():
    if sys.version_info < (3, 12):
        raise RuntimeError("The pinned MADE integration requires Python 3.12 or newer")
    checkout, config = sys.argv[1], json.loads(sys.argv[2])
    load_upstream(checkout)
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from ase.build import bulk
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.io.ase import AseAtomsAdaptor
        from scipy.optimize import minimize_scalar
        from made.data.chemical_system import PhaseDiagramDataset
        from made.envs.base import Environment
        from made.oracles.classic.analytic import AnalyticOracle

        class LocalDataset(PhaseDiagramDataset):
            def _load_dataset(self):
                return PhaseDiagram([PDEntry("Ar", 0.0)])

        class CellEnvironment(Environment):
            def __init__(self):
                oracle = AnalyticOracle(model_name="lj", relax=False, num_workers=1)
                oracle.set_element_reference_energies({"Ar": 0.0})
                super().__init__(LocalDataset(["Ar"]), oracle, config["query_budget"])
                self.adaptor = AseAtomsAdaptor()
                self.evaluation_calls = 0

            def reset(self):
                self.query_count, self.history, self.done = 0, [], False
                return self.get_state()

            def evaluate(self, prototype, a, private=False):
                if prototype not in ("sc", "bcc", "fcc") or type(a) not in (int, float):
                    raise ValueError("Invalid prototype or lattice length")
                if not math.isfinite(a) or not 1.2 <= a <= 3.0:
                    raise ValueError("Lattice length is outside [1.2,3.0]")
                structure = self.adaptor.get_structure(bulk("Ar", prototype, a=float(a), cubic=True))
                result = self.oracle.evaluate(structure)
                if not math.isfinite(result["energy_per_atom"]):
                    raise ValueError("Nonfinite oracle energy")
                if private:
                    self.evaluation_calls += 1
                return result

            def step(self, query):
                if self.is_done():
                    raise ValueError("MADE agent-facing query budget exhausted")
                self.query_count += 1
                result = self.evaluate(query["prototype"], query["a"])
                self.history.append({**query, **result})
                return result

            def get_state(self):
                return {"queries": self.query_count, "budget": self.budget}

            def grade(self, query):
                base = {"metric": "controlled-made-lj-regret-v1",
                        "official_paper_benchmark": False, "tolerance": 0.01,
                        "agent_oracle_calls": self.query_count}
                if not any(row["prototype"] == query["prototype"]
                           and abs(row["a"] - query["a"]) < 1e-12 for row in self.history):
                    return dict(base, passed=False, error="Submitted candidate was never measured")
                references = []
                for prototype in ("sc", "bcc", "fcc"):
                    f = lambda a: self.evaluate(prototype, float(a), private=True)["energy_per_atom"]
                    xs = np.linspace(1.2, 3.0, 101)
                    ys = [f(x) for x in xs]
                    i = int(np.argmin(ys))
                    fit = minimize_scalar(f, bounds=(float(xs[max(0, i-1)]),
                                                    float(xs[min(100, i+1)])), method="bounded",
                                          options={"xatol": 1e-9, "maxiter": 60})
                    if not fit.success:
                        raise RuntimeError("Private MADE numerical reference search failed")
                    references.append({"prototype": prototype, "a": float(fit.x),
                                       "energy_per_atom": float(fit.fun)})
                reference = min(references, key=lambda row: row["energy_per_atom"])
                final = self.evaluate(query["prototype"], query["a"], private=True)["energy_per_atom"]
                baseline = self.evaluate("sc", 3.0, private=True)["energy_per_atom"]
                regret = max(0.0, final - reference["energy_per_atom"]) / max(
                    abs(reference["energy_per_atom"]), 1e-12)
                return dict(base, passed=bool(regret <= 0.01 and final < baseline),
                            relative_regret=regret, final_energy_per_atom=final,
                            baseline_energy_per_atom=baseline, reference=reference,
                            private_reference_oracle_calls=self.evaluation_calls)

        world = CellEnvironment()
        world.reset()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["command"] == "describe":
                    result = {"ready": True, "oracle": "real-MADE-AnalyticOracle",
                              "environment": "registered-custom-Environment-subclass",
                              "model_weights": False, "external_dataset": False,
                              "agent_oracle_calls": world.query_count,
                              "private_reference_oracle_calls": world.evaluation_calls,
                              "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "imports": "fixed-upstream-source-namespaces-no-registry-initializers"}
                elif request["command"] == "measure":
                    result = world.step({k: request[k] for k in ("prototype", "a")})
                elif request["command"] == "grade":
                    result = world.grade({k: request[k] for k in ("prototype", "a")})
                else:
                    raise ValueError("Unknown worker command")
            response = {"result": result}
        except (ValueError, KeyError, TypeError, ArithmeticError, RuntimeError) as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
