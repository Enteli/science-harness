"""Verified upstream code for the controlled, trajectory-interface comparison."""
import ast
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

REVISION = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
BOUNDARY_REVISION = "6fd13ec46187bb6510c6dde72eac8dcd4e0172f5"
TRANSPORT_REVISION = "71ed448e9d0c764d5831820cb85a2d124f5355e2"
NUMERICAL_NAMES = {
    "MAX_FIT_PARAMETERS", "FIT_MAXITER", "FIT_TIME_BUDGET_S", "LAW_CALL_TIMEOUT_S",
    "FIT_MAX_TRAJECTORIES", "FIT_MAX_TIMES_PER_TRAJ",
    "_FitTimeBudgetExceeded", "_LawCallTimeout", "_wrap_with_timeout",
    "clean_law_source", "_compile_law", "_compile_fit_parameters",
    "_subsample_training", "_two_particle_loss", "_validate_fit_spec",
    "_fit_law_parameters",
}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verified_source(checkout, relative):
    checkout = Path(checkout).resolve()
    expected = subprocess.check_output(
        ["git", "-C", str(checkout), "show", f"{REVISION}:{relative}"])
    actual = (checkout / relative).read_bytes()
    if actual != expected:
        raise RuntimeError(f"Pinned upstream source changed: {relative}")
    return actual.decode("utf-8")


def numerical_source(checkout):
    """Extract unchanged function bodies, not upstream simulator/grading imports."""
    source = verified_source(checkout, "ScienceAgent/scienceagent/evaluator.py")
    selected, found = [], set()
    for node in ast.parse(source).body:
        names = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & NUMERICAL_NAMES:
            selected.append(ast.get_source_segment(source, node))
            found.update(names & NUMERICAL_NAMES)
    if found != NUMERICAL_NAMES:
        raise RuntimeError("Pinned numerical function inventory changed")
    return "\n\n".join(selected)


def load_upstream(checkout):
    checkout = Path(checkout).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise RuntimeError("Wrong DiscoverPhysics revision")
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"],
                   check=True, capture_output=True)
    expected = checkout / "ScienceAgent"
    if "scienceagent" in sys.modules:
        actual = Path(sys.modules["scienceagent"].__file__).resolve().parent.parent
        if actual != expected:
            raise RuntimeError("A different upstream checkout is already imported")
    else:
        sys.path.insert(0, str(expected))
    return importlib.import_module("scienceagent.agent")


def provenance(checkout):
    files = [
        "ScienceAgent/scienceagent/agent.py",
        "ScienceAgent/scienceagent/evaluator.py",
        "ScienceAgent/scienceagent/executor.py",
        "PhysicsSchool/prompts/_template_interactive.md",
        "PhysicsSchool/prompts/2particle_instructions.md",
    ]
    return {
        "upstream_revision": REVISION,
        "boundary_reference_revision": BOUNDARY_REVISION,
        "shared_transport_reference_revision": TRANSPORT_REVISION,
        "notebook_prompt_policy": "explicit-final-note-v2",
        "upstream_files": {name: digest(verified_source(checkout, name)) for name in files},
        "numerical_source_sha256": digest(numerical_source(checkout)),
        "implementation_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("physics_*"))
            if path.suffix in (".py", ".mjs")
        },
        "variant": "native-trajectory-notebook-comparison-v1",
    }
