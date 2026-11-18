"""Isolated numerics, native measurements, and replayable operation receipts."""
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from .physics_reference import digest, load_upstream, numerical_source
from .harness.io import atomic_json, encode

TIMES = [1, 2, 3, 4, 5]
NOISE_STD = 0.15089155792810338


class Journal:
    """Replay completed operations only; an ambiguous dispatch cannot be retried."""
    def __init__(self, path, mirror=None):
        self.path, self.index = Path(path), 0
        self.path.mkdir(parents=True, exist_ok=True)
        self.mirror = Path(mirror) if mirror is not None else None
        if self.mirror is not None:
            self.mirror.mkdir(parents=True, exist_ok=True)

    def call(self, kind, payload, operation):
        path = self.path / f"{self.index:04d}-{kind}.json"
        self.index += 1
        fingerprint = digest(encode(payload))
        mirror = self.mirror / path.name if self.mirror is not None else None
        if mirror is not None and mirror.exists():
            durable = json.loads(mirror.read_text())
            if durable["input_sha256"] != fingerprint:
                raise RuntimeError("Persistent trace would change an original operation")
            if not path.exists():
                if durable["status"] != "completed":
                    raise RuntimeError("Persistent trace has an unresolved operation; no retry")
                atomic_json(path, durable)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["input_sha256"] != fingerprint:
                raise RuntimeError("Replay would change an original operation")
            if saved["status"] != "completed":
                raise RuntimeError("Unresolved or failed operation; automatic retry forbidden")
            if mirror is not None:
                if mirror.exists() and json.loads(mirror.read_text())["status"] == "completed":
                    if json.loads(mirror.read_text()) != saved:
                        raise RuntimeError("Persistent and workspace operation records disagree")
                else:
                    atomic_json(mirror, saved)
            return copy.deepcopy(saved["result"])
        started = time.monotonic()
        record = {"kind": kind, "input_sha256": fingerprint, "input": payload, "status": "pending",
                  "started_at": datetime.now(timezone.utc).isoformat()}
        if mirror is not None:
            atomic_json(mirror, record)
        atomic_json(path, record)
        try:
            result = operation()
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                          elapsed_seconds=time.monotonic() - started)
            atomic_json(path, record)
            if mirror is not None:
                atomic_json(mirror, record)
            raise
        completed = {**record, "status": "completed", "result": result,
                     "elapsed_seconds": time.monotonic() - started}
        atomic_json(path, completed)
        if mirror is not None:
            atomic_json(mirror, completed)
        return result


def noise_seed(seed, index):
    return int(np.random.SeedSequence([seed, index, 7331]).generate_state(1)[0])


def proposal(seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 8841]))
    return {"p1": float(rng.uniform(.1, 10)), "p2": float(rng.uniform(.1, 10)),
            "pos2": rng.uniform(-10, 10, 2).tolist(),
            "velocity2": rng.uniform(-5, 5, 2).tolist(),
            "measurement_times": TIMES, "duration": 5.0}


def validate_design(design):
    if not isinstance(design, dict):
        raise ValueError("Experiment must be an object")
    required = {"p1", "p2", "pos2", "velocity2", "measurement_times"}
    if set(design) not in (required, required | {"duration"}):
        raise ValueError("Unexpected experiment fields")
    if (design["measurement_times"] != TIMES
            or any(type(t) not in (int, float) for t in design["measurement_times"])
            or type(design.get("duration", 5)) not in (int, float)
            or design.get("duration", 5) != 5):
        raise ValueError("Use exactly times 1..5 and duration 5")
    for name, low, high in (("p1", .1, 10), ("p2", .1, 10),
                            ("pos2", -10, 10), ("velocity2", -5, 5)):
        values = design[name] if name in ("pos2", "velocity2") else [design[name]]
        if not isinstance(values, list) or len(values) != (2 if name in ("pos2", "velocity2") else 1):
            raise ValueError(f"Invalid {name} shape")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not low <= v <= high for v in values):
            raise ValueError(f"Invalid {name} bounds")
    return {**copy.deepcopy(design), "duration": 5.0}


class NativeWorld:
    def __init__(self, checkout, world):
        load_upstream(checkout)
        from scienceagent import executor
        if world == "yukawa":
            self.executor = executor.NBodySimulationExecutor(
                operators=[{"type": "screening", "params": {"strength": 1.0, "screening_length": 2.0}}],
                integrator="dopri5", noise_std=0, noise_seed=0)
        elif world == "oscillator":
            self.executor = executor.NBodyOscillatorExecutor(
                integrator="dopri5", noise_std=0, noise_seed=0)
        elif world == "extra_dimensions":
            self.executor = executor.NBodyExtraDimensionsExecutor(
                integrator="dopri5", noise_std=0, noise_seed=0)
            self.forbid_coincident_launch = True
        else:
            raise ValueError("Only registered hosted worlds are supported")

    def measure(self, design, seed, clean=False):
        design = validate_design(design)
        self.validate_initial_conditions(design)
        self.executor._init_noise(NOISE_STD, seed)
        if clean:
            with self.executor.noise_disabled():
                result = self.executor.run([design])[0]
        else:
            result = self.executor.run([design])[0]
        if result["measurement_times"] != TIMES:
            raise RuntimeError("Native cadence changed")
        for key in ("pos1", "pos2", "velocity1", "velocity2"):
            values = np.asarray(result[key])
            if values.shape != (5, 2) or not np.isfinite(values).all():
                raise RuntimeError("Native execution produced invalid measurements")
        return {key: copy.deepcopy(result[key]) for key in
                ("measurement_times", "pos1", "pos2", "velocity1", "velocity2")}

    def validate_initial_conditions(self, design):
        if getattr(self, "forbid_coincident_launch", False) and design["pos2"] == [0, 0]:
            raise ValueError("Probe and source initial positions cannot coincide; native force direction is undefined")


class IsolatedNumerics:
    def __init__(self, checkout, python=None):
        self.python = Path(python or sys.executable).absolute()
        self.source = numerical_source(checkout)
        self.worker = Path(__file__).with_name("physics_worker.py").resolve()

    def profile(self):
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
            raise RuntimeError("Qualified macOS OS-isolation backend is required")
        roots = ["/System", "/usr", "/Library/Developer/CommandLineTools", "/Library/Apple",
                 str(self.python.parent.parent)]
        reads = " ".join(f"(subpath {json.dumps(path)})" for path in roots)
        return f"""(version 1)
(deny default)
(allow file-read-metadata)
(allow file-read* {reads} (literal "/") (literal {json.dumps(str(self.worker))})
 (literal "/dev/urandom") (literal "/dev/random") (subpath "/private/var/db/dyld"))
(allow process-exec (subpath "/usr") (subpath "/Library/Developer/CommandLineTools"))
(allow sysctl-read)
(allow mach-lookup)
"""

    def request(self, operation, source="", *, wall_limit=200, **data):
        if (isinstance(wall_limit, bool) or not isinstance(wall_limit, (int, float))
                or not math.isfinite(wall_limit) or not 0 < wall_limit <= 200):
            raise ValueError("Worker wall limit must be positive and no larger than200 seconds")
        fields = {
            "fit": {"observations"}, "final_fit": {"observations"},
            "predict": {"parameters", "queries"}, "probe": {"forbidden", "write_path", "port"},
        }
        if operation not in fields or set(data) != fields[operation]:
            raise RuntimeError(
                "Numerical request contract violation: fit and held-out data must remain separated")
        payload = {"operation": operation, "source": source, "numerical_source": self.source,
                   "numerical_sha256": digest(self.source), **data}
        environment = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-p", self.profile(), str(self.python),
             "-I", "-B", str(self.worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd="/", env=environment, start_new_session=True)
        deadline = time.monotonic() + wall_limit
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(process.communicate, encode(payload))
            while True:
                try:
                    output, error = future.result(timeout=.1)
                    break
                except FutureTimeout:
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                         capture_output=True, text=True, timeout=2)
                    over_memory = rss.returncode == 0 and rss.stdout.strip() and int(rss.stdout) > 512 * 1024
                    if time.monotonic() > deadline or over_memory:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        future.result()
                        return {"ok": False, "failure_kind": "candidate_resource_limit",
                                "error": "Isolated worker exceeded its declared resource limit",
                                "loss_before": None, "loss_after": None, "fitted_params": {},
                                "declared_params": {}, "n_training": 0}
        if process.returncode == -signal.SIGXCPU:
            return {"ok": False, "failure_kind": "candidate_resource_limit",
                    "error_kind": "SIGXCPU", "returncode": process.returncode,
                    "error": "Candidate worker exited on the CPU-resource-limit signal",
                    "loss_before": None, "loss_after": None, "fitted_params": {},
                    "declared_params": {}, "n_training": 0}
        if process.returncode:
            raise RuntimeError(
                f"Numerical worker infrastructure failure (returncode={process.returncode}): {error[-1200:]}")
        return json.loads(output)

    def verify_boundary(self, forbidden):
        forbidden = Path(forbidden).resolve()
        if not forbidden.is_file():
            raise ValueError("Read probe must target an existing forbidden file")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            result = self.request("probe", forbidden=str(forbidden),
                                  write_path=str(forbidden.with_name("forbidden-worker-write")),
                                  port=listener.getsockname()[1])
        if result != {"read_denied": True, "write_denied": True, "network_denied": True}:
            raise RuntimeError(f"Isolation qualification failed: {result}")
        return result
