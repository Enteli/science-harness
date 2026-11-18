"""A managed trusted simulator process, not an OS sandbox."""
import json
import os
import select
import subprocess
import sys
from pathlib import Path

from ..books import encode


class PinnedWorker:
    def __init__(self, checkout, revision, module, configuration, *, python=None, network=True):
        if sys.platform == "win32":
            raise ValueError("Simulator workers currently require Linux or macOS")
        checkout = Path(checkout).resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise ValueError(f"Unsupported simulator revision {actual}; use {revision}")
        dirty = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if dirty:
            raise ValueError("Simulator tracked files are modified; use a clean pinned checkout")
        command = [str(python or sys.executable), "-m", module, str(checkout), encode(configuration)]
        env = None
        if not network:
            env = {k: v for k, v in os.environ.items() if k in {
                "PATH", "HOME", "TMPDIR", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled")
            if sys.platform == "darwin":
                command = ["/usr/bin/sandbox-exec", "-p",
                           "(version 1)(allow default)(deny network*)", *command]
        self.process = subprocess.Popen(
            command, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def request(self, payload):
        self.process.stdin.write(encode(payload) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 90)
        if not ready:
            self.close()
            raise TimeoutError("Simulator request exceeded 90 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Simulator worker exited; check stderr and install its optional dependencies")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"])
        return response["result"]

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
