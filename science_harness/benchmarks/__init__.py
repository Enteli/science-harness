"""Lazy benchmark registry. Register trusted adapters, not model-controlled code."""
from dataclasses import asdict, dataclass
from importlib import import_module
import re
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    protocol: str
    worlds: tuple
    providers: tuple
    profiles: tuple
    runtime: str
    description: str

    def __post_init__(self):
        names = (self.name, self.protocol, *self.worlds, *self.providers, *self.profiles)
        if any(not isinstance(n, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", n) for n in names):
            raise ValueError("Benchmark identifiers must be safe lowercase names, not paths")
        if not self.worlds or not self.providers or not self.profiles:
            raise ValueError("Declare benchmark worlds, providers and profiles explicitly")

    def record(self):
        return asdict(self)


class BenchmarkAdapter(Protocol):
    @staticmethod
    def validate(config): ...

    def prepare(self, config, output, approved_credits): ...

    def run_episode(self, cell, directory): ...

    def close(self): ...


_REGISTRY = {}


def register(spec, loader):
    if not isinstance(spec, BenchmarkSpec) or not callable(loader):
        raise ValueError("A benchmark needs a spec and trusted adapter loader")
    if spec.name in _REGISTRY:
        raise ValueError(f"Benchmark already registered: {spec.name}")
    _REGISTRY[spec.name] = (spec, loader)


def available():
    return [spec.record() for spec, _ in _REGISTRY.values()]


def get(name):
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError(f"Unknown benchmark {name!r}; use 'science-harness bench list'")
    return _REGISTRY[name]


def _adapter(module, name):
    return lambda: getattr(import_module(module, package=__name__), name)


register(BenchmarkSpec(
    "discoverphysics", "native-trajectory-v1", ("yukawa", "oscillator", "extra_dimensions"),
    ("copilot",), ("legacy-v1", "memory-advisor-v1"),
    "Qualified macOS sandbox, pinned upstream checkout, Node SDK and Copilot CLI",
    "Native Python trajectory discovery; not the older expression adapter."),
    _adapter(".discoverphysics", "DiscoverPhysicsAdapter"))
register(BenchmarkSpec(
    "spring", "scientist-reference-v1",
    ("clean", "noise", "offset", "outliers", "gain", "unidentifiable", "mixed",
     "noise_control", "measurement_noise", "configuration_noise", "combined_noise"),
    ("baseline",), ("reference-v1",), "Python; no API or external checkout",
    "Offline calibration reference through the existing generic Scientist; not an LLM ablation."),
    _adapter(".reference", "SpringAdapter"))
register(BenchmarkSpec(
    "newtonbench", "scientist-hooke-reference-v1", ("vanilla_equation",),
    ("baseline",), ("reference-v1", "notebook-reference-v1"),
    "Pinned NewtonBench checkout and its worker dependencies",
    "Easy vanilla Hooke-law reference, including explicit notebook read/write; not an LLM study or full suite."),
    _adapter(".reference", "NewtonBenchAdapter"))
register(BenchmarkSpec(
    "newtonbench-discovery", "scientist-hooke-discovery-v1", ("vanilla_equation",),
    ("copilot",), ("notebook-discovery-v1",),
    "Pinned NewtonBench checkout, trusted analysis extra, Node SDK and Copilot CLI",
    "LLM-selected experiments and safe expression discovery; private predictive evaluation, not the reference policy."),
    _adapter(".discovery", "NewtonDiscoveryAdapter"))
register(BenchmarkSpec(
    "made", "scientist-made-analytic-v1", ("lj-crystal",),
    ("copilot",), ("notebook-discovery-v1",),
    "Pinned MADE checkout and isolated Python 3.12 analytic worker; no model weights or API key",
    "Controlled CPU Lennard-Jones crystal optimization through MADE's real analytic oracle; not the paper's materials benchmark."),
    _adapter(".discovery", "MadeDiscoveryAdapter"))
