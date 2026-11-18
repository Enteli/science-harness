"""Explicit, versioned policy bindings; unsupported combinations are rejected."""
from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class Arm:
    name: str
    memory: str = "inline"
    advisor: str = "off"
    selection: str = "active"
    profile: str = "memory-advisor-v1"

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or "name" not in value or set(value) - {
                "name", "memory", "advisor", "selection", "profile"}:
            raise ValueError("An arm requires name and only declared policy fields")
        arm = cls(**value)
        if not isinstance(arm.name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", arm.name):
            raise ValueError("Arm names must be distinct lowercase identifiers")
        if arm.memory not in ("none", "inline", "external"):
            raise ValueError("Memory must be none, inline or external")
        if arm.advisor not in ("off", "design"):
            raise ValueError("Advisor must be off or design")
        if arm.selection not in ("active", "random"):
            raise ValueError("Selection must be active or random")
        if not isinstance(arm.profile, str) or not re.fullmatch(r"[a-z][a-z0-9-]*-v[1-9][0-9]*", arm.profile):
            raise ValueError("Policy profiles need a versioned identifier, such as memory-advisor-v1")
        return arm

    def record(self):
        return asdict(self)
