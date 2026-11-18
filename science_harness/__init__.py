"""A small, uncertainty-aware scientist harness."""
from .agent import Scientist
from .books import BookStore
from .llm import LLM, OpenAICompatLLM, ScriptedLLM
from .tools import AcquisitionContract, Observation, Tool, ToolRegistry

__all__ = [
    "Scientist", "BookStore", "LLM", "OpenAICompatLLM", "ScriptedLLM",
    "Observation", "Tool", "ToolRegistry", "AcquisitionContract",
]
