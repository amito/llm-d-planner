"""LLM Backend — pluggable provider interface for intent extraction and conversation."""

from planner.llm.client import LLMClient
from planner.llm.factory import create_llm_client
from planner.llm.ollama_client import OllamaClient

__all__ = ["LLMClient", "OllamaClient", "create_llm_client"]


def __getattr__(name: str):  # noqa: N807
    if name == "VertexClient":
        from planner.llm.vertex_client import VertexClient

        return VertexClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
