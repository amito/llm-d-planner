"""Factory for creating LLM clients based on configuration."""

from __future__ import annotations

import logging
import os

from planner.llm.client import LLMClient
from planner.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = ("ollama", "vertex")


def create_llm_client() -> LLMClient:
    """Create an LLM client based on the LLM_PROVIDER environment variable.

    Returns OllamaClient by default; set LLM_PROVIDER=vertex for Vertex AI.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        logger.info("Using Ollama LLM provider")
        return OllamaClient()

    if provider == "vertex":
        from planner.llm.vertex_client import VertexClient

        logger.info("Using Vertex AI LLM provider")
        return VertexClient()

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. Valid options: {', '.join(_VALID_PROVIDERS)}"
    )
