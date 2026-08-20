"""Factory for creating LLM clients based on configuration."""

from __future__ import annotations

import logging
import os

from planner.llm.client import LLMClient

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = ("ollama", "vertex", "openai")


def create_llm_client(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client.

    Args:
        provider: LLM provider name ("ollama", "openai", "vertex").
                  Falls back to LLM_PROVIDER env var, then "ollama".
        api_key: API key (passed to client; falls back to provider-specific env var).
        base_url: Base URL override (OpenAI-compatible endpoints, Ollama host).
        model: Model name override.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")
    provider = provider.lower()

    if provider == "ollama":
        try:
            from planner.llm.ollama_client import OllamaClient

            logger.info("Using Ollama LLM provider")
            return OllamaClient(model=model, host=base_url)
        except ImportError as e:
            raise ImportError(
                "Ollama provider requires ollama. Install with: pip install llm-d-planner[llm]"
            ) from e

    if provider == "vertex":
        try:
            from planner.llm.vertex_client import VertexClient

            logger.info("Using Vertex AI LLM provider")
            return VertexClient(model=model)
        except ImportError as e:
            raise ImportError(
                "Vertex AI provider requires anthropic[vertex]. "
                "Install with: pip install llm-d-planner[vertex]"
            ) from e

    if provider == "openai":
        try:
            from planner.llm.openai_client import OpenAIClient

            logger.info("Using OpenAI-compatible LLM provider")
            return OpenAIClient(api_key=api_key, base_url=base_url, model=model)
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires openai. Install with: pip install llm-d-planner[openai]"
            ) from e

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. Valid options: {', '.join(_VALID_PROVIDERS)}"
    )
