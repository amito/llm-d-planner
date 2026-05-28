"""Ollama client wrapper for LLM interactions."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama library not available. LLM features will be limited.")


from planner.llm.client import log_llm_request, log_llm_response

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM service."""

    def __init__(self, model: str | None = None, host: str | None = None):
        """
        Initialize Ollama client.

        Args:
            model: Model name to use. Falls back to LLM_MODEL env var,
                   then "qwen2.5:7b".
            host: Optional Ollama host URL. Falls back to OLLAMA_HOST env var,
                  then localhost:11434.
        """
        default_model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.model: str = model if model else default_model
        self.host = host or os.getenv("OLLAMA_HOST")

        self._client: ollama.Client | None = None
        if OLLAMA_AVAILABLE:
            client_kwargs = {"host": self.host} if self.host else {}
            self._client = ollama.Client(**client_kwargs)
        else:
            logger.error("Ollama library not installed. Install with: pip install ollama")

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Send chat messages to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Response dict with 'message' containing 'content'
        """
        if not OLLAMA_AVAILABLE or not self._client:
            raise RuntimeError("Ollama library not available")

        try:
            log_llm_request(logger, messages)
            if messages:
                logger.debug(f"[LLM PROMPT] {messages[-1].get('content', '')[:500]}...")

            fmt: Literal["", "json"] = "json" if format_json else ""
            response = self._client.chat(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                format=fmt,
                options={"temperature": temperature},
            )

            response_content = response.get("message", {}).get("content", "")
            log_llm_response(logger, self.model, response_content)
            logger.debug("[LLM RESPONSE CONTENT] %s", response_content)

            return dict(response)

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input prompt string
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        logger.info(
            f"[LLM GENERATE] Prompt length: {len(prompt)} chars, JSON format: {format_json}, Temperature: {temperature}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, format_json=format_json, temperature=temperature)
        return str(response["message"]["content"])

    def extract_structured_data(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Extract structured data from prompt using JSON format.

        Args:
            prompt: Input prompt (should include schema and instructions)
            temperature: Lower temperature for more consistent extraction

        Returns:
            Parsed JSON dict
        """
        response_text = self.generate_completion(prompt, format_json=True, temperature=temperature)

        try:
            result: dict[str, Any] = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            logger.error(f"JSON error: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        if not self._client:
            return False

        try:
            # Try to list models to verify connection
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            return False

    def ensure_model_pulled(self) -> bool:
        """
        Ensure the configured model is pulled locally.

        Returns:
            True if model is available, False otherwise
        """
        if not self._client:
            return False

        try:
            models = self._client.list()
            model_names = [m["name"] for m in models.get("models", [])]

            if self.model not in model_names:
                logger.info(f"Pulling model {self.model}...")
                self._client.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to pull model {self.model}: {e}")
            return False
