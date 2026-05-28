"""OpenAI-compatible client for any service implementing the OpenAI API."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from planner.llm.client import log_llm_request, log_llm_response, strip_markdown_fences

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"


class OpenAIClient:
    """LLM client using any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model: str = model or os.getenv("LLM_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.effective_url: str = self.base_url or "https://api.openai.com/v1"
        self._client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAIClient initialized: model=%s, base_url=%s",
            self.model,
            self.effective_url,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Send chat messages to an OpenAI-compatible endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            format_json: If True, request JSON formatted response.
            temperature: Sampling temperature (0.0 to 1.0).

        Returns:
            Response dict with 'message' containing 'content'.
        """
        log_llm_request(logger, messages, base_url=self.effective_url)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if format_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        if not response.choices:
            logger.warning("LLM returned no choices in response")
            response_text = ""
        else:
            response_text = response.choices[0].message.content or ""
        log_llm_response(logger, self.model, response_text, base_url=self.effective_url)

        return {"message": {"content": response_text}}

    def extract_structured_data(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Extract structured data from prompt, expecting JSON response.

        Args:
            prompt: Input prompt (should include schema and instructions).
            temperature: Lower temperature for more consistent extraction.

        Returns:
            Parsed JSON dict.
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.chat(messages, format_json=True, temperature=temperature)
        response_text = result["message"]["content"]
        stripped = strip_markdown_fences(response_text)

        try:
            parsed: dict[str, Any] = json.loads(stripped)
            return parsed
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s", response_text)
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    def is_available(self) -> bool:
        """Check if the OpenAI-compatible service is reachable."""
        try:
            self._client.models.list()
            return True
        except Exception as e:
            logger.warning("OpenAI-compatible service not available: %s", e)
            return False
