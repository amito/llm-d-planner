"""Vertex AI client for Claude on Google Cloud."""

from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
from typing import Any, cast

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 4096


class VertexClient:
    """LLM client using Claude on Vertex AI via AnthropicVertex SDK."""

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        model: str | None = None,
    ):
        self.project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
        if not self.project_id:
            raise ValueError("VERTEX_PROJECT_ID environment variable is required for Vertex AI")

        self.region: str = region or os.getenv("VERTEX_REGION", "global") or "global"
        self.model: str = model or os.getenv("VERTEX_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL

        self._setup_credentials()

        from anthropic import AnthropicVertex

        self._client = AnthropicVertex(
            project_id=self.project_id,
            region=self.region,
        )
        logger.info(
            "VertexClient initialized: project=%s, region=%s, model=%s",
            self.project_id,
            self.region,
            self.model,
        )

    def _setup_credentials(self) -> None:
        """Write GCP_CREDENTIALS_JSON env var to a temp file for ADC discovery."""
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return
        creds_json = os.getenv("GCP_CREDENTIALS_JSON")
        if not creds_json:
            return
        fd, path = tempfile.mkstemp(suffix=".json", prefix="gcp-creds-")
        with os.fdopen(fd, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        atexit.register(os.unlink, path)
        logger.info("Wrote GCP credentials to %s", path)

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Send chat messages to Claude on Vertex AI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            format_json: Ignored (Anthropic API doesn't have a format flag).
            temperature: Sampling temperature (0.0 to 1.0).

        Returns:
            Response dict with 'message' containing 'content'.
        """
        if messages:
            last_msg = messages[-1]
            logger.info(
                "[LLM REQUEST] Role: %s, Content length: %d chars",
                last_msg.get("role"),
                len(last_msg.get("content", "")),
            )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=cast(Any, messages),
            temperature=temperature,
        )

        first_block = response.content[0]
        response_text: str = (
            first_block.text  # type: ignore[union-attr]
            if hasattr(first_block, "text")
            else str(first_block)
        )
        logger.info(
            "[LLM RESPONSE] Model: %s, Response length: %d chars",
            self.model,
            len(response_text),
        )

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

        # Strip markdown code fences (```json ... ```) that Claude often wraps around JSON
        stripped = response_text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```json) and last line (```)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        try:
            parsed: dict[str, Any] = json.loads(stripped)
            return parsed
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s", response_text)
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    def is_available(self) -> bool:
        """Check if Vertex AI Claude service is reachable."""
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.warning("Vertex AI service not available: %s", e)
            return False
