"""Common LLM client protocol for provider-agnostic usage."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the contract for LLM providers.

    Uses structural typing — any class with matching method signatures
    satisfies this protocol without explicit inheritance.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]: ...

    def extract_structured_data(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]: ...

    def is_available(self) -> bool: ...


def log_llm_request(
    logger: logging.Logger,
    messages: list[dict[str, str]],
    **extra: object,
) -> None:
    """Log an [LLM REQUEST] line with the last message's role and length."""
    if not messages:
        return
    last = messages[-1]
    parts = f"Role: {last.get('role')}, Content length: {len(last.get('content', ''))} chars"
    for key, value in extra.items():
        parts += f", {key}: {value}"
    logger.info("[LLM REQUEST] %s", parts)


def log_llm_response(
    logger: logging.Logger,
    model: str,
    response_text: str,
    **extra: object,
) -> None:
    """Log an [LLM RESPONSE] line with model and response length."""
    parts = f"Model: {model}, Response length: {len(response_text)} chars"
    for key, value in extra.items():
        parts += f", {key}: {value}"
    logger.info("[LLM RESPONSE] %s", parts)


def strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences that LLMs sometimes wrap around JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
