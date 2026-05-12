"""Common LLM client protocol for provider-agnostic usage."""

from __future__ import annotations

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
