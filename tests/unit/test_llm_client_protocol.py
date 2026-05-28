"""Tests that LLM clients structurally satisfy the LLMClient Protocol."""

import pytest

from planner.llm.client import LLMClient
from planner.llm.ollama_client import OllamaClient


@pytest.mark.unit
def test_ollama_client_satisfies_protocol():
    """OllamaClient has the methods required by LLMClient."""
    assert hasattr(OllamaClient, "chat")
    assert hasattr(OllamaClient, "extract_structured_data")
    assert hasattr(OllamaClient, "is_available")


@pytest.mark.unit
def test_ollama_client_is_runtime_checkable():
    """OllamaClient passes runtime isinstance check against LLMClient."""
    client = OllamaClient.__new__(OllamaClient)
    assert isinstance(client, LLMClient)


@pytest.mark.unit
def test_openai_client_satisfies_protocol():
    """OpenAIClient has the methods required by LLMClient."""
    from planner.llm.openai_client import OpenAIClient

    assert hasattr(OpenAIClient, "chat")
    assert hasattr(OpenAIClient, "extract_structured_data")
    assert hasattr(OpenAIClient, "is_available")


@pytest.mark.unit
def test_openai_client_is_runtime_checkable():
    """OpenAIClient passes runtime isinstance check against LLMClient."""
    from planner.llm.openai_client import OpenAIClient

    client = OpenAIClient.__new__(OpenAIClient)
    assert isinstance(client, LLMClient)
