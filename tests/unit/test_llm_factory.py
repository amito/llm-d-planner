"""Unit tests for LLM client factory."""

from unittest.mock import patch

import pytest

from planner.llm.ollama_client import OllamaClient


@pytest.mark.unit
def test_factory_defaults_to_ollama():
    """No LLM_PROVIDER env var returns OllamaClient."""
    from planner.llm.factory import create_llm_client

    with patch.dict("os.environ", {}, clear=False):
        os_env = dict(__import__("os").environ)
        os_env.pop("LLM_PROVIDER", None)
        with patch.dict("os.environ", os_env, clear=True):
            client = create_llm_client()
            assert isinstance(client, OllamaClient)


@pytest.mark.unit
def test_factory_explicit_ollama():
    """LLM_PROVIDER=ollama returns OllamaClient."""
    from planner.llm.factory import create_llm_client

    with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
        client = create_llm_client()
        assert isinstance(client, OllamaClient)


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "LLM_PROVIDER": "vertex",
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
@patch("planner.llm.vertex_client.VertexClient")
def test_factory_vertex(mock_vertex_class):
    """LLM_PROVIDER=vertex returns VertexClient."""
    from planner.llm.factory import create_llm_client

    mock_vertex_class.return_value = "mock-vertex-client"
    client = create_llm_client()
    assert client == "mock-vertex-client"
    mock_vertex_class.assert_called_once()


@pytest.mark.unit
def test_factory_unknown_provider_raises():
    """Unknown LLM_PROVIDER raises ValueError."""
    from planner.llm.factory import create_llm_client

    with (
        patch.dict("os.environ", {"LLM_PROVIDER": "openai"}),
        pytest.raises(ValueError, match="Unknown LLM provider"),
    ):
        create_llm_client()
