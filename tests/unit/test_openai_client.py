"""Unit tests for OpenAI-compatible LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from planner.llm.client import LLMClient


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


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_chat(mock_openai_class):
    """chat() returns response in expected format."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello there!"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient()
    result = client.chat(
        [{"role": "user", "content": "Hi"}],
        format_json=False,
        temperature=0.7,
    )

    assert result == {"message": {"content": "Hello there!"}}
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_chat_json_format(mock_openai_class):
    """chat() passes response_format when format_json=True."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = '{"key": "value"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient()
    client.chat(
        [{"role": "user", "content": "Return JSON"}],
        format_json=True,
        temperature=0.3,
    )

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_extract_structured_data(mock_openai_class):
    """extract_structured_data() returns parsed JSON dict."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = '{"use_case": "chatbot", "user_count": 100}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient()
    result = client.extract_structured_data("Extract intent from: I need a chatbot")

    assert result == {"use_case": "chatbot", "user_count": 100}


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_extract_strips_markdown_fences(mock_openai_class):
    """extract_structured_data() strips markdown code fences from response."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"key": "value"}\n```'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient()
    result = client.extract_structured_data("Extract data")

    assert result == {"key": "value"}


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_extract_invalid_json_raises(mock_openai_class):
    """extract_structured_data() raises ValueError on invalid JSON."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = "not json at all"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    client = OpenAIClient()
    with pytest.raises(ValueError, match="LLM did not return valid JSON"):
        client.extract_structured_data("Extract data")


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_is_available_true(mock_openai_class):
    """is_available() returns True when API responds."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.return_value = []

    client = OpenAIClient()
    assert client.is_available() is True


@pytest.mark.unit
@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_is_available_false(mock_openai_class):
    """is_available() returns False when API is unreachable."""
    from planner.llm.openai_client import OpenAIClient

    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.models.list.side_effect = Exception("Connection refused")

    client = OpenAIClient()
    assert client.is_available() is False


@pytest.mark.unit
@patch.dict("os.environ", {}, clear=True)
def test_openai_client_missing_api_key_raises():
    """OpenAIClient raises ValueError when OPENAI_API_KEY is not set."""
    from planner.llm.openai_client import OpenAIClient

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient()


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {"OPENAI_API_KEY": "test-key", "LLM_MODEL": "my-custom-model"},
)
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_reads_llm_model(mock_openai_class):
    """OpenAIClient reads model from LLM_MODEL env var."""
    from planner.llm.openai_client import OpenAIClient

    mock_openai_class.return_value = MagicMock()
    client = OpenAIClient()
    assert client.model == "my-custom-model"


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://my-litellm.example.com/v1",
    },
)
@patch("planner.llm.openai_client.OpenAI")
def test_openai_client_custom_base_url(mock_openai_class):
    """OpenAIClient passes OPENAI_BASE_URL to the SDK."""
    from planner.llm.openai_client import OpenAIClient

    mock_openai_class.return_value = MagicMock()
    OpenAIClient()

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://my-litellm.example.com/v1"
