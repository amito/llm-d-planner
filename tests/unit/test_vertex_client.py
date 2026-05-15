"""Unit tests for VertexClient."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from planner.llm.client import LLMClient

# Create a fake anthropic module so VertexClient can be imported
# even when anthropic[vertex] is not installed
_mock_anthropic = ModuleType("anthropic")
_mock_anthropic.AnthropicVertex = MagicMock()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _fake_anthropic():
    """Inject a fake anthropic module for all tests in this file."""
    had_it = "anthropic" in sys.modules
    original = sys.modules.get("anthropic")
    sys.modules["anthropic"] = _mock_anthropic
    # Reset the mock between tests
    _mock_anthropic.AnthropicVertex.reset_mock()  # type: ignore[attr-defined]
    yield
    if had_it:
        sys.modules["anthropic"] = original  # type: ignore[assignment]
    else:
        sys.modules.pop("anthropic", None)


@pytest.mark.unit
def test_vertex_client_satisfies_protocol():
    """VertexClient has the methods required by LLMClient."""
    from planner.llm.vertex_client import VertexClient

    assert hasattr(VertexClient, "chat")
    assert hasattr(VertexClient, "extract_structured_data")
    assert hasattr(VertexClient, "is_available")


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_chat():
    """chat() translates messages and returns dict with message.content."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello from Claude")]
    mock_client.messages.create.return_value = mock_response

    client = VertexClient()
    result = client.chat(
        [{"role": "user", "content": "Hi"}],
        format_json=False,
        temperature=0.7,
    )

    assert result == {"message": {"content": "Hello from Claude"}}
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert call_kwargs["temperature"] == 0.7


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_extract_structured_data():
    """extract_structured_data() returns parsed JSON dict."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"use_case": "chatbot_conversational"}')]
    mock_client.messages.create.return_value = mock_response

    client = VertexClient()
    result = client.extract_structured_data("Extract intent from: I need a chatbot")

    assert result == {"use_case": "chatbot_conversational"}


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_extract_structured_data_strips_markdown_fences():
    """extract_structured_data() strips ```json fences from response."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='```json\n{"use_case": "code_completion"}\n```')]
    mock_client.messages.create.return_value = mock_response

    client = VertexClient()
    result = client.extract_structured_data("Extract intent")

    assert result == {"use_case": "code_completion"}


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_extract_structured_data_invalid_json():
    """extract_structured_data() raises ValueError on non-JSON response."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="not json")]
    mock_client.messages.create.return_value = mock_response

    client = VertexClient()
    with pytest.raises(ValueError, match="did not return valid JSON"):
        client.extract_structured_data("Extract intent")


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_is_available_success():
    """is_available() returns True when SDK responds."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="pong")]
    mock_client.messages.create.return_value = mock_response

    client = VertexClient()
    assert client.is_available() is True


@pytest.mark.unit
@patch.dict(
    "os.environ",
    {
        "VERTEX_PROJECT_ID": "test-project",
        "VERTEX_REGION": "global",
    },
)
def test_vertex_client_is_available_failure():
    """is_available() returns False when SDK raises."""
    from planner.llm.vertex_client import VertexClient

    mock_client = MagicMock()
    _mock_anthropic.AnthropicVertex.return_value = mock_client  # type: ignore[attr-defined]
    mock_client.messages.create.side_effect = Exception("auth failed")

    client = VertexClient()
    assert client.is_available() is False


@pytest.mark.unit
def test_vertex_client_missing_project_id():
    """VertexClient raises ValueError when VERTEX_PROJECT_ID is not set."""
    from planner.llm.vertex_client import VertexClient

    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="VERTEX_PROJECT_ID"),
    ):
        VertexClient()
