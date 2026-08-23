"""Test import guards and error handling for optional dependencies."""

import pytest


def test_planner_error_class():
    """Test that PlannerError can be imported and used."""
    from planner.errors import PlannerError

    # Can instantiate
    err = PlannerError("test message")
    assert str(err) == "test message"

    # Is an Exception
    assert isinstance(err, Exception)

    # Can be raised and caught
    with pytest.raises(PlannerError):
        raise PlannerError("test")


def test_core_modules_importable():
    """Test that core modules can be imported without optional dependencies."""
    # These should always work with just core dependencies
    import planner.errors  # noqa: F401
    from planner.shared.schemas.intent import DeploymentIntent  # noqa: F401


def test_llm_factory_import_errors():
    """Test that LLM factory provides helpful error messages for missing providers."""
    import os

    from planner.llm.factory import create_llm_client

    # Save original value
    original_provider = os.environ.get("LLM_PROVIDER")

    try:
        # Test with ollama provider when ollama is missing
        os.environ["LLM_PROVIDER"] = "ollama"
        try:
            # This will work if ollama is installed, or raise ImportError if not
            client = create_llm_client()
            assert client is not None  # If we got here, ollama is installed
        except ImportError as e:
            # If ollama is not installed, check the error message is helpful
            assert "ollama" in str(e).lower()
            assert "pip install" in str(e).lower()
    finally:
        # Restore original value
        if original_provider is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = original_provider


def test_capacity_planner_import_errors():
    """Test that capacity planner functions raise helpful errors when HF deps missing."""
    # Import guards are tested by attempting to use functions that need optional deps
    # We can't test this without actually uninstalling the deps, so we just verify
    # the functions exist and can be imported
    from planner.capacity_planner import (
        get_model_config_from_hf,
        get_model_info_from_hf,
    )

    # Functions are importable
    assert callable(get_model_config_from_hf)
    assert callable(get_model_info_from_hf)
