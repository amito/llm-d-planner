"""Tests that the core import chain works without optional dependencies."""

import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestCoreImportWithoutOptionalDeps:
    """Verify that importing the public API works when optional deps are missing."""

    def test_planner_import_without_optional_deps(self):
        """from planner import Planner must work without huggingface_hub,
        transformers, llm_optimizer, psycopg2, fastapi, or ollama."""
        optional_deps = [
            "huggingface_hub",
            "huggingface_hub.hf_api",
            "transformers",
            "llm_optimizer",
            "llm_optimizer.performance",
            "llm_optimizer.predefined",
            "llm_optimizer.predefined.gpus",
            "psycopg2",
            "psycopg2.extras",
            "psycopg2.extensions",
            "fastapi",
            "ollama",
            "httpx",
        ]

        # Build a dict that makes these modules raise ImportError
        blocked = {}
        for dep in optional_deps:
            if dep in sys.modules:
                blocked[dep] = None  # None causes ImportError on import

        # We must also reload the guarded modules so the try/except re-runs
        modules_to_reload = [
            "planner.capacity_planner",
            "planner.gpu_recommender",
            "planner.knowledge_base.benchmarks",
        ]

        with patch.dict(sys.modules, blocked):
            for mod_name in modules_to_reload:
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])

            # This is the critical assertion: public API must import cleanly
            from planner import Planner

            assert Planner is not None

        # Reload again to restore normal state for other tests
        for mod_name in modules_to_reload:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])

    def test_capacity_planner_importable_without_hf(self):
        """capacity_planner module must be importable without huggingface_hub/transformers."""
        hf_deps = [
            "huggingface_hub",
            "huggingface_hub.hf_api",
            "transformers",
        ]

        blocked = {}
        for dep in hf_deps:
            blocked[dep] = None

        # Remove capacity_planner from cache so it re-imports
        if "planner.capacity_planner" in sys.modules:
            del sys.modules["planner.capacity_planner"]

        with patch.dict(sys.modules, blocked):
            import planner.capacity_planner as cp

            assert cp._HF_AVAILABLE is False

        # Reload to restore normal state
        if "planner.capacity_planner" in sys.modules:
            del sys.modules["planner.capacity_planner"]
        import planner.capacity_planner  # noqa: F401

    def test_gpu_recommender_importable_without_optimizer(self):
        """gpu_recommender module must be importable without llm_optimizer."""
        optimizer_deps = [
            "llm_optimizer",
            "llm_optimizer.performance",
            "llm_optimizer.predefined",
            "llm_optimizer.predefined.gpus",
        ]

        blocked = {}
        for dep in optimizer_deps:
            blocked[dep] = None

        # Remove gpu_recommender from cache so it re-imports
        if "planner.gpu_recommender" in sys.modules:
            del sys.modules["planner.gpu_recommender"]

        with patch.dict(sys.modules, blocked):
            import planner.gpu_recommender as gr

            assert gr._LLM_OPTIMIZER_AVAILABLE is False

        # Reload to restore normal state
        if "planner.gpu_recommender" in sys.modules:
            del sys.modules["planner.gpu_recommender"]
        import planner.gpu_recommender  # noqa: F401


@pytest.mark.unit
class TestCapacityPlannerGuard:
    """Verify capacity_planner.py exposes _HF_AVAILABLE flag."""

    def test_hf_available_flag_exists(self):
        from planner.capacity_planner import _HF_AVAILABLE

        # In the test environment, huggingface_hub IS installed
        assert isinstance(_HF_AVAILABLE, bool)

    def test_hf_functions_raise_when_unavailable(self):
        """Functions that need HuggingFace raise ImportError with helpful message."""
        import planner.capacity_planner as cp

        original = cp._HF_AVAILABLE
        try:
            cp._HF_AVAILABLE = False
            with pytest.raises(ImportError, match="gpu-recommender"):
                cp.get_model_config_from_hf("some-model")
            with pytest.raises(ImportError, match="gpu-recommender"):
                cp.get_model_info_from_hf("some-model")
            with pytest.raises(ImportError, match="gpu-recommender"):
                cp._get_safetensors_metadata_cached("some-model")
        finally:
            cp._HF_AVAILABLE = original


@pytest.mark.unit
class TestGPURecommenderGuard:
    """Verify gpu_recommender.py exposes _LLM_OPTIMIZER_AVAILABLE flag."""

    def test_llm_optimizer_available_flag_exists(self):
        from planner.gpu_recommender import _LLM_OPTIMIZER_AVAILABLE

        assert isinstance(_LLM_OPTIMIZER_AVAILABLE, bool)


@pytest.mark.unit
class TestCLIGuard:
    """Verify CLI exits gracefully when llm_optimizer is not available."""

    def test_cli_exits_when_optimizer_unavailable(self):
        """main() should call sys.exit(1) when _LLM_OPTIMIZER_AVAILABLE is False."""
        import planner.cli.planner_cli as cli

        original = cli._LLM_OPTIMIZER_AVAILABLE
        try:
            cli._LLM_OPTIMIZER_AVAILABLE = False
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 1
        finally:
            cli._LLM_OPTIMIZER_AVAILABLE = original
