"""Test benchmark source selection via environment variable."""

from unittest.mock import patch

import pytest

import neuralnav.api.dependencies as deps


@pytest.fixture(autouse=True)
def _reset_workflow_singleton():
    """Ensure dependency singletons are reset before and after each test."""
    prev_workflow = deps._workflow
    prev_client = deps._model_catalog_client
    deps._workflow = None
    deps._model_catalog_client = None
    try:
        yield
    finally:
        deps._workflow = prev_workflow
        deps._model_catalog_client = prev_client


@pytest.mark.unit
@patch.dict("os.environ", {}, clear=False)
def test_default_source_is_postgresql():
    """When NEURALNAV_BENCHMARK_SOURCE is not set, default to postgresql."""
    import os

    os.environ.pop("NEURALNAV_BENCHMARK_SOURCE", None)
    assert deps._get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_explicit_postgresql_source():
    """When NEURALNAV_BENCHMARK_SOURCE=postgresql, return postgresql."""
    assert deps._get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_source():
    """When NEURALNAV_BENCHMARK_SOURCE=model_catalog, return model_catalog."""
    assert deps._get_benchmark_source_type() == "model_catalog"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": " Model_Catalog "}, clear=False)
def test_benchmark_source_normalization():
    """Whitespace and case in NEURALNAV_BENCHMARK_SOURCE are normalized."""
    assert deps._get_benchmark_source_type() == "model_catalog"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "invalid_source"}, clear=False)
def test_unknown_benchmark_source_defaults_to_postgresql():
    """Unknown NEURALNAV_BENCHMARK_SOURCE values default to postgresql."""
    assert deps._get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_workflow_creates_correct_components():
    """When source is model_catalog, get_workflow() should wire up Model Catalog components."""
    with (
        patch(
            "neuralnav.knowledge_base.model_catalog_client.ModelCatalogClient"
        ) as mock_client_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_benchmarks.ModelCatalogBenchmarkSource"
        ) as mock_bench_src_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_models.ModelCatalogModelSource"
        ) as mock_model_src_cls,
        patch(
            "neuralnav.knowledge_base.model_catalog_quality.ModelCatalogQualityScorer"
        ) as mock_quality_cls,
        patch("neuralnav.recommendation.config_finder.ConfigFinder") as mock_finder_cls,
        patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
        patch("neuralnav.api.dependencies._preload_model_catalog_async"),
    ):
        deps.get_workflow()

        # Verify client created once
        mock_client_cls.assert_called_once()
        client_instance = mock_client_cls.return_value

        # Verify all Model Catalog components wired with the same client
        mock_bench_src_cls.assert_called_once_with(client_instance)
        mock_model_src_cls.assert_called_once_with(client_instance)
        mock_quality_cls.assert_called_once_with(client_instance)

        # Verify ConfigFinder created with all three components
        mock_finder_cls.assert_called_once_with(
            benchmark_repo=mock_bench_src_cls.return_value,
            catalog=mock_model_src_cls.return_value,
            quality_scorer=mock_quality_cls.return_value,
        )

        # Verify workflow created with the custom config_finder
        mock_wf_cls.assert_called_once_with(config_finder=mock_finder_cls.return_value)


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_postgresql_workflow_uses_defaults():
    """When source is postgresql, get_workflow() creates default RecommendationWorkflow."""
    with patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls:
        deps.get_workflow()
        mock_wf_cls.assert_called_once_with()
