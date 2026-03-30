"""Test benchmark source selection and app state initialization."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import neuralnav.api.dependencies as deps


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


def _make_mock_app():
    """Create a mock FastAPI app with a state namespace."""
    app = MagicMock()
    app.state = SimpleNamespace()
    return app


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_mode_creates_client_and_syncs():
    """When source is model_catalog, init_app_state() creates client and starts sync."""
    app = _make_mock_app()
    with (
        patch(
            "neuralnav.knowledge_base.model_catalog_client.ModelCatalogClient"
        ) as mock_client_cls,
        patch("neuralnav.api.dependencies._sync_model_catalog_async") as mock_sync,
        patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
        patch("neuralnav.api.dependencies.ModelCatalog") as mock_mc,
        patch("neuralnav.api.dependencies.SLOTemplateRepository"),
        patch("neuralnav.api.dependencies.DeploymentGenerator"),
        patch("neuralnav.api.dependencies.YAMLValidator"),
        patch(
            "neuralnav.recommendation.quality.usecase_scorer.UseCaseQualityScorer"
        ) as mock_qs_cls,
        patch("neuralnav.recommendation.config_finder.ConfigFinder") as mock_cf_cls,
    ):
        deps.init_app_state(app)

        # Client created
        mock_client_cls.assert_called_once()

        # Workflow wired with shared ConfigFinder using shared instances
        mock_cf_cls.assert_called_once_with(
            catalog=mock_mc.return_value, quality_scorer=mock_qs_cls.return_value
        )
        mock_wf_cls.assert_called_once_with(config_finder=mock_cf_cls.return_value)

        # Sync started, thread stored
        mock_sync.assert_called_once()
        assert app.state.model_catalog_sync_thread == mock_sync.return_value
        assert app.state.model_catalog == mock_mc.return_value
        assert app.state.model_catalog_client == mock_client_cls.return_value


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_postgresql_workflow_uses_defaults():
    """When source is postgresql, init_app_state() creates default RecommendationWorkflow."""
    app = _make_mock_app()
    with (
        patch("neuralnav.api.dependencies.RecommendationWorkflow") as mock_wf_cls,
        patch("neuralnav.api.dependencies.ModelCatalog"),
        patch("neuralnav.api.dependencies.SLOTemplateRepository"),
        patch("neuralnav.api.dependencies.DeploymentGenerator"),
        patch("neuralnav.api.dependencies.YAMLValidator"),
    ):
        deps.init_app_state(app)
        mock_wf_cls.assert_called_once_with()
        assert app.state.workflow == mock_wf_cls.return_value
        assert app.state.model_catalog_client is None


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_init_app_state_sets_all_singletons():
    """init_app_state() populates all expected attributes on app.state."""
    app = _make_mock_app()
    with (
        patch("neuralnav.api.dependencies.RecommendationWorkflow"),
        patch("neuralnav.api.dependencies.ModelCatalog") as mock_mc,
        patch("neuralnav.api.dependencies.SLOTemplateRepository") as mock_slo,
        patch("neuralnav.api.dependencies.DeploymentGenerator") as mock_dg,
        patch("neuralnav.api.dependencies.YAMLValidator") as mock_yv,
    ):
        deps.init_app_state(app)

        assert app.state.model_catalog == mock_mc.return_value
        assert app.state.slo_repo == mock_slo.return_value
        assert app.state.deployment_generator == mock_dg.return_value
        assert app.state.yaml_validator == mock_yv.return_value
        assert app.state.cluster_managers == {}
