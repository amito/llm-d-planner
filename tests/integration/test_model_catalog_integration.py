"""Integration test: verify Model Catalog benchmark source against a live cluster.

Run manually with:
    oc port-forward -n rhoai-model-registries svc/model-catalog 9443:8443 &
    MODEL_CATALOG_URL=https://localhost:9443 \
    MODEL_CATALOG_TOKEN=$(oc whoami -t) \
    MODEL_CATALOG_VERIFY_SSL=false \
    uv run pytest tests/integration/test_model_catalog_integration.py -v
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    url = os.getenv("MODEL_CATALOG_URL")
    token = os.getenv("MODEL_CATALOG_TOKEN")
    if not url or not token:
        pytest.skip("MODEL_CATALOG_URL and MODEL_CATALOG_TOKEN required")
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

    verify_raw = os.getenv("MODEL_CATALOG_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_raw not in {"0", "false", "no", "off"}
    return ModelCatalogClient(base_url=url, token=token, verify_ssl=verify_ssl)


def test_list_models(client):
    models = client.list_models()
    if not models:
        pytest.skip("No models returned from catalog")
    assert all("name" in m for m in models)


def test_get_performance_artifacts(client):
    models = client.list_models()
    if not models:
        pytest.skip("No models returned from catalog")
    perf = []
    for model in models:
        name = model.get("name")
        if not name:
            continue
        artifacts = client.get_model_artifacts(name)
        perf = [a for a in artifacts if a.get("metricsType") == "performance-metrics"]
        if perf:
            break
    if not perf:
        pytest.skip("No performance artifacts found for any catalog model")
    props = perf[0]["customProperties"]
    assert "ttft_p95" in props
    assert "hardware_type" in props


def test_find_configurations_meeting_slo(client):
    from neuralnav.knowledge_base.model_catalog_benchmarks import ModelCatalogBenchmarkSource

    source = ModelCatalogBenchmarkSource(client)
    # Use generous SLO thresholds to avoid flaky failures when catalog data changes
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=5000,
        itl_p95_max_ms=1000,
        e2e_p95_max_ms=60000,
    )
    if not results:
        pytest.skip("No configurations met the SLO criteria in current catalog data")
    for bench in results:
        assert bench.ttft_p95 <= 5000
        assert bench.itl_p95 <= 1000
        assert bench.e2e_p95 <= 60000
