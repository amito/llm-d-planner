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

    return ModelCatalogClient(base_url=url, token=token, verify_ssl=False)


def test_list_models(client):
    models = client.list_models()
    assert len(models) > 0
    assert all("name" in m for m in models)


def test_get_performance_artifacts(client):
    models = client.list_models()
    model_name = models[0]["name"]
    artifacts = client.get_model_artifacts(model_name)
    perf = [a for a in artifacts if a.get("metricsType") == "performance-metrics"]
    # Not all models have performance artifacts, but at least one should
    if not perf:
        pytest.skip(f"No performance artifacts for {model_name}")
    props = perf[0]["customProperties"]
    assert "ttft_p95" in props
    assert "hardware_type" in props


def test_find_configurations_meeting_slo(client):
    from neuralnav.knowledge_base.model_catalog_benchmarks import ModelCatalogBenchmarkSource

    source = ModelCatalogBenchmarkSource(client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512,
        output_tokens=256,
        ttft_p95_max_ms=500,
        itl_p95_max_ms=100,
        e2e_p95_max_ms=30000,
    )
    assert len(results) > 0
    for bench in results:
        assert bench.ttft_p95 <= 500
        assert bench.itl_p95 <= 100
        assert bench.e2e_p95 <= 30000
