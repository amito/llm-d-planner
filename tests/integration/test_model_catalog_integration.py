"""Integration test: verify Model Catalog sync against a live cluster.

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
    from planner.knowledge_base.model_catalog_client import ModelCatalogClient

    verify_raw = os.getenv("MODEL_CATALOG_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_raw not in {"0", "false", "no", "off"}
    return ModelCatalogClient(base_url=url, token=token, verify_ssl=verify_ssl)


def test_list_models(client):
    models = client.list_models()
    if not models:
        pytest.skip("No models returned from catalog")
    assert all("name" in m for m in models)


def test_sync_writes_to_database(client):
    from planner.knowledge_base.benchmarks import BenchmarkRepository
    from planner.knowledge_base.model_catalog import ModelCatalog
    from planner.knowledge_base.model_catalog_sync import sync_model_catalog

    repo = BenchmarkRepository()
    catalog = ModelCatalog()
    result = sync_model_catalog(
        client=client,
        benchmark_repo=repo,
        model_catalog=catalog,
    )
    assert len(result.errors) == 0
    if result.benchmarks_inserted == 0:
        pytest.skip("No performance artifacts returned from catalog")
    cursor = repo._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM exported_summaries WHERE source = 'model_catalog'")
    db_count = cursor.fetchone()[0]
    cursor.close()
    assert db_count == result.benchmarks_inserted
