# RHOAI Model Catalog Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable NeuralNav to load performance benchmarks, accuracy scores, and model metadata from the RHOAI Model Catalog API as an alternative to PostgreSQL + CSV files, controlled by an explicit environment variable.

**Architecture:** Provider pattern — define protocols for each data source (`BenchmarkSource`, `QualityScorer`, `ModelSource`), keep existing implementations unchanged, add new RHOAI-backed implementations, select at startup via `NEURALNAV_BENCHMARK_SOURCE` env var.

**Tech Stack:** Python 3.12, `httpx` for HTTP client, `typing.Protocol` for interfaces, `pytest` with mocks for testing

**Design doc:** `docs/plans/2026-03-02-rhoai-model-catalog-integration-design.md`

---

### Task 1: BenchmarkSource Protocol

Define the protocol that both `BenchmarkRepository` (PostgreSQL) and the new Model Catalog source must implement.

**Files:**
- Create: `src/neuralnav/knowledge_base/benchmark_source.py`
- Test: `tests/unit/test_benchmark_source_protocol.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_benchmark_source_protocol.py
"""Verify BenchmarkSource protocol is satisfied by existing and new implementations."""
import pytest
from neuralnav.knowledge_base.benchmark_source import BenchmarkSource


class FakeBenchmarkSource:
    """Minimal implementation to verify protocol shape."""

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0,
        percentile: str = "p95",
        gpu_types: list[str] | None = None,
    ) -> list:
        return []


@pytest.mark.unit
def test_fake_satisfies_protocol():
    source: BenchmarkSource = FakeBenchmarkSource()
    result = source.find_configurations_meeting_slo(
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=200, itl_p95_max_ms=30, e2e_p95_max_ms=8000,
    )
    assert result == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_benchmark_source_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'neuralnav.knowledge_base.benchmark_source'`

**Step 3: Write minimal implementation**

```python
# src/neuralnav/knowledge_base/benchmark_source.py
"""Protocol definition for benchmark data sources."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neuralnav.knowledge_base.benchmarks import BenchmarkData


@runtime_checkable
class BenchmarkSource(Protocol):
    """Protocol for benchmark data providers.

    Implementations:
    - BenchmarkRepository: PostgreSQL (standalone/upstream)
    - ModelCatalogBenchmarkSource: RHOAI Model Catalog API
    """

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0,
        percentile: str = "p95",
        gpu_types: list[str] | None = None,
    ) -> list[BenchmarkData]: ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_benchmark_source_protocol.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add BenchmarkSource protocol for pluggable benchmark data sources
```

---

### Task 2: Update ConfigFinder to accept BenchmarkSource

Change `ConfigFinder.__init__` type hint from `BenchmarkRepository` to `BenchmarkSource`. Existing code continues to work — `BenchmarkRepository` satisfies the protocol.

**Files:**
- Modify: `src/neuralnav/recommendation/config_finder.py` (lines 23-24, 44-54)
- Test: `tests/unit/test_config_finder_protocol.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_config_finder_protocol.py
"""Verify ConfigFinder works with any BenchmarkSource implementation."""
import pytest
from unittest.mock import MagicMock

from neuralnav.knowledge_base.benchmark_source import BenchmarkSource
from neuralnav.knowledge_base.benchmarks import BenchmarkData
from neuralnav.recommendation.config_finder import ConfigFinder


def _make_bench(model: str = "test/model", hardware: str = "H100", hw_count: int = 1) -> BenchmarkData:
    return BenchmarkData({
        "model_hf_repo": model, "hardware": hardware, "hardware_count": hw_count,
        "framework": "vllm", "framework_version": "0.8.4",
        "prompt_tokens": 512, "output_tokens": 256,
        "mean_input_tokens": 512, "mean_output_tokens": 256,
        "ttft_mean": 50, "ttft_p90": 70, "ttft_p95": 80, "ttft_p99": 100,
        "itl_mean": 10, "itl_p90": 15, "itl_p95": 20, "itl_p99": 25,
        "e2e_mean": 3000, "e2e_p90": 4000, "e2e_p95": 5000, "e2e_p99": 6000,
        "tps_mean": 1000, "tps_p90": 900, "tps_p95": 800, "tps_p99": 700,
        "tokens_per_second": 1000, "requests_per_second": 7,
    })


@pytest.mark.unit
def test_config_finder_accepts_benchmark_source():
    """ConfigFinder should accept any BenchmarkSource, not just BenchmarkRepository."""
    mock_source = MagicMock(spec=BenchmarkSource)
    mock_source.find_configurations_meeting_slo.return_value = [_make_bench()]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    finder = ConfigFinder(benchmark_repo=mock_source, catalog=mock_catalog)
    assert finder.benchmark_repo is mock_source
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_finder_protocol.py -q`
Expected: The test should actually pass already since Python doesn't enforce type hints at runtime — but this test documents the intended contract. If using `basedpyright`, the type checker would flag mismatches.

**Step 3: Update the type hints in ConfigFinder**

In `src/neuralnav/recommendation/config_finder.py`:

Change line 23:
```python
# Before:
from neuralnav.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
# After:
from neuralnav.knowledge_base.benchmarks import BenchmarkData
from neuralnav.knowledge_base.benchmark_source import BenchmarkSource
```

Change the `__init__` signature (line 44-54):
```python
# Before:
def __init__(
    self, benchmark_repo: BenchmarkRepository | None = None, catalog: ModelCatalog | None = None
):
    ...
    self.benchmark_repo = benchmark_repo or BenchmarkRepository()

# After:
def __init__(
    self, benchmark_repo: BenchmarkSource | None = None, catalog: ModelCatalog | None = None
):
    ...
    if benchmark_repo is not None:
        self.benchmark_repo = benchmark_repo
    else:
        from neuralnav.knowledge_base.benchmarks import BenchmarkRepository
        self.benchmark_repo = BenchmarkRepository()
```

The lazy import of `BenchmarkRepository` inside the `else` branch avoids requiring PostgreSQL when a different source is provided.

**Step 4: Run tests to verify everything passes**

Run: `uv run pytest tests/unit/test_config_finder_protocol.py tests/unit/test_benchmark_source_protocol.py -q`
Expected: PASS

**Step 5: Commit**

```
refactor: update ConfigFinder to accept any BenchmarkSource implementation
```

---

### Task 3: Model Catalog HTTP Client

A thin HTTP wrapper for the Model Catalog REST API. Returns raw dicts — no domain mapping.

**Files:**
- Create: `src/neuralnav/knowledge_base/model_catalog_client.py`
- Test: `tests/unit/test_model_catalog_client.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_model_catalog_client.py
"""Tests for Model Catalog HTTP client."""
import json
import pytest
from unittest.mock import patch, MagicMock

from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient


@pytest.fixture
def client():
    return ModelCatalogClient(
        base_url="https://localhost:8443",
        token="test-token",
        source_id="redhat_ai_validated_models",
        verify_ssl=False,
    )


FAKE_MODELS_RESPONSE = {
    "items": [
        {
            "name": "RedHatAI/granite-3.1-8b-instruct",
            "provider": "IBM",
            "description": "Granite model",
            "tasks": ["text-to-text"],
            "customProperties": {
                "size": {"string_value": "8B params", "metadataType": "MetadataStringValue"},
                "validated": {"string_value": "", "metadataType": "MetadataStringValue"},
            },
        }
    ],
    "size": 1,
    "pageSize": 100,
    "nextPageToken": "",
}

FAKE_ARTIFACTS_RESPONSE = {
    "items": [
        {
            "artifactType": "metrics-artifact",
            "metricsType": "performance-metrics",
            "customProperties": {
                "hardware_type": {"string_value": "H100", "metadataType": "MetadataStringValue"},
                "hardware_count": {"int_value": 4, "metadataType": "MetadataIntValue"},
                "ttft_p95": {"double_value": 98.15, "metadataType": "MetadataDoubleValue"},
                "requests_per_second": {"double_value": 7.0, "metadataType": "MetadataDoubleValue"},
            },
        },
        {
            "artifactType": "metrics-artifact",
            "metricsType": "accuracy-metrics",
            "customProperties": {
                "overall_average": {"double_value": 57.67, "metadataType": "MetadataDoubleValue"},
                "mmlu": {"double_value": 82.04, "metadataType": "MetadataDoubleValue"},
            },
        },
    ],
    "size": 2,
    "pageSize": 200,
    "nextPageToken": "",
}


@pytest.mark.unit
@patch("neuralnav.knowledge_base.model_catalog_client.httpx")
def test_list_models(mock_httpx, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = FAKE_MODELS_RESPONSE
    mock_response.raise_for_status = MagicMock()
    mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_response)))
    mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

    models = client.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "RedHatAI/granite-3.1-8b-instruct"


@pytest.mark.unit
@patch("neuralnav.knowledge_base.model_catalog_client.httpx")
def test_get_artifacts(mock_httpx, client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = FAKE_ARTIFACTS_RESPONSE
    mock_response.raise_for_status = MagicMock()
    mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_response)))
    mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

    artifacts = client.get_model_artifacts("RedHatAI/granite-3.1-8b-instruct")
    perf = [a for a in artifacts if a.get("metricsType") == "performance-metrics"]
    acc = [a for a in artifacts if a.get("metricsType") == "accuracy-metrics"]
    assert len(perf) == 1
    assert len(acc) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_catalog_client.py -q`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/neuralnav/knowledge_base/model_catalog_client.py
"""HTTP client for the RHOAI Model Catalog API."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Default ServiceAccount token path inside a pod
_SA_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


class ModelCatalogClient:
    """Thin HTTP wrapper for the Model Catalog v1alpha1 REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        source_id: str | None = None,
        verify_ssl: bool | None = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("MODEL_CATALOG_URL", "https://model-catalog.rhoai-model-registries.svc:8443")
        ).rstrip("/")
        self.token = token or os.getenv("MODEL_CATALOG_TOKEN") or self._read_sa_token()
        self.source_id = source_id or os.getenv("MODEL_CATALOG_SOURCE_ID", "redhat_ai_validated_models")

        verify_env = os.getenv("MODEL_CATALOG_VERIFY_SSL", "true")
        self.verify_ssl = verify_ssl if verify_ssl is not None else (verify_env.lower() == "true")

        self._api_base = f"{self.base_url}/api/model_catalog/v1alpha1"

    @staticmethod
    def _read_sa_token() -> str:
        """Read ServiceAccount token from pod mount."""
        if _SA_TOKEN_PATH.exists():
            return _SA_TOKEN_PATH.read_text().strip()
        return ""

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_json(self, path: str) -> dict:
        """GET request returning parsed JSON."""
        url = f"{self._api_base}{path}"
        with httpx.Client(verify=self.verify_ssl, timeout=30.0) as http:
            resp = http.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def list_models(self, page_size: int = 200) -> list[dict]:
        """List all models from the configured source."""
        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(
                f"/sources/{self.source_id}/models?pageSize={page_size}{token_param}"
            )
            all_items.extend(data.get("items", []))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        logger.info(f"Fetched {len(all_items)} models from Model Catalog")
        return all_items

    def get_model_artifacts(self, model_name: str, page_size: int = 200) -> list[dict]:
        """Get all artifacts (model + metrics) for a model."""
        encoded = quote(model_name, safe="")
        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(
                f"/sources/{self.source_id}/models/{encoded}/artifacts?pageSize={page_size}{token_param}"
            )
            all_items.extend(data.get("items", []))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        return all_items
```

**Step 4: Add `httpx` dependency and run tests**

Run: `uv add httpx && uv run pytest tests/unit/test_model_catalog_client.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add Model Catalog HTTP client for RHOAI integration
```

---

### Task 4: ModelCatalogBenchmarkSource

The core implementation — fetches performance-metrics artifacts, maps them to `BenchmarkData`, and implements `find_configurations_meeting_slo()` with in-memory filtering.

**Files:**
- Create: `src/neuralnav/knowledge_base/model_catalog_benchmarks.py`
- Test: `tests/unit/test_model_catalog_benchmarks.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_model_catalog_benchmarks.py
"""Tests for ModelCatalogBenchmarkSource."""
import pytest
from unittest.mock import MagicMock

from neuralnav.knowledge_base.model_catalog_benchmarks import ModelCatalogBenchmarkSource


def _perf_artifact(
    model_id="RedHatAI/test-model",
    hardware="H100",
    hw_count=4,
    ttft_p95=80.0,
    itl_p95=20.0,
    e2e_p95=5000.0,
    rps=7.0,
    prompt_tokens=512,
    output_tokens=256,
):
    """Build a fake performance-metrics artifact dict."""
    return {
        "artifactType": "metrics-artifact",
        "metricsType": "performance-metrics",
        "customProperties": {
            "model_id": {"string_value": model_id, "metadataType": "MetadataStringValue"},
            "hardware_type": {"string_value": hardware, "metadataType": "MetadataStringValue"},
            "hardware_configuration": {"string_value": f"{hardware} x {hw_count}", "metadataType": "MetadataStringValue"},
            "hardware_count": {"int_value": hw_count, "metadataType": "MetadataIntValue"},
            "ttft_mean": {"double_value": ttft_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "ttft_p90": {"double_value": ttft_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "ttft_p95": {"double_value": ttft_p95, "metadataType": "MetadataDoubleValue"},
            "ttft_p99": {"double_value": ttft_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "itl_mean": {"double_value": itl_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "itl_p90": {"double_value": itl_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "itl_p95": {"double_value": itl_p95, "metadataType": "MetadataDoubleValue"},
            "itl_p99": {"double_value": itl_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "e2e_mean": {"double_value": e2e_p95 * 0.7, "metadataType": "MetadataDoubleValue"},
            "e2e_p90": {"double_value": e2e_p95 * 0.9, "metadataType": "MetadataDoubleValue"},
            "e2e_p95": {"double_value": e2e_p95, "metadataType": "MetadataDoubleValue"},
            "e2e_p99": {"double_value": e2e_p95 * 1.3, "metadataType": "MetadataDoubleValue"},
            "tps_mean": {"double_value": 1500.0, "metadataType": "MetadataDoubleValue"},
            "tps_p90": {"double_value": 1200.0, "metadataType": "MetadataDoubleValue"},
            "tps_p95": {"double_value": 1000.0, "metadataType": "MetadataDoubleValue"},
            "tps_p99": {"double_value": 800.0, "metadataType": "MetadataDoubleValue"},
            "requests_per_second": {"double_value": rps, "metadataType": "MetadataDoubleValue"},
            "mean_input_tokens": {"double_value": float(prompt_tokens), "metadataType": "MetadataDoubleValue"},
            "mean_output_tokens": {"double_value": float(output_tokens), "metadataType": "MetadataDoubleValue"},
            "framework_type": {"string_value": "vllm", "metadataType": "MetadataStringValue"},
            "framework_version": {"string_value": "v0.8.4", "metadataType": "MetadataStringValue"},
            "use_case": {"string_value": "chatbot", "metadataType": "MetadataStringValue"},
            "profiler_config": {
                "string_value": '{"args": {"prompt_tokens": ' + str(prompt_tokens) + ', "output_tokens": ' + str(output_tokens) + '}}',
                "metadataType": "MetadataStringValue",
            },
        },
    }


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {"name": "RedHatAI/test-model", "source_id": "redhat_ai_validated_models"},
        {"name": "RedHatAI/other-model", "source_id": "redhat_ai_validated_models"},
    ]
    client.get_model_artifacts.side_effect = lambda name: {
        "RedHatAI/test-model": [
            _perf_artifact(model_id="RedHatAI/test-model", hardware="H100", hw_count=4, ttft_p95=80, itl_p95=20, e2e_p95=5000, rps=7),
            _perf_artifact(model_id="RedHatAI/test-model", hardware="L4", hw_count=1, ttft_p95=300, itl_p95=50, e2e_p95=15000, rps=2),
        ],
        "RedHatAI/other-model": [
            _perf_artifact(model_id="RedHatAI/other-model", hardware="H100", hw_count=2, ttft_p95=100, itl_p95=25, e2e_p95=6000, rps=5),
        ],
    }.get(name, [])
    return client


@pytest.mark.unit
def test_find_configs_meeting_slo(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=200, itl_p95_max_ms=30, e2e_p95_max_ms=8000,
    )
    # H100x4 (ttft=80, itl=20, e2e=5000) and H100x2 (ttft=100, itl=25, e2e=6000) meet SLO
    # L4x1 (ttft=300) does NOT meet TTFT SLO of 200ms
    assert len(results) == 2
    hardware_configs = {(r.hardware, r.hardware_count) for r in results}
    assert ("H100", 4) in hardware_configs
    assert ("H100", 2) in hardware_configs


@pytest.mark.unit
def test_find_configs_gpu_filter(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=500, itl_p95_max_ms=60, e2e_p95_max_ms=20000,
        gpu_types=["L4"],
    )
    assert len(results) == 1
    assert results[0].hardware == "L4"


@pytest.mark.unit
def test_find_configs_no_match(mock_client):
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=10, itl_p95_max_ms=5, e2e_p95_max_ms=100,
    )
    assert results == []


@pytest.mark.unit
def test_benchmark_data_fields(mock_client):
    """Verify mapped BenchmarkData has all required fields."""
    source = ModelCatalogBenchmarkSource(mock_client)
    results = source.find_configurations_meeting_slo(
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=200, itl_p95_max_ms=30, e2e_p95_max_ms=8000,
    )
    bench = results[0]
    assert bench.model_hf_repo is not None
    assert bench.hardware is not None
    assert bench.hardware_count > 0
    assert bench.ttft_p95 > 0
    assert bench.itl_p95 > 0
    assert bench.e2e_p95 > 0
    assert bench.requests_per_second > 0
    assert bench.tokens_per_second > 0
    assert bench.prompt_tokens == 512
    assert bench.output_tokens == 256
    assert bench.estimated is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_catalog_benchmarks.py -q`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/neuralnav/knowledge_base/model_catalog_benchmarks.py
"""RHOAI Model Catalog benchmark source — loads performance data from Model Catalog artifacts."""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from neuralnav.knowledge_base.benchmarks import BenchmarkData

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

# Cache TTL in seconds (1 hour)
_CACHE_TTL = 3600


def _prop_str(props: dict, key: str, default: str = "") -> str:
    """Extract string value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return entry.get("string_value", default)


def _prop_float(props: dict, key: str, default: float = 0.0) -> float:
    """Extract double/float value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return float(entry.get("double_value", default))


def _prop_int(props: dict, key: str, default: int = 0) -> int:
    """Extract int value from customProperties."""
    entry = props.get(key)
    if entry is None:
        return default
    return int(entry.get("int_value", entry.get("double_value", default)))


def _parse_profiler_config(props: dict) -> tuple[int, int]:
    """Extract prompt_tokens and output_tokens from profiler_config JSON."""
    raw = _prop_str(props, "profiler_config")
    if not raw:
        # Fall back to mean_input/output_tokens
        return int(_prop_float(props, "mean_input_tokens")), int(_prop_float(props, "mean_output_tokens"))
    try:
        config = json.loads(raw)
        args = config.get("args", {})
        return int(args.get("prompt_tokens", 0)), int(args.get("output_tokens", 0))
    except (json.JSONDecodeError, TypeError):
        return int(_prop_float(props, "mean_input_tokens")), int(_prop_float(props, "mean_output_tokens"))


def _artifact_to_benchmark_data(artifact: dict) -> BenchmarkData | None:
    """Map a performance-metrics artifact to a BenchmarkData instance."""
    props = artifact.get("customProperties", {})

    prompt_tokens, output_tokens = _parse_profiler_config(props)
    if prompt_tokens == 0 or output_tokens == 0:
        return None

    tps_mean = _prop_float(props, "tps_mean")

    data = {
        "model_hf_repo": _prop_str(props, "model_id"),
        "hardware": _prop_str(props, "hardware_type"),
        "hardware_count": _prop_int(props, "hardware_count"),
        "framework": _prop_str(props, "framework_type", "vllm"),
        "framework_version": _prop_str(props, "framework_version"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "mean_input_tokens": _prop_float(props, "mean_input_tokens"),
        "mean_output_tokens": _prop_float(props, "mean_output_tokens"),
        "ttft_mean": _prop_float(props, "ttft_mean"),
        "ttft_p90": _prop_float(props, "ttft_p90"),
        "ttft_p95": _prop_float(props, "ttft_p95"),
        "ttft_p99": _prop_float(props, "ttft_p99"),
        "itl_mean": _prop_float(props, "itl_mean"),
        "itl_p90": _prop_float(props, "itl_p90"),
        "itl_p95": _prop_float(props, "itl_p95"),
        "itl_p99": _prop_float(props, "itl_p99"),
        "e2e_mean": _prop_float(props, "e2e_mean"),
        "e2e_p90": _prop_float(props, "e2e_p90"),
        "e2e_p95": _prop_float(props, "e2e_p95"),
        "e2e_p99": _prop_float(props, "e2e_p99"),
        "tps_mean": tps_mean,
        "tps_p90": _prop_float(props, "tps_p90"),
        "tps_p95": _prop_float(props, "tps_p95"),
        "tps_p99": _prop_float(props, "tps_p99"),
        "tokens_per_second": tps_mean,  # Legacy field
        "requests_per_second": _prop_float(props, "requests_per_second"),
        "estimated": False,
    }

    if not data["model_hf_repo"] or not data["hardware"]:
        return None

    return BenchmarkData(data)


class ModelCatalogBenchmarkSource:
    """Load performance benchmarks from the RHOAI Model Catalog API.

    Implements the BenchmarkSource protocol.
    """

    def __init__(self, client: ModelCatalogClient):
        self._client = client
        self._benchmarks: list[BenchmarkData] = []
        self._loaded_at: float = 0

    def _ensure_loaded(self) -> None:
        """Load or refresh benchmark cache."""
        if self._benchmarks and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        """Fetch all models and their performance-metrics artifacts."""
        benchmarks: list[BenchmarkData] = []
        models = self._client.list_models()
        for model in models:
            model_name = model.get("name", "")
            if not model_name:
                continue
            artifacts = self._client.get_model_artifacts(model_name)
            for artifact in artifacts:
                if (
                    artifact.get("artifactType") == "metrics-artifact"
                    and artifact.get("metricsType") == "performance-metrics"
                ):
                    bench = _artifact_to_benchmark_data(artifact)
                    if bench is not None:
                        benchmarks.append(bench)

        self._benchmarks = benchmarks
        self._loaded_at = time.time()
        logger.info(f"Loaded {len(benchmarks)} performance benchmarks from Model Catalog")

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0,
        percentile: str = "p95",
        gpu_types: list[str] | None = None,
    ) -> list[BenchmarkData]:
        """Find configurations meeting SLO from cached Model Catalog data."""
        self._ensure_loaded()

        valid_percentiles = {"mean", "p90", "p95", "p99"}
        if percentile not in valid_percentiles:
            percentile = "p95"

        # Normalize GPU filter to uppercase for comparison
        gpu_filter = {g.upper() for g in gpu_types} if gpu_types else None

        results: list[BenchmarkData] = []
        for bench in self._benchmarks:
            # Match traffic profile
            if bench.prompt_tokens != prompt_tokens or bench.output_tokens != output_tokens:
                continue

            # GPU filter
            if gpu_filter and bench.hardware.upper() not in gpu_filter:
                continue

            # SLO check at requested percentile
            ttft = getattr(bench, f"ttft_{percentile}", 0) or 0
            itl = getattr(bench, f"itl_{percentile}", 0) or 0
            e2e = getattr(bench, f"e2e_{percentile}", 0) or 0

            if ttft > ttft_p95_max_ms or itl > itl_p95_max_ms or e2e > e2e_p95_max_ms:
                continue

            # Min QPS check
            if bench.requests_per_second < min_qps:
                continue

            results.append(bench)

        # Deduplicate: keep highest RPS per (model, hardware, hardware_count)
        best: dict[tuple, BenchmarkData] = {}
        for bench in results:
            key = (bench.model_hf_repo, bench.hardware, bench.hardware_count)
            existing = best.get(key)
            if existing is None or bench.requests_per_second > existing.requests_per_second:
                best[key] = bench

        final = sorted(best.values(), key=lambda b: (b.model_hf_repo, b.hardware, b.hardware_count))
        logger.info(f"Found {len(final)} benchmarks meeting SLO criteria (from Model Catalog)")
        return final
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_model_catalog_benchmarks.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add ModelCatalogBenchmarkSource for RHOAI performance data
```

---

### Task 5: ModelCatalogQualityScorer

Fetch accuracy-metrics artifacts and provide quality scores.

**Files:**
- Create: `src/neuralnav/knowledge_base/model_catalog_quality.py`
- Test: `tests/unit/test_model_catalog_quality.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_model_catalog_quality.py
"""Tests for ModelCatalogQualityScorer."""
import pytest
from unittest.mock import MagicMock

from neuralnav.knowledge_base.model_catalog_quality import ModelCatalogQualityScorer


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {"name": "RedHatAI/model-a"},
        {"name": "RedHatAI/model-b"},
    ]
    client.get_model_artifacts.side_effect = lambda name: {
        "RedHatAI/model-a": [
            {
                "artifactType": "metrics-artifact",
                "metricsType": "accuracy-metrics",
                "customProperties": {
                    "overall_average": {"double_value": 57.67, "metadataType": "MetadataDoubleValue"},
                    "mmlu": {"double_value": 82.04, "metadataType": "MetadataDoubleValue"},
                },
            }
        ],
        "RedHatAI/model-b": [
            {
                "artifactType": "metrics-artifact",
                "metricsType": "accuracy-metrics",
                "customProperties": {
                    "overall_average": {"double_value": 42.0, "metadataType": "MetadataDoubleValue"},
                },
            }
        ],
    }.get(name, [])
    return client


@pytest.mark.unit
def test_get_quality_score(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("RedHatAI/model-a", "chatbot_conversational")
    assert score == pytest.approx(57.67)


@pytest.mark.unit
def test_get_quality_score_unknown_model(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("unknown/model", "chatbot_conversational")
    assert score == 0.0


@pytest.mark.unit
def test_get_quality_score_case_insensitive(mock_client):
    scorer = ModelCatalogQualityScorer(mock_client)
    score = scorer.get_quality_score("redhatai/model-a", "chatbot_conversational")
    assert score == pytest.approx(57.67)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_catalog_quality.py -q`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/neuralnav/knowledge_base/model_catalog_quality.py
"""Quality/accuracy scorer using RHOAI Model Catalog accuracy-metrics artifacts."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600


class ModelCatalogQualityScorer:
    """Score model quality using accuracy-metrics from Model Catalog.

    Drop-in replacement for UseCaseQualityScorer when running with RHOAI.
    Uses overall_average from accuracy-metrics artifacts.
    """

    def __init__(self, client: ModelCatalogClient):
        self._client = client
        self._scores: dict[str, float] = {}  # model_name (lowercase) -> overall_average
        self._loaded_at: float = 0

    def _ensure_loaded(self) -> None:
        if self._scores and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        scores: dict[str, float] = {}
        models = self._client.list_models()
        for model in models:
            model_name = model.get("name", "")
            if not model_name:
                continue
            artifacts = self._client.get_model_artifacts(model_name)
            for artifact in artifacts:
                if (
                    artifact.get("artifactType") == "metrics-artifact"
                    and artifact.get("metricsType") == "accuracy-metrics"
                ):
                    props = artifact.get("customProperties", {})
                    avg_entry = props.get("overall_average")
                    if avg_entry:
                        scores[model_name.lower()] = float(avg_entry.get("double_value", 0))
                    break  # One accuracy artifact per model
        self._scores = scores
        self._loaded_at = time.time()
        logger.info(f"Loaded accuracy scores for {len(scores)} models from Model Catalog")

    def get_quality_score(self, model_name: str, use_case: str) -> float:
        """Get quality score for a model. use_case is accepted for API compat but not used for weighting."""
        self._ensure_loaded()
        return self._scores.get(model_name.lower(), 0.0)

    def get_available_use_cases(self) -> list[str]:
        """Return available use cases (for API compat with UseCaseQualityScorer)."""
        return ["chatbot_conversational"]  # Model Catalog doesn't have per-use-case weighting
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_model_catalog_quality.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add ModelCatalogQualityScorer for RHOAI accuracy data
```

---

### Task 6: ModelCatalogModelSource

Map Model Catalog models to NeuralNav's `ModelInfo` objects.

**Files:**
- Create: `src/neuralnav/knowledge_base/model_catalog_models.py`
- Test: `tests/unit/test_model_catalog_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_model_catalog_models.py
"""Tests for ModelCatalogModelSource."""
import pytest
from unittest.mock import MagicMock

from neuralnav.knowledge_base.model_catalog_models import ModelCatalogModelSource


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = [
        {
            "name": "RedHatAI/granite-3.1-8b-instruct",
            "provider": "IBM",
            "description": "Granite model",
            "tasks": ["text-to-text"],
            "license": "Apache 2.0",
            "customProperties": {
                "size": {"string_value": "8B params", "metadataType": "MetadataStringValue"},
                "validated": {"string_value": "", "metadataType": "MetadataStringValue"},
                "validated_on": {"string_value": '["RHOAI 2.20"]', "metadataType": "MetadataStringValue"},
            },
        },
    ]
    return client


@pytest.mark.unit
def test_get_all_models(mock_client):
    source = ModelCatalogModelSource(mock_client)
    models = source.get_all_models()
    assert len(models) == 1
    m = models[0]
    assert m.model_id == "RedHatAI/granite-3.1-8b-instruct"
    assert m.provider == "IBM"
    assert m.approval_status == "approved"


@pytest.mark.unit
def test_get_model(mock_client):
    source = ModelCatalogModelSource(mock_client)
    m = source.get_model("RedHatAI/granite-3.1-8b-instruct")
    assert m is not None
    assert m.model_id == "RedHatAI/granite-3.1-8b-instruct"


@pytest.mark.unit
def test_get_model_not_found(mock_client):
    source = ModelCatalogModelSource(mock_client)
    assert source.get_model("unknown/model") is None


@pytest.mark.unit
def test_parse_size_parameters(mock_client):
    source = ModelCatalogModelSource(mock_client)
    models = source.get_all_models()
    assert models[0].size_parameters == "8B"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_catalog_models.py -q`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/neuralnav/knowledge_base/model_catalog_models.py
"""Model metadata source from RHOAI Model Catalog."""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from neuralnav.knowledge_base.model_catalog import ModelInfo

if TYPE_CHECKING:
    from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600

# Map Model Catalog tasks to NeuralNav supported_tasks
_TASK_MAP = {
    "text-to-text": ["chatbot_conversational", "summarization_short", "content_generation", "translation"],
    "text-generation": ["chatbot_conversational", "code_completion", "content_generation"],
}

_SIZE_RE = re.compile(r"([\d.]+)\s*[Bb]\b")


def _parse_size(size_str: str) -> str:
    """Parse '8B params' or '70B params' into '8B' or '70B'."""
    match = _SIZE_RE.search(size_str)
    return match.group(0).upper().replace(" ", "") if match else size_str


def _extract_family(name: str) -> str:
    """Extract model family from name. E.g. 'RedHatAI/granite-3.1-8b-instruct' -> 'granite'."""
    base = name.split("/")[-1].lower()
    for family in ["granite", "llama", "mistral", "qwen", "gemma", "phi", "deepseek", "mixtral", "kimi", "nemotron"]:
        if family in base:
            return family
    return base.split("-")[0]


def _catalog_model_to_model_info(model: dict) -> ModelInfo:
    """Map a Model Catalog model dict to a ModelInfo instance."""
    name = model.get("name", "")
    props = model.get("customProperties", {})

    size_str = props.get("size", {}).get("string_value", "")
    validated_on = props.get("validated_on", {}).get("string_value", "")
    has_validated = "validated" in props or bool(validated_on)

    # Map tasks
    catalog_tasks = model.get("tasks", [])
    supported_tasks = []
    for t in catalog_tasks:
        supported_tasks.extend(_TASK_MAP.get(t, [t]))

    # Estimate min GPU memory from size
    size_parsed = _parse_size(size_str)
    try:
        param_billions = float(size_parsed.rstrip("Bb"))
    except ValueError:
        param_billions = 7.0  # default
    # Rough heuristic: ~2 bytes per param for FP16, ~1 byte for FP8/INT8
    min_gpu_gb = max(8, int(param_billions * 1.5))

    data = {
        "model_id": name,
        "name": name.split("/")[-1],
        "provider": model.get("provider", "Unknown"),
        "family": _extract_family(name),
        "size_parameters": _parse_size(size_str) if size_str else f"{param_billions}B",
        "context_length": 128000,  # Default; not available in Model Catalog
        "supported_tasks": list(set(supported_tasks)),
        "domain_specialization": [],
        "license": model.get("license", "Unknown"),
        "license_type": "permissive" if "apache" in model.get("license", "").lower() else "restricted",
        "min_gpu_memory_gb": min_gpu_gb,
        "recommended_for": supported_tasks,
        "approval_status": "approved" if has_validated else "pending",
    }
    return ModelInfo(data)


class ModelCatalogModelSource:
    """Model metadata from RHOAI Model Catalog.

    Drop-in for ModelCatalog when running with RHOAI.
    """

    def __init__(self, client: ModelCatalogClient):
        self._client = client
        self._models: dict[str, ModelInfo] = {}
        self._loaded_at: float = 0

    def _ensure_loaded(self) -> None:
        if self._models and (time.time() - self._loaded_at) < _CACHE_TTL:
            return
        self._load_all()

    def _load_all(self) -> None:
        models: dict[str, ModelInfo] = {}
        for raw in self._client.list_models():
            info = _catalog_model_to_model_info(raw)
            models[info.model_id] = info
        self._models = models
        self._loaded_at = time.time()
        logger.info(f"Loaded {len(models)} models from Model Catalog")

    def get_model(self, model_id: str) -> ModelInfo | None:
        self._ensure_loaded()
        return self._models.get(model_id)

    def get_all_models(self) -> list[ModelInfo]:
        self._ensure_loaded()
        return [m for m in self._models.values() if m.approval_status == "approved"]

    def find_models_for_use_case(self, use_case: str) -> list[ModelInfo]:
        self._ensure_loaded()
        return [m for m in self._models.values() if use_case in m.recommended_for and m.approval_status == "approved"]

    def find_models_by_task(self, task: str) -> list[ModelInfo]:
        self._ensure_loaded()
        return [m for m in self._models.values() if task in m.supported_tasks and m.approval_status == "approved"]

    def get_gpu_type(self, gpu_type: str):
        """Delegate to local JSON catalog for GPU pricing (not available in Model Catalog)."""
        from neuralnav.knowledge_base.model_catalog import ModelCatalog
        return ModelCatalog().get_gpu_type(gpu_type)

    def calculate_gpu_cost(self, gpu_type: str, gpu_count: int, hours_per_month: float = 730, provider: str | None = None) -> float | None:
        """Delegate to local JSON catalog for GPU pricing."""
        from neuralnav.knowledge_base.model_catalog import ModelCatalog
        return ModelCatalog().calculate_gpu_cost(gpu_type, gpu_count, hours_per_month, provider)

    def get_all_gpu_types(self):
        """Delegate to local JSON catalog."""
        from neuralnav.knowledge_base.model_catalog import ModelCatalog
        return ModelCatalog().get_all_gpu_types()
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_model_catalog_models.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add ModelCatalogModelSource for RHOAI model metadata
```

---

### Task 7: Startup Source Selection in dependencies.py

Wire up the env-var-based selection in the API dependency injection layer.

**Files:**
- Modify: `src/neuralnav/api/dependencies.py`
- Test: `tests/unit/test_dependencies_source_selection.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_dependencies_source_selection.py
"""Test benchmark source selection via environment variable."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "postgresql"}, clear=False)
def test_default_source_is_postgresql():
    """Default source should be PostgreSQL."""
    from neuralnav.api.dependencies import _get_benchmark_source_type
    assert _get_benchmark_source_type() == "postgresql"


@pytest.mark.unit
@patch.dict("os.environ", {"NEURALNAV_BENCHMARK_SOURCE": "model_catalog"}, clear=False)
def test_model_catalog_source():
    from neuralnav.api.dependencies import _get_benchmark_source_type
    assert _get_benchmark_source_type() == "model_catalog"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dependencies_source_selection.py -q`
Expected: FAIL — `ImportError: cannot import name '_get_benchmark_source_type'`

**Step 3: Update dependencies.py**

Add the source selection logic. The key changes:

1. Add `_get_benchmark_source_type()` helper
2. Modify `get_workflow()` to use the selected source
3. Keep all existing functions working unchanged

In `src/neuralnav/api/dependencies.py`, add after the existing imports:

```python
def _get_benchmark_source_type() -> str:
    """Get configured benchmark source type."""
    return os.getenv("NEURALNAV_BENCHMARK_SOURCE", "postgresql")
```

Modify `get_workflow()`:

```python
def get_workflow() -> RecommendationWorkflow:
    """Get the recommendation workflow singleton."""
    global _workflow
    if _workflow is None:
        source_type = _get_benchmark_source_type()
        if source_type == "model_catalog":
            from neuralnav.knowledge_base.model_catalog_client import ModelCatalogClient
            from neuralnav.knowledge_base.model_catalog_benchmarks import ModelCatalogBenchmarkSource
            from neuralnav.knowledge_base.model_catalog_models import ModelCatalogModelSource
            from neuralnav.recommendation.config_finder import ConfigFinder

            client = ModelCatalogClient()
            benchmark_source = ModelCatalogBenchmarkSource(client)
            catalog = ModelCatalogModelSource(client)
            config_finder = ConfigFinder(benchmark_repo=benchmark_source, catalog=catalog)
            logger.info("Using Model Catalog as benchmark source")
            _workflow = RecommendationWorkflow(config_finder=config_finder)
        else:
            logger.info("Using PostgreSQL as benchmark source")
            _workflow = RecommendationWorkflow()
    return _workflow
```

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_dependencies_source_selection.py -q`
Expected: PASS

**Step 5: Commit**

```
feat: add env-var-based benchmark source selection in dependencies
```

---

### Task 8: Wire quality scorer into ConfigFinder for model_catalog source

When using `model_catalog` source, `ConfigFinder.plan_all_capacities()` calls `score_model_quality()` which uses the CSV-based singleton. We need to make the quality scorer configurable too.

**Files:**
- Modify: `src/neuralnav/recommendation/config_finder.py` (add quality_scorer parameter)
- Modify: `src/neuralnav/api/dependencies.py` (pass quality scorer when using model_catalog)
- Test: `tests/unit/test_config_finder_quality_scorer.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_config_finder_quality_scorer.py
"""Test ConfigFinder uses injected quality scorer."""
import pytest
from unittest.mock import MagicMock, patch

from neuralnav.knowledge_base.benchmarks import BenchmarkData
from neuralnav.recommendation.config_finder import ConfigFinder
from neuralnav.shared.schemas import TrafficProfile, SLOTargets, DeploymentIntent


def _make_bench():
    return BenchmarkData({
        "model_hf_repo": "RedHatAI/test-model", "hardware": "H100", "hardware_count": 1,
        "framework": "vllm", "framework_version": "0.8.4",
        "prompt_tokens": 512, "output_tokens": 256,
        "mean_input_tokens": 512, "mean_output_tokens": 256,
        "ttft_mean": 50, "ttft_p90": 70, "ttft_p95": 80, "ttft_p99": 100,
        "itl_mean": 10, "itl_p90": 15, "itl_p95": 20, "itl_p99": 25,
        "e2e_mean": 3000, "e2e_p90": 4000, "e2e_p95": 5000, "e2e_p99": 6000,
        "tps_mean": 1000, "tps_p90": 900, "tps_p95": 800, "tps_p99": 700,
        "tokens_per_second": 1000, "requests_per_second": 7,
    })


@pytest.mark.unit
def test_config_finder_uses_injected_quality_scorer():
    mock_source = MagicMock()
    mock_source.find_configurations_meeting_slo.return_value = [_make_bench()]

    mock_catalog = MagicMock()
    mock_catalog.get_all_models.return_value = []
    mock_catalog.calculate_gpu_cost.return_value = 2.70

    mock_scorer = MagicMock()
    mock_scorer.get_quality_score.return_value = 75.0

    finder = ConfigFinder(
        benchmark_repo=mock_source,
        catalog=mock_catalog,
        quality_scorer=mock_scorer,
    )

    traffic = TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=5)
    slo = SLOTargets(ttft_p95_target_ms=200, itl_p95_target_ms=30, e2e_p95_target_ms=8000)
    intent = DeploymentIntent(use_case="chatbot_conversational", user_count=100)

    results = finder.plan_all_capacities(traffic, slo, intent)
    assert len(results) > 0
    mock_scorer.get_quality_score.assert_called()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config_finder_quality_scorer.py -q`
Expected: FAIL — `TypeError: ConfigFinder.__init__() got an unexpected keyword argument 'quality_scorer'`

**Step 3: Implement**

Add `quality_scorer` parameter to `ConfigFinder.__init__`:

```python
def __init__(
    self,
    benchmark_repo: BenchmarkSource | None = None,
    catalog: ModelCatalog | None = None,
    quality_scorer=None,
):
    if benchmark_repo is not None:
        self.benchmark_repo = benchmark_repo
    else:
        from neuralnav.knowledge_base.benchmarks import BenchmarkRepository
        self.benchmark_repo = BenchmarkRepository()
    self.catalog = catalog or ModelCatalog()
    self._quality_scorer = quality_scorer
```

In `plan_all_capacities()`, replace the direct `score_model_quality()` call (around line 252-260):

```python
# Before:
from .quality import score_model_quality
model_name_for_scoring = model.name if model else bench.model_hf_repo
raw_accuracy = score_model_quality(model_name_for_scoring, intent.use_case)
if raw_accuracy == 0 and bench.model_hf_repo:
    raw_accuracy = score_model_quality(bench.model_hf_repo, intent.use_case)

# After:
model_name_for_scoring = model.name if model else bench.model_hf_repo
if self._quality_scorer is not None:
    raw_accuracy = self._quality_scorer.get_quality_score(model_name_for_scoring, intent.use_case)
    if raw_accuracy == 0 and bench.model_hf_repo:
        raw_accuracy = self._quality_scorer.get_quality_score(bench.model_hf_repo, intent.use_case)
else:
    from .quality import score_model_quality
    raw_accuracy = score_model_quality(model_name_for_scoring, intent.use_case)
    if raw_accuracy == 0 and bench.model_hf_repo:
        raw_accuracy = score_model_quality(bench.model_hf_repo, intent.use_case)
```

Update `dependencies.py` to pass the quality scorer:

```python
# In the model_catalog branch of get_workflow():
from neuralnav.knowledge_base.model_catalog_quality import ModelCatalogQualityScorer
quality_scorer = ModelCatalogQualityScorer(client)
config_finder = ConfigFinder(benchmark_repo=benchmark_source, catalog=catalog, quality_scorer=quality_scorer)
```

**Step 4: Run all tests**

Run: `uv run pytest tests/unit/ -q`
Expected: PASS

**Step 5: Commit**

```
feat: make quality scorer injectable in ConfigFinder for RHOAI source
```

---

### Task 9: Integration Test with Live Cluster (Optional)

A manual integration test that verifies the full flow against a real RHOAI cluster. Skip in CI. Run manually with `oc port-forward`.

**Files:**
- Create: `tests/integration/test_model_catalog_integration.py`

**Step 1: Write the test**

```python
# tests/integration/test_model_catalog_integration.py
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
        prompt_tokens=512, output_tokens=256,
        ttft_p95_max_ms=500, itl_p95_max_ms=100, e2e_p95_max_ms=30000,
    )
    assert len(results) > 0
    for bench in results:
        assert bench.ttft_p95 <= 500
        assert bench.itl_p95 <= 100
        assert bench.e2e_p95 <= 30000
```

**Step 2: Run unit tests only (integration skipped without env vars)**

Run: `uv run pytest tests/ -q -m "not integration"`
Expected: PASS (integration tests skipped)

**Step 3: Commit**

```
test: add integration test for Model Catalog benchmark source
```

---

### Summary

| Task | What it builds | New files | Modified files |
|------|---------------|-----------|---------------|
| 1 | `BenchmarkSource` protocol | `benchmark_source.py`, test | — |
| 2 | Update ConfigFinder type hints | test | `config_finder.py` |
| 3 | Model Catalog HTTP client | `model_catalog_client.py`, test | — |
| 4 | Performance benchmark source | `model_catalog_benchmarks.py`, test | — |
| 5 | Accuracy quality scorer | `model_catalog_quality.py`, test | — |
| 6 | Model metadata source | `model_catalog_models.py`, test | — |
| 7 | Startup source selection | test | `dependencies.py` |
| 8 | Injectable quality scorer | test | `config_finder.py`, `dependencies.py` |
| 9 | Integration test (optional) | test | — |
