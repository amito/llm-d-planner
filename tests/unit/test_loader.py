"""Unit tests for planner.knowledge_base.loader functions."""

from unittest.mock import MagicMock, patch

import pytest

from planner.knowledge_base.loader import (
    ensure_schema,
    extract_metadata,
    generate_config_id,
    normalize_benchmark_fields,
    prepare_benchmark_for_insert,
)


def _minimal_benchmark(**overrides):
    """Return a minimal benchmark dict with required fields."""
    base = {
        "model_hf_repo": "meta-llama/Llama-3-8B",
        "hardware": "H100",
        "hardware_count": 1,
        "prompt_tokens": 512,
        "output_tokens": 256,
        "requests_per_second": 10,
    }
    base.update(overrides)
    return base


# --- generate_config_id ---


@pytest.mark.unit
class TestGenerateConfigId:
    def test_deterministic(self):
        b = _minimal_benchmark()
        assert generate_config_id(b) == generate_config_id(b)

    def test_different_model_different_id(self):
        a = _minimal_benchmark(model_hf_repo="model-a")
        b = _minimal_benchmark(model_hf_repo="model-b")
        assert generate_config_id(a) != generate_config_id(b)

    def test_different_hardware_different_id(self):
        a = _minimal_benchmark(hardware="H100")
        b = _minimal_benchmark(hardware="A100")
        assert generate_config_id(a) != generate_config_id(b)

    def test_different_tokens_different_id(self):
        a = _minimal_benchmark(prompt_tokens=512)
        b = _minimal_benchmark(prompt_tokens=1024)
        assert generate_config_id(a) != generate_config_id(b)

    def test_length_is_32(self):
        assert len(generate_config_id(_minimal_benchmark())) == 32

    def test_missing_rps_uses_empty_string(self):
        b = _minimal_benchmark()
        del b["requests_per_second"]
        cid = generate_config_id(b)
        assert len(cid) == 32


# --- normalize_benchmark_fields ---


@pytest.mark.unit
class TestNormalizeBenchmarkFields:
    def test_passthrough_when_canonical(self):
        b = _minimal_benchmark(mean_input_tokens=512, mean_output_tokens=256)
        assert normalize_benchmark_fields(b) == b

    def test_maps_model_id_to_model_hf_repo(self):
        b = {"model_id": "my-model", "hardware": "H100"}
        result = normalize_benchmark_fields(b)
        assert result["model_hf_repo"] == "my-model"

    def test_does_not_overwrite_existing_model_hf_repo(self):
        b = {"model_hf_repo": "original", "model_id": "other"}
        result = normalize_benchmark_fields(b)
        assert result["model_hf_repo"] == "original"

    def test_maps_hardware_type_to_hardware(self):
        b = {"model_hf_repo": "m", "hardware_type": "A100"}
        result = normalize_benchmark_fields(b)
        assert result["hardware"] == "A100"

    def test_maps_gpu_type_to_hardware(self):
        b = {"model_hf_repo": "m", "gpu_type": "L4"}
        result = normalize_benchmark_fields(b)
        assert result["hardware"] == "L4"

    def test_hardware_type_preferred_over_gpu_type(self):
        b = {"model_hf_repo": "m", "hardware_type": "A100", "gpu_type": "L4"}
        result = normalize_benchmark_fields(b)
        assert result["hardware"] == "A100"

    def test_maps_tokens_per_second_mean(self):
        b = {"model_hf_repo": "m", "hardware": "H100", "tokens_per_second_mean": 42.5}
        result = normalize_benchmark_fields(b)
        assert result["tokens_per_second"] == 42.5

    def test_maps_mean_input_output_tokens(self):
        b = {"model_hf_repo": "m", "hardware": "H100", "prompt_tokens": 128, "output_tokens": 64}
        result = normalize_benchmark_fields(b)
        assert result["mean_input_tokens"] == 128
        assert result["mean_output_tokens"] == 64

    def test_does_not_mutate_original(self):
        b = {"model_id": "m", "hardware": "H100"}
        original_keys = set(b.keys())
        normalize_benchmark_fields(b)
        assert set(b.keys()) == original_keys


# --- prepare_benchmark_for_insert ---


@pytest.mark.unit
class TestPrepareBenchmarkForInsert:
    def test_adds_id_and_config_id(self):
        result = prepare_benchmark_for_insert(_minimal_benchmark())
        assert "id" in result
        assert "config_id" in result
        assert len(result["config_id"]) == 32

    def test_unique_ids_per_call(self):
        b = _minimal_benchmark()
        r1 = prepare_benchmark_for_insert(b)
        r2 = prepare_benchmark_for_insert(b)
        assert r1["id"] != r2["id"]

    def test_applies_source_and_confidence(self):
        result = prepare_benchmark_for_insert(
            _minimal_benchmark(), source="blis", confidence_level="benchmarked"
        )
        assert result["source"] == "blis"
        assert result["confidence_level"] == "benchmarked"

    def test_defaults_source_and_confidence(self):
        result = prepare_benchmark_for_insert(_minimal_benchmark())
        assert result["source"] == "local"
        assert result["confidence_level"] == "estimated"

    def test_sets_timestamps(self):
        result = prepare_benchmark_for_insert(_minimal_benchmark())
        assert result["created_at"] is not None
        assert result["updated_at"] is not None
        assert result["jbenchmark_created_at"] is not None

    def test_optional_fields_default_to_none(self):
        result = prepare_benchmark_for_insert(_minimal_benchmark())
        for field in [
            "framework",
            "framework_version",
            "docker_image",
            "entrypoint",
            "profiler_type",
            "profiler_image",
            "profiler_tag",
            "model_uri",
        ]:
            assert result[field] is None, f"{field} should default to None"

    def test_normalizes_field_names(self):
        b = {
            "model_id": "m",
            "gpu_type": "L4",
            "hardware_count": 1,
            "prompt_tokens": 512,
            "output_tokens": 256,
        }
        result = prepare_benchmark_for_insert(b)
        assert result["model_hf_repo"] == "m"
        assert result["hardware"] == "L4"


# --- extract_metadata ---


@pytest.mark.unit
class TestExtractMetadata:
    def test_extracts_from_underscore_metadata(self):
        data = {
            "_metadata": {"source": "blis", "confidence_level": "benchmarked"},
            "benchmarks": [],
        }
        result = extract_metadata(data)
        assert result["source"] == "blis"
        assert result["confidence_level"] == "benchmarked"

    def test_extracts_from_metadata(self):
        data = {
            "metadata": {"source": "estimated", "confidence_level": "estimated"},
            "benchmarks": [],
        }
        result = extract_metadata(data)
        assert result["source"] == "estimated"
        assert result["confidence_level"] == "estimated"

    def test_underscore_metadata_takes_precedence(self):
        data = {
            "_metadata": {"source": "from_underscore"},
            "metadata": {"source": "from_plain"},
        }
        result = extract_metadata(data)
        assert result["source"] == "from_underscore"

    def test_returns_none_when_no_metadata(self):
        result = extract_metadata({"benchmarks": []})
        assert result["source"] is None
        assert result["confidence_level"] is None

    def test_returns_none_for_missing_fields(self):
        result = extract_metadata({"_metadata": {"other_field": "value"}})
        assert result["source"] is None
        assert result["confidence_level"] is None


# --- ensure_schema ---


@pytest.mark.unit
class TestEnsureSchema:
    def test_executes_bundled_schema(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        ensure_schema(mock_conn)

        mock_cursor.executescript.assert_called_once()
        mock_conn.commit.assert_called_once()
        schema_sql = mock_cursor.executescript.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS exported_summaries" in schema_sql
