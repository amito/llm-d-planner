"""Unit tests for data_path resolver."""

import pytest

from planner.data._resolver import data_path


@pytest.mark.unit
class TestDataPath:
    def test_resolves_json_config(self):
        path = data_path("configuration/model_catalog.json")
        assert path.exists()
        assert path.name == "model_catalog.json"

    def test_resolves_csv_weighted_scores(self):
        path = data_path("benchmarks/accuracy/weighted_scores/opensource_chatbot_conversational.csv")
        assert path.exists()
        assert path.name == "opensource_chatbot_conversational.csv"

    def test_resolves_benchmark_json(self):
        path = data_path("benchmarks/performance/benchmarks_BLIS.json")
        assert path.exists()
        assert path.name == "benchmarks_BLIS.json"

    def test_resolves_directory(self):
        path = data_path("configuration")
        assert path.is_dir()

    def test_returns_path_type(self):
        from pathlib import Path
        result = data_path("configuration/model_catalog.json")
        assert isinstance(result, Path)

    def test_nonexistent_file_returns_path(self):
        """data_path does not validate existence -- callers handle that."""
        path = data_path("nonexistent/file.json")
        assert not path.exists()
