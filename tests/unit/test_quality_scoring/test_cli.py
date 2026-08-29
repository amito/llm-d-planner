"""Tests for scripts/check_model_resolution.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_model_resolution.py"
_spec = importlib.util.spec_from_file_location("check_model_resolution", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
main = _mod.main


@pytest.fixture()
def catalog_file(tmp_path):
    catalog = {
        "models": [
            {"model_id": "model-a"},
            {"model_id": "model-b"},
        ]
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    return path


@pytest.fixture()
def full_cache(tmp_path):
    """Cache dir with both Arena and AA data."""
    arena = {
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "row_count": 2,
        "rows": [
            {
                "model_name": "model-a",
                "category": "overall",
                "rating": 1200,
                "rating_lower": 1190,
                "rating_upper": 1210,
            },
            {
                "model_name": "fuzzy-match-target",
                "category": "overall",
                "rating": 1100,
                "rating_lower": 1090,
                "rating_upper": 1110,
            },
        ],
    }
    aa = {
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "model_count": 1,
        "models": [
            {
                "name": "Model A",
                "slug": "model-a",
                "intelligence_index": 80,
                "coding_index": 75,
                "agentic_index": 70,
            },
        ],
    }
    (tmp_path / "arena_models.json").write_text(json.dumps(arena))
    (tmp_path / "aa_models.json").write_text(json.dumps(aa))
    return tmp_path


@pytest.mark.unit
class TestCliHelp:
    def test_help_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "check_model_resolution" in captured.out
        assert "--catalog" in captured.out
        assert "--models" in captured.out
        assert "--fuzzy" in captured.out

    def test_mutually_exclusive_inputs(self, capsys, catalog_file):
        with pytest.raises(SystemExit) as exc_info:
            main(["--catalog", str(catalog_file), "-m", "model-a"])
        assert exc_info.value.code != 0

    def test_requires_input(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0


@pytest.mark.unit
class TestCliWithModelsFlag:
    def test_resolves_exact_match(self, capsys, full_cache):
        main(["-m", "model-a", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert '"model-a" -> model-a (exact)' in out

    def test_not_found_shows_suggestion(self, capsys, full_cache):
        main(["-m", "totally-unknown", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "not found" in out

    def test_multiple_models(self, capsys, full_cache):
        main(["-m", "model-a,totally-unknown", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "Checking 2 model(s)" in out
        assert "model-a" in out
        assert "not found" in out

    def test_summary_counts(self, capsys, full_cache):
        main(["-m", "model-a,totally-unknown", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "Summary: 2 models checked" in out


@pytest.mark.unit
class TestCliWithCatalog:
    def test_loads_from_catalog(self, capsys, catalog_file, full_cache):
        main(["--catalog", str(catalog_file), "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "Checking 2 model(s)" in out
        assert '"model-a"' in out
        assert '"model-b"' in out

    def test_missing_catalog_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--catalog", "/nonexistent/catalog.json"])
        assert exc_info.value.code == 1


@pytest.mark.unit
class TestCliFuzzyFlag:
    def test_fuzzy_off_rejects_fuzzy(self, capsys, full_cache):
        main(["-m", "fuzzy-match-targe", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "not found" in out or "fuzzy, not accepted" in out

    def test_fuzzy_on_accepts_fuzzy(self, capsys, full_cache):
        main(["-m", "model-a-instruct", "--fuzzy", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        lines_with_model = [ln for ln in out.splitlines() if '"model-a-instruct"' in ln]
        fuzzy_lines = [ln for ln in lines_with_model if "fuzzy)" in ln]
        assert len(fuzzy_lines) > 0, "Expected at least one fuzzy match line"
        for line in fuzzy_lines:
            assert "not accepted" not in line


@pytest.mark.unit
class TestCliFallbackToBundled:
    def test_loads_bundled_when_no_cache(self, capsys, tmp_path):
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        main(["-m", "gpt-4o", "--cache-dir", str(empty_cache)])
        out = capsys.readouterr().out
        assert "Arena" in out
        assert "known models" in out
        assert "0 known models" not in out

    def test_warns_when_cache_dir_falls_back(self, capsys, tmp_path):
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()
        main(["-m", "gpt-4o", "--cache-dir", str(empty_cache)])
        err = capsys.readouterr().err
        assert "Warning: no Arena data found in" in err
        assert "Warning: no AA data found in" in err


@pytest.mark.unit
class TestCliCatalogErrors:
    def test_malformed_json_exits(self, capsys, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not json{{{")
        with pytest.raises(SystemExit) as exc_info:
            main(["--catalog", str(bad_json)])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not valid JSON" in err

    def test_missing_models_key_exits(self, capsys, tmp_path):
        no_models = tmp_path / "no_models.json"
        no_models.write_text('{"data": []}')
        with pytest.raises(SystemExit) as exc_info:
            main(["--catalog", str(no_models)])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert '"models" array' in err

    def test_missing_model_id_exits(self, capsys, tmp_path):
        no_id = tmp_path / "no_id.json"
        no_id.write_text('{"models": [{"name": "foo"}]}')
        with pytest.raises(SystemExit) as exc_info:
            main(["--catalog", str(no_id)])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert '"model_id" field' in err


@pytest.mark.unit
class TestCliSourceHeaders:
    def test_prints_both_source_headers(self, capsys, full_cache):
        main(["-m", "model-a", "--cache-dir", str(full_cache)])
        out = capsys.readouterr().out
        assert "=== Arena" in out
        assert "=== Artificial Analysis" in out
