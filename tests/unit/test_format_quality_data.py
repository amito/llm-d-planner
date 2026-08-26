from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "format_quality_data",
    Path(__file__).resolve().parents[2] / "scripts" / "format_quality_data.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
# Prevent top-level fmt() calls from executing during import by stubbing is_file.
_orig_is_file = Path.is_file
Path.is_file = lambda self: False  # type: ignore[assignment]
try:
    _spec.loader.exec_module(_mod)
finally:
    Path.is_file = _orig_is_file  # type: ignore[assignment]
fmt = _mod.fmt


@pytest.mark.unit
class TestFmt:
    def test_sorts_models_by_key(self, tmp_path: Path) -> None:
        data = {"models": [{"slug": "zeta"}, {"slug": "alpha"}, {"slug": "mid"}]}
        f = tmp_path / "models.json"
        f.write_text(json.dumps(data))

        fmt(str(f), sort_key=lambda m: m.get("slug") or "")

        result = json.loads(f.read_text())
        assert [m["slug"] for m in result["models"]] == ["alpha", "mid", "zeta"]

    def test_sorts_rows_by_composite_key(self, tmp_path: Path) -> None:
        data = {
            "rows": [
                {"model_name": "b", "category": "coding"},
                {"model_name": "a", "category": "overall"},
                {"model_name": "a", "category": "coding"},
            ]
        }
        f = tmp_path / "rows.json"
        f.write_text(json.dumps(data))

        fmt(str(f), sort_key=lambda r: (r.get("model_name") or "", r.get("category") or ""))

        result = json.loads(f.read_text())
        names = [(r["model_name"], r["category"]) for r in result["rows"]]
        assert names == [("a", "coding"), ("a", "overall"), ("b", "coding")]

    def test_strips_keys(self, tmp_path: Path) -> None:
        data = {"models": [{"slug": "a", "accessed_date": "2026-01-01", "score": 42}]}
        f = tmp_path / "models.json"
        f.write_text(json.dumps(data))

        fmt(str(f), strip_keys={"accessed_date"})

        result = json.loads(f.read_text())
        assert result["models"] == [{"slug": "a", "score": 42}]

    def test_preserves_keys_not_in_strip_set(self, tmp_path: Path) -> None:
        data = {"models": [{"slug": "a", "name": "A", "score": 10}]}
        f = tmp_path / "models.json"
        f.write_text(json.dumps(data))

        fmt(str(f), strip_keys={"nonexistent"})

        result = json.loads(f.read_text())
        assert result["models"] == [{"slug": "a", "name": "A", "score": 10}]

    def test_missing_file_warns(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fmt(str(tmp_path / "nonexistent.json"))

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "nonexistent.json" in captured.err

    def test_writes_formatted_json_with_trailing_newline(self, tmp_path: Path) -> None:
        data = {"models": [{"slug": "a"}]}
        f = tmp_path / "models.json"
        f.write_text(json.dumps(data))

        fmt(str(f))

        text = f.read_text()
        assert text.endswith("\n")
        assert "  " in text  # indent=2
