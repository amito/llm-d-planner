"""Tests for planner quality scoring helper."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from planner.recommendation.quality.scoring import (
    build_scoring_engine,
    compute_quality_score,
    load_quality_weights,
    validate_quality_weights,
)


def _make_mock_scorecard():
    """Create a mock scorecard with known percentiles."""
    from quality_scoring.models import (
        CompositeScore,
        MatchType,
        ModelScorecard,
        NormalizedScore,
    )

    def _cs(category: str, percentile: float) -> CompositeScore:
        return CompositeScore(
            category=category,
            percentile=percentile,
            arena_score=NormalizedScore(
                raw_score=1400.0,
                percentile=percentile,
                tied_rank=1,
                population_size=100,
                source="arena",
            ),
            aa_score=None,
        )

    return ModelScorecard(
        model_name="test-model",
        arena_name="test-model",
        aa_name=None,
        arena_match_type=MatchType.EXACT,
        aa_match_type=MatchType.NONE,
        overall=_cs("overall", 80.0),
        categories={
            "overall": _cs("overall", 80.0),
            "coding": _cs("coding", 90.0),
            "math": _cs("math", 70.0),
            "creative_writing": _cs("creative_writing", 60.0),
            "instruction_following": _cs("instruction_following", 85.0),
        },
    )


class TestComputeQualityScore:
    def test_weighted_average(self):
        sc = _make_mock_scorecard()
        score = compute_quality_score(sc, {"coding": 6, "math": 4})
        expected = (90.0 * 6 + 70.0 * 4) / (6 + 4)
        assert score == round(expected, 2)

    def test_missing_category_uses_overall_fill_in(self):
        sc = _make_mock_scorecard()
        # "nonexistent" is missing — should use overall percentile (80.0) as fill-in
        score = compute_quality_score(sc, {"coding": 5, "nonexistent": 5})
        expected = (90.0 * 5 + 80.0 * 5) / (5 + 5)  # 85.0
        assert score == round(expected, 2)

    def test_all_categories_missing_uses_overall_for_all(self):
        sc = _make_mock_scorecard()
        score = compute_quality_score(sc, {"nonexistent_a": 5, "nonexistent_b": 5})
        assert score == 80.0  # all fill-ins use overall

    def test_empty_weights_returns_overall(self):
        sc = _make_mock_scorecard()
        score = compute_quality_score(sc, {})
        assert score == 80.0

    def test_no_overall_returns_zero(self):
        from quality_scoring.models import MatchType, ModelScorecard

        sc = ModelScorecard(
            model_name="test",
            arena_name=None,
            aa_name=None,
            arena_match_type=MatchType.NONE,
            aa_match_type=MatchType.NONE,
            overall=None,
            categories={},
        )
        score = compute_quality_score(sc, {"coding": 1})
        assert score == 0.0

    def test_missing_category_with_no_overall_uses_zero(self):
        from quality_scoring.models import MatchType, ModelScorecard

        sc = ModelScorecard(
            model_name="test",
            arena_name=None,
            aa_name=None,
            arena_match_type=MatchType.NONE,
            aa_match_type=MatchType.NONE,
            overall=None,
            categories={},
        )
        score = compute_quality_score(sc, {"coding": 5, "math": 5})
        assert score == 0.0  # no overall to fill in, no categories match


class TestLoadQualityWeights:
    def test_loads_from_json(self, tmp_path):
        weights = {"chatbot": {"categories": {"overall": 5, "coding": 5}}}
        p = tmp_path / "weights.json"
        p.write_text(json.dumps(weights))
        result = load_quality_weights(p)
        assert result["chatbot"]["categories"]["overall"] == 5

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_quality_weights(tmp_path / "nonexistent.json")
        assert result == {}


class TestValidateQualityWeights:
    def test_valid_categories_no_warnings(self, caplog):
        weights = {"test_case": {"categories": {"overall": 5, "coding": 5}}}
        validate_quality_weights(weights)
        assert "Unknown category" not in caplog.text

    def test_unknown_category_logs_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            weights = {"test_case": {"categories": {"bogus_category": 1}}}
            validate_quality_weights(weights)
        assert "bogus_category" in caplog.text


@pytest.mark.unit
class TestBuildScoringEngine:
    """Tests for build_scoring_engine() covering auto-update, cache, and fallback paths."""

    @patch("planner.recommendation.quality.scoring.arena_client")
    @patch("planner.recommendation.quality.scoring.aa_client")
    def test_default_loads_checked_in_data(self, mock_aa, mock_arena, tmp_path):
        """With auto-update disabled, loads from checked-in data/ directory."""
        # Ensure auto-update is off
        with patch.dict(os.environ, {"QUALITY_AUTO_UPDATE": "false"}, clear=False):
            engine, metadata = build_scoring_engine()

        # Clients should not have been called for sync
        mock_arena.sync.assert_not_called()
        mock_aa.sync.assert_not_called()
        assert isinstance(metadata, dict)
        assert "arena_count" in metadata
        assert "aa_count" in metadata

    @patch("planner.recommendation.quality.scoring.arena_client")
    @patch("planner.recommendation.quality.scoring.aa_client")
    def test_auto_update_disabled_ignores_runtime_cache(self, mock_aa, mock_arena, tmp_path):
        """Auto-update disabled: uses checked-in data even if runtime cache exists."""
        runtime_cache = tmp_path / "runtime_cache"
        runtime_cache.mkdir()
        # Write a fake cache file
        arena_cache = runtime_cache / "arena_models.json"
        arena_cache.write_text(
            json.dumps({"rows": [{"model": "cached"}], "fetched_at": "2026-01-01T00:00:00"})
        )

        with patch.dict(os.environ, {"QUALITY_AUTO_UPDATE": "false"}, clear=False):
            engine, metadata = build_scoring_engine(cache_dir=runtime_cache)

        # Should NOT have called load_cache on clients (auto-update is off)
        mock_arena.load_cache.assert_not_called()
        mock_aa.load_cache.assert_not_called()

    @patch("planner.recommendation.quality.scoring.arena_client")
    @patch("planner.recommendation.quality.scoring.aa_client")
    @patch("planner.recommendation.quality.scoring.is_cache_stale", return_value=True)
    def test_auto_update_with_stale_cache_calls_sync(
        self, mock_stale, mock_aa, mock_arena, tmp_path
    ):
        """Auto-update enabled, stale cache: calls sync, loads from runtime cache."""
        runtime_cache = tmp_path / "runtime_cache"

        mock_arena.load_cache.return_value = ([], None)
        mock_arena.sync.return_value = (10, runtime_cache / "arena_models.json")
        # After sync, load_cache returns fresh data
        mock_arena.load_cache.side_effect = [
            ([], None),  # First call: empty
            ([{"model": "synced"}], "2026-07-12T00:00:00"),  # After sync
        ]

        mock_aa.load_cache.return_value = ([], None)

        with patch.dict(
            os.environ,
            {"QUALITY_AUTO_UPDATE": "true", "AA_API_KEY": ""},
            clear=False,
        ):
            engine, metadata = build_scoring_engine(cache_dir=runtime_cache)

        mock_arena.sync.assert_called_once_with(cache_dir=runtime_cache)

    @patch("planner.recommendation.quality.scoring.arena_client")
    @patch("planner.recommendation.quality.scoring.aa_client")
    @patch("planner.recommendation.quality.scoring.is_cache_stale", return_value=False)
    def test_auto_update_with_fresh_cache_skips_sync(
        self, mock_stale, mock_aa, mock_arena, tmp_path
    ):
        """Auto-update enabled, fresh cache: uses cached data without syncing."""
        runtime_cache = tmp_path / "runtime_cache"

        mock_arena.load_cache.return_value = (
            [{"model": "cached"}],
            "2026-07-12T00:00:00",
        )
        mock_aa.load_cache.return_value = (
            [{"name": "cached-model"}],
            "2026-07-12T00:00:00",
        )

        with patch.dict(
            os.environ,
            {"QUALITY_AUTO_UPDATE": "true"},
            clear=False,
        ):
            engine, metadata = build_scoring_engine(cache_dir=runtime_cache)

        mock_arena.sync.assert_not_called()
        mock_aa.sync.assert_not_called()

    @patch("planner.recommendation.quality.scoring.arena_client")
    @patch("planner.recommendation.quality.scoring.aa_client")
    @patch("planner.recommendation.quality.scoring.is_cache_stale", return_value=True)
    def test_auto_update_sync_fails_falls_back(self, mock_stale, mock_aa, mock_arena, tmp_path):
        """Auto-update enabled, sync fails: falls back to checked-in data."""
        runtime_cache = tmp_path / "runtime_cache"

        mock_arena.load_cache.return_value = ([], None)
        mock_arena.sync.side_effect = Exception("network error")

        mock_aa.load_cache.return_value = ([], None)

        with patch.dict(
            os.environ,
            {"QUALITY_AUTO_UPDATE": "true"},
            clear=False,
        ):
            engine, metadata = build_scoring_engine(cache_dir=runtime_cache)

        # Should still succeed (falls back to checked-in data)
        assert metadata["arena_count"] >= 0
        assert metadata["aa_count"] >= 0

    def test_no_environ_mutation(self):
        """Verify that build_scoring_engine() does not mutate os.environ."""
        original_env = os.environ.get("LLM_QUALITY_CACHE_DIR")

        with patch.dict(os.environ, {"QUALITY_AUTO_UPDATE": "false"}, clear=False):
            build_scoring_engine()

        after_env = os.environ.get("LLM_QUALITY_CACHE_DIR")
        assert original_env == after_env, (
            f"LLM_QUALITY_CACHE_DIR was mutated: {original_env!r} -> {after_env!r}"
        )
