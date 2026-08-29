from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quality_scoring.aa_client import (
    _MAX_PAGES,
    _map_api_model,
    cache_age_display,
    compute_distribution,
    fetch_from_api,
    is_cache_stale,
    load_cache,
    load_dist_cache,
    save_cache,
    save_dist_cache,
)


@pytest.mark.unit
class TestMapApiModel:
    def test_basic_mapping(self) -> None:
        api_obj = {
            "name": "Test Model",
            "slug": "test-model",
            "release_date": "2025-06-15",
            "model_creator": {"name": "TestOrg"},
            "evaluations": {
                "artificial_analysis_intelligence_index": 42,
                "artificial_analysis_coding_index": 35,
                "artificial_analysis_agentic_index": 30,
            },
            "performance": {
                "median_output_tokens_per_second": 100.5,
                "median_time_to_first_token_seconds": 0.85,
            },
            "pricing": {
                "price_1m_input_tokens": 1.0,
                "price_1m_output_tokens": 3.0,
            },
        }
        result = _map_api_model(api_obj)
        assert result["name"] == "Test Model"
        assert result["slug"] == "test-model"
        assert result["intelligence_index"] == 42
        assert result["coding_index"] == 35
        assert result["agentic_index"] == 30
        # All original fields preserved, plus promoted index fields
        assert result["release_date"] == "2025-06-15"
        assert result["model_creator"] == {"name": "TestOrg"}
        assert result["performance"]["median_output_tokens_per_second"] == 100.5
        assert result["pricing"]["price_1m_input_tokens"] == 1.0

    def test_missing_fields(self) -> None:
        api_obj = {"name": "Minimal", "slug": "minimal"}
        result = _map_api_model(api_obj)
        assert result["name"] == "Minimal"
        assert result["intelligence_index"] is None
        assert result["coding_index"] is None
        assert result["agentic_index"] is None


@pytest.mark.unit
class TestFetchFromApi:
    def _mock_response(self, data: list[dict], has_more: bool = False, page: int = 1) -> Any:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "tier": "free",
            "pagination": {"page": page, "page_size": 50, "total_pages": 2, "has_more": has_more},
            "data": data,
        }
        return resp

    def test_single_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = self._mock_response(
            [{"name": "A", "slug": "a"}, {"name": "B", "slug": "b"}],
            has_more=False,
        )
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        result = fetch_from_api("test-key")
        assert len(result) == 2
        assert result[0]["name"] == "A"
        client.get.assert_called_once()

    def test_multi_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        page1 = self._mock_response([{"name": "A", "slug": "a"}], has_more=True, page=1)
        page2 = self._mock_response([{"name": "B", "slug": "b"}], has_more=False, page=2)

        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = [page1, page2]
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        result = fetch_from_api("test-key")
        assert len(result) == 2
        assert result[0]["name"] == "A"
        assert result[1]["name"] == "B"
        assert client.get.call_count == 2

    def test_auth_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 401
        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = resp
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        with pytest.raises(RuntimeError, match="401"):
            fetch_from_api("bad-key")

    def test_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 429
        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = resp
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        with pytest.raises(RuntimeError, match="429"):
            fetch_from_api("test-key")

    def test_max_pages_cap(self) -> None:
        assert _MAX_PAGES == 100

    def test_unexpected_response_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{"name": "A"}]  # bare list, not {"data": [...]}
        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = resp
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        with pytest.raises(RuntimeError, match="Unexpected API response structure"):
            fetch_from_api("test-key")

    def test_data_field_not_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": "not-a-list"}
        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.return_value = resp
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        with pytest.raises(RuntimeError, match="Expected list"):
            fetch_from_api("test-key")

    def test_multi_page_correct_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        page1 = self._mock_response([{"name": "A", "slug": "a"}], has_more=True, page=1)
        page2 = self._mock_response([{"name": "B", "slug": "b"}], has_more=False, page=2)

        client = MagicMock()
        client.__enter__ = lambda s: s
        client.__exit__ = MagicMock(return_value=False)
        client.get.side_effect = [page1, page2]
        monkeypatch.setattr("httpx.Client", lambda **kw: client)

        fetch_from_api("test-key")

        calls = client.get.call_args_list
        assert calls[0].kwargs["params"] == {"page": 1}
        assert calls[1].kwargs["params"] == {"page": 2}


@pytest.mark.unit
class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_QUALITY_CACHE_DIR", str(tmp_path))
        models = [{"name": "Test", "slug": "test", "organization": "Org", "intelligence_index": 30}]
        path = save_cache(models)
        assert path.exists()

        loaded, fetched_at = load_cache()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Test"
        assert fetched_at is not None

    def test_load_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_QUALITY_CACHE_DIR", str(tmp_path))
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None

    def test_load_corrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_QUALITY_CACHE_DIR", str(tmp_path))
        cache_dir = tmp_path / ".model_cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "aa_models.json"
        cache_file.write_text("not json{{{")
        loaded, fetched_at = load_cache()
        assert loaded == []
        assert fetched_at is None


@pytest.mark.unit
class TestCacheAgeDisplay:
    def test_just_now(self) -> None:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        assert cache_age_display(now) == "just now"

    def test_invalid(self) -> None:
        assert cache_age_display("not-a-date") == "unknown age"

    def test_hours_ago(self) -> None:
        from datetime import UTC, datetime, timedelta

        two_hours_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert cache_age_display(two_hours_ago) == "2 hours ago"

    def test_days_ago(self) -> None:
        from datetime import UTC, datetime, timedelta

        three_days_ago = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        assert cache_age_display(three_days_ago) == "3 days ago"


@pytest.mark.unit
class TestIsCacheStale:
    def test_none_is_stale(self) -> None:
        assert is_cache_stale(None) is True

    def test_recent_is_fresh(self) -> None:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        assert is_cache_stale(now) is False

    def test_old_is_stale(self) -> None:
        from datetime import UTC, datetime, timedelta

        old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
        assert is_cache_stale(old) is True

    def test_invalid_is_stale(self) -> None:
        assert is_cache_stale("not-a-date") is True


@pytest.mark.unit
class TestDistribution:
    def test_compute_distribution(self) -> None:
        models = [{"intelligence_index": i} for i in range(10, 60)]
        dist = compute_distribution(models)
        assert dist["stats"]["count"] == 50
        assert dist["stats"]["min"] == 10
        assert dist["stats"]["max"] == 59
        assert len(dist["scores"]) == 50

    def test_compute_distribution_filters_none(self) -> None:
        models: list[dict[str, Any]] = [
            {"intelligence_index": 50},
            {"intelligence_index": None},
            {"intelligence_index": 30},
        ]
        dist = compute_distribution(models)
        assert dist["stats"]["count"] == 2

    def test_compute_distribution_all_none(self) -> None:
        models = [{"intelligence_index": None}]
        with pytest.raises(ValueError, match="No models with intelligence_index"):
            compute_distribution(models)

    def test_dist_cache_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_QUALITY_CACHE_DIR", str(tmp_path))
        dist = {
            "stats": {
                "count": 5,
                "min": 10.0,
                "max": 50.0,
                "median": 30.0,
                "mean": 30.0,
                "stdev": 15.0,
                "p25": 20.0,
                "p75": 40.0,
            },
            "scores": [10, 20, 30, 40, 50],
        }
        path = save_dist_cache(dist)
        assert path.exists()

        loaded = load_dist_cache()
        assert loaded is not None
        assert loaded["stats"]["count"] == 5
