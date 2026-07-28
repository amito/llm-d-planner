"""In-process benchmark storage backed by SQLite.

Provides the same query interface as BenchmarkRepository (PostgreSQL)
but requires no external database. Benchmark data is loaded from JSON
files or raw dicts into an in-memory SQLite database.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path

from planner.knowledge_base.benchmarks import BenchmarkData
from planner.knowledge_base.loader import (
    extract_metadata,
    generate_config_id,
    normalize_benchmark_fields,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exported_summaries (
    id TEXT PRIMARY KEY,
    config_id TEXT UNIQUE,
    model_hf_repo TEXT NOT NULL,
    provider TEXT,
    type TEXT NOT NULL DEFAULT 'local',
    ttft_mean REAL NOT NULL,
    ttft_p90 REAL NOT NULL,
    ttft_p95 REAL NOT NULL,
    ttft_p99 REAL NOT NULL,
    e2e_mean REAL NOT NULL,
    e2e_p90 REAL NOT NULL,
    e2e_p95 REAL NOT NULL,
    e2e_p99 REAL NOT NULL,
    itl_mean REAL,
    itl_p90 REAL,
    itl_p95 REAL,
    itl_p99 REAL,
    tps_mean REAL,
    tps_p90 REAL,
    tps_p95 REAL,
    tps_p99 REAL,
    hardware TEXT,
    hardware_count INTEGER,
    framework TEXT,
    framework_version TEXT,
    requests_per_second REAL NOT NULL,
    tokens_per_second REAL NOT NULL,
    mean_input_tokens REAL NOT NULL,
    mean_output_tokens REAL NOT NULL,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    model_uri TEXT,
    source TEXT NOT NULL DEFAULT 'local',
    confidence_level TEXT NOT NULL DEFAULT 'estimated'
);
CREATE INDEX IF NOT EXISTS idx_benchmark_lookup
    ON exported_summaries(model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens);
CREATE INDEX IF NOT EXISTS idx_traffic_patterns
    ON exported_summaries(prompt_tokens, output_tokens);
"""

# Columns inserted by add_benchmarks / save_benchmarks
_INSERT_COLUMNS = (
    "id",
    "config_id",
    "model_hf_repo",
    "provider",
    "type",
    "ttft_mean",
    "ttft_p90",
    "ttft_p95",
    "ttft_p99",
    "e2e_mean",
    "e2e_p90",
    "e2e_p95",
    "e2e_p99",
    "itl_mean",
    "itl_p90",
    "itl_p95",
    "itl_p99",
    "tps_mean",
    "tps_p90",
    "tps_p95",
    "tps_p99",
    "hardware",
    "hardware_count",
    "framework",
    "framework_version",
    "requests_per_second",
    "tokens_per_second",
    "mean_input_tokens",
    "mean_output_tokens",
    "prompt_tokens",
    "output_tokens",
    "model_uri",
    "source",
    "confidence_level",
)


class LocalBenchmarkRepository:
    """In-process benchmark storage backed by SQLite.

    Provides the same query interface as BenchmarkRepository (PostgreSQL)
    but requires no external database. Loads benchmark data from JSON files
    or pre-built dicts into an in-memory SQLite database.
    """

    def __init__(self) -> None:
        """Create an empty repository with in-memory SQLite database."""
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def _prepare_row(
        self,
        benchmark: dict,
        source: str = "local",
        confidence_level: str = "estimated",
    ) -> dict:
        """Normalize a raw benchmark dict into a row ready for insertion."""
        row = normalize_benchmark_fields(benchmark)
        row["id"] = str(uuid.uuid4())
        row["config_id"] = generate_config_id(row)
        row.setdefault("type", "local")
        row.setdefault("provider", None)
        row.setdefault("framework", None)
        row.setdefault("framework_version", None)
        row.setdefault("tps_mean", None)
        row.setdefault("tps_p90", None)
        row.setdefault("tps_p95", None)
        row.setdefault("tps_p99", None)
        row.setdefault("model_uri", None)
        row["source"] = source
        row["confidence_level"] = confidence_level
        return row

    @classmethod
    def from_files(cls, json_paths: list[Path]) -> LocalBenchmarkRepository:
        """Load benchmarks from one or more JSON files.

        Each JSON file must have a top-level "benchmarks" key containing
        a list of benchmark dicts. Optionally, a "_metadata" key with
        "source" and "confidence_level" fields.

        Args:
            json_paths: List of Path objects to JSON files.

        Returns:
            LocalBenchmarkRepository with loaded benchmarks.
        """
        repo = cls()
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
            benchmarks = data.get("benchmarks", [])
            meta = extract_metadata(data)
            source = meta.get("source") or "local"
            confidence = meta.get("confidence_level") or "estimated"
            repo.add_benchmarks(benchmarks, source=source, confidence_level=confidence)
            logger.info("Loaded %d benchmarks from %s", len(benchmarks), path)
        return repo

    @classmethod
    def from_benchmarks(cls, benchmarks: list[dict]) -> LocalBenchmarkRepository:
        """Load benchmarks from pre-built dicts.

        Dicts are raw benchmark records (same format as entries in the
        JSON "benchmarks" array). Field normalization is applied
        automatically.

        Args:
            benchmarks: List of benchmark dicts.

        Returns:
            LocalBenchmarkRepository with loaded benchmarks.
        """
        repo = cls()
        repo.add_benchmarks(benchmarks)
        return repo

    def add_benchmarks(
        self,
        benchmarks: list[dict],
        source: str = "local",
        confidence_level: str = "estimated",
    ) -> int:
        """Add benchmark rows to the repository.

        Appends to existing data. Duplicates (same config_id) are
        silently skipped.

        Args:
            benchmarks: Raw benchmark dicts (normalized automatically).
            source: Data source identifier (e.g., 'blis', 'model_catalog').
            confidence_level: Trust level ('benchmarked' or 'estimated').

        Returns:
            Number of rows inserted.
        """
        if not benchmarks:
            return 0

        cols = ", ".join(_INSERT_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in _INSERT_COLUMNS)
        sql = f"INSERT OR IGNORE INTO exported_summaries ({cols}) VALUES ({placeholders})"

        rows = [self._prepare_row(b, source, confidence_level) for b in benchmarks]
        cursor = self._conn.cursor()
        before_result = cursor.execute("SELECT COUNT(*) FROM exported_summaries").fetchone()
        before = int(before_result[0]) if before_result else 0
        cursor.executemany(sql, rows)
        self._conn.commit()
        after_result = cursor.execute("SELECT COUNT(*) FROM exported_summaries").fetchone()
        after = int(after_result[0]) if after_result else 0
        inserted = after - before
        logger.info("Inserted %d of %d benchmarks (source=%s)", inserted, len(benchmarks), source)
        return inserted

    def replace_benchmarks(self, source: str, benchmarks: list[dict]) -> int:
        """Replace all benchmarks for a given source.

        Deletes existing rows with matching source, then inserts the new
        rows. This mirrors the catalog sync pattern.

        Args:
            source: Data source identifier to replace.
            benchmarks: New benchmark dicts to insert.

        Returns:
            Number of rows inserted.
        """
        self._conn.execute("DELETE FROM exported_summaries WHERE source = ?", (source,))
        self._conn.commit()
        if not benchmarks:
            return 0
        return self.add_benchmarks(benchmarks, source=source)

    def save_benchmarks(
        self,
        benchmarks: list[BenchmarkData],
        source: str = "llm-optimizer",
        confidence_level: str = "estimated",
    ) -> None:
        """Persist BenchmarkData objects to the in-memory database.

        Used by the estimator to cache roofline estimates.
        Same signature as BenchmarkRepository.save_benchmarks().

        Args:
            benchmarks: List of BenchmarkData objects.
            source: Data source identifier (default 'llm-optimizer').
            confidence_level: Trust level ('benchmarked' or 'estimated').
        """
        benchmark_dicts = [b.to_dict() for b in benchmarks]
        for d in benchmark_dicts:
            d.setdefault("prompt_tokens", d.get("mean_input_tokens"))
            d.setdefault("output_tokens", d.get("mean_output_tokens"))
        self.add_benchmarks(benchmark_dicts, source=source, confidence_level=confidence_level)

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
        exclude_estimated: bool = False,
    ) -> list[BenchmarkData]:
        """Find configurations meeting SLO requirements.

        Same signature and semantics as BenchmarkRepository.
        For each unique (model_hf_repo, hardware, hardware_count),
        returns only the benchmark with the highest requests_per_second
        that still meets SLO requirements.

        Args:
            prompt_tokens: Target prompt length
            output_tokens: Target output length
            ttft_p95_max_ms: Maximum acceptable TTFT (ms)
            itl_p95_max_ms: Maximum acceptable ITL (ms/token)
            e2e_p95_max_ms: Maximum acceptable E2E (ms)
            min_qps: Minimum required QPS
            percentile: Which percentile column to use (mean, p90, p95, p99)
            gpu_types: Optional list of GPU types to filter by
            exclude_estimated: If True, exclude rows with confidence_level='estimated'

        Returns:
            List of benchmarks meeting all criteria
        """
        percentile_columns = {
            "mean": ("ttft_mean", "itl_mean", "e2e_mean"),
            "p90": ("ttft_p90", "itl_p90", "e2e_p90"),
            "p95": ("ttft_p95", "itl_p95", "e2e_p95"),
            "p99": ("ttft_p99", "itl_p99", "e2e_p99"),
        }
        cols = percentile_columns.get(percentile)
        if cols is None:
            logger.warning("Invalid percentile '%s', defaulting to p95", percentile)
            cols = percentile_columns["p95"]
        ttft_col, itl_col, e2e_col = cols

        # Build dynamic WHERE clauses
        conditions = [
            "prompt_tokens = ?",
            "output_tokens = ?",
            f"{ttft_col} <= ?",
            f"{itl_col} <= ?",
            f"{e2e_col} <= ?",
            "requests_per_second >= ?",
        ]
        params: list = [
            prompt_tokens,
            output_tokens,
            ttft_p95_max_ms,
            itl_p95_max_ms,
            e2e_p95_max_ms,
            min_qps,
        ]

        if gpu_types:
            placeholders = ", ".join("?" for _ in gpu_types)
            conditions.append(f"hardware IN ({placeholders})")
            params.extend(gpu_types)

        if exclude_estimated:
            conditions.append("confidence_level != 'estimated'")

        where = " AND ".join(conditions)

        query = f"""
            WITH ranked_configs AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_hf_repo, hardware, hardware_count
                           ORDER BY requests_per_second DESC, {e2e_col} ASC
                       ) as rn
                FROM exported_summaries
                WHERE {where}
            )
            SELECT
                id, config_id, model_hf_repo, provider, type,
                ttft_mean, ttft_p90, ttft_p95, ttft_p99,
                e2e_mean, e2e_p90, e2e_p95, e2e_p99,
                itl_mean, itl_p90, itl_p95, itl_p99,
                tps_mean, tps_p90, tps_p95, tps_p99,
                hardware, hardware_count, framework, framework_version,
                requests_per_second, tokens_per_second,
                mean_input_tokens, mean_output_tokens,
                prompt_tokens, output_tokens,
                model_uri, source, confidence_level
            FROM ranked_configs
            WHERE rn = 1
            ORDER BY model_hf_repo, hardware, hardware_count
        """

        cursor = self._conn.execute(query, params)
        results = [BenchmarkData(dict(row)) for row in cursor.fetchall()]
        logger.info("Found %d benchmarks meeting SLO criteria", len(results))
        return results
