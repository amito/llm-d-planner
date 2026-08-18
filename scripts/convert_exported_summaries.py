#!/usr/bin/env python3
"""
Convert exported benchmark summaries JSON to GuideLLM format.

Reads benchmarks-exported-summaries.json (bare array, string-typed values)
and writes benchmarks_GuideLLM.json (metadata-wrapped, native types).

Usage:
    python scripts/convert_exported_summaries.py
    python scripts/convert_exported_summaries.py -i src/planner/data/performance/benchmarks-exported-summaries.json
    python scripts/convert_exported_summaries.py -o src/planner/data/performance/benchmarks_GuideLLM.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FLOAT_FIELDS = {
    "ttft_mean", "ttft_p90", "ttft_p95", "ttft_p99",
    "itl_mean", "itl_p90", "itl_p95", "itl_p99",
    "e2e_mean", "e2e_p90", "e2e_p95", "e2e_p99",
    "tps_mean", "tps_p90", "tps_p95", "tps_p99",
    "tokens_per_second", "mean_input_tokens", "mean_output_tokens",
}

INT_FIELDS = {
    "hardware_count", "requests_per_second",
    "prompt_tokens", "prompt_tokens_min", "prompt_tokens_max", "prompt_tokens_stdev",
    "output_tokens", "output_tokens_min", "output_tokens_max", "output_tokens_stdev",
}

NULLABLE_FIELDS = {"huggingface_prompt_dataset", "profiler_image", "profiler_tag"}

DROP_FIELDS = {"id", "created_at", "updated_at", "loaded_at", "jbenchmark_created_at"}


def convert_record(record: dict) -> dict:
    """Convert a single benchmark record from string types to native types."""
    converted = {}
    for key, value in record.items():
        if key in DROP_FIELDS:
            continue

        if key in NULLABLE_FIELDS and (value is None or value == "NULL"):
            converted[key] = None
        elif key in FLOAT_FIELDS:
            converted[key] = float(value) if value and value != "NULL" else 0.0
        elif key in INT_FIELDS:
            converted[key] = int(float(value)) if value and value != "NULL" else 0
        else:
            converted[key] = value

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert exported summaries to GuideLLM format")
    parser.add_argument(
        "-i", "--input",
        default="src/planner/data/performance/benchmarks-exported-summaries.json",
        help="Input file path",
    )
    parser.add_argument(
        "-o", "--output",
        default="src/planner/data/performance/benchmarks_GuideLLM.json",
        help="Output file path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"Converting {len(records)} records...")
    converted = [convert_record(r) for r in records]

    unique_models = sorted(set(r.get("model_hf_repo", "unknown") for r in converted))

    output = {
        "_metadata": {
            "description": "GuideLLM benchmark data converted from exported summaries",
            "version": "1.0",
            "source": "guidellm",
            "confidence_level": "benchmarked",
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(converted),
        },
        "benchmarks": converted,
    }

    print(f"Writing: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Done: {len(converted)} records, {len(unique_models)} unique models")
    for m in unique_models:
        count = sum(1 for r in converted if r.get("model_hf_repo") == m)
        print(f"  {m}: {count} benchmarks")


if __name__ == "__main__":
    main()
