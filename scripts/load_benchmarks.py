#!/usr/bin/env python3
"""Load benchmark data from JSON into the database.

This script reads benchmark data from a JSON file and inserts it
into the exported_summaries table.

Core loading logic lives in planner.knowledge_base.loader and is
shared with the /api/v1/db/* API endpoints.
"""

import argparse
import json
import sys
from pathlib import Path

from planner.knowledge_base.benchmarks import BenchmarkRepository
from planner.knowledge_base.loader import extract_metadata


def load_benchmarks_json(json_file=None):
    """Load benchmarks from JSON file."""
    if json_file:
        json_path = Path(__file__).parent.parent / json_file
    else:
        json_path = (
            Path(__file__).parent.parent
            / "data"
            / "benchmarks"
            / "performance"
            / "benchmarks_BLIS.json"
        )

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    with open(json_path) as f:
        return json.load(f)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Load benchmark data into database")
    parser.add_argument("json_file", nargs="?", default=None, help="Path to benchmark JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading Benchmark Data")
    print("=" * 60)
    print()

    data = load_benchmarks_json(args.json_file)
    benchmarks = data.get("benchmarks", [])
    print(f"Loaded {len(benchmarks)} benchmarks from JSON")

    meta = extract_metadata(data)
    source = meta["source"] or "local"
    confidence_level = meta["confidence_level"] or "estimated"
    print(f"  source: {source}, confidence_level: {confidence_level}")

    repo = BenchmarkRepository()
    print("Connected to database")
    print()

    try:
        stats = repo.load_benchmarks(
            benchmarks, source=source, confidence_level=confidence_level
        )

        print("\nDatabase Statistics:")
        print(f"  Models: {stats['num_models']}")
        print(f"  Hardware types: {stats['num_hardware_types']}")
        print(f"  Traffic profiles: {stats['num_traffic_profiles']}")
        print(f"  Total benchmarks: {stats['total_benchmarks']}")

        print("\nTraffic Profile Distribution:")
        for tp in stats.get("traffic_distribution", []):
            print(f"  ({tp['prompt_tokens']}, {tp['output_tokens']}): {tp['count']} benchmarks")

    except Exception as e:
        print(f"\nError inserting benchmarks: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Load complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  make db-query-traffic  # View traffic patterns")
    print("  make db-query-models   # View available models")
    print("  make db-shell          # Open database shell")


if __name__ == "__main__":
    main()
