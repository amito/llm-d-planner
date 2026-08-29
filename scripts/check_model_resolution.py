"""CLI for checking model name resolution against Arena and AA data sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quality_scoring import aa_client, arena_client
from quality_scoring.data._resolver import quality_data_path
from quality_scoring.models import MatchType
from quality_scoring.resolver import resolve_model_names, suggest_similar


def _load_arena_names(cache_dir: Path | None) -> list[str]:
    rows, _ = arena_client.load_cache(cache_dir)
    if not rows:
        if cache_dir is not None:
            print(
                f"Warning: no Arena data found in {cache_dir}, falling back to bundled data",
                file=sys.stderr,
            )
        rows, _ = arena_client.load_cache(quality_data_path())
    return sorted({r["model_name"] for r in rows if r.get("category") == "overall"})


def _load_aa_names(cache_dir: Path | None) -> list[str]:
    models, _ = aa_client.load_cache(cache_dir)
    if not models:
        if cache_dir is not None:
            print(
                f"Warning: no AA data found in {cache_dir}, falling back to bundled data",
                file=sys.stderr,
            )
        models, _ = aa_client.load_cache(quality_data_path())
    return sorted({m["name"] for m in models})


def _load_model_names(args: argparse.Namespace) -> list[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Error: catalog file not found: {catalog_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(catalog_path) as f:
            catalog = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: catalog file is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if "models" not in catalog or not isinstance(catalog["models"], list):
        print('Error: catalog JSON must contain a "models" array', file=sys.stderr)
        sys.exit(1)

    try:
        return [m["model_id"] for m in catalog["models"]]
    except (KeyError, TypeError) as e:
        print(f'Error: each catalog entry must have a "model_id" field: {e}', file=sys.stderr)
        sys.exit(1)


def _print_source_results(
    source_name: str,
    model_names: list[str],
    known_names: list[str],
    fuzzy: bool,
) -> int:
    print(f"\n=== {source_name} ({len(known_names)} known models) ===")

    if not known_names:
        print("  No data available for this source.")
        return len(model_names)

    results = resolve_model_names(model_names, known_names)
    not_found_count = 0

    for mr in results:
        if mr.match_type == MatchType.NONE:
            label = "not found"
        elif mr.match_type == MatchType.FUZZY and not fuzzy:
            label = f"{mr.matched_name} (fuzzy, not accepted)"
        else:
            print(f'  "{mr.user_name}" -> {mr.matched_name} ({mr.match_type.value})')
            continue

        print(f'  "{mr.user_name}" -> {label}')
        suggestions = suggest_similar(mr.user_name, known_names, n=3)
        if suggestions:
            print(f"    Similar: {', '.join(suggestions)}")
        not_found_count += 1

    return not_found_count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="check_model_resolution",
        description="Check how model names resolve against Arena and AA data sources.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--catalog",
        metavar="PATH",
        help='Path to model catalog JSON (format: {"models": [{"model_id": "..."}]})',
    )
    group.add_argument(
        "-m",
        "--models",
        metavar="NAMES",
        help="Comma-separated model names to check",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Accept fuzzy matches instead of treating them as not-found",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Override cache directory for quality data (falls back to bundled snapshots)",
    )
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    model_names = _load_model_names(args)

    if not model_names:
        print("No model names to check.", file=sys.stderr)
        sys.exit(1)

    print(f"Checking {len(model_names)} model(s)...")

    arena_names = _load_arena_names(cache_dir)
    aa_names = _load_aa_names(cache_dir)

    arena_missed = _print_source_results("Arena", model_names, arena_names, args.fuzzy)
    aa_missed = _print_source_results("Artificial Analysis", model_names, aa_names, args.fuzzy)

    print(f"\nSummary: {len(model_names)} models checked")
    print(f"  Arena:    {len(model_names) - arena_missed}/{len(model_names)} resolved")
    print(f"  AA:       {len(model_names) - aa_missed}/{len(model_names)} resolved")


if __name__ == "__main__":
    main()
