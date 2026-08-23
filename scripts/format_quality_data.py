"""Format, sort, and strip volatile fields from quality data JSON files for readable git diffs."""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Callable
from typing import Any


def fmt(
    file_path: str,
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
    strip_keys: set[str] | None = None,
) -> None:
    p = pathlib.Path(file_path)
    if not p.is_file():
        print(f"WARNING: skipping {file_path} (file not found)", file=sys.stderr)
        return
    d = json.loads(p.read_text())
    for k in ("rows", "models"):
        if k in d and sort_key:
            d[k].sort(key=sort_key)
        if k in d and strip_keys:
            d[k] = [{kk: vv for kk, vv in m.items() if kk not in strip_keys} for m in d[k]]
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


# Sort keys must match the corresponding sync() functions for consistent ordering:
#   arena: arena_client.py sync()
#   aa:    aa_client.py sync()
fmt(
    "src/quality_scoring/data/arena_models.json",
    lambda r: (r.get("model_name") or "", r.get("category") or ""),
)
fmt(
    "src/quality_scoring/data/aa_models.json",
    lambda m: (m.get("slug") or ""),
    strip_keys={"accessed_date"},
)
fmt("src/quality_scoring/data/arena_dist.json")
fmt("src/quality_scoring/data/aa_dist.json")
