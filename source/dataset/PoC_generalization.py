#!/usr/bin/env python3
"""Create the PoC-generalization split for the XBOW benchmark suite.

Split strategy:
  - Fixed categories: XSS, SSTI, SQL Injection.
  - Val: hardest per category (1 each → 3 total); SQL Injection val is pinned to XBEN-071-24.
  - Train: everything else in those 3 categories.
  - No test set.

Usage:
    uv run python source/dataset/PoC_generalization.py
"""

import json
from collections import defaultdict
from pathlib import Path

from main_splits import _collect, _DEFAULT_BENCHMARKS_DIR

_DEFAULT_OUTPUT = Path(__file__).parent / "PoC_generalization.json"


_CATEGORIES = ["XSS", "SSTI", "SQL Injection"]
_PINNED_VAL = {"SQL Injection": "XBEN-029-24"}


def create_splits(benchmarks_dir: Path) -> dict:
    records = _collect(benchmarks_dir)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["category"] in _CATEGORIES:
            by_category[r["category"]].append(r)

    val, train = [], []
    for cat in _CATEGORIES:
        recs = sorted(by_category[cat], key=lambda r: (-r["level"], r["id"]))
        pinned = _PINNED_VAL.get(cat)
        if pinned:
            val.append(pinned)
            train.extend(r["id"] for r in recs if r["id"] != pinned)
        else:
            val.append(recs[0]["id"])
            train.extend(r["id"] for r in recs[1:])

    return {
        "_meta": {
            "strategy": "fixed_categories_hardest_val",
            "generated_by": "source/dataset/PoC_generalization.py",
            "categories": _CATEGORIES,
            "split_counts": {"train": len(train), "val": len(val)},
        },
        "train": sorted(train),
        "val": sorted(val),
    }


def main() -> None:
    result = create_splits(_DEFAULT_BENCHMARKS_DIR)
    meta = result["_meta"]

    print(f"Top 3 categories: {meta['categories']}")
    print(f"Val  ({meta['split_counts']['val']}): {result['val']}")
    print(f"Train ({meta['split_counts']['train']}): {result['train']}")

    _DEFAULT_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten → {_DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
