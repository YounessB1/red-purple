#!/usr/bin/env python3
"""Create a controlled solution-family transfer split for the XBOW benchmark suite.

This split is designed for a GEPA generalization PoC:
  - Keep a fixed set of 5 validation challenges to control validation cost.
  - Use only strong, repeated exploit families that remain after forbidden
    challenges are excluded.
  - Prefer curated families where the train members clearly teach one reusable
    exploit technique that can plausibly transfer to the held-out validation
    benchmark.
  - For each selected family, hold out the hardest benchmark as validation,
    except where a specific benchmark is intentionally pinned.
  - No test set.

Why this split exists:
  - It avoids the "one skill per benchmark" story.
  - Train contains multiple benchmarks that should map to the same reusable skill.
  - Validation checks whether that learned skill transfers to an unseen benchmark
    from the same solution family.

Usage:
    uv run python source/dataset/PoC_generalization.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from main_splits import _classify

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOLUTIONS_INDEX = _REPO_ROOT / "xbow" / "solutions" / "index.json"
_DEFAULT_OUTPUT = Path(__file__).parent / "PoC_generalization.json"
_FORBIDDEN_PATH = Path(__file__).parent / "forbidden_challenges.json"
_EXCLUDED_BENCHMARKS: set[str] = set(json.loads(_FORBIDDEN_PATH.read_text(encoding="utf-8")).keys())

_FOCUS_CATEGORIES = {"XSS", "SSTI", "SQL Injection", "Command Injection", "IDOR"}
_SELECTED_FAMILIES = [
    "ssti_python_object_traversal",
    "xss_event_handler_bypass",
    "xss_filter_bypass",
    "idor_object_reference_curated",
    "cmdi_ping_family",
]

_PINNED_VAL_BY_FAMILY = {
    "cmdi_ping_family": "XBEN-045-24",
    "idor_object_reference_curated": "XBEN-090-24",
}


def _solution_family(solution: dict) -> str | None:
    """Map a benchmark solution to a reusable exploit family."""
    tags = set(solution.get("tags", []))
    category = _classify(list(tags))
    if category not in _FOCUS_CATEGORIES:
        return None

    text = " ".join(solution.get("workflow", []) + solution.get("vulnerability_summary", []))
    text = text.lower()

    if category == "XSS":
        if any(token in text for token in ["ontoggle", "onerror", "onload", "<svg", "svg", "<details", "details"]):
            return "xss_event_handler_bypass"
        if any(token in text for token in ["javascript context", "inline javascript", "js variable", "quote", "backslash"]):
            return "xss_js_context_breakout"
        if any(token in text for token in ["filter", "waf", "blocked", "bypass"]):
            return "xss_filter_bypass"
        return "xss_reflected_basic"

    if category == "SSTI":
        if any(token in text for token in ["__globals__", "__builtins__", "popen", "mro", "subclass traversal"]):
            return "ssti_python_object_traversal"
        if "<%=" in text:
            return "ssti_erb_exec"
        if "twig" in text:
            return "ssti_twig"
        if any(token in text for token in ["debug", "flag variable", "{{ flag }}", "local context"]):
            return "ssti_context_leak"
        return "ssti_template_eval"

    if category == "SQL Injection":
        if "nosqli" in tags:
            return "nosqli_graphql_operator_injection"
        if "graphql" in tags:
            return "sqli_graphql"
        if any(token in text for token in ["case-sensitive", "mixed-case", "allowlist", "keyword filter"]):
            return "sqli_filter_bypass"
        if any(token in text for token in ["blind", "boolean-based", "boolean based"]):
            return "sqli_blind_extraction"
        if any(token in text for token in ["union", "error-based", "error based"]):
            return "sqli_union_error_extraction"
        if any(token in text for token in ["login", "' or '1'='1", "auth bypass", "signin"]):
            return "sqli_auth_bypass"
        return "sqli_generic"

    if category == "Command Injection":
        if "/ping" in text or "ip_address" in text:
            return "cmdi_ping_family"
        if any(token in text for token in ["upload", "webshell", ".php", "shell.php"]):
            return "cmdi_upload_rce"
        return "cmdi_generic"

    if category == "IDOR":
        return "idor_generic"

    return None


_CURATED_FAMILY_MEMBERS = {
    "idor_object_reference_curated": {
        "XBEN-001-24",
        "XBEN-002-24",
        "XBEN-058-24",
        "XBEN-069-24",
        "XBEN-090-24",
    }
}


def _load_solution_records() -> list[dict]:
    index = json.loads(_SOLUTIONS_INDEX.read_text(encoding="utf-8"))
    records = []
    for solution in index["solutions"]:
        if solution["benchmark_id"] in _EXCLUDED_BENCHMARKS:
            continue
        category = _classify(solution.get("tags", []))
        family = _solution_family(solution)
        if family is None:
            continue
        records.append(
            {
                "id": solution["benchmark_id"],
                "level": int(solution["level"]),
                "tags": solution.get("tags", []),
                "category": category,
                "family": family,
                "name": solution.get("benchmark_name", solution["benchmark_id"]),
                "solution_file": solution.get("solution_file"),
            }
        )
    return records


def create_splits() -> dict:
    records = _load_solution_records()
    if not records:
        raise RuntimeError(f"No solution records found in {_SOLUTIONS_INDEX}")

    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    curated_items = [
        record
        for record in records
        if record["id"] in _CURATED_FAMILY_MEMBERS["idor_object_reference_curated"]
    ]
    if curated_items:
        by_family["idor_object_reference_curated"] = curated_items

    selected_families: dict[str, list[dict]] = {}
    for family in _SELECTED_FAMILIES:
        items = by_family.get(family, [])
        if len(items) < 2:
            raise RuntimeError(
                f"Selected family {family!r} has only {len(items)} usable benchmarks after exclusions"
            )
        selected_families[family] = sorted(items, key=lambda item: (-item["level"], item["id"]))

    train: list[str] = []
    val: list[str] = []
    family_meta: dict[str, dict] = {}

    for family in _SELECTED_FAMILIES:
        items = selected_families[family]
        pinned_val_id = _PINNED_VAL_BY_FAMILY.get(family)
        if pinned_val_id is None:
            val_item = items[0]
            train_items = items[1:]
        else:
            try:
                val_item = next(item for item in items if item["id"] == pinned_val_id)
            except StopIteration as exc:
                raise RuntimeError(f"Pinned val benchmark {pinned_val_id!r} not found in family {family!r}") from exc
            train_items = [item for item in items if item["id"] != pinned_val_id]

        val.append(val_item["id"])
        train.extend(item["id"] for item in train_items)

        family_meta[family] = {
            "category": val_item["category"],
            "count": len(items),
            "train_count": len(train_items),
            "val_id": val_item["id"],
            "train_ids": sorted(item["id"] for item in train_items),
            "all_ids": sorted(item["id"] for item in items),
            "selection_rule": (
                f"pinned val benchmark {pinned_val_id}"
                if pinned_val_id is not None
                else "hardest benchmark in family goes to val"
            ),
            "solution_files": {item["id"]: item["solution_file"] for item in items if item.get("solution_file")},
        }

    selected_ids = {bench_id for bench_id in train + val}
    excluded_families = {}
    for family, items in sorted(by_family.items()):
        if family in _SELECTED_FAMILIES:
            continue
        remaining_ids = sorted(item["id"] for item in items if item["id"] not in selected_ids)
        if remaining_ids:
            excluded_families[family] = remaining_ids

    categories = sorted({meta["category"] for meta in family_meta.values()})

    return {
        "_meta": {
            "strategy": "controlled_five_family_transfer_holdout",
            "generated_by": "source/dataset/PoC_generalization.py",
            "goal": "Demonstrate GEPA learns reusable skills that transfer across held-out benchmarks from a fixed set of exploit families.",
            "categories": categories,
            "selected_families": _SELECTED_FAMILIES,
            "family_count": len(family_meta),
            "split_counts": {"train": len(train), "val": len(val)},
            "family_summary": family_meta,
            "excluded_families": excluded_families,
        },
        "train": sorted(train),
        "val": sorted(val),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = create_splits()
    meta = result["_meta"]

    print(f"Strategy : {meta['strategy']}")
    print(f"Goal     : {meta['goal']}")
    print(f"Families : {meta['family_count']} selected solution families")
    print(f"Train    : {meta['split_counts']['train']}")
    print(f"Val      : {meta['split_counts']['val']}")

    print("\nValidation holdouts by family:")
    for family, info in sorted(meta["family_summary"].items()):
        print(
            f"  {family:<30} category={info['category']:<13} "
            f"val={info['val_id']} train={','.join(info['train_ids'])}"
        )

    if meta["excluded_families"]:
        print("\nExcluded non-selected families:")
        for family, ids in sorted(meta["excluded_families"].items()):
            print(f"  {family:<30} ids={','.join(ids)}")

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten -> {args.output}")


if __name__ == "__main__":
    main()
