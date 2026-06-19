#!/usr/bin/env python3
"""Create a workflow-oriented train/val split for prompt-basics experiments.

This split is designed to test whether prompt changes that improve *general
pentesting heuristics* transfer to held-out benchmarks without teaching a
benchmark-specific exploit tutorial.

Design principles:
  - Validation is fixed to 5 benchmarks whose dominant bottleneck is workflow:
    recon, state inspection, cheap validation, and disciplined extraction.
  - Training is intentionally smaller and cleaner than a category-based split.
    Each selected challenge was kept only if its official solution reinforces a
    reusable workflow habit needed by one or more validation tasks.
  - Forbidden challenges are excluded automatically.

The intended prompt lessons are:
  - Authenticate early when low-privilege access is available
  - Inspect client-controlled state carefully: headers, cookies, hidden fields
  - Validate a suspected primitive cheaply before escalating
  - Enumerate nearby routes, objects, and admin surfaces after the first signal
  - Extract immediately once a primitive is confirmed instead of drifting

Usage:
    uv run python source/dataset/PoC_basics.py
"""

import argparse
import json
from pathlib import Path

from main_splits import _classify

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOLUTIONS_INDEX = _REPO_ROOT / "xbow" / "solutions" / "index.json"
_DEFAULT_OUTPUT = Path(__file__).parent / "PoC_basics.json"
_FORBIDDEN_PATH = Path(__file__).parent / "forbidden_challenges.json"
_EXCLUDED_BENCHMARKS: set[str] = set(json.loads(_FORBIDDEN_PATH.read_text(encoding="utf-8")).keys())

_VALIDATION_SET = [
    {
        "name": "idor_route_and_object_enumeration",
        "category": "IDOR",
        "heuristic_focus": [
            "authenticate first when a low-privilege account is available",
            "enumerate adjacent routes, usernames, and object identifiers before escalating further",
            "re-check sensitive objects immediately after the first unauthorized access signal",
        ],
        "selection_rationale": (
            "XBEN-090-24 is a clean metric for authenticated object enumeration. The solve path "
            "is dominated by logging in, trying nearby usernames, and checking draft IDs."
        ),
        "val_id": "XBEN-090-24",
    },
    {
        "name": "ssrf_fetcher_internal_pivot",
        "category": "SSRF",
        "heuristic_focus": [
            "identify user-controlled URL fetchers during recon",
            "probe loopback and internal targets once a fetch primitive is confirmed",
            "enumerate nearby internal admin routes after the first successful internal response",
        ],
        "selection_rationale": (
            "XBEN-033-24 mostly measures whether the agent notices a fetcher surface and pivots "
            "inward methodically rather than guessing unrelated exploit classes."
        ),
        "val_id": "XBEN-033-24",
    },
    {
        "name": "parameter_discovery_then_file_read",
        "category": "LFI",
        "heuristic_focus": [
            "map endpoints before committing to a bug class",
            "fuzz for hidden parameters and low-risk listing behavior before direct extraction",
            "once a file-read primitive exists, extract immediately instead of over-exploring",
        ],
        "selection_rationale": (
            "XBEN-098-24 is a good workflow metric because the critical step is discovering "
            "hidden parameters and using a listing primitive before direct file retrieval."
        ),
        "val_id": "XBEN-098-24",
    },
    {
        "name": "authenticated_profile_tampering",
        "category": "Business Logic",
        "heuristic_focus": [
            "inspect authenticated profile and update flows for hidden trust-boundary fields",
            "tamper identity- and role-bearing inputs before hunting exotic bugs",
            "revisit sensitive profile state immediately after a successful mutation",
        ],
        "selection_rationale": (
            "XBEN-102-24 is mostly about authenticated recon and hidden-field abuse, not a "
            "specialized exploit primitive."
        ),
        "val_id": "XBEN-102-24",
    },
    {
        "name": "ambient_state_identity_override",
        "category": "IDOR",
        "heuristic_focus": [
            "inspect cookies and other state-bearing client inputs, not just visible form fields",
            "decode or replay weak client-controlled identity state before attempting deeper exploitation",
            "keep the exploit focused on the discovered identity override vector",
        ],
        "selection_rationale": (
            "XBEN-021-24 is a lightweight validation benchmark for cookie/state inspection. "
            "The main lesson is to distrust client-stored identity context and enumerate it carefully."
        ),
        "val_id": None,
    },
]

_TRAINING_MODULES = [
    {
        "name": "authenticated_id_enumeration",
        "category": "IDOR",
        "heuristic_focus": [
            "authenticate early and keep session state stable",
            "enumerate adjacent identifiers and nearby authorized routes before escalating",
            "pivot from unauthorized reads into account takeover or privileged data access when available",
        ],
        "selection_rationale": (
            "These benchmarks all reward disciplined enumeration after login. They are the main "
            "support curriculum for XBEN-090-24 and also reinforce the route/ID mindset useful for XBEN-102-24."
        ),
        "train_ids": [
            "XBEN-001-24",
            "XBEN-002-24",
            "XBEN-058-24",
            "XBEN-069-24",
        ],
    },
    {
        "name": "client_controlled_identity_inputs",
        "category": "Access Control",
        "heuristic_focus": [
            "inspect hidden fields, headers, and user-controlled identity selectors",
            "tamper the simplest trust-boundary field first before reaching for advanced exploits",
            "treat any client-provided role or user identifier as suspect by default",
        ],
        "selection_rationale": (
            "These are the cleanest train benchmarks for the habit needed by XBEN-021-24 and XBEN-102-24: "
            "check whether the application is trusting client-controlled identity context."
        ),
        "train_ids": [
            "XBEN-003-24",
            "XBEN-021-24",
            "XBEN-043-24",
            "XBEN-052-24",
        ],
    },
    {
        "name": "ssrf_fetcher_and_internal_pivot",
        "category": "SSRF",
        "heuristic_focus": [
            "spot fetcher surfaces during baseline recon",
            "confirm internal reachability with the cheapest loopback probe available",
            "enumerate nearby internal endpoints immediately after the first internal response",
        ],
        "selection_rationale": (
            "These directly support XBEN-033-24 by teaching the agent to recognize and exploit "
            "URL-fetch features methodically rather than randomly probing the app."
        ),
        "train_ids": [
            "XBEN-020-24",
            "XBEN-024-24",
        ],
    },
    {
        "name": "primitive_confirmation_and_extraction",
        "category": "Command Injection",
        "heuristic_focus": [
            "validate a suspected primitive with the cheapest observable signal first",
            "once the primitive is real, extract immediately instead of switching exploit class",
            "use the endpoint's intended behavior to shape confirmation and extraction",
        ],
        "selection_rationale": (
            "These command-injection supports are included not to teach shell syntax, but to reinforce "
            "a general loop: confirm the primitive cheaply, then pivot straight to extraction. "
            "That habit matters for XBEN-033-24 and XBEN-098-24."
        ),
        "train_ids": [
            "XBEN-037-24",
            "XBEN-073-24",
        ],
    },
]


def _load_solution_records() -> dict[str, dict]:
    index = json.loads(_SOLUTIONS_INDEX.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for solution in index["solutions"]:
        bench_id = solution["benchmark_id"]
        if bench_id in _EXCLUDED_BENCHMARKS:
            continue
        records[bench_id] = {
            "id": bench_id,
            "level": int(solution["level"]),
            "tags": solution.get("tags", []),
            "category": _classify(solution.get("tags", [])),
            "name": solution.get("benchmark_name", bench_id),
            "solution_file": solution.get("solution_file"),
        }
    return records


def _validate_selection(records: dict[str, dict], bench_ids: list[str]) -> None:
    missing = [bench_id for bench_id in bench_ids if bench_id not in records]
    if missing:
        raise RuntimeError(f"Unknown or forbidden benchmark IDs in PoC_basics selection: {missing}")


def create_splits() -> dict:
    records = _load_solution_records()
    if not records:
        raise RuntimeError(f"No solution records found in {_SOLUTIONS_INDEX}")

    train: list[str] = []
    val: list[str] = []
    validation_meta: dict[str, dict] = {}
    training_meta: dict[str, dict] = {}

    for validation_case in _VALIDATION_SET:
        if validation_case["val_id"] is None:
            continue
        _validate_selection(records, [validation_case["val_id"]])
        val.append(validation_case["val_id"])
        val_record = records[validation_case["val_id"]]
        validation_meta[validation_case["name"]] = {
            "category": validation_case["category"],
            "heuristic_focus": validation_case["heuristic_focus"],
            "selection_rationale": validation_case["selection_rationale"],
            "val_id": validation_case["val_id"],
            "val_level": val_record["level"],
            "solution_files": {validation_case["val_id"]: val_record["solution_file"]},
        }

    for module in _TRAINING_MODULES:
        _validate_selection(records, module["train_ids"])
        train.extend(module["train_ids"])
        training_meta[module["name"]] = {
            "category": module["category"],
            "heuristic_focus": module["heuristic_focus"],
            "selection_rationale": module["selection_rationale"],
            "train_ids": list(module["train_ids"]),
            "train_levels": {bench_id: records[bench_id]["level"] for bench_id in module["train_ids"]},
            "train_count": len(module["train_ids"]),
            "solution_files": {
                bench_id: records[bench_id]["solution_file"]
                for bench_id in module["train_ids"]
                if records[bench_id].get("solution_file")
            },
        }

    if len(set(train)) != len(train):
        raise RuntimeError("Duplicate training benchmark selected in PoC_basics")
    if len(set(val)) != len(val):
        raise RuntimeError("Duplicate validation benchmark selected in PoC_basics")
    if set(train) & set(val):
        raise RuntimeError("PoC_basics train and val overlap")

    categories = sorted(
        {
            *(meta["category"] for meta in validation_meta.values()),
            *(meta["category"] for meta in training_meta.values()),
        }
    )

    return {
        "_meta": {
            "strategy": "workflow_metric_validation_with_curated_training_modules",
            "generated_by": "source/dataset/PoC_basics.py",
            "goal": (
                "Demonstrate that prompt improvements focused on general pentesting heuristics "
                "(recon, validation, state handling, extraction discipline, anti-drift) transfer "
                "across held-out workflow metrics without teaching exploit-specific tutorials."
            ),
            "forbidden_excluded": sorted(_EXCLUDED_BENCHMARKS),
            "split_counts": {"train": len(train), "val": len(val)},
            "validation_cases": validation_meta,
            "training_modules": training_meta,
            "categories": categories,
        },
        "train": list(train),
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
    print(f"Train    : {meta['split_counts']['train']}")
    print(f"Val      : {meta['split_counts']['val']}")

    print("\nValidation cases:")
    for name, info in sorted(meta["validation_cases"].items()):
        print(
            f"  {name:<34} category={info['category']:<16} "
            f"val={info['val_id']}"
        )

    print("\nTraining modules:")
    for name, info in sorted(meta["training_modules"].items()):
        print(
            f"  {name:<34} category={info['category']:<16} "
            f"train={','.join(info['train_ids'])}"
        )

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten -> {args.output}")


if __name__ == "__main__":
    main()
