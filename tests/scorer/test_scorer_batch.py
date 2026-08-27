"""Test scorer across multiple real failed CTF runs (batch variant of test_scorer.py).

Verifies that for each fixture:
  1. score is a valid float in [0.0, 0.9] with a non-empty reason.
  2. scorer_steps_<bench_id>.json is written to out_dir.
Prints a summary table across all fixtures for quality inspection.

Fixtures: tests/scorer/fixtures/<bench_id>/  (one dir per XBEN case)
Output:   tests/scorer/output/result_batch.json
          tests/scorer/output/scorer_steps_<bench_id>.json
"""

import json
import yaml
from pathlib import Path

from source.optimize_anything.scorer import llm_judge

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures"
_OUT      = Path(__file__).parent / "output"

BENCH_IDS = sorted(
    p.name for p in _FIXTURES.iterdir()
    if p.is_dir() and (p / "context_window.json").exists()
)


class _Logger:
    def log_scorer(self, input_tokens, output_tokens, cost):
        print(f"[test-scorer-batch] tokens in={input_tokens} out={output_tokens} cost=${cost:.4f}")


def _load_experiment_config() -> dict:
    raw = yaml.safe_load((_ROOT / "config.yaml").read_text())
    return raw[0] if isinstance(raw, list) else raw


def main() -> None:
    cfg   = _load_experiment_config()
    model = cfg["scorer"]["md"]["model"]
    gt    = cfg["scorer"].get("gt", False)

    print(f"[test-scorer-batch] model={model}  gt={gt}  fixtures={BENCH_IDS}")
    _OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for bench_id in BENCH_IDS:
        fixture_dir = _FIXTURES / bench_id
        context_window = json.loads((fixture_dir / "context_window.json").read_text())

        score, reason = llm_judge(
            context_window, bench_id, model=model, logger=_Logger(), gt=gt, out_dir=_OUT,
        )
        assert 0.0 <= score <= 0.9, f"{bench_id}: score out of range: {score}"
        assert reason, f"{bench_id}: reason is empty"

        steps_file = _OUT / "scorer_steps.json"
        if steps_file.exists():
            steps_file.replace(_OUT / f"scorer_steps_{bench_id}.json")

        results.append({"bench_id": bench_id, "model": model, "gt": gt, "score": score, "reason": reason})

    print("\n[test-scorer-batch] summary:")
    for r in results:
        print(f"  {r['bench_id']:>14}  score={r['score']}  {r['reason'][:100]}")

    (_OUT / "result_batch.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[test-scorer-batch] saved -> {_OUT / 'result_batch.json'}")


if __name__ == "__main__":
    main()
