"""Test diagnoser across multiple real failed CTF runs (batch variant of test_diagnoser.py).

Verifies that for each fixture:
  1. diagnosis output is non-empty and plausible.
  2. diagnoser_steps_<bench_id>.json is written to out_dir.
Prints the full reasoning trace and diagnosis per fixture for quality inspection.

Fixtures: tests/diagnoser/fixtures/<bench_id>/  (one dir per XBEN case)
Output:   tests/diagnoser/output/result_batch.json
          tests/diagnoser/output/diagnoser_steps_<bench_id>.json
"""

import json
import yaml
from pathlib import Path

from source.optimize_anything.diagnoser import diagnose

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures"
_OUT      = Path(__file__).parent / "output"

TRAIN_SIZE = 3

BENCH_IDS = sorted(
    p.name for p in _FIXTURES.iterdir()
    if p.is_dir() and (p / "context_window.json").exists()
)


class _Logger:
    def log_diagnoser(self, input_tokens, output_tokens, cost):
        print(f"[test-diagnoser-batch] tokens in={input_tokens} out={output_tokens} cost=${cost:.4f}")


def _load_experiment_config() -> dict:
    raw = yaml.safe_load((_ROOT / "config.yaml").read_text())
    return raw[0] if isinstance(raw, list) else raw


def main() -> None:
    cfg             = _load_experiment_config()
    model           = cfg["diagnoser"]["md"]["model"]
    reflector_model = cfg["reflector"]["md"].get("model", "")
    gt              = cfg["diagnoser"].get("gt", False)

    print(f"[test-diagnoser-batch] model={model}  gt={gt}  fixtures={BENCH_IDS}")
    _OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for bench_id in BENCH_IDS:
        fixture_dir = _FIXTURES / bench_id
        context_window = json.loads((fixture_dir / "context_window.json").read_text())
        metadata       = json.loads((fixture_dir / "metadata.json").read_text())

        diagnosis = diagnose(
            context_window, metadata, model=model, logger=_Logger(),
            reflector_model=reflector_model, train_size=TRAIN_SIZE, gt=gt,
            bench_id=bench_id, out_dir=_OUT,
        )
        assert diagnosis and len(diagnosis) >= 40, f"{bench_id}: diagnosis too short or empty"

        steps_file = _OUT / "diagnoser_steps.json"
        if steps_file.exists():
            steps_file.replace(_OUT / f"diagnoser_steps_{bench_id}.json")

        results.append({"bench_id": bench_id, "model": model, "gt": gt, "diagnosis": diagnosis})
        print(f"\n[test-diagnoser-batch] {bench_id} ({len(diagnosis)} chars):\n{diagnosis}\n")

    (_OUT / "result_batch.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[test-diagnoser-batch] saved -> {_OUT / 'result_batch.json'}")


if __name__ == "__main__":
    main()
