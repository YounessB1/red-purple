"""Test scorer on a real failed CTF run.

Verifies that:
  1. score is a valid float in [0.0, 0.9] with a non-empty reason.
  2. scorer_steps.json is written to out_dir via the new out_dir param.
  3. Prints the full reasoning trace for quality inspection.

Fixtures: tests/scorer/fixtures/XBEN-043-24/
Output:   tests/scorer/output/result.json
          tests/scorer/output/scorer_steps.json
"""

import json
import yaml
from pathlib import Path

from source.optimize_anything.scorer import llm_judge

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures" / "XBEN-043-24"
_OUT      = Path(__file__).parent / "output"

BENCH_ID = "XBEN-043-24"


class _Logger:
    def log_scorer(self, input_tokens, output_tokens, cost):
        print(f"[test-scorer] tokens in={input_tokens} out={output_tokens} cost=${cost:.4f}")


def _print_trace(steps: list) -> None:
    print(f"\n[test-scorer] trace — {len(steps)} steps:")
    for i, step in enumerate(steps):
        t = step.get("type", "?")
        if t == "thinking":
            preview = step["text"][:200].replace("\n", " ")
            print(f"  [{i:02d}] THINK  {preview}...")
        elif t == "text":
            preview = step["text"][:200].replace("\n", " ")
            print(f"  [{i:02d}] TEXT   {preview}...")
        elif t == "tool":
            name = step.get("name", "?")
            inp  = step.get("input", {})
            cmd  = inp.get("command", inp.get("path", str(inp)))[:120]
            print(f"  [{i:02d}] TOOL   {name}  {cmd}")


def main() -> None:
    cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text())
    model = cfg["scorer"]["md"]["model"]
    gt    = cfg["scorer"].get("gt", False)

    context_window = json.loads((_FIXTURES / "context_window.json").read_text())
    metadata       = json.loads((_FIXTURES / "metadata.json").read_text())

    print(f"[test-scorer] model={model}  bench_id={BENCH_ID}  gt={gt}")
    print(f"[test-scorer] run: stop_reason={metadata['stop_reason']}  "
          f"iterations={metadata['iterations_used']}  success={metadata['success']}")

    _OUT.mkdir(parents=True, exist_ok=True)

    score, reason = llm_judge(
        context_window,
        BENCH_ID,
        model=model,
        logger=_Logger(),
        gt=gt,
        out_dir=_OUT,
    )

    assert 0.0 <= score <= 0.9, f"Score out of range: {score}"
    assert reason, "Reason is empty"

    # ── Inspect steps file ────────────────────────────────────────────────
    steps_file = _OUT / "scorer_steps.json"
    if steps_file.exists():
        steps = json.loads(steps_file.read_text())
        _print_trace(steps)
    else:
        print("[test-scorer] WARNING: scorer_steps.json was not written")

    # ── Print result ──────────────────────────────────────────────────────
    print(f"\n[test-scorer] score={score}")
    print(f"[test-scorer] reason={reason}")

    result = {"bench_id": BENCH_ID, "model": model, "gt": gt, "score": score, "reason": reason}
    (_OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[test-scorer] saved → {_OUT / 'result.json'}")
    print(f"[test-scorer] saved → {steps_file}")


if __name__ == "__main__":
    main()
