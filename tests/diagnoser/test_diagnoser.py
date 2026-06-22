"""Test diagnoser on a real failed CTF run.

Verifies that:
  1. diagnosis output is non-empty and plausible.
  2. diagnoser_steps.json is written to out_dir via the new out_dir param.
  3. Prints the full reasoning trace for quality inspection.

Fixtures: tests/diagnoser/fixtures/XBEN-043-24/
Output:   tests/diagnoser/output/result.json
          tests/diagnoser/output/diagnoser_steps.json
"""

import json
import yaml
from pathlib import Path

from source.optimize_anything.diagnoser import diagnose

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures" / "XBEN-043-24"
_OUT      = Path(__file__).parent / "output"

BENCH_ID   = "XBEN-043-24"
TRAIN_SIZE = 3


class _Logger:
    def log_diagnoser(self, input_tokens, output_tokens, cost):
        print(f"[test-diagnoser] tokens in={input_tokens} out={output_tokens} cost=${cost:.4f}")


def _print_trace(steps: list) -> None:
    print(f"\n[test-diagnoser] trace — {len(steps)} steps:")
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
    cfg             = yaml.safe_load((_ROOT / "config.yaml").read_text())
    model           = cfg["diagnoser"]["md"]["model"]
    reflector_model = cfg["reflector"]["md"].get("model", "")
    gt              = cfg["diagnoser"].get("gt", False)

    context_window = json.loads((_FIXTURES / "context_window.json").read_text())
    metadata       = json.loads((_FIXTURES / "metadata.json").read_text())

    print(f"[test-diagnoser] model={model}  bench_id={BENCH_ID}  gt={gt}")
    print(f"[test-diagnoser] run: stop_reason={metadata['stop_reason']}  "
          f"iterations={metadata['iterations_used']}  success={metadata['success']}")

    _OUT.mkdir(parents=True, exist_ok=True)

    diagnosis = diagnose(
        context_window,
        metadata,
        model=model,
        logger=_Logger(),
        reflector_model=reflector_model,
        train_size=TRAIN_SIZE,
        gt=gt,
        bench_id=BENCH_ID,
        out_dir=_OUT,
    )

    assert diagnosis and len(diagnosis) >= 40, "Diagnosis too short or empty"

    # ── Inspect steps file ────────────────────────────────────────────────
    steps_file = _OUT / "diagnoser_steps.json"
    if steps_file.exists():
        steps = json.loads(steps_file.read_text())
        _print_trace(steps)
    else:
        print("[test-diagnoser] WARNING: diagnoser_steps.json was not written")

    # ── Print diagnosis ───────────────────────────────────────────────────
    print(f"\n[test-diagnoser] diagnosis ({len(diagnosis)} chars):")
    print(diagnosis)

    result = {"bench_id": BENCH_ID, "model": model, "gt": gt, "diagnosis": diagnosis}
    (_OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[test-diagnoser] saved → {_OUT / 'result.json'}")
    print(f"[test-diagnoser] saved → {steps_file}")


if __name__ == "__main__":
    main()
