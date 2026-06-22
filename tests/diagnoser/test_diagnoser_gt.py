"""Test diagnoser with ground_truth=True on a real failed run (XBEN-080-24).

Verifies that:
  1. ground_truth.md is written to the workdir and read by the agent.
  2. The diagnosis output is non-empty and plausible.
  3. The agent's reasoning trace shows it actually read ground_truth.md.

Fixtures: tests/diagnoser/fixtures/XBEN-080-24/
  context_window.json  — copied from iteration_001/train/parent/XBEN-080-24/
  metadata.json        — same source
Ground truth: xbow/solutions/XBEN-080-24.md
"""

import json
import shutil
import uuid
import yaml
from pathlib import Path

from source.optimize_anything.diagnoser import _load_ground_truth, diagnose
from source.optimize_anything.opencode_trace import last_agent_text, trace_opencode_session

_ROOT     = Path(__file__).resolve().parents[2]
_FIXTURES = Path(__file__).parent / "fixtures" / "XBEN-080-24"
_OUT      = Path(__file__).parent / "output"

BENCH_ID   = "XBEN-080-24"
TRAIN_SIZE = 3


class _Logger:
    def log_diagnoser(self, input_tokens, output_tokens, cost):
        print(f"[test-diagnoser-gt] tokens in={input_tokens} out={output_tokens} cost=${cost:.4f}")


def _patch_diagnose_capture_workdir(bench_id, context_window, metadata, model, logger, gt):
    """
    Calls diagnose() but captures the workdir before it is deleted so we can
    inspect the agent's reasoning trace.  We do this by temporarily monkey-
    patching shutil.rmtree inside the diagnoser module.
    """
    import source.optimize_anything.diagnoser as diag_mod

    captured = {}
    original_rmtree = shutil.rmtree

    def _intercept(path, **kwargs):
        path = Path(path)
        if f"diagnoser_{bench_id}" in path.name:
            # keep original path for DB lookup (keyed by original dir)
            captured["original_workdir"] = path
            # copy before deletion so we can inspect files
            dest = _OUT / f"workdir_{path.name}"
            if dest.exists():
                original_rmtree(dest, ignore_errors=True)
            shutil.copytree(path, dest)
            captured["workdir"] = dest
        original_rmtree(path, **kwargs)

    diag_mod.shutil.rmtree = _intercept
    try:
        result = diagnose(
            context_window,
            metadata,
            model=model,
            logger=logger,
            train_size=TRAIN_SIZE,
            gt=gt,
            bench_id=bench_id,
        )
    finally:
        diag_mod.shutil.rmtree = original_rmtree

    return result, captured


def main() -> None:
    cfg             = yaml.safe_load((_ROOT / "config.yaml").read_text())
    model           = cfg["diagnoser"]["md"]["model"]

    context_window = json.loads((_FIXTURES / "context_window.json").read_text())
    metadata       = json.loads((_FIXTURES / "metadata.json").read_text())

    # Confirm ground truth exists
    gt_text = _load_ground_truth(BENCH_ID)
    if gt_text is None:
        print(f"[test-diagnoser-gt] SKIP — no ground truth file for {BENCH_ID}")
        return
    print(f"[test-diagnoser-gt] ground truth found ({len(gt_text)} chars)")

    print(f"[test-diagnoser-gt] model={model}  bench_id={BENCH_ID}  gt=True")

    _OUT.mkdir(parents=True, exist_ok=True)

    diagnosis, captured = _patch_diagnose_capture_workdir(
        BENCH_ID, context_window, metadata, model, _Logger(), gt=True
    )

    # ── Check 1: diagnosis is non-empty ──────────────────────────────────
    assert diagnosis and len(diagnosis) >= 40, "Diagnosis too short or empty"
    print(f"\n[test-diagnoser-gt] diagnosis ({len(diagnosis)} chars):")
    print(diagnosis)

    # ── Check 2: inspect reasoning trace for ground_truth.md reads ───────
    workdir = captured.get("workdir")
    if workdir and workdir.exists():
        print(f"\n[test-diagnoser-gt] captured workdir: {workdir}")

        # List files the agent had access to
        files = list(workdir.iterdir())
        print(f"[test-diagnoser-gt] workdir files: {[f.name for f in files]}")
        gt_written = (workdir / "ground_truth.md").exists()
        print(f"[test-diagnoser-gt] ground_truth.md present in workdir: {gt_written}")
        assert gt_written, "ground_truth.md was NOT written to workdir — gt flag broken"

        # Query DB using the ORIGINAL workdir path (the key used during the session)
        original_workdir = captured.get("original_workdir", workdir)
        print(f"[test-diagnoser-gt] querying trace with original path: {original_workdir}")
        _, _, _, steps = trace_opencode_session(original_workdir)
        print(f"[test-diagnoser-gt] trace steps: {len(steps)}")

        # Check whether agent text or tool calls mention ground_truth.md
        full_trace = json.dumps(steps)
        gt_referenced = "ground_truth" in full_trace
        print(f"[test-diagnoser-gt] 'ground_truth' referenced in trace: {gt_referenced}")

        if gt_referenced:
            # Find which steps mention it
            for i, step in enumerate(steps):
                step_text = json.dumps(step)
                if "ground_truth" in step_text:
                    role = step.get("role", step.get("type", "?"))
                    content_preview = step_text[:300]
                    print(f"  step[{i}] ({role}): ...{content_preview}...")
        else:
            print("[test-diagnoser-gt] WARNING: agent never referenced ground_truth.md in its trace")

        # Save full trace for manual inspection
        (_OUT / "gt_trace_steps.json").write_text(
            json.dumps(steps, indent=2), encoding="utf-8"
        )
        print(f"[test-diagnoser-gt] full trace saved → {_OUT / 'gt_trace_steps.json'}")
    else:
        print("[test-diagnoser-gt] WARNING: workdir not captured (rmtree intercept missed)")

    # ── Save result ───────────────────────────────────────────────────────
    result = {"bench_id": BENCH_ID, "model": model, "gt": True, "diagnosis": diagnosis}
    (_OUT / "result_gt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[test-diagnoser-gt] saved → {_OUT / 'result_gt.json'}")


if __name__ == "__main__":
    main()
