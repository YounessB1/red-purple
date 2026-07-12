"""Diagnoser — spawns an OpenCode agent to produce a compact failure analysis."""

import json
import shutil
import uuid
from pathlib import Path

from source.optimize_anything.opencode_trace import last_agent_text, run_opencode_agent

_ROOT = Path(__file__).resolve().parents[2]
_SOLUTIONS_DIR = _ROOT / "xbow" / "solutions"

_MAX_RETRIES = 2
_TIMEOUT = 300
_MIN_OUTPUT_LEN = 40  # anything shorter is likely an incomplete response


def _load_ground_truth(bench_id: str) -> str | None:
    p = _SOLUTIONS_DIR / f"{bench_id}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def diagnose(
    context_window: list[dict],
    metadata: dict,
    model: str,
    logger,
    reflector_model: str = "",
    train_size: int = 1,
    gt: bool = False,
    bench_id: str = "",
    out_dir: Path | None = None,
) -> str | None:
    bench_id = bench_id or metadata.get("benchmark_id", "unknown")
    workdir = _ROOT / "tmp" / f"diagnoser_{bench_id}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    input_tokens = output_tokens = 0
    cost = 0.0
    steps: list = []
    try:
        (workdir / "context_window.json").write_text(
            json.dumps(context_window, indent=2), encoding="utf-8"
        )
        (workdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        gt_tag = ""
        if gt:
            gt_text = _load_ground_truth(bench_id)
            if gt_text:
                (workdir / "ground_truth.md").write_text(gt_text, encoding="utf-8")
                gt_tag = " [gt]"

        label = f"diagnoser {bench_id}"
        for attempt in range(_MAX_RETRIES):
            timed_out, input_tokens, output_tokens, cost, steps = run_opencode_agent(
                "diagnoser", model, workdir, "Diagnose this CTF agent failure.", _TIMEOUT, label,
            )
            if timed_out:
                print(f"[diagnoser] {bench_id} — timeout after {_TIMEOUT}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                if attempt + 1 < _MAX_RETRIES:
                    continue
                break

            result = last_agent_text(steps)
            if result and len(result) >= _MIN_OUTPUT_LEN:
                print(f"[diagnoser]{gt_tag} {bench_id} — ok ({len(result)} chars)")
                return result

            print(f"[diagnoser]{gt_tag} {bench_id} — output too short or empty (attempt {attempt + 1}/{_MAX_RETRIES})")
            if attempt + 1 < _MAX_RETRIES:
                print(f"[diagnoser]{gt_tag} {bench_id} — retrying ({attempt + 2}/{_MAX_RETRIES})")

        print(f"[diagnoser]{gt_tag} {bench_id} — all attempts failed, giving up")
        return None
    except Exception as e:
        print(f"[diagnoser] {bench_id} — unexpected error: {e}")
        return None
    finally:
        try:
            logger.log_diagnoser(input_tokens, output_tokens, cost)
        except Exception:
            pass
        if out_dir is not None and steps:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "diagnoser_steps.json").write_text(
                    json.dumps(steps, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass
        shutil.rmtree(workdir, ignore_errors=True)
