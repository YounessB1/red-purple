import json
import shutil
import uuid
from pathlib import Path

from source.optimize_anything.opencode_trace import last_agent_text, run_opencode_agent

_ROOT = Path(__file__).resolve().parents[2]
_SOLUTIONS_DIR = _ROOT / "xbow" / "solutions"

_MAX_RETRIES = 2
_TIMEOUT = 300


def _load_ground_truth(bench_id: str) -> str | None:
    p = _SOLUTIONS_DIR / f"{bench_id}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def _extract_json_obj(text: str) -> dict | None:
    """Find and parse the last JSON object in text, handling nested braces."""
    for start in reversed([i for i, c in enumerate(text) if c == "{"]):
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and "score" in obj:
                            return obj
                    except json.JSONDecodeError:
                        break
    return None


def llm_judge(
    context_window: list,
    bench_id: str,
    model: str,
    logger=None,
    gt: bool = False,
    out_dir: Path | None = None,
) -> tuple[float, str]:
    """Score agent progress 0.0–0.9 for a failed run. Returns (score, reason)."""
    workdir = _ROOT / "tmp" / f"scorer_{bench_id}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    input_tokens = output_tokens = 0
    cost = 0.0
    steps: list = []
    try:
        (workdir / "context_window.json").write_text(
            json.dumps(context_window, indent=2), encoding="utf-8"
        )
        last_step = context_window[-1] if context_window else {}
        metadata = {
            "benchmark_id": bench_id,
            "iterations": len(context_window),
            "stop_reason": last_step.get("finish_reason", "unknown"),
            "success": False,
        }
        (workdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        gt_tag = ""
        if gt:
            gt_text = _load_ground_truth(bench_id)
            if gt_text:
                (workdir / "ground_truth.md").write_text(gt_text, encoding="utf-8")
                gt_tag = " [gt]"

        label = f"judge {bench_id}"
        for attempt in range(_MAX_RETRIES):
            timed_out, input_tokens, output_tokens, cost, steps = run_opencode_agent(
                "scorer", model, workdir, "Score this CTF agent run.", _TIMEOUT, label,
            )
            if timed_out:
                print(f"[judge] {bench_id} — timeout after {_TIMEOUT}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                if attempt + 1 < _MAX_RETRIES:
                    continue
                break

            content = last_agent_text(steps)
            data = _extract_json_obj(content)
            if data is not None:
                try:
                    score = max(0.0, min(0.9, round(float(data["score"]), 1)))
                    reason = str(data.get("reason", ""))
                    if logger is not None:
                        logger.log_scorer(input_tokens, output_tokens, cost)
                    print(f"[judge]{gt_tag} {bench_id} — {score} | {reason}")
                    return score, reason
                except (ValueError, TypeError) as e:
                    print(f"[judge] {bench_id} — bad score value: {e} (attempt {attempt + 1}/{_MAX_RETRIES})")

            if attempt + 1 < _MAX_RETRIES:
                print(f"[judge] {bench_id} — no valid JSON in output, retrying ({attempt + 2}/{_MAX_RETRIES})")

        if logger is not None:
            logger.log_scorer(input_tokens, output_tokens, cost)
        print(f"[judge]{gt_tag} {bench_id} — 0.0 | fallback after {_MAX_RETRIES} attempts")
    except Exception as e:
        print(f"[judge] {bench_id} — unexpected error: {e}")
    finally:
        if out_dir is not None and steps:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "scorer_steps.json").write_text(
                    json.dumps(steps, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass
        shutil.rmtree(workdir, ignore_errors=True)
    return 0.0, ""
