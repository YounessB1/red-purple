"""Diagnoser — spawns an OpenCode agent to produce a compact failure analysis."""

import json
import shutil
import subprocess
import uuid
from pathlib import Path

from source.optimize_anything.opencode_trace import last_agent_text, trace_opencode_session

_ROOT = Path(__file__).resolve().parents[2]

_MAX_RETRIES = 2
_TIMEOUT = 300
_MIN_OUTPUT_LEN = 40  # anything shorter is likely an incomplete response


def diagnose(
    context_window: list[dict],
    metadata: dict,
    model: str,
    logger,
    reflector_model: str = "",
    train_size: int = 1,
) -> str:
    bench_id = metadata.get("benchmark_id", "unknown")
    workdir = _ROOT / "tmp" / f"diagnoser_{bench_id}_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    input_tokens = output_tokens = 0
    try:
        (workdir / "context_window.json").write_text(
            json.dumps(context_window, indent=2), encoding="utf-8"
        )
        (workdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        for attempt in range(_MAX_RETRIES):
            try:
                proc = subprocess.run(
                    ["opencode", "run", "--agent", "diagnoser", "--model", model,
                     "--dir", str(workdir), "Diagnose this CTF agent failure."],
                    capture_output=True, text=True, timeout=_TIMEOUT,
                )
                if proc.returncode != 0 and proc.stderr:
                    print(f"[diagnoser] {bench_id} — opencode stderr: {proc.stderr.strip()[:200]}")
            except subprocess.TimeoutExpired:
                print(f"[diagnoser] {bench_id} — timeout after {_TIMEOUT}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                if attempt + 1 < _MAX_RETRIES:
                    continue
                break

            input_tokens, output_tokens, _, steps = trace_opencode_session(workdir)
            result = last_agent_text(steps)
            if result and len(result) >= _MIN_OUTPUT_LEN:
                return result

            print(f"[diagnoser] {bench_id} — output too short or empty (attempt {attempt + 1}/{_MAX_RETRIES})")
            if attempt + 1 < _MAX_RETRIES:
                print(f"[diagnoser] {bench_id} — retrying ({attempt + 2}/{_MAX_RETRIES})")

        print(f"[diagnoser] {bench_id} — all attempts failed, using fallback")
        return f"Diagnosis unavailable for {bench_id} (agent produced no output after {_MAX_RETRIES} attempts)."
    except Exception as e:
        print(f"[diagnoser] {bench_id} — unexpected error: {e}")
        return f"Diagnosis unavailable for {bench_id} (error: {e})."
    finally:
        try:
            logger.log_diagnoser(input_tokens, output_tokens)
        except Exception:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
