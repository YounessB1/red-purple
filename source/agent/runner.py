"""Agent runner — spawns an OpenCode CTF agent per run."""

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_BIN = _REPO_ROOT / "node_modules" / ".bin" / "opencode"
_OPENCODE_BIN: str = str(_LOCAL_BIN) if _LOCAL_BIN.exists() else (shutil.which("opencode") or "opencode")


def _materialize_files(workdir: Path, files: dict) -> None:
    for rel_path, content in files.items():
        dest = workdir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")



def _inject_prompt(workdir: Path) -> None:
    """Append prompt.md content as the body of ctf-agent.md (after frontmatter)."""
    prompt_path = workdir / "prompt.md"
    agent_md_path = workdir / ".opencode" / "agents" / "ctf-agent.md"
    if not prompt_path.exists() or not agent_md_path.exists():
        return
    prompt_content = prompt_path.read_text(encoding="utf-8")
    agent_md = agent_md_path.read_text(encoding="utf-8")
    parts = agent_md.split("---", 2)
    if len(parts) >= 3:
        agent_md_path.write_text(f"---{parts[1]}---\n\n{prompt_content}", encoding="utf-8")



def _trace_session(workdir: Path) -> tuple[dict, list]:
    """Read token usage and conversation parts from OpenCode's SQLite DB."""
    db_path = Path.home() / ".local/share/opencode/opencode.db"
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception:
        return {}, []

    try:
        row = db.execute(
            "SELECT id, cost, tokens_input, tokens_output FROM session "
            "WHERE directory=? ORDER BY time_created DESC LIMIT 1",
            (str(workdir),),
        ).fetchone()
        if not row:
            return {}, []

        session_id, cost, input_tokens, output_tokens = row

        parts = db.execute(
            "SELECT data FROM part WHERE session_id=? ORDER BY time_created",
            (session_id,),
        ).fetchall()

        steps: list[dict] = []
        all_text_parts: list[str] = []
        current_step: dict | None = None
        step_num = 0

        for (raw_part,) in parts:
            try:
                part = json.loads(raw_part)
            except Exception:
                continue

            ptype = part.get("type")

            if ptype == "step-start":
                step_num += 1
                current_step = {
                    "step": step_num,
                    "thought": None,
                    "tool": None,
                    "input": None,
                    "output": None,
                    "truncated": False,
                    "tokens": None,
                    "cost_usd": None,
                    "finish_reason": None,
                }

            elif ptype == "step-finish" and current_step is not None:
                current_step["finish_reason"] = part.get("reason")
                tok = part.get("tokens", {})
                cache = tok.get("cache", {})
                current_step["tokens"] = {
                    "input": tok.get("input", 0),
                    "output": tok.get("output", 0),
                    "cache_read": cache.get("read", 0),
                    "cache_write": cache.get("write", 0),
                }
                current_step["cost_usd"] = part.get("cost")
                steps.append(current_step)
                current_step = None

            elif ptype == "text" and part.get("text", "").strip():
                text = part["text"].strip()
                all_text_parts.append(text)
                if current_step is not None:
                    current_step["thought"] = (
                        current_step["thought"] + "\n\n" + text
                        if current_step["thought"] else text
                    )

            elif ptype == "reasoning" and part.get("text", "").strip():
                text = part["text"].strip()
                all_text_parts.append(text)
                if current_step is not None:
                    block = f"<thinking>\n{text}\n</thinking>"
                    current_step["thought"] = (
                        block + "\n\n" + current_step["thought"]
                        if current_step["thought"] else block
                    )

            elif ptype == "tool" and current_step is not None:
                state = part.get("state", {}) if isinstance(part.get("state"), dict) else {}
                output = str(state.get("output", ""))
                truncated = bool((state.get("metadata") or {}).get("truncated", False))
                current_step["tool"] = part.get("tool", "?")
                current_step["input"] = state.get("input", {})
                current_step["output"] = output or None
                current_step["truncated"] = truncated
                if output:
                    all_text_parts.append(output)

        tool_calls = sum(1 for s in steps if s["tool"] is not None)
        llm_calls = len(steps)

        extras = {
            "all_text": "\n".join(all_text_parts),
            "total_input_tokens": input_tokens or 0,
            "total_output_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
            "total_cost_usd": round(cost or 0, 6),
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "context_messages": len(steps),
        }
        return extras, steps

    finally:
        db.close()


def _find_flag(text: str) -> str | None:
    m = re.search(r'FLAG\{([0-9a-f]{64})\}', text)
    return f"FLAG{{{m.group(1)}}}" if m else None


def _read_agent_field(files: dict, field: str) -> str:
    """Extract a frontmatter field value from ctf-agent.md."""
    content = files.get(".opencode/agents/ctf-agent.md", "")
    m = re.search(rf'(?m)^{field}:\s*"?([^"\n]+)"?', content)
    return m.group(1).strip() if m else ""


def _stop_reason(
    *,
    returncode: int | None,
    killed_by_cancel: bool,
    timed_out: bool,
    context_window: list,
    max_steps: int,
    llm_calls: int,
    success: bool,
) -> str:
    agent_said_stop = bool(context_window) and context_window[-1].get("finish_reason") == "stop"

    if not success and (timed_out or llm_calls >= max_steps):
        return "max_steps_reached"
    if agent_said_stop or (returncode == 0 and not killed_by_cancel and not timed_out):
        return "agent_finished"
    if killed_by_cancel:
        return "cancelled"
    return "error"


def run(
    target: str,
    run_id: str | None = None,
    candidate: dict | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict, list]:
    """Run the OpenCode CTF agent. Returns (metadata, context_window)."""
    run_id = run_id or f"run-{uuid4().hex[:8]}"
    workdir = Path(tempfile.mkdtemp(prefix=f"agent_{run_id}_"))
    started_at = datetime.now(timezone.utc).isoformat()

    print(f"[red-purple] {run_id} | {target}", flush=True)

    files = (candidate or {}).get("files", {})
    model = _read_agent_field(files, "model")
    max_steps = int(_read_agent_field(files, "maxSteps") or 50)

    stdout = ""
    extras: dict = {}
    context_window: list = []
    tool_counts: dict[str, int] = {}
    timed_out = False
    killed_by_cancel = threading.Event()
    proc_returncode: int | None = None
    error_detail: str | None = None

    try:
        _materialize_files(workdir, files)
        _inject_prompt(workdir)

        proc = subprocess.Popen(
            [
                _OPENCODE_BIN, "run",
                "--agent", "ctf-agent",
                "--dir", str(workdir),
                f"The target is at {target}. Find and report the flag.",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        watcher_stop = threading.Event()

        def _watch() -> None:
            while not watcher_stop.is_set() and proc.poll() is None:
                if cancel_event and cancel_event.is_set():
                    proc.kill()
                    killed_by_cancel.set()
                    return
                time.sleep(1)

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

        try:
            stdout, _ = proc.communicate(timeout=max_steps * 120)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            stdout, _ = proc.communicate()
            timed_out = True

        watcher_stop.set()
        watcher.join(timeout=2)
        proc_returncode = proc.returncode

        extras, context_window = _trace_session(workdir)
        for s in context_window:
            if s.get("tool"):
                tool_counts[s["tool"]] = tool_counts.get(s["tool"], 0) + 1

    except Exception as e:
        error_detail = f"{type(e).__name__}: {e}"
        print(f"[red-purple] {run_id} exception: {error_detail}", flush=True)

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    finished_at = datetime.now(timezone.utc).isoformat()
    duration = (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    ).total_seconds()

    all_text = extras.get("all_text", "") + "\n" + stdout
    flag = _find_flag(all_text)
    success = flag is not None

    stop_reason = "error" if error_detail else _stop_reason(
        returncode=proc_returncode,
        killed_by_cancel=killed_by_cancel.is_set(),
        timed_out=timed_out,
        context_window=context_window,
        max_steps=max_steps,
        llm_calls=extras.get("llm_calls", 0),
        success=success,
    )
    if stop_reason == "error" and not error_detail and proc_returncode is not None:
        error_detail = f"exit code {proc_returncode}"

    metadata = {
        "run_id": run_id,
        "target": target,
        "model": model,
        "success": success,
        "flag": flag,
        "stop_reason": stop_reason,
        "error_detail": error_detail,
        "duration_seconds": round(duration, 2),
        "iterations_used": extras.get("tool_calls", 0),
        "max_iterations": max_steps,
        "llm_calls": extras.get("llm_calls", 0),
        "tool_calls": extras.get("tool_calls", 0),
        "total_input_tokens": extras.get("total_input_tokens", 0),
        "total_output_tokens": extras.get("total_output_tokens", 0),
        "total_tokens": extras.get("total_tokens", 0),
        "total_cost_usd": extras.get("total_cost_usd", 0.0),
        "context_messages": extras.get("context_messages", 0),
        "tool_counts": tool_counts,
    }

    outcome = f"FLAG {flag}" if success else f"no flag ({stop_reason})"
    print(f"[red-purple] {run_id} done | {outcome}", flush=True)

    return metadata, context_window
