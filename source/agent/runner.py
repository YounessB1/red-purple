"""Agent runner — executes a seed module and traces the run.

Accepts any seed module/namespace that exposes: TOOL_SCHEMAS, TOOLS, PROMPT.
Defaults to source.agent.seed when no module is provided.

Usage:
    # From CLI / server (reads env vars)
    from source.agent.runner import run
    run()

    # From optimize-anything (explicit params, returns metadata)
    metadata = run(target="http://localhost:8080", run_id="exp1-bench42", seed=my_seed)
"""

import os
from pathlib import Path
from uuid import uuid4

from source.agent import seed as default_seed
from source.llm import LLM
from source.agent.utils import parse_tool_calls, run_tool
from source.tracer import Tracer

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def run(
    target: str | None = None,
    run_id: str | None = None,
    seed=None,
    max_iter: int | None = None,
    task: str | None = None,
    runs_dir: Path | None = None,
    model: str | None = None,
) -> dict:
    """Run the agent loop. Returns metadata dict.

    All parameters are optional — when omitted, falls back to env vars
    (TARGET, RUN_ID, MAX_ITER, TASK) for backward compatibility with
    the container server.
    """
    if seed is None:
        seed = default_seed

    target = target or os.environ.get("TARGET", "")
    if not target:
        print("ERROR: TARGET not set.")
        return {"success": False, "stop_reason": "error"}

    max_iter = max_iter or int(os.environ.get("MAX_ITER", "100"))
    run_id = run_id or os.environ.get("RUN_ID") or f"run-{uuid4().hex[:8]}"
    task_override = task or os.environ.get("TASK", "")
    prompt = _build_prompt(seed=seed, target=target, task_override=task_override)
    runs_dir = runs_dir or RUNS_DIR

    tracer = Tracer(run_id=run_id, target=target, task=prompt,
                    model=model,
                    runs_dir=runs_dir, max_iterations=max_iter)
    llm = LLM(model=model, tracer=tracer)

    history = [
        {"role": "system", "content": prompt},
    ]

    print(f"[red-purple] {run_id} | {target}\n")

    stop_reason = "unknown"
    try:
        for iteration in range(1, max_iter + 1):
            content = llm.generate(history)
            calls = parse_tool_calls(content)
            history.append({"role": "assistant", "content": content})

            if iteration == max_iter:
                stop_reason = "max_iterations"
                break

            if not calls:
                history.append({"role": "user", "content": "You must call a tool to continue. Use terminal_execute to run a command, or finish to report your result."})
                continue

            finished = False
            for call in calls:
                result, should_finish = run_tool(call, tracer, seed.TOOLS)
                history.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                if should_finish:
                    stop_reason = "agent_finished"
                    finished = True
                    break

            if finished:
                break
    except Exception:
        stop_reason = "error"
        raise
    finally:
        tracer.set_stop_reason(stop_reason)
        metadata, context_window = tracer.finish(history)
        flag = metadata.get("flag")
        outcome = f"FLAG {flag}" if flag else f"no flag ({stop_reason})"
        print(f"[red-purple] {run_id} done | {outcome}")

    return metadata, context_window


def _build_prompt(seed, target: str, task_override: str) -> str:
    """Format the single prompt sent to the agent model."""
    if hasattr(seed, "PROMPT"):
        prompt_template = seed.PROMPT
    else:
        # Backward compatibility during migration from two prompt components.
        prompt_template = f"{seed.SYSTEM_PROMPT}\n\n{seed.DEFAULT_TASK}"

    tool_str = str(seed.TOOL_SCHEMAS)
    prompt = prompt_template.replace("{target}", target)
    if "{tools}" in prompt:
        prompt = prompt.replace("{tools}", tool_str)
    else:
        prompt = f"{prompt}\n\n# TOOLS\n{tool_str}"
    if task_override:
        prompt = f"{prompt}\n\n# TASK OVERRIDE\n{task_override}"
    return prompt
