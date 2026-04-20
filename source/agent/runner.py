"""Agent runner — executes a seed module and traces the run.

Accepts any seed module that exposes: TOOL_SCHEMAS, TOOLS, SYSTEM_PROMPT, DEFAULT_TASK.
Defaults to source.agent.seed when no module is provided.
"""

import os
from pathlib import Path
from uuid import uuid4

from source.agent import seed as default_seed
from source.llm import LLM
from source.agent.utils import parse_tool_calls, run_tool
from source.tracer import Tracer

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def run(seed=None) -> None:
    if seed is None:
        seed = default_seed

    target   = os.environ.get("TARGET", "")
    if not target:
        print("ERROR: TARGET not set.")
        return
    max_iter = int(os.environ.get("MAX_ITER", "100"))
    task     = os.environ.get("TASK") or seed.DEFAULT_TASK.format(target=target)
    run_id   = os.environ.get("RUN_ID") or f"run-{uuid4().hex[:8]}"

    tracer = Tracer(run_id=run_id, target=target, task=task,
                    model=os.environ.get("REDPURPLE_LLM", ""),
                    runs_dir=RUNS_DIR, max_iterations=max_iter)
    llm = LLM(tracer=tracer)

    history = [
        {"role": "system", "content": seed.SYSTEM_PROMPT.format(tools=seed.TOOL_SCHEMAS)},
        {"role": "user",   "content": task},
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
                print(f"[tool] {call['name']}({call['args']})")
                result, should_finish = run_tool(call, tracer, seed.TOOLS)
                print(f"[result] {result}\n")
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
        tracer.finish(history)
