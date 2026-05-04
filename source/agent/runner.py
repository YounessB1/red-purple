"""Agent runner — executes a seed module and traces the run.

Accepts any seed module/namespace that exposes: TOOL_SCHEMAS, TOOLS, PROMPT.
Defaults to source.agent.seed when no seed is provided.
"""

import functools
import shutil
import tempfile
import threading
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from source import seed as default_seed
from source.agent import base_prompt as agent_base, tools as agent_tools
from source.llm import LLM
from source.agent.utils import parse_tool_calls, run_tool
from source.agent import compactor, extractor
from source.tracer import Tracer

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def _build_prompt(target: str, seed_instructions: str) -> str:
    return agent_base.BASE_PROMPT.replace("{target}", target) + seed_instructions


def run(
    target: str,
    model: str,
    run_id: str | None = None,
    prompt: str | None = None,
    max_iter: int = 50,
    runs_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run the agent loop. Returns (metadata, context_window)."""
    run_id = run_id or f"run-{uuid4().hex[:8]}"
    prompt = _build_prompt(target, prompt or default_seed.PROMPT)
    runs_dir = runs_dir or RUNS_DIR

    workdir = Path(tempfile.mkdtemp(prefix=f"agent_{run_id}_"))
    tools = {
        "terminal_execute": functools.partial(agent_tools.terminal_execute, cwd=str(workdir)),
        "finish": agent_tools.finish,
    }

    tracer = Tracer(run_id=run_id, target=target, task=prompt,
                    model=model,
                    runs_dir=runs_dir, max_iterations=max_iter)
    llm = LLM(model=model)

    history = [
        {"role": "system", "content": prompt},
    ]

    print(f"[red-purple] {run_id} | {target}\n")

    stop_reason = "unknown"
    try:
        for iteration in tqdm(range(1, max_iter + 1), desc=run_id, unit="iter", leave=False):
            if cancel_event and cancel_event.is_set():
                stop_reason = "cancelled"
                break
            if compactor.should_compact(history, model):
                history = compactor.compact(history, model, tracer=tracer)
            try:
                content, input_tokens, output_tokens = llm.generate(history)
                tracer.log_llm_call(input_tokens, output_tokens)
            except Exception as e:
                stop_reason = f"llm_error: {type(e).__name__}"
                history.append({"role": "user", "content": f"<error>\n{e}\n</error>"})
                print(f"[red-purple] {run_id} | stopping: {stop_reason}", flush=True)
                break
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
                result, should_finish, latency_ms = run_tool(call, tools)
                tracer.log_tool_call(call["name"], call["args"], latency_ms)
                result = extractor.extract(result, model=model, tracer=tracer)
                history.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                if should_finish:
                    stop_reason = "agent_finished"
                    finished = True
                    break

            if finished:
                break
    except Exception as e:
        if stop_reason == "unknown":
            stop_reason = f"error: {type(e).__name__}"
        import traceback; traceback.print_exc()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        tracer.set_stop_reason(stop_reason)
        metadata, context_window = tracer.finish(history)
        flag = metadata.get("flag")
        outcome = f"FLAG {flag}" if flag else f"no flag ({stop_reason})"
        print(f"[red-purple] {run_id} done | {outcome}")

    return metadata, context_window
