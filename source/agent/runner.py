"""Agent runner — executes a seed module and traces the run.

Accepts any seed module/namespace that exposes: TOOL_SCHEMAS, TOOLS, PROMPT.
Defaults to source.agent.seed when no seed is provided.
"""

import threading
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from source import seed as default_seed
from source.agent import tools as agent_tools
from source.llm import LLM
from source.agent.utils import parse_tool_calls, run_tool
from source.agent import compactor, extractor
from source.tracer import Tracer

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


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
    prompt = (prompt or default_seed.PROMPT).replace("{target}", target) + agent_tools.TOOLS_PROMPT
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
        for iteration in tqdm(range(1, max_iter + 1), desc=run_id, unit="iter", leave=False):
            if cancel_event and cancel_event.is_set():
                stop_reason = "cancelled"
                break
            if compactor.should_compact(history, model):
                history = compactor.compact(history, model, tracer=tracer)
            try:
                content = llm.generate(history)
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
                result, should_finish, latency_ms = run_tool(call, agent_tools.TOOLS)
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
        tracer.set_stop_reason(stop_reason)
        metadata, context_window = tracer.finish(history)
        flag = metadata.get("flag")
        outcome = f"FLAG {flag}" if flag else f"no flag ({stop_reason})"
        print(f"[red-purple] {run_id} done | {outcome}")

    return metadata, context_window
