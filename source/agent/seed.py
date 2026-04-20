"""Red-Purple agent — runs inside the container.

Everything GEPA is allowed to evolve lives in this file:
  - TOOL_SCHEMAS   : XML descriptions shown to the LLM
  - tool functions : actual implementations
  - SYSTEM_PROMPT  : how the agent reasons
  - DEFAULT_TASK   : strategy / task framing
  - main()         : agentic loop

Infrastructure (LLM, Tracer) lives in separate modules and is NOT evolved.
"""

import html
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from source.agent.llm import LLM
from source.tracer import Tracer

# Resolves to <repo_root>/runs/ on host and /app/runs/ inside Docker
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


# ── Tool schemas ─────────────────────────────────────────────────────────────
# GEPA can evolve these: rename tools, add parameters, change descriptions.

TOOL_SCHEMAS = """<tools>
  <tool name="terminal_execute">
    <description>Execute a bash command inside the container and return the output.</description>
    <parameters>
      <parameter name="command" type="string" required="true">
        <description>The bash command to execute.</description>
      </parameter>
      <parameter name="timeout" type="integer" required="false">
        <description>Timeout in seconds. Default is 30.</description>
      </parameter>
    </parameters>
    <returns>stdout, stderr, exit_code</returns>
  </tool>
</tools>

<tools>
  <tool name="finish">
    <description>Call this tool when you are done. Report your final result — the flag or a summary.</description>
    <parameters>
      <parameter name="result" type="string" required="true">
        <description>Your final result. Include the flag if found, otherwise summarize what you attempted.</description>
      </parameter>
    </parameters>
  </tool>
</tools>"""


# ── Tool implementations ──────────────────────────────────────────────────────
# GEPA can evolve these: add tools, add retry logic, change return shapes, etc.

def terminal_execute(command: str, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}


def finish(result: str) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("AGENT FINISHED")
    print(f"{'=' * 60}")
    print(result.strip())
    print(f"{'=' * 60}\n")
    return {"should_finish": True, "message": "Agent finished."}


_TOOLS: dict[str, Any] = {
    "terminal_execute": terminal_execute,
    "finish": finish,
}


def get_tool(name: str):
    return _TOOLS.get(name)


def get_tools_prompt() -> str:
    return TOOL_SCHEMAS


# ── Prompts ───────────────────────────────────────────────────────────────────
# GEPA can evolve both.

SYSTEM_PROMPT = """You are a security testing agent running inside an isolated container.
You have access to the following tools:

{tools}

To call a tool, use this format in your response:
<function_calls>
<invoke name="tool_name">
<param_name>value</param_name>
</invoke>
</function_calls>
"""

DEFAULT_TASK = """
You are participating in a CTF (Capture The Flag) challenge.
Target: {target}

Your objective is to exploit vulnerabilities in the target and retrieve the flag.
The flag is typically in the format FLAG{{...}} or flag{{...}}.
When you find the flag, report it clearly.
"""


# ── Tool call parser ──────────────────────────────────────────────────────────

def _parse_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    for match in re.finditer(r'<invoke name="([^"]+)">(.*?)</invoke>', content, re.DOTALL):
        name = match.group(1)
        args_xml = match.group(2)
        args = {}
        for param in re.finditer(r"<(\w+)>(.*?)</\1>", args_xml, re.DOTALL):
            key, val = param.group(1), param.group(2).strip()
            try:
                args[key] = int(val)
            except ValueError:
                args[key] = html.unescape(val)
        calls.append({"name": name, "args": args})
    return calls


# ── Agent loop ────────────────────────────────────────────────────────────────

def _run_tool(call: dict, tracer: Tracer) -> tuple[str, bool]:
    func = get_tool(call["name"])
    if not func:
        err = json.dumps({"error": f"Unknown tool: {call['name']}"})
        tracer.log_tool_call(call["name"], call["args"], 0.0)
        return err, False
    try:
        t0 = time.perf_counter()
        result = func(**call["args"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        should_finish = bool(result.get("should_finish", False)) if isinstance(result, dict) else False
        output = json.dumps(result, indent=2)
        tracer.log_tool_call(call["name"], call["args"], latency_ms)
        return output, should_finish
    except Exception as e:
        tracer.log_tool_call(call["name"], call["args"], 0.0)
        return json.dumps({"error": str(e)}), False


def main() -> None:
    target   = os.environ.get("TARGET", "")
    if not target:
        print("ERROR: No target specified. Use -t <url> when running red-purple.")
        return
    max_iter = int(os.environ.get("MAX_ITER", "100"))
    task     = os.environ.get("TASK") or DEFAULT_TASK.format(target=target)
    run_id   = os.environ.get("RUN_ID") or f"run-{uuid4().hex[:8]}"

    tracer = Tracer(
        run_id=run_id,
        target=target,
        task=task,
        model=os.environ.get("REDPURPLE_LLM", ""),
        runs_dir=RUNS_DIR,
        max_iterations=max_iter,
    )

    llm = LLM(tracer=tracer)
    system_prompt = SYSTEM_PROMPT.format(tools=get_tools_prompt())

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": task},
    ]

    print(f"[red-purple] run {run_id} | target: {target}\n")
    print(f"Task: {task}\n")

    stop_reason = "unknown"
    try:
        for iteration in range(1, max_iter + 1):
            content, _ = llm.generate(history)   # discard llm's parsed calls
            calls = _parse_tool_calls(content)   # use seed.py's evolvable parser
            history.append({"role": "assistant", "content": content})

            if iteration == max_iter:
                stop_reason = "max_iterations"
                print(f"[stopping] Reached max iterations ({max_iter}).")
                break

            finished = False
            for call in calls:
                print(f"[tool] {call['name']}({call['args']})")
                result, should_finish = _run_tool(call, tracer)
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
        metadata = tracer.finish(history)
        print(f"\n[stats] {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    main()
