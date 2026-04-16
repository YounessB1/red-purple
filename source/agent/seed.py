"""Red-Purple agent — runs inside the container."""

import json
import os

from source.agent.llm import LLM
from source.agent.tools import get_tool, get_tools_prompt
from source.agent.tracer import Tracer

try:
    from langfuse import Langfuse
    _langfuse = Langfuse() if os.environ.get("LANGFUSE_PUBLIC_KEY") else None
except ImportError:
    _langfuse = None


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



def execute_tool(call: dict) -> tuple[str, bool]:
    name = call["name"]
    args = call["args"]
    func = get_tool(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"}), False
    try:
        result = func(**args)
        should_finish = bool(result.get("should_finish", False)) if isinstance(result, dict) else False
        return json.dumps(result, indent=2), should_finish
    except Exception as e:
        return json.dumps({"error": str(e)}), False


def main() -> None:
    target = os.environ.get("TARGET", "")
    if not target:
        print("ERROR: No target specified. Use -t <url> when running red-purple.")
        return
    max_iter = int(os.environ.get("MAX_ITER", "100"))
    task = os.environ.get("TASK") or DEFAULT_TASK.format(target=target)

    llm = LLM()
    tracer = Tracer(target=target)
    system_prompt = SYSTEM_PROMPT.format(tools=get_tools_prompt())

    lf_trace = _langfuse.trace(
        name="agent-run",
        input={"target": target, "task": task},
        tags=["red-purple"],
    ) if _langfuse else None

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tracer.log_message("user", task)
    print(f"Task: {task}\n")

    final_result: str | None = None
    try:
        for iteration in range(1, max_iter + 1):
            lf_metadata = {"trace_id": lf_trace.id, "generation_name": f"iteration-{iteration}"} if lf_trace else None
            content, tool_calls = llm.generate(history, metadata=lf_metadata)
            history.append({"role": "assistant", "content": content})
            tracer.log_message("assistant", content)

            if iteration == max_iter:
                print(f"[stopping] Reached max iterations ({max_iter}).")
                break

            finished = False
            for call in tool_calls:
                print(f"[tool] {call['name']}({call['args']})")
                tracer.log_tool_start(call["name"], call["args"])

                lf_span = lf_trace.span(name=f"tool:{call['name']}", input=call["args"]) if lf_trace else None
                result, should_finish = execute_tool(call)
                if lf_span:
                    lf_span.end(output={"result": result})

                tracer.log_tool_end(call["name"], result)
                print(f"[result] {result}\n")
                history.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
                if should_finish:
                    finished = True
                    final_result = result
                    break

            if finished:
                break
    finally:
        tracer.finish()
        if lf_trace:
            lf_trace.update(output={"result": final_result})
        if _langfuse:
            _langfuse.flush()


if __name__ == "__main__":
    main()
