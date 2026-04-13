"""Red-Purple agent — runs inside the container."""

import json
import os

from redpurple.agent.llm import LLM
from redpurple.agent.tools import get_tool, get_tools_prompt
from redpurple.agent.tracer import Tracer


SYSTEM_PROMPT = """You are a security testing agent running inside an isolated container.
You have access to the following tools:

{tools}

To call a tool, use this format in your response:
<function_calls>
<invoke name="tool_name">
<param_name>value</param_name>
</invoke>
</function_calls>

You can call one tool per response. After seeing the tool result, decide what to do next.
When you have completed the task, respond with your findings and no tool calls.
"""


def execute_tool(call: dict) -> str:
    name = call["name"]
    args = call["args"]
    func = get_tool(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = func(**args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def main() -> None:
    target = os.environ.get("TARGET", "")
    if not target:
        print("ERROR: No target specified. Use -t <url> when running red-purple.")
        return
    max_iter = int(os.environ.get("MAX_ITER", "100"))
    task = f"You are pentesting the following target: {target}\nStart with reconnaissance and identify vulnerabilities."

    llm = LLM()
    tracer = Tracer(target=target)
    system_prompt = SYSTEM_PROMPT.format(tools=get_tools_prompt())

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tracer.log_message("user", task)
    print(f"Task: {task}\n")

    try:
        for iteration in range(1, max_iter + 1):
            content, tool_calls = llm.generate(history)
            history.append({"role": "assistant", "content": content})
            tracer.log_message("assistant", content)

            if not tool_calls:
                print(f"Agent: {content}")
                break

            if iteration == max_iter:
                print(f"[stopping] Reached max iterations ({max_iter}).")
                break

            for call in tool_calls:
                print(f"[tool] {call['name']}({call['args']})")
                tracer.log_tool_start(call["name"], call["args"])
                result = execute_tool(call)
                tracer.log_tool_end(call["name"], result)
                print(f"[result] {result}\n")
                history.append({"role": "user", "content": f"<tool_result>\n{result}\n</tool_result>"})
    finally:
        tracer.finish()


if __name__ == "__main__":
    main()
