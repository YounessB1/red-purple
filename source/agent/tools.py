import subprocess


# ── TOOLS ─────────────────────────────────────────────────────────────────────

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
  <tool name="finish">
    <description>Call this when you are done. Report the flag you found, or summarize what you attempted.</description>
    <parameters>
      <parameter name="result" type="string" required="true">
        <description>Your final result.</description>
      </parameter>
    </parameters>
  </tool>
</tools>"""


def terminal_execute(command: str, timeout: int = 30) -> dict:
    try:
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
        return {"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.returncode}
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"error": str(e), "exit_code": -1}


def finish(result: str) -> dict:
    return {"should_finish": True}


TOOLS = {
    "terminal_execute": terminal_execute,
    "finish": finish,
}

TOOLS_PROMPT = f"""
# TOOLS
You have access to the following tools:

{TOOL_SCHEMAS}

Large tool outputs may be summarised in <extracted_output> blocks. Context from earlier in the conversation may be compressed into <context_summary> blocks. Treat both as reliable — do not repeat work already described in them.

# TOOL CALL FORMAT
To call a tool, use XML tags where each tag name is the parameter name. Example:
<function_calls>
<invoke name="terminal_execute">
<command>ls -la</command>
</invoke>
</function_calls>
"""