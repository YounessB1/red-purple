"""Test: verify the ctf-agent actually calls the skill tool when an obvious skill exists.

Two things are being tested:
  1. CONFIG — is skill: "allow" + skills.paths enough to expose the skill tool to the model?
  2. BEHAVIOR — given a blatant trigger in the skill frontmatter, does the model call it?

Run with:
    python tests/runner/test_skill_loading.py

Requires no benchmark server. Uses a throwaway temp workspace.
"""

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SEED_AGENT = _ROOT / "source" / "seed" / ".opencode" / "agents" / "ctf-agent.md"

# ── Test skill content ────────────────────────────────────────────────────────
# Trigger is impossible to miss: "Always. Before any bash command."
_SKILL_CONTENT = """\
---
name: test
description: Verification skill — call this before doing anything else to confirm skill loading works.
---

# TEST SKILL — Skill loading verification

If you are reading this, the skill tool works correctly.

Your task: run exactly this command and nothing else:
    echo "SKILL_LOADED_SUCCESSFULLY"

Do not use curl. Do not explore. Just run the echo command.
"""

_AGENT_MD = """\
---
description: "Skill loading test agent"
model: "openrouter/google/gemma-4-31b-it"
temperature: 0.0
steps: 5
permission:
  bash: "allow"
  skill: "allow"
  read: "allow"
  glob: "allow"
  external_directory: "allow"
  doom_loop: "allow"
  webfetch: "deny"
  websearch: "deny"
  question: "deny"
  task: "deny"
  lsp: "deny"
---

# INSTRUCTIONS

You have ONE task: call the `skill` tool with name "test" before doing anything else.
The skill will tell you exactly what command to run. Follow its instructions precisely.
Do not call bash before calling skill.
"""

def _build_workspace(tmp: Path) -> None:
    # Agents dir
    agents_dir = tmp / ".opencode" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "skill-test-agent.md").write_text(_AGENT_MD)

    # Skills go in .opencode/skills/<name>/SKILL.md — the standard auto-discovery path.
    # OpenCode walks up from --dir (CWD) looking for .opencode/skills/.
    # name + description are required frontmatter fields; trigger is not recognized.
    skills_dir = tmp / ".opencode" / "skills" / "test"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(_SKILL_CONTENT)

    # opencode.json is still needed for schema reference but skills.paths is no longer required
    opencode_json = json.dumps({"$schema": "https://opencode.ai/config.json"}, indent=2)
    (tmp / "opencode.json").write_text(opencode_json)


def _get_last_session_steps(workdir: Path) -> list[dict]:
    """Pull tool call steps from the most recent opencode session in workdir."""
    db_path = Path.home() / ".local/share/opencode/opencode.db"
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except Exception as e:
        print(f"  [warn] Cannot open opencode.db: {e}")
        return []
    try:
        row = db.execute(
            "SELECT id FROM session WHERE directory=? ORDER BY time_created DESC LIMIT 1",
            (str(workdir),),
        ).fetchone()
        if not row:
            return []
        session_id = row[0]
        parts = db.execute(
            "SELECT data FROM part WHERE session_id=? ORDER BY time_created",
            (session_id,),
        ).fetchall()
        steps = []
        for (raw,) in parts:
            try:
                part = json.loads(raw)
            except Exception:
                continue
            if part.get("type") == "tool":
                steps.append({
                    "tool": part.get("tool", "?"),
                    "input": (part.get("state") or {}).get("input", {}),
                })
        return steps
    finally:
        db.close()


def run_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="opencode_skill_test_"))
    print(f"Workspace: {tmp}")
    try:
        _build_workspace(tmp)

        prompt = (
            "Call the skill tool with name 'test' right now. "
            "Then follow its instructions exactly. "
            "Do not call bash first."
        )
        print(f"Running opencode with agent 'skill-test-agent'…")
        proc = subprocess.run(
            [
                "opencode", "run",
                "--agent", "skill-test-agent",
                "--dir", str(tmp),
                prompt,
            ],
            capture_output=True, text=True, timeout=120,
        )

        print(f"Exit code: {proc.returncode}")
        if proc.stdout.strip():
            print(f"Stdout:\n{proc.stdout.strip()[:800]}")
        if proc.stderr.strip():
            print(f"Stderr:\n{proc.stderr.strip()[:400]}")

        steps = _get_last_session_steps(tmp)
        tool_names = [s["tool"] for s in steps]
        print(f"\nTool calls observed: {tool_names}")

        skill_calls = [s for s in steps if s["tool"] == "skill"]
        bash_calls  = [s for s in steps if s["tool"] == "bash"]

        print("\n── Results ─────────────────────────────────────────")
        print(f"  skill calls : {len(skill_calls)}")
        print(f"  bash calls  : {len(bash_calls)}")
        if skill_calls:
            print(f"  skill input : {skill_calls[0]['input']}")
        if bash_calls:
            cmds = [s["input"].get("command","?")[:80] for s in bash_calls]
            print(f"  bash cmds   : {cmds}")

        # ── Assertions ────────────────────────────────────────
        failures = []

        if not skill_calls:
            failures.append("FAIL: skill tool was never called — tool may not be exposed to the model")
        else:
            print("\n  PASS: skill tool was called")

        if skill_calls and bash_calls:
            first_skill = tool_names.index("skill")
            first_bash  = tool_names.index("bash")
            if first_bash < first_skill:
                failures.append(f"FAIL: bash called at step {first_bash} before skill at step {first_skill}")
            else:
                print(f"  PASS: skill called before bash ({first_skill} < {first_bash})")
        elif skill_calls and not bash_calls:
            print("  PASS: skill called, no bash calls (model followed instructions perfectly)")

        # Check bash output contains the expected string
        echo_found = any("SKILL_LOADED_SUCCESSFULLY" in (s["input"].get("command","")) for s in bash_calls)
        if skill_calls and not echo_found:
            failures.append("FAIL: skill was called but agent did not follow skill instructions (no echo command)")
        elif echo_found:
            print("  PASS: agent followed skill instructions (ran echo command)")

        print()
        if failures:
            for f in failures:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("  All checks passed.")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_test()
