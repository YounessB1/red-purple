# Plan: Replace Penetration Testing Agent with OpenCode

## Context

The current penetration testing agent is a hand-crafted Python loop in `source/agent/runner.py`. The goal is to replace it with an OpenCode instance. The client-server architecture (`server.py` → `runner.run()`) is preserved unchanged. GEPA wiring comes later; this PR covers the agent replacement and the candidate schema redesign.

---

## Candidate Schema

The candidate is a **file snapshot dict** — a flat map of relative path → file content. This is the single representation used for everything: GEPA hashing/caching, HTTP transfer to the container, and runner materialization. No separate volume mount or shared module needed.

```python
candidate = {
    "files": {
        ".opencode/agents/ctf-agent.md": "---\nmodel: ...\n---\n...",
        "AGENTS.md":                      "",
        "skills/web_basics.md":           "...",
        # reflector can add any file here — no schema change ever needed
    }
}
```

**Why this works end-to-end:**
- `_candidate_hash` hashes the whole dict → same files = same hash → cache hit ✓
- `_run_via_server` sends `seed_json = json.dumps(candidate)` → no new HTTP params ✓
- Runner writes every `{path: content}` pair into the per-run workdir ✓
- Reflector edits files freely on disk; `AgenticReflector.__call__` reads updated dir and returns snapshot ✓

---

## Iteration directory structure

Each iteration stores the agent state **before** and **after** the reflector for full visibility:

```
experiments/experiment5/
├── iteration_001/
│   ├── agent/                      ← files that RAN in this iteration (snapshot = candidate used)
│   │   ├── .opencode/agents/
│   │   │   └── ctf-agent.md
│   │   ├── AGENTS.md
│   │   └── skills/
│   ├── agent_improved/             ← files AFTER the reflector ran (= next iteration's agent)
│   │   └── ...
│   ├── train/parent/XBEN-xxx/...
│   ├── train/child/XBEN-xxx/...
│   └── reflector.json
├── iteration_002/
│   ├── agent/                      ← copy of iteration_001/agent_improved/
│   └── ...
```

`agent/` is written at the **start** of each iteration (by `agentic_reflector.__call__` or by `_build_seed_candidate` for iteration 0).
`agent_improved/` is written by the **reflector** at the **end** of the iteration.

---

## Docker — no volume mount needed

The file snapshot travels through the existing HTTP interface. The container receives the files as JSON and materializes them. `docker-compose.yml` stays as-is.

OpenCode is installed in the container via multi-stage build (already done in `Dockerfile`):
```dockerfile
FROM node:22-slim AS node
# ... copy node binary, recreate npm/npx symlinks, npm install -g opencode-ai@1.14.50
```

---

## Files to change

| File | Action |
|------|--------|
| `source/seed/` | **Create** — replaces `source/seed.py`; initial agent files live here |
| `source/seed/.opencode/agents/ctf-agent.md` | **Create** — initial agent definition |
| `source/seed/AGENTS.md` | **Create** — empty initially |
| `source/seed/skills/` | **Create** — empty initially |
| `source/agent/runner.py` | **Replace** — OpenCode-based implementation |
| `source/agent/server.py` | **Modify** — pass full candidate dict (not just `prompt` string) |
| `source/optimize_anything/core_loop.py` | **Modify** — `_build_seed_candidate()` reads from `source/seed/` |
| `source/optimize_anything/agentic_reflector.py` | **Modify** — returns file snapshot dict; writes `agent/` + `agent_improved/` dirs |
| `source/optimize_anything/utils.py` | **Modify** — `extract_new_prompt` → `extract_new_candidate` returning dict |
| `.opencode/agents/reflector.md` | **Modify** — enable `edit`/`write` permissions; update output format |
| `source/seed.py` | **Delete** |
| `source/agent/tools.py`, `compactor.py`, `extractor.py` | **Keep, unused** — delete in cleanup PR |
| `source/agent/base_prompt.py` | **Keep** — `BASE_PROMPT_SUMMARY` still referenced by reflector |

---

## Step 1 — Create `source/seed/`

The seed folder IS the initial candidate materialized as real files. `_build_seed_candidate()` reads it into the file snapshot dict.

**`source/seed/.opencode/agents/ctf-agent.md`:**
```markdown
---
description: "CTF penetration testing agent"
model: "openrouter/deepseek/deepseek-v3.2"
temperature: 0.6
maxSteps: 50
tools:
  bash: true
  read: true
  write: true
permission:
  bash: "allow"
  edit: "allow"
---

# ROLE
You are a CTF security agent running inside an isolated container.
Capture the flag from the vulnerable environment at the target URL.

# HOW TO USE BASH
Use the bash tool to run shell commands: curl, nmap, sqlmap, flask-unsign, etc.
All standard CTF tools are installed.

# REPORTING
When you find the flag (FLAG{...} or flag{...}), state it clearly in your final message.

# OBJECTIVE
Exploit vulnerabilities in the target and retrieve the flag.
The flag is typically in the format FLAG{...} or flag{...}.

# SKILLS
- Use curl to interact with the target: curl -v {target}
```

**`source/seed/AGENTS.md`** — empty for now.

**`source/seed/skills/`** — empty for now.

---

## Step 2 — Update `_build_seed_candidate()` in `core_loop.py`

```python
_SEED_DIR = Path(__file__).resolve().parents[2] / "source" / "seed"

def _build_seed_candidate() -> dict:
    files = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")
    return {"files": files}
```

Remove `from source.seed import PROMPT`.

---

## Step 3 — Update `source/agent/server.py`

```python
# Before (line 50-51):
prompt = json.loads(seed_json).get("prompt") if seed_json else None
run(..., prompt=prompt, ...)

# After:
candidate = json.loads(seed_json) if seed_json else {}
run(..., candidate=candidate, ...)
```

---

## Step 4 — Replace `source/agent/runner.py`

### Signature

```python
def run(
    target: str,
    model: str,
    run_id: str | None = None,
    candidate: dict | None = None,
    max_iter: int = 50,
    runs_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict, list]:
```

### Workdir materialization

Create a temp dir, patch `model` and `maxSteps` in the agent `.md` file, prepend the target URL to `AGENTS.md`, write all files:

```python
workdir = Path(tempfile.mkdtemp(prefix=f"agent_{run_id}_"))
files = (candidate or {}).get("files", {})

for rel_path, content in files.items():
    dest = workdir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    # patch model and maxSteps in agent .md files
    if rel_path.endswith(".md") and rel_path.startswith(".opencode/agents/"):
        content = re.sub(r'(?m)^model:.*$', f'model: "{model}"', content)
        content = re.sub(r'(?m)^maxSteps:.*$', f'maxSteps: {max_iter}', content)
    dest.write_text(content, encoding="utf-8")

# Prepend target to AGENTS.md (always, regardless of candidate content)
agents_md = workdir / "AGENTS.md"
existing = agents_md.read_text() if agents_md.exists() else ""
agents_md.write_text(f"# Target\nThe target URL is: {target}\n\n{existing}".strip())
```

### Spawning OpenCode

```python
proc = subprocess.Popen(
    ["opencode", "run", "--agent", "ctf-agent", "--dir", str(workdir),
     f"The target is at {target}. Find and report the flag."],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
```

### Cancellation

Daemon watcher thread polls `cancel_event` every second, calls `proc.kill()` if set. Main thread blocks on `proc.communicate(timeout=max_iter * 120)`.

### SQLite trace

Query `~/.local/share/opencode/opencode.db` (same pattern as `agentic_reflector._trace_session`):
- Match session by `directory = str(workdir)`
- Extract `cost`, `tokens_input`, `tokens_output`
- Reconstruct `context_window` from `part` table

### context_window reconstruction

From `part` table: `text` → assistant, `tool` → append to assistant, `tool-result` → user in `<tool_result>`, `reasoning` → `<thinking>` prefix, `step-start`/`step-finish` → skip.

### Flag detection

Scan all `text` and `tool-result` parts + `proc.stdout` for `FLAG\{[^}]+\}` (case-insensitive).

### Metadata shape

```python
{
    "run_id": str, "target": str, "model": str,
    "success": bool, "flag": str | None, "stop_reason": str,
    "started_at": str, "finished_at": str, "duration_seconds": float,
    "iterations_used": int, "max_iterations": int,
    "llm_calls": int, "extractor_calls": 0, "compactor_calls": 0, "tool_calls": int,
    "total_input_tokens": int, "total_output_tokens": int, "total_tokens": int,
    "total_cost_usd": float, "context_messages": int,
}
```

---

## Step 5 — Update `agentic_reflector.py`

The reflector now:
1. Writes `iter_dir/agent/` from the current candidate (for visibility + reflector to read from)
2. Runs OpenCode with access to `iter_dir/` — reflector edits files in `iter_dir/agent_improved/`
3. Reads `agent_improved/` back into a snapshot dict and returns it to GEPA

```python
def __call__(self, prompt: str | list[dict]) -> dict:
    iteration = _get_iteration()
    iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

    # Write current candidate to agent/ for visibility and reflector reference
    current = self._current_candidate  # set from outside or passed in
    _materialize_dir(iter_dir / "agent", current.get("files", {}))

    # Run reflector — it writes to agent_improved/
    (iter_dir / "agent_improved").mkdir(exist_ok=True)
    message = (
        "Analyze the iteration artifacts and return an improved agent.\n"
        "Current agent files are in agent/. Write improved files to agent_improved/.\n"
        "You may change any file, add new files, or restructure entirely."
    )
    proc = subprocess.run(
        ["opencode", "run", "--agent", "reflector", "--dir", str(iter_dir), message],
        capture_output=True, text=True, timeout=600,
    )

    # Read updated files back into snapshot dict
    agent_improved_dir = iter_dir / "agent_improved"
    new_files = _snapshot_dir(agent_improved_dir)
    new_candidate = {"files": new_files}

    # Log
    input_tokens, output_tokens, cost, steps = self._trace_session(iter_dir)
    self._logger.log_reflector(input_tokens, output_tokens, message, proc.stdout, cost=cost, steps=steps)

    return new_candidate
```

```python
def _snapshot_dir(d: Path) -> dict:
    return {
        str(f.relative_to(d)): f.read_text(encoding="utf-8")
        for f in sorted(d.rglob("*")) if f.is_file()
    }

def _materialize_dir(d: Path, files: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        dest = d / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
```

**Note on return type:** `__call__` now returns `dict` instead of `str`. GEPA wiring needed to consume this properly — handled in the next phase.

---

## Step 6 — Update `.opencode/agents/reflector.md`

Enable write access and update the workflow to edit files rather than return JSON:

```yaml
permission:
  edit: "allow"
  bash: "deny"
  webfetch: "deny"
  external_directory: "deny"
```

Workflow update: reflector reads `agent/` files + diagnosis artifacts, writes improved files to `agent_improved/`.

---

## Verification

1. Start agent server: `uvicorn source.agent.server:app --port 8000`
2. Start a benchmark and hit the endpoint:
   ```bash
   curl -s -X POST "http://localhost:8000/run?target=http://localhost:<port>&max_iter=20&model=openrouter/deepseek/deepseek-v3.2"
   ```
3. Confirm: `metadata.success` bool, `metadata.flag` present or null, `context_window` non-empty
4. Run `python main.py` with small `max_calls=2` — confirm `iteration_001/agent/` and `iteration_001/agent_improved/` both exist
5. Verify `agent_improved/` files differ from `agent/` (reflector made changes)
6. Verify `iteration_002/agent/` matches `iteration_001/agent_improved/`
