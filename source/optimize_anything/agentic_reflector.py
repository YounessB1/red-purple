"""Agentic prompt reflector — spawns an OpenCode agent to browse iteration artifacts."""

import json
import re
import subprocess
from pathlib import Path

from source.optimize_anything.logger import Logger
from source.optimize_anything.utils import extract_new_prompt
from source.optimize_anything.evaluator import _get_iteration

# ── OpenCode permissions (written temporarily to iter_dir) ─────────────

_OPENCODE_PERMISSIONS = {
    "permission": {
        "read":               "allow",
        "glob":               "allow",
        "grep":               "allow",
        "list":               "allow",
        "edit":               "deny",
        "bash":               "deny",
        "task":               "deny",
        "external_directory": "deny",
        "webfetch":           "deny",
        "websearch":          "deny",
    }
}

# ── Agent instructions (written as AGENTS.md to iter_dir) ──────────────

_AGENTIC_REFLECTOR_PROMPT = """
You are a CTF security research assistant acting as a prompt optimizer.

Your job is to analyze how a security agent performed on this iteration and propose an improved strategy prompt.

# Current strategy prompt (what you must improve)

{current_prompt}

# Iteration artifacts

All files below are in your current working directory.

## train/parent/<benchmark_id>/metadata.json
One file per benchmark the agent ran on. Contains outcome details:
- "success": true/false — whether the agent found the flag
- "stop_reason": why the agent stopped ("flag_found", "max_iterations", "error", ...)
- "iterations_used": how many LLM turns the agent used
- "target": the benchmark URL/target

## train/parent/<benchmark_id>/diagnosis.json
Present only for failed runs. Schema: {{"diagnosis": "..."}}
Plain-text analysis: vulnerability type, what the agent tried, where it went wrong, what the correct approach should have been.
This is your primary signal.

## train/parent/<benchmark_id>/context_window.json
Full LLM conversation history — list of message dicts with role and content.
Large file. Read only when the diagnosis alone is insufficient to understand a failure.

## train/parent/<benchmark_id>/judge_score.json
Present only when an LLM judge is configured. Schema: {{"model": "...", "score": float, "reason": "..."}}
Partial credit score (0–1) with the judge's reasoning about how close the agent got.

## pool.json (optional, can be ignored)
Snapshot of all candidate prompts in the optimization pool. Useful only if you want to compare the current strategy against other candidates. The current strategy is already provided above.

# Steps

1. List train/parent/ to see which benchmarks were evaluated.
2. For each benchmark, read metadata.json to check success and stop_reason.
3. For failed benchmarks, read diagnosis.json.
4. For failures where the diagnosis is unclear, read context_window.json.

# Your task

Based on your analysis, produce an improved strategy prompt.
Do NOT redefine the agent role, target URL, tools, or tool-call format — those are fixed infrastructure.
Focus on attack strategy, vulnerability playbooks, concrete commands, and step-by-step methodologies.

**Prioritize generalizing patterns across failures**: if multiple benchmarks share the same root cause or vulnerability type, extract a reusable rule or technique rather than patching each case individually. The goal is a strategy that transfers to unseen challenges.

Return your result as a JSON object with exactly two fields:
- "changes": 3-5 bullet points explaining what you changed and why, referencing specific failure patterns
- "prompt": the full new strategy instructions as a multi-paragraph string

Output only the JSON object, no preamble.
"""


# ── Agentic reflector ──────────────────────────────────────────────────

class AgenticReflector:
    """Spawns an OpenCode agent that browses iteration artifacts and proposes an improved prompt."""

    def __init__(self, model: str, logger: Logger, experiment_dir: Path) -> None:
        self._model = model
        self._logger = logger
        self._experiment_dir = experiment_dir

    # for gepa signature prompt is passes to __call__, but we ignore it since the reflector reads the current prompt directly from the iteration artifacts
    def __call__(self, prompt: str | list[dict]) -> str:
        iteration = _get_iteration()
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

        current_prompt = self._extract_current_prompt(prompt)
        task = self._build_task(current_prompt)
        print(f"\n[agentic-reflector] Starting OpenCode for iteration {iteration}…", flush=True)

        opencode_json = iter_dir / "opencode.json"
        agents_md = iter_dir / "AGENTS.md"
        try:
            opencode_json.write_text(json.dumps(_OPENCODE_PERMISSIONS, indent=2), encoding="utf-8")
            agents_md.write_text(task, encoding="utf-8")

            proc = subprocess.run(
                [
                    "opencode", "run",
                    "--model", self._model,
                    "--dir", str(iter_dir),
                    "Analyze the iteration artifacts and return an improved strategy prompt as JSON.",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
        finally:
            opencode_json.unlink(missing_ok=True)
            agents_md.unlink(missing_ok=True)

        raw = proc.stdout.strip()
        if not raw:
            print(f"[agentic-reflector] No output — stderr:\n{proc.stderr}", flush=True)

        self._logger.log_reflector(0, 0, task, raw)
        return extract_new_prompt(raw, self._logger)

    def _extract_current_prompt(self, prompt: str | list[dict]) -> str:
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        m = re.search(r"## Current strategy instructions.*?```\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else "(could not extract current prompt)"

    def _build_task(self, current_prompt: str) -> str:
        return _AGENTIC_REFLECTOR_PROMPT.format(current_prompt=current_prompt)
