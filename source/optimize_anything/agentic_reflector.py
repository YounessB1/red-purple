"""Agentic prompt reflector — spawns an OpenCode agent that edits agent files in workspace/."""

import shutil
import subprocess
from pathlib import Path

from source.optimize_anything import candidate_store
from source.optimize_anything.logger import Logger
from source.optimize_anything.evaluator import _get_iteration, get_current_candidate, set_reflection_was_merge, set_reflection_merge_parent_b_hash
from source.optimize_anything.opencode_trace import trace_opencode_session
from source.optimize_anything.utils import candidate_hash, dict_to_folder, folder_to_dict, get_candidates_pool, log

_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE = _ROOT / "workspace"
_WORKSPACE_AGENT = _WORKSPACE / "agent"


def _copy_artifacts(iter_dir: Path) -> None:
    artifacts_dst = _WORKSPACE / "artifacts"
    shutil.rmtree(artifacts_dst, ignore_errors=True)
    artifacts_src = iter_dir / "train" / "parent"
    if artifacts_src.exists():
        shutil.copytree(artifacts_src, artifacts_dst)
    else:
        artifacts_dst.mkdir(parents=True)


def _compute_overlap(val_a: dict, val_b: dict) -> float:
    """Jaccard similarity on win-sets (benchmarks where score > 0)."""
    wins_a = {b for b, s in val_a.items() if s > 0}
    wins_b = {b for b, s in val_b.items() if s > 0}
    if not wins_a and not wins_b:
        return 1.0
    return len(wins_a & wins_b) / len(wins_a | wins_b)


# ── Agentic reflector ──────────────────────────────────────────────────

class AgenticReflector:
    """Spawns an OpenCode agent that reads artifacts and edits agent files in workspace/ in-place."""

    def __init__(
        self,
        model: str,
        logger: Logger,
        experiment_dir: Path,
        merge_threshold: float = 0.3,
        reflector_agent: str = "reflector",
        merger_agent: str = "merger",
    ) -> None:
        self._model = model
        self._logger = logger
        self._experiment_dir = experiment_dir
        self._merge_threshold = merge_threshold
        self._reflector_agent = reflector_agent
        self._merger_agent = merger_agent

    def _find_merge_candidates(self, pool: list[dict], current_hash: str) -> tuple[str, str] | None:
        current_entry = next((c for c in pool if c.get("files") == current_hash), None)
        if current_entry is None:
            return None
        val_a = current_entry.get("val", {})

        best_hash, best_overlap = None, 1.0
        for c in pool:
            if c.get("files") == current_hash:
                continue
            if not c.get("on_pareto_front"):
                continue
            overlap = _compute_overlap(val_a, c.get("val", {}))
            if overlap < best_overlap:
                best_overlap = overlap
                best_hash = c["files"]

        if best_hash and best_overlap < self._merge_threshold:
            return current_hash, best_hash
        return None

    def _run_tweak(self, iter_dir: Path) -> None:
        _copy_artifacts(iter_dir)

        message = "Analyze artifacts and improve the agent strategy."
        log(f"\n[agentic-reflector] Starting OpenCode for iteration {_get_iteration()}…")

        proc = subprocess.run(
            ["opencode", "run", "--agent", self._reflector_agent, "--model", self._model, "--dir", str(_ROOT), message],
            capture_output=True, text=True, timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            log(f"[agentic-reflector] No stdout — stderr:\n{proc.stderr}")

        input_tokens, output_tokens, cost, steps = trace_opencode_session(_ROOT)
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)

    def _run_merge(self, hash_a: str, hash_b: str, iter_dir: Path) -> None:
        # workspace/agent/ already restored to hash_a (current parent) by __call__
        candidate_store.restore_workspace(hash_b, _WORKSPACE / "agent_to_merge")

        message = (
            "Analyze the two candidate agents and the failure artifacts, "
            "then write a synthesized agent to workspace/agent/ that combines "
            "the best ideas from both."
        )
        log(f"\n[agentic-reflector] MERGE — {hash_a[:10]}… + {hash_b[:10]}…")

        proc = subprocess.run(
            ["opencode", "run", "--agent", self._merger_agent, "--model", self._model, "--dir", str(_ROOT), message],
            capture_output=True, text=True, timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            log(f"[agentic-reflector] merger no stdout — stderr:\n{proc.stderr}")

        input_tokens, output_tokens, cost, steps = trace_opencode_session(_ROOT)
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)

        shutil.rmtree(_WORKSPACE / "agent_to_merge", ignore_errors=True)
        set_reflection_was_merge(True)

    def __call__(self, _prompt: str | list[dict]) -> str:
        iteration = _get_iteration()
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

        # Clear entire workspace before repopulating for this iteration
        if _WORKSPACE.exists():
            shutil.rmtree(_WORKSPACE)
        _WORKSPACE.mkdir(parents=True)

        # Restore workspace/agent/ to current parent and snapshot for visibility
        current_hash = get_current_candidate().get("files", "")
        if candidate_store.restore_workspace(current_hash, _WORKSPACE_AGENT):
            dict_to_folder(iter_dir / "parent", folder_to_dict(_WORKSPACE_AGENT))
        else:
            log(f"[agentic-reflector] Warning: hash {current_hash[:12]}… not in store, using current workspace")

        # Route: merge if a Pareto-front complement exists, else tweak
        pool = get_candidates_pool(self._experiment_dir, iteration)
        pair = self._find_merge_candidates(pool, current_hash)

        if pair:
            hash_a, hash_b = pair
            set_reflection_merge_parent_b_hash(hash_b)
            log(f"[agentic-reflector] iteration {iteration} → merge")
            self._run_merge(hash_a, hash_b, iter_dir)
        else:
            log(f"[agentic-reflector] iteration {iteration} → tweak")
            self._run_tweak(iter_dir)

        # Common exit: snapshot, hash, store, persist, return
        new_files = folder_to_dict(_WORKSPACE_AGENT)
        new_hash = candidate_hash(new_files)
        candidate_store.store(new_hash, new_files)
        dict_to_folder(iter_dir / "child", new_files)

        changes_path = _WORKSPACE / "reflector_changes.md"
        changes = changes_path.read_text(encoding="utf-8").strip() if changes_path.exists() else ""
        if changes:
            self._logger.log_reflector_changes(changes)

        return f"```\n{new_hash}\n```"
