"""Agentic prompt reflector — spawns an OpenCode agent that proposes structured patches."""

import json
import shutil
import subprocess
from pathlib import Path

from source.optimize_anything import candidate_store
from source.optimize_anything.logger import Logger
from source.optimize_anything.evaluator import (
    _get_iteration,
    get_current_candidate,
    set_reflection_was_merge,
    set_reflection_merge_parent_b_hash,
)
from source.optimize_anything.lr_scheduler import LRScheduler
from source.optimize_anything.opencode_trace import trace_opencode_session, OPENCODE_BIN
from source.optimize_anything.patch_applier import apply_patches
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


def _load_json(path: Path, default):
    if path.exists():
        try:
            # strict=False: the reflector/merger LLM occasionally mixes literal
            # and escaped newlines within a multi-line skill-content string —
            # still valid intent, just not RFC-strict JSON. Without this, a
            # single stray control character silently discards an entire
            # (often good) patch set as "no valid patches proposed".
            return json.loads(path.read_text(encoding="utf-8"), strict=False)
        except Exception:
            pass
    return default


# ── Agentic reflector ──────────────────────────────────────────────────

class AgenticReflector:
    """Spawns an OpenCode agent that proposes JSON patches; Python applies top-N per edit budget."""

    def __init__(
        self,
        model: str,
        logger: Logger,
        experiment_dir: Path,
        merge_threshold: float = 0.3,
        reflector_agent: str = "reflector",
        merger_agent: str = "merger",
        merger_model: str = "",
        edit_budget: int = 4,
        min_edit_budget: int = 2,
        lr_scheduler: str = "cosine",
        total_iterations: int = 20,
        evolution: str = "skill",
    ) -> None:
        self._model = model
        self._merger_model = merger_model or model
        self._logger = logger
        self._experiment_dir = experiment_dir
        self._merge_threshold = merge_threshold
        self._reflector_agent = reflector_agent
        self._merger_agent = merger_agent
        self._scheduler = LRScheduler(edit_budget, min_edit_budget, lr_scheduler)
        self._total_iterations = total_iterations
        self._evolution = evolution
        self._last_applied_patches: list[dict] = []
        self._last_patch_report: dict = {}
        self._merged_pairs: set[frozenset] = set()

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
            pair_key = frozenset([current_hash, best_hash])
            if pair_key in self._merged_pairs:
                return None  # already merged this pair → fall through to tweak
            self._merged_pairs.add(pair_key)
            return current_hash, best_hash
        return None

    def _run_tweak(self, iter_dir: Path) -> None:
        _copy_artifacts(iter_dir)

        message = (
            f"Analyze artifacts and propose improvements to the agent strategy. "
            f"Workspace root: {_ROOT}/ — "
            f"write your proposed edits to {_ROOT}/workspace/proposed_patches.json "
            f"and a human-readable summary to {_ROOT}/workspace/reflector_changes.md"
        )
        log(f"\n[agentic-reflector] Starting OpenCode for iteration {_get_iteration()}…")

        proc = subprocess.run(
            [OPENCODE_BIN, "run", "--agent", self._reflector_agent, "--model", self._model, "--dir", str(_ROOT), message],
            capture_output=True, text=True, timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            log(f"[agentic-reflector] No stdout — stderr:\n{proc.stderr}")

        input_tokens, output_tokens, cost, steps = trace_opencode_session(_ROOT)
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)

    def _apply_tweak_patches(self, current_files: dict, iter_dir: Path) -> dict:
        """Read proposed_patches.json, apply budget-clipped subset, return updated files dict."""
        patches_path = _WORKSPACE / "proposed_patches.json"
        patches = _load_json(patches_path, [])

        if not isinstance(patches, list) or not patches:
            log("[agentic-reflector] No valid patches proposed — agent files unchanged")
            self._last_applied_patches = []
            self._last_patch_report = {"budget": None, "proposed": 0, "selected": 0, "report": [], "dropped_by_budget": []}
            return current_files

        if self._evolution == "prompt":
            before = len(patches)
            patches = [p for p in patches if not p.get("file", "").startswith(".opencode/skills/")]
            if len(patches) < before:
                log(f"[agentic-reflector] Filtered {before - len(patches)} skill patches (evolution=prompt)")

        iteration = _get_iteration()
        budget = self._scheduler.get(iteration, self._total_iterations)
        log(f"[agentic-reflector] {len(patches)} patches proposed, edit budget={budget}")

        selected = patches[:budget]
        dropped = patches[budget:]

        new_files, report = apply_patches(current_files, selected)
        self._last_applied_patches = selected

        applied = sum(1 for r in report if r["status"].startswith("applied"))
        skipped = sum(1 for r in report if r["status"].startswith("skipped"))
        log(f"[agentic-reflector] applied={applied} skipped={skipped} (of {len(selected)} selected, {len(patches)} proposed)")

        dropped_by_budget = [
            {"index": budget + i, "op": p.get("op", ""), "file": p.get("file", "")}
            for i, p in enumerate(dropped)
        ]
        self._last_patch_report = {
            "budget": budget, "proposed": len(patches), "selected": len(selected),
            "report": report, "dropped_by_budget": dropped_by_budget,
        }
        (iter_dir / "patch_report.json").write_text(
            json.dumps(self._last_patch_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return new_files

    def _build_changes_summary(self) -> str:
        """Deterministic human-readable changelog built from the real patch_report.json
        outcome, instead of the reflector's own narrative — which is written *before*
        the edit-budget clip runs and can describe changes that never actually landed
        (confirmed: it once narrated creating a skill file that never existed in the
        child). Format matches Logger.log_reflector_changes()'s parsing: "- " bullets
        become `changes`, "file: description" lines become `changes_summary`.
        """
        pr = self._last_patch_report
        if not pr or not pr.get("proposed"):
            return "- No valid patches proposed — agent files unchanged"

        bullets: list[str] = []
        summary: list[str] = []
        for entry in pr["report"]:
            file_key, op, status = entry["file"], entry["op"], entry["status"]
            if status.startswith("applied"):
                bullets.append(f"- Applied {op} to {file_key}")
                summary.append(f"{file_key}: applied ({op})")
            else:
                bullets.append(f"- NOT applied to {file_key} ({op}, {status})")
                summary.append(f"{file_key}: not applied — {status} ({op})")
        for entry in pr.get("dropped_by_budget", []):
            file_key, op = entry["file"], entry["op"]
            bullets.append(f"- NOT applied to {file_key} ({op}, exceeded edit budget)")
            summary.append(f"{file_key}: not applied — exceeded edit budget ({op})")

        return "\n".join(bullets) + "\n\n" + "\n".join(summary)

    def _run_merge(self, hash_a: str, hash_b: str, iter_dir: Path) -> None:
        # workspace/agent/ already restored to hash_a (current parent) by __call__
        candidate_store.restore_workspace(hash_b, _WORKSPACE / "agent_to_merge")

        message = (
            f"Analyze both candidate agents and propose patches that synthesize the best of both. "
            f"Workspace root: {_ROOT}/ — "
            f"write your proposed edits to {_ROOT}/workspace/proposed_patches.json "
            f"and a human-readable summary to {_ROOT}/workspace/reflector_changes.md"
        )
        log(f"\n[agentic-reflector] MERGE — {hash_a[:10]}… + {hash_b[:10]}…")

        proc = subprocess.run(
            [OPENCODE_BIN, "run", "--agent", self._merger_agent, "--model", self._merger_model, "--dir", str(_ROOT), message],
            capture_output=True, text=True, timeout=600,
        )

        raw = proc.stdout.strip()
        if not raw:
            log(f"[agentic-reflector] merger no stdout — stderr:\n{proc.stderr}")

        input_tokens, output_tokens, cost, steps = trace_opencode_session(_ROOT)
        self._logger.log_reflector(input_tokens, output_tokens, message, raw, cost=cost, steps=steps)

        shutil.rmtree(_WORKSPACE / "agent_to_merge", ignore_errors=True)
        set_reflection_was_merge(True)

    def _on_child_rejected(self) -> None:
        """Called by TracingCallback when GEPA rejects the child. Appends applied patches to blocklist."""
        if not self._last_applied_patches:
            return
        blocklist_path = _WORKSPACE / "patch_blocklist.json"
        existing: list[dict] = _load_json(blocklist_path, [])
        existing.extend(self._last_applied_patches)
        blocklist_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[agentic-reflector] blocklist +{len(self._last_applied_patches)} (child rejected, total={len(existing)})")

    def __call__(self, _prompt: str | list[dict]) -> str:
        iteration = _get_iteration()
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"

        # Persist blocklist across the workspace clear below
        blocklist_path = _WORKSPACE / "patch_blocklist.json"
        blocklist: list[dict] = _load_json(blocklist_path, [])

        # Clear entire workspace before repopulating for this iteration
        if _WORKSPACE.exists():
            shutil.rmtree(_WORKSPACE)
        _WORKSPACE.mkdir(parents=True)

        # Restore blocklist so the reflector agent can read it
        if blocklist:
            blocklist_path.write_text(json.dumps(blocklist, indent=2, ensure_ascii=False), encoding="utf-8")

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
            current_files = folder_to_dict(_WORKSPACE_AGENT)
            self._run_merge(hash_a, hash_b, iter_dir)
            new_files = self._apply_tweak_patches(current_files, iter_dir)
        else:
            log(f"[agentic-reflector] iteration {iteration} → tweak")
            current_files = folder_to_dict(_WORKSPACE_AGENT)
            self._run_tweak(iter_dir)
            new_files = self._apply_tweak_patches(current_files, iter_dir)

        # Common exit: snapshot, hash, store, return
        new_hash = candidate_hash(new_files)
        candidate_store.store(new_hash, new_files)
        dict_to_folder(iter_dir / "child", new_files)

        self._logger.log_reflector_changes(self._build_changes_summary())

        return f"```\n{new_hash}\n```"
