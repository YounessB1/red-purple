"""TracingCallback — persists GEPA engine events to disk for post-hoc inspection."""

import json
from pathlib import Path

from gepa.gepa_utils import remove_dominated_programs

from source.optimize_anything import evaluator


class TracingCallback:

    def __init__(
        self,
        experiment_dir: Path,
        seed_candidate: dict | None = None,
        trainset: list[dict] | None = None,
        valset: list[dict] | None = None,
    ) -> None:
        self._experiment_dir = experiment_dir
        self._train_ids = [ex["benchmark_id"] for ex in (trainset or [])]
        self._val_ids = [ex["benchmark_id"] for ex in (valset or [])]
        self._current_child_instructions: dict | None = None

    def on_optimization_start(self, event) -> None:
        evaluator.set_gepa_iteration(0)
        evaluator.set_gepa_role("val")

    def on_iteration_start(self, event) -> None:
        evaluator.set_gepa_iteration(event["iteration"])
        evaluator.set_gepa_role("val")
        self._current_child_instructions = None
        self._write_pool(event["iteration"], event["state"])
        seed_evo_path = self._experiment_dir / "iteration_000" / "evolution.json"
        if not seed_evo_path.exists():
            self._write_seed_evolution(event["state"])

    def on_evaluation_start(self, event) -> None:
        if event.get("capture_traces"):
            evaluator.set_gepa_role("parent")
        else:
            evaluator.set_gepa_role("child")

    def on_proposal_end(self, event) -> None:
        self._current_child_instructions = event.get("new_instructions")

    def on_iteration_end(self, event) -> None:
        self._write_evolution(event["iteration"], event["state"], event["proposal_accepted"])

    # ── Seed evolution (iteration 000) ────────────────────────────────────

    def _write_seed_evolution(self, state) -> None:
        iter_dir = self._experiment_dir / "iteration_000"
        iter_dir.mkdir(parents=True, exist_ok=True)

        seed_prompt = state.program_candidates[0].get("prompt", "") if state.program_candidates else ""
        val_scores_raw = state.prog_candidate_val_subscores[0] if state.prog_candidate_val_subscores else {}
        val_scores = {
            (self._val_ids[vi] if vi < len(self._val_ids) else str(vi)): score
            for vi, score in val_scores_raw.items()
        }

        evolution = {
            "parent": {
                "candidate_idx": None,
                "prompt": "",
                "train": None,
                "val": None,
            },
            "child": {
                "accepted": True,
                "candidate_idx": 0,
                "prompt": seed_prompt,
                "train": None,
                "val": val_scores,
            },
        }
        (iter_dir / "evolution.json").write_text(
            json.dumps(evolution, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (iter_dir / "ACCEPTED").touch()

    # ── Per-iteration evolution snapshot ──────────────────────────────────

    def _write_evolution(self, iteration: int, state, proposal_accepted: bool) -> None:
        iter_dir = self._experiment_dir / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        trace = state.full_program_trace[-1]
        parent_idx = trace.get("selected_program_candidate")
        subsample_ids = trace.get("subsample_ids", [])
        parent_train_scores = trace.get("subsample_scores", [])
        child_train_scores = trace.get("new_subsample_scores", [])
        new_program_idx = trace.get("new_program_idx")

        def _train_map(scores):
            return {
                self._train_ids[sid]: score
                for sid, score in zip(subsample_ids, scores)
                if sid < len(self._train_ids)
            } if scores else None

        def _val_map(raw: dict):
            return {
                (self._val_ids[vi] if vi < len(self._val_ids) else str(vi)): score
                for vi, score in raw.items()
            }

        # parent
        if parent_idx is not None:
            parent_prompt = state.program_candidates[parent_idx].get("prompt", "")
            parent_val = _val_map(state.prog_candidate_val_subscores[parent_idx])
        else:
            parent_prompt = ""
            parent_val = {}

        # child
        if proposal_accepted and new_program_idx is not None:
            child_prompt = state.program_candidates[new_program_idx].get("prompt", "")
            child_val = _val_map(state.prog_candidate_val_subscores[new_program_idx])
        else:
            child_prompt = (self._current_child_instructions or {}).get("prompt", "")
            child_val = None

        evolution = {
            "parent": {
                "candidate_idx": parent_idx,
                "prompt": parent_prompt,
                "train": _train_map(parent_train_scores),
                "val": parent_val,
            },
            "child": {
                "accepted": proposal_accepted,
                "candidate_idx": new_program_idx,
                "prompt": child_prompt,
                "train": _train_map(child_train_scores),
                "val": child_val,
            },
        }
        (iter_dir / "evolution.json").write_text(
            json.dumps(evolution, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        marker = "ACCEPTED" if proposal_accepted else "REJECTED"
        (iter_dir / marker).touch()

    # ── Pool snapshot ──────────────────────────────────────────────────────

    def _write_pool(self, iteration: int, state) -> None:
        candidate_train: dict[int, dict[str, float]] = {}
        for entry in state.full_program_trace[:-1]:
            new_idx = entry.get("new_program_idx")
            if new_idx is None:
                continue
            subsample_ids = entry.get("subsample_ids", [])
            new_scores = entry.get("new_subsample_scores", [])
            candidate_train[new_idx] = {
                self._train_ids[sid]: score
                for sid, score in zip(subsample_ids, new_scores)
                if sid < len(self._train_ids)
            }

        pruned_mapping = remove_dominated_programs(
            state.get_pareto_front_mapping(),
            scores=state.per_program_tracked_scores,
        )
        candidate_pareto: dict[int, list[str]] = {}
        for val_idx, cand_set in pruned_mapping.items():
            bench_id = self._val_ids[val_idx] if val_idx < len(self._val_ids) else str(val_idx)
            for cand_idx in cand_set:
                candidate_pareto.setdefault(cand_idx, []).append(bench_id)

        candidates = []
        for idx, candidate in enumerate(state.program_candidates):
            val_scores_raw = state.prog_candidate_val_subscores[idx]
            val_scores = {
                (self._val_ids[vi] if vi < len(self._val_ids) else str(vi)): score
                for vi, score in val_scores_raw.items()
            }
            avg = sum(val_scores.values()) / len(val_scores) if val_scores else 0.0
            parents = [p for p in state.parent_program_for_candidate[idx] if p is not None]
            candidates.append({
                "idx": idx,
                "parent_ids": parents,
                "val_avg": round(avg, 4),
                "on_pareto_front": sorted(candidate_pareto.get(idx, [])),
                "val": val_scores,
                "train_subsample": candidate_train.get(idx),
                "files": candidate.get("files", ""),
            })

        pool_dir = self._experiment_dir / f"iteration_{iteration:03d}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        (pool_dir / "pool.json").write_text(
            json.dumps({"iteration_snapshot": iteration, "candidates": candidates},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
