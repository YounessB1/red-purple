"""RedPurpleAdapter — custom GEPAAdapter with parallelized evaluation."""

import random
from concurrent.futures import ThreadPoolExecutor

from gepa import EvaluationBatch, GEPAAdapter
from gepa.strategies.eval_policy import FullEvaluationPolicy

from source.optimize_anything import evaluator


class SubsetValPolicy(FullEvaluationPolicy):
    """Evaluates a random subset of k val examples per accepted candidate."""

    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)

    def get_eval_batch(self, loader, _state, _target_program_idx=None):
        all_ids = list(loader.all_ids())
        if self.k >= len(all_ids):
            return all_ids
        return self.rng.sample(all_ids, self.k)


class RedPurpleAdapter(GEPAAdapter):

    def __init__(self, *, workers: int) -> None:
        self.workers = workers

    def evaluate(self, batch, candidate, capture_traces=False) -> EvaluationBatch:
        def _eval_one(example):
            try:
                score, side_info = evaluator.evaluate(candidate, example)
                output = {k: v for k, v in side_info.items() if k not in ("context_window", "log")}
                return score, output
            except Exception as e:
                bench_id = example.get("benchmark_id", "unknown")
                print(f"[eval] {bench_id} — evaluation error: {e}")
                return 0.0, {"benchmark_id": bench_id, "error": str(e)}

        if self.workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(_eval_one, batch))
        else:
            results = [_eval_one(ex) for ex in batch]

        trajectories = [{}] * len(results) if capture_traces else None
        return EvaluationBatch(
            outputs=[r[1] for r in results],
            scores=[r[0] for r in results],
            trajectories=trajectories,
        )

    def make_reflective_dataset(self, _candidate, _eval_batch, components_to_update):
        return {name: [{}] for name in components_to_update}
