"""RedPurpleAdapter — custom GEPAAdapter with full control over evaluation and reflection."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from gepa import EvaluationBatch, GEPAAdapter

from source.optimize_anything import evaluator


@dataclass
class RedPurpleTrajectory:
    benchmark_id: str
    success: bool
    stop_reason: str
    iterations: int
    context_window: list[dict]
    error: str | None = None


class RedPurpleAdapter(GEPAAdapter):
    """Custom GEPA adapter — owns evaluation parallelism and reflective dataset construction."""

    def __init__(self, *, workers: int) -> None:
        self.workers = workers

    def evaluate(self, batch, candidate, capture_traces=False) -> EvaluationBatch:
        def _eval_one(example):
            try:
                score, side_info = evaluator.evaluate(candidate, example)
                output = {
                    k: v for k, v in side_info.items() if k not in ("context_window", "log")
                }
                trajectory = None
                if capture_traces:
                    trajectory = RedPurpleTrajectory(
                        benchmark_id=side_info["benchmark_id"],
                        success=side_info["success"],
                        stop_reason=side_info["stop_reason"],
                        iterations=side_info["iterations"],
                        context_window=side_info.get("context_window") or [],
                    )
                return score, output, trajectory
            except Exception as e:
                bench_id = example.get("benchmark_id", "unknown")
                print(f"[eval] {bench_id} — evaluation error: {e}")
                output = {"benchmark_id": bench_id, "error": str(e)}
                trajectory = None
                if capture_traces:
                    trajectory = RedPurpleTrajectory(
                        benchmark_id=bench_id,
                        success=False,
                        stop_reason="error",
                        iterations=0,
                        context_window=[],
                        error=str(e),
                    )
                return 0.0, output, trajectory

        if self.workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                results = list(ex.map(_eval_one, batch))
        else:
            results = [_eval_one(ex) for ex in batch]

        scores = [r[0] for r in results]
        outputs = [r[1] for r in results]
        trajectories = [r[2] for r in results] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, _candidate, eval_batch, components_to_update):
        records = []
        for traj in (eval_batch.trajectories or []):
            records.append({
                "Inputs": {"benchmark_id": traj.benchmark_id},
                "Generated Outputs": {
                    "stop_reason": traj.stop_reason,
                    "iterations_used": traj.iterations,
                },
                "Trace": traj.context_window,
            })
        return {name: records for name in components_to_update}
