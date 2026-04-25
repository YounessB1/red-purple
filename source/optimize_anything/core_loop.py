"""Core GEPA optimization loop — all logic lives here."""

import json
import os
import random
import re
import shutil
from pathlib import Path

from gepa import optimize
from gepa.strategies.eval_policy import FullEvaluationPolicy

from source.optimize_anything import cache, evaluator
from source.optimize_anything.adapter import RedPurpleAdapter
from source.optimize_anything.callbacks import TracingCallback
from source.optimize_anything.dataset import load_dataset
from source.agent.seed import PROMPT


class SubsetValPolicy(FullEvaluationPolicy):
    """Evaluates a random subset of k val examples per accepted candidate."""

    def __init__(self, k: int, seed: int = 0):
        self.k = k
        self.rng = random.Random(seed)

    def get_eval_batch(self, loader, state, target_program_idx=None):
        all_ids = list(loader.all_ids())
        if self.k >= len(all_ids):
            return all_ids
        return self.rng.sample(all_ids, self.k)


def _next_experiment_dir(base: Path) -> Path:
    """Find the next experiment number: experiment1, experiment2, ..."""
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for d in base.iterdir()
        if d.is_dir() and (m := re.match(r"experiment(\d+)$", d.name))
    ]
    n = max(existing, default=0) + 1
    return base / f"experiment{n}"


def _build_seed_candidate() -> dict[str, str]:
    return {
        "prompt": PROMPT,
    }


# ── Main entry point ───────────────────────────────────────────────────

def run(
    experiments_dir: Path,
    max_calls: int,
    workers: int,
    agent_max_iter: int,
    agent_model: str,
    config_path: Path,
    reflection_lm: str | None,
    use_wandb: bool = False,
    train_minibatch_size: int | None = None,
    val_minibatch_size: int | None = None,
    experiment_name: str | None = None,
) -> None:
    """Run the full GEPA optimization loop."""
    # Resolve experiment directory
    if experiment_name:
        experiment_dir = experiments_dir / experiment_name
    else:
        experiment_dir = _next_experiment_dir(experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Configure evaluator + cache module state
    evaluator.configure_runtime(
        experiment_dir=experiment_dir,
        agent_max_iter=agent_max_iter,
        agent_model=agent_model,
    )
    cache.CACHE_DIR = experiments_dir / ".eval_cache"

    # Load dataset
    train, val = load_dataset()

    # Copy config.json into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    adapter = RedPurpleAdapter(workers=workers)
    callbacks = [TracingCallback(log_dir=experiment_dir / "reflection_logs")]

    print(f"[red-purple] Experiment: {experiment_dir.name}")
    print(f"[red-purple] Train: {len(train)} benchmarks, Val: {len(val)} benchmarks")
    print(f"[red-purple] Budget: {max_calls} calls, {workers} workers")
    print(f"[red-purple] Output: {experiment_dir}\n")

    wandb_kwargs = {}
    if use_wandb:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError("use_wandb=True but WANDB_API_KEY is not set in .env")
        wandb_kwargs = {
            "use_wandb": True,
            "wandb_api_key": wandb_api_key,
            "wandb_init_kwargs": {"name": experiment_dir.name},
        }

    result = optimize(
        seed_candidate=_build_seed_candidate(),
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=train_minibatch_size,
        max_metric_calls=max_calls,
        run_dir=str(experiment_dir / "oa_state"),
        callbacks=callbacks,
        val_evaluation_policy=(
            SubsetValPolicy(k=val_minibatch_size) if val_minibatch_size is not None else "full_eval"
        ),
        skip_perfect_score=False,
        use_cloudpickle=True,
        seed=0,
        **wandb_kwargs,
    )

    # Save best candidate
    (experiment_dir / "best_candidate.json").write_text(
        json.dumps(result.best_candidate, indent=2), encoding="utf-8"
    )

    print(f"\n[red-purple] Done! Best candidate saved to {experiment_dir / 'best_candidate.json'}")
