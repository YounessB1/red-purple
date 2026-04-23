"""Core GEPA optimization loop — all logic lives here."""

import json
import os
import random
import re
import shutil
from pathlib import Path

from gepa.optimize_anything import (
    optimize_anything,
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
    RefinerConfig,
    TrackingConfig,
)
from gepa.strategies.eval_policy import FullEvaluationPolicy

from source.agent.seed import SYSTEM_PROMPT, DEFAULT_TASK
from source.optimize_anything.dataset import load_dataset
from source.optimize_anything import cache, evaluator


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


# ── Component config builders ──────────────────────────────────────────

def _build_engine_config(
    max_calls: int,
    workers: int,
    run_dir: Path,
    val_minibatch_size: int | None,
) -> EngineConfig:
    kwargs: dict = {
        "max_metric_calls": max_calls,
        "run_dir": str(run_dir),
        "parallel": workers > 1,
        "max_workers": workers,
    }
    if val_minibatch_size is not None:
        kwargs["val_evaluation_policy"] = SubsetValPolicy(k=val_minibatch_size)
    return EngineConfig(**kwargs)


def _build_reflection_config(
    reflection_lm: str | None,
    train_minibatch_size: int | None,
) -> ReflectionConfig:
    kwargs: dict = {}
    if reflection_lm:
        kwargs["reflection_lm"] = reflection_lm
    if train_minibatch_size is not None:
        kwargs["reflection_minibatch_size"] = train_minibatch_size
    return ReflectionConfig(**kwargs)


def _build_refiner_config(refiner_lm: str | None) -> RefinerConfig | None:
    if not refiner_lm:
        return None
    return RefinerConfig(refiner_lm=refiner_lm)


def _build_tracking_config(use_wandb: bool, run_name: str) -> TrackingConfig:
    if not use_wandb:
        return TrackingConfig()
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("use_wandb=True but WANDB_API_KEY is not set in .env")
    return TrackingConfig(
        use_wandb=True,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs={"name": run_name},
    )


# ── Main entry point ───────────────────────────────────────────────────

def run(
    experiments_dir: Path,
    max_calls: int,
    workers: int,
    agent_max_iter: int,
    agent_model: str,
    background: str,
    config_path: Path,
    reflection_lm: str | None,
    refiner_lm: str | None = None,
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
    evaluator.EXPERIMENT_DIR = experiment_dir
    evaluator.AGENT_MAX_ITER = agent_max_iter
    evaluator.AGENT_MODEL = agent_model
    cache.CACHE_DIR = experiments_dir / ".eval_cache"

    # Load dataset
    train, val = load_dataset()

    # Copy config.json into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    # Build GEPA config
    gepa_config = GEPAConfig(
        engine=_build_engine_config(
            max_calls=max_calls,
            workers=workers,
            run_dir=experiment_dir / "oa_state",
            val_minibatch_size=val_minibatch_size,
        ),
        reflection=_build_reflection_config(
            reflection_lm=reflection_lm,
            train_minibatch_size=train_minibatch_size,
        ),
        tracking=_build_tracking_config(
            use_wandb=use_wandb,
            run_name=experiment_dir.name,
        ),
        refiner=_build_refiner_config(refiner_lm),
        merge=None,           # None to disable (default)
        stop_callbacks=None,  # custom stopping logic
    )

    print(f"[red-purple] Experiment: {experiment_dir.name}")
    print(f"[red-purple] Train: {len(train)} benchmarks, Val: {len(val)} benchmarks")
    print(f"[red-purple] Budget: {max_calls} calls, {workers} workers")
    print(f"[red-purple] Output: {experiment_dir}\n")

    # Run GEPA optimization
    result = optimize_anything(
        seed_candidate={
            "system_prompt": SYSTEM_PROMPT,
            "default_task": DEFAULT_TASK,
        },
        evaluator=evaluator.evaluate,
        dataset=train,
        valset=val,
        objective="Maximize CTF flag capture rate across diverse web vulnerability types and difficulty levels",
        background=background,
        config=gepa_config,
    )

    # Save best candidate
    (experiment_dir / "best_candidate.json").write_text(
        json.dumps(result.best_candidate, indent=2), encoding="utf-8"
    )

    print(f"\n[red-purple] Done! Best candidate saved to {experiment_dir / 'best_candidate.json'}")
