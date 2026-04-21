"""Core GEPA optimization loop — all logic lives here."""

import json
import re
from pathlib import Path

from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

from source.agent.seed import SYSTEM_PROMPT, DEFAULT_TASK
from source.optimize_anything.dataset import load_dataset
from source.optimize_anything import evaluator


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


def run(
    experiments_dir: Path,
    max_calls: int,
    workers: int,
    agent_max_iter: int,
    background: str,
    reflection_lm: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Run the full GEPA optimization loop.

    Args:
        experiments_dir: Base dir for experiments (e.g. repo/experiments/).
        max_calls: Total evaluator budget (candidate x benchmark pairs).
        workers: Number of parallel benchmark evaluations.
        agent_max_iter: Max LLM iterations per agent run.
        background: Context string for GEPA's reflection LLM.
        reflection_lm: Model for GEPA reflection (None = GEPA default).
        experiment_name: Explicit experiment name, or None for auto-increment.
    """
    # Resolve experiment directory
    if experiment_name:
        experiment_dir = experiments_dir / experiment_name
    else:
        experiment_dir = _next_experiment_dir(experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Configure evaluator module-level state
    evaluator.EXPERIMENT_DIR = experiment_dir
    evaluator.AGENT_MAX_ITER = agent_max_iter

    # Load dataset
    train, val = load_dataset()

    # Save experiment config for reproducibility
    (experiment_dir / "config.json").write_text(
        json.dumps({
            "experiment": experiment_dir.name,
            "max_calls": max_calls,
            "workers": workers,
            "reflection_lm": reflection_lm,
            "agent_max_iter": agent_max_iter,
            "train_size": len(train),
            "val_size": len(val),
        }, indent=2),
        encoding="utf-8",
    )

    # Build GEPA config
    engine_kwargs = {
        "max_metric_calls": max_calls,
        "run_dir": str(experiment_dir / "gepa_state"),
        "parallel": workers > 1,
        "max_workers": workers,
    }

    reflection_kwargs = {}
    if reflection_lm:
        reflection_kwargs["reflection_lm"] = reflection_lm

    gepa_config = GEPAConfig(
        engine=EngineConfig(**engine_kwargs),
        reflection=ReflectionConfig(**reflection_kwargs),
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
