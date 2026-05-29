"""Core GEPA optimization loop — all logic lives here."""

import json
import re
import shutil
from pathlib import Path

from gepa import optimize

from source.optimize_anything import cache, candidate_store, evaluator
from source.optimize_anything.adapter import RedPurpleAdapter, SubsetValPolicy
from source.optimize_anything.callbacks import TracingCallback
from source.optimize_anything.dataset import load_dataset
from source.optimize_anything.logger import Logger
from source.optimize_anything.agentic_reflector import AgenticReflector
from source.optimize_anything.utils import candidate_hash, dict_to_folder, folder_to_dict, next_experiment_dir

_SEED_DIR = Path(__file__).resolve().parents[2] / "source" / "seed"
_WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"

_active_logger: "Logger | None" = None


def flush_logger() -> None:
    """Write the experiment summary to disk. Safe to call from a signal handler."""
    if _active_logger is not None:
        _active_logger.write_summary()


def _setup_seed(model: str, max_steps: int) -> None:
    """Patch model and maxSteps into the seed ctf-agent.md from config values."""
    agent_md = _SEED_DIR / ".opencode" / "agents" / "ctf-agent.md"
    content = agent_md.read_text(encoding="utf-8")
    content = re.sub(r'(?m)^model:.*$', f'model: "{model}"', content)
    content = re.sub(r'(?m)^maxSteps:.*$', f'maxSteps: {max_steps}', content)
    agent_md.write_text(content, encoding="utf-8")


def _build_seed_candidate() -> dict:
    seed_files = folder_to_dict(_SEED_DIR)
    agent_dir = _WORKSPACE / "agent"
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    dict_to_folder(agent_dir, seed_files)
    files_hash = candidate_hash(seed_files)
    candidate_store.store(files_hash, seed_files)
    return {"files": files_hash}


# ── Main entry point ───────────────────────────────────────────────────

def run(
    experiments_dir: Path,
    max_calls: int,
    workers: int,
    agent_max_iter: int,
    agent_model: str,
    config_path: Path,
    reflection_lm: str | None,
    judge_model: str = "",
    diagnoser_model: str = "",
    gt: bool = False,
    train_minibatch_size: int | None = None,
    val_minibatch_size: int | None = None,
    experiment_name: str | None = None,
    splits_name: str = "splits",
) -> None:
    """Run the full GEPA optimization loop."""
    # Resolve experiment directory
    if experiment_name:
        experiment_dir = experiments_dir / experiment_name
    else:
        experiment_dir = next_experiment_dir(experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    global _active_logger
    logger = Logger(
        reflector_model=reflection_lm or "",
        judge_model=judge_model,
        agent_model=agent_model,
        diagnoser_model=diagnoser_model,
        log_dir=experiment_dir,
    )
    _active_logger = logger

    # Configure evaluator + cache module state
    evaluator.configure_runtime(
        experiment_dir=experiment_dir,
        agent_max_iter=agent_max_iter,
        agent_model=agent_model,
        judge_model=judge_model,
        diagnoser_model=diagnoser_model,
        reflector_model=reflection_lm or "",
        train_size=train_minibatch_size if train_minibatch_size is not None else len(train),
        gt=gt,
        logger=logger,
    )
    cache.CACHE_DIR = experiments_dir / ".eval_cache"
    candidate_store.configure(experiment_dir / ".candidates")

    # Load dataset
    train, val = load_dataset(splits_name=splits_name)

    # Copy config.json into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    _setup_seed(agent_model, agent_max_iter)
    adapter = RedPurpleAdapter(workers=workers)
    seed = _build_seed_candidate()
    callbacks = [TracingCallback(experiment_dir=experiment_dir, seed_candidate=seed, trainset=train, valset=val)]

    print(f"[red-purple] Experiment: {experiment_dir.name}")
    print(f"[red-purple] Train: {len(train)} benchmarks, Val: {len(val)} benchmarks")
    print(f"[red-purple] Budget: {max_calls} calls, {workers} workers")
    print(f"[red-purple] Output: {experiment_dir}\n")

    lm = AgenticReflector(reflection_lm, logger, experiment_dir) if reflection_lm else None
    val_policy = SubsetValPolicy(k=val_minibatch_size) if val_minibatch_size is not None else "full_eval"

    logger.start_logger()

    try:
        result = optimize(
            seed_candidate=seed,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=lm,
            reflection_minibatch_size=train_minibatch_size,
            reflection_prompt_template=None,
            max_metric_calls=max_calls,
            run_dir=str(experiment_dir / "oa_state"),
            callbacks=callbacks,
            val_evaluation_policy=val_policy,
            skip_perfect_score=True,
            use_cloudpickle=True,
            cache_evaluation=True,
            seed=0,
        )
    finally:
        logger.stop_logger()
        _active_logger = None

    # Save best candidate
    (experiment_dir / "best_candidate.json").write_text(
        json.dumps(result.best_candidate, indent=2), encoding="utf-8"
    )

    print(f"\n[red-purple] Done! Best candidate saved to {experiment_dir / 'best_candidate.json'}")
