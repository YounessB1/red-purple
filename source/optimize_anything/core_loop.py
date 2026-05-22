"""Core GEPA optimization loop — all logic lives here."""

import json
import random
import re
import shutil
from pathlib import Path

from gepa import optimize
from gepa.strategies.eval_policy import FullEvaluationPolicy

from source.optimize_anything import cache, candidate_store, evaluator
from source.optimize_anything.adapter import RedPurpleAdapter
from source.optimize_anything.callbacks import TracingCallback
from source.optimize_anything.dataset import load_dataset
from source.optimize_anything.logger import Logger
from source.optimize_anything.agentic_reflector import AgenticReflector
from source.optimize_anything.utils import candidate_hash

_SEED_DIR = Path(__file__).resolve().parents[2] / "source" / "seed"
_WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"

_active_logger: "Logger | None" = None


def flush_logger() -> None:
    """Write the experiment summary to disk. Safe to call from a signal handler."""
    if _active_logger is not None:
        _active_logger.write_summary()


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


def _build_seed_candidate() -> dict:
    seed_files = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            seed_files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")
    # Populate workspace/agent/ from seed (preserve directory structure, skip .gitkeep)
    agent_dir = _WORKSPACE / "agent"
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    for d in sorted(_SEED_DIR.rglob("*")):
        if d.is_dir():
            (agent_dir / d.relative_to(_SEED_DIR)).mkdir(parents=True, exist_ok=True)
    for rel, content in seed_files.items():
        dest = agent_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
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
) -> None:
    """Run the full GEPA optimization loop."""
    # Resolve experiment directory
    if experiment_name:
        experiment_dir = experiments_dir / experiment_name
    else:
        experiment_dir = _next_experiment_dir(experiments_dir)
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
    train, val = load_dataset()

    # Copy config.json into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    adapter = RedPurpleAdapter(workers=workers)
    seed = _build_seed_candidate()
    callbacks = [TracingCallback(experiment_dir=experiment_dir, seed_candidate=seed, trainset=train, valset=val)]

    print(f"[red-purple] Experiment: {experiment_dir.name}")
    print(f"[red-purple] Train: {len(train)} benchmarks, Val: {len(val)} benchmarks")
    print(f"[red-purple] Budget: {max_calls} calls, {workers} workers")
    print(f"[red-purple] Output: {experiment_dir}\n")

    lm = AgenticReflector(reflection_lm, logger, experiment_dir) if reflection_lm else None

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
            val_evaluation_policy=(
                SubsetValPolicy(k=val_minibatch_size) if val_minibatch_size is not None else "full_eval"
            ),
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
