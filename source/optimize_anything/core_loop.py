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
_AGENTS_DIR = Path(__file__).resolve().parents[2] / ".opencode" / "agents"

_active_logger: "Logger | None" = None


def flush_logger() -> None:
    """Write the experiment summary to disk. Safe to call from a signal handler."""
    if _active_logger is not None:
        _active_logger.write_summary()


def patch_agent(md_path: Path, md_params: dict) -> None:
    """Write config md_params into an OpenCode agent .md frontmatter.

    For each key/value pair, replaces the matching `key: ...` line in the
    frontmatter.  Silently skips None values so absent config keys leave the
    .md default intact.  String values are double-quoted; numbers/bools are
    written as-is.
    """
    content = md_path.read_text(encoding="utf-8")
    for key, value in md_params.items():
        if value is None:
            continue
        formatted = f'"{value}"' if isinstance(value, str) else str(value)
        content = re.sub(rf'(?m)^{re.escape(key)}:.*$', f'{key}: {formatted}', content)
    md_path.write_text(content, encoding="utf-8")


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
    ctf_agent: dict,
    scorer: dict,
    diagnoser: dict,
    reflector: dict,
    merger: dict,
    config_path: Path,
    experiment_name: str | None = None,
    splits_name: str = "splits",
) -> None:
    """Run the full GEPA optimization loop.

    ctf_agent / scorer / diagnoser / reflector / merger are the config sections
    from config.yaml.  Each has an ``md`` sub-dict with the keys to patch into
    the corresponding OpenCode agent .md, and runtime keys at the top level.
    """
    # ── Extract runtime params from each section ───────────────────────
    agent_model    = ctf_agent["md"]["model"]
    agent_max_iter = ctf_agent["md"]["maxSteps"]

    judge_model     = scorer["md"].get("model", "")
    gt              = scorer.get("gt", False)

    diagnoser_model = diagnoser["md"].get("model", "")
    diagnoser_gt    = diagnoser.get("gt", False)

    reflector_model  = reflector["md"].get("model", "")
    reflector_agent  = reflector.get("agent", "reflector")
    agentic          = reflector.get("agentic", True)
    train_minibatch  = reflector.get("train_minibatch_size")
    val_minibatch    = reflector.get("val_minibatch_size")

    merger_model    = merger["md"].get("model", "")
    merger_agent    = merger.get("agent", "merger")
    merge_threshold = merger.get("merge_threshold", 0.3)

    # ── Patch agent .md files from config ─────────────────────────────
    patch_agent(_SEED_DIR / ".opencode" / "agents" / "ctf-agent.md", ctf_agent["md"])
    patch_agent(_AGENTS_DIR / "scorer.md",   scorer["md"])
    patch_agent(_AGENTS_DIR / "diagnoser.md", diagnoser["md"])
    patch_agent(_AGENTS_DIR / f"{reflector_agent}.md", reflector["md"])
    patch_agent(_AGENTS_DIR / f"{merger_agent}.md",    merger["md"])

    # ── Resolve experiment directory ───────────────────────────────────
    if experiment_name:
        experiment_dir = experiments_dir / experiment_name
    else:
        experiment_dir = next_experiment_dir(experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    global _active_logger
    logger = Logger(
        reflector_model=reflector_model,
        judge_model=judge_model,
        agent_model=agent_model,
        diagnoser_model=diagnoser_model,
        log_dir=experiment_dir,
    )
    _active_logger = logger

    # Load dataset early so train_size is available for configure_runtime
    train, val = load_dataset(splits_name=splits_name)

    # Configure evaluator + cache module state
    evaluator.configure_runtime(
        experiment_dir=experiment_dir,
        agent_max_iter=agent_max_iter,
        agent_model=agent_model,
        judge_model=judge_model,
        diagnoser_model=diagnoser_model,
        reflector_model=reflector_model,
        train_size=train_minibatch if train_minibatch is not None else len(train),
        gt=gt,
        diagnoser_gt=diagnoser_gt,
        logger=logger,
    )
    cache.CACHE_DIR = experiments_dir / ".eval_cache"
    candidate_store.configure(experiment_dir / ".candidates")

    # Copy config.yaml into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    adapter = RedPurpleAdapter(workers=workers)
    seed = _build_seed_candidate()
    cache.SEED_CANDIDATE_HASH = candidate_hash(seed)
    callbacks = [TracingCallback(experiment_dir=experiment_dir, seed_candidate=seed, trainset=train, valset=val)]

    print(f"[red-purple] Experiment: {experiment_dir.name}")
    print(f"[red-purple] Train: {len(train)} benchmarks, Val: {len(val)} benchmarks")
    print(f"[red-purple] Budget: {max_calls} calls, {workers} workers")
    print(f"[red-purple] Output: {experiment_dir}\n")

    lm = (
        AgenticReflector(
            reflector_model, logger, experiment_dir,
            merge_threshold=merge_threshold,
            reflector_agent=reflector_agent,
            merger_agent=merger_agent,
            merger_model=merger_model or reflector_model,
        )
        if agentic and reflector_model else None
    )
    val_policy = SubsetValPolicy(k=val_minibatch) if val_minibatch is not None else "full_eval"

    logger.start_logger()

    try:
        result = optimize(
            seed_candidate=seed,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=lm,
            reflection_minibatch_size=train_minibatch,
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
