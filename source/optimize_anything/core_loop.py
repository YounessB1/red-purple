"""Core GEPA optimization loop — all logic lives here."""

import json
import os
import random
import re
import shutil
from pathlib import Path

from gepa import optimize
from gepa.strategies.eval_policy import FullEvaluationPolicy

from source.llm import LLM
from source.optimize_anything import cache, evaluator
from source.optimize_anything.adapter import RedPurpleAdapter
from source.optimize_anything.callbacks import TracingCallback
from source.optimize_anything.dataset import load_dataset
from source.optimize_anything.logger import Logger
from source.seed import PROMPT


_active_logger: "Logger | None" = None


def flush_logger() -> None:
    """Write the experiment summary to disk. Safe to call from a signal handler."""
    if _active_logger is not None:
        _active_logger.write_summary()


class ReflectorLLM:
    """Thin wrapper around LLM that logs reflector token usage to the experiment Logger."""

    def __init__(self, model: str, logger: Logger) -> None:
        self._llm = LLM(model)
        self._logger = logger

    def __call__(self, prompt: str | list[dict]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        content, input_tokens, output_tokens = self._llm.generate(messages)
        self._logger.log_reflector(input_tokens, output_tokens, messages, content)
        return content


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

def _build_reflection_prompt_template(background_context: str) -> str:
    return (
        f"## Domain Context\n\n{background_context}\n\n"
        "I provided an assistant with the following instructions to perform a task for me:\n"
        "```\n<curr_param>\n```\n\n"
        "The following are examples of different task inputs provided to the assistant along with "
        "the assistant's response for each of them, and some feedback on how the assistant's response "
        "could be better:\n"
        "```\n<side_info>\n```\n\n"
        "Your task is to write a new instruction for the assistant.\n\n"
        "Read the inputs carefully and identify the input format and infer detailed task description "
        "about the task I wish to solve with the assistant.\n\n"
        "Read all the assistant responses and the corresponding feedback. Identify all niche and domain "
        "specific factual information about the task and include it in the instruction, as a lot of it "
        "may not be available to the assistant in the future. The assistant may have utilized a "
        "generalizable strategy to solve the task, if so, include that in the instruction as well.\n\n"
        "Provide the new instructions within ``` blocks."
    )


def run(
    experiments_dir: Path,
    max_calls: int,
    workers: int,
    agent_max_iter: int,
    agent_model: str,
    config_path: Path,
    reflection_lm: str | None,
    judge_model: str = "",
    gt: bool = False,
    use_wandb: bool = False,
    train_minibatch_size: int | None = None,
    val_minibatch_size: int | None = None,
    experiment_name: str | None = None,
    background_context: str | None = None,
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
        log_dir=experiment_dir,
    )
    _active_logger = logger

    # Configure evaluator + cache module state
    evaluator.configure_runtime(
        experiment_dir=experiment_dir,
        agent_max_iter=agent_max_iter,
        agent_model=agent_model,
        judge_model=judge_model,
        gt=gt,
        logger=logger,
    )
    cache.CACHE_DIR = experiments_dir / ".eval_cache"

    # Load dataset
    train, val = load_dataset()

    # Copy config.json into experiment dir for reproducibility
    shutil.copy2(config_path, experiment_dir / "config.json")

    adapter = RedPurpleAdapter(workers=workers)
    seed = _build_seed_candidate()
    callbacks = [TracingCallback(log_dir=experiment_dir / "reflection_logs", seed_candidate=seed)]

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

    lm = ReflectorLLM(reflection_lm, logger) if reflection_lm else None

    logger.start_logger()

    try:
        result = optimize(
            seed_candidate=seed,
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=lm,
            reflection_minibatch_size=train_minibatch_size,
            reflection_prompt_template=_build_reflection_prompt_template(background_context) if background_context else None,
            max_metric_calls=max_calls,
            run_dir=str(experiment_dir / "oa_state"),
            callbacks=callbacks,
            val_evaluation_policy=(
                SubsetValPolicy(k=val_minibatch_size) if val_minibatch_size is not None else "full_eval"
            ),
            skip_perfect_score=False,
            use_cloudpickle=True,
            cache_evaluation=True,
            seed=0,
            **wandb_kwargs,
        )
    finally:
        logger.stop_logger()
        _active_logger = None

    # Save best candidate
    (experiment_dir / "best_candidate.json").write_text(
        json.dumps(result.best_candidate, indent=2), encoding="utf-8"
    )

    print(f"\n[red-purple] Done! Best candidate saved to {experiment_dir / 'best_candidate.json'}")
