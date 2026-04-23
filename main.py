#!/usr/bin/env python3
"""Red-Purple — entry point. Reads config.json and launches the GEPA optimization loop."""

import json
import os
import signal
from pathlib import Path

from dotenv import load_dotenv

from source import benchmark
from source.optimize_anything.core_loop import run

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"


def _on_sigint(signum, frame):
    print("\n[red-purple] Ctrl+C — stopping all benchmarks and exiting…", flush=True)
    try:
        benchmark.force_stop_all()
    finally:
        os._exit(130)


def main():
    signal.signal(signal.SIGINT, _on_sigint)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    run(
        experiments_dir=REPO_ROOT / cfg["experiments_dir"],
        max_calls=cfg["max_calls"],
        workers=cfg["workers"],
        agent_max_iter=cfg["agent_max_iter"],
        agent_model=cfg["agent_model"],
        train_minibatch_size=cfg.get("train_minibatch_size"),
        val_minibatch_size=cfg.get("val_minibatch_size"),
        background=cfg["background_context"],
        config_path=CONFIG_PATH,
        reflection_lm=cfg.get("reflection_lm"),
        refiner_lm=cfg.get("refiner_lm"),
        use_wandb=cfg.get("use_wandb", False),
    )


if __name__ == "__main__":
    main()
