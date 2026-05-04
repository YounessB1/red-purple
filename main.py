#!/usr/bin/env python3
"""Red-Purple — entry point. Reads config.json and launches the GEPA optimization loop."""

import json
import os
import signal
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from source import benchmark
from source.optimize_anything.core_loop import run, flush_logger

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"


def _on_sigint(signum, frame):
    print("\n[red-purple] Ctrl+C — stopping all benchmarks and exiting…", flush=True)
    try:
        benchmark.force_stop_all()
        try:
            urllib.request.urlopen(
                urllib.request.Request("http://localhost:8000/cancel", method="POST"),
                timeout=5,
            )
        except Exception:
            pass
    finally:
        flush_logger()
        os._exit(130)


def _load_background_context(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    path = base / value
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return value


def main():
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        urllib.request.urlopen(
            urllib.request.Request("http://localhost:8000/reset", method="POST"),
            timeout=5,
        )
    except Exception:
        pass

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    run(
        experiments_dir=REPO_ROOT / cfg["experiments_dir"],
        max_calls=cfg["max_calls"],
        workers=cfg["workers"],
        agent_max_iter=cfg["agent_max_iter"],
        agent_model=cfg["agent_model"],
        judge_model=cfg.get("judge_model", ""),
        gt=cfg.get("gt", False),
        train_minibatch_size=cfg.get("train_minibatch_size"),
        val_minibatch_size=cfg.get("val_minibatch_size"),
        config_path=CONFIG_PATH,
        reflection_lm=cfg.get("reflection_lm"),
        use_wandb=cfg.get("use_wandb", False),
        experiment_name=cfg.get("experiment_name"),
        background_context=_load_background_context(cfg.get("background_context"), REPO_ROOT),
    )


if __name__ == "__main__":
    main()
