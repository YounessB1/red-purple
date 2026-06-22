#!/usr/bin/env python3
"""Red-Purple — entry point. Reads config.json and launches the GEPA optimization loop."""

import os
import signal
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv

from source import benchmark
from source.optimize_anything.core_loop import run, flush_logger

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


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


def main():
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        urllib.request.urlopen(
            urllib.request.Request("http://localhost:8000/reset", method="POST"),
            timeout=5,
        )
    except Exception:
        pass

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    run(
        experiments_dir=REPO_ROOT / cfg["experiments_dir"],
        max_calls=cfg["max_calls"],
        workers=cfg["workers"],
        ctf_agent=cfg["ctf_agent"],
        scorer=cfg["scorer"],
        diagnoser=cfg["diagnoser"],
        reflector=cfg["reflector"],
        merger=cfg["merger"],
        config_path=CONFIG_PATH,
        experiment_name=cfg.get("experiment_name"),
        splits_name=cfg.get("splits", "splits"),
    )


if __name__ == "__main__":
    main()
