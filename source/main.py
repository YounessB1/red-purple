#!/usr/bin/env python3
"""Red-Purple — entry point. Reads config.py and launches the GEPA optimization loop."""

from dotenv import load_dotenv

from source import config
from source.optimize_anything.core_loop import run

load_dotenv()


def main():
    run(
        experiments_dir=config.EXPERIMENTS_DIR,
        max_calls=config.MAX_CALLS,
        workers=config.WORKERS,
        agent_max_iter=config.AGENT_MAX_ITER,
        background=config.BACKGROUND_CONTEXT,
        reflection_lm=config.REFLECTION_LM,
    )


if __name__ == "__main__":
    main()
