#!/usr/bin/env python3
"""Red-Purple — entry point. Reads config.yaml and launches the GEPA optimization loop.

config.yaml can be either a single experiment (flat dict, original format) or a list
of experiments that run sequentially.  Both formats are supported transparently.
"""

import os
import signal
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Many benchmark base images (e.g. mysql:5.7) were never published for
# arm64. Force amd64 + emulation instead of requiring every benchmark's
# docker-compose.yml to be patched with `platform: linux/amd64`. Scoped to
# this process's subprocess calls only (source.benchmark's docker/make
# invocations inherit it) — doesn't touch the caller's shell environment,
# and is a no-op on native linux/amd64 hosts (see xbow/test_benchmarks.py).
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source import benchmark
from source.optimize_anything.core_loop import run, flush_logger

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
_AGENTS_DIR = REPO_ROOT / ".opencode" / "agents"
_WORKSPACE  = REPO_ROOT / "workspace"


def _reset_server() -> None:
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://localhost:8000/reset", method="POST"),
            timeout=5,
        )
    except Exception:
        pass


def _on_sigint(signum, frame) -> None:
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


def main() -> None:
    signal.signal(signal.SIGINT, _on_sigint)

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # Support both a single config (dict) and a list of configs
    experiments: list[dict] = raw if isinstance(raw, list) else [raw]
    total = len(experiments)

    # Snapshot agent .md files before any experiment modifies them.
    # patch_agent() and build_agent_md() edit these in-place; each experiment
    # must start from the same unpatched originals.
    agent_originals: dict[Path, str] = {
        p: p.read_text(encoding="utf-8")
        for p in sorted(_AGENTS_DIR.glob("*.md"))
        if p.is_file()
    }

    for i, cfg in enumerate(experiments, 1):
        if total > 1:
            name = cfg.get("experiment_name", f"experiment_{i}")
            print(f"\n{'=' * 60}")
            print(f"[red-purple] Experiment {i}/{total}: {name}")
            print(f"{'=' * 60}\n")

        # Restore agent .md files so every experiment starts from clean originals.
        for path, content in agent_originals.items():
            path.write_text(content, encoding="utf-8")

        # Clear the patch blocklist: it accumulates within an experiment by design,
        # but must not carry over to the next one.
        blocklist = _WORKSPACE / "patch_blocklist.json"
        if blocklist.exists():
            blocklist.unlink()

        _reset_server()

        run(
            experiments_dir=REPO_ROOT / cfg["experiments_dir"],
            max_calls=cfg["max_calls"],
            workers=cfg["workers"],
            ctf_agent=cfg["ctf_agent"],
            scorer=cfg["scorer"],
            diagnoser=cfg["diagnoser"],
            reflector=cfg["reflector"],
            merger=cfg["merger"],
            experiment_config=cfg,
            experiment_name=cfg.get("experiment_name"),
            splits_name=cfg.get("splits", "splits"),
        )

    if total > 1:
        print(f"\n[red-purple] All {total} experiments complete.")


if __name__ == "__main__":
    main()
