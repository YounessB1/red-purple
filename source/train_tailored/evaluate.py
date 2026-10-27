#!/usr/bin/env python3
"""Standalone evaluation harness for the train-set-tailored CTF agent.

Runs the agent defined in this folder (prompt.md/AGENTS.md/skills) against
XBOW benchmarks, driving the same lifecycle GEPA evaluation uses
(source.benchmark for Docker; the containerized agent-server for the
OpenCode agent run) without any of the optimizer/judge/diagnoser
machinery — this is for measuring one fixed, hand-built agent, not
evolving one.

Unlike source/owasp_baseline/ (a methodology-generic agent meant to
transfer to unseen targets), this agent is deliberately overfit to this
exact 104-benchmark XBOW train set: its skills encode the specific
vulnerable parameter, payload, and quirk extracted from each benchmark's
own source code (see the per-tag skill files' "Benchmark notes" sections),
not general technique ladders alone. It exists as a ceiling/oracle
baseline — how much does knowing the train set's exact answers move the
pass rate, as a point of comparison against agents that have to
generalize. It deliberately does NOT use the FLAG=sha256(BENCHMARK_ID)
shortcut (see xbow/common.mk) discovered while building it — that would
skip exploitation entirely and make the pass rate meaningless as a
capability signal.

The agent MUST run inside the `agent-server` container (source/agent/),
not as a bare `opencode` subprocess on the host: that container is a Kali
image with sqlmap/nmap/hydra/gobuster/etc. preinstalled, which several of
this agent's skills assume are available. Start it first if it isn't
already running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.train_tailored.evaluate                       # all 104 benchmarks
    python3 -m source.train_tailored.evaluate --benchmarks XBEN-001-24 XBEN-004-24
    python3 -m source.train_tailored.evaluate --workers 6
    python3 -m source.train_tailored.evaluate --limit 5              # cheap dry run
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = Path(__file__).resolve().parent
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()

# mysql:5.7 and other old base images were never published for arm64.
# Force amd64 + emulation instead of requiring every benchmark's
# docker-compose.yml to be patched (see xbow/test_benchmarks.py). No-op on
# native linux/amd64 hosts, and scoped to this process only.
os.environ.setdefault("DOCKER_DEFAULT_PLATFORM", "linux/amd64")

from source.benchmark import BENCHMARKS_DIR  # noqa: E402
from source.eval_harness import agent_model, check_agent_server, load_candidate, run_sweep  # noqa: E402

_EXCLUDE = frozenset({"evaluate.py", "__init__.py", "results", "__pycache__", "PROVENANCE.md"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="*", default=None, help="specific benchmark IDs (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N benchmarks (dry run)")
    args = parser.parse_args()

    check_agent_server(_AGENT_SERVER_URL)

    if args.benchmarks:
        benchmark_ids = args.benchmarks
    else:
        benchmark_ids = sorted(
            p.name for p in BENCHMARKS_DIR.iterdir()
            if p.is_dir() and (p / "benchmark.json").exists()
        )

    candidate = load_candidate(_AGENT_DIR, _EXCLUDE)
    out_dir = _REPO_ROOT / "experiments" / "train_tailored" / agent_model(candidate).split("/")[-1]

    run_sweep(
        benchmark_ids, candidate, out_dir,
        benchmarks_dir=BENCHMARKS_DIR, agent_server_url=_AGENT_SERVER_URL,
        workers=args.workers, limit=args.limit,
    )


if __name__ == "__main__":
    main()
