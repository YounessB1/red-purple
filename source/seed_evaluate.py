#!/usr/bin/env python3
"""Standalone evaluation harness for the seed CTF agent against nyu_ctf.

Runs the agent defined in source/seed/ (prompt.md/opencode.json/ctf-agent.md
— the same starting config GEPA optimizes from) against the nyu_ctf
benchmarks that have a confirmed, verified flag-retrieval path (see
nyu_ctf/PoC.md), driving the same lifecycle GEPA evaluation uses
(source.benchmark for Docker; the containerized agent-server for the
OpenCode agent run) without any of the optimizer/judge/diagnoser
machinery — this measures the seed agent as-is against a benchmark source
it was never tuned against.

Deliberately NOT placed inside source/seed/ itself: that directory is read
as-is via folder_to_dict() with no exclude filter by several places in
source/optimize_anything/ (core_loop.py's _build_seed_candidate,
evaluator.py's fallback, cache.py, eval_test_set.py) to build GEPA's actual
seed candidate payload — dropping this script in there would leak it into
every future GEPA run's agent workdir.

The agent MUST run inside the `agent-server` container (source/agent/),
not as a bare `opencode` subprocess on the host: that container is a Kali
image with sqlmap/nmap/hydra/gobuster/etc. preinstalled. Start it first if
it isn't already running:
    docker compose -f source/agent/docker-compose.yml up -d --build

Usage:
    python3 -m source.seed_evaluate                            # all 20 verified benchmarks
    python3 -m source.seed_evaluate --benchmarks NYU-023 NYU-001
    python3 -m source.seed_evaluate --workers 3
    python3 -m source.seed_evaluate --limit 1                  # cheap dry run
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_DIR = _REPO_ROOT / "source" / "seed"
_AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8000")

load_dotenv()

# Unlike xbow (some legacy base images, e.g. mysql:5.7, were never published
# for arm64 — see source/train_tailored/evaluate.py), nyu_ctf's images build and
# were verified natively on both arm64 and amd64 hosts this session, so no
# DOCKER_DEFAULT_PLATFORM override is needed — forcing amd64 here would only
# break already-built native-arch images ("found but does not provide the
# specified platform") and add needless emulation overhead.

from source.benchmark import NYU_BENCHMARKS_DIR  # noqa: E402
from source.eval_harness import agent_model, check_agent_server, load_candidate, run_sweep  # noqa: E402

# Only the nyu_ctf benchmarks with a confirmed, verified flag-retrieval path
# through the intended vulnerability — see nyu_ctf/PoC.md. Deliberately NOT a
# directory scan: nyu_ctf/benchmarks also contains NYU-003/004/009/014 (no
# confirmed exploit path, marked "Status: unresolved" in PoC.md) alongside
# the excluded NYU-011/022 (infra-incompatible, listed in
# nyu_ctf/excluded_challenges.json), which would just produce noisy failures
# unrelated to the seed agent's actual capability.
_NYU_VERIFIED_BENCHMARKS = (
    "NYU-001", "NYU-002", "NYU-005", "NYU-006", "NYU-007", "NYU-008",
    "NYU-010", "NYU-012", "NYU-013", "NYU-015", "NYU-016", "NYU-017",
    "NYU-018", "NYU-019", "NYU-020", "NYU-021", "NYU-023", "NYU-024",
    "NYU-025", "NYU-026",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="specific benchmark IDs (default: all 20 verified-working nyu_ctf benchmarks)",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel workers (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N benchmarks (dry run)")
    args = parser.parse_args()

    check_agent_server(_AGENT_SERVER_URL)

    benchmark_ids = args.benchmarks if args.benchmarks else list(_NYU_VERIFIED_BENCHMARKS)

    candidate = load_candidate(_SEED_DIR, frozenset())
    out_dir = _REPO_ROOT / "experiments" / "seed_nyu_ctf" / agent_model(candidate).split("/")[-1]

    run_sweep(
        benchmark_ids, candidate, out_dir,
        benchmarks_dir=NYU_BENCHMARKS_DIR, agent_server_url=_AGENT_SERVER_URL,
        workers=args.workers, limit=args.limit,
    )


if __name__ == "__main__":
    main()
