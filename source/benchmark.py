"""Benchmark lifecycle management — start, stop, port discovery for XBOW challenges."""

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"

# Matches "0.0.0.0:35167->80/tcp" — captures the host port
_HOST_PORT_RE = re.compile(r"0\.0\.0\.0:(\d+)->\d+/tcp")

_active_benchmarks: set[str] = set()


def start_benchmark(benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR) -> int:
    """Start a benchmark's Docker containers and return the host port.

    Runs `make run` (which builds + docker compose up --wait), then discovers
    the dynamically assigned host port via `docker ps`.
    """
    bench_dir = benchmarks_dir / benchmark_id
    result = subprocess.run(
        ["make", "run"],
        cwd=bench_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`make run` failed for {benchmark_id} (exit {result.returncode})\n"
            f"cwd: {bench_dir}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    _active_benchmarks.add(benchmark_id)
    return find_host_port(benchmark_id)


def stop_benchmark(benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR) -> None:
    """Tear down a benchmark's Docker containers."""
    _active_benchmarks.discard(benchmark_id)
    bench_dir = benchmarks_dir / benchmark_id
    subprocess.run(
        ["docker", "compose", "down", "--remove-orphans"],
        cwd=bench_dir,
        capture_output=True,
        timeout=60,
    )


def force_stop_all(benchmarks_dir: Path = BENCHMARKS_DIR) -> None:
    """Fire-and-forget teardown of every active benchmark.

    Spawns detached `docker compose down` processes that survive the Python
    process exiting. Used from the SIGINT handler to recover the terminal
    instantly without waiting for stuck worker threads.
    """
    for benchmark_id in list(_active_benchmarks):
        try:
            subprocess.Popen(
                ["docker", "compose", "down", "--remove-orphans"],
                cwd=benchmarks_dir / benchmark_id,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass
    _active_benchmarks.clear()


def find_host_port(benchmark_id: str) -> int:
    """Find the host-mapped port for a benchmark using `docker ps`.

    Filters containers by benchmark name and returns the host port of the
    container that has a 0.0.0.0:PORT->CONTAINER/tcp mapping (the externally
    accessible service, not internal-only containers).
    """
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={benchmark_id.lower()}", "--format", "{{.Ports}}"],
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.strip().splitlines():
        match = _HOST_PORT_RE.search(line)
        if match:
            return int(match.group(1))

    raise RuntimeError(
        f"No host-mapped port found for benchmark '{benchmark_id}'. "
        f"docker ps output:\n{result.stdout}"
    )
