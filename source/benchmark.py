"""Benchmark lifecycle management — start, stop, port discovery for XBOW challenges."""

import json
import random
import re
import subprocess
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"

# Matches "0.0.0.0:35167->80/tcp" — captures the host port
_HOST_PORT_RE = re.compile(r"0\.0\.0\.0:(\d+)->\d+/tcp")

_active_benchmarks: set[str] = set()

# `docker compose up --wait` (what `make run` used to call) fails transiently
# under concurrent load: a DB container that's merely slow to boot gets a
# false "unhealthy" verdict from Docker's own healthcheck loop (~50s window
# baked into the benchmark's docker-compose.yml — interval*retries, no
# start_period — not something we control here), and compose exits the
# instant it sees that first unhealthy event even though Docker keeps
# re-probing afterward and the container often recovers seconds later.
#
# Two real mitigations, not just a bigger timeout guess:
#   1. Throttle how many containers are simultaneously *in that vulnerable
#      boot window*, independent of how many workers are running agents —
#      that's the actual mechanism making boots slow (host CPU/IO
#      contention), not the healthcheck's timing itself.
#   2. Poll real container health ourselves instead of trusting compose's
#      single-shot exit-on-first-unhealthy-event.
_START_RETRIES = 4
_START_RETRY_DELAY_RANGE = (8, 15)  # seconds, randomized to avoid thundering herd
_START_CONCURRENCY = 6  # max containers concurrently mid-boot, regardless of total workers
_HEALTH_TIMEOUT = 300  # seconds to wait for real container health after `up -d`
_HEALTH_POLL_INTERVAL = 3  # seconds between health polls

_start_semaphore = threading.Semaphore(_START_CONCURRENCY)


def start_benchmark(benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR) -> int:
    """Start a benchmark's Docker containers and return the host port.

    Builds, starts detached, then polls real container health ourselves
    (see module-level comment above) instead of trusting `docker compose up
    --wait`'s brittle single-shot verdict. Retries the whole cycle on
    failure — concurrent workers routinely contend for host resources
    during container boot.
    """
    bench_dir = benchmarks_dir / benchmark_id
    last_error: Exception | None = None

    for attempt in range(_START_RETRIES):
        try:
            subprocess.run(
                ["make", "build"],
                cwd=bench_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            # Held across up+health-poll, not just the `up -d` dispatch —
            # `up -d` returns almost instantly, so only throttling the
            # dispatch wouldn't limit how many containers are actually
            # mid-boot at once, which is the thing causing contention.
            with _start_semaphore:
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=bench_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                _wait_for_healthy(bench_dir)
            _active_benchmarks.add(benchmark_id)
            return find_host_port(benchmark_id)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as e:
            last_error = e
            if attempt < _START_RETRIES - 1:
                # Clear any half-started stack before retrying so it doesn't
                # collide with the next attempt (stale network/container names).
                # Best-effort only: under the same load that caused the failure,
                # this cleanup can itself hang past its timeout. An uncaught
                # TimeoutExpired here would propagate all the way up through
                # evaluate() and get silently converted to a fake score by
                # RedPurpleAdapter — worse than just skipping a cleanup attempt.
                try:
                    subprocess.run(
                        ["docker", "compose", "down", "--remove-orphans"],
                        cwd=bench_dir,
                        capture_output=True,
                        timeout=60,
                    )
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(random.uniform(*_START_RETRY_DELAY_RANGE))

    detail = str(last_error)
    if isinstance(last_error, subprocess.CalledProcessError):
        detail = (
            f"exit {last_error.returncode}\n"
            f"--- stdout ---\n{last_error.stdout}\n--- stderr ---\n{last_error.stderr}"
        )
    raise RuntimeError(
        f"starting {benchmark_id} failed after {_START_RETRIES} attempts\n"
        f"cwd: {bench_dir}\n{detail}"
    )


def _wait_for_healthy(
    bench_dir: Path,
    timeout: float = _HEALTH_TIMEOUT,
    poll_interval: float = _HEALTH_POLL_INTERVAL,
) -> None:
    """Poll real container health instead of trusting `docker compose up
    --wait`'s single-shot verdict. Docker keeps re-probing on its own
    interval even after an 'unhealthy' event, so a merely-slow container
    (common under this pipeline's concurrent load) can still recover if
    given a genuinely long, observed window rather than a fixed guessed
    threshold. Fails fast if a container actually exits/crashes, though,
    rather than waiting out the full timeout for something that's dead.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            cwd=bench_dir,
            capture_output=True,
            text=True,
            timeout=20,
        )
        services = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        if not services:
            time.sleep(poll_interval)
            continue
        exited = [s["Service"] for s in services if s.get("State") == "exited"]
        if exited:
            raise RuntimeError(f"{bench_dir.name}: service(s) exited during startup: {exited}")
        if all(s.get("State") == "running" and s.get("Health", "") in ("healthy", "") for s in services):
            return
        time.sleep(poll_interval)
    raise RuntimeError(f"{bench_dir.name} did not become healthy within {timeout:.0f}s")


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

    Queries docker ps directly instead of relying on _active_benchmarks to
    avoid a race condition where threads start containers after the set is cleared.
    """
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    running_names = result.stdout.strip().splitlines()

    # Collect every benchmark_id whose containers are still up
    to_stop: set[str] = set(_active_benchmarks)
    for name in running_names:
        # Container names look like "xben-054-24-web-1" — reconstruct benchmark_id
        m = re.match(r"(xben-\d+-\d+)-", name)
        if m:
            bench_id = m.group(1).upper()  # "XBEN-054-24"
            to_stop.add(bench_id)

    for benchmark_id in to_stop:
        bench_dir = benchmarks_dir / benchmark_id
        if not bench_dir.is_dir():
            continue
        try:
            subprocess.Popen(
                ["docker", "compose", "down", "--remove-orphans"],
                cwd=bench_dir,
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
