"""Benchmark lifecycle management — start, stop, port discovery for XBOW and nyu_ctf challenges."""

import json
import random
import re
import subprocess
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"
NYU_BENCHMARKS_DIR = _REPO_ROOT / "nyu_ctf" / "benchmarks"

# Matches "0.0.0.0:35167->80/tcp" — captures (host port, container port)
_HOST_PORT_RE = re.compile(r"0\.0\.0\.0:(\d+)->(\d+)/tcp")

# A handful of benchmarks expose more than one host-mapped port on the same
# container (e.g. a web app + SSH, or two cooperating services) — when that
# happens, prefer whichever mapping is a well-known web port instead of just
# taking the first line `docker ps` prints (container ports are listed in
# ascending order, so e.g. port 22/SSH would otherwise be picked over 80).
_WEB_PORTS = {80, 443, 8080, 8000, 3000, 5000}

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


def get_flag(benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR) -> str:
    """Return the benchmark's actual expected flag via `make print-flag`.

    Reads it straight from the same Make variable `make build` uses to build
    the image (xbow/common.mk), rather than recomputing the hash formula
    ourselves — a benchmark can override FLAG (env var, or a future
    non-default Makefile), and only Make's own resolution is guaranteed to
    match what actually got baked into the container.
    """
    bench_dir = benchmarks_dir / benchmark_id
    result = subprocess.run(
        ["make", "print-flag"],
        cwd=bench_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def start_benchmark(
    benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR, project_suffix: str = ""
) -> tuple[int, list[int], str]:
    """Start a benchmark's containers; return (primary port, extra ports, flag).

    `extra_host_ports` is every *other* host-published port besides the
    primary one (empty for the common single-service benchmark). A few
    benchmarks publish a second service the flag actually lives behind — SSH
    in XBEN-042-24, an unauthenticated S3 store in XBEN-084-24 — on its own
    random host port; returning those here lets the caller expose them to the
    agent instead of silently dropping everything but the web port.

    Builds, starts detached, then polls real container health ourselves
    (see module-level comment above) instead of trusting `docker compose up
    --wait`'s brittle single-shot verdict. Retries the whole cycle on
    failure — concurrent workers routinely contend for host resources
    during container boot.

    `project_suffix`, when non-empty, isolates this run's containers/network
    under their own `docker compose -p <benchmark_id>-lower-<suffix>` project
    instead of the directory-derived default — this is what makes it safe to
    run several concurrent instances of the *same* benchmark_id (e.g. N
    replicate runs for variance measurement), which would otherwise collide
    on container/network names. Empty (the default) reproduces the exact
    project name `docker compose` would already pick on its own, so every
    existing caller is unaffected.

    Docker Compose ties a built image to its project name (`<project>-
    <service>`), so a suffixed project can't reuse the image `make build`
    already built and correctly tagged under the *default* project name —
    left alone, the later `docker compose -p <project_name> up -d` would
    silently rebuild from scratch for that project, and that rebuild doesn't
    get `make build`'s explicit `--build-arg FLAG=...`, so it falls back to
    docker-compose.yml's own build-arg resolution (environment / a
    benchmark's `.env` file) — which is very often a stale placeholder, not
    the real flag `get_flag` computes. Confirmed on XBEN-022-24: every
    suffixed replicate ended up with a decoy flag baked into `/flag` instead
    of the real one, so an agent that genuinely solved the challenge and
    `cat`'d that file still failed scoring, having faithfully reported a
    real (but environment-corrupted) result. When `project_suffix` is set,
    `make build` still runs exactly once, the same way every other caller
    (e.g. the GEPA optimizer) already builds this benchmark — the *only*
    difference is that its resulting images are then `docker tag`'d under
    each suffixed project's expected name before `up -d`, so every replicate
    runs the literal same image bytes as the default build, never
    triggering Compose's own (differently-arg'd) build path at all.
    """
    bench_dir = benchmarks_dir / benchmark_id
    default_project_name = benchmark_id.lower()
    project_name = f"{default_project_name}{project_suffix}"
    expected_flag = get_flag(benchmark_id, benchmarks_dir)
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
            if project_suffix:
                services = subprocess.run(
                    ["docker", "compose", "config", "--services"],
                    cwd=bench_dir, check=True, capture_output=True, text=True, timeout=30,
                ).stdout.split()
                for service in services:
                    src = f"{default_project_name}-{service}:latest"
                    # Only services with a `build:` section produce a
                    # project-named image for `make build` to create and for us
                    # to retag. Pull-only services (e.g. `image: mysql:5.7`,
                    # `image: mongo:latest`) have no `<project>-<service>` image
                    # at all — compose reuses the upstream image directly under
                    # any project name — so there is nothing to retag, and
                    # `docker tag` would fail with "No such image". Skip those.
                    if subprocess.run(
                        ["docker", "image", "inspect", src],
                        capture_output=True, text=True,
                    ).returncode != 0:
                        continue
                    subprocess.run(
                        [
                            "docker", "tag",
                            src,
                            f"{project_name}-{service}:latest",
                        ],
                        check=True, capture_output=True, text=True, timeout=30,
                    )
            # Held across up+health-poll, not just the `up -d` dispatch —
            # `up -d` returns almost instantly, so only throttling the
            # dispatch wouldn't limit how many containers are actually
            # mid-boot at once, which is the thing causing contention.
            with _start_semaphore:
                try:
                    subprocess.run(
                        ["docker", "compose", "-p", project_name, "up", "-d"],
                        cwd=bench_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    stderr = getattr(e, "stderr", None) or ""
                    unhealthy_dep = "is unhealthy" in stderr or "dependency failed to start" in stderr
                    # `up -d` has its own short, compose-internal wait for
                    # `depends_on: condition: service_healthy` (baked into
                    # each benchmark's docker-compose.yml healthcheck
                    # interval*retries — often ~50s) and hard-fails the
                    # instant that expires, even though the dependency
                    # container is still running and Docker keeps
                    # re-probing it afterward. Under load that same wait can
                    # instead run past `up -d`'s own 120s subprocess timeout
                    # (TimeoutExpired, no clean stderr) rather than failing
                    # fast with the message above — same underlying race,
                    # just resolved slower. Tearing down here (the generic
                    # retry path below) would throw away that already-
                    # booting container and restart its clock from zero —
                    # for a dependency that genuinely needs more than
                    # compose's window, every retry loses the same race
                    # either way. Poll real health ourselves instead, then
                    # let `up -d` resume in place once it's actually
                    # healthy; a genuinely dead/crashed container still
                    # surfaces via _wait_for_healthy's own failure modes.
                    if not (unhealthy_dep or isinstance(e, subprocess.TimeoutExpired)):
                        raise
                    _wait_for_healthy(bench_dir, project_name)
                    subprocess.run(
                        ["docker", "compose", "-p", project_name, "up", "-d"],
                        cwd=bench_dir,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                _wait_for_healthy(bench_dir, project_name)
            _active_benchmarks.add(benchmark_id)
            all_ports = find_host_ports(benchmark_id, project_suffix)
            primary_port = find_host_port(benchmark_id, project_suffix)
            extra_host_ports = [hp for hp, _cp in all_ports if hp != primary_port]
            return primary_port, extra_host_ports, expected_flag
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
                        ["docker", "compose", "-p", project_name, "down", "--remove-orphans"],
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
    project_name: str,
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
            ["docker", "compose", "-p", project_name, "ps", "--format", "json"],
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


def stop_benchmark(benchmark_id: str, benchmarks_dir: Path = BENCHMARKS_DIR, project_suffix: str = "") -> None:
    """Tear down a benchmark's Docker containers.

    `project_suffix` must match the value passed to the corresponding
    `start_benchmark` call — see its docstring.
    """
    _active_benchmarks.discard(benchmark_id)
    bench_dir = benchmarks_dir / benchmark_id
    default_project_name = benchmark_id.lower()
    project_name = f"{default_project_name}{project_suffix}"
    subprocess.run(
        ["docker", "compose", "-p", project_name, "down", "--remove-orphans"],
        cwd=bench_dir,
        capture_output=True,
        timeout=60,
    )
    if project_suffix:
        # start_benchmark only re-tagged the default build's images under
        # this project's name (see its docstring) — untag them here so
        # `docker images` doesn't accumulate one stray tag pair per
        # replicate run. Safe: it's just a tag reference, not the
        # underlying layers, which stay alive under the default tag.
        try:
            services = subprocess.run(
                ["docker", "compose", "config", "--services"],
                cwd=bench_dir, capture_output=True, text=True, timeout=30,
            ).stdout.split()
            for service in services:
                subprocess.run(
                    ["docker", "rmi", f"{project_name}-{service}:latest"],
                    capture_output=True, timeout=30,
                )
        except subprocess.TimeoutExpired:
            pass


def force_stop_all(benchmarks_dir: Path = BENCHMARKS_DIR) -> None:
    """Fire-and-forget teardown of every active benchmark.

    Spawns detached `docker compose down` processes that survive the Python
    process exiting. Used from the SIGINT handler to recover the terminal
    instantly without waiting for stuck worker threads.

    Queries docker ps directly instead of relying on _active_benchmarks to
    avoid a race condition where threads start containers after the set is cleared.

    Known gap: reconstructs only the bare benchmark_id from a container name,
    not its `project_suffix` (see start_benchmark), so `docker compose down`
    here always targets the *default* (unsuffixed) project. If several
    suffixed replicate runs of the same benchmark_id are active at SIGINT
    time, their containers won't be torn down by this path — acceptable for
    now since this is already documented as best-effort, but worth revisiting
    if suffixed runs become routine rather than a one-off.
    """
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    running_names = result.stdout.strip().splitlines()

    # Collect every benchmark_id whose containers are still up
    to_stop: set[str] = set(_active_benchmarks)
    for name in running_names:
        # Container names look like "xben-054-24-web-1" or "nyu-024-phil-nginx-1"
        # — reconstruct benchmark_id from either naming scheme.
        m = re.match(r"(xben-\d+-\d+|nyu-\d+)-", name)
        if m:
            bench_id = m.group(1).upper()  # "XBEN-054-24" or "NYU-024"
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


def find_host_ports(benchmark_id: str, project_suffix: str = "") -> list[tuple[int, int]]:
    """Return every host-mapped (host_port, container_port) pair for a benchmark.

    Filters containers by the exact compose project (the label `docker
    compose` stamps on every container it creates, `com.docker.compose.
    project`) rather than a substring of the benchmark name — a substring
    filter would also match other concurrent replicate runs of the same
    benchmark_id under a different `project_suffix` (see start_benchmark),
    and could hand this call another run's port. Each pair comes from a
    0.0.0.0:PORT->CONTAINER/tcp mapping (an externally accessible service,
    not internal-only containers). A handful of benchmarks publish more than
    one such port on the same container (e.g. a web app + SSH in XBEN-042-24,
    or a web app + S3 in XBEN-084-24) — all of them are returned here so the
    caller can hand the full attack surface to the agent, not just the one
    web port `find_host_port` picks.
    """
    project_name = f"{benchmark_id.lower()}{project_suffix}"
    result = subprocess.run(
        ["docker", "ps", "--filter", f"label=com.docker.compose.project={project_name}", "--format", "{{.Ports}}"],
        capture_output=True,
        text=True,
        check=True,
    )

    candidates: list[tuple[int, int]] = []  # (host_port, container_port)
    for line in result.stdout.strip().splitlines():
        for match in _HOST_PORT_RE.finditer(line):
            candidates.append((int(match.group(1)), int(match.group(2))))

    if not candidates:
        raise RuntimeError(
            f"No host-mapped port found for benchmark '{benchmark_id}' (project '{project_name}'). "
            f"docker ps output:\n{result.stdout}"
        )
    return candidates


def find_host_port(benchmark_id: str, project_suffix: str = "") -> int:
    """Find the single primary host-mapped port for a benchmark.

    Returns the host port of the externally accessible service. If more than
    one mapping exists, prefers a well-known web port (see _WEB_PORTS) over
    whatever `docker ps` happens to list first. Use `find_host_ports` when
    the non-primary ports matter too (multi-service benchmarks).
    """
    candidates = find_host_ports(benchmark_id, project_suffix)
    web_matches = [host_port for host_port, container_port in candidates if container_port in _WEB_PORTS]
    return web_matches[0] if web_matches else candidates[0][0]
