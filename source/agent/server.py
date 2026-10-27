"""Red-Purple — agent server."""

import asyncio
import json
import os
import threading
import traceback
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request

from source.agent.runner import run, sweep_orphan_workdirs

# Per-run cancel events so one cancellation never bleeds into another run.
_active_run_events: dict[str, threading.Event] = {}
_runs_lock = threading.Lock()

# Background GC for orphaned per-run workdirs. A workdir is cleaned by the run's
# own `finally` on any normal return; this reclaims the ones left behind when the
# process is hard-killed (docker restart/stop, OOM, crash) — which survive a
# `restart` because it reuses the same writable layer. See sweep_orphan_workdirs.
_GC_ENABLED = os.environ.get("AGENT_GC_ENABLED", "1") not in ("0", "false", "False", "")
_GC_INTERVAL_SEC = int(os.environ.get("AGENT_GC_INTERVAL_SEC", "600"))
_GC_GRACE_SEC = int(os.environ.get("AGENT_GC_GRACE_SEC", "3600"))
_gc_stop = threading.Event()


def _gc_loop() -> None:
    # First sweep runs almost immediately so prior-lifetime orphans are reclaimed
    # shortly after startup, then every interval until shutdown.
    while not _gc_stop.is_set():
        try:
            with _runs_lock:
                active = set(_active_run_events)
            stats = sweep_orphan_workdirs(active, _GC_GRACE_SEC)
            if stats["removed"]:
                print(
                    f"[red-purple] workdir-gc removed {stats['removed']} orphan(s), "
                    f"freed {stats['freed_bytes'] / 1e6:.1f} MB (kept {stats['kept']})",
                    flush=True,
                )
        except Exception as e:  # a GC failure must never take the server down
            print(f"[red-purple] workdir-gc error: {type(e).__name__}: {e}", flush=True)
        _gc_stop.wait(_GC_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    gc_thread: threading.Thread | None = None
    if _GC_ENABLED:
        gc_thread = threading.Thread(target=_gc_loop, name="workdir-gc", daemon=True)
        gc_thread.start()
        print(
            f"[red-purple] workdir-gc started (interval={_GC_INTERVAL_SEC}s, "
            f"grace={_GC_GRACE_SEC}s)",
            flush=True,
        )
    try:
        yield
    finally:
        _gc_stop.set()
        if gc_thread is not None:
            gc_thread.join(timeout=5)


app = FastAPI(lifespan=lifespan)


def _rewrite_localhost(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


def _rewrite_host_token(token: str) -> str:
    """Rewrite a bare `host:port` token (no scheme) so the benchmark is
    reachable from inside the agent container, the same way `_rewrite_localhost`
    does for full URLs. Used for the extra, non-primary services of a
    multi-port benchmark, which are passed as `host:port` (protocol unknown —
    e.g. SSH), not as HTTP URLs."""
    host, sep, port = token.rpartition(":")
    if sep and host in ("localhost", "127.0.0.1", "::1"):
        return f"host.docker.internal{sep}{port}"
    return token


@app.post("/reset")
async def reset_endpoint() -> dict:
    with _runs_lock:
        for ev in _active_run_events.values():
            ev.set()
        _active_run_events.clear()
    return {"status": "reset"}


@app.post("/cancel")
async def cancel_endpoint() -> dict:
    with _runs_lock:
        for ev in _active_run_events.values():
            ev.set()
    return {"status": "cancelling"}


@app.post("/run")
async def run_endpoint(
    request: Request,
    target: str,
    expected_flag: str = "",
    extra: list[str] = Query(default=[]),
) -> dict:
    # The candidate's file contents (seed_json) arrive as the raw POST body,
    # not a query parameter: they routinely exceed a request line's practical
    # size once an agent has more than a couple of skills, which uvicorn/h11
    # rejects outright ("Invalid HTTP request received.") rather than
    # reporting a clear size error — keeping it in the body avoids that
    # limit entirely. `target`/`expected_flag` stay as query params since
    # they're always short.
    target = _rewrite_localhost(target)
    extra_targets = [_rewrite_host_token(t) for t in extra]
    run_id = f"run-{uuid4().hex[:8]}"
    body = await request.body()
    candidate = json.loads(body) if body else {}

    cancel_event = threading.Event()
    with _runs_lock:
        _active_run_events[run_id] = cancel_event

    loop = asyncio.get_event_loop()
    try:
        metadata, context_window = await loop.run_in_executor(
            None, lambda: run(
                target=target, run_id=run_id, candidate=candidate,
                cancel_event=cancel_event, expected_flag=expected_flag or None,
                extra_targets=extra_targets,
            )
        )
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        with _runs_lock:
            _active_run_events.pop(run_id, None)

    return {"metadata": metadata, "context_window": context_window}
