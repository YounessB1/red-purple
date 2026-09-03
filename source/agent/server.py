"""Red-Purple — agent server."""

import asyncio
import json
import threading
import traceback
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from source.agent.runner import run

app = FastAPI()

# Per-run cancel events so one cancellation never bleeds into another run.
_active_run_events: dict[str, threading.Event] = {}
_runs_lock = threading.Lock()


def _rewrite_localhost(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


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
async def run_endpoint(target: str, seed_json: str = "", expected_flag: str = "") -> dict:
    target = _rewrite_localhost(target)
    run_id = f"run-{uuid4().hex[:8]}"
    candidate = json.loads(seed_json) if seed_json else {}

    cancel_event = threading.Event()
    with _runs_lock:
        _active_run_events[run_id] = cancel_event

    loop = asyncio.get_event_loop()
    try:
        metadata, context_window = await loop.run_in_executor(
            None, lambda: run(
                target=target, run_id=run_id, candidate=candidate,
                cancel_event=cancel_event, expected_flag=expected_flag or None,
            )
        )
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        with _runs_lock:
            _active_run_events.pop(run_id, None)

    return {"metadata": metadata, "context_window": context_window}
