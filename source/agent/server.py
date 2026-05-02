"""Red-Purple — agent server."""

import asyncio
import json
import threading
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import traceback

from fastapi import FastAPI, HTTPException

from source.agent.runner import run

app = FastAPI()

_cancel_event = threading.Event()
_active_runs = 0
_runs_lock = threading.Lock()


def _rewrite_localhost(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


@app.post("/cancel")
async def cancel_endpoint() -> dict:
    _cancel_event.set()
    return {"status": "cancelling"}


@app.post("/run")
async def run_endpoint(target: str, max_iter: int = 100, seed_json: str = "", model: str = "") -> dict:
    global _active_runs
    target = _rewrite_localhost(target)
    run_id = f"run-{uuid4().hex[:8]}"
    prompt = json.loads(seed_json).get("prompt") if seed_json else None

    with _runs_lock:
        if _active_runs == 0:
            _cancel_event.clear()
        _active_runs += 1

    loop = asyncio.get_event_loop()
    try:
        metadata, context_window = await loop.run_in_executor(
            None, lambda: run(target=target, run_id=run_id, max_iter=max_iter, model=model, prompt=prompt, cancel_event=_cancel_event)
        )
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    finally:
        with _runs_lock:
            _active_runs -= 1

    return {"metadata": metadata, "context_window": context_window}
