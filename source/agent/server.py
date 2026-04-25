"""Red-Purple — agent server. Spawns an agent process per run and returns artifacts when done."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from fastapi import FastAPI

app = FastAPI()

_RUNS_DIR = Path("/app/runs")


def _rewrite_localhost(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


@app.post("/run")
async def run_agent(target: str, max_iter: int = 100, seed_json: str = "", model: str = "") -> dict:
    run_id = f"run-{uuid4().hex[:8]}"
    target = _rewrite_localhost(target)

    env = {**os.environ, "TARGET": target, "MAX_ITER": str(max_iter), "RUN_ID": run_id}
    if seed_json:
        env["SEED_OVERRIDE"] = seed_json
    if model:
        env["MODEL"] = model

    proc = await asyncio.create_subprocess_exec(
        "python3", "-m", "source.agent",
        env=env,
        cwd="/app",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()

    run_dir = _RUNS_DIR / run_id
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        context_window = json.loads((run_dir / "context_window.json").read_text())
    except FileNotFoundError:
        metadata = {
            "run_id": run_id,
            "success": False,
            "flag": None,
            "stop_reason": "error",
            "iterations_used": 0,
            "_exit_code": proc.returncode,
            "_server_logs": stdout.decode(errors="replace")[-3000:],
        }
        context_window = []

    return {"metadata": metadata, "context_window": context_window}
