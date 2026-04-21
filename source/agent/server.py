"""Red-Purple — agent server. Runs persistently inside the container and spawns agent processes on demand."""

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

_runs: dict[str, dict] = {}
_RUNS_DIR = Path("/app/runs")


def _rewrite_localhost(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal so agents can reach the host."""
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = parsed.netloc.replace(parsed.hostname, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


@app.post("/run")
async def start_run(target: str, max_iter: int = 100, task: str = "", seed_json: str = "") -> dict:
    run_id = f"run-{uuid4().hex[:8]}"
    target = _rewrite_localhost(target)

    env = {
        **os.environ,
        "TARGET": target,
        "MAX_ITER": str(max_iter),
        "TASK": task,
        "RUN_ID": run_id,
    }
    if seed_json:
        env["SEED_OVERRIDE"] = seed_json

    proc = await asyncio.create_subprocess_exec(
        "python3", "-m", "source.agent",
        env=env,
        cwd="/app",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    _runs[run_id] = {"target": target, "status": "running", "logs": []}
    asyncio.create_task(_collect(run_id, proc))

    return {"run_id": run_id, "target": target}


async def _collect(run_id: str, proc: asyncio.subprocess.Process) -> None:
    """Read subprocess output line by line and store it."""
    async for line in proc.stdout:
        _runs[run_id]["logs"].append(line.decode(errors="replace"))
    await proc.wait()
    _runs[run_id]["status"] = "done" if proc.returncode == 0 else "failed"
    _runs[run_id]["exit_code"] = proc.returncode


@app.get("/runs")
def list_runs() -> dict:
    return {
        rid: {k: v for k, v in info.items() if k != "logs"}
        for rid, info in _runs.items()
    }


@app.get("/runs/{run_id}/logs")
async def stream_logs(run_id: str) -> StreamingResponse:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = _runs[run_id]

    async def generate():
        idx = 0
        while True:
            while idx < len(run["logs"]):
                yield run["logs"][idx]
                idx += 1
            if run["status"] != "running":
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/runs/{run_id}/artifacts")
def get_artifacts(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    if _runs[run_id]["status"] == "running":
        raise HTTPException(status_code=409, detail="Run still in progress")
    run_dir = _RUNS_DIR / run_id
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        context_window = json.loads((run_dir / "context_window.json").read_text())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifacts not found")
    return {"metadata": metadata, "context_window": context_window}
