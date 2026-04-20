#!/usr/bin/env python3
"""Red-Purple — host-side CLI. Ensures the agent server is running, then triggers a scan."""

import argparse
import json
import pathlib
import subprocess
import sys
import time

_COMPOSE_FILE = pathlib.Path(__file__).resolve().parent / "docker-compose.yml"
_RUNS_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs"

import httpx
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = "http://localhost:8000"


def ensure_server() -> None:
    subprocess.run(["docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d"], check=True)
    for _ in range(30):
        try:
            httpx.get(f"{SERVER_URL}/runs", timeout=1)
            return
        except Exception:
            time.sleep(1)
    print("ERROR: agent server did not start in time.")
    sys.exit(1)


def save_artifacts(run_id: str) -> None:
    r = httpx.get(f"{SERVER_URL}/runs/{run_id}/artifacts", timeout=30)
    r.raise_for_status()
    artifacts = r.json()

    run_dir = _RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(artifacts["metadata"], indent=2))
    (run_dir / "context_window.json").write_text(json.dumps(artifacts["context_window"], indent=2))
    print(f"\n[red-purple] artifacts saved → {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="red-purple")
    parser.add_argument("-t", "--target", required=True, help="Target URL to scan")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--task", default="")
    args = parser.parse_args()

    ensure_server()

    r = httpx.post(
        f"{SERVER_URL}/run",
        params={"target": args.target, "max_iter": args.max_iter, "task": args.task},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    run_id = data["run_id"]
    print(f"[red-purple] run {run_id} started → {data['target']}\n")

    with httpx.stream("GET", f"{SERVER_URL}/runs/{run_id}/logs", timeout=None) as r:
        for chunk in r.iter_text():
            print(chunk, end="", flush=True)

    save_artifacts(run_id)


if __name__ == "__main__":
    main()
