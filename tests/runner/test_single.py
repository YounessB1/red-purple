"""End-to-end test — start a benchmark, run the OpenCode agent against it, print result."""

import json
from pathlib import Path

import httpx
from source.benchmark import start_benchmark, stop_benchmark

BENCHMARK_ID = "XBEN-001-24"
AGENT_SERVER = "http://localhost:8000"

_ROOT = Path(__file__).resolve().parents[2]
_SEED_DIR = _ROOT / "source" / "seed"
_OUT_DIR = _ROOT / "tests" / "runner" / "output"


def _build_candidate() -> dict:
    files = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")
    return {"files": files}


def main() -> None:
    print(f"[test] Starting benchmark {BENCHMARK_ID}...")
    port = start_benchmark(BENCHMARK_ID)
    target = f"http://localhost:{port}"
    print(f"[test] Target: {target}")

    candidate = _build_candidate()
    print(f"[test] Candidate files: {list(candidate['files'].keys())}")

    try:
        print(f"[test] Sending run request to {AGENT_SERVER}")
        resp = httpx.post(
            f"{AGENT_SERVER}/run",
            params={
                "target": target,
                "seed_json": json.dumps(candidate),
            },
            timeout=None,
        )
        resp.raise_for_status()
        result = resp.json()

        metadata = result.get("metadata", {})
        context_window = result.get("context_window", [])

        out_dir = _OUT_DIR / BENCHMARK_ID
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out_dir / "context_window.json").write_text(json.dumps(context_window, indent=2), encoding="utf-8")
        print(f"[test] Saved to {out_dir}/")

    finally:
        stop_benchmark(BENCHMARK_ID)
        print("\n[test] Benchmark stopped.")


if __name__ == "__main__":
    main()
