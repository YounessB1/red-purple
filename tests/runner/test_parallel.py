"""Run the first 3 benchmarks in parallel and save results to tests/runner/output/<bench_id>/."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from source.benchmark import start_benchmark, stop_benchmark

BENCHMARKS = ["XBEN-001-24", "XBEN-002-24", "XBEN-003-24"]
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


def run_one(bench_id: str, candidate: dict) -> str:
    port = start_benchmark(bench_id)
    target = f"http://localhost:{port}"
    print(f"[{bench_id}] up on port {port}", flush=True)

    try:
        resp = httpx.post(
            f"{AGENT_SERVER}/run",
            params={"target": target, "seed_json": json.dumps(candidate)},
            timeout=None,
        )
        resp.raise_for_status()
        result = resp.json()

        out_dir = _OUT_DIR / bench_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(
            json.dumps(result.get("metadata", {}), indent=2), encoding="utf-8"
        )
        (out_dir / "context_window.json").write_text(
            json.dumps(result.get("context_window", []), indent=2), encoding="utf-8"
        )

        meta = result.get("metadata", {})
        status = f"{'FLAG ' + meta['flag'] if meta.get('flag') else 'no flag'} | {meta.get('stop_reason')} | {meta.get('tool_calls')} tools | ${meta.get('total_cost_usd', 0):.4f}"
        print(f"[{bench_id}] done → {status}", flush=True)
        return f"{bench_id}: {status}"

    finally:
        stop_benchmark(bench_id)
        print(f"[{bench_id}] stopped", flush=True)


def main() -> None:
    candidate = _build_candidate()
    print(f"Candidate files: {list(candidate['files'].keys())}")
    print(f"Running {len(BENCHMARKS)} benchmarks in parallel...\n")

    with ThreadPoolExecutor(max_workers=len(BENCHMARKS)) as pool:
        futures = {pool.submit(run_one, bid, candidate): bid for bid in BENCHMARKS}
        for fut in as_completed(futures):
            bid = futures[fut]
            try:
                print(f"  PASS  {fut.result()}")
            except Exception as e:
                print(f"  FAIL  {bid}: {e}")

    print(f"\nResults saved to {_OUT_DIR}/")


if __name__ == "__main__":
    main()
