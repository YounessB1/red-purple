"""Smoke test — start 3 benchmarks in parallel, verify reachable, then stop all."""

from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from source.benchmark import start_benchmark, stop_benchmark

BENCHMARKS = ["XBEN-001-24", "XBEN-002-24", "XBEN-003-24"]


def run_one(bench_id: str) -> str:
    """Start benchmark, hit it, stop it. Returns a status string."""
    port = start_benchmark(bench_id)
    print(f"  [{bench_id}] up on port {port}")

    try:
        r = httpx.get(f"http://localhost:{port}/", timeout=15)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        print(f"  [{bench_id}] reachable — HTTP {r.status_code}")
        return f"{bench_id}: OK (port {port})"
    finally:
        stop_benchmark(bench_id)
        print(f"  [{bench_id}] stopped")


def main():
    print(f"Starting {len(BENCHMARKS)} benchmarks in parallel...\n")

    with ThreadPoolExecutor(max_workers=len(BENCHMARKS)) as pool:
        futures = {pool.submit(run_one, bid): bid for bid in BENCHMARKS}

        for fut in as_completed(futures):
            bid = futures[fut]
            try:
                result = fut.result()
                print(f"  PASS  {result}")
            except Exception as e:
                print(f"  FAIL  {bid}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
