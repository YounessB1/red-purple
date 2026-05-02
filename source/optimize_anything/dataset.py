"""Load dataset splits into GEPA-compatible format."""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_PATH = _REPO_ROOT / "source" / "dataset" / "splits.json"
BENCHMARKS_DIR = _REPO_ROOT / "xbow" / "benchmarks"


def load_dataset(
    splits_path: Path = SPLITS_PATH,
    benchmarks_dir: Path = BENCHMARKS_DIR,
) -> tuple[list[dict], list[dict]]:
    """Load splits.json and return (trainset, valset) as lists of example dicts."""
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train = [_make_example(bid, benchmarks_dir, "train") for bid in splits["train"]]
    val = [_make_example(bid, benchmarks_dir, "val") for bid in splits["val"]]
    return train, val


def _make_example(benchmark_id: str, benchmarks_dir: Path, split: str) -> dict:
    """Create a single dataset example from benchmark metadata."""
    meta_path = benchmarks_dir / benchmark_id / "benchmark.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "benchmark_id": benchmark_id,
        "split": split,
        "level": str(meta["level"]),
        "tags": meta.get("tags", []),
        "name": meta.get("name", benchmark_id),
    }
