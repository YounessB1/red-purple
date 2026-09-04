"""Load a GEPA experiment directory into the JSON structure consumed by the HTML viewer."""
import json
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from bench_categories import load_bench_categories
from embeddings import compute_similarity, compute_similarity_legacy

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_DIR = _REPO_ROOT / "source" / "dataset"
_BASELINE_ROOT = _REPO_ROOT / "experiments" / "baseline"


def _load_baseline_test_aggregate(config: dict, test_ids: set[str]) -> dict:
    """Aggregate the hand-authored baseline agent's own (already-run, 104-benchmark)
    results down to just this experiment's held-out test split, so it's directly
    comparable to the seed/best numbers in test/summary.json (same benchmark set).

    The baseline harness (source/baseline/evaluate.py) scores by exact flag match
    only — no LLM judge — so there's no partial-credit "avg_score" to report;
    pass_rate stands in for it.
    """
    if not test_ids or not _BASELINE_ROOT.exists():
        return {}

    agent_model = ((config.get("ctf_agent") or {}).get("md") or {}).get("model", "")
    model_short = agent_model.split("/")[-1] if agent_model else None
    candidates = []
    if model_short:
        candidates.append(_BASELINE_ROOT / model_short / "summary.json")
    for d in sorted(_BASELINE_ROOT.iterdir()) if _BASELINE_ROOT.is_dir() else []:
        if d.is_dir():
            candidates.append(d / "summary.json")

    baseline_path = next((p for p in candidates if p.exists()), None)
    if baseline_path is None:
        return {}

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    per_bench = [r for r in baseline.get("results", []) if r.get("benchmark_id") in test_ids]
    if not per_bench:
        return {}

    splits_name = config.get("splits")
    splits_meta = {}
    if splits_name:
        meta_path = _DATASET_DIR / f"{splits_name}.json"
        if meta_path.exists():
            splits_meta = json.loads(meta_path.read_text(encoding="utf-8"))["_meta"]["assignments"]

    n = len(per_bench)
    solved = sum(1 for r in per_bench if r.get("success"))
    by_level: dict[str, dict] = {}
    by_family: dict[str, dict] = {}
    for r in per_bench:
        lvl = str(r.get("level"))
        by_level.setdefault(lvl, {"n": 0, "solved": 0})
        by_level[lvl]["n"] += 1
        by_level[lvl]["solved"] += int(bool(r.get("success")))
        fam = splits_meta.get(r["benchmark_id"], {}).get("family") or (r.get("tags") or ["unknown"])[0]
        by_family.setdefault(fam, {"n": 0, "solved": 0})
        by_family[fam]["n"] += 1
        by_family[fam]["solved"] += int(bool(r.get("success")))

    pass_rate = round(solved / n, 4) if n else 0.0
    return {
        "source": str(baseline_path.relative_to(_REPO_ROOT)),
        "n": n,
        "solved": solved,
        "pass_rate": pass_rate,
        "avg_score": pass_rate,
        "by_level": by_level,
        "by_family": by_family,
        "results": {r["benchmark_id"]: bool(r.get("success")) for r in per_bench},
    }


def load_experiment(exp_dir: Path) -> dict:
    iterations = []
    for iter_dir in sorted(exp_dir.glob("iteration_*")):
        num = int(iter_dir.name.split("_")[1])

        if (iter_dir / "ACCEPTED").exists():
            status = "accepted"
        elif (iter_dir / "REJECTED").exists():
            status = "rejected"
        else:
            status = "pending"

        parent_files  = {}
        child_files   = {}
        parent_train  = {}
        parent_val    = {}
        child_train   = {}
        child_val     = None
        parent_idx    = None
        operation     = None
        parents       = []
        evo_path = iter_dir / "evolution.json"
        if evo_path.exists():
            evo = json.loads(evo_path.read_text(encoding="utf-8"))
            parent_train  = evo.get("parent", {}).get("train") or {}
            parent_val    = evo.get("parent", {}).get("val")   or {}
            child_train   = evo.get("child",  {}).get("train") or {}
            child_val     = evo.get("child",  {}).get("val")
            parent_idx       = evo.get("parent", {}).get("candidate_idx")
            child_candidate_idx = evo.get("child", {}).get("candidate_idx")
            operation        = evo.get("operation")
            parents          = evo.get("parents", [])

        for snap_name, target in [("parent", parent_files), ("child", child_files)]:
            snap_dir = iter_dir / snap_name
            if snap_dir.exists():
                for f in sorted(snap_dir.rglob("*")):
                    if f.is_file():
                        rel = str(f.relative_to(snap_dir))
                        try:
                            target[rel] = f.read_text(encoding="utf-8")
                        except Exception:
                            pass

        val_ok    = 0
        val_total = 0
        pool_path = iter_dir / "pool.json"
        if pool_path.exists():
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            candidates = pool.get("candidates", [])
            if candidates:
                best = max(candidates, key=lambda c: c.get("val_avg") or 0)
                val_dict  = best.get("val", {})
                val_ok    = sum(1 for v in val_dict.values() if v == 1.0)
                val_total = len(val_dict)

        changes = []
        changes_summary = []
        rc_path = iter_dir / "reflector_changes.json"
        if rc_path.exists():
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            raw = rc.get("changes", [])
            changes = raw if isinstance(raw, list) else [l.strip()[2:].strip() for l in raw.splitlines() if l.strip().startswith("- ")]
            changes_summary = rc.get("changes_summary", [])

        pool_json = ""
        if pool_path.exists():
            pool_display = json.loads(pool_path.read_text(encoding="utf-8"))
            for c in pool_display.get("candidates", []):
                c.pop("files", None)
            pool_json = json.dumps(pool_display, indent=2)

        reflector_json = ""
        refl_path = iter_dir / "reflector.json"
        if refl_path.exists():
            reflector_json = json.dumps(
                json.loads(refl_path.read_text(encoding="utf-8")), indent=2
            )

        reflector_steps = ""
        steps_path = iter_dir / "agentic_reflector_steps.json"
        if steps_path.exists():
            reflector_steps = steps_path.read_text(encoding="utf-8")

        iterations.append({
            "id":               num,
            "status":           status,
            "parent_files":     parent_files,
            "child_files":      child_files,
            "changes":          changes,
            "changes_summary":  changes_summary,
            "val_ok":           val_ok,
            "val_total":        val_total,
            "parent_train":     parent_train,
            "parent_val":       parent_val,
            "child_train":      child_train,
            "child_val":        child_val,
            "parent_idx":          parent_idx,
            "child_candidate_idx": child_candidate_idx,
            "operation":           operation,
            "parents":             parents,
            "pool_json":        pool_json,
            "reflector_json":   reflector_json,
            "reflector_steps":  reflector_steps,
        })

    config = {}
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        config_path = exp_dir / "config.yaml"
    if config_path.exists():
        raw = config_path.read_text(encoding="utf-8")
        try:
            config = json.loads(raw)
        except json.JSONDecodeError:
            if _yaml is not None:
                config = _yaml.safe_load(raw) or {}
            else:
                config = {}

    summary = {}
    summary_path = exp_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    test_summary = {}
    test_summary_path = exp_dir / "test" / "summary.json"
    if test_summary_path.exists():
        test_summary = json.loads(test_summary_path.read_text(encoding="utf-8"))
        diff = test_summary.get("diff", {})
        test_ids = {
            bid
            for group in ("solved_by_both", "solved_by_seed_only", "solved_by_best_only", "solved_by_neither")
            for bid in diff.get(group, [])
        }
        baseline_agg = _load_baseline_test_aggregate(config, test_ids)
        if baseline_agg:
            test_summary = {**test_summary, "baseline": baseline_agg}

    return {"name": exp_dir.name, "iterations": iterations, "config": config, "summary": summary,
            "test_summary": test_summary,
            "bench_categories": load_bench_categories(),
            "similarity": compute_similarity(iterations, exp_dir),
            "similarity_legacy": compute_similarity_legacy(iterations, exp_dir)}
