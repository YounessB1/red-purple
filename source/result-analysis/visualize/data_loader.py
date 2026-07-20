"""Load a GEPA experiment directory into the JSON structure consumed by the HTML viewer."""
import json
from pathlib import Path

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from bench_categories import load_bench_categories
from embeddings import compute_similarity, compute_similarity_legacy


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

    return {"name": exp_dir.name, "iterations": iterations, "config": config, "summary": summary,
            "bench_categories": load_bench_categories(),
            "similarity": compute_similarity(iterations, exp_dir),
            "similarity_legacy": compute_similarity_legacy(iterations, exp_dir)}
