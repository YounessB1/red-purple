#!/usr/bin/env python3
"""Generate a self-contained HTML prompt viewer for a red-purple experiment.
Usage: python visualize.py [experiment_dir]
Output: <experiment_dir>/viewer.html
"""
import json
import sys
from pathlib import Path
try:
    import yaml as _yaml
except ImportError:
    _yaml = None


_TAG_TO_CATEGORY: dict[str, str] = {
    "xss": "XSS", "ssti": "SSTI", "idor": "IDOR",
    "privilege_escalation": "Privilege Escalation", "command_injection": "Command Injection",
    "ssh": "Command Injection", "default_credentials": "Default Credentials",
    "sqli": "SQL Injection", "blind_sqli": "SQL Injection", "nosqli": "SQL Injection",
    "insecure_deserialization": "Deserialization", "lfi": "LFI",
    "path_traversal": "LFI", "xxe": "LFI", "business_logic": "Business Logic",
    "graphql": "Business Logic", "race_condition": "Business Logic",
    "brute_force": "Business Logic", "http_method_tamper": "Business Logic",
    "smuggling_desync": "Business Logic", "crypto": "Business Logic",
    "jwt": "Business Logic", "ssrf": "Business Logic",
    "information_disclosure": "Business Logic", "arbitrary_file_upload": "Business Logic",
    "cve": "Business Logic",
}
_CATEGORY_PRIORITY = [
    "XSS", "SSTI", "SQL Injection", "Command Injection",
    "Deserialization", "LFI", "IDOR", "Privilege Escalation",
    "Default Credentials", "Business Logic",
]


def _classify_tags(tags: list) -> str:
    cats = {_TAG_TO_CATEGORY[t] for t in tags if t in _TAG_TO_CATEGORY}
    if not cats:
        return "Business Logic"
    return min(cats, key=_CATEGORY_PRIORITY.index)


def _load_bench_categories() -> dict:
    benchmarks_dir = Path(__file__).resolve().parent / "xbow" / "benchmarks"
    result: dict[str, str] = {}
    if not benchmarks_dir.exists():
        return result
    for bench_dir in sorted(benchmarks_dir.iterdir()):
        bj = bench_dir / "benchmark.json"
        if not bj.exists():
            continue
        try:
            data = json.loads(bj.read_text(encoding="utf-8"))
            result[bench_dir.name] = _classify_tags(data.get("tags", []))
        except Exception:
            pass
    return result


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
            "bench_categories": _load_bench_categories()}


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__NAME__ — Prompt Explorer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #ffffff; color: #24292f;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}
#hdr {
  background: #f6f8fa; border-bottom: 1px solid #d0d7de;
  padding: 10px 18px; display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
#hdr h1 { font-size: 15px; font-weight: 600; color: #1a1a1a; }
#hdr .hint { font-size: 12px; color: #6e7781; }
#body { display: flex; flex: 1; overflow: hidden; }
#sidebar {
  width: 210px; flex-shrink: 0; background: #f6f8fa;
  border-right: 1px solid #d0d7de; overflow-y: auto; padding: 8px;
}
#main { flex: 1; overflow-y: auto; padding: 18px 24px; min-width: 200px; }
.resizer {
  width: 4px; flex-shrink: 0; cursor: col-resize;
  background: #d0d7de; transition: background 0.15s;
}
.resizer:hover, .resizer.dragging { background: #0969da; }
#info-sidebar {
  width: 310px; flex-shrink: 0; background: #f6f8fa;
  border-left: 1px solid #d0d7de; overflow-y: auto; padding: 12px;
}
.info-panel {
  background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px;
  margin-bottom: 12px; overflow: hidden;
}
.info-panel-hdr {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
  color: #57606a; padding: 7px 12px; background: #eaeef2;
  border-bottom: 1px solid #d0d7de;
}
.info-row {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 8px; padding: 5px 12px; border-bottom: 1px solid #f6f8fa;
  font-size: 12.5px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #57606a; white-space: nowrap; flex-shrink: 0; }
.info-val { color: #24292f; text-align: right; word-break: break-all; }
.info-val.mono { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11.5px; }
.info-sub-hdr {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  color: #9198a1; padding: 6px 12px 3px; background: #ffffff;
}
.info-cost { color: #9a6700; }
.info-calls { color: #0969da; }
.card {
  border: 1px solid #d0d7de; border-radius: 7px; padding: 9px 11px;
  margin-bottom: 5px; cursor: pointer;
  transition: border-color 0.12s, background 0.12s;
}
.card:hover { border-color: #0969da; background: #f6f8fa; }
.card.sel   { border-color: #0969da; background: #ddf4ff; box-shadow: 0 0 0 2px #0969da33; }
.card-top   { display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }
.card-num   { font-size: 12px; font-weight: 600; color: #1a1a1a; }
.badge {
  font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 9px;
  letter-spacing: 0.4px; text-transform: uppercase;
}
.b-accepted { background: #dafbe1; color: #1a7f37; border: 1px solid #2da44e; }
.b-rejected { background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; }
.b-pending  { background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; }
.card-score { font-size: 10px; color: #57606a; }
#placeholder {
  height: 100%; display: flex; align-items: center; justify-content: center;
  color: #9198a1; font-size: 14px;
}
.section-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
  color: #57606a; margin-bottom: 8px; margin-top: 22px;
}
.section-title:first-child { margin-top: 0; }

/* ── Tab bar ── */
.tab-bar {
  display: flex; gap: 0; margin-bottom: 20px;
  border-bottom: 1px solid #d0d7de;
}
.tab {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: #57606a; font-size: 13px; font-family: inherit;
  padding: 7px 18px; cursor: pointer; margin-bottom: -1px;
  transition: color 0.1s;
}
.tab:hover { color: #24292f; }
.tab.sel { color: #1a1a1a; border-bottom-color: #f06633; font-weight: 600; }
.json-view {
  background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px;
  padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px; line-height: 1.65; color: #24292f;
  white-space: pre-wrap; word-break: break-word;
}

.parent-tag {
  display: inline-block; font-size: 11px; color: #0969da;
  background: #ddf4ff; border: 1px solid #54aeff;
  border-radius: 4px; padding: 2px 9px;
  font-family: 'JetBrains Mono', Consolas, monospace;
}
.op-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.op-badge {
  font-size: 10px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px;
}
.op-tweak { background: #dafbe1; color: #1a7f37; border: 1px solid #2da44e; }
.op-merge { background: #fbefff; color: #8250df; border: 1px solid #bf8cf2; }
.op-seed  { background: #ddf4ff; color: #0969da; border: 1px solid #54aeff; }

/* ── Scores table ── */
.scores-wrap { display: flex; gap: 16px; margin-bottom: 0; }
.scores-panel {
  flex: 1; background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px;
  overflow: hidden; min-width: 0;
}
.scores-panel-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; background: #eaeef2; border-bottom: 1px solid #d0d7de;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; color: #57606a;
}
.scores-panel-hdr .s-summary { font-size: 10px; font-weight: 400; color: #6e7781; }
.scores-tbl { width: 100%; border-collapse: collapse; }
.scores-tbl th {
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
  color: #6e7781; padding: 4px 10px; text-align: center; border-bottom: 1px solid #d0d7de;
}
.scores-tbl th.bench-col { text-align: left; }
.scores-tbl td {
  padding: 3px 10px; font-size: 11.5px;
  font-family: 'JetBrains Mono', Consolas, monospace; border-bottom: 1px solid #f6f8fa;
}
.scores-tbl tr:last-child td { border-bottom: none; }
.scores-tbl td.bench-name { color: #57606a; }
.bench-cat {
  font-size: 10px; color: #6e7781; font-family: inherit;
  background: #eaeef2; border: 1px solid #d0d7de;
  border-radius: 3px; padding: 1px 5px; margin-left: 7px;
  font-weight: 500; letter-spacing: 0.2px; vertical-align: middle;
}
.s-pass  { color: #1a7f37; text-align: center; font-weight: 700; }
.s-mid   { color: #9a6700; text-align: center; }
.s-fail  { color: #cf222e; text-align: center; }
.s-none  { color: #9198a1; text-align: center; }
.s-delta-pos { color: #1a7f37; text-align: center; font-weight: 700; font-size: 13px; }
.s-delta-neg { color: #cf222e; text-align: center; font-size: 13px; }
.s-delta-eq  { color: #9198a1; text-align: center; }

/* ── Reflector changes ── */
.changes-box {
  background: #ffffff; border: 1px solid #d0d7de; border-left: 3px solid #9a6700;
  border-radius: 6px; padding: 14px 16px;
  font-size: 13px; line-height: 1.7; color: #24292f;
}
.changes-box ul { list-style: none; padding: 0; }
.changes-box li {
  padding: 5px 0 5px 16px; position: relative;
  border-bottom: 1px solid #eaeef2;
}
.changes-box li:last-child { border-bottom: none; }
.changes-box li::before { content: '•'; position: absolute; left: 0; color: #9a6700; }

/* ── Pool view ── */
.pool-header { font-size: 11px; color: #57606a; margin-bottom: 14px; }
.pool-wrap   { display: flex; flex-wrap: wrap; gap: 14px; }
.pool-card   {
  background: #ffffff; border: 1px solid #d0d7de; border-radius: 7px;
  overflow: hidden; flex: 1; min-width: 260px;
}
.pool-card-hdr {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 14px; background: #eaeef2; border-bottom: 1px solid #d0d7de;
}
.pool-card-title { font-size: 12px; font-weight: 600; color: #1a1a1a; }
.pool-avg   { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', Consolas, monospace; }
.pool-avg.g { color: #1a7f37; } .pool-avg.y { color: #9a6700; } .pool-avg.r { color: #cf222e; }
.pool-card-body { padding: 10px 14px; }
.pool-row {
  display: flex; align-items: baseline; gap: 8px;
  font-size: 11.5px; padding: 3px 0; border-bottom: 1px solid #eaeef2;
}
.pool-row:last-child { border-bottom: none; }
.pool-lbl  { color: #57606a; min-width: 56px; flex-shrink: 0; }
.pool-mval { color: #24292f; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11px; }
.pool-slbl {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.4px; color: #9198a1; padding: 8px 0 4px;
}
.pool-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.pool-chip  {
  font-family: 'JetBrains Mono', Consolas, monospace;
  font-size: 11px; padding: 2px 7px; border-radius: 4px; font-weight: 600;
}
.pc-pass { background: #dafbe1; color: #1a7f37; border: 1px solid #2da44e; }
.pc-mid  { background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; }
.pc-fail { background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; }
.pc-none { background: #f6f8fa; color: #57606a; border: 1px solid #d0d7de; }

/* ── Reflector view ── */
.refl-block {
  background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px;
  margin-bottom: 14px; overflow: hidden;
}
.refl-block-hdr {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
  color: #57606a; padding: 5px 14px; background: #eaeef2;
  border-bottom: 1px solid #d0d7de;
}
.refl-body {
  padding: 12px 14px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px; line-height: 1.65; color: #24292f;
  white-space: pre-wrap; word-break: break-word;
  max-height: 420px; overflow-y: auto;
}
.rh  { color: #0969da; font-weight: 700; }
.rf  { color: #9198a1; }

/* ── Unified diff ── */
.diff-meta {
  display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 12px;
}
.diff-meta .stat-del { color: #cf222e; }
.diff-meta .stat-ins { color: #1a7f37; }
.unified-diff {
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12.5px; line-height: 1.65;
  background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px;
  overflow: hidden;
}
.u-hunk {
  background: #ddf4ff; color: #0969da;
  padding: 2px 14px; font-size: 11px;
  border-top: 1px solid #d0d7de; border-bottom: 1px solid #d0d7de;
}
.u-hunk:first-child { border-top: none; }
.u-del  { background: #ffebe9; color: #cf222e; padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-ins  { background: #dafbe1; color: #1a7f37; padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-ctx  { background: #ffffff; color: #57606a;  padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-sign { display: inline-block; width: 14px; font-weight: 700; color: inherit; user-select: none; }
.no-diff {
  background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
  padding: 14px 16px; font-size: 13px; color: #9198a1; font-style: italic;
}

/* ── Files overview ── */
.files-overview {
  background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px;
  padding: 10px 14px; margin-bottom: 16px;
}
.file-row {
  display: flex; align-items: center; gap: 10px;
  padding: 3px 0; border-bottom: 1px solid #f6f8fa;
  font-size: 12px; font-family: 'JetBrains Mono', Consolas, monospace;
}
.file-row:last-child { border-bottom: none; }
.file-badge {
  font-size: 9px; font-weight: 700; padding: 1px 6px; border-radius: 4px;
  letter-spacing: 0.4px; text-transform: uppercase; white-space: nowrap;
  min-width: 68px; text-align: center;
}
.fb-modified  { background: #fff8c5; color: #9a6700; border: 1px solid #d4a72c; }
.fb-added     { background: #dafbe1; color: #1a7f37; border: 1px solid #2da44e; }
.fb-removed   { background: #ffebe9; color: #cf222e; border: 1px solid #ff8182; }
.fb-unchanged { background: #f6f8fa; color: #57606a; border: 1px solid #d0d7de; }
.file-name { color: #57606a; }
.file-summary { font-size: 11.5px; color: #6e7781; margin-left: auto; font-style: italic; }

/* ── Reasoning steps ── */
.rsn-feed { display: flex; flex-direction: column; gap: 6px; }
.rsn-step { border-radius: 6px; overflow: hidden; background: #ffffff; }
.rsn-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
  padding: 4px 12px;
}
.rsn-body {
  padding: 9px 14px; font-size: 12.5px; line-height: 1.7; color: #24292f;
  white-space: pre-wrap; word-break: break-word; background: #ffffff;
}

/* thinking — blue */
.rsn-step-thinking { border: 1px solid #0969da; }
.rsn-step-thinking .rsn-label { color: #0969da; border-bottom: 1px solid #0969da; }
.rsn-step-thinking .rsn-body strong { color: #0969da; }

/* text — green */
.rsn-step-text { border: 1px solid #2da44e; }
.rsn-step-text .rsn-label { color: #1a7f37; border-bottom: 1px solid #2da44e; }

/* tool — amber */
.rsn-step-tool { border: 1px solid #d4a72c; }
.rsn-step-tool .rsn-label { color: #9a6700; border-bottom: 1px solid #d4a72c; }
.rsn-step-tool .rsn-body  {
  font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px;
}
.rsn-tool-name { color: #9a6700; font-weight: 700; margin-right: 10px; }
.rsn-params { margin-top: 6px; display: flex; flex-direction: column; gap: 2px; }
.rsn-param-row { display: flex; gap: 8px; font-size: 11.5px; }
.rsn-param-key { color: #57606a; flex-shrink: 0; }
.rsn-param-val { color: #24292f; word-break: break-all; }

/* ── Evolution view ── */
#evo-tree-wrap { position: relative; overflow-x: auto; padding: 24px; }
#evo-svg { position: absolute; top: 0; left: 0; pointer-events: none; overflow: visible; }
.evo-levels { display: flex; flex-direction: column; gap: 56px; align-items: center; }
.evo-level  { display: flex; gap: 32px; align-items: flex-start; justify-content: center; }
.evo-candidate {
  border: 1px solid #d0d7de; border-radius: 8px;
  padding: 12px 16px; background: #fff;
}
.evo-candidate-hdr { font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 10px; }
.evo-candidate-sub { font-size: 10px; color: #57606a; font-weight: 400; margin-left: 6px; }
.evo-files { display: flex; flex-direction: column; gap: 5px; }
.evo-file-block {
  height: 32px; border-radius: 5px; border: 1px solid;
  display: inline-flex; align-items: center; padding: 0 8px;
  font-size: 10px; font-family: 'JetBrains Mono', Consolas, monospace;
  white-space: nowrap; cursor: default;
}
.evo-val-badge {
  display: inline-block; margin-left: 10px; padding: 2px 8px;
  border-radius: 10px; font-size: 11px; font-weight: 600;
  background: #ddf4ff; color: #0550ae; border: 1px solid #80ccff;
  vertical-align: middle;
}
</style>
</head>
<body>
<div id="hdr">
  <h1 id="exp-name"></h1>
  <span class="hint">Click an iteration to view reflector changes and prompt diff</span>
</div>
<div id="body">
  <div id="sidebar"></div>
  <div class="resizer" id="resizer-left"></div>
  <div id="main">
    <div id="placeholder">← Select an iteration</div>
    <div id="content" style="display:none"></div>
  </div>
  <div class="resizer" id="resizer-right"></div>
  <div id="info-sidebar"></div>
</div>
<script>
const DATA = __DATA__;
let selId  = null;
let selTab = 'diff';

document.getElementById('exp-name').textContent = DATA.name;

function shortModel(m) { return (m || '').replace('openrouter/', ''); }
function fmtDuration(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m ${sec}s` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtTokens(n) {
  if (!n) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(n);
}
function row(key, val, cls) {
  return `<div class="info-row"><span class="info-key">${key}</span><span class="info-val${cls ? ' ' + cls : ''}">${val}</span></div>`;
}

function renderInfoSidebar() {
  const cfg = DATA.config || {};
  const sum = DATA.summary || {};
  const el  = document.getElementById('info-sidebar');

  // ── Config panel ──
  let cfgRows = '';
  if (cfg.agent_model)          cfgRows += row('agent model',    esc(shortModel(cfg.agent_model)), 'mono');
  if (cfg.reflection_lm)        cfgRows += row('reflector',      esc(shortModel(cfg.reflection_lm)), 'mono');
  if (cfg.diagnoser_model)      cfgRows += row('diagnoser',      esc(shortModel(cfg.diagnoser_model)), 'mono');
  if (cfg.judge_model)          cfgRows += row('judge',          esc(shortModel(cfg.judge_model)), 'mono');
  if (cfg.max_calls != null)    cfgRows += row('budget',         `${cfg.max_calls} calls`);
  if (cfg.workers != null)      cfgRows += row('workers',        cfg.workers);
  if (cfg.agent_max_iter != null) cfgRows += row('agent max iter', cfg.agent_max_iter);
  if (cfg.train_minibatch_size != null) cfgRows += row('train batch', cfg.train_minibatch_size);
  cfgRows += row('val batch', cfg.val_minibatch_size != null ? cfg.val_minibatch_size : 'full');
  if (cfg.gt != null)           cfgRows += row('ground truth',   cfg.gt ? 'yes' : 'no');
  if (cfg.background_context)   cfgRows += row('context file',   esc(cfg.background_context), 'mono');

  // ── Summary panel ──
  let sumRows = '';
  if (sum.duration_seconds != null) sumRows += row('duration', fmtDuration(sum.duration_seconds));
  if (sum.total_cost_usd   != null) sumRows += row('total cost', `$${sum.total_cost_usd.toFixed(2)}`, 'info-cost');
  if (sum.total_tokens     != null) sumRows += row('total tokens', fmtTokens(sum.total_tokens));

  for (const key of ['agents', 'reflector', 'diagnoser', 'scorer']) {
    const s = sum[key];
    if (!s || s.calls === 0) continue;
    sumRows += `<div class="info-sub-hdr">${key} (${s.calls} calls)</div>`;
    if (s.model) sumRows += row('model', esc(shortModel(s.model)), 'mono');
    sumRows += row('cost', `$${s.cost_usd.toFixed(2)}`, 'info-cost');
    sumRows += row('tokens in/out', `${fmtTokens(s.input_tokens)} / ${fmtTokens(s.output_tokens)}`);
  }

  el.innerHTML =
    `<div class="info-panel"><div class="info-panel-hdr">Config</div>${cfgRows}</div>` +
    (sumRows ? `<div class="info-panel"><div class="info-panel-hdr">Summary</div>${sumRows}</div>` : '');
}

function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  const evoCard = document.createElement('div');
  evoCard.className = 'card' + (selId === 'evolution' ? ' sel' : '');
  evoCard.innerHTML = `<div class="card-top"><span class="card-num">Evolution</span></div>`;
  evoCard.addEventListener('click', () => { selId = 'evolution'; renderSidebar(); renderMain(); });
  sb.appendChild(evoCard);
  DATA.iterations.forEach(it => {
    const card = document.createElement('div');
    card.className = 'card' + (selId === it.id ? ' sel' : '');
    card.innerHTML =
      `<div class="card-top">
         <span class="card-num">Iter ${pad(it.id)}</span>
         <span class="badge b-${it.status}">${it.status}</span>
       </div>
       ${it.val_total ? `<div class="card-score">${it.val_ok}/${it.val_total} solved</div>` : ''}`;
    card.addEventListener('click', () => { selId = it.id; renderSidebar(); renderMain(); });
    sb.appendChild(card);
  });
}

function renderMain() {
  const ph = document.getElementById('placeholder');
  const ct = document.getElementById('content');
  if (selId === null) { ph.style.display = 'flex'; ct.style.display = 'none'; return; }
  ph.style.display = 'none'; ct.style.display = 'block';
  if (selId === 'evolution') { renderEvolutionView(ct); return; }
  renderIteration(ct, DATA.iterations.find(x => x.id === selId));
}

// ── Evolution view ────────────────────────────────────────────────────────
function fileColor(f) {
  if (f.startsWith('skills/')) return { bg: '#fff8c5', border: '#d4a72c', text: '#9a6700' };
  return { bg: '#ffebe9', border: '#cf222e', text: '#cf222e' };
}

function fileDisplayName(f) {
  if (f.startsWith('skills/')) {
    const parts = f.split('/');
    return parts.length >= 2 ? parts[1] : f;
  }
  return f.split('/').pop();
}

function buildEvolutionTree() {
  const nodes    = {};  // candidateIdx → { idx, iter, files }
  const parentOf = {};  // childIdx → parentIdx
  const children = {};  // parentIdx → [childIdx, ...]

  // Seed node (candidate 0)
  const seedIter = DATA.iterations[0];
  nodes[0] = { idx: 0, iter: 0, files: seedIter?.parent_files || {}, val_avg: null };

  for (const it of DATA.iterations) {
    if (it.status !== 'accepted') continue;
    if (!it.child_files || !Object.keys(it.child_files).length) continue;
    const newIdx = it.child_candidate_idx;
    if (newIdx === null || newIdx === undefined || newIdx === 0) continue;

    nodes[newIdx] = { idx: newIdx, iter: it.id, files: it.child_files, val_avg: null };
    if (it.parent_idx !== null && it.parent_idx !== undefined) {
      parentOf[newIdx] = it.parent_idx;
      if (!children[it.parent_idx]) children[it.parent_idx] = [];
      children[it.parent_idx].push(newIdx);
    }
  }

  // Fill val_avg from pool.json (for candidates that appear there)
  for (const it of DATA.iterations) {
    if (!it.pool_json) continue;
    const pool = JSON.parse(it.pool_json);
    for (const c of pool.candidates) {
      if (nodes[c.idx] && c.val_avg != null) nodes[c.idx].val_avg = c.val_avg;
    }
  }
  // Fill val_avg from child_val in evolution.json (for candidates absent from pool)
  for (const it of DATA.iterations) {
    if (it.status !== 'accepted') continue;
    const idx = it.child_candidate_idx;
    if (idx == null || !nodes[idx] || nodes[idx].val_avg != null) continue;
    const vals = it.child_val ? Object.values(it.child_val).filter(v => v != null) : [];
    if (vals.length) nodes[idx].val_avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  // Seed val_avg from first iteration's parent_val
  if (nodes[0] && nodes[0].val_avg == null) {
    const it0 = DATA.iterations[0];
    const vals = it0?.parent_val ? Object.values(it0.parent_val).filter(v => v != null) : [];
    if (vals.length) nodes[0].val_avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  return { nodes, parentOf, children };
}

function evoNodeHtml(idx, node) {
  const label = idx === 0 ? 'Seed' : `Cand #${idx}`;
  const sub   = node.iter > 0 ? `iter ${pad(node.iter)}` : '';
  const valBadge = node.val_avg != null
    ? `<span class="evo-val-badge">val avg ${node.val_avg.toFixed(2)}</span>`
    : '';
  const blocks = Object.keys(node.files).filter(f => !f.includes('ctf-agent.md')).sort().map(f => {
    const lines    = (node.files[f] || '').split('\n').length;
    const name     = fileDisplayName(f);
    const col      = fileColor(f);
    const isSkill  = f.startsWith('skills/');
    const radius   = isSkill ? '14px' : '5px';
    const extraPad = Math.min(60, Math.max(0, lines * 3 - 8));
    return `<div class="evo-file-block"
      style="border-radius:${radius};padding-right:${8 + extraPad}px;background:${col.bg};border-color:${col.border};color:${col.text}"
      title="${esc(f)} — ${lines} lines">${esc(name)}</div>`;
  }).join('');
  const clickAction = node.iter > 0
    ? `style="cursor:pointer;" onclick="selId=${JSON.stringify(node.iter)};renderSidebar();renderMain();"`
    : '';
  return `<div class="evo-candidate" data-evo-idx="${idx}" ${clickAction}>
    <div class="evo-candidate-hdr">${esc(label)}<span class="evo-candidate-sub">${esc(sub)}</span>${valBadge}</div>
    <div class="evo-files">${blocks}</div>
  </div>`;
}

function renderEvolutionView(el) {
  const { nodes, parentOf, children } = buildEvolutionTree();
  if (!Object.keys(nodes).length) {
    el.innerHTML = '<div class="no-diff">No candidates found.</div>';
    return;
  }

  // Compute horizontal offsets: spread each parent's children left/right of parent
  const xOffset = {};  // idx → x offset in px relative to center
  xOffset[0] = 0;
  const SPREAD = 240;
  const bfsQueue = [0];
  while (bfsQueue.length) {
    const pIdx   = bfsQueue.shift();
    const childs = children[pIdx] || [];
    const n      = childs.length;
    childs.forEach((cIdx, i) => {
      xOffset[cIdx] = xOffset[pIdx] + (i - (n - 1) / 2) * SPREAD;
      bfsQueue.push(cIdx);
    });
  }

  const STEP = 270;  // fixed vertical distance per candidate level
  let containerH = 0;
  let nodesHtml  = '';
  for (const [idxStr, node] of Object.entries(nodes)) {
    const idx = parseInt(idxStr);
    const top = idx * STEP;          // idx == acceptance order → fixed spacing
    const x   = xOffset[idx] ?? 0;
    containerH = Math.max(containerH, top + 220);
    nodesHtml += `<div style="position:absolute;top:${top}px;left:50%;transform:translateX(calc(-50% + ${x}px));">
      ${evoNodeHtml(idx, node)}
    </div>`;
  }

  el.innerHTML = `<div id="evo-tree-wrap" style="position:relative;min-height:${containerH}px;overflow:auto;padding:16px 0;">
    <svg id="evo-svg" style="position:absolute;top:0;left:0;pointer-events:none;"></svg>
    ${nodesHtml}
  </div>`;

  requestAnimationFrame(() => {
    const wrap = document.getElementById('evo-tree-wrap');
    const svg  = document.getElementById('evo-svg');
    if (!wrap || !svg) return;
    const W = wrap.scrollWidth;
    const H = wrap.scrollHeight;
    svg.style.width  = W + 'px';
    svg.style.height = H + 'px';
    const wRect = wrap.getBoundingClientRect();

    // Collect edge data
    const edgeData = [];
    for (const [childStr, parentIdx] of Object.entries(parentOf)) {
      const childIdx = parseInt(childStr);
      const pEl = wrap.querySelector(`[data-evo-idx="${parentIdx}"]`);
      const cEl = wrap.querySelector(`[data-evo-idx="${childIdx}"]`);
      if (!pEl || !cEl) continue;
      const pr = pEl.getBoundingClientRect();
      const cr = cEl.getBoundingClientRect();
      const x1 = pr.left + pr.width  / 2 - wRect.left + wrap.scrollLeft;
      const y1 = pr.bottom - wRect.top + wrap.scrollTop;
      const x2 = cr.left + cr.width  / 2 - wRect.left + wrap.scrollLeft;
      const y2 = cr.top   - wRect.top + wrap.scrollTop;
      edgeData.push({ x1, y1, x2, y2 });
    }

    // Dashed lines between every consecutive candidate (sorted by vertical position)
    const candEls = [...wrap.querySelectorAll('[data-evo-idx]')];
    const candBands = candEls.map(el => {
      const r = el.getBoundingClientRect();
      return { idx: parseInt(el.dataset.evoIdx),
               bottom: r.bottom - wRect.top + wrap.scrollTop,
               top:    r.top    - wRect.top + wrap.scrollTop };
    }).sort((a, b) => a.top - b.top);
    let lines = '';
    for (let i = 0; i < candBands.length - 1; i++) {
      const y = (candBands[i].bottom + candBands[i + 1].top) / 2;
      lines += `<line x1="0" y1="${y}" x2="${W}" y2="${y}"
        stroke="#d0d7de" stroke-width="1" stroke-dasharray="4 6"/>`;
    }

    // Bezier edges
    let paths = '';
    for (const { x1, y1, x2, y2 } of edgeData) {
      const mid = (y1 + y2) / 2;
      paths += `<path d="M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}"
        fill="none" stroke="#b0b7be" stroke-width="3"/>`;
    }

    svg.innerHTML = lines + paths;
  });
}

function renderScores(it) {
  function scoreCls(v) { return v >= 0.7 ? 's-pass' : v >= 0.35 ? 's-mid' : 's-fail'; }
  function scoreStr(v) { return v === 1.0 ? '1' : v === 0.0 ? '0' : v.toFixed(2); }
  function scoreCell(v) {
    if (v === undefined || v === null) return `<td class="s-none">—</td>`;
    return `<td class="${scoreCls(v)}">${scoreStr(v)}</td>`;
  }
  function avg(map, keys) { return keys.reduce((s, k) => s + (map[k] || 0), 0) / keys.length; }

  function scorePanel(label, parentMap, childMap) {
    const keys = Object.keys(parentMap && Object.keys(parentMap).length ? parentMap : (childMap || {}));
    if (!keys.length) return '';
    const pAvg = avg(parentMap, keys);
    const cAvg = childMap ? avg(childMap, keys) : null;
    const summary = cAvg !== null
      ? `avg ${pAvg.toFixed(2)} → ${cAvg.toFixed(2)}`
      : `avg ${pAvg.toFixed(2)}`;
    let rows = '';
    for (const k of keys) {
      const p = parentMap[k];
      const c = childMap ? childMap[k] : undefined;
      let delta = '';
      if (c !== undefined && c !== null) {
        if (c > p)      delta = `<td class="s-delta-pos">↑</td>`;
        else if (c < p) delta = `<td class="s-delta-neg">↓</td>`;
        else            delta = `<td class="s-delta-eq">·</td>`;
      } else {
        delta = `<td class="s-none"></td>`;
      }
      const cat = (DATA.bench_categories || {})[k];
      const catTag = cat ? `<span class="bench-cat">${esc(cat)}</span>` : '';
      rows += `<tr><td class="bench-name">${esc(k)}${catTag}</td>${scoreCell(p)}${scoreCell(c)}${delta}</tr>`;
    }
    return `
      <div class="scores-panel">
        <div class="scores-panel-hdr">
          <span>${label}</span>
          <span class="s-summary">${summary}</span>
        </div>
        <table class="scores-tbl">
          <thead><tr>
            <th class="bench-col">Benchmark</th>
            <th>Parent</th><th>Child</th><th>Δ</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }
  const trainPanel = scorePanel('Train', it.parent_train, it.child_train);
  const valPanel   = scorePanel('Val',   it.parent_val,   it.child_val);
  if (!trainPanel && !valPanel) return '';
  return `<div class="section-title">Benchmark scores</div>
          <div class="scores-wrap">${trainPanel}${valPanel}</div>`;
}

function renderIteration(el, it) {
  const tabs = ['diff', 'pool', 'reflector', 'reasoning'];
  const tabBar = `<div class="tab-bar">${
    tabs.map(t => `<button class="tab${selTab===t?' sel':''}" onclick="selTab='${t}';renderMain()">${t.charAt(0).toUpperCase()+t.slice(1)}</button>`).join('')
  }</div>`;

  let body = '';

  if (selTab === 'diff') {
    const op = it.operation;
    if (op) {
      const parents = (it.parents || []).filter(p => p != null);
      let parentLabel = '';
      if (op === 'seed') {
        parentLabel = '';
      } else if (op === 'merge' && parents.length >= 2) {
        parentLabel = `<span class="parent-tag">Candidates #${parents[0]} × #${parents[1]}</span>`;
      } else if (parents.length >= 1) {
        parentLabel = `<span class="parent-tag">Candidate #${parents[0]}</span>`;
      }
      body += `<div class="op-row"><span class="op-badge op-${op}">${op}</span>${parentLabel}</div>`;
    } else if (it.parent_idx != null) {
      body += `<div class="op-row"><span class="parent-tag">Candidate #${it.parent_idx}</span></div>`;
    }
    body += renderScores(it);

    const pf = it.parent_files || {};
    const cf = it.child_files  || {};
    const allFiles = [...new Set([...Object.keys(pf), ...Object.keys(cf)])].sort();

    if (!allFiles.length) {
      body += `<div class="no-diff">No file data for this iteration.</div>`;
    } else {
      const added     = allFiles.filter(f => !(f in pf) &&  (f in cf));
      const removed   = allFiles.filter(f =>  (f in pf) && !(f in cf));
      const modified  = allFiles.filter(f =>  (f in pf) &&  (f in cf) && pf[f] !== cf[f]);
      const unchanged = allFiles.filter(f =>  (f in pf) &&  (f in cf) && pf[f] === cf[f]);

      const summaryMap = {};
      (it.changes_summary || []).forEach(l => {
        const ci = l.indexOf(':');
        if (ci > 0) summaryMap[l.slice(0, ci).trim()] = l.slice(ci + 1).trim();
      });
      body += `<div class="section-title">Files</div>`;
      body += renderFilesOverview(added, removed, modified, unchanged, summaryMap);

      const bullets = Array.isArray(it.changes) ? it.changes : (it.changes || '').split(/\n(?=- )/).map(s => s.replace(/^- /, '').trim()).filter(Boolean);
      if (bullets.length) {
        body += `<div class="section-title">Reflector changes</div>`;
        body += `<div class="changes-box"><ul>${bullets.map(b => `<li>${esc(b)}</li>`).join('')}</ul></div>`;
      }

      for (const f of [...modified, ...added].sort()) {
        const diff = lineDiff(pf[f] || '', cf[f] || '');
        const dels = diff.filter(d => d.t === 'd').length;
        const ins  = diff.filter(d => d.t === 'i').length;
        const isNew = !(f in pf);
        body += `<div class="section-title" style="margin-top:18px">${esc(f)}</div>`;
        body += `<div class="diff-meta">`;
        if (!isNew) body += `<span class="stat-del">−${dels} lines</span>`;
        body += `<span class="stat-ins">+${ins} lines</span>`;
        if (isNew) body += `<span style="color:#6e7681;font-size:11px;margin-left:4px">new file</span>`;
        body += `</div>`;
        body += renderUnifiedDiff(diff);
      }
      for (const f of removed) {
        body += `<div class="section-title" style="margin-top:18px">${esc(f)}</div>`;
        body += `<div class="no-diff" style="border-left:3px solid #f85149;color:#f85149">deleted</div>`;
      }
    }

  } else if (selTab === 'pool') {
    body += renderPool(it);

  } else if (selTab === 'reflector') {
    body += renderReflector(it);

  } else {
    body += renderReasoning(it);
  }

  el.innerHTML = tabBar + body;
}

// ── Pool renderer ─────────────────────────────────────────────────────────
function renderPool(it) {
  if (!it.pool_json) return `<div class="no-diff">No pool data for this iteration.</div>`;
  const data = JSON.parse(it.pool_json);
  const cands = data.candidates || [];
  if (!cands.length) return `<div class="no-diff">No candidates in pool.</div>`;

  function chipCls(v) { return v >= 0.7 ? 'pc-pass' : v >= 0.35 ? 'pc-mid' : 'pc-fail'; }
  function chipStr(v) { return v === 1.0 ? '1' : v === 0.0 ? '0' : v.toFixed(2); }
  function chips(map) {
    return Object.entries(map || {}).map(([k, v]) =>
      `<span class="pool-chip ${chipCls(v)}">${esc(k)} ${chipStr(v)}</span>`
    ).join('');
  }

  let html = `<div class="pool-header">Snapshot ${data.iteration_snapshot ?? '—'} · ${cands.length} candidate${cands.length !== 1 ? 's' : ''}</div>
              <div class="pool-wrap">`;

  for (const c of cands) {
    const avg    = c.val_avg != null ? c.val_avg : null;
    const avgStr = avg != null ? avg.toFixed(2) : '—';
    const avgCls = avg == null ? '' : avg >= 0.7 ? 'g' : avg >= 0.4 ? 'y' : 'r';
    const parents = c.parent_ids && c.parent_ids.length ? `#${c.parent_ids.join(', #')}` : 'seed';
    const pareto  = (c.on_pareto_front || []).length;

    const valChips   = chips(c.val);
    const trainChips = chips(c.train_subsample);

    html += `<div class="pool-card">
      <div class="pool-card-hdr">
        <span class="pool-card-title">Candidate #${c.idx}</span>
        <span class="pool-avg ${avgCls}">${avgStr}</span>
      </div>
      <div class="pool-card-body">
        <div class="pool-row"><span class="pool-lbl">parents</span><span class="pool-mval">${esc(parents)}</span></div>
        <div class="pool-row"><span class="pool-lbl">pareto</span><span class="pool-mval">${pareto} benchmark${pareto !== 1 ? 's' : ''}</span></div>
        ${valChips   ? `<div class="pool-slbl">Val</div><div class="pool-chips">${valChips}</div>` : ''}
        ${trainChips ? `<div class="pool-slbl">Train sample</div><div class="pool-chips">${trainChips}</div>` : ''}
      </div>
    </div>`;
  }
  html += '</div>';
  return html;
}

// ── Reflector renderer ────────────────────────────────────────────────────
function reflContent(text) {
  return text.split('\n').map(line => {
    if (/^## /.test(line))  return `<span class="rh">${esc(line)}</span>`;
    if (/^```/.test(line))  return `<span class="rf">${esc(line)}</span>`;
    return esc(line);
  }).join('\n');
}

function renderReflector(it) {
  if (!it.reflector_json) return `<div class="no-diff">No reflector data for this iteration.</div>`;
  const data = JSON.parse(it.reflector_json);
  let html = '';

  // Input — may be a string (agentic reflector) or an array of messages (LLM reflector)
  const inputMsgs = Array.isArray(data.input) ? data.input : (data.input ? [{role: 'user', content: data.input}] : []);
  for (const msg of inputMsgs) {
    html += `<div class="refl-block">
      <div class="refl-block-hdr">${esc(msg.role)}</div>
      <div class="refl-body">${reflContent(msg.content || '')}</div>
    </div>`;
  }

  // Output
  const raw = data.output || '';
  if (raw) {
    html += `<div class="section-title" style="margin-top:4px">Output</div>`;
    let changes = [], prompt = '';
    try {
      const out = JSON.parse(raw);
      changes = out.changes || [];
      prompt  = out.prompt  || '';
    } catch(_) {
      // Fallback: unescaped quotes inside the prompt string break JSON.parse.
      // Extract changes (clean array before "prompt") and prompt (greedy to last " before }) separately.
      const cm = raw.match(/"changes"\s*:\s*(\[[\s\S]*?\])\s*,\s*"prompt"/);
      if (cm) { try { changes = JSON.parse(cm[1]); } catch(__) {} }
      const pm = raw.match(/"prompt"\s*:\s*"([\s\S]*)"\s*\}\s*$/);
      if (pm) {
        prompt = pm[1].replace(/\\(.)/g, (_, c) =>
          ({n:'\n', t:'\t', r:'\r', '\\':'\\', '"':'"'}[c] || c));
      }
    }
    const items = Array.isArray(changes)
      ? changes
      : String(changes).split(/\n(?=- )/).map(s => s.replace(/^- /, '').trim()).filter(Boolean);
    if (items.length) {
      html += `<div class="changes-box"><ul>${items.map(b => `<li>${esc(b)}</li>`).join('')}</ul></div>`;
    }
    if (prompt) {
      html += `<div class="refl-block" style="margin-top:12px">
        <div class="refl-block-hdr">New prompt</div>
        <div class="refl-body">${esc(prompt)}</div>
      </div>`;
    }
    if (!items.length && !prompt) {
      html += `<div class="refl-block"><div class="refl-body">${esc(raw)}</div></div>`;
    }
  }

  return html;
}

// ── Reasoning steps renderer ─────────────────────────────────────────────
function renderReasoning(it) {
  if (!it.reflector_steps) return `<div class="no-diff">No agentic reflector steps for this iteration.</div>`;
  let steps;
  try { steps = JSON.parse(it.reflector_steps); } catch(_) {
    return `<div class="no-diff">Could not parse steps data.</div>`;
  }
  if (!Array.isArray(steps) || !steps.length) return `<div class="no-diff">No steps recorded.</div>`;

  // Strip final output message (the JSON blob the agent writes as its answer)
  if (steps[steps.length - 1].type === 'text' &&
      steps[steps.length - 1].text.trimStart().startsWith('{')) {
    steps = steps.slice(0, -1);
  }

  function shortPath(p) {
    const m = (p || '').match(/iteration_\d+\/(.*)/);
    return m ? m[1] : p;
  }
  function fmtThinking(text) {
    return esc(text).replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  }

  let html = '<div class="rsn-feed">';
  for (const step of steps) {
    if (step.type === 'thinking') {
      html += `<div class="rsn-step rsn-step-thinking">
        <div class="rsn-label">Thinking</div>
        <div class="rsn-body">${fmtThinking(step.text)}</div>
      </div>`;
    } else if (step.type === 'text') {
      html += `<div class="rsn-step rsn-step-text">
        <div class="rsn-label">Text</div>
        <div class="rsn-body">${esc(step.text)}</div>
      </div>`;
    } else if (step.type === 'tool') {
      const name = step.name || '?';
      const inp  = step.input || {};
      const params = Object.entries(inp)
        .filter(([k, v]) => v !== null && v !== undefined && v !== '' && v !== 0)
        .map(([k, v]) => {
          const display = (k === 'filePath' || k === 'path') ? shortPath(String(v)) : String(v);
          return `<div class="rsn-param-row"><span class="rsn-param-key">${esc(k)}</span><span class="rsn-param-val">${esc(display)}</span></div>`;
        }).join('');
      html += `<div class="rsn-step rsn-step-tool">
        <div class="rsn-label">Tool</div>
        <div class="rsn-body"><span class="rsn-tool-name">${esc(name)}</span>${params ? `<div class="rsn-params">${params}</div>` : ''}</div>
      </div>`;
    }
  }
  html += '</div>';
  return html;
}

// ── Files overview ────────────────────────────────────────────────────────
function renderFilesOverview(added, removed, modified, unchanged, summaryMap) {
  summaryMap = summaryMap || {};
  function lookup(f) {
    if (summaryMap[f]) return summaryMap[f];
    return summaryMap[f.split('/').pop()] || '';
  }
  function fileRows(cls, label, files) {
    return files.map(f => {
      const desc = lookup(f);
      return `<div class="file-row"><span class="file-badge fb-${cls}">${label}</span><span class="file-name">${esc(f)}</span>${desc ? `<span class="file-summary">${esc(desc)}</span>` : ''}</div>`;
    }).join('');
  }
  const rows = fileRows('modified', 'modified', modified)
             + fileRows('added',    'added',    added)
             + fileRows('removed',  'deleted',  removed)
             + fileRows('unchanged','unchanged',unchanged);
  return `<div class="files-overview">${rows}</div>`;
}

// ── Render unified diff (git-style) ───────────────────────────────────────
function renderUnifiedDiff(diff) {
  const CTX = 4;
  // Assign line numbers
  const lines = [];
  let lA = 1, lB = 1;
  for (const d of diff) {
    if (d.t === 'e') { lines.push({ t: 'ctx', lA: lA++, lB: lB++, text: d.line }); }
    else if (d.t === 'd') { lines.push({ t: 'del', lA: lA++, lB: null, text: d.line }); }
    else               { lines.push({ t: 'ins', lA: null, lB: lB++, text: d.line }); }
  }

  // Mark lines within CTX of a change
  const show = new Array(lines.length).fill(false);
  lines.forEach((l, i) => {
    if (l.t !== 'ctx') {
      for (let j = Math.max(0, i - CTX); j <= Math.min(lines.length - 1, i + CTX); j++)
        show[j] = true;
    }
  });

  // Group into contiguous hunks
  const hunks = [];
  let i = 0;
  while (i < lines.length) {
    if (!show[i]) { i++; continue; }
    let j = i;
    while (j < lines.length && show[j]) j++;
    hunks.push(lines.slice(i, j));
    i = j;
  }

  let html = '<div class="unified-diff">';
  for (const hunk of hunks) {
    const firstOld = hunk.find(l => l.lA !== null)?.lA ?? 1;
    const firstNew = hunk.find(l => l.lB !== null)?.lB ?? 1;
    const cntOld   = hunk.filter(l => l.t !== 'ins').length;
    const cntNew   = hunk.filter(l => l.t !== 'del').length;
    html += `<div class="u-hunk">@@ -${firstOld},${cntOld} +${firstNew},${cntNew} @@</div>`;
    for (const l of hunk) {
      if (l.t === 'del') html += `<div class="u-del"><span class="u-sign">-</span>${esc(l.text)}</div>`;
      else if (l.t === 'ins') html += `<div class="u-ins"><span class="u-sign">+</span>${esc(l.text)}</div>`;
      else html += `<div class="u-ctx"><span class="u-sign"> </span>${esc(l.text)}</div>`;
    }
  }
  html += '</div>';
  return html;
}

// ── LCS line diff ──────────────────────────────────────────────────────────
function lineDiff(a, b) {
  const al = a ? a.split('\n') : [];
  const bl = b ? b.split('\n') : [];
  const m = al.length, n = bl.length;
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = al[i] === bl[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && al[i] === bl[j]) { out.push({ t: 'e', line: al[i] }); i++; j++; }
    else if (j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j])) { out.push({ t: 'i', line: bl[j] }); j++; }
    else { out.push({ t: 'd', line: al[i] }); i++; }
  }
  return out;
}

function pad(n) { return String(n).padStart(3, '0'); }
function esc(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

(function() {
  function makeResizer(resizerId, targetId, side) {
    const resizer = document.getElementById(resizerId);
    const target  = document.getElementById(targetId);
    let startX, startW;
    resizer.addEventListener('mousedown', e => {
      startX = e.clientX;
      startW = target.getBoundingClientRect().width;
      resizer.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      function onMove(e) {
        const dx  = e.clientX - startX;
        const newW = side === 'left' ? startW + dx : startW - dx;
        if (newW >= 80 && newW <= 700) target.style.width = newW + 'px';
      }
      function onUp() {
        resizer.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      e.preventDefault();
    });
  }
  makeResizer('resizer-left',  'sidebar',      'left');
  makeResizer('resizer-right', 'info-sidebar', 'right');
})();

renderInfoSidebar();
renderSidebar();
renderMain();
</script>
</body>
</html>
"""


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/experiment1")
    data = load_experiment(exp_dir)
    safe_json = (json.dumps(data, ensure_ascii=False)
                 .replace("<", "\\u003c")
                 .replace(">", "\\u003e")
                 .replace("&", "\\u0026"))
    html = HTML.replace("__NAME__", data["name"]).replace("__DATA__", safe_json)
    out = exp_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated: {out}")


main()
