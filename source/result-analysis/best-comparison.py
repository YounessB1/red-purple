#!/usr/bin/env python3
"""Compare the best candidate found across one or more red-purple experiments.

For each experiment directory passed on the command line, this scans every
iteration_*/pool.json for the candidate with the highest val_avg, resolves
its files via .candidates/<hash>/, and renders a side-by-side comparison of
scores, prompt.md, AGENTS.md and skills across experiments.

Usage: python best-comparison.py <experiment_dir> [<experiment_dir> ...] [-o OUTPUT]
Output: <experiments_dir>/best-comparison.html (next to the experiment dirs,
unless -o is given).
"""
import argparse
import difflib
import html
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

_SEED_BASELINE_REL = Path("seed/gemma-4-31b")


def find_best_candidate(exp_dir: Path) -> dict | None:
    """Scan every iteration_*/pool.json under exp_dir and return the metadata
    for the candidate with the highest val_avg seen anywhere in the run."""
    best = None
    for pool_path in sorted(exp_dir.glob("iteration_*/pool.json")):
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        iteration = pool.get("iteration_snapshot")
        for cand in pool.get("candidates", []):
            val_avg = cand.get("val_avg")
            if val_avg is None:
                continue
            if best is None or val_avg > best["val_avg"]:
                best = {
                    "val_avg": val_avg,
                    "idx": cand.get("idx"),
                    "iteration": iteration,
                    "val": cand.get("val", {}),
                    "on_pareto_front": cand.get("on_pareto_front", []),
                    "files": cand.get("files"),
                    "pool_path": pool_path,
                }
    return best


def load_candidate_files(exp_dir: Path, files_hash: str) -> dict:
    """Load prompt.md, AGENTS.md and skill files for a candidate's file hash."""
    result = {"prompt": "", "agents": "", "skills": {}}
    if not files_hash:
        return result
    cand_dir = exp_dir / ".candidates" / files_hash
    if not cand_dir.exists():
        return result
    prompt_path = cand_dir / "prompt.md"
    if prompt_path.exists():
        result["prompt"] = prompt_path.read_text(encoding="utf-8", errors="replace")
    agents_path = cand_dir / "AGENTS.md"
    if agents_path.exists():
        result["agents"] = agents_path.read_text(encoding="utf-8", errors="replace")
    skills_dir = cand_dir / ".opencode" / "skills"
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            result["skills"][skill_file.parent.name] = skill_file.read_text(
                encoding="utf-8", errors="replace"
            )
    return result


def load_summary(exp_dir: Path) -> dict:
    summary_path = exp_dir / "experiment_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_experiment_data(exp_dir: Path) -> dict:
    exp_dir = exp_dir.resolve()
    data = {"name": exp_dir.name, "path": str(exp_dir), "best": None, "summary": load_summary(exp_dir)}
    best = find_best_candidate(exp_dir)
    if best is None:
        return data
    files = load_candidate_files(exp_dir, best["files"])
    val = best["val"]
    passed = sum(1 for v in val.values() if v == 1.0)
    data["best"] = {
        **best,
        "pool_path": str(best["pool_path"].relative_to(exp_dir)),
        "pass_count": passed,
        "task_count": len(val),
        **files,
    }
    return data


def load_seed_entry(exp_dir: Path) -> dict | None:
    """The unoptimized starting candidate (iteration_000, candidate 0) — used
    as the fixed baseline for diffs, instead of whichever experiment happens
    to be listed first on the command line."""
    exp_dir = exp_dir.resolve()
    evo_path = exp_dir / "iteration_000" / "evolution.json"
    child_dir = exp_dir / "iteration_000" / "child"
    if not evo_path.exists() or not child_dir.exists():
        return None
    try:
        evo = json.loads(evo_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    val = evo.get("child", {}).get("val") or {}
    vals = [v for v in val.values() if v is not None]
    val_avg = sum(vals) / len(vals) if vals else 0.0

    prompt = agents = ""
    skills: dict = {}
    prompt_path = child_dir / "prompt.md"
    if prompt_path.exists():
        prompt = prompt_path.read_text(encoding="utf-8", errors="replace")
    agents_path = child_dir / "AGENTS.md"
    if agents_path.exists():
        agents = agents_path.read_text(encoding="utf-8", errors="replace")
    skills_dir = child_dir / ".opencode" / "skills"
    if skills_dir.exists():
        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            skills[skill_file.parent.name] = skill_file.read_text(encoding="utf-8", errors="replace")

    best = {
        "val_avg": val_avg,
        "idx": 0,
        "iteration": 0,
        "val": val,
        "on_pareto_front": [],
        "files": None,
        "pool_path": "iteration_000/evolution.json",
        "pass_count": sum(1 for v in val.values() if v == 1.0),
        "task_count": len(val),
        "prompt": prompt,
        "agents": agents,
        "skills": skills,
    }
    return {"name": "Seed", "path": str(exp_dir), "best": best, "summary": {}}


def run_task_maps_from_results(exp_dir: Path, split: str) -> list[dict[str, bool]]:
    """One {benchmark_id: success} map per repeated held-out run, from
    exp_dir/test/run*/<split>/results.json's per-task 'results' list."""
    maps = []
    for run_dir in sorted(exp_dir.glob("test/run*")):
        results_path = run_dir / split / "results.json"
        if not results_path.exists():
            continue
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_map = {
            item["benchmark_id"]: bool(item.get("success"))
            for item in data.get("results", [])
            if item.get("benchmark_id")
        }
        if run_map:
            maps.append(run_map)
    return maps


def run_task_maps_from_metadata(baseline_dir: Path, split: str) -> list[dict[str, bool]]:
    """One {benchmark_id: success} map per run for the seed baseline, from
    baseline_dir/run*/<split>/*/metadata.json's 'success' flag — this baseline
    predates results.json and only has raw task dirs."""
    maps = []
    for run_dir in sorted(baseline_dir.glob("run*")):
        split_dir = run_dir / split
        if not split_dir.is_dir():
            continue
        run_map = {}
        for task_dir in split_dir.iterdir():
            meta_path = task_dir / "metadata.json"
            if not task_dir.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            run_map[task_dir.name] = bool(meta.get("success"))
        if run_map:
            maps.append(run_map)
    return maps


def _run_group(
    name: str,
    run_maps: list[dict[str, bool]],
    driving_tasks: set[str] | None = None,
    pareto_tasks: set[str] | None = None,
) -> dict:
    rates = [sum(1 for v in m.values() if v) / len(m) for m in run_maps if m]
    return {
        "name": name,
        "rates": rates,
        "runs": run_maps,
        "driving_tasks": driving_tasks or set(),
        "pareto_tasks": pareto_tasks or set(),
    }


def _accepted_children_by_idx(exp_dir: Path) -> dict[int, dict]:
    """candidate_idx -> that iteration's evolution.json, for every accepted
    iteration (an idx is produced by exactly one accepted iteration)."""
    mapping: dict[int, dict] = {}
    for evo_path in sorted(exp_dir.glob("iteration_*/evolution.json")):
        try:
            evo = json.loads(evo_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        child = evo.get("child", {})
        idx = child.get("candidate_idx")
        if child.get("accepted") and idx is not None:
            mapping[idx] = evo
    return mapping


def find_driving_tasks(exp_dir: Path, best_idx: int | None) -> set[str]:
    """Training-set challenges that drove the evolution of the best candidate:
    walk the accepted-iteration ancestry from best_idx back to the seed (idx 0)
    via each iteration's "parents" links, and collect every train-minibatch
    task that scored < 1.0 for the parent at that step (i.e. the imperfect
    task(s) that triggered that step's diagnosis/reflection)."""
    if best_idx is None:
        return set()
    idx_to_evo = _accepted_children_by_idx(exp_dir)
    driving: set[str] = set()
    visited: set[int] = set()
    stack = [best_idx]
    while stack:
        idx = stack.pop()
        if idx in visited or idx == 0:
            visited.add(idx)
            continue
        visited.add(idx)
        evo = idx_to_evo.get(idx)
        if evo is None:
            continue
        parent_train = evo.get("parent", {}).get("train") or {}
        driving.update(tid for tid, score in parent_train.items() if score < 1.0)
        stack.extend(evo.get("parents", []))
    return driving


def build_run_groups(experiments_root: Path, experiment_dirs: list[Path], split: str) -> list[dict]:
    """One group per candidate (seed baseline + each named experiment), each holding
    the per-run pass rates (for the box plot) and the raw per-task/per-run success
    maps (for the solved/not-solved table) on the given split."""
    groups = []
    seed_dir = experiments_root / _SEED_BASELINE_REL
    if seed_dir.exists():
        seed_maps = run_task_maps_from_metadata(seed_dir, split)
        if seed_maps:
            groups.append(_run_group("Seed", seed_maps))
    for exp_dir in experiment_dirs:
        exp_dir = exp_dir.resolve()
        maps = run_task_maps_from_results(exp_dir, split)
        if maps:
            best = find_best_candidate(exp_dir)
            driving_tasks = find_driving_tasks(exp_dir, best["idx"] if best else None)
            pareto_tasks = set(best.get("on_pareto_front") or []) if best else set()
            groups.append(_run_group(exp_dir.name, maps, driving_tasks, pareto_tasks))
    return groups


def _tukey_hinges(values: list[float]) -> tuple[float, float, float]:
    """Median and Tukey hinges (q1, q3) — a simple, well-defined quartile method
    that behaves sanely on the small samples (n=5) these run counts produce."""
    s = sorted(values)
    n = len(s)
    median = statistics.median(s)
    mid = n // 2
    lower = s[:mid]
    upper = s[mid + (n % 2):]
    q1 = statistics.median(lower) if lower else median
    q3 = statistics.median(upper) if upper else median
    return q1, median, q3


_BOXPLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _nice_step(raw_step: float) -> float:
    """Round an axis step up to a human-friendly 1/2/5x10^k value."""
    if raw_step <= 0:
        return 1.0
    exp = math.floor(math.log10(raw_step))
    base = raw_step / (10 ** exp)
    nice = 1 if base <= 1 else 2 if base <= 2 else 5 if base <= 5 else 10
    return nice * (10 ** exp)


def _axis_bounds_and_ticks(values: list[float]) -> tuple[float, float, list[float]]:
    """Auto-scale the y-axis to the data's actual spread (plus padding) instead of
    a fixed 0-100 — a tight cluster of pass rates would otherwise collapse into a
    sliver a few pixels tall."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        lo, hi = lo - 5, hi + 5
    pad = (hi - lo) * 0.2
    raw_lo, raw_hi = lo - pad, hi + pad
    step = _nice_step((raw_hi - raw_lo) / 4)
    y0 = max(0.0, math.floor(raw_lo / step) * step)
    y1 = min(100.0, math.ceil(raw_hi / step) * step)
    ticks = []
    t = y0
    while t <= y1 + 1e-6:
        ticks.append(round(t, 6))
        t += step
    return y0, y1, ticks


def render_boxplot_panels(groups: list[dict]) -> str:
    """One self-contained mini box plot per group (small multiples), laid out side
    by side but sharing a single y-scale so heights stay honestly comparable —
    bigger and less cluttered than cramming every box onto one shared x-axis."""
    plot_h = 340
    margin_t, margin_b = 40, 34
    panel_w = 190
    box_w = 84
    cap_w = box_w * 0.32
    jitter_steps = [-0.30, -0.16, 0.0, 0.16, 0.30]

    all_rates_pct = [r * 100 for g in groups for r in g["rates"]]
    y0, y1, ticks = _axis_bounds_and_ticks(all_rates_pct) if all_rates_pct else (0.0, 100.0, [0, 25, 50, 75, 100])

    def y_of(pct: float) -> float:
        return margin_t + plot_h * (1 - (pct - y0) / (y1 - y0))

    total_h = margin_t + plot_h + margin_b
    panels = []
    for i, g in enumerate(groups):
        color = _BOXPLOT_COLORS[i % len(_BOXPLOT_COLORS)]
        show_axis = i == 0
        axis_w = 40 if show_axis else 10
        cx = axis_w + panel_w / 2

        grid = []
        for t in ticks:
            y = y_of(t)
            grid.append(
                f'<line x1="{axis_w:.1f}" y1="{y:.1f}" x2="{axis_w + panel_w:.1f}" y2="{y:.1f}" '
                'stroke="#e1e0d9" stroke-width="1"/>'
            )
            if show_axis:
                label = f"{t:.0f}" if abs(t - round(t)) < 1e-6 else f"{t:.1f}"
                grid.append(
                    f'<text x="{axis_w - 10:.1f}" y="{y + 3:.1f}" text-anchor="end" class="bp-axis-lbl">'
                    f'{label}</text>'
                )

        rates_pct = [r * 100 for r in g["rates"]]
        q1, med, q3 = _tukey_hinges(rates_pct)
        lo, hi = min(rates_pct), max(rates_pct)
        y_lo, y_hi = y_of(lo), y_of(hi)
        y_q1, y_q3 = y_of(q1), y_of(q3)
        y_med = y_of(med)

        marks = [
            f'<line x1="{cx:.1f}" y1="{y_lo:.1f}" x2="{cx:.1f}" y2="{y_hi:.1f}" '
            f'stroke="{color}" stroke-width="2"/>'
        ]
        for y in (y_lo, y_hi):
            marks.append(
                f'<line x1="{cx - cap_w / 2:.1f}" y1="{y:.1f}" x2="{cx + cap_w / 2:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        box_top, box_bottom = min(y_q1, y_q3), max(y_q1, y_q3)
        marks.append(
            f'<rect x="{cx - box_w / 2:.1f}" y="{box_top:.1f}" width="{box_w:.1f}" '
            f'height="{max(1.0, box_bottom - box_top):.1f}" fill="{color}" fill-opacity="0.14" '
            f'stroke="{color}" stroke-width="2" rx="4"/>'
        )
        marks.append(
            f'<line x1="{cx - box_w / 2:.1f}" y1="{y_med:.1f}" x2="{cx + box_w / 2:.1f}" y2="{y_med:.1f}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        for j, val in enumerate(rates_pct):
            jit = jitter_steps[j % len(jitter_steps)]
            px = cx + jit * box_w
            py = y_of(val)
            tip = html.escape(f"{g['name']} · run {j + 1} · {val:.1f}%").replace("'", "&#39;")
            marks.append(
                f'<circle class="bp-dot" cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" '
                f'stroke="#fcfcfb" stroke-width="2" aria-label="{tip}" '
                f"onmouseenter=\"bpTipShow(event,'{tip}')\" onmousemove=\"bpTipMove(event)\" "
                f'onmouseleave="bpTipHide()"></circle>'
            )
        value_label = (
            f'<text x="{cx:.1f}" y="{y_hi - 10:.1f}" text-anchor="middle" class="bp-value-lbl">'
            f'{med:.1f}%</text>'
        )

        svg = (
            f'<svg width="{axis_w + panel_w:.0f}" height="{total_h:.0f}" overflow="visible">'
            f'{"".join(grid)}{"".join(marks)}{value_label}'
            "</svg>"
        )
        panels.append(
            '<div class="bp-panel">'
            f'<div class="bp-panel-hdr">{html.escape(g["name"])}</div>'
            f'{svg}'
            "</div>"
        )
    return f'<div class="bp-panels">{"".join(panels)}</div>'


def render_run_summary_table(groups: list[dict]) -> str:
    n_runs = max((len(g["runs"]) for g in groups), default=0)
    header = (
        "<th>Experiment</th>"
        + "".join(f"<th>Run {i + 1}</th>" for i in range(n_runs))
        + "<th>Mean Solved</th><th>Variance</th>"
    )
    rows = []
    for g in groups:
        runs = g["runs"]
        solved_counts = [sum(1 for v in run_map.values() if v) for run_map in runs]
        totals = [len(run_map) for run_map in runs]
        cells = "".join(f"<td>{solved}/{total}</td>" for solved, total in zip(solved_counts, totals))
        mean_solved = statistics.mean(solved_counts) if solved_counts else 0.0
        variance_solved = statistics.variance(solved_counts) if len(solved_counts) > 1 else 0.0

        rows.append(
            "<tr>"
            f'<td class="exp-name">{html.escape(g["name"])}</td>{cells}'
            f'<td class="val-avg">{mean_solved:.1f}</td>'
            f'<td class="val-avg">{variance_solved:.2f}</td>'
            "</tr>"
        )
    return (
        '<table class="summary-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_run_summary_section(groups: list[dict], split_label: str) -> str:
    if not groups:
        return ""
    return (
        f'<h3 class="bp-title">{html.escape(split_label)} Split — Benchmarks Solved per Run</h3>'
        f"{render_run_summary_table(groups)}"
    )


def render_boxplot_section(groups: list[dict], split_label: str) -> str:
    if not groups:
        return (
            '<div class="no-diff">'
            f'No {html.escape(split_label.lower())}-split run data found (expected '
            f'test/run*/{html.escape(split_label.lower())}/results.json per experiment, plus a '
            f'{_SEED_BASELINE_REL} baseline).</div>'
        )
    n_runs = max((len(g["rates"]) for g in groups), default=0)
    caption = (
        f'Pass rate per run on the {html.escape(split_label.lower())} split &middot; up to {n_runs} runs '
        "per group &middot; box = interquartile range (Tukey hinges), line = median, "
        "whiskers = min/max, dots = individual runs (hover a dot for its exact value)"
    )
    return (
        f'<h3 class="bp-title">{html.escape(split_label)} Split — Pass Rate Across Runs</h3>'
        f'{render_boxplot_panels(groups)}'
        f'<div class="bp-caption">{caption}</div>'
    )


def _shared_task_order(groups: list[dict]) -> list[str]:
    """Order benchmarks by total solve count across every run of every experiment
    combined (up to len(groups) * runs-per-group solves) — most-solved first, never
    solved last, with everything in between naturally graded by how consistently it
    was solved. Computed once and reused for every experiment's table so rows line
    up across them."""
    task_ids = sorted({tid for g in groups for run_map in g["runs"] for tid in run_map})

    def total_solves(tid: str) -> int:
        return sum(
            1
            for g in groups
            for run_map in g["runs"]
            if run_map.get(tid)
        )

    return sorted(task_ids, key=lambda tid: (-total_solves(tid), tid))


def render_pass_fail_table(group: dict, task_order: list[str]) -> str:
    runs = group["runs"]
    driving_tasks = group.get("driving_tasks") or set()
    pareto_tasks = group.get("pareto_tasks") or set()
    task_ids = [tid for tid in task_order if any(tid in run_map for run_map in runs)]
    header = "<th>Benchmark</th>" + "".join(f"<th>Run {i + 1}</th>" for i in range(len(runs)))
    rows = []
    for tid in task_ids:
        cells = []
        for i, run_map in enumerate(runs):
            success = run_map.get(tid)
            if success is None:
                cls, title = "pf-na", f"Run {i + 1}: not evaluated"
            elif success:
                cls, title = "pf-solved", f"Run {i + 1}: solved"
            else:
                cls, title = "pf-unsolved", f"Run {i + 1}: not solved"
            cells.append(f'<td class="pf-cell {cls}" title="{html.escape(title)}"></td>')
        is_driver = tid in driving_tasks
        is_pareto = tid in pareto_tasks
        row_classes = " ".join(c for c in ("pf-driver-row" if is_driver else "", "pf-pareto-row" if is_pareto else "") if c)
        row_cls = f' class="{row_classes}"' if row_classes else ""
        task_classes = " ".join(c for c in ("task-id", "pf-task", "pf-driver" if is_driver else "", "pf-pareto" if is_pareto else "") if c)
        titles = []
        if is_driver:
            titles.append("Drove this candidate's evolution (imperfect on this task at an accepted mutation step)")
        if is_pareto:
            titles.append("On this candidate's val Pareto front (the single best candidate found on this task)")
        task_title = f' title="{html.escape(" · ".join(titles))}"' if titles else ""
        rows.append(f'<tr{row_cls}><td class="{task_classes}"{task_title}>{html.escape(tid)}</td>{"".join(cells)}</tr>')
    return (
        '<div class="pf-table-wrap"><table class="pf-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def render_pass_fail_section(groups: list[dict], split_label: str) -> str:
    if not groups:
        return ""
    task_order = _shared_task_order(groups)
    blocks = [
        f'<div class="pf-block"><div class="pf-block-hdr">{html.escape(g["name"])}</div>'
        f'{render_pass_fail_table(g, task_order)}</div>'
        for g in groups
    ]
    has_driver = any(g.get("driving_tasks") for g in groups)
    has_pareto = any(g.get("pareto_tasks") for g in groups)
    legend_items = []
    if has_driver:
        legend_items.append('<span class="pf-legend-item"><span class="pf-legend-swatch pf-driver"></span>drove this candidate\'s evolution (imperfect train task at an accepted mutation)</span>')
    if has_pareto:
        legend_items.append('<span class="pf-legend-item"><span class="pf-legend-swatch pf-pareto"></span>on the val Pareto front (best candidate found so far on this task)</span>')
    legend = f'<div class="pf-legend">{"".join(legend_items)}</div>' if legend_items else ""
    return (
        f'<h3 class="bp-title">{html.escape(split_label)} Split — Solved / Not Solved per Run</h3>'
        f'{legend}'
        f'<div class="pf-grid">{"".join(blocks)}</div>'
    )


def diff_lines(base_text: str, other_text: str) -> str:
    """Render other_text as HTML lines, highlighting insertions/deletions
    relative to base_text."""
    base_lines = base_text.splitlines()
    other_lines = other_text.splitlines()
    sm = difflib.SequenceMatcher(a=base_lines, b=other_lines, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for line in other_lines[j1:j2]:
                out.append(f'<div class="diff-line ctx">{html.escape(line) or "&nbsp;"}</div>')
        elif tag == "insert":
            for line in other_lines[j1:j2]:
                out.append(f'<div class="diff-line add">+ {html.escape(line) or "&nbsp;"}</div>')
        elif tag == "delete":
            for line in base_lines[i1:i2]:
                out.append(f'<div class="diff-line del">- {html.escape(line) or "&nbsp;"}</div>')
        elif tag == "replace":
            for line in base_lines[i1:i2]:
                out.append(f'<div class="diff-line del">- {html.escape(line) or "&nbsp;"}</div>')
            for line in other_lines[j1:j2]:
                out.append(f'<div class="diff-line add">+ {html.escape(line) or "&nbsp;"}</div>')
    return "\n".join(out) or '<div class="diff-line ctx">&nbsp;</div>'


def plain_lines(text: str) -> str:
    lines = text.splitlines() or [""]
    return "\n".join(
        f'<div class="diff-line ctx">{html.escape(line) or "&nbsp;"}</div>' for line in lines
    )


def file_color(is_skill: bool) -> dict:
    if is_skill:
        return {"bg": "#fff8c5", "border": "#d4a72c", "text": "#9a6700"}
    return {"bg": "#ffebe9", "border": "#cf222e", "text": "#cf222e"}


def render_file_block(display_name: str, content: str, is_skill: bool) -> str:
    """Same graphical file-block used by the visualize Evolution tree / Test
    tab (evolution.js::evoNodeHtml, test_results.js::agentCardHtml) — a
    colored pill sized roughly by line count."""
    lines = content.count("\n") + 1 if content else 0
    col = file_color(is_skill)
    radius = "14px" if is_skill else "5px"
    extra_pad = min(60, max(0, lines * 3 - 8))
    style = (
        f"border-radius:{radius};padding-right:{8 + extra_pad}px;"
        f"background:{col['bg']};border-color:{col['border']};color:{col['text']}"
    )
    title = html.escape(f"{display_name} — {lines} lines")
    return f'<div class="evo-file-block" style="{style}" title="{title}">{html.escape(display_name)}</div>'


def render_agent_card(exp: dict) -> str:
    best = exp["best"]
    if best is None:
        return (
            '<div class="evo-candidate test-agent-card">'
            f'<div class="evo-candidate-hdr">{html.escape(exp["name"])}</div>'
            '<div class="evo-candidate-sub">no data</div></div>'
        )
    blocks = []
    if best.get("prompt"):
        blocks.append(render_file_block("prompt.md", best["prompt"], False))
    if best.get("agents"):
        blocks.append(render_file_block("AGENTS.md", best["agents"], False))
    for skill_name, content in sorted(best.get("skills", {}).items()):
        blocks.append(render_file_block(skill_name, content, True))
    blocks_html = "".join(blocks) or '<span class="evo-candidate-sub">no prompt/AGENTS/skill files</span>'
    return (
        '<div class="evo-candidate kept test-agent-card">'
        f'<div class="evo-candidate-hdr">{html.escape(exp["name"])}</div>'
        f'<div class="evo-files">{blocks_html}</div></div>'
    )


def render_overview(experiments: list[dict]) -> str:
    cards = "".join(render_agent_card(e) for e in experiments)
    return (
        "<section>"
        '<div class="section-title">Agent definitions</div>'
        f'<div class="test-agent-cards-row">{cards}</div>'
        "</section>"
    )


def render_text_comparison(title: str, experiments: list[dict], field: str) -> str:
    if not any(e["best"] is not None and e["best"].get(field) for e in experiments):
        return ""
    # experiments[0] (the seed, when found) is always the baseline, even if
    # its own text for this field is empty — the diffs still need a fixed
    # reference point rather than whichever experiment has content first.
    baseline = experiments[0]
    baseline_text = (baseline["best"] or {}).get(field, "")
    baseline_body = (
        plain_lines(baseline_text) if baseline_text
        else '<div class="diff-line ctx missing">(empty)</div>'
    )
    panels = [
        f'<div class="text-panel"><div class="text-panel-hdr">{html.escape(baseline["name"])} '
        f'<span class="baseline-tag">baseline</span></div>'
        f'<div class="diff-body">{baseline_body}</div></div>'
    ]
    for exp in experiments[1:]:
        if exp["best"] is None:
            continue
        text = exp["best"].get(field, "")
        if not text:
            continue
        panels.append(
            f'<div class="text-panel"><div class="text-panel-hdr">{html.escape(exp["name"])}</div>'
            f'<div class="diff-body">{diff_lines(baseline_text, text)}</div></div>'
        )
    return f'<section><h2>{title}</h2><div class="text-panels">{"".join(panels)}</div></section>'


def render_skill_stack(exp: dict) -> str:
    """One experiment's own skills, listed in full one under another. Skill
    names aren't comparable across experiments (each run invents its own), so
    there's no cross-experiment diffing here — just a plain per-experiment
    listing, and experiments with no skills contribute no panel at all."""
    skills = (exp["best"] or {}).get("skills") if exp["best"] else None
    if not skills:
        return ""
    blocks = []
    for name, content in sorted(skills.items()):
        blocks.append(f'<div class="skill-name">{html.escape(name)}</div>')
        blocks.append(f'<div class="skill-body">{plain_lines(content)}</div>')
    return (
        f'<div class="text-panel"><div class="text-panel-hdr">{html.escape(exp["name"])}</div>'
        f'<div class="skill-stack">{"".join(blocks)}</div></div>'
    )


def render_skills_comparison(experiments: list[dict]) -> str:
    panels = [p for p in (render_skill_stack(e) for e in experiments) if p]
    if not panels:
        return ""
    return f'<section><h2>Skills</h2><div class="text-panels">{"".join(panels)}</div></section>'


_STYLES = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #ffffff; color: #24292f; margin: 0; padding: 24px 32px 60px;
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 28px 0 10px; color: #1a1a1a; }
.subtitle { color: #6e7781; font-size: 13px; margin-bottom: 24px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #f6f8fa; font-weight: 600; color: #57606a; }
.exp-name { font-weight: 600; }
.val-avg { font-weight: 700; color: #0969da; }
.task-id { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11.5px; }
.no-data { color: #9198a1; font-style: italic; }
.text-panels { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 4px; }
.text-panel {
  flex: 1 1 0; min-width: 340px; border: 1px solid #d0d7de; border-radius: 7px;
  overflow: hidden; background: #ffffff;
}
.text-panel-hdr {
  background: #f6f8fa; border-bottom: 1px solid #d0d7de; padding: 6px 10px;
  font-size: 12px; font-weight: 600;
}
.baseline-tag {
  font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
  background: #ddf4ff; color: #0969da; border-radius: 9px; padding: 1px 6px; margin-left: 4px;
}
.diff-body {
  max-height: 520px; overflow: auto; font-family: 'JetBrains Mono', Consolas, monospace;
  font-size: 11.5px; line-height: 1.5;
}
.diff-line { white-space: pre-wrap; word-break: break-word; padding: 0 8px; }
.diff-line.ctx { background: #ffffff; }
.diff-line.add { background: #dafbe1; color: #1a7f37; }
.diff-line.del { background: #ffebe9; color: #cf222e; }
.diff-line.missing { color: #9198a1; font-style: italic; }
.skill-stack { max-height: 640px; overflow-y: auto; }
.skill-name {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
  color: #57606a; background: #f6f8fa; padding: 5px 10px;
  border-top: 1px solid #d0d7de; border-bottom: 1px solid #d0d7de;
}
.skill-name:first-child { border-top: none; }
.skill-body {
  font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11.5px; line-height: 1.5;
}
.section-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
  color: #57606a; margin-bottom: 8px; margin-top: 22px;
}
.test-agent-cards-row { display: flex; gap: 12px; margin-top: 16px; align-items: flex-start; }
.test-agent-card { flex: 1; min-width: 0; cursor: default; }
.evo-candidate {
  position: relative; border: 2px solid #d0d7de; border-radius: 8px;
  padding: 12px 16px; background: #fff;
}
.evo-candidate.kept { border-color: #2da44e; background: #f7fdf8; }
.evo-candidate-hdr { font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 0; }
.evo-candidate-sub { font-size: 10px; color: #57606a; font-weight: 400; margin-left: 6px; }
.evo-files { display: flex; flex-direction: column; gap: 5px; margin-top: 8px; }
.evo-file-block {
  height: 32px; border-radius: 5px; border: 1px solid;
  display: inline-flex; align-items: center; padding: 0 8px;
  font-size: 10px; font-family: 'JetBrains Mono', Consolas, monospace;
  white-space: nowrap; cursor: default;
}
.tab-bar { display: flex; gap: 0; margin-bottom: 20px; border-bottom: 1px solid #d0d7de; }
.tab {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: #57606a; font-size: 13px; font-family: inherit;
  padding: 7px 18px; cursor: pointer; margin-bottom: -1px;
}
.tab:hover { color: #24292f; }
.tab.sel { color: #1a1a1a; border-bottom-color: #f06633; font-weight: 600; }
.no-diff {
  background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px;
  padding: 14px 16px; font-size: 13px; color: #9198a1; font-style: italic;
}
.bp-title { font-size: 15px; font-weight: 600; color: #24292f; margin: 0 0 14px; }
.bp-panels {
  display: flex; flex-direction: row; align-items: flex-start; flex-wrap: nowrap;
  gap: 4px; background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px;
  padding: 20px 20px 12px; overflow-x: auto;
}
.bp-panel { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; }
.bp-panel-hdr { font-size: 12.5px; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; }
.bp-axis-lbl { font-size: 10.5px; fill: #898781; }
.bp-value-lbl { font-size: 12px; font-weight: 700; fill: #52514e; }
.bp-dot { cursor: pointer; transition: r 0.12s ease, stroke-width 0.12s ease; }
.bp-dot:hover { r: 7.5; stroke-width: 3; }
.bp-caption { font-size: 11.5px; color: #6e7781; margin-top: 10px; font-style: italic; }
.bp-tooltip {
  position: fixed; pointer-events: none; z-index: 50; transform: translate(-50%, -135%);
  background: #24292f; color: #ffffff; font-size: 11px; font-weight: 600;
  padding: 4px 8px; border-radius: 5px; white-space: nowrap;
}
.split-section { margin-top: 44px; }
.split-section:first-child { margin-top: 0; }
.pf-grid {
  display: flex; flex-direction: row; align-items: flex-start; flex-wrap: nowrap;
  gap: 22px; margin-top: 20px; overflow-x: auto; padding-bottom: 4px;
}
.pf-block { flex: 0 0 auto; }
.pf-block-hdr { font-size: 12px; font-weight: 700; color: #1a1a1a; margin-bottom: 6px; }
.pf-table-wrap { overflow-x: auto; }
.pf-table { border-collapse: collapse; font-size: 11px; width: auto; }
.pf-table th, .pf-table td { border: 1px solid #ffffff; padding: 0; text-align: center; }
.pf-table th {
  background: #f6f8fa; padding: 4px 8px; font-weight: 600; color: #57606a;
  border: 1px solid #d0d7de;
}
.pf-task {
  border: 1px solid #d0d7de !important; padding: 3px 10px; text-align: left;
  white-space: nowrap; background: #f6f8fa;
}
.pf-cell { width: 22px; height: 18px; cursor: default; }
.pf-cell.pf-solved   { background: #0ca30c; }
.pf-cell.pf-unsolved { background: #d03b3b; }
.pf-cell.pf-na        { background: #e1e0d9; }
.pf-task.pf-driver {
  background: #fff8c5; border-color: #d4a72c !important; font-weight: 700;
  position: relative;
}
.pf-task.pf-driver::before {
  content: "\\2605"; color: #9a6700; margin-right: 4px; font-size: 10px;
}
tr.pf-driver-row .pf-cell { box-shadow: inset 0 0 0 1px #d4a72c; }
.pf-task.pf-pareto {
  background: #cde2fb; border-color: #2a78d6 !important; font-weight: 700;
}
.pf-task.pf-pareto::after {
  content: "\\2191"; color: #184f95; margin-left: 4px; font-size: 10px; font-weight: 900;
}
tr.pf-pareto-row .pf-cell { box-shadow: inset 0 0 0 1px #2a78d6; }
.pf-task.pf-driver.pf-pareto { background: linear-gradient(90deg, #fff8c5 50%, #cde2fb 50%); }
.pf-legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 4px; font-size: 11px; color: #57606a; }
.pf-legend-item { display: flex; align-items: center; gap: 5px; }
.pf-legend-swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px; border: 1px solid; }
.pf-legend-swatch.pf-driver { background: #fff8c5; border-color: #d4a72c; }
.pf-legend-swatch.pf-pareto { background: #cde2fb; border-color: #2a78d6; }
"""


def render_split_tab_body(groups: list[dict], split_label: str) -> str:
    groups = groups or []
    return (
        '<section class="split-section">' + render_run_summary_section(groups, split_label) + "</section>"
        '<section class="split-section">' + render_boxplot_section(groups, split_label) + "</section>"
        '<section class="split-section">' + render_pass_fail_section(groups, split_label) + "</section>"
    )


_TABS = [
    ("summary", "Summary"),
    ("train", "Train"),
    ("val", "Val"),
    ("test", "Test"),
]


def render_html(
    experiments: list[dict],
    train_groups: list[dict] | None = None,
    val_groups: list[dict] | None = None,
    test_groups: list[dict] | None = None,
) -> str:
    names = ", ".join(e["name"] for e in experiments)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_body = [
        render_overview(experiments),
        render_text_comparison("Prompt (prompt.md)", experiments, "prompt"),
        render_text_comparison("AGENTS.md", experiments, "agents"),
        render_skills_comparison(experiments),
    ]
    tab_bodies = {
        "summary": "".join(summary_body),
        "train": render_split_tab_body(train_groups, "Train"),
        "val": render_split_tab_body(val_groups, "Val"),
        "test": render_split_tab_body(test_groups, "Test"),
    }
    tab_buttons = "\n".join(
        f'  <button class="tab{" sel" if key == "summary" else ""}" id="tab-btn-{key}" '
        f'onclick="showTab(\'{key}\')">{label}</button>'
        for key, label in _TABS
    )
    tab_panels = "\n".join(
        f'<div id="tab-{key}"{"" if key == "summary" else " hidden"}>{tab_bodies[key]}</div>'
        for key, _ in _TABS
    )
    tab_names_js = ", ".join(f"'{key}'" for key, _ in _TABS)
    body = f"""
<h1>Best Candidate Comparison</h1>
<div class="subtitle">{html.escape(names)} — generated {generated}</div>
<div class="tab-bar">
{tab_buttons}
</div>
{tab_panels}
<div id="bp-tooltip" class="bp-tooltip" hidden></div>
<script>
const TABS = [{tab_names_js}];
function showTab(name) {{
  TABS.forEach(t => {{
    document.getElementById('tab-' + t).hidden = t !== name;
    document.getElementById('tab-btn-' + t).classList.toggle('sel', t === name);
  }});
}}
function bpTipShow(evt,text){{
  var t=document.getElementById('bp-tooltip');
  t.textContent=text; t.style.left=evt.clientX+'px'; t.style.top=evt.clientY+'px'; t.hidden=false;
}}
function bpTipMove(evt){{
  var t=document.getElementById('bp-tooltip');
  t.style.left=evt.clientX+'px'; t.style.top=evt.clientY+'px';
}}
function bpTipHide(){{ document.getElementById('bp-tooltip').hidden=true; }}
</script>"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Best Candidate Comparison</title>"
        f"<style>{_STYLES}</style></head><body>{body}</body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path, help="Experiment directories to compare")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML file path")
    args = parser.parse_args()

    experiments = [build_experiment_data(exp_dir) for exp_dir in args.experiments]

    seed = None
    for exp_dir in args.experiments:
        seed = load_seed_entry(exp_dir)
        if seed is not None:
            break
    if seed is not None:
        experiments = [seed] + experiments
    else:
        print("[best-comparison] warning: no seed (iteration_000) found, falling back to "
              f"{experiments[0]['name']} as the diff baseline")

    for exp in experiments:
        best = exp["best"]
        if best is None:
            print(f"[best-comparison] {exp['name']}: no candidates found")
        else:
            print(
                f"[best-comparison] {exp['name']}: iteration {best['iteration']}, "
                f"candidate {best['idx']}, val_avg {best['val_avg'] * 100:.1f}%"
            )

    output = args.output
    if output is None:
        output = args.experiments[0].resolve().parent / "best-comparison.html"

    experiments_root = args.experiments[0].resolve().parent
    split_groups = {}
    for split in ("train", "val", "test"):
        groups = build_run_groups(experiments_root, args.experiments, split)
        if not groups:
            print(f"[best-comparison] {split}-split run data: none found")
        split_groups[split] = groups

    output.write_text(
        render_html(experiments, split_groups["train"], split_groups["val"], split_groups["test"]),
        encoding="utf-8",
    )
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
