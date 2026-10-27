#!/usr/bin/env python3
"""Compare a successful run's context window against a failed run's context
window, for challenges with inconsistent outcomes across a multi-run
experiment (e.g. the seed baseline, which has run1..run5 per task), and
render a tabbed HTML report pinpointing where and why the two runs diverge.

Expected layout per run: <experiment_dir>/run*/<split>/<task_id>/{metadata.json,
context_window.json} — i.e. the seed baseline layout.

Usage:
  python3 context_divergence.py <experiment_dir> --tasks TASK_ID [TASK_ID ...]
  python3 context_divergence.py <experiment_dir> --n-random 3 [--seed 42]
  python3 context_divergence.py <experiment_dir> --tasks TASK_ID ... \
      --explanations-file explanations.json

Output defaults to experiments/<experiment_dir basename>_inconsistencies.html.
"""
import argparse
import difflib
import html
import json
import math
import random
import re
from pathlib import Path

_DIGIT_RE = re.compile(r"\d+")
_FLAG_RE = re.compile(r"-[a-zA-Z]+\b")
_DIVERGENCE_SIMILARITY_THRESHOLD = 0.4


def discover_runs(experiment_dir: Path) -> list[Path]:
    return sorted(
        (p for p in experiment_dir.glob("run*") if p.is_dir()),
        key=lambda p: p.name,
    )


def load_task_metadata(run_dirs: list[Path], split: str) -> dict[str, dict[str, dict]]:
    """task_id -> run_name -> metadata dict (only for tasks present in the split)."""
    tasks: dict[str, dict[str, dict]] = {}
    for run_dir in run_dirs:
        split_dir = run_dir / split
        if not split_dir.exists():
            continue
        for task_dir in sorted(split_dir.iterdir()):
            meta_path = task_dir / "metadata.json"
            if not task_dir.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            tasks.setdefault(task_dir.name, {})[run_dir.name] = meta
    return tasks


def find_inconsistent_tasks(tasks: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Subset of tasks whose runs don't all share the same success value."""
    result = {}
    for task_id, by_run in tasks.items():
        outcomes = {meta.get("success") for meta in by_run.values()}
        if len(outcomes) > 1:
            result[task_id] = by_run
    return result


def pick_pair(by_run: dict[str, dict]) -> tuple[str, str] | None:
    """Deterministic default pair: the successful run with the fewest steps
    (cleanest solve) vs the failed run with the highest total token cost
    (the one that diverged hardest). Returns (success_run, fail_run) run
    names, or None if either side is empty."""
    solved = [(name, m) for name, m in by_run.items() if m.get("success")]
    failed = [(name, m) for name, m in by_run.items() if not m.get("success")]
    if not solved or not failed:
        return None
    success_run = min(solved, key=lambda nm: nm[1].get("iterations_used") or math.inf)[0]
    fail_run = max(failed, key=lambda nm: nm[1].get("total_tokens") or 0)[0]
    return success_run, fail_run


def load_trajectory(run_dir: Path, split: str, task_id: str) -> list[dict]:
    """Parse context_window.json into a per-step list with cumulative tokens."""
    cw_path = run_dir / split / task_id / "context_window.json"
    if not cw_path.exists():
        return []
    steps = json.loads(cw_path.read_text(encoding="utf-8"))
    trajectory = []
    cumulative = 0
    for i, step in enumerate(steps):
        tok = step.get("tokens") or {}
        cumulative += (tok.get("input") or 0) + (tok.get("cache_read") or 0)
        cmd = step.get("input")
        command = cmd.get("command") if isinstance(cmd, dict) else cmd
        trajectory.append(
            {
                "step": i,
                "tool": step.get("tool"),
                "command": command or "",
                "thought": (step.get("thought") or "").strip(),
                "cum_tokens": cumulative,
            }
        )
    return trajectory


def _normalize_command(cmd: str) -> str:
    """Strip digit runs (ports, hashes, guessed ids) and short flags (-v, -s,
    -I, ...) so commands that differ only by the per-run randomized port or
    incidental curl flags still compare as similar."""
    cmd = _DIGIT_RE.sub("#", cmd or "")
    cmd = _FLAG_RE.sub("", cmd)
    return re.sub(r"\s+", " ", cmd).strip().lower()


def find_first_divergence(success_traj: list[dict], fail_traj: list[dict]) -> int | None:
    """Index of the first step whose commands are no longer similar enough to
    call "the same action" between the two trajectories (fuzzy match, so
    wording/flag differences on an otherwise-identical recon step don't count
    as a divergence), or None if none found in the overlapping range."""
    for i in range(min(len(success_traj), len(fail_traj))):
        a = _normalize_command(success_traj[i]["command"])
        b = _normalize_command(fail_traj[i]["command"])
        similarity = difflib.SequenceMatcher(None, a, b).ratio()
        if similarity < _DIVERGENCE_SIMILARITY_THRESHOLD:
            return i
    return None


def build_task_report(
    experiment_dir: Path,
    split: str,
    task_id: str,
    by_run: dict[str, dict],
    content: dict | str | None,
) -> dict | None:
    """content is either a plain explanation string, or a dict with optional
    "explanation" (str) and "agent_diff" (dict, see _render_agent_diff) keys —
    both are external data, not computed by this script, since isolating which
    part of a multi-task patch belongs to one specific challenge requires
    reading the reflector's reasoning, not just diffing files mechanically."""
    pair = pick_pair(by_run)
    if pair is None:
        return None
    success_run, fail_run = pair
    success_traj = load_trajectory(experiment_dir / success_run, split, task_id)
    fail_traj = load_trajectory(experiment_dir / fail_run, split, task_id)
    if not success_traj or not fail_traj:
        return None
    divergence_step = find_first_divergence(success_traj, fail_traj)
    if isinstance(content, dict):
        explanation = content.get("explanation") or ""
        agent_diff = content.get("agent_diff")
    else:
        explanation = content or ""
        agent_diff = None
    return {
        "task_id": task_id,
        "success_run": success_run,
        "fail_run": fail_run,
        "success_meta": by_run[success_run],
        "fail_meta": by_run[fail_run],
        "success_traj": success_traj,
        "fail_traj": fail_traj,
        "divergence_step": divergence_step,
        "explanation": explanation,
        "agent_diff": agent_diff,
    }


_STYLES = """
.viz-root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page-plane:     #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --status-good:     #0ca30c;
  --status-critical: #d03b3b;
  --status-info:     #0969da;
  --status-accent:   #8250df;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .viz-root {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border: rgba(255,255,255,0.10);
    --status-info:     #6cb6ff;
    --status-accent:   #c297ff;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page-plane:     #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border: rgba(255,255,255,0.10);
  --status-info:     #6cb6ff;
  --status-accent:   #c297ff;
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page-plane);
  color: var(--text-primary);
  padding: 24px 16px 48px;
}
.viz-header { max-width: 1100px; margin: 0 auto 20px; }
.viz-header h1 { font-size: 22px; margin: 0 0 4px; }
.viz-header p { color: var(--text-secondary); margin: 0; font-size: 14px; }
.tabs {
  max-width: 1100px; margin: 0 auto 16px;
  display: flex; gap: 4px; border-bottom: 1px solid var(--gridline);
  flex-wrap: wrap;
}
.tab-btn {
  appearance: none; border: none; background: transparent; cursor: pointer;
  padding: 10px 16px; font-size: 14px; font-weight: 600;
  color: var(--text-secondary); border-bottom: 2px solid transparent;
  font-family: inherit;
}
.tab-btn.active { color: var(--text-primary); border-bottom-color: var(--status-good); }
.tab-panel { max-width: 1100px; margin: 0 auto; }
.tab-panel[hidden] { display: none; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 20px; margin-bottom: 16px;
}
.card h2 { font-size: 16px; margin: 0 0 12px; }
.run-summary { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px; font-size: 13px; }
.run-summary .run-stat { min-width: 200px; }
.run-stat .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.run-stat.solved .swatch { background: var(--status-good); }
.run-stat.failed .swatch { background: var(--status-critical); }
.run-stat .label { font-weight: 700; }
.run-stat .meta { color: var(--text-secondary); }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }
.legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.legend .solved .swatch { background: var(--status-good); }
.legend .failed .swatch { background: var(--status-critical); }
.explanation { font-size: 13px; line-height: 1.6; color: var(--text-primary); margin: 12px 0 16px; }
table.cmd-compare { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
table.cmd-compare th {
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline);
  color: var(--text-muted); font-weight: 700; position: sticky; top: 0; background: var(--surface-1);
}
table.cmd-compare th.step-col { width: 44px; }
table.cmd-compare th.solved-col { color: var(--status-good); }
table.cmd-compare th.failed-col { color: var(--status-critical); }
table.cmd-compare td {
  padding: 6px 10px; border-bottom: 1px solid var(--gridline); vertical-align: top;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; white-space: pre-wrap; word-break: break-word;
}
table.cmd-compare td.step-col { font-family: inherit; color: var(--text-muted); text-align: right; }
table.cmd-compare td.empty { color: var(--text-muted); font-family: inherit; }
table.cmd-compare tr.diverge-row td { background: color-mix(in srgb, var(--status-critical) 10%, transparent); border-top: 2px solid var(--status-critical); border-bottom: 2px solid var(--status-critical); }
table.cmd-compare tr.diverge-row td.step-col::after { content: " \\2190 diverge"; color: var(--status-critical); font-weight: 700; }
.cmd-table-wrap { max-height: 640px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
.agent-diff-section { margin: 4px 0 20px; }
.agent-diff-title { font-size: 13px; font-weight: 700; margin: 0 0 4px; }
.agent-diff-iter { font-size: 11px; color: var(--text-muted); font-family: ui-monospace, monospace; margin-bottom: 8px; }
.agent-diff-note {
  font-size: 12px; color: var(--text-secondary); background: color-mix(in srgb, var(--text-muted) 10%, transparent);
  border-left: 3px solid var(--text-muted); padding: 6px 10px; border-radius: 4px; margin-bottom: 10px;
}
.diff-file { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 10px; }
.diff-file-header {
  background: color-mix(in srgb, var(--text-muted) 12%, transparent); padding: 6px 10px;
  font-family: ui-monospace, monospace; font-size: 12px; font-weight: 700; display: flex; gap: 8px; align-items: center;
}
.diff-file-status { font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 4px; text-transform: uppercase; }
.diff-file-status.new { background: color-mix(in srgb, var(--status-good) 18%, transparent); color: var(--status-good); }
.diff-file-status.modified { background: color-mix(in srgb, var(--status-critical) 12%, transparent); color: var(--text-secondary); }
.diff-body { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; overflow-x: auto; }
.diff-line { padding: 1px 10px; white-space: pre-wrap; word-break: break-word; }
.diff-line.meta { color: var(--text-muted); background: color-mix(in srgb, var(--text-muted) 6%, transparent); }
.diff-line.hunk { color: var(--text-muted); background: color-mix(in srgb, var(--status-good) 6%, transparent); }
.diff-line.add { background: color-mix(in srgb, var(--status-good) 14%, transparent); color: var(--text-primary); }
.diff-line.del { background: color-mix(in srgb, var(--status-critical) 14%, transparent); color: var(--text-primary); }
.diff-line.ctx { color: var(--text-secondary); }
"""


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _diff_lines_html(a_lines: list[str], b_lines: list[str], fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="")
    out = []
    for line in diff:
        text = html.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            out.append(f'<div class="diff-line meta">{text}</div>')
        elif line.startswith("@@"):
            out.append(f'<div class="diff-line hunk">{text}</div>')
        elif line.startswith("+"):
            out.append(f'<div class="diff-line add">{text}</div>')
        elif line.startswith("-"):
            out.append(f'<div class="diff-line del">{text}</div>')
        else:
            out.append(f'<div class="diff-line ctx">{text}</div>')
    return "".join(out)


def _render_agent_diff(agent_diff: dict | None) -> str:
    """Git-diff-style view of the specific agent-definition change(s)
    attributable to this challenge — the file(s) the reflector's patch
    touched because of THIS task's diagnosis, isolated from any other task
    that shared the same 3-task minibatch at that iteration (that isolation
    is supplied as data, not computed here — see build_task_report)."""
    if not agent_diff:
        return ""
    file_blocks = []
    for f in agent_diff.get("files", []):
        path = f["path"]
        if f.get("status") == "new":
            new_lines = f["new_text"].splitlines()
            body = _diff_lines_html([], new_lines, "/dev/null", f"b/{path}")
            status_label = "new"
        else:
            parent_lines = f.get("parent_text", "").splitlines()
            child_lines = parent_lines + [f["added_line"]]
            body = _diff_lines_html(parent_lines, child_lines, f"a/{path}", f"b/{path}")
            status_label = "modified"
        file_blocks.append(
            f'<div class="diff-file"><div class="diff-file-header">'
            f'<span class="diff-file-status {status_label}">{status_label}</span>'
            f'<span>{html.escape(path)}</span></div>'
            f'<div class="diff-body">{body}</div></div>'
        )
    note_html = f'<div class="agent-diff-note">{html.escape(agent_diff["note"])}</div>' if agent_diff.get("note") else ""
    iter_html = f'<div class="agent-diff-iter">{html.escape(agent_diff["iteration"])}</div>' if agent_diff.get("iteration") else ""
    return f"""
    <div class="agent-diff-section">
      <div class="agent-diff-title">How this challenge changed the agent definition</div>
      {iter_html}
      {note_html}
      {"".join(file_blocks)}
    </div>
    """


def _render_command_table(report: dict) -> str:
    """Row-per-step table: the raw bash command issued by the successful run
    next to the raw command issued by the failed run, at the same step index.
    The row where the two commands first stop matching is highlighted."""
    success_traj = report["success_traj"]
    fail_traj = report["fail_traj"]
    div_step = report["divergence_step"]
    n_rows = max(len(success_traj), len(fail_traj))

    def cell(traj: list[dict], i: int) -> str:
        if i >= len(traj):
            return '<td class="empty">&mdash;</td>'
        cmd = traj[i]["command"]
        if not cmd:
            return '<td class="empty">&mdash;</td>'
        return f"<td>{html.escape(cmd)}</td>"

    rows = []
    for i in range(n_rows):
        row_cls = ' class="diverge-row"' if i == div_step else ""
        rows.append(
            f"<tr{row_cls}><td class=\"step-col\">{i}</td>{cell(success_traj, i)}{cell(fail_traj, i)}</tr>"
        )

    return f"""
    <div class="cmd-table-wrap">
    <table class="cmd-compare">
      <thead><tr>
        <th class="step-col">#</th>
        <th class="solved-col">{html.escape(report['success_run'])} &mdash; solved ({len(success_traj)} steps)</th>
        <th class="failed-col">{html.escape(report['fail_run'])} &mdash; failed ({len(fail_traj)} steps)</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    """


def _run_stat_html(run_name: str, meta: dict, cls: str) -> str:
    steps = meta.get("iterations_used")
    tokens = meta.get("total_tokens")
    stop = meta.get("stop_reason")
    return (
        f'<div class="run-stat {cls}"><span class="swatch"></span>'
        f'<span class="label">{html.escape(run_name)} &mdash; {"solved" if cls == "solved" else "failed"}</span><br/>'
        f'<span class="meta">{steps} steps &middot; {_fmt_int(tokens or 0)} tokens &middot; {html.escape(str(stop))}</span></div>'
    )


def render_task_panel(report: dict, index: int, active: bool) -> str:
    task_id = report["task_id"]

    explanation_html = (
        f'<div class="explanation">{html.escape(report["explanation"])}</div>' if report["explanation"] else ""
    )

    return f"""
    <div class="tab-panel" id="panel-{index}" role="tabpanel" {'' if active else 'hidden'}>
      <div class="card">
        <h2>{html.escape(task_id)}</h2>
        <div class="run-summary">
          {_run_stat_html(report['success_run'], report['success_meta'], 'solved')}
          {_run_stat_html(report['fail_run'], report['fail_meta'], 'failed')}
        </div>
        {explanation_html}
        {_render_agent_diff(report.get("agent_diff"))}
        {_render_command_table(report)}
      </div>
    </div>
    """


def render_html(experiment_name: str, reports: list[dict]) -> str:
    tabs = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-target="panel-{i}">{html.escape(r["task_id"])}</button>'
        for i, r in enumerate(reports)
    )
    panels = "".join(render_task_panel(r, i, i == 0) for i, r in enumerate(reports))

    script = """
    <script>
    function scrollToDivergence(panel) {
      const row = panel.querySelector('.diverge-row');
      if (row) row.scrollIntoView({ block: 'center' });
    }
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.hidden = true);
        btn.classList.add('active');
        const panel = document.getElementById(btn.dataset.target);
        panel.hidden = false;
        scrollToDivergence(panel);
      });
    });
    const initialPanel = document.querySelector('.tab-panel:not([hidden])');
    if (initialPanel) scrollToDivergence(initialPanel);
    </script>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(experiment_name)} &mdash; run divergence</title>
<style>{_STYLES}</style>
</head>
<body>
<div class="viz-root">
  <div class="viz-header">
    <h1>{html.escape(experiment_name)} &mdash; why runs diverge</h1>
    <p>For each inconsistent challenge: one successful run vs. one failed run of the same challenge, compared step by step.</p>
  </div>
  <div class="tabs" role="tablist">{tabs}</div>
  {panels}
</div>
{script}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", nargs="+", default=None, help="Explicit task ids to include")
    parser.add_argument("--n-random", type=int, default=None, help="Randomly sample N inconsistent tasks")
    parser.add_argument("--seed", type=int, default=None, help="Seed for --n-random sampling")
    parser.add_argument(
        "--explanations-file",
        type=Path,
        default=None,
        help=(
            "JSON file: {task_id: explanation text} or {task_id: {explanation, agent_diff}}. "
            "agent_diff (optional) = {iteration, note, files: [{path, status: new|modified, "
            "new_text} or {path, status: modified, parent_text, added_line}]} — the specific "
            "agent-definition change(s) attributable to this task, rendered as a git diff."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    run_dirs = discover_runs(experiment_dir)
    if not run_dirs:
        raise SystemExit(f"No run* directories found under {experiment_dir}")

    tasks = load_task_metadata(run_dirs, args.split)
    inconsistent = find_inconsistent_tasks(tasks)
    if not inconsistent:
        raise SystemExit(f"No inconsistent tasks found under {experiment_dir} ({args.split})")

    if args.tasks:
        selected = {tid: inconsistent[tid] for tid in args.tasks if tid in inconsistent}
        missing = set(args.tasks) - set(selected)
        if missing:
            print(f"Warning: not inconsistent (or not found), skipping: {sorted(missing)}")
    elif args.n_random:
        rng = random.Random(args.seed)
        chosen_ids = rng.sample(sorted(inconsistent), min(args.n_random, len(inconsistent)))
        selected = {tid: inconsistent[tid] for tid in chosen_ids}
    else:
        selected = inconsistent

    explanations = {}
    if args.explanations_file and args.explanations_file.exists():
        explanations = json.loads(args.explanations_file.read_text(encoding="utf-8"))

    reports = []
    for task_id, by_run in selected.items():
        report = build_task_report(experiment_dir, args.split, task_id, by_run, explanations.get(task_id))
        if report is not None:
            reports.append(report)
        else:
            print(f"Warning: could not build report for {task_id} (missing trajectory data)")

    if not reports:
        raise SystemExit("No reports could be built.")

    experiment_name = experiment_dir.name
    output = args.output or (Path("experiments") / f"{experiment_name}_inconsistencies.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(experiment_name, reports), encoding="utf-8")
    print(f"Wrote {output} ({len(reports)} tasks)")


if __name__ == "__main__":
    main()
