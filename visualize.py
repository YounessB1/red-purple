#!/usr/bin/env python3
"""Generate a self-contained HTML prompt viewer for a red-purple experiment.
Usage: python visualize.py [experiment_dir]
Output: <experiment_dir>/viewer.html
"""
import json
import sys
from pathlib import Path


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

        parent_prompt = ""
        child_prompt  = ""
        parent_train  = {}
        parent_val    = {}
        child_train   = {}
        child_val     = None
        parent_idx    = None
        evo_path = iter_dir / "evolution.json"
        if evo_path.exists():
            evo = json.loads(evo_path.read_text(encoding="utf-8"))
            parent_prompt = evo.get("parent", {}).get("prompt", "")
            child_prompt  = evo.get("child",  {}).get("prompt", "")
            parent_train  = evo.get("parent", {}).get("train") or {}
            parent_val    = evo.get("parent", {}).get("val")   or {}
            child_train   = evo.get("child",  {}).get("train") or {}
            child_val     = evo.get("child",  {}).get("val")
            parent_idx    = evo.get("parent", {}).get("candidate_idx")

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

        changes = ""
        rc_path = iter_dir / "reflector_changes.json"
        if rc_path.exists():
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            changes = rc.get("changes", "")

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
            "parent_prompt":    parent_prompt,
            "child_prompt":     child_prompt,
            "changes":          changes,
            "val_ok":           val_ok,
            "val_total":        val_total,
            "parent_train":     parent_train,
            "parent_val":       parent_val,
            "child_train":      child_train,
            "child_val":        child_val,
            "parent_idx":       parent_idx,
            "pool_json":        pool_json,
            "reflector_json":   reflector_json,
            "reflector_steps":  reflector_steps,
        })

    config = {}
    config_path = exp_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    summary = {}
    summary_path = exp_dir / "experiment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    return {"name": exp_dir.name, "iterations": iterations, "config": config, "summary": summary}


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__NAME__ — Prompt Explorer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0d1117; color: #c9d1d9;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}
#hdr {
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 10px 18px; display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
#hdr h1 { font-size: 15px; font-weight: 600; color: #e6edf3; }
#hdr .hint { font-size: 12px; color: #6e7681; }
#body { display: flex; flex: 1; overflow: hidden; }
#sidebar {
  width: 210px; flex-shrink: 0; background: #161b22;
  border-right: 1px solid #30363d; overflow-y: auto; padding: 8px;
}
#main { flex: 1; overflow-y: auto; padding: 18px 24px; }
#info-sidebar {
  width: 310px; flex-shrink: 0; background: #161b22;
  border-left: 1px solid #30363d; overflow-y: auto; padding: 12px;
}
.info-panel {
  background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
  margin-bottom: 12px; overflow: hidden;
}
.info-panel-hdr {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
  color: #6e7681; padding: 7px 12px; background: #1c2128;
  border-bottom: 1px solid #30363d;
}
.info-row {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 8px; padding: 5px 12px; border-bottom: 1px solid #161b22;
  font-size: 12.5px;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #6e7681; white-space: nowrap; flex-shrink: 0; }
.info-val { color: #c9d1d9; text-align: right; word-break: break-all; }
.info-val.mono { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11.5px; }
.info-sub-hdr {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  color: #484f58; padding: 6px 12px 3px; background: #0d1117;
}
.info-cost { color: #d29922; }
.info-calls { color: #79c0ff; }
.card {
  border: 1px solid #30363d; border-radius: 7px; padding: 9px 11px;
  margin-bottom: 5px; cursor: pointer;
  transition: border-color 0.12s, background 0.12s;
}
.card:hover { border-color: #58a6ff; background: #1c2128; }
.card.sel   { border-color: #1f6feb; background: #1c2128; box-shadow: 0 0 0 2px #1f6feb55; }
.card-top   { display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }
.card-num   { font-size: 12px; font-weight: 600; color: #e6edf3; }
.badge {
  font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 9px;
  letter-spacing: 0.4px; text-transform: uppercase;
}
.b-accepted { background: #1a4228; color: #3fb950; border: 1px solid #238636; }
.b-rejected { background: #3d1a1a; color: #f85149; border: 1px solid #da3633; }
.b-pending  { background: #2d2516; color: #d29922; border: 1px solid #9e6a03; }
.card-score { font-size: 10px; color: #6e7681; }
#placeholder {
  height: 100%; display: flex; align-items: center; justify-content: center;
  color: #3d444d; font-size: 14px;
}
.section-title {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;
  color: #6e7681; margin-bottom: 8px; margin-top: 22px;
}
.section-title:first-child { margin-top: 0; }

/* ── Tab bar ── */
.tab-bar {
  display: flex; gap: 0; margin-bottom: 20px;
  border-bottom: 1px solid #30363d;
}
.tab {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: #8b949e; font-size: 13px; font-family: inherit;
  padding: 7px 18px; cursor: pointer; margin-bottom: -1px;
  transition: color 0.1s;
}
.tab:hover { color: #c9d1d9; }
.tab.sel { color: #e6edf3; border-bottom-color: #f78166; font-weight: 600; }
.json-view {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  padding: 16px; font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px; line-height: 1.65; color: #c9d1d9;
  white-space: pre-wrap; word-break: break-word;
}

.parent-tag {
  display: inline-block; font-size: 11px; color: #79c0ff;
  background: #1b2333; border: 1px solid #1f6feb;
  border-radius: 4px; padding: 2px 9px; margin-bottom: 14px;
  font-family: 'JetBrains Mono', Consolas, monospace;
}

/* ── Scores table ── */
.scores-wrap { display: flex; gap: 16px; margin-bottom: 0; }
.scores-panel {
  flex: 1; background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  overflow: hidden; min-width: 0;
}
.scores-panel-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; background: #1c2128; border-bottom: 1px solid #30363d;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.5px; color: #8b949e;
}
.scores-panel-hdr .s-summary { font-size: 10px; font-weight: 400; color: #6e7681; }
.scores-tbl { width: 100%; border-collapse: collapse; }
.scores-tbl th {
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
  color: #6e7681; padding: 4px 10px; text-align: center; border-bottom: 1px solid #21262d;
}
.scores-tbl th.bench-col { text-align: left; }
.scores-tbl td {
  padding: 3px 10px; font-size: 11.5px;
  font-family: 'JetBrains Mono', Consolas, monospace; border-bottom: 1px solid #161b22;
}
.scores-tbl tr:last-child td { border-bottom: none; }
.scores-tbl td.bench-name { color: #8b949e; }
.s-pass  { color: #3fb950; text-align: center; font-weight: 700; }
.s-mid   { color: #d29922; text-align: center; }
.s-fail  { color: #f85149; text-align: center; }
.s-none  { color: #3d444d; text-align: center; }
.s-delta-pos { color: #3fb950; text-align: center; font-weight: 700; font-size: 13px; }
.s-delta-neg { color: #f85149; text-align: center; font-size: 13px; }
.s-delta-eq  { color: #3d444d; text-align: center; }

/* ── Reflector changes ── */
.changes-box {
  background: #161b22; border: 1px solid #30363d; border-left: 3px solid #d29922;
  border-radius: 6px; padding: 14px 16px;
  font-size: 13px; line-height: 1.7; color: #c9d1d9;
}
.changes-box ul { list-style: none; padding: 0; }
.changes-box li {
  padding: 5px 0 5px 16px; position: relative;
  border-bottom: 1px solid #21262d;
}
.changes-box li:last-child { border-bottom: none; }
.changes-box li::before { content: '•'; position: absolute; left: 0; color: #d29922; }

/* ── Pool view ── */
.pool-header { font-size: 11px; color: #6e7681; margin-bottom: 14px; }
.pool-wrap   { display: flex; flex-wrap: wrap; gap: 14px; }
.pool-card   {
  background: #161b22; border: 1px solid #30363d; border-radius: 7px;
  overflow: hidden; flex: 1; min-width: 260px;
}
.pool-card-hdr {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 14px; background: #1c2128; border-bottom: 1px solid #30363d;
}
.pool-card-title { font-size: 12px; font-weight: 600; color: #e6edf3; }
.pool-avg   { font-size: 22px; font-weight: 700; font-family: 'JetBrains Mono', Consolas, monospace; }
.pool-avg.g { color: #3fb950; } .pool-avg.y { color: #d29922; } .pool-avg.r { color: #f85149; }
.pool-card-body { padding: 10px 14px; }
.pool-row {
  display: flex; align-items: baseline; gap: 8px;
  font-size: 11.5px; padding: 3px 0; border-bottom: 1px solid #21262d;
}
.pool-row:last-child { border-bottom: none; }
.pool-lbl  { color: #6e7681; min-width: 56px; flex-shrink: 0; }
.pool-mval { color: #c9d1d9; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11px; }
.pool-slbl {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.4px; color: #484f58; padding: 8px 0 4px;
}
.pool-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.pool-chip  {
  font-family: 'JetBrains Mono', Consolas, monospace;
  font-size: 11px; padding: 2px 7px; border-radius: 4px; font-weight: 600;
}
.pc-pass { background: #122416; color: #3fb950; border: 1px solid #238636; }
.pc-mid  { background: #2d2516; color: #d29922; border: 1px solid #9e6a03; }
.pc-fail { background: #2d1515; color: #f85149; border: 1px solid #6e1a1a; }
.pc-none { background: #1c2128; color: #484f58; border: 1px solid #30363d; }

/* ── Reflector view ── */
.refl-block {
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  margin-bottom: 14px; overflow: hidden;
}
.refl-block-hdr {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
  color: #6e7681; padding: 5px 14px; background: #1c2128;
  border-bottom: 1px solid #30363d;
}
.refl-body {
  padding: 12px 14px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12px; line-height: 1.65; color: #c9d1d9;
  white-space: pre-wrap; word-break: break-word;
  max-height: 420px; overflow-y: auto;
}
.rh  { color: #79c0ff; font-weight: 700; }
.rf  { color: #3d444d; }

/* ── Unified diff ── */
.diff-meta {
  display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 12px;
}
.diff-meta .stat-del { color: #f85149; }
.diff-meta .stat-ins { color: #3fb950; }
.unified-diff {
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12.5px; line-height: 1.65;
  background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
  overflow: hidden;
}
.u-hunk {
  background: #1b2333; color: #79c0ff;
  padding: 2px 14px; font-size: 11px;
  border-top: 1px solid #21262d; border-bottom: 1px solid #21262d;
}
.u-hunk:first-child { border-top: none; }
.u-del  { background: #2d1515; color: #ffa198; padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-ins  { background: #122416; color: #7ee787; padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-ctx  { background: #0d1117; color: #8b949e;  padding: 0 14px; white-space: pre-wrap; word-break: break-word; }
.u-sign { display: inline-block; width: 14px; font-weight: 700; color: inherit; user-select: none; }
.no-diff {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 14px 16px; font-size: 13px; color: #484f58; font-style: italic;
}

/* ── Reasoning steps ── */
.rsn-feed { display: flex; flex-direction: column; gap: 6px; }
.rsn-step { border-radius: 6px; overflow: hidden; background: #161b22; }
.rsn-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;
  padding: 4px 12px;
}
.rsn-body {
  padding: 9px 14px; font-size: 12.5px; line-height: 1.7; color: #c9d1d9;
  white-space: pre-wrap; word-break: break-word; background: #161b22;
}

/* thinking — light blue */
.rsn-step-thinking { border: 1px solid #58a6ff; }
.rsn-step-thinking .rsn-label { color: #58a6ff; border-bottom: 1px solid #58a6ff; }
.rsn-step-thinking .rsn-body strong { color: #79c0ff; }

/* text — green */
.rsn-step-text { border: 1px solid #3fb950; }
.rsn-step-text .rsn-label { color: #3fb950; border-bottom: 1px solid #3fb950; }

/* tool — orange */
.rsn-step-tool { border: 1px solid #d29922; }
.rsn-step-tool .rsn-label { color: #d29922; border-bottom: 1px solid #d29922; }
.rsn-step-tool .rsn-body  {
  font-family: 'JetBrains Mono', Consolas, monospace; font-size: 12px;
}
.rsn-tool-name { color: #d29922; font-weight: 700; margin-right: 10px; }
.rsn-params { margin-top: 6px; display: flex; flex-direction: column; gap: 2px; }
.rsn-param-row { display: flex; gap: 8px; font-size: 11.5px; }
.rsn-param-key { color: #6e7681; flex-shrink: 0; }
.rsn-param-val { color: #c9d1d9; word-break: break-all; }
</style>
</head>
<body>
<div id="hdr">
  <h1 id="exp-name"></h1>
  <span class="hint">Click an iteration to view reflector changes and prompt diff</span>
</div>
<div id="body">
  <div id="sidebar"></div>
  <div id="main">
    <div id="placeholder">← Select an iteration</div>
    <div id="content" style="display:none"></div>
  </div>
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
  renderIteration(ct, DATA.iterations.find(x => x.id === selId));
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
      rows += `<tr><td class="bench-name">${esc(k)}</td>${scoreCell(p)}${scoreCell(c)}${delta}</tr>`;
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
    if (it.parent_idx != null) {
      body += `<div class="parent-tag">Parent: Candidate #${it.parent_idx}</div>`;
    }
    body += renderScores(it);

    if (it.changes) {
      body += `<div class="section-title">Reflector changes</div>`;
      const bullets = it.changes.split(/\n(?=- )/).map(s => s.replace(/^- /, '').trim()).filter(Boolean);
      if (bullets.length > 1) {
        body += `<div class="changes-box"><ul>${bullets.map(b => `<li>${esc(b)}</li>`).join('')}</ul></div>`;
      } else {
        body += `<div class="changes-box">${esc(it.changes)}</div>`;
      }
    }

    body += `<div class="section-title">Prompt diff — parent → child</div>`;
    if (!it.parent_prompt && !it.child_prompt) {
      body += `<div class="no-diff">No prompt data for this iteration.</div>`;
    } else if (it.parent_prompt && !it.child_prompt) {
      body += `<div class="no-diff">Skipped — parent scored perfectly.</div>`;
    } else if (it.parent_prompt === it.child_prompt) {
      body += `<div class="no-diff">Prompts are identical — no changes.</div>`;
    } else {
      const diff = lineDiff(it.parent_prompt, it.child_prompt);
      const dels = diff.filter(d => d.t === 'd').length;
      const ins  = diff.filter(d => d.t === 'i').length;
      body += `<div class="diff-meta"><span class="stat-del">−${dels} lines</span><span class="stat-ins">+${ins} lines</span></div>${renderUnifiedDiff(diff)}`;
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
