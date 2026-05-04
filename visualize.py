#!/usr/bin/env python3
"""Generate a self-contained HTML prompt viewer for a red-purple experiment.
Usage: python visualize.py [experiment_dir]
Output: <experiment_dir>/viewer.html
"""
import json
import sys
from pathlib import Path


def load_experiment(exp_dir: Path) -> dict:
    def load_benches(split_dir: Path) -> list:
        benches = []
        if split_dir.exists():
            for bd in sorted(split_dir.iterdir()):
                mf = bd / "metadata.json"
                if mf.exists():
                    m = json.loads(mf.read_text())
                    benches.append({
                        "bench": bd.name,
                        "success": bool(m.get("success")),
                        "duration_s": round(m.get("duration_seconds", 0)),
                        "iterations_used": m.get("iterations_used", 0),
                        "max_iterations": m.get("max_iterations", 0),
                        "stop_reason": m.get("stop_reason", ""),
                        "cost_usd": round(m.get("total_cost_usd", 0), 4),
                    })
        return benches

    logs_dir = exp_dir / "reflection_logs"
    proposal_files = sorted(logs_dir.glob("iter_*_proposal.json")) if logs_dir.exists() else []

    iterations = []
    for pf in proposal_files:
        num = int(pf.stem.split("_")[1])
        data = json.loads(pf.read_text())
        prompt = (data.get("new_instructions") or {}).get("prompt", "")

        if num == 0:
            status, reject_reason = "seed", None
        else:
            acc_f = logs_dir / f"iter_{num:03d}_accepted.json"
            rej_f = logs_dir / f"iter_{num:03d}_rejected.json"
            if acc_f.exists():
                status, reject_reason = "accepted", None
            elif rej_f.exists():
                rej = json.loads(rej_f.read_text())
                status = "rejected"
                reject_reason = rej.get("reason", "")
            else:
                status, reject_reason = "pending", None

        iterations.append({
            "id": num,
            "status": status,
            "prompt": prompt,
            "reject_reason": reject_reason,
            "train": load_benches(exp_dir / f"iteration_{num:03d}" / "train"),
            "val":   load_benches(exp_dir / f"iteration_{num:03d}" / "val"),
        })

    return {"name": exp_dir.name, "iterations": iterations}


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
/* ── Header ─────────────────────────────────────────────────────────── */
#hdr {
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 10px 18px; display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
#hdr h1 { font-size: 15px; font-weight: 600; color: #e6edf3; }
#hdr .hint { font-size: 12px; color: #6e7681; }
/* ── Layout ──────────────────────────────────────────────────────────── */
#body { display: flex; flex: 1; overflow: hidden; }
#sidebar {
  width: 230px; flex-shrink: 0; background: #161b22;
  border-right: 1px solid #30363d; overflow-y: auto; padding: 8px;
}
#main { flex: 1; overflow-y: auto; padding: 18px 22px; }
/* ── Sidebar cards ───────────────────────────────────────────────────── */
.card {
  border: 1px solid #30363d; border-radius: 7px; padding: 9px 11px;
  margin-bottom: 5px; cursor: pointer; transition: border-color 0.12s, background 0.12s;
  position: relative;
}
.card:hover { border-color: #58a6ff; background: #1c2128; }
.card.sel-a { border-color: #1f6feb; background: #1c2128; box-shadow: 0 0 0 2px #1f6feb55; }
.card.sel-b { border-color: #8957e5; background: #1c2128; box-shadow: 0 0 0 2px #8957e555; }
.card-top { display: flex; align-items: center; gap: 7px; margin-bottom: 3px; }
.card-num { font-size: 12px; font-weight: 600; color: #e6edf3; }
.badge {
  font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 9px;
  letter-spacing: 0.4px; text-transform: uppercase;
}
.b-seed     { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.b-accepted { background: #1a4228; color: #3fb950; border: 1px solid #238636; }
.b-rejected { background: #3d1a1a; color: #f85149; border: 1px solid #da3633; }
.b-pending  { background: #2d2516; color: #d29922; border: 1px solid #9e6a03; }
.sel-dot {
  position: absolute; top: 7px; right: 8px;
  font-size: 10px; font-weight: 700; width: 18px; height: 18px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
}
.dot-a { background: #1f6feb; color: #fff; }
.dot-b { background: #8957e5; color: #fff; }
.card-scores { font-size: 10px; color: #6e7681; }
.spill {
  display: inline-block; background: #21262d; border-radius: 3px;
  padding: 1px 5px; margin-right: 3px;
}
/* ── Main placeholder ────────────────────────────────────────────────── */
#placeholder {
  height: 100%; display: flex; align-items: center; justify-content: center;
  color: #3d444d; font-size: 14px;
}
/* ── Score boxes ─────────────────────────────────────────────────────── */
.score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 18px; }
.sbox {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 13px 15px;
}
.sbox.box-a { border-top: 3px solid #1f6feb; }
.sbox.box-b { border-top: 3px solid #8957e5; }
.sbox.box-single { border-top: 3px solid #30363d; grid-column: 1 / -1; }
.sbox h3 { font-size: 12px; font-weight: 600; color: #e6edf3; margin-bottom: 8px; }
.big-num { font-size: 26px; font-weight: 700; line-height: 1; }
.big-sub { font-size: 10px; color: #6e7681; margin-top: 2px; margin-bottom: 8px; }
.bench-row { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }
.bpill {
  font-size: 9.5px; padding: 2px 6px; border-radius: 4px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  cursor: default;
}
.bpill-ok   { background: #1a4228; color: #3fb950; }
.bpill-fail { background: #3d1a1a; color: #f85149; }
.reject-note { font-size: 10px; color: #d29922; margin-top: 7px; }
/* ── Diff controls ───────────────────────────────────────────────────── */
.diff-bar {
  display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;
}
.diff-bar .la { color: #58a6ff; font-size: 12px; font-weight: 600; }
.diff-bar .arrow { color: #484f58; }
.diff-bar .lb { color: #a371f7; font-size: 12px; font-weight: 600; }
.diff-bar .stats { font-size: 11px; color: #6e7681; }
/* ── Diff content ────────────────────────────────────────────────────── */
.diff-wrap {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  overflow: hidden; font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 12.5px; line-height: 1.65;
}
.dline { display: flex; }
.dgutter {
  width: 34px; flex-shrink: 0; text-align: center; font-size: 10px;
  color: #484f58; user-select: none; border-right: 1px solid #21262d; padding: 0 3px;
  display: flex; align-items: center; justify-content: center;
}
.dcontent { flex: 1; padding: 0 12px; white-space: pre-wrap; word-break: break-word; }
.d-eq  .dcontent { color: #8b949e; }
.d-del { background: #3d1a1a; }
.d-del .dcontent { color: #ffa198; }
.d-del .dgutter  { background: #3d1a1a; color: #f85149; }
.d-ins { background: #1a3528; }
.d-ins .dcontent { color: #7ee787; }
.d-ins .dgutter  { background: #1a3528; color: #3fb950; }
.d-skip {
  background: #161b22; color: #484f58; font-size: 10px; text-align: center;
  padding: 3px 0; border-top: 1px solid #21262d; border-bottom: 1px solid #21262d;
  font-family: sans-serif; cursor: pointer; user-select: none;
}
.d-skip:hover { color: #8b949e; }
/* ── Single prompt ───────────────────────────────────────────────────── */
.prompt-box {
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  padding: 18px 20px; margin-top: 14px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
  color: #c9d1d9;
}
</style>
</head>
<body>
<div id="hdr">
  <h1 id="exp-name"></h1>
  <span class="hint">Click a card to view prompt · Click two cards to diff</span>
</div>
<div id="body">
  <div id="sidebar"></div>
  <div id="main">
    <div id="placeholder">← Select an iteration</div>
    <div id="content" style="display:none"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
let selA = null, selB = null;

document.getElementById('exp-name').textContent = DATA.name;

// ── Sidebar ────────────────────────────────────────────────────────────
function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  DATA.iterations.forEach(it => {
    const trainOk = it.train.filter(b => b.success).length;
    const valOk   = it.val.filter(b => b.success).length;
    const scoreStr = it.val.length
      ? `<span class="spill">val ${valOk}/${it.val.length}</span>`
      : it.train.length
        ? `<span class="spill">train ${trainOk}/${it.train.length}</span>`
        : '';

    const isA = selA === it.id, isB = selB === it.id;
    const card = document.createElement('div');
    card.className = 'card' + (isA ? ' sel-a' : isB ? ' sel-b' : '');
    card.innerHTML =
      `<div class="card-top">
         <span class="card-num">Iter ${pad(it.id)}</span>
         <span class="badge b-${it.status}">${it.status}</span>
         ${isA ? '<span class="sel-dot dot-a">A</span>' : isB ? '<span class="sel-dot dot-b">B</span>' : ''}
       </div>
       <div class="card-scores">${scoreStr}</div>`;
    card.addEventListener('click', () => pick(it.id));
    sb.appendChild(card);
  });
}

function pick(id) {
  if      (selA === id) { selA = selB; selB = null; }
  else if (selB === id) { selB = null; }
  else if (selA === null) { selA = id; }
  else if (selB === null) { selB = id; }
  else { selA = selB; selB = id; }
  renderSidebar();
  renderMain();
}

// ── Main ───────────────────────────────────────────────────────────────
function renderMain() {
  const ph = document.getElementById('placeholder');
  const ct = document.getElementById('content');
  if (selA === null) { ph.style.display = 'flex'; ct.style.display = 'none'; return; }
  ph.style.display = 'none'; ct.style.display = 'block';
  const itA = DATA.iterations.find(x => x.id === selA);
  if (selB === null) {
    renderSingle(ct, itA);
  } else {
    renderCompare(ct, itA, DATA.iterations.find(x => x.id === selB));
  }
}

// ── Score box ──────────────────────────────────────────────────────────
function scoreBox(it, cls) {
  const trainOk = it.train.filter(b => b.success).length;
  const valOk   = it.val.filter(b => b.success).length;

  const primary = it.val.length
    ? { ok: valOk, total: it.val.length, label: 'val' }
    : it.train.length
      ? { ok: trainOk, total: it.train.length, label: 'train' }
      : null;

  const pct = primary ? Math.round(primary.ok / primary.total * 100) : null;
  const color = pct === null ? '#6e7681' : pct >= 50 ? '#3fb950' : '#f85149';

  function pills(benches) {
    return benches.map(b =>
      `<span class="bpill ${b.success ? 'bpill-ok' : 'bpill-fail'}"
             title="${b.bench}&#10;${b.stop_reason}&#10;${b.duration_s}s  $${b.cost_usd}"
       >${b.bench}</span>`
    ).join('');
  }

  const extraSplit = (it.val.length && it.train.length)
    ? `<div class="big-sub">train ${trainOk}/${it.train.length}</div><div class="bench-row">${pills(it.train)}</div>`
    : '';

  return `<div class="sbox ${cls}">
    <h3>Iter ${pad(it.id)} — ${it.status.toUpperCase()}</h3>
    ${primary ? `<div class="big-num" style="color:${color}">${pct}%</div>
                 <div class="big-sub">${primary.label} ${primary.ok}/${primary.total}</div>` : ''}
    <div class="bench-row">${pills(it.val.length ? it.val : it.train)}</div>
    ${extraSplit}
    ${it.reject_reason ? `<div class="reject-note">⚠ ${esc(it.reject_reason)}</div>` : ''}
  </div>`;
}

// ── Single ─────────────────────────────────────────────────────────────
function renderSingle(el, it) {
  el.innerHTML =
    `<div class="score-grid">${scoreBox(it, 'box-single')}</div>
     <div class="prompt-box">${esc(it.prompt)}</div>`;
}

// ── Compare ────────────────────────────────────────────────────────────
function renderCompare(el, itA, itB) {
  const diff = lineDiff(itA.prompt, itB.prompt);
  const dels = diff.filter(d => d.t === 'd').length;
  const ins  = diff.filter(d => d.t === 'i').length;

  el.innerHTML =
    `<div class="score-grid">
       ${scoreBox(itA, 'box-a')}
       ${scoreBox(itB, 'box-b')}
     </div>
     <div class="diff-bar">
       <span class="la">A: Iter ${pad(itA.id)} (${itA.status})</span>
       <span class="arrow">→</span>
       <span class="lb">B: Iter ${pad(itB.id)} (${itB.status})</span>
       <span class="stats">−${dels} lines &nbsp; +${ins} lines</span>
     </div>
     <div class="diff-wrap">${renderDiff(diff)}</div>`;
}

function renderDiff(diff) {
  const CTX = 4;
  const show = new Array(diff.length).fill(false);
  diff.forEach((d, i) => {
    if (d.t !== 'e') {
      for (let j = Math.max(0, i - CTX); j <= Math.min(diff.length - 1, i + CTX); j++)
        show[j] = true;
    }
  });

  let html = '', lineN = 1, i = 0;
  while (i < diff.length) {
    if (!show[i]) {
      let j = i;
      while (j < diff.length && !show[j]) {
        if (diff[j].t === 'e') lineN++;
        j++;
      }
      const n = j - i;
      html += `<div class="d-skip">··· ${n} unchanged line${n !== 1 ? 's' : ''} ···</div>`;
      i = j;
    } else {
      const d = diff[i];
      const cls = d.t === 'e' ? 'd-eq' : d.t === 'd' ? 'd-del' : 'd-ins';
      const glyph = d.t === 'e' ? lineN++ : d.t === 'd' ? '−' : '+';
      html += `<div class="dline ${cls}">
        <span class="dgutter">${glyph}</span>
        <span class="dcontent">${esc(d.line)}</span>
      </div>`;
      i++;
    }
  }
  return html;
}

// ── Line-level LCS diff ────────────────────────────────────────────────
function lineDiff(a, b) {
  const al = a ? a.split('\n') : [];
  const bl = b ? b.split('\n') : [];
  const m = al.length, n = bl.length;
  // DP table
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = al[i] === bl[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
  // Traceback
  const out = [];
  let i = 0, j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && al[i] === bl[j]) { out.push({ t: 'e', line: al[i] }); i++; j++; }
    else if (j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j])) { out.push({ t: 'i', line: bl[j] }); j++; }
    else { out.push({ t: 'd', line: al[i] }); i++; }
  }
  return out;
}

// ── Utils ──────────────────────────────────────────────────────────────
function pad(n) { return String(n).padStart(3, '0'); }
function esc(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Init ───────────────────────────────────────────────────────────────
renderSidebar();
renderMain();
</script>
</body>
</html>
"""


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/experiment1")
    data = load_experiment(exp_dir)
    # Escape < > & so embedded JSON can't be parsed as HTML tags inside <script>
    safe_json = (json.dumps(data, ensure_ascii=False)
                 .replace("<", "\\u003c")
                 .replace(">", "\\u003e")
                 .replace("&", "\\u0026"))
    html = HTML.replace("__NAME__", data["name"]).replace("__DATA__", safe_json)
    out = exp_dir / "viewer.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated: {out}")


main()
