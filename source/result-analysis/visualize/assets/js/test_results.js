// ── Evolution → Test tab ─────────────────────────────────────────────────
// Held-out test-split comparison across three reference points, all scored
// on the exact same test benchmark IDs:
//   - baseline: the hand-authored agent (source/baseline/), aggregated down
//     to this experiment's test split from its own full 104-benchmark run
//     (see data_loader.py::_load_baseline_test_aggregate) — optional, only
//     present when experiments/baseline/<model>/summary.json exists.
//   - seed / best: source/optimize_anything/eval_test_set.py's output,
//     written to <experiment_dir>/test/summary.json.
// Shown as its own top-level sidebar entry (not nested under Evolution) —
// this is the number that actually answers "did optimizing help, and does
// it beat a well-informed human-written agent", since train/val are what
// GEPA searched over.

// buckets: ordered list of { label, data } — each data is a by_family dict or null
function testByBucketRows(buckets) {
  const keys = [...new Set(buckets.flatMap(b => Object.keys(b.data || {})))];
  return keys
    .map(k => {
      const cells = buckets.map(b => {
        const v = b.data?.[k] || { n: 0, solved: 0 };
        return { n: v.n, solved: v.solved, rate: v.n ? v.solved / v.n : 0 };
      });
      const n = Math.max(...cells.map(c => c.n));
      return { key: k, n, cells };
    })
    .sort((a, b) => b.n - a.n);
}

function testBucketTable(title, buckets) {
  const rows = testByBucketRows(buckets);
  const head = buckets.map(b => `<th>${esc(b.label)}</th>`).join('');
  const body = rows.map(r => {
    const cellsHtml = r.cells.map(c =>
      `<td class="${c.n === 0 ? 's-none' : c.rate === 1 ? 's-pass' : c.rate === 0 ? 's-fail' : 's-mid'}">${c.n ? `${c.solved}/${c.n}` : '—'}</td>`
    ).join('');
    return `<tr><td class="bench-name">${esc(r.key)} <span class="bench-cat">n=${r.n}</span></td>${cellsHtml}</tr>`;
  }).join('');
  return `
    <div class="scores-panel">
      <div class="scores-panel-hdr"><span>${esc(title)}</span></div>
      <table class="scores-tbl">
        <thead><tr><th class="bench-col">${esc(title)}</th>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

function testDiffList(label, ids, cls) {
  if (!ids || !ids.length) return '';
  return `
    <div class="test-diff-group">
      <div class="test-diff-hdr ${cls}">${esc(label)} <span class="bench-cat">${ids.length}</span></div>
      <div class="test-diff-ids">${ids.map(id => `<span class="test-diff-id">${esc(id)}</span>`).join('')}</div>
    </div>`;
}

// Big percentage-only cards, one per agent — just the number, nothing else.
function testSummaryCards(buckets) {
  return `<div class="test-pct-row">${buckets.map(b => {
    const rate = b.agg.n ? b.agg.solved / b.agg.n : 0;
    return `
      <div class="test-pct-card">
        <div class="test-pct-val">${(rate * 100).toFixed(1)}%</div>
        <div class="test-pct-lbl">${esc(b.label)}</div>
      </div>`;
  }).join('')}</div>`;
}

// Same graphical file-block card used by the Evolution → Tree view (evolution.js:
// isSkillFile/fileColor/fileDisplayName), reused here so Seed/Baseline/Best read
// as the same visual "shape" as any tree node — just always expanded, no
// click-to-toggle/val-badge/pruned styling since these aren't tree candidates.
function agentCardHtml(label, files) {
  const blocks = Object.keys(files || {})
    .filter(f => f === 'prompt.md' || f === 'AGENTS.md' || isSkillFile(f))
    .sort()
    .map(f => {
      const lines    = (files[f] || '').split('\n').length;
      const name     = fileDisplayName(f);
      const col      = fileColor(f);
      const isSkill  = isSkillFile(f);
      const radius   = isSkill ? '14px' : '5px';
      const extraPad = Math.min(60, Math.max(0, lines * 3 - 8));
      return `<div class="evo-file-block"
        style="border-radius:${radius};padding-right:${8 + extraPad}px;background:${col.bg};border-color:${col.border};color:${col.text}"
        title="${esc(f)} — ${lines} lines">${esc(name)}</div>`;
    }).join('');
  return `
    <div class="evo-candidate kept test-agent-card">
      <div class="evo-candidate-hdr">${esc(label)}</div>
      <div class="evo-files">${blocks || '<span class="evo-candidate-sub">no prompt/AGENTS/skill files</span>'}</div>
    </div>`;
}

// labels/order must match testSummaryCards' `buckets` column order so the two
// rows line up (Baseline-under-Baseline, Seed-under-Seed, Best-under-Best).
function renderAgentCardsRow(buckets, defs) {
  const labelToFiles = { Baseline: defs.baseline, Seed: defs.seed, Best: defs.best };
  const cards = buckets.map(b => agentCardHtml(b.label, labelToFiles[b.label]));
  if (!cards.length) return '';
  return `<div class="section-title">Agent definitions</div><div class="test-agent-cards-row">${cards.join('')}</div>`;
}

// Full (non-diff) file dump for a candidate's whole file set — used by the
// Seed/Baseline/Best definition tabs, as opposed to the Diff tab's parent→child comparison.
function renderFullDefinition(files) {
  if (!files || !Object.keys(files).length) {
    return `<div class="no-diff">No file definition available.</div>`;
  }
  const names = Object.keys(files).sort();
  let html = `<div class="files-overview">${
    names.map(f => `<div class="file-row"><span class="file-name">${esc(f)}</span></div>`).join('')
  }</div>`;
  for (const f of names) {
    html += `<div class="section-title" style="margin-top:18px">${esc(f)}</div>`;
    html += `<pre class="code-block">${esc(files[f] || '')}</pre>`;
  }
  return html;
}

function renderTestResults(el) {
  const ts = DATA.test_summary;
  if (!ts || !ts.seed || !ts.best) {
    el.innerHTML = `<div class="no-diff">No held-out test evaluation found for this experiment.<br>
      Run: <code>python3 -m source.optimize_anything.eval_test_set experiments/${esc(DATA.name)}</code></div>`;
    return;
  }

  const defs = DATA.test_definitions || {};
  const tabs = [{ id: 'results', label: 'Results' }];
  if (defs.seed)     tabs.push({ id: 'seed',     label: 'Seed definition' });
  if (defs.baseline) tabs.push({ id: 'baseline', label: 'Baseline definition' });
  if (defs.best)     tabs.push({ id: 'best',     label: 'Best candidate definition' });
  if (!tabs.find(t => t.id === selTestTab)) selTestTab = 'results';

  const tabBar = `<div class="tab-bar">${
    tabs.map(t => `<button class="tab${selTestTab===t.id?' sel':''}" onclick="selTestTab='${t.id}';renderMain()">${esc(t.label)}</button>`).join('')
  }</div>`;

  let body = '';
  if (selTestTab === 'seed')          body = renderFullDefinition(defs.seed);
  else if (selTestTab === 'baseline') body = renderFullDefinition(defs.baseline);
  else if (selTestTab === 'best')     body = renderFullDefinition(defs.best);
  else {
    const baseline = ts.baseline || null;
    const seed = ts.seed, best = ts.best;
    const diff = ts.diff || {};

    const buckets = [];
    if (baseline) buckets.push({ label: 'Baseline', data: null, agg: baseline });
    buckets.push({ label: 'Seed', data: null, agg: seed });
    buckets.push({ label: 'Best', data: null, agg: best });

    const familyBuckets = buckets.map(b => ({ label: b.label, data: b.agg.by_family }));

    body = `
      ${testSummaryCards(buckets)}

      ${renderAgentCardsRow(buckets, defs)}

      <div class="scores-wrap" style="margin-top:16px;">
        ${testBucketTable('Family', familyBuckets)}
      </div>

      <div class="test-diff-wrap">
        ${testDiffList('Solved by best only (gained vs. seed)', diff.solved_by_best_only, 'test-diff-gain')}
        ${testDiffList('Solved by seed only (regressed vs. seed)', diff.solved_by_seed_only, 'test-diff-loss')}
        ${testDiffList('Solved by both (seed & best)', diff.solved_by_both, 'test-diff-both')}
        ${testDiffList('Solved by neither (seed & best)', diff.solved_by_neither, 'test-diff-neither')}
      </div>`;
  }

  el.innerHTML = tabBar + body;
}
