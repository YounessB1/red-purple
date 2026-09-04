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

function trPct(x) {
  return (x == null) ? '—' : (x * 100).toFixed(1) + '%';
}

function testDeltaClass(delta) {
  if (delta > 0.0001) return 's-delta-pos';
  if (delta < -0.0001) return 's-delta-neg';
  return 's-delta-eq';
}

function testDeltaLabel(delta) {
  return (delta > 0 ? '+' : '') + (delta * 100).toFixed(1) + 'pp';
}

// buckets: ordered list of { label, data } — each data is a by_level/by_family dict or null
function testByBucketRows(buckets) {
  const keys = [...new Set(buckets.flatMap(b => Object.keys(b.data || {})))];
  return keys
    .map(k => {
      const cells = buckets.map(b => {
        const v = b.data?.[k] || { n: 0, solved: 0 };
        return { n: v.n, solved: v.solved, rate: v.n ? v.solved / v.n : 0 };
      });
      const n = Math.max(...cells.map(c => c.n));
      const last = cells[cells.length - 1], first = cells[0];
      return { key: k, n, cells, delta: last.rate - first.rate };
    })
    .sort((a, b) => b.n - a.n);
}

function testBucketTable(title, buckets) {
  const rows = testByBucketRows(buckets);
  const head = buckets.map(b => `<th>${esc(b.label)}</th>`).join('') + (buckets.length > 1 ? '<th>Δ</th>' : '');
  const body = rows.map(r => {
    const cellsHtml = r.cells.map(c =>
      `<td class="${c.n === 0 ? 's-none' : c.rate === 1 ? 's-pass' : c.rate === 0 ? 's-fail' : 's-mid'}">${c.n ? `${c.solved}/${c.n}` : '—'}</td>`
    ).join('');
    const deltaHtml = buckets.length > 1
      ? `<td class="${testDeltaClass(r.delta)}">${r.delta > 0 ? '+' : ''}${(r.delta * 100).toFixed(0)}pp</td>`
      : '';
    return `<tr><td class="bench-name">${esc(r.key)} <span class="bench-cat">n=${r.n}</span></td>${cellsHtml}${deltaHtml}</tr>`;
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

function testCardHtml(label, agg, extraSub) {
  return `
    <div class="test-card">
      <div class="test-card-lbl">${esc(label)}</div>
      <div class="test-card-val">${trPct(agg.pass_rate)}</div>
      <div class="test-card-sub">${agg.solved}/${agg.n} solved${extraSub ? ' · ' + esc(extraSub) : ''}</div>
    </div>`;
}

function renderTestResults(el) {
  const ts = DATA.test_summary;
  if (!ts || !ts.seed || !ts.best) {
    el.innerHTML = `<div class="no-diff">No held-out test evaluation found for this experiment.<br>
      Run: <code>python3 -m source.optimize_anything.eval_test_set experiments/${esc(DATA.name)}</code></div>`;
    return;
  }

  const baseline = ts.baseline || null;
  const seed = ts.seed, best = ts.best;
  const deltaSeedBest = ts.delta_pass_rate ?? (best.pass_rate - seed.pass_rate);
  const diff = ts.diff || {};

  const buckets = [];
  if (baseline) buckets.push({ label: 'Baseline', data: null, agg: baseline });
  buckets.push({ label: 'Seed', data: null, agg: seed });
  buckets.push({ label: 'Best', data: null, agg: best });

  const levelBuckets  = buckets.map(b => ({ label: b.label, data: b.agg.by_level }));
  const familyBuckets = buckets.map(b => ({ label: b.label, data: b.agg.by_family }));

  let cardsHtml = '';
  if (baseline) {
    cardsHtml += testCardHtml('Baseline', baseline, `flag-match only, ${esc(baseline.source || '')}`);
    cardsHtml += `<div class="test-card test-card-arrow">→</div>`;
  }
  cardsHtml += testCardHtml('Seed', seed, `avg score ${(seed.avg_score ?? 0).toFixed(2)}`);
  cardsHtml += `<div class="test-card test-card-arrow">→</div>`;
  cardsHtml += testCardHtml('Best', best, `avg score ${(best.avg_score ?? 0).toFixed(2)}`);
  cardsHtml += `
    <div class="test-card test-card-delta">
      <div class="test-card-lbl">Δ Best − Seed</div>
      <div class="test-card-val ${testDeltaClass(deltaSeedBest)}">${testDeltaLabel(deltaSeedBest)}</div>
      <div class="test-card-sub">held-out test split, n=${best.n}</div>
    </div>`;
  if (baseline) {
    const deltaBestBaseline = best.pass_rate - baseline.pass_rate;
    cardsHtml += `
      <div class="test-card test-card-delta">
        <div class="test-card-lbl">Δ Best − Baseline</div>
        <div class="test-card-val ${testDeltaClass(deltaBestBaseline)}">${testDeltaLabel(deltaBestBaseline)}</div>
        <div class="test-card-sub">optimized vs. hand-authored agent</div>
      </div>`;
  }

  el.innerHTML = `
    <div class="test-summary-cards">${cardsHtml}</div>

    <div class="scores-wrap" style="margin-top:16px;">
      ${testBucketTable('Level', levelBuckets)}
      ${testBucketTable('Family', familyBuckets)}
    </div>

    <div class="test-diff-wrap">
      ${testDiffList('Solved by best only (gained vs. seed)', diff.solved_by_best_only, 'test-diff-gain')}
      ${testDiffList('Solved by seed only (regressed vs. seed)', diff.solved_by_seed_only, 'test-diff-loss')}
      ${testDiffList('Solved by both (seed & best)', diff.solved_by_both, 'test-diff-both')}
      ${testDiffList('Solved by neither (seed & best)', diff.solved_by_neither, 'test-diff-neither')}
    </div>`;
}
