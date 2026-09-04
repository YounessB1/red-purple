// ── Evolution → Validation performance chart ────────────────────────────────
// One bar per accepted candidate (plus the seed) — the lineage that's
// actually being evolved on — each scored on the full held-out val set.
function buildValidationSeries() {
  const rows = [];
  for (const it of DATA.iterations) {
    if (it.status !== 'accepted') continue;
    const vals = it.child_val ? Object.values(it.child_val).filter(v => v != null) : [];
    if (!vals.length) continue;
    rows.push({
      iter: it.id,
      status: it.parent_idx == null ? 'seed' : 'accepted',
      idx: it.child_candidate_idx,
      val: vals.reduce((a, b) => a + b, 0) / vals.length,
      n: vals.length,
    });
  }
  return rows.sort((a, b) => a.iter - b.iter);
}

// ── Slope of the running-best val score ─────────────────────────────────
// A rolling rate of improvement: (bestVal[i] - bestVal[i-window]) / window,
// evaluated at every accepted iteration. Since the running-best series is
// monotonic non-decreasing, this is always >= 0 and decays toward 0 as the
// run plateaus — a diminishing-returns / learning-curve view that's more
// useful than the raw step line for judging "is this still worth running".
function computeSlopeSeries(bestPoints, window) {
  const slopes = [];
  for (let i = 0; i < bestPoints.length; i++) {
    const targetIter = bestPoints[i].iter - window;
    let j = -1;
    for (let k = i; k >= 0; k--) {
      if (bestPoints[k].iter <= targetIter) { j = k; break; }
    }
    if (j === -1) continue;  // not enough history yet for a full window
    const dIter = bestPoints[i].iter - bestPoints[j].iter;
    const dVal  = bestPoints[i].val  - bestPoints[j].val;
    slopes.push({ iter: bestPoints[i].iter, slope: dIter > 0 ? dVal / dIter : 0 });
  }
  return slopes;
}

function renderSlopeChart(el, bestPoints) {
  if (bestPoints.length < 2) { el.innerHTML = ''; return; }

  const maxIter = Math.max(1, ...bestPoints.map(p => p.iter));
  // window ~10% of the run, clamped to something both stable and legible
  const window = Math.max(3, Math.min(15, Math.round(maxIter * 0.1)));
  const slopes = computeSlopeSeries(bestPoints, window);
  if (!slopes.length) {
    el.innerHTML = `<div class="no-diff">Not enough iterations yet for a ${window}-iteration rolling slope.</div>`;
    return;
  }

  const W = 900, H = 200;
  const PAD_L = 62, PAD_R = 20, PAD_T = 16, PAD_B = 30;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const maxSlope = Math.max(1e-6, ...slopes.map(s => s.slope));

  const px = iter  => PAD_L + (maxIter === 0 ? 0 : (iter / maxIter) * plotW);
  const py = slope => PAD_T + (1 - slope / maxSlope) * plotH;

  let grid = '';
  [0, 0.5, 1].forEach(f => {
    const v = maxSlope * f;
    const gy = py(v);
    grid += `<line x1="${PAD_L}" y1="${gy.toFixed(1)}" x2="${W - PAD_R}" y2="${gy.toFixed(1)}" class="vc-grid"/>`;
    grid += `<text x="${PAD_L - 8}" y="${(gy + 3).toFixed(1)}" text-anchor="end" class="vc-axis-lbl">${v.toFixed(3)}</text>`;
  });

  const xTicks = [...new Set([slopes[0].iter, slopes[slopes.length - 1].iter])];
  let xAxis = `<line x1="${PAD_L}" y1="${PAD_T + plotH}" x2="${W - PAD_R}" y2="${PAD_T + plotH}" class="vc-axis-line"/>`;
  xTicks.forEach(iter => {
    xAxis += `<text x="${px(iter).toFixed(1)}" y="${H - 6}" text-anchor="middle" class="vc-axis-lbl">${pad(iter)}</text>`;
  });

  let line = '', area = '';
  slopes.forEach((s, i) => {
    const x = px(s.iter).toFixed(1), y = py(s.slope).toFixed(1);
    line += i === 0 ? `M${x},${y}` : ` L${x},${y}`;
  });
  area = `${line} L${px(slopes[slopes.length - 1].iter).toFixed(1)},${py(0).toFixed(1)} L${px(slopes[0].iter).toFixed(1)},${py(0).toFixed(1)} Z`;

  let hits = '';
  slopes.forEach(s => {
    hits += `<circle cx="${px(s.iter).toFixed(1)}" cy="${py(s.slope).toFixed(1)}" r="7" class="vc-slope-hit"
      data-iter="${s.iter}" data-slope="${s.slope}"></circle>`;
  });

  el.innerHTML = `
    <div class="vc-wrap vc-slope-wrap">
      <div class="vc-legend">
        <span class="evo-legend-item"><span class="vc-legend-line-swatch"></span>Δ best-val / iteration, ${window}-iter rolling window</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" class="vc-svg" preserveAspectRatio="xMidYMid meet">
        ${grid}
        ${xAxis}
        <path d="${area}" class="vc-slope-area"></path>
        <path d="${line}" class="vc-slope-line"></path>
        ${hits}
      </svg>
      <div class="vc-tooltip" id="vc-slope-tooltip"></div>
    </div>`;

  const wrap    = el.querySelector('.vc-slope-wrap');
  const tooltip = el.querySelector('#vc-slope-tooltip');
  el.querySelectorAll('.vc-slope-hit').forEach(hit => {
    hit.addEventListener('mouseenter', () => {
      const d = hit.dataset;
      tooltip.textContent = `Iter ${pad(Number(d.iter))} · slope ${parseFloat(d.slope).toFixed(4)} / iter (last ${window})`;
      tooltip.style.display = 'block';
    });
    hit.addEventListener('mousemove', e => {
      const r = wrap.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 12) + 'px';
      tooltip.style.top  = (e.clientY - r.top - 10) + 'px';
    });
    hit.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  });
}

function renderValidationChart(el) {
  const rows = buildValidationSeries();
  if (!rows.length) {
    el.innerHTML = '<div class="no-diff">No validation scores recorded yet.</div>';
    return;
  }

  const seed = rows.find(r => r.status === 'seed') || rows[0];
  const best = rows.reduce((b, p) => (p.val > b.val ? p : b), rows[0]);

  const W = 900, H = 340;
  const PAD_L = 62, PAD_R = 20, PAD_T = 20, PAD_B = 30;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const maxIter = Math.max(1, ...rows.map(r => r.iter));

  const px = iter => PAD_L + (maxIter === 0 ? 0 : (iter / maxIter) * plotW);
  const py = val  => PAD_T + (1 - val) * plotH;

  // ── recessive gridlines + y-axis labels ──
  let grid = '';
  [0, 0.25, 0.5, 0.75, 1].forEach(v => {
    const gy = py(v);
    grid += `<line x1="${PAD_L}" y1="${gy.toFixed(1)}" x2="${W - PAD_R}" y2="${gy.toFixed(1)}" class="vc-grid"/>`;
    grid += `<text x="${PAD_L - 8}" y="${(gy + 3).toFixed(1)}" text-anchor="end" class="vc-axis-lbl">${v.toFixed(2)}</text>`;
  });

  // x-axis: only label first, last, and best iteration to avoid clutter across many iterations
  const xTicks = [...new Set([rows[0].iter, rows[rows.length - 1].iter, best.iter])].sort((a, b) => a - b);
  let xAxis = `<line x1="${PAD_L}" y1="${PAD_T + plotH}" x2="${W - PAD_R}" y2="${PAD_T + plotH}" class="vc-axis-line"/>`;
  xTicks.forEach(iter => {
    xAxis += `<text x="${px(iter).toFixed(1)}" y="${H - 6}" text-anchor="middle" class="vc-axis-lbl">${pad(iter)}</text>`;
  });

  // ── histogram bars, one per accepted candidate (+ seed), colored by
  // whether the candidate ever survived the val-set Pareto pruning (green)
  // or was dominated out of every pool.json snapshot (red) — see
  // utils.js::computeEverPooledSet ──
  const everPooled = computeEverPooledSet();
  const barW = Math.max(2, Math.min(14, (plotW / (maxIter + 1)) * 0.72));
  let bars = '';
  for (const r of rows) {
    const x = px(r.iter) - barW / 2;
    const yTop = py(r.val);
    const yBase = py(0);
    const h = Math.max(0.5, yBase - yTop);
    const cls = everPooled.has(r.idx) ? 'vc-bar-kept' : 'vc-bar-pruned';
    bars += `<rect x="${x.toFixed(1)}" y="${yTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" class="vc-bar ${cls}"></rect>`;
  }

  // ── running-best step line: at each accepted iteration, the highest val
  // avg seen so far — flat until a new best displaces it, never a diagonal ──
  let bestSoFar = -Infinity;
  const bestPoints = rows.map(r => { bestSoFar = Math.max(bestSoFar, r.val); return { iter: r.iter, val: bestSoFar }; });
  let bestLine = '';
  bestPoints.forEach((p, i) => {
    const x = px(p.iter).toFixed(1), y = py(p.val).toFixed(1);
    bestLine += i === 0 ? `M${x},${y}` : ` L${x},${py(bestPoints[i - 1].val).toFixed(1)} L${x},${y}`;
  });

  // ── direct labels + markers for seed & best only. Seed sits on the
  // leftmost bar where a centered label collides with the next few bars, so
  // it's parked in the left margin (level with its own value, same column as
  // the y-axis labels) instead of floating over the plot ──
  const cySeed = py(seed.val), cyBest = py(best.val);
  const calloutSeed = `<text x="${(PAD_L - 8).toFixed(1)}" y="${(cySeed + 3).toFixed(1)}"
    text-anchor="end" class="vc-callout vc-callout-seed">Seed ${seed.val.toFixed(2)}</text>`;
  const calloutBest = best.idx === seed.idx ? '' : `<text x="${px(best.iter).toFixed(1)}" y="${(cyBest > 40 ? cyBest - 10 : cyBest + 20).toFixed(1)}"
    text-anchor="middle" class="vc-callout vc-callout-best">Best ${best.val.toFixed(2)}</text>`;
  const callouts = calloutSeed + calloutBest;
  let endpoints = '';
  for (const p of [seed, best]) {
    endpoints += `<circle cx="${px(p.iter).toFixed(1)}" cy="${py(p.val).toFixed(1)}" r="5" class="${p === seed ? 'vc-dot vc-dot-seed' : 'vc-dot vc-dot-best'}"></circle>`;
  }

  // ── full-height hit rects, one per bar, drive the tooltip ──
  let hits = '';
  for (const r of rows) {
    const w = Math.max(barW, 6);
    const x = px(r.iter) - w / 2;
    hits += `<rect x="${x.toFixed(1)}" y="${PAD_T}" width="${w.toFixed(1)}" height="${plotH.toFixed(1)}" class="vc-hit-bar"
      data-iter="${r.iter}" data-status="${r.status}" data-idx="${r.idx ?? ''}" data-val="${r.val}" data-n="${r.n}"
      data-pooled="${everPooled.has(r.idx)}"></rect>`;
  }

  el.innerHTML = `
    <div class="vc-wrap">
      <div class="vc-legend">
        <span class="evo-legend-item"><span class="vc-legend-swatch vc-bar-kept"></span>In pool (kept on Pareto front)</span>
        <span class="evo-legend-item"><span class="vc-legend-swatch vc-bar-pruned"></span>Pruned (dominated)</span>
        <span class="evo-legend-item"><span class="vc-legend-line-swatch"></span>Best so far</span>
      </div>
      <svg viewBox="0 0 ${W} ${H}" class="vc-svg" preserveAspectRatio="xMidYMid meet">
        ${grid}
        ${xAxis}
        ${bars}
        <path d="${bestLine}" class="vc-line"></path>
        ${endpoints}
        ${callouts}
        ${hits}
      </svg>
      <div class="vc-tooltip" id="vc-tooltip"></div>
    </div>
    <div class="vc-summary">
      <span class="vc-summary-item"><span class="vc-dot-swatch vc-dot-seed"></span>Seed (iter ${pad(seed.iter)}) — <strong>${seed.val.toFixed(2)}</strong></span>
      <span class="vc-summary-item"><span class="vc-dot-swatch vc-dot-best"></span>Best (iter ${pad(best.iter)}, candidate #${best.idx}) — <strong>${best.val.toFixed(2)}</strong></span>
    </div>
    <div class="vc-slope-title">Rate of improvement (slope of best-so-far)</div>
    <div id="vc-slope-chart"></div>`;

  renderSlopeChart(el.querySelector('#vc-slope-chart'), bestPoints);

  const wrap    = el.querySelector('.vc-wrap');
  const tooltip = el.querySelector('#vc-tooltip');
  el.querySelectorAll('.vc-hit-bar').forEach(hit => {
    hit.addEventListener('mouseenter', () => {
      const d = hit.dataset;
      const statusLbl = d.status === 'seed' ? 'Seed' : 'Accepted';
      const poolLbl = d.pooled === 'true' ? 'kept' : 'pruned';
      tooltip.textContent = `Iter ${pad(Number(d.iter))} · Cand #${d.idx} · ${statusLbl} · ${poolLbl} · val avg ${parseFloat(d.val).toFixed(2)} (n=${d.n})`;
      tooltip.style.display = 'block';
    });
    hit.addEventListener('mousemove', e => {
      const r = wrap.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 12) + 'px';
      tooltip.style.top  = (e.clientY - r.top - 10) + 'px';
    });
    hit.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  });
}
