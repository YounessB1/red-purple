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

  // ── histogram bars, one per accepted candidate (+ seed) ──
  const barW = Math.max(2, Math.min(14, (plotW / (maxIter + 1)) * 0.72));
  let bars = '';
  for (const r of rows) {
    const x = px(r.iter) - barW / 2;
    const yTop = py(r.val);
    const yBase = py(0);
    const h = Math.max(0.5, yBase - yTop);
    const cls = r.status === 'seed' ? 'vc-bar-seed' : 'vc-bar-accepted';
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
      data-iter="${r.iter}" data-status="${r.status}" data-idx="${r.idx ?? ''}" data-val="${r.val}" data-n="${r.n}"></rect>`;
  }

  el.innerHTML = `
    <div class="vc-wrap">
      <div class="vc-legend">
        <span class="evo-legend-item"><span class="vc-legend-swatch vc-bar-seed"></span>Seed</span>
        <span class="evo-legend-item"><span class="vc-legend-swatch vc-bar-accepted"></span>Accepted (val avg)</span>
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
    </div>`;

  const wrap    = el.querySelector('.vc-wrap');
  const tooltip = el.querySelector('#vc-tooltip');
  el.querySelectorAll('.vc-hit-bar').forEach(hit => {
    hit.addEventListener('mouseenter', () => {
      const d = hit.dataset;
      const statusLbl = d.status === 'seed' ? 'Seed' : 'Accepted';
      tooltip.textContent = `Iter ${pad(Number(d.iter))} · Cand #${d.idx} · ${statusLbl} · val avg ${parseFloat(d.val).toFixed(2)} (n=${d.n})`;
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
