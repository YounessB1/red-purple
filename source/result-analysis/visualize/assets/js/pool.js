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
