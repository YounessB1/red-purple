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
