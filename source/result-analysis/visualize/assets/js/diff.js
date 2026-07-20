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
