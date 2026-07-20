// ── Evolution → Similarity heatmap ──────────────────────────────────────────
// Similarity is precomputed server-side (embeddings.py) at generation time via
// OpenRouter embeddings — real semantic vectors, not text/line matching — and
// shipped in DATA.similarity as a ready-to-render matrix.
//
// The same mutation tree (seed → generations of accepted edits, val_avg only
// — no pruned/kept coloring, that's the Tree tab's job) is drawn on BOTH axes:
// horizontally on the left (feeding the heatmap's rows) and vertically on top
// (feeding its columns) — same depth/slot assignment, same leaf order, just
// rotated. Leaves (candidates never mutated further) are flush-aligned on one
// line per axis via a dashed extension where a branch ended early. Having the
// tree on both sides makes it possible to read patterns off either axis and
// see where they cross: a dark cell whose row and column both trace back to
// nearby branches is expected inheritance; a dark cell whose row and column
// trace to distant branches is convergent evolution.

// Diverging scale, not sequential: this data has a genuinely meaningful zero
// (a "typical" pair, post mean-centering) and two qualitatively different
// directions — more similar than typical, or less — so a single white-to-red
// ramp would make "typical" and "below typical" look like the same thing at
// different intensities, when they actually mean different things. Neutral
// gray at 0, orange for positive (more similar than typical), blue for
// negative (less similar than typical) — the standard treatment for
// correlation-shaped data (seaborn/matplotlib's RdBu/coolwarm, ColorBrewer's
// diverging category, etc.), not a novel choice.
function simColor(v) {
  const t = Math.max(-1, Math.min(1, v));
  const mid = [0xea, 0xee, 0xf2];   // neutral gray, v = 0
  if (t >= 0) {
    const hi = [0x7a, 0x1a, 0x05]; // deep orange/maroon, v = +1 (more similar than typical)
    const c = mid.map((m, i) => Math.round(m + (hi[i] - m) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  const lo = [0x0b, 0x3d, 0x91];   // deep blue, v = -1 (less similar than typical)
  const c = mid.map((m, i) => Math.round(m + (lo[i] - m) * -t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const HTREE_MARGIN = 10;

// Depth (generation, via BFS from the seed) and the classic leaf-counting
// tidy-tree slot assignment (sequential integer slots for leaves in
// traversal order; internal nodes center over their children) — computed
// once and shared by both the row tree and the column tree, since it's the
// same tree either way, just laid out along a different axis.
function computeTreeDepthSlot(children) {
  const depth = { 0: 0 };
  const bfsQueue = [0];
  while (bfsQueue.length) {
    const p = bfsQueue.shift();
    for (const c of (children[p] || [])) { depth[c] = depth[p] + 1; bfsQueue.push(c); }
  }
  const maxDepth = Math.max(0, ...Object.values(depth));

  let nextLeafSlot = 0;
  const slot = {};
  const leafOrder = [];
  (function assignSlot(idx) {
    const kids = children[idx] || [];
    if (!kids.length) { slot[idx] = nextLeafSlot++; leafOrder.push(idx); return slot[idx]; }
    const childSlots = kids.map(assignSlot);
    slot[idx] = (childSlots[0] + childSlots[childSlots.length - 1]) / 2;
    return slot[idx];
  })(0);

  return { depth, slot, leafOrder, maxDepth };
}

// Renders one axis's tree — either the row tree (orientation 'row': depth
// drives X, growing rightward, elbow is horizontal-then-vertical) or the
// column tree (orientation 'col': depth drives Y, growing downward, elbow is
// vertical-then-horizontal). Leaves are pulled flush to `flushMain` (the
// deepest column/row reached by any branch) with a dashed extension line.
function renderTreeAxis({ nodes, parentOf, children, depth, slot, orientation, mainOf, crossOf, flushMain, nodeSize }) {
  let edges = '', boxes = '';
  const isRow = orientation === 'row';

  // Same square box on both axes — sized to fit the heatmap's row/column
  // spacing exactly, so a row-tree node and a column-tree node look
  // identical (not a flattened rectangle on one axis and a square on the
  // other) and the whole thing reads as one symmetric diagram.
  const boxW = nodeSize, boxH = nodeSize;
  const fontSize = Math.max(6, Math.min(13, nodeSize * 0.4));

  for (const idxStr of Object.keys(nodes)) {
    const idx = parseInt(idxStr);
    const node = nodes[idx];
    const isLeaf = !(children[idx] || []).length;
    const naturalMain = mainOf(depth[idx] ?? 0);
    const cross = crossOf(slot[idx]);
    const renderMain = isLeaf ? flushMain : naturalMain;

    const parentIdx = parentOf[idx];
    if (parentIdx !== undefined) {
      const pMain = mainOf(depth[parentIdx] ?? 0);
      const pCross = crossOf(slot[parentIdx]);
      if (isRow) {
        const parentRight = pMain + boxW;
        edges += `<path d="M ${parentRight.toFixed(1)} ${pCross.toFixed(1)} H ${naturalMain.toFixed(1)} V ${cross.toFixed(1)}" class="sh-tree-edge"></path>`;
      } else {
        const parentBottom = pMain + boxH;
        edges += `<path d="M ${pCross.toFixed(1)} ${parentBottom.toFixed(1)} V ${naturalMain.toFixed(1)} H ${cross.toFixed(1)}" class="sh-tree-edge"></path>`;
      }
    }
    if (isLeaf && Math.abs(renderMain - naturalMain) > 0.5) {
      edges += isRow
        ? `<line x1="${naturalMain.toFixed(1)}" y1="${cross.toFixed(1)}" x2="${renderMain.toFixed(1)}" y2="${cross.toFixed(1)}" class="sh-tree-ext"></line>`
        : `<line x1="${cross.toFixed(1)}" y1="${naturalMain.toFixed(1)}" x2="${cross.toFixed(1)}" y2="${renderMain.toFixed(1)}" class="sh-tree-ext"></line>`;
    }

    const boxX = isRow ? renderMain : cross - boxW / 2;
    const boxY = isRow ? cross - boxH / 2 : renderMain;
    const label_ = idx === 0 ? 'Seed' : (node.val_avg != null ? node.val_avg.toFixed(2) : '–');
    boxes += `<g class="${isLeaf ? 'sh-tree-leaf' : 'sh-tree-node'}">
      <rect x="${boxX.toFixed(1)}" y="${boxY.toFixed(1)}" width="${boxW.toFixed(1)}" height="${boxH.toFixed(1)}" rx="4"></rect>
      <text x="${(boxX + boxW / 2).toFixed(1)}" y="${(boxY + boxH / 2 + fontSize * 0.35).toFixed(1)}" text-anchor="middle" style="font-size:${fontSize.toFixed(1)}px">${esc(label_)}</text>
      <title>${idx === 0 ? 'Seed' : `#${idx}`}${node.val_avg != null ? ` · val_avg ${node.val_avg.toFixed(2)}` : ''}</title>
    </g>`;
  }
  return edges + boxes;
}

// Renders one heatmap instance into `container` (a dedicated wrapper div, not
// shared with the other instance) — everything DOM-id-based below is prefixed
// with `idPrefix` so two instances can coexist on the same page without their
// zoom/pan controls or tooltips colliding.
function renderOneSimilarityHeatmap(container, sim, { idPrefix, title, captionSuffix, getViewState, setViewState }) {
  if (!sim || !sim.available) {
    const reason = sim && sim.reason ? esc(sim.reason) : 'Similarity data not available.';
    container.innerHTML = `<h3 class="sh-title">${esc(title)}</h3><div class="no-diff">${reason}</div>`;
    return;
  }

  const { nodes, parentOf, children } = buildEvolutionTree();
  const simByIdx = new Map(sim.items.map((it, i) => [it.idx, i]));

  const n = Object.keys(nodes).filter(idxStr => !(children[idxStr] || []).length).length;
  if (n < 2) {
    container.innerHTML = `<h3 class="sh-title">${esc(title)}</h3><div class="no-diff">Not enough leaf candidates yet to compare.</div>`;
    return;
  }

  const ROW_H = Math.max(28, Math.min(50, 1000 / n));
  const { depth, slot, leafOrder, maxDepth } = computeTreeDepthSlot(children);

  const items = leafOrder.map(idx => sim.items[simByIdx.get(idx)]);
  const simMatrix = leafOrder.map(a => leafOrder.map(b => sim.matrix[simByIdx.get(a)][simByIdx.get(b)]));
  function label(pos) { return items[pos].idx === 0 ? 'Seed' : `#${items[pos].idx}`; }

  // Observed range, kept only for the caption's informational note below.
  let simMin = 1, simMax = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const v = simMatrix[i][j];
      if (v < simMin) simMin = v;
      if (v > simMax) simMax = v;
    }
  }
  // No rescaling at all: color and the displayed numbers are the actual raw
  // mean-centered cosine similarity, on its true fixed range of -1 to 1 (-1
  // and 1 are the mathematical extremes; +1 is only reachable by
  // identical/self content). 0 = a typical pair in this run, not "no
  // similarity" — that's what mean-centering means.
  function cellFill(pi, pj) { return simColor(simMatrix[pi][pj]); }

  // ── Geometry: one square node size, shared by both trees, sized to fit the
  // heatmap's per-cell spacing (ROW_H) exactly — so a row-tree box and a
  // column-tree box are visually identical, not a flattened rectangle on one
  // axis and a square on the other. Depth-axis (generation) spacing is the
  // same on both axes too, for the same symmetry.
  const NODE_SIZE = Math.max(18, Math.min(40, ROW_H - 4));
  const HTREE_STEP = NODE_SIZE + 26;

  const rowMainOf = d => HTREE_MARGIN + d * HTREE_STEP;
  const rowFlushMain = rowMainOf(maxDepth);
  const rowTreeExtent = rowFlushMain + NODE_SIZE;

  const colMainOf = d => HTREE_MARGIN + d * HTREE_STEP;
  const colFlushMain = colMainOf(maxDepth);
  const colTreeExtent = colFlushMain + NODE_SIZE;

  const IDX_LABEL_W = 34, IDX_LABEL_H = 40;
  const HEATMAP_START_X = rowTreeExtent + IDX_LABEL_W + 14;
  const HEATMAP_START_Y = colTreeExtent + IDX_LABEL_H + 14;
  const gridSize = n * ROW_H;

  const rowCrossOf = s => HEATMAP_START_Y + s * ROW_H + ROW_H / 2;
  const colCrossOf = s => HEATMAP_START_X + s * ROW_H + ROW_H / 2;

  const rowTreeSvg = renderTreeAxis({
    nodes, parentOf, children, depth, slot, orientation: 'row',
    mainOf: rowMainOf, crossOf: rowCrossOf, flushMain: rowFlushMain, nodeSize: NODE_SIZE,
  });
  const colTreeSvg = renderTreeAxis({
    nodes, parentOf, children, depth, slot, orientation: 'col',
    mainOf: colMainOf, crossOf: colCrossOf, flushMain: colFlushMain, nodeSize: NODE_SIZE,
  });

  let rowIdxLabels = '';
  for (let i = 0; i < n; i++) {
    const y = rowCrossOf(i);
    rowIdxLabels += `<text x="${(rowTreeExtent + IDX_LABEL_W - 6).toFixed(1)}" y="${(y + 3.3).toFixed(1)}" text-anchor="end" class="sh-axis-lbl">${esc(label(i))}</text>`;
  }
  let colIdxLabels = '';
  const colLblFontSize = Math.max(6, Math.min(11, ROW_H * 0.32));
  for (let j = 0; j < n; j++) {
    const x = colCrossOf(j);
    const y = colTreeExtent + IDX_LABEL_H - 6;
    colIdxLabels += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="start" class="sh-axis-lbl" style="font-size:${colLblFontSize.toFixed(1)}px" transform="rotate(-60 ${x.toFixed(1)} ${y.toFixed(1)})">${esc(label(j))}</text>`;
  }

  // ── Heatmap cells + colorbar ────────────────────────────────────────────
  let cells = '';
  for (let pi = 0; pi < n; pi++) {
    for (let pj = 0; pj < n; pj++) {
      const x = HEATMAP_START_X + pj * ROW_H;
      const y = HEATMAP_START_Y + pi * ROW_H;
      cells += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(ROW_H - 1).toFixed(1)}" height="${(ROW_H - 1).toFixed(1)}"
        fill="${cellFill(pi, pj)}" class="sh-cell" data-pi="${pi}" data-pj="${pj}"></rect>`;
    }
  }

  const CBAR_GAP = 22, CBAR_W = 14, CBAR_LABEL_W = 46;
  const cbarX = HEATMAP_START_X + gridSize + CBAR_GAP;
  const cbarY = HEATMAP_START_Y, cbarH = gridSize;
  let legendStops = '';
  for (let t = 0; t <= 20; t++) legendStops += `<stop offset="${t * 5}%" stop-color="${simColor(1 - t / 10)}"/>`;

  const W = cbarX + CBAR_W + CBAR_LABEL_W;
  const H = HEATMAP_START_Y + gridSize + 20;

  container.innerHTML = `
    <h3 class="sh-title">${esc(title)}</h3>
    <div class="evo-tree-toolbar">
      <button class="evo-zoom-btn" id="${idPrefix}-zoom-in" title="Zoom in">+</button>
      <button class="evo-zoom-btn" id="${idPrefix}-zoom-out" title="Zoom out">−</button>
      <button class="evo-zoom-btn evo-zoom-reset" id="${idPrefix}-zoom-reset" title="Reset view">Reset</button>
      <span class="evo-zoom-hint">Scroll to zoom · drag to pan</span>
    </div>
    <div class="evo-tree-viewport" id="${idPrefix}-viewport">
      <div class="evo-tree-stage" id="${idPrefix}-stage" style="width:${W}px;height:${H}px;">
        <svg width="${W}" height="${H}" overflow="visible">
          <defs><linearGradient id="${idPrefix}-grad" x1="0" y1="0" x2="0" y2="1">${legendStops}</linearGradient></defs>
          ${rowTreeSvg}
          ${colTreeSvg}
          ${rowIdxLabels}
          ${colIdxLabels}
          ${cells}
          <rect x="${cbarX.toFixed(1)}" y="${cbarY}" width="${CBAR_W}" height="${cbarH}" fill="url(#${idPrefix}-grad)" stroke="#d0d7de"></rect>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + 4).toFixed(1)}" class="sh-axis-lbl">1.00</text>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + cbarH / 2 + 3).toFixed(1)}" class="sh-axis-lbl">0.00</text>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + cbarH).toFixed(1)}" class="sh-axis-lbl">-1.00</text>
        </svg>
      </div>
      <div class="sh-tooltip" id="${idPrefix}-tooltip"></div>
    </div>
    <div class="sh-caption">${n} leaf candidates out of ${sim.items.length} total · embedding model: ${esc(sim.model)}, mean-centered · ${captionSuffix} · excludes static boilerplate (agent wrapper, provider config) · row/column order follows the tree's leaf order on both axes · diverging orange/blue scale, fixed -1.00 to 1.00, actual values, no rescaling · this run's observed range was ${simMin.toFixed(2)}–${simMax.toFixed(2)}</div>`;

  setupPanZoom({
    viewportId: `${idPrefix}-viewport`, stageId: `${idPrefix}-stage`,
    zoomInId: `${idPrefix}-zoom-in`, zoomOutId: `${idPrefix}-zoom-out`, zoomResetId: `${idPrefix}-zoom-reset`,
    contentW: W, contentH: H,
    getState: getViewState,
    setState: setViewState,
    initialK: 1,  // open at true size (bigger, legible squares) rather than shrinking to fit — pan/zoom covers the rest
  });

  const viewport = container.querySelector(`#${idPrefix}-viewport`);
  const tooltip = container.querySelector(`#${idPrefix}-tooltip`);
  container.querySelectorAll('.sh-cell').forEach(c => {
    c.addEventListener('mouseenter', () => {
      const pi = +c.dataset.pi, pj = +c.dataset.pj;
      if (pi === pj) {
        tooltip.textContent = `${label(pi)} · self`;
      } else {
        tooltip.textContent = `${label(pi)} vs ${label(pj)} · ${simMatrix[pi][pj].toFixed(2)} mean-centered similarity`;
      }
      tooltip.style.display = 'block';
    });
    c.addEventListener('mousemove', e => {
      const r = viewport.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 12) + 'px';
      tooltip.style.top = (e.clientY - r.top - 10) + 'px';
    });
    c.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  });
}

// Builds a *pruned* ancestry tree over {seed, top-N candidates, and every real
// intermediate ancestor that lies on a path between them} — the union of the
// seed-to-candidate paths for each top-N candidate, walked via the real
// parent links (not a skip-connection). A branch that never leads to a top-N
// candidate is dropped entirely (never even visited), but a branch that does
// lead to one keeps every real hop along the way — including nodes that
// aren't themselves top-N (e.g. #45's real chain runs through #13, #11, #10,
// #2, #1 before the seed; only #2 happens to also be top-N, the rest are kept
// purely because they're on the path). Two top-N candidates that are directly
// related (e.g. #38 → #41) end up linked with nothing in between, same as
// before — that's just what "the real path" collapses to when there's no
// intermediate hop.
//
// Row/column order comes from the tree itself, not from a val_avg ranking —
// ranking two candidates next to each other just because their scores happen
// to be close says nothing about whether they're related, and unrelated
// neighbors are exactly what makes a tree's edges cross the grid. A pre-order
// walk (visit a top-N node the instant it's reached, then recurse into its
// children) visits every top-N node exactly once, and — since it's a single
// depth-first walk — each subtree occupies one contiguous block of positions.
// That's what guarantees the edges never cross: no top-N node's position ever
// needs to jump outside its own branch's block. Every other kept node (seed +
// pass-through ancestors, which have no grid row of their own) is positioned
// bottom-up afterward: centered between its own children's positions.
function buildTopNPrunedTree(rows, fullParentOf) {
  const topSet = new Set(rows.map(r => r.idx));

  const keep = new Set([0]);
  for (const r of rows) {
    let idx = r.idx;
    while (idx !== undefined && idx !== null && !keep.has(idx)) {
      keep.add(idx);
      idx = fullParentOf[idx];
    }
  }

  const parentOf = {};
  const children = {};
  for (const idx of keep) {
    if (idx === 0) continue;
    const p = fullParentOf[idx];
    parentOf[idx] = p;
    (children[p] = children[p] || []).push(idx);
  }

  const depth = { 0: 0 };
  const bfsQueue = [0];
  while (bfsQueue.length) {
    const p = bfsQueue.shift();
    for (const c of (children[p] || [])) { depth[c] = depth[p] + 1; bfsQueue.push(c); }
  }
  const maxDepth = Math.max(0, ...Object.values(depth));

  const dfsOrder = [];
  (function walk(idx) {
    if (topSet.has(idx)) dfsOrder.push(idx);
    for (const c of (children[idx] || [])) walk(c);
  })(0);
  const rankOf = new Map(dfsOrder.map((idx, i) => [idx, i]));

  const cross = {};
  (function assignCross(idx) {
    // Always recurse into children first, even for a top-N node — a top-N
    // candidate can itself be an ancestor of others (e.g. #2), and its own
    // fixed rank position must not short-circuit computing its descendants'.
    const kids = children[idx] || [];
    for (const k of kids) assignCross(k);
    if (rankOf.has(idx)) { cross[idx] = rankOf.get(idx); return cross[idx]; }
    if (!kids.length) { cross[idx] = (rows.length - 1) / 2; return cross[idx]; }
    const kidCrosses = kids.map(k => cross[k]);
    cross[idx] = (Math.min(...kidCrosses) + Math.max(...kidCrosses)) / 2;
    return cross[idx];
  })(0);

  return { keep, parentOf, children, depth, maxDepth, cross, rankOf, dfsOrder };
}

// Same visual language as renderTreeAxis (elbow edges, rounded boxes) but no
// leaf-flush alignment — every node here sits at its true depth, since depth
// is part of what this view is showing. Top-N candidates render like the full
// tree's leaves (orange border); the seed and pass-through ancestors render
// like its internal nodes (plain border) — same classes, same meaning.
function renderPrunedTreeAxis({ keep, parentOf, depth, cross, rankOf, valOf, orientation, mainOf, crossOf, nodeSize }) {
  let edges = '', boxes = '';
  const isRow = orientation === 'row';
  const boxW = nodeSize, boxH = nodeSize;
  const fontSize = Math.max(6, Math.min(13, nodeSize * 0.4));

  for (const idx of keep) {
    const main = mainOf(depth[idx] ?? 0);
    const c = crossOf(cross[idx]);
    const p = parentOf[idx];
    if (p !== undefined) {
      const pMain = mainOf(depth[p] ?? 0);
      const pCross = crossOf(cross[p]);
      if (isRow) {
        edges += `<path d="M ${(pMain + boxW).toFixed(1)} ${pCross.toFixed(1)} H ${main.toFixed(1)} V ${c.toFixed(1)}" class="sh-tree-edge"></path>`;
      } else {
        edges += `<path d="M ${pCross.toFixed(1)} ${(pMain + boxH).toFixed(1)} V ${main.toFixed(1)} H ${c.toFixed(1)}" class="sh-tree-edge"></path>`;
      }
    }
    const boxX = isRow ? main : c - boxW / 2;
    const boxY = isRow ? c - boxH / 2 : main;
    const v = valOf(idx);
    const label_ = idx === 0 ? 'Seed' : (v != null ? v.toFixed(2) : `#${idx}`);
    const isTopN = rankOf.has(idx);
    boxes += `<g class="${isTopN ? 'sh-tree-leaf' : 'sh-tree-node'}">
      <rect x="${boxX.toFixed(1)}" y="${boxY.toFixed(1)}" width="${boxW.toFixed(1)}" height="${boxH.toFixed(1)}" rx="4"></rect>
      <text x="${(boxX + boxW / 2).toFixed(1)}" y="${(boxY + boxH / 2 + fontSize * 0.35).toFixed(1)}" text-anchor="middle" style="font-size:${fontSize.toFixed(1)}px">${esc(label_)}</text>
      <title>${idx === 0 ? 'Seed' : `#${idx}`}${v != null ? ` · val_avg ${v.toFixed(2)}` : ''}${isTopN ? '' : ' · not in top-N (kept: on path to a top-N candidate)'}</title>
    </g>`;
  }
  return edges + boxes;
}

// Same idea as renderOneSimilarityHeatmap, but for a fixed top-N-by-val_avg
// subset instead of every tree leaf: membership is by val_avg (the top N
// scores), but row/column *order* follows the pruned ancestry tree's
// depth-first structure, not the ranking — see buildTopNPrunedTree for why.
function renderTopNSimilarityHeatmap(container, sim, { idPrefix, title, captionSuffix, topN, getViewState, setViewState }) {
  if (!sim || !sim.available) {
    const reason = sim && sim.reason ? esc(sim.reason) : 'Similarity data not available.';
    container.innerHTML = `<h3 class="sh-title">${esc(title)}</h3><div class="no-diff">${reason}</div>`;
    return;
  }

  const simByIdx = new Map(sim.items.map((it, i) => [it.idx, i]));
  const topRows = buildValidationSeries()
    .filter(r => simByIdx.has(r.idx))
    .slice()
    .sort((a, b) => b.val - a.val)
    .slice(0, topN);
  const n = topRows.length;
  if (n < 2) {
    container.innerHTML = `<h3 class="sh-title">${esc(title)}</h3><div class="no-diff">Not enough scored candidates yet to compare.</div>`;
    return;
  }

  const { nodes: fullNodes, parentOf: fullParentOf } = buildEvolutionTree();
  const {
    keep: prunedKeep, parentOf: prunedParentOf, depth: prunedDepth,
    maxDepth: prunedMaxDepth, cross: prunedCross, rankOf: prunedRankOf, dfsOrder,
  } = buildTopNPrunedTree(topRows, fullParentOf);

  // Grid rows/columns follow the tree's depth-first order (dfsOrder), not the
  // val_avg ranking used only to decide *membership* in the top-N set.
  const topRowByIdx = new Map(topRows.map(r => [r.idx, r]));
  const rows = dfsOrder.map(idx => topRowByIdx.get(idx));

  // The candidate idx (#45, #2, ...) is an internal bookkeeping number — what
  // actually matters here is how each candidate placed. valRankByIdx is the
  // 1-based rank by val_avg (1 = best), independent of the tree-order the
  // grid itself is laid out in.
  const valRankByIdx = new Map(topRows.map((r, i) => [r.idx, i + 1]));

  const simMatrix = rows.map(a => rows.map(b => sim.matrix[simByIdx.get(a.idx)][simByIdx.get(b.idx)]));
  function label(pos) { return rows[pos].idx === 0 ? 'Seed' : `#${valRankByIdx.get(rows[pos].idx)}`; }
  function sublabel(pos) { return rows[pos].val.toFixed(2); }

  let simMin = 1, simMax = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const v = simMatrix[i][j];
      if (v < simMin) simMin = v;
      if (v > simMax) simMax = v;
    }
  }
  function cellFill(pi, pj) { return simColor(simMatrix[pi][pj]); }
  const valOf = idx => fullNodes[idx]?.val_avg ?? null;

  const ROW_H = Math.max(32, Math.min(60, 480 / n));
  const NODE_SIZE = Math.max(16, Math.min(30, ROW_H - 4));
  const HTREE_STEP = NODE_SIZE + 18;
  const rowMainOf = d => HTREE_MARGIN + d * HTREE_STEP;
  const colMainOf = d => HTREE_MARGIN + d * HTREE_STEP;
  const rowTreeExtent = rowMainOf(prunedMaxDepth) + NODE_SIZE;
  const colTreeExtent = colMainOf(prunedMaxDepth) + NODE_SIZE;

  const IDX_LABEL_W = 60, IDX_LABEL_H = 56;
  const HEATMAP_START_X = rowTreeExtent + IDX_LABEL_W + 14;
  const HEATMAP_START_Y = colTreeExtent + IDX_LABEL_H + 14;
  const gridSize = n * ROW_H;

  const rowCrossOf = s => HEATMAP_START_Y + s * ROW_H + ROW_H / 2;
  const colCrossOf = s => HEATMAP_START_X + s * ROW_H + ROW_H / 2;

  const rowTreeSvg = renderPrunedTreeAxis({
    keep: prunedKeep, parentOf: prunedParentOf, depth: prunedDepth, cross: prunedCross, rankOf: prunedRankOf, valOf,
    orientation: 'row', mainOf: rowMainOf, crossOf: rowCrossOf, nodeSize: NODE_SIZE,
  });
  const colTreeSvg = renderPrunedTreeAxis({
    keep: prunedKeep, parentOf: prunedParentOf, depth: prunedDepth, cross: prunedCross, rankOf: prunedRankOf, valOf,
    orientation: 'col', mainOf: colMainOf, crossOf: colCrossOf, nodeSize: NODE_SIZE,
  });

  let rowLabels = '';
  for (let i = 0; i < n; i++) {
    const y = rowCrossOf(i);
    rowLabels += `<text x="${(rowTreeExtent + IDX_LABEL_W - 6).toFixed(1)}" y="${(y - 2).toFixed(1)}" text-anchor="end" class="sh-axis-lbl">${esc(label(i))}</text>`;
    rowLabels += `<text x="${(rowTreeExtent + IDX_LABEL_W - 6).toFixed(1)}" y="${(y + 10).toFixed(1)}" text-anchor="end" class="sh-axis-sublbl">${esc(sublabel(i))}</text>`;
  }
  let colLabels = '';
  for (let j = 0; j < n; j++) {
    const x = colCrossOf(j);
    const y = colTreeExtent + IDX_LABEL_H - 6;
    colLabels += `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="start" class="sh-axis-lbl" transform="rotate(-60 ${x.toFixed(1)} ${y.toFixed(1)})">${esc(label(j))} · ${esc(sublabel(j))}</text>`;
  }

  let cells = '';
  for (let pi = 0; pi < n; pi++) {
    for (let pj = 0; pj < n; pj++) {
      const x = HEATMAP_START_X + pj * ROW_H;
      const y = HEATMAP_START_Y + pi * ROW_H;
      cells += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(ROW_H - 1).toFixed(1)}" height="${(ROW_H - 1).toFixed(1)}"
        fill="${cellFill(pi, pj)}" class="sh-cell" data-pi="${pi}" data-pj="${pj}"></rect>`;
    }
  }

  const CBAR_GAP = 22, CBAR_W = 14, CBAR_LABEL_W = 46;
  const cbarX = HEATMAP_START_X + gridSize + CBAR_GAP;
  const cbarY = HEATMAP_START_Y, cbarH = gridSize;
  let legendStops = '';
  for (let t = 0; t <= 20; t++) legendStops += `<stop offset="${t * 5}%" stop-color="${simColor(1 - t / 10)}"/>`;

  const W = cbarX + CBAR_W + CBAR_LABEL_W;
  const H = HEATMAP_START_Y + gridSize + 20;

  container.innerHTML = `
    <h3 class="sh-title">${esc(title)}</h3>
    <div class="evo-tree-toolbar">
      <button class="evo-zoom-btn" id="${idPrefix}-zoom-in" title="Zoom in">+</button>
      <button class="evo-zoom-btn" id="${idPrefix}-zoom-out" title="Zoom out">−</button>
      <button class="evo-zoom-btn evo-zoom-reset" id="${idPrefix}-zoom-reset" title="Reset view">Reset</button>
      <span class="evo-zoom-hint">Scroll to zoom · drag to pan</span>
    </div>
    <div class="evo-tree-viewport" id="${idPrefix}-viewport">
      <div class="evo-tree-stage" id="${idPrefix}-stage" style="width:${W}px;height:${H}px;">
        <svg width="${W}" height="${H}" overflow="visible">
          <defs><linearGradient id="${idPrefix}-grad" x1="0" y1="0" x2="0" y2="1">${legendStops}</linearGradient></defs>
          ${rowTreeSvg}
          ${colTreeSvg}
          ${rowLabels}
          ${colLabels}
          ${cells}
          <rect x="${cbarX.toFixed(1)}" y="${cbarY}" width="${CBAR_W}" height="${cbarH}" fill="url(#${idPrefix}-grad)" stroke="#d0d7de"></rect>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + 4).toFixed(1)}" class="sh-axis-lbl">1.00</text>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + cbarH / 2 + 3).toFixed(1)}" class="sh-axis-lbl">0.00</text>
          <text x="${(cbarX + CBAR_W + 5).toFixed(1)}" y="${(cbarY + cbarH).toFixed(1)}" class="sh-axis-lbl">-1.00</text>
        </svg>
      </div>
      <div class="sh-tooltip" id="${idPrefix}-tooltip"></div>
    </div>
    <div class="sh-caption">top ${n} candidates by val_avg (membership only — rows/columns are NOT ranked by score, they follow the tree's depth-first order so its edges never cross the grid) · embedding model: ${esc(sim.model)}, mean-centered · ${captionSuffix} · pruned ancestry tree: the real seed-to-candidate path for every top-${n} candidate is kept in full (intermediate ancestors included, plain boxes), any branch that never reaches a top-${n} candidate is dropped entirely · excludes static boilerplate (agent wrapper, provider config) · diverging orange/blue scale, fixed -1.00 to 1.00, actual values, no rescaling · this run's observed range was ${simMin.toFixed(2)}–${simMax.toFixed(2)}</div>`;

  setupPanZoom({
    viewportId: `${idPrefix}-viewport`, stageId: `${idPrefix}-stage`,
    zoomInId: `${idPrefix}-zoom-in`, zoomOutId: `${idPrefix}-zoom-out`, zoomResetId: `${idPrefix}-zoom-reset`,
    contentW: W, contentH: H,
    getState: getViewState,
    setState: setViewState,
    initialK: 1,
  });

  function idxTag(pos) { return rows[pos].idx === 0 ? '' : ` (candidate #${rows[pos].idx})`; }

  const viewport = container.querySelector(`#${idPrefix}-viewport`);
  const tooltip = container.querySelector(`#${idPrefix}-tooltip`);
  container.querySelectorAll('.sh-cell').forEach(c => {
    c.addEventListener('mouseenter', () => {
      const pi = +c.dataset.pi, pj = +c.dataset.pj;
      if (pi === pj) {
        tooltip.textContent = `${label(pi)}${idxTag(pi)} (val_avg ${sublabel(pi)}) · self`;
      } else {
        tooltip.textContent = `${label(pi)}${idxTag(pi)} (${sublabel(pi)}) vs ${label(pj)}${idxTag(pj)} (${sublabel(pj)}) · ${simMatrix[pi][pj].toFixed(2)} mean-centered similarity`;
      }
      tooltip.style.display = 'block';
    });
    c.addEventListener('mousemove', e => {
      const r = viewport.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 12) + 'px';
      tooltip.style.top = (e.clientY - r.top - 10) + 'px';
    });
    c.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
  });
}

// Four independent instances stacked one under the other: the two tree-based
// leaf heatmaps (file×file and whole-candidate), then the same two similarity
// computations again but restricted to the top-10 by val_avg with no tree —
// same underlying data, different subset/layout for each pair.
function renderSimilarityHeatmap(el) {
  el.innerHTML = `
    <div id="sh-new-wrap"></div>
    <div id="sh-legacy-wrap" class="sh-section-divider"></div>
    <div id="sh-top10-wrap" class="sh-section-divider"></div>
    <div id="sh-top10-legacy-wrap" class="sh-section-divider"></div>`;

  renderOneSimilarityHeatmap(el.querySelector('#sh-new-wrap'), DATA.similarity, {
    idPrefix: 'sh',
    title: 'File × File Leaf Similarity Heatmap',
    captionSuffix: 'one-to-one file matching (prompt.md, AGENTS.md, each skills/*/SKILL.md embedded separately, not pooled)',
    getViewState: () => simViewState,
    setViewState: s => { simViewState = s; },
  });

  renderOneSimilarityHeatmap(el.querySelector('#sh-legacy-wrap'), DATA.similarity_legacy, {
    idPrefix: 'shleg',
    title: 'Whole-Candidate Leaf Similarity Heatmap',
    captionSuffix: 'whole candidate pooled into a single vector, truncated at ~8k tokens (24,000 chars)',
    getViewState: () => simLegacyViewState,
    setViewState: s => { simLegacyViewState = s; },
  });

  renderTopNSimilarityHeatmap(el.querySelector('#sh-top10-wrap'), DATA.similarity, {
    idPrefix: 'shtop',
    title: 'File × File Top-10 Similarity Heatmap',
    captionSuffix: 'one-to-one file matching (prompt.md, AGENTS.md, each skills/*/SKILL.md embedded separately, not pooled)',
    topN: 10,
    getViewState: () => simTop10ViewState,
    setViewState: s => { simTop10ViewState = s; },
  });

  renderTopNSimilarityHeatmap(el.querySelector('#sh-top10-legacy-wrap'), DATA.similarity_legacy, {
    idPrefix: 'shtopleg',
    title: 'Whole-Candidate Top-10 Similarity Heatmap',
    captionSuffix: 'whole candidate pooled into a single vector, truncated at ~8k tokens (24,000 chars)',
    topN: 10,
    getViewState: () => simTop10LegacyViewState,
    setViewState: s => { simTop10LegacyViewState = s; },
  });
}
