// ── Evolution view ────────────────────────────────────────────────────────
// Skill files live at ".opencode/skills/<name>/SKILL.md" (or "skills/<name>/SKILL.md"
// for older snapshots) — match on the "skills/" segment anywhere in the path,
// not just at the start.
function isSkillFile(f) {
  return /(^|\/)skills\//.test(f);
}

function fileColor(f) {
  if (isSkillFile(f)) return { bg: '#fff8c5', border: '#d4a72c', text: '#9a6700' };
  return { bg: '#ffebe9', border: '#cf222e', text: '#cf222e' };
}

function fileDisplayName(f) {
  if (isSkillFile(f)) {
    const parts = f.split('/');
    const i = parts.indexOf('skills');
    return (i >= 0 && parts.length > i + 1) ? parts[i + 1] : f;
  }
  return f.split('/').pop();
}

function buildEvolutionTree() {
  const nodes    = {};  // candidateIdx → { idx, iter, files }
  const parentOf = {};  // childIdx → parentIdx
  const children = {};  // parentIdx → [childIdx, ...]

  // Seed node (candidate 0)
  const seedIter = DATA.iterations[0];
  nodes[0] = { idx: 0, iter: 0, files: seedIter?.parent_files || {}, val_avg: null };

  for (const it of DATA.iterations) {
    if (it.status !== 'accepted') continue;
    if (!it.child_files || !Object.keys(it.child_files).length) continue;
    const newIdx = it.child_candidate_idx;
    if (newIdx === null || newIdx === undefined || newIdx === 0) continue;

    nodes[newIdx] = { idx: newIdx, iter: it.id, files: it.child_files, val_avg: null };
    if (it.parent_idx !== null && it.parent_idx !== undefined) {
      parentOf[newIdx] = it.parent_idx;
      if (!children[it.parent_idx]) children[it.parent_idx] = [];
      children[it.parent_idx].push(newIdx);
    }
  }

  // Fill val_avg from pool.json (for candidates that appear there), and track
  // pool membership via the shared everPooled helper (utils.js).
  for (const it of DATA.iterations) {
    if (!it.pool_json) continue;
    const pool = JSON.parse(it.pool_json);
    for (const c of pool.candidates) {
      if (nodes[c.idx] && c.val_avg != null) nodes[c.idx].val_avg = c.val_avg;
    }
  }
  const everPooled = computeEverPooledSet();
  // Fill val_avg from child_val in evolution.json (for candidates absent from pool)
  for (const it of DATA.iterations) {
    if (it.status !== 'accepted') continue;
    const idx = it.child_candidate_idx;
    if (idx == null || !nodes[idx] || nodes[idx].val_avg != null) continue;
    const vals = it.child_val ? Object.values(it.child_val).filter(v => v != null) : [];
    if (vals.length) nodes[idx].val_avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  // Seed val_avg from first iteration's parent_val
  if (nodes[0] && nodes[0].val_avg == null) {
    const it0 = DATA.iterations[0];
    const vals = it0?.parent_val ? Object.values(it0.parent_val).filter(v => v != null) : [];
    if (vals.length) nodes[0].val_avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  for (const idxStr of Object.keys(nodes)) {
    nodes[idxStr].pruned = !everPooled.has(parseInt(idxStr));
  }

  return { nodes, parentOf, children };
}

// Collapsed by default (just label + val avg); click expands to show
// prompt/AGENTS.md/skills; clicking an already-expanded candidate collapses it
// back — a plain toggle.
function toggleTreeNode(idx) {
  if (treeExpanded.has(idx)) treeExpanded.delete(idx); else treeExpanded.add(idx);
  renderEvolutionTree(document.getElementById('evo-tab-body'));
}

function evoNodeHtml(idx, node) {
  const label    = idx === 0 ? 'Seed' : `Cand #${idx}`;
  const sub      = node.iter > 0 ? `iter ${pad(node.iter)}` : '';
  const valBadge = node.val_avg != null
    ? `<div class="evo-val-badge">val avg ${node.val_avg.toFixed(2)}</div>`
    : '';
  const expanded = treeExpanded.has(idx);
  const caret    = `<span class="evo-caret">${expanded ? '▾' : '▸'}</span>`;
  const pruneCls = node.pruned ? 'pruned' : 'kept';
  const pruneHint = node.pruned ? 'Pruned (dominated on the Pareto front)' : 'Kept (on the Pareto front)';
  const hint     = (expanded ? 'Click to collapse' : 'Click to expand') + ' · ' + pruneHint;

  let body = '';
  if (expanded) {
    const blocks = Object.keys(node.files)
      .filter(f => f === 'prompt.md' || f === 'AGENTS.md' || isSkillFile(f))
      .sort()
      .map(f => {
        const lines    = (node.files[f] || '').split('\n').length;
        const name     = fileDisplayName(f);
        const col      = fileColor(f);
        const isSkill  = isSkillFile(f);
        const radius   = isSkill ? '14px' : '5px';
        const extraPad = Math.min(60, Math.max(0, lines * 3 - 8));
        return `<div class="evo-file-block"
          style="border-radius:${radius};padding-right:${8 + extraPad}px;background:${col.bg};border-color:${col.border};color:${col.text}"
          title="${esc(f)} — ${lines} lines">${esc(name)}</div>`;
      }).join('');
    body = `<div class="evo-files">${blocks}</div>`;
  }

  return `<div class="evo-candidate ${pruneCls}${expanded ? ' expanded' : ''}" data-evo-idx="${idx}" onclick="toggleTreeNode(${idx})" title="${esc(hint)}">
    <div class="evo-candidate-hdr">${caret}${esc(label)}<span class="evo-candidate-sub">${esc(sub)}</span></div>
    ${valBadge}
    ${body}
  </div>`;
}

function renderEvolutionView(el) {
  const tabs = [
    { id: 'tree',       label: 'Tree' },
    { id: 'validation', label: 'Validation Performance' },
    { id: 'similarity', label: 'Similarity Heatmap' },
  ];
  const tabBar = `<div class="tab-bar">${
    tabs.map(t => `<button class="tab${selEvoTab===t.id?' sel':''}" onclick="selEvoTab='${t.id}';renderMain()">${t.label}</button>`).join('')
  }</div>`;
  el.innerHTML = tabBar + '<div id="evo-tab-body"></div>';
  const body = document.getElementById('evo-tab-body');

  if (selEvoTab === 'validation') {
    renderValidationChart(body);
  } else if (selEvoTab === 'similarity') {
    renderSimilarityHeatmap(body);
  } else {
    renderEvolutionTree(body);
  }
}

// ── Tree layout ──────────────────────────────────────────────────────────
// Every position (nodes AND edges) is computed here, purely from the tree
// structure — never measured from the rendered DOM. That's the point: with a
// single CSS transform driving pan/zoom on a container that holds both nodes
// and edges, they can never drift out of sync, at any zoom level, because
// there's nothing to re-measure or get stale.
const TREE_NODE_W   = 210;
const TREE_GUTTER_X = 50;
const TREE_GUTTER_Y = 70;
const TREE_PAD_V     = 24, TREE_HEADER_H = 26, TREE_BADGE_H = 40;
const TREE_BLOCK_H   = 32, TREE_BLOCK_GAP = 5;

function evoBlockCount(node) {
  return Object.keys(node.files).filter(f => f === 'prompt.md' || f === 'AGENTS.md' || isSkillFile(f)).length;
}

function evoNodeHeight(node) {
  const bc = treeExpanded.has(node.idx) ? evoBlockCount(node) : 0;
  return TREE_PAD_V + TREE_HEADER_H + TREE_BADGE_H + (bc > 0 ? bc * TREE_BLOCK_H + Math.max(0, bc - 1) * TREE_BLOCK_GAP : 0);
}

function computeTreeLayout(nodes, children) {
  // depth (level) of every node via BFS from the seed
  const depth = { 0: 0 };
  const bfsQueue = [0];
  while (bfsQueue.length) {
    const p = bfsQueue.shift();
    for (const c of (children[p] || [])) { depth[c] = depth[p] + 1; bfsQueue.push(c); }
  }

  // row height per level = tallest node at that level, so rows pack tightly
  const levelHeight = {};
  for (const idxStr of Object.keys(nodes)) {
    const idx = parseInt(idxStr);
    const d = depth[idx] ?? 0;
    levelHeight[d] = Math.max(levelHeight[d] || 0, evoNodeHeight(nodes[idx]));
  }
  const maxDepth = Math.max(...Object.keys(levelHeight).map(Number));
  const levelY = {};
  let cumY = 0;
  for (let d = 0; d <= maxDepth; d++) {
    levelY[d] = cumY;
    cumY += (levelHeight[d] || 0) + TREE_GUTTER_Y;
  }

  // classic leaf-counting tidy-tree X assignment: each node centers over its
  // children; leaves get one slot each, in traversal order.
  let nextLeafSlot = 0;
  const slotX = {};
  (function assignSlot(idx) {
    const kids = children[idx] || [];
    if (!kids.length) { slotX[idx] = nextLeafSlot++; return slotX[idx]; }
    const childCenters = kids.map(assignSlot);
    slotX[idx] = (childCenters[0] + childCenters[childCenters.length - 1]) / 2;
    return slotX[idx];
  })(0);

  const SLOT_W = TREE_NODE_W + TREE_GUTTER_X;
  const positions = {};
  for (const idxStr of Object.keys(nodes)) {
    const idx = parseInt(idxStr);
    positions[idx] = {
      x: slotX[idx] * SLOT_W,
      y: levelY[depth[idx] ?? 0],
      w: TREE_NODE_W,
      h: evoNodeHeight(nodes[idx]),
    };
  }

  return { positions, totalW: nextLeafSlot * SLOT_W, totalH: cumY };
}

// ── Pan / zoom ───────────────────────────────────────────────────────────
// One CSS transform on a stage inside a fixed-size viewport; content (nodes,
// edges, cells, ...) is sized once in real pixels and never recomputed on
// zoom. Generic so both the Tree tab and the Similarity heatmap tab (which
// has the same "scroll to zoom, drag to pan, +/-/Reset" needs) share one
// implementation instead of drifting apart.
function setupPanZoom({ viewportId, stageId, zoomInId, zoomOutId, zoomResetId, contentW, contentH, getState, setState, dragIgnoreSelector, initialK }) {
  const viewport = document.getElementById(viewportId);
  const stage    = document.getElementById(stageId);
  if (!viewport || !stage) return;

  const vw = viewport.clientWidth  || 900;
  const vh = viewport.clientHeight || 600;
  const fitK = Math.min(1, (vw - 60) / contentW, (vh - 60) / contentH);
  // Some content (e.g. the similarity heatmap) wants to open at a fixed,
  // legible size rather than always shrinking to fit the viewport — pass
  // initialK to use that instead of fitK, panning/zooming covers the rest.
  const startK = initialK != null ? initialK : Math.max(0.1, fitK);

  // Re-renders (e.g. expanding a tree node) can shift content, but that
  // shouldn't yank the view back to "fit" every time — restore whatever
  // pan/zoom the user had, and only compute a fresh fit on first render.
  let k, tx, ty;
  const saved = getState();
  if (saved) {
    ({ k, tx, ty } = saved);
  } else {
    k  = startK;
    tx = Math.max(20, (vw - contentW * k) / 2);
    ty = 20;
  }

  function apply() {
    stage.style.transform = `translate(${tx}px, ${ty}px) scale(${k})`;
    setState({ k, tx, ty });
  }
  apply();

  viewport.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = viewport.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const prevK = k;
    k = Math.min(3, Math.max(0.08, k * (1 - e.deltaY * 0.0015)));
    tx = mx - ((mx - tx) / prevK) * k;
    ty = my - ((my - ty) / prevK) * k;
    apply();
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  viewport.addEventListener('mousedown', e => {
    if (dragIgnoreSelector && e.target.closest(dragIgnoreSelector)) return;  // let clicks through
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    viewport.classList.add('dragging');
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    tx += e.clientX - lastX; ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    apply();
  });
  window.addEventListener('mouseup', () => { dragging = false; viewport.classList.remove('dragging'); });

  const zoomIn    = document.getElementById(zoomInId);
  const zoomOut   = document.getElementById(zoomOutId);
  const zoomReset = document.getElementById(zoomResetId);
  if (zoomIn)    zoomIn.onclick    = () => { k = Math.min(3, k * 1.25); apply(); };
  if (zoomOut)   zoomOut.onclick   = () => { k = Math.max(0.08, k / 1.25); apply(); };
  if (zoomReset) zoomReset.onclick = () => {
    k = startK;
    tx = Math.max(20, (viewport.clientWidth - contentW * k) / 2);
    ty = 20;
    apply();
  };
}

function setupTreePanZoom(contentW, contentH) {
  setupPanZoom({
    viewportId: 'evo-tree-viewport', stageId: 'evo-tree-stage',
    zoomInId: 'evo-zoom-in', zoomOutId: 'evo-zoom-out', zoomResetId: 'evo-zoom-reset',
    contentW, contentH,
    getState: () => treeViewState,
    setState: s => { treeViewState = s; },
    dragIgnoreSelector: '.evo-candidate',
  });
}

function renderEvolutionTree(el) {
  const { nodes, parentOf, children } = buildEvolutionTree();
  if (!Object.keys(nodes).length) {
    el.innerHTML = '<div class="no-diff">No candidates found.</div>';
    return;
  }

  const { positions, totalW, totalH } = computeTreeLayout(nodes, children);

  let nodesHtml = '';
  for (const [idxStr, node] of Object.entries(nodes)) {
    const idx = parseInt(idxStr);
    const pos = positions[idx];
    nodesHtml += `<div style="position:absolute;left:${pos.x}px;top:${pos.y}px;width:${pos.w}px;">${evoNodeHtml(idx, node)}</div>`;
  }

  let edgesHtml = '';
  for (const [childStr, parentIdx] of Object.entries(parentOf)) {
    const childIdx = parseInt(childStr);
    const p = positions[parentIdx], c = positions[childIdx];
    if (!p || !c) continue;
    const x1 = p.x + p.w / 2, y1 = p.y + p.h;
    const x2 = c.x + c.w / 2, y2 = c.y;
    const midY = (y1 + y2) / 2;
    edgesHtml += `<path d="M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}" fill="none" stroke="#b0b7be" stroke-width="3"/>`;
  }

  el.innerHTML = `
    <div class="evo-tree-toolbar">
      <button class="evo-zoom-btn" id="evo-zoom-in" title="Zoom in">+</button>
      <button class="evo-zoom-btn" id="evo-zoom-out" title="Zoom out">−</button>
      <button class="evo-zoom-btn evo-zoom-reset" id="evo-zoom-reset" title="Reset view">Reset</button>
      <span class="evo-zoom-hint">Scroll to zoom · drag to pan</span>
      <span class="evo-tree-legend">
        <span class="evo-legend-item"><span class="evo-legend-swatch kept"></span>Kept (on Pareto front)</span>
        <span class="evo-legend-item"><span class="evo-legend-swatch pruned"></span>Pruned (dominated)</span>
      </span>
    </div>
    <div class="evo-tree-viewport" id="evo-tree-viewport">
      <div class="evo-tree-stage" id="evo-tree-stage" style="width:${totalW}px;height:${totalH}px;">
        <svg class="evo-tree-edges" width="${totalW}" height="${totalH}">${edgesHtml}</svg>
        ${nodesHtml}
      </div>
    </div>`;

  setupTreePanZoom(totalW, totalH);
}
