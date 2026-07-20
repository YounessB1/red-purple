function renderIteration(el, it) {
  const tabs = ['diff', 'pool', 'reflector', 'reasoning'];
  const tabBar = `<div class="tab-bar">${
    tabs.map(t => `<button class="tab${selTab===t?' sel':''}" onclick="selTab='${t}';renderMain()">${t.charAt(0).toUpperCase()+t.slice(1)}</button>`).join('')
  }</div>`;

  let body = '';

  if (selTab === 'diff') {
    const op = it.operation;
    if (op) {
      const parents = (it.parents || []).filter(p => p != null);
      let parentLabel = '';
      if (op === 'seed') {
        parentLabel = '';
      } else if (op === 'merge' && parents.length >= 2) {
        parentLabel = `<span class="parent-tag">Candidates #${parents[0]} × #${parents[1]}</span>`;
      } else if (parents.length >= 1) {
        parentLabel = `<span class="parent-tag">Candidate #${parents[0]}</span>`;
      }
      body += `<div class="op-row"><span class="op-badge op-${op}">${op}</span>${parentLabel}</div>`;
    } else if (it.parent_idx != null) {
      body += `<div class="op-row"><span class="parent-tag">Candidate #${it.parent_idx}</span></div>`;
    }
    body += renderScores(it);

    const pf = it.parent_files || {};
    const cf = it.child_files  || {};
    const allFiles = [...new Set([...Object.keys(pf), ...Object.keys(cf)])].sort();

    if (!allFiles.length) {
      body += `<div class="no-diff">No file data for this iteration.</div>`;
    } else {
      const added     = allFiles.filter(f => !(f in pf) &&  (f in cf));
      const removed   = allFiles.filter(f =>  (f in pf) && !(f in cf));
      const modified  = allFiles.filter(f =>  (f in pf) &&  (f in cf) && pf[f] !== cf[f]);
      const unchanged = allFiles.filter(f =>  (f in pf) &&  (f in cf) && pf[f] === cf[f]);

      const summaryMap = {};
      (it.changes_summary || []).forEach(l => {
        const ci = l.indexOf(':');
        if (ci > 0) summaryMap[l.slice(0, ci).trim()] = l.slice(ci + 1).trim();
      });
      body += `<div class="section-title">Files</div>`;
      body += renderFilesOverview(added, removed, modified, unchanged, summaryMap);

      const bullets = Array.isArray(it.changes) ? it.changes : (it.changes || '').split(/\n(?=- )/).map(s => s.replace(/^- /, '').trim()).filter(Boolean);
      if (bullets.length) {
        body += `<div class="section-title">Reflector changes</div>`;
        body += `<div class="changes-box"><ul>${bullets.map(b => `<li>${esc(b)}</li>`).join('')}</ul></div>`;
      }

      for (const f of [...modified, ...added].sort()) {
        const diff = lineDiff(pf[f] || '', cf[f] || '');
        const dels = diff.filter(d => d.t === 'd').length;
        const ins  = diff.filter(d => d.t === 'i').length;
        const isNew = !(f in pf);
        body += `<div class="section-title" style="margin-top:18px">${esc(f)}</div>`;
        body += `<div class="diff-meta">`;
        if (!isNew) body += `<span class="stat-del">−${dels} lines</span>`;
        body += `<span class="stat-ins">+${ins} lines</span>`;
        if (isNew) body += `<span style="color:#6e7681;font-size:11px;margin-left:4px">new file</span>`;
        body += `</div>`;
        body += renderUnifiedDiff(diff);
      }
      for (const f of removed) {
        body += `<div class="section-title" style="margin-top:18px">${esc(f)}</div>`;
        body += `<div class="no-diff" style="border-left:3px solid #f85149;color:#f85149">deleted</div>`;
      }
    }

  } else if (selTab === 'pool') {
    body += renderPool(it);

  } else if (selTab === 'reflector') {
    body += renderReflector(it);

  } else {
    body += renderReasoning(it);
  }

  el.innerHTML = tabBar + body;
}

// ── Files overview ────────────────────────────────────────────────────────
function renderFilesOverview(added, removed, modified, unchanged, summaryMap) {
  summaryMap = summaryMap || {};
  function lookup(f) {
    if (summaryMap[f]) return summaryMap[f];
    return summaryMap[f.split('/').pop()] || '';
  }
  function fileRows(cls, label, files) {
    return files.map(f => {
      const desc = lookup(f);
      return `<div class="file-row"><span class="file-badge fb-${cls}">${label}</span><span class="file-name">${esc(f)}</span>${desc ? `<span class="file-summary">${esc(desc)}</span>` : ''}</div>`;
    }).join('');
  }
  const rows = fileRows('modified', 'modified', modified)
             + fileRows('added',    'added',    added)
             + fileRows('removed',  'deleted',  removed)
             + fileRows('unchanged','unchanged',unchanged);
  return `<div class="files-overview">${rows}</div>`;
}
