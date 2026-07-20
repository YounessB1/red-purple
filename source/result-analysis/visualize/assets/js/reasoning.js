// ── Reasoning steps renderer ─────────────────────────────────────────────
function renderReasoning(it) {
  if (!it.reflector_steps) return `<div class="no-diff">No agentic reflector steps for this iteration.</div>`;
  let steps;
  try { steps = JSON.parse(it.reflector_steps); } catch(_) {
    return `<div class="no-diff">Could not parse steps data.</div>`;
  }
  if (!Array.isArray(steps) || !steps.length) return `<div class="no-diff">No steps recorded.</div>`;

  // Strip final output message (the JSON blob the agent writes as its answer)
  if (steps[steps.length - 1].type === 'text' &&
      steps[steps.length - 1].text.trimStart().startsWith('{')) {
    steps = steps.slice(0, -1);
  }

  function shortPath(p) {
    const m = (p || '').match(/iteration_\d+\/(.*)/);
    return m ? m[1] : p;
  }
  function fmtThinking(text) {
    return esc(text).replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  }

  let html = '<div class="rsn-feed">';
  for (const step of steps) {
    if (step.type === 'thinking') {
      html += `<div class="rsn-step rsn-step-thinking">
        <div class="rsn-label">Thinking</div>
        <div class="rsn-body">${fmtThinking(step.text)}</div>
      </div>`;
    } else if (step.type === 'text') {
      html += `<div class="rsn-step rsn-step-text">
        <div class="rsn-label">Text</div>
        <div class="rsn-body">${esc(step.text)}</div>
      </div>`;
    } else if (step.type === 'tool') {
      const name = step.name || '?';
      const inp  = step.input || {};
      const params = Object.entries(inp)
        .filter(([k, v]) => v !== null && v !== undefined && v !== '' && v !== 0)
        .map(([k, v]) => {
          const display = (k === 'filePath' || k === 'path') ? shortPath(String(v)) : String(v);
          return `<div class="rsn-param-row"><span class="rsn-param-key">${esc(k)}</span><span class="rsn-param-val">${esc(display)}</span></div>`;
        }).join('');
      html += `<div class="rsn-step rsn-step-tool">
        <div class="rsn-label">Tool</div>
        <div class="rsn-body"><span class="rsn-tool-name">${esc(name)}</span>${params ? `<div class="rsn-params">${params}</div>` : ''}</div>
      </div>`;
    }
  }
  html += '</div>';
  return html;
}
