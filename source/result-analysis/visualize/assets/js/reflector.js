// ── Reflector renderer ────────────────────────────────────────────────────
function reflContent(text) {
  return text.split('\n').map(line => {
    if (/^## /.test(line))  return `<span class="rh">${esc(line)}</span>`;
    if (/^```/.test(line))  return `<span class="rf">${esc(line)}</span>`;
    return esc(line);
  }).join('\n');
}

function renderReflector(it) {
  if (!it.reflector_json) return `<div class="no-diff">No reflector data for this iteration.</div>`;
  const data = JSON.parse(it.reflector_json);
  let html = '';

  // Input — may be a string (agentic reflector) or an array of messages (LLM reflector)
  const inputMsgs = Array.isArray(data.input) ? data.input : (data.input ? [{role: 'user', content: data.input}] : []);
  for (const msg of inputMsgs) {
    html += `<div class="refl-block">
      <div class="refl-block-hdr">${esc(msg.role)}</div>
      <div class="refl-body">${reflContent(msg.content || '')}</div>
    </div>`;
  }

  // Output
  const raw = data.output || '';
  if (raw) {
    html += `<div class="section-title" style="margin-top:4px">Output</div>`;
    let changes = [], prompt = '';
    try {
      const out = JSON.parse(raw);
      changes = out.changes || [];
      prompt  = out.prompt  || '';
    } catch(_) {
      // Fallback: unescaped quotes inside the prompt string break JSON.parse.
      // Extract changes (clean array before "prompt") and prompt (greedy to last " before }) separately.
      const cm = raw.match(/"changes"\s*:\s*(\[[\s\S]*?\])\s*,\s*"prompt"/);
      if (cm) { try { changes = JSON.parse(cm[1]); } catch(__) {} }
      const pm = raw.match(/"prompt"\s*:\s*"([\s\S]*)"\s*\}\s*$/);
      if (pm) {
        prompt = pm[1].replace(/\\(.)/g, (_, c) =>
          ({n:'\n', t:'\t', r:'\r', '\\':'\\', '"':'"'}[c] || c));
      }
    }
    const items = Array.isArray(changes)
      ? changes
      : String(changes).split(/\n(?=- )/).map(s => s.replace(/^- /, '').trim()).filter(Boolean);
    if (items.length) {
      html += `<div class="changes-box"><ul>${items.map(b => `<li>${esc(b)}</li>`).join('')}</ul></div>`;
    }
    if (prompt) {
      html += `<div class="refl-block" style="margin-top:12px">
        <div class="refl-block-hdr">New prompt</div>
        <div class="refl-body">${esc(prompt)}</div>
      </div>`;
    }
    if (!items.length && !prompt) {
      html += `<div class="refl-block"><div class="refl-body">${esc(raw)}</div></div>`;
    }
  }

  return html;
}
