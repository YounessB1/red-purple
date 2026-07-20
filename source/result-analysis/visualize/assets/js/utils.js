function shortModel(m) { return (m || '').replace('openrouter/', ''); }
function fmtDuration(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h > 0 ? `${h}h ${m}m ${sec}s` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtTokens(n) {
  if (!n) return '0';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(n);
}
function row(key, val, cls) {
  return `<div class="info-row"><span class="info-key">${key}</span><span class="info-val${cls ? ' ' + cls : ''}">${val}</span></div>`;
}
function pad(n) { return String(n).padStart(3, '0'); }
function esc(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
