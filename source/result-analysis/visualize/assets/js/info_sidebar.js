function renderInfoSidebar() {
  const cfg = DATA.config || {};
  const sum = DATA.summary || {};
  const el  = document.getElementById('info-sidebar');

  // ── Config panel — supports nested format (ctf_agent/scorer/…) and legacy flat format ──
  let cfgRows = '';
  // Resolve values from nested format first, fall back to legacy flat keys
  const agentModel    = cfg.ctf_agent?.md?.model      || cfg.agent_model;
  const reflModel     = cfg.reflector?.md?.model      || cfg.reflection_lm;
  const diagModel     = cfg.diagnoser?.md?.model      || cfg.diagnoser_model;
  const judgeModel    = cfg.scorer?.md?.model         || cfg.judge_model;
  const maxSteps      = cfg.ctf_agent?.md?.steps       || cfg.ctf_agent?.md?.maxSteps || cfg.agent_max_iter;
  const trainBatch    = cfg.reflector?.train_minibatch_size ?? cfg.train_minibatch_size;
  const valBatch      = cfg.reflector?.val_minibatch_size   ?? cfg.val_minibatch_size;
  const gt            = cfg.scorer?.gt                ?? cfg.gt;
  const evolution     = cfg.reflector?.evolution;
  const splits        = cfg.splits;

  if (cfg.experiment_name) cfgRows += row('experiment',    esc(cfg.experiment_name));
  if (splits)              cfgRows += row('splits',        esc(splits));
  if (agentModel)          cfgRows += row('agent model',   esc(shortModel(agentModel)), 'mono');
  if (reflModel)           cfgRows += row('reflector',     esc(shortModel(reflModel)), 'mono');
  if (diagModel)           cfgRows += row('diagnoser',     esc(shortModel(diagModel)), 'mono');
  if (judgeModel)          cfgRows += row('judge',         esc(shortModel(judgeModel)), 'mono');
  if (cfg.max_calls != null) cfgRows += row('budget',      `${cfg.max_calls} calls`);
  if (cfg.workers   != null) cfgRows += row('workers',     cfg.workers);
  if (maxSteps      != null) cfgRows += row('agent max steps', maxSteps);
  if (trainBatch    != null) cfgRows += row('train batch', trainBatch);
  cfgRows += row('val batch', valBatch != null ? valBatch : 'full');
  if (gt            != null) cfgRows += row('ground truth', gt ? 'yes' : 'no');
  if (evolution)             cfgRows += row('evolution',   esc(evolution));
  if (cfg.background_context) cfgRows += row('context file', esc(cfg.background_context), 'mono');

  // ── Summary panel ──
  let sumRows = '';
  if (sum.duration_seconds != null) sumRows += row('duration', fmtDuration(sum.duration_seconds));
  if (sum.total_cost_usd   != null) sumRows += row('total cost', `$${sum.total_cost_usd.toFixed(2)}`, 'info-cost');
  if (sum.total_tokens     != null) sumRows += row('total tokens', fmtTokens(sum.total_tokens));

  for (const key of ['agents', 'reflector', 'diagnoser', 'scorer']) {
    const s = sum[key];
    if (!s || s.calls === 0) continue;
    sumRows += `<div class="info-sub-hdr">${key} (${s.calls} calls)</div>`;
    if (s.model) sumRows += row('model', esc(shortModel(s.model)), 'mono');
    sumRows += row('cost', `$${s.cost_usd.toFixed(2)}`, 'info-cost');
    sumRows += row('tokens in/out', `${fmtTokens(s.input_tokens)} / ${fmtTokens(s.output_tokens)}`);
  }

  el.innerHTML =
    `<div class="info-panel"><div class="info-panel-hdr">Config</div>${cfgRows}</div>` +
    (sumRows ? `<div class="info-panel"><div class="info-panel-hdr">Summary</div>${sumRows}</div>` : '');
}
