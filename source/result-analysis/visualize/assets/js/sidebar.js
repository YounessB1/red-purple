function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  if (DATA.test_summary && Object.keys(DATA.test_summary).length) {
    const testCard = document.createElement('div');
    testCard.className = 'card' + (selId === 'test' ? ' sel' : '');
    testCard.innerHTML = `<div class="card-top"><span class="card-num">Test</span></div>`;
    testCard.addEventListener('click', () => { selId = 'test'; renderSidebar(); renderMain(); });
    sb.appendChild(testCard);
  }
  const evoCard = document.createElement('div');
  evoCard.className = 'card' + (selId === 'evolution' ? ' sel' : '');
  evoCard.innerHTML = `<div class="card-top"><span class="card-num">Evolution</span></div>`;
  evoCard.addEventListener('click', () => { selId = 'evolution'; renderSidebar(); renderMain(); });
  sb.appendChild(evoCard);
  DATA.iterations.forEach(it => {
    const card = document.createElement('div');
    card.className = 'card' + (selId === it.id ? ' sel' : '');
    card.innerHTML =
      `<div class="card-top">
         <span class="card-num">Iter ${pad(it.id)}</span>
         <span class="badge b-${it.status}">${it.status}</span>
       </div>
       ${it.val_total ? `<div class="card-score">${it.val_ok}/${it.val_total} solved</div>` : ''}`;
    card.addEventListener('click', () => { selId = it.id; renderSidebar(); renderMain(); });
    sb.appendChild(card);
  });
}

function renderMain() {
  const ph = document.getElementById('placeholder');
  const ct = document.getElementById('content');
  if (selId === null) { ph.style.display = 'flex'; ct.style.display = 'none'; return; }
  ph.style.display = 'none'; ct.style.display = 'block';
  if (selId === 'test') { renderTestResults(ct); return; }
  if (selId === 'evolution') { renderEvolutionView(ct); return; }
  renderIteration(ct, DATA.iterations.find(x => x.id === selId));
}
