const DATA = __DATA__;
let selId  = null;
let selTab = 'diff';
let selEvoTab = 'tree';
let treeExpanded = new Set();
let treeViewState = null;
let simViewState = null;
let simLegacyViewState = null;
let simTop10ViewState = null;
let simTop10LegacyViewState = null;

document.getElementById('exp-name').textContent = DATA.name;

renderInfoSidebar();
renderSidebar();
renderMain();

// Both side panels (iterations on the left, config/summary on the right)
// toggle independently — #main is a flex:1 child of #body, so it naturally
// reclaims whatever width a hidden panel (and its resizer) frees up.
function setupPanelToggle(btnId, panelId, resizerId) {
  const btn = document.getElementById(btnId);
  const panel = document.getElementById(panelId);
  const resizer = document.getElementById(resizerId);
  btn.classList.add('active');
  btn.addEventListener('click', () => {
    const collapsed = panel.classList.toggle('collapsed');
    resizer.classList.toggle('collapsed', collapsed);
    btn.classList.toggle('active', !collapsed);
  });
}
setupPanelToggle('toggle-sidebar', 'sidebar', 'resizer-left');
setupPanelToggle('toggle-info-sidebar', 'info-sidebar', 'resizer-right');
