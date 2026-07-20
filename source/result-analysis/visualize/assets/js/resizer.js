(function() {
  function makeResizer(resizerId, targetId, side) {
    const resizer = document.getElementById(resizerId);
    const target  = document.getElementById(targetId);
    let startX, startW;
    resizer.addEventListener('mousedown', e => {
      startX = e.clientX;
      startW = target.getBoundingClientRect().width;
      resizer.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      function onMove(e) {
        const dx  = e.clientX - startX;
        const newW = side === 'left' ? startW + dx : startW - dx;
        if (newW >= 80 && newW <= 700) target.style.width = newW + 'px';
      }
      function onUp() {
        resizer.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      e.preventDefault();
    });
  }
  makeResizer('resizer-left',  'sidebar',      'left');
  makeResizer('resizer-right', 'info-sidebar', 'right');
})();
