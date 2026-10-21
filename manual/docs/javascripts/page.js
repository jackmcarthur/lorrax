// Page-level structure for the LORRAX manual: theory section cards and indentation,
// keyword entry wrappers, and figure-kit figures. Authors never call this directly.
(function () {
  function wrapUntil(start, stop, cls) {
    const box = document.createElement('div'); box.className = cls;
    let n = start.nextElementSibling;
    while (n && !stop(n)) { const next = n.nextElementSibling; box.appendChild(n); n = next; }
    start.after(box);
    return box;
  }
  function theory(root) {
    const h2s = [...root.querySelectorAll('h2')].filter(h => h.querySelector('.n'));
    const cards = root.querySelector('.section-cards');
    if (cards && h2s.length) {
      cards.innerHTML = h2s.map(h => {
        const n = h.querySelector('.n').textContent.trim(), title = h.textContent.replace(n, '').trim();
        const subs = [];
        let s = h.nextElementSibling;
        while (s && s.tagName !== 'H2') { if (s.tagName === 'H3' && s.querySelector('.n')) subs.push(s); s = s.nextElementSibling; }
        const list = subs.map(u => { const un = u.querySelector('.n').textContent.trim(); return `<div class="cs"><span>${un}</span>${u.textContent.replace(un, '').trim()}</div>`; }).join('');
        return `<a href="#${h.id}"><div><span class="cn">${n}</span><span class="ct">${title}</span></div><div>${list}</div></a>`;
      }).join('');
    }
    h2s.forEach(h => { if (!h.nextElementSibling || !h.nextElementSibling.classList.contains('sec-body')) wrapUntil(h, x => x.tagName === 'H2', 'sec-body'); });
  }
  function keywords(root) {
    root.querySelectorAll('h3.kw').forEach(h => {
      if (h.parentElement.classList.contains('kw-entry')) return;
      const box = document.createElement('div'); box.className = 'kw-entry';
      h.before(box); box.appendChild(h);
      let n = box.nextElementSibling;
      while (n && !/^H[1-3]$/.test(n.tagName)) { const next = n.nextElementSibling; box.appendChild(n); n = next; }
    });
  }
  function figures(root) {
    if (!window.FigureKit || !window.SPECS) return;
    root.querySelectorAll('.algo-figure[data-spec]').forEach(el => { const s = window.SPECS[el.dataset.spec]; if (s) window.FigureKit.algorithm(el, s); });
    root.querySelectorAll('.pipeline-figure[data-spec]').forEach(el => { const s = window.SPECS[el.dataset.spec]; if (s) window.FigureKit.pipeline(el, s, { focus: el.dataset.focus || null }); });
  }
  function run() {
    const root = document.querySelector('.md-content article') || document.body;
    theory(root); keywords(root);
    (document.fonts ? document.fonts.ready : Promise.resolve()).then(() => figures(root));
  }
  if (window.document$) window.document$.subscribe(run); else document.addEventListener('DOMContentLoaded', run);
})();
