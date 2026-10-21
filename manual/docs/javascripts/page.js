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
      // Consecutive "Reference: ..." sections share one card that lists each of them.
      const isRef = h => /^Reference\b/.test(h.textContent.replace(h.querySelector('.n').textContent, '').trim());
      const groups = [];
      h2s.forEach(h => { const last = groups[groups.length - 1]; if (isRef(h) && last && isRef(last[0])) last.push(h); else groups.push([h]); });
      cards.innerHTML = groups.map(g => {
        if (g.length > 1) {
          const ns = g.map(h => h.querySelector('.n').textContent.trim());
          const list = g.map((h, i) => `<div class="cs"><span>${ns[i]}</span>${h.textContent.replace(ns[i], '').trim().replace(/^Reference:\s*/, '').replace(/^./, c => c.toUpperCase())}</div>`).join('');
          return `<a href="#${g[0].id}"><div><span class="cn">${ns[0]}–${ns[ns.length - 1]}</span><span class="ct">Reference</span></div><div>${list}</div></a>`;
        }
        const h = g[0];
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
