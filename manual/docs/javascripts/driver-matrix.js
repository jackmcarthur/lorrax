// Driver × page grid at the top of the left sidebar (direction 4a).
// Agents: add a driver to DRIVERS once its pages exist under docs/drivers/<id>/.
(function () {
  const DRIVERS = [
    { id: 'nscf', stage: 'plane-wave bands' },
    { id: 'gw', stage: 'self-energy · QP' },
    { id: 'bse', stage: 'absorption' }
    // exciton-bands: pages unlisted for now (mkdocs.yml not_in_nav); add it back here with them.
  ];
  const PAGES = [
    { id: 'overview', path: '', label: 'Overview' },
    { id: 'keywords', path: 'keywords/', label: 'Input keywords' },
    { id: 'theory', path: 'theory/', label: 'Theory' },
    { id: 'files', path: 'files/', label: 'Files' }
    // Dependencies pages exist but are unlisted (mkdocs.yml not_in_nav).
  ];
  const P = d => ['p', d], C = (x, y, r, f) => ['c', x, y, r, f];
  const ICONS = {
    overview: [C(8, 8, 5.8), P('M9.9 6.1L8.9 8.9L6.1 9.9L7.1 7.1Z')],
    keywords: [P('M2.5 5H13.5M2.5 11H13.5'), C(6, 5, 1.9, 1), C(10.5, 11, 1.9, 1)],
    theory: [P('M12 3.5H4.5L8.6 8L4.5 12.5H12')],
    files: [P('M4 2.5H9.5L12.5 5.5V13.5H4Z'), P('M9.5 2.5V5.5H12.5')],
    deps: [C(4, 4, 1.7), C(4, 12, 1.7), C(12, 8, 1.7), P('M5.6 4.8L10.4 7.2M5.6 11.2L10.4 8.8')],
    terminal: [P('M3 5L6 8L3 11'), P('M8.5 11.5H13')],
    info: [C(8, 8, 5.8), P('M8 7.3V11'), C(8, 5, 0.7, 1)],
    ban: [C(8, 8, 5.8), P('M3.9 12.1L12.1 3.9')],
    list: [P('M6 4H13M6 8H13M6 12H13'), C(3.2, 4, 0.8, 1), C(3.2, 8, 0.8, 1), C(3.2, 12, 0.8, 1)],
    edit: [P('M3 13L3.6 10.4L10.6 3.4L12.6 5.4L5.6 12.4Z')],
    prev: [P('M13 8H3M7 4L3 8L7 12')],
    next: [P('M3 8H13M9 4L13 8L9 12')]
  };
  function icon(name, size) {
    const parts = (ICONS[name] || []).map(p => p[0] === 'p' ? `<path d="${p[1]}"/>` : `<circle cx="${p[1]}" cy="${p[2]}" r="${p[3]}" ${p[4] ? 'fill="currentColor" stroke="none"' : ''}/>`).join('');
    return `<svg width="${size || 15}" height="${size || 15}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${parts}</svg>`;
  }
  window.LX_ICONS = { ICONS, icon };

  function base() { try { return new URL(window.__md_scope || '/', location.href).href; } catch (e) { return location.origin + '/'; } }
  function where() {
    const rel = location.href.slice(base().length).split('#')[0].split('?')[0];
    const m = rel.match(/^drivers\/([^/]+)\/(?:([^/.]+)(?:\/|\.html)?)?/);
    if (!m) return { driver: null, page: null };
    const seg = m[2] || '';
    const page = seg === '' || seg === 'index' ? 'overview' : seg === 'dependencies' ? 'deps' : seg;
    return { driver: m[1], page };
  }
  function run() {
    const host = document.querySelector('.md-sidebar--primary .md-sidebar__inner');
    if (!host) return;
    const old = host.querySelector('.lx-matrix'); if (old) old.remove();
    const { driver, page } = where(), b = base();
    let g = '<span></span>' + PAGES.map(p => `<span class="lx-matrix__icon${p.id === page ? ' on' : ''}" title="${p.label}">${icon(p.id, 15)}</span>`).join('');
    DRIVERS.forEach(d => {
      const me = d.id === driver;
      g += `<div><div class="lx-matrix__id${me ? ' on' : ''}">${d.id}</div><div class="lx-matrix__stage">${d.stage}</div></div>`;
      g += PAGES.map(p => `<a class="lx-matrix__cell${me ? ' row' : ''}${me && p.id === page ? ' on' : ''}" href="${b}drivers/${d.id}/${p.path}" title="${d.id} · ${p.label}"></a>`).join('');
    });
    const el = document.createElement('div');
    el.className = 'lx-matrix';
    el.innerHTML = `<div class="lx-matrix__title">Drivers</div><div class="lx-matrix__grid">${g}</div>`;
    // One column per page: keep the CSS grid in step with PAGES.
    el.querySelector('.lx-matrix__grid').style.gridTemplateColumns = `minmax(0,1fr) repeat(${PAGES.length}, 22px)`;
    host.insertBefore(el, host.firstChild);
  }
  if (window.document$) window.document$.subscribe(run); else document.addEventListener('DOMContentLoaded', run);
})();
