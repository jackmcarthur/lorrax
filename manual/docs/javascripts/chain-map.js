// Chain map for LORRAX driver and utility pages: the pipeline figure of the flowchart design (FigureKit.pipeline with
// SPECS.lorraxPipeline): Mean field -> GW -> BSE stepping down and to the right, utilities hanging below the stage that
// feeds them. The page's own stage, helper or option chip is highlighted (2026-10-20; replaces the 4a staircase).
// Usage in Markdown:  <div class="chain-map" data-driver="gw" data-page="files"></div>
// data-page: overview | keywords | theory | files | deps (files adds the reads/writes row).
// Optional data-treatment on the gw theory page selects the Σ chip.
(function () {
  // data-driver -> what to highlight: a stage (focus), a helper box (helper) or an option chip of the focused stage (chip)
  const FOCUS = {
    nscf: { focus: 'mf', chip: 'nscf-lorrax' }, pw2bgw: { helper: 'pw2bgw' },
    centroids: { helper: 'centroids' }, 'kin-ion': { helper: 'kin-ion' }, dipole: { helper: 'dipole' },
    gw: { focus: 'gw' }, htransform: { helper: 'htransform' }, downfold: { helper: 'downfold' },
    bse: { focus: 'bse', chip: 'bse' }, 'exciton-bands': { focus: 'bse', chip: 'exciton-bands' }
  };
  // files each driver reads and writes (the Files page row); in-process hand-offs are drawn dashed
  const EDGES = [
    ['scf', 'nscf', 'prefix.save'], ['scf', 'pw2bgw', 'prefix.save'],
    ['nscf', 'dipole', 'WFN.h5'], ['nscf', 'centroids', 'WFN.h5'], ['nscf', 'kin-ion', 'WFN.h5'], ['nscf', 'gw', 'WFN.h5'],
    ['pw2bgw', 'dipole', 'WFN.h5'], ['pw2bgw', 'centroids', 'WFN.h5'], ['pw2bgw', 'kin-ion', 'WFN.h5'], ['pw2bgw', 'gw', 'WFN.h5'],
    ['centroids', 'gw', 'centroids_frac_<N>.txt'], ['dipole', 'gw', 'dipole.h5'], ['kin-ion', 'gw', 'kin_ion.h5'],
    ['gw', 'downfold', 'restart bundle'], ['gw', 'bse', 'restart bundle'], ['gw', 'bse', 'eqp1.dat'], ['downfold', 'bse', 'compressed restart'],
    ['gw', 'htransform', 'qp_wfn_rotations.h5'], ['gw', 'exciton-bands', 'restart bundle'],
    // not a file hand-off: exciton_bands runs the H-transform in process and reads no htransform output
    ['htransform', 'exciton-bands', 'H-transform, in process (no file)', true]
  ].map(([from, to, file, inproc]) => ({ from, to, file, inproc: !!inproc }));
  const MONO = "'IBM Plex Mono',monospace";
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  function caption(d, mode) {
    if (mode === 'files') return 'Files ' + d + ' reads and writes.';
    if (mode === 'deps') return 'Where ' + d + ' sits in the chain: what it needs and what reads its output.';
    return 'Where ' + d + ' sits in the chain.';
  }
  function fileRow(d) {
    const u = a => a.filter((x, i) => a.indexOf(x) === i);
    const pick = (dir, ip) => u(EDGES.filter(e => e[dir] === d && e.inproc === ip).map(e => e.file));
    const chip = (x, c, b, bg, dashed) => `<span style="font-family:${MONO};border:1px ${dashed ? 'dashed' : 'solid'} ${b};color:${c};border-radius:2px;padding:1px 7px;background:${bg};white-space:nowrap">${esc(x)}</span>`;
    return `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center;font-size:12.5px">` +
      pick('to', false).map(x => chip(x, '#1f2a44', '#1f2a44', '#ffffff')).join('') +
      pick('to', true).map(x => chip(x, '#6b7894', '#a3abbb', '#ffffff', true)).join('') +
      `<span style="color:#6b7894;padding:0 4px">→ ${esc(d)} →</span>` +
      pick('from', false).map(x => chip(x, '#ffffff', '#131b2e', '#1f2a44')).join('') +
      pick('from', true).map(x => chip(x, '#6b7894', '#a3abbb', '#ffffff', true)).join('') + `</div>`;
  }

  function render(el) {
    if (!window.FigureKit || !window.SPECS || !window.SPECS.lorraxPipeline) return;
    const d = el.dataset.driver, mode = el.dataset.page || 'overview', f = FOCUS[d] || {};
    const chip = d === 'gw' ? (el.dataset.treatment || null) : (f.chip || null);
    el.innerHTML = `<div class="lx-map-frame" style="position:relative;overflow:hidden"><div class="lx-map" style="transform-origin:0 0;width:max-content"></div></div>` +
      `<div style="margin-top:8px;font-size:13px;color:#6b7894;line-height:1.4">${esc(caption(d, mode))}</div>` + (mode === 'files' ? fileRow(d) : '');
    window.FigureKit.pipeline(el.querySelector('.lx-map'), window.SPECS.lorraxPipeline, { focus: f.focus || null, focusHelper: f.helper || null, focusChip: chip });
    fit(el);
  }
  function fit(el) {
    const frame = el.querySelector('.lx-map-frame'), map = el.querySelector('.lx-map'), fig = map && map.firstElementChild;
    if (!frame || !fig) return;
    const W = fig.offsetWidth, H = fig.offsetHeight, s = Math.min(1, el.clientWidth / (W + 2));
    map.style.transform = s < 1 ? `scale(${s})` : 'none';
    frame.style.height = Math.ceil(H * s) + 'px';
  }

  // the gw theory page's treatment tabs move the Σ chip
  const TAB = { COHSEX: 'cohsex', 'GN-PPM': 'gn_ppm', 'HL-GPP': 'hl_ppm', TANGENTIAL: 'tangential' };
  function bindTabs() {
    document.querySelectorAll('.tabbed-set > input').forEach(inp => {
      if (inp.dataset.lxBound) return; inp.dataset.lxBound = '1';
      inp.addEventListener('change', () => {
        const lab = document.querySelector(`label[for="${inp.id}"]`), txt = lab ? lab.textContent.trim().toUpperCase() : '';
        const key = Object.keys(TAB).find(k => txt.startsWith(k));
        if (!key) return;
        document.querySelectorAll('.chain-map[data-page="theory"][data-driver="gw"]').forEach(el => { el.dataset.treatment = TAB[key]; render(el); });
      });
    });
  }
  function run() { document.querySelectorAll('.chain-map[data-driver]').forEach(render); bindTabs(); }
  window.addEventListener('resize', () => document.querySelectorAll('.chain-map').forEach(fit));
  if (window.document$) window.document$.subscribe(run); else document.addEventListener('DOMContentLoaded', run);
  window.LX_CHAIN = { FOCUS, EDGES, render };
})();
