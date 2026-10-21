/* LORRAX figure kit: renders algorithm figures and pipeline figures from plain JS specs.
   Needs KaTeX (katex.min.js + katex.min.css) and the IBM Plex Sans / Mono fonts on the page.
   Usage:
     FigureKit.algorithm(document.getElementById('fig'), spec)
     FigureKit.pipeline(document.getElementById('fig'), spec, { focus: 'gw' })            */
(function () {
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const css = o => Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== false)
    .map(([k, v]) => k.replace(/[A-Z]/g, m => '-' + m.toLowerCase()) + ':' + v).join(';');
  const el = (tag, style, inner) => `<${tag} style="${esc(css(style || {}))}">${inner == null ? '' : inner}</${tag}>`;
  const div = (s, i) => el('div', s, i), span = (s, i) => el('span', s, i);
  const list = x => Array.isArray(x) ? x : [x];

  function tex(t) {
    if (!window.katex) return '<i>' + esc(t) + '</i>';
    try { return window.katex.renderToString(t, { throwOnError: false }); } catch (e) { return esc(t); }
  }
  // Mixed text: $math$ and `code` inside plain words.
  function rich(s) {
    if (s == null || s === '') return '';
    return String(s).split(/(\$[^$]+\$|`[^`]+`)/g).map(p => {
      if (!p) return '';
      if (p[0] === '$' && p.length > 2 && p.endsWith('$')) return tex(p.slice(1, -1));
      if (p[0] === '`' && p.length > 1) return `<code style="font-family:'IBM Plex Mono',monospace;font-size:0.92em">${esc(p.slice(1, -1))}</code>`;
      return esc(p);
    }).join('');
  }
  const macroFn = macros => s => Object.keys(macros || {}).sort((a, b) => b.length - a.length).reduce((x, k) => x.split(k).join(macros[k]), s);

  const SANS = "'IBM Plex Sans',sans-serif", MONO = "'IBM Plex Mono',monospace";

  /* ---------------- Algorithm figure ---------------- */
  const ALGO_VARS = {
    '--unit': '10px', '--navy': '#1f2a44', '--navy-deep': '#131b2e', '--navy-mid': '#2a3653', '--grey-text': '#6b7894', '--soft-text': '#3d4760',
    '--rule': '#e3e6ec', '--panel': '#f6f7f9', '--panel-edge': '#d9dee6', '--strip': '#e8ebf0', '--strip-edge': '#cdd3dd', '--arrow-red': '#d8322a', '--dash': '#7d879c',
    '--stroke': '1.2px',
    '--text-figure': 'calc(var(--unit) * 2.7)', '--text-stage': 'calc(var(--unit) * 2.4)', '--text-label': 'calc(var(--unit) * 1.65)',
    '--text-math': 'calc(var(--unit) * 2.3)', '--text-equation': 'calc(var(--unit) * 2)', '--text-arrow': 'calc(var(--unit) * 1.85)',
    '--text-file': 'calc(var(--unit) * 1.55)', '--text-loop': 'calc(var(--unit) * 1.95)', '--text-definition': 'calc(var(--unit) * 2)',
    '--box-min-height': 'calc(var(--unit) * 4.8)', '--box-pad-x': 'calc(var(--unit) * 1.3)', '--box-pad-y': 'calc(var(--unit) * 0.75)', '--box-line-gap': 'calc(var(--unit) * 0.7)',
    '--file-strip': 'calc(var(--unit) * 2.55)', '--row-gap': 'calc(var(--unit) * 2.2)', '--stage-gap': 'calc(var(--unit) * 2.6)',
    '--badge': 'calc(var(--unit) * 4.5)', '--badge-gap': 'calc(var(--unit) * 1.6)', '--panel-pad-x': 'calc(var(--unit) * 1.6)', '--panel-pad-y': 'calc(var(--unit) * 1.5)',
    '--loop-strip': 'calc(var(--unit) * 3.5)', '--arrow-pad': 'calc(var(--unit) * 1.1)', '--arrow-min': 'calc(var(--unit) * 5.5)', '--join-width': 'calc(var(--unit) * 1.3)',
    '--extra-drop': 'calc(var(--text-equation) * 1.7 + var(--unit) * 0.6)', '--column-gap': 'calc(var(--unit) * 2.8)'
  };
  const LOOK = {
    output: { fill: 'var(--navy)', ink: '#ffffff', edge: 'var(--navy-deep)', stripFill: 'var(--navy-mid)', stripInk: '#c3cbdb', stripRule: 'rgba(255,255,255,0.14)' },
    input: { fill: '#ffffff', ink: 'var(--navy)', edge: 'var(--navy)', stripFill: 'var(--strip)', stripInk: 'var(--soft-text)', stripRule: 'var(--strip-edge)' },
    plain: { fill: '#ffffff', ink: 'var(--navy)', edge: 'var(--navy)' }
  };
  const label = t => div({ fontSize: 'var(--text-label)', fontWeight: 600, color: 'var(--grey-text)' }, t);

  function algorithm(target, spec, opts) {
    opts = opts || {};
    const M = m => tex(macroFn(spec.macros)(m)), T = t => rich(macroFn(spec.macros)(t));
    const objects = spec.objects || {};
    const objOf = x => typeof x === 'string' && x[0] === '@' ? objects[x.slice(1)] : null;
    const isArrow = x => x && typeof x === 'object' && !Array.isArray(x);

    const fileStrip = (lk, text, abs) => div({ position: abs ? 'absolute' : undefined, left: abs ? '0' : undefined, right: abs ? '0' : undefined, top: abs ? '100%' : undefined,
      marginTop: '-1px', height: 'var(--file-strip)', padding: '0 var(--box-pad-x)', boxSizing: 'border-box', display: 'flex', alignItems: 'center',
      background: lk.stripFill, color: lk.stripInk, border: '1px solid ' + lk.edge, borderTop: '1px solid ' + lk.stripRule,
      fontFamily: MONO, fontSize: 'var(--text-file)', whiteSpace: 'nowrap' }, esc(text));

    function box(x) {
      const o = objOf(x), lk = LOOK[o ? o.kind : 'plain'] || LOOK.plain;
      const lines = list(o ? o.math : x).map(m => span({}, M(m))).join('');
      const face = div({ minHeight: 'var(--box-min-height)', boxSizing: 'border-box', padding: 'var(--box-pad-y) var(--box-pad-x)', display: 'flex', flexDirection: 'column',
        justifyContent: 'center', alignItems: 'center', gap: 'var(--box-line-gap)', background: lk.fill, color: lk.ink, border: '1px solid ' + lk.edge,
        fontSize: 'var(--text-math)', lineHeight: 1.2, whiteSpace: 'nowrap' }, lines);
      const file = o && o.file ? div({ height: 0, overflow: 'hidden', visibility: 'hidden', padding: '0 var(--box-pad-x)', fontFamily: MONO, fontSize: 'var(--text-file)', whiteSpace: 'nowrap' }, esc(o.file)) + fileStrip(lk, o.file, true) : '';
      return div({ display: 'flex', flexDirection: 'column' }, face + file);
    }
    function arrow(x) {
      const ex = x.extraInput, eo = objOf(ex);
      const extra = ex ? (
        div({ height: 0, overflow: 'hidden', visibility: 'hidden', fontSize: 'var(--text-equation)', whiteSpace: 'nowrap' }, M(eo ? list(eo.math).join(' ') : ex)) +
        div({ position: 'absolute', left: '50%', top: 0, width: 'calc(var(--unit) * 0.7)', height: 'calc(var(--unit) * 0.7)', transform: 'translate(-50%,-50%)', borderRadius: '50%', background: 'var(--navy)' }) +
        div({ position: 'absolute', left: '50%', top: 0, height: 'var(--extra-drop)', borderLeft: '1px dashed var(--dash)' }) +
        div({ position: 'absolute', left: '50%', top: 'var(--extra-drop)', transform: 'translateX(-50%)', paddingTop: 'calc(var(--unit) * 0.3)', fontSize: 'var(--text-equation)', color: 'var(--soft-text)', whiteSpace: 'nowrap', lineHeight: 1.25 }, M(eo ? list(eo.math).join(' ') : ex))
      ) : '';
      return div({ display: 'grid', gridTemplateColumns: 'auto 1fr', gridTemplateRows: '1fr 0 1fr', minWidth: 'var(--arrow-min)' },
        div({ gridRow: 1, gridColumn: '1 / 3', alignSelf: 'end', padding: '0 var(--arrow-pad) calc(var(--unit) * 0.45)', textAlign: 'center', fontFamily: MONO, fontSize: 'var(--text-arrow)', fontWeight: 600, color: 'var(--arrow-red)', lineHeight: 1, whiteSpace: 'nowrap' }, esc(x.arrow)) +
        div({ gridRow: 2, gridColumn: '1 / 3', height: 0, display: 'flex', alignItems: 'center' },
          div({ flex: 1, height: 'var(--stroke)', background: 'var(--navy)' }) +
          div({ flex: 'none', width: 0, height: 0, borderLeft: 'calc(var(--unit) * 0.8) solid var(--navy)', borderTop: 'calc(var(--unit) * 0.45) solid transparent', borderBottom: 'calc(var(--unit) * 0.45) solid transparent' })) +
        div({ gridRow: 3, gridColumn: 1, position: 'relative', padding: '0 ' + (ex ? 'var(--arrow-pad)' : '0px') }, extra) +
        div({ gridRow: 3, gridColumn: 2, padding: 'calc(var(--unit) * 0.55) var(--arrow-pad) 0', textAlign: 'center', fontSize: 'var(--text-equation)', whiteSpace: 'nowrap', lineHeight: 1.25 }, x.equation ? M(x.equation) : ''));
    }
    const cell = (x, row, col) => div({ gridRow: row, gridColumn: col, alignSelf: 'center', position: 'relative' }, isArrow(x) ? arrow(x) : box(x));
    const joinCell = (row, col, first, last) => div({ gridRow: row, gridColumn: col, alignSelf: 'stretch', position: 'relative' },
      div({ position: 'absolute', left: 0, right: 0, top: '50%', height: 'var(--stroke)', marginTop: 'calc(var(--stroke) / -2)', background: 'var(--navy)' }) +
      div({ position: 'absolute', right: 0, width: 'var(--stroke)', top: first ? '50%' : '0', bottom: last ? '50%' : 'calc(var(--row-gap) * -1)', background: 'var(--navy)' }) +
      div({ minWidth: 'var(--join-width)', height: '100%' }));

    function stage(p, isRef, linksToNext) {
      const rows = p.rows || [], n = rows.length, width = Math.max(...rows.map(r => r.length)), join = p.thenJoin || [];
      let cells = '';
      rows.forEach((r, i) => {
        r.forEach((x, j) => { cells += cell(x, String(i + 1), String(j + 1)); });
        if (n > 1) cells += joinCell(String(i + 1), `${r.length + 1} / ${width + 2}`, i === 0, i === n - 1);
      });
      join.forEach((x, k) => { cells += cell(x, `1 / span ${n}`, String(width + (n > 1 ? 2 : 1) + k)); });
      const last = (rows[n - 1] || []).concat(join);
      const padBottom = last.some(x => isArrow(x) && x.extraInput) ? 'calc(var(--panel-pad-y) + var(--extra-drop) + var(--text-equation) * 1.4 - var(--box-min-height) / 2)'
        : last.some(x => { const o = objOf(x); return o && o.file; }) ? 'calc(var(--panel-pad-y) + var(--file-strip))' : 'var(--panel-pad-y)';
      const edge = isRef ? '1px dashed #9aa3b5' : '1px solid var(--panel-edge)';
      const loop = p.loop ? div({ height: 'var(--loop-strip)', margin: 'calc(var(--panel-pad-y) * -1) calc(var(--panel-pad-x) * -1) var(--panel-pad-y)', padding: '0 var(--box-pad-x)',
        display: 'flex', alignItems: 'center', gap: 'calc(var(--unit) * 0.5)', background: isRef ? '#ffffff' : 'var(--strip)', borderBottom: isRef ? '1px dashed #9aa3b5' : '1px solid var(--strip-edge)',
        fontSize: 'var(--text-loop)', whiteSpace: 'nowrap' }, T(p.loop)) : '';
      const badge = div({ position: 'relative', flex: 'none', width: 'var(--badge)' },
        (linksToNext ? div({ position: 'absolute', left: 'calc(var(--badge) / 2 - 0.5px)', top: 'calc(var(--badge) + var(--unit) * 0.6)', bottom: 'calc(var(--stage-gap) * -1 + var(--unit) * 0.6)', width: '1px', background: 'var(--navy)' }) : '') +
        div({ width: 'var(--badge)', height: 'var(--badge)', boxSizing: 'border-box', border: '1.5px solid var(--navy)', background: '#ffffff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 'calc(var(--unit) * 2.05)', fontWeight: 600 }, esc(p.number)));
      const body = div({ flex: 1, minWidth: 0 },
        div({ height: 'var(--badge)', display: 'flex', alignItems: 'center', fontSize: 'var(--text-stage)', fontWeight: 600, whiteSpace: 'nowrap' }, T(p.title)) +
        div({ marginTop: 'calc(var(--unit) * 0.8)', background: isRef ? '#ffffff' : 'var(--panel)', border: edge, padding: `var(--panel-pad-y) var(--panel-pad-x) ${padBottom}` },
          loop + div({ display: 'grid', gridAutoRows: '1fr', rowGap: 'var(--row-gap)', justifyContent: 'start', alignItems: 'center' }, cells)));
      return div({ display: 'flex', gap: 'var(--badge-gap)' }, badge + body);
    }

    const main = spec.stages || [], refs = spec.referenceStages || [];
    let stagesHtml = div({ marginLeft: 'calc(var(--badge) + var(--badge-gap))', marginBottom: 'calc(var(--stage-gap) * -0.5)', paddingBottom: 'calc(var(--unit) * 0.5)', borderBottom: '1px solid var(--rule)' }, label('Algorithm'));
    main.forEach((p, i) => { stagesHtml += stage(p, false, i < main.length - 1); });
    if (refs.length) {
      stagesHtml += div({ display: 'flex', alignItems: 'center', gap: 'var(--unit)', marginBottom: 'calc(var(--stage-gap) * -0.5)' }, label('Reference') + span({ flex: 1, height: '1px', background: 'var(--rule)' }));
      refs.forEach(p => { stagesHtml += stage(p, true, false); });
    }

    const usedIn = p => (p.rows || []).flat().concat(p.thenJoin || []).flatMap(x => isArrow(x) ? [x.extraInput] : [x]).filter(x => objOf(x)).map(x => x.slice(1));
    const sourceOf = k => { const s = main.find(p => usedIn(p).includes(k)); return s ? s.number : ''; };
    const card = (title, kind, headFill, headInk, headEdge) => {
      const keys = Object.keys(objects).filter(k => objects[k].kind === kind);
      if (!keys.length) return '';
      const lk = LOOK[kind];
      const items = keys.map((k, i) => {
        const o = objects[k], src = kind === 'output' ? sourceOf(k) : '';
        return div({ padding: 'var(--box-pad-x)', borderTop: i ? '1px solid var(--rule)' : 'none', display: 'flex', flexDirection: 'column', gap: 'var(--unit)' },
          div({ display: 'flex', alignItems: 'flex-start', gap: 'var(--unit)' },
            div({ flex: 'none', display: 'flex', flexDirection: 'column' },
              div({ minHeight: 'var(--box-min-height)', boxSizing: 'border-box', padding: 'var(--box-pad-y) var(--box-pad-x)', display: 'flex', alignItems: 'center', background: lk.fill, color: lk.ink, border: '1px solid ' + lk.edge, fontSize: 'var(--text-math)', lineHeight: 1.2, whiteSpace: 'nowrap' }, M(list(o.math).join(' '))) +
              (o.file ? fileStrip(lk, o.file, false) : '')) +
            span({ flex: 1 }) +
            (src ? span({ height: 'var(--box-min-height)', display: 'flex', alignItems: 'center', gap: 'calc(var(--unit) * 0.6)', fontSize: 'var(--text-label)', color: 'var(--grey-text)' },
              'from' + span({ width: 'calc(var(--unit) * 2.4)', height: 'calc(var(--unit) * 2.4)', boxSizing: 'border-box', border: '1.5px solid var(--navy)', color: 'var(--navy)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 'var(--text-file)', fontWeight: 600 }, esc(src))) : '')) +
          div({ display: 'flex', flexDirection: 'column', gap: 'calc(var(--unit) * 0.5)', fontSize: 'var(--text-definition)', lineHeight: 1.35, whiteSpace: 'nowrap' }, list(o.definition || []).map(d => div({}, M(d))).join('')));
      }).join('');
      return div({ border: '1.5px solid var(--navy)', background: '#ffffff' },
        div({ height: 'var(--loop-strip)', padding: '0 var(--box-pad-x)', display: 'flex', alignItems: 'center', background: headFill, color: headInk, borderBottom: headEdge }, span({ fontSize: 'calc(var(--unit) * 2)', fontWeight: 600 }, title)) + items);
    };
    const cards = card('Inputs', 'input', 'var(--strip)', 'var(--navy)', '1px solid var(--strip-edge)') + card('Outputs', 'output', 'var(--navy)', '#ffffff', 'none');
    const objectsCol = cards ? div({ display: 'flex', flexDirection: 'column', gap: 'var(--badge-gap)', width: 'max-content' },
      div({ marginBottom: 'calc(var(--badge-gap) * -0.5)', paddingBottom: 'calc(var(--unit) * 0.5)', borderBottom: '1px solid var(--rule)' }, label('Objects')) + cards) : '';

    const vars = Object.assign({}, ALGO_VARS, opts.vars || {});
    target.innerHTML = div(Object.assign({}, vars, { width: 'max-content', boxSizing: 'border-box', background: '#ffffff', border: opts.frame === false ? 'none' : '1px solid #dfe3ea',
      padding: 'calc(var(--unit) * 2) calc(var(--unit) * 2.2) calc(var(--unit) * 2.4)', color: 'var(--navy)', fontFamily: SANS }),
      div({ display: 'flex', alignItems: 'baseline', gap: 'var(--unit)', paddingBottom: 'calc(var(--unit) * 0.9)', borderBottom: '1.5px solid var(--navy)' },
        span({ fontSize: 'var(--text-figure)', fontWeight: 600, whiteSpace: 'nowrap' }, T(spec.title || '')) + span({ flex: 1 }) +
        span({ fontFamily: MONO, fontSize: 'var(--text-label)', color: 'var(--grey-text)', whiteSpace: 'nowrap' }, esc(spec.label || ''))) +
      div({ display: 'flex', alignItems: 'flex-start', gap: 'var(--column-gap)', marginTop: 'var(--unit)' },
        div({ display: 'flex', flexDirection: 'column', gap: 'var(--stage-gap)', width: 'max-content' }, stagesHtml) + objectsCol));
  }

  /* ---------------- Pipeline figure ---------------- */
  const PIPE = { navy: '#1f2a44', navyDeep: '#131b2e', navyMid: '#2a3653', idle: '#a3abbb', idleSoft: '#7d879c', idleEdge: '#b3bac8', mute: '#6b7894', soft: '#4b5670', bar: '#e8ebf0', barRule: '#cdd3dd' };
  // headerHeight: the badge-and-title row ABOVE the stage box (flowchart style, 2026-10-20); the box is the rest of stageHeight
  const PIPE_LAYOUT = { stageWidth: 252, stageHeight: 106, stageStepX: 344, stageStepY: 64, marginX: 32, marginY: 32, helperStep: 44, helperHeight: 36, headerHeight: 28, footerHeight: 24 };

  function pipeline(target, spec, opts) {
    opts = opts || {};
    // focus: a stage id; focusHelper: a helper name (its box filled, its lines lit); focusChip: an option chip label
    // (or chip id) in the focused stage, shown filled (2026-10-20, chain map on the manual's pages)
    const focus = opts.focus ?? spec.focus ?? null, header = opts.header || spec.header || 'navy';
    const fHelper = opts.focusHelper ?? null, fChip = opts.focusChip ?? null;
    const chipId = c => typeof c === 'string' ? c : (c.id || c.label), chipLabel = c => typeof c === 'string' ? c : c.label;
    const Lo = Object.assign({}, PIPE_LAYOUT, spec.layout || {}, opts.layout || {});
    const SW = Lo.stageWidth, SH = Lo.stageHeight, DX = Lo.stageStepX, STEP = Lo.stageStepY, ML = Lo.marginX, MT = Lo.marginY, HS = Lo.helperStep, HH = Lo.helperHeight;
    const T = t => rich(macroFn(spec.macros)(t));
    const px = v => v + 'px';
    const dn = (x, y) => `M${x - 5} ${y - 8} L${x} ${y} L${x + 5} ${y - 8} Z`, rt = (x, y) => `M${x - 7} ${y - 4.5} L${x} ${y} L${x - 7} ${y + 4.5} Z`;
    const st = spec.stages.map((s, i) => Object.assign({}, s, { x: ML + i * DX, y: MT + i * STEP }));
    const segs = [], dots = [];
    let html = '', bottom = 0;
    st.forEach((s, i) => {
      const n = st[i + 1];
      if (s.helpers && s.helpers.length) {
        const top = s.y + SH + 26, sx = s.x + 14, hx = s.x + 28, hw = SW - 44, bus = s.x + SW + 16;
        const cs = s.helpers.map((_, k) => top + k * HS + HH / 2), last = cs.length - 1, ho = focus === s.id, hi = !!n && focus === n.id;
        const kh = s.helpers.findIndex(h => h.name === fHelper);   // the focused helper of this stage, if any
        if (s.helperLabel) html += span({ position: 'absolute', left: px(hx), top: px(top - 15), fontSize: '10.5px', fontWeight: 600, color: PIPE.mute, whiteSpace: 'nowrap', lineHeight: 1 }, esc(s.helperLabel));
        s.helpers.forEach((h, k) => {
          const hf = k === kh;
          html += div({ position: 'absolute', left: px(hx), top: px(cs[k] - HH / 2), width: px(hw), height: px(HH), boxSizing: 'border-box', background: hf ? PIPE.navy : '#ffffff', border: '1px solid ' + (hf ? PIPE.navyDeep : PIPE.navy), display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '0 9px' },
            span({ fontFamily: MONO, fontSize: '11px', fontWeight: 600, color: hf ? '#ffffff' : PIPE.navy, lineHeight: 1.2 }, esc(h.name)) +
            span({ fontSize: '10.5px', color: hf ? '#c3cbdb' : PIPE.soft, lineHeight: 1.25, whiteSpace: 'nowrap' }, T(h.note || '')));
        });
        // a focused helper lights the spine down to it and the bus from it into the next stage
        const spineHot = k => ho || (kh >= 0 && k <= kh);
        cs.forEach((c, k) => segs.push({ main: false, hot: spineHot(k), d: `M${sx} ${k ? cs[k - 1] : s.y + SH} V${c}` }));
        cs.forEach((c, k) => { segs.push({ main: false, hot: ho || k === kh, d: `M${sx} ${c} H${hx}` }); if (k < last) dots.push({ x: sx, y: c, hot: spineHot(k) }); });
        if (n) {
          cs.forEach((c, k) => segs.push({ main: false, hot: hi || k === kh, d: `M${hx + hw} ${c} H${bus}` }));
          if (last > 0) { cs.forEach((c, k) => { if (k) segs.push({ main: false, hot: hi || (kh >= k), d: `M${bus} ${cs[k - 1]} V${c}` }); if (k < last) dots.push({ x: bus, y: c, hot: hi || (kh >= k) }); }); }
          segs.push({ main: false, hot: hi || kh >= 0, d: `M${bus} ${cs[0]} H${n.x}`, head: rt(n.x, cs[0]) });
        }
        bottom = Math.max(bottom, cs[last] + HH / 2);
      }
      if (n) {
        // the hand-off leaves the middle of the box's math row and drops onto the next box (below its title row)
        const HD = Lo.headerHeight, my = s.y + HD + (SH - HD - Lo.footerHeight) / 2, mx = n.x + SW / 2, hot = focus === s.id || focus === n.id;
        segs.push({ main: true, hot, d: `M${s.x + SW} ${my} H${mx} V${n.y + HD}`, head: dn(mx, n.y + HD) });
        // the handed-over file, drawn as a file strip of the algorithm figures
        if (s.handoff) html += span({ position: 'absolute', left: px((s.x + SW + mx) / 2), top: px(my), transform: 'translate(-50%,-50%)', fontFamily: MONO, fontSize: '10.5px', lineHeight: 1.45, padding: '1px 8px',
          background: PIPE.bar, color: hot ? PIPE.navy : PIPE.idleSoft, border: '1px solid ' + (hot ? PIPE.navy : PIPE.barRule), whiteSpace: 'nowrap' }, T(s.handoff));
      }
      bottom = Math.max(bottom, s.y + SH);
    });
    // A stage in the algorithm figures' vocabulary: a square number badge and the title on a row above the box; the box
    // white with a navy rule (an input box), navy when it is the page's own stage (an output box); the options in a
    // mono strip under the math, like a file strip.
    const stagesHtml = st.map(s => {
      const f = s.id === focus, chips = s.options || [], HD = Lo.headerHeight;
      const badge = span({ width: '20px', height: '20px', flex: 'none', boxSizing: 'border-box', border: '1.5px solid ' + PIPE.navy, background: f ? PIPE.navy : '#ffffff', color: f ? '#ffffff' : PIPE.navy,
        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11.5px', fontWeight: 600 }, esc(String(s.number || '').replace(/^0+(?=\d)/, '')));
      const head = div({ position: 'absolute', left: px(s.x), top: px(s.y), height: px(HD - 6), display: 'flex', alignItems: 'center', gap: '8px', whiteSpace: 'nowrap' },
        badge + span({ fontSize: '14px', fontWeight: 600, color: PIPE.navy }, esc(s.title)));
      return head + div({ position: 'absolute', left: px(s.x), top: px(s.y + HD), width: px(SW), height: px(SH - HD), boxSizing: 'border-box', background: f ? PIPE.navy : '#ffffff', border: '1px solid ' + (f ? PIPE.navyDeep : PIPE.navy), display: 'flex', flexDirection: 'column' },
        div({ flex: 1, display: 'flex', alignItems: 'center', padding: '0 11px', fontSize: '14px', whiteSpace: 'nowrap', color: f ? '#ffffff' : PIPE.navy }, T(s.summary || '')) +
        div({ height: px(Lo.footerHeight), flex: 'none', borderTop: '1px solid ' + (f ? 'rgba(255,255,255,0.14)' : PIPE.barRule), background: f ? PIPE.navyMid : PIPE.bar, display: 'flex', alignItems: 'center', gap: '8px', padding: '0 9px' },
          span({ fontSize: '10px', whiteSpace: 'nowrap', color: f ? '#c3cbdb' : '#56617a' }, esc(s.optionsLabel || '')) +
          div({ display: 'flex', alignItems: 'center' }, chips.map((c, i) => { const on = f && fChip != null && chipId(c) === fChip;
            return span({ fontFamily: MONO, fontSize: '10px', whiteSpace: 'nowrap', color: on ? PIPE.navyDeep : (f ? '#e6eaf2' : PIPE.navy), background: on ? '#e0b54a' : 'transparent',
              padding: on ? '1px 5px' : '0', borderRadius: on ? '2px' : '0', paddingRight: on ? '5px' : '6px', marginRight: '6px',
              borderRight: !on && i < chips.length - 1 ? (f ? '1px solid rgba(255,255,255,0.25)' : '1px solid #b3bbc8') : 'none' }, esc(chipLabel(c))); }).join(''))));
    }).join('');
    const W = 2 * ML + (st.length - 1) * DX + SW, H = bottom + 32;
    const col = s => s.hot ? PIPE.navy : PIPE.idle;
    const svg = `<svg width="${W}" height="${H}" style="position:absolute;left:0;top:0;pointer-events:none">` +
      segs.slice().sort((a, b) => a.hot - b.hot).map(s => `<path d="${s.d}" fill="none" stroke="${col(s)}" stroke-width="${s.main ? (s.hot ? 2.2 : 1.6) : (s.hot ? 1.2 : 1)}"${s.main ? '' : ' stroke-dasharray="4 3"'} stroke-linejoin="miter"/>` + (s.head ? `<path d="${s.head}" fill="${col(s)}"/>` : '')).join('') +
      dots.slice().sort((a, b) => a.hot - b.hot).map(d => `<circle cx="${d.x}" cy="${d.y}" r="2.6" fill="${d.hot ? PIPE.navy : PIPE.idle}"/>`).join('') + '</svg>';
    target.innerHTML = div({ position: 'relative', width: px(W), height: px(H), background: '#ffffff', border: opts.frame === false ? 'none' : '1px solid #dfe3ea', fontFamily: SANS, color: PIPE.navy }, svg + html + stagesHtml);
  }

  window.FigureKit = { algorithm, pipeline, rich, tex, ALGO_VARS, PIPE_LAYOUT };
})();
