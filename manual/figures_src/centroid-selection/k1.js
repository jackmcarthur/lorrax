// Fig. K1 — ISDF centroid selection, the sketched-QRCP selector (docs/assets/figures/k1.png;
// used by docs/theory/centroid-selection.md and docs/utilities/centroids.md). Render: bash make_k1.sh (this folder).
//
// A figure-kit algorithm spec. The kit, the KaTeX harness and the shared macros (window.LORRAX_MACROS) live in the
// lorrax_flowcharts project (~/projects/lorrax_flowcharts: figure-kit/, render/, specs/_shared.js); this file replaces
// lorrax_flowcharts/specs/k1.js, which still draws the retired dense-Gram pivoted-Cholesky prune.
//
// Source: LORRAX branch claude/rqrcp-centroids @ 0cab74596 (src/centroid/). The dense M x M Gram and its pivoted Cholesky
// (centroid/pivoted_cholesky.py) are deleted; the sketched QRCP is the only selector.
// Stage 1: sampling_metric.build_feature_metric_diagonal (s = K(r,r)); kmeans_cli._resolve_weight :323, w = s^P with
//          P = --weight-power, default 5/6 (:72, :361), so Lloyd's point density ~ w^(3/5) = s^(1/2).
// Stage 2: kmeans_isdf.weighted_kmeans_jax (Lloyd, k-means++ seed); kmeans_cli._snap_and_unfold :374;
//          M = ceil(3 N_mu) (--oversample 3.0, :32; M_cand :510).
// Stage 3: sketch_select.build_sketch_q0_via_loadwfns :425 (module docstring :1-60). Y = Omega Z, Khatri-Rao Gaussian:
//          Y_j(a) = r'^(-1/2) sum_k sqrt(w_k) (sum_{m in L} g^L_jkm psi*_mk(a)) (sum_{n in R} g^R_jkn psi_nk(a)),
//          g ~ CN(0,1) from jax.random.fold_in on (k, row, --seed), never stored beyond one k; E[Y^H Y] = Z^H Z = K.
//          r' = N + max(64, ceil(0.1 N)) (sketch_rows_default :135, --sketch-rows). Current mode (--density-mode current):
//          Z_i(a) = <Psi^L(a)|alpha^i|Psi^R(a)>, one independent draw per component, E[Y^H Y] = sum_i Z_i^H Z_i
//          (sketch_vertices :236) -- the stage note.
// Stage 4: sketch_select.sketch_block_select :763 (block kernel _select_block_kernel :562, deflation _deflate_kernel :737).
//          Per block: orbits scored by the mean shrink-corrected residual d_hat = d r'/(r'-t) over members above the floor
//          (sqrt(eps) max d0, :838-840); the top b orbits' columns C gathered, H = C^H C; orbits admitted in order while
//          their in-block score >= gamma x the leading score (--sketch-gamma 0.7) and they fit the point budget, each member
//          pivoted by pivoted Cholesky on H, largest residual first; the admitted directions orthonormalized by CholQR2;
//          Y <- Y - Q (Q^H Y) once. Stops when a block admits nothing (:869). A member under the floor is kept with its
//          orbit but adds no direction (not drawn). Whole orbits only: candidates.emit_complete_groups :249.
// Stage 5: the consumer, the zeta fit of the GW run (Fig. G1, docs/theory/parallel-algorithms.md; docs/theory/isdf.md §2):
//          zeta_q mu(G) on the irreducible q. Drawn as its own stage so the block loop strip of stage 4 does not cover it.
// Kept from the retired K1 (lorrax_flowcharts/specs/k1.js, rounds 3-5): objects, stage 1 and stage 2 layout, captions.
window.SPECS = window.SPECS || {};
window.SPECS.k1 = {
  title: 'ISDF centroid selection',
  label: 'Fig. K1',
  // Inputs and Outputs side by side: keeps the five-stage figure near the old K1's 1734 x 1515 css px
  objectsLayout: 'top-inline',
  macros: Object.assign({}, window.LORRAX_MACROS, {
    '\\ra': '\\mathbf{r}_{a}', '\\rb': '\\mathbf{r}_{b}', '\\dhat': '\\hat d', '\\Qb': 'Q_{\\mathrm b}'
  }),

  objects: {
    psiG: { kind: 'input', math: '\\psi_{n\\bk}(\\bG)', file: 'WFN.h5',
      definition: ['w_{\\bk}:\\ \\text{full-BZ weights},\\ \\textstyle\\sum_{\\bk}w_{\\bk}=1'] },
    LR: { kind: 'input', math: 'L,\\ R', file: 'band windows',
      definition: ['L=[0,\\,N_v+N^{\\Sigma}_c),\\quad R=[0,\\,N_b)',
                   '\\text{unit weight per band; occupations unused}'] },
    rmu: { kind: 'output', math: '\\rmu', file: 'centroids_frac_<N>.txt',
      definition: ['\\psi^{*}_{m\\bk}(\\br)\\psi_{n\\bk+\\bq}(\\br)\\approx\\textstyle\\sum_{\\mu}\\zeta_{\\bq\\mu}(\\br)\\,\\psi^{*}_{m\\bk}(\\rmu)\\psi_{n\\bk+\\bq}(\\rmu)',
                   '\\mu=1\\ldots \\Nmu;\\ \\text{whole symmetry orbits}'] }
  },

  stages: [
    { number: '1', title: 'Feature weight',
      rows: [
        [ '@psiG', { arrow: 'IFFT', equation: '\\bG\\to\\br' }, '\\psi_{n\\bk}(\\br)' ],
        [ '@LR' ] ],
      thenJoin: [
        { arrow: 'sum bands', equation: '' },
        '\\rho^{A}_{\\bk}(\\br)=\\textstyle\\sum_{n\\in A}|\\psi_{n\\bk}(\\br)|^{2},\\ \\ A=L,R',
        { arrow: 'k-sum', equation: '' },
        { box: ['s(\\br)=\\textstyle\\sum_{\\bk}w_{\\bk}\\,\\rho^{L}_{\\bk}\\rho^{R}_{\\bk}=K(\\br,\\br)', 'w(\\br)=s(\\br)^{5/6}'], desc: 'k-means weight' } ] },

    { number: '2', title: 'Weighted Lloyd k-means', loop: 'until the centroids stop moving (k-means++ seed)',
      rows: [[
        'w(\\br)',
        { arrow: 'assign', equation: '' },
        { box: 'V_a=\\{\\br:\\ \\ra\\ \\text{is the nearest centroid}\\}', desc: 'Voronoi cell' },
        { arrow: 'update', equation: '' },
        '\\ra=\\textstyle\\sum_{V_a}w\\,\\br\\,/\\sum_{V_a}w',
        { arrow: 'snap, unfold', equation: '\\text{orbit closure}' },
        { box: '\\ra,\\ \\ a=1\\ldots M,\\ \\ M=\\lceil 3\\,\\Nmu\\rceil', desc: 'candidate pool (default oversampling)' } ]] },

    { number: '3', title: 'Candidate sketch', note: 'current: one draw per $\\alpha^i$',
      rows: [[
        '\\psi_{n\\bk}(\\br)',
        { arrow: 'evaluate', equation: '\\text{at the candidates}' },
        '\\psi_{n\\bk}(\\ra)',
        { arrow: 'sketch', equation: 'Y=\\Omega Z' },
        // Z is never formed: its definition rides in the sketch box (line 2), not in a box of its own
        { box: ['Y_j(\\ra)=\\tfrac{1}{\\sqrt{r\'}}\\textstyle\\sum_{\\bk}\\sqrt{w_{\\bk}}\\,\\big(\\sum_{m\\in L}g^{L}_{j\\bk m}\\,\\psi^{*}_{m\\bk}(\\ra)\\big)\\big(\\sum_{n\\in R}g^{R}_{j\\bk n}\\,\\psi_{n\\bk}(\\ra)\\big)',
                'Z_{\\bk mn}(\\ra)=\\sqrt{w_{\\bk}}\\,\\psi^{*}_{m\\bk}(\\ra)\\,\\psi_{n\\bk}(\\ra),\\quad \\mathbb{E}\\,[Y^{\\dagger}Y]=Z^{\\dagger}Z=K(\\ra,\\rb)',
                'j=1\\ldots r\',\\ \\ r\'=\\Nmu+\\max(64,\\lceil 0.1\\,\\Nmu\\rceil),\\quad g\\sim\\mathcal{CN}(0,1)'],
          desc: 'Khatri–Rao Gaussian sketch of the $\\mathbf q=0$ pair matrix $Z$' } ]] },

    { number: '4', title: 'Sketched QRCP pruning',
      loop: 'one block per pass; repeat while a whole orbit still fits in the $\\Nmu$ budget',
      rows: [[
        { box: '\\dhat(\\ra)=\\frac{r\'}{r\'-t}\\,\\big\\|(1-QQ^{\\dagger})\\,Y(\\ra)\\big\\|^{2}', desc: 'shrink-corrected residual; $t$ directions kept' },
        { arrow: 'gather a block', equation: '' },
        { box: ['C=Y(\\ra),\\ \\ \\ra\\in O_1\\cup\\ldots\\cup O_b', 'H=C^{\\dagger}C'], desc: 'the $b$ best orbits by $\\langle\\hat d\\rangle_O$' },
        { arrow: 'admit, pivot', equation: '\\langle\\dhat\\rangle_{O}\\ge\\gamma\\,\\langle\\dhat\\rangle_{O_1}' },
        { box: 'H=\\Pi\\,LL^{\\dagger}\\,\\Pi^{\\mathsf T}', desc: 'block pivoted Cholesky' },
        { wrap: true },
        { arrow: 'CholQR2', equation: '' },
        { box: 'C_{\\mathrm{adm}}=\\Qb R,\\ \\ \\Qb^{\\dagger}\\Qb=1', desc: 'admitted columns, in pivot order' },
        { arrow: 'deflate once', equation: '' },
        { box: ['Y\\leftarrow Y-\\Qb\\,(\\Qb^{\\dagger}Y)', 'Q\\leftarrow[\\,Q,\\ \\Qb\\,]'], desc: 'every column, one pass' },
        { arrow: 'keep whole orbits', equation: '' },
        '@rmu' ]] },

    // no-break spaces: the title row is a flex box, which drops plain spaces next to the math
    { number: '5', title: 'The $\\zeta$ fit', ref: 'Fig. G1',
      rows: [[
        '@rmu',
        { arrow: 'fit', equation: '\\text{irreducible }\\bq' },
        '\\zeta_{\\bq\\mu}(\\bG)' ]] }
  ]
};
