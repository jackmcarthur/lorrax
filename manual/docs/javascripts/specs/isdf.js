// Algorithm figure spec: ISDF fit and bare Coulomb.
// See algorithm-figure.md for the vocabulary.
window.SPECS = window.SPECS || {};
window.SPECS.isdf = {
    title: 'ISDF fit and bare Coulomb',
    label: 'Fig. 2',
    macros: { '\\bk': '\\mathbf{k}', '\\bq': '\\mathbf{q}', '\\br': '\\mathbf{r}', '\\bG': '\\mathbf{G}' },

    objects: {
      psiG: { kind: 'input', math: '\\psi_{n\\bk}(\\bG)', file: 'WFN.h5',
        definition: ['\\psi_{n\\bk}(\\br) = \\tfrac{1}{\\sqrt{\\Omega}} \\textstyle\\sum_{\\bG} e^{i(\\bk+\\bG)\\cdot \\br}\\, \\psi_{n\\bk}(\\bG)'] },
      rmu: { kind: 'input', math: '\\br_{\\mu}', file: 'centroids_file',
        definition: ['\\br_{\\mu} \\in \\Omega, \\quad \\mu = 1, \\ldots, N_{\\mu}'] },
      LR: { kind: 'input', math: 'L,\\ R', file: 'band windows',
        definition: ['m \\in L = \\{0, \\ldots, N_{L}-1\\}', 'n \\in R = \\{0, \\ldots, N_{R}-1\\}'] },
      vq: { kind: 'input', math: 'v(\\bq+\\bG)', file: 'kernel',
        definition: ['v(\\bq+\\bG) = 4\\pi / |\\bq+\\bG|^{2}'] },
      zeta: { kind: 'output', math: '\\zeta_{\\bq\\mu}(\\br)', file: 'tmp/zeta_q.h5',
        definition: ['\\psi^{*}_{m\\bk}(\\br)\\,\\psi_{n,\\bk-\\bq}(\\br) \\approx \\textstyle\\sum_{\\mu} \\zeta_{\\bq\\mu}(\\br)', '\\qquad \\times\\, \\psi^{*}_{m\\bk}(\\br_{\\mu})\\,\\psi_{n,\\bk-\\bq}(\\br_{\\mu})'] },
      V: { kind: 'output', math: 'V_{\\bq,\\mu\\nu}', file: 'to screening',
        definition: ['V_{\\bq,\\mu\\nu} = \\tfrac{1}{\\Omega} \\textstyle\\sum_{\\bG} \\zeta^{*}_{\\bq\\mu}(\\bq+\\bG)', '\\qquad \\times\\, v(\\bq+\\bG)\\, \\zeta_{\\bq\\nu}(\\bq+\\bG)'] }
    },

    stages: [
      { number: '1', title: 'Real-space orbitals', rows: [[
        '@psiG', { arrow: 'ifft', equation: '\\psi(\\br) = \\mathcal{F}^{-1}\\psi(\\bG)' },
        '\\psi_{n\\bk}(\\br)', { arrow: 'gather', equation: '\\psi(\\br) \\to \\psi(\\br_{\\mu})', extraInput: '@rmu' },
        '\\psi_{n\\bk}(\\br_{\\mu})']] },
      { number: '2', title: 'Band projectors', rows: [['\\psi_{n\\bk}(\\br)'], ['\\psi_{n\\bk}(\\br_{\\mu})']],
        thenJoin: [{ arrow: 'gemm', equation: 'P = \\psi\\,\\psi^{\\dagger}' }, ['P^{L}_{\\bk}(\\br,\\br_{\\mu}),', 'P^{R}_{\\bk}(\\br,\\br_{\\mu})']] },
      { number: '3', title: 'Right-hand side', loop: 'for $\\bk \\in \\mathrm{BZ}$', rows: [[
        ['P^{L}_{\\bk}(\\br,\\br_{\\mu}),', 'P^{R}_{\\bk-\\bq}(\\br,\\br_{\\mu})'],
        { arrow: 'k-convolution', equation: 'ZC^{\\dagger} = P^{L*} \\ast P^{R}' },
        '(ZC^{\\dagger})_{\\bq}(\\br,\\br_{\\mu})']] },
      { number: '4', title: 'Least-squares fit', rows: [[
        '(ZC^{\\dagger})_{\\bq}', { arrow: 'gather rows', equation: 'CC^{\\dagger} \\subset ZC^{\\dagger}' },
        '(CC^{\\dagger})_{\\bq}(\\br_{\\nu},\\br_{\\mu})', { arrow: 'solve', equation: 'ZC^{\\dagger} = \\zeta\\, CC^{\\dagger}', extraInput: '(ZC^{\\dagger})_{\\bq}' },
        '@zeta']] },
      { number: '5', title: 'Bare Coulomb', rows: [[
        '\\zeta_{\\bq\\mu}(\\br)', { arrow: 'fft', equation: '\\zeta(\\bG) = \\mathcal{F}\\zeta(\\br)' },
        '\\zeta_{\\bq\\mu}(\\bq+\\bG)', { arrow: 'contract', equation: 'V = \\zeta^{\\dagger} v\\, \\zeta', extraInput: '@vq' },
        '@V']] }
    ],

    referenceStages: [
      { number: 'R', title: 'Direct pair densities', loop: 'for $\\bk,\\ m \\in L,\\ n \\in R$',
        rows: [['\\psi_{m\\bk}(\\bG)', { arrow: 'ifft' }, '\\psi^{*}_{m\\bk}(\\br)'], ['\\psi_{n,\\bk-\\bq}(\\bG)', { arrow: 'ifft' }, '\\psi_{n,\\bk-\\bq}(\\br)']],
        thenJoin: [{ arrow: 'multiply', equation: 'Z = \\psi^{*}\\psi' }, 'Z_{mn}(\\br)', { arrow: 'fft', equation: 'M = \\mathcal{F}Z' }, 'M_{mn}(\\bq+\\bG)'] }
    ]
};
