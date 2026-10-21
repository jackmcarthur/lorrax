// Pipeline figure spec: the LORRAX driver chain (the flowchart design's staircase: stages step down and to the right,
// utilities hang below the stage that feeds them). Drawn by chain-map.js on every driver and utility page; see
// lorrax_flowcharts/figure-kit/pipeline-figure.md. Chips may be { id, label }: the id is what a page's data-driver
// or data-treatment selects.
window.SPECS = window.SPECS || {};
window.SPECS.lorraxPipeline = {
  macros: { '\\bk': '\\mathbf{k}', '\\br': '\\mathbf{r}' },
  layout: { stageWidth: 300, stageStepX: 392, stageStepY: 88 },
  stages: [
    { id: 'mf', number: '01', title: 'Mean field', summary: '$\\psi_{n\\bk}(\\br),\\ E_{n\\bk}$',
      optionsLabel: 'QE', options: ['scf', 'nscf', 'kgrid', { id: 'nscf-lorrax', label: 'or LORRAX nscf' }], handoff: 'WFN.h5',
      helperLabel: 'GW inputs', helpers: [
        { name: 'pw2bgw', note: 'writes `WFN.h5` from QE' },
        { name: 'centroids', note: 'ISDF interpolation points $\\br_{\\mu}$' },
        { name: 'kin-ion', note: '$\\langle m\\bk|T+V^{\\mathrm{ion}}|n\\bk\\rangle$' },
        { name: 'dipole', note: 'velocity $\\langle m\\bk|\\hat{v}|n\\bk\\rangle$' }] },
    { id: 'gw', number: '02', title: 'GW', summary: '$\\Sigma^{xc}_{n\\bk}(\\omega)\\ \\rightarrow\\ E^{\\mathrm{QP}}_{n\\bk}$',
      optionsLabel: 'Σ', options: [{ id: 'mpa', label: 'shared pole' }, { id: 'gn_ppm', label: 'GN-PPM' }, { id: 'cohsex', label: 'COHSEX' }, { id: 'hl_ppm', label: 'HL-GPP' }],
      handoff: 'restart · eqp1.dat',
      helperLabel: 'BSE inputs', helpers: [
        { name: 'htransform', note: 'QP bands on a $\\bk$ path' },
        { name: 'downfold', note: '$V$, $W$ on a band subspace' }] },
    { id: 'bse', number: '03', title: 'BSE', summary: '$H^{\\mathrm{BSE}} A^{S} = \\Omega_{S}\\, A^{S}$',
      optionsLabel: 'calc', options: [{ id: 'bse', label: 'absorption' }, { id: 'exciton-bands', label: 'exciton bands' }] }
  ]
};
