## LORRAX: Low-scaling Real-space Real-Axis eXcited state package

LORRAX implements an efficient GW workflow accelerated by Interpolative Separable Density Fitting (ISDF). The main GW driver is called GWJAX (`gw.gw_jax`). It reads plane-wave DFT wavefunctions (WFN.h5) and computes quasiparticle corrections via static COHSEX or GN-PPM (Godby-Needs Generalized Plasmon Pole) frequency dependence.

- **Input**: DFT wavefunctions on a plane-wave grid, symmetry maps, and k-point sampling
- **Core idea**: Replace dense charge-density products with a compact ISDF basis defined by centroids r_mu
- **Outcome**: Exchange and screened-exchange self-energy matrix elements and band-edge corrections

### High-level pipeline

1. Charge density from selected bands → choose ISDF points r_mu via k-means/CVT
2. Read wavefunctions cnk(G), FFT to real space psi_nk(r)
3. For each q, construct zeta_q,mu(r) by solving C_q zeta_q = Z_q using least-squares
4. Compute V_q,mu,nu from zeta_q,mu in G-space with the Coulomb kernel v_q(G)
5. Build Green’s function G and (optionally) chi0 and screened interaction W
6. Form sigma_X/SX/COH and project to band representation Sigma_kij

See formalism details in formalism.md. For runnable examples, see examples/.

### API reference (Markdown-only)

Generated Markdown lives under `docs/api/`.

Generate locally (no server):

```bash
uv add pdoc  # once per environment
uv run -- bash docs/gen_api_docs.sh
```

Then browse the `.md` files in `docs/api/` in your editor or on GitHub.

### Key modules

- `src/gw/gw_jax.py`: COHSEX driver (JAX, sharded)
- `src/gw/w_isdf.py`: static screening and chi0 helpers
- `src/isdf/centroid/kmeans_isdf.py`: centroid selection
- `src/isdf/common/wfnreader.py`: wavefunction I/O

### Frequency-Integration Docs

- `docs/MINIMAX_QUADRATURE.md`: GL/HGL theory, minimax solver methods, and CTSP derivations
- `docs/FREQ_INTEGRATION_REWRITE_PLAN.md`: rewrite blueprint for a unified chi/sigma frequency-integration engine

### Getting started

1. Create the environment (uv recommended) and run tests
2. Provide WFN.h5 and centroids
3. Run the COHSEX driver

See the root README for full instructions.

### Quick runs

- Regression fixture:
  - `python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in`
- Console entry point:
  - `gw_jax -i tests/regression/cohsex_debug/cohsex_test.in`
