# Measured campaign cost and remaining scope

This note records measurements, not projected speedups. The material is the
CrI3 electronic **8×8×1** grid, **750 physical bands in the internal sum**, **3000 logical
ISDF centroids**, and a **160-band protected SC Hamiltonian**. The coarse
response is the full-BZ **4×4×1** grid with the explicitly selected experimental
Fejér real-space window; it is not a near-Γ q patch. Electronic Green functions
still use all 64 k points. The coarse approximation and its non-Φ qualification
are described in [k-convolution](../../../docs/architecture/kconv.md#bosonic-response-grid).

## Actual scalar SC timings

On native P16, source `861d228`, job `60056663`, proper q4 SC reached its native
1 meV convergence verdict at map 7. These are individual **protected160 SC**
maps, not 44-band collision sweeps or full-coherent TD refreshes:

| Map | W response (s) | Σ τ sweep (s) | Map wall (s) | Actual window/τ pairs |
| --- | ---: | ---: | ---: | ---: |
| 0, cold | 116.7 | 32.7 | 160.5 | 192, 8 windows |
| 1 | 47.7 | 22.4 | 72.5 | 225, 8 windows |
| 2 | 45.5 | 17.4 | 64.9 | 238, 9 windows |
| 7, held rules | 45.2 | 17.3 | 64.5 | 238, 9 windows |

Maps 3–7 held 238 branch-distinct τ nodes in nine windows; map 7 built zero
new rules. A window/τ pair is not an individual FFT or an isolated GPU kernel.
The Σ executable includes W(τ) synthesis, Green construction, G/W convolution,
band projection, and the frequency fold. W response includes sampling, Dyson,
model construction, checks, and I/O; it is not just a dielectric solve.

Across all eight maps, the timing tree reports Σ τ sweep **158.785 s**,
`tau.synthesis_setup` **3.996 s**, `tau.setup` **1.806 s**, and
`tau.initial_compile_and_probe` **9.429 s**. Setup printed 1.6 s on map 0 and
0.0 s on maps 1–7 (one-decimal display, not exact zero). Initial compile/probe
printed 5.0 s and 4.4 s on maps 0 and 1, then 0.0 s. This is **not evidence of
a material warm setup/recompilation bottleneck**. Do not optimize the fresh
`sigma_subtile_operands` JIT closure on function-identity suspicion alone.

Evidence is under
`/home/jackm/mgkba_campaign_20261019/qgrid_20261020/`:

- `material_q4_fejer861_60056663.rank0.log`: map rows near lines 7130–14638;
  last quadrature at 14536; timing tree at 14990–14998; explicit
  `LORRAX_STAGE_SYNC=1` device attribution at 15085.
- `material_q4_fejer861_accepted_60056663/`: actual accepted-input frame and
  native driver/solver verdicts. An SC output spectrum alone is not that join.

## A measured optimization: checked Coulomb-root reuse

The first q4 constructor receipt records **all 16 physical q rows** borrowing
the existing checked cover `[0,16]`: `hit=true`, `fresh_solve=false`, source
layout `local`, requested solve layout `distributed`. The three requested
spans are `[0,6]`, `[6,12]`, and `[12,16]`. Its total constructor Coulomb phase
is **0.540 s**; it did not recompute those roots. This is not a measured total
W speedup against a matched uncached run. Bare-root construction still has a
cold cost elsewhere; cache residency and live-room admission remain charged.

Evidence: `material_q4_fejer861_60056663/constructor_receipt.json`,
`q_receipts[*].constructor.coulomb.root_cache` and `seconds.coulomb`.
Owner: `src/gw/response_bank.py::_coulomb_batch` and
`src/gw/shared_pole_constructor.py` (cache-only borrowing preserves the
original fresh-solve route and all checks).

## What the compact44 benchmark really requests

`active_sigma_probe.py:31–52` constructs external energies only for
`[100,144)` and requests `projection_bands=(100,144)`. In
`src/gw/mpa/sigma.py:2693–2694`, the planner uses the request's actual
frequencies, not an all160 frequency grid. Lines 1503–1542 select the
destination faces on device and set a 44-band output carrier. The internal
Green faces remain full-band. In `src/gw/ppm_tau_kernel.py:359–414`, the
Green GEMM keeps the physical750/padded carrier, while the projector and
accumulator use 44 destinations. W's authenticated q domain has 16 rows;
the electronic convolution still has 64 rows. There is no crop-after160
execution in this benchmark.

The actual accepted q4 input has protected `[0,160)` energies
`[-78.4640192095, +1.8408462531]` eV (width **80.3048654626 eV**) and active
`[100,144)` energies `[-11.2051361769, -2.4157639738]` eV (width
**8.7893722031 eV**), in the same native absolute energy zero. These are from
`material_q4_fejer861_accepted_60056663/frame_0007_call_0007.npz`, not the older
q8 reference. They establish the narrower external support, not linear
runtime scaling with bandwidth or band count.

**A successful compact44 timing is still pending in this evidence set.**
The earlier source309 callback failed before Σ and supplies no sweep timing.
The prepared two-sweep benchmark measures raw750 equilibrium correlation
BODY only: no head, spectral-tail extrapolation, density-coherent internal
line, refreshed-W TD fixed point, or propagation. Its second sweep uses the
same physical W but may use different certified held/padded rules, so it is
not automatically an identical-executable warm-kernel measurement.

Full-coherent TD uses the compact ordered path
(`src/gw/ppm_sigma.py:2203–2272`), retaining active rates and full virtual MH
from bracketed contributions in one sweep. Its **material cost is not yet
measured** here. Reducing output bands does not remove the full750 Green
sum, μ-space work, electronic FFT, or current-W refresh.

## Specific remaining opportunities, not measured gains

- The older *passive* `window_sigma_c` path still executes prefix
  `[0,b_hi)` destinations and crops afterward: `ppm_sigma.py:2130–2140`,
  `2532–2534`, `2576–2579`, `2647–2656`. Its `window_only` option narrows
  frequencies, not the destination carrier. The current ordered full-ρ
  branch bypasses it at `2560–2566` and passes the compact interval at
  `2267–2272`. A future passive fix must compact the request, projection,
  block indices and head join together; it cannot be credited to current TD.
- `linalg=local` is **not a universal backend switch**. Passive shared-pole
  synthesis first prices replicated pole columns, then may use whole-parent
  local execution, otherwise face panels (`mpa/sigma.py:956–1085`). Read the
  actual schedule receipt. Ordered general synthesis explicitly uses face
  GEMM/distributed workspace (`2053–2081`), and the bounded general-pencil
  reduction uses host SciPy (`shared_pole_constructor.py:191–330`). A local
  flag does not change those paths. No unpriced route change is proposed.

## Software validation is not material BSE validation

Native P4 source `4b24c0e` tested the actual BSE matvec against the literal
coarse `1/Nq` screened term plus fine `1/Nk` bare exchange, with distinct
heads injected once. Maximum complex error was **4.518280359883027e−16**;
the sparse q-comb error was exactly zero. Receipt:
`p4_bse4b_60056663/receipt.json` under the evidence directory above.
This qualifies the tiny software mapping only, not material exciton roots,
optical convergence, a TD trajectory, or general positivity.
