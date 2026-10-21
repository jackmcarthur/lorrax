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

A later standalone replay on the accepted q4 input (step `60056663.7`,
source `4b24c0e`) measured **78.0244 s** for W construction and **20.290 s**
for its protected160 Sigma sweep (219 nodes, nine windows, 18 rules built).
Its response bank used the host tier, 3.36 GB/rank, rather than the warmed SC
device route. The enclosing W timer was 87.1 s because it also included the
**8.91045 s BSE callback**; do not count that BSE twice. These are distinct
cold/warm and residency conditions, not a single universal refresh cost.

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

The band limits have different meanings; the screening space is **not**
reduced to the propagated density window:

| Object | Electronic band support |
| --- | --- |
| Propagated density | Active44 block; spectators retain their reference occupations |
| χ and W construction | All750 physical screening bands, including FF, FA, AF and AA |
| External on-shell Σ matrix | Active44 rows/columns and their requested energies |
| Virtual internal Σ line | All750 bands; real collision destinations are separately active-only |

In particular, FA and AF retain active-to-outside virtual transitions even
though the propagated active–spectator density coherences are zero. A narrow
external on-shell sampling range is not a transition cutoff in χ. The current
full-density builder embeds the active matrix into the full-band density,
uses current right-column energy phases, and sums all four response blocks
before Dyson. The separate bosonic q grid changes spatial sampling, not this
band support. The coherent wrapper is now a thin operand adapter onto the
existing equilibrium direct stream, including its typed-parent mode-11
contraction, row passes and carry. Independent density left endpoints and
node-specific references remain distinct runtime arguments; this is not a
stationary-ρ shortcut. The old separate full-density contraction was removed.
CPU literal/orientation/moment checks pass, and the scalar default matches
the actual parent21fcc77 owner bit-for-bit on a tiny CPU fixture. The native
P4 proof passed on source `dbf45a4`, step `60056663.13`: 20 paired calls,
full+FF/FA/AF/AA, two current E/rho inputs, fine/coarse grids and exact
moments through actual mode11 and compiled execution admission. Maximum
literal error was `5.55179e-16`, moment error `1.22128e-15`; the fixture
receipt took 29.6548 s (39 s Slurm step). See
[the software-proof summary](evidence/COMMON_CHI_P4_PASS_60056663.json) and
the local `qgrid_20261020/p4_commonchiDBF_60056663/receipt.json` under the
campaign artifact root. This is not a material/general-W timing, hsuite,
speedup or positivity result. Earlier measured timings above are unchanged.

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

**The compact44 benchmark completed** on source `4b24c0e`, step
`60056663.8`, with output shape `[2,64,44,44]` and the same authenticated q4
reference/W:

| Sweep | Wall (s) | Window/τ pairs | Windows | Rules built |
| --- | ---: | ---: | ---: | ---: |
| Cold | 6.378333 | 14 | 2 | 4 |
| Held session | 2.249600 | 26 | 2 | 0 |

The two full complex pairs differed by relative Frobenius **2.07e-6** and
maximum **9.97e-7 Ry** (0.01356 meV). This is a diagnostic, not a convergence
verdict. Actual output/receipts are in
`material_active_sigma4bq4_60056663/coarse_pre_sigma/` under the evidence root;
the committed summary is `evidence/ACTIVE44_SIGMA_TIMING_60056663.json`.
The earlier source309 callback failed before Σ and supplied no sweep timing.
This two-sweep benchmark measures raw750 equilibrium correlation
BODY only: no head, spectral-tail extrapolation, density-coherent internal
line, refreshed-W TD fixed point, or propagation. Its second sweep uses the
same physical W but may use different certified held/padded rules, so it is
not automatically an identical-executable warm-kernel measurement.

Full-coherent TD uses the compact ordered path
(`src/gw/ppm_sigma.py:2203–2272`), retaining active rates and full virtual MH
from bracketed contributions in one sweep. A verification-only material
source-frozen measurement is recorded below; a successful updated-W TD
refresh remains unmeasured. Reducing output bands does not remove the full750 Green
sum, μ-space work, electronic FFT, or current-W refresh.

## Source-frozen ordered44 and one-site dielectric: measured, not admitted

Source `707874d`, owned P16 step `60063961.0`, completed the one-site
microscopic q0 dielectric diagnostic at the previously fitted suspect pole,
then the full native ordered Sigma44 t0 snapshot. This control uses the
**unshifted positive equilibrium source W** converted exactly to signed
ordered factors: applied W-lag gamma is **zero**, unlike the direct dielectric
response's configured `.05 eV`. It is explicitly `SOURCE_FROZEN_DIAGNOSTIC`,
not refreshed W, production `joint_frozen` initialization, or matched B/C
admission. The current full-rho lines, 750 virtual bands, active-only real
destinations `[100,144)`, native head/tail and HF/MH reference anchor remain.

| Actual phase | Seconds | Scope |
| --- | ---: | --- |
| Fresh equilibrium source W | 87.23 | Cold host-bank/file-artifact path, not warm SC's 45.2 s |
| Direct one-site response stream | 17.9564 | Four panels, 215 Green pairs, full750 response |
| Positive-to-ordered source adapter | 27.1816 | Exact native store conversion and authentication |
| Ordered44 native Sigma tau sweep | 13 | 77 pairs, four sweeps, Kmax4836; compile/probe3.405 s is inside |
| Full t0 snapshot | 67.4418 | Includes one-site probe, adapter and ordered Sigma; not Sigma alone |

One-site setup/planning took 1.4074/.1751 s; the dielectric inverse stage
including its basic record took 1.6040 s. These timers are nested and must not
be summed indiscriminately. No isolated inclusive ordered-Sigma-only timer
or total updated-W refresh below60s is established by this run.

The computed logical3000 dielectric at
`z=(-.16154070956060437+1.0066523909325316i) Ry` had inverse residual
`1.08714e-13`, with spectral residual upper bound `1.32917e-7` including
roundoff allowance `1.32917e-7`. Its algebraic smallest-singular-value lower
bound was **.0185205**, strictly positive. This excludes a singularity of
the computed sampled dielectric within that numerical certificate; it is
not a proof of exact physical pole absence or a continuation/TD admission.

The source-frozen t0 was **not stationary**: net number rate
`-.163963 /fs`; gross injection/removal `+2.34439/-2.50835 /fs` gave a
**6.7575% gross-normalized imbalance**. This normalization is not the
actual empty-state gain versus filled-state loss normalization. The native
projector-resolved rates independently reconstructed from the saved full
operators are gain **.269177 /fs** and loss **.433140 /fs**, so gain/loss is
.621455: a **37.8545% deficit relative to loss** (46.6920% symmetric
imbalance). The6.76% gross figure must not be presented as this transition
imbalance. The native integrator subsequently refused25as: its measured stability certificate
required at most24.47as. No trajectory, clipping, trace subtraction, gate
relaxation or matched B/C admission followed. Initial positive rate spectra
and inward projector diagnostics do not establish nonlinear Pauli invariance.

Reservation was359s×16 = **1.59556 GPUh**; the numeric step was350s×16 =
**1.55556 active GPUh**. These are alternative views, not additive charges.
The allocation was released immediately. Exact artifacts, hashes and scope:
[SOURCE_FROZEN707_RESULT_60063961.json](evidence/SOURCE_FROZEN707_RESULT_60063961.json).

### Actual native flags and why405s is not a missing-GPU-flag measurement

Both93C and707 startup receipts resolve CUDA A100-SXM4-40GB,16 processes with
one device each, a4×4 NCCL mesh, FP64/complex128 and **native NVIDIA MathDx
cuFFTDx k-convolution**. The native FFI closure is authenticated. The
persistent np16 compile cache is armed; XLA GPU autotune level0 and parallel
LLVM module compilation are the repository defaults. Runtime pool policy is
cuda_async/preallocate with fraction.8302, measured budget31.58GB/device;
OPENBLAS uses4 threads under8 allocated CPU cores/rank. The deck requests
`linalg=local`; the actual constructor nevertheless uses its priced face
route when the local resident reduction does not fit. No `xla_reference`
fallback or missing native-GPU-enable flag is evidenced.

The93C405.294s stream evaluated6,137 certified Green pairs/28panels, nine
passes of84 local rows, three nodes per accumulation, and distinct FF/FA/AF/AA
blocks. Warm equilibrium W45.2s is a different algorithm/residency/node-count
case, not a like-for-like performance baseline. A single preserved93C NVML
sample showed all16 GPUs at100% instantaneous utilization; that neither
measures sustained efficiency nor proves the tiling optimal. The requested
below60s **total general-W refresh** remains a target, not an achieved result;
quadrature/tiling improvements need matched numerical certificates and actual
new measurements, not flag speculation or accuracy-gate relaxation.

## Specific remaining opportunities, not measured gains

### Implemented value-only contour path; native timing still required

Source `7aa8f92` switches only the full-density general-W caller to the
existing public `minimax.analytic_box_rule` value primitives. The old path
used real-time crossing rules and also certified derivatives that this
value-only realization does not consume. Default value-plus-derivative
callers are unchanged. Exact full-matrix column supports retain even a
`1e-30` coherence; this is not diagonal-population pruning.

On the actual saved750-band state and all26 original transfer-frequency
sites, an owner-level scalar replay gives **856 pairs**, versus6,137:
FF83, FA274, AF242, AA257. The same `1e-8` value certificate, full virtual
space and coefficient-mass checks remain. Forward/reverse contour phases
are formed with non-growing endpoint references to avoid zero-times-infinity
arithmetic. Tiny full-complex CPU oracles cover all four blocks, current
coherences, and both fine and Fejer-restricted response grids.

This node reduction is **not a measured GPU speedup or a sub60s cycle**.
Full750 Green GEMMs, momentum transforms, Dyson, direction selection and
frequency reduction still need native timing and admission. See
[the bounded host certificate record](evidence/COMPLEX_BOX_HOST_CERTIFICATES.json).
The separate native P4 proof subsequently passed20 cases on source7aa:
maximum literal complex error `2.14446e-9`, maximum scalar certificate
`9.76432e-9 < 1e-8`, full response plus all four blocks, two current E/rho
inputs, and both fine and Fejer grids. Its22.8645s tiny-fixture time is not
a material refresh time. See [native proof evidence](evidence/COMPLEX_BOX_NATIVE_P4_60065589.json).

### Correction: the707 t0 request was not on its instantaneous shells

The preserved707 measurements above remain raw historical evidence, but
must **not** be interpreted as a number-conservation verdict for the
specified on-shell MGKBA closure. A subsequent code audit found that
`window_sigma_plan` inherited the producer's initial narrow frequency
extent, rather than the effective extent grown inside SC. Legacy `cover`
semantics map remaining uncovered states to Sigma(0). Here all2,816 active
shells could collapse to zero. A scalar replay with the actual pole census
and zero external frequency reproduces exactly the observed77 pairs/four
windows; requesting the true44 shells gives a different192-pair plan.
Tightening quadrature alone does not cure a wrong evaluation frequency.

Fix `46e3dfb` introduces one shared `ordered_window_request` owner for the
ordered Sigma and its W support-read validation: actual current row/column
shells, a scalar extent extending their minimum/maximum by1eV, explicit
all-covered/no-static-fallback witnesses, and no added boundary evaluation
shells. All750 virtual lines and the legacy SC request semantics remain
unchanged. The fourteen focused request/support tests pass; new native
material verification is still required. The earlier one-site dielectric
witness is independent of this Sigma-request defect and remains valid in
its stated numerical scope.

Source `17dd65a` also separates the **numerical common frequency origin**
from physical chemical-potential metadata. A binary gapped logical reference
uses its accepted global gap midpoint; absolute H/E, occupations, physical
census mu, transfer poles and spectral-tail ladder remain unchanged. Both
external and internal bulk frequencies shift together, and the existing
head owner shifts covariantly. Twenty-one focused tests check exact shells,
support reads, scalar denominator and actual head value/derivative invariance.
This avoids an unnecessary factored-growth refusal in the old skewed
coordinate frame without relaxing the growth certificate. New material
verification remains required; choosing a coordinate origin is not a cure
for a physical collective instability or an inaccurate W realization.

### First general full-density stream: completed 93C stream, refused fit

Source `93c448f`, step `60062281.0`, passed scalar planning in **5.229 s**:
**28 panels / 6,137 Green pairs**, maximum certificate `8.852e-9` at the
unchanged `1e-8` tolerance, coefficient mass **175.9**. The stream then
completed in **405.294 s**. Subsequent synchronized phases were Dyson
**3.171 s**, direction selection **13.651 s**, and pencil assembly
**11.510 s**. These are measured phases of this full750/full-ρ/16q refresh,
not the earlier equilibrium compact44 Σ timing.

The q0 reduced realization was **refused**, not admitted: order 424,
largest pole imaginary part **1.00665239 Ry**, versus electronic evaluation
height **0.01837465 Ry**. No ordered Σ, t0 collision, or propagation followed.
Thus the completed stream is not a successful total-W or TD benchmark.
The two direct microscopic W probes near the suspect pole had relative
fit defects 0.957% and 0.706%; neither was at the pole or supplied a native
dielectric inverse witness. Agreement against a large W background does
not validate the pole. No pole clipping or numerical gate relaxation is used.

One **5.85 s host-only scalar replay**, using the actual logged sites,
current active energy span, block intervals and identical existing owners,
reproduced exactly all 28 panels / 6,137 nodes:

| Block | Green pairs | Panels | Share of nodes, not wall time |
| --- | ---: | ---: | ---: |
| FF | 60 | 3 | 0.98% |
| FA | 3,484 | 12 | 56.77% |
| AF | 1,873 | 9 | 30.52% |
| AA | 720 | 4 | 11.73% |

The low-height crossing groups alone use FA 2,898 + AF 1,604 + AA 579 nodes:
**82.79%** of the total. These include the added `.25 eV` training/held
supports with separate `.05 eV` response-lag decay, on broad transition
intervals; FA spans about 5.59 Ry. In contrast, the newly certified
high-imaginary FA fallback is only 152 nodes (**2.48%**). AA alone and the
scalar planner are therefore not the main *node-count* cause. This does not
authorize dropping verification/training sites, spectator transitions or
coherences, nor loosening the certificate. A future numerical improvement
must preserve the same full-density primitives and requested accuracy.

Actual logs show **9 row passes of 84 local rows**, 3 nodes per accumulate.
Panel 1 (22 nodes) took 4.217 s inclusive, with printed 1.1 s compilation and
2.6 s native dispatch. Panel 4 (384 nodes) took 16.981 s inclusive, with 1.1 s
compilation and 15.7 s dispatch; the next panel's compile time was 0.0 s.
Thus this measured panel is dominated by execution, not compilation repeated
per node. Distinct output counts and final partial-node shapes can compile
once each. The loop skips inactive padded node slots; 384 is a panel carrier,
not a requirement to evaluate 384 physical nodes in a shorter panel.

See [the exact scalar replay and partial timing record](evidence/GENERAL_CHI_SCALAR_REPLAY_93C.json),
which binds the local partial log snapshot and lists all group/panel counts.
Node shares cannot be converted directly to time shares: band-range widths,
FFT work and output counts differ. The final phase/refusal evidence is
`material_ownq4_zero44_93C_60062281/q0_diagnostic/q_000_diagnostic.json`
under the artifact root, alongside `q_000_direct_probes.json` and the
frequency-only NPZ. No successful total-W runtime or speedup is claimed.

### Other scoped opportunities

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
