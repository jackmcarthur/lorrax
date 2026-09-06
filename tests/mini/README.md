# Four-process mini suite

From the checkout, on Perlmutter:

```bash
lx run -N 1 -G 4 -n 4 -- python3 -m tests.mini --output /path/to/new/run
```

The runner requires four processes, one GPU per process, and a 2×2 mesh.
It uses one runtime for the whole sequence and calls the production entry
points directly. No pytest collection, xdist, nested MPI launches, or QE runs
are part of the timed command. The ordinary core and full tiers are separate.
Service primitives run before the physics drivers to catch transport or algebra
failures early. The primitive eigensolve/Cholesky/LU cell omits backend and route
overrides, and records the service's selected plans in `summary.json`.

Every invocation creates a new output directory, retains each rank's logs,
and updates `summary.json` after each check. An existing directory refuses.
The default budget is 120 seconds measured from Python entry, including JAX
startup, imports, compilation, staging, and numerical checks. Allocation and
launcher time are outside that clock. An over-budget completed suite fails;
it cannot quietly drop checks to pass. `--budget-seconds 600` is available for
profiling. Collective operations have no rank-local
timer; the launch supervisor owns termination of a stalled job.

The new mean-field fixture is a deliberately cheap periodic
H₂⁺ crystal with an 8 Ry wavefunction cutoff, a 5×5×1 k-grid, nine stored
two-component bands, and a ferromagnetic moment along its screw axis. Its
SCF is converged; its energies are a software fixture, not a materials result.
Seven bands are active in GW and three appear in Sigma. The real WFN and
QE schema authenticate broken TR and fractional translations at runtime.
All inputs, pseudopotential bytes, and WFN are hashed in `PROVENANCE.json`.

The default coverage is:

| Check | Mechanisms |
|---|---|
| Fresh charge centroids | Real P4 Lloyd, orbit closure, oversampling and Gram pruning |
| Fresh current centroids | Magnetic spinors, current Gram, orbit closure, oversampling and pruning |
| Literal charge centroids | Non-orbit path, no oversampling/pruning, 19 points |
| Fresh kinetic/ionic preprocessing | Pseudopotential loader and parallel HDF5 writer |
| Three fresh COHSEX calculations | Local/high, distributed/low, local/low band memory; symmetry unfolding, zeta, Coulomb, screening, Sigma, QP output |
| One COHSEX restart | Distributed/high band memory from a copy of local/high's completed restart state; real loader invocation, no refitting, fresh/restart numerical parity |
| Fresh scalar MPA | Existing helium fixture B, 7 active bands, 13 centroids, 2 poles, frequency-dependent Sigma, frozen quadrature rules |
| SlabIO primitives | Real parallel HDF5; complex 5×7 data, two-axis and product-axis sharding, empty padded rank, zero padding, append/offset updates, packed reads, metadata, invalid-extent refusals |
| Existing dense-algebra bodies | Default-plan eigensolve/Cholesky/LU with a ragged five-matrix batch, plus real cuSolverMp/cuBLASMp, hostile extents, factor/solve residuals and route refusals |

The four COHSEX results must agree with one another and with hashed numerical
references: all 27 k/band rows, all printed Sigma components, and EQPs within
0.02 meV. MPA uses the existing core reference's 0.2-meV tolerance, checks
all 25 frequencies of its small Sigma tensor, and regenerates its fit/stores.
The distributed/high combination covers restart loading and downstream GW;
its fresh-fit combination is deliberately omitted to fit restart coverage
into the existing four runs. Each fresh arm must fit once and never load a
restart; the restart arm must load once and never fit. Copied EQP/Sigma
outputs are removed before the restart runs, so stale parent results cannot
satisfy the comparison. No mode is accepted by a skip or xfail.
The GW calculation currently disables head completion and stochastic
body averaging to make the comparison deterministic. MPA temporarily uses
the same development permission for runtime minimax solving as the core B
test; this is not a certification of that setting for production.

The magnetic case exercises 25 full / 9 stored k-points, 7 active bands,
and 6 orbit-closed centroids; none is divisible by four. The literal selector's
19 and MPA's 13 centroids additionally exercise odd extents on the 2×2 mesh.
The restart arm uses its own writable copy of the completed local/high run's
`tmp/` state, preserving the original run. All arms share the runtime's compiled
executables. No persistent compilation cache was used for the recorded runs.

GN/HL-PPM, SC, BSE, htransform, head/body corrections, and production-scale
memory remain in the other tiers.

Measurement and final coverage are recorded in the sandbox report identified
in the feature commit. The two-minute target is not a convergence claim.


Fixture maintenance uses the same tool validation and subprocess helpers as
`tests/core/fixtures/build_fixtures.py` and always writes to a new directory:

```bash
lx run -N 1 -G 0 -n 1 -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  python3 -m tests.mini.build_fixture --output /path/to/new/qe
```

`BUILD.json` deliberately says `built_unvalidated`: use the sandbox's
registered `qe_scf_xml_report.py` on `scf-schema.xml`, and
`check_wfn_band_edges.py` on `WFN.h5`, before publishing a replacement.
The builder never updates the committed fixture or its reference stamp.
Vxc/KIH export is disabled in `pw2bgw.in` because QE's magnetic Vxc writer
refuses spatial symmetries; the mini suite runs LORRAX's own preprocessing.
