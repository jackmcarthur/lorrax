# LORRAX test suite

> ## ⚠ THE FOUR-GPU RULE — every GPU leg runs at P=4
>
> Whichever tier you run, **every GPU verification leg runs at P=4**. A
> P=1-only verification is never sufficient for landing; unit and CPU cells
> are exempt. The owner's rationale, verbatim:
>
> > *"use four gpus for 100% of all testing so that never ever do we run
> > something on one GPU and then learn it doesn't generalize later"*
>
> `lx test` already takes all four GPUs on the node; a driver leg wants
> `-G 4` rather than the one-GPU default. See
> [`AGENT_PREAMBLE.md`](../AGENT_PREAMBLE.md).

## Two tiers: the default gate, and the census (2026-08-09)

The owner's instruction, verbatim:

> the test suite for lorrax became a super clodgy mess because of the llm
> test-for-everything habit; really we should run that Si test calculation
> (granted for all drivers that were touched since last ran) and the tests
> for the services and have that basically be it.

So:

```bash
pytest                 # DEFAULT GATE — minutes.  The Si end-to-end test
                       # calculation for the drivers this branch TOUCHED,
                       # plus every service's own suite.
pytest --census        # THE CENSUS — everything.  Byte-for-byte the run a
pytest -m census       # bare `pytest` was before the split (3371 cells at
                       # the 2026-08-10 rebase; the count tracks the tree).
lx test                # Perlmutter: the default gate on a compute node
lx test --census       # Perlmutter: the census
```

Nothing was deleted and nothing changed meaning. **`tests/KNOWN_FAILURES.md`
accounts for the CENSUS run**, and that accounting is untouched: `--census`
collects exactly the set the old default collected, measured node id by node
id at the split.

The default gate's roster and its file→driver map live in
[`fast_gate.py`](fast_gate.py); the twenty lines of pytest wiring are in
[`conftest.py`](conftest.py). Every cell the default gate runs already
existed — no deck and no tolerance was authored for this split. Drivers with
no runnable in-tree deck are named in `fast_gate.UNDECKED` and the run says
so out loud when you touch one.

The gate stands down — you get the census — whenever you have already said
what you want: `--census`, `-m`, `-k`, a named path, `--no-services`,
`--only-service`, or `LX_CENSUS=1`. Override the driver selection directly
with `LX_GATE_DRIVERS=all|none|<comma list>`, and the diff base with
`LX_GATE_REF` (default: merge-base with `origin/main`).

---

## The census, in detail

Redesigned 2026-07-09 (see `lorrax_sandbox/reports/test_suite_redesign_2026-07-09/`).
Plain invocation, one GPU, no xdist/srun overrides required:

```bash
LORRAX_NGPU=1 lxrun python3 -m pytest -q --census tests   # Perlmutter (~4 min)
uv run python -m pytest -q --census tests                 # local dev
```

Optional 4-GPU parallel run (kept working, never required):

```bash
lxrun python3 -m pytest -q --census tests -p xdist -n 4   # conftest pins worker→GPU
```

## Architecture — three tiers

**Tier 1 — frozen e2e pins** (`test_gw_jax_regression.py`): seven fresh
`gw.gw_jax` runs over six decks (hBN runs twice, once per arm) covering whole
pipelines transitively (ζ-fit → V_q → χ₀ → W → PPM fit → 4-branch Σ → head →
QP extraction → writers).  Five are frozen against LORRAX's own past output;
**one is checked against BerkeleyGW**, and it is the only place in the suite
where an external code enters the loop; and **one is not a freeze at all but a
LIVENESS control** — the hBN `mc_average_vcoul_body` A/B, which asserts a
physics knob still MOVES Σ:

| gate | fixture | unique coverage |
|------|---------|-----------------|
| `si_cohsex_3d` (production) | `regression/si_cohsex_debug`, `cohsex_si_test.in` | **The BerkeleyGW anchor.** sys_dim=3 Coulomb + analytic head. Two gates on one run: bit-identity vs `eqp_si_ref.dat`, AND `test_si_production_matches_berkeleygw` vs literal BGW columns in `bgw_sigma_hp_noavg.dat` (MEASURED sigTOT 0.644 meV MAE, full BZ). **Do not shrink/re-freeze** — use the fast deck instead. |
| `si_cohsex_fast` | `regression/si_cohsex_debug`, `cohsex_si_fast.in` | Same crystal, 20 bands / 144 centroids, ~12 s. A PURE SELF-FREEZE — MEASURED 2109 meV from BGW (the band cut, not the centroid count). Gates code changes fast; says nothing about BGW. |
| `hbn_cohsex_3d` | `regression/hbn_cohsex_debug`, `cohsex_hbn_test.in` | **The only NON-CUBIC 3D deck.** Si FCC satisfies `bvec.T = P·bvec` for a cyclic signed permutation, so the mini-BZ draw-convention bug class (358bb0b) is a pure RESEED there and no Si gate can see it (measured z = 3.0 = noise); hBN's hexagonal `bvec` admits no such P and the same defect is a bias at z = 293.7. Also the only pinned deck that pins no `vhead` → the native q→0 head ladder runs e2e. Two cells: the self-freeze at atol 1e-5 eV, and `test_hbn_mc_average_vcoul_body_moves_sigma`, the NEGATIVE CONTROL (flip the key, Σ must move >5 meV MAE; measured 13.995 vs a 0.396 meV MC seed width). NOT a BerkeleyGW anchor. |
| `cohsex` | `regression/cohsex_debug` | only IBZ-stored WFN → ψ k-unfold e2e; 12-op group; nspinor=2 static SX/COH; K_POINTS path |
| `gnppm` | `regression/gnppm_debug` | dynamic GN-PPM workhorse; IBZ cascade active (log-asserted); session state for Tier 2 |
| `bispinor` | `regression/bispinor_debug` | bispinor GN-PPM: Σ^B + 4 ζ channels + 7 V_q tiles + transverse γ̃ |

**Tier 2 — invariance gates** (`test_invariance_gates.py`): two paths that
must agree, self-checking (no frozen refs), run as `restart = true`
variants from a COPY of the gnppm session state (`conftest.py` session
fixtures) so the ζ-fit/V_q are not redone per gate: restart≡fresh, μ-pad
flips (gnppm + bispinor), kij↔kij_stream, SC-iter-1≡one-shot, fixed-point
frozen rotations, IBZ≡full-BZ.  Every major 2026 bug class lives here.

**Tier 3 — unit tests** (18 files): only what the gates cannot see —
config parsing (`test_qp_solver_config`), quadrature math vs analytic
values (`test_minimax_quadrature`), TRS/symmetry-unfold invariants on
synthetic data (`test_symmetry_unfold`), loader/IO contracts
(`test_wfn_loader_eager`, `test_file_io` — headers + ZetaReader + SlabIO),
kernel identities vs independent references (`test_wfn_transforms`,
`test_zq_from_psi_sm_bit_identity` — z_q streaming + PsiGStore slicer,
`test_compute_all_V_q_g_flat` — incl. per-q Coulomb sphere,
`test_compute_V_q_bispinor_g_flat`, `test_bispinor_route_exhaustive`), head-fit
sign regressions (`test_head_correction`), PPM window freeze
(`test_sigma_ppm_gates` G2), planner floors (`test_band_chunk_size_floor`),
QSGW band partition (`test_band_partition`), restart pad roundtrip
(`test_restart_pad_roundtrip`), BGW eqp format (`test_eqp_bgw`), the
k-basis of the energy files (`test_eqp_kpoint_basis` — eqp{0,1}.dat on the
IBZ wedge, sigma_diag/eqp_g0w0 on the full BZ, every block naming its own
k so nothing downstream has to pair by position), plus ONE
kmeans smoke test (`test_kmeans_smoke` — kmeans is a fixture-generation
tool; deeper breakage fails visibly at regen).

**`extra` marker** (deselected by default via pyproject `addopts`; run
with `-m extra`): tooling/experimental/out-of-repo-fixture suites —
`test_sternheimer_solvers`, `test_head_wing_schur`, `test_aot_memory`,
`test_cartesian_actions_cri3`, `test_reshard_all_to_all`.

`-m "not regression"` remains the 1–2 min unit-only loop.

## multi_device/

Cross-P invariance scripts (different GPU counts) — driven by
`multi_device/run_tier2.sh` under SLURM, not collected by pytest.  The same
directory now also holds the arms that need a real `.so` pair rather than a
second GPU count: `multi_device/restart_q_storage_ab.sh` is the full-BZ vs
IBZ-wedge restart A/B, which cannot be a pytest cell because SlabIO needs the
phdf5 FFI and every restart-writer cell in the tree is red without it.  See
`multi_device/README.md` for the roster and the measured record of each.
