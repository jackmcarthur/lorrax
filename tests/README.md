# LORRAX test suite

Redesigned 2026-07-09 (see `lorrax_sandbox/reports/test_suite_redesign_2026-07-09/`).
Plain invocation, one GPU, no xdist/srun overrides required:

```bash
LORRAX_NGPU=1 lxrun python3 -m pytest -q tests      # Perlmutter (~4 min)
uv run python -m pytest -q tests                    # local dev
```

Optional 4-GPU parallel run (kept working, never required):

```bash
lxrun python3 -m pytest -q tests -p xdist -n 4      # conftest pins worker→GPU
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

**Tier 3 — unit tests** (17 files): only what the gates cannot see —
config parsing (`test_qp_solver_config`), quadrature math vs analytic
values (`test_minimax_quadrature`), TRS/symmetry-unfold invariants on
synthetic data (`test_symmetry_unfold`), loader/IO contracts
(`test_wfn_loader_eager`, `test_file_io` — headers + ZetaReader + SlabIO),
kernel identities vs independent references (`test_wfn_transforms`,
`test_zq_from_psi_sm_bit_identity` — z_q streaming + PsiGStore slicer,
`test_compute_all_V_q_g_flat` — incl. per-q Coulomb sphere,
`test_compute_V_q_bispinor_g_flat`, `test_sigma_x_bispinor`), head-fit
sign regressions (`test_head_correction`), PPM window freeze
(`test_sigma_ppm_gates` G2), planner floors (`test_band_chunk_size_floor`),
QSGW band partition (`test_band_partition`), restart pad roundtrip
(`test_restart_pad_roundtrip`), BGW eqp format (`test_eqp_bgw`), plus ONE
kmeans smoke test (`test_kmeans_smoke` — kmeans is a fixture-generation
tool; deeper breakage fails visibly at regen).

**`extra` marker** (deselected by default via pyproject `addopts`; run
with `-m extra`): tooling/experimental/out-of-repo-fixture suites —
`test_sternheimer_solvers`, `test_head_wing_schur`, `test_aot_memory`,
`test_R_proper_cri3`, `test_reshard_all_to_all`.

`-m "not regression"` remains the 1–2 min unit-only loop.

## multi_device/

Cross-P invariance scripts (different GPU counts) — driven by
`multi_device/run_tier2.sh` under SLURM, not collected by pytest.
