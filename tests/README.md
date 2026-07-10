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

**Tier 1 — frozen e2e pins** (`test_gw_jax_regression.py`): four fresh
`gw.gw_jax` runs covering whole pipelines transitively (ζ-fit → V_q → χ₀ →
W → PPM fit → 4-branch Σ → head → QP extraction → writers):

| gate | fixture | unique coverage |
|------|---------|-----------------|
| `si_cohsex_3d` | `regression/si_cohsex_debug` | **BGW anchor** (0.12 meV), sys_dim=3 Coulomb + analytic head. Do not shrink/re-freeze. |
| `cohsex` | `regression/cohsex_debug` | only IBZ-stored WFN → ψ k-unfold e2e; 12-op group; nspinor=2 static SX/COH; K_POINTS path |
| `gnppm` | `regression/gnppm_debug` | dynamic GN-PPM workhorse; IBZ cascade active (log-asserted); session state for Tier 2 |
| `bispinor` | `regression/bispinor_debug` | bispinor GN-PPM: Σ^B + 4 ζ channels + 7 V_q tiles + transverse γ̃ |

**Tier 2 — invariance gates** (`test_invariance_gates.py`): two paths that
must agree, self-checking (no frozen refs), run as `restart = true`
variants from a COPY of the gnppm session state (`conftest.py` session
fixtures) so the ζ-fit/V_q are not redone per gate: restart≡fresh, μ-pad
flips (gnppm + bispinor), kij↔kij_stream, SC-iter-1≡one-shot, fixed-point
frozen rotations, IBZ≡full-BZ.  Every major 2026 bug class lives here.

**Tier 3 — unit tests**: only what the gates cannot see — config parsing
(`test_qp_solver_config`), quadrature math vs analytic values
(`test_minimax_quadrature`), TRS/symmetry-unfold invariants on synthetic
data (`test_symmetry_unfold`), loader/IO contracts (`test_wfn_loader_eager`,
`test_zeta_reader`, `test_slab_io_ffi_contract`, `test_mf_isdf_header_roundtrip`),
kernel identities vs independent references (`test_wfn_transforms`,
`test_zq_from_psi_sm_bit_identity`, `test_compute_all_V_q_g_flat`,
`test_compute_V_q_bispinor_g_flat`, `test_per_q_sphere`), head-fit sign
regressions (`test_head_correction`), planner floors
(`test_band_chunk_size_floor`), QSGW band partition (`test_band_partition`),
restart pad roundtrip (`test_restart_pad_roundtrip`), BGW eqp format
(`test_eqp_bgw`), plus ONE kmeans smoke test (`test_kmeans_smoke` — kmeans
is a fixture-generation tool; deeper breakage fails visibly at regen).

**`extra` marker** (deselected by default via pyproject `addopts`; run
with `-m extra`): tooling/experimental/out-of-repo-fixture suites —
`test_sternheimer_solvers`, `test_head_wing_schur`, `test_aot_memory`,
`test_R_proper_cri3`, `test_reshard_all_to_all`.

`-m "not regression"` remains the 1–2 min unit-only loop.

## multi_device/

Cross-P invariance scripts (different GPU counts) — driven by
`multi_device/run_tier2.sh` under SLURM, not collected by pytest.
