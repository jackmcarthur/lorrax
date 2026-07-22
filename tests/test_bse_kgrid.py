"""Gates for the ``bse_k_grid`` coarse→fine general-BSE-init densification.

The knob lives in the GENERAL init
(``bse.bse_io.load_bse_data_from_restart_sharded``): when set and DIFFERENT
from the coarse restart grid it interpolates the WHOLE bundle (ψ/ε via
htransform, V_Q via vq_interp, W via zero-pad in R) so every solver runs on the
fine grid.  Unset or == coarse → the coarse bundle is returned byte-identically.

  1. **Parser** (fixture-free, CPU): ``_parse_grid_spec`` accepts
     ``"nx ny nz"`` / ``"nx,ny,nz"`` / tuple / empty / None, rejects the rest.
  2. **On-grid identity** (fixture+GPU): ``bse_k_grid == coarse`` returns a
     byte-identical bundle vs the no-flag path (the fast-path guarantee).
  3. **Densify** (fixture+GPU): a small 3×3→6×6 run returns a fine-grid bundle
     with the right shapes (nk=36, band/μ dims unchanged) and a W_q on the fine
     k-lattice, and it is solvable (finite, ordered, positive eigenvalues).

All 1-GPU (no 16-GPU gating).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")

FXDIR = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
         "05_lorrax_cohsex_native")

# Modest htransform window (nband=40 <= 48) so the fine-grid conduction ENERGY
# recovery stays accurate (10_lorrax_exciton_bands nband<=48 finding).
_KG_INPUT = """[cohsex]
restart = false
centroids_file = centroids_frac_640.txt
nval = 26
ncond = 14
nband = 40
sys_dim = 2
do_screened = true
screening_method = minimax
fermi_reference = midgap
sigma_at_dft_energies = true
memory_per_device_gb = 28
wfn_file = WFN.h5
bare_coulomb_cutoff = 30.0
"""


def test_parse_grid_spec():
    from bse.bse_io import _parse_grid_spec
    assert _parse_grid_spec(None) is None
    assert _parse_grid_spec("") is None
    assert _parse_grid_spec("   ") is None
    assert _parse_grid_spec("12 12 1") == (12, 12, 1)
    assert _parse_grid_spec("12,12,1") == (12, 12, 1)
    assert _parse_grid_spec("6, 6, 1") == (6, 6, 1)
    assert _parse_grid_spec((8, 8, 2)) == (8, 8, 2)
    assert _parse_grid_spec([4, 4, 1]) == (4, 4, 1)
    for bad in ("12 12", "1 2 3 4", "12x12x1"):
        with pytest.raises((ValueError,)):
            _parse_grid_spec(bad)


def _make_run(tmp_path):
    run = tmp_path / "kg"
    run.mkdir()
    os.symlink(f"{FXDIR}/tmp", run / "tmp")
    os.symlink(f"{FXDIR}/WFN.h5", run / "WFN.h5")
    os.symlink(f"{FXDIR}/centroids_frac_640.txt", run / "centroids_frac_640.txt")
    (run / "cohsex_kg.in").write_text(_KG_INPUT)
    return run


pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{FXDIR}/tmp/zeta_q.h5"),
    reason="MoS2 3x3 640-centroid fixture not present")


def _load(restart, inp, mesh_xy, bse_k_grid):
    from bse.bse_io import load_bse_data_from_restart_sharded
    return load_bse_data_from_restart_sharded(
        restart, n_val=2, n_cond=2, mesh_xy=mesh_xy, input_file=str(inp),
        inject_head=True, bse_k_grid=bse_k_grid)


@pytest.mark.gpu
def test_on_grid_identity_byte_equal(tmp_path):
    """bse_k_grid == coarse → the bundle is byte-identical to the no-flag path."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)
    cg = (int(d0["nkx"]), int(d0["nky"]), int(d0["nkz"]))
    d1 = _load(restart, inp, mesh_xy, cg)          # == coarse → fast path
    for k in ("psi_c_X", "psi_v_X", "M_X", "eps_c", "eps_v", "W_q", "V_q0"):
        a, b = np.asarray(jax.device_get(d0[k])), np.asarray(jax.device_get(d1[k]))
        assert np.array_equal(a, b), f"{k} not byte-identical on the fast path"


@pytest.mark.gpu
def test_densify_3to6_shapes_and_solvable(tmp_path):
    """3×3→6×6 densification: fine-grid bundle shapes + a solvable spectrum."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_w_exact import _create_mesh_xy
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from solvers.lanczos import block_lanczos_eig_jit
    import jax.numpy as jnp

    run = _make_run(tmp_path)
    restart = str(run / "tmp" / "isdf_tensors_640.h5")
    inp = run / "cohsex_kg.in"
    mesh_xy = _create_mesh_xy(1, 1)
    d0 = _load(restart, inp, mesh_xy, None)
    df = _load(restart, inp, mesh_xy, (6, 6, 1))

    assert (int(df["nkx"]), int(df["nky"]), int(df["nkz"])) == (6, 6, 1)
    nk_f = 36
    # band / μ dims are grid-invariant; only the k axis (and W's k-axes) grow.
    assert df["n_val"] == d0["n_val"] and df["n_cond"] == d0["n_cond"]
    assert df["n_rmu_pad"] == d0["n_rmu_pad"]
    assert df["psi_c_X"].shape[0] == nk_f
    assert df["psi_v_X"].shape[0] == nk_f
    assert df["eps_c"].shape[0] == nk_f
    assert tuple(df["W_q"].shape[-3:]) == (6, 6, 1)
    assert df["V_q0"].shape == d0["V_q0"].shape        # μ-basis tile unchanged shape
    assert bool(np.all(np.isfinite(np.asarray(jax.device_get(df["eps_c"])))))

    # solvable: lowest TDA excitons finite, ordered, positive
    nc_pad, nv_pad = int(df["n_cond_pad"]), int(df["n_val_pad"])
    n_flat = nc_pad * nv_pad * nk_f
    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_stack_matvec(mesh_xy, 6, 6, 1, kernel="bse")
    W_R = jnp.fft.ifftn(df["W_q"], axes=(2, 3, 4), norm="ortho")
    bs = 4

    def mvb(Vb):
        X = Vb.reshape(bs, nc_pad, nv_pad, nk_f)
        X = jax.lax.with_sharding_constraint(X, sh.X)
        HX = matvec(X, df["psi_c_X"], df["psi_c_Y"], df["psi_v_X"], df["psi_v_Y"],
                    df["eps_c"], df["eps_v"], W_R, df["V_q0"], df["M_X"], df["M_Y"])
        return HX.reshape(bs, -1)

    evs, _ = block_lanczos_eig_jit(mvb, n_flat, n_eig=4, block_size=bs,
                                   max_iter=20, n_reorth=20)
    evs = np.asarray(jax.device_get(evs))[:4].real
    assert np.all(np.isfinite(evs))
    assert np.all(np.diff(evs) >= -1e-6)               # ascending
    assert evs[0] > 0.0                                # bound state above zero
