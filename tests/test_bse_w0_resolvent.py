"""Gate: the non-TDA RPA-screening resolvent reproduces the GW static W.

``bse_w_exact`` computes W(0) - v = v (0 - H_RPA)^{-1} v with the non-TDA
symplectic RPA density-response Hamiltonian (ring kernel V in both blocks).
This must reproduce the restart's head-less (W0_qmunu - V_qmunu) q=0 tile to the
GW minimax-integration floor (~1e-9 on the gnppm fixture).  Exercises the whole
chain touched by the W(0) cross-check: ``inject_head=False`` loading, the
``screening=True`` B-block, ``ensure_W_R``, the probe generator/snapshot, and the
[f; -f] / X+Y symplectic convention.
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh, PartitionSpec as P

from bse import bse_io
from bse.bse_w_exact import (
    _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols,
    build_finite_q_data, _symmetry_reduced_q_list,
)
from symmetry_maps import kgrid_shift_map                        # noqa: E402


@pytest.mark.gpu
def test_w0_resolvent_matches_restart(gnppm_session):
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))

    # FULL chi0 window (n_occ x n_cond), head-less bodies on both sides.
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False)
    nlog = int(data["n_rmu"])
    T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
         - np.asarray(jax.device_get(data["V_q0"])))

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)
    cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
    W_tile, resids = _resolve_wc_columns(
        cols, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh,
        max_iter=200, tol=1e-10)

    # The resolvent projects back to the assembled (mu_X, nu_Y) tile directly:
    # the reduce-scatter snapshot must land nu on y and mu on x (P('x','y')), so
    # no replicated (mu, nu) is ever materialized.
    assert W_tile.sharding.spec == P("x", "y"), (
        f"W tile carries {W_tile.sharding.spec}, expected P('x','y') = (mu_X, nu_Y)")

    wc = np.asarray(jax.device_get(W_tile))     # (n_rmu, n_pad); column i = cols[i]
    resids = np.asarray(jax.device_get(resids))
    for i, nu0 in enumerate(cols):
        tcol = T[:nlog, int(nu0)]
        rel = float(np.linalg.norm(wc[:nlog, i] - tcol) / np.linalg.norm(tcol))
        assert resids[i] < 1e-6, f"col {nu0}: GMRES not converged (resid={resids[i]:.2e})"
        assert rel < 1e-6, f"col {nu0}: W(0) resolvent closure rel_err={rel:.2e} (>1e-6)"


def test_kgrid_shift_map_matches_roll():
    """``kgrid_shift_map`` is a permutation, folds k+q correctly, has G∈{0,1}³ for
    on-grid q, is the identity at q=0, and its gather EQUALS ``jnp.roll`` — the ONE
    place the finite-q k+q arithmetic lives (pure numpy, no GPU)."""
    nkx, nky, nkz = 4, 4, 4
    nk = nkx * nky * nkz
    a = np.arange(nk).astype(np.float64)
    ar = a.reshape(nkx, nky, nkz)
    ix, iy, iz = np.meshgrid(np.arange(nkx), np.arange(nky), np.arange(nkz), indexing="ij")
    kc = np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)], axis=1)
    for q in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (3, 3, 3), (1, 2, 3)]:
        kpq, G_umk = kgrid_shift_map(nkx, nky, nkz, q)
        assert np.array_equal(np.sort(kpq), np.arange(nk)), f"q={q}: not a permutation"
        assert G_umk.min() >= 0 and G_umk.max() <= 1, f"q={q}: G_umk not in {{0,1}}"
        # fold identity: k + q == kpq_coords + kgrid·G_umk
        kpq_c = kc[kpq]
        assert np.array_equal(kc + np.asarray(q), kpq_c + np.asarray([nkx, nky, nkz]) * G_umk)
        # gather ≡ jnp.roll(shift=-q): out[k] = a[k+q]
        rolled = np.roll(ar, shift=(-q[0], -q[1], -q[2]), axis=(0, 1, 2)).reshape(-1)
        assert np.array_equal(rolled.astype(int), a[kpq].astype(int)), f"q={q}: roll≠gather"
    kpq0, G0 = kgrid_shift_map(nkx, nky, nkz, (0, 0, 0))
    assert np.array_equal(kpq0, np.arange(nk)) and np.all(G0 == 0), "q=0 not identity"


@pytest.mark.gpu
def test_wq_resolvent_matches_restart_finite_q(gnppm_session):
    """Finite-q generalization of the W(0) gate: the RPA screening resolvent at the
    smallest nonzero symmetry-reduced q reproduces the restart's own
    (W0_qmunu - V_qmunu)[q_flat] tile to the GW minimax-quadrature floor.  Exercises
    build_finite_q_data (roll conduction + finite-q V tile, no umklapp phase), the
    per-operator GMRES cache, and reorthogonalized/lstsq GMRES on the stiff
    (head-carrying) finite-q operator.  Compares each q vs its OWN tile (finite-q
    tiles are ~3% non-covariant under centroid permutation — never sym-unfold)."""
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))

    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])

    # Smallest nonzero symmetry-reduced q (cheapest finite-q case).
    q_list = _symmetry_reduced_q_list(input_path)
    nz = q_list[np.any(q_list != 0, axis=1)]
    q = nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))]
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    assert (qx, qy, qz) != (0, 0, 0), "no nonzero symmetry-reduced q on this fixture"

    dq = build_finite_q_data(data, (qx, qy, qz), mesh)
    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, dq)
    W0 = np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
    Vq = np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz]))
    T = W0 - Vq

    cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
    W_tile, resids = _resolve_wc_columns(
        cols, 0.0 + 0.0j, dq, matvec, diag_h, gen, snapshot, sh,
        max_iter=200, tol=1e-10)
    assert W_tile.sharding.spec == P("x", "y"), (
        f"W_q tile carries {W_tile.sharding.spec}, expected P('x','y')")

    wc = np.asarray(jax.device_get(W_tile))
    resids = np.asarray(jax.device_get(resids))
    for i, nu0 in enumerate(cols):
        tcol = T[:nlog, int(nu0)]
        rel = float(np.linalg.norm(wc[:nlog, i] - tcol) / np.linalg.norm(tcol))
        assert resids[i] < 1e-6, (
            f"q={(qx, qy, qz)} col {nu0}: GMRES not converged (resid={resids[i]:.2e})")
        assert rel < 1e-6, (
            f"q={(qx, qy, qz)} col {nu0}: W_q resolvent closure rel_err={rel:.2e} (>1e-6)")
