"""A metal's static head row is the Schur complement of one NSD static chi0.

The static row folds the intraband wings ``Y = -C sum_kn w_kn u_kn`` (``u =
|psi|^2`` at the centroids, ``w`` the Fermi-surface table of ``kappa_TF^2``)
through a Gamma body.  A body built from q = 0 pair sums has no diagonal pairs,
hence no intraband block, and the fold through it is not a Schur complement:
bcc Fe 8^3 on main 23d904a67 gave ``kappa_eff^2 = -12.9 bohr^-2``.  Checked on
CPU on a toy static chi0 (two bands with Fermi-surface weights, three
centroids; four on a 2 x 2 mesh):

* ``qsgw_head.static_intraband_body_sharded`` equals ``-C sum w u u^T`` on
  full and on parents-only storage (one operation class that permutes
  centroids), and the static wings read the same operands;
* the fold through ``metal_static_gamma_body(W_inter)`` equals the exact
  Schur complement of ``[head, wing; wing, body]`` and gives
  ``0 < kappa_eff^2 <= kappa_TF^2``;
* the fold through ``W_inter`` alone (the tangential route before the fix)
  is not that Schur complement and refuses its negative ``kappa_eff^2``.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)

NK, NS, NB, OMEGA = 2, 1, 2, 40.0
#: Two bands with Fermi-surface weights at two k; one state is off the surface.
SURFACE = np.array([[0.7, 0.0], [0.3, 1.1]])


def _meshes():
    out = [(Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y")), 3)]
    if len(jax.devices()) >= 4:
        out.append((Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")), 4))
    return out


def _toy(mu, seed=3):
    """psi, a PD bare V, an NSD interband body and W_inter = (V^-1 - B_inter)^-1."""
    rng = np.random.default_rng(seed)
    psi = (rng.normal(size=(NK, NS, mu, NB))
           + 1j * rng.normal(size=(NK, NS, mu, NB))).astype(np.complex128)
    M = rng.normal(size=(mu, mu))
    V = M @ M.T + 8.0 * np.eye(mu)
    A = rng.normal(size=(mu, 2))
    B_inter = -0.05 * A @ A.T
    W_inter = np.linalg.inv(np.linalg.inv(V) - B_inter)
    return psi, V, B_inter, W_inter


def _face(psi):
    return SimpleNamespace(layout="face", psi_mun=jnp.asarray(psi),
                           psi_nmu=jnp.asarray(psi.transpose(0, 3, 1, 2)),
                           enk=None, occ=None, slices=None, green_parent=None)


def _parents_only(psi, mu, n_shards):
    """Four full-k rows from NK parents in two operation classes (identity and
    a swap of the first two centroids of every shard, shard-local as the
    orbit-packed carrier); returns the bundle and the rows' densities."""
    loc = mu // n_shards
    ident = np.tile(np.arange(loc), n_shards)
    swap = ident.copy()
    swap[0::loc], swap[1::loc] = ident[1::loc], ident[0::loc]
    local_perm = np.stack([ident, swap]).astype(np.int32)
    irr_idx = np.array([0, 1, 0, 1])
    class_of_row = np.array([0, 0, 1, 1], dtype=np.int32)
    plan = SimpleNamespace(
        irr_idx=irr_idx, n_parent=NK, n_full=4,
        operation_classes=lambda: SimpleNamespace(
            ops=np.arange(2), class_of_row=class_of_row, local_perm=local_perm))
    parent = SimpleNamespace(**vars(_face(psi)), plan=plan)
    wfns = SimpleNamespace(layout="face", psi_mun=None, psi_nmu=None,
                           slices=None, green_parent=parent)
    u = np.sum(np.abs(psi) ** 2, axis=1)                        # (NK, mu, NB)
    shard = (np.arange(mu) // loc) * loc
    rows = np.stack([u[irr_idx[r]][shard + local_perm[class_of_row[r]]]
                     for r in range(4)])                        # (4, mu, NB)
    return wfns, rows


def _response(wfns, surface, mesh, mu, nk):
    from gw.qsgw_head import static_head_wings_sharded
    C = 2.0 / nk
    Y, Z = static_head_wings_sharded(wfns, surface, mesh=mesh, nb_logical=NB,
                                     nk_tot=nk, nspin=1, nspinor=NS)
    kappa2 = 8.0 * np.pi * C * float(np.sum(surface)) / OMEGA
    return SimpleNamespace(static_kappa2_bohr2=kappa2, static_Y_x=Y, static_Z_y=Z,
                           static_surface=(surface, NB, 1, NS))


def _schur(u, surface, V, B_inter, nk):
    """Exact static fold of [h, y^T; y, B_inter + B_FS] through (V^-1 - B)^-1."""
    C = 2.0 / nk
    h = -C * np.sum(surface) / OMEGA
    y = -C * np.einsum("kn,kmn->m", surface, u)
    B_fs = -C * np.einsum("kn,kmn,kvn->mv", surface, u, u)
    W = np.linalg.inv(np.linalg.inv(V) - B_inter - B_fs)
    full = np.block([[np.array([[h]]), y[None] / np.sqrt(OMEGA)],
                     [y[:, None] / np.sqrt(OMEGA), B_inter + B_fs]])
    assert np.linalg.eigvalsh(full).max() < 1e-12          # one NSD static chi0
    return B_fs, y, -8.0 * np.pi * (h + y @ W @ y / OMEGA)


@pytest.mark.parametrize("parents", [False, True])
@pytest.mark.parametrize("mesh_mu", _meshes(), ids=lambda m: f"mesh{m[0].devices.size}")
def test_static_fold_is_schur_complement(mesh_mu, parents):
    from gw.qsgw_head import (_fold_static_kappa2, metal_static_gamma_body,
                              static_intraband_body_sharded)
    mesh, mu = mesh_mu
    psi, V, B_inter, W_inter = _toy(mu)
    if parents:
        wfns, u = _parents_only(psi, mu, mesh.devices.shape[0])
        surface = np.concatenate([SURFACE, SURFACE[::-1]])      # (4 rows, NB)
    else:
        wfns, u, surface = _face(psi), np.sum(np.abs(psi) ** 2, axis=1), SURFACE
    nk = surface.shape[0]
    B_fs, y, kappa2_exact = _schur(u, surface, V, B_inter, nk)

    body = static_intraband_body_sharded(wfns, surface, mesh=mesh, nb_logical=NB,
                                         nk_tot=nk, nspin=1, nspinor=NS)
    np.testing.assert_allclose(np.asarray(body)[0], B_fs, rtol=0, atol=1e-12)
    response = _response(wfns, surface, mesh, mu, nk)
    np.testing.assert_allclose(np.asarray(response.static_Y_x), y, rtol=0, atol=1e-12)

    meta = SimpleNamespace(nk_tot=nk, n_rmu=mu, mu_solve_extent=mu, mu_basis=None,
                           nspinor_wfnfile=NS, nspin=1)
    config = SimpleNamespace(backend=SimpleNamespace(w_dyson_solver="local"))
    W_fs = metal_static_gamma_body(jnp.asarray(W_inter, jnp.complex128)[None], wfns, response,
                                   meta=meta, config=config, mesh=mesh)
    kappa2 = _fold_static_kappa2(response, W_fs[0], OMEGA, mesh)
    assert kappa2_exact > 0.0
    assert kappa2 == pytest.approx(kappa2_exact, rel=1e-10)
    assert kappa2 <= response.static_kappa2_bohr2 * (1 + 1e-12)

    # The tangential route before the fix: the intraband wings over the
    # interband body.  Its "fold" is no Schur complement and goes negative.
    C = 2.0 / nk
    unfolded = -8.0 * np.pi * (-C * np.sum(surface) / OMEGA + y @ W_inter @ y / OMEGA)
    assert unfolded < 0.0 < kappa2_exact
    with pytest.raises(ValueError, match="nonphysical screening"):
        _fold_static_kappa2(response, jnp.asarray(W_inter, jnp.complex128), OMEGA, mesh)
