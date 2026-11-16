"""A metal's static head row is the Schur complement of one NSD static chi0, on every route.

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
* the fold through ``W_inter`` alone (the tangential route before 46e3f9780,
  and the MPA route until the dressing moved into the common fold owner) is
  not that Schur complement and refuses its negative ``kappa_eff^2``;
* ``qsgw_head.iteration_head_sample_terms``, the fold every route calls with
  its own undressed Gamma body (tangential model, MPA origin sample), dresses a
  metal's static row and only that row; an insulator's z = 0 row is folded
  through the body it is given, bit for bit.
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
    """The static row's Fermi-surface block, from its one owner."""
    from gw.qsgw_head import _metal_static_head
    meta = SimpleNamespace(nk_tot=nk, cell_volume=OMEGA)
    config = SimpleNamespace(mpa=SimpleNamespace(metal_origin_shift_ry=None))
    kappa2, Y, Z, chi_fs = _metal_static_head(
        wfns, surface, (0j,), mesh=mesh, meta=meta, config=config,
        nb_logical=NB, nspin=1, nspinor=NS)
    return SimpleNamespace(static_kappa2_bohr2=kappa2, static_Y_x=Y, static_Z_y=Z,
                           static_chi_fs=chi_fs)


def _meta(nk, mu):
    return SimpleNamespace(nk_tot=nk, n_rmu=mu, mu_solve_extent=mu, mu_basis=None,
                           nspinor_wfnfile=NS, nspin=1, cell_volume=OMEGA)


def _config():
    from gw.gw_config import HeadCorrection
    return SimpleNamespace(backend=SimpleNamespace(w_dyson_solver="local"),
                           head=SimpleNamespace(correction=HeadCorrection.FULL),
                           mpa=SimpleNamespace(metal_origin_shift_ry=None))


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
    np.testing.assert_allclose(np.asarray(response.static_chi_fs)[0], B_fs, rtol=0, atol=1e-12)

    meta, config = _meta(nk, mu), _config()
    W_fs = metal_static_gamma_body(jnp.asarray(W_inter, jnp.complex128)[None], response,
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


def _route_response(wfns, surface, mesh, mu, nk, *, metal):
    """A two-row head response: the static row (z = 0) and one dynamic row."""
    rng = np.random.default_rng(11)
    static = _response(wfns, surface, mesh, mu, nk)
    S = np.stack([-0.01 * np.eye(3), -0.02 * np.eye(3)]).astype(np.complex128)
    Y = 0.05 * (rng.normal(size=(2, 3, mu)) + 1j * rng.normal(size=(2, 3, mu)))
    return SimpleNamespace(
        omegas=(0j, 0.5j), S_direct=jnp.asarray(S), Y_x=jnp.asarray(Y),
        Z_y=jnp.asarray(np.conj(np.swapaxes(Y, 1, 2))), fold_scale=None,
        static_kappa2_bohr2=static.static_kappa2_bohr2 if metal else None,
        static_Y_x=static.static_Y_x, static_Z_y=static.static_Z_y,
        static_chi_fs=static.static_chi_fs if metal else None,
        drude_tensor=np.eye(3) if metal else None,
        fermi_surface=object() if metal else None)


@pytest.mark.parametrize("mesh_mu", _meshes(), ids=lambda m: f"mesh{m[0].devices.size}")
def test_every_route_dresses_the_static_row(mesh_mu):
    """The common fold owner dresses a metal's static row from the raw Gamma body.

    Every route hands ``iteration_head_sample_terms`` its own Gamma body: the
    tangential model at z = 0 and the MPA origin sample are both q = 0 pair
    sums (no diagonal pairs).  On main 46e3f9780 only the tangential route
    dressed it, upstream; the MPA static row folded ``W_inter`` and refused.
    """
    from gw.head_correction import fold_cartesian_head_wings_sharded
    from gw.qsgw_head import iteration_head_sample_terms, metal_static_gamma_body
    mesh, mu = mesh_mu
    psi, V, B_inter, W_inter = _toy(mu)
    wfns, u, surface = _face(psi), np.sum(np.abs(psi) ** 2, axis=1), SURFACE
    nk = surface.shape[0]
    _, _, kappa2_exact = _schur(u, surface, V, B_inter, nk)
    response = _route_response(wfns, surface, mesh, mu, nk, metal=True)
    meta, config = _meta(nk, mu), _config()
    W = jnp.asarray(W_inter, jnp.complex128)

    static = iteration_head_sample_terms(response, 0, W, meta=meta, config=config, mesh=mesh)
    assert static["static_row"]
    assert static["static_kappa2"] == pytest.approx(kappa2_exact, rel=1e-10)
    assert 0.0 < static["static_kappa2"] <= response.static_kappa2_bohr2 * (1 + 1e-12)
    W_st = metal_static_gamma_body(W[None], response, meta=meta, config=config, mesh=mesh)[0]
    want = fold_cartesian_head_wings_sharded(response.S_direct[0], response.Y_x[0], W_st,
                                             response.Z_y[0], OMEGA, mesh_xy=mesh)
    np.testing.assert_allclose(static["S_effective"], np.asarray(want), rtol=0, atol=1e-13)

    dynamic = iteration_head_sample_terms(response, 1, W, meta=meta, config=config, mesh=mesh)
    assert not dynamic["static_row"]
    plain = fold_cartesian_head_wings_sharded(response.S_direct[1], response.Y_x[1], W,
                                              response.Z_y[1], OMEGA, mesh_xy=mesh)
    np.testing.assert_array_equal(dynamic["S_effective"], np.asarray(plain))


@pytest.mark.parametrize("mesh_mu", _meshes(), ids=lambda m: f"mesh{m[0].devices.size}")
def test_insulator_static_row_is_not_dressed(mesh_mu):
    """An insulator's z = 0 row folds the body it is given, bit for bit."""
    from gw.head_correction import fold_cartesian_head_wings_sharded
    from gw.qsgw_head import iteration_head_sample_terms
    mesh, mu = mesh_mu
    psi, _, _, W_inter = _toy(mu)
    wfns, surface = _face(psi), SURFACE
    response = _route_response(wfns, surface, mesh, mu, surface.shape[0], metal=False)
    W = jnp.asarray(W_inter, jnp.complex128)
    terms = iteration_head_sample_terms(response, 0, W, meta=_meta(surface.shape[0], mu),
                                        config=_config(), mesh=mesh)
    plain = fold_cartesian_head_wings_sharded(response.S_direct[0], response.Y_x[0], W,
                                              response.Z_y[0], OMEGA, mesh_xy=mesh)
    assert terms["static_row"] and terms["static_kappa2"] is None
    np.testing.assert_array_equal(terms["S_effective"], np.asarray(plain))


def test_metal_static_row_without_its_block_refuses():
    """A metal static row without ``static_chi_fs`` refuses by name, never folds undressed."""
    from gw.qsgw_head import metal_static_gamma_body
    mesh, mu = _meshes()[0]
    W = jnp.zeros((1, mu, mu), jnp.complex128)
    with pytest.raises(ValueError, match="GATE metal_static_body"):
        metal_static_gamma_body(W, SimpleNamespace(static_chi_fs=None), meta=_meta(2, mu),
                                config=_config(), mesh=mesh)


@pytest.mark.parametrize("mesh_mu", _meshes(), ids=lambda m: f"mesh{m[0].devices.size}")
def test_static_role_swap_is_the_dressed_body(mesh_mu):
    """The static role's Gamma W equals the dressed interband body.

    ``static_fermi_surface_swap`` is ``chi_FS - chi_FS^FD`` with ``-df/dE =
    f (1 - f) / kT`` on the same operands; added to a Matsubara-like Gamma row
    ``chi_inter + chi_FS^FD`` it gives ``(V^-1 - chi_inter - chi_FS)^-1``, which
    is ``metal_static_gamma_body((V^-1 - chi_inter)^-1)`` (Woodbury).
    """
    from gw.qsgw_head import metal_static_gamma_body, static_fermi_surface_swap
    mesh, mu = mesh_mu
    psi, V, B_inter, W_inter = _toy(mu)
    wfns, u, surface = _face(psi), np.sum(np.abs(psi) ** 2, axis=1), SURFACE
    nk = surface.shape[0]
    kT = 0.02
    f = np.array([[0.2, 0.9], [0.55, 0.0]])
    occ = SimpleNamespace(f_kn=f, smearing_family="fd", smearing_width_ry=kT)
    response = _response(wfns, surface, mesh, mu, nk)
    response.static_fs_shape = (NB, 1, NS)
    swap = static_fermi_surface_swap(response, wfns, occ, meta=_meta(nk, mu), mesh=mesh)
    C = 2.0 / nk
    fd = f * (1 - f) / kT
    want = (-C * np.einsum("kn,kmn,kvn->mv", surface, u, u)
            + C * np.einsum("kn,kmn,kvn->mv", fd, u, u))
    np.testing.assert_allclose(np.asarray(swap)[0], want, rtol=0, atol=1e-11)

    B_fd = -C * np.einsum("kn,kmn,kvn->mv", fd, u, u)
    W_swapped = np.linalg.inv(np.linalg.inv(V) - (B_inter + B_fd + np.asarray(swap)[0]))
    W_st = metal_static_gamma_body(jnp.asarray(W_inter, jnp.complex128)[None], response,
                                   meta=_meta(nk, mu), config=_config(), mesh=mesh)
    np.testing.assert_allclose(np.asarray(W_st)[0], W_swapped, rtol=1e-10, atol=1e-12)
    assert static_fermi_surface_swap(None, wfns, occ, meta=_meta(nk, mu), mesh=mesh) is None
