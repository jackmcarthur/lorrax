"""The two-component charge head is the scalar head with spin traced inside each vertex.

Owner statement: the ``gw.shared_pole_head`` module docstring.  The Coulomb
kernel couples the charge density only, so on an N_spinor = 2 store every head
vertex is the spin-traced pair density ``rho_ij(mu) = sum_s conj(psi_is(mu))
psi_js(mu)``, the capacity is ``2/(n_spin n_spinor)`` states per band, and the
body is the spin-traced ``n_mu x n_mu`` charge operator.  Two oracles pin that:

1. A spin-doubled two-component store -- every scalar band split into an up and
   a down copy with the same spatial part, block-diagonal velocity -- reproduces
   the scalar direct head ``S``, the wings ``Y``/``Z``, the static density wings
   and the folded ``S_eff`` through one planted spin-scalar body.  A missing
   ``2/(n_spin n_spinor)`` capacity would fail this by a factor of two.
2. A global SU(2) rotation of the spinor axis leaves every one of them
   invariant.  A packed ``(mu, spin)`` endpoint, which is not spin-rotation
   covariant, would fail this.

Every cell builds the 2x2 mesh the production kernels shard on
(``@pytest.mark.mesh(4)``): under ``lx test`` it runs on the node's four
GPUs in the mesh child; a direct invocation supplies
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.  It is never
quietly collapsed to a smaller mesh.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.head_correction import fold_cartesian_head_wings_sharded  # noqa: E402
from gw.qsgw_head import (  # noqa: E402
    head_s_tensor_sharded,
    head_wings_sharded,
    static_head_wings_sharded,
)
from gw.wavefunction_bundle import BandSlices, build_wavefunctions_face  # noqa: E402

pytestmark = pytest.mark.mesh(4)


def _mesh_xy():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(f"needs 4 devices for the 2x2 mesh, have {len(devices)}")
    return Mesh(np.asarray(devices[:4], dtype=object).reshape(2, 2), ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _host(a):
    return np.asarray(jax.device_get(a))


def _face(mesh, psi, enk, occ_cut):
    """Canonical faces from host ``psi[k, n, s, mu]`` (the wing tests' convention)."""
    nk, nb, _ns, _nmu = psi.shape
    slices = BandSlices.from_band_edges(0, 0, occ_cut, nb, nb)
    y_in = _put(psi, mesh, P(None, None, None, "y"))
    x_in = _put(np.conj(psi).transpose(0, 3, 1, 2), mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))
    return build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)


def _scalar_store(rng, *, nk, nb, nmu):
    psi = (rng.standard_normal((nk, nb, 1, nmu))
           + 1j * rng.standard_normal((nk, nb, 1, nmu)))
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    v = (rng.standard_normal((3, nk, nb, nb))
         + 1j * rng.standard_normal((3, nk, nb, nb)))
    surface = rng.uniform(0.1, 1.0, size=(nk, nb))
    return psi, enk, v, surface


def _spin_doubled(psi, enk, v, surface):
    """Band ``2n + sigma`` is the scalar band ``n`` placed entirely in spin ``sigma``."""
    nk, nb, _one, nmu = psi.shape
    psi2 = np.zeros((nk, 2 * nb, 2, nmu), dtype=np.complex128)
    v2 = np.zeros((3, nk, 2 * nb, 2 * nb), dtype=np.complex128)
    for sigma in range(2):
        psi2[:, sigma::2, sigma, :] = psi[:, :, 0, :]
        v2[:, :, sigma::2, sigma::2] = v
    enk2 = np.repeat(enk, 2, axis=1)
    surface2 = np.repeat(surface, 2, axis=1)
    return psi2, enk2, v2, surface2


def _random_su2(rng):
    q, r = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
    q = q * (np.diag(r) / np.abs(np.diag(r)))[None, :]
    return q / np.sqrt(np.linalg.det(q))


def _head_pieces(mesh, face, psi_shape, enk, v, omega, *, nspinor, eta, cell_volume,
                 surface=None):
    nk, nb = enk.shape
    occ = _host(face.occ)
    common = dict(mesh=mesh, nb_logical=nb, nk_tot=nk, nspin=1, nspinor=nspinor)
    S = head_s_tensor_sharded(
        v, jnp.asarray(enk), jnp.asarray(occ), omega, cell_volume=cell_volume,
        eta_ry=eta, **common)
    Y, Z = head_wings_sharded(
        v, face, jnp.asarray(enk), jnp.asarray(occ), omega, eta_ry=eta, **common)
    static = None
    if surface is not None:
        static = static_head_wings_sharded(face, jnp.asarray(surface), **common)
    return S, Y, Z, static


def _assert_close(got, want, *, rtol=1e-10):
    got, want = _host(got), _host(want)
    scale = float(np.max(np.abs(want)))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-13 * max(scale, 1.0))


def test_spin_doubled_store_reproduces_the_scalar_charge_head():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260921)
    nk, nb, nmu = 2, 6, 8
    cell_volume, eta = 97.0, 0.01
    omega = np.asarray([0.1 + 0.0j, 0.3 + 0.02j, 0.0 + 0.4j])

    psi, enk, v, surface = _scalar_store(rng, nk=nk, nb=nb, nmu=nmu)
    psi2, enk2, v2, surface2 = _spin_doubled(psi, enk, v, surface)
    face1 = _face(mesh, psi, enk, nb // 2)
    face2 = _face(mesh, psi2, enk2, nb)
    # The builders derive occupations from the slices: the doubled store
    # occupies both spin copies of every occupied scalar band.
    np.testing.assert_array_equal(_host(face2.occ), np.repeat(_host(face1.occ), 2, axis=1))

    S1, Y1, Z1, C1 = _head_pieces(mesh, face1, psi.shape, enk, v, omega,
                                  nspinor=1, eta=eta, cell_volume=cell_volume,
                                  surface=surface)
    S2, Y2, Z2, C2 = _head_pieces(mesh, face2, psi2.shape, enk2, v2, omega,
                                  nspinor=2, eta=eta, cell_volume=cell_volume,
                                  surface=surface2)
    assert tuple(Y2.shape) == tuple(Y1.shape) == (len(omega), 3, nmu)
    assert tuple(Z2.shape) == tuple(Z1.shape) == (len(omega), nmu, 3)
    _assert_close(S2, S1)
    _assert_close(Y2, Y1)
    _assert_close(Z2, Z1)
    _assert_close(C2[0], C1[0])
    _assert_close(C2[1], C1[1])

    # One planted spin-scalar body serves both stores: the fold is one
    # contraction on the n_mu x n_mu charge basis.
    a = rng.standard_normal((len(omega), nmu, nmu)) + 1j * rng.standard_normal((len(omega), nmu, nmu))
    W = _put(a @ np.conj(np.swapaxes(a, -1, -2)) + 0.4 * np.eye(nmu), mesh, P(None, "x", "y"))
    S1_eff = fold_cartesian_head_wings_sharded(S1, Y1, W, Z1, cell_volume, mesh_xy=mesh)
    S2_eff = fold_cartesian_head_wings_sharded(S2, Y2, W, Z2, cell_volume, mesh_xy=mesh)
    _assert_close(S2_eff, S1_eff)
    # The fold moved the head: this is not a trivially satisfied identity.
    assert float(np.max(np.abs(_host(S1_eff) - _host(S1)))) > 1e-6 * float(np.max(np.abs(_host(S1))))


def test_global_spin_rotation_leaves_the_charge_head_invariant():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260922)
    nk, nb, nmu = 2, 6, 8
    cell_volume, eta = 97.0, 0.01
    omega = np.asarray([0.1 + 0.0j, 0.3 + 0.02j])

    psi = (rng.standard_normal((nk, nb, 2, nmu))
           + 1j * rng.standard_normal((nk, nb, 2, nmu)))
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    v = (rng.standard_normal((3, nk, nb, nb))
         + 1j * rng.standard_normal((3, nk, nb, nb)))
    surface = rng.uniform(0.1, 1.0, size=(nk, nb))
    U = _random_su2(rng)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-13)
    psi_rot = np.einsum("st,kntm->knsm", U, psi)

    face = _face(mesh, psi, enk, nb // 2)
    face_rot = _face(mesh, psi_rot, enk, nb // 2)
    _, Y, Z, C = _head_pieces(mesh, face, psi.shape, enk, v, omega,
                              nspinor=2, eta=eta, cell_volume=cell_volume, surface=surface)
    _, Y_rot, Z_rot, C_rot = _head_pieces(mesh, face_rot, psi.shape, enk, v, omega,
                                          nspinor=2, eta=eta, cell_volume=cell_volume,
                                          surface=surface)
    _assert_close(Y_rot, Y)
    _assert_close(Z_rot, Z)
    _assert_close(C_rot[0], C[0])
    _assert_close(C_rot[1], C[1])
    # A packed (mu, spin) endpoint would have moved: the rotation is not the identity
    # on any spinor component.
    assert float(np.max(np.abs(psi_rot - psi))) > 0.1
