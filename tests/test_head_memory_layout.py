"""Rank-local layouts of the zeta-fit pad diagonal, the LR+RL completion and the head wings.

Three memory-layout changes (the two whole-array OOMs of 2026-09-21) are pinned
against the expressions they replaced, at sizes that exercise the sharding:

* ``add_pad_diagonal_sharded`` against the former replicated
  ``C + (tr C / n) diag(~active)`` on a ``(nq, n, n)`` carrier at
  ``P(None, 'x', 'y')``;
* ``complete_ordered_pair_normal_equations`` on a sharded carrier against the
  jitted global ``N + conj(N[q_neg])`` (bit-identical: the permutation acts on
  the replicated q axis and the conjugate is elementwise);
* ``head_wings_sharded`` against the previous four-operand contraction written
  out in NumPy, at a centroid extent that needs more than one 64-wide block
  per rank and a frequency count that needs more than one frequency block.

Every cell builds the 2x2 mesh the production kernels shard on
(``@pytest.mark.mesh(4)``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import gw.qsgw_head as qsgw_head  # noqa: E402
from gw.isdf_fitting import add_pad_diagonal_sharded  # noqa: E402
from gw.qsgw_head import head_wings_sharded  # noqa: E402
from gw.wavefunction_bundle import BandSlices, build_wavefunctions_face  # noqa: E402
from isdf.core import (  # noqa: E402
    _ordered_pair_normal_equations,
    complete_ordered_pair_normal_equations,
)

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


def _complex(rng, shape, scale=1.0):
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def test_pad_diagonal_sharded_matches_the_replicated_expression():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260921)
    nq, n, n_logical = 3, 16, 13
    a = _complex(rng, (nq, n, n))
    C = a @ np.conj(np.swapaxes(a, -1, -2))
    active = np.ones(n, dtype=bool)
    active[[2, 9, 15]] = False                      # interleaved pad slots on both shards
    C[:, ~active, :] = 0.0
    C[:, :, ~active] = 0.0
    scale = np.trace(C, axis1=-2, axis2=-1).real / n_logical
    expected = C + scale[:, None, None] * np.diag(~active).astype(np.complex128)[None]
    got = add_pad_diagonal_sharded(_put(C, mesh, P(None, "x", "y")), active, float(n_logical),
                                   mesh_xy=mesh)
    assert got.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    np.testing.assert_allclose(_host(got), expected, rtol=1e-13, atol=1e-13 * np.max(np.abs(expected)))
    # The pad diagonal really carries the mean diagonal, and only there.
    diff = _host(got) - C
    assert np.allclose(np.diagonal(diff, axis1=-2, axis2=-1)[:, ~active], scale[:, None])
    assert np.allclose(diff[:, active, :], 0.0) and np.allclose(diff[:, :, active], 0.0)


def test_ordered_completion_on_a_sharded_carrier_is_the_global_sum():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260922)
    nq, n = 8, 12
    N = _complex(rng, (nq, n, n))
    neg = np.array([0, 7, 6, 5, 4, 3, 2, 1])          # q -> -q on an 8-point line
    reference = _host(_ordered_pair_normal_equations(jnp.asarray(N), jnp.asarray(neg)))
    got = complete_ordered_pair_normal_equations(_put(N, mesh, P(None, "x", "y")), neg)
    assert got.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    np.testing.assert_array_equal(_host(got), reference)
    np.testing.assert_allclose(reference, N + np.conj(N[neg]), rtol=0, atol=0)


def _face(mesh, psi, enk, occ_cut):
    nk, nb, _ns, _nmu = psi.shape
    slices = BandSlices.from_band_edges(0, 0, occ_cut, nb, nb)
    y_in = _put(psi, mesh, P(None, None, None, "y"))
    x_in = _put(np.conj(psi).transpose(0, 3, 1, 2), mesh, P(None, "x", None, None))
    return build_wavefunctions_face(
        y_in, x_in, enk_full=_put(enk, mesh, P(None, None)), slices=slices, mesh_xy=mesh)


def _four_operand_wings(v, enk, occ, psi, omegas, *, nk_tot, nspin, nspinor, eta):
    """The contraction the kernel used before 0094b6b3, written out in NumPy."""
    nk, nb = enk.shape
    pref = -4.0 / (float(nk_tot) * max(nspin, 1) * max(nspinor, 1))
    dE = enk[:, :, None] - enk[:, None, :]
    fdiff = occ[:, None, :] - occ[:, :, None]
    Y, Z = [], []
    for om in omegas:
        z = om + 1j * eta
        denom = z * z - dE * dE
        w = np.where((dE > 0) & (np.abs(denom) > 1e-16), pref * fdiff / (denom + 0j), 0.0)
        Y.append(np.einsum("akij,kij,kism,kjsm->am", np.conj(v), w, np.conj(psi), psi, optimize=False))
        Z.append(np.einsum("kism,kjsm,kij,bkij->mb", psi, np.conj(psi), w, v, optimize=False))
    return np.stack(Y), np.stack(Z)


def test_wing_contraction_matches_the_four_operand_expression_across_blocks(monkeypatch):
    mesh = _mesh_xy()
    # Two 64-wide centroid blocks per rank (70 local of 140) and three frequency
    # blocks of the production width 2 (five frequencies).
    monkeypatch.setattr(qsgw_head, "_HEAD_WING_MU_BLOCK", 64)
    monkeypatch.setattr(qsgw_head, "_HEAD_WING_FREQUENCY_BLOCK", 2)
    qsgw_head._KERNEL_CACHE.clear()
    rng = np.random.default_rng(20260923)
    nk, nb, ns, nmu = 3, 8, 2, 140
    psi = _complex(rng, (nk, nb, ns, nmu))
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    v = _complex(rng, (3, nk, nb, nb))
    omegas = np.asarray([0.1, 0.3 + 0.02j, 0.0 + 0.4j, 0.7 - 0.05j, 1.1 + 0.1j])
    eta = 0.01
    face = _face(mesh, psi, enk, nb // 2)
    occ = _host(face.occ)
    Y, Z = head_wings_sharded(
        v, face, jnp.asarray(enk), jnp.asarray(occ), omegas, mesh=mesh, nb_logical=nb,
        nk_tot=nk, nspin=1, nspinor=ns, eta_ry=eta)
    Y_ref, Z_ref = _four_operand_wings(v, enk, occ, psi, omegas, nk_tot=nk, nspin=1,
                                       nspinor=ns, eta=eta)
    assert tuple(Y.shape) == (len(omegas), 3, nmu) and tuple(Z.shape) == (len(omegas), nmu, 3)
    scale = max(float(np.max(np.abs(Y_ref))), float(np.max(np.abs(Z_ref))))
    np.testing.assert_allclose(_host(Y), Y_ref, rtol=1e-10, atol=1e-12 * scale)
    np.testing.assert_allclose(_host(Z), Z_ref, rtol=1e-10, atol=1e-12 * scale)
