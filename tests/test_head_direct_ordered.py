"""The direct head ``S_ab(z)`` needs no time-reversal assumption.

Owner statement: the ``gw.shared_pole_head`` module docstring.  The exact
long-wavelength charge response is the Lehmann sum over ALL ordered band
pairs at every ``k``,

    chi_00(q -> 0, z) = q_a q_b (pref/2) sum_k sum_{n != m}
        (f_n - f_m) conj(v^a_nm) v^b_nm / (D_mn^2 (z + i eta - D_mn)),
    D_mn = e_m - e_n,  pref = 4 / (Omega N_k n_spin n_spinor),

with no pairing of ``k`` with ``-k``.  ``head_s_tensor_sharded`` sums each
unordered pair once with the combined denominator ``D [(z+i eta)^2 - D^2]``;
combining the ``(n, m)`` and ``(m, n)`` terms at ONE ``k`` gives

    (f_j - f_i)/D^2 [conj(T)/(z - D) - T/(z + D)]
        = 2 (f_j - f_i) [Re T - i z Im T / D] / (D (z^2 - D^2)),

so the signed occupation difference, the energy-ordered pair sum, the
conjugation ``conj(v^a) v^b`` and the denominator all survive without time
reversal; the kernel's only departure from the Lehmann sum is the
antisymmetric ``Im T_ab`` part (zero after a time-reversal-paired k sum,
nonzero on a magnet), which the mini-BZ quadratic form ``q.S.q`` annihilates
whatever weight it carries.  These cells pin exactly that:

* TR-broken velocities: ``q.S.q`` equals the Lehmann oracle for random real
  ``q``; ``S(-z) = S(z)``; the kernel-minus-oracle difference is purely
  antisymmetric in ``(a, b)``.
* TR-paired velocities (``v(-k) = -conj v(k)``, the zero-magnetisation
  limit): the kernel equals the oracle as a full 3x3 and is symmetric.
* Magnetisation to zero: the antisymmetric part of the kernel's tensor
  vanishes linearly with the time-odd perturbation while ``q.S.q`` stays
  exact at every step.
* Fractional (metallic) occupations: the same identities with signed
  ``f_n - f_m`` in ``(0, 1)``.

Every cell builds the 2x2 mesh the production kernel shards on
(``@pytest.mark.mesh(4)``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh  # noqa: E402

from gw.qsgw_head import head_s_tensor_sharded  # noqa: E402

pytestmark = pytest.mark.mesh(4)

CELL_VOLUME, NSPIN, NSPINOR, ETA = 97.0, 1, 2, 0.01
OMEGAS = np.asarray([0.1 + 0.0j, 0.3 + 0.02j, 0.0 + 0.4j, 0.7 - 0.05j])


def _mesh_xy():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(f"needs 4 devices for the 2x2 mesh, have {len(devices)}")
    return Mesh(np.asarray(devices[:4], dtype=object).reshape(2, 2), ("x", "y"))


def _hermitian_velocity(rng, nk, nb):
    v = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal((3, nk, nb, nb))
    return 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))


def _time_reversal_pair(v, enk, occ):
    """Append the ``-k`` partner of every k: ``v(-k) = -conj v(k)``, same energies."""
    return (np.concatenate([v, -np.conj(v)], axis=1),
            np.concatenate([enk, enk], axis=0),
            np.concatenate([occ, occ], axis=0))


def _lehmann_oracle(v, enk, occ, omegas):
    """Independent O(nk nb^2) sum over ALL ordered pairs; no pair combination."""
    nk, nb = enk.shape
    pref = 4.0 / (CELL_VOLUME * nk * NSPIN * NSPINOR)
    S = np.zeros((len(omegas), 3, 3), dtype=np.complex128)
    for iw, om in enumerate(omegas):
        z = om + 1j * ETA
        for k in range(nk):
            for n in range(nb):
                for m in range(nb):
                    if n == m:
                        continue
                    d = enk[k, m] - enk[k, n]
                    if abs(d) < 1e-12:
                        continue
                    weight = (occ[k, n] - occ[k, m]) / (d * d * (z - d))
                    S[iw] += 0.5 * pref * weight * np.outer(np.conj(v[:, k, n, m]), v[:, k, n, m])
    return S


def _kernel(mesh, v, enk, occ, omegas):
    return np.asarray(jax.device_get(head_s_tensor_sharded(
        v, jnp.asarray(enk), jnp.asarray(occ), omegas, mesh=mesh,
        nb_logical=enk.shape[1], cell_volume=CELL_VOLUME, nk_tot=enk.shape[0],
        nspin=NSPIN, nspinor=NSPINOR, eta_ry=ETA)))


def _quadratic_forms(q, S):
    """``q.S.q`` for a fixed set of real directions ``q`` (count, 3)."""
    return np.einsum("qa,wab,qb->wq", q, S, q)


def _scale(S):
    return max(float(np.max(np.abs(S))), 1.0e-30)


def _data(rng, nk, nb, *, fractional):
    v = _hermitian_velocity(rng, nk, nb)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    if fractional:
        occ = np.sort(rng.uniform(0.0, 1.0, size=(nk, nb)), axis=1)[:, ::-1].copy()
    else:
        occ = np.where(np.arange(nb)[None, :] < nb // 2, 1.0, 0.0) * np.ones((nk, 1))
    return v, enk, occ


@pytest.mark.parametrize("fractional", [False, True])
def test_direct_head_matches_the_lehmann_sum_without_time_reversal(fractional):
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260921 + int(fractional))
    v, enk, occ = _data(rng, nk=3, nb=6, fractional=fractional)
    S = _kernel(mesh, v, enk, occ, OMEGAS)
    S_ref = _lehmann_oracle(v, enk, occ, OMEGAS)
    scale = _scale(S_ref)
    q = rng.standard_normal((6, 3))
    # The observable: every quadratic form agrees.
    np.testing.assert_allclose(_quadratic_forms(q, S), _quadratic_forms(q, S_ref),
                               rtol=1e-10, atol=1e-12 * scale)
    # The only departure is antisymmetric in (a, b) ...
    diff = S - S_ref
    np.testing.assert_allclose(diff + np.swapaxes(diff, -1, -2), 0.0, atol=1e-12 * scale)
    # ... and on a magnet it is not zero (the test discriminates).
    assert float(np.max(np.abs(diff))) > 1e-6 * scale
    # Even in z: S is a function of z^2 only.
    S_minus = _kernel(mesh, v, enk, occ, -OMEGAS - 2j * ETA)   # z -> -z with z = omega + i eta
    np.testing.assert_allclose(S_minus, S, rtol=1e-10, atol=1e-12 * scale)


def test_time_reversal_paired_data_is_the_zero_magnetisation_limit():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260922)
    v1, enk1, occ1 = _data(rng, nk=2, nb=6, fractional=False)
    v, enk, occ = _time_reversal_pair(v1, enk1, occ1)
    S = _kernel(mesh, v, enk, occ, OMEGAS)
    S_ref = _lehmann_oracle(v, enk, occ, OMEGAS)
    scale = _scale(S_ref)
    # Under time reversal the k and -k Lehmann terms cancel Im T, so the
    # kernel's tensor is the full Lehmann tensor and it is symmetric.
    np.testing.assert_allclose(S, S_ref, rtol=1e-10, atol=1e-12 * scale)
    np.testing.assert_allclose(S - np.swapaxes(S, -1, -2), 0.0, atol=1e-12 * scale)


def test_antisymmetric_part_vanishes_linearly_with_the_magnetisation():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260923)
    v1, enk1, occ1 = _data(rng, nk=2, nb=6, fractional=True)
    v_even, enk, occ = _time_reversal_pair(v1, enk1, occ1)
    # A time-odd perturbation: the same Hermitian table at k and -k, which
    # violates v(-k) = -conj v(k) and models a magnetisation.
    odd = _hermitian_velocity(rng, 2, 6)
    v_odd = np.concatenate([odd, odd], axis=1)
    S_even = _kernel(mesh, v_even, enk, occ, OMEGAS)
    scale = _scale(S_even)
    q = rng.standard_normal((6, 3))
    antisym = []
    for eps in (1e-1, 1e-2, 1e-3):
        v = v_even + eps * v_odd
        S = _kernel(mesh, v, enk, occ, OMEGAS)
        S_ref = _lehmann_oracle(v, enk, occ, OMEGAS)
        np.testing.assert_allclose(_quadratic_forms(q, S), _quadratic_forms(q, S_ref),
                                   rtol=1e-10, atol=1e-12 * scale)
        antisym.append(float(np.max(np.abs(S - np.swapaxes(S, -1, -2)))))
        np.testing.assert_allclose(S, S_even, atol=8.0 * eps * scale)
    # Linear in the time-odd amplitude: each decade of eps drops it a decade
    # (the odd part is bilinear in v, so the cross term dominates).
    assert antisym[0] > 1e-3 * scale
    assert antisym[1] < 0.2 * antisym[0]
    assert antisym[2] < 0.2 * antisym[1]
