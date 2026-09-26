"""The metallic q->0 head: Drude prefactor and the fixed-N occupation state.

Two questions.

1. **Prefactor.**  For free electrons (E = k^2 Ry, v = 2k Ry bohr) the
   head's Fermi-surface tensor must be D = 2n (S convention), i.e.
   omega_p^2 = 8 pi D = 16 pi n Ry^2 = 4 pi n e^2 / m, and the Thomas-Fermi
   screening 8 pi N(E_F) / Omega must be 4 k_F / pi.  The production owners
   (:func:`gw.fermi_surface.metal_head_surface_weights` and
   :func:`gw.qsgw_head.head_drude_tensor_sharded`) are evaluated on a Fermi
   sphere inside a cubic zone; the linear-tetrahedron error is O(h^2)
   (0.971, 0.988, 0.9986 of the exact value at 12^3, 16^3, 20^3).

2. **Occupations.**  A metal's head must take the fixed-N Fermi-Dirac state,
   not the bundle's 0/1 step by band index.  The step splits a degenerate
   multiplet at the cut; on Na 8^3 (a pair split by 8e-15 Ry, 2.2 eV above
   mu) that made the head W = 0 at every frequency.  The fixture plants the
   same split: the step table reproduces the divergence (negative control)
   and :func:`gw.qsgw_head.build_dft_head_response` with the state does not.
"""

from __future__ import annotations

import itertools
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw import qsgw_head
from gw.efermi import OccupationState
from gw.fermi_surface import metal_head_surface_weights
from gw.qsgw_head import head_drude_tensor_sharded, head_s_tensor_sharded

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        return Mesh(devices[:4].reshape(2, 2), ("x", "y"))
    return Mesh(devices[:1].reshape(1, 1), ("x", "y"))


def _cubic_ops():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.int64)
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            ops.append(m)
    return np.asarray(ops)


def _star_labels(idx, grid, ops):
    key = {tuple(r): i for i, r in enumerate(np.mod(idx, grid))}
    label = np.full(len(idx), -1, dtype=np.int64)
    nstar = 0
    for i, r in enumerate(np.mod(idx, grid)):
        if label[i] >= 0:
            continue
        for m in ops:
            label[key[tuple(np.mod(m @ r, grid))]] = nstar
        nstar += 1
    return label


def _cubic_sym(n):
    grid = np.asarray((n, n, n))
    idx = np.asarray(list(np.ndindex(n, n, n)), dtype=np.int64)
    ops = _cubic_ops()
    return idx, SimpleNamespace(
        unfolded_kpts=idx / grid, sym_mats_k=ops,
        irr_idx_k=_star_labels(idx, grid, ops))


@pytest.mark.parametrize("nspinor", (1, 2))
def test_free_electron_drude_tensor_and_thomas_fermi(nspinor):
    n, a = 16, 6.0
    idx, sym = _cubic_sym(n)
    frac = idx / n
    frac = np.where(frac >= 0.5, frac - 1.0, frac)     # Fermi sphere inside
    k = frac * (2.0 * np.pi / a)
    energy = np.sum(k * k, axis=1)[:, None]
    velocity = (2.0 * k).T[:, :, None, None]           # (3, nk, 1, 1)
    if nspinor == 2:                                  # two spin states per k
        energy = np.repeat(energy, 2, axis=1)
        v = np.zeros((3, n ** 3, 2, 2))
        v[:, :, 0, 0] = v[:, :, 1, 1] = velocity[:, :, 0, 0]
        velocity = v
    kf = 0.3 * 2.0 * np.pi / a
    surface = metal_head_surface_weights(energy, kf * kf, sym=sym, kgrid=(n, n, n))
    volume = a ** 3
    drude = np.asarray(head_drude_tensor_sharded(
        jnp.asarray(velocity, dtype=jnp.complex128), jnp.asarray(surface),
        mesh=_mesh(), nb_logical=energy.shape[1], cell_volume=volume,
        nk_tot=n ** 3, nspin=1, nspinor=nspinor))
    density = kf ** 3 / (3.0 * np.pi ** 2)
    np.testing.assert_allclose(np.diag(drude.real) / (2.0 * density), 1.0,
                               atol=0.015)
    assert np.max(np.abs(drude - np.diag(np.diag(drude)))) < 1e-12 * density
    capacity = 2.0 / nspinor
    kappa2 = 8.0 * np.pi * capacity * surface.sum() / n ** 3 / volume
    assert kappa2 / (4.0 * kf / np.pi) == pytest.approx(1.0, abs=0.015)


def _split_pair_fixture():
    """Band 0 crosses mu; band 1 touches it at Gamma, split by 1e-14 Ry."""
    n, a = 6, 6.0
    idx, sym = _cubic_sym(n)
    frac = idx / n
    frac = np.where(frac >= 0.5, frac - 1.0, frac)
    k = frac * (2.0 * np.pi / a)
    k2 = np.sum(k * k, axis=1)
    e0 = k2 - 0.4
    e1 = e0 + 0.6 * k2 + 1.0e-14
    energies = np.stack((e0, e1, e0 + 3.0), axis=1)
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(3, n ** 3, 3, 3)) + 1j * rng.normal(
        size=(3, n ** 3, 3, 3))
    velocity = 0.5 * (raw + np.swapaxes(raw.conj(), -1, -2))
    state = OccupationState.solve_smearing(
        energies, np.full(n ** 3, 1.0 / n ** 3), 2.0, 0.02,
        state_capacity=2.0, family="fd", logical_nband=3)
    step = np.zeros_like(energies)
    step[:, 0] = 1.0                                   # 2 electrons = 1 band
    return SimpleNamespace(n=n, a=a, sym=sym, energies=energies,
                           velocity=velocity, state=state, step=step)


def test_step_by_band_index_splits_a_degenerate_pair_and_diverges():
    """Negative control: the table the one-shot/off heads used to take."""
    fx = _split_pair_fixture()
    common = dict(mesh=_mesh(), nb_logical=3, cell_volume=fx.a ** 3,
                  nk_tot=fx.n ** 3, nspin=1, nspinor=1)
    z = np.asarray([0.2j, 0.5 + 0.2j])
    s_step = np.asarray(head_s_tensor_sharded(
        jnp.asarray(fx.velocity), jnp.asarray(fx.energies),
        jnp.asarray(fx.step), z, **common))
    s_fd = np.asarray(head_s_tensor_sharded(
        jnp.asarray(fx.velocity), jnp.asarray(fx.energies),
        jnp.asarray(np.asarray(fx.state.f_kn)), z, **common))
    assert np.max(np.abs(s_step)) > 1.0e6
    assert np.max(np.abs(s_fd)) < 1.0e3


def test_dft_head_takes_the_fixed_n_state_and_its_drude_term(monkeypatch, tmp_path):
    fx = _split_pair_fixture()
    (tmp_path / "dipole.h5").write_bytes(b"")
    monkeypatch.setattr(qsgw_head, "read_authenticated_dipole_velocity",
                        lambda *a, **k: fx.velocity)
    wfn = SimpleNamespace(nspin=1, kgrid=(fx.n,) * 3, symmetry=lambda: fx.sym)
    meta = SimpleNamespace(b_id_0=0, b_id_4_chi_user=3, nk_tot=fx.n ** 3,
                           cell_volume=fx.a ** 3, nspinor_wfnfile=1, nelec=1,
                           nb_sigma=3)
    config = SimpleNamespace(head=SimpleNamespace(wcoul0_eta=0.0))
    wfns = SimpleNamespace(enk=fx.energies, occ=fx.step)
    z = np.asarray([2e-5j, 0.2j, 0.5 + 0.2j])
    got = qsgw_head.build_dft_head_response(
        wfns, z, input_dir=str(tmp_path), mesh=_mesh(), wfn=wfn, meta=meta,
        config=config, wings=False, occupation_state=fx.state)

    surface = metal_head_surface_weights(
        fx.energies, fx.state.mu_ry, sym=fx.sym, kgrid=(fx.n,) * 3)
    common = dict(mesh=_mesh(), nb_logical=3, cell_volume=fx.a ** 3,
                  nk_tot=fx.n ** 3, nspin=1, nspinor=1)
    want = np.asarray(head_s_tensor_sharded(
        jnp.asarray(fx.velocity), jnp.asarray(fx.energies),
        jnp.asarray(np.asarray(fx.state.f_kn)), z,
        surface_weight_kn=jnp.asarray(surface), **common))
    np.testing.assert_allclose(np.asarray(got.S_direct), want, rtol=1e-12,
                               atol=1e-12)
    np.testing.assert_array_equal(
        got.sigma_occupations, np.asarray(fx.state.f_kn)[:, :3])
    assert got.efermi_ry == float(fx.state.mu_ry)
    assert np.max(np.abs(got.drude_tensor)) > 0.0
    kappa2 = 8.0 * np.pi * 2.0 * surface.sum() / fx.n ** 3 / fx.a ** 3
    assert got.static_kappa2_bohr2 == pytest.approx(kappa2, rel=1e-14)
    # The insulating call (no state) keeps the bundle table and no intraband.
    plain = qsgw_head.build_dft_head_response(
        wfns, z[1:], input_dir=str(tmp_path), mesh=_mesh(), wfn=wfn,
        meta=meta, config=config, wings=False)
    assert plain.drude_tensor is None and plain.static_kappa2_bohr2 is None
    assert np.max(np.abs(np.asarray(plain.S_direct))) > 1.0e6
