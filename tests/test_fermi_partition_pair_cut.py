"""Fermi energy partition (gw.efermi.fermi_energy_partition) and the pair kernel's window cut.

The partition splits (k, band) states by the sign of e - mu and derives the window w = 10 widths;
the ordered-pair kernel skips the tile block whose bands all lie above mu + w at every k.  On a
clamped MP1 table that block's weights are exactly zero, so the cut is BITWISE inert; on a
Fermi-Dirac table it moves the result by less than the partition's truncation bound.  CPU only.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.efermi import FERMI_WINDOW_WIDTHS, fermi_energy_partition, fd_occupations, mp1_occupations  # noqa: E402

jax.config.update("jax_enable_x64", True)

NK, NB, NMU, TILE, WIDTH, MU = 4, 24, 5, 4, 0.02, 0.0


def _bands(rng):
    # Sorted energies per k: a metal-like band bottom crossing mu and a wide empty manifold above.
    e = np.sort(rng.uniform(-0.4, 1.5, size=(NK, NB)), axis=1)
    e[:, 0] = -0.6
    return e


def _scan(psi_mun, psi_nmu, e, f, s, z, band_cut):
    from common.shard_map import shard_map
    from common.wfn_layout import psi_specs
    from gw.w_isdf import _fractional_pair_scan_face
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    PSI_NMU_SPEC, PSI_MUN_SPEC = psi_specs("face")
    kmq = np.roll(np.arange(NK), 1).astype(np.int32)

    def _local(pm, pn, e_, f_, s_, z_):
        return _fractional_pair_scan_face(
            pm, pn, jnp.take(pm, kmq, axis=0), jnp.take(pn, kmq, axis=0), e_, jnp.take(e_, kmq, axis=0),
            f_, jnp.take(f_, kmq, axis=0), s_, jnp.take(s_, kmq, axis=0), z_,
            nb_full=NB, nb_logical=NB, tile=TILE, band_cut=band_cut)
    kern = jax.jit(shard_map(_local, mesh=mesh,
                             in_specs=(PSI_MUN_SPEC, PSI_NMU_SPEC, P(None, None), P(None, None), P(None, None), P(None)),
                             out_specs=P(None, "x", "y"), check_vma=False))
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(jnp.asarray(a), rep)
    return np.asarray(kern(put(psi_mun), put(psi_nmu), put(e), put(f), put(s), put(z)))


def _operands(rng):
    psi = rng.standard_normal((NK, NB, NMU)) + 1j * rng.standard_normal((NK, NB, NMU))
    psi_mun = psi.transpose(0, 2, 1)[:, None, :, :]      # (nk, s, mu, nb)
    psi_nmu = psi[:, :, None, :]                          # (nk, nb, s, mu)
    return psi_mun, psi_nmu


def test_partition_invariants():
    e = _bands(np.random.default_rng(1))
    f = np.asarray(fd_occupations(e, MU, WIDTH))
    part = fermi_energy_partition(e, f, mu_ry=MU, width_ry=WIDTH, family="fd")
    assert part.window_ry == pytest.approx(FERMI_WINDOW_WIDTHS * WIDTH)
    assert np.array_equal(part.below, e < MU) and np.array_equal(part.near, np.abs(e - MU) < part.window_ry)
    # Every band at or above the cut is above mu + w at every k; the band just below the cut is not, somewhere.
    assert np.all(e[:, part.band_cut_hi:] > MU + part.window_ry)
    assert part.band_cut_hi == 0 or np.any(e[:, part.band_cut_hi - 1] <= MU + part.window_ry)
    assert np.all(e[:, :part.band_cut_lo] < MU - part.window_ry)
    # The four factor tables: A, A' over states above mu; B, B' below; f + (1 - f) = 1 on each side.
    assert np.allclose(part.upper_weights.sum(axis=0), (~part.below).astype(float))
    assert np.allclose(part.lower_weights.sum(axis=0), part.below.astype(float))
    assert part.truncation_bound == pytest.approx(np.exp(-FERMI_WINDOW_WIDTHS))
    with pytest.raises(ValueError, match="fermi_partition_family"):
        fermi_energy_partition(e, f, mu_ry=MU, width_ry=0.0, family="fixed")


def test_pair_cut_bitwise_on_clamped_mp1():
    rng = np.random.default_rng(2)
    e = _bands(rng)
    f = np.asarray(mp1_occupations(e, MU, WIDTH))
    s = np.zeros_like(e)
    part = fermi_energy_partition(e, f, mu_ry=MU, width_ry=WIDTH, family="mp1")
    assert 0 < part.band_cut_hi < NB and part.truncation_bound == 0.0
    # The dropped block is exactly zero on the table: every band above the cut has f == 0 exactly.
    assert np.all(f[:, part.band_cut_hi:] == 0.0)
    z = np.asarray([0.31j, 0.2 + 0.31j])
    psi_mun, psi_nmu = _operands(rng)
    full = _scan(psi_mun, psi_nmu, e, f, s, z, None)
    cut = _scan(psi_mun, psi_nmu, e, f, s, z, part.band_cut_hi)
    assert np.array_equal(full, cut), "the window cut must be bit-for-bit inert on a clamped MP1 table"


def test_pair_cut_within_truncation_bound_on_fd():
    rng = np.random.default_rng(3)
    e = _bands(rng)
    f = np.asarray(fd_occupations(e, MU, WIDTH))
    s = np.zeros_like(e)
    part = fermi_energy_partition(e, f, mu_ry=MU, width_ry=WIDTH, family="fd")
    z = np.asarray([0.31j])
    psi_mun, psi_nmu = _operands(rng)
    full = _scan(psi_mun, psi_nmu, e, f, s, z, None)
    cut = _scan(psi_mun, psi_nmu, e, f, s, z, part.band_cut_hi)
    rel = np.linalg.norm(full - cut) / np.linalg.norm(full)
    assert 0.0 < rel < part.truncation_bound, (rel, part.truncation_bound)
