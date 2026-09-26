"""SlabIO writers for ``dipole.h5`` and ``kin_ion.h5`` (ARCH H4).

Both preprocessing artifacts are written from the sweep's band shards; no
rank gathers the table.  The cells here write a padded, sharded operand on an
emulated 2x2 CPU mesh (SlabIO's serial tier), read the file back with h5py,
and compare against the NumPy truth: values, the dropped pad rows, the
attributes, and ΔE derived on read against the expression the writer used to
store.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _emulated_mesh():
    import jax
    from jax.sharding import Mesh

    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("needs 4 cpu devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))


def _sharded(host, mesh, spec):
    import jax
    from jax.sharding import NamedSharding
    return jax.device_put(host, NamedSharding(mesh, spec))


def _random(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_band_energies_follow_the_star_parent_and_refuse_a_short_map():
    from file_io.dipole import band_energies_on_full_bz

    energies = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    wfn = SimpleNamespace(energies=energies)
    sym = SimpleNamespace(nk_tot=4, irr_idx_k=np.array([0, 2, 2, 1]))
    got = band_energies_on_full_bz(wfn, sym, 4)
    np.testing.assert_array_equal(got, energies[0][[0, 2, 2, 1], :4])
    # Red twin: the fallback this replaces paired k with its own index.
    short = SimpleNamespace(nk_tot=4, irr_idx_k=np.array([0, 2, 2]))
    with pytest.raises(ValueError, match="GATE dipole_energy_star_map"):
        band_energies_on_full_bz(wfn, short, 4)


def test_delta_e_is_derived_bitwise_and_reads_a_legacy_table(tmp_path):
    from file_io.dipole import delta_e, delta_e_cv

    e = np.random.default_rng(3).standard_normal((4, 6))
    stored = np.zeros((4, 6, 6))
    for k in range(4):                     # the writer's old expression
        stored[k] = e[k][:, None] - e[k][None, :]
    new, old = tmp_path / "new.h5", tmp_path / "old.h5"
    with h5py.File(new, "w") as h5:
        h5.create_dataset("band_energies", data=e)
    with h5py.File(old, "w") as h5:
        h5.create_dataset("deltaE", data=stored)
    with h5py.File(new, "r") as h5:
        assert np.array_equal(delta_e(h5), stored)
        assert np.array_equal(delta_e_cv(h5, nv=2, nc=3), stored[:, 2:5, :2])
    with h5py.File(old, "r") as h5:
        assert np.array_equal(delta_e(h5), stored)
        assert delta_e_cv(h5, nv=2, nc=3) is None


@pytest.mark.mesh(4)
def test_write_dipole_round_trips_from_shards(tmp_path):
    from jax.sharding import PartitionSpec as P
    from file_io.dipole import delta_e, finite_q_payload, write_dipole

    mesh = _emulated_mesh()
    nk, nb, nb_pad = 3, 5, 6                       # one pad row per band axis
    v = _random((nk, 3, nb_pad, nb_pad), 1)
    v[..., nb:, :] = 0.0
    v[..., :, nb:] = 0.0
    e = np.random.default_rng(2).standard_normal((nk, nb))
    fq = finite_q_payload(
        rho_cvkq=_random((2, 3, nk, 2), 4), v_cvkq=_random((3, 2, 3, nk, 2), 5),
        kminq_idx=np.zeros((nk, 2), dtype=np.int64), iq_list=[0, 1],
        n_occ=3, v_lo=0, c_hi=5)
    path = tmp_path / "dipole.h5"
    write_dipole(path, _sharded(v, mesh, P(None, None, "x", "y")), e,
                 mesh=mesh, attrs={"nbands": 7, "nk": nk, "note": "n"},
                 finite_q=fq)
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(
            h5["dipole_cart"][...], np.moveaxis(v, 1, 0)[..., :nb, :nb])
        np.testing.assert_array_equal(h5["band_energies"][...], e)
        assert "deltaE" not in h5
        assert np.array_equal(delta_e(h5), e[:, :, None] - e[:, None, :])
        assert int(h5.attrs["nbands"]) == 7 and h5.attrs["note"] == "n"
        np.testing.assert_array_equal(h5["finite_q/iq_list"][...], [0, 1])
        assert int(h5["finite_q"].attrs["c_hi"]) == 5


@pytest.mark.mesh(4)
def test_write_kin_ion_round_trips_and_the_check_reads_blocks(tmp_path):
    from jax.sharding import PartitionSpec as P
    from file_io.kin_ion import K_STORAGE_ATTR, K_STORAGE_IBZ, write_kin_ion
    from psp.operator_checks import check_degeneracy_consistency

    mesh = _emulated_mesh()
    n_rows, nb, nb_pad = 2, 5, 6
    h = _random((n_rows, nb_pad, nb_pad), 7)
    h = h + np.conj(np.swapaxes(h, -1, -2))
    h[:, nb:, :] = 0.0
    h[:, :, nb:] = 0.0
    star = (np.array([0, 1, 1, 0], dtype=np.int32),
            np.array([0, 1, 2, 0], dtype=np.int32), 2)
    path = tmp_path / "kin_ion.h5"
    write_kin_ion(path, _sharded(h, mesh, P(None, "x", "y")), mesh=mesh,
                  nb=nb, star=star, attrs={"nk": 4, "nrk": n_rows})
    en = np.repeat(np.array([[0.0, 0.0, 1.0, 1.0, 2.0]]), n_rows, axis=0)
    with h5py.File(path, "r") as h5:
        ds = h5["kin_ion"]
        np.testing.assert_array_equal(ds[...], h[:, :nb, :nb])
        assert ds.attrs[K_STORAGE_ATTR] == K_STORAGE_IBZ
        assert int(ds.attrs["n_sym_spatial"]) == 2
        np.testing.assert_array_equal(h5["irr_idx_k"][...], star[0])
        np.testing.assert_array_equal(h5["sym_idx_k"][...], star[1])
        lazy = check_degeneracy_consistency(ds, en, print_fn=lambda *_: None)
    eager = check_degeneracy_consistency(h[:, :nb, :nb], en,
                                         print_fn=lambda *_: None)
    assert lazy == eager
