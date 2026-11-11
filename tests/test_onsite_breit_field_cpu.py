"""On-site Breit deck keys, field assembly and debug writer (``gw.gw_config``, ``gw.hartree``,
``file_io.onsite_breit_output``).

A CPU/unit cell, exempt from the four-GPU rule: no driver runs and no WFN file is
read.  ``gw.hartree.onsite_breit_field`` runs on a synthetic stand-in for the
loader (orthonormal two-component psi(G) on a CrCl3-like cell, the loader's
band-carrier pad reproduced) against a direct numpy evaluation of
``P``, ``D = sum_k w_k (P U) f (P U)^dagger`` and ``H_OS = P^dagger h P``; every
gate it adds is shown to fire.  On one CPU device the mesh is 1x1; with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` a 2x2 mesh also runs and
nb = 6 then exercises the carrier pad (8) and its strip.  Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_THREAD_TIMEOUT=1 \\
        python -m pytest tests/test_onsite_breit_field_cpu.py -q -p no:cacheprovider

NOT covered here (the P4 leg): the real ``WfnLoader.load(k="full_bz")`` unfold,
``padded_gvectors`` on a real WFN, ``gw.onsite_breit_cli`` end to end, the
smooth ``direct_field_matrices`` arm.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

jax.config.update("jax_enable_x64", True)

from gw import hartree  # noqa: E402
from gw.gw_config import LorraxConfig  # noqa: E402
from psp import onsite_breit as ob  # noqa: E402
from tests.test_onsite_breit import (DATA, SIDECARS, UPFS, _fake_wfn, _gtable,  # noqa: E402
                                     _meshes, _reference_P)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


# ---------------------------------------------------------------- deck keys
def _config(tmp_path, extra=""):
    path = tmp_path / "deck.in"
    path.write_text("[cohsex]\nsys_dim = 3\nnumber_bands = 30\nnval = 4\nncond = 4\n" + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False, runtime_platform="cpu",
                                        print_fn=lambda *_a, **_k: None)


ON = ("onsite_breit = true\nonsite_breit_files = Cr.onsite_breit.json\n")


def test_deck_key_refusals_and_controls(tmp_path):
    c = _config(tmp_path)                                                    # control: absent
    assert c.onsite_breit is False and c.onsite_breit_projector == "ortho-atomic"
    c = _config(tmp_path, ON + "onsite_breit_double_counting = ONSITE_ONLY\n"
                "onsite_breit_exchange = Full\n")                            # control: complete
    assert (c.onsite_breit, c.onsite_breit_double_counting, c.onsite_breit_exchange,
            c.onsite_breit_files, c.onsite_breit_pseudo_dir) == (
        True, "onsite_only", "full", "Cr.onsite_breit.json", "")
    refused = {
        "dependent key with the switch off": "onsite_breit_exchange = full\n",
        "switch off, explicit default named": "onsite_breit = false\nonsite_breit_projector = ortho-atomic\n",
        "no files": "onsite_breit = true\nonsite_breit_double_counting = onsite_only\n"
                    "onsite_breit_exchange = full\n",
        "double counting unset": ON + "onsite_breit_exchange = full\n",
        "exchange unset": ON + "onsite_breit_double_counting = onsite_only\n",
        "paw on a scalar deck": ON + "onsite_breit_double_counting = paw\nonsite_breit_exchange = full\n",
        "unknown value": ON + "onsite_breit_double_counting = onsite_only\nonsite_breit_exchange = on\n",
        "frozen density in SC": ON + "onsite_breit_double_counting = onsite_only\n"
                                "onsite_breit_exchange = full\nqp_solver = self_consistent\n"
                                "density_self_consistent = false\n",
    }
    for what, extra in refused.items():
        with pytest.raises(ValueError, match="GATE onsite_breit_keys") as err:
            _config(tmp_path, extra)
        assert "got:" in str(err.value) and "fix:" in str(err.value), what


def _fingerprinted(wfn, nk=1, nb=4):
    """A stand-in carries what common.parallel_transport.wfn_fingerprint reads."""
    wfn.energies = np.zeros((1, nk, nb))
    wfn.kpoints = np.zeros((nk, 3))
    wfn.nelec, wfn.nbands, wfn.path = 2, nb, None
    return wfn


def test_setup_for_refusals_cache_and_controls(tmp_path):
    def cfg(**kw):
        base = dict(onsite_breit=True, onsite_breit_files=os.path.basename(SIDECARS["Cr"]),
                    onsite_breit_double_counting="onsite_only", onsite_breit_exchange="full",
                    onsite_breit_projector="ortho-atomic", onsite_breit_pseudo_dir="", bispinor=False)
        return SimpleNamespace(**{**base, **kw})

    wfn = _fingerprinted(_fake_wfn())
    assert hartree.onsite_breit_setup_for(cfg(onsite_breit=False), wfn, input_dir=DATA,
                                          tt_exchange=False) is None
    s1 = hartree.onsite_breit_setup_for(cfg(), wfn, input_dir=DATA, tt_exchange=False)     # control
    s2 = hartree.onsite_breit_setup_for(cfg(), _fingerprinted(_fake_wfn()), input_dir=DATA,
                                        tt_exchange=False)
    assert s1 is s2 and s1.R_os == 10                       # one setup per value, not per object
    with pytest.raises(ValueError, match="GATE onsite_breit_keys"):          # paw without TT exchange
        hartree.onsite_breit_setup_for(cfg(onsite_breit_exchange="paw"), wfn, input_dir=DATA,
                                       tt_exchange=False)
    with pytest.raises(ValueError, match="GATE onsite_breit_keys"):          # full beside TT exchange
        hartree.onsite_breit_setup_for(cfg(), wfn, input_dir=DATA, tt_exchange=True)
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
        hartree.onsite_breit_setup_for(cfg(onsite_breit_files="absent.json"), wfn, input_dir=DATA,
                                       tt_exchange=False)
    (tmp_path / "upf").mkdir()                                                # a species without UPF
    os.symlink(UPFS["Cr"], tmp_path / "upf" / "Cr.upf")
    with pytest.raises(ValueError, match="GATE onsite_breit_projector"):
        hartree.onsite_breit_setup_for(cfg(onsite_breit_pseudo_dir=str(tmp_path / "upf")), wfn,
                                       input_dir=DATA, tt_exchange=False)


# ---------------------------------------------------------------- field assembly
class _Loader:
    """The loader surface onsite_breit_field reads, on synthetic data.

    ``load`` pads the band axis to the mesh carrier with zero bands and places the
    array at the requested spec, as ``WfnLoader.load`` does.
    """

    def __init__(self, psi, occ, kv, mesh, exact=False):
        base = _fake_wfn()
        self.__dict__.update(vars(base))
        self._psi, self._occ, self._mesh = psi, occ, mesh
        nk, nb = psi.shape[:2]
        self.energies = np.zeros((1, nk, nb))
        self.kpoints, self.nelec, self.nbands, self.path = kv, 3, nb, None
        self.occupation_state_capacity = 1.0
        self.physical_density_band_stop = int(occ.shape[1])
        self.occupations_are_exact_integer = exact
        self.loads = 0

    def physical_density_occupations(self, *, k, unit_as_none=False):
        assert k == "full_bz"
        return self._occ

    def load(self, *, bands, k, sharding, bispinor=False):
        from runtime.padding import padded_axis
        assert k == "full_bz" and not bispinor
        self.loads += 1
        win = self._psi[:, bands[0]:bands[1]]
        carrier = padded_axis(win.shape[1], self._mesh, name="test", spec=sharding, axis=1).carrier
        out = np.zeros((win.shape[0], carrier) + win.shape[2:], np.complex128)
        out[:, :win.shape[1]] = win
        return jax.device_put(out, NamedSharding(self._mesh, sharding))


def _orthonormal_psi(rng, nk, nb, ngk, ngkmax):
    psi = np.zeros((nk, nb, 2, ngkmax), np.complex128)
    for k in range(nk):
        A = rng.normal(size=(2 * ngk[k], nb)) + 1j * rng.normal(size=(2 * ngk[k], nb))
        Q = np.linalg.qr(A)[0]                                                 # (2 ngk, nb)
        psi[k, :, :, :ngk[k]] = Q.T.reshape(nb, 2, ngk[k])
    return psi


@pytest.fixture(scope="module")
def setup():
    return ob.build_onsite_setup(_fake_wfn(), UPFS, SIDECARS, projector="ortho-atomic",
                                 double_counting="onsite_only", exchange="full")


@pytest.fixture(scope="module")
def cell():
    wfn = _fake_wfn()
    B = wfn.blat * wfn.bvec
    rng = np.random.default_rng(23)
    nk, ngkmax, nb_file = 3, 90, 8
    kv = rng.uniform(-0.5, 0.5, size=(nk, 3))
    ngk = [70 + 7 * k for k in range(nk)]
    tabs = [_gtable(kv[k], ngk[k], ngkmax, B, k) for k in range(nk)]
    gv, gm = np.stack([t[0] for t in tabs]), np.stack([t[1] for t in tabs])
    psi = _orthonormal_psi(rng, nk, nb_file, ngk, ngkmax)
    occ = np.zeros((nk, nb_file))
    occ[:, :6] = rng.uniform(0.05, 1.0, size=(nk, 6))                         # nothing above band 6
    return SimpleNamespace(kv=kv, gv=gv, gm=gm, psi=psi, occ=occ, rng=rng, nk=nk)


def test_onsite_breit_field_against_numpy(setup, cell, monkeypatch):
    from psp import dft_operators
    monkeypatch.setattr(dft_operators, "padded_gvectors", lambda wfn, *, k: SimpleNamespace(
        gvecs=cell.gv, mask=cell.gm, kvecs=cell.kv, ngk=cell.gm.sum(1).astype(int)))
    sym = SimpleNamespace(nk_tot=cell.nk, unfolded_kpts=cell.kv)
    b0, b1 = 1, 7                                       # GLOBAL band range, b0 > 0 on purpose
    nb, w = b1 - b0, np.full(cell.nk, 1.0 / cell.nk)
    P_ref = _reference_P(cell.psi[:, b0:b1], cell.gv, cell.gm, cell.kv, setup)
    for mesh in _meshes():
        hartree._ONSITE_PROJECTIONS.clear()
        wfn = _Loader(cell.psi, cell.occ, cell.kv, mesh)
        lines = []
        # DFT level: U = 1, the WFN's occupations on the window
        fd = hartree.onsite_breit_field(wfn, sym, setup, band_range=(b0, b1), mesh=mesh,
                                        print_fn=lines.append)
        f = cell.occ[:, b0:b1]
        D_ref = np.einsum("k,kn,kan,kbn->ab", w, f, P_ref, P_ref.conj())
        h_ref, _ = ob.onsite_block_potential(D_ref, setup)
        H_ref = np.einsum("kan,ab,kbm->knm", P_ref.conj(), h_ref, P_ref)
        errs = (_rel(fd.projections, P_ref), _rel(fd.density_matrix, D_ref), _rel(fd.matrix, H_ref))
        print(f"\nmesh {dict(mesh.shape)}: |P|, |D|, |H_OS| rel errors {errs[0]:.1e} {errs[1]:.1e} "
              f"{errs[2]:.1e}; max|H_OS| {np.abs(H_ref).max() * ob.RY_TO_MEV:.4f} meV")
        assert fd.projections.shape == (cell.nk, setup.R_os, nb) and max(errs) < 1e-12
        assert fd.matrix.sharding.is_fully_replicated and fd.projections.sharding.is_fully_replicated
        assert lines == fd.report and "H_OS: bands [1, 7)" in lines[-1]
        # the map level: a unitary U and its occupations, with P from the cache (one load)
        U = np.linalg.qr(cell.rng.normal(size=(cell.nk, nb, nb))
                         + 1j * cell.rng.normal(size=(cell.nk, nb, nb)))[0]
        fq = cell.rng.uniform(0, 1, size=(cell.nk, nb))
        fu = hartree.onsite_breit_field(wfn, sym, setup, band_range=(b0, b1), mesh=mesh,
                                        U=jnp.asarray(U), occupations=fq, print_fn=lambda _l: None)
        PU = np.einsum("kan,kni->kai", P_ref, U)
        D_u = np.einsum("k,ki,kai,kbi->ab", w, fq, PU, PU.conj())
        H_u = np.einsum("kan,ab,kbm->knm", P_ref.conj(), ob.onsite_block_potential(D_u, setup)[0], P_ref)
        assert _rel(fu.density_matrix, D_u) < 1e-12 and _rel(fu.matrix, H_u) < 1e-12
        assert wfn.loads == 1
        with pytest.raises(ValueError, match="U needs the occupations"):
            hartree.onsite_breit_field(wfn, sym, setup, band_range=(b0, b1), mesh=mesh, U=U)


def test_onsite_breit_field_gates_fire(setup, cell, monkeypatch):
    from psp import dft_operators
    monkeypatch.setattr(dft_operators, "padded_gvectors", lambda wfn, *, k: SimpleNamespace(
        gvecs=cell.gv, mask=cell.gm, kvecs=cell.kv, ngk=cell.gm.sum(1).astype(int)))
    mesh = _meshes()[0]
    sym = SimpleNamespace(nk_tot=cell.nk, unfolded_kpts=cell.kv)
    quiet = dict(mesh=mesh, print_fn=lambda _l: None)
    hartree._ONSITE_PROJECTIONS.clear()
    # k order: the loader's full-BZ k rows are not sym.unfolded_kpts
    with pytest.raises(ValueError, match="GATE onsite_breit_k_order"):
        hartree.onsite_breit_field(_Loader(cell.psi, cell.occ, cell.kv, mesh),
                                   SimpleNamespace(nk_tot=cell.nk, unfolded_kpts=cell.kv[::-1]),
                                   setup, band_range=(0, 8), **quiet)
    # occupied weight above the window
    with pytest.raises(ValueError, match="GATE onsite_breit_band_cut"):
        hartree.onsite_breit_field(_Loader(cell.psi, cell.occ, cell.kv, mesh), sym, setup,
                                   band_range=(0, 4), **quiet)
    # a non-orthonormal psi breaks the bound D <= f (control: the same call with the true psi)
    hartree.onsite_breit_field(_Loader(cell.psi, cell.occ, cell.kv, mesh), sym, setup,
                               band_range=(0, 8), **quiet)
    hartree._ONSITE_PROJECTIONS.clear()
    with pytest.raises(ValueError, match="GATE onsite_breit_density"):
        hartree.onsite_breit_field(_Loader(10.0 * cell.psi, cell.occ, cell.kv, mesh), sym, setup,
                                   band_range=(0, 8), **quiet)
    hartree._ONSITE_PROJECTIONS.clear()
    # a non-Hermitian h (a broken potential) breaks H_OS's Hermiticity
    true_potential = ob.onsite_block_potential

    def broken(D, s):
        h, p = true_potential(D, s)
        return h + 1e-3j * np.triu(np.ones_like(h), 1), p

    monkeypatch.setattr(ob, "onsite_block_potential", broken)
    with pytest.raises(ValueError, match="GATE onsite_breit_matrix"):
        hartree.onsite_breit_field(_Loader(cell.psi, cell.occ, cell.kv, mesh), sym, setup,
                                   band_range=(0, 8), **quiet)


# ---------------------------------------------------------------- debug writer
def test_debug_writer_round_trip(setup, cell, monkeypatch, tmp_path):
    import h5py
    from file_io.onsite_breit_output import read_onsite_breit_h5, write_onsite_breit_h5
    from common.units import RYD_TO_EV
    from psp import dft_operators
    monkeypatch.setattr(dft_operators, "padded_gvectors", lambda wfn, *, k: SimpleNamespace(
        gvecs=cell.gv, mask=cell.gm, kvecs=cell.kv, ngk=cell.gm.sum(1).astype(int)))
    mesh = _meshes()[0]
    sym = SimpleNamespace(nk_tot=cell.nk, unfolded_kpts=cell.kv)
    hartree._ONSITE_PROJECTIONS.clear()
    fd = hartree.onsite_breit_field(_Loader(cell.psi, cell.occ, cell.kv, mesh), sym, setup,
                                    band_range=(0, 8), mesh=mesh, print_fn=lambda _l: None)
    path = str(tmp_path / "onsite_breit.h5")
    dH = np.asarray(fd.matrix)
    with pytest.raises(TypeError, match="device array"):                      # host arrays only
        write_onsite_breit_h5(path, field=fd, dH_kij_ry=fd.matrix, kpoints=cell.kv, kset="full_bz",
                              band_start_1based=1, attrs={})
    write_onsite_breit_h5(path, field=fd, dH_kij_ry=dH, kpoints=cell.kv, kset="full_bz",
                          band_start_1based=1, attrs={"wfn_fingerprint": "test"},
                          extra={"P_kan": np.asarray(fd.projections)})
    got = read_onsite_breit_h5(path)
    assert _rel(got["dH_kij_ev"], dH * RYD_TO_EV) < 1e-15 and got["attrs"]["kset"] == "full_bz"
    with h5py.File(path, "r") as f:
        dims = f["shell_dim_I"][()]
        assert list(dims) == [sa["stop"] - sa["start"] for sa in setup.shell_atoms]
        for i, sa in enumerate(setup.shell_atoms):
            sl = slice(sa["start"], sa["stop"])
            assert np.array_equal(f["density_matrix_Iab"][i, :dims[i], :dims[i]], fd.density_matrix[sl, sl])
            assert np.allclose(f["h_Iab_ev/total"][i, :dims[i], :dims[i]],
                               fd.pieces[i]["total"] * RYD_TO_EV, rtol=1e-15, atol=0)
            mo = ob.shell_moments(fd.density_matrix[sl, sl], sa["l"])
            assert np.allclose(f["moments/S_Ic"][i], mo["S"]) and f["moments/N_I"][i] == mo["N"]
        assert f.attrs["double_counting"] == "onsite_only" and "P_kan" in f
        assert f.attrs["setup_provenance"] == setup.provenance
    with pytest.raises(ValueError, match="collide"):
        write_onsite_breit_h5(path, field=fd, dH_kij_ry=dH, kpoints=cell.kv, kset="full_bz",
                              band_start_1based=1, attrs={}, extra={"moments": np.zeros(1)})
