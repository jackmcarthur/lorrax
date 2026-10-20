"""``tdagw.model.load_td_model`` options (API v2): what reaches the BSE loader, and ``model["window"]``.

CPU-only and device-free: the loader is replaced by stand-ins with the parameter lists of the
main-line loader, the BSEDIP loader (``velocity_file``) and a loader with ``velocity_full_window``,
so the tests prove the forwarding rule, not the loader.  The eigenvalue check of the eqp
pass-through on a real ``bse_k_grid`` bundle (and its wrong-eqp red twin) is a GPU leg
(docs/theory/td_agw.md, loading).
"""
import inspect
import os
import types

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np
import pytest

from tdagw import model as M


def _main_loader(restart_file, n_val=4, n_cond=4, fermi_energy=0.0, mesh_xy=None, *,
                 input_file=None, n_occ=None, degeneracy_mode="strict", eqp_file=None):
    raise AssertionError("stand-in")


def _bsedip_loader(restart_file, n_val=4, n_cond=4, fermi_energy=0.0, mesh_xy=None, *,
                   input_file=None, n_occ=None, degeneracy_mode="strict", eqp_file=None,
                   velocity_file=None):
    raise AssertionError("stand-in")


def _full_loader(restart_file, n_val=4, n_cond=4, fermi_energy=0.0, mesh_xy=None, *,
                 input_file=None, n_occ=None, degeneracy_mode="strict", eqp_file=None,
                 velocity_file=None, velocity_full_window=False):
    raise AssertionError("stand-in")


MESH = types.SimpleNamespace(devices=np.empty((2, 2), dtype=object))


def test_load_td_model_v2_signature():
    sig = inspect.signature(M.load_td_model)
    want = {"eqp_file": None, "velocity_file": None, "velocity_full_window": False,
            "keep_window": False}
    for name, default in want.items():
        p = sig.parameters[name]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is default


@pytest.mark.parametrize("loader", [_main_loader, _bsedip_loader, _full_loader])
def test_defaults_forward_nothing(loader):
    assert M._loader_options(loader) == {}
    assert M._loader_options(loader, densifies=True) == {}


def test_eqp_forwarded_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = M._loader_options(_main_loader, eqp_file="eqp1.dat")
    assert got == {"eqp_file": os.path.join(str(tmp_path), "eqp1.dat")}


def test_velocity_forwarded_on_densified_route():
    got = M._loader_options(_full_loader, velocity_file="/a/dipole.h5",
                            velocity_full_window=True, densifies=True)
    assert got == {"velocity_file": "/a/dipole.h5", "velocity_full_window": True}
    assert M._loader_options(_bsedip_loader, velocity_file="/a/dipole.h5",
                             densifies=True) == {"velocity_file": "/a/dipole.h5"}


@pytest.mark.parametrize("loader, kw, gate", [
    (_main_loader, dict(velocity_file="/a/d.h5", densifies=True), "td_velocity_loader"),
    (_bsedip_loader, dict(velocity_file="/a/d.h5", densifies=False), "td_velocity_route"),
    (_bsedip_loader, dict(velocity_file="/a/d.h5", velocity_full_window=True, densifies=True),
     "td_velocity_full_window_loader"),
    (_full_loader, dict(velocity_full_window=True, densifies=True), "td_velocity_full_window"),
])
def test_velocity_requests_refuse_by_name(loader, kw, gate):
    with pytest.raises(ValueError, match=f"GATE {gate}:"):
        M._loader_options(loader, **kw)


def _data(densified):
    return dict(n_val=2, n_cond=3, eps_v=np.arange(4.0).reshape(2, 2),
                eps_c=np.arange(6.0).reshape(2, 3), densified=densified, dipoles="D")


def _no_reslice(*a, **k):
    raise AssertionError("re-sliced a densified window")


def test_window_densified_keeps_loader_energies(monkeypatch):
    import bse.bse_window as bw
    monkeypatch.setattr(bw, "apply_eqp_and_reslice_bands", _no_reslice)
    d = _data(True)
    for eqp, source in ((None, "htransform (dft)"), ("/e/eqp1.dat", "htransform (eqp)")):
        w = M._window_record(d, restart_file="r.h5", input_file="d.in", n_occ=130,
                             eqp_file=eqp, mesh_xy=MESH, degeneracy_mode="strict")
        assert w["eps_v"] is d["eps_v"] and w["eps_c"] is d["eps_c"]
        assert (w["n_occ"], w["n_val"], w["n_cond"], w["dipoles"]) == (130, 2, 3, "D")
        assert w["densified"] and w["energy_source"] == source and w["eqp_file"] == eqp


def test_window_coarse_eqp_reslices_like_bse_jax(monkeypatch):
    import bse.bse_window as bw
    seen = {}

    def fake(restart_file, eqp_file, input_file, n_val, n_cond, n_occ, gx, gy, **kw):
        seen.update(args=(restart_file, eqp_file, input_file, n_val, n_cond, n_occ, gx, gy))
        return "EV", "EC", seen.get("n_occ_out", n_occ)

    monkeypatch.setattr(bw, "apply_eqp_and_reslice_bands", fake)
    w = M._window_record(_data(False), restart_file="r.h5", input_file="d.in", n_occ=130,
                         eqp_file="/e/eqp1.dat", mesh_xy=MESH, degeneracy_mode="strict")
    assert (w["eps_v"], w["eps_c"], w["n_occ"]) == ("EV", "EC", 130)
    assert seen["args"] == ("r.h5", "/e/eqp1.dat", "d.in", 2, 3, 130, 2, 2)
    assert w["energy_source"].startswith("eqp re-slice")
    seen["n_occ_out"] = 131                           # a QP gap that moves n_occ
    with pytest.raises(ValueError, match="GATE td_eqp_n_occ:"):
        M._window_record(_data(False), restart_file="r.h5", input_file="d.in", n_occ=130,
                         eqp_file="/e/eqp1.dat", mesh_xy=MESH, degeneracy_mode="strict")


def test_window_coarse_without_eqp_is_the_restart():
    d = _data(False)
    w = M._window_record(d, restart_file="r.h5", input_file="d.in", n_occ=130,
                         eqp_file=None, mesh_xy=MESH, degeneracy_mode="strict")
    assert w["eps_v"] is d["eps_v"] and w["energy_source"] == "restart" and not w["densified"]


@pytest.mark.parametrize("line, want", [("bse_k_grid = 30 30 1\n", True),
                                        ("bse_k_grid = 8 8 1\n", False), ("", False)])
def test_deck_densifies_is_the_loaders_test(tmp_path, line, want):
    deck = tmp_path / "gnppm.in"
    deck.write_text("[cohsex]\nsys_dim = 2\nnval = 130\n" + line)
    assert M._deck_densifies(str(deck), {"grid": np.array([8, 8, 1])}) is want
