"""A TD-aGW restart continues bitwise from a checkpoint of either carry format.

The two-level Rabi system of the validation ladder (five k copies, rk4 at a
fixed 10 as step, density matrix every 100 as, a checkpoint every 5 samples)
runs through the driver's stages:

* run A to 2000 as, its checkpoints written by the slotted writer (since
  2026-11-22), and again with the writer before it (``checkpoint/lorrax``,
  reproduced verbatim below);
* run B from each A's last checkpoint to 4000 as
  (``load_checkpoint_from``, a new outdir);
* run C straight to 4000 as.

Every B's rows and final carry must equal C's bit for bit.  A header file
with no LORRAX carry (``init_output`` only) must refuse with
``td_restart_carry``.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
h5py = pytest.importorskip("h5py")

import _validation_systems as vs  # noqa: E402
from tdagw import tdgw_io  # noqa: E402


def _main_write_checkpoint(path, *, it_next, tchkp, lessG, carry=None):
    """``tdgw_io.write_checkpoint`` as on main before 2026-11-22 (the old format)."""
    with h5py.File(path, "a") as f:
        c = f["tdgw_data/checkpoint"]
        c["it_next"][...] = it_next
        c["tchkp"][...] = tchkp
        c["lessG_knn"][...] = lessG
        if carry is not None:
            if "lorrax" in c:
                del c["lorrax"]
            g = c.create_group("lorrax")
            for i, leaf in enumerate(carry["leaves"]):
                g.create_dataset(f"leaf_{i:03d}", data=np.asarray(leaf))
            g.attrs["n_leaves"] = len(carry["leaves"])
            g.attrs["step"] = int(carry["step"])
            g.attrs["fingerprint"] = carry["fingerprint"]


def _run(outdir, header, system, *, tf_as, load_from=""):
    from common.collectives import resolve_mesh
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    nb = int(system["e_ev"].shape[1])
    raw = {"control": {"calc_type": "IP",
                       "timesteps": {"t0": 0, "tf": float(tf_as), "dt_measure": 100.0, "ntcheck": 5},
                       "ivp": {"rtol": 1e-10, "atol": 1e-14},
                       "files": {"outdir": str(outdir), "load_header_from": str(header),
                                 "load_checkpoint_from": str(load_from)}},
           "system": {"ibndmin": 1, "ibndmax": nb, "ivbm": int(system["n_val"]),
                      "geometry": "slab", "depopulation": 0.005, "dephasing": 0.02},
           "measurements": {"types": ["density_matrix"]},
           "lorrax": {"method": "rk4", "dt": 10.0, "n_corr": 2, "intraband": False,
                      "trials": [vs.circular_trial(2.0e9, 2.0, sign=+1)]}}
    deck = parse_td_deck(raw, source=str(outdir))
    mesh = resolve_mesh(None)
    sysd = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, sysd, mesh)
    outs = stages.open_outputs(deck, sysd)
    stages.propagate(deck, sysd, model, tdgw_io.efield_function(deck.trials), outs, mesh,
                     say=lambda s: None)
    path = outs.paths[0]
    with h5py.File(path, "r") as f:
        rows = f["tdgw_data/time_domain/density_matrix/data"][()]
    return path, rows, tdgw_io.read_checkpoint(path)


def test_restart_from_both_formats_is_bitwise(tmp_path, monkeypatch):
    tl = vs.two_level()
    header = vs.write_header(str(tmp_path / "hdr"), tl)
    _, rows_c, rec_c = _run(tmp_path / "C", header, tl, tf_as=4000.0)
    for fmt in ("slotted", "before"):
        with monkeypatch.context() as mp:
            if fmt == "before":
                mp.setattr(tdgw_io, "write_checkpoint", _main_write_checkpoint)
            path_a, rows_a, rec_a = _run(tmp_path / f"A_{fmt}", header, tl, tf_as=2000.0)
        with h5py.File(path_a, "r") as f:
            has = sorted(k for k in f["tdgw_data/checkpoint"])
        assert ("lorrax" in has) == (fmt == "before") and ("lorrax_record" in has) == (fmt == "slotted"), has
        assert rec_a["it_next"] == rows_a.shape[0] > 5 and not rec_a["torn"], rec_a["it_next"]
        _, rows_b, rec_b = _run(tmp_path / f"B_{fmt}", header, tl, tf_as=4000.0,
                                load_from=tmp_path / f"A_{fmt}")
        assert rows_b.shape == rows_c.shape and np.array_equal(rows_b, rows_c), (
            fmt, float(np.max(np.abs(rows_b - rows_c))))
        assert rec_b["it_next"] == rec_c["it_next"] and rec_b["step"] == rec_c["step"]
        assert all(np.array_equal(a, b) for a, b in zip(rec_b["leaves"], rec_c["leaves"])), fmt


def test_restart_without_carry_refuses(tmp_path):
    tl = vs.two_level()
    header = vs.write_header(str(tmp_path / "hdr"), tl)
    os.makedirs(tmp_path / "plain", exist_ok=True)
    # A TDGW-style file: init_output alone, no LORRAX carry.
    os.symlink(os.path.join(header, "tdgw.h5"), tmp_path / "plain" / "tdgw.h5")
    with pytest.raises(ValueError, match="td_restart_carry"):
        _run(tmp_path / "R", header, tl, tf_as=2000.0, load_from=tmp_path / "plain")
