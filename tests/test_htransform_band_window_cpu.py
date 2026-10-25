"""htransform's band window comes from the band request, never from a BSE key.

``fh_interp.band_window`` returns nval + ncond (``resolve_band_request``, so
``number_bands_protected`` too) and fits ``--guard-bands`` above it; the BSE's
``number_cond_bands_coarse`` is not read, so an input file shared with the BSE
(the Si SOC example's ``gw.in``: ncond = 20, number_cond_bands_coarse = 16)
runs unchanged.  Under ``--qp-rotations`` without an SC energy ladder, the
fitted rows above the QP block take DFT + the block's mean conduction shift
(``fh_interp.resolve_qp_hamiltonian_state``), as ``--eqp-file`` does above the
eqp file, so the returned window may reach the top of the QP block.

Host-side window logic and one compact-row update on a 1-device CPU mesh: a
CPU unit cell, exempt from the four-GPU rule under its unit-and-CPU-cell clause.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SI_SOC_GW_IN = os.path.join(HERE, "..", "examples", "Si_soc", "lorrax", "gw.in")
quiet = lambda *a, **k: None


def _window(params, nelec, guard_bands=4):
    from bandstructure.fh_interp import band_window
    from gw.gw_config import resolve_band_request
    resolve_band_request(params, SimpleNamespace(nelec=nelec), quiet)
    lines = []
    return band_window(params, guard_bands, lines.append), lines


def test_si_soc_gw_input_file_returns_its_qp_bands():
    from gw.gw_config import read_lorrax_input
    params = read_lorrax_input(SI_SOC_GW_IN)
    (n_return, n_fit), lines = _window(params, nelec=8)
    assert (n_return, n_fit) == (28, 32)      # bands 1-28 as sigma.inp, 4 guards
    assert len(lines) == 1 and "number_cond_bands_coarse" in lines[0]
    assert params["kpoints_crystal_b"]["segments"]   # the path is in gw.in too


def test_number_bands_protected_sets_the_window(tmp_path):
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "gw.in"
    deck.write_text("[cohsex]\nsys_dim = 3\nnumber_bands = 60\n"
                    "number_bands_protected = 28\nnumber_cond_bands_coarse = 16\n")
    (n_return, n_fit), _lines = _window(read_lorrax_input(str(deck)), nelec=8,
                                        guard_bands=2)
    assert (n_return, n_fit) == (28, 30)


def test_qp_rotations_scissor_above_the_qp_block(tmp_path, monkeypatch, caplog):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import file_io.qp_wfn as qp_wfn
    import file_io.restart_bundle as restart_bundle
    from bandstructure import fh_interp

    nk, nb_fit, nb_qp, nelec, rank = 4, 8, 6, 2, 3
    rng = np.random.default_rng(20261025)
    dft = np.sort(rng.uniform(-0.5, 1.0, (nk, nb_fit)), axis=1)          # Ry
    e_qp = dft[:, :nb_qp] + rng.uniform(0.0, 0.05, (nk, nb_qp))
    u = np.broadcast_to(np.eye(nb_qp), (nk, nb_qp, nb_qp)).astype(np.complex128)
    path = tmp_path / "qp_wfn_rotations.h5"
    path.write_bytes(b"")
    monkeypatch.setattr(qp_wfn, "refuse_conflicting_qp_state_sources",
                        lambda **k: None)
    monkeypatch.setattr(qp_wfn, "authenticate_qp_rotations_source_wfn",
                        lambda *a, **k: None)
    monkeypatch.setattr(restart_bundle, "read_qp_rotations_artifact", lambda p: {})
    monkeypatch.setattr(restart_bundle, "validate_qp_rotations_frame",
                        lambda *a, **k: (u, e_qp, (0, nb_qp)))
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
    ctilde = jax.device_put(
        jnp.asarray(rng.standard_normal((nk, nb_fit, rank)).astype(np.complex128)),
        NamedSharding(mesh, P()))
    with caplog.at_level("INFO", logger="lorrax.bandstructure.fh_interp"):
        _c, enk, corrected = fh_interp.resolve_qp_hamiltonian_state(
            basis=SimpleNamespace(ctilde=ctilde, band_range=(0, nb_fit)),
            enk_sigma=dft.T, sym=SimpleNamespace(unfolded_kpts=None),
            meta=SimpleNamespace(kgrid=None), wfn=SimpleNamespace(nelec=nelec),
            wfn_path="WFN.h5", qp_rotations_file=str(path))
    shift = float(np.mean((e_qp - dft[:, :nb_qp])[:, nelec:]))
    enk = np.asarray(enk)
    assert corrected == (0, nb_fit)    # no DFT guard boundary left to refuse
    assert np.allclose(enk[:nb_qp], e_qp.T, atol=1e-6)
    assert np.allclose(enk[nb_qp:], dft[:, nb_qp:].T + shift, atol=1e-6)
    lines = [r.getMessage() for r in caplog.records if "QP scissor" in r.getMessage()]
    assert len(lines) == 1 and f"[{nb_qp}, {nb_fit})" in lines[0]
