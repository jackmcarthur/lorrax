"""The SC stop test reads each trusted label's own sorted column (CPU, ms).

``gw.sc_iteration._sc_identity_for_call`` on a one-k toy with DFT levels
-1, 0, 0 (+50 ueV), 1 eV.  A map that keeps the pair's centre but changes
its splitting from 0 to 10 meV moved each member 5 meV; the retired block
mean read 0.  An exact doublet in an arbitrary internal gauge reads its true
motion.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import common.collectives as collectives
from gw import sc_iteration


@dataclass(frozen=True)
class _Outputs:
    sigma_basis_U: np.ndarray
    identity: dict | None = None


@dataclass(frozen=True)
class _State:
    outputs: _Outputs = field(default=None)


def _call(monkeypatch, e_in, e_out, u_in, u_out, history):
    nb = e_in.shape[1]
    trusted = np.ones((1, nb), dtype=bool)
    part = SimpleNamespace(protected_mask=trusted, in_range_mask=~trusted)
    monkeypatch.setattr(sc_iteration, "_state_partition", lambda s, i: part)
    monkeypatch.setattr(sc_iteration, "_partition_on_loop", lambda p, i: p)
    monkeypatch.setattr(sc_iteration, "_record_sc", lambda i, line: None)
    monkeypatch.setattr(collectives, "gather_to_host", np.asarray)
    inputs = SimpleNamespace(initial_state_role="dft_seed")
    verdict, _ = sc_iteration._sc_identity_for_call(
        inputs, _State(_Outputs(u_in)), e_in, e_out, history,
        cutoff_ev=1e-3, u_out=u_out)
    return verdict


def test_split_change_inside_an_accidental_block_is_seen(monkeypatch):
    eye = np.eye(4)[None]
    dft = np.array([[-1.0, 0.0, 5e-5, 1.0]])
    history = {}
    _call(monkeypatch, dft, dft, eye, eye, history)        # map 0 sets labels
    e_in = np.array([[-1.0, 0.0, 0.0, 1.0]])               # pair degenerate
    e_out = np.array([[-1.0, -0.005, 0.005, 1.0]])         # split 10 meV, same centre
    verdict = _call(monkeypatch, e_in, e_out, eye, eye, history)
    assert np.isclose(verdict.max_abs_ev, 5e-3)
    assert not verdict.converged


def test_exact_doublet_in_any_gauge_reads_its_motion(monkeypatch):
    eye = np.eye(4)[None]
    dft = np.array([[-1.0, 0.0, 0.0, 1.0]])
    history = {}
    _call(monkeypatch, dft, dft, eye, eye, history)
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.eye(4, dtype=complex)
    rot[1:3, 1:3] = [[c, -s], [s, c]]                      # gauge inside the doublet
    e_in = np.array([[-1.0, 2e-4, 2e-4, 1.0]])
    e_out = np.array([[-1.0, 6e-4, 6e-4, 1.0]])
    verdict = _call(monkeypatch, e_in, e_out, rot[None], rot[None], history)
    assert np.isclose(verdict.max_abs_ev, 4e-4)
    assert verdict.converged


def test_retired_degeneracy_key_refuses_by_name(tmp_path):
    import pytest
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "sc.in"
    deck.write_text("[cohsex]\nsc_exact_degeneracy_tol_ev = 1e-4\n")
    with pytest.raises(ValueError, match="'sc_exact_degeneracy_tol_ev' is retired"):
        read_lorrax_input(str(deck))
