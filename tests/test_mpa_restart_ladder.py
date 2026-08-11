"""The restart ladder's drop branches: does a changed input rebuild it?

THE FAILURE THESE GUARD is the one that nearly shipped on 2026-08-10: a
stored object reused because nothing compared the thing that had changed.
Both tests are RED FIRST in the sense that matters -- each one changes
exactly one input and asserts the ladder gives up exactly one rung, and
each would pass trivially if the ladder always rebuilt, so each is paired
with a green case asserting the rung IS taken when nothing changed.

These are pure-logic tests on purpose.  ``resolve_mpa_restart`` takes the
zeta verdict and the sampling expectation as VALUES, so the drop branches
need no filesystem, no mesh and no ISDF fit -- which is what lets them run
on CPU in the light gate while the sharded driver-seam gate is still owed.
"""

import numpy as np
import pytest

from gw.mpa import w_restart as WR


ZGRID = np.array([0.0, 0.1j, 0.25j, 0.5j], dtype=np.complex128)


def _header(omega, **prov):
    """A W(omega) header shaped like ``mpa_store.read_w_header``'s."""
    n = int(np.asarray(omega).size)
    return {
        "omega": np.asarray(omega, dtype=np.complex128),
        # every slab written: an unwritten one is an IDENTITY failure and
        # raises, which is a different test from these staleness drops
        "data_ready": np.ones(n, dtype=bool),
        "n_omega": int(np.asarray(omega).size),
        "grid_hash": "sha256:deadbeefcafef00d",
        "omega_units": "Ha",
        "screening_content": "W_c",
        "q_storage": "ibz",
        "provenance": dict(prov),
    }


# --------------------------------------------------------------------------
# RED 1: a changed centroid file must drop all the way to the zeta refit
# --------------------------------------------------------------------------

def test_changed_centroids_drops_to_zeta_refit():
    """A stale centroid table costs the zeta rung, and says which key."""
    d = WR.resolve_mpa_restart(
        "/nonexistent/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=False,
                          reason="centroids_md5"))
    assert d.stage == WR.STAGE_BUILD, (
        "a zeta fit whose centroid table changed describes a different "
        "basis; reusing it would put this run's W on someone else's "
        "centroids")
    assert any("centroids_md5" in r for r in d.drops), (
        f"the drop must NAME the key that caused it; got {d.drops!r}")


def test_unchanged_centroids_keeps_the_zeta_rung():
    """The green half: nothing changed, so the ISDF fit is skipped."""
    d = WR.resolve_mpa_restart(
        "/nonexistent/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=True))
    assert d.stage == WR.STAGE_BUILD_W
    assert d.drops == (), (
        "a run that resumed everything it could should report no drops")


def test_absent_zeta_is_not_a_drop():
    """Absence is not staleness: nothing to name, nothing to announce."""
    d = WR.resolve_mpa_restart("/nonexistent/bundle.h5",
                               zeta=WR.ZetaState(present=False))
    assert d.stage == WR.STAGE_BUILD
    assert d.drops == ()


# --------------------------------------------------------------------------
# RED 2: a changed z-grid must drop the W sweep -- and the poles with it
# --------------------------------------------------------------------------

def test_changed_z_grid_drops_the_w_sweep(monkeypatch):
    """A different frequency grid re-sweeps, naming omega and the index."""
    moved = ZGRID.copy()
    moved[2] = 0.30j
    monkeypatch.setattr(WR.os.path, "exists", lambda p: True)
    monkeypatch.setattr(WR, "_has",
                        lambda path, name: name == WR.W_OMEGA_DATASET)
    monkeypatch.setattr(WR.mpa_store, "read_w_header",
                        lambda path, name: _header(ZGRID))
    monkeypatch.setattr(WR.mpa_store, "require_correlation_part",
                        lambda *a, **k: None)
    monkeypatch.setattr(WR.mpa_store, "canonical_energy_unit",
                        lambda *a, **k: "Ha")
    monkeypatch.setattr(WR, "_refuse_a_foreign_wfn",
                        lambda *a, **k: None)

    d = WR.resolve_mpa_restart(
        "/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=True),
        sampling=WR.SamplingExpectation(omega=moved))
    assert d.stage == WR.STAGE_BUILD_W, (
        "the sweep is stale, so it must be redone -- but the zeta fit "
        "under it is still good and must NOT be")
    assert any("omega" in r for r in d.drops), d.drops
    assert any("index 2" in r for r in d.drops), (
        f"the drop should point at WHERE the grids diverge; got {d.drops!r}")


def test_unchanged_z_grid_keeps_the_sweep(monkeypatch):
    """The green half: same grid, so the sweep is skipped and the fit runs."""
    monkeypatch.setattr(WR.os.path, "exists", lambda p: True)
    monkeypatch.setattr(WR, "_has",
                        lambda path, name: name == WR.W_OMEGA_DATASET)
    monkeypatch.setattr(WR.mpa_store, "read_w_header",
                        lambda path, name: _header(ZGRID))
    monkeypatch.setattr(WR.mpa_store, "require_correlation_part",
                        lambda *a, **k: None)
    monkeypatch.setattr(WR.mpa_store, "canonical_energy_unit",
                        lambda *a, **k: "Ha")
    monkeypatch.setattr(WR, "_refuse_a_foreign_wfn", lambda *a, **k: None)

    d = WR.resolve_mpa_restart(
        "/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=True),
        sampling=WR.SamplingExpectation(omega=ZGRID.copy()))
    assert d.stage == WR.STAGE_FIT
    assert d.drops == ()


def test_changed_sampling_key_drops_and_names_it(monkeypatch):
    """A deck key that feeds the sweep is as good as a changed grid."""
    monkeypatch.setattr(WR.os.path, "exists", lambda p: True)
    monkeypatch.setattr(WR, "_has",
                        lambda path, name: name == WR.W_OMEGA_DATASET)
    monkeypatch.setattr(WR.mpa_store, "read_w_header",
                        lambda path, name: _header(ZGRID, n_rmu=1128))
    monkeypatch.setattr(WR.mpa_store, "require_correlation_part",
                        lambda *a, **k: None)
    monkeypatch.setattr(WR.mpa_store, "canonical_energy_unit",
                        lambda *a, **k: "Ha")
    monkeypatch.setattr(WR, "_refuse_a_foreign_wfn", lambda *a, **k: None)

    d = WR.resolve_mpa_restart(
        "/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=True),
        sampling=WR.SamplingExpectation(omega=ZGRID.copy(),
                                        keys={"n_rmu": 1394}))
    assert d.stage == WR.STAGE_BUILD_W
    assert any("n_rmu" in r for r in d.drops), d.drops


def test_a_key_the_store_never_stamped_is_not_a_mismatch(monkeypatch):
    """Silence on either side is not disagreement -- the settled rule."""
    monkeypatch.setattr(WR.os.path, "exists", lambda p: True)
    monkeypatch.setattr(WR, "_has",
                        lambda path, name: name == WR.W_OMEGA_DATASET)
    monkeypatch.setattr(WR.mpa_store, "read_w_header",
                        lambda path, name: _header(ZGRID))
    monkeypatch.setattr(WR.mpa_store, "require_correlation_part",
                        lambda *a, **k: None)
    monkeypatch.setattr(WR.mpa_store, "canonical_energy_unit",
                        lambda *a, **k: "Ha")
    monkeypatch.setattr(WR, "_refuse_a_foreign_wfn", lambda *a, **k: None)

    d = WR.resolve_mpa_restart(
        "/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=True),
        sampling=WR.SamplingExpectation(omega=ZGRID.copy(),
                                        keys={"line_heights": [0.1, 1.0]}))
    assert d.stage == WR.STAGE_FIT, (
        "a store that predates a sampling key must not be rebuilt on "
        "the strength of its silence")


# --------------------------------------------------------------------------
# The extension must be strict: an untaught driver keeps its old behaviour
# --------------------------------------------------------------------------

def test_no_zeta_and_no_sampling_is_the_old_ladder(monkeypatch):
    """A caller that says nothing gets exactly the pre-ladder answer."""
    monkeypatch.setattr(WR.os.path, "exists", lambda p: False)
    d = WR.resolve_mpa_restart("/bundle.h5")
    assert d.stage == WR.STAGE_BUILD
    assert d.drops == ()


def test_announce_drops_prints_one_line_each():
    """Every dropped rung is stated; a silent rebuild is the bug."""
    lines = []
    d = WR.resolve_mpa_restart(
        "/nonexistent/bundle.h5",
        zeta=WR.ZetaState(present=True, valid=False, reason="n_rmu"))
    d.announce_drops(lines.append)
    assert len(lines) == len(d.drops) == 1
    assert "n_rmu" in lines[0]
    assert "NOT resuming" in lines[0]


@pytest.mark.parametrize("stage", [WR.STAGE_BUILD, WR.STAGE_BUILD_W,
                                   WR.STAGE_FIT, WR.STAGE_SIGMA])
def test_every_stage_describes_itself(stage):
    """No rung may print a bare repr -- the log is the audit trail."""
    d = WR.MpaRestartDecision(path="/tmp/b.h5", stage=stage,
                              w_header=_header(ZGRID) if stage in
                              (WR.STAGE_FIT, WR.STAGE_SIGMA) else None,
                              fit_ledger={"n_p": 8, "n_q": 8,
                                          "energy_unit": "Ha",
                                          "screening_content": "W_c"}
                              if stage == WR.STAGE_SIGMA else None)
    text = WR.describe(d)
    assert "b.h5" in text and len(text) > 40


# --------------------------------------------------------------------------
# The seam: the verdict travels, and does not linger
# --------------------------------------------------------------------------

def test_recorded_zeta_state_is_single_use():
    """A stale verdict is the failure this ladder exists to prevent."""
    WR.record_zeta_state(present=True, valid=True)
    first = WR.take_zeta_state()
    assert first is not None and first.valid
    assert WR.take_zeta_state() is None, (
        "a second seam that ran no ISDF stage must get 'cannot say', not "
        "the previous run's answer")


def test_recorded_invalid_zeta_carries_its_reason():
    WR.record_zeta_state(present=True, valid=False, reason="centroids_md5")
    st = WR.take_zeta_state()
    assert st.present and not st.valid and st.reason == "centroids_md5"
    d = WR.resolve_mpa_restart("/nonexistent/bundle.h5", zeta=st)
    assert d.stage == WR.STAGE_BUILD
    assert any("centroids_md5" in r for r in d.drops)
