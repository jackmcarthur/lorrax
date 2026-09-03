"""ONE owner for each policy the PPM and MPA Sigma branches both need.

Three sibling drifts, one file, because they are the same defect three
times: a rule that one ansatz enforced and its sibling did not.

**(a) The sharded-cube divisibility precondition.**  ``ppm_sigma`` refused
an indivisible sigma band window by name under
``sigma_omega_layout=sharded`` while ``gw.mpa.sigma`` padded, accumulated
into a ``P(None,None,'x','y')`` array and stripped the same cube with no
divisibility check anywhere in the module.  One contract at one seam now:
``ppm_sigma.assert_sharded_sigma_window_divides_mesh``.

**(b) The effective Sigma broadening xi.**  GN/HL-PPM and MPA use the
requested ``sigma_regularization_ev`` by default.  The bounded HGL cell
planner, not an omega-extent-dependent xi floor, owns conditioning.  One
resolver (``ppm_windows.resolve_sigma_regularization``), one explicit floor
key (``sigma_regularization_floor_ev``), one stamp.

**(c) fermi_reference.**  MPA read ``wfn.efermi`` and never looked at the
key.  ``wfn.efermi`` IS the midgap, so the default agreed by coincidence
and only ``vbm`` was silently wrong -- the worst shape a defect can have.
One resolver (``gw.efermi.resolve_sigma_efermi_ry``) returning the energy
AND the provenance string the h5 stamp needs.

SCOPE.  Host-side policy cells: they drive the resolvers and the
preconditions directly with synthetic inputs.  They do NOT run a Sigma,
do not compare Sigma values between the two ansaetze, and therefore make
no claim that PPM and MPA now agree numerically -- only that they can no
longer disagree about these three policies without saying so.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import jax
from jax.sharding import Mesh

from common.units import RYD_TO_EV


# ---------------------------------------------------------------------------
# (a) one divisibility precondition
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh22():
    d = jax.devices()
    if len(d) < 4:
        pytest.skip("needs 4 devices for a 2x2 mesh")
    return Mesh(np.array(d[:4]).reshape(2, 2), ("x", "y"))


def test_both_ansatze_call_the_same_sharded_window_precondition():
    """Source gate: the MPA executor must not re-derive the divisibility rule.

    A source cell rather than a run cell because reaching the MPA executor
    needs a pole store on disk.  The point being pinned is structural --
    that there is ONE function and both branches call it -- and that is
    exactly what a source cell can see.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "gw"
    mpa = (root / "mpa" / "sigma.py").read_text()
    ppm = (root / "ppm_sigma.py").read_text()
    assert "assert_sharded_sigma_window_divides_mesh(" in mpa, (
        "the MPA executor pads and strips a sharded cube without the "
        "precondition the PPM branch refuses on")
    assert ppm.count("def assert_sharded_sigma_window_divides_mesh") == 1
    # And the rule is not ALSO spelled out inline in the MPA module.
    mpa_code = "\n".join(l.split("#", 1)[0] for l in mpa.splitlines())
    assert "% p_x" not in mpa_code and "% p_y" not in mpa_code


def test_the_precondition_refuses_an_indivisible_window(mesh22):
    from gw.ppm_sigma import assert_sharded_sigma_window_divides_mesh

    # Divides both axes -> silent.
    assert_sharded_sigma_window_divides_mesh(8, mesh22, ansatz="gn_ppm")
    # ODD counts only: on a 2x2 mesh anything even divides both axes,
    # so an even nb would be testing the silent branch by accident.
    for nb in (7, 9, 15):
        with pytest.raises(ValueError) as exc:
            assert_sharded_sigma_window_divides_mesh(
                nb, mesh22, ansatz="compute_mode = mpa")
        msg = str(exc.value)
        assert str(nb) in msg and "2x2" in msg
        # The MPA arm must NOT advise a layout it has no plan for.
        assert "sigma_omega_layout = replicated" not in msg
    with pytest.raises(ValueError) as exc:
        assert_sharded_sigma_window_divides_mesh(7, mesh22, ansatz="gn_ppm")
    assert "sigma_omega_layout = replicated" in str(exc.value)


# ---------------------------------------------------------------------------
# (b) one effective-xi resolver
# ---------------------------------------------------------------------------

def test_auto_keeps_requested_xi_for_every_ansatz():
    """Omega extent is absent from the resolver and cannot change xi."""
    import inspect

    from gw.ppm_windows import resolve_sigma_regularization

    assert "omega_grid_ry" not in inspect.signature(
        resolve_sigma_regularization).parameters
    resolved = [resolve_sigma_regularization(
        requested_ry=0.25 / RYD_TO_EV, ansatz=ansatz)
        for ansatz in ("gn_ppm", "hl_ppm", "mpa")]
    assert all(not value.raised for value in resolved)
    assert all(value.resolved_ev == pytest.approx(0.25, rel=1e-12)
               for value in resolved)
    assert all(value.floor_ry == 0.0 for value in resolved)


def test_an_explicit_floor_equalises_xi_across_ansatze():
    """The knob the register asked for: one number, both ansaetze."""
    from gw.ppm_windows import resolve_sigma_regularization

    kw = dict(requested_ry=0.25 / RYD_TO_EV, floor_ev=0.8)
    a = resolve_sigma_regularization(ansatz="gn_ppm", **kw)
    b = resolve_sigma_regularization(ansatz="mpa", **kw)
    assert a.resolved_ry == b.resolved_ry
    assert a.resolved_ev == pytest.approx(0.8, rel=1e-12)
    assert a.floor_policy == b.floor_policy == "explicit"
    # An explicit 0 means "do not raise" -- spellable on purpose, and it
    # says so in the stamp rather than being indistinguishable from auto.
    z = resolve_sigma_regularization(ansatz="gn_ppm", **{**kw, "floor_ev": 0})
    assert not z.raised and z.floor_policy == "explicit"


def test_a_requested_xi_above_the_floor_is_never_lowered():
    from gw.ppm_windows import resolve_sigma_regularization

    r = resolve_sigma_regularization(
        requested_ry=3.0 / RYD_TO_EV, ansatz="gn_ppm")
    assert r.resolved_ev == pytest.approx(3.0, rel=1e-12) and not r.raised


def test_the_log_line_is_the_same_sentence_for_both_ansatze():
    """A comparison can only assert equal xi if both runs print it alike."""
    from gw.ppm_windows import resolve_sigma_regularization

    lines = [resolve_sigma_regularization(
        requested_ry=0.25 / RYD_TO_EV,
        ansatz=a).describe() for a in ("gn_ppm", "mpa")]
    for line in lines:
        assert "Σ broadening ξ:" in line and "requested" in line
    assert all("RAISED" not in line for line in lines)


def test_a_bad_floor_refuses_rather_than_resolving_to_auto():
    from gw.ppm_windows import resolve_sigma_regularization

    with pytest.raises(ValueError):
        resolve_sigma_regularization(
            requested_ry=0.25 / RYD_TO_EV,
            ansatz="gn_ppm", floor_ev=-1.0)


# ---------------------------------------------------------------------------
# (c) one fermi_reference resolver
# ---------------------------------------------------------------------------

def _wfn(vbm=0.10, cbm=0.30):
    return SimpleNamespace(vbm=vbm, cbm=cbm, efermi=0.5 * (vbm + cbm))


def test_fermi_reference_is_honored_including_the_value_mpa_used_to_drop():
    from gw.efermi import resolve_sigma_efermi_ry

    wfn = _wfn()
    e_mid, p_mid = resolve_sigma_efermi_ry(
        "midgap", occupation_state=None, wfn=wfn)
    e_vbm, p_vbm = resolve_sigma_efermi_ry(
        "vbm", occupation_state=None, wfn=wfn)
    assert e_mid == pytest.approx(0.20) and p_mid == "midgap"
    assert e_vbm == pytest.approx(0.10) and p_vbm == "vbm"
    # THE DEFECT: 'midgap' and 'vbm' must not resolve to the same energy.
    # MPA read wfn.efermi for both, and wfn.efermi IS the midgap -- so the
    # default agreed by coincidence and this pair was the only witness.
    assert e_mid != e_vbm
    assert p_mid != p_vbm, (
        "the provenance stamp must distinguish them too, or sigma_mnk.h5 "
        "cannot say which reference its omega axis is measured from")


def test_mp1_fixed_n_takes_the_iterations_chemical_potential():
    from gw.efermi import resolve_sigma_efermi_ry

    state = SimpleNamespace(mu_ry=0.4242)
    e, p = resolve_sigma_efermi_ry(
        "mp1_fixed_n", occupation_state=state, wfn=_wfn())
    assert e == pytest.approx(0.4242) and p == "fixed-N mu"


def test_a_gap_reference_beside_a_fixed_n_state_refuses():
    """Not a preference -- the occupations were solved at the OTHER one."""
    from gw.efermi import resolve_sigma_efermi_ry

    state = SimpleNamespace(mu_ry=0.4242)
    with pytest.raises(ValueError) as exc:
        resolve_sigma_efermi_ry("midgap", occupation_state=state, wfn=_wfn())
    assert "mp1_fixed_n" in str(exc.value)
    with pytest.raises(ValueError):
        resolve_sigma_efermi_ry(
            "mp1_fixed_n", occupation_state=None, wfn=_wfn())


def test_an_unknown_reference_refuses_by_name():
    from gw.efermi import resolve_sigma_efermi_ry

    with pytest.raises(ValueError):
        resolve_sigma_efermi_ry("cbm", occupation_state=None, wfn=_wfn())


def test_the_provenance_stamp_is_honored_rather_than_inferred():
    """The `efermi_ry is None` proxy must not overwrite a real provenance.

    With MPA now passing an explicit reference for EVERY fermi_reference,
    the old proxy would stamp a midgap run as "fixed-N mu".
    """
    import inspect
    from gw.dynamic_sigma import eval_sigma_c_at_dft_energies

    sig = inspect.signature(eval_sigma_c_at_dft_energies)
    assert "efermi_provenance" in sig.parameters
    src = inspect.getsource(eval_sigma_c_at_dft_energies)
    assert "if efermi_provenance is not None:" in src
