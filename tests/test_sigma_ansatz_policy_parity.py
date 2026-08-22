"""ONE owner for each policy the PPM and MPA Sigma branches both need.

Three sibling drifts, one file, because they are the same defect three
times: a rule that one ansatz enforced and its sibling did not.

**(a) The sharded-cube divisibility precondition.**  ``ppm_sigma`` refused
an indivisible sigma band window by name under
``sigma_omega_layout=sharded`` while ``gw.mpa.sigma`` padded, accumulated
into a ``P(None,None,'x','y')`` array and stripped the same cube with no
divisibility check anywhere in the module.  One contract at one seam now:
``ppm_sigma.assert_sharded_sigma_window_divides_mesh``.

**(b) The effective Sigma broadening xi.**  GN-PPM silently raised the
deck's ``sigma_regularization_ev`` to a window-dependent conditioning
floor; MPA passed it straight through.  1.90x apart on a +/-5 eV grid,
5.7x on +/-15 eV, and the resolved value appeared in no artifact -- so a
cross-ansatz comparison could not assert that two runs shared xi.  One
resolver (``ppm_windows.resolve_sigma_regularization``), one deck key
(``sigma_regularization_floor_ev``), one stamp.

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
    for nb in (7, 9, 70):
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

def _omega_grid_ry(half_width_ev):
    half = half_width_ev / RYD_TO_EV
    return np.linspace(-half, half, 41)


def test_auto_raises_the_hgl_ansatze_and_leaves_mpa_alone():
    """The measured 1.90x / 5.7x, reproduced from the shipped formula."""
    from gw.ppm_windows import (crossing_regularization_floor,
                                resolve_sigma_regularization)

    for half_ev, expect_ev in ((5.0, 0.4762), (15.0, 1.4286)):
        grid = _omega_grid_ry(half_ev)
        ppm = resolve_sigma_regularization(
            requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=grid,
            edge_factor=1.5, ansatz="gn_ppm")
        mpa = resolve_sigma_regularization(
            requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=grid,
            edge_factor=1.5, ansatz="mpa")
        assert ppm.raised and not mpa.raised
        assert ppm.resolved_ev == pytest.approx(expect_ev, abs=1e-3)
        assert mpa.resolved_ev == pytest.approx(0.25, rel=1e-12)
        # The DRIFT this row exists for, quoted from the resolver itself.
        assert ppm.resolved_ry / mpa.resolved_ry == pytest.approx(
            expect_ev / 0.25, rel=1e-3)
        # ...and it is the same closed form the driver used before.
        assert ppm.floor_ry == pytest.approx(
            crossing_regularization_floor(
                float(np.max(np.abs(grid))), 1.5), rel=1e-12)
    # hl_ppm takes the same floor as gn_ppm: it is the same quadrature.
    hl = resolve_sigma_regularization(
        requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=_omega_grid_ry(5.0),
        edge_factor=1.5, ansatz="hl_ppm")
    assert hl.raised


def test_an_explicit_floor_equalises_xi_across_ansatze():
    """The knob the register asked for: one number, both ansaetze."""
    from gw.ppm_windows import resolve_sigma_regularization

    grid = _omega_grid_ry(15.0)
    kw = dict(requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=grid,
              edge_factor=1.5, floor_ev=0.8)
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
        requested_ry=3.0 / RYD_TO_EV, omega_grid_ry=_omega_grid_ry(5.0),
        edge_factor=1.5, ansatz="gn_ppm")
    assert r.resolved_ev == pytest.approx(3.0, rel=1e-12) and not r.raised


def test_the_log_line_is_the_same_sentence_for_both_ansatze():
    """A comparison can only assert equal xi if both runs print it alike."""
    from gw.ppm_windows import resolve_sigma_regularization

    grid = _omega_grid_ry(5.0)
    lines = [resolve_sigma_regularization(
        requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=grid,
        edge_factor=1.5, ansatz=a).describe() for a in ("gn_ppm", "mpa")]
    for line in lines:
        assert "Σ broadening ξ:" in line and "requested" in line
    assert "RAISED" in lines[0] and "RAISED" not in lines[1]


def test_a_bad_floor_refuses_rather_than_resolving_to_auto():
    from gw.ppm_windows import resolve_sigma_regularization

    with pytest.raises(ValueError):
        resolve_sigma_regularization(
            requested_ry=0.25 / RYD_TO_EV, omega_grid_ry=_omega_grid_ry(5.0),
            edge_factor=1.5, ansatz="gn_ppm", floor_ev=-1.0)


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
