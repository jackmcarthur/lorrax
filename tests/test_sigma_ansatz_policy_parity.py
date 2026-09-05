"""ONE owner for each policy the PPM and MPA Sigma branches both need.

Three sibling drifts, one file, because they are the same defect three
times: a rule that one ansatz enforced and its sibling did not.

**(a) The sharded-cube carrier.** Both ansaetze now ask
``runtime.padding`` for the same square mesh carrier. An indivisible logical
window is represented by exact-zero rows and never published with an illegal
sharding.

**(b) The effective Sigma broadening xi.**  GN-PPM silently raised the
deck's ``sigma_regularization_ev`` to a window-dependent conditioning
floor; MPA passed it straight through.  1.90x apart on a +/-5 eV grid,
5.7x on +/-15 eV, and the resolved value appeared in no artifact.  Since
2026-09-02 (owner ruling) the floor is gone: one resolver
(``ppm_windows.resolve_sigma_regularization``) returns the deck's value
for every ansatz, and one stamp records it.

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
# (a) one mesh-carrier owner
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh22():
    d = jax.devices()
    if len(d) < 4:
        pytest.skip("needs 4 devices for a 2x2 mesh")
    return Mesh(np.array(d[:4]).reshape(2, 2), ("x", "y"))


def test_both_ansatze_call_the_same_sharded_window_owner():
    """Source gate: neither executor may re-derive divisibility arithmetic.

    A source cell rather than a run cell because reaching the MPA executor
    needs a pole store on disk.  The point being pinned is structural --
    that there is ONE function and both branches call it -- and that is
    exactly what a source cell can see.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "gw"
    mpa = (root / "mpa" / "sigma.py").read_text()
    ppm = (root / "ppm_sigma.py").read_text()
    assert "sigma_band_axis(" in mpa
    assert ppm.count("def sigma_band_axis") == 1
    assert "assert_sharded_sigma_window_divides_mesh" not in ppm
    # And the rule is not ALSO spelled out inline in the MPA module.
    mpa_code = "\n".join(l.split("#", 1)[0] for l in mpa.splitlines())
    assert "% p_x" not in mpa_code and "% p_y" not in mpa_code


def test_an_indivisible_window_gets_the_same_carrier_for_both_ansatze(mesh22):
    from gw.ppm_sigma import sigma_band_axis

    for nb in (7, 9, 15):
        tags = [sigma_band_axis(nb, mesh22, ansatz=ansatz)
                for ansatz in ("gn_ppm", "compute_mode = mpa")]
        assert [(t.logical, t.carrier, t.divisor) for t in tags] == [
            (nb, nb + 1, 2), (nb, nb + 1, 2)]


# ---------------------------------------------------------------------------
# (b) one effective-xi resolver
# ---------------------------------------------------------------------------

def _omega_grid_ry(half_width_ev):
    half = half_width_ev / RYD_TO_EV
    return np.linspace(-half, half, 41)


def test_xi_is_literal_for_every_ansatz():
    """Owner ruling 2026-09-02: the deck's eta IS the broadening.  The HGL
    conditioning floor (1.90x on +-5 eV, 5.7x on +-15 eV) went with the HGL
    executor it belonged to; nothing raises xi for any ansatz."""
    from gw.ppm_windows import resolve_sigma_regularization

    for ansatz in ("gn_ppm", "hl_ppm", "mpa"):
        r = resolve_sigma_regularization(
            requested_ry=0.25 / RYD_TO_EV, ansatz=ansatz)
        assert not r.raised and r.floor_ry == 0.0
        assert r.floor_policy == "literal"
        assert r.resolved_ev == pytest.approx(0.25, rel=1e-12)
    with pytest.raises(ValueError):
        resolve_sigma_regularization(requested_ry=0.0, ansatz="mpa")


def test_the_log_line_is_the_same_sentence_for_both_ansatze():
    """A comparison can only assert equal xi if both runs print it alike."""
    from gw.ppm_windows import resolve_sigma_regularization

    grid = _omega_grid_ry(5.0)
    del grid
    lines = [resolve_sigma_regularization(
        requested_ry=0.25 / RYD_TO_EV, ansatz=a).describe() for a in ("gn_ppm", "mpa")]
    for line in lines:
        assert "Σ broadening ξ:" in line and "requested" in line
    assert lines[0].replace("gn_ppm", "mpa") == lines[1]


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
