"""The GN/HL plasmon-pole Sigma refuses a metallic occupation table.

WHY THE MATERIAL DOOR IS NOT ENOUGH.  ``gw_config.validate_material_inputs``
refuses GN-PPM on fractional WFN occupations (``GATE gn_ppm_refuses_metals``,
owner ruling 2026-09-17), but a WFN written with integer occupations can still
carry a band crossing the step table's Fermi level.  Whether a band crosses
E_F is a property of the SPECTRUM, and only that can be measured here.  There
is no override: the former ``LORRAX_PPM_ALLOW_CROSSING_BANDS`` hatch was the
one way to run GN-PPM on a metal and was removed with the ruling.

WHAT THE DRIVER DOES IF IT RUNS ANYWAY.  ``_prepare_sigma_state`` derives
``vbm = max(enk | occupied)`` and ``cbm = min(enk | empty)``.  With a
crossing band ``vbm > cbm``, so the "midgap" ``0.5*(vbm+cbm)`` is not in any
gap, and it then clips ``E_cond = max(enk - efermi, 0)`` and
``H_val = max(efermi - enk, 0)`` -- a band on the wrong side of that
pseudo-Fermi level cannot even be REPRESENTED.  Every array keeps its shape
and the run exits 0.  This is the ``TODO(metal-greens)`` scope limit the
module has documented in prose since it was written; a documented limitation
that nothing enforces is a limitation only the reader has.

THE PREDICATE NEEDS NO TOLERANCE.  ``wavefunction_bundle._build_occ`` fills
``occ`` as ``(enk <= efermi)``, exactly 0.0/1.0, so "band n is occupied at
some k and empty at others" is a statement about an integer table.

Evidence for the failure this gate stops: Na bcc SOC 48b,
``reports/occupation_threshold_all_paths_2026-08-16/evidence/probe_na.log``
(JID 57138992) -- MP1 gives f in [-0.002194, +1.031587] with 150
negative-lobe and 12 over-one (k,n) entries, and the step split assigns
every one of them fully to one branch with weight 1.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from gw.ppm_sigma import (                                       # noqa: E402
    assert_gapped_occupations_for_ppm,
)

_REMOVED_OVERRIDE = "LORRAX_PPM_ALLOW_CROSSING_BANDS"


def _gapped(nk=6, nb=10, n_occ=4):
    """Every band uniformly occupied or uniformly empty -- an insulator."""
    occ = np.zeros((nk, nb), dtype=float)
    occ[:, :n_occ] = 1.0
    return occ


def _metallic(nk=6, nb=10, n_occ=4, crossing=(4,)):
    """One band occupied at half the k and empty at the other half."""
    occ = _gapped(nk, nb, n_occ)
    for b in crossing:
        occ[: nk // 2, b] = 1.0
        occ[nk // 2 :, b] = 0.0
    return occ


def test_a_gapped_table_passes_and_reports_zero(monkeypatch):
    """GREEN arm, and the one that makes the red arm evidence.

    Every insulating deck in the tree runs through this call.  It returns a
    NUMBER rather than merely not raising, so "checked and clean" and "never
    reached" are distinguishable.
    """
    monkeypatch.delenv(_REMOVED_OVERRIDE, raising=False)
    assert assert_gapped_occupations_for_ppm(
        _gapped(), print_fn=lambda *_a: None) == 0


def test_a_crossing_band_refuses_by_name_and_says_which_band(monkeypatch):
    monkeypatch.delenv(_REMOVED_OVERRIDE, raising=False)
    with pytest.raises(ValueError) as exc:
        assert_gapped_occupations_for_ppm(
            _metallic(crossing=(4,)), print_fn=lambda *_a: None)
    msg = str(exc.value)
    assert "GATE ppm_sigma_gapped_occupations" in msg
    assert "[4]" in msg, msg
    # It must name the SUPPORTED route, not merely say no.
    assert "compute_mode = mpa" in msg and "sigma_w_model = shared_pole" in msg
    assert "owner ruling 2026-09-17" in msg
    # and the mechanism, so the reader can tell this from a shape complaint
    assert "vbm > cbm" in msg


def test_several_crossing_bands_are_all_counted(monkeypatch):
    monkeypatch.delenv(_REMOVED_OVERRIDE, raising=False)
    with pytest.raises(ValueError, match=r"3 Fermi-crossing"):
        assert_gapped_occupations_for_ppm(
            _metallic(nb=12, crossing=(3, 4, 5)), print_fn=lambda *_a: None)


def test_the_removed_override_no_longer_admits_a_crossing_spectrum(monkeypatch):
    """Owner ruling 2026-09-17: GN-PPM is not allowed for metals, so the
    former debugging hatch refuses exactly like the unset case."""
    monkeypatch.setenv(_REMOVED_OVERRIDE, "1")
    with pytest.raises(ValueError, match="GATE ppm_sigma_gapped_occupations"):
        assert_gapped_occupations_for_ppm(
            _metallic(crossing=(4,)), print_fn=lambda *_a: None)


def test_the_gate_is_called_before_the_state_prep_that_needs_it():
    """CALL-CHAIN PIN.  The gate protects ``_prepare_sigma_state``'s derived
    Fermi level, so it has to run BEFORE it -- a gate placed after the
    quantity it guards measures a value already computed from bad inputs."""
    import ast
    import os

    src_path = os.path.join(os.path.dirname(__file__), "..", "src", "gw",
                            "ppm_sigma.py")
    src = open(src_path, encoding="utf8").read()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "compute_sigma_c_ppm_omega_grid")
    body = ast.unparse(fn)
    assert "assert_gapped_occupations_for_ppm(" in body, (
        "the driver no longer calls the gate; a metallic deck would reach "
        "the 0/1 band split again with nothing objecting")
    assert body.index("assert_gapped_occupations_for_ppm(") < body.index(
        "_prepare_sigma_state("), (
        "the gate must run before the Fermi level it protects is derived")
