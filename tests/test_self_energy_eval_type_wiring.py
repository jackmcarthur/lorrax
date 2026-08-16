"""``self_energy_eval_type`` must SELECT a reporting path, not describe one.

THE DEFECT THIS EXISTS TO PREVENT, stated exactly.  Between 2026-08-15 and
2026-08-16 the key was parsed, validated, documented and refused on the one
bad pairing -- and consumed by NOTHING.  ``git grep self_energy_eval_type``
on ``feat/staged-sc-2026-08-15`` returned two files: ``gw/gw_config.py``,
which defines it, and ``tests/test_staged_sc.py``, which tests the
definition.  No solver, writer or driver read it.  Choosing ``hermitianized``
over ``linearized`` changed no number in any output file, because
self-consistency already ran the hermitianized path and one-shot already ran
the linearized one; the key DESCRIBED the behaviour its neighbours had
already fixed instead of SELECTING it.

That is a silent defect: every unit test of the enum passed, the refusal
worked, the documentation was accurate, and the one configuration the key
was introduced to unlock -- ONE-SHOT x HERMITIANIZED, a single-shot
QSGW-style rediagonalisation with no self-consistency -- was unreachable.

So the cells below do not test the enum.  ``tests/test_staged_sc.py`` does
that, and it kept passing throughout the defect.  These test that the
RESOLVED VALUE REACHES THE NUMBERS ON DISK:

1.  the two values produce DIFFERENT ``eqp0.dat`` from one identical
    ``GWResults`` -- the cell that fails the moment the key stops selecting
    anything, which is the whole point of this file;
2.  ``hermitianized`` reports the H_qp EIGENVALUES specifically, not merely
    something-different (a writer that emitted zeros would pass cell 1);
3.  ``hermitianized`` writes ``eqp1 == eqp0``, because off-diagonal Sigma has
    already moved the bands and there is no Newton step left to
    Z-linearize;
4.  the DEFAULT container reports exactly what ``linearized`` reports, byte
    for byte -- the bit-identity contract for every deck that never names
    the key.

Everything runs on synthetic arrays -- no WFN, no GPU, no jit.  The driver
seam (``gw_jax`` reading ``config.self_energy_eval_type`` and the
band-window guard) is exercised on the cluster; see the lane report.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.gw_config import SelfEnergyEvalType
from gw.gw_output import GWResults, write_results


_NK = 2
_NB = 3
_KPTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
_KIRR_TO_KFULL = np.array([0, 1], dtype=np.int32)


def _synthetic_results(eval_type=None) -> GWResults:
    """One Sigma, one H_qp, two possible reports of it.

    Built so the two arms CANNOT agree by accident: ``sig_sx`` carries a
    real off-diagonal block, so the eigenvalues of H = kin_ion + Sigma are
    pushed away from its diagonal by level repulsion.  A wiring that
    silently fell back to the linearized formula would have to reproduce
    that repulsion to pass, and it cannot.
    """
    e_dft_ry = np.array([[0.10, 0.40, 0.95],
                         [0.15, 0.45, 0.90]], dtype=np.float64)

    kin_ion = np.zeros((_NK, _NB, _NB), dtype=np.complex128)
    for k in range(_NK):
        kin_ion[k] = np.diag(e_dft_ry[k]).astype(np.complex128)

    # Screened exchange: a diagonal shift PLUS the off-diagonal block that
    # makes hermitianized a different physical answer from linearized.
    sig_sx = np.zeros((_NK, _NB, _NB), dtype=np.complex128)
    for k in range(_NK):
        sig_sx[k] = np.diag(np.array([-0.30, -0.22, -0.11]))
        sig_sx[k, 0, 1] = sig_sx[k, 1, 0] = 0.08 + 0.01 * k
        sig_sx[k, 1, 2] = sig_sx[k, 2, 1] = 0.05 - 0.01 * k

    sig_coh = np.zeros((_NK, _NB, _NB), dtype=np.complex128)
    for k in range(_NK):
        sig_coh[k] = np.diag(np.array([0.06, 0.04, 0.03]))

    sig_h = np.zeros((_NK, _NB, _NB), dtype=np.complex128)
    sig_x = np.zeros((_NK, _NB, _NB), dtype=np.complex128)

    # E_qp_ry as the DRIVER computes it: eigh of the hermitianized
    # H = kin_ion + Sigma_xc + V_H.  Same formula as gw_jax's H-build seam.
    H = kin_ion + sig_sx + sig_coh + sig_h
    H = 0.5 * (H + np.conj(np.swapaxes(H, -1, -2)))
    e_qp_ry = np.linalg.eigvalsh(H)

    kwargs = {}
    if eval_type is not None:
        kwargs["eval_type"] = eval_type
    return GWResults(
        sig_sx=sig_sx, sig_coh=sig_coh, sig_h=sig_h, sig_x=sig_x,
        E_qp_ry=e_qp_ry,
        U_qp=np.zeros((_NK, _NB, _NB), dtype=np.complex128),
        E_dft_ry=e_dft_ry,
        kin_ion_ry=kin_ion,
        band_start=0, band_stop=_NB,
        use_ppm=False, self_consistent=False,
        **kwargs)


def _write(tmp_path, eval_type, tag: str) -> dict[str, str]:
    """Run the real writer; return the paths it produced."""
    d = tmp_path / tag
    d.mkdir()
    paths = {
        "sigma_diag": str(d / "sigma_diag.dat"),
        "eqp0": str(d / "eqp0.dat"),
        "eqp1": str(d / "eqp1.dat"),
    }
    write_results(
        _synthetic_results(eval_type),
        sigma_diag_file=paths["sigma_diag"],
        eqp0_file=paths["eqp0"],
        eqp1_file=paths["eqp1"],
        input_dir=str(d),
        kpoints_crys=_KPTS,
        kgrid=(2, 1, 1),
        kpoints_irr_frac=_KPTS,
        kirr_to_kfull=_KIRR_TO_KFULL,
        # h5py is not a test dependency here, and the rotations file is not
        # what this module is about.
        write_qp_rotations=False,
        print_fn=lambda *a, **k: None,
    )
    return paths


def _body(path: str) -> str:
    """File contents WITHOUT the provenance stamp.

    ``common.provenance.provenance_header`` writes a UTC timestamp, so two
    runs of identical physics are never literally byte-identical.  Every
    identity claim in this module is therefore over the DATA BODY -- every
    line the writer emits that is not the generated-at stamp.
    """
    with open(path) as fh:
        return "".join(
            ln for ln in fh if not ln.startswith("# Generated by LORRAX"))


def _parse_eqp(path: str) -> np.ndarray:
    """Read the E_QP column back out of a BGW eqp{0,1}.dat.

    BOTH line kinds carry four fields -- the per-k header is
    ``(kx, ky, kz, nspin*nb)`` and the band row is
    ``(ispin, iband, E_DFT, E_QP)`` -- so the field COUNT cannot tell them
    apart.  The band row is the one whose first two fields are integers.
    """
    rows: list[float] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            if "." in parts[0] or "." in parts[1]:    # k-point header
                continue
            rows.append(float(parts[3]))              # ispin iband E_DFT E_QP
    return np.array(rows, dtype=np.float64).reshape(_NK, _NB)


# ---------------------------------------------------------------------------
# 1. THE CELL THIS FILE EXISTS FOR
# ---------------------------------------------------------------------------

def test_the_key_selects_something(tmp_path):
    """linearized and hermitianized must not agree on eqp0.dat.

    THE REGRESSION GUARD.  If this passes trivially -- i.e. if the two
    bodies are equal -- the key has stopped selecting a reporting path and
    is once again a validated no-op, which is the exact defect of
    2026-08-15.  No tolerance is involved: the arms are different formulas
    applied to the same Sigma, so they differ or the wiring is gone.
    """
    lin = _write(tmp_path, SelfEnergyEvalType.LINEARIZED, "lin")
    herm = _write(tmp_path, SelfEnergyEvalType.HERMITIANIZED, "herm")

    assert _body(lin["eqp0"]) != _body(herm["eqp0"]), (
        "self_energy_eval_type selected NOTHING: linearized and "
        "hermitianized produced identical eqp0.dat from one Sigma.  The "
        "key is a no-op again -- see this module's docstring.")


def test_hermitianized_reports_the_rediagonalized_eigenvalues(tmp_path):
    """Different is not enough -- it must be the EIGENVALUES.

    Cell 1 would also pass for a writer that emitted zeros, or E_DFT, or
    the linearized numbers with a sign error.  This pins the actual
    quantity: the eigenvalues of H_qp that the driver already computed.
    """
    herm = _write(tmp_path, SelfEnergyEvalType.HERMITIANIZED, "herm")
    expected = _synthetic_results().E_qp_ry * RYD_TO_EV
    got = _parse_eqp(herm["eqp0"])
    # 9 decimal places in the file; compare at the format's own precision.
    assert np.allclose(got, expected, atol=1e-8, rtol=0.0), (
        f"hermitianized eqp0.dat is not the H_qp spectrum.\n"
        f"  got      {got}\n  expected {expected}")


def test_hermitianized_moves_the_bands_by_the_off_diagonal_sigma(tmp_path):
    """The DIRECTION of the difference is the off-diagonal Sigma.

    With the diagonal held fixed, level repulsion pushes the outer
    eigenvalues apart relative to the linearized (diagonal-only) answer.
    Pinning the SIGN of that spread keeps a future refactor from passing
    cell 1 by reporting some other array that merely happens to differ.
    """
    lin = _parse_eqp(_write(tmp_path, SelfEnergyEvalType.LINEARIZED,
                            "lin")["eqp0"])
    herm = _parse_eqp(_write(tmp_path, SelfEnergyEvalType.HERMITIANIZED,
                             "herm")["eqp0"])
    # Bottom band pushed down, top band pushed up, at every k.
    assert np.all(herm[:, 0] < lin[:, 0]), (herm[:, 0], lin[:, 0])
    assert np.all(herm[:, -1] > lin[:, -1]), (herm[:, -1], lin[:, -1])
    # And the total spread must widen -- the signature of level repulsion.
    assert np.all((herm[:, -1] - herm[:, 0]) > (lin[:, -1] - lin[:, 0]))


def test_hermitianized_writes_eqp1_equal_to_eqp0(tmp_path):
    """No Newton step survives a rediagonalisation, so there is no twin.

    Not a placeholder: the static-COHSEX linearized path writes the
    identical pair for the same structural reason (Z = 1).  Downstream
    ``eqp1.dat`` consumers (the BSE reads it) therefore keep working.
    """
    herm = _write(tmp_path, SelfEnergyEvalType.HERMITIANIZED, "herm")
    assert _body(herm["eqp0"]) == _body(herm["eqp1"])


# ---------------------------------------------------------------------------
# 2. THE BIT-IDENTITY CONTRACT
# ---------------------------------------------------------------------------

def test_default_container_is_linearized_byte_for_byte(tmp_path):
    """A deck that never names the key must produce the historical files.

    The field defaults to LINEARIZED and the writer's hermitianized branch
    is not entered, so the default path runs UNTOUCHED code rather than
    re-derived numbers.  This cell pins that as an observable property of
    the bytes, over eqp0, eqp1 AND sigma_diag.
    """
    default = _write(tmp_path, None, "default")
    explicit = _write(tmp_path, SelfEnergyEvalType.LINEARIZED, "explicit")
    for name in ("eqp0", "eqp1", "sigma_diag"):
        assert _body(default[name]) == _body(explicit[name]), name


def test_sigma_diag_is_identical_across_both_arms(tmp_path):
    """eval_type is a REPORTING axis: it must not perturb Sigma itself.

    ``sigma_diag.dat`` is the Sigma-decomposition dump.  Both arms built
    the same Sigma from the same inputs, so this file must not move.  If
    it ever does, the key has grown a side effect on the physics and is no
    longer the orthogonal third axis its docstring claims.
    """
    lin = _write(tmp_path, SelfEnergyEvalType.LINEARIZED, "lin")
    herm = _write(tmp_path, SelfEnergyEvalType.HERMITIANIZED, "herm")
    assert _body(lin["sigma_diag"]) == _body(herm["sigma_diag"])


# ---------------------------------------------------------------------------
# 3. THE BAND-WINDOW GUARD MUST BE ABLE TO FIRE
# ---------------------------------------------------------------------------
#
# A guard that cannot fire is the same defect as a key that selects
# nothing, one level down -- and the first draft of this wiring had
# exactly that bug.  It handed ``check_band_window`` the ACTIVE-WINDOW
# energies (``enk_dft``, whose columns are 0..b3-b0) together with GLOBAL
# edges b0/b3.  With b0 == 0 -- the normal case, and the only one the SC
# driver permits -- b_min lands on 0 and b_max lands on nb, both outside
# the guard's own ``0 < b < nb`` test, so EVERY window passed without
# being looked at.  The fix is the array, not the guard: full-band
# energies (``wfn.energies[0]``), against which a global edge means what
# it says.  These two cells pin both halves.

def _degenerate_deck_energies() -> np.ndarray:
    """(nk, nbands) Ry with an EXACTLY degenerate pair straddling band 4."""
    e = np.array([
        [0.00, 0.10, 0.20, 0.30, 0.40, 0.40, 0.70, 0.90],
        [0.02, 0.12, 0.22, 0.32, 0.45, 0.45, 0.72, 0.92],
    ], dtype=np.float64)
    return e                      # bands 4 and 5 are degenerate at every k


def test_the_window_guard_fires_on_a_sliced_multiplet():
    """Global edges against FULL-BAND energies: the edge at 5 must refuse."""
    from common.band_degeneracy import (BandWindowDegeneracyError,
                                        check_band_window)
    e = _degenerate_deck_energies()
    with pytest.raises(BandWindowDegeneracyError):
        # Window [0, 5) cuts the 4/5 pair in half.
        check_band_window(e, 0, 5, mode="strict", log=lambda *a, **k: None)


def test_the_window_guard_is_a_noop_on_the_active_window_array():
    """THE BUG, pinned so it cannot come back.

    Same physical window, but described against the ACTIVE-WINDOW array
    the way the first draft did it: the array is only as wide as the
    window, so the upper edge coincides with its width and the guard
    treats it as cutting nothing.  It passes -- which is precisely why
    the driver must hand it the FULL-band energies instead.
    """
    from common.band_degeneracy import check_band_window
    e = _degenerate_deck_energies()
    sliced = e[:, 0:5]            # what `enk_dft` looks like for [0, 5)
    # No exception: b_max == nb is exempt, so the sliced multiplet at the
    # boundary is invisible.  Documenting the no-op IS the regression test.
    check_band_window(sliced, 0, 5, mode="strict", log=lambda *a, **k: None)


def test_the_driver_hands_the_guard_full_band_energies():
    """Structural: the driver's guard must read ``wfn.energies``.

    Pinned by source inspection rather than by a full run, because
    reaching this seam otherwise costs a WFN, a mean field and four GPUs.
    If the array ever reverts to ``enk_dft`` the guard silently stops
    guarding, and no numerical test would notice.

    READ AS TEXT, not imported: importing ``gw.gw_jax`` brings up the
    runtime stack and the FFI gate refuses on a CPU cell, which would
    turn this into an environment test instead of a source contract.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "gw_jax.py").read_text()
    assert "check_band_window(" in src, "the guard is gone entirely"
    guard = src.split("check_band_window(")[-1].split(",")[0]
    assert "_enk_ry" in guard, f"guard reads {guard.strip()!r}"
    assign = [ln for ln in src.splitlines() if "_enk_ry =" in ln]
    assert assign, "no _enk_ry assignment in gw_jax"
    assert "wfn.energies" in assign[0], (
        f"the hermitianized band-window guard is reading {assign[0].strip()!r}; "
        f"it must read the FULL-band table (wfn.energies), or a global "
        f"upper edge equal to the active window's width makes it a no-op.")


# ---------------------------------------------------------------------------
# 4. THE REFUSAL SURVIVES THE WIRING
# ---------------------------------------------------------------------------

BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra: str = ""):
    from gw.gw_config import LorraxConfig
    path = tmp_path / "eval_type.in"
    path.write_text(BASE_INPUT + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


def test_linearized_under_self_consistency_still_refuses(tmp_path):
    """The one bad pairing stays an error BY NAME after the wiring.

    Duplicated from ``tests/test_staged_sc.py`` on purpose: that module
    tests the resolver, this one tests the axis end to end, and the
    refusal is the one property that must hold in both readings.
    """
    with pytest.raises(ValueError, match="self_energy_eval_type"):
        _config(tmp_path,
                "qp_solver = self_consistent\n"
                "self_energy_eval_type = linearized\n")


def test_the_driver_reads_the_axis_through_a_named_accessor(tmp_path):
    """``config.self_energy_eval_type`` is what the driver consumes.

    Pinned because the value's HOME is ``config.sc.eval_type``, on a
    record whose own docstring says it is read only under
    self-consistency.  A one-shot driver reaching into ``config.sc`` would
    read as a bug to the next person and invite a 'fix' that reverts the
    feature; the accessor is what makes the one-shot read legible.
    """
    assert _config(tmp_path).self_energy_eval_type is (
        SelfEnergyEvalType.LINEARIZED)
    assert _config(tmp_path, "self_energy_eval_type = hermitianized\n"
                   ).self_energy_eval_type is SelfEnergyEvalType.HERMITIANIZED
    # ONE-SHOT x HERMITIANIZED -- the configuration the key was introduced
    # to unlock, and the one that was unreachable while it selected nothing.
    cfg = _config(tmp_path, "self_energy_eval_type = hermitianized\n")
    from gw.gw_config import QPSolver
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT
    assert cfg.self_energy_eval_type is SelfEnergyEvalType.HERMITIANIZED
