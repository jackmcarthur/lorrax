"""``write_sigma_to_file`` emits ONE column layout per file.

The real-vs-complex decision used to be taken PER SCALAR against a hard-coded
1e-10: each row asked whether its OWN imaginary part cleared the threshold and
appended either an ``+x.xxxxxxi`` field or a filler.  A run where one band's
Sigma is measurably complex and its neighbours are not then wrote rows of two
different shapes in the same column — the malformed dump recorded in the
frontier artifacts.  The ``VH`` branch was worse: it had no filler at all, so
the trailing ``Eo=`` column moved horizontally from row to row.

A file format is a property of the FILE, so the question is asked once per
component over every element that will be written.  These cells pin that, and
each carries the case where it comes out FALSE.
"""
from __future__ import annotations

import numpy as np
import pytest

from file_io.sigma_output import write_sigma_to_file


NK, NB = 2, 4

#: Crystal coordinates for the NK rows.  Required by the writer (it
#: length-checks them against the Sigma rows) so every block can say
#: which k it is; these cells are about column layout, not the k-basis,
#: so any distinct pair does.
KPTS = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=np.float64)


def _diag_array(values):
    """``(NK, NB, NB)`` with ``values`` on the diagonal."""
    out = np.zeros((NK, NB, NB), dtype=np.complex128)
    idx = np.arange(NB)
    out[:, idx, idx] = np.asarray(values, dtype=np.complex128)
    return out


def _rows(path):
    """The data rows only (no header, no rulers, no blanks)."""
    return [ln for ln in path.read_text().splitlines()
            if ln.startswith("n=")]


def _one_hot_complex():
    """A diagonal where exactly ONE element has an imaginary part.

    This is the geometry the per-scalar decision could not write: every
    other row takes the other branch.
    """
    vals = np.zeros((NK, NB), dtype=np.complex128)
    vals += np.arange(NK * NB).reshape(NK, NB) * 0.25
    vals[1, 2] += 3.5e-3j          # far above 1e-10, one element only
    return vals


def test_one_complex_element_makes_every_row_of_that_column_complex(tmp_path):
    """The property: a column complex ANYWHERE is written complex EVERYWHERE."""
    out = tmp_path / "sigma.dat"
    write_sigma_to_file(_diag_array(_one_hot_complex()), filename=str(out),
                        kpoints_crys=KPTS)
    rows = _rows(out)
    assert len(rows) == NK * NB
    n_with_i = sum(1 for r in rows if "i" in r.split("sigSX=")[1])
    assert n_with_i == len(rows), (
        f"only {n_with_i} of {len(rows)} rows carry the imaginary field, so "
        f"the file has two row shapes in one column — the malformed dump")


def test_a_fully_real_component_carries_no_imaginary_field_anywhere(tmp_path):
    """THE RED TWIN of the cell above.

    If the writer simply always emitted the imaginary field, the first cell
    would pass while saying nothing.  A real run must produce a real column,
    or the marker means nothing.
    """
    out = tmp_path / "sigma_real.dat"
    vals = np.arange(NK * NB).reshape(NK, NB) * 0.25 + 0j
    write_sigma_to_file(_diag_array(vals), filename=str(out),
                        kpoints_crys=KPTS)
    for r in _rows(out):
        assert "i" not in r.split("sigSX=")[1], (
            f"a fully real sigSX column emitted an imaginary field: {r!r}")


@pytest.mark.parametrize("complex_somewhere", [True, False])
def test_every_row_has_the_same_width_with_every_column_present(
        tmp_path, complex_somewhere):
    """Constant row width is the thing a column parser actually depends on.

    Exercised with sigSX + sigCOH + sigTOT + VH + Eo all present, because the
    ``VH`` branch is the one that had no filler and therefore shifted ``Eo=``.
    """
    vals = _one_hot_complex() if complex_somewhere else (
        np.arange(NK * NB).reshape(NK, NB) * 0.25 + 0j)
    hv = np.arange(NK * NB).reshape(NK, NB) * 0.1 + 0j
    if complex_somewhere:
        hv = hv.astype(np.complex128)
        hv[0, 1] += 1.0e-2j
    out = tmp_path / "full.dat"
    write_sigma_to_file(
        _diag_array(vals), filename=str(out), kpoints_crys=KPTS,
        sigma_coh_kij_eV=_diag_array(vals * 0.5),
        hartree_kij_eV=_diag_array(hv),
        energies_dft_ev=np.arange(NK * NB).reshape(NK, NB) * 1.0)
    rows = _rows(out)
    widths = {len(r) for r in rows}
    assert len(widths) == 1, (
        f"rows have {len(widths)} different widths {sorted(widths)}; a "
        f"whitespace-splitting parser reads the wrong column")

    # ...and the Eo= column starts at the same offset on every row, which is
    # the specific thing the unpadded VH branch broke.
    offsets = {r.index("Eo=") for r in rows}
    assert len(offsets) == 1, (
        f"the Eo= column starts at {sorted(offsets)} on different rows")


def test_the_totals_column_decides_on_its_own_sum(tmp_path):
    """sigTOT is sigSX+sigCOH, so its complexity is its OWN question.

    Constructed so the parts cancel: both components are complex, the sum is
    real to well within the tolerance.  Deciding sigTOT's format from either
    part would print an imaginary field that is identically zero.
    """
    vals = np.arange(NK * NB).reshape(NK, NB) * 0.25 + 0j
    sx = vals + 2.0e-3j
    coh = vals - 2.0e-3j
    out = tmp_path / "cancel.dat"
    write_sigma_to_file(_diag_array(sx), filename=str(out),
                        kpoints_crys=KPTS,
                        sigma_coh_kij_eV=_diag_array(coh))
    rows = _rows(out)
    assert rows, "no rows written"
    for r in rows:
        tot = r.split("sigTOT=")[1]
        assert "i" not in tot, (
            f"sigTOT printed an imaginary field for a sum that cancels: "
            f"{tot!r}")
        # ...while both PARTS keep theirs, or the cell proves nothing.
        assert "i" in r.split("sigSX=")[1].split("sigCOH=")[0]


def test_bispinor_lorentz_columns_are_additive_and_scalar_schema_is_unchanged(
        tmp_path):
    sx = np.arange(NK * NB).reshape(NK, NB) * 0.25 - 2.0
    coh = np.arange(NK * NB).reshape(NK, NB) * 0.05 + 0.1
    total = sx + coh
    parts = np.stack((0.9 * total, 0.025 * total, 0.075 * total))
    scalar = tmp_path / "scalar.dat"
    bispinor = tmp_path / "bispinor.dat"
    kwargs = dict(
        sigma_coh_kij_eV=_diag_array(coh),
        hartree_kij_eV=_diag_array(np.zeros_like(total)),
        kpoints_crys=KPTS,
    )
    write_sigma_to_file(_diag_array(sx), filename=str(scalar), **kwargs)
    write_sigma_to_file(
        _diag_array(sx), filename=str(bispinor),
        sigma_lorentz_skn_eV=parts, **kwargs)

    scalar_rows = _rows(scalar)
    assert all("sigCC=" not in row and "sigTT=" not in row
               and "sigCT=" not in row for row in scalar_rows)
    for row in _rows(bispinor):
        assert row.index("sigCC=") < row.index("sigTT=") < row.index("sigCT=")


def test_lorentz_column_closure_refuses_a_bad_decomposition(tmp_path):
    values = np.ones((NK, NB))
    bad = np.zeros((3, NK, NB))
    with pytest.raises(ValueError, match="sigma_lorentz_column_closure"):
        write_sigma_to_file(
            _diag_array(values), filename=str(tmp_path / "bad.dat"),
            sigma_coh_kij_eV=_diag_array(values),
            sigma_lorentz_skn_eV=bad,
            kpoints_crys=KPTS)
