"""A QP WFN says so on disk, and ``--eqp`` on top of one is refused.

THE DEFECT.  ``file_io.qp_wfn.write_qp_wfn_h5`` writes a BGW-compatible
WFN.h5 whose ψ and E are a **matched pair**: the rotated orbitals carry the
QP eigenvalues that produced the rotation.  ``exciton_bands --eqp`` supplies
a diagonal ladder written against **DFT band labels**.  Applying the second
on top of the first discards the canonical eigenvalues and relabels the
rotated orbitals with the mean-field ordering — and every array keeps the
right shape, so nothing downstream notices.

Measured on the MoS2 run-82 parent smoke, JID 57269074 step ``.128``
(``00_lorrax_gw/exciton_parent_smoke.in:9`` selects ``WFN_qp.h5``; the
wrapper passes ``--eqp eqp1.dat``).  The shape guard already in
``exciton_bands`` sees this only after a ``bse_k_grid`` densification has
moved ψ to a different k axis; without densification the two tables are the
same size and the overwrite is silent.

WHY A STAMP AND NOT A FILENAME.  Before this, the only available
discriminator was the string ``_qp`` in a path, which is not a fact about
the contents.  ``write_qp_wfn_h5`` now stamps ``qp_wfn_scheme``; the
refusal reads it back.

THE ABSENCE IS NOT A "NO".  A ``pw2bgw`` WFN and a WFN written before the
stamp both return ``None``, which means UNVERIFIABLE.  Only a POSITIVE
identification refuses — an absence is a claim about what was searched
(``TASTE.md``), and refusing on it would break every mean-field deck in the
tree.  The three cells below are that trichotomy; without the ``None`` arm
the refusal would be indistinguishable from a blanket ban on ``--eqp``.
"""
from __future__ import annotations

import os

import pytest

h5py = pytest.importorskip("h5py")


def _deck(tmp_path, wfn_name):
    deck = tmp_path / "cohsex.in"
    deck.write_text(f"[cohsex]\nnval = 4\nncond = 4\nwfn_file = {wfn_name}\n")
    return str(deck)


def _stamped_qp_wfn(path, *, scheme=None, band_start=8, band_stop=40):
    from file_io.qp_wfn import QP_WFN_ATTR, QP_WFN_SCHEME

    with h5py.File(str(path), "w") as f:
        f.create_dataset("mf_header/kpoints/nrk", data=1)
        f.attrs[QP_WFN_ATTR] = QP_WFN_SCHEME if scheme is None else scheme
        f.attrs["qp_wfn_band_start"] = int(band_start)
        f.attrs["qp_wfn_band_stop"] = int(band_stop)
        f.attrs["qp_wfn_source"] = "WFN.h5"


def _mean_field_wfn(path):
    """What ``pw2bgw`` writes: no stamp of any kind."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("mf_header/kpoints/nrk", data=1)


# ---------------------------------------------------------------------------
# The stamp reader's three answers
# ---------------------------------------------------------------------------

def test_the_reader_identifies_a_stamped_qp_wfn(tmp_path):
    from file_io.qp_wfn import QP_WFN_SCHEME, read_qp_wfn_stamp

    p = tmp_path / "WFN_qp.h5"
    _stamped_qp_wfn(p)
    stamp = read_qp_wfn_stamp(p)
    assert stamp is not None
    assert stamp["scheme"] == QP_WFN_SCHEME
    assert (stamp["band_start"], stamp["band_stop"]) == (8, 40)


def test_an_unstamped_wfn_is_unverifiable_not_mean_field(tmp_path):
    """The arm that keeps every existing deck running.

    A pw2bgw WFN carries no stamp, and so does a QP WFN written before the
    stamp existed.  The reader must not pretend to tell them apart.
    """
    from file_io.qp_wfn import read_qp_wfn_stamp

    p = tmp_path / "WFN.h5"
    _mean_field_wfn(p)
    assert read_qp_wfn_stamp(p) is None
    assert read_qp_wfn_stamp(tmp_path / "does_not_exist.h5") is None


def test_a_foreign_scheme_is_reported_verbatim(tmp_path):
    """A future scheme must be visible as a different word, not mapped."""
    from file_io.qp_wfn import QP_WFN_SCHEME, read_qp_wfn_stamp

    p = tmp_path / "WFN_qp.h5"
    _stamped_qp_wfn(p, scheme="lorrax-qp-wfn-v99")
    stamp = read_qp_wfn_stamp(p)
    assert stamp["scheme"] == "lorrax-qp-wfn-v99" != QP_WFN_SCHEME


# ---------------------------------------------------------------------------
# The refusal, both arms
# ---------------------------------------------------------------------------

def test_eqp_on_a_qp_wfn_refuses_by_name(tmp_path):
    """RED arm.  The message must say WHICH file and WHAT to do."""
    from bse.bse_window import refuse_eqp_on_a_qp_wfn

    _stamped_qp_wfn(tmp_path / "WFN_qp.h5")
    deck = _deck(tmp_path, "WFN_qp.h5")
    with pytest.raises(ValueError) as exc:
        refuse_eqp_on_a_qp_wfn(deck, "eqp1.dat")
    msg = str(exc.value)
    assert "WFN_qp.h5" in msg and "eqp1.dat" in msg
    assert "drop --eqp" in msg
    assert "DFT band LABELS" in msg, (
        "the refusal must name the MECHANISM (a ladder written against DFT "
        "labels), not merely assert redundancy")


def test_eqp_on_a_mean_field_wfn_is_the_supported_route(tmp_path):
    """GREEN arm, and the one that makes the red arm evidence.

    Refusing on the absent stamp would ban ``--eqp`` outright — the whole
    production route of a mean-field WFN plus diagonal QP corrections.
    """
    from bse.bse_window import refuse_eqp_on_a_qp_wfn

    _mean_field_wfn(tmp_path / "WFN.h5")
    deck = _deck(tmp_path, "WFN.h5")
    refuse_eqp_on_a_qp_wfn(deck, "eqp1.dat")        # must not raise


def test_a_foreign_stamp_still_refuses_and_says_the_version(tmp_path):
    """A file written by a different QP writer version is still a QP WFN.

    Accepting it because the version string did not match would be the
    superseded-convention trap in ``TASTE.md`` one index over.
    """
    from bse.bse_window import refuse_eqp_on_a_qp_wfn

    _stamped_qp_wfn(tmp_path / "WFN_qp.h5", scheme="lorrax-qp-wfn-v99")
    deck = _deck(tmp_path, "WFN_qp.h5")
    with pytest.raises(ValueError, match="different version"):
        refuse_eqp_on_a_qp_wfn(deck, "eqp1.dat")


def test_the_writer_stamps_what_the_reader_reads():
    """One contract, checked over the SOURCE so the two cannot drift apart.

    A round trip through ``write_qp_wfn_h5`` needs a full WFN plus the FFI
    loader; this cell is deliberately the cheap half — that the writer sets
    the attribute the reader keys on, and does it in the writer rather than
    at a call site.
    """
    import ast

    src = os.path.join(os.path.dirname(__file__), "..", "src", "file_io",
                       "qp_wfn.py")
    tree = ast.parse(open(src, encoding="utf8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "write_qp_wfn_h5")
    body = ast.unparse(fn)
    assert "QP_WFN_ATTR" in body and "QP_WFN_SCHEME" in body, (
        "write_qp_wfn_h5 no longer stamps its output; the eqp refusal would "
        "silently stop firing and every QP WFN would read as mean-field")
    for extra in ("qp_wfn_band_start", "qp_wfn_band_stop"):
        assert extra in body, extra
