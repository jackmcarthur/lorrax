"""``write_qsgw_datasets`` — the four datasets that had no producer.

WHAT THIS RESTORES.  ``sigma_xc_qsgw_kij_ev`` and the three ``qp_*``
ladders were written by the pre-2026-04 driver, deleted on 2026-04-11 by
the QP/output rewrite along with a pile of code that genuinely was dead,
and thereafter existed only in ``tests/regression/cohsex_debug/``.  The
k_irr landing found their names sitting in ``SIGMA_K_AXIS`` with nothing
on the other end; the owner's 2026-08-08 ruling is that the writer should
exist, gated by the deck, because "a lot of people will want to plot
that".

WHAT IS GATED HERE, and why in this shape.  The writer composes with the
landed k_irr extraction rather than sitting beside it, so what has to be
pinned is the composition:

1.  **The selection is a selection.**  The rows the appended datasets
    hold are the rows the full-BZ arrays held, ``array_equal`` and not
    ``allclose`` — the same standard ``test_sigma_kirr_extraction.py``
    holds the cubes to, for the same reason: a tolerance would pass a
    writer that had started doing arithmetic on the way out.
2.  **The file decides the k set, not the caller.**  ``append_qsgw_
    datasets_h5`` reads its own storage off the cubes already in the file
    and matches them.  A heterogeneous artifact — some datasets on the
    wedge, some on the full BZ, every one of them plausibly shaped — is
    the failure this design forecloses, and no reader in the tree checks
    across datasets.
3.  **The red twins.**  Key off writes nothing at all; a file whose
    tables and slab disagree refuses THROUGH ``kin_ion.read_star_map``,
    which already owns that refusal, rather than through a second copy of
    it here.
4.  **The plotting contract.**  A consumer reads the written file, learns
    its k-set through the landed reader path, and gets back something
    shaped, typed and finite.  That is the whole reason the datasets are
    coming back.

The cells build their files with plain ``h5py`` rather than through
``write_sigma_omega_h5``.  That is not a shortcut around ``SlabIO``: the
appended arrays are replicated by construction (``build_qsgw_sigma_xc``
pins its result replicated before it Hermitises; the ladders are host
eigenvalues), so the append genuinely is a rank-0 serial write, and
gating it behind a library this box does not have would leave the
load-bearing claims untested everywhere.  The one cell that reads a real
artifact reads the committed fixture.
"""

from __future__ import annotations

import ast
import os
import pathlib
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from file_io.sigma_output import (                              # noqa: E402
    QSGW_PLOT_DATASETS,
    SIGMA_K_AXIS,
    SPREAD_ATTR_PREFIX,
    append_qsgw_datasets_h5,
    compact_star_tables,
    extract_and_stamp_k_irr,
    k_irr_rows_for,
)
from file_io.kin_ion import (                                   # noqa: E402
    K_STORAGE_ATTR,
    K_STORAGE_IBZ,
    K_STORAGE_VERSION,
    K_STORAGE_VERSION_ATTR,
    N_SYM_SPATIAL_ATTR,
    read_star_map,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"

#: ``cohsex_debug``'s topology, the same one
#: ``test_sigma_kirr_extraction.py`` types: 9 full-BZ k over 3 stars,
#: ntran 12, and an ``irr_idx_k`` whose labels are not contiguous.
_IRR = np.array([0, 2, 2, 2, 3, 2, 2, 2, 3])
_SIDX = np.array([0, 12, 0, 3, 0, 14, 15, 2, 1])
_NSS = 12
_NB = 4


def _payload(seed=0, nk=9, nb=_NB):
    """The four datasets, on the FULL BZ, as a producer hands them over."""
    rng = np.random.default_rng(seed)
    cube = (rng.standard_normal((nk, nb, nb))
            + 1j * rng.standard_normal((nk, nb, nb)))
    return {
        "sigma_xc_qsgw_kij_ev": 0.5 * (cube + np.conj(np.swapaxes(
            cube, -1, -2))),
        "qp_static_cohsex_ev": rng.standard_normal((nk, nb)),
        "qp_omega0_ev": rng.standard_normal((nk, nb)),
        "qp_diag_self_consistent_ev": rng.standard_normal((nk, nb)),
    }


def _make_file(path, *, on_ibz, nk_full=9, nb=_NB, n_omega=5):
    """A ``sigma_mnk.h5`` carrying just the cubes the real writer creates.

    ``on_ibz`` builds the stamped form: the cubes hold one row per star
    and the two unfold tables sit beside them, which is exactly what
    ``write_sigma_omega_h5(..., star=...)`` produces.
    """
    rows, compact = compact_star_tables(_IRR)
    nk = len(rows) if on_ibz else nk_full
    rng = np.random.default_rng(99)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("omega_ev", data=np.linspace(-2.0, 2.0, n_omega))
        for name, shape in (
                ("sigma_c_kij_ev", (n_omega, nk, nb, nb)),
                ("sigma_sx_kij_ev", (nk, nb, nb)),
                ("hartree_kij_ev", (nk, nb, nb))):
            ds = f.create_dataset(
                name, data=(rng.standard_normal(shape)
                            + 1j * rng.standard_normal(shape)))
            if on_ibz:
                ds.attrs[K_STORAGE_ATTR] = K_STORAGE_IBZ
                ds.attrs[K_STORAGE_VERSION_ATTR] = K_STORAGE_VERSION
                ds.attrs[N_SYM_SPATIAL_ATTR] = _NSS
                ds.attrs["nk_full"] = nk_full
        if on_ibz:
            f.create_dataset("irr_idx_k", data=compact.astype(np.int32))
            f.create_dataset("sym_idx_k", data=_SIDX.astype(np.int32))
    return str(path)


def _read(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k][()]) for k in f}


# ---------------------------------------------------------------------------
# 1.  The selection is a selection
# ---------------------------------------------------------------------------

def test_the_appended_rows_are_bit_identical_to_the_full_bz_rows(tmp_path):
    """THE gate.  Every one of the four, taken not computed.

    ``array_equal`` on every dataset, including the real ``qp_*`` ladders
    where a stray ``float32`` round trip or an accidental degeneracy
    average would still be ``allclose``.
    """
    pytest.importorskip("jax")
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    rows, _ = compact_star_tables(_IRR)
    src = _payload()

    written = append_qsgw_datasets_h5(path, dict(src))
    assert written == list(QSGW_PLOT_DATASETS), (
        "the appendix is written in one fixed order so a reader meets the "
        "datasets the same way on every run")

    got = _read(path)
    for name in QSGW_PLOT_DATASETS:
        expected = np.take(src[name], rows, axis=SIGMA_K_AXIS[name])
        assert np.array_equal(got[name], expected), (
            f"{name}: the appended rows are not bit-identical to the "
            f"full-BZ rows they were taken from — the appender is doing "
            f"arithmetic on the way out, which is reconstruction")


def test_a_full_bz_file_takes_the_datasets_whole(tmp_path):
    """The other arm: an unstamped file gets unstamped, unextracted rows.

    Without this the cell above could pass for an appender that extracted
    unconditionally, which would silently drop 6 of 9 k on every legacy
    run.
    """
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    src = _payload(seed=1)
    append_qsgw_datasets_h5(path, dict(src))

    got = _read(path)
    with h5py.File(path, "r") as f:
        for name in QSGW_PLOT_DATASETS:
            assert np.array_equal(got[name], src[name])
            assert K_STORAGE_ATTR not in f[name].attrs, (
                f"{name} carries a k_storage attr on a full-BZ file; "
                f"no-attr-means-full is the back-compat direction and the "
                f"appender must not add one it cannot honour")
            assert read_star_map(path, name,
                                 k_axis=SIGMA_K_AXIS[name]) is None


def test_the_spread_stats_reach_every_appended_dataset(tmp_path):
    """Same four numbers, same prefix, as the cubes beside them.

    The whole point of stamping at write time is that an extracted file
    cannot be measured afterwards.  A dataset that arrived without the
    stamp would be the one row of the file nobody could price.
    """
    pytest.importorskip("jax")
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    append_qsgw_datasets_h5(path, _payload(seed=2))
    keys = [SPREAD_ATTR_PREFIX + k for k in
            ("raw_ev", "diag_ev", "frobenius_ev", "trace_ev", "omega_index")]
    with h5py.File(path, "r") as f:
        for name in QSGW_PLOT_DATASETS:
            attrs = dict(f[name].attrs)
            assert attrs[K_STORAGE_ATTR] == K_STORAGE_IBZ
            assert int(attrs["nk_full"]) == _IRR.size
            for k in keys:
                assert k in attrs, f"{name}: {k} missing"
            # Random rows satisfy no star relation, so a zero here would
            # mean the statistic ran AFTER the drop, where each star has
            # one member left and every arm is zero by construction.
            assert float(attrs[SPREAD_ATTR_PREFIX + "diag_ev"]) > 1e-6


def test_the_ordering_is_the_one_the_creating_writer_uses(tmp_path):
    """Both writers go through ``extract_and_stamp_k_irr``.

    Parses the module rather than trusting the comment: the ordering
    (measure on the complete arrays, then drop) is the owner's ruling, and
    a second copy of it is a second place for the two steps to swap.  They
    agree today because there is one implementation.
    """
    body = (_SRC / "file_io" / "sigma_output.py").read_text()
    tree = ast.parse(body)
    callers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "extract_and_stamp_k_irr"):
                callers.add(fn.name)
    assert callers == {"write_sigma_omega_h5", "append_qsgw_datasets_h5"}, (
        f"expected exactly the two writers to share the ordering; got "
        f"{sorted(callers)}")
    # ...and neither may re-spell the measure/drop pair itself.
    for fn_name in ("append_qsgw_datasets_h5",):
        assert f"sigma_star_spread_stats(" not in body.split(
            f"def {fn_name}")[1], (
            f"{fn_name} measures the spread itself instead of through "
            f"extract_and_stamp_k_irr")


def test_the_helper_refuses_an_already_extracted_array():
    """RED TWIN of the ordering: a wedge-shaped array cannot be handed in.

    Dropping rows from an array that already has one member per star
    would produce a correctly-shaped file whose stamped spread is
    identically zero — a run that looks orbit-closed because the
    measurement had nothing left to measure.
    """
    pytest.importorskip("jax")
    rows, _ = compact_star_tables(_IRR)
    wedge = np.zeros((len(rows), _NB, _NB), dtype=np.complex128)
    with pytest.raises(ValueError, match="COMPLETE full-BZ"):
        extract_and_stamp_k_irr(
            {"sigma_xc_qsgw_kij_ev": wedge}, (_IRR, _SIDX, _NSS))


def test_the_helper_refuses_a_dataset_with_no_declared_k_axis():
    """A name outside ``SIGMA_K_AXIS`` is refused, not defaulted to axis 0.

    An array dropped along the wrong axis is still complex128 and still
    plausibly shaped; there is no downstream check that would catch it.
    """
    with pytest.raises(ValueError, match="SIGMA_K_AXIS"):
        extract_and_stamp_k_irr(
            {"not_a_sigma_dataset": np.zeros((9, 3))}, (_IRR, _SIDX, _NSS))


# ---------------------------------------------------------------------------
# 2.  The file decides the k set
# ---------------------------------------------------------------------------

def test_appending_to_a_file_that_is_not_there_refuses_by_name(tmp_path):
    """No file, no appendix.  This function never CREATES sigma_mnk.h5.

    A static compute mode writes no Σ_c(ω) cube at all.  Creating a file
    holding only the appendix would turn today's honest
    ``FileNotFoundError`` in ``eqp_bgw`` into a ``KeyError`` several
    frames deeper.
    """
    with pytest.raises(FileNotFoundError, match="APPENDIX"):
        append_qsgw_datasets_h5(
            str(tmp_path / "absent.h5"), _payload())


def test_a_file_that_is_not_a_sigma_mnk_refuses(tmp_path):
    """No cube to read the k storage off means no guess at the k axis."""
    path = tmp_path / "something_else.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("unrelated", data=np.zeros(3))
    with pytest.raises(ValueError, match="it is not a sigma_mnk"):
        append_qsgw_datasets_h5(str(path), _payload())


def test_a_k_set_that_disagrees_with_the_file_refuses(tmp_path):
    """A full-BZ file will not take rows from a different mesh.

    One file must mean one thing.  The IBZ arm of this refusal is
    ``extract_and_stamp_k_irr``'s (above); this is the arm that fires when
    there are no tables to check against.
    """
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    wrong = {"qp_omega0_ev": np.zeros((7, _NB))}
    with pytest.raises(ValueError, match="stores 9 on the full BZ"):
        append_qsgw_datasets_h5(path, wrong)


def test_a_mislabelled_file_refuses_through_the_existing_machinery(tmp_path):
    """RED TWIN.  The refusal is ``read_star_map``'s, not a copy of it.

    A slab whose row count and whose tables describe different
    calculations is what a truncated or mislabelled file looks like from
    the outside, and ``kin_ion.read_star_map`` already refuses it by name.
    The appender must inherit that rather than re-deriving a weaker
    version — so the file here is broken in the way that reader owns, and
    the message asserted is the one it writes.
    """
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    with h5py.File(path, "a") as f:
        # The tables now describe four stars for a three-row slab.
        del f["irr_idx_k"]
        f.create_dataset("irr_idx_k",
                         data=np.array([0, 1, 1, 1, 2, 1, 1, 1, 3],
                                       dtype=np.int32))
    with pytest.raises(ValueError, match="do not describe the same"):
        append_qsgw_datasets_h5(path, _payload())


def test_a_rerun_replaces_its_own_datasets(tmp_path):
    """Values and stamp describe THIS run, not a merge of two.

    Reruns in one directory are the normal way these files are made; a
    second append that silently kept the first run's rows would be a file
    whose Σ cubes and whose appendix came from different calculations.
    """
    pytest.importorskip("jax")
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    rows, _ = compact_star_tables(_IRR)
    append_qsgw_datasets_h5(path, _payload(seed=3))
    second = _payload(seed=4)
    append_qsgw_datasets_h5(path, dict(second))
    got = _read(path)
    for name in QSGW_PLOT_DATASETS:
        assert np.array_equal(
            got[name], np.take(second[name], rows, axis=SIGMA_K_AXIS[name]))


def test_a_mode_that_built_nothing_writes_nothing(tmp_path):
    """``None`` in the payload is how a mode says it has no such quantity.

    ``qp_static_cohsex_ev`` is the live case: Σ_SX and Σ_COH are built
    only by ``compute_mode = cohsex``, and a PPM run omits the dataset
    rather than putting Σ_x + Σ_c(ω=0) — which already has the name
    ``qp_omega0_ev`` — under it.
    """
    pytest.importorskip("jax")
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    src = _payload(seed=5)
    src["qp_static_cohsex_ev"] = None
    written = append_qsgw_datasets_h5(path, src)
    assert "qp_static_cohsex_ev" not in written
    with h5py.File(path, "r") as f:
        assert "qp_static_cohsex_ev" not in f
        assert "qp_omega0_ev" in f


# ---------------------------------------------------------------------------
# 3.  The committed fixture, which predates all of this
# ---------------------------------------------------------------------------

def _fixture():
    p = _REPO / "tests" / "regression" / "cohsex_debug" / "sigma_mnk.h5"
    if not p.is_file():
        pytest.skip("cohsex_debug fixture blob absent from this checkout")
    return str(p)


def test_the_committed_fixture_already_holds_all_four(tmp_path):
    """The format this writer restores is the format that file has.

    Names, ranks and dtypes, checked against the artifact rather than
    against this branch's own idea of them — that fixture is the only
    surviving output of the producer this commit replaces, and a plotting
    script written for it must keep working.
    """
    with h5py.File(_fixture(), "r") as f:
        for name in QSGW_PLOT_DATASETS:
            assert name in f, f"{name} missing from the fixture"
        assert f["sigma_xc_qsgw_kij_ev"].ndim == 3
        assert np.iscomplexobj(f["sigma_xc_qsgw_kij_ev"][()])
        for name in ("qp_static_cohsex_ev", "qp_omega0_ev",
                     "qp_diag_self_consistent_ev"):
            assert f[name].ndim == 2
            assert f[name].dtype == np.float64


def test_the_committed_fixture_still_reads_as_full_bz():
    """No-attr-means-full, on the appendix as well as on the cubes.

    The fixture predates the k_storage stamp entirely, so every one of
    its nine datasets must read as a nine-row full-BZ array.  A writer
    that had started stamping unconditionally would reinterpret it.
    """
    p = _fixture()
    for name in QSGW_PLOT_DATASETS:
        assert read_star_map(p, name, k_axis=SIGMA_K_AXIS[name]) is None
    with h5py.File(p, "r") as f:
        assert f["sigma_xc_qsgw_kij_ev"].shape[0] == 9
        assert f["qp_omega0_ev"].shape[0] == 9


def test_the_fixtures_qsgw_cube_is_hermitian():
    """The property the QSGW ansatz closes on, read back off the disk.

    ``build_qsgw_sigma_xc`` ends in ``½(M + M†)``, so a file whose cube
    is not Hermitian did not come from that kernel — which is the one
    thing about this dataset a consumer may safely assume.
    """
    with h5py.File(_fixture(), "r") as f:
        q = np.asarray(f["sigma_xc_qsgw_kij_ev"][()])
    resid = float(np.abs(q - np.conj(np.swapaxes(q, -1, -2))).max())
    assert resid == 0.0, f"non-Hermitian by {resid:.3e} eV"


def test_the_three_ladders_are_three_approximations_not_one():
    """The fixture's own evidence that the trio is worth writing.

    They were three separate quantities in the run that produced this
    file, and the plot exists because they disagree: static COHSEX sits
    ~3.6 eV off the ω=0 ladder here.  A branch that quietly made two of
    them the same expression would still write three datasets.
    """
    with h5py.File(_fixture(), "r") as f:
        static = np.asarray(f["qp_static_cohsex_ev"][()])
        omega0 = np.asarray(f["qp_omega0_ev"][()])
        diag = np.asarray(f["qp_diag_self_consistent_ev"][()])
    assert np.abs(static - omega0).max() == pytest.approx(3.63, abs=0.05)
    assert not np.array_equal(omega0, diag)


# ---------------------------------------------------------------------------
# 4.  The plotting contract
# ---------------------------------------------------------------------------

def test_a_plotting_consumer_can_reconstruct_the_full_bz_ladder(tmp_path):
    """THE "people will plot this" cell, end to end.

    A plotter opens the file, learns its k-set through the landed reader
    (``kin_ion.read_star_map``), and puts a ladder back on the full BZ
    with the landed broadcast.  Shape, dtype and finiteness are what a
    plotting call needs and all it needs.

    THE BROADCAST IS LEGITIMATE HERE AND IS NOT ON THE Σ CUBES, and the
    difference is worth stating because ``file_io.sigma_output`` refuses
    the second in as many words.  A spectrum is star-invariant when the
    quadrature is orbit-closed, so spreading a ladder over the star is
    the plotter's own choice about its own axis; spreading a Σ MATRIX
    replaces nk−nrk independent evaluations with reconstructions.  The
    stamped spread numbers are how a plotter sees which of the two
    situations its deck is in.
    """
    pytest.importorskip("jax")
    from file_io.kin_ion import broadcast_ibz_to_full_bz

    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    append_qsgw_datasets_h5(path, _payload(seed=6))

    star = read_star_map(path, "qp_omega0_ev", k_axis=0)
    assert star is not None, "the appended ladder must carry the stamp"
    with h5py.File(path, "r") as f:
        stored = np.asarray(f["qp_omega0_ev"][()])
        spread = float(f["qp_omega0_ev"].attrs[SPREAD_ATTR_PREFIX + "diag_ev"])
    full = np.asarray(broadcast_ibz_to_full_bz(stored, *star))

    assert full.shape == (_IRR.size, _NB)
    assert full.dtype == np.float64
    assert np.all(np.isfinite(full))
    # The rows that were stored come back verbatim; the rest are the
    # star members they stand for.
    rows, _ = compact_star_tables(_IRR)
    assert np.array_equal(full[rows], stored)
    assert spread > 0.0, "the plotter is told how far the star relation is"


def test_a_k_irr_side_consumer_reaches_the_cube_and_is_refused_elsewhere(
        tmp_path):
    """The Σ cube's consumer path: an index remap that REFUSES.

    ``k_irr_rows_for`` is what ``gw.eqp_bgw`` takes to this file, and the
    appended cube must be reachable the same way — with the same refusal
    on a k whose row was dropped, because handing back another member of
    its star is the substitution this format exists to prevent.
    """
    pytest.importorskip("jax")
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    src = _payload(seed=7)
    append_qsgw_datasets_h5(path, dict(src))

    compact = read_star_map(path, "sigma_xc_qsgw_kij_ev", k_axis=0)[0]
    with h5py.File(path, "r") as f:
        cube = np.asarray(f["sigma_xc_qsgw_kij_ev"][()])
    # 0, 1 and 4 are the stored full-BZ rows.
    got = k_irr_rows_for([0, 1, 4], compact, what="plotting smoke")
    assert np.array_equal(cube[got], src["sigma_xc_qsgw_kij_ev"][[0, 1, 4]])
    with pytest.raises(ValueError, match="not stored rows"):
        k_irr_rows_for([2], compact, what="plotting smoke")


# ---------------------------------------------------------------------------
# The deck key
# ---------------------------------------------------------------------------

def test_the_key_defaults_off_so_todays_file_is_unchanged():
    """DEFAULT PRESERVES TODAY'S BEHAVIOUR — the standing rule.

    These datasets have had no producer since April; ``false`` is
    byte-for-byte the file every current run writes, and the owner's
    ruling was "gated by input file", not "on".
    """
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["write_qsgw_datasets"] is False


def test_a_deck_that_never_heard_of_the_key_writes_nothing_new(tmp_path):
    """Every archived deck keeps working, unchanged."""
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n")
    assert read_lorrax_input(str(deck))["write_qsgw_datasets"] is False


@pytest.mark.parametrize("spelling,want",
                         [("true", True), ("false", False),
                          ("yes", True), ("no", False),
                          ("1", True), ("0", False)])
def test_the_key_parses_the_decks_boolean_grammar(tmp_path, spelling, want):
    """The grammar every other boolean deck key uses, not a private one."""
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(
        f"[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
        f"write_qsgw_datasets = {spelling}\n")
    assert read_lorrax_input(str(deck))["write_qsgw_datasets"] is want


def test_the_key_is_not_an_unknown_deck_key_and_reaches_the_config(
        tmp_path, monkeypatch):
    """Always-strict parsing accepts it, and the dataclass carries it.

    The parse has three layers and a key wired into two of them reads as
    its default forever with nothing to show for it.  The writers consult
    the DATACLASS, so that is what this asserts.
    """
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
                    "write_qsgw_datasets = true\n")
    cfg = LorraxConfig.from_input_file(str(deck),
                                       print_fn=lambda *a, **k: None)
    assert cfg.write_qsgw_datasets is True
    assert cfg.write_restart_tensors is True, "independent axes"


def test_the_key_has_a_row_in_the_input_reference():
    """A new deck key does not land without its reference row.

    ``docs/input_reference.md`` is the only place the deck's surface is
    written down for an operator, and the row has to name the datasets —
    "write the QSGW datasets" tells nobody what appears in their file.
    """
    doc = (_REPO / "docs" / "input_reference.md").read_text()
    assert "`write_qsgw_datasets`" in doc
    row = [ln for ln in doc.splitlines()
           if ln.startswith("| `write_qsgw_datasets`")]
    assert len(row) == 1, row
    for name in QSGW_PLOT_DATASETS:
        assert name in row[0], f"the reference row does not name {name}"


# ---------------------------------------------------------------------------
# The seams: key off writes nothing, key on writes at the right moment
# ---------------------------------------------------------------------------

def _cube_seam_cfg(key):
    return types.SimpleNamespace(write_qsgw_datasets=key)


def test_the_cube_seam_is_a_no_op_when_the_key_is_off(tmp_path):
    """RED TWIN.  Key off, file untouched — byte for byte."""
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    before = pathlib.Path(path).read_bytes()
    from gw.qsgw_utils import write_qsgw_sigma_cube

    assert write_qsgw_sigma_cube(
        path, np.zeros((9, _NB, _NB), dtype=np.complex128),
        config=_cube_seam_cfg(False), print_fn=lambda *a, **k: None) is False
    assert pathlib.Path(path).read_bytes() == before
    with h5py.File(path, "r") as f:
        assert "sigma_xc_qsgw_kij_ev" not in f


def test_the_cube_seam_writes_and_converts_to_ev_when_the_key_is_on(tmp_path):
    """The control arm, and the unit seam.

    The in-memory cube is in Ry (everything in the Σ path is); the file is
    in eV, like every other dataset in it.  A missing conversion here
    would be a factor of 13.6 in a plot with no other symptom.
    """
    pytest.importorskip("jax")
    from common.units import RYD_TO_EV
    from gw.qsgw_utils import write_qsgw_sigma_cube

    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=True)
    rows, _ = compact_star_tables(_IRR)
    cube_ry = _payload(seed=8)["sigma_xc_qsgw_kij_ev"]
    assert write_qsgw_sigma_cube(
        path, cube_ry, config=_cube_seam_cfg(True),
        print_fn=lambda *a, **k: None) is True
    with h5py.File(path, "r") as f:
        got = np.asarray(f["sigma_xc_qsgw_kij_ev"][()])
    assert np.array_equal(got, (RYD_TO_EV * cube_ry)[rows])


def test_the_ladder_seam_says_so_when_the_run_wrote_no_file(capsys):
    """A static mode has nowhere to put the ladders, and says which key.

    Silence here would look exactly like the key not being read at all,
    which is the failure mode an operator cannot debug from a log.
    """
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    zeros = np.zeros((2, 3, 3), dtype=np.complex128)
    results = GWResults(
        sig_sx=zeros, sig_coh=zeros, sig_h=zeros, sig_x=zeros,
        E_qp_ry=np.zeros((2, 3)), U_qp=zeros, E_dft_ry=np.zeros((2, 3)),
        kin_ion_ry=zeros, band_start=0, band_stop=3,
        use_ppm=False, sigma_omega_h5_path=None)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=True,
        compute_mode=types.SimpleNamespace(value="cohsex"))
    out = write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=None, omega_grid_ev=None, print_fn=print)
    assert out == []
    printed = capsys.readouterr().out
    assert "write_qsgw_datasets" in printed
    assert "sigma_mnk.h5" in printed


def test_the_ladder_seam_is_silent_and_writes_nothing_when_off(capsys,
                                                               tmp_path):
    """RED TWIN of the cell above: the default path says nothing at all."""
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    zeros = np.zeros((9, _NB, _NB), dtype=np.complex128)
    results = GWResults(
        sig_sx=zeros, sig_coh=zeros, sig_h=zeros, sig_x=zeros,
        E_qp_ry=np.zeros((9, _NB)), U_qp=zeros,
        E_dft_ry=np.zeros((9, _NB)), kin_ion_ry=zeros,
        band_start=0, band_stop=_NB, use_ppm=True,
        sigma_omega_h5_path=path)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=False,
        compute_mode=types.SimpleNamespace(value="gn_ppm"))
    assert write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=np.zeros((5, 9, _NB)),
        omega_grid_ev=np.linspace(-2, 2, 5), print_fn=print) == []
    assert capsys.readouterr().out.strip() == ""
    with h5py.File(path, "r") as f:
        assert not any(n in f for n in QSGW_PLOT_DATASETS)


def test_the_ladders_are_three_ladders_of_one_h0(tmp_path, capsys):
    """The seam, run: H₀ shared, Σ_xc differing, three ladders out.

    Built on a static (COHSEX-shaped) result so all three arms fire at
    once — that is the only configuration in which the trio is complete,
    and it is the configuration the plot was invented for.
    """
    pytest.importorskip("jax")
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    nk, nb, n_omega = 9, _NB, 5
    rng = np.random.default_rng(13)

    def _herm(scale=1.0):
        a = (rng.standard_normal((nk, nb, nb))
             + 1j * rng.standard_normal((nk, nb, nb))) * scale
        return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))

    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    omega_ev = np.linspace(-2.0, 2.0, n_omega)
    results = GWResults(
        sig_sx=_herm(), sig_coh=_herm(), sig_h=_herm(), sig_x=_herm(),
        E_qp_ry=rng.standard_normal((nk, nb)), U_qp=_herm(),
        E_dft_ry=rng.standard_normal((nk, nb)) * 0.01,
        kin_ion_ry=_herm(2.0), band_start=0, band_stop=nb,
        use_ppm=False, efermi_ev=0.0, sigma_omega_h5_path=path)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=True,
        compute_mode=types.SimpleNamespace(value="cohsex"))
    written = write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=rng.standard_normal((n_omega, nk, nb)),
        omega_grid_ev=omega_ev,
        sigma_c_omega=(rng.standard_normal((n_omega, nk, nb, nb))
                       + 0j),
        print_fn=print)
    assert set(written) == {"qp_static_cohsex_ev", "qp_omega0_ev",
                            "qp_diag_self_consistent_ev"}
    got = _read(path)
    for name in written:
        assert got[name].shape == (nk, nb)
        assert got[name].dtype == np.float64
        assert np.all(np.isfinite(got[name]))
    # Three approximations, not one expression written three times.
    assert not np.array_equal(got["qp_static_cohsex_ev"],
                              got["qp_omega0_ev"])
    assert not np.array_equal(got["qp_omega0_ev"],
                              got["qp_diag_self_consistent_ev"])
    # Ladders are eigenvalues: sorted ascending per k, by construction.
    for name in written:
        if name != "qp_diag_self_consistent_ev":
            assert np.all(np.diff(got[name], axis=1) >= 0.0)


def test_a_ppm_run_omits_the_static_cohsex_ladder_and_names_it(tmp_path,
                                                               capsys):
    """The omission, and its one line.

    Σ_SX and Σ_COH are not built by a dynamic mode, and the alternative —
    putting Σ_x + Σ_c(ω=0) under that name — would write one quantity
    twice under two labels.  The plot is better off short a curve than
    wrong about one.
    """
    pytest.importorskip("jax")
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    nk, nb, n_omega = 9, _NB, 5
    rng = np.random.default_rng(17)
    zeros = np.zeros((nk, nb, nb), dtype=np.complex128)
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    results = GWResults(
        sig_sx=zeros, sig_coh=zeros, sig_h=zeros,
        sig_x=np.zeros((nk, nb, nb), dtype=np.complex128),
        E_qp_ry=np.zeros((nk, nb)), U_qp=zeros,
        E_dft_ry=np.zeros((nk, nb)),
        kin_ion_ry=np.zeros((nk, nb, nb), dtype=np.complex128),
        band_start=0, band_stop=nb, use_ppm=True, efermi_ev=0.0,
        sigma_omega_h5_path=path)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=True,
        compute_mode=types.SimpleNamespace(value="gn_ppm"))
    written = write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=rng.standard_normal((n_omega, nk, nb)),
        omega_grid_ev=np.linspace(-2.0, 2.0, n_omega),
        sigma_c_omega=(rng.standard_normal((n_omega, nk, nb, nb)) + 0j),
        print_fn=print)
    assert "qp_static_cohsex_ev" not in written
    assert set(written) == {"qp_omega0_ev", "qp_diag_self_consistent_ev"}
    printed = capsys.readouterr().out
    assert "qp_static_cohsex_ev" in printed and "omitting" in printed


def test_a_band_sharded_cube_is_skipped_rather_than_gathered(tmp_path,
                                                             capsys):
    """RED TWIN of the ω₀ arm: no host transfer of a tiled cube.

    This seam is inside ``gw_jax``'s rank-0 output block, so pulling one
    ω slice of a ``sigma_omega_layout = sharded`` cube back would be the
    "spans non-addressable devices" error at P>1 — and gathering it would
    put a collective inside a single-rank block, which deadlocks.  The
    layout is read off the array, so the skip cannot disagree with the
    producer about which path the tensor took.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    nk, nb, n_omega = 9, _NB, 5
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    cube = jax.device_put(
        jnp.zeros((n_omega, nk, nb, nb), dtype=jnp.complex128),
        NamedSharding(mesh, P(None, None, "x", "y")))

    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    zeros = np.zeros((nk, nb, nb), dtype=np.complex128)
    results = GWResults(
        sig_sx=zeros, sig_coh=zeros, sig_h=zeros, sig_x=zeros,
        E_qp_ry=np.zeros((nk, nb)), U_qp=zeros, E_dft_ry=np.zeros((nk, nb)),
        kin_ion_ry=zeros, band_start=0, band_stop=nb, use_ppm=True,
        efermi_ev=0.0, sigma_omega_h5_path=path)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=True,
        compute_mode=types.SimpleNamespace(value="gn_ppm"))
    written = write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=np.zeros((n_omega, nk, nb)),
        omega_grid_ev=np.linspace(-2.0, 2.0, n_omega),
        sigma_c_omega=cube, print_fn=print)
    assert "qp_omega0_ev" not in written
    printed = capsys.readouterr().out
    assert "sigma_omega_layout" in printed, printed


def test_the_log_reports_each_ladder_against_the_runs_own_e_qp(tmp_path,
                                                               capsys):
    """The headline number, so the log is readable without the file.

    ``e_qp_ry`` is the run's own ladder — the eigh ``eqp0.dat`` is built
    from — and the point of the appendix is the distance between it and
    each approximation.  Reported before the append, on the full-BZ
    arrays, because afterwards the ladders are one row per star and the
    difference would be against the wrong rows.
    """
    pytest.importorskip("jax")
    from gw.gw_output import GWResults, write_qsgw_qp_ladders

    nk, nb, n_omega = 9, _NB, 5
    rng = np.random.default_rng(23)
    zeros = np.zeros((nk, nb, nb), dtype=np.complex128)
    path = _make_file(tmp_path / "sigma_mnk.h5", on_ibz=False)
    results = GWResults(
        sig_sx=zeros, sig_coh=zeros, sig_h=zeros, sig_x=zeros,
        E_qp_ry=np.zeros((nk, nb)), U_qp=zeros, E_dft_ry=np.zeros((nk, nb)),
        kin_ion_ry=zeros, band_start=0, band_stop=nb, use_ppm=True,
        efermi_ev=0.0, sigma_omega_h5_path=path)
    cfg = types.SimpleNamespace(
        write_qsgw_datasets=True,
        compute_mode=types.SimpleNamespace(value="gn_ppm"))
    write_qsgw_qp_ladders(
        results, config=cfg, e_qp_ry=results.E_qp_ry,
        sigma_c_omega_diag_ev=rng.standard_normal((n_omega, nk, nb)),
        omega_grid_ev=np.linspace(-2.0, 2.0, n_omega),
        sigma_c_omega=(rng.standard_normal((n_omega, nk, nb, nb)) + 0j),
        print_fn=print)
    printed = capsys.readouterr().out
    assert "max |Δ|" in printed, printed
    assert "omega0" in printed and "diag_self_consistent" in printed


# ---------------------------------------------------------------------------
# Where the cube is written, and why it cannot move
# ---------------------------------------------------------------------------

def _calls_of(path, name):
    tree = ast.parse(pathlib.Path(path).read_text(), filename=str(path))
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id",
                                getattr(node.func, "attr", "")) == name):
                out.append(fn.name)
    return out


def test_the_cube_is_written_at_both_seams_and_only_there():
    """One writer, two call sites, each in the basis its path is in.

    Under self-consistency ``sigma_xc_kij_ry`` is in the QP basis — the
    basis the file's Σ_c cube is in — and ``run_sc_driver`` rotates it to
    the DFT basis a few frames after ``dump_sigma_omega_h5_final``
    returns.  Appending after that rotation would put one DFT-basis
    matrix into a file of QP-basis ones with matching shape, dtype and
    stamp, which nothing downstream checks.  The one-shot seam is in the
    Σ dispatch for the mirror reason: that is where the file it appends
    to was just written.
    """
    sc = _calls_of(_SRC / "gw" / "sc_iteration.py", "write_qsgw_sigma_cube")
    disp = _calls_of(_SRC / "gw" / "sigma_dispatch.py",
                     "write_qsgw_sigma_cube")
    util = _calls_of(_SRC / "gw" / "qsgw_utils.py", "write_qsgw_sigma_cube")
    assert sc == ["dump_sigma_omega_h5_final"], sc
    assert disp == ["finalize_dynamic_sigma"], disp
    # ``solve_qp``'s fixed_point branch rebuilds at the solved energies
    # and must overwrite the at-DFT cube the dispatch wrote, or the file
    # disagrees with eqp0.dat by the whole on-shell correction.
    assert util == ["solve_qp"], util


def test_the_one_shot_cube_write_is_behind_the_file_write_flag():
    """It must not fire on an SC iteration, which writes no file.

    ``write_sigma_omega_h5`` is exactly "did THIS call create
    sigma_mnk.h5".  Without the guard, every intermediate SC iteration
    would append its own cube to whatever file the previous run left in
    the directory.
    """
    src = _SRC / "gw" / "sigma_dispatch.py"
    tree = ast.parse(src.read_text(), filename=str(src))
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if not (isinstance(t, ast.Name) and t.id == "write_sigma_omega_h5"):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and getattr(sub.func, "id",
                                getattr(sub.func, "attr", ""))
                    == "write_qsgw_sigma_cube"):
                guarded = True
    assert guarded, (
        "the one-shot QSGW cube append is not inside "
        "``if write_sigma_omega_h5:``; on the SC path it would append an "
        "intermediate iteration's cube to a stale file")
