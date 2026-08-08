"""The k_irr extraction of ``sigma_mnk.h5``: it is a SELECTION, and exactly.

The claim this file pins is narrow and therefore checkable: the rows the
extracted file holds are the rows the full-BZ file held, bit for bit.  That
is not a tolerance question — a selection either takes the row or it does
not — so every equality here is ``array_equal``, never ``allclose``.  A
gate written with a tolerance would pass a writer that had started doing
arithmetic on the way out, which is the one thing this format must never
do (``file_io.sigma_output``'s block: reconstruction is refused, selection
is not).

Four things are gated:

1.  **Bit-identity of the selection**, on both the 3-D and the 4-D
    datasets, against the same writer run without ``star``.
2.  **Back-compat by absence.**  A file written with no ``star`` carries no
    ``k_storage`` attr and reads as full-BZ, which is what every committed
    fixture and every older run is.  The discriminator has to be an
    attribute the old writer never wrote, or old files get reinterpreted.
3.  **The refusal twin.**  ``k_irr_rows_for`` must REFUSE a full-BZ index
    whose row was dropped, rather than silently returning another member
    of the same star — on ``cohsex_debug`` those members differ by 113 eV
    (see ``tools/sigma_star_spread_decompose.py``), so a silent
    substitution is a physics error with no symptom.
4.  **The association order of the derived total**, ``(c + sx) + h``,
    bit-identically.  This was previously filed as "unpinned association
    costs 4.4e-05 eV"; it was measured on 2026-08-08 and that diagnosis
    was wrong (all three orders give bit-identically the same residual on
    that fixture, which is a widened float32 artifact).  The association
    was in fact already pinned by construction in both write paths.  What
    was missing is this gate, so the property stops depending on nobody
    editing either path.

The spread statistic gets its own cell: it must be computed on the
COMPLETE cube, which is checkable by construction — a stat computed after
the drop would have one member per star to compare and would read exactly
zero.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from file_io.sigma_output import (                              # noqa: E402
    K_STORAGE_ATTR,
    K_STORAGE_IBZ,
    K_STORAGE_VERSION,
    K_STORAGE_VERSION_ATTR,
    SPREAD_ATTR_PREFIX,
    compact_star_tables,
    k_irr_rows_for,
    derive_sigma_total,
    sigma_star_spread_stats,
    star_select_k_irr,
    write_sigma_omega_h5,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# A synthetic deck, so the gate does not depend on a fixture blob
# ---------------------------------------------------------------------------
#: ``cohsex_debug``'s own topology, typed once: 9 full-BZ k over 3 stars,
#: ntran 12, and — the detail that makes this deck worth copying — an
#: ``irr_idx_k`` whose labels are NOT contiguous (``{0, 2, 3}`` out of a
#: 4-row wedge).  A writer that filed that table verbatim would claim 4
#: stored rows for a 3-row slab, which is exactly what
#: ``kin_ion.read_star_map`` refuses on.
_IRR = np.array([0, 2, 2, 2, 3, 2, 2, 2, 3])
_SIDX = np.array([0, 12, 0, 3, 0, 14, 15, 2, 1])
_NSS = 12


def _mesh():
    jax = pytest.importorskip("jax")
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _need_slab_io():
    """Skip the FILE-FORMAT cells where SlabIO has no transport.

    ``file_io.slab_io`` has exactly one backend (collective MPI-IO through
    ``liblorrax_ffi*.so``) and REFUSES rather than falling back, so on a
    box without that library nothing can write this file at all.  The
    claims that do not need a file — the selection, the association, the
    refusal, the statistic — are gated above WITHOUT this skip, on purpose:
    they are the load-bearing ones and they must run everywhere.
    """
    pytest.importorskip("jax")
    from ffi.common.ffi_loader import has_phdf5_write
    if not has_phdf5_write("cpu"):
        pytest.skip("no phdf5 write symbol on this platform; SlabIO has one "
                    "transport and does not fall back")


def _cube(n_omega=4, nk=9, nb=5, seed=0):
    rng = np.random.default_rng(seed)

    def _c(shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    return {
        "omega_ev": np.linspace(-2.0, 2.0, n_omega),
        "sigma_c_kij_ev": _c((n_omega, nk, nb, nb)),
        "sigma_sx_kij_ev": _c((nk, nb, nb)),
        "hartree_kij_ev": _c((nk, nb, nb)),
    }


def _write(tmp_path, data, star):
    tag = 'irr' if star else 'full'
    path = os.path.join(str(tmp_path), f"sigma_mnk_{tag}.h5")
    write_sigma_omega_h5(
        path, data["omega_ev"], None,
        sigma_c_kij_ev=data["sigma_c_kij_ev"],
        sigma_sx_kij_ev=data["sigma_sx_kij_ev"],
        hartree_kij_ev=data["hartree_kij_ev"],
        mesh=_mesh(), star=star)
    return path


def _read(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k][()]) for k in f}


# ---------------------------------------------------------------------------
# 1.  The selection is a selection
# ---------------------------------------------------------------------------

def test_compaction_renumbers_a_noncontiguous_wedge():
    """The stored table must index the STORED rows, not the upstream wedge."""
    rows, compact = compact_star_tables(_IRR)
    assert rows.tolist() == [0, 1, 4], (
        "the kept rows are the first occurrence of each star, in order")
    assert compact.tolist() == [0, 1, 1, 1, 2, 1, 1, 1, 2]
    # The property ``read_star_map`` checks: max label + 1 == stored rows.
    assert int(compact.max()) + 1 == len(rows)


def test_the_extracted_rows_are_bit_identical_to_the_full_bz_rows(tmp_path):
    """THE gate.  Not a tolerance — a row is taken or it is not."""
    _need_slab_io()
    data = _cube()
    rows, _ = compact_star_tables(_IRR)

    full = _read(_write(tmp_path, data, None))
    irr = _read(_write(tmp_path, data, (_IRR, _SIDX, _NSS)))

    for name, k_axis in (("sigma_total_kij_ev", 1), ("sigma_c_kij_ev", 1),
                         ("sigma_sx_kij_ev", 0), ("hartree_kij_ev", 0)):
        assert full[name].shape[k_axis] == len(_IRR)
        assert irr[name].shape[k_axis] == len(rows)
        expected = np.take(full[name], rows, axis=k_axis)
        assert np.array_equal(irr[name], expected), (
            f"{name}: the extracted rows are not bit-identical to the "
            f"full-BZ rows they were taken from — the writer is doing "
            f"arithmetic on the way out, which is reconstruction, not "
            f"selection")

    assert np.array_equal(full["omega_ev"], irr["omega_ev"])


def test_star_select_is_a_take_and_nothing_else():
    """``star_select_k_irr`` must not touch the values it moves."""
    a = np.arange(9 * 3 * 3, dtype=np.complex128).reshape(9, 3, 3)
    rows, _ = compact_star_tables(_IRR)
    assert np.array_equal(star_select_k_irr(a, rows), a[rows])
    b = np.arange(4 * 9 * 2 * 2, dtype=np.complex128).reshape(4, 9, 2, 2)
    assert np.array_equal(star_select_k_irr(b, rows, k_axis=1), b[:, rows])


# ---------------------------------------------------------------------------
# 2.  Back-compat is by ABSENCE of the attribute
# ---------------------------------------------------------------------------

def test_no_star_means_no_attr_means_full(tmp_path):
    """An unstamped file must stay readable as exactly what it was."""
    _need_slab_io()
    from file_io.kin_ion import read_star_map

    path = _write(tmp_path, _cube(), None)
    with h5py.File(path, "r") as f:
        for name in ("sigma_total_kij_ev", "sigma_c_kij_ev",
                     "sigma_sx_kij_ev", "hartree_kij_ev"):
            assert K_STORAGE_ATTR not in f[name].attrs, (
                f"{name} carries a k_storage attr on a full-BZ write; the "
                f"back-compat default is that OLD files have no attr, so "
                f"the writer must not add one when it has no star tables")
        assert "irr_idx_k" not in f and "sym_idx_k" not in f
    assert read_star_map(path, "sigma_c_kij_ev", k_axis=1) is None


def test_the_committed_fixture_still_reads_as_full_bz():
    """The fixture predates the stamp and must never be reinterpreted."""
    from file_io.kin_ion import read_star_map

    p = os.path.join(_REPO, "tests", "regression", "cohsex_debug",
                     "sigma_mnk.h5")
    if not os.path.isfile(p):
        pytest.skip("cohsex_debug fixture blob absent from this checkout")
    for name, k_axis in (("sigma_c_kij_ev", 1), ("hartree_kij_ev", 0)):
        assert read_star_map(p, name, k_axis=k_axis) is None
    with h5py.File(p, "r") as f:
        assert f["hartree_kij_ev"].shape[0] == 9, (
            "the committed fixture is a 9-row full-BZ file and the "
            "extraction must not have touched it")


def test_the_stamps_are_kin_ions_and_the_tables_are_filed_with_them(tmp_path):
    """One stamp vocabulary for both files, and one reader for it."""
    _need_slab_io()
    from file_io.kin_ion import read_star_map

    rows, compact = compact_star_tables(_IRR)
    path = _write(tmp_path, _cube(), (_IRR, _SIDX, _NSS))
    with h5py.File(path, "r") as f:
        ds = f["sigma_c_kij_ev"]
        assert ds.attrs[K_STORAGE_ATTR] == K_STORAGE_IBZ
        assert int(ds.attrs[K_STORAGE_VERSION_ATTR]) == K_STORAGE_VERSION
        assert int(ds.attrs["n_sym_spatial"]) == _NSS
        assert int(ds.attrs["nk_full"]) == len(_IRR)
        assert np.array_equal(np.asarray(f["irr_idx_k"]), compact)
        assert np.array_equal(np.asarray(f["sym_idx_k"]), _SIDX)
    # and the one reader implementation accepts it on both ranks of shape
    got = read_star_map(path, "sigma_c_kij_ev", k_axis=1)
    assert got is not None and np.array_equal(got[0], compact)
    assert read_star_map(path, "hartree_kij_ev", k_axis=0) is not None


# ---------------------------------------------------------------------------
# 3.  The refusal twin
# ---------------------------------------------------------------------------

def test_a_dropped_row_is_refused_not_substituted():
    """The red twin.  A dropped k must raise, never return its star-mate.

    This is the whole safety argument for the extraction.  ``compact[i]``
    is defined for every i, so the tempting implementation returns the
    star's first member for a k that was dropped — and on a deck whose
    quadrature is not orbit-closed those two matrices differ by up to
    113 eV with no symptom anywhere downstream.
    """
    _, compact = compact_star_tables(_IRR)
    # 0, 1 and 4 are stored rows; 2, 3, 5, 6, 7, 8 were dropped.
    assert k_irr_rows_for([0, 1, 1, 4], compact).tolist() == [0, 1, 1, 2]
    for dropped in (2, 3, 5, 6, 7, 8):
        with pytest.raises(ValueError, match="not stored rows"):
            k_irr_rows_for([0, dropped], compact)
    with pytest.raises(ValueError, match="out of range"):
        k_irr_rows_for([0, 99], compact)


def test_the_live_wedge_map_of_the_fixture_deck_resolves():
    """``kirr_to_kfull`` on the real deck must land on stored rows.

    The anti-tautology arm of the cell above: the refusal is only useful
    if the SUPPORTED case actually passes, and on ``cohsex_debug`` the
    wedge map is ``[0, 1, 1, 4]`` over a 4-row WFN wedge that covers 3
    stars — a redundancy that would break a naive remap.
    """
    p = os.path.join(_REPO, "tests", "regression", "cohsex_debug",
                     "qp_wfn_rotations.h5")
    if not os.path.isfile(p):
        pytest.skip("cohsex_debug fixture blob absent from this checkout")
    with h5py.File(p, "r") as f:
        kmap = np.asarray(f["kirr_to_kfull"], dtype=np.int64)
    _, compact = compact_star_tables(_IRR)
    assert k_irr_rows_for(kmap, compact).tolist() == [0, 1, 1, 2]


# ---------------------------------------------------------------------------
# 4.  The association order of the derived total
# ---------------------------------------------------------------------------

def test_the_derived_total_is_c_plus_sx_plus_h_bit_identically(tmp_path):
    """``(c + sx[None]) + h[None]``, pinned.

    Both write paths already spell this association — the replicated
    derivation in ``write_sigma_omega_h5`` and the sharded one in
    ``gw.ppm_pipeline._write_sigma_omega_h5`` — and the point of the gate
    is that they keep spelling the SAME one.  Bit-identical, in float64,
    because a reassociation of a 3-term float64 sum shows up at 1e-16
    relative and any gate loose enough to miss that is not pinning
    anything.
    """
    _need_slab_io()
    data = _cube(seed=3)
    full = _read(_write(tmp_path, data, None))
    expected = ((data["sigma_c_kij_ev"]
                 + data["sigma_sx_kij_ev"][None, ...])
                + data["hartree_kij_ev"][None, ...])
    assert np.array_equal(full["sigma_total_kij_ev"], expected)


def test_the_extraction_and_the_sum_commute(tmp_path):
    """Extracting the total equals totalling the extraction.

    Only true because the sum runs before the drop; if a future edit
    reordered them, float arithmetic would still agree to 1e-16 and this
    is the cell that would notice.
    """
    _need_slab_io()
    data = _cube(seed=4)
    rows, _ = compact_star_tables(_IRR)
    irr = _read(_write(tmp_path, data, (_IRR, _SIDX, _NSS)))
    expected = ((data["sigma_c_kij_ev"][:, rows]
                 + data["sigma_sx_kij_ev"][rows][None, ...])
                + data["hartree_kij_ev"][rows][None, ...])
    assert np.array_equal(irr["sigma_total_kij_ev"], expected)


# ---------------------------------------------------------------------------
# The spread statistic
# ---------------------------------------------------------------------------

def test_the_spread_stat_is_measured_before_the_drop(tmp_path):
    """A stat taken after the drop would read zero.  This one does not."""
    _need_slab_io()
    data = _cube(seed=7)
    path = _write(tmp_path, data, (_IRR, _SIDX, _NSS))
    with h5py.File(path, "r") as f:
        attrs = dict(f["hartree_kij_ev"].attrs)
    keys = [SPREAD_ATTR_PREFIX + k for k in
            ("raw_ev", "diag_ev", "frobenius_ev", "trace_ev", "omega_index")]
    for k in keys:
        assert k in attrs, f"{k} missing — the stat did not reach the file"
    # Random rows do not satisfy any star relation, so every arm must be
    # comfortably non-zero.  Zero here means the stat ran on the extracted
    # slab (one member per star) rather than on the complete one.
    for k in keys[:4]:
        assert float(attrs[k]) > 1e-6, (
            f"{k} is ~0 on random data; the statistic was computed AFTER "
            f"the extraction, where each star has one member left and "
            f"every spread is identically zero by construction")


def test_the_gauge_blind_arms_vanish_on_an_exact_star_relation():
    """Build a cube that DOES satisfy the star relation; the stat sees it.

    The anti-tautology twin of the cell above.  ``star_broadcast`` of a
    wedge slab is by construction star-exact, so every arm must collapse
    to round-off — which is what makes a NON-zero reading elsewhere
    meaningful rather than an artefact of the metric.
    """
    pytest.importorskip("jax")
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps

    rows, compact = compact_star_tables(_IRR)
    rng = np.random.default_rng(11)
    wedge = (rng.standard_normal((len(rows), 4, 4))
             + 1j * rng.standard_normal((len(rows), 4, 4)))
    full = np.asarray(symmetry_maps.star_broadcast(
        wedge, compact, _SIDX, _NSS,
        irr_labels=np.arange(len(rows), dtype=np.int32)))

    stats = sigma_star_spread_stats(full, rows, compact, _SIDX, _NSS)
    for arm in ("raw_ev", "frobenius_ev", "trace_ev"):
        assert stats[arm] < 1e-12, (
            f"{arm} = {stats[arm]:.3e} on a slab that satisfies the star "
            f"relation exactly; the metric is measuring something other "
            f"than the relation")


def test_the_stat_reports_a_real_break_on_the_committed_fixture():
    """The published 113.08 eV, reproduced through the shipped stat.

    Ties the in-run statistic to the number in ``sigma_output``'s block and
    in ``NOTE_sigma_star_spread.md``, so the two cannot drift.  The
    gauge-blind arms are asserted LARGE here as well, because on this deck
    the breaking is real (non-orbit-closed centroids) and not gauge — that
    is the diagnostic verdict, and this is where it is pinned.
    """
    p = os.path.join(_REPO, "tests", "regression", "cohsex_debug",
                     "sigma_mnk.h5")
    if not os.path.isfile(p):
        pytest.skip("cohsex_debug fixture blob absent from this checkout")
    pytest.importorskip("jax")
    rows, compact = compact_star_tables(_IRR)
    with h5py.File(p, "r") as f:
        hartree = np.asarray(f["hartree_kij_ev"][()])
    stats = sigma_star_spread_stats(hartree, rows, compact, _SIDX, _NSS)
    assert stats["raw_ev"] == pytest.approx(113.08, abs=0.01)
    assert stats["diag_ev"] == pytest.approx(61.51, abs=0.01)
    # Gauge-blind and still large: the break survives the basis freedom.
    assert stats["frobenius_ev"] > 20.0
    assert stats["trace_ev"] > 70.0


# ---------------------------------------------------------------------------
# The same claims, WITHOUT a file — so they run on a box with no SlabIO
# ---------------------------------------------------------------------------
# Everything above that needs ``_need_slab_io`` is gating the FILE.  These
# cells gate the same properties on the functions the writer composes, so
# a box that cannot write the format still refuses a broken selection, a
# reassociated total, or a statistic taken at the wrong moment.  They are
# not duplicates: they fail for a narrower reason, which is what makes a
# skipped file cell survivable rather than a hole.

def test_the_association_is_c_plus_sx_then_h_bit_identically():
    """``derive_sigma_total`` IS ``(c + sx[None]) + h[None]``.

    Bit-identical in float64: a reassociation moves a three-term sum at
    1e-16 relative, so ``allclose`` here would pin nothing at all.  Both
    write paths call this one function (the replicated branch in
    ``write_sigma_omega_h5``, the jitted ``_ev_tensors`` in
    ``gw.ppm_pipeline``), which is what makes one gate enough.
    """
    rng = np.random.default_rng(17)
    c = (rng.standard_normal((3, 9, 4, 4))
         + 1j * rng.standard_normal((3, 9, 4, 4)))
    sx = rng.standard_normal((9, 4, 4)) + 1j * rng.standard_normal((9, 4, 4))
    h = rng.standard_normal((9, 4, 4)) + 1j * rng.standard_normal((9, 4, 4))

    assert np.array_equal(derive_sigma_total(c, sx, h),
                          (c + sx[None, ...]) + h[None, ...])
    # The other orders are genuinely different bytes — otherwise the cell
    # above would pass for a writer that had reassociated.
    assert not np.array_equal(derive_sigma_total(c, sx, h),
                              c + (sx + h)[None, ...])
    # Optional operands still behave.
    assert np.array_equal(derive_sigma_total(c, None, h), c + h[None, ...])
    assert np.array_equal(derive_sigma_total(c, None, None), c)


def test_both_write_paths_call_the_one_association():
    """Neither path may re-spell the sum inline.

    Parses the two files rather than trusting a comment: the sharded
    branch used to carry its own copy of the expression, and a copy is
    exactly what this gate exists to stop coming back.
    """
    import re
    for rel in ("src/file_io/sigma_output.py", "src/gw/ppm_pipeline.py"):
        body = open(os.path.join(_REPO, rel)).read()
        # the derivation site must go through the named function
        assert "derive_sigma_total(" in body, (
            f"{rel} no longer calls derive_sigma_total; if the sum moved, "
            f"move the gate with it rather than deleting the pin")
        # ...and must not spell the three-term sum inline any more
        inline = re.search(r"\+\s*\(?RYD_TO_EV\s*\*\s*h_ry\)?\[None", body)
        assert inline is None, (
            f"{rel} spells the c+sx+h association inline again; the two "
            f"write paths must produce bit-identical bytes and that only "
            f"holds while they share one expression")


def test_the_extraction_commutes_with_the_sum_on_arrays():
    """Extract-then-total equals total-then-extract, bit for bit.

    True only because the writer sums BEFORE it drops rows.  If a future
    edit swapped the order the results would still agree to 1e-16 and no
    physics gate would notice — this cell would.
    """
    rng = np.random.default_rng(23)
    c = (rng.standard_normal((3, 9, 4, 4))
         + 1j * rng.standard_normal((3, 9, 4, 4)))
    sx = rng.standard_normal((9, 4, 4)) + 1j * rng.standard_normal((9, 4, 4))
    h = rng.standard_normal((9, 4, 4)) + 1j * rng.standard_normal((9, 4, 4))
    rows, _ = compact_star_tables(_IRR)

    sum_then_drop = star_select_k_irr(
        derive_sigma_total(c, sx, h), rows, k_axis=1)
    drop_then_sum = derive_sigma_total(
        star_select_k_irr(c, rows, k_axis=1),
        star_select_k_irr(sx, rows),
        star_select_k_irr(h, rows))
    assert np.array_equal(sum_then_drop, drop_then_sum)


def test_a_stat_taken_after_the_drop_reads_zero():
    """Why the ORDER in the writer is load-bearing, demonstrated.

    The extracted slab has one member per star, so every spread arm over
    it is identically zero. That is the failure mode the writer's ordering
    avoids, and this cell is the proof that the ordering is not cosmetic.
    """
    pytest.importorskip("jax")
    rng = np.random.default_rng(29)
    full = (rng.standard_normal((9, 4, 4))
            + 1j * rng.standard_normal((9, 4, 4)))
    rows, compact = compact_star_tables(_IRR)

    before = sigma_star_spread_stats(full, rows, compact, _SIDX, _NSS)
    assert before["diag_ev"] > 1e-6 and before["frobenius_ev"] > 1e-6

    # The same statistic, computed on the extracted slab: each star now
    # has exactly one member, so there is nothing left to disagree.
    dropped = star_select_k_irr(full, rows)
    trivial_rows, trivial_compact = compact_star_tables(np.arange(len(rows)))
    after = sigma_star_spread_stats(
        dropped, trivial_rows, trivial_compact,
        np.zeros(len(rows), dtype=np.int32), _NSS)
    assert after["diag_ev"] == 0.0 and after["frobenius_ev"] == 0.0, (
        "a post-extraction statistic must read exactly zero; if it does "
        "not, this cell is measuring something other than the star spread")
