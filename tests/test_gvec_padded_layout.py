"""The ONE padded G representation shared by ψ (WFN.h5) and ζ (zeta_q.h5).

``common.gvec_fft_box`` owns the pad sentinel, the routine that builds
the padded table, the detector that spots a padded table which lost its
mask, and the FFT-box gather index.  Both producers go through it:

* ζ — ``common.coulomb_sphere.compute_per_q_bare_coulomb_components``
  turns a per-q ``|q+G|² ≤ cutoff`` sphere into ``(n_q, 3, ngkmax)``
  components + ``(n_q, ngkmax)`` flat indices for ``isdf_header``.
* ψ — ``file_io.wfn_loader.WfnLoader.gvecs`` turns the ragged on-disk
  ``wfns/gvecs`` into ``(n_k, ngkmax, 3)``.

What each test here rules out is stated on the test.  The theme is that
the pad marker is only useful if "sentinel row ⇔ pad row" is TRUE, so
every check below is paired with the construction that makes it false.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from common.gvec_fft_box import (
    build_g_index_for_fft_box,
    fft_box_pad_sentinel,
    pad_gvecs_to_sentinel,
    pad_mask,
    refuse_padded_gvecs_without_mask,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WFN_FIXTURES = sorted(glob.glob(os.path.join(_HERE, "regression", "*",
                                              "WFN*.h5")))


# ---------------------------------------------------------------------------
# 1. The sentinel itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [(6, 6, 8), (15, 15, 60), (5, 5, 5),
                                  (24, 24, 80), (1, 2, 3)])
def test_sentinel_components_and_flat_index_are_the_same_cell(grid):
    """The two halves of the sentinel must not drift apart.

    ``components`` is read off ``fftfreq``; ``flat_index`` is computed
    arithmetically.  A consumer holding either one must land on the same
    FFT-box cell, or the ζ writer's ``sphere_idx_padded`` gather and the
    reader's ``gvec_components`` pad marker describe different slots.
    Falsified by e.g. writing ``-nx//2`` for the components: correct for
    even extents, off by one for odd ones (grid 5 → fftfreq gives +2,
    ``-5//2`` gives −3, which wraps to cell 2 as well but is a Miller
    index the box never produces).
    """
    comps, flat = fft_box_pad_sentinel(grid)
    nx, ny, nz = grid
    cell = tuple(int(c) % int(n) for c, n in zip(comps, grid))
    assert cell == (nx // 2, ny // 2, nz // 2)
    assert flat == cell[0] * ny * nz + cell[1] * nz + cell[2]
    assert 0 <= flat < nx * ny * nz

    # The Miller triple is the one the fftfreq table actually holds, so a
    # reader that rebuilds G from the flat index agrees.
    tables = [np.rint(np.fft.fftfreq(n) * n).astype(np.int32) for n in grid]
    assert [int(t[c]) for t, c in zip(tables, cell)] == \
        [int(v) for v in comps]


def test_sentinel_matches_the_documented_minus_half_grid_for_even_extents():
    """``isdf_header`` documents the pad as ``(-nx/2, -ny/2, -nz/2)``.

    That statement is only true for EVEN extents, which is what BGW FFT
    grids are; this pins the agreement where it is claimed and shows the
    odd case differs, so the docstring is not silently over-general.
    """
    comps_even, _ = fft_box_pad_sentinel((24, 24, 80))
    assert [int(v) for v in comps_even] == [-12, -12, -40]
    comps_odd, _ = fft_box_pad_sentinel((15, 15, 61))
    assert [int(v) for v in comps_odd] == [7, 7, 30]      # NOT (-7, -7, -30)


# ---------------------------------------------------------------------------
# 2. The shared build: one routine, both producers
# ---------------------------------------------------------------------------

def test_pad_table_fills_only_the_pad_and_reports_the_extents():
    grid = (6, 6, 8)
    sent, _ = fft_box_pad_sentinel(grid)
    rows = [np.array([[0, 0, 0], [1, 0, 0], [-1, 2, 0]], dtype=np.int32),
            np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)]
    out, ngk = pad_gvecs_to_sentinel(rows, grid)

    assert out.shape == (2, 3, 3) and out.dtype == np.int32
    assert ngk.tolist() == [3, 2]
    for j, r in enumerate(rows):
        assert np.array_equal(out[j, :len(r)], r)          # physical verbatim
        assert np.array_equal(
            out[j, len(r):], np.broadcast_to(sent, (3 - len(r), 3)))
    assert pad_mask(ngk, 3).tolist() == [[True] * 3, [True, True, False]]


def test_rectangular_input_refuses_to_guess_its_own_extents():
    """A rectangular table carries no extents; inferring them from the
    pad value would make the pad marker define itself, which is exactly
    the circularity the sentinel exists to remove."""
    grid = (6, 6, 8)
    tab, ngk = pad_gvecs_to_sentinel(
        [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
         np.array([[0, 0, 0]], dtype=np.int32)], grid)
    with pytest.raises(ValueError, match="ngk_valid"):
        pad_gvecs_to_sentinel(tab, grid)
    with pytest.raises(ValueError, match="ngk_valid"):
        build_g_index_for_fft_box(tab, grid, 2)
    # With the extents it is idempotent — re-padding an already-padded
    # table is a no-op, which is what lets a consumer normalise anything.
    again, _ = pad_gvecs_to_sentinel(tab, grid, ngk_valid=ngk)
    assert np.array_equal(again, tab)


@pytest.mark.parametrize("grid,cutoff", [((8, 8, 8), 4.0), ((6, 6, 10), 6.0)])
def test_zeta_sphere_uses_the_same_pad_as_the_shared_routine(grid, cutoff):
    """The ζ producer must not have its own copy of the pad.

    Rebuilds ``gvec_components_padded`` from the sphere's own logical
    G-lists through ``pad_gvecs_to_sentinel`` and demands equality; a
    second, drifted sentinel in ``coulomb_sphere`` would show up here.
    Also pins the writer's ``sphere_idx_padded`` against the components
    table, since those are the two things ``isdf_header`` stores and a
    reader cross-references.
    """
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components

    nx, ny, nz = grid
    bvec = np.diag([1.0, 1.1, 0.9])
    q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.25, 0.0]])
    pkg = compute_per_q_bare_coulomb_components(
        fft_grid=grid, bvec=bvec, q_irr_frac=q, vcoul_cutoff_ry=cutoff)

    comps = pkg["gvec_components_padded"]            # (n_q, 3, ngkmax)
    ngk = np.asarray(pkg["ngk_per_q"])
    ngkmax = int(pkg["ngkmax"])
    assert comps.shape == (q.shape[0], 3, ngkmax)
    assert int(ngk.min()) < ngkmax, "fixture must actually pad some q"

    gvecs = np.ascontiguousarray(comps.transpose(0, 2, 1))
    rebuilt, _ = pad_gvecs_to_sentinel(
        [gvecs[j, :int(ngk[j])] for j in range(q.shape[0])], grid,
        ngkmax=ngkmax)
    assert np.array_equal(rebuilt, gvecs)

    # sphere_idx_padded is the wrapped-flat index of the SAME rows.
    flat = ((gvecs[..., 0] % nx) * ny * nz + (gvecs[..., 1] % ny) * nz
            + (gvecs[..., 2] % nz))
    assert np.array_equal(flat.astype(np.int32), pkg["sphere_idx_padded"])
    _, sentinel_flat = fft_box_pad_sentinel(grid)
    for j in range(q.shape[0]):
        assert np.all(pkg["sphere_idx_padded"][j, int(ngk[j]):]
                      == sentinel_flat)
        assert int(pkg["sphere_idx_padded"][j, 0]) == 0    # G=(0,0,0) first


@pytest.mark.skipif(not _WFN_FIXTURES, reason="no WFN fixtures in tests/")
@pytest.mark.parametrize("path", _WFN_FIXTURES,
                         ids=[os.path.basename(os.path.dirname(p))
                              for p in _WFN_FIXTURES])
def test_wfn_loader_pads_with_the_shared_sentinel(path):
    """ψ's producer goes through the same routine as ζ's."""
    from wfn_loader import WfnLoader

    with WfnLoader(path) as ld:
        grid = tuple(int(s) for s in ld.fft_grid)
        sent, _ = fft_box_pad_sentinel(grid)
        for spec in ("ibz", "full_bz"):
            g = np.asarray(ld.gvecs(k=spec))
            nv = np.asarray(ld.ngk_valid(k=spec))
            assert g.shape[1] == int(ld.ngkmax)
            for j in range(g.shape[0]):
                n = int(nv[j])
                assert np.array_equal(
                    g[j, n:], np.broadcast_to(sent, (g.shape[1] - n, 3))), \
                    f"{spec} k={j} pad rows are not the sentinel"


def test_zeta_loader_and_wfn_loader_expose_the_same_G_surface(tmp_path):
    """``ZetaLoader.gvecs`` / ``ngk_valid`` mirror ``WfnLoader``'s.

    The two files store the G axis differently — WFN.h5 flattens a
    RAGGED axis behind ``kpt_starts``, zeta_q.h5 is already
    ``ngkmax``-rectangular — and the whole point of the shared layer is
    that the difference stops at the read.  This asserts the two
    accessors agree in NAME, SHAPE, DTYPE and PAD, and that the ζ reader
    re-validates the on-disk components rather than trusting them.

    ``fft_grid`` here is ``(16, 16, 16)`` because that is what the shared
    fake WFN header carries, and ``isdf_header/gvec_components`` is
    meaningless on any other grid — something the existing
    ``_build_g_flat_zeta_file`` call sites get away with only because
    nothing before now read the components THROUGH the header's grid.
    """
    from tests.test_compute_all_V_q_g_flat import _build_g_flat_zeta_file
    from zeta_loader import ZetaLoader

    grid = (16, 16, 16)                    # == the fake WFN's mf_header
    path, _z, _q, comps, ngk_per_q, ngkmax = _build_g_flat_zeta_file(
        tmp_path, fft_grid=grid, kgrid=(2, 2, 1),
        bvec=np.diag([0.7, 0.7, 0.3]), cutoff=8.0, n_rmu=4, sys_dim=2)
    assert int(np.min(ngk_per_q)) < ngkmax, "fixture must actually pad"
    sent, _ = fft_box_pad_sentinel(grid)

    with ZetaLoader(path) as zl:                   # header-only, mesh=None
        assert tuple(int(s) for s in zl.fft_grid) == grid
        g = zl.gvecs(q="ibz")
        nv = zl.ngk_valid(q="ibz")
        assert g.shape == (comps.shape[0], ngkmax, 3) and g.dtype == np.int32
        assert nv.tolist() == np.asarray(ngk_per_q).tolist()
        # Same content as the on-disk (n_q, 3, ngkmax) table, transposed.
        assert np.array_equal(g, comps.transpose(0, 2, 1))
        for j in range(g.shape[0]):
            n = int(nv[j])
            assert np.array_equal(
                g[j, n:], np.broadcast_to(sent, (ngkmax - n, 3)))
        # An explicit q-subset resolves like WfnLoader's explicit k-list.
        assert np.array_equal(zl.gvecs(q=[1, 0]), g[[1, 0]])

    # The components table and mf_header's FFTgrid are stored SEPARATELY
    # and nothing used to check they agree — yet the components mean
    # nothing without the grid they were built on.  Rewriting the pad
    # rows to another grid's sentinel is what that mismatch looks like on
    # disk, and it must be caught at READ time rather than silently
    # overwritten by the re-pad.
    import shutil
    import h5py
    bad = str(tmp_path / "zeta_bad.h5")
    shutil.copy(path, bad)
    other, _ = fft_box_pad_sentinel((6, 6, 8))
    with h5py.File(bad, "a") as f:
        ds = f["isdf_header/gvec_components"]
        arr = ds[...]
        j = int(np.argmin(ngk_per_q))
        arr[j, :, int(ngk_per_q[j]):] = np.asarray(other)[:, None]
        ds[...] = arr
    with ZetaLoader(bad) as zl:
        with pytest.raises(ValueError, match="not the pad sentinel"):
            zl.gvecs(q="ibz")


# ---------------------------------------------------------------------------
# 3. "No physical G on the sentinel cell" — checked, not assumed
# ---------------------------------------------------------------------------

def test_refuses_a_physical_G_on_the_sentinel_cell_in_a_PADDED_row():
    """The ambiguity is per-row: the offending row must also have pad."""
    grid = (6, 6, 8)
    nx, ny, nz = grid
    sent, _ = fft_box_pad_sentinel(grid)
    wide = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int32)

    # Row 0 holds the corner AND is padded (ngk=2 < ngkmax=3).
    with pytest.raises(ValueError, match="pad sentinel"):
        pad_gvecs_to_sentinel(
            [np.array([[0, 0, 0], list(sent)], dtype=np.int32), wide], grid)
    # A different Miller index on the SAME cell is refused too — the cell
    # is what the gather reads, so Miller-triple equality would miss it.
    with pytest.raises(ValueError, match="pad sentinel"):
        pad_gvecs_to_sentinel(
            [np.array([[0, 0, 0], [nx // 2, ny // 2, nz // 2]],
                      dtype=np.int32), wide], grid)


def test_accepts_a_corner_G_in_a_row_that_has_no_pad():
    """Construct the case where the refusal must NOT fire.

    A row with ``ngk == ngkmax`` has no pad slots, so its corner G is
    unambiguous — nothing to confuse it with.  This is the NORMAL shape
    of a ζ sphere set, not a curiosity: the corner is the hardest G to
    admit, so the q whose sphere contains it is the widest sphere, i.e.
    the one that DEFINES ngkmax.  MEASURED on the ``(6,6,8)`` / cutoff
    8.0 fixture in ``tests/test_compute_all_V_q_g_flat.py``:
    ngk = [280, 282, 282, 288], and only the ngk=288 row admits the
    corner.  A whole-table check would refuse that file for an ambiguity
    it does not have — and that file is an input to two existing test
    modules.
    """
    grid = (6, 6, 8)
    sent, _ = fft_box_pad_sentinel(grid)
    # Row 0 is the widest (ngk == ngkmax == 3) and holds the corner;
    # row 1 is padded but corner-free.
    out, ngk = pad_gvecs_to_sentinel(
        [np.array([[0, 0, 0], [1, 0, 0], list(sent)], dtype=np.int32),
         np.array([[0, 0, 0], [2, 0, 0]], dtype=np.int32)], grid)
    assert out.shape == (2, 3, 3) and ngk.tolist() == [3, 2]
    assert np.array_equal(out[0, 2], np.asarray(sent))   # kept as PHYSICAL
    assert np.array_equal(out[1, 2], np.asarray(sent))   # written as PAD

    # …and a single unpadded row, e.g. a Γ-only deck.
    out1, ngk1 = pad_gvecs_to_sentinel(
        [np.array([[0, 0, 0], list(sent)], dtype=np.int32)], grid)
    assert out1.shape == (1, 2, 3) and ngk1.tolist() == [2]


@pytest.mark.skipif(not _WFN_FIXTURES, reason="no WFN fixtures in tests/")
@pytest.mark.parametrize("path", _WFN_FIXTURES,
                         ids=[os.path.basename(os.path.dirname(p))
                              for p in _WFN_FIXTURES])
def test_no_fixture_has_a_physical_G_on_the_sentinel_cell(path):
    """The invariant, MEASURED rather than argued, on every real deck.

    ``WfnLoader.gvecs`` would already have refused, so this restates it
    with an independent expression (a direct count over the physical
    rows) and reports the MARGIN — how far the sphere is from the corner
    — so a deck that is merely close shows up before it fails.
    """
    from wfn_loader import WfnLoader

    with WfnLoader(path) as ld:
        grid = tuple(int(s) for s in ld.fft_grid)
        corner = np.asarray([n // 2 for n in grid])
        for spec in ("ibz", "full_bz"):
            g = np.asarray(ld.gvecs(k=spec), dtype=np.int64)
            nv = np.asarray(ld.ngk_valid(k=spec), dtype=np.int64)
            phys = g[pad_mask(nv, g.shape[1])]
            wrapped = phys % np.asarray(grid, dtype=np.int64)[None, :]
            assert not np.any(np.all(wrapped == corner[None, :], axis=1))
            # Margin: the sphere's extent along each axis vs the corner.
            assert np.all(np.abs(phys).max(axis=0) < corner), \
                f"{path} {spec}: G-sphere reaches the box corner"


# ---------------------------------------------------------------------------
# 4. The masked scatter
# ---------------------------------------------------------------------------

def test_masked_scatter_equals_the_ragged_build():
    grid = (6, 6, 8)
    rows = [np.array([[0, 0, 0], [1, 0, 0], [-1, 2, 3], [2, 2, 2]],
                     dtype=np.int32),
            np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int32)]
    tab, ngk = pad_gvecs_to_sentinel(rows, grid, ngkmax=5)
    rect = build_g_index_for_fft_box(tab, grid, 5, ngk_valid=ngk)
    ragged = build_g_index_for_fft_box(rows, grid, 5)
    assert np.array_equal(rect, ragged)


def test_pad_then_scatter_everything_would_clobber_a_real_G():
    """Why MASKED and not pad-then-scatter-everything.

    Constructed so a physical G sits on the sentinel cell — a table
    ``pad_gvecs_to_sentinel`` refuses, which is the point: without the
    refusal, the unmasked scatter writes pad rows LAST (higher ``g``
    wins in a numpy fancy-index assignment) and the real coefficient
    index is replaced by a pad index.  The masked build is immune
    regardless, and this shows both halves.
    """
    grid = (6, 6, 8)
    sent, _ = fft_box_pad_sentinel(grid)
    cell = tuple(int(c) % int(n) for c, n in zip(sent, grid))
    ngkmax = 4
    # Physical: Γ at g=0, the corner at g=1.  Pad: g=2, 3.
    tab = np.broadcast_to(sent, (1, ngkmax, 3)).astype(np.int32).copy()
    tab[0, 0] = [0, 0, 0]
    tab[0, 1] = sent
    ngk = np.asarray([2], dtype=np.int32)

    masked = build_g_index_for_fft_box(tab, grid, ngkmax, ngk_valid=ngk)
    assert int(masked[0][cell]) == 1                 # the REAL G survives

    # The rejected alternative, written out: scatter every row.
    everything = build_g_index_for_fft_box(
        tab, grid, ngkmax, ngk_valid=np.asarray([ngkmax], dtype=np.int32))
    assert int(everything[0][cell]) == ngkmax - 1    # a PAD row won

    # …and this table is exactly what the producer refuses to build.
    with pytest.raises(ValueError, match="pad sentinel"):
        pad_gvecs_to_sentinel(tab[:, :2], grid, ngkmax=ngkmax,
                              ngk_valid=ngk)


def test_empty_cells_take_the_ngkmax_sentinel():
    grid = (4, 4, 4)
    rows = [np.array([[0, 0, 0]], dtype=np.int32)]
    gi = build_g_index_for_fft_box(rows, grid, 7)
    assert gi.shape == (1, 4, 4, 4)
    assert int(gi[0, 0, 0, 0]) == 0
    assert int(np.count_nonzero(gi == 7)) == 4 * 4 * 4 - 1


# ---------------------------------------------------------------------------
# 5. The consumer-side detector
# ---------------------------------------------------------------------------

def test_detector_arms_and_their_blind_spot():
    grid = (6, 6, 8)
    sent, _ = fft_box_pad_sentinel(grid)
    phys = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 3]], dtype=np.int32)

    refuse_padded_gvecs_without_mask(phys, grid)          # ragged: accepted
    refuse_padded_gvecs_without_mask(phys, None)

    one = np.concatenate([phys, np.asarray(sent)[None, :]], axis=0)
    with pytest.raises(ValueError, match="pad sentinel cell"):
        refuse_padded_gvecs_without_mask(one, grid)       # arm (a) sees 1 row
    refuse_padded_gvecs_without_mask(one, None)           # arm (b) is BLIND

    two = np.concatenate([phys, np.broadcast_to(sent, (2, 3))], axis=0)
    with pytest.raises(ValueError, match="duplicate row"):
        refuse_padded_gvecs_without_mask(two, None)       # arm (b) sees 2

    # Arm (b) does not depend on the pad VALUE — a legacy zero-padded
    # list is caught too, which is what makes it a backstop rather than a
    # restatement of arm (a).
    zeros = np.concatenate([phys, np.zeros((2, 3), dtype=np.int32)], axis=0)
    with pytest.raises(ValueError, match="duplicate row"):
        refuse_padded_gvecs_without_mask(zeros, None)
