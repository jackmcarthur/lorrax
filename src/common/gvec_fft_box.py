"""Sparse-G → dense-FFT-box: the padded G-axis, and the gather onto it.

This module owns TWO things that must not drift apart:

1. **The padded representation.**  Both ψ (WFN.h5 ``wfns``) and ζ
   (``zeta_q.h5`` ``zeta_q_G`` + ``isdf_header/gvec_components``) carry a
   per-k / per-q G-list of *different* logical length ``ngk`` on a
   *fixed* axis of width ``ngkmax``.  Coefficients in the pad slots are
   zero; G-vectors in the pad slots hold the **pad sentinel**, the Miller
   index sitting in the FFT box's Nyquist-corner cell
   ``(nx//2, ny//2, nz//2)``, read off the ``fftfreq`` tables.
   :func:`fft_box_pad_sentinel` is the single definition of that value;
   :func:`pad_gvecs_to_sentinel` is the single routine that builds the
   table, and it REFUSES a G-list that would make the sentinel ambiguous
   (see below).

   ζ produces the table from a cutoff sphere
   (:func:`common.coulomb_sphere.compute_per_q_bare_coulomb_components`);
   ψ produces it from the ragged on-disk ``wfns/gvecs``
   (:meth:`file_io.wfn_loader.WfnLoader.gvecs`).  Both call the routines
   here, so the two on-disk layouts converge on ONE in-memory form.

   *Why a sentinel and not zero.*  Zero pad rows are the Miller index
   ``(0,0,0)`` — the physical Γ component, which every G-sphere contains.
   A consumer that forgets the ``ngk`` mask therefore does not crash, it
   silently adds ``ngkmax − ngk`` extra copies of ψ(Γ) (see
   ``gw.kin_ion_io.get_kin_ion_k``, inside H₀'s ~500 eV cancellation).
   The sentinel cell is one no physical G of a PADDED row occupies, so
   the same mistake becomes *detectable*: within any row that has pad,
   "this row is the sentinel" ⇔ "this row is pad".  That equivalence is
   what :func:`pad_gvecs_to_sentinel` enforces — it is a checked
   precondition, not an assumption, and the check is scoped per row
   because that is exactly the scope in which the equivalence is used
   (a row with ``ngk == ngkmax`` has no pad to be confused with, and
   MEASURED, that is precisely where a ζ sphere does reach the corner).
   MEASURED on the ψ side: across all five WFN fixtures under ``tests/``,
   for both ``k='ibz'`` and ``k='full_bz'``, ZERO physical G lands on the
   sentinel cell in ANY row, padded or not — max ``|G|`` per axis runs
   ``(8, 8, 27)`` against a corner at ``(15, 15, 60)``, and similar.

2. **The gather.**  See ``GVEC_FFT_BOX_GATHER.md``.  For each k-point a
   lookup table maps every ``(nx, ny, nz)`` FFT-box cell to the
   ``g``-index within that k's coefficient slab (or ``ngkmax`` if the
   cell has no coefficient).  At runtime one ``jnp.take`` fills the whole
   FFT box — no scatter, no per-k loop.

Public API
----------
``fft_box_pad_sentinel``      — the pad Miller index + its flat-FFT slot.
``pad_mask``                  — ``(nk, ngkmax)`` bool, True on physical G.
``pad_gvecs_to_sentinel``     — build/validate the padded G table.
``refuse_padded_gvecs_without_mask``
                              — the consumer-side detector for a padded
                                list that lost its mask.
``build_g_index_for_fft_box`` — host-side precompute, once per WFN.
``make_fft_box_kernel``       — jitted shard_map kernel, reused per
                                band chunk.

Only ``make_fft_box_kernel`` needs jax, and it imports it in its body.
Everything above is pure numpy: the pad sentinel and the padded-table
build are host-side facts about an FFT grid, reasonable and testable
without a device, and a host-side producer (``coulomb_sphere``) does not
acquire an accelerator dependency by asking what the pad row is.  (The
``common`` package's own ``__init__`` still pulls jax, so `import
common.gvec_fft_box` does too — this is about the module's dependency,
not the package's.)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


__all__ = [
    "fft_box_pad_sentinel",
    "pad_mask",
    "pad_gvecs_to_sentinel",
    "refuse_padded_gvecs_without_mask",
    "build_g_index_for_fft_box",
    "make_fft_box_kernel",
]


# ===========================================================================
#  The padded representation
# ===========================================================================

def fft_box_pad_sentinel(
    fft_grid: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    """The pad slot for ``fft_grid``: its Miller index and its flat slot.

    Returns ``(components, flat_index)``:

    ``components`` : ``(3,)`` int32
        Miller index of the FFT box's Nyquist corner, taken from the
        ``fftfreq`` tables — i.e. ``(-nx/2, -ny/2, -nz/2)`` for even
        extents and ``((nx-1)/2, …)`` for odd ones.  Reading it off
        ``fftfreq`` (rather than writing ``-nx//2``) is what makes a
        consumer that recomputes ``G`` from ``flat_index`` land on the
        same triple, for even AND odd extents.
    ``flat_index`` : int
        ``(nx//2)·ny·nz + (ny//2)·nz + (nz//2)`` — the C-order flat
        position of that cell in an ``(nx, ny, nz)`` fftfreq-ordered box.
        Always in bounds, which is why a gather at a pad slot is safe
        even before the coefficients are zeroed.

    This is the ONE definition.  ``isdf_header``'s documented
    ``(-nx/2, -ny/2, -nz/2)``, ``coulomb_sphere``'s ``sphere_idx_padded``
    fill, ``WfnLoader.gvecs``'s pad rows and ``kin_ion_io``'s
    unmasked-pad refusal all resolve through here.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(f"fft_box_pad_sentinel: bad fft_grid {fft_grid!r}")
    components = np.array(
        [int(np.rint(np.fft.fftfreq(nx)[nx // 2] * nx)),
         int(np.rint(np.fft.fftfreq(ny)[ny // 2] * ny)),
         int(np.rint(np.fft.fftfreq(nz)[nz // 2] * nz))],
        dtype=np.int32)
    flat = (nx // 2) * ny * nz + (ny // 2) * nz + (nz // 2)
    return components, int(flat)


def pad_mask(ngk_valid, ngkmax: int) -> np.ndarray:
    """``(nk, ngkmax)`` bool — True on the physical G of each row.

    The one expression for "which slots are real"; every consumer that
    needs a 1/0 weight builds it from this (``psp.dft_operators``'s
    ``PaddedGVectors.mask`` is ``pad_mask(...).astype(float64)``).
    """
    ngk = np.asarray(ngk_valid, dtype=np.int64).reshape(-1)
    return np.arange(int(ngkmax), dtype=np.int64)[None, :] < ngk[:, None]


def _refuse_sentinel_collision(
    gvecs_padded: np.ndarray,
    ngk_valid: np.ndarray,
    fft_grid: tuple[int, int, int],
    *,
    where: str,
) -> None:
    """Refuse if a PHYSICAL G shares the pad sentinel's FFT-box cell.

    The padded representation says "sentinel row ⇔ pad slot".  That
    biconditional has one failure mode: a G-sphere large enough to reach
    the box's Nyquist corner.  Then the sentinel is also a real G, and
    every consumer that identifies pad by the row value (``isdf_header``'s
    documented pad marker, ``kin_ion_io``'s refusal, any future
    pad-then-scatter-everything g_index build) either drops a physical
    coefficient or double-counts it.  Refuse instead of trusting.

    Tested on the FFT-box CELL, not on Miller-triple equality: cell
    equality is the weaker premise (``+nx/2`` and ``-nx/2`` are different
    Miller indices but the same cell), so passing this check also licenses
    the cheaper pad-then-scatter build, whose pad rows WOULD win the
    sentinel cell on a ``g``-max scatter.

    Scoped PER ROW, and this is load-bearing rather than lenient: the
    biconditional is a statement about one row's own slots, so it can
    only be violated by a row that has BOTH a pad slot AND a physical G
    at the corner.  A row with ``ngk == ngkmax`` has no pad slot, so its
    corner G is unambiguous.  That case is not hypothetical — it is the
    normal shape of a ζ sphere set: the corner is the hardest G to admit,
    so the q whose sphere contains it is (weakly) the LARGEST sphere,
    i.e. the one that defines ``ngkmax`` and is therefore unpadded.
    MEASURED on the ``(6,6,8)`` / cutoff 8.0 ζ fixture in
    ``tests/test_compute_all_V_q_g_flat.py``: ngk = [280, 282, 282, 288],
    and the only q admitting the corner is the ngk=288 row.  A global
    check would refuse that file for an ambiguity it does not have.
    """
    ngk = np.asarray(ngk_valid, dtype=np.int64).reshape(-1)
    ngkmax = int(gvecs_padded.shape[1])
    padded_rows = ngk < ngkmax
    if not bool(padded_rows.any()):
        return                       # nothing is padded — nothing to confuse
    grid = [int(v) for v in fft_grid]
    g = np.asarray(gvecs_padded, dtype=np.int32)
    # Cell equality axis by axis rather than via a flat index: the whole
    # test is separable, and this keeps the temporaries at one
    # ``(nk, ngkmax)`` int32 at a time instead of a full int64
    # ``(nk, ngkmax, 3)`` (35 MB vs 210 MB at a 144-k / 20000-G deck).
    hit = pad_mask(ngk, ngkmax) & padded_rows[:, None]
    for ax in range(3):
        hit &= (g[..., ax] % np.int32(grid[ax])) == np.int32(grid[ax] // 2)
        if not bool(hit.any()):
            return
    k_bad = np.unique(np.nonzero(hit)[0])
    comps, _ = fft_box_pad_sentinel(fft_grid)
    raise ValueError(
        f"{where}: {int(hit.sum())} PHYSICAL G-vector(s) in PADDED rows k="
        f"{k_bad[:8].tolist()}{'…' if k_bad.size > 8 else ''} wrap to the "
        f"pad sentinel's FFT-box cell "
        f"{tuple(int(v) // 2 for v in fft_grid)} (Miller "
        f"{tuple(int(v) for v in comps)}) on grid "
        f"{tuple(int(v) for v in fft_grid)}.  The ngkmax-padded G layout "
        f"marks pad slots with exactly that cell, so 'this row is the "
        f"sentinel' would no longer mean 'this row is pad' and every "
        f"consumer that reads the pad marker (isdf_header/gvec_components, "
        f"gw.kin_ion_io's unmasked-pad refusal) becomes unsound.  This "
        f"means the G-sphere reaches the FFT box's Nyquist corner: either "
        f"the cutoff that built it is too large for this FFT grid, or the "
        f"FFT grid is too small for the cutoff.  Enlarge the grid.")


def pad_gvecs_to_sentinel(
    gvecs_per_k,
    fft_grid: tuple[int, int, int],
    *,
    ngkmax: int | None = None,
    ngk_valid=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the shared ``(nk, ngkmax, 3)`` sentinel-padded G table.

    THE single producer of the padded representation.  ζ calls it from
    ``common.coulomb_sphere`` on per-q cutoff spheres; ψ calls it from
    ``file_io.wfn_loader.WfnLoader.gvecs`` on the ragged on-disk
    ``wfns/gvecs``.  Both get the same in-memory object, so everything
    downstream (``build_g_index_for_fft_box``, ``pad_mask``,
    ``psp.dft_operators.padded_gvectors``) is written once.

    Parameters
    ----------
    gvecs_per_k
        Either a length-``nk`` sequence of ragged ``(ngk[k], 3)`` int
        arrays, or an already-rectangular ``(nk, ngkmax, 3)`` int array
        whose rows ``[k, ngk_valid[k]:]`` are pad of ANY value (they are
        overwritten).  ``ngk_valid`` is required for the rectangular
        form — a rectangular table does not carry its own extents, and
        guessing them from the pad value is exactly the circularity this
        module exists to remove.
    fft_grid
        ``(nx, ny, nz)``.
    ngkmax
        Width of the padded axis.  Defaults to ``max_k ngk[k]``; pass the
        file's own ``ngkmax`` when it may exceed that (BGW's header
        ``ngkmax`` is a max over the FILE's k, which for a subset request
        is larger than the subset's max).
    ngk_valid
        ``(nk,)`` logical extents.  Derived from the ragged shapes when
        ``gvecs_per_k`` is a sequence.

    Returns
    -------
    ``(gvecs_padded (nk, ngkmax, 3) int32, ngk_valid (nk,) int32)``

    Raises
    ------
    ValueError
        If some ``ngk[k] > ngkmax``, or if a physical G collides with the
        pad sentinel (see :func:`_refuse_sentinel_collision`).
    """
    sentinel, _ = fft_box_pad_sentinel(fft_grid)

    rect = (isinstance(gvecs_per_k, np.ndarray) and gvecs_per_k.ndim == 3)
    if rect:
        if ngk_valid is None:
            raise ValueError(
                "pad_gvecs_to_sentinel: a rectangular (nk, ngkmax, 3) "
                "gvecs table carries no logical extents; pass "
                "ngk_valid=... (e.g. WfnLoader.ngk_valid(k=...)).  "
                "Inferring the extents from the pad value would make the "
                "pad marker define itself.")
        src = np.asarray(gvecs_per_k, dtype=np.int32)
        ngk = np.asarray(ngk_valid, dtype=np.int32).reshape(-1)
        nk = int(src.shape[0])
        width = int(ngkmax) if ngkmax is not None else int(src.shape[1])
    else:
        rows = [np.asarray(g, dtype=np.int32).reshape(-1, 3)
                for g in gvecs_per_k]
        nk = len(rows)
        ngk = (np.asarray(ngk_valid, dtype=np.int32).reshape(-1)
               if ngk_valid is not None
               else np.asarray([r.shape[0] for r in rows], dtype=np.int32))
        width = int(ngkmax) if ngkmax is not None else (
            int(ngk.max()) if nk else 0)

    if nk != int(ngk.shape[0]):
        raise ValueError(
            f"pad_gvecs_to_sentinel: {nk} k-rows but ngk_valid has "
            f"{int(ngk.shape[0])} entries")
    if nk and int(ngk.max()) > width:
        bad = int(np.argmax(ngk))
        raise ValueError(
            f"pad_gvecs_to_sentinel: k={bad} has ngk={int(ngk[bad])} > "
            f"ngkmax={width}")

    out = np.empty((nk, width, 3), dtype=np.int32)
    out[...] = sentinel[None, None, :]
    if rect:
        keep = pad_mask(ngk, min(width, int(src.shape[1])))
        out[:, :keep.shape[1]][keep] = src[:, :keep.shape[1]][keep]
    else:
        for j, r in enumerate(rows):
            out[j, :int(ngk[j])] = r[:int(ngk[j])]

    _refuse_sentinel_collision(out, ngk, fft_grid,
                               where="pad_gvecs_to_sentinel")
    return out, ngk.astype(np.int32, copy=False)


def refuse_padded_gvecs_without_mask(
    gvecs, fft_grid=None, *, where: str = "G-list",
) -> None:
    """Refuse a single-k ``(nG, 3)`` padded G-list that lost its mask.

    The consumer-side counterpart of :func:`pad_gvecs_to_sentinel`, and
    it lives here so the two share one invariant instead of two copies of
    it.  A pad slot is a VALID FFT-box index (it has to be — a gather
    there must not run off the box), so an unmasked padded list does not
    crash: it returns a plausible matrix wrong by ``ngkmax − ngk`` extra
    copies of one component.  Its first caller is
    ``gw.kin_ion_io.get_kin_ion_k``, where that sits inside H₀'s ~500 eV
    cancellation.

    Two independent arms, both cheap host reductions over ``(nG, 3)``:

    **(a) The pad marker.**  Pad rows are
    :func:`fft_box_pad_sentinel`, so on any table this codebase produces,
    "a row wraps to the sentinel cell" ⇒ "that row is pad", and ONE such
    row fires.  Needs ``fft_grid``; skipped when the caller cannot supply
    one.  Tested on the box CELL, not on Miller equality, because the
    cell is what the ψ gather actually reads.

    *Where (a) is FALSE on a genuinely padded list:* nowhere — every pad
    row IS the sentinel by construction.

    *Where (a) fires on a genuinely ragged list — a REAL false positive,
    stated plainly:* if that k's own G-sphere contains the box's Nyquist
    corner.  :func:`pad_gvecs_to_sentinel` does NOT rule this out for
    such a k, because its refusal is scoped to rows that HAVE pad, and a
    ragged single-k list arriving here has none.  What rules it out is
    measurement, not construction: across every WFN available — the five
    fixtures under ``tests/`` and the production MoS₂ decks on Perlmutter
    (36×36×135 grid, 80 Ry, 144 k; 24×24×80, 144 k) — ZERO physical G
    lands on the corner cell in any k, with max ``|G|`` per axis
    ``(9, 9, 32)`` against a corner at ``(18, 18, 67)``.  That is a
    consequence of BGW sizing the FFT box for the DENSITY sphere while
    this list is the ψ sphere; a deck that violated it would be one whose
    ψ cutoff reached the density grid's Nyquist corner.  If one ever
    appears, the fix is to pass ``g_mask`` (the caller always can) or to
    enlarge the FFT grid — not to weaken the check, whose other direction
    guards a ~500 eV cancellation.

    **(b) The list is not a set.**  A physical G-sphere holds each Miller
    index at most once, and the full-BZ unfold ``G ↦ S·G − G_umklapp`` is
    injective, so it stays a set.  Any duplicate row therefore means
    padding — whatever the pad value happens to be.  Deliberately
    pad-value-agnostic: it is the backstop that survives a future change
    to the marker, and the only arm available without ``fft_grid``.

    *Where (b) is FALSE on a genuinely padded list:* exactly one pad row,
    with (a) unavailable.  That is the same blind spot the pre-2026-08
    guard had (it tested for ``> 1`` all-zero row); arm (a) closes it
    whenever the grid is known.  MEASURED premise for (b): across all
    five WFN fixtures under ``tests/``, for both ``k='ibz'`` and
    ``k='full_bz'``, the maximum duplicate count within one k's physical
    G-list is 0.
    """
    g = np.asarray(gvecs)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(
            f"refuse_padded_gvecs_without_mask: expected (nG, 3), got "
            f"{g.shape}")
    n_g = int(g.shape[0])
    if n_g < 2:
        return
    g = g.astype(np.int64, copy=False)

    if fft_grid is not None:
        grid = np.asarray([int(v) for v in fft_grid], dtype=np.int64)
        if grid.shape == (3,) and bool((grid > 0).all()):
            _, sentinel_flat = fft_box_pad_sentinel(tuple(grid))
            w = g % grid[None, :]
            flat = w[:, 0] * grid[1] * grid[2] + w[:, 1] * grid[2] + w[:, 2]
            n_sent = int(np.count_nonzero(flat == sentinel_flat))
            if n_sent:
                raise ValueError(
                    f"{where} has {n_sent} row(s) on the FFT-box pad "
                    f"sentinel cell {tuple(int(v) // 2 for v in grid)} "
                    f"(grid {tuple(int(v) for v in grid)}), so it is "
                    f"ngkmax-padded, but no g_mask was passed.  Pass the "
                    f"mask from psp.dft_operators.padded_gvectors(...)"
                    f".at(ik).  Unmasked, those rows add {n_sent} spurious "
                    f"copies of the Nyquist-corner component to every "
                    f"matrix element — silently.")

    # (b) Duplicate-row backstop.  Exact integer packing, not hashing:
    # |G| never approaches 2**20 on any FFT grid, so three components fit
    # losslessly in one int64 key and ``unique`` is exact.
    if int(np.abs(g).max(initial=0)) < (1 << 20):
        key = (((g[:, 0] + (1 << 20)) << 42)
               | ((g[:, 1] + (1 << 20)) << 21)
               | (g[:, 2] + (1 << 20)))
    else:                                # pathological grid: exact, slower
        key = np.ascontiguousarray(g).view(np.dtype(
            [('a', g.dtype), ('b', g.dtype), ('c', g.dtype)])).ravel()
    n_dup = n_g - int(np.unique(key).size)
    if n_dup:
        raise ValueError(
            f"{where} has {n_dup} duplicate row(s), so it is ngkmax-padded "
            f"(a physical G-sphere is a SET — the full-BZ unfold "
            f"G ↦ S·G − G_umklapp is injective, so it stays one), but no "
            f"g_mask was passed.  Pass the mask from "
            f"psp.dft_operators.padded_gvectors(...).at(ik).  Unmasked, "
            f"each duplicate re-adds a physical component to every matrix "
            f"element — silently.")


# ===========================================================================
#  The gather index
# ===========================================================================

def build_g_index_for_fft_box(
    gvecs_per_k: Sequence[np.ndarray] | np.ndarray,
    fft_grid: tuple[int, int, int],
    ngkmax: int,
    *,
    ngk_valid=None,
) -> np.ndarray:
    """Precompute ``g_index[k, nx, ny, nz]``.

    For each k and each FFT-box cell ``(nx, ny, nz)``, the entry is the
    position ``g`` in ``[0, ngk[k])`` such that ``gvecs[k, g] % fft_grid
    == (nx, ny, nz)``, or ``ngkmax`` (a sentinel used by the runtime
    kernel to gather zero for empty cells).

    Parameters
    ----------
    gvecs_per_k : ``(nk, ngkmax, 3)`` int array (preferred) …
        …the shared padded representation, straight from
        :func:`pad_gvecs_to_sentinel` / ``WfnLoader.gvecs``.  ``ngk_valid``
        is then REQUIRED.  A length-``nk`` sequence of ragged
        ``(ngk[k], 3)`` arrays is also accepted (extents come from the
        shapes).
    fft_grid : (nx, ny, nz)
    ngkmax : int
        Max G-count across all k; also the empty-cell sentinel VALUE of
        the returned table (unrelated to the pad G sentinel).
    ngk_valid : ``(nk,)`` int, keyword-only
        Logical extent per k.  Required for the rectangular form.

    Returns
    -------
    g_index : ``(nk, nx, ny, nz)`` int32

    Notes
    -----
    The scatter is **masked**, not pad-then-scatter-everything: only
    slots ``g < ngk[k]`` are written.  The alternative — scatter all
    ``ngkmax`` rows and let the pad rows land on the sentinel cell —
    would let a pad row (higher ``g``, hence last writer) WIN that cell
    and shadow a physical G sitting there.  Masking removes the ordering
    dependence entirely; :func:`pad_gvecs_to_sentinel` separately refuses
    tables where such a physical G exists at all.

    Vectorised over k: one fancy-index scatter for the whole table, no
    Python k-loop, no ragged ``Sequence`` in the hot path.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    # int32 throughout: Miller indices and their residues both fit, and
    # the ``(nk, ngkmax, 3)`` wrap temporary below is the largest array
    # this function touches besides its own output.
    grid = np.asarray((nx, ny, nz), dtype=np.int32)

    if isinstance(gvecs_per_k, np.ndarray) and gvecs_per_k.ndim == 3:
        gvecs = np.asarray(gvecs_per_k, dtype=np.int32)
        if ngk_valid is None:
            raise ValueError(
                "build_g_index_for_fft_box: a rectangular (nk, ngkmax, 3) "
                "gvecs table needs ngk_valid=... — its pad rows carry the "
                "FFT-box pad sentinel, and scattering them would shadow "
                "whatever real G shares that cell.")
        ngk = np.asarray(ngk_valid, dtype=np.int64).reshape(-1)
    else:
        rows = [np.asarray(g, dtype=np.int32).reshape(-1, 3)
                for g in gvecs_per_k]
        ngk = np.asarray([r.shape[0] for r in rows], dtype=np.int64)
        width = int(ngk.max()) if len(rows) else 0
        gvecs = np.zeros((len(rows), width, 3), dtype=np.int32)
        for k, r in enumerate(rows):
            gvecs[k, :r.shape[0]] = r

    nk, width = int(gvecs.shape[0]), int(gvecs.shape[1])
    if nk != int(ngk.shape[0]):
        raise ValueError(
            f"build_g_index_for_fft_box: {nk} k-rows but ngk_valid has "
            f"{int(ngk.shape[0])} entries")
    if nk and int(ngk.max()) > int(ngkmax):
        bad = int(np.argmax(ngk))
        raise ValueError(
            f"k={bad}: ngk={int(ngk[bad])} > ngkmax={int(ngkmax)}")

    g_index = np.full((nk, nx, ny, nz), int(ngkmax), dtype=np.int32)
    if nk == 0 or width == 0:
        return g_index

    wrapped = gvecs % grid[None, None, :]                  # (nk, width, 3)
    valid = pad_mask(ngk, width)                           # (nk, width) bool
    k_of = np.broadcast_to(
        np.arange(nk, dtype=np.int32)[:, None], (nk, width))
    g_of = np.broadcast_to(
        np.arange(width, dtype=np.int32)[None, :], (nk, width))
    g_index[k_of[valid],
            wrapped[..., 0][valid],
            wrapped[..., 1][valid],
            wrapped[..., 2][valid]] = g_of[valid]
    return g_index


def make_fft_box_kernel(
    mesh: "Mesh",                                       # noqa: F821
    nk: int,
    ngkmax: int,
    nb_padded: int,
    nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Build a jitted ``(cnk_slab, g_index) → psi_G_box`` kernel.

    Inputs (global, cnk_slab band-sharded over the combined ``(x,y)``
    mesh axes, g_index replicated):

    ``cnk_slab``     ``(nb_padded, nspinor, nk, ngkmax, 2)`` f64 — the
                     ``re/im``-packed output of ``read_kchunk_union_sharded``
                     with ``kchunk_axis=2``.
    ``g_index``      ``(nk, nx, ny, nz)`` int32 from
                     :func:`build_g_index_for_fft_box`.

    Output:

    ``psi_G_box``    ``(nk, nb_padded, nspinor, nx, ny, nz)`` c128
                     sharded ``P(None, ('x','y'), None, None, None, None)``.
    """
    # jax is imported HERE, not at module scope: everything above this
    # function is pure numpy, and host-side producers (coulomb_sphere,
    # and any tool that only wants the pad sentinel) must not have to
    # bring up the accelerator stack to ask what the pad row is.
    import jax
    import jax.numpy as jnp
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    mesh_x = int(mesh.shape["x"])
    mesh_y = int(mesh.shape["y"])
    world_size = mesh_x * mesh_y
    if nb_padded % world_size != 0:
        raise ValueError(f"nb_padded={nb_padded} not divisible by {world_size}")
    bands_per_rank = nb_padded // world_size
    nx, ny, nz = (int(v) for v in fft_grid)

    def _per_rank(cnk_slab, g_index):
        # (bpr, ns, nk, ngkmax, 2) f64 → (bpr, ns, nk, ngkmax) c128
        cnk = cnk_slab[..., 0] + 1j * cnk_slab[..., 1]
        # Append one zero slot so the sentinel index (== ngkmax) gathers
        # zero.  `_padded` has the extra slot on the G axis only.
        zero_slot = jnp.zeros(
            (bands_per_rank, nspinor, nk, 1), dtype=jnp.complex128)
        cnk_padded = jnp.concatenate([cnk, zero_slot], axis=-1)

        # Single flat-axis gather: view cnk_padded as
        # (bpr, ns, nk * (ngkmax + 1)) and look up a k-and-g-combined
        # index per FFT cell.  One kernel launch for every (k, box cell).
        cnk_flat = cnk_padded.reshape(
            bands_per_rank, nspinor, nk * (ngkmax + 1))
        k_stride = ngkmax + 1
        flat_index = (
            jnp.arange(nk, dtype=jnp.int32)[:, None, None, None] * k_stride
            + g_index
        )  # (nk, nx, ny, nz)
        gathered = jnp.take(cnk_flat, flat_index, axis=2)
        # Move nk to the front for the canonical output layout.
        return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))

    return jax.jit(shard_map(
        _per_rank, mesh=mesh,
        in_specs=(
            P(("x", "y"), None, None, None, None),   # cnk_slab
            P(None, None, None, None),                # g_index
        ),
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_vma=False,
    ))
