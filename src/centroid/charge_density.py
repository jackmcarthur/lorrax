"""Weighting-density providers for k-means ISDF centroid selection.

The k-means weight decides WHERE the ISDF quadrature has points, hence
which states the ISDF can represent at all.  Two weights are offered
(``kmeans_cli --centroid-weight``):

* ``charge_density`` — the ground-state (OCCUPIED) ρ(r), sources 1 & 2
  below.  Correct for valence/near-gap work; starves any region the
  occupied states do not occupy.
* ``band_range`` — :func:`rho_from_band_range`, w(r) = Σ_{n∈range} Σ_k
  w_k |ψ_nk(r)|² over the bands the calculation actually uses.

Two ρ(r) sources are supported:

1. ``rho_from_qe_save(save_dir)`` — read the already-symmetrized valence
   density from QE's ``<prefix>.save/charge-density.hdf5``. This is what QE
   wrote at the end of SCF and is the physically correct, point-group-
   symmetric ρ_val(r). Fastest of the two paths — no recomputation, just
   a single FFT of ρ(G) → ρ(r). Requires access to the QE ``.save`` dir.

2. ``rho_from_wfn_ibz(wfn, sym, n_val=None)`` — compute
   ρ(r) = Σ_k w_k Σ_n |ψ_nk(r)|² directly from the IBZ wavefunctions in
   ``WFN.h5``, using the k-weights stored in the file. This is cheaper
   than the previous unfold-every-k approach in
   ``centroid/get_charge_density.py`` (now removed): for a 4×4×4 k-grid
   with 48 symmetries → 8 IBZ points, this does 8× fewer FFTs.

   Caveat: the raw IBZ sum is *not* point-group symmetrized. For
   symmetry-preserving centroid selection prefer the ``qe_save`` path, or
   apply ``SymMaps`` symmetrization afterwards.

``get_charge_density(...)`` is the unified entry point. When called without
an explicit source, it auto-detects an adjacent ``<prefix>.save`` directory
and uses it if present, falling back to the IBZ-sum path otherwise.

Both providers return a real numpy array of shape ``wfn.fft_grid`` = (Nx,
Ny, Nz) on the QE FFT grid, in the same convention as the previous
implementation (total electron number ≈ ``np.sum(ρ)``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from file_io import WfnLoader as WFNReader
from common import symmetry_maps


# ═══════════════════════════════════════════════════════════════════════
# Source 1: read QE's symmetrized ρ_val(r) directly from charge-density.hdf5
# ═══════════════════════════════════════════════════════════════════════

def rho_from_qe_save(save_dir: str | os.PathLike) -> np.ndarray:
    """Read ρ_val(r) from a QE ``<prefix>.save/charge-density.hdf5``.

    Returns ρ already summed over bands and k-points, point-group
    symmetrized, and on the QE dense FFT grid — exactly what QE wrote.
    Valence only (NLCC excluded).

    Args:
        save_dir: path to the ``<prefix>.save`` directory (e.g.
            ``runs/.../qe/scf/silicon.save``). Must contain
            ``data-file-schema.xml`` and ``charge-density.hdf5``.

    Returns:
        rho_r: (Nx, Ny, Nz) float64 real-space valence density.
    """
    # Imported lazily because qe_save_reader depends on h5py + xml parsing
    # that aren't needed on the wfn-ibz path.
    from file_io.qe_save_reader import CrystalData

    save_dir = Path(save_dir)
    if not (save_dir / "charge-density.hdf5").is_file():
        raise FileNotFoundError(f"{save_dir}/charge-density.hdf5")

    crystal = CrystalData.from_qe_save(str(save_dir))
    rho_r, _rho_G = crystal.load_charge_density()
    return np.asarray(rho_r, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Source 2: IBZ wavefunction sum (no full-BZ unfold)
# ═══════════════════════════════════════════════════════════════════════

def _load_wfn_k_fftbox_ibz(wfn: WFNReader, n_val: int) -> jnp.ndarray:
    """Load IBZ wavefunctions into the FFT box, shape (nk_irr, n_val, nspinor, Nx, Ny, Nz).

    Raw IBZ coefficients scattered onto the QE FFT grid — no fractional-
    translation phase is applied because we only want |ψ|² downstream and
    phases drop out of the modulus.

    Uses :class:`file_io.wfn_loader.WfnLoader` (eager backend) plus
    :func:`common.wfn_transforms.to_box`.  P5 will switch the caller to
    pass a ``WfnLoader`` directly so this transient construction goes
    away.

    PROCESS-LOCAL by construction.  ρ here is a pure function of the WFN,
    identical on every rank, so it is computed redundantly per rank rather
    than as a global object — the same contract
    :func:`common.wfn_transforms.load_kpoint_fftbox_local` uses.  The
    previous body paired a GLOBAL band-sharded ``loader.load`` with a 1x1
    mesh built from ``jax.devices()[:1]``; ``jax.devices()`` is the global
    device list, so that mesh is process 0's device **on every rank**, and
    every rank but 0 SIGSEGVs the moment the boxed FFT executes on it.
    Measured at N_mu=10015: rc=139 at P=4 AND at P=64, faulting in
    ``psp/get_DFT_mtxels.py:196``'s ``jnp.fft.ifftn`` (wk_REL jobs 7879470 /
    7879492 / 7879495, all frames identical).  Rank 0 sailed through, which
    is why the logs showed exactly one rank's worth of progress banners.
    """
    from file_io.wfn_loader import WfnLoader
    from common.wfn_transforms import to_box

    # single_device_mesh IS the process-local mesh: wfn_transforms.
    # process_local_mesh is an alias for this exact object (one per
    # process, owned by common.collectives).  Import it from its owner.
    from common.collectives import single_device_mesh

    with WfnLoader(wfn._filename) as loader:
        psi = loader.load_process_local(bands=(0, n_val), k="ibz")
        return to_box(psi, loader.box_index(k="ibz"), loader.fft_grid,
                      mesh=single_device_mesh())


def rho_from_wfn_ibz(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    n_val: int | None = None,
) -> np.ndarray:
    """Sum ρ(r) = Σ_k w_k Σ_n |ψ_nk(r)|² over IBZ k-points.

    Delegates the actual arithmetic to ``psp.get_DFT_mtxels.compute_valence_density``,
    which handles both the "wfn grid == ρ grid" case (one iFFT per band)
    and the "ecutrho > 4·ecutwfc" case (scatter into a larger ρ grid). It
    picks up the IBZ k-weights automatically when the leading dim of
    ``wfn_k`` equals ``len(wfn.kweights)``.

    Args:
        wfn: open ``WFNReader``.
        sym: matching ``SymMaps`` (used only for the wfn→ρ re-embedding
            path when ``ecutrho > 4·ecutwfc``).
        n_val: number of (occupied) bands to include. Defaults to
            ``wfn.nelec`` (all valence electrons for spinor wfn, i.e. one
            band per electron).

    Returns:
        rho_r: (Nx, Ny, Nz) float64 real-space valence density.

        Not point-group symmetrized. The density is the correct IBZ sum
        weighted by ``wfn.kweights``, but cross-star-member averaging is
        not applied — two k-points related by a rotation contribute the
        un-rotated |ψ|² at each real-space point. For centroid selection
        this is usually fine; for cases where symmetric centroid output
        is required, use ``rho_from_qe_save`` instead.
    """
    # Lazy import: psp brings in JAX + pseudos code that the qe_save path
    # does not need.
    from psp.get_DFT_mtxels import compute_valence_density

    if n_val is None:
        n_val = int(wfn.nelec)

    wfn_k = _load_wfn_k_fftbox_ibz(wfn, n_val)
    rho_jax = compute_valence_density(wfn_k, sym, wfn)
    return np.asarray(rho_jax, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Source 3: band-range weight — Σ_{n ∈ range} Σ_k w_k |ψ_nk(r)|²
# ═══════════════════════════════════════════════════════════════════════

def symmetrize_on_grid(field: np.ndarray, sym_ops: np.ndarray) -> np.ndarray:
    """Average ``field`` over a symmorphic integer point group on the grid.

    ``sym_ops`` are the r-action matrices ``M`` (BGW ``Rinv``, τ=0) of a
    group that maps the FFT grid to itself, e.g. the output of
    :func:`centroid.orbit_syms.recover_symmorphic_density_point_group`.
    Returns ``(1/|G|) Σ_M field[(M·n) mod N]`` — invariant because the
    group is closed.
    """
    f = np.asarray(field, dtype=np.float64)
    N = np.asarray(f.shape, dtype=np.int64)
    ops = np.asarray(sym_ops, dtype=np.int64).reshape(-1, 3, 3)
    if ops.shape[0] <= 1:
        return f
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in N), indexing="ij")
    n_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)   # (P,3)
    flat = f.ravel()
    acc = np.zeros_like(flat)
    for M in ops:
        img = (n_idx @ M.T) % N[None, :]
        acc += flat[img[:, 0] * (N[1] * N[2]) + img[:, 1] * N[2] + img[:, 2]]
    return (acc / ops.shape[0]).reshape(f.shape)


def rho_from_band_range(
    wfn: WFNReader,
    band_range: tuple[int, int],
    *,
    sym_ops: np.ndarray | None = None,
    chunk_gb: float = 4.0,
    verbose: bool = True,
    dist_mesh=None,
    chunk_bands: int | None = None,
) -> np.ndarray:
    """k-means weight from the density of the BAND RANGE IN USE.

    ``w(r) = Σ_{n ∈ [b_lo, b_hi)} Σ_k w_k |ψ_nk(r)|²`` (the k-average over
    the WFN's stored k-set, using its k-weights), in the same
    normalisation as :func:`rho_from_qe_save`.

    WHY THIS FEATURE EXISTS: the occupied-only ρ(r) is entirely inside the
    slab, so a ρ-weighted k-means puts ZERO centroids in the vacuum and the
    vacuum-localized far-conduction states have no quadrature support —
    their ⟨nk|V_H|nk⟩ (a pure centroid sum) comes back +139.75 eV where the
    truth is −139.6 eV, and the whole error lands on Vxc = E_dft − kin_ion
    − V_H (|ΔVxc| vs QE correlates with the vacuum weight at +0.958).
    Weighting by the bands actually in use puts centroids where those
    states live.

    Parameters
    ----------
    wfn : open ``WFNReader`` (only ``_filename``/``kweights``/
        ``cell_volume`` are used; ψ is streamed through ``WfnLoader``).
    band_range : ``(b_lo, b_hi)``, 0-based half-open.
    sym_ops : optional (n_op, 3, 3) integer r-action matrices.  The raw
        k-sum is NOT point-group symmetric (star members contribute
        un-rotated |ψ|² at each r); pass the recovered density point group
        to symmetrize, so the weight cannot itself break the k-star
        symmetry the orbit closure is there to protect.
    chunk_gb : band-chunk size target for the r-space buffer.
    dist_mesh : the run's GLOBAL mesh (``devices.size == process_count()``).
        Supplied ⇒ the band sweep is split across ranks and reassembled with
        one psum per chunk; omitted ⇒ every rank recomputes the whole sweep
        (the historical behaviour).  This is the only knob: there is no env
        opt-out, because the distributed form is a work split, not a
        different quadrature.  NOTE the split also cuts the per-rank ψ read
        by ``P`` — this loader is opened process-locally (eager h5py), so
        ranks reading disjoint bands is safe; do not "improve" it into a
        collective read without making the call counts rank-independent.

    Returns
    -------
    (Nx, Ny, Nz) float64 real-space weight on the WFN FFT grid.
    """
    import jax
    from file_io.wfn_loader import WfnLoader
    from common.wfn_transforms import to_rbox

    from common.collectives import (single_device_mesh, process_count,
                                    process_rank, psum_replicate)

    b_lo, b_hi = int(band_range[0]), int(band_range[1])
    if b_hi <= b_lo:
        raise ValueError(f"empty band range: {band_range}")

    with WfnLoader(wfn._filename) as loader:
        nb_file = int(loader.nbands)
        if b_hi > nb_file:
            raise ValueError(
                f"--weight-bands upper edge {b_hi} exceeds the WFN's "
                f"{nb_file} bands; lower it or regenerate the NSCF.")
        fft_grid = tuple(int(s) for s in loader.fft_grid)
        n_r = int(np.prod(fft_grid))
        nspinor = int(loader.nspinor)
        g_index = loader.box_index(k="ibz")
        n_k = int(g_index.shape[0])
        kw = np.asarray(wfn.kweights, dtype=np.float64)[:n_k]
        # PROCESS-LOCAL, same reasoning (and same measured SIGSEGV) as
        # ``_load_wfn_k_fftbox_ibz`` above: this weight is a pure function of
        # the WFN and must not be built as a global band-sharded object on a
        # mesh pinned to ``jax.devices()[0]``.  single_device_mesh() IS the
        # process-local mesh (process_local_mesh is its alias).
        mesh = single_device_mesh()
        # r-space buffer is (n_k, nb, nspinor, Nr) complex128 → size the
        # band chunk against the budget (≥1 band, ≤ the whole range).
        per_band = n_k * nspinor * n_r * 16
        nb_chunk = int(max(1, min(b_hi - b_lo,
                                  (chunk_gb * 1024 ** 3) // max(per_band, 1))))
        # The memory budget alone sizes chunks for a SERIAL sweep, where
        # fewer/larger chunks are strictly better.  Distributed, the chunk is
        # also the unit of work: at 600 bands / 4 GB this deck yields THREE
        # chunks, which caps the split at 3 ranks no matter how many are
        # available (measured job 7885968: 0.70x, i.e. slower).  Shrink to at
        # least one chunk per rank -- never above the budget, so peak memory
        # only falls.  Chunk boundaries are part of the summation grouping,
        # so this changes the weight in the last bits; that is measured
        # against the legacy chunking rather than assumed harmless.
        if chunk_bands is not None:
            nb_chunk = int(max(1, min(b_hi - b_lo, chunk_bands)))
        elif dist_mesh is not None and process_count() > 1:
            nb_chunk = int(max(1, min(
                nb_chunk,
                -(-(b_hi - b_lo) // int(process_count())))))
        scale = float(np.sqrt(n_r / float(wfn.cell_volume)))
        kw_j = jnp.asarray(kw).reshape(-1, 1, 1, 1, 1, 1)  # (n_k,b,s,x,y,z)
        chunks = [(lo, min(lo + nb_chunk, b_hi))
                  for lo in range(b_lo, b_hi, nb_chunk)]
        world, rank = process_count(), process_rank()
        # The band sum is separable, so at P>1 each rank computes only the
        # chunks it owns and one psum per chunk puts the whole back on every
        # rank.  Ownership is round-robin over the SAME canonical chunk list
        # on every rank, so all ranks make the same number of psum calls (a
        # non-owner contributes exact zeros) — a rank-dependent call count
        # would deadlock.  Accumulating the psummed chunk totals in canonical
        # order reproduces the serial left-fold; adding zeros is exact in
        # IEEE-754, so the only thing that can move a bit is XLA compiling
        # the per-chunk reduction differently when it is not fused into the
        # accumulate.  That is measured, not assumed (see docs); P=1 keeps
        # the original fused loop verbatim and cannot regress.
        distribute = world > 1 and dist_mesh is not None
        if verbose:
            how = (f"{len(chunks)} chunk(s) over {world} ranks, one psum each"
                   if distribute else
                   f"{len(chunks)} chunk(s), replicated on every rank"
                   + ("" if world <= 1 else
                      "  [no dist_mesh passed — sweep NOT distributed]"))
            print(f"[band_range weight] bands [{b_lo},{b_hi}) over {n_k} "
                  f"stored k (Σw_k={kw.sum():.4f}), grid {fft_grid}, "
                  f"chunk={nb_chunk} bands, {how}")
        if not distribute:
            w = jnp.zeros(fft_grid, dtype=jnp.float64)
            for lo, hi in chunks:
                psi = loader.load_process_local(bands=(lo, hi), k="ibz")
                psi_r = to_rbox(psi, g_index, fft_grid, mesh=mesh, norm="ortho")
                # ``load_process_local`` returns exactly ``hi - lo`` bands (no
                # mesh-divisibility padding — nothing about it is global), so
                # this slice is a no-op there and still drops ``load``'s pad
                # rows at P=1.
                psi_r = psi_r[:, :hi - lo] * scale   # drop band-axis pad rows
                w = w + jnp.sum(
                    (psi_r.real ** 2 + psi_r.imag ** 2) * kw_j, axis=(0, 1, 2))
            w = np.asarray(jax.device_get(w), dtype=np.float64)
        else:
            # ONE psum for the whole sweep, not one per chunk.  Per-chunk
            # psums are bit-identical to the serial left-fold, but the
            # collective's FIXED cost dominates at this payload (368 KB):
            # measured 0.85 s per call, job 7885969 — 4 chunks made the
            # distributed sweep 12x SLOWER than replicated on the sparse leg,
            # and at P=64 it would have cost ~50 s of pure latency.  Summing
            # locally first and reducing once regroups the additions, which
            # perturbs the last bits exactly like re-chunking does (measured
            # 6.6e-16 relative, ~3 ulp).  The weight is a SAMPLING DENSITY
            # whose consumers snap to grid points, so the gate that matters
            # is "the centroid set does not move", not bit-identity.
            w_loc = jnp.zeros(fft_grid, dtype=jnp.float64)
            for i, (lo, hi) in enumerate(chunks):
                if (i % world) != rank:
                    continue
                psi = loader.load_process_local(bands=(lo, hi), k="ibz")
                psi_r = to_rbox(psi, g_index, fft_grid, mesh=mesh,
                                norm="ortho")
                psi_r = psi_r[:, :hi - lo] * scale
                w_loc = w_loc + jnp.sum(
                    (psi_r.real ** 2 + psi_r.imag ** 2) * kw_j, axis=(0, 1, 2))
            w = psum_replicate(
                np.asarray(jax.device_get(w_loc), dtype=np.float64), dist_mesh)

    if sym_ops is not None:
        n_op = int(np.asarray(sym_ops).reshape(-1, 3, 3).shape[0])
        w_sym = symmetrize_on_grid(w, sym_ops)
        if verbose:
            dev = float(np.max(np.abs(w_sym - w)) / (np.max(np.abs(w)) or 1.0))
            print(f"[band_range weight] symmetrized over {n_op} op(s); "
                  f"raw k-sum asymmetry was {dev:.2e} (relative)")
        w = w_sym
    if verbose:
        print(f"[band_range weight] Σ_r w = {w.sum():.4f} over {b_hi - b_lo} "
              f"bands (same normalisation as rho_from_qe_save, whose Σ_r ρ "
              f"covers the {int(wfn.nelec)} occupied)")
    return w


# ═══════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════

def _autodetect_save_dir(start: str | os.PathLike = ".") -> Path | None:
    """Look for ``<prefix>.save/charge-density.hdf5`` in cwd and its parents.

    The kmeans CLI is typically launched from a ``00_lorrax/`` run-subdir
    containing a symlinked ``WFN.h5``; the QE ``<prefix>.save`` lives in a
    sibling ``qe/scf/`` (or ``qe/nscf/`` via a nested symlink). We search
    (a) the current directory, (b) its direct children named ``*.save``,
    and (c) the nearest ancestor's ``qe/scf`` and ``qe/nscf``.
    """
    start = Path(start).resolve()
    # (a) & (b): save dir in cwd or as child
    candidates = [start] + sorted(start.glob("*.save"))
    # (c): common layouts relative to the run tree
    for ancestor in (start, *start.parents):
        for sub in ("qe/scf", "qe/nscf"):
            candidates.extend(sorted((ancestor / sub).glob("*.save")))
        if ancestor.name == "runs" or ancestor == ancestor.parent:
            break  # do not walk above the sandbox root
    for c in candidates:
        if (c / "charge-density.hdf5").is_file():
            return c
    return None


def get_charge_density(
    wfn: WFNReader | None = None,
    sym: symmetry_maps.SymMaps | None = None,
    *,
    source: str = "auto",
    save_dir: str | os.PathLike | None = None,
    n_val: int | None = None,
) -> np.ndarray:
    """Return ρ_val(r) on the QE FFT grid. Dispatch to one of two sources.

    Args:
        wfn: open ``WFNReader``. Required for ``source='wfn_ibz'`` and for
            the ``'auto'`` fallback path.
        sym: matching ``SymMaps``. Same requirement as ``wfn``.
        source: which provider to use.

            * ``'qe_save'`` — read from ``save_dir/charge-density.hdf5``.
            * ``'wfn_ibz'`` — compute Σ_k w_k Σ_n |ψ_nk|² from ``wfn``.
            * ``'auto'`` (default) — prefer ``qe_save`` if a ``*.save``
              directory with a charge-density.hdf5 is findable (either via
              ``save_dir`` argument or by walking up from ``$PWD``); else
              fall back to ``wfn_ibz``.

        save_dir: explicit QE ``<prefix>.save`` path. If ``None`` and
            ``source`` is ``'qe_save'`` or ``'auto'``, the sandbox layout
            (``./qe/scf/<prefix>.save``, etc.) is probed.
        n_val: occupied-band count for the ``wfn_ibz`` path. Defaults to
            ``wfn.nelec``.

    Returns:
        (Nx, Ny, Nz) float64 real-space density, in the same normalization
        convention as the old ``centroid.get_charge_density`` it replaces.
    """
    if source not in {"auto", "qe_save", "wfn_ibz"}:
        raise ValueError(f"source must be 'auto'|'qe_save'|'wfn_ibz', got {source!r}")

    # Decide which path to take.
    use_save = source == "qe_save"
    if source == "auto":
        resolved = Path(save_dir) if save_dir is not None else _autodetect_save_dir()
        if resolved is not None and (resolved / "charge-density.hdf5").is_file():
            save_dir = resolved
            use_save = True
            print(f"[charge_density] source='auto' → reading {save_dir}/charge-density.hdf5")
        else:
            print("[charge_density] source='auto' → no QE .save found, "
                  "falling back to IBZ wavefunction sum from WFN.h5")

    if use_save:
        if save_dir is None:
            raise ValueError("source='qe_save' requires save_dir=...")
        return rho_from_qe_save(save_dir)

    # wfn_ibz path
    if wfn is None or sym is None:
        raise ValueError("source='wfn_ibz' requires both wfn=... and sym=...")
    return rho_from_wfn_ibz(wfn, sym, n_val=n_val)
