"""Charge-density providers for k-means ISDF centroid selection.

Two sources are supported:

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

from file_io import WFNReader
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

    Uses ``wfn.get_cnk_batch`` (raw irreducible coefficients) and scatters
    onto the QE FFT grid via the per-k G-vector list — no fractional-
    translation phase is applied because we only want |ψ|² downstream and
    phases drop out of the modulus.
    """
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    nspinor = int(wfn.nspinor)
    band_indices = np.arange(n_val)

    # Pre-allocate one host buffer; fill in a Python k-loop (nk_irr is small).
    psi = np.zeros((wfn.nkpts, n_val, nspinor, nx, ny, nz), dtype=np.complex128)
    for ik in range(wfn.nkpts):
        gvecs_k = np.asarray(wfn.get_gvec_nk(ik))                      # (ngk, 3)
        cnk = wfn.get_cnk_batch(ik, band_indices)                      # (n_val, nspinor, ngk)
        psi[ik, :, :, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = cnk
    return jnp.asarray(psi)


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
