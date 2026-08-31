"""Physical real-space charge fields and scalar-grid symmetrization.

Centroid feature weights live in :mod:`centroid.sampling_metric`; physical
ground-state density readers remain here for Hartree and diagnostics.

Two ρ(r) sources are supported:

1. ``rho_from_qe_save(save_dir)`` — read the already-symmetrized valence
   density from QE's ``<prefix>.save/charge-density.hdf5``. This is what QE
   wrote at the end of SCF and is the physically correct, point-group-
   symmetric ρ_val(r). Fastest of the two paths — no recomputation, just
   a single FFT of ρ(G) → ρ(r). Requires access to the QE ``.save`` dir.

2. ``rho_from_wfn_ibz(wfn, sym, n_val=None)`` — compute
   ρ(r) = Σ_k w_k Σ_n f_nk |ψ_nk(r)|² directly from the IBZ wavefunctions in
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
import warnings
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
import symmetry_maps                                            # noqa: E402


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

def _load_wfn_k_fftbox_ibz(wfn: WfnLoader, n_val: int) -> jnp.ndarray:
    """Load IBZ wavefunctions into the FFT box, shape (nk_irr, n_val, nspinor, Nx, Ny, Nz).

    Raw IBZ coefficients scattered onto the QE FFT grid — no fractional-
    translation phase is applied because we only want |ψ|² downstream and
    phases drop out of the modulus.

    Uses :class:`wfn_loader.WfnLoader` (eager backend) plus
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
    from wfn_loader import WfnLoader
    from common.wfn_transforms import to_box

    # single_device_mesh IS the process-local mesh: wfn_transforms.
    # process_local_mesh is an alias for this exact object (one per
    # process, owned by common.collectives).  Import it from its owner.
    from common.collectives import single_device_mesh

    with WfnLoader(wfn.path) as loader:
        psi = loader.load_process_local(bands=(0, n_val), k="ibz")
        return to_box(psi, loader.box_index(k="ibz"), loader.fft_grid,
                      mesh=single_device_mesh())


def rho_from_wfn_ibz(
    wfn: WfnLoader,
    sym: symmetry_maps.SymMaps,
    n_val: int | None = None,
    *,
    warn: bool = True,
) -> np.ndarray:
    """Sum ρ(r) = Σ_k w_k Σ_n f_nk |ψ_nk(r)|² over IBZ k-points.

    Delegates the actual arithmetic to ``psp.get_DFT_mtxels.compute_valence_density``,
    which handles both the "wfn grid == ρ grid" case (one iFFT per band)
    and the "ecutrho > 4·ecutwfc" case (scatter into a larger ρ grid). It
    picks up the IBZ k-weights automatically when the leading dim of
    ``wfn_k`` equals ``len(wfn.kweights)``.

    Args:
        wfn: open ``WFNReader``.
        sym: matching ``SymMaps`` (used only for the wfn→ρ re-embedding
            path when ``ecutrho > 4·ecutwfc``).
        n_val: number of bands to include. Defaults to ``wfn.nelec`` for an
            exact integer occupation table.  Fractional/smeared WFN density
            is intentionally not built by this legacy resident-IBZ carrier;
            use the canonical QE charge-density source instead.

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

    if not wfn.occupations_are_exact_integer:
        raise ValueError(
            "rho_from_wfn_ibz: fractional/smeared physical density requires "
            "every exactly nonzero WFN band (including signed tails beyond "
            "ifmax), "
            "which this legacy resident-IBZ FFT-box path may not materialize. "
            "Select the canonical QE density with "
            "--charge-density-source qe_save (rho_from_qe_save).")
    if n_val is None:
        n_val = int(wfn.physical_density_band_stop)

    wfn_k = _load_wfn_k_fftbox_ibz(wfn, n_val)
    with warnings.catch_warnings():
        if not warn:
            warnings.simplefilter("ignore")
        rho_jax = compute_valence_density(
            wfn_k, sym, wfn, k_source="file")
    return np.asarray(rho_jax, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Scalar-grid symmetry projection used by centroid feature weights
# ═══════════════════════════════════════════════════════════════════════

def symmetrize_on_grid(
    field: np.ndarray,
    sym_ops: np.ndarray,
    translations_frac: np.ndarray | None = None,
) -> np.ndarray:
    """Average ``field`` over spatial Seitz operations on the FFT grid.

    With no translations, ``sym_ops`` are direct integer r-actions (the
    historical symmorphic API).  With ``translations_frac``, ``sym_ops`` are
    BGW reciprocal-space ``mtrx`` rows and the service builds the exact
    nonsymmorphic real-space pullback.  The latter is the centroid driver's
    atom-derived space-group path.
    """
    f = np.asarray(field, dtype=np.float64)
    N = np.asarray(f.shape, dtype=np.int64)
    ops = np.asarray(sym_ops, dtype=np.int64).reshape(-1, 3, 3)
    if ops.shape[0] <= 1 and translations_frac is None:
        return f
    flat = f.ravel()
    acc = np.zeros_like(flat)
    if translations_frac is None:
        for M in ops:
            acc += flat[symmetry_maps.grid_point_image_perm(N, M)]
    else:
        tau = np.asarray(translations_frac, dtype=np.float64).reshape(-1, 3)
        if tau.shape[0] != ops.shape[0]:
            raise ValueError(
                "translations_frac must have one row per symmetry; "
                f"got {tau.shape[0]} for {ops.shape[0]} operations")
        pullback = symmetry_maps.fft_grid_pullback_perm(
            ops, tau * (2.0 * np.pi), N, validate=True)
        for row in pullback:
            acc += flat[row]
    return (acc / ops.shape[0]).reshape(f.shape)




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
    wfn: WfnLoader | None = None,
    sym: symmetry_maps.SymMaps | None = None,
    *,
    source: str = "auto",
    save_dir: str | os.PathLike | None = None,
    n_val: int | None = None,
    print_fn=print,
    warn: bool = True,
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
            print_fn(f"[charge_density] source='auto' → reading {save_dir}/charge-density.hdf5")
        else:
            print_fn("[charge_density] source='auto' → no QE .save found, "
                     "falling back to IBZ wavefunction sum from WFN.h5")

    if use_save:
        if save_dir is None:
            raise ValueError("source='qe_save' requires save_dir=...")
        return rho_from_qe_save(save_dir)

    # wfn_ibz path
    if wfn is None or sym is None:
        raise ValueError("source='wfn_ibz' requires both wfn=... and sym=...")
    return rho_from_wfn_ibz(wfn, sym, n_val=n_val, warn=warn)
