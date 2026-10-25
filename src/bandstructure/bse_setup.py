"""BSE interpolation setup — fine-k wfn recovery via standard htransform.

Top-level entry point for the BSE-interpolation handoff. Takes the
coarse-grid htransform outputs (``ctilde``, ``B_at_mu``, ``enk_sigma``)
plus a finer uniform k-grid spec, and produces wfns at the
already-selected coarse-grid centroids on the fine grid via the
standard htransform pipeline:

    fH_R, L_R  = build_fH_R(ctilde, enk_sigma, kgrid_co,       # one program:
                            backmap=Φ^T)                      # node-pair kernel and
    L_R        = ifft_k Φ^T_k (−f(E_k)) C̃_k†                  # back-map kernel
    X_q        = Σ_R Σ_T w_ij e^{-2πi q·(R+LT)} X_R           # per-pair WS sum
    (lam, c)   = top_block_eigh(fH_q, C̃ of nearest coarse k)  # fitted f(ε), eigvecs in C̃'s frame
    ψ_n,q(r_μ) = [L_q c_n,q]_μ / (−lam_n)                    # at the centroids (exact on-grid)

``Φ^T`` are the fitted states at the ISDF centroids (the back-map targets:
the nodes themselves when the nodes are the centroids).

The resulting wfns are returned as the canonical X/Y-sharded bundle
used elsewhere in LORRAX (matching ``load_centroids_band_chunked``),
ready to drop in as a BSE interpolation input.

This module deliberately stays narrow: it calls the core htransform
routines (``build_fH_R``, ``node_pair_table``, ``newton_inv``) and
adds only the fine-grid k-list construction, the per-q top-block path, and
the X/Y reshard.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from distrib_la import mesh_key as _mesh_key   # noqa: E402
from common.log import get_logger, message
from common.collectives import device_put_process_local
# Physical target and q counts need not divide the process grid. The
# projection and bundle epilogue below keep their legal carriers until the
# explicit output layout is established; fit_sharding is not their padding.
from common.sharding_fit import fit_sharding as _fit
from runtime.padding import padded_axis
# The (i,j)-face → q-batch move below is a layout change XLA cannot lower in
# one step; ``common.staged_reshard`` is the two-all_to_all staging that makes
# it legal.  See the block at ``fH_q(qbatch)``.
from common.staged_reshard import (
    face_to_batch_reshard as _face_to_batch_reshard,
    face_to_batch_reshard_supported as _face_to_batch_supported,
    batch_to_face_reshard,
)

from .fh_interp import (
    NODE_ROUTES,
    admit_node_route,
    build_fH_R,
    coarse_start_block,
    log_top_block,
    newton_inv,
    node_pair_table,
    node_route_map,
    require_newton_converged,
    start_columns,
    top_block_eigh,
)
from common.ws_images import pair_ws_code_fourier

# ``logger``, not ``log``: ``compute_wfns_fi`` names its ``log_fn`` seam ``log``.
logger = get_logger(__name__)


#: Per-process cache of this module's small epilogue jits (``newton_inv`` and
#: the bundle reshard), keyed by EVERY closed-over value: the mesh (by
#: ``distrib_la.mesh_key``, which carries device identity, not ``id``), the
#: extents and ``newton_inv``'s (a, n, shift).  The chunk kernel is not
#: here: ``fh_interp.admit_node_route`` compiles it per call at the call's
#: operands, as the admission prices it (the persistent compile cache serves a
#: repeat).
_WFNS_FI_KERNELS: dict = {}


def _cached_kernel(key, build):
    """The jit for ``key``, built once per process.

    Deliberately a plain dict and not ``functools.lru_cache``: the key is
    assembled by the caller from things an argument-keyed decorator cannot
    see (closed-over constants), and an eviction policy would silently
    reintroduce the retrace this exists to remove.  The tree's own prior art
    is ``htransform._accum_G_cache`` / ``vq_interp._REFIT_KERNELS`` /
    ``distrib_la.plan._SCAN_CACHE``; this is the same shape.
    """
    fn = _WFNS_FI_KERNELS.get(key)
    if fn is None:
        fn = _WFNS_FI_KERNELS[key] = build()
    return fn


def _parse_kgrid_fine(spec) -> tuple[int, int, int]:
    """Accept '8 8 1', '8,8,1', or a tuple/list."""
    if isinstance(spec, (tuple, list)):
        parts = list(spec)
    else:
        parts = str(spec).replace(',', ' ').split()
    if len(parts) != 3:
        raise ValueError(f"kgrid_fine must have 3 entries, got {spec!r}")
    return tuple(int(x) for x in parts)


def parse_kgrid_shift(spec) -> tuple[float, float, float]:
    """``kgrid_fine_shift``: kgrid.x's dk, ``"dx dy dz"`` in fractions of one
    fine-grid step, each reduced to [0, 1).  Empty or None -> (0, 0, 0)."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return (0.0, 0.0, 0.0)
    parts = list(spec) if isinstance(spec, (tuple, list)) else \
        str(spec).replace(',', ' ').split()
    if len(parts) != 3:
        raise ValueError(f"kgrid_fine_shift takes three numbers dx dy dz (kgrid.x's dk, "
                         f"in fractions of one fine-grid step), got {spec!r}")
    vals = tuple(float(str(x).replace('d', 'e').replace('D', 'e')) for x in parts)
    if not all(np.isfinite(vals)):
        raise ValueError(f"kgrid_fine_shift must be finite, got {spec!r}")
    return tuple(float(v % 1.0) for v in vals)


def uniform_kgrid_frac(kgrid: tuple[int, int, int],
                       shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """The uniform k grid k_i = (i + dk)/N in crystal coords, C order (ix
    slowest), wrapped to (-0.5, 0.5]; ``shift`` is kgrid.x's dk (fractions of
    one step).  Host float64."""
    nx, ny, nz = kgrid
    dx, dy, dz = (float(s) for s in shift)
    ix = (np.arange(nx, dtype=np.float64) + dx) / nx
    iy = (np.arange(ny, dtype=np.float64) + dy) / ny
    iz = (np.arange(nz, dtype=np.float64) + dz) / nz
    grid = np.stack(np.meshgrid(ix, iy, iz, indexing='ij'), axis=-1).reshape(-1, 3)
    return (grid + 0.5) % 1.0 - 0.5


def _uniform_kgrid_frac(kgrid: tuple[int, int, int], shift=(0.0, 0.0, 0.0)) -> jax.Array:
    """:func:`uniform_kgrid_frac` on the device (Γ-centred when ``shift`` is 0)."""
    return jnp.asarray(uniform_kgrid_frac(kgrid, shift))


#: The f-shoulder tripwire's floor on ``min_k |f(ε_b)| / max|f|``, over the
#: bands ``compute_wfns_fi`` RETURNS.  Default 0.0 — refuse only when a
#: returned band's f is EXACTLY zero somewhere in the zone, which is the
#: measured failure and not a tolerance judgement.
FI_FSHOULDER_TOL_DEFAULT = 0.0


def _f_shoulder_gate(f_eps, b_min: int, b_max: int, shift: float, log_fn,
                     rank: int | None = None,
                     where: str = "compute_wfns_fi"):
    """REFUSE when a RETURNED band is invisible to ``fH`` at some k.

    THE MECHANISM, in one sentence: ``f`` is identically zero for ε ≥
    ``shift`` and ``htransform._f_params_from_energies`` sets ``shift =
    max_k ε[nb−1]``, so the top band of the htransform's own window
    contributes EXACTLY NOTHING to ``fH = Σ_n f(ε_n) c_n c_nᴴ`` at the k that
    attains that maximum, and ``f`` vanishes to second order approaching it
    — which puts a whole SHOULDER of bands below the top at a per-cent or
    less of fH's weight.  Those eigenvector slots are degenerate with fH's
    own ``(rank − nb)``-dimensional null space and ``eigh`` fills them with
    arbitrary directions out of it.  The returned ψ for such a band is not
    slightly wrong; it is a different vector.

    WHY A GATE AND NOT A DOC LINE.  This cost five lanes.  The representation
    was fine every time — ``ctilde`` orthonormality 8.9e-15 and the Galerkin
    residual 4.3e-15 on the parent where the tile null read 140 — so every
    instrument the tree already had exonerated the htransform, correctly, and
    the defect was in the REQUEST.  Measured signature
    (``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md``
    §2), from the runs' own ``enk_sigma``:

        dp2628n20  nb 20, ZERO guards:  bands 16..19 read min_k |f|/max|f| =
                   0.000000, exactly zero at 3 k each; α-space overlap
                   ‖O[m,:]‖ collapses to 0.23…0.27; tile null 1.267.
        p2628n52   nb 52, ZERO guards:  bands 50, 51 zero at 8 k each.
        The bands that SURVIVE never get closer to zero than ~4e-4 of
        max|f| and come back at ‖O[m,:]‖ = 1.000000 at every k.

    So the good and the bad are separated by four decades with nothing in
    between, and a floor at exactly zero is the whole discrimination.

    THE FIX IS GUARD BANDS, not a wider tolerance: widen the htransform
    window ABOVE what you ask back (``initialize_wfns(n_guard_bands=…)``,
    ``bse.vq_interp.refit_prepare``), so the shoulder lands on bands nobody
    reads.  ``b_min``/``b_max`` index ``ctilde``'s band axis, which is the
    same axis ``f_eps`` carries and — ``f`` being monotone in ε — the same
    ascending-eigenvalue slice the caller takes out of ``fH_q``.
    """
    tol = FI_FSHOULDER_TOL_DEFAULT
    fa = np.abs(np.asarray(f_eps, dtype=np.float64))     # (nb_co, nk_co)
    fmax = float(fa.max()) if fa.size else 0.0
    if fmax <= 0.0:
        raise ValueError(
            f"{where}: max|f(ε)| over the whole htransform window is "
            "zero — every band is at or above the f-transform's shift, so fH "
            "is identically zero and its eigenvectors are pure noise.  This "
            "is an energy-array problem (a zeroed or sentinel enk_sigma), "
            "not a window one.")
    ratio = fa[b_min:b_max, :].min(axis=1) / fmax        # per RETURNED band
    n_zero_k = int((fa[b_min:b_max, :] == 0.0).sum())
    worst_b = int(np.argmin(ratio)) + b_min
    worst = float(ratio.min())
    n_guard = int(fa.shape[0]) - int(b_max)
    if log_fn is not None:
        log_fn(f"  [gate] f-shoulder over the RETURNED bands "
               f"[{b_min}, {b_max}) of {int(fa.shape[0])} "
               f"({n_guard} guard band(s) above): "
               f"min_b min_k |f(ε_b)|/max|f| = {worst:.6e} at band {worst_b} "
               f"(shift = {shift:.8f} Ry; {n_zero_k} exactly-zero (band, k) "
               f"slot(s) in the returned window)")
    if worst <= tol:
        raise ValueError(
            f"{where}: band {worst_b} of the RETURNED window "
            f"[{b_min}, {b_max}) is invisible to fH — "
            f"min_k |f(ε_{worst_b})|/max|f| = {worst:.6e} at or below the "
            f"{tol:.1e} floor, with {n_zero_k} exactly-zero (band, k) slot(s) "
            f"in the window.  f(ε) ≡ 0 for ε ≥ shift = {shift:.8f} Ry "
            f"(= max_k ε of the TOP band of ctilde's own window), so this "
            f"band contributes nothing to fH = Σ_n f(ε_n) c_n c_nᴴ at those "
            f"k; its eigenvector slot is degenerate with fH's "
            + (f"{int(rank) - int(fa.shape[0])}-dimensional "
               if rank is not None else "")
            + f"null space and eigh fills it with an arbitrary direction out "
            f"of that space. "
            f"The returned ψ for that band is a DIFFERENT VECTOR, not an "
            f"approximate one, and every pair density built from it is wrong "
            f"— measured at 1.267 on a 5.0e-02 tile-null bracket.\n"
            f"  THE FIX IS GUARD BANDS, and it is the ONLY fix: widen the "
            f"htransform window above what you ask back, so the shoulder "
            f"lands on bands nobody reads.  There are currently {n_guard}; "
            f"how many you need is the DEPTH OF THE DEAD SET printed above "
            f"({n_zero_k} exactly-zero (band, k) slots), which was four bands "
            f"on a 20-band window and two on a 52-band one.  In a deck: raise "
            f"``number_cond_bands_coarse`` (or ``ncond``) above the window you "
            f"return (``number_cond_bands_fine``, the ζ-fit window).  In code: "
            f"``htransform.initialize_wfns(n_guard_bands=…)``, which "
            f"``bse.vq_interp.refit_prepare`` drives from "
            f"``--refit-guard-bands``.")
    return worst


def resolve_conduction_window(enk_sigma, *, nb_window: int, nval_in: int,
                              n_cond: int, tol_ry: float, mode: str,
                              input_file: str, log_fn=print):
    """``(b_min, b_max, n_guard)``: the BSE conduction bands inside the fH.

    fH is built from every band of ``ctilde`` and only the sub-window
    ``[b_min, b_max)`` is returned, so a full-band ``ctilde`` gives a full-band
    fH however few conduction bands the BSE keeps, and the kept bands sit
    strictly inside it, guarded above.  A sliver window whose top edge cuts a
    near-degenerate (Kramers) pair rings 100–1000 meV off-grid.  The window is
    two-sided: a wider one asks the nodal basis to represent more states (the
    metric eps of the fit and the node count per fitted band), which the
    all-coarse transformed-energy receipt and the consumer's on-grid energy
    gate measure.

    ``nb_window`` is the band count of the htransform fH (``ctilde``);
    ``enk_sigma`` is its ``(nb, nk)`` energies, window-relative like
    ``b_min = nval_in`` and ``b_max = nval_in + n_cond``.  The window must fit (it refuses otherwise), and its edges are
    checked for a cut multiplet (``common.band_degeneracy``, report-only:
    ``b_max`` is pinned to the loader's ``n_cond``, and widening it here would
    desynchronise the conduction caches from the BSE window).  ``n_guard`` is
    the conduction bands above the selection: fewer than 4 lets the top edge
    ring off-grid; more than 16 asks the nodal basis to represent states the
    kernel never uses.  Both warn.
    """
    from common.band_degeneracy import check_band_window
    nb_window = int(nb_window)
    b_min, b_max = int(nval_in), int(nval_in) + int(n_cond)
    n_guard = nb_window - b_max
    if b_max > nb_window:
        raise ValueError(
            f"BSE conduction window [{b_min},{b_max}) exceeds the htransform "
            f"fH window ({nb_window} bands): raise nband in {input_file} to "
            f">= {b_max}, or drop number_cond_bands_fine to <= {nb_window - int(nval_in)}")
    check_band_window(
        np.asarray(enk_sigma).T, b_min, b_max, tol_ry=tol_ry, mode=mode,
        where="exciton_bands htransform conduction window", log=log_fn)
    if n_guard < 4:
        logger.warning(
            f"only {n_guard} conduction guard band(s) above the BSE selection",
            extra=message(
                id="bse_guard_bands_few",
                found=("conduction guard bands", n_guard),
                needs="at least 4: a selection boundary near a Kramers pair "
                      "can ring off-grid",
                fix=f"widen the input's ncond/nband (>= {b_max + 4} bands)"))
    if n_guard > 16:
        logger.warning(
            f"htransform fH spans {nb_window} bands with {n_guard} "
            f"conduction guards above the BSE window",
            extra=message(
                id="bse_guard_bands_many",
                found=("conduction guard bands", n_guard),
                needs="a window no wider than 16 guards: a LARGE interp "
                      "window is not automatically more accurate",
                fix="keep nband just above the BSE window unless both the "
                    "all-coarse transformed-energy receipt and this driver's "
                    "on-grid energy gate remain controlled"))
    log_fn(f"  full-band htransform: fH over {nb_window} bands "
           f"({int(nval_in)}v + {nb_window - int(nval_in)}c); BSE conduction "
           f"[{b_min},{b_max}) = {int(n_cond)} band(s) + {n_guard} guard(s)")
    return b_min, b_max, n_guard


def _project_fine_wfns(lam, U, L, b_min, b_max, nspinor, n_mu, n_mu_out):
    """Project nodal states directly to a legal padded spin/centroid carrier.

    The back-map packs physical targets at ``s*n_mu + mu``. Reindexing that
    flat padded carrier avoids an unshardable intermediate of width n_mu;
    every added centroid is exactly zero, independently of target pad data.
    """
    c = U[..., b_min:b_max]
    nodal = (L @ c) / (-lam[..., None, b_min:b_max])
    if n_mu_out == n_mu:
        # Keep the original projection when its physical width already tiles.
        targets = nodal[..., :nspinor*n_mu, :].reshape(
            nodal.shape[:-2] + (nspinor, n_mu_out, c.shape[-1]))
    else:
        spin = jnp.arange(nspinor)[:, None]
        mu = jnp.arange(n_mu_out)[None, :]
        indices = jnp.where(mu < n_mu, spin*n_mu+mu, 0).reshape(-1)
        targets = jnp.take(nodal, indices, axis=-2)
        targets = targets.reshape(nodal.shape[:-2] + (nspinor, n_mu_out, c.shape[-1]))
        targets = jnp.where((mu < n_mu)[..., None], targets, 0)

    return lam[..., b_min:b_max], jnp.moveaxis(targets, -1, -3), c


def compute_wfns_fi(
    *,
    ctilde: jax.Array,
    B_at_mu,
    enk_sigma: jax.Array,
    kgrid_co: tuple[int, int, int],
    kgrid_fine=None,
    band_window_fi: tuple[int, int],
    mesh_xy: Mesh,
    a_band_index: int | None = None,
    batch_size: int | None = None,
    q_list=None,
    kgrid_shift=None,
    return_coeffs: bool = False,
    n_mu_out: int | None = None,
    use_low_mem_eigh: bool = False,
    htransform_quality_record_fn=None,
    log_fn=None,
):
    """Recover ψ at the ISDF centroids on a finer uniform k-grid — or on an
    EXPLICIT list of arbitrary q (the arbitrary-Q generalization,
    arbitrary_q_bse.md §1/§2a: e.g. the shifted grid {k + Q}).

    Args:
        ctilde:    (nk_co, nb, rank) orthonormal nodal coefficients C̃_k.
                   ``build_fH_R`` consumes paired rank-sharded views, so this
                   may arrive rank-sharded or replicated.
        B_at_mu:   :class:`isdf.galerkin.NodalInterp` — the nodal values,
                   the states at the centroids (the back-map targets) and
                   the geometry; ``shape`` = (rank, ns, n_μ).
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        kgrid_fine:  (nkx_fi, nky_fi, nkz_fi) fine k-grid; tuple/list/string.
                   Mutually exclusive with ``q_list``.
        band_window_fi: (b_min, b_max) — sub-window of the htransform band
                   axis to RETURN on the fine grid (0-based, exclusive end).
                   fH is ALWAYS built from every band in ``ctilde``; this
                   window only selects which eigenpairs are returned, so the
                   returned bands sit interior to fH, guarded above.  The
                   top-block solve converges exactly these returned bands
                   (the guards above are not read; the topmost ones sit at
                   f = 0 beside the null space and need not converge).
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index for the f-transform 'a' parameter
                   (defaults to the top of the htransform window).
        batch_size: q-points per chunk (deck key ``wfn_fi_q_chunk``);
                   ``None`` / ``<= 0`` (the DEFAULT) means N_q_co =
                   prod(kgrid_co): one chunk then costs, on the q-batched
                   route, the bytes of fH_R itself, which the run has already
                   afforded.  Rounded UP to a multiple of the device count so
                   the q axis splits.
        q_list:    optional (nq, 3) fractional q — evaluate at exactly these
                   points instead of a uniform grid (wrapped to (−0.5, 0.5]).
        kgrid_shift: with ``kgrid_fine``, kgrid.x's dk ``(dx, dy, dz)`` in
                   fractions of one fine-grid step: k_i = (i + dk)/N.
        n_mu_out: optional output centroid carrier, at least the physical
                   target count and divisible by both mesh axes. Defaults to
                   their least common multiple above that count. Padding is
                   exact zero inside the priced projection program.
        return_coeffs: also return the eigenvector coefficients
                   ``.coeffs_fi`` (nk_fi, rank, nb_fi), rank-sharded over
                   all mesh ranks (y,x) (the canonical Galerkin rank carrier).
        use_low_mem_eigh: the deck's ``linalg = distributed``: no rank may
                   hold a whole (rank, rank) matrix, so only the face route
                   is admitted.  Otherwise ``lxkit.memory`` admits the
                   q-batched route (whole matrices of a rank's own q rows,
                   one batched solve) where its compiled program fits the
                   room, else the face route (one q at a time, every matrix
                   tiled over the mesh) — the mesh backup of route
                   completeness (GATE nmu2_no_backup).
        log_fn:    optional logger.

    Returns:
        SimpleNamespace bundle (matching ``file_io.tagged_arrays`` convention):
            .n_mu_logical: physical target count (B_at_mu geometry unchanged)
            .n_mu_carrier: padded centroid extent of both wavefunction views
            .psi_rmu_Y:  (nk_fi, nb_fi, ns, mu_out)  P(None, None, None, 'y')
            .psi_rmuT_X: (nk_fi, mu_out, nb_fi, ns)  P(None, 'x', None, None)
            .enk_full:   (nk_fi, nb_fi)           recovered DFT energies (Ry)
                                                  via ``newton_inv`` of fH_q eigvals
            .lam_fi:     (nk_fi, nb_fi)           raw fH_q eigenvalues (= f(ε_n,q))
            .coeffs_fi:  (nk_fi, rank, nb_fi)     only when ``return_coeffs``
            .q_frac:     (nk_fi, 3) host float64  fractional q of each row,
                                                  wrapped to (−0.5, 0.5]

    Both wfn copies live on-device and are sharding-distinct so any
    contraction over (n_μ) along either mesh axis stays local.
    """
    log = log_fn if log_fn is not None else (lambda *a, **kw: None)
    if (kgrid_fine is None) == (q_list is None):
        raise ValueError("pass exactly one of kgrid_fine / q_list")
    nb_co = int(ctilde.shape[1])
    rank = int(ctilde.shape[2])
    # Galerkin fit/read always supply this rank carrier. Keeping the returned
    # coefficient bank rank-sharded avoids a ragged-q replication fallback.
    if return_coeffs and rank % int(mesh_xy.size):
        raise ValueError('fine-wavefunction coefficients require the canonical Galerkin '
                         'rank carrier padded over all mesh ranks')
    nspinor = int(B_at_mu.shape[1])
    n_mu = int(B_at_mu.shape[2])
    # Both canonical views shard mu, on different axes. The source geometry
    # remains physical; only its returned wavefunction carrier is padded.
    mu_divisor = int(np.lcm(int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])))
    mu_out = (-(-n_mu//mu_divisor)*mu_divisor if n_mu_out is None else int(n_mu_out))
    if n_mu < 1 or mu_out < n_mu or mu_out % mu_divisor:
        raise ValueError("fine wavefunction centroid carrier must contain the targets and tile both mesh axes")

    b_min, b_max = int(band_window_fi[0]), int(band_window_fi[1])
    nb_fi = b_max - b_min
    if nb_fi <= 0 or b_min < 0 or b_max > nb_co:
        raise ValueError(
            f"band_window_fi {band_window_fi} not a valid sub-range of the "
            f"{nb_co} fitted bands")

    nk_co = int(kgrid_co[0]) * int(kgrid_co[1]) * int(kgrid_co[2])
    if int(ctilde.shape[0]) != nk_co:
        raise ValueError(
            f"kgrid_co {tuple(kgrid_co)} implies N_q_co={nk_co} but ctilde has "
            f"{int(ctilde.shape[0])} k-points; the q-chunk default and "
            f"build_fH_R's ifft-reshape both read this extent.")
    if batch_size is None or int(batch_size) <= 0:
        batch_size, _bs_origin = nk_co, f"default N_q_co={nk_co}"
    else:
        batch_size = int(batch_size)
        _bs_origin = f"wfn_fi_q_chunk={batch_size}"
    # The q-batch width is a PLACEMENT extent: a width that does not divide
    # the mesh drops the q sharding and replicates the batch (measured 8x
    # slower at P=64), so it is rounded UP to a multiple of ndev.  Padded q
    # rows are zeros, solved and discarded by the [:nq] slice.
    bs = padded_axis(
        batch_size, mesh_xy, name="BSE q batch",
        spec=P(("x", "y"), None, None), axis=0).carrier
    from runtime.padding import mesh_divisor
    _ndev = mesh_divisor(mesh_xy)
    _batch_note = (f"q-chunk={bs} [{_bs_origin}]" if bs == batch_size else
                   f"q-chunk={bs} ({_bs_origin}, padded to a multiple of "
                   f"ndev={_ndev} so the q axis splits)")

    if q_list is None:
        kgrid_fine = _parse_kgrid_fine(kgrid_fine)
        nq = kgrid_fine[0] * kgrid_fine[1] * kgrid_fine[2]
        kgrid_shift = parse_kgrid_shift(kgrid_shift)
        log(f"  bse_setup: {kgrid_co} → {kgrid_fine} ({nq} q-pts"
            + (f", shift {kgrid_shift}" if any(kgrid_shift) else "") + "), "
            f"bands [{b_min}, {b_max}) of {nb_co} fitted, rank {rank}, "
            f"{_batch_note}")
    else:
        q_list = np.asarray(q_list, dtype=np.float64).reshape(-1, 3)
        nq = q_list.shape[0]
        log(f"  bse_setup: {kgrid_co} → explicit q-list ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {nb_co} fitted, rank {rank}, "
            f"{_batch_note}")

    # ── fH_R and the back-map L_R: one admitted program ───────────────────
    # Both stay SHARDED P(None, 'x', 'y'): the q-Fourier sum contracts ONLY
    # over R, so every device builds its own tile of fH_q / L_q with no
    # communication.  (A replicated fH_R was nk_co·rank²·16 B on EVERY device
    # and a host gather of the same size per process: the reason a converged
    # k-grid could not run the htransform, gw_converged_12x12_80ry_2026-07-21
    # §5.)
    fH_R, (a_f, n_f, shift), _f_eps, _active_R, L_R = build_fH_R(
        ctilde, enk_sigma, kgrid_co, mesh_xy,
        a_band_index=a_band_index, log_fn=log,
        quality_record_fn=htransform_quality_record_fn,
        backmap=B_at_mu.target_values)
    _f_shoulder_gate(_f_eps, b_min, b_max, shift, log, rank=rank)
    ws_table = node_pair_table(B_at_mu, kgrid_co, mesh_xy)
    ws_code, ws_target = ws_table["code"], ws_table["target_code"]
    rep = NamedSharding(mesh_xy, P())
    columns = device_put_process_local(start_columns(rank, nb_co), rep)

    # ── q-points (uniform fine grid or explicit list), padded to batch ───
    if q_list is None:
        q_all = _uniform_kgrid_frac(kgrid_fine, kgrid_shift)
    else:
        q_all = jnp.asarray((q_list + 0.5) % 1.0 - 0.5)
    n_pad = padded_axis(nq, bs, name="BSE setup q batch").pad
    q_pad = (jnp.concatenate([q_all, jnp.zeros((n_pad, 3), dtype=q_all.dtype)])
             if n_pad else q_all)
    q_pad = jax.device_put(q_pad, rep)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    # ── The (i,j)-face → q-batch reshard of the q-batched route, STAGED ──
    # Built HERE, outside every jit: the factory issues a collective
    # (``warm_mesh_cliques``) and so runs synchronously on every rank.  A
    # single ``with_sharding_constraint`` {1,px,py} -> {ndev,1,1} lowers as
    # replicate-then-repartition, the WHOLE (bs, rank, rank) batch on every
    # device ("Involuntary full rematerialization", job 7882974); the staged
    # pair of single-axis all_to_alls keeps every intermediate one shard
    # (job 7883154: P=64 231 -> 146 s).  ``bs`` is padded to ndev above.
    _stage_q = {}
    for extent in {rank, int(L_R.shape[1])}:
        _stage_q[extent] = (_face_to_batch_reshard(mesh_xy, axes=('x', 'y'))
                            if _face_to_batch_supported(
                                mesh_xy, (bs, extent, rank), axes=('x', 'y'))
                            else None)

    face_pin = lambda x: jax.lax.with_sharding_constraint(
        x, _fit(mesh_xy, P(None, 'x', 'y'), x.shape, "bse_setup.X_q(face)"))

    def _fourier(q, X_R, code, route, hermitian=True):
        """X_q = Σ_R Σ_T w_ij e^{-2πi q·(R+LT)} X_R (per-pair WS images) for
        a q batch: face tiles, local over R; on the q-batched route then the
        staged face→q exchange (each rank: whole matrices of its q rows),
        after which the Hermitian transpose is local; on the face route the
        batch (one q) stays tiled and the transpose is a face collective."""
        X_q = face_pin(pair_ws_code_fourier(X_R, ws_table, q, code=code,
                                            pin=face_pin))
        if route == "q-batched":
            stage = _stage_q[int(X_q.shape[1])]
            X_q = (stage(X_q) if stage is not None else
                   jax.lax.with_sharding_constraint(X_q, _fit(
                       mesh_xy, P(('x', 'y'), None, None), X_q.shape,
                       "bse_setup.X_q(qbatch)")))
        if not hermitian:
            return X_q
        out = 0.5 * (X_q + jnp.swapaxes(X_q, -1, -2).conj())
        return out if route == "q-batched" else face_pin(out)

    def _project(lam, U, L, b_min, b_max):
        """Band-window slice + ψ_n,q(r_μ) = [L_q c_n,q]_μ / (−lam_n).

        ``L`` is the back-map ``L_q`` (t_carrier, rank): exact on-grid, and
        the division only touches returned window bands, which the
        f-shoulder gate keeps away from ``f = 0``.
        """
        return _project_fine_wfns(lam, U, L, b_min, b_max, nspinor, n_mu, mu_out)

    out_shardings = (
        _fit(mesh_xy, P(('x', 'y'), None), (bs, nb_fi), "bse_setup.lam"),
        _fit(mesh_xy, P(('x', 'y'), None, None, None),
             (bs, nb_fi, nspinor, mu_out), "bse_setup.psi"),
        _fit(mesh_xy, P(('x', 'y'), None, None), (bs, rank, nb_fi),
             "bse_setup.coeffs"),
        rep, rep)
    q_batch_x0 = _fit(mesh_xy, P(('x', 'y'), None, None),
                      (bs, rank, int(columns.shape[1]) + nb_co),
                      "bse_setup.start_block")

    def _program(route):
        """One chunk on ``route``: Fourier sum → top-block solve over every
        fitted band → back-map projection, ONE fused jit (the full
        (bs, rank, rank) eigenvector batch is never an output; it killed a
        16 x A100-80GB run when it was, 10.1 GiB/device at rank 4452)."""
        @partial(jax.jit, out_shardings=out_shardings)
        def _chunk(q_batch, fH_R, L_R, code, target_code, ctilde, columns):
            def one(q):
                x0 = coarse_start_block(ctilde, kgrid_co, q, columns)
                if route == "q-batched":
                    x0 = jax.lax.with_sharding_constraint(x0, q_batch_x0)
                lam, U, steps, res = top_block_eigh(
                    _fourier(q, fH_R, code, route), x0, (b_min, b_max))
                L = _fourier(q, L_R, target_code, route, hermitian=False)
                return _project(lam, U, L, b_min, b_max) + (steps, res)
            return node_route_map(route, one, q_batch)
        return _chunk

    route, _chunk = admit_node_route(
        f"htransform fine-q fH(q) top block ({rank}^2, q-chunk {bs})",
        _program, (q_pad[:bs], fH_R, L_R, ws_code, ws_target, ctilde,
                   columns), rank=rank, mesh_xy=mesh_xy,
        routes=("face",) if use_low_mem_eigh else NODE_ROUTES)
    log(f"  fH_q route {route}: {nq} q in {q_pad.shape[0] // bs} chunk(s) of "
        f"{bs}" + (f" ({bs // _ndev} whole ({rank}, {rank}) matrices per rank)"
                   if route == "q-batched" else
                   f", one q at a time, ({rank}, {rank}) tiled over {px}x{py}")
        + ("  [linalg = distributed]" if use_low_mem_eigh else ""))

    lam_chunks, psi_chunks, c_chunks, block_stats = [], [], [], []
    t_loop, logged = time.time(), 0
    for i in range(0, q_pad.shape[0], bs):
        lam_s, psi_s, c_s, steps, res = _chunk(
            q_pad[i:i + bs], fH_R, L_R, ws_code, ws_target, ctilde, columns)
        block_stats.append((steps, res))
        lam_chunks.append(lam_s)
        psi_chunks.append(psi_s)
        if return_coeffs:
            c_chunks.append(c_s)
        # One sync per chunk: each chunk's executable materialises a
        # (bs, rank, rank) fH_q (q-batched) inside it, and N chunks in flight
        # would hold N such transients (two OOMs in this module's history).
        jax.block_until_ready(psi_s)
        done = min(i + bs, nq)
        if done - logged >= max(32, bs) or done >= nq:
            logged = done
            dt = time.time() - t_loop
            log(f"    fH_q top-block {done}/{nq} q  {dt:.0f}s  "
                f"{1e3 * dt / max(done, 1):.0f} ms/q")
    log_top_block(log, f"fine-grid fH_q, returned bands [{b_min}, {b_max})",
                  nb_co,
                  [np.asarray(s) for s, _ in block_stats],
                  [np.asarray(r) for _, r in block_stats])
    del fH_R, L_R
    # Retain the legal q carrier until the explicit-output epilogue. An
    # eager ragged [:nq] can drop q sharding and replicate the full payload.
    lam_fi = jnp.concatenate(lam_chunks, axis=0)
    psi_fi = jnp.concatenate(psi_chunks, axis=0)
    coeffs_fi = (jnp.concatenate(c_chunks, axis=0)
                 if return_coeffs else None)
    jax.block_until_ready((lam_fi, psi_fi, coeffs_fi))
    del lam_chunks, psi_chunks, c_chunks, lam_s, psi_s, c_s
    _mesh_id = _mesh_key(mesh_xy)

    # ── Newton-invert lam_fi → DFT-equivalent energies ───────────────────
    # (a_f, n_f, shift) are TRACE CONSTANTS here, not arguments, so they are
    # in the key: two decks that differ only in their f-transform parameters
    # must not share this executable.
    def _build_inv():
        @jax.jit
        def _inv(lam):
            return newton_inv(a_f, n_f, shift, lam.real)
        return _inv

    energies_fi, inverse_residual = _cached_kernel(
        ("inv", _mesh_id, float(a_f), float(n_f), float(shift)),
        _build_inv)(lam_fi)
    require_newton_converged(
        float(inverse_residual), where="compute_wfns_fi")

    # ── Reshard into the canonical (Y, X) wfn-bundle layout ──────────────
    # Matches ``common.wfn_transforms.load_centroids_band_chunked``:
    #   psi_rmu_Y:  (nk, nb, ns, n_μ)   P(None, None, None, 'y')   — n_μ on Y
    #   psi_rmuT_X: (nk, n_μ, nb, ns)   P(None, 'x', None, None)   — n_μ on X
    # Both views use the legal centroid carrier produced inside _project;
    # neither can drop its mu axis because of a prime physical target count.
    out_Y = _fit(mesh_xy, P(None, None, None, 'y'),
                 (nq, nb_fi, nspinor, mu_out),
                 "bse_setup.psi_rmu_Y")
    out_X = _fit(mesh_xy, P(None, 'x', None, None),
                 (nq, mu_out, nb_fi, nspinor),
                 "bse_setup.psi_rmuT_X")

    coeff_sh = NamedSharding(mesh_xy, P(None, ('y', 'x'), None))

    def _build_bundle():
        # The existing staged mover guarantees one shard per intermediate.
        # A direct q-product -> mu-axis constraint can replicate all of psi.
        to_face = batch_to_face_reshard(mesh_xy)
        band_spin = nb_fi*nspinor
        band_carrier = -(-band_spin//mu_divisor)*mu_divisor
        coeff_bands = -(-nb_fi//int(mesh_xy.shape['x']))*int(mesh_xy.shape['x'])

        @partial(jax.jit, out_shardings=(out_Y, out_X, rep, rep,
                                        coeff_sh if return_coeffs else None))
        def _make_bundle(psi, energies, lam, coeffs):
            packed = psi.reshape(psi.shape[0], band_spin, mu_out)
            packed = jnp.pad(packed, ((0, 0), (0, band_carrier-band_spin), (0, 0)))
            # Face first, then gather only bands; mu remains split throughout.
            py = jax.lax.with_sharding_constraint(to_face(packed),
                NamedSharding(mesh_xy, P(None, None, 'y')))
            px = jax.lax.with_sharding_constraint(
                to_face(jnp.swapaxes(packed, -1, -2)),
                NamedSharding(mesh_xy, P(None, 'x', None)))
            psi_Y = py[:nq, :band_spin].reshape(nq, nb_fi, nspinor, mu_out)
            psi_X = px[:nq, :, :band_spin].reshape(nq, mu_out, nb_fi, nspinor)
            if return_coeffs:
                cp = jnp.pad(jnp.swapaxes(coeffs, -1, -2),
                    ((0, 0), (0, coeff_bands-nb_fi), (0, 0)))
                cf = jnp.swapaxes(to_face(cp), -1, -2)
                # Move only x from bands to rank; y remains on rank. The
                # product order is therefore (y,x), with no full C replica.
                c = jax.lax.with_sharding_constraint(cf, coeff_sh)[:nq, :, :nb_fi]
            else:
                c = None
            return psi_Y, psi_X, energies[:nq], lam[:nq], c
        return _make_bundle

    # Every closed-over carrier/output choice belongs to this geometry key.
    program = _cached_kernel(
        ("make_bundle", _mesh_id, int(nq), int(nb_fi), int(nspinor),
         int(mu_out), int(rank), bool(return_coeffs)), _build_bundle)
    arguments = (psi_fi, energies_fi, lam_fi, coeffs_fi)
    executable = program.lower(*arguments).compile()
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    from lxkit import memory
    needed = -agreed_chunk(-compiled_new_bytes(executable))
    memory.admit('fine wavefunction bundle', (('staged q/centroid redistribution',
        memory.priced(0, 'compiled fine-wavefunction epilogue',
            spread=needed*int(mesh_xy.size), ranks=int(mesh_xy.size))),),
        memory.room(bucketed=False))
    psi_rmu_Y, psi_rmuT_X, energies_fi, lam_fi, coeffs_fi = executable(*arguments)

    log(f"  bundle: psi_rmu_Y={psi_rmu_Y.shape}, psi_rmuT_X={psi_rmuT_X.shape}")
    out = SimpleNamespace(
        n_mu_logical=n_mu, n_mu_carrier=mu_out,
        psi_rmu_Y=psi_rmu_Y,
        psi_rmuT_X=psi_rmuT_X,
        enk_full=energies_fi,
        lam_fi=lam_fi,
    )
    if return_coeffs:
        out.coeffs_fi = coeffs_fi
    out.q_frac = np.asarray(q_all, dtype=np.float64)
    return out
