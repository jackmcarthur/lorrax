import os
import argparse
import numpy as np

# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  MUST run before this module's own
# `import jax`; idempotent, so importing htransform as a LIBRARY from an
# already-started driver (bse.exciton_bands does) returns the same stack.
from runtime import initialize_communicator_stack
RUNTIME = initialize_communicator_stack()

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erf
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta
from common import rank_criterion
from common import timing
from common.units import RYD_TO_EV
from runtime.padding import round_up
from common.wfn_transforms import get_enk_bandrange
from isdf.galerkin import (
    _lowdin_orthonormalize_band_rows,
    fit_galerkin_basis,
    validate_rank_multiplier,
)
# ``eigh_backend`` + ``use_low_mem_eigh`` are ONE axis with ONE resolver;
# this driver reads a raw params dict rather than a LorraxConfig, which is
# exactly the case that function exists for.
from gw.gw_config import (
    distrib_la_batched_route_choices,
    eigh_backend_choices,
    resolve_distrib_la_batched_route,
    resolve_eigh_backend,
)
from common.fft_helpers import make_flat_k_ifftn
# Q's free r axis is zero-padded through ``runtime.padding`` and split over
# the full mesh product.  ``common.staged_reshard`` owns the exact
# product-band → product-r exchange used to put streamed wavefunctions there.
from common.collectives import gather_to_host
from common.sharding_fit import padded_extent as _pad_to
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402


def _build_mesh_xy() -> Mesh:
    """The run's ('x','y') mesh — the one the module-top startup call built.

    History, shortest form: this was seven lines hand-rolling a mesh (one of
    five dialects, 2026-07-30 audit), then ``resolve_mesh``, then
    ``prepare_mesh`` once measurement answered the warm-up question — without
    the clique warm-up this driver's first collective fires from an XLA
    parallel-executor worker thread and jaxlib refuses communicator creation
    (MPI_Is_thread_main false; P=16 job 7884867 failed on all 16 ranks at the
    first ``load_centroids`` collective).  Since 2026-08-01 the mesh and its
    warm-up come from ``initialize_communicator_stack()`` at the top of this
    module, so this helper only hands back ``RUNTIME.mesh`` for the library
    callers (``initialize_wfns`` with ``mesh_xy=None``) that resolve the mesh
    late.  A second ``prepare_mesh()`` here would be a second Mesh object —
    a second set of communicators and a second copy of every shape-keyed jit
    cache.
    """
    return RUNTIME.mesh


# Per-device ceiling on one streamed Galerkin Q r-chunk.
#
# The outer-r / inner-band loop mirrors zeta fitting: all band chunks are
# summed into Q for ONE r chunk before its Gram contribution is formed and Q
# is discarded.  This ceiling bounds that Q shard on each device.  Splitting
# r changes only the Gram reduction order; splitting bands before the outer
# product would drop cross terms and is forbidden.
# Override with LORRAX_GALERKIN_CHUNK_GIB (GiB, float).
#
# Resolved INSIDE the consuming function, not at module scope: the old
# module-level ``float(os.environ.get(...))`` meant a malformed export
# crashed ``import bandstructure.htransform`` itself — a bare
# ``ValueError: could not convert string to float`` from the import
# storm, naming neither the variable nor the fix (the import-time-crash
# class, P1 audit).  ``resolve_galerkin_chunk_bytes`` refuses BY NAME,
# from the call that actually consumes the budget.


def resolve_galerkin_chunk_bytes() -> int:
    """Per-device ``LORRAX_GALERKIN_CHUNK_GIB`` Q-tile budget in bytes.

    Blank/unset → the default; garbage REFUSES naming the variable
    (``gw_config.env_float`` refuse mode); non-positive values refuse too
    — a zero-byte accumulation budget is never what anyone meant.
    """
    from gw.gw_config import env_float
    gib = env_float("LORRAX_GALERKIN_CHUNK_GIB", 6.0, refuse=True)
    if gib <= 0.0:
        raise ValueError(
            f"LORRAX_GALERKIN_CHUNK_GIB={gib!r} must be > 0 (GiB budget "
            f"for one Galerkin Q r-chunk; unset/blank = 6).")
    return int(gib * 1024 ** 3)


def resolve_extra_rank_pad() -> int:
    """``LORRAX_EXTRA_RANK_PAD`` (default 0) — TEST-ONLY pad-invariance knob.

    The one env resolver for the rank axis's extra null directions (see
    the block comment at the use site); blank/unset → 0, negative or
    non-integer REFUSES naming the variable.  Never set in production.
    """
    raw = os.environ.get("LORRAX_EXTRA_RANK_PAD")
    if raw is None or not raw.strip():
        return 0
    try:
        extra = int(raw)
    except ValueError:
        raise ValueError(
            f"LORRAX_EXTRA_RANK_PAD={raw!r} is not an integer.  Accepted: "
            f"a non-negative int, or unset/blank for 0 (no extra pad).") \
            from None
    if extra < 0:
        raise ValueError(
            f"LORRAX_EXTRA_RANK_PAD must be >= 0; got {extra}")
    return extra


def resolve_fh_ortho_tol(log_fn=None) -> float:
    """``LORRAX_FH_ORTHO_TOL`` (default 1e-6) — the build_fH_R gate cap.

    Routed through ``gw_config.env_float`` in refuse mode: blank/unset →
    the default (the old inline ``float(get(...) or 0.0)`` made a BLANK
    export silently DISABLE the orthonormality gate — the failure it
    guards is a wrong number, not a crash, so silent-off is the worst
    possible reading of a typo); garbage REFUSES naming the variable; a
    NON-DEFAULT value is announced, and ``0`` (gate off) is announced as
    exactly that.
    """
    from gw.gw_config import env_float
    tol = env_float("LORRAX_FH_ORTHO_TOL", 1e-6, refuse=True)
    if tol != 1e-6 and log_fn is not None:
        log_fn(f"  [gate] LORRAX_FH_ORTHO_TOL={tol:.3e} overrides the "
               f"default 1e-6"
               + ("  ** THE ORTHONORMALITY GATE IS DISABLED — only ever "
                  "to reproduce a known-bad run **" if tol == 0.0 else ""))
    return tol


def resolve_rank_policy_mode() -> str:
    """``LORRAX_RANK_POLICY`` (default ``refuse``) — the truncation gate's authority.

    The name and the grammar live once, in ``common/rank_criterion``, which is
    L2 and must be a function of its arguments (``tests/test_layering.py``);
    this driver does the lookup and hands the answer over.  Same shape as
    :func:`resolve_fh_ortho_tol` two functions up, and it exists for the same
    reason: an L1 library dial reaches the environment through EXACTLY ONE
    named resolver, so a reader can find every env-dependent decision this
    module makes by grepping for ``resolve``.

    A mis-spelled mode REFUSES naming the variable — a gate disarmed by a
    typo reads clean in the log, which is the worst possible reading of one.

    WHY THE NAME IS SPELLED OUT HERE.  Everywhere else in the tree the lookup
    is ``os.environ.get(rank_criterion.POLICY_MODE_ENV)`` — one source of
    truth for the spelling.  This module is under the EXACT-SET env ratchet in
    ``tests/test_layering.py``, whose AST scan reads the argument literally
    and records a constant as ``<dynamic>``; pinning that would make the
    ratchet stop naming the variable, which is the degenerate-value failure
    ``TASTE.md`` #18 is about.  So the literal is written for the scanner and
    checked against the constant immediately below, which makes a rename a
    loud refusal at exactly one line instead of a silent second spelling.
    """
    if rank_criterion.POLICY_MODE_ENV != "LORRAX_RANK_POLICY":
        raise RuntimeError(
            f"resolve_rank_policy_mode: common/rank_criterion renamed its "
            f"policy dial to {rank_criterion.POLICY_MODE_ENV!r}, but this "
            f"resolver still reads 'LORRAX_RANK_POLICY' — the literal exists "
            f"only so the L1 env ratchet can see the name.  Update both.")
    return rank_criterion.resolve_policy_mode(
        os.environ.get("LORRAX_RANK_POLICY"))


def resolve_galerkin_rank_multiplier(value) -> float:
    """Resolve the whole-state randomized-QRCP search ceiling.

    The published default is ``20*N_band``.  ``0`` remains a compatibility
    spelling of 20 for archived decks, not an alternate exact-span path.
    The QR threshold—not this ceiling—selects the delivered physical rank.
    """
    return validate_rank_multiplier(
        value, name="htransform_rank_multiplier")


def validate_centroid_subset_idx(selection, n_parent: int) -> np.ndarray:
    """Validate the ordered parent-to-child centroid row selection."""
    idx = np.asarray(selection)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise ValueError(
            "htransform centroid subset must be a one-dimensional integer "
            f"row list; got shape={idx.shape}, dtype={idx.dtype}.")
    idx = idx.astype(np.int64, copy=False)
    if idx.size == 0:
        raise ValueError("htransform centroid subset may not be empty.")
    if int(idx.min()) < 0 or int(idx.max()) >= int(n_parent):
        raise ValueError(
            "htransform centroid subset escapes its parent table: "
            f"min/max={int(idx.min())}/{int(idx.max())}, parent rows="
            f"{int(n_parent)}.")
    if np.unique(idx).size != idx.size:
        raise ValueError(
            "htransform centroid subset contains duplicate parent rows; the "
            "downfold basis must be a strict ordered subset.")
    return idx


def streaming_galerkin_solve(wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
                             band_range: tuple[int, int],
                             rtol: float = 1e-8, log_fn=None,
                             band_chunk_size: int = 64,
                             bispinor: bool = False,
                             return_full_proj: bool = False,
                             eigh_backend: str = "auto",
                             rank_multiplier: float = 20.0,
                             qr_eps: float = 1.0e-3,
                             qrcp_seed: int = 0):
    """Htransform policy adapter for :func:`isdf.galerkin.fit_galerkin_basis`."""
    rank_multiplier = resolve_galerkin_rank_multiplier(rank_multiplier)
    from common.gpu_utils import _get_jax_gpu_memory_bytes
    device_pool_limit, _, _ = _get_jax_gpu_memory_bytes()
    basis = fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy, band_range,
        rtol=rtol,
        log_fn=log_fn,
        band_chunk_size=band_chunk_size,
        bispinor=bispinor,
        include_projector=return_full_proj,
        eigh_backend=eigh_backend,
        rank_multiplier=rank_multiplier,
        qr_eps=qr_eps,
        qrcp_seed=qrcp_seed,
        q_tile_budget=resolve_galerkin_chunk_bytes(),
        device_pool_limit=device_pool_limit,
        rank_policy_mode=resolve_rank_policy_mode(),
        extra_rank_pad=resolve_extra_rank_pad(),
    )
    return basis.as_legacy_tuple(
        mesh_xy, include_projector=return_full_proj)

def _f_params_from_energies(enk_nb_nk: jax.Array, top_band_index: int,
                            a_band_index: int | None = None) -> tuple[float, float, float]:
    """Compute f-transform parameters from eigenvalues.

    Parameters
    ----------
    top_band_index : int
        Index of the highest band (sets epsilon0 = max eigenvalue of this band).
    a_band_index : int or None
        Index of the band whose bandwidth sets `a = 4 * bandwidth`.
        If None, defaults to top_band_index (original behavior).
        Set this to the highest band you want to keep accurately.
    """
    if a_band_index is None:
        a_band_index = top_band_index
    # ONE jit for the two host floats this returns.  Eagerly these five
    # lines are six single-primitive XLA modules — ``dynamic_slice`` and
    # ``squeeze`` for each band index (one compile each, shapes match),
    # ``_reduce_max`` x2, ``_reduce_min``, ``subtract``, ``add`` — measured
    # at 6 of this driver's 137 (job 7884866).  The band indices are static,
    # so the whole thing folds into one module per (shape, index pair).
    #
    # Bit-identical: the ops and their order are unchanged, ``max``/``min``
    # are order-independent reductions, and ``(max - min) + 1e-14`` is a
    # subtract feeding an add with no multiply for a fusion to contract into
    # an FMA.  ``* 4.0`` stays on the host exactly where it was.
    stats = np.asarray(jax.device_get(
        _f_params_jit(enk_nb_nk, int(top_band_index), int(a_band_index))))
    shift = float(stats[0])
    gap = 4.0 * float(stats[1])
    n = 3.0
    return gap, n, shift


@partial(jax.jit, static_argnames=('top_band_index', 'a_band_index'))
def _f_params_jit(enk_nb_nk: jax.Array, top_band_index: int,
                  a_band_index: int) -> jax.Array:
    """``[max ε_top, max ε_a - min ε_a + 1e-14]`` — see the caller."""
    E_top_k = enk_nb_nk[top_band_index]
    E_a_k = enk_nb_nk[a_band_index]
    return jnp.stack([jnp.max(E_top_k),
                      jnp.max(E_a_k) - jnp.min(E_a_k) + 1e-14])


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _fun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    f_left = y + 0.5 * a
    arg = n * (0.5 + y / a)
    term1 = a * (jnp.exp(-(n * 0.5) ** 2) - jnp.exp(-((n * (a + 2 * y)) / (2 * a)) ** 2)) / (2 * n * jnp.sqrt(jnp.pi) * erf_half)
    term2 = (a + 2 * y) * (erf_half - erf(arg)) / (4 * erf_half)
    f_mid = term1 + term2
    f = jnp.where(cond_left, f_left, 0.0)
    f = jnp.where(cond_mid, f_mid, f)
    f = jnp.where(y >= 0, 0.0, f)
    return jnp.where(f > 0, 0.0, f)


def fun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Transform function f(x) — matches Fortran fun(). Works in unshifted space."""
    return _fun_jit(float(a), float(n), float(shift), x)


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _dfun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    df = jnp.zeros_like(y)
    df = jnp.where(cond_left, 1.0, df)
    arg = n * (0.5 + y / a)
    df = jnp.where(cond_mid, 0.5 - erf(arg) / (2 * erf_half), df)
    return jnp.where(y >= 0, 0.0, df)


def dfun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Derivative of transform function — matches Fortran dfun()."""
    return _dfun_jit(float(a), float(n), float(shift), x)


def f_transform_eigs(enk_nb_nk: jax.Array,
                     a_band_index: int | None = None) -> tuple[jax.Array, float, float, float]:
    """Apply f-transform to eigenvalues. Returns (f_eps, a, n, shift)."""
    nb, _ = enk_nb_nk.shape
    a, n, shift = _f_params_from_energies(enk_nb_nk, top_band_index=nb - 1,
                                          a_band_index=a_band_index)
    f_eps = fun(a, n, shift, enk_nb_nk)
    return f_eps, a, n, shift


def build_R_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Lattice-R grid (nk, 3) matching the IFFT-shift convention used by
    ``make_flat_k_ifftn`` — symmetric around 0, with R_i ∈ {-n/2, ..., n/2-1}."""
    def _shift(n):
        a = np.arange(n, dtype=np.float64)
        return np.where(a >= (n + 1) // 2, a - n, a)
    return np.stack(np.meshgrid(
        _shift(kgrid[0]), _shift(kgrid[1]), _shift(kgrid[2]), indexing='ij'),
        axis=-1).reshape(-1, 3)


def build_fH_R(ctilde: jax.Array, enk_sigma: jax.Array,
               kgrid_co: tuple[int, int, int], mesh_xy: Mesh,
               *, a_band_index: int | None = None,
               log_fn=None):
    """f-transformed Hamiltonian in real-space lattice representation.

    Math (htransform paper):
        fH_k  = -Σ_n |sqrt(-f(ε_n,k))|² ctilde_n,k ctilde_n,k^H
              = Σ_n f(ε_n,k) ctilde_n,k ctilde_n,k^H
        fH_R  = (1/N_k) Σ_k e^{+2πi k·R} fH_k                          # IFFT
    where f(ε) is the smooth bandwidth-bound transform from
    ``f_transform_eigs`` (≤0 for ε<shift, =0 for ε≥shift). For any q,
    fH_q = Herm[Σ_R e^{-2πi q·R} fH_R] recovers the rank-α-basis Hamiltonian
    whose eigenvalues are f(ε_n,q) and whose eigenvectors are c_n,q,
    enabling both bandstructure interpolation (eigvalsh + newton_inv on
    eigvals) and wfn recovery (eigh, then ψ_n,q(r_μ) = Σ_α c_n,q[α]·B[α,s,μ]).

    The paper/archive use the conjugate sign pair.  Because every ``fH_k`` is
    Hermitian and both implementations Hermitize the reconstructed matrix,
    the two spellings are exactly equivalent; keeping the signs here aligned
    with the actual IFFT prevents a documentation-only convention mismatch.

    Args:
        ctilde:    (nk_co, nb, rank) Galerkin coefficients in the rank-α basis,
                   replicated. Output of ``streaming_galerkin_solve``.
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index whose bandwidth sets ``a``;
                   defaults to top of the htransform window (nb-1).
        log_fn:    optional logger.

    Returns:
        fH_k:    (nk_co, rank, rank), sharded P(None, 'x', 'y').
        fH_R:    (nk_co, rank, rank), sharded P(None, 'x', 'y') (lattice-R index).
        params:  (a_f, n_f, shift) — for ``newton_inv`` on the eigvals of fH_q.
        f_eps:   (nb, nk_co) f-transformed eigenvalues, replicated.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    f_eps, a_f, n_f, shift = f_transform_eigs(enk_sigma, a_band_index=a_band_index)
    log_fn(f"  f-transform: a={a_f:.6f} Ry, n={n_f:.2f}, shift={shift:.6f} Ry"
           + (f" (a from band {a_band_index})" if a_band_index is not None else ""))

    flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    spec_3d = P(None, None, None, 'x', 'y')
    # 'backward' = 1/N normalisation in IFFT — matches Σ_R e^{-2πik·R} fH_R = fH_k.
    local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid_co, spec_3d, norm='backward')

    # ``f_eps.T`` at the call site was its own eager ``transpose`` module;
    # transposing inside costs nothing and removes one compile (exact — a
    # transpose is a permutation of the same f64 values).
    @partial(jax.jit, out_shardings=(flat_xy, flat_xy))
    def _build(ctilde_in, f_eps_in):
        f_eps_T = f_eps_in.T
        f_eps_ki = jnp.where(f_eps_T > 0, 0.0, f_eps_T)
        bw = jnp.sqrt(jnp.clip(-f_eps_ki, 0.0, None))
        weighted = ctilde_in * bw[..., None]
        fH_k = -jnp.einsum('kim,kin->kmn', weighted, jnp.conj(weighted), optimize=True)
        # Constrain the CONTRACTION OUTPUT, not just the hermitized copy.
        # ``ctilde`` arrives replicated, so without this the SPMD partitioner
        # has no reason to split the (nk, rank, rank) product and materialises
        # it — and then its transpose, and then the hermitized sum — at FULL
        # size on every device.  Measured on MoS2 12x12 / n_mu=2412 / nb=20
        # (rank 2880): the module wanted 57.84 GiB per device and OOMed an
        # A100-80GB, against a true sharded footprint of 2 x 1.2 GiB on a 4x4
        # mesh.  Pinning the einsum output makes the whole chain local.
        # Memory-only: same values, same shardings out.
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_k = 0.5 * (fH_k + jnp.swapaxes(fH_k, -1, -2).conj())
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_R = local_ifftn(fH_k)
        return fH_k, fH_R

    # ── THE ON-GRID RECOVERY GATE ─────────────────────────────────────────
    # fH_k = Σ_n f(ε_n,k) c_n,k c_n,kᴴ has eigenvalues EXACTLY {f(ε_n,k)} (plus
    # rank−nb exact zeros) if and only if the rows of ctilde[k] are
    # orthonormal.  When they are not, the eigenvalues are no longer f(ε_n)
    # and ``newton_inv`` returns wrong ENERGIES on the coarse grid — silently,
    # with a Hermitian positive-semidefinite fH, a successful Cholesky
    # upstream and a plot that looks fine.
    #
    # WHY HERE.  This is the one function BOTH consumers pass through — the
    # ``bandstructure.htransform`` CLI and ``bandstructure.bse_setup.
    # compute_wfns_fi`` (hence ``bse.exciton_bands``).  Gating here covers the
    # exciton path without reaching into it.
    #
    # WHY ALL k.  ``streaming_galerkin_solve`` prints ``ctilde[0]`` only, and
    # k=0 is Γ, the most symmetric point in the zone.  Cheap to fix: this is
    # nk·nb²·rank, four orders below the nk·rank³ of an eigendecomposition.
    #
    # THE THRESHOLD IS MEASURED, not chosen.  Jobs 7883150 / 7883160, MoS2 4×4
    # / n_μ=785 / rtol 1e-8, walking ncond across the capacity limit
    # (nk·nb → nspinor·n_μ = 1570).  ``ortho`` is this number; ``on-grid'' is
    # the order-matched max|Δε| over the whole band window against the stored
    # eigenvalues:
    #
    #   ncond  nk·nb  rank   ortho      on-grid max   on-grid over BSE cond
    #     70    1536  1536   1.87e-14   3.2e-05 meV        8.8e-11 meV
    #     71    1552  1551   2.97e-04   2.955   meV        0.201   meV   ←
    #     72    1568  1562   6.31e-04   6.939   meV        0.242   meV
    #     74    1600  1570   3.11e-03   23.60   meV        0.860   meV
    #     78    1632  1570   9.35e-03   88.92   meV        3.142   meV
    #
    # Losing ONE direction of 1552 (rank 1551) is what crosses the owner's
    # 0.1 meV requirement.  Over that range on-grid max|Δε| ≈ 9.0e3 · ortho
    # (7.6e3 … 1.1e4), so a cap of 1e-6 holds the on-grid energy error under
    # ~0.01 meV with a decade of margin, while every healthy configuration
    # measured — every window from nb=30 to nb=96 — sits at 1.9e-14, EIGHT
    # decades below the cap.  There is no false-positive room here.
    #
    # AND IT IS NOT A CONDITIONING PROBLEM, so do not reach for ``rtol``.
    # Same job, ncond=72, tightening the amplification cap makes it WORSE
    # because it discards more of a basis that already cannot span:
    #     rtol 1e-8 → rank 1562, ortho 6.31e-04, on-grid  6.9 meV
    #     rtol 1e-6 → rank 1494, ortho 1.04e-02, on-grid 80.3 meV
    #     rtol 1e-4 → rank 1086, ortho 2.61e-02, on-grid 219.3 meV
    # The lever for the EXACT-SPAN route is n_μ (or a narrower window), never
    # the truncation tolerance.  ``htransform_rank_multiplier`` is different:
    # it declares a reduced cross-k MODEL and restores the row-isometry with a
    # per-k polar factor before this gate.  Its accuracy is decided by an
    # observable A/B, not by relabeling its cut as numerical noise.
    #
    # ``LORRAX_FH_ORTHO_TOL`` overrides; 0 disables. Never disable to make a
    # run finish — the failure it catches is a wrong number, not a crash.
    rep_ = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=rep_)
    def _ortho_all_k(c):
        nb_ = c.shape[1]
        G = jnp.einsum('kim,kjm->kij', c, jnp.conj(c), optimize=True)
        return jnp.max(jnp.abs(G - jnp.eye(nb_, dtype=G.dtype)[None]))

    _ortho = float(_ortho_all_k(ctilde))
    _tol = resolve_fh_ortho_tol(log_fn)
    log_fn(f"  [gate] ctilde orthonormality over ALL {int(ctilde.shape[0])} k: "
           f"max|C Cᴴ − I| = {_ortho:.3e}  (cap {_tol:.1e}; measured "
           f"conversion: on-grid max|Δε| ≈ 9.0e3 × this, so this run's "
           f"on-grid energy error is ≈ {9.0e3 * _ortho:.2e} meV)")
    if _tol > 0.0 and _ortho > _tol:
        raise ValueError(
            f"build_fH_R: the Galerkin coefficients are NOT orthonormal — "
            f"max|C Cᴴ − I| = {_ortho:.3e} over all k, above the {_tol:.1e} "
            f"cap.  fH's eigenvalues are then not f(ε_n) and the recovered "
            f"ENERGIES are wrong on the coarse grid by roughly "
            f"{9.0e3 * _ortho:.2e} meV, silently.  Cause, in order of "
            f"likelihood: (1) ψ-at-centroids cannot span the band window — "
            f"check the rank line from streaming_galerkin_solve for "
            f"rank < nk·nb.  THREE repairs, and which one applies depends on "
            f"WHY the span failed: MORE CENTROIDS or a NARROWER window if the "
            f"basis is genuinely too small for the bands; or an explicit "
            f"reduced model order, htransform_rank_multiplier >= 1, if the "
            f"span failed because nk·nb grew with the K-POINT COUNT.  The "
            f"exact-span route retains up to nk·nb directions, so on a dense "
            f"metal grid it demands a basis that grows with N_k, which the "
            f"method's own scaling contract does not (Wu et al. S1 estimates "
            f"the QRCP basis at 20–40·N_b, INDEPENDENT of N_k, and reports "
            f"N_mu saturating as N_k grows).  Measured: Na 512 k × 12 bands "
            f"with the validated c1016 basis retains the full column rank "
            f"2032 and refuses here at 2.324e-4, while the SAME basis "
            f"completes at three bands — buying centroids is the wrong "
            f"repair for that shape.  Never rtol: same job, tightening the "
            f"amplification cap made ortho WORSE (1e-8 → 6.31e-4, 1e-6 → "
            f"1.04e-2, 1e-4 → 2.61e-2).  (2) the G accumulation summed Q Qᴴ "
            f"per band chunk instead of summing into Q first.  Override with "
            f"LORRAX_FH_ORTHO_TOL only to reproduce a known-bad run.")

    fH_k, fH_R = _build(ctilde, f_eps)
    return fH_k, fH_R, (a_f, n_f, shift), f_eps


def newton_inv(a: float, n: float, shift: float, y: jax.Array,
               max_iter: int = 50) -> jax.Array:
    """Newton inversion — matches Fortran newton_inv(). Works in unshifted space."""
    dxmax = a / 2.0
    x0 = y + shift  # Fortran initial guess: x = y + s

    def body_fun(_, x_curr):
        res = fun(a, n, shift, x_curr) - y
        df_val = dfun(a, n, shift, x_curr)
        dx = jnp.where(jnp.abs(df_val) > 1e-14, -res / df_val, 0.0)
        dx = jnp.clip(dx, -dxmax, dxmax)
        return x_curr + dx

    return lax.fori_loop(0, max_iter, body_fun, x0)


def load_wfns_and_enk_for_sigma(wfn, sym, nval: int, ncond: int, nband: int):
    nelec = int(wfn.nelec)
    nsigmarange = (int(nelec - nval), int(nelec + ncond))
    enk_sigma, _ = get_enk_bandrange(wfn, sym, nsigmarange, nsigmarange)
    return nsigmarange, jnp.asarray(enk_sigma).transpose(1, 0)


def setup_wfn_and_sym(wfn_file: str, mesh_xy: Mesh | None = None):
    # Pass the device mesh so the loader can pick a sharded read backend
    # (the collective phdf5 FFI read, on GPU or the CUDA-free host lib),
    # band-sharding ψ instead of replicating the whole WFN on every rank.
    # Single-process / no-mesh transparently stays eager.
    wfn = WfnLoader(wfn_file, mesh=mesh_xy)
    sym = symmetry_maps.SymMaps(wfn)
    return wfn, sym


def _clean_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip()
    if not label:
        return None
    # Map common aliases to actual Unicode Greek letters so Matplotlib displays them
    if label.lower() in {'gg', 'gamma', '\\u0393'}:
        label = 'Γ'
    elif label.lower() in {'gl', 'lambda', '\\u039b'}:
        label = 'Λ'
    elif label.lower() in {'gs', 'sigma', '\\u03a3'}:
        label = 'Σ'
    return label


def generate_kpath_from_qe_segments(params: dict, wfn) -> tuple[jnp.ndarray, np.ndarray, list[str | None]] | None:
    seginfo = params.get("kpoints_crystal_b")
    if not seginfo:
        return None
    segments = seginfo.get("segments", [])
    if len(segments) < 2:
        return None
    nodes_crys = [np.asarray(seg["k"], dtype=float) for seg in segments]
    labels = [_clean_label(seg.get("label")) for seg in segments]
    pts_crys = [nodes_crys[0]]
    node_indices = [0]
    for i in range(len(nodes_crys) - 1):
        k0 = nodes_crys[i]
        k1 = nodes_crys[i + 1]
        n = max(1, int(segments[i + 1].get("n", 1)))
        for t in range(1, n + 1):
            alpha = t / float(n)
            pts_crys.append((1.0 - alpha) * k0 + alpha * k1)
        node_indices.append(len(pts_crys) - 1)
    kpoints = np.stack(pts_crys, axis=0)
    return jnp.asarray(kpoints, dtype=jnp.float64), np.asarray(node_indices, dtype=int), labels


def _load_centroids(centroids_path: str, fft_grid: tuple[int, int, int]) -> np.ndarray:
    centroids_frac = np.loadtxt(centroids_path, ndmin=2)
    if centroids_frac.size == 0:
        raise ValueError(f"Centroids file {centroids_path} is empty")
    fft_grid = np.asarray(fft_grid, dtype=int)
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    centroid_indices = np.mod(centroid_indices, fft_grid)
    return centroid_indices


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


def _make_logger(verbose: bool):
    return print if verbose else (lambda *_, **__: None)


def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Full-BZ QP energies from the IRREDUCIBLE-WEDGE ``eqp{0,1}.dat``.

    Reads the BerkeleyGW-columnar eqp file LORRAX's GW writes — one block
    per ``wfn.kpoints`` entry, the crystal coordinate in the block header,
    energies in eV — and returns ``(nb, nk_full)`` in RYDBERG over
    ``band_window``, which is what this module's DFT path returns.

    THE UNFOLD IS THE SERVICE'S, NOT THIS MODULE'S.  Every IBZ→full-BZ
    map in the tree goes through ``symmetry_maps.star_broadcast``, reached
    here by :func:`symmetry_maps.unfold_file_wedge_to_full_bz` — the FILE
    wedge, ``wfn.kpoints``, which is what ``eqp1.dat`` is indexed by and
    what BerkeleyGW means by the IBZ.  It shares its backend with the
    kin_ion read path; no index table crosses into this module.

    WHAT THIS REPLACED, AND WHY.  It used to require a PRE-UNFOLDED
    full-BZ text file (``nk == sym.nk_tot``, refused otherwise) whose
    ``k-point N:`` blocks it paired to full-BZ k BY POSITION, with no
    coordinate ever read.  Nothing in the tree wrote that file: it came
    from an out-of-tree ``make_eqp_htformat.py`` that joined
    ``eqp_g0w0.dat`` against ``eqp1.dat`` to do the unfold by hand.  That
    is bespoke unfolding one hop upstream, plus a positional pairing that
    a re-ordered file passes silently.  Now the wedge file is read
    directly and the service does the unfold, so the converter has no job
    left and the position never enters.

    The block coordinates are CHECKED against the deck's own wedge
    (``sym.unfolded_kpts[sym.kirr_fullids]``) rather than trusted — a
    file from another deck, or in another order, is refused here instead
    of producing a quasiparticle bandstructure with the energies on the
    wrong k.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    from gw.eqp_bgw import read_bgw_eqp
    from symmetry_maps import unfold_file_wedge_to_full_bz

    kpts_file, _e_dft_ev, e_qp_ev, band_offset = read_bgw_eqp(eqp_file)
    nk_file, nb_file = e_qp_ev.shape

    # ---- the file must be THIS deck's wedge, in the wedge's order -------
    kirr = np.asarray(sym.unfolded_kpts, dtype=np.float64)[
        np.asarray(sym.kirr_fullids, dtype=np.int64)]
    if nk_file != kirr.shape[0]:
        raise ValueError(
            f"{os.path.basename(eqp_file)} holds {nk_file} k-blocks but this "
            f"deck's irreducible wedge has {kirr.shape[0]} (full BZ "
            f"{int(sym.nk_tot)}).  This reader takes the wedge file LORRAX's "
            f"GW writes; the pre-unfolded full-BZ form is no longer read.")
    # Written by ``%13.9f``, so equality is to that many places; the
    # comparison is modulo a lattice vector because either side may carry
    # a k in a different periodic image.
    dk = np.asarray(kpts_file, dtype=np.float64) - kirr
    dk -= np.rint(dk)
    worst = float(np.max(np.abs(dk))) if dk.size else 0.0
    if worst > 1e-6:
        bad = int(np.argmax(np.max(np.abs(dk), axis=1)))
        raise ValueError(
            f"{os.path.basename(eqp_file)} block {bad} is at "
            f"{np.asarray(kpts_file)[bad].tolist()} but this deck's wedge "
            f"point {bad} is {kirr[bad].tolist()} (worst |Δk| = {worst:.2e} "
            f"over {nk_file} blocks).  The eqp file does not belong to this "
            f"wavefunction, or its k-order differs — either way its energies "
            f"would land on the wrong k.")

    # ---- absolute band window -> the file's columns ---------------------
    lo, hi = start - band_offset, end - band_offset
    if lo < 0 or hi > nb_file:
        raise ValueError(
            f"{os.path.basename(eqp_file)} covers absolute bands "
            f"[{band_offset}, {band_offset + nb_file}) but the requested "
            f"window is [{start}, {end}) — the eqp file does not span the "
            f"htransform sigma window.")
    window_ev = np.asarray(e_qp_ev)[:, lo:hi]
    if np.isnan(window_ev).any():
        n_missing = int(np.isnan(window_ev).sum())
        raise ValueError(
            f"{os.path.basename(eqp_file)} is missing {n_missing} of "
            f"{window_ev.size} (k, band) entries inside the requested window "
            f"[{start}, {end}) — a short block cannot be silently padded.")

    # ---- THE unfold: wedge -> full BZ, through the service --------------
    # The FILE wedge: ``eqp1.dat`` is indexed by ``wfn.kpoints``, which is a
    # different (and on two of three committed decks a different-LENGTH)
    # k-set from the star wedge — see the register.  The call site says
    # which without the reader needing to know what ``trs_reference`` is.
    full_ev = np.asarray(unfold_file_wedge_to_full_bz(sym, window_ev))
    if full_ev.shape[0] != int(sym.nk_tot):
        raise ValueError(
            f"star_broadcast returned {full_ev.shape[0]} k-points, expected "
            f"{int(sym.nk_tot)}")

    # eV on disk (BGW convention); Ry is this module's internal unit.
    return jnp.asarray(full_ev.T / RYD_TO_EV, dtype=jnp.float64)


def initialize_wfns(input_path: str, params: dict, log_fn, eqp_file: str | None = None,
                    mesh_xy: Mesh | None = None, return_full_proj: bool = False,
                    n_guard_bands: int = 0, centroid_subset_idx=None):
    """Load ψ, build the Galerkin ``ctilde``/``B_at_mu`` over the deck's window.

    ``n_guard_bands`` widens the htransform window ABOVE the deck's
    ``(nelec − nval, nelec + ncond)`` by that many bands, and only that.  It
    is the two-window contract's one lever: ``f(ε)`` is identically zero for
    ε ≥ ``shift`` := ``max_k ε[nb−1]``, so the top of the htransform's OWN
    window contributes nothing to ``fH`` and ``jnp.linalg.eigh`` fills those
    slots out of fH's null space.  A caller that needs the top of its window
    BACK — ``bse.vq_interp.refit_prepare``, whose ζ' must be fitted on exactly
    the producer's window — buys guard bands above it so every band it asks
    for is interior.  Default ``0`` is the historical behaviour, byte for
    byte: no caller that omits it loads a band it did not load before.

    The widening moves ``ncond`` (and, with it, ``nband``, which is what
    ``Meta`` zero-pads ψ above — a guard band past ``nband`` would arrive as
    exact zeros and the Galerkin solve would never say so).  It does NOT move
    ``nval``: the shoulder is at the TOP of the window, and the bottom edge is
    the deck's own.

    Measured, on the two parents of
    ``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md``:
    with zero guards the top four bands of a 20-band window collapse to
    ``min_k ‖O[m,:]‖`` = 0.23…0.27 in the α-space overlap, and the on-grid
    tile null reads 1.267 against a 5.0e-02 bracket.
    """
    from file_io.centroids import load_centroids as _shared_load_centroids

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    # Build the mesh up front so the loader is mesh-aware (sharded read).
    if mesh_xy is None:
        mesh_xy = _build_mesh_xy()
    wfn_file = _resolve(params["wfn_file"])
    wfn, sym = setup_wfn_and_sym(wfn_file, mesh_xy=mesh_xy)
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    _, centroid_indices, n_rmu = _shared_load_centroids(
        centroid_path, tuple(int(x) for x in wfn.fft_grid))
    if centroid_subset_idx is not None:
        _n_parent = int(n_rmu)
        _subset = validate_centroid_subset_idx(
            centroid_subset_idx, _n_parent)
        centroid_indices = np.asarray(centroid_indices)[_subset]
        n_rmu = int(_subset.size)
        log_fn(
            f"  [reduced-galerkin] centroid fit born in the downfold basis: "
            f"ordered parent subset {_n_parent} -> {n_rmu} rows.  No "
            "parent-width B_at_mu or projected wavefunction is formed.")

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    n_guard_bands = int(n_guard_bands)
    if n_guard_bands < 0:
        raise ValueError(
            f"initialize_wfns: n_guard_bands={n_guard_bands} is negative; the "
            f"guard bands sit ABOVE the deck's window and there is no such "
            f"thing as a negative one.  Pass 0 for the historical window.")
    if n_guard_bands:
        _ncond_deck, _nband_deck = ncond, nband
        ncond = _ncond_deck + n_guard_bands
        nband = max(_nband_deck, int(wfn.nelec) + ncond)
        _b_hi = int(wfn.nelec) + ncond
        if _b_hi > int(wfn.nbands):
            raise SystemExit(
                f"initialize_wfns: {n_guard_bands} guard band(s) above the "
                f"deck's window would need bands up to {_b_hi}, and "
                f"{os.path.basename(wfn_file)} carries {int(wfn.nbands)}.  "
                f"A guard band the file cannot supply arrives as EXACT ZEROS "
                f"(``Meta``'s past-mnband pad), which the Galerkin solve "
                f"absorbs without complaint and the f-transform then reports "
                f"as a perfectly representable band.  Re-run the WFN with "
                f"more bands, or drop the guard count to "
                f"{max(0, int(wfn.nbands) - int(wfn.nelec) - _ncond_deck)}.")
        log_fn(f"  [two-window] htransform window WIDENED by "
               f"{n_guard_bands} guard band(s) above the deck's: ncond "
               f"{_ncond_deck} → {ncond} (bands "
               f"[{int(wfn.nelec) - nval}, {_b_hi}) of {int(wfn.nbands)}); "
               f"nband {_nband_deck} → {nband} so the guards are READ rather "
               f"than zero-padded.  f(ε) ≡ 0 for ε ≥ max_k ε[nb−1], so the "
               f"guards absorb the shoulder and every band the caller asks "
               f"for is interior to fH.")
    # Htransform interpolates the one-particle Hamiltonian.  A bispinor GW
    # deck changes the QP energies supplied through eqp1.dat; it does not
    # require rebuilding a four-component interpolation basis.  The source
    # WFN's canonical spinors therefore remain the sole wavefunction route.
    # This also keeps scalar/bare/full energy arms in one fixed gauge.
    _gw_bispinor = bool(params.get("bispinor", False))
    bispinor = False
    if _gw_bispinor:
        log_fn("  [route] bispinor QP energies with the canonical spinor "
               "WFN interpolation basis (no synthetic small components)")
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    # ``sys_dim`` IS A DECK KEY AND ``Meta`` HAS NO FIELD FOR IT.  The GW
    # driver stamps it on the Meta it builds (``gw_jax.main``:
    # ``meta.sys_dim = config.sys_dim``) and every dimension-aware consumer
    # reads ``meta.sys_dim`` off that stamp.  This is the OTHER Meta in the
    # tree — the one the bandstructure driver, ``bse.bse_densify``'s
    # ``bse_k_grid`` densification, ``bse.exciton_bands``' ``--w-coarse-grid``
    # leg and ``bse.vq_interp.refit_prepare`` all use — so it is stamped here,
    # from the same key, once, rather than at each of those four call sites.
    #
    # WITHOUT THIS the stamp was simply absent, and every consumer that reads
    # it defensively fell back to bulk 3D: ``gw.coulomb.get_kernel(None)``
    # returns ``Bulk3D``, and the former head-densify guard returned without
    # deciding anything.  The second one was the expensive
    # case — on a slab deck the C1 head channel re-attached ``8π/|q|²``, the
    # untruncated 3D pole, where the true 2D head goes as ``8π·z_c/|q_∥|``,
    # and the error GROWS as the fine grid densifies.  See
    # ``gw.head_densify.build_fine_head_scalars`` and
    # ``bse.bse_densify.build_w_head_channel``.
    #
    # ``read_lorrax_input`` resolves ``sys_dim`` from ``gw_config._DEFAULTS``
    # for every deck, so the key is present on every real caller; a params
    # dict assembled by hand and missing it is refused rather than defaulted,
    # because a default here is exactly the silent 3D assumption above.
    if "sys_dim" not in params:
        raise KeyError(
            "initialize_wfns: params carries no 'sys_dim', so the Meta this "
            "builds cannot be stamped with the deck's dimensionality and "
            "every dimension-aware consumer downstream would silently assume "
            "bulk 3D.  Build params with gw.gw_config.read_lorrax_input (it "
            "fills the key from _DEFAULTS for every deck), or set it "
            "explicitly.  Do NOT default it here.")
    meta.sys_dim = int(params["sys_dim"])
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn, sym, nval, ncond, nband)

    # Optionally override energies with EQP values from a file only if explicitly requested via CLI
    if eqp_file:
        # --eqp-file is an explicit request: a file that cannot be found or
        # parsed must refuse, not silently fall back to the DFT energies —
        # the output would be a DFT bandstructure labeled as quasiparticle.
        eqp_path = _resolve(eqp_file)
        if not os.path.isfile(eqp_path):
            raise SystemExit(f"FATAL: --eqp-file was given but not found: {eqp_path}")
        try:
            enk_sigma = read_eqp_energies(eqp_path, sym, nsigmarange)
            log_fn(f"Using EQP energies from {os.path.basename(eqp_path)} for band window {nsigmarange}")
        except Exception as exc:
            raise SystemExit(
                f"FATAL: --eqp-file {os.path.basename(eqp_path)} could not be "
                f"consumed: {exc}\nExpected LORRAX's GW ``eqp1.dat`` as "
                f"written: BerkeleyGW columns on the IRREDUCIBLE WEDGE — "
                f"{getattr(sym, 'nk_red', '?')} k-blocks for this deck, each "
                f"a '(3f13.9,i8)' crystal-coordinate header followed by that "
                f"many '(2i8,2f15.9)' band rows in eV.  Pass it directly; the "
                f"unfold to this deck's {getattr(sym, 'nk_tot', '?')} full-BZ "
                f"k-points happens here, through the symmetry service.  No "
                f"pre-conversion step is needed or accepted."
            ) from exc

    band_range = (int(nsigmarange[0]), int(nsigmarange[1]))
    with mesh_xy:
        out = streaming_galerkin_solve(
            wfn, sym, meta, centroid_indices, mesh_xy, band_range,
            rtol=1e-8, log_fn=log_fn, bispinor=bispinor,
            return_full_proj=return_full_proj,
            # Deck key, same family as the fH_q eigh; the CLI --eigh-backend
            # override is scoped to compute_wfns_fi and deliberately not
            # threaded here — the deck is the source of truth for the fit.
            #
            # RESOLVED, not raw.  ``eigh_backend`` and ``use_low_mem_eigh``
            # are two spellings of ONE axis and ``gw_config.resolve_eigh_
            # backend`` is the single place they combine; reading the raw
            # key here meant a deck that said ``use_low_mem_eigh = true``
            # got the native replicated Gram eigh anyway — the key was
            # parsed, defaulted, stored and read by nobody on this driver.
            eigh_backend=resolve_eigh_backend(params),
            rank_multiplier=params.get("htransform_rank_multiplier", 20.0),
            qr_eps=params.get("htransform_qr_eps", 1.0e-3),
            qrcp_seed=params.get("htransform_qrcp_seed", 0),
        )
    S, ctilde, B_at_mu = out[:3]
    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, nb={band_range[1]-band_range[0]}, rank={ctilde.shape[2]}")
    if return_full_proj:
        return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma, out[3]
    return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma


def initialize_kpath(wfn, params):
    info = generate_kpath_from_qe_segments(params, wfn)
    if info is None:
        return None, None, None, None, []
    kpath_frac, node_indices, node_labels = info
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    k_cart = np.asarray(kpath_frac) @ bvec * blat * (2.0 * np.pi)
    seg_len = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    # Compare against the CANONICAL label ``_clean_label`` emits (the real
    # 'Γ' character).  Until 2026-08-01 this tested the eight-character
    # literal ``'\\u0393'``, which no cleaned label ever equals, so
    # gamma_positions was always empty and the ``argmin(norm)`` fallback in
    # ``h_transform`` picked index 0 — right only because Γ headed every
    # path used so far and ties (the zero pad rows) resolve to the first
    # index (scorecard BE).
    gamma_positions = [int(idx) for idx, lbl in zip(node_indices, node_labels)
                       if (lbl or '').strip() == 'Γ']
    return kpath_frac, x_path, node_indices, node_labels, gamma_positions


def resolve_local_vbm_index(nelec: int, band_start: int,
                            n_return_bands: int) -> int:
    """Return the VBM column inside one absolute contiguous band window."""
    idx = int(nelec) - 1 - int(band_start)
    if idx < 0 or idx >= int(n_return_bands):
        raise ValueError(
            "htransform: the returned absolute band window does not contain "
            f"the VBM: nelec={int(nelec)}, band_start={int(band_start)}, "
            f"n_return_bands={int(n_return_bands)} gives local index {idx}. "
            "A bandstructure referenced to the VBM must retain that level.")
    return idx


def h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log_fn, mesh_xy: Mesh,
                a_band_index: int | None = None, diagnostics: bool = True,
                band_start: int = 0, n_return_bands: int | None = None):
    from time import perf_counter as _perf   # instrument:
    nk = int(meta.nkx * meta.nky * meta.nkz)
    states = ctilde.shape[1]
    rank = ctilde.shape[2]
    kgrid = (meta.nkx, meta.nky, meta.nkz)
    nb_keep = states if n_return_bands is None else int(n_return_bands)
    if nb_keep <= 0 or nb_keep > states:
        raise ValueError(
            f"htransform: n_return_bands={nb_keep} must be in [1, {states}] "
            "for the fitted Galerkin window.")
    fermi_band_idx = resolve_local_vbm_index(
        int(wfn.nelec), int(band_start), nb_keep)
    n_guard_bands = int(states) - nb_keep
    log_fn(
        f"  [two-window] standalone htransform returns {nb_keep} band(s) "
        f"from absolute window [{int(band_start)}, "
        f"{int(band_start) + nb_keep}) and fits {states} band(s) "
        f"({n_guard_bands} guard band(s) above).")

    rep = NamedSharding(mesh_xy, P())  # fully replicated, used for diagnostics

    _t0 = _perf()                                          # instrument:
    coeffs = ctilde.reshape(nk, states, rank)
    fH_k, fH_R, (a_f, n_f, shift), f_eps = build_fH_R(
        coeffs, enk_sigma, kgrid, mesh_xy, a_band_index=a_band_index, log_fn=log_fn)
    jax.block_until_ready((fH_k, fH_R, f_eps))             # instrument:
    timing.record("ht.build_fH_R", _perf() - _t0)          # instrument:

    # The top of the fit window is identically invisible to fH at the k where
    # it sets ``shift``.  Returning that band gives an arbitrary null-space
    # direction even when ctilde is exactly orthonormal.  The BSE consumer has
    # enforced this two-window contract since the fifth-wall diagnosis; the
    # standalone band writer must pass through the same gate before it can
    # publish a curve.
    from .bse_setup import _f_shoulder_gate
    _f_shoulder_gate(
        f_eps, 0, nb_keep, shift, log_fn, rank=rank,
        where="htransform")

    # Diagnostics. Split into two small jits:
    #   _diag_stats_fast  — sharding-respecting reductions on the full fH_k
    #     (no gather needed; psum on the (x,y) face).
    #   _diag_eig_at_gamma — eigvalsh on fH_k[0] only. Pull fH_k[0:1] to
    #     replicated FIRST (7.4 MB at our scale), so the eigvalsh runs
    #     single-device locally rather than driving an all-gather of the
    #     full (rank, rank) face inside the eigvalsh module.
    @jax.jit
    def _diag_stats_fast(fH_k):
        return (jnp.min(jnp.real(fH_k)),
                jnp.max(jnp.real(fH_k)),
                jnp.max(jnp.abs(jnp.imag(fH_k))))

    # Takes the WHOLE ``f_eps`` and returns the four printed 5-element
    # windows as well: ``f_eps[:, 0]`` and each ``np.array(x[a:b])`` below
    # was its own eager ``dynamic_slice``/``squeeze`` module (4 of this
    # driver's 137, job 7884866).  Slicing is exact, and these values reach
    # the log only — never ``bandstructure.dat``.
    @partial(jax.jit, static_argnames=('states',))
    def _diag_eig_at_gamma(fH_k0_rep, f_eps_in, states):
        eigs0 = jnp.sort(jnp.linalg.eigvalsh(fH_k0_rep))
        f_exp0 = jnp.sort(f_eps_in[:, 0])
        eig_err = jnp.max(jnp.abs(eigs0[:f_exp0.shape[0]] - f_exp0))
        return (eig_err, f_exp0[:5], eigs0[:5], f_exp0[-5:],
                eigs0[states - 5:states])

    _t0 = _perf()                                          # instrument:
    # ── THE DIAGNOSTICS GATE ──────────────────────────────────────────────
    # This block plus ``_gamma_rt`` below is 1.442 s of the 4.327 s cache-cold
    # ``h_transform`` stage — 33 %, 7 XLA programs — and 0.120 s warm
    # (PROFILE_htransform_exciton §1.2).  Every value it computes reaches a
    # ``log_fn`` line and nothing else; none of them reaches
    # ``bandstructure.dat``.  And ``log_fn`` is ``print if verbose else a
    # no-op`` (``_make_logger``), so WITHOUT ``--verbose`` the driver was
    # spending a third of the stage computing numbers it then discarded.
    #
    # ``fH_k`` exists ONLY for this block: the kpath solve consumes ``fH_R``
    # alone.  At the reference shape it is (64, 768, 768) complex128 = 576 MiB
    # global, held alive across the whole solve for four log lines.  Dropping
    # the reference right after the block frees it before the kpath batches
    # allocate their own (bs, rank, rank) temporaries — the two peaks stop
    # overlapping.  (The buffer is still PRODUCED, because it is an output of
    # ``build_fH_R``'s single jit and ``fH_R`` is its ifft; only its residency
    # is at stake here, which is the 576 MiB the profile prices.)
    if diagnostics:
        re_min, re_max, im_max = _diag_stats_fast(fH_k)
        log_fn("fH_k real range: [{:.3e}, {:.3e}], |imag|max={:.3e}".format(
            float(re_min), float(re_max), float(im_max)))
        fH_k0_rep = jax.device_put(fH_k[0], rep)  # (rank, rank) replicated, ~7 MB
        _eig_err, _f_head, _e_head, _f_tail, _e_tail = jax.device_get(
            _diag_eig_at_gamma(fH_k0_rep, f_eps, int(states)))
        fH_eig_err = float(_eig_err)
        log_fn(f"fH(k=0) eigenvalue error vs f(eps): {fH_eig_err:.6f} Ry = {fH_eig_err * 13.6057:.3f} eV")
        log_fn(f"  f(eps) first 5: {np.asarray(_f_head)}")
        log_fn(f"  fH eig first 5: {np.asarray(_e_head)}")
        log_fn(f"  f(eps) last 5:  {np.asarray(_f_tail)}")
        log_fn(f"  fH eig last 5:  {np.asarray(_e_tail)}")
    else:
        log_fn("  [route] fH diagnostics: OFF (fH_k stats, Γ eigen-check and "
               "the Γ round-trip are not computed; --fh-diagnostics=on "
               "restores them)")
    timing.record("ht.diagnostics", _perf() - _t0)         # instrument:

    _t0 = _perf()                                          # instrument:
    # ── THE METRIC ROUTE — S is the identity, and the code already knows it ──
    #
    # ``_kpath_batch`` below reduces a GENERALIZED eigenproblem fH_q c = λ S c
    # by two triangular solves against chol(S) per q.  But S has exactly ONE
    # producer — ``streaming_galerkin_solve`` returns
    #     S = jax.jit(lambda: jnp.eye(rank, dtype=jnp.complex128), …)()
    # (see its closing lines) — because the α basis is Cholesky-orthogonalised
    # upstream, which is the whole point of ``_finalize``'s ``coeffs @ L``.  The
    # other consumer of the identical math, ``bse_setup._q_batch``, already
    # takes S = I for granted: it calls ``jnp.linalg.eigh(fH_q)`` with no
    # metric at all.  The two solves here are vestigial.
    #
    # WHAT THEY COST.  Each ``solve_triangular`` of a (rank, rank) triangular
    # factor against a (rank, rank) right-hand side is rank³/2 complex MACs, so
    # the pair is ~8·rank³ real flops per q against the ~5.3·rank³ of the
    # eigenvalues-only Hermitian eigensolve they feed.  They are the LARGER
    # half of the arithmetic in the batch — and the Fourier sum this whole
    # workstream is about is 8·N_k·rank², a further factor rank/N_k below both.
    #
    # WHAT THEY CHANGE.  With S = I the ridge makes S_sym = (1+1e-10)·I, so
    # chol(S) = sqrt(1+1e-10)·I and the pair divides every eigenvalue by
    # (1+1e-10) — a systematic -1e-10 RELATIVE shift of λ, applied for no
    # reason.  Skipping them is therefore not bit-identical; it is analytically
    # exact and removes a small bias rather than adding one.  The measured
    # energy delta on the reference deck is in HTRANSFORM_FFT.md §6.
    #
    # The route is decided ONCE, from the data, and announced.  A caller that
    # ever hands this driver a non-identity S keeps the Cholesky path
    # unchanged — the branch is on a measured deviation, not on an assumption.
    @partial(jax.jit, out_shardings=rep)
    def _s_dev_from_eye(S_in):
        return jnp.max(jnp.abs(S_in - jnp.eye(rank, dtype=S_in.dtype)))

    _s_dev = float(_s_dev_from_eye(S))
    metric_route = "identity" if _s_dev == 0.0 else "cholesky"
    log_fn(f"  [route] fH_q metric: {metric_route}  (max|S - I| = {_s_dev:.3e}; "
           f"'identity' skips 2 triangular solves of {rank}³/2 complex MACs per q)")

    S_chol = None
    if metric_route == "cholesky":
        # Built ONLY on the path that uses it — on the identity route this
        # compile (one XLA program, 0.34 s cache-cold) is not paid at all,
        # which is what keeps the route program-count-neutral.
        @jax.jit
        def _build_S_chol(S):
            S_sym = (S + S.conj().T) * 0.5
            S_sym += 1e-10 * jnp.mean(jnp.real(jnp.diag(S_sym))) * jnp.eye(rank, dtype=S_sym.dtype)
            return jnp.linalg.cholesky(S_sym)
        S_chol = _build_S_chol(S)

    R_grid = jnp.asarray(build_R_grid_np(kgrid))
    jax.block_until_ready(                                 # instrument:
        (R_grid,) if S_chol is None else (S_chol, R_grid)) # instrument:
    timing.record("ht.S_chol", _perf() - _t0)              # instrument:

    # ── Kpath-batch processing ───────────────────────────────────────────
    # fH_R stays SHARDED P(None, 'x', 'y'): the (rank, rank) face is split
    # across the mesh, the lattice-R axis is not.  The q-Fourier sum contracts
    # ONLY over R, so every device builds its own (i, j) tile with NO
    # communication; the single collective is the reshard onto the q axis just
    # before the eigvalsh, after which each device owns whole (rank, rank)
    # matrices for its own q-rows and the eigvalsh runs ndev-parallel.
    #
    # It used to be ``fH_R_rep = jax.device_put(fH_R, rep)``.  That is
    # nk · rank² · 16 B on EVERY device (MoS2 12×12: ~51 GB/rank at rank 4716,
    # and it is the term that breaks a 90 GB rank envelope at n_μ ≈ 3.1k) and,
    # because the source is sharded, JAX routes it through ``x._value`` — a
    # host gather of the same size per process.  Identical de-replication to
    # ``bandstructure/bse_setup.py::_fourier`` (see its lines 166-178 / 247-259
    # for the measured OOM this removes); the arithmetic is unchanged.

    # Sharding specs for batched (bs, rank, rank) → (bs, rank) eigvalsh.
    batch_mat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    batch_eig_shard = NamedSharding(mesh_xy, P(('x', 'y'), None))
    face_ij_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=batch_eig_shard)
    def _kpath_batch(batch_k, fH_R, S_chol):
        # batch_k: (bs, 3) replicated; fH_R: (nk, rank, rank) at P(None,'x','y');
        # S_chol: (rank, rank) replicated.  The einsum contracts over the
        # (replicated) R axis only → local on each (i, j) tile.
        phase = jnp.exp(-2j * jnp.pi * (batch_k @ R_grid.T))           # (bs, nk)
        mat = 0.5 * jnp.einsum('bk,kij->bij', phase, fH_R)             # (bs, rank, rank)
        # Pin the contraction OUTPUT to the (i, j) layout: left free, XLA
        # materialises the whole (bs, rank, rank) batch on every device before
        # the reshard below (bs·rank²·16 = 11.4 GiB/device at bs=32/rank 4716).
        mat = jax.lax.with_sharding_constraint(mat, face_ij_shard)
        # Reshard (i,j)→q FIRST, then hermitize: on the q-sharded layout each
        # device owns whole matrices, so the transpose is local.  The other
        # order costs a second all-to-all for ``swapaxes``.
        mat = jax.lax.with_sharding_constraint(mat, batch_mat_shard)
        mat = mat + jnp.swapaxes(mat, 1, 2).conj()

        if S_chol is None:
            # S = I (see the metric-route block above).  ``mat`` was hermitized
            # two lines up and ``eigvalsh`` reads one triangle, so the dropped
            # ``(z + zᴴ)/2`` cannot move an eigenvalue either.
            return jax.vmap(jnp.linalg.eigvalsh)(mat)

        def _solve_one(m):
            y = jsp_linalg.solve_triangular(S_chol, m, lower=True)
            z = jsp_linalg.solve_triangular(S_chol, y, lower=True, trans=2)
            return jnp.linalg.eigvalsh((z + z.conj().T) * 0.5)

        return jax.vmap(_solve_one)(mat)

    # Round-trip diagnostic at Γ — single q, kept (i, j)-sharded (never whole
    # on any device: rank²·16/ndev instead of rank²·16).
    # ``q0`` is a numpy constant, not ``jnp.zeros``: eagerly that was two more
    # single-primitive modules (``convert_element_type``, ``broadcast_in_dim``)
    # for three zeros that only ever appear as a jit constant anyway.
    q0 = np.zeros((1, 3), dtype=np.float64)
    face_one_shard = NamedSharding(mesh_xy, P('x', 'y'))

    # The residual reduction is INSIDE the jit.  Eagerly it was four more
    # modules (``subtract``, ``abs``, ``_reduce_max``, ``_reduce_min``); the
    # arithmetic and the shardings of both operands are unchanged, and the
    # scalar reaches the log only.
    @jax.jit
    def _gamma_rt(fH_R, fH_k):
        phase0 = jnp.exp(-2j * jnp.pi * (q0 @ R_grid.T))
        m = 0.5 * jnp.einsum('bk,kij->bij', phase0, fH_R)
        m = jax.lax.with_sharding_constraint(m, face_ij_shard)
        m = (m + jnp.swapaxes(m, 1, 2).conj())[0]
        m = jax.lax.with_sharding_constraint(m, face_one_shard)
        # The canonical flattened coarse grid starts at Gamma.  ``m`` is the
        # q=0 reconstruction, so comparing it with any other k row can hide a
        # genuine ordering/sign error behind an accidental smaller residual.
        return jnp.max(jnp.abs(fH_k[0] - m))

    _t0 = _perf()                                          # instrument:
    if diagnostics:
        rt_err = float(_gamma_rt(fH_R, fH_k))
        log_fn(f"FFT Γ round-trip max error: {rt_err:.3e}")
    timing.record("ht.gamma_roundtrip", _perf() - _t0)     # instrument:
    # Last reader of ``fH_k``; see the diagnostics-gate block above.
    del fH_k

    fermi_energy = float(wfn.efermi)
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = kpath_data
    energies_on_path = None
    energies_sorted = None
    path_range = None
    gamma_exact = None

    if kpath_frac is not None:
        # Wrap + pad in ONE jit.  Eagerly this was five single-primitive
        # modules — ``add``, ``remainder``, ``subtract`` for the wrap and
        # ``broadcast_in_dim`` + ``concatenate`` for the pad (job 7884866).
        # Op-for-op identical, so the k-points fed to ``_kpath_batch`` are
        # bit-unchanged: an add feeding a remainder feeding a subtract, with
        # no multiply anywhere for a fusion to contract into an FMA.
        @partial(jax.jit, static_argnames=('n_pad',))
        def _prep_kpath(kf, n_pad):
            wk = (kf + 0.5) % 1.0 - 0.5
            if n_pad:
                wk = jnp.concatenate(
                    [wk, jnp.zeros((n_pad, 3), dtype=wk.dtype)], axis=0)
            return wk

        # Pad nq to a multiple of batch_size — every batch has the same
        # shape, _kpath_batch compiles ONCE.
        #
        # The q axis is split over the WHOLE mesh by
        # ``P(('x','y'), None, None)``.  Its smallest legal nonempty extent is
        # therefore exactly ndev, which gives each device ONE whole
        # ``(rank, rank)`` matrix.  A fixed width of 32 was placement-legal at
        # P16 but assigned TWO rank-4800 matrices/device; the native eigvalsh
        # workspace then requested 13.73 GiB and killed the CrI3 200-band arm
        # after every fH/Gamma gate had passed (JID 57538651).  This is the
        # native path's memory plan: minimize concurrent whole matrices while
        # retaining ndev-way q parallelism.  A distributed-within-matrix path
        # belongs to ``distrib_la.plan``, not a second eig implementation here.
        #
        # ``padded_extent`` is the canonical placement owner also used by
        # bse_setup.  Starting from one means this policy cannot drift from
        # the divisor that judges the actual NamedSharding.  Extra q are zero
        # rows and ``_post_kpath[:nq]`` discards them, so only scheduling and
        # the static executable shape change, never a retained value.
        _t0 = _perf()                                      # instrument:
        batch_size = _pad_to(mesh_xy, ('x', 'y'), 1)
        ndev = spec_divisor(mesh_xy, P(('x', 'y'), None), axis=0)
        matrices_per_device = batch_size // ndev
        matrix_bytes = rank * rank * np.dtype(np.complex128).itemsize
        log_fn(
            f"  kpath native eig ledger: q-batch={batch_size}, ndev={ndev}, "
            f"whole matrices/device={matrices_per_device}; one "
            f"({rank}, {rank}) complex128 operand={matrix_bytes / 2**30:.3f} "
            f"GiB/device (backend eigensolver workspace excluded)")
        nq = int(kpath_frac.shape[0])
        n_pad = round_up(nq, batch_size) - nq
        wrapped_k = _prep_kpath(kpath_frac, int(n_pad))
        nq_padded = wrapped_k.shape[0]
        jax.block_until_ready(wrapped_k)                   # instrument:
        timing.record("ht.kpath_prep", _perf() - _t0)      # instrument:
        _t0 = _perf()                                      # instrument:
        lambda_q_list = []
        for i in range(0, nq_padded, batch_size):
            batch_eigs = _kpath_batch(wrapped_k[i:i+batch_size], fH_R, S_chol)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)
        timing.record("ht.kpath_loop", _perf() - _t0,      # instrument:
                      count=len(lambda_q_list))            # instrument:

        # Bundle concat + slice + vmap(newton_inv) + sort into ONE jit so the
        # post-loop processing emits one compile rather than 4 (concatenate,
        # sort, gather, vmap-newton).
        # ``out_shardings=(rep, rep)`` is a CORRECTNESS requirement at P>1, not
        # a placement preference.  ``batches`` arrive q-sharded
        # (``batch_eig_shard = P(('x','y'), None)``); left to inference the
        # concatenate propagates that sharding onto the outputs, and whether
        # the SPMD partitioner then replicates them or leaves them split is
        # decided by whether ``nq`` divides the device count.  When it does,
        # the ``np.asarray`` on the next line raises
        #     RuntimeError: Fetching value for `jax.Array` that spans
        #     non-addressable (non process local) devices is not possible.
        # — the whole kpath solve completes and the run dies on the final host
        # fetch.  Measured on the survey's own reference path lengths at P=4
        # (2x2 mesh): nq=13 OK, nq=40 DIED, nq=139 OK, i.e. TWO OF THE FOUR
        # reference decks could not run multi-process at all
        # (PROFILE_htransform_exciton §1.5).
        #
        # Replication is the right answer and not merely the safe one: BOTH
        # outputs are consumed on the host by every process immediately below
        # (``np.asarray``, ``np.max``, the writer gate), and they are small —
        # ``energies`` is (nq, rank) f64 and ``energies_sorted`` (nq, nb_keep),
        # 854 KB and 13 KB at nq=139 / rank 768, against the (nk, rank, rank)
        # 576 MiB arrays this driver already holds.  Values are untouched:
        # ``out_shardings`` moves data, it does not compute.
        # Gate: ``tests/test_htransform_post_kpath_sharding.py``.
        @partial(jax.jit, static_argnames=('nq', 'nb_keep'),
                 out_shardings=(rep, rep))
        def _post_kpath(batches, nq, nb_keep):
            lambda_q = jnp.concatenate(batches, axis=0)[:nq]
            energies = jax.vmap(lambda row: newton_inv(a_f, n_f, shift, row.real))(lambda_q)
            energies_sorted = jnp.sort(energies, axis=1)[:, :nb_keep]
            return energies, energies_sorted

        _t0 = _perf()                                      # instrument:
        energies_on_path, energies_sorted_jax = _post_kpath(
            tuple(lambda_q_list), int(nq), int(nb_keep))
        # Report the sharding that was ACTUALLY produced, not the one asked
        # for.  Anything other than ``P()`` here is the non-addressable-fetch
        # crash of PROFILE_htransform_exciton §1.5 waiting for a P>1 run with
        # nq divisible by the device count.  ``_post_kpath`` now pins
        # ``out_shardings=(rep, rep)``, so this line should always read ``P()``;
        # it earns its keep by reporting the spec that came back rather than
        # the one that was requested.  Gated by
        # ``tests/test_htransform_kpath_gates.py::test_post_kpath_outputs_are_replicated``.
        log_fn(f"  [gate] _post_kpath out spec: "
               f"{energies_sorted_jax.sharding.spec} "
               f"(must be P() — a q-sharded fetch dies at P>1)")
        # ``gather_to_host``, not ``np.asarray``: the line above DECLARES the
        # convention, this one SURVIVES its violation.  ``np.asarray`` goes
        # through the same ``_value`` path ``device_get`` does and raises
        # identically at P>1, so a future edit that drops the ``out_shardings``
        # pin would turn the gate line into a crash on the very next statement
        # instead of a report.  Both halves are kept deliberately.
        energies_sorted = gather_to_host(energies_sorted_jax)
        timing.record("ht.post_kpath", _perf() - _t0)      # instrument:
        _t0 = _perf()                                      # instrument:
        # The VBM index is LOCAL to the fitted band window.  Using the
        # absolute electron count here silently selected a conduction level
        # whenever the window started above band zero.
        fermi_energy = float(np.max(energies_sorted[:, fermi_band_idx]))
        if not gamma_positions:
            # Label-less path: nearest-to-Γ point, EXCLUDING the batch pad
            # rows (they are exact zeros and would win the tie on any path
            # that does not start at Γ).  Host numpy — two values for a log
            # line and the plot markers, no reason for two eager XLA modules.
            _k_np = np.asarray(jax.device_get(wrapped_k))[:nq]
            gamma_positions = [int(np.argmin(np.linalg.norm(_k_np, axis=1)))]
        gamma_exact = np.sort(np.asarray(enk_sigma[:, 0]))[:nb_keep]
        # Shift all reported energies so VBM is at 0
        if energies_on_path is not None:
            energies_on_path = energies_on_path - fermi_energy
        if energies_sorted is not None:
            energies_sorted = energies_sorted - fermi_energy
        if gamma_exact is not None:
            gamma_exact = gamma_exact - fermi_energy
        # Recompute path range after shift and report
        path_range = (float(energies_sorted.min()), float(energies_sorted.max()))
        log_fn(f"Path energy range: {path_range[0]:.6f} to {path_range[1]:.6f} Ry (VBM@0)")
        # Γ deltas (in mRy) remain unchanged by uniform shift
        delta = (energies_sorted[gamma_positions[0]] - gamma_exact) * 1000.0
        log_fn("Γ Δε (mRy): " + ", ".join(f"{d:+.2f}" for d in delta[:6]))
        # After shifting, the Fermi level indicator is at 0
        fermi_energy = 0.0
        timing.record("ht.kpath_host_tail", _perf() - _t0)  # instrument:

    return {
        "nk_total": nk,
        "nb_keep": nb_keep,
        "nb_fit": int(states),
        "band_start": int(band_start),
        "n_guard_bands": n_guard_bands,
        "fermi_energy": fermi_energy,
        "energies_on_path": energies_on_path,
        "energies_sorted": energies_sorted,
        "path_range": path_range,
        "gamma_exact": gamma_exact,
        "kpath_data": (kpath_frac, x_path, node_indices, node_labels, gamma_positions),
    }


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    for band in range(nb_keep):
        ax.plot(x_path, energies_sorted[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_sorted[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path,
                        *, band_start: int = 0, nb_fit: int | None = None):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    # Same family as ``bse_io.write_eigenvectors_stream``: a WRITER must not
    # assume the layout of what it is handed.  ``energies_on_path`` comes
    # straight out of ``_post_kpath`` with the q axis tiled over the mesh.
    energies = gather_to_host(energies_on_path)
    kpoints = gather_to_host(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy\n')
        if nb_fit is not None:
            fh.write(
                f"# absolute_band_window=[{int(band_start)},"
                f"{int(band_start) + energies.shape[1]}) "
                f"fit_bands={int(nb_fit)} "
                f"guard_bands={int(nb_fit) - energies.shape[1]}\n")
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    import time as _time
    # ── The honesty row, and why these three lines exist ──────────────────
    # ``timing.report(wall=…)`` closes the table with
    #     (untimed) = wall - Σ(top-level rows)
    # and PROFILING_TOOLS calls that the honesty row.  On this driver it read
    # -2.414 s (-60.5 %) warm and -1.283 s (-11.3 %) cache-cold at P=4
    # (PROFILE_htransform_exciton §1.6) — a NEGATIVE honesty row, which cannot
    # be read at all.
    #
    # The cause is an attribution boundary, not a mis-measurement.
    # ``initialize_communicator_stack()`` runs in this module's BODY, above
    # every import, and ``prepare_mesh`` records a ``collective_warmup``
    # section (~2.4 s at P=4) into the global collector while doing it.  That
    # is before ``_t_main``, so the row was inside Σ(rows) but outside the wall
    # it is subtracted from, and the difference came out negative — the table
    # was reporting more seconds than the clock it divides by.
    #
    # The fix is the idiom ``gw_jax`` already uses at its own ``_t_main``
    # (gw_jax.py:180-200) and ``exciton_bands`` uses via ``process_elapsed_s``:
    # RESET the collector here so the pre-main section is not double-counted,
    # read the true process start from /proc, and DECOMPOSE that pre-main span
    # into rows at report time rather than adding rows to it.  The table then
    # closes against the process wall, ``(untimed)`` is non-negative by
    # construction, and the 2.4 s that used to be unreadable appears by name.
    _t_main = _time.perf_counter()
    timing.reset()
    _pre_main = timing.process_elapsed_s()
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Hamiltonian interpolation driver")
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    parser.add_argument("--eqp-file", default=None, help="Path to EQP/sigX file to override DFT band energies")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic details")
    parser.add_argument("--a-band", type=int, default=None,
                        help="Band index (0-based) whose bandwidth sets 'a'. "
                             "E.g. nval+ncond_keep-1. Default: top band.")
    parser.add_argument(
        "--guard-bands", type=int, default=4,
        help="Bands fitted above the requested nval+ncond output window. "
             "The f-transform is identically zero at the top of its own "
             "window, so standalone output requires interior returned bands. "
             "Default: 4 (the measured shoulder depth). Zero is retained only "
             "as a red/reproduction arm and will normally refuse.")
    parser.add_argument("--fh-diagnostics", default="auto",
                        choices=("auto", "on", "off"),
                        help="fH_k range stats, the Γ eigenvalue check against "
                             "f(eps) and the Γ round-trip.  They are 33%% of "
                             "the cache-cold h_transform stage (1.442 s, 7 XLA "
                             "programs) and hold fH_k — 576 MiB at the "
                             "reference shape — alive across the whole solve, "
                             "for four log lines that never reach "
                             "bandstructure.dat.  auto (default) = follow "
                             "--verbose, i.e. compute them only if anything "
                             "will print them; on = always; off = never.")
    parser.add_argument("--eigh-backend", default=None,
                        choices=eigh_backend_choices(),
                        help="Eigensolver for the fH_q eigendecomposition of "
                             "the get_centroids_fi handoff.  auto|off = the "
                             "q-batched native path; distributed|cusolvermp|"
                             "slate|scalapack spread ONE (rank, rank) tile "
                             "over the mesh through the distrib_la door (wide "
                             "band windows).  ``distributed`` is the portable "
                             "spelling and the ONLY one that exists on a host "
                             "mesh, where it means ScaLAPACK pzheevd.  "
                             "OVERRIDES the input-file ``eigh_backend`` key "
                             "(default: use the key, which defaults to auto).")
    parser.add_argument(
        "--distrib-la-batched-route", default=None,
        choices=distrib_la_batched_route_choices(),
        help="OVERRIDES the input-file distrib_la_batched_route key for "
             "every Plan.batched call in this driver. auto preserves the "
             "backend's robust distributed route; batch_reshard moves q "
             "onto the mesh and runs whole-matrix local JAX linalg.")
    args = parser.parse_args(argv)
    log = _make_logger(args.verbose)

    # JAX persistent compile cache — the same call/pattern as gw_jax's
    # _warm_start (and run_nscf / run_sternheimer / kmeans_cli).  This CLI
    # was the one driver that never enabled it, so every htransform run paid
    # a cold XLA compile of the Galerkin/G-accum kernels.  It is now safe and
    # effective at EVERY process count (scorecard AH) — measured at P=8 on the
    # fixture, a warm run compiles 0 of 152 modules per rank instead of 152.
    # Opt out with ISDF_JAX_CACHE_DIR="" — that env value is honoured inside
    # ensure_jax_compile_cache, so nothing here needs to test for it.
    # Failures are logged and swallowed.
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as exc:
        print(f"  [jax compile cache] skipped: {exc}", flush=True)

    from gw.gw_init import read_cohsex_input
    params = read_cohsex_input(args.input)
    # Input file is the source of truth; the CLI flag is an override — and
    # BOTH go through the one resolver, which is where ``use_low_mem_eigh``
    # folds in.  This driver used to inline the precedence and never call
    # it, so the deck key moved nothing here.
    eigh_backend = resolve_eigh_backend(params, override=args.eigh_backend)
    distrib_la_batched_route = resolve_distrib_la_batched_route(
        params, override=args.distrib_la_batched_route)
    # The INTENT travels too.  ``compute_wfns_fi`` refuses at resolve time
    # under this flag rather than falling back to the whole-matrix native
    # path (bse_setup's no-fallback contract); passing only the resolved
    # library name would leave that refusal disarmed on this driver.
    use_low_mem_eigh = bool(params.get("use_low_mem_eigh", False))
    n_return_bands = int(params["nval"]) + int(params["ncond"])
    
    # Override WFN file if provided via CLI
    if args.wfn_file is not None:
        params["wfn_file"] = args.wfn_file
        log(f"Using WFN file from CLI: {args.wfn_file}")

    from common import sanity

    with timing.section("initialize_wfns"):
        wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma = initialize_wfns(
            args.input, params, log, args.eqp_file,
            n_guard_bands=args.guard_bands)
    # ── Galerkin-input gate ───────────────────────────────────────────
    # S is the ISDF overlap Gram matrix (Hermitian positive-definite by
    # construction) and ``enk_sigma`` is the band energies the whole
    # interpolation is anchored to — including, when ``--eqp-file`` is
    # given, energies read from a GW run that may itself have produced
    # garbage.  A −136 eV QP energy fed into htransform yields a
    # bandstructure.dat that is numerically finite, plots fine, and is
    # wrong.  Bracket it here, where the file name is still in scope.
    sanity.check_finite("htransform S (ISDF overlap)", S, print_fn=log)
    sanity.check_finite("htransform ctilde", ctilde, print_fn=log)
    sanity.check_finite("htransform E_nk (Ry)", enk_sigma, print_fn=log)
    # Bandwidth, not absolute energy: the zero of a pseudopotential
    # eigenvalue is convention-dependent, but the *spread* of a Σ-window
    # band set is not — 20 Ry (272 eV) is far wider than any real
    # semicore-to-conduction window and so only fires on gross
    # corruption.  A subtler check (comparing --eqp-file energies against
    # the DFT ones they replace) is proposed but not implemented here;
    # see the workstream-O report.
    _enk = np.asarray(jax.device_get(enk_sigma), dtype=np.float64)
    if _enk.size:
        _spread = float(_enk.max() - _enk.min())
        log(f"  E_nk spread: {_spread:.4f} Ry over {_enk.size} states")
        sanity.check_in_range(
            "htransform E_nk bandwidth (Ry)", np.array([_spread]),
            0.0, 20.0, unit="Ry", print_fn=log)

    kpath_data = initialize_kpath(wfn, params)
    _diag_on = (args.verbose if args.fh_diagnostics == "auto"
                else args.fh_diagnostics == "on")
    with mesh_xy, timing.section("h_transform"):
        result = h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                             a_band_index=args.a_band, diagnostics=_diag_on,
                             band_start=int(wfn.nelec) - int(params["nval"]),
                             n_return_bands=n_return_bands)

    # Optional BSE interpolation handoff: fine-k wfns at coarse centroids.
    # Driven by ``get_centroids_fi`` + ``kgrid_fi`` + ``wfn_fi_{min,max}``;
    # see ``bandstructure.bse_setup.compute_wfns_fi`` for the contract.
    if params.get("get_centroids_fi", False):
        from .bse_setup import compute_wfns_fi
        b_min = int(params["wfn_fi_min"])
        b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])
        # SPLASH RADIUS OF THE f-SHOULDER, named by
        # ``2026-08-11-fifth-wall-is-the-f-transform-shoulder.md`` §7 and
        # audited here.  ``wfn_fi_max`` unset DEFAULTS to the full band count
        # — zero guard bands, i.e. exactly the configuration that row
        # convicts.  ``compute_wfns_fi``'s f-shoulder gate is what refuses;
        # this warns first, in the vocabulary of the deck key the user would
        # have to change, so the refusal is not the first news of it.
        _n_guard = int(ctilde.shape[1]) - b_max
        if _n_guard < 4:
            log(f"  [warn] wfn_fi_max={b_max} leaves only {_n_guard} guard "
                f"band(s) below the top of the htransform window "
                f"({int(ctilde.shape[1])} bands)"
                + (" — this is the ZERO-GUARD default (wfn_fi_max unset "
                   "means 'the whole window')" if not
                   int(params["wfn_fi_max"]) else "")
                + f".  f(eps) is identically zero at and above "
                f"max_k eps of the window's own top band, so the top of what "
                f"you are asking BACK may be an arbitrary direction out of "
                f"fH's null space.  Raise nband/ncond so the window extends "
                f"above wfn_fi_max; the f-shoulder gate decides.")
        with mesh_xy, timing.section("wfns_fi"):
            wfns_fi = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                kgrid_fi=params["kgrid_fi"],
                band_window_fi=(b_min, b_max),
                mesh_xy=mesh_xy, a_band_index=args.a_band,
                eigh_backend=eigh_backend,
                use_low_mem_eigh=use_low_mem_eigh, log_fn=log,
                distrib_la_batched_route=distrib_la_batched_route,
            )
        log(f"BSE setup: psi_rmu_Y={wfns_fi.psi_rmu_Y.shape} "
            f"P{wfns_fi.psi_rmu_Y.sharding.spec}, "
            f"psi_rmuT_X={wfns_fi.psi_rmuT_X.shape} "
            f"P{wfns_fi.psi_rmuT_X.sharding.spec}, "
            f"enk_full={wfns_fi.enk_full.shape}")

    if args.plot:
        plot_bands(result)

    # ── Writer gate ───────────────────────────────────────────────────
    # bandstructure.dat is the file downstream tooling and the regression
    # gate diff against ground truth; a NaN row silently changes the file
    # length rather than the exit code.
    sanity.check_finite("bandstructure.dat energies",
                        result['energies_sorted'], print_fn=log)

    # Rank-0 writer gate: at P>1 every process reaches this line with the
    # same (replicated) energies and used to write the SAME shared-FS file
    # concurrently — a race that can interleave partial writes.  Same idiom
    # as the gw_output/gw_init writers.
    output_dir = os.path.dirname(os.path.abspath(args.input))
    if jax.process_index() == 0:
        write_bands_to_file(
            os.path.join(output_dir, 'bandstructure.dat'),
            result['energies_sorted'],  # sorted & truncated to nb_keep, not raw eigenvalues
            kpath_data[0],
            kpath_data[1],
            band_start=result["band_start"],
            nb_fit=result["nb_fit"],
        )

    # ── Outputs barrier, then CLOSE THE LOADER EXPLICITLY ─────────────
    # The block above is rank-0-only, so without this barrier the other
    # ranks arrive at the collective close while rank 0 is still writing
    # ``bandstructure.dat``.  The barrier is what makes every rank enter
    # the close at the same point; the close is the load-bearing half.
    #
    # ``initialize_wfns`` hands back a MESH-AWARE ``WfnLoader``
    # (``setup_wfn_and_sym`` -> ``WfnLoader(wfn_file, mesh=mesh_xy)``, and
    # this driver passes ``mesh_xy=None`` so ``initialize_wfns`` builds the
    # mesh itself), so at P>1 the loader picks the phdf5 backend and owns a
    # ``SlabIO`` whose ``close()`` runs an UNCONDITIONAL COLLECTIVE barrier
    # (``file_io/_slab_io_ffi.py``, ``_barrier("slab_io_ffi_close_attrs")``).
    # Left to ``WfnLoader.__del__``, that collective fires whenever the
    # garbage collector happens to drop the object during interpreter
    # shutdown — a moment no two ranks agree on, and rank 0's object graph
    # differs from the others' because of the writer block above.
    #
    # This is the defect measured and cured in ``bse.exciton_bands`` at
    # ``b3813d8f`` (FIX_exciton_exit_hang.md): three ranks parked in
    # ``__del__`` -> ``SlabIO.close`` -> ``sync_global_devices`` while the
    # fourth had already reached ``ffi.io._atexit_close_all`` ->
    # ``H5Fclose`` -> ``MPI_Barrier``; two disjoint collective domains,
    # neither ever satisfied, the payload complete and every output written,
    # and the step holding its GPUs at 4x100% CPU until the allocation died.
    # That report's §6 named THIS driver as the one remaining sibling with
    # the identical shape.  ``close()`` is idempotent and nulls the handles,
    # so the later ``__del__`` becomes a no-op on every rank; at P=1 both
    # the barrier and the SlabIO collective are already no-ops.
    from common.collectives import barrier
    barrier("htransform.outputs_written")
    try:
        wfn.close()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [htransform] WfnLoader.close() failed "
              f"({type(exc).__name__}: {exc}); continuing to exit")

    summary = (f"HT complete: {result['nb_keep']} returned / "
               f"{result['nb_fit']} fit bands "
               f"({result['n_guard_bands']} guards), nk={result['nk_total']}, "
               f"fermi={result['fermi_energy']:.6f} Ry")
    if result['path_range'] is not None:
        summary += f", path range [{result['path_range'][0]:.6f}, {result['path_range'][1]:.6f}] Ry"
    print(summary)
    # Close the table against the PROCESS wall, not against ``main()``'s.
    # ``_pre_main`` is everything above this function: the module body's
    # ``initialize_communicator_stack()`` (env, jax.distributed, backend init,
    # mesh + clique warm-up) and the import storm under it.  It is decomposed
    # into rows from the numbers the startup call measured for itself, with
    # the remainder attributed to imports — the same shape as
    # ``gw_jax.py``'s epilogue, and for the same reason: recording the phases
    # AND the whole span would double-count and break the
    # "rows + (untimed) == wall" property that makes the table readable.
    _wall = _time.perf_counter() - _t_main
    if _pre_main is not None:
        _phases = {}
        try:
            _phases = dict(RUNTIME.facts.get("elapsed", {}) or {})
        except Exception:      # noqa: BLE001 — observability never kills a run
            _phases = {}
        for _phase, _secs in sorted(_phases.items()):
            if _phase != "total":
                timing.record(f"htransform.runtime_stack.{_phase}", float(_secs))
        timing.record("htransform.imports",
                      max(_pre_main - float(_phases.get("total", 0.0)), 0.0))
        _wall = _pre_main + _wall
    timing.report(title="--- Timing (seconds) ---", wall=_wall)
    return 0


if __name__ == "__main__":
    # ``runtime.finalize_process``, not a bare ``SystemExit``: the explicit
    # ``wfn.close()`` above removes the collective from ``__del__``, and this
    # removes the SECOND unordered collective — ``ffi.io._atexit_close_all``,
    # whose ``H5Fclose`` on the restart/zeta contexts is collective too and
    # otherwise runs at whatever point each rank's interpreter teardown
    # reaches it.  ``finalize_process`` runs the effects barrier, the
    # distributed shutdown and the atexit hooks in ONE stated order on every
    # rank and then ends with ``os._exit``, so GC-driven ``__del__``s at
    # shutdown never run at all.  Same pattern as ``bse.exciton_bands``
    # (``b3813d8f``) and ``gw.gw_jax``, the sibling that has never hung.
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
