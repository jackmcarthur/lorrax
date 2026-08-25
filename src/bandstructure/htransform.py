import os
import argparse
import numpy as np

# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  MUST run before this module's own
# `import jax`; idempotent, so importing htransform as a LIBRARY from an
# already-started driver (bse.exciton_bands does) returns the same stack.
from runtime import (
    debug_print,
    debug_print_enabled,
    initialize_communicator_stack,
    rank0_print,
)
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta
from common import timing
from common.units import RYD_TO_EV
from runtime.padding import round_up, spec_divisor
from common.wfn_transforms import get_enk_bandrange
from isdf.galerkin import (
    fit_galerkin_basis,
    validate_rank_multiplier,
)
# ``eigh_backend`` + ``use_low_mem_eigh`` are ONE axis with ONE resolver;
# this driver reads a raw params dict rather than a LorraxConfig, which is
# exactly the case that function exists for.
from gw.gw_config import (
    distrib_la_batched_route_choices,
    eigh_backend_choices,
    read_cohsex_input,
    resolve_distrib_la_batched_route,
    resolve_eigh_backend,
)
from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
# Q's free r axis is zero-padded through ``runtime.padding`` and split over
# the full mesh product.  ``common.staged_reshard`` owns the exact
# product-band → product-r exchange used to put streamed wavefunctions there.
from common.collectives import gather_to_host
from common.sharding_fit import padded_extent as _pad_to
from runtime.production_stream import ProductionStdout
from .production_report import HTransformProductionReport

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


# Per-device ceiling on one bounded whole-state real-space tile.  The
# randomized sketch, exact selected-state Gram and physical projection all use
# the canonical ``PsiGStore`` outer-r / inner-band stream; the shared planner
# chooses one carrier that bounds their selected/random rows and WFN transform
# workspace.  No full-r Galerkin basis is materialized.
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
    """Per-device ``LORRAX_GALERKIN_CHUNK_GIB`` stream budget in bytes.

    Blank/unset → the default; garbage REFUSES naming the variable
    (``gw_config.env_float`` refuse mode); non-positive values refuse too
    — a zero-byte accumulation budget is never what anyone meant.
    """
    from gw.gw_config import env_float
    gib = env_float("LORRAX_GALERKIN_CHUNK_GIB", 6.0, refuse=True)
    if gib <= 0.0:
        raise ValueError(
            f"LORRAX_GALERKIN_CHUNK_GIB={gib!r} must be > 0 (GiB budget "
            f"for one whole-state real-space tile; unset/blank = 6).")
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
    from file_io.centroids import validate_centroid_selection
    try:
        return validate_centroid_selection(selection, n_parent)
    except ValueError as exc:
        raise ValueError(f"htransform centroid subset: {exc}") from exc


def streaming_galerkin_solve(wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
                             band_range: tuple[int, int],
                             log_fn=None,
                             band_chunk_size: int = 64,
                             bispinor: bool = False,
                             rank_multiplier: float = 20.0,
                             qr_eps: float = 1.0e-3,
                             qrcp_seed: int = 0,
                             progress_fn=None, rank_record_fn=None):
    """Htransform policy adapter for :func:`isdf.galerkin.fit_galerkin_basis`."""
    rank_multiplier = resolve_galerkin_rank_multiplier(rank_multiplier)
    # The whole-state ledger describes allocations made *after this point*.
    # Compare it with the allocator budget still available to the fit, not
    # with the arena limit: the driver, WFN metadata and symmetry service are
    # already live.  ``get_device_memory_info`` is the public owner of that
    # policy and retains its standard 10% reserve for allocator fragmentation
    # and state materialised between planning and the first streamed slab.
    from common.gpu_utils import get_device_memory_info
    memory = get_device_memory_info()
    local_fit_budget = int(float(memory["budget_gb"]) * 1.0e9)
    # The carrier and r-chunk extents are shared static control flow.  BFC
    # residency is rank-local and asynchronous, so one rank choosing from its
    # own larger budget can send its peers into a different compile/loop
    # schedule.  Resolve one worst-rank budget through the process-collective
    # service before planning any static extent.
    from common.collectives import all_gather_processes
    process_budgets = np.asarray(
        all_gather_processes(np.asarray(local_fit_budget, dtype=np.int64)),
        dtype=np.int64,
    )
    if process_budgets.size == 0 or np.any(process_budgets <= 0):
        raise RuntimeError(
            "htransform live-capacity gather returned no positive budget")
    device_fit_budget = float(np.min(process_budgets))
    if log_fn is not None:
        log_fn(
            "  Whole-state live fit budget: "
            f"{device_fit_budget/2**30:.2f} GiB/device from "
            f"worst-rank reserve (this rank "
            f"{local_fit_budget/2**30:.2f} GiB; "
            f"{float(memory['available_gb'])*1.0e9/2**30:.2f} GiB available "
            f"({memory['source']}); the allocator limit is not reusable "
            "capacity while earlier driver state remains live)")
    basis = fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy, band_range,
        log_fn=log_fn,
        band_chunk_size=band_chunk_size,
        bispinor=bispinor,
        rank_multiplier=rank_multiplier,
        qr_eps=qr_eps,
        qrcp_seed=qrcp_seed,
        q_tile_budget=resolve_galerkin_chunk_bytes(),
        device_pool_limit=device_fit_budget,
        extra_rank_pad=resolve_extra_rank_pad(),
        progress_fn=progress_fn,
        rank_record_fn=rank_record_fn,
    )
    return basis


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


def outer_r_shell_mask(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Mask the outer lattice-vector shell of the interpolation supercell.

    Each periodic direction with more than one point contributes its largest
    represented ``|R_i|``.  The definition works for odd and even meshes and
    excludes singleton/non-periodic directions.  It is host-only metadata;
    the large ``f(H)_R`` tensor remains sharded.
    """
    grid = np.asarray(tuple(int(v) for v in kgrid), dtype=np.int64)
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(f"outer_r_shell_mask: invalid k grid {tuple(grid)}")
    r_grid = np.asarray(build_R_grid_np(tuple(grid)), dtype=np.int64)
    edge = grid // 2
    active = grid > 1
    return np.any((np.abs(r_grid) == edge[None, :]) & active[None, :], axis=1)


@jax.jit
def _fh_locality_metrics(fh_r: jax.Array,
                         outer_shell: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Two scale-free locality metrics without gathering ``f(H)_R``.

    ``fh_r`` is face-sharded on its matrix axes.  JAX therefore reduces each
    local tile first and communicates only the resulting R-vector/scalars.
    """
    norm_sq_r = jnp.sum(jnp.real(fh_r * jnp.conj(fh_r)), axis=(1, 2))
    shell = jnp.asarray(outer_shell, dtype=bool)
    total_sq = jnp.sum(norm_sq_r)
    shell_sq = jnp.sum(jnp.where(shell, norm_sq_r, 0.0))
    shell_max_sq = jnp.max(jnp.where(shell, norm_sq_r, 0.0))
    r0_sq = norm_sq_r[0]
    shell_fraction = jnp.where(
        total_sq > 0.0, jnp.sqrt(shell_sq / total_sq), 0.0)
    shell_max_over_r0 = jnp.where(
        r0_sq > 0.0, jnp.sqrt(shell_max_sq / r0_sq), 0.0)
    return shell_fraction, shell_max_over_r0


def build_fH_R(ctilde: jax.Array, enk_sigma: jax.Array,
               kgrid_co: tuple[int, int, int], mesh_xy: Mesh,
               *, a_band_index: int | None = None,
               log_fn=None, quality_record_fn=None):
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
        quality_record_fn: optional callback receiving the shared
                   row-isometry and real-space-locality receipt.  The
                   reduction preserves the matrix sharding and transfers only
                   scalars to the host.

    Returns:
        fH_k:    (nk_co, rank, rank), sharded P(None, 'x', 'y').
        fH_R:    (nk_co, rank, rank), sharded P(None, 'x', 'y') (lattice-R index).
        params:  (a_f, n_f, shift) — for ``newton_inv`` on the eigvals of fH_q.
        f_eps:   (nb, nk_co) f-transformed eigenvalues, replicated.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    f_eps, a_f, n_f, shift = f_transform_eigs(enk_sigma, a_band_index=a_band_index)
    log_fn(f"  f-transform: a={a_f * RYD_TO_EV:.5f} eV, n={n_f:.2f}, "
           f"shift={shift * RYD_TO_EV:.5f} eV"
           + (f" (a from band {a_band_index})" if a_band_index is not None else ""))

    flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    spec_3d = P(None, None, None, 'x', 'y')
    rep_ = NamedSharding(mesh_xy, P())
    # 'backward' = 1/N normalisation in IFFT — matches Σ_R e^{-2πik·R} fH_R = fH_k.
    local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid_co, spec_3d, norm='backward')
    local_fftn = make_flat_k_fftn(mesh_xy, kgrid_co, spec_3d, norm='backward')

    # ``f_eps.T`` at the call site was its own eager ``transpose`` module;
    # transposing inside costs nothing and removes one compile (exact — a
    # transpose is a permutation of the same f64 values).
    @partial(jax.jit,
             out_shardings=(flat_xy, flat_xy, rep_, rep_, rep_, rep_))
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

        # The rank-by-rank fH eigensolve is not needed to certify its coarse
        # spectrum.  If A = sqrt(-f(eps)) C, the nonzero eigenvalues of
        # -A^H A are exactly those of the much smaller -A A^H.  This checks
        # every coarse k in the nb-by-nb state space while the weighted rows
        # are already live, rather than compiling a rank-by-rank Gamma-only
        # eigensolve beside the production batched eigensolver.
        small_fH = -jnp.einsum(
            'kim,kjm->kij', weighted, jnp.conj(weighted), optimize=True)
        small_fH = 0.5 * (small_fH + jnp.swapaxes(small_fH, -1, -2).conj())
        recovered_f = jax.vmap(jnp.linalg.eigvalsh)(small_fH)
        expected_f = jnp.sort(f_eps_T, axis=1)
        f_recovery_error = jnp.max(jnp.abs(recovered_f - expected_f))
        recovered_e, inverse_residual = jax.vmap(
            lambda row: newton_inv(a_f, n_f, shift, row.real))(recovered_f)
        energy_recovery_error = jnp.max(
            jnp.abs(jnp.sort(recovered_e, axis=1)
                    - jnp.sort(enk_sigma.T, axis=1)))

        # Use the paired canonical flat-k service for the complete inverse
        # transform.  A one-point explicit Fourier sum can certify Gamma while
        # still missing an ordering, normalization or non-Gamma defect.
        fft_roundtrip_error = jnp.max(jnp.abs(local_fftn(fH_R) - fH_k))
        return (fH_k, fH_R, fft_roundtrip_error, f_recovery_error,
                energy_recovery_error, jnp.max(inverse_residual))

    # Projection receipt, not a rank gate.  The published whole-state QRCP
    # basis is deliberately approximate, so C C^H need not be the identity.
    # Applying a per-k Löwdin map would change the one shared alpha gauge and
    # is therefore forbidden.  Accuracy is decided directly from recovered
    # coarse-grid energies/wavefunctions and the independent fine-QE oracle.
    @partial(jax.jit, out_shardings=rep_)
    def _ortho_all_k(c):
        nb_ = c.shape[1]
        G = jnp.einsum('kim,kjm->kij', c, jnp.conj(c), optimize=True)
        return jnp.max(jnp.abs(G - jnp.eye(nb_, dtype=G.dtype)[None]))

    _projection_defect = float(_ortho_all_k(ctilde))
    log_fn(
        f"  [receipt] whole-state projection over all "
        f"{int(ctilde.shape[0])} coarse k: max|C Cᴴ − I|="
        f"{_projection_defect:.3e} (diagnostic only; no per-k gauge repair)")

    (fH_k, fH_R, fft_roundtrip_error, f_recovery_error,
     energy_recovery_error, inverse_residual) = _build(ctilde, f_eps)
    log_fn(
        f"  [receipt] all-{int(ctilde.shape[0])}-coarse-k canonical FFT "
        f"round-trip max|FFT(IFFT(fH))-fH|="
        f"{float(fft_roundtrip_error):.3e}")
    log_fn(
        f"  [receipt] all-coarse transformed spectrum: "
        f"max|eig(-A A^H)-f(eps)|={float(f_recovery_error):.3e} Ry; "
        f"max recovered-energy residual="
        f"{float(energy_recovery_error):.3e} Ry; Newton residual="
        f"{float(inverse_residual):.3e} Ry")
    require_newton_converged(
        float(inverse_residual), where="build_fH_R all-coarse recovery")
    if quality_record_fn is not None:
        from time import perf_counter as _quality_clock
        _quality_t0 = _quality_clock()
        _outer_shell = outer_r_shell_mask(kgrid_co)
        _shell_fraction, _shell_max_over_r0 = jax.device_get(
            _fh_locality_metrics(fH_R, jnp.asarray(_outer_shell)))
        quality_record_fn({
            "row_isometry_max": _projection_defect,
            "outer_shell_l2_fraction": float(_shell_fraction),
            "outer_shell_max_over_r0": float(_shell_max_over_r0),
            "outer_shell_vectors": int(np.count_nonzero(_outer_shell)),
            "r_vectors": int(_outer_shell.size),
            "locality_wall_seconds": float(_quality_clock() - _quality_t0),
        })
    return fH_k, fH_R, (a_f, n_f, shift), f_eps


NEWTON_RESIDUAL_MAX = 1.0e-12


def newton_inv(a: float, n: float, shift: float, y: jax.Array,
               max_iter: int = 50) -> tuple[jax.Array, jax.Array]:
    """Invert ``f`` and return ``(x, max|f(x)-y|)``.

    The 50-step cap, ``a/2`` step clip, exact ``df != 0`` condition and
    1e-12 caller refusal are the archived implementation's convergence
    contract.  Returning the residual keeps one inverse implementation while
    letting each JITted caller enforce that contract at its existing host seam.
    """
    dxmax = a / 2.0
    x0 = y + shift  # Fortran initial guess: x = y + s

    def body_fun(_, x_curr):
        res = fun(a, n, shift, x_curr) - y
        df_val = dfun(a, n, shift, x_curr)
        dx = jnp.where(df_val != 0.0, -res / df_val, 0.0)
        dx = jnp.clip(dx, -dxmax, dxmax)
        return x_curr + dx

    x = lax.fori_loop(0, max_iter, body_fun, x0)
    residual = jnp.max(jnp.abs(fun(a, n, shift, x) - y))
    return x, residual


def require_newton_converged(residual: float, *, where: str) -> None:
    """Host refusal shared by the two htransform consumers."""
    if not np.isfinite(residual) or residual > NEWTON_RESIDUAL_MAX:
        raise ValueError(
            f"{where}: f-transform Newton inverse did not converge: "
            f"max|f(x)-y|={residual:.6e} Ry exceeds the archived "
            f"{NEWTON_RESIDUAL_MAX:.1e}-Ry residual cap after 50 steps")


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


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


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
                    mesh_xy: Mesh | None = None,
                    n_guard_bands: int = 0, centroid_subset_idx=None,
                    progress_fn=None, centroid_record_fn=None,
                    rank_record_fn=None, wfn_sym=None, *,
                    require_all_occupied: bool = False):
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
    from file_io.centroids import load_centroid_basis

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    # Build the mesh up front so the loader is mesh-aware (sharded read).
    if mesh_xy is None:
        mesh_xy = _build_mesh_xy()
    wfn_file = _resolve(params["wfn_file"])
    if wfn_sym is None:
        wfn, sym = setup_wfn_and_sym(wfn_file, mesh_xy=mesh_xy)
    else:
        wfn, sym = wfn_sym
    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    if require_all_occupied and nval != int(wfn.nelec):
        raise ValueError(
            "standalone htransform output requires every occupied band in "
            "its one contiguous Hamiltonian: "
            f"nval={nval}, occupied bands={int(wfn.nelec)}, so absolute "
            f"bands [{int(wfn.nelec)-nval},{int(wfn.nelec)}) would be "
            "omitted. A lower spectral boundary can be recovered exactly at "
            "sampled k while producing uncontrolled off-grid ringing. Set "
            f"nval={int(wfn.nelec)}. Internal BSE interpolation retains its "
            "explicit window contract and does not use this standalone gate.")
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    centroid_basis = load_centroid_basis(
        centroid_path, tuple(int(x) for x in wfn.fft_grid), sym=sym,
        selection=centroid_subset_idx)
    centroid_indices = centroid_basis.centroid_indices
    n_rmu = centroid_basis.n_rmu
    if centroid_subset_idx is not None:
        log_fn(
            f"  [reduced-galerkin] centroid fit born in the downfold basis: "
            f"ordered parent subset {centroid_basis.source_n_rmu} -> "
            f"{n_rmu} rows.  No "
            "parent-width B_at_mu or projected wavefunction is formed.")
    if centroid_record_fn is not None:
        centroid_record_fn(centroid_basis)

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
        basis = streaming_galerkin_solve(
            wfn, sym, meta, centroid_indices, mesh_xy, band_range,
            log_fn=log_fn, bispinor=bispinor,
            rank_multiplier=params.get("htransform_rank_multiplier", 20.0),
            qr_eps=params.get("htransform_qr_eps", 1.0e-3),
            qrcp_seed=params.get("htransform_qrcp_seed", 0),
            progress_fn=progress_fn,
            rank_record_fn=rank_record_fn,
        )
    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, "
           f"nb={band_range[1]-band_range[0]}, rank={basis.rank_carrier}")
    return wfn, sym, meta, mesh_xy, basis, enk_sigma


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


def h_transform(meta, ctilde, enk_sigma, wfn, kpath_data, log_fn, mesh_xy: Mesh,
                a_band_index: int | None = None,
                band_start: int = 0, n_return_bands: int | None = None,
                progress_fn=None, quality_record_fn=None):
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
        coeffs, enk_sigma, kgrid, mesh_xy, a_band_index=a_band_index,
        log_fn=log_fn, quality_record_fn=quality_record_fn)
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
    # The path solve consumes ``fH_R`` alone.  Release the coarse-grid image
    # before its q-batched matrices are allocated (576 MiB at the reference
    # 64 x 768 x 768 complex128 shape).
    del fH_k

    _t0 = _perf()                                          # instrument:
    # The selected-state Cholesky basis is orthonormal by construction.  There
    # is no separately mutable metric: carrying a dense identity S was a
    # compatibility object that cost rank² storage and enabled a dead
    # generalized-eigenproblem branch beside bse_setup's canonical identity
    # eigensolve.
    log_fn(f"  [route] fH_q metric: identity (selected-state Galerkin basis; "
           f"no dense {rank}x{rank} identity object)")

    R_grid = jnp.asarray(build_R_grid_np(kgrid))
    jax.block_until_ready(R_grid)                            # instrument:
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
    def _kpath_batch(batch_k, fH_R):
        # batch_k: (bs, 3) replicated; fH_R: (nk, rank, rank) at P(None,'x','y');
        # The einsum contracts over the replicated R axis only, so it is local
        # on each (i, j) tile.
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

        return jax.vmap(jnp.linalg.eigvalsh)(mat)

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
        from common.progress import LoopProgress
        _n_batches = int(nq_padded // batch_size)
        _path_progress = LoopProgress(
            _n_batches, progress_fn or (lambda *_: None),
            title="Hamiltonian interpolation along k path",
            item_name="k batch", max_updates=min(_n_batches, 12),
            enabled=progress_fn is not None).start()
        for i in range(0, nq_padded, batch_size):
            batch_eigs = _kpath_batch(wrapped_k[i:i+batch_size], fH_R)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)
            _path_progress.step()
        _path_progress.finish()
        timing.record("ht.kpath_loop", _perf() - _t0,      # instrument:
                      count=len(lambda_q_list))            # instrument:

        # Bundle concat + slice + vmap(newton_inv) + sort into ONE jit so the
        # post-loop processing emits one compile rather than 4 (concatenate,
        # sort, gather, vmap-newton).
        # ``out_shardings=(rep, rep, rep)`` is a CORRECTNESS requirement at
        # P>1, not
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
        # Replication is the right answer and not merely the safe one: the two
        # arrays are consumed on the host by every process immediately below
        # (``np.asarray``, ``np.max``, the writer gate), and the third output is
        # one scalar Newton receipt.  They are small —
        # ``energies`` is (nq, rank) f64 and ``energies_sorted`` (nq, nb_keep),
        # 854 KB and 13 KB at nq=139 / rank 768, against the (nk, rank, rank)
        # 576 MiB arrays this driver already holds.  Values are untouched:
        # ``out_shardings`` moves data, it does not compute.
        # Gate: ``tests/test_htransform_post_kpath_sharding.py``.
        @partial(jax.jit, static_argnames=('nq', 'nb_keep'),
                 out_shardings=(rep, rep, rep))
        def _post_kpath(batches, nq, nb_keep):
            lambda_q = jnp.concatenate(batches, axis=0)[:nq]
            energies, inverse_residual = jax.vmap(
                lambda row: newton_inv(a_f, n_f, shift, row.real))(lambda_q)
            energies_sorted = jnp.sort(energies, axis=1)[:, :nb_keep]
            return energies, energies_sorted, jnp.max(inverse_residual)

        _t0 = _perf()                                      # instrument:
        energies_on_path, energies_sorted_jax, inverse_residual = _post_kpath(
            tuple(lambda_q_list), int(nq), int(nb_keep))
        require_newton_converged(
            float(inverse_residual), where="htransform path")
        # Report the sharding that was ACTUALLY produced, not the one asked
        # for.  Anything other than ``P()`` here is the non-addressable-fetch
        # crash of PROFILE_htransform_exciton §1.5 waiting for a P>1 run with
        # nq divisible by the device count.  ``_post_kpath`` now pins
        # ``out_shardings=(rep, rep, rep)``, so this line should always read
        # ``P()``;
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
        _k_np = np.asarray(jax.device_get(wrapped_k))[:nq]
        if not gamma_positions:
            # Label-less path: nearest-to-Γ point, EXCLUDING the batch pad
            # rows (they are exact zeros and would win the tie on any path
            # that does not start at Γ).  Host numpy — two values for a log
            # line and the plot markers, no reason for two eager XLA modules.
            gamma_positions = [int(np.argmin(np.linalg.norm(_k_np, axis=1)))]
        _gamma_position = int(gamma_positions[0])
        _gamma_is_exact = bool(
            np.linalg.norm(_k_np[_gamma_position]) <= 1.0e-10)
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
        log_fn(
            f"Path energy range: {path_range[0] * RYD_TO_EV:.5f} to "
            f"{path_range[1] * RYD_TO_EV:.5f} eV (VBM@0)")
        delta = ((energies_sorted[gamma_positions[0]] - gamma_exact)
                 * RYD_TO_EV * 1000.0)
        log_fn(("Γ Δε (meV): " if _gamma_is_exact else
                "nearest-path-point Δε vs Γ source (meV): ")
               + ", ".join(f"{d:+.2f}" for d in delta[:6]))
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
        "f_transform": {
            "a_ry": float(a_f),
            "n": float(n_f),
            "shift_ry": float(shift),
            "scale_band_local": int(
                states - 1 if a_band_index is None else a_band_index),
            "shoulder_band_local": int(states - 1),
        },
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
    energies_ev = np.asarray(energies_sorted) * RYD_TO_EV
    gamma_exact_ev = (None if gamma_exact is None else
                      np.asarray(gamma_exact) * RYD_TO_EV)
    for band in range(nb_keep):
        ax.plot(x_path, energies_ev[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact_ev is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact_ev, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_ev[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy * RYD_TO_EV, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (eV)')
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
    energies = gather_to_host(energies_on_path) * RYD_TO_EV
    kpoints = gather_to_host(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy_eV\n')
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
    parser.add_argument("-o", "--output-file", default="bandstructure.dat",
                        help="Interpolated band table (relative paths are "
                             "resolved beside the input deck)")
    parser.add_argument("--report-file", default="htransform.out",
                        help="Human-readable calculation report (relative "
                             "paths are resolved beside the input deck)")
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
    input_dir = os.path.dirname(os.path.abspath(args.input))

    def _output_path(value: str) -> str:
        return value if os.path.isabs(value) else os.path.join(input_dir, value)

    output_path = _output_path(args.output_file)
    report_path = _output_path(args.report_file)
    _debug = debug_print_enabled()
    report = HTransformProductionReport(
        report_path, runtime=RUNTIME, debug=_debug, stdout=rank0_print)
    production_stdout = ProductionStdout(
        debug=_debug, rank=RUNTIME.process_index,
        warning_fn=report.legacy_print)
    production_stdout.install()
    report.stdout = rank0_print if _debug else production_stdout.emit
    log = report.legacy_print
    _energy_source = (f"quasiparticle energies from {os.path.basename(args.eqp_file)}"
                      if args.eqp_file else "DFT eigenvalues from the WFN")
    report.begin(input_file=args.input, output_file=output_path,
                 energy_source=_energy_source)
    report.architecture()

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

    # Resolve the concrete fine-k plan before the first setup/progress line.
    # Whole-state QRCP owns basis selection and has no Gram eigensolve.
    mesh_xy = _build_mesh_xy()
    _wfn_path = params["wfn_file"]
    if not os.path.isabs(_wfn_path):
        _wfn_path = os.path.join(input_dir, _wfn_path)
    wfn, sym = setup_wfn_and_sym(_wfn_path, mesh_xy=mesh_xy)
    from distrib_la import plan as _linalg_plan
    _fine_enabled = bool(params.get("get_centroids_fi", False))
    _fine_plan = (_linalg_plan(
        "eigh", mesh_xy, backend=eigh_backend, n=None,
        batched_route=distrib_la_batched_route)
        if _fine_enabled else None)
    report.environment(
        params=params, wfn=wfn,
        fine_plan=_fine_plan, fine_enabled=_fine_enabled)

    from common import sanity

    from common.progress import LoopProgress
    _setup_progress = LoopProgress(
        1, report.progress, title="wavefunction and Galerkin setup",
        item_name="stage", max_updates=1).start()
    _centroid_records = []
    _rank_records = []
    with timing.section("initialize_wfns"):
        wfn, sym, meta, mesh_xy, basis, enk_sigma = initialize_wfns(
            args.input, params, log, args.eqp_file,
            mesh_xy=mesh_xy, wfn_sym=(wfn, sym),
            n_guard_bands=args.guard_bands, progress_fn=report.progress,
            centroid_record_fn=_centroid_records.append,
            rank_record_fn=_rank_records.append,
            require_all_occupied=True)
    _setup_progress.step()
    _setup_progress.finish()
    ctilde, B_at_mu = basis.ctilde, basis.basis_at_nodes
    # ── Galerkin-input gate ───────────────────────────────────────────
    # ``ctilde`` is the compact Galerkin coefficient table and ``enk_sigma``
    # is the band energies the whole
    # interpolation is anchored to — including, when ``--eqp-file`` is
    # given, energies read from a GW run that may itself have produced
    # garbage.  A −136 eV QP energy fed into htransform yields a
    # bandstructure.dat that is numerically finite, plots fine, and is
    # wrong.  Bracket it here, where the file name is still in scope.
    sanity.check_finite("htransform ctilde", ctilde, print_fn=log)
    sanity.check_finite("htransform band energies", enk_sigma, print_fn=log)
    # Bandwidth, not absolute energy: the zero of a pseudopotential
    # eigenvalue is convention-dependent, but the *spread* of a Σ-window
    # band set is not — 272 eV is far wider than any real
    # semicore-to-conduction window and so only fires on gross
    # corruption.  A subtler check (comparing --eqp-file energies against
    # the DFT ones they replace) is proposed but not implemented here;
    # see the workstream-O report.
    _enk = np.asarray(jax.device_get(enk_sigma), dtype=np.float64)
    if _enk.size:
        _spread = float(_enk.max() - _enk.min()) * RYD_TO_EV
        log(f"  E_nk spread: {_spread:.4f} eV over {_enk.size} states")
        sanity.check_in_range(
            "htransform E_nk bandwidth", np.array([_spread]),
            0.0, 20.0 * RYD_TO_EV, unit="eV", print_fn=log)

    kpath_data = initialize_kpath(wfn, params)
    if len(_centroid_records) != 1:
        raise RuntimeError(
            "htransform initialize_wfns did not return exactly one centroid "
            f"closure record; got {len(_centroid_records)}.")
    if len(_rank_records) != 1:
        raise RuntimeError(
            "htransform initialize_wfns did not return exactly one Galerkin "
            f"rank record; got {len(_rank_records)}.")
    report.sampling(
        wfn=wfn, sym=sym, centroids=_centroid_records[0])
    _transform_progress = LoopProgress(
        1, report.progress, title="fH construction and path solution",
        item_name="stage", max_updates=1).start()
    _quality_records = []
    with mesh_xy, timing.section("h_transform"):
        result = h_transform(meta, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                             a_band_index=args.a_band,
                             band_start=int(wfn.nelec) - int(params["nval"]),
                             n_return_bands=n_return_bands,
                             progress_fn=report.progress,
                             quality_record_fn=_quality_records.append)
    _transform_progress.step()
    _transform_progress.finish()

    _centroid_path = params.get("centroids_file", "centroids_frac.txt")
    _centroid_path = (_centroid_path if os.path.isabs(_centroid_path) else
                      os.path.join(input_dir, _centroid_path))
    report.interpolation_space(
        params=params, wfn=wfn, meta=meta, result=result,
        enk_sigma_ry=enk_sigma, ctilde=ctilde,
        centroid_file=_centroid_path, energy_source=_energy_source,
        centroids=_centroid_records[0])
    report.spectral_compression(_rank_records[0])
    if len(_quality_records) != 1:
        raise RuntimeError(
            "htransform returned "
            f"{len(_quality_records)} interpolation-quality receipts; "
            "expected one")
    report.htransform_quality(_quality_records[0])
    report.path_summary(result=result)

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
    sanity.check_finite(f"{os.path.basename(output_path)} energies",
                        result['energies_sorted'], print_fn=log)

    # Rank-0 writer gate: at P>1 every process reaches this line with the
    # same (replicated) energies and used to write the SAME shared-FS file
    # concurrently — a race that can interleave partial writes.  Same idiom
    # as the gw_output/gw_init writers.
    _write_progress = LoopProgress(
        1, report.progress, title="interpolated-band output",
        item_name="file", max_updates=1).start()
    if jax.process_index() == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_bands_to_file(
            output_path,
            result['energies_sorted'],  # sorted & truncated to nb_keep, not raw eigenvalues
            kpath_data[0],
            kpath_data[1],
            band_start=result["band_start"],
            nb_fit=result["nb_fit"],
        )
    _write_progress.step()
    _write_progress.finish()

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
        log(f"WARNING: WfnLoader.close() failed "
            f"({type(exc).__name__}: {exc}); continuing to exit")
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
    if _debug:
        timing.report(print_fn=log, title="--- Timing (seconds) ---", wall=_wall)

    _wfn_path = params["wfn_file"]
    _wfn_path = (_wfn_path if os.path.isabs(_wfn_path) else
                 os.path.join(input_dir, _wfn_path))
    _file_rows = [
        ("input deck", "read", args.input),
        ("DFT wavefunctions", "read", _wfn_path),
        ("ISDF centroids", "read", _centroid_path),
    ]
    if args.eqp_file:
        _eqp_path = (args.eqp_file if os.path.isabs(args.eqp_file) else
                     os.path.join(input_dir, args.eqp_file))
        _file_rows.append(("QP energies", "read", _eqp_path))
    _file_rows.extend([
        ("interpolated bands",
         "written" if os.path.exists(output_path) else "absent", output_path),
        ("calculation report", "written", report.path),
    ])
    report.timings(timing.records(), wall=_wall)
    report.warnings()
    report.files(_file_rows)
    report.finish()
    production_stdout.close()
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
