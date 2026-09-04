"""
solvers/davidson.py — Block Davidson iterative eigensolver (shape-agnostic).

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
callables for the matvec, preconditioner, and initial guess.
No DFT or physics knowledge — works for any Hermitian eigenproblem.

Shape convention
----------------
The state vector batch ``V`` has the form ``(m, *trailing)`` where ``m`` is
the batch axis (number of subspace vectors) and ``trailing`` is whatever
shape encodes one vector — flat ``(dim,)``, plane-wave ``(n_channels, dim)``,
exciton ``(n_cond, n_val, nk)``, or anything else. All internal contractions
are written with numpy/JAX einsum ellipsis (``'m...,n...->mn'``,
``'mn,m...->n...'``) so a single implementation supports every layout.

The trailing axes may be sharded; Davidson never reshapes them, never
flattens them, and never gathers them. The Ritz-vector reconstruction
``X = V x`` uses ``jnp.einsum('mn,m...->n...', x, V)`` whose output
inherits the sharding pattern of ``V``.

Algorithm (after QE cegterg.f90):
  1. init_fn builds the starting subspace
  2. Each iteration: Gram projection → generalized eigh via Cholesky →
     Ritz vectors → residuals → precond_fn → expand
  3. Fixed block-size expansion (avoids JIT recompilation)
  4. Restart to Ritz vectors when subspace exceeds m_max

Usage
-----
    from solvers.davidson import davidson
    eigenvalues, eigenvectors = davidson(
        apply_H, n_eig=12, precond_fn=precond, init_fn=init,
    )
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import gather_to_host


# ═══════════════════════════════════════════════════════════════════════
#  Instrumentation (BSE perf campaign, 2026-08-08)
# ═══════════════════════════════════════════════════════════════════════
#  These counters are incremented INSIDE jit bodies.  A jit body runs on
#  host exactly once per TRACE, so one increment == one trace of that
#  program at that shape signature.  That is the instrument a fixed-shape
#  claim has to be proven against: ``compile_cache_stats()['compiles']``
#  reads 0 on any run whose persistent cache is warm, no matter how many
#  distinct programs the loop dispatches, so it can only ever prove the
#  cache works -- never that the shapes are fixed.
TRACE_COUNTS: dict = {}
MATVEC_APPLICATIONS = [0]      # count of VECTORS H has been applied to
LAST_RUN: dict = {}            # per-iteration history of the last davidson()


def _tally(name, *shapes) -> None:
    key = (name,) + tuple(str(t) for t in shapes)
    TRACE_COUNTS[key] = TRACE_COUNTS.get(key, 0) + 1


def reset_instrumentation() -> None:
    TRACE_COUNTS.clear()
    MATVEC_APPLICATIONS[0] = 0
    LAST_RUN.clear()


def instrumentation_summary() -> str:
    lines = [f"[dav-instr] matvec applications (vectors): "
             f"{MATVEC_APPLICATIONS[0]}",
             f"[dav-instr] distinct traced programs: {len(TRACE_COUNTS)}"]
    for k in sorted(TRACE_COUNTS, key=str):
        lines.append(f"[dav-instr]   {k} -> {TRACE_COUNTS[k]} trace(s)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Subspace cap
# ═══════════════════════════════════════════════════════════════════════
#: ``m_max = DEFAULT_M_MAX_FACTOR * n_eig`` is the restart cap.
#:
#: This was 4 and 4 is a defect-shaped default, not merely a conservative one.
#: On the Si 4x4x4 record deck (post exchange-conjugation, band-edge cluster
#: 0.0177 meV inside a 7.8 eV spectrum) the shipped 4*n_eig **cannot reach
#: 10 ueV at any budget**: it stalled at 30.8 ueV after 2420 matvec
#: applications, where unrestarted Lanczos reaches 2.9 ueV in 265.  The cause
#: is structural -- with m_max = 4*n_eig the subspace runs 20, 40, 60, 80 and
#: then hard-restarts to 20, so a restart fires every FOUR iterations and
#: discards all accumulated Krylov information.  The subspace never exceeds 80
#: on a problem needing ~265 dimensions to resolve its band edge.
#:
#: Measured applications to target (0 missed), block = n_eig, eps = 1e-3:
#:
#:     m_max      1 meV    10 ueV   floor      vectors stored (V + HV)
#:     4n           560      2360   6.54 ueV       160
#:     6n           360      1320   0.24 ueV       240
#:     8n           200       480   0.00 ueV       320
#:     10n          180       240   0.01 ueV       400   <- knee
#:     12n          180       240   0.01 ueV       480
#:     16n          180       240   0.00 ueV       640
#:
#: 10 is the knee: 12n/16n/20n are identical on every column and cost more
#: memory.  Davidson stores BOTH ``V`` and ``HV``, so the memory is
#: ``2 * m_max`` vectors -- 20*n_eig, which at n_eig = 20 is 400 copies of one
#: trial vector.  That is cheap at bse_dim = 1024 and is NOT cheap at
#: production bse_dim; DAVIDSON_COMPETITIVE.md prices the crossover.
DEFAULT_M_MAX_FACTOR = 10


# ═══════════════════════════════════════════════════════════════════════
#  Conditioning policy — why every threshold below is RELATIVE
# ═══════════════════════════════════════════════════════════════════════
#
# This solver used to regularise every Cholesky with an ABSOLUTE ``1e-12 * I``
# and then invert the factor with ``jnp.linalg.inv``.  That pair is the reason
# Davidson diverged above n_eig ≈ 50.  The mechanism, instrumented end to end:
#
#   1. Some roots converge.  Their residuals fall to ~1e-9, so the
#      preconditioned corrections ``P = R / (ΔE − λ + ε)`` are near-zero.
#   2. CGS2 against V removes what little signal is left; those columns of P
#      are pure round-off, ‖p‖ ~ 1e-16·‖R‖.
#   3. ``S_P = Pᴴ P`` for those columns is ~1e-30, so the ABSOLUTE 1e-12
#      regularisation DOMINATES S_P.  The Cholesky factor of (noise + 1e-12·I)
#      is ~1e-6·I on those directions, and ``L⁻¹`` therefore rescales the noise
#      UP to unit norm instead of rejecting it.  A regularisation meant as a
#      floor became an amplifier.
#   4. Those unit-norm noise columns are appended to V.  V is no longer
#      orthonormal in any useful sense: measured ``min eig(Sc)`` collapses
#      1.0 → 1.17e-11 → 2.5e-15 over ~50 iterations.
#   5. In ``_generalized_eigh`` the same absolute floor now sits *below*
#      Sc's smallest eigenvalue, ``inv(L)`` amplifies by ~1e6, and
#      ``C = L⁻¹ H L⁻ᴴ`` by ~1e12.  The Ritz values explode.
#      MEASURED offline against the stored dense Si BSE matrix:
#      n_eig=100 → e₁ = −1.66e7 eV; n_eig=200 → −1.83e10 eV.  n_eig=20 self-
#      destructs by iteration 56 too — it only looks clean because it
#      converges and exits first.
#
# NOT the cause, and the falsifier was built and run: clamping the
# preconditioner denominator (``|ΔE − λ + ε| ≥ 0.05 Ry``) alone STILL diverges
# (e₁ = −1.35e9 eV).  Fixing the denominator is not a fix.
#
# The policy here is therefore:
#   * every Cholesky floor scales with the matrix it regularises
#     (``_chol_floor``: 1e-12 · tr(S)/m, the mean diagonal),
#   * the subspace expansion is RANK-REVEALING (``_whiten_rank_revealing``):
#     directions whose Gram eigenvalue is below 1e-10 · λ_max are *dropped*,
#     not rescaled, so noise never enters V at all,
#   * no Cholesky factor is inverted with the general ``jnp.linalg.inv``;
#     the small (m, m) reductions use triangular solves.
#
# MEASURED, n_eig=100, 60 iterations, stored dense Si BSE matrix:
#     as shipped                          diverges
#     relative Cholesky floor alone       converges, 2.48e-9 meV
#     rank-revealing drop                 converges, 2.49e-9 meV, fewest matvecs
# ═══════════════════════════════════════════════════════════════════════

#: Cholesky floor as a fraction of the matrix's own mean diagonal.
_CHOL_REL_EPS = 1e-12
#: Gram eigenvalues below this fraction of λ_max are not real directions.
_RANK_DROP_RTOL = 1e-10
#: Absolute underflow guard — only ever used to keep a divide finite.
_TINY = 1e-300


def _chol_floor(S):
    """Scale-aware Cholesky regularisation for a Hermitian PSD Gram matrix.

    ``1e-12 · tr(S)/m`` — a fraction of the matrix's own mean diagonal, never
    an absolute constant.  See the conditioning-policy block above for what
    the absolute version did.  Clamped at ``_TINY`` so an all-zero S still
    yields a positive floor rather than a Cholesky of a singular matrix.
    """
    m = S.shape[0]
    scale = jnp.real(jnp.trace(S)) / m
    return _CHOL_REL_EPS * jnp.maximum(scale, _TINY)


def _whiten_rank_revealing(S, P):
    """Whiten P by its Gram matrix S, dropping numerically null directions.

    Returns ``(P_w, rank)``.

    ``S`` is ``(m, m)`` Hermitian PSD with ``S[i,j] = Σ conj(P[i])·P[j]``.
    We need ``M`` with ``conj(M) S Mᵀ = I``; writing ``S = U diag(e) Uᴴ``
    the solution is ``M[m,i] = e_m^(-1/2) U[i,m]``, i.e. the einsum below.
    Eigenvalues are returned in DESCENDING order so the surviving directions
    occupy the leading rows and the caller can drop the tail with one slice.

    ``rank`` counts the directions with ``e > 1e-10 · e_max``.  Everything
    below that threshold is round-off, not a search direction: its column is
    zeroed rather than divided by ``sqrt(e)``, which is what turned noise
    into unit-norm basis vectors before.  The caller MUST slice to ``rank``
    — appending an exact-zero column to V puts a zero row/column into ``Sc``
    and ``Hc``, and the resulting Ritz value is exactly 0 (Ry), i.e. a
    spurious state below the entire physical spectrum.

    Shape-agnostic: P is ``(m, *trailing)`` and only the batch axis is
    contracted, so trailing sharding is untouched.
    """
    e, U = jnp.linalg.eigh(S)                 # ascending, e real
    e = e[::-1]
    U = U[:, ::-1]                            # descending
    thresh = _RANK_DROP_RTOL * jnp.maximum(e[0], 0.0)
    keep = e > thresh
    rank = jnp.sum(keep.astype(jnp.int32))
    inv_sqrt = jnp.where(keep, 1.0 / jnp.sqrt(jnp.maximum(e, _TINY)), 0.0)
    M = U * inv_sqrt[None, :].astype(U.dtype)     # M[i, m] = U[i,m]·e_m^-1/2
    P_w = jnp.einsum('im,i...->m...', M, P, optimize=True)
    return P_w, rank


# ═══════════════════════════════════════════════════════════════════════
#  JIT'd subspace projection kernel (shape-agnostic via ellipsis einsum)
# ═══════════════════════════════════════════════════════════════════════

def _generalized_eigh(A, B):
    """Solve A v = λ B v via Cholesky reduction.  JIT-compatible.

    Operates on small (m, m) replicated matrices — fine to keep replicated.

    The floor on B is relative (``_chol_floor``) and the factor is never
    inverted: ``C = L⁻¹ A L⁻ᴴ`` is two triangular solves and the back
    transform is a third.  ``jnp.linalg.inv(L)`` did a general LU of a matrix
    it already knew to be triangular, and it was the amplifier in step 5 of
    the conditioning-policy note above.
    """
    m = B.shape[0]
    B_reg = B + _chol_floor(B) * jnp.eye(m, dtype=B.dtype)
    L = jnp.linalg.cholesky(B_reg)
    # Y = L⁻¹ A ; then C = Y L⁻ᴴ, obtained as (L⁻¹ Yᴴ)ᴴ.
    Y = jax.scipy.linalg.solve_triangular(L, A, lower=True)
    C = jax.scipy.linalg.solve_triangular(
        L, Y.conj().T, lower=True).conj().T
    C = 0.5 * (C + C.conj().T)
    eigenvalues, V = jnp.linalg.eigh(C)
    eigenvectors = jax.scipy.linalg.solve_triangular(
        L.conj().T, V, lower=False)
    return eigenvalues, eigenvectors


@functools.partial(jax.jit, static_argnames=('n_eig',))
def _ritz_and_residuals(V, HV, n_eig):
    """Project → solve → Ritz vectors → residuals (shape-agnostic).

    V, HV are (m, *trailing) — batch on axis 0, vector content on the rest.
    The Gram/Hamiltonian matrices are (m, m) and stay replicated.
    Ritz vectors are reconstructed as ``X = V x`` via ellipsis einsum so
    sharding on ``trailing`` propagates from V to X.

    Returns (eigenvalues, X, HX, R, res_norms).
    """
    _tally('ritz_and_residuals', V.shape, n_eig)
    # (m, m) projections — ellipsis sums all trailing axes
    Hc = jnp.einsum('m...,n...->mn', jnp.conj(V), HV, optimize=True)
    Sc = jnp.einsum('m...,n...->mn', jnp.conj(V), V, optimize=True)
    Hc = 0.5 * (Hc + Hc.conj().T)
    Sc = 0.5 * (Sc + Sc.conj().T)

    eig_all, C_all = _generalized_eigh(Hc, Sc)
    Lambda = eig_all[:n_eig]
    C_N = C_all[:, :n_eig]

    # Ritz vector reconstruction. The 'mn,m...->n...' contraction has the
    # SUBSPACE m axis on V — which is replicated — and the TRAILING axes
    # (whatever they are) untouched. XLA / shard_map keeps the trailing
    # sharding identical to V's, so X inherits V's PartitionSpec.
    X = jnp.einsum('mn,m...->n...', C_N, V, optimize=True)
    HX = jnp.einsum('mn,m...->n...', C_N, HV, optimize=True)

    # Residual: HX - λ X. Broadcast Lambda along all trailing axes.
    # ``Lambda`` is (n_eig,); pad to (n_eig, *ones) matching X's rank.
    lam_shape = (n_eig,) + (1,) * (X.ndim - 1)
    R = HX - X * Lambda.reshape(lam_shape)

    # ‖R‖ per state — sum over every trailing axis.
    trailing_axes = tuple(range(1, R.ndim))
    res_norms = jnp.sqrt(jnp.sum(jnp.abs(R) ** 2, axis=trailing_axes))

    return Lambda, X, HX, R, res_norms


def _count_converged(conv, n_eig: int) -> int:
    """How many of the ``n_eig`` requested states meet the residual test.

    A TOTAL count, and it used to be a PREFIX count: an inline loop that walked
    from state 0 and stopped at the first unconverged one.  So a solve with
    state 0 still moving and states 1..n-1 all converged reported ``0/n`` and
    was indistinguishable in the log from a solve where nothing had converged
    at all.  The convergence census read exactly that line — ``WARNING: did not
    converge in 200 iterations. Best: 0/20`` — on a run whose eigenvalues were
    already within 1.4 ueV of the exact dense reference.  The number was not
    wrong about any single state; it was answering a different question than
    its own label asked.

    THE CONTROL FLOW IS UNCHANGED BY THIS, and that is checkable rather than
    hoped for.  ``n_conv`` has exactly two uses in :func:`davidson`, the early
    return and the print cadence, and both test ``n_conv == n_eig``.  A prefix
    count reaches ``n_eig`` precisely when every one of the requested states is
    converged — which is precisely when the total count reaches it too.  The
    two rules agree on that test and differ only on the number printed, so the
    iteration at which this solver stops is bit-for-bit what it was.  Gated,
    both directions, by ``tests/test_davidson_convergence_report.py``.
    """
    return int(sum(1 for i in range(n_eig) if conv[i]))


def _default_precond(R, eigenvalues):
    """Identity preconditioner (no-op): returns R / ‖R‖ per state."""
    trailing_axes = tuple(range(1, R.ndim))
    norms = jnp.sqrt(jnp.sum(jnp.abs(R) ** 2, axis=trailing_axes))
    norm_shape = (R.shape[0],) + (1,) * (R.ndim - 1)
    return R / jnp.maximum(norms, 1e-30).reshape(norm_shape)


@jax.jit
def _orthonormalise_batch(P):
    """Self-orthonormalise the batch axis of P, dropping null directions.

    Same self-orthonormalisation step as ``_ortho_expand`` without the
    against-V projection — used to clean up the initial subspace V0.
    Returns ``(P_w, rank)``; see :func:`_whiten_rank_revealing`.
    """
    _tally('orthonormalise_batch', P.shape)
    S_P = jnp.einsum('m...,n...->mn', jnp.conj(P), P, optimize=True)
    S_P = 0.5 * (S_P + S_P.conj().T)
    return _whiten_rank_revealing(S_P, P)


@jax.jit
def _ortho_expand(V, P):
    """Orthonormalise P against V (CGS2) and self-orthonormalise P columns.

    Without this, the Davidson subspace V loses orthonormality across
    iterations: ‖V^H V − I‖ grows, the Cholesky in ``_generalized_eigh``
    becomes ill-conditioned, and eigenvalues blow up to ~1e+40 within
    ~50 iterations. Maintaining V orthonormal keeps Sc ≈ I throughout.

    V    : (m_V, *trailing) — assumed already orthonormal in batch axis.
    P    : (n_eig, *trailing) — preconditioned residuals.

    Returns ``(P_w, rank)``.  ``P_w`` has P's batch width, ordered by
    decreasing Gram eigenvalue; only ``P_w[:rank]`` is a set of genuine,
    orthonormal search directions and only that prefix may be appended to V.
    The rows past ``rank`` are exactly zero — they are the CGS2 round-off of
    already-converged roots, and rescaling them to unit norm (what the old
    absolute ``1e-12·I`` Cholesky floor did) is the divergence mechanism
    documented in the conditioning-policy block at the top of this module.
    """
    _tally('ortho_expand', V.shape, P.shape)
    # Iterated classical Gram-Schmidt against V (twice for full numerical
    # orthogonality at the cost of one extra projection).
    for _ in range(2):
        overlap = jnp.einsum('m...,n...->mn', jnp.conj(V), P, optimize=True)
        P = P - jnp.einsum('mn,m...->n...', overlap, V, optimize=True)

    # Rank-revealing self-orthonormalisation of P (eigh of Pᴴ P).
    S_P = jnp.einsum('m...,n...->mn', jnp.conj(P), P, optimize=True)
    S_P = 0.5 * (S_P + S_P.conj().T)
    return _whiten_rank_revealing(S_P, P)


# ═══════════════════════════════════════════════════════════════════════
#  Warmup
# ═══════════════════════════════════════════════════════════════════════

def warmup_davidson_jit(
    n_eig: int,
    trailing_shape: tuple[int, ...],
    m_max: int | None = None,
    *,
    dtype=jnp.complex128,
    sharding=None,
):
    """Pre-compile _ritz_and_residuals at all subspace sizes.

    Parameters
    ----------
    n_eig : int
        Number of eigenvalues being sought; controls the static argument.
    trailing_shape : tuple[int, ...]
        Shape of *one* state vector (everything after the batch axis 0).
        Examples: ``(n_channels, dim)``, ``(nc, nv, nk)``, ``(dim,)``.
    m_max : int, optional
        Largest subspace dimension that will be reached. Default 10·n_eig,
        matching :func:`davidson`.
    dtype : jnp dtype, optional
        Subspace-vector dtype. Default complex128.
    sharding : jax.sharding.Sharding, optional
        If given, dummy buffers are placed under this sharding so the
        compile cache key matches the production shardings.

    Notes
    -----
    Placement uses ``common.collectives.device_put_process_local``, NOT a
    bare ``jax.device_put``.  ``jnp.zeros(...)`` is an **uncommitted**
    ``jax.Array``, and ``jax/_src/dispatch.py::_device_put_sharding_impl``
    takes the ``multihost_utils.assert_equal`` branch for an uncommitted
    operand whenever the target sharding is not fully addressable — a real
    ``process_allgather`` that materialises ``P × V.nbytes`` on **every**
    rank (AA.1 / AO.1).  The BSE caller
    (``bse.bse_lanczos``, ``solver_kind='davidson'``) passes
    ``NamedSharding(mesh_xy, P(None, 'x', 'y', None))`` — a multi-process
    sharding — and calls this once per subspace size in
    ``{n_eig, 2·n_eig, …, m_max}``, so the antipattern fires four times on
    the full trial-vector block before a single matvec runs.
    The precondition ``device_put_process_local`` documents holds
    trivially: the operand is all-zeros, hence bit-identical on every rank.
    The host seed is a **zero-copy** ``np.broadcast_to`` view (the same
    trick AO.1 used for ``bse_w_exact``'s probe seed), so no rank ever
    materialises the global buffer on the host either — the helper's
    ``np.ascontiguousarray(arr[idx])`` realises only this rank's shard.

    COVERAGE IS PARTIAL, and knowing which part matters.  This warms the
    subspace sizes ``{n_eig, 2·n_eig, …, m_max}`` — exactly the sizes the
    loop visits while every expansion block is full rank, i.e. the early
    iterations.  Once roots start converging the rank-revealing expansion
    (see :func:`_whiten_rank_revealing`) appends fewer than ``n_eig``
    columns, ``m`` stops landing on multiples of ``n_eig``, and the
    remaining sizes compile lazily.  MEASURED on the dense Si BSE matrix at
    the production setting (20 roots, ``m_max = 10·n_eig``): 23 expansions
    over 8 distinct block widths, so the lazy tail is bounded and small.
    Widening ``m_max`` REDUCES it further (3 distinct widths at 20·n_eig)
    because fewer restarts means fewer near-converged expansions.
    """
    if m_max is None:
        m_max = DEFAULT_M_MAX_FACTOR * n_eig
    for m in range(n_eig, m_max + n_eig, n_eig):
        m_eff = min(m, m_max)
        shape = (m_eff,) + tuple(trailing_shape)
        if sharding is None:
            V = jnp.zeros(shape, dtype=dtype)
        else:
            from common.collectives import device_put_process_local
            V = device_put_process_local(
                np.broadcast_to(np.zeros((), dtype=dtype), shape), sharding)
        HV = V
        _ritz_and_residuals(V, HV, n_eig)


# ═══════════════════════════════════════════════════════════════════════
#  Davidson solver
# ═══════════════════════════════════════════════════════════════════════

def davidson(
    apply_H,
    *,
    n_eig: int,
    precond_fn=None,
    init_fn=None,
    X0: jax.Array | None = None,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    stall_patience: int = 20,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Block Davidson iterative eigensolver — shape-agnostic.

    Parameters
    ----------
    apply_H : (m, *trailing) → (m, *trailing)
        Hermitian matvec. Trailing shape is arbitrary; sharding (if any)
        is the caller's responsibility — Davidson never reshapes the
        trailing axes and never inserts collectives.
    n_eig : int
        Number of lowest eigenvalues to converge.
    precond_fn : (R, eigenvalues) → P
        Preconditioner: maps residuals (n_eig, *trailing) and eigenvalues
        (n_eig,) to corrections P with the same shape and sharding.
        Default: identity (normalised residuals).
    init_fn : (apply_H, n_eig) → (X0, HX0)
        Initial subspace builder. Returns (n_eig, *trailing) arrays.
        Default: must provide X0 instead.
    X0 : (n_eig, *trailing)
        Explicit initial vectors (alternative to init_fn).
    m_max : int, optional
        Max subspace dimension before restart. Default **10 x n_eig**.

        MEASURED, not guessed (see the module note below), on the stored dense
        Si BSE matrix (n=1024, 20 roots, to 1 meV / 1e-3 meV on the lowest 20):
        ``4*n_eig`` needs 279 matvecs, ``10*n_eig`` needs ~140 / ~160, and
        ``20*n_eig`` buys nothing over ``10*n_eig``.  The former ``4*n_eig``
        default caps the subspace below what a clustered band edge needs and
        restarts often enough to throw away most of the subspace it just paid
        for; ``10*n_eig`` is the knee, and larger is pure memory.
    max_iter : int, optional
        Iteration cap. Default 100.
    tol : float, optional
        Convergence tolerance; per-state ‖R‖ < tol·max(1,|λ|). Default 1e-8.
    stall_patience : int, optional
        Stop after this many consecutive iterations in which the largest
        residual did not improve by at least 1 %.  Default 20; set to 0 to
        disable and always run to ``max_iter``.

        WHY THIS EXISTS.  ``tol`` is only reachable if the operator is
        Hermitian to better than ``tol``.  It is not, on the decks this
        stack runs: the BSE ``H`` carries ``max|H − Hᴴ| = 8.08e-06`` from an
        asymmetry in W (see the α-Hermiticity gate in ``solvers/lanczos.py``
        for the full calibration), and a Ritz pair extracted from the
        SYMMETRISED projection cannot have a residual below the size of the
        antihermitian part.  MEASURED, production Si 4v4c BSE, 20 roots:
        the eigenvalues reach their final values — matching the dense
        spectrum to 8 significant figures — at **200 matvecs**, at which
        point ‖R‖ ∈ [1.7e-6, 3.3e-6]; ``tol=1e-8`` can then never be met,
        and the solver spent **3820 further matvecs** re-deriving the same
        answer, ending at ‖R‖ ∈ [4.3e-6, 5.7e-6] — slightly WORSE.  With no
        stall guard, "did not converge" is the guaranteed outcome of every
        Davidson BSE run, which trains the reader to ignore it.
    verbose : bool, optional
        Print per-iteration progress. Default True.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_eig,)
    eigenvectors : jax.Array, shape (n_eig, *trailing)
        Returned as a JAX array so callers can keep its sharding; cast
        with ``np.asarray`` if you want host memory.

    Notes
    -----
    The expansion block is RANK-REVEALING: directions whose Gram eigenvalue
    is below ``_RANK_DROP_RTOL·λ_max`` are dropped rather than rescaled, so
    the block width shrinks as roots converge and ``apply_H`` is called on
    fewer vectors.  Each distinct width costs one XLA compile of the caller's
    matvec; in exchange the noise directions that used to be manufactured by
    the absolute Cholesky floor never enter V.  See the conditioning-policy
    block at the top of this module for the measurement.
    """
    import time as _time

    if precond_fn is None:
        precond_fn = _default_precond
    if m_max is None:
        m_max = DEFAULT_M_MAX_FACTOR * n_eig
    # Floor clamp kept from main: a caller-supplied m_max below
    # 2*n_eig cannot restart, so it is raised rather than obeyed.
    m_max = max(int(m_max), 2 * n_eig)

    # Instrumentation: count the vectors H is applied to.  This is the
    # campaign's comparison currency -- the matvec is bandwidth-bound at 84%
    # of HBM peak (KERNEL_DEEPDIVE.md), so matvec count tracks time.
    _apply_H_raw = apply_H

    def apply_H(_V, _f=_apply_H_raw):
        MATVEC_APPLICATIONS[0] += int(_V.shape[0])
        return _f(_V)

    LAST_RUN.clear()
    LAST_RUN.update({'iter': [], 'mv': [], 'm': [], 'eig': [], 'res': [],
                     't': []})
    _t0 = _time.perf_counter()

    # ── initial subspace ──
    if X0 is not None:
        # Explicit X0 — caller controls dtype and sharding.
        V = jnp.asarray(X0[:n_eig])
    elif init_fn is not None:
        V, _HV = init_fn(apply_H, n_eig)
    else:
        raise ValueError("Provide either init_fn or X0")

    # Orthonormalise V0 so the very first ``_ritz_and_residuals`` call
    # operates on a well-conditioned Gram matrix. ``init_bse_subspace`` mixes
    # indicator vectors (mostly orthogonal) with a random Gaussian tail —
    # close to orthonormal but not exactly, and CGS2 in ``_ortho_expand``
    # below assumes V is orthonormal when projecting subsequent residuals.
    V, rank0 = _orthonormalise_batch(V)
    n_keep0 = int(rank0)
    if n_keep0 < V.shape[0]:
        # A rank-deficient START is a caller bug (duplicate indicator vectors,
        # or n_eig larger than the physical block), not round-off — say so
        # rather than silently solving a smaller problem than was asked for.
        raise ValueError(
            f"davidson: initial subspace has rank {n_keep0} < n_eig="
            f"{V.shape[0]}; the {V.shape[0] - n_keep0} dependent trial "
            f"vector(s) cannot seed distinct roots.  Check init_fn / X0.")
    HV = apply_H(V)
    n_matvec = V.shape[0]

    if verbose:
        print(f"Davidson: n_eig={n_eig}, trailing={V.shape[1:]}, m_max={m_max}")

    eigenvalues = None
    X = V  # in case the loop never runs
    n_conv = 0
    best_res = np.inf
    n_stalled = 0

    for it in range(1, max_iter + 1):
        # ── GPU: project + solve + Ritz + residual (one JIT) ──
        Lambda, X, HX, R, res = _ritz_and_residuals(V, HV, n_eig)

        # ── precondition (caller-provided, possibly JIT'd) ──
        # An Olsen-corrected preconditioner needs the current Ritz vectors to
        # project against; a plain Jacobi one does not.  Offer X and fall back,
        # so both signatures work and neither caller has to know about the
        # other.
        try:
            P = precond_fn(R, Lambda, X)
        except TypeError:
            P = precond_fn(R, Lambda)

        # ── convergence check (CPU) ──
        res_np = gather_to_host(res)
        Lambda_np = gather_to_host(Lambda)
        rel_tol = tol * np.maximum(1.0, np.abs(Lambda_np))
        conv = res_np < rel_tol
        # Count EVERY converged root, not the leading run of them.  The old
        # prefix count (break on the first unconverged root) meant one lagging
        # LOW root held the exit closed while every other root was already at
        # 1e-12 — so the solver kept expanding a subspace whose corrections
        # were pure round-off and iterated straight into its own instability.
        # It also mis-reported progress: "conv=1/100" with 99 roots converged.
        # The rule itself lives in ``_count_converged``, which bounds the
        # count at the ``n_eig`` states actually requested; see its docstring
        # for why the prefix→total switch leaves the control flow bit-identical.
        n_conv = _count_converged(conv, n_eig)

        eigenvalues = Lambda_np
        LAST_RUN['iter'].append(it)
        LAST_RUN['mv'].append(MATVEC_APPLICATIONS[0])
        LAST_RUN['m'].append(int(V.shape[0]))
        LAST_RUN['eig'].append(np.asarray(Lambda_np, dtype=np.float64).copy())
        LAST_RUN['res'].append(np.asarray(res_np, dtype=np.float64).copy())
        LAST_RUN['t'].append(_time.perf_counter() - _t0)
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_eig):
            print(f"  iter {it:3d}: m={V.shape[0]:3d}  mv={n_matvec:5d}  "
                  f"eig[0]={float(Lambda_np[0]):12.6f}  "
                  f"eig[{n_eig-1}]={float(Lambda_np[n_eig-1]):12.6f}  "
                  f"res=[{res_np.min():.1e},{res_np.max():.1e}]  "
                  f"conv={n_conv}/{n_eig}")

        if n_conv == n_eig:
            if verbose:
                print(f"  Converged all {n_eig} in {it} iterations, "
                      f"{n_matvec} matvecs.")
            return eigenvalues, X

        # ── stall detector ──
        # Not "converged", but "this is as good as this operator gets".
        # The distinction matters: the residual floor is a property of the
        # OPERATOR's Hermiticity, not of the iteration, so more iterations
        # cannot help and the caller needs to be told which of the two
        # happened.  See the ``stall_patience`` docstring for the numbers.
        res_max = float(res_np.max())
        if res_max < 0.99 * best_res:
            best_res = res_max
            n_stalled = 0
        else:
            n_stalled += 1
            if stall_patience > 0 and n_stalled >= stall_patience:
                if verbose:
                    print(f"  STALLED: largest residual has not improved by "
                          f"1% in {n_stalled} iterations (best {best_res:.2e}, "
                          f"now {res_max:.2e}); {n_conv}/{n_eig} met "
                          f"tol={tol:.1e} after {n_matvec} matvecs.  A "
                          f"residual floor is the size of the operator's "
                          f"ANTIHERMITIAN part — check the alpha-Hermiticity "
                          f"gate; if that reports ~{res_max:.0e}, the "
                          f"eigenvalues are converged and the tolerance is "
                          f"simply unreachable on this operator.")
                break

        # ── expand subspace ──
        # Re-orthonormalise P against V before computing HP. This keeps V's
        # batch-axis Gram matrix close to the identity across iterations;
        # otherwise the Cholesky-based generalized eigh in
        # ``_ritz_and_residuals`` blows up at large m.
        #
        # ``_ortho_expand`` returns the block ordered by decreasing Gram
        # eigenvalue plus the numerical rank; only the leading ``n_keep`` rows
        # are real search directions and the rest are exactly zero.  The slice
        # is a leading-axis (replicated) slice, so it inserts no collective and
        # does not touch the sharded trailing axes.
        P, rank = _ortho_expand(V, P)
        n_keep = int(rank)
        if n_keep == 0:
            # Every correction was round-off: the subspace cannot grow.  This
            # is not convergence — report what was reached and stop rather
            # than append noise (which is what the old code did, and it is
            # the divergence mechanism).
            if verbose:
                print(f"  iter {it}: expansion block has rank 0 — subspace "
                      f"cannot grow; stopping at {n_conv}/{n_eig} converged, "
                      f"{n_matvec} matvecs.")
            break
        P = P[:n_keep]
        HP = apply_H(P)
        n_matvec += n_keep
        V = jnp.concatenate([V, P], axis=0)
        HV = jnp.concatenate([HV, HP], axis=0)

        # ── restart if too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, HV = X, HX

    if verbose and n_conv != n_eig and n_stalled < max(stall_patience, 1):
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_eig} after {n_matvec} matvecs "
              f"(smallest max-residual reached: {best_res:.2e}).")
    return eigenvalues, X
