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


def _to_host(arr) -> np.ndarray:
    """Bring a (possibly multi-process) jax.Array to host as numpy.

    Delegates to :func:`common.collectives.gather_to_host`.

    The docstring this replaces said it mirrored
    ``common.collectives.gather_to_host`` — but that one uses
    ``tiled=True`` and this one used ``tiled=False``, so it mirrored the
    reference incorrectly, and the same mis-citation appears in two other
    copies of the same body.  ``tiled=False`` raises on an array whose shards
    live on other processes; ``process_count()`` does not answer that
    question, ``is_fully_addressable`` does.  One implementation, in the
    service, with the branch and the reason for it in one place.
    """
    if isinstance(arr, np.ndarray):
        return arr
    from common.collectives import gather_to_host
    return gather_to_host(arr)


# ═══════════════════════════════════════════════════════════════════════
#  JIT'd subspace projection kernel (shape-agnostic via ellipsis einsum)
# ═══════════════════════════════════════════════════════════════════════

def _generalized_eigh(A, B):
    """Solve A v = λ B v via Cholesky reduction.  JIT-compatible.

    Operates on small (m, m) replicated matrices — fine to keep replicated.
    """
    m = B.shape[0]
    B_reg = B + 1e-12 * jnp.eye(m, dtype=B.dtype)
    L = jnp.linalg.cholesky(B_reg)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.conj().T
    C = 0.5 * (C + C.conj().T)
    eigenvalues, V = jnp.linalg.eigh(C)
    eigenvectors = jnp.linalg.solve(L.conj().T, V)
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
    """Self-orthonormalise the batch axis of P via Cholesky of P^H P.

    Same self-orthonormalisation step as ``_ortho_expand`` without the
    against-V projection — used to clean up the initial subspace V0.
    """
    S_P = jnp.einsum('m...,n...->mn', jnp.conj(P), P, optimize=True)
    S_P = 0.5 * (S_P + S_P.conj().T)
    n = S_P.shape[0]
    L_P = jnp.linalg.cholesky(S_P + 1e-12 * jnp.eye(n, dtype=S_P.dtype))
    L_P_inv = jnp.linalg.inv(L_P)
    return jnp.einsum('mi,i...->m...', jnp.conj(L_P_inv), P, optimize=True)


@jax.jit
def _ortho_expand(V, P):
    """Orthonormalise P against V (CGS2) and self-orthonormalise P columns.

    Without this, the Davidson subspace V loses orthonormality across
    iterations: ‖V^H V − I‖ grows, the Cholesky in ``_generalized_eigh``
    becomes ill-conditioned, and eigenvalues blow up to ~1e+40 within
    ~50 iterations. Maintaining V orthonormal keeps Sc ≈ I throughout.

    V    : (m_V, *trailing) — assumed already orthonormal in batch axis.
    P    : (n_eig, *trailing) — preconditioned residuals.

    Returns P with V^H P ≈ 0 and P^H P ≈ I (in the (m, n) Gram metric
    summing over all trailing axes).
    """
    # Iterated classical Gram-Schmidt against V (twice for full numerical
    # orthogonality at the cost of one extra projection).
    for _ in range(2):
        overlap = jnp.einsum('m...,n...->mn', jnp.conj(V), P, optimize=True)
        P = P - jnp.einsum('mn,m...->n...', overlap, V, optimize=True)

    # Self-orthonormalisation of P via Cholesky factorisation of P^H P.
    S_P = jnp.einsum('m...,n...->mn', jnp.conj(P), P, optimize=True)
    S_P = 0.5 * (S_P + S_P.conj().T)
    n = S_P.shape[0]
    L_P = jnp.linalg.cholesky(S_P + 1e-12 * jnp.eye(n, dtype=S_P.dtype))
    L_P_inv = jnp.linalg.inv(L_P)
    P = jnp.einsum('mi,i...->m...', jnp.conj(L_P_inv), P, optimize=True)
    return P


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
        Largest subspace dimension that will be reached. Default 4·n_eig.
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
    """
    if m_max is None:
        m_max = 4 * n_eig
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
        Max subspace dimension before restart. Default 4 × n_eig.
    max_iter : int, optional
        Iteration cap. Default 100.
    tol : float, optional
        Convergence tolerance; per-state ‖R‖ < tol·max(1,|λ|). Default 1e-8.
    verbose : bool, optional
        Print per-iteration progress. Default True.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_eig,)
    eigenvectors : jax.Array, shape (n_eig, *trailing)
        Returned as a JAX array so callers can keep its sharding; cast
        with ``np.asarray`` if you want host memory.
    """
    if precond_fn is None:
        precond_fn = _default_precond
    if m_max is None:
        m_max = 4 * n_eig

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
    V = _orthonormalise_batch(V)
    HV = apply_H(V)

    if verbose:
        print(f"Davidson: n_eig={n_eig}, trailing={V.shape[1:]}, m_max={m_max}")

    eigenvalues = None
    X = V  # in case the loop never runs

    for it in range(1, max_iter + 1):
        # ── GPU: project + solve + Ritz + residual (one JIT) ──
        Lambda, X, HX, R, res = _ritz_and_residuals(V, HV, n_eig)

        # ── precondition (caller-provided, possibly JIT'd) ──
        P = precond_fn(R, Lambda)

        # ── convergence check (CPU) ──
        res_np = _to_host(res)
        Lambda_np = _to_host(Lambda)
        rel_tol = tol * np.maximum(1.0, np.abs(Lambda_np))
        conv = res_np < rel_tol
        n_conv = _count_converged(conv, n_eig)

        eigenvalues = Lambda_np
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_eig):
            print(f"  iter {it:3d}: m={V.shape[0]:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_eig-1}]={float(Lambda[n_eig-1]):12.6f}  "
                  f"res=[{res_np.min():.1e},{res_np.max():.1e}]  "
                  f"conv={n_conv}/{n_eig}")

        if n_conv == n_eig:
            if verbose:
                print(f"  Converged all {n_eig} in {it} iterations.")
            return eigenvalues, X

        # ── expand subspace ──
        # Re-orthonormalise P against V before computing HP. This keeps V's
        # batch-axis Gram matrix close to the identity across iterations;
        # otherwise the Cholesky-based generalized eigh in
        # ``_ritz_and_residuals`` blows up at large m.
        P = _ortho_expand(V, P)
        HP = apply_H(P)
        V = jnp.concatenate([V, P], axis=0)
        HV = jnp.concatenate([HV, HP], axis=0)

        # ── restart if too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, HV = X, HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_eig}")
    return eigenvalues, X
