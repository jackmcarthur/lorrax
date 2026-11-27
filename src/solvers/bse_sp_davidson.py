"""
solvers/bse_sp_davidson.py — structure-preserving block Davidson for the lowest
roots of the full (non-Tamm-Dancoff) Bethe-Salpeter problem.

WHAT THIS SOLVES
----------------
    H [X; Y] = omega [X; Y],     H = [[ A,  B ], [ -conj(B), -conj(A) ]],
    A = A^H,  B = B^T,           normalisation  X^H X - Y^H Y = +1.

The only operator access is a real-linear block applier

    pair(X, s) = A X + s B conj(X),        M(x) := pair(x, +1)            (1)

(``bse.bse_stack_matvec.build_bse_stack_pair_matvec`` in production).  Only
``s = +1`` is used: ``pair(x, -1) = -i pair(i x, +1)``.  Equation numbers are
those of ``docs/architecture/iterative_eigensolvers.md#structure-preserving-davidson``.

THE METHOD
----------
The basis is a set ``w_j`` orthonormal in the REAL inner product
(``Re(W^H W) = I``) and the trial space is ``span_C{[w_j; conj(w_j)]}`` (2).
With

    M_r = Re(W^H M W)   real symmetric,       S = Im(W^H W)   real antisymmetric,

the Galerkin projection of ``H z = omega z`` is ``M_r c = omega (i S) c`` (3).
``M_r`` is the compression of ``K = [[A, B], [conj B, conj A]]`` onto the trial
space, so when ``K`` is positive definite (3) is a Hermitian definite pencil:
the Ritz values are real, come in exact +- pairs, lie on or above the exact
roots, and the Ritz vectors

    X = sum_j c_j w_j,     Y = conj( sum_j conj(c_j) w_j )                (4)

satisfy ``X_i^H X_j - Y_i^H Y_j = c_i^H (2 i S) c_j = delta_ij``.  The residual
of (4) needs no operator application (5).  Each open root adds two real
directions per iteration, the paired form of its preconditioned residual
``[t_X; t_Y]`` (6): two pair calls.  It is the paired-trial-vector response
solver of Olsen, Jensen and Jorgensen, J. Comput. Phys. 74 (1988) 265, written
for a complex symmetric ``B`` through the real-linear applier; the projected
solve is Algorithm 1 of Shao, da Jornada, Yang, Deslippe and Louie, Linear
Algebra Appl. 488 (2016) 148.

REAL COEFFICIENTS ONLY
----------------------
``M`` is real-linear, not complex-linear: ``M(alpha x) != alpha M(x)``.  Every
coefficient that multiplies a basis vector while its image ``M w`` is carried
along (Gram-Schmidt, whitening, restart rotation) is therefore REAL.  The
planned ``orthogonalize`` of ``distrib_la`` uses complex coefficients and is
not used here; ``_real_cgs2`` is its real-coefficient form on the same plan.
The complex Ritz coefficients ``c`` never rotate the basis: they only form
``X``, ``Y`` and the residual, where ``H [w; conj w] = [M w; -conj(M w)]`` is
complex-linear in ``c``.

WHEN K IS NOT POSITIVE DEFINITE
-------------------------------
``lam_min(M_r) >= lam_min(K)`` (Cauchy interlacing), so a projected ``M_r``
that is not positive definite is a CERTIFICATE that ``K`` is not: an
instability of the reference state, found with no extra operator application.
``on_indefinite='refuse'`` (the default) raises with the measured value.
``'continue'`` separates the few unstable roots (complex ``omega``, or a real
root of positive type with ``omega <= 0``) with a general solve of the
projected pencil, and keeps the Cholesky solve for all the others on the
``M_r``-orthogonal complement, where ``M_r`` is positive definite again
(``_solve_projected``).

HOST LOOP ON PLANNED BUFFERS — AND WHY NOT ONE ``while_loop``
---------------------------------------------------------------
``solvers.davidson_fixed`` runs its whole iteration in one ``lax.while_loop``
because every dense step it needs is one Hermitian eigensolve that
``distrib_la`` provides on each device leg, and because NSCF runs it once per
k-point.  The same three questions give the other answer here:

* the projected problem needs a real Cholesky factor, two triangular solves
  and, for an unstable reference state, a general complex eigensolve.  JAX has
  no device kernel for the last and ``distrib_la`` no vetted per-leg route for
  the others, so one ``while_loop`` would need a host callback in every
  iteration (what the planned Davidson removed) or a second, host-side loop
  for the unstable case (two copies of the iteration);
* the full BSE is one solve per run of a few tens of iterations, each
  dominated by ``2 n_open`` pair calls; the loop below adds, per iteration,
  three kernel dispatches (four at a restart), the applier's power-of-two
  pieces, ten small coefficient reductions and one host solve of at most
  ``capacity^2`` reals;
* fixed shapes are kept all the same.  The basis ``W``, its images ``M W`` and
  every block have a fixed capacity, the kernels are the plan's active-window
  operations (``store``, ``gram``, ``reconstruct``) with traced counts, so
  nothing recompiles as the basis grows, and the kernel set is reused across
  solves (``plan_key``).

Measured on four emulated CPU devices: the planned operations run as separate
programs on 1 to 6 host threads, where the ``while_loop`` Davidson is refused
below 16 (``bse_lanczos`` GATE cpu_davidson_threads).

Every process runs this loop on the same replicated small arrays (``M_r``,
``S``, residuals, ranks come out of all-reduces and are bit-identical on every
rank).  The host solves are rank 0's on every rank (``_from_rank0``), so the
control flow cannot diverge on a host LAPACK that rounds differently.

WHAT IS NOT HERE
----------------
No physics, no input keys, no energy units.  The BSE layer
(``bse.bse_nontda``) owns the operator, the preconditioner's diagonal, the
start block and the residual gate in meV.
"""
from __future__ import annotations

import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable

import numpy as np
import scipy.linalg as sla

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import (gather_to_host, process_count, process_rank, psum_replicate,
                                replicate_to_mesh)
from common.log import block, get_logger
from distrib_la import plan_subspace
from solvers.davidson_fixed import (BAD_INITIAL, BAD_TOLERANCE, CONVERGED, ITERATION_LIMIT,
                                    NO_DIRECTIONS, NONFINITE, RUNNING, STALLED)
from solvers.subspace_numerics import _RANK_DROP_RTOL, _rank_whitener

log = get_logger(__name__)

#: Statuses beyond ``solvers.davidson_fixed``'s (whose codes are shared).
INDEFINITE = 8        # M_r not positive definite under on_indefinite='refuse'
NOT_SYMMETRIC = 9     # Re(P^H M P) asymmetric: A is not Hermitian or B not symmetric
STATUS_NAMES = {RUNNING: "RUNNING", CONVERGED: "CONVERGED", ITERATION_LIMIT: "ITERATION_LIMIT",
                NO_DIRECTIONS: "NO_DIRECTIONS", BAD_INITIAL: "BAD_INITIAL", NONFINITE: "NONFINITE",
                BAD_TOLERANCE: "BAD_TOLERANCE", STALLED: "STALLED", INDEFINITE: "INDEFINITE",
                NOT_SYMMETRIC: "NOT_SYMMETRIC"}

#: Basis capacity in real-form vectors per requested root: the depth in
#: iterations of the Tamm-Dancoff default (``solvers.davidson``, 10 vectors a
#: root), at two real-form vectors per vector.
DEFAULT_CAPACITY_FACTOR = 20
#: Smallest capacity in blocks of ``n_block`` rows: the restarted basis
#: (2 blocks) and two iterations of corrections.
MIN_CAPACITY_BLOCKS = 6
#: ``M_r`` counts as positive definite above this multiple of its largest
#: diagonal entry.  Below it the sign of ``lam_min`` is rounding.
PD_RTOL = 1e-13
#: Largest relative asymmetry of ``Re(P^H M P)`` accepted on a new block.  A
#: correct operator gives round-off; ``bse_nontda``'s matrix-free route uses the
#: same number for the same property (``_MF_METRIC_TOL``).
METRIC_TOL = 1e-8
#: A projected root is real when ``|Im omega|`` is below this fraction of
#: ``|omega|`` plus ``IMAG_ATOL`` times the scale of ``M_r``.
IMAG_RTOL = 1e-9
IMAG_ATOL = 1e-12
#: A projected root near zero is known only to about ``eps * scale^2 / |omega|``
#: (``omega^2 = lam_- lam_+`` is what the pencil fixes to ``eps * scale^2``).
#: The real part of a non-real root counts as zero below ``ROOT_NOISE`` times
#: that: an imaginary root whose real part is rounding-negative is still the
#: representative of its pair.
ROOT_NOISE = 100.0

#: Kernel sets kept by the caller's ``plan_key`` (as ``solvers.davidson._PLANS``).
_KERNELS: dict = {}
_KERNELS_MAX = 8


# ═══════════════════════════════════════════════════════════════════════
#  The projected problem (host; at most capacity x capacity reals)
# ═══════════════════════════════════════════════════════════════════════

def _real_range(columns, rtol=_RANK_DROP_RTOL):
    """Orthonormal real basis of the range of real ``columns`` (m, k): left
    singular vectors above ``rtol`` of the largest singular value."""
    if columns.shape[1] == 0:
        return np.zeros((columns.shape[0], 0))
    U, s, _ = np.linalg.svd(columns, full_matrices=False)
    return U[:, s > rtol * max(s[0], 1e-300)]


def solve_projected_definite(Mr, S, n_roots):
    """Lowest positive roots of ``M_r c = omega (i S) c``, Eq. (3), for a
    positive definite ``M_r``.

    ``M_r = L L^T``; the Hermitian ``i L^-1 S L^-T`` has the eigenvalues
    ``+-1/omega`` (Shao et al., Linear Algebra Appl. 488 (2016) 148,
    Algorithm 1, on the projected matrices).

    Parameters
    ----------
    Mr : (m, m) real symmetric positive definite.
    S : (m, m) real antisymmetric.
    n_roots : int

    Returns
    -------
    omega : (k,) ascending, ``k <= n_roots`` (fewer when the pencil has fewer
        positive roots).
    C : (m, k) complex with ``c_i^H (2 i S) c_j = delta_ij``, that is
        ``X_i^H X_j - Y_i^H Y_j = delta_ij`` in Eq. (4).

    Raises ``numpy.linalg.LinAlgError`` when ``M_r`` is not positive definite.
    """
    L = np.linalg.cholesky(Mr)
    T = sla.solve_triangular(L, S, lower=True)
    Sh = sla.solve_triangular(L, T.T, lower=True).T
    mu, Yv = np.linalg.eigh(0.5j * (Sh - Sh.T))
    mu, Yv = mu[::-1], Yv[:, ::-1]
    keep = int(min(n_roots, np.sum(mu > 1e-14 * max(mu[0], 0.0))))
    mu, Yv = mu[:keep], Yv[:, :keep]
    C = sla.solve_triangular(L.T, Yv, lower=False) / np.sqrt(2.0 * mu)[None, :]
    return 1.0 / mu, C


def _sigma_orthonormal(V, S):
    """Columns of positive type made Sigma-orthonormal: ``V R^-1`` with
    ``V^H (2 i S) V = R^H R``, so that ``c_i^H (2 i S) c_j = delta_ij``.

    Eigenvectors of distinct roots of the pencil are Sigma-orthogonal already;
    inside a multiplet a general solve returns any basis, and scaling each
    column alone leaves ``X_i^H X_j - Y_i^H Y_j`` off the identity there.
    Falls back to the per-column scale when the Gram matrix is not positive
    definite to rounding.
    """
    if V.shape[1] == 0:
        return V
    G = V.conj().T @ ((2j * S) @ V)
    G = 0.5 * (G + G.conj().T)
    try:
        R = np.linalg.cholesky(G).conj().T
    except np.linalg.LinAlgError:
        return V / np.sqrt(np.abs(np.real(np.diag(G))))[None, :]
    return sla.solve_triangular(R, V.T, trans="T", lower=False).T


def _solve_projected(Mr, S, n_roots, scale, *, definite):
    """Roots of ``M_r c = omega (i S) c`` in the solver's order.

    ``definite``: Cholesky route (:func:`solve_projected_definite`).
    Otherwise the unstable sector is separated first.  A general (QZ) solve of
    the pencil gives its eigenpairs; the special ones are (a) complex roots,
    one representative ``Im omega > 0, Re omega >= 0`` per pair or quadruplet,
    and (b) real roots of positive type (``c^H i S c > 0``) with
    ``omega <= 0``.  Eigenvectors of roots that are not conjugate partners are
    ``M_r``- and ``S``-orthogonal, so on the ``M_r``-orthogonal complement of
    the special sector the pencil has only the stable roots and ``M_r`` is
    positive definite there: the Cholesky route gives them, Sigma-orthonormal
    inside a multiplet as in the definite case.

    Returns a dict: ``omega`` (k,) complex in the order unstable (descending
    ``Im``), negative real (ascending), stable real (ascending); ``C`` (m, k);
    ``n_unstable``, ``n_negative``; ``keep`` (m, r) real columns spanning the
    special eigenvectors and their partners (the restart keeps them); and
    ``general`` (True when the complement was not positive definite and the
    stable roots come from the general solve, unorthogonalised).
    """
    m = Mr.shape[0]
    if definite:
        om, C = solve_projected_definite(Mr, S, n_roots)
        return dict(omega=om.astype(complex), C=C, n_unstable=0, n_negative=0,
                    keep=np.zeros((m, 0)), general=False)
    w, V = sla.eig(Mr, 1j * S)
    ok = np.isfinite(w)
    w, V = w[ok], V[:, ok]
    sig = np.real(np.einsum("ij,ij->j", V.conj(), (1j * S) @ V))
    real = np.abs(w.imag) <= IMAG_RTOL * np.abs(w) + IMAG_ATOL * scale
    # The representative has Im omega > 0 and Re omega >= 0.  An imaginary root
    # has Re omega = 0 only to the accuracy of a root near zero (ROOT_NOISE):
    # a rounding-negative real part must not drop it.  The two members of a
    # quadruplet whose real parts are below that are both kept: they are two
    # imaginary roots to working precision.
    re_zero = np.maximum(IMAG_RTOL * np.abs(w) + IMAG_ATOL * scale,
                         ROOT_NOISE * np.finfo(float).eps * scale ** 2
                         / np.maximum(np.abs(w), np.finfo(float).tiny))
    iu = np.where(~real & (w.imag > 0) & (w.real >= -re_zero))[0]
    iu = iu[np.argsort(-w.imag[iu])]
    ineg = np.where(real & (sig > 0) & (w.real <= 0))[0]
    ineg = ineg[np.argsort(w.real[ineg])]
    # The partners: conj(omega) for a complex root (its eigenvector is a second
    # one), conj(c) for a real root (the same real plane).  One partner per
    # representative: degenerate roots must not share one.
    below = [int(j) for j in np.where(~real & (w.imag < 0))[0]]
    mirror = []
    for j in iu:
        if not below:
            break
        k = min(below, key=lambda i: abs(w[i] - np.conj(w[j])))
        mirror.append(k)
        below.remove(k)
    special = np.concatenate([V[:, iu], V[:, mirror], V[:, ineg]], axis=1)
    keep = _real_range(np.concatenate([special.real, special.imag], axis=1))
    om_u, C_u = w[iu].astype(complex), V[:, iu].astype(complex)
    C_u = C_u / (np.sqrt(2.0) * np.linalg.norm(C_u, axis=0))[None, :]
    om_n, C_n = w[ineg].real.astype(complex), _sigma_orthonormal(V[:, ineg].astype(complex), S)
    n_stable = max(0, n_roots - len(iu) - len(ineg))
    general = False
    try:
        # N: real orthonormal basis of the complement of the special sector,
        # {d : keep^T S d = 0} = {d : keep^T M_r d = 0}.  The two descriptions
        # are the same space; the better conditioned one is used.  M_r keep
        # loses eps / |lam| digits in a soft direction of the sector (its
        # column there is a difference of O(scale) terms of size |lam|), and
        # the stable roots then stall at that level; S keep does not.
        cands = [G for G in (S @ keep, Mr @ keep / max(scale, np.finfo(float).tiny))
                 if G.shape[1]]
        if cands:
            spread = [np.linalg.svd(G, compute_uv=False) for G in cands]
            pick = int(np.argmax([s[-1] / max(s[0], np.finfo(float).tiny) for s in spread]))
            Qc, _ = np.linalg.qr(cands[pick], mode="complete")
        else:
            Qc = np.eye(m)
        Nc = Qc[:, keep.shape[1]:]
        om_s, C_s = solve_projected_definite(Nc.T @ Mr @ Nc, Nc.T @ S @ Nc, n_stable)
        C_s = Nc @ C_s
    except np.linalg.LinAlgError:
        general = True
        ist = np.where(real & (sig > 0) & (w.real > 0))[0]
        ist = ist[np.argsort(w.real[ist])][:n_stable]
        om_s, C_s = w[ist].real, _sigma_orthonormal(V[:, ist].astype(complex), S)
    omega = np.concatenate([om_u, om_n, om_s.astype(complex)])[:n_roots]
    C = np.concatenate([C_u, C_n, C_s], axis=1)[:, :n_roots]
    return dict(omega=omega, C=C, n_unstable=min(len(iu), n_roots),
                n_negative=min(len(ineg), max(0, n_roots - len(iu))), keep=keep, general=general)


def _restart_rotation(keep, C_stable, width):
    """Real orthonormal ``Q`` (m, r <= width) spanning ``keep`` and the real
    forms ``[Re c, Im c]`` of as many leading stable Ritz vectors as fit:
    the restarted basis ``Q^T W`` holds those Ritz vectors and their partners,
    Eq. (7).  Real, so the images follow by the same rotation."""
    k = C_stable.shape[1]
    while True:
        Q = _real_range(np.concatenate([keep, C_stable[:, :k].real, C_stable[:, :k].imag], axis=1))
        if Q.shape[1] <= width or k == 0:
            return Q[:, :width]
        k -= 1


def _projected_step(Mr, S, n_block, capacity, *, solve_indefinite):
    """One iteration's host work on the active projected matrices, returned as
    arrays whose shapes depend on the plan only, so that every rank can take
    rank 0's through one fixed-shape all-reduce (``_from_rank0``).

    Parameters
    ----------
    Mr, S : (m, m) real symmetric / antisymmetric
        The active blocks of ``M_r`` and ``S``.
    solve_indefinite : bool
        Solve also when ``M_r`` is not positive definite; otherwise only the
        measurement is returned for that case.

    Returns
    -------
    omega : (n_block,) complex — roots in the solver's order, 0 where absent.
    C : (capacity, n_block) complex — Ritz coefficients in the first ``m``
        rows, zero columns where absent.
    present : (n_block,) bool.
    Q : (capacity, 2 n_block) real — the restart rotation of Eq. (7) in its
        first ``m`` rows and ``r`` columns.
    counts : (5,) int — definite, n_special (unstable and non-positive real
        roots), n_unstable, r, general (1 when the stable roots came from the
        general solve).
    reals : (2,) — ``lam_min(M_r)`` and the positive-definiteness floor.
    """
    m, nb = Mr.shape[0], n_block
    scale = float(np.max(np.abs(np.diag(Mr))))
    lam_min = float(sla.eigh(Mr, eigvals_only=True, subset_by_index=[0, 0])[0])
    floor = PD_RTOL * scale
    definite = lam_min > floor
    sol = None
    if definite:
        try:
            sol = _solve_projected(Mr, S, nb, scale, definite=True)
        except np.linalg.LinAlgError:
            definite = False
    if sol is None and solve_indefinite:
        sol = _solve_projected(Mr, S, nb, scale, definite=False)
    om, C = np.zeros(nb, complex), np.zeros((capacity, nb), complex)
    Q = np.zeros((capacity, 2 * nb))
    n_special = n_un = r = general = k = 0
    if sol is not None:
        k = sol["omega"].shape[0]
        om[:k], C[:m, :k] = sol["omega"], sol["C"]
        n_un, n_special = sol["n_unstable"], sol["n_unstable"] + sol["n_negative"]
        rot = _restart_rotation(sol["keep"], sol["C"][:, n_special:], 2 * nb)
        r, general = rot.shape[1], int(sol["general"])
        Q[:m, :r] = rot
    return [om, C, np.arange(nb) < k, Q,
            np.array([int(definite), n_special, n_un, r, general], np.int64),
            np.array([lam_min, floor])]


# ═══════════════════════════════════════════════════════════════════════
#  Device kernels on the planned buffers
# ═══════════════════════════════════════════════════════════════════════
#  Rows are vectors; the trailing axes are the caller's and keep its sharding.
#  A Gram contracts every trailing axis (one all-reduce of a small coefficient
#  matrix when they are sharded); a combination sums over the replicated row
#  axis (no collective).  No vector is gathered and no vector-sized array is
#  replicated.

def _real_cgs2(basis, W, Pb, m):
    """Two classical Gram-Schmidt passes of the block ``Pb`` against the first
    ``m`` rows of ``W`` with REAL coefficients ``Re(W^H p)``: the plan's
    ``gram``, its real part, ``reconstruct``, a subtraction.  ``W`` is
    orthonormal in the real inner product only, and its images follow real
    combinations only (module docstring)."""
    for _ in range(2):
        c = jnp.real(basis.gram(W, Pb, m)).astype(jnp.complex128)
        x, _ = basis.reconstruct(W, W, c, m, Pb, compute_image=False)
        Pb = basis.constrain(Pb - x)
    return Pb


def _real_whiten(blockp, Pb, room):
    """Orthonormalise the rows of ``Pb`` in the real inner product and drop
    dependent ones: the rank rule of ``solvers.subspace_numerics`` on the REAL
    Gram matrix ``Re(Pb Pb^H)``.  Returns the block (rows past the rank zero)
    and the rank, at most ``room``."""
    n = Pb.shape[0]
    T, rank = _rank_whitener(jnp.real(blockp.gram(Pb, Pb, n)))
    rank = jnp.minimum(rank.astype(jnp.int32), room)
    out, _ = blockp.reconstruct(Pb, Pb, T.astype(jnp.complex128), n, Pb,
                                columns=rank, compute_image=False)
    return blockp.constrain(out), rank


@lru_cache(maxsize=_KERNELS_MAX)
def _algebra(n_block, capacity, vector_shape, sharding):
    """The operator-free kernels of one geometry, compiled once per process.

    ``sharding`` is the vectors' ``NamedSharding`` (row axis unsharded) or
    None for one device.  Vector outputs keep it; coefficient outputs are
    replicated.  ``extend`` and ``restart`` donate the two capacity buffers,
    which they update in place.
    """
    nb, nb2 = n_block, 2 * n_block
    basis = plan_subspace(capacity=capacity, n_eig=nb2, vector_sharding=sharding,
                          max_block_size=nb2)
    blockp = plan_subspace(capacity=nb2, n_eig=nb2, vector_sharding=sharding)
    rep = None if sharding is None else NamedSharding(sharding.mesh, P())

    def jit(fn, outs, donate=()):
        if sharding is None:
            return jax.jit(fn, donate_argnums=donate,
                           in_shardings=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
        return jax.jit(fn, out_shardings=outs, donate_argnums=donate)

    def buffers():
        zero = jnp.zeros((capacity,) + vector_shape, jnp.complex128)
        return basis.constrain(zero), basis.constrain(zero)

    def first(X0, room):
        """``[X0; i X0]`` orthonormalised in the real inner product."""
        Pb = blockp.constrain(jnp.concatenate([X0, 1j * X0], axis=0))
        Pb, _ = _real_whiten(blockp, Pb, room)
        return _real_whiten(blockp, Pb, room)

    def ortho(W, Pc, mask, m, room):
        """The correction block made orthonormal to ``W[:m]`` and to itself.

        ``mask`` (2 n_block,) zeroes the rows of locked roots first.  Real
        CGS2, whitening with the rank rule, and both once more (the second
        pass removes what the whitening's rescaling brought back)."""
        m, room = jnp.asarray(m, jnp.int32), jnp.asarray(room, jnp.int32)
        Pb = jnp.where(mask.reshape((nb2,) + (1,) * len(vector_shape)), Pc, 0)
        Pb = blockp.constrain(Pb)
        rank = jnp.int32(0)
        for _ in range(2):
            Pb = _real_cgs2(basis, W, Pb, m)
            Pb, rank = _real_whiten(blockp, Pb, room)
        return Pb, rank

    def extend(W, MW, Pb, MPb, m, rank):
        """Store ``rank`` new rows at ``m`` and return the two new panels
        ``W[:m+rank]^H MP`` and ``W[:m+rank]^H P`` (capacity, 2 n_block)."""
        m, rank = jnp.asarray(m, jnp.int32), jnp.asarray(rank, jnp.int32)
        W, MW = basis.store(W, MW, Pb, MPb, m, rank)
        W, MW = basis.constrain(W), basis.constrain(MW)
        return W, MW, basis.gram(W, MPb, m + rank), basis.gram(W, Pb, m + rank)

    def restart(W, MW, Qc, m, r, template):
        """Rows ``[0, r)`` of the buffers become ``Q^T W`` and ``Q^T M W``, Eq. (7)."""
        m, r = jnp.asarray(m, jnp.int32), jnp.asarray(r, jnp.int32)
        Wr, MWr = basis.reconstruct(W, MW, Qc, m, template, columns=r)
        W, MW = basis.store(W, MW, blockp.constrain(Wr), blockp.constrain(MWr), jnp.int32(0), r)
        return basis.constrain(W), basis.constrain(MW)

    def sigma_gram(X, Y):
        """``X_i^H X_j - Y_i^H Y_j`` (n_block, n_block): the metric of Eq. (4)."""
        G = blockp.gram(blockp.constrain(jnp.concatenate([X, Y], axis=0)),
                        blockp.constrain(jnp.concatenate([X, Y], axis=0)), nb2)
        return G[:nb, :nb] - G[nb:, nb:]

    return SimpleNamespace(
        basis=basis, blockp=blockp, rep=rep,
        buffers=jit(buffers, (sharding, sharding)),
        first=jit(first, (sharding, rep)),
        ortho=jit(ortho, (sharding, rep)),
        extend=jit(extend, (sharding, sharding, rep, rep), donate=(0, 1)),
        restart=jit(restart, (sharding, sharding), donate=(0, 1)),
        sigma_gram=jit(sigma_gram, rep),
        jit=jit)


def _operator_kernels(alg, apply_pair, precondition, n_block, vector_shape, sharding):
    """The two kernels that hold the caller's programs.

    ``ritz``: Ritz vectors, residual norms and the paired corrections of every
    root, Eqs. (4)-(6).  ``apply(width)``: ``M`` on ``width`` rows of a block,
    written back into the image block; widths are powers of two, so the
    applier compiles for at most ``log2`` widths and no padded row is ever
    multiplied (the rule of ``davidson_fixed._active_blocks``).
    """
    nb = n_block
    basis, blockp = alg.basis, alg.blockp
    axes = tuple(range(1, 1 + len(vector_shape)))
    tail = (1,) * len(vector_shape)

    def ritz(data, W, MW, Cc, omega, m, template):
        m = jnp.asarray(m, jnp.int32)
        comb, hcomb = basis.reconstruct(W, MW, Cc, m, template)
        X, Y = comb[:nb], jnp.conj(comb[nb:])                    # Eq. (4)
        HX, HY = hcomb[:nb], -jnp.conj(hcomb[nb:])               # H z = sum_j c_j [M w_j; -conj(M w_j)]
        om = omega.reshape((nb,) + tail)
        RX, RY = HX - om * X, HY - om * Y                        # Eq. (5)
        r2 = jnp.sum(jnp.abs(RX) ** 2, axis=axes) + jnp.sum(jnp.abs(RY) ** 2, axis=axes)
        z2 = jnp.sum(jnp.abs(X) ** 2, axis=axes) + jnp.sum(jnp.abs(Y) ** 2, axis=axes)
        resid = jnp.where(z2 > 0, jnp.sqrt(r2 / jnp.where(z2 > 0, z2, 1.0)), jnp.inf)
        tX, tY = precondition(data, RX, RY, omega)
        # Eq. (6): the two real-form components of [t_X; t_Y].  One real scale
        # per root: the relative size of t_X and t_Y is information.
        a = tX + jnp.conj(tY)
        b = -1j * (tX - jnp.conj(tY))
        n2 = jnp.sum(jnp.abs(tX) ** 2, axis=axes) + jnp.sum(jnp.abs(tY) ** 2, axis=axes)
        scale = (1.0 / jnp.sqrt(jnp.maximum(n2, 1e-300))).reshape((nb,) + tail)
        Pc = blockp.constrain(jnp.concatenate([a * scale, b * scale], axis=0))
        return X, Y, Pc, resid

    applies: dict = {}

    def apply(width):
        if width not in applies:
            def run(data, Pb, MPb, start):
                piece = jax.lax.dynamic_slice_in_dim(Pb, start, width)
                if sharding is not None:
                    piece = jax.lax.with_sharding_constraint(piece, sharding)
                out = apply_pair(data, piece, jnp.ones((width,), jnp.float64))
                return blockp.constrain(jax.lax.dynamic_update_slice_in_dim(
                    MPb, out.astype(MPb.dtype), start, axis=0))
            applies[width] = alg.jit(run, sharding)
        return applies[width]

    def zero_block():
        return blockp.constrain(jnp.zeros((2 * nb,) + vector_shape, jnp.complex128))

    return SimpleNamespace(ritz=alg.jit(ritz, (sharding, sharding, sharding, alg.rep)),
                           apply=apply, zero_block=alg.jit(zero_block, sharding))


def sigma_gradient(data, RX, RY, omega):
    """The preconditioner used when none is given: ``[R_X; -R_Y]``, the
    gradient of the Thouless functional (Eq. 6 with ``D -> infinity``)."""
    return RX, -RY


def sp_diagonal_precond(*, epsilon_shift=1e-3):
    """Paired diagonal preconditioner ``(H_D - omega)^-1`` with
    ``H_D = diag(D, -D)``, Eq. (6):

        t_X = R_X / (D - omega + eps),     t_Y = -R_Y / (D + omega + eps).

    ``D`` (real, the vectors' shape and sharding) is the diagonal of ``A``; the
    diagonal of ``H`` is ``[D; -D]`` because ``A`` is Hermitian and ``B`` sits
    in the off-diagonal blocks.  Elementwise: no collective.  It does not
    normalise ``t_X`` and ``t_Y`` separately; they are one direction.

    Returns ``precondition(D, RX, RY, omega) -> (tX, tY)`` on (n, *vs) blocks;
    pass ``D`` as the solver's ``data`` (or a function of it).
    """
    def precondition(D, RX, RY, omega):
        om = omega.reshape((RX.shape[0],) + (1,) * (RX.ndim - 1))
        eps = jnp.asarray(epsilon_shift, dtype=D.dtype)
        dx, dy = D[None] - om + eps, D[None] + om + eps
        dx = jnp.where(jnp.abs(dx) < 1e-12, 1e-12, dx)
        dy = jnp.where(jnp.abs(dy) < 1e-12, 1e-12, dy)
        return RX / dx, -RY / dy
    return precondition


# ═══════════════════════════════════════════════════════════════════════
#  Solver
# ═══════════════════════════════════════════════════════════════════════

def _from_rank0(arrays, mesh):
    """Rank 0's copy of small host arrays, on every process.

    The host solves feed coefficients and control flow, which must be
    identical on every rank; a host LAPACK may round differently from rank to
    rank (it does with the thread count).  The arrays are packed into one
    complex vector, every rank but 0 contributes zeros, and one all-reduce
    (``common.collectives.psum_replicate``) hands rank 0's values to all.
    The shapes depend on the plan only, so the reduction compiles once.  In
    one process, or without a mesh, it is the identity and nothing is sent.
    """
    if mesh is None or process_count() <= 1:
        return arrays
    arrays = [np.asarray(a) for a in arrays]
    flat = np.concatenate([a.astype(np.complex128).reshape(-1) for a in arrays])
    flat = psum_replicate(flat if process_rank() == 0 else np.zeros_like(flat), mesh)
    out, at = [], 0
    for a in arrays:
        part = flat[at:at + a.size].reshape(a.shape)
        at += a.size
        out.append(part if np.iscomplexobj(a) else
                   np.real(part) > 0.5 if a.dtype == bool else np.real(part).astype(a.dtype))
    return out


def basis_capacity(requested, n_block, n_converge, dimension):
    """Real-form basis capacity the solver runs with: the request (default
    ``DEFAULT_CAPACITY_FACTOR * n_converge``), at least ``MIN_CAPACITY_BLOCKS``
    blocks of ``n_block`` rows, at most the real dimension ``2 * dimension``
    of the vector space (padding included).  A caller prices ``2 * capacity``
    vectors with it before the solve."""
    want = int(requested) if requested is not None else DEFAULT_CAPACITY_FACTOR * n_converge
    return min(2 * dimension, max(want, MIN_CAPACITY_BLOCKS * n_block))


def sp_davidson(
    apply_pair: Callable,
    X0,
    *,
    precondition: Callable | None = None,
    data=(),
    n_converge: int | None = None,
    capacity: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    stall_patience: int = 20,
    lock_converged: bool = True,
    margin_lock: float = 1.0,
    on_indefinite: str = "refuse",
    metric_tol: float | None = METRIC_TOL,
    refusal_notes: dict | None = None,
    plan_key=None,
    verbose: bool = True,
):
    """Lowest roots of the full BSE with paired eigenvectors, structure preserving.

    Parameters
    ----------
    apply_pair : (data, X (b, *vs) complex128, s (b,) float64) -> (b, *vs)
        ``A X + s B conj(X)`` per row, real-linear, JAX-traceable.  Called
        with ``s = +1`` and power-of-two ``b`` only.  One row is one pair call.
    X0 : (n_block, *vs) complex128
        Start block, linearly independent; its sharding (a ``NamedSharding``
        with an unsharded row axis, or one device) is the vectors' layout for
        the whole solve.  Entries the operator leaves at zero (mesh padding)
        must be zero here: every later vector is a combination of images and
        preconditioned residuals, so it stays zero there.  The first basis is
        ``[X0, i X0]``: ``2 n_block`` pair calls.
    precondition : (data, RX, RY (n_block, *vs), omega (n_block,) complex128) -> (tX, tY)
        The paired correction of Eq. (6), e.g. :func:`sp_diagonal_precond`.
        Default: :func:`sigma_gradient`.
    data : pytree
        The operator's and the preconditioner's device arrays.  Pass them
        here, not through a closure: a distributed array cannot be captured as
        a compiled constant.
    n_converge : int, optional
        Stop when the lowest ``n_converge`` roots pass; the other
        ``n_block - n_converge`` are margin roots, which keep a multiplet cut
        at the boundary inside the block.  Default: every root.
    capacity : int, optional
        Most real-form basis vectors held; see :func:`basis_capacity`.  Storage is
        ``2 * capacity`` vectors (``W`` and ``M W``).  When the next block
        would not fit, the basis restarts on the real form of the Ritz block.
    max_iter : int
        Iteration cap (one projected solve each).
    tol : float
        A root passes when ``||H z - omega z||_2 / ||z||_2 < tol * max(1, |omega|)``,
        the Tamm-Dancoff solver's rule, in the operator's energy units.
    stall_patience : int
        Stop (``STALLED``) after this many iterations without a 1 % reduction
        of the largest tested residual; 0 disables it.
    lock_converged : bool
        Soft locking: a passing root adds no correction.  It stays in the
        basis and is tested again every iteration.
    margin_lock : float, >= 1
        With locking, a margin root locks at ``margin_lock * tol``.
    on_indefinite : {'refuse', 'continue'}
        What to do when the projected stability matrix ``M_r`` is not positive
        definite (module docstring).
    metric_tol : float or None
        Refuse when a new block's ``Re(P^H M P)`` is asymmetric beyond this
        (relative): the operator is not ``A = A^H`` with ``B = B^T``.
    refusal_notes : dict, optional
        Text appended to a refusal, by gate: ``'definite'`` (GATE
        sp_davidson_definite) and ``'operator'`` (GATE sp_davidson_operator).
        The caller's own remedy: its input key, its energy unit.
    plan_key : hashable, optional
        Reuse the compiled kernels of an earlier call with this key.  The key
        must fix the two programs; shapes, sharding, block and capacity are
        added here.
    verbose : bool
        One summary line at INFO, and one line at INFO when ``on_indefinite='continue'`` meets an
        unstable reference state; each restart is logged at DEBUG.

    Returns
    -------
    omega : (n_block,) float64 host array, ascending; complex128 when unstable
        roots were found (they come first, then real roots ``<= 0``, then the
        positive ones).  Roots the projected pencil does not yet hold are
        ``inf``.
    X, Y : (n_block, *vs) complex128, the layout of ``X0``
        ``X_i^H X_j - Y_i^H Y_j = delta_ij`` for the real roots.  An unstable
        root has ``X^H X - Y^H Y = 0`` and is scaled to ``X^H X + Y^H Y = 1``.
    info : dict
        ``status`` (``STATUS_NAMES``), ``iterations``, ``restarts``,
        ``active_size``, ``capacity``, ``n_block``, ``n_converge``, ``n_tested``,
        ``n_stable_tested`` (the unstable and non-positive roots take block
        rows: ``n_tested = min(n_block, n_special + n_converge)``, so fewer than
        ``n_converge`` positive roots are certified when the block has fewer
        margin rows than special roots; the caller checks this number);
        ``pair_calls`` (total, the start block's ``pair_calls_start``
        included), ``pair_calls_per_root`` (total over ``n_converge``),
        ``root_corrections`` (n_block,) (iterations in which each root asked
        for its two directions);
        ``residuals`` (n_block,), ``converged`` (n_block,) bool;
        ``lam_min`` (lowest eigenvalue of ``M_r`` seen: ``lam_min(K)`` is at or
        below it, and equal once the softest mode is among the roots),
        ``pd_floor``, ``indefinite_at`` (iteration, or None), ``n_unstable``,
        ``n_negative``, ``general_solve`` (iterations whose stable roots came
        from the general solve);
        ``metric_asym``, ``orth_err`` (largest ``|Re(W^H P) - I|`` on a new
        block), ``sigma_err`` (largest ``|X_i^H X_j - Y_i^H Y_j - delta_ij|``
        over the real roots returned), ``seconds``.

    Raises
    ------
    ValueError
        ``GATE sp_davidson_definite`` (``on_indefinite='refuse'``),
        ``GATE sp_davidson_operator`` (``metric_tol``), a rank-deficient start
        block, bad arguments.
    FloatingPointError
        Non-finite operator or preconditioner output.

    Any other stop is returned: read ``info['status']``; only ``CONVERGED``
    certifies the lowest ``n_tested`` roots.
    """
    if on_indefinite not in ("refuse", "continue"):
        raise ValueError(
            f"GATE sp_davidson_argument: on_indefinite is not a known policy (got: {on_indefinite!r}; "
            f"want: 'refuse' or 'continue').  Why: the policy decides what an unstable reference state "
            f"does to the solve.  Fix: pass 'refuse' or 'continue'.")
    if not (np.isfinite(tol) and tol > 0):
        raise ValueError(
            f"GATE sp_davidson_argument: tol is not a residual stop (got: {tol!r}; want: finite and "
            f"> 0).  Why: the stop compares every residual with it.  Fix: pass a positive finite tol.")
    if not float(margin_lock) >= 1.0:
        raise ValueError(
            f"GATE sp_davidson_argument: margin_lock is not a multiple of tol (got: {margin_lock!r}; "
            f"want: >= 1).  Why: margin roots lock at margin_lock * tol, never tighter than the stop.  "
            f"Fix: pass margin_lock >= 1.")
    if int(max_iter) < 1:
        raise ValueError(
            f"GATE sp_davidson_argument: max_iter allows no iteration (got: {max_iter!r}; want: "
            f">= 1).  Why: a solve with no iteration returns the start block.  Fix: pass max_iter >= 1.")
    started = time.perf_counter()
    X0 = jnp.asarray(X0)
    if X0.dtype != jnp.complex128:
        raise TypeError(
            f"GATE sp_davidson_argument: X0 has the wrong dtype (got: {X0.dtype}; want: complex128).  "
            f"Why: the paired form [X, conj(X)] and the projected solve are complex128 throughout.  "
            f"Fix: pass X0 as complex128.")
    nb, vs = int(X0.shape[0]), tuple(int(n) for n in X0.shape[1:])
    nb2, dim = 2 * nb, int(np.prod(vs))
    n_test = nb if n_converge is None else int(n_converge)
    if not 1 <= n_test <= nb:
        raise ValueError(
            f"GATE sp_davidson_argument: n_converge is outside the start block (got: {n_test}; want: "
            f"1 <= n_converge <= {nb}, the rows of X0).  Why: the stop tests that many lowest roots "
            f"of the rows it was given.  Fix: pass n_converge in range, or more start rows.")
    if nb > dim:
        raise ValueError(
            f"GATE sp_davidson_argument: more start vectors than the space holds (got: {nb} rows of "
            f"X0 in dimension {dim}; want: at most {dim}).  Why: the start block [X0, i X0] must "
            f"span {2 * nb} real directions.  Fix: pass fewer start vectors.")
    cap = basis_capacity(capacity, nb, n_test, dim)
    sharding = X0.sharding if isinstance(X0.sharding, NamedSharding) else None
    mesh = None if sharding is None else sharding.mesh
    precondition = sigma_gradient if precondition is None else precondition

    alg = _algebra(nb, cap, vs, sharding)
    full_key = None if plan_key is None else (plan_key, nb, cap, vs, sharding)
    ops = _KERNELS.get(full_key) if full_key is not None else None
    if ops is None:
        ops = _operator_kernels(alg, apply_pair, precondition, nb, vs, sharding)
        if full_key is not None:
            if len(_KERNELS) >= _KERNELS_MAX:
                _KERNELS.pop(next(iter(_KERNELS)))
            _KERNELS[full_key] = ops

    def host(x):
        return np.asarray(gather_to_host(x))

    def device(a):
        """A host array every rank holds, as a replicated operand (no collective)."""
        return jnp.asarray(a) if mesh is None else replicate_to_mesh(np.asarray(a), mesh)

    def apply_rows(Pb, MPb, rank):
        """``M`` on rows ``[0, rank)`` of ``Pb`` in power-of-two pieces."""
        start = 0
        for bit in reversed(range(int(rank).bit_length())):
            width = 1 << bit
            if rank - start >= width:
                MPb = ops.apply(width)(data, Pb, MPb, np.int32(start))
                start += width
        return MPb

    Mr, S = np.zeros((cap, cap)), np.zeros((cap, cap))
    track = dict(metric_asym=0.0, orth_err=0.0)

    def add_panels(pm, ps, m, rank):
        """The new rows and columns of ``M_r = Re(W^H M W)`` and ``S = Im(W^H W)``;
        returns False on a non-finite panel."""
        pm, ps = host(pm)[:m + rank, :rank], host(ps)[:m + rank, :rank]
        if not (np.all(np.isfinite(pm)) and np.all(np.isfinite(ps))):
            return False
        Mr[:m + rank, m:m + rank] = pm.real
        Mr[m:m + rank, :m] = pm.real[:m].T
        S[:m + rank, m:m + rank] = ps.imag
        S[m:m + rank, :m] = -ps.imag[:m].T
        blk = pm.real[m:]                         # Re(P^H M P): symmetric iff A = A^H, B = B^T
        track["metric_asym"] = max(track["metric_asym"], float(
            np.max(np.abs(blk - blk.T)) / max(np.max(np.abs(blk)), 1e-300)))
        gram = ps.real.copy()                     # Re(W^H P) = [0; I]
        gram[m:] -= np.eye(rank)
        track["orth_err"] = max(track["orth_err"], float(np.max(np.abs(gram))))
        return True

    # ── first basis: [X0, i X0] ─────────────────────────────────────────
    status, it, m, restarts = RUNNING, 0, 0, 0
    calls, calls_start = 0, 0
    W, MW = alg.buffers()
    Pb, rank = alg.first(X0, np.int32(min(nb2, cap)))
    rank = int(host(rank))
    MPb = ops.zero_block()
    if rank != nb2:
        status = BAD_INITIAL
    else:
        MPb = apply_rows(Pb, MPb, rank)
        calls = calls_start = rank
        W, MW, pm, ps = alg.extend(W, MW, Pb, MPb, np.int32(0), np.int32(rank))
        if add_panels(pm, ps, 0, rank):
            m = rank
        else:
            status = NONFINITE

    omega = np.full(nb, np.inf, complex)
    resid = np.full(nb, np.inf)
    passes = np.zeros(nb, bool)
    corrections = np.zeros(nb, np.int64)
    X = Y = None
    n_un = n_neg = n_general = 0
    n_tested = n_test
    indefinite_at, lam_min_seen, pd_floor = None, np.inf, 0.0
    best, stalled = np.inf, 0
    while status == RUNNING:
        if metric_tol is not None and track["metric_asym"] > metric_tol:
            status = NOT_SYMMETRIC
            break
        it += 1
        # ── the projected problem on the active block, Eq. (3) ───────────
        A_ = 0.5 * (Mr[:m, :m] + Mr[:m, :m].T)
        S_ = 0.5 * (S[:m, :m] - S[:m, :m].T)
        om, C, present, Qp, counts, reals = _from_rank0(_projected_step(
            A_, S_, nb, cap, solve_indefinite=(on_indefinite == "continue")), mesh)
        definite, n_special, n_un, r, general = (int(n) for n in counts)
        lam_min, pd_floor = (float(x) for x in reals)
        lam_min_seen = min(lam_min_seen, lam_min)
        if not definite:
            if indefinite_at is None:
                indefinite_at = it
                if verbose and on_indefinite == "continue":
                    log.info(f"full-BSE Davidson: the projected stability matrix is not positive "
                             f"definite at iteration {it} (lam_min = {lam_min:.6e}); continuing "
                             "with the unstable sector separated", extra=block("sp_davidson"))
            if on_indefinite == "refuse":
                status = INDEFINITE
                break
        n_general += general
        n_neg = n_special - n_un
        n_tested = min(nb, n_special + n_test)

        # ── Ritz vectors, residuals, the gate, Eqs. (4)-(5) ──────────────
        Cc = np.concatenate([C, np.conj(C)], axis=1)       # (capacity, 2 n_block)
        X, Y, Pc, res = ops.ritz(data, W, MW, device(Cc), device(om), np.int32(m), Pb)
        resid = host(res)
        omega = np.where(present, om, np.inf)
        if not np.all(np.isfinite(resid[present])):
            status = NONFINITE
            break
        gate = tol * np.maximum(1.0, np.abs(om))
        passes = present & (resid < gate)
        if np.all(passes[:n_tested]):
            status = CONVERGED
            break
        if it >= max_iter:
            status = ITERATION_LIMIT
            break
        worst = float(np.max(resid[:n_tested]))
        if worst < 0.99 * best:
            best, stalled = worst, 0
        else:
            stalled += 1
        if stall_patience > 0 and stalled >= stall_patience:
            status = STALLED
            break

        # ── corrections: the paired direction of every open root, Eq. (6)
        open_ = present & (~passes if lock_converged else True)
        if lock_converged and float(margin_lock) != 1.0:
            open_ &= ~((resid < float(margin_lock) * gate) & (np.arange(nb) >= n_tested))
        corrections += open_
        if m + 2 * int(np.sum(open_)) > cap:
            # Restart on the real form of the Ritz block, Eq. (7).
            W, MW = alg.restart(W, MW, device(Qp.astype(complex)), np.int32(m), np.int32(r), Pb)
            Qm = Qp[:m, :r]
            Mr_new, S_new = np.zeros_like(Mr), np.zeros_like(S)
            Mr_new[:r, :r] = Qm.T @ A_ @ Qm
            S_new[:r, :r] = Qm.T @ S_ @ Qm
            Mr, S, m = Mr_new, S_new, r
            restarts += 1
            if verbose:
                log.debug(f"full-BSE Davidson: restart {restarts} at iteration {it}, "
                          f"basis {r} real-form vectors", extra=block("sp_davidson"))
        Pb, rank = alg.ortho(W, Pc, device(np.concatenate([open_, open_])), np.int32(m),
                             np.int32(cap - m))
        rank = int(host(rank))
        if rank == 0:
            status = NO_DIRECTIONS
            break
        MPb = apply_rows(Pb, MPb, rank)
        calls += rank
        W, MW, pm, ps = alg.extend(W, MW, Pb, MPb, np.int32(m), np.int32(rank))
        if not add_panels(pm, ps, m, rank):
            status = NONFINITE
            break
        m += rank

    sigma_err = np.nan
    if X is not None and status not in (NONFINITE, BAD_INITIAL):
        G = host(alg.sigma_gram(X, Y))
        sel = np.where((np.arange(nb) >= n_un) & np.isfinite(omega))[0]
        if sel.size:
            sigma_err = float(np.max(np.abs(G[np.ix_(sel, sel)] - np.eye(sel.size))))
    info = dict(
        status=int(status), status_name=STATUS_NAMES[int(status)], iterations=it,
        restarts=restarts, active_size=m, capacity=cap, n_block=nb, n_converge=n_test,
        n_tested=int(n_tested), n_stable_tested=int(n_tested) - int(n_un) - int(n_neg),
        pair_calls=int(calls), pair_calls_start=int(calls_start),
        pair_calls_per_root=float(calls) / n_test, root_corrections=corrections,
        residuals=resid, converged=passes, lam_min=float(lam_min_seen), pd_floor=float(pd_floor),
        indefinite_at=indefinite_at, n_unstable=int(n_un), n_negative=int(n_neg),
        general_solve=int(n_general), metric_asym=track["metric_asym"],
        orth_err=track["orth_err"], sigma_err=sigma_err,
        seconds=time.perf_counter() - started)
    notes = refusal_notes or {}
    note = {gate: (f"  {notes[gate]}" if notes.get(gate) else "") for gate in ("definite", "operator")}
    if status == BAD_INITIAL:
        raise ValueError(
            f"GATE sp_davidson_start: the start block [X0, i X0] is rank deficient (got: rank "
            f"{rank} of {nb2}; want: full rank).  Why: the first basis is its orthonormalisation, "
            f"and a dependent start vector would leave a direction the solve never sees.  Fix: pass "
            f"linearly independent start vectors (over the reals, X0 and i X0 together).")
    if status == NONFINITE:
        raise FloatingPointError(
            "GATE sp_davidson_finite: the operator or the preconditioner returned a non-finite value "
            "(got: NaN or Inf in the applied block; want: finite output).  Why: every later "
            "projection, Ritz value and residual would be meaningless.  Fix: check the operator "
            "inputs (the kernel and the orbitals) and the preconditioner diagonal for zeros, NaN or "
            "Inf.")
    if status == INDEFINITE:
        raise ValueError(
            f"GATE sp_davidson_definite: the projected stability matrix M_r = Re(W^H M W) is not "
            f"positive definite (got: lam_min = {lam_min_seen:.6e} at iteration {indefinite_at}, "
            f"in the operator's energy units, an upper bound of lam_min(K) taken when the sign "
            f"first showed and not a converged value; want: > {pd_floor:.3e} = {PD_RTOL:g} x its "
            f"largest diagonal entry).  Why: lam_min(K) <= lam_min(M_r) for K = [[A, B], [B*, A*]], so K "
            f"is not positive definite: the reference state is unstable (or marginal to "
            f"rounding), the lowest root is imaginary or not positive, and a definite solve "
            f"would return roots this pencil does not have.  Fix: on_indefinite='continue' "
            f"continues the solve and returns the unstable root(s) with the real ones, or "
            f"change the window or the screening.{note['definite']}")
    if status == NOT_SYMMETRIC:
        raise ValueError(
            f"GATE sp_davidson_operator: Re(P^H M P) of a new block is asymmetric (got: "
            f"{track['metric_asym']:.3e} relative, after {calls} pair calls; want: <= "
            f"{metric_tol:g}).  Why: it is symmetric exactly when A = A^H and B = B^T, the two "
            f"properties the method needs; on any other operator the iteration stalls or "
            f"returns roots of a different problem.  Fix: check the applier "
            f"pair(X, s) = A X + s B conj(X) and its operands.{note['operator']}")
    if verbose:
        ok = np.sum(passes[:n_tested])
        word = ("converged" if status == CONVERGED else
                "STALLED" if status == STALLED else "NOT converged")
        log.info(f"full-BSE Davidson: {word}; {int(ok)}/{int(n_tested)} roots, {it} iterations, "
                 f"{calls} pair calls ({calls / n_test:.1f} per root), {restarts} restarts; "
                 f"max residual={np.max(resid[:n_tested]):.3e}, lam_min={lam_min_seen:.3e}",
                 extra=block("sp_davidson"))
    out_omega = omega if n_un else np.real(omega)
    return out_omega, X, Y, info
