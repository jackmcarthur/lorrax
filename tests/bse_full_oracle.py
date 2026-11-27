"""Dense oracle and synthetic pencils for the full (non-Tamm-Dancoff) BSE eigen routes.

Importable by any solver test (``from tests.bse_full_oracle import ...``); it imports
only NumPy and nothing from ``src/``, so it is an independent reference for every
route that does.

The full BSE eigenproblem on a window of ``N`` transitions is

    H z = omega z,    H = [[A, B], [-B*, -A*]],    z = [X; Y],

with ``A = A^H`` (resonant block) and ``B = B^T`` (coupling block), both ``N x N`` complex
and energies in Ry.  Equivalently ``K z = omega Sigma z`` with ``Sigma = diag(1, -1)`` and
``K = [[A, B], [B*, A*]]`` Hermitian.  When ``K`` is positive definite every root is real, the
roots come in pairs ``+-omega`` (the partner of ``[X; Y]`` is ``[Y*; X*]``), and the physical
eigenvectors are normalised ``X^H X - Y^H Y = +1``.  A not positive definite ``K`` gives
imaginary root pairs (a triplet or charge instability): the oracle reports them, it does not
hide them.

What is here

* ``assemble_dense_ab(pair, window_shape)``: ``A`` and ``B`` from identity columns of a pair
  applier ``pair(X, s) = A X + s B conj(X)`` (``X`` of shape ``(b, *window_shape)``); the
  identity columns are real, so ``pair(E, +1) = (A + B) E`` and ``pair(E, -1) = (A - B) E``.
* ``structure_errors(A, B)``: ``|A - A^H| / |A|`` and ``|B - B^T| / |B|`` (Frobenius).
* ``solve_structure_preserving(A, B)``: the dense Cholesky route, positive definite ``K`` only.
* ``solve_brute_force(A, B)``: ``numpy.linalg.eig`` of ``H``, positive real roots grouped by
  degeneracy and Sigma-orthonormalised inside each group; reports imaginary pairs.
* ``reference_roots(A, B, n_eig=None)``: the one entry a solver test calls.
* Residual, normalisation and subspace-distance measures for comparing another solver's
  ``(omega, X, Y)`` with the reference: ``eigpair_residuals``, ``sigma_norms``,
  ``sigma_gram``, ``degenerate_groups``, ``sigma_projector_distance``.
* Synthetic pencils with BSE structure: ``dominant_pencil``, ``near_singular_pencil``,
  ``indefinite_pencil``, ``degenerate_pencil``, ``clustered_pencil`` and the list
  ``standard_pencils``.  Each is a dict with ``A``, ``B``, ``name``, ``definite``,
  ``n_imag_pairs`` and, where an analytic value exists, ``omega_exact`` (ascending positive
  real roots, from a route independent of the oracle's two solvers).

Vectors are columns: ``X`` and ``Y`` are ``(N, n_roots)``.
"""
import numpy as np

__all__ = [
    "pair_from_dense", "assemble_dense_ab", "structure_errors", "assert_bse_structure",
    "bse_K", "bse_H", "solve_structure_preserving", "solve_brute_force", "reference_roots",
    "eigpair_residuals", "sigma_norms", "sigma_gram", "degenerate_groups",
    "sigma_projector_distance", "rotate_unitary", "random_unitary",
    "dominant_pencil", "near_singular_pencil", "indefinite_pencil", "degenerate_pencil",
    "clustered_pencil", "standard_pencils",
    "general_K", "general_apply", "solve_general_pencil", "general_residuals", "mode_roots",
    "general_mode_pencil", "dominant_general_pencil",
]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def bse_H(A, B):
    """``H = [[A, B], [-B*, -A*]]`` (2N x 2N), the non-Hermitian full-BSE matrix."""
    A, B = np.asarray(A), np.asarray(B)
    return np.block([[A, B], [-B.conj(), -A.conj()]])


def bse_K(A, B):
    """``K = [[A, B], [B*, A*]]`` (2N x 2N), Hermitian when ``A = A^H`` and ``B = B^T``."""
    A, B = np.asarray(A), np.asarray(B)
    return np.block([[A, B], [B.conj(), A.conj()]])


def pair_from_dense(A, B):
    """The pair applier ``pair(X, s) = A X + s B conj(X)`` of dense ``A`` and ``B``.

    Parameters
    ----------
    A, B : (N, N) complex
        Resonant and coupling blocks.

    Returns
    -------
    pair : callable
        ``pair(X, s)`` with ``X`` of shape ``(b, *window)`` (``prod(window) = N``) and ``s``
        a real scalar; returns an array of the same shape.  It is real-linear in ``X``, not
        complex-linear, like the production applier.
    """
    A, B = np.asarray(A), np.asarray(B)

    def pair(X, s):
        X = np.asarray(X)
        flat = X.reshape(X.shape[0], -1)
        return (flat @ A.T + float(s) * (flat.conj() @ B.T)).reshape(X.shape)

    return pair


def assemble_dense_ab(pair, window_shape, *, chunk=64):
    """Assemble dense ``A`` and ``B`` from identity columns of a pair applier.

    Parameters
    ----------
    pair : callable
        ``pair(X, s) = A X + s B conj(X)`` with ``X`` of shape ``(b, *window_shape)`` and
        ``s`` a real scalar; the result must convert with ``numpy.asarray`` (a device or
        sharded array is fine).
    window_shape : tuple of int
        Shape of one vector; ``N = prod(window_shape)``.  The flat index is C order.
    chunk : int
        Identity columns per call.

    Returns
    -------
    A, B : (N, N) complex128
        ``A = (F + G) / 2`` and ``B = (F - G) / 2`` with ``F = pair(E, +1)`` and
        ``G = pair(E, -1)``; exact because the identity columns ``E`` are real.
    """
    window_shape = tuple(int(n) for n in window_shape)
    N = int(np.prod(window_shape))
    eye = np.eye(N, dtype=np.complex128).reshape((N,) + window_shape)
    F = np.empty((N, N), dtype=np.complex128)
    G = np.empty((N, N), dtype=np.complex128)
    for c0 in range(0, N, int(chunk)):
        blk = eye[c0:c0 + int(chunk)]
        b = blk.shape[0]
        F[:, c0:c0 + b] = np.asarray(pair(blk, 1.0)).reshape(b, N).T
        G[:, c0:c0 + b] = np.asarray(pair(blk, -1.0)).reshape(b, N).T
    return 0.5 * (F + G), 0.5 * (F - G)


def structure_errors(A, B):
    """Relative Frobenius departures ``(|A - A^H| / |A|, |B - B^T| / |B|)``.

    A zero block has error 0.  These are the two conditions the full-BSE solvers rely on
    (the kappa metric ``Re(x^H F(x'))`` is symmetric iff both hold).
    """
    A, B = np.asarray(A), np.asarray(B)
    nA, nB = np.linalg.norm(A), np.linalg.norm(B)
    herm = np.linalg.norm(A - A.conj().T) / nA if nA > 0 else 0.0
    sym = np.linalg.norm(B - B.T) / nB if nB > 0 else 0.0
    return float(herm), float(sym)


def assert_bse_structure(A, B, tol=1e-12):
    """Raise ``AssertionError`` unless ``A = A^H`` and ``B = B^T`` to ``tol`` (relative)."""
    herm, sym = structure_errors(A, B)
    assert herm <= tol, f"A is not Hermitian: |A - A^H|/|A| = {herm:.3e} > {tol:g}"
    assert sym <= tol, f"B is not complex symmetric: |B - B^T|/|B| = {sym:.3e} > {tol:g}"


# ---------------------------------------------------------------------------
# Reference solvers
# ---------------------------------------------------------------------------

def solve_structure_preserving(A, B, *, pd_rtol=1e-10):
    """Dense structure-preserving solve of ``K z = omega Sigma z`` (K positive definite).

    With ``K = L L^H`` and ``M = L^-1 Sigma L^-H`` (Hermitian), ``M y = (1/omega) y``; the
    positive eigenvalues of ``M`` are the physical roots and ``z = L^-H y / sqrt(1/omega)``
    is Sigma-normalised, ``X^H X - Y^H Y = +1``.  Degenerate roots come out Sigma-orthonormal
    because ``eigh`` returns orthonormal ``y``.

    Raises
    ------
    ValueError
        If ``min eig(K) <= pd_rtol * max eig(K)`` (imaginary roots, or a singular ``K``).

    Returns
    -------
    dict
        ``omega`` (N,) ascending, ``X`` and ``Y`` (N, N) columns, ``k_min`` (smallest
        eigenvalue of K).
    """
    A, B = np.asarray(A, dtype=np.complex128), np.asarray(B, dtype=np.complex128)
    N = A.shape[0]
    K = bse_K(A, B)
    K = 0.5 * (K + K.conj().T)
    kw = np.linalg.eigvalsh(K)
    if kw[0] <= pd_rtol * kw[-1]:
        raise ValueError(f"K = [[A,B],[B*,A*]] is not positive definite: min eig {kw[0]:.3e}, "
                         f"max eig {kw[-1]:.3e}")
    L = np.linalg.cholesky(K)
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    Linv_S = np.linalg.solve(L, np.diag(sig).astype(np.complex128))      # L^-1 Sigma
    M = np.linalg.solve(L, Linv_S.conj().T)                               # L^-1 Sigma L^-H
    M = 0.5 * (M + M.conj().T)
    mu, Y = np.linalg.eigh(M)
    idx = np.where(mu > 0)[0]
    idx = idx[np.argsort(-mu[idx])]                                       # largest 1/omega first
    if idx.size != N:
        raise ValueError(f"expected {N} positive roots, found {idx.size}")
    Z = np.linalg.solve(L.conj().T, Y[:, idx]) / np.sqrt(mu[idx])[None, :]
    return {"omega": 1.0 / mu[idx], "X": Z[:N], "Y": Z[N:], "k_min": float(kw[0])}


def degenerate_groups(omega, rtol=1e-9):
    """Index groups of (numerically) degenerate roots, in ascending order of ``omega``.

    Two neighbours (after sorting) are in one group when they differ by at most
    ``rtol * max(1, |omega|)``.
    """
    omega = np.asarray(omega, dtype=float)
    order = np.argsort(omega)
    groups, cur = [], [int(order[0])]
    for i in order[1:]:
        if abs(omega[i] - omega[cur[-1]]) <= rtol * max(1.0, abs(omega[i])):
            cur.append(int(i))
        else:
            groups.append(cur)
            cur = [int(i)]
    return groups + [cur]


def solve_brute_force(A, B, *, real_rtol=1e-9, group_rtol=1e-9):
    """``numpy.linalg.eig`` of ``H = [[A,B],[-B*,-A*]]``; positive real roots, Sigma-normalised.

    Roots with ``|Im| <= real_rtol * max|lambda|`` are real; the positive ones are the
    physical roots.  Each degenerate group (``group_rtol``) is Sigma-orthonormalised with
    ``Z <- Z G^(-1/2)``, ``G = Z^H Sigma Z`` (Hermitian and positive for a positive-signature
    group), so the result is basis-independent inside a group.  Complex roots are not
    normalised; their positive imaginary parts are returned in ``imag``.

    Returns
    -------
    dict
        ``omega`` (n,) ascending, ``X`` and ``Y`` (N, n), ``imag`` (n_imag,) ascending
        positive imaginary parts of the complex-pair roots, ``n_real_negative`` (the count of
        negative real roots, which equals ``len(omega)`` for a definite ``K``).
    """
    A, B = np.asarray(A, dtype=np.complex128), np.asarray(B, dtype=np.complex128)
    N = A.shape[0]
    lam, V = np.linalg.eig(bse_H(A, B))
    scale = float(np.max(np.abs(lam)))
    is_real = np.abs(lam.imag) <= real_rtol * scale
    pos = np.where(is_real & (lam.real > 0))[0]
    neg = np.where(is_real & (lam.real < 0))[0]
    imag = np.sort(lam.imag[(~is_real) & (lam.imag > 0)])
    order = pos[np.argsort(lam.real[pos])]
    omega = lam.real[order]
    Z = V[:, order]
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    if omega.size:
        for g in degenerate_groups(omega, group_rtol):
            Zg = Z[:, g]
            G = Zg.conj().T @ (sig[:, None] * Zg)
            G = 0.5 * (G + G.conj().T)
            w, U = np.linalg.eigh(G)
            if w[0] <= 0:
                raise ValueError("a positive root has non-positive Sigma norm (wrong Krein "
                                 f"signature or a defective cluster): min {w[0]:.3e}")
            Z[:, g] = Zg @ ((U / np.sqrt(w)[None, :]) @ U.conj().T)
    return {"omega": omega, "X": Z[:N], "Y": Z[N:], "imag": imag,
            "n_real_negative": int(neg.size)}


def reference_roots(A, B, n_eig=None, *, pd_rtol=1e-10):
    """Reference positive roots with ``X^H X - Y^H Y = 1``: the call a solver test makes.

    Uses the structure-preserving solve when ``K`` is positive definite and the brute-force
    ``eig`` otherwise (so a pencil with an imaginary pair still has its real roots and the
    imaginary parts reported).

    Parameters
    ----------
    A, B : (N, N) complex
        Blocks with ``A = A^H`` and ``B = B^T`` (checked to 1e-8; raises ``ValueError``).
    n_eig : int, optional
        Keep the lowest ``n_eig`` roots (default all).
    pd_rtol : float
        Positive-definiteness threshold on ``min eig(K) / max eig(K)``.

    Returns
    -------
    dict
        ``omega`` (n,), ``X`` and ``Y`` (N, n), ``definite`` (bool), ``imag`` (ascending
        positive imaginary parts of complex-pair roots, empty when definite), ``route``
        (``'structure_preserving'`` or ``'brute_force'``).
    """
    herm, sym = structure_errors(A, B)
    if herm > 1e-8 or sym > 1e-8:
        raise ValueError(f"not BSE-structured: |A-A^H|/|A| = {herm:.2e}, |B-B^T|/|B| = {sym:.2e}")
    try:
        out = solve_structure_preserving(A, B, pd_rtol=pd_rtol)
        out.update(definite=True, imag=np.zeros(0), route="structure_preserving")
    except ValueError:
        out = solve_brute_force(A, B)
        out.update(definite=False, route="brute_force")
    if n_eig is not None:
        n = int(n_eig)
        out["omega"], out["X"], out["Y"] = out["omega"][:n], out["X"][:, :n], out["Y"][:, :n]
    return out


# ---------------------------------------------------------------------------
# Measures for comparing a solver's output with the reference
# ---------------------------------------------------------------------------

def eigpair_residuals(A, B, omega, X, Y):
    """Relative residual of each ``(omega_i, [X_i; Y_i])`` of ``H z = omega z``.

    ``|H z - omega z| / (|z| |H|_2)`` per column; ``X`` and ``Y`` are ``(N, n)`` columns.
    """
    H = bse_H(A, B)
    Z = np.concatenate([np.asarray(X), np.asarray(Y)], axis=0)
    R = H @ Z - Z * np.asarray(omega)[None, :]
    return np.linalg.norm(R, axis=0) / (np.linalg.norm(Z, axis=0) * np.linalg.norm(H, 2))


def sigma_norms(X, Y):
    """``X^H X - Y^H Y`` per column (real, length n); 1 for a normalised physical root."""
    X, Y = np.asarray(X), np.asarray(Y)
    return np.real(np.sum(X.conj() * X, axis=0) - np.sum(Y.conj() * Y, axis=0))


def sigma_gram(X, Y):
    """``Z^H Sigma Z = X^H X - Y^H Y`` as an (n, n) matrix; the identity for a Sigma-orthonormal set."""
    X, Y = np.asarray(X), np.asarray(Y)
    return X.conj().T @ X - Y.conj().T @ Y


def sigma_projector_distance(Xa, Ya, Xb, Yb):
    """Frobenius distance of the Sigma-projectors of two sets spanning one eigenspace.

    ``P = Z (Z^H Sigma Z)^-1 Z^H Sigma``.  It does not depend on the basis chosen inside the
    space, so it compares degenerate or clustered roots between two solvers.  Zero for equal
    spans; ``sqrt(2 r)`` for orthogonal spans of dimension ``r``.
    """
    def proj(X, Y):
        Z = np.concatenate([np.asarray(X), np.asarray(Y)], axis=0)
        sig = np.concatenate([np.ones(np.asarray(X).shape[0]), -np.ones(np.asarray(Y).shape[0])])
        SZ = sig[:, None] * Z
        return Z @ np.linalg.solve(Z.conj().T @ SZ, SZ.conj().T)
    return float(np.linalg.norm(proj(Xa, Ya) - proj(Xb, Yb)))


# ---------------------------------------------------------------------------
# Synthetic BSE-structured pencils
# ---------------------------------------------------------------------------

def random_unitary(n, rng):
    """Haar-random ``n x n`` unitary (QR of a complex Gaussian with the diagonal phases fixed)."""
    Q, R = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return Q * (np.diag(R) / np.abs(np.diag(R)))[None, :]


def rotate_unitary(A, B, U):
    """``A -> U A U^H``, ``B -> U B U^T``: the roots are unchanged (``X -> U X``, ``Y -> U* Y``).

    Keeps ``A = A^H``, ``B = B^T`` and positive definiteness of ``K``; makes the blocks dense
    and complex so a solver cannot lean on a diagonal or real structure.
    """
    return U @ A @ U.conj().T, U @ B @ U.T


def _pencil(name, A, B, *, omega_exact=None, definite=True, n_imag_pairs=0, note=""):
    K = bse_K(A, B)
    k_min = float(np.linalg.eigvalsh(0.5 * (K + K.conj().T))[0])
    return {"name": name, "A": A, "B": B, "omega_exact": omega_exact, "definite": definite,
            "n_imag_pairs": n_imag_pairs, "k_min": k_min, "n": A.shape[0], "note": note}


def _real_pm_pencil(n, seed, lam_min, name):
    """Real symmetric ``P = A + B`` (well conditioned) and ``M = A - B`` with smallest eigenvalue
    ``lam_min``, rotated by a random unitary.  ``omega^2`` are the eigenvalues of
    ``P^(1/2) M P^(1/2)`` (symmetric, independent of the structure-preserving solve)."""
    rng = np.random.default_rng(seed)
    q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.concatenate([[lam_min], 0.3 + 0.7 * rng.random(n - 1)])
    mu = 0.5 + rng.random(n)
    M = (q1 * lam[None, :]) @ q1.T
    P = (q2 * mu[None, :]) @ q2.T
    ph = (q2 * np.sqrt(mu)[None, :]) @ q2.T
    w2 = np.linalg.eigvalsh(ph @ M @ ph)
    A, B = 0.5 * (P + M), 0.5 * (P - M)
    return rotate_unitary(A, B, random_unitary(n, rng)), w2


def dominant_pencil(n=48, seed=0, *, gap=0.4, window=0.6, coupling=0.02):
    """Diagonally dominant ``A`` with a gap and small ``B`` (the optical-spectrum regime).

    ``A = diag(e) + coupling * (C + C^H) / (2 sqrt(n))`` with ``e`` in ``[gap, gap + window]``
    (Ry), ``B = coupling * (S + S^T) / (2 sqrt(n))``, ``C`` and ``S`` complex Gaussian; no
    analytic roots.  Positive definite by construction (``|B|_2 << gap``).
    """
    rng = np.random.default_rng(seed)
    cplx = lambda: rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))  # noqa: E731
    e = gap + window * np.sort(rng.random(n))
    C, S = cplx(), cplx()
    A = np.diag(e) + coupling * (C + C.conj().T) / (2 * np.sqrt(n))
    B = coupling * (S + S.T) / (2 * np.sqrt(n))
    return _pencil("dominant", A, B, note=f"gap {gap}, coupling {coupling}")


def near_singular_pencil(n=40, seed=1, *, eps=1e-4):
    """``A - B`` with smallest eigenvalue ``eps`` (a soft, nearly unstable triplet-like mode).

    Real ``P = A + B`` and ``M = A - B`` rotated by a unitary; lowest ``omega`` is of order
    ``sqrt(eps)``.  ``omega_exact`` from ``eigvalsh(P^(1/2) M P^(1/2))``.
    """
    (A, B), w2 = _real_pm_pencil(n, seed, eps, "near_singular")
    return _pencil("near_singular", A, B, omega_exact=np.sqrt(np.sort(w2)),
                   note=f"lambda_min(A - B) = {eps:g}")


def indefinite_pencil(n=40, seed=2, *, delta=1e-3):
    """Slightly indefinite: ``A - B`` has one eigenvalue ``-delta``, so one imaginary pair.

    ``K`` is then not positive definite, ``omega^2`` has exactly one negative value and the
    spectrum holds ``+-i sqrt(|omega^2|)``.  ``omega_exact`` lists the ``n - 1`` positive real
    roots; ``imag_exact`` the one imaginary part.
    """
    (A, B), w2 = _real_pm_pencil(n, seed, -delta, "indefinite")
    w2 = np.sort(w2)
    assert w2[0] < 0 < w2[1], "indefinite_pencil: expected exactly one negative omega^2"
    out = _pencil("indefinite", A, B, omega_exact=np.sqrt(w2[1:]), definite=False,
                  n_imag_pairs=1, note=f"lambda_min(A - B) = {-delta:g}")
    out["imag_exact"] = np.sqrt(-w2[:1])
    return out


def _diag_pencil(a, b, seed, name, note):
    """Rotated ``A = diag(a)``, ``B = diag(b)``: roots ``sqrt(a^2 - |b|^2)`` exactly."""
    rng = np.random.default_rng(seed)
    n = a.size
    A, B = rotate_unitary(np.diag(a).astype(complex), np.diag(b).astype(complex),
                          random_unitary(n, rng))
    w = np.sqrt(a ** 2 - np.abs(b) ** 2)
    return _pencil(name, A, B, omega_exact=np.sort(w), note=note)


def degenerate_pencil(n_groups=6, mult=3, seed=3):
    """Exactly degenerate roots: ``n_groups`` distinct roots, each of multiplicity ``mult``.

    ``A = diag(a)`` and ``B = diag(b)`` with ``a`` and ``b`` constant on each group, rotated
    by a random unitary.  ``omega_g = sqrt(a_g^2 - |b_g|^2)`` exactly; ``N = n_groups * mult``.
    """
    rng = np.random.default_rng(seed)
    ag = 0.4 + 0.1 * np.arange(n_groups) + 0.02 * rng.random(n_groups)
    bg = 0.03 * (rng.standard_normal(n_groups) + 1j * rng.standard_normal(n_groups))
    return _diag_pencil(np.repeat(ag, mult), np.repeat(bg, mult), seed, "degenerate",
                        f"{n_groups} roots x multiplicity {mult}")


def clustered_pencil(n_clusters=6, size=4, seed=4, *, spread=1e-7):
    """Clusters of ``size`` roots separated by ``spread`` Ry inside a cluster, 0.1 Ry apart.

    Same construction as ``degenerate_pencil`` with the in-cluster energies split by
    ``spread``; ``omega`` known exactly, each cluster an eigenspace to compare by projector.
    """
    rng = np.random.default_rng(seed)
    centres = 0.4 + 0.1 * np.arange(n_clusters)
    a = (centres[:, None] + spread * np.arange(size)[None, :]).reshape(-1)
    b = np.repeat(0.03 * (rng.standard_normal(n_clusters) + 1j * rng.standard_normal(n_clusters)),
                  size)
    return _diag_pencil(a, b, seed, "clustered", f"{n_clusters} clusters x {size}, spread {spread:g} Ry")


# ---------------------------------------------------------------------------
# The general pencil: no conjugation symmetry (a finite pair momentum Q)
# ---------------------------------------------------------------------------
#
#     K z = omega Sigma z,   K = [[A, B], [B^H, A_low]] = K^H,   Sigma = diag(1, -1).
#
# At Q = 0, A_low = conj(A) and B = B^T and this is the pencil above.  In general the roots are
# omega_i(Q) > 0 with z^H Sigma z > 0 and -omega_i(-Q) < 0 with z^H Sigma z < 0: no +- pairs.

def general_K(A, B, A_low):
    """``K = [[A, B], [B^H, A_low]]`` (2N x 2N); Hermitian when ``A`` and ``A_low`` are."""
    A, B, A_low = np.asarray(A), np.asarray(B), np.asarray(A_low)
    return np.block([[A, B], [B.conj().T, A_low]])


def general_apply(K):
    """The applier ``Z (b, 2, *window) -> K Z`` of a dense ``K`` (rows are pairs ``[X; Y]``)."""
    K = np.asarray(K)

    def apply(Z):
        Z = np.asarray(Z)
        return (Z.reshape(Z.shape[0], -1) @ K.T).reshape(Z.shape)

    return apply


def solve_general_pencil(K, *, real_rtol=1e-9, group_rtol=1e-9):
    """``numpy.linalg.eig`` of ``Sigma K``, every root classified by type and sign.

    A root is real when ``|Im| <= real_rtol * max|lambda|``; its type is the sign of
    ``z^H Sigma z``.  Returns a dict:

    ``omega`` (n,) ascending, ``X``, ``Y`` (N, n): the physical roots, real, positive, of positive
        type, Sigma-orthonormalised inside each degenerate group (``z_i^H Sigma z_j = delta_ij``);
    ``complex`` : roots with ``Im > 0``, descending ``Im`` (each has a conjugate partner);
    ``negative`` : real roots of positive type at ``omega <= 0``, ascending;
    ``mirror`` : real roots of negative type at ``omega >= 0``, ascending (non-positive roots of
        the ``-Q`` problem);
    ``k_min`` : the smallest eigenvalue of ``K``; ``definite`` : ``k_min > 0`` to 1e-10 relative.
    """
    K = np.asarray(K, dtype=np.complex128)
    N = K.shape[0] // 2
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    kw = np.linalg.eigvalsh(0.5 * (K + K.conj().T))
    lam, V = np.linalg.eig(sig[:, None] * K)
    scale = float(np.max(np.abs(lam)))
    is_real = np.abs(lam.imag) <= real_rtol * scale
    typ = np.real(np.einsum("ij,ij->j", V.conj(), sig[:, None] * V))
    pos = np.where(is_real & (typ > 0) & (lam.real > 0))[0]
    order = pos[np.argsort(lam.real[pos])]
    omega, Z = lam.real[order], V[:, order]
    if omega.size:
        for g in degenerate_groups(omega, group_rtol):
            Zg = Z[:, g]
            G = Zg.conj().T @ (sig[:, None] * Zg)
            w, U = np.linalg.eigh(0.5 * (G + G.conj().T))
            if w[0] <= 0:
                raise ValueError(f"a positive-type group has a non-positive Sigma norm: min {w[0]:.3e}")
            Z[:, g] = Zg @ ((U / np.sqrt(w)[None, :]) @ U.conj().T)
    cplx = lam[(~is_real) & (lam.imag > 0)]
    return {"omega": omega, "X": Z[:N], "Y": Z[N:],
            "complex": cplx[np.argsort(-cplx.imag)],
            "negative": np.sort(lam.real[is_real & (typ > 0) & (lam.real <= 0)]),
            "mirror": np.sort(lam.real[is_real & (typ < 0) & (lam.real >= 0)]),
            "k_min": float(kw[0]), "definite": bool(kw[0] > 1e-10 * kw[-1])}


def general_residuals(K, omega, X, Y):
    """``|Sigma K z - omega z| / (|z| |K|_2)`` per column of ``z = [X; Y]``."""
    K = np.asarray(K)
    N = K.shape[0] // 2
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    Z = np.concatenate([np.asarray(X), np.asarray(Y)], axis=0)
    R = sig[:, None] * (K @ Z) - Z * np.asarray(omega)[None, :]
    return np.linalg.norm(R, axis=0) / (np.linalg.norm(Z, axis=0) * np.linalg.norm(K, 2))


def mode_roots(a, a_low, b):
    """The two roots of each 2 x 2 pencil ``[[a, b], [conj b, a_low]] z = omega sigma_z z``:
    ``omega = (a - a_low)/2 +- sqrt(((a + a_low)/2)^2 - |b|^2)``.  Returns ``(plus, minus)``
    complex arrays; for real roots ``plus`` is the one of positive type."""
    a, a_low, b = (np.asarray(v) for v in (a, a_low, b))
    root = np.sqrt((0.25 * (a + a_low) ** 2 - np.abs(b) ** 2).astype(complex))
    return 0.5 * (a - a_low) + root, 0.5 * (a - a_low) - root


def general_mode_pencil(a, a_low, b, seed=0):
    """A direct sum of 2 x 2 pencils rotated by ``diag(U1, U2)`` with random unitaries (a rotation
    that commutes with Sigma): dense, complex, no symmetry between the blocks, roots known exactly
    (:func:`mode_roots`).  Returns ``(A, B, A_low)`` with ``A = U1 diag(a) U1^H``,
    ``B = U1 diag(b) U2^H``, ``A_low = U2 diag(a_low) U2^H``."""
    rng = np.random.default_rng(seed)
    a, a_low, b = (np.asarray(v) for v in (a, a_low, b))
    U1, U2 = random_unitary(a.size, rng), random_unitary(a.size, rng)
    return ((U1 * a[None, :]) @ U1.conj().T, (U1 * b[None, :]) @ U2.conj().T,
            (U2 * a_low[None, :]) @ U2.conj().T)


def dominant_general_pencil(n=48, seed=0, *, gap=0.4, window=0.6, coupling=0.02, low_shift=0.05):
    """Diagonally dominant blocks with a gap and a small coupling without symmetry: the regime of
    a finite-Q exciton solve.  ``A = diag(e) + c (C + C^H)/(2 sqrt n)``, ``A_low = diag(e') + c
    (C' + C'^H)/(2 sqrt n)`` with independent energies ``e'`` in ``[gap + low_shift, ...]``, and
    ``B = c S / sqrt(n)`` a general complex matrix.  Returns ``(A, B, A_low)``; positive definite
    by construction (``|B|_2 << gap``)."""
    rng = np.random.default_rng(seed)
    cplx = lambda: rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))  # noqa: E731
    e = gap + window * np.sort(rng.random(n))
    e_low = gap + low_shift + window * np.sort(rng.random(n))
    C, C_low, S = cplx(), cplx(), cplx()
    A = np.diag(e) + coupling * (C + C.conj().T) / (2 * np.sqrt(n))
    A_low = np.diag(e_low) + coupling * (C_low + C_low.conj().T) / (2 * np.sqrt(n))
    return A, coupling * S / (2 * np.sqrt(n)), A_low


def standard_pencils(seed=0):
    """The five synthetic pencils at their default sizes, as a list of dicts.

    Definite ones first (``dominant``, ``near_singular``, ``degenerate``, ``clustered``),
    then ``indefinite``.  ``seed`` shifts every generator's seed.
    """
    return [dominant_pencil(seed=seed), near_singular_pencil(seed=seed + 1),
            degenerate_pencil(seed=seed + 3), clustered_pencil(seed=seed + 4),
            indefinite_pencil(seed=seed + 2)]
