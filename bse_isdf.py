"""Haydock/Lanczos diagonalization demo for the Bethe--Salpeter equation.

The BSE Hamiltonian is Hermitian but far too large to form explicitly in
real calculations.  A practical solver therefore only needs a routine
that applies the matrix to a trial vector.  The Haydock recursion
produces a Krylov basis from repeated applications of this routine.  In
that basis the Hamiltonian reduces to a tridiagonal matrix whose
eigenpairs approximate those of the full operator.  By mapping the
tridiagonal eigenvectors back to the Krylov basis we obtain
approximations to the desired eigenvectors.

This file illustrates the procedure on a small dense matrix.  The code
is written with ``gpu_utils`` so that all operations run on either NumPy
or CuPy.  The only unavoidable Python loop is over the number of Krylov
iterations.  Expensive steps such as reorthogonalization are expressed as
matrix--vector products so that they execute efficiently on the GPU.
In a full implementation ``apply_matrix_to_vector`` would call a
specialised low-rank kernel.
"""

from gpu_utils import cp, xp


def apply_matrix_to_vector(mat, vec):
    """Apply the matrix ``mat`` to ``vec``."""
    return xp.matmul(mat, vec)


def haydock_eig(mat, n_eig, max_iter=40):
    """Compute ``n_eig`` lowest eigenpairs of Hermitian ``mat``.

    Parameters
    ----------
    mat : xp.ndarray
        Hermitian matrix.
    n_eig : int
        Number of eigenvalues to compute.
    max_iter : int, optional
        Size of the Lanczos/Haydock basis.
    """
    n = mat.shape[0]
    rng = xp.random.default_rng(0)
    q = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    xp.divide(q, xp.linalg.norm(q), out=q)

    Q = xp.zeros((n, max_iter + 1), dtype=mat.dtype)
    alpha = xp.zeros(max_iter, dtype=mat.real.dtype)
    beta = xp.zeros(max_iter, dtype=mat.real.dtype)
    Q[:, 0] = q

    actual_iter = max_iter
    for j in range(max_iter):
        z = apply_matrix_to_vector(mat, q)
        alpha[j] = xp.vdot(q, z).real
        if j > 0:
            z -= beta[j - 1] * Q[:, j - 1]
        z -= alpha[j] * q

        # Full reorthogonalization using matrix--vector products
        if j > 0:
            proj = xp.matmul(Q[:, :j].conj().T, z)
            z -= xp.matmul(Q[:, :j], proj)

        beta[j] = xp.linalg.norm(z)
        if beta[j] < 1e-12 or j == max_iter - 1:
            actual_iter = j + 1
            break
        q = xp.divide(z, beta[j])
        Q[:, j + 1] = q

    T = xp.diag(alpha[:actual_iter])
    if actual_iter > 1:
        off = beta[: actual_iter - 1]
        T += xp.diag(off, k=1) + xp.diag(off, k=-1)

    evals_T, vecs_T = xp.linalg.eigh(T)
    idx = xp.argsort(evals_T)
    evals = evals_T[idx][:n_eig]
    vecs = xp.matmul(Q[:, :actual_iter], vecs_T[:, idx[:n_eig]])
    return evals, vecs


def main():
    n = 100
    n_eig = 5
    rng = xp.random.default_rng(42)
    mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    mat = 0.5 * (mat + mat.conj().T)

    evals_hay, evecs_hay = haydock_eig(mat, n_eig)
    evals_exact, evecs_exact = xp.linalg.eigh(mat)
    idx = xp.argsort(evals_exact)
    evals_exact = evals_exact[idx][:n_eig]
    evecs_exact = evecs_exact[:, idx[:n_eig]]

    print("Haydock eigenvalues:", evals_hay)
    print("Exact eigenvalues:", evals_exact)
    diff = xp.abs(evals_hay - evals_exact)
    print("Max abs difference:", diff.max())

    # Check orthonormality of Haydock eigenvectors
    overlap = xp.matmul(evecs_hay.conj().T, evecs_hay) - xp.eye(n_eig)
    print("Orthonormality error:", xp.linalg.norm(overlap))


if __name__ == "__main__":
    main()
