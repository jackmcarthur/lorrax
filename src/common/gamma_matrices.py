import jax
import jax.numpy as jnp

"""Dirac gamma matrices represented as JAX arrays.

The matrices are provided in the standard Dirac representation
and stored as ``jax.numpy`` arrays.
"""

# Pauli matrices
sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

# Standard Dirac gamma matrices (4x4)
# JM: actually I replace gamma0-3 with gamma0*gamma0-3, so that I can use psidag = conj(psi) rather than psibar = conj(psi) gamma0

gamma0 = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=jnp.complex128)

gamma1 = jnp.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=jnp.complex128)

gamma2 = jnp.array([[0, 0, 0, -1j],
                   [0, 0, 1j, 0],
                   [0, -1j, 0, 0],
                   [1j, 0, 0, 0]], dtype=jnp.complex128)

gamma3 = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, -1],
                   [1, 0, 0, 0],
                   [0, -1, 0, 0]], dtype=jnp.complex128)

# gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3
gamma5 = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0]], dtype=jnp.complex128)

def _to_sparse(mat):
    """Return row indices, column indices, and values of nonzero entries."""
    r, c = jnp.nonzero(mat)
    return r, c, mat[r, c]


def _to_perm_phase(mat):
    """Decompose a 4×4 monomial matrix into (perm, phase) so that
    ``mat[α, β] = phase[α] · δ_{β, perm[α]}``.

    Every γ̃^μ ∈ {γ̃^0, γ̃^1, γ̃^2, γ̃^3} is monomial (one non-zero per
    row and per column, value ∈ {±1, ±i}), so ``Σ_α γ̃_{αα'} · X[α]``
    reduces to ``X[perm⁻¹[α']] · phase[perm⁻¹[α']]`` — a permuted
    gather + element-wise phase multiply.  Eliminates 4×4 matmul at
    every spin contraction site.
    """
    import numpy as _np
    m = _np.asarray(mat)
    n = m.shape[0]
    perm = _np.zeros(n, dtype=_np.int32)
    phase = _np.zeros(n, dtype=_np.complex128)
    for a in range(n):
        cols = _np.flatnonzero(_np.abs(m[a]) > 0)
        assert cols.size == 1, (
            f"_to_perm_phase requires a monomial matrix; row {a} has "
            f"{cols.size} non-zeros."
        )
        c = int(cols[0])
        perm[a] = c
        phase[a] = m[a, c]
    return jnp.asarray(perm), jnp.asarray(phase)


gammas = [gamma0, gamma1, gamma2, gamma3]
gammas_sparse = [_to_sparse(g) for g in gammas]

# Per-row permutation + phase decomposition of each γ̃^μ.  Used by the
# bispinor ζ-fit and Σ^B kernels to replace 4×4 matmuls on the spinor
# axis with a gather + per-element phase mul.  ``gammas_perm[μ][α] =
# perm[α]`` and ``gammas_phase[μ][α] = phase[α]`` such that
# ``γ̃^μ_{αβ} = phase[α] · δ_{β, perm[α]}``.
_perm_phase = [_to_perm_phase(g) for g in gammas]
gammas_perm = jnp.stack([pp[0] for pp in _perm_phase])     # (4, 4) int32
gammas_phase = jnp.stack([pp[1] for pp in _perm_phase])    # (4, 4) c128


def gamma_perm_phase(mu_lorentz: int) -> tuple[jax.Array, jax.Array]:
    """Return ``(perm, phase)`` for γ̃^{μ_L}.

    γ̃^{μ_L}_{αβ} = phase[α] · δ_{β, perm[α]} (every γ̃ ∈ {γ̃^0..3} is
    monomial, value ∈ {±1, ±i}).  Use with :func:`gamma_apply` to fold
    a γ̃-on-spin-axis contraction into a gather + element-wise phase
    multiply.
    """
    mu = int(mu_lorentz)
    return gammas_perm[mu], gammas_phase[mu]


def gamma_apply(X: jax.Array, perm: jax.Array, phase: jax.Array,
                axis: int, is_identity: bool = False) -> jax.Array:
    """Apply a monomial γ̃ matrix on ``axis`` of X via gather + phase mul.

    Computes ``Y[..., β, ...] = Σ_α γ̃_{βα} · X[..., α, ...]`` where
    ``γ̃_{βα} = phase[β] · δ_{α, perm[β]}``, i.e.

        Y[..., β, ...] = phase[β] · X[..., perm[β], ...]

    Replaces a 4×4 matmul on the spin axis with a single
    ``jnp.take`` + broadcasted element-wise multiply.  Generic over
    rank: the spin axis can sit anywhere, all other axes are
    untouched.

    For γ̃^0 = I_4 (perm = identity, phase = 1), pass ``is_identity=True``
    (Python bool — branch resolves at JIT trace time) to skip the
    gather + multiply entirely and return ``X`` byte-identical.
    """
    if is_identity:
        return X
    X_p = jnp.take(X, perm, axis=axis)
    bcast = [1] * X_p.ndim
    bcast[axis] = phase.shape[0]
    return X_p * phase.reshape(bcast)


def gamma_double_contract(
    P_l_conj: jax.Array, P_r: jax.Array,
    perm_L: jax.Array | None = None,
    phase_L: jax.Array | None = None,
    perm_R: jax.Array | None = None,
    phase_R: jax.Array | None = None,
    spin_axes: tuple[int, int] = (1, 2),
) -> jax.Array:
    """γ̃·γ̃ double contraction over two spin axes → drop spin rank by 2.

        Σ_{αβα'β'} γ̃_L_{αα'} γ̃_R_{ββ'} P_l_conj[..., α, β, ...] P_r[..., α', β', ...]
          = Σ_{αβ} phase_L[α]·phase_R[β]
                · P_l_conj[..., α, β, ...]
                · P_r[..., perm_L[α], perm_R[β], ...]

    γ̃ identity shortcut: pass ``perm_L=None`` (and/or ``perm_R=None``)
    to mean γ̃_L = I_4 (and/or γ̃_R = I_4); the corresponding
    ``gamma_apply`` is skipped at JIT trace time.  Both None → pure
    Σ_{αβ} P_l_conj · P_r Frobenius reduction (charge channel).

    Inputs:
        P_l_conj, P_r : same shape; the two specified ``spin_axes``
                        are size ns=4 each.
        spin_axes     : tuple ``(left_axis, right_axis)`` indexing the
                        two spin axes to contract.  Default ``(1, 2)``
                        matches the (k, a, b, μ, ν/r) layout used by
                        ``pair_density``.

    Output: same shape as P_l_conj minus the two spin axes.
    """
    a_axis, b_axis = spin_axes
    P_r_p = P_r
    if perm_L is not None:
        P_r_p = gamma_apply(P_r_p, perm_L, phase_L, axis=a_axis)
    if perm_R is not None:
        P_r_p = gamma_apply(P_r_p, perm_R, phase_R, axis=b_axis)
    return jnp.sum(P_l_conj * P_r_p, axis=spin_axes)


__all__ = [
    "sigma_x", "sigma_y", "sigma_z",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma5",
    "gammas_perm", "gammas_phase",
    "gamma_perm_phase", "gamma_apply", "gamma_double_contract",
]
