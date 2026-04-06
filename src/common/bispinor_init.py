"""Bispinor initialization: compute small-component spinors from large components.

The 4-component (Dirac) spinor has large (upper) and small (lower) components.
For a non-relativistic wavefunction ψ_nk(G), the small component is:
    ψ_small = (α/2) (σ · (k+G)) ψ_large
where α is the fine-structure constant and σ are the Pauli matrices.
"""

import jax.numpy as jnp


def get_small_psi_component(gvecs, kvec, bvec, psi_G):
    """Compute the small (lower) spinor component from the large (upper).

    Computes (α/2)(σ·(k+G)) ψ_nk(G) for bispinor functionality.

    Args:
        gvecs: (ngk, 3) integer G-vectors in crystal coordinates
        kvec: (3,) k-vector in crystal coordinates
        bvec: (3, 3) reciprocal lattice vectors (rows = b1, b2, b3)
        psi_G: (nb, nspinor, ngk) wavefunction coefficients (large component)

    Returns:
        psi_small: (nb, 2, ngk) small-component spinor coefficients

    Note:
        Not @jax.jit because ngk varies per k-point → recompilation overhead.
        Possible improvements: σ·v with v = p + [r, V_NL + Σ], DKH4 contribution.
    """
    halfalpha = jnp.complex128(0.00364867628215)  # 1/2 * fine-structure constant
    gvecsk_cart = jnp.matmul(gvecs + kvec, bvec)

    # Pauli matrix contraction: (σ·p)_{ab} where a,b are spinor indices
    sigmadotp = jnp.array([
        [gvecsk_cart[:, 2], gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]],
        [gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1], -gvecsk_cart[:, 2]],
    ], dtype=jnp.complex128)

    return jnp.multiply(
        halfalpha, jnp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
    )
