"""Bispinor initialization: compute small-component spinors from large components.

The 4-component (Dirac) spinor has large (upper) and small (lower) components.
For a non-relativistic wavefunction ψ_nk(G), the small component is:
    ψ_small = (α/2) (σ · (k+G)) ψ_large
where α is the fine-structure constant and σ are the Pauli matrices.
"""

import jax.numpy as jnp


# Half the fine-structure constant α (Hartree atomic units).  Single home
# for the bispinor small-component scale — both the non-batched reference
# ``get_small_psi_component`` and the k-batched ``lift_to_4spinor`` (the
# WfnLoader production path) use it.
HALFALPHA = 0.00364867628215


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
    halfalpha = jnp.complex128(HALFALPHA)  # 1/2 * fine-structure constant
    gvecsk_cart = jnp.matmul(gvecs + kvec, bvec)

    # Pauli matrix contraction: (σ·p)_{ab} where a,b are spinor indices
    sigmadotp = jnp.array([
        [gvecsk_cart[:, 2], gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]],
        [gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1], -gvecsk_cart[:, 2]],
    ], dtype=jnp.complex128)

    return jnp.multiply(
        halfalpha, jnp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
    )


def lift_to_4spinor(psi_2, gvecs, kvecs, bvec):
    """k-batched 2-spinor ψ → 4-spinor ψ via the small-component lift.

    Appends ``ψ_S = (α/2)(σ·(k+G)) ψ_L`` to the large components: the same
    physics as :func:`get_small_psi_component`, vectorised across a leading
    k-axis and concatenated into a 4-spinor.  Pure ``jnp`` (no jit / sharding
    here — the caller wraps it; see ``WfnLoader._get_bispinor_lift_jit``).

    This is the single home of the k-batched lift, imported by
    ``file_io.wfn_loader``; do not re-copy the constant + σ·p block there.

    Parameters
    ----------
    psi_2 : (n_k, nb, 2, ngkmax) complex
        Large-component ψ.
    gvecs : (n_k, ngkmax, 3) float64
        G-vectors (crystal), already cast to float.
    kvecs : (n_k, 3) float64
        k-vectors (crystal).
    bvec : (3, 3) float64
        Reciprocal lattice vectors (rows = b1, b2, b3).

    Returns
    -------
    (n_k, nb, 4, ngkmax) complex
        4-spinor ψ: ``[ψ_L ; ψ_S]`` along the spinor axis.
    """
    halfalpha = jnp.complex128(HALFALPHA)
    # (k + G) in cartesian, per (k, g).
    pkG = gvecs + kvecs[:, None, :]                          # (n_k, ngkmax, 3)
    p_cart = pkG @ bvec                                       # (n_k, ngkmax, 3)
    px = p_cart[..., 0].astype(jnp.complex128)
    py = p_cart[..., 1].astype(jnp.complex128)
    pz = p_cart[..., 2].astype(jnp.complex128)
    # σ·p as a (2, 2) Pauli-contraction stacked per (k, g):
    #   [[ pz, px - i*py],
    #    [px + i*py, -pz]]
    sdp = jnp.stack(
        [jnp.stack([pz,  px - 1j * py], axis=-1),
         jnp.stack([px + 1j * py, -pz], axis=-1)],
        axis=-2,
    )                                                         # (n_k, ngkmax, 2, 2)
    # ψ_S[k, b, i, g] = halfalpha · Σ_j σ·p[k, g, i, j] · ψ_L[k, b, j, g]
    psi_S = halfalpha * jnp.einsum(
        "kgij,kbjg->kbig", sdp, psi_2[:, :, 0:2, :])
    return jnp.concatenate([psi_2, psi_S], axis=2)           # (n_k, nb, 4, ngkmax)
