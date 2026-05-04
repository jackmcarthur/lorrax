"""Bispinor initialization: compute small-component spinors from large components.

The 4-component (Dirac) spinor has large (upper) and small (lower) components.
For a non-relativistic wavefunction ψ_nk(G), the small component is:
    ψ_small = (α_FS / 2) (σ · (k+G)) ψ_large
where α_FS is the fine-structure constant and σ are the Pauli matrices.
The (k+G) here must be in **atomic units (Bohr⁻¹)** for the formula to give
the correct physical magnitude.
"""

import math

import jax.numpy as jnp


def get_small_psi_component(gvecs, kvec, bvec, psi_G, alat):
    """Compute the small (lower) spinor component from the large (upper).

    Computes (α_FS/2)(σ·(k+G)) ψ_nk(G) for bispinor functionality.

    Args:
        gvecs: (ngk, 3) integer G-vectors in crystal coordinates
        kvec: (3,) k-vector in crystal coordinates
        bvec: (3, 3) reciprocal lattice vectors (rows = b_i) in BGW WFN.h5
            convention — i.e. **2π/alat units, not Bohr⁻¹**.  Inside this
            function we multiply by 2π/alat to convert to Bohr⁻¹ before
            applying the Dirac kinetic-balance formula.
        psi_G: (nb, nspinor, ngk) wavefunction coefficients (large component)
        alat: lattice constant ``a_lat`` in Bohr (= ``wfn.alat``).

    Returns:
        psi_small: (nb, 2, ngk) small-component spinor coefficients

    Note:
        Not @jax.jit because ngk varies per k-point → recompilation overhead.
        Possible improvements: σ·v with v = p + [r, V_NL + Σ], DKH4 contribution.
    """
    halfalpha = jnp.complex128(0.00364867628215)  # 1/2 * fine-structure constant
    # Convert (G + k) from BGW's 2π/alat-unit Cartesian to physical Bohr⁻¹.
    # Without this factor (k+G) is too small (when alat > 2π Bohr) or too
    # large (alat < 2π) by alat/(2π); for CrI3 (alat ~ 13 Bohr) the lift was
    # off by ~2× before this fix.  See `runs/MoS2/B_bispinor_pd_smoke_*` and
    # `docs/BISPINOR_DHFB_DESIGN.md` §2.2 for the diagnostic.
    tpi_over_alat = jnp.complex128(2.0 * math.pi / float(alat))
    gvecsk_cart = tpi_over_alat * jnp.matmul(gvecs + kvec, bvec)

    # Pauli matrix contraction: (σ·p)_{ab} where a,b are spinor indices
    sigmadotp = jnp.array([
        [gvecsk_cart[:, 2], gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]],
        [gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1], -gvecsk_cart[:, 2]],
    ], dtype=jnp.complex128)

    return jnp.multiply(
        halfalpha, jnp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
    )
