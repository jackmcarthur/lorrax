"""Bispinor initialization: compute small-component spinors from large components.

The 4-component (Dirac) spinor has large (upper) and small (lower) components.
For a non-relativistic wavefunction ψ_nk(G), the small component is:
    ψ_small = (α/2) (σ · (k+G)) ψ_large
where α is the fine-structure constant, σ are the Pauli matrices, and the
momentum is Cartesian in bohr⁻¹.  The caller that owns the WFN-file convention
must therefore fold ``blat`` into the stored dimensionless ``bvec`` exactly
once before entering this module.
"""

import jax.numpy as jnp

from common.gamma_matrices import paulis


# Half the fine-structure constant α (Hartree atomic units).  Single home
# for the bispinor small-component scale — both the non-batched reference
# ``get_small_psi_component`` and the k-batched ``lift_to_4spinor`` (the
# WfnLoader production path) use it.
HALFALPHA = 0.00364867628215
# The full fine-structure constant is derived here, beside the lift scale,
# rather than re-spelled by current-vertex consumers.
ALPHA_FS = 2.0 * HALFALPHA

# Public provenance for the exact model component this module defines.  The
# lift plus the dimensionless alpha_i vertex determines a positive-energy
# no-pair PARAMAGNETIC current correlator.  It deliberately says nothing about
# the same-Hamiltonian contact, gauged nonlocal pseudopotential, or eliminated
# negative-energy sector required for an electromagnetic response.
NO_PAIR_DIRAC_CURRENT_MODEL = (
    "positive_energy_kinetic_balance_dirac_current_v1"
)
KINETIC_BALANCE_LIFT_PROVENANCE = (
    "psi_S=(alpha_fs/2)*sigma.p*psi_L"
)
DIRAC_ALPHA_VERTEX_PROVENANCE = (
    "j=c*psi^dagger*alpha*psi; raw_paramagnetic_vertex_no_contact"
)


def sigma_dot_cartesian(psi_L, vector_cart):
    r"""Apply ``sigma . vector_cart`` to a large-component spinor block.

    ``psi_L`` has shape ``(..., band, 2, G)`` and ``vector_cart`` has the
    paired shape ``(..., G, 3)``.  The Cartesian components use the canonical
    Pauli matrices from :mod:`common.gamma_matrices`; no charge, velocity, or
    unit-conversion prefactor is included here.

    The explicit two-component boundary is intentional.  A four-component
    input is already lifted and must not be sliced or lifted a second time.
    """
    psi = jnp.asarray(psi_L)
    vector = jnp.asarray(vector_cart)
    if psi.ndim < 3 or int(psi.shape[-2]) != 2:
        raise ValueError(
            "sigma_dot_cartesian requires explicit Psi_L with shape "
            f"(..., band, 2, G), got {tuple(psi.shape)}")
    expected_vector_shape = psi.shape[:-3] + (psi.shape[-1], 3)
    if vector.shape != expected_vector_shape:
        raise ValueError(
            "sigma_dot_cartesian requires vector_cart paired with Psi_L: "
            f"expected {expected_vector_shape}, got {tuple(vector.shape)}")
    return jnp.einsum(
        "aij,...ga,...bjg->...big", paulis, vector, psi,
        optimize=True)


def kinetic_balance_lift_jet(psi_L, K_cart_bohr_inv):
    r"""Return the kinetic-balance lift and its three endpoint derivatives.

    Holding the explicit two-component ``psi_L`` fixed, this is the exact
    Cartesian jet of

    ``Psi(K) = [psi_L; (ALPHA_FS/2) (sigma.K) psi_L]``.

    ``K_cart_bohr_inv`` has shape ``(..., G, 3)`` in bohr^-1, paired with
    ``psi_L`` of shape ``(..., band, 2, G)``.  The returned value has shape
    ``(..., band, 4, G)`` and ``dPsi_dK`` has shape
    ``(3, ..., band, 4, G)`` with the Cartesian derivative axis first.

    The embedding is affine in each endpoint momentum, so
    ``d2Psi/dK_a dK_b = 0`` exactly.  No ninefold zero wavefunction carrier
    is allocated or returned.  Bra/ket endpoint routing, product rules, and
    charge/current/contact prefactors belong to the downstream operator
    owner; this neutral helper adds none of them.
    """
    psi = jnp.asarray(psi_L)
    sigma_K_psi = sigma_dot_cartesian(psi, K_cart_bohr_inv)
    halfalpha = jnp.complex128(HALFALPHA)
    lifted = jnp.concatenate((psi, halfalpha * sigma_K_psi), axis=-2)

    # First form (..., cart, band, spin, G), then keep the bounded Cartesian
    # jet axis first for direct endpoint selection by a response consumer.
    dpsi_small = halfalpha * jnp.einsum(
        "aij,...bjg->...abig", paulis, psi, optimize=True)
    cart_axis = psi.ndim - 3
    dpsi_small = jnp.moveaxis(dpsi_small, cart_axis, 0)
    dpsi_dK = jnp.concatenate(
        (jnp.zeros_like(dpsi_small), dpsi_small), axis=-2)
    return lifted, dpsi_dK


def get_small_psi_component(gvecs, kvec, bvec_cart_bohr, psi_G):
    """Compute the small (lower) spinor component from the large (upper).

    Computes (α/2)(σ·(k+G)) ψ_nk(G) for bispinor functionality.

    Args:
        gvecs: (ngk, 3) integer G-vectors in crystal coordinates
        kvec: (3,) k-vector in crystal coordinates
        bvec_cart_bohr: (3, 3) Cartesian reciprocal lattice vectors in
            bohr⁻¹ (rows = b1, b2, b3)
        psi_G: (nb, nspinor, ngk) wavefunction coefficients (large component)

    Returns:
        psi_small: (nb, 2, ngk) small-component spinor coefficients

    Note:
        Not @jax.jit because ngk varies per k-point → recompilation overhead.
        Possible improvements: σ·v with v = p + [r, V_NL + Σ], DKH4 contribution.
    """
    # Compatibility wrapper only.  The σ·p algebra has one implementation:
    # the k-batched production kernel below.
    return lift_to_4spinor(
        psi_G[None, ...],
        gvecs[None, ...],
        kvec[None, ...],
        bvec_cart_bohr,
    )[0, :, 2:4, :]


def lift_to_4spinor(psi_2, gvecs, kvecs, bvec_cart_bohr):
    """k-batched 2-spinor ψ → 4-spinor ψ via the small-component lift.

    Appends ``ψ_S = (α/2)(σ·(k+G)) ψ_L`` to the large components: the same
    physics as :func:`get_small_psi_component`, vectorised across a leading
    k-axis and concatenated into a 4-spinor.  Pure ``jnp`` (no jit / sharding
    here — the caller wraps it; see ``WfnLoader._get_bispinor_lift_jit``).

    This is the single home of the k-batched lift, imported by
    ``wfn_loader``; do not re-copy the constant + σ·p block there.

    Parameters
    ----------
    psi_2 : (n_k, nb, 2, ngkmax) complex
        Large-component ψ.
    gvecs : (n_k, ngkmax, 3) float64
        G-vectors (crystal), already cast to float.
    kvecs : (n_k, 3) float64
        k-vectors (crystal).
    bvec_cart_bohr : (3, 3) float64
        Cartesian reciprocal lattice vectors in bohr⁻¹
        (rows = b1, b2, b3).  This API does not accept the WFN file's raw
        dimensionless ``bvec`` because silently omitting ``blat`` rescales
        every small component.

    Returns
    -------
    (n_k, nb, 4, ngkmax) complex
        4-spinor ψ: ``[ψ_L ; ψ_S]`` along the spinor axis.
    """
    # (k + G) in cartesian, per (k, g).
    pkG = gvecs + kvecs[:, None, :]                          # (n_k, ngkmax, 3)
    p_cart = pkG @ bvec_cart_bohr                             # (n_k, ngkmax, 3)
    psi_S = jnp.complex128(HALFALPHA) * sigma_dot_cartesian(
        psi_2, p_cart)
    return jnp.concatenate([psi_2, psi_S], axis=2)           # (n_k, nb, 4, ngkmax)
