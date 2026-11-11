"""Bispinor initialization: compute small-component spinors from large components.

The 4-component (Dirac) spinor has large (upper) and small (lower) components.
For a non-relativistic wavefunction ψ_nk(G), the small component is:
    ψ_small = (α/2) (σ · (k+G)) ψ_large
where α is the fine-structure constant, σ are the Pauli matrices, and the
momentum is Cartesian in bohr⁻¹.  The caller that owns the WFN-file convention
must therefore fold ``blat`` into the stored dimensionless ``bvec`` exactly
once before entering this module.

A bispinor run uses two four-component carriers of the same two-component
ψ_L (docs/theory/bispinor-gw.md#carrier); ``common.four_current_model`` says
which vertex takes which:

* ``RAW_KINETIC_BALANCE_LIFT``, ``[ψ_L; (α/2) σ·p ψ_L]``: every current
  vertex.  ``c Ψ† α Ψ`` on it is exactly the Pauli current (convective plus
  spin magnetization), conserved together with the charge ``ψ_L† ψ_L``.
* ``LARGE_BLOCK_LIFT``, ``[ψ_L; 0]``: every charge vertex, so the charge
  density and the scalar (CC) channel see ``ψ_L† ψ_L``.
"""

import jax.numpy as jnp

from common.gamma_matrices import paulis


# Half the fine-structure constant α (Hartree atomic units).  Single home
# for the bispinor small-component scale, used by the k-batched
# ``lift_to_4spinor`` (the WfnLoader production path).
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
LARGE_BLOCK_LIFT_PROVENANCE = "Psi=[psi_L;0]"
# The carrier of every current vertex (docs/theory/bispinor-gw.md#carrier).
RAW_KINETIC_BALANCE_LIFT = "raw"
# The carrier of every charge vertex: psi_L in the large block and a zero
# small block (docs/theory/bispinor-gw.md#carrier).
LARGE_BLOCK_LIFT = "large_block"
LIFT_PROVENANCE = {
    RAW_KINETIC_BALANCE_LIFT: KINETIC_BALANCE_LIFT_PROVENANCE,
    LARGE_BLOCK_LIFT: LARGE_BLOCK_LIFT_PROVENANCE,
}
DIRAC_ALPHA_VERTEX_PROVENANCE = (
    "j=c*psi^dagger*alpha*psi; raw_paramagnetic_vertex_no_contact"
)


def kinetic_balance_lift_provenance(representation: str) -> str:
    """Return the authenticated provenance for one lift selector."""
    mode = str(representation).strip().lower()
    if mode not in LIFT_PROVENANCE:
        raise ValueError(
            f"unknown kinetic-balance representation {representation!r}; "
            f"expected one of {sorted(LIFT_PROVENANCE)}")
    return LIFT_PROVENANCE[mode]


def apply_dirac_velocity_to_ket(psi):
    """Apply c alpha_i to (...,band,4,G) kets, in Ry Bohr.

    Returns (3,...,band,4,G). This is the paramagnetic Dirac velocity;
    nonlocal-potential and QP corrections belong to their existing owners.
    c=2/alpha_fs in Rydberg units. The input must already carry the selected
    current carrier (the raw kinetic-balance lift); no second lift occurs.
    """
    from common.gamma_matrices import gamma1, gamma2, gamma3
    if psi.ndim < 3 or psi.shape[-2] != 4:
        raise ValueError('Dirac velocity requires (...,band,4,G) wavefunctions')
    return (2.0 / ALPHA_FS) * jnp.stack(tuple(
        jnp.einsum('st,...ntg->...nsg', alpha, psi)
        for alpha in (gamma1, gamma2, gamma3)))


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


def lift_to_4spinor(
    psi_2, gvecs, kvecs, bvec_cart_bohr, *,
    representation: str,
):
    """k-batched 2-spinor ψ → 4-spinor ψ for one bispinor carrier.

    ``representation`` has no default: a bispinor load names the vertex it
    serves (``common.four_current_model.resolve_four_current_representation``).

    * ``'raw'`` (``RAW_KINETIC_BALANCE_LIFT``, the current carrier) appends
      ``ψ_S = (α/2)(σ·(k+G)) ψ_L`` to the large components.
    * ``'large_block'`` (``LARGE_BLOCK_LIFT``, the charge carrier) appends
      zeros, so ``Ψ†Ψ = ψ_L†ψ_L`` exactly.

    Pure ``jnp`` (no jit / sharding here — the caller wraps it; see
    ``WfnLoader._get_bispinor_lift_jit``).

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
    representation : {'raw', 'large_block'}
        The carrier (keyword only, required).

    Returns
    -------
    (n_k, nb, 4, ngkmax) complex
        4-spinor ψ: ``[ψ_L ; ψ_S]`` along the spinor axis.
    """
    mode = str(representation).strip().lower()
    kinetic_balance_lift_provenance(mode)
    if mode == LARGE_BLOCK_LIFT:
        return jnp.concatenate([psi_2, jnp.zeros_like(psi_2)], axis=2)
    # (k + G) in cartesian, per (k, g).
    pkG = gvecs + kvecs[:, None, :]                          # (n_k, ngkmax, 3)
    p_cart = pkG @ bvec_cart_bohr                             # (n_k, ngkmax, 3)
    psi_S = jnp.complex128(HALFALPHA) * sigma_dot_cartesian(
        psi_2, p_cart)
    return jnp.concatenate([psi_2, psi_S], axis=2)
