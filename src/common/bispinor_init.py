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
ISOMETRIC_KINETIC_BALANCE_LIFT_PROVENANCE = (
    "Psi=[I;X](I+X^dagger*X)^(-1/2)*psi_L;X=(alpha_fs/2)*sigma.p"
)
# Velocity balance (owner request 2026-09-04): the SPATIAL-CURRENT carrier
# only, ONE CARRIER PER CARTESIAN CHANNEL.  For channel a,
#
#   psi_S^(a) = (alpha/4) [ sigma.(2(k+G) + dV_SR/dk) psi_L
#                           + sigma^a (dV_SO/dk_a) psi_L ]          (Ry velocities)
#
# so that psi_L^dagger sigma^a psi_S^(a) + h.c. = (alpha/2) <2(k+G)_a + dV_NL/dk_a>
# EXACTLY: the spin-scalar part rides the sigma sandwich (sigma^a sigma^b V_b
# + V_b sigma^b sigma^a = 2 V_a because [sigma, V_SR] = 0) and the spin-orbit
# part sits behind sigma^a alone (sigma^a sigma^a = 1), which is what keeps
# the i eps_abc [sigma^c, dV_SO/dk_b] artifact of a single sigma.v carrier
# out.  The alpha^a vertex of channel a is then the pseudo-Hamiltonian's
# velocity at first order in q, plus the Gordon spin current.  The charge
# carrier keeps ``sigma.p``: its small-component density is the O(alpha^2)
# Dirac density, and a V_NL-dressed |psi_S|^2 has no place in it.
RAW_KINETIC_BALANCE_LIFT = "raw"
ISOMETRIC_KINETIC_BALANCE_LIFT = "isometric"
#: The family selector (deck value ``bispinor_current_balance = velocity``;
#: the resolver's ``current_lift``).  It names the per-channel scheme, not
#: one carrier; the loader and the lift take the channel selectors below.
VELOCITY_KINETIC_BALANCE_LIFT = "velocity"
VELOCITY_KINETIC_BALANCE_LIFTS = ("velocity_1", "velocity_2", "velocity_3")
VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE = (
    "psi_S^(a)=(alpha_fs/4)*[sigma.(2(k+G)+dV_SR/dk)+sigma^a*dV_SO/dk_a]*psi_L"
    ";one_carrier_per_channel"
)


def velocity_lift_channel(representation: str) -> int | None:
    """``"velocity_a"`` -> ``a`` in {1,2,3}; anything else -> ``None``."""
    mode = str(representation).strip().lower()
    if mode in VELOCITY_KINETIC_BALANCE_LIFTS:
        return int(mode[-1])
    return None
DIRAC_ALPHA_VERTEX_PROVENANCE = (
    "j=c*psi^dagger*alpha*psi; raw_paramagnetic_vertex_no_contact"
)


def kinetic_balance_lift_provenance(representation: str) -> str:
    """Return the authenticated provenance for one lift selector."""
    mode = str(representation).strip().lower()
    if mode == RAW_KINETIC_BALANCE_LIFT:
        return KINETIC_BALANCE_LIFT_PROVENANCE
    if mode == ISOMETRIC_KINETIC_BALANCE_LIFT:
        return ISOMETRIC_KINETIC_BALANCE_LIFT_PROVENANCE
    if mode == VELOCITY_KINETIC_BALANCE_LIFT or velocity_lift_channel(mode):
        # One provenance for the family: the three channel carriers are
        # the same scheme, and every artifact that names a lift names the
        # scheme (its channel is fixed by the Lorentz label it serves).
        return VELOCITY_KINETIC_BALANCE_LIFT_PROVENANCE
    raise ValueError(
        f"unknown kinetic-balance representation {representation!r}; "
        f"expected {RAW_KINETIC_BALANCE_LIFT!r}, "
        f"{ISOMETRIC_KINETIC_BALANCE_LIFT!r}, "
        f"{VELOCITY_KINETIC_BALANCE_LIFT!r} or one of "
        f"{VELOCITY_KINETIC_BALANCE_LIFTS}")


def _isometric_kinetic_balance_factor(K_cart_bohr_inv):
    """The sole pointwise ``(I + X^dagger X)^(-1/2)`` spelling."""
    K = jnp.asarray(K_cart_bohr_inv)
    h2 = jnp.float64(HALFALPHA) ** 2
    return jnp.reciprocal(jnp.sqrt(
        jnp.float64(1.0) + h2 * jnp.sum(K * K, axis=-1)))


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


def sigma_dot_cartesian_kets(kets):
    r"""Contract ``sigma`` with a Cartesian stack of two-component kets.

    ``kets`` has shape ``(..., 3, band, 2, G)`` — one ket per Cartesian
    component, e.g. ``(dV_NL/dk_a) psi_L`` from
    ``psp.vnl_ops.apply_vnl_velocity_to_ket``.  Returns
    ``sum_a sigma^a ket_a`` with shape ``(..., band, 2, G)``.  This is
    :func:`sigma_dot_cartesian` for an operator-valued vector: there the
    vector multiplies ``psi_L`` pointwise in G, here each component has
    already acted on ``psi_L``.  No prefactor is included.
    """
    kets = jnp.asarray(kets)
    if kets.ndim < 4 or int(kets.shape[-4]) != 3 or int(kets.shape[-2]) != 2:
        raise ValueError(
            "sigma_dot_cartesian_kets requires kets with shape "
            f"(..., 3, band, 2, G), got {tuple(kets.shape)}")
    return jnp.einsum("aij,...abjg->...big", paulis, kets, optimize=True)


def kinetic_balance_lift_jet(
    psi_L,
    K_cart_bohr_inv,
    *,
    representation: str = RAW_KINETIC_BALANCE_LIFT,
    cartesian_K_derivative_axes: tuple[int, ...] | None = None,
):
    r"""Return one representation-aware kinetic-balance endpoint jet.

    ``psi_L`` is the large block *in the selected representation* at the
    expansion point.  For ``raw`` it is the source Pauli spinor.  For
    ``isometric`` it is already ``r(K) psi_source``; the function must not
    normalize it a second time.  Holding ``psi_source`` fixed, the lifted
    endpoint is

    ``Psi(K) = [I; h sigma.K] psi_L(K)``, ``h=ALPHA_FS/2``.

    ``K_cart_bohr_inv`` has shape ``(..., G, 3)`` in bohr^-1, paired with
    ``psi_L`` of shape ``(..., band, 2, G)``.  The returned value has shape
    ``(..., band, 4, G)``.

    With ``cartesian_K_derivative_axes=None`` the historical return contract
    is retained: ``(Psi, dPsi_dK)`` with the Cartesian derivative axis first.
    The raw branch is the historical operation sequence unchanged.  Supplying
    one or two axes returns only that ``K`` first- or second-derivative family.
    This helper never inserts the signs from ``K_bra=k-q``: the downstream
    operator supplies one minus at q-first order and none at q-second order.
    The bounded selector lets that operator consume a derivative immediately
    instead of materializing a three- or nine-WFN jet.

    For the isometric representation,

    ``r=(1+h^2 K^2)^(-1/2)``,
    ``r_a/r=-h^2 K_a r^2``, and
    ``r_ab/r=-h^2 delta_ab r^2+3 h^4 K_a K_b r^4``.

    These analytic factors include both product-rule terms in the endpoint
    derivative.  For the raw affine embedding the selected second derivative
    is exact zero.  Bra/ket orientation and vertex/contact prefactors remain
    with the downstream operator owner.
    """
    psi = jnp.asarray(psi_L)
    K = jnp.asarray(K_cart_bohr_inv)
    sigma_K_psi = sigma_dot_cartesian(psi, K)
    halfalpha = jnp.complex128(HALFALPHA)
    lifted = jnp.concatenate((psi, halfalpha * sigma_K_psi), axis=-2)

    mode = str(representation).strip().lower()
    # Validate through the one representation/provenance owner.  Keeping the
    # returned string unused is deliberate: this numerical helper does not
    # manufacture an artifact identity.
    kinetic_balance_lift_provenance(mode)
    axes = (None if cartesian_K_derivative_axes is None else tuple(
        int(axis) for axis in cartesian_K_derivative_axes))
    if axes is not None and len(axes) not in (1, 2):
        raise ValueError(
            "kinetic_balance_lift_jet cartesian_K_derivative_axes must "
            f"contain one or two axes; got {axes}")
    if axes is not None and any(axis < 0 or axis >= 3 for axis in axes):
        raise ValueError(
            "kinetic_balance_lift_jet Cartesian axes must lie in [0,3); "
            f"got {axes}")

    if axes is None and mode == RAW_KINETIC_BALANCE_LIFT:
        # Historical raw path: preserve its operation sequence and returned
        # bytes.  New selected-family code stays entirely off this branch.
        dpsi_small = halfalpha * jnp.einsum(
            "aij,...bjg->...abig", paulis, psi, optimize=True)
        cart_axis = psi.ndim - 3
        dpsi_small = jnp.moveaxis(dpsi_small, cart_axis, 0)
        dpsi_dK = jnp.concatenate(
            (jnp.zeros_like(dpsi_small), dpsi_small), axis=-2)
        return lifted, dpsi_dK

    def raw_first(axis: int):
        small = halfalpha * jnp.einsum(
            "ij,...bjg->...big", paulis[axis], psi, optimize=True)
        return jnp.concatenate((jnp.zeros_like(small), small), axis=-2)

    if mode == RAW_KINETIC_BALANCE_LIFT:
        if len(axes) == 1:
            return raw_first(axes[0])
        return jnp.zeros_like(lifted)

    h2 = jnp.float64(HALFALPHA) ** 2
    r = _isometric_kinetic_balance_factor(K)
    r2 = r * r

    def first(axis: int):
        r_a_over_r = -h2 * K[..., axis] * r2
        return (r_a_over_r[..., None, None, :] * lifted
                + raw_first(axis))

    if axes is None:
        return lifted, jnp.stack(tuple(first(axis) for axis in range(3)))
    if len(axes) == 1:
        return first(axes[0])

    a, b = axes
    r_a_over_r = -h2 * K[..., a] * r2
    r_b_over_r = -h2 * K[..., b] * r2
    r_ab_over_r = (
        -h2 * jnp.asarray(a == b, dtype=jnp.float64) * r2
        + 3.0 * h2 * h2 * K[..., a] * K[..., b] * r2 * r2)
    return (
        r_ab_over_r[..., None, None, :] * lifted
        + r_a_over_r[..., None, None, :] * raw_first(b)
        + r_b_over_r[..., None, None, :] * raw_first(a))


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


def lift_to_4spinor(
    psi_2, gvecs, kvecs, bvec_cart_bohr, *,
    representation: str = RAW_KINETIC_BALANCE_LIFT,
    sigma_vnl_velocity_ry=None,
):
    """k-batched 2-spinor ψ → 4-spinor ψ via the small-component lift.

    Appends ``ψ_S = (α/2)(σ·(k+G)) ψ_L`` to the large components.
    The default ``representation='raw'`` is the historical map, unchanged.
    ``representation='isometric'`` applies the pointwise scalar

    ``r(G) = 1/sqrt(1 + [(α/2)|k+G|]^2)``

    to both upper and lower blocks.  Since ``(σ·K)†(σ·K)=|K|² I``, this is
    exactly ``[I;X](I+X†X)^(-1/2)`` without a redundant 2×2 matrix inverse.

    ``representation='velocity_a'`` (a = 1, 2, 3) builds the channel-``a``
    carrier of the per-channel velocity balance (module header):

    ``ψ_S^(a) = (α/2) σ·p ψ_L + (α/4) · sigma_vnl_velocity_ry``

    where ``sigma_vnl_velocity_ry`` is ``Σ_b σ^b (dV_SR/dk_b ψ_L) + σ^a
    (dV_SO/dk_a ψ_L)`` in Rydberg velocity units (``(n_k, nb, 2, ngkmax)``,
    from ``psp.vnl_ops.nonlocal_velocity_lift(setup)(..., channel=a)``) and
    the ¼ is (α/2) times the Ry→Hartree velocity conversion ½.  It is
    REQUIRED for these representations and REFUSED for the others: a
    supplied ket that was silently ignored is the failure class this tree
    pays for.  The family name ``"velocity"`` is not a carrier and is
    refused here; callers resolve it to a channel first.
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
    mode = str(representation).strip().lower()
    if mode == VELOCITY_KINETIC_BALANCE_LIFT:
        raise ValueError(
            "lift_to_4spinor: 'velocity' names the per-channel scheme, not a "
            f"carrier; pass one of {VELOCITY_KINETIC_BALANCE_LIFTS} (the "
            "Lorentz label the carrier serves).")
    if velocity_lift_channel(mode):
        if sigma_vnl_velocity_ry is None:
            raise ValueError(
                f"lift_to_4spinor(representation={mode!r}) requires "
                "sigma_vnl_velocity_ry (the channel's projector velocity "
                "ket); attach psp.vnl_ops.nonlocal_velocity_lift(setup) to "
                "the loader (WfnLoader.nonlocal_velocity_lift).")
        sigma_v = jnp.asarray(sigma_vnl_velocity_ry)
        if tuple(sigma_v.shape) != tuple(psi_S.shape):
            raise ValueError(
                "lift_to_4spinor: sigma_vnl_velocity_ry shape "
                f"{tuple(sigma_v.shape)} != psi_L shape {tuple(psi_S.shape)}")
        # (alpha/2) * (1/2): dV_NL/dk arrives in Rydberg velocity units
        # (dH_Ry/dk), and the Hartree velocity is half of that.
        psi_S = psi_S + jnp.complex128(0.5 * HALFALPHA) * sigma_v
    elif sigma_vnl_velocity_ry is not None:
        raise ValueError(
            "lift_to_4spinor: sigma_vnl_velocity_ry is only meaningful for "
            f"the velocity_a representations, got {representation!r}")
    lifted = jnp.concatenate([psi_2, psi_S], axis=2)
    if mode == RAW_KINETIC_BALANCE_LIFT or velocity_lift_channel(mode):
        return lifted
    if mode != ISOMETRIC_KINETIC_BALANCE_LIFT:
        raise ValueError(
            f"unknown kinetic-balance representation {representation!r}; "
            f"expected {RAW_KINETIC_BALANCE_LIFT!r}, "
            f"{ISOMETRIC_KINETIC_BALANCE_LIFT!r} or one of "
            f"{VELOCITY_KINETIC_BALANCE_LIFTS}")
    r = _isometric_kinetic_balance_factor(p_cart)
    return lifted * r[:, None, None, :]
