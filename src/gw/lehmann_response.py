"""Ordered Lehmann weights shared by centroid and plane-wave response owners.

No normalization or density orientation enters here. The caller supplies
``de = E_a-E_b``, ``df = f_a-f_b`` and the physical pair mask, and owns the
outer k/spin/volume normalization. Static ``z=0`` has a separate response
owner; reference callers authenticate their nonzero complex sites first.
"""
import jax.numpy as jnp
import operator


def centroid_pair_density(psi_left, psi_right):
    """The direct scanner's local density ``ψ_a conj(ψ_b)``.

    Inputs are ordinary, typed-unfolded ``[k,spin,mu,band]`` tiles on one
    endpoint slab; the output is ``[k,mu,a,b]``. Physical ordered response
    uses its conjugate on the row endpoint. This function owns only the
    spin trace; transport, band selection and validity stay with callers.
    """
    return jnp.einsum("ksma,ksmb->kmab", psi_left, jnp.conj(psi_right), optimize=True)


def centroid_response_denominator(n_k):
    """The raw centroid pair sum's canonical ``sqrt(Nk)`` denominator.

    Spin and the second k normalization enter the existing Dyson owner,
    not this raw response. Kept as a JAX float64 square root to preserve
    the direct scanner's original arithmetic.
    """
    if isinstance(n_k, bool):
        raise ValueError("centroid response requires a positive integer full-k count")
    count = operator.index(n_k)
    if count < 1:
        raise ValueError("centroid response requires a positive integer full-k count")
    return jnp.sqrt(jnp.asarray(count, jnp.float64))


def lehmann_pair_weights(energy_difference, occupation_difference, z_values, *,
                         with_derivative=False, pair_mask=None):
    """Return ``df/(de+z)`` and optionally its exact ``d/ds``, ``s=z²``.

    ``de`` and ``df`` have the same arbitrary pair shape. A one-dimensional
    frequency axis is prepended; with ``with_derivative`` the slope rows
    follow the value rows, preserving the direct centroid scanner's layout.
    ``pair_mask`` is a boolean array broadcastable to the pair shape, not an occupation
    complement: both transition directions vanish on an invalid state.
    """
    de = jnp.asarray(energy_difference)
    df = jnp.asarray(occupation_difference)
    z = jnp.asarray(z_values, dtype=jnp.complex128)
    if de.shape != df.shape or z.ndim != 1 or z.size < 1:
        raise ValueError("Lehmann weights require equal pair shapes and nonempty 1-D sites")
    z_broadcast = z.reshape((z.size,) + (1,) * de.ndim)
    denominator = de[None] + z_broadcast
    weights = df[None] / denominator
    if with_derivative:
        slope = -weights / (denominator * (2 * z_broadcast))
        weights = jnp.concatenate((weights, slope), axis=0)
    if pair_mask is not None:
        mask = jnp.asarray(pair_mask)
        if mask.dtype != jnp.bool_ or jnp.broadcast_shapes(mask.shape, de.shape) != de.shape:
            raise ValueError("Lehmann pair mask must be boolean and broadcast to the pair shape")
        weights = jnp.where(mask[None], weights, 0.0)
    return weights
