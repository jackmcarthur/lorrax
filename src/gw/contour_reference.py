"""Ordered contour-deformation algebra for small scalar GW references.

This reference helper consumes contractions of *minus* the decaying
correlation interaction, ``-W_c``. It builds neither a response nor a Green
function. The caller retains their shared physics owners and all-P layouts,
and supplies bounded external-state blocks. Frequencies and spacings are in
Ry. Values/derivatives have shape ``(k, a, b, m)``, external-minus-internal
energies ``x`` and occupations have shape ``(k, a, E, m)``. Derivatives are
with respect to ``s=z**2``. The output is ``(k, a, b, E)``.

The ordered split is the authenticated historical TRREF algebra, with
``W(-iu)=W(iu)^dagger`` and occupied residues from ``W_-q(|x|+i eta)^T``.
A decaying two-pole anchor matches the exactly known value and slope at
``i eta``. Its integral is analytic; subtraction occurs before multiplying
the singular imaginary-axis kernel. Infinite-domain refinements are empirical
checks, not a certified bound on an unsampled physical response. An exactly
known instantaneous term must be removed by its owner before this helper and
its self-energy added separately; it is not a decaying ``W_c`` sample.

The default finite-eta convention is ``time_ordered_fractional``:
occupied pole denominators have ``x+Omega-i eta`` and empty denominators
``x-Omega+i eta``. A retarded self-energy uses ``+i eta`` on both poles;
their imaginary parts must not be compared as the same observable. Their
diagonal real parts coincide for real Hermitian pole residues.
``analytic_convention="retarded"`` selects the common upper-half-plane
external energy and conjugates the occupied residue's complete derivative.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from common.units import RYD_TO_EV


def _endpoint_face(value, mesh, label):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    actual = getattr(value, "sharding", None)
    if (not isinstance(actual, NamedSharding) or actual.mesh != mesh
            or actual.spec != P(None, "x", "y") or value.ndim != 3
            or value.dtype != jnp.complex128 or min(value.shape) < 1):
        raise ValueError(f"CD endpoint {label} requires a nonempty complex128 all-P face")


@lru_cache(maxsize=16)
def _density_endpoint_program(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la import matmul
    face = NamedSharding(mesh, P(None, "x", "y"))

    def project(pair, endpoint_map):
        endpoint_map = jax.lax.with_sharding_constraint(jnp.broadcast_to(
            endpoint_map, (pair.shape[0],) + endpoint_map.shape[-2:]), face)
        return matmul(pair, endpoint_map, mesh=mesh, backend="distributed",
                      batched_route="auto")

    return jax.jit(project, in_shardings=(face, face), out_shardings=face)


def project_density_endpoints(pair, endpoint_map, *, mesh):
    """Change density endpoints by the explicit linear map ``pair @ map``.

    Both operands and the result remain complex128 all-P faces. Pair rows
    have ``[batch,T,mu]`` and the declared map ``[batch or 1,mu,G]``. A fitted
    ISDF density uses the same saved ζ that builds its bare interaction,
    packed at the canonical I/O seam. No normalization, conjugation, basis
    completeness or relation between ordered partners is inferred here.
    An asymmetric complex map requires mapping both actual densities;
    mapping a conjugated density is not conjugating its mapped result.
    """
    _endpoint_face(pair, mesh, "pair")
    _endpoint_face(endpoint_map, mesh, "map")
    if (pair.shape[-1] != endpoint_map.shape[-2]
            or endpoint_map.shape[0] not in (1, pair.shape[0])):
        raise ValueError("CD density endpoint map has incompatible pair/map axes")
    return _density_endpoint_program(mesh)(pair, endpoint_map)


@lru_cache(maxsize=16)
def _interaction_endpoint_program(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    from distrib_la import matmul
    face = NamedSharding(mesh, P(None, "x", "y"))
    replicated = NamedSharding(mesh, P())

    def lift(interaction, endpoint_map, prefactor):
        batch = max(interaction.shape[0], endpoint_map.shape[0])
        interaction = jax.lax.with_sharding_constraint(jnp.broadcast_to(
            interaction, (batch,) + interaction.shape[-2:]), face)
        endpoint_map = jax.lax.with_sharding_constraint(jnp.broadcast_to(
            endpoint_map, (batch,) + endpoint_map.shape[-2:]), face)
        left = matmul(endpoint_map.conj(), interaction, mesh=mesh,
                      backend="distributed", batched_route="auto")
        right = transpose_xy(endpoint_map, mesh)
        return matmul(left, right, mesh=mesh, backend="distributed",
                      batched_route="auto") * prefactor

    return jax.jit(lift, in_shardings=(face, face, replicated), out_shardings=face)


def lift_interaction_endpoints(interaction, endpoint_map, *, mesh, prefactor):
    """Lift a reduced operator as ``conj(map) @ interaction @ map.T * pref``.

    ``interaction[batch,G,G]`` and ``map[batch or 1,mu,G]`` stay on all-P
    faces, including both GEMM temporaries and the lifted ``[batch,mu,mu]``.
    The right endpoint is a transpose, not an adjoint; the operator may be
    non-Hermitian at a complex frequency. Broadcast is allowed when either
    batch extent is one. The caller supplies the finite positive physical
    prefactor. Saved charge ζ with a conventional PW interaction requires
    ``1/Omega``; this function neither inserts a volume nor changes units.
    It builds no response, inverse Coulomb root or ordered partner.
    """
    import jax.numpy as jnp
    _endpoint_face(interaction, mesh, "interaction")
    _endpoint_face(endpoint_map, mesh, "map")
    if (interaction.shape[-1] != interaction.shape[-2]
            or interaction.shape[-1] != endpoint_map.shape[-1]
            or (interaction.shape[0] != endpoint_map.shape[0]
                and min(interaction.shape[0], endpoint_map.shape[0]) != 1)):
        raise ValueError("CD interaction endpoint lift has incompatible operator/map axes")
    prefactor = float(prefactor)
    if not np.isfinite(prefactor) or prefactor <= 0:
        raise ValueError("CD interaction lift needs an explicit positive finite physical prefactor")
    return _interaction_endpoint_program(mesh)(interaction, endpoint_map,
                                               jnp.asarray(prefactor, jnp.float64))


@lru_cache(maxsize=16)
def _diagonal_projection_program(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la import matmul

    face = NamedSharding(mesh, P(None, "x", "y"))
    replicated = NamedSharding(mesh, P())

    def project(interaction, pair, prefactor):
        operator = jax.lax.with_sharding_constraint(jnp.broadcast_to(
            interaction, (pair.shape[0],) + interaction.shape[-2:]), face)
        product = matmul(pair.conj(), operator, mesh=mesh,
                         backend="distributed", batched_route="auto")
        diagonal = jnp.sum(product * pair, axis=-1) * prefactor
        return product, diagonal

    return jax.jit(project, in_shardings=(face, face, replicated),
                   out_shardings=(face, replicated))


def project_interaction_diagonal(interaction, pair, *, mesh, prefactor,
                                 scalar_replication_bound_bytes):
    """Contract ordered pair rows against an all-P interaction.

    ``pair[batch,T,G]`` and ``interaction[batch or 1,G,G]`` are complex128
    faces at ``P(None,'x','y')``. The visible contraction is
    ``conj(pair) @ interaction``, followed by ``sum(product*pair,G)``.
    The product remains all-P. Only the caller-bounded scalar result
    ``[batch,T]`` is replicated, to feed the scalar contour integrator.

    The caller supplies the physical positive prefactor: conventional
    plane-wave vertices need ``1/(Nk*Omega)``; the ISDF contraction needs
    ``1/Nk``. Supply ``-W_c`` to the correlation integrator, removing its
    exactly known instantaneous term with the interaction's own owner.
    This routine neither builds a response nor infers normalization,
    occupations, the ordered partner or a head correction.
    """
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    expected = P(None, "x", "y")
    for name, value in (("interaction", interaction), ("pair", pair)):
        actual = getattr(value, "sharding", None)
        if (not isinstance(actual, NamedSharding) or actual.mesh != mesh
                or actual.spec != expected or value.ndim != 3
                or value.dtype != jnp.complex128):
            raise ValueError(f"CD projection {name} needs a complex128 all-P {expected} face")
    if (interaction.shape[-2] != interaction.shape[-1]
            or pair.shape[-1] != interaction.shape[-1]
            or interaction.shape[0] not in (1, pair.shape[0])
            or min(pair.shape) < 1):
        raise ValueError("CD projection requires compatible nonempty pair/operator faces")
    prefactor = float(prefactor)
    bound = int(scalar_replication_bound_bytes)
    scalar_bytes = pair.shape[0] * pair.shape[1] * pair.dtype.itemsize
    if not np.isfinite(prefactor) or prefactor <= 0:
        raise ValueError("CD projection needs an explicit positive finite physical prefactor")
    if bound < scalar_bytes:
        raise ValueError("CD projection scalar output exceeds its explicit replication bound")
    return _diagonal_projection_program(mesh)(
        interaction, pair, jnp.asarray(prefactor, dtype=jnp.float64))


@lru_cache(maxsize=16)
def _target_block_program(mesh, n_targets):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    face = NamedSharding(mesh, P(None, "x", "y"))
    grouped = NamedSharding(mesh, P(None, "x", None, "y"))
    replicated = NamedSharding(mesh, P())

    def contract(product, pair, prefactor):
        shape = (pair.shape[0], pair.shape[1] // n_targets,
                 n_targets, pair.shape[2])
        left = jax.lax.with_sharding_constraint(product.reshape(shape), grouped)
        right = jax.lax.with_sharding_constraint(pair.reshape(shape), grouped)
        return jnp.sum(left[:, :, :, None, :] * right[:, :, None, :, :],
                       axis=-1) * prefactor

    return jax.jit(contract, in_shardings=(face, face, replicated),
                   out_shardings=replicated)


def project_interaction_block(interaction, pair, *, mesh, n_targets,
                              prefactor, scalar_replication_bound_bytes):
    """Retain the complete target block for each internal state.

    Pair rows on the same all-P face as :func:`project_interaction_diagonal`
    are ordered ``[internal_m,target_a]``. The existing diagonal projector
    supplies the unchanged product ``conj(pair) @ interaction``. The result
    is the explicit contraction ``sum_G product[m,a,G]*pair[m,b,G]*pref``
    with shape ``[batch,m,a,b]``. Only this caller-bounded small block is
    replicated; product and both grouped contraction operands remain all-P.
    Internal groups must divide the X axis, so reshaping never gathers pair
    rows. Any padded target/internal entries must be zero at the caller.

    Supply the physical prefactor and ``-W_c`` exactly as for the diagonal
    projection. This function neither builds a response nor selects a
    contour sheet or partner. A minus-q interaction contracted against the
    reversed density roles needs its *external target block transpose*
    before it represents the contour helper's already-transposed partner.
    That transpose is distinct from an adjoint or a causal conjugation.
    """
    import jax.numpy as jnp
    from common.collectives import resolve_mesh

    # The common owner resolves the full square run mesh; caller-mesh
    # validation alone permits a proper subset and cannot certify all-P.
    canonical_mesh = resolve_mesh()
    if (mesh.devices.size != canonical_mesh.devices.size
            or set(mesh.devices.flat) != set(canonical_mesh.devices.flat)
            or tuple(mesh.axis_names) != ("x", "y")
            or int(mesh.shape["x"]) != int(mesh.shape["y"])):
        raise ValueError("CD block projection mesh must cover every global JAX device on the square X/Y grid")
    if (not isinstance(n_targets, (int, np.integer))
            or isinstance(n_targets, (bool, np.bool_)) or n_targets < 1):
        raise ValueError("CD block projection needs an exact positive target count")
    n_targets = int(n_targets)
    _endpoint_face(pair, mesh, "block pair")
    if (pair.shape[1] % n_targets
            or (pair.shape[1] // n_targets) % int(mesh.shape["x"])):
        raise ValueError("CD block projection requires complete internal groups on the X face")
    if (not isinstance(scalar_replication_bound_bytes, (int, np.integer))
            or isinstance(scalar_replication_bound_bytes, (bool, np.bool_))):
        raise ValueError("CD block projection needs an exact scalar replication bound")
    scalar_bytes = (pair.shape[0] * pair.shape[1] * n_targets
                    * pair.dtype.itemsize)
    if scalar_replication_bound_bytes < scalar_bytes:
        raise ValueError("CD block projection scalar output exceeds its explicit replication bound")
    product, _ = project_interaction_diagonal(
        interaction, pair, mesh=mesh, prefactor=prefactor,
        scalar_replication_bound_bytes=scalar_replication_bound_bytes)
    block = _target_block_program(mesh, n_targets)(
        product, pair, jnp.asarray(prefactor, dtype=jnp.float64))
    return product, block


def imaginary_rule(n, eta, *, scale=None):
    """Gauss rules on ``[0, eta]`` and ``[eta, infinity)``, both open.

    ``n`` nodes are divided between the finite segment and the rationally
    mapped tail. No finite cutoff or missing tail changes with ``n``. ``scale``
    sets the tail map's stretch in Ry and is held fixed across refinements.
    Returns positive nodes and weights in Ry; no node equals the Green pole.
    """
    eta = float(eta)
    scale = eta if scale is None else float(scale)
    if int(n) != n or int(n) < 8 or not np.isfinite([eta, scale]).all() or min(eta, scale) <= 0:
        raise ValueError("CD imaginary rule needs n >= 8 and positive finite eta/scale")
    low = max(2, int(n) // 5)
    out = []
    for count, tail in ((low, False), (int(n) - low, True)):
        t, w = np.polynomial.legendre.leggauss(count)
        t, w = (t + 1) / 2, w / 2
        if tail:
            out.append((eta + scale * t / (1 - t), w * scale / (1 - t)**2))
        else:
            out.append((eta * t, eta * w))
    return tuple(np.concatenate([part[j] for part in out]) for j in (0, 1))


def _inputs(x, occ, eta):
    x = np.asarray(x, np.float64)
    f = np.broadcast_to(np.asarray(occ, np.float64), x.shape)
    if x.ndim != 4 or not np.isfinite(x).all() or not np.isfinite(f).all():
        raise ValueError("CD x/occupations must be finite (k,a,E,m) arrays")
    if np.any((f < 0) | (f > 1)) or not np.isfinite(eta) or eta <= 0:
        raise ValueError("CD needs occupations in [0,1] and positive finite eta")
    return x, f


def _apply(a, weights, xp):
    return xp.einsum("kabl,kael->kabe", a, weights)


def _dagger(a, xp):
    return xp.conj(xp.swapaxes(a, -3, -2))


def _kernels(x, occ, eta, analytic_convention, xp):
    if analytic_convention not in ("time_ordered_fractional", "retarded"):
        raise ValueError("CD analytic_convention must be time_ordered_fractional or retarded")
    f = xp.broadcast_to(xp.asarray(occ), x.shape)
    dc = x + 1j * eta
    return f, (dc if analytic_convention == "retarded" else x - 1j * eta), dc


def _physical_band_mask(x, n_active, band_valid):
    count = x.shape[-1] if n_active is None else int(n_active)
    if not 0 < count <= x.shape[-1]:
        raise ValueError("CD n_active must include a positive physical band extent")
    active = np.arange(x.shape[-1])[None, None, None, :] < count
    if band_valid is not None:
        valid = np.asarray(band_valid)
        if valid.dtype != np.bool_:
            raise ValueError("CD band_valid must be an explicit boolean physical-state mask")
        active = active & np.broadcast_to(valid, x.shape)
    return active


def anchor_coefficients(value, derivative_s, eta, *, betas=None):
    """Two decaying poles matching value and exact ``d/ds`` at ``z=i eta``.

    Returns ``(c1,c2)`` for ``A(u)=sum_j c_j/(u**2+beta_j**2)``.
    The anchor poles are numerical auxiliaries, held fixed in every refinement.
    """
    beta = np.asarray([.5, 4.], np.float64) / RYD_TO_EV if betas is None else np.asarray(betas, np.float64)
    if beta.shape != (2,) or not np.isfinite(beta).all() or np.any(beta <= 0) or beta[0] == beta[1]:
        raise ValueError("CD anchor needs two distinct positive finite beta values")
    d1, d2 = eta**2 + beta**2
    y1 = (-derivative_s + value / d2) / (1 / d2 - 1 / d1)
    return (y1 * d1, (value - y1) * d2), beta


def _odd_anchor(d, x, beta, zero_angle, xp, *, external_derivative=False):
    d2 = d * d
    angle = xp.where(x == 0, zero_angle, xp.angle(d2))
    numerator = np.log(beta**2) - (xp.log(xp.abs(d2)) + 1j * angle)
    if external_derivative:
        return (d * numerator - (beta**2 - d2) / d) / (beta**2 - d2)**2
    return numerator / (2 * (beta**2 - d2))


def anchor_part(value0, deriv0, x, occ, *, eta, betas=None, n_active=None,
                band_valid=None,
                analytic_convention="time_ordered_fractional", xp=np,
                external_derivative=False):
    """Analytic anchor integral, or its external-energy derivative.

    Response inputs and auxiliary poles stay fixed when differentiating.
    At ``x=0`` use the same branch as the value; the exact residue's slope
    cancels the anchor's branch jump because both anchor conditions match.
    """
    if type(external_derivative) is not bool:
        raise ValueError("CD external_derivative must be a static boolean")
    xh, fh = _inputs(x, occ, eta)
    x, f = xp.asarray(xh), xp.asarray(fh)
    cp, beta = anchor_coefficients(value0, deriv0, eta, betas=betas)
    cm, _ = anchor_coefficients(_dagger(value0, xp), _dagger(deriv0, xp), eta, betas=beta)
    f, dv, dc = _kernels(x, f, eta, analytic_convention, xp)
    angle_v = np.pi if analytic_convention == "retarded" else -np.pi
    sign = xp.where(x >= 0, 1., -1.)
    total = 0
    for p, m, b in zip(cp, cm, beta):
        if external_derivative:
            even = -f / (2 * b * (b + sign * dv)**2) - (1 - f) / (2 * b * (b + sign * dc)**2)
        else:
            even = f * sign / (2 * b * (b + sign * dv)) + (1 - f) * sign / (2 * b * (b + sign * dc))
        odd = (f * _odd_anchor(dv, x, b, angle_v, xp, external_derivative=external_derivative)
               + (1 - f) * _odd_anchor(dc, x, b, np.pi, xp, external_derivative=external_derivative)) * (1j / (2 * np.pi))
        if n_active is not None or band_valid is not None:
            active = xp.asarray(_physical_band_mask(xh, n_active, band_valid))
            even, odd = xp.where(active, even, 0.), xp.where(active, odd, 0.)
        total = total + _apply(.5 * (p + m), even, xp) + _apply(p - m, odd, xp)
    return total, cp, cm, beta


def imag_remainder_node(value_i, u, weight, x, occ, cp, cm, betas, *, eta,
                        n_active=None, band_valid=None,
                        analytic_convention="time_ordered_fractional", xp=np,
                        external_derivative=False):
    """One imaginary node with its known Green-pole limit subtracted first.

    ``value_i`` is the contraction of ``-W_c(iu)``. The two independently
    ordered half-lines are retained. Rules must avoid ``u=eta``; their open
    intervals do so even for the own-energy column ``x=0``.
    """
    if float(u) <= 0 or not np.isfinite([u, weight]).all() or weight <= 0 or float(u) == float(eta):
        raise ValueError("CD imaginary node needs u>0, weight>0, u!=eta")
    if type(external_derivative) is not bool:
        raise ValueError("CD external_derivative must be a static boolean")
    f, dv, dc = _kernels(x, occ, eta, analytic_convention, xp)
    if external_derivative:
        minus = (-f / (dv - 1j * u)**2 - (1 - f) / (dc - 1j * u)**2) * (weight / (2 * np.pi))
        plus = (-f / (dv + 1j * u)**2 - (1 - f) / (dc + 1j * u)**2) * (weight / (2 * np.pi))
    else:
        minus = (f / (dv - 1j * u) + (1 - f) / (dc - 1j * u)) * (weight / (2 * np.pi))
        plus = (f / (dv + 1j * u) + (1 - f) / (dc + 1j * u)) * (weight / (2 * np.pi))
    if n_active is not None or band_valid is not None:
        active = xp.asarray(_physical_band_mask(np.asarray(x), n_active, band_valid))
        minus, plus = xp.where(active, minus, 0.), xp.where(active, plus, 0.)
    ap, am = 0, 0
    for p, m, beta in zip(cp, cm, betas):
        ap = ap + p / (u * u + beta * beta)
        am = am + m / (u * u + beta * beta)
    return _apply(value_i - ap, minus, xp) + _apply(_dagger(value_i, xp) - am, plus, xp)


def _residue_weights(x, f, n_active, band_valid):
    return _physical_band_mask(x, n_active, band_valid), np.where(x < 0, f, -(1 - f))


def _exact_residue_inputs(x, occ, node, *, eta, n_active, band_valid,
                          analytic_convention, xp):
    xh, fh = _inputs(x, occ, eta)
    _kernels(xh, fh, eta, analytic_convention, np)
    node = float(node)
    if not np.isfinite(node) or node < 0:
        raise ValueError("CD exact residue node must be finite and nonnegative")
    active, residue = _residue_weights(xh, fh, n_active, band_valid)
    selected = xp.asarray(np.where(active & (np.abs(xh) == node), residue, 0.))
    return node, selected, xp.asarray(xh < 0)


def real_residue_node(value, value_t, x, occ, node, *, eta, n_active=None,
                      band_valid=None, analytic_convention="time_ordered_fractional",
                      xp=np):
    """One exact positive-line crossing, without an interpolating real grid.

    Both values are evaluated at the common frequency ``node+i eta``;
    ``value_t`` is the -q interaction already transposed and contracted.
    Only states with exactly ``abs(x)==node`` contribute. The caller must
    enumerate every active distinct crossing from its original x table and
    authenticate that coverage; it must not round or cluster crossings.
    Different external read energies are selected after the common-frequency
    matrix sample is formed, never before its ordered partner dagger.
    """
    node, selected, sign = _exact_residue_inputs(x, occ, node, eta=eta,
        n_active=n_active, band_valid=band_valid, analytic_convention=analytic_convention, xp=xp)
    partner = _dagger(value_t, xp) if analytic_convention == "retarded" else value_t
    return (_apply(value, xp.where(sign, 0., selected), xp)
            + _apply(partner, xp.where(sign, selected, 0.), xp))


def real_residue_derivative_node(derivative_s, derivative_t_s, x, occ, node, *,
                                 eta, n_active=None, band_valid=None,
                                 analytic_convention="time_ordered_fractional", xp=np):
    """External-energy derivative at one authenticated exact crossing.

    Both inputs are actual slopes ``d/ds`` at ``z=node+i eta`` with
    ``s=z_Ry²``. Convert the complete slope to ``d/dz`` before the occupied
    retarded partner's dagger, then apply ``d|x|/dE``. The ``x=0`` branch
    matches :func:`real_residue_node`; the matched analytic anchor restores
    the continuous derivative. This is not an interpolation of saved values.
    """
    node, selected, sign = _exact_residue_inputs(x, occ, node, eta=eta,
        n_active=n_active, band_valid=band_valid, analytic_convention=analytic_convention, xp=xp)
    z = node + 1j * eta
    slope, partner = 2 * z * derivative_s, 2 * z * derivative_t_s
    if analytic_convention == "retarded":
        partner = _dagger(partner, xp)
    return (_apply(slope, xp.where(sign, 0., selected), xp)
            - _apply(partner, xp.where(sign, selected, 0.), xp))


def real_part(values, derivatives_s, values_t, derivatives_t_s, x, occ, nodes,
              *, eta, n_active=None, band_valid=None, spacing=None,
              analytic_convention="time_ordered_fractional", xp=np):
    """Cubic Hermite residues on an authenticated uniform positive real grid.

    Values are sampled at ``nodes+i eta``; the partner samples are already
    transposed and contracted with the current pair density. Both derivative
    arrays use the derivative of that same positive real coordinate. Every
    nonzero physical residue must have two bracketing nodes. ``n_active``
    excludes explicitly zero carrier bands, never a physical band tail.
    ``band_valid`` optionally masks per-k ragged spectrum padding explicitly;
    padded wavefunction coefficients must already be exact zero at the caller.
    ``spacing`` can select an integer coarsening of the same node grid.
    """
    xh, fh = _inputs(x, occ, eta)
    _kernels(xh, fh, eta, analytic_convention, np)
    nodes = np.asarray(nodes, np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or nodes[0] != 0 or not np.isfinite(nodes).all():
        raise ValueError("CD residues need a finite uniform grid starting at zero")
    base = nodes[1] - nodes[0]
    if base <= 0 or not np.allclose(np.diff(nodes), base, rtol=2e-12, atol=2e-14 * base):
        raise ValueError("CD residue grid must be strictly increasing and uniform")
    spacing = base if spacing is None else float(spacing)
    stride = round(spacing / base)
    if stride < 1 or not np.isclose(stride * base, spacing, rtol=2e-12, atol=0):
        raise ValueError("CD residue spacing must be an integer multiple of its base grid")
    active, residue = _residue_weights(xh, fh, n_active, band_valid)
    query = np.abs(xh)
    last = nodes[((nodes.size - 1) // stride) * stride]
    if np.any(active & (residue != 0) & (query > last)):
        raise ValueError("CD residue coverage: an active crossing exceeds the last coarse-grid node")
    for samples in (values, derivatives_s, values_t, derivatives_t_s):
        if len(samples) != len(nodes):
            raise ValueError("CD values/derivatives must cover every declared real node")
    index = np.floor(query / spacing).astype(np.int64)
    index = np.minimum(index, max(0, (nodes.size - 1) // stride - 1))
    t = query / spacing - index
    coeff = (2*t**3 - 3*t**2 + 1, -2*t**3 + 3*t**2,
             spacing*(t**3 - 2*t**2 + t), spacing*(t**3 - t**2))
    sign = xp.asarray(xh < 0)
    active = xp.asarray(active)
    index = xp.asarray(index)
    residue = xp.asarray(residue)
    coeff = tuple(xp.asarray(c) for c in coeff)
    total = 0
    for inode in range(0, len(nodes), stride):
        node = inode // stride
        z = nodes[inode] + 1j * eta
        partner_value = values_t[inode]
        partner_slope = 2*z*derivatives_t_s[inode]
        if analytic_convention == "retarded":
            partner_value = _dagger(partner_value, xp)
            partner_slope = _dagger(partner_slope, xp)
        for deriv, value, partner in ((False, values[inode], partner_value),
                                      (True, 2*z*derivatives_s[inode], partner_slope)):
            left, right = coeff[2:] if deriv else coeff[:2]
            selected = xp.where(active, residue * (xp.where(index == node, left, 0.)
                                                   + xp.where(index + 1 == node, right, 0.)), 0.)
            total = total + _apply(value, xp.where(sign, 0., selected), xp)
            total = total + _apply(partner, xp.where(sign, selected, 0.), xp)
    return total


def integrate_ordered(values, derivatives_s, values_t, derivatives_t_s, x, occ,
                      schedule, *, eta, n_active=None, band_valid=None, spacings=None,
                      analytic_convention="time_ordered_fractional", xp=np):
    """Combine analytic anchor, infinite imaginary integral and real residues.

    ``schedule`` declares ``real_nodes`` and ``imaginary`` rules, each with
    ``indices``, ``nodes`` and ``weights``. The sample at index zero is
    ``z=i eta``. Returns a dictionary keyed by ``(n_imaginary, spacing_Ry)``.
    Refinement comparisons must keep eta, physical bands, states and tail-map
    scale fixed. No embedded estimator or accuracy refusal is implied.
    """
    x, occ = _inputs(x, occ, eta)
    anchor, cp, cm, beta = anchor_part(values[0], derivatives_s[0], x, occ, eta=eta,
                                      n_active=n_active, band_valid=band_valid,
                                      analytic_convention=analytic_convention, xp=xp)
    nodes = np.asarray(schedule["real_nodes"], np.float64)
    spacings = (nodes[1] - nodes[0],) if spacings is None else tuple(spacings)
    real = {h: real_part(values[:len(nodes)], derivatives_s[:len(nodes)],
                        values_t[:len(nodes)], derivatives_t_s[:len(nodes)], x, occ, nodes,
                        eta=eta, n_active=n_active, band_valid=band_valid, spacing=h,
                        analytic_convention=analytic_convention, xp=xp) for h in spacings}
    out = {}
    for rule in schedule["imaginary"]:
        indices, u, weights = rule["indices"], rule["nodes"], rule["weights"]
        if len(indices) != len(u) or len(u) != len(weights) or not len(u):
            raise ValueError("CD imaginary rule has inconsistent or empty arrays")
        remainder = 0
        for i, ui, wi in zip(indices, u, weights):
            remainder = remainder + imag_remainder_node(values[i], ui, wi, xp.asarray(x), occ,
                                                         cp, cm, beta, eta=eta,
                                                         n_active=n_active, band_valid=band_valid,
                                                         analytic_convention=analytic_convention, xp=xp)
        for h, residues in real.items():
            out[(len(u), h)] = anchor + remainder + residues
    return out
