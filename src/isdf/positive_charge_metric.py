"""Physical low-G charge density with a completed periodic local correction.

The ordinary ISDF factor solves the smooth, delta and exact-monopole columns.
The radial owner supplies free delta-delta minus compensation and both
periodic neutral-potential adjoints. The global compensation Gram stays on
the two-dimensional face and uses the public distributed matrix action.
"""
from __future__ import annotations

import numpy as np

BODY_METRIC = 'physical_low_local_high'


def plan_positive_periodic_action(mesh, *, geometry, lm, centroid_basis,
        fft_points, cache_path, cache_file_sha256):
    """Price and retain the public action before allocating the cache.

    The shape descriptor authenticates ownership and plans native workspace;
    it asserts no physical payload. The loader must still authenticate the
    externally pinned file and the exact geometry before the action is used.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_axis
    from isdf.atomic_coulomb import make_periodic_compensation_action
    from isdf.coulomb_fourier_cache import PERIODIC_SCHEMA
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    nq = len(geometry['operator_q_fractional'])
    nm = len(geometry['atom_centres_bohr'])*len(lm)
    axis = padded_axis(nm, mesh, name='periodic moment',
        specs=((face.spec, 1), (face.spec, 2)))
    descriptor = dict(gram=jax.ShapeDtypeStruct((nq, axis.carrier, axis.carrier),
        np.complex128, sharding=face), moment_axis=axis,
        metadata=dict(schema=PERIODIC_SCHEMA, geometry=geometry, logical_shape=[nq, nm, nm]),
        path=str(cache_path), file_sha256=cache_file_sha256)
    action, receipt = make_periodic_compensation_action(mesh, descriptor,
        centroid_basis=centroid_basis, fft_points=fft_points)
    arrays = receipt['arrays_per_rank']
    price = (arrays['gram_bytes']+3*arrays['moment_rows_bytes']+arrays['result_bytes']
             +sum(receipt['vendor_gemm_workspace_bytes_per_rank']))
    return dict(action=action, receipt=receipt, geometry=geometry,
                moment_axis=axis, resident_bound_bytes_per_rank=price)


def positive_radial_coulomb_provider(zeta_g, rhs, *, monopole_rhs, radius,
        weights_dr, lm, centers_cart, q_frac, gvec_components, q_plus_G_cart,
        cell_volume, fft_points, support_radius, minimum_atom_image_distance,
        body_cutoff_ry, periodic_cache, interpolation_degree, quadrature_order,
        fourier_points=4097, prepared_cache=None, periodic_plan=None):
    """Bind the local-high completion to the existing charge solve.

    ``onsite`` returns the q-owned free-local and periodic-mean action.
    ``periodic_moment_rows`` returns grid-sum multipoles on the same q owners,
    with canonical atom/lm carrier tails zero. ZetaG moves those rows to its
    existing face before ``periodic_action``; no global Gram is replicated
    or converted into q-owned full matrices.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from common.collectives import device_put_process_local
    from runtime.padding import pad_to_axis
    from isdf.atomic_coulomb import (_angular_channels,
        make_periodic_compensation_action, radial_coulomb_provider)
    from isdf.atomic_hartree import neutral_potential_mean_rows
    from vcoul import CoulombGeometry, get_kernel, v_qG_table

    store, mesh = zeta_g.store, zeta_g.mesh
    harmonics = _angular_channels(lm)
    centers, q = np.asarray(centers_cart, float), np.asarray(q_frac, float)
    g, kg = np.asarray(gvec_components, float), np.asarray(q_plus_G_cart, float)
    geometry = periodic_cache['metadata']['geometry']
    reciprocal = np.asarray(geometry['reciprocal_rows_bohr_inverse'], float)
    nr, nh, na = len(radius), len(harmonics), len(centers)
    nf, qpad, mu = na*nh*nr, int(store.Q_pad), int(store.mu_pad)
    qspec = P(('x', 'y'), None, None)
    if (not na or centers.shape != (na, 3) or q.shape != (store.Q, 3)
            or g.shape != (store.Q, 3, store.g_axis.logical)
            or kg.shape != (store.Q, store.g_axis.logical, 3)
            or not all(np.isfinite(a).all() for a in (centers, q, g, kg))
            or not np.isfinite(minimum_atom_image_distance)
            or minimum_atom_image_distance <= 2*support_radius
            or not np.allclose(kg, (g.transpose(0, 2, 1)+q[:, None])@reciprocal,
                               rtol=2e-13, atol=2e-13)
            or not np.array_equal(centers, geometry['atom_centres_bohr'])
            or not np.array_equal(q, geometry['operator_q_fractional'])
            or float(cell_volume) != float(geometry['cell_volume_bohr3'])
            or float(support_radius) != float(geometry['support_radius_bohr'])
            or not np.array_equal(harmonics, periodic_cache['metadata']['lm'])
            or zeta_g.mu_basis is None):
        raise ValueError('positive charge requires the authenticated periodic q/G/atom/centroid geometry')
    bare = v_qG_table(get_kernel(3), q, g,
        geometry=CoulombGeometry(reciprocal, float(cell_volume)),
        vcoul_cutoff_ry=float(body_cutoff_ry), v_head_fn=None)
    provider = radial_coulomb_provider(zeta_g, rhs, monopole_rhs=monopole_rhs,
        radius=radius, weights_dr=weights_dr, lm=harmonics, centers_cart=centers,
        q_plus_G_cart=kg, cell_volume=cell_volume, fft_points=fft_points,
        support_radius=support_radius, interpolation_degree=interpolation_degree,
        quadrature_order=quadrature_order, fourier_points=fourier_points,
        prepared_cache=prepared_cache)
    if (provider['rhs'].shape != (qpad, mu, nf+na)
            or provider['exact_monopole_offset'] != nf):
        raise ValueError('positive charge solves delta and exact M0 columns only')
    tables = provider['radial_tables']
    degree_rows = np.searchsorted(tables['degrees'], harmonics[:, 0])
    monopole = int(np.flatnonzero(np.all(harmonics == (0, 0), axis=1))[0])
    mean = neutral_potential_mean_rows(tables, support_radius=support_radius,
        fft_points=fft_points, cell_volume=cell_volume)
    put = lambda a, spec=P(): device_put_process_local(
        np.asarray(a), NamedSharding(mesh, spec))
    moments = put(tables['moments'][degree_rows])
    mean_row = put(mean['potential_mean_row'])
    gamma = np.all(q == 0, axis=1)
    if np.count_nonzero(gamma) != 1:
        raise ValueError('positive charge q domain requires one physical Gamma')
    gamma_pad = np.zeros(qpad, bool)
    gamma_pad[:store.Q] = gamma
    gamma_device = put(gamma_pad, P(('x', 'y')))
    charge_scale = np.sqrt(4*np.pi)*float(fft_points)/float(cell_volume)
    moment_axis = periodic_cache['moment_axis']

    def local_mean(free, coefficients, phi_row, qzero):
        density = coefficients[..., :nf].reshape(coefficients.shape[0], mu, na, nh, nr)
        exact = coefficients[..., nf:]
        charge = charge_scale*jnp.sum(exact, axis=-1)
        # Equal epsilon*g0 enrichment in delta and compensation cancels in Phi.
        phi = jnp.einsum('qmar,r->qm', density[:, :, :, monopole], phi_row)
        adjoints = -2/float(cell_volume)*(charge.conj()[:, :, None]*phi[:, None, :]
                                        +phi.conj()[:, :, None]*charge[:, None, :])
        return free+jnp.where(qzero[:, None, None], adjoints, 0.)

    complete_mean = jax.jit(shard_map(local_mean, mesh=mesh,
        in_specs=(qspec, qspec, P(), P(('x', 'y'))), out_specs=qspec,
        check_vma=False), donate_argnums=(0,))
    free_onsite = provider['onsite']

    def onsite(coefficients):
        return complete_mean(free_onsite(coefficients), coefficients, mean_row, gamma_device)

    def moment_rows(coefficients, radial_rows):
        density = coefficients[..., :nf].reshape(coefficients.shape[0], mu, na, nh, nr)
        values = jnp.einsum('qmahr,hr->qmah', density, radial_rows)
        values = values.at[:, :, :, monopole].set(coefficients[..., nf:])
        return pad_to_axis(values.reshape(coefficients.shape[0], mu, na*nh),
                           moment_axis, axis=2)

    rows_kernel = jax.jit(shard_map(moment_rows, mesh=mesh,
        in_specs=(qspec, P()), out_specs=qspec, check_vma=False))
    if periodic_plan is None:
        action, receipt = make_periodic_compensation_action(mesh, periodic_cache,
            centroid_basis=zeta_g.mu_basis, fft_points=fft_points)
    else:
        if (periodic_plan['moment_axis'] != moment_axis
                or any(not np.array_equal(periodic_plan['geometry'][key], value)
                       for key, value in geometry.items())
                or periodic_plan['receipt']['cache_file_sha256'] != periodic_cache['file_sha256']):
            raise ValueError('prepared periodic action differs from the admitted physical cache')
        action, receipt = periodic_plan['action'], periodic_plan['receipt']
    # Preserve the radial handles and expose the already-compiled positive
    # completion stages with their original explicit table operands.
    aot_metadata = dict(provider['aot_metadata'])
    aot_metadata.update(
        mean_completion=dict(kernel=complete_mean, table_operands=(mean_row, gamma_device)),
        periodic_moments=dict(kernel=rows_kernel, table_operands=(moments,)))
    provider.update(onsite=onsite, body_metric=BODY_METRIC, bare_v_table=bare,
        aot_metadata=aot_metadata,
        periodic_moment_rows=lambda coefficients: rows_kernel(coefficients, moments),
        periodic_action=action, periodic_cache=periodic_cache,
        periodic_action_receipt=receipt, raw_columns=nf+na, PS_solved_columns=0,
        local_cross_policy='physical_low_plus_completed_bare_delta_high')
    return provider
