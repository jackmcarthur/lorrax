"""Direct current Hartree from fitting's physical sample products.

These upstream kernels consume explicit full-zone four-spinor samples and
one Gamma current-interaction block at a time. They neither select a GW
model nor relax the reconstructed resident-charge admission gate. The
current interaction includes its signed metric and fitted Coulomb integral;
sample products require no spatial quadrature or additional exchange sign.
"""
from __future__ import annotations

import numpy as np


def _current_basis_rows(rows):
    """Validate the current-channel change of coordinates, never band frame."""
    basis = np.eye(3, dtype=np.complex128) if rows is None else np.asarray(rows)
    if (basis.shape != (3, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis.conj().T @ basis, np.eye(3),
                               rtol=0, atol=2e-12)):
        raise ValueError('Current Hartree requires a unitary current-component basis')
    return np.asarray(basis, dtype=np.complex128)


def make_occupied_current_trace(mesh, occupations, full_kweights, *,
                                point_active, current_basis_rows=None):
    """Form ``d_a(mu)=sum_kn w_k f_kn psi^dagger Gamma_a psi``.

    Parameters
    ----------
    mesh : Mesh
        The actual fitting mesh, with band axis X and sample axis Y.
    occupations : ndarray, (n_full_k, n_band_carrier)
        Physical occupations in [0,1], including zero band padding. For
        CrI3 these are the same 130 occupied labels on all nine rows. ISDF
        occupied loss weights are not occupations.
    full_kweights : ndarray, (n_full_k,)
        Nonnegative physical BZ weights summing to one.
    point_active : ndarray, (n_mu_carrier,)
        Actual packed sample mask. Ghost samples never contribute.
    current_basis_rows : ndarray, (3,3), optional
        Unitary B with Gamma_a=sum_i B_ai alpha_i. Canonical stored TT
        blocks are already Cartesian and require this argument to be None.

    Returns
    -------
    callable
        Takes physical, unweighted complex128 samples of shape
        (n_full_k,n_band_carrier,4,n_mu_carrier), P(None,'x',None,'y').
        Returns three current traces at P(None,'y'), in sample-product
        units. No N/Omega or radial/angular weight is introduced.
    """
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.gamma_matrices import gamma_apply, gamma_perm_phase

    occ, weights, active = map(np.asarray, (occupations, full_kweights, point_active))
    if (occ.ndim != 2 or np.iscomplexobj(occ) or not np.isfinite(occ).all()
            or np.any(occ < 0) or np.any(occ > 1)
            or weights.shape != (occ.shape[0],) or np.iscomplexobj(weights)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0, atol=2e-12)
            or active.ndim != 1 or active.dtype != np.dtype(bool)
            or occ.shape[1] % int(mesh.shape['x'])
            or len(active) % int(mesh.shape['y'])):
        raise ValueError('Current Hartree requires physical full-zone occupations and packed sample masks')
    basis = _current_basis_rows(current_basis_rows)
    put = lambda value, spec: device_put_process_local(
        np.asarray(value), NamedSharding(mesh, spec))
    weighted = put(weights[:, None] * occ, P(None, 'x'))
    live = put(active, P('y'))
    B = put(basis, P())
    alpha = tuple(gamma_perm_phase(i) for i in (1, 2, 3))

    def trace_local(face, f, mask, B):
        face = jnp.where((f > 0)[:, :, None, None] & mask[None, None, None], face, 0.)
        current = jnp.stack(tuple(jnp.einsum('knsm,knsm,kn->m', face.conj(),
            gamma_apply(face, perm, phase, axis=2), f)
            for perm, phase in alpha))
        current = jax.lax.psum(current, 'x')
        return jnp.where(mask[None], B @ current, 0.)

    kernel = jax.jit(shard_map(trace_local, mesh=mesh,
        in_specs=(P(None, 'x', None, 'y'), P(None, 'x'), P('y'), P()),
        out_specs=P(None, 'y'), check_vma=False))

    def trace(face):
        expected = (*occ.shape, 4, len(active))
        if (face.shape != expected or np.dtype(face.dtype) != np.dtype(np.complex128)
                or face.sharding != NamedSharding(mesh, P(None, 'x', None, 'y'))):
            raise ValueError('Current Hartree samples differ from the explicit full-zone physical carrier')
        return kernel(face, weighted, live, B)
    trace.lower = lambda face: kernel.lower(face, weighted, live, B)
    return trace


def make_gamma_current_block_action(mesh, point_active):
    """Form ``a(mu)=sum_nu T_Gamma(mu,nu)d(nu)`` for one block.

    T is complex128 at P('x','y'), d at P('y'), and the output is the
    small replicated sample vector at P(). The operator is already in Ry
    with the transverse metric included. This is a direct contraction:
    no exchange-minus, energy half-factor, or sample quadrature weight.
    """
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local

    active = np.asarray(point_active)
    if (active.ndim != 1 or active.dtype != np.dtype(bool)
            or len(active) % int(mesh.shape['x'])
            or len(active) % int(mesh.shape['y'])):
        raise ValueError('Current Hartree interaction requires the same packed sample basis on both legs')
    mask_x = device_put_process_local(active, NamedSharding(mesh, P('x')))
    mask_y = device_put_process_local(active, NamedSharding(mesh, P('y')))

    def action_local(T, d, left_live, right_live):
        T = jnp.where(left_live[:, None] & right_live[None], T, 0.)
        d = jnp.where(right_live, d, 0.)
        partial = jax.lax.psum(T @ d, 'y')
        return jax.lax.all_gather(partial, 'x', axis=0, tiled=True)

    count = len(active)
    kernel = jax.jit(shard_map(action_local, mesh=mesh,
        in_specs=(P('x', 'y'), P('y'), P('x'), P('y')),
        out_specs=P(), check_vma=False))

    def action(T, d):
        if (T.shape != (count, count) or d.shape != (count,)
                or np.dtype(T.dtype) != np.dtype(np.complex128)
                or np.dtype(d.dtype) != np.dtype(np.complex128)
                or T.sharding != NamedSharding(mesh, P('x', 'y'))
                or d.sharding != NamedSharding(mesh, P('y'))):
            raise ValueError('Current Hartree block and source differ from their packed basis/sharding')
        return kernel(T, d, mask_x, mask_y)
    action.lower = lambda T, d: kernel.lower(T, d, mask_x, mask_y)
    return action


def make_sampled_current_potential_matrix(mesh, point_active, band_active, *,
                                          current_basis_rows=None):
    """Contract the Hermitian-density derivative of ``E=1/2 d^dagger T d``.

    Physical receiving samples are (k,band,4,mu) at
    P(None,'x',None,'y'). The potential is (3,mu), in Ry/sample-product
    units, replicated. Gamma blocks contain the fitted metric, so there
    is no centroid/radial/angular quadrature weight. A fit-basis field is
    converted once with B^dagger; canonical stored TT fields are Cartesian.
    Cartesian d is real for Hermitian occupied density. Its constrained
    energy derivative is A=Re(Td), so Re is taken AFTER basis conversion,
    without a half in H. The caller retains raw Td to diagnose fit reality;
    this is a defined variational operator, not a numerical symmetrization.

    The ket's band axis is gathered inside the mesh-local kernel, then
    the point contraction is reduced and its second band axis sliced.
    No band-pair-by-sample density tensor is formed. Returns a matrix at
    P(None,'x','y'); padded bands and samples are exact zero.
    """
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.gamma_matrices import gamma_apply, gamma_perm_phase

    points, bands = map(np.asarray, (point_active, band_active))
    if (points.ndim != 1 or bands.ndim != 1
            or points.dtype != np.dtype(bool) or bands.dtype != np.dtype(bool)
            or len(points) % int(mesh.shape['y'])
            or len(bands) % int(mesh.shape['x'])
            or len(bands) % int(mesh.shape['y'])):
        raise ValueError('Current Hartree receiving masks require the canonical band/sample carrier')
    basis = _current_basis_rows(current_basis_rows)
    put = lambda value, spec: device_put_process_local(np.asarray(value), NamedSharding(mesh, spec))
    bx, pm, B = put(bands, P('x')), put(points, P('y')), put(basis, P())
    alpha = tuple(gamma_perm_phase(i) for i in (1, 2, 3))
    n_y = len(bands) // int(mesh.shape['y'])
    n_mu_y = len(points) // int(mesh.shape['y'])

    def matrix_local(face, potential, live_bands, live_points, B):
        face = jnp.where(live_bands[None, :, None, None]
                         & live_points[None, None, None], face, 0.)
        all_kets = jax.lax.all_gather(face, 'x', axis=1, tiled=True)
        cartesian = jnp.real(B.conj().T @ potential)
        local_field = jax.lax.dynamic_slice_in_dim(cartesian,
            jax.lax.axis_index('y') * n_mu_y, n_mu_y, axis=1)
        phi = jnp.zeros_like(all_kets)
        for i, (perm, phase) in enumerate(alpha):
            phi = phi + local_field[i] * gamma_apply(all_kets, perm, phase, axis=2)
        matrix = jax.lax.psum(jnp.einsum('kmsr,knsr->kmn', face.conj(), phi), 'y')
        return jax.lax.dynamic_slice_in_dim(matrix,
            jax.lax.axis_index('y') * n_y, n_y, axis=2)

    kernel = jax.jit(shard_map(matrix_local, mesh=mesh,
        in_specs=(P(None, 'x', None, 'y'), P(), P('x'), P('y'), P()),
        out_specs=P(None, 'x', 'y'), check_vma=False))

    def contract(face, potential):
        if (face.ndim != 4 or face.shape[1:] != (len(bands), 4, len(points))
                or potential.shape != (3, len(points))
                or np.dtype(face.dtype) != np.dtype(np.complex128)
                or np.dtype(potential.dtype) != np.dtype(np.complex128)
                or face.sharding != NamedSharding(mesh, P(None, 'x', None, 'y'))
                or potential.sharding != NamedSharding(mesh, P())):
            raise ValueError('Current Hartree receiving samples/field differ from their canonical packed carrier')
        return kernel(face, potential, bx, pm, B)
    contract.lower = lambda face, potential: kernel.lower(face, potential, bx, pm, B)
    return contract


def sampled_transverse_hartree(reader, samples, source, *, resident_bytes):
    """Finish the direct spatial-current operator before orbital donation.

    Parameters
    ----------
    reader : BispinorVqReader
        Canonical complete current interaction, opened with the actual
        current centroid basis and typed family plan. Its stored blocks
        are Cartesian, even when the current fits used a helical basis.
    samples : dict
        ``source_full`` is the actual full-BZ unweighted four-spinor sample
        bank after the full source-band frame. ``receiving`` is the matching
        receiving view, with the same packed sample axis. Both are
        complex128 at P(None,'x',None,'y'). ``point_active`` and
        ``receiving_band_active`` are canonical boolean masks.
    source : dict
        ``occupations`` and ``full_kweights`` come from the physical
        occupation resolver. This numerical seam does not authenticate a
        reconstructed source/frame or produce a resident provenance record;
        those joins must close before its output is served by GW.
    resident_bytes : float
        Caller-priced external live device bytes per rank, including the
        source/receiving banks. AOT scratch plus one Gamma block is admitted
        in addition. Existing source/provider banks cannot be omitted.

    Returns
    -------
    dict
        Native receiving matrix ``parent_kij_ry`` at P(None,'x','y') and
        the three current traces/raw potential, with an AOT memory receipt
        and explicit current/potential reality diagnostics.
        This is not a paired charge/current resident field admission.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.restart_bundle import BispinorVqReader
    from common.gpu_utils import device_budget_bytes, record_stage_price
    from common.collectives import gather_to_host

    if (not isinstance(reader, BispinorVqReader)
            or not reader._current_family_authenticated
            or samples['source_full'].shape[0] != int(np.prod(reader.kgrid))
            or not np.isfinite(resident_bytes) or resident_bytes < 0):
        raise ValueError('Current Hartree requires the authenticated current reader, explicit full BZ and live price')
    mesh = reader._mesh
    points = np.asarray(samples['point_active'])
    bands = np.asarray(samples['receiving_band_active'])
    trace = make_occupied_current_trace(mesh, source['occupations'], source['full_kweights'],
        point_active=points)
    action = make_gamma_current_block_action(mesh, points)
    contract = make_sampled_current_potential_matrix(mesh, points, bands)
    vector = jax.ShapeDtypeStruct((len(points),), jnp.complex128,
                                 sharding=NamedSharding(mesh, P('y')))
    block = jax.ShapeDtypeStruct((len(points), len(points)), jnp.complex128,
                                sharding=NamedSharding(mesh, P('x', 'y')))
    potential = jax.ShapeDtypeStruct((3, len(points)), jnp.complex128,
                                    sharding=NamedSharding(mesh, P()))
    hermitian = jax.jit(lambda T: (jnp.max(jnp.abs(T - T.conj().T)),
                                    jnp.max(jnp.abs(T))))
    quotes = []
    for name, lowered in (
            ('occupied_full_BZ_current', trace.lower(samples['source_full'])),
            ('Gamma_current_block_matvec', action.lower(block, vector)),
            ('Gamma_diagonal_Hermiticity', hermitian.lower(block)),
            ('sampled_alpha_A_matrix', contract.lower(samples['receiving'], potential))):
        stats = lowered.compile().memory_analysis()
        quotes.append(dict(owner=name, arguments=int(stats.argument_size_in_bytes),
            outputs=int(stats.output_size_in_bytes), temporary=int(stats.temp_size_in_bytes),
            aliases=int(stats.alias_size_in_bytes)))
    # One TT block, two three-current small vectors, and the receiving
    # matrix coexist. Exact AOT argument/output/scratch is conservative
    # in addition to the separately priced caller-owned live banks.
    persistent = (16 * len(points)**2 / int(mesh.size)
                  + 6 * 3 * 16 * len(points)
                  + 16 * samples['receiving'].shape[0] * len(bands)**2 / int(mesh.size))
    aot = max(q['arguments'] + q['outputs'] + q['temporary'] - q['aliases'] for q in quotes)
    price = dict(external_resident_bytes_per_rank=float(resident_bytes),
        incremental_persistent_bytes_per_rank=persistent, exact_AOT_bytes_per_rank=aot,
        planned_bytes_per_rank=float(resident_bytes) + persistent + aot,
        budget_bytes_per_rank=float(device_budget_bytes()), quotes=quotes)
    if price['planned_bytes_per_rank'] > price['budget_bytes_per_rank']:
        raise MemoryError('Current Hartree occupied trace/field/receiving AOT exceeds live budget')
    record_stage_price('isdf.current_hartree', price['planned_bytes_per_rank'])
    d = trace(samples['source_full'])
    d_imag = float(gather_to_host(jnp.max(jnp.abs(jnp.imag(d)))))
    d_scale = float(gather_to_host(jnp.max(jnp.abs(d))))
    if d_imag > 2e-12 + 256*np.finfo(float).eps*d_scale:
        raise ValueError('Occupied Cartesian current is not real for its Hermitian physical density')
    # The source is a physical Hermitian occupation trace. Retain its
    # roundoff diagnostic; the actual constrained energy uses real d.
    d_real = jnp.real(d).astype(jnp.complex128)
    source_rows = tuple(jax.device_put(d_real[j], NamedSharding(mesh, P('y'))) for j in range(3))
    a = jax.device_put(jnp.zeros((3, len(points)), jnp.complex128), NamedSharding(mesh, P()))
    for i in range(3):
        for j in range(i, 3):
            T = reader.get_current_gamma_tile(i + 1, j + 1)
            if i == j:
                skew, scale = hermitian(T)
                skew, scale = float(gather_to_host(skew)), float(gather_to_host(scale))
                if skew > 2e-12 + 256*np.finfo(float).eps*scale:
                    raise ValueError('Gamma current energy requires Hermitian diagonal interaction blocks')
            a = a.at[i].add(action(T, source_rows[j]))
            if i != j:
                adjoint = jax.device_put(T.conj().T, NamedSharding(mesh, P('x', 'y')))
                a = a.at[j].add(action(adjoint, source_rows[i]))
                a.block_until_ready()
                del adjoint
            a.block_until_ready()
            del T
    matrix = contract(samples['receiving'], a)
    matrix.block_until_ready()
    a_imag = float(gather_to_host(jnp.max(jnp.abs(jnp.imag(a)))))
    a_scale = float(gather_to_host(jnp.max(jnp.abs(a))))
    return dict(parent_kij_ry=matrix, occupied_current=d, sample_potential_ry=a,
        effective_sample_potential_ry=jnp.real(a), memory_receipt=price,
        reality_diagnostics=dict(occupied_current_imaginary_max=d_imag,
            occupied_current_scale=d_scale, raw_potential_imaginary_max=a_imag,
            raw_potential_scale=a_scale),
        variational_convention='Hermitian-density derivative of 1/2 d^dagger T_Gamma d; Re after Cartesian conversion; nohalf in H',
        scope='Upstream sampled current Hartree only; paired source/operator admission required')


def prepare_paired_current_hartree(charge_record, current_parent, reader, *,
                                   wfn, mesh, resident_bytes):
    """Finish the current derivative from the same physical full-source carrier.

    The current parent's typed spin/Bloch action unfolds explicit full k;
    scalar star weights are never used for the current. The full physical
    source-band frame has acted before the public carrier selection; all
    physically occupied rows must remain present. The independently produced
    canonical current descriptor must match charge's WFN/frame/FILE source
    and this reader's actual current centroid/q chart before allocation.
    Charge and current receiving ranges are the same independent public
    logical interval, not the ISDF fit's left-band interval. Unoccupied
    physical source bands outside that interval are not fabricated as ghosts.
    """
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.gpu_utils import device_budget_bytes, record_stage_price
    from runtime.padding import pad_to_axis
    from file_io.tagged_arrays import _hartree_band_axis
    from .isdf_augmentation import _hartree_source_request
    from .augmentation_hartree import _current_operator_binding, pair_resident_hartree

    provenance = charge_record['provenance']
    current_operator_record = reader.physical_current_operator_record()
    _current_operator_binding(provenance, current_operator_record)
    plan = current_parent.plan
    face, _ = current_parent.projection_faces()
    basis = reader._mu_bases[1]
    count = int(wfn.nbands)
    lo, hi = map(int, provenance['band_range'])
    spec = P(None, 'x', None, 'y')
    if (plan.mesh_xy is not mesh or plan.nspinor != 4
            or face.dtype != jnp.complex128 or face.ndim != 4
            or face.shape[0] != plan.n_parent or face.shape[1] < hi
            or face.shape[2:] != (4, basis.n_packed)
            or face.sharding != NamedSharding(mesh, spec)
            or not np.array_equal(plan.parent_full_rows, provenance['parent_full_rows'])
            or not np.array_equal(plan.k_parent_frac, provenance['parent_k_frac'])
            or not 0 == lo < hi <= count):
        raise ValueError('GATE resident_current_source: occupied current carrier/FILE receiving domain changed')
    physical = _hartree_source_request(
        dict(occupations=None, full_kweights=None, spin_degeneracy=1.),
        wfn=wfn, plan=plan, public_range=(lo, hi))
    source_binding = provenance['source_binding']
    if any(physical[key] != source_binding[key] for key in
           ('occupations_sha256', 'full_kweights_sha256', 'spin_degeneracy')):
        raise ValueError('GATE resident_current_source: charge/current physical occupations differ')
    covered = min(count, face.shape[1])
    if np.any(physical['occupations'][:, covered:]):
        raise ValueError('GATE resident_current_source: physical occupied support exceeds the actual current carrier')
    axis = _hartree_band_axis(hi-lo, mesh)
    receiving = jax.jit(lambda value: pad_to_axis(value[:, lo:hi], axis, axis=1),
                        out_shardings=NamedSharding(mesh, spec))
    unfold = jax.jit(shard_map(lambda value: plan.unfold_face(
        value, spin_axis=2, mu_axis=3, mesh_axis='y'), mesh=mesh,
        in_specs=spec, out_specs=spec, check_vma=False))
    quotes = []
    for name, lowered in (('physical_full_BZ_current_samples', unfold.lower(face)),
                          ('full_receiving_current_samples', receiving.lower(face))):
        stats = lowered.compile().memory_analysis()
        quotes.append(dict(owner=name, arguments=int(stats.argument_size_in_bytes),
            outputs=int(stats.output_size_in_bytes), temporary=int(stats.temp_size_in_bytes),
            aliases=int(stats.alias_size_in_bytes)))
    full_bytes = 16*plan.n_full*face.shape[1]*4*basis.n_packed/int(mesh.size)
    receiving_bytes = 16*plan.n_parent*axis.carrier*4*basis.n_packed/int(mesh.size)
    aot = max(q['arguments']+q['outputs']+q['temporary']-q['aliases'] for q in quotes)
    planned = float(resident_bytes)+full_bytes+receiving_bytes+aot
    if (not np.isfinite(resident_bytes) or resident_bytes < 0
            or planned > float(device_budget_bytes())):
        raise MemoryError('Current Hartree explicit full-BZ/receiving sample AOT exceeds live budget')
    record_stage_price('isdf.current_hartree_sample_transport', planned)
    source_full = unfold(face)
    receive = receiving(face)
    occ = np.zeros((plan.n_full, face.shape[1]), np.float64)
    occ[:, :covered] = physical['occupations'][np.asarray(plan.irr_idx), :covered]
    result = sampled_transverse_hartree(reader,
        dict(source_full=source_full, receiving=receive,
            point_active=basis.active_mask,
            receiving_band_active=np.arange(axis.carrier) < hi-lo),
        dict(occupations=occ, full_kweights=physical['full_kweights']),
        resident_bytes=float(resident_bytes)+full_bytes+receiving_bytes)
    paired = pair_resident_hartree(charge_record, result['parent_kij_ry'],
                                  current_operator_record)
    result['source_transport_memory_receipt'] = dict(planned_bytes_per_rank=planned,
        external_resident_bytes_per_rank=float(resident_bytes),
        full_source_samples_bytes_per_rank=full_bytes,
        receiving_samples_bytes_per_rank=receiving_bytes, exact_AOT_quotes=quotes)
    return paired, result
