"""Exact source-bound receiving charge Hartree from an existing prepared state.

The orbital frame/atomic caches are inputs already authenticated by preparation.
No WFN coefficient read, projector, Gram or inverse-root factory occurs here.
Endpoint faces remain distributed; fixed-source point responses precede pairs.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib,json,time
from pathlib import Path
import numpy as np

def _digest(value):
    array = np.ascontiguousarray(value)
    header = repr((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


@lru_cache(maxsize=None)
def _receiving_point_kernel(mesh, band_tile, source_count):
    """Retain one sharded receiving contraction for equivalent packet shapes.

    Responses and endpoint arrays remain runtime operands; this cache owns
    only the compiled-callable factory keyed by mesh and static band geometry.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax import shard_map

    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    tile, nu = int(band_tile), int(source_count)
    fs = P(None, 'x', None, 'y')
    def apply_local(ps, delta, exact, wd, wp, m0, point_live):
        nrow, npoint = ps.shape[1], ps.shape[-1]
        nb = nrow*px
        pr = jnp.where(point_live[None, None, None], ps, 0.)[0]
        dr = jnp.where(point_live[None, None, None], delta, 0.)[0]
        initial = jnp.zeros((6*nu, nrow, nb), jnp.complex128)

        def weighted_overlap(bra, ket, weight):
            return jnp.einsum('ism,cjsm->cij', bra.conj(),
                              weight[:, None, None, :]*ket[None])

        def add_columns(matrix, step):
            ids = step*tile+jnp.arange(tile)-jax.lax.axis_index('x')*nrow
            live = (ids >= 0) & (ids < nrow)
            ids = jnp.clip(ids, 0, nrow-1)
            pc, dc = [jax.lax.psum(jnp.where(live[:, None, None],
                jnp.take(value, ids, axis=0), 0.), 'x') for value in (pr, dr)]
            # Retain both mixed adjoints and delta-delta explicitly.
            value = (weighted_overlap(pr, dc, wd)
                     + weighted_overlap(dr, pc, wd)
                     + weighted_overlap(dr, dc, wd)).reshape(6, nu, nrow, tile)
            value = value.at[3].add(weighted_overlap(pr, pc, wp))
            value = jax.lax.psum(value.reshape(6*nu, nrow, tile), 'y')
            return jax.lax.dynamic_update_slice(matrix, value,
                (jnp.int32(0), jnp.int32(0), step*tile)), None

        matrix = jax.lax.scan(add_columns, initial,
            jnp.arange(nb//tile, dtype=jnp.int32), unroll=1)[0]
        matrix = jax.lax.dynamic_slice(matrix,
            (jnp.int32(0), jnp.int32(0), jax.lax.axis_index('y')*(nb//py)),
            (6*nu, nrow, nb//py))
        matrix = matrix.reshape(6, nu, nrow, nb//py)
        return matrix+jnp.einsum('cua,ija->cuij', m0, exact[0])

    return jax.jit(shard_map(apply_local, mesh=mesh,
        in_specs=(fs, fs, P(None, 'x', 'y', None), P(None, 'y'),
                  P(None, 'y'), P(), P('y')),
        out_specs=P(None, None, 'x', 'y'), check_vma=False))


def _receiving_angular_contract(directions, weights, lm, harmonics, source):
    """Authenticate the original spherical rule, including stored R roundoff.

    A source-bound orbit rule must match the SAME stage's nodes, weights and
    harmonic table exactly. The symmetry owner stores Cartesian rotations
    rounded to ten decimal places, so these original nodes need not have
    unit norm within 2e-12. Their norm envelope follows from the actual
    rotation metric defect, not a caller-selected tolerance. Unbound unit
    rules retain the existing 2e-12 guard. No node is renormalized here.
    """
    norm = np.linalg.norm(directions, axis=1)
    if source is None:
        if not np.allclose(norm, 1., rtol=0, atol=2e-12):
            raise ValueError('Receiving adjoints require the canonical atomic geometry and tile extent')
        return dict(policy='unbound_unit_spherical_rule',
            maximum_direction_norm_defect=float(np.max(abs(norm-1.))),
            direction_norm_tolerance=2e-12)
    if not isinstance(source, dict) or set(source) != {'control', 'cartesian_rotations'}:
        raise ValueError('Receiving angular source requires exact stage controls and Cartesian rows')
    from gw import isdf_augmentation as stage

    rotations = np.asarray(source['cartesian_rotations'], dtype=np.float64)
    if not np.isfinite(rotations).all():
        raise ValueError('Receiving angular source requires finite Cartesian rows')
    expected = stage._orbit_angular_quadrature(source['control'], rotations)
    originals = (expected[0], expected[1], expected[2], expected[3])
    supplied = (directions, weights, lm, harmonics)
    if any(a.dtype != b.dtype or not np.array_equal(a, b)
           for a, b in zip(supplied, originals)):
        raise ValueError('Receiving angular cloud differs from the SAME stage rule')
    defect = rotations @ rotations.transpose(0, 2, 1)-np.eye(3)
    epsilon = float(np.max(np.linalg.norm(defect, ord=2, axis=(1, 2))))
    margin = 64*np.finfo(np.float64).eps*max(
        1., float(np.max(np.linalg.norm(rotations, ord=2, axis=(1, 2))))**2)
    lower, upper = np.sqrt(1.-epsilon)-margin, np.sqrt(1.+epsilon)+margin
    if np.any(norm < lower) or np.any(norm > upper):
        raise ValueError('Canonical receiving directions exceed their rotation metric roundoff bound')
    return dict(policy='exact_stage_orbit_angular_rule',
        maximum_direction_norm_defect=float(np.max(abs(norm-1.))),
        rotation_metric_spectral_defect=epsilon, floating_point_margin=margin,
        direction_norm_interval=[float(lower), float(upper)],
        cartesian_rotations_sha256=_digest(rotations),
        exact_nodes_weights_labels_harmonics=True, angular_Gram_error=float(expected[4]))


def _make_point_contraction(mesh, *, functional, radius, directions,
                            angular_weights, lm, Y, band_tile=8,
                            point_active=None, operator_contract=None,
                            angular_source=None):
    """Apply fixed-source Hartree adjoints to distributed receiving endpoints.

    ``contract(ps, delta, exact)`` consumes SAME-frame grid-normalized
    ``(1,band,4,point)`` faces at ``P(None,'x',None,'y')`` and physical exact
    Y00 integrals at ``P(None,'x','y',None)``. Natural points have direction
    fastest, then radius, then atom, with an optional inert suffix. The six
    components retain the atomic owner's order and return at
    ``P(None,None,'x','y')`` (component,source,bra,ket).

    The radial/angular projection adjoint is applied once to the fixed
    source. A scan gathers only one bounded ket band tile over X and reduces
    band actions over Y. It never forms pair densities or pair harmonics,
    gathers complete endpoint faces, or conjugates a receiving response.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw import isdf_augmentation as stage
    from isdf.atomic_hartree import charge_hartree_operator_contract

    r, dirs, aw, labels, harmonics = map(np.asarray,
        (radius, directions, angular_weights, lm, Y))
    na, nh, nr = map(int, functional['local_geometry'])
    tile = int(band_tile)
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    expected = ('compensation_body', 'difference', 'PS_delta', 'delta_PS',
                'enriched', 'periodic_mean')
    scale = float(functional['local_to_grid'])
    contract = (charge_hartree_operator_contract(None) if operator_contract is None
                else operator_contract)
    if (not isinstance(contract, dict)
            or set(contract) not in ({'operator', 'neutral_mean_policy'},
                                    {'operator', 'neutral_mean_policy', 'kernel'})
            or int(functional.get('sys_dim', 3)) != int(contract.get('kernel', {}).get('sys_dim', 3))
            or any(functional.get(key) != value for key, value in contract.items())
            or functional.get('receiving_component_order') != expected
            or not np.isfinite(scale) or scale <= 0 or na < 1 or nr < 1 or nh < 1
            or tile != band_tile or isinstance(band_tile, (bool, np.bool_))
            or tile < 1 or tile % px or r.shape != (nr,)
            or np.any(r < 0) or np.any(np.diff(r) <= 0)
            or dirs.ndim != 2 or dirs.shape[1:] != (3,) or not len(dirs)
            or aw.shape != (len(dirs),) or np.any(aw <= 0) or labels.shape != (nh, 2)
            or not np.issubdtype(labels.dtype, np.integer)
            or len(set(map(tuple, labels))) != len(labels)
            or np.any(labels[:, 0] < 0) or np.any(abs(labels[:, 1]) > labels[:, 0])
            or harmonics.shape != (nh, len(dirs))
            or not all(np.isfinite(a).all() for a in (r, dirs, aw, harmonics))):
        raise ValueError('Receiving adjoints require the canonical atomic geometry and tile extent')
    angular_contract = _receiving_angular_contract(dirs, aw, labels, harmonics, angular_source)
    logical = na*nr*len(dirs)
    active = (np.ones(logical, bool) if point_active is None
              else np.asarray(point_active).copy())
    if (active.ndim != 1 or active.dtype != np.dtype(bool)
            or len(active) % py or len(active) < logical
            or not np.all(active[:logical]) or np.any(active[logical:])):
        raise ValueError('Receiving points require natural physical order and inert suffix padding')
    response = functional['receiving_components']
    delta, ps, m0 = (np.asarray(response[k]) for k in ('delta', 'PS', 'exact_Y00'))
    nu = delta.shape[1]
    if (delta.shape != (6, nu, na, nh, nr) or ps.shape != delta.shape
            or m0.shape != (6, nu, na)
            or not all(np.isfinite(a).all() for a in (delta, ps, m0))
            or np.any(ps[[0, 1, 2, 4, 5]] != 0)):
        raise ValueError('Receiving component responses differ from the atomic Hartree owner')
    if 'kernel' in contract and (np.any(ps != 0) or np.any(delta[5] != 0) or np.any(m0[5] != 0)):
        raise ValueError('Receiving slab adjoints must omit local delta_PS and bulk periodic means')
    # Density harmonics are sum_d rho(r,d) Y_h*(d) w_d. Transpose that
    # map, without complex conjugating its response. Only grid-normalized
    # point products need Nfft/Omega; exact Y00 remains physical.
    def point_response(radial):
        value = scale*np.einsum(
            'uahr,hd,d->uard', radial, harmonics.conj(), aw, optimize=True)
        return np.pad(value.reshape(len(radial), logical),
                      ((0, 0), (0, len(active)-logical)))
    wd = point_response(delta.reshape(6*nu, na, nh, nr))
    wp = point_response(ps[3])
    weights_delta = stage._put(wd, mesh, P(None, 'y'))
    weights_ps = stage._put(wp, mesh, P(None, 'y'))
    weights_m0 = stage._put(m0, mesh, P())
    point_live = stage._put(active, mesh, P('y'))
    fs = P(None, 'x', None, 'y')

    kernel = _receiving_point_kernel(mesh, tile, nu)

    def contract(ps, delta, exact):
        if (ps.shape != delta.shape or ps.ndim != 4 or ps.shape[0] != 1
                or ps.shape[2:] != (4, len(active)) or ps.shape[1] % tile
                or ps.shape[1] % py or exact.shape != (1, ps.shape[1], ps.shape[1], na)
                or ps.sharding != NamedSharding(mesh, fs)
                or delta.sharding != NamedSharding(mesh, fs)
                or exact.sharding != NamedSharding(mesh, P(None, 'x', 'y', None))
                or any(np.dtype(a.dtype) != np.dtype(np.complex128)
                       for a in (ps, delta, exact))):
            raise ValueError('Receiving endpoints or exact Y00 differ from their distributed domain')
        return kernel(ps, delta, exact, weights_delta, weights_ps, weights_m0, point_live)

    receipt = dict(schema='lorrax.sharded_receiving_point_hartree.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        endpoint_layout="P(None,'x',None,'y')", parent_extent=1,
        point_order='atom_radius_direction', logical_points=logical,
        carrier_points=len(active), atoms=na, radii=nr, directions=len(dirs),
        harmonics=nh, band_tile=tile, source_count=nu,
        component_order=list(expected), x_gather_bands=tile,
        y_reduction='psum_of_weighted_band_actions', pair_density_materialized=False,
        pair_harmonics_materialized=False, complete_parent_endpoint_replication=False,
        point_response_bytes_per_rank=(6*nu+nu)*(len(active)//py)*16,
        bounded_ket_gather_bytes_per_rank=2*tile*4*(len(active)//py)*16,
        exact_response_units='physical_Y00', point_pair_conversion=functional['local_to_grid'],
        radius_sha256=_digest(r), directions_sha256=_digest(dirs),
        angular_weights_sha256=_digest(aw), lm_sha256=_digest(labels),
        Y_sha256=_digest(harmonics), point_active_sha256=_digest(active),
        angular_source_contract=angular_contract)
    return contract, receipt


def _make_delta_face(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax import shard_map
    def build(C,field,phase):
        return jnp.einsum('pnf,fsm->pnsm',C,field)*phase[:,None,None]
    return jax.jit(shard_map(build,mesh=mesh,
        in_specs=(P(None,'x',None),P(None,None,'y'),P(None,'y')),
        out_specs=P(None,'x',None,'y'),check_vma=False))


def _receiving_packet_functional(functional, start, stop):
    """Slice radial receiving adjoints, carrying the exact Y00 term once.

    The source's Poisson/local operators have already acted on the full
    radial grid. Only their receiving adjoints are sliced, so summing point
    contractions over packets is the original full-grid bilinear form.
    Exact Y00 is independent of the radial samples and belongs to the first
    packet alone. No quadrature weights or atomic fields are modified.
    """
    na, nh, nr = map(int, functional['local_geometry'])
    if (any(isinstance(v, (bool, np.bool_)) or int(v) != v for v in (start, stop))
            or not 0 <= start < stop <= nr):
        raise ValueError('Receiving radial packet must lie in the physical grid')
    start, stop = int(start), int(stop)
    response = functional['receiving_components']
    return dict(functional, local_geometry=(na, nh, stop-start),
        receiving_components=dict(delta=np.asarray(response['delta'])[..., start:stop],
            PS=np.asarray(response['PS'])[..., start:stop],
            exact_Y00=(response['exact_Y00'] if start == 0
                       else np.zeros_like(response['exact_Y00']))))


def _build_receiving_parts(*, wfn, mesh, state, artifact,
                               receiving_range=(0,64), band_tile=8,
                               parent_rows=None, source_capture=None):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.mtxel_sweep import (SweepGeometry, Operator,
        local_potential_operator, sweep_matrix_elements,blocks_to_host)
    from common.wfn_layout import band_sphere_spec
    from common.collectives import all_gather_processes
    from runtime.padding import padded_axis
    import math
    from gw import isdf_augmentation as stage
    from psp.reconstruction_overlap import rotate_band_rows
    from isdf.atomic_moments import exact_pair_moments
    from isdf.atomic_hartree import (prepare_charge_hartree, charge_hartree_functional,
                                    charge_hartree_operator_contract)

    started=time.perf_counter()
    memory_before=[d.memory_stats() for d in jax.local_devices()]
    parent=state['parent_psi'];overlap=state['overlap_receipt']
    source=state['hartree_source'] if source_capture is None else source_capture
    dimension = state.get('sys_dim', 3)
    operator_contract = charge_hartree_operator_contract(wfn, sys_dim=dimension)
    if (source['source_binding'].get('sys_dim', 3) != dimension
            or ('kernel' in operator_contract
                and source['source_binding'].get('hartree_kernel') != operator_contract['kernel'])
            or ('kernel' not in operator_contract and 'hartree_kernel' in source['source_binding'])):
        raise ValueError('Occupied source and receiving Coulomb kernel differ')
    if hashlib.sha256(json.dumps(source['source_binding'],sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()!=source['source_identity']:
        raise ValueError('Occupied source identity/binding changed')
    if (len(receiving_range)!=2 or any(isinstance(v,(bool,np.bool_)) or int(v)!=v
            for v in receiving_range)):
        raise ValueError('Receiving labels must be an integer band interval')
    lo,hi=map(int,receiving_range);requested_tile=int(band_tile)
    if requested_tile<1 or isinstance(band_tile,(bool,np.bool_)) or requested_tile!=band_tile:
        raise ValueError('Receiving band tile must be a positive integer')
    tile=padded_axis(requested_tile,mesh,name='receiving endpoint tile',
        spec=P(None,'x',None,'y'),axis=1).carrier
    smooth=parent.psi_G;nk,nb,ns,ng=map(int,smooth.shape)
    public=tuple(map(int,parent.band_range));physical=int(wfn.nbands)
    # Preparation clips the authenticated physical source window before
    # padding its reciprocal transport carrier. Keep those domains separate:
    # a physical120 source may legitimately use a public128 carrier.
    source_range=tuple(source['source_binding']['public_band_range'])
    if (len(source_range)!=2 or any(isinstance(v,(bool,np.bool_)) or int(v)!=v
            for v in source_range)):
        raise ValueError('Occupied source physical labels must be an integer band interval')
    source_range=tuple(map(int,source_range))
    rawk=np.asarray(parent.kvecs_frac);sym=state['sym']
    grid=tuple(map(int,wfn.fft_grid));N=int(np.prod(grid));volume=float(wfn.cell_volume)
    if (not public[0]==source_range[0]<=lo<hi<=source_range[1]<=min(public[1],physical)
            or public[1]-public[0]>nb or tile<1 or tile%int(mesh.shape['x'])
            or ns!=4 or source['source_binding']['augmentation_identity']!=artifact['identity']
            or state['identity']!=artifact['identity'] or state['fft_points']!=N
            or state['cell_volume']!=volume
            or not np.array_equal(rawk,np.asarray(wfn.kvecs(k=sym.parent_k_domain)))
            or not np.array_equal(rawk,np.asarray(overlap['k_parent_frac']))):
        raise ValueError('Resident source/receiving physical domain, geometry or state identity changed')
    if source['source_binding']['physical_bands']!=physical:
        raise ValueError('Occupied source and receiving full-WFN windows differ')
    compact = artifact.get('compact_target')
    if compact is not None:
        from psp.augmentation_cache import paired_field_policy_contract
        contract = paired_field_policy_contract(compact['binding']['field_policy'])
    if compact is not None and (state.get('compact_target_binding') != compact['binding']
            or overlap.get('compact_target_binding') != compact['binding']
            or source['source_binding'].get('compact_target_binding') != compact['binding']
            or overlap.get('overlap_operator') != contract['overlap_operator']
            or source['source_binding'].get('source_frame_policy') != contract['source_frame_policy']):
        raise ValueError('Receiving J must use the SAME compact target C/D/B/A and paired field policy')
    rows=np.arange(nk) if parent_rows is None else np.asarray(parent_rows,int)
    if rows.ndim!=1 or len(set(rows.tolist()))!=len(rows) or np.any((rows<0)|(rows>=nk)):
        raise ValueError('Receiving raw parent rows must be distinct and in domain')
    logical=hi-lo
    band_axis=padded_axis(logical,mesh,name='receiving Hartree band domain',
        specs=((P(None,'x','y'),1),(P(None,'x','y'),2)))
    endpoint_axis=padded_axis(logical,math.lcm(tile,band_axis.divisor),
        name='bounded receiving endpoint carrier')
    nt=endpoint_axis.carrier;indices=np.arange(lo-public[0],hi-public[0])
    matrix_carrier=band_axis.carrier
    radius=np.asarray(state['radius']);weights=np.asarray(state['weights_dr'])
    centers=np.asarray(wfn.atom_crys)%1.;types=np.asarray(wfn.atom_types,int)
    lattice=float(wfn.alat)*np.asarray(wfn.avec)
    directions,aw,lm,Y,angular_error=stage._orbit_angular_quadrature(
        artifact['angular'],np.asarray(sym.R_cart))
    expected_radius, expected_weights, expected_support = stage._radial_grid(artifact['radial'])
    if (not np.array_equal(lm,state['lm']) or not np.array_equal(
            centers@lattice,state['centers_cart'])
            or not np.array_equal(radius,expected_radius)
            or not np.array_equal(weights,expected_weights)
            or state['support_radius'] != expected_support):
        raise ValueError('Resident atomic geometry differs from prepared source')
    captured=dict(source)
    operand=prepare_charge_hartree(wfn,np.asarray(captured['smooth_density'])[None],
        np.asarray(captured['local_ps_density']),np.asarray(captured['local_delta_density']),
        np.asarray(captured['exact_monopole']),radius=radius,weights_dr=weights,lm=lm,
        centers_cart=state['centers_cart'],support_radius=state['support_radius'],
        minimum_atom_image_distance=state['nearest_atom_image'],
        electron_count=captured['electron_count'],
        interpolation_degree=state['interpolation_degree'],quadrature_order=state['quadrature_order'],
        sys_dim=dimension,fourier_points=state.get('fourier_points',4097))
    source_done=time.perf_counter()

    # Small tables receive the SAME factor before public selection. The
    # already-rotated reciprocal carrier is consumed without a second A.
    A=np.asarray(overlap['inverse_sqrt'])[:,:physical,:physical]
    source_A=np.asarray(captured.get('full150_A',A))
    if (source_A.shape!=A.shape or not np.isfinite(source_A).all()
            or hashlib.sha256(np.ascontiguousarray(source_A).tobytes()).hexdigest()
                !=captured['source_binding']['full150_frame_sha256']):
        raise ValueError('Occupied source physical factor payload/binding changed')
    source_receiving_factor_error=float(np.max(abs(source_A-A)))
    if source_receiving_factor_error>2e-11:
        raise ValueError('Occupied source and receiving factor differ beyond declared cross-mesh roundoff')
    C=[];D=[];B=[np.asarray(b) for b in overlap['atomic_delta_grams']]
    for rawC,rawD in zip(overlap['atomic_coefficients'],overlap['delta_overlaps']):
        C.append(rotate_band_rows(jnp.asarray(np.asarray(rawC)[:,:physical]),
            jnp.asarray(A),band_axis=1)[:,lo:hi])
        D.append(rotate_band_rows(jnp.asarray(np.asarray(rawD)[:,:physical]),
            jnp.asarray(A),band_axis=1)[:,lo:hi])
    C=[jnp.pad(c,((0,0),(0,nt-logical),(0,0))) for c in C]
    D=[jnp.pad(d,((0,0),(0,nt-logical),(0,0))) for d in D]
    SG=np.asarray(overlap['source_gram'])[:,:physical,:physical]
    expected_Q=np.pad(np.einsum('kmi,kmn,knj->kij',A.conj(),SG,A,optimize=True)[:,lo:hi,lo:hi],((0,0),(0,nt-logical),(0,nt-logical)))
    # Reuse the canonical bounded FFT/band-reduction sweep. Its identity G
    # operator measures Q_PS; no receiving-I substitution is made.
    gvec=np.asarray(wfn.gvecs(k=sym.parent_k_domain),dtype=np.int32)
    counts=np.asarray(wfn.ngk_valid(k=sym.parent_k_domain),int)
    if gvec.shape[1]>ng:raise ValueError('Resident G carrier shorter than loader geometry')
    if gvec.shape[1]<ng:gvec=np.pad(gvec,((0,0),(0,ng-gvec.shape[1]),(0,0)))
    gmask=(np.arange(ng)[None]<counts[:,None]).astype(float)
    K=(gvec+rawk[:,None])@np.asarray(state['reciprocal']);K*=gmask[...,None]
    geom=SweepGeometry(mesh=mesh,fft_grid=grid,ngkmax=ng,nb=public[1]-public[0],
        ns=ns,nk=nk,cell_volume=volume)
    identity=Operator(apply_g=lambda psi,g,m,k:psi,post=1.,key=('resident_J_identity',))
    bandG=jax.jit(lambda g:g,out_shardings=NamedSharding(mesh,band_sphere_spec()))(smooth)
    psbody,Q=sweep_matrix_elements(bandG,geom=geom,
        operator=(local_potential_operator(geom,operand['potential'][0]),identity),
        gvecs=gvec,gmask=gmask,box_index=parent.sphere_index,kvecs=rawk)
    crop=jax.jit(lambda h:jnp.pad(h[:,indices][:,:,indices],((0,0),(0,nt-logical),(0,nt-logical))),out_shardings=NamedSharding(mesh,P(None,'x','y')))
    PS=crop(psbody);Q=crop(Q)
    Q_error=float(np.asarray(jax.jit(lambda q:jnp.max(abs(q-jnp.asarray(expected_Q))),
        out_shardings=NamedSharding(mesh,P()))(Q)))
    if Q_error>2e-11:raise ValueError('Measured resident smooth overlap differs from SAME-A source Gram')
    smooth_done=time.perf_counter()

    # Apply the SAME source's receiving adjoint before forming band pairs.
    # Smooth PSbody/Q above retain the original potential, so its source
    # neutral mean is rejoined once below, not via smooth_potential here.
    functional=charge_hartree_functional(operand)
    packet=artifact['runtime']['radial_packet']
    if isinstance(packet,(bool,np.bool_)) or int(packet)!=packet or packet<1:
        raise ValueError('Receiving radial packet must be a positive integer')
    packet=int(packet)
    gs=P(None,None,None,('x','y'))
    kernels=stage._tile_kernels(mesh,1,nt)
    phase_kernel=stage._point_phase_kernel(mesh)
    delta_face=_make_delta_face(mesh)
    # Only the receiving sample view may require extra suffix zeros. The
    # original public reciprocal source/Poisson/sweep remain unchanged.
    extra=max(0,int(indices[0])+nt-int(smooth.shape[1]))
    endpoint_source=(smooth if extra==0 else jax.jit(lambda g:jnp.pad(g,
        ((0,0),(0,extra),(0,0),(0,0))),out_shardings=NamedSharding(mesh,gs))(smooth))
    band_live=stage._put(np.arange(nt)<logical,mesh,P('x'))
    mask_faces=jax.jit(lambda f,band_mask:jnp.where(band_mask[None,:,None,None],f,0.),
        out_shardings=NamedSharding(mesh,P(None,'x',None,'y')))
    shape=(len(rows),nt,nt);ms=NamedSharding(mesh,P(None,'x','y'))
    parts=jnp.zeros((6,*shape),jnp.complex128,device=NamedSharding(mesh,P(None,None,'x','y')))
    coefficients=[stage._put(np.asarray(c),mesh,P(None,'x',None)) for c in C]
    take_coefficient=jax.jit(lambda a,p:jax.lax.dynamic_slice(a,
        (p,jnp.int32(0),jnp.int32(0)),(1,a.shape[1],a.shape[2])),out_shardings=NamedSharding(mesh,P(None,'x',None)))
    take_phase=jax.jit(lambda a,p:jax.lax.dynamic_slice(a,
        (p,jnp.int32(0)),(1,a.shape[1])),out_shardings=NamedSharding(mesh,P(None,'y')))
    def exact_matrix(c,d):
        return jnp.stack([exact_pair_moments(ci,di,ci,di,jnp.asarray(b),array_api=jnp)
            for ci,di,b in zip(c,d,B)],axis=-1)[None]/np.sqrt(4*np.pi)
    exact_matrix=jax.jit(exact_matrix,
        out_shardings=NamedSharding(mesh,P(None,'x','y',None)))
    exact_by_parent=[exact_matrix(tuple(c[p] for c in C),tuple(d[p] for d in D))
                     for p in rows]
    # Charge is an exact physical C/D/B moment, independent of the radial
    # packet partition. Compute and store it once for every receiving row.
    delta_charge=jax.jit(lambda values:np.sqrt(4*np.pi)*jnp.stack(
        [jnp.sum(value[0],axis=-1) for value in values]),out_shardings=ms)(tuple(exact_by_parent))
    def store_parent(parts,value,p):
        zero=jnp.int32(0)
        previous=jax.lax.dynamic_slice(parts,(zero,p,zero,zero),(6,1,nt,nt))
        return jax.lax.dynamic_update_slice(parts,previous+value[:,0,None],(zero,p,zero,zero))
    store_parent=jax.jit(store_parent,out_shardings=NamedSharding(mesh,P(None,None,'x','y')))
    fs=NamedSharding(mesh,P(None,'x',None,'y'));peak_endpoint=0
    projection_receipts=[];peak_phase=peak_fields=0
    samplers={}
    for r0 in range(0,len(radius),packet):
        r1=min(r0+packet,len(radius));packet_radius=radius[r0:r1]
        points=(centers[:,None,None]+packet_radius[None,:,None,None]*
            directions[None,None]@np.linalg.inv(lattice)).reshape(-1,3)
        active=np.ones(len(points),bool)
        if len(points)%int(mesh.shape['y']):
            raise ValueError('Unwrapped atomic endpoint points need Y-compatible extent')
        if len(points) not in samplers:
            samplers[len(points)]=stage._point_samples_kernel(mesh,1,nt,len(points),
                int(artifact['factory']['g_block']) if 'factory' in artifact and 'g_block' in artifact['factory'] else 256,N)
        sampler=samplers[len(points)]
        cart=stage._put(points@lattice,mesh,P());live=stage._put(active.astype(float),mesh,P())
        contract,projection_receipt=_make_point_contraction(mesh,
            functional=_receiving_packet_functional(functional,r0,r1),
            radius=packet_radius,directions=directions,angular_weights=aw,lm=lm,Y=Y,
            band_tile=tile,point_active=active,operator_contract=operator_contract,
            angular_source=dict(control=artifact['angular'],
                cartesian_rotations=np.asarray(sym.R_cart)))
        projection_receipts.append(dict(projection_receipt,radial_interval=[r0,r1],
            exact_Y00_included=r0==0))
        # The existing atomic image owner supplies unchanged point fields
        # and Bloch phases, bounded to this packet and reused across parents.
        delta_geometry=[(stage._put(field,mesh,P(None,None,'y')),
            stage._put(phase,mesh,P(None,'y'))) for field,phase in stage._sample_geometry(
            points,active,centers,lattice,artifact['normalized_caches'],types,
            rawk,state['support_radius'],np.sqrt(volume/N))]
        peak_fields=max(peak_fields,sum(int(field.size+phase.size)*16//int(mesh.shape['y'])
            for field,phase in delta_geometry))
        for outrow,parent_index in enumerate(rows):
            print(f'Resident distributed J parent {int(parent_index)} radial [{r0},{r1}): native shards',flush=True)
            k=stage._put(K[parent_index:parent_index+1],mesh,P(None,('x','y'),None))
            phase=phase_kernel(k,cart)
            peak_phase=max(peak_phase,int(phase.size)*16//int(mesh.size))
            ps=sampler(kernels['source'](endpoint_source,jnp.int32(parent_index),jnp.int32(indices[0])),phase,live)
            ps=mask_faces(ps,band_live)
            delta=jnp.zeros((1,nt,4,len(points)),jnp.complex128,device=fs)
            for atom,(field,image_phase) in enumerate(delta_geometry):
                coeff=take_coefficient(coefficients[atom],jnp.int32(parent_index))
                delta=delta+delta_face(coeff,field,take_phase(image_phase,jnp.int32(parent_index)))
            jax.block_until_ready((ps,delta));del phase,k
            peak_endpoint=max(peak_endpoint,int(ps.size+delta.size)*16//int(mesh.size))
            parts=store_parent(parts,contract(ps,delta,exact_by_parent[outrow]),jnp.int32(outrow))
            parts.block_until_ready();del ps,delta
        del delta_geometry,cart,live,contract
    source_mean=(-2/volume*operand['source_phi'][0].conj()*Q[rows]
        if dimension == 3 else jnp.zeros_like(Q[rows]))
    def assemble(parts,delta_charge,PS,Q,source_mean):
        body=PS+parts[0];local=parts[1:5];mean=parts[5]+source_mean
        return body+jnp.sum(local,axis=0)+mean,body,local,mean,Q+delta_charge
    assemble=jax.jit(assemble,out_shardings=(ms,ms,
        NamedSharding(mesh,P(None,None,'x','y')),ms,ms))
    total,body,local,mean,charge=assemble(parts,delta_charge,PS[rows],Q[rows],source_mean)
    # Only scalar diagnostics reduce; production does not gather native J.
    scalar=jax.jit(lambda a:jnp.max(abs(a)),out_shardings=NamedSharding(mesh,P()))
    finite=jax.jit(lambda a:jnp.all(jnp.isfinite(a)),out_shardings=NamedSharding(mesh,P()))
    closure=float(np.asarray(scalar(total-body-local.sum(axis=0)-mean)))
    if closure>2e-11 or any(not bool(np.asarray(finite(a))) for a in (total,body,local,mean,charge)):
        raise ValueError('Resident Hartree tile closure or finite matrix failed')
    arrays=dict(hartree_ry=total,body_ry=body,local_terms_ry=local,
        periodic_mean_ry=mean,receiving_charge=charge,PSbody_ry=PS[rows],Q_PS=Q[rows],
        raw_smooth_truth_ry=PS[rows]+source_mean,
        raw_local_truth_ry=total-PS[rows]-source_mean)
    memory_after=[d.memory_stats() for d in jax.local_devices()]
    local_peaks=np.asarray([max((int(m.get('peak_bytes_in_use',0)) for m in memory_after if m),default=0),
        max((int(m.get('bytes_in_use',0)) for m in memory_before if m),default=0)],dtype=np.int64)
    process_peaks=np.asarray(all_gather_processes(local_peaks))
    diagnostics=dict(source_prepare_seconds=source_done-started,
        smooth_sweep_seconds=smooth_done-source_done,
        receiving_local_seconds=time.perf_counter()-smooth_done,
        total_seconds=time.perf_counter()-started,
        distributed_endpoint_bytes_per_rank=peak_endpoint,sharded_projection_receipts=projection_receipts,
        receiving_radial_packet=packet,receiving_radial_packets=len(projection_receipts),
        phase_packet_bytes_per_rank=peak_phase,atomic_field_packet_bytes_per_rank=peak_fields,
        exact_Y00_response_included_once=True,delta_charge_computed_once=True,
        pair_tile_shape=[tile,tile,len(types),len(lm),min(packet,len(radius))],
        device_memory_before=memory_before,device_memory_after=memory_after,
        all_process_peak_bytes_in_use=process_peaks[...,0].tolist(),
        all_process_resident_bytes_before=process_peaks[...,1].tolist(),
        receiving_pair_density_materialized=False,receiving_pair_harmonics_materialized=False,
        source_identity=captured['source_identity'],source_binding=captured['source_binding'],
        raw_parent_rows=rows.tolist(),parent_full_rows=np.asarray(sym.kirr_fullids)[rows].tolist(),
        raw_parent_k=rawk[rows].tolist(),receiving_band_labels=(np.arange(lo,hi)+1).tolist(),
        receiving_band_range=[lo,hi],receiving_band_logical=logical,
        receiving_band_carrier=matrix_carrier,receiving_endpoint_carrier=nt,
        receiving_transport_tile=tile,receiving_requested_tile=requested_tile,
        measured_Q_PS_error=Q_error,decomposition_error_Ry=closure,
        measured_receiving_isometry_error=float(np.asarray(scalar(charge-np.pad(np.eye(logical),((0,nt-logical),(0,nt-logical)))))),
        hermitian_skew_Ry=float(np.asarray(scalar(total-total.swapaxes(-1,-2).conj()))),
        angular_gram_error=angular_error,operator=operand['operator'],
        neutral_mean_policy=operand['neutral_mean_policy'],
        physical_bands=physical,public_band_range=list(source_range),
        reciprocal_transport_band_range=list(public),
        source_receiving_factor_error=source_receiving_factor_error,
        no_WFN_coefficient_read=True,no_projection_or_overlap_rebuild=True,
        no_mu_or_F_operand=True,source_point_independence_scope='callable consumes physical state only',
        native_matrix_sharding="P(None,'x','y')",complete_endpoint_replication=False)
    landing=jax.jit(lambda a:a[...,:matrix_carrier,:matrix_carrier],
        out_shardings=NamedSharding(mesh,P(None,'x','y')))
    arrays={k:(jax.jit(lambda a:a[...,:matrix_carrier,:matrix_carrier],
        out_shardings=NamedSharding(mesh,P(None,None,'x','y')))(v)
        if k=='local_terms_ry' else landing(v)) for k,v in arrays.items()}
    diagnostics['numerical_owner_sources_sha256']=numerical_owner_sources_sha256()
    return arrays,diagnostics


def numerical_owner_sources_sha256():
    from gw import isdf_augmentation
    from isdf import atomic_hartree,atomic_moments
    from common import mtxel_sweep
    from psp import reconstruction_overlap,augmentation_spinors
    from runtime import padding
    owners=(isdf_augmentation,atomic_hartree,atomic_moments,mtxel_sweep,
            reconstruction_overlap,augmentation_spinors,padding)
    return {**{m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in owners},
        __name__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def build_resident_receiving_J(*,wfn,mesh,state,artifact,receiving_range=(0,64),band_tile=8):
    """Return native two-band sharded J and source/operator diagnostics.

    Receiving labels are logical; the returned array has its canonical
    mesh-compatible carrier and exact zero band ghosts. Source occupations,
    full-WFN factor and public fitting interval remain unchanged.
    """
    arrays,diagnostics=_build_receiving_parts(wfn=wfn,mesh=mesh,state=state,artifact=artifact,
        receiving_range=receiving_range,band_tile=band_tile)
    return arrays['hartree_ry'],diagnostics
