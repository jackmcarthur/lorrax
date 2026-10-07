"""Exact source-bound receiving charge Hartree from an existing prepared state.

The orbital frame/atomic caches are inputs already authenticated by preparation.
No WFN coefficient read, projector, Gram or inverse-root factory occurs here.
Endpoint faces remain distributed; only bounded pair harmonics replicate.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def _digest(value):
    array = np.ascontiguousarray(value)
    header = repr((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _make_tile_project(mesh, *, radius, directions, angular_weights, lm, Y,
                      atom_count, fft_points, cell_volume, band_tile=8,
                      point_active=None):
    """Return ``(tile_project, receipt)`` for grid-normalized four-spinors.

    ``tile_project(ps, delta, rowstart, colstart)`` accepts SAME-frame
    ``(1, band, 4, point)`` faces at ``P(None,'x',None,'y')``. Points have
    direction fastest, then radius, then atom; only a suffix may be padding.
    It returns physical PS and correction charge harmonics with shape
    ``(1,tile,tile,atom,lm,radius)``. The correction includes BOTH mixed
    adjoints and delta-delta. N_fft/Omega is applied exactly once here.

    Only two bounded eight-band endpoint pairs gather over X, at fixed Y.
    Bucket gathers retain only radial rows owned by that Y shard. The
    resulting small harmonic tile reduces over Y, without an X reduction.
    The caller's full receiving endpoint storage stays distributed.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from gw import isdf_augmentation as stage

    r = np.asarray(radius, dtype=np.float64).copy()
    dirs = np.asarray(directions, dtype=np.float64).copy()
    aw = np.asarray(angular_weights, dtype=np.float64).copy()
    labels = np.asarray(lm).copy()
    harmonics = np.asarray(Y, dtype=np.complex128).copy()
    na, tile = int(atom_count), int(band_tile)
    nfft, volume = int(fft_points), float(cell_volume)
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    if (isinstance(atom_count, (bool, np.bool_)) or atom_count != na or na < 1
            or isinstance(band_tile, (bool, np.bool_)) or band_tile != tile
            or tile < 1 or tile % px or isinstance(fft_points, (bool, np.bool_))
            or fft_points != nfft or nfft < 1 or not np.isfinite(volume) or volume <= 0
            or r.ndim != 1 or not len(r) or not np.isfinite(r).all()
            or np.any(r < 0) or np.any(np.diff(r) <= 0)
            or dirs.ndim != 2 or dirs.shape[1:] != (3,) or not len(dirs)
            or not np.isfinite(dirs).all()
            or not np.allclose(np.linalg.norm(dirs, axis=1), 1., rtol=0, atol=2e-12)
            or aw.shape != (len(dirs),) or not np.isfinite(aw).all() or np.any(aw <= 0)
            or labels.ndim != 2 or labels.shape[1:] != (2,) or not len(labels)
            or not np.issubdtype(labels.dtype, np.integer)
            or len(set(map(tuple, labels))) != len(labels)
            or np.any(labels[:, 0] < 0) or np.any(abs(labels[:, 1]) > labels[:, 0])
            or harmonics.shape != (len(labels), len(dirs))
            or not np.isfinite(harmonics).all()):
        raise ValueError('Receiving harmonics require the finite canonical atomic geometry and tile extent')
    logical = na * len(r) * len(dirs)
    live = (np.ones(logical, bool) if point_active is None
            else np.asarray(point_active).copy())
    if (live.ndim != 1 or live.dtype != np.dtype(bool) or len(live) % py
            or len(live) < logical or not np.all(live[:logical]) or np.any(live[logical:])):
        raise ValueError('Receiving point axis must be the natural physical order with inert suffix padding')
    npoint = len(live)
    # These metadata describe the existing natural sample axis. They do not
    # create a centroid geometry or invent a symmetry transport plan.
    natural_axis = SimpleNamespace(packed_to_canonical=np.arange(npoint),
        active_mask=live, n_logical=logical)
    natural = SimpleNamespace(layout=SimpleNamespace(axis=natural_axis))
    columns, weights = stage._angular_bucket_tables(
        natural, na, len(r), harmonics.conj() * aw[None], py)
    rows = na * len(r)
    row_lists = [np.flatnonzero(np.any(weights[y] != 0, axis=(0, 2)))
                 for y in range(py)]
    local_rows = max(1, max(map(len, row_lists)))
    width = columns.shape[-1]
    local_columns = np.zeros((py, local_rows, width), np.int32)
    local_weights = np.zeros((py, len(labels), local_rows, width), np.complex128)
    row_ids = np.zeros((py, local_rows), np.int32)
    for y, selected in enumerate(row_lists):
        row_ids[y, :len(selected)] = selected
        local_columns[y, :len(selected)] = columns[y, selected]
        local_weights[y, :, :len(selected)] = weights[y][:, selected]
    ids = stage._put(row_ids, mesh, P('y', None))
    cols = stage._put(local_columns, mesh, P('y', None, None))
    angle = stage._put(local_weights, mesh, P('y', None, None, None))
    active = stage._put(live, mesh, P('y'))
    fs = P(None, 'x', None, 'y')
    face = stage._tile_kernels(mesh, 1, tile)['face']
    scale = nfft / volume

    def project_local(pr, pc, dr, dc, columns, weights, row_ids, live):
        # Mask before multiplication so arbitrarily poisoned dead samples
        # cannot create NaNs or overflow in a charge product.
        endpoints = [jnp.where(live[None, None, None], value, 0.)
                     for value in (pr, pc, dr, dc)]
        pr, pc, dr, dc = [jax.lax.all_gather(value, 'x', axis=1, tiled=True)[0]
                          for value in endpoints]
        ps = jnp.einsum('ism,jsm->ijm', pr.conj(), pc)
        delta = (jnp.einsum('ism,jsm->ijm', pr.conj(), dc)
                 + jnp.einsum('ism,jsm->ijm', dr.conj(), pc)
                 + jnp.einsum('ism,jsm->ijm', dr.conj(), dc))
        weighted_live = jnp.any(weights[0] != 0, axis=0)

        def angular_project(pair):
            values = jnp.take(pair, columns[0], axis=-1)
            values = jnp.where(weighted_live[None, None], values, 0.)
            partial = jnp.einsum('ijrb,hrb->ijrh', values, weights[0])
            out = jnp.zeros((tile, tile, rows, len(labels)), jnp.complex128)
            out = out.at[:, :, row_ids[0], :].add(partial)
            out = jax.lax.psum(out, 'y')
            return (scale * out.reshape(tile, tile, na, len(r), len(labels))
                    .transpose(0, 1, 2, 4, 3))[None]
        return angular_project(ps), angular_project(delta)

    kernel = jax.jit(shard_map(project_local, mesh=mesh,
        in_specs=(fs, fs, fs, fs, P('y', None, None),
                  P('y', None, None, None), P('y', None), P('y')),
        out_specs=(P(), P()), check_vma=False))

    def tile_project(ps, delta, rowstart, colstart):
        if (ps.shape != delta.shape or ps.ndim != 4 or ps.shape[0] != 1
                or ps.shape[2:] != (4, npoint) or ps.shape[1] % px
                or ps.sharding != NamedSharding(mesh, fs)
                or delta.sharding != NamedSharding(mesh, fs)
                or np.dtype(ps.dtype) != np.dtype(np.complex128)
                or np.dtype(delta.dtype) != np.dtype(np.complex128)
                or any(isinstance(v, (bool, np.bool_)) or int(v) != v
                       or not 0 <= int(v) <= ps.shape[1] - tile
                       for v in (rowstart, colstart))):
            raise ValueError('Receiving endpoints or selected bands differ from the distributed natural face')
        zero = jnp.int32(0)
        i, j = jnp.int32(rowstart), jnp.int32(colstart)
        return kernel(face(ps, zero, i), face(ps, zero, j),
                      face(delta, zero, i), face(delta, zero, j),
                      cols, angle, ids, active)

    receipt = dict(schema='lorrax.sharded_receiving_harmonics.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage_owner_sha256=hashlib.sha256(Path(stage.__file__).read_bytes()).hexdigest(),
        endpoint_layout="P(None,'x',None,'y')", parent_extent=1,
        point_order='atom_radius_direction', point_axis='natural_with_suffix_padding',
        logical_points=logical, carrier_points=npoint, atoms=na, radii=len(r),
        directions=len(dirs), harmonics=len(labels), band_tile=tile,
        x_gather_bands=tile, y_reduction='psum_of_canonical_angular_buckets',
        local_bucket_rows=local_rows, bucket_width=width,
        radius_sha256=_digest(r), directions_sha256=_digest(dirs),
        angular_weights_sha256=_digest(aw), lm_sha256=_digest(labels), Y_sha256=_digest(harmonics),
        point_active_sha256=_digest(live), fft_points=nfft, cell_volume=volume,
        output_units='physical_density_harmonics', grid_pair_conversion=scale,
        correction_terms=['PS_bra_delta_ket', 'delta_bra_PS_ket', 'delta_bra_delta_ket'],
        complete_parent_endpoint_replication=False,
        bounded_endpoint_gather_bytes_per_rank=4*tile*4*(npoint//py)*16,
        bucket_pair_bytes_per_rank=2*tile*tile*local_rows*width*16,
        replicated_output_tile_bytes=2*tile*tile*na*len(r)*len(labels)*16)
    return tile_project, receipt


def _make_matrix_tiles(mesh,band_tile=8):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    tile=int(band_tile)
    def read(m,p,i,j):
        ri=i+jnp.arange(tile)-jax.lax.axis_index('x')*m.shape[-2]
        cj=j+jnp.arange(tile)-jax.lax.axis_index('y')*m.shape[-1]
        valid=(ri[:,None]>=0)&(ri[:,None]<m.shape[-2])&(cj[None]>=0)&(cj[None]<m.shape[-1])
        value=m[p,jnp.clip(ri,0,m.shape[-2]-1)[:,None],jnp.clip(cj,0,m.shape[-1]-1)[None]]
        return jax.lax.psum(jnp.where(valid,value,0.),('x','y'))
    def store(m,value,p,i,j):
        ri=i+jnp.arange(tile)-jax.lax.axis_index('x')*m.shape[-2]
        cj=j+jnp.arange(tile)-jax.lax.axis_index('y')*m.shape[-1]
        ri=jnp.where((ri>=0)&(ri<m.shape[-2]),ri,m.shape[-2])
        cj=jnp.where((cj>=0)&(cj<m.shape[-1]),cj,m.shape[-1])
        return m.at[...,p,ri[:,None],cj[None]].set(value,mode='drop')
    read=jax.jit(shard_map(read,mesh=mesh,in_specs=(P(None,'x','y'),P(),P(),P()),out_specs=P(),check_vma=False))
    write=jax.jit(shard_map(store,mesh=mesh,in_specs=(P(None,'x','y'),P(),P(),P(),P()),
        out_specs=P(None,'x','y'),check_vma=False))
    write4=jax.jit(shard_map(store,mesh=mesh,in_specs=(P(None,None,'x','y'),P(),P(),P(),P()),
        out_specs=P(None,None,'x','y'),check_vma=False))
    return read,write,write4


def _make_delta_face(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    def build(C,field,phase):
        return jnp.einsum('pnf,fsm->pnsm',C,field)*phase[:,None,None]
    return jax.jit(shard_map(build,mesh=mesh,
        in_specs=(P(None,'x',None),P(None,None,'y'),P(None,'y')),
        out_specs=P(None,'x',None,'y'),check_vma=False))


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
    from isdf.atomic_hartree import prepare_charge_hartree,make_charge_hartree_tile

    started=time.perf_counter()
    memory_before=[d.memory_stats() for d in jax.local_devices()]
    parent=state['parent_psi'];overlap=state['overlap_receipt']
    source=state['hartree_source'] if source_capture is None else source_capture
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
    if (not np.array_equal(lm,state['lm']) or not np.array_equal(
            centers@lattice,state['centers_cart'])
            or not np.array_equal(radius,np.asarray(artifact['radial']['radius']))
            or not np.array_equal(weights,np.asarray(artifact['radial']['weights_dr']))):
        raise ValueError('Resident atomic geometry differs from prepared source')
    captured=dict(source)
    operand=prepare_charge_hartree(wfn,np.asarray(captured['smooth_density'])[None],
        np.asarray(captured['local_ps_density']),np.asarray(captured['local_delta_density']),
        np.asarray(captured['exact_monopole']),radius=radius,weights_dr=weights,lm=lm,
        centers_cart=state['centers_cart'],support_radius=state['support_radius'],
        minimum_atom_image_distance=state['nearest_atom_image'],
        electron_count=captured['electron_count'],
        interpolation_degree=state['interpolation_degree'],quadrature_order=state['quadrature_order'])
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

    # Zero boxes are literals inside this jit: existing tile algebra supplies
    # all atomic terms, and the canonical smooth sweep is rejoined below.
    # XLA can eliminate the identically zero grid contractions, retaining no
    # receiving FFT boxes or alternate local Coulomb implementation.
    contract=make_charge_hartree_tile(operand)
    @jax.jit
    def local_only(tp,td,exact):
        z=jnp.zeros((1,tile,4,*grid),jnp.complex128)
        return contract(z,z,tp,td,exact)
    geometry=(len(types),len(radius),len(directions))
    points=(centers[:,None,None]+radius[None,:,None,None]*
        directions[None,None]@np.linalg.inv(lattice)).reshape(-1,3)
    active=np.ones(len(points),bool)
    if len(points)%int(mesh.shape['y']):raise ValueError('Unwrapped atomic endpoint points need Y-compatible extent')
    gs=P(None,None,None,('x','y'))
    kernels=stage._tile_kernels(mesh,1,nt)
    sampler=stage._point_samples_kernel(mesh,1,nt,len(points),
        int(artifact['factory']['g_block']) if 'factory' in artifact and 'g_block' in artifact['factory'] else 256,N)
    phase_kernel=stage._point_phase_kernel(mesh)
    cart=stage._put(points@lattice,mesh,P());live=stage._put(active.astype(float),mesh,P())
    tile_project,projection_receipt=_make_tile_project(mesh,radius=radius,directions=directions,
        angular_weights=aw,lm=lm,Y=Y,atom_count=len(types),fft_points=N,
        cell_volume=volume,band_tile=tile,point_active=active)
    read,store,store4=_make_matrix_tiles(mesh,tile);delta_face=_make_delta_face(mesh)
    # Only the receiving sample view may require extra suffix zeros. The
    # original public reciprocal source/Poisson/sweep remain unchanged.
    extra=max(0,int(indices[0])+nt-int(smooth.shape[1]))
    endpoint_source=(smooth if extra==0 else jax.jit(lambda g:jnp.pad(g,
        ((0,0),(0,extra),(0,0),(0,0))),out_shardings=NamedSharding(mesh,gs))(smooth))
    band_live=stage._put(np.arange(nt)<logical,mesh,P('x'))
    mask_faces=jax.jit(lambda f,band_mask:jnp.where(band_mask[None,:,None,None],f,0.),
        out_shardings=NamedSharding(mesh,P(None,'x',None,'y')))
    shape=(len(rows),nt,nt);ms=NamedSharding(mesh,P(None,'x','y'))
    total=jnp.zeros(shape,jnp.complex128,device=ms)
    body=jnp.zeros(shape,jnp.complex128,device=ms);mean=jnp.zeros(shape,jnp.complex128,device=ms)
    local=jnp.zeros((4,*shape),jnp.complex128,device=NamedSharding(mesh,P(None,None,'x','y')))
    charge=jnp.zeros(shape,jnp.complex128,device=ms)
    fs=NamedSharding(mesh,P(None,'x',None,'y'));peak_endpoint=0
    for outrow,parent_index in enumerate(rows):
        print(f'Resident distributed J parent {int(parent_index)}: native band/point shards',flush=True)
        k=stage._put(K[parent_index:parent_index+1],mesh,P(None,('x','y'),None))
        phase=phase_kernel(k,cart)
        ps=sampler(kernels['source'](endpoint_source,jnp.int32(parent_index),jnp.int32(indices[0])),phase,live)
        ps=mask_faces(ps,band_live)
        delta=jnp.zeros((1,nt,4,len(points)),jnp.complex128,device=fs)
        for atom,(field,image_phase) in enumerate(stage._sample_geometry(points,active,
                centers,lattice,artifact['normalized_caches'],types,
                rawk[parent_index:parent_index+1],state['support_radius'],np.sqrt(volume/N))):
            coeff=stage._put(np.asarray(C[atom][parent_index:parent_index+1]),mesh,P(None,'x',None))
            delta=delta+delta_face(coeff,stage._put(field,mesh,P(None,None,'y')),
                stage._put(image_phase,mesh,P(None,'y')))
        jax.block_until_ready((ps,delta));del phase,k
        peak_endpoint=max(peak_endpoint,int(ps.size+delta.size)*16//int(mesh.size))
        for i in range(0,nt,tile):
            for j in range(0,nt,tile):
                ri,cj=slice(i,i+tile),slice(j,j+tile)
                tp,td=tile_project(ps,delta,i,j)
                exact=jnp.stack([exact_pair_moments(c[parent_index,ri],d[parent_index,ri],
                    c[parent_index,cj],d[parent_index,cj],jnp.asarray(b),array_api=jnp)
                    for c,d,b in zip(C,D,B)],axis=-1)[None]/np.sqrt(4*np.pi)
                m,p,l,z,q=local_only(tp,td,exact)
                ids=(jnp.int32(parent_index),jnp.int32(i),jnp.int32(j))
                pt,qt=read(PS,*ids),read(Q,*ids)
                sm=-2/volume*operand['source_phi'][0].conj()*qt
                outids=(jnp.int32(outrow),jnp.int32(i),jnp.int32(j))
                total=store(total,m[0,0]+pt+sm,*outids)
                body=store(body,p[0,0]+pt,*outids)
                local=store4(local,l[:,0,0],*outids)
                mean=store(mean,z[0,0]+sm,*outids)
                charge=store(charge,q[0]+qt,*outids)
        jax.block_until_ready((total,body,local,mean,charge));del ps,delta
    # Only scalar diagnostics reduce; production does not gather native J.
    scalar=jax.jit(lambda a:jnp.max(abs(a)),out_shardings=NamedSharding(mesh,P()))
    finite=jax.jit(lambda a:jnp.all(jnp.isfinite(a)),out_shardings=NamedSharding(mesh,P()))
    closure=float(np.asarray(scalar(total-body-local.sum(axis=0)-mean)))
    if closure>2e-11 or any(not bool(np.asarray(finite(a))) for a in (total,body,local,mean,charge)):
        raise ValueError('Resident Hartree tile closure or finite matrix failed')
    source_mean=-2/volume*complex(np.asarray(operand['source_phi'])[0].conjugate())*Q[rows]
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
        distributed_endpoint_bytes_per_rank=peak_endpoint,sharded_projection_receipt=projection_receipt,pair_tile_shape=[tile,tile,len(types),len(lm),len(radius)],
        device_memory_before=memory_before,device_memory_after=memory_after,
        all_process_peak_bytes_in_use=process_peaks[...,0].tolist(),
        all_process_resident_bytes_before=process_peaks[...,1].tolist(),
        literal_zero_box_elimination_not_assumed=True,
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
