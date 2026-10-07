"""Periodic charge Hartree from fitting's reconstructed occupied density.

The source is a physical occupation trace, independent of the ISDF loss.
Its compensated full-FFT Poisson field and bounded receiving-vertex tiles
include both smooth/local adjoints and the periodic neutral-potential mean.
This numerical seam does not select a source frame or enable a GW consumer.
"""
from __future__ import annotations

import numpy as np


def make_occupied_point_trace(plan, occupations, full_kweights, *,
                              cell_volume, spin_degeneracy):
    """Physical occupied charge on a typed radial packet, never ISDF loss.

    Occupations select the SAME raw-parent bands as the input sample face.
    Full-zone k weights are normalized independently of star multiplicity.
    The canonical plan transports each operation class's scalar density;
    spin actions and Bloch phases cancel in the LL+SS occupation trace.
    Returned values are charge per volume at P('y'), replicated over X.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map

    occ=np.asarray(occupations);weights=np.asarray(full_kweights)
    volume=float(cell_volume);fspin=float(spin_degeneracy)
    if (occ.ndim!=2 or occ.shape[0]!=plan.n_parent or np.iscomplexobj(occ)
            or not np.isfinite(occ).all() or np.any(occ<0) or np.any(occ>1)
            or weights.shape!=(plan.n_full,) or np.iscomplexobj(weights)
            or not np.isfinite(weights).all() or np.any(weights<0)
            or not np.isclose(weights.sum(),1.,rtol=0,atol=2e-12)
            or not np.isfinite(volume) or volume<=0 or fspin not in (1.,2.)
            or plan.nspinor!=1 and fspin!=1.):
        raise ValueError('Hartree point trace requires canonical physical occupations and normalized full-k weights')
    spin=np.asarray(plan.spin_action_full)
    if (spin.ndim!=3 or spin.shape[1:]!=(plan.nspinor,plan.nspinor)
            or not np.allclose(spin.conj().transpose(0,2,1)@spin,
                               np.eye(plan.nspinor),rtol=0,atol=2e-12)):
        raise ValueError('Hartree occupation trace requires the typed unitary spin action')
    classes=plan.operation_classes()
    active=np.asarray(plan.layout.axis.active_mask,dtype=bool)
    global_perm=np.asarray(plan.sym_perm)[classes.ops]
    if (np.any(global_perm<0) or np.any(global_perm>=len(active))
            or not np.all(active[global_perm]==active[None])):
        raise ValueError('Hartree scalar transport crosses the physical point/ghost boundary')
    counts=np.zeros_like(classes.counts)
    np.add.at(counts,(classes.class_of_row,np.asarray(plan.irr_idx)),weights)
    occ=np.where(counts.sum(axis=0)[:,None]>0,occ,0.)
    mesh=plan.mesh_xy
    put=lambda a,spec:device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    O=put(occ,P(None,'x'));W=put(counts,P());perm=put(classes.local_perm,P())
    mask=put(active,P('y'));scale=float(np.prod(plan.fft_grid))*fspin/volume

    def trace_local(face,o,w,permutation,live):
        # Rejecting poisoned inactive carriers after squaring is too late:
        # finite dead values can overflow, and zero times NaN is not zero.
        used=(o>0)[:,:,None,None]&live[None,None,None,:]
        face=jnp.where(used,face,0.)
        partial=jnp.einsum('pnsr,pn->pr',jnp.abs(face)**2,o)
        parents=jax.lax.psum(partial,'x')
        by_class=jnp.einsum('cp,pr->cr',w,parents)
        result=plan.transport_classes(by_class,permutation,class_axis=0,
            mu_axis=1,mesh_axis='y')
        return jnp.where(live,scale*result,0.)
    kernel=jax.jit(shard_map(trace_local,mesh=mesh,
        in_specs=(P(None,'x',None,'y'),P(None,'x'),P(),P(),P('y')),
        out_specs=P('y'),check_vma=False))

    def trace(face):
        shape=(plan.n_parent,occ.shape[1],plan.nspinor,plan.n_centroid_packed)
        if (face.shape!=shape or face.sharding!=NamedSharding(mesh,P(None,'x',None,'y'))
                or np.dtype(face.dtype)!=np.dtype(np.complex128)):
            raise ValueError('Hartree occupation trace face differs from its typed raw-parent/point/physical-band layout')
        return kernel(face,O,W,perm,mask)
    return trace


def make_occupied_density_projection(mesh, indices, weights, *, output_shape):
    """Project small physical source fields with the caller's canonical buckets.

    Parameters
    ----------
    indices : (P_y,N_row,N_bucket) integer
        Existing point-owner local bucket indices, never a second geometry
        or symmetry map. Rows are in the caller's canonical order.
    weights : (P_y,N_channel,N_row,N_bucket) complex or real
        Existing angular/integration weights; zero buckets are inert.
    output_shape : tuple of int
        A reshape of ``(N_row,N_channel)`` in row-major order. For atomic
        harmonics this is ``(N_atom,N_radius,N_lm)``; the caller transposes
        its radius/harmonic axes to the radial-provider order.

    Returns
    -------
    callable
        Maps physical ``(source,N_point_packed)`` at ``P(None,'y')`` to a
        small replicated ``(source,*output_shape)`` table. Bands have already
        been summed over X by the occupation trace; only Y is reduced here.
        No band-pair cloud or per-state Fourier transform is constructed.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map

    columns, table = np.asarray(indices), np.asarray(weights)
    shape = tuple(output_shape)
    py = int(mesh.shape['y'])
    if (columns.ndim != 3 or columns.shape[0] != py
            or not np.issubdtype(columns.dtype, np.integer) or np.any(columns < 0)
            or table.ndim != 4 or table.shape[0] != py
            or table.shape[2:] != columns.shape[1:]
            or min(table.shape) < 1 or not np.isfinite(table).all()
            or not shape or any(isinstance(s, (bool, np.bool_))
                or not isinstance(s, (int, np.integer)) or s < 1 for s in shape)
            or int(np.prod(shape)) != table.shape[1]*columns.shape[1]):
        raise ValueError('Hartree density projection requires finite canonical bucket weights and output ordering')
    put = lambda a, spec: device_put_process_local(a, NamedSharding(mesh, spec))
    index = put(columns.astype(np.int32), P('y', None, None))
    angular = put(table, P('y', None, None, None))

    def project_local(density, cols, angular_weights):
        values = jnp.take(density, cols[0], axis=-1)
        live = jnp.any(angular_weights[0] != 0, axis=0)
        values = jnp.where(live[None], values, 0.)
        partial = jnp.einsum('sfj,hfj->sfh', values, angular_weights[0])
        return jax.lax.psum(partial, 'y').reshape((density.shape[0],) + shape)
    kernel = jax.jit(shard_map(project_local, mesh=mesh,
        in_specs=(P(None, 'y'), P('y', None, None), P('y', None, None, None)),
        out_specs=P(), check_vma=False))

    def project(density):
        if (density.ndim != 2 or density.shape[1] % py
                or int(columns.max()) >= density.shape[1]//py
                or density.sharding != NamedSharding(mesh, P(None, 'y'))):
            raise ValueError('Hartree density projection differs from the typed local point layout')
        return kernel(density, index, angular)
    return project


def prepare_charge_hartree(wfn, smooth_density, local_ps_density,
                          local_delta_density, local_monopole, *, radius,
                          weights_dr, lm, centers_cart, support_radius,
                          minimum_atom_image_distance, electron_count,
                          interpolation_degree=5, quadrature_order=16):
    """Prepare full periodic 3D/G0-zero potentials for occupation traces.

    ``smooth_density`` is physical charge per volume, ``(source,nx,ny,nz)``.
    Local PS/delta densities are physical spherical coefficients with shape
    ``(source,atom,lm,radius)``; ``local_monopole`` is the exact correction
    integral against Y00, ``(source,atom)``. Every operand must come from the
    same served field and orbital frame. The fitting caller authenticates
    that binding and its physical occupations before invoking this seam.

    The small source fields and FFT potential are replicated by design,
    as in the canonical charge-Hartree owner. No band-pair FFT is created.
    """
    import jax
    import jax.numpy as jnp
    from scipy.special import beta, sph_harm_y
    from common.fourier_plan import LocalFourierPlan
    from common.wfn_transforms import process_local_mesh
    from psp.dft_operators import (poisson_potential_from_rhoG,
                                   _poisson_reciprocal_geometry)
    from isdf.atomic_coulomb import (atomic_radial_metrics,
                                    _angular_channels, _smooth_neutral_tables)
    from isdf.augmentation_breit import _sonine_profile_transform

    grid=tuple(map(int,wfn.fft_grid));N=int(np.prod(grid))
    volume=float(wfn.cell_volume);R=float(support_radius)
    reciprocal=float(wfn.blat)*np.asarray(wfn.bvec,dtype=float)
    metric=np.asarray(wfn.bdot,dtype=float)
    if (reciprocal.shape!=(3,3) or metric.shape!=(3,3)
            or not np.isfinite(reciprocal).all() or not np.isfinite(metric).all()
            or not np.allclose(metric,reciprocal@reciprocal.T,rtol=2e-11,atol=2e-12)):
        raise ValueError('Hartree canonical reciprocal metric and Cartesian Fourier vectors differ')
    rho=np.asarray(smooth_density);sp=np.asarray(local_ps_density)
    sd=np.asarray(local_delta_density);exact=np.asarray(local_monopole)
    centers=np.asarray(centers_cart,dtype=float);r=np.asarray(radius,dtype=float)
    channels=_angular_channels(lm);nh=len(channels)
    if (rho.ndim!=4 or rho.shape[1:]!=grid or sd.ndim!=4
            or sp.shape!=sd.shape or sd.shape[0]!=rho.shape[0]
            or sd.shape[2:]!=(nh,len(r)) or not rho.shape[0]
            or centers.shape!=(sd.shape[1],3) or not sd.shape[1]
            or exact.shape!=sd.shape[:2] or not np.isfinite(volume) or volume<=0
            or not np.isfinite(minimum_atom_image_distance)
            or minimum_atom_image_distance<=2*R
            or any(not np.isfinite(v).all() for v in (rho,sp,sd,exact,centers))
            or np.max(abs(rho.imag))>1e-12):
        raise ValueError('Hartree occupation trace has invalid physical density/local geometry')
    lm_rows={tuple(row):i for i,row in enumerate(channels)}
    for h,(l,m) in enumerate(channels):
        partner=lm_rows[(int(l),-int(m))]
        if any(not np.allclose(v[:,:,partner],(-1.)**int(m)*v[:,:,h].conj(),
                               rtol=2e-11,atol=2e-12) for v in (sp,sd)):
            raise ValueError('Hartree source harmonics must represent real occupied charge')
    if not np.allclose(exact,exact.real,rtol=2e-11,atol=2e-12):
        raise ValueError('Hartree source monopoles must represent real occupied charge')
    expected=np.asarray(electron_count)
    if np.iscomplexobj(expected):
        raise ValueError('Hartree physical electron counts must be real')
    expected=expected.astype(float)
    if expected.ndim==0:expected=np.full(rho.shape[0],float(expected))
    if expected.shape!=(rho.shape[0],) or not np.isfinite(expected).all() or np.any(expected<0):
        raise ValueError('Hartree requires the physical electron count for every source')
    tables=atomic_radial_metrics(r,weights_dr,channels[:,0],support_radius=R,
        fft_points=1,cell_volume=1,interpolation_degree=interpolation_degree,
        quadrature_order=quadrature_order)
    cross=_smooth_neutral_tables(tables,support_radius=R,fft_points=1,cell_volume=1)
    rows=np.searchsorted(tables['degrees'],channels[:,0])
    mono=int(np.flatnonzero(np.all(channels==(0,0),axis=1))[0]);row0=int(rows[mono])
    moments=np.asarray(tables['moments'][rows])
    source_moments=np.einsum('bahr,hr->bah',sd,moments)
    epsilon=exact-source_moments[:,:,mono]
    source_comp=source_moments.copy();source_comp[:,:,mono]=exact
    nodal=np.asarray(tables['interpolation_map']).copy()
    nodal[:tables['origin_row_count'],0]=tables['origin_factors'][row0]
    q2=(tables['quadrature_weights_dr']*tables['quadrature_radius']**4)@nodal
    source_phi=-(2*np.pi/3)*np.sqrt(4*np.pi)*np.sum(
        np.einsum('bar,r->ba',sd[:,:,mono],q2)
        -3*R**2/17*source_moments[:,:,mono],axis=-1)
    mesh=process_local_mesh()
    forward=LocalFourierPlan(grid,(-3,-2,-1),sign=-1,norm='backward',mesh=mesh)
    source_F=forward(jnp.asarray(rho.real*(volume/N),jnp.complex128)).reshape(len(rho),N)
    G2,zero,Gcart,_=_poisson_reciprocal_geometry(grid,jnp.asarray(wfn.bdot),
        jnp.asarray(wfn.bvec),float(wfn.blat),False,need_g_cart=True)
    vectors=np.asarray(Gcart).reshape(3,N).T;length=np.linalg.norm(vectors,axis=1)
    v=jnp.where(zero,0.,8*np.pi/(volume*G2)).reshape(N)
    degrees=np.asarray(tables['degrees']);scale=2/beta(degrees+1.5,7.)
    comp_source=jnp.zeros((len(rho),N),jnp.complex128)
    response=jnp.zeros((len(rho),len(centers),nh),jnp.complex128)
    tile=2048

    @jax.jit
    def response_tile(sf,sm,angle,vf):
        full=sf+jnp.einsum('bah,ahg->bg',sm,angle)
        return full,jnp.einsum('bg,g,ahg->bah',full.conj(),vf,angle)

    for first in range(0,N,tile):
        last=min(first+tile,N);n=last-first
        vec,ka=vectors[first:last],length[first:last]
        theta=np.arccos(np.clip(np.divide(vec[:,2],ka,out=np.ones_like(ka),where=ka>0),-1,1))
        phi=np.arctan2(vec[:,1],vec[:,0])
        radial=np.stack([_sonine_profile_transform(int(l),6,R,ka)*s
                         for l,s in zip(degrees,scale)])[rows]
        angular=np.stack([4*np.pi*(-1j)**int(l)*sph_harm_y(int(l),int(m),theta,phi)
                          for l,m in channels])*radial
        angle=np.exp(-1j*(vec@centers.T)).T[:,None]*angular[None]
        full,part=response_tile(jnp.pad(source_F[:,first:last],((0,0),(0,tile-n))),
            jnp.asarray(source_comp),jnp.asarray(np.pad(angle,((0,0),(0,0),(0,tile-n)))),
            jnp.pad(v[first:last],((0,tile-n),)))
        comp_source=comp_source.at[:,first:last].set(full[:,:n]);response=response+part
    charges=np.asarray(comp_source[:,0])
    if np.max(abs(charges-expected))>2e-10:
        raise ValueError('Hartree reconstructed occupation trace differs from its physical electron count')
    potential=poisson_potential_from_rhoG(
        (comp_source*np.sqrt(N)/volume).reshape((len(rho),)+grid),
        jnp.asarray(wfn.bdot),jnp.asarray(wfn.bvec),float(wfn.blat),False)
    return dict(potential=potential,compensation_response=response,
        source_delta=jnp.asarray(sd),source_ps=jnp.asarray(sp),
        source_epsilon=jnp.asarray(epsilon),source_phi=jnp.asarray(source_phi),
        source_charge=jnp.asarray(charges.real),moments=jnp.asarray(moments),q2_row=jnp.asarray(q2),
        metric_difference=jnp.asarray((tables['delta_metric']-tables['compensation_metric'])[rows]),
        smooth_neutral=jnp.asarray(cross['smooth_neutral_metric'][rows]),
        m0_cross=jnp.asarray(cross['smooth_compensation_cross'][row0]
            -2*tables['compensation_self'][row0]*tables['moments'][row0]),
        mono=mono,support_radius=R,volume=volume,fft_grid=grid,
        operator='ordinary_3D_periodic_full_FFT_G0_zero',
        neutral_mean_policy='subtract_free_space_neutral_cell_mean')


def make_charge_hartree_tile(operand):
    """Compile bounded receiving vertices in the SAME reconstructed frame.

    Row/column smooth FFT boxes have ``(vertex,band,spin,nx,ny,nz)`` shape
    and orthonormal FFT scaling. Local PS/delta pair densities have shape
    ``(vertex,row_band,column_band,atom,lm,radius)``. Exact local monopoles
    have ``(vertex,row_band,column_band,atom)`` shape. Returns the matrix,
    PW body, four local terms, periodic mean and target overlap. Matrices
    retain both source and receiving-vertex axes for independent controls.
    """
    import jax
    import jax.numpy as jnp

    if (operand.get('operator')!='ordinary_3D_periodic_full_FFT_G0_zero'
            or operand.get('neutral_mean_policy')!='subtract_free_space_neutral_cell_mean'):
        raise ValueError('Hartree requires its own full-FFT/G0-zero and periodic-mean operator')
    mono=int(operand['mono']);volume=float(operand['volume']);R=float(operand['support_radius'])
    potential=operand['potential'];response=operand['compensation_response']
    sd,sp=operand['source_delta'],operand['source_ps']
    se,q2,mom=operand['source_epsilon'],operand['q2_row'],operand['moments']
    kd,kn,ke=(operand[k] for k in ('metric_difference','smooth_neutral','m0_cross'))
    sf,sq=operand['source_phi'],operand['source_charge']

    @jax.jit
    def contract(row_boxes,col_boxes,tp,td,exact):
        if (row_boxes.ndim!=6 or col_boxes.ndim!=6 or tp.shape!=td.shape
                or td.ndim!=6 or exact.shape!=td.shape[:4]
                or row_boxes.shape[0]!=col_boxes.shape[0]
                or row_boxes.shape[:2]!=td.shape[:2]
                or col_boxes.shape[:2]!=(td.shape[0],td.shape[2])
                or row_boxes.shape[2:]!=col_boxes.shape[2:]
                or row_boxes.shape[3:]!=operand['fft_grid']
                or td.shape[3:]!=sd.shape[1:]):
            raise ValueError('Hartree receiving vertices differ from the source geometry or matrix tile')
        tm=jnp.einsum('bijahr,hr->bijah',td,mom)
        epsilon=exact-tm[:,:,:,:,mono]
        comp=tm.at[:,:,:,:,mono].set(exact)
        psbody=jnp.einsum('visxyz,uxyz,vjsxyz->uvij',row_boxes.conj(),potential,col_boxes)
        compbody=jnp.einsum('uah,bijah->ubij',response,comp)
        body=psbody+compbody
        diff=jnp.einsum('uahr,hrt,vijaht->uvij',sd.conj(),kd,td)
        c1=jnp.einsum('uahr,hrt,vijaht->uvij',sp.conj(),kn,td)
        c2=jnp.einsum('uahr,htr,vijaht->uvij',sd.conj(),kn,tp)
        sc=jnp.einsum('uar,r->ua',sd[:,:,mono],ke)
        tc=jnp.einsum('vijar,r->vija',td[:,:,:,:,mono],ke)
        enriched=jnp.einsum('ua,vija->uvij',se.conj(),tc)+jnp.einsum('ua,vija->uvij',sc.conj(),epsilon)
        phi=-(2*np.pi/3)*np.sqrt(4*np.pi)*jnp.sum(
            jnp.einsum('bijar,r->bija',td[:,:,:,:,mono],q2)
            -3*R**2/17*tm[:,:,:,:,mono],axis=-1)
        charge=jnp.einsum('visxyz,vjsxyz->vij',row_boxes.conj(),col_boxes)
        charge=charge+np.sqrt(4*np.pi)*jnp.sum(exact,axis=-1)
        mean=-2/volume*(sq.conj()[:,None,None,None]*phi[None]
                         +sf.conj()[:,None,None,None]*charge[None])
        local=jnp.stack((diff,c1,c2,enriched))
        return body+jnp.sum(local,axis=0)+mean,body,local,mean,charge
    return contract


def charge_hartree_functional(operand):
    """Linear Hartree functional on the existing charge normal-equation RHS.

    This is the receiving-vertex linearization of
    :func:`make_charge_hartree_tile`, including both periodic neutral means.
    It computes ``F_mu = integral(zeta_mu * V_H)`` without conjugating the
    receiving density. The same charge factor therefore solves this scalar
    RHS; it never needs full-FFT zeta functions or band-pair radial clouds.

    Returns
    -------
    dict
        ``smooth_potential`` is a real physical Rydberg potential with shape
        ``(source,nx,ny,nz)``. Its contraction with grid-normalized smooth
        density has no 1/N factor. ``local_response`` has shape
        ``(source,2*N_atom*N_lm*N_r+N_atom)`` in the provider's existing
        delta, PS, exact-Y00 order. It includes the single N_fft/Omega
        conversion for local densities in that provider's grid units.
        Source identity, q0 placement and point layout are bound by fitting.
    """
    import jax.numpy as jnp

    if (operand.get('operator') != 'ordinary_3D_periodic_full_FFT_G0_zero'
            or operand.get('neutral_mean_policy') != 'subtract_free_space_neutral_cell_mean'):
        raise ValueError('Hartree functional requires its full-FFT/G0-zero and periodic-mean operator')
    mono = int(operand['mono'])
    volume = float(operand['volume'])
    radius = float(operand['support_radius'])
    sd, sp = operand['source_delta'], operand['source_ps']
    se = operand['source_epsilon']
    moment = operand['moments']
    kd, kn, ke = (operand[k] for k in
                  ('metric_difference', 'smooth_neutral', 'm0_cross'))
    response = operand['compensation_response']
    sf, sq = operand['source_phi'], operand['source_charge']
    source_mean = np.asarray(sf)
    if (not np.isfinite(source_mean).all()
            or not np.allclose(source_mean, source_mean.real, rtol=2e-11, atol=2e-12)):
        raise ValueError('Hartree functional requires the real occupied-source neutral mean')

    delta = (jnp.einsum('uahr,hrt->uaht', sd.conj(), kd)
             + jnp.einsum('uahr,hrt->uaht', sp.conj(), kn))
    ps = jnp.einsum('uahr,htr->uaht', sd.conj(), kn)
    # Compensation's Y00 coefficient is the exact served M0, not the
    # radial-interpolant moment. Keep its receiving adjoint separate.
    multipole_response = response.at[:, :, mono].set(0.)
    delta = delta + multipole_response[..., None] * moment[None, None]
    source_cross = jnp.einsum('uar,r->ua', sd[:, :, mono], ke)
    delta = delta.at[:, :, mono].add(
        se.conj()[..., None]*ke - source_cross.conj()[..., None]*moment[mono])
    m0 = response[:, :, mono] + source_cross.conj()

    phi_row = ((2*np.pi/3)*np.sqrt(4*np.pi)
               * (operand['q2_row'] - 3*radius**2/17*moment[mono]))
    delta = delta.at[:, :, mono].add(
        (2/volume)*sq.conj()[:, None, None]*phi_row)
    m0 = m0 - (2/volume)*np.sqrt(4*np.pi)*sf.conj()[:, None]
    # The source is real occupied charge; its Y00 neutral mean is real.
    # Apply that scalar to the smooth overlap here, and to the local exact
    # overlap in m0 above. There is no Hartree energy's factor one-half.
    smooth = operand['potential'] - (2/volume)*sf.real[:, None, None, None]
    local_to_grid = float(np.prod(operand['fft_grid']))/volume
    local = jnp.concatenate((delta.reshape(len(sd), -1),
                             ps.reshape(len(sd), -1), m0), axis=-1)
    return dict(smooth_potential=smooth, local_response=local_to_grid*local,
                local_geometry=tuple(map(int, sd.shape[1:])),
                local_to_grid=local_to_grid,
                local_feature_order=('delta', 'PS', 'exact_Y00'),
                operator=operand['operator'],
                neutral_mean_policy=operand['neutral_mean_policy'])


def make_charge_hartree_rhs_contractor(functional, mesh, *, q0_slot):
    """Reduce paired local RHSs to a q-owned scalar before the SAME OWN solve.

    Input and output use ``P(('x','y'),None,None)``. The input is the existing
    provider's concatenated ``(Q_pad,mu_packed,local_feature)`` complex128
    RHS, in grid-normalized density units. Only its authenticated physical
    Gamma slot contributes; all other q rows return exact zero. The source
    response is a small replicated species/radial table, never a band cloud.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map

    if (functional.get('operator') != 'ordinary_3D_periodic_full_FFT_G0_zero'
            or functional.get('local_feature_order') != ('delta', 'PS', 'exact_Y00')
            or functional.get('neutral_mean_policy') != 'subtract_free_space_neutral_cell_mean'
            or isinstance(q0_slot, (bool, np.bool_))
            or not isinstance(q0_slot, (int, np.integer)) or q0_slot < 0):
        raise ValueError('Hartree local RHS requires its exact operator, feature order and physical Gamma slot')
    layout = NamedSharding(mesh, P(('x', 'y'), None, None))
    response = device_put_process_local(
        np.asarray(functional['local_response']), NamedSharding(mesh, P()))

    def reduce_local(rhs, table):
        qrow = (jax.lax.axis_index(('x', 'y'))*rhs.shape[0]
                + jnp.arange(rhs.shape[0]))
        gamma = jnp.where((qrow == q0_slot)[:, None, None], rhs, 0.)
        return jnp.einsum('qmf,uf->qmu', gamma, table)
    kernel = jax.jit(shard_map(reduce_local, mesh=mesh,
        in_specs=(P(('x', 'y'), None, None), P()),
        out_specs=P(('x', 'y'), None, None), check_vma=False))

    def contract(rhs):
        if (rhs.ndim != 3 or rhs.shape[-1] != response.shape[-1]
                or q0_slot >= rhs.shape[0] or rhs.sharding != layout
                or np.dtype(rhs.dtype) != np.dtype(np.complex128)):
            raise ValueError('Hartree local RHS differs from its paired provider features or q-owned layout')
        return kernel(rhs, response)
    return contract
