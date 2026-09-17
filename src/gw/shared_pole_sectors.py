"""Charge/current cross pencils on stacks of parent-local arrays.

The joint projection is plan section 12.2. Both endpoint output panels
must be retained: C[X_C,X_T] and T[X_C,X_T]. For the ordered response the
definite member is H, not G (shared-pole report equation 5.5).
These functions run inside the constructor's batched linalg stage; they
neither move an operator to the host nor prescribe a processor mesh.
"""

import jax.numpy as jnp


def read_sector_round(io, meta, bank, header, ids, endpoints, *, sample_span=None,
                      fields=('Wc','dWc_ds'), retained=()):
    """Read one bounded sector sample at a time into its packed endpoint bases.

    The canonical photon store is mesh-interleaved; each selected family is
    converted to the existing MuBasis order before the constructor sees it.
    ``endpoints`` names C=0 or T=1 on each side. Returned TT/CT rows are
    mu-major, Cartesian-component-minor. Parents remain at P(('x','y')).
    The store selects native sector hyperslabs, never full photon panels.
    It owns all I/O,
    authentication and transport; this function only selects sector rows.
    """
    import jax
    import numpy as np
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from file_io.shared_pole_store import read_shared_pole_bank

    layout=bank['photon_layout']
    indices=[];masks=[]
    for family in endpoints:
        basis=bank['mu_bases'][family]
        mu=basis.pack_host(np.arange(basis.n_canonical,dtype=np.int32),axis=0)
        components=3 if family else 1
        width=layout.carrier_extent(family)//layout.mesh_side
        local=components*width
        index=(mu[:,None]//width*local+np.arange(components)[None]*width+mu[:,None]%width)
        indices.append(index.reshape(-1))
        masks.append(np.repeat(basis.active_mask,components))
    spec=P(('x','y'))
    def select(a):
        a=jnp.take(jnp.take(a,jnp.asarray(indices[0]),axis=-2),jnp.asarray(indices[1]),axis=-1)
        return jnp.where(jnp.asarray(masks[0])[:,None]&jnp.asarray(masks[1])[None,:],a,0)
    select=jax.jit(shard_map(select,mesh=io.mesh,in_specs=spec,out_specs=spec,check_vma=False))
    join=jax.jit(shard_map(lambda *a:jnp.concatenate(a,axis=1),mesh=io.mesh,
                          in_specs=spec,out_specs=spec,check_vma=False))
    ledger=meta.shared_pole_capacity
    ambient=ledger.live_stages
    keep=list(retained)
    out={}
    try:
        for field in fields:
            sample=field in ('Wc','dWc_ds')
            if sample and (sample_span is None or sample_span[1] <= sample_span[0]):
                raise ValueError('sector sample reads require a nonempty bounded sample_span')
            spans=([(i,i+1) for i in range(*sample_span)] if sample else [None])
            parts=[]
            for span in spans:
                size=sum(a.size*a.dtype.itemsize//io.mesh.size for a in keep)
                row=ledger.reserve(f'sector.read.retained.{len(ledger.entries)}',
                    resident_bytes_per_rank=size,workspace_bytes_per_rank=0,concurrent_with=ambient)
                ledger.live_stages=(*ambient,row['stage'])
                value=read_shared_pole_bank(io,meta=meta,header=header,q_ids=ids,
                    sample_span=span,fields=(field,),partition_spec=spec,
                    sector=tuple('T' if family else 'C' for family in endpoints))[field]
                # Reserve the selected output alongside the bounded input.
                output=value.shape[0]*(1 if not sample else value.shape[1])*len(indices[0])*len(indices[1])*value.dtype.itemsize//io.mesh.size
                workspace=value.shape[0]*(1 if not sample else value.shape[1])*len(indices[0])*value.shape[-1]*value.dtype.itemsize//io.mesh.size
                ledger.reserve(f'sector.read.select.{len(ledger.entries)}',
                    resident_bytes_per_rank=value.size*value.dtype.itemsize//io.mesh.size+output,
                    workspace_bytes_per_rank=workspace,concurrent_with=ledger.live_stages)
                selected=select(value)
                selected.block_until_ready()
                del value
                parts.append(selected);keep.append(selected)
            if len(parts)>1:
                ledger.reserve(f'sector.read.join.{len(ledger.entries)}',
                    resident_bytes_per_rank=sum(a.size*a.dtype.itemsize//io.mesh.size for a in keep+parts),
                    workspace_bytes_per_rank=0,concurrent_with=ambient)
            out[field]=join(*parts) if len(parts)>1 else parts[0]
            out[field].block_until_ready()
            keep=list(retained)+list(out.values())
    finally:
        ledger.live_stages=ambient
    return out


def construct_diagonal_sector_round(samples, moments, meta, config, geometry, *, mesh_xy,
                                     retained=()):
    """Run the production selection/reduction program for CC or TT.

    Samples and M0..M3 are parent-local stacks, [P,S,n,n] and [P,n,n].
    ``geometry`` carries the existing recipe, endpoint basis/tables/header,
    round ids/real/partner slots and symmetry partner rows. TT uses
    n=3*n_T with mu-major Cartesian rows. The unchanged recipe fractions
    set its widths and 1.8*n pole budget. Capacity remains the whole-map
    ledger; no fictitious independent sector allowance is created.

    Returns the sorted positive model, signed model, original-pencil span,
    selected state/infinity panels and replicated diagnostics. The latter
    are needed by the CT joint projection in the same map, never cached.
    """
    import copy
    import math
    import jax
    import numpy as np
    import distrib_la
    from runtime.padding import padded_axis
    from jax.sharding import PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity
    from gw.shared_pole_directions import _round_kernels,select_round_states
    from gw.shared_pole_local import partner_realization,round_tables,reduce_round,_batch_put
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b

    components=int(geometry['components'])
    basis=geometry['basis']
    n=components*basis.n_logical
    local_meta=copy.copy(meta)
    local_meta.mu_basis=basis
    local_meta.n_rmu=n
    local_meta.n_rmu_padded=components*basis.n_packed
    recipe=dict(meta.shared_pole_recipe,n=n)
    policy=shared_real_pole_v1_r3b[recipe['accuracy']]
    for field in ('imaginary_width','infinity_width','line_direction_cap','pole_budget'):
        fraction=policy.get(field+'_fraction')
        recipe[field]=None if fraction is None else math.ceil(n*fraction)
    budget=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
                               mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,upstream=())
    budget.batch_width=len(geometry['ids'])
    budget.retained_panels=tuple(retained)
    budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1])
    eig=budget.eigenplan(local_meta.n_rmu_padded)
    svd=budget.eigenplan(2*local_meta.n_rmu_padded)
    extent=lambda width:padded_axis(width,mesh_xy,name='shared_pole_port',
        specs=((P('x','y'),0),(P('x','y'),1))).carrier
    qi,values=distrib_la.leading_eigenvectors(moments['M1'],min(n,recipe['infinity_width']),
        eigh_plan=eig,column_extent=extent,multiplet_tol=recipe['multiplet_relative_tolerance'],
        real_rows=geometry['real'])
    kernels=_round_kernels(mesh_xy)
    infinity=(qi,*(kernels.apply(moments[name],qi) for name in ('M0','M1','M2','M3')))
    action=partner_realization(local_meta,geometry['header'],geometry['ids'],
        geometry['partner_parent'],geometry['partner_row'],mesh_xy=mesh_xy,components=components)
    rotation=(None if components==1 else _batch_put(mesh_xy,geometry['sym'].cartesian_action(
        np.asarray(geometry['partner_row'])[geometry['ids']],axial=False,time_odd=True)))
    states,counts,roles=select_round_states(samples,recipe,sample_lo=geometry['sample_lo'],
        real=geometry['real'],mesh_xy=mesh_xy,eigh_plan=eig,svd_plan=svd,column_extent=extent,
        logical_n=n,ordered=True,exchange=(geometry['slots'],*action),current_rotation=rotation)
    tables=round_tables(counts,[s[1].shape[-1] for s in states],[s[0] for s in states],
        [v.shape[-1] for v in values],qi.shape[-1],column_extent=extent,ordered=True,odd_moments=True)
    side=tables['active'].shape[-1]
    budget.retained_panels=(*retained,*infinity,*(v for s in states for v in s[1:]),
                            *samples.values(),*moments.values())
    budget.plan(side,phase='reduction')
    reduced=reduce_round(states,infinity,tables,real=geometry['real'],mesh_xy=mesh_xy,
        native_eigh=budget.eigenplan(side).native_fn,ordered=True,odd_moments=True,
        keep_budget=recipe['pole_budget'],retain_span=True)
    model,signed,vectors,diagnostics,y=reduced
    reduction,zero,_,_=jax.tree.map(np.asarray,diagnostics)
    for name in ('orientation_paired','gram_diagonal_positive','gram_valid','retained_metric_positive'):
        if not np.all(reduction[name][:geometry['real']]):
            raise ValueError(f"GATE shared_pole_sector_{name}: sector={geometry['sector']}, "
                             f"parents={geometry['ids'][:geometry['real']]}, "
                             f"Gram min/max={reduction['gram_min_relative'][:geometry['real']].tolist()}; no repair")
    if not np.all(zero['zero_policy'][:geometry['real']]):
        raise ValueError(f"GATE shared_pole_sector_zero_ritz: sector={geometry['sector']}")
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                endpoint_action=(*action,rotation),recipe=recipe,budget=budget)


def cross_round_actions(samples, states, roles, recipe, *, sample_lo, mesh_xy,
                        partner_slots, endpoint_actions):
    """Apply rectangular samples to the diagonal sectors' selected directions.

    ``samples=(W_LR,W_RL,dW_LR/ds,dW_RL/ds)`` are parent-local
    [P,S,n_L,n_R] (reverse blocks have reversed endpoint extents).
    ``states``/``roles`` are the existing selection round's source states.
    ``endpoint_actions`` is ((alpha,inverse,phase,rotation)_L, ..._R);
    a charge endpoint has rotation=None, a current endpoint uses the
    symmetry service's polar time-odd [P,3,3] action and mu-major rows.
    Only direction/output panels cross ranks, through the same partner
    permutation as the diagonal-sector construction. Outputs follow the
    paired state order and the derivative is d/dz, report equation 5.3.
    """
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_directions import _sample_point

    batch=P(('x','y'))
    perm=tuple((i,int(p)) for i,p in enumerate(partner_slots))
    left,right=endpoint_actions

    def rotate(panel, rotation, transpose):
        if rotation is None:
            return panel
        shape=panel.shape
        panel=panel.reshape(shape[0],shape[1]//3,3,shape[-1])
        return jnp.einsum('bij,bmir->bmjr' if transpose else 'bij,bmjr->bmir',
                          rotation,panel).reshape(shape)

    outputs=[]
    for state,role in zip(states,roles[0]):
        sid=int(role['sample_id'])
        z=_sample_point(recipe,sid)
        conjugate=bool(role.get('conjugate',False))
        mirror=bool(role.get('mirror',False))
        exchange=mirror and z.real!=0
        use_adjoint=not conjugate if mirror and not exchange else conjugate
        node=state[0]

        def apply(w,wr,d,dr,q,left,right):
            a,b=w[:,sid-sample_lo],wr[:,sid-sample_lo]
            da,db=d[:,sid-sample_lo],dr[:,sid-sample_lo]
            if exchange:
                alpha,inverse,phase,rotation=right
                q=rotate(jnp.take_along_axis(phase[:,:,None]*q,inverse[:,:,None],axis=1),rotation,True)
                q=jax.lax.ppermute(q,('x','y'),perm)
                if conjugate:
                    a,da=jnp.conj(a),jnp.conj(da)
                else:
                    a,da=jnp.swapaxes(b,-1,-2),jnp.swapaxes(db,-1,-2)
            elif use_adjoint:
                a,da=jnp.swapaxes(jnp.conj(b),-1,-2),jnp.swapaxes(jnp.conj(db),-1,-2)
            o,d_o=a@q,(da@q)*(2*node)
            if exchange:
                alpha,inverse,phase,rotation=left
                def back(x):
                    x=jax.lax.ppermute(x,('x','y'),perm)
                    return rotate(jnp.conj(phase)[:,:,None]*jnp.take_along_axis(x,alpha[:,:,None],axis=1),rotation,False)
                o,d_o=back(o),back(d_o)
            return o,d_o

        program=jax.jit(shard_map(apply,mesh=mesh_xy,in_specs=(batch,)*7,
                                  out_specs=(batch,batch),check_vma=False))
        outputs.append(program(*samples,state[1],left,right))
    return tuple(outputs)


def reduce_cross_round(charge, transverse, cross, moments, *, mesh_xy, native_eigh, gates):
    """Construct CT on the two retained original-pencil spans, one parent per rank.

    Each sector tuple contains (points, order, states, infinity, Y, signed),
    where ``states`` is (Q tuple, WQ tuple), ``signed`` is (c,mu,active),
    and all operands carry the round's leading parent sharding. ``cross``
    contains the TC-on-C and CT-on-T (output, derivative) panel tuples;
    ``moments`` is M0_CT..M3_CT. No full operator leaves the admitted
    parent-local program. Returns two signed CT endpoint factors, inverse
    poles, active columns and the unchanged joint-metric diagnostics.
    """
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm

    def body(charge,transverse,cross,moments):
        def unpack(sector,actions):
            points,order,states,infinity,y,signed=sector
            def pack(panels):
                return jnp.take(jnp.concatenate(panels,axis=-1),order[0],axis=-1,mode='fill',fill_value=0)
            return points,pack(states[0]),infinity[0],pack(tuple(a[0] for a in actions)),pack(tuple(a[1] for a in actions))
        zc,qc,ic,tc,dtc=unpack(charge,cross[0])
        zt,qt,it,ct,dct=unpack(transverse,cross[1])
        g,h,otc,oct=ordered_cross_pencil((zc,qc,ic),(zt,qt,it),(tc,ct,dct),moments,matmul=_mm)
        # The diagonal output panels are O_original. Reconstruct their
        # infinity columns from the same physical moments already in hand.
        def diagonal(sector):
            _,order,states,infinity,y,signed=sector
            own=jnp.take(jnp.concatenate(states[1],axis=-1),order[0],axis=-1,mode='fill',fill_value=0)
            return jnp.concatenate((own,2*infinity[1],2*infinity[2]),axis=-1)
        cc=(charge[4],charge[5][1],diagonal(charge),otc)
        tt=(transverse[4],transverse[5][1],diagonal(transverse),oct)
        return reduce_sector_pencil(joint_sector_pencil(cc,tt,(h,g),matmul=_mm),
                                   eigh=native_eigh,matmul=_mm,gates=gates)
    spec=P(('x','y'))
    return jax.jit(shard_map(body,mesh=mesh_xy,in_specs=(spec,)*4,
                            out_specs=(spec,spec),check_vma=False))(charge,transverse,cross,moments)


def ordered_cross_pencil(charge, transverse, cross_actions, cross_moments, *, matmul):
    """Assemble rectangular G_CT, H_CT and both cross outputs, report A.1/5.4.

    ``charge`` and ``transverse`` are (nodes, directions, infinity_directions),
    with shapes [b,R_f], [b,n,R_f], [b,n,r_inf]. The finite columns already
    follow the round's paired order. ``cross_actions`` contains W_TC Q_C,
    W_CT Q_T and (dW_CT/dz) Q_T at each column's own node, with shapes
    [b,n_T,R_C], [b,n_C,R_T], [b,n_C,R_T]. ``cross_moments`` contains
    M0_CT..M3_CT [b,n_C,n_T], half the physical z-series coefficients.
    Arrays remain parent-local inside an admitted batched linalg program.

    Returns G_CT, H_CT [b,R_C+2r_C,R_T+2r_T] and the full cross output
    panels O_TC, O_CT. No adjoint symmetry is imposed on a cross tile.
    """
    from gw.shared_pole_pencil import _finite_column_g

    zc, qc, ic = charge
    zt, qt, it = transverse
    tc, ct, derivative = cross_actions
    left = matmul(tc, qt, transa="C")
    right = matmul(qc, ct, transa="C")
    g = _finite_column_g(left, right, matmul(qc, derivative, transa="C"), zc, zt)
    h = zt[:, None, :] * g - left
    mt = tuple(2 * matmul(m, it) for m in cross_moments)
    mc = tuple(2 * matmul(m, ic, transa="C") for m in cross_moments)
    # Infinity rows on the C side and columns on the T side use the
    # same resolvent identity, with each side's own complex coordinate.
    bottom0 = matmul(ic, ct, transa="C")
    bottom1 = zt[:, None, :] * bottom0 - matmul(mc[0], qt, transa="C")
    bottom2 = zt[:, None, :] * bottom1 - matmul(mc[1], qt, transa="C")
    top0 = matmul(tc, it, transa="C")
    top1 = jnp.conj(zc)[:, :, None] * top0 - matmul(qc, mt[0], transa="C")
    top2 = jnp.conj(zc)[:, :, None] * top1 - matmul(qc, mt[1], transa="C")
    p = tuple(matmul(ic, m, transa="C") for m in mt)
    block = lambda a, b, c, d: jnp.concatenate((
        jnp.concatenate((a, b), axis=-1), jnp.concatenate((c, d), axis=-1)), axis=-2)
    g = block(g, jnp.concatenate((top0, top1), axis=-1),
              jnp.concatenate((bottom0, bottom1), axis=-2), block(p[0], p[1], p[1], p[2]))
    h = block(h, jnp.concatenate((top1, top2), axis=-1),
              jnp.concatenate((bottom1, bottom2), axis=-2), block(p[1], p[2], p[2], p[3]))
    return g, h, jnp.concatenate((tc, mc[0], mc[1]), axis=-1), jnp.concatenate((ct, mt[0], mt[1]), axis=-1)


def cross_pencil_block(left, right, samples, *, matmul):
    """Compute one rectangular Hermite block, report equation A.1.

    Parameters
    ----------
    left, right : tuple
        Node and direction panel, ``(s, Q[b,n,r])``. Nodes are z² for
        the even pencil and z for the signed pencil, in Ry² and Ry.
    samples : tuple
        CT at conjugate(left node), CT at right node, and its derivative
        at right node. Arrays are complex ``[b,n_C,n_T]``. The derivative
        is with respect to the pencil coordinate. CT at the conjugate
        node must come from TC's adjoint, never from CT's own adjoint.
    matmul : callable
        Constructor's local/service GEMM with transa/transb support.

    Returns
    -------
    tuple
        Cross G and H, complex ``[b,r_C,r_T]``. Units follow the input
        state normalization. Parent sharding is inherited from the caller.
    """
    a, qc = left
    b, qt = right
    wa, wb, derivative = samples
    project = lambda w: matmul(qc, matmul(w, qt), transa="C")
    at_left = project(wa)
    delta = b - jnp.conj(a)
    confluent = delta == 0
    g = jnp.where(confluent, -project(derivative),
                  (at_left - project(wb)) / jnp.where(confluent, 1, delta))
    return g, b * g - at_left


def joint_sector_pencil(charge, transverse, cross, *, matmul):
    """Project CT on the two metric-corrected sector spans (plan 12.2).

    Parameters
    ----------
    charge, transverse : tuple
        ``(Y, values, own_output, cross_output)``. Y is ``[b,R,K]``;
        values ``[b,K]`` are squared Ritz poles for even data or signed
        inverse poles for ordered data. Output panels before projection
        have ``[b,n_endpoint,R]``. Charge cross_output has T rows;
        transverse cross_output has C rows. Y is normalized in G for
        even data, H for ordered data. Only retained columns enter.
    cross : tuple
        Cross definite and value members ``[b,R_C,R_T]``: (G_CT,H_CT)
        for even data, (H_CT,G_CT) for ordered data.
    matmul : callable
        Constructor's service/local GEMM. All arrays stay parent-local
        inside the admitted batched linalg stage.

    Returns
    -------
    tuple
        Joint definite member, value member, and the two full output
        panels. No Hermitization, clipping or Gram repair is performed.
    """
    yc, vc, oc, tc = charge
    yt, vt, ot, ct = transverse
    project = lambda a: matmul(yc, matmul(a, yt), transa="C")
    metric, value = map(project, cross)
    adj = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))
    block = lambda a, b, d: jnp.concatenate((
        jnp.concatenate((a, b), axis=-1),
        jnp.concatenate((adj(b), d), axis=-1)), axis=-2)
    # Inactive columns of a batched retained span are exactly zero. Their
    # metric is zero too; assigning them an identity invents latent states.
    ic = jnp.eye(vc.shape[-1], dtype=metric.dtype)[None] * jnp.any(yc != 0, axis=-2)[:, None, :]
    it = jnp.eye(vt.shape[-1], dtype=metric.dtype)[None] * jnp.any(yt != 0, axis=-2)[:, None, :]
    return (block(ic, metric, it),
            block(ic * vc[:, None, :], value, it * vt[:, None, :]),
            jnp.concatenate((matmul(oc, yc), matmul(ct, yt)), axis=-1),
            jnp.concatenate((matmul(tc, yc), matmul(ot, yt)), axis=-1))


def reduce_sector_pencil(pencil, *, eigh, matmul, gates):
    """Solve the definite joint pair without modifying either member.

    ``pencil`` is (metric, value, O_C, O_T), stacked on [b,...], from
    :func:`joint_sector_pencil`. The returned (c_C,c_T,lambda,active)
    evaluates as c_C (s-lambda)^-1 c_T.H for even data, and as
    c_C (z*lambda-1)^-1 c_T.H for ordered data. The latter follows
    report equation 5.5. The constructor must refuse false diagnostics.

    ``eigh`` and ``matmul`` are the admitted batched linalg callables.
    Existing normalized-Gram thresholds control rank revelation; no
    cross-sector passivity or PSD repair is applied.
    """
    from gw.shared_pole_reduction import _metric_inverse_root

    metric, value, oc, ot = pencil
    gamma, u = eigh(metric)
    top = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(top > 0, top, 1)
    valid = ((top > 0) & jnp.all(jnp.isfinite(gamma), axis=-1)
             & (ratio >= gates["normalized_gram_validity"]["threshold"]))
    keep = ((top[:, None] > 0) &
            (gamma > gates["normalized_gram_keep"]["threshold"] * top[:, None]))
    y = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]
    null = jnp.eye(metric.shape[-1], dtype=metric.dtype)[None] * (~keep)[:, None, :]
    reduced_metric = matmul(y, matmul(metric, y), transa="C")
    correction, corrected, diagnostics = _metric_inverse_root(
        reduced_metric + null, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    y = matmul(y, correction) * keep[:, None, :]
    reduced = matmul(y, matmul(value, y), transa="C")
    sentinel = -(jnp.linalg.norm(reduced, axis=(-2, -1)) + 1)
    values, rotation = eigh(reduced + null * sentinel[:, None, None])
    count = jnp.sum(keep, axis=-1)
    active = jnp.arange(metric.shape[-1])[None] >= metric.shape[-1] - count[:, None]
    coefficients = matmul(y, rotation) * active[:, None, :]
    return (matmul(oc, coefficients), matmul(ot, coefficients),
            jnp.where(active, values, 1), active), dict(
                diagnostics, gram_valid=valid, gram_min_relative=ratio,
                retained_metric_positive=corrected, retained_rank=count)


def sector_cauchy_schwarz(metrics, *, eigh_charge, eigh_current, matmul, gates):
    """Report the squared cross norm in a positive sector metric.

    ``metrics=(C,CT,T)`` consists of [b,n_C,n_C], [b,n_C,n_T], and
    [b,n_T,n_T] Hermitian diagonal metrics and a rectangular cross block,
    for example the positive spectral moment. Cauchy--Schwarz is
    ||C^(-1/2) CT T^(-1/2)||_2² <= 1. This is not an inequality on an
    arbitrary complex-frequency W tile. No input is repaired or replaced.
    A cross block outside either metric support is reported as infinity.
    Service eigenplans and the caller's parent-local GEMM own all algebra.
    """
    c, cross, t = metrics
    cutoff = gates["normalized_gram_keep"]["threshold"]

    def inverse_root(a, eigh):
        values, vectors = eigh(a)
        keep = values > cutoff * values[:, -1:]
        scale = keep / jnp.sqrt(jnp.where(keep, values, 1))
        return (matmul(vectors * scale[:, None, :], vectors, transb="C"),
                matmul(vectors * keep[:, None, :], vectors, transb="C"),
                values[:, 0])

    ic, pc, min_c = inverse_root(c, eigh_charge)
    it, pt, min_t = inverse_root(t, eigh_current)
    whitened = matmul(ic, matmul(cross, it))
    eigenvalues, _ = eigh_charge(matmul(whitened, whitened, transb="C"))
    norm = jnp.linalg.norm(cross, axis=(-2, -1))
    outside = jnp.linalg.norm(cross-matmul(pc, matmul(cross, pt)), axis=(-2, -1))
    defect = outside / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny)
    supported = defect <= gates["retained_subspace_moments"]["threshold"]
    return dict(cauchy_schwarz_squared=jnp.where(supported, eigenvalues[:, -1], jnp.inf),
                support_relative=defect, charge_metric_min=min_c, current_metric_min=min_t)
