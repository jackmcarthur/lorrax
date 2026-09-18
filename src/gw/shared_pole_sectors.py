"""Charge/current cross pencils on stacks of parent-local arrays.

The joint projection is plan section 12.2. Both endpoint output panels
must be retained: C[X_C,X_T] and T[X_C,X_T]. For the ordered response the
definite member is H, not G (shared-pole report equation 5.5).
These functions run inside the constructor's batched linalg stage; they
neither move an operator to the host nor prescribe a processor mesh.
"""

from functools import lru_cache

import jax.numpy as jnp
from gw.shared_pole_pencil import _matrix_layout


@lru_cache(maxsize=None)
def _sector_read_programs(mesh,indices,masks,execution):
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_execution import face_program
    def select(a):
        a=jnp.take(jnp.take(a,jnp.asarray(indices[0]),axis=-2),jnp.asarray(indices[1]),axis=-1)
        return jnp.where(jnp.asarray(masks[0])[:,None]&jnp.asarray(masks[1])[None,:],a,0)
    if execution=='face':
        return face_program(select,mesh),face_program(lambda *a:jnp.concatenate(a,axis=1),mesh)
    spec=P(('x','y'))
    return (jax.jit(shard_map(select,mesh=mesh,in_specs=spec,out_specs=spec,check_vma=False)),
            jax.jit(shard_map(lambda *a:jnp.concatenate(a,axis=1),mesh=mesh,
                             in_specs=spec,out_specs=spec,check_vma=False)))


def read_sector_round(io, meta, bank, header, ids, endpoints, *, sample_span=None,
                      fields=('Wc','dWc_ds'), retained=(), execution='local'):
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
    from gw.shared_pole_execution import face_program
    select,join=_sector_read_programs(io.mesh,tuple(tuple(map(int,i)) for i in indices),
                                     tuple(tuple(map(bool,m)) for m in masks),execution)
    ledger=meta.shared_pole_capacity
    ambient=ledger.live_stages
    keep=list(retained)
    out={}
    try:
        for field in fields:
            sample=field in ('Wc','dWc_ds','Wc_mirror','dWc_mirror_ds')
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
                    sample_span=span,fields=(field,),partition_spec=None if execution == "face" else spec,
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


def construct_sector_poles(bank, meta, config, *, mesh_xy, output):
    """Construct current-map CC, TT and joint-span CT models and their stores.

    The bank contains W-W_infinity and its ordered moments. CC uses n_C
    rows, TT uses 3*n_T rows, with the same physical supports and separate
    directions, reductions and 1.8*n budgets. All retained arrays count
    against the existing whole-map capacity ledger.

    Signed stability is the positive retained H of the ordered pencil; see
    docs/architecture/shared_pole_model.md. Scalar positive-V upper passivity
    is inapplicable. Held W and moment residuals are diagnostics; physical
    accuracy is measured independently on the integrated sector Sigma.
    """
    import jax
    import numpy as np
    from pathlib import Path
    from common import timing
    from common.collectives import rank0_transaction
    from file_io.slab_io import SlabIO
    from file_io.shared_pole_store import (validate_shared_pole_bank, _metadata,
        write_shared_pole_model,write_shared_pole_sector_manifest)
    from gw.shared_pole_local import parent_rounds,batch_to_face,canonical_factors,face_rows
    from gw.shared_pole_screening import _json
    from gw.shared_pole_directions import _sample_point
    from jax.sharding import NamedSharding,PartitionSpec as P
    from runtime.padding import padded_axis
    from symmetry_maps import minus_q_parent_partners

    header=validate_shared_pole_bank(bank['path'],expected_identity=bank['identity'],
                                    mesh_xy=mesh_xy,require_complete=True)
    if header.get('mirror_mode')!='literal_same_operator_v1':
        raise ValueError('GATE shared_pole_sector_mirror: photon bank requires authenticated literal same-operator mirrors')
    sample_fields=('Wc','dWc_ds','Wc_mirror','dWc_mirror_ds')
    recipe=meta.shared_pole_recipe
    if header['identity'] != bank['identity']:
        raise ValueError('GATE shared_pole_bank_state: current sector bank identity mismatch')
    qt,operations=header['qirr'],header['operations']
    partner,row=minus_q_parent_partners(header['q_irr_full_idx'],qt['irr_idx_q'],
        qt['sym_idx_q'],kgrid=header['grid'],sym_mats_k=np.asarray(operations['rotation']),
        antiunitary=np.asarray(operations['antiunitary'],bool),
        authorized_rows=operations['authorized_rows'])
    sector_headers=[_metadata(meta,table,recipe,bank['identity'],True,basis=basis,sector=sector)
               for table,basis,sector in zip(bank['sector_tables'],bank['mu_bases'],('CC','TT'))]
    fit_span=(int(min(recipe['fit_ids'])),int(max(recipe['fit_ids']))+1)
    receipts=[];stores={};placed=[]
    root=Path(output).parent
    to_face=batch_to_face(mesh_xy)
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    from gw.shared_pole_execution import sector_round_schedule,is_face
    for ids,real,slots,execution in sector_round_schedule(bank,header,meta,config,mesh_xy,partner):
        sectors=[];retained=[]
        for family,name in enumerate(('CC','TT')):
            with timing.fenced_section('spole.sector.'+name):
                with SlabIO(bank['path'],mode='r',mesh=mesh_xy) as io:
                    exact=read_sector_round(io,meta,bank,header,ids,(family,family),
                        fields=('M0','M1','M2','M3'),retained=retained,execution=execution)
                    samples=read_sector_round(io,meta,bank,header,ids,(family,family),
                        sample_span=fit_span,fields=sample_fields,retained=(*retained,*exact.values()),execution=execution)
                geometry=dict(components=3 if family else 1,basis=bank['mu_bases'][family],
                    ids=ids,real=real,header=sector_headers[family],partner_parent=partner,
                    partner_row=row,slots=slots,sym=bank['tables']['sym'],sample_lo=fit_span[0],
                    sector=name)
                model=construct_diagonal_sector_round(samples,exact,meta,config,geometry,
                    mesh_xy=mesh_xy,retained=retained)
                del samples,exact
                sectors.append(model)

                retained.extend(jax.tree.leaves((model['model'],model['signed'],
                    model['coefficients'],model['infinity'],tuple(s[1:] for s in model['states']))))
                reduction,zero,_,_=model['diagnostics']
                counts=np.asarray(model['vectors'][1]).sum(axis=-1).tolist()
                receipts.append(dict(sector=name,parents=ids[:real],K=counts[:real],
                    gram_min_relative=np.asarray(reduction['gram_min_relative'])[:real].tolist(),
                    zero_policy=np.asarray(zero['zero_policy'])[:real].tolist()))
                path=Path(output).with_name('sector_diagonal_receipt.json')
                rank0_transaction(path,stage='sector.diagonal_receipt',
                    write=lambda:path.write_text(_json(dict(identity=bank['identity'],
                        status='DIAGONAL_SPANS_ONLY',rounds=receipts))+'\n'))
        with timing.fenced_section('spole.sector.CT'):
            with SlabIO(bank['path'],mode='r',mesh=mesh_xy) as io:
                ct=read_sector_round(io,meta,bank,header,ids,(0,1),sample_span=fit_span,
                                      fields=sample_fields,retained=retained,execution=execution)
                tc=read_sector_round(io,meta,bank,header,ids,(1,0),sample_span=fit_span,
                                      fields=sample_fields,retained=(*retained,*ct.values()),execution=execution)
                cm=read_sector_round(io,meta,bank,header,ids,(0,1),fields=('M0','M1','M2','M3'),
                                      retained=(*retained,*ct.values(),*tc.values()),execution=execution)
            cross=construct_cross_sector_round(sectors,(ct,tc),cm,meta,config,mesh_xy=mesh_xy,
                sample_lo=fit_span[0],partner_slots=slots,real=real)
            with SlabIO(bank['path'],mode='r',mesh=mesh_xy) as io:
                c1=read_sector_round(io,meta,bank,header,ids,(0,0),fields=('M1',),
                    retained=(*retained,*ct.values(),*tc.values(),*cm.values(),*jax.tree.leaves(cross['models'])),execution=execution)['M1']
                t1=read_sector_round(io,meta,bank,header,ids,(1,1),fields=('M1',),
                    retained=(*retained,*ct.values(),*tc.values(),*cm.values(),c1,*jax.tree.leaves(cross['models'])),execution=execution)['M1']
            cauchy=sector_moment_cauchy((c1,cm['M1'],t1),sectors,mesh_xy=mesh_xy)
            del c1,t1
            del ct,tc,cm
        models=(sectors[0]['model'],sectors[1]['model'],*cross['models'])
        signed=(tuple((s['signed'][0],s['signed'][0],*s['signed'][1:]) for s in sectors)
                +(cross['signed'],))
        budget=cross['budget']
        budget.retained_panels=tuple(jax.tree.leaves((models,signed)))
        budget.live(())
        # Each held tile is read, scored, and released before the next support.
        held_rows={name:[] for name in ('CC','TT','CT')}
        for name,endpoint_pair,model in zip(held_rows,((0,0),(1,1),(0,1)),signed):
            for sample_id in recipe['held_ids']:
                sample_id=int(sample_id)
                with SlabIO(bank['path'],mode='r',mesh=mesh_xy) as io:
                    held=read_sector_round(io,meta,bank,header,ids,endpoint_pair,
                                           sample_span=(sample_id,sample_id+1),execution="face" if is_face(model[0]) else "local")
                errors=sector_held_errors(model,held,_sample_point(recipe,sample_id),mesh_xy=mesh_xy)
                held_rows[name].append(dict(sample_id=sample_id,
                    Wc=np.asarray(errors)[:real,0].tolist(),dWc_ds=np.asarray(errors)[:real,1].tolist()))
                del held
        for key,value in cauchy.items():
            values=np.asarray(value)[:real]
            expected_infinity=(key=='cauchy_schwarz_squared') & np.isposinf(values)
            if not np.all(np.isfinite(values) | expected_infinity):
                raise ValueError(f'GATE shared_pole_sector_nonfinite: spectral moment {key}')
        row_receipt=dict(parents=ids[:real],execution=execution,
            mesh_shape={axis:int(mesh_xy.shape[axis]) for axis in ("x","y")},held=held_rows,
            CT_gram_min_relative=np.asarray(cross['diagnostics']['gram_min_relative'])[:real].tolist(),
            CT_retained_metric_positive=np.asarray(cross['diagnostics']['retained_metric_positive'])[:real].tolist(),
            CT_zero_policy=np.asarray(cross['zero']['zero_policy'])[:real].tolist(),
            spectral_moment_cauchy={key:[float(v) if np.isfinite(v) else 'OUTSIDE_METRIC_SUPPORT'
                for v in np.asarray(value)[:real]] for key,value in cauchy.items()},
            scalar_upper_passivity='NOT_APPLICABLE_SIGNED_V',
            stability_scope='retained ordered H; exact full-space stability not established',
            sigma_accuracy='NOT_MEASURED')
        receipts.append(row_receipt)
        # Stage each canonical parent once. One round of face factors is live;
        # no all-parent factor stack or full photon operator is materialized.
        for name,family,model in zip(('CC','TT','CT_C','CT_T'),(0,1,0,1),models):
            poles,active=jax.tree.map(lambda a:np.asarray(jax.device_put(a,NamedSharding(mesh_xy,P()))),model[1:])
            counts=active.sum(axis=-1,dtype=np.int64)
            width=padded_axis(int(counts[:real].max()),mesh_xy,name='shared_pole_port',
                specs=((P('x','y'),0),(P('x','y'),1))).carrier
            factor=face_rows(mesh_xy,tuple(range(real)),width)(model[0] if is_face(model[0]) else to_face(model[0]))
            for slot,q in enumerate(ids[:real]):
                public=canonical_factors(mesh_xy,(slot,),components=3 if family else 1)(factor)
                filename=root/(name+'.h5')
                store_header=write_shared_pole_model(filename,public,
                    jax.device_put(poles[slot:slot+1,:width],NamedSharding(mesh_xy,P())),
                    counts[slot:slot+1],q_span=(q,q+1),meta=meta,tables=bank['sector_tables'][family],
                    recipe=recipe,receipts=dict(identity=bank['identity'],constructor=row_receipt),
                    ordered=True,basis=bank['mu_bases'][family],sector=name)
                stores[name]=(str(filename),store_header)
                del public
            del factor
        placed.extend(ids[:real])
        budget.retained_panels=()
        ledger.live_stages=upstream
        del sectors,cross,models,signed,retained,model
    if sorted(placed)!=list(range(header['n_q_irr'])):
        raise ValueError('GATE shared_pole_sector_rounds: each parent must be written once')
    handle=write_shared_pole_sector_manifest(root/'sectors.json',models=stores,bank=bank,
        identity=bank['identity'],receipts=dict(rounds=receipts,status='CONSTRUCTED',
            acceptance='signed-retained-H-v1',sigma_accuracy='NOT_MEASURED'),mesh_xy=mesh_xy)
    return dict(handle=handle,identity=bank['identity'],status='CONSTRUCTED',
                q_receipts=receipts,capacity=ledger.receipt())


def sector_held_errors(signed, samples, z, *, mesh_xy):
    """Relative W and dW/ds diagnostics at one held physical complex z.

    Signed endpoints are (C_L,C_R,mu,active), parent-sharded; sample tiles
    have one support. Only the resulting two scalars per parent replicate.
    """
    import jax
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_local import _mm
    from gw.shared_pole_execution import is_face,face_program,face_matmul
    face=is_face(signed[0])
    mm=face_matmul(mesh_xy) if face else _mm
    from functools import partial
    body=partial(_sector_held_equations,z=z,mm=mm)
    spec=P(('x','y'))
    if face:
        from gw.shared_pole_execution import held_program
        value=held_program(mesh_xy)(*signed,samples['Wc'],samples['dWc_ds'],jnp.asarray(z))
    else:
        program=jax.jit(shard_map(body,mesh=mesh_xy,in_specs=(spec,)*6,out_specs=spec,check_vma=False))
        value=program(*signed,samples['Wc'],samples['dWc_ds'])
    return jax.device_put(value,NamedSharding(mesh_xy,P()))


def _sector_held_equations(left,right,mu,active,w,d,z,*,mm):
    denominator=z*mu-1
    weights=jnp.where(active,1/denominator,0)
    slopes=jnp.where(active,-mu/denominator**2/(2*z),0)
    result=[]
    for coefficient,exact in ((weights,w[:,0]),(slopes,d[:,0])):
        value=mm(left*coefficient[:,None,:],right,transb='C')
        norm=jnp.linalg.norm(exact,axis=(-2,-1))
        result.append(jnp.linalg.norm(value-exact,axis=(-2,-1))/jnp.maximum(norm,jnp.finfo(norm.dtype).tiny))
    return jnp.stack(result,axis=-1)


def sector_moment_cauchy(metrics, sectors, *, mesh_xy):
    """Report the Cauchy–Schwarz diagnostic of the common physical M1 metric."""
    import jax
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_local import _mm
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_execution import is_face,face_program,face_matmul,face_eigh
    face=is_face(metrics[0])
    mm=face_matmul(mesh_xy) if face else _mm
    plans=[(lambda a:face_eigh(mesh_xy,a.shape[-1]).batched(a)) if face else s['budget'].eigenplan(m.shape[-1]).native_fn
           for s,m in zip(sectors,(metrics[0],metrics[2]))]
    def body(c,ct,t):
        return sector_cauchy_schwarz((c,ct,t),eigh_charge=plans[0],eigh_current=plans[1],matmul=mm,gates=gates)
    spec=P(('x','y'))
    from gw.shared_pole_execution import cauchy_program
    program=cauchy_program(mesh_xy) if face else jax.jit(shard_map(body,mesh=mesh_xy,in_specs=(spec,)*3,out_specs=spec,check_vma=False))
    result=program(*metrics)
    return jax.tree.map(lambda a:jax.device_put(a,NamedSharding(mesh_xy,P())),result)


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

    from gw.shared_pole_execution import is_face, face_program, face_reduce_round
    execution='face' if is_face(samples['Wc']) else 'local'
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
                               mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,upstream=(),execution=execution)
    budget.batch_width=len(geometry['ids'])
    budget.retained_panels=tuple(retained)
    # Four photon sample fields at every fit support and four moment fields
    # are resident during selection. Derive the face count from these exact
    # read dictionaries so admission prices the live panel set.
    selection_faces=(sum(int(panel.shape[1]) for panel in samples.values())
                     +len(moments))
    budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                selection_faces=selection_faces)
    eig=budget.eigenplan(local_meta.n_rmu_padded)
    svd=budget.eigenplan(2*local_meta.n_rmu_padded)
    extent=lambda width:padded_axis(width,mesh_xy,name='shared_pole_port',
        specs=((P('x','y'),0),(P('x','y'),1))).carrier
    qi,values=distrib_la.leading_eigenvectors(moments['M1'],min(n,recipe['infinity_width']),
        eigh_plan=eig,column_extent=extent,multiplet_tol=recipe['multiplet_relative_tolerance'],
        real_rows=None if execution=='face' else geometry['real'])
    kernels=_round_kernels(mesh_xy,'face' if execution=='face' else 'batch')
    infinity=(qi,*(kernels.apply(moments[name],qi) for name in ('M0','M1','M2','M3')))
    # Literal mirrors use the original parent's stored operator directly;
    # no spatial partner action is needed in either execution layout.
    action=(None,None,None)
    rotation=None
    states,counts,roles=select_round_states(samples,recipe,sample_lo=geometry['sample_lo'],
        real=geometry['real'],mesh_xy=mesh_xy,eigh_plan=eig,svd_plan=svd,column_extent=extent,
        logical_n=n,ordered=True,exchange=(geometry['slots'],*action),current_rotation=rotation)
    tables=round_tables(counts,[s[1].shape[-1] for s in states],[s[0] for s in states],
        [v.shape[-1] for v in values],qi.shape[-1],column_extent=extent,ordered=True,odd_moments=True)
    side=tables['active'].shape[-1]
    budget.retained_panels=(*retained,*infinity,*(v for s in states for v in s[1:]),
                            *samples.values(),*moments.values())
    if execution == 'face':
        reduced=face_reduce_round(states,infinity,tables,real=geometry['real'],mesh=mesh_xy,
            budget=budget,ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True)
    else:
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
    # The returned planner must not retain the just-consumed full sample and
    # moment arrays through its accounting view after the caller releases them.
    budget.retained_panels=tuple(retained)
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                endpoint_action=(*action,rotation),recipe=recipe,budget=budget,execution=execution)


def construct_cross_sector_round(sectors, samples, moments, meta, config, *,
                                 mesh_xy, sample_lo, partner_slots, real):
    """Run CT on the two current-map diagonal spans, keeping both outputs.

    ``samples=(CT,TC)`` contains the native rectangular Wc/dWc_ds rounds;
    moments is the CT M0..M3 round. All operators are parent-sharded. The
    signed physical photon interaction is admitted by the unchanged positive
    retained-H checks, not by the scalar positive-V upper passivity bound.
    """
    import copy
    import jax
    import numpy as np
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity
    from gw.shared_pole_local import _batch_put
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates

    from gw.shared_pole_execution import is_face,face_program
    execution="face" if is_face(samples[0]["Wc"]) else "local"
    charge, transverse = sectors
    ct, tc = samples
    retained = jax.tree.leaves(tuple((s['model'],s['signed'],s['coefficients'],
        s['infinity'],tuple(state[1:] for state in s['states'])) for s in sectors))
    local_meta = copy.copy(meta)
    local_meta.n_rmu_padded = sum(s['model'][0].shape[-2] for s in sectors)
    budget = ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
        mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,upstream=(),execution=execution)
    budget.batch_width = charge['model'][0].shape[0]
    budget.retained_panels = (*retained,*ct.values(),*tc.values(),*moments.values())
    # Cross assembly still has the original rectangular pencil. The sum
    # prices its envelope even when the retained joint pair is much smaller.
    original_side = sum(s['coefficients'].shape[-2] for s in sectors)
    budget.plan(original_side,phase='reduction')
    actions=[]
    for source, forward, reverse, left, right in (
            (charge,tc,ct,transverse,charge),
            (transverse,ct,tc,charge,transverse)):
        panels=(forward['Wc'],reverse['Wc'],forward['dWc_ds'],reverse['dWc_ds'],
                forward['Wc_mirror'],reverse['Wc_mirror'],
                forward['dWc_mirror_ds'],reverse['dWc_mirror_ds'])
        actions.append(cross_round_actions(panels,source['states'],source['roles'],
            source['recipe'],sample_lo=sample_lo,mesh_xy=mesh_xy,
            partner_slots=partner_slots,
            endpoint_actions=(left['endpoint_action'],right['endpoint_action'])))

    spec=P(('x','y'))
    packed=[]
    for sector in sectors:
        # Drop only exactly inactive carrier columns. This is a storage
        # compaction of the retained span, not a second physical rank cut.
        width=int(jnp.max(jnp.sum(sector['signed'][2],axis=-1)))
        if execution == 'face':
            from runtime.padding import padded_axis
            width=padded_axis(width,mesh_xy,name='shared_pole_port',
                specs=((P('x','y'),0),(P('x','y'),1))).carrier
        from functools import partial
        compact=partial(_compact_sector_equations,width=width)
        if execution=='face':
            from gw.shared_pole_execution import compact_program
            compact=compact_program(mesh_xy,width)
        else:
            compact=jax.jit(shard_map(compact,mesh=mesh_xy,in_specs=(spec,spec),out_specs=(spec,spec),check_vma=False))
        y,signed=compact(sector['coefficients'],sector['signed'])
        put=(lambda a:jax.device_put(a,NamedSharding(mesh_xy,P()))) if execution=='face' else (lambda a:_batch_put(mesh_xy,a))
        packed.append((put(sector['tables']['points']),
            put(sector['tables']['order']),
            (tuple(s[1] for s in sector['states']),tuple(s[2] for s in sector['states'])),
            sector['infinity'],y,signed))
    side=sum(s[4].shape[-1] for s in packed)
    budget.retained_panels=(*budget.retained_panels,*jax.tree.leaves((actions,packed)))
    budget.plan(max(side,original_side),phase='reduction')
    signed,diagnostics=reduce_cross_round(*packed,tuple(actions),
        tuple(moments[f'M{i}'] for i in range(4)),mesh_xy=mesh_xy,
        native_eigh=None if execution=="face" else budget.eigenplan(side).native_fn,gates=gates)
    for name in ('gram_valid','retained_metric_positive'):
        if not bool(jnp.all(diagnostics[name][:real])):
            raise ValueError(f'GATE shared_pole_sector_{name}: sector=CT; no repair')
    models,zero=positive_cross_models(signed,mesh_xy=mesh_xy,gates=gates)
    if not bool(jnp.all(zero['zero_policy'][:real])):
        raise ValueError('GATE shared_pole_sector_zero_ritz: sector=CT')
    budget.retained_panels=tuple(retained)
    replicated=NamedSharding(mesh_xy,P())
    return dict(models=models,signed=signed,
                diagnostics=jax.tree.map(lambda a:jax.device_put(a,replicated),diagnostics),
                zero=jax.tree.map(lambda a:jax.device_put(a,replicated),zero),budget=budget)


def _compact_sector_equations(y,signed,*,width):
    c,mu,active=signed
    order=jnp.argsort(~active,axis=-1,stable=True)[:,:width]
    return (jnp.take_along_axis(y,order[:,None,:],axis=-1),
        (jnp.take_along_axis(c,order[:,None,:],axis=-1),
         jnp.take_along_axis(mu,order,axis=-1),
         jnp.take_along_axis(active,order,axis=-1)))


def positive_cross_models(signed, *, mesh_xy, gates):
    """Positive-pole CT endpoint models with the same ordering and zero mask.

    Finite/infinite and low-pole dropped weight are checked independently
    on each physical endpoint; a large charge norm cannot hide a lost current
    factor. The signed model remains available for held-frequency checks.
    """
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_gates import apply_shared_pole_zero_policy,sort_shared_pole_columns

    from functools import partial
    body=partial(_positive_cross_equations,gates=gates)
    from gw.shared_pole_execution import is_face,face_program
    if is_face(signed[0]):
        from gw.shared_pole_execution import positive_cross_program
        return positive_cross_program(mesh_xy)(*signed)
    spec=P(('x','y'))
    return jax.jit(shard_map(body,mesh=mesh_xy,in_specs=(spec,)*4,
                              out_specs=(spec,spec),check_vma=False))(*signed)


def _positive_cross_equations(left,right,mu,active,*,gates):
    from gw.shared_pole_gates import apply_shared_pole_zero_policy,sort_shared_pole_columns
    cut=gates['normalized_gram_keep']['threshold']*jnp.max(jnp.abs(jnp.where(active,mu,0)),axis=-1)
    retained=active & (jnp.abs(mu)>cut[:,None])
    positive=retained & (mu>0)
    safe=jnp.where(positive,mu,1)
    models=[];checks=[]
    for c in (left,right):
        weight=jnp.sum(jnp.abs(c)**2,axis=-2)
        total=jnp.sum(jnp.where(active,weight,0),axis=-1)
        lost=jnp.sum(jnp.where(active & ~retained,weight,0),axis=-1)
        infinite=lost/jnp.where(total>0,total,1)
        b=c*(jnp.sqrt(2.)/safe*positive)[:,None,:]
        model,zero=apply_shared_pole_zero_policy((b,jnp.where(positive,1/safe**2,1),positive),gates=gates)
        zero['infinite_weight_fraction']=infinite
        zero['zero_policy'] &= infinite<=gates['zero_ritz_policy']['threshold']['max_dropped_weight_fraction']
        models.append(model);checks.append(zero)
    # The physical pole cutoff sets one mask independently of endpoint
    # weight. Admit only if BOTH endpoint loss budgets pass, then use one
    # permutation for the two files, including degenerate poles.
    same=jnp.all(models[0][2]==models[1][2],axis=-1)
    charge,order=sort_shared_pole_columns(models[0])
    current=(jnp.take_along_axis(models[1][0],order[:,None,:],axis=-1),charge[1],charge[2])
    return (charge,current),dict(zero_policy=same & checks[0]['zero_policy'] & checks[1]['zero_policy'],
                              charge=checks[0],current=checks[1])


def _literal_cross_products(panels,q,node,*,sample,mirror,conjugate,mm):
    """One literal same-operator CT action for both execution layouts."""
    w,wr,d,dr,wm,wrm,dwm,dwrm=panels
    if mirror:
        a,da=(wm,dwm) if conjugate else (wrm,dwrm)
        adjoint=not conjugate
    else:
        a,da=(wr,dr) if conjugate else (w,d)
        adjoint=conjugate
    a,da=a[:,sample],da[:,sample]
    if adjoint:
        a,da=jnp.swapaxes(jnp.conj(a),-1,-2),jnp.swapaxes(jnp.conj(da),-1,-2)
    return mm(a,q),mm(da,q)*(2*node)


def cross_round_actions(samples, states, roles, recipe, *, sample_lo, mesh_xy,
                        partner_slots, endpoint_actions):
    """Apply rectangular samples to the diagonal sectors' selected directions.

    ``samples=(W_LR,W_RL,dW_LR/ds,dW_RL/ds)`` are parent-local
    [P,S,n_L,n_R] (reverse blocks have reversed endpoint extents).
    With literal mirrors, four matching W/dW panels at ``-conj(z)`` follow.
    ``states``/``roles`` are the existing selection round's source states.
    ``endpoint_actions`` is ((alpha,inverse,phase,rotation)_L, ..._R);
    a charge endpoint has rotation=None, a current endpoint uses the
    symmetry service's polar time-odd [P,3,3] action and mu-major rows.
    Literal mirror panels already belong to the original q operator and
    require no partner-rank exchange or spatial action. The legacy four-panel
    route exchanges only direction/output panels through the authenticated
    partner permutation. Outputs follow the paired state order and the
    derivative is d/dz, report equation 5.3.
    """
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_directions import _sample_point

    from gw.shared_pole_execution import is_face,face_program,face_matmul
    from gw.shared_pole_local import _mm
    face=is_face(samples[0])
    mm=face_matmul(mesh_xy) if face else _mm
    batch=P(('x','y'))
    literal=len(samples)==8
    if len(samples) not in (4,8):
        raise ValueError('GATE shared_pole_sector_mirror: expected four direct or eight direct/mirror panels')
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
        exchange=mirror and z.real!=0 and not literal
        use_adjoint=not conjugate if mirror and not exchange else conjugate
        node=state[0]

        if face:
            if not literal:
                raise ValueError('distributed photon CT requires authenticated literal mirrors')
            from gw.shared_pole_execution import cross_action_program
            outputs.append(cross_action_program(mesh_xy,sid-sample_lo,mirror,conjugate)(
                samples,state[1],jnp.asarray(node)))
            continue

        def apply(*args):
            *panels,q,left,right=args
            if literal:
                return _literal_cross_products(panels,q,node,sample=sid-sample_lo,
                    mirror=mirror,conjugate=conjugate,mm=mm)
            w,wr,d,dr=panels[:4]
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
            o,d_o=mm(a,q),mm(da,q)*(2*node)
            if exchange:
                alpha,inverse,phase,rotation=left
                def back(x):
                    x=jax.lax.ppermute(x,('x','y'),perm)
                    return rotate(jnp.conj(phase)[:,:,None]*jnp.take_along_axis(x,alpha[:,:,None],axis=1),rotation,False)
                o,d_o=back(o),back(d_o)
            return o,d_o

        program=jax.jit(shard_map(apply,mesh=mesh_xy,
            in_specs=(batch,)*(len(samples)+3),out_specs=(batch,batch),check_vma=False))
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

    from gw.shared_pole_execution import is_face,face_program,face_matmul,face_eigh
    face=is_face(charge[4])
    mm=face_matmul(mesh_xy) if face else _mm
    eigh=(lambda a:face_eigh(mesh_xy,a.shape[-1]).batched(a)) if face else native_eigh
    from functools import partial
    body=partial(_cross_reduce_equations,mm=mm,eigh=eigh,gates=gates)
    spec=P(('x','y'))
    from gw.shared_pole_execution import cross_parent_program
    program=cross_parent_program(mesh_xy) if face else jax.jit(shard_map(body,mesh=mesh_xy,in_specs=(spec,)*4,
                            out_specs=(spec,spec),check_vma=False))
    return program(charge,transverse,cross,moments)


def _cross_reduce_equations(charge,transverse,cross,moments,*,mm,eigh,gates,matrix_sharding=None):
    def unpack(sector,actions):
        points,order,states,infinity,y,signed=sector
        def pack(panels):
            return _matrix_layout(jnp.take(jnp.concatenate(panels,axis=-1),order[0],axis=-1,mode='fill',fill_value=0), matrix_sharding)
        return points,pack(states[0]),infinity[0],pack(tuple(a[0] for a in actions)),pack(tuple(a[1] for a in actions))
    zc,qc,ic,tc,dtc=unpack(charge,cross[0])
    zt,qt,it,ct,dct=unpack(transverse,cross[1])
    g,h,otc,oct=ordered_cross_pencil((zc,qc,ic),(zt,qt,it),(tc,ct,dct),moments,matmul=mm,matrix_sharding=matrix_sharding)
    # The diagonal output panels are O_original. Reconstruct their
    # infinity columns from the same physical moments already in hand.
    def diagonal(sector):
        _,order,states,infinity,y,signed=sector
        own=jnp.take(jnp.concatenate(states[1],axis=-1),order[0],axis=-1,mode='fill',fill_value=0)
        return _matrix_layout(jnp.concatenate((own,2*infinity[1],2*infinity[2]),axis=-1), matrix_sharding)
    cc=(charge[4],charge[5][1],diagonal(charge),otc)
    tt=(transverse[4],transverse[5][1],diagonal(transverse),oct)
    return reduce_sector_pencil(joint_sector_pencil(cc,tt,(h,g),matmul=mm,matrix_sharding=matrix_sharding),
                               eigh=eigh,matmul=mm,gates=gates,matrix_sharding=matrix_sharding)


def ordered_cross_pencil(charge, transverse, cross_actions, cross_moments, *, matmul, matrix_sharding=None):
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
    return tuple(_matrix_layout(a, matrix_sharding) for a in
                 (g, h, jnp.concatenate((tc, mc[0], mc[1]), axis=-1),
                  jnp.concatenate((ct, mt[0], mt[1]), axis=-1)))


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


def joint_sector_pencil(charge, transverse, cross, *, matmul, matrix_sharding=None):
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
    return tuple(_matrix_layout(a, matrix_sharding) for a in (block(ic, metric, it),
            block(ic * vc[:, None, :], value, it * vt[:, None, :]),
            jnp.concatenate((matmul(oc, yc), matmul(ct, yt)), axis=-1),
            jnp.concatenate((matmul(tc, yc), matmul(ot, yt)), axis=-1)))


def reduce_sector_pencil(pencil, *, eigh, matmul, gates, matrix_sharding=None):
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
    null = _matrix_layout(jnp.eye(metric.shape[-1], dtype=metric.dtype)[None] * (~keep)[:, None, :], matrix_sharding)
    reduced_metric = matmul(y, matmul(metric, y), transa="C")
    correction, corrected, diagnostics = _metric_inverse_root(
        reduced_metric + null, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"], matrix_sharding=matrix_sharding)
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
