"""Charge/current cross pencils on stacks of parent-local arrays.

The joint projection is plan section 12.2. Both endpoint output panels
must be retained: C[X_C,X_T] and T[X_C,X_T]. For the ordered response the
definite member is H, not G (shared-pole report equation 5.5).
These functions run inside the constructor's batched linalg stage; they
neither move an operator to the host nor prescribe a processor mesh.
"""

from functools import lru_cache

import jax.numpy as jnp
from gw.shared_pole_pencil import _matrix_layout, _matrix_take_columns, _matrix_concat


def _contiguous_q_spans(ids, real, limit=4):
    """Canonical q spans from a (possibly permuted) constructor round.

    The small limit bounds the extra public factor and store conversion panel;
    the store still admits each panel against the current map capacity ledger.
    """
    ordered=sorted(range(real),key=lambda slot:ids[slot])
    spans=[]
    for slot in ordered:
        if not spans or len(spans[-1])==limit or ids[slot]!=ids[spans[-1][-1]]+1:
            spans.append([])
        spans[-1].append(slot)
    return tuple((ids[slots[0]],ids[slots[-1]]+1,tuple(slots)) for slots in spans)


@lru_cache(maxsize=None)
def _sector_read_programs(mesh,indices,masks,execution):
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_execution import face_program
    def select(a):
        if execution == 'face':
            from common.staged_reshard import permute_sharded_axis
            face = P(*([None]*(a.ndim-2)), 'x', 'y')
            for axis,index in zip((-2,-1),indices):
                a=permute_sharded_axis(a,axis,index,mesh,face)
        else:
            a=jnp.take(jnp.take(a,jnp.asarray(indices[0]),axis=-2),jnp.asarray(indices[1]),axis=-1)
        return jnp.where(jnp.asarray(masks[0])[:,None]&jnp.asarray(masks[1])[None,:],a,0)
    if execution=='face':
        return face_program(select,mesh),face_program(lambda *a:jnp.concatenate(a,axis=1),mesh)
    spec=P(('x','y'))
    return (jax.jit(shard_map(select,mesh=mesh,in_specs=spec,out_specs=spec,check_vma=False)),
            jax.jit(shard_map(lambda *a:jnp.concatenate(a,axis=1),mesh=mesh,
                             in_specs=spec,out_specs=spec,check_vma=False)))


def _sector_indices(bank, endpoints):
    """Row/column index and active mask of each endpoint family inside its stored rectangle."""
    import numpy as np
    layout=bank['photon_layout']
    indices=[];masks=[]
    for family in endpoints:
        basis=bank['mu_bases'][family]
        mu=basis.pack_host(np.arange(basis.n_canonical,dtype=np.int32),axis=0)
        components=3 if family else 1
        width=layout.carrier_extent(family)//layout.mesh_side
        local=components*width
        index=(mu[:,None]//width*local+np.arange(components)[None]*width+mu[:,None]%width)
        indices.append(tuple(map(int,index.reshape(-1))))
        masks.append(tuple(map(bool,np.repeat(basis.active_mask,components))))
    return tuple(indices),tuple(masks)


def read_sector_round(io, meta, bank, header, ids, endpoints, *, sample_span=None, sample_ids=None,
                      fields=('Wc','dWc_ds'), retained=(), execution='local'):
    """Read a bounded sector span into its packed endpoint bases.

    The canonical photon store is mesh-interleaved; each selected family is
    converted to the existing MuBasis order before the constructor sees it.
    ``endpoints`` names C=0 or T=1 on each side. Returned TT/CT rows are
    mu-major, Cartesian-component-minor. Local rounds keep parents at
    P(('x','y')); a face round keeps one parent at P(None,'x','y').
    The store selects native sector hyperslabs, never full photon panels.
    It owns all I/O,
    authentication and transport; this function only selects sector rows.
    """
    import jax
    import numpy as np
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from file_io.shared_pole_store import read_shared_pole_bank

    indices,masks=_sector_indices(bank,endpoints)
    spec=P(('x','y'))
    select,join=_sector_read_programs(io.mesh,indices,masks,execution)
    ledger=meta.shared_pole_capacity
    ambient=ledger.live_stages
    keep=list(retained)
    out={}
    try:
        for field in fields:
            sample=field in ('Wc','dWc_ds')
            if sample and sample_ids is None and (sample_span is None or sample_span[1] <= sample_span[0]):
                raise ValueError('sector sample reads require a nonempty bounded sample_span or sample_ids')
            size=sum(a.size*a.dtype.itemsize//io.mesh.size for a in keep)
            row=ledger.reserve(f'sector.read.retained.{len(ledger.entries)}',
                resident_bytes_per_rank=size,workspace_bytes_per_rank=0,concurrent_with=ambient)
            ledger.live_stages=(*ambient,row['stage'])
            value=read_shared_pole_bank(io,meta=meta,header=header,q_ids=ids,
                sample_span=sample_span if sample else None,
                sample_ids=sample_ids if sample else None,fields=(field,),
                partition_spec=None if execution == "face" else spec,
                sector=tuple('T' if family else 'C' for family in endpoints))[field]
            # Reserve the selected output alongside the bounded input.
            output=value.shape[0]*(1 if not sample else value.shape[1])*len(indices[0])*len(indices[1])*value.dtype.itemsize//io.mesh.size
            workspace=value.shape[0]*(1 if not sample else value.shape[1])*len(indices[0])*value.shape[-1]*value.dtype.itemsize//io.mesh.size
            ledger.reserve(f'sector.read.select.{len(ledger.entries)}',
                resident_bytes_per_rank=value.size*value.dtype.itemsize//io.mesh.size+output,
                workspace_bytes_per_rank=workspace,concurrent_with=ledger.live_stages)
            out[field]=select(value)
            del value
            keep=list(retained)+list(out.values())
    finally:
        ledger.live_stages=ambient
    return out


def sector_line_selection(bank, meta, *, mesh_xy, execution, nq):
    """The photon bank's C and T families for the producer's line selection.

    Each endpoint block is cut from the face-tiled photon operator exactly as
    a sector read cuts it from the bank (the family rectangle of each rank's
    tile, then the packed-basis take and active mask), so the producer selects
    on the constructor's bits. Each family also stores the CT/TC actions on
    its directions.
    """
    from file_io.shared_pole_store import _resident_sector_select
    from gw.shared_pole_directions import LineSelection, selection_layout
    layout=bank['photon_layout']
    side=int(layout.mesh_side)
    c,t=(int(layout.carrier_extent(family))//side for family in (0,1))
    move,axis,_=selection_layout(mesh_xy,execution,int(nq))

    def block(value,endpoints):
        rectangle=_resident_sector_select(mesh_xy,3,tuple(0 if f==0 else c for f in endpoints),
                                          tuple(c if f==0 else 3*t for f in endpoints))(value)
        indices,masks=_sector_indices(bank,endpoints)
        select=_sector_read_programs(mesh_xy,indices,masks,execution)[0]
        return axis(select(move(rectangle)))
    families=[]
    for family,basis in enumerate(bank['mu_bases']):
        n=(3 if family else 1)*int(basis.n_logical)
        families.append(dict(name=('C','T')[family],index=family,
            recipe=sector_recipe(meta.shared_pole_recipe,n),logical_n=n,
            rows=(3 if family else 1)*int(basis.n_packed),cross=True))
    return LineSelection(families,block,mesh_xy=mesh_xy,ordered=True,execution=execution,nq=nq)


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
    from common.collectives import (device_put_process_local, gather_to_host,
                                    rank0_transaction)
    from file_io.shared_pole_store import (validate_shared_pole_bank, _metadata, open_shared_pole_bank,
        read_line_panels,write_shared_pole_model,write_shared_pole_sector_manifest)
    from gw.shared_pole_local import batch_to_face,canonical_factors,face_rows
    from gw.shared_pole_screening import _json
    from gw.shared_pole_directions import _sample_point
    from jax.sharding import NamedSharding,PartitionSpec as P

    header=validate_shared_pole_bank(bank['path'],expected_identity=bank['identity'],
                                    mesh_xy=mesh_xy,require_complete=True)
    line_lo,line_hi=(int(v) for v in header['line_panels']['sample_span'])
    if line_hi<=line_lo:
        raise ValueError('GATE shared_pole_line_panel: photon recipe has no fitted line sample')
    recipe=meta.shared_pole_recipe
    if header['identity'] != bank['identity']:
        raise ValueError('GATE shared_pole_bank_state: current sector bank identity mismatch')
    sector_headers=[_metadata(meta,table,recipe,bank['identity'],True,basis=basis,sector=sector)
               for table,basis,sector in zip(bank['sector_tables'],bank['mu_bases'],('CC','TT'))]
    # Line supports off the imaginary axis arrive as their stored states and
    # cross actions; the dense fitted samples (imaginary axis) are read whole.
    dense_fit=[int(i) for i in recipe['fit_ids'] if not line_lo<=int(i)<line_hi]

    def read_samples(io,endpoints,retained):
        # Wc/dWc_ds at the dense fitted samples.
        return read_sector_round(io,meta,bank,header,ids,endpoints,sample_ids=dense_fit,
            fields=('Wc','dWc_ds'),retained=retained,execution=execution)

    def read_line(io,family,cross=False):
        spec=None if execution=='face' else P(('x','y'))
        name=('C','T')[family]
        return {sid:read_line_panels(io,meta=meta,header=header,family=name,sample=sid,cross=cross,
                                     q_ids=ids,partition_spec=spec)
                for sid in range(line_lo,line_hi)}
    receipts=[];stores={};placed=[]
    root=Path(output).parent
    to_face=batch_to_face(mesh_xy)
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_execution import sector_round_schedule,is_face
    resolved_execution,execution_rows=sector_execution(
        meta,config,bank['mu_bases'],header['n_q_irr'],mesh_xy=mesh_xy,upstream=upstream)
    batch_width = int(mesh_xy.size)
    if resolved_execution == 'face':
        from gw.shared_pole_execution import sector_batch_width
        batch_width, batch_admission = sector_batch_width(
            meta,linalg_resolution({'linalg':config.backend.linalg}),recipe,execution_rows,
            mesh=mesh_xy,ledger=ledger,nq=header['n_q_irr'])
        for row in execution_rows:
            row['batch_admission'] = batch_admission
            row['parent_batch'] = batch_width
    sector_models,model_residence=_sector_model_residence(meta,config,header,bank['mu_bases'],
        execution_rows,mesh_xy=mesh_xy,root=root,upstream=upstream,route=resolved_execution)
    if sector_models is not None:
        upstream=ledger.live_stages=(*upstream,model_residence['stage'])
    for ids,real,slots,execution in sector_round_schedule(
            bank,header,meta,config,mesh_xy,execution=resolved_execution,
            batch_width=batch_width):
        sectors=[];retained=[]
        for family,name in enumerate(('CC','TT')):
            with timing.section('spole.sector.'+name, announce=True):
                with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                    exact=read_sector_round(io,meta,bank,header,ids,(family,family),
                        fields=('M0','M1','M2','M3'),retained=retained,execution=execution)
                    samples=read_samples(io,(family,family),(*retained,*exact.values()))
                    line=read_line(io,family)
                geometry=dict(components=3 if family else 1,basis=bank['mu_bases'][family],
                    ids=ids,real=real,header=sector_headers[family],sample_ids=dense_fit,
                    sector=name)
                model=construct_diagonal_sector_round(samples,exact,meta,config,geometry,
                    mesh_xy=mesh_xy,retained=retained,line=line)
                del samples,exact,line
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
        with timing.section('spole.sector.CT', announce=True):
            with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                ct=read_samples(io,(0,1),retained)
                tc=read_samples(io,(1,0),(*retained,*ct.values()))
                cm=read_sector_round(io,meta,bank,header,ids,(0,1),fields=('M0','M1','M2','M3'),
                                      retained=(*retained,*ct.values(),*tc.values()),execution=execution)
                line_cross=[read_line(io,family,cross=True) for family in (0,1)]
            cross=construct_cross_sector_round(sectors,(ct,tc),cm,meta,config,mesh_xy=mesh_xy,
                sample_ids=dense_fit,line_cross=line_cross,real=real)
            del line_cross
            with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                c1=read_sector_round(io,meta,bank,header,ids,(0,0),fields=('M1',),
                    retained=(*retained,*ct.values(),*tc.values(),*cm.values(),*jax.tree.leaves(cross['models'])),execution=execution)['M1']
                t1=read_sector_round(io,meta,bank,header,ids,(1,1),fields=('M1',),
                    retained=(*retained,*ct.values(),*tc.values(),*cm.values(),c1,*jax.tree.leaves(cross['models'])),execution=execution)['M1']
            cauchy=sector_moment_cauchy((c1,cm['M1'],t1),sectors,mesh_xy=mesh_xy)
            del c1,t1
            del ct,tc,cm
        models=(sectors[0]['model'],sectors[1]['model'],*cross['models'])
        treatment_policy=recipe.get('sector_pole_treatment')
        treatment=None
        treatment_masks=(models[0][2],models[1][2],models[2][2],models[3][2])
        if treatment_policy is not None:
            from gw.shared_pole_gates import shared_pole_treatment_mask
            cc_mask,cc_row=shared_pole_treatment_mask(
                models[0][1],models[0][2],ceiling_ry=treatment_policy['ceiling_ry'])
            tt_mask,tt_row=shared_pole_treatment_mask(
                models[1][1],models[1][2],ceiling_ry=treatment_policy['ceiling_ry'])
            ct_common=(jnp.all(models[2][1]==models[3][1],axis=-1)
                       &jnp.all(models[2][2]==models[3][2],axis=-1))
            ct_mask,ct_row=shared_pole_treatment_mask(
                models[2][1],models[2][2],ceiling_ry=treatment_policy['ceiling_ry'])
            treatment=dict(CC=dict(cc_row,common_census=jnp.ones_like(ct_common)),
                           TT=dict(tt_row,common_census=jnp.ones_like(ct_common)),
                           CT=dict(ct_row,common_census=ct_common))
            treatment_masks=(cc_mask,tt_mask,ct_mask,ct_mask)
            for name in treatment:
                if not bool(jnp.all(treatment[name]['common_census'][:real])):
                    raise ValueError(f'GATE shared_pole_sector_treatment_census: sector={name}')
                if not bool(jnp.all(treatment[name]['active_prefix'][:real])):
                    raise ValueError(f'GATE shared_pole_sector_treatment_order: sector={name}')
                if not bool(jnp.all(treatment[name]['retained_nonempty'][:real])):
                    raise ValueError(
                        f'GATE shared_pole_sector_treatment_empty: sector={name}')
        signed=(tuple((s['signed'][0],s['signed'][0],*s['signed'][1:]) for s in sectors)
                +(cross['signed'],))
        budget=cross['budget']
        budget.retained_panels=tuple(jax.tree.leaves((models,signed,treatment_masks)))
        budget.live(())
        # Each held tile is read, scored, and released before the next support.
        held_rows={name:[] for name in ('CC','TT','CT')}
        with timing.section('spole.sector.held', announce=True):
            for name,endpoint_pair,model in zip(held_rows,((0,0),(1,1),(0,1)),signed):
                with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                    for sample_id in recipe['held_ids']:
                        sample_id=int(sample_id)
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
        treatment_receipt=None
        if treatment is not None:
            treatment_receipt=dict(
                policy=dict(treatment_policy),
                scope='stored positive-pole models; held rows below score the pre-treatment signed fit',
                sigma_accuracy='NOT_MEASURED_BY_CONSTRUCTOR',
                sectors={name:{key:np.asarray(gather_to_host(value))[:real].tolist()
                    for key,value in values.items()}
                    for name,values in treatment.items()})
        row_receipt=dict(parents=ids[:real],execution=execution,
            mesh_shape=dict(mesh_xy.shape),held=held_rows,
            held_scope='pre-treatment signed fit; not stored-model or Sigma accuracy',
            pole_treatment=treatment_receipt,
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
        with timing.section('spole.sector.write', announce=True):
            ct_host_census=None
            for name,family,model,active_mask in zip(
                    ('CC','TT','CT_C','CT_T'),(0,1,0,1),models,treatment_masks):
                ambient=ledger.live_stages
                if treatment_policy is None:
                    treated_factor,treated_poles=model[:2]
                else:
                    local_factor_bytes=model[0].size*model[0].dtype.itemsize//mesh_xy.size
                    local_pole_bytes=model[1].size*model[1].dtype.itemsize//mesh_xy.size
                    treatment_stage=ledger.reserve(
                        f'sector.treatment.{name}.{ids[0]}',
                        resident_bytes_per_rank=local_factor_bytes+local_pole_bytes,
                        workspace_bytes_per_rank=local_factor_bytes+local_pole_bytes,
                        concurrent_with=ambient)
                    ledger.live_stages=(*ambient,treatment_stage['stage'])
                    try:
                        treated_factor=jnp.where(active_mask[:,None,:],model[0],0)
                        treated_poles=jnp.where(active_mask,model[1],1)
                    except Exception:
                        ledger.live_stages=ambient
                        raise
                try:
                    # CT_C and CT_T have one physical eigenproblem. Reuse the
                    # exact host pole bytes and mask at the writer boundary;
                    # their endpoint factors still travel independently.
                    census=_host_sector_census(treated_poles,active_mask,mesh_xy,real,
                        common=ct_host_census if name=='CT_T' else None)
                    if name=='CT_C':
                        ct_host_census=census
                    poles,active,counts,width=census
                    factor=face_rows(mesh_xy,tuple(range(real)),width)(treated_factor if is_face(treated_factor) else to_face(treated_factor))
                    for q0,q1,slots in _contiguous_q_spans(ids,real):
                        public=canonical_factors(mesh_xy,slots,components=3 if family else 1)(factor)
                        filename=root/(name+'.h5') if sector_models is None else sector_models[name]
                        store_header=write_shared_pole_model(filename,public,
                            device_put_process_local(poles[list(slots),:width],NamedSharding(mesh_xy,P())),
                            counts[list(slots)],q_span=(q0,q1),meta=meta,tables=bank['sector_tables'][family],
                            recipe=recipe,receipts=dict(identity=bank['identity'],constructor=row_receipt),
                            ordered=True,basis=bank['mu_bases'][family],sector=name)
                        stores[name]=(str(filename) if sector_models is None else filename,store_header)
                        del public
                    del factor,treated_factor,treated_poles
                finally:
                    if treatment_policy is not None:
                        ledger.live_stages=ambient
        placed.extend(ids[:real])
        budget.retained_panels=()
        ledger.live_stages=upstream
        del sectors,cross,models,signed,retained,model
    if sorted(placed)!=list(range(header['n_q_irr'])):
        raise ValueError('GATE shared_pole_sector_rounds: each parent must be written once')
    handle=write_shared_pole_sector_manifest(root/'sectors.json',models=stores,bank=bank,
        identity=bank['identity'],receipts=dict(rounds=receipts,status='CONSTRUCTED',
            acceptance='signed-retained-H-v1',sigma_accuracy='NOT_MEASURED'),mesh_xy=mesh_xy)
    if sector_models is not None:
        # Sigma keeps this stage live while it reads the models, then releases both.
        handle['model_stage']=model_residence['stage']
    return dict(handle=handle,identity=bank['identity'],status='CONSTRUCTED',
                q_receipts=receipts,capacity=ledger.receipt(),
                execution=execution_rows,model_residence=model_residence)


def _sector_model_residence(meta,config,header,mu_bases,execution_rows,*,mesh_xy,root,
                            upstream,route):
    """Keep this map's four sector models on the devices for Sigma when they fit.

    The constructor writes CC, TT, CT_C and CT_T once and Sigma reads each once
    in the same map; the files only carry them across that boundary. Under the
    bank's admission rule they stay resident when R, their bytes at the
    retained-pole bound (``signed_side_bound``; CT at the sum of both), and
    one copy fit in half the device budget and the constructor keeps its route
    with R live. Otherwise the files are written as before.
    Returns ``(models or None, receipt)``; a resident R is reserved as
    ``receipt["stage"]``.
    """
    from file_io.shared_pole_store import ResidentSectorModel
    from runtime.padding import combined_divisor, padded_axis
    ledger=meta.shared_pole_capacity
    nq=int(header['n_q_irr'])
    divisor=combined_divisor(mesh_xy.shape['x'],mesh_xy.shape['y'])
    bound=[min(row['signed_side_bound'],row['conservative_pencil_side']) for row in execution_rows]
    rows=dict(CC=(mu_bases[0].n_canonical,bound[0]),TT=(3*mu_bases[1].n_canonical,bound[1]),
              CT_C=(mu_bases[0].n_canonical,sum(bound)),CT_T=(3*mu_bases[1].n_canonical,sum(bound)))
    R=sum(16*nq*n*padded_axis(k,divisor,name="shared_pole_sector_K").carrier//int(mesh_xy.size)+8*nq*k for n,k in rows.values())
    receipt=dict(residence='file',payload_bytes_per_rank=int(R))
    both=ledger.preview(resident_bytes_per_rank=2*R,workspace_bytes_per_rank=0,
                        concurrent_with=upstream)
    receipt['half_budget_bytes_per_rank']=both['available_device_bytes_per_rank']//2
    if both['aggregate_bytes_per_rank']>receipt['half_budget_bytes_per_rank']:
        receipt['reason']='models and one copy exceed half the device budget'
        return None,receipt
    stage=f"sector_models.{header['identity']['iteration_id']}"
    ledger.reserve(stage,resident_bytes_per_rank=R,workspace_bytes_per_rank=0,
                   concurrent_with=upstream)
    if sector_execution(meta,config,mu_bases,nq,mesh_xy=mesh_xy,
                        upstream=(*upstream,stage))[0]!=route:
        receipt['reason']='constructor would change route with the models live'
        return None,receipt
    receipt.update(residence='device',stage=stage,
                   reason='models, one copy and the unchanged constructor route fit')
    return {name:ResidentSectorModel(mesh_xy,label=str(root/(name+'.h5'))) for name in rows},receipt


def _host_sector_census(poles,mask,mesh_xy,real,*,common=None):
    """Materialize one writer census, or reuse the CT_C bytes for CT_T."""
    if common is not None:
        return common
    import jax
    import numpy as np
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from runtime.padding import ladder_extent, padded_axis

    poles,active=jax.tree.map(lambda a:np.asarray(device_put_process_local(
        a,NamedSharding(mesh_xy,P()))),(poles,mask))
    counts=active.sum(axis=-1,dtype=np.int64)
    # The writer carrier sits on the extent ladder, as the scalar
    # constructor's export does: K moves every SC map, and every program keyed
    # by this width (the store's factor check, the face and canonical
    # handoffs) then repeats across maps. The file keeps K.max columns.
    width=padded_axis(ladder_extent(int(counts[:real].max()),poles.shape[-1]),
        mesh_xy,name='shared_pole_port',
        specs=((P('x','y'),0),(P('x','y'),1))).carrier
    # The factor carrier is Py aligned; only inactive pole columns are added.
    if poles.shape[-1] < width:
        poles=np.pad(poles,((0,0),(0,width-poles.shape[-1])),constant_values=1.0)
    return poles,active,counts,width


def sector_held_errors(signed, samples, z, *, mesh_xy):
    """Relative W and dW/ds diagnostics at one held physical complex z.

    Signed endpoints are (C_L,C_R,mu,active), parent-sharded; sample tiles
    have one support. Only the resulting two scalars per parent replicate.
    """
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_execution import is_face,held_program
    if is_face(signed[0]):
        return held_program(mesh_xy)(*signed,samples['Wc'],samples['dWc_ds'],jnp.asarray(z))
    value=_local_held_program(mesh_xy)(*signed,samples['Wc'],samples['dWc_ds'],jnp.asarray(z))
    return device_put_process_local(value,NamedSharding(mesh_xy,P()))


@lru_cache(maxsize=None)
def _local_held_program(mesh):
    from functools import partial
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm
    spec=P(('x','y'))
    return jax.jit(shard_map(partial(_sector_held_equations,mm=_mm),mesh=mesh,
        in_specs=(spec,)*6+(P(),),out_specs=spec,check_vma=False))


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
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_execution import is_face,cauchy_program
    if is_face(metrics[0]):
        return cauchy_program(
            mesh_xy, metrics[0].shape[-1], metrics[2].shape[-1])(*metrics)
    plans=[s['budget'].eigenplan(m.shape[-1]).native_fn
           for s,m in zip(sectors,(metrics[0],metrics[2]))]
    result=_local_cauchy_program(mesh_xy,*plans)(*metrics)
    return jax.tree.map(lambda a:device_put_process_local(a,NamedSharding(mesh_xy,P())),result)


@lru_cache(maxsize=None)
def _local_cauchy_program(mesh,charge_eigh,current_eigh):
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    def body(c,ct,t):
        return sector_cauchy_schwarz((c,ct,t),eigh_charge=charge_eigh,
                                     eigh_current=current_eigh,matmul=_mm,gates=gates)
    spec=P(('x','y'))
    return jax.jit(shard_map(body,mesh=mesh,in_specs=(spec,)*3,
                            out_specs=spec,check_vma=False))


def sector_recipe(recipe, n):
    """The map recipe resized to one sector of ``n`` rows: widths, line cap and pole budget."""
    import math
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b
    policy=shared_real_pole_v1_r3b[recipe['accuracy']]
    result=dict(recipe,n=n)
    for field in ('imaginary_width','infinity_width','line_direction_cap','pole_budget'):
        fraction=policy.get(field+'_fraction')
        result[field]=None if fraction is None else math.ceil(n*fraction)
    return result


def construct_diagonal_sector_round(samples, moments, meta, config, geometry, *, mesh_xy,
                                     retained=(), line=None):
    """Run the production selection/reduction program for CC or TT.

    Samples and M0..M3 are parent-local stacks, [P,S,n,n] and [P,n,n], the
    samples at the dense fitted ids ``geometry['sample_ids']``; ``line`` maps
    each line-panel sample to its stored (panels, counts) for this sector.
    ``geometry`` carries the endpoint basis, header, round ids/real and the
    sector name. TT uses n=3*n_T with mu-major Cartesian rows. The unchanged
    recipe fractions set its widths and 1.8*n pole budget. Capacity remains
    the whole-map ledger; no fictitious independent sector allowance is created.

    Returns the sorted positive model, signed model, original-pencil span,
    selected state/infinity panels and replicated diagnostics. The latter
    are needed by the CT joint projection in the same map, never cached.
    """
    import copy
    import jax
    import numpy as np
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,round_padding_output_bytes
    from gw.shared_pole_directions import (_round_kernels,line_panel_states,port_extent,
                                           select_round_states,leading_response_directions)
    from gw.shared_pole_local import round_tables,reduce_round,grow_round,carrier_history

    from gw.shared_pole_execution import is_face, face_reduce_round
    execution='face' if is_face(samples['Wc']) else 'local'
    components=int(geometry['components'])
    basis=geometry['basis']
    n=components*basis.n_logical
    local_meta=copy.copy(meta)
    local_meta.mu_basis=basis
    local_meta.n_rmu=n
    local_meta.n_rmu_padded=components*basis.n_packed
    recipe=sector_recipe(meta.shared_pole_recipe,n)
    budget=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
                               mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,
                               upstream=meta.shared_pole_capacity.live_stages,
                               execution=execution)
    budget.batch_width=len(geometry['ids'])
    budget.retained_panels=tuple(retained)
    # The dense sample and moment fields and the stored line panels are
    # resident during selection. Derive the face count from these exact read
    # dictionaries so admission prices the live panel set.
    line={} if line is None else line
    panel_elements=sum(int(np.prod(panels.shape[1:])) for panels,_ in line.values())
    selection_faces=(sum(int(panel.shape[1]) for panel in samples.values())
                     +len(moments)+-(-panel_elements//local_meta.n_rmu_padded**2))
    budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                selection_faces=selection_faces)
    eig=budget.eigenplan(local_meta.n_rmu_padded)
    svd=budget.eigenplan(2*local_meta.n_rmu_padded)
    extent=port_extent(mesh_xy)
    qi,values=leading_response_directions(moments['M1'],min(n,recipe['infinity_width']),
        eigh_plan=eig,column_extent=extent,multiplet_tol=recipe['multiplet_relative_tolerance'],
        real_rows=None if execution=='face' else geometry['real'])
    kernels=_round_kernels(mesh_xy,'face' if execution=='face' else 'batch')
    infinity=(qi,*(kernels.apply(moments[name],qi) for name in ('M0','M1','M2','M3')))
    # Synthetic local slots select nothing, as a dense selection's do.
    real=geometry['real']
    line_states={sid:line_panel_states(panels,np.where(np.arange(len(counts))<real,counts,0),
                                       recipe,sid=sid,ordered=True,mesh_xy=mesh_xy)
                 for sid,(panels,counts) in line.items()}
    states,counts,roles=select_round_states(samples,recipe,sample_ids=geometry['sample_ids'],
        real=real,mesh_xy=mesh_xy,eigh_plan=eig,svd_plan=svd,column_extent=extent,
        logical_n=n,ordered=True,line_states=line_states)
    del line_states
    # Reuse admitted carrier widths across this model's later SC maps.
    round_key=('sector',geometry['sector'],n)
    # The reduction envelope already includes current Q/O/dO and infinity
    # panels. Full sample/moment stacks remain caller-live through this call,
    # so those and earlier-sector outputs are the only additional arrays.
    budget.retained_panels=(*retained,*samples.values(),*moments.values(),
                            *(panels for panels,_ in line.values()))
    history=carrier_history(meta)
    def _tables(widths,infinity_width,reuse):
        return round_tables(counts,widths,[s[0] for s in states],
            [v.shape[-1] for v in values],infinity_width,column_extent=extent,
            ordered=True,odd_moments=True,key=round_key if reuse else None,
            history=history if reuse else None)
    def _preview(widths,infinity_width,reuse):
        table=_tables(widths,infinity_width,reuse)
        padding_bytes=round_padding_output_bytes(states,infinity,widths,infinity_width)
        return budget.preview(table['active'].shape[-1],phase='reduction',
            padding_output_bytes_per_rank=padding_bytes)[
            'device_budget_status']=='PASS'
    def _admit(widths,infinity_width,reuse):
        table=_tables(widths,infinity_width,reuse)
        side=table['active'].shape[-1]
        padding_bytes=round_padding_output_bytes(states,infinity,widths,infinity_width)
        budget.plan(side,phase='reduction',padding_output_bytes_per_rank=padding_bytes)
        if reuse:
            history[(round_key,'extent',2,len(widths))]=(table['order'].shape[1]//2,)
        return table,side
    states,infinity,(tables,side)=grow_round(
        round_key,states,infinity,history=history,preview=_preview,admit=_admit)
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    gram_keep = shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']
    if execution == 'face':
        reduced=face_reduce_round(states,infinity,tables,real=geometry['real'],mesh=mesh_xy,
            budget=budget,ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True,
            gram_keep=gram_keep,admit=False)
    else:
        reduced=reduce_round(states,infinity,tables,real=geometry['real'],mesh_xy=mesh_xy,
            native_eigh=budget.eigenplan(side).native_fn,ordered=True,odd_moments=True,
            keep_budget=recipe['pole_budget'],retain_span=True,gram_keep=gram_keep)
    model,signed,vectors,diagnostics,y=reduced
    reduction,zero,_,_=jax.tree.map(np.asarray,diagnostics)
    for name in ('orientation_paired','gram_diagonal_positive','gram_valid','retained_metric_positive'):
        if not np.all(reduction[name][:geometry['real']]):
            raise ValueError(f"GATE shared_pole_sector_{name}: sector={geometry['sector']}, "
                             f"parents={geometry['ids'][:geometry['real']]}, "
                             f"passed={reduction[name][:geometry['real']].tolist()}, "
                             f"Gram min/max={reduction['gram_min_relative'][:geometry['real']].tolist()}, "
                             f"paired H_r min/max={reduction['paired_min_relative'][:geometry['real']].tolist()}; no repair")
    if not np.all(zero['zero_policy'][:geometry['real']]):
        raise ValueError(f"GATE shared_pole_sector_zero_ritz: sector={geometry['sector']}")
    # The returned planner must not retain the just-consumed full sample and
    # moment arrays through its accounting view after the caller releases them.
    budget.retained_panels=tuple(retained)
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                recipe=recipe,budget=budget,execution=execution)


def construct_cross_sector_round(sectors, samples, moments, meta, config, *,
                                 mesh_xy, sample_ids, line_cross, real):
    """Run CT on the two current-map diagonal spans, keeping both outputs.

    ``samples=(CT,TC)`` contains the native rectangular Wc/dWc_ds rounds at
    the dense fitted ``sample_ids``; ``line_cross[family]`` maps each line-panel
    sample to that family's stored cross panel (``read_line_panels(cross=True)``);
    moments is the CT M0..M3 round. All operators are parent-sharded. The
    signed physical photon interaction is admitted by the unchanged positive
    retained-H checks, not by the scalar positive-V upper passivity bound.
    """
    import copy
    import jax
    import numpy as np
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity
    from gw.shared_pole_local import _batch_put
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates

    from gw.shared_pole_execution import is_face
    execution="face" if is_face(samples[0]["Wc"]) else "local"
    charge, transverse = sectors
    ct, tc = samples
    retained = jax.tree.leaves(tuple((s['model'],s['signed'],s['coefficients'],
        s['infinity'],tuple(state[1:] for state in s['states'])) for s in sectors))
    local_meta = copy.copy(meta)
    local_meta.n_rmu_padded = sum(s['model'][0].shape[-2] for s in sectors)
    budget = ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
        mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,
        upstream=meta.shared_pole_capacity.live_stages,execution=execution)
    budget.batch_width = charge['model'][0].shape[0]
    budget.retained_panels = (*retained,*ct.values(),*tc.values(),*moments.values(),
        *(panels for stored in line_cross for panels,_ in stored.values()))
    # Cross assembly has rectangular original pencils; only the projected
    # retained pair is square. Keep those two extents distinct in the ledger.
    original_sides = tuple(s['coefficients'].shape[-2] for s in sectors)
    # The compacted span is the round's largest retained count on the extent
    # ladder.  A wider historical span is optional but the second cross
    # capacity admission includes actions that do not exist yet at this
    # point, so only this round's admitted span is used.
    from runtime.padding import ladder_extent
    widths = [min(s['signed'][2].shape[-1],
                  ladder_extent(int(jnp.max(jnp.sum(s['signed'][2],axis=-1)))))
              for s in sectors]
    if execution == 'face':
        from runtime.padding import padded_axis
        widths = [padded_axis(width,mesh_xy,name='shared_pole_port',
            specs=((P('x','y'),0),(P('x','y'),1))).carrier for width in widths]
    side = sum(widths)
    budget.plan(side,phase='cross_reduction',cross_original_sides=original_sides)
    actions=[]
    for source, forward, reverse, stored in ((charge,tc,ct,line_cross[0]),(transverse,ct,tc,line_cross[1])):
        panels=(forward['Wc'],reverse['Wc'],forward['dWc_ds'],reverse['dWc_ds'])
        actions.append(cross_round_actions(panels,source['states'],source['roles'],
            source['recipe'],sample_ids=sample_ids,mesh_xy=mesh_xy,line_cross=stored))

    spec=P(('x','y'))
    packed=[]
    for sector,width in zip(sectors,widths):
        # Drop only exactly inactive carrier columns. This is a storage
        # compaction of the retained span, not a second physical rank cut.
        if execution=='face':
            from gw.shared_pole_execution import compact_program
            compact=compact_program(mesh_xy,width)
        else:
            compact=_local_compact_program(mesh_xy,width)
        y,signed=compact(sector['coefficients'],sector['signed'])
        # Host role coordinates/order are replicated metadata, not matrices.
        put=(lambda a:jax.make_array_from_callback(a.shape,NamedSharding(mesh_xy,P()),
                                                   lambda index:a[index])) if execution=='face' else (lambda a:_batch_put(mesh_xy,a))
        packed.append((put(sector['tables']['points']),
            put(sector['tables']['order']),
            (tuple(s[1] for s in sector['states']),tuple(s[2] for s in sector['states'])),
            sector['infinity'],y,signed))
    side=sum(s[4].shape[-1] for s in packed)
    budget.retained_panels=(*budget.retained_panels,*jax.tree.leaves((actions,packed)))
    budget.plan(side,phase='cross_reduction',cross_original_sides=original_sides)
    cross_eigh=budget.eigenplan(side)
    signed,diagnostics=reduce_cross_round(*packed,tuple(actions),
        tuple(moments[f'M{i}'] for i in range(4)),mesh_xy=mesh_xy,
        eigh_plan=cross_eigh)
    for name in ('gram_valid','retained_metric_positive'):
        if not bool(jnp.all(diagnostics[name][:real])):
            raise ValueError(f'GATE shared_pole_sector_{name}: sector=CT; '
                f"Gram min/max={float(jnp.min(diagnostics['gram_min_relative'][:real])):.9e}; "
                f"threshold={gates['normalized_gram_validity']['threshold']}; no repair")
    models,zero=positive_cross_models(signed,mesh_xy=mesh_xy)
    if not bool(jnp.all(zero['zero_policy'][:real])):
        raise ValueError('GATE shared_pole_sector_zero_ritz: sector=CT')
    budget.retained_panels=tuple(retained)
    replicated=NamedSharding(mesh_xy,P())
    return dict(models=models,signed=signed,
                diagnostics=jax.tree.map(lambda a:device_put_process_local(a,replicated),diagnostics),
                zero=jax.tree.map(lambda a:device_put_process_local(a,replicated),zero),budget=budget)


@lru_cache(maxsize=None)
def _local_compact_program(mesh,width):
    from functools import partial
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    spec=P(('x','y'))
    return jax.jit(shard_map(partial(_compact_sector_equations,width=width),mesh=mesh,
        in_specs=(spec,spec),out_specs=(spec,spec),check_vma=False))


def _compact_sector_equations(y,signed,*,width,matrix_sharding=None):
    c,mu,active=signed
    order=jnp.argsort(~active,axis=-1,stable=True)[:,:width]
    return (_matrix_take_columns(y,order,matrix_sharding),
        (_matrix_take_columns(c,order,matrix_sharding),
         jnp.take_along_axis(mu,order,axis=-1),
         jnp.take_along_axis(active,order,axis=-1)))


def positive_cross_models(signed, *, mesh_xy):
    """Positive-pole CT endpoint models with the same ordering and zero mask.

    Finite/infinite and low-pole dropped weight are checked independently
    on each physical endpoint; a large charge norm cannot hide a lost current
    factor. The signed model remains available for held-frequency checks.
    """
    from gw.shared_pole_execution import is_face,positive_cross_program
    if is_face(signed[0]):
        return positive_cross_program(mesh_xy)(*signed)
    return _local_positive_cross_program(mesh_xy)(*signed)


@lru_cache(maxsize=None)
def _local_positive_cross_program(mesh):
    from functools import partial
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    spec=P(('x','y'))
    return jax.jit(shard_map(partial(_positive_cross_equations,gates=gates),mesh=mesh,
        in_specs=(spec,)*4,out_specs=(spec,spec),check_vma=False))


def _positive_cross_equations(left,right,mu,active,*,gates,matrix_sharding=None):
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
    charge,order=sort_shared_pole_columns(models[0],matrix_sharding=matrix_sharding)
    current=(_matrix_take_columns(models[1][0],order,matrix_sharding),charge[1],charge[2])
    return (charge,current),dict(zero_policy=same & checks[0]['zero_policy'] & checks[1]['zero_policy'],
                              charge=checks[0],current=checks[1])


def _cross_products(panels,q,node,sample,*,mirror,imaginary,conjugate,mm):
    """One CT action with the parent's own operator, for both execution layouts.

    ``panels`` are the forward and reverse rectangles and their derivatives,
    then (a mirror state off the imaginary axis only) the same four of the
    minus-q partner W_q(-conj z); ``sample`` indexes their stacks. A mirror
    state acts with W(-conj z): the partner panels or, at an imaginary node
    where -conj z = z, the sample itself.
    """
    w,wr,d,dr,wm,wrm,dwm,dwrm=(*panels,None,None,None,None)[:8]
    if mirror and not imaginary:
        a,da=(wm,dwm) if conjugate else (wrm,dwrm)
        adjoint=not conjugate
    elif mirror:
        a,da=(w,d) if conjugate else (wr,dr)
        adjoint=not conjugate
    else:
        a,da=(wr,dr) if conjugate else (w,d)
        adjoint=conjugate
    a,da=a[:,sample],da[:,sample]
    if adjoint:
        a,da=jnp.swapaxes(jnp.conj(a),-1,-2),jnp.swapaxes(jnp.conj(da),-1,-2)
    return mm(a,q),mm(da,q)*(2*node)


def cross_round_actions(samples, states, roles, recipe, *, sample_ids, mesh_xy, line_cross):
    """Apply rectangular samples to the diagonal sectors' selected directions.

    ``samples=(W_LR,W_RL,dW_LR/ds,dW_RL/ds)`` are parent-local
    [P,S,n_L,n_R] at the dense fitted ``sample_ids`` (reverse blocks have
    reversed endpoint extents). ``states``/``roles`` are the diagonal round's
    source states. A state of a line-panel sample takes its stored cross
    output and action (``line_cross[sid]``: (panels [.., 2S, n_L, r], counts),
    the producer's ``sector_line_panels``); a dense sample's are formed here.
    Outputs follow the paired state order and the derivative is d/dz, report
    equation 5.3.
    """
    import numpy as np
    from gw.shared_pole_directions import _round_kernels, _sample_point, replicated
    from gw.shared_pole_execution import is_face,cross_action_program
    from gw.shared_pole_local import _pad_columns
    face=is_face(samples[0])
    k=_round_kernels(mesh_xy,'face' if face else 'batch')
    index={int(sid):i for i,sid in enumerate(sample_ids)}
    put=replicated(mesh_xy)
    stored={}
    outputs=[]
    for state,role in zip(states,roles[0]):
        sid=int(role['sample_id'])
        conjugate=bool(role.get('conjugate',False))
        mirror=bool(role.get('mirror',False))
        if sid in line_cross:
            if sid not in stored:
                panels=line_cross[sid][0]
                stored[sid]=[k.column(panels,put(np.int32(i))) for i in range(int(panels.shape[1]))]
            # The round padded the state's direction to its carrier
            # (grow_round); its padding columns act as zero.
            width=int(state[1].shape[-1])
            field=2*(2*mirror+conjugate)
            outputs.append(tuple(a if int(a.shape[-1])==width else
                                 _pad_columns(a.sharding,a.shape,width)(a)
                                 for a in stored[sid][field:field+2]))
            continue
        imaginary=_sample_point(recipe,sid).real==0
        if not imaginary:
            raise ValueError(f'GATE shared_pole_line_panel: line sample {sid} has no stored cross panel')
        node=jnp.asarray(state[0])
        sample=jnp.asarray(index[sid],jnp.int32)
        if face:
            outputs.append(cross_action_program(mesh_xy,mirror,imaginary,conjugate)(samples,state[1],node,sample))
        else:
            outputs.append(_local_cross_action_program(mesh_xy,mirror,imaginary,conjugate)(samples,state[1],node,sample))
    return tuple(outputs)


@lru_cache(maxsize=None)
def _local_cross_action_program(mesh,mirror,imaginary,conjugate):
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm
    def apply(panels,q,node,sample):
        return _cross_products(panels,q,node,sample,mirror=mirror,
            imaginary=imaginary,conjugate=conjugate,mm=_mm)
    batch=P(('x','y'))
    return jax.jit(shard_map(apply,mesh=mesh,in_specs=(batch,batch,P(),P()),
                            out_specs=(batch,batch),check_vma=False))


def reduce_cross_round(charge, transverse, cross, moments, *, mesh_xy, eigh_plan):
    """Construct CT on the two retained original-pencil spans.

    Each sector tuple contains (points, order, states, infinity, Y, signed),
    where ``states`` is (Q tuple, WQ tuple), ``signed`` is (c,mu,active),
    and all operands carry the round's leading parent sharding. ``cross``
    contains the TC-on-C and CT-on-T (output, derivative) panel tuples;
    ``moments`` is M0_CT..M3_CT. No full operator leaves the admitted
    local rounds remain parent-local; a face round keeps one physical parent
    over the complete mesh. ``eigh_plan`` is the already-resolved public
    service plan matching that layout. Returns two signed CT endpoint factors,
    inverse poles, active columns and the unchanged joint-metric diagnostics.
    """
    from gw.shared_pole_execution import is_face,cross_parent_program
    if is_face(charge[4]):
        side=charge[4].shape[-1]+transverse[4].shape[-1]
        return cross_parent_program(mesh_xy,side)(charge,transverse,cross,moments)
    return _local_cross_parent_program(mesh_xy,eigh_plan.native_fn)(charge,transverse,cross,moments)


@lru_cache(maxsize=None)
def _local_cross_parent_program(mesh,native_eigh):
    from functools import partial
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm, zero_row_safe_eigh
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    spec=P(('x','y'))
    body=partial(_cross_reduce_equations,mm=_mm,eigh=zero_row_safe_eigh(native_eigh),gates=gates)
    return jax.jit(shard_map(body,mesh=mesh,in_specs=(spec,)*4,
                            out_specs=(spec,spec),check_vma=False))


def _cross_reduce_equations(charge,transverse,cross,moments,*,mm,eigh,gates,matrix_sharding=None):
    join = lambda arrays, axis: _matrix_concat(arrays, axis, matrix_sharding)
    def pack(panels,order):
        pad=1 if matrix_sharding is None else int(matrix_sharding.mesh.shape['y'])
        values=join((*panels,jnp.zeros_like(panels[0][...,:pad])),axis=-1)
        return _matrix_take_columns(values,order,matrix_sharding)
    def unpack(sector,actions):
        points,order,states,infinity,y,signed=sector
        return points,pack(states[0],order),infinity[0],pack(tuple(a[0] for a in actions),order),pack(tuple(a[1] for a in actions),order)
    zc,qc,ic,tc,dtc=unpack(charge,cross[0])
    zt,qt,it,ct,dct=unpack(transverse,cross[1])
    g,h,otc,oct=ordered_cross_pencil((zc,qc,ic),(zt,qt,it),(tc,ct,dct),moments,matmul=mm,matrix_sharding=matrix_sharding)
    # The diagonal output panels are O_original. Reconstruct their
    # infinity columns from the same physical moments already in hand.
    def diagonal(sector):
        _,order,states,infinity,y,signed=sector
        own=pack(states[1],order)
        return _matrix_layout(join((own,2*infinity[1],2*infinity[2]),axis=-1), matrix_sharding)
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

    join = lambda arrays, axis: _matrix_concat(arrays, axis, matrix_sharding)
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
    block = lambda a, b, c, d: join((
        join((a, b), axis=-1), join((c, d), axis=-1)), axis=-2)
    g = block(g, join((top0, top1), axis=-1),
              join((bottom0, bottom1), axis=-2), block(p[0], p[1], p[1], p[2]))
    h = block(h, join((top1, top2), axis=-1),
              join((bottom1, bottom2), axis=-2), block(p[1], p[2], p[2], p[3]))
    return tuple(_matrix_layout(a, matrix_sharding) for a in
                 (g, h, join((tc, mc[0], mc[1]), axis=-1),
                  join((ct, mt[0], mt[1]), axis=-1)))


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
    join = lambda arrays, axis: _matrix_concat(arrays, axis, matrix_sharding)
    yc, vc, oc, tc = charge
    yt, vt, ot, ct = transverse
    project = lambda a: matmul(yc, matmul(a, yt), transa="C")
    metric, value = map(project, cross)
    adj = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))
    block = lambda a, b, d: join((
        join((a, b), axis=-1),
        join((adj(b), d), axis=-1)), axis=-2)
    # Inactive columns of a batched retained span are exactly zero. Their
    # metric is zero too; assigning them an identity invents latent states.
    ic = jnp.eye(vc.shape[-1], dtype=metric.dtype)[None] * jnp.any(yc != 0, axis=-2)[:, None, :]
    it = jnp.eye(vt.shape[-1], dtype=metric.dtype)[None] * jnp.any(yt != 0, axis=-2)[:, None, :]
    return tuple(_matrix_layout(a, matrix_sharding) for a in (block(ic, metric, it),
            block(ic * vc[:, None, :], value, it * vt[:, None, :]),
            join((matmul(oc, yc), matmul(ct, yt)), axis=-1),
            join((matmul(tc, yc), matmul(ot, yt)), axis=-1)))


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


def sector_execution(meta, config, mu_bases, nq, *, mesh_xy, upstream):
    """The CC/TT/CT constructor layout ('local' or 'face') and its per-sector rows.

    One owner for the route: the constructor and the bank-residence admission
    (which requires that keeping the bank resident does not change it) both
    resolve it against the whole-map ledger with ``upstream`` live.
    """
    import copy
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_execution import constructor_execution, line_panel_count, selection_face_count
    recipe=meta.shared_pole_recipe
    ledger=meta.shared_pole_capacity
    extent=port_extent(mesh_xy)
    execution_rows=[]
    rows=[(1 if family==0 else 3)*basis.n_packed for family,basis in enumerate(mu_bases)]
    for family,basis in enumerate(mu_bases):
        components=3 if family else 1
        local_meta=copy.copy(meta)
        local_meta.mu_basis=basis
        local_meta.n_rmu=components*basis.n_logical
        local_meta.n_rmu_padded=components*basis.n_packed
        local_recipe=sector_recipe(recipe,local_meta.n_rmu)
        # The diagonal selection holds this sector's dense samples, moments
        # and its own line panels; the cross panels are the CT round's.
        faces=selection_face_count(local_recipe,n=local_meta.n_rmu_padded,logical_n=local_meta.n_rmu,
            states=4,rows=rows[family],dense_fields=2,moment_fields=4,column_extent=extent)
        mode,route=constructor_execution(
            local_meta,linalg_resolution({'linalg':config.backend.linalg}),local_recipe,
            mesh=mesh_xy,ledger=ledger,upstream=upstream,ordered=True,
            odd_moments=True,selection_faces=faces,
            sample_batch=len(recipe['fit_ids'])-line_panel_count(recipe),column_extent=extent)
        cap=local_recipe['line_direction_cap']
        execution_rows.append(dict(sector=('CC','TT')[family],mode=mode,
                                   packed_extent=local_meta.n_rmu_padded,
                                   line_width=extent(max(1,local_meta.n_rmu if cap is None else min(cap,local_meta.n_rmu))),
                                   signed_side_bound=extent(2*local_recipe['pole_budget']) if local_recipe['pole_budget'] is not None else route['conservative_pencil_side'],**route))
    # CT retains both diagonal spans and both rectangular sample stacks. The
    # CC/TT admission alone cannot promise that their joint pencil fits one
    # rank. Resolve its conservative route before opening the bank so the
    # complete round uses one layout and the face batch can be priced below.
    joint_meta=copy.copy(meta)
    joint_meta.n_rmu=sum((3 if family else 1)*basis.n_logical
                         for family,basis in enumerate(mu_bases))
    joint_meta.n_rmu_padded=sum(row['packed_extent'] for row in execution_rows)
    joint_recipe=sector_recipe(recipe,joint_meta.n_rmu)
    # Dense CT and TC (W, dW/ds) at the dense fitted samples, the moments and
    # both families' cross panels.
    lines=line_panel_count(recipe)
    cross=lines*8*(rows[1]*execution_rows[0]['line_width']+rows[0]*execution_rows[1]['line_width'])
    joint_faces=4*(len(recipe['fit_ids'])-lines)+4+-(-cross//joint_meta.n_rmu_padded**2)
    joint_mode,joint_route=constructor_execution(
        joint_meta,linalg_resolution({'linalg':config.backend.linalg}),
        joint_recipe,mesh=mesh_xy,ledger=ledger,
        upstream=upstream,ordered=True,odd_moments=True,
        selection_faces=joint_faces,sample_batch=len(recipe['fit_ids'])-lines,parent_count=nq,
        retained_output_families=2,column_extent=extent,
        defer_reduction=True,
        cross_original_sides=tuple(row['conservative_pencil_side'] for row in execution_rows),
        cross_retained_side=sum(min(row['signed_side_bound'],row['conservative_pencil_side'])
                                for row in execution_rows))
    resolved_execution=('face' if joint_mode=='face' or
                        any(row['mode']=='face' for row in execution_rows)
                        else 'local')
    return resolved_execution,execution_rows

