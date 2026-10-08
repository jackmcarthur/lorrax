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


def _contiguous_q_spans(ids, real):
    """Canonical q spans from a (possibly permuted) constructor round.

    A round's contiguous parents are one span: one store write per model and
    round. Its public factor copy is the round's own factor's size, and the
    store admits it against the current map capacity ledger.
    """
    ordered=sorted(range(real),key=lambda slot:ids[slot])
    spans=[]
    for slot in ordered:
        if not spans or ids[slot]!=ids[spans[-1][-1]]+1:
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


def construct_sector_poles(bank, meta, config, *, mesh_xy, output, print_fn=print):
    """Construct current-map CC, TT and joint-span CT models and their stores.

    The bank contains W-W_infinity and its ordered moments. CC uses n_C
    rows, TT uses 3*n_T rows, with the same physical supports and separate
    directions, reductions and 1.8*n budgets. All retained arrays count
    against the existing whole-map capacity ledger.

    The route is decided once per map from the recipe shapes (``sector_route``):
    q-local rounds of P parents, or rounds of R = min(nq, P) parents on the face
    through the staged reduction (``staged_round``). Each round writes its
    parents' models before the next round reads.

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
    from gw.shared_pole_local import batch_to_face,canonical_factors,carrier_history,face_rows,parent_rounds
    from gw.shared_pole_screening import _json
    from gw.shared_pole_directions import _sample_point
    from gw.shared_pole_execution import is_face
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

    def read_samples(io,endpoints,retained,ids,layout):
        # Wc/dWc_ds at the dense fitted samples.
        return read_sector_round(io,meta,bank,header,ids,endpoints,sample_ids=dense_fit,
            fields=('Wc','dWc_ds'),retained=retained,execution=layout)

    def read_line(io,family,ids,layout,cross=False):
        spec=None if layout=='face' else P(('x','y'))
        return {sid:read_line_panels(io,meta=meta,header=header,family=('C','T')[family],sample=sid,
                                     cross=cross,q_ids=ids,partition_spec=spec)
                for sid in range(line_lo,line_hi)}

    def read_diagonal(family,ids,real,layout='face',retained=()):
        # One sector's moments, dense samples and line panels for ``ids``.
        with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
            exact=read_sector_round(io,meta,bank,header,ids,(family,family),
                fields=('M0','M1','M2','M3'),retained=retained,execution=layout)
            refuse_nonfinite_moment(('CC','TT')[family],exact['M1'],real,mesh_xy=mesh_xy)
            samples=read_samples(io,(family,family),(*retained,*exact.values()),ids,layout)
            line=read_line(io,family,ids,layout)
        return samples,exact,line

    def read_cross(ids,real,layout='face',retained=()):
        # CT and TC at the dense samples, the CT moments and both families' cross panels.
        with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
            ct=read_samples(io,(0,1),retained,ids,layout)
            tc=read_samples(io,(1,0),(*retained,*ct.values()),ids,layout)
            cm=read_sector_round(io,meta,bank,header,ids,(0,1),fields=('M0','M1','M2','M3'),
                                  retained=(*retained,*ct.values(),*tc.values()),execution=layout)
            refuse_nonfinite_moment('CT',cm['M1'],real,mesh_xy=mesh_xy)
            line_cross=[read_line(io,family,ids,layout,cross=True) for family in (0,1)]
        return (ct,tc),cm,line_cross

    receipts=[];stores={};placed=[]
    root=Path(output).parent
    to_face=batch_to_face(mesh_xy)
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    execution,execution_rows,route=sector_route(
        meta,config,bank['mu_bases'],header['n_q_irr'],mesh_xy=mesh_xy,upstream=upstream,print_fn=print_fn)
    # The resident models are reserved first, so every round runs beside them.
    sector_models,model_residence=_sector_model_residence(meta,config,header,bank['mu_bases'],
        execution_rows,mesh_xy=mesh_xy,root=root,upstream=upstream,route=execution)
    if sector_models is not None:
        upstream=ledger.live_stages=(*upstream,model_residence['stage'])
    geometry=lambda family:dict(components=3 if family else 1,basis=bank['mu_bases'][family],
        header=sector_headers[family],sample_ids=dense_fit,sector=('CC','TT')[family])

    def diagonal_receipt(name,model,ids,real):
        reduction,zero,_,_=model['diagnostics']
        counts=np.asarray(model['vectors'][1]).sum(axis=-1).tolist()
        receipts.append(dict(sector=name,parents=ids[:real],K=counts[:real],
            gram_min_relative=np.asarray(reduction['gram_min_relative'])[:real].tolist(),
            zero_policy=np.asarray(zero['zero_policy'])[:real].tolist()))
    staged_line=None
    for ids,real,slots in parent_rounds(int(header['n_q_irr']),route['width']):
        if execution=='face':
            sectors,cross=staged_round(read_diagonal,read_cross,ids,real,meta,config,geometry,
                mesh_xy=mesh_xy,sample_ids=dense_fit,receipt=diagonal_receipt,rows=execution_rows)
            line="; ".join(f"{name} side {r['side']}, stages of {r['stage']}, eighs {'/'.join(r['eigh_routes'])}"
                           for name,r in (('TT',sectors[1]['staged']),('CC',sectors[0]['staged']),('CT',cross['staged'])))
            if line!=staged_line:
                staged_line=line
                print_fn(f"Shared-pole sector constructor: round of parents {ids[0]}..{ids[real-1]}: {line}")
        else:
            sectors=[];retained=[]
            for family,name in enumerate(('CC','TT')):
                with timing.section('spole.sector.'+name, announce=True):
                    samples,exact,line=read_diagonal(family,ids,real,'local',tuple(retained))
                    model=construct_diagonal_sector_round(samples,exact,meta,config,
                        dict(geometry(family),ids=ids,real=real),mesh_xy=mesh_xy,retained=retained,line=line)
                    del samples,exact,line
                    sectors.append(model)
                    retained.extend(jax.tree.leaves((model['model'],model['signed'],
                        model['coefficients'],model['infinity'],tuple(s[1:] for s in model['states']))))
                    diagonal_receipt(name,model,ids,real)
            with timing.section('spole.sector.CT', announce=True):
                samples,cm,line_cross=read_cross(ids,real,'local',tuple(retained))
                cross=construct_cross_sector_round(sectors,samples,cm,meta,config,mesh_xy=mesh_xy,
                    sample_ids=dense_fit,line_cross=line_cross,real=real)
                del samples,cm,line_cross,retained
        path=Path(output).with_name('sector_diagonal_receipt.json')
        rank0_transaction(path,stage='sector.diagonal_receipt',
            write=lambda:path.write_text(_json(dict(identity=bank['identity'],
                status='DIAGONAL_SPANS_ONLY',rounds=receipts))+'\n'))
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
        # Every array this round holds beside its upstream: its sector and CT outputs (models,
        # signed factors, vectors, diagnostics) and the treatment masks.
        budget.retained_panels=tuple(a for a in jax.tree.leaves((sectors,cross,models,signed,treatment_masks))
                                     if hasattr(a,'sharding'))
        budget.live(())
        # Each held tile is read, scored, and released before the next support.
        held_rows={name:[] for name in ('CC','TT','CT')}
        with timing.section('spole.sector.held', announce=True):
            held_ids=[int(i) for i in recipe['held_ids']]
            for name,endpoint_pair,model in zip(held_rows,((0,0),(1,1),(0,1)),signed):
                # Every held sample of the sector in one read (one open, one transfer).
                with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                    held=read_sector_round(io,meta,bank,header,ids,endpoint_pair,sample_ids=held_ids,
                                           execution="face" if is_face(model[0]) else "local")
                for j,sample_id in enumerate(held_ids):
                    one={k:v[:,j:j+1] for k,v in held.items()}
                    errors=sector_held_errors(model,one,_sample_point(recipe,sample_id),mesh_xy=mesh_xy)
                    held_rows[name].append(dict(sample_id=sample_id,
                        Wc=np.asarray(errors)[:real,0].tolist(),dWc_ds=np.asarray(errors)[:real,1].tolist()))
                    del one
                del held
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
            scalar_upper_passivity='NOT_APPLICABLE_SIGNED_V',
            stability_scope='retained ordered H; exact full-space stability not established',
            sigma_accuracy='NOT_MEASURED')
        receipts.append(row_receipt)
        # Stage each canonical parent once. One round of face factors is live;
        # no all-parent factor stack or full photon operator is materialized.
        with timing.section('spole.sector.write', announce=True):
            ct_host_census=None
            # One writer carrier per model (CC, TT, CT) held across rounds and
            # maps (held_writer_width); CT_C and CT_T share the CT census.
            writer_history=carrier_history(meta)
            writer_capacity=getattr(meta,'shared_pole_rank_capacity',None)
            writer_events=None if writer_capacity is None else writer_capacity.setdefault('_events',[])
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
                        common=ct_host_census if name=='CT_T' else None,
                        hold=(writer_history,('writer',name[:2]),writer_events))
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
        del sectors,cross,models,signed
    ledger.live_stages=upstream
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


@lru_cache(maxsize=None)
def _finite_parents_program(mesh):
    import jax
    from jax.sharding import NamedSharding,PartitionSpec as P
    return jax.jit(lambda a: jnp.all(jnp.isfinite(a), axis=tuple(range(1, a.ndim))),
                   out_shardings=NamedSharding(mesh,P()))


def refuse_nonfinite_moment(name, m1, real, *, mesh_xy):
    """GATE shared_pole_sector_nonfinite: a sector's M1 spectral moment is finite on every real parent.

    One reduction of the moment the round already holds, replicated, so every
    rank refuses alike.
    """
    import numpy as np
    finite = np.asarray(_finite_parents_program(mesh_xy)(m1))[:int(real)]
    if not finite.all():
        raise ValueError(f'GATE shared_pole_sector_nonfinite: spectral moment M1 of sector {name} '
                         f'is not finite on parent slots {np.flatnonzero(~finite).tolist()}')


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
    from file_io.shared_pole_store import ResidentSectorModel,admit_resident_model
    ledger=meta.shared_pole_capacity
    nq=int(header['n_q_irr'])
    bound=[min(row['signed_side_bound'],row['conservative_pencil_side']) for row in execution_rows]
    rows=dict(CC=(mu_bases[0].n_canonical,bound[0]),TT=(3*mu_bases[1].n_canonical,bound[1]),
              CT_C=(mu_bases[0].n_canonical,sum(bound)),CT_T=(3*mu_bases[1].n_canonical,sum(bound)))
    R=sum(ResidentSectorModel.payload_bytes(mesh_xy,nq,n,k) for n,k in rows.values())
    receipt=admit_resident_model(ledger,R,f"sector_models.{header['identity']['iteration_id']}",upstream)
    if receipt['residence']!='device':
        return None,receipt
    if sector_execution(meta,config,mu_bases,nq,mesh_xy=mesh_xy,
                        upstream=(*upstream,receipt['stage']))[0]!=route:
        receipt.update(residence='file',reason='constructor would change route with the models live')
        del receipt['stage']
        return None,receipt
    receipt['reason']='models, one copy and the unchanged constructor route fit'
    return {name:ResidentSectorModel(mesh_xy,label=str(root/(name+'.h5'))) for name in rows},receipt


def _host_sector_census(poles,mask,mesh_xy,real,*,common=None,hold=None):
    """Materialize one writer census, or reuse the CT_C bytes for CT_T.

    ``hold`` is ``(history, key, events)`` for the held writer carrier
    (:func:`held_writer_width`); without it the carrier is this round's own.
    """
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
    if hold is not None:
        width=held_writer_width(width,int(counts[:real].max()),*hold)
    # The factor carrier is Py aligned; only inactive pole columns are added.
    if poles.shape[-1] < width:
        poles=np.pad(poles,((0,0),(0,width-poles.shape[-1])),constant_values=1.0)
    return poles,active,counts,width


def held_writer_width(live, kmax, history, key, events):
    """The writer carrier of one sector model: the largest one so far, or this round's.

    The live carrier (this round's Kmax on the extent ladder) moves between
    rounds and SC maps, and each new width lowered the face handoff, the
    canonical stack and the store's staging, check and finalize programs
    (Fe 4^3 bispinor: 2 at map 2, 1 at map 3, about 20 per map at maps 0-1).
    ``history[key]`` keeps the largest live carrier of any earlier round or
    map of this model (``carrier_history``); it grows only when a live carrier
    exceeds it, noted in ``events`` (the SC log's list, None outside an SC
    run), and never shrinks. The extra columns are what the ladder pad
    already writes past each parent's K: zero factor columns and unit poles,
    which the store checks and never reads as poles.
    """
    before=int(history.get(key,0))
    if live>before:
        if before and events is not None:
            events.append(f"shared-pole writer carrier ({key[1]}): Kmax {kmax} exceeds the held "
                          f"width {before}; grown to {live}")
        history[key]=live
        return live
    return before


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
    """Run the q-local selection/reduction program for CC or TT (one whole parent per rank).

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
    import numpy as np
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,round_padding_output_bytes
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import reduce_round,recipe_panel_widths,recipe_infinity_width,pad_states
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
                               execution='local')
    budget.batch_width=len(geometry['ids'])
    budget.retained_panels=tuple(retained)
    # Priced as sector_execution's route decision prices the local round.
    budget.ritz_budget=recipe['pole_budget']
    budget.retain_span=True
    # The dense sample and moment fields and the stored line panels are
    # resident during selection. Derive the face count from these exact read
    # dictionaries so admission prices the live panel set.
    line={} if line is None else line
    budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                selection_faces=_selection_faces(samples,moments,line,local_meta.n_rmu_padded))
    extent=port_extent(mesh_xy)
    states,counts,roles,infinity,values=_sector_selection(samples,moments,line,recipe,geometry,n,
        budget.eigenplan(local_meta.n_rmu_padded),mesh_xy=mesh_xy)
    budget.retained_panels=(*retained,*samples.values(),*moments.values(),
                            *(panels for panels,_ in line.values()))
    # Every state panel is padded to its recipe carrier, so the round
    # program's inputs have one shape; the pencil extent is grow-only over
    # this sector's rounds and SC maps (round_tables).
    widths=recipe_panel_widths(roles[0],states,recipe,column_extent=extent,logical_n=n)
    infinity_width=recipe_infinity_width(infinity,recipe,column_extent=extent,logical_n=n)
    tables=_sector_tables(meta,geometry,n,counts,widths,states,values,infinity_width,extent)
    side=int(tables['active'].shape[-1])
    budget.plan(side,phase='reduction',padding_output_bytes_per_rank=round_padding_output_bytes(
        states,infinity,widths,infinity_width))
    states,infinity=pad_states(states,widths,infinity,infinity_width)
    reduced=reduce_round(states,infinity,tables,real=geometry['real'],mesh_xy=mesh_xy,
        native_eigh=budget.eigenplan(side).native_fn,ordered=True,odd_moments=True,
        keep_budget=recipe['pole_budget'],retain_span=True,gram_keep=_SECTOR_GRAM_KEEP())
    model,signed,vectors,diagnostics,y=reduced
    _sector_gates(diagnostics,geometry['sector'],geometry['ids'],geometry['real'])
    # The returned planner must not retain the just-consumed full sample and
    # moment arrays through its accounting view after the caller releases them.
    budget.retained_panels=tuple(retained)
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                recipe=recipe,budget=budget,execution='local')


def _SECTOR_GRAM_KEEP():
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    return shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']


def _selection_faces(samples, moments, line, packed):
    """Resident [n, n] face equivalents of one selection's read dictionaries."""
    import numpy as np
    panel_elements=sum(int(np.prod(panels.shape[1:])) for panels,_ in line.values())
    return (sum(int(panel.shape[1]) for panel in samples.values())+len(moments)
            +-(-panel_elements//int(packed)**2))


def _sector_tables(meta, geometry, n, counts, widths, states, values, infinity_width, extent):
    """A round's column tables at the recipe carriers; the pencil extent is grow-only over this
    sector's rounds and SC maps (``round_tables``: discovered in map 0, held from map 1), and an
    SC map past 0 that still grows it says so in its log."""
    from gw.shared_pole_local import round_tables,carrier_history
    history=carrier_history(meta)
    name=(('sector',geometry['sector'],n),'extent',2,len(states))
    before=int(history.get(name,(0,))[0])
    tables=round_tables(counts,widths,[s[0] for s in states],[v.shape[-1] for v in values],
        infinity_width,column_extent=extent,ordered=True,odd_moments=True,key=name[0],history=history)
    capacity=getattr(meta,'shared_pole_rank_capacity',None)
    if before and capacity is not None and int(history[name][0])>before:
        capacity.setdefault('_events',[]).append(
            f"shared-pole {geometry['sector']} round: a state's selection exceeds its held carrier; "
            f"pencil extent {before} -> {int(history[name][0])}")
    return tables


def _sector_gates(diagnostics, sector, ids, real):
    """Refuse a diagonal sector reduction whose leading ``real`` parents fail a gate (no repair)."""
    import jax
    import numpy as np
    reduction,zero,_,_=jax.tree.map(np.asarray,diagnostics)
    for name in ('orientation_paired','gram_diagonal_positive','gram_valid','retained_metric_positive'):
        if not np.all(reduction[name][:real]):
            raise ValueError(f"GATE shared_pole_sector_{name}: sector={sector}, "
                             f"parents={list(ids)[:real]}, "
                             f"passed={reduction[name][:real].tolist()}, "
                             f"Gram min/max={reduction['gram_min_relative'][:real].tolist()}, "
                             f"paired Schur S min/max={reduction['paired_min_relative'][:real].tolist()}; no repair")
    if not np.all(zero['zero_policy'][:real]):
        bad=np.flatnonzero(~np.asarray(zero['zero_policy'][:real]))
        raise ValueError(f"GATE shared_pole_sector_zero_ritz: sector={sector}, parents={[list(ids)[i] for i in bad]}, "
                         f"dropped weight fraction={zero['dropped_factor_weight_fraction'][bad].tolist()}, "
                         f"dropped={zero['dropped_count'][bad].tolist()}, retained={zero['retained_rank'][bad].tolist()}, "
                         f"factor weight={zero['factor_weight'][bad].tolist()}, "
                         f"infinite weight fraction={reduction['infinite_weight_fraction'][bad].tolist()}")


def _sector_selection(samples, moments, line, recipe, geometry, n, eig, *, mesh_xy):
    """The selection half of a diagonal sector round: infinity directions and the state panels
    of one batch of parents (face stacks or batch layout), the eighs on ``eig``. Returns
    (states, counts, roles, infinity, values); the caller forms the tables and pads the panels."""
    import numpy as np
    from gw.shared_pole_directions import (_round_kernels,line_panel_states,port_extent,
                                           select_round_states,infinity_directions)
    from gw.shared_pole_execution import is_face
    face=is_face(samples['Wc'])
    extent=port_extent(mesh_xy)
    kernels=_round_kernels(mesh_xy,'face' if face else 'batch')
    qi,values=infinity_directions(kernels,moments['M1'],min(n,recipe['infinity_width']),
        eigh_plan=eig,column_extent=extent,multiplet_tol=recipe['multiplet_relative_tolerance'],
        real_rows=None if face else geometry['real'])
    infinity=(qi,*(kernels.apply(moments[name],qi) for name in ('M0','M1','M2','M3')))
    real=geometry['real']
    line_states={sid:line_panel_states(panels,np.where(np.arange(len(counts))<real,counts,0),
                                       recipe,sid=sid,ordered=True,mesh_xy=mesh_xy)
                 for sid,(panels,counts) in line.items()}
    states,counts,roles=select_round_states(samples,recipe,sample_ids=geometry['sample_ids'],
        real=real,mesh_xy=mesh_xy,eigh_plan=eig,column_extent=extent,
        logical_n=n,ordered=True,line_states=line_states)
    del line_states
    return states,counts,roles,infinity,values


def staged_sector(read, ids, real, meta, config, geometry, *, mesh_xy, route):
    """One round's CC or TT on the face: the selection in sub-batches of the fixed tile
    (``read(ids, real)`` returns their samples, moments and line panels), one set of tables
    for the round's ``len(ids)`` slots, and the staged reduction (``face_reduce_decoupled``):
    GEMM stages over the widest halving of the round whose program fits beside its stacks
    (``stage_width``), each eigh once over the round's stack, on route (c) where its shape
    price fits beside its boundary (``staged_eigh``). Returns what the q-local round returns."""
    import copy
    import jax
    import numpy as np
    import distrib_la
    from common import timing
    from contextlib import contextmanager
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,staged_sector_bytes,_shard_bytes,_local_eigenplan
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import recipe_panel_widths,recipe_infinity_width,pad_states,parent_rounds
    from gw.shared_pole_execution import (face_reduce_decoupled,face_ritz_carrier,face_reduction_bytes,
                                          _stack,parent_rows,staged_eigh,stage_width)
    components=int(geometry['components'])
    basis=geometry['basis']
    name=geometry['sector']
    n=components*basis.n_logical
    local_meta=copy.copy(meta)
    local_meta.mu_basis=basis
    local_meta.n_rmu=n
    local_meta.n_rmu_padded=components*basis.n_packed
    recipe=sector_recipe(meta.shared_pole_recipe,n)
    ledger=meta.shared_pole_capacity
    ambient=ledger.live_stages
    budget=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
                               mesh_xy=mesh_xy,ledger=ledger,upstream=ambient,execution='face')
    extent=port_extent(mesh_xy)
    packed=int(local_meta.n_rmu_padded)
    width=len(ids)
    tile=route[name]
    budget.batch_width=int(tile['selection'])
    # The selection eighs: one [tile x dense samples] stack per kind, beside the tile's inputs and
    # every selected panel of the round at the recipe bound.
    stack=(int(tile['selection'])*max(1,int(tile['dense'])),packed,packed)
    eig=staged_eigh(mesh_xy,stack,int(tile['selection'])*int(tile['unit'])+int(tile['panels']),
                    ledger=ledger,live=ambient,label=f'{name} selection')
    eigh_bytes=distrib_la.eigh_stack_bytes(eig,stack,np.complex128)
    parts=[]
    with timing.section('decoupled.selection'):
        for sub,sub_real,_ in parent_rounds(ids,int(tile['selection'])):
            samples,moments,line=read(sub,sub_real)
            line={} if line is None else line
            # Earlier sub-batches' selected panels stay live beside this selection.
            budget.retained_panels=tuple(a for a in jax.tree.leaves([(part[0],part[3]) for part in parts])
                                         if hasattr(a,'sharding'))
            price=budget.resident_quote(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                selection_faces=_selection_faces(samples,moments,line,packed))['resident_bytes_per_rank']
            ledger.reserve(f"sector.staged.{name}.selection.{len(ledger.entries)}",
                resident_bytes_per_rank=price+eigh_bytes,workspace_bytes_per_rank=0,concurrent_with=ambient)
            states,counts,roles,infinity,values=_sector_selection(samples,moments,line,recipe,
                dict(geometry,ids=sub,real=sub_real),n,eig,mesh_xy=mesh_xy)
            # A short last sub-batch repeats its last slot: keep the leading slots only.
            parts.append((states,np.asarray(counts)[:int(sub_real)],roles,infinity,list(values)[:int(sub_real)],sub_real))
            del samples,moments,line
    budget.retained_panels=()
    with timing.section('decoupled.stack'):
        # One carrier per state over every sub-batch (the widest selection), then one stack.
        widths=[max(ws) for ws in zip(*(recipe_panel_widths(part[2][0],part[0],recipe,column_extent=extent,logical_n=n)
                                        for part in parts))]
        infinity_width=max(recipe_infinity_width(part[3],recipe,column_extent=extent,logical_n=n) for part in parts)
        padded=[pad_states(part[0],widths,part[3],infinity_width) for part in parts]
        take=lambda a,keep:a if int(a.shape[0])==int(keep) else parent_rows(mesh_xy,a,np.arange(int(keep)))
        def node(a):
            z=padded[0][0][a][0]
            if np.ndim(z)==0:
                return z
            return np.concatenate([np.asarray(sub[0][a][0])[:part[5]] for sub,part in zip(padded,parts)])
        states=[(node(a),*_stack(mesh_xy,[tuple(take(x,part[5]) for x in sub[0][a][1:]) for sub,part in zip(padded,parts)]))
                for a in range(len(padded[0][0]))]
        infinity=_stack(mesh_xy,[tuple(take(x,part[5]) for x in sub[1]) for sub,part in zip(padded,parts)])
        counts=np.concatenate([part[1] for part in parts])
        values=[v for part in parts for v in part[4]]
        roles=parts[0][2]
        del padded,parts
    tables=_sector_tables(meta,geometry,n,counts,widths,states,values,infinity_width,extent)
    side=int(tables['active'].shape[-1])
    carrier=face_ritz_carrier(mesh_xy,recipe['pole_budget'])
    # The selected (node, Q, O) and infinity panels stay to the end; the dW Q panels only
    # through the members stage. Each stage and eigh is priced from the round's shapes.
    held=sum(_shard_bytes(a) for st in states for a in st[1:3])+sum(_shard_bytes(a) for a in infinity)
    dw=sum(_shard_bytes(a) for st in states for a in st[3:])
    stacks,boundaries=staged_sector_bytes(parents=width,ranks=mesh_xy.size,side=side,carrier=carrier,packed=packed)
    program=lambda w:face_reduction_bytes(mesh_xy,int(w),rows=packed,side=side,carrier=carrier,retain_span=True)
    stage=stage_width(ledger,stacks+held+dw,program,width,live=ambient)
    row=ledger.reserve(f"sector.staged.{name}.stacks.{len(ledger.entries)}",
                       resident_bytes_per_rank=stacks+held+dw+program(stage),workspace_bytes_per_rank=0,
                       concurrent_with=ambient)
    ledger.live_stages=(*ambient,row['stage'])
    labels=('H_vv','Schur','reduced')
    plans=tuple(staged_eigh(mesh_xy,(width,m,m),bound+held,ledger=ledger,live=ambient,label=f'{name} {label}')
                for (m,bound),label in zip(boundaries,labels))

    @contextmanager
    def eigh_row(k,plan,stack):
        # The eigh of stack k runs beside its boundary: one row for both while it runs.
        stage=ledger.reserve(f"sector.staged.{name}.eigh{k}.{len(ledger.entries)}",
            resident_bytes_per_rank=boundaries[k][1]+held+distrib_la.eigh_stack_bytes(plan,stack.shape,stack.dtype),
            workspace_bytes_per_rank=0,concurrent_with=ambient)['stage']
        ledger.live_stages=(*ambient,stage)
        try:
            yield
        finally:
            ledger.live_stages=(*ambient,row['stage'])
    try:
        reduced=face_reduce_decoupled(states,infinity,tables,mesh=mesh_xy,eigh_plans=plans,width=stage,
            ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True,
            gram_keep=_SECTOR_GRAM_KEEP(),carrier=carrier,eigh_rows=eigh_row)
    finally:
        ledger.live_stages=ambient
    model,signed,vectors,diagnostics,y=reduced
    _sector_gates(diagnostics,name,ids,int(real))
    residual=lambda key:float(np.max(np.asarray(diagnostics[0][key])[:int(real)]))
    receipt=dict(parents=width,side=side,selection=int(tile['selection']),stage=int(stage),
                 eigh_routes=['c' if p.batched_route==distrib_la.ROUTE_BATCH_RESHARD else 'mesh' for p in plans],
                 stacks_bytes_per_rank=int(stacks+held+dw),keep_residual=residual('metric_inverse_root_residual_relative'))
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                recipe=recipe,budget=budget,execution='face',staged=receipt)


#: What ``release_selection_panels`` drops from a staged diagonal sector.
RELEASED_SELECTION=('states','infinity','coefficients')


def release_selection_panels(sectors):
    """Drop the diagonal sectors' selection panels (node, Q, O, infinity) and spans once the
    round's CT pencils are built: the round then reads only the sectors' models and signed
    factors (treatment, held checks, writes), so their device bytes return before the CT eighs."""
    for sector in sectors:
        for key in RELEASED_SELECTION:
            sector[key]=None


def slice_sector(sector, slots, mesh_xy):
    """A CT sub-batch's view of a round's sector: the ``slots`` rows of every per-parent array
    and table; shared records (roles, recipe, budget) pass through."""
    import jax
    import numpy as np
    from gw.shared_pole_execution import parent_rows
    index=np.asarray(slots,np.int64)
    def rows(a):
        if isinstance(a,(int,float,bool,str,dict)) or a is None:
            return a
        if isinstance(a,np.ndarray):
            return a[index] if a.ndim>=1 and a.shape[0]>int(index.max()) else a
        if isinstance(a,(complex,float,int,bool,np.generic)):
            return a
        if hasattr(a,'shape') and hasattr(a,'sharding'):
            return parent_rows(mesh_xy,a,index)
        return a
    out=dict(sector)
    out['model']=tuple(rows(a) for a in sector['model'])
    out['signed']=tuple(rows(a) for a in sector['signed'])
    out['coefficients']=rows(sector['coefficients'])
    out['states']=[tuple(rows(a) for a in st) for st in sector['states']]
    out['infinity']=tuple(rows(a) for a in sector['infinity'])
    out['tables']={k:rows(np.asarray(v)) for k,v in sector['tables'].items()}
    out['vectors']=tuple(rows(a) for a in sector['vectors'])
    out['diagnostics']=jax.tree.map(rows,sector['diagnostics'])
    return out


def staged_cross(sectors, read, ids, real, meta, *, mesh_xy, sample_ids, cross_elements, released, release):
    """One round's CT from its CC and TT on the face: each sub-batch's joint pencil from its own
    samples (``read(ids, real)``; ``cross_elements`` its cross panels per parent, priced) at one
    compacted span for the round, written in place into
    one stack, then the joint reduction (``face_cross_decoupled``) with each eigh once over the
    round's stack. The pencils run beside the sectors' held outputs (the ledger's live stages);
    ``release()`` then drops the sectors' selection panels and spans, and the eighs run beside
    ``released``. Priced at the round's joint side; sub-batches are the widest halving of the
    round whose pencil program fits beside the stacks. Returns the round's CT outputs."""
    import jax
    import numpy as np
    import distrib_la
    from common import timing
    from contextlib import contextmanager
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_capacity import staged_cross_bytes,_shard_bytes
    from gw.shared_pole_execution import (_assemble,face_cross_bytes,face_cross_decoupled,parent_rows,
                                          staged_eigh,stage_width)
    from gw.shared_pole_local import parent_rounds
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    live,held=(_cross_carriers(w,mesh_xy,'face') for w in cross_span_widths(meta,sectors))
    widths=[max(a,b) for a,b in zip(live,held)]
    side=sum(widths)
    rows=tuple(int(s['model'][0].shape[-2]) for s in sectors)
    width=len(ids)
    stacks,boundaries=staged_cross_bytes(parents=width,ranks=mesh_xy.size,side=side,rows=rows)
    # A pencil sub-batch of w parents: its program and its reads (CT and TC at the dense
    # samples, the CT moments, both families' cross panels), priced from the shapes.
    sides=[int(s['coefficients'].shape[-2]) for s in sectors]
    reads=lambda w:-(-16*int(w)*((4*len(sample_ids)+8)*sum(rows)**2+int(cross_elements))//int(mesh_xy.size))
    program=lambda w:face_cross_bytes(mesh_xy,int(w),sum(rows),sides,widths)+reads(w)
    stage=stage_width(ledger,stacks,program,width,live=upstream)
    stack_row=ledger.reserve(f"sector.staged.CT.stacks.{len(ledger.entries)}",
        resident_bytes_per_rank=stacks+program(stage),workspace_bytes_per_rank=0,concurrent_with=upstream)
    ledger.live_stages=(*upstream,stack_row['stage'])

    def parts():
        for slots,slot_real,_ in parent_rounds(width,stage):
            subs=[slice_sector(sec,slots,mesh_xy) for sec in sectors]
            samples,cm,line_cross=read([ids[i] for i in slots],slot_real)
            pencil=_cross_pencil(subs,samples,cm,mesh_xy=mesh_xy,sample_ids=sample_ids,
                                 line_cross=line_cross,widths=widths)
            del samples,cm,line_cross,subs
            yield int(slots[0]),(pencil if int(slot_real)==int(stage)
                                 else parent_rows(mesh_xy,pencil,np.arange(int(slot_real))))
    try:
        with timing.section('decoupled.pencils'):
            pencil=list(_assemble(mesh_xy,width,parts()))
        release()
        ledger.live_stages=(*released,stack_row['stage'])
        plans=tuple(staged_eigh(mesh_xy,(width,m,m),bound,ledger=ledger,live=released,label=f'CT {label}')
                    for (m,bound),label in zip(boundaries,('metric','Ritz')))

        @contextmanager
        def eigh_row(k,plan,stack):
            stage_name=ledger.reserve(f"sector.staged.CT.eigh{k}.{len(ledger.entries)}",
                resident_bytes_per_rank=boundaries[k][1]+distrib_la.eigh_stack_bytes(plan,stack.shape,stack.dtype),
                workspace_bytes_per_rank=0,concurrent_with=released)['stage']
            ledger.live_stages=(*released,stage_name)
            try:
                yield
            finally:
                ledger.live_stages=(*released,stack_row['stage'])
        signed,diagnostics=face_cross_decoupled(pencil,mesh=mesh_xy,eigh_plans=plans,width=stage,
                                                eigh_rows=eigh_row)
        del pencil
    finally:
        ledger.live_stages=upstream
    for name in ('gram_valid','retained_metric_positive'):
        if not bool(jnp.all(diagnostics[name][:int(real)])):
            bad=np.flatnonzero(~np.asarray(diagnostics[name][:int(real)])).tolist()
            raise ValueError(f'GATE shared_pole_sector_{name}: sector=CT, parents={[ids[i] for i in bad]}; '
                f"Gram min/max={float(jnp.min(diagnostics['gram_min_relative'][:int(real)])):.9e}; "
                f"threshold={gates['normalized_gram_validity']['threshold']}; no repair")
    models,zero=positive_cross_models(signed,mesh_xy=mesh_xy)
    if not bool(jnp.all(zero['zero_policy'][:int(real)])):
        raise ValueError('GATE shared_pole_sector_zero_ritz: sector=CT')
    replicated=NamedSharding(mesh_xy,P())
    staged=dict(parents=width,side=side,stage=int(stage),stacks_bytes_per_rank=int(stacks),
                eigh_routes=['c' if p.batched_route==distrib_la.ROUTE_BATCH_RESHARD else 'mesh' for p in plans])
    return dict(models=models,signed=signed,
                diagnostics=jax.tree.map(lambda a:device_put_process_local(a,replicated),diagnostics),
                zero=jax.tree.map(lambda a:device_put_process_local(a,replicated),zero),staged=staged)


def _cross_carriers(widths, mesh_xy, execution):
    """Each sector's compacted CT span carrier: its width, on the face padded to the port axis."""
    if execution != 'face':
        return list(widths)
    from jax.sharding import PartitionSpec as P
    from runtime.padding import padded_axis
    return [padded_axis(width,mesh_xy,name='shared_pole_port',
        specs=((P('x','y'),0),(P('x','y'),1))).carrier for width in widths]


def _cross_actions(sectors, samples, *, mesh_xy, sample_ids, line_cross):
    """The TC-on-C and CT-on-T actions on the diagonal sectors' selected directions."""
    ct,tc=samples
    return tuple(cross_round_actions((forward['Wc'],reverse['Wc'],forward['dWc_ds'],reverse['dWc_ds']),
                                     source['states'],source['roles'],source['recipe'],sample_ids=sample_ids,
                                     mesh_xy=mesh_xy,line_cross=stored)
                 for source,forward,reverse,stored in ((sectors[0],tc,ct,line_cross[0]),
                                                      (sectors[1],ct,tc,line_cross[1])))


def _cross_pencil(sectors, samples, moments, *, mesh_xy, sample_ids, line_cross, widths):
    """One face sub-batch's CT joint pencil (metric, value, O_C, O_T) at the round's spans."""
    from gw.shared_pole_execution import cross_pencil_program
    actions=_cross_actions(sectors,samples,mesh_xy=mesh_xy,sample_ids=sample_ids,line_cross=line_cross)
    packed=_pack_cross_spans(sectors,widths,mesh_xy=mesh_xy,execution='face')
    return cross_pencil_program(mesh_xy)(*packed,actions,tuple(moments[f'M{i}'] for i in range(4)))


def construct_cross_sector_round(sectors, samples, moments, meta, config, *,
                                 mesh_xy, sample_ids, line_cross, real):
    """Run the q-local CT on the two current-map diagonal spans, keeping both outputs.

    ``samples=(CT,TC)`` contains the native rectangular Wc/dWc_ds rounds at
    the dense fitted ``sample_ids``; ``line_cross[family]`` maps each line-panel
    sample to that family's stored cross panel (``read_line_panels(cross=True)``);
    moments is the CT M0..M3 round. All operators are parent-local. The
    signed physical photon interaction is admitted by the unchanged positive
    retained-H checks, not by the scalar positive-V upper passivity bound.
    The compacted spans are the held widths of an SC run (``cross_span_widths``).
    """
    import copy
    import jax
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    charge, transverse = sectors
    ct, tc = samples
    retained = jax.tree.leaves(tuple((s['model'],s['signed'],s['coefficients'],
        s['infinity'],tuple(state[1:] for state in s['states'])) for s in sectors))
    local_meta = copy.copy(meta)
    local_meta.n_rmu_padded = sum(s['model'][0].shape[-2] for s in sectors)
    budget = ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
        mesh_xy=mesh_xy,ledger=meta.shared_pole_capacity,
        upstream=meta.shared_pole_capacity.live_stages,execution='local')
    budget.batch_width = charge['model'][0].shape[0]
    # Cross assembly has rectangular original pencils; only the projected
    # retained pair is square. Keep those two extents distinct in the ledger.
    original_sides = tuple(s['coefficients'].shape[-2] for s in sectors)
    _,widths=cross_span_widths(meta,sectors)
    actions=_cross_actions(sectors,samples,mesh_xy=mesh_xy,sample_ids=sample_ids,line_cross=line_cross)
    packed=_pack_cross_spans(sectors,widths,mesh_xy=mesh_xy,execution='local')
    budget.retained_panels = (*retained,*ct.values(),*tc.values(),*moments.values(),
        *(panels for stored in line_cross for panels,_ in stored.values()),*jax.tree.leaves((actions,packed)))
    side=sum(s[4].shape[-1] for s in packed)
    budget.plan(side,phase='cross_reduction',cross_original_sides=original_sides)
    signed,diagnostics=reduce_cross_round(*packed,actions,
        tuple(moments[f'M{i}'] for i in range(4)),mesh_xy=mesh_xy,
        eigh_plan=budget.eigenplan(side))
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


def cross_span_widths(meta, sectors):
    """The CT round's compacted span width of CC and TT: ``(live, held)``.

    ``live`` is each sector's largest retained rank in the round on the extent
    ladder (``runtime.padding.ladder_extent``), at most its span's columns.
    Ranks drift across a ladder step between rounds and SC maps (Fe 4^3
    bispinor CC 1152 <-> 1280, TT 1408 <-> 1536), and each step recompiled
    the cross reduction (2 x 11.2 s at map 2). Every SC map binds
    ``meta.shared_pole_rank_capacity`` (a dict the quadrature session keeps):
    ``held`` is then the largest live width of any earlier round or map, grown
    only when a live width exceeds it, with the growth noted in
    ``held["_events"]`` for the SC log; it never shrinks. The extra columns
    are inactive retained columns, exact zeros in Y and c: the joint pencil
    gives them zero metric and the zero-row-safe eigensolver keeps them out of
    the spectrum. No binding (one-shot): ``held`` is ``live``.
    """
    from runtime.padding import ladder_extent
    live=[];held=[]
    capacity=getattr(meta,'shared_pole_rank_capacity',None)
    for name,sector in zip(('CC','TT'),sectors):
        active=sector['signed'][2]
        cap=int(sector['coefficients'].shape[-1])
        rank=int(jnp.max(jnp.sum(active,axis=-1)))
        width=min(cap,ladder_extent(rank))
        live.append(width)
        if capacity is None:
            held.append(width)
            continue
        before=int(capacity.get(name,0))
        if width>before:
            if before:
                capacity.setdefault('_events',[]).append(
                    f"shared-pole CT span ({name}): retained rank {rank} exceeds the "
                    f"held width {before}; grown to {width}")
            capacity[name]=width
        held.append(min(cap,int(capacity[name])))
    return live,held


def _pack_cross_spans(sectors, widths, *, mesh_xy, execution):
    """Each diagonal sector's CT operands with its retained span compacted to ``widths``."""
    import jax
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.shared_pole_local import _batch_put
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
    return packed


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
    # Y holds the solve's columns only; the signed model's columns past them are zero padding.
    order=jnp.argsort(~active[:,:y.shape[-1]],axis=-1,stable=True)[:,:width]
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
    from gw.shared_pole_directions import _round_kernels, _sample_point
    from gw.shared_pole_execution import is_face,cross_action_program
    from gw.shared_pole_local import _pad_columns
    face=is_face(samples[0])
    k=_round_kernels(mesh_xy,'face' if face else 'batch')
    index={int(sid):i for i,sid in enumerate(sample_ids)}
    stored={}
    outputs=[]
    for state,role in zip(states,roles[0]):
        sid=int(role['sample_id'])
        conjugate=bool(role.get('conjugate',False))
        mirror=bool(role.get('mirror',False))
        if sid in line_cross:
            if sid not in stored:
                panels=line_cross[sid][0]
                stored[sid]=list(k.columns(int(panels.shape[1]))(panels))
            # The round padded the state's direction to its carrier
            # (its recipe carrier); its padding columns act as zero.
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
    ``moments`` is M0_CT..M3_CT. The q-local round keeps each parent on its
    rank; the staged face route runs the same equations in stages
    (``face_cross_decoupled``). ``eigh_plan`` is the local service plan.
    Returns two signed CT endpoint factors, inverse poles, active columns and
    the unchanged joint-metric diagnostics.
    """
    return _local_cross_parent_program(mesh_xy,eigh_plan.native_fn)(charge,transverse,cross,moments)


@lru_cache(maxsize=None)
def _local_cross_parent_program(mesh,native_eigh):
    from functools import partial
    import jax
    from common.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_local import _mm
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    spec=P(('x','y'))
    body=partial(_cross_reduce_equations,mm=_mm,eigh=native_eigh,gates=gates)
    return jax.jit(shard_map(body,mesh=mesh,in_specs=(spec,)*4,
                            out_specs=(spec,spec),check_vma=False))


def _cross_reduce_equations(charge,transverse,cross,moments,*,mm,eigh,gates,matrix_sharding=None):
    return reduce_sector_pencil(_cross_pencil_equations(charge,transverse,cross,moments,mm=mm,
                                                        matrix_sharding=matrix_sharding),
                               eigh=eigh,matmul=mm,gates=gates,matrix_sharding=matrix_sharding)


def _cross_pencil_equations(charge,transverse,cross,moments,*,mm,matrix_sharding=None):
    """The CT joint pencil (metric, value, O_C, O_T) on the two retained spans (plan 12.2)."""
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
    return joint_sector_pencil(cc,tt,(h,g),matmul=mm,matrix_sharding=matrix_sharding)


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
    gamma, u = eigh(pencil[0])
    stage = joint_keep_stage(pencil, gamma, u, matmul=matmul, gates=gates, matrix_sharding=matrix_sharding)
    values, rotation = eigh(stage["reduced"])
    return joint_output_stage(stage, values, rotation, matmul=matmul)


# The joint CT reduction's two GEMM stages around its two eighs (``reduce_sector_pencil``
# composes them in one program; the staged CT runs each stage over a batch of parents
# and each eigh over the round's stack, ``shared_pole_execution.face_cross_decoupled``).

def joint_keep_stage(pencil, gamma, u, *, matmul, gates, matrix_sharding=None):
    """The metric's keep cut and its corrected basis Y; ``reduced`` (Y^H V Y, the null block at
    the sentinel) goes to the second eigh."""
    from distrib_la import hermitian_part
    from gw.shared_pole_reduction import _metric_inverse_root

    metric, value, oc, ot = pencil
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
    # Y^H V Y is Hermitian only to round-off amplified by 1/gamma near the keep
    # cut (3e-11 relative at gamma/top 1e-7, n 1024, above 64 n eps): the
    # whole-mesh eigh, checked against its operand, refuses it, while the local
    # eigh symmetrizes. Its Hermitian part is what both routes solve.
    reduced = _matrix_layout(hermitian_part(matmul(y, matmul(value, y), transa="C")), matrix_sharding)
    sentinel = -(jnp.linalg.norm(reduced, axis=(-2, -1)) + 1)
    return dict(reduced=_matrix_layout(reduced + null * sentinel[:, None, None], matrix_sharding), y=y,
                oc=oc, ot=ot, keep=keep, valid=valid, ratio=ratio, corrected=corrected, diagnostics=diagnostics)


def joint_output_stage(stage, values, rotation, *, matmul):
    """The CT endpoint factors c_C, c_T, inverse poles and active columns, and the diagnostics."""
    y, keep = stage["y"], stage["keep"]
    count = jnp.sum(keep, axis=-1)
    side = y.shape[-1]
    active = jnp.arange(side)[None] >= side - count[:, None]
    coefficients = matmul(y, rotation) * active[:, None, :]
    return (matmul(stage["oc"], coefficients), matmul(stage["ot"], coefficients),
            jnp.where(active, values, 1), active), dict(
                stage["diagnostics"], gram_valid=stage["valid"], gram_min_relative=stage["ratio"],
                retained_metric_positive=stage["corrected"], retained_rank=count)


def staged_round(read_diagonal, read_cross, ids, real, meta, config, geometry, *, mesh_xy,
                 sample_ids, receipt, rows):
    """One round of the staged face route: TT, then CC beside TT's held outputs, then CT from
    both. Each sector's held outputs are a ledger row live through the CT pencils, its models
    alone (``kept``) another, live through the CT eighs. Returns ``(sectors, cross)`` with
    ``cross['budget']`` the round's live-row planner beside the round's upstream."""
    import copy
    import jax
    from functools import partial
    from common import timing
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,_shard_bytes
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    row_bytes=lambda tree:sum(_shard_bytes(a) for a in {id(a):a for a in jax.tree.leaves(tree)
                                                         if hasattr(a,'sharding')}.values())
    route=rows[0]['staged_route']
    whole={};held=[];kept=[]
    try:
        for family in (1,0):
            name=('CC','TT')[family]
            with timing.section('spole.sector.'+name+'.all', announce=True):
                model=staged_sector(partial(read_diagonal,family),ids,real,meta,config,geometry(family),
                                    mesh_xy=mesh_xy,route=route)
            whole[family]=model
            rows[family]['staged']=model['staged']
            receipt(name,model,ids,real)
            held.append(ledger.reserve(f"sector.staged.held.{name}.{len(ledger.entries)}",
                resident_bytes_per_rank=row_bytes(model),workspace_bytes_per_rank=0,
                concurrent_with=ledger.live_stages)['stage'])
            kept.append(ledger.reserve(f"sector.staged.kept.{name}.{len(ledger.entries)}",
                resident_bytes_per_rank=row_bytes({k:v for k,v in model.items() if k not in RELEASED_SELECTION}),
                workspace_bytes_per_rank=0,concurrent_with=upstream)['stage'])
            ledger.live_stages=(*upstream,*held)
        sectors=[whole[0],whole[1]]
        with timing.section('spole.sector.CT.all', announce=True):
            cross=staged_cross(sectors,read_cross,ids,real,meta,mesh_xy=mesh_xy,cross_elements=route['CT'],
                sample_ids=sample_ids,released=(*upstream,*kept),release=lambda:release_selection_panels(sectors))
        rows[0]['joint']['staged']=cross['staged']
    finally:
        ledger.live_stages=upstream
    local_meta=copy.copy(meta)
    local_meta.n_rmu_padded=sum(int(s['model'][0].shape[-2]) for s in sectors)
    cross['budget']=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
        mesh_xy=mesh_xy,ledger=ledger,upstream=upstream,execution='face')
    return sectors,cross


def sector_route(meta, config, mu_bases, nq, *, mesh_xy, upstream, print_fn=print):
    """The map's sector route from the recipe shapes, decided once and printed on one line
    (``Shared-pole sector constructor: route ...``, a prefix the production report keeps).

    q-local, rounds of P parents with one whole parent per rank, when every sector's local
    round fits (``sector_execution``); otherwise rounds of R = min(nq, P) parents (balanced)
    on the face through the staged reduction (``staged_round``). R is set by P, never by the
    budget. The selection runs in sub-batches of the fixed tile (``runtime.tiles``), priced per
    parent from the recipe's selection inputs; stage widths and eigh routes are decided per
    round at the round's shapes (``stage_width``, ``staged_eigh``). Returns
    ``(execution, rows, route)``: ``route['width']`` parents per round; the staged route's
    tiles ride on ``rows[0]['staged_route']``.
    """
    import jax
    from types import SimpleNamespace
    from gw.shared_pole_capacity import shared_pole_byte_terms
    from gw.shared_pole_execution import line_panel_count
    from runtime.tiles import tile_units
    execution,rows=sector_execution(meta,config,mu_bases,nq,mesh_xy=mesh_xy,upstream=upstream)
    nq,ranks=int(nq),int(mesh_xy.size)
    rounds=-(-nq//min(nq,ranks))
    width=ranks if execution=='local' else -(-nq//rounds)
    joint=rows[0]['joint']
    need=lambda row:max(row[k]['aggregate_bytes_per_rank'] for k in ('local_selection','local_reduction') if row.get(k))
    budget=rows[0]['local_selection']['device_budget_bytes_per_rank']
    local=", ".join(f"{name} {need(row)/1e9:.1f}" for name,row in (('CC',rows[0]),('TT',rows[1]),('CT',joint)))
    if execution=='face':
        recipe=meta.shared_pole_recipe
        lines=line_panel_count(recipe)
        dense=len(recipe['fit_ids'])-lines
        tiles={}
        for row in rows:
            packed,side,iw=int(row['packed_extent']),int(row['conservative_pencil_side']),int(row['infinity_width'])
            unit=shared_pole_byte_terms(SimpleNamespace(n_rmu_padded=packed),mesh_xy=mesh_xy,
                resolution=SimpleNamespace(layout='distributed'),pencil_side=0,parent_batch=1,
                sample_batch=max(1,dense),phase='selection',selection_faces=int(row['selection_faces']))['resident_bytes_per_rank']
            # Every selected (Q, O) and infinity panel of the round at the recipe bound.
            panels=-(-16*(2*packed*(side-2*iw)+5*packed*iw)*width//ranks)
            tiles[row['sector']]=dict(selection=tile_units(unit,width),unit=int(unit),dense=dense,panels=panels)
        tiles['CT']=lines*8*(int(rows[1]['packed_extent'])*int(rows[0]['line_width'])
                             +int(rows[0]['packed_extent'])*int(rows[1]['line_width']))
        rows[0]['staged_route']=tiles
        text=(f"rounds of {width} of {nq} parents ({-(-nq//width)} round(s)) on the face, staged; "
              f"selection tiles CC {tiles['CC']['selection']}, TT {tiles['TT']['selection']} parents; "
              f"q-local needs GB/rank {local} of {budget/1e9:.1f}")
    else:
        text=f"q-local, {-(-nq//width)} round(s) of {width} parents (local GB/rank {local} of {budget/1e9:.1f})"
    print_fn(f"Shared-pole sector constructor: route {text}")
    return execution,rows,dict(width=width,rounds=-(-nq//width))


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
    from gw.shared_pole_capacity import held_sector_bytes, retained_span_columns
    extent=port_extent(mesh_xy)
    execution_rows=[]
    rows=[(1 if family==0 else 3)*basis.n_packed for family,basis in enumerate(mu_bases)]
    # TT runs beside CC's held round outputs, CT beside both (rank-local:
    # one parent per rank), at each sector's conservative side.
    held=0
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
            sample_batch=len(recipe['fit_ids'])-line_panel_count(recipe),column_extent=extent,
            ritz_budget=local_recipe['pole_budget'],retain_span=True,carry=held)
        side=route['conservative_pencil_side']
        budget=local_recipe['pole_budget']
        kept=side if budget is None else 2*min(side//2,int(budget))
        held+=held_sector_bytes(local_meta.n_rmu_padded,side,kept)
        cap=local_recipe['line_direction_cap']
        execution_rows.append(dict(sector=('CC','TT')[family],mode=mode,
                                   packed_extent=local_meta.n_rmu_padded,selection_faces=faces,
                                   line_width=extent(max(1,local_meta.n_rmu if cap is None else min(cap,local_meta.n_rmu))),
                                   signed_side_bound=extent(2*local_recipe['pole_budget']) if local_recipe['pole_budget'] is not None else route['conservative_pencil_side'],
                                   infinity_width=extent(max(1,int(local_recipe['infinity_width']))),
                                   pole_budget=local_recipe['pole_budget'],
                                   span_columns=retained_span_columns(side,kept),**route))
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
        cross_original_sides=tuple(row['conservative_pencil_side'] for row in execution_rows),
        cross_retained_side=sum(min(row['signed_side_bound'],row['span_columns'])
                                for row in execution_rows),carry=held)
    if (joint_mode=='face' and int(nq)>=int(mesh_xy.size)
            and joint_route['local_selection']['device_budget_status']=='PASS'):
        # Whole q-local models per rank (owner 2026-10-03); the round admits the
        # joint pencil at its actual spans, not both at twice their pole budgets.
        joint_mode,joint_route='local',dict(joint_route,reason='parents >= ranks; actual-side joint pencil')
    resolved_execution=('face' if joint_mode=='face' or
                        any(row['mode']=='face' for row in execution_rows)
                        else 'local')
    for row in execution_rows:
        row['joint']=dict(mode=joint_mode,**joint_route)
    return resolved_execution,execution_rows

