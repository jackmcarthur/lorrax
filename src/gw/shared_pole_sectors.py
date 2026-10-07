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
    from functools import partial
    from file_io.shared_pole_store import (validate_shared_pole_bank, _metadata, open_shared_pole_bank,
        preview_model_write,read_line_panels,write_shared_pole_model,write_shared_pole_sector_manifest)
    from gw.shared_pole_local import batch_to_face,canonical_factors,carrier_history,face_rows
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

    def read_samples(io,endpoints,retained,ids=None,layout=None):
        # Wc/dWc_ds at the dense fitted samples.
        return read_sector_round(io,meta,bank,header,ids_of(ids),endpoints,sample_ids=dense_fit,
            fields=('Wc','dWc_ds'),retained=retained,execution=layout or execution)

    def read_line(io,family,cross=False,ids=None,layout=None):
        spec=None if (layout or execution)=='face' else P(('x','y'))
        name=('C','T')[family]
        return {sid:read_line_panels(io,meta=meta,header=header,family=name,sample=sid,cross=cross,
                                     q_ids=ids_of(ids),partition_spec=spec)
                for sid in range(line_lo,line_hi)}
    ids_of=lambda given:ids if given is None else given
    receipts=[];stores={};placed=[]
    root=Path(output).parent
    to_face=batch_to_face(mesh_xy)
    ledger=meta.shared_pole_capacity
    upstream=ledger.live_stages
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_execution import sector_round_schedule,is_face
    resolved_execution,execution_rows=sector_execution(
        meta,config,bank['mu_bases'],header['n_q_irr'],mesh_xy=mesh_xy,upstream=upstream)
    # The resident models are reserved first, so the batch is admitted beside them.
    sector_models,model_residence=_sector_model_residence(meta,config,header,bank['mu_bases'],
        execution_rows,mesh_xy=mesh_xy,root=root,upstream=upstream,route=resolved_execution)
    if sector_models is not None:
        upstream=ledger.live_stages=(*upstream,model_residence['stage'])
    batch_width,sizes,face_room = int(mesh_xy.size),{},None

    def used_room(row,budget):
        # The room a sector's reduction ran against, the least over rounds.
        rooms=row['face_eigh_room_bytes_per_rank']
        rooms['reduction']=min(rooms.get('reduction',1<<62),budget.face_room or 0)
    def admit_face(nq):
        nonlocal batch_width,sizes,face_room
        from gw.shared_pole_capacity import face_eigh_room
        from gw.shared_pole_execution import sector_batch_width
        batch_width, batch_admission = sector_batch_width(
            meta,linalg_resolution({'linalg':config.backend.linalg}),recipe,execution_rows,
            mesh=mesh_xy,ledger=ledger,nq=nq)
        sizes = batch_admission['sector_program_bytes_per_rank']
        # distrib_la decides every face eigh stack against the room beside the
        # admitted batch, whose row holds the largest program price, so it
        # bounds every round's retry (whole matrices per rank where they fit).
        face_room = face_eigh_room(batch_admission['reduction'])
        # The decision is printed when it is made; the constructor's summary
        # line comes only after the stage, so a leg that ends mid-stage has none.
        if jax.process_index()==0:
            print(f"Shared-pole face batch: {batch_width} parent(s) of {int(nq)} per round "
                  f"(priced from the shapes)",flush=True)
        for row in (*execution_rows,execution_rows[0]['joint']):
            row['batch_admission'] = batch_admission
            name = row.get('sector','CT')
            row['face_batch'] = dict(parent_batch=batch_width,program_bytes_per_rank=sizes[name])
            row['face_eigh_room_bytes_per_rank'] = dict(selection=face_room)
    if resolved_execution == 'face':
        admit_face(header['n_q_irr'])
    rounds=list(sector_round_schedule(bank,header,meta,config,mesh_xy,
        execution=resolved_execution,batch_width=batch_width))
    # The decoupled face route: CC and TT for every parent at once (the selection in
    # sub-batches of the admitted width, every eigh once over the stack), then the CT
    # rounds over slices of the held outputs. The stacks are reserved at their actual
    # side; stacks over budget reduce in face rounds (construct_diagonal_sector_all warns).
    whole=None
    round_upstream=upstream
    if resolved_execution=='face' and int(header['n_q_irr'])>batch_width:
        # TT first: its 2c stacks are the largest eighs, so they run beside no other
        # sector's held outputs. Each sector's held outputs (every parent's models, span
        # and (Q, O) panels, kept for CT) are one ledger row, live through the CT rounds.
        from gw.shared_pole_capacity import _shard_bytes
        whole={};held_rows=[]
        try:
            for family,name in ((1,'TT'),(0,'CC')):
                with timing.section('spole.sector.'+name+'.all', announce=True):
                    def read(ids,family=family,name=name):
                        with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                            exact=read_sector_round(io,meta,bank,header,ids,(family,family),
                                fields=('M0','M1','M2','M3'),retained=(),execution='face')
                            refuse_nonfinite_moment(name,exact['M1'],len(ids),mesh_xy=mesh_xy)
                            samples=read_samples(io,(family,family),tuple(exact.values()),ids=ids,layout='face')
                            line=read_line(io,family,ids=ids,layout='face')
                        return samples,exact,line
                    geometry=dict(components=3 if family else 1,basis=bank['mu_bases'][family],
                        header=sector_headers[family],sample_ids=dense_fit,sector=name,
                        face_room=face_room,program_bytes=sizes.get(name))
                    model=construct_diagonal_sector_all(read,int(header['n_q_irr']),meta,config,geometry,
                        mesh_xy=mesh_xy,retained=(),width=batch_width)
                    whole[family]=model
                    execution_rows[family]['decoupled']=model['decoupled']
                    used_room(execution_rows[family],model['budget'])
                    leaves=jax.tree.leaves((model['model'],model['signed'],
                        model['coefficients'],model['infinity'],tuple(s[1:] for s in model['states'])))
                    unique={id(a):a for a in leaves if hasattr(a,'sharding')}
                    held_rows.append(ledger.reserve(f"sector.decoupled.held.{name}",
                        resident_bytes_per_rank=sum(_shard_bytes(a) for a in unique.values()),
                        workspace_bytes_per_rank=0,concurrent_with=ledger.live_stages)['stage'])
                    ledger.live_stages=(*upstream,*held_rows)
                    reduction,zero,_,_=model['diagnostics']
                    counts=np.asarray(model['vectors'][1]).sum(axis=-1).tolist()
                    receipts.append(dict(sector=name,parents=list(range(int(header['n_q_irr']))),K=counts,
                        gram_min_relative=np.asarray(reduction['gram_min_relative']).tolist(),
                        zero_policy=np.asarray(zero['zero_policy']).tolist()))
            whole=[whole[0],whole[1]]
        finally:
            round_upstream=(*upstream,*held_rows)
            ledger.live_stages=round_upstream
    cross_all=None
    if whole is not None:
        def read_cross(ids,real):
            with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                ct=read_samples(io,(0,1),(),ids=ids,layout='face')
                tc=read_samples(io,(1,0),tuple(ct.values()),ids=ids,layout='face')
                cm=read_sector_round(io,meta,bank,header,ids,(0,1),fields=('M0','M1','M2','M3'),
                                      retained=(*ct.values(),*tc.values()),execution='face')
                refuse_nonfinite_moment('CT',cm['M1'],real,mesh_xy=mesh_xy)
                line_cross=[read_line(io,family,cross=True,ids=ids,layout='face') for family in (0,1)]
            return (ct,tc),cm,line_cross
        with timing.section('spole.sector.CT.all', announce=True):
            cross_all=construct_cross_sector_all(whole,list(rounds),read_cross,meta,config,mesh_xy=mesh_xy,
                sample_ids=dense_fit,nq=int(header['n_q_irr']),width=batch_width,program_bytes=sizes.get('CT'),
                upstream=round_upstream)
            execution_rows[0]['joint']['decoupled']=cross_all['decoupled']
    while rounds:
        ids,real,slots,execution=rounds.pop(0)
        face=execution=='face'
        sectors=[];retained=[];first_receipt=len(receipts)
        model=None
        if whole is not None and face:
            # The held outputs are their ledger rows (round_upstream); only this
            # round's slices are new arrays, priced where they are made.
            sectors=[slice_sector(sec,ids,mesh_xy) for sec in whole]
        for family,name in (() if sectors else enumerate(('CC','TT'))):
            with timing.section('spole.sector.'+name, announce=True):
                with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                    exact=read_sector_round(io,meta,bank,header,ids,(family,family),
                        fields=('M0','M1','M2','M3'),retained=retained,execution=execution)
                    refuse_nonfinite_moment(name,exact['M1'],real,mesh_xy=mesh_xy)
                    samples=read_samples(io,(family,family),(*retained,*exact.values()))
                    line=read_line(io,family)
                geometry=dict(components=3 if family else 1,basis=bank['mu_bases'][family],
                    ids=ids,real=real,header=sector_headers[family],sample_ids=dense_fit,
                    sector=name,face_room=face_room if face else None,
                    program_bytes=sizes.get(name) if face else None)
                model=construct_diagonal_sector_round(samples,exact,meta,config,geometry,
                    mesh_xy=mesh_xy,retained=retained,line=line)
                del samples,exact,line
                sectors.append(model)
                if execution=='face':
                    used_room(execution_rows[family],model['budget'])

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
        if cross_all is not None:
            cross=slice_cross(cross_all,ids,mesh_xy)
        else:
            with timing.section('spole.sector.CT', announce=True):
                with open_shared_pole_bank(bank['path'],mesh_xy=mesh_xy) as io:
                    ct=read_samples(io,(0,1),retained)
                    tc=read_samples(io,(1,0),(*retained,*ct.values()))
                    cm=read_sector_round(io,meta,bank,header,ids,(0,1),fields=('M0','M1','M2','M3'),
                                          retained=(*retained,*ct.values(),*tc.values()),execution=execution)
                    refuse_nonfinite_moment('CT',cm['M1'],real,mesh_xy=mesh_xy)
                    line_cross=[read_line(io,family,cross=True) for family in (0,1)]
                cross=construct_cross_sector_round(sectors,(ct,tc),cm,meta,config,mesh_xy=mesh_xy,
                    sample_ids=dense_fit,line_cross=line_cross,real=real,
                    program_bytes=sizes.get('CT') if face else None)
                del line_cross
                del ct,tc,cm
                if execution=='face':
                    used_room(execution_rows[0]['joint'],cross['budget'])
        if cross is None:
            # The slow fallback (owner: warn, never refuse on budget): this
            # round's parents run again as face batches, CC and TT included.
            import warnings
            del sectors,retained,model,cross,receipts[first_receipt:]
            ledger.live_stages=upstream
            if not sizes:
                admit_face(real)
            warnings.warn(f'shared-pole CT: the local round of parents {ids[:real]} does not '
                          'fit at its actual spans; it reruns on the face route in batches '
                          f'of {batch_width} parents (slow fallback)',RuntimeWarning,stacklevel=2)
            rounds[:0]=face_rerun_rounds(ids,real,batch_width)
            continue
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
            spans=_contiguous_q_spans(ids,real)
            rows=max(q1-q0 for q0,q1,_ in spans)

            def writer_fits(model_name,width):
                families=(0,1) if model_name=='CT' else ((0,) if model_name=='CC' else (1,))
                try:
                    return all(preview_model_write(meta,(rows,bank['mu_bases'][f].n_packed,3 if f else 1,width),
                                                   basis=bank['mu_bases'][f]) for f in families)
                except (ValueError,MemoryError,RuntimeError):
                    return False
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
                        hold=(writer_history,('writer',name[:2]),partial(writer_fits,name[:2]),writer_events))
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
        ledger.live_stages=round_upstream
        del sectors,cross,models,signed,retained,model
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


def face_rerun_rounds(ids, real, width):
    """Fixed-width face rounds over a local round's ``real`` leading parents
    (``parent_rounds``); its padded slots are never rerun."""
    from gw.shared_pole_local import parent_rounds
    return [(*row,'face') for row in parent_rounds(ids[:int(real)],width)]


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

    ``hold`` is ``(history, key, fits, events)`` for the held writer carrier
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


def held_writer_width(live, kmax, history, key, fits, events):
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
    which the store checks and never reads as poles. ``fits(width)`` prices
    the store's write at the held carrier; if it does not fit, the round
    writes at its live carrier and the hold is kept.
    """
    before=int(history.get(key,0))
    if live>before:
        if before and events is not None:
            events.append(f"shared-pole writer carrier ({key[1]}): Kmax {kmax} exceeds the held "
                          f"width {before}; grown to {live}")
        history[key]=live
        return live
    return before if before==live or fits(before) else live


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
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import (round_tables,reduce_round,recipe_panel_widths,
                                      recipe_infinity_width,pad_states,carrier_history)

    from gw.shared_pole_execution import is_face, face_reduce_round, face_ritz_carrier
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
    budget.face_room=geometry.get('face_room')
    # Face: the reduction is priced at its sector's compiled program (sector_batch_width).
    budget.program_bytes=geometry.get('program_bytes')
    # Priced as sector_execution's route decision prices the local round.
    if execution=='local':
        budget.ritz_budget=recipe['pole_budget']
        budget.retain_span=True
    # The dense sample and moment fields and the stored line panels are
    # resident during selection. Derive the face count from these exact read
    # dictionaries so admission prices the live panel set.
    line={} if line is None else line
    panel_elements=sum(int(np.prod(panels.shape[1:])) for panels,_ in line.values())
    selection_faces=(sum(int(panel.shape[1]) for panel in samples.values())
                     +len(moments)+-(-panel_elements//local_meta.n_rmu_padded**2))
    selection=budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                          selection_faces=selection_faces)
    if execution=='face' and budget.face_room is not None:
        # The selection eighs run beside the selection's own admitted live set.
        from gw.shared_pole_capacity import face_eigh_room
        budget.face_room=min(budget.face_room,face_eigh_room(selection) or 0) or None
    extent=port_extent(mesh_xy)
    states,counts,roles,infinity,values=_sector_selection(samples,moments,line,recipe,geometry,local_meta,n,
        budget,mesh_xy=mesh_xy,execution=execution,retained=retained)
    # Every state panel is padded to its recipe carrier, so the round
    # program's inputs have one shape; the pencil extent is grow-only over
    # this sector's rounds and SC maps (round_tables): discovered in map 0,
    # held from map 1.
    widths=recipe_panel_widths(roles[0],states,recipe,column_extent=extent,logical_n=n)
    infinity_width=recipe_infinity_width(infinity,recipe,column_extent=extent,logical_n=n)
    history=carrier_history(meta)
    name=(('sector',geometry['sector'],n),'extent',2,len(states))
    before=int(history.get(name,(0,))[0])
    tables=round_tables(counts,widths,[s[0] for s in states],[v.shape[-1] for v in values],
        infinity_width,column_extent=extent,ordered=True,odd_moments=True,key=name[0],history=history)
    side=int(tables['active'].shape[-1])
    budget.plan(side,phase='reduction',padding_output_bytes_per_rank=round_padding_output_bytes(
        states,infinity,widths,infinity_width))
    states,infinity=pad_states(states,widths,infinity,infinity_width)
    # An SC map past 0 that still grows its extent says so in its log, as
    # the CT span and K holds do.
    capacity=getattr(meta,'shared_pole_rank_capacity',None)
    if before and capacity is not None and int(history[name][0])>before:
        capacity.setdefault('_events',[]).append(
            f"shared-pole {geometry['sector']} round: a state's selection exceeds its held carrier; "
            f"pencil extent {before} -> {int(history[name][0])}")
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    gram_keep = shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']
    if execution == 'face':
        reduced=face_reduce_round(states,infinity,tables,mesh=mesh_xy,
            budget=budget,ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True,
            gram_keep=gram_keep,admit=False,room=budget.face_room,
            carrier=face_ritz_carrier(mesh_xy,recipe['pole_budget']))
    else:
        reduced=reduce_round(states,infinity,tables,real=geometry['real'],mesh_xy=mesh_xy,
            native_eigh=budget.eigenplan(side).native_fn,ordered=True,odd_moments=True,
            keep_budget=recipe['pole_budget'],retain_span=True,gram_keep=gram_keep)
    model,signed,vectors,diagnostics,y=reduced
    _sector_gates(diagnostics,geometry['sector'],geometry['ids'],geometry['real'])
    # The returned planner must not retain the just-consumed full sample and
    # moment arrays through its accounting view after the caller releases them.
    budget.retained_panels=tuple(retained)
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                recipe=recipe,budget=budget,execution=execution)


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


def _sector_selection(samples, moments, line, recipe, geometry, local_meta, n, budget, *, mesh_xy, execution, retained):
    """The selection half of a diagonal sector round: infinity directions and the state panels
    of one batch of parents (face stacks or batch layout). Returns (states, counts, roles,
    infinity, values); the caller forms the tables and pads the panels."""
    import numpy as np
    from gw.shared_pole_directions import (_round_kernels,line_panel_states,port_extent,
                                           select_round_states,infinity_directions)
    eig=budget.eigenplan(local_meta.n_rmu_padded)
    extent=port_extent(mesh_xy)
    kernels=_round_kernels(mesh_xy,'face' if execution=='face' else 'batch')
    qi,values=infinity_directions(kernels,moments['M1'],min(n,recipe['infinity_width']),
        eigh_plan=eig,column_extent=extent,multiplet_tol=recipe['multiplet_relative_tolerance'],
        real_rows=None if execution=='face' else geometry['real'])
    infinity=(qi,*(kernels.apply(moments[name],qi) for name in ('M0','M1','M2','M3')))
    real=geometry['real']
    line_states={sid:line_panel_states(panels,np.where(np.arange(len(counts))<real,counts,0),
                                       recipe,sid=sid,ordered=True,mesh_xy=mesh_xy)
                 for sid,(panels,counts) in line.items()}
    states,counts,roles=select_round_states(samples,recipe,sample_ids=geometry['sample_ids'],
        real=real,mesh_xy=mesh_xy,eigh_plan=eig,column_extent=extent,
        logical_n=n,ordered=True,line_states=line_states)
    del line_states
    budget.retained_panels=(*retained,*samples.values(),*moments.values(),
                            *(panels for panels,_ in line.values()))
    return states,counts,roles,infinity,values


def construct_diagonal_sector_all(read, nq, meta, config, geometry, *, mesh_xy, retained=(), width):
    """CC or TT for every parent at once: the selection in sub-batches of ``width`` parents
    (``read(ids)`` returns that batch's samples, moments and line panels on the face), one
    set of tables for every parent, and the decoupled reduction (stage programs over
    sub-batches, each eigh once over the stack). Returns what the round returns, for every parent."""
    import copy
    import jax
    import numpy as np
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,face_eigh_room
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import (round_tables,recipe_panel_widths,recipe_infinity_width,pad_states,
                                      carrier_history,parent_rounds)
    from gw.shared_pole_execution import face_reduce_decoupled,face_eigh,face_ritz_carrier,_stack
    components=int(geometry['components'])
    basis=geometry['basis']
    n=components*basis.n_logical
    local_meta=copy.copy(meta)
    local_meta.mu_basis=basis
    local_meta.n_rmu=n
    local_meta.n_rmu_padded=components*basis.n_packed
    recipe=sector_recipe(meta.shared_pole_recipe,n)
    ledger=meta.shared_pole_capacity
    budget=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
                               mesh_xy=mesh_xy,ledger=ledger,upstream=ledger.live_stages,execution='face')
    budget.batch_width=int(width)
    budget.retained_panels=tuple(retained)
    budget.face_room=geometry.get('face_room')
    budget.program_bytes=geometry.get('program_bytes')
    extent=port_extent(mesh_xy)
    from common import timing
    parts=[]
    with timing.section('decoupled.selection'):
        for ids,real,slots in parent_rounds(nq,int(width)):
            samples,moments,line=read(ids)
            line={} if line is None else line
            panel_elements=sum(int(np.prod(panels.shape[1:])) for panels,_ in line.values())
            selection_faces=(sum(int(panel.shape[1]) for panel in samples.values())
                             +len(moments)+-(-panel_elements//local_meta.n_rmu_padded**2))
            # Earlier sub-batches' selected panels stay live beside this selection.
            budget.retained_panels=(*retained,*(a for a in jax.tree.leaves([(part[0],part[3]) for part in parts])
                                                if hasattr(a,'sharding')))
            selection=budget.plan(0,phase='selection',sample_batch=samples['Wc'].shape[1],
                                  selection_faces=selection_faces)
            if budget.face_room is not None:
                budget.face_room=min(budget.face_room,face_eigh_room(selection) or 0) or None
            sub=dict(geometry,ids=ids,real=real)
            states,counts,roles,infinity,values=_sector_selection(samples,moments,line,recipe,sub,local_meta,n,budget,
                mesh_xy=mesh_xy,execution='face',retained=budget.retained_panels)
            # a short last sub-batch repeats its last parent: keep the real slots only
            keep=slice(0,int(real))
            parts.append((states,np.asarray(counts)[keep],roles,infinity,[v for v in values][keep],real))
            del samples,moments,line
            budget.retained_panels=tuple(retained)
    with timing.section('decoupled.stack'):
        # One carrier per state over every sub-batch (the widest selection), then one stack.
        widths=[max(ws) for ws in zip(*(recipe_panel_widths(part[2][0],part[0],recipe,column_extent=extent,logical_n=n)
                                        for part in parts))]
        infinity_width=max(recipe_infinity_width(part[3],recipe,column_extent=extent,logical_n=n) for part in parts)
        padded=[pad_states(part[0],widths,part[3],infinity_width) for part in parts]
        from gw.shared_pole_execution import parent_rows
        take=lambda a,real:a if int(a.shape[0])==int(real) else parent_rows(mesh_xy,a,np.arange(int(real)))
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
    history=carrier_history(meta)
    name=(('sector',geometry['sector'],n),'extent',2,len(states))
    tables=round_tables(counts,widths,[s[0] for s in states],[v.shape[-1] for v in values],
        infinity_width,column_extent=extent,ordered=True,odd_moments=True,key=name[0],history=history)
    side=int(tables['active'].shape[-1])
    budget.retained_panels=(*retained,*(a for s in states for a in s[1:]),*infinity)
    # The stage programs run at the sub-batch width (priced as the face round is); the
    # stacks of every parent are reserved below, apart from them.
    budget.plan(side,phase='reduction',padding_output_bytes_per_rank=0)
    budget.batch_width=int(nq)
    # The stage stacks of every parent sit beside the eigh stacks. A stage's run holds
    # its input stack and the stack it writes in place (face_reduce_decoupled): the
    # keep stage the paired members and the restricted pencil, the paired stage the
    # restricted pencil and (Y, Y^H G_r Y). Each eigh runs beside the (node, Q, O) state
    # panels (the dW Q panels are released after the pencil) and its own boundary stack:
    # H'_vv beside the paired members and its Hermitian copy, the Schur complement beside
    # the restricted pencil, Y^H G_r Y beside (Y, Y^H G_r Y, O_r, the paired span).
    carrier_columns=face_ritz_carrier(mesh_xy,recipe['pole_budget'])
    packed=int(local_meta.n_rmu_padded)
    held=sum(a.size*a.dtype.itemsize//int(mesh_xy.size) for st in states for a in st[1:3])
    held+=sum(a.size*a.dtype.itemsize//int(mesh_xy.size) for a in infinity)
    dw_panels=sum(a.size*a.dtype.itemsize//int(mesh_xy.size) for st in states for a in st[3:])
    # A stage program of ``width`` parents is bounded by the face round program's price
    # at that width (its whole chain); its inputs and outputs are sub-batch slices.
    from gw.shared_pole_execution import face_reduction_bytes,decoupled_stage_bytes
    program=face_reduction_bytes(mesh_xy,int(width),rows=packed,side=side,carrier=carrier_columns,retain_span=True)
    resident,boundaries=decoupled_stage_bytes(nq=nq,ranks=mesh_xy.size,side=side,carrier=carrier_columns,
        packed=packed,held=held,dw_panels=dw_panels,program=program)
    stacks=resident-held-dw_panels-program
    panels=held+dw_panels
    row=ledger.reserve(f"sector.decoupled.{geometry['sector']}.stacks",resident_bytes_per_rank=resident,
                       workspace_bytes_per_rank=0,concurrent_with=ledger.live_stages)
    ambient=ledger.live_stages
    ledger.live_stages=(*ambient,row['stage'])
    rooms=tuple(face_eigh_room(ledger.preview(resident_bytes_per_rank=boundary,
                                              workspace_bytes_per_rank=0,concurrent_with=ambient))
                for boundary in boundaries)

    from contextlib import contextmanager
    from gw.shared_pole_execution import eigh_program_bytes
    @contextmanager
    def eigh_row(k,plan,stack):
        # The eigh of stack k runs beside its boundary: one row for both while it runs, its
        # program priced from the service's decision (eigh_program_bytes).
        stage=ledger.reserve(f"sector.decoupled.{geometry['sector']}.eigh{k}.{len(ledger.entries)}",
                             resident_bytes_per_rank=boundaries[k]+eigh_program_bytes(plan,stack,mesh=mesh_xy),
                             workspace_bytes_per_rank=0,concurrent_with=ambient)['stage']
        ledger.live_stages=(*ambient,stage)
        try:
            yield
        finally:
            ledger.live_stages=(*ambient,row['stage'])
    admitted=row['device_budget_status']=='PASS'
    receipt=dict(parents=int(nq),sub_batch=int(width),pencil_side=int(side),stacks_bytes_per_rank=int(stacks),
                 panels_bytes_per_rank=int(panels),eigh_room_bytes_per_rank=rooms,admitted=bool(admitted))
    if not admitted:
        import warnings
        warnings.warn(f"shared-pole {geometry['sector']}: the decoupled stacks of {int(nq)} parents "
                      f"({stacks/1e9:.1f} GB/rank) do not fit beside the live set; the stacked panels "
                      f"reduce in face rounds of {int(width)} (the slow fallback)",RuntimeWarning)
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    gram_keep=shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']
    carrier=face_ritz_carrier(mesh_xy,recipe['pole_budget'])
    try:
        if admitted:
            # Only the states list holds the dW Q panels now, so the reduction can release them.
            budget.retained_panels=(*retained,*(a for s in states for a in s[1:3]),*infinity)
            reduced=face_reduce_decoupled(states,infinity,tables,mesh=mesh_xy,
                eigh_plans=tuple(face_eigh(mesh_xy,side,r) for r in rooms),
                width=int(width),ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True,
                gram_keep=gram_keep,carrier=carrier,eigh_rows=eigh_row)
        else:
            # The stacks do not fit beside the live set (warn, never refuse): the same
            # stacked panels reduce in face rounds of the admitted width, the eighs per round.
            from gw.shared_pole_execution import face_reduce_round,_take
            ledger.live_stages=ambient
            parts=[]
            for i0 in range(0,int(nq),int(width)):
                i1=min(i0+int(width),int(nq))
                sub_states=[(st[0] if np.ndim(st[0])==0 else np.asarray(st[0])[i0:i1],*_take(mesh_xy,tuple(st[1:]),i0,i1))
                            for st in states]
                sub_tables={k:np.asarray(v)[i0:i1] for k,v in tables.items()}
                parts.append(face_reduce_round(sub_states,_take(mesh_xy,infinity,i0,i1),sub_tables,mesh=mesh_xy,
                    budget=budget,ordered=True,odd_moments=True,keep_budget=recipe['pole_budget'],retain_span=True,
                    gram_keep=gram_keep,admit=False,room=budget.face_room,carrier=carrier))
            reduced=_stack(mesh_xy,parts)
    finally:
        ledger.live_stages=ambient
    model,signed,vectors,diagnostics,y=reduced
    _sector_gates(diagnostics,geometry['sector'],list(range(int(nq))),int(nq))
    residual=lambda key:float(np.max(np.asarray(diagnostics[0][key])[:int(nq)]))
    receipt.update(keep_residual=residual('metric_inverse_root_residual_relative'),
                   paired_residual=residual('paired_metric_inverse_root_residual_relative'),
                   paired_iterations=int(residual('paired_metric_inverse_root_iterations')))
    budget.retained_panels=tuple(retained)
    return dict(model=model,signed=signed,coefficients=y,states=states,infinity=infinity,
                tables=tables,roles=roles,diagnostics=diagnostics,vectors=vectors,
                recipe=recipe,budget=budget,execution='face',decoupled=receipt)


def slice_sector(sector, slots, mesh_xy):
    """One CT round's view of a sector built for every parent: the ``slots`` rows of every
    per-parent array and table; shared records (roles, recipe, budget) pass through."""
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


def construct_cross_sector_all(whole, rounds, read, meta, config, *, mesh_xy, sample_ids, nq, width,
                               program_bytes, upstream):
    """CT for every parent at once beside the decoupled CC and TT (``whole``): each face round's
    joint pencil from its own samples (``read(ids, real)``) at one compacted span for every
    parent, written in place into one stack, then the joint reduction with each eigh once over
    the stack (``face_cross_decoupled``), the gates and the positive models for every parent.
    Returns what ``construct_cross_sector_round`` returns, for every parent, and a receipt."""
    import copy
    import jax
    import numpy as np
    from contextlib import contextmanager
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity,face_eigh_room
    from gw.shared_pole_execution import _assemble,face_cross_decoupled,face_eigh,parent_rows
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    ledger=meta.shared_pole_capacity
    live,held=(_cross_carriers(w,mesh_xy,'face') for w in cross_span_widths(meta,whole))
    widths=[max(a,b) for a,b in zip(live,held)]
    side=sum(widths)
    rows=tuple(int(s['model'][0].shape[-2]) for s in whole)
    from gw.shared_pole_execution import decoupled_cross_bytes
    stacks,boundaries=decoupled_cross_bytes(nq=nq,ranks=mesh_xy.size,side=side,rows=rows)
    # The pencil stack and the keep stage's (Y^H V Y, Y) beside it, plus one round's program.
    stack=ledger.reserve(f"sector.decoupled.CT.stacks",
        resident_bytes_per_rank=stacks+int(program_bytes or 0),
        workspace_bytes_per_rank=0,concurrent_with=upstream)
    ledger.live_stages=(*upstream,stack['stage'])
    budgets=[]

    def parts():
        for ids,real,slots,execution in rounds:
            sectors=[slice_sector(sec,ids,mesh_xy) for sec in whole]
            samples,cm,line_cross=read(ids,real)
            out=construct_cross_sector_round(sectors,samples,cm,meta,config,mesh_xy=mesh_xy,
                sample_ids=sample_ids,line_cross=line_cross,real=real,program_bytes=program_bytes,
                widths=widths,pencil_only=True)
            del samples,cm,line_cross,sectors
            budgets.append(out['budget'])
            pencil=out['pencil']
            yield int(ids[0]),(pencil if int(pencil[0].shape[0])==int(real)
                               else parent_rows(mesh_xy,pencil,np.arange(int(real))))
    try:
        from common import timing
        with timing.section('decoupled.pencils'):
            pencil=_assemble(mesh_xy,int(nq),parts())
        rooms=tuple(face_eigh_room(ledger.preview(resident_bytes_per_rank=b,workspace_bytes_per_rank=0,
                                                  concurrent_with=upstream)) for b in boundaries)

        from gw.shared_pole_execution import eigh_program_bytes
        @contextmanager
        def eigh_row(k,plan,stack):
            row=ledger.reserve(f"sector.decoupled.CT.eigh{k}.{len(ledger.entries)}",
                               resident_bytes_per_rank=boundaries[k]+eigh_program_bytes(plan,stack,mesh=mesh_xy),
                               workspace_bytes_per_rank=0,concurrent_with=upstream)['stage']
            ledger.live_stages=(*upstream,row)
            try:
                yield
            finally:
                ledger.live_stages=(*upstream,stack['stage'])
        signed,diagnostics=face_cross_decoupled(pencil,mesh=mesh_xy,
            eigh_plans=tuple(face_eigh(mesh_xy,side,r) for r in rooms),width=int(width),eigh_rows=eigh_row)
        del pencil
    finally:
        ledger.live_stages=upstream
    for name in ('gram_valid','retained_metric_positive'):
        if not bool(jnp.all(diagnostics[name][:nq])):
            bad=np.flatnonzero(~np.asarray(diagnostics[name][:nq])).tolist()
            raise ValueError(f'GATE shared_pole_sector_{name}: sector=CT, parents={bad}; '
                f"Gram min/max={float(jnp.min(diagnostics['gram_min_relative'][:nq])):.9e}; "
                f"threshold={gates['normalized_gram_validity']['threshold']}; no repair")
    models,zero=positive_cross_models(signed,mesh_xy=mesh_xy)
    if not bool(jnp.all(zero['zero_policy'][:nq])):
        raise ValueError('GATE shared_pole_sector_zero_ritz: sector=CT')
    local_meta=copy.copy(meta)
    local_meta.n_rmu_padded=sum(rows)
    budget=ConstructorCapacity(local_meta,linalg_resolution({'linalg':config.backend.linalg}),
        mesh_xy=mesh_xy,ledger=ledger,upstream=upstream,execution='face')
    budget.batch_width=int(width)
    replicated=NamedSharding(mesh_xy,P())
    return dict(models=models,signed=signed,
                diagnostics=jax.tree.map(lambda a:device_put_process_local(a,replicated),diagnostics),
                zero=jax.tree.map(lambda a:device_put_process_local(a,replicated),zero),budget=budget,
                decoupled=dict(parents=int(nq),sub_batch=int(width),pencil_side=int(side),
                               stacks_bytes_per_rank=int(stacks),
                               eigh_room_bytes_per_rank=rooms,admitted=stack['device_budget_status']=='PASS',
                               keep_residual=float(np.max(np.asarray(diagnostics['metric_inverse_root_residual_relative'])[:nq])),
                               paired_iterations=int(np.max(np.asarray(diagnostics['metric_inverse_root_iterations'])[:nq]))))


def slice_cross(cross, slots, mesh_xy):
    """One round's view of the CT built for every parent: the ``slots`` rows of its models,
    signed factors, diagnostics and zero receipts; the budget passes through."""
    from gw.shared_pole_execution import parent_rows
    import numpy as np
    index=np.asarray(slots,np.int64)
    rows=lambda tree:parent_rows(mesh_xy,tree,index)
    return dict(models=rows(cross['models']),signed=rows(cross['signed']),
                diagnostics=rows(cross['diagnostics']),zero=rows(cross['zero']),budget=cross['budget'])


def _cross_carriers(widths, mesh_xy, execution):
    """Each sector's compacted CT span carrier: its width, on the face padded to the port axis."""
    if execution != 'face':
        return list(widths)
    from jax.sharding import PartitionSpec as P
    from runtime.padding import padded_axis
    return [padded_axis(width,mesh_xy,name='shared_pole_port',
        specs=((P('x','y'),0),(P('x','y'),1))).carrier for width in widths]


def construct_cross_sector_round(sectors, samples, moments, meta, config, *,
                                 mesh_xy, sample_ids, line_cross, real, program_bytes=None,
                                 widths=None, pencil_only=False):
    """Run CT on the two current-map diagonal spans, keeping both outputs.

    ``samples=(CT,TC)`` contains the native rectangular Wc/dWc_ds rounds at
    the dense fitted ``sample_ids``; ``line_cross[family]`` maps each line-panel
    sample to that family's stored cross panel (``read_line_panels(cross=True)``);
    moments is the CT M0..M3 round. All operators are parent-sharded. The
    signed physical photon interaction is admitted by the unchanged positive
    retained-H checks, not by the scalar positive-V upper passivity bound.
    A face round is priced at CT's ``program_bytes`` (``sector_batch_width``).
    Returns None for a local round whose CT does not fit at its actual spans.
    ``widths`` fixes the compacted spans (the decoupled CT: one joint side for every
    round); ``pencil_only`` returns the round's joint pencil (metric, value, O_C, O_T)
    and its budget instead of reducing it.
    """
    import copy
    import jax
    import numpy as np
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_capacity import ConstructorCapacity
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
    budget.face_room = charge['budget'].face_room
    budget.program_bytes = program_bytes
    budget.retained_panels = (*retained,*ct.values(),*tc.values(),*moments.values(),
        *(panels for stored in line_cross for panels,_ in stored.values()))
    # Cross assembly has rectangular original pencils; only the projected
    # retained pair is square. Keep those two extents distinct in the ledger.
    original_sides = tuple(s['coefficients'].shape[-2] for s in sectors)
    # The compacted span of each sector: its held width in an SC run
    # (cross_span_widths), this round's own when the held one does not fit.
    carrier=lambda widths:_cross_carriers(widths,mesh_xy,execution)

    def fits(widths):
        try:
            return budget.preview(sum(widths),phase='cross_reduction',
                cross_original_sides=original_sides)['device_budget_status']=='PASS'
        except (ValueError,MemoryError,RuntimeError):
            return False
    if widths is None:
        live,held=(carrier(w) for w in cross_span_widths(meta,sectors))
        widths=held if held==live or fits(held) else live
    else:
        live=widths=list(widths)
    side = sum(widths)
    budget.plan(side,phase='cross_reduction',cross_original_sides=original_sides)
    actions=[]
    for source, forward, reverse, stored in ((charge,tc,ct,line_cross[0]),(transverse,ct,tc,line_cross[1])):
        panels=(forward['Wc'],reverse['Wc'],forward['dWc_ds'],reverse['dWc_ds'])
        actions.append(cross_round_actions(panels,source['states'],source['roles'],
            source['recipe'],sample_ids=sample_ids,mesh_xy=mesh_xy,line_cross=stored))

    base_panels=budget.retained_panels
    packed=_pack_cross_spans(sectors,widths,mesh_xy=mesh_xy,execution=execution)
    budget.retained_panels=(*base_panels,*jax.tree.leaves((actions,packed)))
    if widths!=live and not fits(widths):
        # The held span fits the reduction but not with this round's actions.
        budget.retained_panels=base_panels
        del packed
        widths=live
        packed=_pack_cross_spans(sectors,widths,mesh_xy=mesh_xy,execution=execution)
        budget.retained_panels=(*base_panels,*jax.tree.leaves((actions,packed)))
    side=sum(s[4].shape[-1] for s in packed)
    if execution=='local' and not fits([side]):
        # One whole parent per rank does not fit at the
        # actual spans: the caller reruns the round on the face (construct_sector_poles).
        budget.retained_panels=tuple(retained)
        return None
    budget.plan(side,phase='cross_reduction',cross_original_sides=original_sides)
    if pencil_only:
        from gw.shared_pole_execution import cross_pencil_program
        pencil=cross_pencil_program(mesh_xy)(*packed,tuple(actions),tuple(moments[f'M{i}'] for i in range(4)))
        budget.retained_panels=tuple(retained)
        return dict(pencil=pencil,budget=budget)
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
    ``moments`` is M0_CT..M3_CT. No full operator leaves the admitted
    local rounds remain parent-local; a face round keeps one physical parent
    over the complete mesh. ``eigh_plan`` is the already-resolved public
    service plan matching that layout. Returns two signed CT endpoint factors,
    inverse poles, active columns and the unchanged joint-metric diagnostics.
    """
    from gw.shared_pole_execution import is_face,cross_parent_program
    if is_face(charge[4]):
        return cross_parent_program(mesh_xy,eigh_plan)(charge,transverse,cross,moments)
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
# composes them in one program; the decoupled CT runs each stage over a batch of parents
# and each eigh over every parent, ``shared_pole_execution.face_cross_decoupled``).

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
                                   packed_extent=local_meta.n_rmu_padded,
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

