"""Cold and held-space current general-W constructions; no Sigma or TD step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('setup-deck','accepted-deck','frames','snapshot','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    # Native bootstrap must precede NumPy/BLAS imports.
    from gw import gw_jax,sc_iteration,shared_pole_screening as screening
    from tests.tdagw.cri3_qgrid_750_campaign import native_frozen_map_probe as frozen
    from tests.tdagw.cri3_qgrid_750_campaign.same_w_rate_replay import load_projector_snapshot,snapshot_energy
    from tdagw.snapshot_runtime import prepare_native_source,_native_window_support_reads
    from gw.gw_init import screening_coulomb_view
    from gw.shared_pole_recipe import bind_shared_pole_census,resolve_shared_pole_recipe
    from gw.ppm_sigma import window_sigma_plan
    from tdagw.dynsig import chi0_active_group
    from centroid.sampling_metric import full_k_quadrature_weights
    from common.units import RYD_TO_EV
    from common.collectives import rank0_transaction,device_put_process_local,gather_to_host
    from common.parallel_transport import fingerprint_from_binding
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    arrays,frame=frozen.load_accepted_frame(a.frames)
    policy=frozen.accepted_response_policy(frame,a.accepted_deck)
    saved=load_projector_snapshot(a.snapshot)
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('fresh benchmark output required')
    provenance=dict(source_revision=os.environ['LEG_HEAD'],
                    source_archive_sha256=os.environ['LEG_SOURCE_ARCHIVE_SHA256'])
    original_solver=sc_iteration.run_self_consistency
    original_screen=screening.screen_general_shared_poles
    finished=[]

    def stop_after_screen(*args,**kwargs):
        held={};records=[]
        for index,phase in enumerate(('cold','held_spaces')):
            call=dict(kwargs,label=f'snapshot_{index+1:06d}',fit_basis=held)
            started=time.monotonic()
            built=original_screen(*args,**call)
            record=dict(status='CURRENT_FULL_W_COMPLETE_NO_SIGMA_NO_TD',phase=phase,
                **provenance,
                source_frame=frame['frame'],snapshot_sha256=hashlib.sha256(Path(a.snapshot).read_bytes()).hexdigest(),
                gpu_count=jax.process_count(),wall_seconds=time.monotonic()-started,
                model=built['shared_pole']['path'],digest=built['shared_pole']['digest'],
                receipt=json.loads(screening._json(built['receipt'])),
                model_identity=built['shared_pole']['identity'],fit_tolerance=.05,physical_steps=0,
                scope='Same saved t0 source with the current device reducer; current chi/W/residues and '
                      'all native checks recomputed. Only numerical fit spaces/rules may be held. '
                      'Pilot W tolerance 5%, not Sigma accuracy or dynamics.')
            records.append(record)
            rank0_transaction(out/('benchmark_'+phase+'.json'),stage='screen_device_benchmark/'+phase,
                write=lambda:(out/('benchmark_'+phase+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n'))
        rank0_transaction(out/'benchmark.json',stage='screen_device_benchmark/complete',
            write=lambda:(out/'benchmark.json').write_text(json.dumps(dict(
                status='COLD_AND_HELD_FULL_W_COMPLETE_NO_SIGMA_NO_TD',rows=records),indent=2,allow_nan=False)+'\n'))
        finished.append(True)
        raise StopIteration('ONE_CURRENT_FULL_W_COMPLETE')

    def intercept(initial,inputs,**kwargs):
        if jax.process_count()!=16:raise ValueError('original sixteen-rank source layout required')
        frozen.validate_callback_reference(frame,inputs.meta.qgrid,policy,inputs.config.qgrid_response_window)
        if (fingerprint_from_binding(inputs.wfn_fingerprint_binding,inputs.wfn)!=frame['wfn_fingerprint']
                or not np.array_equal(inputs.meta.mu_basis.canonical_indices,arrays['centroid_canonical_indices'])
                or not np.allclose(inputs.sym.unfolded_kpts,arrays['kpoints_full_frac'],rtol=0,atol=1e-12)):
            raise ValueError('source WFN/centroids/grid differs from accepted frame')
        mesh=inputs.mesh_xy
        put=lambda v,sh:device_put_process_local(np.asarray(v),sh)
        rep=NamedSharding(mesh,P());face=NamedSharding(mesh,P(None,'x','y'))
        current=sc_iteration.rotate_wavefunctions(inputs.wfns_dft,put(arrays['U_active'],face),
            enk_active_new=put(arrays['energy_padded_ry'][:,:160],rep),
            enk_base=put(arrays['energy_padded_ry'],rep),occupations=put(arrays['occupation_padded'],rep),
            efermi=None,active_slice=slice(0,160),mesh_xy=mesh)
        if (not np.array_equal(gather_to_host(current.enk),arrays['energy_padded_ry'])
                or not np.array_equal(gather_to_host(current.occ),arrays['occupation_padded'])):
            raise ValueError('current full750 energies/occupations differ from accepted frame')
        source=dict(wfns=current,meta=inputs.meta,config=inputs.config,V_q=inputs.V_q,efermi_ry=0.)
        prepared=prepare_native_source(source,mesh)
        if prepared['nb']!=750 or prepared['nmu_logical']!=3000 or prepared['qgrid']!=(4,4,1):
            raise ValueError('exact750band/3000mu/q4 benchmark required')
        rank0_transaction(out,stage='screen_device_benchmark/output',write=lambda:out.mkdir(exist_ok=False,parents=True))
        # This is the native evaluator's exact current snapshot/support setup,
        # without constructing a Sigma plan's irrelevant authenticated-tail
        # prerequisite. Source band-extrapolation policy stays untouched.
        energy=snapshot_energy(prepared,saved)
        active=slice(100,144)
        origin=prepared['frequency_origin_ry']
        plan=window_sigma_plan(inputs.config,head=None,window_only=True)
        reads=_native_window_support_reads(saved['E_ry'],origin,plan)
        a0,a1,_=chi0_active_group(energy[:,:prepared['nb']],active,'auto')
        snapshot=dict(U_active=saved['U'],E_active_ry=saved['E_ry'],rho_active=saved['rho_in_basis'],
            active_slice=active,response_active_slice=slice(a0,a1),energy_ry=energy,
            gamma_ry=.05/RYD_TO_EV,eta_ry=float(inputs.config.sigma.regularization_ev)/RYD_TO_EV,
            spm_support_sites_ev='',support_reads_ev=reads,general_refinement=True,fit_tolerance=.05)
        # The pre-SC interception precedes the map's normal census/recipe bind.
        # Bind that same owner explicitly; no upstream screening is evaluated.
        bind_shared_pole_census(current,inputs.meta,occupation_state=None,
            trs_allowed=inputs.sym.trs_allowed,state_capacity=inputs.wfn.occupation_state_capacity,
            kweights=full_k_quadrature_weights(inputs.wfn,inputs.wfn.symmetry()))
        inputs.meta.shared_pole_recipe=resolve_shared_pole_recipe(inputs.config,current,inputs.meta,
            mesh_xy=mesh,print_fn=inputs.print_fn,support_reads_ev=reads)
        # No upstream W or ledger-owned reservation exists in this pre-map
        # diagnostic. Driver arrays are included in the freshly bound baseline.
        inputs.meta.shared_pole_capacity.live_stages=()
        inputs.meta.shared_pole_capacity.at_phase()
        v_screen=screening_coulomb_view(inputs.V_q,meta=inputs.meta,mesh_xy=mesh)
        stop_after_screen(current,v_screen,inputs.meta,inputs.config,mesh_xy=mesh,sym=inputs.sym,
            snapshot=snapshot,run_dir=str(out/'models'),label='snapshot_000001',print_fn=inputs.print_fn)

    sc_iteration.run_self_consistency=intercept
    try:
        try:gw_jax.main(['-i',str(Path(a.setup_deck).resolve())])
        except StopIteration as exc:
            if str(exc)!='ONE_CURRENT_FULL_W_COMPLETE' or not finished:raise
        if not finished:raise RuntimeError('native full W benchmark did not finish')
    finally:
        sc_iteration.run_self_consistency=original_solver


if __name__=='__main__':main()
