"""Exact radial sample storage, typed transport and three distinct OWN solves.

Run on canonical P4 with --output inside the run's scratch directory.
This fixture proves byte identity and refusals, not material cost or capacity.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json


def check_radial_sample_reuse(runtime, directory):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gamma_matrices import current_fit_terms
    from common.staged_reshard import face_to_batch_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.isdf_augmentation import _face_complement, _rhs_storage_kernels
    from gw.augmentation_current import (prepare_radial_sample_store,
        write_radial_samples, read_radial_samples, DeferredCurrentDensity)
    from isdf.local_rhs import local_density_rhs
    from isdf.zeta_mubatch import _local_coefficient_solve
    from symmetry_maps import q_negation_index
    from file_io import _slab_io_rank

    mesh = runtime.mesh
    assert mesh.size == 4 and runtime.process_count == 4
    eye = np.eye(4, dtype=complex)
    sym = SimpleNamespace(sym_matrices=np.eye(3, dtype=np.int32)[None],
        translations=np.zeros((1,3)), irr_idx_k=np.arange(3, dtype=np.int32),
        sym_idx_k=np.zeros(3, np.int32), kirr_fullids=np.arange(3),
        spinor_action=lambda rows,nspinor:np.repeat(eye[None],len(rows),axis=0))
    kpar = np.zeros((3,3)); kpar[:,0] = np.arange(3)/3.
    points = np.array([[.07,0,0],[0,.11,0],[0,0,.15],[.10,.08,0],
                       [0,.09,.13],[.09,0,.18],[.03,.05,.10]])
    plan = build_centroid_k_unfold_plan(sym,points,(4,4,4),mesh,nspinor=4,
        parent_k_frac=kpar,coordinate_kind='fractional')
    packed = plan.layout.axis.pack_host(points,axis=0,fill_value=0.)
    npnt = plan.n_centroid_packed
    shape = (3,4,4,npnt)
    fs = NamedSharding(mesh,P(None,'x',None,'y'))
    qs = NamedSharding(mesh,P(('x','y'),None,None))
    put = lambda a:device_put_process_local(a,fs)
    host = lambda a:np.asarray(gather_to_host(a))
    def exact(a,b):
        np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint64),
                                      np.ascontiguousarray(b).view(np.uint64))
    def refuses(fn, fragment):
        try:
            fn()
        except (ValueError, RuntimeError) as exc:
            assert fragment in str(exc), str(exc)
        else:
            raise AssertionError('negative control was accepted: '+fragment)
    rng = np.random.default_rng(31721)
    original = []
    for _ in range(2):
        pair = []
        for _ in range(2):
            a = (rng.normal(size=shape)+1j*rng.normal(size=shape))*.07
            a[...,~plan.layout.axis.active_mask] = 0.
            # Preserve signed zeros as well as generic complex bits.
            a[0,0,0,0] = complex(-0.,0.)
            pair.append(a)
        original.append(pair)
    store, receipt = prepare_radial_sample_store(mesh,root=directory,shape=shape,
                                                  packets=2,resident_bytes=0.)
    assert store is not None and receipt['output_records'] == 4
    try:
        for i,(ps,ae) in enumerate(original):
            write_radial_samples(store,i,put(ps),put(ae),plan,packed)
            if i == 0:
                refuses(lambda:read_radial_samples(store,0,plan,packed),'incomplete')
        current = dict(identity='field',angular_support_identity='angular',
            source_frame_policy='same_actual_served_four_spinor_full_WFN_Lowdin',
            endpoint_source_binding={'physical_bands':4,'frame_sha256':'original'},
            q_full_indices=np.arange(3),plan=plan,phase_prices={})
        deferred = DeferredCurrentDensity(current,{},source=None,coefficients=(),
                                           parameters={},samples=store)
        deferred.bind_charge_zeta(SimpleNamespace(mesh=mesh,L_q=None,lu_piv=None,
            local_augmentation=None,shell=None,q0_response_rhs=None))
        deferred._authenticate()
        current['endpoint_source_binding']['frame_sha256'] = 'different-actual-frame'
        refuses(deferred._authenticate,'source-frame binding changed')
        current['endpoint_source_binding']['frame_sha256'] = 'original'
        moved = points.copy(); moved[0,0] += .013
        other = build_centroid_k_unfold_plan(sym,moved,(4,4,4),mesh,nspinor=4,
            parent_k_frac=kpar,coordinate_kind='fractional')
        refuses(lambda:read_radial_samples(store,0,other,
            other.layout.axis.pack_host(moved,axis=0,fill_value=0.)),'binding changed')
        from dataclasses import replace
        spin = np.array(plan.spin_action_full,copy=True)
        spin[1] *= -1
        changed_spin = replace(plan,spin_action_full=spin)
        refuses(lambda:read_radial_samples(store,0,changed_spin,packed),'binding changed')
        rows = np.array([2,0,1],np.int32)
        select,_ = _rhs_storage_kernels(mesh,np.arange(3),4,1,1,npnt,npnt,npnt)
        to_q = face_to_batch_reshard(mesh)
        left = put(original[0][1])
        left_faces = (left,_face_complement(left,mesh))
        circular = np.array([[1,1j,0],[1,-1j,0],[0,0,2**.5]],complex)/2**.5
        for i,(ps,ae) in enumerate(original):
            saved_ps,saved_ae = read_radial_samples(store,i,plan,packed)
            exact(host(saved_ps),ps); exact(host(saved_ae),ae)
            source_ps,source_ae = put(ps),put(ae)
            old_faces = ((source_ae,_face_complement(source_ae,mesh)),
                         (source_ps,_face_complement(source_ps,mesh)))
            new_faces = ((saved_ae,_face_complement(saved_ae,mesh)),
                         (saved_ps,_face_complement(saved_ps,mesh)))
            for old,new in zip(old_faces,new_faces):
                exact(host(old[1]),host(new[1]))
            for channel in (1,2,3):
                rhs = []
                for af,pf in (old_faces,new_faces):
                    value = local_density_rhs(centroid_faces=left_faces,atom_ae_faces=af,
                        atom_ps_faces=pf,left_plan=plan,right_plan=plan,
                        weight_l=np.array([1.,1.,0.,0.]),weight_r=np.array([0.,1.,1.,0.]),
                        kgrid=(3,1,1),mesh_xy=mesh,q_neg_idx=q_negation_index((3,1,1)),
                        q_indices=rows,vertex_terms=current_fit_terms(channel,circular))
                    rhs.append(to_q(select(value)))
                exact(host(rhs[0]),host(rhs[1]))
                d = np.arange(1,npnt+1)+channel
                factor = device_put_process_local(np.repeat(
                    np.diag(1/np.sqrt(d))[None],4,axis=0).astype(complex),qs)
                solve = _local_coefficient_solve(mesh,'positive_rank_truncate',npnt,3)
                exact(host(solve(factor,rhs[0])),host(solve(factor,rhs[1])))
        assert store['next_read'] == 2
        refuses(lambda:read_radial_samples(store,0,plan,packed),'binding changed')
        # Corrupt actual stored bytes on every rank; the tier must refuse after
        # its one authenticated re-read. No resampling fallback is involved.
        store['next_read'] = 0
        for local in store['bank'].stores.values():
            data = bytearray(store['bank'].records[0])
            local.read(memoryview(data),0); data[0] ^= 1
            local.write(memoryview(data),0)
        refuses(lambda:read_radial_samples(store,0,plan,packed),'write digests')
    finally:
        store['bank'].release()
    # Explicit storage-unavailability refusal, collectively resolved before any
    # source can be discarded. No failure injection enters production code.
    free = _slab_io_rank._free_bytes
    _slab_io_rank._free_bytes = lambda root:0
    try:
        declined, fallback = prepare_radial_sample_store(mesh,root=directory,
            shape=shape,packets=2,resident_bytes=0.)
    finally:
        _slab_io_rank._free_bytes = free
    assert declined is None and fallback['mode'] == 'resample'
    assert 'capacity' in fallback['reason']
    result = dict(scope='P4 exact stored endpoints, complements, three RHS/OWN solves and refusals',
        byte_identity=True,packets=2,channels=3,storage=receipt,fallback=fallback,
        controls=['changed physical point','changed spin transport','changed source frame','incomplete records',
                  'repeated read','corrupt bytes','collective disk refusal'],
        material_capacity_or_speed_claim=False)
    if jax.process_index() == 0:
        Path(directory).mkdir(parents=True,exist_ok=True)
        (Path(directory)/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    parser = argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    runtime = initialize_communicator_stack()
    def main():
        check_radial_sample_reuse(runtime,args.output)
        return 0
    run_main_and_finalize(main)
