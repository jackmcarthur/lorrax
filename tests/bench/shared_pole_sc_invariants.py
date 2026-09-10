"""Observe a real scalar SC replay before projections and Gram reduction.

This diagnostic wraps existing Python stage boundaries, returning their
unchanged results. Large wavefunctions and projector matrices remain sharded
over both mesh axes. Only per-k/per-frequency scalar diagnostics reach JSON.
No sampled-centroid overlap is identified with the physical Hilbert metric.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--entry', type=Path,
                        help='reconstruct a saved entering rotation; inspect state and stop before W/Sigma')
    args = parser.parse_args()
    # The real driver owns runtime initialization and all physics execution.
    from gw import gw_jax as driver
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host
    from distrib_la import gemm_plan
    from gw import response_bank as rb, sc_iteration as sc

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'invariants.jsonl'
    records = []
    contexts = {}
    rotations = 0
    map_inputs = None

    class StateObserved(Exception):
        pass

    def emit(kind, **fields):
        row = dict(kind=kind, job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], **fields)
        records.append(row)
        if jax.process_index() == 0:
            with path.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print('SC_INVARIANT '+json.dumps(row, allow_nan=False), flush=True)

    def host(value):
        return np.asarray(gather_to_host(value)).tolist()

    @jax.jit
    def matrix_stats(a):
        adj = jnp.conj(jnp.swapaxes(a, -1, -2))
        norm = jnp.linalg.norm(a, axis=(-2, -1))
        defect = jnp.linalg.norm(a-adj, axis=(-2, -1))
        trace = jnp.trace(a, axis1=-2, axis2=-1)
        return dict(norm=norm, antihermitian_abs=defect,
                    antihermitian_relative=defect/jnp.maximum(norm, 1e-300),
                    trace_real=jnp.real(trace), trace_imag=jnp.imag(trace),
                    max_abs=jnp.max(jnp.abs(a), axis=(-2, -1)))

    @jax.jit
    def difference(a, b):
        return (jnp.linalg.norm(a-b, axis=(-2, -1)) /
                jnp.maximum(jnp.linalg.norm(b, axis=(-2, -1)), 1e-300))

    def stats(a):
        return {k: host(v) for k, v in matrix_stats(a).items()}

    def face_checks(old, new, *, label):
        """Preserve sum_n |psi_n><psi_n| under the full band rotation."""
        if old.psi_mun is None or new.psi_mun is None:
            return dict(label=label, status='ABSENT')
        assert old.psi_mun.shape[1] == 1
        nk, _, mu, nb = old.psi_mun.shape
        product = gemm_plan(mesh, m=mu, k=nb, n=mu, nq=nk, dtype=jnp.complex128)
        face = NamedSharding(mesh, P(None, 'x', 'y'))

        @jax.jit
        def projector(left, right):
            # Both stored orientations contain psi, not psi conjugate.
            return jax.lax.with_sharding_constraint(
                product(left[:, 0], jnp.conj(right[:, :, 0])), face)

        @jax.jit
        def orientation(left, right):
            peer = jnp.swapaxes(right[:, :, 0], -1, -2)
            return difference(left[:, 0], peer)

        before = projector(old.psi_mun, old.psi_nmu)
        after = projector(new.psi_mun, new.psi_nmu)
        return dict(label=label, shape=list(old.psi_mun.shape),
                    full_projector_relative=host(difference(after, before)),
                    orientation_before=host(orientation(old.psi_mun, old.psi_nmu)),
                    orientation_after=host(orientation(new.psi_mun, new.psi_nmu)),
                    projector_before=stats(before), projector_after=stats(after),
                    scope='centroid full-band projector; not occupied density or physical normalization')

    original_rotate = sc.rotate_wavefunctions

    def state_probe(old, new, inputs):
        """Gamma spectral projectors and raw time samples, no quadrature."""
        meta = inputs.meta
        assert int(meta.nspinor) == 1 and bool(inputs.sym.trs_allowed)
        # This bounded fixture is the unshifted scalar-Si 4^3 replay.
        assert (int(meta.nkx),int(meta.nky),int(meta.nkz)) == (4,4,4)
        mu, nb = old.psi_mun.shape[2:]
        assert mu <= 400 and nb <= 36
        product = gemm_plan(mesh,m=mu,k=nb,n=mu,nq=1,dtype=jnp.complex128)
        face = NamedSharding(mesh,P(None,'x','y'))

        @jax.jit
        def gamma(left,right,weight):
            return jax.lax.with_sharding_constraint(
                product(left[:1,0]*weight[:1,None,:],jnp.conj(right[:1,:,0])),face)

        @jax.jit
        def imaginary_norm(a):
            return jnp.linalg.norm(jnp.imag(a))/jnp.maximum(jnp.linalg.norm(a),1e-300)

        for name,wfns in (('dft',old),('saved_sc',new)):
            energy,f,u,reference,census = rb.response_weights(wfns,meta)
            weights = dict(all=f+u,occupied=f,all_energy=(f+u)*(energy-reference),
                           occupied_energy=f*(energy-reference))
            for label,weight in weights.items():
                value = gamma(wfns.psi_mun,wfns.psi_nmu,
                              rb.stream_weights(wfns,weight,mesh))
                emit('gamma_spectral_projector',state=name,weight=label,
                     imaginary_relative=float(imaginary_norm(value)),matrix=stats(value))
            qids = tuple(map(int,np.asarray(inputs.sym.q_irr_full_idx)))
            kernel,fixed = rb.response_stream(wfns,meta,mesh_xy=mesh,q_ids=qids,n_outputs=3)
            common = (jnp.asarray([0.,.3,1.]),jnp.asarray(np.eye(3,dtype=np.complex128)),*fixed,
                      rb.stream_weights(wfns,f,mesh).astype(jnp.complex128))
            values = kernel(*common,rb.stream_weights(wfns,u,mesh).astype(jnp.complex128),
                            jnp.asarray(reference))
            # -i on the upper weight turns the t=0 difference into the sum,
            # exactly the existing bare-moment producer's convention.
            positive = kernel(*common,rb.stream_weights(wfns,-1j*u,mesh),jnp.asarray(reference))
            numerator = jnp.linalg.norm(values[:,0],axis=(-2,-1))
            denominator = jnp.linalg.norm(positive[:,0],axis=(-2,-1))
            emit('raw_time_response',state=name,qids=qids,time_ry_inverse=[0.,.3,1.],
                 matrix=stats(values),equal_time_relative=host(numerator/jnp.maximum(denominator,1e-300)),
                 scope='existing Green/FFT kernel at individual times; no integration, Dyson solve or pole construction')

    def rotate(old, u, **kwargs):
        nonlocal rotations
        if args.entry is not None:
            from common.collectives import device_put_process_local
            with np.load(args.entry) as saved:
                a_lo,a_hi = map(int,saved['active'])
                u = device_put_process_local(saved['U'],NamedSharding(mesh,P(None,'x','y')))
                kwargs['enk_base'] = device_put_process_local(saved['enk'],NamedSharding(mesh,P()))
                kwargs['enk_active_new'] = kwargs['enk_base'][:,a_lo:a_hi]
                kwargs['active_slice'] = slice(a_lo,a_hi)
                kwargs['efermi'] = float(sc._midgap_efermi(saved['enk'][:,:a_hi],int(map_inputs.meta.nelec)))
        new = original_rotate(old, u, **kwargs)
        # Small band-space arrays only; no wavefunction/Green host gather.
        uh = np.asarray(gather_to_host(u))
        active = kwargs.get('active_slice') or old.slices.sigma
        lo, hi = int(active.start or 0), int(active.stop)
        logical = min(hi, int(old.slices.b4_logical)-int(old.slices.b0))-lo
        unit = uh[:, :logical, :logical]
        unit_error = np.linalg.norm(unit.conj().transpose(0, 2, 1) @ unit-np.eye(logical), axis=(-2, -1))
        f0 = np.asarray(gather_to_host(old.occ))
        f1 = np.asarray(gather_to_host(new.occ))
        e1 = np.asarray(gather_to_host(new.enk))
        # The Si diagnostic's small band-space state is sufficient to
        # rebuild the entering QP carrier from the immutable DFT restart.
        # This is <= nk*nb**2*16 host bytes, never a wavefunction gather.
        if jax.process_index() == 0:
            np.savez(args.output/f'entry_rotation_{rotations:04d}.npz',
                     U=uh, enk=e1, occ=f1, active=np.asarray([lo,hi]))
        count = int(old.slices.b4_logical)-int(old.slices.b0)
        order = np.argsort(e1[:, :count], axis=1)
        sorted_f = np.take_along_axis(f1[:, :count], order, axis=1)
        emit('rotation', index=rotations, active=[lo,hi], logical_active=logical,
             unitarity_fro_per_k=unit_error.tolist(),
             occupied_count_before_per_k=f0[:, :count].sum(axis=1).tolist(),
             occupied_count_after_per_k=f1[:, :count].sum(axis=1).tolist(),
             occupation_min=float(f1[:, :count].min()), occupation_max=float(f1[:, :count].max()),
             upward_occupation_increase=float(np.max(np.diff(sorted_f, axis=1))),
             full=face_checks(old,new,label='full_k'),
             parent=(face_checks(old.green_parent,new.green_parent,label='raw_parent')
                     if old.green_parent is not None else None))
        if new.green_parent is not None and new.psi_mun is not None:
            ids = np.asarray(new.green_parent.plan.parent_full_rows, dtype=np.int32)
            @jax.jit
            def agreement(full, parent):
                selected = jnp.take(full, ids, axis=0)
                return jnp.linalg.norm(selected-parent) / jnp.maximum(jnp.linalg.norm(parent), 1e-300)
            emit('parent_binding', index=rotations,
                 psi_mun_relative=float(agreement(new.psi_mun,new.green_parent.psi_mun)),
                 psi_nmu_relative=float(agreement(new.psi_nmu,new.green_parent.psi_nmu)),
                 energy_max_abs=float(jnp.max(jnp.abs(new.enk[ids]-new.green_parent.enk))),
                 occupation_max_abs=float(jnp.max(jnp.abs(new.occ[ids]-new.green_parent.occ))))
        rotations += 1
        if args.entry is not None:
            state_probe(old,new,map_inputs)
            raise StateObserved('saved SC state inspected before screening')
        return new

    original_produce = rb.produce_sample_bank

    def produce(wfns, meta, config, **kwargs):
        bank = str(kwargs['bank_io']['path'])
        z = rb.bank_points(kwargs['sample_plan'])
        contexts[bank] = dict(z=z, qids=np.asarray(kwargs['sym'].q_irr_full_idx).tolist())
        emit('bank_context', bank=bank, z_ry=np.stack([z.real,z.imag],axis=-1).tolist(),
             qids=contexts[bank]['qids'], census=rb.response_weights(wfns,meta)[-1])
        return original_produce(wfns,meta,config,**kwargs)

    original_execution = rb._bank_execution

    def execution(meta, mesh_xy, bank_io, receipt, config):
        execute = original_execution(meta,mesh_xy,bank_io,receipt,config)
        bank = str(bank_io['path'])
        sample_q = 0
        moment_q = 0

        def observed(kernel, values, stage):
            nonlocal sample_q, moment_q
            # Measure the raw inputs before the native operation, and return
            # exactly its original result. No Hermitian repair or reordering.
            before = None
            if stage == 'sample_dyson':
                before = dict(chi=stats(values[1]), dchi_ds=stats(values[2]), coulomb_root=stats(values[0]))
            elif stage == 'moment_dyson':
                before = dict(A0=stats(values[1]), A1=stats(values[2]))
            result = execute(kernel,values,stage)
            if stage == 'sample_dyson':
                context = contexts[bank]
                # This replay requires a full frequency panel, and refuses
                # to label partial-panel indices as complete samples.
                assert values[1].shape[0] == len(context['z'])
                emit('sample_dyson', bank=bank, q=int(context['qids'][sample_q]),
                     before=before, Wc=stats(result[0]), dWc_ds=stats(result[1]))
                sample_q += 1
            elif stage == 'moment_dyson':
                emit('moment_dyson', bank=bank, q_index=moment_q, before=before,
                     M1=stats(result[0]), M3=stats(result[1]))
                moment_q += 1
            return result
        return observed

    # Validate sensitivity without changing a production operand.
    from common.collectives import device_put_process_local
    good = np.eye(4, dtype=np.complex128)[None]
    bad = good.copy(); bad[0,0,1] = 1e-5
    spec = NamedSharding(mesh,P(None,'x','y'))
    negative = stats(device_put_process_local(bad,spec))
    assert negative['antihermitian_relative'][0] > 1e-6
    emit('negative_control', asymmetric_matrix=negative,
         scaled_unitary_defect=float(np.linalg.norm((1.0001*np.eye(4)).T @ (1.0001*np.eye(4))-np.eye(4))))

    sc.rotate_wavefunctions = rotate
    original_map = sc.gw_iteration_map

    def observe_map(state, inputs):
        nonlocal map_inputs
        map_inputs = inputs
        return original_map(state,inputs)

    sc.gw_iteration_map = observe_map
    rb.produce_sample_bank = produce
    rb._bank_execution = execution
    try:
        driver.main(['-i', args.input])
    except StateObserved as exc:
        emit('terminal',status='STATE_OBSERVED',message=str(exc),entry=str(args.entry))
    except Exception as exc:
        emit('terminal', status='DRIVER_REFUSED', exception=type(exc).__name__, message=str(exc))
        if 'shared_pole_gram_valid' not in str(exc):
            raise
    else:
        emit('terminal', status='DRIVER_COMPLETED')
    finally:
        sc.rotate_wavefunctions = original_rotate
        sc.gw_iteration_map = original_map
        rb.produce_sample_bank = original_produce
        rb._bank_execution = original_execution


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
