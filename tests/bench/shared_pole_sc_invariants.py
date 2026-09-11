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
    parser.add_argument('--first-map-stages', action='store_true',
                        help='observe Gamma band operators across Sigma/H assembly and stop after map zero')
    parser.add_argument('--full-k-stages', action='store_true',
                        help='observe full-BZ band-operator star covariance before wedge selection; stop after map zero')
    parser.add_argument('--synthesis-pairs', action='store_true',
                        help='observe first three W and projected Sigma time samples in map zero')
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
    stage_wfns = None

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
        full_product = gemm_plan(mesh,m=mu,k=nb,n=mu,nq=int(meta.nk_tot),dtype=jnp.complex128)
        face = NamedSharding(mesh,P(None,'x','y'))
        from symmetry_maps import q_negation_index
        negative = jnp.asarray(q_negation_index((4,4,4)))

        @jax.jit
        def full_pairs(left,right,weight):
            value = jax.lax.with_sharding_constraint(
                full_product(left[:,0]*weight[:,None,:],jnp.conj(right[:,:,0])),face)
            peer = jnp.take(value,negative,axis=0).swapaxes(-1,-2)
            norm = jnp.linalg.norm(value,axis=(-2,-1))
            defect = jnp.linalg.norm(value-peer,axis=(-2,-1))
            return defect/jnp.maximum(norm,1e-300),defect,norm

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
                relative,absolute,norm = full_pairs(
                    wfns.psi_mun,wfns.psi_nmu,rb.stream_weights(wfns,weight,mesh))
                emit('full_k_spectral_projector',state=name,weight=label,
                     pair_relative=host(relative),pair_absolute=host(absolute),norm=host(norm),
                     negative_index=host(negative),
                     scope='centroid Bloch-kernel paired transpose, calibrated against immutable DFT state')
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
        nonlocal rotations, stage_wfns
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
        stage_wfns = new
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

    def band_operator(label, value, wfns=None):
        """Gamma kernel psi M psi.H: gauge-independent TRS observation.

        Only one Gamma tile and one narrow face are formed, both on all P.
        This is an operator-symmetry diagnostic, not a physical trace rule.
        """
        if value is None or getattr(value, 'ndim', 0) != 3:
            return
        wfns = stage_wfns if wfns is None else wfns
        mu, nb = wfns.psi_mun.shape[2:]
        assert value.shape[-2:] == (nb, nb) and mu <= 400 and nb <= 36
        right = gemm_plan(mesh,m=nb,k=nb,n=mu,nq=1,dtype=jnp.complex128)
        left = gemm_plan(mesh,m=mu,k=nb,n=mu,nq=1,dtype=jnp.complex128)
        face = NamedSharding(mesh,P(None,'x','y'))

        @jax.jit
        def project(a, psi_x, psi_y):
            p = left(psi_x[:1,0],right(a[:1],jnp.conj(psi_y[:1,:,0])))
            p = jax.lax.with_sharding_constraint(p,face)
            return (jnp.linalg.norm(jnp.imag(p))/jnp.maximum(jnp.linalg.norm(p),1e-300),
                    jnp.linalg.norm(jnp.imag(p)),jnp.linalg.norm(p))

        relative, absolute, norm = project(value,wfns.psi_mun,wfns.psi_nmu)
        emit('gamma_band_operator',label=label,imaginary_relative=float(relative),
             imaginary_absolute=float(absolute),norm=float(norm),matrix=stats(value[:1]))

    original_sigma = sc.compute_sigma_xc
    original_partition = sc._apply_scissor_partition_policy
    original_output_seam = sc._sc_output_tables_on_loop_kset
    full_k_records = []

    def output_seam(sigma_result, delta_h_qp_full, delta_h_qp_unextrap_full,
                    sigma_basis_U_full, exact_hartree_full, kstar):
        if args.full_k_stages:
            from symmetry_maps import KStarMap
            # The physical map is meaningful even when the loop itself uses
            # the identity map. Compare actual full-BZ results before any take.
            physical = KStarMap.from_sym(map_inputs.sym, int(map_inputs.wfn.ntran))
            nk, nb, _ = sigma_basis_U_full.shape
            assert nk == physical.nk_full == 64 and nb <= 36

            @jax.jit
            def residuals(value, reference):
                defect = value-reference
                absolute = jnp.linalg.norm(defect, axis=(-2,-1))
                norm = jnp.linalg.norm(value, axis=(-2,-1))
                return (absolute/jnp.maximum(norm, 1e-300), absolute, norm,
                        jnp.argmax(jnp.abs(defect)), jnp.max(jnp.abs(defect)))

            operators = [('delta_h', delta_h_qp_full),
                         ('delta_h_unextrapolated', delta_h_qp_unextrap_full)]
            operators += [(name, getattr(sigma_result, name)) for name in
                          ('v_h_kij_ry', 'sigma_x_kij_ry', 'sigma_xc_kij_ry')]
            for label, value in operators:
                if value is None or getattr(value, 'ndim', 0) != 3:
                    continue
                assert value.shape == sigma_basis_U_full.shape
                # Independent eigenvectors at different k need not share a
                # QP gauge. Restore the immutable loader's DFT band gauge.
                dft = sc._rotate_to_dft_basis(value, sigma_basis_U_full, mesh=mesh)
                reference = physical.broadcast(physical.select(dft))
                rel, absolute, norm, worst, maximum = residuals(dft, reference)
                rel, absolute, norm = host(rel), host(absolute), host(norm)
                location = np.unravel_index(int(worst), tuple(dft.shape))
                row = dict(label=label, stage='before_sc_output_wedge_selection',
                           basis='immutable_DFT', canonical_star_spread_relative=float(physical.spread_rel(dft)),
                           per_k_relative_fro=rel, per_k_absolute_fro_ry=absolute,
                           per_k_norm_fro_ry=norm, worst_full_k=int(np.argmax(rel)),
                           max_relative_fro=max(rel), max_absolute_entry_ry=float(maximum),
                           worst_entry_kij=[int(v) for v in location],
                           physical_nk_full=physical.nk_full, physical_nk_irr=physical.nk_irr,
                           loop_nk_full=kstar.nk_full, loop_nk_irr=kstar.nk_irr,
                           loop_identity=kstar.is_identity,
                           matrix_shape=list(dft.shape), matrix_entries_per_rank=int(np.prod(dft.shape))//4,
                           scope='Star relation of actual full-BZ band operators; parent little-group invariance is separate')
                emit('full_k_band_star', **row)
                full_k_records.append(row)
                if jax.process_index() == 0:
                    (args.output/'full_k_stages.json').write_text(json.dumps(dict(
                        job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                        rows=full_k_records), indent=2)+'\n')
        return original_output_seam(sigma_result, delta_h_qp_full,
                                    delta_h_qp_unextrap_full, sigma_basis_U_full,
                                    exact_hartree_full, kstar)

    def sigma(*pos, **kwargs):
        result = original_sigma(*pos,**kwargs)
        if args.first_map_stages:
            for field in ('v_h_kij_ry','sigma_x_kij_ry','sigma_xc_kij_ry'):
                band_operator(field,getattr(result,field))
            cube=result.sigma_c_omega_kij_ry
            for i in (0,cube.shape[0]//2,cube.shape[0]-1):
                value=cube[i]
                band_operator(f'sigma_c_omega_{i}_hermitian',.5*(value+value.conj().swapaxes(-1,-2)))
        return result

    def partition(value, *pos, **kwargs):
        if args.first_map_stages:
            band_operator('H_before_partition',value,map_inputs.wfns_dft)
        result = original_partition(value,*pos,**kwargs)
        if args.first_map_stages:
            band_operator('H_after_partition',result[0],map_inputs.wfns_dft)
        return result

    def observe_map(state, inputs):
        nonlocal map_inputs
        map_inputs = inputs
        result = original_map(state,inputs)
        if args.first_map_stages:
            band_operator('kin_ion_dft',inputs.kin_ion_dft,inputs.wfns_dft)
        if args.first_map_stages or args.full_k_stages:
            raise StateObserved('first map Sigma/H stages inspected')
        return result

    sc.gw_iteration_map = observe_map
    rb.produce_sample_bank = produce
    rb._bank_execution = execution
    sc.compute_sigma_xc = sigma
    sc._apply_scissor_partition_policy = partition
    sc._sc_output_tables_on_loop_kset = output_seam
    from gw.mpa import sigma as mpa_sigma
    original_synthesis = mpa_sigma._shared_pole_w_synthesis
    original_tau = mpa_sigma.get_shared_sigma_tau_kernel

    def synthesis(*pos,**kwargs):
        build=original_synthesis(*pos,**kwargs)
        if not args.synthesis_pairs:
            return build
        header=pos[2]
        from symmetry_maps import q_negation_index
        negative=jnp.asarray(q_negation_index(header['grid']))
        calls=0
        @jax.jit
        def pairs(a):
            peer=jnp.take(a,negative,axis=0).swapaxes(-1,-2)
            return jnp.linalg.norm(a-peer,axis=(-2,-1))/jnp.maximum(jnp.linalg.norm(a,axis=(-2,-1)),1e-300)
        def observed(*values):
            nonlocal calls
            result=build(*values)
            if calls<3:
                emit('synthesis_q_pairs',call=calls,relative=host(pairs(result)),
                     time=[float(jnp.real(values[-1])),float(jnp.imag(values[-1]))],
                     rewired=mpa_sigma._shared_pole_fixed_q_policy(header).n_pair_rewired)
            calls+=1
            return result
        return observed

    def tau(**kwargs):
        kernel=original_tau(**kwargs)
        if not args.synthesis_pairs or kwargs.get('w_synthesis') is None:
            return kernel
        calls=0
        seen=set()
        def observed(*values):
            nonlocal calls
            result=kernel(*values)
            if isinstance(result,jax.core.Tracer):
                # The inherited-memory comparison traces a synthetic phased-W
                # kernel. Observe only real execution, never compile tracers.
                return result
            # Selector is constructed once per product window in the owner.
            key=id(values[5])
            if calls<3 or key not in seen:
                band_operator(f'sigma_tau_{calls}_hermitian',.5*(result+result.conj().swapaxes(-1,-2)))
                energy=np.asarray(gather_to_host(values[4]))
                selector=np.asarray(gather_to_host(values[5])).reshape(energy.shape)
                emit('sigma_window_selector',call=calls,energy_gamma=energy[0].tolist(),
                     selector_gamma=selector[0].tolist(),time=[float(jnp.real(values[-1])),float(jnp.imag(values[-1]))])
                band_operator(f'selector_{calls}',jnp.diag(jnp.asarray(selector[0]))[None].astype(jnp.complex128))
                seen.add(key)
            calls+=1
            return result
        return observed
    mpa_sigma._shared_pole_w_synthesis=synthesis
    mpa_sigma.get_shared_sigma_tau_kernel=tau
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
        sc.compute_sigma_xc = original_sigma
        sc._apply_scissor_partition_policy = original_partition
        sc._sc_output_tables_on_loop_kset = original_output_seam
        mpa_sigma._shared_pole_w_synthesis=original_synthesis
        mpa_sigma.get_shared_sigma_tau_kernel=original_tau


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
