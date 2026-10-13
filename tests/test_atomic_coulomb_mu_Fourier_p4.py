"""Exact mu-balanced charge FT against physical quadrature and complete V.

Canonical P4 fixture: small-q domains, coherent served M0, complex atomic
phases, unequal G counts, poisoned q/mu padding and physical writer layout.
This is a numerical owner/functional proof, not material or timing admission.
"""
from pathlib import Path
import json
import os


def check_mu_Fourier(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common import gpu_utils
    from isdf.atomic_coulomb import radial_coulomb_provider, atomic_radial_metrics, _smooth_neutral_tables
    from isdf.augmentation import local_basis_fourier
    from isdf.zeta_mubatch import ZetaG, ZStore, _local_coefficient_solve
    from runtime.padding import padded_axis
    from common.centroid_basis import PackedCentroidBasis
    from types import SimpleNamespace

    mesh = runtime.mesh
    assert mesh.size == 4 and runtime.process_count == 4
    # Physical canonical rows interleave three inversion orbits. This is
    # the actual public nonidentity basis, including its two ghost slots,
    # rather than a mocked writer permutation or the former mu_basis=None.
    first = np.asarray([[.11, .17, .23], [.19, .31, .43], [.27, .41, .59]])
    points = np.concatenate((first, 1.-first))
    sym = SimpleNamespace(sym_matrices=np.asarray([np.eye(3), -np.eye(3)], np.int32),
                          translations=np.zeros((2, 3)))
    basis = PackedCentroidBasis.build(points, sym, (4, 4, 6), mesh,
                                     coordinate_kind='fractional')
    literal = np.asarray([0, 3, 2, 5, 1, 4, -1, -1])
    assert basis.n_logical == 6 and basis.n_canonical == basis.n_packed == 8
    assert not basis.is_identity
    np.testing.assert_array_equal(basis.layout.axis.packed_to_canonical, literal)
    inverse = np.argsort(literal[:basis.n_logical])
    def canonical(array, axis):
        # Literal host reference independent of the tested device unpack.
        result = np.take(array, inverse, axis=axis)
        pads = [(0, 0)]*result.ndim
        pads[axis] = (0, basis.n_canonical-basis.n_logical)
        return np.pad(result, pads)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    host = lambda a: np.asarray(gather_to_host(a))
    qs = P(('x', 'y'), None, None)
    results = []
    for Q in (1, 3, 4):
        Qp, mu, na, nr, ng, gt, R = 4, 8, 2, 12, 7, 4, 1.7
        lm = np.asarray([(l, m) for l in range(3) for m in range(-l, l+1)])
        nh, nf = len(lm), na*len(lm)*nr
        rng = np.random.default_rng(78134+Q)
        random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
        r = R*np.linspace((1e-5/R)**(1/3), 1., nr)**3
        w = np.diff(np.r_[0., (r[:-1]+r[1:])/2, R])
        B = random((Qp, mu, mu))*.1
        B[Q:] = 0.; B[:, -2:] = 0.
        rhs, exact = random((Qp, mu, nf))*.004, random((Qp, mu, na))*.006
        rhs[Q:] = 1e7-3e7j; exact[Q:] = 3e7+1e8j
        rhs[:, -2:] = 1e6+2e6j; exact[:, -2:] = 5e6-1e6j
        K = rng.normal(size=(Q, ng, 3))*.8; K[0, 0] = 0.
        counts = np.asarray([7, 5, 6, 4][:Q], np.int32)
        centers = np.asarray([[.1, -.3, .7], [5.7, .4, -.2]])
        volume, N = 1728., 160
        store = ZStore(mesh=mesh, q_axis=padded_axis(Q, 4, name='mu FT q'),
            mu_pad=mu, g_axis=padded_axis(ng, gt, name='mu FT G'), b=mu, placement='host')
        zeta = ZetaG(store, mesh=mesh, L_q=put(B, qs), lu_piv=None,
            solver_kind='rank_truncate', batched_route='batch_reshard',
            n_rmu_solve=mu, n_rmu=basis.n_logical, mu_basis=basis, ngk_per_q=counts,
            gvec_components=np.zeros((Q, 3, ng), np.int32), path='physical charge fixture')
        provider = radial_coulomb_provider(zeta, put(rhs, qs), monopole_rhs=put(exact, qs),
            radius=r, weights_dr=w, lm=lm, centers_cart=centers, q_plus_G_cart=K,
            cell_volume=volume, fft_points=N, support_radius=R,
            interpolation_degree=5, quadrature_order=16)
        coefficients = _local_coefficient_solve(mesh, zeta.solver_kind, mu, Q)(
            zeta.factor, provider['rhs'])
        prepared = provider['prepare_fourier_coefficients'](coefficients)
        assert provider['fourier_distribution_receipt']['mode'] == ('mu' if Q < 4 else 'q')
        if Q < 4:
            assert prepared.shape == (Q, mu, nf+na)
            assert prepared.sharding.is_equivalent_to(NamedSharding(mesh, P(None, ('x', 'y'), None)), 3)
            quote = next(v for v in provider['fourier_distribution_receipt']['actual_AOT']
                         if v['owner'] == 'mu_Fourier')
            assert not any(quote['collective_counts'].values()), quote
        else:
            assert prepared is coefficients
        solved = B@(B.conj().swapaxes(-1, -2)@np.concatenate((rhs, exact), axis=-1))
        solved[Q:] = 0.
        assert np.max(abs(host(coefficients)-solved)) < 3e-14
        d = solved[..., :nf].reshape(Qp, mu, na, nh, nr)
        e = solved[..., nf:]
        tables = atomic_radial_metrics(r, w, lm[:, 0], support_radius=R,
            fft_points=N, cell_volume=volume, interpolation_degree=5, quadrature_order=16)
        cross = _smooth_neutral_tables(tables, support_radius=R, fft_points=N, cell_volume=volume)
        qr, qw = tables['quadrature_radius'], tables['quadrature_weights_dr']
        delta_r = np.zeros((Qp, mu, na, nh, len(qr)), complex)
        comp_r = np.zeros_like(delta_r)
        onsite = np.zeros((Qp, mu, mu), complex)
        for atom in range(na):
            for h, (ell, _) in enumerate(lm):
                row = int(ell); values = d[:, :, atom, h]
                sample_map = tables['interpolation_map'].copy()
                sample_map[:tables['origin_row_count'], 0] = tables['origin_factors'][row]
                field = values@sample_map.T
                moment = values@tables['moments'][row]
                ordinary = np.einsum('qmr,rt,qnt->qmn', values.conj(), tables['delta_metric'][row], values)
                gself = 2*(N/volume)**2*tables['compensation_self'][row]
                if h == 0:
                    epsilon = e[:, :, atom]-moment
                    field += epsilon[..., None]*tables['compensation_quadrature_shapes'][row]
                    dg = values@cross['smooth_compensation_cross'][row]
                    ordinary += (epsilon.conj()[:, :, None]*dg[:, None, :]
                        +dg.conj()[:, :, None]*epsilon[:, None, :]
                        +gself*epsilon.conj()[:, :, None]*epsilon[:, None, :])
                    moment = e[:, :, atom]
                delta_r[:, :, atom, h] = field
                comp_r[:, :, atom, h] = moment[..., None]*tables['compensation_quadrature_shapes'][row]
                onsite += ordinary-gself*moment.conj()[:, :, None]*moment[:, None, :]
        expected_d = np.zeros((Q, mu, ng), complex); expected_g = np.zeros_like(expected_d)
        for q in range(Q):
            n = int(counts[q])
            for atom in range(na):
                expected_d[q, :, :n] += local_basis_fourier(delta_r[q, :, atom], qr, qw, lm,
                    K[q, :n], center_cart=centers[atom])*(N/volume)
                expected_g[q, :, :n] += local_basis_fourier(comp_r[q, :, atom], qr, qw, lm,
                    K[q, :n], center_cart=centers[atom])*(N/volume)
        layout_error = quadrature_error = 0.
        for t in range(2):
            original = provider['fourier_tile'](t, coefficients)
            balanced = provider['fourier_tile'](t, prepared)
            for part, (old, new) in enumerate(zip(original, balanced)):
                old, new = host(old), host(new)
                layout_error = max(layout_error, float(np.max(abs(old-new))))
                assert np.count_nonzero(new[Q:]) == np.count_nonzero(new[:, -2:]) == 0
                expected = np.zeros_like(new)
                n = min(gt, ng-t*gt)
                expected[:Q, :, :n] = (expected_d if part == 0 else expected_g)[..., t*gt:t*gt+n]
                quadrature_error = max(quadrature_error, float(np.max(abs(new-expected))))
        assert layout_error < 3e-14 and quadrature_error < 4e-11
        raw_z = random((Q, mu, store.n_G))*.01
        raw_z[:, -2:] = 1e6-1e6j
        store.write_batch(0, put(raw_z, P(None, ('x', 'y'), None)))
        smooth = (B[:Q]@(B[:Q].conj().swapaxes(-1, -2)@raw_z))[..., :ng]
        live = np.arange(ng)[None, :] < counts[:, None]
        smooth *= live[:, None, :]
        v = rng.random((Q, ng))*live
        product = lambda a, b: np.einsum('qmg,qg,qng->qmn', a.conj(), v, b)
        reference = (product(smooth, smooth)+product(smooth, expected_d)
            +product(expected_d, smooth)+product(expected_g, expected_g)+onsite[:Q])
        reference = canonical(canonical(reference, 1), 2)
        physical_reference = canonical(smooth+expected_d, 1)
        # A payload left in packed order must fail the same equality. This
        # proves the nonidentity writer check distinguishes the old seam.
        packed_writer_mismatch = float(np.max(abs(physical_reference-smooth-expected_d)))
        assert packed_writer_mismatch > 1e-7

        class PhysicalWriter:
            def __init__(self): self.payload = np.zeros((Q, mu, ng), complex)
            def write_slab(self, name, value, offset):
                assert name == 'zeta_q_G'
                assert value.sharding.is_equivalent_to(NamedSharding(mesh, P(None, ('x', 'y'), None)), 3)
                first = offset[2]; n = min(gt, ng-first)
                self.payload[..., first:first+n] = host(value)[..., :n]

        completion_inputs = []
        onsite_owner = provider['onsite']
        def record_onsite(values):
            assert not values.is_deleted()
            completion_inputs.append(values)
            return onsite_owner(values)
        provider['onsite'] = record_onsite
        writer = PhysicalWriter(); zeta.local_augmentation = provider
        V = host(zeta.contract_v(v, keep=np.zeros((Q, 1), np.int32), zeta_io=writer))
        matrix_error = float(np.max(abs(V-reference)))
        physical_error = float(np.max(abs(writer.payload-physical_reference)))
        assert matrix_error < 2e-11 and physical_error < 4e-11
        assert not np.any(V[:, basis.n_logical:])
        assert not np.any(V[:, :, basis.n_logical:])
        assert not np.any(writer.payload[:, basis.n_logical:])
        assert np.max(abs(V-V.conj().swapaxes(-1, -2))) < 2e-13
        assert np.max(abs(host(zeta.shell)[..., 0]-writer.payload[..., 0])) < 2e-13
        receipt = dict(provider['fourier_distribution_receipt'])
        assert len(completion_inputs) == 1
        assert completion_inputs[0].is_deleted() == (Q < 4)
        assert not provider['rhs'].is_deleted(), 'raw provider RHS was consumed'
        if Q < 4:
            assert receipt['solved_q_bank_released'] and receipt['raw_rhs_retained']
            assert receipt['post_release_Fourier_peak'] <= gpu_utils.device_budget_bytes()
            assert provider['local_completion_capacity_receipt']['actual_AOT']
            assert zeta.local_stage_seconds['onsite_and_moments'] >= 0.
            assert zeta.local_stage_seconds['G_loop_and_body_finish'] >= 0.
            for quote in receipt['actual_AOT']:
                assert quote['collective_counts']['all-gather'] == 0
            try:
                provider['prepare_fourier_coefficients'](provider['rhs'], consume_q_coefficients=True)
            except ValueError as exc:
                assert 'distinct from the raw RHS' in str(exc)
            else:
                raise AssertionError('raw RHS was accepted as a consumable solve result')
        if Q < 4:
            saved = gpu_utils._RUN_DEVICE_BUDGET_GB
            try:
                gpu_utils.set_device_budget_gb(1e-7)
                declined = provider['prepare_fourier_coefficients'](coefficients)
                assert declined is coefficients and provider['fourier_distribution_receipt']['mode'] == 'q'
            finally:
                gpu_utils.set_device_budget_gb(saved)
        if Q == 1:
            # A write-only pass requires no onsite or periodic result. Its
            # successful consuming transport must still preserve raw RHS.
            completion_inputs.clear()
            writer = PhysicalWriter()
            assert zeta.contract_v(v, keep=np.zeros((Q, 1), np.int32),
                                   zeta_io=writer, with_v=False) is None
            assert not completion_inputs and not provider['rhs'].is_deleted()
            assert provider['fourier_distribution_receipt']['solved_q_bank_released']
            assert np.max(abs(writer.payload-physical_reference)) < 4e-11
        fallback_error = None
        positive_lifecycle_error = None
        if Q == 3:
            # Refuse only the optional redistribution after actual completion
            # pricing. The already prepared onsite result is reused once.
            prepare_owner = provider['prepare_fourier_coefficients']
            def decline_layout(values, **options):
                saved = gpu_utils._RUN_DEVICE_BUDGET_GB
                try:
                    gpu_utils.set_device_budget_gb(1e-7)
                    return prepare_owner(values, **options)
                finally:
                    gpu_utils.set_device_budget_gb(saved)
            provider['prepare_fourier_coefficients'] = decline_layout
            completion_inputs.clear()
            writer = PhysicalWriter()
            fallback = host(zeta.contract_v(v, keep=np.zeros((Q, 1), np.int32), zeta_io=writer))
            fallback_error = float(np.max(abs(fallback-reference)))
            assert fallback_error < 2e-11 and len(completion_inputs) == 1
            assert not completion_inputs[0].is_deleted() and not provider['rhs'].is_deleted()
            assert provider['fourier_distribution_receipt']['mode'] == 'q'
            provider['prepare_fourier_coefficients'] = prepare_owner
            # Exercise retained exact-M0 rows and the actual public distributed
            # periodic action. The finite complex Gram is deliberately planted;
            # this proves lifecycle/order, not a physical Slab completion.
            import jax.numpy as jnp
            from jax import shard_map
            from runtime.padding import pad_to_axis
            from isdf.atomic_coulomb import make_periodic_compensation_action
            from isdf.coulomb_fourier_cache import PERIODIC_SCHEMA
            face = NamedSharding(mesh, P(None, 'x', 'y'))
            moment_axis = padded_axis(na*nh, mesh, name='periodic moment',
                                     specs=((face.spec, 1), (face.spec, 2)))
            radial_rows = put(tables['moments'][lm[:, 0]], P())
            def moment_rows(values, rows):
                density = values[..., :nf].reshape(values.shape[0], mu, na, nh, nr)
                result = jnp.einsum('qmahr,hr->qmah', density, rows)
                result = result.at[:, :, :, 0].set(values[..., nf:])
                return pad_to_axis(result.reshape(values.shape[0], mu, na*nh), moment_axis, axis=2)
            moment_kernel = jax.jit(shard_map(moment_rows, mesh=mesh,
                in_specs=(qs, P()), out_specs=qs, check_vma=False))
            gram0 = random((Q, na*nh, na*nh))*.03
            gram0 = gram0.conj().swapaxes(-1, -2)@gram0
            gram = np.zeros((Q, moment_axis.carrier, moment_axis.carrier), complex)
            gram[:, :na*nh, :na*nh] = gram0
            cache = dict(gram=put(gram, face.spec), moment_axis=moment_axis,
                path='planted finite lifecycle fixture', file_sha256='0'*64,
                metadata=dict(schema=PERIODIC_SCHEMA, logical_shape=[Q, na*nh, na*nh],
                              geometry=dict(cell_volume_bohr3=volume)))
            action, _ = make_periodic_compensation_action(mesh, cache,
                centroid_basis=basis, fft_points=N)
            provider.update(body_metric='physical_low_local_high', bare_v_table=v,
                periodic_cache=cache, periodic_action=action,
                periodic_moment_rows=lambda values: moment_kernel(values, radial_rows))
            provider['aot_metadata']['periodic_moments'] = dict(kernel=moment_kernel,
                                                               table_operands=(radial_rows,))
            mrows = np.einsum('qmahr,hr->qmah', d, tables['moments'][lm[:, 0]])
            mrows[:, :, :, 0] = e
            mrows = mrows[:Q].reshape(Q, mu, na*nh)
            periodic = (N/volume)**2*(mrows.conj()@gram0@mrows.swapaxes(-1, -2))
            physical = smooth+expected_d
            positive_reference = (product(physical, physical)-product(expected_d, expected_d)
                                  +onsite[:Q]+periodic)
            positive_reference = canonical(canonical(positive_reference, 1), 2)
            completion_inputs.clear()
            positive = host(zeta.contract_v(v, keep=np.zeros((Q, 1), np.int32)))
            positive_lifecycle_error = float(np.max(abs(positive-positive_reference)))
            assert positive_lifecycle_error < 2e-11 and len(completion_inputs) == 1
            assert completion_inputs[0].is_deleted() and not provider['rhs'].is_deleted()
            assert {'periodic_moments', 'moments_to_face'} <= {
                row['owner'] for row in provider['local_completion_capacity_receipt']['actual_AOT']}
        results.append(dict(Q=Q, layout_max_abs=layout_error, physical_quadrature_max_abs=quadrature_error,
            complete_mixed_V_max_abs=matrix_error, physical_writer_layout_max_abs=physical_error,
            nonidentity_centroid_basis=True, logical_mu=basis.n_logical,
            canonical_mu=basis.n_canonical, packed_mu=basis.n_packed,
            literal_packed_to_canonical=literal.tolist(),
            omitted_unpack_same_criterion_error=packed_writer_mismatch,
            capacity=receipt, poisoned_pads_zero=True, capacity_decline_keeps_same_q_arithmetic=True))
        results[-1].update(consuming_solved_bank_only=Q < 4,
                           complete_optional_layout_fallback_error=fallback_error,
                           planted_positive_periodic_lifecycle_error=positive_lifecycle_error)
        zeta.close()
    result = dict(status='PASS_P4_NUMERICAL_OWNER_AND_MIXED_FUNCTIONAL_ONLY', cases=results,
        scope='Exact Fourier distribution, coherent M0 and physical writer transport. No material, Slab periodic completion, full4262 capacity or <=2x timing admission.')
    if jax.process_index() == 0:
        print(json.dumps(result), flush=True)
        Path(os.environ['CHARGE_MU_FOURIER_REPORT']).write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: (check_mu_Fourier(runtime), 0)[1])
