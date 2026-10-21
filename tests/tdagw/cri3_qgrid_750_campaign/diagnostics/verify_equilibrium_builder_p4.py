"""Native P4 padded chi + same-state EQ frequency-owner software proof.

Not a material W admission, TD trajectory, or screening-cycle timing.
"""
from __future__ import annotations


def equilibrium_stage(c):
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from ffi import ffi_dial_key
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.response_bank import _bank_execution
    from gw.shared_pole_pencil import join_vectors
    from gw.shared_pole_screening import (_general_frequency_programs,
        _equilibrium_frequency_programs, _equilibrium_definite_eligibility)
    from gw.shared_pole_constructor import collective_general_pencil, EQUILIBRIUM_METRIC_SOLVER
    from verify_common_coherent_chi_p4 import bank_config

    mesh = c['mesh']; grid = c['coarse']; nq = int(np.prod(grid)); mu = c['mu']
    coords = np.indices(grid).reshape(3, -1).T
    minus = np.ravel_multi_index(((-coords) % grid).T, grid)
    signature = np.diag([1., 1., -1., -1.]); gamma = .037
    sites = np.repeat([.21+.18j, -.21+.18j, .47j, .72+.22j, -.72+.22j], 2)
    rng = np.random.default_rng(20261021)
    plus = rng.normal(size=(nq, mu, 2))+1j*rng.normal(size=(nq, mu, 2))
    omega = rng.uniform(.6, 1.9, (nq, 2))
    directions = rng.normal(size=(nq, mu, len(sites)))+1j*rng.normal(size=(nq, mu, len(sites)))
    qi = rng.normal(size=(nq, mu, 2))+1j*rng.normal(size=(nq, mu, 2))
    side = int(mesh.shape['y'])
    order = join_vectors((np.arange(len(sites)), np.arange(2)+len(sites)), side)
    units = join_vectors((np.ones(len(sites)), np.full(2, .5)), side)
    host = lambda a: np.asarray(gather_to_host(a))
    put = c['put']; key = (mesh, P(None, 'x', 'y'), ffi_dial_key())
    common, gram = _equilibrium_frequency_programs(*key)
    _, infinity, adjoint, _ = _general_frequency_programs(*key)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=1, nk_tot=64, nq_tot=nq, nspinor=2,
                          n_rmu=mu, qgrid=grid, qgrid_response_window='fejer')
    meta = fresh_shared_pole_capacity(meta, mesh); meta.shared_pole_capacity.at_phase()
    row = meta.shared_pole_capacity.reserve('eq_software_matrices',
        resident_bytes_per_rank=16*nq*20*(len(sites)+2)**2//mesh.size, workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages = (row['stage'],)
    execution = dict(seconds={}, memory=[], compiled=[], purpose='native frequency GEMM programs')
    execute = _bank_execution(meta, mesh, execution, bank_config())
    maximum = 0.; gram_error = 0.; reports = []; previous = None
    held_columns = {}
    for current in range(2):
        cp = plus*(1+.05*current*np.sin(np.arange(nq))[:, None, None])
        coupling = np.array([np.concatenate((cp[q], cp[minus[q]].conj()), -1) for q in range(nq)])
        metric = np.array([np.diag(np.r_[omega[q], omega[minus[q]]]+.13*current) for q in range(nq)])
        def exact(q, z):
            return coupling[q]@np.linalg.solve((z+1j*gamma)*signature-metric[q], coupling[q].conj().T)
        output = np.array([np.stack([exact(q, z)@directions[q, :, i]
            for i, z in enumerate(sites)], -1) for q in range(nq)])
        zeta = sites+1j*gamma
        states = np.array([np.stack([np.linalg.solve(z*signature-metric[q],
            coupling[q].conj().T@directions[q, :, i]) for i, z in enumerate(zeta)], -1) for q in range(nq)])
        c1 = coupling@signature@coupling.conj().swapaxes(-1, -2)
        c2 = coupling@signature@metric@signature@coupling.conj().swapaxes(-1, -2)
        assert min(np.linalg.norm(v) for v in c1) > .01
        zd, qd, od = put(zeta, P()), put(directions, P(None, 'x', 'y')), put(output, P(None, 'x', 'y'))
        g, h, b, out = execute(common, (zd, qd, od), 'direct')
        # The production owner reuses the general infinity program, with
        # exactly ONE undamping of physical C2_gamma before the state join.
        c2_gamma = c2-1j*gamma*c1
        args = (zd.conj(), zd, g, h, b, out, qd, adjoint(od), qd, od,
                put(qi, P(None, 'x', 'y')), put(c1, P(None, 'x', 'y')),
                put(c2_gamma+1j*gamma*c1, P(None, 'x', 'y')))
        g, h, b, out = execute(infinity, args, 'direct')
        output_gram = execute(gram, (out,), 'direct')
        gh, hh, nh, oh = map(host, (g, h, output_gram, out))
        # Only this declared tiny four-port software fixture is host-read;
        # no material/source microscopic matrix or transition bank is read.
        states = np.concatenate((states, signature@coupling.conj().swapaxes(-1, -2)@qi), -1)[..., order]
        expected_g = states.conj().swapaxes(-1, -2)@signature@states
        expected_h = states.conj().swapaxes(-1, -2)@metric@states
        for actual, target in ((gh, expected_g), (hh, expected_h)):
            error = np.linalg.norm(actual-target)/np.linalg.norm(target)
            gram_error = max(gram_error, float(error)); assert error < 2e-12
        np.testing.assert_allclose(nh, oh.conj().swapaxes(-1, -2)@oh, rtol=2e-12, atol=2e-11)
        structure = dict(kind='equilibrium_definite', gamma_ry=gamma,
            state_selection='principal-rrqr-v1', state_units=units, frequency_scale_ry=2.,
            metric_solver=EQUILIBRIUM_METRIC_SOLVER)
        # q=0 and a genuinely distinct q/-q pair use the REAL all-rank
        # canonical host-owner handoff, not independent per-rank LAPACK.
        values = {}
        for iq in (0, 1, int(minus[1])):
            model = collective_general_pencil(gh[iq], hh[iq], max_order=len(units),
                rank_tolerance=1e-8, structure=structure, output_gram=nh[iq],
                fixed_spaces=held_columns.get(iq))
            receipt = model['receipt']
            assert receipt['metric_solver'] == EQUILIBRIUM_METRIC_SOLVER
            assert receipt['space_representation'] == 'original-column-selector-v1'
            assert receipt['fixed_congruence_count'] == 2
            assert max(receipt['original_H_extended_whitening_errors']) <= 1e-9
            columns = model['spaces']
            np.testing.assert_array_equal(columns['right'], columns['left'])
            selector = np.asarray(columns['right'])
            assert selector.shape == (len(units), 4)
            indices = np.argmax(abs(selector), axis=0)
            assert len(np.unique(indices)) == 4
            expected_selector = np.zeros_like(selector)
            expected_selector[indices, np.arange(4)] = 1
            np.testing.assert_array_equal(selector, expected_selector)
            if current == 0:
                assert receipt['state_selection']['selected_order'] == 4
                held_columns[iq] = {name: np.asarray(value).copy() for name, value in columns.items()}
            else:
                assert receipt['state_selection'] is None
                np.testing.assert_array_equal(selector, held_columns[iq]['right'])
            np.testing.assert_array_equal(model['poles_ry'].imag, np.full(4, -gamma))
            l, r = oh[iq]@model['output_map'], oh[iq]@model['input_map']
            values[iq] = []
            for z in (.31+.04j, -.43+.2j, 1.5+.08j):
                value = (l/(z-model['poles_ry']))@r.conj().T
                error = np.linalg.norm(value-exact(iq, z))/np.linalg.norm(exact(iq, z))
                maximum = max(maximum, float(error)); assert error < 2e-9
                values[iq].append(value)
            reports.append(dict(current=current, q=iq, receipt=model['receipt']))
        z = .31+.04j
        assert np.linalg.norm(exact(1, z)-exact(1, -z)) > .01
        np.testing.assert_allclose(exact(1, z), exact(int(minus[1]), -z.conjugate()).conj(),
                                   rtol=2e-13, atol=2e-13)
        if previous is not None:
            assert np.linalg.norm(values[0][0]-previous) > 1e-4
        previous = values[0][0]
    # A resolved indefinite current metric cannot be hidden by RRQR.
    try:
        collective_general_pencil(np.diag([1., -1.]), np.diag([1., -.1]), max_order=2,
            rank_tolerance=1e-8, output_gram=np.eye(2), structure=dict(kind='equilibrium_definite',
                gamma_ry=gamma, state_selection='principal-rrqr-v1', state_units=np.ones(2),
                frequency_scale_ry=1., metric_solver=EQUILIBRIUM_METRIC_SOLVER))
    except ValueError as error:
        if ('GATE shared_equilibrium_metric:' not in str(error)
                or 'negative H' not in str(error) or 'no PSD repair' not in str(error)):
            raise
        refusal = str(error)
    else:
        raise AssertionError('resolved negative H was not refused')
    e = np.broadcast_to([-.8, -.4, .4, .9], (64, 4)).copy()
    rho = np.broadcast_to(np.diag([1., 1., 0., 0.]), (64, 4, 4)).astype(complex).copy()
    assert _equilibrium_definite_eligibility(e, rho, 4, c['grid'], grid, 'fejer')['eligible']
    rho[:, 1, 2] = 1e-30; rho[:, 2, 1] = 1e-30
    assert not _equilibrium_definite_eligibility(e, rho, 4, c['grid'], grid, 'fejer')['eligible']
    return dict(max_full_complex_held_relative_error=maximum, latent_gram_relative_error=gram_error,
        current_inputs=2, nonzero_C1=True, distinct_q_crossing=True, real_zeta_poles=True,
        canonical_structured_handoff=True, original_state_RRQR=True, tiny_host_port_extent=mu,
        metric_solver=EQUILIBRIUM_METRIC_SOLVER, exact_original_selector_held_across_current_H=True,
        material_microscopic_host_gather=False, resolved_negative_H_refusal=refusal,
        eigenmodels=reports, execution=execution)


def combined_stage(c):
    from verify_complex_box_chi_p4 import response_stage
    return dict(padded_complex_chi=response_stage(c), equilibrium_frequency=equilibrium_stage(c))


if __name__ == '__main__':
    from verify_complex_box_chi_p4 import main
    raise SystemExit(main(combined_stage))
