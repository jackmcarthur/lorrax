"""Uncached current moments use the native plan and admitted response owner."""
import ast
import builtins
import inspect
import symtable
import numpy as np
import pytest


def test_extracted_fit_has_no_unresolved_globals_and_uses_current_bundle():
    from gw import shared_pole_screening as owner
    source = inspect.getsource(owner)
    symbols = symtable.symtable(source, owner.__file__, 'exec')
    known = set(symbols.get_identifiers()) | set(dir(builtins))
    def check(table):
        assert not [s.get_name() for s in table.get_symbols()
                    if s.is_referenced() and s.is_global() and s.get_name() not in known]
        for child in table.get_children():
            check(child)
    for table in symbols.get_children():
        if table.get_name() in ('_general_current_fit', '_general_current_site_samples',
                                '_general_model_frequency_diagnostic'):
            check(table)
    tree = ast.parse(inspect.getsource(owner.screen_general_shared_poles))
    bundle = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'fit_current' for t in n.targets))
    fields = {k.arg: ast.unparse(k.value) for k in bundle.keywords}
    assert fields['plan'] == 'plan'
    assert fields['meta'] == 'current_meta'
    assert fields['execute_response'] == 'execute_response'


def test_uncached_native_moments_receive_current_plan_meta_and_executor(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import ffi_dial_key, fft as F
    from gw.greens_function_kernel import density_left_endpoint
    from gw.response_bank import full_density_bare_moments
    from gw import response_bank
    from gw.shared_pole_execution import face_matmul
    from gw.shared_pole_screening import _general_current_fit, _general_frequency_programs
    from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture

    case = _native_port_fixture(monkeypatch)
    mesh, meta = case['mesh'], case['meta']
    carrier = case['wfns'].green_parent
    pm, pn, energy_d = carrier.psi_mun, carrier.psi_nmu, carrier.enk
    nk, ns, nmu, nb = pm.shape
    rho = np.eye(nb, dtype=complex)[None] * np.asarray(carrier.occ)[:, :, None]
    # Coherence makes the current raw first moment nontrivial.
    rho[:, 1, 2], rho[:, 2, 1] = .07+.03j, .07-.03j
    ph = density_left_endpoint(mesh, pm.shape, layout=carrier.layout)(pm, jnp.asarray(rho))
    face, rep = NamedSharding(mesh, P(None, 'x', 'y')), NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a, complex), face)
    V = put(np.broadcast_to(.15*np.eye(nmu), (nk, nmu, nmu)))
    qi = put(np.broadcast_to(np.eye(nmu)[:, :1], (nk, nmu, 1)))
    mm = face_matmul(mesh)
    pencil, _, adjoint, normal = _general_frequency_programs(mesh, face.spec, ffi_dial_key())
    calls = []
    def execute(program, args, stage, runtime_bytes=0):
        calls.append((stage, runtime_bytes))
        return program(*args)
    class StopAtInfinity(Exception):
        pass
    captured = []
    def infinity(*args):
        captured.append(args[-2:])
        raise StopAtInfinity
    # All other fields are irrelevant before the infinity stage. Keep the
    # complete existing bundle boundary rather than a second moment path.
    current = dict.fromkeys(('H adjoint_program algebra begin blocks carrier common_program '
        'definite dyson end energy energy_d equilibrium eta execute_response face fit_basis '
        'fit_ids gamma grid held_ids infinity_program label layout mesh_xy meta mm nb nk nmu '
        'normal_program nq ns output_gram_program pencil_program ph phases plan pm pn print_fn '
        'qgrid recipe rep representation response_active response_window rho run_dir snapshot '
        'started stop tolerance verification_geometry').split())
    current.update(V=V, adjoint_program=adjoint, algebra=dict(prefactor=.37),
        begin=lambda *a: None, end=lambda *a: None, carrier=carrier, definite=False,
        energy=np.asarray(energy_d), energy_d=energy_d, eta=.14, execute_response=execute,
        face=face, fit_basis=dict(directions={0: qi, 1: qi, 'infinity': qi}), gamma=.043,
        infinity_program=infinity, mesh_xy=mesh, meta=meta, mm=mm, nb=nb, nk=nk, nmu=nmu,
        normal_program=normal, nq=nk, ns=ns, pencil_program=pencil, ph=ph, plan=carrier.plan,
        pm=pm, pn=pn, qgrid=(nk, 1, 1), rep=rep, stop=case['nb'],
        recipe=dict(imaginary_width=1, direction_cutoff=1e-8))
    samples = dict(sites=np.array([.2j, .3j]), values=[V, 2*V],
                   fit_positions=[0, 1], held_positions=[])
    moment_calls = []
    def checked_moments(operands, current_meta, **kwargs):
        assert current_meta is meta
        assert kwargs['k_unfold_plan'] is carrier.plan
        assert kwargs['execute'] is execute
        moment_calls.append(kwargs['q_ids'])
        return full_density_bare_moments(operands, current_meta, **kwargs)
    monkeypatch.setattr(response_bank, 'full_density_bare_moments', checked_moments)
    assert 'moments' not in current
    with F.xla_reference():
        x1, x2 = full_density_bare_moments((pm, pn, ph, energy_d), meta,
            mesh_xy=mesh, q_ids=np.arange(nk))
        with pytest.raises(StopAtInfinity):
            _general_current_fit(current, samples)
        assert [stage for stage, _ in calls] == ['direct', 'moment_correlation', 'moment_correlation']
        pref, gamma = .37, .043
        c1 = mm(V, mm(pref*x1, V))
        c2 = mm(V, mm(pref*(x2-1j*gamma*x1), V)) + mm(V, mm(pref*x1, mm(V, mm(pref*x1, V))))
        for got, expected in zip(current['moments'], (c1, c2)):
            np.testing.assert_allclose(got, expected, rtol=3e-12, atol=2e-14)
        assert np.linalg.norm(np.asarray(c1)) > 1e-6
        with pytest.raises(StopAtInfinity):
            _general_current_fit(current, samples)
    assert len(calls) == 3  # Current moments are cached only after native execution.
    assert len(moment_calls) == 1
    np.testing.assert_array_equal(moment_calls[0], np.arange(nk))
    for first, second in zip(captured[0], captured[1]):
        assert first is second
