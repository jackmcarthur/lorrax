"""Small parser/gauge/source-contract oracles, not GPU or material admission."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


def module(name):
    path = Path(__file__).with_name(name+".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


callback = module("sc750_td_callback")


def deck():
    from tdagw.td_config import read_td_deck
    return read_td_deck(str(Path(__file__).with_name("inputs")/"sc750_zero_joint_0p6fs.toml"))


def test_real_zero_deck_uses_legal_full_MH_mode_and_no_stale_source():
    value = deck()
    assert callback.validate_probe_deck(value) == pytest.approx(.6)
    assert value.values["lorrax.collision_hermitian"] == "dynamic"
    assert value.values["lorrax.collision_hamiltonian"] == "reference"
    assert value.values["lorrax.bundle_input"] == "LIVE_AUTHENTICATED_SC750_CALLBACK"
    assert value.values["lorrax.collision_refresh_fs"]*24.188843265857/1000 == pytest.approx(.5)


def test_explicit_matching44_zero_gate_has_full_measurement_and_same_prescription():
    from tdagw.td_config import read_td_deck
    wide=read_td_deck(str(Path(__file__).with_name('inputs')/'sc750_zero_joint_44band_0p6fs.toml'))
    assert callback.validate_probe_deck(wide)==pytest.approx(.6)
    active=callback.probe_active_window(wide)
    assert (active.start,active.stop)==(100,144)
    assert (130-active.start,active.stop-130)==(30,14)
    small=deck()
    excluded={'system.ibndmin','system.ibndmax'}
    assert {k:v for k,v in wide.values.items() if k not in excluded}=={
        k:v for k,v in small.values.items() if k not in excluded}
    assert wide.measurements[-1]==('rho_full','density_matrix',
                                  dict(brange1=[101,145],brange2=[101,145]))
    stale=wide._replace(measurements=small.measurements)
    with pytest.raises(ValueError,match='matching full active rho_full'):
        callback.validate_probe_deck(stale)


def test_callback_uses_validated_window_for_reference_size_and_diagnostic_scope():
    tree=ast.parse(Path(callback.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_td_probe')
    build=next(n for n in ast.walk(run) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='build_sc_td_reference')
    assert ast.unparse(next(k.value for k in build.keywords if k.arg=='active'))=='active'
    system=next(n for n in ast.walk(run) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='TdSystem')
    assert ast.unparse(system.args[4])=='n_active'
    source=ast.get_source_segment(Path(callback.__file__).read_text(),run)
    assert '[128,132)' not in source and '4x4 accepted' not in source
    assert 'actual_active=active_bands' in source


def test_single_trial_observer_retains_only_latest_complete_row_and_requires_identity():
    import gc
    import weakref
    tree=ast.parse(Path(callback.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_td_probe')
    factory=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='factory')
    held=[];events=[];cache=object()
    def original_factory():
        def evaluate(basis,**kwargs):
            return dict(retarded_pair=np.full((2,64,44,44),complex(basis['value'])))
        return evaluate,cache
    scope=dict(original_factory=original_factory,evaluated=held,
               resources=lambda label:events.append(label))
    exec(compile(ast.Module(body=[factory],type_ignores=[]),'<actual callback factory>','exec'),scope)
    observed,returned_cache=scope['factory']()
    assert returned_cache is cache
    init=dict(value=1);result=observed(init,equilibrium=True)
    first=weakref.ref(result['retarded_pair'])
    assert callback.latest_snapshot_row(held,init) is result
    del result
    intermediate=dict(value=2);observed(intermediate)
    final=dict(value=3);result=observed(final)
    gc.collect()
    assert first() is None and len(held)==1
    assert callback.latest_snapshot_row(held,final) is result
    np.testing.assert_array_equal(result['retarded_pair'],3.)
    assert result['retarded_pair'].shape==(2,64,44,44)
    with pytest.raises(ValueError,match='latest single-trial evaluated basis'):
        callback.latest_snapshot_row(held,init)
    assert events==['before_native_snapshot','after_native_snapshot']*3


@pytest.mark.parametrize("key,value", [
    ("system.Efunc.E_x", 1.e9), ("lorrax.collision", "subtracted"),
    ("lorrax.collision_hamiltonian", "hf"), ("lorrax.collision_hermitian", "none"),
    ("lorrax.intraband", True), ("system.ibndmax", 144)])
def test_zero_gate_refuses_scope_changes(key, value):
    original = deck()
    values = dict(original.values, **{key: value})
    with pytest.raises(ValueError, match="GATE"):
        callback.validate_probe_deck(original._replace(values=values))


def test_boundary_compressions_rotate_density_into_actual_rate_gauge():
    u = np.array([[1., 1.j], [1.j, 1.]])/np.sqrt(2)
    rho_basis = np.diag([1., 0.])[None].astype(complex)
    U = u[None]
    gamma = U@np.diag([.03, .02])@U.conj().swapaxes(-1, -2)
    injection = U@np.diag([0., .03])@U.conj().swapaxes(-1, -2)
    rho, loss, raw, result = callback.snapshot_boundary_report(gamma, injection, U, rho_basis)
    np.testing.assert_allclose(rho, U@rho_basis@U.conj().swapaxes(-1, -2))
    np.testing.assert_allclose(raw, injection-.5*(gamma@rho+rho@gamma), atol=1e-16)
    assert result["empty_count"].tolist() == [1]
    assert result["filled_count"].tolist() == [1]
    assert result["empty_min_injection"][0] == pytest.approx(.03)
    assert result["filled_min_blocking"][0] == pytest.approx(.03)
    assert result["min_blocking"][0] == pytest.approx(-.01)
    assert result["trace_rate"][0] == pytest.approx(0., abs=1e-16)
    from tdagw.collision import boundary_diagnostics
    wrong = boundary_diagnostics(gamma, injection, rho_basis, occupation_tolerance=1e-8)
    assert abs(wrong["empty_min_injection"][0]-.03) > .01


def test_literal_source_record_propagates_native_current_tail_head_and_frame():
    """Execute the actual nested record builder with inert host references."""
    probe = module("native_frozen_map_probe")
    tree = ast.parse(Path(probe.__file__).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "source_record")
    compiled = compile(ast.Module(body=[node], type_ignores=[]), str(probe.__file__), "exec")
    original, current, tail, head, energy, unitary = (object() for _ in range(6))
    inputs = NS(wfns_dft=NS(green_parent=NS(projection_faces=lambda: (original,))),
        sym=object(), parallel_transport=NS(velocity_dft_cart=object()),
        V_q=object(), input_dir="native_dir", mesh_xy=object())
    frame = dict(accepted_driver_complete=dict(reference_ready=True))
    scope = dict(inputs=inputs, Vscreen=object(), meta=object(), config=object(),
        U=unitary, E=np.zeros((64, 752)), f=np.zeros((64, 752)), frame=frame,
        arrays=object(), input_deck="native.in", G0_canonical=None, G0_receipt=None, resolver=object(),
        reference_policy=None,
        captured=dict(spectral_tail_reference=tail, sigma_head=head, efermi_ry=-.1),
        time=NS(perf_counter=lambda: 5.), leg_started=2.)
    exec(compiled, scope)
    row = scope["source_record"](current, dict(iteration_head=object()))
    assert row["wfns"] is current and row["original_psi"] is original
    assert row["screening_inputs"]["wfns"] is current
    assert row["screening_inputs"]["V_q"] is scope["Vscreen"]
    assert row["screening_inputs"]["kwargs"]["sym"] is inputs.sym
    assert row["spectral_tail_reference"] is tail and row["head"] is head
    assert row["driver_complete"] is frame["accepted_driver_complete"]
    assert row["frame_arrays"] is scope["arrays"] and row["efermi_ry"] == -.1
    assert row["wall_s"] == 3.


def test_probe_frame_timer_is_off_by_default_and_watchdog_has_no_walk():
    probe = module("native_frozen_map_probe")
    tree = ast.parse(Path(probe.__file__).read_text())
    timers = [kw.value.value for n in ast.walk(tree) if isinstance(n, ast.Call)
              and n.args and isinstance(n.args[0], ast.Constant)
              and n.args[0].value == "--stack-after"
              for kw in n.keywords if kw.arg == "default"]
    assert timers == [0.]
    watchdog = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "watchdog")
    assert "dump_traceback" not in ast.unparse(watchdog)


def test_completed_native_GW_evidence_is_published_before_optional_TD():
    """A later raw finite-width TD refusal must not erase completed GW work."""
    probe = module("native_frozen_map_probe")
    tree = ast.parse(Path(probe.__file__).read_text())
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "evaluate_frozen_leg")
    calls = [n for n in ast.walk(evaluate) if isinstance(n, ast.Call)]
    callback_line = next(n.lineno for n in calls if isinstance(n.func, ast.Name)
                         and n.func.id == "post_sigma_callback")
    for suffix in ("_spectral_fit.npz", "_band_operators.npz", "_receipt.json",
                   "_post_sigma_callback_started.json"):
        publish = next(n for n in calls if isinstance(n.func, ast.Name)
                       and n.func.id == "publish" and n.args
                       and any(isinstance(x, ast.Constant) and x.value == suffix
                               for x in ast.walk(n.args[0])))
        assert publish.lineno < callback_line
    completed = next(n for n in calls if isinstance(n.func, ast.Name)
                     and n.func.id == "publish" and n.args
                     and any(isinstance(x, ast.Constant)
                             and x.value == "_post_sigma_callback_complete.json"
                             for x in ast.walk(n.args[0])))
    assert completed.lineno > callback_line


def test_callback_does_not_enter_collective_IO_from_an_arbitrary_exception():
    tree = ast.parse(Path(callback.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "run_td_probe")
    # The rank0-only OSError read is explicitly joined by agree_io_refusal;
    # an arbitrary execution failure must not call the collective save helper.
    handlers = [n for n in ast.walk(run) if isinstance(n, ast.ExceptHandler)]
    assert all(isinstance(n.type, ast.Name) and n.type.id == "OSError" for n in handlers)
    assert not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                   and x.func.id == "save" for n in handlers for x in ast.walk(n))
