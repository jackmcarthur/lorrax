"""Native verdicts must reject poisoned comparisons before tolerance reduction."""
from types import SimpleNamespace

import numpy as np


def test_native_gate_relative_metrics_reject_poison_and_overflow():
    import jax
    from tests.test_active_local_native_gate import _rel as active_rel
    from tests.test_kconv_xla_gate import _rel as kconv_rel
    from tests.test_kdft_parent_gate import _sharded_rel

    reference = np.array([1 + 2j, -3j])
    for metric in (active_rel, kconv_rel, jax.jit(_sharded_rel)):
        assert float(metric(reference, reference)) == 0
        assert np.isclose(float(metric(reference + .03, reference)), .01)
        assert float(metric(np.zeros(2), np.zeros(2))) == 0
        with np.errstate(over="ignore", invalid="ignore"):
            for value in (np.nan, np.inf, -np.inf):
                poisoned = reference.copy(); poisoned[-1] = value
                assert np.isinf(float(metric(poisoned, reference)))
                assert np.isinf(float(metric(reference, poisoned)))
            huge = np.array([1.7e308 + 1.7e308j])
            assert np.isinf(float(metric(huge, huge)))
            assert np.isinf(float(metric(-huge, huge)))
    for metric in (active_rel, kconv_rel, jax.jit(_sharded_rel)):
        assert np.isinf(metric(reference[:1], reference))


def test_active_gate_refuses_late_nan_metric(monkeypatch, capsys):
    import runtime
    from ffi import gate
    from tests import test_active_local_native_gate as active

    monkeypatch.setattr(runtime, "initialize_communicator_stack", lambda: SimpleNamespace(
        mesh=SimpleNamespace(devices=np.empty((1, 1)))))
    monkeypatch.setattr(gate, "mesh_ffi_platform", lambda mesh: "ROCM")
    monkeypatch.setattr(active, "plan_cases", lambda *args: {})
    exits = []
    monkeypatch.setattr(runtime, "finalize_process", exits.append)
    for pair in ((np.nan, 0.), (0., np.nan), (np.inf, 0.)):
        monkeypatch.setattr(active, "raw_cases", lambda platform: (
            {"late_poison": {"finite_first": (0., 0.), "poisoned_later": pair}}, {}))
        assert active.main() == 1
        assert exits[-1] == 1
        assert "ACTIVE_LOCAL_NATIVE_GATE FAIL" in capsys.readouterr().out
