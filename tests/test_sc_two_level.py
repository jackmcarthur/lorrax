"""Two-level QSGW orchestration without running a physics kernel."""

import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from gw import sc_iteration, sc_two_level


def _verdict(value, *, converged):
    return sc_iteration.ConvergenceVerdict(
        converged=bool(converged), max_abs_ev=float(value),
        rms_protected_ev=float(value), rms_all_ev=float(value),
        n_protected=1, n_total=1, worst_k=0, worst_band=0,
        cutoff_ev=1.0e-3)


@dataclass(frozen=True)
class _Inputs:
    frozen_screening: object = None
    mesh_xy: object = None
    partition: object = None
    fixed_quadrature_session: object = None
    config: object = field(default_factory=lambda: SimpleNamespace(
        sc=SimpleNamespace(outer_refit_policy="fixed_poles")))
    print_fn: object = lambda *_args: None


def _state(value, iteration, *, outputs=None, converged=False):
    partition = SimpleNamespace(
        protected_mask=np.asarray([True]),
        in_range_mask=np.asarray([True]))
    return sc_iteration.SCState(
        H_qp_dft=np.asarray([[[float(value)]]]),
        iteration=int(iteration), partition=partition,
        outputs=outputs,
        convergence_verdict=_verdict(
            0.0 if converged else 1.0, converged=converged),
    )


class _Outputs:
    """Weak-referenceable stand-in for the output-sized map transaction."""

    def __init__(self, screening):
        self.screening = screening


def test_one_map_diagnostic_delegates_without_two_level_wrapping(monkeypatch):
    initial = _state(0.0, 0)
    inputs = _Inputs()
    sentinel = (_state(1.0, 1), [])
    calls = []

    def _run(state, passed_inputs, **kwargs):
        calls.append((state, passed_inputs, kwargs))
        return sentinel

    monkeypatch.setattr(sc_iteration, "run_self_consistency", _run)
    got = sc_two_level.run_two_level_self_consistency(
        initial, inputs, max_iter=1, tol_ev=1.0e-3,
        accelerator="rcrop", history_depth=5, mixing=1.0)

    assert got is sentinel
    assert len(calls) == 1
    assert calls[0][1] is inputs
    assert calls[0][2]["max_iter"] == 1


def test_outer_refits_freeze_each_model_and_reset_inner_history(monkeypatch):
    inputs = _Inputs()
    initial = _state(0.0, 0)
    calls = []
    live_count = [0]
    producer_output_refs = []

    monkeypatch.setattr(
        sc_iteration, "_kshard_eigh_kernels",
        lambda _mesh: (None, lambda H: np.real(np.asarray(H)[..., 0])))

    def _run(state, passed_inputs, **kwargs):
        calls.append((float(np.asarray(state.H_qp_dft)[0, 0, 0]),
                      passed_inputs.frozen_screening, dict(kwargs),
                      passed_inputs.fixed_quadrature_session))
        if passed_inputs.frozen_screening is None:
            live_count[0] += 1
            model = {"outer": live_count[0]}
            screening = sc_iteration.SCMapScreeningArtifacts(
                static_w=None, iteration_head=None,
                static_head_terms=None, sigma_model=model)
            outputs = _Outputs(screening)
            producer_output_refs.append(weakref.ref(outputs))
            return _state(
                float(np.asarray(state.H_qp_dft)[0, 0, 0]) + 10.0,
                state.iteration + 1, outputs=outputs), []

        # The inner call may retain the screening object itself, but the
        # producer's output-sized Sigma/writer transaction must already be
        # dead before this next map allocation.
        assert producer_output_refs[-1]() is None
        # First inner fixed point is 1.0.  With alpha=0.25 the next outer
        # producer must therefore see 0.25, not the unmixed candidate.  The
        # second inner answer is within the 1 meV outer cutoff of that input.
        target = 1.0 if live_count[0] == 1 else 0.25001
        screening = passed_inputs.frozen_screening
        outputs = SimpleNamespace(screening=screening)
        return _state(
            target, state.iteration + 2, outputs=outputs, converged=True), [
                0.2, 0.1]

    monkeypatch.setattr(sc_iteration, "run_self_consistency", _run)

    final, history = sc_two_level.run_two_level_self_consistency(
        initial, inputs, max_iter=3, tol_ev=1.0e-3,
        accelerator="rcrop", history_depth=5, mixing=0.25)

    assert final.convergence_verdict.converged
    np.testing.assert_allclose(final.H_qp_dft, 0.25001)
    assert len(history) == 6  # one producer map + two inner calls, twice
    assert [call[0] for call in calls] == [0.0, 10.0, 0.25, 10.25]
    assert calls[0][1] is None and calls[2][1] is None
    assert calls[1][1].sigma_model == {"outer": 1}
    assert calls[3][1].sigma_model == {"outer": 2}
    # One artifact sequence, while each inner runner is a fresh invocation
    # (and therefore a fresh rCROP history).
    assert [call[2]["reset_diagnostics"] for call in calls] == [
        True, False, False, False]
    assert [call[2]["snapshot_offset"] for call in calls] == [0, 1, 3, 4]
    assert calls[0][3] is None and calls[2][3] is None
    assert calls[1][3] is not calls[3][3]
    assert calls[1][3] == calls[3][3] == {
        "state_edge_padding_ev": 0.25,
        "pole_extent_padding_fraction": 0.0,
    }
    assert calls[1][1].w_time_factor_cache == {}
    assert calls[3][1].w_time_factor_cache == {}
    assert calls[1][1].w_time_factor_cache is not calls[3][1].w_time_factor_cache


def test_two_level_refuses_a_pre_frozen_initial_input():
    inputs = _Inputs(frozen_screening=object())
    with np.testing.assert_raises_regex(ValueError, "owns the frozen"):
        sc_two_level.run_two_level_self_consistency(
            _state(0.0, 0), inputs, max_iter=2, tol_ev=1.0e-3,
            accelerator="rcrop", history_depth=5, mixing=1.0)
