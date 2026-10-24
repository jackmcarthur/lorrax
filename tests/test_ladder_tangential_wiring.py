"""Admission, model identity and micro-reducible head wiring without a WFN."""
import __future__
import ast
import builtins
from contextlib import nullcontext
import enum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


def _definitions(relative, names, namespace):
    nodes = [node for node in ast.parse((ROOT / relative).read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
             or isinstance(node, (ast.Assign, ast.AnnAssign))
             and any(isinstance(target, ast.Name) and target.id in names
                     for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _imports(modules):
    def load(name, *args, **kwargs):
        return modules[name] if name in modules else builtins.__import__(name, *args, **kwargs)
    return dict(vars(builtins), __import__=load)


def _enums():
    return _definitions("src/gw/gw_config.py", {"ComputeMode", "ScreeningDiagrams",
        "QPSolver", "HeadCorrection"}, {"enum": enum})


def test_parent_admission_uses_actual_tangential_mode_and_preserves_refusals():
    ns = _enums()
    _definitions("src/file_io/restart_bundle.py", {"require_parent_screening_consumer"}, ns)
    _definitions("src/gw/gw_config.py", {"_W_BSE_REFUSALS", "_W_RPA_RESOLVENT_REFUSALS",
        "_RESOLVENT_REFUSAL_TABLES", "coerce_screening_diagrams",
        "refuse_unsupported_screening_diagrams"}, ns)
    modes, diagrams = ns["ComputeMode"], ns["ScreeningDiagrams"]

    def config(mode, diagram, *, self_consistent=False):
        return NS(compute_mode=mode, screening=NS(diagrams=diagram),
                  sigma=NS(w_model="tangential" if mode is modes.TANGENTIAL else "mpa"),
                  qp_solver=ns["QPSolver"].SELF_CONSISTENT if self_consistent
                  else ns["QPSolver"].ONE_SHOT_DFT,
                  head=NS(mc_average_placement="off"))

    gate = ns["require_parent_screening_consumer"]
    for mode in (modes.COHSEX, modes.GN_PPM, modes.TANGENTIAL):
        gate(config(mode, diagrams.W_BSE))
    for mode, diagram in ((modes.MPA, diagrams.W_BSE),
                          (modes.TANGENTIAL, diagrams.W_RPA_RESOLVENT)):
        with pytest.raises(ValueError, match="parent_screening_diagrams"):
            gate(config(mode, diagram))
    with pytest.raises(ValueError, match="w_bse_self_consistency_unimplemented"):
        ns["refuse_unsupported_screening_diagrams"](
            config(modes.TANGENTIAL, diagrams.W_BSE, self_consistent=True))


def test_ladder_diagram_and_rung_digest_reach_restart_validation():
    from bse.w_ladder import LADDER_RESPONSE_CONVENTION
    from gw.tangential_recipe import shared_real_pole_gates_v1_r3b
    base = {name: "same-state" for name in ("iteration_id", "hamiltonian", "energies",
        "occupations", "wavefunctions", "centroids", "recipe_hash", "gate_hash")}
    captured = {}
    class AtRestart(Exception):
        pass
    def restart(_path, *, expected_identity, **_kwargs):
        captured.update(expected_identity)
        raise AtRestart
    static = np.eye(2)
    modules = {
        "bse.w_ladder": NS(LADDER_RESPONSE_CONVENTION=LADDER_RESPONSE_CONVENTION),
        "screening_bse": NS(_GMRES_MAX_ITER=300, _GMRES_TOL=1e-6,
            _assert_wedge_matches_run=None, _finalize_ladder_head=None,
            prepare_ladder_restart=lambda *a, **k: static),
        "response_bank": NS(operator_digest=lambda a, _m, **_kw: hashlib.sha256(a.tobytes()).hexdigest()),
        "gw_config": NS(uses_full_bispinor_tangential=lambda _c: False),
        "ppm_pipeline": NS(observing_window_source=lambda: False, publish_window_source=None),
        "file_io.tangential_store": NS(initialize_tangential_bank=None),
        "file_io.tagged_arrays": NS(register_tangential_restart_member=None),
        "tangential_recipe": NS(tangential_restart_handle=restart,
            shared_real_pole_gates_v1_r3b=shared_real_pole_gates_v1_r3b),
        "tangential_constructor": NS(construct_tangential_poles=None),
        "w_isdf": NS(produce_w_bank=None, compute_response_moments=None),
    }
    ns = {"__builtins__": _imports(modules), "time": time, "np": np,
          "timing": NS(section=lambda *a, **k: nullcontext()),
          "tangential_identity": lambda *a, **k: dict(base)}
    _definitions("src/gw/ladder_tangential.py", {"_solver_accuracy", "make_ladder_tangential_source"}, ns)
    _definitions("src/gw/tangential_screening.py", {"screen_tangential_poles"}, ns)
    config = NS(screening=NS(diagrams="w_bse"), debug=NS(write_w=False),
                write_poles=False, restart=True, head=NS(correction="off"))
    meta = NS(tangential_recipe={"bank_rule_tolerance": 1e-8}, tangential_capacity=NS(live_stages=()))
    source = ns["make_ladder_tangential_source"](None, None, quad=None, e_ref=0,
        sym=None, centroid_indices=[], config=config, meta=meta, mesh_xy=None,
        tensors_filename="restart.h5", head_resolver=None)
    with pytest.raises(AtRestart):
        ns["screen_tangential_poles"](None, None, meta, config, mesh_xy=None, sym=None,
            centroid_indices=[], run_dir="unused", label="oneshot", wfn=None,
            wfn_fingerprint_binding=None, tensors_filename="restart.h5",
            occupation_state=None, print_fn=lambda *a: None,
            material_class="insulator", bank_source=source)
    assert captured["screening_diagrams"] == "w_bse"
    assert captured["ladder_response_convention"] == "physical-ra-v1"
    assert captured["ladder_static_w_sha256"] == hashlib.sha256(static.tobytes()).hexdigest()
    assert captured["ladder_head_policy"] == "off"
    assert captured["ladder_gmres_tol"] == pytest.approx(1e-9)
    assert captured["ladder_gmres_max_iter"] == 300
    assert captured["ladder_solve_accuracy_policy"] == "bank-relative-true-residual-v1"

    def refuse(message):
        raise ValueError(message)
    ns.update(_json=lambda value: json.dumps(value, sort_keys=True), _refuse=refuse)
    _definitions("src/file_io/tangential_store.py", {"_IDENTITY_KEYS", "_check_identity"}, ns)
    ns["_check_identity"](captured, captured)
    for stale in (base, dict(captured, screening_diagrams="w_rpa"),
                  dict(captured, ladder_response_convention="folded-hybrid"),
                  dict(captured, ladder_static_w_sha256="other-rung"),
                  dict(captured, ladder_head_policy="full"),
                  dict(captured, ladder_gmres_tol=1e-6),
                  dict(captured, ladder_gmres_max_iter=200),
                  dict(captured, ladder_solve_accuracy_policy="old")):
        with pytest.raises(ValueError, match="stale input/SC identity"):
            ns["_check_identity"](stale, captured)


def test_tangential_solve_budget_and_delivered_residual_gate():
    from gw.ladder_tangential import _solver_accuracy, _gate_solve
    from gw.screening_bse import _GMRES_TOL, _GMRES_MAX_ITER
    from gw.tangential_recipe import shared_real_pole_v1_r3b
    for tier, tolerance in (("production", 1e-9), ("relaxed", 1e-8)):
        recipe = shared_real_pole_v1_r3b[tier]
        accuracy = _solver_accuracy(recipe)
        assert accuracy["gmres_tol"] == pytest.approx(tolerance)
        assert accuracy["gmres_max_iter"] == _GMRES_MAX_ITER == 300
        tol = accuracy["gmres_tol"]
        _gate_solve(np.array([0., .9*tol, tol]), np.array([0, 5, 299]),
                    accuracy=accuracy, where="value/slope/head control")
        # The legacy GN guard allowed 10x the requested tolerance. The
        # Hermite producer must instead enforce the actual input budget.
        for bad in (np.nextafter(tol, np.inf), 1e-6, np.nan, np.inf):
            with pytest.raises(ValueError, match="GATE ladder_tangential_solve.*requested"):
                _gate_solve(np.array([bad]), np.array([7]), accuracy=accuracy, where=tier)
        with pytest.raises(ValueError, match="iterations 300/300"):
            _gate_solve(np.zeros(1), np.array([300]), accuracy=accuracy, where=tier)
    for bad in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="GATE ladder_tangential_accuracy"):
            _solver_accuracy({"bank_rule_tolerance": bad})
    assert _GMRES_TOL == 1e-6  # The production GN-PPM numerical contract is unchanged.


@pytest.mark.parametrize("micro", [True, False])
def test_head_reuses_micro_samples_and_preserves_direct_response_route(micro):
    enums = _enums()
    samples = tuple(NS(response_kind="micro_reducible" if micro else "direct_irreducible",
                       token=i) for i in range(3))
    points = (1j, 2j, 3j)
    response = NS(omegas=points, samples=samples) if micro else NS(omegas=points)
    seen = {}
    def forbidden(*_a, **_k):
        raise AssertionError("A micro-reducible head must not read or Schur-fold the body")
    def fit(values, z, _poles, **kwargs):
        assert all(a is b for a, b in zip(values, samples[:2]))
        np.testing.assert_array_equal(z, points[:2])
        seen.update(kwargs)
        return {}
    def moment(values, nodes, **_kwargs):
        assert all(a is b for a, b in zip(values, samples))
        assert tuple(nodes) == points
        return 0.7
    modules = {
        "file_io.tangential_store": NS(open_tangential_model=forbidden,
                                         read_tangential_matrix=forbidden),
        "gw_config": NS(HeadCorrection=enums["HeadCorrection"]),
        "mpa.model": NS(fit_head_samples=fit, head_moment_report=forbidden),
        "mpa.sample_plan": NS(plan_z=lambda _plan: np.array(points[:2])),
        "qsgw_head": NS(IterationHeadSamples=lambda **kwargs: NS(**kwargs),
            finalize_iteration_head_rows=forbidden if micro else lambda *_a, **_k: (),
            head_moment_m1=moment,
            iteration_head_sample_terms=forbidden if micro else
                lambda _response, index, _body, **_kwargs: samples[index],
            metal_response_nodes=lambda p, *_a, **_k: p),
        "response_bank": NS(_reserve=forbidden),
        "qgrid_symmetry": NS(tangential_operator_realizer=forbidden),
        "distrib_la": NS(),
    }
    ns = {"__builtins__": _imports(modules), "np": np, "jnp": np,
        "jax": NS(process_index=lambda: 1),
        "timing": NS(section=lambda *a, **k: nullcontext())}
    _definitions("src/gw/tangential_head.py", {"build_tangential_head"}, ns)
    config = NS(head=NS(uses_bgw_metal_q0shift=False,
                        correction=enums["HeadCorrection"].FULL if micro else
                            enums["HeadCorrection"].OFF),
                mpa=NS(n_poles=2, pole_solver="pade"))
    meta = NS(tangential_capacity=NS(live_stages=()),
              tangential_recipe={"census": {"mu_ry": 0.}})
    wfns = NS(enk=np.zeros((1, 2)), occ=np.ones((1, 2)), slices=NS(sigma=slice(None)))
    head, iteration = ns["build_tangential_head"]({"identity": {"screening_diagrams": "w_bse"},
        "digest": "ladder-body"}, None, None, wfns, meta, config,
        mesh_xy=None, wfn=None, response=response, head_resolver=None,
        plan=object(), material_class="insulator", occupation_state=None)
    assert seen["model"] == ("bse_resolvent_micro_pade" if micro else "dft_direct_pade")
    assert seen["moment_m1"] == 0.7 and iteration.samples == samples
    assert head["body_digest"] == "ladder-body"


def test_temporary_operator_digest_does_not_retain_static_rung():
    import gc
    import weakref
    ns = dict(np=np, hashlib=hashlib, _OPERATOR_DIGESTS={},
              _operator_digest_program=lambda _mesh, _shape: lambda value: np.array([np.sum(value), 0]))
    _definitions("src/gw/response_bank.py", {"operator_digest"}, ns)
    operator = np.eye(2)[None]
    reference = weakref.ref(operator)
    mesh = NS(shape={"x": 1, "y": 1})
    expected = ns["operator_digest"](operator, mesh, cache=False)
    assert not ns["_OPERATOR_DIGESTS"]
    del operator
    gc.collect()
    assert reference() is None
    held = np.eye(2)[None]
    assert ns["operator_digest"](held, mesh) == expected
    assert ns["_OPERATOR_DIGESTS"][id(held)][0] is held
