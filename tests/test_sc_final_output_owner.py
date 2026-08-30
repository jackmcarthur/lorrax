"""The generic SC result is the eigensystem of the accepted SC carry.

The last Sigma map and the accepted carry are deliberately not equivalent:
the protected/in-range/out-of-grid partition can zero couplings and replace a
diagonal by a scissor value.  Re-diagonalising raw ``kin_ion + Sigma`` in the
driver therefore silently publishes a different result.
"""
from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SC_PATH = _ROOT / "src" / "gw" / "sc_iteration.py"
_DRIVER_PATH = _ROOT / "src" / "gw" / "gw_jax.py"

# These are the distrib_la eigensolver accuracy contracts, not CPU LAPACK
# equivalence tolerances.  The real-mesh gates use 1e-10 for eigenvalues
# against NumPy and 1e-11 for the eigensystem residual/orthogonality of a
# distributed solve; see test_distrib_la_multiproc.py and
# test_distrib_la_contract.py.  Both measures are normalized by the scale of
# the reference matrix so the same physical Hamiltonian in Ry or eV is judged
# identically.
_DISTRIB_EIGH_VALUE_RTOL = 1.0e-10
_DISTRIB_EIGH_VECTOR_RTOL = 1.0e-11
_DISTRIB_EIGH_RECONSTRUCTION_RTOL = 2.0 * _DISTRIB_EIGH_VECTOR_RTOL


def _max_relative_error(actual, expected) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return float(np.max(np.abs(actual - expected))) / max(
        float(np.max(np.abs(expected))), 1.0e-300)


def _frobenius_relative_error(actual, expected) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return float(np.linalg.norm(actual - expected)) / max(
        float(np.linalg.norm(expected)), 1.0e-300)


def _function(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected one {name} in {path}, got {len(matches)}"
    return matches[0]


def _assigned_names(nodes) -> set[str]:
    return {
        target.id
        for node in nodes
        for child in ast.walk(node)
        if isinstance(child, (ast.Assign, ast.AnnAssign))
        for target in (
            child.targets if isinstance(child, ast.Assign) else [child.target]
        )
        if isinstance(target, ast.Name)
    }


def _attribute_paths(nodes) -> set[str]:
    paths = set()
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute):
                continue
            parts = [child.attr]
            value = child.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                paths.add(".".join([value.id, *reversed(parts)]))
    return paths


def test_partitioned_accepted_h_owns_full_bz_output(monkeypatch):
    """Red/green fixture: raw map H and accepted partitioned H disagree."""
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from gw import sc_iteration
    from gw.band_partition import apply_band_partition

    # ``raw_h`` represents kin_ion + the last raw Sigma map.  Band 2 is
    # outside the Sigma grid: acceptance removes its couplings and installs
    # the fitted scissor diagonal.  The protected 0/1 block deliberately has
    # a complex coupling so the TR star member also exercises conjugation.
    raw_h = np.array([[[
        -1.0, 0.0 + 0.15j, 0.20,
    ], [
        0.0 - 0.15j, 0.35, -0.10j,
    ], [
        0.20, 0.10j, 0.55,
    ]]], dtype=np.complex128)
    accepted_h = apply_band_partition(
        jnp.asarray(raw_h),
        protected_mask=jnp.asarray([True, True, False]),
        in_range_mask=jnp.asarray([True, True, False]),
        scissor_E_qp_kn=jnp.asarray([[0.0, 0.0, 2.50]]),
    )
    assert not np.allclose(np.asarray(accepted_h), raw_h)

    solve_calls = []

    def _fake_sc_eigh(H, *, kind, mesh_xy, config):
        solve_calls.append((kind, H))
        E, U = np.linalg.eigh(np.asarray(H))
        return jnp.asarray(E), jnp.asarray(U)

    monkeypatch.setattr(sc_iteration, "_resolve_sc_eigh",
                        lambda *_args, **_kwargs: "native")
    monkeypatch.setattr(sc_iteration, "_sc_eigh_bands", _fake_sc_eigh)
    state = sc_iteration.SCState(H_qp_dft=accepted_h, iteration=4)
    config = SimpleNamespace()
    E_star, U_star, _ = sc_iteration.final_qp_eigenstates(
        state, n_occ=1, mesh_xy=object(), config=config,
        print_fn=lambda *_args: None)

    assert len(solve_calls) == 1
    expected_E, expected_U = np.linalg.eigh(np.asarray(accepted_h))
    np.testing.assert_allclose(E_star, expected_E)
    np.testing.assert_allclose(U_star, expected_U)
    raw_E = np.linalg.eigvalsh(raw_h)
    assert np.max(np.abs(E_star - raw_E)) > 1.0

    # A real incumbent KStarMap, including one antiunitary star member.
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import KStarMap

    kstar = KStarMap(
        irr_idx=np.array([0, 0], dtype=np.int32),
        sym_idx=np.array([0, 1], dtype=np.int32),
        n_sym_spatial=1,
    )
    E_full, U_full = sc_iteration._loop_arrays_on_full_bz(
        (E_star, U_star), kstar=kstar, state_on_ibz=True)
    np.testing.assert_array_equal(E_full[1], E_full[0])
    np.testing.assert_allclose(U_full[1], np.conj(U_full[0]))


def test_final_owner_routes_through_service_and_matches_host():
    """Exercise the real native distrib_la route under its accuracy contract."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from common.collectives import device_put_process_local
    from gw import sc_iteration

    rng = np.random.default_rng(8)
    A = (rng.normal(size=(2, 5, 5))
         + 1j * rng.normal(size=(2, 5, 5)))
    H = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    devices = jax.devices()
    # The landing P4 leg gets a real 2x2 mesh and exercises nb=5 -> 8
    # service padding.  Ordinary login/unit runs retain a safe 1x1 route.
    mesh_shape = (2, 2) if len(devices) == 4 else (1, 1)
    mesh_devices = devices if len(devices) == 4 else devices[:1]
    mesh = Mesh(np.asarray(mesh_devices).reshape(mesh_shape), ("x", "y"))
    config = SimpleNamespace(
        sc=SimpleNamespace(eigh="native"),
        memory=SimpleNamespace(per_device_gb=40.0),
        backend=SimpleNamespace(distrib_la_batched_route="batch_reshard"),
    )
    H_device = device_put_process_local(
        H, NamedSharding(mesh, P(None, None, None)))
    E, U, _ = sc_iteration.final_qp_eigenstates(
        sc_iteration.SCState(H_qp_dft=H_device, iteration=2),
        n_occ=2, mesh_xy=mesh, config=config, print_fn=lambda *_args: None)

    E_ref, _ = np.linalg.eigh(H)
    eigenvalue_error = _max_relative_error(E, E_ref)
    assert eigenvalue_error <= _DISTRIB_EIGH_VALUE_RTOL, (
        "distributed eigenvalues versus NumPy: relative error "
        f"{eigenvalue_error:.3e} exceeds "
        f"{_DISTRIB_EIGH_VALUE_RTOL:.1e}")

    # These checks are invariant to eigenvector phases and rotations within a
    # degenerate subspace.  Comparing U directly to NumPy would turn harmless
    # gauge choices into failures.  The residual and orthogonality bounds are
    # the service contract.  Reconstruction composes both errors, hence its
    # derived 2 * vector-bound rather than another ad-hoc absolute tolerance.
    eigenpair_error = max(
        float(np.linalg.norm(H[q] @ U[q] - U[q] * E[q][None, :]))
        / max(float(np.linalg.norm(H[q])), 1.0e-300)
        for q in range(H.shape[0]))
    orthogonality_error = max(
        float(np.linalg.norm(
            np.conj(U[q].T) @ U[q] - np.eye(U.shape[-1])))
        for q in range(H.shape[0]))
    assert eigenpair_error <= _DISTRIB_EIGH_VECTOR_RTOL, (
        "distributed eigenpair residual: relative error "
        f"{eigenpair_error:.3e} exceeds "
        f"{_DISTRIB_EIGH_VECTOR_RTOL:.1e}")
    assert orthogonality_error <= _DISTRIB_EIGH_VECTOR_RTOL, (
        "distributed eigenvector orthogonality: error "
        f"{orthogonality_error:.3e} exceeds "
        f"{_DISTRIB_EIGH_VECTOR_RTOL:.1e}")

    rebuilt = U @ (E[:, :, None] * np.conj(np.swapaxes(U, -1, -2)))
    reconstruction_error = max(
        _frobenius_relative_error(rebuilt[q], H[q])
        for q in range(H.shape[0]))
    assert reconstruction_error <= _DISTRIB_EIGH_RECONSTRUCTION_RTOL, (
        "distributed eigensystem reconstruction: relative error "
        f"{reconstruction_error:.3e} exceeds "
        f"{_DISTRIB_EIGH_RECONSTRUCTION_RTOL:.1e}")
    print(
        "SC_FINAL_EIGH_ACCURACY "
        f"eigenvalue_rel={eigenvalue_error:.6e} "
        f"eigenpair_rel={eigenpair_error:.6e} "
        f"orthogonality={orthogonality_error:.6e} "
        f"reconstruction_rel={reconstruction_error:.6e}")


def test_terminal_solve_is_single_owned_and_optional_dump_reuses_it():
    run_sc_driver = _function(_SC_PATH, "run_sc_driver")
    dump = _function(_SC_PATH, "dump_qp_wfn_artifacts")
    final = _function(_SC_PATH, "final_qp_eigenstates")

    calls = [
        node for node in ast.walk(run_sc_driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "final_qp_eigenstates"
    ]
    assert len(calls) == 1
    timing_rows = {
        node.args[0].value
        for node in ast.walk(run_sc_driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "timing" and node.func.attr == "section"
        and node.args and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "sc.final_eigh" in timing_rows
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "final_qp_eigenstates"
        for node in ast.walk(dump)
    ), "the optional artifact writer must consume, not recompute, terminal E/U"
    assert "_sc_eigh_bands" in {
        node.func.id for node in ast.walk(final)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_diagonalize_and_get_efermi" not in {
        node.func.id for node in ast.walk(final)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_sc_driver_output_arm_has_no_private_eigh_or_eigvalsh():
    """The one-shot vmap(eigh) remains, but cannot be reached by SC."""
    main = _function(_DRIVER_PATH, "main")
    output_ifs = [
        node for node in ast.walk(main)
        if isinstance(node, ast.If)
        and {"E_full", "U_full"} <= _assigned_names(node.body)
    ]
    assert len(output_ifs) == 1
    branch = output_ifs[0]
    test_text = ast.unparse(branch.test)
    assert "qp_solver is QPSolver.SELF_CONSISTENT" in test_text

    sc_paths = _attribute_paths(branch.body)
    assert "sc_result.E_qp_full_ry" in sc_paths
    assert "sc_result.U_dft_to_qp_full" in sc_paths
    assert not ({"jnp.linalg.eigh", "jnp.linalg.eigvalsh"} & sc_paths)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "jax" and node.func.attr == "vmap"
        for stmt in branch.body for node in ast.walk(stmt)
    )

    # The established one-shot branch remains exactly one local vmap(eigh).
    one_shot_paths = _attribute_paths(branch.orelse)
    assert "jnp.linalg.eigh" in one_shot_paths
    assert "jnp.linalg.eigvalsh" not in one_shot_paths
