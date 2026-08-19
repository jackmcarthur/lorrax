"""QSGW head samples must survive the restart-writer seam."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gw import gw_output, restart_q_storage
from gw.head_correction import HeadResponseKind, HeadSample


class _HeadSource:
    def __init__(self, samples):
        self.samples = {complex(key): value for key, value in samples.items()}
        self.calls = []

    def at(self, omega):
        key = complex(omega)
        self.calls.append(key)
        return self.samples[key]


class _QStorage:
    def with_capture(self, _capture):
        return self


def _config(*, material_class="metal", correction="full", head_update="off"):
    return SimpleNamespace(
        do_screened=True,
        qp_solver=SimpleNamespace(value="self_consistent"),
        mpa=SimpleNamespace(material_class=material_class),
        compute_mode=SimpleNamespace(
            value="gn_ppm", ppm_model="gn", is_dynamic=True),
        head=SimpleNamespace(correction=SimpleNamespace(value=correction)),
        sc=SimpleNamespace(head_update=head_update),
        ppm=SimpleNamespace(omega_p=2.0),
        screening=SimpleNamespace(diagrams=None),
    )


def _patch_writer_plumbing(monkeypatch, written):
    import file_io

    monkeypatch.setattr(
        gw_output, "restart_tensor_writes_enabled", lambda *_args: True)
    monkeypatch.setattr(
        gw_output, "_stamp_screening_diagrams", lambda *_args: None)
    monkeypatch.setattr(
        restart_q_storage, "resolve_restart_q_storage_for_run",
        lambda *_args, **_kwargs: _QStorage())
    monkeypatch.setattr(
        restart_q_storage, "take_pre_unfold", lambda _name: object())
    monkeypatch.setattr(
        file_io, "write_w0_qmunu_to_h5",
        lambda *_args, **kwargs: written.setdefault("w0", kwargs))
    monkeypatch.setattr(
        file_io, "write_head_scalars_to_h5",
        lambda *_args, **kwargs: written.setdefault("head", kwargs))


def test_qsgw_iteration_samples_are_the_only_persisted_head_source(
    tmp_path, monkeypatch,
):
    """Removing ``iteration_head`` makes this call use the forbidden resolver."""
    restart = tmp_path / "isdf_tensors_1.h5"
    restart.touch()
    S_static = np.arange(9, dtype=np.float64).reshape(3, 3).astype(np.complex128)
    S_probe = S_static + 10.0
    iteration_head = _HeadSource({
        0.0j: HeadSample(
            vc0=101.0 + 0.0j, wcoul0=41.0 + 0.0j,
            source="qsgw_parallel_transport", omega=0.0j, S_cart=S_static,
            response_kind=HeadResponseKind.FULL_LOCAL_FIELDS),
        2.0j: HeadSample(
            vc0=102.0 + 0.0j, wcoul0=42.0 + 0.0j,
            source="qsgw_parallel_transport(omega=2j Ry)",
            omega=2.0j, S_cart=S_probe,
            response_kind=HeadResponseKind.FULL_LOCAL_FIELDS),
    })
    forbidden_resolver = _HeadSource({})
    written = {}
    _patch_writer_plumbing(monkeypatch, written)

    gw_output.persist_w0_and_head(
        np.zeros((1, 1, 1), dtype=np.complex128),
        tensors_filename=str(restart),
        head_resolver=forbidden_resolver,
        iteration_head=iteration_head,
        config=_config(),
        meta=SimpleNamespace(n_rmu=1),
        mesh_xy=object(),
        print_fn=lambda *_args: None,
    )

    assert forbidden_resolver.calls == []
    assert iteration_head.calls == [0.0j, 2.0j]
    assert written["head"]["vhead"] == 101.0 + 0.0j
    np.testing.assert_array_equal(
        written["head"]["whead"],
        np.array([41.0 + 0.0j, 42.0 + 0.0j], dtype=np.complex128))
    np.testing.assert_array_equal(written["head"]["S_cart"], S_static)
    assert written["head"]["head_correction"] == "full"
    assert written["head"]["response_kind"] == "full_local_fields"
    assert written["head"]["head_source"] == "qsgw_parallel_transport"


def test_static_metal_without_s_cart_refuses_before_any_restart_write(
    tmp_path, monkeypatch,
):
    """The TF scalar cannot be paired with a later DFT ``dipole.h5`` rebuild."""
    restart = tmp_path / "isdf_tensors_1.h5"
    restart.touch()
    iteration_head = _HeadSource({
        0.0j: HeadSample(
            vc0=101.0 + 0.0j, wcoul0=17.0 + 0.0j,
            source="qsgw_parallel_transport_tf", omega=0.0j, S_cart=None),
    })
    written = {}
    _patch_writer_plumbing(monkeypatch, written)

    with pytest.raises(
        ValueError, match="persist_iteration_head_requires_s_cart",
    ):
        gw_output.persist_w0_and_head(
            np.zeros((1, 1, 1), dtype=np.complex128),
            tensors_filename=str(restart),
            head_resolver=_HeadSource({}),
            iteration_head=iteration_head,
            config=_config(),
            meta=SimpleNamespace(n_rmu=1),
            mesh_xy=object(),
            static_head_only=True,
            print_fn=lambda *_args: None,
        )

    assert written == {}


def test_full_qsgw_insulator_cannot_fall_back_to_the_dft_seed_head(
    tmp_path, monkeypatch,
):
    """The final-map requirement is physics-based, not metal-only."""
    restart = tmp_path / "isdf_tensors_1.h5"
    restart.touch()
    written = {}
    _patch_writer_plumbing(monkeypatch, written)

    with pytest.raises(ValueError, match="persist_sc_requires_iteration_head"):
        gw_output.persist_w0_and_head(
            np.zeros((1, 1, 1), dtype=np.complex128),
            tensors_filename=str(restart),
            head_resolver=_HeadSource({}),
            iteration_head=None,
            config=_config(material_class="insulator"),
            meta=SimpleNamespace(n_rmu=1),
            mesh_xy=object(),
            print_fn=lambda *_args: None,
        )

    assert written == {}


def test_the_final_qsgw_map_threads_its_head_to_the_writer():
    """Pin both halves of the production seam, not only the writer signature."""
    gw_dir = Path(gw_output.__file__).resolve().parent
    sc_tree = ast.parse((gw_dir / "sc_iteration.py").read_text())
    outputs = next(
        node for node in sc_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SCOutputs")
    screening_payload = next(
        node for node in sc_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "SCMapScreeningArtifacts")
    payload_fields = {
        node.target.id for node in screening_payload.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)}
    assert payload_fields == {"static_w", "iteration_head", "static_head_terms"}
    assert any(
        isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "screening"
        for node in outputs.body)

    tree = sc_tree
    persist_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "persist_w0_and_head"
    ]
    assert len(persist_calls) == 1
    assert all(
        any(keyword.arg == "iteration_head" for keyword in call.keywords)
        for call in persist_calls
    )
    driver = next(
        node for node in sc_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sc_driver")
    gates = [
        node for node in ast.walk(driver)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("GATE sc_final_map_requires_iteration_head")]
    assert gates, "the final-map head gate must not depend on restart writes"


def test_final_map_capture_is_released_even_when_persistence_is_skipped():
    """The driver's ownership seam must release the pre-unfold W in finally."""
    gw_dir = Path(gw_output.__file__).resolve().parent
    tree = ast.parse((gw_dir / "sc_iteration.py").read_text())
    driver = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_sc_driver")
    persistence_try = next(
        node for node in ast.walk(driver)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.Call)
            and getattr(child.func, "id", None) == "persist_w0_and_head"
            for child in ast.walk(node)))
    cleanup_calls = [
        child for statement in persistence_try.finalbody
        for child in ast.walk(statement)
        if isinstance(child, ast.Call)
        and getattr(child.func, "id", None) == "take_pre_unfold"]
    assert len(cleanup_calls) == 1
    assert (len(cleanup_calls[0].args) == 1
            and isinstance(cleanup_calls[0].args[0], ast.Constant)
            and cleanup_calls[0].args[0].value == "W0_qmunu")


def test_qp_diagonal_head_diagnostic_is_rotated_to_dft_on_four_devices():
    """Exercise the host-diagonal/sharded-U seam, not a one-device surrogate."""
    jax = pytest.importorskip("jax")
    from jax.sharding import Mesh
    from gw import sc_iteration

    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires a 2x2 device mesh")
    mesh = Mesh(np.asarray(devices[:4]).reshape(2, 2), ("x", "y"))
    c, s = np.sqrt(0.75), 0.5
    U = sc_iteration._place(
        np.asarray([[[c, -s], [s, c]]], dtype=np.complex128),
        mesh, sc_iteration._band_rotation_spec())
    diag_qp = np.asarray([1.0, 3.0], dtype=np.complex128)
    got = sc_iteration._rotate_head_diagonal_to_dft(
        diag_qp, U, mesh=mesh)

    np.testing.assert_allclose(np.asarray(got), [[1.5, 2.5]], rtol=0, atol=1e-7)
