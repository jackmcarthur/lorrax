"""The SC Hamiltonian is one all-P carrier from seed through rCROP.

Login-node cells cover algebra and source structure on a 1x1 mesh.  The
multi-device layout/collective claim remains a P4 gate: a 1x1 NamedSharding
can prove the declared contract but cannot prove that GSPMD kept it.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                      # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import (                           # noqa: E402
    device_put_process_local,
    gather_to_host,
    resolve_mesh,
    single_device_mesh,
)
from gw import sc_iteration                                  # noqa: E402


def _mesh1():
    # Process-local by contract: safe when this cell is selected beside a
    # one-process-per-GPU P4 cell.
    return single_device_mesh()


def test_seed_and_iteration_zero_inspection_keep_the_matrix_layout():
    mesh = _mesh1()
    E = jnp.asarray([
        [-1.25, -0.4, 0.3, 1.7],
        [-1.1, -0.35, 0.45, 1.9],
    ], dtype=jnp.float64)
    from_diag, inspect_initial, identity = (
        sc_iteration._sc_carrier_kernels(mesh))

    H = from_diag(E)
    E_got, exact = inspect_initial(H)
    U = identity(E_got)

    assert isinstance(H.sharding, NamedSharding)
    assert H.sharding.spec == sc_iteration._band_rotation_spec()
    assert U.sharding.spec == sc_iteration._band_rotation_spec()
    assert E_got.sharding.spec == jax.sharding.PartitionSpec(None, None)
    assert bool(exact)
    np.testing.assert_array_equal(np.asarray(E_got), np.asarray(E))
    np.testing.assert_array_equal(
        np.asarray(H),
        np.asarray(E)[:, :, None] * np.eye(E.shape[1])[None])

    # Red twin: the gate is exact, not a tolerance-based diagonality claim.
    H_nondiag = H.at[0, 0, 1].set(1.0e-30j)
    _, exact_nondiag = inspect_initial(H_nondiag)
    assert not bool(exact_nondiag)


def test_complex_rotation_returns_the_canonical_matrix_spec_and_parity():
    mesh = _mesh1()
    rng = np.random.default_rng(31)
    nk, nb = 3, 5
    raw = (rng.normal(size=(nk, nb, nb))
           + 1j * rng.normal(size=(nk, nb, nb)))
    O = 0.5 * (raw + np.conj(np.swapaxes(raw, -1, -2)))
    U = np.empty_like(O)
    for k in range(nk):
        q, _ = np.linalg.qr(
            rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
        U[k] = q

    spec = sc_iteration._band_rotation_spec()
    sh = NamedSharding(mesh, spec)
    got = sc_iteration._rotate_to_dft_basis(
        jax.device_put(O, sh), jax.device_put(U, sh), mesh=mesh)
    want = U @ O @ np.conj(np.swapaxes(U, -1, -2))

    assert got.sharding.spec == spec
    np.testing.assert_allclose(np.asarray(got), want, rtol=2e-14, atol=2e-14)


def test_sc_module_has_no_private_repeated_eigensolver():
    """Every nontrivial SC spectrum routes through distrib_la's service."""
    src = inspect.getsource(sc_iteration)
    assert "def _make_kshard_eigh(" not in src
    assert "def _kshard_eigh_kernels(" not in src
    assert "jnp.linalg.eigh" not in src
    assert "jnp.linalg.eigvalsh" not in src
    spectrum = inspect.getsource(sc_iteration._sc_eigenvalues)
    assert "_sc_eigh_bands(" in spectrum


def test_rcrop_map_boundary_changes_extent_not_layout():
    body = inspect.getsource(sc_iteration._run_rcrop)
    seam = body[body.index("def _to_carry("):
                body.index("# Bookkeeping")]
    assert "_place(" not in seam
    assert seam.count("with_sharding_constraint") == 2
    assert "entry_sh" in seam
    assert "A[:, :nb, :nb]" in seam


def test_only_small_or_terminal_values_cross_to_host_in_sc_carrier_helpers():
    initial = inspect.getsource(sc_iteration.make_initial_state_from_dft)
    iteration = inspect.getsource(sc_iteration.gw_iteration_map)
    scissor = inspect.getsource(sc_iteration._scissor_E_qp_for_outofrange)
    terminal = inspect.getsource(sc_iteration._terminal_replicate)

    assert "np.eye(nb_active)" not in initial
    assert "np.asarray(state.H_qp_dft)" not in iteration
    assert "_sc_matrix_diagonal(" in scissor
    assert "gather_to_host(state.H_qp_dft)" not in iteration
    assert "gather_to_host(is_exact_diagonal)" in iteration
    assert "gather_to_host" in terminal


@pytest.mark.mesh(4)
def test_nondivisible_carrier_and_repeated_eigh_keep_logical_shape_on_p4():
    """Hostile nb=5 on a 2x2 mesh: pad internally, return logical shapes.

    This is intentionally the direct falsifier for the risky boundary: the
    SC seed lands at the all-P matrix spec, and every repeated spectrum goes
    through the qsgw_density/distrib_la service-owned padding seam.
    """
    if len(jax.devices()) != 4:
        pytest.skip("requires the canonical four-device process mesh")
    mesh = resolve_mesh()
    rng = np.random.default_rng(20260830)
    nk, nb = 2, 5
    E0_host = np.sort(rng.normal(size=(nk, nb)), axis=1)
    E0 = device_put_process_local(
        E0_host, NamedSharding(mesh, P(None, None)))
    from_diag, inspect_initial, _ = sc_iteration._sc_carrier_kernels(mesh)
    H = from_diag(E0)
    E_inspect, exact = inspect_initial(H)

    assert H.shape == (nk, nb, nb)
    assert H.sharding.spec == sc_iteration._band_rotation_spec()
    assert bool(np.asarray(gather_to_host(exact)))
    np.testing.assert_array_equal(np.asarray(E_inspect), E0_host)

    config = SimpleNamespace(backend=SimpleNamespace(
        distrib_la_batched_route="batch_reshard"))
    for _ in range(3):
        E, U = sc_iteration._sc_eigh_bands(
            H, kind="native", mesh_xy=mesh, config=config)
        assert E.shape == (nk, nb)
        assert U.shape == (nk, nb, nb)
        assert U.sharding.spec == sc_iteration._band_rotation_spec()
        np.testing.assert_allclose(np.asarray(E), E0_host,
                                   rtol=0.0, atol=2.0e-14)

    # The inspection is allowed to gather its O(nk*nb) diagonal and reduce
    # one scalar.  It must not all-gather the O(nk*nb^2) complex carrier.
    hlo = inspect_initial.lower(H).compile().as_text().lower()
    full_shape = f"c128[{nk},{nb},{nb}]"
    assert not any(
        "all-gather" in line and full_shape in line.replace(" ", "")
        for line in hlo.splitlines())


def _synthetic_inputs(mesh):
    return SimpleNamespace(
        mesh_xy=mesh,
        eigh_kind="native",
        config=SimpleNamespace(
            sc=SimpleNamespace(dump_dir=None),
            backend=SimpleNamespace(distrib_la_batched_route="batch_reshard"),
        ),
        partition=SimpleNamespace(
            protected_mask=np.ones(4, dtype=bool),
            in_range_mask=np.ones(4, dtype=bool),
        ),
        print_fn=lambda *_a, **_k: None,
        input_dir=".",
    )


@pytest.mark.mesh(4)
@pytest.mark.parametrize(
    "accelerator,max_iter,want_maps,want_eigh",
    [("linear", 1, 1, 2),
     ("linear", 2, 2, 5),
     ("rcrop", 1, 3, 7)],
)
def test_accelerators_keep_one_carrier_and_one_eigh_owner_on_p4(
        monkeypatch, accelerator, max_iter, want_maps, want_eigh):
    """One-shot, repeated linear, and repeated rCROP calls on one layout."""
    if len(jax.devices()) != 4:
        pytest.skip("requires the canonical four-device process mesh")
    mesh = resolve_mesh()
    inputs = _synthetic_inputs(mesh)
    E0_host = np.asarray([[-1.2, -0.3, 0.4, 1.1],
                          [-1.0, -0.2, 0.5, 1.3]])
    E0 = device_put_process_local(
        E0_host, NamedSharding(mesh, P(None, None)))
    from_diag, _, _ = sc_iteration._sc_carrier_kernels(mesh)
    H0 = from_diag(E0)
    target = H0.at[:, 0, 1].set(0.04 + 0.025j)
    target = target.at[:, 1, 0].set(0.04 - 0.025j)
    state0 = sc_iteration.SCState(H_qp_dft=H0, iteration=0)

    # Keep this as an SC/dataflow test; artifact serialization has its own
    # gates and would obscure exactly which map/eigh calls are counted.
    monkeypatch.setattr(sc_iteration, "_clear_sc_eqp_snapshots",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(sc_iteration, "_write_sc_eqp_snapshot",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(sc_iteration, "_maybe_dump_e_history",
                        lambda *_a, **_k: None)

    map_calls = []

    def fake_map(state, _inputs):
        map_calls.append(state.H_qp_dft.sharding.spec)
        H_out = 0.7 * state.H_qp_dft + 0.3 * target
        return sc_iteration.SCState(
            H_qp_dft=H_out,
            iteration=state.iteration + 1,
            occupation_state=state.occupation_state,
            head_surface_weight_kn=state.head_surface_weight_kn,
            outputs=SimpleNamespace(call=len(map_calls)),
        )

    monkeypatch.setattr(sc_iteration, "gw_iteration_map", fake_map)
    from gw import qsgw_density
    service = qsgw_density.distributed_eigh_bands
    eigh_calls = []

    def counted_service(*args, **kwargs):
        eigh_calls.append(1)
        return service(*args, **kwargs)

    monkeypatch.setattr(qsgw_density, "distributed_eigh_bands",
                        counted_service)
    final, _ = sc_iteration.run_self_consistency(
        state0, inputs, max_iter=max_iter, tol_ev=0.0,
        accelerator=accelerator, history_depth=2, mixing=0.6)

    spec = sc_iteration._band_rotation_spec()
    assert map_calls == [spec] * want_maps
    assert len(eigh_calls) == want_eigh
    assert final.H_qp_dft.sharding.spec == spec
