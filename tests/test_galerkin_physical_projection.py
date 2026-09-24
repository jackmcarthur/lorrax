"""P4 placement and parity gate for Galerkin physical coefficients."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax as _jax_boot

    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _kwargs = {"local_device_ids": [0]} if _visible and "," not in _visible else {}
    _jax_boot.distributed.initialize(**_kwargs)

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from isdf.galerkin import (
    GalerkinOperatorProjection,
    _coefficients_from_projection,
    _make_basis_solve_kernel,
    _make_projection_accum_kernel,
    _make_spin_operator_fold_kernel,
    _reduce_device_partials,
    plan_galerkin_operator_stream,
    rotate_galerkin_operator,
)


def _mesh(count: int = 4) -> Mesh:
    devices = jax.devices()
    if len(devices) >= count:
        side = int(np.sqrt(count))
        return Mesh(
            np.asarray(devices[:count]).reshape(side, side), ("x", "y"))
    return Mesh(np.asarray(devices[:1]).reshape(1, 1), ("x", "y"))


def _put_global(value: np.ndarray, sharding: NamedSharding):
    value = np.asarray(value)
    return jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index])


def _host(value) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(value))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


@pytest.mark.mesh(4)
def test_physical_solve_and_projection_are_local_r_blocks_with_exact_parity():
    mesh = _mesh()
    rank, ns, r_extent = 4, 2, 32
    nk, band_carrier = 2, 3
    row = NamedSharding(mesh, P(None, None, ("y", "x")))
    psi_layout = NamedSharding(mesh, P(None, None, None, ("y", "x")))
    rep = NamedSharding(mesh, P())

    rng = np.random.default_rng(17)
    factor_np = np.tril(
        rng.normal(size=(rank, rank))
        + 1j * rng.normal(size=(rank, rank)))
    factor_np[np.diag_indices(rank)] += 6.0
    factor_np = factor_np.astype(np.complex128)
    rows_np = (
        rng.normal(size=(rank, ns, r_extent))
        + 1j * rng.normal(size=(rank, ns, r_extent))).astype(np.complex128)

    factor = jax.device_put(factor_np, rep)
    rows = _put_global(rows_np, row)
    solve = _make_basis_solve_kernel(
        mesh=mesh, rank=rank, nspinor=ns,
        r_carrier=r_extent, row_layout=row)
    solve_compiled = solve.lower(factor, rows).compile()
    solve_hlo = solve_compiled.as_text().lower()
    assert "all-gather" not in solve_hlo and "all_gather" not in solve_hlo
    local_r = r_extent // int(mesh.size)
    solve_mem = solve_compiled.memory_analysis()
    full_rhs_pair = 2 * rank * ns * r_extent * np.dtype(np.complex128).itemsize
    solve_peak = (
        int(solve_mem.temp_size_in_bytes)
        + int(solve_mem.argument_size_in_bytes)
        + int(solve_mem.output_size_in_bytes)
        - int(solve_mem.alias_size_in_bytes))
    assert solve_peak < full_rhs_pair, (
        f"compiled solve peak {solve_peak} is not below one full-r input+output "
        f"pair {full_rhs_pair}; local-r={local_r}")

    basis = solve(factor, rows)
    expected_basis = np.moveaxis(
        np.stack([
            np.linalg.solve(factor_np, rows_np[:, spin, :])
            for spin in range(ns)
        ]), 0, 1)
    np.testing.assert_allclose(
        _host(basis), expected_basis, rtol=2e-13, atol=2e-13)

    # The fit's projection: per-device partials of Psi X^H with no
    # collective, one reduction, then C = (Psi X^H) L^-H on the rank face.
    psi_np = (
        rng.normal(size=(nk, band_carrier, ns, r_extent))
        + 1j * rng.normal(size=(nk, band_carrier, ns, r_extent))
    ).astype(np.complex128)
    psi = _put_global(psi_np, psi_layout)
    x_rows = _put_global(rows_np, row)          # the solve donated ``rows``
    accum = _make_projection_accum_kernel(
        mesh=mesh, nk=nk, band_carrier=band_carrier, rank=rank,
        nspinor=ns, r_carrier=r_extent)
    acc_layout = NamedSharding(mesh, P(("x", "y"), None, None))
    acc = _put_global(np.zeros((int(mesh.size) * nk, band_carrier, rank),
                               dtype=np.complex128), acc_layout)
    accum_hlo = accum.lower(psi, x_rows, acc).compile().as_text().lower()
    for collective in ("all-gather", "all-reduce", "all-to-all",
                       "reduce-scatter", "collective-permute"):
        assert collective not in accum_hlo, collective
    projection = _reduce_device_partials(accum(psi, x_rows, acc), mesh)
    expected_projection = np.einsum(
        "kbsr,asr->kba", psi_np, np.conj(rows_np), optimize=True)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(projection)), expected_projection,
        rtol=2e-13, atol=2e-12)

    coefficients = jax.jit(
        _coefficients_from_projection, in_shardings=(rep, rep),
        out_shardings=rep)(projection, factor)
    expected = np.einsum(
        "kbsr,asr->kba", psi_np, np.conj(expected_basis), optimize=True)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(coefficients)), expected,
        rtol=2e-12, atol=2e-12)


@pytest.mark.mesh(16)
def test_full_grid_spin_operator_stays_on_rank_face_and_rotates_coefficients():
    mesh = _mesh(16)
    rank, ns, r_extent = 8, 2, 32
    nq, nb = 5, 3
    left = NamedSharding(mesh, P("x", None, "y"))
    right = NamedSharding(mesh, P("y", None, "x"))
    face = NamedSharding(mesh, P("x", "y"))
    rep = NamedSharding(mesh, P())

    panel_budget = 64 * 1024
    plan = plan_galerkin_operator_stream(
        rank=rank, nspinor=ns, n_rtot=97, mesh_xy=mesh,
        q_tile_budget=panel_budget)
    assert plan.q_tile_local_bytes <= panel_budget
    assert plan.max_r_carrier % int(mesh.size) == 0

    rng = np.random.default_rng(23)
    basis_np = (
        rng.normal(size=(rank, ns, r_extent))
        + 1j * rng.normal(size=(rank, ns, r_extent))).astype(np.complex128)
    spin_np = np.asarray([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    basis_left = _put_global(basis_np, left)
    basis_right = _put_global(basis_np, right)
    spin = jax.device_put(spin_np, rep)
    zero_operator = _put_global(
        np.zeros((rank, rank), dtype=np.complex128), face)
    zero_metric = _put_global(
        np.zeros((rank, rank), dtype=np.complex128), face)

    fold = _make_spin_operator_fold_kernel(
        rank=rank, nspinor=ns, r_carrier=r_extent,
        mesh=mesh, basis_left_layout=left, basis_right_layout=right,
        face_layout=face)
    compiled = fold.lower(
        basis_left, basis_right, spin, zero_operator, zero_metric).compile()
    hlo = compiled.as_text().lower()
    if int(mesh.size) > 1:
        assert "all-gather" in hlo or "all_gather" in hlo
        assert "reduce-scatter" not in hlo and "reduce_scatter" not in hlo
        assert "all-reduce" not in hlo and "all_reduce" not in hlo

    operator, metric = fold(
        basis_left, basis_right, spin, zero_operator, zero_metric)
    expected_operator = np.einsum(
        "asr,st,btr->ab", np.conj(basis_np), spin_np, basis_np,
        optimize=True)
    expected_metric = np.einsum(
        "asr,bsr->ab", np.conj(basis_np), basis_np, optimize=True)
    np.testing.assert_allclose(
        _host(operator), expected_operator, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        _host(metric), expected_metric, rtol=2e-13, atol=2e-13)

    coeff_np = (
        rng.normal(size=(nq, rank, nb))
        + 1j * rng.normal(size=(nq, rank, nb))).astype(np.complex128)
    result = rotate_galerkin_operator(
        jax.device_put(coeff_np, rep),
        GalerkinOperatorProjection(operator=operator, metric=metric), mesh,
        logical_q_count=nq)
    expected_num = np.einsum(
        "qan,ab,qbn->qn", np.conj(coeff_np), expected_operator, coeff_np,
        optimize=True)
    expected_norm = np.einsum(
        "qan,ab,qbn->qn", np.conj(coeff_np), expected_metric, coeff_np,
        optimize=True)
    np.testing.assert_allclose(
        _host(result.numerator), expected_num, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        _host(result.norm), expected_norm, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        _host(result.value), expected_num.real / expected_norm.real,
        rtol=3e-13, atol=3e-13)
