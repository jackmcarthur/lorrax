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
    _make_basis_solve_kernel,
    _make_physical_project_kernel,
)


def _mesh() -> Mesh:
    devices = jax.devices()
    if len(devices) >= 4:
        return Mesh(np.asarray(devices[:4]).reshape(2, 2), ("x", "y"))
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

    psi_np = (
        rng.normal(size=(nk, band_carrier, ns, r_extent))
        + 1j * rng.normal(size=(nk, band_carrier, ns, r_extent))
    ).astype(np.complex128)
    coeff_np = (
        rng.normal(size=(nk, band_carrier, rank))
        + 1j * rng.normal(size=(nk, band_carrier, rank))
    ).astype(np.complex128)
    psi = _put_global(psi_np, psi_layout)
    coefficients = jax.device_put(coeff_np, rep)
    project = _make_physical_project_kernel(
        mesh=mesh, nk=nk, band_carrier=band_carrier, rank=rank,
        nspinor=ns, r_carrier=r_extent,
        psi_layout=psi_layout, basis_layout=row)
    project_compiled = project.lower(psi, basis, coefficients).compile()
    project_hlo = project_compiled.as_text().lower()
    assert "all-gather" not in project_hlo and "all_gather" not in project_hlo
    assert "all-reduce" in project_hlo or "all_reduce" in project_hlo

    actual = project(psi, basis, coefficients)
    expected = coeff_np + np.einsum(
        "kbsr,asr->kba", psi_np, np.conj(expected_basis), optimize=True)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(actual)), expected,
        rtol=2e-13, atol=2e-13)
