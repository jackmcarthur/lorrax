"""Algebra and structure gates for the one-dispatch tiled q=0 Gram."""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh11() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _problem(seed: int = 20260901, *, npoint: int = 7):
    rng = np.random.default_rng(seed)
    nk, nl, nr, ns = 2, 3, 5, 4
    left = (rng.standard_normal((nk, nl, ns, npoint))
            + 1j * rng.standard_normal((nk, nl, ns, npoint)))
    right = (rng.standard_normal((nk, nr, ns, npoint))
             + 1j * rng.standard_normal((nk, nr, ns, npoint)))
    return (
        left,
        right,
        jnp.asarray(np.conj(left).transpose(0, 3, 1, 2)),
        jnp.asarray(left),
        jnp.asarray(np.conj(right).transpose(0, 3, 1, 2)),
        jnp.asarray(right),
        jnp.asarray([0.375, 0.625], dtype=jnp.float64),
    )


def _slice_pad(a, start: int, *, axis: int, size: int):
    ids = jnp.arange(start, start + size, dtype=jnp.int32)
    safe = jnp.minimum(ids, jnp.int32(a.shape[axis] - 1))
    out = jnp.take(a, safe, axis=axis)
    mask_shape = [1] * a.ndim
    mask_shape[axis] = size
    return jnp.where(
        (ids < a.shape[axis]).reshape(mask_shape), out,
        jnp.zeros((), dtype=a.dtype))


def _incumbent_tile_dispatches(faces, weights, mesh, *, width, mode):
    """Literal one-device form of the former c-outer/r-inner schedule."""
    from isdf import gram_q0_from_psi_sm

    left_x, left_y, right_x, right_y = faces
    npoint = int(left_x.shape[1])
    G = jnp.zeros((npoint, npoint), dtype=left_x.dtype)
    for c0 in range(0, npoint, width):
        for r0 in range(0, npoint, width):
            tile = gram_q0_from_psi_sm(
                _slice_pad(left_x, r0, axis=1, size=width),
                _slice_pad(left_y, c0, axis=3, size=width),
                _slice_pad(right_x, r0, axis=1, size=width),
                _slice_pad(right_y, c0, axis=3, size=width),
                weights, mesh_xy=mesh, gamma_mode=mode, symmetrize=False,
            )
            rows = r0 + jnp.arange(width, dtype=jnp.int32)
            cols = c0 + jnp.arange(width, dtype=jnp.int32)
            G = G.at[rows[:, None], cols[None, :]].set(tile, mode="drop")
    return G


@pytest.mark.parametrize("mode", ("charge", "transverse"))
@pytest.mark.parametrize("npoint,width", ((7, 3), (6, 3)))
def test_tiled_scan_is_bit_exact_to_incumbent_dispatches(
        mode, npoint, width):
    """Fusion preserves tile arithmetic/order with and without a tail."""
    from isdf import gram_q0_tiled_from_psi_sm

    _, _, left_x, left_y, right_x, right_y, weights = _problem(
        npoint=npoint)
    mesh = _mesh11()
    faces = (left_x, left_y, right_x, right_y)
    expected = _incumbent_tile_dispatches(
        faces, weights, mesh, width=width, mode=mode)
    got = gram_q0_tiled_from_psi_sm(
        jnp.zeros((npoint, npoint), dtype=jnp.complex128), *faces, weights,
        mesh_xy=mesh, tile_width=width, gamma_mode=mode,
    )
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_tiled_transverse_keeps_gamma2_conjugation_and_all_components():
    """The oracle distinguishes the PSD feature Gram from raw CCT phases."""
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from isdf import gram_q0_tiled_from_psi_sm, pair_density

    left, right, left_x, left_y, right_x, right_y, weights = _problem(
        seed=260901, npoint=5)
    mesh = _mesh11()
    got = np.asarray(gram_q0_tiled_from_psi_sm(
        jnp.zeros((5, 5), dtype=jnp.complex128),
        left_x, left_y, right_x, right_y, weights,
        mesh_xy=mesh, tile_width=3, gamma_mode="transverse",
    ))
    gamma = np.asarray(jax.device_get(jnp.stack((gamma1, gamma2, gamma3))))
    features = np.einsum(
        "knsa,ist,kmta->ikanm", np.conj(left), gamma, right,
        optimize=True)
    expected = np.einsum(
        "k,ikanm,ikbnm->ab", np.asarray(weights),
        np.conj(features), features, optimize=True)

    # Negative control: applying gamma_i rather than gamma_i* on the first
    # endpoint is the raw transverse CCT convention.  gamma2 then contributes
    # with the wrong sign; random complex spinors make the difference O(1).
    P_l = np.asarray(pair_density(left_x, left_y, mesh))
    P_r = np.asarray(pair_density(right_x, right_y, mesh))
    wrong = np.zeros((5, 5), dtype=np.complex128)
    for gamma_i in gamma:
        prod = np.einsum(
            "kabxy,aA,bB,kABxy->kxy", np.conj(P_l), gamma_i, gamma_i,
            P_r, optimize=True)
        wrong += np.einsum("k,kxy->xy", np.asarray(weights), prod)
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert np.max(np.abs(wrong - expected)) > 1.0e-3 * scale
    np.testing.assert_allclose(got, expected, rtol=4e-13, atol=4e-13)


def test_tiled_executor_scan_and_donation_are_explicit_in_source():
    """The structural acceleration cannot regress into Python unrolling."""
    from isdf import core

    tree = ast.parse(textwrap.dedent(inspect.getsource(
        core._gram_q0_tiled_from_psi_kernel)))
    scans = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "scan"]
    assert len(scans) == 1
    unroll = next((kw.value for kw in scans[0].keywords
                   if kw.arg == "unroll"), None)
    assert isinstance(unroll, ast.Constant) and unroll.value == 1

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    jits = [
        node for node in calls
        if ((isinstance(node.func, ast.Attribute)
             and node.func.attr == "jit")
            or (isinstance(node.func, ast.Name)
                and node.func.id == "partial"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "jit"))
    ]
    donate = [kw.value for call in jits for kw in call.keywords
              if kw.arg == "donate_argnums"]
    assert len(donate) == 1
    assert ast.literal_eval(donate[0]) == (0,)
    scatter_updates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "at"
    ]
    assert scatter_updates == []
    dynamic_stores = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dynamic_update_slice"
    ]
    assert dynamic_stores


def test_terminal_hermitian_fold_preserves_value_and_xy_layout():
    from centroid.pivoted_cholesky import (
        _candidate_gram_hermitian_fold_kernel,
        _candidate_gram_zero_kernel,
    )
    from common.collectives import device_put_process_local

    mesh = _mesh11()
    xy = NamedSharding(mesh, P("x", "y"))
    raw = np.asarray([
        [1.0 + 0.2j, 2.0 - 0.4j, -0.5 + 0.7j],
        [0.3 + 0.1j, -2.0 + 0.8j, 1.1 - 0.2j],
        [0.6 - 0.9j, -0.7 + 0.5j, 3.0 - 0.3j],
    ], dtype=np.complex128)
    fold = _candidate_gram_hermitian_fold_kernel(mesh)
    assert fold is _candidate_gram_hermitian_fold_kernel(mesh)
    assert (_candidate_gram_zero_kernel(mesh, 3)
            is _candidate_gram_zero_kernel(mesh, 3))
    got = fold(
        device_put_process_local(raw, xy))
    assert tuple(got.sharding.spec) == ("x", "y")
    np.testing.assert_array_equal(
        np.asarray(got), 0.5 * (raw + np.conj(raw.T)))
