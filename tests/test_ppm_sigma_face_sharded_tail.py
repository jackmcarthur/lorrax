"""Real-mesh parity gate for the GN-PPM face-layout sharded tail.

The face Sigma kernel accumulates on the wider ``nb_full`` carrier, while
the published QP cube ends at ``nb_sigma``.  The replicated tail has always
assembled that wider carrier on host and selected its leading logical block.
This gate drives the same deterministic per-branch host tiles through both
production finalizers and requires bit equality after the sharded route's
canonical ``strip_sigma_window`` slice+reshard.

The hostile geometry is deliberately tiny and non-degenerate:
``nb_full=8``, ``nb_sigma=4`` on a real 2x2 mesh.  Values outside the logical
block are nonzero, so a gate that merely assumes a zero pad cannot pass.
There is no random input and no physics-kernel twin here: the seam under test
is movement-only finalization after the already-gated tau projection.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la", "minimax"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    # The module environment also carries a LORRAX checkout.  Put THIS
    # gate's source first so the exact commit under test owns ``gw``.
    _SRC = os.path.join(_REPO, "src")
    if _SRC in sys.path:
        sys.path.remove(_SRC)
    sys.path.insert(0, _SRC)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack

    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import argparse
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw import ppm_sigma
from gw.wavefunction_bundle import BandSlices


PX = PY = 2
NB_FULL = 8
NB_SIGMA = 4
NK = 2


def _tiny_problem(layout: str):
    slices = BandSlices.from_band_edges(
        0, 0, 2, NB_SIGMA, NB_FULL,
        b4_chi=NB_FULL, b4_sigma=NB_FULL)
    enk = jnp.asarray([
        [-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [-1.8, -0.8, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
    ], dtype=jnp.float64)
    occ = jnp.asarray([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float64)
    wfns = SimpleNamespace(
        layout="face", slices=slices, enk=enk, occ=occ)
    ppm = SimpleNamespace(
        B_q=jnp.ones((1, 2, 2), dtype=jnp.complex128),
        Omega_q=jnp.full((1, 2, 2), 2.0, dtype=jnp.float64),
        valid_mask_q=jnp.ones((1, 2, 2), dtype=bool),
    )
    meta = SimpleNamespace(
        nk_tot=NK, nkx=NK, nky=1, nkz=1,
        b_id_4_sigma_user=NB_FULL)
    ppm_cfg = SimpleNamespace(invalid_mode="zero")
    sigma_cfg = SimpleNamespace(
        regularization_ev=0.25,
        regularization_floor_ev=0.0,
        window_edge_factor=1.5,
        fermi_reference="midgap",
        omega_layout=layout,
    )
    quad = SimpleNamespace(
        target_error=1.0e-8,
        max_nodes=16,
        crossing_max_nodes=32,
        crossing_eps_q=1.0e-8,
        use_shipped_tables=True,
    )
    return wfns, ppm, meta, ppm_cfg, sigma_cfg, quad


def _analytic_branch_tiles(*, omega_nonneg_ry, log_tag, wfns, mesh_xy,
                           meta, **_unused):
    """Return nonzero-tail host tiles at the production finalizer seam."""
    n_omega = int(np.asarray(omega_nonneg_ry).size)
    shape = (1, n_omega, NK, NB_FULL, NB_FULL)
    sharding = NamedSharding(
        mesh_xy, P(None, None, None, "x", "y"))
    devices = list(sharding.addressable_devices)
    dmap = sharding.devices_indices_map(shape)

    # The production tags contain Greek omega.  Give each physical branch a
    # distinct exact coefficient so the cond/val fold is observable.
    if log_tag.startswith("ω≥"):
        branch_code = 1.0 if log_tag.endswith("cond") else 10.0
    elif log_tag.startswith("ω<"):
        branch_code = 100.0 if log_tag.endswith("cond") else 1000.0
    else:
        raise AssertionError(f"unexpected Sigma branch tag {log_tag!r}")

    iw = np.arange(n_omega, dtype=np.float64)[None, :, None, None, None]
    ik = np.arange(NK, dtype=np.float64)[None, None, :, None, None]
    im = np.arange(NB_FULL, dtype=np.float64)[None, None, None, :, None]
    jn = np.arange(NB_FULL, dtype=np.float64)[None, None, None, None, :]
    full = ((branch_code + 3.0 * iw + 5.0 * ik + 7.0 * im + 11.0 * jn)
            + 1j * (2.0 * branch_code + iw + 13.0 * im - 17.0 * jn))
    tile_index = [tuple(dmap[d]) for d in devices]
    tiles = [np.asarray(full[ix], dtype=np.complex128) for ix in tile_index]
    return ppm_sigma._SigmaBranchTiles(
        tiles=tiles,
        tile_index=tile_index,
        devices=devices,
        spatial_padded=(1, NK, NB_FULL, NB_FULL),
        sharding=sharding,
        nb_real=NB_SIGMA,
    ), []


def _to_host(x):
    from jax.experimental import multihost_utils as mhu
    return np.asarray(mhu.process_allgather(x, tiled=True))


def check_face_sharded_tail_parity(mesh):
    omega = np.asarray([-0.2, 0.1], dtype=np.float64)

    def run(layout):
        wfns, ppm, meta, ppm_cfg, sigma_cfg, quad = _tiny_problem(layout)
        with mock.patch.object(
                ppm_sigma, "_run_sigma_branch",
                side_effect=_analytic_branch_tiles):
            with mesh:
                return ppm_sigma.compute_sigma_c_ppm_omega_grid(
                    wfns, ppm, meta, mesh,
                    ppm_cfg=ppm_cfg,
                    sigma_cfg=sigma_cfg,
                    quad=quad,
                    omega_grid_ry=omega,
                    ansatz="gn_ppm",
                    print_fn=lambda *_a, **_k: None,
                ).sigma_c_kij

    replicated = run("replicated")
    sharded = jax.block_until_ready(run("sharded"))
    got = _to_host(sharded)
    want = np.asarray(jax.device_get(replicated))
    assert got.shape == want.shape == (1, omega.size, NK, NB_SIGMA, NB_SIGMA)
    assert tuple(sharded.sharding.spec) == (None, None, None, "x", "y")
    np.testing.assert_array_equal(got, want)


def test_face_sharded_tail_matches_replicated_bit_exactly():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} real processes for a live sharded tail; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_face_sharded_tail_parity(mesh)


def _cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *_a, **_k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if (px, py) != (PX, PY) or jax.device_count() != px * py:
        p0(f"REFUSE: this gate requires exactly {PX}x{PY}={PX * PY} "
           f"devices; got mesh={args.mesh}, devices={jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    try:
        check_face_sharded_tail_parity(mesh)
    except Exception as exc:
        p0(f"FAIL face_sharded_tail_parity: {type(exc).__name__}: {exc}")
        raise
    p0("PASS face_sharded_tail_parity: sharded == replicated bit-exact")
    p0("done: 1/1 cases passed")
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
