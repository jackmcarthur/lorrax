"""Debug/benchmark helpers for ring-collective BSE matvecs.

This module intentionally collects non-core utilities (timing, micro-benchmarks)
in one place to keep `bse_ring_comm.py` focused on kernels.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

import isdf.common.timing as timing

from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
from .bse_serial import apply_D


def ring_matvec_timing(
    input_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    px: int = 2,
    py: int = 2,
    n_repeat: int = 5,
    n_warmup: int = 1,
    component_timing: bool = True,
    low_mem: bool = True,
    use_nohead: bool = False,
) -> None:
    """Time the ring matvec on data loaded from a restart file."""
    restart_file = _find_restart_file(input_file)
    devices = jax.devices()
    if len(devices) < px * py:
        raise RuntimeError(
            f"Need {px*py} devices, found {len(devices)}. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=... before running."
        )
    mesh = Mesh(np.array(devices[: px * py]).reshape(px, py), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)

    timing.reset()
    with timing.section("bse_ring_debug.restart_load"):
        payload = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=n_val,
            n_cond=n_cond,
            fermi_energy=0.0,
            mesh_xy=mesh,
            pad_bands=True,
            use_nohead=use_nohead,
        )
        psi_c_X = payload["psi_c_X"]
        psi_c_Y = payload["psi_c_Y"]
        psi_v_X = payload["psi_v_X"]
        psi_v_Y = payload["psi_v_Y"]
        eps_c = payload["eps_c"]
        eps_v = payload["eps_v"]
        W_q = payload["W_q"]
        V_q0 = payload["V_q0"]
        nkx = int(payload["nkx"])
        nky = int(payload["nky"])
        nkz = int(payload["nkz"])
        n_cond_pad = int(payload["n_cond_pad"])
        n_val_pad = int(payload["n_val_pad"])
        nk = nkx * nky * nkz
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (1, n_cond_pad, n_val_pad, nk)) + 1j * jax.random.normal(
            key, (1, n_cond_pad, n_val_pad, nk)
        )

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c_X, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c_Y, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v_X, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v_Y, sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(W_q, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(V_q0, sh.V)
        X = jax.lax.with_sharding_constraint(X, sh.X)

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz, low_mem=low_mem)

        with timing.section("bse_ring_debug.matvec_compile"):
            W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
            W_R.block_until_ready()
            HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
            HX.block_until_ready()

        for _ in range(n_warmup):
            HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
            HX.block_until_ready()

        with timing.section("bse_ring_debug.matvec_run"):
            for _ in range(n_repeat):
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0)
                HX.block_until_ready()

        if component_timing:
            with timing.section("bse_ring_debug.D_term"):
                D_ring = apply_D(X, eps_c, eps_v)
                D_ring.block_until_ready()
            with timing.section("bse_ring_debug.V_term"):
                V_ring = matvec(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R * 0.0, V_q0
                )
                V_ring.block_until_ready()
            with timing.section("bse_ring_debug.W_term"):
                W_ring = matvec(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R, V_q0 * 0.0
                )
                W_ring.block_until_ready()

    timing.report(print_fn=print, title="--- Ring Matvec Timing ---")

