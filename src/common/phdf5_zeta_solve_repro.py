"""Small zeta-like PHDF5 FFI write reproducer.

This keeps the important producer path from ``fit_zeta_chunked_to_h5``:

  C_q --compute_L_q_from_CCT--> L_q
  Z_q --solve_zeta_from_L_q--> zeta(q, mu, r)
  zeta.transpose(0, 2, 1) --SlabIO PHDF5_FFI--> HDF5

It avoids wavefunctions, FFTs, and pair-density accumulation.  The point is to
separate "PHDF5 can write synthetic arrays" from "PHDF5 can write arrays
produced by the zeta solve".

Usage under Shifter, from the source root::

    LORRAX_NGPU=4 LORRAX_NTASKS=4 LORRAX_NNODES=1 \\
      src/ffi/common/cpp/run_shifter.sh \\
      python3 -u -m common.phdf5_zeta_solve_repro --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys

from runtime import set_default_env
set_default_env()

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime import init_jax_distributed
from common.isdf_fitting import compute_L_q_from_CCT, solve_zeta_from_L_q
from file_io.slab_io import SlabIO
from gw.gw_config import SlabIOBackend


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _parse_mesh(spec: str, world: int) -> tuple[int, int]:
    if spec:
        px_s, py_s = spec.lower().split("x")
        px, py = int(px_s), int(py_s)
    elif world == 16:
        px, py = 4, 4
    elif world == 4:
        px, py = 2, 2
    else:
        px, py = world, 1
    if px * py != world:
        raise ValueError(f"mesh {px}x{py} != process_count {world}")
    return px, py


def _build_hpd(nq: int, nmu: int, mesh: Mesh, seed: int) -> jax.Array:
    sh = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def build() -> jax.Array:
        k_re, k_im = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_re, (nq, nmu, nmu), dtype=jnp.float64)
        b = jax.random.normal(k_im, (nq, nmu, nmu), dtype=jnp.float64)
        z = (a + 1j * b).astype(jnp.complex128)
        c = z @ jnp.swapaxes(jnp.conj(z), -1, -2)
        c = c + (10.0 * nmu) * jnp.eye(nmu, dtype=jnp.complex128)[None, :, :]
        return jax.lax.with_sharding_constraint(c, sh)

    out = build()
    out.block_until_ready()
    return out


def _build_rhs(nq: int, nmu: int, ncols: int, mesh: Mesh, seed: int) -> jax.Array:
    sh = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def build() -> jax.Array:
        k_re, k_im = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_re, (nq, nmu, ncols), dtype=jnp.float64)
        b = jax.random.normal(k_im, (nq, nmu, ncols), dtype=jnp.float64)
        z = (a + 1j * b).astype(jnp.complex128)
        return jax.lax.with_sharding_constraint(z, sh)

    out = build()
    out.block_until_ready()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="", help="Mesh, e.g. 2x2 or 4x4.")
    ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--nmu", type=int, default=64)
    ap.add_argument("--ncols", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--q-chunk-size", type=int, default=0)
    ap.add_argument(
        "--outer-jit",
        action="store_true",
        help="Build RHS and solve inside one enclosing jit, like fit_one_rchunk.",
    )
    ap.add_argument("--path", default="/tmp/phdf5_zeta_solve_repro.h5")
    args = ap.parse_args()

    init_jax_distributed()

    world = jax.process_count()
    px, py = _parse_mesh(args.mesh, world)
    if args.nmu % px or args.nmu % py:
        raise ValueError(f"nmu={args.nmu} must be divisible by mesh axes {px} and {py}")
    if args.ncols % world:
        raise ValueError(f"ncols={args.ncols} must be divisible by process_count {world}")

    devices = np.asarray(jax.devices()).reshape(px, py)
    mesh = Mesh(devices, axis_names=("x", "y"))
    q_chunk_size = args.q_chunk_size or args.nq

    path = os.path.abspath(args.path)
    if jax.process_index() == 0:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    multihost_utils.sync_global_devices("remove_old_file")

    _log(
        f"world={world} mesh={px}x{py} nq={args.nq} nmu={args.nmu} "
        f"ncols={args.ncols} iters={args.iters} q_chunk_size={q_chunk_size}")

    with mesh:
        C_q = _build_hpd(args.nq, args.nmu, mesh, seed=17)
        L_q = compute_L_q_from_CCT(C_q, mesh)
        L_q.block_until_ready()
        _log(f"L_q shape={L_q.shape} spec={getattr(L_q.sharding, 'spec', None)}")

        rhs_sh = NamedSharding(mesh, P(None, "x", "y"))

        @jax.jit
        def build_solve_outer(L_q_arg: jax.Array, seed: jax.Array) -> jax.Array:
            k_re, k_im = jax.random.split(jax.random.key(seed), 2)
            a = jax.random.normal(
                k_re, (args.nq, args.nmu, args.ncols), dtype=jnp.float64)
            b = jax.random.normal(
                k_im, (args.nq, args.nmu, args.ncols), dtype=jnp.float64)
            Z_q = (a + 1j * b).astype(jnp.complex128)
            Z_q = jax.lax.with_sharding_constraint(Z_q, rhs_sh)
            return solve_zeta_from_L_q(
                L_q_arg, Z_q, mesh, q_chunk_size=q_chunk_size, vertex_mu_L=0)

        with SlabIO(path, mode="w", mesh=mesh, backend=SlabIOBackend.PHDF5_FFI) as io:
            io.create_dataset(
                "zeta_q",
                shape=(args.nq, args.iters * args.ncols, args.nmu),
                dtype=np.complex128,
                chunks=(1, args.ncols, args.nmu),
            )
            for it in range(args.iters):
                if args.outer_jit:
                    zeta = build_solve_outer(
                        L_q, jnp.asarray(100 + it, dtype=jnp.uint32))
                else:
                    Z_q = _build_rhs(args.nq, args.nmu, args.ncols, mesh, seed=100 + it)
                    zeta = solve_zeta_from_L_q(
                        L_q, Z_q, mesh, q_chunk_size=q_chunk_size, vertex_mu_L=0)
                zeta.block_until_ready()
                A = zeta.transpose(0, 2, 1)
                A.block_until_ready()
                if jax.process_index() == 0:
                    print(
                        f"write {it}: A.shape={A.shape} "
                        f"spec={getattr(A.sharding, 'spec', None)} "
                        f"local={[s.data.shape for s in A.addressable_shards]}",
                        flush=True,
                    )
                io.write_slab(
                    "zeta_q",
                    A,
                    offset=(0, it * args.ncols, 0),
                    global_shape=(args.nq, args.iters * args.ncols, args.nmu),
                )

    if jax.process_index() == 0:
        with h5py.File(path, "r") as f:
            d = f["zeta_q"]
            print(f"wrote {path}: shape={d.shape} dtype={d.dtype}", flush=True)
        os.remove(path)
        print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
