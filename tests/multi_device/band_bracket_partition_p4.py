"""P=4 gate: the Σ_c band brackets PARTITION the band sum, on a real 2x2.

    srun -n 4 python3 tests/multi_device/band_bracket_partition_p4.py

WHY THIS EXISTS SEPARATELY FROM ``tests/test_band_extrapolation.py``.  That
file gates the same two claims in process on a 1x1 mesh, which is enough to
catch a bracketing arithmetic error and cheap enough to run in the default
suite.  What it CANNOT reach is the part of the bracket plumbing that only
exists at P>1:

  * the stacked sigma(tau) carries a new LEADING axis, and the shared device
    accumulator inserts omega behind it;
  * the reduce-scatter projector's output sharding has to survive the stack.

An in-process 4-device mesh cannot be used for this: the flat-k FFT FFI
aborts uncatchably under one process holding four devices (rc=-6, no junit —
``tests/harness`` names the case).  So this is a multi-process script, in the
directory the repo already keeps such gates in.

WHAT IT ASSERTS, on every rank, on every addressable shard:

  1. ``cumsum(3 brackets, axis=0)[-1] == 1 full-band bracket`` to 1e-12
     relative — the partition claim through the sole shared carrier;
  2. the single-bracket kernel is BIT-IDENTICAL (``max|Δ| == 0``) to the
     un-bracketed ``brackets=None`` kernel MPA still uses — the default-path
     claim, at the shape where a sharding drift would show;
  3. the stacked output's sharding is ``P(None, None, 'x', 'y')`` with the
     bracket axis replicated, because the tile bookkeeping downstream reads
     it and assumes exactly that.

Exit 0 on success; a nonzero exit with the failing quantity named otherwise.
Prints its device count and mesh so a leg that silently ran at P=1 is
visible in the log rather than passing quietly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P    # noqa: E402


def _fail(msg: str) -> None:
    print(f"[band-bracket-p4] FAIL: {msg}", flush=True)
    sys.exit(1)


def main() -> None:
    n_dev = jax.device_count()
    rank = jax.process_index()
    if n_dev != 4:
        _fail(f"needs exactly 4 global devices, jax reports {n_dev} "
              f"({jax.process_count()} process(es)) — this is an ABSENCE, "
              f"not a measurement")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    if rank == 0:
        print(f"[band-bracket-p4] processes={jax.process_count()} "
              f"devices={n_dev} mesh=2x2 platform={jax.devices()[0].platform}",
              flush=True)

    from gw.ppm_tau_kernel import (_get_sigma_kij_kernel,
                                   get_shared_sigma_tau_kernel)
    from gw.band_extrapolation import plan_band_brackets
    from common.collectives import device_put_process_local

    nk, nb, n_mu, m = 8, 12, 8, 4
    kgrid = (2, 2, 2)
    rng = np.random.default_rng(20260815)

    def c(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(np.complex128)

    def put(x, spec):
        return device_put_process_local(x, NamedSharding(mesh, spec))

    from full_photon_head_sigma_gate import _bundle
    from gw.wavefunction_bundle import BandSlices, parent_sigma_operands, sigma_face_kernel_kwargs
    psi = c(nk, nb, 1, n_mu)
    energy = np.abs(rng.standard_normal((nk, nb)))
    wfns = _bundle(mesh, psi, energy, np.zeros((nk, nb)),
        BandSlices.from_band_edges(0, 0, 0, m, nb), kgrid=kgrid)
    psi_xn, psi_yr, psi_xr, psi_yn, E_A, _ = parent_sigma_operands(wfns)
    face_kwargs = sigma_face_kernel_kwargs(wfns)
    mask_A = jnp.asarray(rng.random((nk, nb)) > 0.3)
    B_q = put(c(nk, n_mu, n_mu), P(None, 'x', 'y'))
    Omega_q = put(np.abs(rng.standard_normal((nk, n_mu, n_mu))) + 0.1,
                  P(None, 'x', 'y'))
    mask_B = put(np.ones((nk, n_mu, n_mu), dtype=bool), P(None, 'x', 'y'))

    tau_args = (psi_xn, psi_yr, psi_xr, psi_yn, E_A, mask_A,
                jnp.where(mask_B, B_q, 0.0)[None, ...],
                Omega_q.astype(jnp.complex128)[None, ...],
                jnp.asarray([0], dtype=jnp.int32),
                jnp.asarray([[0.0, np.inf, -np.inf, -np.inf,
                              np.inf, np.inf]], dtype=jnp.float64),
                jnp.asarray([False]),
                jnp.asarray(0.25, dtype=jnp.float64),
                jnp.asarray(0.10, dtype=jnp.float64),
                jnp.asarray(0.3 - 0.7j, dtype=jnp.complex128))
    # Exercise the production opt-in planner rather than a hand-written
    # arbitrary partition.  The kernel contract is still only the resolved
    # bounds; this proves those bounds retain the P=4 leading-axis/sharding
    # behaviour the estimator relies on.
    plan = plan_band_brackets(
        enabled=True,
        enk_ry=np.tile(np.arange(nb, dtype=float) ** 2, (nk, 1)),
        n_occ=4, nb_logical=nb, nb_padded=nb,
        bracket_scheme="conduction_energy_midpoint")
    brk3 = plan.bounds
    if rank == 0:
        print(f"[band-bracket-p4] scheme={plan.bracket_scheme} "
              f"counts={plan.counts} bounds={plan.bounds}", flush=True)

    # ---- 1 + 3: partition through the shared carrier ---------------------
    with mesh:
        one = get_shared_sigma_tau_kernel(
            mesh_xy=mesh, kgrid=kgrid, brackets=((0, nb),), **face_kwargs)(*tau_args)
        three = get_shared_sigma_tau_kernel(
            mesh_xy=mesh, kgrid=kgrid, brackets=brk3, **face_kwargs)(*tau_args)
    spec = tuple(three.sharding.spec)
    if spec != (None, None, 'x', 'y'):
        _fail(f"stacked sigma(tau) sharding is {spec}, expected "
              "(None, None, 'x', 'y')")
    if one.shape[0] != 1 or three.shape[0] != 3:
        _fail(f"bracket extents {one.shape[0]}/{three.shape[0]}, expected 1/3")
    for s1, s3 in zip(one.addressable_shards, three.addressable_shards):
        t1 = np.asarray(s1.data)
        t3 = np.asarray(s3.data)
        tot = np.cumsum(t3, axis=0)[-1]
        scale = max(float(np.max(np.abs(t1[0]))), 1e-300)
        rel = float(np.max(np.abs(tot - t1[0]))) / scale
        if not rel < 1e-12:
            _fail(f"rank{rank} shard{s1.index}: cumulative bracket sum != "
                  f"full-band sum, rel {rel:.3e}")
    if rank == 0:
        print("[band-bracket-p4] shared-carrier partition OK", flush=True)

    # ---- 2: the default path is bit-identical -----------------------------
    kij_args = (psi_xn, psi_yr, psi_xr, psi_yn, E_A, mask_A,
                jnp.asarray(0.25, dtype=jnp.float64),
                jnp.asarray(0.3 - 0.7j, dtype=jnp.complex128), B_q)
    with mesh:
        a = _get_sigma_kij_kernel(mesh_xy=mesh, kgrid=kgrid, merged_x=True,
                                  brackets=None, **face_kwargs)(*kij_args)
        b = _get_sigma_kij_kernel(mesh_xy=mesh, kgrid=kgrid, merged_x=True,
                                  brackets=((0, nb),), **face_kwargs)(*kij_args)
    for sa, sb in zip(a.addressable_shards, b.addressable_shards):
        ta, tb = np.asarray(sa.data), np.asarray(sb.data)
        if not np.array_equal(ta, tb[0]):
            _fail(f"rank{rank} shard{sa.index}: length-1 bracket axis is not "
                  f"bit-identical, max|d| = {np.max(np.abs(ta - tb[0])):.3e}")
    if rank == 0:
        print("[band-bracket-p4] default path bit-identical (max|d| = 0)",
              flush=True)
        print("[band-bracket-p4] PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
