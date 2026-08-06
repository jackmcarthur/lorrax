#!/usr/bin/env python3
"""What does rotating a SHARDED Σ_c(ω,k,m,n) by a replicated W lower to?

THE QUESTION.  ``sigma_mnk.h5`` is written from ``state_final`` in the basis
of ``last_sigma_basis_U`` (= U_{N-1}); ``WFN_qp.h5`` is written from the eigh
of ``state_final.H_qp_dft`` (= H_N).  The two artifacts are one SC iteration
apart.  Putting them in one basis means rotating the cube by
``W = U_{N-1}^† U_N`` before the write:

    Σ'[ω,k,m,n] = Σ_pq W[k,m,p] · Σ[ω,k,p,q] · conj(W[k,n,q])

Under ``sigma_omega_layout = replicated`` that is a local contraction on an
already-replicated cube.  Under ``sharded`` the cube is P(None,None,'x','y')
and the contraction runs over BOTH tiled indices — p on 'x', q on 'y'.  If
GSPMD answers that by gathering the cube, the rotation reintroduces exactly
the per-rank full-cube residency that 712a866 removed, and the SC refusal at
``gw_config.py`` becomes load-bearing for this corner.

WHAT THIS MEASURES.  Nothing is executed at production shape: the arrays are
``jax.ShapeDtypeStruct`` and the probe reads the COMPILED HLO plus
``memory_analysis()``.  So the numbers are the compiler's, at the real
(nω, nk, nb) of a production deck, without needing the memory to run it.
A separate small-shape leg EXECUTES all formulations and checks they agree,
because an HLO reading of a formulation that computes the wrong thing is
worthless.

FORMULATIONS
  rep       cube replicated, einsum — the reference; no collective is
            expected and the operand itself is the full cube.
  naive     cube tiled, one einsum, output pinned to the cube's own tiling.
            This is what a one-line port of ``_rotate_to_dft_basis`` gives.
  naive_free  same, output sharding left to GSPMD.
  blocked   two half-rotations inside shard_map, each scanning over the
            DESTINATION band block so no rank ever holds more than one tile
            of any intermediate: p is summed on 'x' (psum per m-block), then
            q on 'y' (psum per n-block).  The design sketch to beat.

REPORTED PER FORMULATION
  * every collective in the compiled HLO with its operand shape and bytes;
  * the largest collective operand (bytes/rank);
  * ``memory_analysis().temp_size_in_bytes`` — the peak transient;
  * those two against the full cube nω·nk·nb²·16 B, which is the number the
    layout exists to avoid.

RUN IT AT TWO MESH SIZES.  At P=4 four tiles and one cube are the same number
of bytes, so a transient that scales as 1/P cannot be told from one that does
not.  The scaling is the whole question.

MEASURED, job 7889790, nω=41 nk=16 nb=512 (cube 2624 MiB), jax 0.9.1, XLA:CPU:

  formulation   P=4 (2x2)                        P=16 (4x4)
  rep           0 collectives, temp 1.02 cube,   identical — P-independent
                11 full-cube buffers             (the cube IS replicated)
  naive         2 all-gather 1312 MB (cube/2),   2 all-gather 656 MB (cube/4),
                temp 1.02 cube, 0 full-cube      temp 0.51 cube, 0 full-cube
  naive_free    all-reduce 2624 MB = FULL CUBE,  IDENTICAL at P=16
                temp 1.00 cube, 6 full-cube      — P-INDEPENDENT
  blocked       2 all-reduce 656 MB (one tile),  2 all-reduce 164 MB (one tile),
                temp 1.02 cube, 0 full-cube      temp 0.26 cube, 0 full-cube

So: naive's transient is ~2·cube/p_axis and shrinks only as 1/sqrt(P);
naive_free reintroduces the full-cube per-rank residency that 712a866
removed, and does not improve with P at all; blocked's largest collective
operand is exactly one tile and its transient ~4 tiles, i.e. 1/P.

All three agree with a host reference to <= 5e-16 relative at nb=8, both
mesh sizes.

Env: SOR_NW, SOR_NK, SOR_NB (production shape), SOR_SMALL_NB (exec leg).

    srun -n 4 python3 -m sigma_omega_rotate_probe
"""
import os
import re
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402
from common.shard_map import shard_map              # noqa: E402

from common.collectives import process_rank, resolve_mesh     # noqa: E402

NW = int(os.environ.get("SOR_NW", "41"))       # ω points, -10..10 eV @ 0.5
NK = int(os.environ.get("SOR_NK", "16"))       # 4x4x1 full BZ
NB = int(os.environ.get("SOR_NB", "512"))      # the nb of the 2751 MB/rank claim
SMALL_NB = int(os.environ.get("SOR_SMALL_NB", "8"))

COLLECTIVES = ("all-reduce", "all-gather", "collective-permute",
               "reduce-scatter", "all-to-all")


# --------------------------------------------------------------------------
# The four formulations.  All compute the same tensor.
# --------------------------------------------------------------------------

def rot_einsum(sig, W):
    """Σ'[w,k,m,n] = Σ_pq W[k,m,p] Σ[w,k,p,q] conj(W[k,n,q]).

    Same index convention as ``sc_iteration._rotate_to_dft_basis`` with an
    ω axis prepended.
    """
    return jnp.einsum('kmp,wkpq,knq->wkmn', W, sig, jnp.conj(W),
                      optimize=True)


def rot_blocked(mesh):
    """Two half-rotations, each scanning over the DESTINATION band block.

    The transient bound is the point.  A single psum of the p-contraction
    partial would hold (nω, nk, nb, nb/p_y) per rank — the full cube divided
    by ONE mesh axis only.  Scanning over the destination block j and psumming
    one (nω, nk, nb/p_x, nb/p_y) block at a time keeps every intermediate at
    exactly one tile, at the cost of p_x (then p_y) separate collectives
    moving the same total bytes.
    """
    p_x = int(mesh.shape['x'])
    p_y = int(mesh.shape['y'])

    @jax.jit
    @_sm(mesh, (P(None, None, 'x', 'y'), P(None, None, None)),
         P(None, None, 'x', 'y'))
    def _kernel(tile, W):
        ix = jax.lax.axis_index('x')
        iy = jax.lax.axis_index('y')
        mb = tile.shape[2]
        nbl = tile.shape[3]

        # --- half 1: contract p (on 'x').  W rows are the destination m. ---
        W_p = jax.lax.dynamic_slice_in_dim(W, ix * mb, mb, axis=2)   # (k,nb,mb)

        def body_x(j, acc):
            Wj = jax.lax.dynamic_slice_in_dim(W_p, j * mb, mb, axis=1)
            blk = jnp.einsum('kmp,wkpq->wkmq', Wj, tile)             # one tile
            blk = jax.lax.psum(blk, axis_name='x')
            return jnp.where(j == ix, blk, acc)

        A = jax.lax.fori_loop(0, p_x, body_x,
                              jnp.zeros((tile.shape[0], tile.shape[1], mb, nbl),
                                        dtype=tile.dtype))

        # --- half 2: contract q (on 'y').  conj(W) rows are the dest n. ---
        Wc = jnp.conj(W)
        Wc_q = jax.lax.dynamic_slice_in_dim(Wc, iy * nbl, nbl, axis=2)

        def body_y(j, acc):
            Wj = jax.lax.dynamic_slice_in_dim(Wc_q, j * nbl, nbl, axis=1)
            blk = jnp.einsum('knq,wkmq->wkmn', Wj, A)                # one tile
            blk = jax.lax.psum(blk, axis_name='y')
            return jnp.where(j == iy, blk, acc)

        return jax.lax.fori_loop(0, p_y, body_y,
                                 jnp.zeros((tile.shape[0], tile.shape[1],
                                            mb, nbl), dtype=tile.dtype))

    return _kernel


def _sm(mesh, in_specs, out_specs):
    from functools import partial
    return partial(shard_map, mesh=mesh, in_specs=in_specs,
                   out_specs=out_specs, check_vma=False)


# --------------------------------------------------------------------------
# HLO reading
# --------------------------------------------------------------------------

def _operand_bytes(line):
    """Largest cNNN[...] literal on the line, in bytes."""
    best = 0
    for m in re.finditer(r'c(\d+)\[([\d,]*)\]', line):
        width = int(m.group(1)) // 8
        dims = [int(x) for x in m.group(2).split(',') if x]
        n = 1
        for d in dims:
            n *= d
        best = max(best, n * width)
    return best


def analyse(tag, fn, args, out_sharding, p0, cube_bytes, cube_shape):
    """Compile without executing; print the collective census + peak temp.

    The compiled HLO is already SPMD-PARTITIONED, so every shape in it is a
    PER-DEVICE shape.  A buffer whose shape is the full cube therefore means
    one rank holds the whole cube — that is the detector for "GSPMD gathered
    it", and it is reported separately from the collective census because a
    gather can also appear as a fused copy with no collective opcode.
    """
    try:
        jf = (jax.jit(fn, out_shardings=out_sharding)
              if out_sharding is not None else jax.jit(fn))
        comp = jf.lower(*args).compile()
    except Exception as exc:                      # noqa: BLE001
        p0(f"[sor] {tag:11s} COMPILE FAILED: {type(exc).__name__}: {exc}")
        return None
    txt = comp.as_text()
    worst = 0
    rows = []
    for line in txt.splitlines():
        for kind in COLLECTIVES:
            if f" {kind}(" in line or f"= {kind}" in line:
                b = _operand_bytes(line)
                worst = max(worst, b)
                rows.append((kind, b))
                break
    temp = -1
    try:
        ma = comp.memory_analysis()
        if ma is not None:
            temp = int(ma.temp_size_in_bytes)
    except Exception:                              # noqa: BLE001
        temp = -1
    full = "c128[" + ",".join(str(d) for d in cube_shape) + "]"
    n_full = txt.count(full)
    p0(f"[sor] {tag:11s} collectives={len(rows):3d}  "
       f"largest_operand={worst / 2**20:9.1f} MB  "
       f"peak_temp={temp / 2**20 if temp >= 0 else -1.0:9.1f} MB "
       f"(= {temp / cube_bytes if temp >= 0 else -1.0:5.2f} cube)  "
       f"full-cube buffers={n_full}")
    seen = {}
    for kind, b in rows:
        seen[(kind, b)] = seen.get((kind, b), 0) + 1
    for (kind, b), cnt in sorted(seen.items(), key=lambda kv: -kv[0][1])[:8]:
        p0(f"[sor]     {cnt:4d} x {kind:19s} {b / 2**20:9.1f} MB")
    return dict(tag=tag, collectives=len(rows), largest=worst, temp=temp,
                full_cube_buffers=n_full)


def main():
    rank = process_rank()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    p_x, p_y = (int(s) for s in mesh.devices.shape)
    cube_spec = P(None, None, 'x', 'y')
    cube_shd = NamedSharding(mesh, cube_spec)
    rep_shd = NamedSharding(mesh, P(None, None, None))
    rep4_shd = NamedSharding(mesh, P(None, None, None, None))

    cube_bytes = NW * NK * NB * NB * 16
    p0(f"[sor] mesh=({p_x},{p_y})  cube=(nω={NW}, nk={NK}, nb={NB})  "
       f"= {cube_bytes / 2**20:.1f} MB replicated, "
       f"{cube_bytes / (p_x * p_y) / 2**20:.1f} MB/rank tiled")
    if NB % p_x or NB % p_y:
        p0(f"[sor] REFUSING: nb={NB} does not divide the mesh {p_x}x{p_y}")
        return 2

    sig_t = jax.ShapeDtypeStruct((NW, NK, NB, NB), jnp.complex128,
                                 sharding=cube_shd)
    sig_r = jax.ShapeDtypeStruct((NW, NK, NB, NB), jnp.complex128,
                                 sharding=rep4_shd)
    W_r = jax.ShapeDtypeStruct((NK, NB, NB), jnp.complex128, sharding=rep_shd)

    p0("[sor] --- compiled-HLO census at production shape (nothing executed) ---")
    cshape = (NW, NK, NB, NB)
    analyse("rep", rot_einsum, (sig_r, W_r), rep4_shd, p0, cube_bytes, cshape)
    analyse("naive", rot_einsum, (sig_t, W_r), cube_shd, p0, cube_bytes, cshape)
    analyse("naive_free", rot_einsum, (sig_t, W_r), None, p0, cube_bytes, cshape)
    analyse("blocked", rot_blocked(mesh), (sig_t, W_r), cube_shd,
            p0, cube_bytes, cshape)

    # ---- small-shape execution: the formulations must agree ----
    nb = SMALL_NB
    if nb % p_x or nb % p_y:
        p0(f"[sor] small leg skipped: SOR_SMALL_NB={nb} indivisible by mesh")
        return 0
    nw, nk = 3, 2
    rng = np.random.default_rng(20260805)
    S = (rng.standard_normal((nw, nk, nb, nb))
         + 1j * rng.standard_normal((nw, nk, nb, nb)))
    Wh = (rng.standard_normal((nk, nb, nb))
          + 1j * rng.standard_normal((nk, nb, nb)))
    for k in range(nk):                             # make W unitary
        Q, R = np.linalg.qr(Wh[k])
        Wh[k] = Q * (np.diagonal(R) / np.abs(np.diagonal(R)))[None, :]
    ref = np.einsum('kmp,wkpq,knq->wkmn', Wh, S, np.conj(Wh))

    # A global jax.Array spans non-addressable devices at P>1, so it cannot be
    # np.asarray'd directly (job 7889784 died exactly there).  Gather per
    # layout: tiled arrays concatenate, replicated ones are taken from
    # process 0's copy.
    import jax.experimental.multihost_utils as mhu

    def _host(x):
        spec = tuple(getattr(x.sharding, "spec", ()) or ())
        if any(s is not None for s in spec):
            return np.asarray(mhu.process_allgather(x, tiled=True))
        # Fully replicated: every device already holds the whole array, so
        # this rank's own shard IS the global one.  process_allgather refuses
        # tiled=False on a non-fully-addressable global array (job 7889785).
        return np.asarray(x.addressable_shards[0].data)

    S_t = jax.device_put(S, NamedSharding(mesh, cube_spec))
    S_r = jax.device_put(S, NamedSharding(mesh, P(None, None, None, None)))
    W_d = jax.device_put(Wh, rep_shd)
    got_rep = _host(jax.jit(rot_einsum)(S_r, W_d))
    got_naive = _host(jax.jit(rot_einsum, out_shardings=NamedSharding(
        mesh, cube_spec))(S_t, W_d))
    got_blk = _host(rot_blocked(mesh)(S_t, W_d))
    scale = np.max(np.abs(ref))
    for tag, got in (("rep", got_rep), ("naive", got_naive),
                     ("blocked", got_blk)):
        d = float(np.max(np.abs(got - ref))) / scale
        p0(f"[sor] exec nb={nb}: {tag:8s} max|Δ|/scale = {d:.3e} "
           f"{'PASS' if d <= 1e-13 else 'FAIL'}")
        if d > 1e-13:
            return 1
    return 0


if __name__ == "__main__":
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:                          # noqa: BLE001
        traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()
    finalize_process(rc)
