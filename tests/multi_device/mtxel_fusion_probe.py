"""Where ``mtxel.sweep``'s wall goes at the MoS2 4x4 shape, and whether the
``shard_map`` boundaries in ``local_potential_operator`` are the reason.

THE CLAIM UNDER TEST.  ``local_potential_operator`` crosses a ``shard_map``
edge four times per k (ifftn in/out, fftn in/out).  A ``shard_map`` is a hard
fusion boundary, so the box is said to be re-materialised at each crossing —
189 MB at this deck (nb=128, ns=2, 24x24x80, c128) — and that traffic is
proposed as the explanation for the 30x gap between the ~0.2 s of FFT
arithmetic and the 5.72 s/iteration measured in job 7889362.

THE INSTRUMENT IS A DIFFERENCE, NOT A ROW.  Six operators run through the
SAME ``sweep_matrix_elements`` skeleton; each adds one stage to the previous,
so every stage's cost is a subtraction and no stage is attributed by
assertion:

    identity   psi -> psi                      scan + both reshards + einsum
    boxonly    +sphere->box, box->sphere       the two gathers, box resident
    boxmul     +V(r) multiply on the box       one elementwise pass
    onefft     +ifftn                          ONE shard_map pair
    prod       +fftn  == production            TWO shard_map pairs
    fused      the same five stages in ONE shard_map

``prod`` and ``fused`` do identical arithmetic and differ ONLY in how many
``shard_map`` edges the box crosses (4 against 2 — the entry and exit of the
one region).  If the boundary is what costs, ``fused`` is faster than ``prod``
by the box traffic the crossings force; if it is not, they agree.

``fftfloor`` is the measured arithmetic floor: the same two transforms on the
same box, nk times, with nothing else in the program.  The sweep cannot go
below it, and the distance from it is what any fix has to close.

Env: MTX_NB, MTX_NS, MTX_NK, MTX_GRID, MTX_NGK, MTX_NPAD, MTX_REPS,
MTX_ARMS (comma list, default all), MTX_HLO (1 = also print the HLO census).
Defaults are the MoS2 4x4 deck: nb=128, ns=2, nk=10 (the IBZ k-set the
density-SC rebuild sweeps), grid 24x24x80, ngkmax=1964.
"""
import os
import re
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402
from common.shard_map import shard_map              # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from common.mtxel_sweep import (SweepGeometry, Operator,        # noqa: E402
                                sweep_matrix_elements)
from common.wfn_transforms import _box_kernel, _KERNEL_CACHE   # noqa: E402
from common.fft_helpers import (make_sharded_fftn_3d,          # noqa: E402
                                make_sharded_ifftn_3d,
                                local_fftn3, local_ifftn3)
from common.jax_compile_cache import _STATE as CC_STATE        # noqa: E402

NB = int(os.environ.get("MTX_NB", "128"))
NS = int(os.environ.get("MTX_NS", "2"))
NK = int(os.environ.get("MTX_NK", "10"))
GRID = tuple(int(v) for v in os.environ.get("MTX_GRID", "24,24,80").split(","))
NGK = int(os.environ.get("MTX_NGK", "1964"))
NPAD = int(os.environ.get("MTX_NPAD", "4"))
REPS = int(os.environ.get("MTX_REPS", "3"))
ARMS = [a for a in os.environ.get(
    "MTX_ARMS",
    "identity,boxonly,boxmul,onefft,prod,fused").split(",") if a.strip()]
WANT_HLO = os.environ.get("MTX_HLO", "0") == "1"


# ---------------------------------------------------------------------------
# The staged operators.  Each is the previous one plus exactly one stage, so
# the wall difference between neighbours is that stage and nothing else.
# ---------------------------------------------------------------------------

def staged_operator(geom: SweepGeometry, V_r, kind: str) -> Operator:
    from psp.get_DFT_mtxels import local_potential_scalars

    mesh = geom.mesh
    box_spec = geom.spec_box_xy
    ifftn = make_sharded_ifftn_3d(mesh, box_spec, box_spec,
                                  norm='ortho', axes=(-3, -2, -1))
    fftn = make_sharded_fftn_3d(mesh, box_spec, box_spec,
                                norm='ortho', axes=(-3, -2, -1))
    _sc = local_potential_scalars(geom.cell_volume, geom.ngrid)
    scale, deltaV, fft_norm = (float(_sc.scale), float(_sc.deltaV),
                               float(_sc.fft_norm))
    V_r_j = jnp.asarray(V_r, dtype=jnp.complex128)
    ngkmax = geom.ngkmax

    def _out(phi_G, gvec, gmask):
        out = phi_G[..., gvec[:, 0], gvec[:, 1], gvec[:, 2]]
        return out * gmask[None, None, None, :].astype(out.dtype)

    if kind == 'identity':
        def op(psi_n, gvec, gmask, bidx, kvec, V):
            return psi_n * gmask[None, None, None, :].astype(psi_n.dtype)
    elif kind == 'boxonly':
        def op(psi_n, gvec, gmask, bidx, kvec, V):
            return _out(_box_kernel(psi_n, bidx, ngkmax=ngkmax), gvec, gmask)
    elif kind == 'boxmul':
        def op(psi_n, gvec, gmask, bidx, kvec, V):
            box = _box_kernel(psi_n, bidx, ngkmax=ngkmax)
            return _out(box * V, gvec, gmask)
    elif kind == 'onefft':
        def op(psi_n, gvec, gmask, bidx, kvec, V):
            box = _box_kernel(psi_n, bidx, ngkmax=ngkmax)
            psi_r = ifftn(box) * scale
            return _out(psi_r * V, gvec, gmask)
    elif kind == 'prod':
        def op(psi_n, gvec, gmask, bidx, kvec, V):
            box = _box_kernel(psi_n, bidx, ngkmax=ngkmax)
            psi_r = ifftn(box) * scale
            phi_G = fftn(psi_r * V) * (deltaV * fft_norm)
            return _out(phi_G, gvec, gmask)
    elif kind == 'prod_flatg':
        # Production, with the box->sphere gather done on a FLATTENED r axis
        # instead of as advanced indexing on the three trailing axes.  Same
        # values (the flat index is the row-major offset, and the reshape is
        # free in the default layout).  The point is the LAYOUT: a 3-axis
        # trailing gather prefers an r-major/band-minor box, which is not the
        # layout XLA's fft demands, and job 7889385's HLO shows the resulting
        # relayout as ``fft(%copy_bitcast_fusion)``.
        _nyz = (int(geom.fft_grid[1]), int(geom.fft_grid[2]))
        ngrid_ = geom.ngrid

        def op(psi_n, gvec, gmask, bidx, kvec, V):
            box = _box_kernel(psi_n, bidx, ngkmax=ngkmax)
            psi_r = ifftn(box) * scale
            phi_G = fftn(psi_r * V) * (deltaV * fft_norm)
            gflat = (gvec[:, 0] * _nyz[0] + gvec[:, 1]) * _nyz[1] + gvec[:, 2]
            flat = phi_G.reshape(phi_G.shape[:-3] + (ngrid_,))
            out = jnp.take(flat, gflat, axis=-1)
            return out * gmask[None, None, None, :].astype(out.dtype)
    elif kind == 'fused':
        # THE PROPOSED FIX: the five stages inside ONE shard_map, so the box
        # crosses two edges instead of eight.  Communication-free — bands are
        # the sharded axis, the FFT axes and both index tables are replicated.
        sph = geom.spec_sphere_xy
        rep1, rep2, rep3 = P(None), P(None, None), P(None, None, None)

        def _body(psi_l, gvec_l, gmask_l, bidx_l, V_l):
            box = _box_kernel(psi_l, bidx_l, ngkmax=ngkmax)
            psi_r = local_ifftn3(box, axes=(-3, -2, -1), norm='ortho') * scale
            phi = local_fftn3(psi_r * V_l, axes=(-3, -2, -1),
                              norm='ortho') * (deltaV * fft_norm)
            out = phi[..., gvec_l[:, 0], gvec_l[:, 1], gvec_l[:, 2]]
            return out * gmask_l[None, None, None, :].astype(out.dtype)

        smap = shard_map(_body, mesh=mesh,
                         in_specs=(sph, rep2, rep1, P(None, None, None, None),
                                   rep3),
                         out_specs=sph)

        def op(psi_n, gvec, gmask, bidx, kvec, V):
            return smap(psi_n, gvec, gmask, bidx, V)
    else:
        raise ValueError(f"unknown arm {kind!r}")

    return Operator(apply=op, post=float(_sc.post), consts=(V_r_j,),
                    key=('probe', kind, geom.fft_grid, geom.ngkmax, geom.ns))


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


# ---------------------------------------------------------------------------
# HLO census: count the boundary custom calls and the big buffers that the
# fusion argument is about.  Rank 0 only; the module is identical on all.
# ---------------------------------------------------------------------------

_BOUNDARY = ("SPMDFullToShardShape", "SPMDShardToFullShape")


#: ``%name = c128[1,128,2,24,24,80]{5,4,3,2,1,0} fusion(...)`` — the opcode
#: is the token between the shape+layout and the operand list.
_OPCODE = re.compile(r'=\s+\S+\s+([a-z][\w-]*)\(')
#: Shape WITH its layout, which is the whole point: a box that reaches the
#: fft in a non-default layout is a 180 MB relayout copy, not a free bitcast.
_SHAPED = re.compile(r'c128\[([\d,]+)\](\{[\d,]+\})?')


def hlo_census(txt, label, p0, box_elems, show_ops=("fft",)):
    counts, big, lines = {}, {}, []
    for line in txt.splitlines():
        for k in _BOUNDARY:
            if k in line:
                counts[k] = counts.get(k, 0) + 1
        m = _OPCODE.search(line)
        op = m.group(1) if m else None
        if op:
            counts[op] = counts.get(op, 0) + 1
        for sm in _SHAPED.finditer(line):
            n = 1
            for d in sm.group(1).split(','):
                if d:
                    n *= int(d)
            if n >= box_elems // 2:
                tag = sm.group(0)
                big[tag] = (big.get(tag, (0, n))[0] + 1, n)
        if op in show_ops:
            lines.append(line.strip()[:220])
    keep = {k: v for k, v in counts.items()
            if k in ('fft', 'fusion', 'copy', 'gather', 'transpose',
                     'custom-call', 'bitcast', 'while', 'dot')
            or k in _BOUNDARY}
    p0(f"[probe] HLO {label}: " + "  ".join(
        f"{k}={v}" for k, v in sorted(keep.items())))
    for shp, (n, elems) in sorted(big.items(), key=lambda kv: -kv[1][0])[:6]:
        p0(f"[probe] HLO {label}: box-class operand {shp} named {n}x "
           f"({elems * 16 / 2**20:.0f} MB)")
    for ln in lines[:6]:
        p0(f"[probe] HLO {label}: {ln}")


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nx, ny, nz = GRID
    ngkmax = NGK + NPAD
    volume = 702.20                      # MoS2 4x4 cell volume, a.u.^3

    ngrid = nx * ny * nz
    box_mb = NB * NS * ngrid * 16 / 2**20
    # 5 N log2 N per transform, 2 transforms per (band, spinor, k).
    gflop = (5.0 * ngrid * np.log2(ngrid) * NB * NS * 2 * NK) / 1e9
    p0(f"[probe] world={world} mesh=({px},{py}) nk={NK} nb={NB} ns={NS} "
       f"grid={GRID} ngk={NGK} ngkmax={ngkmax} reps={REPS}")
    p0(f"[probe] one k's box {box_mb:.0f} MB global "
       f"({box_mb/world:.0f} MB/rank) | transforms {NB*NS*2*NK} of {ngrid} "
       f"points = {gflop:.1f} GFLOP for the whole sweep")

    rng = np.random.default_rng(20260805)
    gv = np.zeros((NK, ngkmax, 3), dtype=np.int32)
    bidx = np.full((NK, nx, ny, nz), ngkmax, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(ngrid, size=NGK, replace=False)
        gv[ik, :NGK, 0] = cells // (ny * nz)
        gv[ik, :NGK, 1] = (cells // nz) % ny
        gv[ik, :NGK, 2] = cells % nz
        bidx[ik, gv[ik, :NGK, 0], gv[ik, :NGK, 1], gv[ik, :NGK, 2]] = \
            np.arange(NGK, dtype=np.int32)
    gmask = np.zeros((NK, ngkmax), dtype=np.float64)
    gmask[:, :NGK] = 1.0

    psi = (rng.standard_normal((NK, NB, NS, ngkmax))
           + 1j * rng.standard_normal((NK, NB, NS, ngkmax))
           ).astype(np.complex128)
    psi[..., NGK:] = 0.0
    V_r = np.asarray(rng.standard_normal(GRID))
    kvecs = rng.standard_normal((NK, 3)) * 0.25

    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=ngkmax,
                         nb=NB, ns=NS, nk=NK, cell_volume=volume)
    psi_pad = psi
    if geom.nb != NB:
        psi_pad = np.pad(psi, ((0, 0), (0, geom.nb - NB), (0, 0), (0, 0)))
        p0(f"[probe] band pad {NB} -> {geom.nb} (p_prod={geom.p_prod})")
    sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
    psi_j = jax.make_array_from_callback(psi_pad.shape, sharding,
                                         lambda idx: psi_pad[idx])
    kw = dict(geom=geom, gvecs=gv, gmask=gmask, box_index=bidx, kvecs=kvecs)

    # ---- the measured arithmetic floor -----------------------------------
    # The same two transforms on the same box, nk times, with nothing else in
    # the program.  This is what "0.2 s of arithmetic" has to be checked
    # against on THIS machine, not a peak-FLOPs estimate.
    box_spec = geom.spec_box_xy
    _if = make_sharded_ifftn_3d(mesh, box_spec, box_spec, norm='ortho',
                                axes=(-3, -2, -1))
    _f = make_sharded_fftn_3d(mesh, box_spec, box_spec, norm='ortho',
                              axes=(-3, -2, -1))
    # Index the host array with the callback's own index tuple: a replicated
    # axis hands back ``slice(None, None)``, so deriving the shard extent
    # from ``stop - start`` raises TypeError (first run of this probe, job
    # 7889382).
    box_np = np.zeros((1, geom.nb, NS, nx, ny, nz), dtype=np.complex128)
    box0 = jax.make_array_from_callback(
        box_np.shape, NamedSharding(mesh, box_spec), lambda idx: box_np[idx])
    fft_pair = jax.jit(lambda b: _f(_if(b)))
    fft_pair(box0).block_until_ready()
    tf = []
    for _ in range(REPS):
        t0 = time.time()
        for _ik in range(NK):
            fft_pair(box0).block_until_ready()
        tf.append(time.time() - t0)
    t_floor = float(np.median(tf))
    p0(f"[probe] FFT FLOOR (nk x ifftn+fftn on the box, nothing else): "
       f"{t_floor:.3f} s   -> {gflop/max(t_floor,1e-9):.1f} GFLOP/s")

    # ---- where the sweep's excess over that floor lives -------------------
    # The floor's operand is a jit PARAMETER, so XLA gets to pick its layout
    # and hand it to the fft unchanged.  The sweep's box is produced by a
    # gather and consumed by a gather.  Two standalone jits, no scan and no
    # einsum, separate the two effects:
    #   floor_g   box built in-jit, box returned      (producer side only)
    #   floor_gs  box built in-jit, sphere returned   (producer + consumer)
    # ``floor_gs`` IS the operator, standalone.  Whatever it costs over
    # ``floor`` is a layout/relayout effect, not the scan and not the mesh.
    if os.environ.get("MTX_LOCALISE", "0") == "1":
        from psp.get_DFT_mtxels import local_potential_scalars
        _s = local_potential_scalars(volume, ngrid)
        sc_, dV_, fn_ = float(_s.scale), float(_s.deltaV), float(_s.fft_norm)
        Vj = jnp.asarray(V_r, dtype=jnp.complex128)

        def _chain(psi_k, bidx_k, V):
            box = _box_kernel(psi_k, bidx_k, ngkmax=ngkmax)
            return _f(_if(box) * sc_ * V) * (dV_ * fn_)

        j_g = jax.jit(lambda p_, b_, V: _chain(p_, b_, V))
        j_gs = jax.jit(lambda p_, b_, g_, m_, V: (
            lambda phi: phi[..., g_[:, 0], g_[:, 1], g_[:, 2]]
            * m_[None, None, None, :].astype(phi.dtype))(_chain(p_, b_, V)))
        bidx_j = jnp.asarray(bidx, dtype=jnp.int32)
        gv_j = jnp.asarray(gv, dtype=jnp.int32)
        gm_j = jnp.asarray(gmask, dtype=jnp.float64)
        for nm, fnj, args in (
                ("floor_g", j_g, lambda ik: (psi_j[ik:ik+1], bidx_j[ik:ik+1],
                                             Vj)),
                ("floor_gs", j_gs, lambda ik: (psi_j[ik:ik+1],
                                               bidx_j[ik:ik+1], gv_j[ik],
                                               gm_j[ik], Vj))):
            fnj(*args(0)).block_until_ready()
            tl = []
            for _ in range(REPS):
                t0 = time.time()
                for ik in range(NK):
                    fnj(*args(ik)).block_until_ready()
                tl.append(time.time() - t0)
            p0(f"[probe] {nm:<9} median={float(np.median(tl)):.3f}s  "
               f"({float(np.median(tl))/max(t_floor,1e-9):.2f}x the floor)")
        if WANT_HLO:
            hlo_census(jax.jit(lambda b: _f(_if(b))).lower(box0)
                       .compile().as_text(), "floor/optimized", p0,
                       geom.nb * NS * ngrid)
            hlo_census(j_gs.lower(psi_j[0:1], bidx_j[0:1], gv_j[0], gm_j[0],
                                 Vj).compile().as_text(),
                       "floor_gs/optimized", p0, geom.nb * NS * ngrid)

    # ---- the staged arms -------------------------------------------------
    results = {}
    ref = None
    for arm in ARMS:
        op = staged_operator(geom, V_r, arm)
        c0 = CC_STATE.compiles
        out = sweep_matrix_elements(psi_j, operator=op, **kw)
        out.block_until_ready()
        comp = CC_STATE.compiles - c0
        t = []
        for _ in range(REPS):
            t0 = time.time()
            o = sweep_matrix_elements(psi_j, operator=op, **kw)
            o.block_until_ready()
            t.append(time.time() - t0)
        med = float(np.median(t))
        results[arm] = med
        # prod vs fused must agree numerically as well as in wall.
        loc = np.concatenate([np.asarray(s.data).ravel()
                              for s in out.addressable_shards])
        if arm == 'prod':
            ref = loc
        delta = ''
        if arm != 'prod' and arm in ('fused', 'prod_flatg') and ref is not None:
            sc = max(float(np.abs(ref).max()), 1e-300)
            delta = (f"  vs prod max rel delta "
                     f"{float(np.abs(loc - ref).max())/sc:.3e}")
        p0(f"[probe] arm={arm:<9} median={med:.3f}s  best={min(t):.3f}s  "
           f"compiles={comp}{delta}", flush=True)
        if WANT_HLO and arm in ('prod', 'fused'):
            key = [k for k in _KERNEL_CACHE
                   if k[0] == 'sweep_matrix_elements'
                   and ('probe', arm) == k[6][:2]]
            if key:
                fn = _KERNEL_CACHE[key[0]]
                # Lower with the SAME operands the call used — psi_j is the
                # sharded one, and the sharding is what the census is about.
                low = fn.lower(psi_j, jnp.asarray(gv, dtype=jnp.int32),
                               jnp.asarray(gmask, dtype=jnp.float64),
                               jnp.asarray(bidx, dtype=jnp.int32),
                               jnp.asarray(kvecs, dtype=jnp.float64),
                               *[jnp.asarray(c) for c in op.consts])
                hlo_census(low.as_text(), f"{arm}/stablehlo", p0,
                           geom.nb * NS * ngrid)
                hlo_census(low.compile().as_text(), f"{arm}/optimized", p0,
                           geom.nb * NS * ngrid)

    # ---- the decomposition ------------------------------------------------
    p0(f"[probe] ---- stage attribution (nk={NK}, P={world}) ----")
    order = [a for a in ('identity', 'boxonly', 'boxmul', 'onefft', 'prod')
             if a in results]
    prev = None
    for a in order:
        if prev is None:
            p0(f"[probe]   {a:<9} {results[a]:7.3f} s   (skeleton: scan, "
               f"both reshards, einsum)")
        else:
            p0(f"[probe]   {a:<9} {results[a]:7.3f} s   "
               f"+{results[a]-results[prev]:6.3f} s over {prev}")
        prev = a
    if 'prod' in results and 'fused' in results:
        d = results['prod'] - results['fused']
        p0(f"[probe]   fused     {results['fused']:7.3f} s   "
           f"{d:+.3f} s against prod "
           f"({100.0*d/max(results['prod'],1e-9):+.1f} %) — this number IS "
           f"the shard_map-boundary hypothesis")
    if 'prod' in results:
        p0(f"[probe]   FFT floor {t_floor:7.3f} s   prod/floor = "
           f"{results['prod']/max(t_floor,1e-9):.1f}x")
    p0(f"[probe] rank={rank:03d} VmHWM={vmhwm_gib():.3f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    # finalize_process() ends with os._exit and DOES NOT RETURN, so a bare
    # `finally: finalize_process()` swallows the exception and exits 0.
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)
