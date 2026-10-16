"""The BSE W term over (μ, ν) blocks equals the whole-tile W term (``bse.bse_stack_matvec``).

The W term is an exact sum over (μ, ν) blocks of the rank's ``W_R`` tile
(``bse_stack_matvec._blocked_w_term``); blocking only reorders the μ and ν sums.  Each case
builds a matvec on a 2x2 mesh with ε = 0 and V = 0, so it returns ``-W X`` alone, forces a
route (the fused decode, the outer load with a stored U, the XLA encode) and a block plan
(``runtime.tiles.TILE_BYTES`` lowered for the stored-U routes, the 2^31 handler cap lowered for
the fused one), and requires ``max|a - b| <= 1e-12 * max|b|`` against

* the whole-tile matvec of the same route (the unblocked program), and
* an independent NumPy evaluation of the definition (``docs/architecture/bse.md#hamiltonian``),
  ``(WX)[c,v,k] = (1/√Nk) Σ conj(ψ_c) ψ_v fftn(ifftn(T) · W_R)`` with
  ``T[k,t,μ,s,ν] = Σ_cv ψ_c[k,c,t,μ] conj(ψ_v[k,v,s,ν]) X[c,v,k]``, sharing no code with either.

Usages: the TDA matvec (``build_bse_stack_matvec``), the TD-aGW model's application of the same
W term to a density matrix (both legs the whole window, one trial), and the non-TDA pair
applier fused and unfused (``build_bse_stack_pair_matvec``); spinor counts 1 and 2; block widths
with remainders (29 = 8 + 8 + 8 + 5 rows per rank, the last block clamped onto its neighbour).
Red twins: a plan with one μ block dropped, and one whose W_R block start is off by one row; the
same check must report both.  The over-budget case lowers the device budget below the W term's
price: the stage warns and still runs (TASTE 96).  The memory case compiles (abstractly; nothing
is allocated) the route-3 matvec on a 30x30 grid in 8 x 8 blocks of a 480 x 480 tile per rank and
requires its temps below half of the rank's ``W_R``; its red twin slices the blocks from the 5-D
tile, which GPU layout assignment turns into one k-leading relayout of the whole tile.

CPU, one process with four host devices: ``JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 tests/test_bse_w_blocks.py``.
CUDA, four ranks: ``lx run -N 1 -G 4 -n 4 -- python3 tests/test_bse_w_blocks.py``.  The
planner tests (``test_plan_*``) run under pytest anywhere; the mesh cases skip there.
"""
import contextlib
import warnings

TOL = 1e-12
KGRID = (3, 2, 1)                 # nk = 6: a 2-D grid, odd along x
N_MU = 58                         # 29 rows per rank on a 2x2 mesh: blocks of 8 leave remainders


def _rel(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


@contextlib.contextmanager
def _patched(module, **values):
    """Set module attributes for the block; restore them after."""
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def _w_reference(X, pc, pv, W_R, kgrid):
    """NumPy W term of the definition: (b, c, v, nk) for X (b, c, v, nk), ψ (nk, n, ns, μ), W_R (μ, ν, kx, ky, kz)."""
    import numpy as np
    nk = int(np.prod(kgrid))
    T = np.einsum("kctM,kvsN,bcvk->bktMsN", pc, np.conj(pv), X)
    shape = T.shape
    TR = np.fft.ifftn(T.reshape((shape[0],) + tuple(kgrid) + shape[2:]), axes=(1, 2, 3), norm="ortho")
    W = np.moveaxis(W_R, (2, 3, 4), (0, 1, 2))                      # (kx, ky, kz, μ, ν)
    U = np.fft.fftn(TR * W[None, :, :, :, None, :, None, :], axes=(1, 2, 3), norm="ortho")
    return np.einsum("kctM,kvsN,bktMsN->bcvk", np.conj(pc), pv, U.reshape(shape)) / np.sqrt(nk)


def _w_reference_B(Xb, pcX, pvX, pcY, pvY, W_R, kgrid):
    """NumPy coupling-block W term: T_B[k,t,μ,s,ν] = Σ ψ^X_v[k,v,t,μ] conj(ψ^Y_c[k,c,s,ν]) Xb[c,v,k]."""
    import numpy as np
    nk = int(np.prod(kgrid))
    T = np.einsum("kvtM,kcsN,bcvk->bktMsN", pvX, np.conj(pcY), Xb)
    shape = T.shape
    TR = np.fft.ifftn(T.reshape((shape[0],) + tuple(kgrid) + shape[2:]), axes=(1, 2, 3), norm="ortho")
    W = np.moveaxis(W_R, (2, 3, 4), (0, 1, 2))
    U = np.fft.fftn(TR * W[None, :, :, :, None, :, None, :], axes=(1, 2, 3), norm="ortho")
    return np.einsum("kctM,kvsN,bktMsN->bcvk", np.conj(pcX), pvY, U.reshape(shape)) / np.sqrt(nk)


def _operands(rng, n_c, n_v, ns, n_trials, td_model=False):
    """Host operands: X, ψ_c, ψ_v, W_R.  ``td_model``: one ψ set on both legs (the TD window)."""
    import numpy as np
    nk = int(np.prod(KGRID))
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    pc = cplx(nk, n_c, ns, N_MU)
    pv = pc.copy() if td_model else cplx(nk, n_v, ns, N_MU)
    return dict(X=cplx(n_trials, n_c, n_v, nk), pc=pc, pv=pv, pcY=pc.copy(), pvX=pv.copy(),
                W_R=cplx(N_MU, N_MU, *KGRID))


def _apply(mesh, ops, *, pair=None):
    """``-W X`` (pair: ``-(W_A X + s W_B conj X)``) from the matvec on ``mesh``, gathered to host."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from bse import bse_stack_matvec as B
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    n_c, n_v, nk = ops["X"].shape[1], ops["X"].shape[2], ops["X"].shape[3]
    args = (put(ops["pc"], P(None, None, None, "x")), put(ops["pcY"], P(None, None, None, "y")),
            put(ops["pvX"], P(None, None, None, "x")), put(ops["pv"], P(None, None, None, "y")),
            put(np.zeros((nk, n_c)), P(None, None)), put(np.zeros((nk, n_v)), P(None, None)),
            put(ops["W_R"], P("x", "y", None, None, None)), put(np.zeros((N_MU, N_MU), complex), P("x", "y")),
            put(np.zeros((nk, n_c, n_v, N_MU), complex), P(None, "x", "y", None)))
    X = put(ops["X"], P(None, "x", "y", None))
    if pair is None:
        out = B.build_bse_stack_matvec(mesh, *KGRID, kernel="bse")(X, *args)
    else:
        fuse, s = pair
        out = B.build_bse_stack_pair_matvec(mesh, *KGRID, kernel="bse", fuse=fuse)(
            X, put(np.asarray(s, complex), P()), *args)
    return np.asarray(multihost_utils.process_allgather(out, tiled=True))


def _route_patches(route):
    """Module attributes that force ``route`` on any backend (the refusals the router reads)."""
    forced = lambda *a, **k: "forced by tests/test_bse_w_blocks.py"
    return {"fused": {}, "outer": {"klead_outer_decode_refusal": forced},
            "xla": {"klead_outer_refusal": forced}}[route]


def _plan_patches(route, plan, ns):
    """(tiles attributes, bse_stack_matvec attributes) that make the planner pick ``plan``."""
    import numpy as np
    nk = int(np.prod(KGRID))
    if plan == "whole":
        return {}, {}
    if route == "fused":                       # the fused route blocks only at the handler cap
        return {}, {"_HANDLER_ELEMENTS": {"cap_a": 1500, "cap_b": 800}[plan]}
    cells = {"cap_a": 64, "cap_b": 200, "cap_c": 464}[plan]   # 8x8; 24x8; 29 (whole) x 16
    return {"TILE_BYTES": 16 * nk * (ns * ns + 1) * cells}, {}


def _receipts(B):
    """Patches recording each trace's route and block plan, the cases' activation receipts."""
    seen = []
    route, plan = B._w_route, B._w_plan

    def _w_route(*a, **k):
        got = route(*a, **k)
        seen.append(("route", got[0]))
        return got

    def _w_plan(*a, **k):
        got = plan(*a, **k)
        seen.append(("plan", got))
        return got
    return seen, {"_w_route": _w_route, "_w_plan": _w_plan}


def run_cases(mesh, *, xla_reference=False):
    """Every (usage, ns, route, plan) case; returns ``{name: rel}`` and the red-twin results.

    A case name carries its receipt, the route and block counts its trace recorded; a case whose
    receipt is not the forced route, or a blocked plan that traced the whole tile, reports
    ``inf`` (fails).
    """
    import numpy as np
    from bse import bse_stack_matvec as B
    from ffi import fft as F
    from runtime import tiles
    ctx = F.xla_reference if xla_reference else contextlib.nullcontext
    rng = np.random.default_rng(20261016)
    out = {}
    usages = {"tda": dict(n_c=4, n_v=6, n_trials=3), "td_model": dict(n_c=6, n_v=6, n_trials=1, td_model=True),
              "pair_fused": dict(n_c=4, n_v=6, n_trials=2), "pair_unfused": dict(n_c=4, n_v=6, n_trials=2)}
    for usage, shape in usages.items():
        pair = {"pair_fused": (True, -1.0), "pair_unfused": (False, -1.0)}.get(usage)
        for ns in (1, 2):
            ops = _operands(rng, shape["n_c"], shape["n_v"], ns, shape["n_trials"], shape.get("td_model", False))
            ref = -_w_reference(ops["X"], ops["pc"], ops["pv"], ops["W_R"], KGRID)
            if pair is not None:
                ref = ref - pair[1] * _w_reference_B(np.conj(ops["X"]), ops["pc"], ops["pvX"], ops["pcY"],
                                                     ops["pv"], ops["W_R"], KGRID)
            for route in ("fused", "outer", "xla"):
                whole = None
                for plan in ("whole", "cap_a", "cap_b") + (("cap_c",) if route != "fused" else ()):
                    tile_attrs, mod_attrs = _plan_patches(route, plan, ns)
                    seen, spies = _receipts(B)
                    with ctx(), _patched(B, **_route_patches(route), **mod_attrs, **spies), \
                            _patched(tiles, **tile_attrs):
                        got = _apply(mesh, ops, pair=pair)
                    routes = {v for k, v in seen if k == "route"}
                    counts = {p[0][1] * p[1][1] for k, p in seen if k == "plan"}
                    blocks = "x".join(f"{p[0][1]}x{p[1][1]}" for k, p in seen if k == "plan")
                    activated = routes == {route} and (counts == {1} if plan == "whole" else
                                                       bool(counts) and min(counts) > 1)
                    key = f"{usage}/ns{ns}/{route}/{plan} [{blocks} blocks]"
                    if plan == "whole":
                        whole = got
                    else:
                        out[key + " vs whole"] = _rel(got, whole) if activated else float("inf")
                    out[key + " vs numpy"] = _rel(got, ref) if activated else float("inf")
    # Red twins: the same check must catch a dropped block and a misaligned block start.
    ops = _operands(rng, 4, 6, 2, 3)
    ref = -_w_reference(ops["X"], ops["pc"], ops["pv"], ops["W_R"], KGRID)
    take = B._take_block

    def drop_second_mu_block(arrays, i, b, n):
        blocks, start = take(arrays, i, b, n)
        if b == 16:                                    # plan cap_b: μ 2 x 16, ν 4 x 8
            blocks = tuple(a * (i != 1) for a in blocks)
        return blocks, start

    def shifted_start(arrays, i, b, n):
        blocks, start = take(arrays, i, b, n)
        return blocks, start + (i == 1) * (start + 1 <= n - b)

    red = {}
    for name, fake in (("drop a mu block", drop_second_mu_block), ("W_R block start + 1", shifted_start)):
        tile_attrs, _ = _plan_patches("xla", "cap_b", 2)
        with ctx(), _patched(B, _take_block=fake, **_route_patches("xla")), _patched(tiles, **tile_attrs):
            red[name] = _rel(_apply(mesh, ops), ref)
    # Over budget: the price exceeds a budget lowered below it; the stage warns and runs.
    from common import gpu_utils
    keep = gpu_utils._RUN_DEVICE_BUDGET_GB
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        gpu_utils.set_device_budget_gb(1e-9)
        try:
            tile_attrs, _ = _plan_patches("xla", "cap_a", 2)
            with ctx(), _patched(B, **_route_patches("xla")), _patched(tiles, **tile_attrs):
                out["over budget: still exact vs numpy"] = _rel(_apply(mesh, ops), ref)
        finally:
            gpu_utils._RUN_DEVICE_BUDGET_GB = keep
    warned = any("memory over budget at bse.w_term" in str(w.message) for w in seen)
    return out, red, warned


def run_densify_cases(mesh):
    """The densifier's two in-place W updates against NumPy: the SR/LR rank term in q blocks
    (``bse_densify._add_rank_lr``; whole grid, and blocks of 4 of 6 q, the last clamped onto its
    neighbour) and the Γ head channel (``gw.head_densify.attach_head_channel``)."""
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from bse.bse_densify import _add_rank_lr
    from gw.head_densify import attach_head_channel
    from runtime import tiles
    rng = np.random.default_rng(7)
    nq, n_l = int(np.prod(KGRID)), 3
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    W, A, D = cplx(N_MU, N_MU, *KGRID), cplx(nq, n_l, N_MU), cplx(nq, n_l, n_l)
    put_w = lambda: device_put_process_local(W.copy(), NamedSharding(mesh, P("x", "y", None, None, None)))
    host = lambda a: np.asarray(multihost_utils.process_allgather(a, tiled=True))
    term = np.einsum("qam,qab,qbn->mnq", np.conj(A), D, A).reshape(W.shape)
    out = {}
    for name, tile in (("whole", None), ("q blocks 4 of 6", 2 * 16 * (N_MU // 2) ** 2 * 4)):
        with _patched(tiles, **({} if tile is None else {"TILE_BYTES": tile})):
            out[f"densify rank-LR add, {name} vs numpy"] = _rel(host(_add_rank_lr(put_w(), A, D, mesh, -1.0)),
                                                               W - term)
    g0 = cplx(N_MU)
    S = rng.standard_normal(KGRID)
    g0_X = device_put_process_local(g0, NamedSharding(mesh, P("x")))
    g0_Y = device_put_process_local(g0, NamedSharding(mesh, P("y")))
    ref = W + (np.conj(g0)[:, None] * g0[None, :])[:, :, None, None, None] * (S / 2.5)[None, None]
    out["densify head channel vs numpy"] = _rel(host(attach_head_channel(put_w(), g0_X, g0_Y, S, 2.5)), ref)
    return out


def run_memory_case(mesh):
    """Per-rank compiled temps of the route-3 matvec, ``(blocks from _w_block, red twin, W_R bytes)``.

    CrI3-like 30x30 grid, ns 2, a 480 x 480 tile per rank in 8 x 8 blocks of 64 rows, abstract
    operands.  The red twin slices each block from the 5-D tile, the program the AOT gate refused at
    98.5 GB per rank on CrI3 30x30, N_μ 3000, P4: there GPU layout assignment gave the
    loop-invariant tile the k-major layout of the k-leading kernel operand, one relayout of the
    WHOLE tile ahead of the scans.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from bse import bse_stack_matvec as B
    from runtime import tiles
    kg, nk, n_c, n_v, ns, n_mu, nb = (30, 30, 1), 900, 4, 6, 2, 960, 2
    sds = lambda shape, spec, dt=jnp.complex128: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))
    args = (sds((nb, n_c, n_v, nk), P(None, "x", "y", None)),
            sds((nk, n_c, ns, n_mu), P(None, None, None, "x")), sds((nk, n_c, ns, n_mu), P(None, None, None, "y")),
            sds((nk, n_v, ns, n_mu), P(None, None, None, "x")), sds((nk, n_v, ns, n_mu), P(None, None, None, "y")),
            sds((nk, n_c), P(None, None), jnp.float64), sds((nk, n_v), P(None, None), jnp.float64),
            sds((n_mu, n_mu) + kg, P("x", "y", None, None, None)), sds((n_mu, n_mu), P("x", "y")),
            sds((nk, n_c, n_v, n_mu), P(None, "x", "y", None)))
    tile = 16 * nk * (ns * ns + 1) * 64 * 64

    def temp(block_fn):
        with _patched(B, **_route_patches("xla"), _w_block=block_fn), _patched(tiles, TILE_BYTES=tile):
            matvec = B.build_bse_stack_matvec(mesh, *kg, kernel="bse")
            return matvec.lower(*args).compile().memory_analysis().temp_size_in_bytes

    def five_d(W_R, m0, n0, bm, bn):
        return lax.dynamic_slice(W_R, (m0, n0) + (0,) * (W_R.ndim - 2), (bm, bn) + tuple(W_R.shape[2:]))
    mu_loc = n_mu // 2
    return temp(B._w_block), temp(five_d), 16 * nk * mu_loc * mu_loc


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    mesh = stack.mesh
    if tuple(mesh.devices.shape) != (2, 2):
        print(f"SKIP: needs a 2x2 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out, red, warned = run_cases(mesh)
    out.update(run_densify_cases(mesh))
    temp, twin, w_bytes = run_memory_case(mesh)
    on_gpu = jax.devices()[0].platform == "gpu"
    mem_ok = temp < 0.5 * w_bytes
    twin_ok = twin >= w_bytes if on_gpu else True    # the relayout is GPU layout assignment's
    bad = {k: v for k, v in out.items() if not v <= TOL}
    caught = {k: v for k, v in red.items() if v > 1e-6}
    warned = warned or jax.process_index() != 0      # the warning is rank 0's
    ok = not bad and len(caught) == len(red) and warned and mem_ok and twin_ok
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:44s} rel {v:.3e}", flush=True)
        for k, v in red.items():
            print(f"{'ok  ' if v > 1e-6 else 'FAIL'} red twin {k:35s} rel {v:.3e} (must exceed 1e-6)", flush=True)
        print(f"{'ok  ' if warned else 'FAIL'} over-budget price warned (memory over budget at bse.w_term)", flush=True)
        print(f"{'ok  ' if mem_ok else 'FAIL'} W term temps {temp / 1e9:.3f} GB per rank "
              f"(must stay below half of W_R, {w_bytes / 1e9:.2f} GB)", flush=True)
        print(f"{'ok  ' if twin_ok else 'FAIL'} red twin 5-D block slice: temps {twin / 1e9:.3f} GB per rank "
              + ("(must reach W_R: the whole-tile relayout)" if on_gpu else "(cpu: no relayout to catch)"),
              flush=True)
        print(f"BSE_W_BLOCKS {'PASS' if ok else 'FAIL'}: {len(out) - len(bad)}/{len(out)} cases within "
              f"{TOL:g}, {len(caught)}/{len(red)} red twins caught, over-budget warning "
              f"{'seen' if warned else 'MISSING'}, W term temps {temp / 1e9:.2f} GB "
              f"{'<' if mem_ok else '>='} W_R/2 (twin {twin / 1e9:.2f} GB)", flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_w_blocks_need_a_2x2_mesh():
    """Collected by pytest: the cases need their own 2x2 mesh, so they run as a script."""
    import pytest
    pytest.skip("run as a script on a 2x2 mesh: python3 tests/test_bse_w_blocks.py")


def test_w_block_matches_the_5d_slice():
    """``_w_block`` (the slice of the tile's 2-D view) is W_R's 5-D block at interior, edge and clamped offsets."""
    import jax.numpy as jnp
    import numpy as np
    from bse import bse_stack_matvec as B
    rng = np.random.default_rng(31)
    W = jnp.asarray(rng.standard_normal((29, 23, 3, 2, 1)) + 1j * rng.standard_normal((29, 23, 3, 2, 1)))
    ref = np.asarray(W)
    for m0, n0, bm, bn in ((0, 0, 8, 8), (8, 16, 8, 7), (21, 15, 8, 8), (0, 0, 29, 23)):
        assert np.array_equal(np.asarray(B._w_block(W, m0, n0, bm, bn)), ref[m0:m0 + bm, n0:n0 + bn])


def test_plan_whole_tile_when_it_fits():
    from bse.bse_stack_matvec import plan_w_blocks
    assert plan_w_blocks("xla", nk=64, ns=2, mu_loc=100, nu_loc=100, n_c=14, rank=14) == ((100, 1), (100, 1))
    assert plan_w_blocks("fused", nk=64, ns=2, mu_loc=1504, nu_loc=1504, n_c=14, rank=14) == ((1504, 1), (1504, 1))


def test_plan_stored_u_fits_the_tile():
    from bse.bse_stack_matvec import plan_w_blocks
    from runtime.tiles import TILE_BYTES
    for nk, mu, nc, route in ((900, 1504, 14, "xla"), (900, 1500, 14, "xla"), (64, 3000, 30, "outer"),
                              (144, 1504, 14, "outer"), (900, 1504, 60, "xla")):
        (bm, nm), (bn, nn) = plan_w_blocks(route, nk=nk, ns=2, mu_loc=mu, nu_loc=mu, n_c=nc, rank=nc)
        assert 16 * nk * 5 * bm * bn <= TILE_BYTES, (nk, mu, bm, bn)
        assert bm % 8 == 0 and bn % 8 == 0 and (nm - 1) * bm < mu <= nm * bm and (nn - 1) * bn < mu <= nn * bn
        assert nk * 4 * bm * bn < 2 ** 31
    # CrI3 30x30, N_mu 3000, P4: 13 x 13 blocks of 120 x 120 (U block 0.83 GB)
    assert plan_w_blocks("xla", nk=900, ns=2, mu_loc=1504, nu_loc=1504, n_c=14, rank=14) == ((120, 13), (120, 13))


def test_plan_handler_cap_shrinks_and_refuses():
    import pytest
    from bse.bse_stack_matvec import plan_w_blocks
    # the unfused CrI3 8x8 42v x 30c at N_mu 6000, P4: whole U = 2.30e9 >= 2^31 -> blocks
    (bm, nm), (bn, nn) = plan_w_blocks("outer", nk=64, ns=2, mu_loc=3000, nu_loc=3000, n_c=30, rank=30)
    assert nm * nn > 1 and 64 * 4 * bm * bn < 2 ** 31
    # the fused route stays whole until an operand reaches 2^31 (V = mu_loc nu_loc nk here)
    (bm, nm), (bn, nn) = plan_w_blocks("fused", nk=128, ns=2, mu_loc=6000, nu_loc=6000, n_c=8, rank=8)
    assert nm * nn > 1 and bm * bn * 128 < 2 ** 31
    # red twin of the refusal: the largest grid whose 8 x 8 block fits plans; one doubling more refuses
    assert plan_w_blocks("outer", nk=2 ** 22, ns=2, mu_loc=64, nu_loc=64, n_c=4, rank=4)[0][0] == 8
    with pytest.raises(ValueError, match="GATE bse_w_block_operand"):
        plan_w_blocks("outer", nk=2 ** 23, ns=2, mu_loc=64, nu_loc=64, n_c=4, rank=4)


if __name__ == "__main__":
    raise SystemExit(main())
