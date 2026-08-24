#!/usr/bin/env python3
"""Track O7 — where the ladder-W direct rung's time actually goes, and what the
flat-k cuFFT FFI engine buys it.

Two modes, deliberately independent so the cheap one can answer the first
question without a deck:

``rung``    STAGE BREAKDOWN + the three engine/layout arms, on synthetic
            ``T`` / ``W_R`` tensors at the rung's real shapes.  No payload, no
            restart, no driver run — the rung is a pure function of shapes, so
            a fixture would only add I/O to the measurement.  Answers: is the
            FFT the cost, or the elementwise multiply, or the transposes?  Each
            arm is ONE jit, timed and HLO-censused (fft / custom-call /
            transpose / copy / fusion counts + temp bytes), and every arm's
            output is checked against the XLA arm's.

``matvec``  END-TO-END: the real ladder matvec on a real payload, baseline vs
            an FFI-served twin obtained by rebinding the two factory names
            ``bse_stack_matvec`` imports (``w_ladder_fftffi.make_sharded_*_ffi``)
            BEFORE the matvec is built.  No production file is edited — the
            integration agent owns those concurrently.  The RPA matvec
            (``include_w=False``) is timed beside both, which is what makes the
            rung's share of the matvec a measurement rather than an inference.

Cache-cold is FORCED, both variables (INVARIANTS row 4 + the 2026-08-16
finding that ``ISDF_JAX_CACHE_DIR=""`` alone is not cold under ``lx``, which
exports ``JAX_COMPILATION_CACHE_DIR``).

Run (Perlmutter, one GPU is enough — this is an exploration leg, not a
verification one; the four-GPU rule governs verification):

    lx run -G 1 -n 1 bash -lc 'source <prelude>; cd $LXA && \
        python3 tests/bench/bench_w_ladder_fftffi.py --mode rung --out <ev>'

Not collected by pytest (``tests/bench`` is in ``norecursedirs``); argv-driven
per ``docs/architecture/layers.md`` §5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cache-cold BEFORE `import jax`, both variables (see the module docstring).
os.environ["ISDF_JAX_CACHE_DIR"] = ""
os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)

import numpy as np                                               # noqa: E402

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
jax.config.update("jax_compilation_cache_dir", None)

from bse import w_ladder_fftffi as X                             # noqa: E402
from ffi.mklfft import fft_ffi_enabled                           # noqa: E402


# ---------------------------------------------------------------------------
# timing + HLO census
# ---------------------------------------------------------------------------
def timeit(fn, args, reps: int, rounds: int = 3):
    """Seconds per call — the MINIMUM over ``rounds`` blocks of ``reps`` calls.

    Minimum, not mean: on a co-tenanted card every perturbation (a neighbour's
    kernel, a clock step, an allocator stall) is additive, so the mean of a
    short block estimates the interference and the minimum estimates the
    kernel.  The first block is discarded as warm-up on top of the compile
    call, because the first dispatch after a compile also pays the first
    allocation of every buffer in the program.
    """
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


_CENSUS_OPS = ("fft", "custom-call", "transpose", "copy", "fusion", "bitcast",
               "dot", "all-gather", "collective-permute", "reduce-scatter")


def census(fn, args, tag: str) -> dict:
    """Optimized-HLO opcode census + XLA temp bytes for one jit.

    Counts by opcode on the ROOT computation and every called computation, the
    same way ``docs/HLO_HOWTO.md`` does.  ``copy``/``fusion`` are both here on
    purpose: on GPU the layout materialization shows up as a fusion with a big
    temp, not as a ``transpose`` opcode (``flat_k_fft_service.md`` §8), so a
    census that counts only transposes is the defective instrument that
    document names.
    """
    try:
        comp = jax.jit(fn).lower(*args).compile()
        text = comp.as_text()
    except Exception as exc:                       # pragma: no cover
        return {"tag": tag, "error": f"{type(exc).__name__}: {exc}"}
    counts = {op: 0 for op in _CENSUS_OPS}
    for ln in text.splitlines():
        s = ln.strip()
        if "=" not in s or "(" not in s:
            continue
        # `%x = f64[2,3]{1,0} transpose(...)` — the opcode is the whitespace-
        # separated token immediately before the first '('.
        head = s.split("=", 1)[1].split("(", 1)[0].strip().split(" ")[-1]
        if head in counts:
            counts[head] += 1
    mem = {}
    try:
        ma = comp.memory_analysis()
        mem = {"temp_MB": ma.temp_size_in_bytes / 1e6,
               "argument_MB": ma.argument_size_in_bytes / 1e6,
               "output_MB": ma.output_size_in_bytes / 1e6}
    except Exception:
        pass
    return {"tag": tag, "ops": counts, **mem}


def relerr(a, b) -> float:
    a = np.asarray(jax.device_get(a))
    b = np.asarray(jax.device_get(b))
    den = np.max(np.abs(a))
    return float(np.max(np.abs(a - b)) / (den if den > 0 else 1.0))


# ---------------------------------------------------------------------------
# mode: rung
# ---------------------------------------------------------------------------
def make_operands(mesh, nb, nmu, nnu, ns, kgrid, seed=3):
    """The same random rung state in four layouts, plus BOTH W forms.

    ``W_q`` is the primitive here and ``W_R = ifftn(W_q, norm='ortho')`` is
    derived from it exactly as ``bse_feast.ensure_W_R`` derives it
    (``bse_densify.make_w_densifier`` :190) — that is what makes the fused
    ``gw_conv`` arm (which consumes ``W_q``) comparable to the production arms
    (which consume ``W_R``) rather than merely similar.
    """
    nk = int(np.prod(kgrid))
    rng = np.random.default_rng(seed)
    sh = X.rung_shardings(mesh)

    def _c(*shape):
        return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                / 8.0).astype(np.complex128)

    Tn = _c(nb, nmu, nnu, ns, ns, nk)
    Wq = _c(nmu, nnu, *kgrid)
    W_Rn = np.fft.ifftn(Wq, axes=(2, 3, 4), norm="ortho")
    with mesh:
        T = jax.device_put(Tn, sh.T6)                                  # b,μ,ν,t,s,k
        W = jax.device_put(W_Rn, sh.W5)                                # μ,ν,kx,ky,kz
        T_flat = jax.device_put(np.transpose(Tn, (5, 0, 1, 2, 3, 4)),
                                sh.T_flat)                             # k,b,μ,ν,t,s
        W_flat = jax.device_put(
            np.transpose(W_Rn.reshape(nmu, nnu, nk), (2, 0, 1)), sh.W_flat)
        T_g = jax.device_put(
            np.transpose(Tn, (5, 0, 1, 3, 4, 2)).reshape(
                nk, nb, nmu, ns * ns, nnu), sh.T_g)                     # k,b,μ,ts,ν
        Wq_f = jax.device_put(
            np.transpose(Wq.reshape(nmu, nnu, nk), (2, 0, 1)), sh.W_flat)
    return T, W, T_flat, W_flat, T_g, Wq_f


def mode_rung(args, mesh):
    rows, cens, checks = [], [], []
    ns = args.nspinor
    for kgrid in args.kgrids:
        nk = int(np.prod(kgrid))
        for nb in args.batches:
            nmu = nnu = args.nmu
            T, W, T_flat, W_flat, T_g, Wq_f = make_operands(
                mesh, nb, nmu, nnu, ns, kgrid)
            tag = f"k{'x'.join(str(v) for v in kgrid)}_mu{nmu}_b{nb}"
            tbytes = T.size * 16 / 1e6
            print(f"\n=== {tag}: T {tuple(T.shape)} = {tbytes:.1f} MB, "
                  f"W_R {tuple(W.shape)}", flush=True)

            conv_xla = X.make_conv_xla(mesh, kgrid)
            conv_kmin = X.make_conv_ffi_kminor(mesh, kgrid)
            conv_klead = X.make_conv_ffi_kleading(mesh, kgrid)

            # --- the two halves of the rung, and the whole thing, per arm ---
            jits = {}
            jits["xla_conv"] = (jax.jit(conv_xla), (T, W))
            jits["xla_rung"] = (jax.jit(lambda t, w: X.to_O_from_kminor(conv_xla(t, w))),
                                (T, W))
            jits["ffi_kmin_conv"] = (jax.jit(conv_kmin), (T, W))
            jits["ffi_kmin_rung"] = (
                jax.jit(lambda t, w: X.to_O_from_kminor(conv_kmin(t, w))), (T, W))
            jits["ffi_klead_conv"] = (jax.jit(conv_klead), (T_flat, W_flat))
            jits["ffi_klead_rung"] = (
                jax.jit(lambda t, w: X.to_O_from_kleading(conv_klead(t, w))),
                (T_flat, W_flat))
            try:
                conv_fused = X.make_conv_ffi_fused(mesh, kgrid)
                jits["ffi_fused_conv"] = (jax.jit(conv_fused), (T_g, Wq_f))
                jits["ffi_fused_rung"] = (
                    jax.jit(lambda t, w: X.to_O_from_fused(conv_fused(t, w), ns)),
                    (T_g, Wq_f))
            except Exception as exc:
                print(f"  fused gw_conv UNAVAILABLE: {type(exc).__name__}: {exc}",
                      flush=True)

            # --- stage isolation (each its own jit: an upper bound per stage,
            #     because XLA cannot fuse across a dispatch boundary) ---
            spec8 = P(None, "x", "y", None, None, None, None, None)
            from common.fft_helpers import (make_sharded_fftn_3d,
                                            make_sharded_ifftn_3d)
            _if8 = make_sharded_ifftn_3d(mesh, spec8, spec8, axes=(5, 6, 7),
                                         norm="ortho")
            _f8 = make_sharded_fftn_3d(mesh, spec8, spec8, axes=(5, 6, 7),
                                       norm="ortho")
            r8 = (nb, nmu, nnu, ns, ns, *kgrid)
            jits["stage_xla_ifft"] = (jax.jit(lambda t: _if8(t.reshape(r8))), (T,))
            jits["stage_xla_fft"] = (jax.jit(lambda t: _f8(t.reshape(r8))), (T,))
            jits["stage_mult"] = (
                jax.jit(lambda t, w: (w[None, :, :, None, None, :, :, :]
                                      * t.reshape(r8))), (T, W))
            from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
            specf = P(None, None, None, None, "x", "y", None, None)
            _iff = make_flat_k_ifftn(mesh, kgrid, specf, norm="ortho")
            _ff = make_flat_k_fftn(mesh, kgrid, specf, norm="ortho")
            jits["stage_ffi_ifft"] = (jax.jit(_iff), (T_flat,))
            jits["stage_ffi_fft"] = (jax.jit(_ff), (T_flat,))
            jits["stage_flatten"] = (jax.jit(X.flatten_T), (T,))
            jits["stage_toO_kminor"] = (jax.jit(X.to_O_from_kminor), (T,))
            jits["stage_toO_klead"] = (jax.jit(X.to_O_from_kleading), (T_flat,))
            jits["stage_wflatten"] = (jax.jit(X.flatten_W_R), (W,))

            out = {}
            for name, (fn, a) in jits.items():
                try:
                    dt = timeit(fn, a, args.reps)
                    out[name] = fn(*a)
                except Exception as exc:
                    print(f"  {name:22s}  FAILED  {type(exc).__name__}: {exc}",
                          flush=True)
                    rows.append(dict(tag=tag, kgrid=list(kgrid), nb=nb, nmu=nmu,
                                     arm=name, error=f"{type(exc).__name__}: {exc}"))
                    continue
                print(f"  {name:22s}  {dt*1e3:9.3f} ms", flush=True)
                rows.append(dict(tag=tag, kgrid=list(kgrid), nb=nb, nmu=nmu,
                                 arm=name, ms=dt * 1e3, T_MB=tbytes))
                if args.census and name.endswith("rung"):
                    c = census(fn, a, f"{tag}/{name}")
                    cens.append(c)
                    print(f"      census {c.get('ops')} temp="
                          f"{c.get('temp_MB', float('nan')):.1f} MB", flush=True)

            # --- numerics: every arm against the XLA arm, in O layout ---
            ref = out.get("xla_rung")
            for name in ("ffi_kmin_rung", "ffi_klead_rung", "ffi_fused_rung"):
                if ref is not None and name in out:
                    e = relerr(ref, out[name])
                    checks.append(dict(tag=tag, arm=name, rel=e))
                    print(f"  numerics {name:16s} rel={e:.3e}", flush=True)
            del out, T, W, T_flat, W_flat, T_g, Wq_f
    return dict(mode="rung", rows=rows, census=cens, numerics=checks)


# ---------------------------------------------------------------------------
# mode: matvec  (real payload, real operator)
# ---------------------------------------------------------------------------
def mode_matvec(args, mesh):
    import harness                                               # noqa: F401
    from runtime import bootstrap
    bootstrap()
    from bse import bse_io, bse_stack_matvec
    from bse.bse_feast import ladder_matvec_operands, matvec_operands
    from bse.bse_w_exact import _symmetry_tables, build_finite_q_data
    from bse.w_ladder import build_ladder_resolvent
    from bse import w_ladder_shifts as wls

    run_dir = Path(args.run_dir)
    input_path = str(run_dir / args.deck)
    restart = bse_io._find_restart_file(input_path)
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    print(f"[payload] n_rmu={int(data['n_rmu'])} "
          f"mu_pad={int(data['V_q0'].shape[0])} "
          f"nc={int(data['eps_c'].shape[1])} nv={int(data['eps_v'].shape[1])} "
          f"nk={int(data['nkx']*data['nky']*data['nkz'])} "
          f"nspinor={int(data['psi_c_X'].shape[2])}", flush=True)
    sym = _symmetry_tables(input_path)
    q = tuple(int(v) for v in np.asarray(sym.q_irr_kgrid_int, dtype=int)[0])

    from common.fft_helpers import (make_sharded_fftn_3d,
                                    make_sharded_ifftn_3d)
    rows, outs, cens = [], {}, []
    for arm in args.arms:
        if arm == "ffi":
            bse_stack_matvec.make_sharded_ifftn_3d = X.make_sharded_ifftn_3d_ffi
            bse_stack_matvec.make_sharded_fftn_3d = X.make_sharded_fftn_3d_ffi
        else:
            bse_stack_matvec.make_sharded_ifftn_3d = make_sharded_ifftn_3d
            bse_stack_matvec.make_sharded_fftn_3d = make_sharded_fftn_3d
        for include_w in ([True, False] if arm == "xla" else [True]):
            stack = build_ladder_resolvent(mesh, data, include_w=include_w)
            matvec, _, gen, snapshot, sh = stack
            dq = build_finite_q_data(data, q, mesh)
            # The LADDER matvec carries ladder_rung_slots -> 14 operands (the
            # ordinary 10 + the rung's four PHYSICAL psi); the RPA delegate
            # takes the ordinary 10.
            ops = tuple(ladder_matvec_operands(dq) if include_w
                        else matvec_operands(dq))
            n_pad = int(dq["V_q0"].shape[0])
            label = f"{arm}{'' if include_w else '_rpa'}"
            for width in args.batches:
                G = np.eye(n_pad)[:width, :]
                rhs = wls.seed_probe_block(G, dq, gen, sh)

                def _one(x, _mv=matvec, _ops=ops):
                    return _mv(x, *_ops)

                @jax.jit
                def _rep(x, _mv=matvec, _ops=ops):
                    return jax.lax.fori_loop(
                        0, args.reps, lambda _, y: _mv(y, *_ops), x)

                jax.block_until_ready(_rep(rhs))
                t0 = time.perf_counter()
                jax.block_until_ready(_rep(rhs))
                dt = (time.perf_counter() - t0) / args.reps
                print(f"  {label:10s} batch={width:3d}  {dt*1e3:9.3f} ms/matvec",
                      flush=True)
                rows.append(dict(arm=label, batch=width, ms=dt * 1e3,
                                 include_w=include_w))
                if width == args.batches[0]:
                    outs[label] = np.asarray(jax.device_get(_one(rhs)))
                    if args.census:
                        c = census(_one, (rhs,), f"matvec/{label}/b{width}")
                        cens.append(c)
                        print(f"      census {c.get('ops')} temp="
                              f"{c.get('temp_MB', float('nan')):.1f} MB",
                              flush=True)
    checks = []
    if "xla" in outs and "ffi" in outs:
        e = relerr(outs["xla"], outs["ffi"])
        checks.append(dict(pair="xla_vs_ffi", rel=e))
        print(f"\n  numerics matvec xla vs ffi: rel={e:.3e}", flush=True)
    return dict(mode="matvec", q=list(q), rows=rows, reps=args.reps,
                census=cens, numerics=checks, run_dir=str(run_dir))


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="rung", choices=["rung", "matvec"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--nmu", type=int, default=399)
    ap.add_argument("--nspinor", type=int, default=2)
    ap.add_argument("--kgrids", default="3x3x1")
    ap.add_argument("--batches", default="1")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--census", type=int, default=1)
    ap.add_argument("--arms", default="xla,ffi")
    ap.add_argument("--run-dir",
                    default="/pscratch/sd/j/jackm/opt_fftffi_20260816/gnppm_run")
    ap.add_argument("--deck", default="gnppm_test.in")
    args = ap.parse_args(argv)
    args.kgrids = [tuple(int(v) for v in g.split("x"))
                   for g in args.kgrids.split(",") if g]
    args.batches = [int(v) for v in args.batches.split(",") if v]
    args.arms = [a for a in args.arms.split(",") if a]

    devs = jax.devices()
    mesh = Mesh(np.array(devs[:1]).reshape(1, 1), axis_names=("x", "y"))
    print(f"[env] jax {jax.__version__}  devices={devs}  mesh=1x1  "
          f"fft_ffi_enabled={fft_ffi_enabled()}  "
          f"LORRAX_FFI_SO={os.environ.get('LORRAX_FFI_SO', '<unset>')}",
          flush=True)
    print(f"[cold] ISDF_JAX_CACHE_DIR={os.environ.get('ISDF_JAX_CACHE_DIR')!r} "
          f"JAX_COMPILATION_CACHE_DIR="
          f"{os.environ.get('JAX_COMPILATION_CACHE_DIR', '<popped>')!r}",
          flush=True)

    with mesh:
        result = (mode_rung(args, mesh) if args.mode == "rung"
                  else mode_matvec(args, mesh))
    result["argv"] = sys.argv[1:]
    result["jax"] = jax.__version__
    result["devices"] = [str(d) for d in devs]
    # Allocator travels with every wall time (AGENT_PREAMBLE, "the machine").
    result["allocator"] = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    result["mem_fraction"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION",
                                            "<default>")
    result["reps"] = args.reps
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        p = out / f"bench_fftffi_{args.mode}_{stamp}.json"
        p.write_text(json.dumps(result, indent=1))
        print(f"\n[out] {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
