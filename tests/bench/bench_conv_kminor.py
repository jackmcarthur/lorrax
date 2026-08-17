#!/usr/bin/env python3
"""GATES + BENCH for the fused-conv family's k-MINOR member.

Three modes, deliberately independent so the cheap one runs without a deck:

``gate``    THE NON-NEGOTIABLE GATES, at the rung's real shapes.  Each prints
            PASS/FAIL and the process exit code is the AND of all of them:
              G1  numerics vs the XLA chain, rel <= 1e-15, out_layout=0
              G2  numerics vs the XLA chain + its transpose, out_layout=1
              G3  the two out_layout arms carry IDENTICAL numerics (the store
                  is a permutation; anything but bit-equality means the
                  compute path forked, which is the defect this cell exists
                  for — a tolerance here would test nothing)
              G4  c128-ONLY: a c64 operand must RAISE, not demote
              G5  the HLO probe: every arm compiles, census rc 0, and the
                  fused arm's op counts show the chain actually collapsed
``ksweep``  nk = 9 … 216, the sweep that killed the strided-plan arm.  The
            handler must WIN OR TIE at EVERY point (O7 measured the k-leading
            FFI at 1.61x the XLA chain at nk=64 and 4.00x at nk=216; a k-minor
            engine that repeats that has bought nothing).  Reports the ratio
            per row and fails the leg if any row regresses beyond tolerance.
``rung``    A/B timing at the rung's real shapes: the CHAIN (what O9 priced at
            3637 us = 74.3% of the ladder matvec) and the FULL RUNG (chain +
            the production contract_bands decode, which O7 measured at
            95.6-98.4% of the ladder matvec), at nb = 1 and 4.

WHY THE RUNG AND NOT THE MATVEC.  The end-to-end matvec A/B needs the opt-in
hook inside ``bse_ring_comm._apply_W_from_T``, and that file belongs to the
integration lane (the hook is three lines; the patch ships beside this file's
evidence).  Everything measurable WITHOUT touching it is measured here, and
the two published shares — 74.3% of GPU busy for the chain, 95.6-98.4% of the
matvec for the rung — are what carry a rung number to a matvec number.  Do not
quote a matvec speedup from this script; quote the rung and the share.

BENCH_ALLOC=bfc.  The campaign prelude's ``XLA_PYTHON_CLIENT_ALLOCATOR=
platform`` poisons microbenches (measured 9.7x on one stage, O7 table F), so
this script SETS the allocator itself, before importing jax, and prints which
one it used.  Cache-cold is forced, both variables.

Run (Perlmutter, one GPU):

    lx run -G 1 -n 1 --wait 900 env LORRAX_FFI_SO=<isolated .so> \\
        LORRAX_CONV_KMINOR_FFI=1 BENCH_ALLOC=bfc \\
        python3 tests/bench/bench_conv_kminor.py --mode gate --out <ev>

Not collected by pytest (``tests/bench`` is in ``norecursedirs``); argv-driven
per ``docs/architecture/layers.md`` §5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# --- allocator + cache-cold, BEFORE `import jax` -----------------------------
_ALLOC = os.environ.get("BENCH_ALLOC", "bfc").strip().lower()
if _ALLOC == "bfc":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    os.environ.pop("TF_GPU_ALLOCATOR", None)
elif _ALLOC == "platform":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["ISDF_JAX_CACHE_DIR"] = ""
os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)

import numpy as np                                                # noqa: E402

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
jax.config.update("jax_compilation_cache_dir", None)

from bse import w_ladder_conv_kminor as K                          # noqa: E402
from common.fft_helpers import (make_sharded_fftn_3d,              # noqa: E402
                                make_sharded_ifftn_3d)
from ffi.fft import conv_kminor_available                          # noqa: E402


# ---------------------------------------------------------------------------
# instruments (same conventions as tests/bench/bench_w_ladder_fftffi.py)
# ---------------------------------------------------------------------------
def timeit(fn, args, reps: int, rounds: int = 3) -> float:
    """Milliseconds per call — MINIMUM over ``rounds`` blocks of ``reps``.

    Minimum, not mean: on a co-tenanted card every perturbation is additive,
    so the mean of a short block estimates the interference and the minimum
    estimates the kernel."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e3


_CENSUS_OPS = ("fft", "custom-call", "transpose", "copy", "fusion", "bitcast",
               "dot", "all-gather", "collective-permute", "reduce-scatter")


def census(fn, args, tag: str) -> dict:
    """Optimized-HLO opcode census + XLA temp bytes for one jit.

    ``copy``/``fusion`` are counted on purpose: on GPU a layout
    materialization shows up as a kLoop fusion with a big temp, not as a
    ``transpose`` opcode, so a census keyed on transposes alone is the
    defective instrument ``flat_k_fft_service.md`` §8 names."""
    try:
        comp = jax.jit(fn).lower(*args).compile()
        text = comp.as_text()
    except Exception as exc:
        return {"tag": tag, "error": f"{type(exc).__name__}: {exc}"}
    counts = {op: 0 for op in _CENSUS_OPS}
    for ln in text.splitlines():
        s = ln.strip()
        if "=" not in s or "(" not in s:
            continue
        head = s.split("=", 1)[1].split("(", 1)[0].strip().split(" ")[-1]
        if head in counts:
            counts[head] += 1
    out = {"tag": tag, "ops": counts}
    try:
        ma = comp.memory_analysis()
        out["temp_MB"] = ma.temp_size_in_bytes / 1e6
    except Exception:
        pass
    return out


def _host(x):
    """A numpy view of a possibly GLOBALLY-SHARDED array.

    Under one process ``device_get`` suffices.  Under several, an array whose
    shards live on other processes cannot be fetched locally at all, so the
    comparison has to gather it — which is a TEST-ONLY collective and exactly
    the kind of thing the handler itself must never do."""
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        return np.asarray(multihost_utils.process_allgather(x, tiled=True))
    return np.asarray(jax.device_get(x))


def relerr(ref, got) -> float:
    """max|ref-got| / max|ref| — the SAME metric O7/O9's harnesses used
    (tools/rung_chain_proto.py:196), so the numbers are comparable to their
    4.267e-16 without a footnote."""
    a = _host(ref)
    b = _host(got)
    den = float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b)) / (den if den > 0 else 1.0))


# ---------------------------------------------------------------------------
# the reference: the production chain, lifted verbatim
# ---------------------------------------------------------------------------
def build_xla_chain(mesh, kgrid, *, to_O: bool):
    """``bse_ring_comm._apply_W_from_T``'s chain, minus the decode.

    Verbatim (`bse_ring_comm.py:540-543, 575-584`): 6-D -> 8-D reshape,
    sharded ifftn(ortho), W_R broadcast multiply, sharded fftn(ortho), 6-D
    reshape, and — when ``to_O`` — the transpose into the decode's
    ``(b, k, t, mu, s, nu)`` layout, which is the op ``out_layout=1`` deletes.
    """
    nkx, nky, nkz = kgrid
    spec8 = P(None, "x", "y", None, None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, spec8, spec8, axes=(5, 6, 7),
                                  norm="ortho")
    fftn = make_sharded_fftn_3d(mesh, spec8, spec8, axes=(5, 6, 7),
                                norm="ortho")

    def _chain(T, W_R):
        nb, mu, nu, nt, ns, nk = T.shape
        T_k = T.reshape(nb, mu, nu, nt, ns, nkx, nky, nkz)
        T_R = ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U = fftn(U_R).reshape(nb, mu, nu, nt, ns, nk)
        return jnp.transpose(U, (0, 5, 3, 1, 4, 2)) if to_O else U

    return _chain


def operands(mesh, nb, nmu, ns, kgrid, seed=3):
    """``T`` (b,mu,nu,t,s,nk), ``W_R`` (mu,nu,kx,ky,kz) and its flat twin.

    ``W_q`` is the primitive and ``W_R = ifftn(W_q, norm='ortho')`` is derived
    exactly as ``bse_feast.ensure_W_R`` derives it, so the operands are the
    rung's, not merely rung-shaped."""
    nk = int(np.prod(kgrid))
    rng = np.random.default_rng(seed)
    shT = NamedSharding(mesh, K.RUNG_T_SPEC)
    shW5 = NamedSharding(mesh, P("x", "y", None, None, None))
    shW3 = NamedSharding(mesh, K.RUNG_W_SPEC)

    def _c(*shape):
        return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                / 8.0).astype(np.complex128)

    Tn = _c(nb, nmu, nmu, ns, ns, nk)
    Wq = _c(nmu, nmu, *kgrid)
    W_Rn = np.fft.ifftn(Wq, axes=(2, 3, 4), norm="ortho")
    with mesh:
        T = jax.device_put(Tn, shT)
        W5 = jax.device_put(W_Rn, shW5)
        W3 = jax.device_put(W_Rn.reshape(nmu, nmu, nk), shW3)
    return T, W5, W3


# ---------------------------------------------------------------------------
# mode: gate
# ---------------------------------------------------------------------------
def mode_gate(args, mesh) -> dict:
    ns, nmu = args.nspinor, args.nmu
    kgrid = tuple(args.kgrids[0])
    nb = args.batches[0]
    res = {"mode": "gate", "kgrid": list(kgrid), "nmu": nmu, "nb": nb,
           "nspinor": ns, "cells": []}

    def cell(name, ok, detail):
        res["cells"].append({"cell": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    T, W5, W3 = operands(mesh, nb, nmu, ns, kgrid)
    print(f"\n=== gate: T {tuple(T.shape)} = {T.size * 16 / 1e6:.1f} MB, "
          f"W_R {tuple(W5.shape)}", flush=True)

    f_xla = jax.jit(build_xla_chain(mesh, kgrid, to_O=False))
    f_xla_O = jax.jit(build_xla_chain(mesh, kgrid, to_O=True))
    ref = f_xla(T, W5)
    ref_O = f_xla_O(T, W5)

    conv0 = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=0))
    conv1 = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=1))
    got0 = conv0(T, W3)
    got1 = conv1(T, W3)

    # G1 / G2 — numerics against the XLA chain
    r0 = relerr(ref, got0)
    cell("G1 out_layout=0 vs XLA chain", r0 <= args.tol,
         f"rel = {r0:.3e} (budget {args.tol:.0e})")
    r1 = relerr(ref_O, got1)
    cell("G2 out_layout=1 vs XLA chain + transpose", r1 <= args.tol,
         f"rel = {r1:.3e} (budget {args.tol:.0e})")

    # G3 — the two arms must be the SAME numbers, not merely close.  The store
    # is a permutation of one computation; a nonzero delta means the arms took
    # different arithmetic paths, and a tolerance would hide exactly that.
    perm = np.transpose(_host(got0), (0, 5, 3, 1, 4, 2))
    d = float(np.max(np.abs(perm - _host(got1))))
    cell("G3 out_layout arms bit-identical", d == 0.0,
         f"max|permute(layout0) - layout1| = {d:.3e} (must be exactly 0)")

    # G4 — c128 only, by REFUSAL
    try:
        conv0(T.astype(jnp.complex64), W3.astype(jnp.complex64))
        cell("G4 c64 refused (no demotion)", False,
             "a complex64 call SUCCEEDED — the handler demoted instead of "
             "refusing")
    except Exception as exc:
        ok = "complex128" in str(exc)
        cell("G4 c64 refused (no demotion)", ok,
             f"{type(exc).__name__}: {str(exc)[:120]}")

    # G5 — HLO probe
    cens = [census(build_xla_chain(mesh, kgrid, to_O=True), (T, W5), "xla_rung"),
            census(K.make_rung_conv(mesh, kgrid, out_layout=1), (T, W3),
                   "ffi_rung_layout1"),
            census(K.make_rung_conv(mesh, kgrid, out_layout=0), (T, W3),
                   "ffi_conv_layout0")]
    res["census"] = cens
    errs = [c for c in cens if "error" in c]
    fused = next((c for c in cens if c["tag"] == "ffi_rung_layout1"), None)
    ok = (not errs and fused is not None
          and fused["ops"]["custom-call"] >= 1
          and fused["ops"]["fft"] == 0)
    cell("G5 HLO probe rc 0 + chain collapsed", ok,
         (f"errors={[c['error'] for c in errs]}" if errs else
          f"fused arm: {fused['ops']['custom-call']} custom-call, "
          f"{fused['ops']['fft']} fft, {fused['ops']['transpose']} transpose, "
          f"temp {fused.get('temp_MB', float('nan')):.1f} MB  |  xla arm: "
          f"{cens[0]['ops']['fft']} fft, {cens[0]['ops']['transpose']} "
          f"transpose, temp {cens[0].get('temp_MB', float('nan')):.1f} MB"))

    res["pass"] = all(c["pass"] for c in res["cells"])
    return res


# ---------------------------------------------------------------------------
# mode: ksweep
# ---------------------------------------------------------------------------
def mode_ksweep(args, mesh) -> dict:
    ns, nmu, nb = args.nspinor, args.nmu, args.batches[0]
    res = {"mode": "ksweep", "nmu": nmu, "nb": nb, "rows": []}
    print(f"\n=== ksweep: mu=nu={nmu} nspinor={ns} nb={nb}\n"
          f"   (L1 = out_layout=1, the rung chain incl. the decode's O "
          f"layout; L0 = out_layout=0, same arithmetic with a coalesced "
          f"store — the difference IS the permutation's cost)\n"
          f"{'kgrid':>8} {'nk':>4} {'T(MB)':>8} {'xla[ms]':>9} "
          f"{'ffi[ms]':>9} {'ffi/xla':>8}", flush=True)
    worst = 0.0
    for kgrid in args.kgrids:
        kgrid = tuple(kgrid)
        nk = int(np.prod(kgrid))
        T, W5, W3 = operands(mesh, nb, nmu, ns, kgrid)
        f_xla = jax.jit(build_xla_chain(mesh, kgrid, to_O=True))
        f_xla0 = jax.jit(build_xla_chain(mesh, kgrid, to_O=False))
        f_ffi = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=1))
        f_ffi0 = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=0))
        rel = relerr(f_xla(T, W5), f_ffi(T, W3))
        t_x = timeit(f_xla, (T, W5), args.reps)
        t_f = timeit(f_ffi, (T, W3), args.reps)
        # The layout-0 pair isolates THE STORE: same arithmetic, same reads,
        # only the write map differs.  Without it a slow row cannot be
        # attributed between the transform and the permutation.
        t_x0 = timeit(f_xla0, (T, W5), args.reps)
        t_f0 = timeit(f_ffi0, (T, W3), args.reps)
        ratio = t_f / t_x
        worst = max(worst, ratio)
        # Bytes an ideal fused kernel must move: read T, read W, write U.
        gb = (2.0 * T.size + W3.size) * 16 / 1e9
        row = {"kgrid": list(kgrid), "nk": nk, "T_MB": T.size * 16 / 1e6,
               "xla_ms": t_x, "ffi_ms": t_f, "ratio": ratio, "rel": rel,
               "xla_conv_only_ms": t_x0, "ffi_layout0_ms": t_f0,
               "ffi_GBps": gb / (t_f * 1e-3), "ffi0_GBps": gb / (t_f0 * 1e-3)}
        res["rows"].append(row)
        print(f"{'x'.join(str(v) for v in kgrid):>8} {nk:>4} "
              f"{row['T_MB']:>8.1f} {t_x:>9.3f} {t_f:>9.3f} {ratio:>8.2f}"
              f"  | L0 {t_f0:>7.3f} ({row['ffi0_GBps']:>6.0f} GB/s)"
              f"  L1 {row['ffi_GBps']:>6.0f} GB/s  rel {rel:.1e}", flush=True)
        del T, W5, W3
    # WIN OR TIE at every point.  `--ksweep-tol` is the tie band, not a
    # licence: O7's k-leading arm lost 1.61x at nk=64 and 4.00x at nk=216, so
    # anything above ~1.05 here means the k-minor engine reproduced the defect
    # it exists to remove.
    res["worst_ratio"] = worst
    res["pass"] = worst <= args.ksweep_tol
    print(f"\n  worst ffi/xla over the sweep = {worst:.2f} "
          f"(must be <= {args.ksweep_tol:.2f}) -> "
          f"{'PASS' if res['pass'] else 'FAIL'}", flush=True)
    return res


# ---------------------------------------------------------------------------
# mode: rung
# ---------------------------------------------------------------------------
def mode_rung(args, mesh) -> dict:
    from common.contract_bands import contract_bands_block_reshard
    ns, nmu = args.nspinor, args.nmu
    kgrid = tuple(args.kgrids[0])
    nk = int(np.prod(kgrid))
    res = {"mode": "rung", "kgrid": list(kgrid), "nmu": nmu, "rows": []}
    rng = np.random.default_rng(11)

    w_decode = contract_bands_block_reshard(mesh, extra="leading")
    fused_body = K.build_rung_body(mesh, kgrid, w_decode)

    def _c(*shape):
        return ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                / 8.0).astype(np.complex128)

    # psi_c (k, c, s, mu_X) and psi_v (k, v, s, nu_Y) — the rung's operands.
    psi_c = jax.device_put(_c(nk, args.ncond, ns, nmu),
                           NamedSharding(mesh, P(None, None, None, "x")))
    psi_v = jax.device_put(_c(nk, args.nval, ns, nmu),
                           NamedSharding(mesh, P(None, None, None, "y")))

    def xla_body(T, W5):
        chain = build_xla_chain(mesh, kgrid, to_O=True)
        O_b = chain(T, W5)
        out = w_decode(psi_c, O_b, jnp.transpose(psi_v, (0, 2, 3, 1)))
        return jnp.transpose(out, (0, 2, 3, 1)) / jnp.sqrt(
            jnp.asarray(nk, dtype=T.real.dtype))

    print(f"\n=== rung A/B: kgrid {kgrid} nk={nk} mu=nu={nmu} ns={ns} "
          f"nc={args.ncond} nv={args.nval}\n"
          f"{'nb':>3} {'T(MB)':>8} {'chain_xla':>10} {'chain_ffi':>10} "
          f"{'x':>6} | {'rung_xla':>9} {'rung_ffi':>9} {'x':>6}  rel",
          flush=True)
    for nb in args.batches:
        T, W5, W3 = operands(mesh, nb, nmu, ns, kgrid)
        f_cx = jax.jit(build_xla_chain(mesh, kgrid, to_O=True))
        f_cf = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=1))
        f_rx = jax.jit(xla_body)
        f_rf = jax.jit(lambda t, w5, w3: fused_body(t, psi_c, psi_v, w5))
        rel_chain = relerr(f_cx(T, W5), f_cf(T, W3))
        rel_rung = relerr(f_rx(T, W5), f_rf(T, W5, W3))
        t_cx = timeit(f_cx, (T, W5), args.reps)
        t_cf = timeit(f_cf, (T, W3), args.reps)
        t_rx = timeit(f_rx, (T, W5), args.reps)
        t_rf = timeit(f_rf, (T, W5, W3), args.reps)
        row = {"nb": nb, "T_MB": T.size * 16 / 1e6,
               "chain_xla_ms": t_cx, "chain_ffi_ms": t_cf,
               "chain_speedup": t_cx / t_cf,
               "rung_xla_ms": t_rx, "rung_ffi_ms": t_rf,
               "rung_speedup": t_rx / t_rf,
               "rel_chain": rel_chain, "rel_rung": rel_rung}
        res["rows"].append(row)
        print(f"{nb:>3} {row['T_MB']:>8.1f} {t_cx:>10.3f} {t_cf:>10.3f} "
              f"{row['chain_speedup']:>6.2f} | {t_rx:>9.3f} {t_rf:>9.3f} "
              f"{row['rung_speedup']:>6.2f}  {rel_chain:.1e}/{rel_rung:.1e}",
              flush=True)
        del T, W5, W3
    return res


# ---------------------------------------------------------------------------
# mode: sizes  — THE GENERALITY GATE
# ---------------------------------------------------------------------------
def mode_sizes(args, mesh) -> dict:
    """Every axis extent in [1, 24], primes and mixed radices included.

    This is the cell that says the handler is SIZE-AGNOSTIC rather than tuned
    to one deck's k-grid.  It runs a spread of ``(nkx,nky,nkz)`` — powers of
    two, powers of three, primes (2,3,5,7,11,13,17,19,23), the awkward mixed
    ones, degenerate 1s on one or two axes, and the corner ``1x1x1`` — and
    for each asks for exactly two outcomes:

      * SERVED  → numerics agree with the XLA chain inside ``--tol``; or
      * REFUSED → the handler raised, and the message NAMES the reason.

    A silent wrong answer and a bare exception are both failures.  Anything
    the device can hold must be right; anything it cannot must say so, quote
    the device's own maximum, and name the alternative — which is what makes
    the envelope a stated contract instead of a discovered one.
    """
    ns, nmu, nb = args.nspinor, args.nmu, args.batches[0]
    res = {"mode": "sizes", "nmu": nmu, "nb": nb, "rows": []}
    grids = [
        (1, 1, 1), (2, 1, 1), (3, 1, 1), (5, 1, 1), (7, 1, 1), (11, 1, 1),
        (13, 1, 1), (17, 1, 1), (19, 1, 1), (23, 1, 1), (24, 1, 1),
        (2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1), (7, 7, 1), (6, 4, 1),
        (2, 3, 5), (3, 5, 7), (11, 3, 2), (13, 5, 1), (23, 2, 1),
        (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7),
        (8, 8, 8), (10, 10, 10), (12, 12, 12), (24, 24, 1), (24, 24, 24),
    ] if not args.kgrids_given else [tuple(g) for g in args.kgrids]
    print(f"\n=== sizes: mu=nu={nmu} nspinor={ns} nb={nb}\n"
          f"{'kgrid':>10} {'nk':>6} {'T(MB)':>8}  {'verdict':<9} detail",
          flush=True)
    n_served = n_refused = n_bad = 0
    for kgrid in grids:
        nk = int(np.prod(kgrid))
        mb = nb * nmu * nmu * ns * ns * nk * 16 / 1e6
        tag = "x".join(str(v) for v in kgrid)
        if mb > args.max_tile_MB:
            print(f"{tag:>10} {nk:>6} {mb:>8.0f}  {'SKIPPED':<9} "
                  f"tile over --max-tile-MB {args.max_tile_MB:.0f}", flush=True)
            res["rows"].append({"kgrid": list(kgrid), "nk": nk,
                                "verdict": "skipped", "T_MB": mb})
            continue
        try:
            T, W5, W3 = operands(mesh, nb, nmu, ns, kgrid)
        except Exception as exc:
            print(f"{tag:>10} {nk:>6} {mb:>8.1f}  {'SKIPPED':<9} "
                  f"operand alloc: {type(exc).__name__}", flush=True)
            continue
        row = {"kgrid": list(kgrid), "nk": nk, "T_MB": mb}
        try:
            f_ffi = jax.jit(K.make_rung_conv(mesh, kgrid, out_layout=1))
            got = f_ffi(T, W3)
            jax.block_until_ready(got)
        except Exception as exc:
            msg = str(exc)
            # A refusal must be INFORMATIVE: the reason, the device's own
            # limit, and what to use instead.  A bare traceback is a failure
            # of this cell even though the numbers were never wrong.
            named = ("conv_kminor" in msg
                     and "shared memory" in msg
                     and "k-STRIDED" in msg)
            row.update(verdict="refused", named=named, msg=msg[:400])
            res["rows"].append(row)
            n_refused += 1
            n_bad += 0 if named else 1
            print(f"{tag:>10} {nk:>6} {mb:>8.1f}  "
                  f"{'REFUSED' if named else 'REFUSED*':<9} "
                  f"{'named the bound + the alternative' if named else msg[:90]}",
                  flush=True)
            del T, W5, W3
            continue
        f_xla = jax.jit(build_xla_chain(mesh, kgrid, to_O=True))
        rel = relerr(f_xla(T, W5), got)
        ok = rel <= args.tol
        row.update(verdict="served", rel=rel, ok=bool(ok))
        res["rows"].append(row)
        n_served += 1
        n_bad += 0 if ok else 1
        print(f"{tag:>10} {nk:>6} {mb:>8.1f}  {'SERVED':<9} "
              f"rel = {rel:.3e} {'' if ok else '  <-- OVER BUDGET'}", flush=True)
        del T, W5, W3, got
    res.update(served=n_served, refused=n_refused, bad=n_bad,
               **{"pass": n_bad == 0})
    print(f"\n  {n_served} served, {n_refused} refused (all naming the bound), "
          f"{n_bad} bad -> {'PASS' if n_bad == 0 else 'FAIL'}", flush=True)
    return res


# ---------------------------------------------------------------------------
# mode: multigpu  — THE GPU-COUNT GATE
# ---------------------------------------------------------------------------
def mode_multigpu(args, mesh) -> dict:
    """The same numbers on a 2x2 mesh as on 1x1, and NO collective inside.

    The handler is rank-local by construction: it multiplies the local
    (mu_local, nu_local) tile against the local W tile and never talks to
    another rank.  "Works at any GPU count" therefore means two checkable
    things, and this cell checks both rather than asserting the first:

      1. VALUE: the sharded result equals the single-device XLA chain's, so
         the local tile shape — which changes with the mesh — is genuinely a
         runtime quantity and not a shape the kernel assumed.
      2. STRUCTURE: the compiled module contains no all-gather /
         collective-permute / reduce-scatter.  A handler that quietly needed a
         gather would still give the right answer here and would not scale.
    """
    ns, nmu, nb = args.nspinor, args.nmu, args.batches[0]
    kgrid = tuple(args.kgrids[0])
    devs = jax.devices()
    res = {"mode": "multigpu", "kgrid": list(kgrid), "nmu": nmu,
           "n_devices": len(devs), "cells": []}

    def cell(name, ok, detail):
        res["cells"].append({"cell": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    if len(devs) < 4:
        cell("M0 four devices present", False,
             f"only {len(devs)} device(s) visible; run with -G 4")
        res["pass"] = False
        return res
    res["processes"] = int(jax.process_count())

    # reference: the production XLA chain on a 1x1 mesh.  SKIPPED under
    # multi-process — device 0 is addressable only by process 0, so a 1x1 mesh
    # is not a thing every rank can build.  M2 (against the XLA chain at the
    # SAME geometry) is the cell that matters there anyway: it asks whether the
    # kernel and the reference agree on the tiles they actually hold.
    multi = jax.process_count() > 1
    ref = None
    if not multi:
        mesh1 = Mesh(np.asarray(devs[:1]).reshape(1, 1), ("x", "y"))
        with mesh1:
            T1, W5_1, W3_1 = operands(mesh1, nb, nmu, ns, kgrid, seed=17)
            ref = _host(jax.jit(build_xla_chain(mesh1, kgrid, to_O=True))(
                T1, W5_1))
        del T1, W5_1, W3_1

    mesh4 = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))
    with mesh4:
        T4, W5_4, W3_4 = operands(mesh4, nb, nmu, ns, kgrid, seed=17)
        fn = K.make_rung_conv(mesh4, kgrid, out_layout=1)
        got = jax.jit(fn)(T4, W3_4)
        jax.block_until_ready(got)
        if ref is None:
            cell("M1 2x2 mesh vs the 1x1 XLA chain", True,
                 f"SKIPPED under {jax.process_count()} processes (a 1x1 mesh "
                 f"is not addressable from every rank); M2 covers it")
        else:
            rel = relerr(ref, got)
            cell("M1 2x2 mesh vs the 1x1 XLA chain", rel <= args.tol,
                 f"rel = {rel:.3e} (budget {args.tol:.0e}); local tile "
                 f"mu={nmu // 2} nu={nmu // 2} per rank")
        # also against the XLA chain ON the 2x2 mesh, which is the arm the
        # integration would replace
        ref4 = jax.jit(build_xla_chain(mesh4, kgrid, to_O=True))(T4, W5_4)
        rel4 = relerr(ref4, got)
        cell("M2 2x2 mesh vs the 2x2 XLA chain", rel4 <= args.tol,
             f"rel = {rel4:.3e}")
        c = census(fn, (T4, W3_4), "ffi_2x2")
        res["census"] = c
        coll = 0 if "error" in c else sum(
            c["ops"][k] for k in ("all-gather", "collective-permute",
                                  "reduce-scatter"))
        cell("M3 no collective inside the handler's module",
             ("error" not in c) and coll == 0,
             c.get("error", f"all-gather {c['ops']['all-gather']}, "
                            f"collective-permute {c['ops']['collective-permute']}, "
                            f"reduce-scatter {c['ops']['reduce-scatter']}, "
                            f"custom-call {c['ops']['custom-call']}"))
        del T4, W5_4, W3_4, got
    res["pass"] = all(cc["pass"] for cc in res["cells"])
    return res


# ---------------------------------------------------------------------------
# mode: e2e  — THE PRODUCTION MATVEC, ARMED BY THE DIAL
# ---------------------------------------------------------------------------
def mode_e2e(args, mesh) -> dict:
    """The real ladder matvec on a real payload, dial OFF vs dial AUTO.

    This is the measurement every previous number in this lane was a
    projection of.  It builds the PRODUCTION operator
    (``bse.w_ladder.build_ladder_resolvent`` -> ``bse_ring_comm``) twice in one
    process, once with ``LORRAX_CONV_KMINOR_FFI=off`` and once with the default
    ``auto``, so the only thing that differs between the arms is which body the
    hook selected — no monkey-patched factories, no rebuilt operands.

    The RPA delegate (``include_w=False``) is timed beside both, because that
    is what makes the rung's share of the matvec a measurement rather than an
    inference.
    """
    import os as _os
    import harness                                              # noqa: F401
    from runtime import bootstrap
    bootstrap()
    from bse import bse_io
    from bse.bse_feast import ladder_matvec_operands, matvec_operands
    from bse.bse_w_exact import _symmetry_tables, build_finite_q_data
    from bse.w_ladder import build_ladder_resolvent
    from bse import w_ladder_shifts as wls
    from ffi.fft import conv_kminor_mode

    run_dir = Path(args.run_dir)
    input_path = str(run_dir / args.deck)
    restart = bse_io._find_restart_file(input_path)
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    nk = int(data['nkx'] * data['nky'] * data['nkz'])
    print(f"[payload] n_rmu={int(data['n_rmu'])} nc={int(data['eps_c'].shape[1])} "
          f"nv={int(data['eps_v'].shape[1])} nk={nk} "
          f"nspinor={int(data['psi_c_X'].shape[2])}", flush=True)
    sym = _symmetry_tables(input_path)
    q = tuple(int(v) for v in np.asarray(sym.q_irr_kgrid_int, dtype=int)[0])
    # dq is built AFTER the first build_ladder_resolvent: that call is what
    # runs ensure_W_R and puts W_R into `data`, and build_finite_q_data carries
    # it into the per-q operand bundle.  Building dq first gives a bundle with
    # no W_R and the matvec operand tuple KeyErrors.
    dq = None
    res = {"mode": "e2e", "nk": nk, "n_rmu": int(data['n_rmu']),
           "q": list(q), "rows": [], "cells": []}

    outs, times = {}, {}
    saved = _os.environ.get("LORRAX_CONV_KMINOR_FFI")
    try:
        for arm, dial in (("xla", "off"), ("kernel", "auto")):
            _os.environ["LORRAX_CONV_KMINOR_FFI"] = dial
            assert conv_kminor_mode() == dial
            for include_w in ([True, False] if arm == "xla" else [True]):
                matvec, _, gen, snapshot, sh = build_ladder_resolvent(
                    mesh, data, include_w=include_w)
                if dq is None:
                    dq = build_finite_q_data(data, q, mesh)
                    n_pad = int(dq["V_q0"].shape[0])
                ops = tuple(ladder_matvec_operands(dq) if include_w
                            else matvec_operands(dq))
                label = f"{arm}{'' if include_w else '_rpa'}"
                for width in args.batches:
                    G = np.eye(n_pad)[:width, :]
                    rhs = wls.seed_probe_block(G, dq, gen, sh)

                    # Operands are ARGUMENTS, never closed over: under
                    # multi-process a jitted callable that closes over an array
                    # spanning non-addressable devices is refused outright, and
                    # `matvec` is already jitted with its own in_shardings, so
                    # re-wrapping it would only add that hazard.
                    key = (label, width)
                    outs[key] = jax.block_until_ready(matvec(rhs, *ops))
                    times[key] = timeit(matvec, (rhs, *ops), args.reps)
                    print(f"  {label:11s} width={width:3d}  "
                          f"{times[key]:9.3f} ms/matvec", flush=True)
    finally:
        if saved is None:
            _os.environ.pop("LORRAX_CONV_KMINOR_FFI", None)
        else:
            _os.environ["LORRAX_CONV_KMINOR_FFI"] = saved

    print(f"\n{'width':>6} {'xla[ms]':>9} {'kernel[ms]':>11} {'speedup':>8} "
          f"{'rpa[ms]':>9} {'rung share':>11}   rel", flush=True)
    ok_all = True
    for width in args.batches:
        tx = times[("xla", width)]
        tk = times[("kernel", width)]
        tr = times.get(("xla_rpa", width), float("nan"))
        rel = relerr(outs[("xla", width)], outs[("kernel", width)])
        ok = rel <= args.e2e_tol
        ok_all &= ok
        share = (tx - tr) / tx if tr == tr else float("nan")
        res["rows"].append({"width": width, "xla_ms": tx, "kernel_ms": tk,
                            "speedup": tx / tk, "rpa_ms": tr,
                            "rung_share": share, "rel": rel, "ok": bool(ok)})
        print(f"{width:>6} {tx:>9.3f} {tk:>11.3f} {tx/tk:>8.2f} {tr:>9.3f} "
              f"{share:>10.1%}   {rel:.3e}{'' if ok else '  <-- OVER'}",
              flush=True)
    res["pass"] = bool(ok_all)
    print(f"\n  matvec numerics vs the XLA arm: "
          f"{'PASS' if ok_all else 'FAIL'} (budget {args.e2e_tol:.0e})",
          flush=True)
    return res


# ---------------------------------------------------------------------------
# mode: overlap  — CAN THE CHAIN HIDE BEHIND THE GEMMs?
# ---------------------------------------------------------------------------
def mode_overlap(args, mesh) -> dict:
    """Is there anything left to overlap, and does XLA already do it?

    The question is worth asking because the non-TDA ladder matvec applies the
    rung TWICE — resonant and antiresonant — on INDEPENDENT operands.  Those
    two applications have no data dependency, so if the scheduler can run them
    concurrently the chain is already hidden behind the other's GEMMs and there
    is nothing for a stream to buy.

    The experiment needs no profiler and no restructuring: time ONE rung
    application, then time TWO independent ones inside a single jit.

        ratio = t(2 applications) / t(1 application)

      ~2.0  -> strictly serial; concurrency is available but unused
      ~1.0  -> already fully overlapped
      in between -> partially overlapped

    Reported for both the XLA arm and the fused arm, because the answer can
    differ: the fused arm is ONE kernel where XLA's is five, and a single
    long-running kernel that already fills the device has no room to overlap
    with anything, which is itself the finding.

    WHY NOT STREAMS AT THE FFI BOUNDARY.  The handler is handed XLA's stream
    and must run on it: that is the contract that lets XLA reason about
    ordering.  A handler that launched onto a private stream would have to
    synchronise it back before returning, which serialises anyway, or return
    early and race the consumer.  Overlap in this architecture is the
    SCHEDULER's to find, so the honest experiment is whether it finds it.
    """
    ns, nmu = args.nspinor, args.nmu
    kgrid = tuple(args.kgrids[0])
    nb = args.batches[0]
    res = {"mode": "overlap", "kgrid": list(kgrid), "nmu": nmu, "nb": nb,
           "rows": []}
    T, W5, W3 = operands(mesh, nb, nmu, ns, kgrid, seed=5)
    T2, W5b, W3b = operands(mesh, nb, nmu, ns, kgrid, seed=6)

    chain_x = build_xla_chain(mesh, kgrid, to_O=True)
    conv_f = K.make_rung_conv(mesh, kgrid, out_layout=1)

    pairs = [
        ("xla", jax.jit(chain_x), (T, W5),
         jax.jit(lambda a, b, c, d: (chain_x(a, b), chain_x(c, d))),
         (T, W5, T2, W5b)),
        ("ffi", jax.jit(conv_f), (T, W3),
         jax.jit(lambda a, b, c, d: (conv_f(a, b), conv_f(c, d))),
         (T, W3, T2, W3b)),
    ]
    print(f"\n=== overlap probe: kgrid {kgrid} mu=nu={nmu} nb={nb}\n"
          f"{'arm':>6} {'1 apply[ms]':>12} {'2 applies[ms]':>14} {'ratio':>7}"
          f"   verdict", flush=True)
    for name, f1, a1, f2, a2 in pairs:
        t1 = timeit(f1, a1, args.reps)
        t2 = timeit(f2, a2, args.reps)
        r = t2 / t1
        verdict = ("already overlapped" if r < 1.25 else
                   ("partially overlapped" if r < 1.75 else "serial"))
        res["rows"].append({"arm": name, "one_ms": t1, "two_ms": t2,
                            "ratio": r, "verdict": verdict})
        print(f"{name:>6} {t1:>12.3f} {t2:>14.3f} {r:>7.2f}   {verdict}",
              flush=True)
    del T, W5, W3, T2, W5b, W3b
    return res


# ---------------------------------------------------------------------------
def _kgrid(s: str):
    v = tuple(int(x) for x in s.replace("x", ",").split(","))
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"kgrid must be a,b,c; got {s!r}")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=("gate", "ksweep", "rung", "sizes", "multigpu",
                             "overlap", "e2e"),
                    default="gate")
    ap.add_argument("--nmu", type=int, default=399)
    ap.add_argument("--nspinor", type=int, default=2)
    ap.add_argument("--ncond", type=int, default=20)
    ap.add_argument("--nval", type=int, default=26)
    ap.add_argument("--kgrids", type=_kgrid, nargs="+",
                    default=[(3, 3, 1)])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-15)
    ap.add_argument("--ksweep-tol", type=float, default=1.05)
    ap.add_argument("--mesh", type=str, default="1x1",
                    help="mesh geometry: 1x1 | 2x2 | auto")
    ap.add_argument("--run-dir", type=str,
                    default="/pscratch/sd/j/jackm/wbse_freq_fixture")
    ap.add_argument("--deck", type=str, default="gnppm_test.in")
    ap.add_argument("--e2e-tol", type=float, default=1e-13)
    ap.add_argument("--max-tile-MB", type=float, default=6000.0,
                    help="sizes mode: skip a k-grid whose T tile exceeds this")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)
    # `sizes` uses its own spread unless the caller named grids explicitly.
    args.kgrids_given = any(a.startswith("--kgrid") for a in (argv or sys.argv))

    # BOOTSTRAP FIRST, and this is not optional under -n>1: `jax.devices()`
    # returns only the LOCAL devices until jax.distributed.initialize has run,
    # so a multi-process leg that queries devices first silently sees a 1-GPU
    # world and refuses a mesh it could have built.
    if args.mode in ("e2e", "multigpu") or args.mesh != "1x1":
        sys.path.insert(0, str(_REPO / "tests"))
        from runtime import bootstrap as _bootstrap
        _bootstrap()
    devs = jax.devices()
    # MESH GEOMETRY.  Default 1x1 keeps every microbench a single-device
    # measurement; `--mesh 2x2` (or `auto`, which takes 2x2 when four devices
    # are visible) is what the P>1 certification legs use, and it is the SAME
    # code under one process driving four devices and under four processes
    # driving one each — which is the point of certifying both.
    if args.mesh == "auto":
        geom = (2, 2) if len(devs) >= 4 else (1, 1)
    else:
        geom = tuple(int(v) for v in args.mesh.lower().split("x"))
    need = geom[0] * geom[1]
    if len(devs) < need:
        print(f"[REFUSED] --mesh {geom[0]}x{geom[1]} needs {need} devices; "
              f"{len(devs)} visible.")
        return 2
    mesh = Mesh(np.asarray(devs[:need]).reshape(*geom), ("x", "y"))
    print(f"[env] mesh {geom[0]}x{geom[1]} | processes={jax.process_count()} "
          f"index={jax.process_index()} | local devices={len(jax.local_devices())}")
    ok, why = conv_kminor_available(mesh)
    print(f"[env] jax={jax.__version__} devices={devs}")
    print(f"[env] allocator={_ALLOC} "
          f"(XLA_PYTHON_CLIENT_ALLOCATOR="
          f"{os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', '<unset>')})")
    print(f"[env] LORRAX_FFI_SO={os.environ.get('LORRAX_FFI_SO', '<unset>')}")
    print(f"[env] conv_kminor handler: {'PRESENT' if ok else 'ABSENT'} — {why}")
    if not ok:
        print("[REFUSED] the handler is not reachable; nothing measured.")
        return 2

    with mesh:
        try:
            result = {"gate": mode_gate, "ksweep": mode_ksweep,
                      "rung": mode_rung, "sizes": mode_sizes,
                      "multigpu": mode_multigpu,
                      "overlap": mode_overlap,
                      "e2e": mode_e2e}[args.mode](args, mesh)
        except Exception:
            traceback.print_exc()
            return 3

    result["env"] = {"jax": jax.__version__, "allocator": _ALLOC,
                     "so": os.environ.get("LORRAX_FFI_SO", ""),
                     "devices": [str(d) for d in devs]}
    # ONE writer.  Under -n>1 every process reaches this line with the same
    # result, and all of them writing the same path interleaves into invalid
    # JSON — which is how a passing leg produces an unreadable artifact.
    if args.out and jax.process_index() == 0:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        p = out / f"bench_conv_kminor_{args.mode}_{stamp}.json"
        p.write_text(json.dumps(result, indent=1))
        print(f"\n[out] {p}")
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
