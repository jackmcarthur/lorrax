#!/usr/bin/env python
"""THE FIT-STAGE PROBE: does the batched Pade fit compile, and where does
the time actually go?

MEASUREMENT ONLY.  Nothing here is imported by production code; it calls
the shipped entry points (``pade_fit.fit_mpa_poles_batched``,
``diagnostics.diagnostics_batched``) on a REAL column block off the
production W_c store and reports, in one leg:

  A  the shipped per-block cost, split fit / diagnostic
  B  whether ``jax.jit`` of the batched fit compiles AT ALL on a GPU,
     and whether its answer is bit-identical to the eager one
  C  what the compiled HLO actually calls -- which is the only way to
     answer "does the companion-root eigvals have a GPU lowering"
     without guessing
  D  eigvals cost against batch size and against n_p, which separates
     per-element call overhead from n^3 arithmetic
  E  whether XLA common-subexpression-eliminates a duplicated Pade
     solve (if it does, the redundancy the docstring admits is free
     under jit and the restructure is unnecessary)
  F  the 1-fit-1-solve candidate, timed and checked for bit-identity
     against the shipped two-fit path

Usage:
  probe_fit_jit.py [--sections ABCDEF] [--n-elements N] [--q Q]
Environment: WC_STORE, WC_NAME as in scripts/perf_mpa16/env.sh.
"""
from __future__ import annotations

import argparse
import os
import time
import traceback

import numpy as np

T_ENTRY = time.time()

from file_io import mpa_store  # noqa: E402
from gw.mpa import diagnostics, fit_driver, pade_fit, tiling  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def banner(text):
    print("", flush=True)
    print("=" * 72, flush=True)
    print(text, flush=True)
    print("=" * 72, flush=True)


def timeit(fn, *, warmup=1, reps=3):
    """min-of-reps wall seconds, with the result of the last call."""
    out = None
    for _ in range(warmup):
        out = fn()
        jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return min(ts), ts, out


def solve_conditioning_only(W_samples, z_samples, n_p, *, rcond=1.0e-13):
    """``diagnostics.solve_conditioning`` MINUS its redundant refit.

    Same expression tree, op for op, for ``cond``, ``sigma_max``,
    ``sigma_min`` and ``backward_error``; the ``fit_mpa_poles`` call
    whose three outputs the driver discards is simply not made.  This is
    the candidate the restructure lands, staged here so the probe can
    check bit-identity before any production file moves.
    """
    pade_fit._require_x64()
    pade_fit._check_sample_support(W_samples, z_samples, n_p)
    n = int(n_p)
    w = jnp.asarray(W_samples, dtype=jnp.complex128)
    z = jnp.asarray(z_samples, dtype=jnp.complex128)

    A, rhs, _, _ = pade_fit.build_pade_system(w, z, n)
    row_norm = jnp.linalg.norm(A, axis=1)
    row_norm = jnp.where(row_norm > 0, row_norm, 1.0)
    A_n = A / row_norm[:, None]
    rhs_n = rhs / row_norm

    y, cond, s_max, s_min = pade_fit._solve_normalised(A, rhs, rcond)
    num = jnp.linalg.norm(A_n @ y - rhs_n)
    den = (jnp.linalg.norm(A_n) * jnp.linalg.norm(y)
           + jnp.linalg.norm(rhs_n))
    den = jnp.where(den > 0, den, 1.0)
    return {
        "cond": cond,
        "sigma_max": s_max,
        "sigma_min": s_min,
        "backward_error": num / den,
    }


def bitcmp(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"  {name:24s} SHAPE MISMATCH {a.shape} vs {b.shape}",
              flush=True)
        return False
    same = np.array_equal(a.view(np.uint8), b.view(np.uint8))
    if same:
        print(f"  {name:24s} BYTE-IDENTICAL ({a.size} values)", flush=True)
        return True
    d = np.abs(a - b)
    finite = np.isfinite(d)
    worst = float(np.max(d[finite])) if np.any(finite) else float("nan")
    scale = float(np.max(np.abs(b[finite]))) if np.any(finite) else 1.0
    n_diff = int(np.count_nonzero(
        a.view(np.uint8) != b.view(np.uint8)))
    print(f"  {name:24s} DIFFERS: {n_diff} bytes, max|d| {worst:.3e} "
          f"against max|ref| {scale:.3e} "
          f"(rel {worst / scale if scale else float('nan'):.3e})",
          flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="ABCDEF")
    ap.add_argument("--n-elements", type=int, default=0,
                    help="subsample the block's elements; 0 = the whole block")
    ap.add_argument("--q", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    banner("0. ENVIRONMENT")
    print(f"  jax {jax.__version__}", flush=True)
    print(f"  devices {jax.devices()}", flush=True)
    print(f"  process {jax.process_index()} of {jax.process_count()}",
          flush=True)
    print(f"  x64 {pade_fit._x64_is_on()}", flush=True)
    for k in ("XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
              "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_ENABLE_X64",
              "JAX_PLATFORMS", "LORRAX_GPU_DEVICE"):
        print(f"  {k}={os.environ.get(k)}", flush=True)
    for flag in ("jax_use_magma", "jax_default_matmul_precision"):
        try:
            print(f"  config {flag}={jax.config.read(flag)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  config {flag}: unavailable ({type(exc).__name__})",
                  flush=True)
    print(f"  import+init {time.time() - T_ENTRY:.2f} s", flush=True)

    w_src = os.environ["WC_STORE"]
    w_name = os.environ.get("WC_NAME", "W_qmunu_omega")
    header = mpa_store.read_w_header(w_src, w_name)
    n_mu = header["n_mu"]
    n_omega = header["n_omega"]
    n_p = n_omega // 2
    z = np.asarray(header["omega"], dtype=np.complex128)
    plan = tiling.plan_column_walk(n_mu, n_omega, None)
    lo, hi = plan["blocks"][0]

    t_read = time.perf_counter()
    block = mpa_store.read_w_columns(
        w_src, w_name, args.q, np.arange(lo, hi), tile_bytes=None,
        out_spec=tiling.row_shard_spec())
    t_read = time.perf_counter() - t_read
    tile_np = fit_driver._elements_from_block(block)
    if args.n_elements:
        tile_np = tile_np[:args.n_elements]
    n_elem = tile_np.shape[0]
    print(f"  block q={args.q} cols[{lo}:{hi}] read in {t_read:.2f} s -> "
          f"tile {tile_np.shape} (n_p={n_p}, N_mu={n_mu}, "
          f"{plan['n_blocks']} blocks/q)", flush=True)

    tile = jnp.asarray(tile_np, dtype=jnp.complex128)
    zj = jnp.asarray(z, dtype=jnp.complex128)

    def us(sec):
        return 1.0e6 * sec / n_elem

    shipped_fit = None
    shipped_diag = None

    # ---------------------------------------------------------------- A
    if "A" in args.sections:
        banner("A. THE SHIPPED PATH, AS fit_one_block CALLS IT (eager)")
        t_fit, all_fit, shipped_fit = timeit(
            lambda: pade_fit.fit_mpa_poles_batched(tile, zj, n_p),
            reps=args.reps)
        t_dia, all_dia, shipped_diag = timeit(
            lambda: diagnostics.diagnostics_batched(
                diagnostics.solve_conditioning, tile, zj, n_p),
            reps=args.reps)
        print(f"  fit_mpa_poles_batched   {t_fit:.3f} s  "
              f"{us(t_fit):8.1f} us/element   reps {['%.3f' % v for v in all_fit]}",
              flush=True)
        print(f"  solve_conditioning      {t_dia:.3f} s  "
              f"{us(t_dia):8.1f} us/element   reps {['%.3f' % v for v in all_dia]}",
              flush=True)
        print(f"  SHIPPED BLOCK TOTAL     {t_fit + t_dia:.3f} s  "
              f"{us(t_fit + t_dia):8.1f} us/element", flush=True)

    # ---------------------------------------------------------------- B
    jit_fit = None
    if "B" in args.sections:
        banner("B. DOES THE BATCHED FIT jit AT ALL?")

        def _f(t):
            return pade_fit.fit_mpa_poles_batched(t, zj, n_p)

        try:
            jf = jax.jit(_f)
            t0 = time.perf_counter()
            lowered = jf.lower(tile)
            t_lower = time.perf_counter() - t0
            t0 = time.perf_counter()
            compiled = lowered.compile()
            t_compile = time.perf_counter() - t0
            print(f"  VERDICT: COMPILES.  lower {t_lower:.2f} s, "
                  f"compile {t_compile:.2f} s", flush=True)
            t_jit, all_jit, jit_fit = timeit(lambda: jf(tile),
                                             reps=args.reps)
            print(f"  jit(fit) run            {t_jit:.3f} s  "
                  f"{us(t_jit):8.1f} us/element  reps "
                  f"{['%.3f' % v for v in all_jit]}", flush=True)
            if shipped_fit is not None:
                print("  bit-identity of jit vs eager:", flush=True)
                bitcmp("Omega", jit_fit[0], shipped_fit[0])
                bitcmp("B", jit_fit[1], shipped_fit[1])
                bitcmp("max_abs_residual", jit_fit[2]["max_abs_residual"],
                       shipped_fit[2]["max_abs_residual"])
                bitcmp("n_valid", jit_fit[2]["n_valid"],
                       shipped_fit[2]["n_valid"])
            try:
                mem = compiled.memory_analysis()
                print(f"  compiled memory: temp {mem.temp_size_in_bytes} B, "
                      f"output {mem.output_size_in_bytes} B", flush=True)
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"  VERDICT: DOES NOT COMPILE -- "
                  f"{type(exc).__name__}", flush=True)
            print("  " + "\n  ".join(
                traceback.format_exc().splitlines()[-14:]), flush=True)

    # ---------------------------------------------------------------- C
    if "C" in args.sections:
        banner("C. WHAT THE COMPILED FIT ACTUALLY CALLS (HLO census)")
        try:
            text = jax.jit(
                lambda t: pade_fit.fit_mpa_poles_batched(t, zj, n_p)
            ).lower(tile).as_text()
            import re
            targets = re.findall(r'custom_call_target\s*=\s*"([^"]+)"', text)
            from collections import Counter
            for name, cnt in sorted(Counter(targets).items()):
                print(f"  custom_call {name:40s} x{cnt}", flush=True)
            if not targets:
                print("  (no custom calls in the lowered StableHLO)",
                      flush=True)
            for probe in ("eig", "svd", "lapack", "magma", "cusolver",
                          "host", "TransferTo", "while", "sort"):
                n = len(re.findall(probe, text, flags=re.IGNORECASE))
                print(f"  token {probe:12s} appears {n} times", flush=True)
            # And the eigvals call on its own, which is the question.
            comp = jnp.zeros((n_p, n_p), dtype=jnp.complex128)
            etext = jax.jit(jnp.linalg.eigvals).lower(comp).as_text()
            etargets = re.findall(
                r'custom_call_target\s*=\s*"([^"]+)"', etext)
            print(f"  eigvals(8x8) lowering custom calls: "
                  f"{sorted(set(etargets)) or 'NONE'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  HLO census failed: {type(exc).__name__}: {exc}",
                  flush=True)

    # ---------------------------------------------------------------- D
    if "D" in args.sections:
        banner("D. THE COMPANION-ROOT eigvals, BY BATCH AND BY n_p")
        print("  (batched jnp.linalg.eigvals on complex128 companion "
              "matrices)", flush=True)
        rng = np.random.default_rng(20260810)
        for n in (4, 8, 12, 16):
            for nb in (1, 64, 1024, min(n_elem, 16384)):
                mats = (rng.standard_normal((nb, n, n))
                        + 1j * rng.standard_normal((nb, n, n)))
                dm = jnp.asarray(mats, dtype=jnp.complex128)
                try:
                    t_e, _, _ = timeit(lambda: jnp.linalg.eigvals(dm),
                                       reps=max(2, args.reps - 1))
                    per = 1.0e6 * t_e / nb
                    print(f"  n_p={n:3d} batch={nb:6d}  {t_e:8.4f} s  "
                          f"{per:9.1f} us/element (eager)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  n_p={n:3d} batch={nb:6d}  EAGER FAILED "
                          f"{type(exc).__name__}: {str(exc)[:120]}",
                          flush=True)
                try:
                    je = jax.jit(jnp.linalg.eigvals)
                    t_e, _, _ = timeit(lambda: je(dm),
                                       reps=max(2, args.reps - 1))
                    per = 1.0e6 * t_e / nb
                    print(f"  n_p={n:3d} batch={nb:6d}  {t_e:8.4f} s  "
                          f"{per:9.1f} us/element (jit)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  n_p={n:3d} batch={nb:6d}  JIT FAILED "
                          f"{type(exc).__name__}: {str(exc)[:120]}",
                          flush=True)
        # The same on the CPU backend, for the documented 35-41 us row.
        try:
            cpu = jax.devices("cpu")[0]
            for n in (8,):
                for nb in (1024, min(n_elem, 16384)):
                    mats = (rng.standard_normal((nb, n, n))
                            + 1j * rng.standard_normal((nb, n, n)))
                    dm = jax.device_put(
                        jnp.asarray(mats, dtype=jnp.complex128), cpu)
                    t_e, _, _ = timeit(lambda: jnp.linalg.eigvals(dm), reps=2)
                    print(f"  CPU BACKEND n_p={n} batch={nb:6d} "
                          f"{t_e:8.4f} s {1.0e6 * t_e / nb:9.1f} us/element",
                          flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  cpu-backend eigvals unavailable: "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)

        # And the fit WITHOUT its roots, to price the rest of the kernel.
        banner("D2. THE FIT'S OTHER STAGES, PRICED SEPARATELY")

        def _system_and_solve(t):
            def one(w):
                A, rhs, x_hat, x_max = pade_fit.build_pade_system(w, zj, n_p)
                coef, cond, s_max, s_min = pade_fit._solve_normalised(
                    A, rhs, 1.0e-13)
                return coef, cond
            return jax.vmap(one)(t)

        try:
            t_s, _, _ = timeit(lambda: _system_and_solve(tile),
                               reps=args.reps)
            print(f"  build+SVD solve only    {t_s:.3f} s  "
                  f"{us(t_s):8.1f} us/element (eager)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  build+solve failed: {type(exc).__name__}: {exc}",
                  flush=True)

        def _roots_only(t):
            def one(w):
                A, rhs, x_hat, x_max = pade_fit.build_pade_system(w, zj, n_p)
                coef, _, _, _ = pade_fit._solve_normalised(A, rhs, 1.0e-13)
                return pade_fit._companion_roots(coef[n_p:])
            return jax.vmap(one)(t)

        try:
            t_r, _, _ = timeit(lambda: _roots_only(tile), reps=args.reps)
            print(f"  build+solve+roots       {t_r:.3f} s  "
                  f"{us(t_r):8.1f} us/element (eager)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  roots leg failed: {type(exc).__name__}: {exc}",
                  flush=True)

    # ---------------------------------------------------------------- E
    if "E" in args.sections:
        banner("E. DOES XLA CSE A DUPLICATED PADE SOLVE?")

        def one_solve(t):
            def one(w):
                A, rhs, _, _ = pade_fit.build_pade_system(w, zj, n_p)
                coef, cond, _, _ = pade_fit._solve_normalised(A, rhs, 1.0e-13)
                return coef, cond
            return jax.vmap(one)(t)

        def two_solves(t):
            def one(w):
                A, rhs, _, _ = pade_fit.build_pade_system(w, zj, n_p)
                c1, k1, _, _ = pade_fit._solve_normalised(A, rhs, 1.0e-13)
                c2, k2, _, _ = pade_fit._solve_normalised(A, rhs, 1.0e-13)
                return c1 + c2, k1 + k2
            return jax.vmap(one)(t)

        try:
            j1 = jax.jit(one_solve)
            j2 = jax.jit(two_solves)
            t1, _, _ = timeit(lambda: j1(tile), reps=args.reps)
            t2, _, _ = timeit(lambda: j2(tile), reps=args.reps)
            print(f"  jit one solve  {t1:.3f} s", flush=True)
            print(f"  jit two solves {t2:.3f} s  -> ratio {t2 / t1:.2f}x "
                  f"({'CSE: the duplicate is free' if t2 < 1.35 * t1 else 'NO CSE: the duplicate is paid'})",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  CSE probe failed: {type(exc).__name__}: {exc}",
                  flush=True)

        # The same question for the whole fit, which is what the driver
        # would actually be duplicating.
        try:
            jfit1 = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(
                t, zj, n_p))
            jfit2 = jax.jit(lambda t: (
                pade_fit.fit_mpa_poles_batched(t, zj, n_p),
                diagnostics.diagnostics_batched(
                    diagnostics.solve_conditioning, t, zj, n_p)))
            ta, _, _ = timeit(lambda: jfit1(tile), reps=args.reps)
            tb, _, _ = timeit(lambda: jfit2(tile), reps=args.reps)
            print(f"  jit fit alone            {ta:.3f} s  "
                  f"{us(ta):8.1f} us/element", flush=True)
            print(f"  jit fit + solve_cond     {tb:.3f} s  "
                  f"{us(tb):8.1f} us/element  -> ratio {tb / ta:.2f}x",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  whole-fit CSE probe failed: {type(exc).__name__}: "
                  f"{exc}", flush=True)

    # ---------------------------------------------------------------- F
    if "F" in args.sections:
        banner("F. THE 1-FIT CANDIDATE: fit once, condition without refitting")
        t_c, all_c, cond_only = timeit(
            lambda: diagnostics.diagnostics_batched(
                solve_conditioning_only, tile, zj, n_p),
            reps=args.reps)
        print(f"  conditioning-only       {t_c:.3f} s  "
              f"{us(t_c):8.1f} us/element  reps "
              f"{['%.3f' % v for v in all_c]}", flush=True)
        if shipped_diag is not None:
            print("  bit-identity against solve_conditioning:", flush=True)
            bitcmp("cond", cond_only["cond"], shipped_diag["cond"])
            bitcmp("backward_error", cond_only["backward_error"],
                   shipped_diag["backward_error"])
            bitcmp("sigma_max", cond_only["sigma_max"],
                   shipped_diag["sigma_max"])
            bitcmp("sigma_min", cond_only["sigma_min"],
                   shipped_diag["sigma_min"])
        if shipped_fit is not None:
            print("  the fit's own cond_pade against solve_conditioning's "
                  "cond (the driver stores the latter):", flush=True)
            bitcmp("cond_pade vs cond", shipped_fit[2]["cond_pade"],
                   shipped_diag["cond"] if shipped_diag is not None
                   else shipped_fit[2]["cond_pade"])

        # One vmap instead of two: the same work, one dispatch.
        def fused(t):
            def one(w):
                Om, B, dg = pade_fit.fit_mpa_poles(w, zj, n_p)
                ck = solve_conditioning_only(w, zj, n_p)
                return Om, B, dg, ck
            return jax.vmap(one)(t)

        try:
            t_f, all_f, fused_out = timeit(lambda: fused(tile),
                                           reps=args.reps)
            print(f"  ONE fused vmap          {t_f:.3f} s  "
                  f"{us(t_f):8.1f} us/element  reps "
                  f"{['%.3f' % v for v in all_f]}", flush=True)
            if shipped_fit is not None:
                print("  fused vs shipped fit, bitwise:", flush=True)
                bitcmp("Omega", fused_out[0], shipped_fit[0])
                bitcmp("B", fused_out[1], shipped_fit[1])
            if shipped_diag is not None:
                bitcmp("backward_error", fused_out[3]["backward_error"],
                       shipped_diag["backward_error"])
                bitcmp("cond", fused_out[3]["cond"], shipped_diag["cond"])
        except Exception as exc:  # noqa: BLE001
            print(f"  fused vmap failed: {type(exc).__name__}: {exc}",
                  flush=True)

        try:
            jfused = jax.jit(fused)
            t_jf, _, jfused_out = timeit(lambda: jfused(tile),
                                         reps=args.reps)
            print(f"  ONE fused vmap, jitted  {t_jf:.3f} s  "
                  f"{us(t_jf):8.1f} us/element", flush=True)
            if shipped_fit is not None:
                print("  jitted fused vs shipped fit, bitwise:", flush=True)
                bitcmp("Omega", jfused_out[0], shipped_fit[0])
                bitcmp("B", jfused_out[1], shipped_fit[1])
        except Exception as exc:  # noqa: BLE001
            print(f"  jitted fused failed: {type(exc).__name__}: {exc}",
                  flush=True)

    banner("PROBE COMPLETE")


if __name__ == "__main__":
    main()
