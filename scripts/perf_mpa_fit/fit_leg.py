#!/usr/bin/env python
"""ONE FARM LEG of the MPA Pade fit: fit q in [qlo, qhi) and report cost.

VERBATIM COPY of ``scripts/perf_mpa16/fit_leg.py`` from
``perf/mpa-16gpu-2026-08-10`` @ ``f729f131``, which is the harness the
14.7-minute sixteen-GPU baseline was measured on.  Two strings differ and
nothing else: the provenance ``lane`` this leg stamps into its store, and
this paragraph.  The point of copying rather than improving it is that a
before/after wall is only a comparison if the stopwatch is the same
stopwatch.

MEASUREMENT HARNESS, NOT A DRIVER.  It calls ``fit_driver.fit_one_block``
-- the same entry point ``run_fit_driver`` walks -- and changes nothing
about what is computed.  It exists because ``run_fit_driver`` walks ALL q
in one process, and the question this lane is asked is how the walk
divides across a farm.

WHAT IT ADDS OVER ``run_fit_driver``: a q window, and a stopwatch on the
things a farm plan needs separated -- the leg's bring-up (python import,
lorrax import, jax device init) against its steady-state per-block cost.
Bring-up is charged once per leg, so it is the term that decides how fine
a farm may be sliced, and it cannot be read off a whole-run wall clock.

Usage:  fit_leg.py <qlo> <qhi> <out.h5> [--max-blocks N] [--tag T]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# THE FIRST STAMP IS BEFORE EVERY IMPORT.  Everything above this line is
# the container and the interpreter; everything below is chargeable to the
# leg, and the two are separated because only the second one is ours to
# fix.
T_ENTRY = time.time()
print(f"[leg] python entry, {T_ENTRY:.3f} epoch", flush=True)

import numpy as np  # noqa: E402

T_NUMPY = time.time()

from file_io import mpa_store  # noqa: E402
from gw.mpa import fit_driver, tiling  # noqa: E402

T_LORRAX = time.time()
print(f"[leg] imports: numpy {T_NUMPY - T_ENTRY:.2f} s, "
      f"lorrax {T_LORRAX - T_NUMPY:.2f} s", flush=True)

import jax  # noqa: E402

_DEVS = jax.devices()
T_JAX = time.time()
print(f"[leg] jax {jax.__version__} devices={_DEVS} "
      f"init {T_JAX - T_LORRAX:.2f} s", flush=True)
print(f"[leg] BRING-UP TOTAL (entry -> first device): "
      f"{T_JAX - T_ENTRY:.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qlo", type=int)
    ap.add_argument("qhi", type=int)
    ap.add_argument("out")
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="stop after N blocks (calibration); 0 = all")
    ap.add_argument("--tag", default="leg")
    args = ap.parse_args()

    w_src = os.environ["WC_STORE"]
    w_name = os.environ.get("WC_NAME", "W_qmunu_omega")

    header = mpa_store.read_w_header(w_src, w_name)
    n_mu = header["n_mu"]
    n_omega = header["n_omega"]
    n_q = header["n_q_on_disk"]
    n_p = n_omega // 2
    z = np.asarray(header["omega"], dtype=np.complex128)

    # The same refusal ``run_fit_driver`` makes, at the same seam: the
    # multipole method fits W - v and a store filled from the Dyson solve
    # holds W, whose poles are dominated by the bare Coulomb interaction.
    content = mpa_store.require_correlation_part(
        header.get("screening_content"),
        where=f"fit_leg[{args.tag}]", source=f"{w_src} :: {w_name}")

    plan = tiling.plan_column_walk(n_mu, n_omega, None)
    print(f"[leg] geometry n_q={n_q} N_mu={n_mu} n_omega={n_omega} "
          f"n_p={n_p} content={content}", flush=True)
    print(f"[leg] walk {plan['n_blocks']} blocks per q x "
          f"{args.qhi - args.qlo} q, {plan['n_cols']} columns per block",
          flush=True)

    t_alloc = time.perf_counter()
    mpa_store.allocate_fit_store(
        args.out, n_q=n_q, n_mu=n_mu, n_p=n_p,
        screening_content=content,
        energy_unit=header["omega_units"],
        grid_hash=header["grid_hash"],
        table_hash=header["table_hash"],
        centroid_hash=header["centroid_hash"],
        provenance={"lane": "perf/mpa-fit-efficiency-2026-08-10",
                    "leg": str(args.tag),
                    "q_window": f"[{args.qlo},{args.qhi})"})
    t_alloc = time.perf_counter() - t_alloc
    print(f"[leg] store allocated in {t_alloc:.2f} s -> {args.out}",
          flush=True)

    spec = tiling.row_shard_spec()
    sched = [s for s in tiling.fit_schedule(n_q, n_mu, n_omega, None)
             if args.qlo <= s[0] < args.qhi]
    if args.max_blocks:
        sched = sched[:args.max_blocks]

    tot = {"read": 0.0, "fit": 0.0, "write": 0.0}
    n_cols_done = 0
    n_elem = 0
    t_walk = time.perf_counter()
    T_FIRST = None
    for i, (q, lo, hi) in enumerate(sched):
        t_b = time.perf_counter()
        stats = fit_driver.fit_one_block(
            w_src, w_name, args.out, q, np.arange(lo, hi), z, n_p,
            tile_bytes=None, out_spec=spec)
        dt = time.perf_counter() - t_b
        if T_FIRST is None:
            T_FIRST = dt
            print(f"[leg] FIRST BLOCK {dt:.2f} s "
                  f"(carries the jit compile)", flush=True)
        tot["read"] += stats["seconds_read"]
        tot["fit"] += stats["seconds_fit"]
        tot["write"] += stats["seconds_write"]
        n_cols_done += stats["n_cols"]
        n_elem += stats["n_elements"]
        if i % 5 == 0 or i == len(sched) - 1:
            el = time.perf_counter() - t_walk
            print(f"[leg] block {i + 1}/{len(sched)} q={q} "
                  f"cols[{lo}:{hi}] {dt:.2f} s | walk {el:.1f} s | "
                  f"r/f/w {stats['seconds_read']:.2f}/"
                  f"{stats['seconds_fit']:.2f}/"
                  f"{stats['seconds_write']:.2f}", flush=True)
    t_walk = time.perf_counter() - t_walk

    # Steady state EXCLUDES the first block, which carries the compile.
    n_rest = max(len(sched) - 1, 1)
    steady = (t_walk - (T_FIRST or 0.0)) / n_rest
    print("", flush=True)
    print(f"=== fit_leg {args.tag} COST "
          f"(allocator: BFC@0.85) ===", flush=True)
    print(f"  q window        [{args.qlo},{args.qhi})  "
          f"blocks {len(sched)}", flush=True)
    print(f"  bring-up        {T_JAX - T_ENTRY:.2f} s "
          f"(numpy {T_NUMPY - T_ENTRY:.2f} + lorrax "
          f"{T_LORRAX - T_NUMPY:.2f} + jax {T_JAX - T_LORRAX:.2f})",
          flush=True)
    print(f"  store alloc     {t_alloc:.2f} s", flush=True)
    print(f"  first block     {T_FIRST:.2f} s (compile included)",
          flush=True)
    print(f"  steady block    {steady:.3f} s "
          f"(mean of the remaining {n_rest})", flush=True)
    print(f"  walk total      {t_walk:.1f} s", flush=True)
    print(f"  per stage       read {tot['read']:.1f} s  "
          f"fit {tot['fit']:.1f} s  write {tot['write']:.1f} s",
          flush=True)
    print(f"  columns         {n_cols_done}  elements {n_elem}", flush=True)
    print(f"  THROUGHPUT      {n_cols_done / t_walk:.2f} columns/s  "
          f"= {n_cols_done / t_walk * 3600:.0f} columns/h/GPU", flush=True)
    print(f"  LEG WALL        {time.time() - T_ENTRY:.1f} s "
          f"(python entry to here)", flush=True)
    print("=" * 52, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
