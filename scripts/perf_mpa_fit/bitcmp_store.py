#!/usr/bin/env python
"""Compare two MPA fit stores BYTE FOR BYTE over a q window.

THE GATE THIS SERVES.  The fit store is read by another lane, so a
performance change to the fit is allowed to make the file faster and not
to make it different.  "Different" is decided here on the uint8 view of
every dataset the store carries — poles, residues and all four
diagnostics — and not on a tolerance, because a tolerance would pass a
restructure that changed the arithmetic as happily as one that did not.

Neither store needs jax and neither is opened for writing; this runs on a
login node or a CPU allocation, one q at a time so the peak is about a
third of a gigabyte rather than the forty the pair of stores weighs.

Usage:
  bitcmp_store.py <A.h5> <B.h5> [--q-lo 0] [--q-hi 64] [--group /]
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

#: Everything the fit store holds per element.  ``fit_residual`` and
#: ``fit_n_valid`` are created on first write rather than at allocation,
#: so a store may legitimately lack them only if no block was ever
#: written — which is itself a failure, and is reported as one.
DATASETS = ("Omega_p", "B_p", "fit_condition", "fit_backward_error",
            "fit_residual", "fit_n_valid")


def q_slice(dset, q):
    """The q-th slab of a store dataset, whichever axis carries q.

    ``Omega_p`` and ``B_p`` are ``(n_p, n_q, N_mu, N_mu)``; the
    diagnostics are ``(n_q, N_mu, N_mu)``.  The q axis is the one whose
    length is the store's ``n_q``, and it is axis 1 for the first pair
    and axis 0 for the second — read off the rank rather than guessed.
    """
    return dset[:, q] if dset.ndim == 4 else dset[q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--q-lo", type=int, default=0)
    ap.add_argument("--q-hi", type=int, default=0)
    ap.add_argument("--group", default="/")
    args = ap.parse_args()

    fa = h5py.File(args.a, "r")
    fb = h5py.File(args.b, "r")
    ga = fa[args.group]
    gb = fb[args.group]

    print(f"A {args.a}")
    print(f"B {args.b}")
    for label, g in (("A", ga), ("B", gb)):
        prov = {k: g.attrs[k] for k in sorted(g.attrs)
                if "prov" in k.lower() or "content" in k.lower()
                or "hash" in k.lower() or "unit" in k.lower()}
        print(f"  [{label}] attrs {prov}")

    names = [n for n in DATASETS if n in ga and n in gb]
    missing = [n for n in DATASETS if n not in ga or n not in gb]
    if missing:
        print(f"  NOTE datasets absent from one or both stores: {missing}")
    if not names:
        print("  REFUSED: the two stores share no comparable dataset.")
        return 2

    n_q = ga[names[0]].shape[1 if ga[names[0]].ndim == 4 else 0]
    q_hi = args.q_hi or n_q
    print(f"  comparing {names} over q [{args.q_lo}, {q_hi}) of {n_q}")

    total_bytes = 0
    total_diff = 0
    worst = {}
    for q in range(args.q_lo, q_hi):
        for name in names:
            a = np.ascontiguousarray(q_slice(ga[name], q))
            b = np.ascontiguousarray(q_slice(gb[name], q))
            if a.shape != b.shape or a.dtype != b.dtype:
                print(f"  q={q} {name}: SHAPE/DTYPE MISMATCH "
                      f"{a.shape}{a.dtype} vs {b.shape}{b.dtype}")
                return 2
            va = a.view(np.uint8)
            vb = b.view(np.uint8)
            n_bytes = va.size
            n_diff = int(np.count_nonzero(va != vb))
            total_bytes += n_bytes
            total_diff += n_diff
            if n_diff:
                d = np.abs(a - b)
                fin = np.isfinite(d)
                mx = float(np.max(d[fin])) if np.any(fin) else float("nan")
                ref = float(np.max(np.abs(b[fin]))) if np.any(fin) else 1.0
                prev = worst.get(name, (0, 0.0, 0.0))
                worst[name] = (prev[0] + n_diff, max(prev[1], mx), ref)
        if (q - args.q_lo) % 8 == 0:
            print(f"  ... q={q} cumulative {total_diff} differing bytes "
                  f"of {total_bytes}", flush=True)

    print("")
    if total_diff == 0:
        print(f"  VERDICT: BYTE-IDENTICAL — 0 of {total_bytes} bytes "
              f"differ across {len(names)} datasets and "
              f"{q_hi - args.q_lo} q")
        return 0
    print(f"  VERDICT: DIFFERS — {total_diff} of {total_bytes} bytes")
    for name, (n, mx, ref) in sorted(worst.items()):
        print(f"    {name:22s} {n} bytes, max|d| {mx:.6e} against "
              f"max|ref| {ref:.6e} (rel {mx / ref if ref else np.nan:.3e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
