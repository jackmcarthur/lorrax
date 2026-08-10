#!/usr/bin/env python3
"""Fold the per-pole censuses, strike the balance, write the LEG MANIFEST.

Pure stdlib on purpose: it runs on a login node between two farms and has
no reason to want a container, a GPU or numpy.

Usage: balance.py <census_dir> <n_legs> <partial_dir> <manifest.json>
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.environ["WT"], "src"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "window_farm",
    os.path.join(os.environ["WT"], "src", "gw", "mpa", "window_farm.py"))
WF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WF)


def main():
    census_dir, n_legs, partial_dir, out = sys.argv[1:5]
    paths = sorted(glob.glob(os.path.join(census_dir, "*.json")))
    if not paths:
        raise SystemExit(f"balance.py: no census files in {census_dir}")
    census = WF.merge_census_files(paths)
    universe = WF.universe_from_census(census)
    total = sum(int(r["n_tau"]) for r in census["rows"])
    print(f"census: {len(paths)} files, {len(census['rows'])} window groups "
          f"over {len(universe)} (pole, branch) pairs, {total} tau "
          f"dispatches, sha {census['sha'][:12]}")
    by_pole = {}
    for r in census["rows"]:
        by_pole[r["pole"]] = by_pole.get(r["pole"], 0) + int(r["n_tau"])
    for p in sorted(by_pole):
        print(f"  pole {p}: {by_pole[p]} tau dispatches")
    print(f"  pole skew (max/min): "
          f"{max(by_pole.values()) / min(by_pole.values()):.2f}x  "
          f"-- this is what the eight-piece farm could not remove")

    legs = WF.balance_legs(census, int(n_legs))
    taus = [leg["n_tau"] for leg in legs]
    ideal = total / len(legs)
    print(f"balance: {len(legs)} legs, max {max(taus)}, min {min(taus)}, "
          f"ideal {ideal:.0f}, imbalance {max(taus) / ideal:.4f}")
    WF.write_manifest(
        out, legs, kind="pass", fit_store=census["fit_store"],
        n_p=census["n_p"], sha=census["sha"], out_dir=partial_dir,
        census=census,
        extra={"projected_max_leg_tau": max(taus),
               "projected_imbalance": max(taus) / ideal})
    print(f"manifest: {out}")
    for leg in WF.read_manifest(out)["legs"]:
        print(f"  {leg['id']}: {leg['n_tau']:>6d} tau  "
              f"{leg['n_groups']:>4d} groups  poles {leg['poles']}  "
              f"-> {os.path.basename(leg['output'])}")


if __name__ == "__main__":
    main()
