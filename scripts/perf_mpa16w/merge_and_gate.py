#!/usr/bin/env python3
"""Merge the window farm, compare it against the pole farm, run the red twins.

The three gates of this lane, in one step so they share one allocation:

  (a) the window-farmed Σ_c against the pole-farmed Σ_c and, pole by pole,
      against the single-leg computation of that pole;
  (c) the manifest refusal — one leg's output is moved aside and the merge
      must refuse BY NAME, on the pass manifest (a group range), a fit
      manifest (a q range, the [48, 52) that went missing on 2026-08-10),
      and a declared INPUT (a window plan the farm read);
  and the recombination audit the combiner prints either way.

With a reference given, the merge additionally answers the question this
lane exists to answer: whether the farm's arithmetic can tell that its
window groups came off disk instead of out of the planner.  It cannot —
the partition is the same partition and the recombination order is the
same order — so the comparison is against BIT-IDENTITY and not against
the re-association floor §9.6 documents, per leg cube and in total.

Usage: merge_and_gate.py <manifest.json> <pole_partial_dir> <out_dir>
                         [<reference_sigma.npy> [<reference_partial_dir>]]
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import h5py
import numpy as np

from gw.mpa import window_farm as WF
from gw.mpa.sigma_pass import combine_pass_partials


def _read_grid(path):
    with h5py.File(path, "r") as f:
        return (np.asarray(f["omega_grid_ry"][()], dtype=np.float64),
                str(f.attrs["mpa_partial_fit_store"]),
                int(f.attrs["mpa_partial_n_p"]))


def _cmp(name, a, b):
    d = np.abs(a - b)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    mx = float(np.max(d)) if d.size else 0.0
    n_diff = int(np.count_nonzero(d))
    print(f"  {name}: max|Δ| = {mx:.6e} Ry "
          f"({mx / scale if scale else 0.0:.3e} relative), "
          f"{n_diff} of {a.size} elements differ "
          f"({'BIT-IDENTICAL' if n_diff == 0 else 'not bit-identical'})")
    return mx, (mx / scale if scale else 0.0), n_diff


def main():
    man_path, pole_dir, out_dir = sys.argv[1:4]
    ref_npy = sys.argv[4] if len(sys.argv) > 4 else ""
    ref_dir = sys.argv[5] if len(sys.argv) > 5 else ""
    os.makedirs(out_dir, exist_ok=True)
    manifest = WF.read_manifest(man_path)
    win_paths = [leg["output"] for leg in manifest["legs"]]
    pole_paths = sorted(glob.glob(os.path.join(pole_dir, "*.h5")))
    om, fit_src, n_p = _read_grid(win_paths[0])

    print("=" * 72)
    print("GATE (c) RED TWIN 1 -- pass manifest, one leg's output removed")
    victim = manifest["legs"][len(manifest["legs"]) // 2]
    hidden = os.path.join(out_dir, "hidden_" + os.path.basename(victim["output"]))
    shutil.move(victim["output"], hidden)
    try:
        combine_pass_partials(win_paths, n_p=n_p, omega_grid_ry=om,
                              fit_src=fit_src, manifest=manifest)
    except WF.FarmIncomplete as exc:
        print("REFUSED, as it must:")
        print(str(exc))
        assert victim["id"] in str(exc), "the refusal did not name the leg"
        assert victim["range_label"] in str(exc), \
            "the refusal did not name the range"
        print(f"twin verdict: RED as designed -- named leg {victim['id']} "
              f"and its range")
    else:
        raise SystemExit("*** GATE (c) FAILED: the merge summed a subset ***")
    finally:
        shutil.move(hidden, victim["output"])

    print("=" * 72)
    print("GATE (c) RED TWIN 2 -- fit manifest, the 2026-08-10 shape")
    fit_legs = WF.plan_fit_legs(64, 16)
    fit_dir = os.path.join(out_dir, "fit_twin")
    os.makedirs(fit_dir, exist_ok=True)
    for leg in fit_legs:
        with open(os.path.join(fit_dir, f"{leg['id']}.h5"), "wb") as f:
            f.write(b"\0" * 8192)          # a stand-in leg store
    fit_man_path = os.path.join(out_dir, "fit_manifest.json")
    WF.write_manifest(fit_man_path, fit_legs, kind="fit",
                      fit_store="(stand-in)", out_dir=fit_dir)
    fit_man = WF.read_manifest(fit_man_path)
    os.remove(os.path.join(fit_dir, "fit12.h5"))       # leg 12, q [48, 52)
    try:
        WF.refuse_incomplete(fit_man)
    except WF.FarmIncomplete as exc:
        print("REFUSED, as it must:")
        print(str(exc))
        assert "q [48, 52)" in str(exc), "the refusal did not name the q range"
        print("twin verdict: RED as designed -- named leg fit12 and q [48, 52)")
    else:
        raise SystemExit("*** GATE (c) FAILED on the fit side ***")

    if manifest.get("inputs"):
        print("=" * 72)
        print("GATE (c) RED TWIN 3 -- a declared INPUT removed (a window plan)")
        vic = manifest["inputs"][len(manifest["inputs"]) // 2]
        hid = os.path.join(out_dir, "hidden_" + os.path.basename(vic["output"]))
        shutil.move(vic["output"], hid)
        try:
            WF.refuse_incomplete(manifest)
        except WF.FarmIncomplete as exc:
            print("REFUSED, as it must:")
            print(str(exc))
            assert vic["id"] in str(exc), "the refusal did not name the plan"
            assert vic["address"] in str(exc), \
                "the refusal did not name the plan's address"
            print(f"twin verdict: RED as designed -- named {vic['id']} and "
                  f"its address")
        else:
            raise SystemExit(
                "*** GATE (c) FAILED: the merge accepted a farm whose plan "
                "is gone ***")
        finally:
            shutil.move(hid, vic["output"])

    print("=" * 72)
    print("GATE (a) -- the window farm merged under its manifest")
    tot_w, order, audit_w = combine_pass_partials(
        win_paths, n_p=n_p, omega_grid_ry=om, fit_src=fit_src,
        manifest=manifest)
    print("GATE (a) -- the pole farm, same deck, same sha")
    tot_p, _order, audit_p = combine_pass_partials(
        pole_paths, n_p=n_p, omega_grid_ry=om, fit_src=fit_src)

    print("-" * 72)
    print("window-farmed Σ_c against pole-farmed Σ_c:")
    mx, rel, n_diff = _cmp("total", tot_p, tot_w)
    print(f"  for scale, the combiner's own re-association audit on the "
          f"SAME cubes:")
    print(f"    pole farm, ascending vs descending: "
          f"{audit_p['reassoc_descending_max_abs_ry']:.6e} Ry "
          f"({audit_p['reassoc_descending_rel']:.3e} rel)")
    print(f"    pole farm, ascending vs shuffled:   "
          f"{audit_p['reassoc_shuffled_max_abs_ry']:.6e} Ry "
          f"({audit_p['reassoc_shuffled_rel']:.3e} rel)")
    print(f"    window farm, ascending vs shuffled: "
          f"{audit_w['reassoc_shuffled_max_abs_ry']:.6e} Ry "
          f"({audit_w['reassoc_shuffled_rel']:.3e} rel)")

    # Pole by pole, where the partition allows it.  A pole leg's cube IS
    # the single-process computation of that pole, so comparing it against
    # the sum of the window legs covering that pole is the single-leg arm
    # of the gate.  IT IS ONLY A COMPARISON WHERE THE POLE IS COVERED
    # EXCLUSIVELY: a balanced contiguous partition puts leg boundaries
    # inside poles, so most poles are shared with a leg that also carries
    # part of the next one, and summing only the legs that touch nothing
    # else would compare a fraction of a pole against all of it.  That is
    # not a small difference and it is not a defect — it is the wrong
    # question, so it is not asked.
    print("-" * 72)
    print("per pole, where the partition covers a pole exclusively:")
    legs_by_pole, exclusive = {}, {}
    for leg in manifest["legs"]:
        spec = WF.parse_group_subset(leg["group_subset"])
        poles_here = {p for (p, _b) in spec}
        key = min((p, WF.BRANCH_KEYS.index(b), lo)
                  for (p, b), (lo, _hi, _t) in spec.items())
        for p in poles_here:
            legs_by_pole.setdefault(p, []).append((key, leg, len(poles_here)))
    for p, entries in legs_by_pole.items():
        if all(n == 1 for _k, _l, n in entries):
            exclusive[p] = sorted(entries, key=lambda e: e[0])
    worst = 0.0
    for path in pole_paths:
        with h5py.File(path, "r") as f:
            poles = np.asarray(f.attrs["mpa_partial_poles"]).tolist()
            ref = np.asarray(f["sigma_c_partial"][()], dtype=np.complex128)
        if len(poles) != 1 or poles[0] not in exclusive:
            continue
        acc = np.zeros_like(ref)
        for _k, leg, _n in exclusive[poles[0]]:
            with h5py.File(leg["output"], "r") as f:
                acc = acc + np.asarray(f["sigma_c_partial"][()],
                                       dtype=np.complex128)
        _mxp, relp, _ndp = _cmp(
            f"pole {poles[0]} ({len(exclusive[poles[0]])} window legs)",
            ref, acc)
        worst = max(worst, relp)
    shared = sorted(set(legs_by_pole) - set(exclusive))
    print(f"  worst exclusively-covered pole, relative: {worst:.3e}")
    print(f"  poles {shared} are split across legs that also carry a "
          f"neighbouring pole, so they have no single-pole arm; they are "
          f"covered by the total comparison above.")

    np.save(os.path.join(out_dir, "sigma_c_window.npy"), tot_w)
    np.save(os.path.join(out_dir, "sigma_c_pole.npy"), tot_p)

    # THE PLAN-ONCE GATE.  A reference is another farm's cubes over the
    # same manifest partition.  Where the split is identical the two runs
    # differ only in where the window groups came from, so the answer must
    # be exact equality: any nonzero difference is a difference in what a
    # certified rule was, not a re-association, and §9.6's floor is the
    # wrong scale to judge it on.
    if ref_npy:
        print("=" * 72)
        print(f"PLAN-ONCE GATE -- this farm against {ref_npy}")
        ref = np.load(ref_npy)
        if ref.shape != tot_w.shape:
            raise SystemExit(
                f"*** reference Σ_c has shape {ref.shape} against this "
                f"farm's {tot_w.shape}: these are not the same "
                f"calculation ***")
        rmx, rrel, rnd = _cmp("merged Σ_c vs reference", ref, tot_w)
        if rnd != 0:
            print("*** NOT BIT-IDENTICAL.  For scale, the re-association "
                  "floor of this same recombination: "
                  f"{audit_w['reassoc_shuffled_max_abs_ry']:.6e} Ry ***")
        else:
            print("  verdict: BIT-IDENTICAL -- the farm cannot tell that "
                  "its window groups came off disk.")
        del rmx, rrel
    if ref_dir:
        print("-" * 72)
        print(f"per leg cube, against {ref_dir}")
        worst_leg, n_bad = 0.0, 0
        for leg in manifest["legs"]:
            base = os.path.basename(leg["output"])
            other = os.path.join(ref_dir, base)
            if not os.path.exists(other):
                print(f"  {leg['id']}: no reference cube at {other}")
                continue
            with h5py.File(leg["output"], "r") as f:
                mine = np.asarray(f["sigma_c_partial"][()],
                                  dtype=np.complex128)
            with h5py.File(other, "r") as f:
                theirs = np.asarray(f["sigma_c_partial"][()],
                                    dtype=np.complex128)
            mx, _rel, nd = _cmp(f"{leg['id']}", theirs, mine)
            worst_leg = max(worst_leg, mx)
            n_bad += (nd != 0)
        print(f"  {len(manifest['legs']) - n_bad} of "
              f"{len(manifest['legs'])} leg cubes bit-identical; worst "
              f"max|Δ| = {worst_leg:.6e} Ry")
    print("=" * 72)
    print(f"VERDICT total: max|Δ| = {mx:.6e} Ry, relative {rel:.3e}, "
          f"{n_diff} elements differ of {tot_w.size}")
    print(f"VERDICT poles {list(order)} covered exactly once by both arms")


if __name__ == "__main__":
    main()
