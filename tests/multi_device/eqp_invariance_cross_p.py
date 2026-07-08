#!/usr/bin/env python3
"""Tier-2 device-invariance gate: cross-P comparison (1 GPU vs 4 GPU) — COMPARE step.

NOT part of the default pytest suite (the legs need multiple GPUs, and LORRAX
runs one process per device — a single process cannot host the P=4 leg).
Drive the whole gate with the sibling ``run_tier2.sh`` (Perlmutter / lxrun),
or run the legs yourself with your launcher and then:

    python3 eqp_invariance_cross_p.py compare <case> <p1_run_dir> <p4_run_dir>

with ``<case>`` = ``gnppm`` | ``bispinor``.  Each run dir must contain the
fixture outputs (sigma_diag file, eqp0/eqp1.dat, run.log, tmp/zeta_q*.h5).

Rationale + tolerances (reports/device_invariance_2026-07-08/ROOT_CAUSE.md;
each bound justified by measurement):

  charge (gnppm fixture)
    ζ_C                    frob-rel ≤ 1e-6   (measured cross-P floor 2–9e-7)
    Σ_X diag               ≤ 1e-5 eV         (measured ≤1e-6 eV across P)
    minimax node counts    exactly equal     (integers; census-fix acceptance)
    logical invalid count  exactly equal
    eqp0/eqp1, OFF-pole    ≤ 1 meV           (bands with |Im Σ_C| < 100 eV)
    eqp0/eqp1, ON-pole     reported, NOT gated (Σ_C(E_dft) on a GN-PPM pole is
                           ill-posed; ~1e-7 W noise moves it O(0.1 eV))

  bispinor fixture
    ζ charge               frob-rel ≤ 1e-6
    ζ transverse           PROVISIONAL loose 1e-3 (near-null modes are
                           solver-noise-dominated at fixed shape until the
                           rank-revealing CCT treatment lands — Fix 3)
    eqp0 (all bands)       ≤ 50 meV PROVISIONAL (Σ^B-carried; measured
                           post-fix 4g↔16g = 1.45e-7 eV, so this is
                           generous; tighten after Fix 3)

The fixture centroid counts (642 charge / 640+668 bispinor) are deliberately
NOT divisible by 4, so the P=4 leg exercises a different pad topology than
P=1 — this gate covers BOTH the pad-extent fixes (sharp version: the fixed-P
Tier-1 gate ``tests/test_mu_pad_invariance.py``) and the irreducible ~1e-7
cross-P reduction-order floor.

Exit code 0 = all gated quantities pass.  Non-gated quantities are printed
either way.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

CASES = {
    "gnppm": dict(
        sigma_out="sigma_diag_gnppm_test.dat",
        labels=("sigX", "sigC", "sigXC"),
        zeta_files=("tmp/zeta_q.h5",),
        charge=True,
    ),
    "bispinor": dict(
        sigma_out="sigma_diag_bispinor_test.dat",
        labels=("sigSX", "sigCOH", "sigTOT"),
        zeta_files=("tmp/zeta_q.h5", "tmp/zeta_q_mu1.h5",
                    "tmp/zeta_q_mu2.h5", "tmp/zeta_q_mu3.h5"),
        charge=False,
    ),
}

TOL_ZETA_C = 1.0e-6
TOL_ZETA_T_PROVISIONAL = 1.0e-3
TOL_SIGX_EV = 1.0e-5
TOL_EQP_OFFPOLE_EV = 1.0e-3
TOL_BISPINOR_EV = 0.05
ONPOLE_IM_EV = 100.0


# --------------------------------------------------------------------------
# Parsers (sigma_diag / eqp / run.log census) — mirror
# lorrax_sandbox skills/compare conventions.
# --------------------------------------------------------------------------

def parse_sigma_diag(path: Path, labels) -> dict:
    a, b, c = labels
    out, ik = {}, None
    cplx = (r'=\s*(-?[\d.]+(?:[eE][+-]?\d+)?)'
            r'(?:\+\s*(-?[\d.]+(?:[eE][+-]?\d+)?)i)?')
    for line in path.read_text().splitlines():
        m = re.search(r'k-point\s+(\d+)\s*:', line)
        if m:
            ik = int(m.group(1)); continue
        m = re.match(r'\s*n=\s*(\d+)\s', line)
        if not (m and ik is not None):
            continue
        n = int(m.group(1))
        rec = {}
        for name in (a, b, c):
            mm = re.search(re.escape(name) + cplx, line)
            if mm is None:
                rec = None
                break
            rec[name] = complex(float(mm.group(1)),
                                float(mm.group(2)) if mm.group(2) else 0.0)
        if rec is not None:
            out[(ik, n)] = rec
    if not out:
        sys.exit(f"[tier2] no rows parsed from {path}")
    return out


def parse_eqp(path: Path) -> dict:
    out, ik = {}, -1
    for line in path.read_text().splitlines():
        s = line.split()
        if not s or line.startswith('#'):
            continue
        if len(s) == 4 and '.' in s[0]:
            ik += 1; continue
        if len(s) == 4:
            out[(ik, int(s[1]) - 1)] = float(s[3])
    return out


def parse_census(log: Path) -> dict:
    """Node counts per window line + GN invalid census from run.log."""
    text = log.read_text()
    nodes = re.findall(
        r'window "(\w+)" \((?:crossing|Laplace)\): (\d+) nodes', text)
    m = re.search(r'GN invalid modes: (\d+)/(\d+)', text)
    return {
        "windows": tuple(nodes),
        "invalid": (int(m.group(1)), int(m.group(2))) if m else None,
    }


def frob_rel(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a.ravel())
    return float(np.linalg.norm((a - b).ravel()) / max(na, 1e-300))


def compare_zeta(dir1: Path, dir4: Path, rel_path: str) -> float:
    import h5py
    worst = 0.0
    with h5py.File(dir1 / rel_path, "r") as f1, \
         h5py.File(dir4 / rel_path, "r") as f4:
        keys = [k for k in f1.keys() if k in f4
                and getattr(f1[k], "ndim", 0) >= 2]
        for k in keys:
            a, b = np.asarray(f1[k]), np.asarray(f4[k])
            if a.shape != b.shape:
                sys.exit(f"[tier2] {rel_path}:{k} shape {a.shape} vs {b.shape}"
                         " — on-disk extents must be logical (P-independent)")
            worst = max(worst, frob_rel(a, b))
    return worst


def compare_case(name: str, d1: Path, d4: Path) -> bool:
    case = CASES[name]
    print(f"\n===== Tier-2 cross-P compare: {name} =====")
    print(f"  P=1: {d1}\n  P=4: {d4}")
    ok = True

    for zf in case["zeta_files"]:
        rel = compare_zeta(d1, d4, zf)
        transverse = "mu" in zf
        tol = TOL_ZETA_T_PROVISIONAL if transverse else TOL_ZETA_C
        status = "PASS" if rel <= tol else "FAIL"
        gate = "provisional" if transverse else "gate"
        print(f"  zeta {zf}: frob-rel {rel:.3e}  ({gate} tol {tol:g}) {status}")
        ok &= (rel <= tol)

    s1 = parse_sigma_diag(d1 / case["sigma_out"], case["labels"])
    s4 = parse_sigma_diag(d4 / case["sigma_out"], case["labels"])
    keys = sorted(set(s1) & set(s4))
    lx = case["labels"][0]
    d_sigx = np.array([abs(s4[k][lx].real - s1[k][lx].real) for k in keys])
    print(f"  Σ_{lx}: max|Δ| = {d_sigx.max():.3e} eV  (tol {TOL_SIGX_EV:g}) "
          f"{'PASS' if d_sigx.max() <= TOL_SIGX_EV else 'FAIL'}")
    ok &= (d_sigx.max() <= TOL_SIGX_EV)

    if case["charge"]:
        c1, c4 = parse_census(d1 / "run.log"), parse_census(d4 / "run.log")
        eq_w = c1["windows"] == c4["windows"]
        eq_i = c1["invalid"] == c4["invalid"]
        print(f"  minimax windows: P1={c1['windows']}")
        print(f"                   P4={c4['windows']}  "
              f"{'PASS' if eq_w else 'FAIL'}")
        print(f"  invalid census:  P1={c1['invalid']} P4={c4['invalid']}  "
              f"{'PASS' if eq_i else 'FAIL'}")
        ok &= eq_w and eq_i

        # eqp gated OFF-pole (|Im Σ_C| < 100 eV at P=1)
        lc = case["labels"][1]
        onpole = {k for k in keys if abs(s1[k][lc].imag) >= ONPOLE_IM_EV}
        for eqp_name in ("eqp0.dat", "eqp1.dat"):
            e1, e4 = parse_eqp(d1 / eqp_name), parse_eqp(d4 / eqp_name)
            ek = sorted(set(e1) & set(e4))
            off = [k for k in ek if k not in onpole]
            on = [k for k in ek if k in onpole]
            d_off = max((abs(e4[k] - e1[k]) for k in off), default=0.0)
            d_on = max((abs(e4[k] - e1[k]) for k in on), default=0.0)
            st = "PASS" if d_off <= TOL_EQP_OFFPOLE_EV else "FAIL"
            print(f"  {eqp_name}: off-pole ({len(off)} bands) max|Δ| = "
                  f"{d_off:.3e} eV (tol {TOL_EQP_OFFPOLE_EV:g}) {st}; "
                  f"on-pole ({len(on)} bands) max|Δ| = {d_on:.3e} eV "
                  f"[reported, not gated]")
            ok &= (d_off <= TOL_EQP_OFFPOLE_EV)
    else:
        # bispinor: Σ^B rides in eqp0 via sig_x; loose provisional bound.
        e1, e4 = parse_eqp(d1 / "eqp0.dat"), parse_eqp(d4 / "eqp0.dat")
        ek = sorted(set(e1) & set(e4))
        d_all = max((abs(e4[k] - e1[k]) for k in ek), default=0.0)
        st = "PASS" if d_all <= TOL_BISPINOR_EV else "FAIL"
        print(f"  eqp0 (all bands, Σ^B-carried): max|Δ| = {d_all:.3e} eV "
              f"(PROVISIONAL tol {TOL_BISPINOR_EV:g}) {st}")
        ok &= (d_all <= TOL_BISPINOR_EV)

    print(f"===== {name}: {'PASS' if ok else 'FAIL'} =====")
    return ok


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[1] != "compare":
        sys.exit(__doc__)
    name = sys.argv[2]
    if name not in CASES:
        sys.exit(f"unknown case {name!r}; choose from {list(CASES)}")
    ok = compare_case(name, Path(sys.argv[3]), Path(sys.argv[4]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
