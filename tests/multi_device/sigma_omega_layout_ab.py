#!/usr/bin/env python3
"""A/B gate for ``sigma_omega_layout`` — replicated vs sharded Σ_c(ω,k,m,n).

NOT part of the default pytest suite: each arm is a full driver run, and the
sharded layout only differs from the replicated one at P>1 (at P=1 the mesh
has a single shard and both paths hold the whole cube anyway).  Run the two
arms with your launcher on the SAME deck, changing ONLY
``sigma_omega_layout``, then:

    python3 sigma_omega_layout_ab.py compare <replicated_dir> <sharded_dir>
            [--solver one_shot_dft|self_consistent]

``sigma_omega_layout = sharded`` is a MOVEMENT-ONLY change: the per-rank
(m_X, n_Y) host tiles produced by the stacked psum_scatter are published as
one P(None,None,'x','y') jax.Array instead of being gathered into the full
cube on every rank (``ppm_sigma.py`` sigma.host_gather → sigma.tile_finalize),
and five consumers read the tiles at their native sharding.  No arithmetic
changes, so the outputs must agree.  This gate measures by how much.

WHAT IS COMPARED, and at what resolution

  eqp0.dat / eqp1.dat   %15.9f text (eqp_bgw.write_bgw_eqp).  The file
                        carries 1e-9 eV; exact equality of the E_QP column
                        therefore bounds |Δ| < 5e-10 eV and is the STRONGEST
                        statement the file supports.  It is NOT a 1e-12
                        relative measurement — the h5 arrays below are.
  sigma_diag.dat        %12.6f text.  Same caveat, 1e-6 eV.
  sigma_mnk.h5          complex128, full precision.  THE load-bearing
                        comparison: sigma_total/sigma_c carry the cube
                        itself, sigma_sx/hartree carry the (nk,nb,nb)
                        operands the sharded h5 writer mixes with it.
  WFN_qp.h5             float64 QP eigenvalues (Ry) of the converged H.
                        Under SC these come from the eigh of H_qp_dft, i.e.
                        downstream of every cube consumer in the loop.

TOLERANCE.  Gated on ``max|Δ| / max|A|`` (max-norm relative to the array's
own scale) at 1e-12.  Frobenius-relative and the exact-inequality element
count are reported alongside.  Bit-identity is NOT asserted: the sharded
diagonal extractor is a psum onto exact zeros, which is bit-exact for every
x != -0.0 (``qsgw_utils._extract_diag_sharded_kernel`` docstring), so an
off-diagonal exact -0.0 can come back +0.0.  Signed-zero-only differences
are counted separately and never fail the gate.

TIMING EVIDENCE (``--log NAME``, default gw_rank0.out; skipped if absent).
The point of the layout is the elided collective, so the gate checks that it
is elided rather than renamed:
  * ``sigma.host_gather``   MUST appear in the replicated log and MUST NOT
    appear in the sharded one;
  * ``sigma.tile_finalize`` MUST appear in the sharded log only.
Under ``qp_solver = self_consistent`` the gather runs ONCE PER ITERATION, so
its call count in the replicated log is also reported — that count times the
per-call bytes is the whole payoff.

MEASURED, job 7889781, P=4 (2x2 mesh, 2 nodes x 2 ranks), frozen tree
/scratch2/08271/jackmc/omegacube_ab/frozen_24673e6/lorrax (= main 24673e6),
deck MoS2 4x4x1 / 128 bands / nω=41 / nk=16 (cube 171.97 MB/rank),
``qp_solver = one_shot_dft``, restart from the legacy tmp/:

    eqp0.dat, eqp1.dat        max|ΔE_QP| = 0.000e+00 eV over 1280 rows
    sigma_diag.dat            max|Δ| = 0.000e+00 eV, all five columns
    sigma_mnk.h5 (4 datasets) BIT-IDENTICAL — max|Δ| = 0.0, zero elements
                              differ, so the signed-zero exception did not
                              fire on this deck either
    sigma.host_gather         replicated 1 call / 0.306 s; ABSENT sharded
    sigma.tile_finalize       sharded    1 call / 0.009 s; absent replicated
    wall                      42.054 s replicated → 39.685 s sharded
    VmHWM (max over 4 ranks)  5783.3 MB replicated → 5558.8 MB sharded

Bit-identity held here but is NOT asserted (see the tolerance note above).

MEASURED, job 7889782, same deck and mesh, ``qp_solver = self_consistent``,
sc_max_iter=3 / sc_tol_ev=1e-12 / default rcrop accelerator (= 7 Σ
evaluations), frozen tree
/scratch2/08271/jackmc/omegacube_ab/frozen_24673e6_noguard/lorrax, which is
24673e6 with the 12-line SC refusal at gw_config.py:1985-1996 deleted and
nothing else changed:

    eqp0.dat, eqp1.dat        max|ΔE_QP| = 0.000e+00 eV
    sigma_diag.dat            max|Δ| = 0.000e+00 eV, all five columns
    sigma_mnk.h5 (4 datasets) BIT-IDENTICAL
    WFN_qp.h5 el              BIT-IDENTICAL
    qp_wfn_rotations.h5       BIT-IDENTICAL, E_qp AND U_mnk
    SC trajectory             7 iterations both arms; final RMS ΔE
                              1.3130e-02 eV both
    sigma.host_gather         replicated 7 calls / 2.010 s (once per Σ
                              evaluation); ABSENT sharded
    sigma.tile_finalize       sharded 7 calls / 0.057 s
    wall                      232.162 s replicated → 232.636 s sharded
    VmHWM (max over 4 ranks)  6752.6 MB replicated → 6476.2 MB sharded

The five SC-path cube consumers and the branch each takes (read at 24673e6):

  head injection      ppm_pipeline.py:178 branches on
                      is_band_sharded_sigma_omega — sharded takes
                      add_band_diag_sharded (rank-local scatter-add, no dense
                      head), replicated takes the _embed_dense + `+` at :189.
                      Once per Σ evaluation.
  diag(Σ_c) at E_DFT  qsgw_utils.py:337 — sharded takes
                      _extract_diag_sharded_kernel (shard_map + one psum of
                      an (nω,nk,nb) array), replicated takes
                      _extract_diag_kernel (with_sharding_constraint to a
                      replicated 4-D, then the einsum trace).  Called from
                      ppm_pipeline.py:230, once per Σ evaluation.
  sigma_mnk.h5 write  ppm_pipeline.py:267 — sharded derives the eV tensors
                      under out_shardings pinned to the cube's own sharding
                      and hands per-rank tiles to SlabIO; replicated takes
                      the dense call at :301.  Once per SC run, from
                      sc_iteration.py:1297.
  eqp1 Z-factor diag  gw_jax.py:651-653 — the same call as the second row,
                      so the same branch.  Once, after the driver returns.
  QSGW Σ_xc build     qsgw_utils._qsgw_build_kernel:355-389 — NO branch, and
                      none is needed: take_along_axis indexes axis 0 (ω),
                      which is None in both layouts' PartitionSpec, so it is
                      shard-local either way, and the
                      with_sharding_constraint(M, rep_3d) at :383 replicates
                      the (nk,nb,nb) result before the Hermitisation.  That
                      line is where the two layouts converge.  Called from
                      sigma_dispatch.py:445, once per Σ evaluation.

Exit code 0 = every gated quantity passes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# max|Δ| / max|A|.  Movement-only means the only admissible differences are
# IEEE-neutral (psum onto zeros, reassociation-free); 1e-12 is the standing
# repo tolerance for "same answer", not a fitted bound.
TOL_REL = 1.0e-12

# sigma_mnk.h5 datasets written by file_io.sigma_output.write_sigma_omega_h5.
# sigma_total/sigma_c are (nω,nk,nb,nb); sigma_sx/hartree are (nk,nb,nb).
H5_SIGMA_DATASETS = (
    "sigma_total_kij_ev",
    "sigma_c_kij_ev",
    "sigma_sx_kij_ev",
    "hartree_kij_ev",
)

# QP eigenvalues inside WFN_qp.h5 (Ry).  'elda' is absent in some writers'
# output, so the list is probed, not required.
WFN_QP_DATASETS = ("mf_header/kpoints/el", "mf_header/kpoints/elda")

# qp_wfn_rotations.h5 (file_io.qp_wfn.write_qp_rotations_h5): the eigenvalues
# and eigenvectors of the converged H_qp_dft.  E_qp is gated; U_mnk is
# REPORTED ONLY, because an eigenvector is defined up to a phase and, inside a
# degenerate multiplet, up to a unitary mixing — a difference there is not
# evidence of a physics difference unless E_qp also moved.  (This deck is
# heavily degenerate: eqp0.dat shows every level twice.)
QP_ROT_DATASETS = ("E_qp_nk_hartree", "U_mnk")
QP_ROT_REPORT_ONLY = ("U_mnk",)

# "SC done: 7 iterations, final RMS ΔE = 1.3130e-02 eV" (sc_iteration).
_SC_DONE = re.compile(
    r'SC done:\s*(\d+)\s*iterations'
    r'(?:,\s*final RMS\s*\S*E\s*=\s*([-\d.eE+]+)\s*eV)?')


# --------------------------------------------------------------------------
# Text parsers
# --------------------------------------------------------------------------

def parse_eqp(path: Path) -> dict[tuple[int, int], tuple[float, float]]:
    """BGW eqp file → {(ik, band): (E_DFT, E_QP)} in eV.

    Format (``gw.eqp_bgw.write_bgw_eqp``): a 4-token k-header line
    ``kx ky kz nbands`` where the first three are floats, then ``nbands``
    rows ``spin band E_DFT E_QP``.  The k index is positional.
    """
    out: dict[tuple[int, int], tuple[float, float]] = {}
    ik = -1
    for line in path.read_text().splitlines():
        s = line.split()
        if not s or line.lstrip().startswith('#'):
            continue
        if len(s) == 4 and '.' in s[0]:
            ik += 1
            continue
        if len(s) == 4:
            out[(ik, int(s[1]))] = (float(s[2]), float(s[3]))
    if not out:
        sys.exit(f"[layout-ab] no rows parsed from {path}")
    return out


def parse_sigma_diag(path: Path) -> dict[tuple[int, int], dict[str, complex]]:
    """sigma_diag.dat → {(ik, n): {label: value}} in eV.

    Labels are whatever the writer used (sigX/sigC/sigXC for dynamic modes,
    sigSX/sigCOH/sigTOT for static), so they are discovered from the line
    rather than assumed — the gate must not silently compare nothing
    because a deck used the other naming.
    """
    out: dict[tuple[int, int], dict[str, complex]] = {}
    ik = None
    tok = re.compile(r'(\w+)=\s*(-?[\d.]+(?:[eE][+-]?\d+)?)'
                     r'(?:\+\s*(-?[\d.]+(?:[eE][+-]?\d+)?)i)?')
    for line in path.read_text().splitlines():
        m = re.search(r'k-point\s+(\d+)\s*:', line)
        if m:
            ik = int(m.group(1))
            continue
        if ik is None or not line.startswith('n='):
            continue
        rec = {}
        n = None
        for name, re_s, im_s in tok.findall(line):
            val = complex(float(re_s), float(im_s) if im_s else 0.0)
            if name == 'n':
                n = int(val.real)
            else:
                rec[name] = val
        if n is not None and rec:
            out[(ik, n)] = rec
    if not out:
        sys.exit(f"[layout-ab] no rows parsed from {path}")
    return out


_TIMING_ROW = re.compile(r'^\s*([\w.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s')


def parse_timing_rows(log: Path) -> dict[str, tuple[int, float]]:
    """run log → {section name: (call count, total seconds)}.

    Reads the ``--- Timing ---`` table emitted by common.timing.report.
    Only rows whose name is a dotted/underscored identifier are kept.
    """
    rows: dict[str, tuple[int, float]] = {}
    for line in log.read_text(errors="replace").splitlines():
        m = _TIMING_ROW.match(line)
        if m:
            rows[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return rows


# --------------------------------------------------------------------------
# Array comparison
# --------------------------------------------------------------------------

def _norm_zero(a: np.ndarray) -> np.ndarray:
    """Map -0.0 to +0.0 (real and imaginary parts) without touching anything
    else.  ``x + 0.0`` is the identity on every other IEEE value including
    inf and nan payloads."""
    return a + np.zeros((), dtype=a.dtype)


def compare_arrays(name: str, a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Report and gate max|Δ|/max|A|.  Returns (rel, passed)."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE {a.shape} vs {b.shape}  FAIL")
        return (float('inf'), False)
    d = np.abs(a - b)
    scale = float(np.max(np.abs(a)))
    max_abs = float(np.max(d)) if d.size else 0.0
    rel = max_abs / scale if scale > 0 else max_abs
    frob = (float(np.linalg.norm(d.ravel()) / np.linalg.norm(a.ravel()))
            if scale > 0 else 0.0)
    raw_ne = int(np.count_nonzero(a != b))
    zer_ne = int(np.count_nonzero(_norm_zero(a) != _norm_zero(b)))
    ok = rel <= TOL_REL
    tag = ("BIT-IDENTICAL" if raw_ne == 0 else
           f"{raw_ne}/{a.size} elements differ"
           + (f" ({raw_ne - zer_ne} of them signed-zero only)"
              if raw_ne != zer_ne else ""))
    print(f"  {name}: max|Δ|={max_abs:.6e} rel={rel:.3e} frob-rel={frob:.3e} "
          f"scale={scale:.6e}  [{tag}]  (tol {TOL_REL:g}) "
          f"{'PASS' if ok else 'FAIL'}")
    return (rel, ok)


def compare_h5(name: str, pa: Path, pb: Path, datasets, required: bool,
               report_only=()) -> bool:
    import h5py
    if not pa.exists() or not pb.exists():
        miss = pa if not pa.exists() else pb
        print(f"  {name}: MISSING {miss}  "
              f"{'FAIL' if required else '[skipped, not required]'}")
        return not required
    ok = True
    seen = 0
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        for ds in datasets:
            if ds not in fa or ds not in fb:
                if ds in fa or ds in fb:
                    print(f"  {name}[{ds}]: present in ONE arm only  FAIL")
                    ok = False
                continue
            seen += 1
            _, good = compare_arrays(
                f"{name}[{ds}]",
                np.asarray(fa[ds]), np.asarray(fb[ds]))
            if ds in report_only:
                print(f"    ^ {ds}: reported, NOT gated (eigenvector gauge)")
            else:
                ok &= good
    if required and seen == 0:
        print(f"  {name}: none of {datasets} present  FAIL")
        ok = False
    return ok


# --------------------------------------------------------------------------
# Whole-run comparison
# --------------------------------------------------------------------------

def compare_eqp(tag: str, da: Path, db: Path) -> bool:
    """Exact equality of both columns.  See the module docstring on why this
    is equality and not a tolerance: the file resolution (1e-9 eV) is coarser
    than the 1e-12-relative bound the gate asserts on the arrays."""
    pa, pb = da / tag, db / tag
    if not pa.exists() or not pb.exists():
        print(f"  {tag}: MISSING ({pa if not pa.exists() else pb})  FAIL")
        return False
    ea, eb = parse_eqp(pa), parse_eqp(pb)
    if set(ea) != set(eb):
        print(f"  {tag}: (k, band) sets differ — {len(ea)} vs {len(eb)} rows"
              f"  FAIL")
        return False
    keys = sorted(ea)
    d_dft = max(abs(ea[k][0] - eb[k][0]) for k in keys)
    d_qp = max(abs(ea[k][1] - eb[k][1]) for k in keys)
    ok = (d_dft == 0.0 and d_qp == 0.0)
    worst = max(keys, key=lambda k: abs(ea[k][1] - eb[k][1]))
    print(f"  {tag}: {len(keys)} rows  max|ΔE_DFT|={d_dft:.3e} eV  "
          f"max|ΔE_QP|={d_qp:.3e} eV  (file resolution 1e-9 eV) "
          f"{'IDENTICAL' if ok else 'DIFFER'} {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    worst at k={worst[0]} band={worst[1]}: "
              f"{ea[worst][1]:.9f} vs {eb[worst][1]:.9f}")
    return ok


def compare_sigma_diag(name: str, da: Path, db: Path) -> bool:
    pa, pb = da / name, db / name
    if not pa.exists() or not pb.exists():
        print(f"  {name}: MISSING  [skipped]")
        return True
    sa, sb = parse_sigma_diag(pa), parse_sigma_diag(pb)
    keys = sorted(set(sa) & set(sb))
    if not keys:
        print(f"  {name}: no shared (k, n) rows  FAIL")
        return False
    labels = sorted(set(sa[keys[0]]) & set(sb[keys[0]]))
    ok = True
    for lab in labels:
        d = max(abs(sa[k][lab] - sb[k][lab]) for k in keys)
        ok &= (d == 0.0)
        print(f"  {name}[{lab}]: {len(keys)} rows max|Δ|={d:.3e} eV "
              f"(file resolution 1e-6 eV) {'PASS' if d == 0.0 else 'FAIL'}")
    return ok


def compare_timing(da: Path, db: Path, log_name: str) -> bool:
    """The collective must be ELIDED, not renamed."""
    la, lb = da / log_name, db / log_name
    if not la.exists() or not lb.exists():
        print(f"  timing: {log_name} absent in one arm  [skipped]")
        return True
    ta, tb = parse_timing_rows(la), parse_timing_rows(lb)
    ok = True
    gath_a, gath_b = ta.get("sigma.host_gather"), tb.get("sigma.host_gather")
    fin_a, fin_b = ta.get("sigma.tile_finalize"), tb.get("sigma.tile_finalize")
    if gath_a is None:
        print("  timing: sigma.host_gather ABSENT from the replicated arm — "
              "the reference arm did not take the gather path  FAIL")
        ok = False
    else:
        print(f"  timing: replicated sigma.host_gather  {gath_a[0]} calls, "
              f"{gath_a[1]:.3f} s  (one per Σ stage = one per SC iteration)")
    if gath_b is not None:
        print(f"  timing: sharded arm STILL RAN sigma.host_gather "
              f"({gath_b[0]} calls, {gath_b[1]:.3f} s)  FAIL")
        ok = False
    else:
        print("  timing: sharded sigma.host_gather absent  PASS")
    if fin_b is None:
        print("  timing: sharded sigma.tile_finalize ABSENT — the sharded arm "
              "did not take the tile path  FAIL")
        ok = False
    else:
        print(f"  timing: sharded sigma.tile_finalize  {fin_b[0]} calls, "
              f"{fin_b[1]:.3f} s")
    if fin_a is not None:
        print(f"  timing: replicated arm ran sigma.tile_finalize "
              f"({fin_a[0]} calls)  FAIL")
        ok = False
    return ok


def compare_sc(da: Path, db: Path, log_name: str) -> bool:
    """SC-only checks.

    The guard this gate exists to test claims SC is special because the driver
    "captures Σ_c(ω) across iterations".  So the SC arm must show (i) that the
    two layouts walked the SAME trajectory, not just landed on the same point —
    a different number of iterations with the same endpoint would hide a
    per-iteration difference; and (ii) that the replicated arm really did pay
    the full-cube gather once per Σ evaluation, which is what makes SC the
    workload with the most to gain.
    """
    la, lb = da / log_name, db / log_name
    if not la.exists() or not lb.exists():
        print(f"  SC: {log_name} absent in one arm  FAIL")
        return False
    ma = _SC_DONE.search(la.read_text(errors="replace"))
    mb = _SC_DONE.search(lb.read_text(errors="replace"))
    if ma is None or mb is None:
        print("  SC: no 'SC done:' line — the arms did not run the SC driver"
              "  FAIL")
        return False
    na, nb = int(ma.group(1)), int(mb.group(1))
    ok = (na == nb)
    print(f"  SC iterations: replicated={na} sharded={nb} "
          f"{'PASS' if ok else 'FAIL'}")
    if ma.group(2) and mb.group(2):
        ra, rb = float(ma.group(2)), float(mb.group(2))
        same = (ra == rb)
        print(f"  SC final RMS ΔE: {ra:.4e} vs {rb:.4e} eV  |Δ|={abs(ra - rb):.3e}"
              f"  {'IDENTICAL' if same else 'DIFFER'} {'PASS' if same else 'FAIL'}")
        ok &= same
    # The payoff claim: the gather is per Σ evaluation, not per run.
    ta = parse_timing_rows(la)
    g = ta.get("sigma.host_gather")
    if g is None:
        print("  SC: replicated arm has no sigma.host_gather row  FAIL")
        ok = False
    elif g[0] < 2:
        print(f"  SC: replicated sigma.host_gather ran {g[0]}x — the "
              f"per-iteration cost is not exercised by this deck  FAIL")
        ok = False
    else:
        print(f"  SC: replicated sigma.host_gather ran {g[0]}x "
              f"(once per Σ evaluation) — {g[1]:.3f} s total  PASS")
    return ok


def compare_run(da: Path, db: Path, *, log_name: str,
                sigma_diag_name: str, solver: str) -> bool:
    sc = (solver == "self_consistent")
    print(f"\n===== sigma_omega_layout A/B  (qp_solver = {solver}) =====")
    print(f"  replicated: {da}")
    print(f"  sharded   : {db}")
    ok = True
    for tag in ("eqp0.dat", "eqp1.dat"):
        ok &= compare_eqp(tag, da, db)
    ok &= compare_sigma_diag(sigma_diag_name, da, db)
    ok &= compare_h5("sigma_mnk.h5", da / "sigma_mnk.h5", db / "sigma_mnk.h5",
                     H5_SIGMA_DATASETS, required=True)
    # SC always dumps the QP artifacts (sc_iteration.dump_qp_wfn_artifacts);
    # the one-shot dump refuses when Σ is on the full BZ and the WFN carries
    # the IBZ, so they are optional there.
    ok &= compare_h5("WFN_qp.h5", da / "WFN_qp.h5", db / "WFN_qp.h5",
                     WFN_QP_DATASETS, required=sc)
    ok &= compare_h5("qp_wfn_rotations.h5",
                     da / "qp_wfn_rotations.h5", db / "qp_wfn_rotations.h5",
                     QP_ROT_DATASETS, required=sc,
                     report_only=QP_ROT_REPORT_ONLY)
    if sc:
        ok &= compare_sc(da, db, log_name)
    ok &= compare_timing(da, db, log_name)
    print(f"===== {'PASS' if ok else 'FAIL'} =====")
    return ok


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "compare":
        sys.exit(__doc__)
    log_name = "gw_rank0.out"
    sigma_diag_name = "sigma_diag.dat"
    solver = "one_shot_dft"
    pos = []
    i = 1
    while i < len(argv):
        if argv[i] == "--log":
            log_name = argv[i + 1]; i += 2
        elif argv[i] == "--sigma-diag":
            sigma_diag_name = argv[i + 1]; i += 2
        elif argv[i] == "--solver":
            solver = argv[i + 1]; i += 2
        else:
            pos.append(argv[i]); i += 1
    if len(pos) != 2 or solver not in ("one_shot_dft", "self_consistent"):
        sys.exit(__doc__)
    ok = compare_run(Path(pos[0]), Path(pos[1]),
                     log_name=log_name, sigma_diag_name=sigma_diag_name,
                     solver=solver)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
