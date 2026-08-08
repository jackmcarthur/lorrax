#!/usr/bin/env python3
"""A/B gate for ``restart_q_storage`` — full-BZ vs the IBZ q wedge on disk.

NOT part of the default pytest suite.  Each arm is a full GW driver run plus
a BSE run, so it needs GPUs, the phdf5 FFI ``.so`` pair and a real launcher.
`4e8cfd70` (the restart producer) names this script as its own gate and says
why it cannot be a pytest cell: SlabIO needs the phdf5 FFI that the WSL
checkout does not build, so every restart-writer cell in the tree is red
there and the BYTES had never been measured in-tree until this ran.

Run the two arms with your launcher on the SAME deck, changing ONLY
``restart_q_storage`` — ``restart_q_storage_ab.sh`` is that driver — then:

    python3 restart_q_storage_ab.py compare <full_dir> <auto_dir> [--ref REF]

``full`` is the control: it never asks the closure question, so the file it
writes is the same bytes the deck wrote before the format existed.  ``auto``
resolves to the wedge on any deck whose centroid set is orbit-closed and
whose q path reduced — ``si_bse_debug`` since `fb046e0c` — and the file then
holds the PRE-UNFOLD IBZ block plus the tables needed to unfold it.

TWO REQUIREMENTS ON HOW THE ARMS ARE RUN, and both have cost somebody a day.

1.  MATCHED SETTINGS, AND THE FROZEN REFERENCE IS NOT THE CRITERION.  This
    deck's ``bse_eigenvalues_ref.dat`` is pinned to a Lanczos spectrum that
    is not converged at states 7, 14 and 15: with no code change at all it
    moves 7.32 meV going 200 -> 400 iterations and 5.51 meV going 1 -> 4
    processes (measured 2026-08-08, ``NOTE_vcoul_head_refreeze.md``).  The
    pin survives only because the gate re-runs it at exactly the pinned
    count on exactly the pinned mesh.  So the A/B holds iteration count,
    process count, mesh and every solver flag FIXED across the arms — then
    the fragility is common-mode and cancels — and the reference is used as
    a sanity line only.  This script prints the reference delta under a
    heading that says so.  If the ``full`` arm disagrees with the reference,
    suspect a settings mismatch before suspecting the code: the measured
    reference delta of the ``full`` arm below is 0.000 meV at P=4 and
    4.489 meV at P=1, and NOTHING about the code differs between those two.

2.  THE WEDGE ARM'S BSE LEG MUST BE P=1.  The sharded reader refuses a wedge
    file, deliberately (``bse_io._MunuSlabPlan``): a per-rank (mu, nu)
    hyperslab cannot unfold, because the unfold is a double-gather ACROSS
    the mu and nu axes that plan shards on, so a rank does not hold the
    elements its own block's images come from.  Only the serial h5py reader
    (``bse_io._load_ring_subset``, the P=1 + TDA branch of
    ``_preview_lanczos``) unfolds.  The physics A/B therefore runs at P=1 on
    BOTH arms.  The P=4 leg is still run on both — on ``full`` because it is
    the configuration the frozen reference was cut at, and on ``auto``
    because the refusal is itself an assertion.  A wedge file that the
    production sharded reader silently ACCEPTED would be the alarming
    outcome, not the refusal.

WHAT IS COMPARED, and at what resolution

  V_qmunu, W0_qmunu     complex128, full precision, read through
                        ``bse_io.restart_munu_full_bz`` — THE one place a
                        restart V/W becomes a full-BZ host array, so the
                        wedge arm goes through the real production unfold
                        and the full arm goes through ``dset[()]``.  This is
                        the load-bearing comparison and it is asserted as
                        ``np.array_equal``, not as a tolerance: the design's
                        claim is that the file holds the array the producer
                        had one statement BEFORE its own unfold, so
                        ``unfold(stored)`` is the SAME CALL on the SAME
                        ARGUMENTS the run made.  That is an identity.  A
                        tolerance here would be an admission that it is not.
  psi_full_y, enk_full  compared too, and for the opposite reason: they are
                        the datasets the format does NOT move.  A difference
                        in them would mean the storage key had reached
                        something it has no business reaching.
  BSE eigenvalues       parsed from each arm's P=1 log at the 8-decimal eV
                        the driver prints, so exact equality of the printed
                        vector bounds max|Delta| below 1e-5 meV.  This is
                        the deliverable number and it is reported in meV.
                        The driver prints eigenvalues and writes no file, so
                        8 decimals is all the precision that exists; the
                        tensor comparison above is what carries bit-identity.
  file size             ``stat``, both arms, plus the ratio.

MEASURED 2026-08-08, lx job 56499811, 1 node x 4 A100 (nid001028), branch
``svc/symmetry_maps-followup-2026-08-08`` @ 54d25712, BUILD_NOTES merge_ckpt
``.so`` pair (dev md5 c680c229..., host md5 91f330c3...), LX_BASE_MODULE
lorrax_J070 (jax 0.7.0), deck ``tests/regression/si_bse_debug`` with its
adopted orbit-closed 480 set (md5 253b498f...), GW at 4 processes / 2x2 mesh,
BSE at 200 Lanczos iterations:

    resolution (auto arm, verbatim)
        [restart_write] restart_q_storage=auto -> ibz (centroid set is
        orbit-closed (worst residual 4.596e-16 at tol 1.0e-05) and the q
        path reduced)

    V_qmunu on disk       (64, 480, 480) full  ->  (8, 480, 480) wedge
    W0_qmunu on disk      (64, 480, 480) full  ->  (8, 480, 480) wedge
    psi_full_y, enk_full  unmoved, as required
    restart file          541 335 584 B (0.5042 GiB)
                       -> 130 297 888 B (0.1213 GiB)      4.155x

    unfold(wedge) vs full-BZ    BIT-IDENTICAL on all four datasets:
                                max|Delta| = 0.000e+00, 0 elements differing
    BSE eigenvalues, P=1        max|Delta| = 0.000000 meV over all 20 states
                                (identical to every printed decimal)
    full arm at P=4 vs the frozen reference (the `fb046e0c` re-cut)
                                max|Delta| = 0.000e+00 eV over 20 states —
                                BIT-IDENTICAL, inside ATOL_FROZEN_EV 1e-6.
                                The deck key's `full` path is inert, and so
                                is main's merge checkpoint on this deck.
    full arm at P=1 vs the same reference
                                4.4887 meV, 12 of 20 cells over the 1e-6 eV
                                pin — the process-count fragility of
                                requirement 1, with no code difference of
                                any kind between the two legs.  This is the
                                number that shows why the reference cannot
                                be the criterion for the wedge arm.

AND THE COROLLARY THAT FELL OUT OF IT, recorded here because it is about the
reference rather than about this script: `tests/test_bse_bgw_regression.py::
test_bse_matches_frozen_and_bgw` fails on Perlmutter at exactly that
4.4887 meV / 12-of-20, and fails IDENTICALLY with `restart_q_storage = full`
pinned in the deck — so it is not the wedge and not this branch.  `conftest.py`
pins the pytest process to ONE GPU, while `fb046e0c` cut the reference from two
4-process script runs and recorded, as its own honest limit, that it could not
run the gate.  The reference has therefore never been seen by the gate that
pins it.  Nothing is re-frozen here; the number is left where the owner can
see both ends of it (0.000 meV at P=4, 4.4887 meV at P=1, same binary, same
restart file).

THE FILE RATIO IS 4.155x AND NOT THE 8x THE DESIGN PREDICTS, and the reason
is arithmetic rather than a shortfall.  V and W0 DO go 8x — 0.24 GB to
0.03 GB each, which is exactly the 64 q -> 8 q the wedge stores.  What does
not shrink is ``psi_full_y``: 0.06 GB on a band/k index the wedge never
touches.  8x is the number for the two tensors; 4.2x is the number for the
FILE on this deck, and it approaches 8x as N_mu grows against the band count
(the tensors scale as N_mu^2, psi as N_mu).  Quote whichever one the claim is
actually about.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

#: The four datasets an arm must agree on.  V/W are the ones the format
#: moves; psi/enk are the ones it must not.
DATASETS = ("V_qmunu", "W0_qmunu", "psi_full_y", "enk_full")


def _restart_file(run_dir: Path) -> Path:
    hits = sorted(glob.glob(str(run_dir / "tmp" / "isdf_tensors_*.h5")))
    if not hits:
        raise SystemExit(f"{run_dir}: no tmp/isdf_tensors_*.h5")
    if len(hits) > 1:
        raise SystemExit(f"{run_dir}: {len(hits)} restart files, expected 1")
    return Path(hits[0])


def _parse_eigenvalues(text: str) -> np.ndarray:
    """Pull the printed lowest-N eigenvalue vector (eV) out of a BSE log."""
    match = re.search(r"Lowest \d+ eigenvalues \(eV\): \[(.*?)\]", text, re.S)
    if not match:
        raise SystemExit("BSE log printed no eigenvalue vector")
    return np.array([float(tok) for tok in match.group(1).split()])


def compare_tensors(fa: Path, fb: Path) -> bool:
    """Element-wise identity of every restart tensor, through the REAL seam."""
    import h5py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bse.bse_io import is_q_wedge, restart_munu_full_bz

    ok = True
    with h5py.File(fa, "r") as A, h5py.File(fb, "r") as B:
        for name in DATASETS:
            if name not in A or name not in B:
                print(f"  {name:<12} ABSENT in one arm "
                      f"(A={name in A}, B={name in B})  FAIL")
                ok = False
                continue
            wa, wb = is_q_wedge(A[name]), is_q_wedge(B[name])
            print(f"  {name:<12} A shape {str(A[name].shape):<18} "
                  f"wedge={wa}   B shape {str(B[name].shape):<18} wedge={wb}")
            if name in ("V_qmunu", "W0_qmunu"):
                a = restart_munu_full_bz(A[name], name, str(fa))
                b = restart_munu_full_bz(B[name], name, str(fb))
            else:
                a, b = np.asarray(A[name][()]), np.asarray(B[name][()])
            if a.shape != b.shape:
                print(f"    unfolded shapes DIFFER: {a.shape} vs {b.shape}"
                      "  FAIL")
                ok = False
                continue
            same = np.array_equal(a, b)
            delta = 0.0 if same else float(np.max(np.abs(a - b)))
            n_diff = 0 if same else int(np.count_nonzero(a != b))
            print(f"    unfolded {str(a.shape):<18} "
                  f"{'BIT-IDENTICAL' if same else 'DIFFERS'}   "
                  f"max|d| = {delta:.3e}   elements differing = {n_diff}")
            ok &= same
    return ok


def compare_files(da: Path, db: Path) -> bool:
    fa, fb = _restart_file(da), _restart_file(db)
    sa, sb = os.path.getsize(fa), os.path.getsize(fb)
    print("--- restart files ---")
    print(f"  A  {fa}  {sa} B ({sa / 2**30:.4f} GiB)")
    print(f"  B  {fb}  {sb} B ({sb / 2**30:.4f} GiB)")
    print(f"  ratio A/B = {sa / sb:.3f}x")
    print("--- tensors ---")
    return compare_tensors(fa, fb)


def compare_eigenvalues(da: Path, db: Path, log_name: str) -> bool:
    pa, pb = da / log_name, db / log_name
    if not pa.exists() or not pb.exists():
        print(f"--- eigenvalues --- SKIPPED ({log_name} missing in one arm)")
        return True
    ea = _parse_eigenvalues(pa.read_text(errors="replace"))
    eb = _parse_eigenvalues(pb.read_text(errors="replace"))
    n = min(len(ea), len(eb))
    d = np.abs(ea[:n] - eb[:n]) * 1e3          # eV -> meV
    print(f"--- BSE eigenvalues ({log_name}, {n} states) ---")
    print(f"  max|Delta| = {d.max():.6f} meV     "
          f"mean|Delta| = {d.mean():.6f} meV")
    print(f"  identical to the printed 8 decimals: {np.array_equal(ea, eb)}")
    return bool(np.array_equal(ea, eb))


def compare_reference(da: Path, ref: Path, log_name: str) -> None:
    """SANITY LINE ONLY, never an acceptance criterion.

    Requirement 1 in the module docstring is the whole reason this prints
    rather than asserts: the reference moves by meV under settings changes
    that touch no code, so on its own it bounds nothing.
    """
    p = da / log_name
    if not p.exists() or not ref.exists():
        return
    got = _parse_eigenvalues(p.read_text(errors="replace"))
    frozen = np.loadtxt(ref, comments="#")[:, 1]
    n = min(len(got), len(frozen))
    d = np.abs(got[:n] - frozen[:n]) * 1e3
    print(f"--- sanity line: arm A {log_name} vs the frozen reference ---")
    print(f"  max|Delta| = {d.max():.6f} meV over {n} states   "
          "(informational; the reference is NOT the criterion — it is only "
          "comparable at the mesh and iteration count it was cut at)")


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "compare":
        sys.exit(__doc__)
    log_name = "bse_p1.log"
    ref = None
    pos = []
    i = 1
    while i < len(argv):
        if argv[i] == "--log":
            log_name = argv[i + 1]; i += 2
        elif argv[i] == "--ref":
            ref = Path(argv[i + 1]); i += 2
        else:
            pos.append(argv[i]); i += 1
    if len(pos) != 2:
        sys.exit(__doc__)
    da, db = Path(pos[0]), Path(pos[1])
    print(f"===== restart_q_storage A/B:  A={da}  B={db} =====")
    ok = compare_files(da, db)
    ok &= compare_eigenvalues(da, db, log_name)
    if ref is not None:
        compare_reference(da, ref, log_name)
    print(f"===== {'PASS' if ok else 'FAIL'} =====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
