#!/usr/bin/env python3
"""D6 item 4: instrument the S-tensor's dE->0 branch on the real Na deck.

Reproduces `_s_tensor_kernel`'s exact per-(k,n,m) weight-selection logic
(src/gw/qsgw_head.py, post-D1) in plain NumPy over the REAL WFN energies and
the SAME MP1 occupation solve production uses
(sc_iteration._solve_occupation_state: target_electrons = wfn.num_electrons,
capacity = spin_degeneracy_factor(wfn), width_ry = deck's occ_broadening_ry),
to answer: on a real Fermi surface, how many band pairs would the OLD
`|denom| <= 1e-16` hard clip have zeroed, vs how many now take the new
analytic MP1 limit, and what is that limit's typical/max magnitude there.

Writes nothing to the WFN or any production artifact; this is a read-only
census script.  Single process, no jax mesh required (dense (nk,nb,nb)
census, nb=48, nk<=512 -- trivially small).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _run(args) -> int:
    sys.path.insert(0, os.environ["LORRAX_CHECKOUT"] + "/src")
    import h5py

    from gw.efermi import mp1_negative_derivative, solve_mp1_occupations
    from psp.get_DFT_mtxels import spin_degeneracy_factor

    with h5py.File(args.wfn, "r") as f:
        kp = f["mf_header/kpoints"]
        el_ry = np.asarray(kp["el"])  # (ns, nk, mnband), Ry
        num_electrons = None
        # BGW-style header: ifmax integrated over occ, or read directly if
        # present.  Fall back to summing the stored `occ` array (electrons
        # per unit cell) the same way psp.get_DFT_mtxels.wfn.num_electrons
        # does, to avoid re-deriving a different definition.
        occ = np.asarray(kp["occ"])  # (ns, nk, mnband)
        w = np.asarray(kp["w"])      # (nk,) BZ weight, sums to 1
        nspinor = int(np.asarray(kp["nspinor"])[()])
        nspin = int(np.asarray(kp["nspin"])[()])
        mnband = int(np.asarray(kp["mnband"])[()])
        kgrid = np.asarray(kp["kgrid"])

    print(f"# WFN: {args.wfn}")
    print(f"# nspin={nspin} nspinor={nspinor} mnband={mnband} kgrid={kgrid.tolist()}")

    nb = int(args.nband)
    ns, nk, _ = el_ry.shape
    if ns != 1:
        raise SystemExit(f"expected ns=1 (spinor or scalar single channel), got {ns}")
    e = el_ry[0, :, :nb].astype(np.float64)          # (nk, nb) Ry
    kweights = np.full(nk, 1.0 / nk, dtype=np.float64)

    # Capacity / target electrons: same primitives production calls
    # (sc_iteration._solve_occupation_state).  spin_degeneracy_factor takes
    # a wfn-like object; build the minimal shim it actually reads.
    class _WfnShim:
        pass
    shim = _WfnShim()
    shim.nspin = nspin
    shim.nspinor = nspinor
    capacity = float(spin_degeneracy_factor(shim))
    # num_electrons: EXACT formula from services/wfn_loader/loader.py
    # (WfnLoader.num_electrons, the value production's
    # sc_iteration._solve_occupation_state reads as target_electrons):
    # capacity * sum_k (w_k/sum(w)) * sum_{s,b} occ[s,k,b].
    w_norm = w / float(np.sum(w))
    target_electrons = float(capacity * np.einsum("k,skb->", w_norm, occ, optimize=True))
    print(f"# capacity(spin_degeneracy_factor)={capacity} target_electrons={target_electrons}")

    width_ry = float(args.width_ry)
    mu_ry, f_kn = solve_mp1_occupations(
        e, kweights, target_electrons, width_ry,
        state_capacity=capacity, clamp_tol=float(args.clamp_tol))
    mu_ry = float(mu_ry)
    f_kn = np.asarray(f_kn)
    s_kn = np.asarray(mp1_negative_derivative(e, mu_ry, width_ry))  # -df/dE, (nk,nb)
    print(f"# solved mu_ry={mu_ry:.10f} width_ry={width_ry}")
    print(f"# occ range [{f_kn.min():.6f}, {f_kn.max():.6f}], "
          f"-df/dE range [{s_kn.min():.6e}, {s_kn.max():.6e}]")

    # --- reproduce _s_tensor_kernel's per-(k,n,m) selection exactly ---
    dE = e[:, :, None] - e[:, None, :]            # (nk,nb,nb): bra=n, ket=m
    f_diff = f_kn[:, None, :] - f_kn[:, :, None]   # f_ket - f_bra
    transition = dE > 0.0

    scale = np.maximum(1.0, np.maximum(
        np.abs(e)[:, :, None], np.abs(e)[:, None, :]))
    eps = np.finfo(np.float64).eps
    near_degenerate = np.abs(dE) <= 64.0 * eps * scale
    s_avg = 0.5 * (s_kn[:, :, None] + s_kn[:, None, :])

    new_branch = transition & near_degenerate
    n_transitions = int(np.sum(transition))
    n_new_branch = int(np.sum(new_branch))
    print(f"# total (k,n,m) transitions with dE>0: {n_transitions}")
    print(f"# of those, near-degenerate (new MP1-limit branch fires): {n_new_branch} "
          f"({100.0 * n_new_branch / max(n_transitions, 1):.4f}%)")

    # For the new-branch pairs: what would the OLD hard clip have done?
    # At a representative set of frequencies (z = omega + i*eta) spanning
    # the deck's own Sigma omega grid, since the weight's z-dependence
    # matters for whether |denom| clears 1e-16.
    omegas_ev = np.array(args.omegas_ev, dtype=np.float64)
    eta_ry = float(args.eta_ry)
    RY_TO_EV = 13.605693009  # matches this tree's RYD_TO_EV to quoted precision
    prefactor = 1.0  # weight magnitude only; overall physical prefactor cancels in the count

    rows = []
    for om_ev in omegas_ev:
        z = (om_ev / RY_TO_EV) + 1j * eta_ry
        denom = dE * (z * z - dE * dE)
        old_zeroed = np.abs(denom) <= 1.0e-16
        # Among the pairs the NEW code routes to the degenerate branch:
        old_zeroed_here = int(np.sum(old_zeroed & new_branch))
        # The old "regular" (unclipped) formula's value where it was NOT
        # clipped but IS near-degenerate -- this is the catastrophic-
        # cancellation regime: f_diff/dE computed as a raw divided
        # difference of two nearly-equal floats, not simply "the right
        # answer with no fix needed."
        with np.errstate(divide="ignore", invalid="ignore"):
            old_regular = prefactor * f_diff / denom
        old_regular_here = old_regular[new_branch & ~old_zeroed]
        new_degenerate = prefactor * s_avg / (z * z)
        new_here = new_degenerate[new_branch]
        rows.append((
            om_ev, old_zeroed_here, n_new_branch - old_zeroed_here,
            float(np.max(np.abs(new_here))) if new_here.size else 0.0,
            float(np.median(np.abs(new_here))) if new_here.size else 0.0,
            float(np.max(np.abs(old_regular_here))) if old_regular_here.size else float("nan"),
        ))

    print("#")
    print("# omega_eV | old_clip_zeroed | old_clip_left_nonzero(unstable) | "
          "new_MP1_limit_max|w| | new_MP1_limit_median|w| | old_unclipped_max|w|(unstable ref)")
    for om, oz, onz, nmax, nmed, omax in rows:
        print(f"  {om:8.4f} | {oz:16d} | {onz:30d} | {nmax:.6e} | {nmed:.6e} | {omax:.6e}")

    print("#")
    print("# VERDICT: at every sampled omega, ALL new-branch pairs that the old")
    print("# |denom|<=1e-16 clip would have zeroed now carry the finite MP1")
    print("# limit shown above; the pairs the old clip did NOT zero (nonzero")
    print("# 'old_clip_left_nonzero') were computed by the old code as a raw")
    print("# divided difference of two nearly machine-degenerate floats --")
    print("# numerically unstable, not a validated regularization -- and are")
    print("# now replaced uniformly by the same closed-form limit.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--nband", type=int, required=True)
    ap.add_argument("--width-ry", type=float, default=0.01)
    ap.add_argument("--clamp-tol", type=float, default=1.0e-10)
    ap.add_argument("--eta-ry", type=float, default=0.25 / 13.605693009)
    ap.add_argument("--omegas-ev", type=float, nargs="+",
                     default=[-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
    parsed = ap.parse_args()
    sys.exit(_run(parsed))
