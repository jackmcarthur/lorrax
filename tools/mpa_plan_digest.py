"""Hash one MPA Sigma plan's panes and rules, so two trees can be compared.

WHY THIS EXISTS.  ``feat/mpa-binned-width-clause`` claims to be strictly
additive: with ``binned_width_clause`` off, the plan a pass builds must be
the plan ``0f5da1ef`` built, to the byte.  Nothing in the tree could state
that claim -- the nearest gate compared derived scalars through a pass loop
-- so this is the canonicalization the claim is defined against, and it is
a TOOL rather than a test helper because the comparison it exists for runs
two DIFFERENT CHECKOUTS against each other and neither can import the
other's test module.

Usage, one checkout at a time, from that checkout's root::

    PYTHONPATH=<checkout>/src:<checkout>/services/minimax/src \\
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
    python tools/mpa_plan_digest.py --field <spec> [--binned r]

It prints one line per branch and a total, so a disagreement names which
branch moved rather than only that something did.

WHAT GOES IN THE DIGEST, AND WHAT DOES NOT.  Everything the device tau
loop reads: the group ORDER (panes are summed in it and re-association is
the one difference an order change makes), each group's ``idx_B``, its
field shape and omega operand, and per window the nodes, weights, both
energy references, the omega sign, the prefactor, the projection and the
A-side mask.  Excluded: ``name``, ``provenance`` and ``b_mass``.  The
first two are prose carrying the pane's width range at three significant
figures; the third is a sum over the ``idx_B`` already hashed.  A digest
that moved when a format string moved would be a gate nobody keeps green.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

RYD = 13.605693122994


def plan_digest(groups):
    h = hashlib.sha256()
    for grp in groups:
        h.update(b"\x00GROUP")
        h.update(np.asarray(grp.idx_B, dtype="<i8").tobytes())
        h.update(repr(tuple(int(x) for x in grp.field_shape)).encode())
        h.update(np.asarray(grp.omega_operand, dtype="<c16").tobytes())
        for win in grp.windows:
            h.update(b"\x01WINDOW")
            h.update(np.asarray(win.nodes.t, dtype="<c16").tobytes())
            h.update(np.asarray(win.nodes.alpha, dtype="<c16").tobytes())
            h.update(np.asarray(win.mask_A, dtype=bool).tobytes())
            h.update(np.asarray(
                [win.E_ref_A, win.E_ref_B, win.prefactor],
                dtype="<f8").tobytes())
            h.update(np.asarray([win.omega_sign], dtype="<i8").tobytes())
            h.update(str(win.project).encode())
    return h.hexdigest()


def synthetic_field(n_modes, seed, decades, gamma_lo, gamma_hi):
    """A pole field inside the fitter's fourth guard, by construction.

    ``Gamma_p <= a_p`` element by element, which is what
    ``pade_fit``'s fourth guard enforces and what both width clauses'
    derivations rest on.  A field drawn otherwise would be comparing two
    planners on input neither of them can be handed.
    """

    rng = np.random.default_rng(int(seed))
    a = np.sort(10.0 ** rng.uniform(0.0, float(decades),
                                    size=int(n_modes))) / RYD
    g = a * rng.uniform(float(gamma_lo), float(gamma_hi), size=int(n_modes))
    return a, g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-modes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decades", type=float, default=4.0)
    ap.add_argument("--gamma-lo", type=float, default=0.2)
    ap.add_argument("--gamma-hi", type=float, default=0.95)
    ap.add_argument("--n-a", type=int, default=6)
    ap.add_argument("--e-lo", type=float, default=0.3)
    ap.add_argument("--e-hi", type=float, default=6.0)
    ap.add_argument("--binned", type=float, default=None,
                    help="bin ratio; omitted means the flag is OFF")
    ap.add_argument("--npz", default=None,
                    help="a real pole field: keys 'a_ry' and 'gamma_ry'")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from gw.mpa import sigma_pass as SP

    if args.npz:
        with np.load(args.npz) as d:
            a = np.asarray(d["a_ry"], dtype=np.float64)
            g = np.asarray(d["gamma_ry"], dtype=np.float64)
    else:
        a, g = synthetic_field(args.n_modes, args.seed, args.decades,
                               args.gamma_lo, args.gamma_hi)

    E_A = np.linspace(args.e_lo, args.e_hi, args.n_a) / RYD
    mask_A = np.ones(args.n_a, dtype=bool)
    live = np.ones(a.shape, dtype=bool)

    kw = {}
    if args.binned is not None:
        # Only pass the kwarg when it is on, so this tool runs unchanged
        # against a checkout whose planner does not have it.
        kw["binned_width_clause"] = float(args.binned)

    out = {"branches": {}, "n_panes": 0, "n_tau": 0}
    h = hashlib.sha256()
    for space in ("cond", "val"):
        for neg in (False, True):
            groups, stats = SP.plan_branch_groups(
                a_ry=a, gamma_ry=g, live_mask=live,
                E_A_host=E_A, base_mask_A_host=mask_A,
                omega_nonneg_ry=np.linspace(0.0, 5.0 / RYD, 4),
                space=space, neg_omega_half=neg,
                xi_ry=1.0e-9, edge_factor=1.5, rel_tol=1.0e-8,
                target_error=1.0e-8, laplace_max_nodes=64,
                crossing_eps_q=1.0e-10, crossing_max_nodes=400,
                use_shipped_minimax_tables=True,
                print_fn=lambda *a, **k: None, **kw)
            d = plan_digest(groups)
            ntau = int(sum(w.n_tau for grp in groups for w in grp.windows))
            key = f"{space}{'-neg' if neg else '-pos'}"
            out["branches"][key] = {"digest": d, "n_panes": len(groups),
                                    "n_tau": ntau}
            out["n_panes"] += len(groups)
            out["n_tau"] += ntau
            h.update(key.encode())
            h.update(bytes.fromhex(d))
    out["total_digest"] = h.hexdigest()
    out["binned_width_clause"] = args.binned

    if args.json:
        print(json.dumps(out, indent=1))
    else:
        for key, v in out["branches"].items():
            print(f"{key:>10}  panes={v['n_panes']:5d}  "
                  f"tau={v['n_tau']:7d}  {v['digest']}")
        print(f"{'TOTAL':>10}  panes={out['n_panes']:5d}  "
              f"tau={out['n_tau']:7d}  {out['total_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
