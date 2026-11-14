r"""Linearity of the bispinor transverse terms in $\alpha^2$ (a P = 4 leg, two processes).

The transverse direct field $\langle m|\boldsymbol\alpha\cdot\mathbf A[\mathbf J]|n\rangle$
and the bare transverse exchange $\Sigma^B$ each carry two current vertices,
so scaling the fine-structure constant by $s$ scales both by $s^2$, while the
charge part (V_H, the charge exchange, $\Sigma_c$ on $W_C$) does not contain
$\alpha$.  $s = 1/\sqrt2$ must halve them.  The scale is the import-time
constant ``common.bispinor_init.HALFALPHA`` (``LORRAX_DIAG_ALPHA_FS_SCALE``), so
the scaled run is its own process and this check is not an hsuite stage.

Three runs of one ``bispinor_gw = bare_transverse`` one-shot deck, each with
``restart = false`` in its own directory:

* ``full``: the deck as it is;
* ``scaled``: the same with ``LORRAX_DIAG_ALPHA_FS_SCALE=0.7071067811865476``;
* ``charge``: the deck plus ``bispinor_transverse_exchange = false``.

Usage (CPU, after the three runs)::

    python3 tests/hsuite/bispinor_linearity.py full/x_sigma.h5 scaled/x_sigma.h5 \
        charge/x_sigma.h5 [--scale 0.70710678] [--rtol 1e-4]

Exit status 0 and ``linearity PASS`` when every check holds.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _members(path):
    import h5py
    with h5py.File(path, "r") as h5:
        return {key: np.asarray(h5[key][()]) for key in (
            "hartree_scalar_kij_ev", "hartree_transverse_kij_ev",
            "sigma_sx_kij_ev", "sigma_c_kij_ev") if key in h5}


def linearity_problems(full, scaled, charge, *, scale, rtol):
    """Checks of the alpha^2 law on three runs' ``{member: array}`` (eV).

    ``H_T(scaled) = s^2 H_T(full)``; ``Sigma^B(scaled) = s^2 Sigma^B(full)``
    with ``Sigma^B = sigma_sx - sigma_sx(charge)`` (the charge run has the
    transverse exchange off); the scalar Hartree and ``sigma_c`` do not move.
    Each bound is ``rtol`` times the largest element of the full-run term.
    """
    s2 = float(scale) ** 2
    out = []

    def close(label, got, want, ref):
        bound = rtol * max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
        diff = float(np.max(np.abs(got - want)))
        if not np.isfinite(diff) or diff > bound:
            out.append(f"{label}: max |diff| {diff:.3e} eV > {bound:.3e}")

    h_t = full["hartree_transverse_kij_ev"]
    if not np.any(h_t):
        out.append("hartree_transverse_kij_ev of the full run is zero: nothing to scale")
    close("H_T(scaled) vs s^2 H_T(full)", scaled["hartree_transverse_kij_ev"], s2 * h_t, h_t)
    b_full = full["sigma_sx_kij_ev"] - charge["sigma_sx_kij_ev"]
    b_scaled = scaled["sigma_sx_kij_ev"] - charge["sigma_sx_kij_ev"]
    if not np.any(b_full):
        out.append("Sigma^B of the full run is zero: nothing to scale")
    close("Sigma^B(scaled) vs s^2 Sigma^B(full)", b_scaled, s2 * b_full, b_full)
    for member in ("hartree_scalar_kij_ev", "sigma_c_kij_ev"):
        close(f"{member} (alpha-free)", scaled[member], full[member], full[member])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("full")
    parser.add_argument("scaled")
    parser.add_argument("charge")
    parser.add_argument("--scale", type=float, default=2.0 ** -0.5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    args = parser.parse_args(argv)
    problems = linearity_problems(_members(args.full), _members(args.scaled),
                                  _members(args.charge), scale=args.scale,
                                  rtol=args.rtol)
    for line in problems:
        print(f"linearity FAIL {line}")
    print("linearity PASS" if not problems else "linearity FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
