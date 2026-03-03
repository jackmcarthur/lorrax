#!/usr/bin/env python3
"""Quick diagnostic plot for local pseudopotential before/after FFT build."""

from __future__ import annotations

import argparse
from pathlib import Path

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
import matplotlib.pyplot as plt

from psp.load_upf import load_upf
from psp.normalize import normalize_dataclass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upf", type=Path, help="Path to UPF file")
    parser.add_argument("--out", type=Path, default=Path("vloc_radial.png"), help="Output figure path")
    args = parser.parse_args()

    pseudo = normalize_dataclass(load_upf(args.upf))

    r_mesh = np.asarray(pseudo.pp_mesh.pp_r.value, dtype=float)
    vloc_radial = np.asarray(pseudo.pp_local.value, dtype=float)
    Zval = float(getattr(pseudo.pp_header, 'z_valence', 0.0))
    e2 = 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        coul = -Zval * e2 / r_mesh
    if np.isfinite(coul[1]):
        coul[0] = coul[1]

    from scipy.special import erf

    short = vloc_radial.copy()
    ratio = np.zeros_like(r_mesh)
    nonzero = r_mesh > 0
    ratio[nonzero] = erf(r_mesh[nonzero]) / r_mesh[nonzero]
    if not nonzero[0]:
        ratio[0] = 2.0 / np.sqrt(np.pi)
    short += Zval * e2 * ratio

    ymin = np.min(vloc_radial)
    ymax = np.max(np.concatenate([vloc_radial, short]))
    ymin_plot = ymin - 0.05 * abs(ymin) if ymin < 0 else ymin * 0.95

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(r_mesh, vloc_radial, label="V_loc(r) from UPF")
    ax.plot(r_mesh, short, label="Short-range V_loc(r) + Z erf(r)/r")
    ax.plot(r_mesh, coul, label="-Z e^2 / r")
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("Potential (Ry)")
    ax.set_xlim(0.0, r_mesh[-1])
    ax.set_ylim(ymin_plot, ymax * 1.05)
    ax.set_title(args.upf.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved figure to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
