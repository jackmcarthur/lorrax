#!/usr/bin/env python3
"""Plot radial pseudopotential projectors and their Fourier transforms.

Usage
-----
python scripts/plot_upf_projectors.py path/to/PSEUDO.upf [output_dir]

Generates PNGs showing each projector β_l(r) and its Fourier–Bessel transform
F_l(q) so that real- and reciprocal-space normalisations can be inspected.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn

from isdf.psp.load_upf import load_upf
from isdf.psp.normalize import normalize_dataclass


def _beta_over_r(beta: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return β(r) given UPF-stored r·β(r)."""

    beta = np.asarray(beta, dtype=float)
    r = np.asarray(r, dtype=float)
    if beta.size != r.size:
        raise ValueError("beta and radial grid lengths differ")
    out = np.zeros_like(beta)
    if r[0] == 0.0:
        out[1:] = beta[1:] / r[1:]
        out[0] = out[1]
    else:
        out[:] = beta / r
    return out


def _weights(r: np.ndarray, rab: np.ndarray | None) -> np.ndarray:
    if rab is not None:
        return np.asarray(rab, dtype=float)
    dr = r[1] - r[0]
    w = np.ones_like(r)
    w[0] = 0.5
    w[-1] = 0.5
    return w * dr


def spherical_transform(beta_r: np.ndarray, r: np.ndarray, rab: np.ndarray | None, l: int, q_grid: np.ndarray) -> np.ndarray:
    """Compute F_l(q) = ∫ r² β(r) j_l(q r) dr."""

    weights = _weights(r, rab)
    qr = np.outer(q_grid, r)
    j_l = spherical_jn(l, qr)
    return (j_l * (beta_r * r * r)) @ weights


def plot_projectors(upf_path: Path, out_dir: Path, q_max: float | None, q_points: int) -> None:
    pseudo = normalize_dataclass(load_upf(upf_path))
    ppmesh = getattr(pseudo, "pp_mesh", None)
    ppnl = getattr(pseudo, "pp_nonlocal", None)
    if ppmesh is None or ppnl is None:
        raise RuntimeError("pseudopotential lacks PP_MESH or PP_NONLOCAL section")

    r = np.asarray(ppmesh.pp_r.value, dtype=float)
    rab = None
    try:
        rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
    except Exception:
        rab = None

    betas: Iterable = getattr(ppnl, "pp_beta", [])
    if not betas:
        raise RuntimeError("no PP_BETA projectors found in UPF")

    label = getattr(pseudo.pp_header, "element", upf_path.stem)

    if q_max is None:
        q_max = 60.0
    q_grid = np.linspace(0.0, q_max, q_points)

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, "lll", getattr(beta, "angular_momentum", 0)))
        beta_r = _beta_over_r(np.asarray(beta.value, dtype=float), r)
        F_q = spherical_transform(beta_r, r, rab, l, q_grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(r, beta_r)
        axes[0].set_xlabel("r (Bohr)")
        axes[0].set_ylabel(r"$\beta_l(r)$")
        axes[0].set_title(f"{label} β{idx} (ℓ={l})")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(q_grid, F_q.real, label="Re")
        axes[1].plot(q_grid, F_q.imag, label="Im", linestyle="--")
        axes[1].set_xlabel("q (Bohr$^{-1}$)")
        axes[1].set_ylabel(r"$F_l(q)$")
        axes[1].set_title("Fourier–Bessel transform")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        outfile = out_dir / f"{upf_path.stem}_beta{idx:02d}_l{l}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UPF projectors in real and reciprocal space")
    parser.add_argument("upf", type=Path, help="Path to pseudopotential (.upf)")
    parser.add_argument("output", type=Path, nargs="?", help="Directory for PNG files (default: UPF directory)")
    parser.add_argument("--qmax", type=float, default=None, help="Maximum q (Bohr^{-1}) for Fourier grid")
    parser.add_argument("--qpoints", type=int, default=2000, help="Number of samples along q")
    args = parser.parse_args()

    upf_path = args.upf.resolve()
    if not upf_path.exists():
        raise FileNotFoundError(upf_path)

    out_dir = args.output.resolve() if args.output else upf_path.parent
    plot_projectors(upf_path, out_dir, args.qmax, args.qpoints)


if __name__ == "__main__":
    main()
