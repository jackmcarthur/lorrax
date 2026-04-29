"""Evaluate W_c(omega) columns from a saved pseudopoles file.

This is a lightweight post-processing utility for `bse_pseudopoles.py`.

We interpret the saved pseudopoles as an approximation of the retarded
correlation screened interaction (in the ISDF/r_mu basis).

For TDA:
  W_c(omega) ≈ sum_s  d_s d_s^H / (omega - Omega_s)

For non-TDA (full Casida / Lehmann representation):
  W_c(omega) ≈ sum_s [ d_s d_s^H / (omega - Omega_s)
                      - d_s* d_s^T / (omega + Omega_s) ]

This is implemented via a unified evaluation formula with per-pole weights:
  W_c[mu,nu] = sum_p  w_p * d_p[mu] * conj(d_p[nu]) / (omega - omega_p)

where non-TDA poles are stored as:
  Resonant:      omega_p = +Omega_s,  d_p = d_s,       w_p = +1
  Anti-resonant: omega_p = -Omega_s,  d_p = conj(d_s), w_p = -1

Each residue vector d_s is the *Coulomb-dressed density channel*
used elsewhere in this repo (see `bse_w_exact.py`):
  d_s[mu] = (V * (d X + d* Y))[mu]   (non-TDA)
  d_s[mu] = (V * (d X))[mu]         (TDA)

So the evaluated columns here are directly comparable to:
  - `bse_w_exact.py --write-kind Wc`
  - SOS+Dyson reference columns in `../tests_isdf/.../compare_w_exact_to_sos.py`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class Poles:
    omega_ry: np.ndarray  # (n_poles,)
    d: np.ndarray  # (n_poles, n_mu)
    weight: np.ndarray  # (n_poles,)  +1 resonant, -1 anti-resonant


def _iter_window_groups(h5: h5py.File, windows: Iterable[str] | None) -> list[str]:
    names = [k for k, v in h5.items() if isinstance(v, h5py.Group)]
    names = sorted(names)
    if windows is None:
        return names
    want = set(windows)
    return [n for n in names if n in want]


def load_pseudopoles(path: str, *, windows: list[str] | None = None) -> Poles:
    omega_all: list[np.ndarray] = []
    d_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    with h5py.File(path, "r") as h5:
        for wname in _iter_window_groups(h5, windows):
            g = h5[wname]

            omega_b = np.asarray(g.get("omega_bright_ry", np.zeros((0,), dtype=np.complex128)))
            d_b = np.asarray(g.get("d_bright", np.zeros((0, 0), dtype=np.complex128)))
            w_b = np.asarray(g.get("weight_bright", np.ones(omega_b.size, dtype=np.float64)))
            omega_t = np.asarray(g.get("omega_tail_ry", np.zeros((0,), dtype=np.complex128)))
            d_t = np.asarray(g.get("d_tail", np.zeros((0, 0), dtype=np.complex128)))
            w_t = np.asarray(g.get("weight_tail", np.ones(omega_t.size, dtype=np.float64)))

            if omega_b.size:
                omega_all.append(omega_b.reshape(-1))
                d_all.append(d_b.reshape((omega_b.size, -1)))
                w_all.append(w_b.reshape(-1))
            if omega_t.size:
                omega_all.append(omega_t.reshape(-1))
                d_all.append(d_t.reshape((omega_t.size, -1)))
                w_all.append(w_t.reshape(-1))

    if not omega_all:
        return Poles(
            omega_ry=np.zeros((0,), dtype=np.complex128),
            d=np.zeros((0, 0), dtype=np.complex128),
            weight=np.zeros((0,), dtype=np.float64),
        )

    omega = np.concatenate(omega_all, axis=0)
    d = np.concatenate(d_all, axis=0)
    weight = np.concatenate(w_all, axis=0)
    return Poles(omega_ry=omega, d=d, weight=weight)


def reconstruct_Wc_columns(poles: Poles, *, z_ry: complex, cols: np.ndarray) -> np.ndarray:
    """Return Wc[:, cols] as row-major columns: shape (n_cols, n_mu).

    Uses: Wc[mu, nu] = sum_p w_p * d_p[mu] * conj(d_p[nu]) / (z - omega_p)
    where w_p is +1 for resonant poles and -1 for anti-resonant poles.
    """
    omega = poles.omega_ry
    d = poles.d
    w = poles.weight
    if omega.size == 0:
        return np.zeros((int(cols.size), int(d.shape[1]) if d.ndim == 2 else 0), dtype=np.complex128)
    if d.ndim != 2:
        raise ValueError(f"expected d to have shape (n_poles, n_mu); got {d.shape}")
    if cols.size == 0:
        return np.zeros((0, d.shape[1]), dtype=d.dtype)

    denom = (z_ry - omega).astype(np.complex128)  # (n_poles,)
    coeffs = w / denom  # (n_poles,)  -- w_p / (z - omega_p)

    # Wc[:,nu] = sum_p w_p * d_p * conj(d_p[nu]) / (z - omega_p).
    d_conj_cols = d[:, cols].conj()  # (n_poles, n_cols)
    tmp = d_conj_cols * coeffs[:, None]  # (n_poles, n_cols)
    return tmp.T @ d  # (n_cols, n_mu)


def _parse_cols(col_str: str | None, n_mu: int) -> np.ndarray:
    if not col_str:
        return np.arange(n_mu, dtype=int)
    cols = [int(x) for x in col_str.split(",") if x.strip() != ""]
    cols = [c for c in cols if 0 <= c < n_mu]
    return np.array(cols, dtype=int)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate Wc(omega) columns from a pseudopoles file.")
    ap.add_argument("--poles", required=True, help="Input H5 from bse_pseudopoles.py")
    ap.add_argument("--out", default="Wc_from_pseudopoles.h5", help="Output H5 in bse_w_exact-compatible format")
    ap.add_argument("--omega-ev", type=float, default=0.0)
    ap.add_argument("--eta-ev", type=float, default=0.0)
    ap.add_argument("--cols", type=str, default=None, help="Comma-separated mu column indices (0-based).")
    ap.add_argument(
        "--windows",
        type=str,
        default=None,
        help="Comma-separated window group names to include (default: all groups).",
    )
    args = ap.parse_args(argv)

    with h5py.File(args.poles, "r") as h5:
        ry_to_ev = float(h5.attrs.get("ry_to_ev", 13.605693122994))
        use_tda = int(h5.attrs.get("use_tda", 0))
        include_W = int(h5.attrs.get("include_W", 0))

    windows = None
    if args.windows:
        windows = [w.strip() for w in args.windows.split(",") if w.strip()]

    poles = load_pseudopoles(args.poles, windows=windows)
    if poles.d.size == 0:
        raise SystemExit("No poles/residues found in input file.")

    n_mu = int(poles.d.shape[1])
    cols = _parse_cols(args.cols, n_mu)
    z_ry = (args.omega_ev + 1j * args.eta_ev) / ry_to_ev

    Wc_cols = reconstruct_Wc_columns(poles, z_ry=z_ry, cols=cols)

    with h5py.File(args.out, "w") as h5:
        h5.attrs["omega_ev"] = float(args.omega_ev)
        h5.attrs["eta_ev"] = float(args.eta_ev)
        h5.attrs["ry_to_ev"] = float(ry_to_ev)
        h5.attrs["use_tda"] = int(use_tda)
        h5.attrs["include_W"] = int(include_W)
        h5.attrs["source_poles_file"] = str(args.poles)
        if windows is not None:
            h5.attrs["windows"] = ",".join(windows)
        h5.create_dataset("columns", data=cols.astype(np.int32))
        h5.create_dataset("Wc", data=Wc_cols)

    print(f"Wrote Wc columns to {args.out} (n_cols={cols.size}, n_mu={n_mu}, n_poles={poles.omega_ry.size})")


if __name__ == "__main__":
    main()

