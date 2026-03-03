from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from isdf.bse_isdf.pseudopoles_eval import load_pseudopoles, reconstruct_Wc_columns


def _load_exact(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        cols = np.asarray(f["columns"][:], dtype=int)
        wc = np.asarray(f["Wc"][:])
        attrs = dict(f.attrs)
    return cols, wc, attrs


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b)
    return float(num / den) if den != 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recreate the 12v12c window-inclusion sweep plot for Wc(0) pseudopoles vs exact."
    )
    ap.add_argument(
        "--poles",
        default="projects/test_isdf/tmp/m20_12v12c_w6.h5",
        help="Pseudopoles H5 (from bse_pseudopoles).",
    )
    ap.add_argument(
        "--exact",
        default="projects/test_isdf/tmp/Wc_exact_12v12c_converged.h5",
        help="Exact Wc(0) columns H5 (from bse_w_exact).",
    )
    ap.add_argument("--out", default="projects/test_isdf/tmp/sweep_12v12c_m20_w6.png", help="Output plot path.")
    ap.add_argument(
        "--windows",
        default=None,
        help="Comma-separated window names to consider in order (default: infer W1..Wn from file).",
    )
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cols, wc_exact, exact_attrs = _load_exact(args.exact)
    omega_ev = float(exact_attrs.get("omega_ev", 0.0))
    eta_ev = float(exact_attrs.get("eta_ev", 0.0))

    with h5py.File(args.poles, "r") as f:
        pole_windows = sorted([k for k, v in f.items() if isinstance(v, h5py.Group)])
        ry_to_ev = float(f.attrs.get("ry_to_ev", 13.605693122994))

    if args.windows is not None:
        windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    else:
        windows = pole_windows

    if not windows:
        raise SystemExit("No window groups found.")

    z_ry = (omega_ev + 1j * eta_ev) / ry_to_ev

    errs = []
    for n in range(1, len(windows) + 1):
        poles = load_pseudopoles(args.poles, windows=windows[:n])
        wc_p = reconstruct_Wc_columns(poles, z_ry=z_ry, cols=cols)
        errs.append(_rel_err(wc_p, wc_exact))

    # Plot
    import matplotlib.pyplot as plt

    x = np.arange(1, len(windows) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
    ax.plot(x, errs, marker="o")
    ax.set_xlabel("Number of Windows Included")
    ax.set_ylabel("Relative Error  ||Wc_pseudo - Wc_exact|| / ||Wc_exact||")
    ax.set_yscale("log")
    ttl = args.title
    if ttl is None:
        ttl = f"12v12c Wc({omega_ev:g} eV) Window Sweep"
    ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xticks(x)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

