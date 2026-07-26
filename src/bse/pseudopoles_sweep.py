"""Sweep p_keep from saved pseudopole data.

Works with two H5 formats:
  1. Intermediates (H_w, C_w, J_w) — full brightness re-decomposition per p_keep.
  2. Final poles (d_bright, omega_bright_ry, sigma) — sweep by truncating
     pre-sorted bright poles.

Usage:
    python -m bse.pseudopoles_sweep \
        --poles bse_pseudopoles.h5 \
        --ref Wc_exact.h5 \
        --omega-ev 0.0 --eta-ev 0.0 \
        --cols 0,1,2,3,4,5,6,7,8 \
        --plot-file sweep.png
"""
from __future__ import annotations

import argparse
import numpy as np
import h5py


RY_TO_EV = 13.6056980659


def _detect_format(h5_path: str) -> str:
    """Detect whether the H5 file has intermediates or final poles."""
    with h5py.File(h5_path, "r") as h5:
        for wname in sorted(h5.keys()):
            g = h5[wname]
            if "H_w" in g and "C_w" in g:
                return "intermediates"
            if "d_bright" in g and "omega_bright_ry" in g:
                return "final"
    return "unknown"


def _reconstruct_from_final(
    h5_path: str,
    p_keep: int,
    omega_ry: float,
    eta_ry: float,
    cols: list[int],
    use_tda: bool,
    ry_to_ev: float = RY_TO_EV,
) -> np.ndarray:
    """Reconstruct Wc columns from saved d_bright/omega_bright, keeping p_keep per window."""
    z = complex(omega_ry, eta_ry)

    all_omega = []
    all_d = []
    all_w = []

    with h5py.File(h5_path, "r") as h5:
        for wname in sorted(h5.keys()):
            g = h5[wname]
            if "d_bright" not in g or "omega_bright_ry" not in g:
                continue
            d_all_win = g["d_bright"][:]       # (n_poles, n_mu)
            omega_all_win = g["omega_bright_ry"][:]  # (n_poles,)

            if d_all_win.size == 0:
                continue

            p_use = min(p_keep, d_all_win.shape[0])
            d = d_all_win[:p_use, :]           # (p_use, n_mu)
            omega = omega_all_win[:p_use]      # (p_use,)

            # d_bright is stored as (n_poles, n_mu); transpose to (n_mu, n_poles)
            d = d.T  # (n_mu, p_use)

            weights = np.ones(p_use, dtype=np.float64)

            if not use_tda:
                n_res = p_use
                omega = np.concatenate([omega, -omega.conj()])
                d = np.concatenate([d, np.conj(d)], axis=1)
                weights = np.concatenate([np.ones(n_res), -np.ones(n_res)])

            all_omega.append(omega)
            all_d.append(d)
            all_w.append(weights)

    if not all_omega:
        return np.zeros((len(cols), 0), dtype=np.complex128)

    omega_cat = np.concatenate(all_omega)
    d_cat = np.concatenate(all_d, axis=1)   # (n_mu, n_poles_total)
    w_cat = np.concatenate(all_w)

    denom = z - omega_cat
    coeffs = w_cat / denom
    Wc = d_cat @ (coeffs[:, None] * d_cat[cols, :].conj().T)  # (n_mu, n_cols)

    return Wc.T  # (n_cols, n_mu)


def _reconstruct_from_intermediates(
    h5_path: str,
    p_keep: int,
    omega_ry: float,
    eta_ry: float,
    cols: list[int],
    use_tda: bool,
    ry_to_ev: float = RY_TO_EV,
    j_floor: float = 1e-6,
) -> np.ndarray:
    """Reconstruct Wc columns from saved intermediates with a given p_keep."""
    z = complex(omega_ry, eta_ry)

    all_omega = []
    all_d = []
    all_w = []

    with h5py.File(h5_path, "r") as h5:
        for wname in sorted(h5.keys()):
            g = h5[wname]
            if "H_w" not in g or "C_w" not in g:
                continue
            H_w = g["H_w"][:]
            C_w = g["C_w"][:]
            J_w = g["J_w"][:] if "J_w" in g else np.zeros((0, 0))
            a_eV = float(g.attrs["a_eV"])
            b_eV = float(g.attrs["b_eV"])
            a_ry = a_eV / ry_to_ev
            b_ry = b_eV / ry_to_ev

            if H_w.size == 0 or C_w.size == 0:
                continue

            n_basis = H_w.shape[0]

            # Brightness decomposition
            G_w = C_w.conj().T @ C_w
            G_w = 0.5 * (G_w + G_w.conj().T)
            sigma2, W_w = np.linalg.eigh(G_w)
            order = np.argsort(sigma2)[::-1]
            sigma2 = sigma2[order]
            W_w = W_w[:, order]

            p_use = min(p_keep, W_w.shape[1])
            Wb = W_w[:, :p_use]

            # Bright Rayleigh-Ritz
            H_b = Wb.conj().T @ H_w @ Wb
            evals_b, vecs_b = np.linalg.eig(H_b)
            order_b = np.argsort(evals_b.real)
            evals_b = evals_b[order_b]
            vecs_b = vecs_b[:, order_b]

            # Filter out-of-window
            in_window = (evals_b.real >= a_ry) & (evals_b.real <= b_ry)
            evals_b = evals_b[in_window]
            vecs_b = vecs_b[:, in_window]

            if evals_b.size == 0:
                continue

            g_b = Wb @ vecs_b
            d_bright = C_w @ g_b  # (n_mu, n_poles)

            weights_b = np.ones(evals_b.shape[0], dtype=np.float64)

            if not use_tda and J_w.size > 0 and evals_b.size > 0:
                J_b = Wb.conj().T @ J_w @ Wb
                j_norms = np.array([
                    vecs_b[:, p].conj() @ J_b @ vecs_b[:, p]
                    for p in range(vecs_b.shape[1])
                ])
                j_norms_real = j_norms.real
                for p in range(d_bright.shape[1]):
                    if j_norms_real[p] > j_floor:
                        d_bright[:, p] /= np.sqrt(j_norms_real[p])

                # Anti-resonant poles
                n_res = evals_b.shape[0]
                evals_b = np.concatenate([evals_b, -evals_b])
                d_bright = np.concatenate([d_bright, np.conj(d_bright)], axis=1)
                weights_b = np.concatenate([np.ones(n_res), -np.ones(n_res)])

            all_omega.append(evals_b)
            all_d.append(d_bright)
            all_w.append(weights_b)

    if not all_omega:
        n_mu = 0
        with h5py.File(h5_path, "r") as h5:
            for wname in h5.keys():
                if "C_w" in h5[wname]:
                    n_mu = h5[wname]["C_w"].shape[0]
                    break
        return np.zeros((n_mu, len(cols)), dtype=np.complex128)

    omega_all = np.concatenate(all_omega)
    d_all = np.concatenate(all_d, axis=1)  # (n_mu, n_poles_total)
    w_all = np.concatenate(all_w)

    # Evaluate Wc[mu, cols]
    denom = z - omega_all  # (n_poles,)
    coeffs = w_all / denom  # (n_poles,)
    Wc = d_all @ (coeffs[:, None] * d_all[cols, :].conj().T)  # (n_mu, n_cols)

    return Wc.T  # (n_cols, n_mu) to match pseudopoles_eval convention


def get_max_poles_per_window(h5_path: str) -> int:
    """Get the maximum number of poles per window."""
    max_n = 0
    with h5py.File(h5_path, "r") as h5:
        for wname in h5.keys():
            g = h5[wname]
            if "H_w" in g:
                max_n = max(max_n, g["H_w"].shape[0])
            elif "d_bright" in g:
                max_n = max(max_n, g["d_bright"].shape[0])
    return max_n


def get_sigma_per_window(h5_path: str) -> dict[str, np.ndarray]:
    """Get brightness singular values per window."""
    result = {}
    with h5py.File(h5_path, "r") as h5:
        for wname in sorted(h5.keys()):
            g = h5[wname]
            if "sigma" in g:
                sigma = g["sigma"][:]
                a_eV = float(g.attrs["a_eV"])
                b_eV = float(g.attrs["b_eV"])
                result[f"{wname} [{a_eV:.1f},{b_eV:.1f}]eV"] = sigma
            elif "C_w" in g:
                C_w = g["C_w"][:]
                if C_w.size > 0:
                    G_w = C_w.conj().T @ C_w
                    G_w = 0.5 * (G_w + G_w.conj().T)
                    sigma2 = np.linalg.eigvalsh(G_w)[::-1]
                    sigma = np.sqrt(np.maximum(sigma2, 0.0))
                    a_eV = float(g.attrs["a_eV"])
                    b_eV = float(g.attrs["b_eV"])
                    result[f"{wname} [{a_eV:.1f},{b_eV:.1f}]eV"] = sigma
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sweep p_keep from saved pseudopole intermediates")
    parser.add_argument("--poles", required=True, help="H5 file with intermediates")
    parser.add_argument("--ref", required=True, help="Reference Wc H5 file")
    parser.add_argument("--omega-ev", type=float, default=0.0)
    parser.add_argument("--eta-ev", type=float, default=0.0)
    parser.add_argument("--cols", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--plot-file", type=str, default="sweep_pkeep.png")
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV)
    args = parser.parse_args(argv)

    cols = [int(c) for c in args.cols.split(",")]
    omega_ry = args.omega_ev / args.ry_to_ev
    eta_ry = args.eta_ev / args.ry_to_ev

    with h5py.File(args.poles, "r") as h5:
        use_tda = bool(h5.attrs.get("use_tda", 1))

    # Load reference
    with h5py.File(args.ref, "r") as h5:
        Wc_ref = h5["Wc"][:]

    # Get sigma per window
    sigmas = get_sigma_per_window(args.poles)

    # Detect format and choose reconstruct function
    fmt = _detect_format(args.poles)
    print(f"Detected H5 format: {fmt}")
    if fmt == "intermediates":
        reconstruct = lambda p: _reconstruct_from_intermediates(
            args.poles, p, omega_ry, eta_ry, cols, use_tda, args.ry_to_ev,
        )
    elif fmt == "final":
        reconstruct = lambda p: _reconstruct_from_final(
            args.poles, p, omega_ry, eta_ry, cols, use_tda, args.ry_to_ev,
        )
    else:
        raise RuntimeError(f"Unknown H5 format in {args.poles}")

    # Sweep p_keep
    max_basis = get_max_poles_per_window(args.poles)
    p_values = list(range(1, max_basis + 1))
    errors = []
    alphas = []
    for p in p_values:
        Wc = reconstruct(p)
        rel = float(np.linalg.norm(Wc - Wc_ref) / np.linalg.norm(Wc_ref))
        alpha = float(np.real(np.sum(Wc.conj() * Wc_ref)) / np.sum(np.abs(Wc_ref) ** 2))
        errors.append(rel)
        alphas.append(alpha)
        print(f"  p_keep={p:3d}: error={rel:.4f} ({rel*100:.1f}%), alpha={alpha:.6f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Singular values per window
    ax = axes[0]
    for label, sv in sigmas.items():
        ax.semilogy(np.arange(1, len(sv) + 1), sv, "o-", markersize=3, label=label)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Brightness singular value")
    ax.set_title("Singular values per window")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Relative error vs p_keep
    ax = axes[1]
    ax.plot(p_values, [e * 100 for e in errors], "bo-", markersize=4)
    ax.set_xlabel("p_keep (bright modes per window)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Error vs. bright modes kept")
    ax.grid(True, alpha=0.3)
    ax.axhline(5, color="tab:orange", linestyle="--", alpha=0.5, label="5%")
    ax.legend(fontsize=8)

    # Panel 3: Alpha vs p_keep
    ax = axes[2]
    ax.plot(p_values, alphas, "ro-", markersize=4)
    ax.set_xlabel("p_keep (bright modes per window)")
    ax.set_ylabel("Alpha (overlap)")
    ax.set_title("Overlap vs. bright modes kept")
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Pseudopole p_keep sweep (non-TDA={'no' if use_tda else 'yes'})", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.plot_file, dpi=150)
    print(f"\nPlot saved to {args.plot_file}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
