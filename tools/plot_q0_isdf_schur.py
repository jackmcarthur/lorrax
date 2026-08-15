#!/usr/bin/env python3
"""Plot Lorrax q->0 direct/ISDF-Schur heads against BGW finite-q0 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _peak(w, eps, lo=3.0, hi=10.0):
    keep = (w >= lo) & (w <= hi)
    return float(w[keep][np.argmax(-eps.imag[keep])])


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--bgw", required=True)
    p.add_argument("--lorrax", required=True)
    p.add_argument("--direct-reference")
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    bgw = np.loadtxt(a.bgw)
    lx = np.loadtxt(a.lorrax)
    wb = bgw[:, 0]
    wl = lx[:, 0]
    chi_b = bgw[:, 2] + 1j * bgw[:, 3]
    eb_head = bgw[:, 4] + 1j * bgw[:, 5]
    eb_full = bgw[:, 6] + 1j * bgw[:, 7]
    chi_d = lx[:, 1] + 1j * lx[:, 2]
    chi_f = lx[:, 3] + 1j * lx[:, 4]
    el_direct = lx[:, 5] + 1j * lx[:, 6]
    el_folded = lx[:, 7] + 1j * lx[:, 8]
    wc_direct = lx[:, 9] + 1j * lx[:, 10]
    wc_folded = lx[:, 11] + 1j * lx[:, 12]
    v_candidates = (1.0 - 1.0 / eb_head) / chi_b
    finite = np.isfinite(v_candidates.real)
    v0 = float(np.median(v_candidates.real[finite]))
    wc_b_head = v0 * (eb_head - 1.0)
    wc_b_full = v0 * (eb_full - 1.0)

    direct_error = np.nan
    if a.direct_reference:
        ref = np.loadtxt(a.direct_reference)
        if np.array_equal(ref[:, 0], wl):
            er = ref[:, 3] + 1j * ref[:, 4]
        else:
            er = np.interp(wl, ref[:, 0], ref[:, 3]) + 1j * np.interp(
                wl, ref[:, 0], ref[:, 4])
        direct_error = float(np.max(np.abs(er - el_direct)))

    black = "#111111"
    gray = "#7a7a7a"
    orange = "#d55e00"
    blue = "#0072b2"
    styles = (
        (wb, eb_head, black, "--", "BGW head-only"),
        (wb, eb_full, gray, "-", r"BGW full $\epsilon^{-1}_{00}$"),
        (wl, el_direct, orange, "--", r"Lorrax $q\to0$ direct"),
        (wl, el_folded, blue, "-", r"Lorrax $q\to0$ ISDF Schur"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    for col, (xmin, xmax) in enumerate(((0.0, 20.0), (3.0, 10.0))):
        for w, e, color, ls, label in styles:
            axes[0, col].plot(
                w, e.real, color=color, ls=ls, lw=2.0, label=label)
            axes[1, col].plot(
                w, -e.imag, color=color, ls=ls, lw=2.0, label=label)
        axes[0, col].set_xlim(xmin, xmax)
        axes[1, col].set_xlim(xmin, xmax)
        axes[0, col].set_ylabel(r"$\mathrm{Re}\,\epsilon^{-1}_{00}$")
        axes[1, col].set_ylabel(r"$-\mathrm{Im}\,\epsilon^{-1}_{00}$")
        axes[1, col].set_xlabel(r"$\omega$ (eV)")
        for row in range(2):
            axes[row, col].axhline(0.0, color="0.8", lw=0.7)
            axes[row, col].grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    peaks = {
        "BGW full": _peak(wb, eb_full),
        "Lorrax direct": _peak(wl, el_direct),
        "Lorrax Schur": _peak(wl, el_folded),
    }
    axes[1, 1].text(
        0.98, 0.95,
        "\n".join(f"{name}: {value:.1f} eV" for name, value in peaks.items()),
        transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85})
    fig.suptitle(
        r"Sodium dielectric head: Lorrax $q\to0$ ISDF Schur fold vs "
        r"BerkeleyGW $q_0=(0,0,1/8)$")
    fig.text(
        0.5, 0.012,
        "N=9, 23 bands, MP1 width 0.136056931 eV, eta=0.1 eV; "
        "Lorrax body rank 96. The 0-eV point is the broadened dynamic limit.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    eps_path = out / "sodium_epsinv_q0_isdf_schur.png"
    fig.savefig(eps_path, dpi=200)
    plt.close(fig)

    wc_styles = (
        (wb, wc_b_head / v0, black, "--", "BGW head-only"),
        (wb, wc_b_full / v0, gray, "-", "BGW full"),
        (wl, wc_direct / v0, orange, "--", "Lorrax direct"),
        (wl, wc_folded / v0, blue, "-", "Lorrax ISDF Schur"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True)
    for w, value, color, ls, label in wc_styles:
        axes[0].plot(w, value.real, color=color, ls=ls, lw=2.0, label=label)
        axes[1].plot(w, value.imag, color=color, ls=ls, lw=2.0, label=label)
    axes[0].set_ylabel(r"$\mathrm{Re}\,[W^c_{00}/v_0]$")
    axes[1].set_ylabel(r"$\mathrm{Im}\,[W^c_{00}/v_0]$")
    axes[1].set_xlabel(r"$\omega$ (eV)")
    for ax in axes:
        ax.set_xlim(0.0, 20.0)
        ax.axhline(0.0, color="0.8", lw=0.7)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(
        r"Sodium $W^c_{00}(\omega)$: direct and ISDF-Schur head models")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    wc_path = out / "sodium_Wc_q0_isdf_schur.png"
    fig.savefig(wc_path, dpi=200)
    plt.close(fig)

    table = np.column_stack((
        wl, chi_d.real, chi_d.imag, chi_f.real, chi_f.imag,
        el_direct.real, el_direct.imag, el_folded.real, el_folded.imag))
    np.savetxt(
        out / "lorrax_q0_direct_folded_plot_data.dat", table, fmt="%.16e",
        header="omega_ev Re_chi_direct Im_chi_direct Re_chi_folded "
        "Im_chi_folded Re_epsinv_direct Im_epsinv_direct "
        "Re_epsinv_folded Im_epsinv_folded")
    print(f"wrote {eps_path}")
    print(f"wrote {wc_path}")
    print("peaks " + " ".join(f"{k}={v:.6f}" for k, v in peaks.items()))
    print(f"direct_reference_max_abs={direct_error:.9e}")
    print(f"v0={v0:.16g}")


if __name__ == "__main__":
    main()
