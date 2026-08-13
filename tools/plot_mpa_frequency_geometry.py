#!/usr/bin/env python3
"""Reproduce the data-backed frequency-integration figures in the MPA guide.

The defaults point to the frozen MoS2 frequency artifacts used in the
2026-08-12 study.  Band energies, z samples, quadrature nodes, window bounds,
and measured residuals are read from those inputs.  The displayed plasmon
locations are deliberately illustrative; no pole store is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon


HERE = Path(__file__).resolve().parents[1]
DEFAULT_WFN = Path(
    "/home/jackm/lorrax_mpa_frequency_minimal/tests/regression/gnppm_debug/WFN.h5"
)
DEFAULT_CHI = Path(
    "/home/jackm/lorrax_mpa_frequency_plan/services/minimax/prototypes/"
    "chi_multiz_frozen_candidates/B15_cheb_1em04_00.json"
)
DEFAULT_SIGMA = Path(
    "/home/jackm/lorrax_sigma_crossing_unified_20260812/frozen/"
    "sigma_crossing_tier_1e4.json"
)
DEFAULT_NONCROSS = Path(
    "/home/jackm/lorrax_frontier_20260812/frozen_noncross_frontier.json"
)
DEFAULT_SHARED = Path("/tmp/shared_crossing_analysis.json")
DEFAULT_OUT = HERE / "docs/theory/figures"

RY_TO_EV = 13.605693122994
BLUE = "#2471a3"
RED = "#c0392b"
ORANGE = "#d97706"
PURPLE = "#6c3483"
GRAY = "#666666"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.15,
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _hx(value: str) -> float:
    return float.fromhex(value)


def _complex_pairs(rows) -> np.ndarray:
    return np.asarray([complex(_hx(a), _hx(b)) for a, b in rows])


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def _chi_data(wfn_path: Path, artifact_path: Path):
    art = _load(artifact_path)
    with h5py.File(wfn_path, "r") as handle:
        energies = np.asarray(handle["mf_header/kpoints/el"])[0]
        nv = int(np.asarray(handle["mf_header/kpoints/ifmax"])[0, 0])
    # This artifact used the logical 26-valence/20-conduction window.
    val = energies[:, :nv]
    cond = energies[:, nv:46]
    v_max = val.max()
    c_min = cond.min()
    b = (v_max - val).ravel()
    a = (cond - c_min).ravel()
    delta = (cond[:, :, None] - val[:, None, :]).ravel()
    p = art["physics_manifest"]
    z = _complex_pairs(p["z_values_ry"])
    t_hat = np.asarray([_hx(x) for x in art["representation"]["t_hat"]])
    h_hat = np.asarray([_hx(x) for x in art["representation"]["h_hat"]])
    scale = _hx(p["energy_scale_ry"])
    return {
        "A_ha": a,
        "B_ha": b,
        "delta_ha": delta,
        "gap_ha": c_min - v_max,
        "z_ry": z,
        "t_ry": t_hat / scale,
        "h_hat": h_hat,
        "artifact": art,
    }


def _ellipse_for_panel(left, right, rho):
    center = 0.5 * (left + right)
    half = 0.5 * (right - left)
    return center, half * 0.5 * (rho + 1.0 / rho), half * 0.5 * (
        rho - 1.0 / rho
    )


def _composite_panels(eta: float, freq_max: float, tol: float):
    """Same panel geometry and Bernstein bound as damped_line_rule."""
    t_max = np.log(2.0 / tol) / eta
    panel_target = 16.0 * 2.0 * np.pi / freq_max
    n_panels = max(1, int(np.ceil(t_max / panel_target)))
    width = t_max / n_panels
    half_bandwidth = (freq_max + eta) * width / 4.0
    eps_panel = 0.5 * (tol / eta) / n_panels
    rows = []
    for k in range(n_panels):
        left = k * width
        envelope = np.exp(-eta * left)
        order = int(np.ceil(half_bandwidth)) + 1
        while True:
            n = float(order)
            rho = (n + np.sqrt(n * n - half_bandwidth**2)) / half_bandwidth
            log_bound = (
                half_bandwidth * (rho - 1.0 / rho)
                - 2.0 * n * np.log(rho)
            )
            bound = (
                (64.0 / 15.0)
                * (0.5 * width)
                * envelope
                * np.exp(log_bound)
                / (1.0 - rho ** (-2.0 * n))
            )
            if bound <= eps_panel:
                break
            order += 1
        x, _ = np.polynomial.legendre.leggauss(order)
        nodes = left + 0.5 * width * (x + 1.0)
        rows.append((left, left + width, order, rho, bound, nodes))
    return t_max, rows


def plot_chi_geometry(data, out: Path) -> None:
    # BerkeleyGW WFN eigenvalues and the frozen artifact are both in Ry.
    a = data["A_ha"] * RY_TO_EV
    b = data["B_ha"] * RY_TO_EV
    gap = data["gap_ha"] * RY_TO_EV
    z = data["z_ry"] * RY_TO_EV
    dmin = data["delta_ha"].min() * RY_TO_EV
    dmax = data["delta_ha"].max() * RY_TO_EV

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    ax = axes[0]
    # Actual unique one-particle offsets, shown as the Cartesian product sampled by χ.
    aa, bb = np.meshgrid(np.unique(np.round(a, 10)), np.unique(np.round(b, 10)))
    delta = gap + aa + bb
    lo = np.real(z[3])
    near = np.real(z[5])
    sc = ax.scatter(
        aa,
        bb,
        c=delta,
        s=7,
        cmap="viridis",
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    for value, color, label in [
        (lo, ORANGE, rf"$\mathrm{{Re}}\,z={lo:.1f}$ eV"),
        (near, RED, rf"$\mathrm{{Re}}\,z={near:.1f}$ eV"),
    ]:
        # Eg + A + B = Re z: the resonant ridge in this plane.
        x = np.linspace(max(0.0, value - gap - b.max()), min(a.max(), value - gap), 200)
        y = value - gap - x
        ax.plot(x, y, color=color, lw=1.6, label=label)
    ax.set(xlabel=r"$A=E_c-E_{c,min}$ (eV)", ylabel=r"$B=E_{v,max}-E_v$ (eV)")
    ax.set_title("actual transition window")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.025)
    cbar.set_label(r"$\Delta=E_g+A+B$ (eV)")
    _panel_label(ax, "a")

    ax = axes[1]
    # Shade the actual transition band on the real axis.
    ax.fill_between([dmin, dmax], [0, 0], [0.09, 0.09], color="#777", alpha=0.18)
    ax.plot([dmin, dmax], [0, 0], color="#333", lw=4, solid_capstyle="butt")
    near_line = z[:7]
    far_line = z[7:]
    ax.scatter(near_line.real, near_line.imag, color=RED, s=20, zorder=4, label="near line")
    ax.scatter(far_line.real, far_line.imag, color=BLUE, s=20, zorder=4, label="far line")
    chosen = [near_line[3], far_line[3]]
    for zz, color in zip(chosen, [RED, BLUE], strict=True):
        ax.plot([zz.real, zz.real], [0, zz.imag], color=color, ls=(0, (2, 2)), lw=0.8)
        ax.annotate(
            rf"$z={zz.real:.1f}+{zz.imag:.1f}i$",
            xy=(zz.real, zz.imag),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
            fontsize=7,
        )
    ax.set_xlim(-1.0, dmax + 4.0)
    ax.set_ylim(-0.15, max(far_line.imag) * 1.25)
    ax.set(xlabel=r"$\mathrm{Re}\,z$ (eV)", ylabel=r"$\mathrm{Im}\,z$ (eV)")
    ax.set_title("actual MPA request points")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(
        0.98,
        0.05,
        rf"transition band $[{dmin:.1f},{dmax:.1f}]$ eV",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555",
    )
    _panel_label(ax, "b")

    ax = axes[2]
    t = data["t_ry"]
    weights = data["h_hat"]
    size = 8.0 + 40.0 * np.sqrt(weights / weights.max())
    ax.scatter(np.zeros_like(t), t, s=size, color=PURPLE, alpha=0.55, linewidths=0)
    ax.scatter(np.zeros_like(t), -t, s=size, color=PURPLE, alpha=0.55, linewidths=0)
    ax.axvline(0, color="#444", lw=0.6)
    ax.set_yscale("symlog", linthresh=0.2, linscale=0.7)
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0])
    ax.set_xticklabels([r"$\mathrm{Re}\,\tau=0$"])
    ax.set_ylabel(r"$\mathrm{Im}\,\tau$ (Ry$^{-1}$)")
    ax.set_title("frozen rank-154 sine dictionary")
    ax.text(
        0.04,
        0.04,
        r"154 positive times" "\n" r"308 executed $\pm it$ nodes" "\n" r"marker area $\propto h_\ell$",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    _panel_label(ax, "c")
    fig.suptitle(r"Direct $\chi^0(z)$ evaluation: actual MoS$_2$ bands and sample grid", y=1.015)
    fig.tight_layout()
    _save(fig, out, "mpa_chi_geometry")


def plot_chi_bernstein(data, out: Path) -> None:
    eta = float(np.imag(data["z_ry"][0]))
    omega_max = float(np.max(np.real(data["z_ry"])))
    delta_max = float(np.max(data["delta_ha"]))
    freq_max = omega_max + delta_max
    tol = 1.0e-4
    t_max, panels = _composite_panels(eta, freq_max, tol)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.8), gridspec_kw={"height_ratios": [1.0, 1.4]})
    ax = axes[0]
    for k, (left, right, order, rho, bound, nodes) in enumerate(panels):
        center, ar, ai = _ellipse_for_panel(left, right, rho)
        ax.add_patch(
            Ellipse(
                (center, 0),
                2 * ar,
                2 * ai,
                fill=False,
                ec=plt.cm.viridis(k / max(len(panels) - 1, 1)),
                lw=1.0,
                alpha=0.8,
            )
        )
        ax.scatter(nodes, np.zeros_like(nodes), s=5, color="#222", alpha=0.55)
        ax.plot([left, right], [0, 0], color="#222", lw=1.5)
        ax.text(center, -0.35, rf"$n={order}$", ha="center", va="top", fontsize=6.5)
    ax.axvline(t_max, color=RED, ls="--", lw=0.9)
    ax.set_xlim(-1.5, t_max + 1.5)
    max_ai = max(_ellipse_for_panel(p[0], p[1], p[3])[2] for p in panels)
    ax.set_ylim(-1.18 * max_ai, 1.18 * max_ai)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel=r"$\mathrm{Re}\,t$ (Ry$^{-1}$)", ylabel=r"$\mathrm{Im}\,t$ (Ry$^{-1}$)")
    ax.set_title(r"Bernstein ellipses for $\eta=0.2$ Ry")
    ax.text(
        0.99,
        0.96,
        rf"$F_{{\max}}={freq_max:.2f}$ Ry,  $t_{{\max}}={t_max:.2f}$ Ry$^{{-1}}$\n"
        rf"{len(panels)} panels, {sum(len(p[-1]) for p in panels)} GL nodes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
    )
    _panel_label(ax, "a")

    ax = axes[1]
    tt = np.linspace(0, t_max, 3000)
    for zz, color, label in [
        (data["z_ry"][3], RED, "near sample"),
        (data["z_ry"][10], BLUE, "far sample"),
    ]:
        envelope = np.exp(-zz.imag * tt)
        ax.semilogy(tt, envelope, color=color, label=rf"{label}: $\eta={zz.imag:.1f}$ Ry")
    ax.axhline(tol / 2.0, color="#777", ls=(0, (3, 2)), lw=0.8, label=r"tail scale $\epsilon/2$")
    for left, right, *_ in panels:
        ax.axvline(left, color="#aaa", lw=0.35)
    ax.axvline(t_max, color=RED, ls="--", lw=0.9)
    ax.set_xlim(0, t_max)
    ax.set_ylim(1e-45, 2)
    ax.set(xlabel=r"$t$ (Ry$^{-1}$)", ylabel=r"$|e^{izt}|=e^{-\eta t}$")
    ax.set_title("same panels, different line damping")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _panel_label(ax, "b")
    fig.suptitle("Actual near-line composite construction and its analytic contours", y=0.995)
    fig.tight_layout()
    _save(fig, out, "mpa_chi_bernstein")


def _diamond(a0, a1, b0, b1):
    return [(a + b, b - a) for a, b in [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]]


def plot_sigma_geometry(sigma: dict, noncross: dict, shared: dict, out: Path) -> None:
    xi = _hx(sigma["hgl"]["xi_ry"])
    threshold = _hx(shared["compatibility_hgl"]["groups"][0]["window_identity"]["mask_B_threshold"])
    omega_max = threshold - 2.0 * xi
    exact_rank = int(sigma["exact"]["rank_per_branch"])
    # D609 retained the accepted rank-48 HGL rule; the looser rank-21 trial
    # in sigma_crossing_tier_1e4 was explicitly rejected by the QP gate.
    hgl_rank = int(shared["compatibility_hgl"]["rule"]["rank"])
    side_nodes = int(sigma["sides"]["executed_spatial_dispatches"])
    nc_nodes = int(noncross["cases"]["uniform_1e-04"]["four_pole_executed_spatial_tau_contractions"])

    # Only these representative pole centers are schematic.  Every boundary and
    # node count comes from the frozen D609 artifacts.
    ea = np.array([0.025, 0.12, 0.28, 0.45, 0.70, 1.30, 2.30])
    ap = np.array([0.020, 0.11, 0.25, 0.44, 0.72, 1.45, 2.60])
    gam = np.array([0.010, 0.030, 0.060, 0.12, 0.20, 0.34, 0.48])
    limit = max(ea.max(), ap.max()) + 0.25

    fig = plt.figure(figsize=(9.8, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.0, 0.65], width_ratios=[1.28, 1.0], hspace=0.08, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2 = fig.add_subplot(gs[:, 1])

    windows = [
        (0, omega_max, 0, omega_max, RED, 0.16, "crossing core"),
        (omega_max, limit, 0, omega_max, BLUE, 0.12, "high-$E_A$ stripe"),
        (0, limit, omega_max, limit, BLUE, 0.10, "deep-pole slab"),
    ]
    for a0, a1, b0, b1, color, alpha, _label in windows:
        poly = Polygon(_diamond(a0, a1, b0, b1), closed=True, fc=color, ec=color, alpha=alpha, lw=0.8)
        ax.add_patch(poly)
        cu, cv = 0.5 * (a0 + a1) + 0.5 * (b0 + b1), 0.5 * (b0 + b1) - 0.5 * (a0 + a1)

    for e in ea:
        ax.plot([e + ap.min(), e + ap.max()], [ap.min() - e, ap.max() - e], color="#777", lw=0.32, alpha=0.35)
    for p in ap:
        ax.plot([ea.min() + p, ea.max() + p], [p - ea.min(), p - ea.max()], color="#777", lw=0.32, alpha=0.35)
    for i, e in enumerate(ea):
        for j, p in enumerate(ap):
            u, v = e + p, p - e
            crossing = e <= omega_max and p <= omega_max
            ax.scatter(u, v, s=15, color=RED if crossing else BLUE, edgecolor="white", linewidth=0.3, zorder=5)
            if crossing:
                narrow = gam[j] < xi
                ax.scatter(u, v, s=28, facecolor="none", edgecolor=PURPLE if narrow else RED, linewidth=0.7, zorder=6)

    for w, color in [(0.35 * omega_max, ORANGE), (0.85 * omega_max, RED)]:
        ax.axvline(w, color=color, ls=(0, (3, 2)), lw=1.0)
        ax_strip.axvline(w, color=color, ls=(0, (3, 2)), lw=1.0)
        ax.text(w, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, "", color=color)
    ax.axvline(omega_max, color="#777", ls="--", lw=0.7)
    ax.set_xlim(-0.1, 2 * limit + 0.15)
    ax.set_ylim(-limit - 0.1, limit + 0.1)
    ax.set_ylabel(r"$a_p-E_A$ (Ry)")
    ax.set_title(r"Rotated window plane: horizontal $S=E_A+a_p$")
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(
        handles=[
            Patch(fc=RED, alpha=0.18, ec=RED, label="crossing core"),
            Patch(fc=BLUE, alpha=0.13, ec=BLUE, label="sign-definite sides"),
            Line2D([], [], marker="o", ls="", mfc="none", mec=PURPLE, label=r"narrow $\Gamma<\xi$"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )
    _panel_label(ax, "a")

    ax_strip.axhline(0, color="#aaa", lw=0.7)
    sums = (ea[:, None] + ap[None, :]).ravel()
    ax_strip.plot(sums, np.zeros_like(sums), "|", color="#444", ms=8, mew=0.75)
    ax_strip.axvspan(0, omega_max, color=RED, alpha=0.06)
    ax_strip.set_ylim(-0.55, 0.55)
    ax_strip.set_yticks([])
    ax_strip.set_xlabel(r"$S=E_A+a_p$ (Ry); crossing when $S$ enters the requested $|\omega|$ range")
    for spine in ["left", "right", "top"]:
        ax_strip.spines[spine].set_visible(False)

    # Denominator plane for two real frequencies and illustrative fitted poles.
    x = np.linspace(-omega_max, limit * 1.35, 350)
    gamma = np.geomspace(0.006, 0.55, 180)
    xx, gg = np.meshgrid(x, gamma)
    mag = 1.0 / np.sqrt(xx**2 + gg**2)
    mesh = ax2.pcolormesh(
        xx,
        gg,
        np.log10(mag),
        shading="auto",
        cmap="magma",
        alpha=0.72,
        rasterized=True,
    )
    ax2.axhspan(0, xi, color=PURPLE, alpha=0.13, label=r"HGL: $\Gamma<\xi$")
    ax2.axhline(xi, color=PURPLE, ls="--", lw=1.0)
    for w, color, name in [(0.35 * omega_max, ORANGE, r"$\omega_1$"), (0.85 * omega_max, RED, r"$\omega_2$")]:
        d = w - (ea[:, None] + ap[None, :])
        gg_points = np.broadcast_to(gam[None, :], d.shape)
        ax2.scatter(
            d.ravel(),
            gg_points.ravel(),
            s=12,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.2,
            label=name,
        )
    ax2.set_yscale("log")
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(gamma.min(), gamma.max())
    ax2.axvline(0, color="#222", lw=0.8)
    ax2.set(xlabel=r"$u=\omega-(E_A+a_p)$ (Ry)", ylabel=r"pole width $\Gamma_p$ (Ry)")
    ax2.set_title("same crossing decision in the denominator plane")
    cbar = fig.colorbar(mesh, ax=ax2, fraction=0.045, pad=0.025)
    cbar.set_label(r"$\log_{10}|1/(u+i\Gamma)|$")
    ax2.legend(frameon=False, fontsize=7, loc="upper right")
    ax2.text(
        0.02,
        0.03,
        "actual D609 counts\n"
        rf"noncrossing {nc_nodes}" "\n"
        rf"exact crossing $2\times{exact_rank}$" "\n"
        rf"HGL $2\times{hgl_rank}$" "\n"
        rf"sides {side_nodes}" "\n"
        "total 609",
        transform=ax2.transAxes,
        va="bottom",
        fontsize=7,
        bbox=dict(fc="white", ec="none", alpha=0.78, pad=2),
    )
    _panel_label(ax2, "b")
    fig.suptitle(
        rf"$\Sigma_c$ windows from actual thresholds: $\omega_{{\max}}={omega_max:.3f}$ Ry, "
        rf"$\xi={xi:.4f}$ Ry (pole positions illustrative)",
        y=0.99,
    )
    _save(fig, out, "mpa_sigma_windows")


def plot_sigma_nodes(sigma: dict, noncross: dict, shared: dict, out: Path) -> None:
    nc_rows = noncross["cases"]["uniform_1e-04"]["classes"]
    families = {
        "noncrossing": np.concatenate([_complex_pairs(row["plan_t"]) for row in nc_rows.values()]),
        "exact crossing": _complex_pairs(sigma["exact"]["plan_t"]),
        "accepted HGL": _complex_pairs(shared["compatibility_hgl"]["rule"]["plan_t"]),
        "crossing sides": np.concatenate(
            [_complex_pairs(row["plan_t"]) for row in sigma["sides"]["rules"]]
        ),
    }
    colors = {
        "noncrossing": BLUE,
        "exact crossing": RED,
        "accepted HGL": PURPLE,
        "crossing sides": ORANGE,
    }
    executed = {
        "noncrossing": int(
            noncross["cases"]["uniform_1e-04"]["four_pole_executed_spatial_tau_contractions"]
        ),
        "exact crossing": 2 * int(sigma["exact"]["rank_per_branch"]),
        "accepted HGL": 2 * int(shared["compatibility_hgl"]["rule"]["rank"]),
        "crossing sides": int(sigma["sides"]["executed_spatial_dispatches"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    markers = {"noncrossing": "o", "exact crossing": "s", "accepted HGL": "D", "crossing sides": "^"}
    for name, tau in families.items():
        ax.scatter(
            tau.real,
            tau.imag,
            s=16,
            marker=markers[name],
            color=colors[name],
            alpha=0.56,
            linewidths=0,
            label=name,
        )
    ax.axhline(0, color="#777", lw=0.5)
    ax.axvline(0, color="#777", lw=0.5)
    ax.set_xscale("symlog", linthresh=0.03, linscale=0.7)
    ax.set_yscale("symlog", linthresh=0.03, linscale=0.7)
    ax.set_xlim(-0.02, 250)
    ax.set_ylim(-160, 1.2)
    ax.set(xlabel=r"$\mathrm{Re}\,\tau$ (Ry$^{-1}$)", ylabel=r"$\mathrm{Im}\,\tau$ (Ry$^{-1}$)")
    ax.set_title("actual D609 contour-node locations")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _panel_label(ax, "a")

    ax = axes[1]
    levels = np.arange(len(families))[::-1]
    for y, (name, tau) in zip(levels, families.items(), strict=True):
        ax.scatter(np.abs(tau), np.full(tau.size, y), s=15, color=colors[name], alpha=0.62, linewidths=0)
        ax.text(
            0.98,
            y + 0.20,
            f"{tau.size} stored / {executed[name]} executed",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color=colors[name],
        )
    ax.set_xscale("log")
    ax.set_yticks(levels, list(families))
    ax.set_ylim(-0.55, len(families) - 0.25)
    ax.set(xlabel=r"$|\tau|$ (Ry$^{-1}$)", title="node scales and contraction multiplicity")
    _panel_label(ax, "b")
    fig.suptitle(r"The 609-node $\Sigma_c$ plan: frozen quadrature data, not schematic nodes", y=1.015)
    fig.tight_layout()
    _save(fig, out, "mpa_sigma_nodes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wfn", type=Path, default=DEFAULT_WFN)
    parser.add_argument("--chi-artifact", type=Path, default=DEFAULT_CHI)
    parser.add_argument("--sigma-artifact", type=Path, default=DEFAULT_SIGMA)
    parser.add_argument("--noncross-artifact", type=Path, default=DEFAULT_NONCROSS)
    parser.add_argument("--shared-artifact", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chi = _chi_data(args.wfn, args.chi_artifact)
    sigma = _load(args.sigma_artifact)
    noncross = _load(args.noncross_artifact)
    shared = _load(args.shared_artifact)
    plot_chi_geometry(chi, args.output)
    plot_chi_bernstein(chi, args.output)
    plot_sigma_geometry(sigma, noncross, shared, args.output)
    plot_sigma_nodes(sigma, noncross, shared, args.output)
    print(f"wrote figures to {args.output}")


if __name__ == "__main__":
    main()
