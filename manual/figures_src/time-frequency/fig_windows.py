"""Fig. 4.2: product windows on the crossing branch, and what they buy.

Insulating example (eV, about E_F): conduction states E in [0.6, 100],
poles Omega in [1, 250], on-shell frequencies omega in [0.6, 10] on the
omega >= E_F half; eta = 0.25 (sigma_regularization_ev, src/gw/gw_config.py:1636),
eps = 1e-4 (sigma_quadrature_eps, :1622), edge a = 1.5 eta
(sigma_window_edge_factor, :1649).

Partition: gw.sigma_box_plan._state_products (src/gw/sigma_box_plan.py:214-248)
with Lambda = max|omega| + a + excursion (gw.mpa.sigma_windows.sigma_pole_edges,
src/gw/mpa/sigma_windows.py:87-110; excursion 0 on an insulator).
Boxes: corners omega - (E + Omega) padded as gw.sigma_box_plan._box (:268-276).
Rules: minimax.analytic_box_rule (services/minimax/src/minimax/analytic_box.py:336),
read-only, on the same boxes; "whole branch" is the one box the branch would
take without the partition.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

import style as S
from lorrax_minimax import analytic_box_rule

ETA, EPS = 0.25, 1e-4
A = 1.5 * ETA
E_LO, E_HI = 0.6, 100.0
P_LO, P_HI = 1.0, 250.0
W_LO, W_HI = 0.6, 10.0
LAM = W_HI + A


def pad_box(lo, hi):
    pad = 0.02 * max(hi - lo, ETA)
    lo2 = lo - pad if lo <= 0 else max(lo - pad, 0.7 * lo)
    hi2 = hi + pad if hi >= 0 else min(hi + pad, 0.7 * hi)
    return (lo2, hi2, ETA, ETA)


def box(e, p):
    corners = [w - (x + y) for w in (W_LO, W_HI) for x in e for y in p]
    return pad_box(min(corners), max(corners))


wins = {
    "resonant": box((E_LO, LAM), (P_LO, LAM)),
    "state tail": box((LAM, E_HI), (P_LO, LAM)),
    "pole tail": box((E_LO, E_HI), (LAM, P_HI)),
}
whole_box = box((E_LO, E_HI), (P_LO, P_HI))
rules = {k: analytic_box_rule(b, EPS) for k, b in wins.items()}
whole = analytic_box_rule(whole_box, EPS)
for k, r in rules.items():
    print(k, wins[k], r.times.size, r.sup_error)
print("whole", whole_box, whole.times.size, whole.sup_error)
n_win = sum(r.times.size for r in rules.values())

fig, ax = plt.subplots(figsize=(S.WIDTH, 3.9),
                       gridspec_kw=dict(left=0.075, right=0.985, top=0.84, bottom=0.14))

# ---- (a) the partition in the (E, Omega) plane
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.4, 160)
ax.set_ylim(0.6, 420)
ax.add_patch(Rectangle((E_LO, P_LO), LAM - E_LO, LAM - P_LO, fc=S.RED, alpha=0.13, ec=S.RED, lw=1.0))
ax.add_patch(Rectangle((LAM, P_LO), E_HI - LAM, LAM - P_LO, fc=S.BLUE, alpha=0.13, ec=S.NAVY, lw=0.8))
ax.add_patch(Rectangle((E_LO, LAM), E_HI - E_LO, P_HI - LAM, fc=S.BLUE, alpha=0.06, ec=S.NAVY, lw=0.8))
# where Re d actually changes sign: E + Omega <= omega_max
e = np.geomspace(E_LO, W_HI - P_LO, 200)
ax.fill_between(e, P_LO, np.maximum(W_HI - e, P_LO), color=S.RED, alpha=0.28, lw=0)
ax.plot(e, W_HI - e, color=S.RED, lw=1.0)
ax.text(0.68, 1.25, "under the curve: Re d = 0 at some ω", color="#8f1d17", fontsize=8.8)
ax.text(0.68, 3.4, f"resonant window\ncrossing, {rules['resonant'].times.size} nodes", color="#8f1d17",
        fontsize=9.3, fontweight="semibold", linespacing=1.2)
ax.text(14, 2.6, f"state tail\nsign-definite\n{rules['state tail'].times.size} nodes",
        color=S.NAVY, fontsize=9.3, linespacing=1.15)
ax.text(2.0, 60, f"pole tail:  sign-definite, {rules['pole tail'].times.size} nodes",
        color=S.NAVY, fontsize=9.3)
for v in (LAM,):
    ax.axvline(v, color=S.GREY, lw=0.6, ls=":")
    ax.axhline(v, color=S.GREY, lw=0.6, ls=":")
ax.text(LAM * 1.06, 300, "edge: highest read frequency + 1.5η", color=S.GREY, fontsize=9.0)
ax.set_xlabel(r"state energy $E - E_F$  (eV)", labelpad=2)
ax.set_ylabel("pole Ω  (eV)")
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1", "10", "100"])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1", "10", "100"])
ax.set_title("One branch of Σ cut into three product windows", x=-0.06, pad=22)
ax.text(-0.06, 1.025, f"together {n_win} nodes; one window for the branch: {whole.times.size}",
        transform=ax.transAxes, color=S.GREY, fontsize=9.0, va="bottom")

S.save(fig, "windows")

# ---- (b) per-term relative error, Re d < 0
fig, bx = plt.subplots(figsize=(S.WIDTH, 3.9),
                       gridspec_kw=dict(left=0.085, right=0.985, top=0.84, bottom=0.14))
x_all = -np.geomspace(0.3, 360, 6000)


def rel_err(rule, x):
    """Upper envelope (max per log bin, 20 bins a decade) of |d||Q - 1/d| on Im d = eta."""
    d = x + 1j * ETA
    Q = np.exp(1j * np.outer(d, rule.times)) @ rule.weights
    err = np.abs(d) * np.abs(Q - 1.0 / d)
    key = np.floor(20 * np.log10(np.abs(x)))
    out = np.empty_like(err)
    for k in np.unique(key):
        out[key == k] = err[key == k].max()
    return out


bx.axhline(EPS, color=S.SOFT, lw=0.8, ls=":")
bx.text(330, EPS * 1.25, r"$\varepsilon$ = sigma_quadrature_eps", ha="right", va="bottom", color=S.SOFT,
        fontsize=9.0)
bound = EPS * np.abs(x_all + 1j * ETA) / ETA
bx.plot(-x_all, bound, color=S.IDLE, lw=1.0, ls=(0, (4, 2)))
bx.text(40, 0.035, r"crossing bound $\varepsilon|d|/\eta$", color=S.GREY,
        fontsize=9.0, ha="right", va="bottom")
sel = (x_all >= whole_box[0]) & (x_all <= whole_box[1])
bx.plot(-x_all[sel], rel_err(whole, x_all[sel]), color=S.GREY, lw=1.3,
        label=f"one crossing window for the branch ({whole.times.size} nodes)")
first = True
for k, col in (("resonant", S.RED), ("state tail", S.NAVY), ("pole tail", S.NAVY)):
    b = wins[k]
    xs_ = x_all[(x_all >= b[0]) & (x_all <= b[1])]
    if k == "state tail":
        xs_ = xs_[xs_ <= -LAM + W_HI * 0]  # its members start below -(Lambda - omega_max)
    lab = None
    if k == "resonant":
        lab = f"resonant window, crossing ({rules[k].times.size} nodes)"
    elif first:
        lab = f"tail windows, sign-definite ({rules['state tail'].times.size} + {rules['pole tail'].times.size} nodes)"
        first = False
    bx.plot(-xs_, rel_err(rules[k], xs_), color=col, lw=1.4 if col == S.RED else 1.2,
            ls="-" if k != "pole tail" else (0, (4, 2)), label=lab)
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlim(0.3, 360)
bx.set_ylim(1e-7, 1.0)
bx.set_xlabel("|Re d|  (eV), denominators below resonance", labelpad=2)
bx.set_ylabel(r"relative error  $|d|\,|Q(d) - 1/d|$")
bx.set_title("What each term is certified to", x=-0.07, pad=22)
bx.text(-0.07, 1.025, "relative error of each term below resonance; far terms meet ε only in a sign-definite window", transform=bx.transAxes,
        color=S.GREY, fontsize=9.0, va="bottom")
bx.legend(loc="lower left", fontsize=8.8, handlelength=2.0, borderaxespad=0.3)
bx.grid(True, which="major", color=S.RULE, lw=0.6)
S.save(fig, "windows-error")
