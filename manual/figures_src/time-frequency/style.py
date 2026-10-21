"""Shared look of the time-frequency figures (docs/theory/minimax.md).

Palette: the flowchart kit (_design_src/docs-kit/figure-kit/STYLE.md) and the
site tokens (docs/stylesheets/lorrax.css): navy ink, one red accent, greys.
Type: IBM Plex Sans (the site's body face), bundled in ./fonts.
Output: white background, ~2000 px wide, into docs/assets/figures/.
"""
import os

import matplotlib as mpl
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "assets", "figures"))

NAVY = "#1f2a44"      # all ink, main series (LORRAX)
RED = "#d8322a"       # the one accent: the thing the figure is about
SOFT = "#3d4760"      # secondary text
GREY = "#6b7894"      # labels, generic/naive series
IDLE = "#a3abbb"      # faint lines
RULE = "#d9dee6"      # panel edges, grid
PANEL = "#f6f7f9"     # shaded regions
STRIP = "#e8ebf0"
BLUE = "#2c4f8f"      # site link blue, used sparingly as a second tone of navy

WIDTH = 9.2           # inches; at DPI below this is ~2020 px
DPI = 220

for name in ("Regular", "Italic", "Medium", "SemiBold"):
    font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f"IBMPlexSans-{name}.ttf"))

mpl.rcParams.update({
    "font.family": "IBM Plex Sans",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 10.5,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": NAVY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "text.color": NAVY,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "mathtext.fontset": "custom",
    "mathtext.rm": "IBM Plex Sans",
    "mathtext.it": "IBM Plex Sans:italic",
    "mathtext.bf": "IBM Plex Sans:semibold",
    "mathtext.sf": "IBM Plex Sans",
    "mathtext.fallback": "stixsans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
})


def save(fig, name):
    """Write docs/assets/figures/theory-timefreq-<name>.png."""
    path = os.path.join(OUT, f"theory-timefreq-{name}.png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    print("wrote", path)
    return path


def panel_label(ax, text, x=-0.01, y=1.02):
    """Bold panel tag in the title line, e.g. '(a)'."""
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11.5, fontweight="semibold", color=NAVY)
