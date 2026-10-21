"""Read-only access to LORRAX's quadrature service for the figures.

The figures call the rule builders of LORRAX's ``services/minimax`` package
(pure NumPy/SciPy: no driver, no JAX, no material data) from the read-only
checkout at LORRAX_SRC (default ~/projects/lorrax_main, main be618924).
Nothing is written there: bytecode writing is switched off before the import.

    noncrossing_levelled  services/minimax/src/minimax/levelled.py:316   static 1/x rule
    crossing_nodes        services/minimax/src/minimax/analytic_box.py:110  bent contour
    sector_degree         services/minimax/src/minimax/analytic_box.py:205  sector law
    analytic_box_rule     services/minimax/src/minimax/analytic_box.py:335  certified rule
"""
import os
import sys

sys.dont_write_bytecode = True
LORRAX_SRC = os.environ.get("LORRAX_SRC", os.path.expanduser("~/projects/lorrax_main"))
sys.path.insert(0, os.path.join(LORRAX_SRC, "services", "minimax", "src"))

from minimax.levelled import noncrossing_levelled  # noqa: E402
from minimax.analytic_box import (analytic_box_rule, crossing_nodes,  # noqa: E402
                                  sector_degree)
from minimax.laplace_ritz import place_times  # noqa: E402

__all__ = ["noncrossing_levelled", "analytic_box_rule", "crossing_nodes",
           "sector_degree", "place_times"]
