"""Multipole-W (MPA) staging area — the fit kernel, the schedule, the driver.

STAGING LOCATION, SAID ONCE AT THE PACKAGE LEVEL SO EVERY MODULE
INHERITS IT.  ``gw/mpa/`` and ``file_io/mpa_store.py`` are the two
halves of the multipole-W infrastructure landed ahead of the
minimax-service design review, and both are deliberately
dependency-light and MOVABLE: when the MPA stage settles they are
expected to move as a pair, most likely into a service alongside
``symmetry_maps``.  Placement here is a parking spot, NOT a ruling.

WHAT LIVES WHERE.  The BYTES are ``file_io.mpa_store``: the
frequency-resolved W schema, the column reader and its budget, the
staged B/Ω store.  The SCHEDULE is ``tiling``: how the fit driver walks
the (q, ν-column) grid, what one step is allowed to hold, and the
refusal that keeps the walk 1-D sharded on the row axis.  The FIT is
``sampling`` + ``pade_fit`` + ``diagnostics``: the double-parallel
sample grid, the normalised Padé-in-z² solve with its four ordered
guards, and the conditioning/held-out/perturbation harness.  The
DRIVER is ``fit_driver``: the loop that reads a block, fits it, writes
it, and finalizes — the only module here that touches all of the above.

Contents
--------
``sampling``
    The double-parallel sample-grid constructor (two horizontal lines
    in the complex-frequency plane, semi-homogeneous powers-of-two real
    partition, nested in ``n_p``).
``pade_fit``
    The fit kernel itself: normalised Padé-in-z² linear solve with
    z_max scaling, companion-matrix roots, all-2·n_p-point complex
    least-squares residues, four ordered guards, mandatory residue
    refit.
``diagnostics``
    Conditioning, backward error, held-out-sample residual, and the
    perturbation-refit propagation harness.
``tiling``
    The (q, ν-column) walk and its memory rule.
``fit_driver``
    Read → fit → write → finalize, plus the cost report.

WHY THE JAX HALF IS NOT EAGERLY IMPORTED HERE.  ``tiling`` is pure
arithmetic on ints and ``file_io.mpa_store`` is numpy-and-h5py, so
``import gw.mpa.tiling`` costs nothing and can be reasoned about
without a device.  ``pade_fit``, ``diagnostics`` and ``fit_driver``
import jax.  Re-exporting those at package scope would put a jax import
behind every ``import gw.mpa.tiling``, which is the property the
schedule half was written to keep.  They remain reachable the ordinary
way — ``from gw.mpa import pade_fit`` and ``import gw.mpa.pade_fit``
both work, because Python falls back to importing a submodule when the
package has no such attribute — so nothing is hidden, only deferred.
"""

from gw.mpa.tiling import (column_blocks, fit_schedule, plan_column_walk,
                           row_shard_spec)

__all__ = ["column_blocks", "fit_schedule", "plan_column_walk",
           "row_shard_spec"]
