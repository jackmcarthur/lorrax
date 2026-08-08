"""Multipole-W (MPA) staging area — the fit stage's tiling and schedule.

STAGING LOCATION, SAID ONCE AT THE PACKAGE LEVEL SO EVERY MODULE
INHERITS IT.  ``gw/mpa/`` and ``file_io/mpa_store.py`` are the two
halves of the multipole-W infrastructure landed ahead of the fit
itself, and both are deliberately dependency-light and MOVABLE: when
the MPA stage settles they are expected to move as a pair, most likely
into a service alongside ``symmetry_maps``.  Nothing in this package
imports jax, and its one dependency on the tree is
``file_io.mpa_store`` — the established direction (``gw`` reads
``file_io``, never the reverse).

WHAT LIVES WHERE.  The BYTES are ``file_io.mpa_store``: the
frequency-resolved W schema, the column reader and its budget, the
staged B/Ω store.  The SCHEDULE is here: how the fit driver walks the
(q, ν-column) grid, what one step is allowed to hold, and the refusal
that keeps the walk 1-D sharded on the row axis.
"""

from gw.mpa.tiling import (column_blocks, fit_schedule, plan_column_walk,
                           row_shard_spec)

__all__ = ["column_blocks", "fit_schedule", "plan_column_walk",
           "row_shard_spec"]
