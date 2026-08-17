"""The fit stage's walk over (q, ν-column) blocks, and its memory rule.

STAGING LOCATION: see ``gw/mpa/__init__.py``.  This module is pure
arithmetic on ints — no jax, no h5py, no numpy required for the
schedule itself — so it can be reasoned about and tested without a
file or a device, and moved without either.

THE CONSTRAINT THIS MODULE ENCODES, in the owner's terms.  A small
number of W_q(μ,ν) copies fit in memory at once, but NOT all ω_i.  So:

* every frequency is on DISK, with the frequency axis leading
  (``file_io.mpa_store``);
* the fit reads SUB-TILES — a few ν columns across all ω — sized so
  that one block costs about ONE (N_μ, N_μ) tile;
* the block is 1-D SHARDED ON THE ROW AXIS ONLY, never 2-D;
* B_q and Ω_q go to disk STAGED, as each block's fit completes, rather
  than accumulating until the last block finishes.

The last point is not symmetry with the first: the fit's OUTPUT is
larger than its input.  A block of C columns reads
``n_omega * N_μ * C`` complex numbers and produces ``2 * n_p * N_μ * C``
of them (Ω_p and B_p per element per pole), so for n_p = 8 and
n_omega = 16 the poles outweigh the samples by exactly the factor
``2 n_p / n_omega`` = 1.  At the top of the Cu schedule (n_p = 12,
n_omega = 16) they outweigh them by 1.5x.  Accumulating them is
therefore strictly worse than accumulating the input, which is the
thing already declared not to fit.

WHY THE COLUMN AXIS AND NOT THE ROW AXIS.  W_q(μ,ν) is stored
row-major in (μ, ν), so a column block is a strided read and a row
block is contiguous — the row block looks like the better choice until
the sharding is written down.  The fit is elementwise, the rows are
what gets split across devices, and a row-block walk would put the
sharded axis and the walked axis on top of each other: each step's
rows would live on one rank while the rest idled.  Walking the columns
keeps the row axis whole and shardable at every step, which is what
makes the block's cost a function of the caller's budget rather than of
the mesh shape.
"""

from __future__ import annotations

from file_io.mpa_store import (choose_column_budget, describe_column_cost,
                               one_tile_bytes)

__all__ = ["choose_column_budget", "column_blocks", "describe_column_cost",
           "fit_schedule", "one_tile_bytes", "plan_column_walk",
           "row_shard_spec"]

def row_shard_spec(mesh_axis="x"):
    """The ONLY sharding a column block may carry: rows over one axis.

    Returns the ``PartitionSpec`` entries as a plain tuple —
    ``(None, mesh_axis, None)`` — so a caller can build
    ``NamedSharding(mesh, P(*row_shard_spec()))`` without this module
    importing jax.  Mesh axes are referred to BY NAME and never by
    shape, per AGENTS.md; the default ``'x'`` is the tree's first mesh
    axis and the argument exists so a caller on a differently-named
    mesh does not have to hand-build the tuple and get the position
    wrong.

    ``mpa_store.read_w_columns(out_spec=...)`` checks any spec it is
    given against this shape and refuses a 2-D one by name.
    """
    return (None, mesh_axis, None)


def column_blocks(n_mu, n_cols):
    """The ν-column ranges the fit walks, as ``(start, stop)`` pairs.

    Contiguous and in order, because a contiguous range is one HDF5
    hyperslab and a scattered one is a point selection — same bytes,
    several times the metadata — and because the completion ledger
    records ranges, which a scattered block cannot be.

    The last block is SHORT rather than overlapping: ``N_μ = 480`` in
    blocks of 30 is 16 exact blocks, but ``N_μ = 500`` is 16 blocks of
    30 and one of 20.  Overlapping to keep the width uniform would fit
    twenty columns twice and hand the staged store a double write,
    which it refuses.
    """
    n_mu = int(n_mu)
    n_cols = int(n_cols)
    if n_mu < 1:
        raise ValueError(f"column_blocks: n_mu must be >= 1; got {n_mu}")
    if n_cols < 1:
        raise ValueError(
            f"column_blocks: n_cols must be >= 1; got {n_cols}.  A "
            f"zero-width block is not a smaller step, it is a walk that "
            f"never finishes.")
    return [(lo, min(lo + n_cols, n_mu))
            for lo in range(0, n_mu, n_cols)]


def plan_column_walk(n_mu, n_omega, tile_bytes=None):
    """The budget, the blocks, and what one step costs — as a dict.

    One call a driver can log at startup so the memory decision is on
    the record BEFORE the run rather than in a traceback after it.
    """
    n_cols = choose_column_budget(n_mu, n_omega, tile_bytes)
    blocks = column_blocks(n_mu, n_cols)
    return {
        "n_mu": int(n_mu),
        "n_omega": int(n_omega),
        "n_cols": int(n_cols),
        "n_blocks": len(blocks),
        "blocks": blocks,
        "tile_bytes": one_tile_bytes(n_mu) if tile_bytes is None
        else int(tile_bytes),
        "cost": describe_column_cost(n_mu, n_omega, n_cols, tile_bytes),
        "row_shard_spec": row_shard_spec(),
    }


def fit_schedule(n_q, n_mu, n_omega, tile_bytes=None):
    """``(q, col_start, col_stop)`` steps, q-major.

    Q-MAJOR AND NOT COLUMN-MAJOR.  Both orders visit the same steps,
    but the q axis is the one the read is indexed on — a column block
    is ``ds[:, q, :, lo:hi]`` — so finishing a q before moving on keeps
    each HDF5 chunk in cache for the whole of its column walk instead
    of touching every q's chunks once per block.  It is also the order
    the staged store's completion ledger is easiest to read in: a
    crashed run leaves a prefix of whole q's plus one partial q, which
    ``_ranges`` prints as one line.
    """
    blocks = column_blocks(n_mu, choose_column_budget(n_mu, n_omega,
                                                      tile_bytes))
    return [(q, lo, hi) for q in range(int(n_q)) for lo, hi in blocks]
