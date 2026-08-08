"""COMPAT SHIM — parser for BerkeleyGW's ``vcoul`` file.

The parser moved to :mod:`vcoul.bgw_parity` (2026-08-07) — it was already
numpy-only, and "read BGW's MC-averaged v(q+G) and lay it on our FFT grid"
is Coulomb arithmetic, not a lorrax file format.  ``file_io/__init__.py``
re-exports all three names from here, so ``from file_io import
read_bgw_vcoul`` keeps working unchanged.

File format: one line per (q, G) entry with

    qx qy qz Gx Gy Gz vcoul

where qx,qy,qz are fractional k-point coords and Gx,Gy,Gz are integer
Miller indices.  BGW's vcoul value is ``<8π/|q+G+δq|²>_miniBZ``
(MC-averaged) in Rydberg units (before multiplication by
``fact = 1/(Nk·Ω)``).

This is useful to bypass LORRAX's approximate G=0-only mini-BZ average in
the 3D semiconductor case, when BGW's all-G MC averaging matters (small
k-grids).

REGISTERED, NOT TAKEN: ``BGWVcoulTable.find_q_index``'s April fix never
landed and the ``use_bgw_vcoul`` surface is data-dead in-tree.  The
repair-or-delete decision is the OWNER's (DESIGN_vcoul.md owner row 3);
the extraction moved this code verbatim and fixed nothing.
"""
from __future__ import annotations

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from vcoul import (                                         # noqa: E402,F401
    BGWVcoulTable,
    fill_v_grid_for_q,
    read_bgw_vcoul,
)

__all__ = ["BGWVcoulTable", "read_bgw_vcoul", "fill_v_grid_for_q"]
