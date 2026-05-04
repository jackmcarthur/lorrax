"""
DEPRECATED back-compat shim — re-exports the public V_q surface from the
post-refactor modules.

The 2000+ line ``compute_vcoul.py`` was split (commits f6a3400, 02b333d) into:

    * ``gw.coulomb_kernel`` — pure (q+G)-space Coulomb physics: FFT-grid
      shape helpers, ``compute_sqrt_vcoul_0d``, ``make_v_munu_chunked_kernel``
      (sphere_idx + the ``get_sqrt_v_and_phase`` factory + MC head averaging).
    * ``gw.v_q_tile``      — unified per-tile V_q kernel
      ``_make_V_q_tile_kernel`` + outer ``compute_V_q_tile`` driver +
      memory chooser ``_choose_v_q_chunks``.  The kernel now handles both
      the legacy Case A (full μ × full μ) and Case B (μ × ν tiled) plus
      the upcoming bispinor V^{μ_L, ν_L} (distinct ζ_L / ζ_R) instantiation
      in one cached jit.
    * ``gw.v_q_driver``    — outer drivers ``compute_all_V_q_sharded``
      (production), ``compute_all_V_q_from_zeta_h5`` (replicated fallback),
      and the legacy single-q helpers.

Importers should switch to the new modules.  This shim re-exports the
public symbols at their old names so external callers (notebooks, ad-hoc
scripts) keep working without edits.

If you are reading this because something imported from ``compute_vcoul``
and got an ``AttributeError``: the symbol you wanted was probably an
internal helper (prefixed ``_``) that did not survive the split.  Grep
the new modules for it.
"""

from .coulomb_kernel import (
    compute_sqrt_vcoul_0d,
    exp_ikr_fftbox,
    fft_integer_axes,
    make_v_munu_chunked_kernel,
)
from .v_q_driver import (
    compute_all_V_q_from_zeta_h5,
    compute_all_V_q_sharded,
    compute_V_q_from_zeta_array,
    compute_V_q_from_zeta_h5,
    read_zeta_q_sharded,
)
from .v_q_tile import (
    _choose_v_q_chunks,
    _make_V_q_tile_kernel,
    compute_V_q_tile,
)

__all__ = [
    # coulomb_kernel
    "compute_sqrt_vcoul_0d",
    "exp_ikr_fftbox",
    "fft_integer_axes",
    "make_v_munu_chunked_kernel",
    # v_q_tile
    "compute_V_q_tile",
    "_choose_v_q_chunks",
    "_make_V_q_tile_kernel",
    # v_q_driver
    "compute_all_V_q_sharded",
    "compute_all_V_q_from_zeta_h5",
    "compute_V_q_from_zeta_h5",
    "compute_V_q_from_zeta_array",
    "read_zeta_q_sharded",
]
