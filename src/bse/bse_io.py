"""Compatibility facade for the BSE restart I/O, split 2026-08-10.

This module used to be all of it — loading, band-window authority, q=0 head
injection, coarse-to-fine densification and restart policy in one file that had
reached 2917 lines — and five separate lanes collided in it inside two days.
The code now lives in four modules, each with one responsibility and one
authority rule stated in its own docstring:

``bse_window``
    Which bands are in the window, and how its axes are padded.  The
    ``PAD_EPS_GUARD_RY`` sentinel, the pad masks and counts, the ``--eqp``
    re-slice, and the eigenvector writer's declared window.
``bse_head``
    The q=0 rank-1 Coulomb head: where its scalars come from, the ``w0_ready``
    gate that keeps a screened head off an unscreened tile, and the separate
    ``defer_whead`` question a pending densification asks.
``bse_densify``
    Coarse k-grid to fine: the exact real-space zero-pad, the one sharded
    densifier, C1's analytic per-fine-q head channel, and the whole-bundle
    interpolation the ``bse_k_grid`` key drives.
``bse_loading``
    Reading a GW restart into a BSE bundle: the readiness and q-wedge gates,
    the SlabIO / serial transports, and the two loaders.

NOTHING MOVED THAT CONSUMERS CAN SEE.  Every name this module exported before
the split is re-exported here and resolves to the same function object, so
every consumer's import is byte-identical to what it was.  The only import
lines that changed anywhere are two test GUARDS that reach in to patch or to
read the source of code that moved; a guard has to follow the code or it stops
measuring anything.

Retiring the facade — pointing the twenty-odd import sites at the real modules
and deleting this file — is a separate decision that has not been taken.  One
in-package consumer, ``bse/vq_interp.py``'s lazy ``from . import bse_io``, is
still on it deliberately: the lazy form is what keeps the two modules out of an
import cycle, and repointing it belongs with the retirement rather than ahead
of it.

Write new code against the module that owns the behaviour, not against this
file.
"""
from __future__ import annotations

from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY,
                                    check_band_window, resolve_band_window)

from .bse_densify import (W_HEAD_DENSIFY_MODES, _interpolate_bse_data_to_grid,
                          _parse_grid_spec, _read_lorrax_input_quietly,
                          _resolve_bse_k_grid, _zeropad_R_axis,
                          build_w_head_channel, decimate_W_q_to_subgrid,
                          make_w_densifier, pad_W_R_to_grid,
                          resolve_w_head_densify)
from .bse_head import (_inject_q0_head, _parse_head_overrides,
                       _resolve_head_params)
from .bse_loading import (_BARE_V_FALLBACK_WARNING, _MunuSlabPlan,
                          _assert_local_block, _bse_slabio_usable,
                          _find_restart_file, _get_local_axis_coords,
                          _get_local_mesh_coords, _load_ring_subset,
                          _pad_first_two_axes, _pad_last_axis,
                          _pad_last_two_axes, _read_bse_tensors,
                          _read_psi_mu_sharded, _read_vq0_sharded,
                          _read_wq_sharded, _refuse_unpersisted,
                          _resolve_munu_reader, _slabio_read_munu,
                          _slabio_read_psi, is_q_wedge,
                          load_bse_data_from_restart_sharded,
                          restart_munu_full_bz)
from .bse_window import (PAD_EPS_GUARD_RY, _generate_kpts_grid, _log0,
                         _pad_axis_to_multiple, _parse_wfn_path,
                         apply_eqp_and_reslice_bands, apply_eqp_corrections,
                         n_pad_transitions, pad_zone_mask, pad_zone_mask_np,
                         read_bgw_eqp, resolve_n_occ,
                         write_eigenvectors_stream)

__all__ = [
    "DEFAULT_MODE",
    "DEGENERACY_TOL_RY",
    "PAD_EPS_GUARD_RY",
    "W_HEAD_DENSIFY_MODES",
    "apply_eqp_and_reslice_bands",
    "apply_eqp_corrections",
    "build_w_head_channel",
    "check_band_window",
    "decimate_W_q_to_subgrid",
    "is_q_wedge",
    "load_bse_data_from_restart_sharded",
    "make_w_densifier",
    "n_pad_transitions",
    "pad_W_R_to_grid",
    "pad_zone_mask",
    "pad_zone_mask_np",
    "read_bgw_eqp",
    "resolve_band_window",
    "resolve_n_occ",
    "resolve_w_head_densify",
    "restart_munu_full_bz",
    "write_eigenvectors_stream",
]
