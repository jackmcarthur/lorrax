"""``vcoul`` — the bare and truncated Coulomb interaction, as pure math.

Everything LORRAX knows about ``v(q+G)``: the three truncation schemes,
the mini-BZ Voronoi averaging that regularises the q→0 head, the
bare-Coulomb sphere predicate, and the BerkeleyGW ``vcoul``-file parity
surface.  Numbers in, numbers out — no decks, no HDF5, no ``wfn`` loader,
no ``Meta``.  The deck keys, the head-resolver ladder and the file paths
stay in ``gw``; this package is what they compute WITH.

THE PACKAGE IS THE DOOR.  Everything a consumer needs is a top-level name
here; importing ``vcoul.<submodule>`` from lorrax is a layering violation
the monorepo's ``tests/test_layering.py`` fails on.

Dependencies: numpy and jax, plus **optional** scipy (the scrambled-Sobol
generator, quarantined in :mod:`vcoul.minibz` behind an announce-or-refuse
gate — see :func:`minibz_voronoi_batches`).  Nothing here imports lorrax,
which ``services/vcoul/tests/test_vcoul_import_isolation.py`` measures in
a process where lorrax is absent.

WHO CALLS WHAT
--------------
``get_kernel(sys_dim)`` → ``Bulk3D`` / ``Slab2D`` / ``Box0D``
    The dimension dispatch.  Each kernel implements exactly one
    arithmetic method, ``_v_bare_per_q``; everything
    dimension-INDEPENDENT (the per-q loop, the ``vcoul_cutoff_ry`` mask,
    the ``argmin |q+G|`` head-slot injection, the ``(n_q, ngkmax)``
    float64 contract) lives once in ``v_qG_table``.  So there is exactly
    one ``v(q+G)`` formula per dimensionality in the tree.
``v_qG_table(kernel, q_irr_frac, gvec_components, *, geometry, ...)``
    THE production ``v(q+G)`` driver.  ``gw.compute_vcoul.compute_v_q_per_G``
    is a thin dispatcher over it (reached from
    ``gw.v_q_g_flat.compute_all_V_q_g_flat`` and
    ``gw.v_q_bispinor``); ``v_qG_single`` is the single-q complex128
    wrapper over the same driver, which is what ``kernel.v_qG`` calls.
``CoulombGeometry(bvec, cell_volume, bdot=None, fft_grid=None)``
    The four numbers every routine here needs, with ``blat * bvec``
    taken ONCE (``from_wfn``) instead of at five call sites.
``q0_average(geometry, kgrid, ...)`` on any kernel
    The ``(vc0_mean, wcoul0)`` pair at q→0.  ``gw.vcoul.compute_q0_averages``
    is the deck-facing wrapper (called by ``gw.head_correction``'s
    HeadResolver in both the epshead and the anisotropic-S(ω) flavour).
``minibz_voronoi_batches`` / ``minibz_average`` / ``minibz_inscribed_sphere_r2``
    The BGW ``Common/minibzaverage.f90`` port.  Used by the 3D and 2D
    ``q0_average`` above, and directly by ``bse.vq_interp.minibz_head_vlr``
    for the 2D slab per-Q exchange head.
``minibz_moment_tensor(shift_cart, dq_batches, ...)``
    The SECOND moment ``M_ab = <v(q) q_a q_b>_cell`` on the same draws and
    the same two BGW branches as ``minibz_average`` — six numbers instead
    of one, which is what the BSE exchange head's cell average actually
    needs (``LT_HEAD_PROBLEM.md`` §3).  Bare units, like the scalar.
``iter_minibz_photon_samples(kernel, geometry, kgrid, ...)``
    Streams one Sobol replicate and one fixed-size chunk at a time, returning
    Cartesian q together with the raw bare Coulomb-gauge kernel in the fixed
    ``(C,Tx,Ty,Tz)`` basis.  It is the sample provider for a caller that must
    complete a coupled q0 solve before averaging; it contains no screened
    response physics and applies no cell-volume factor.
``build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume, ...)``
    The 3D body head as a FUNCTION of the Cartesian ``K = q+G``, which
    ``v_qG_table`` evaluates at every ``argmin |q+G|`` slot (all of them
    when the argmin ties, and they all get the tied set's mean).  One
    live call site, double-gated on ``mc_average_vcoul_body and
    sys_dim == 3``.  It replaced the per-q ``(nkx, nky, nkz)`` table
    ``build_v_head_miniBZ_avg_3d`` on 2026-08-08: a per-q scalar is
    attached to ``q_frac @ bvec``, which is not the argmin ``q+G`` on 12
    of Si's 64 q, so that shape could not express the fix.
``build_miniBZ_dq_cart(kgrid, bvec, ...)``
    The centrosymmetric ``{δq} = {−δq}`` mini-BZ draw the head above
    averages over — on the door because the closure is a correctness
    requirement of the injection, not an implementation detail.
``bare_coulomb_sphere_indices`` / ``bare_coulomb_sphere_mask``
    "Which G are in q's sphere".  ``common.coulomb_sphere`` pads the
    answer into the WFN.h5 sentinel layout; the padding is the FFT box's
    business, not Coulomb's.
``compute_vcoul_box``
    The Wigner-Seitz cell-box FFT (BGW ``Common/trunc_cell_box.f90``),
    the 0-D kernel's arithmetic.  numpy only.
``read_bgw_vcoul`` / ``fill_v_grid_for_q`` / ``BGWVcoulTable``
    The BGW ``vcoul``-file parity surface, behind ``use_bgw_vcoul``.
"""

from __future__ import annotations

from vcoul.base import (
    CoulombKernel,
    HeadSlotTable,
    SysDim,
    get_kernel,
    head_slot_table,
    v_qG_single,
    v_qG_table,
)
from vcoul.bgw_parity import BGWVcoulTable, fill_v_grid_for_q, read_bgw_vcoul
from vcoul.box_0d import Box0D
from vcoul.box_fft import (
    N_IN_BOX,
    NCELL,
    TRUNC_SHIFT,
    _round_up_fft_size,
    compute_vcoul_box,
)
from vcoul.bulk_3d import Bulk3D
from vcoul.geometry import CoulombGeometry
from vcoul.minibz import (
    COULOMB_GAUGE_TT_SIGN,
    apply_transverse_projector,
    transverse_projector,
    build_miniBZ_dq_cart,
    build_v_head_miniBZ_fn_3d,
    minibz_average,
    minibz_cell_affine,
    minibz_frac_to_cart,
    minibz_inscribed_sphere_r2,
    minibz_moment_tensor,
    minibz_voronoi_batches,
    iter_minibz_photon_samples,
    sample_minibz_qpoints,
    wrap_points_to_voronoi,
)
from vcoul.slab_2d import Slab2D
from vcoul.sphere import (
    bare_coulomb_sphere_indices,
    bare_coulomb_sphere_mask,
    fft_box_miller,
)

#: The bare-3D mini-BZ kernel.  Underscore-private by history (it is
#: ``Common/minibzaverage.f90``'s inner expression, not a public formula)
#: but it has a real cross-package consumer — ``bse.vq_interp`` builds the
#: 2D slab per-Q exchange head with ``kind="slab_lr"`` — so it is on the
#: door rather than reached for through a submodule.
from vcoul.minibz import _minibz_kernel_bare

__all__ = [
    # geometry
    "CoulombGeometry",
    # dispatch + the v(q+G) driver
    "SysDim", "CoulombKernel", "get_kernel", "v_qG_table", "v_qG_single",
    "HeadSlotTable", "head_slot_table",
    "Bulk3D", "Slab2D", "Box0D",
    # mini-BZ sampling / averaging
    "COULOMB_GAUGE_TT_SIGN",
    "apply_transverse_projector",
    "transverse_projector",
    "wrap_points_to_voronoi", "minibz_voronoi_batches",
    "sample_minibz_qpoints", "minibz_inscribed_sphere_r2",
    "minibz_average", "minibz_moment_tensor", "_minibz_kernel_bare",
    "iter_minibz_photon_samples",
    "build_miniBZ_dq_cart", "build_v_head_miniBZ_fn_3d",
    "minibz_frac_to_cart", "minibz_cell_affine",
    # the sphere predicate
    "fft_box_miller", "bare_coulomb_sphere_mask",
    "bare_coulomb_sphere_indices",
    # 0-D cell box.  ``_round_up_fft_size`` is underscore-private by
    # history and is on the door for the same reason
    # ``_minibz_kernel_bare`` is: ``gw.compute_vcoul_0d`` re-exports it,
    # and a shim reaching a submodule for it would be the door rule's
    # first violation.  The three BGW parameters (``Common/nrtype.f90``)
    # are here for the same reason and no other: they were module-level
    # public names at ``gw.compute_vcoul_0d`` before the extraction, so
    # dropping them would be a compatibility break, and taking them off
    # ``vcoul.box_fft`` is what tests/test_layering.py rule 6 caught when
    # the shim first tried it.
    "compute_vcoul_box", "_round_up_fft_size",
    "N_IN_BOX", "NCELL", "TRUNC_SHIFT",
    # BGW parity
    "BGWVcoulTable", "read_bgw_vcoul", "fill_v_grid_for_q",
]
