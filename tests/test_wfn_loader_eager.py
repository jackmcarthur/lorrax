"""MOVED, 2026-08-07.  This file has no cells; the suite is now the service's.

``WfnLoader`` was extracted to ``services/wfn_loader/`` (charter wave 1,
SERVICE_FORM: *a service owns its own suite*), so its contract cells live
beside it:

    services/wfn_loader/tests/test_wfn_loader_contract.py

All TWELVE collected ids that were here moved there VERBATIM — the same
bodies, the same assertions, the same synthetic-WFN builder:

    test_band_pad_rows_are_zero
    test_g_pad_columns_are_zero
    test_iterator_chunks_band_axis
    test_bispinor_lift_matches_legacy[synth]
    test_bispinor_lift_matches_legacy[real]        (was [mos2] — see below)
    test_phdf5_backend_requires_mesh
    test_auto_backend_resolves_to_eager_on_single_process
    test_phdf5_unfold_kernel_matches_eager_ibz[bands0]
    test_phdf5_unfold_kernel_matches_eager_ibz[bands1]
    test_env_forces_backend
    test_no_ffi_at_P_gt_1_refuses_and_names_both_libraries
    test_deleted_phdf5_host_tier_refuses_rather_than_resolving_elsewhere

The one INPUT change, declared rather than smuggled: the ``mos2``
parametrization named
``/pscratch/.../MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5``, whose machine is
gone, so on every machine anybody runs today it produced
``skip: MoS2 3x3 WFN not present``.  Survey w1_wfn_loader §6.4 established
that ``tests/regression/gnppm_debug/WFN.h5`` is that same file (byte-size
identical; header re-read on Perlmutter: nrk 9, mnband 82, nspinor 2,
ngkmax 1963, ntran 2), so the arm points there, RUNS, and keeps the old
path as the ``LORRAX_WFN_TEST_MOS2`` override.  The id changed from
``[mos2]`` to ``[real]``; nothing else about the cell did.

The service suite adds the three tiers this file never had — the
constructor/env refusal surface, ``adopt_mesh``'s four narrowing
conditions, the ``bands()`` values, an emulated 2x2
(``test_wfn_loader_emulated_mesh.py``) and four REAL ranks
(``test_wfn_loader_multiproc.py``).

WHY THIS FILE STILL EXISTS instead of being deleted: ``git rm`` is
deny-listed for this worker, and a zero-cell module is the honest way to
say "moved, here is where" to anyone who runs ``pytest
tests/test_wfn_loader_eager.py`` from muscle memory or a stale command
line.  Its removal is a one-line follow-up for whoever holds the delete
permission, and it is registered as such in the land report.  The file is
COLLECTED (pytest imports it) and contributes ZERO ids, which is exactly
what the step-2 census reconciles against.
"""
from __future__ import annotations
