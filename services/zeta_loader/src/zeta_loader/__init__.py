"""``zeta_loader`` — the reader and format-contract owner of ``zeta_q.h5``.

One door for the ISDF ζ tensor a GW run writes once and reads many times
(and for its bispinor siblings ``zeta_q_mu{1,2,3}.h5``): the header
surface, the collective slab read that feeds V_q, the sanctioned local
plan, the never-raising probe, and the one post-close append.  It owns no
mathematics — ``zeta_rcond``, the fit and the solver tiers are
producer-side (``isdf`` / ``gw.isdf_fitting``) and stay there.

THE PACKAGE IS THE DOOR.  Everything a consumer needs is a top-level name
here; importing ``zeta_loader.<submodule>`` from lorrax is a layering
violation the monorepo's ``tests/test_layering.py`` fails on.

The surface
-----------
``ZetaLoader(path, *, mesh=None, mode='r')``
    Header-only when ``mesh=None`` — every header attribute works with no
    transport, which is what lets a caller ask the file about its own
    layout on a stack with no phdf5 FFI.  With a mesh, one SlabIO handle
    is opened and held for the loader's lifetime so the phdf5 ctx is
    amortised across reads.  ``close()`` / context manager releases both
    handles; the header attributes are ``bind_mf_attrs`` +
    ``bind_isdf_attrs`` (``nspin``, ``kgrid``, ``fft_grid``,
    ``sym_matrices``, ``r_mu_fft_idx``, ``n_rmu``, ``zeta_layout``,
    ``gvec_components``, ``ngk_per_q``, ``ngkmax_zeta``,
    ``zeta_cutoff_ry``, …).
``.read_zeta_G_slab(*, q_offset, q_count, mu_offset, mu_count, mesh=None)``
    THE hot V_q read.  COLLECTIVE.  ``(Q, μ, ngkmax)`` complex128 at
    ``P(None, ('x','y'), None)``; a ``mu_count`` past the on-disk extent
    comes back zero-filled.
``.read_zeta_G_local(key)``
    The sanctioned NON-collective plan: a host-numpy h5py hyperslab
    through a serial handle the loader owns.  Per-rank independent BY
    CONTRACT — making it collective would turn a rank-0 diagnostic into
    a hang.  Works at ``mesh=None``.
``.load(*, q='ibz'|seq[int], mu=None, sharding=None)``
    The WfnLoader-shaped window.  Collective.
``.gvecs(q=...)`` / ``.ngk_valid(q=...)``
    The VALIDATED per-q G-list, padded to the FFT-box sentinel.
    ``gvecs`` carries the only read-time check that
    ``isdf_header/gvec_components`` agrees with the ``mf_header`` FFT
    grid — consumers that take ``gvec_components`` raw are trusting a
    table nobody checked.
``.slab_io``
    The raw transport handle.  A migration escape hatch, kept.
``probe_zeta_file(path) -> ZetaFileProbe``
    NEVER raises, for any input.  Owns the
    ``(('zeta_q_G',1),('zeta_q',2))`` layout dispatch as the one copy of
    that truth.
``write_g0_mu(path, g0_logical, *, n_rmu_expected=None)``
    The one sanctioned rank-0 serial append after the collective handle
    closes.  The caller owns the gating and the barriers, deliberately.

Contracts
---------
ONE DATA LAYOUT.  Every data method reads G-flat ``zeta_q_G`` and refuses
anything else by name; the r-space read path was deleted on 2026-08-07
because no writer in the tree emits that layout.  ``__init__`` still
opens r-space files, because the HEADER surface is layout-independent.

AGREEMENT IS CHECKED AT OPEN, NOT ASSUMED.  Header μ vs dataset μ, and
header ``ngkmax`` vs the ``zeta_q_G`` G axis — the second because the
collective plan sizes from the header and the local plan sizes from the
dataset, so a disagreement would have the two reading different extents
of one file in silence.

COMPLETENESS IS CHECKED AT OPEN.  ``isdf_header/zeta_is_done`` is the
writer's flag, and a ζ left by a job that died mid-write refuses here
rather than flowing into V_q → W → Σ with rc=0
(``LORRAX_ALLOW_PARTIAL_ZETA=1`` overrides, for debugging).

Standalone, and honestly so
---------------------------
Declared dependencies are lxkit, jax, numpy and h5py.  ``import
zeta_loader`` is clean with no LORRAX checkout on ``sys.path`` at all,
and the FORMAT surface — :func:`probe_zeta_file`, :func:`write_g0_mu` —
is FULLY FUNCTIONAL there, which matters because both of its call sites
run outside a loader's lifetime (the probes gate whether the fit runs;
the append happens after the collective handle is gone).

The DATA path is not standalone, and that is stated rather than blurred:
``ZetaLoader`` reaches ``file_io.mf_header``, ``file_io.isdf_header``,
``file_io.slab_io`` and ``common.gvec_fft_box`` through CALL-TIME imports
that refuse by name.  Those four are the wave-1b seam; when slab_io and
the header binders extract, the lazy imports become package
dependencies.
"""

from __future__ import annotations

from zeta_loader.format import ZetaFileProbe, probe_zeta_file, write_g0_mu
from zeta_loader.loader import ZetaLoader

__all__ = [
    # the reader
    "ZetaLoader",
    # the format surface (pure h5py + numpy; works with no lorrax)
    "probe_zeta_file", "ZetaFileProbe", "write_g0_mu",
]
