"""``zeta_loader`` — the reader and format-contract owner of ``zeta_q.h5``.

One door for the ISDF ζ tensor a GW run writes once and reads many times:
the header surface (the ~40 ``mf_header`` + ``isdf_header`` attributes the
tree consumes), the collective SlabIO slab read that feeds V_q, and the
validated per-q G-list accessors.  It owns no mathematics — ``zeta_rcond``,
the fit and the solver tiers are producer-side (``isdf`` /
``gw.isdf_fitting``) and stay there.

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
    amortised across reads.

Standalone, and honestly so
---------------------------
Declared dependencies are lxkit, jax, numpy and h5py.  ``import
zeta_loader`` is clean with no LORRAX checkout on ``sys.path`` at all —
but the DATA path is not yet standalone, and that is stated rather than
blurred: ``ZetaLoader`` reaches ``file_io.mf_header``,
``file_io.isdf_header``, ``file_io.slab_io`` and ``common.gvec_fft_box``
through CALL-TIME imports that refuse by name.  Those four are the wave-1b
seam; when slab_io and the header binders extract, the lazy imports become
package dependencies.
"""

from __future__ import annotations

from zeta_loader.loader import ZetaLoader

__all__ = [
    "ZetaLoader",
]
