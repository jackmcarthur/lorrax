"""``wfn_loader`` — one entry point for ψ(G) loading from a BerkeleyGW WFN.h5.

:class:`WfnLoader` replaces the {WFNReader + PhdfWfnReader +
SymMaps.get_cnk_fullzone[_batch] + SymMaps.get_gvecs_kfull +
load_wfns.read_Gvecs_to_devices + load_kpoint_fftbox} mess with one class:
a caller says which bands and which k-points it wants and gets ψ back in
**G-flat** layout (per-k, per-band, per-spinor, ngk_max-padded), on the
sharding it asked for.  Which transport moved the bytes — a host h5py read
per rank, or a collective MPI-IO read through the phdf5 FFI — is a
resolved fact it can read off ``loader.backend`` and never has to branch
on: the two backends are byte-identical for the same request, which is the
P2 contract and the only reason ``LORRAX_WFN_BACKEND`` is safe to expose.

THE PACKAGE IS THE DOOR.  Everything a consumer needs is a top-level name
here; importing ``wfn_loader.loader`` from outside is a layering violation
the monorepo's ``tests/test_layering.py`` fails on.

The surface
-----------
``WfnLoader(path, *, mesh=None, backend='auto')``
    The class.  Its ``mf_header`` attributes are the same names the legacy
    ``WFNReader`` exposed (``nkpts``, ``nbands``, ``nspinor``, ``kgrid``,
    ``fft_grid``, ``bvec``, ``sym_matrices``, ``translations``, …), so it
    is a drop-in for the metadata callers too.
``WfnLoader.load`` / ``.load_process_local`` / ``.bands``
    ψ for a (band-range, k-set) window: as a GLOBAL sharded array, as a
    THIS-PROCESS-ONLY array, and as a band-chunked iterator.
``WfnLoader.kvecs`` / ``.gvecs`` / ``.ngk_valid`` / ``.box_index`` /
``.box_index_dev``
    The reciprocal-space side: paired fractional-k and padded per-k
    Miller-index tables, logical lengths, and the FFT-box gather table (host,
    and cached on device). Bloch phases and ``k+G`` must use the paired loader
    tables rather than reconstructing k independently.
``KSpec``
    The k-spec vocabulary: ``'ibz'``, ``'full_bz'``, or an explicit list
    of full-BZ indices.

THE UNDERSCORED NAMES ARE PART OF THE DOOR, deliberately.  They are the
module-level kernel / jit-cache factories the in-tree tests pin by name —
``_phdf5_unfold_kernel`` is the on-device symmetry unfold, the one piece of
the collective path that runs without an ``.so`` and therefore the piece a
single-process cell can pin against the eager backend.  Re-exporting them
here is what lets the transitional shim ``src/file_io/wfn_loader.py`` keep
every name the old module bound while still going through the door — the
alternative is lorrax importing ``wfn_loader.loader``, i.e. trading a
counted re-export for an uncounted past-the-door reach.

``_build_phdf5_clamped_counts`` was on this list until 2026-08-07.  It was
the per-rank hyperslab clip for the collective read, and it is now
``file_io._slab_io_ffi._derive_window_counts``, behind the slab_io door
that performs the read: the service states which windows it wants and how
much of each is real, and knows nothing about hyperslabs, ranks or FFI
targets.  Its cells moved with it.
"""

from __future__ import annotations

from wfn_loader.loader import (
    IBZRows,
    KSpec,
    WfnLoader,
    _bispinor_lift_kernel,
    _get_bispinor_lift_jit,
    _phdf5_unfold_kernel,
    _sharded_zero_proto_fn,
)

__all__ = [
    # the class, and the k-spec vocabulary its methods take
    "WfnLoader", "KSpec", "IBZRows",
    # module-level helpers the in-tree tests pin by name (see above)
    "_phdf5_unfold_kernel",
    "_sharded_zero_proto_fn", "_get_bispinor_lift_jit",
    "_bispinor_lift_kernel",
]
