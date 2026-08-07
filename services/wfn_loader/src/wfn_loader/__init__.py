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
``WfnLoader.gvecs`` / ``.ngk_valid`` / ``.box_index`` / ``.box_index_dev``
    The G-vector side: the padded per-k Miller-index table, its logical
    lengths, and the FFT-box gather table (host, and cached on device).
``KSpec``
    The k-spec vocabulary: ``'ibz'``, ``'full_bz'``, or an explicit list
    of full-BZ indices.

THE UNDERSCORED NAMES ARE PART OF THE DOOR, deliberately.  They are the
module-level helpers ``tests/test_wfn_loader_eager.py`` pins directly
(``_build_phdf5_clamped_counts`` is the per-rank hyperslab clip whose band
bound drifted from ``b_hi`` to ``mnband`` once already, and two cells there
read its SOURCE), plus the three jit/lru-cache factories the phdf5 path
hangs off.  Re-exporting them here is what lets the transitional shim
``src/file_io/wfn_loader.py`` keep every name the old module bound while
still going through the door — the alternative is lorrax importing
``wfn_loader.loader``, i.e. trading a counted re-export for an uncounted
past-the-door reach.
"""

from __future__ import annotations

from wfn_loader.loader import (
    KSpec,
    WfnLoader,
    _bispinor_lift_kernel,
    _build_phdf5_clamped_counts,
    _get_bispinor_lift_jit,
    _phdf5_unfold_kernel,
    _sharded_zero_proto_fn,
)

__all__ = [
    # the class, and the k-spec vocabulary its methods take
    "WfnLoader", "KSpec",
    # module-level helpers the in-tree tests pin by name (see above)
    "_build_phdf5_clamped_counts", "_phdf5_unfold_kernel",
    "_sharded_zero_proto_fn", "_get_bispinor_lift_jit",
    "_bispinor_lift_kernel",
]
