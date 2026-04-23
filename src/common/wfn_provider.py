"""Universal wavefunction provider — DESIGN DOC / NOT YET IMPLEMENTED.

Status: design only.  Do NOT import this module; it has no runtime code.
The actual implementation currently lives in :mod:`common.psi_G_store`
(narrower scope: host-cached ψ(G) for ISDF fit only).  This file captures
the full design we want before pivoting to consolidate.

=============================================================================
  Goal
=============================================================================

One abstraction that every higher-level function uses to access
wavefunctions.  Callers specify WHAT they want (ψ(G) FFT-box, ψ(G) on
the G-sphere, ψ(r) at centroids, ψ(r) at r-chunks) and over WHICH
window (``band_range``, ``k_range``).  They do NOT need to know:
  * which backend reads the file (h5py via ``read_Gvecs_to_devices``
    or phdf5 via ``PhdfWfnReader.coeffs_gspace``)
  * whether there is a host-resident cache
  * whether we pull per iteration or keep tiles resident.

Those decisions are made ONCE at :func:`wfn_init` from config + env.

=============================================================================
  Current functions we're collapsing
=============================================================================

A. Primary readers (file → sharded FFT-box).  Same contract, different
   signatures:
     * common.load_wfns.read_Gvecs_to_devices
         (wfn, sym, band_range, meta, bispinor, mesh_xy, k_range=None)
         → (nk_slice, nb, ns, nx, ny, nz) complex128,
           sharded P(None, ('x','y'), None, None, None, None)
         h5py backend.  NOTE: k_range already present; if not, add it.
     * common.phdf5_wfn_reader.PhdfWfnReader.coeffs_gspace
         (band_range, *, k_ids=None)
         → same shape/sharding
         phdf5 FFI backend.  k_ids is a list; ``wfn_init`` will let
         callers pass k_range and translate internally.

B. Transforms on an already-loaded FFT-box (stateless):
     * common.load_wfns.get_sharded_wfns_rchunk_slice
         IFFT + k-phase + r-linearize + slice to [r_start, r_end]
         → (nk, nb, ns, n_rchunk) on P(None, None, None, 'y')
     * common.load_wfns.get_sharded_wfns_centroids
         IFFT + k-phase + jnp.take at centroid indices
         → psi_rmu_Y + psi_rmuT_X (the pair-density pair)
     * (missing) psi_fftbox_to_flat_G
         jnp.take on the FFT-box at sphere-G indices
         → (nk, nb, ns, n_G)

C. Batch/iter wrappers combining A+C:
     * iter_psi_rchunk_bandwise  — generator yielding (bc, psi_bc_Y)
     * get_psi_rchunk            — concatenates the generator
     * load_centroids_band_chunked — parallel to get_psi_rchunk but
                                      with centroids transform
   All three carry a ``use_phdf5`` flag; that flag disappears once the
   provider picks the backend at init time.

D. Caching layer (my current narrow implementation):
     * common.psi_G_store.PsiGStore + HostPsiGStore + RereadPsiGStore
       host-resident tiles + in-jit io_callback + shard_map fetch
   Extends to be the core of the new provider.

E. Plumbing:
     * common.load_wfns._get_phdf5_reader — singleton cache, keep as-is
     * common.load_wfns._make_kchunk_meta — helper, keep as-is

=============================================================================
  Proposed architecture — four layers
=============================================================================

Layer 1 — Unified primary reader (collapses A)
-----------------------------------------------

Single dispatcher in ``common.load_wfns``::

    def load_psi_G_fftbox(
        *,
        wfn, sym, meta, mesh_xy, bispinor,
        band_range: tuple[int, int],
        k_range: tuple[int, int] | None = None,     # None → all k
        backend: Literal["auto", "phdf5", "h5py"] = "auto",
    ) -> jax.Array:
        '''(nk_slice, nb, ns, nx, ny, nz) sharded P(None, ('x','y'), ...).'''
        ...

Internally: dispatches to ``read_Gvecs_to_devices`` or
``PhdfWfnReader.coeffs_gspace``.  The phdf5 path converts
``k_range=(lo,hi)`` → ``k_ids=np.arange(lo,hi)``.  Keep both existing
readers; this is just a facade.

Layer 2 — Transform helpers (B, unchanged)
-------------------------------------------

Existing ``get_sharded_wfns_rchunk_slice`` /
``get_sharded_wfns_centroids`` stay.  Add one small helper for the
flat-G case::

    def psi_fftbox_to_flat_G(psi_fftbox, g_indices_flat) -> jax.Array

All three are stateless — no cache lookup, just FFT / gather.

Layer 3 — ``WfnProvider`` (collapses C + D)
--------------------------------------------

Primary fetch + convenience methods::

    class WfnProvider:
        # Core: one entry point for ψ(G) FFT-box.  JIT-safe when
        # mode ∈ {host_cache, reread} (io_callback on a host tile);
        # out-of-jit only when mode == "direct" (calls Layer 1 each
        # call and returns a fresh device array).
        def fetch_psi_G_fftbox(self, band_range, k_range=None) -> jax.Array: ...

        # Derived: fetch + Layer 2 transform, in one call.  Callers
        # become backend- and cache-agnostic.
        def fetch_psi_r_chunk(
            self, band_range, k_range=None, *, r_start, r_end) -> jax.Array: ...
        def fetch_psi_centroids(
            self, band_range, k_range=None, *, centroid_indices) -> (Y, X_T): ...
        def fetch_psi_G_flat(
            self, band_range, k_range=None, *, g_indices_flat) -> jax.Array: ...

        # Lifecycle for reread (no-op for host_cache / direct)
        def begin_iteration(self, **kwargs) -> None: ...
        def end_iteration(self) -> None: ...
        def close(self) -> None: ...

Three concrete backends:

  * ``HostCachedWfnProvider``  — my current ``HostPsiGStore``.
                                 Load once at __init__; host tiles
                                 reused across all iterations.
  * ``RereadWfnProvider``      — my current ``RereadPsiGStore``.
                                 Refresh tiles in ``begin_iteration``,
                                 free in ``end_iteration``.  Same
                                 in-jit fetch.
  * ``DirectWfnProvider``      — NEW.  No tiles; each
                                 ``fetch_psi_G_fftbox`` calls Layer 1
                                 and returns a fresh device array.
                                 Cheap; only usable OUTSIDE a jit.
                                 Models what psp/get_dipole_mtxels
                                 does today.

Layer 4 — ``wfn_init`` factory (single entry point)
----------------------------------------------------

::

    def wfn_init(
        *, wfn, sym, cfg, meta, mesh_xy,
        band_chunk_ranges=None,     # required for host_cache / reread
    ) -> WfnProvider:
        '''Picks (backend, cache_mode) from:
            cfg.wfn_backend    ∈ {"auto", "phdf5", "h5py"}
            cfg.wfn_cache_mode ∈ {"auto", "host_cache", "reread", "direct"}
        with env overrides
            LORRAX_WFN_BACKEND, LORRAX_WFN_CACHE_MODE.

        "auto" backend    → phdf5 if the FFI .so is present and the
                           wfn is phdf5-backed; else h5py.
        "auto" cache_mode → host_cache if the predicted tile fits in
                           a host-RAM budget; else reread.  If
                           band_chunk_ranges is None, falls back to
                           direct (no cache).
        '''

=============================================================================
  What changes, what stays
=============================================================================

UNCHANGED:
  * read_Gvecs_to_devices, PhdfWfnReader.coeffs_gspace — primary readers.
  * get_sharded_wfns_rchunk_slice, get_sharded_wfns_centroids — transforms.
  * _get_phdf5_reader, _make_kchunk_meta — plumbing.
  * Existing iter_psi_rchunk_bandwise / get_psi_rchunk / load_centroids_
    band_chunked callers keep working (back-compat).  Deprecated in a
    follow-up once the provider is well-used.

NEW:
  * common.wfn_provider (this file, rewritten from psi_G_store):
    the three provider classes + wfn_init factory + four fetch_*
    methods.
  * Layer 1 dispatcher load_psi_G_fftbox in common.load_wfns (~30 lines).
  * common.load_wfns.psi_fftbox_to_flat_G helper (~10 lines).

DELETED:
  * ``use_phdf5`` flag sprayed across batch wrappers — backend chosen
    once at wfn_init.
  * Config keys ``use_phdf5_gspace`` and ``gspace_mode`` — replaced by
    ``wfn_backend`` + ``wfn_cache_mode``.

=============================================================================
  Band-parallel opportunity
=============================================================================

Today, functions like psp.get_dipole_mtxels Python-iterate over k and
call read_Gvecs_to_devices inline.  That sequence parallelises only
over whatever is inside the FFT/einsum — bands are effectively
replicated at the dipole-accumulate stage.

With WfnProvider, dipole (and similar A_mnk-style quantities) can:
  psi_G = wfn_prov.fetch_psi_G_fftbox(band_range=(0, nb), k_range=...)
  # psi_G is (nk_slice, nb, ns, fftbox) sharded n_XY on BANDS.
  # Downstream computation can now parallelise over bands because the
  # provider already handed back a band-sharded array.

Rewriting those consumers is separate work — not part of this design.
The provider plumbing just makes it *possible*.

=============================================================================
  Effort estimate (for when we come back to this)
=============================================================================

  Layer 1 dispatcher + phdf5 k_range alias           ~30 lines
  Rename psi_G_store → wfn_provider, rename classes  ~20 lines
  DirectWfnProvider                                  ~40 lines
  Convenience methods (r_chunk / centroids / flat_G) ~40 lines
  wfn_init factory with auto-detect                  ~50 lines
  Config flags + env overrides                       ~15 lines
  ISDF fit caller update                             ~10 lines
  psp.get_dipole_mtxels migration (demo)             ~20 lines
  Smoke tests                                        ~30 lines
  Total: ~255 lines, 2 files added, ~4 touched.

=============================================================================
  Open questions to resolve before implementing
=============================================================================

1. DirectWfnProvider inside-jit story.  Out-of-jit direct reads are
   trivial.  For an in-jit caller that really does NOT want a host
   cache, we'd need io_callback on the phdf5 reader — risky because
   coeffs_gspace is a collective that all ranks must enter together.
   Probably defer; the two in-jit modes are host_cache and reread.

2. Arbitrary (non-bc-aligned) band windows.  My current tile layout
   only supports windows that are unions of declared bcs.  For the
   universal loader we might want arbitrary ranges — that needs
   either (a) a fresh read per call for non-aligned windows, or
   (b) a cross-rank reshuffle in fetch.  Defer until a caller needs it.

3. ψ(r) cached view.  If a consumer wants ψ(r) at multiple r-chunks
   for the same band window, we currently IFFT every call.  Could
   add an optional ``cache_psi_r`` flag on the provider that keeps
   the IFFT'd FFT-box alongside the G-box.  Defer; no consumer asks.
"""

# No runtime code — this is a design doc only.  Do not import this
# module.  The actual narrow implementation is in common.psi_G_store.
raise ImportError(
    "common.wfn_provider is a design-doc file, not a module.  "
    "Use common.psi_G_store for the current (narrow) ψ(G) store, "
    "or read this file's docstring for the pending unification plan."
)
