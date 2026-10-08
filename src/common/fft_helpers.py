from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from common.shard_map import shard_map


# Value-level parity contract for the canonical flat-k service.  This is the
# registered Sigma-path class (``docs/architecture/ffi_layout.md``, the
# engine-swap parity rule), not a bit-equality promise between FFT engines.
FLAT_K_FFT_VALUE_RTOL = 1.0e-12


# ============================================================================
# shard_map based FFT - runs FFT independently on each device's local data
# See: https://docs.jax.dev/en/latest/notebooks/shard_map.html
#
# ONE local-FFT form, deliberately.  A second, per-axis
# ``custom_partitioning`` form (``make_jittable_local_{i,}fftn_3d``, three
# chained rank-1 FFTs) lived here until 2026-07-30 with no production caller:
# its only user was a memory-model FFT query (itself deleted 2026-09-25), which
# was sizing three rank-1 cuFFT plans that nothing in the pipeline ever builds.
# Recover it from git history if a per-axis form is ever wanted again; do not
# reintroduce it as a modelling-only path.
# ============================================================================


def local_ifftn3(x_local, *, axes: tuple[int, ...] = (-3, -2, -1), norm: str | None = None):
    """Device-local N-D IFFT — the inner kernel of :func:`make_sharded_ifftn_3d`.

    Call this DIRECTLY from code that is already inside a ``shard_map`` (shard_map
    cannot nest): the operand is a device-local shard whose FFT axes are
    replicated, so a plain ``jnp.fft.ifftn`` runs entirely on-device.  The
    ``make_sharded_ifftn_3d`` factory below is just this kernel wrapped in a
    ``shard_map`` for auto-partitioned callers.  ONE source for the local FFT.

    Do NOT call it eagerly on a (μ,ν)-sharded global array: outside a
    shard_map, jax gathers the full operand onto every rank — an N_μ²-class
    tile, forbidden by the scaling doctrine (audit P0-4).
    ``tests/test_fft_shardmap_context.py`` gates this by AST for src/bse and
    src/gw: every call site's enclosing-function chain must construct a
    shard_map (or be a ratcheted, documented single-device exception).
    """
    return jnp.fft.ifftn(x_local, axes=axes, norm=norm)


def local_fftn3(x_local, *, axes: tuple[int, ...] = (-3, -2, -1), norm: str | None = None):
    """Device-local N-D forward FFT — inner kernel of :func:`make_sharded_fftn_3d`.

    Forward counterpart of :func:`local_ifftn3`; call directly from inside a
    ``shard_map``.
    """
    return jnp.fft.fftn(x_local, axes=axes, norm=norm)


def make_sharded_ifftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)

    Returns:
        A function that performs 3D IFFT on sharded data
    """
    def _wrap(x_local):
        return local_ifftn3(x_local, axes=axes, norm=norm)

    return shard_map(_wrap, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)

# ---------------------------------------------------------------------------
# The DONATED transform, memoised — one program per (mesh, spec, axes, norm)
# ---------------------------------------------------------------------------
# ``make_sharded_ifftn_3d`` is a FACTORY: it returns a fresh ``shard_map``
# closure every call and carries no cache.  Wrapping that fresh closure in a
# fresh ``jax.jit`` — which is what every donated-W call site did — hands jax a
# wrapper object whose dispatch cache is empty, so a byte-identical program is
# re-traced, re-lowered and re-probed against the persistent compile cache on
# every call.  It is the same defect ``bse_davidson_helpers`` fixed for the
# exact-diagonal build (PRECOND_BUILD_FREE.md §3.1, §7.2), and it is measured
# at ~20-23 ms per call against ~1.0 ms of execution on the Si 4x4x4 record
# deck at P=4 (FIX_construction_defects.md §1).
#
# The memo below keys on everything that determines the PROGRAM — the mesh, the
# partition spec, the transformed axes and the norm convention.  It deliberately
# does NOT key on shape or dtype: those are jax's own dispatch-cache business,
# so one memo entry serves every payload shape a run puts through the same
# transform.  ``Mesh`` and ``PartitionSpec`` are both hashable by value, so two
# call sites that ask for the same transform share one program.
#
# WHY A SEPARATE ACCESSOR AND NOT A MEMO ON THE FACTORY.  Ten call sites in the
# tree build sharded transforms through the factory, and several close the
# returned callable into larger objects.  Memoising the factory itself would
# change the OBJECT IDENTITY those call sites see, and identity is load-bearing
# in this tree (``bse_feast._GMRES_SOLVER_CACHE`` keys on ``id(matvec)``).  A
# new accessor changes nothing for anyone who does not call it.
_DONATED_KFFT_KMINOR: dict[tuple, Callable] = {}


def get_donated_kfft_kminor(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str = "ifftn",
    norm: str | None = "ortho",
) -> Callable:
    """The k-minor k-axis transform (``make_kfft_kminor``) as a jitted,
    INPUT-DONATING program, memoised per ``(mesh, kgrid, spec, kind, norm)``.

    Donation is the point of the separate accessor.  The W-transform call sites
    want XLA to alias ``W_R`` onto ``W_q``'s buffer — at production
    ``mu = 10015 / P = 64`` the un-aliased form costs 2 x 404 MB per rank — and
    donation only works from a top-level dispatch boundary, which is what this
    is.  Callers must drop their own reference to the operand right after the
    call; the returned array is the only live copy.  Memoised so the program is
    constructed once per process instead of once per call.
    """
    key = (mesh, tuple(int(v) for v in kgrid), spec, kind, norm)
    hit = _DONATED_KFFT_KMINOR.get(key)
    if hit is None:
        from ffi.fft import make_kfft_kminor as _kfft
        hit = jax.jit(_kfft(mesh, key[1], spec, kind=kind, norm=norm),
                      donate_argnums=(0,))
        _DONATED_KFFT_KMINOR[key] = hit
    return hit


def make_sharded_fftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    shard_map local FFT (forward).

    This is the forward-FFT counterpart to make_sharded_ifftn_3d.
    """
    def _wrap(x_local):
        return local_fftn3(x_local, axes=axes, norm=norm)

    return shard_map(_wrap, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


# ============================================================================
# THE k-CONVOLUTION ROUTER at the factory seam
# ============================================================================
# The physics entry points for every k-axis convolution and k-axis transform.
# Each is the ``ffi.fft`` router factory itself, re-exported so physics code
# imports its FFTs from one module; the router picks nvidia-mathdx on CUDA and
# the host plan route on cpu and the XLA backend elsewhere, from the mesh's
# device vendor, so no caller
# branches on a backend and no environment variable picks one.  The contracts
# live in ``ffi/fft.py`` and ``docs/architecture/kconv.md``.
# ============================================================================
from ffi.fft import (  # noqa: E402,F401  (re-exported entry points)
    KConvStored,
    chi_unfold_scratch_bytes,
    kconv_backend,
    klead_outer_decode_refusal,
    klead_outer_refusal,
    klead_unfold_scratch_bytes,
    make_fused_conv_kpair,
    make_fused_conv_kparent,
    make_fused_conv_kplane,
    make_kconv_chi_unfold,
    make_kconv_chi_vertex,
    make_kconv_klead,
    make_kconv_klead_unfold,
    make_kconv_kminor,
    make_kconv_lorentz_unfold,
    make_kfft_klead,
    make_kfft_klead_unfold,
    make_kfft_kminor,
    make_local_kconv_klead,
    make_local_kconv_klead_outer,
    make_local_kconv_klead_outer_decode,
    make_local_kconv_kminor,
    make_local_kfft_klead,
    make_local_kfft_kminor,
    x_block_rows,
    xla_reference,
)


# Flat-k FFT helpers: callers operate on (nk, *trail) arrays everywhere and the
# k-grid 3-D form exists only in the spec vocabulary.


def make_flat_k_fft(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Return a flat-k FFT: ``(nk, *trail) -> (nk, *trail)``.

    ``spec`` is the ``PartitionSpec`` on the 3-D form
    ``(nkx, nky, nkz, *trail)``.  The three leading k-axes must be
    replicated (``None``) so the per-rank transform sees the full k axis on
    every device.  ``out_spec`` must equal ``spec`` (no post-FFT reshard).

    ``kind='ifftn'`` or ``'fftn'`` selects the direction.  ``norm``
    follows ``jnp.fft.*`` ('ortho', 'forward', 'backward' / None).

    The k-convolution router's k-leading transform
    (:func:`ffi.fft.make_kfft_klead`): nvidia-mathdx on CUDA, the host plan
    handler on cpu, the XLA backend elsewhere.
    """
    if out_spec is not None and tuple(out_spec) != tuple(spec):
        raise ValueError(
            f"the flat-k transform implements no post-FFT reshard (out_spec "
            f"{out_spec} != spec {spec}); drop out_spec.")
    return make_kfft_klead(mesh, kgrid, spec, kind=kind, norm=norm)


def make_flat_k_ifftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k IFFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='ifftn',
                           norm=norm, out_spec=out_spec)


def make_flat_k_fftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k FFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='fftn',
                           norm=norm, out_spec=out_spec)


def make_local_flat_k_fftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    *,
    norm: str | None = 'ortho',
) -> Callable:
    """Device-local flat-k FFT for a caller already inside ``shard_map``.

    The same router transform as :func:`make_flat_k_fftn`
    (:func:`ffi.fft.make_local_kfft_klead`), without its outer ``shard_map``
    wrapper.  It exists for bounded local trailing-axis slabs; callers must
    keep the complete k axis local.
    """
    return make_local_kfft_klead(mesh, kgrid, kind='fftn', norm=norm)
