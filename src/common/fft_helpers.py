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
# FFI backend for the flat-k helpers — REQUIRED (decisions.md 2026-08-01)
# ============================================================================
# The service itself lives in ``src/ffi/fft.py`` (Python) + ``src/ffi/cpp/
# fftw`` and ``src/ffi/cpp/cufft`` (handlers).  THIS file used to carry a
# second, drifting copy of the gate and both bodies (delegated 2026-07-30),
# and then a gated XLA twin of the flat-k transform (DELETED 2026-08-01
# under the FFI-required ruling: where a certified FFI path exists, the
# native-JAX duplicate is not maintained).
#
# What the service is: the flat-k batched 3-D FFTs dispatched to the platform
# FFI library through the ffi.fft router — the FFTW3 ABI on cpu meshes,
# nvidia-mathdx (mode 3) on CUDA meshes — so the call sites are
# platform-agnostic.
# WHY: XLA's fft custom-call wants the transformed axes minor-most, so every
# dot(k-major) <-> fft(k-minor) boundary in the Σ τ kernel pays a full
# transpose of the ~398 MB/rank μ² tile — 65% of the STAGED τ DISPATCH at
# nb=128/P=64, BEFORE this service existed (the line said "of sigma.exec"
# until 2026-08-11; see the denominator note in ffi/fft.py).  Today the FFT's
# share of that dispatch is 16.1% / 60.5% / 84.9% at 9 / 64 / 216 k-points
# (P=4, BFC@0.85, 2026-08-11) — it is governed by nk, so there is no single
# "today" number and the 9-k fixture's 0.07%-of-wall does not generalise.
# Stride descriptors read the dot-layout tile where it lies, so the transposes
# disappear instead of moving.  Contract: ``docs/architecture/services.md``
# (``ffi.fft``); the k-convolution router: ``docs/architecture/ffi_layout.md``.
#
# What stays here: the OWNER RULE that these helpers are the single FFT entry
# point (``make_flat_k_fft`` below is still the only door), and the XLA
# ``make_sharded_*fftn_3d`` / ``local_*fftn3`` layer above — those serve the
# shard_map-INTERIOR call sites (isdf/core, wfn_transforms, BSE) that have
# no FFI route, which the ruling explicitly keeps.
# ============================================================================

from ffi.fft import (  # noqa: E402  (re-export: see the block above)
    GATE,
    fft_ffi_enabled,
)


# ============================================================================
# THE k-CONVOLUTION ROUTER at the factory seam (decisions.md 2026-09-24)
# ============================================================================
# The physics front doors for every k-axis convolution and every k-axis
# transform of a k-MINOR tile.  Each is the ``ffi.fft`` router factory itself,
# re-exported here so physics code imports its FFTs from one module; the
# router picks nvidia-mathdx on CUDA and the plan route on cpu from the mesh,
# so no caller branches on a backend and no environment variable picks one.
#
#     make_kconv_klead        Σ / COHSEX: KConvStored(prep(W), apply(T, W_prep))
#     make_kconv_klead_unfold Σ from the raw-parent Green: fn(G, Gt, W_prep), same prep
#     make_kfft_klead_unfold  that prep read from the q wedge: fn(W_wedge, Wt=None)
#     make_kconv_chi_unfold   chi0 from the raw-parent Green pair: fn(acc, Gv, Gc, alpha)
#     make_kconv_kminor       BSE rung:   fn(X, K_R), out_layout 0 | 1
#     make_kfft_kminor        sharded transform over the three trailing k axes
#     make_local_kconv_kminor / make_local_kfft_kminor  the same inside a shard_map
#     make_local_kconv_klead  the k-leading conv inside a shard_map, V already R space (BSE W term)
#     make_local_kconv_klead_outer  the same with T = sum_K L R formed on the load (BSE encode
#                                   fused); klead_outer_refusal says when it cannot serve
#
# The contracts live in ``ffi/fft.py``.
# ============================================================================
from ffi.fft import (  # noqa: E402,F401  (re-exported front doors)
    KConvStored,
    make_kconv_klead,
    make_kconv_klead_unfold,
    make_kconv_lorentz_unfold,
    make_kfft_klead_unfold,
    make_kconv_chi_unfold,
    make_kconv_kminor,
    make_kfft_klead,
    make_kfft_kminor,
    make_local_kconv_klead,
    make_local_kconv_klead_outer,
    klead_outer_refusal,
    make_local_kconv_kminor,
    make_local_kfft_klead,
    make_local_kfft_kminor,
)


# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists in the spec vocabulary, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).
#
# Backend: the platform FFI handler, unconditionally (FFTW3 ABI on cpu,
# nvidia-mathdx on CUDA — see the block above) — ``(nk, *trail) ->
# (nk, *trail)``, k-major end to end, no 3-D reshape, no layout anchoring.
# The gated XLA twin (reshape -> make_sharded_*fftn_3d -> reshape) was
# DELETED 2026-08-01 (decisions.md: FFI backends are required, not
# optional); LORRAX_FFT_FFI=0 therefore refuses here rather than silently
# selecting a path that no longer exists.  Recover the XLA arm from git
# history for a debugging build.
# ============================================================================


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
    (:func:`ffi.fft.make_kfft_klead`): nvidia-mathdx on CUDA (measured 1.8-7.4x
    faster than the cuFFT advanced-layout plan it replaced at the χ0/Σ
    production tiles, runs/runtime/kconv_stage2_20260924/bench_flatk_v2.log),
    the FFTW3-ABI flat-k plan handler on cpu.  ``LORRAX_FFT_FFI=0`` refuses.
    """
    if not fft_ffi_enabled():
        raise RuntimeError(GATE.off_refuse_msg)
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
    if not fft_ffi_enabled():
        raise RuntimeError(GATE.off_refuse_msg)
    return make_local_kfft_klead(mesh, kgrid, kind='fftn', norm=norm)
