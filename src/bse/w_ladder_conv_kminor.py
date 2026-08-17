"""The ladder-W rung on the k-MINOR FUSED CONV handler — opt-in, DEFAULT OFF.

This module is the BSE-side consumer of the fused-conv family's k-MINOR
member (``ffi.fft.make_conv_kminor_ffi``, reached through the
``common.fft_helpers.make_fused_conv_kminor`` seam).  It contains no
new physics and no new layout: it replaces the five-op chain

    reshape → ifftn(ortho) → W_R broadcast multiply → fftn(ortho)
            → reshape → transpose to (b, k, t, μ, s, ν)

that ``bse_ring_comm._apply_W_from_T`` emits today with ONE custom call that
lands in the decode's layout directly.  Everything after it — the
``contract_bands_block_reshard`` decode, the ``/√nk``, the output sharding —
is byte-for-byte the production spelling and is called from here unchanged.

WHAT IT REPLACES, PRICED
------------------------
Measured on one A100 at the gnppm geometry (n_rmu=399, nk=3×3×1, nspinor=2,
nb=1; ``reports/screening_diagrams_wbse/evidence/opt_kernels/RESULTS.txt``
table A), per non-TDA ladder matvec, GPU busy 4897 µs total:

    loop_transpose_fusion_2  (post-FFT → decode O layout)     1488 µs  30.4%
    regular_fft + vector_fft (the two cuFFT radix passes ×2)  1122 µs  22.9%
    loop_multiply_fusion_2   (W_R broadcast multiply)          379 µs   7.7%
    loop_transpose_fusion_3  (pre-FFT chain layout)            376 µs   7.7%
    scal_kernel_val          (cuFFT's missing 'ortho' scale)   271 µs   5.5%
    ------------------------------------------------------------------------
    THE CHAIN                                                 3637 µs  74.3%

8.1 HBM passes over a 91.7 MB tile where a fused kernel needs 1.12.  O9's
priced target for this module is 0.32–0.52 ms in place of the 3.64 ms, i.e.
2.8–3.1× on the matvec; after it the encode/decode GEMMs (1012 µs) are the
largest remaining term.

WHY A SEPARATE MODULE AND NOT AN EDIT
-------------------------------------
``bse_ring_comm.py`` is owned by the integration lane.  The whole opt-in is
three lines there (see ``integrate_conv_kminor.patch`` beside this file's
evidence directory), and every line of behaviour lives here, so the two lanes
cannot collide and the rung file keeps one spelling of the chain plus a hook.

**DEFAULT OFF.**  ``LORRAX_CONV_KMINOR_FFI=1`` opts in.  The handler is new
custom CUDA device code exercised at P=1 on one A100; the XLA chain is the
certified production path.  ``conv_kminor_rung_supported`` is the honest
answer to "may this run here" and REFUSES rather than demoting — in
particular for the fp32-GMRES arm, whose payload ``bse_feast.
_build_gmres_data_fp32`` casts to c64 and which the c128-only handler must
turn away instead of silently up-casting the arithmetic being measured.

SCOPE, stated because it is narrower than the module reads
----------------------------------------------------------
* The RING rung only.  The TDA stack rung (``bse_stack_matvec._w_stack``)
  sits inside one big ``shard_map`` and calls ``local_fftn3`` directly;
  ``shard_map`` cannot nest, so it has no FFI route at all — the same
  structural limit O7 recorded.
* P=1, 1×1 mesh.  The handler is rank-local by construction (it multiplies
  rank-local tiles and implements no reshard, exactly like every other FFT
  handler here) and the specs below carry μ on ``'x'`` / ν on ``'y'`` as
  production does, so P>1 should be a measurement rather than a port — but
  it is a measurement nobody has taken.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from common.fft_helpers import make_fused_conv_kminor
from ffi.fft import (conv_kminor_available, conv_kminor_enabled,
                     conv_kminor_plan)

__all__ = [
    "conv_kminor_enabled", "conv_kminor_plan", "conv_kminor_rung_supported",
    "rung_uses_conv_kminor",
    "RUNG_T_SPEC", "RUNG_W_SPEC", "RUNG_O_SPEC",
    "make_rung_conv", "build_rung_body",
]

#: The rung's production shardings, in this module's rank-6 vocabulary.
#: ``T`` is ``(b, μ, ν, t, s, nk)`` with μ on ``'x'`` (it comes from ψ_c^X) and
#: ν on ``'y'`` (from ψ_v^Y) — ``bse_ring_comm.make_bse_shardings``'s ``sh.T``.
RUNG_T_SPEC = P(None, "x", "y", None, None, None)
#: ``W_R`` flattened to ``(μ, ν, nk)`` — ``sh.W`` with its three k axes merged.
RUNG_W_SPEC = P("x", "y", None)
#: ``(b, k, t, μ, s, ν)`` — ``contract_bands_block_reshard(extra="leading")``'s
#: canonical ``O`` layout, which ``out_layout=1`` emits from the store.
RUNG_O_SPEC = P(None, None, None, "x", None, "y")


def conv_kminor_rung_supported(mesh: Mesh, dtype) -> tuple[bool, str]:
    """``(ok, reason)`` — may the fused rung run on this mesh at this dtype?

    Both halves in one place because a caller asking "can I" must get ONE
    answer with ONE reason.  The dtype half is a refusal, not a demotion:
    the handler is c128-only and up-casting a c64 payload would change the
    arithmetic of the arm being measured, not merely its speed.
    """
    if jnp.dtype(dtype) != jnp.dtype(jnp.complex128):
        return False, (
            f"the k-minor fused conv handler is complex128 only; the rung is "
            f"{jnp.dtype(dtype)} (the fp32-GMRES ladder arm casts the payload "
            f"in bse_feast._build_gmres_data_fp32).  Routing it here must "
            f"REFUSE, not up-cast.")
    return conv_kminor_available(mesh)


def rung_uses_conv_kminor(mesh: Mesh, kgrid, dtype) -> tuple[bool, str]:
    """``(use, reason)`` — the rung's whole routing question, answered once.

    Wraps :func:`ffi.fft.conv_kminor_plan` with the ONE extra condition the
    rung adds: the payload must be complex128.  Under ``auto`` a c64 payload
    (the ``--gmres-fp32`` arm) falls through to the XLA chain, which serves it
    correctly; under ``on`` the handler's own refusal fires instead, because a
    caller that demanded the kernel must not be silently routed elsewhere.
    """
    if jnp.dtype(dtype) != jnp.dtype(jnp.complex128):
        if conv_kminor_enabled() and CONV_KMINOR_MODE_ON():
            raise TypeError(
                f"LORRAX_CONV_KMINOR_FFI=on, but the rung payload is "
                f"{jnp.dtype(dtype)}.  The handler is complex128 only and "
                f"does NOT up-cast (the fp32-GMRES arm casts in "
                f"bse_feast._build_gmres_data_fp32); use the default `auto`, "
                f"which keeps the XLA chain for this payload.")
        return False, f"payload dtype {jnp.dtype(dtype)} is not complex128"
    return conv_kminor_plan(mesh, kgrid)


def CONV_KMINOR_MODE_ON() -> bool:
    """True when the dial is the explicit ``on`` (not ``auto``)."""
    from ffi.fft import conv_kminor_mode
    return conv_kminor_mode() == "on"


def make_rung_conv(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    *,
    out_layout: int = 1,
    mult: float = 1.0,
) -> Callable:
    """``fn(T, W3) -> U`` — the rung's convolution, one FFI call.

    ``T`` ``(b, μ, ν, t, s, nk)`` and ``W3`` ``(μ, ν, nk)`` (``W_R`` with its
    three k axes merged), both c128.  ``out_layout=1`` (the default here)
    returns ``(b, k, t, μ, s, ν)``, ready for the decode with no transpose;
    ``out_layout=0`` returns ``T``'s own layout and is what the A/B bench uses
    to isolate the CONVOLUTION from the STORE.

    ``norm='ortho'`` on both transforms, matching the production spelling
    (``bse_ring_comm``'s ``make_sharded_{i,}fftn_3d(..., norm='ortho')``);
    the two factors are one folded constant inside the handler.
    """
    return make_fused_conv_kminor(
        mesh, kgrid, RUNG_T_SPEC, RUNG_W_SPEC,
        norm='ortho', mult=mult, out_layout=out_layout)


def build_rung_body(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    w_decode: Callable,
) -> Callable:
    """The drop-in body of ``bse_ring_comm._apply_W_from_T``.

    Returns ``f(T, psi_c_X, psi_v_Y, W_R) -> WX`` with the SAME signature,
    the same operands, the same output sharding and the same value as the
    production body — the chain replaced by one call, everything downstream
    untouched.  The caller wraps it in its own ``jax.jit`` with its own
    ``in_shardings``/``out_shardings`` exactly as it does today.

    ``w_decode`` is the caller's already-built
    ``contract_bands_block_reshard(mesh, extra="leading")``.  It is passed in
    rather than rebuilt: building a second one would give the rung two decode
    primitives with two collective warm-ups for one contraction.
    """
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    conv = make_rung_conv(mesh, (nkx, nky, nkz), out_layout=1)

    def _apply_W_from_T_conv(T, psi_c_X, psi_v_Y, W_R):
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))
        # (μ, ν, kx, ky, kz) -> (μ, ν, nk).  A bitcast: the three k axes are
        # replicated, contiguous and minor-most, so XLA emits no copy — the
        # 3-D k structure exists only inside the handler, the same convention
        # the flat-k service uses.
        W3 = W_R.reshape(W_R.shape[0], W_R.shape[1], nk)
        # ONE call: ifft, W multiply, fft, both 'ortho' factors, AND the
        # (b,μ,ν,t,s,k) -> (b,k,t,μ,s,ν) permutation, emitted by the store.
        O_b = conv(T, W3)
        psi_v_snv = jnp.transpose(psi_v_Y, (0, 2, 3, 1))
        out = w_decode(psi_c_X, O_b, psi_v_snv)      # (b, nk, c_X, v_Y)
        WX = jnp.transpose(out, (0, 2, 3, 1))        # (b, c_X, v_Y, nk)
        return WX / sqrt_nk

    return _apply_W_from_T_conv
