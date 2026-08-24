"""PROTOTYPE (track O7) — routing the ladder-W direct rung's k-FFTs through the
flat-k FFI engine instead of XLA's ``fft`` custom call.

**VERDICT, MEASURED 2026-08-16 (claim 0230) — DO NOT WIRE THIS IN.**  One A100,
jax 0.7.0, cache-cold, BFC@0.85; evidence
``reports/screening_diagrams_wbse/evidence/opt_fftffi/``.  The rung's tile is
ALREADY k-minor, which is XLA's best FFT layout, so the Σ-kernel's
layout-anchor argument does not apply here: the production rung's optimized HLO
carries ONE transpose, ZERO copies and a temp budget equal to exactly one ``T``
tile.  Substituting the strided cuFFT handler wins 17-20% at ``nk=9`` and LOSES
1.6× at ``nk=64`` and 4.0× at ``nk=216`` (crossover ``nk≈16``), because
``cufftPlanMany64(istride=T, idist=1)`` degrades with the batch stride and in
the rung that stride IS the μ·ν·s² tile.  What the measurement DID find is that
the rung's convolution is exactly ``make_gw_conv_ffi``'s expression (see
:func:`make_conv_ffi_fused`) and that the fused form is worth 17-20% at
``nk=9`` purely by removing two HBM round trips — the FUSION is transferable,
the ENGINE is not.  The ask that follows is a k-MINOR fused conv handler, which
the rung could call with no layout change at all.  Keep this module as the A/B
instrument for that; do not promote any arm of it as it stands.

**This module is a measurement instrument, not a production path.**  Nothing in
``src/`` imports it; ``tests/bench/bench_w_ladder_fftffi.py`` is its only
caller.  It exists because the rung
(``bse_stack_matvec._build_bse_stack_matvec_full._apply_W_from_T``) is 96-99% of a
ladder matvec (opt_shifts, 2026-08-16: 7.92 ms with the rung vs 0.255 ms
without, gnppm_debug, 1×A100) and its FFT pair is one of the four BSE modules
carrying the registered eager-FFT debt (``docs/architecture/decisions.md``
2026-08-01: "BSE's XLA FFTs have no FFI route yet").

THE STRUCTURAL FACT THAT MAKES THIS REACHABLE AT ALL.  ``shard_map`` cannot
nest, and the FFI entry (``fft_helpers.make_flat_k_*``) wraps its own — so a
call site already inside a ``shard_map`` has no FFI route
(``docs/dev/flat_k_fft_service.md`` §3; that is why ``isdf/core`` and
``wfn_transforms`` cannot be routed).  The RING rung is the exception: its
``_apply_W_from_T`` is a plain ``jax.jit`` with ``in_shardings``, and it already
reaches its transforms through the ``make_sharded_*fftn_3d`` factories, which
wrap their own ``shard_map``.  The TDA stack rung (``bse_stack_matvec._w_stack``
→ ``_conv_decode``) is NOT reachable — it is inside one big ``shard_map`` and
calls ``local_fftn3`` directly; only its ``gspmd`` audit twin is.  The ladder
uses the ring builder, so the ladder is the reachable one.

AND THE ONE THAT LIMITS THE WIN.  The Σ τ kernel's 6-transposes-killed result
came from a LAYOUT ANCHOR: ``dot`` wants k-major, XLA's ``fft`` wants the
transformed axes minor-most, so every boundary paid a transpose of the μ² tile.
**The rung has no such anchor on the FFT side** — its ``T`` is
``(b, μ, ν, t, s, nk)``, i.e. already k-minor, exactly the layout XLA's FFT
wants.  What the rung DOES pay is on the far side: the decode
(``contract_bands_block_reshard``, ``extra="leading"``) wants
``O = (b, k, t, μ, s, ν)``, so the production rung transposes the whole U tile
k-minor→k-leading after the second FFT (the shared full builder's decode, whose own
comment predicts this: "composes with the future flat-k conv layout … which
emits k-leading natively").  So the FFI's k-major contract is aligned with the
DECODE, not with the encode — which is why this module measures three arms and
not two:

    xla        production spelling: 8-D reshape → sharded ifftn → W_R multiply
               → sharded fftn → 6-D reshape → transpose to O.
    ffi_kmin   FFI engine, production STORAGE: one transpose in
               (b,μ,ν,t,s,k)→(k,b,μ,ν,t,s), FFI ifftn → multiply → FFI fftn,
               then the (cheaper) permutation to O.  What a drop-in swap at the
               existing seam buys, transposes included.
    ffi_klead  FFI engine and k-LEADING storage, i.e. an encode that emits
               ``T`` as (k,b,μ,ν,t,s) — no transpose in, and the O permutation
               only.  The end state route (a) describes; the encode change it
               would need is priced separately by the bench.

Everything goes through ``common.fft_helpers`` factories.  There is no raw
``jnp.fft`` in this file, deliberately: the one-FFT-path rule
(``tests/test_fft_shardmap_context.py``) is a gate, and a prototype that
violates it cannot be promoted.

Dtype: the FFI handlers are **complex128 only** (both platforms).  The fp64
ladder path is c128 and is served; the ``--gmres-fp32`` route casts the whole
payload to c64 in ``bse_feast._build_gmres_data_fp32`` and would have to be
REFUSED here, not silently demoted.  ``rung_dtype_supported`` states that once.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import (
    make_flat_k_fftn,
    make_flat_k_ifftn,
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
)

__all__ = [
    "rung_shardings", "rung_dtype_supported",
    "make_conv_xla", "make_conv_ffi_kminor", "make_conv_ffi_kleading",
    "make_conv_ffi_fused",
    "to_O_from_kminor", "to_O_from_kleading", "to_O_from_fused",
    "flatten_W_R", "flatten_T", "unflatten_T",
    "make_sharded_ifftn_3d_ffi", "make_sharded_fftn_3d_ffi",
]


# ---------------------------------------------------------------------------
# Shardings.  The (μ, ν) placement is the production one everywhere: μ on 'x'
# (it comes from ψ_c^X), ν on 'y' (from ψ_v^Y).  Only the position of the k
# axis moves between the arms, which is the entire experiment.
# ---------------------------------------------------------------------------
def rung_shardings(mesh: Mesh) -> SimpleNamespace:
    return SimpleNamespace(
        # production storage
        T6=NamedSharding(mesh, P(None, "x", "y", None, None, None)),
        W5=NamedSharding(mesh, P("x", "y", None, None, None)),
        # k-leading (flat-k) storage
        T_flat=NamedSharding(mesh, P(None, None, "x", "y", None, None)),
        W_flat=NamedSharding(mesh, P(None, "x", "y")),
        # the fused gw_conv contract: G (nk, a, mx, b, my), W (nk, mx, my)
        T_g=NamedSharding(mesh, P(None, None, "x", None, "y")),
        # the decode's canonical O layout, contract_bands extra="leading"
        O=NamedSharding(mesh, P(None, None, None, "x", None, "y")),
    )


def rung_dtype_supported(dtype) -> tuple[bool, str]:
    """(ok, reason) — the FFI handlers are c128-only, on both platforms.

    Stated as a function rather than a comment because the honest answer for
    the ``--gmres-fp32`` ladder arm is a REFUSAL: ``bse_feast`` casts psi/W to
    ``complex64`` upstream, and a backend that silently up-casts would change
    the measured arm's numerics, not just its speed.
    """
    if jnp.dtype(dtype) == jnp.dtype(jnp.complex128):
        return True, "complex128"
    return False, (
        f"flat-k FFI handlers are complex128 only; the rung is {jnp.dtype(dtype)} "
        "(the fp32-GMRES ladder arm casts the payload in "
        "bse_feast._build_gmres_data_fp32).  Routing it here would have to "
        "refuse, not up-cast.")


# ---------------------------------------------------------------------------
# Layout helpers.  ``T`` production storage is (b, μ, ν, t, s, nk) with k
# MINOR; the flat-k contract is (nk, *trail) with k LEADING.
# ---------------------------------------------------------------------------
_TO_FLAT = (5, 0, 1, 2, 3, 4)      # (b,μ,ν,t,s,k) -> (k,b,μ,ν,t,s)
_FROM_FLAT = (1, 2, 3, 4, 5, 0)    # inverse
_O_FROM_KMINOR = (0, 5, 3, 1, 4, 2)   # (b,μ,ν,t,s,k) -> (b,k,t,μ,s,ν)
_O_FROM_KLEAD = (1, 0, 4, 2, 5, 3)    # (k,b,μ,ν,t,s) -> (b,k,t,μ,s,ν)


def flatten_T(T6):
    """(b, μ, ν, t, s, nk) -> (nk, b, μ, ν, t, s)."""
    return jnp.transpose(T6, _TO_FLAT)


def unflatten_T(T_flat):
    """(nk, b, μ, ν, t, s) -> (b, μ, ν, t, s, nk)."""
    return jnp.transpose(T_flat, _FROM_FLAT)


def flatten_W_R(W_R):
    """(μ, ν, kx, ky, kz) -> (nk, μ, ν).

    A ONCE-PER-SOLVE cost in any real integration: ``W_R`` is built outside the
    matvec by ``bse_feast.ensure_W_R`` and is q-independent, so a k-leading
    ladder would simply have the densifier emit this layout.  The bench times
    it separately for exactly that reason — it must not be charged per matvec.
    """
    mu, nu = W_R.shape[0], W_R.shape[1]
    nk = int(W_R.shape[2] * W_R.shape[3] * W_R.shape[4])
    return jnp.transpose(W_R.reshape(mu, nu, nk), (2, 0, 1))


def to_O_from_kminor(U6):
    """(b, μ, ν, t, s, nk) -> (b, k, t, μ, s, ν) — the production transpose
    (``bse_stack_matvec._build_bse_stack_matvec_full``)."""
    return jnp.transpose(U6, _O_FROM_KMINOR)


def to_O_from_kleading(U_flat):
    """(nk, b, μ, ν, t, s) -> (b, k, t, μ, s, ν)."""
    return jnp.transpose(U_flat, _O_FROM_KLEAD)


# ---------------------------------------------------------------------------
# The three convolution arms.  Each is ``conv(T, W) -> U`` in its own layout;
# the ``to_O_*`` helpers above complete the rung up to the decode, and the
# bench times the two halves separately so the transpose is attributable.
# ---------------------------------------------------------------------------
def make_conv_xla(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    """Production spelling, verbatim: reshape → sharded ifftn → broadcast
    multiply → sharded fftn → reshape.  ``T`` (b,μ,ν,t,s,nk), ``W_R``
    (μ,ν,kx,ky,kz)."""
    nkx, nky, nkz = (int(v) for v in kgrid)
    spec8 = P(None, "x", "y", None, None, None, None, None)
    ifftn = make_sharded_ifftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")
    fftn = make_sharded_fftn_3d(mesh, spec8, spec8, axes=(5, 6, 7), norm="ortho")

    def conv(T, W_R):
        nb, mu, nu, nt, ns, nk = T.shape
        T_k = T.reshape(nb, mu, nu, nt, ns, nkx, nky, nkz)
        T_R = ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        return fftn(U_R).reshape(nb, mu, nu, nt, ns, nk)

    return conv


def make_conv_ffi_kleading(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    """FFI engine on k-LEADING storage: ``T_flat`` (nk,b,μ,ν,t,s), ``W_flat``
    (nk,μ,ν).  No reshape and no transpose anywhere in the chain — the FFI
    reads the tile where it lies, which is the service's whole thesis."""
    spec = P(None, None, None, None, "x", "y", None, None)   # 3-D form + trail
    ifftn = make_flat_k_ifftn(mesh, kgrid, spec, norm="ortho")
    fftn = make_flat_k_fftn(mesh, kgrid, spec, norm="ortho")

    def conv(T_flat, W_flat):
        T_R = ifftn(T_flat)
        U_R = W_flat[:, None, :, :, None, None] * T_R
        return fftn(U_R)

    return conv


def make_conv_ffi_fused(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    """The rung as an instance of the ALREADY-CERTIFIED fused ``gw_conv`` entry.

    ``ensure_W_R`` builds ``W_R = ifftn(W_q, axes=k, norm='ortho')``
    (``bse_densify.make_w_densifier`` :190) and the rung then computes
    ``U = fftn_ortho(ifftn_ortho(T) · W_R)``.  Substituting the first into the
    second:

        U = fftn_ortho( ifftn_ortho(T) · ifftn_ortho(W_q) )

    which is ``make_gw_conv_ffi``'s formula
    ``sigma = fftn(ifftn(G) · ifftn(W)[:, None, :, None, :] · mult)`` at
    ``norm='ortho'``, ``mult=1`` — not an analogy, the same expression.  The
    handler's ``G (nk, a, mx, b, my)`` / ``W (nk, mx, my)`` layout contract is
    satisfied EXACTLY by ``a = n_trial``, ``mx = μ``, ``b = t·s`` (the two
    spinor axes merged — they are broadcast on both sides), ``my = ν``.

    Two consequences beyond speed, and they are the reason this arm exists:
    the R-space tile never materializes (the handler chunks internally), which
    is the ladder matvec's measured peak (audit §4: 5.3× the RPA matvec, "the
    T/T_R/U_R/U_q/U chain"); and ``W_q`` is consumed directly, so
    ``ensure_W_R``'s densifier — the F1 per-q retrace — has nothing left to do
    on this path.

    Operands: ``T_g`` (nk, b, μ, t·s, ν) and ``W_qf`` (nk, μ, ν), both c128.
    """
    from common.fft_helpers import make_flat_k_gw_conv
    g_spec = P(None, None, None, None, "x", None, "y")
    v_spec = P(None, None, None, "x", "y")
    return make_flat_k_gw_conv(mesh, kgrid, g_spec, v_spec, norm="ortho",
                               mult=1.0)


def to_O_from_fused(U_g, nspinor: int):
    """(nk, b, μ, t·s, ν) -> (b, k, t, μ, s, ν)."""
    nk, nb, mu, _, nu = U_g.shape
    return jnp.transpose(U_g.reshape(nk, nb, mu, nspinor, nspinor, nu),
                         (1, 0, 3, 2, 4, 5))


def make_conv_ffi_kminor(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    """FFI engine on PRODUCTION storage: one transpose in, the k-leading chain,
    one transpose out.  This is what a drop-in swap at the existing seam costs
    — the layout anchor moved onto the rung rather than removed."""
    inner = make_conv_ffi_kleading(mesh, kgrid)

    def conv(T, W_R):
        return unflatten_T(inner(flatten_T(T), flatten_W_R(W_R)))

    return conv


# ---------------------------------------------------------------------------
# Signature-compatible drop-in replacements for the two factories
# ``bse_stack_matvec`` imports. Rebinding these two names on that module before
# ``build_bse_stack_matvec(full=True)`` gives an FFI-served REAL matvec with no edit to
# a production file — which is how the end-to-end arm is measured while the
# integration agent owns the rung mechanics concurrently.
#
# This is the PESSIMISTIC wiring by construction: each factory pays its own
# transpose pair, so the rung carries four transposes instead of the one the
# k-leading route would.  It isolates the ENGINE.  Read it against
# ``make_conv_ffi_kminor`` (two transposes) and ``make_conv_ffi_kleading``
# (none) to separate engine from layout.
# ---------------------------------------------------------------------------
def _make_adapter(kind: str):
    def factory(mesh: Mesh, in_spec: P, out_spec: P, *, norm=None,
                axes=(-3, -2, -1)) -> Callable:
        ax = tuple(a % 8 if a < 0 else a for a in axes)
        if ax != (5, 6, 7):
            raise ValueError(
                f"the FFI drop-in adapter is written for the rung's 8-D form "
                f"(axes 5,6,7); got axes={axes}.")
        trail = tuple(in_spec)[:5]
        spec = P(None, None, None, *trail)
        cache: dict = {}

        def f(x8):
            kgrid = tuple(int(v) for v in x8.shape[5:8])
            fn = cache.get(kgrid)
            if fn is None:
                fn = (make_flat_k_ifftn if kind == "ifftn" else make_flat_k_fftn)(
                    mesh, kgrid, spec, norm=norm)
                cache[kgrid] = fn
            lead = x8.shape[:5]
            nk = kgrid[0] * kgrid[1] * kgrid[2]
            xf = jnp.transpose(x8.reshape(*lead, nk), _TO_FLAT)
            yf = fn(xf)
            return jnp.transpose(yf, _FROM_FLAT).reshape(x8.shape)

        return f

    return factory


#: Drop-in for ``common.fft_helpers.make_sharded_ifftn_3d`` at the rung's 8-D
#: call site.  Same signature, FFI engine inside.
make_sharded_ifftn_3d_ffi = _make_adapter("ifftn")
#: Drop-in for ``common.fft_helpers.make_sharded_fftn_3d``.
make_sharded_fftn_3d_ffi = _make_adapter("fftn")
