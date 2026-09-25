"""LocalFourierPlan: one local separable DFT with restricted per-axis supports.

The plan computes, on one device and in the caller's precision,

    y = R_out · F_{s,norm} · E_in · x,

where ``F_{s,norm}`` is exactly ``jnp.fft.fftn`` (``sign=-1``) or
``jnp.fft.ifftn`` (``sign=+1``) over ``axes`` with ``jnp.fft``'s ``norm``
meaning for that direction.  ``E_in`` embeds a compact input: on an axis with
``in_support[ax] = idx`` (length ``K``), ``x`` has extent ``K`` and
``x[..., k, ...]`` sits at full-grid index ``idx[k] mod N``; every other index
is zero.  ``R_out`` restricts the output: on an axis with
``out_support[ax] = idx'`` (length ``K'``), ``y[..., k', ...]`` is the full
result at index ``idx'[k'] mod N``.  Supports are separable (one index set per
axis); a sphere or disk enters through its tight bounding box, the product of
its per-axis projections, which costs ~3 % more line work than exact sphere
pruning and keeps every stage a regular batched operation.

Each axis is one stage, a GEMM with a Fourier matrix stored in the plan,

    A_ax[j', j] = scale_ax · exp(sign · 2πi · (idx'[j'] · idx[j] mod N) / N),

or an FFT.  The integer phase ``idx'·idx mod N`` is reduced exactly before the
float64 exponential.  The GEMM does the embedding, transform and restriction
in one pass; it is chosen when ``N`` is at or below the device's measured
crossover (:data:`GEMM_CROSSOVER`, separately for full and supported axes).
FFT axes go to one ``fft_helpers.local_fftn3``/``local_ifftn3`` call (a single
multidimensional ``jnp.fft`` transform, so the library keeps its
multidimensional algorithm), preceded by one gather per embedded axis
(``take(mode='fill')``: the full axis reads zero off the support) and followed
by one take per restricted axis.  Stages run in increasing ``n_out/n_in``
order: shrinking GEMMs first, the FFT group next, expanding GEMMs last, so the
array is as small as possible at every stage.

The crossover table is a deterministic function of ``device_kind``: every rank
builds the same plan and there is no runtime autotune.  An unknown device kind
(and CPU) uses the FFT on every axis, so the O(N log N) scaling is kept
everywhere; a GEMM replaces an FFT only on axes whose length is a measured,
bounded constant.

The matrices are numpy arrays owned by the plan.  Under ``jit`` or inside a
``shard_map`` body they become constants of the executable; they are never
rebuilt per call.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import jax
import jax.numpy as jnp

from common.fft_helpers import local_fftn3, local_ifftn3


# Per ``device_kind`` prefix: the largest ``N`` at which the stored-matrix GEMM
# beats the library FFT, for a full N→N axis and for an axis with a support
# (whose FFT arm pays the embedding gather or the restriction take).  A missing
# device, and 0, mean FFT.  Sweep: ``tests/bench/bench_fourier_plan.py``.
#
# A100 (complex128, production XLA flags): a batched cuFFT costs about one HBM
# pass for every N in 2..256, so a full axis never gains from a GEMM past the
# launch-latency regime (≤1e3 lines).  A supported axis's FFT arm costs about
# five passes per sphere→box→sphere round trip (gathers, transform, takes); the
# rotated GEMM chain replaces them with one GEMM pass per axis and wins up to
# N = 128 at K = N/2 (0.46–0.90 of the FFT arm on 12³–96³ and 24²–128² boxes).
GEMM_CROSSOVER: dict[str, tuple[int, int]] = {"NVIDIA A100": (0, 128)}

_NORMS = (None, "backward", "ortho", "forward")


def _axis_scale(n: int, sign: int, norm: str | None) -> float:
    """``jnp.fft``'s per-axis scale for this direction (fftn: sign -1)."""
    if norm == "ortho":
        return 1.0 / np.sqrt(n)
    scaled = "forward" if sign < 0 else "backward"
    return 1.0 / n if (norm or "backward") == scaled else 1.0


def dft_matrix(n: int, out_idx, in_idx, *, sign: int, scale: float = 1.0,
               dtype=np.complex128) -> np.ndarray:
    """``A[j', j] = scale · exp(sign·2πi·(out_idx[j']·in_idx[j] mod n)/n)``.

    The phase index is reduced exactly in int64 and centred on ``(-n/2, n/2]``
    before the float64 exponential.
    """
    o = np.asarray(out_idx, dtype=np.int64) % n
    i = np.asarray(in_idx, dtype=np.int64) % n
    m = np.outer(o, i) % n
    m = np.where(2 * m > n, m - n, m)
    a = np.exp((sign * 2.0 * np.pi / n) * 1j * m.astype(np.float64))
    return (scale * a).astype(dtype)


def gemm_crossover(device_kind: str) -> tuple[int, int]:
    """``(full-axis max N, supported-axis max N)`` for ``device_kind``."""
    for prefix, n_max in GEMM_CROSSOVER.items():
        if device_kind.startswith(prefix):
            return n_max
    return 0, 0


class LocalFourierPlan:
    """A jnp.fft-convention transform over ``axes`` with optional supports.

    ``extents`` are the full grid sizes ``N`` aligned with ``axes``;
    ``in_support``/``out_support`` map an axis (as written in ``axes``) to an
    integer index array.  ``plan(x)`` requires ``x.dtype == dtype`` and the
    compact extent on every in-support axis, the full extent elsewhere.
    ``device_kind`` defaults to ``jax.devices()[0].device_kind``.
    ``plan.stages`` lists ``(axis, 'gemm'|'fft', n_in, n_out)`` in execution
    order.
    """

    def __init__(self, extents, axes, *, sign: int, norm: str | None = "backward",
                 dtype=jnp.complex128,
                 in_support: Mapping[int, np.ndarray] | None = None,
                 out_support: Mapping[int, np.ndarray] | None = None,
                 device_kind: str | None = None):
        extents, axes = tuple(int(n) for n in extents), tuple(int(a) for a in axes)
        if len(extents) != len(axes) or len(set(axes)) != len(axes):
            raise ValueError(f"LocalFourierPlan: extents {extents} and axes {axes} "
                             "must align one-to-one with distinct axes")
        if sign not in (-1, 1):
            raise ValueError(f"LocalFourierPlan: sign must be -1 or +1, got {sign!r}")
        if norm not in _NORMS:
            raise ValueError(f"LocalFourierPlan: norm must be one of {_NORMS}, got {norm!r}")
        self.dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(self.dtype, jnp.complexfloating):
            raise ValueError(f"LocalFourierPlan: dtype must be complex, got {self.dtype}")
        in_support, out_support = dict(in_support or {}), dict(out_support or {})
        for name, sup in (("in_support", in_support), ("out_support", out_support)):
            if set(sup) - set(axes):
                raise ValueError(f"LocalFourierPlan: {name} keys {sorted(sup)} "
                                 f"are not all in axes {axes}")
        if device_kind is None:
            device_kind = jax.devices()[0].device_kind
        self.device_kind = str(device_kind)
        n_full, n_sup = gemm_crossover(self.device_kind)

        self.extents, self.axes, self.sign, self.norm = extents, axes, sign, norm
        gemm, fft, embed, take, self.stages = [], [], [], [], []
        for n, ax in zip(extents, axes):
            i_idx = np.asarray(in_support.get(ax, np.arange(n)), dtype=np.int64).ravel() % n
            o_idx = np.asarray(out_support.get(ax, np.arange(n)), dtype=np.int64).ravel() % n
            if i_idx.size == 0 or o_idx.size == 0:
                raise ValueError(f"LocalFourierPlan: empty support on axis {ax}")
            if np.unique(i_idx).size != i_idx.size:
                raise ValueError(f"LocalFourierPlan: in_support on axis {ax} repeats an index")
            supported = ax in in_support or ax in out_support
            if n <= (n_sup if supported else n_full):
                A = dft_matrix(n, o_idx, i_idx, sign=sign,
                               scale=_axis_scale(n, sign, norm), dtype=self.dtype)
                gemm.append((o_idx.size / i_idx.size, ax, A))
                continue
            fft.append(ax)
            if ax in in_support:        # position of each full index in x; K → zero
                pos = np.full(n, i_idx.size, dtype=np.int32)
                pos[i_idx] = np.arange(i_idx.size, dtype=np.int32)
                embed.append(("embed", ax, pos))
            if ax in out_support:
                take.append(("take", ax, o_idx.astype(np.int32)))
            self.stages.append((ax, "fft", i_idx.size, o_idx.size))
        # Shrinking GEMMs first, then the FFT group (its embeddings, one
        # multidimensional transform, its restrictions), expanding GEMMs last;
        # within a ratio the last-listed axis first, so a chain of GEMMs
        # rotates through the trailing axes without transposes (__call__).
        rank = {ax: -i for i, ax in enumerate(axes)}       # minor-most listed axis first
        order = sorted(range(len(gemm)), key=lambda s: (gemm[s][0], rank[gemm[s][1]]))
        pre = [("gemm", gemm[s][1], gemm[s][2]) for s in order if gemm[s][0] < 1]
        post = [("gemm", gemm[s][1], gemm[s][2]) for s in order if gemm[s][0] >= 1]
        mid = embed + ([("fft", tuple(fft), None)] if fft else []) + take
        self._ops = pre + mid + post
        fft_stages = self.stages
        self.stages = ([(ax, "gemm", A.shape[1], A.shape[0]) for _, ax, A in pre]
                       + fft_stages
                       + [(ax, "gemm", A.shape[1], A.shape[0]) for _, ax, A in post])
        self._in_extent = {ax: (np.asarray(in_support[ax]).size if ax in in_support else n)
                           for n, ax in zip(extents, axes)}

    def __call__(self, x):
        if x.dtype != self.dtype:
            raise TypeError(f"LocalFourierPlan: x.dtype {x.dtype} != plan dtype {self.dtype}")
        for ax, k in self._in_extent.items():
            if x.shape[ax] != k:
                raise ValueError(f"LocalFourierPlan: axis {ax} of x has extent "
                                 f"{x.shape[ax]}, the plan expects {k}")
        fft = local_fftn3 if self.sign < 0 else local_ifftn3
        nd = x.ndim
        phys = list(range(nd))          # phys[p]: the logical axis stored at position p
        for i, (kind, ax, A) in enumerate(self._ops):
            if kind == "fft":
                x = fft(x, axes=tuple(phys.index(a % nd) for a in ax), norm=self.norm)
                continue
            a = ax % nd
            p = phys.index(a)
            if kind == "embed":         # one gather; out-of-range K reads zero
                x = jnp.take(x, jnp.asarray(A), axis=p, mode="fill", fill_value=0)
            elif kind == "take":
                x = jnp.take(x, jnp.asarray(A), axis=p)
            else:
                later = any(k == "gemm" for k, _, _ in self._ops[i + 1:])
                x, phys = _apply_axis_matrix(x, jnp.asarray(A), p, phys, rotate=later)
        if phys != sorted(phys):
            x = jnp.transpose(x, [phys.index(a) for a in range(nd)])
        return x


def _apply_axis_matrix(x, A, p: int, phys: list, *, rotate: bool):
    """``Σ_j A[j', j] x[..., j, ...]`` over physical axis ``p`` as ONE GEMM.

    Contracting the major or the minor axis needs no transpose; the new axis
    is written major (``A · Xᵀ``), so the next-minor axis becomes minor for the
    following stage, unless this is the last GEMM on the minor axis, which
    writes it back in place (``X · Aᵀ``).  A middle axis is first moved minor.
    Returns the result and its physical axis order.
    """
    nd = x.ndim
    if 0 < p < nd - 1:
        x = jnp.moveaxis(x, p, -1)
        phys = phys[:p] + phys[p + 1:] + [phys[p]]
        p = nd - 1
    if p == nd - 1 and not rotate:
        return jax.lax.dot_general(x, A, (((p,), (1,)), ((), ()))), phys
    y = jax.lax.dot_general(A, x, (((1,), (p,)), ((), ())))
    return y, [phys[p]] + phys[:p] + phys[p + 1:]
