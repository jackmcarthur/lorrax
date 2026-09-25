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
in one pass; it is chosen when ``N`` lies in the device's measured GEMM range
(:data:`GEMM_CROSSOVER`, separately for full and supported axes; a supported
axis also needs ``n_in·n_out/N² ≤`` the row's bound).
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

A plane read from a cylinder (``in_gather=(plane_from_col, n_col)``, 2-D,
forward, ``norm='backward'``) is not separable: ``x (..., n_col)`` holds the
occupied cells of the ``(n_b, n_c)`` plane, ``plane_from_col[cell]`` the
column of each cell (``n_col`` on an empty one), and the plan returns
``fftn(take(x, plane_from_col, mode='fill').reshape(..., n_b, n_c))``.  It is
served by :func:`ffi.fft.make_plane_fft_gather`: the cuFFTDx gather-on-load
kernel on CUDA when it serves the plane (the zero plane is never written),
else the static-run concatenate + one 2-D FFT.  ``plan(Fa, start, size)``
reads the planes ``[start, start + size)`` of axis 1 of ``Fa`` in place.
"""

from __future__ import annotations

from itertools import permutations, product
from math import prod
from typing import Mapping

import numpy as np
import jax
import jax.numpy as jnp

from common.fft_helpers import local_fftn3, local_ifftn3


# Per ``device_kind`` prefix: the axis lengths ``N`` at which the stored-matrix
# GEMM beats the library FFT, for a full N→N axis and for an axis with a
# support (whose FFT arm pays the embedding gather or the restriction take).
# A missing device, or an ``N`` outside the range, means FFT.  Sweep:
# ``tests/bench/bench_fourier_plan.py``.
#
# A100, complex128, measured through the CUDA leg (one custom call): a batched
# cuFFT costs about one HBM pass for every N in 2..256, and the Fourier ZGEMMs
# (DMMA tensor cores, ~7 TFLOP/s at K ≈ 27–40) cost more than a pass, so a full
# axis never takes the GEMM (1-D 1.14–6.1× at 1e3–1e5 lines, 2-D/3-D grids 1.2–82×).  A supported
# axis's FFT arm pays the zero-fill embed, the transform and the restriction;
# the GEMM does all three in one pass: one way 0.50–0.82 and round trip
# 0.45–0.87 of the FFT arm on 16³–96³ and 24²–128² boxes.  12³ loses (1.16:
# launch-bound batched GEMMs), hence the floor at 16.
#
# A supported axis also needs a small support.  The GEMM costs K = n_in·n_out/N
# multiply-adds per grid element and the FFT arm about one pass, so the row's
# third entry bounds K/N.  Measured on A100 (``runs/runtime/
# f3_fft_adoption_20260925/sweep_support_fraction.out``: 2-D and 3-D, N 24–128,
# in and out supports), the GEMM wins 1.08–2.8× at every K/N ≤ 0.54 except 2-D
# out-support at small N.  The first loss above 0.54 is at N = 128, K/N = 0.55
# (out-support, 0.74–0.83×).  The bound 0.54 keeps Fe's 13/25 union box.
# The fourth entry is the supported-axis N range of a plan over one or two
# axes.  2-D out-supports lose at N 16–30 (0.60–0.97×; N = 27 wins 1.08–1.14×)
# and win from N = 32 (``sweep_small_n.out``).  3-D supports and 2-D
# in-supports win from N = 16/20 and keep the third entry's range.
# Missing entries mean no K/N bound and the same range.
GEMM_CROSSOVER: dict[str, tuple] = {
    "NVIDIA A100": (range(0), range(16, 129), 0.54, range(32, 129)),
}

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


def gemm_crossover(device_kind: str) -> tuple[range, range, float, range]:
    """``(full-axis N range, supported-axis N range, max supported n_in·n_out/N²,
    supported-axis N range of a 1- or 2-axis plan)`` taking the GEMM."""
    row = next((r for p, r in GEMM_CROSSOVER.items() if device_kind.startswith(p)),
               (range(0), range(0)))
    return (row[0], row[1], row[2] if len(row) > 2 else 1.0,
            row[3] if len(row) > 3 else row[1])


def _default_leg():
    """``None``: the leg follows the platform the call is lowered for.  Tests and
    the sweep force ``'xla'`` or ``'ffi'`` by patching this."""
    return None


def _cuda_present(mesh) -> bool:
    """Whether the plan can be lowered for CUDA: the mesh's platform, else any CUDA device."""
    if mesh is not None:
        from ffi.gate import mesh_ffi_platform
        return mesh_ffi_platform(mesh) == "CUDA"
    try:
        return len(jax.devices("cuda")) > 0
    except RuntimeError:
        return False


class LocalFourierPlan:
    """A jnp.fft-convention transform over ``axes`` with optional supports.

    ``extents`` are the full grid sizes ``N`` aligned with ``axes``;
    ``in_support``/``out_support`` map an axis (as written in ``axes``) to an
    integer index array.  ``plan(x)`` requires ``x.dtype == dtype`` and the
    compact extent on every in-support axis, the full extent elsewhere.
    ``out_perm`` returns ``jnp.transpose(y, out_perm)`` instead of ``y``, at no
    cost when the GEMM chain can write that order directly (see ``_route``).
    ``device_kind`` defaults to the kind of ``mesh``'s first device, else
    ``jax.devices()[0]``'s (it chooses GEMM axes only, never correctness).
    The leg is chosen when the call is lowered, from the platform it is
    lowered for (``jax.lax.platform_dependent``): one ``lorrax_fourier_plan``
    custom call on CUDA, XLA ops elsewhere, so a CPU operand in a GPU process
    takes the XLA leg.  Contract on every platform: complex128, at most three
    transform axes (``GATE fourier-plan-contract`` otherwise).
    ``plan.stages`` lists ``(axis, 'gemm'|'fft', n_in, n_out)``; GEMM stages
    of equal ``n_out/n_in`` may execute in either order.  ``in_gather`` (with
    the caller's ``mesh``) is the cylinder-plane form of the module docstring;
    only it takes ``plan(x, start, size)``.
    """

    def __init__(self, extents, axes, *, sign: int, norm: str | None = "backward",
                 dtype=jnp.complex128,
                 in_support: Mapping[int, np.ndarray] | None = None,
                 out_support: Mapping[int, np.ndarray] | None = None,
                 out_perm: tuple[int, ...] | None = None,
                 in_gather: tuple | None = None, mesh=None,
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
        if self.dtype != jnp.complex128 or not 1 <= len(axes) <= 3:
            raise ValueError(
                f"GATE fourier-plan-contract: got dtype {self.dtype} over {len(axes)} axes; "
                "want complex128 over 1 to 3 transform axes; why: the CUDA leg "
                "(lorrax_fourier_plan) serves exactly that, and the plan keeps one contract "
                "on every platform; fix: cast to complex128, or split the transform")
        in_support, out_support = dict(in_support or {}), dict(out_support or {})
        self._gather = None
        if in_gather is not None:
            if (axes != (-2, -1) or sign != -1 or (norm or "backward") != "backward"
                    or in_support or out_support or out_perm is not None or mesh is None):
                raise ValueError(
                    "LocalFourierPlan: in_gather is the forward 'backward'-norm plane "
                    "transform over axes (-2, -1) with no supports or out_perm, and "
                    f"needs the caller's mesh; got extents {extents}, axes {axes}, "
                    f"sign {sign}, norm {norm!r}, mesh {mesh!r}")
            from ffi.fft import make_plane_fft_gather
            plane_from_col, n_col = in_gather
            self._gather = make_plane_fft_gather(mesh, plane_from_col, int(n_col), extents)
            self.extents, self.axes, self.sign, self.norm = extents, axes, sign, norm
            self.stages = [(axes, "gather-fft", int(n_col), extents[0] * extents[1])]
            return
        for name, sup in (("in_support", in_support), ("out_support", out_support)):
            if set(sup) - set(axes):
                raise ValueError(f"LocalFourierPlan: {name} keys {sorted(sup)} "
                                 f"are not all in axes {axes}")
        # A support listing the whole axis in order is no support: that axis
        # is full, and a full axis takes the full-axis row (A100: the FFT).
        n_of = dict(zip(axes, extents))
        for sup in (in_support, out_support):
            for ax in [a for a, idx in sup.items()
                       if np.array_equal(np.asarray(idx).ravel() % n_of[a], np.arange(n_of[a]))]:
                del sup[ax]
        if device_kind is None:
            device_kind = (mesh.devices.flat[0] if mesh is not None else jax.devices()[0]).device_kind
        self.device_kind = str(device_kind)
        n_full, n_sup, kn_max, n_sup12 = gemm_crossover(self.device_kind)
        if len(axes) < 3:
            n_sup = n_sup12

        self.extents, self.axes, self.sign, self.norm = extents, axes, sign, norm
        gemm, fft, embed, take, self.stages = [], [], [], [], []
        self._axis = {}                 # ax -> (n, in_idx, out_idx, supported_in, supported_out)
        for n, ax in zip(extents, axes):
            i_idx = np.asarray(in_support.get(ax, np.arange(n)), dtype=np.int64).ravel() % n
            o_idx = np.asarray(out_support.get(ax, np.arange(n)), dtype=np.int64).ravel() % n
            if i_idx.size == 0 or o_idx.size == 0:
                raise ValueError(f"LocalFourierPlan: empty support on axis {ax}")
            if np.unique(i_idx).size != i_idx.size:
                raise ValueError(f"LocalFourierPlan: in_support on axis {ax} repeats an index")
            supported = ax in in_support or ax in out_support
            self._axis[ax] = (n, i_idx, o_idx, ax in in_support, ax in out_support)
            if (n in n_sup and i_idx.size * o_idx.size <= kn_max * n * n) if supported else n in n_full:
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
        # within a ratio the last-listed axis first (``_route`` may reorder
        # equal ratios and chooses where each GEMM writes its new axis).
        rank = {ax: -i for i, ax in enumerate(axes)}       # minor-most listed axis first
        order = sorted(range(len(gemm)), key=lambda s: (gemm[s][0], rank[gemm[s][1]]))
        pre = [("gemm", gemm[s][1], gemm[s][2]) for s in order if gemm[s][0] < 1]
        post = [("gemm", gemm[s][1], gemm[s][2]) for s in order if gemm[s][0] >= 1]
        mid = embed + ([("fft", tuple(fft), None)] if fft else []) + take
        self._ops = pre + mid + post
        self._groups = (pre, mid, post)
        self.out_perm = None if out_perm is None else tuple(int(a) for a in out_perm)
        self._routes = {}
        if _cuda_present(mesh):         # the CUDA leg's handler must be loaded before lowering
            from ffi.fft import FOURIER_PLAN_TARGET, _require_target
            _require_target(FOURIER_PLAN_TARGET, "CUDA")
        fft_stages = self.stages
        self.stages = ([(ax, "gemm", A.shape[1], A.shape[0]) for _, ax, A in pre]
                       + fft_stages
                       + [(ax, "gemm", A.shape[1], A.shape[0]) for _, ax, A in post])
        self._in_extent = {ax: (np.asarray(in_support[ax]).size if ax in in_support else n)
                           for n, ax in zip(extents, axes)}

    def __call__(self, x, start=None, size=None):
        if self._gather is not None:
            return self._gather(x) if start is None else self._gather(x, start, size)
        if start is not None:
            raise ValueError("LocalFourierPlan: the slab form (start, size) is in_gather only")
        if x.dtype != self.dtype:
            raise TypeError(f"LocalFourierPlan: x.dtype {x.dtype} != plan dtype {self.dtype}")
        for ax, k in self._in_extent.items():
            if x.shape[ax] != k:
                raise ValueError(f"LocalFourierPlan: axis {ax} of x has extent "
                                 f"{x.shape[ax]}, the plan expects {k}")
        leg = _default_leg()
        if leg is not None:
            return self._call_ffi(x) if leg == "ffi" else self._call_xla(x)
        return jax.lax.platform_dependent(x, cuda=self._call_ffi, default=self._call_xla)

    def _call_xla(self, x):
        """The XLA leg: GEMMs as ``dot_general``, the FFT group as one ``jnp.fft`` call."""
        nd = x.ndim
        target = list(range(nd)) if self.out_perm is None else [a % nd for a in self.out_perm]
        if sorted(target) != list(range(nd)):
            raise ValueError(f"LocalFourierPlan: out_perm {self.out_perm} is not a "
                             f"permutation of {nd} axes")
        key = (x.shape, tuple(target))
        if key not in self._routes:
            self._routes[key] = self._route(x.shape, target)
        _, ops, places = self._routes[key]
        fft = local_fftn3 if self.sign < 0 else local_ifftn3
        phys = list(range(nd))          # phys[p]: the logical axis stored at position p
        places = iter(places)
        for kind, ax, A in ops:
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
                if 0 < p < nd - 1:      # a middle axis: one transpose to make it minor
                    x = jnp.moveaxis(x, p, -1)
                    phys.append(phys.pop(p))
                    p = nd - 1
                phys.remove(a)
                if next(places):        # (rest, N'): the new axis minor
                    x = jax.lax.dot_general(x, jnp.asarray(A), (((p,), (1,)), ((), ())))
                    phys.append(a)
                else:                   # (N', rest): the new axis major
                    x = jax.lax.dot_general(jnp.asarray(A), x, (((1,), (p,)), ((), ())))
                    phys.insert(0, a)
        if phys != target:
            x = jnp.transpose(x, [phys.index(a) for a in target])
        return x

    def _call_ffi(self, x):
        """The CUDA leg: the transform axes are moved trailing (free when they
        already are), then one custom call executes the same stages row-major
        (GEMMs by cuBLAS with the matrix broadcast, the FFT group by cuFFT)."""
        from ffi.fft import fourier_plan_ffi
        nd = x.ndim
        phys = sorted(a % nd for a in self.axes)
        trailing = list(range(nd - len(phys), nd))
        if phys != trailing:
            x = jnp.moveaxis(x, phys, trailing)
        by_pos = {a % nd: a for a in self.axes}
        pos = {by_pos[p]: i for i, p in enumerate(phys)}      # plan axis -> trailing slot
        cols = [self._axis[by_pos[p]] for p in phys]
        gemm_axes = {ax for kind, ax, _ in self._ops if kind == "gemm"}
        order = []
        for kind, ax, _ in self._ops:
            if kind == "gemm":
                order.append(pos[ax])
            elif kind == "fft":
                order.append(-1)
        y = fourier_plan_ffi(
            x, n=[c[0] for c in cols], kin=[c[1].size for c in cols],
            kout=[c[2].size for c in cols], in_idx=np.concatenate([c[1] for c in cols]),
            out_idx=np.concatenate([c[2] for c in cols]), sup_in=[int(c[3]) for c in cols],
            sup_out=[int(c[4]) for c in cols],
            gemm=[int(by_pos[p] in gemm_axes) for p in phys],
            scale=[_axis_scale(c[0], self.sign, self.norm) for c in cols], order=order,
            sign=self.sign)
        if phys != trailing:
            y = jnp.moveaxis(y, trailing, phys)
        if self.out_perm is not None:
            y = jnp.transpose(y, [a % nd for a in self.out_perm])
        return y

    def _route(self, shape, target):
        """The GEMM order (within equal ratios) and output placements that
        move the fewest elements through explicit transposes.

        A GEMM contracts its axis where it lies when that axis is major or
        minor, and writes the new axis major or minor for free (the operand
        transposes live inside the GEMM); a middle axis costs one transpose,
        and so does a final order other than ``target``.  At most
        ``3!·2³ = 48`` candidates; the first minimum wins, so every rank
        chooses the same route.
        """
        nd = len(shape)
        pre, mid, post = self._groups

        def orders(group):              # permutations within runs of equal ratio
            runs = {}
            for op in group:
                runs.setdefault(op[2].shape[0] / op[2].shape[1], []).append(op)
            return [sum(c, []) for c in product(*[[list(q) for q in permutations(r)]
                                                  for r in runs.values()])]

        best = None
        for ops in (a + mid + b for a in orders(pre) for b in orders(post)):
            n_g = sum(op[0] == "gemm" for op in ops)
            for places in product((0, 1), repeat=n_g):
                dims, phys, cost, it = list(shape), list(range(nd)), 0, iter(places)
                for kind, ax, A in ops:
                    if kind == "fft":
                        continue
                    a = ax % nd
                    if kind != "gemm":
                        dims[a] = len(A)
                        continue
                    if 0 < phys.index(a) < nd - 1:
                        cost += prod(dims)
                    phys.remove(a)
                    dims[a] = A.shape[0]
                    phys = phys + [a] if next(it) else [a] + phys
                if phys != target:
                    cost += prod(dims)
                if best is None or cost < best[0]:
                    best = (cost, ops, places)
        return best
