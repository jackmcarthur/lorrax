"""Native 2-D face GEMM: batched SUMMA over bounded band panels on the square mesh."""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from jax import shard_map
from .resolve import mesh_platform


# Kernel lessons: the batched 2-D SUMMA Green build (numbers: sandbox claim ids).
# Over plain JAX: none; it is XLA collectives plus the local active-range GEMM, 1.03-1.43x slower
#   than the retired band-complete XLA gather, kept by the layout rule, not for speed (2949).
#   Against cuBLASMp (one SUMMA per k, fenced into the XLA stream): 1.6-2.7x faster, except Fe 8^3
#   at P16 (7.37 vs 5.11 ms) (2949).
# Did not pay: cuBLASMp for G, faster only in CrI3 conduction windows (1.16 vs 1.66 ms), 31.5 vs
#   6.8 ms on Fe 8^3 (2949); two K/2 panels, 1.85 ms at P16 but every band live on a rank, refused
#   (2951); panels held across a Sigma call, 1/p_x of psi per rank, withdrawn (2944); holding only
#   the first panel, Sigma tau -1 to -3% for 30 MB (2953, branch); conjugating the finished partner
#   tile, Fe 4^3 W peak 1.50 -> 1.59 GB (2951).
# Overlap: under XLA's default scheduler the prefetched gather runs on the compute stream, 0.01-0.02
#   ms per tau node overlapped (2953).  The latency-hiding scheduler hides it (Sigma tau -4 to -6% at
#   P16; 2953, 2958) but stays off: +0.5 GB/rank unpriced (2953), and before 676eeb9a2 remat
#   reordered a loop-counter read under it (2961).
# Decides it: at P16 the band-panel exchange is most of a build (all-gathers 1.2 of 1.85 ms; 2949).
def panel_matmul(a, b, *, mesh, panel_bytes, bounds=None, weights=None, partner=False,
                 transa="N", transb="N", compiler_options=None):
    """Multiply face matrices by a batched 2-D SUMMA over bounded contraction panels.

    Parameters
    ----------
    a : jax.Array
        Complex/real [q,m,k], sharded P(None,'x','y').
    b : jax.Array
        [q,k,n] with the same face layout.
    mesh : jax.sharding.Mesh
        Named x/y processor axes, a square mesh (decisions.md, square meshes).
        Matrix extents are already mesh-padded.
    panel_bytes : int
        Caller-admitted bytes per rank for the two live operand panels.
        Output and input faces are accounted for separately by the caller.
    bounds : jax.Array, optional
        Integer (q,2), replicated: per batch row, the half-open contraction
        interval [lo, hi) outside which the caller has already zeroed a (or
        b).  Each panel's local product runs only over the interval's columns
        in that panel (the local active-range GEMM), so the dropped columns
        cost no flops; the result is the same product.
    weights : jax.Array, optional
        (q,k) replicated contraction weights, ``a·diag(w)·b``.  Each panel's
        slice of ``a`` is scaled on its way into the all-gather, so no
        weighted copy of ``a`` is made.
    partner : bool
        Square mesh and 3-D ``b`` only: also return the conjugate-face
        product ``conj(a)·diag(w)·conj(b)`` from the SAME panel exchange,
        each gathered panel conjugated before its own local GEMM (no
        conjugated copy of a tile).  Returns the pair.

    transa, transb : str
        ``'N'``, ``'T'`` or ``'C'`` on the square-mesh route: ``a`` is then
        given as ``op(a)``'s transpose ``[q, k, m]`` (``b`` as ``[q, n, k]``)
        on the face, and its tile is moved to the transposed grid position
        by one ``ppermute`` and transposed locally, which on a square mesh is
        exactly the N-layout tile, so the panel loop runs unchanged. No
        distributed transpose, no second tile copy.
    compiler_options : dict, optional
        Passed to the kernel's ``jax.jit`` (the latency-hiding scheduler,
        which overlaps the prefetched panel gathers with the local GEMM).

    Returns
    -------
    jax.Array
        a @ b, with b's batch/sample axes and an x/y output face. Units
        multiply without any normalization. The output always stays x/y
        tiled; no rank ever holds a band-complete panel (on a p x p mesh a
        panel spans at most K/p contraction columns), and exchanged panels
        never exceed ``panel_bytes`` per rank, or one contraction column.  Every batch row rides in each
        panel exchange and each local GEMM: one collective per panel, not per q.

    Notes
    -----
    Measured against the alternatives (A100-40GB, per complex Green-sized
    product, CrI3 8x8 q=10 m=n=2904 k=144 / Fe 8³ q=59 m=n=2560 k=120):
    at P4 (2x2) the full-k gather (retired: band-complete) takes 2.68 / 6.59
    ms, this route (two K/2 panels) 3.22 / 6.79, cuBLASMp (one SUMMA per q)
    6.68 / 10.95; at P16 (4x4) the gather 1.86 / 5.14, this route (four /
    five K/p-bounded panels) 2.57 / 7.37, two K/2 panels 1.85 / 5.95 (every
    band column live on a rank: refused), cuBLASMp 6.88 / 5.11.
    """
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    if a.ndim != 3 or b.ndim not in (3, 4) or a.dtype != b.dtype:
        raise ValueError('panel_matmul requires rank-3 A and rank-3/4 B of one dtype')
    if transa not in ('N', 'T', 'C') or transb not in ('N', 'T', 'C'):
        raise ValueError('panel_matmul: transa/transb must be N, T or C')
    if (transa != 'N' or transb != 'N') and (px != py or b.ndim != 3 or partner or weights is not None):
        raise ValueError('panel_matmul: transposed operands need a square mesh, 3-D b, no weights or partner')
    q, m, k = (a.shape[0], a.shape[2], a.shape[1]) if transa != 'N' else a.shape
    kb, n = (b.shape[-1], b.shape[-2]) if transb != 'N' else (b.shape[-2], b.shape[-1])
    if b.shape[0] != q or kb != k:
        raise ValueError('panel_matmul batch/contraction extents disagree')
    if m % px or k % px or k % py or n % py:
        raise ValueError('panel_matmul requires producer-padded face extents')
    per_column = a.dtype.itemsize * q * (m // px + n // py)
    limit = max(1, int(panel_bytes) // per_column)   # at least one column per panel
    if b.ndim == 3 and px == py:
        # SUMMA on interleaved panels: every rank contributes `width` of its
        # own K columns to each panel, so a panel is ONE all-gather per operand
        # over the mesh axis, p·width <= K/p columns (one owner block): two
        # live panels (the one multiplied and the one prefetched) never hold
        # a band-complete row or column.  A kl with no such divisor ends in
        # one narrower panel.
        width = _interleaved_width(k, px, limit)
        kernel = _interleaved_kernel(mesh, q, m, k, n, width, bounds is not None,
                                     weights is not None, bool(partner), transa, transb,
                                     None if compiler_options is None else tuple(sorted(compiler_options.items())))
        args = (a, b)
        if bounds is not None:
            args += (jnp.asarray(bounds, jnp.int32).reshape(q, 2),)
        if weights is not None:
            args += (jnp.asarray(weights, a.dtype).reshape(q, k),)
        return kernel(*args)
    raise ValueError('panel_matmul requires a square mesh and a 3-D b')


def _interleaved_width(k, p, limit):
    """Local columns per interleaved panel: p·width <= K/p, and two live panels within ``limit`` columns."""
    kl = k // p
    cap = max(1, min(limit // (2 * p), kl // p if p > 1 else kl))
    n_panel = -(-kl // cap)
    return -(-kl // n_panel)


def _panel_contraction(mesh, active=True):
    """``(left, right, bounds, c)``: ``c + left[:, :, lo:hi] @ right[:, lo:hi]`` per row
    (``c=None``: a fresh product), the local active-range GEMM of this mesh's platform,
    accumulating in place (beta = 1, ``c`` aliased to the result).  ``bounds=None``
    (``active=False``): every column into a given ``c``, through the prepared target,
    whose interval is an attribute, so no bounds are read and the stream does not wait.
    The bounds are valid by construction (``_interleaved_kernel``), so no guard runs."""
    one = np.complex128(1.0)
    if mesh_platform(mesh) == "CUDA":
        from ._active_local_cuda import (_native, _native_out, _prepared_native,
                                         require_active_local_cuda,
                                         require_prepared_active_local_cuda)
        (require_active_local_cuda if active else require_prepared_active_local_cuda)()

        def contract(left, right, bounds, c):
            if bounds is None:
                every = np.array([0, left.shape[2]], np.int64)
                return _prepared_native(left, right, c, active_bounds=every, alpha=one, beta=one)
            if c is None:
                return _native_out(left, right, bounds, alpha=one)
            return _native(left, right, bounds, c, alpha=one, beta=one)
    else:
        from ._active_local import active_local_matmul

        def contract(left, right, bounds, c):
            if bounds is None:
                return c + left @ right
            weights = jnp.ones(left.shape[::2], left.dtype)
            return active_local_matmul(left, right, bounds, weights, c, alpha=one,
                                       beta=np.complex128(0.0) if c is None else one)
    return contract



def batch_gram(b, weights, bounds, *, mesh, nbatch, right=None, partner=False):
    """``W[q] = b[q][:, lo:hi]·diag(w[q])·c[q][:, lo:hi]†`` with whole parents per rank, out on the face.

    ``b`` ``(Bp,m,K)``, ``weights`` ``(Bp,K)`` and int32 ``bounds`` ``(Bp,2)`` are
    in the batch layout (:func:`distrib_la.batch_layout`; ``nbatch`` real rows of
    ``Bp``); ``right`` ``c`` ``(Bp,n,K)`` likewise, ``b`` itself when ``None``.
    Each rank contracts its own rows over their intervals with the same local
    active-range GEMM a :func:`panel_matmul` panel uses, and only ``W`` moves,
    batch to face (:func:`distrib_la.local_batch`): per call a rank sends its
    ``W`` rows, never the factors, which stay resident.  Returns
    ``(nbatch,m,n)`` at ``P(None,'x','y')``.  ``partner``: also the
    conjugate-factor product ``conj(b)·diag(w)·c^T`` from the same resident
    rows (the pair :func:`panel_matmul` returns with ``partner=True``), both
    moved in the one exchange.
    """
    from ._batch_reshard import local_batch
    contract = _panel_contraction(mesh)
    bounds = bounds.astype(jnp.int32)

    def gram(b, c, w, limits):
        value = contract(b * w[:, None, :], jnp.swapaxes(jnp.conj(c), -1, -2), limits, None)
        if not partner:
            return value
        return value, contract(jnp.conj(b) * w[:, None, :], jnp.swapaxes(c, -1, -2), limits, None)
    if right is None:
        return local_batch(lambda b, w, limits: gram(b, b, w, limits), mesh, resident=(0, 1, 2),
                           nbatch=nbatch)(b, weights, bounds)
    return local_batch(gram, mesh, resident=(0, 1, 2, 3), nbatch=nbatch)(b, right, weights, bounds)

@lru_cache(maxsize=64)
def _interleaved_kernel(mesh, q, m, k, n, width, active=False, weighted=False, partner=False,
                        transa="N", transb="N", compiler_options=None):
    """Batched SUMMA on a square mesh: K streamed in interleaved panels, one prefetched.

    Rank ``(x, y)`` holds the K block ``[y·K/p, (y+1)·K/p)`` of A and
    ``[x·K/p, (x+1)·K/p)`` of B.  Panel ``j`` takes local columns
    ``[j·w, (j+1)·w)`` of every block (the last panel may be narrower): the
    all-gather of A over ``y`` and of B over ``x`` then both hold the SAME
    global K set ``{i·K/p + j·w + t}``, in the same order, so the local
    product of the two gathered panels is that panel's exact contribution to
    the rank's own output tile.  No reduction follows.  Panel ``j+1`` is
    gathered before panel ``j`` is multiplied; XLA's default scheduler still
    runs the gather on the compute stream, so it does not overlap the GEMM
    (see the lessons above ``panel_matmul``).  ``active``: each row's
    interval ``[lo, hi)`` meets a panel in ONE run of panel positions (a
    suffix of the first live owner's segment, whole segments, a prefix of the
    last), so the local active-range GEMM contracts only that run.
    ``weighted``: a replicated ``(q, K)`` weight row scales each panel's
    local slice of A before its all-gather.  ``partner``: the
    raw panels are gathered and each product gets its own weighted (and, for
    the partner, conjugated) panel copy; two accumulators, no tile copy.
    """
    p = int(mesh.shape['x'])
    kl = k // p
    n_full, rest = divmod(kl, width)
    # Three or more panels: every panel after the first accumulates in place through the
    # beta = 1 GEMM.  XLA folds one straight-line ``c + a @ b`` into its GEMM, but of two
    # adjacent ones (a one-trip scan is inlined; a tail follows the last full panel) it
    # leaves one as an add that holds the product and the new sum beside the running sum,
    # two output tiles (runs/DEV/701_scanacc_20260930/aot).  Two panels stay on XLA.
    in_place = active or n_full + bool(rest) >= 3
    contract = _panel_contraction(mesh, active) if in_place else None
    owner = np.arange(p, dtype=np.int32)[None, :]

    # A transposed operand arrives as op's transpose on the face: rank (x, y)
    # holds S[k-block x, m-block y]. The N-layout tile A[m-block x, k-block y]
    # is conj(S[k-block y, m-block x])^T, the tile of rank (y, x): one
    # ppermute across the grid's diagonal, then a local (conjugate) transpose.
    across = tuple((x * p + y, y * p + x) for x in range(p) for y in range(p))

    def transposed(t, mode):
        t = lax.ppermute(t, ('x', 'y'), perm=across)
        t = jnp.swapaxes(t, -1, -2)
        return jnp.conj(t) if mode == 'C' else t

    def body(a, b, bounds, weights):
        if transa != 'N':
            a = transposed(a, transa)
        if transb != 'N':
            b = transposed(b, transb)

        def gather(off, wd):
            left = lax.dynamic_slice_in_dim(a, off, wd, axis=2)
            if weighted and not partner:
                start = lax.axis_index('y') * kl + off
                left = left * lax.dynamic_slice_in_dim(weights, start, wd, axis=1)[:, None, :]
            right = lax.dynamic_slice_in_dim(b, off, wd, axis=1)
            return (lax.all_gather(left, 'y', axis=2, tiled=True),
                    lax.all_gather(right, 'x', axis=1, tiled=True))

        def interval(off, wd):
            base = owner * kl + off
            start = owner * wd + jnp.clip(bounds[:, :1] - base, 0, wd)
            stop = owner * wd + jnp.clip(bounds[:, 1:] - base, 0, wd)
            live = stop > start
            hi = jnp.max(jnp.where(live, stop, 0), axis=1)
            lo = jnp.minimum(jnp.min(jnp.where(live, start, p * wd), axis=1), hi)
            return jnp.stack([lo, hi], axis=1).astype(jnp.int32)

        def product(cs, panel, off, wd):
            left, right = panel
            if partner:
                cols = (owner.T * kl + off + jnp.arange(wd)[None, :]).reshape(-1)
                wc = (jnp.take(weights, cols, axis=1)[:, None, :] if weighted
                      else jnp.ones((), left.dtype))
                pairs = ((left * wc, right), (jnp.conj(left) * wc, jnp.conj(right)))
            else:
                pairs = ((left, right),)
            outs = []
            for i, (lhs, rhs) in enumerate(pairs):
                c = None if cs is None else cs[i]
                if active:
                    outs.append(contract(lhs, rhs, interval(off, wd), c))
                elif c is None:
                    outs.append(lhs @ rhs)
                else:
                    outs.append(c + lhs @ rhs if contract is None else contract(lhs, rhs, None, c))
            return tuple(outs)

        cur = gather(0, width)
        cs = None
        if n_full >= 2:
            nxt = gather(width, width)
            cs = product(None, cur, 0, width)

            def step(carry, j):
                cs, panel = carry
                ahead = gather((j + 1) * width, width)
                return (product(cs, panel, j * width, width), ahead), None

            (cs, cur), _ = lax.scan(step, (cs, nxt), jnp.arange(1, n_full - 1), unroll=1)
        tail = gather(n_full * width, rest) if rest else None
        cs = product(cs, cur, (n_full - 1) * width, width)
        if tail is not None:
            cs = product(cs, tail, n_full * width, rest)
        return cs if partner else cs[0]

    face = NamedSharding(mesh, P(None, 'x', 'y'))
    rep = NamedSharding(mesh, P())
    extra = (('bounds',) if active else ()) + (('weights',) if weighted else ())

    def local(a, b, *rest_args):
        named = dict(zip(extra, rest_args))
        return body(a, b, named.get('bounds'), named.get('weights'))

    n_out = 2 if partner else 1
    out = (P(None, 'x', 'y'),) * 2 if partner else P(None, 'x', 'y')
    kernel = shard_map(local, mesh=mesh, in_specs=(P(None, 'x', 'y'),) * 2 + (P(),) * len(extra),
                       out_specs=out, check_vma=False)
    options = None if compiler_options is None else dict(compiler_options)
    return jax.jit(kernel, in_shardings=(face, face) + (rep,) * len(extra),
                   out_shardings=(face,) * n_out if partner else face, compiler_options=options)
