"""Placement-independent reductions over mesh axes, for use inside ``shard_map``.

A collective library's all-reduce and reduce-scatter add the P per-rank terms in
an order its rings or trees take from the physical topology of the devices the
job was given, so the same job on another device set, or with ranks paired
differently across packages, differs in the last bit (RCCL on MI250X, det lane
2026-10-16: Fe V2 sigma_c 4e-7 eV and eqp print flips from a 1-ulp psi(r_mu)
reduce-scatter amplified by the ISDF fit).  These functions move the terms with
``all_to_all`` / ``all_gather`` (pure data movement, bitwise whatever the route)
and add them on each device in rank order, so the result depends only on the
rank count.

Exact cases go straight to the library collective, so they cost nothing:
two ranks (``a + b == b + a`` in IEEE arithmetic, so every route gives the same
bits), one rank, and integer or boolean operands.

Memory.  The library reduce-scatter writes a ``1/n`` output; the ordered one
also needs the ``all_to_all`` output, a second buffer the size of the operand.
Above ``slab_bytes`` (default 8 MiB, ``LORRAX_ORDERED_SLAB_MB``) the operand is
reduced in slabs inside a ``fori_loop`` (the all-reduce in place, in the operand's own
buffer, as the library all-reduce is), so the extra transient is a few slabs whatever
the operand size.  Every element is still summed ``c[0] + ... +
c[n-1]``, so the slabbed and one-shot results are bitwise equal (the last slab's
start is clamped and its overlap is recomputed to the same bits).

Every other floating sum over a mesh axis in the tree must call these (the
reduction lint in ``tests/test_reduction_lint.py``); a raw ``lax.psum`` stays
only where the sum is exact at any rank count and says so with an
``# exact-reduce: <reason>`` comment.  :func:`install_reduction_guard` (env
``LORRAX_REDUCTION_GUARD=log|raise``, installed by ``runtime``) checks the same rule
on the programs a run actually traces and records each ordered call's payload.

JAX imports stay inside the functions so importing :mod:`lxkit` stays stdlib-only.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ordered_psum_scatter", "ordered_psum", "ordered_reduce", "slab_bytes",
           "install_reduction_guard", "install_reduction_guard_from_env",
           "reduction_trace", "EXACT_MARKER"]

#: The comment that exempts a raw library reduction: the sum is exact at any rank
#: count (integer operand, one owner per element, zeros), followed by the reason.
EXACT_MARKER = "exact-reduce:"

DEFAULT_SLAB_BYTES = 8 << 20
_TRACE: list = []
_GUARD = {"mode": None}


def slab_bytes() -> int:
    """``LORRAX_ORDERED_SLAB_MB`` (float MiB, default 8; malformed or ``<= 0``: the default)."""
    try:
        mb = float(os.environ.get("LORRAX_ORDERED_SLAB_MB", "") or 0)
    except ValueError:
        mb = 0.0
    return int(mb * (1 << 20)) if mb > 0 else DEFAULT_SLAB_BYTES


def _axis_size(axis_name) -> int:
    from jax import lax
    names = axis_name if isinstance(axis_name, (tuple, list)) else (axis_name,)
    n = 1
    for a in names:
        n *= int(lax.axis_size(a))
    return n


def _inexact(dtype) -> bool:
    import numpy as np
    return np.issubdtype(np.dtype(dtype), np.inexact)


def _rank_order_sum(c):
    """``c[0] + c[1] + ... + c[n-1]``, left to right (the order IS the contract)."""
    acc = c[0]
    for j in range(1, int(c.shape[0])):
        acc = acc + c[j]
    return acc


def _caller_site():
    """``file:line`` of the first frame outside this module and outside jax."""
    f = sys._getframe(1)
    here = os.path.abspath(__file__)
    while f is not None:
        fn = f.f_code.co_filename
        if os.path.abspath(fn) != here and f"{os.sep}jax{os.sep}" not in fn \
                and f"{os.sep}jaxlib{os.sep}" not in fn:
            return f"{fn}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _record(op, x, n, path, extra_bytes, rccl_extra_bytes):
    if _GUARD["mode"] is None:
        return
    _TRACE.append({"site": _caller_site(), "op": op, "n": n,
                   "shape": [int(s) for s in x.shape], "dtype": str(x.dtype),
                   "bytes": int(x.size) * x.dtype.itemsize, "path": path,
                   "extra_bytes": int(extra_bytes),
                   "rccl_extra_bytes": int(rccl_extra_bytes)})


def _a2a_sum(xc, axis_name):
    """(a, n, b) slab -> (a, b): row-block p of every rank to rank p, summed in rank order."""
    import jax.numpy as jnp
    from jax import lax
    c = lax.all_to_all(jnp.moveaxis(xc, 1, 0), axis_name, 0, 0, tiled=True)
    return _rank_order_sum(c)


def _ordered_rs_3d(x3, axis_name, n, slab):
    """Ordered reduce-scatter of ``x3`` (A, n, B) over its middle axis -> (A, B).

    Extra transient: the slab copy, its ``all_to_all`` output and the transposed
    copy, each at most ``slab`` bytes (one-shot: ``2 * x3.nbytes``)."""
    import jax.numpy as jnp
    from jax import lax
    A, _, B = (int(s) for s in x3.shape)
    item = x3.dtype.itemsize
    if A * n * B * item <= slab:
        return _a2a_sum(x3, axis_name), "one-shot", 2 * A * n * B * item
    row = n * B * item
    if row <= slab:
        a, b = min(A, max(1, slab // row)), B
    else:
        a, b = 1, min(B, max(1, slab // (n * item)))
    ka, kb = -(-A // a), -(-B // b)
    out0 = jnp.zeros_like(x3[:, 0, :])

    def body(i, out):
        ia = jnp.minimum((i // kb) * a, A - a)
        ib = jnp.minimum((i % kb) * b, B - b)
        xc = lax.dynamic_slice(x3, (ia, 0, ib), (a, n, b))
        return lax.dynamic_update_slice(out, _a2a_sum(xc, axis_name), (ia, ib))

    out = lax.fori_loop(0, ka * kb, body, out0)
    return out, f"slabbed {ka * kb}x({a},{n},{b})", 3 * a * n * b * item


def ordered_psum_scatter(x, axis_name, *, scatter_dimension: int = 0, tiled: bool = False,
                         slab_bytes: int | None = None):
    """``lax.psum_scatter`` with the terms added in rank order on every placement.

    Same arguments and result as ``jax.lax.psum_scatter(x, axis_name,
    scatter_dimension=..., tiled=...)``: device ``p`` (row-major index over
    ``axis_name``) gets the sum over ranks of slice ``p`` of ``x`` along
    ``scatter_dimension``.  ``all_to_all`` then ``n - 1`` adds, in slabs of at most
    ``slab_bytes`` (default :func:`slab_bytes`).
    """
    import jax.numpy as jnp
    from jax import lax
    x = jnp.asarray(x)
    n = _axis_size(axis_name)
    d = scatter_dimension % x.ndim
    if n <= 2 or not _inexact(x.dtype):
        return lax.psum_scatter(x, axis_name, scatter_dimension=d, tiled=tiled)
    size = int(x.shape[d])
    if tiled:
        if size % n:
            raise ValueError(f"ordered_psum_scatter: dimension {d} of size {size} is not divisible by {n} devices")
        m = size // n
    elif size != n:
        raise ValueError(f"ordered_psum_scatter: untiled dimension {d} has size {size}; want the axis size {n}")
    else:
        m = 1
    pre, post = x.shape[:d], x.shape[d + 1:]
    A = 1
    for s in pre:
        A *= int(s)
    B = m
    for s in post:
        B *= int(s)
    slab = int(slab_bytes if slab_bytes is not None else globals()["slab_bytes"]())
    out, path, extra = _ordered_rs_3d(x.reshape(A, n, B), axis_name, n, slab)
    _record("psum_scatter", x, n, path, extra, 0)
    out_shape = pre + ((m,) if tiled else ()) + post
    return out.reshape(out_shape)


def _ordered_psum_leaf(x, axis_name, slab):
    import jax.numpy as jnp
    from jax import lax
    x = jnp.asarray(x)
    n = _axis_size(axis_name)
    if n <= 2 or not _inexact(x.dtype):
        return lax.psum(x, axis_name)
    shape, L = x.shape, int(x.size)
    item = x.dtype.itemsize
    flat = x.reshape(-1)
    if L * item <= slab or L < 2 * n:
        pad = (-L) % n
        if pad:
            flat = jnp.concatenate([flat, jnp.zeros((pad,), flat.dtype)])
        part = _a2a_sum(flat.reshape(1, n, -1), axis_name)[0]
        full = lax.all_gather(part, axis_name, tiled=True)
        _record("psum", x, n, "one-shot", 3 * (L + pad) * item, 0)
        return (full[:L] if pad else full).reshape(shape)
    b = max(1, min(L // n, slab // (n * item)))
    w = n * b
    k = L // w

    def body(i, buf):
        # Full slabs are disjoint, so the carry is the operand itself: each slab is read
        # before it is overwritten and no summed value is read back (no new S buffer).
        part = _a2a_sum(lax.dynamic_slice(buf, (i * w,), (w,)).reshape(1, n, b), axis_name)[0]
        return lax.dynamic_update_slice(buf, lax.all_gather(part, axis_name, tiled=True), (i * w,))

    out = lax.fori_loop(0, k, body, flat)
    t = L - k * w
    if t:                                           # static tail, padded to the rank count
        tp = -(-t // n) * n
        tail = jnp.concatenate([out[k * w:], jnp.zeros((tp - t,), flat.dtype)]) if tp > t else out[k * w:]
        part = _a2a_sum(tail.reshape(1, n, tp // n), axis_name)[0]
        out = lax.dynamic_update_slice(out, lax.all_gather(part, axis_name, tiled=True)[:t], (k * w,))
    _record("psum", x, n, f"slabbed {k}x{w}+{t}", 3 * w * item, 0)
    return out.reshape(shape)


def ordered_psum(x, axis_name, *, slab_bytes: int | None = None):
    """``lax.psum`` with the terms added in rank order on every placement (any pytree).

    Per leaf: an ordered reduce-scatter of the flattened operand, then an
    ``all_gather`` -- the two halves of a ring all-reduce, with the adds in rank
    order -- in slabs of at most ``slab_bytes`` (default :func:`slab_bytes`)."""
    import jax
    slab = int(slab_bytes if slab_bytes is not None else globals()["slab_bytes"]())
    return jax.tree_util.tree_map(lambda leaf: _ordered_psum_leaf(leaf, axis_name, slab), x)


def _spec_axes(spec) -> tuple:
    out = []
    for entry in tuple(spec):
        if entry is None:
            continue
        out += list(entry) if isinstance(entry, (tuple, list)) else [entry]
    return tuple(out)


_REDUCE_PROGRAMS: dict = {}


def ordered_reduce(local_fn, *arrays):
    """A global sum over sharded ``jax.Array`` s, outside ``shard_map``, in rank order.

    ``jnp.sum`` / ``tensordot`` / ``vdot`` / ``norm`` on a mesh-sharded array lower to a
    library all-reduce that XLA inserts itself (no ``lax.psum`` in the source, so neither
    the lint nor the guard sees it), and its order follows the device topology.  Here
    ``local_fn(*shards)`` maps each rank's shards to that rank's PARTIAL of the global
    sum -- it must sum away every sharded dimension of its operands -- and the partials
    are added with :func:`ordered_psum` over the mesh axes the operands are sharded on.
    The result is replicated.  Operands without a ``NamedSharding`` enter replicated; when
    none is mesh-sharded ``local_fn`` runs on the whole arrays.  ``local_fn`` is a cache
    key (one program per function, mesh and specs): pass a module-level function.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        raise TypeError("ordered_reduce runs on concrete arrays (eager code): inside a traced "
                        "program the operand layout is not known here, and falling back to "
                        "local_fn would hand the sum to an XLA all-reduce.  Put the reduction "
                        "in a shard_map and use lxkit.ordered_psum.")
    shs = [getattr(a, "sharding", None) for a in arrays]
    named = [sh for sh in shs if isinstance(sh, NamedSharding)]
    if not named or all(not _spec_axes(sh.spec) for sh in named):
        return local_fn(*arrays)
    mesh = named[0].mesh
    specs = tuple(sh.spec if isinstance(sh, NamedSharding) else P() for sh in shs)
    axes = tuple(a for a in mesh.axis_names if any(a in _spec_axes(sp) for sp in specs))
    key = (local_fn, mesh, specs, axes)
    prog = _REDUCE_PROGRAMS.get(key)
    if prog is None:
        if len(_REDUCE_PROGRAMS) > 64:
            _REDUCE_PROGRAMS.clear()
        prog = _REDUCE_PROGRAMS[key] = jax.jit(jax.shard_map(
            lambda *loc: ordered_psum(local_fn(*loc), axes), mesh=mesh, in_specs=specs,
            out_specs=P(), check_vma=False))
    return prog(*arrays)


# -- the runtime guard ---------------------------------------------------------------

def reduction_trace() -> list:
    """Records of the reductions traced since the guard was installed (dicts)."""
    return list(_TRACE)


def _call_lines(frame) -> tuple:
    """First and last source line of the call expression the frame is executing."""
    try:
        pos = list(frame.f_code.co_positions())[frame.f_lasti // 2]
        if pos[0] is not None:
            return pos[0], pos[1] if pos[1] is not None else pos[0]
    except Exception:
        pass
    return frame.f_lineno, frame.f_lineno


def _marked(site_file: str, lines: tuple) -> bool:
    """The ``exact-reduce:`` comment on a line of the call itself (as the lint requires)."""
    import linecache
    return any(EXACT_MARKER in linecache.getline(site_file, ln) for ln in range(lines[0], lines[1] + 1))


def install_reduction_guard(mode: str = "log", dump_path: str | None = None) -> None:
    """Wrap ``jax.lax.psum`` / ``psum_scatter`` / ``pmean`` so a raw call from outside
    jax and lxkit, on a floating operand over more than two ranks and without the
    ``exact-reduce:`` comment, is recorded (``mode="log"``) or refused
    (``mode="raise"``).  Installed before the drivers import, so ``from jax.lax
    import psum`` binds the wrapper too.  ``dump_path`` (``{rank}`` expanded) gets
    the JSON trace at exit."""
    import atexit
    import json
    import jax
    from jax import lax
    if mode not in ("log", "raise"):
        raise ValueError(f"install_reduction_guard: mode {mode!r}; want 'log' or 'raise'")
    if _GUARD["mode"] is not None:
        _GUARD["mode"] = mode
        return
    _GUARD["mode"] = mode
    here = os.path.abspath(__file__)

    def wrap(name, raw):
        def guarded(x, axis_name, *args, **kw):
            f = sys._getframe(1)
            fn = f.f_code.co_filename
            if os.path.abspath(fn) == here or f"{os.sep}jax{os.sep}" in fn:
                return raw(x, axis_name, *args, **kw)
            try:
                n = _axis_size(axis_name)
            except Exception:
                return raw(x, axis_name, *args, **kw)
            leaves = jax.tree_util.tree_leaves(x)
            floating = [l for l in leaves if _inexact(getattr(l, "dtype", type(l)))]
            if n > 2 and floating:
                marked = _marked(fn, _call_lines(f))
                rec = {"site": f"{fn}:{f.f_lineno}", "op": f"raw {name}", "n": n,
                       "shape": [int(s) for s in getattr(floating[0], "shape", ())],
                       "dtype": str(getattr(floating[0], "dtype", "")),
                       "bytes": int(sum(getattr(l, "size", 1) * getattr(getattr(l, "dtype", None), "itemsize", 8)
                                        for l in floating)),
                       "path": "exact-marked" if marked else "UNORDERED"}
                _TRACE.append(rec)
                if not marked and _GUARD["mode"] == "raise":
                    raise RuntimeError(
                        f"raw lax.{name} over {n} ranks on a floating operand at {rec['site']}: "
                        f"the sum order follows the device topology.  Use lxkit.ordered_{'psum_scatter' if name == 'psum_scatter' else 'psum'}, "
                        f"or mark the line '# {EXACT_MARKER} <reason>' if the sum is exact at any rank count.")
            return raw(x, axis_name, *args, **kw)
        guarded.__wrapped__ = raw
        return guarded

    for name in ("psum", "psum_scatter", "pmean"):
        raw = getattr(lax, name)
        setattr(lax, name, wrap(name, raw))
    if dump_path:
        def _dump():
            try:
                rank = jax.process_index()
            except Exception:
                rank = 0
            with open(dump_path.replace("{rank}", str(rank)), "w") as fh:
                json.dump(_TRACE, fh, indent=0)
        atexit.register(_dump)


def install_reduction_guard_from_env() -> None:
    """``LORRAX_REDUCTION_GUARD=log|raise`` installs the guard; ``LORRAX_REDUCTION_TRACE=<path>``
    (``{rank}`` expanded) dumps the trace at exit.  Unset: nothing happens."""
    mode = os.environ.get("LORRAX_REDUCTION_GUARD", "").strip()
    if mode:
        install_reduction_guard(mode, os.environ.get("LORRAX_REDUCTION_TRACE") or None)
