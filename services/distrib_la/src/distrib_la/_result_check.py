"""Checked dense results: no distributed library result leaves this service unchecked.

The distributed libraries fail silently. cuSOLVERMp 0.9.1's syevd returned
wrong eigenvectors with status 0 and info 0 on valid Hermitian matrices: an
exact-zero block (Fe 4^3 bispinor CT metric, n 2688), rank-deficient PSD
responses (n 432 to 18304) and a sentinel-padded H'_vv (CrI3, n 9152), all
reproduced outside LORRAX. So every eigh takes its exact-zero rows out of the
solver (:func:`deflate_zero_rows`), and every distributed eigh and LU solve is
checked on its own output against fixed-seed random probes before it is
returned (:func:`checked`), at O(n^2 k) beside the O(n^3) solve. There is one
attempt: a result that fails its check is NaN-poisoned on every rank (an
eigh's eigenvalues, a solve's X), named on rank 0's stderr (GATE distrib_la_result_check) and raised, as a consumed LU
solve always was. The device returns each check's
errors with its flag (:func:`checked`) and the host prints the notices after
the call, so no program carries a host callback: JAX's persistent compile
cache never stores a program with one, and every checked program would
compile again in every process. There is no dial.

Acceptance (:func:`accept`) is on backward errors, which a backward-stable
solver keeps near n * eps whatever the condition number:
  eigh   ||(A V - V diag(w)) X|| / (||A|| ||X||) and ||(V^H V - I) X|| / ||X||
  solve  ||W^H (A X - B)|| / (sqrt(k) (||A|| ||X|| + ||B||))
with X, W fixed-seed Gaussian probes (k = :data:`PROBES` columns, identical on
every rank) and Frobenius norms, accepted at distrib_la.roundoff_tol(n, dtype)
= 64 n eps (4e-12 at n 432, 2.6e-10 at n 18304). Every correct result measured
sits at or below 1e-14; every failure at or above 8e-6.

Scope: the probes catch gross errors and missing, duplicated or
non-orthogonal eigenpairs. They are a projection, so one eigenvalue wrong by
delta passes while delta <~ accept(n) sqrt(n) ||A||_F.
"""
from __future__ import annotations

import contextvars
import os
import sys
import traceback

import jax
from lxkit import ordered_norm, ordered_sum
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

#: Probe columns per check.
PROBES = 8
#: The probe seed: every rank draws the same probes, so every rank reaches
#: the same verdict from the same reduced norms.
_SEED = 20261002


def accept(n, dtype=np.complex128):
    """The accepted backward error of a checked result of side ``n``: roundoff_tol(n, dtype)."""
    from distrib_la.tolerance import roundoff_tol
    return roundoff_tol(n, dtype)


def spectral_shift(a, *, mesh=None):
    """The always-on eigh shift ``s = ||A||_F / sqrt(n)`` of each matrix of ``a`` (batch shape).

    A checked eigh solves ``A + s I`` and moves the values back by ``s``
    (:func:`shifted`), one attempt on every backend. Measured (largenmu,
    2026-10-19; checked on the columns ``w > 1e-8 max(w)``): cuSOLVERMp 0.9.1
    (P=4, 2x2) returns wrong kept pairs for the exactly low-rank operand
    ``B^H B / n`` (``B`` n/4 x n) at n=9152 in 4 of 6 seeds x blocks (residual ~3e-4,
    orthogonality ~0.1), and this shift repairs 9 of 9 measured cases (those six,
    flat and graded 18304, full-rank Hermitian 9152; residual <= 1e-16,
    orthogonality <= 7e-15). ``+max|diag|`` and ``+||A||_F`` do not. On ELPA
    2026.02.001 it is harmless (all columns pass). It does NOT give orthonormal
    vectors inside an exactly degenerate cluster (rounding-level eigenvalues,
    all-column orthogonality 2e-2..3e-1 there): a caller that checks ALL
    columns of such an operand refuses by name; none of our operands measured
    has one. The eigenvalues move by ~``s eps <= ||A||_2 eps``.
    """
    square = jnp.abs(a) ** 2      # rank-ordered on a face (deflate_zero_rows' rule): no topology order
    total = (jnp.sum(square, axis=(-2, -1)) if mesh is None else
             ordered_sum(square, axis=(-2, -1), mesh=mesh, spec=_face(a.ndim)))
    return jnp.sqrt(total / a.shape[-1])


def shifted(eigh, s):
    """``eigh`` (``a -> (w, v)``) solved on ``a + s I``, its values moved back by ``s``."""
    def run(a):
        w, v = eigh(a + s[..., None, None].astype(a.dtype) * _eye_like(a))
        return w - s[..., None].astype(w.dtype), v
    return run


def _eye_like(a):
    """Identity broadcast to ``a``'s matrix axes, built inside the caller's program."""
    n = a.shape[-1]
    return (jax.lax.broadcasted_iota(jnp.int32, (n, n), 0)
            == jax.lax.broadcasted_iota(jnp.int32, (n, n), 1)).astype(a.dtype)


def _face(nd):
    """The face tile of a (..., n, n) stack: rows on 'x', columns on 'y'."""
    return P(*((None,) * (nd - 2)), "x", "y")


def _rows(nd):
    """A tall (..., n, k) stack: rows on 'x' (k probe columns whole)."""
    return P(*((None,) * (nd - 2)), "x", None)


def _norm(a, mesh=None, spec=None):
    """Frobenius norm of the matrix axes.  With ``mesh`` the ranks' partials are added in
    rank order on ``spec`` (lxkit.ordered_norm), not by an XLA all-reduce in topology order."""
    if mesh is None:
        return jnp.linalg.norm(a, axis=(-2, -1))
    return ordered_norm(a, axis=(-2, -1), mesh=mesh, spec=spec)


def ascend(a, values, vectors, *, constrain=None):
    """``(values, vectors)`` of ``a`` in ascending order where main reordered: only when ``a`` has an
    exact zero row and a live eigenvalue is negative (the deflated zeros come out first, so they
    sit out of order only then). A PSD operand, or one with no zero row, passes through unchanged;
    the reorder is a column permutation of a distributed result, so it runs only in that case.
    A positional consumer (bandstructure.bse_setup's band window on the mesh-aligned f(H)) reads
    the spectrum of the zero-padded matrix, as on main.
    """
    pin = (lambda r: r) if constrain is None else constrain
    need = jnp.any(jnp.any(jnp.all(a == 0, axis=-1), axis=-1) & jnp.any(values < 0, axis=-1))

    def reorder(operands):
        w, v = operands
        order = jnp.argsort(w, axis=-1, stable=True)
        return pin((jnp.take_along_axis(w, order, axis=-1), jnp.take_along_axis(v, order[..., None, :], axis=-1)))
    return pin(jax.lax.cond(need, reorder, pin, (values, vectors)))


def deflate_zero_rows(eigh, *, constrain=None, mesh=None, ascending=True):
    """Wrap a Hermitian eigensolver so exact zero rows never reach it.

    Capacity padding and unselected columns reach eigensolvers as exact zero
    rows/columns, and a large zero block breaks them: the native solver
    returned nonfinite values (Na P16, 1460 of 2584 rows), cuSOLVERMp wrong
    eigenpairs with info 0. Those rows are decoupled, so they are replaced by
    distinct diagonal sentinels in [-4b, -2b], b the Gershgorin bound of the
    rest (relative to A, so a small-norm matrix keeps the gap), solved, and
    reported back as exact zero eigenvalues with their unit eigenvectors.

    THE CONTRACT (main's): the ascending spectrum of the zero-padded matrix.
    The deflated pairs (value 0, unit vectors, lowest row index first) leave
    the solver first, which is in order unless a live eigenvalue is negative;
    only then are the pairs sorted (:func:`ascend`, a column permutation of a
    distributed result). Positional consumers rely on it (bandstructure.
    bse_setup's band window on the mesh-aligned f(H): 1405 meV without it,
    hsuite 2026-10-19). ``ascending=False`` keeps the solver's order for
    Plan._checked_eigh, which drops its leading pads by count and then
    ascends. A matrix with no zero row passes through unchanged, sorted.
    The service pads a solve side the same way (:func:`pad_side`): its zero
    rows have the lowest indices, so they come out first and are dropped by
    count. (ELPA returns wrong pairs when the zero rows form a block at the
    global TAIL, 4e-3 at n 4608, and not at the head: Frontier 2x2, 2026-10-19.) ``constrain`` pins a distributed result to its layout (values
    replicated, vectors on the operand's faces).
    """
    pin = (lambda r: r) if constrain is None else constrain
    def solve(a, cut=None):     # ``cut`` is the checked routes' (checked_eigh); a bare solve ignores it
        n = a.shape[-1]
        dead = jnp.all(a == 0, axis=-1)
        # Every live eigenvalue lies in [-b, b] (Gershgorin).
        rows = (jnp.sum(jnp.abs(a), axis=-1) if mesh is None else
                ordered_sum(jnp.abs(a), axis=-1, mesh=mesh, spec=_face(a.ndim)))
        bound = jnp.max(rows, axis=-1)
        bound = jnp.where(bound > 0, bound, 1)
        sentinel = -2 * bound[..., None] * (2 - jnp.arange(n, dtype=bound.dtype) / n)   # row 0 lowest
        diagonal = jnp.where(dead, sentinel, 0).astype(a.dtype)
        values, vectors = pin(eigh(a + diagonal[..., :, None] * _eye_like(a)))
        values = jnp.where(values < -1.5 * bound[..., None], 0, values)
        return ascend(a, values, vectors, constrain=pin) if ascending else pin((values, vectors))
    return solve


def pad_side(a, side: int):
    """``a`` (.., n, n) zero-padded to ``side`` at the HEAD: the pads deflate first and are dropped by count."""
    n = int(a.shape[-1])
    return jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(side - n, 0), (side - n, 0)])


def drop_side(values, vectors, n: int):
    """The ``n``-row result of a :func:`pad_side` solve: the first ``side - n`` pairs and rows are the pads."""
    side = int(values.shape[-1])
    return values[..., side - n:], vectors[..., side - n:, side - n:]


#: The local (one-device) eigh of every native route: eigh_local (jnp.linalg.eigh with the
#: leg's implementation, :mod:`distrib_la.local_eigh`), deflated.
from distrib_la.local_eigh import eigh_local as _eigh_local
native_eigh = jax.jit(deflate_zero_rows(_eigh_local))
#: The same deflated eigh by Jacobi: the one re-solve of a round whose native eigh came back
#: non-finite (shared_pole_constructor / shared_pole_sectors, on their host-read diagnostics).
from distrib_la.local_eigh import eigh_local_jacobi as _eigh_local_jacobi
native_eigh_jacobi = jax.jit(deflate_zero_rows(_eigh_local_jacobi))


def probes(n, dtype, k=PROBES, salt=0):
    """Fixed-seed Gaussian [n, k] probe columns in ``dtype``, the same on every rank."""
    key = jax.random.fold_in(jax.random.key(_SEED), salt)
    real = jax.random.normal(key, (n, k), dtype=jnp.float64)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        imag = jax.random.normal(jax.random.fold_in(key, 1), (n, k), dtype=jnp.float64)
        return (real + 1j * imag).astype(dtype)
    return real.astype(dtype)


def _all_ranks(flag, mesh):
    """A boolean that is true only if it is true on every device (one pmin over the mesh).

    A library's 'replicated' output (out_specs=P(), check_vma=False) is not
    checked to agree; a rank-local NaN read locally would split the verdict
    and the ranks' retry branches (INVARIANTS 21).
    """
    if mesh is None:
        return flag
    from jax import shard_map
    axes = tuple(mesh.axis_names)
    reduce = shard_map(lambda f: jax.lax.pmin(f.astype(jnp.int32), axes), mesh=mesh,
                       in_specs=P(), out_specs=P(), check_vma=False)
    return reduce(flag) > 0


def used_probes(values, dtype, cut=None):
    """The probes, restricted to the columns a caller uses: rows ``w > cut * max(w)`` (all with ``cut`` None).

    Check what you use: a caller that keeps only ``w > c max(w)`` passes ``cut <= c``, so a
    defect inside the discarded near-zero cluster (cuSOLVERMp's non-orthonormal vectors
    there) is outside the check and needs no retry. ``(X_used, mask or None)``."""
    x = probes(values.shape[-1], dtype)
    if cut is None:
        return x, None
    used = values > cut * jnp.max(values, axis=-1, keepdims=True)
    return x * used[..., :, None].astype(dtype), used[..., :, None]


def eigh_errors(a, values, vectors, *, mesh=None, cut=None):
    """Largest probe residual and orthogonality error of a (stack of) eigh result(s), on the used columns."""
    x, used = used_probes(values, vectors.dtype, cut)
    vx = vectors @ x
    nd = a.ndim
    gram = jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ vx - x
    norm_x = jnp.maximum(_norm(x), jnp.finfo(values.dtype).tiny)      # replicated, per matrix
    residual = _norm(a @ vx - vectors @ (values[..., :, None] * x), mesh, _rows(nd)) / (
        jnp.maximum(_norm(a, mesh, _face(nd)), jnp.finfo(values.dtype).tiny) * norm_x)
    orthogonality = _norm(gram if used is None else jnp.where(used, gram, 0), mesh, _rows(nd)) / norm_x
    finite = _all_ranks(jnp.all(jnp.isfinite(values)), mesh)
    return (jnp.where(finite, jnp.max(residual), jnp.inf),
            jnp.where(finite, jnp.max(orthogonality), jnp.inf))


def call_site():
    """The first frame outside distrib_la and JAX: the role the refusal names."""
    for frame in reversed(traceback.extract_stack()[:-1]):
        path = frame.filename.replace(os.sep, "/")
        if "/distrib_la/" in path or "/jax/" in path or "/jaxlib/" in path or path.startswith("<"):
            continue
        parts = path.split("/")
        return f"{'/'.join(parts[-2:])}:{frame.lineno} {frame.name}"
    return "unknown caller"


def refusal_message(op, n, dtype, site, *errors):
    detail = ", ".join(f"{name} {float(value):.2e}" for name, value in
                       zip(("residual", "orthogonality"), errors))
    return (f"GATE distrib_la_result_check: {op} n={n} at {site}: "
            f"{detail or 'failed its check'} "
            f"(accept {accept(n, dtype):.1e}); the distributed solver returned a wrong result; "
            f"result set to NaN")


#: Host reads of check statuses: ``id(array) -> (array, bit)`` for a failure flag
#: (``status[0]``), so a flag is read at most once. The array is held so its id
#: cannot be reused while the entry lives.
_FAILED: dict = {}


def _memo(table, array):
    hit = table.get(id(array))
    return None if hit is None or hit[0] is not array else hit[1]


def _remember(table, array, bit):
    if len(table) > 64:
        table.clear()
    table[id(array)] = (array, bit)


def _flag(status):
    """Whether a replicated check status ``(failed, errors)`` failed: one read of its flag, every rank alike."""
    failed = _memo(_FAILED, status[0])
    if failed is None:
        failed = bool(np.any(np.asarray(jax.device_get(status[0].addressable_data(0)))))
        _remember(_FAILED, status[0], failed)
    return failed


def _notices(status, op, n, dtype, site):
    """Print the refusal of each failed matrix of a status on rank 0 (a stack names each one).

    Called only once :func:`_flag` says the status failed, so the error table is
    read only for a refusal. Only the printing depends on the rank.
    """
    if jax.process_index() != 0:
        return
    errors = status[1]
    errors = np.asarray(errors if isinstance(errors, np.ndarray) else jax.device_get(errors.addressable_data(0)))
    tol = accept(n, dtype)
    for row in errors.reshape(-1, errors.shape[-1]):
        if not np.all(row <= tol):
            print(refusal_message(op, n, dtype, site, *row), file=sys.stderr, flush=True)


def checked(op, solve, errors_of, operands, *, n=None, dtype=None, constrain=None):
    """Run ``solve`` once and check it with ``errors_of``: ``(result, (failed, errors))``.

    ``solve`` maps ``operands`` to a result; ``errors_of(result)`` returns
    replicated (mesh-reduced) scalars, accepted at ``accept(n, dtype)``.
    ``failed`` is a replicated bool, ``errors`` [e] the check's errors; the
    caller's host names a failure (:func:`raise_if_failed`). A failure poisons
    the result's first part with NaN on every rank (an eigh's eigenvalues, a
    solve's X: n or n x nrhs elements, selected in place; poisoning the
    vectors too would copy them: +16 n^2/P at P16, 16 matrices, route (c)),
    and the failure is refused by name, eager or at its checked_program's
    finish. ``constrain`` pins the result to its layout.
    """
    pin = (lambda r: r) if constrain is None else constrain
    n = int(operands[0].shape[-1]) if n is None else int(n)
    dtype = operands[0].dtype if dtype is None else dtype
    result = pin(solve(*operands))
    errors = jnp.stack(errors_of(result)).astype(jnp.float64)
    ok = jnp.all(errors <= accept(n, dtype))
    nan = lambda a: jnp.where(ok, a, jnp.array(jnp.nan, a.dtype))
    out = (nan(result[0]), *result[1:]) if isinstance(result, tuple) else nan(result)
    return pin(out), (~ok, errors)


def eigh_layout(mesh, ndim):
    """Pin an eigh result to the service contract: values replicated, vectors on the faces."""
    values = NamedSharding(mesh, P())
    vectors = NamedSharding(mesh, P(*([None] * (ndim - 2)), "x", "y"))
    return lambda r: (jax.lax.with_sharding_constraint(r[0], values),
                      jax.lax.with_sharding_constraint(r[1], vectors))


def checked_eigh(solve, a, *, mesh=None, cut=None):
    """A distributed eigh, checked on the columns ``w > cut * max(w)`` (all by default): ``(result, failed)``."""
    constrain = None if mesh is None else eigh_layout(mesh, a.ndim)
    return checked("eigh", solve, lambda r: eigh_errors(a, *r, mesh=mesh, cut=cut), (a,),
                   constrain=constrain)


#: Inside :func:`checked_program`'s trace: (entries, trace). A checked solve
#: traced at that level hands its status and (op, n, dtype, site) here;
#: elsewhere (no program, or a nested scan, cond or shard_map trace) it
#: names its own failure (:func:`raise_if_failed`).
_TRACED = contextvars.ContextVar("distrib_la_traced", default=None)


def _program_context():
    """The enclosing :func:`checked_program` when tracing at its level, else None."""
    from jax._src.core import trace_ctx
    ctx = _TRACED.get()
    return ctx if ctx is not None and ctx[1] is trace_ctx.trace else None


def _signature(args):
    return jax.tree.structure(args), tuple((tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(args))


def _collected(fn):
    """``fn`` traced with its checked solves collected: ``(out, ())`` or
    ``(out, (reduced failure flag, every solve's error table))``; each trace's
    (op, n, dtype, site) per solve is kept in ``run.infos`` by signature."""
    infos = {}

    def run(*args):
        from jax._src.core import trace_ctx
        token = _TRACED.set(([], trace_ctx.trace))
        try:
            out, entries = fn(*args), _TRACED.get()[0]
        finally:
            _TRACED.reset(token)
        infos[_signature(args)] = tuple(info for _, info in entries)
        if not entries:
            return out, ()
        return out, (jnp.any(jnp.stack([jnp.any(s[0]) for s, _ in entries])),
                     tuple(s[1] for s, _ in entries))
    run.infos = infos
    return run


def checked_shapes(fn, *args):
    """``jax.eval_shape`` of ``fn`` as :func:`checked_program` traces it."""
    return jax.eval_shape(_collected(fn), *args)[0]


def checked_program(fn, mesh, out_shardings, in_shardings=None, compiler_options=None):
    """A caller's jitted program over checked solves: one program, its failure read once.

    The program returns each checked solve's result and the mesh-reduced
    failure flag with every solve's error table; the host reads the flag
    (one transfer) and, only when it is set, prints each failed solve's
    refusal and raises by name. A program with no checked solve returns no
    flag and is never synced on. A caller that runs
    ``call.lower(*args).compile()`` itself hands that executable's
    ``(out, status)`` to ``call.finish(args, result)``.
    """
    rep = NamedSharding(mesh, P())
    traced = _collected(fn)
    shardings = {} if in_shardings is None else dict(in_shardings=in_shardings)
    if compiler_options is not None:
        # Top-level jit only (the XLA option set, e.g. the latency-hiding scheduler).
        shardings["compiler_options"] = dict(compiler_options)
    program = jax.jit(traced, out_shardings=(out_shardings, rep), **shardings)

    def finish(args, result):
        out, status = result
        if status and _flag(status):
            for (op, n, dtype, site), table in zip(traced.infos[_signature(args)], status[1]):
                _notices((status[0], table), op, n, dtype, site)
            raise ValueError(f"GATE distrib_la_result_check: a checked solve in the program at "
                             f"{call_site()} failed its check (named above); result set to NaN")
        return out

    def call(*args):
        return finish(args, program(*args))
    call.lower, call.finish = program.lower, finish
    return call


#: Sites whose traced status took the host-callback fallback (warned once each).
_UNCACHEABLE: set = set()


def raise_if_failed(status, op, n, dtype, site):
    """An eager call's refusal: raise by name when any matrix failed its check.

    ``status`` is :func:`checked`'s ``(failed, errors)`` (or a stack of them,
    from a scan), replicated, so every rank reads the same values and raises
    together; only the printing depends on the rank. A traced status goes to
    the enclosing :func:`checked_program`, if any; a traced status outside
    one prints its notices through a host callback (the only program that
    keeps one, and so is not stored by JAX's persistent cache: it warns once
    per site, naming it).
    """
    if isinstance(status[0], jax.core.Tracer):
        if _program_context() is not None:
            _program_context()[0].append((status, (op, int(n), dtype, site)))
        else:
            if site not in _UNCACHEABLE and jax.process_index() == 0:
                print(f"distrib_la: UNCACHEABLE: a checked {op} at {site} is traced outside "
                      f"distrib_la.checked_program, so the program holding it carries a host callback "
                      f"and JAX's persistent cache never stores it (it compiles again in every "
                      f"process); build that program with distrib_la.checked_program",
                      file=sys.stderr, flush=True)
            _UNCACHEABLE.add(site)
            # No array math in the callback: np.any on a jax Array compiles a reduction, and
            # only on the ranks that reach it (a rank-divergent compile). _notices prints only
            # failed rows, on rank 0, from host copies.
            jax.debug.callback(lambda failed, errors: _notices((failed, errors), op, int(n), dtype, site),
                               *status)
        return
    if _flag(status):                         # every rank: the failure bit, one read
        _notices(status, op, n, dtype, site)
        raise ValueError(refusal_message(op, n, dtype, site))


def matrix_sketch(a, mesh=None):
    """Before a factorization or solve that may consume ``a``: (A^H W, ||A||).  ``mesh``: ``a``
    is face-tiled there and its norm is added in rank order."""
    w = probes(a.shape[-1], a.dtype, salt=1)
    return jnp.conj(jnp.swapaxes(a, -1, -2)) @ w, _norm(a, mesh, _face(a.ndim))


def rhs_sketch(b, mesh=None):
    """Before a solve that may consume ``b``: (W^H B, ||B||) (``b`` face-tiled on ``mesh``)."""
    w = probes(b.shape[-2], b.dtype, salt=1)
    return jnp.conj(w.T) @ b, _norm(b, mesh, _face(b.ndim))


def solve_errors(sketch, x, mesh=None):
    """||W^H (A X - B)|| / (sqrt(k) (||A|| ||X|| + ||B||)) from the pre-solve sketches, largest over the stack.

    ``sketch`` is ``matrix_sketch(A) + rhs_sketch(B)``; W^H A X = (A^H W)^H X.
    """
    ahw, norm_a, whb, norm_b = sketch
    projected = jnp.conj(jnp.swapaxes(ahw, -1, -2)) @ x - whb
    nd = x.ndim
    scale = np.sqrt(PROBES) * (norm_a * _norm(x, mesh, _face(nd)) + norm_b)
    # W^H (A X - B): k probe rows, the right-hand-side columns on 'y' as in X
    error = _norm(projected, mesh, P(*((None,) * (nd - 1)), "y")) / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
    return (jnp.where(jnp.all(jnp.isfinite(x)), jnp.max(error), jnp.inf),)
