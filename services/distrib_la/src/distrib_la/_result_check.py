"""Checked dense results: no distributed library result leaves this service unchecked.

The distributed libraries fail silently. cuSOLVERMp 0.9.1's syevd returned
wrong eigenvectors with status 0 and info 0 on valid Hermitian matrices: an
exact-zero block (Fe 4^3 bispinor CT metric, n 2688), rank-deficient PSD
responses (n 432 to 18304) and a sentinel-padded H'_vv (CrI3, n 9152), all
reproduced outside LORRAX. So every eigh takes its exact-zero rows out of the
solver (:func:`deflate_zero_rows`), and every distributed eigh and LU solve is
checked on its own output against fixed-seed random probes before it is
returned (:func:`checked`), at O(n^2 k) beside the O(n^3) solve. A failed
check retries where the operands survive; a result that still fails is
NaN-poisoned on every rank, named on rank 0's stderr (GATE
distrib_la_result_check) and raised. The device returns each check's
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
#: Newton-Schulz steps the shifted retry may take (it stops at n * eps).
ORTHONORMALIZE_STEPS = 12


def accept(n, dtype=np.complex128):
    """The accepted backward error of a checked result of side ``n``: roundoff_tol(n, dtype)."""
    from distrib_la.tolerance import roundoff_tol
    return roundoff_tol(n, dtype)


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


def _live_norm(a, mesh=None):
    """||A||_F without the rows that carry only a diagonal entry (sentinels, zero rows)."""
    diagonal = jnp.diagonal(a, axis1=-2, axis2=-1)
    decoupled = jnp.all((a == 0) | (_eye_like(a) != 0), axis=-1)
    if mesh is None:
        live = jnp.sum(jnp.abs(a) ** 2, axis=(-2, -1)) - jnp.sum(
            jnp.where(decoupled, jnp.abs(diagonal) ** 2, 0), axis=-1)
    else:   # rank order on the face; the diagonal is one-owner (exact), summed in rank order too
        live = ordered_sum(jnp.abs(a) ** 2, axis=(-2, -1), mesh=mesh, spec=_face(a.ndim)) - ordered_sum(
            jnp.where(decoupled, jnp.abs(diagonal) ** 2, 0), axis=-1, mesh=mesh, spec=P(*((None,) * (a.ndim - 1))))
    return jnp.sqrt(jnp.maximum(live, 0))


def deflate_zero_rows(eigh, *, constrain=None, shift=False, mesh=None, defer=False, in_place=False):
    """Wrap a Hermitian eigensolver so exact zero rows never reach it.

    Capacity padding and unselected columns reach eigensolvers as exact zero
    rows/columns, and a large zero block breaks them: the native solver
    returned nonfinite values (Na P16, 1460 of 2584 rows), cuSOLVERMp wrong
    eigenpairs with info 0. Those rows are decoupled, so they are replaced by
    distinct diagonal sentinels in [-4b, -2b], b the Gershgorin bound of the
    rest (relative to A, so a small-norm matrix keeps the gap), solved, and
    reported back as exact zero eigenvalues with their unit eigenvectors in
    ascending order: the spectrum and vectors of the zero-padded matrix,
    without a zero block inside the solver. The eigenvector columns are
    reordered only when a live eigenvalue is negative (a PSD matrix comes out
    of the solver in order already), because reordering a distributed matrix's
    columns is a cross-device permutation. A matrix with no zero row passes
    through unchanged. ``constrain`` pins a distributed result to its layout
    (values replicated, vectors on the operand's faces), so the cond's output
    is never left to propagation, which replicated it (n^2 per rank).
    ``shift`` also adds s = ||live A||_F to the live diagonal in the same
    pass (one copy of A, as without it) and takes s off the eigenvalues: the
    solver sees the live spectrum in [s - b, s + b], so a (near-)zero
    cluster sits at s, away from the origin (:func:`shifted`, without its
    re-orthonormalization). The eigenvalues come back with an absolute error
    ~ eps (||A|| + s) <= eps sqrt(n) ||A||_2.

    ``defer`` leaves the reorder to the caller: the solve returns
    ``(values, vectors, live_negative)`` in the solver's order, with the
    replicated flag that says whether :func:`sort_eigenpairs` must still run
    (a distributed checked chain reorders once, after its check, and an eager
    call in a separate program only when the flag is set, so no program
    reserves the reorder's buffers beside the solve: EIGHMEM).
    ``in_place`` (the donated eigh, whose ``a`` is consumed; True or a scalar
    token, see below) adds the diagonal
    with a scatter on each rank's own tile (:func:`_add_diagonal`), an in-place
    update XLA performs in ``a``'s buffer, where ``a + diag I`` is a new
    buffer. Its off-diagonal entries are ``a``'s exactly; ``a + diag I`` turns
    a ``-0.0`` in a live row into ``+0.0``, so the two operands can differ in
    the sign of a zero, nowhere else.
    """
    pin = (lambda r: r) if constrain is None else constrain
    def solve(a):
        n = a.shape[-1]
        dead = jnp.all(a == 0, axis=-1)
        # Every live eigenvalue lies in [-b, b] (Gershgorin).
        rows = (jnp.sum(jnp.abs(a), axis=-1) if mesh is None else
                ordered_sum(jnp.abs(a), axis=-1, mesh=mesh, spec=_face(a.ndim)))
        bound = jnp.max(rows, axis=-1)
        bound = jnp.where(bound > 0, bound, 1)
        sentinel = -2 * bound[..., None] * (1 + jnp.arange(n, dtype=bound.dtype) / n)
        s = _live_norm(a, mesh)[..., None] if shift else 0
        diagonal = jnp.where(dead, sentinel, s).astype(a.dtype)
        if in_place is not False and in_place is not None:
            # A token (the donated chain's sketch, times 0) the diagonal depends on, so
            # every read of ``a`` it stands for is ordered before the in-place update;
            # +0 leaves every finite diagonal value bitwise as it was.
            token = jnp.asarray(0 if in_place is True else in_place)
            operand = _add_diagonal(a, diagonal + token.astype(diagonal.dtype), mesh)
        else:
            operand = a + diagonal[..., :, None] * _eye_like(a)
        values, vectors = pin(eigh(operand))
        restored = jnp.where(values < -1.5 * bound[..., None], 0, values - s)
        # Sentinels come out first; only a negative live eigenvalue must move ahead of the zeros.
        live_negative = jnp.any((restored < 0) & jnp.any(dead, axis=-1, keepdims=True))
        if defer:
            return (*pin((restored, vectors)), live_negative)
        return reorder_if(live_negative, restored, vectors, mesh=mesh, constrain=constrain)
    return solve


def _add_diagonal(a, diagonal, mesh):
    """``a`` with ``diagonal`` added to its diagonal, by a scatter-add on each rank's tile.

    A scatter updates its operand in place where the operand is dead after it (a
    donated ``a``), so the deflated operand takes ``a``'s buffer. Each rank adds the
    entries of the global diagonal its ``(n/p, n/q)`` tile holds (none off the
    diagonal blocks): ``min(n/p, n/q)`` slots, the ones past the overlap dropped.
    """
    n = int(a.shape[-1])
    if mesh is None:
        k = jnp.arange(n)
        return a.at[..., k, k].add(diagonal, mode="drop")
    from jax import shard_map
    p, q = int(mesh.shape["x"]), int(mesh.shape["y"])
    rl, cl = n // p, n // q
    nd = a.ndim
    lead = (None,) * (nd - 2)

    def local(d, tile):
        x, y = jax.lax.axis_index("x"), jax.lax.axis_index("y")
        lo = jnp.maximum(x * rl, y * cl)
        hi = jnp.minimum((x + 1) * rl, (y + 1) * cl)
        g = lo + jnp.arange(min(rl, cl))
        ok = g < hi
        row = jnp.where(ok, g - x * rl, rl)          # out of range: dropped
        col = jnp.where(ok, g - y * cl, cl)
        vals = jnp.take(d, jnp.minimum(g, n - 1), axis=-1)
        return tile.at[..., row, col].add(vals, mode="drop")
    return shard_map(local, mesh=mesh, in_specs=(P(*lead, None), P(*lead, "x", "y")),
                     out_specs=P(*lead, "x", "y"), check_vma=False)(diagonal, a)


def _permute_columns(vectors, order, mesh):
    """``take_along_axis(vectors, order[..., None, :], -1)`` of a face-sharded matrix, gather-free.

    The column permutation of a ``P(.., 'x', 'y')`` matrix: one all_to_all over
    ``'y'`` gives each rank ``n/(p q)`` whole rows, the replicated ``order``
    permutes them locally, and the inverse all_to_all restores the faces. Two
    ``n^2/P`` exchanges and ``n^2/P`` temporaries per rank, where XLA's own
    lowering of the global take all-gathers ``n^2/p`` (q tiles; 4.0 n^2/P at
    4x4, EIGHMEM step 1). Pure data movement: the result is bitwise the take's.
    A tile whose rows do not split over ``'y'`` keeps the global take.
    """
    nd = vectors.ndim
    p, q = int(mesh.shape["x"]), int(mesh.shape["y"])
    n = int(vectors.shape[-1])
    if n % p or n % q or (n // p) % q:
        return jnp.take_along_axis(vectors, order[..., None, :], axis=-1)
    from jax import shard_map
    lead = (None,) * (nd - 2)

    def local(o, v):
        rows = jax.lax.all_to_all(v, "y", nd - 2, nd - 1, tiled=True)     # (.., n/(pq), n)
        rows = jnp.take_along_axis(rows, o[..., None, :], axis=-1)
        return jax.lax.all_to_all(rows, "y", nd - 1, nd - 2, tiled=True)  # (.., n/p, n/q)
    return shard_map(local, mesh=mesh, in_specs=(P(*lead, None), P(*lead, "x", "y")),
                     out_specs=P(*lead, "x", "y"), check_vma=False)(order, vectors)


def sort_eigenpairs(values, vectors, *, mesh=None, constrain=None):
    """Eigenpairs in ascending (stable) order: the reorder :func:`deflate_zero_rows` defers."""
    pin = (lambda r: r) if constrain is None else constrain
    order = jnp.argsort(values, axis=-1, stable=True)
    taken = (jnp.take_along_axis(vectors, order[..., None, :], axis=-1) if mesh is None
             else _permute_columns(vectors, order, mesh))
    return pin((jnp.take_along_axis(values, order, axis=-1), taken))


def reorder_if(flag, values, vectors, *, mesh=None, constrain=None):
    """:func:`sort_eigenpairs` when the replicated ``flag`` is set (``lax.cond``), else unchanged."""
    pin = (lambda r: r) if constrain is None else constrain
    return pin(jax.lax.cond(flag, lambda r: sort_eigenpairs(*r, mesh=mesh, constrain=constrain),
                            pin, (values, vectors)))


#: The local (one-device) eigh of every native route: eigh_local (jnp.linalg.eigh with the
#: leg's implementation, :mod:`distrib_la.local_eigh`), deflated.
from distrib_la.local_eigh import eigh_local as _eigh_local
native_eigh = jax.jit(deflate_zero_rows(_eigh_local))
#: The same deflated eigh by Jacobi: the one re-solve of a round whose native eigh came back
#: non-finite (shared_pole_constructor / shared_pole_sectors, on their host-read diagnostics).
from distrib_la.local_eigh import eigh_local_jacobi as _eigh_local_jacobi
native_eigh_jacobi = jax.jit(deflate_zero_rows(_eigh_local_jacobi))


def orthonormalize(vectors, *, matmul, mesh=None):
    """Newton-Schulz polar iteration V <- V (3I - V^H V) / 2, GEMMs only.

    Returns the vectors and the number of steps taken. A solve whose vectors
    are eigenvectors but not orthonormal inside a degenerate cluster
    (cuSOLVERMp at n 18304: residual 2e-15, orthogonality 2e-2) is repaired
    without leaving the cluster: V^H V couples only near-parallel columns,
    which share an eigenvalue. The iteration converges, quadratically, while
    the singular values of V lie in (0, sqrt 3); it stops at n * eps or after
    ORTHONORMALIZE_STEPS, and the result check decides either way (a failure
    falls through to the next attempt). n 18304, 4x4: 6 steps, 2e-2 -> 8e-15.
    """
    n = vectors.shape[-1]
    eye = _eye_like(vectors)
    target = n * float(np.finfo(vectors.real.dtype).eps)

    def gram(v):
        g = matmul(v, v, transa="C")
        return g, jnp.max(_norm(g - eye, mesh, _face(g.ndim))) / np.sqrt(n)

    def unfinished(state):
        steps, _, _, defect = state
        return (steps < ORTHONORMALIZE_STEPS) & (defect > target)

    def step(state):
        steps, v, g, _ = state
        v = matmul(v, (3 * eye - g) / 2)
        g, defect = gram(v)
        return steps + 1, v, g, defect

    g, defect = gram(vectors)
    steps, vectors, _, _ = jax.lax.while_loop(unfinished, step, (jnp.int32(0), vectors, g, defect))
    return vectors, steps


def shifted(eigh, *, matmul, mesh=None):
    """Solve A + s I, s = ||live part of A||_F, re-orthonormalize, return A's eigenpairs.

    The shift moves a large (near-)zero cluster away from the origin.
    cuSOLVERMp's silent failures on rank-deficient PSD responses (n 432, no
    zero row, every block size), on the CT metric's zero block (n 2688, block
    224) and on a sentinel-padded H'_vv (n 9152, 4x4) all pass shifted (probe
    residual <= 1.3e-15). Where the shifted vectors are still not orthonormal
    inside a cluster (n 18304, a 3/4-wide near-zero cluster: orthogonality
    2e-2), ``orthonormalize`` repairs them. s excludes rows that carry only a
    diagonal entry (deflation sentinels). The eigenvalues come back with an
    absolute error ~ eps (||A|| + s) <= eps sqrt(n) ||A||_2, below the
    n eps lambda_max support line of the response callers.
    """
    def solve(a):
        s = _live_norm(a, mesh)
        values, vectors = eigh(a + s[..., None, None] * _eye_like(a))
        vectors, _ = orthonormalize(vectors, matmul=matmul, mesh=mesh)
        return values - s[..., None], vectors
    return solve


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


def eigh_errors(a, values, vectors, *, mesh=None):
    """Largest probe residual and orthogonality error of a (stack of) eigh result(s)."""
    x = probes(a.shape[-1], vectors.dtype)
    vx = vectors @ x
    nd = a.ndim
    residual = _norm(a @ vx - vectors @ (values[..., :, None] * x), mesh, _rows(nd)) / (
        jnp.maximum(_norm(a, mesh, _face(nd)), jnp.finfo(values.dtype).tiny) * _norm(x))
    orthogonality = _norm(jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ vx - x, mesh, _rows(nd)) / _norm(x)
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


def refusal_message(op, n, dtype, site, attempts, *errors):
    detail = ", ".join(f"{name} {float(value):.2e}" for name, value in
                       zip(("residual", "orthogonality"), errors))
    return (f"GATE distrib_la_result_check: {op} n={n} at {site}: "
            f"{detail or 'no attempt passed'} after {attempts} attempt(s) "
            f"(accept {accept(n, dtype):.1e}); the distributed solver returned a wrong result "
            f"and no retry repaired it; result set to NaN")


#: Host reads of check statuses: ``id(array) -> (array, bit)`` for a failure flag
#: (``status[0]``) and for a notice bit (keyed by its error table ``status[1]``, which
#: several tables of one program share no flag with).  The array is held so its id
#: cannot be reused while the entry lives.
_FAILED: dict = {}
_NOTICE: dict = {}


def _memo(table, array):
    hit = table.get(id(array))
    return None if hit is None or hit[0] is not array else hit[1]


def _remember(table, array, bit):
    if len(table) > 64:
        table.clear()
    table[id(array)] = (array, bit)


def _word_program(tols, extras=0):
    """``[failed, notice_0, ..]`` of a status and its tables on the device: one int32 word, one host read.

    ``notice_i`` is exactly :func:`_notices`' print condition for table ``i``: some
    matrix's first attempt neither all-NaN nor within ``tols[i]`` (its loop stops at
    the first such row).
    """
    @jax.jit
    def word(failed, tables, flags=()):
        bits = [jnp.any(failed)]
        for tol, errors in zip(tols, tables):
            first = errors.reshape(-1, *errors.shape[-2:])[:, 0]
            shown = ~jnp.all(jnp.isnan(first), axis=-1) & ~jnp.all(first <= tol, axis=-1)
            bits.append(jnp.any(shown))
        bits.extend(jnp.any(f) for f in flags)
        return jnp.stack(bits).astype(jnp.int32)
    return word


_WORDS: dict = {}


def _read_status(failed, tables, infos, flags=()):
    """Read a replicated status's failure bit and every table's notice bit in ONE transfer.

    Every rank calls it for the same status (it gates the retry and the refusal on all of
    them), so every rank compiles and runs the same program, on its own replica
    (INVARIANTS 21: no rank-conditional compile); only the printing is rank 0's.
    ``infos`` are the tables' ``(n, dtype)``. ``flags`` are further replicated
    booleans read in the same word (a deferred reorder's, :func:`read_flags`);
    their bits are returned after the failure bit.
    """
    tols = tuple(float(accept(int(n), dtype)) for n, dtype in infos)
    key = (tols, len(flags))
    program = _WORDS.get(key) or _WORDS.setdefault(key, _word_program(tols, len(flags)))
    word = np.asarray(program(failed.addressable_data(0),
                              tuple(t.addressable_data(0) for t in tables),
                              tuple(f.addressable_data(0) for f in flags)))
    for table, bit in zip(tables, word[1:1 + len(tables)]):
        _remember(_NOTICE, table, bool(bit))
    _remember(_FAILED, failed, bool(word[0]))
    if flags:
        return (bool(word[0]), *(bool(b) for b in word[1 + len(tables):]))
    return bool(word[0])


def read_flags(status, flag, n, dtype):
    """``(failed, flag)`` of an eager checked eigh: its status and a replicated flag, ONE transfer.

    The failure and notice bits are remembered as :func:`_flag` reads them, so
    the :func:`raise_if_failed` that follows reads nothing more. Every rank
    reads the same word (INVARIANTS 21).
    """
    if isinstance(status[1], np.ndarray):
        failed = _flag(status, n, dtype)
        return failed, bool(np.asarray(jax.device_get(flag.addressable_data(0))))
    return _read_status(status[0], (status[1],), ((n, dtype),), (flag,))


def _status(status, n=None, dtype=None):
    """``(failed, notice)`` of a replicated status, each read at most once per array.

    With ``(n, dtype)`` (and a device table) both bits come in one transfer, on
    every rank alike; without them only the failure bit is read (notice None).
    """
    failed, notice = _memo(_FAILED, status[0]), None
    want = n is not None and not isinstance(status[1], np.ndarray)
    if want:
        notice = _memo(_NOTICE, status[1])
    if (failed is not None) and (notice is not None or not want):
        return failed, notice
    if want:
        _read_status(status[0], (status[1],), ((n, dtype),))
        return _memo(_FAILED, status[0]), _memo(_NOTICE, status[1])
    failed = bool(np.any(np.asarray(jax.device_get(status[0].addressable_data(0)))))
    _remember(_FAILED, status[0], failed)
    return failed, notice


def _flag(status, n=None, dtype=None):
    """Whether a replicated check status ``(failed, errors)`` failed (every rank reads the same bytes).

    With ``(n, dtype)`` rank 0 reads the notice bit in the same transfer, so the
    :func:`_notices` that follows reads the error table only when it will print.
    """
    return _status(status, n, dtype)[0]


def _notices(status, op, n, dtype, site, *, final):
    """Print a check status's notices on rank 0: each failed attempt, then the refusal.

    ``errors`` [..., k, e] holds every attempt's errors (NaN where an attempt
    did not run); a stack (a scan) reports each matrix that failed. A failed
    last attempt is the refusal when ``final``, else the first program's
    retry notice. Only the printing depends on the rank.
    """
    if jax.process_index() != 0:
        return
    if not isinstance(status[1], np.ndarray) and _memo(_NOTICE, status[1]) is False:
        return              # read with the failure bit (_flag/_read_status): no attempt to name
    errors = status[1]
    errors = np.asarray(errors if isinstance(errors, np.ndarray) else jax.device_get(errors.addressable_data(0)))
    tol = accept(n, dtype)
    for table in errors.reshape(-1, *errors.shape[-2:]):
        for i, row in enumerate(table):
            if np.isnan(row).all() or np.all(row <= tol):
                break
            if i + 1 == len(table) and final:
                print(refusal_message(op, n, dtype, site, len(table), *row), file=sys.stderr, flush=True)
                break
            what = ("solving again in a separate program" if i + 1 == len(table) else
                    f"attempt {i + 1} of {len(table)} failed; solving again")
            detail = ", ".join(f"{name} {float(value):.2e}" for name, value in
                               zip(("residual", "orthogonality"), row))
            print(f"distrib_la: {op} n={n} at {site} failed its result check ({detail}, "
                  f"accept {tol:.1e}); {what}", file=sys.stderr, flush=True)


def checked(op, attempts, errors_of, operands, *, n=None, dtype=None, constrain=None, final=True):
    """Run ``attempts`` until one passes ``errors_of``: ``(result, (failed, errors))``.

    Each attempt maps ``operands`` to a result; ``errors_of(result)`` returns
    replicated (mesh-reduced) scalars, accepted at ``accept(n, dtype)``. Later
    attempts run only when the earlier ones failed (``lax.cond``). ``failed``
    is a replicated bool, ``errors`` [len(attempts), e] every attempt's
    errors (NaN where it did not run); the caller's host prints the notices
    (:func:`raise_if_failed`). A final failure poisons the result with NaN on
    every rank. ``constrain`` pins each result (and each cond's output) to its
    layout. ``final=False`` (a first program, whose retries run in a separate
    program on its flag) returns zeros, not NaN, so the rest of that program
    stays finite.
    """
    pin = (lambda r: r) if constrain is None else constrain
    n = int(operands[0].shape[-1]) if n is None else int(n)
    dtype = operands[0].dtype if dtype is None else dtype
    k = len(attempts)

    def verdict(errors):
        return jnp.all(errors <= accept(n, dtype))

    def run(index, table):
        result = pin(attempts[index](*operands))
        errors = jnp.stack(errors_of(result)).astype(jnp.float64)
        if table is None:
            table = jnp.full((k, errors.shape[0]), jnp.nan, jnp.float64)
        table = table.at[index].set(errors)

        def accepted(r):
            return pin(r), jnp.bool_(False), table
        if index + 1 == k:
            def refuse(r):
                # A flag riding with the result (a deferred reorder) is cleared, not poisoned.
                return (pin(jax.tree.map(lambda a: jnp.zeros_like(a) if a.dtype == jnp.bool_ else
                                         jnp.full_like(a, jnp.nan if final else 0), r)),
                        jnp.bool_(True), table)
            out, failed, table = jax.lax.cond(verdict(errors), accepted, refuse, result)
            return pin(out), failed, table
        out, failed, table = jax.lax.cond(verdict(errors), accepted,
                                          lambda _: run(index + 1, table), result)
        return pin(out), failed, table
    out, failed, table = run(0, None)
    return out, (failed, table)


def eigh_layout(mesh, ndim):
    """Pin an eigh result to the service contract: values replicated, vectors on the faces."""
    values = NamedSharding(mesh, P())
    vectors = NamedSharding(mesh, P(*([None] * (ndim - 2)), "x", "y"))
    return lambda r: (jax.lax.with_sharding_constraint(r[0], values),
                      jax.lax.with_sharding_constraint(r[1], vectors))


def checked_eigh(attempts, a, *, mesh=None, final=True, errors_of=None):
    """A distributed eigh, checked: ``attempts`` solve ``a`` in order of preference; ``(result, failed)``.

    A result is ``(values, vectors)`` or, from attempts that defer their reorder
    (:func:`deflate_zero_rows` ``defer``), ``(values, vectors, live_negative)``:
    the check reads the first two, which are an eigendecomposition in any column
    order. ``errors_of`` replaces :func:`eigh_errors` (the donated route's sketch
    check, :func:`sketch_eigh_errors`).
    """
    layout = None if mesh is None else eigh_layout(mesh, a.ndim)
    constrain = None if layout is None else (lambda r: (*layout(r[:2]), *r[2:]))
    errors_of = errors_of or (lambda r: eigh_errors(a, r[0], r[1], mesh=mesh))
    return checked("eigh", attempts, errors_of, (a,), n=a.shape[-1], dtype=a.dtype,
                   constrain=constrain, final=final)


def eigh_sketch(a, *, mesh=None):
    """What :func:`sketch_eigh_errors` keeps of ``a``: ``(a @ X, X, ||a||_F)``, ``X`` the probes.

    Taken before a donated solve destroys ``a`` (n x PROBES per rank row, not n^2).
    """
    x = probes(a.shape[-1], a.dtype)
    return a @ x, x, _norm(a, mesh, _face(a.ndim))


def sketch_eigh_errors(sketch, values, vectors, *, mesh=None):
    """:func:`eigh_errors` from a sketch of the (consumed) operand.

    The residual is ``||V^H (A X) - W V^H X|| / (||A|| ||X||)``, the left
    eigen-relation ``V^H A = W V^H`` on the probes: for Hermitian ``A`` it is
    the adjoint of ``A V - V W``, the relation :func:`eigh_errors` probes from
    the right, so the same defect is measured with ``A`` read once, before the
    solve. The orthogonality is :func:`eigh_errors`' (V alone).
    """
    ax, x, norm = sketch
    nd = vectors.ndim
    vh = jnp.conj(jnp.swapaxes(vectors, -1, -2))
    vx = vectors @ x
    residual = _norm(vh @ ax - values[..., :, None] * (vh @ x), mesh, _rows(nd)) / (
        jnp.maximum(norm, jnp.finfo(values.dtype).tiny) * _norm(x))
    orthogonality = _norm(vh @ vx - x, mesh, _rows(nd)) / _norm(x)
    finite = _all_ranks(jnp.all(jnp.isfinite(values)), mesh)
    return (jnp.where(finite, jnp.max(residual), jnp.inf),
            jnp.where(finite, jnp.max(orthogonality), jnp.inf))


#: Inside :func:`checked_program`'s trace: (phase, entries, trace). A checked
#: solve traced at that level holds only that phase of its chain and hands
#: its status and (op, n, dtype, site) here; elsewhere (no program, or a
#: nested scan, cond or shard_map trace) it holds the whole chain ("all").
_TRACED = contextvars.ContextVar("distrib_la_traced", default=None)


def _program_context():
    """The enclosing :func:`checked_program` when tracing at its level, else None."""
    from jax._src.core import trace_ctx
    ctx = _TRACED.get()
    return ctx if ctx is not None and ctx[2] is trace_ctx.trace else None


def traced_phase():
    """The phase a traced checked solve compiles: its program's, else "all"."""
    ctx = _program_context()
    return "all" if ctx is None else ctx[0]


def in_checked_retry():
    """Tracing a :func:`checked_program`'s whole-chain program, whose stacks run on the whole mesh."""
    return traced_phase() == "all" and _program_context() is not None


def _signature(args):
    return jax.tree.structure(args), tuple((tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(args))


def _in_phase(fn, phase):
    """``fn`` traced with its checked solves on ``phase``: ``(out, ())`` or
    ``(out, (reduced failure flag, every solve's error table))``; each trace's
    (op, n, dtype, site) per solve is kept in ``run.infos`` by signature."""
    infos = {}

    def run(*args):
        from jax._src.core import trace_ctx
        token = _TRACED.set((phase, [], trace_ctx.trace))
        try:
            out, entries = fn(*args), _TRACED.get()[1]
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
    """``jax.eval_shape`` of ``fn`` as :func:`checked_program`'s first program traces it."""
    return jax.eval_shape(_in_phase(fn, "first"), *args)[0]


def checked_program(fn, mesh, out_shardings, in_shardings=None, compiler_options=None):
    """A caller's jitted program over checked solves, run as an eager eigh is.

    The first program holds each checked solve's first attempt and check and
    returns their mesh-reduced failure flag and error tables; only when the
    flag is set does the whole-chain program run: every solve's whole chain,
    every stack on the whole mesh, refused by name if a solve fails every
    attempt. The notices print on the host from the returned tables. A
    program with no checked solve returns no flag and is never synced on.
    ``call.lower`` lowers the whole-chain program, so a caller that admits it
    by its compiled size has sized the retry, and a first program run beside
    that size fits. A caller that runs ``call.lower(*args).compile()`` itself
    hands that executable's ``(out, status)`` to ``call.finish(args, result)``,
    which prints the notices, refuses by name and returns ``out``.
    """
    rep = NamedSharding(mesh, P())
    phases = {phase: _in_phase(fn, phase) for phase in ("first", "all")}
    shardings = {} if in_shardings is None else dict(in_shardings=in_shardings)
    if compiler_options is not None:
        # Top-level jit only (the XLA option set, e.g. the latency-hiding scheduler).
        shardings["compiler_options"] = dict(compiler_options)
    first, again = (jax.jit(phases[phase], out_shardings=(out_shardings, rep), **shardings)
                    for phase in ("first", "all"))

    def report(phase, args, status, final):
        for (op, n, dtype, site), table in zip(phases[phase].infos[_signature(args)], status[1]):
            _notices((status[0], table), op, n, dtype, site, final=final)

    def finish(args, result):
        out, status = result
        if status:
            infos = phases["all"].infos[_signature(args)]
            _read_status(status[0], tuple(status[1]), [(n, dtype) for _, n, dtype, _ in infos])
            report("all", args, status, final=True)
            if _flag(status):
                raise ValueError(f"GATE distrib_la_result_check: a checked solve in the program at "
                                 f"{call_site()} failed every attempt (named above); result set to NaN")
        return out

    def call(*args):
        out, status = first(*args)
        if not status or not _flag(status):
            return out
        report("first", args, status, final=False)
        del out
        return finish(args, again(*args))
    call.lower, call.finish = again.lower, finish
    return call


#: Sites whose traced status took the host-callback fallback (warned once each).
_UNCACHEABLE: set = set()


def raise_if_failed(status, op, n, dtype, site):
    """An eager call's notices and refusal: raise by name when any matrix failed every attempt.

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
            _program_context()[1].append((status, (op, int(n), dtype, site)))
        else:
            if site not in _UNCACHEABLE and jax.process_index() == 0:
                print(f"distrib_la: UNCACHEABLE: a checked {op} at {site} is traced outside "
                      f"distrib_la.checked_program, so the program holding it carries a host callback "
                      f"and JAX's persistent cache never stores it (it compiles again in every "
                      f"process); build that program with distrib_la.checked_program",
                      file=sys.stderr, flush=True)
            _UNCACHEABLE.add(site)
            jax.debug.callback(lambda failed, errors: _notices((failed, errors), op, int(n), dtype, site,
                                                               final=True), *status)
        return
    failed = _flag(status, n, dtype)          # every rank: the failure and notice bits, one read
    _notices(status, op, n, dtype, site, final=True)
    if failed:
        raise ValueError(refusal_message(op, n, dtype, site, "every"))


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
