"""Non-negative SPD quadratics with principal systems on distributed faces."""
from __future__ import annotations

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from lxkit import memory
from distrib_la._result_check import checked_program
from distrib_la.plan import plan, _exchange
from distrib_la.resolve import resolve_backend, mesh_platform
from distrib_la.workspace import local_lu_workspace, lu_workspace_query


def _lower(fn, operands, mesh, label, *, linear=None, agree=False):
    """Compile and price the complete program plus its native workspace."""
    executable = fn.lower(*operands).compile()
    analysis = executable.memory_analysis()
    if analysis is None:
        raise RuntimeError(f"{label}: compiler returned no memory price")
    new = (int(analysis.output_size_in_bytes) + int(analysis.temp_size_in_bytes)
           - int(analysis.alias_size_in_bytes))
    vendor, outside = 0, memory.Term()
    if linear is not None and mesh_platform(mesh) == 'CUDA':
        n = int(operands[0].shape[0])
        if linear.backend == 'one_device':
            vendor = local_lu_workspace(mesh, n, operands[0].dtype)
        else:
            # One RHS column per rank leaves exactly one getrs chunk, so
            # the native query is cap-independent; no cap state is posted.
            vendor = lu_workspace_query(mesh, n, int(mesh.shape['y']), operands[0].dtype, 0)
            outside = memory.contexts()
    if agree:
        # Positive metadata only: the shared budget hook rejects negative values.
        figures = _exchange(f'nonnegative/{linear.backend}/{operands[0].shape[0]}',
                            dict(xla=max(0, new), vendor=vendor))
        new = max(row['xla'] for row in figures)
        vendor = max(row['vendor'] for row in figures)
    price = memory.Price(
        xla=(memory.Term(max(0, new), source='compiled one-system principal program')
             if linear is not None and linear.backend == 'one_device' else
             memory.Term(0.0, max(0, new) * int(mesh.size), 'compiled principal-pivot program')),
        vendor=memory.Term(float(vendor), source='native LU workspace query'),
        outside=outside, ranks=int(mesh.size), label=label,
        whole=int(operands[0].shape[0]) if linear is not None and linear.backend == 'one_device' else 0)
    run = (lambda *args: fn.finish(args, executable(*args))) if hasattr(fn, 'finish') else executable
    return run, price


def _compile(fn, operands, mesh, label):
    run, price = _lower(fn, operands, mesh, label)
    memory.admit(label, (("distributed face", price),), memory.room())
    return run


@lru_cache(maxsize=8)
def _programs(mesh, n, n_logical, linear):
    # Plan equality excludes room. This is one system with nrhs=Py<=n,
    # so a concrete provider always has the same one-round executable.
    face = NamedSharding(mesh, P("x", "y"))
    rows = NamedSharding(mesh, P("x"))
    replicated = NamedSharding(mesh, P())
    logical = jnp.arange(n) < n_logical
    # One RHS copy per y column keeps its layout legal without gathering A.
    nrhs = int(mesh.shape["y"])

    def normalize(a, b):
        diagonal = jnp.diag(a)
        valid = jnp.all(jnp.where(logical, diagonal > 0, True))
        scale = jnp.sqrt(jnp.where(logical, diagonal, 1.0))
        h = a / scale[:, None] / scale[None, :]
        h = jnp.where(logical[:, None] & logical[None, :], h, 0.0)
        h = h + jnp.diag((~logical).astype(a.dtype))
        c = jnp.where(logical, b / scale, 0.0)
        valid = valid & jnp.all(jnp.isfinite(h)) & jnp.all(jnp.isfinite(c))
        return h, c, scale, logical & (c > 0), valid

    def solve(h, c, free):
        system = jnp.where(free[:, None] & free[None, :], h, 0.0)
        system = system + jnp.diag((~free).astype(h.dtype))
        rhs = jnp.broadcast_to(jnp.where(free, c, 0.0)[:, None], (n, nrhs))
        return linear.batched(system[None], rhs[None])[0, :, 0]

    def inspect(h, c, z, free):
        hz = h @ z
        gradient = hz - c
        # A backward-error scale, invariant to a common rescaling of A/b.
        eps = jnp.finfo(h.dtype).eps
        tiny = jnp.finfo(h.dtype).tiny
        xtol = 64 * n_logical * eps * jnp.maximum(tiny, jnp.max(jnp.abs(z)))
        gtol = 64 * n_logical * eps * jnp.maximum(
            tiny, jnp.maximum(jnp.max(jnp.abs(c)), jnp.max(jnp.abs(hz))))
        bad = logical & jnp.where(free, z < -xtol, gradient < -gtol)
        residual = jnp.max(jnp.where(free, jnp.abs(gradient), 0.0))
        finite = jnp.all(jnp.isfinite(z)) & jnp.all(jnp.isfinite(gradient))
        return bad, jnp.array([jnp.sum(bad), finite,
                               residual / jnp.maximum(gtol, tiny)])

    def pivot(free, bad, block):
        last = jnp.max(jnp.where(bad, jnp.arange(n), -1))
        flip = jnp.where(block, bad, jnp.arange(n) == last)
        return logical & (free ^ flip)

    def finish(z, scale):
        return jnp.where(logical, jnp.maximum(z, 0.0) / scale, 0.0)

    return (
        jax.jit(normalize, in_shardings=(face, rows),
                out_shardings=(face, rows, rows, rows, replicated),
                donate_argnums=(0, 1)),
        checked_program(solve, mesh, rows, in_shardings=(face, rows, rows)),
        jax.jit(inspect, in_shardings=(face, rows, rows, rows),
                out_shardings=(rows, replicated)),
        jax.jit(pivot, in_shardings=(rows, rows, replicated), out_shardings=rows),
        jax.jit(finish, in_shardings=(rows, rows), out_shardings=rows),
    )


def _principal(mesh, n, n_logical, linear, operands, programs):
    """Admit the complete first provider with a real, lazily priced mesh backup."""
    label = 'nonnegative quadratic principal solve'
    run, price = _lower(programs[1], operands, mesh, label, linear=linear,
                        agree=linear.backend == 'one_device')
    held = {linear.backend: (run, programs)}

    def backup():
        backend = resolve_backend('solve_lu', 'distributed', mesh, n=n, size_route=False)
        other = plan('solve_lu', mesh, backend=backend, n=n, batched_route='auto')
        bundle = _programs(mesh, n, n_logical, other)
        run, price = _lower(bundle[1], operands, mesh, label, linear=other, agree=True)
        held['distributed face'] = (run, bundle)
        return price

    routes = [(linear.backend, price)]
    if linear.backend == 'one_device':
        routes.append(('distributed face', backup))
    room = memory.room()
    if linear.backend == 'one_device' and price.bytes() > room:
        # A whole matrix is eligible only within the planning room.
        routes = [('distributed face', backup())]
    chosen = memory.admit(label, routes, room)
    run, bundle = held[chosen]
    return run, *bundle[2:]


def solve_nonnegative_quadratic(a, b, *, mesh, n_logical, maxiter=None):
    """Minimize ``0.5*x.T@a@x - b.T@x`` with ``x >= 0``.

    Parameters
    ----------
    a : jax.Array
        Real float64 SPD ``(n_pad, n_pad)`` on ``P('x','y')``. Units are
        objective / x**2. Any regularization belongs to the caller. Consumed
        by donation while normalizing the problem; do not reuse afterwards.
    b : jax.Array
        Real float64 ``(n_pad,)`` on ``P('x')``, replicated along y. Units
        are objective / x. Consumed by donation, as ``a`` is.
    mesh : jax.sharding.Mesh
        The named x/y device mesh. The CUDA leg uses its existing local
        capacity route where the complete program and native scratch fit;
        a whole-mesh LU is its lazy backup. No matrix is gathered to hosts.
    n_logical : int
        Physical variable count in ``[1,n_pad]``. Pad variables are fixed
        exactly to zero and are excluded from convergence checks.
    maxiter : int, optional
        Maximum principal solves; defaults to ``20*n_logical``. A failed
        solve or non-convergence raises instead of returning partial weights.

    Returns
    -------
    jax.Array
        Non-negative ``(n_pad,)`` solution on ``P('x')``, in x units.

    Notes
    -----
    The caller resolves the shared ``lxkit.memory`` run budget first.
    Positive diagonal scaling preserves the quadratic and its active set.
    Block principal pivots exchange variables with negative primal or dual
    values. Three unsuccessful block pivots switch to one variable per
    pivot until the violation count decreases. Only scalar convergence
    information leaves the mesh. Every masked principal matrix returns to
    the original face; inactive rows/columns are identity, so the physical
    active submatrix has its exact SPD solution without dynamic gathering.
    """
    n, n_logical = int(a.shape[0]), int(n_logical)
    if (a.ndim != 2 or a.shape != (n, n) or b.shape != (n,)
            or a.dtype != np.dtype("float64") or b.dtype != a.dtype
            or not 0 < n_logical <= n):
        raise ValueError("nonnegative quadratic requires float64 square A, vector b, "
                         "and 0 < n_logical <= padded extent")
    if n % int(mesh.shape["x"]) or n % int(mesh.shape["y"]):
        raise ValueError("nonnegative quadratic extent must be mesh padded")
    maxiter = 20 * n_logical if maxiter is None else int(maxiter)
    if maxiter < 1:
        raise ValueError("nonnegative quadratic maxiter must be positive")
    room = memory.room()
    backend = resolve_backend("solve_lu", "distributed", mesh, n=n, room=room,
                              size_route=mesh_platform(mesh) == 'CUDA')
    linear = plan("solve_lu", mesh, backend=backend, n=n,
                  batched_route="auto", budget_bytes=room)
    normalize, solve, inspect, pivot, finish = _programs(mesh, n, n_logical, linear)
    normalize = _compile(normalize, (a, b), mesh, "nonnegative quadratic scaling")
    h, c, scale, free, valid = normalize(a, b)
    if not bool(np.asarray(valid)):
        raise ValueError("nonnegative quadratic requires finite data and positive diagonal")
    solve, inspect, pivot, finish = _principal(
        mesh, n, n_logical, linear, (h, c, free), (normalize, solve, inspect, pivot, finish))
    # Shape-only dummy outputs avoid executing a solve merely to price checks.
    z = jax.ShapeDtypeStruct(c.shape, c.dtype, sharding=c.format)
    inspect = _compile(inspect, (h, c, z, free), mesh, "nonnegative quadratic KKT check")
    finish = _compile(finish, (z, scale), mesh, "nonnegative quadratic result")
    best, patience = n_logical + 1, 3
    for _ in range(maxiter):
        z = solve(h, c, free)
        bad, stats = inspect(h, c, z, free)
        count, finite, residual = np.asarray(stats)
        if not finite or residual > 32:
            raise RuntimeError("GATE nonnegative_quadratic: principal solve failed "
                               f"the KKT residual check ({residual:g} tolerance units)")
        count = int(count)
        if count == 0:
            return finish(z, scale)
        if count < best:
            best, patience, use_block = count, 3, True
        elif patience:
            patience, use_block = patience - 1, True
        else:
            use_block = False
        block = jax.device_put(np.array(use_block), NamedSharding(mesh, P()))
        free = pivot(free, bad, block)
    raise RuntimeError("GATE nonnegative_quadratic: principal pivots did not "
                       f"converge after {maxiter} solves ({count} KKT violations)")
