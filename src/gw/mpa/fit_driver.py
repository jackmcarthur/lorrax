"""The MPA fit stage's driver: read a column block, fit it, write it, finalize.

STAGING LOCATION: see ``gw/mpa/__init__.py``.  This is the module that
touches every other half of the multipole-W infrastructure — the
schedule (``tiling``), the bytes (``file_io.mpa_store``), the sample
grid (``sampling``) and the fit itself (``pade_fit``)
— and it exists because none of them can demonstrate the pipeline
alone.  The fit kernel proves it recovers planted poles from samples in
memory; the store proves it refuses a half-written file.  Neither
proves that a fit whose samples came off disk in ``n_cols``-wide blocks
and whose poles went back to disk block by block reconstructs the field
that was planted.  That claim spans both, so it is tested here.

THE LOOP, WHICH IS THE WHOLE MODULE
-----------------------------------
``plan_column_walk`` -> for each ``(q, column block)``: collective SlabIO
read -> row-local ``fit_mpa_poles_batched`` -> collective SlabIO write ->
rank-zero ledger commit.  Nothing accumulates across blocks except the
cost counters: the fit's OUTPUT is larger than its input at the
scheduled ``n_p`` (``tiling`` derives the factor), so holding the poles
until the last block would hold more than the samples that were already
declared not to fit.

THE ORDER OF OPERATIONS INSIDE ONE BLOCK IS LOAD-BEARING.  The read
comes first and the write comes last, with nothing partial in between,
so a block that is refused — a busted column budget, an unready
frequency slab — leaves the staged store byte-identical to what it was
before the block started.  That is what makes a refused block RESUMABLE
rather than corrupting: the ledger never learned about it, so the next
walk simply fits it again.  :func:`fit_one_block` is exposed separately
from :func:`run_fit_driver` for exactly this reason, and not only for
the tests: a driver resuming a crashed run walks the blocks its ledger
says are missing, which is a loop over ``fit_one_block`` and not a
re-entry into ``run_fit_driver``.

WHAT THIS MODULE DOES NOT DO.  It does not unfold — poles fitted on the
q wedge unfold the way W does, per q, and the staged store is explicit
that doing it at write time would store ``n_q_full`` copies of a tensor
the symmetry says is ``n_q_ibz`` of them.  It does not choose ``n_p``;
that is the deck's (``sampling.POLE_SCHEDULE``).  It does not evaluate
W — the samples arrive on disk from the screening sweep, and the
protocol grid they were evaluated on is stamped beside them.
"""

from __future__ import annotations

import functools
import time

import numpy as np

from file_io import mpa_store
from gw.mpa import pade_fit, tiling

__all__ = [
    "accumulate_over_pole_passes",
    "fit_one_block",
    "format_cost_report",
    "pole_pass_order",
    "run_fit_driver",
]

#: The two diagnostics ``mpa_store.write_fit_block`` REQUIRES, plus the
#: two this driver adds.  Fixed here rather than assembled per block
#: because the store stamps the key set on the first block and refuses a
#: later block that reports a different one — a quantity measured for
#: some blocks and not others reads back as zero for the rest, and a
#: zero condition number is a perfectly conditioned solve.
_DIAGNOSTIC_KEYS = ("condition", "backward_error", "residual", "n_valid")


@functools.lru_cache(maxsize=None)
def _sharded_fit_kernel(mesh_xy, n_p, guard_items, rcond):
    """One compiled local-row Padé fit; columns remain replicated."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from common.shard_map import shard_map

    row_axes = ("x", "y")
    block_spec = P(None, None, row_axes, None)
    pole_spec = P(None, None, row_axes, None)
    diag_spec = P(None, row_axes, None)
    guards = dict(guard_items)
    n = int(n_p)

    def _local(block, z, row_ids, n_mu_logical, n_cols_logical):
        samples = jnp.transpose(block[:, 0], (1, 2, 0))
        n_rows, n_cols, n_omega = samples.shape
        tile = samples.reshape(n_rows * n_cols, n_omega)
        Omega, Bp, diag = pade_fit.fit_mpa_poles_batched(
            tile, z, n, guards=guards, rcond=rcond)
        Omega = jnp.transpose(
            Omega.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        Bp = jnp.transpose(
            Bp.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        valid = ((row_ids[:, None] < n_mu_logical)
                 & (jnp.arange(n_cols)[None, :] < n_cols_logical))

        def _diag(x):
            return jnp.where(valid, x.reshape(n_rows, n_cols), 0.0)[None]

        condition = _diag(diag["cond_pade"])
        backward = _diag(diag["backward_error"])
        residual = _diag(diag["max_abs_residual"])
        n_valid = _diag(diag["n_valid"])
        Omega = jnp.where(valid[None, None], Omega, 0.0 + 0.0j)
        Bp = jnp.where(valid[None, None], Bp, 0.0 + 0.0j)
        maxima = lax.pmax(
            jnp.stack((jnp.max(condition), jnp.max(backward))), row_axes)
        finite = (
            jnp.all(jnp.isfinite(Omega)) & jnp.all(jnp.isfinite(Bp))
            & jnp.all(jnp.isfinite(condition))
            & jnp.all(jnp.isfinite(backward))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(n_valid)))
        finite = lax.pmin(finite.astype(jnp.int32), row_axes)
        return (Omega, Bp, condition, backward, residual, n_valid,
                maxima, finite)

    mapped = shard_map(
        _local, mesh=mesh_xy,
        in_specs=(block_spec, P(None), P(row_axes), P(), P()),
        out_specs=(pole_spec, pole_spec, diag_spec, diag_spec, diag_spec,
                   diag_spec, P(None), P()),
        check_vma=True)
    return jax.jit(mapped)


def fit_one_block(
    w_src,
    w_name,
    fit_dest,
    q,
    mu_cols,
    z_samples,
    n_p,
    *,
    mesh_xy,
    n_cols_buffer=None,
    tile_bytes=None,
    guards=None,
    rcond=1.0e-13,
    header=None,
):
    """Read one ``(q, column block)``, fit it, stage it.  Returns stats.

    Parameters
    ----------
    w_src, w_name
        The frequency-resolved W(omega) file and its dataset name.
    fit_dest
        The staged B/Omega store, already allocated.
    q
        Index into the STORED q axis — the same axis the W file uses,
        wedge or full.  The fit does not unfold.
    mu_cols
        The nu columns of this block.  Their count is checked against
        ``choose_column_budget`` by the reader, which refuses with the
        full arithmetic rather than truncating.
    z_samples
        ``(2*n_p,)`` complex — the protocol sample grid the W file was
        evaluated on.  Build it with ``sampling.double_parallel_grid``
        and check it against the file's stamped ``omega``; this
        function requires them to agree, because a fit against the
        wrong abscissae is the one failure the stamped grid exists to
        prevent.
    mesh_xy
        The run mesh.  The full ``('x', 'y')`` mesh shards the row axis;
        frequency and the scheduled column buffer remain replicated.

    Returns
    -------
    dict
        ``ledger`` (the store's completion ledger after this block) plus
        the block's counters: columns, elements, bytes read, dispatches
        and the per-stage seconds.  :func:`run_fit_driver` sums these
        into the run's cost report.

    The Padé kernel returns its condition number, backward error and sample
    residual from the same solve; diagnostics therefore add no second fit.

    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common.collectives import device_put_process_local

    n = int(n_p)
    hdr = mpa_store.read_w_header(w_src, w_name) if header is None else header
    n_mu = int(hdr["n_mu"])
    cols = mpa_store.normalise_columns(mu_cols, n_mu)
    if n_cols_buffer is None:
        n_cols_buffer = mpa_store.choose_column_budget(
            n_mu, int(hdr["n_omega"]), tile_bytes)
    t_read = time.perf_counter()
    block = mpa_store.read_w_columns_collective(
        w_src, w_name, q, cols, mesh_xy=mesh_xy,
        n_cols_buffer=n_cols_buffer, tile_bytes=tile_bytes, header=hdr)
    t_read = time.perf_counter() - t_read

    n_omega, _, n_mu_padded, _ = map(int, block.shape)
    n_cols = int(cols.size)
    z = np.asarray(z_samples, dtype=np.complex128)
    if z.shape != (2 * n,):
        raise ValueError(
            f"fit_one_block: z_samples has shape {z.shape} but n_p={n} "
            f"needs exactly {2 * n} samples.  FALSE case: "
            f"z_samples.shape == (2*n_p,) — the Pade system is square "
            f"on the sample support and n_p is not free to differ from "
            f"the grid the file was evaluated on.")
    if n_omega != 2 * n:
        raise ValueError(
            f"fit_one_block: the W file carries {n_omega} frequency "
            f"slabs but n_p={n} demands {2 * n}.  FALSE case: "
            f"n_omega == 2*n_p.  The double-parallel protocol samples "
            f"exactly 2*n_p points — n_p per line — so a mismatch means "
            f"either the fit is being run at an n_p the file was not "
            f"sampled for, or the file is not a double-parallel grid.")

    t_fit = time.perf_counter()
    resolved = pade_fit._resolve_guards(guards)
    guard_items = tuple(sorted(resolved.items()))
    kernel = _sharded_fit_kernel(mesh_xy, n, guard_items, float(rcond))
    z_dev = device_put_process_local(
        z, NamedSharding(mesh_xy, P(None)))
    row_ids = device_put_process_local(
        np.arange(n_mu_padded, dtype=np.int32),
        NamedSharding(mesh_xy, P(("x", "y"))))
    (Omega, B, condition, backward, residual, n_valid,
     maxima, finite) = kernel(
        block, z_dev, row_ids, np.int32(n_mu), np.int32(n_cols))
    Omega.block_until_ready()
    t_fit = time.perf_counter() - t_fit

    diag_block = {
        "condition": condition,
        "backward_error": backward,
        "residual": residual,
        "n_valid": n_valid,

    }
    maxima_host = np.asarray(maxima.addressable_data(0), dtype=np.float64)
    finite_host = bool(np.asarray(finite.addressable_data(0)))

    t_write = time.perf_counter()
    ledger = mpa_store.write_fit_block_collective(
        fit_dest, q, cols, Omega, B, diag_block, mesh_xy=mesh_xy,
        block_condition_max=maxima_host[0],
        block_backward_error_max=maxima_host[1],
        diagnostics_finite=finite_host)
    t_write = time.perf_counter() - t_write

    return {
        "ledger": ledger,
        "q": int(q),
        "n_cols": int(n_cols),
        "n_elements": int(n_mu * n_cols),
        "bytes_read": (n_omega * n_mu * n_cols
                       * mpa_store.COMPLEX128_BYTES),
        "fit_dispatches": 1,
        "diagnostic_dispatches": 0,
        "full_fits": int(n_mu * n_cols),
        "pade_solves": int(n_mu * n_cols),
        "seconds_read": t_read,
        "seconds_fit": t_fit,
        "seconds_write": t_write,
    }


def run_fit_driver(
    w_src,
    w_name,
    fit_dest,
    z_samples,
    n_p,
    *,
    mesh_xy,
    tile_bytes=None,
    guards=None,
    rcond=1.0e-13,
    certification=None,
    provenance=None,
    report_stream=None,
):
    """The whole fit stage: allocate, walk, stage, finalize, report.

    Reads the W(omega) header for its extents and its stamps, allocates
    the staged B/Omega store against them, walks
    ``tiling.fit_schedule`` q-major, and finalizes.  The W file's
    ``grid_hash``, ``table_hash`` and ``centroid_hash`` are carried into
    the fit store so the Sigma stage can assert that these poles came
    from that screening on that centroid set.

    Returns ``(ledger, report)`` — the store's completion ledger and the
    cost report, whose contents :func:`format_cost_report` prints.  The
    report is PRINTED ONCE, at finalize, to ``report_stream`` when one
    is given; a memory decision that only appears in a traceback is a
    decision nobody reads.
    """
    n = int(n_p)
    rcond = float(rcond)
    if not 0.0 < rcond < 1.0:
        raise ValueError("run_fit_driver requires 0 < rcond < 1")
    if certification is None:
        # These are solver-consistency guards, not material tolerances.
        # A solve beyond 1/rcond is numerically rank deficient by its own
        # truncation policy; sqrt(eps) is the ordinary backward-stability
        # ceiling for complex128 arithmetic.  Observable accuracy remains a
        # separate held-out/direct-denominator gate.
        certification = {
            "condition_max_allowed": 1.0 / rcond,
            "backward_error_max_allowed": float(
                np.sqrt(np.finfo(np.float64).eps)),
        }
    t_total = time.perf_counter()
    header = mpa_store.read_w_header(w_src, w_name)
    n_mu = header["n_mu"]
    n_omega = header["n_omega"]
    n_q = header["n_q_on_disk"]

    z = np.asarray(z_samples, dtype=np.complex128)
    stamped = np.asarray(header["omega"], dtype=np.complex128)
    if z.shape != stamped.shape or not np.array_equal(z, stamped):
        raise ValueError(
            f"run_fit_driver: the sample grid handed in does not match "
            f"the grid {w_name!r} was evaluated on.  Given "
            f"{z.shape} against a stamped {stamped.shape}"
            + ("" if z.shape != stamped.shape else
               f", worst |dz| = {np.max(np.abs(z - stamped)):.3e}")
            + ".  FALSE case: z_samples IS the file's stamped omega.  "
            "The file carries its grid and the protocol that made it "
            "precisely so a fit cannot be run against the wrong "
            "abscissae; re-deriving the grid and hoping it agrees is "
            "the failure that stamp exists to catch.")

    plan = tiling.plan_column_walk(n_mu, n_omega, tile_bytes)
    mpa_store.allocate_fit_store_collective(
        fit_dest, mesh_xy=mesh_xy, n_q=n_q, n_mu=n_mu, n_p=n,
        diagnostic_keys=_DIAGNOSTIC_KEYS,
        energy_unit=header["omega_units"],
        grid_hash=header["grid_hash"],
        table_hash=header["table_hash"],
        centroid_hash=header["centroid_hash"],
        unfold_tables=mpa_store.read_w_tables(w_src, w_name),
        provenance=provenance)

    report = {
        "n_q": int(n_q),
        "n_mu": int(n_mu),
        "n_omega": int(n_omega),
        "n_p": n,
        "n_cols_budget": int(plan["n_cols"]),
        "n_blocks_per_q": int(plan["n_blocks"]),
        "tile_bytes": int(plan["tile_bytes"]),
        "cost_sentence": plan["cost"],
        "blocks_walked": 0,
        "columns_read": 0,
        "elements_fitted": 0,
        "bytes_read": 0,
        "peak_block_bytes": 0,
        "fit_dispatches": 0,
        "diagnostic_dispatches": 0,
        "full_fits": 0,
        "pade_solves": 0,
        "seconds": {"read": 0.0, "fit": 0.0, "write": 0.0,
                    "finalize": 0.0, "total": 0.0},
    }

    ledger = None
    for q, lo, hi in tiling.fit_schedule(n_q, n_mu, n_omega, tile_bytes):
        stats = fit_one_block(
            w_src, w_name, fit_dest, q, np.arange(lo, hi), z, n,
            mesh_xy=mesh_xy, n_cols_buffer=plan["n_cols"],
            tile_bytes=tile_bytes, guards=guards, rcond=rcond,
            header=header)
        ledger = stats["ledger"]
        report["blocks_walked"] += 1
        report["columns_read"] += stats["n_cols"]
        report["elements_fitted"] += stats["n_elements"]
        report["bytes_read"] += stats["bytes_read"]
        report["peak_block_bytes"] = max(report["peak_block_bytes"],
                                         stats["bytes_read"])
        report["fit_dispatches"] += stats["fit_dispatches"]
        report["diagnostic_dispatches"] += stats["diagnostic_dispatches"]
        report["full_fits"] += stats["full_fits"]
        report["pade_solves"] += stats["pade_solves"]
        report["seconds"]["read"] += stats["seconds_read"]
        report["seconds"]["fit"] += stats["seconds_fit"]
        report["seconds"]["write"] += stats["seconds_write"]

    t_fin = time.perf_counter()
    from common.collectives import barrier, process_rank
    before = mpa_store.fit_completion_ledger(fit_dest)
    if not bool(np.asarray(before["blocks_done"]).all()):
        raise ValueError(
            "run_fit_driver: refusing collective finalize because the fit "
            "ledger still has unfinished columns")
    if process_rank() == 0:
        mpa_store.finalize_fit_store(
            fit_dest, certification=certification)
    barrier("mpa_fit_finalized")
    ledger = mpa_store.fit_completion_ledger(fit_dest)
    report["seconds"]["finalize"] = time.perf_counter() - t_fin

    report["logical_outputs"] = 2 * n * report["elements_fitted"]
    report["bytes_budget_total"] = (report["blocks_walked"]
                                    * report["tile_bytes"])
    report["condition_max"] = ledger["condition_max"]
    report["backward_error_max"] = ledger["backward_error_max"]
    report["seconds"]["total"] = time.perf_counter() - t_total

    if report_stream is not None and process_rank() == 0:
        report_stream.write(format_cost_report(report))
    return ledger, report


def format_cost_report(report):
    """The run's cost, as the theory plan demands it be stated.

    T section B: *"Cost reports must state 2n_p logical outputs, actual
    tau-node dispatches normalized to a GN sweep, and per-output R->q
    transforms and Dyson solves; a line-batched calculation is not 'one
    build' merely because it is one physical sweep."*  The fit stage's
    analogue of that demand is this: a vmapped block is not one solve
    merely because it is one dispatch, and the two numbers are printed
    beside each other so nobody has to infer the second from the first.

    The three counts distinguish scheduler launches, elementwise fits and
    Padé solves.  They are equal up to the number of elements per block:
    conditioning, backward error and the finished-model residual are all
    produced by the same fit.
    """
    r = report
    sec = r["seconds"]
    mib = 2 ** 20
    lines = [
        "",
        "MPA fit stage — cost report",
        "-" * 60,
        f"  geometry        n_q={r['n_q']} N_mu={r['n_mu']} "
        f"n_omega={r['n_omega']} n_p={r['n_p']}",
        f"  walk            {r['blocks_walked']} blocks "
        f"({r['n_blocks_per_q']} per q x {r['n_q']} q), "
        f"{r['n_cols_budget']} columns per block",
        f"  columns read    {r['columns_read']} "
        f"(= {r['n_q']} q x {r['n_mu']} columns)",
        f"  elements fitted {r['elements_fitted']}",
        f"  logical outputs {r['logical_outputs']} "
        f"(= 2*n_p per element: n_p Omega + n_p B)",
        f"  dispatches      {r['fit_dispatches']} fit + "
        f"{r['diagnostic_dispatches']} diagnostic, vmapped",
        f"  full fits       {r['full_fits']} "
        f"(1 per element; diagnostics reuse the same solve)",
        f"  Pade solves     {r['pade_solves']} "
        f"(1 per element)",
        f"  bytes read      {r['bytes_read']} B "
        f"({r['bytes_read'] / mib:.2f} MiB) against a budget of "
        f"{r['bytes_budget_total']} B "
        f"({r['bytes_budget_total'] / mib:.2f} MiB)",
        f"  peak block      {r['peak_block_bytes']} B "
        f"({r['peak_block_bytes'] / mib:.3f} MiB) against one tile of "
        f"{r['tile_bytes']} B ({r['tile_bytes'] / mib:.3f} MiB)",
        f"  budget rule     {r['cost_sentence']}",
        f"  worst condition {r['condition_max']:.3e}, worst backward "
        f"error {r['backward_error_max']:.3e}",
        f"  seconds         read {sec['read']:.3f}  fit "
        f"{sec['fit']:.3f}  write {sec['write']:.3f}  finalize "
        f"{sec['finalize']:.3f}  total {sec['total']:.3f}",
        "-" * 60,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one-pole-per-pass read, and the accumulation it exists to license
# ---------------------------------------------------------------------------

def pole_pass_order(n_p):
    """THE PINNED PASS ORDER: ascending pole index, 0 .. n_p - 1.

    Pinned, and stated as its own function, because the accumulation
    below is a re-associated floating-point sum and re-association is
    only reproducible if the order is.  Ascending index is not an
    arbitrary choice: ``pade_fit.fit_mpa_poles`` returns its poles
    sorted ascending in ``Re Omega`` (then ``Im Omega``), and the staged
    store keeps that axis leading, so pass ``p`` is the ``p``-th
    lowest-energy pole of every element at once.  Summing the small
    contributions before the large ones is also the better-conditioned
    direction for the accumulation, which is a happy consequence of an
    ordering chosen for reproducibility rather than an argument for it.
    """
    n = int(n_p)
    if n < 1:
        raise ValueError(
            f"pole_pass_order: n_p={n_p!r} is not a positive integer. "
            "FALSE case: int(n_p) >= 1.")
    return tuple(range(n))


def accumulate_over_pole_passes(
    fit_src, per_pole_fn, *, order=None, allow_partial=False
):
    """Run ``per_pole_fn`` once per pole and accumulate; return the sum.

    THIS IS THE CORRECTNESS LEMMA OF THE MEMORY-SAFE DESIGN, NOT A
    SIGMA IMPLEMENTATION.  It computes nothing physical and knows
    nothing about self-energies, Green's functions, quadratures or
    windows.  What it does is exhibit the pass STRUCTURE that the
    owner's Sigma design rests on — S section 5.5: *"run the existing
    sigma integration ~14 times, one B_q/Omega_q pair at a time, in its
    standard sharded form; wasteful of Green's-function construction,
    memory-safe by choice"* — in a form where its one mathematical
    claim can be checked without a Sigma stage existing.

    THE CLAIM.  A quantity that is a SUM OVER POLES of a per-pole term
    is unchanged when the sum is re-associated into one pass per pole,
    with only one ``(B_q, Omega_q)`` pair resident at a time.  That is
    the whole of it, and it is why the memory-shaped plan costs
    fourteen sets of spatial FFTs and no accuracy.  The real Sigma is
    such a sum: ``W(tau) = sum_p B_p exp(-i Omega_p tau)`` enters
    linearly, so every downstream contraction distributes over ``p``.

    THE ONE CAVEAT, AND IT IS FLOATING-POINT AND NOT ALGEBRAIC.  The
    re-association is exact in exact arithmetic and NOT bit-exact in
    floating point: ``sum_p`` evaluated as a vectorised reduction over
    a pole axis and ``sum_p`` evaluated as a sequential accumulation
    associate differently and differ at the last ulp or two.  So the
    pass order is PINNED (:func:`pole_pass_order`) and the accumulation
    is sequential in that order, which makes the result reproducible
    run to run even though it is not identical to the vectorised one.
    ``tests/test_mpa_fit_driver.py`` asserts both halves: agreement
    with the vectorised reference at machine precision, and BIT-EXACT
    agreement with an association-matched reference, which is what
    isolates the difference to association and nothing else.

    Parameters
    ----------
    fit_src
        The finalized staged store, as a path or an open h5py group.
    per_pole_fn
        ``f(p, Omega_p, B_p) -> array`` — the per-pass quantity.
        ``Omega_p`` and ``B_p`` are ``(n_q, N_mu, N_mu)``, this pole's
        slab and no other.  A real driver's ``f`` is the existing sigma
        integration; here it is whatever the caller passes.
    order
        Pass order.  ``None`` takes :func:`pole_pass_order`.  Passing
        one explicitly is how a test shows the association caveat is
        real.

    Returns
    -------
    ``(total, info)``
        ``total`` is the accumulated sum; ``info`` records the order
        actually walked, the number of passes, and the per-pass
        provenance the design asks the driver to own (S section 5.5:
        *"the 14 partial Sigmas only add coherently if each pass
        records the (pole index, E_ref_B, quadrature provenance) triple
        it used"* — the pole index is this module's half).
    """
    ledger = mpa_store.fit_completion_ledger(fit_src)
    n_p = ledger["n_p"]
    walk = (pole_pass_order(n_p) if order is None
            else tuple(int(p) for p in order))

    total = None
    passes = []
    for p in walk:
        Omega_p, B_p = mpa_store.read_pole_slice(
            fit_src, p, allow_partial=allow_partial)
        contrib = per_pole_fn(p, Omega_p, B_p)
        total = contrib if total is None else total + contrib
        passes.append({
            "pole_index": int(p),
            "re_omega_min": float(np.min(np.real(Omega_p))),
            "re_omega_max": float(np.max(np.real(Omega_p))),
            "gamma_max": float(np.max(np.abs(np.imag(Omega_p)))),
        })

    return total, {
        "order": walk,
        "n_passes": len(walk),
        "n_p": n_p,
        "passes": passes,
    }
