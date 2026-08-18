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
    "fit_scalar_samples",
    "fit_one_block",
    "format_cost_report",
    "run_fit_driver",
]

#: The two diagnostics ``mpa_store.write_fit_block`` REQUIRES, plus the
#: two this driver adds.  Fixed here rather than assembled per block
#: because the store stamps the key set on the first block and refuses a
#: later block that reports a different one — a quantity measured for
#: some blocks and not others reads back as zero for the rest, and a
#: zero condition number is a perfectly conditioned solve.
_DIAGNOSTIC_KEYS = ("condition", "backward_error", "residual", "n_valid")
_COMPANION_DIAGNOSTIC_KEYS = _DIAGNOSTIC_KEYS + (
    "condition_support", "condition_denominator",
    "backward_error_support", "backward_error_denominator")

# Both stamped solve modes use the fully JAX-resident QR root finder.  Loewner
# remains the deck default; selecting the Leon companion solve is explicit in
# the deck and fit-store provenance.
_FIT_EIG = "jax_qr"


@functools.lru_cache(maxsize=None)
def _scalar_fit_kernel(n_p, guard_items, rcond, solve):
    """One replicated scalar fit using the selected stamped policy."""
    import jax
    import jax.numpy as jnp

    guards = dict(guard_items)
    n = int(n_p)

    @jax.jit
    def _kernel(samples, z):
        return pade_fit.fit_mpa_poles(
            samples, z, n, guards=guards, rcond=float(rcond),
            eig=_FIT_EIG, solve=solve)

    return _kernel


def fit_scalar_samples(
    Wc, z_samples, n_p, *, guards=None, rcond=1.0e-13,
    solve="loewner",
):
    """Fit one scalar Wc sample vector with the body driver's exact policy."""
    import jax
    import jax.numpy as jnp

    n = int(n_p)
    z = np.asarray(z_samples, dtype=np.complex128)
    wc = np.asarray(Wc, dtype=np.complex128)
    if z.shape != (2 * n,) or wc.shape != z.shape:
        raise ValueError(
            "scalar MPA head requires Wc and z with shape (2*n_p,); "
            f"got Wc={wc.shape}, z={z.shape}, n_p={n}")
    resolved = pade_fit._resolve_guards(guards)
    pade_fit._check_solve_mode(solve)
    kernel = _scalar_fit_kernel(
        n, tuple(sorted(resolved.items())), float(rcond), solve)
    Omega, B, diagnostics = kernel(
        jnp.asarray(wc), jnp.asarray(z))
    Omega, B, diagnostics = jax.device_get((Omega, B, diagnostics))
    valid = np.asarray(diagnostics["valid"], dtype=bool)
    if not np.any(valid):
        raise ValueError("scalar MPA head fit rejected every pole")
    return {
        "Omega_p": np.asarray(Omega, dtype=np.complex128)[valid],
        "B_p": np.asarray(B, dtype=np.complex128)[valid],
        "condition": float(np.asarray(diagnostics["cond_pade"])),
        "condition_support": float(
            np.asarray(diagnostics["cond_support"])),
        "condition_denominator": float(
            np.asarray(diagnostics["cond_denominator"])),
        "backward_error": float(np.asarray(diagnostics["backward_error"])),
        "backward_error_support": float(
            np.asarray(diagnostics["backward_error_support"])),
        "backward_error_denominator": float(
            np.asarray(diagnostics["backward_error_denominator"])),
        "max_abs_residual": float(
            np.asarray(diagnostics["max_abs_residual"])),
        "n_valid": int(np.asarray(diagnostics["n_valid"])),
        "solve": solve,
        "affine": solve == "loewner",
        "eig": _FIT_EIG,
        "rcond": float(rcond),
    }


@functools.lru_cache(maxsize=None)
def _sharded_fit_kernel(mesh_xy, n_p, rcond, solve):
    """One compiled local-row fit; columns remain replicated."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from common.shard_map import shard_map

    row_axes = ("x", "y")
    block_spec = P(None, None, row_axes, None)
    pole_spec = P(None, None, row_axes, None)
    diag_spec = P(None, row_axes, None)
    guards = pade_fit._resolve_guards(None)
    n = int(n_p)

    def _local(block, z, row_ids, n_mu_logical, n_cols_logical):
        samples = jnp.transpose(block[:, 0], (1, 2, 0))
        n_rows, n_cols, n_omega = samples.shape
        tile = samples.reshape(n_rows * n_cols, n_omega)
        Omega, Bp, diag = pade_fit.fit_mpa_poles_batched(
            tile, z, n, guards=guards, rcond=rcond,
            eig=_FIT_EIG, solve=solve)
        Omega = jnp.transpose(
            Omega.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        Bp = jnp.transpose(
            Bp.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        valid = ((row_ids[:, None] < n_mu_logical)
                 & (jnp.arange(n_cols)[None, :] < n_cols_logical))

        def _diag(x):
            return jnp.where(valid, x.reshape(n_rows, n_cols), 0.0)[None]

        condition = _diag(diag["cond_pade"])
        condition_support = _diag(diag["cond_support"])
        condition_denominator = _diag(diag["cond_denominator"])
        backward = _diag(diag["backward_error"])
        backward_support = _diag(diag["backward_error_support"])
        backward_denominator = _diag(diag["backward_error_denominator"])
        residual = _diag(diag["max_abs_residual"])
        n_valid = _diag(diag["n_valid"])
        Omega = jnp.where(valid[None, None], Omega, 0.0 + 0.0j)
        Bp = jnp.where(valid[None, None], Bp, 0.0 + 0.0j)
        finite = (
            jnp.all(jnp.isfinite(Omega)) & jnp.all(jnp.isfinite(Bp))
            & jnp.all(jnp.isfinite(condition))
            & jnp.all(jnp.isfinite(condition_support))
            & jnp.all(jnp.isfinite(condition_denominator))
            & jnp.all(jnp.isfinite(backward))
            & jnp.all(jnp.isfinite(backward_support))
            & jnp.all(jnp.isfinite(backward_denominator))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(n_valid)))
        # One scalar-vector reduction, not a pmax plus a second synchronous
        # integer pmin.  Pole fits are otherwise completely row-local; this
        # is the only cross-rank communication in the compute kernel and it
        # exists solely to certify the block before its collective write.
        summary = lax.pmax(jnp.stack((
            jnp.max(condition),
            jnp.max(backward),
            (~finite).astype(jnp.float64),
        )), row_axes)
        maxima = summary[:2]
        finite = (summary[2] == 0.0).astype(jnp.int32)
        return (Omega, Bp, condition, condition_support,
                condition_denominator, backward, backward_support,
                backward_denominator, residual, n_valid, maxima, finite)

    mapped = shard_map(
        _local, mesh=mesh_xy,
        in_specs=(block_spec, P(None), P(row_axes), P(), P()),
        out_specs=(pole_spec, pole_spec, diag_spec, diag_spec, diag_spec,
                   diag_spec, diag_spec, diag_spec, diag_spec, diag_spec,
                   P(None), P()),
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
    rcond=1.0e-13,
    solve="loewner",
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

    The Loewner kernel returns its condition number, backward error and sample
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
    pade_fit._check_solve_mode(solve)
    kernel = _sharded_fit_kernel(mesh_xy, n, float(rcond), solve)
    z_dev = device_put_process_local(
        z, NamedSharding(mesh_xy, P(None)))
    row_ids = device_put_process_local(
        np.arange(n_mu_padded, dtype=np.int32),
        NamedSharding(mesh_xy, P(("x", "y"))))
    (Omega, B, condition, condition_support, condition_denominator,
     backward, backward_support, backward_denominator, residual, n_valid,
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
    if solve == "companion":
        diag_block.update({
            "condition_support": condition_support,
            "condition_denominator": condition_denominator,
            "backward_error_support": backward_support,
            "backward_error_denominator": backward_denominator,
        })
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
    rcond=1.0e-13,
    solve="loewner",
    provenance=None,
    occupation_state=None,
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
    pade_fit._check_solve_mode(solve)
    # Solver-consistency guards, not material tolerances.  A solve beyond
    # 1/rcond is numerically rank deficient by its own truncation policy;
    # sqrt(eps) is the ordinary backward-stability ceiling for complex128.
    # Observable accuracy remains a separate held-out gate.
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
    fit_provenance = dict(provenance or {})
    fit_provenance.update({
        "solve_mode": solve,
        "solve_rcond": rcond,
        "eig_mode": _FIT_EIG,
        "fit_fused": True,
    })
    mpa_store.allocate_fit_store_collective(
        fit_dest, mesh_xy=mesh_xy, n_q=n_q, n_mu=n_mu, n_p=n,
        diagnostic_keys=(
            _DIAGNOSTIC_KEYS if solve == "loewner"
            else _COMPANION_DIAGNOSTIC_KEYS),
        energy_unit=header["omega_units"],
        grid_hash=header["grid_hash"],
        table_hash=header["table_hash"],
        centroid_hash=header["centroid_hash"],
        unfold_tables=mpa_store.read_w_tables(w_src, w_name),
        provenance=fit_provenance,
        occupation_state=occupation_state)

    report = {
        "n_q": int(n_q),
        "n_mu": int(n_mu),
        "n_omega": int(n_omega),
        "n_p": n,
        "solve": solve,
        "eig": _FIT_EIG,
        "rcond": rcond,
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
        "pade_solves": 0,
        "seconds": {"read": 0.0, "fit": 0.0, "write": 0.0,
                    "finalize": 0.0, "total": 0.0},
    }

    ledger = None
    for q, lo, hi in tiling.fit_schedule(n_q, n_mu, n_omega, tile_bytes):
        stats = fit_one_block(
            w_src, w_name, fit_dest, q, np.arange(lo, hi), z, n,
            mesh_xy=mesh_xy, n_cols_buffer=plan["n_cols"],
            tile_bytes=tile_bytes, rcond=rcond,
            solve=solve,
            header=header)
        ledger = stats["ledger"]
        report["blocks_walked"] += 1
        report["columns_read"] += stats["n_cols"]
        report["elements_fitted"] += stats["n_elements"]
        report["bytes_read"] += stats["bytes_read"]
        report["peak_block_bytes"] = max(report["peak_block_bytes"],
                                         stats["bytes_read"])
        report["fit_dispatches"] += stats["fit_dispatches"]
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

    The two counts distinguish scheduler launches from per-element pole
    solves; conditioning, backward error and the finished-model residual
    are all produced by the same solve.
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
        f"  solve           {r['solve']}, rcond={r['rcond']:.1e}",
        f"  eig             {r['eig']}, fused=yes",
        f"  walk            {r['blocks_walked']} blocks "
        f"({r['n_blocks_per_q']} per q x {r['n_q']} q), "
        f"{r['n_cols_budget']} columns per block",
        f"  columns read    {r['columns_read']} "
        f"(= {r['n_q']} q x {r['n_mu']} columns)",
        f"  elements fitted {r['elements_fitted']}",
        f"  logical outputs {r['logical_outputs']} "
        f"(= 2*n_p per element: n_p Omega + n_p B)",
        f"  dispatches      {r['fit_dispatches']} fit, vmapped",
        f"  pole solves     {r['pade_solves']} "
        f"(1 per element; diagnostics reuse the same solve)",
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
