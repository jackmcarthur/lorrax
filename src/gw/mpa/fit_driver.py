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
``plan_column_walk`` -> authenticate the partial ledger -> divide unfinished
``(q, column block)`` entries into fixed 32-block checkpoint epochs.  Within
each epoch one payload writer performs collective SlabIO read -> row-local
``fit_mpa_poles_batched`` -> queue and drain collective pole writes for every
block.  Only after that handle closes does rank zero publish the epoch's
completion journal.  No pole tensor accumulates across blocks: the persistent
object within an epoch is the file handle and its cached dataset handles.  The
per-block drain is the required ownership transfer before the next W-file
read; the epoch close is the bounded durability transaction.

THE ORDER OF OPERATIONS INSIDE ONE BLOCK IS LOAD-BEARING.  The read comes
first and the write comes last, so a block refused before its write leaves
the staged store byte-identical.  In the production session, the on-disk
ledger remains unchanged until every queued payload write has drained and
the collective handle has closed; a failed session can therefore leave
uncommitted payload bytes but can never certify them as consumable.  The
one-block :func:`fit_one_block` spelling remains the surgical resume API and
commits that one block before returning.

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
import os
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

# Loewner and the linear companion diagnostic use the fully JAX-resident QR
# root finder.  The Padé--Thiele diagnostic uses LAPACK/cuSOLVER ``geev``
# because that is the root step in Yambo's published PT implementation; its
# lack of GPU batching is accepted and printed as part of this diagnostic
# route's cost rather than changing the algorithm being compared.
_FIT_EIG = "jax_qr"
# One checkpoint transaction after this many scheduled q/column blocks.  The
# receipt schema owns the value because it authenticates opens against epochs;
# this alias keeps the loop readable without creating a second policy value.
_FIT_CHECKPOINT_BLOCKS = mpa_store.FIT_IO_CHECKPOINT_BLOCKS
# The source block is sharded over every process before fitting.  Let a
# default run spend one legacy dense-matrix tile *per process* on that read,
# but never increase the local allowance beyond this cap.  The Loewner
# kernel still consumes one legacy global tile at a time; separating these
# two widths is what removes HDF5 calls and output drains without changing a
# single elementwise solve.
_FIT_IO_LOCAL_EXPANSION_CAP_BYTES = 64 * 2 ** 20


def _fit_eig(solve):
    return "lapack" if solve == "thiele" else _FIT_EIG


def _fit_column_plans(n_mu, n_omega, tile_bytes, mesh_xy):
    """Return independent source-I/O and Loewner-compute column plans.

    An explicit ``tile_bytes`` retains its historical meaning and disables
    automatic expansion.  With the default one-tile policy, the collective
    source block can use the processes that shard its row axis: its global
    allowance grows by ``P`` while its local increase is capped at 64 MiB.
    The compute plan remains the historical one-global-tile shape so the
    compiled kernel, poles and arithmetic are unchanged.
    """
    compute_bytes = (
        mpa_store.one_tile_bytes(n_mu) if tile_bytes is None
        else int(tile_bytes))
    processes = max(1, int(getattr(mesh_xy, "size", 1)))
    if tile_bytes is not None or processes == 1:
        io_bytes = compute_bytes
    else:
        full_q_bytes = (
            int(n_omega) * int(n_mu) * int(n_mu)
            * mpa_store.COMPLEX128_BYTES)
        local_increment = min(
            compute_bytes, _FIT_IO_LOCAL_EXPANSION_CAP_BYTES)
        io_bytes = min(
            full_q_bytes,
            max(compute_bytes, processes * local_increment))
    return (
        tiling.plan_column_walk(n_mu, n_omega, io_bytes),
        tiling.plan_column_walk(n_mu, n_omega, compute_bytes),
        processes,
    )


@functools.lru_cache(maxsize=None)
def _scalar_fit_kernel(n_p, guard_items, rcond, solve):
    """One replicated scalar fit using the selected stamped policy."""
    import jax
    import jax.numpy as jnp

    guards = dict(guard_items)
    n = int(n_p)
    eig = _fit_eig(solve)

    @jax.jit
    def _kernel(samples, z):
        return pade_fit.fit_mpa_poles(
            samples, z, n, guards=guards, rcond=float(rcond),
            eig=eig, solve=solve)

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
        "eig": _fit_eig(solve),
        "rcond": float(rcond),
    }


@functools.lru_cache(maxsize=None)
def _sharded_fit_kernel(mesh_xy, n_p, rcond, solve, ordered=False):
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
    eig = _fit_eig(solve)

    def _local(block, negative_block, z, row_ids, n_mu_logical,
               n_cols_logical):
        samples = jnp.transpose(block[:, 0], (1, 2, 0))
        n_rows, n_cols, n_omega = samples.shape
        tile = samples.reshape(n_rows * n_cols, n_omega)
        if ordered:
            negative = jnp.transpose(
                negative_block[:, 0], (1, 2, 0)).reshape(
                    n_rows * n_cols, n_omega)
            Omega, Bp, Dp, diag = pade_fit.fit_mpa_poles_batched(
                tile, z, n, W_negative_tile=negative, return_odd=True,
                guards=guards, rcond=rcond, eig=eig, solve=solve)
        else:
            Omega, Bp, diag = pade_fit.fit_mpa_poles_batched(
                tile, z, n, guards=guards, rcond=rcond,
                eig=eig, solve=solve)
            Dp = jnp.zeros_like(Bp)
        Omega = jnp.transpose(
            Omega.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        Bp = jnp.transpose(
            Bp.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        Dp = jnp.transpose(
            Dp.reshape(n_rows, n_cols, n), (2, 0, 1))[:, None]
        valid = ((row_ids[:, None] < n_mu_logical)
                 & (jnp.arange(n_cols)[None, :] < n_cols_logical))

        def _diag(x):
            return jnp.where(valid, x.reshape(n_rows, n_cols), 0.0)[None]

        def _diag_finite(x):
            shaped = x.reshape(n_rows, n_cols)
            return jnp.all(jnp.isfinite(jnp.where(valid, shaped, 0.0)))

        condition = _diag(diag["cond_pade"])
        backward = _diag(diag["backward_error"])
        Omega = jnp.where(valid[None, None], Omega, 0.0 + 0.0j)
        Bp = jnp.where(valid[None, None], Bp, 0.0 + 0.0j)
        Dp = jnp.where(valid[None, None], Dp, 0.0 + 0.0j)
        finite = (
            jnp.all(jnp.isfinite(Omega)) & jnp.all(jnp.isfinite(Bp))
            & jnp.all(jnp.isfinite(Dp))
            & jnp.all(jnp.isfinite(condition))
            & jnp.all(jnp.isfinite(backward))
            & _diag_finite(diag["cond_support"])
            & _diag_finite(diag["cond_denominator"])
            & _diag_finite(diag["backward_error_support"])
            & _diag_finite(diag["backward_error_denominator"])
            & _diag_finite(diag["max_abs_residual"])
            & _diag_finite(diag["n_valid"]))
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
        return Omega, Bp, Dp, condition, maxima, finite

    out_specs = (pole_spec, pole_spec, pole_spec, diag_spec, P(None), P())
    if ordered:
        mapped = shard_map(
            _local, mesh=mesh_xy,
            in_specs=(
                block_spec, block_spec, P(None), P(row_axes), P(), P()),
            out_specs=out_specs, check_vma=True)
    else:
        def _incumbent(block, z, row_ids, n_mu_logical, n_cols_logical):
            return _local(
                block, block, z, row_ids, n_mu_logical, n_cols_logical)

        mapped = shard_map(
            _incumbent, mesh=mesh_xy,
            in_specs=(block_spec, P(None), P(row_axes), P(), P()),
            out_specs=out_specs, check_vma=True)
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
    w_negative_name=None,
    negative_header=None,
    mesh_xy,
    n_cols_buffer=None,
    tile_bytes=None,
    fit_tile_bytes=None,
    rcond=1.0e-13,
    solve="loewner",
    header=None,
    fit_writer=None,
    w_reader=None,
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
    tile_bytes, fit_tile_bytes
        Collective source-I/O and compute-microbatch budgets, respectively.
        The production driver may read a wider all-P-sharded source block and
        pass the historical one-tile budget as ``fit_tile_bytes``.  This
        changes only batching: every element still enters the same kernel
        exactly once.  Direct callers that omit ``fit_tile_bytes`` retain
        one fit dispatch per source block.
    fit_writer
        Optional live :class:`file_io.mpa_store.FitWriter`.  The production
        driver supplies one for the checkpoint epoch.  ``None`` uses the
        one-block committed spelling for surgical resume callers.
    w_reader
        Optional live :class:`file_io.mpa_store.WColumnReader`.  Production
        supplies the same reader for every block in the checkpoint epoch;
        ``None`` keeps the one-shot source lifetime for surgical callers.

    Returns
    -------
    dict
        ``ledger`` (the session ledger after this block; on-disk committed
        when no ``fit_writer`` was supplied) plus
        the block's counters: columns, elements, bytes read, dispatches
        and the per-stage seconds.  :func:`run_fit_driver` sums these
        into the run's cost report.

    The Loewner kernel returns its condition number, backward error and sample
    residual from the same solve; diagnostics therefore add no second fit.

    """
    import jax.numpy as jnp
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
    if w_reader is None:
        block = mpa_store.read_w_columns_collective(
            w_src, w_name, q, cols, mesh_xy=mesh_xy,
            n_cols_buffer=n_cols_buffer, tile_bytes=tile_bytes, header=hdr)
    else:
        block = w_reader.read(
            w_name, q, cols, n_cols_buffer=n_cols_buffer,
            tile_bytes=tile_bytes)
    negative_block = block
    if w_negative_name is not None:
        if w_reader is None:
            negative_block = mpa_store.read_w_columns_collective(
                w_src, w_negative_name, q, cols, mesh_xy=mesh_xy,
                n_cols_buffer=n_cols_buffer, tile_bytes=tile_bytes,
                header=negative_header)
        else:
            negative_block = w_reader.read(
                w_negative_name, q, cols,
                n_cols_buffer=n_cols_buffer, tile_bytes=tile_bytes)
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
    ordered = w_negative_name is not None
    kernel = _sharded_fit_kernel(
        mesh_xy, n, float(rcond), solve, ordered)
    z_dev = device_put_process_local(
        z, NamedSharding(mesh_xy, P(None)))
    row_ids = device_put_process_local(
        np.arange(n_mu_padded, dtype=np.int32),
        NamedSharding(mesh_xy, P(("x", "y"))))
    fit_n_cols = (
        int(n_cols_buffer) if fit_tile_bytes is None else
        mpa_store.choose_column_budget(
            n_mu, n_omega * (2 if ordered else 1), fit_tile_bytes))
    outputs = []
    for start in range(0, int(n_cols_buffer), fit_n_cols):
        stop = min(start + fit_n_cols, int(n_cols_buffer))
        logical = max(0, min(n_cols, stop) - start)
        positive = block[..., start:stop]
        if ordered:
            value = kernel(
                positive, negative_block[..., start:stop], z_dev, row_ids,
                np.int32(n_mu), np.int32(logical))
        else:
            value = kernel(
                positive, z_dev, row_ids,
                np.int32(n_mu), np.int32(logical))
        outputs.append(value)
    if len(outputs) == 1:
        Omega, B, B_odd, condition, maxima, finite = outputs[0]
    else:
        Omega, B, B_odd, condition = (
            jnp.concatenate([value[index] for value in outputs], axis=-1)
            for index in range(4))
        maxima = jnp.max(jnp.stack(
            [value[4] for value in outputs]), axis=0)
        finite = jnp.min(jnp.stack([value[5] for value in outputs]))
    Omega.block_until_ready()
    t_fit = time.perf_counter() - t_fit

    diag_block = {"condition": condition}
    maxima_host = np.asarray(maxima.addressable_data(0), dtype=np.float64)
    finite_host = bool(np.asarray(finite.addressable_data(0)))

    t_write = time.perf_counter()
    write = (mpa_store.write_fit_block_collective
             if fit_writer is None else fit_writer.write_block)
    if fit_writer is None:
        ledger = write(
            fit_dest, q, cols, Omega, B, diag_block, mesh_xy=mesh_xy,
            B_odd_p_block=B_odd if ordered else None,
            block_condition_max=maxima_host[0],
            block_backward_error_max=maxima_host[1],
            diagnostics_finite=finite_host)
    else:
        ledger = write(
            q, cols, Omega, B, diag_block,
            B_odd_p_block=B_odd if ordered else None,
            block_condition_max=maxima_host[0],
            block_backward_error_max=maxima_host[1],
            diagnostics_finite=finite_host)
    t_write = time.perf_counter() - t_write

    return {
        "ledger": ledger,
        "q": int(q),
        "n_cols": int(n_cols),
        "n_elements": int(n_mu * n_cols),
        "bytes_read": ((2 if ordered else 1) * n_omega * n_mu * n_cols
                       * mpa_store.COMPLEX128_BYTES),
        "fit_dispatches": len(outputs),
        "pade_solves": int(n_mu * n_cols),
        "seconds_read": t_read,
        "seconds_fit": t_fit,
        "seconds_write": t_write,
    }


def _bind_sample_artifact_identities(header, negative_header, provenance):
    """Bind fit provenance to WFN and charge-zeta sample identities."""
    identity_groups = (
        ("WFN", mpa_store.WFN_IDENTITY_PROVENANCE_KEYS),
        ("charge-zeta", mpa_store.CHARGE_ZETA_IDENTITY_PROVENANCE_KEYS),
    )

    def text(value):
        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value).decode("utf-8")
        return str(value)

    def identity(row, label, identity_label, keys):
        values = dict(row.get("provenance", {}))
        present = [key for key in keys if key in values]
        if not present:
            return None
        if len(present) != len(keys):
            raise ValueError(
                f"run_fit_driver: {label} sample {identity_label} identity "
                "is partial; "
                f"present={present}, required={list(keys)}")
        return {key: text(values[key]) for key in keys}

    bound = dict(provenance or {})
    for identity_label, keys in identity_groups:
        positive = identity(header, "positive", identity_label, keys)
        negative = (
            None if negative_header is None else
            identity(negative_header, "negative", identity_label, keys))
        if negative_header is not None and negative != positive:
            raise ValueError(
                "run_fit_driver: ordered positive/negative sample stores "
                f"carry different {identity_label} identities: "
                f"positive={positive}, negative={negative}")
        caller_identity = {
            key: text(bound[key]) for key in keys if key in bound}
        if positive is None:
            if caller_identity:
                raise ValueError(
                    "run_fit_driver: caller provenance cannot inject a "
                    f"{identity_label} identity absent from the sample store")
            # Legacy direct/offline sample fitting remains possible, but the
            # resulting fit is unauthenticated and explicit reuse supplies the
            # missing expected pair and refuses it.
            continue
        mismatched = [
            key for key, value in caller_identity.items()
            if value != positive[key]
        ]
        if mismatched:
            raise ValueError(
                f"run_fit_driver: caller {identity_label} provenance "
                f"disagrees with the sample store in {mismatched}")
        bound.update(positive)
    return bound


def run_fit_driver(
    w_src,
    w_name,
    fit_dest,
    z_samples,
    n_p,
    *,
    w_negative_name=None,
    mesh_xy,
    tile_bytes=None,
    fit_tile_bytes=None,
    rcond=1.0e-13,
    solve="loewner",
    provenance=None,
    occupation_state=None,
    report_stream=None,
    overwrite_incompatible=False,
    finalize=True,
):
    """The whole fit stage: resume/allocate, checkpoint, optionally finalize.

    Reads the W(omega) header for its extents and its stamps, allocates
    the staged B/Omega store against them, walks
    ``tiling.fit_schedule`` q-major, and finalizes.  The W file's
    ``grid_hash``, ``table_hash`` and ``centroid_hash`` are carried into
    the fit store so the Sigma stage can assert that these poles came
    from that screening on that centroid set.  The sample's canonical WFN
    fingerprint and scheme are copied as one identity pair; caller provenance
    may agree with that pair but cannot manufacture or override it.

    Returns ``(ledger, report)`` — the store's completion ledger and the
    cost report, whose contents :func:`format_cost_report` prints.  The
    report is PRINTED ONCE, at finalize, to ``report_stream`` when one
    is given; a memory decision that only appears in a traceback is a
    decision nobody reads.

    RECEIPT SCOPE.  The production ``finalize=False`` body path persists the
    successful attempt's I/O receipt before returning to the separate scalar
    head/root-COMPLETE transaction.  A standalone ``finalize=True`` call
    returns and optionally prints the same report but deliberately does not
    append it after COMPLETE: its own finalize duration is unknowable before
    that last write, and mutating a finalized artifact to record it would
    violate the store's write-once contract.

    DEFAULT MEMORY POLICY.  The on-disk walk and the elementwise Loewner
    launch have independent widths.  When ``tile_bytes`` is omitted, the
    collective read can spend up to one legacy matrix tile per row-sharding
    process (with a 64 MiB local expansion cap), while each fit dispatch
    retains the historical one-global-tile budget.  Passing ``tile_bytes``
    explicitly preserves the old single-budget behavior.
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
    negative_header = None
    ordered = w_negative_name is not None
    if ordered:
        negative_header = mpa_store.read_w_header(w_src, w_negative_name)
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
    if ordered:
        negative_stamped = np.asarray(
            negative_header["omega"], dtype=np.complex128)
        identity_keys = (
            "n_mu", "n_omega", "n_q_on_disk", "omega_units",
            "table_hash", "centroid_hash")
        mismatched = [key for key in identity_keys
                      if negative_header[key] != header[key]]
        if mismatched or negative_stamped.shape != z.shape \
                or not np.array_equal(negative_stamped, -z):
            raise ValueError(
                "run_fit_driver: ordered W(-z) samples do not match the "
                f"positive sample store; mismatched={mismatched}, "
                f"negative_shape={negative_stamped.shape}, z_shape={z.shape}, "
                "required negative omega == -z exactly")

    effective_n_omega = n_omega * (2 if ordered else 1)
    plan, compute_plan, io_processes = _fit_column_plans(
        n_mu, effective_n_omega, tile_bytes, mesh_xy)
    if fit_tile_bytes is not None:
        compute_plan = tiling.plan_column_walk(
            n_mu, effective_n_omega, fit_tile_bytes)
        if compute_plan["n_cols"] > plan["n_cols"]:
            raise ValueError(
                "run_fit_driver: fit_tile_bytes may not make a compute "
                "microbatch wider than its source-I/O block; got "
                f"{compute_plan['n_cols']} fit columns against "
                f"{plan['n_cols']} I/O columns")
    fit_provenance = _bind_sample_artifact_identities(
        header, negative_header, provenance)
    fit_provenance.update({
        "solve_mode": solve,
        "solve_rcond": rcond,
        "eig_mode": _fit_eig(solve),
        "fit_fused": True,
    })
    schedule = tuple(tiling.fit_schedule(
        n_q, n_mu, effective_n_omega, plan["tile_bytes"]))
    ledger = None
    must_allocate = not os.path.exists(os.fspath(fit_dest))
    if not must_allocate:
        try:
            ledger = mpa_store.validate_fit_store_for_resume(
                fit_dest, n_q=n_q, n_mu=n_mu, n_p=n,
                energy_unit=header["omega_units"],
                grid_hash=header["grid_hash"],
                table_hash=header["table_hash"],
                centroid_hash=header["centroid_hash"],
                provenance=fit_provenance, ordered_residues=ordered,
                schedule=schedule, occupation_state=occupation_state)
            if ledger["complete"]:
                raise ValueError(
                    "completed MPA fit stores are write-once; consume the "
                    "fit through mpa_fit_reuse_file or choose a new variant")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            if not bool(overwrite_incompatible):
                raise ValueError(
                    "GATE mpa_partial_fit_compatible: existing fit store is "
                    "not a compatible resumable transaction and will not be "
                    "truncated. Set mpa_overwrite_completed_artifacts = true "
                    "only for destructive regeneration. Cause: "
                    f"{type(exc).__name__}: {exc}") from exc
            must_allocate = True
    from common.collectives import barrier, process_rank
    # ``validate_fit_store_for_resume`` is an all-rank serial-h5py read.
    # Do not let a fast rank enter either collective allocation or an
    # append-mode FitWriter while a slower peer still owns that reader.
    barrier("mpa_fit_resume_readers_closed")
    if must_allocate:
        ledger = mpa_store.allocate_fit_store_collective(
            fit_dest, mesh_xy=mesh_xy, n_q=n_q, n_mu=n_mu, n_p=n,
            energy_unit=header["omega_units"],
            grid_hash=header["grid_hash"],
            table_hash=header["table_hash"],
            centroid_hash=header["centroid_hash"],
            unfold_tables=mpa_store.read_w_tables(w_src, w_name),
            provenance=fit_provenance,
            occupation_state=occupation_state,
            ordered_residues=ordered)
    # Fresh allocation returns a final all-rank serial ledger read; resume
    # validation does too.  No FitWriter may open append-mode until every
    # rank has closed whichever reader produced ``ledger``.
    barrier("mpa_fit_ledger_readers_closed")

    report = {
        "n_q": int(n_q),
        "n_mu": int(n_mu),
        "n_omega": int(n_omega),
        "n_p": n,
        "ordered_residues": ordered,
        "solve": solve,
        "eig": _fit_eig(solve),
        "rcond": rcond,
        "certification": dict(certification),
        "n_cols_budget": int(plan["n_cols"]),
        "n_blocks_per_q": int(plan["n_blocks"]),
        "tile_bytes": int(plan["tile_bytes"]),
        "cost_sentence": plan["cost"],
        "fit_n_cols_budget": int(compute_plan["n_cols"]),
        "fit_tile_bytes": int(compute_plan["tile_bytes"]),
        "io_processes": int(io_processes),
        "blocks_walked": 0,
        "blocks_skipped": 0,
        "checkpoint_blocks": _FIT_CHECKPOINT_BLOCKS,
        "checkpoint_epochs_planned": 0,
        "checkpoint_epochs_committed": 0,
        "sample_source_opens": 0,
        "sample_source_closes": 0,
        "sample_h5d_reads": 0,
        "columns_read": 0,
        "elements_fitted": 0,
        "bytes_read": 0,
        "peak_block_bytes": 0,
        "fit_dispatches": 0,
        "pade_solves": 0,
        "seconds": {"source_open": 0.0, "read": 0.0,
                    "source_close": 0.0, "fit": 0.0, "write": 0.0,
                    "finalize": 0.0, "total": 0.0},
    }

    pending = []
    for q, lo, hi in schedule:
        block_done = np.asarray(ledger["blocks_done"])[int(q), int(lo):int(hi)]
        if bool(block_done.all()):
            report["blocks_skipped"] += 1
        elif bool(block_done.any()):
            raise ValueError(
                f"run_fit_driver: schedule block {(q, lo, hi)} is partly "
                "committed after ledger validation")
        else:
            pending.append((q, lo, hi))
    epochs = tuple(
        tuple(pending[start:start + _FIT_CHECKPOINT_BLOCKS])
        for start in range(0, len(pending), _FIT_CHECKPOINT_BLOCKS))
    report["checkpoint_epochs_planned"] = len(epochs)
    for epoch in epochs:
        headers = {w_name: header}
        if ordered:
            headers[w_negative_name] = negative_header
        fit_writer = None
        w_reader = None
        source_accounted = False
        source_closed = False

        def close_source_reader():
            nonlocal source_accounted, source_closed
            if w_reader is None or source_closed:
                return
            if not source_accounted:
                report["sample_h5d_reads"] += w_reader.h5d_reads
                source_accounted = True
            t_source_lifecycle = time.perf_counter()
            w_reader.close()
            report["seconds"]["source_close"] += (
                time.perf_counter() - t_source_lifecycle)
            report["sample_source_closes"] += 1
            source_closed = True

        try:
            # FitWriter authenticates its ledger through serial h5py before
            # opening the output payload.  Finish that transfer before a
            # collective source handle exists, then close in reverse order.
            t_write_lifecycle = time.perf_counter()
            fit_writer = mpa_store.FitWriter(fit_dest, mesh_xy=mesh_xy)
            report["seconds"]["write"] += (
                time.perf_counter() - t_write_lifecycle)
            t_source_lifecycle = time.perf_counter()
            w_reader = mpa_store.open_w_column_reader(
                w_src, mesh_xy=mesh_xy, headers=headers)
            report["seconds"]["source_open"] += (
                time.perf_counter() - t_source_lifecycle)
            report["sample_source_opens"] += 1
            for q, lo, hi in epoch:
                stats = fit_one_block(
                    w_src, w_name, fit_dest, q, np.arange(lo, hi), z, n,
                    mesh_xy=mesh_xy, n_cols_buffer=plan["n_cols"],
                    w_negative_name=w_negative_name,
                    negative_header=negative_header,
                    tile_bytes=plan["tile_bytes"],
                    fit_tile_bytes=(
                        compute_plan["tile_bytes"]
                        if fit_tile_bytes is None else fit_tile_bytes),
                    rcond=rcond,
                    solve=solve, header=header, fit_writer=fit_writer,
                    w_reader=w_reader)
                ledger = stats["ledger"]
                report["blocks_walked"] += 1
                report["columns_read"] += stats["n_cols"]
                report["elements_fitted"] += stats["n_elements"]
                report["bytes_read"] += stats["bytes_read"]
                report["peak_block_bytes"] = max(
                    report["peak_block_bytes"], stats["bytes_read"])
                report["fit_dispatches"] += stats["fit_dispatches"]
                report["pade_solves"] += stats["pade_solves"]
                report["seconds"]["read"] += stats["seconds_read"]
                report["seconds"]["fit"] += stats["seconds_fit"]
                report["seconds"]["write"] += stats["seconds_write"]
            close_source_reader()
        except BaseException:
            # Both constructors and every block phase land here.  Source
            # closes first; only then may the output payload close/abort and
            # perform its serial-ledger ownership transfer.
            try:
                close_source_reader()
            finally:
                if fit_writer is not None:
                    fit_writer.close(commit=False)
            raise
        # The source reader is closed on every rank before FitWriter closes
        # its payload and rank zero publishes this epoch's completion ledger.
        t_write_lifecycle = time.perf_counter()
        ledger = fit_writer.close()
        report["seconds"]["write"] += time.perf_counter() - t_write_lifecycle
        report["checkpoint_epochs_committed"] += 1

    t_fin = time.perf_counter()
    if not bool(np.asarray(ledger["blocks_done"]).all()):
        raise ValueError(
            "run_fit_driver: refusing collective finalize because the fit "
            "ledger still has unfinished columns")
    if finalize and process_rank() == 0:
        mpa_store.finalize_fit_store(
            fit_dest, certification=certification)
    barrier("mpa_fit_finalized" if finalize else "mpa_fit_body_complete")
    ledger = mpa_store.fit_completion_ledger(fit_dest)
    if not finalize:
        # Every rank just performed a serial-h5py ledger read.  Transfer
        # ownership after the slowest reader has CLOSED, before rank zero
        # opens the same inode to commit the receipt.
        barrier("mpa_fit_body_ledger_readers_closed")
    report["seconds"]["finalize"] = time.perf_counter() - t_fin

    report["logical_outputs"] = (
        (3 if ordered else 2) * n * report["elements_fitted"])
    report["bytes_budget_total"] = (report["blocks_walked"]
                                    * report["tile_bytes"])
    report["condition_max"] = ledger["condition_max"]
    report["backward_error_max"] = ledger["backward_error_max"]
    report["seconds"]["total"] = time.perf_counter() - t_total

    # Production leaves root COMPLETE for the later scalar-head transaction.
    # Persist this successful body attempt now, while metadata is still
    # legally writable.  A ready incumbent is authenticated and preserved,
    # so a head-crash restart with no pending blocks cannot replace the real
    # work receipt with its zero-new-work pass.
    if not finalize:
        if process_rank() == 0:
            mpa_store.write_fit_io_receipt(fit_dest, report)
        barrier("mpa_fit_io_receipt_committed")

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
        f"{r['n_cols_budget']} I/O columns per block",
        f"  compute batch   {r.get('fit_n_cols_budget', r['n_cols_budget'])} "
        f"columns per fit dispatch",
        f"  resume          {r['blocks_skipped']} blocks skipped; "
        f"checkpoint epoch={r['checkpoint_blocks']} blocks, "
        f"{r['checkpoint_epochs_committed']}/"
        f"{r['checkpoint_epochs_planned']} new epochs committed",
        f"  sample I/O      {r['sample_source_opens']} source opens, "
        f"{r['sample_source_closes']} closes, "
        f"{r['sample_h5d_reads']} H5Dreads",
        f"  columns read    {r['columns_read']} newly fitted "
        f"(full store = {r['n_q']} q x {r['n_mu']} columns)",
        f"  elements fitted {r['elements_fitted']}",
        f"  logical outputs {r['logical_outputs']} "
        + ("(= 3*n_p per element: n_p Omega + n_p B + n_p D)"
           if r.get("ordered_residues", False)
           else "(= 2*n_p per element: n_p Omega + n_p B)"),
        f"  dispatches      {r['fit_dispatches']} fit, vmapped",
        f"  pole solves     {r['pade_solves']} "
        f"(1 per element; diagnostics reuse the same solve)",
        f"  bytes read      {r['bytes_read']} B "
        f"({r['bytes_read'] / mib:.2f} MiB) against a budget of "
        f"{r['bytes_budget_total']} B "
        f"({r['bytes_budget_total'] / mib:.2f} MiB)",
        f"  peak block      {r['peak_block_bytes']} B "
        f"({r['peak_block_bytes'] / mib:.3f} MiB) against the collective "
        f"I/O budget of {r['tile_bytes']} B "
        f"({r['tile_bytes'] / mib:.3f} MiB), row-sharded over "
        f"{r.get('io_processes', 1)} processes",
        f"  fit tile        {r.get('fit_tile_bytes', r['tile_bytes'])} B "
        f"({r.get('fit_tile_bytes', r['tile_bytes']) / mib:.3f} MiB)",
        f"  budget rule     {r['cost_sentence']}",
        f"  worst condition {r['condition_max']:.3e}, worst backward "
        f"error {r['backward_error_max']:.3e}",
        f"  seconds         source open {sec['source_open']:.3f}  read "
        f"{sec['read']:.3f}  source close {sec['source_close']:.3f}  fit "
        f"{sec['fit']:.3f}  write {sec['write']:.3f}  finalize "
        f"{sec['finalize']:.3f}  total {sec['total']:.3f}",
        "-" * 60,
        "",
    ]
    return "\n".join(lines)
