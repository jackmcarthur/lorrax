"""The MPA fit stage's driver: read a column block, fit it, write it, finalize.

STAGING LOCATION: see ``gw/mpa/__init__.py``.  This is the module that
touches every other half of the multipole-W infrastructure — the
schedule (``tiling``), the bytes (``file_io.mpa_store``), the sample
grid (``sampling``) and the fit itself (``pade_fit``, ``diagnostics``)
— and it exists because none of them can demonstrate the pipeline
alone.  The fit kernel proves it recovers planted poles from samples in
memory; the store proves it refuses a half-written file.  Neither
proves that a fit whose samples came off disk in ``n_cols``-wide blocks
and whose poles went back to disk block by block reconstructs the field
that was planted.  That claim spans both, so it is tested here.

THE LOOP, WHICH IS THE WHOLE MODULE
-----------------------------------
``plan_column_walk`` -> for each ``(q, column block)``: ``read_w_columns``
(budgeted) -> ``fit_mpa_poles_batched`` (vmapped over the block's
elements) -> ``write_fit_block`` (staged, as the block completes) ->
``finalize_fit_store``.  Nothing accumulates across blocks except the
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

import contextlib
import time

import numpy as np

from file_io import mpa_store
from gw.mpa import diagnostics, pade_fit, tiling

__all__ = [
    "accumulate_over_pole_passes",
    "fit_one_block",
    "format_cost_report",
    "pole_pass_order",
    "read_pole_slice",
    "run_fit_driver",
]

#: The two diagnostics ``mpa_store.write_fit_block`` REQUIRES, plus the
#: two this driver adds.  Fixed here rather than assembled per block
#: because the store stamps the key set on the first block and refuses a
#: later block that reports a different one — a quantity measured for
#: some blocks and not others reads back as zero for the rest, and a
#: zero condition number is a perfectly conditioned solve.
_DIAGNOSTIC_KEYS = ("condition", "backward_error", "residual", "n_valid")


def _elements_from_block(block):
    """``(n_omega, N_mu, n_cols)`` -> ``(N_mu * n_cols, n_omega)``.

    The fit kernel takes one element's sample vector per row and is
    vmapped over the leading axis, so the frequency axis — outermost on
    disk, because the fit reads all of omega for a few columns — becomes
    the innermost one in the solve.  The element index is
    ``mu * n_cols + col``, which is what :func:`_block_from_elements`
    inverts; the two are written next to each other so the ordering has
    one definition and not two.
    """
    n_omega, n_mu, n_cols = block.shape
    return np.transpose(block, (1, 2, 0)).reshape(n_mu * n_cols, n_omega)


def _block_from_elements(arr, n_mu, n_cols):
    """``(n_elements, n_p)`` -> ``(n_p, N_mu, n_cols)``, the write shape.

    The pole axis leads on disk because the Sigma stage consumes
    ``W(tau) = sum_p B_p exp(-i Omega_p tau)`` with ``p`` outermost —
    one pass per pole, one contiguous slab per pass.
    """
    a = np.asarray(arr)
    return np.transpose(a, (1, 0)).reshape(a.shape[1], n_mu, n_cols)


def fit_one_block(
    w_src,
    w_name,
    fit_dest,
    q,
    mu_cols,
    z_samples,
    n_p,
    *,
    tile_bytes=None,
    guards=None,
    rcond=1.0e-13,
    out_spec=None,
    solve=pade_fit.SOLVE_MODES[0],
    affine=True,
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
    out_spec
        Optional row-axis sharding spec for the read block, checked (not
        applied) by the reader.  ``tiling.row_shard_spec()`` is the only
        shape it accepts.

    Returns
    -------
    dict
        ``ledger`` (the store's completion ledger after this block) plus
        the block's counters: columns, elements, bytes read, dispatches
        and the per-stage seconds.  :func:`run_fit_driver` sums these
        into the run's cost report.

    Notes
    -----
    THE BACKWARD ERROR COSTS A SECOND DISPATCH, and it is not optional.
    ``mpa_store.write_fit_block`` REQUIRES ``condition`` and
    ``backward_error``: the Sigma stage's certification refuses poles
    that fail its gates, and a pole whose conditioning nobody recorded
    can only be trusted, never refused.  But ``pade_fit.fit_mpa_poles``
    returns ``cond_pade`` and no backward error — the backward error
    lives in ``diagnostics.solve_conditioning``, which re-solves the
    system AND re-runs the whole fit internally to report its forward
    residual.  So one logical block costs TWO complete fits per element
    and THREE solves of the Pade-in-z^2 system: one in the fit, one
    standalone in ``solve_conditioning``, and one more inside the fit
    that function runs for itself.  The cost report states the
    dispatches, the fits and the Pade solves separately rather than
    quoting whichever is smallest; see :func:`format_cost_report`.
    """
    n = int(n_p)
    t_read = time.perf_counter()
    block = mpa_store.read_w_columns(
        w_src, w_name, q, mu_cols, tile_bytes=tile_bytes,
        out_spec=out_spec)
    t_read = time.perf_counter() - t_read

    n_omega, n_mu, n_cols = block.shape
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

    tile = _elements_from_block(block)

    t_fit = time.perf_counter()
    Omega, B, diag = pade_fit.fit_mpa_poles_batched(
        tile, z, n, guards=guards, rcond=rcond, solve=solve, affine=affine)
    cond_diag = diagnostics.diagnostics_batched(
        diagnostics.solve_conditioning, tile, z, n, rcond=rcond,
        solve=solve, affine=affine)
    Omega = np.asarray(Omega)
    B = np.asarray(B)
    t_fit = time.perf_counter() - t_fit

    diag_block = {
        "condition": np.asarray(cond_diag["cond"],
                                dtype=np.float64).reshape(n_mu, n_cols),
        "backward_error": np.asarray(
            cond_diag["backward_error"],
            dtype=np.float64).reshape(n_mu, n_cols),
        "residual": np.asarray(
            diag["max_abs_residual"],
            dtype=np.float64).reshape(n_mu, n_cols),
        "n_valid": np.asarray(
            diag["n_valid"], dtype=np.float64).reshape(n_mu, n_cols),
    }

    t_write = time.perf_counter()
    ledger = mpa_store.write_fit_block(
        fit_dest, q, mu_cols,
        _block_from_elements(Omega, n_mu, n_cols),
        _block_from_elements(B, n_mu, n_cols),
        diag_block)
    t_write = time.perf_counter() - t_write

    return {
        "ledger": ledger,
        "q": int(q),
        "n_cols": int(n_cols),
        "n_elements": int(n_mu * n_cols),
        "bytes_read": int(block.size) * mpa_store.COMPLEX128_BYTES,
        "fit_dispatches": 1,
        "diagnostic_dispatches": 1,
        "full_fits": 2 * int(n_mu * n_cols),
        "pade_solves": 3 * int(n_mu * n_cols),
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
    tile_bytes=None,
    guards=None,
    rcond=1.0e-13,
    certification=None,
    provenance=None,
    report_stream=None,
    solve=pade_fit.SOLVE_MODES[0],
    affine=True,
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
    if solve not in pade_fit.SOLVE_MODES:
        raise ValueError(
            f"GATE fit_solve_mode_known: solve={solve!r} is not one of "
            f"{pade_fit.SOLVE_MODES}.  FALSE case: the deck names a "
            "denominator algebra the fit kernel implements.  This is a "
            "REFUSAL and not a fallback on purpose: the whole hazard the "
            "stamp below exists to close is a fit store that cannot say "
            "which algebra made it, and silently substituting the default "
            "for a typo is how that store gets written.")
    n = int(n_p)
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

    # WHICH SCREENING OBJECT THIS IS, asked before a byte is allocated.
    # The multipole method fits W_c = W - v and nothing else; a store
    # filled from the Dyson solve holds W, its poles are dominated by the
    # bare Coulomb interaction, and NOTHING downstream can tell — the fit
    # reproduces whatever it is handed (this deck's backward error is
    # 4e-16), the tensor stays Hermitian and the k-star relation survives,
    # because v_q is symmetry-covariant.  The 2026-08-09 n_p = 1 bridge
    # measured what that costs: Sigma_c = -130.651 eV against Godby-Needs
    # +0.6754 eV, with |v| at 104-119 % of |W| at the probe frequency.
    # So it is a declaration the producer makes and this driver refuses
    # on, at the one seam that reads the W file and writes the pole file.
    content = mpa_store.require_correlation_part(
        header.get("screening_content"),
        where="run_fit_driver", source=f"{w_src} :: {w_name}")
    plan = tiling.plan_column_walk(n_mu, n_omega, tile_bytes)
    mpa_store.allocate_fit_store(
        fit_dest, n_q=n_q, n_mu=n_mu, n_p=n,
        # CARRIED, NOT RE-DERIVED.  The Sigma stage refuses a pole field
        # that was fitted to the full W, and by the time it runs the W
        # file may be gone; the fit store therefore says for itself what
        # it was fitted to, and this driver is the only thing that can
        # know.
        screening_content=content,
        # THE POLE AXIS INHERITS THE ABSCISSAE'S UNIT.  The Pade solve
        # returns Omega_p in whatever unit the z samples were stated in
        # and B_p carries one power of it, and the z samples are the W
        # store's own stamped grid (asserted bit-equal above) -- so the
        # unit is READ OFF THAT STAMP, never typed by a human.  This is
        # the declaration the Sigma-side readers convert on; the
        # first-light field was fitted on a Hartree grid and read as
        # Rydberg, halving every pole with no symptom, which is the
        # defect this line retires at the only seam that knows the
        # answer for free.
        energy_unit=header["omega_units"],
        grid_hash=header["grid_hash"],
        table_hash=header["table_hash"],
        centroid_hash=header["centroid_hash"],
        provenance=dict(provenance or {},
                        solve_mode=str(solve),
                        solve_affine=bool(affine),
                        solve_rcond=float(rcond)))

    report = {
        "solve": str(solve),
        "affine": bool(affine),
        "rcond": float(rcond),
        "screening_content": content,
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

    spec = tiling.row_shard_spec()
    ledger = None
    for q, lo, hi in tiling.fit_schedule(n_q, n_mu, n_omega, tile_bytes):
        stats = fit_one_block(
            w_src, w_name, fit_dest, q, np.arange(lo, hi), z, n,
            tile_bytes=tile_bytes, guards=guards, rcond=rcond,
            out_spec=spec, solve=solve, affine=affine)
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
    ledger = mpa_store.finalize_fit_store(
        fit_dest, certification=certification)
    report["seconds"]["finalize"] = time.perf_counter() - t_fin

    report["logical_outputs"] = 2 * n * report["elements_fitted"]
    report["bytes_budget_total"] = (report["blocks_walked"]
                                    * report["tile_bytes"])
    report["condition_max"] = ledger["condition_max"]
    report["backward_error_max"] = ledger["backward_error_max"]
    report["seconds"]["total"] = time.perf_counter() - t_total

    if report_stream is not None:
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

    SO THREE COUNTS ARE PRINTED, NOT ONE.  ``dispatches`` is what the
    scheduler sees: two vmapped launches per block.  ``full fits`` is
    what the elements see: TWO per element, because the store requires
    a backward error, ``pade_fit.fit_mpa_poles`` does not return one,
    and the only supplier — ``diagnostics.solve_conditioning`` — runs a
    complete fit of its own to report a forward residual beside it.
    ``Pade solves`` is what the linear algebra sees: THREE per element,
    the two fits' plus the standalone equilibrated solve
    ``solve_conditioning`` does before them.

    None of that factor is this driver's doing; it is a property of the
    seam between the fit kernel and the staged store, and it is printed
    rather than absorbed because the design review is where it can be
    removed — a ``backward_error`` returned from ``fit_mpa_poles``
    beside the ``cond_pade`` it already computes would collapse all
    three counts to one dispatch, one fit and one solve.
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
        # THE SOLVE, NAMED, IN THE LOG.  The kernel gained a second
        # denominator algebra on 2026-08-10 and production reaches it
        # through a DEFAULT, so without this line an A/B arm's log is not
        # evidence of which arm it was -- and the standing spec gap is
        # that an unknown deck key logs and proceeds, which means a typo
        # in the deck reads as the default everywhere except here.
        f"  solve           {r['solve']}"
        + (f" (affine={r['affine']})" if r['solve'] == 'pade' else "")
        + f", rcond={r['rcond']:.1e}"
        + "  [stamped as prov_solve_mode / prov_solve_affine /"
        " prov_solve_rcond]",
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
        f"(2 per element: the fit, and the one solve_conditioning "
        f"runs to report a forward residual)",
        f"  Pade solves     {r['pade_solves']} "
        f"(3 per element: the two fits', plus solve_conditioning's own "
        f"equilibrated solve)",
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


@contextlib.contextmanager
def _as_group(src, mode="r"):
    """An open h5py group, from either a path or an already-open one.

    A three-line mirror of ``symmetry_maps.QirrDest``, kept for ONE
    remaining reason after the pole-sliced reader moved into
    ``mpa_store``: :func:`accumulate_over_pole_passes` opens the store
    once and reads ``n_p`` slabs through the same handle, so the pass
    loop pays one open rather than fourteen.  ``mpa_store``'s readers all
    accept an already-open group, which is what makes that possible; this
    is the object that produces one.
    """
    if hasattr(src, "attrs"):
        yield src
        return
    import h5py
    with h5py.File(str(src), mode) as handle:
        yield handle


def read_pole_slice(fit_src, p, *, allow_partial=False):
    """``(Omega_p, B_p)`` for ONE pole -- ``(n_q, N_mu, N_mu)`` each.

    THE STOPGAP IS OVER: this is now
    :func:`file_io.mpa_store.read_pole_slice`, which is where the
    docstring this replaces said it belonged -- beside the other two
    readers, sharing their ledger refusal instead of restating it.  The
    name stays exported here because the pass loop below is its caller
    and the design review named it at this address.
    """
    return mpa_store.read_pole_slice(
        fit_src, p, allow_partial=allow_partial)


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
    with _as_group(fit_src) as grp:
        ledger = mpa_store.fit_completion_ledger(grp)
        n_p = ledger["n_p"]
        walk = (pole_pass_order(n_p) if order is None
                else tuple(int(p) for p in order))

        total = None
        passes = []
        for p in walk:
            Omega_p, B_p = read_pole_slice(
                grp, p, allow_partial=allow_partial)
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
