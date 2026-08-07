"""Microbench: the psi(G) read, per deck x window x path, on a real mesh.

RECORD, DO NOT OPTIMIZE.  Nothing here is a test and nothing here has a
threshold -- a bench cell that fails a run is a slow test wearing a
different name, and the charter is explicit that the default suite stays
seconds-fast and that perf lands as recorded baseline files.  ``bench`` is
in ``norecursedirs`` (``pyproject.toml``), so pytest never collects this.

    lx run --cpu -N 1 -n 4 python3 \\
        services/wfn_loader/bench/bench_wfn_loader.py --mesh 2x2 --tag cpu2x2
    lx run -N 1 -G 4 -n 4 python3 \\
        services/wfn_loader/bench/bench_wfn_loader.py --mesh 2x2 --tag gpu2x2
    # ...and the production deck, which is not checked in:
    lx run --cpu -N 1 -n 4 python3 \\
        services/wfn_loader/bench/bench_wfn_loader.py --mesh 2x2 --tag cpu2x2 \\
        --deck mos2_400b --wfn /pscratch/.../mos2_12x12_400b/WFN.h5

One JSON row per (deck, window, path) into ``baselines/``, the claims-style
shape the charter names.  Regression detection is DIFFING BASELINE FILES
ACROSS BRANCHES, not asserting a number in a test: a threshold that has to
hold on a shared machine either gets loosened until it means nothing or
fails on somebody else's contention.

THE THREE PATHS, and why all three have rows.

``door``      -- ``SlabIO.read_slabs``: n windows of one slab shape in ONE
                 collective H5Dread.  This is what the loader does.
``n_read_slab`` -- the SAME windows as n separate ``SlabIO.read_slab``
                 calls.  **This shape is REJECTED** (DESIGN DECISION 1),
                 and its rows are kept precisely because they are the
                 ruling's evidence.  A baseline table that records only
                 the shape that won cannot answer "was this worth it" the
                 next time somebody proposes the loop -- and somebody
                 will, because the loop is the obvious way to write it.
``load``      -- ``WfnLoader.load`` end to end, per backend, so the read
                 numbers above can be read as a FRACTION of the thing a
                 driver actually waits for.  ``--backend eager`` is the
                 comparison arm: the two are byte-identical by contract,
                 so the only difference a row can show is time.

WHAT IS TIMED, precisely.  The FIRST call of every row is a compile + cold
read and is recorded SEPARATELY as ``cold_s`` -- "how long does the first
one take" is a real question with a different answer, and conflating it
with the warm number is how a caching regression goes unnoticed.  Then
``warmup`` calls settle the executable cache, then ``reps`` timed calls,
each ``block_until_ready``-ed.  Median, min and max are all recorded.  The
baseline files quote **warm-min** as the headline because that is the
number least contaminated by a shared machine's tails, and because it is
the number the step-0 fold measurement quoted -- comparing a median
against a min is how the draft of the docs page got a row wrong.

MB/s IS COMPUTED FROM THE BYTES THE REQUEST ACTUALLY NAMES, not from the
padded slab: ``sum(prod(valid_shapes[i])) * itemsize`` over the windows.
Rating a read by its padding would make a raggeder deck look faster.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_BENCH = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_BENCH))
_REPO = os.path.dirname(_SERVICES)
for _p in (os.path.join(_REPO, "src"),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
for _svc in ("lxkit", "wfn_loader"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from lxkit.gate import platform_from_env
        import jax
        if platform_from_env() == "CUDA":
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()

#: ``deck -> (relative path, nrk, mnband, ngkmax, ngkmin)``.  SINGLE SOURCE
#: is ``services/wfn_loader/tests/conftest.py::DECKS`` -- these rows are
#: copied from it and ``--wfn`` overrides the path.  gnppm is the hostile
#: deck the whole L-c tier is written against: mnband 82 (82 % 4 == 2) and
#: ragged ngk 1917-1963, so neither the band axis nor the G axis divides.
#: ``mos2_400b`` is the PRODUCTION-scale deck and is NOT checked in (15.6
#: GB); it needs ``--wfn``.
_DECKS = {
    "gnppm":     ("tests/regression/gnppm_debug/WFN.h5", 9, 82, 1963, 1917),
    "mos2_400b": (None, 144, 400, None, None),
}

#: ``deck -> [(b_lo, b_hi, label)]``.  The hostile window is first on
#: purpose: ``(0, 10)`` at world 4 gives per-rank clamped band counts
#: [3,3,3,1], which is the geometry the band-pad bug lived in for months.
_WINDOWS = {
    "gnppm":     [(0, 10, "hostile"), (0, 82, "full")],
    "mos2_400b": [(0, 400, "full")],
}


def _mesh_from(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _time_one(fn, *, warmup, reps):
    """``(cold_s, [timed seconds])``.  The first call is compile + cold.

    REFUSES to time anything ``block_until_ready`` cannot wait on, and
    that refusal is the difference between a baseline and a fiction.  The
    sibling service learned this the expensive way: a route that returned
    a library HANDLE rather than an array came back at 0.032 ms, flat
    across every problem size, and flatness was the only tell.  A number
    that does not move with the problem is not a measurement of it.
    """
    import jax
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    cold_s = time.perf_counter() - t0
    if not hasattr(out, "shape"):
        raise TypeError(
            f"refusing to time a {type(out).__name__}: block_until_ready "
            f"cannot wait on it, so the elapsed time would be dispatch "
            f"overhead wearing the name of a collective read")
    del out
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return cold_s, ts


def _window_tables(loader, b_lo, b_hi, *, unfold=False):
    """The offsets / valid_shapes the loader itself would build.

    Derived by asking the loader for its own k-plan rather than
    re-deriving it here: a bench that re-implements the request measures a
    request nothing makes.  This is the ONE place the bench reaches past
    the public door, and it does so to stay honest about what it is
    timing -- flagged, not hidden.
    """
    k_idxs, _unfold_from_spec = loader._resolve_k("ibz")
    ibz_unique_sorted, n_reads, _pos, _n_k = loader._kplan(k_idxs, unfold)
    ns = int(loader.nspinor)
    band_extent = min(b_hi, int(loader.nbands))
    offsets = np.stack([
        [b_lo, 0, int(loader.kpt_starts[i]), 0] for i in ibz_unique_sorted
    ], axis=0).astype(np.int64)
    valid_shapes = np.stack([
        [band_extent - b_lo, ns, int(loader.ngk[i]), 2]
        for i in ibz_unique_sorted
    ], axis=0).astype(np.int64)
    return offsets, valid_shapes, n_reads, ns


def _bytes_named(valid_shapes, itemsize=8):
    """Bytes the REQUEST names -- the logical extents, not the padding."""
    return int(sum(int(np.prod(v)) for v in valid_shapes)) * itemsize


def _read_rows(WfnLoader, path, mesh, common, b_lo, b_hi, *, paths,
               warmup, reps):
    """The ``door`` and ``n_read_slab`` rows for one window.

    Both go through ONE open handle, so a difference between them is the
    request SHAPE and nothing else -- not two opens, not two contexts, not
    two stripe layouts.  That is the same discipline the step-0 fold
    measurement ran under ("both arms through the SAME PhdfCtx"), and it is
    the only way the comparison means what it says.
    """
    from jax.sharding import PartitionSpec as P
    rows = []
    # A machine with no phdf5-capable .so REFUSES -- at the constructor if
    # the auto-pick got there, at the collective OPEN otherwise, because
    # the door probes when it opens and not before.  Either way it is a
    # fact about the machine, so it is a ROW: a crash here would cost the
    # run its `load` rows too and leave a silent gap in the baseline file
    # exactly where the interesting platform was.
    loader = None
    try:
        loader = WfnLoader(path, mesh=mesh, backend="phdf5")
        off, vs, n_reads, ns = _window_tables(loader, b_lo, b_hi)
        nb = b_hi - b_lo
        ngkmax = int(loader.ngkmax)
        io = loader._ensure_slab_io()
        mb = _bytes_named(vs) / 1e6
        spec = P(("x", "y"), None, None, None)

        if "door" in paths:
            def _door():
                return io.read_slabs(
                    "wfns/coeffs", shape=(nb, ns, ngkmax, 2),
                    offsets=off, valid_shapes=vs, partition_spec=spec,
                    window_axis=2, dtype=np.float64)
            rows.append(_row(common, "door", "SlabIO.read_slabs",
                             n_reads, mb, _door, warmup=warmup, reps=reps))

        if "n_read_slab" in paths:
            import jax
            def _loop():
                outs = [
                    io.read_slab(
                        "wfns/coeffs", shape=(nb, ns, ngkmax, 2),
                        offset=tuple(int(x) for x in off[i]),
                        valid_shape=tuple(int(x) for x in vs[i]),
                        partition_spec=spec, dtype=np.float64)
                    for i in range(n_reads)]
                return jax.numpy.stack(outs, axis=2)
            rows.append(_row(
                common, "n_read_slab",
                f"{n_reads} x SlabIO.read_slab + stack", n_reads, mb,
                _loop, warmup=warmup, reps=reps,
                rejected="DESIGN DECISION 1: measured slower on every deck; "
                         "kept as the ruling's evidence"))
    except Exception as exc:                                   # noqa: BLE001
        rows.append(dict(common, path="door",
                         refused=" ".join(str(exc).split())[:240]))
    finally:
        if loader is not None:
            loader.close()
    return rows


def run(path, deck, mesh, *, windows, paths, backends, warmup, reps):
    import jax
    from jax.sharding import PartitionSpec as P
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    rows = []
    for (b_lo, b_hi, wlabel) in windows:
        common = dict(deck=deck, window=[b_lo, b_hi], window_label=wlabel,
                      mesh=f"{px}x{py}", nodes=int(
                          os.environ.get("SLURM_JOB_NUM_NODES", "1")))

        # ---- the two READ paths, both through the SAME open handle, so a
        # difference between them is the request shape and nothing else.
        if {"door", "n_read_slab"} & set(paths):
            rows.extend(_read_rows(WfnLoader, path, mesh, common, b_lo, b_hi,
                                   paths=paths, warmup=warmup, reps=reps))

        # ---- end to end, per backend.  Byte-identical by contract, so the
        # only thing these rows can differ in is time.
        if "load" in paths:
            for backend in backends:
                loader = WfnLoader(path, mesh=mesh, backend=backend)
                try:
                    def _load(_l=loader):
                        return _l.load(bands=(b_lo, b_hi), k="ibz")
                    rows.append(_row(
                        dict(common, backend=backend), "load",
                        f"WfnLoader.load(bands=({b_lo},{b_hi}), k='ibz')",
                        None, None, _load, warmup=warmup, reps=reps))
                finally:
                    loader.close()

    return rows


def _row(common, path, call, n_windows, mb, fn, *, warmup, reps,
         rejected=None):
    import jax
    row = dict(common, path=path, call=call, reps=reps, warmup=warmup)
    if n_windows is not None:
        row["n_windows"] = int(n_windows)
    if mb is not None:
        row["mb_named"] = round(mb, 3)
    if rejected:
        row["rejected"] = rejected
    try:
        cold_s, ts = _time_one(fn, warmup=warmup, reps=reps)
    except Exception as exc:                                   # noqa: BLE001
        # RECORDED, not dropped: "no row" and "row failed" are different
        # facts, and a baseline file that omits the second is unreadable
        # later.
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:240]}"
        if jax.process_index() == 0:
            print(f"ERROR {path} {common.get('window')}: "
                  f"{type(exc).__name__}", flush=True)
        return row
    row.update(cold_s=cold_s, warm_s=statistics.median(ts),
               warm_min_s=min(ts), warm_max_s=max(ts), warm_reps_s=ts)
    if mb is not None:
        row["mbps_warm_min"] = round(mb / min(ts), 1)
    if jax.process_index() == 0:
        rate = f"  {row.get('mbps_warm_min', '')} MB/s" if mb else ""
        print(f"{path:12s} {common['deck']:10s} {str(common['window']):10s} "
              f"cold {cold_s:8.4f} s  warm-min {min(ts):8.4f} s{rate}",
              flush=True)
    return row


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--tag", required=True,
                    help="baseline file name stem, e.g. cpu2x2")
    ap.add_argument("--deck", default="gnppm", choices=sorted(_DECKS))
    ap.add_argument("--wfn", default="",
                    help="override the deck's path (required for decks that "
                         "are not checked in)")
    ap.add_argument("--paths", default="door,n_read_slab,load")
    ap.add_argument("--backends", default="phdf5,eager")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing baseline file instead of "
                         "replacing it (decks are measured on separate runs)")
    args = ap.parse_args()

    rel = _DECKS[args.deck][0]
    path = args.wfn or (os.path.join(_REPO, rel) if rel else "")
    if not path or not os.path.exists(path):
        print(f"deck {args.deck!r}: no readable WFN at {path!r}; pass --wfn",
              file=sys.stderr)
        return 2

    mesh = _mesh_from(args.mesh)
    rows = run(path, args.deck, mesh,
               windows=_WINDOWS[args.deck],
               paths=set(args.paths.split(",")),
               backends=[b for b in args.backends.split(",") if b],
               warmup=args.warmup, reps=args.reps)

    if jax.process_index() != 0:
        return 0
    out = os.path.join(_BENCH, "baselines", f"{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = dict(
        tag=args.tag,
        jobid=os.environ.get("SLURM_JOB_ID", os.environ.get("SLURM_JOBID", "")),
        nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        processes=jax.process_count(),
        devices=jax.device_count(),
        backend=jax.default_backend(),
        mesh=args.mesh,
        jax_version=jax.__version__,
        machine=os.environ.get("NERSC_HOST", os.environ.get("LX_MACHINE", "")),
        ffi_so=os.environ.get("LORRAX_FFI_SO", ""),
        ffi_host_so=os.environ.get("LORRAX_FFI_HOST_SO", ""),
        recorded=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        rows=rows,
    )
    if args.merge and os.path.exists(out):
        with open(out) as fh:
            old = json.load(fh)
        keep = [r for r in old.get("rows", []) if r.get("deck") != args.deck]
        doc["rows"] = keep + rows
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    timed = sum(1 for r in rows if "warm_min_s" in r)
    print(f"wrote {out}: {len(rows)} rows, {timed} timed "
          f"({os.path.getsize(out)} bytes)", flush=True)
    # Judge by artifacts: a baseline file with no timed rows is not a
    # measurement, and saying so here is cheaper than discovering it in a
    # diff three steps later.
    return 0 if timed else 1


if __name__ == "__main__":
    sys.exit(main())
