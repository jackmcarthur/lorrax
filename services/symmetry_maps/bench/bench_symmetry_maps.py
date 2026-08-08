"""Microbench: op x backend x shape, at 1x1 and at an emulated 2x2.

RECORD, DO NOT OPTIMIZE.  Nothing here is a test and nothing here has a
threshold — a bench cell that fails a run is a slow test wearing a
different name, and the charter is explicit that the default suite stays
seconds-fast and that perf lands as recorded baseline files.  ``bench/``
is ``norecursedirs`` in this service's ``pyproject.toml`` (and in
``distrib_la``'s, which this file is modelled on), so pytest never
collects it.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 \\
        services/symmetry_maps/bench/bench_symmetry_maps.py \\
        --mesh 1x1 --tag wsl1x1 --machine wsl
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 \\
        services/symmetry_maps/bench/bench_symmetry_maps.py \\
        --mesh 2x2 --tag wsl2x2 --machine wsl

One JSON claims file per leg into ``baselines/``: ``(op, backend, shape,
mesh, seconds)`` per row, ``(nodes, jobid, machine, devices, jax_version,
recorded)`` per document.  That split is ``distrib_la``'s, not a new
invention — SERVICE_FORM §Perf names the seven fields and the mold puts
the four run-scoped ones where they belong, once.  Regression detection is
DIFFING BASELINE FILES ACROSS BRANCHES, not asserting a number in a test.

WHAT ``backend`` MEANS HERE.  ``distrib_la``'s backend axis is the linear
algebra route (scalapack / slate / cusolvermp / native).  This service has
no FFI at all; its one real dispatch axis is WHERE THE OPERAND LIVES, and
it is a genuine fork in the code — ``_take_rows``, ``_broadcast_rows`` and
``_star_stats`` each test ``isinstance(A, jax.Array)`` and run either a
numpy expression or a cached ``jax.jit``.  So ``backend`` is ``numpy``
(host) or ``jax`` (device), and the two rows for one op are the two
branches of that ``if``.

WHAT IS TIMED, precisely.  The FIRST call of every (op, backend, shape) is
a compile (or, on the numpy branch, a cold cache) and is recorded
separately as ``compile_seconds``; then ``warmup`` calls settle the
executable cache, then ``reps`` timed calls, each ``block_until_ready``-ed,
and the MEDIAN is recorded along with min and max.  Median rather than
mean because a laptop under WSL produces occasional long tails that a mean
folds into the number and hides.  ``compile_seconds`` is recorded
separately because "how long does the first call take" is a real question
with a different answer, and conflating them is how a plan-caching
regression goes unnoticed — ``symmetry_maps`` keeps a module-level
``_STAR_JIT_CACHE`` keyed on the index tables' BYTES, so a caller that
rebuilds its ``KStarMap`` every SC iteration pays ``compile_seconds`` every
iteration and the median row would never say so.

N = 25 (``--reps``), warmup 5, AND WHAT N CAN AND CANNOT FIX WAS MEASURED.
Within one burst of 25, the median settles: three back-to-back 1x1 legs
under a steady load gave ``SymMaps(si_cohsex_debug)`` 54.23 / 53.70 /
53.82 ms and ``star_select`` numpy 0.0128 / 0.0130 / 0.0129 ms, and the
18 MiB-allocating rows (the numpy spreads, ``star_broadcast`` on the
device) repeated to ±10%.  BETWEEN legs minutes apart the SAME median on
the SAME code moved 51.4 → 93.4 → 124.6 → 124.4 → 117.8 ms, tracking a
1-minute load average that ran 2.96–5.58 while other work shared the box
— and ``seconds_min`` moved with it (49.6 → 84.6 → 77.4 ms), so the min
is not a clean floor either.  That is a 2.4× band that no N closes,
because the variable is not sample size.

SIX CORES, NOT TWELVE, and that is why the count is a recorded field and
not a sentence.  ``os.cpu_count()`` and ``os.sched_getaffinity`` both
answer 6 inside the benching process, matching ``/proc/cpuinfo``, while
``nproc`` in an interactive shell on this same box answers 12 from a
different namespace.  A load average of 5.5 against 6 usable cores is
the box saturated, which is the reading that makes the drift above
make sense — and it is the reading a prose "12 cores" would have hidden.

Two consequences, both structural rather than advisory.  (1) The doc
header records ``cpu_count`` and ``load_average``, so a row that looks
like a regression can be checked against the machine it was taken on
before anyone goes looking for it in the code.  (2) A WSL baseline is
comparable to another WSL baseline taken the same way and to nothing else;
the Perlmutter legs, which run in a dedicated allocation, are the ones a
cross-branch diff should lean on.  This is also why every row carries
``seconds_min``/``seconds_max`` and why regression detection is a diff of
files and never a threshold: the recorded BAND is the honest object, and a
threshold on this box would have fired five times this afternoon without a
line of code changing.

THREE GROUPS, AND WHICH MESH EACH BELONGS TO.

``construct`` — ``SymMaps(deck)`` for all four in-tree decks, built from
    the tests' ``_deck_stub.read_deck`` so the bench and the acceptance
    gate read the same eleven header attributes the same way.  Host numpy
    throughout (``SymMaps`` builds its tables eagerly), so it is
    mesh-independent and runs on the 1x1 leg only.

``star`` — ``star_select`` / ``star_broadcast`` / ``star_spread`` /
    ``KStarMap.spread_rel`` on a production-shaped band-index operand:
    ``nk=64, nb=60`` complex128 = 18.4 MiB, the Si class.  The index
    tables are ``si_cohsex_debug``'s OWN (``nk_tot=64``, ``nkpts=8``,
    ``ntran=48``), read through the same stub, so the gather widths and
    the star sizes are a real deck's and not a shape someone picked.  Host
    and device rows both, at 1x1 — this is the pair the docs page's
    "one reduction and one 16-byte transfer against two full host
    readbacks" claim is about.  1x1 leg only: nothing in the star surface
    shards its own operand (``_row_out_sharding`` inherits whatever the
    caller had), so a 2x2 row would measure the caller's sharding choice
    and attribute it to this service.

``unfold`` — ``unfold_isdf_operator`` at whatever ``--mesh`` says.  This
    is the one
    group that is genuinely mesh-shaped: 1x1 takes neither ``all_to_all``
    branch and 2x2 takes both.  Runs on every leg.

THE 2x2 IS EMULATED, and that is stated in the row (``emulated: true``)
rather than left for a reader to infer from ``machine``.  Four host
devices come from ``--xla_force_host_platform_device_count=4``, set below
before the first jax import for the same reason the service conftest sets
it: XLA reads that flag once per process and never again.  An emulated
mesh is four threads over one memory, so its ``all_to_all`` is a memcpy
and its number is NOT a network measurement; what it does measure, and
measure honestly, is the redistribution's own arithmetic and dispatch.
The Perlmutter legs below are where the interconnect enters.

PERLMUTTER ROW SCHEMA (the orchestrator runs these legs separately; the
rows land in the same ``baselines/`` directory and diff against the WSL
ones field for field)::

    lx run --cpu -N 1 -n 1 python3 \\
        services/symmetry_maps/bench/bench_symmetry_maps.py \\
        --mesh 1x1 --tag pm_cpu1x1
    lx run --cpu -N 1 -n 4 python3 \\
        services/symmetry_maps/bench/bench_symmetry_maps.py \\
        --mesh 2x2 --tag pm_cpu2x2
    lx run -N 1 -G 4 -n 4 python3 \\
        services/symmetry_maps/bench/bench_symmetry_maps.py \\
        --mesh 2x2 --tag pm_gpu2x2

On those legs ``machine`` comes from ``NERSC_HOST`` (``perlmutter``),
``jobid`` from ``SLURM_JOB_ID`` and ``nodes`` from
``SLURM_JOB_NUM_NODES`` — all three default to the WSL answer
(``wsl`` / ``null`` / ``1``) and none is ever invented.  ``jobid`` is JSON
``null`` and not ``""`` on a machine with no batch system, because ``""``
is what an unset ``SLURM_JOB_ID`` inside a real job would also give and
the two facts are not the same.  ``emulated`` is FALSE on a real ``-n 4``
leg — the process count, not the flag, is what decides it — so a reader
diffing pm_cpu2x2 against wsl2x2 can see at a glance that only one of them
crossed a process boundary.
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
_SERVICE = os.path.dirname(_BENCH)
_SERVICES = os.path.dirname(_SERVICE)
for _svc in ("lxkit", "symmetry_maps"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)
# The service's own tests directory: ``_deck_stub`` is the header reader
# the acceptance gate uses and ``test_symmetry_maps_emulated_mesh`` owns
# the orbit-closed synthetic geometry.  Importing them rather than copying
# them is deliberate — a bench that re-typed the twelve seeds would drift
# from the cells that prove the geometry is orbit-closed and that its
# umklapp phase is non-trivial, and then measure a different problem while
# looking identical.
_SVC_TESTS = os.path.join(_SERVICE, "tests")
if _SVC_TESTS not in sys.path:
    sys.path.insert(0, _SVC_TESTS)

if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    # Before the first jax import, and only if the caller did not say
    # otherwise.  Copied, not shared, from the service conftest for the
    # reason its comment gives: the shared home would have to be lxkit.
    _flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _flags:
        _force = "--xla_force_host_platform_device_count=4"
        os.environ["XLA_FLAGS"] = (
            f"{_flags} {_force}".strip() if _flags else _force)
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        import jax
        jax.distributed.initialize()

import _deck_stub                                              # noqa: E402
import test_symmetry_maps_emulated_mesh as _lb                 # noqa: E402
from symmetry_maps import (KStarMap, SymMaps, star_broadcast,  # noqa: E402
                           star_select, star_spread,
                           unfold_isdf_operator)

#: The band-index operand every star row is measured on: ``(nk, nb, nb)``
#: complex128.  nk is ``si_cohsex_debug``'s full BZ and is therefore not a
#: free parameter; nb=60 is the Si class's band count.  18.4 MiB — large
#: enough that the gather is not pure dispatch, small enough that a WSL box
#: does the whole sweep in seconds.
_STAR_DECK = "si_cohsex_debug"
_STAR_NB = 60

#: A second orbit-closed centroid set, built by the L-b tests' OWN
#: ``_geometry`` so closure and the non-trivial umklapp wrap are properties
#: of that function and not of this file.  12 centroids (the L-b geometry)
#: is a latency measurement: at ``(5, 12, 12)`` complex128 the ``2x2``
#: all_to_all moves 1.1 KiB per rank and the row is dispatch. 120 keeps the
#: same two self-image seeds (so ``L_μ − L_ν`` is still not identically
#: zero — the trap the L-b docstring measured) and adds 59 pairs, giving
#: ``2 + 118 = 120``, divisible by 4.  Both are recorded: the small one
#: because it is the shape the correctness cells prove, the large one
#: because it is the shape the number means something at.
_SEEDS_120 = (((0, 0), (3, 6))
              + tuple((x, y) for y in range(1, 6) for x in range(12))[:59])


def _star_case():
    """``(kmap, irr, sym, nss, A_full_host, A_irr_host)`` for the Si class."""
    sym = SymMaps(_deck_stub.read_deck(_STAR_DECK))
    nss = len(sym.sym_mats_k) // 2
    kmap = KStarMap.from_sym(sym, nss)
    rng = np.random.default_rng(20260807)
    shape = (kmap.nk_full, _STAR_NB, _STAR_NB)
    A = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    A = A.astype(np.complex128)
    return kmap, sym.irr_idx_k, sym.sym_idx_k, nss, A, star_select(
        A, sym.irr_idx_k)[0]


def _geometry_120():
    """n_rmu = 120, orbit-closed, divisible by Px*Py for 1x1/2x2/4x1."""
    perm, L, n = _lb._geometry(_SEEDS_120)
    assert n == 120 and n % 4 == 0, f"expected 120 centroids, got {n}"
    return perm, L, n


def _time_one(fn, *, warmup, reps, blocking=True):
    """``(compile_seconds, [timed seconds])`` — first call is the compile.

    REFUSES to time an ASYNC result the timer will not wait on.  A jax
    call returns a future, so ``blocking=True`` rows must return something
    ``block_until_ready`` understands — a ``jax.Array``, a float, a numpy
    array, or a container of those.  ``blocking=False`` is the host-only
    route (``SymMaps.__init__`` returns an object, not an array) and its
    refusal is the MIRROR one: a ``jax.Array`` coming back from a call
    nobody blocks on would be dispatch overhead wearing the name of the
    work — the class of mistake that made ``distrib_la``'s cuSOLVERMp
    cholesky row read 0.032 ms flat across n = 64 and n = 1024.
    """
    import jax
    out = fn()
    if blocking and not isinstance(
            out, (float, int, np.ndarray, jax.Array, tuple, list)):
        raise TypeError(
            f"refusing to time a {type(out).__name__}: block_until_ready "
            f"cannot wait on it, so the elapsed time would be dispatch "
            f"overhead wearing the name of the work")
    if not blocking and isinstance(out, jax.Array):
        raise TypeError(
            "refusing to time a jax.Array without blocking: the call is "
            "asynchronous and the elapsed time would be dispatch overhead "
            "wearing the name of the work")
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        if blocking:
            jax.block_until_ready(r)
        ts.append(time.perf_counter() - t0)
    return compile_s, ts


def _mib(shape, itemsize=16):
    return float(np.prod(shape)) * itemsize / (1 << 20)


def _row(rows, op, backend, shape, mesh, compile_s, ts,
         dtype="complex128", **extra):
    row = dict(op=op, backend=backend, shape=list(shape), mesh=mesh,
               dtype=dtype, seconds=statistics.median(ts),
               seconds_min=min(ts), seconds_max=max(ts),
               compile_seconds=compile_s, reps=len(ts), **extra)
    rows.append(row)
    print(f"{op + '/' + backend:34s} {str(list(shape)):16s} {mesh:5s} -> "
          f"median {row['seconds'] * 1e3:9.4f} ms  "
          f"(min {min(ts) * 1e3:.4f} max {max(ts) * 1e3:.4f}, "
          f"compile {compile_s * 1e3:.2f} ms)", flush=True)
    return row


def run_construct(rows, *, warmup, reps):
    """``SymMaps(deck)`` per deck.  Host numpy; no mesh, no jax."""
    for deck in _deck_stub.DECKS:
        if not _deck_stub.deck_available(deck):
            # RECORDED, not skipped: "no fixture" and "no row" are
            # different facts and a baseline file that omits the first is
            # unreadable later.
            rows.append(dict(op="SymMaps.__init__", backend="numpy",
                             deck=deck, mesh="1x1",
                             refused="fixture absent: "
                                     + _deck_stub.deck_path(deck)))
            continue
        # The header read is OUTSIDE the timed call: it is h5py touching a
        # 14-52 MiB file's mf_header group, and folding it in would make
        # this a measurement of the filesystem's page cache.
        hdr = _deck_stub.read_deck(deck)
        sym = SymMaps(hdr)
        compile_s, ts = _time_one(lambda h=hdr: SymMaps(h),
                                  warmup=warmup, reps=reps, blocking=False)
        _row(rows, "SymMaps.__init__", "numpy",
             (int(sym.nk_tot), int(hdr.nkpts), int(hdr.ntran)), "1x1",
             compile_s, ts, dtype="int32/float64", deck=deck,
             shape_names=["nk_full", "nk_irr", "ntran"])


def run_star(rows, *, warmup, reps):
    """The four star entry points, host and device, at the Si shape."""
    import jax.numpy as jnp
    kmap, irr, sidx, nss, A_host, A_irr_host = _star_case()
    A_dev = jnp.asarray(A_host)
    A_irr_dev = jnp.asarray(A_irr_host)
    shape = A_host.shape
    ishape = A_irr_host.shape
    for backend, A, A_irr in (("numpy", A_host, A_irr_host),
                              ("jax", A_dev, A_irr_dev)):
        cases = (
            ("star_select", shape, ishape,
             lambda A=A: star_select(A, irr)),
            ("star_broadcast", ishape, shape,
             lambda A_irr=A_irr: star_broadcast(
                 A_irr, irr, sidx, nss, kmap.labels)),
            ("star_spread", shape, (),
             lambda A=A: star_spread(A, irr, sidx, nss)),
            ("KStarMap.spread_rel", shape, (),
             lambda A=A: kmap.spread_rel(A)),
        )
        for op, sh, out_sh, fn in cases:
            compile_s, ts = _time_one(fn, warmup=warmup, reps=reps)
            _row(rows, op, backend, sh, "1x1", compile_s, ts,
                 deck=_STAR_DECK, nk_full=kmap.nk_full,
                 nk_irr=kmap.nk_irr, n_sym_spatial=nss,
                 operand_mib=_mib(sh), out_shape=list(out_sh),
                 out_mib=_mib(out_sh))


def run_unfold(rows, mesh, mesh_str, *, warmup, reps, emulated):
    """``unfold_isdf_operator`` on the L-b geometry (n_rmu=12) and its
    120 twin."""
    import jax.numpy as jnp
    for name, geom in (("lb12", _lb._divisible_geometry),
                       ("orbit120", _geometry_120)):
        perm, L, n_rmu = geom()
        V = jnp.asarray(_lb._hermitian_ibz(n_rmu))

        def _fn(V=V, perm=perm, L=L):
            return unfold_isdf_operator(
                V, irr_idx=_lb._IRR, sym_idx=_lb._SYM,
                sym_perm=perm, L_table=L,
                q_irr_frac=_lb._Q_IRR, mesh_xy=mesh,
                n_sym_spatial=_lb._NTRAN)
        compile_s, ts = _time_one(_fn, warmup=warmup, reps=reps)
        out_sh = (len(_lb._IRR), n_rmu, n_rmu)
        # HAND-TRACKED THROUGH THE 2026-08-08 RENAME SWEEP — the ``op``
        # field is a BENCHMARK IDENTITY, not a symbol reference.  It is
        # the key every committed row in ``baselines/*.json`` is matched
        # on, and the whole value of those files is that a number from
        # today can be compared with a number from before the rename.
        # Renaming the key would not rename the history; it would only
        # make the history unmatchable.  The CALL above moved to the new
        # name; this string stays until a baseline re-cut retires it.
        _row(rows, "unfold_v_q", "jax", V.shape, mesh_str, compile_s, ts,
             geometry=name, n_q_ibz=int(V.shape[0]),
             operand_mib=_mib(V.shape), out_shape=list(out_sh),
             out_mib=_mib(out_sh), emulated=emulated)


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PxQ, e.g. 1x1 or 2x2")
    ap.add_argument("--tag", required=True,
                    help="baseline file name stem, e.g. wsl1x1")
    ap.add_argument("--machine", default=os.environ.get(
        "NERSC_HOST", os.environ.get("LX_MACHINE", "wsl")))
    ap.add_argument("--groups", default="",
                    help="comma-separated subset of construct,star,unfold; "
                         "default is the set that applies to --mesh")
    ap.add_argument("--reps", type=int, default=25,
                    help="timed calls per row; 25 was measured to settle "
                         "the median on this box (see the module note)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    from jax.sharding import Mesh
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh_str = f"{px}x{py}"
    have = jax.device_count()
    if have < px * py:
        # A LOUD refusal, not a silent 1x1: a bench that quietly ran the
        # wrong mesh and wrote the requested one into the row is a
        # baseline file that lies.
        print(f"REFUSED: --mesh {mesh_str} needs {px * py} devices, "
              f"this process has {have}.  On a laptop that means XLA_FLAGS "
              f"was already set without "
              f"--xla_force_host_platform_device_count; on a cluster it "
              f"means the launcher gave fewer ranks than the mesh.",
              file=sys.stderr, flush=True)
        return 2
    mesh = Mesh(np.asarray(jax.devices()[:px * py]).reshape(px, py),
                ("x", "y"))
    # EMULATED means one process pretending to be several.  Keyed on the
    # process count and not on XLA_FLAGS: a real ``-n 4`` leg sets no flag
    # and must not be labelled emulated, and a single-process run with the
    # flag must not be labelled real.
    emulated = px * py > 1 and jax.process_count() == 1

    default = ("construct", "star", "unfold") if px * py == 1 else ("unfold",)
    groups = tuple(g.strip() for g in args.groups.split(",")
                   if g.strip()) or default
    unknown = [g for g in groups if g not in
               ("construct", "star", "unfold")]
    if unknown:
        print(f"REFUSED: unknown group(s) {unknown}", file=sys.stderr)
        return 2

    rows = []
    for group in groups:
        try:
            if group == "construct":
                run_construct(rows, warmup=args.warmup, reps=args.reps)
            elif group == "star":
                run_star(rows, warmup=args.warmup, reps=args.reps)
            else:
                run_unfold(rows, mesh, mesh_str, warmup=args.warmup,
                           reps=args.reps, emulated=emulated)
        except Exception as exc:                               # noqa: BLE001
            # Recorded as an ERROR row rather than quietly dropped — see
            # distrib_la's bench, where the donated-buffer failures were
            # only findable because they were written down.
            rows.append(dict(op=group, backend="", mesh=mesh_str,
                             error=f"{type(exc).__name__}: "
                                   f"{' '.join(str(exc).split())[:240]}"))
            print(f"ERROR {group}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)

    if jax.process_index() != 0:
        return 0
    jobid = (os.environ.get("SLURM_JOB_ID")
             or os.environ.get("SLURM_JOBID") or None)
    doc = dict(
        tag=args.tag,
        jobid=jobid,
        nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        processes=jax.process_count(),
        devices=jax.device_count(),
        backend=jax.default_backend(),
        mesh=mesh_str,
        emulated=emulated,
        groups=list(groups),
        reps=args.reps,
        warmup=args.warmup,
        jax_version=jax.__version__,
        numpy_version=np.__version__,
        machine=args.machine,
        # NOT decoration.  A row can move 2.4x on this box for reasons
        # that are not in the code (see the module note); recording what
        # the machine was doing is what lets a later diff tell the two
        # apart instead of guessing.
        cpu_count=os.cpu_count(),
        load_average=[round(v, 2) for v in os.getloadavg()],
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        recorded=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        rows=rows,
    )
    out = os.path.join(_BENCH, "baselines", f"{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    timed = sum(1 for r in rows if "seconds" in r)
    print(f"wrote {out}: {len(rows)} rows, {timed} timed "
          f"({os.path.getsize(out)} bytes)", flush=True)
    # Judge by artifacts: a baseline file with no timed rows is not a
    # measurement, and saying so here is cheaper than discovering it in a
    # diff three steps later.
    return 0 if timed else 1


if __name__ == "__main__":
    sys.exit(main())
