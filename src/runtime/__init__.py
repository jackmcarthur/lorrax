"""THE LORRAX runtime: one startup call, one order, one honest report.

Every core driver needs the identical thing done in the identical order
before any physics runs::

    from runtime import initialize_communicator_stack
    RUNTIME = initialize_communicator_stack()   # BEFORE ``import jax``
    import jax
    ...
    mesh_xy = RUNTIME.mesh

The adopters (2026-08-01) are the seven chain drivers — ``gw.gw_jax``,
``centroid.kmeans_cli``, ``psp.get_dipole_mtxels``, ``gw.kin_ion_io``,
``bandstructure.htransform``, ``bse.bse_jax`` and ``bse.exciton_bands`` —
each making exactly ONE module-top call, gated per driver by
``tests/test_crossfile_requests.py`` (the R1 audit).  The two BSE CLIs that
take ``--px/--py`` still start on the canonical square mesh; a different
(square) shape goes through :meth:`RuntimeStack.reshape` (or the BSE factory
``bse.bse_ring_comm.create_mesh_xy``, which reuses the startup mesh whenever
the requested shape matches it).  Only square meshes are supported (repo
``docs/architecture/decisions.md`` 2026-08-01); both paths refuse
``px != py``.

:func:`initialize_communicator_stack` is that call.  It resolves the JAX
environment, installs the fail-fast excepthook, states the CPU-collectives
transport, skips the CUDA plugin on a CPU-only run, initialises
``jax.distributed``, resolves GPU-or-CPU, builds the run's device mesh with
every MPI/NCCL communicator it needs already created, arms the persistent
compile cache, and prints a rank-0 block stating every choice it made where
more than one was possible.  The order is load-bearing and is documented,
step by step with the reason for each step, on the function itself.

WHY IT IS ONE FUNCTION.  These pieces existed before; they were called by
five drivers in four different orders, and the order decides the answer.
The first ``jax.devices()`` freezes the platform and the allocator, so an
``os.environ.setdefault`` after it sets a string and changes nothing —
measured, job 7882443: two runs with byte-identical ``os.environ`` and
``bytes_limit`` 11.805 GB vs 0.000 GB.  ``os.environ`` is a false witness
for allocator state, which is why the report reads the live client.

THE PIECES ARE STILL PUBLIC, and :func:`bootstrap` still bundles the
env/distributed/CPU-fallback half exactly as it did — that is what
:func:`initialize_communicator_stack` calls, so there is one implementation
of the order rather than two.  A driver that genuinely cannot take a mesh
at import time can still call :func:`bootstrap`; it then owns the mesh, the
warm-up and the report itself, and it will drift, which is the situation
this module exists to end.

This module imports jax only inside function bodies, so importing it costs
nothing and the env defaults it sets are in place before jax reads them.
The distributed guard (``_LORRAX_JAX_DISTRIBUTED_DONE``) is an env sentinel
so it survives the re-import path ``python -m gw.gw_jax`` → ``gw_init`` →
``gw.gw_jax``; the mesh guard (``_STACK``) is a module global because a
second mesh is a per-process hazard, not a per-job one.
"""
from __future__ import annotations

import os
import subprocess

# Step 5b's gate.  A SIBLING module (see its "WHY IT LIVES IN runtime/"), and
# imported at module scope on purpose: it pulls only ``inspect``/``os`` at
# import, so it costs this package nothing and — decisively — brings no jax
# into a module that must be importable before jax reads its environment.
# It used to be ``common.jax_support``, reached through a function-local
# import; that was ``runtime`` reaching UP a layer, and hiding the edge inside
# a function body is not the same as not having it.  Same shape and same
# remedy as numbered request R9 (``runtime.xla_memory``, LANDED 2026-07-31).
#
# RELATIVE, like ``.xla_memory`` below, and that is not cosmetic: this package
# is imported under BOTH names in this tree — as ``runtime`` (``sys.path``
# has ``src/``) and as ``src.runtime`` (``tests/test_aot_memory.py``).  An
# absolute ``from runtime.jax_support import …`` here resolves the top-level
# name in the second case, which on a machine with an editable install of
# LORRAX is a DIFFERENT checkout's ``runtime`` package.  Measured: it raised
# ``ModuleNotFoundError: No module named 'runtime.jax_support'`` from the
# other tree.
from .jax_support import enforce as _enforce_jax_support


_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"

# Canonical LORRAX boolean grammar for env knobs — ONE token set, case- and
# whitespace-insensitive, shared by every knob in this module.  Before the
# consolidation (release audit 2026-07-28) LORRAX_MALLOC_TUNE,
# LORRAX_FAILFAST and the (now removed) LORRAX_GLOO_IFNAME check each parsed a
# different subset, so e.g. LORRAX_MALLOC_TUNE=OFF (or "False", "no", " 0 ")
# silently left the tuning ENABLED — the falsy-parse bug class this
# workstream had already fixed once for LORRAX_CHECK_REPLICA.
#
# BLANK IS NOT FALSE.  ``""`` used to be in this tuple, which made
# ``export LORRAX_MALLOC_TUNE=`` DISABLE a default-on knob while every other
# live vocabulary in the tree — ``ffi/gate.py`` ("unset or whitespace
# always maps to the gate's declared default"),
# ``file_io._slab_io_mpi_host._env_flag`` and ``gw.gw_config.env_bool``
# (which ``isdf.core`` imports since P1.3) —
# reads blank as UNSET and returns the knob's default.  A blank export is
# what a shell leaves behind for an unset variable expansion
# (``export X=$UNDEFINED``), so under the old rule a harness typo in the
# variable *name* silently turned the knob off.
_FALSY_TOKENS = ("0", "false", "no", "off")


def _env_falsy(name: str, default: str = "1") -> bool:
    """True when env knob ``name`` parses as falsy (0, false, no, off).

    Unset OR blank/whitespace falls back to ``default`` — the same reading
    of blank as every other LORRAX boolean parser (see ``_FALSY_TOKENS``).
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raw = default
    return raw.strip().lower() in _FALSY_TOKENS


#: Seconds spent inside the process's FIRST ``import jax``, measured where it
#: actually happens rather than guessed.
#:
#: WHY THIS EXISTS.  Until the 2026-08-09 import audit, every driver's startup
#: table charged this to ``env_and_distributed`` — a row whose name says
#: "jax.distributed and backend init" — so roughly two of that row's six
#: seconds were the Python import storm wearing a distributed-init label, and
#: an owner reading the table would have gone looking for a handshake problem
#: that was not there.  MEASURED at P=4 on an A100 node: ``import jax`` +
#: ``jax.numpy`` 2.165 s, ``jax.distributed.initialize()`` 1.023 s, first
#: ``jax.devices()`` 3.782 s.  Splitting the row is the whole fix; nothing
#: about the ORDER changes, which is why this is an accumulator read at the
#: existing import sites rather than a new import placed somewhere tidier.
_JAX_IMPORT_SECONDS = [0.0]


def _import_jax():
    """``import jax``, charging the first (uncached) one to a phase counter.

    Every later call is a ``sys.modules`` hit and adds nothing, so the counter
    holds the one-time cost no matter how many bring-up pieces call it.
    """
    import sys as _sys
    if "jax" in _sys.modules:
        import jax
        return jax
    import time as _time
    _t0 = _time.perf_counter()
    import jax
    _JAX_IMPORT_SECONDS[0] += _time.perf_counter() - _t0
    return jax


__all__ = [
    "initialize_communicator_stack",
    "finalize_process",
    "RuntimeStack",
    "collect_startup_facts",
    "format_startup_report",
    "bootstrap",
    "set_default_env",
    "announce_cpu_collectives",
    "skip_gpu_plugin_discovery",
    "init_jax_distributed",
    "fallback_to_cpu_if_no_gpu_backend",
    "install_failfast_excepthook",
]


# ---------------------------------------------------------------------------
# Demotion ledger — the record the startup report is built from
# ---------------------------------------------------------------------------
#
# Standing doctrine: ``auto`` MAY demote, but it must ANNOUNCE, from the rank
# it happened on.  Announcing is necessary and not sufficient: a line printed
# during a 40-second import storm, interleaved with every rank's stderr, is
# not something anybody reads.  So every demotion is ALSO recorded here and
# re-stated, in one place, in the startup report.  A demotion that reaches
# neither is a bug; a demotion that reaches only the scrolling log is a bug
# the owner asked to have fixed ("we should be using the logs more often to
# check performance").
#
# Rank-local by construction: this is a per-process list, and the report is
# printed from rank 0, so a demotion that happened only on rank 7 appears in
# rank 7's inline announcement and NOT in the report.  That asymmetry is
# deliberate and is stated in the report itself.
_DEMOTIONS: list = []

#: Filled in by :func:`tune_glibc_malloc` so the report can state whether the
#: OOM mitigation is actually armed rather than assuming it is.
_MALLOC_TUNE: dict = {"applied": None, "mmap_mb": None, "trim_mb": None,
                      "reason": None}

#: Filled in by :func:`init_jax_distributed`: which of the two
#: ``jax.distributed.initialize()`` forms actually ran.  A module global (not
#: an env var) on purpose — it is a report field, not a cross-import guard,
#: and the guard (``_DISTRIBUTED_SENTINEL``) already exists.
_DISTRIBUTED_FORM: list = []


def _record_demotion(msg: str) -> None:
    """Record one resolved-differently-than-requested event for the report."""
    if msg not in _DEMOTIONS:
        _DEMOTIONS.append(msg)


def bootstrap(*, platform: str = "gpu") -> None:
    """Canonical CLI bootstrap: env defaults + distributed init + CPU fallback.

    One call replaces the three-call header every LORRAX CLI used to
    carry.  MUST run before the caller's own ``import jax``:
    :func:`set_default_env` only works if jax has not been imported yet
    (jax reads its env at import time).  The jax imports *inside*
    :func:`init_jax_distributed` / :func:`fallback_to_cpu_if_no_gpu_backend`
    happen after the env is set, so they are safe.
    :func:`announce_cpu_collectives` must run after :func:`set_default_env`
    (it reads the resolved ``JAX_PLATFORMS``) — this slot satisfies that.

    Idempotent (each piece guards itself); no-op-ish in single-process
    runs.  ``platform`` forwards to :func:`set_default_env`.
    """
    set_default_env(platform=platform)
    announce_cpu_collectives()
    skip_gpu_plugin_discovery()
    init_jax_distributed()
    fallback_to_cpu_if_no_gpu_backend()
    install_failfast_excepthook()


def install_failfast_excepthook() -> None:
    """Make an uncaught per-rank exception kill the *job*, not just the rank.

    The exit-code problem this solves
    ---------------------------------
    In a ``jax.distributed`` run the ranks are peers in a collective
    program.  When one rank raises, CPython unwinds it normally: module
    ``atexit`` handlers run, the collectives backend tries to tear down, and
    the interpreter may block indefinitely inside a communicator whose
    peers are still sitting in a collective the dead rank will now never
    join.  Meanwhile the surviving ranks are blocked in that collective.
    The step ends when srun's timeout or the scheduler reaps it — and the
    campaign's logs repeatedly show that ending as **rc=0 at the sbatch
    level**, with a partial or absent set of outputs and no error anywhere
    near the top of the log.

    The fix is to make the *first* rank to fail exit non-zero,
    immediately, without unwinding:

    * print a rank-tagged banner (so ``tail`` on any log finds it),
    * flush stdout/stderr explicitly (``os._exit`` does not),
    * ``os._exit(1)`` — skipping atexit handlers and backend teardown,
      which is exactly what would otherwise hang.

    srun then sees a non-zero task exit, kills the remaining tasks in the
    step, and the job's exit code is non-zero.  A failure that used to
    look like success now looks like a failure.

    No-op in single-process runs (normal traceback + rc 1 already works,
    and ``os._exit`` would suppress useful teardown).  Opt out with
    ``LORRAX_FAILFAST=0``.
    """
    if _env_falsy("LORRAX_FAILFAST"):
        return
    if _resolve_proc_count() <= 1:
        return

    import sys

    if getattr(sys, "_lorrax_failfast_installed", False):
        return

    previous = sys.excepthook

    def _failfast(exc_type, exc_value, exc_tb):
        # SystemExit never reaches sys.excepthook, so an intentional
        # ``raise SystemExit(0)`` (e.g. LORRAX_EXIT_AFTER_ZETA) is
        # unaffected by this hook.
        rank = _resolve_proc_id()
        n = _resolve_proc_count()
        # Emit on BOTH streams: under srun+apptainer, stderr written just
        # before os._exit has been observed to vanish from the captured
        # logs (validation jobs 7884599/7884602: 4 crashes, zero stderr
        # text survived) while stdout survives.
        import traceback
        text = (
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            + f"\n*** LORRAX FAIL-FAST: rank {rank}/{n} died with "
            f"{exc_type.__name__}: {exc_value}\n"
            f"*** Exiting rc=1 WITHOUT teardown so this failure reaches "
            f"the job's exit code.  Peer ranks are blocked in a "
            f"collective this rank will never join; srun will now kill "
            f"the step.  (Disable with LORRAX_FAILFAST=0.)\n\n")
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass
        os._exit(1)

    sys.excepthook = _failfast
    sys._lorrax_failfast_installed = True


def set_default_env(*, platform: str = "gpu") -> None:
    """Set LORRAX's canonical JAX env defaults.

    Must be called BEFORE the first JAX **backend init** — not merely
    before ``import jax``.  jaxlib reads the GPU knobs in
    ``jaxlib/xla_client.py:generate_pjrt_gpu_plugin_options()``, called
    from ``jax_plugins/xla_cuda12/__init__.py:initialize()`` when the CUDA
    client is created; in this package that is the ``jax.devices()`` inside
    :func:`fallback_to_cpu_if_no_gpu_backend`.  A ``setdefault`` after that
    point sets the string and changes nothing (see the WHY note below).
    Uses ``setdefault`` so any caller-provided override wins.

    ``platform="gpu"`` (default) sets ``JAX_PLATFORMS="cuda,cpu"`` so
    JAX tries CUDA and falls back to CPU.  ``platform="cpu"`` forces CPU.

    WHY ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` IS SET HERE
    -------------------------------------------------------
    LORRAX's FFI handlers allocate OUTSIDE the XLA allocator — the cuFFT
    handler keeps a grow-only ``cudaMalloc`` arena
    (``ffi/cufft/__init__.py:30``), and cuSOLVERMp/libcal stage through
    plain ``cudaMalloc`` too.  Whatever XLA hoards is memory those cannot
    have, so the preallocation default is a correctness knob for this
    codebase, not a tuning one.

    Left unset, jaxlib omits the ``preallocate`` option entirely and the
    PJRT GPU client defaults to preallocating 75 % of the card.  Measured
    on 8 Quadro RTX 5000 (15.74 GB) across 2 nodes, jobs 7882442/7882447,
    each cell run twice with the second rep in reverse order (every rep-2
    number reproduced its rep-1 twin to 3 decimals):

    ================= ============== ================== =================
    PREALLOCATE       XLA holds      free for the FFI   largest cuFFT
    (allocator unset) for 6 GiB live  arena             plan creatable
    ================= ============== ================== =================
    unset (today)     11.93 GB       3.50 GB            3.07 GB
    ``false``          8.13 GB       13.50 GB           7.16 GB
    ================= ============== ================== =================

    That is a 2.3x improvement in the largest allocatable cuFFT plan, and
    it is the mechanism behind the failure recorded at
    ``scripts/profiling/aot_cufft_sanity.py:24`` (CrI3 Q=13: cuFFT plan
    creation failed although the compiled peak, 66.32 GB, fit in 80 GB).

    ``XLA_PYTHON_CLIENT_ALLOCATOR`` is deliberately NOT set, and the four
    accepted values are not interchangeable:

    * ``default``/``bfc`` — BFC.  What we get by leaving this unset.
      Keeps ``memory_stats()`` fully populated, which ``gw_init``'s
      high-water report, ``gw_output``'s XLA-pool banner and
      ``runtime/aot_memory`` all read.
    * ``platform`` — plain ``cudaMalloc``, **not** cudaMallocAsync
      (the plugin logs "Using platform allocator." vs "Using BFC
      allocator." and carries a separate ``CudaAsyncAllocator``).  Best
      headroom, but ``memory_stats()`` returns ``bytes_limit=0`` and
      ``peak_bytes_in_use=0`` — it would silently blind every memory
      report in the codebase.  Measured, job 7882447.
    * ``cuda_async`` — cudaMallocAsync.  Measurably the best of the three
      (0.19 GB overhead, 9.20 GB largest plan) AND it keeps
      ``peak_bytes_in_use``.  It is NOT the default here only because on
      Frontera rtx (sm_75) it needs the command-buffer restriction in
      ``config/frontera/ffi_env.sh:44-51``; that script sets both together
      and its explicit ``export`` correctly overrides this ``setdefault``.
      Promote it here only together with that XLA_FLAGS mitigation.

    ``TF_GPU_ALLOCATOR`` is a TensorFlow variable and is **inert for JAX**:
    a cell setting only ``TF_GPU_ALLOCATOR=cuda_malloc_async`` was
    byte-identical to the unset cell on every metric, including an 11.805
    GB BFC pool that the real ``cuda_async`` allocator never has (job
    7882442).  Do not add it back.

    A caller-supplied ``XLA_PYTHON_CLIENT_ALLOCATOR`` is VALIDATED here —
    see :func:`_check_allocator_env`.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # HDF5 FILE LOCKING OFF BY DEFAULT (audit A1, 2026-08-15).  A LORRAX
    # process maps TWO independent HDF5 library instances — h5py's bundled
    # wheel libhdf5 and the FFI's cray libhdf5_parallel — and the stores
    # are touched through both, sequentially.  Each instance takes its own
    # advisory locks and cannot see the other's, so one instance's stale
    # lock refuses the other's legitimate open with "unable to lock file"
    # on a file nothing is actually writing.
    #
    # A MITIGATION, NOT A FIX, and the difference matters: this masks
    # LOCK-level refusals only.  The metadata-cache divergence between the
    # two instances is untouched by it, and that is the actual A1 hazard —
    # ``file_io.hdf5_owner`` is what refuses a live cross-stack overlap and
    # counts the sequential alternation that remains.  HDF5 reads this at
    # file-open time, so setting it here (before any store is opened)
    # rather than at some module's import is what makes it order-independent.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        raise ValueError(f"platform must be 'gpu' or 'cpu', got {platform!r}")
    _check_allocator_env()
    tune_glibc_malloc()
    skip_gpu_plugin_discovery()


#: The four values jaxlib accepts for ``XLA_PYTHON_CLIENT_ALLOCATOR``.
#: Read straight off ``jaxlib/xla_client.py``'s own tuple in
#: ``generate_pjrt_gpu_plugin_options()``; anything else raises there.
ALLOCATOR_SPELLINGS = ("default", "platform", "bfc", "cuda_async")

#: How jaxlib normalises the value before that comparison: ``.lower()``,
#: and **no** ``.strip()``.  Mirrored exactly below, so this check accepts
#: precisely the set jaxlib accepts — a laxer check here would pass a value
#: on to the failure mode it exists to prevent.
_ALLOCATOR_ENV = "XLA_PYTHON_CLIENT_ALLOCATOR"


def _check_allocator_env(*, print_fn=print) -> None:
    """Refuse an unrecognised ``XLA_PYTHON_CLIENT_ALLOCATOR``, HERE.

    WHY THIS IS WORTH THE LINES.  jaxlib does validate the value —
    ``jaxlib/xla_client.py``::

        allocator = os.getenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'default').lower()
        ...
        if allocator not in ('default', 'platform', 'bfc', 'cuda_async'):
          raise ValueError(...)

    but it does so inside ``generate_pjrt_gpu_plugin_options()``, which runs
    from ``jax_plugins.xla_cuda12.initialize()`` while the CUDA client is
    being created — i.e. inside backend DISCOVERY, where every exception is
    caught and recorded as "this backend is unavailable".  jaxlib's precise
    ValueError never reaches the user.  What reaches the user is::

        RuntimeError: Backend 'cuda' is not in the list of known backends

    which names neither the variable nor the value, and reads as *missing
    hardware*.  Worse, on a build whose ``JAX_PLATFORMS`` lists a CPU
    fallback it is not an error at all: the run continues on CPU, at a
    fraction of the speed, because of a misspelling in a harness.

    Raising BEFORE the first ``jax.devices()`` (for this package, the one
    inside :func:`fallback_to_cpu_if_no_gpu_backend`) turns that into a
    message naming the knob, the bad value and the four legal ones.

    Two normalisation details are deliberate, because getting either wrong
    would re-open the hole:

    * jaxlib lowercases but does NOT strip, so ``" bfc"`` is a value jaxlib
      REJECTS.  This check must reject it too — accepting it here would
      hand a known-bad value straight to the anonymous failure.
    * blank is LORRAX-canonically "unset", but jaxlib compares ``''``
      against its tuple and rejects it.  So blank is not merely allowed
      through: the variable is REMOVED, which is what the author of
      ``export XLA_PYTHON_CLIENT_ALLOCATOR=`` meant and what jaxlib needs
      in order to apply its own ``'default'``.
    """
    raw = os.environ.get(_ALLOCATOR_ENV)
    if raw is None:
        return
    if not raw.strip():
        del os.environ[_ALLOCATOR_ENV]
        if _resolve_proc_id() == 0:
            print_fn(f"  [runtime] {_ALLOCATOR_ENV} was set but blank; "
                     f"removed it (blank = unset).  jaxlib rejects '' "
                     f"outright, so leaving it would have surfaced as "
                     f"\"Backend 'cuda' is not in the list of known "
                     f"backends\".", flush=True)
        return
    if raw.lower() in ALLOCATOR_SPELLINGS:            # jaxlib: .lower(), no strip
        return
    raise ValueError(
        f"{_ALLOCATOR_ENV}={raw!r} is not a value jaxlib accepts.  Legal: "
        f"{' | '.join(ALLOCATOR_SPELLINGS)} — compared after .lower() and "
        f"WITHOUT stripping, so surrounding whitespace is itself a "
        f"rejection.  Unset (or blank, which LORRAX removes) leaves jaxlib "
        f"its own default, BFC.  Left to jaxlib this raises inside CUDA "
        f"plugin discovery, where the exception is swallowed and reported "
        f"as \"Backend 'cuda' is not in the list of known backends\" — a "
        f"message that names neither this variable nor its value, and "
        f"reads as missing hardware rather than a typo.")


# ---------------------------------------------------------------------------
# CPU-only runs must not pay for the CUDA PJRT plugin
# ---------------------------------------------------------------------------

# "done"  -- the import blocker is installed
# "said"  -- one of the announcements has already been printed.  This function
#            is called from four entry points on purpose (see below), and two
#            of them fire in every driver, so without this the opt-out warning
#            prints twice (observed, job 7882076 cell c4_knob0).
_GPU_PLUGIN_SKIP_STATE = {"done": False, "said": False}


def _gpu_plugin_say(msg: str) -> None:
    if _GPU_PLUGIN_SKIP_STATE["said"] or _resolve_proc_id() != 0:
        return
    _GPU_PLUGIN_SKIP_STATE["said"] = True
    print(msg, flush=True)


def skip_gpu_plugin_discovery(*, announce: bool = True) -> bool:
    """When this run cannot use a GPU, stop jax from loading the CUDA plugin.

    Returns True when the skip is in force.

    WHAT IT COSTS NOT TO DO THIS (measured, job 7882055, fresh Frontera
    compute node, MoS2 4x4 deck, nb=256)::

        cold start, CUDA plugin discovery ON  ..... 88.2 s to the first
                                                     physics line
        same node class, discovery stubbed .......  11.3 s
        of which ``jax.devices()`` alone ..........  63.9 s  ->  0.06 s

    ``JAX_PLATFORMS=cpu`` does NOT prevent this on its own.
    ``xla_bridge.backends()`` calls ``_discover_and_register_pjrt_plugins()``
    BEFORE it looks at ``jax_platforms`` (jax 0.9.1,
    ``jax/_src/xla_bridge.py:797`` vs ``:808``), discovery imports
    ``jax_plugins.xla_cuda12``, and the FIRST statement of that module's
    ``initialize()`` is ``_load_nvidia_libraries()`` -- a ``ctypes`` dlopen of
    libcudart, libnvrtc, libcublas, libcublasLt, libnccl, libcupti,
    libcusparse, libcusolver, libcufft, libnvshmem_host and libcudnn.  On a
    CPU node every one of those is a cold dlopen of a library that will never
    be used, and it ends in ``cuInit`` failing anyway -- the
    ``RuntimeError: ... operation cuInit(0) failed`` banner every LORRAX CPU
    log carries is the receipt.

    WHY IT IS SLOW, since the byte count looks small.  ``mincore(2)`` over the
    venv right after a cold start (same job) shows only 173 MB resident, 119 MB
    of it CUDA -- at Lustre's 70 MB/s streaming rate that would be ~2 s.  It is
    not a bandwidth cost: dlopen relocation faults scattered 4 KB pages, each
    one a separate Lustre RPC, and it is the RPC latency times tens of
    thousands of faults that produces the minute.  Removing the libraries from
    the critical path removes the faults.

    HOW.  A ``sys.meta_path`` finder answers imports under the ``jax_plugins``
    namespace with a stub module whose ``initialize()`` does nothing.  jax's
    discovery loop then finds a plugin, calls its ``initialize()``, and gets a
    no-op; nothing is dlopened and nothing is registered.  No jax file is
    modified and no package is removed -- the same installed venv still runs
    GPU jobs, because this arms ONLY when the caller has said CPU and nothing
    but CPU.

    THE GATE has two arms, and each one is a case where jax was *already*
    going to end up on CPU -- the skip only lets it find that out before
    paying, never after:

    1. ``JAX_PLATFORMS`` resolves to exactly ``cpu``.  A GPU backend is
       forbidden outright, so the plugin can only be dead weight.

    2. ``JAX_PLATFORMS`` asks for a GPU but this node has no visible NVIDIA
       device (:func:`_gpu_is_present`).  jax's own ``backends()`` does
       ``if platform == "cuda" and not has_visible_nvidia_gpu(): continue``
       (jax 0.9.1 ``xla_bridge.py:829``) -- it skips CUDA on exactly this
       test, but only AFTER discovery has dlopened the libraries.  This arm
       is what makes the fix reach a user who never set ``JAX_PLATFORMS``:
       ``bootstrap(platform="gpu")`` and every driver's own
       ``setdefault("JAX_PLATFORMS", "cuda,cpu")`` land here on a CPU node.

       Because arm 2 means "this run is CPU", it does what the demotion in
       :func:`fallback_to_cpu_if_no_gpu_backend` would have done anyway --
       pins ``JAX_PLATFORMS=cpu`` and re-states the CPU collectives
       implementation -- so nothing downstream can tell the difference except
       the clock.  A run on a real GPU node NEVER reaches arm 2: the device
       nodes exist, and the CUDA driver cannot work without them.

    Opt out with ``LORRAX_CPU_SKIP_GPU_PLUGINS=0``; the opt-out announces
    itself, because "startup got a minute slower" is otherwise invisible.
    """
    plat = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    plats = [p.strip() for p in plat.split(",") if p.strip()]
    wants_gpu = any(p in ("cuda", "gpu", "rocm") for p in plats)

    if _GPU_PLUGIN_SKIP_STATE["done"]:
        # The import blocker is already installed, but arm 2's OTHER half —
        # pinning ``JAX_PLATFORMS=cpu`` so the resolved platform agrees with
        # the hardware — is per-CALL state: a later caller (a second
        # ``set_default_env()`` in the same process, e.g. another driver
        # entry or a test) may have re-set ``JAX_PLATFORMS`` to a GPU
        # request after the first arming.  Re-apply the demotion for this
        # call's env; without it the function returns True while leaving a
        # ``cuda,cpu`` request standing on a GPU-less node (the exact
        # disagreement ``test_set_default_env_defaults`` pins).
        if wants_gpu and not _gpu_is_present():
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "cpu"
        return True
    if plats == ["cpu"]:
        arm = "JAX_PLATFORMS=cpu"
    elif wants_gpu and not _gpu_is_present():
        arm = f"JAX_PLATFORMS={plat!r} but no NVIDIA device on this node"
    else:
        # No platform stated at all, or a GPU is stated and present: leave
        # jax's discovery exactly as it is.  Guessing here would be the one
        # way this function could cost someone a GPU.
        return False

    if _env_falsy("LORRAX_CPU_SKIP_GPU_PLUGINS"):
        _record_demotion(
            "LORRAX_CPU_SKIP_GPU_PLUGINS is off, so jax discovered and "
            "loaded the CUDA PJRT plugin on this CPU-only run, which "
            "measured 76.9 s of extra cold start on a Frontera node "
            "(job 7882076).")
        if announce:
            _gpu_plugin_say(
                "[runtime] LORRAX_CPU_SKIP_GPU_PLUGINS=0: jax will discover "
                "and load the CUDA PJRT plugin on this CPU-only run.  "
                "Measured cost of loading it on a cold Frontera node: 76.9 s "
                "(job 7882076, cell c4_knob0 -- 10.5 s to this line, 87.4 s to "
                "cuInit failing).")
        return False

    import sys

    if "jax_plugins.xla_cuda12" in sys.modules:
        # Too late: something already imported and initialised the plugin, so
        # the libraries are loaded and stubbing now would only hide it.  The
        # STARTUP SAVING is lost, but the arm-2 demotion is not: reaching
        # this line at all means the run is CPU-only (arm 1 or arm 2 was
        # chosen above), so pin ``JAX_PLATFORMS=cpu`` exactly as the normal
        # arm-2 path does.  Without this, a caller asking ``cuda,cpu`` on a
        # GPU-less node where the plugin import already happened (any pytest
        # process: jax's own discovery imports the plugin during the first
        # backend init) is left with a resolved platform that disagrees
        # with the hardware.
        if plats != ["cpu"] and not _gpu_is_present():
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "cpu"
            _record_demotion(
                f"JAX_PLATFORMS was requested as {plat!r} and was pinned to "
                f"'cpu' (no NVIDIA device on this node); the CUDA plugin was "
                f"already imported, so only the demotion — not the startup "
                f"saving — is in force.")
        if announce:
            _gpu_plugin_say(
                "[runtime] NOTE: the CUDA PJRT plugin was already imported "
                "before skip_gpu_plugin_discovery() ran; the CPU-only "
                "startup saving is NOT in force for this process.")
        return False

    import importlib.abc
    import importlib.machinery
    from types import ModuleType

    class _NoopPluginLoader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = ModuleType(spec.name)
            mod.__path__ = []           # so pkgutil.iter_modules finds nothing
            mod.initialize = lambda: None
            return mod

        def exec_module(self, module):
            return None

    class _JaxPluginBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "jax_plugins" or fullname.startswith("jax_plugins."):
                return importlib.machinery.ModuleSpec(
                    fullname, _NoopPluginLoader(), is_package=True)
            return None

    sys.meta_path.insert(0, _JaxPluginBlocker())
    _GPU_PLUGIN_SKIP_STATE["done"] = True
    _record_demotion(
        f"GPU PJRT plugin discovery was skipped ({arm}), so no CUDA library "
        f"was loaded; the measured saving on a cold Frontera node is 76.9 s "
        f"(job 7882076), and LORRAX_CPU_SKIP_GPU_PLUGINS=0 restores it.")
    if announce:
        _gpu_plugin_say(
            f"[runtime] {arm}: skipping GPU PJRT plugin discovery (no CUDA "
            f"libraries are loaded; measured 76.9 s of cold start on "
            f"Frontera, job 7882076).  LORRAX_CPU_SKIP_GPU_PLUGINS=0 "
            f"restores it.")
    if plats != ["cpu"]:
        # Arm 2: this IS the CPU demotion, taken early.  Do the rest of what
        # fallback_to_cpu_if_no_gpu_backend() does on demotion, so the run's
        # observable state is the same as if it had discovered the absence of
        # a GPU the slow way -- including SAYING which wire the collectives
        # are on, which is load-bearing (gloo's reduce-scatter corrupts).
        os.environ.pop("JAX_PLATFORM_NAME", None)
        os.environ["JAX_PLATFORMS"] = "cpu"
        _record_demotion(
            f"JAX_PLATFORMS was requested as {plat!r} and was pinned to 'cpu' "
            f"before backend init because this node exposes no NVIDIA device "
            f"node; this run has no GPU.")
        announce_cpu_collectives()
    return True


# glibc mallopt parameter numbers (malloc.h).
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3


def tune_glibc_malloc() -> bool:
    """Pin glibc's mmap/trim thresholds so freed XLA:CPU transients go back
    to the OS.  Returns True when the tuning was applied; every path that
    leaves it unapplied (opt-out, missing libc/mallopt, mallopt rejecting
    the value) announces itself on rank 0 — this tuning is a load-bearing
    OOM mitigation, and a guard that matters must say when it is not in
    force (QUALITY_PATTERNS #5/#7).

    WHY (workstream T, measured on Frontera).  XLA:CPU allocates and frees
    every intermediate through plain ``malloc``/``free``.  glibc's mmap
    threshold is DYNAMIC: the first time an mmap'd block is freed, glibc
    raises the threshold to that block's size (capped at 32 MB) and sets
    ``trim_threshold = 2 x mmap_threshold``.  From then on every allocation
    below 32 MB is served from the sbrk heap / per-thread arenas, and heap
    memory is returned to the OS only when the *top* of the heap happens to
    be free.  With 28 XLA worker threads churning multi-MB contraction
    scratch that condition is essentially never met, so RSS ratchets up
    monotonically for as long as the process keeps doing work — anonymous
    memory, page cache flat.

    In the ISDF zeta fit that showed up as a per-r-chunk ramp proportional to
    the back-solve FLOP count: +0.35 GB/rank/r-chunk at MoS2 12x12 / 606
    centroids / P=80, and +6.6 GB/rank/r-chunk at 1998 centroids / P=144,
    which is what killed jobs 7874803 / 7875070 / 7875071 with
    ``std::bad_alloc`` mid-loop while the planner's static estimate was
    comfortable.  ``jax.live_arrays()`` stayed EXACTLY constant throughout —
    nothing was retained on the JAX side.

    Pinning ``M_MMAP_THRESHOLD`` also DISABLES the dynamic adjustment, so
    every allocation at or above the threshold is mmap'd and ``munmap``'d on
    free — returned to the OS immediately, no fragmentation.  Measured cost:
    within noise (~4% on the r-chunk wall at 40 nodes), and steady-state RSS
    dropped 2.3 -> 1.7 GB/rank as a bonus.

    Knobs: ``LORRAX_MALLOC_TUNE=0`` disables; ``LORRAX_MALLOC_MMAP_MB`` /
    ``LORRAX_MALLOC_TRIM_MB`` override the thresholds (MB).
    """
    if _env_falsy("LORRAX_MALLOC_TUNE"):
        _MALLOC_TUNE.update(applied=False,
                            reason="disabled by LORRAX_MALLOC_TUNE")
        _record_demotion(
            "glibc malloc tuning is DISABLED by LORRAX_MALLOC_TUNE, so long "
            "XLA:CPU runs regain the per-r-chunk RSS ramp that OOM-killed "
            "jobs 7874803, 7875070 and 7875071 (workstream T).")
        if _resolve_proc_id() == 0:
            print("[runtime] glibc malloc tuning DISABLED by "
                  "LORRAX_MALLOC_TUNE: long XLA:CPU runs regain the "
                  "per-r-chunk RSS ramp this tuning removes (workstream T).",
                  flush=True)
        return False
    reason = None
    mmap_mb = trim_mb = None
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mmap_mb = int(os.environ.get("LORRAX_MALLOC_MMAP_MB", "1"))
        trim_mb = int(os.environ.get("LORRAX_MALLOC_TRIM_MB", "128"))
        ok = libc.mallopt(_M_MMAP_THRESHOLD, mmap_mb * 1024 * 1024)
        ok &= libc.mallopt(_M_TRIM_THRESHOLD, trim_mb * 1024 * 1024)
        if not ok:
            reason = "mallopt() returned 0 (threshold value rejected)"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    _MALLOC_TUNE.update(applied=(reason is None), mmap_mb=mmap_mb,
                        trim_mb=trim_mb, reason=reason)
    if reason is not None:
        _record_demotion(
            f"glibc malloc tuning was NOT applied ({reason}), so the "
            f"per-r-chunk RSS ramp that OOM-killed jobs 7874803, 7875070 and "
            f"7875071 is not mitigated on this rank (workstream T).")
        # This is the load-bearing mitigation for the +GB/rank/r-chunk RSS
        # ramp that killed jobs 7874803/7875070/7875071 with std::bad_alloc
        # (workstream T).  It used to fail silently here — a run whose OOM
        # guard never armed left zero log evidence (release audit
        # 2026-07-28).  Rank 0 speaks: glibc/mallopt behaviour is uniform
        # across the ranks of one homogeneous job.
        if _resolve_proc_id() == 0:
            print(f"[runtime] WARNING: glibc malloc tuning NOT applied "
                  f"({reason}); expect the RSS ramp on long CPU runs that "
                  f"OOM-killed jobs 7874803/7875070/7875071 — see "
                  f"workstream T.", flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# CPU collectives announcement
# ---------------------------------------------------------------------------
#
# LORRAX runs its multi-process CPU collectives on MPI
# (JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi -> MPItrampoline -> Intel MPI).  The
# harness sets that, not this module: MPITRAMPOLINE_LIB names a build artifact
# outside the repo, so the choice has to stay visible in the launch script
# (docs/dev/mpi_collectives.md).
#
# What this function does is make the resolved choice AUDIBLE.  jax's own
# default is gloo, and gloo is not merely slower here -- its reduce-scatter
# SILENTLY CORRUPTS ~5% of executions with a plausible wrong value and a zero
# exit code.  A forgotten export must therefore not be a silent event
# (standing doctrine #3: a demotion may happen, but it must announce itself
# from the rank it happens on).  One rank-0 line, no branching on transport
# anywhere else in src/.


def announce_cpu_collectives() -> None:
    """Print the resolved CPU collectives implementation once, from rank 0.

    No-op for single-process runs and for non-CPU platforms (GPU collectives
    are NCCL's).  WARNS when a multi-process CPU run has landed on gloo,
    because that is the corrupting reduce-scatter backend and the failure it
    produces is silent (see docs/dev/mpi_collectives.md).
    """
    if _resolve_proc_count() <= 1:
        return
    plat = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if plat and "cpu" not in [p.strip() for p in plat.split(",")]:
        return                          # pure-GPU platform: NCCL owns these
    # A multi-platform value ("cuda,cpu" — the bootstrap(platform="gpu")
    # default) lands on the CPU backend only when no GPU is actually there.
    # On a real GPU node it does not, and warning about the CPU collectives
    # implementation would be a false positive.  (Same test the Gloo pin used
    # to make, and the same one fallback_to_cpu_if_no_gpu_backend makes.)
    if plat and plat != "cpu" and _gpu_is_present():
        return
    impl = os.environ.get(
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "gloo").strip().lower() or "gloo"
    if _resolve_proc_id() != 0:
        return
    if impl == "mpi":
        wrap = os.environ.get("MPITRAMPOLINE_LIB", "")
        print(f"[runtime] CPU collectives: mpi (MPItrampoline -> "
              f"{wrap or '<MPITRAMPOLINE_LIB UNSET>'}).", flush=True)
        if not wrap:
            print("[runtime] WARNING: JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi "
                  "with MPITRAMPOLINE_LIB unset — MPItrampoline has no "
                  "wrapper to load.  See docs/dev/mpi_collectives.md.",
                  flush=True)
    else:
        import sys
        print(f"[runtime] WARNING: CPU collectives implementation is {impl!r}, "
              "not 'mpi'.  gloo's reduce-scatter is MEASURED to return wrong "
              "data silently (~5% of executions, plausible values, rc=0) and "
              "is 14-30x slower on this fabric.  Export "
              "JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi plus MPITRAMPOLINE_LIB, "
              "LORRAX_MPI_FORCE_THREAD_MAIN=1 and LORRAX_MPI_FINALIZE_FIX="
              "skip_atexit — see docs/dev/mpi_collectives.md.",
              file=sys.stderr, flush=True)


def _resolve_proc_count() -> int:
    """Process count: JAX_PROCESS_COUNT → JAX_NUM_PROCESSES → SLURM_NTASKS → 1."""
    return int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get(
            "JAX_NUM_PROCESSES",
            os.environ.get("SLURM_NTASKS", "1"))))


def _resolve_proc_id() -> int:
    """Process index: JAX_PROCESS_INDEX → SLURM_PROCID → 0."""
    return int(os.environ.get(
        "JAX_PROCESS_INDEX",
        os.environ.get("SLURM_PROCID", "0")))


def _resolve_coordinator_address() -> str:
    """Coordinator address for jax.distributed.

    JAX_COORDINATOR_ADDRESS overrides everything.  Otherwise resolve the
    first host of SLURM_NODELIST via ``scontrol show hostnames`` and
    append port 12355.  Final fallback: SLURMD_NODENAME / HOSTNAME /
    'localhost' + port 12355.
    """
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coord:
        return coord
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, check=True,
            )
            first_host = result.stdout.strip().split("\n")[0]
            return f"{first_host}:12355"
        except Exception:
            pass
    host = (os.environ.get("SLURMD_NODENAME")
            or os.environ.get("HOSTNAME")
            or "localhost")
    return f"{host}:12355"


def init_jax_distributed() -> None:
    """Call ``jax.distributed.initialize()`` idempotently.

    Safe to call multiple times — the ``_LORRAX_JAX_DISTRIBUTED_DONE``
    env sentinel persists across re-imports within a process (module-
    level Python globals don't, which is why the previous per-driver
    copies sometimes double-initialised when ``python -m gw.gw_jax``
    pulled ``gw.gw_jax`` in again through ``gw_init``).

    The Cray MPICH stack on Perlmutter runs each rank with
    ``CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`` — exactly one GPU per
    process.  ``jax.distributed.initialize()`` with no args then hangs
    in the topology exchange because it assumes each process owns
    *all* local GPUs.  We pass ``local_device_ids`` explicitly,
    derived from CUDA_VISIBLE_DEVICES.  First try that; on failure
    fall back to the explicit ``(coordinator_address, num_processes,
    process_id)`` form.

    ``JAX_COORDINATOR_ADDRESS``, when set, SKIPS the auto-detected form and
    goes straight to the explicit one.  Auto-detection derives the coordinator
    port from ``SLURM_JOB_ID``, so every step of one allocation lands on the
    SAME port: two concurrent runs in a shared interactive allocation (two
    agents attached to one salloc, or one agent's two launches) join each
    other's coordinator and die with ``ABORTED: task N unexpectedly tried to
    connect with a different incarnation``, or hang until srun SIGKILLs them.
    Set a per-launch address (``--env=JAX_COORDINATOR_ADDRESS=$HOST:$PORT``
    with a port unique to the launch) to keep the runs independent.
    """
    # Drivers with the three-call header (gw.kin_ion_io, psp.run_nscf, ...)
    # reach ``runtime`` here first, and this is still BEFORE anything has
    # asked jax for a backend -- the only window in which the CPU-only GPU
    # plugin skip can be armed.  Idempotent and self-gating; a GPU run
    # (JAX_PLATFORMS != "cpu") passes straight through.
    skip_gpu_plugin_discovery()

    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return

    proc_count = _resolve_proc_count()
    if proc_count <= 1:
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        _DISTRIBUTED_FORM.append(
            "jax.distributed.initialize() was not called because this is a "
            "single-process run")
        return

    jax = _import_jax()

    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_local = len([x for x in cv.split(",") if x.strip()]) if cv else 0
    init_kwargs = {"local_device_ids": list(range(n_local))} if n_local else {}
    if not os.environ.get("JAX_COORDINATOR_ADDRESS"):
        try:
            jax.distributed.initialize(**init_kwargs)
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            _DISTRIBUTED_FORM.append(
                f"jax.distributed.initialize() took its auto-detected form "
                f"with local_device_ids={init_kwargs.get('local_device_ids')}")
            return
        except Exception as exc:
            _record_demotion(
                f"The auto-detected jax.distributed.initialize() form failed "
                f"({type(exc).__name__}: {exc}), so the explicit "
                f"coordinator/num_processes/process_id form was used instead.")

    # ``local_device_ids`` matters on BOTH paths: without it the explicit form
    # assumes each process owns every local GPU and dies with
    # "CUDA_ERROR_INVALID_DEVICE: invalid device ordinal" under the one-GPU-
    # per-process binding select_gpu.sh sets up.
    coord = _resolve_coordinator_address()
    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=proc_count,
        process_id=_resolve_proc_id(),
        **init_kwargs,
    )
    os.environ[_DISTRIBUTED_SENTINEL] = "1"
    _DISTRIBUTED_FORM.append(
        f"jax.distributed.initialize() took its explicit form with "
        f"coordinator_address={coord!r}, num_processes={proc_count} and "
        f"local_device_ids={init_kwargs.get('local_device_ids')}")


def nccl_warmup(mesh_xy) -> None:
    """Pre-initialise every NCCL communicator we'll need later.

    First call on a new NCCL communicator pays ``ncclCommInitRank`` cost
    (~1-2 s on A100) — topology discovery.  Each unique ``replica_groups``
    pattern is a separate communicator.  Our mesh uses three patterns:

      * full-mesh psum     — ``{{0,1,2,3}}``   (used by ``jnp.mean``,
                                                 reductions with no axis
                                                 arg, etc.)
      * 'x'-axis psum      — ``{{0,1},{2,3}}`` (sigma reduce-scatter x
                                                 stage; also triggered by
                                                 any axis-'x' psum)
      * 'y'-axis psum      — ``{{0,2},{1,3}}`` (sigma reduce-scatter y
                                                 stage; any axis-'y' psum)

    Firing a dummy psum on each pattern at driver init moves the
    multi-second first-call cost off whatever timed section would
    otherwise have hit it (most recently: a 1.9 s single ``all-reduce-start``
    inside ``jit(_mean)/reduce_sum`` during the sigma phase, traced via
    the profiling stack).  No-op in single-process mode.
    """
    import jax
    import jax.numpy as jnp
    if jax.process_count() <= 1:
        return
    from jax.sharding import NamedSharding, PartitionSpec as P
    # Each (axis_spec) shape below has its reduction emit a distinct NCCL
    # communicator at XLA lower time.  ``jnp.sum`` on an array sharded
    # over the given axes lowers to the right psum; ``jax.lax.psum`` isn't
    # callable from top-level jit (needs shard_map/pmap context), so we
    # route the warmup through the implicit-reduction path instead.
    shape2d = tuple(mesh_xy.shape[ax] for ax in mesh_xy.axis_names)
    warm_specs = [(shape2d, P(*mesh_xy.axis_names))]       # full-mesh psum
    for ax in mesh_xy.axis_names:
        n_ax = int(mesh_xy.shape[ax])
        warm_specs.append(((n_ax,), P(ax)))                # per-axis psum
    for shape, spec in warm_specs:
        sharding = NamedSharding(mesh_xy, spec)
        x = jax.device_put(jnp.ones(shape, dtype=jnp.float64), sharding)
        _ = jax.jit(jnp.sum)(x).block_until_ready()


def _gpu_is_present() -> bool:
    """True if an NVIDIA GPU is actually visible to this process.

    Used to decide whether a JAX GPU-backend init failure is a benign
    "no GPU here, run on CPU" (login/CPU node) or a genuine GPU-init
    failure that must NOT be masked (GPU node with a driver/library
    problem).  Signals, cheapest first:

      * ``CUDA_VISIBLE_DEVICES=""`` → explicitly masked, no GPU.
      * any ``/dev/nvidia[0-9]*`` device node (or ``/dev/nvidiactl``) →
        a GPU is physically present on this node.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    import glob
    return bool(glob.glob("/dev/nvidia[0-9]*")) or os.path.exists("/dev/nvidiactl")


def fallback_to_cpu_if_no_gpu_backend() -> None:
    """If ``jax.devices()`` fails because no GPU backend came up, retry on CPU.

    Two failure strings count as "no GPU backend":

      * ``Unknown backend: 'gpu'``            — JAX_PLATFORMS unset / 'gpu'
        with no CUDA runtime (sandbox / test contexts).
      * ``Unable to initialize backend 'cuda'`` (and the 'gpu'/'rocm'
        variants) — what JAX raises when ``JAX_PLATFORMS='cuda,cpu'`` (the
        value ``set_default_env()`` sets) is tried on a CPU node.

    We downgrade to CPU ONLY when no GPU is actually present
    (:func:`_gpu_is_present`): on a real GPU node a cuda-init failure is a
    genuine error and is re-raised rather than silently masked by a
    catastrophically-slow CPU run.  On downgrade we clear JAX_PLATFORM_NAME,
    force JAX_PLATFORMS=cpu, and drop any cached (failed) backend so the
    caller's next ``jax.devices()`` re-initialises cleanly on CPU.

    The downgrade also RE-RUNS :func:`announce_cpu_collectives`: a run that
    started on a GPU platform value skipped the announcement at bootstrap,
    and after the downgrade it IS a multi-process CPU run, so the resolved
    collectives implementation has to be stated.  An environmental accident
    must not silently change the wire under every collective
    (quality-pattern #8).
    """
    # Last arming point before the first ``jax.devices()`` in the canonical
    # driver header: this call IS that first backend init, and the CUDA plugin
    # dlopen happens inside it.  A driver that calls only this piece (no
    # bootstrap, no init_jax_distributed) still gets the CPU-only skip.
    skip_gpu_plugin_discovery()
    jax = _import_jax()
    caught = None
    try:
        jax.devices()
        return
    except RuntimeError as exc:
        caught = exc            # bind to a name the except block won't delete
        msg = str(exc)
    no_gpu_backend = (
        "Unknown backend: 'gpu'" in msg
        or "Unable to initialize backend 'cuda'" in msg
        or "Unable to initialize backend 'gpu'" in msg
        or "Unable to initialize backend 'rocm'" in msg)
    if not (no_gpu_backend and not _gpu_is_present()):
        # Genuine failure (real GPU-init error on a GPU node, or a
        # non-backend RuntimeError): re-raise the ORIGINAL exception.  A
        # bare ``raise`` here would throw "No active exception to re-raise"
        # since the except block has exited.
        raise caught
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    _record_demotion(
        f"The GPU backend failed to initialise and no NVIDIA device is "
        f"present, so JAX_PLATFORMS was forced to 'cpu' after backend init "
        f"({str(msg).splitlines()[0][:160]}).")
    try:                       # drop the half-initialised cuda backend cache
        jax.clear_backends()
    except Exception:
        pass
    # The platform is CPU now, so state the collectives implementation (see
    # docstring).  announce_cpu_collectives self-guards on P<=1 and rank.
    announce_cpu_collectives()


# ===========================================================================
#  THE startup entry point
# ===========================================================================
#
# ONE function every core LORRAX driver calls, so that gw_jax, htransform,
# bse_jax, run_nscf and kmeans_cli bring the machine up in the SAME order,
# resolve GPU-vs-CPU by the SAME rule, and print the SAME account of what
# they resolved.  Before this existed the six pieces below were called by
# five drivers in four different orders, and the order is load-bearing: the
# first ``jax.devices()`` freezes the allocator, and every ``setdefault``
# after it sets a string and changes nothing (measured, job 7882443 --
# identical ``os.environ``, ``bytes_limit`` 11.805 GB vs 0.000 GB).
#
# WHY IT RETURNS THE MESH.  "Bring up the communicator stack" is not done
# when ``jax.distributed.initialize`` returns: the MPI cliques and the NCCL
# communicators do not exist yet, and whichever physics kernel fires the
# first collective creates them -- from an XLA pool worker, which is the
# refusal that killed the BSE TDA Lanczos (32 refusals at P=16, gate
# 7881216).  The mesh and its warm-up are therefore part of startup, not
# part of physics, and a driver that gets a mesh back cannot forget to warm
# it.
# ===========================================================================

#: Set by the first :func:`initialize_communicator_stack`; every later call
#: returns it.  A module global rather than an env sentinel because it holds
#: a live ``Mesh`` -- and because the thing it guards (a SECOND mesh, hence a
#: second set of communicators) is a per-process hazard, not a per-job one.
_STACK = None


class RuntimeStack:
    """What :func:`initialize_communicator_stack` resolved, for the driver.

    Attributes
    ----------
    mesh
        THE run's ``('x','y')`` device mesh, already clique-warmed.  A
        driver should pass this everywhere a mesh is wanted and never build
        a second one: a second ``Mesh`` object is a second set of
        communicators and a second copy of every shape-keyed jit cache.
    platform, device_kind, n_devices, n_local_devices
        The RESOLVED backend, after any CPU demotion -- never the request.
    process_index, process_count
        This rank and the world size, from jax (not from SLURM).
    facts
        The dict :func:`collect_startup_facts` produced; the report is a
        pure function of it, so a test can assert on either.
    report
        The rank-0 startup block, as a tuple of sentences.  Kept so a
        driver (or a test) can re-emit or grep it without re-collecting.
    """

    __slots__ = ("mesh", "platform", "device_kind", "n_devices",
                 "n_local_devices", "process_index", "process_count",
                 "facts", "report")

    def __init__(self, *, mesh, platform, device_kind, n_devices,
                 n_local_devices, process_index, process_count, facts, report):
        self.mesh = mesh
        self.platform = platform
        self.device_kind = device_kind
        self.n_devices = n_devices
        self.n_local_devices = n_local_devices
        self.process_index = process_index
        self.process_count = process_count
        self.facts = facts
        self.report = tuple(report)

    @property
    def mesh_shape(self):
        return tuple(int(n) for n in self.mesh.devices.shape)

    def reshape(self, px: int, py: int, *, print_fn=None):
        """Replace the startup mesh with a ``px`` x ``py`` one, warmed.

        FOR THE ONE CASE THE STARTUP CALL CANNOT SERVE.  The entry point
        runs above ``import jax``, so it cannot know a mesh shape that comes
        from the command line — and ``bse.exciton_bands`` / ``bse.bse_w_exact``
        take ``--px``/``--py``.  Those drivers still make ONE startup call;
        they follow it with this, after argparse.

        Not a second mesh in the sense the re-entry guard refuses: the old
        one is dropped, the new one is warmed by the same
        :func:`common.collectives.prepare_mesh`, and the swap ANNOUNCES
        itself, because the startup block above it named the old shape and a
        report that quietly describes a mesh the run is not using is worse
        than no report.  A no-op when the shape already matches.
        """
        say = print_fn if print_fn is not None else _print_rank0
        if (int(px), int(py)) == self.mesh_shape:
            return self.mesh
        if int(px) != int(py):
            raise ValueError(
                f"RuntimeStack.reshape({px}, {py}): only square 2-D meshes "
                f"are supported (repo docs/architecture/decisions.md, "
                f"2026-08-01) — rectangular meshes complicate ScaLAPACK "
                f"grid geometry and the divisibility contracts for no "
                f"measured benefit.  Pass --px == --py.")
        import jax
        import numpy as np
        from jax.sharding import Mesh
        from common.collectives import prepare_mesh
        devices = jax.devices()
        if int(px) * int(py) != len(devices):
            raise ValueError(
                f"RuntimeStack.reshape({px}, {py}) needs {int(px) * int(py)} "
                f"devices and this job has {len(devices)}.")
        old = self.mesh_shape
        wanted = Mesh(np.asarray(devices).reshape(int(px), int(py)),
                      axis_names=tuple(self.mesh.axis_names))
        self.mesh = prepare_mesh(wanted, axis_names=tuple(self.mesh.axis_names),
                                 print_fn=say)
        say(f"  The startup mesh {old[0]}x{old[1]} was replaced by the "
            f"{int(px)}x{int(py)} mesh this driver's arguments asked for, and "
            f"the new mesh's communicator cliques were warmed before the "
            f"first physics jit.")
        return self.mesh

    def __repr__(self):                                       # pragma: no cover
        return (f"RuntimeStack(platform={self.platform!r}, "
                f"mesh={self.mesh_shape}, "
                f"proc={self.process_index}/{self.process_count})")


def initialize_communicator_stack(*, platform: str = "gpu",
                                  axis_names=("x", "y"),
                                  print_fn=None) -> RuntimeStack:
    """Bring up the whole runtime, in the one order that is correct.

    Call this ONCE, at the TOP of a driver module, **above the driver's own
    ``import jax``** -- exactly where ``bootstrap()`` used to sit.  It is
    importable without jax (this module imports jax only inside function
    bodies), so the env defaults it sets are in place before jax reads them.

    THE ORDER, AND WHY EACH STEP IS WHERE IT IS
    -------------------------------------------
    0. :func:`install_failfast_excepthook` -- FIRST, earlier than
       ``bootstrap()`` puts it.  Everything below can raise on one rank
       while its peers block in a collective (``jax.distributed.initialize``
       and the clique warm-up are both collective).  The hook is what turns
       that from "the job ends rc=0 with no output" into a non-zero exit,
       so installing it after the collectives would leave the riskiest part
       of startup unprotected.  Idempotent, so ``bootstrap()``'s own call
       below is a no-op.
    1. :func:`set_default_env` -- JAX_ENABLE_X64, JAX_PLATFORMS,
       XLA_PYTHON_CLIENT_PREALLOCATE=false, the allocator-spelling refusal,
       the glibc malloc tuning, and the first arming of the CPU-only plugin
       skip.  MUST precede the first backend init; jax reads x64 at import
       and the GPU knobs when the CUDA client is built.
    2. :func:`announce_cpu_collectives` -- after (1), because it reads the
       RESOLVED ``JAX_PLATFORMS``; before any collective, because gloo's
       reduce-scatter corrupts silently and the operator has to learn that
       before the run, not after.
    3. :func:`skip_gpu_plugin_discovery` -- idempotent re-arm; the last
       point at which the CUDA dlopen (76.9 s cold, job 7882076) can still
       be avoided.
    4. :func:`init_jax_distributed` -- one ``jax.distributed.initialize()``
       per process, guarded by an env sentinel that survives re-imports.
    5. :func:`fallback_to_cpu_if_no_gpu_backend` -- THIS is the first
       ``jax.devices()``, i.e. backend init.  After it the platform and the
       allocator are frozen and ``os.environ`` stops being evidence.
       Demotes to CPU only when no GPU is physically present; a cuda-init
       failure ON a GPU node is re-raised, never masked.

       (2)-(5) are ``bootstrap()``, unchanged and still separately tested;
       this function calls it rather than re-listing its steps, so there is
       exactly one implementation of the order.
    5b. :func:`runtime.jax_support.enforce` -- the running JAX is the one this
       tree is written against, or the run REFUSES here.  Needs (5): the
       backend must exist so ``jax._src`` is fully populated and its
       signatures are readable.  Before (6) because (6) performs the first
       ``jit``, and the failure this catches is a ``TypeError`` inside that
       jit -- or its silent variant, a compile cache reporting
       ``enabled=True`` while writing zero entries.  Costs one
       ``inspect.signature`` per patched hook.
    6. :func:`common.collectives.prepare_mesh` -- the run's mesh, then
       ``warm_mesh_cliques`` (CPU/MPI) and ``nccl_warmup`` (GPU/NCCL).
       Collective: every rank must reach it.  Needs (4) and (5).
    6b. :func:`_enforce_required_ffi` -- the FFI layer is REQUIRED
       (decisions.md 2026-08-01): each gate's :meth:`ffi.gate.Gate.enforce`
       runs against the fresh mesh, so a missing or unloadable FFI library
       refuses AT STARTUP, naming the ``.so`` and the fix, instead of at
       the first kernel factory mid-run; an explicit ``=0`` either refuses
       (deleted duplicate) or announces the uncertified opt-out here.
       Needs (6): the refusal/announce vocabulary is mesh-platform-aware.
    7. ``ensure_jax_compile_cache`` -- needs ``jax.process_count()`` from
       (4)/(5), and its P>1 agreement layer rides the coordination service
       that (4) started.  Deliberately AFTER (6): the clique warm-up's jit
       must stay the small, sequentially-executed program that makes the
       warm-up work at all, and running it before the cache layer is
       installed keeps that measured behaviour byte-identical.
    8. The startup report -- LAST, because it reads the live client, and
       the live client does not exist until (5).

    Parameters
    ----------
    platform
        ``"gpu"`` (default; JAX_PLATFORMS=``cuda,cpu``, so a GPU-less node
        demotes) or ``"cpu"`` (forced).  The ``cpu`` in ``cuda,cpu`` is
        load-bearing: ``solvers/lanczos.py`` uses ``jax.debug.callback``,
        which needs a local CPU device, so ``JAX_PLATFORMS=cuda`` alone
        breaks BSE.
    axis_names
        The mesh axis names.  Everything in-tree uses ``('x','y')``.
    print_fn
        Where the report goes; default is a rank-0-gated ``print``.

    Idempotent.  Drivers import each other (``python -m gw.gw_jax`` re-imports
    ``gw.gw_jax`` through ``gw_init``), so the second call must not build a
    second mesh -- it returns the first :class:`RuntimeStack`.  A second call
    asking for DIFFERENT ``axis_names`` is not a re-entry, it is a request for
    a second mesh, and it RAISES rather than silently handing back the first.
    """
    global _STACK
    if _STACK is not None:
        if tuple(axis_names) != tuple(_STACK.mesh.axis_names):
            raise RuntimeError(
                f"initialize_communicator_stack() was already called in this "
                f"process with axis_names={tuple(_STACK.mesh.axis_names)} and "
                f"is now asked for {tuple(axis_names)}.  That is a request for "
                f"a SECOND mesh, i.e. a second set of MPI/NCCL communicators "
                f"and a second copy of every shape-keyed jit cache -- not a "
                f"re-entry.  Pass the first mesh (RuntimeStack.mesh) to "
                f"common.collectives.prepare_mesh() if you need it reshaped.")
        return _STACK

    say = print_fn if print_fn is not None else _print_rank0
    import time

    # Every phase is timed.  Two of these are the campaign's largest
    # unexplained startup costs -- 43.8 s of ``jax.distributed`` init at
    # P=16, and 75.0 s to first output on a cold node (jobs 7881949 /
    # 7882055) -- and they were invisible because they happen before any
    # driver's ``timing.reset()``.  Handing the numbers back in ``facts``
    # lets a driver re-record them into its own table AFTER that reset, so
    # the stage table still sums to the wall.
    _t0 = time.perf_counter()
    # -- 0 ------------------------------------------------------------------
    install_failfast_excepthook()
    # -- 1..5 ---------------------------------------------------------------
    bootstrap(platform=platform)
    # -- 5b -----------------------------------------------------------------
    _enforce_supported_jax(say)
    _t_boot = time.perf_counter()
    # -- 6 ------------------------------------------------------------------
    from common.collectives import prepare_mesh
    mesh = prepare_mesh(axis_names=tuple(axis_names), print_fn=say)
    # -- 6b -----------------------------------------------------------------
    _enforce_required_ffi(mesh)
    _t_mesh = time.perf_counter()
    # -- 7 ------------------------------------------------------------------
    cache_error = None
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as exc:                                  # noqa: BLE001
        # Not fatal (the cache is an optimisation), but it must not be
        # SILENT: a run that quietly lost its compile cache looks like a
        # performance regression with no cause in the log.
        cache_error = f"{type(exc).__name__}: {exc}"
    _t_cache = time.perf_counter()
    # -- 8 ------------------------------------------------------------------
    facts = collect_startup_facts(mesh, cache_error=cache_error)
    # ``jax_import`` is CARVED OUT of ``env_and_distributed``, not added
    # beside it: the rows have to keep summing to ``total`` or every driver
    # epilogue that re-records them (htransform, gw_jax, bse_jax) breaks its
    # "rows + (untimed) == wall" property.  See ``_JAX_IMPORT_SECONDS``.
    _jax_import = min(_JAX_IMPORT_SECONDS[0], _t_boot - _t0)
    facts["elapsed"] = {
        "jax_import": _jax_import,
        "env_and_distributed": (_t_boot - _t0) - _jax_import,
        "mesh_and_warmup": _t_mesh - _t_boot,
        "compile_cache": _t_cache - _t_mesh,
        "measurement": time.perf_counter() - _t_cache,
        "total": time.perf_counter() - _t0,
    }
    report = format_startup_report(facts)
    import jax
    _STACK = RuntimeStack(
        mesh=mesh,
        platform=facts["backend"],
        device_kind=facts["device_kind"],
        n_devices=facts["n_devices"],
        n_local_devices=facts["n_local_devices"],
        process_index=facts["process_index"],
        process_count=facts["process_count"],
        facts=facts,
        report=report,
    )
    if jax.process_index() == 0:
        for line in report:
            say(line)
    return _STACK


def _print_rank0(*a, **k):
    if _resolve_proc_id() == 0:
        k.setdefault("flush", True)
        print(*a, **k)


def finalize_process(rc: int = 0):
    """End the process by EXPLICIT, ORDERED finalization.  Does not return.

    Why this exists (measured, not conjectured).  jax registers an atexit
    hook (``jax._src.api.clean_up``) that destroys every backend client at
    interpreter exit.  After a run that cold-compiled its whole program set
    in-process (~195 XLA:CPU compiles on the fastloop mini-deck), that
    destructor deadlocks in its thread-pool shutdown: the main thread parks
    in a futex join and ~53 ``tf_XLAEigen`` pool workers are left spinning
    on-CPU, indefinitely — 20 minutes measured before the harness timeout
    (jobs 7884928, 7884989 phases 0a/0b; thread census in the 7884989
    artifacts).  Every Python-side duty (the driver's timing table, the
    compile-cache atexit report) completes FIRST; the block is pure C++
    teardown, unreachable from here, and warm runs (0 in-process compiles)
    never hit it — 6 clean bare runs across every FFI dial combination
    (jobs 7884986/7884987).

    So this function performs each remaining teardown duty explicitly, in
    order, and then ends the process with ``os._exit`` so the interpreter
    finalization that would run the deadlocking destructor never starts:

    1. ``jax.effects_barrier()`` — drains the runtime tokens that jax's own
       ``wait_for_tokens`` atexit would have drained.
    2. ``atexit.unregister(jax._src.api.clean_up)`` — the deadlocking hook.
    3. ``jax.distributed.shutdown()`` — the piece of ``clean_up`` that must
       still happen (P>1 coordination service; a no-op at P=1).
    4. ``atexit._run_exitfuncs()`` — every remaining registered duty runs
       NOW: the compile-cache report, the ``impl=mpi`` collectives
       ``Finalize``, h5py cleanup.  Nothing is silently skipped.
    5. One rank-0 line stating what happened, flush, ``os._exit(rc)``.

    The hard exit is not a workaround bolted on a mystery: the same
    mechanism (``os._exit`` after ``main()``) is what the fastloop's interim
    GW_WRAPPER proved green on the certified cold runs (job 7884936) while
    the bare interpreter exit hung.  Call it from a driver's ``__main__``
    block with the return code of ``main()``; ``gw.gw_jax`` — the one chain
    driver measured to hang — is the adopter.
    """
    import atexit
    import sys

    try:
        import jax
        try:
            jax.effects_barrier()
        except Exception as exc:                          # noqa: BLE001
            _print_rank0(f"  [finalize] effects_barrier failed "
                         f"({type(exc).__name__}: {exc}); continuing")
        try:
            from jax._src import api as _jax_api
            atexit.unregister(_jax_api.clean_up)
        except Exception as exc:                          # noqa: BLE001
            _print_rank0(f"  [finalize] could not unregister jax clean_up "
                         f"({type(exc).__name__}: {exc}); the hard exit "
                         f"below still prevents the interpreter-teardown "
                         f"path")
        try:
            jax.distributed.shutdown()
        except Exception as exc:                          # noqa: BLE001
            _print_rank0(f"  [finalize] jax.distributed.shutdown failed "
                         f"({type(exc).__name__}: {exc}); continuing")
    except Exception:                                     # jax never imported
        pass
    try:
        atexit._run_exitfuncs()
    except Exception as exc:                              # noqa: BLE001
        _print_rank0(f"  [finalize] an atexit hook failed "
                     f"({type(exc).__name__}: {exc}); continuing")
    _print_rank0(
        "[runtime] process finalized explicitly (effects barrier, "
        "distributed shutdown, atexit hooks) and ending with os._exit: the "
        "interpreter-teardown destruction of the XLA:CPU client is skipped "
        "deliberately — it deadlocks in pool shutdown after cold-compile "
        "storms (jobs 7884928/7884989).")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(rc))


# ---------------------------------------------------------------------------
#  The startup report:  collect (needs jax)  ->  format (pure)
# ---------------------------------------------------------------------------
#
# SPLIT IN TWO ON PURPOSE.  :func:`collect_startup_facts` needs a live
# backend and a mesh; :func:`format_startup_report` is a pure function of a
# dict and imports nothing, so the WORDING -- which is the product the owner
# reads -- is testable on a login node, with hand-built facts, including the
# fact shapes that only occur on hardware nobody can allocate on demand
# (``platform`` allocator, gloo transport, a failed FFI probe).
#
# THE RULE THE FORMATTER ENFORCES.  Every sentence ends in a period, and
# every choice where more than one outcome was possible is stated even when
# it resolved to the boring one.  "LORRAX_FFT_FFI is off" is not noise: the
# absence of that line is indistinguishable from the flag being on, and the
# whole point of the block is that a performance question can be answered
# from the log without re-running anything.


def _thread_env() -> dict:
    """Thread-count environment + the affinity XLA:CPU actually gets.

    ``sched_getaffinity`` rather than ``cpu_count``: production launches
    under ``taskset -c <lo>-<hi>`` (one 28-core slice per rank on Frontera),
    so ``os.cpu_count()`` reports the whole 56-core node on every rank and
    would make a correctly-pinned run look like a 2x oversubscription.
    """
    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:                                         # noqa: BLE001
        affinity = None
    return {
        "affinity": affinity,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "LORRAX_MKLBLAS_THREADS": os.environ.get("LORRAX_MKLBLAS_THREADS"),
        "LORRAX_SCALAPACK_MKL_THREADS": os.environ.get(
            "LORRAX_SCALAPACK_MKL_THREADS"),
    }


def _enforce_supported_jax(say) -> None:
    """Startup enforcement of the declared JAX window — step 5b.

    Refuses when ``jax.version.__version_info__`` is outside
    ``runtime.jax_support``'s window, or when any ``jax._src`` private that
    ``common/jax_compile_cache.py`` patches has a shape this tree is not
    written against.  Both are startup facts; neither can change mid-run.

    WHY IT IS HERE AT ALL, given that it was deliberately left unwired for a
    day.  ``jax_compile_cache`` used to carry five compatibility shims that
    absorbed exactly these conditions, and while they existed a refusal keyed
    on them would have stopped runs that worked.  Four of the five went with
    jax 0.5.3 support on 2026-08-06.  Unabsorbed, the conditions surface as a
    ``TypeError`` on the first ``jit`` — or, in the variant that cost the most
    time to find, as no surface at all: ``ensure_jax_compile_cache`` printing
    ``enabled=True`` over a cache that wrote zero entries for months.

    Raising here is intended, not startup fragility, and for the same reason
    :func:`_enforce_required_ffi` raises one step later: the fail-fast
    excepthook is already installed (step 0), so a refusal on any rank exits
    the job non-zero with the message at the top of the log.

    ``LORRAX_JAX_UNSUPPORTED_OK=1`` downgrades every refusal to one announced
    line — the module's single declared silence, and it is announced on rank 0
    rather than swallowed.
    """
    _enforce_jax_support(announce=say)


def _enforce_required_ffi(mesh) -> None:
    """Startup enforcement of the REQUIRED FFI layer (decisions.md
    2026-08-01) — step 6b of :func:`initialize_communicator_stack`.

    Runs each capability gate's :meth:`ffi.gate.Gate.enforce` against the
    run's mesh: a missing or unloadable FFI library REFUSES here, at
    startup, quoting ``probe_target``'s three-way reason (which names the
    ``.so`` and the LD_LIBRARY_PATH/build fix), instead of surfacing at the
    first kernel factory mid-run; an explicit ``=0`` refuses (where the
    native duplicate was deleted — LORRAX_FFT_FFI) or announces the
    uncertified debug opt-out (where a native path is structurally
    retained — LORRAX_BANDS_GEMM_FFI, LORRAX_FFT_FFI_FUSED's decomposed
    FFI chain).  Out-of-scope platforms are skipped per each gate's
    declared policy (the platform's native lowering IS the required path
    there, e.g. cuBLAS dot on CUDA for the GEMM dial).

    Raising here is the intended behaviour, not a startup fragility: the
    fail-fast excepthook is already installed (step 0), so a refusal on
    any rank exits the job non-zero with the message at the top of the
    log.  An import failure of the gate modules themselves is a broken
    build and propagates for the same reason.
    """
    from ffi.fft import (CONV_KLEAD_GATE, CONV_KMINOR_GATE, FUSED_GATE,
                         GATE as _FFT_GATE)
    from ffi.gemm import GATE as _GEMM_GATE

    for gate in (_FFT_GATE, FUSED_GATE, _GEMM_GATE):
        gate.enforce(mesh)
    # The CONV_K* dials are ACCELERATOR gates, so enforce() reports their
    # platform/handler capability at startup while each caller resolves the
    # runtime shape through its plan helper.  `auto` never raises; `on`
    # refuses missing platform/handler capability by name; `off` announces
    # the explicit opt-out.  The k-leading member defaults off and has no
    # production consumer until its separately-reviewed Sigma seam lands.
    CONV_KMINOR_GATE.enforce(mesh)
    CONV_KLEAD_GATE.enforce(mesh)


def _ffi_dial_facts() -> list:
    """(env, mode, enabled, detail) for every FFI capability dial.

    Read through the services' own ``Gate`` objects, never by re-parsing the
    variables here: the gate owns the strict grammar (``=Y`` is a grammar
    error, not a silent no-op), so asking it HERE both reports the answer
    and pins the answer every later consumer will key its kernel cache on.
    Since the FFI-required ruling (decisions.md 2026-08-01) the vocabulary
    is two-valued -- ``on``/``off`` answer from the env alone, no probe --
    and the hard availability check is :func:`_enforce_required_ffi`
    (step 6b), which has already run by the time this collector reports.
    """
    out = []
    try:
        from ffi.gemm import GATE as _GEMM_GATE
        from ffi.fft import (CONV_KLEAD_GATE, CONV_KMINOR_GATE,
                             GATE as _FFT_GATE, FUSED_GATE)
    except Exception as exc:                                  # noqa: BLE001
        return [{"env": "<ffi dials>", "mode": None, "enabled": None,
                 "detail": f"the FFI gate modules could not be imported "
                           f"({type(exc).__name__}: {exc})"}]
    for gate, what in ((_GEMM_GATE, "the contract_bands right-GEMM "
                                    "contraction"),
                       (_FFT_GATE, "the flat-k 3-D FFT helper path"),
                       (FUSED_GATE, "the fused IFFT-multiply-FFT tau kernel"),
                       (CONV_KMINOR_GATE,
                        "the fused k-MINOR ifft-multiply-fft conv (the BSE "
                        "ladder-W rung; accelerator)"),
                       (CONV_KLEAD_GATE,
                        "the direct fused k-LEADING IFFT(G)-IFFT(W)-FFT "
                        "conv (Sigma; accelerator, default off)")):
        try:
            mode = gate.mode()
            enabled = gate.enabled()
            detail = what
        except Exception as exc:                              # noqa: BLE001
            mode, enabled = None, None
            detail = f"{what} (gate raised {type(exc).__name__}: {exc})"
        out.append({"env": gate.env, "mode": mode, "enabled": enabled,
                    "default": gate.default, "target": gate.target,
                    "platforms": tuple(gate.platforms), "detail": detail,
                    "off_label": gate.off_label,
                    "off_policy": gate.off_policy,
                    # Declared capability test, for the `auto` sentence above.
                    # Empty for every gate that does not offer the mode.
                    "auto_capability": getattr(gate, "auto_capability", ""),
                    # Captured HERE, not re-read in the formatter: the
                    # formatter is pure, and a report that re-read os.environ
                    # could print a value the gate never saw.
                    "raw": os.environ.get(gate.env)})
    return out


def _linalg_facts(mesh) -> dict:
    """Which distributed dense-linalg backends this mesh can actually serve.

    ``distrib_la.list_backends`` runs the same guards the real call
    will run (platform, compiled capability, one-process-per-device, mesh
    geometry), so "available" here is a promise, not a guess.  It never
    prints and never raises for an availability reason.

    Cost: at most ONE dlopen of the platform FFI library, shared with every
    other probe in the process; the linalg targets are then symbol lookups.
    Which backend a given op *takes* is an input-file key
    (``eigh_backend`` / ``distributed_cholesky``), resolved later against the
    parsed config, so this reports CAPABILITY -- what the machine can do --
    and the config echo reports the choice.
    """
    try:
        from ffi import _services
        _services.ensure_on_path()
        from distrib_la import list_backends as _list_backends
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    out = {}
    for op in ("eigh", "cholesky", "solve_lu"):
        try:
            status = _list_backends(op, mesh)
            out[op] = sorted(b for b, s in status.items()
                             if s.startswith("available"))
        except Exception as exc:                              # noqa: BLE001
            out[op] = None
            out.setdefault("errors", {})[op] = f"{type(exc).__name__}: {exc}"
    return out


def _ffi_library_facts(platform_key: str) -> dict:
    """Which ``.so`` the FFI layer resolved, and whether it loaded."""
    try:
        from ffi.common import ffi_loader
    except Exception as exc:                                  # noqa: BLE001
        return {"platform": platform_key, "path": None, "loaded": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    try:
        ffi_loader.get_lib(platform_key)
        return {"platform": platform_key,
                "path": ffi_loader._loaded_path(platform_key),
                # The BYTES that were dlopened, not the path a shell echoed.
                # See ffi_loader.library_provenance for why a path alone has
                # proved insufficient.
                "provenance": ffi_loader.library_provenance(platform_key),
                "loaded": True, "reason": "",
                "env_var": ffi_loader._PLATFORMS[platform_key]["env"],
                "env_value": os.environ.get(
                    ffi_loader._PLATFORMS[platform_key]["env"])}
    except Exception as exc:                                  # noqa: BLE001
        return {"platform": platform_key, "path": None, "loaded": False,
                "reason": f"{type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:200]}",
                "env_var": "LORRAX_FFI_SO/LORRAX_FFI_HOST_SO",
                "env_value": None}


def collect_startup_facts(mesh, *, cache_error: str | None = None) -> dict:
    """Everything the startup report states, read from the live process.

    Called once, on every rank (cheap; the report is printed from rank 0
    only, but collecting everywhere keeps ``RuntimeStack.facts`` meaningful
    for a per-rank assertion in a test or a probe).

    Every field is wrapped so that a failure to MEASURE something becomes a
    recorded string rather than an exception: a startup report that can kill
    the run would be a worse instrument than no report.  Nothing is
    swallowed -- an unmeasurable field is printed as unmeasurable, with the
    exception text.
    """
    import jax

    f: dict = {}
    f["demotions"] = list(_DEMOTIONS)
    f["distributed_form"] = (_DISTRIBUTED_FORM[-1] if _DISTRIBUTED_FORM else
                             "jax.distributed was already initialised earlier "
                             "in this process")
    f["jax_platforms_env"] = os.environ.get("JAX_PLATFORMS")
    # Read the RESOLVED flag, not the env var: a driver may call
    # ``jax.config.update("jax_enable_x64", True)`` after import, and x64
    # silently off is how complex128 degrades to complex64 everywhere.
    try:
        f["x64"] = bool(jax.config.read("jax_enable_x64"))
    except Exception:                                         # noqa: BLE001
        f["x64"] = bool(getattr(jax.config, "jax_enable_x64", False))
    f["jax_version"] = getattr(jax, "__version__", "unknown")

    f["backend"] = jax.default_backend()
    f["process_index"] = int(jax.process_index())
    f["process_count"] = int(jax.process_count())
    devices = jax.devices()
    local = jax.local_devices()
    f["n_devices"] = len(devices)
    f["n_local_devices"] = len(local)
    f["device_kind"] = local[0].device_kind if local else "unknown"

    f["mesh_shape"] = tuple(int(n) for n in mesh.devices.shape)
    f["mesh_axes"] = tuple(mesh.axis_names)

    # -- CPU collectives transport -----------------------------------------
    impl = (os.environ.get("JAX_CPU_COLLECTIVES_IMPLEMENTATION", "")
            .strip().lower() or None)
    f["collectives"] = {
        "applicable": (f["process_count"] > 1 and f["backend"] == "cpu"),
        "impl": impl or "gloo",
        "impl_was_set": impl is not None,
        "wrapper": os.environ.get("MPITRAMPOLINE_LIB") or None,
        "finalize_fix": os.environ.get("LORRAX_MPI_FINALIZE_FIX") or None,
        "force_thread_main": os.environ.get(
            "LORRAX_MPI_FORCE_THREAD_MAIN") or None,
    }

    # -- allocator: the CLIENT, corroborated against the environment -------
    #
    # LAYERING NOTE (numbered request R9, LANDED 2026-07-31): the two
    # corroboration helpers used to live in ``gw.gw_config``, and this import
    # was lazy and exception-guarded because ``runtime`` must not grow a
    # dependency on ``gw``.  They are now ``runtime.xla_memory``, a sibling
    # module, so the direction is right; the guard stays because a
    # ``memory_stats()`` shape this code has not seen must degrade the REPORT,
    # not kill the run.  The CLIENT read below is unconditional and does not
    # depend on that import: os.environ is a false witness for allocator
    # state (job 7882443 -- identical environ, bytes_limit 11.805 GB vs 0.000
    # GB), so the number in the report always comes from the device.
    stats, stats_err = None, None
    try:
        # local_devices(), not devices(): jax.devices() is the GLOBAL list,
        # so [0] is process 0's device on every rank.
        stats = local[0].memory_stats() if local else None
    except Exception as exc:                                  # noqa: BLE001
        stats_err = f"{type(exc).__name__}: {exc}"
    pool = {"stats": dict(stats) if stats else None, "error": stats_err,
            "corroboration": None, "disagreement": "", "env": None}
    try:
        from .xla_memory import (classify_xla_pool,
                                 resolve_xla_gpu_memory_env)
        xm = resolve_xla_gpu_memory_env()
        reading = classify_xla_pool(stats, backend=f["backend"], env=xm)
        pool["env"] = {
            "allocator": xm.allocator,
            "allocator_raw": xm.allocator_raw,
            "allocator_is_valid": xm.allocator_is_valid,
            "preallocate": xm.preallocate,
            "preallocate_raw": xm.preallocate_raw,
            "preallocate_looks_like_a_typo": xm.preallocate_looks_like_a_typo,
            "mem_fraction": xm.mem_fraction,
            "mem_fraction_var": xm.mem_fraction_var,
            "tf_gpu_allocator_raw": xm.tf_gpu_allocator_raw,
        }
        pool["corroboration"] = reading.peak_source
        pool["disagreement"] = reading.disagreement
        pool["accounting_present"] = reading.accounting_present
    except Exception as exc:                                  # noqa: BLE001
        pool["corroboration"] = "unavailable"
        pool["disagreement"] = (
            f"the environment/client corroboration could not run "
            f"({type(exc).__name__}: {exc}); the figures above are the "
            f"client's, uncorroborated")
    f["pool"] = pool

    # -- FFI ---------------------------------------------------------------
    plat_key = "CUDA" if f["backend"] in ("gpu", "cuda") else f["backend"]
    f["ffi_library"] = _ffi_library_facts(plat_key)
    f["ffi_dials"] = _ffi_dial_facts()
    f["linalg"] = _linalg_facts(mesh)

    # -- threads, cache, guards -------------------------------------------
    f["threads"] = _thread_env()
    try:
        from common.jax_compile_cache import compile_cache_stats
        f["compile_cache"] = compile_cache_stats()
    except Exception as exc:                                  # noqa: BLE001
        f["compile_cache"] = {"error": f"{type(exc).__name__}: {exc}"}
    f["compile_cache_error"] = cache_error
    f["malloc_tune"] = dict(_MALLOC_TUNE)
    import sys as _sys
    f["failfast"] = bool(getattr(_sys, "_lorrax_failfast_installed", False))
    f["failfast_env"] = os.environ.get("LORRAX_FAILFAST")
    return f


_RULE = "=" * 78


def format_startup_report(f: dict) -> list:
    """The rank-0 startup block: a list of complete sentences.

    PURE -- a function of ``f`` alone, so every branch is testable without
    the hardware that produces it.  Sentences, with periods, because the
    owner reads this block to answer performance questions and a table of
    bare tokens does not say WHICH of two things happened or why.
    """
    L = [_RULE, "  LORRAX runtime — resolved startup configuration", _RULE]
    add = L.append

    # -- topology ----------------------------------------------------------
    P = f.get("process_count", 1)
    if P > 1:
        add(f"  This is rank {f['process_index']} of {P}, and it addresses "
            f"{f['n_local_devices']} of the {f['n_devices']} devices in the "
            f"job.")
    else:
        add(f"  This is a single-process run with {f['n_devices']} "
            f"addressable device(s).")
    add(f"  {f['distributed_form']}.")
    add(f"  The JAX platform resolved to {f['backend']!r} on devices of kind "
        f"{f['device_kind']!r}, from JAX_PLATFORMS="
        f"{f.get('jax_platforms_env')!r}, under jax "
        f"{f.get('jax_version')} with 64-bit values "
        f"{'enabled' if f.get('x64') else 'DISABLED'}.")
    gx, gy = (list(f.get("mesh_shape", (1, 1))) + [1, 1])[:2]
    add(f"  The run's device mesh is {gx}x{gy} over axes "
        f"{tuple(f.get('mesh_axes', ()))}, and its communicator cliques were "
        f"warmed before the first physics jit.")
    el = f.get("elapsed")
    if el:
        # ``jax_import`` is stated separately because it is the one piece of
        # this block that is NOT distributed bring-up — it is the Python
        # import storm, and it used to hide inside the next clause.  Optional
        # so a probe that hand-builds ``elapsed`` still formats.
        _ji = el.get("jax_import")
        _ji_clause = (f"{_ji:.1f} s to import jax itself, " if _ji else "")
        add(f"  Bringing this stack up took {el['total']:.1f} s in total: "
            f"{_ji_clause}"
            f"{el['env_and_distributed']:.1f} s for the environment, "
            f"jax.distributed and backend init, "
            f"{el['mesh_and_warmup']:.1f} s to build the mesh and warm its "
            f"communicators, {el['compile_cache']:.1f} s to arm the compile "
            f"cache and {el['measurement']:.1f} s to measure everything in "
            f"this block.")

    # -- demotions ---------------------------------------------------------
    for d in f.get("demotions", ()):
        add(f"  DEMOTION: {d}")
    if P > 1:
        add("  Demotions listed above are this rank's; a demotion that "
            "happened only on another rank is announced in that rank's own "
            "output and does not appear here.")

    # -- transport ---------------------------------------------------------
    c = f.get("collectives", {})
    if not c.get("applicable"):
        if P <= 1:
            add("  There are no cross-process collectives in this run, so "
                "JAX_CPU_COLLECTIVES_IMPLEMENTATION does not apply.")
        else:
            add("  Cross-process collectives run on NCCL because this is a "
                "GPU platform, so JAX_CPU_COLLECTIVES_IMPLEMENTATION does "
                "not apply.")
    elif c.get("impl") == "mpi":
        add(f"  Multi-process CPU collectives run on the MPI implementation "
            f"(JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi) through the "
            f"MPItrampoline wrapper "
            f"{c.get('wrapper') or '<MPITRAMPOLINE_LIB UNSET>'}.")
        if not c.get("wrapper"):
            add("  WARNING: MPITRAMPOLINE_LIB is unset, so MPItrampoline has "
                "no wrapper to load and the MPI transport cannot come up; see "
                "docs/dev/mpi_collectives.md.")
        add(f"  The MPI transport knobs resolved to "
            f"LORRAX_MPI_FINALIZE_FIX={c.get('finalize_fix')!r} and "
            f"LORRAX_MPI_FORCE_THREAD_MAIN={c.get('force_thread_main')!r}; "
            f"the thread-main refusal is handled by warming the cliques from "
            f"the main thread, so the wrapper override is not required.")
    else:
        why = ("set to that value" if c.get("impl_was_set")
               else "unset, and gloo is jax's own default")
        add(f"  WARNING: multi-process CPU collectives resolved to "
            f"{c.get('impl')!r} because "
            f"JAX_CPU_COLLECTIVES_IMPLEMENTATION is {why}.")
        add("  This is not merely slower: gloo's reduce_scatter is measured "
            "to return wrong data silently in about 5% of executions, with "
            "plausible values and a zero exit code, so results from this run "
            "cannot be trusted (docs/dev/mpi_collectives.md).")

    # -- allocator, from the client ---------------------------------------
    pool = f.get("pool", {})
    st = pool.get("stats") or {}
    env = pool.get("env")
    is_gpu = f.get("backend") in ("gpu", "cuda", "rocm")
    if st.get("bytes_limit") or st.get("peak_bytes_in_use") or \
            st.get("bytes_in_use"):
        add(f"  The XLA memory pool, read from "
            f"jax.local_devices()[0].memory_stats() and not from os.environ, "
            f"reports a limit of {st.get('bytes_limit', 0)/1e9:.2f} GB with "
            f"{st.get('bytes_in_use', 0)/1e9:.2f} GB in use and a peak of "
            f"{st.get('peak_bytes_in_use', 0)/1e9:.2f} GB so far.")
    elif pool.get("error"):
        add(f"  The XLA memory pool could not be read from the client "
            f"({pool['error']}), so no allocator figure is reported.")
    elif is_gpu:
        add("  The live client reports no arena accounting at all, so there "
            "is no XLA pool figure for this run and any memory number it "
            "prints later came from an nvidia-smi sample of the whole GPU.")
    else:
        add(f"  The XLA GPU pool knobs do not apply on the "
            f"{f.get('backend')} backend, and that backend keeps no arena "
            f"accounting, so no allocator figure is reported.")
    if env is not None and is_gpu:
        canonical = (not env["preallocate"]) and env["allocator_raw"] is None
        why = (" — LORRAX's canonical pair: preallocation off so the cuFFT "
               "and cuSOLVERMp arenas can allocate outside XLA, and the "
               "allocator left unset because BFC is the only kind that keeps "
               "memory_stats() populated" if canonical else
               " — NOT LORRAX's canonical pair, which is preallocate=false "
               "with the allocator left unset (BFC); a caller overrode it")
        add(f"  XLA_PYTHON_CLIENT_PREALLOCATE resolved to "
            f"{'true' if env['preallocate'] else 'false'} (raw "
            f"{env['preallocate_raw']!r}) and XLA_PYTHON_CLIENT_ALLOCATOR "
            f"resolved to {env['allocator']!r} (raw "
            f"{env['allocator_raw']!r}){why}.")
        if env.get("mem_fraction"):
            add(f"  The XLA client memory fraction is "
                f"{env['mem_fraction']} from {env['mem_fraction_var']}.")
        if env.get("tf_gpu_allocator_raw"):
            add(f"  TF_GPU_ALLOCATOR={env['tf_gpu_allocator_raw']!r} is set "
                f"but is INERT for jax; it selects nothing here.")
        if not env.get("allocator_is_valid"):
            add(f"  WARNING: XLA_PYTHON_CLIENT_ALLOCATOR="
                f"{env['allocator_raw']!r} is not a value jaxlib accepts, and "
                f"jaxlib swallows that error inside CUDA plugin discovery.")
        if env.get("preallocate_looks_like_a_typo"):
            add(f"  WARNING: XLA_PYTHON_CLIENT_PREALLOCATE="
                f"{env['preallocate_raw']!r} reads as 'off' to a human but "
                f"jax's test is case-sensitive, so preallocation is ON.")
    if pool.get("disagreement"):
        add(f"  WARNING: {pool['disagreement']}")

    # -- FFI ---------------------------------------------------------------
    lib = f.get("ffi_library", {})
    if lib.get("loaded"):
        add(f"  The {lib.get('platform')} FFI library loaded from "
            f"{lib.get('path')}"
            + (f", selected by {lib.get('env_var')}."
               if lib.get("env_value") else
               f", which is the in-tree default because "
               f"{lib.get('env_var')} is unset."))
        # The path is not the identity: stage dirs are hand-named and a
        # harness can echo one path and export another.  State the BYTES.
        if lib.get("provenance"):
            add(f"  FFI build provenance: {lib.get('provenance')}")
    else:
        add(f"  No {lib.get('platform')} FFI library could be loaded "
            f"({lib.get('reason')}).  The FFI layer is REQUIRED "
            f"(decisions.md 2026-08-01): startup enforcement refuses this "
            f"state, so a run printing this line bypassed "
            f"initialize_communicator_stack — see "
            f"docs/environment/overview.md for the library build.")
    # The dial sentences are PLATFORM-AWARE: an enabled dial whose platform
    # scope excludes this run's backend is not "routed through the FFI
    # handler" — it does not exist here, and the gate's own enforce() skips
    # it silently by declared policy (Gate.silent_platform_demote).  Found
    # by the P1.2 GPU certification (job 7885151): on a CUDA mesh the block
    # claimed the LORRAX_BANDS_GEMM_FFI contraction rode the FFI handler
    # while the actual GEMMs ride XLA:GPU's native cuBLAS lowering.
    _plat_key = ("CUDA" if f.get("backend") in ("gpu", "cuda")
                 else f.get("backend"))
    for d in f.get("ffi_dials", ()):
        if d.get("mode") is None:
            add(f"  The {d['env']} dial could not be resolved: {d['detail']}.")
            continue
        raw = d.get("raw")
        how = f"set to {raw!r}" if raw not in (None, "") else "unset"
        plats = tuple(d.get("platforms") or ())
        if d["enabled"] and plats and _plat_key not in plats:
            route = (f"rides the platform's native lowering — the dial "
                     f"exists on {'/'.join(plats)} only and this run's "
                     f"backend is {f.get('backend')!r}, where startup "
                     f"enforcement skips it by the gate's declared platform "
                     f"policy (the native lowering IS the required path "
                     f"there)")
        elif d["mode"] == "auto":
            # An OPT-IN ACCELERATOR, not a required layer: `auto` uses the
            # handler where the capability is present and takes the caller's
            # own certified path where it is not.  The startup block is the
            # ONE place that says which happened, so it must not borrow the
            # required-layer sentence, which would claim a route this run may
            # not have taken.
            route = (f"is routed through the FFI handler WHERE AVAILABLE and "
                     f"through {d.get('off_label', 'the native path')} "
                     f"otherwise (capability: {d.get('auto_capability', '')})")
        elif d["enabled"]:
            route = "is routed through the FFI handler (the required layer)"
        elif d.get("off_policy") == "refuse":
            route = ("has NOTHING to run — the native duplicate was "
                     "deleted (decisions.md 2026-08-01) and startup "
                     "enforcement refuses this setting")
        elif d.get("default") == "off":
            # An OPT-IN dial, where `off` is the CERTIFIED state.  The
            # sentence below assumes the required-layer shape (dial defaults
            # ON, so `off` means somebody opted out of the certified path)
            # and would be exactly backwards here: it would call the
            # production path uncertified and imply the run had asked for
            # something.  Keyed on the DEFAULT, not on a name, so the next
            # opt-in dial gets the right sentence for free.
            route = (f"keeps {d.get('off_label', 'the default path')} — "
                     f"this dial is OPT-IN and off is its certified state")
        else:
            route = (f"runs the retained opt-out path "
                     f"({d.get('off_label', 'native lowering')}), which is "
                     f"uncertified for production")
        add(f"  The {d['env']} dial is {how} and resolved to "
            f"{d['mode']}, so {d['detail']} {route}.")
    lin = f.get("linalg", {})
    if lin.get("error"):
        add(f"  The distributed dense-linalg facade could not be queried "
            f"({lin['error']}).")
    else:
        for op in ("eigh", "cholesky", "solve_lu"):
            avail = lin.get(op)
            if avail is None:
                continue
            add(f"  The distributed backends available for {op} on this mesh "
                f"are {', '.join(avail)}; which one runs is the input-file "
                f"key, not an environment variable.")

    # -- threads -----------------------------------------------------------
    t = f.get("threads", {})
    add(f"  This process is pinned to {t.get('affinity')} schedulable CPUs, "
        f"with OMP_NUM_THREADS={t.get('OMP_NUM_THREADS')!r}, "
        f"MKL_NUM_THREADS={t.get('MKL_NUM_THREADS')!r} and "
        f"OPENBLAS_NUM_THREADS={t.get('OPENBLAS_NUM_THREADS')!r}.")
    add(f"  Inside the FFI handlers the BLAS team size is "
        f"LORRAX_MKLBLAS_THREADS={t.get('LORRAX_MKLBLAS_THREADS')!r} for the "
        f"batched GEMM, where unset means the ambient thread count, and "
        f"LORRAX_SCALAPACK_MKL_THREADS="
        f"{t.get('LORRAX_SCALAPACK_MKL_THREADS')!r} for the ScaLAPACK "
        f"handlers, where unset means a cap of 4 because pzheevd measured "
        f"11.28 s per q at 14 MKL threads against 0.463 s at 4.")
    aff = t.get("affinity")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        val = t.get(name)
        if aff and val and val.strip().isdigit() and int(val) > aff:
            add(f"  WARNING: {name}={val} exceeds this process's "
                f"{aff}-CPU affinity, which oversubscribes every threaded "
                f"library in the run.")

    # -- compile cache -----------------------------------------------------
    cc = f.get("compile_cache", {})
    if f.get("compile_cache_error"):
        add(f"  WARNING: the JAX persistent compile cache failed to arm "
            f"({f['compile_cache_error']}), so every rank will compile every "
            f"module in this run.")
    elif cc.get("error"):
        add(f"  The JAX persistent compile-cache state could not be read "
            f"({cc['error']}).")
    elif cc.get("enabled"):
        _np = cc.get("n_proc") or 1
        _share = ("used by this single rank" if int(_np) <= 1 else
                  f"shared by all {_np} ranks through the hit/miss agreement "
                  f"layer")
        add(f"  The JAX persistent compile cache is enabled at "
            f"{cc.get('dir')}, {_share}.")
    else:
        add("  The JAX persistent compile cache is OFF, so every rank "
            "compiles every module in this run; set ISDF_JAX_CACHE_DIR to a "
            "rank-visible directory, or unset it to take the $SCRATCH "
            "default, to turn it on.")
    add("  The cache key includes every array shape, so a system size this "
        "machine has not run before misses every entry no matter how warm "
        "the cache looks.")

    # -- guards ------------------------------------------------------------
    if f.get("failfast"):
        add("  The fail-fast excepthook is installed, so an uncaught "
            "exception on any rank exits the step non-zero instead of "
            "leaving its peers blocked in a collective that rank will never "
            "join.")
    elif P > 1:
        add(f"  WARNING: the fail-fast excepthook is NOT installed "
            f"(LORRAX_FAILFAST={f.get('failfast_env')!r}), so a rank that "
            f"dies can leave the job hanging and still exit rc=0.")
    else:
        add("  The fail-fast excepthook is not installed because a "
            "single-process run already fails with a traceback and rc=1.")
    mt = f.get("malloc_tune", {})
    if mt.get("applied"):
        add(f"  glibc malloc tuning is in force with M_MMAP_THRESHOLD="
            f"{mt.get('mmap_mb')} MB and M_TRIM_THRESHOLD="
            f"{mt.get('trim_mb')} MB, which is the mitigation for the "
            f"per-r-chunk RSS ramp on long XLA:CPU runs.")
    elif mt.get("applied") is False:
        add(f"  WARNING: glibc malloc tuning is NOT in force "
            f"({mt.get('reason')}), so long XLA:CPU runs regain the RSS ramp "
            f"that OOM-killed jobs 7874803, 7875070 and 7875071.")
    else:
        add("  glibc malloc tuning was never attempted in this process, "
            "which means set_default_env() did not run and this report is "
            "describing an environment LORRAX did not configure.")
    L.append(_RULE)
    return L
