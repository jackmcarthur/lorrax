"""Centralised JAX env-var setup + jax.distributed initialization.

Every LORRAX driver (gw.gw_jax, psp.run_nscf, centroid.kmeans_isdf, the
phdf5 plumbing tests, ...) needs the same three things:

  * JAX_ENABLE_X64 = 1 and a sensible JAX_PLATFORMS default
  * ``jax.distributed.initialize()`` called exactly once per process with
    the SLURM-aware argument pattern that actually works on Cray MPICH
    (explicit ``local_device_ids`` derived from CUDA_VISIBLE_DEVICES;
    explicit coordinator from SLURM_NODELIST when the no-args default
    hangs)
  * A fallback to CPU when the GPU backend is unavailable (common in
    sandbox tests without a live CUDA context)

This module owns all three.  Each driver should do::

    from runtime import bootstrap
    bootstrap()          # BEFORE the driver's own ``import jax``
    import jax

:func:`bootstrap` bundles the canonical three-call header
(``set_default_env`` → ``init_jax_distributed`` →
``fallback_to_cpu_if_no_gpu_backend``).  It is importable without pulling
in jax (this module only imports jax lazily, inside functions), and it
sets the env vars before anything imports jax — so as long as the CLI
calls it above its own ``import jax``, the before-import contract holds.
Drivers with a non-standard header (e.g. no CPU fallback) can still call
the three pieces individually.

Previously five different modules had their own copies of this logic,
drifting apart over time (gw.gw_jax had the SLURM-coordinator fallback;
psp.run_nscf and centroid.kmeans_isdf didn't; the phdf5 tests had yet
another flavour).  The sentinel-env-var guard (``_LORRAX_JAX_DISTRIBUTED_DONE``)
now persists across re-imports so the re-entry path
``python -m gw.gw_jax`` → ``gw_init`` imports ``gw.gw_jax`` again and
previously double-initialised no longer does.
"""
from __future__ import annotations

import os
import subprocess


_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"

__all__ = [
    "bootstrap",
    "set_default_env",
    "pin_gloo_interface",
    "init_jax_distributed",
    "fallback_to_cpu_if_no_gpu_backend",
    "install_failfast_excepthook",
]


def bootstrap(*, platform: str = "gpu") -> None:
    """Canonical CLI bootstrap: env defaults + distributed init + CPU fallback.

    One call replaces the three-call header every LORRAX CLI used to
    carry.  MUST run before the caller's own ``import jax``:
    :func:`set_default_env` only works if jax has not been imported yet
    (jax reads its env at import time).  The jax imports *inside*
    :func:`init_jax_distributed` / :func:`fallback_to_cpu_if_no_gpu_backend`
    happen after the env is set, so they are safe.
    :func:`pin_gloo_interface` must run after :func:`set_default_env`
    (it reads the resolved ``JAX_PLATFORMS``) and before anything touches
    ``jax.devices()`` (a backend factory cannot be replaced once the
    backend is initialized) — this slot satisfies both.

    Idempotent (each piece guards itself); no-op-ish in single-process
    runs.  ``platform`` forwards to :func:`set_default_env`.
    """
    set_default_env(platform=platform)
    pin_gloo_interface()
    init_jax_distributed()
    fallback_to_cpu_if_no_gpu_backend()
    install_failfast_excepthook()


def install_failfast_excepthook() -> None:
    """Make an uncaught per-rank exception kill the *job*, not just the rank.

    The exit-code problem this solves
    ---------------------------------
    In a ``jax.distributed`` run the ranks are peers in a collective
    program.  When one rank raises, CPython unwinds it normally: module
    ``atexit`` handlers run, the NCCL/gloo backend tries to tear down, and
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
    if os.environ.get("LORRAX_FAILFAST", "1").strip().lower() in (
            "0", "off", "false", "no"):
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
        try:
            previous(exc_type, exc_value, exc_tb)
        except Exception:
            import traceback
            traceback.print_exception(exc_type, exc_value, exc_tb)
        try:
            sys.stderr.write(
                f"\n*** LORRAX FAIL-FAST: rank {rank}/{n} died with "
                f"{exc_type.__name__}: {exc_value}\n"
                f"*** Exiting rc=1 WITHOUT teardown so this failure reaches "
                f"the job's exit code.  Peer ranks are blocked in a "
                f"collective this rank will never join; srun will now kill "
                f"the step.  (Disable with LORRAX_FAILFAST=0.)\n\n")
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(1)

    sys.excepthook = _failfast
    sys._lorrax_failfast_installed = True


def set_default_env(*, platform: str = "gpu") -> None:
    """Set LORRAX's canonical JAX env defaults.

    Must be called BEFORE ``import jax`` — JAX reads these at import time.
    Uses ``setdefault`` so any caller-provided override wins.

    ``platform="gpu"`` (default) sets ``JAX_PLATFORMS="cuda,cpu"`` so
    JAX tries CUDA and falls back to CPU.  ``platform="cpu"`` forces CPU.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        raise ValueError(f"platform must be 'gpu' or 'cpu', got {platform!r}")
    tune_glibc_malloc()


# glibc mallopt parameter numbers (malloc.h).
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3


def tune_glibc_malloc() -> bool:
    """Pin glibc's mmap/trim thresholds so freed XLA:CPU transients go back
    to the OS.  Returns True when the tuning was applied.

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
    if os.environ.get("LORRAX_MALLOC_TUNE", "1") in ("0", "off", "false"):
        return False
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        mmap_mb = int(os.environ.get("LORRAX_MALLOC_MMAP_MB", "1"))
        trim_mb = int(os.environ.get("LORRAX_MALLOC_TRIM_MB", "128"))
        ok = libc.mallopt(_M_MMAP_THRESHOLD, mmap_mb * 1024 * 1024)
        ok &= libc.mallopt(_M_TRIM_THRESHOLD, trim_mb * 1024 * 1024)
        return bool(ok)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Gloo transport interface pin (workstreams AK/AL, 2026-07)
# ---------------------------------------------------------------------------

_GLOO_PIN_SENTINEL = "_LORRAX_GLOO_PIN_DONE"

# High-speed-fabric name patterns, in preference order: InfiniBand (Frontera
# ib0), then Cray Slingshot (Perlmutter hsn0).  Machine capability, not
# policy: the pin binds the SAME collectives to a different NIC.
_FABRIC_PREFIXES = ("ib", "hsn")

# SIOCGIFADDR — Linux ioctl for an interface's IPv4 (struct ifreq).
_SIOCGIFADDR = 0x8915


def _iface_ipv4(name: str):
    """IPv4 address assigned to interface ``name``, or None (Linux only)."""
    import fcntl
    import socket
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(
            s.fileno(), _SIOCGIFADDR,
            struct.pack("256s", name.encode()[:15]))
        return socket.inet_ntoa(packed[20:24])
    except OSError:
        return None
    finally:
        s.close()


def _iface_is_up(name: str) -> bool:
    """True unless the kernel reports the interface administratively down."""
    try:
        with open(f"/sys/class/net/{name}/operstate") as f:
            return f.read().strip() != "down"
    except OSError:
        return False


def _detect_fabric_iface():
    """(name, ipv4) of the preferred UP high-speed fabric NIC, else None.

    Preference: ``ib*`` before ``hsn*`` (numeric order within a prefix);
    a candidate must be UP and carry an assigned IPv4 — an interface Gloo
    could not actually bind is not a candidate.
    """
    try:
        names = sorted(os.listdir("/sys/class/net"))
    except OSError:
        return None
    for prefix in _FABRIC_PREFIXES:
        for name in names:
            if not (name.startswith(prefix)
                    and name[len(prefix):len(prefix) + 1].isdigit()):
                continue
            if not _iface_is_up(name):
                continue
            addr = _iface_ipv4(name)
            if addr:
                return name, addr
    return None


def pin_gloo_interface() -> None:
    """Bind JAX's Gloo CPU collectives to the high-speed fabric NIC.

    THE PROBLEM (measured, scorecard AK.4/AK.10).  ``jax.distributed`` CPU
    runs use Gloo TCP collectives, and jax 0.9.1 constructs them with no
    ``interface=`` argument (``jax/_src/xla_bridge.py::make_cpu_client``
    calls ``make_gloo_tcp_collectives(distributed_client=...)`` only).
    Gloo then binds the NIC that routes to the coordinator's hostname —
    on Frontera the 1 GbE management NIC ``em1`` (129.114.x.x, MTU 1500),
    not InfiniBand ``ib0`` (192.168.x.x).  ``GLOO_SOCKET_IFNAME`` appears
    nowhere in the shipped jax/jaxlib and is inert.  Measured on the 4x4
    785c deck at P=16 (job 7876536): pinning ``interface=ib0`` is 3.3x on
    the whole pipeline (zeta back-solve 13.7x, sigma.exec 3.5x) with
    eqp0/eqp1/sigma_diag/eqp_g0w0 byte-identical and every compile-only
    stage row at ratio 1.00 — only communication moves.

    THE MECHANISM.  ``xla_bridge.register_backend_factory("cpu", ...)``
    overwrites the stock factory and refuses only once the backend is
    already initialized, so re-registering before anything touches
    ``jax.devices()`` installs a wrapper that builds the Gloo collectives
    with ``interface=`` set and forwards everything else to the stock
    ``make_cpu_client``.

    SCOPE — this is a transport (machine-capability) choice, never physics:
      * single-process runs: silent no-op (no collectives exist);
      * GPU-platform runs (``JAX_PLATFORMS`` not exactly ``cpu``): silent
        no-op (multi-process GPU uses NCCL, whose NIC selection is NCCL's);
      * no ``ib*``/``hsn*`` interface with an IPv4 found: announced no-op,
        stock jax behaviour;
      * jax internals moved / registration refused / bad interface at
        collectives-construction time: LOUD warning, stock behaviour —
        degrade, never crash and never hang.  (A pinned interface with no
        route to a peer fails at the first collective with Gloo's 30 s
        connect timeout — an exception the failfast hook turns into a job
        exit, not a hang; ``LORRAX_GLOO_IFNAME=off`` is the escape hatch.)

    Override: ``LORRAX_GLOO_IFNAME=<name>`` forces the interface,
    ``off``/``none``/``0`` disables the pin.  Either way the decision is
    announced on rank 0 — an env var that silently changed the wire under
    every collective would violate quality-pattern #8.
    """
    if os.environ.get(_GLOO_PIN_SENTINEL):
        return
    os.environ[_GLOO_PIN_SENTINEL] = "1"

    if _resolve_proc_count() <= 1:
        return                          # no cross-process collectives at all
    if os.environ.get("JAX_PLATFORMS", "").strip().lower() != "cpu":
        return                          # GPU path: collectives are NCCL's

    rank0 = _resolve_proc_id() == 0

    def _say(msg: str) -> None:
        if rank0:
            print(f"[runtime] {msg}", flush=True)

    override = os.environ.get("LORRAX_GLOO_IFNAME")
    if override is not None and override.strip().lower() in (
            "off", "none", "0", ""):
        _say("Gloo interface pin DISABLED by LORRAX_GLOO_IFNAME="
             f"{override!r}: jax default transport (binds the NIC that "
             "routes to the coordinator — the 1 GbE management NIC on "
             "Frontera).")
        return

    if override is not None:
        iface = override.strip()
        addr = _iface_ipv4(iface)
        if addr is None or not _iface_is_up(iface):
            _say(f"WARNING: LORRAX_GLOO_IFNAME={iface!r} is not an UP "
                 "interface with an IPv4 on this node — Gloo interface pin "
                 "SKIPPED, stock jax transport (likely the 1 GbE management "
                 "NIC).")
            return
        why = "LORRAX_GLOO_IFNAME override"
    else:
        found = _detect_fabric_iface()
        if found is None:
            _say("Gloo interface: jax default (no UP ib*/hsn* interface "
                 "with an IPv4 found on this node).")
            return
        iface, addr = found
        why = "auto-detected high-speed fabric"

    try:
        from jax._src import config as _jax_config
        from jax._src import distributed as _jax_dist
        from jax._src import xla_bridge as _xb
        from jax._src.lib import xla_client as _xc

        _stock_make_cpu_client = _xb.make_cpu_client
        _make_gloo = _xc._xla.make_gloo_tcp_collectives   # AttributeError if moved

        def _pinned_cpu_client(collectives=None):
            if (collectives is None
                    and _jax_dist.global_state.client is not None
                    and _jax_config.cpu_collectives_implementation.value
                        == "gloo"):
                try:
                    collectives = _make_gloo(
                        distributed_client=_jax_dist.global_state.client,
                        interface=iface,
                    )
                except Exception as exc:
                    _say(f"WARNING: could not build Gloo collectives on "
                         f"{iface!r} ({type(exc).__name__}: {exc}); falling "
                         "back to the stock transport.")
                    collectives = None
            return _stock_make_cpu_client(collectives)

        _xb.register_backend_factory(
            "cpu", _pinned_cpu_client, priority=0, fail_quietly=False)
    except Exception as exc:
        _say(f"WARNING: Gloo interface pin unavailable "
             f"({type(exc).__name__}: {exc}) — jax internals moved or the "
             "backend is already initialized; stock jax transport (on "
             "Frontera that is the 1 GbE management NIC). Update "
             "runtime.pin_gloo_interface for this jax version.")
        return

    _say(f"Gloo collectives pinned to {iface} ({addr}; {why}). The jax "
         "default binds the coordinator-route NIC — em1, 1 GbE, on "
         "Frontera compute nodes. Override/disable: LORRAX_GLOO_IFNAME.")


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
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return

    proc_count = _resolve_proc_count()
    if proc_count <= 1:
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return

    import jax

    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_local = len([x for x in cv.split(",") if x.strip()]) if cv else 0
    init_kwargs = {"local_device_ids": list(range(n_local))} if n_local else {}
    if not os.environ.get("JAX_COORDINATOR_ADDRESS"):
        try:
            jax.distributed.initialize(**init_kwargs)
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass

    # ``local_device_ids`` matters on BOTH paths: without it the explicit form
    # assumes each process owns every local GPU and dies with
    # "CUDA_ERROR_INVALID_DEVICE: invalid device ordinal" under the one-GPU-
    # per-process binding select_gpu.sh sets up.
    jax.distributed.initialize(
        coordinator_address=_resolve_coordinator_address(),
        num_processes=proc_count,
        process_id=_resolve_proc_id(),
        **init_kwargs,
    )
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


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
    """
    import jax
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
    try:                       # drop the half-initialised cuda backend cache
        jax.clear_backends()
    except Exception:
        pass
