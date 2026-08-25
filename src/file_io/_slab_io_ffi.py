"""FFI SlabIO backend — collective MPI-IO via ``ffi.phdf5``.

THE transport.  Not a tier: the capability router, the sibling tiers
(``PHDF5_HOST``, ``H5PY_ALLGATHER``), the ``slab_io`` and ``use_ffi_io``
deck keys and the ``SlabIOBackend`` enum were all deleted in the
one-backend port.  :mod:`file_io.slab_io` has one path and it is this
one; if ``liblorrax_ffi*.so`` is missing or lacks the write symbol,
SlabIO REFUSES naming the library rather than moving the bytes some
other way.  Nothing falls back, because the thing it used to fall back
to gathered a global array onto one rank -- an OOM at the design
envelope, not a slow path.

PLATFORM-AGNOSTIC.  Nothing in this module is CUDA-specific: the
``jax.ffi.ffi_call`` sites name only the target string, and
``ffi_loader`` registers ``liblorrax_ffi.so``'s handlers under
platform="CUDA" and ``liblorrax_ffi_host.so``'s under platform="cpu"
against those same strings.  So this backend drives the GPU collective
write and (since workstream AE) the host one — on CPU the C++ side
skips the D2H staging entirely and H5Dwrite reads the XLA buffer in
place.  The capability probe for it lives in ``ffi_loader.has_phdf5_write``;
``gw_config._route_cpu_slab_io``, which used to consume that probe to
pick a tier, was deleted with the router.

Every operation derives per-rank hyperslab offsets from the sharding
spec of the JAX array being written (or a caller-provided one for
reads) plus a global-origin ``offset`` argument.  The C++ handler
un-ravels the rank id through ``mesh_shape`` and advances along every
sharded dim.  See ``ffi/cpp/phdf5/write_ffi.cc`` for the C++ side.
"""
from __future__ import annotations

import functools
import os
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from common.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import barrier as _barrier, device_put_process_local
from runtime import debug_print_enabled

from . import h5_journal as _journal

#: The two HDF5 library instances, spelled as ``file_io.hdf5_owner`` and
#: ``file_io.h5_journal`` spell them.  Every HDF5 call this module makes
#: is journaled AT ISSUE under one of them — the collective transport
#: under ``ffi``, the two serial-h5py metadata touches under ``h5py`` —
#: so the per-rank journal shows both libraries' traffic on one file in
#: one stream (SLAB_IO_ROOT_CAUSE_AUDIT.md §C).
_J_FFI = "ffi"
_J_H5PY = "h5py"


def _rank0() -> bool:
    return jax.process_index() == 0


# ``_barrier`` is ``common.collectives.barrier``.  It used to be a local
# copy whose whole body was ``try: sync_global_devices(tag); except
# Exception: pass`` — seven lines below an import of the very module that
# owns the correct one.  The swallow is the defect: every use of it here
# guards a WRITE ordering (the inode replacement before the collective
# open, the rank-0
# deferred-attr write before close), so a barrier that silently did
# nothing would let a rank read a file whose stripe layout or metadata
# the writer has not finished changing — a data defect that reports rc=0.
# ``collectives.barrier`` instead returns False for the legitimate
# single-process no-op and RAISES (after naming the barrier) when a real
# multi-process barrier fails.


# ---------------------------------------------------------------------------
# Lustre striping policy — ONE function, used by every writer
# ---------------------------------------------------------------------------
#
# THE MECHANISM.  Cray ROMIO picks its collective-buffering aggregator count
# as ``cb_nodes = min(striping_factor, nranks)`` unless cb_nodes is set
# explicitly (it is not — hand-set cb_nodes measured 30% slower, slab_io.md
# §Tuning).  So the stripe count IS the aggregator count.  A FIXED stripe
# count therefore pins the aggregator count forever, and every rank added
# past it is a rank that does not aggregate.  The old default of 16 was
# tuned on a payload that happened to run at 16 ranks, where 16 is also
# exactly ``nranks`` — the coincidence that made a constant look like a
# tuning.
#
# MEASURED (job 56389339 + 16/25-node steps, /pscratch, GiB/s aggregate
# write, phase-separated harness ``tests/bench/slabio_scaling_bench.py``):
#
#   ranks  payload      stripe=16 x 1M   stripe=nranks         gain
#      4     2.00 GiB      0.813            0.906  (4 x 1M)    +11%
#     16     8.00 GiB      3.152            3.152  (16 x 1M)     0%   [1]
#     64    32.00 GiB      5.189           10.630  (64 x 4M)   +105%
#     64   381.47 GiB      7.403           13.222  (64 x 4M)    +79%   [2]
#    100    50.00 GiB      7.872           15.216  (100 x 4M)   +93%
#
#   [1] at 16 ranks the two policies are the SAME configuration; the row is
#       the null control, not a win.
#   [2] the real envelope payload: V_qmunu, (nq=64, 20000, 20000) c128.
#       The old default leaves 44% of the write bandwidth on the floor.
#
# ``stripe=nranks`` won or tied at EVERY rank count measured.
#
# STRIPE UNIT.  1 MiB at <=16 ranks and 4 MiB at >=64 ranks are the two
# measured anchors (64 x 1M = 7.068 vs 64 x 4M = 10.630 at 64 ranks;
# 16 x 4M = 2.07 vs 16 x 1M = 2.93 at 16 ranks).  Between them the unit is
# the power of two NEAREST IN LOG2 to ``nranks/16`` MiB — a geometric ramp
# with no free parameters, anchored at both measured ends, rather than a
# step at 64 chosen for convenience.  It matters that it is not a step:
# a 32-rank run sits halfway between the anchors in exactly the sense that
# log2 measures, and the ladder cannot even test it (``resolve_mesh``
# requires a perfect-square device count, so 32 is not a legal geometry —
# the interpolation region 16 < n < 64 contains NO power-of-two rank count
# that is also a legal mesh).  A rule that cannot be measured at its
# midpoint should at least be the one the two endpoints imply.
#
# WHY THE UNIT MUST STAY <= 4 MiB.  The per-rank tile knee is 4 MiB
# (job 56389339, 16 ranks: tile/rank 0.06/0.25/1/4/16 MiB gave
# 0.126/0.609/1.779/3.165/3.551 GiB/s — flat above 4 MiB, a cliff below).
# At the envelope, ``V_qmunu`` at N_mu=20000 on 1024 ranks is ~6.1 MiB per
# rank per slab: only 1.5x above the knee.  A stripe unit larger than the
# per-rank tile starves aggregators, so 4 MiB is a ceiling, not a trend to
# extrapolate.
#
#: Clamp on the stripe count.  Lower bound: below 4 the file is on too few
#: OSTs to reach even 1 GiB/s (1 x 1M measured 0.61-0.75 GiB/s at every
#: rank count from 4 to 64 — a per-FILE single-OST ceiling that adding
#: ranks does not move).  Upper bound: see _STRIPE_COUNT_MAX.
_STRIPE_COUNT_MIN = 4
_STRIPE_COUNT_MAX = 128

_STRIPE_UNIT_MIN = 1 << 20
_STRIPE_UNIT_MAX = 4 << 20


#: The boolean env grammar, ONE copy.  Mirrors the C++ writer's
#: ``env_flag`` (``ffi/cpp/phdf5/context.cc``) so the Python and C++ halves
#: of the phdf5 writer stay one grammar. This file once spelled the same
#: tuple inline three times; keeping one local copy avoids an uphill
#: L3 -> L1 import.
# ONE PARSER (2026-08-22).  These three used to be a local copy of the
# grammar, and the copy SWALLOWED: an unrecognised token resolved to False
# with nothing printed, so a typo in a knob name's VALUE turned a
# default-on knob off in silence.  ``runtime.env_flags`` (L3, no jax) owns
# the table and the once-per-value announcement now; the local tuples were
# a deliberate duplicate to avoid an uphill L3 -> L1 import into
# ``gw_config``, and moving the grammar down removes the reason for it.
from runtime.env_flags import ENV_FALSE as _FALSE   # noqa: F401
from runtime.env_flags import ENV_TRUE as _TRUE     # noqa: F401
from runtime.env_flags import env_bool as _env_flag  # noqa: F401


#: Paths already announced as taking the legacy serial-h5py introspect.
#: One line per path per process, not one per dataset: the fact being
#: reported is a property of the DEPLOYED LIBRARY, and repeating it per
#: dataset would bury it under itself.
_LEGACY_INTROSPECT_ANNOUNCED: set = set()


def _announce_legacy_introspect(path: str) -> None:
    """Say, once, that this file's geometry came from the OTHER HDF5 stack.

    The FFI has owned dataset introspection since 2026-08-22
    (``lrx_phdf5_dataset_geometry``).  A library built before that exports
    no such entry point, so a read-only handle still falls back to a
    serial-h5py open of the same path — legal, counted by
    ``file_io.hdf5_owner``, and exactly the cohabitation the metadata
    entry points exist to retire.  Announce-or-refuse: the run does not
    get to take the old route silently.
    """
    if path in _LEGACY_INTROSPECT_ANNOUNCED:
        return
    _LEGACY_INTROSPECT_ANNOUNCED.add(path)
    if not debug_print_enabled() or not _rank0():
        return
    print(f"  [SlabIO] {os.path.basename(path)}: dataset geometry read "
          f"through SERIAL h5py — the loaded FFI library predates "
          f"lrx_phdf5_dataset_geometry (2026-08-22), so two HDF5 library "
          f"instances touch this path.  Read-only on both sides, so it is "
          f"allowed and counted; a handle that could WRITE would refuse "
          f"here.  Rebuild to retire it: src/ffi/cpp/build.sh (CUDA leg), "
          f"config/perlmutter/build_ffi_host.sh (host leg).", flush=True)


def _close_log_level() -> int:
    """Return 0=quiet production close or 2=driver debug detail.

    Storage-library progress is debug detail.  Worker errors remain
    unconditional below, but healthy drains, joins and ``H5Fclose`` calls do
    not belong in a production physics log.
    """
    return 2 if debug_print_enabled() else 0


def _stripe_policy(nranks: int) -> tuple[int, int]:
    """``(striping_factor, striping_unit_bytes)`` for a world of ``nranks``.

    Pure function of the rank count — no env, no MPI, no filesystem — so
    the policy can be tested at rank counts no allocation can reach.
    See the block comment above for the measurements behind every number.
    """
    n = max(1, int(nranks))
    count = min(max(n, _STRIPE_COUNT_MIN), _STRIPE_COUNT_MAX)
    # Unit: the power of two nearest in log2 to (nranks/16) MiB, clamped.
    # Integer arithmetic, no floats: double the unit each time nranks
    # passes the geometric midpoint between the current anchor and the
    # next (16*sqrt(2) ~ 22.6, 32*sqrt(2) ~ 45.3).  Written as an
    # exact integer comparison so it cannot drift with libm.
    unit = _STRIPE_UNIT_MIN
    while unit < _STRIPE_UNIT_MAX and 2 * n * n > (
            (unit // _STRIPE_UNIT_MIN) * 32) ** 2:
        unit *= 2
    return count, unit


def _stripe_count(nranks: int | None = None) -> int:
    """Lustre ``striping_factor`` in force: the policy, or the env override.

    ``LORRAX_PHDF5_STRIPE_COUNT`` overrides :func:`_stripe_policy`; unset
    or empty means the policy.  Anything else must be a plain int.  A typo
    here used to crash with a bare ``ValueError`` while the sibling
    ``LORRAX_PHDF5_STRIPE_SIZE_FS`` was silently replaced by ITS default —
    two neighbouring knobs, opposite failure modes (audit 2026-07-28).
    Both refuse loudly, naming the variable and the accepted grammar.

    ``-1`` ("stripe over every OST") is REFUSED rather than passed
    through.  It is the one value that looks like a maximum and measures
    like a failure: 0.105 GiB/s at 64 ranks / 32 GiB and 1.118 at 16
    ranks / 8 GiB, against 10.63 and 3.15 for the policy — 100x and 3x
    slower respectively (job 56389339, ``w_SC64``/``w_SC16``).  A
    number-of-OSTs-wide layout puts every rank on every OST, which is the
    maximum-contention arrangement, not the maximum-bandwidth one.
    """
    raw = os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "").strip()
    if not raw:
        return _stripe_policy(
            jax.process_count() if nranks is None else nranks)[0]
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(
            f"LORRAX_PHDF5_STRIPE_COUNT={raw!r} is not a valid stripe "
            f"count: expected a plain integer (e.g. 16; 0 disables the "
            f"striping hints; -1 is refused, see below).") from None
    if val < 0:
        raise ValueError(
            f"LORRAX_PHDF5_STRIPE_COUNT={raw!r} is refused.  A negative "
            f"striping_factor means 'every OST on the filesystem', which "
            f"is the maximum-CONTENTION layout and measures like one: "
            f"0.105 GiB/s at 64 ranks writing 32 GiB, and 1.118 GiB/s at "
            f"16 ranks writing 8 GiB, against 10.63 and 3.15 for "
            f"stripe=nranks (job 56389339).  Pass a positive count, or "
            f"unset the variable and let the policy pick "
            f"min(max(nranks, {_STRIPE_COUNT_MIN}), {_STRIPE_COUNT_MAX}).")
    return val


def _stripe_size_bytes(nranks: int | None = None) -> int:
    """Lustre ``striping_unit`` in force, in bytes.

    ``LORRAX_PHDF5_STRIPE_SIZE_FS`` is the ``lfs setstripe -S`` spelling
    ("4M") and overrides :func:`_stripe_policy`; MPI-IO's ``striping_unit``
    hint wants bytes, so accept both spellings and normalise here.
    Malformed input REFUSES with the grammar: it used to be silently
    replaced by the 4 MiB default, so an explicit ``=4MiB`` A/B experiment
    quietly measured the default configuration (doctrine 3; audit
    2026-07-28 — the sibling ``LORRAX_PHDF5_STRIPE_COUNT`` refuses the
    same way).

    Lives here, beside the count, since 2026-08-06: the two are one
    policy and were previously resolved in two modules with two
    different notions of "the default".
    """
    raw = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE_FS", "").strip()
    if not raw:
        legacy = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE", "").strip()
        if legacy:
            try:
                v = int(legacy)
            except ValueError:
                raise ValueError(
                    f"LORRAX_PHDF5_STRIPE_SIZE={legacy!r} is not a valid "
                    f"byte count: expected a plain integer.") from None
            if v > 0:
                return v
        return _stripe_policy(
            jax.process_count() if nranks is None else nranks)[1]
    mult = 1
    base = raw
    if raw and raw[-1] in "kKmMgG":
        mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[raw[-1].lower()]
        base = raw[:-1]
    try:
        return int(float(base) * mult)
    except ValueError:
        raise ValueError(
            f"LORRAX_PHDF5_STRIPE_SIZE_FS={raw!r} is not a valid stripe "
            f"size: expected '<number>[k|M|G]' in the `lfs setstripe -S` "
            f"single-letter-suffix grammar (e.g. '4M', '512k'; "
            f"'MiB'/'MB' spellings are not accepted).") from None


def _export_striping_env(nranks: int | None = None) -> tuple[int, int]:
    """Materialise the resolved striping into ``os.environ``; return it.

    WHY THIS EXISTS, and why it is not spooky action.  The FFI writer's
    hints are built in C++ (``ffi/cpp/phdf5/context.cc``) whose only
    input from an operator is ``getenv``: ``ffi.io.open_file`` passes a
    path, a mesh and a mode, so an explicit ``LORRAX_PHDF5_STRIPE_*``
    override has no other channel.  Exporting the resolved values before
    ``open_file`` is that channel.

    Consequences, deliberately:

    * an operator's explicit ``LORRAX_PHDF5_STRIPE_*`` still wins — it is
      read by the resolvers above and simply re-exported unchanged;
    * both writers see one value, so "the environment" remains a
      complete description of the layout, which is what makes the
      ``lfs getstripe`` readback in the bench harness meaningful;
    * the value is visible to anything that dumps ``os.environ`` in a run
      log, instead of being implicit in a library nobody can grep.

    WHAT THIS IS NO LONGER DOING (2026-08-06).  It used to be the ONLY
    thing keeping the two writers on one layout: with the variable unset,
    ``context.cc`` fell back to the literals ``"16"`` and 1 MiB while
    :func:`_stripe_policy` resolved ``clamp(nranks, 4, 128)`` and the
    1→4 MiB ramp, and the disagreement was masked purely by the fact
    that ``_FfiBackend.__init__`` calls this immediately before
    ``_open_file()``.  A caller reaching ``ffi.io.open_file`` DIRECTLY
    got the old constants, silently.  The C++ now derives the same
    policy from ``ctx->world_size`` (``stripe_policy_count`` /
    ``stripe_policy_unit`` there transcribe ``_stripe_policy`` here, and
    ``tests/test_slab_io_routing.py`` compiles them and diffs the two
    over 0..4100 ranks), so the agreement is by construction and this
    function is back to being only what its name says: the operator
    override's channel to C++, and a run log's record of the layout.
    """
    count = _stripe_count(nranks)
    unit = _stripe_size_bytes(nranks)
    os.environ["LORRAX_PHDF5_STRIPE_COUNT"] = str(count)
    # Re-export in the `lfs setstripe -S` spelling both parsers accept, so
    # a run log's environment reads the way the docs and `lfs` do.  A unit
    # that is not a whole number of MiB (only reachable by an explicit
    # override like '512k') keeps the byte spelling, which both parsers
    # also accept — an exact value beats a pretty one.
    os.environ["LORRAX_PHDF5_STRIPE_SIZE_FS"] = (
        f"{unit >> 20}M" if unit >= (1 << 20) and unit % (1 << 20) == 0
        else str(unit))
    return count, unit


def file_stripe_layout(path: str) -> tuple[int, int] | None:
    """The stripe layout the FILE ACTUALLY HAS, or ``None`` if unknowable.

    This is a DIFFERENT question from :func:`_stripe_count`, which is the
    layout we would REQUEST.  A Lustre layout is a property of the inode,
    fixed at create (see :func:`_replace_inode_for_write`), so on a
    ``mode='r'`` open of a file LORRAX did not write the policy is inert
    and *this* is the only number that governs the read.  Every ``WFN.h5``
    is in that class: ``pw2bgw`` creates it, not us.

    WHY THIS EXISTS.  ``striping_factor`` sets ``cb_nodes =
    min(striping_factor, nranks)``, so a one-stripe file pins the read to a
    SINGLE aggregator at any rank count.  Measured 2026-08-06, ``lfs
    getstripe -c -S`` on two production files --
    ``pre_august/ZG_yifan/WFN.h5`` and ``int0807_art/bse/WFN.h5`` -- both
    return ``stripe_count 1, stripe_size 1 MiB``, which is also
    ``/pscratch``'s directory default.  The ledger's cold-read ladder at 64
    ranks puts a 1-stripe file at ~0.6-0.8 GiB/s against 5.6 for a striped
    one.  No LORRAX code change reaches it -- the remedy is ``lfs setstripe``
    on the deck directory BEFORE ``pw2bgw``, or ``lfs migrate`` after.

    Nothing in the tree reported this before: ``_FfiBackend.__init__`` logs
    striping only on ``mode='w'``, deliberately, because on a read the
    POLICY is meaningless.  The file's own layout is not, and it was the
    dominant term nobody could see.

    Returns ``(stripe_count, stripe_size_bytes)``.  ``None`` when ``lfs`` is
    absent, the path is not on Lustre, or the call fails for any reason --
    this is an observation, never a gate, and must not break a read.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["lfs", "getstripe", "-c", "-S", str(path)],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode != 0:
            return None
        nums = [int(tok) for tok in out.stdout.split() if tok.isdigit()]
        # `-c -S` prints the count then the size, one integer each.
        return (nums[0], nums[1]) if len(nums) >= 2 else None
    except Exception:
        return None


def _replace_inode_for_write(path: str) -> None:
    """Rank-0 unlink + barrier so ``mode='w'`` REPLACES the file's inode.

    Called by the one PHDF5 writer before any rank opens the file,
    UNCONDITIONALLY of ``lfs`` availability. Rationale: a Lustre
    stripe layout is a property of the INODE, fixed at create time —
    ``lfs setstripe``, MPI-IO's ``striping_factor`` hint and
    ``H5Fcreate(H5F_ACC_TRUNC)`` are all no-ops against an existing
    inode, so a rerun over an existing 1-stripe file keeps 1 stripe
    forever (measured: job 7876423 funnelled 13.3 GB of ``V_qmunu``
    through a single OST).  The only unlink used to live inside the
    deleted ``lfs setstripe`` prestripe helper AFTER its lfs-missing
    early return, i.e. it never ran in the production apptainer image
    (audit 2026-07-28).

    ``os.path.lexists`` (not ``exists``) so a dangling symlink is also
    replaced; a live symlink is announced before removal because the
    new file lands at ``path`` itself, not at the link's old target.
    A failed unlink RAISES rather than falling through: proceeding
    would H5F_ACC_TRUNC the old inode and silently inherit its stripe
    layout — ``mode='w'`` is a replace contract.
    """
    if _rank0() and os.path.lexists(path):
        if os.path.islink(path):
            try:
                target = os.readlink(path)
            except OSError:
                target = "<unreadable>"
            print(f"  [SlabIO] mode='w': {path} is a symlink -> "
                  f"{target!r}; removing the LINK — the new file is "
                  f"created at {path}, not at the old target.",
                  flush=True)
        try:
            os.remove(path)
        except OSError as e:
            raise OSError(
                f"SlabIO mode='w': could not replace existing file "
                f"{path!r}: {e}.  Refusing to open with H5F_ACC_TRUNC "
                f"instead — truncation reuses the inode, so the file "
                f"would silently keep its existing Lustre stripe layout "
                f"and ignore the striping_factor/striping_unit hints "
                f"(the 1-stripe single-OST defect, job 7876423).  "
                f"Delete the file or fix its permissions, then rerun."
            ) from e
    _barrier("slab_io_replace_inode")


# ---------------------------------------------------------------------------
# The MPI world this context ACTUALLY has, vs the one it was opened for
# ---------------------------------------------------------------------------
#
# MEASURED, job 56389339, 4 nodes / 16 ranks, ``srun --mpi=pmi2`` (the wrong
# PMI flavour for this stack; the right one is ``cray_shasta``):
#
#     MPI_Comm_size(MPI_COMM_WORLD) == 1 on every rank
#     jax.process_count()           == 16
#     ... 8 hostile geometries written and read back BIT-EXACT, rc=0,
#         file 16-striped and fully populated, no warning anywhere.
#
# Nothing in the stack noticed.  ``ffi.io.open_file`` checks ``p*q ==
# jax.process_count()``, and ``shard_index.h::validate_shard_encoding`` checks
# ``prod(mesh_shape) == ctx->world_size`` — but ``ctx->world_size`` is
# ``jax.process_count()``, passed down from Python.  Both checks compare JAX
# to JAX and agree.  The MPI communicator that H5Dwrite actually collects on
# is never consulted, so a PMI-mismatched launch gives every rank a private
# singleton MPI_COMM_WORLD and 16 unsynchronised writers on one file.
#
# It "works" only because the hyperslabs happen to be disjoint and each rank
# is writing its own byte ranges: there is no collective handshake left to
# fail.  Change the geometry so two ranks touch one HDF5 chunk, or let one
# rank's metadata update race another's, and it is silent corruption with
# rc=0 — which is precisely the class of defect the collective path exists
# to make impossible.
#
# With collective writes ON (the current default) the same launch does NOT
# survive, but it does not diagnose either: it dies inside Cray ROMIO as
#     Out of memory in .../ad_cray/ad_cray_write_coll.c, line 669
# followed by MPI_Abort and "HDF5: infinite loop closing library".  That is
# the SAME line ``stage/phdf5_stage_cray.sh`` documents as a known Cray-MPICH
# >=1 GB/rank OOM whose remedy is ``LORRAX_PHDF5_COLLECTIVE_WRITES=0`` /
# ``LORRAX_PHDF5_INDEPENDENT=1``.  An operator who follows that documented
# remedy converts the loud crash into the silent-wrong-answer regime above.
# So the misdiagnosis is not hypothetical; the tree points at it.
#
# Hence: ask MPI, once, at the first collective open.  The verdict is
# rank-invariant by construction (every rank compares its own
# MPI_Comm_size against a replicated jax.process_count()), so it refuses
# everywhere or nowhere — the only kind of refusal a collective tolerates.
#
# CALLED ON EVERY PATH THAT OPENS A FILE COLLECTIVELY.  Both of them:
# ``_FfiBackend.__init__`` (tier 1) and ``_MpiHostBackend.__init__``
# (tier 2).  Tier 2 had no guard until 2026-08-06 even though it is a
# routine multi-process route — a guard installed on one of two doors is
# a guard on neither.  The remaining ``h5py.File`` opens in this package
# are rank-0-only serial handles (``_introspect_dataset``, the deferred
# attribute write in ``close``); they derive no per-rank hyperslab and
# collect on nothing, so there is no MPI world for them to disagree with.
_MPI_WORLD_VERDICT: dict[str, tuple[str, str]] = {}


def _mpi_world_verdict(mpi_size, proc_count, probe_detail, *, require,
                       tier="PHDF5_FFI"):
    """Pure decision function for the MPI-world guard — ``(verdict, msg)``.

    Split out from the ctypes probe so it can be tested without an MPI.
    ``verdict`` is one of ``"ok"``, ``"unprobed"``, ``"refuse"``.

    ``require`` gates the UNPROBED case only, and the MISMATCH case below
    is unconditional.  That asymmetry is deliberate — "we could not check"
    and "we checked and it is broken" are different facts — but the
    mismatch message used to end with "LORRAX_PHDF5_REQUIRE_MPI_WORLD=0
    downgrades this to a warning", which was never true of that branch
    (found 2026-08-06).  A refusal that advertises an escape hatch it
    does not implement is worse than a bare one: it sends the reader
    looking for a variable that cannot help, on a failure whose whole
    point is that it is invisible.  The sentence is gone rather than
    implemented, because CLAIMS 68 measured what proceeding costs — with
    a private singleton ``MPI_COMM_WORLD`` per rank the 8 hostile
    geometries wrote and read back bit-exact at rc=0, correctly striped,
    and only because disjoint hyperslabs need no collective handshake.
    Two ranks on one chunk would be silent corruption. There is nothing
    to downgrade to.
    """
    if mpi_size is None:
        msg = (f"SlabIO {tier}: could not verify the MPI world size "
               f"({probe_detail}).  jax.process_count()={proc_count}; if the "
               f"launcher's PMI flavour does not match this MPI stack every "
               f"rank gets a private singleton MPI_COMM_WORLD and the "
               f"collective write silently degrades to {proc_count} "
               f"unsynchronised writers on one file.  Set "
               f"LORRAX_PHDF5_REQUIRE_MPI_WORLD=1 to make this a refusal.")
        return ("refuse" if require else "unprobed"), msg
    if int(mpi_size) == int(proc_count):
        return "ok", (f"MPI_Comm_size={mpi_size} == "
                      f"jax.process_count()={proc_count} ({probe_detail})")
    return "refuse", (
        f"SlabIO {tier}: MPI_Comm_size(MPI_COMM_WORLD)={mpi_size} but "
        f"jax.process_count()={proc_count} ({probe_detail}).  This context "
        f"would derive {proc_count} distinct per-rank hyperslabs and hand "
        f"them to a communicator of {mpi_size} process(es), so the "
        f"'collective' write has no collective in it: every rank writes "
        f"alone, nothing synchronises the HDF5 metadata, and the result is "
        f"wrong-with-rc=0 whenever two ranks touch one chunk.  The usual "
        f"cause is the launcher's PMI flavour: on Perlmutter/Cray MPICH use "
        f"`srun --mpi=cray_shasta`, NOT --mpi=pmi2.  Refused identically on "
        f"every rank, with NO downgrade: LORRAX_PHDF5_REQUIRE_MPI_WORLD "
        f"gates only the 'could not probe' case, and CLAIMS 68 measured "
        f"that proceeding here yields bit-exact-looking output at rc=0 "
        f"from a wholly broken MPI, so there is nothing safe to downgrade "
        f"to.  Fix the launcher.")


def _probe_mpi_world_size():
    """``(size, detail)`` from the live MPI, or ``(None, why-not)``.

    Reads the ALREADY-INITIALISED MPI that ``open_file`` just brought up
    (``context.cc::ensure_mpi_initialized``); it never initialises MPI
    itself, so it cannot perturb the bring-up it is auditing.
    """
    import ctypes
    cands = [(None, "global symbol table")]
    cands += [(s, s) for s in ("libmpi.so.12", "libmpi.so.40", "libmpi.so")]
    last = "no libmpi found"
    for so, label in cands:
        try:
            lib = ctypes.CDLL(so)
        except OSError as e:
            last = f"{label}: {e}"
            continue
        if not hasattr(lib, "MPI_Comm_size"):
            last = f"{label}: no MPI_Comm_size symbol"
            continue
        try:
            flag = ctypes.c_int(0)
            if lib.MPI_Initialized(ctypes.byref(flag)) != 0:
                last = f"{label}: MPI_Initialized failed"
                continue
            if not flag.value:
                return None, f"{label}: MPI not initialized at this point"
            size = ctypes.c_int(-1)
            # MPICH ABI: MPI_COMM_WORLD is the integer handle 0x44000000.
            comm = ctypes.c_int(0x44000000)
            if lib.MPI_Comm_size(comm, ctypes.byref(size)) == 0 \
                    and size.value > 0:
                return size.value, f"{label}, MPICH ABI"
            # Open MPI ABI: MPI_COMM_WORLD is &ompi_mpi_comm_world.
            try:
                addr = ctypes.addressof(
                    ctypes.c_void_p.in_dll(lib, "ompi_mpi_comm_world"))
                size2 = ctypes.c_int(-1)
                if lib.MPI_Comm_size(ctypes.c_void_p(addr),
                                     ctypes.byref(size2)) == 0 \
                        and size2.value > 0:
                    return size2.value, f"{label}, Open MPI ABI"
            except (ValueError, AttributeError):
                pass
            last = f"{label}: MPI_Comm_size returned no usable size"
        except (AttributeError, OSError) as e:
            last = f"{label}: {type(e).__name__}: {e}"
    return None, last


def _assert_mpi_world(mesh, *, mpi_size=None, probe_detail=None,
                      tier="PHDF5_FFI") -> None:
    """Refuse a collective context whose MPI world is not the JAX world.

    Runs ONCE per process per tier, at that tier's first collective open —
    the answer cannot change afterwards, and the ctypes probe costs one
    call.  A caller that already holds the authoritative communicator
    (``_MpiHostBackend`` has mpi4py's ``COMM_WORLD``) passes ``mpi_size``
    and ``probe_detail`` instead, skipping the probe: same comparison,
    one less ABI guess.
    """
    # This knob ignored half its own documented vocabulary until 2026-08-06:
    # the parse here was a bare os.environ compare with no ``.lower()``, so
    # SKIP_MPI_WORLD_CHECK=ON (or True, or YES) silently did NOT skip.  It
    # failed closed, which is why it survived unnoticed — but the next knob
    # written that way will fail open.  Both fixes for it agreed on the
    # behaviour; this is the one that routes through the shared helper, so
    # the vocabulary lives in _TRUE (line 134) and cannot drift per-site.
    if _env_flag("LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK", False):
        return
    require = os.environ.get(
        "LORRAX_PHDF5_REQUIRE_MPI_WORLD", "").strip().lower() not in _FALSE
    if tier not in _MPI_WORLD_VERDICT:
        if mpi_size is None:
            mpi_size, probe_detail = _probe_mpi_world_size()
        _MPI_WORLD_VERDICT[tier] = _mpi_world_verdict(
            mpi_size, jax.process_count(), probe_detail, require=require,
            tier=tier)
    verdict, msg = _MPI_WORLD_VERDICT[tier]
    if verdict == "refuse":
        import sys
        sys.__stderr__.write(
            f"[SlabIO ERROR rank={jax.process_index()}] {msg}\n")
        sys.__stderr__.flush()
        raise RuntimeError(msg)
    if verdict == "unprobed" and _rank0():
        print(f"  [SlabIO] WARNING: {msg}", flush=True)


# ---------------------------------------------------------------------------
# Availability — one probe, one refusal, no tiers
# ---------------------------------------------------------------------------
#: Where the contract is written down.  Quoted by the refusal: a refusal
#: that does not name its doc sends the reader to grep.
_SLAB_IO_DOC = "docs/architecture/slab_io.md#contract"

#: Fix prose per FAILED PROBE, keyed by the stage the chain stopped at,
#: because that is what decides the repair — and ``probe_target``'s three
#: states are three DIFFERENT repairs, which is why the refusal quotes its
#: reason verbatim rather than reducing it to a bool.
_SLAB_IO_FIX = {
    "loader": (
        "the FFI loader itself did not import, so no probe ever ran.  "
        "Check that PYTHONPATH reaches <lorrax>/src before looking at any "
        "library -- src/ffi/cpp/run_shifter.sh sets it for you."),
    "probe": (
        "probe_target's reason above is three-way, and each state is a "
        "DIFFERENT repair:\n"
        "            * 'unknown target'      -- this platform's library "
        "never carried the handler.  Rebuild it (GPU leg "
        "src/ffi/cpp/build.sh, CPU leg config/perlmutter/build_ffi_host.sh) "
        "and re-stage.\n"
        "            * 'could not be loaded' -- the .so may be perfectly "
        "good; the loader could not open it.  Fix LD_LIBRARY_PATH / the "
        "Shifter bind-mounts, and measure INSIDE the container: a "
        "login-node ldd lies about this closure (slab_io.md, Failure "
        "modes).\n"
        "            * 'does not export'     -- a stale .so is first on "
        "the path.  Read the PROVENANCE file stamped beside it "
        "(src/ffi/cpp/stage/stamp_provenance.sh) and compare it with the "
        "build you meant to be running."),
    "mpi": (
        "the write handler IS present; MPI could not bootstrap, which is a "
        "LAUNCHER fact and not a library one.  Launch under a "
        "PMI-providing launcher: on Perlmutter/Shifter that is "
        "`srun --mpi=cray_shasta` (pmi2 and pmix both yield singleton MPI "
        "against shifter's Cray MPICH -- src/ffi/cpp/run_shifter.sh).  A "
        "bare `python3` inside an salloc has every SLURM variable and no "
        "PMI server, and is not a supported multi-rank launch."),
}

#: PMI/PMIx variable spellings a launcher (srun --mpi=pmi2/pmix,
#: mpiexec.hydra, mpirun) leaves in the environment.  Prefix-matched so the
#: versioned PMIx names (PMIX_SERVER_URI21, ...) are covered.  Plain SLURM
#: batch variables (SLURM_JOB_ID etc.) are deliberately NOT in this list: a
#: bare ``python`` inside an sbatch allocation has all of those and still no
#: PMI server to register with — that is exactly the failing launch shape.
_MPI_LAUNCHER_ENV_PREFIXES = ("PMI_", "PMIX_")
_MPI_LAUNCHER_ENV_VARS = ("HYDI_CONTROL_FD",)

#: ``(ok, stage, reason)`` from the first probe in this process.  The answer
#: cannot change afterwards and the probe can cost a subprocess, so it runs
#: once.
_AVAILABILITY: "tuple[bool, str, str] | None" = None


def _mpi_launcher_env() -> "str | None":
    """The first launcher PMI/PMIx variable present, else None."""
    for name in _MPI_LAUNCHER_ENV_VARS:
        if os.environ.get(name):
            return name
    for name in sorted(os.environ):
        if name.startswith(_MPI_LAUNCHER_ENV_PREFIXES):
            return name
    return None


def _mpi_singleton_probe(child_code: str, what: str,
                         argv_extra=(), timeout_s: float = 60.0
                         ) -> tuple[bool, str]:
    """``(ok, reason)`` — run ``child_code`` in a THROWAWAY subprocess.

    Why a subprocess: on a bare launch (no PMI environment) whether
    ``MPI_Init_thread`` works as a singleton is a property of the MPI
    stack, and on the production stack it does not fail catchably — Intel
    MPI 2020 calls ``abort()`` inside ``MPIR_pmi_init`` (job 7884926), so
    an in-process probe kills the run it was meant to protect.  The child
    runs exactly the init the tier would run; the probe survives the
    child's death.  Only reached on the bare-launch path, so the ~1 s
    child never costs a production srun/mpirun start anything.

    Never raises.  A hung child (the init blocking rather than aborting)
    is killed at ``timeout_s`` and reported as not bootstrappable — the
    conservative direction.
    """
    import subprocess
    import sys
    try:
        res = subprocess.run(
            [sys.executable, "-c", child_code, *argv_extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, (f"the singleton {what} probe hung for {timeout_s:.0f} s "
                       f"and was killed")
    except Exception as exc:                          # pragma: no cover
        return False, (f"the singleton {what} probe could not run "
                       f"({type(exc).__name__}: {exc})")
    if res.returncode == 0:
        return True, f"singleton {what} succeeded in a probe subprocess"
    return False, (f"singleton {what} exited rc={res.returncode} in a probe "
                   f"subprocess (Intel MPI aborts in MPIR_pmi_init on a "
                   f"PMI-less launch)")


def _probe_mpi_bootstrap(platform: str) -> tuple[bool, str]:
    """``(bootstrappable, how)`` for this tier's own MPI init.

    A launcher PMI environment settles it — that is the environment every
    green multi-rank run had (CLAIMS rows 3, 17).  Without one, probe the
    exact call the tier would make (``lrx_phdf5_init_mpi`` in the loaded
    .so) in a throwaway subprocess; see :func:`_mpi_singleton_probe`.

    NOTE what this does and does not prove.  A PMI environment proves a
    launcher registered this process; it does NOT prove the PMI flavour
    matches the MPI library.  A mismatch (``srun --mpi=pmi2`` against
    Shifter's Cray MPICH) yields singleton MPI — every rank sees
    ``MPI_Comm_size()==1`` — which no probe here detects.  That one is
    caught later and unconditionally by :func:`_assert_mpi_world`, at the
    first collective open, against the LIVE communicator.
    """
    var = _mpi_launcher_env()
    if var is not None:
        return True, f"launcher PMI environment present ({var})"
    try:
        from ffi.common.ffi_loader import loaded_lib_path
        so = loaded_lib_path(platform)
    except Exception as exc:                          # pragma: no cover
        return False, f"ffi_loader unavailable ({type(exc).__name__}: {exc})"
    if not so:
        return False, (f"no loaded {platform} FFI library to probe MPI init "
                       f"with")
    # The child is handed a PATH, not a platform, and opens the .so itself —
    # so it cannot go through ffi_loader._bind_c_abi and has to know the
    # naming rule.  cpp/common/c_abi.h suffixes the HOST leg's C ABI with
    # ``_host`` so the two platform libraries share no dynamic symbol
    # (KNOWN_FAILURES L1); a pre-2026-08-08 .so of either leg has only the
    # unsuffixed name.  Ask for both, refuse if neither is there — a probe
    # that silently found no entry point would report "MPI cannot bootstrap"
    # for a build problem.
    child = ("import ctypes, sys\n"
             "lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)\n"
             "for _n in ('lrx_phdf5_init_mpi_host', 'lrx_phdf5_init_mpi'):\n"
             "    _f = getattr(lib, _n, None)\n"
             "    if _f is not None:\n"
             "        _f(); break\n"
             "else:\n"
             "    raise SystemExit('no lrx_phdf5_init_mpi entry point in '\n"
             "                     + sys.argv[1])\n")
    return _mpi_singleton_probe(child, f"MPI_Init_thread ({platform} FFI)",
                                argv_extra=(so,))


def probe_availability(platform: str | None = None) -> tuple[bool, str, str]:
    """``(ok, stage, reason)`` — can this process write one tile per rank?

    Two things must hold, and they fail for unrelated reasons, so the
    stage is reported alongside the verdict:

    1. this platform's FFI library exports ``lorrax_phdf5_write``.  Probed
       with :func:`ffi_loader.probe_target`, not a bare ``has_target``:
       its three-state reason separates "never built with the handler"
       (rebuild) from "could not be loaded" (``LD_LIBRARY_PATH``) from
       "a stale .so is first on the path" — a distinction that cost
       workstream P a day, and the distinction the refusal's *fix* line
       is keyed on.
    2. MPI can bootstrap here.  Handler presence is NOT capability: the
       write calls ``MPI_Init_thread``, and on a bare launch with no PMI
       environment Intel MPI aborts the process inside ``MPIR_pmi_init``
       (job 7884926 — the fastloop's bare P=1 gw stage died at the first
       collective H5Fcreate).

    Cached per process; the answer cannot change and probing can cost a
    subprocess.
    """
    global _AVAILABILITY
    if _AVAILABILITY is not None:
        return _AVAILABILITY
    if platform is None:
        platform = "cpu" if jax.default_backend() == "cpu" else "CUDA"
    stage, reason = "probe", "ffi.common.ffi_loader import failed"
    try:
        from ffi.common.ffi_loader import probe_target
        ok, reason = probe_target("lorrax_phdf5_write", platform)
    except Exception as exc:                          # pragma: no cover
        ok, stage, reason = False, "loader", f"{type(exc).__name__}: {exc}"
    if ok:
        mpi_ok, mpi_how = _probe_mpi_bootstrap(platform)
        if not mpi_ok:
            ok, stage = False, "mpi"
            reason = (f"the {platform} FFI exports the write handler but MPI "
                      f"cannot bootstrap in this process: {mpi_how}")
        else:
            reason = f"{reason}; MPI can bootstrap ({mpi_how})"
    _AVAILABILITY = (bool(ok), stage, reason)
    return _AVAILABILITY


def probe_read_availability(platform: str | None = None) -> tuple[bool, str]:
    """``(ok, reason)`` — can ``platform``'s library serve a slab READ?

    Probes the multi-window read target (``lorrax_phdf5_read_kchunk_union``,
    the one :meth:`_FfiBackend.read_slabs` dispatches) through
    :func:`ffi_loader.probe_target`, whose three-state reason separates
    "not a target of this library" from "the library could not be loaded"
    from "loaded but does not export the symbol" — three different fixes,
    which is why the reason comes back verbatim instead of a bool.

    NOT cached, and deliberately NOT sharing :func:`probe_availability`'s
    ``_AVAILABILITY``.  That one memoises a single verdict with no platform
    key, which is fine for the write door (it asks once, about here) and
    wrong for this one: the only way a caller uses this is a
    ("CUDA", "cpu") ladder, and a platform-blind cache would hand the FIRST
    platform's answer back for the second — reporting a host library that
    exports the handler on a node where only the CUDA one does.  The probe
    is a symbol lookup on an already-loaded library, so there is nothing to
    memoise anyway; ``probe_availability``'s cache is there for the MPI
    bootstrap subprocess, which this door does not run.

    Never raises: a broken ``ffi_loader`` import is a reason, not a crash,
    because the caller is choosing between transports.
    """
    if platform is None:
        platform = "cpu" if jax.default_backend() == "cpu" else "CUDA"
    try:
        from ffi.common.ffi_loader import probe_target
        return probe_target("lorrax_phdf5_read_kchunk_union", platform)
    except Exception as exc:                          # pragma: no cover
        return False, (f"ffi.common.ffi_loader is unavailable "
                       f"({type(exc).__name__}: {exc})")


def assert_available(platform: str | None = None) -> None:
    """Refuse, naming the failed probe, if the tile path cannot run here.

    THE ONLY REFUSAL LEFT ON THIS AXIS.  There used to be three tiers and
    a router, and the router's job was to pick a lesser transport when
    this probe failed; the lesser transports are gone (see
    ``file_io.slab_io``'s module docstring), so a failed probe is now a
    plain "this deployment cannot do the thing" and says which part.

    Nothing here is about process count.  The old refusals all keyed on
    ``process_count() > 1`` because the tier they guarded was legal at
    exactly one process; this one is not legal-at-P=1-only, it is the
    only path there is, so it either works or the deployment is broken.
    """
    ok, stage, reason = probe_availability(platform)
    if ok:
        return
    raise RuntimeError(
        "\n*** SlabIO REFUSED: this stack cannot write one tile per rank. "
        "***\n"
        f"  rule    one rank writes one tile, and nothing larger than one "
        f"rank's tile is ever materialised (owner ruling 2026-08-05).  "
        f"There is exactly one transport that does this and it is not "
        f"available here.\n"
        f"  got     probe stage '{stage}': {reason}\n"
        f"          [{_slab_io_geometry()}]\n"
        f"  wanted  the platform FFI library exports 'lorrax_phdf5_write' "
        f"AND MPI_Init_thread succeeds in this process.\n"
        f"  fix     {_SLAB_IO_FIX.get(stage, _SLAB_IO_FIX['probe'])}\n"
        f"  doc     {_SLAB_IO_DOC}")


def _slab_io_geometry() -> str:
    """The run geometry, as a one-line fragment for the refusal.

    Printed WITH the decision, on purpose.  No archived multi-node log
    records a node count, so settling "does phdf5 work across nodes"
    needed an archaeology pass rather than a grep.  The geometry belongs
    next to the decision it qualifies.

    SLURM's own node count is reported when present but is NOT the
    primary source, and the spelling matters.  Measured inside the
    Shifter container on Perlmutter, 2026-08-05, ``srun -N 4 -n 16``:
    ``SLURM_NNODES=4``, ``SLURM_NTASKS=16`` and ``SLURM_JOB_NODELIST``
    are all present, while ``SLURM_JOB_NUM_NODES`` is ABSENT — it is a
    batch/allocation-level variable, not a step-level one.  JAX's own
    process/device counts do not depend on the launcher's vocabulary at
    all, so they come first.
    """
    parts = []
    try:
        parts.append(f"processes={jax.process_count()}")
        parts.append(f"devices={jax.device_count()}")
        parts.append(f"local_devices={jax.local_device_count()}")
    except Exception:                                 # pragma: no cover
        parts.append("jax process/device counts unavailable")
    for _k in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_NTASKS"):
        _v = os.environ.get(_k)
        if _v:
            parts.append(f"{_k}={_v}")
    return ", ".join(parts)


def mesh_divisible_shape(shape, mesh, partition_spec) -> tuple[int, ...]:
    """``shape`` rounded UP so it is shardable by ``partition_spec``.

    The one place the rounding rule lives.  ``read_slab`` applies it when
    a caller omits ``shape``, which is what makes the easy call the
    correct call — see :meth:`SlabIO.read_slab`.  Exposed because a
    caller that must ALLOCATE a matching buffer (a consumer padding its
    own μ extent) needs the same number and must not re-derive it.
    """
    axis_count_per_dim, axis_flat = _sharding_to_axis_info(
        NamedSharding(mesh, partition_spec), len(shape))
    out = [int(s) for s in shape]
    flat = 0
    for d in range(len(out)):
        n_ax = axis_count_per_dim[d]
        if n_ax <= 0:
            continue
        div = 1
        for k in range(n_ax):
            div *= int(mesh.shape[mesh.axis_names[axis_flat[flat + k]]])
        flat += n_ax
        out[d] = -(-out[d] // div) * div          # ceil to a multiple of div
    return tuple(out)


def _local_shard_and_global_offset(
    A: jax.Array,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return ``(local_numpy, global_offset)`` for the process-local shard.

    LORRAX runs one JAX device per process under multi-process (mesh on
    ``mesh_xy``), so each process has exactly one addressable shard.

    The shard's ``.index`` is a tuple of ``slice`` objects giving the
    GLOBAL start/stop along each axis.  Slabs are always contiguous
    along each axis (no broadcast tiling) so ``.start`` is the offset
    within A.shape.  Replicated axes give ``slice(0, A.shape[ax])`` —
    every process holds the full axis and writes the same overlapping
    rows; under independent MPI-IO that's a redundant write but
    semantically correct (every rank writes identical bytes).
    """
    shards = A.addressable_shards
    if len(shards) != 1:
        # Multi-device-per-process (e.g. GPU with N visible devices
        # under a single process).  Not the LORRAX CPU mesh-xy regime
        # but worth a clear error rather than silent wrong data.
        raise RuntimeError(
            f"SlabIO expects 1 addressable shard per process; "
            f"got {len(shards)} for A.shape={tuple(A.shape)}.  Did you "
            f"set --xla_force_host_platform_device_count > 1 on a "
            f"multi-process run?")
    shard = shards[0]
    local = np.asarray(shard.data)
    # Replicated axes have ``slice(None, None)`` (no explicit bounds);
    # treat ``start=None`` as 0 (the full-axis slab starts at 0).
    offset = tuple(int(s.start) if s.start is not None else 0
                   for s in shard.index)
    return local, offset


# ``_shard_read_plan`` — the per-device (local_shape, dst, disk) hyperslab
# arithmetic — was DELETED on 2026-08-11.  It was written in the 2026-07-28
# audit as the single source of truth for a clip that TWO backends had
# copy-pasted (``_slab_io_allgather``'s sharded fast path and
# ``_slab_io_mpi_host.read_slab``), and both of those backends went away
# with the tier deletion at 233a830d.  Nothing has called it since; the FFI
# read path does its own clipping in ``_normalize_valid_shape`` plus the C
# handler's rank arithmetic.  A de-duplication helper whose duplicates are
# both gone is not a helper.


# The ``lfs setstripe`` prestripe helper that used to live here was
# DELETED (owner-approved, 2026-07-31): the production apptainer image
# does not ship ``lfs``, so it had never once set a stripe (measured,
# job 7876423 — both output files came back ``lmm_stripe_count: 1``).
# The Lustre layout is requested through MPI-IO's ``striping_factor``/
# ``striping_unit`` hints instead, which ROMIO applies via ``llapi``
# with no binary on PATH (``_slab_io_mpi_host._mpi_io_hints``;
# ``ffi/cpp/phdf5/context.cc``).  The ``mode='w'`` inode replace is
# :func:`_replace_inode_for_write`, unconditional of ``lfs``.

# Lazy imports happen inside the class methods; module-level imports
# of ffi.phdf5 would break users who don't build the FFI .so.


def _sharding_to_axis_info(
    sharding: NamedSharding, ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode a NamedSharding's per-dim axis lists for the FFI attrs.

    Returns ``(axis_count_per_dim, axis_flat)``:
      - axis_count_per_dim[d]: number of mesh axes sharding dim d
        (0 = replicated).
      - axis_flat: concatenation of per-dim axis index lists, in dim
        order, each list preserving JAX's leftmost-is-slowest order.

    JAX canonicalises ``PartitionSpec(None, None)`` to
    ``PartitionSpec()``, so iterate by the array's ndim and treat
    missing trailing entries as ``None``.
    """
    axis_names = list(sharding.mesh.axis_names)
    spec = list(sharding.spec)
    counts: list[int] = []
    flat: list[int] = []
    for i in range(ndim):
        s = spec[i] if i < len(spec) else None
        if s is None:
            counts.append(0)
        elif isinstance(s, str):
            if s not in axis_names:
                raise ValueError(
                    f"sharding spec dim {i}: axis '{s}' not in mesh "
                    f"axis_names {axis_names}")
            counts.append(1)
            flat.append(axis_names.index(s))
        elif isinstance(s, (list, tuple)):
            counts.append(len(s))
            for a in s:
                if a not in axis_names:
                    raise ValueError(
                        f"sharding spec dim {i}: axis '{a}' not in mesh "
                        f"axis_names {axis_names}")
                flat.append(axis_names.index(a))
        else:
            raise ValueError(f"unrecognised spec element at dim {i}: {s!r}")
    return tuple(counts), tuple(flat)


def _replicated_sharding(mesh: Mesh, ndim: int) -> NamedSharding:
    """All-None PartitionSpec on `mesh` for an ndim-D array."""
    return NamedSharding(mesh, P(*([None] * ndim)))


def _replicated_i64_vector(values: Sequence[int], mesh: Mesh) -> jax.Array:
    """Small int64 control buffer, explicitly replicated on ``mesh``.

    Do not rely on JAX's default placement for these vectors: the PHDF5
    write path passes offsets through a cached jitted shard_map, and an
    implicitly placed offset buffer once arrived in C++ with dimensions
    permuted in the real CrI3 driver.  Replicating the control buffer is
    both the intended semantics and the safest JIT cache key.
    """
    # Process-local placement: plain ``jax.device_put`` of host numpy onto
    # a multi-process sharding runs JAX's hidden ``assert_equal``
    # all-gather (scorecard AA.1) — a per-call blocking collective on a
    # control buffer that is identical on every rank by construction.
    # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
    #
    # THE WIDTH CONTRACT IS NOT CHECKED HERE, deliberately: this array is
    # handed straight to the cached ``shard_map``, whose body calls
    # ``ffi.io.{ffi_read_call,ffi_write_call}``, and those refuse a
    # non-int64 control operand by name (``require_control_i64``).  Checking
    # it twice would be two places to keep in step with one C++ signature.
    return device_put_process_local(
        np.asarray(tuple(int(v) for v in values), dtype=np.int64),
        NamedSharding(mesh, P()),
    )


def _normalize_slab_request(
    *,
    op: str,
    name: str,
    offset: Sequence[int] | None,
    slab_shape: Sequence[int],
    global_shape: Sequence[int] | None,
    check_bounds: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(offset, slab_shape, global_shape)`` after basic checks."""
    shape = tuple(int(s) for s in slab_shape)
    if not shape:
        raise ValueError(f"{op} {name!r}: slab shape must be non-empty")
    if any(s < 0 for s in shape):
        raise ValueError(f"{op} {name!r}: negative slab shape {shape}")

    off = tuple(int(o) for o in (offset if offset is not None
                                else (0,) * len(shape)))
    gshape = tuple(int(s) for s in (global_shape if global_shape is not None
                                   else shape))

    if len(off) != len(shape) or len(gshape) != len(shape):
        raise ValueError(
            f"{op} {name!r}: rank mismatch offset={off}, "
            f"slab_shape={shape}, global_shape={gshape}")
    if any(o < 0 for o in off):
        raise ValueError(f"{op} {name!r}: negative offset {off}")
    if any(g < 0 for g in gshape):
        raise ValueError(f"{op} {name!r}: negative global shape {gshape}")

    if check_bounds:
        over = [
            (i, off[i], shape[i], gshape[i])
            for i in range(len(shape))
            if off[i] + shape[i] > gshape[i]
        ]
        if over:
            details = ", ".join(
                f"dim {i}: {o}+{s}>{g}" for i, o, s, g in over)
            raise ValueError(
                f"{op} {name!r}: slab exceeds global shape ({details})")
    return off, shape, gshape


# ---------------------------------------------------------------------------
# The logical extent of a slab — derived, not configured
# ---------------------------------------------------------------------------
#
# decisions.md 2026-08-04, "Padding is SlabIO's business, not the caller's":
# a caller states LOGICAL shapes only.  ``valid_shape`` therefore DEFAULTS to
# the operand's own extent clipped to the dataset, and survives only as an
# OVERRIDE for the ragged-chunk case (a chunk buffer whose tail is not part of
# the write, which SlabIO cannot derive because both extents are legitimate).
#
# The derivation is the whole point of the entry: it is exactly the arithmetic
# that every call site used to do by hand, and getting it wrong at any one of
# them produced a wholly-padded rank, an overrun, or a silent prefix write.
def _derive_valid_shape(
    slab_shape: Sequence[int],
    offset: Sequence[int],
    ds_shape: Sequence[int],
) -> tuple[int, ...]:
    """``min(slab, dataset - offset)`` per dim, floored at 0.

    ``slab_shape`` is the PHYSICAL extent the caller handed us (possibly
    padded for mesh divisibility); ``ds_shape`` is the dataset's LOGICAL
    extent.  The clip is what turns "my buffer has pad rows" into "those
    rows are not part of the file", with no caller-side arithmetic.

    A slab that starts past the end of the dataset yields 0 on that dim,
    i.e. a globally empty request, which is a legitimate no-op rendezvous
    (every rank selects nothing) and not a refusal.
    """
    return tuple(max(0, min(int(s), int(g) - int(o)))
                 for s, o, g in zip(slab_shape, offset, ds_shape))


def _normalize_valid_shape(
    *,
    op: str,
    name: str,
    valid_shape: Sequence[int] | None,
    slab_shape: Sequence[int],
    offset: Sequence[int],
    ds_shape: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the logical on-file extent inside a possibly padded slab.

    ``slab_shape`` is the physical JAX array shape.  ``ds_shape`` is the
    dataset's own extent — a REPLICATED quantity (it comes from the
    create_dataset call, the auto-create global shape, or a metadata read
    every rank performs), which is what makes the bounds verdict below
    rank-invariant.

    ``valid_shape=None`` (the ordinary case) derives the extent via
    :func:`_derive_valid_shape`.  An explicit ``valid_shape`` is the
    ragged-chunk override: it must fit inside the physical slab, and
    ``offset + valid_shape`` must fit inside the dataset — an override
    that overruns is a REFUSAL, because the caller asserted an extent
    SlabIO has no licence to silently shrink.

    ``ds_shape=None`` means the dataset does not exist yet and is about
    to be created at exactly ``slab_shape``; then the whole slab is
    valid by construction.
    """
    shape = tuple(int(s) for s in slab_shape)
    off = tuple(int(o) for o in offset)
    gshape = None if ds_shape is None else tuple(int(s) for s in ds_shape)

    if valid_shape is None:
        if gshape is None:
            return shape
        if len(gshape) != len(shape):
            raise ValueError(
                f"{op} {name!r}: dataset rank {len(gshape)} does not match "
                f"slab rank {len(shape)} (dataset={gshape}, slab={shape})")
        return _derive_valid_shape(shape, off, gshape)

    vshape = tuple(int(s) for s in valid_shape)
    if len(vshape) != len(shape):
        raise ValueError(
            f"{op} {name!r}: valid_shape rank mismatch "
            f"valid_shape={vshape}, slab_shape={shape}")
    if any(s < 0 for s in vshape):
        raise ValueError(f"{op} {name!r}: negative valid_shape {vshape}")
    too_large = [
        (i, vshape[i], shape[i])
        for i in range(len(shape))
        if vshape[i] > shape[i]
    ]
    if too_large:
        details = ", ".join(f"dim {i}: {v}>{s}"
                            for i, v, s in too_large)
        raise ValueError(
            f"{op} {name!r}: valid_shape exceeds slab shape ({details})")
    if gshape is not None:
        over = [
            (i, off[i], vshape[i], gshape[i])
            for i in range(len(shape))
            if off[i] + vshape[i] > gshape[i]
        ]
        if over:
            details = ", ".join(
                f"dim {i}: {o}+{s}>{g}" for i, o, s, g in over)
            raise ValueError(
                f"{op} {name!r}: valid slab exceeds dataset extent "
                f"({details}).  valid_shape is an explicit override; drop it "
                f"and SlabIO clips the slab to the dataset instead.")
    return vshape


# ---------------------------------------------------------------------------
# The same clip, once per (rank, window) — the multi-window read's table
# ---------------------------------------------------------------------------
#
# :meth:`_FfiBackend.read_slabs` asks the C++ union handler for n windows of
# ONE common slab shape in ONE H5Dread, and the handler shifts each rank onto
# its own block of every window before it selects.  So each rank needs its OWN
# count row per window, and that row is exactly :func:`_derive_valid_shape` of
# (the rank's block shape, the rank's offset INSIDE the window, the window's
# valid extent) — the same clip the one-rectangle path applies, evaluated
# world x n_windows times.
#
# THE TABLE USED TO LIVE IN THE PSI LOADER (``file_io/wfn_loader.py``'s
# ``_build_phdf5_clamped_counts``), one import away from the clip it was
# applying, and the two DIVERGED on the band axis: the loader's copy clipped
# to ``mnband`` (the FILE extent) while its own serial reader clipped to
# ``b_hi`` (the LOGICAL window) and zeroed the rest.  A load refuses
# ``b_hi > mnband``, so the file clip is never the tighter of the two — it
# fires only past EOF and NOT on the pad rows between ``b_hi`` and
# ``b_lo + nb_padded``.  Those rows came back holding real file bands on the
# collective path and zeros on the serial one, on every request whose band
# count did not divide the world size; the parity harness's own geometry
# happened to divide, which is why it went unseen (22049c3, fixed 2026-08-06).
# Layout and clip now live in THIS file, three definitions apart, and the
# caller states LOGICAL extents only — which is the rule ``valid_shape``
# already follows for one rectangle.  That is the whole reason the table
# moved: the divergence class dies structurally rather than by a test.


def _normalize_window_tables(
    *, name: str, offsets, valid_shapes, ndim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The two per-window tables as ``(n, ndim)`` int64 — or a REFUSAL.

    ``read_slabs`` reads n windows of ONE slab shape, so the tables describe
    the same n windows and must agree with each other on n and with the slab
    on ndim.  Checked HERE, by name and with both shapes, because neither
    mismatch has a symptom downstream: the tables are dispatched into a
    compiled ``shard_map`` whose complaint would name a traced buffer, and a
    table at the wrong ndim is read row-major by the C++ handler as a
    different set of windows entirely — a wrong hyperslab that returns
    rc=0 data.  Rank-invariant: every rank builds these from the same
    request.
    """
    off = np.asarray(offsets, dtype=np.int64)
    val = np.asarray(valid_shapes, dtype=np.int64)
    for what, tbl in (("offsets", off), ("valid_shapes", val)):
        if tbl.ndim != 2:
            raise ValueError(
                f"read_slabs {name!r}: {what} must be a 2-D (n_windows, "
                f"ndim) table; got shape {tuple(tbl.shape)}")
    if off.shape[0] != val.shape[0]:
        raise ValueError(
            f"read_slabs {name!r}: offsets and valid_shapes describe "
            f"different window counts ({off.shape[0]} vs {val.shape[0]}); "
            f"they are two columns of ONE table of windows.")
    if off.shape[1] != ndim or val.shape[1] != ndim:
        raise ValueError(
            f"read_slabs {name!r}: window tables have ndim "
            f"{off.shape[1]}/{val.shape[1]} but the slab shape has {ndim} "
            f"dimensions; every window is a slab of that one shape, so all "
            f"three ranks must agree.")
    if (off < 0).any() or (val < 0).any():
        raise ValueError(
            f"read_slabs {name!r}: negative entry in the window tables "
            f"(offsets min {int(off.min())}, valid_shapes min "
            f"{int(val.min())})")
    return off, val


def _rank_block_offsets(
    *,
    per_rank_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    mesh_shape: Sequence[int],
) -> np.ndarray:
    """``(world, ndim)`` — where each rank's block starts INSIDE a window.

    Transcribes the handler's own arithmetic (``read_ffi.cc``, "per-rank
    coord shift on sharded dims"): un-ravel the rank through ``mesh_shape``
    row-major, then for a dim sharded by several mesh axes combine their
    coordinates leftmost-is-slowest — JAX's order, which is what
    ``axis_flat`` records.  A replicated dim contributes 0, so every rank
    starts that dim at the window origin.

    Built for EVERY rank rather than read off this process's shard index:
    the count table is assembled whole and then sharded, so rank r's row
    has to be computable on rank 0.  Pure arithmetic over replicated
    quantities, so every rank builds the same table.
    """
    ndim = len(tuple(per_rank_shape))
    dims = tuple(int(m) for m in mesh_shape)
    world = 1
    for m in dims:
        world *= m
    out = np.zeros((world, ndim), dtype=np.int64)
    for r in range(world):
        coord = np.unravel_index(r, dims)
        flat = 0
        for d in range(ndim):
            na = int(axis_count_per_dim[d])
            block, stride = 0, 1
            for k in range(na - 1, -1, -1):
                ax = int(axis_flat[flat + k])
                block += int(coord[ax]) * stride
                stride *= dims[ax]
            flat += na
            out[r, d] = block * int(per_rank_shape[d])
    return out


def _derive_window_counts(
    *,
    per_rank_shape: Sequence[int],
    rank_offsets: np.ndarray,
    valid_shapes: np.ndarray,
) -> np.ndarray:
    """``(world * n_windows, ndim)`` per-rank hyperslab counts.

    One :func:`_derive_valid_shape` per (rank, window): the rank's block
    shape, clipped to what is left of that window's VALID extent past the
    rank's own offset into it.  A rank whose block starts past the valid
    extent on a dim gets 0 there — it selects nothing, and the handler's
    pre-zeroed staging buffer makes its tile read as exactly zero, which is
    the pad-row semantics the one-rectangle read has everywhere else.
    Equivalently ``sum_r count[r, w, d] == valid_shapes[w][d]`` on a dim
    the world shards, which is the invariant the regression cells assert.

    Row layout is the handler's: rank-major, window-minor, flattened so the
    leading axis can be sharded across the mesh — each rank's shard_map-local
    view is then exactly its own ``(n_windows, ndim)`` slice, and no rank
    ever sends a row.
    """
    world = int(rank_offsets.shape[0])
    n_win = int(valid_shapes.shape[0])
    ndim = len(tuple(per_rank_shape))
    counts = np.zeros((world, n_win, ndim), dtype=np.int64)
    for r in range(world):
        for w in range(n_win):
            counts[r, w] = _derive_valid_shape(
                per_rank_shape, rank_offsets[r], valid_shapes[w])
    return counts.reshape(world * n_win, ndim)


# ---------------------------------------------------------------------------
# Dataset geometry — the replicated record that makes the derivation legal
# ---------------------------------------------------------------------------
class _DatasetGeometry:
    """Per-handle record of each dataset's LOGICAL shape and dtype.

    Every entry is written from a RANK-INDEPENDENT quantity: the shape
    passed to ``create_dataset`` (SPMD by contract), the ``global_shape``
    an auto-creating ``write_slab`` used, or a metadata read that every
    rank performs on the same file.  So ``_known_shape`` returns the same
    tuple on every rank, which is the precondition for deriving
    ``valid_shape`` from it — a per-rank dataset shape would put the
    ranks back on different sides of the bounds test.

    The record is authoritative because ``create_dataset`` REFUSES a
    shape/dtype change on an existing dataset (decisions.md 2026-08-04);
    without that rule an ``H5Dopen`` of a differently-shaped dataset
    would leave this dict describing geometry the file does not have.

    THAT REFUSAL IS THE C HANDLER'S, not this class's.  A Python
    ``_refuse_geometry_change`` method used to live here and raise the
    reuse-or-refuse ValueError; it was DELETED on 2026-08-11 with zero
    callers.  ``lrx_phdf5_ensure_dataset`` performs the same check
    collectively on every rank, which is where it has to be — a Python
    twin can only see the ranks that reach it, and the two could disagree.
    """

    def _geom_init(self) -> None:
        self._ds_geom: dict[str, tuple[tuple[int, ...], "np.dtype"]] = {}

    def _remember_geom(self, name: str, shape, dtype) -> None:
        self._ds_geom[str(name)] = (
            tuple(int(s) for s in shape), np.dtype(dtype))

    def _known_shape(self, name: str) -> tuple[int, ...] | None:
        got = self._ds_geom.get(str(name))
        return None if got is None else got[0]


def _shard_divisors(
    *,
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    mesh_shape: Sequence[int],
    ndim: int,
) -> tuple[int, ...]:
    """Per-dim product of the mesh axis sizes sharding that dim (1 = replicated).

    The divisor the FFI's equal-block rank arithmetic needs each dim to
    be a multiple of — which is also JAX's own divisor for building a
    ``NamedSharding`` block over that dim.  Read by
    :func:`_validate_block_divisible`.
    """
    divs: list[int] = []
    flat_idx = 0
    for d in range(ndim):
        na = int(axis_count_per_dim[d])
        div = 1
        for k in range(na):
            div *= int(mesh_shape[int(axis_flat[flat_idx + k])])
        flat_idx += na
        divs.append(div)
    return tuple(divs)


def _validate_block_divisible(
    *,
    op: str,
    name: str,
    shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    mesh_shape: Sequence[int],
) -> None:
    """Reject a (shape, spec) pair that cannot form equal block shards.

    NOT a SlabIO padding requirement — decisions.md 2026-08-04 takes
    divisibility out of the caller's contract, and this check is what is
    left after it: a restatement of JAX's OWN constraint, raised early
    and with the numbers named.

    MEASURED (gate ``PADRANK_CASE=nodiv``, job 7888644, jax 0.9.1): a
    ``jax.Array`` whose sharded dim does not divide its mesh-axis product
    cannot be constructed at all — ``device_put`` raises
    ``IndivisibleError``.  So SlabIO is never HANDED such an operand on
    the write path (there is nothing to pad), and on the read path it
    could not RETURN one either: padding the read and trimming would
    only replace this message with JAX's, at the same refusal.  That also
    settles the audit's item 5 — ``_MpiHostBackend``'s absent check is
    not a capability difference, because the case it would accept cannot
    be expressed.

    A caller who wants a padded extent asks for the padded SHAPE, which
    is a legitimate request and needs no other argument.
    """
    divs = _shard_divisors(
        axis_count_per_dim=axis_count_per_dim, axis_flat=axis_flat,
        mesh_shape=mesh_shape, ndim=len(tuple(shape)))
    for d, size in enumerate(tuple(int(s) for s in shape)):
        if divs[d] > 1 and size % divs[d]:
            raise ValueError(
                f"{op} {name!r}: dimension {d} size {size} is not "
                f"divisible by its mesh-axis product {divs[d]}, so the "
                f"array you asked for cannot be sharded this way — JAX "
                f"itself refuses to build it (IndivisibleError).  Ask for "
                f"dimension {d} at {-(-size // divs[d]) * divs[d]} "
                f"instead: SlabIO fills what the dataset covers and zeroes "
                f"the rest, and you state nothing else.")


# ---------------------------------------------------------------------------
# Module-level shard_map kernel factories (read / write)
# ---------------------------------------------------------------------------
#
# A jit'd shard_map per ``(mesh, sharding, mesh_shape, axis layout,
# dtype/shape)`` signature.  Caching at module scope means all
# ``_FfiBackend`` instances share one cache — re-opening the same file,
# or opening any file with a matching FFI signature, reuses the compile.
# The closure is built INSIDE the cached factory so its Python ``id()``
# is stable per cache entry (vs ``functools.partial``, which constructs
# a fresh wrapper each call and defeats JAX's trace-cache identity test).
#
# ``ctx_handle`` and ``ds_id`` are NOT in that signature: they travel as
# a runtime ``(2,)`` int64 buffer (``ffi.io.handle_vector``), so one
# compiled module serves every file, every dataset and every process.
#
# They used to be FFI Attrs, which are baked into the HLO.  ctx_handle is
# a heap address, so the module differed in EVERY PROCESS and the JAX
# persistent compile cache could never hit one — it wrote a fresh entry
# every run, forever.  MEASURED 2026-08-07, byte-identical workload into
# a private cache dir: ``jit__per_rank`` entries went 4 -> 8 -> 12 across
# three runs while a plain ``jax.jit`` control stayed at 1, and the
# shared ``$SCRATCH/lorrax_jax_cache/np1`` had accumulated 6813 such dead
# entries out of 14443 (47%, 27 MB, growing without bound).  The HLO diff
# between two processes was a single token: ``ctx_handle = 56960080`` vs
# ``58537408``.  ds_id moved out with it because as an Attr it forked a
# module per (file, dataset) pair, so the module count scaled with the
# deck rather than with the geometry.
#
# The C++ side reads both out of the buffer at the top of ReadDispatch /
# WriteDispatch (one 16-byte D2H copy, alongside the two already there
# for offset/valid_shape).  Changing one side without the other is a hard
# FFI signature mismatch at the first call — they land together.

@functools.lru_cache(maxsize=None)
def _get_read_sm(mesh, partition_spec, *,
                 mesh_shape, axis_count_per_dim, axis_flat, out_struct):
    """One H5Dread per rank.  Returns a jit'd shard_map; identity-stable
    via lru_cache so JAX's trace cache hits on repeat invocation."""
    from ffi.phdf5.read import ffi_read_call

    def _per_rank(handle_local, offset_local, valid_shape_local):
        return ffi_read_call(
            out_struct, handle_local, offset_local, valid_shape_local,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
        )
    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), P(), P()), out_specs=partition_spec,
        check_vma=False,
    )
    return jax.jit(sm_bare)


@functools.lru_cache(maxsize=None)
def _get_write_sm(mesh, in_specs, *,
                  mesh_shape, axis_count_per_dim, axis_flat, no_jit):
    """One H5Dwrite per rank.  ``LORRAX_WRITE_NO_JIT=1`` (passed via
    ``no_jit``) skips the jit wrapper — diagnostic for chasing the
    jit-argument-retention buffer leak on long write loops."""
    from ffi.phdf5.write import ffi_write_call

    def _per_rank(A_local, handle_local, offset_local, valid_shape_local):
        return ffi_write_call(
            A_local, handle_local, offset_local, valid_shape_local,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
        )
    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(in_specs, P(), P(), P()), out_specs=P(),
        check_vma=False,
    )
    return sm_bare if no_jit else jax.jit(sm_bare)


# ---------------------------------------------------------------------------
# Dataset attributes
# ---------------------------------------------------------------------------
# ``create_dataset(attrs=...)`` used to be DISCARDED here, with a
# ``warnings.warn`` where the write should have been.  That is the
# defect fixed on 2026-08-08, and it was a production one rather than a
# cosmetic one: ``file_io.sigma_output`` stamps ``k_storage="ibz"`` on
# each Σ dataset through exactly this argument, and ``kin_ion``'s reader
# is specified so that a dataset with NO ``k_storage`` attr means stored
# on the full BZ (the back-compat rule that keeps every pre-format file
# readable).  So a wedge cube written through this transport — the only
# transport SlabIO has, and therefore every cluster-written cube — came
# back claiming to be a full-BZ cube with nrk rows, and every k_irr
# consumer indexed full-BZ rows into it.  The warning made that visible
# to nobody: it goes to stderr in the middle of a run's write telemetry,
# and the file it describes is wrong for the rest of its life.
#
# A transport does not get to drop the caller's data and call the
# warning a mitigation.  These attrs are written, by the same rank-0
# h5py machinery ``write_attr`` already defers its small datasets to,
# and through the same ``ds.attrs[key] = value`` assignment that the
# host-side writers (``gw.kin_ion_io``, ``sigma_output``'s QSGW
# appender) use — so a stamp written by the FFI transport and the same
# stamp written host-side are the same bytes in the file, not merely
# the same intent.

def _host_attr_value(value):
    """The value h5py is handed, host-side, for one attribute.

    Almost everything passes through UNTOUCHED, and that is the point:
    ``ds.attrs["k_storage"] = "ibz"`` writes a variable-length UTF-8
    string, while ``ds.attrs["k_storage"] = np.asarray("ibz")`` writes a
    fixed-length byte string that reads back as ``b"ibz"`` — and
    :func:`file_io.kin_ion.read_star_map` does ``str(...)`` on what it
    reads, so the second spelling produces the literal ``"b'ibz'"`` and
    refuses the file it just wrote.  A helpful coercion here would be a
    second format.

    Only a device array needs anything done to it, and what it needs is
    the pull to host that ``write_attr``'s drain does for the same
    reason: h5py cannot serialise a ``jax.Array``.
    """
    if isinstance(value, (str, bytes, bool, int, float,
                          np.generic, np.ndarray)):
        return value
    return np.asarray(jax.device_get(value))


def _apply_dataset_attrs(h5, pending) -> None:
    """Stamp ``pending`` — ``[(dataset_name, attrs_dict), …]`` — onto ``h5``.

    ``h5`` is an OPEN ``h5py.File`` in a writable mode; the caller owns
    the open, because on the transport this runs inside the one rank-0
    reopen that also lands the deferred small datasets.

    A name that is not in the file RAISES.  The alternative — skip it,
    the dataset is gone anyway — is the shape of the defect this
    function exists to close: an attr silently not written is a file
    that lies about what it holds, and the stamps that come through
    here are the ones that say whether an array is a wedge or the full
    BZ.  Every caller of ``create_dataset`` creates the dataset in the
    same breath, so a miss here is a transport bug and should read as
    one.
    """
    for name, attrs in pending:
        if name not in h5:
            raise KeyError(
                f"SlabIO: {os.path.basename(h5.filename)} has no dataset "
                f"{name!r} to stamp {sorted(attrs)} onto.  The attrs were "
                f"handed to create_dataset({name!r}, ...), so the dataset "
                f"should exist — refusing to drop them silently, which is "
                f"the defect this path was written to close.")
        ds = h5[name]
        for key, value in attrs.items():
            ds.attrs[key] = _host_attr_value(value)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class _FfiBackend(_DatasetGeometry):
    """Collective MPI-IO SlabIO backend."""

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # Lazy import — keeps file_io importable without the FFI built.
        from ffi.phdf5 import open_file as _open_file, close_file as _close_file
        from ffi.common import ffi_loader as _loader

        self._open_file = _open_file
        self._close_file = _close_file
        self._loader = _loader

        # Refuse HERE, before the inode is touched, if this stack cannot
        # serve the tile path.  One probe, cached per process.  There is
        # nothing to demote to, which is the whole 2026-08-06 change: the
        # router that used to answer this question by picking a lesser
        # transport has no lesser transport left to pick.
        assert_available()

        self.path = path
        self.mesh = mesh
        self.mode = mode
        # mode='w' must REPLACE the inode (rank-0 unlink + barrier,
        # shared with _MpiHostBackend — see _replace_inode_for_write):
        # Lustre layout is fixed at inode create, so H5Fcreate(TRUNC)
        # over an old file silently keeps its stripe count and the
        # MPI_Info striping hints no-op.  Unconditional of `lfs` (audit
        # 2026-07-28, job 7876423 1-stripe evidence).  'a'/'r' keep the
        # existing inode and its layout by design.  Barrier after the
        # rank-0 unlink so all ranks see the inode state before
        # H5Fcreate.
        if mode == "w":
            _replace_inode_for_write(path)
            _barrier("slab_io_ffi_prestripe")
        # Resolve the striping policy and put it where the C++ writer can
        # see it (its only input is getenv), BEFORE open_file — that call
        # is where context.cc builds the MPI_Info and H5Fcreate applies
        # the layout.  A file opened 'a'/'r' keeps the inode it already
        # has, so the hints are a no-op there; export anyway so the log
        # line and the environment agree on every path.
        _sc, _su = _export_striping_env()
        if mode == "w" and _rank0() and debug_print_enabled():
            print(f"  [SlabIO.phdf5_ffi] {os.path.basename(path)} mode={mode} "
                  f"ranks={jax.process_count()} stripe_count={_sc} "
                  f"stripe_unit={_su} B (policy; "
                  f"LORRAX_PHDF5_STRIPE_COUNT/_SIZE_FS override)",
                  flush=True)
        elif mode == "r" and _rank0() and debug_print_enabled():
            # The read side names its own dominant term, and it is a
            # DIFFERENT number: a Lustre layout is fixed at inode create, so
            # on a file LORRAX did not write (every WFN.h5 — pw2bgw creates
            # it) the policy above is inert and the file's own layout governs
            # the read.  stripe_count=1 pins cb_nodes=1, i.e. ONE aggregator
            # at any rank count.  This announcement was hand-rolled in
            # WfnLoader._ensure_phdf5_ctx, which was the read side's way of
            # reaching around SlabIO; it belongs where the read handle is
            # opened.  See file_stripe_layout for the measurement.
            _lay = file_stripe_layout(path)
            if _lay is not None:
                _c, _s = _lay
                _warn = ("  <-- ONE STRIPE: cb_nodes=1, single-aggregator "
                         "read at any rank count; `lfs setstripe -c 16 -S 4M`"
                         " on the deck dir before the file is created, or "
                         "`lfs migrate` this file" if _c == 1 else "")
                print(f"  [SlabIO.phdf5_ffi] {os.path.basename(path)} "
                      f"mode={mode} file stripe_count={_c} stripe_size={_s} B "
                      f"(the file's own inode, not the policy){_warn}",
                      flush=True)
        # ONE HDF5 LIBRARY INSTANCE PER OPEN FILE (audit A1; claims/0110).
        # Declared BEFORE the collective open, so a path h5py already
        # holds open in a way that can write is refused by name here
        # instead of surfacing as "file signature not found" or a native
        # segfault later.  ``file_io.hdf5_owner`` owns the rule and the
        # message; this line owns only "the FFI is opening this path now".
        from .hdf5_owner import STACK_FFI, note_close, note_open
        self._owner_token: int | None = note_open(
            path, STACK_FFI, mode,
            where=f"SlabIO/_FfiBackend({os.path.basename(path)}, "
                  f"mode={mode!r})")
        try:
            # ISSUE-TIME journal line: this is the last Python statement
            # before ``H5Fopen``/``H5Fcreate``, and the handle it is about
            # to return cannot appear on it (SlabIO writes the completion
            # line that carries the handle).
            with _journal.op_scope("open", path, stack=_J_FFI, mode=mode):
                self.fh: int = self._open_file(path, mesh=mesh, mode=mode)
            # ``open_file`` has now brought MPI up (context.cc::
            # ensure_mpi_initialized).  Ask it how big the world REALLY is
            # before a single hyperslab is derived from
            # jax.process_count().  See _assert_mpi_world for the
            # measurement this exists for.
            _assert_mpi_world(mesh)
        except BaseException:
            # An open that failed holds nothing; leaving it registered
            # would refuse every later legitimate open on this path.
            note_close(path, self._owner_token)
            self._owner_token = None
            raise
        self._ds_ids: dict[str, int] = {}
        # Replicated record of every dataset's LOGICAL geometry — the
        # thing ``valid_shape`` is derived from.  See _DatasetGeometry.
        self._geom_init()
        # write_attr needs plain h5py (the FFI doesn't expose a
        # collective attr-write path), so we defer attr writes to
        # close() — concurrent h5py + MPI-IO on the same file would
        # corrupt HDF5 metadata.
        self._deferred_attrs: list[tuple[str, object]] = []
        # ``create_dataset(attrs=...)`` rides the SAME deferral, for the
        # same reason and into the same rank-0 reopen: an H5 attribute
        # write is metadata on a file MPI-IO still holds open.  See
        # :func:`_apply_dataset_attrs`.
        self._deferred_ds_attrs: list[tuple[str, dict]] = []
        # Python-level async writer.  ``write_slab`` enqueues a callable
        # here; the ``AsyncDispatcher`` worker pops it and calls
        # ``jax.jit(shard_map(_per_rank))(A).block_until_ready()``.
        # Rationale: XLA's ``ffi::Future`` async mechanism registers the
        # Future with XLA's scheduler but still blocks the caller
        # (Python main thread) of ``jit(...)(A)`` until the Future
        # resolves — i.e. until ``H5Dwrite`` completes.  By doing the
        # jit on a dedicated Python worker thread, we leave the main
        # Python thread free to build the next chunk while the current
        # one is still writing.  One worker per backend (FIFO) ensures
        # every rank dispatches in the same order, which is the MPI-IO
        # collective rendezvous requirement.  See
        # ``reports/session_2026-04-18_async_probe/report.md``.
        # Compiled shard_map cache lives at module level — see
        # ``_get_read_sm`` / ``_get_write_sm``.  Instance no longer
        # carries its own ``_sm_cache``; the module-level lru_cache is
        # shared across all _FfiBackend instances, so re-opening the
        # same (or another) file with matching FFI signature reuses
        # the cached compile.
        # Bound the write-dispatch queue to prevent GPU memory growth
        # across chunks.  Each queued ``_task`` closure captures its
        # input ``A`` (the jax.Array being written) by Python reference
        # — XLA's allocator counts A as live-in-use until the closure
        # runs and returns.  With H5Dwrite at ~11 s per chunk and
        # chunk-compute at ~1-2 s, an unbounded queue grows ~1 task per
        # chunk at steady state: each chunk's A accumulates on GPU and
        # ``bytes_in_use`` rises by ~1 zeta_chunk/rank/chunk until OOM.
        #
        # Total in-flight A-holding at queue-cap K = (K queued +
        # 1 being processed + 1 in main-thread transpose view).
        # Throughput cost vs unbounded is small above K=2; writer is
        # already the bottleneck on typical H5Dwrite rates.
        #
        # Measured at Si 4x4x4 60Ry / 2400c / mem16:
        #   K=0 unbounded: 12.91 → 22.48+ GB / 28 s zeta_fit (OOM-bound)
        #   K=2:           12.91 → 16.47 GB (flat) / 97 s zeta_fit
        #   K=4:           12.91 → 18.50 GB (flat) / 92 s zeta_fit
        # K=2 gives identical throughput to K=4 on this system (writer
        # saturates) while saving 2 × zeta_chunk/rank.
        from common.async_io import AsyncDispatcher
        self._dispatcher = AsyncDispatcher(
            name=f"phdf5-dispatch-{path}", maxsize=2)
        # Bytes handed to the writer thread that are not on disk yet.
        # write_slab returns as soon as the task is queued, so the
        # only place that can put a denominator under the flush is the
        # drain — see close().
        self._queued_bytes: int = 0

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        attrs: dict | None = None,
    ) -> None:
        # ``phdf5_ensure_dataset`` is a collective HDF5 op (H5Dcreate)
        # that goes through the same MPI file handle as the writer
        # thread's H5Dwrite.  If we issue it while the writer is still
        # in flight, MPI's datatype-cache state on the file handle
        # interleaves and the next H5Dwrite trips ``MPI_File_set_view:
        # Invalid datatype``.  Drain first.
        #
        # This drain is where a multi-dataset file spends most of its
        # write time — the previous dataset's H5Dwrite completes HERE,
        # not in close() — and it used to be silent, which is why the
        # caller-side per-dataset timing in file_io.tagged_arrays
        # attributed one tensor's transfer to the next dataset in the
        # file.  Report it where it happens.
        import time as _time
        _t0 = _time.perf_counter()
        _flushed = self._drain_pending()
        _dt = _time.perf_counter() - _t0
        _log_level = _close_log_level()
        if (_flushed and jax.process_index() == 0
                and (_log_level >= 2 or _dt >= 1.0 or _flushed >= 1_000_000_000)):
            print(f"  [SlabIO.flush] {os.path.basename(self.path)}: "
                  f"{_flushed / 1e9:.2f} GB written in {_dt:.1f} s "
                  f"({_flushed / 1e6 / max(_dt, 1e-9):.0f} MB/s) before "
                  f"creating {name!r}", flush=True)
        # ``phdf5_ensure_dataset`` REFUSES (on every rank — it is
        # collective and its inputs are replicated) when the dataset
        # exists at a different shape or dtype, and reuses it when they
        # match: decisions.md 2026-08-04.  That refusal is what makes the
        # geometry recorded below authoritative.
        with _journal.op_scope("create", self.path, stack=_J_FFI, ds=name,
                               cnt=tuple(int(s) for s in shape),
                               mode=self.mode, handle=self.fh):
            ds_id = self._loader.phdf5_ensure_dataset(
                self.fh, name, tuple(int(s) for s in shape),
                str(jnp.dtype(dtype).name),
            )
        self._ds_ids[name] = ds_id
        self._remember_geom(name, shape, jnp.dtype(dtype))
        # THE ATTRS ARE WRITTEN.  Queued here and landed by
        # :func:`_apply_dataset_attrs` inside close()'s rank-0 reopen,
        # because an H5 attribute write is metadata on a file collective
        # MPI-IO is still holding — the same constraint that put
        # ``write_attr`` on the same queue, and the same rendezvous
        # pays for both.  The record is a COPY: the caller's dict is
        # theirs to mutate between here and close().
        if attrs:
            self._deferred_ds_attrs.append((name, dict(attrs)))
    # ------------------------------------------------------------------
    def write_attr(self, name: str, value) -> None:
        # Deferred to close() to avoid interleaving rank-0 h5py with
        # active MPI-IO on the same file.  Small arrays only; this is
        # not meant for large data.
        self._deferred_attrs.append((name, value))

    # ------------------------------------------------------------------
    def _drain_pending(self) -> int:
        """Block until all queued write tasks finish; return their bytes.

        The byte count is what the drain actually moved, which is the
        numerator the caller needs to turn a duration into a rate.
        """
        self._dispatcher.drain()
        nbytes, self._queued_bytes = self._queued_bytes, 0
        return nbytes

    # ------------------------------------------------------------------
    def _introspect_dataset(self, name: str) -> tuple[tuple[int, ...], "np.dtype"]:
        """Return ``(shape, dtype)`` of an existing dataset.

        THROUGH THE FFI — the same library instance that already holds
        this file open (``ffi_loader.phdf5_dataset_geometry``).  Cached so
        repeated lookups for the same name are free.

        WHY THIS IS NOT h5py ANY MORE (2026-08-22).  It used to open
        ``self.path`` a second time with serial h5py while the collective
        handle was live.  That is legal only while BOTH stacks are
        read-only, and ``file_io.hdf5_owner`` refuses it by name — as it
        must — the moment the live FFI handle can write.  A real caller
        walked into exactly that: ``get_dipole_mtxels
        --parallel-transport-out`` held ``parallel_transport.h5``
        ``mode='a'`` and then introspected ``links_ibz``, so every rank
        died on the one-owner refusal AFTER the expensive PT tensor was
        already on disk and BEFORE ``dipole.h5`` was written.  The guard
        was right and the caller was wrong; the repair is to stop needing
        a second library at all.

        OLDER LIBRARIES.  A ``.so`` built before 2026-08-22 exports no
        geometry entry point.  On a read-only handle the h5py introspect
        is still legal, so it is taken, announced ONCE per process and
        counted by the registry.  On a handle that can write there is
        nothing legal to fall back to, so this refuses by name and points
        at the rebuild — which is the same failure as before, with a
        message that names its repair.
        """
        cache = getattr(self, "_introspect_cache", None)
        if cache is None:
            cache = {}
            self._introspect_cache = cache
        if name in cache:
            return cache[name]

        if self._loader.has_phdf5_metadata_api(self._platform()):
            # DRAIN FIRST.  This route is a COLLECTIVE HDF5 operation — it
            # routes through the same cached ``H5Dopen`` ``_ds_id`` uses —
            # where the h5py route below was a local POSIX read.  So it
            # inherits ``_ds_id``'s hazard verbatim: entering HDF5/MPI-IO
            # on this file handle while the asynchronous writer thread is
            # still inside it interleaves MPI's datatype-cache state, and
            # a rank that opens while a peer writes mismatches the
            # collective order.  ``read_slab`` / ``read_slabs`` /
            # ``padded_shape_for`` already drain before reaching here;
            # ``read_whole`` reaches it directly, which is why the drain
            # belongs at THIS choke point rather than at each caller.
            self._drain_pending()
            with _journal.op_scope("attr_r", self.path, stack=_J_FFI,
                                   ds=name, mode=self.mode, handle=self.fh):
                shape, dtype_name = self._loader.phdf5_dataset_geometry(
                    self.fh, name, platform=self._platform())
            got = (tuple(int(s) for s in shape), np.dtype(dtype_name))
            cache[name] = got
            return got

        if self.mode != "r":
            raise RuntimeError(
                f"SlabIO({os.path.basename(self.path)}, mode={self.mode!r}): "
                f"learning the geometry of dataset {name!r} needs either the "
                f"FFI metadata entry points (lrx_phdf5_dataset_geometry, "
                f"added 2026-08-22) or a serial-h5py open of a file this "
                f"handle can WRITE — and the second is refused by "
                f"file_io.hdf5_owner, correctly (audit A1; two HDF5 library "
                f"instances, one file, one of them a writer).\n"
                f"  fix= rebuild the FFI library from this tree "
                f"(src/ffi/cpp/build.sh for the CUDA leg, "
                f"config/perlmutter/build_ffi_host.sh for the host leg), or "
                f"pre-register the geometry on this handle with "
                f"create_dataset({name!r}, shape=..., dtype=...) — which is "
                f"idempotent for an identical existing dataset and is the "
                f"ordering file_io/parallel_transport.py already uses.")

        _announce_legacy_introspect(self.path)
        import h5py

        from .hdf5_owner import STACK_H5PY, open_scope
        # Read-only on both sides, which is the one cross-stack overlap the
        # registry allows.  It is still counted, and it is still the route
        # this method exists to retire.
        with open_scope(self.path, STACK_H5PY, "r",
                        where=f"_FfiBackend._introspect_dataset({name!r})"), \
                _journal.op_scope("attr_r", self.path, stack=_J_H5PY,
                                  ds=name, mode="r", handle=self.fh):
            with h5py.File(self.path, "r") as f:
                ds = f[name]
                shape = tuple(int(s) for s in ds.shape)
                dtype = np.dtype(ds.dtype)
        cache[name] = (shape, dtype)
        return shape, dtype

    def _platform(self) -> str | None:
        """Which FFI library owns this handle ("CUDA"/"cpu"), or None.

        Every lifecycle call on a ``PhdfCtx*`` must go through the library
        that allocated it (``ffi.io.platform_for_handle``); the two new
        metadata calls are lifecycle calls like any other.
        """
        from ffi.io import platform_for_handle
        return platform_for_handle(self.fh)

    def read_whole(self, name: str, *, dtype=None):
        """Read a WHOLE small dataset into a host ``np.ndarray``, every rank.

        The rank-0 / scalar door.  ``read_slab`` cannot serve a scalar: a
        rank-0 dataspace has no hyperslab to select, and the request is
        refused before it reaches HDF5 ("slab shape must be non-empty").
        Every stamp ``write_attr`` publishes is such a dataset, which is
        why ``gw.qsgw_head.load_parallel_transport_head`` could never read
        its own artifact.

        FOR SCALARS AND SMALL REPLICATED VECTORS.  See
        ``ffi_loader.phdf5_read_whole``; the payload must be O(1) in the
        design envelope because every rank materialises all of it.

        COLLECTIVE in the same sense as :meth:`read_slab`: the geometry
        query behind it is a collective ``H5Dopen``, so every rank calls
        this with the same name, in the same order.  It drains queued
        writes first for the reasons :meth:`read_slab` lists — nothing
        here may enter HDF5 while the writer thread is inside it.
        """
        self._drain_pending()
        shape, ds_dtype = self._dataset_geom(name)
        want = np.dtype(dtype) if dtype is not None else np.dtype(ds_dtype)
        if not self._loader.has_phdf5_metadata_api(self._platform()):
            # SAME LEGACY ROUTE AS ``_introspect_dataset``, and for the same
            # reason it has one: on a READ-ONLY handle a serial-h5py open is
            # the one cross-stack overlap ``hdf5_owner`` allows, and it is
            # what ``gw.qsgw_head`` was doing on origin/main -- through its
            # own short-lived owner -- when this method did not exist.
            # Refusing here instead would take a path that WORKS today on
            # every deployed .so and break it until a rebuild lands, on the
            # metallic-head route (`sc_head_update = dft_velocity`) every
            # accepted sodium number came through.  A ratchet belongs on the
            # artifact, not on a caller who has no way to satisfy it.
            #
            # On a WRITABLE handle there is still nothing legal to fall back
            # to (two HDF5 instances, one file, one of them a writer -- audit
            # A1), so that arm keeps the refusal and names the rebuild.
            if self.mode != "r":
                raise RuntimeError(
                    f"SlabIO.read_small({name!r}) on "
                    f"SlabIO({os.path.basename(self.path)}, "
                    f"mode={self.mode!r}): the loaded FFI library predates "
                    f"the metadata entry points (lrx_phdf5_read_whole, "
                    f"2026-08-22), and the serial-h5py fallback is legal "
                    f"only while BOTH stacks are read-only -- this handle "
                    f"can write, so file_io.hdf5_owner refuses it, "
                    f"correctly.\n"
                    f"  fix= rebuild the FFI library from this tree "
                    f"(src/ffi/cpp/build.sh for the CUDA leg, "
                    f"config/perlmutter/build_ffi_host.sh for the host leg), "
                    f"or reopen this file mode='r' for the stamp read.")
            _announce_legacy_introspect(self.path)
            import h5py

            from .hdf5_owner import STACK_H5PY, open_scope
            with open_scope(self.path, STACK_H5PY, "r",
                            where=f"_FfiBackend.read_whole({name!r})"), \
                    _journal.op_scope("read", self.path, stack=_J_H5PY,
                                      ds=name, mode="r", handle=self.fh):
                with h5py.File(self.path, "r") as f:
                    out = np.asarray(f[name][()])
            return out.astype(want, copy=False) if dtype is not None else out
        # Journaled by ``SlabIO.read_small``, the public door — one line
        # per op, as for every other method here.
        return self._loader.phdf5_read_whole(
            self.fh, name, shape=shape, dtype_name=str(want.name),
            platform=self._platform())

    def _ds_id(self, name: str, readonly: bool = False) -> int:
        if name in self._ds_ids:
            return self._ds_ids[name]
        # ``phdf5_open_dataset_ro`` is collective on the file handle
        # — same MPI rendezvous + datatype-cache hazard as
        # ``phdf5_ensure_dataset`` (see :meth:`create_dataset`).
        self._drain_pending()
        if readonly:
            with _journal.op_scope("open", self.path, stack=_J_FFI, ds=name,
                                   mode="r", handle=self.fh):
                ds_id = self._loader.phdf5_open_dataset_ro(self.fh, name)
        else:
            raise RuntimeError(
                f"dataset '{name}' not registered — call create_dataset first")
        self._ds_ids[name] = ds_id
        return ds_id

    def _dataset_geom(self, name: str) -> tuple[tuple[int, ...], "np.dtype"]:
        """``(shape, dtype)`` of ``name``, from the record where possible.

        The record FIRST, and the serial-h5py introspect only for a
        dataset this handle did not create.  That ordering is not a
        micro-optimisation: opening this path with h5py while the same
        file is open for collective MPI-IO writing reads a superblock
        that is not durable yet, and h5py fails with "file signature not
        found" (measured, job 7888644 probe/raw, when an unconditional
        introspect was tried here).  A dataset this handle created is
        already described by the record, so the read-after-write-on-one-
        handle case never reaches the file.
        """
        got = self._ds_geom.get(str(name))
        if got is not None:
            return got
        shape, dtype = self._introspect_dataset(name)
        self._remember_geom(name, shape, dtype)
        return tuple(int(s) for s in shape), np.dtype(dtype)

    # ------------------------------------------------------------------
    # FFI write padding contract: shard ``A`` with equal local blocks,
    # then let C++ clip each rank's file hyperslab to ``valid_shape``,
    # which SlabIO derives from the dataset's own extent.
    def write_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
        global_shape: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        dtype=None,
    ) -> None:
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        # Ensure placement: if not sharded on our mesh, put as replicated.
        # Process-local (see _replicated_i64_vector): a host/uncommitted
        # operand here would otherwise pay the hidden assert_equal
        # all-gather at P × A.nbytes — on a WRITE-path tensor, the
        # single biggest assertion payload in the codebase (AA.1 class).
        # A replicated write requires rank-identical A anyway (the
        # collective writer dedups replicas); LORRAX_CHECK_REPLICA=1
        # re-arms the assertion.
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = device_put_process_local(
                A, _replicated_sharding(self.mesh, A.ndim))

        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            A.sharding, A.ndim)
        off, slab_shape, req_gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=A.shape, global_shape=global_shape,
            check_bounds=False)
        mesh_shape = tuple(self.mesh.shape[ax] for ax in self.mesh.axis_names)
        _validate_block_divisible(
            op="write_slab", name=name, shape=slab_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)

        # The dataset's LOGICAL extent, in this order of authority:
        #   1. what this handle already created/opened for ``name``;
        #   2. the caller's ``global_shape`` (creating the dataset now);
        #   3. ``A.shape`` (whole-dataset write of a fresh dataset).
        # A caller that states BOTH must agree with the file, or the
        # dataset it thinks it is writing is not the one on disk.
        ds_shape = self._known_shape(name)
        if ds_shape is None:
            ds_shape = req_gshape
            self._drain_pending()
            # The SECOND ensure_dataset site — a write that creates its
            # own dataset.  Journaled like the first: an ``ensure`` is a
            # collective H5Dcreate, i.e. file metadata, and a create this
            # path made is one the reader of the log will otherwise not
            # find any ``create`` line for.
            with _journal.op_scope("create", self.path, stack=_J_FFI,
                                   ds=name,
                                   cnt=tuple(int(s) for s in ds_shape),
                                   mode=self.mode, handle=self.fh):
                ds_id = self._loader.phdf5_ensure_dataset(
                    self.fh, name, tuple(int(s) for s in ds_shape),
                    str(jnp.dtype(A.dtype).name),
                )
            self._ds_ids[name] = ds_id
            self._remember_geom(name, ds_shape, A.dtype)
        elif global_shape is not None and req_gshape != ds_shape:
            raise ValueError(
                f"write_slab {name!r}: global_shape={req_gshape} contradicts "
                f"the dataset's extent {ds_shape}.  global_shape is only "
                f"needed to CREATE a dataset; drop it and SlabIO uses the "
                f"dataset's own shape.")

        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, ds_shape=ds_shape)
        gshape = ds_shape

        if debug_print_enabled():
            import sys
            local_shapes = [tuple(s.data.shape) for s in A.addressable_shards]
            sys.__stdout__.write(
                f"[ffi-debug proc={jax.process_index()}] "
                f"name={name} shape={tuple(A.shape)} dtype={A.dtype} "
                f"spec={getattr(A.sharding, 'spec', None)} "
                f"offset={off} valid_shape={vshape} gshape={gshape} "
                f"local_shapes={local_shapes}\n")
            sys.__stdout__.flush()


        ds_id = self._ds_ids[name]
        ctx_handle = self.fh
        in_specs = A.sharding.spec  # PartitionSpec

        # Module-level lru_cache shared across all _FfiBackend instances.
        # Keys on the FFI signature (mesh / sharding / geometry) ONLY:
        # ctx_handle, ds_id, offset and valid_shape are all RUNTIME args,
        # so one compile serves every chunk, dataset, file and process.
        sm = _get_write_sm(
            self.mesh, in_specs,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            no_jit=bool(os.environ.get('LORRAX_WRITE_NO_JIT')),
        )

        # Enqueue dispatch onto the Python worker thread.  Main thread
        # returns in ~0.2ms; the worker thread calls ``sm(A, offset)``
        # in FIFO order.  The offset Buffer is tiny (ndim × 8 bytes).
        # Same replicated-control-buffer path as offset/valid_shape:
        # device_put_process_local, so this does NOT add a per-call
        # blocking all-gather at P>1 (scorecard AA.1).
        handle_arr = _replicated_i64_vector((ctx_handle, ds_id), self.mesh)
        offset_arr = _replicated_i64_vector(off, self.mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, self.mesh)

        def _task():
            tok = sm(A, handle_arr, offset_arr, valid_shape_arr)
            tok.block_until_ready()

        self._queued_bytes += int(A.nbytes)
        self._dispatcher.submit(_task)

    # ------------------------------------------------------------------
    # FFI read padding contract: output ``shape`` is equal-block
    # sharded; C++ reads only ``valid_shape`` and zero-fills the rest.
    def padded_shape_for(self, name: str, *, mesh: Mesh, partition_spec: P
                         ) -> tuple[int, ...]:
        """The dataset's shape rounded UP to be shardable by ``partition_spec``.

        What ``SlabIO.read_slab`` uses when the caller omits ``shape``.
        Kept on the backend rather than in ``slab_io.py`` because it needs
        the dataset geometry, which only the backend knows; the rounding
        rule itself is :func:`mesh_divisible_shape`, single-sourced.
        """
        self._drain_pending()
        ds_shape, _ = self._dataset_geom(name)
        return mesh_divisible_shape(ds_shape, mesh, partition_spec)

    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
        as_numpy: bool = False,  # accepted for signature compatibility;
        # the public SlabIO.read_slab handles the numpy conversion.
    ) -> jax.Array:
        mesh = mesh or self.mesh
        # Drain queued writes BEFORE anything in this method touches the
        # file.  Three distinct hazards, all of them silent:
        #
        #  1. Read-after-write ordering.  ``write_slab`` only ENQUEUES; a
        #     read issued before the queue drains sees the pre-write bytes
        #     with no error anywhere.
        #  2. ``ctx->pinned_buf`` is one buffer shared by the writer thread
        #     and this read.  ``ReadImpl`` runs SYNCHRONOUSLY on the XLA
        #     thread and starts with ensure_pinned + memset; on the CUDA
        #     build that is the very buffer an in-flight H5Dwrite is reading
        #     from, and ensure_pinned may free and realloc it underneath.
        #  3. Two threads inside HDF5/MPI-IO on one file handle.  This is
        #     the hazard ``create_dataset`` and ``_ds_id`` already drain for
        #     ("MPI's datatype-cache state on the file handle interleaves"),
        #     and worse for a collective transfer: rank A doing read-then-
        #     write while rank B does write-then-read mismatches the
        #     MPI-IO collective order and hangs.
        #
        # ``_ds_id`` drains too, but only on the first sight of a dataset
        # name — a read of an ALREADY-cached dataset skipped every drain in
        # the method, and ``_introspect_dataset`` (serial h5py on the same
        # path) ran before even that.  One unconditional drain at the top.
        self._drain_pending()
        ds_shape, ds_dtype = self._dataset_geom(name)
        if shape is None:
            # Symmetry with the allgather backend: callers that don't
            # need padding shouldn't have to compute shape themselves.
            shape = ds_shape
        if dtype is None:
            dtype = ds_dtype
        off, read_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=shape, global_shape=None, check_bounds=False)

        # Default: fully replicated.  Caller can provide partition_spec
        # to shard the read.
        if partition_spec is None:
            partition_spec = P(*([None] * len(read_shape)))
        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(read_shape))
        mesh_shape = tuple(mesh.shape[ax] for ax in mesh.axis_names)

        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=read_shape, offset=off, ds_shape=ds_shape)
        _validate_block_divisible(
            op="read_slab", name=name, shape=read_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)

        # Per-rank output shape: divide by the product of the mesh
        # sizes of all axes sharding that dim.
        local_shape = list(read_shape)
        _flat_idx = 0
        for d in range(len(read_shape)):
            na = axis_count_per_dim[d]
            if na > 0:
                div = 1
                for k in range(na):
                    div *= int(mesh_shape[axis_flat[_flat_idx + k]])
                local_shape[d] = int(local_shape[d]) // div
                _flat_idx += na
        out_struct = jax.ShapeDtypeStruct(tuple(local_shape), jnp.dtype(dtype))

        ds_id = self._ds_id(name, readonly=True)
        ctx_handle = self.fh

        # Module-level lru_cache shared across all _FfiBackend instances.
        # ctx_handle / ds_id are runtime args, not part of the key.
        sm = _get_read_sm(
            mesh, partition_spec,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            out_struct=out_struct,
        )

        handle_arr = _replicated_i64_vector((ctx_handle, ds_id), mesh)
        offset_arr = _replicated_i64_vector(off, mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, mesh)
        result = sm(handle_arr, offset_arr, valid_shape_arr)
        result.block_until_ready()
        return result

    # ------------------------------------------------------------------
    # n windows, ONE H5Dread.  Same padding contract as read_slab, one
    # valid extent per window; see SlabIO.read_slabs for the measurement
    # that put this behind the door instead of folding it into n read_slab
    # calls.
    def read_slabs(
        self,
        name: str,
        *,
        shape: Sequence[int],
        offsets,
        valid_shapes,
        partition_spec: P,
        window_axis: int,
        dtype=None,
        mesh: Mesh | None = None,
    ) -> jax.Array:
        from ffi.io import read_kchunk_union_sharded

        mesh = mesh or self.mesh
        # Same unconditional drain, for the same three hazards, as
        # :meth:`read_slab` — read-after-write ordering, the shared
        # ``ctx->pinned_buf``, and two threads inside HDF5/MPI-IO on one
        # file handle.  A multi-window read is one collective like any
        # other, so none of them get weaker.
        self._drain_pending()
        ds_shape, ds_dtype = self._dataset_geom(name)
        slab_shape = tuple(int(s) for s in shape)
        if dtype is None:
            dtype = ds_dtype
        offsets_t, valid_t = _normalize_window_tables(
            name=name, offsets=offsets, valid_shapes=valid_shapes,
            ndim=len(slab_shape))

        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(slab_shape))
        mesh_shape = tuple(int(mesh.shape[ax]) for ax in mesh.axis_names)
        _validate_block_divisible(
            op="read_slabs", name=name, shape=slab_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)
        divs = _shard_divisors(
            axis_count_per_dim=axis_count_per_dim, axis_flat=axis_flat,
            mesh_shape=mesh_shape, ndim=len(slab_shape))
        per_rank_shape = tuple(s // d for s, d in zip(slab_shape, divs))
        counts = _derive_window_counts(
            per_rank_shape=per_rank_shape,
            rank_offsets=_rank_block_offsets(
                per_rank_shape=per_rank_shape,
                axis_count_per_dim=axis_count_per_dim,
                axis_flat=axis_flat, mesh_shape=mesh_shape),
            valid_shapes=valid_t)

        # BOTH tables staged PROCESS-LOCALLY.  ``jax.device_put`` of host
        # numpy onto a multi-process sharding runs JAX's hidden
        # ``assert_equal`` all-gather (scorecard AA.1/Y.3) — a blocking
        # collective per call on buffers that are identical (offsets) or
        # exactly this rank's own rows (counts) by construction.  The counts
        # table is the expensive one to assert: P x (world x n_windows) x
        # ndim x 8 B is O(P^2), 18.9 MB at P=64 and 95.5 MB at P=144.
        # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
        counts_spec = P(tuple(mesh.axis_names), None)
        offsets_dev = device_put_process_local(
            offsets_t, NamedSharding(mesh, P(None, None)))
        counts_dev = device_put_process_local(
            counts, NamedSharding(mesh, counts_spec))

        reader = read_kchunk_union_sharded(
            self.fh, name,
            n_kchunk=int(offsets_t.shape[0]),
            kchunk_axis=int(window_axis),
            file_global_shape=ds_shape,
            per_rank_file_shape=per_rank_shape,
            dtype=dtype,
            mesh=mesh,
            file_partition_spec=partition_spec,
            count_partition_spec=counts_spec,
        )
        # No ``block_until_ready`` here, unlike read_slab: the union handler
        # returns an async ``ffi::Future`` and the caller's next op is what
        # sequences it (measured ~1% end-to-end, read_ffi.cc:819-829).
        # Blocking would be a behaviour change dressed as symmetry.
        return reader(offsets_dev, counts_dev)

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Drain pending writes on the Python worker thread, then stop
        # the worker, THEN close the MPI-IO handle.  Order matters:
        # close_ctx() in C++ also drains its own task queue, but an
        # in-flight Python-side jit dispatch could still be holding a
        # reference to ctx_handle when we call close_file below.
        #
        # The drain can take minutes for multi-GB writes (N collective
        # MPI-IO calls serialised through one writer thread per ctx).
        # Print per-stage timings on rank 0 so a long drain doesn't
        # look like a hang.
        import time as _time
        _rank0 = (jax.process_index() == 0)
        _log_level = _close_log_level() if _rank0 else 0
        _verbose = _log_level >= 2
        _pending = self._dispatcher.pending
        if _log_level and _pending:
            print(f"  [SlabIO.close] draining {_pending} pending writes "
                  f"for {os.path.basename(self.path)} …", flush=True)
        # ── A rank must not skip a collective because of its OWN error ──
        # decisions.md 2026-08-04.  A worker exception surfaces on this
        # rank's ``drain()``; if it propagated from here it would skip the
        # collective ``H5Fclose`` below on THIS rank only, and the peers
        # would sit inside it with no message.  ``AsyncDispatcher.
        # _raise_if_error`` also CLEARS the error as it raises, so nothing
        # downstream would re-raise it either.  Record it, complete the
        # teardown every rank is inside, then raise at the end.
        _worker_error: BaseException | None = None
        _drained_bytes = 0
        _t0 = _time.perf_counter()
        try:
            _drained_bytes = self._drain_pending()
        except BaseException as exc:                          # noqa: BLE001
            _worker_error = exc
        _t_drain = _time.perf_counter() - _t0
        if _verbose:
            # The size and the rate belong HERE and only here.  A caller
            # timing its own write_slab() is timing the enqueue, so the
            # rate it can compute is a fiction of the queue depth; this
            # is the first point at which the bytes are on disk.
            _moved = (f"{_drained_bytes / 1e9:.2f} GB at "
                      f"{_drained_bytes / 1e6 / max(_t_drain, 1e-9):.0f} MB/s"
                      if _drained_bytes else "no queued data")
            print(f"  [SlabIO.close] Python dispatch drained in "
                  f"{_t_drain:.1f} s ({_moved}); joining writer thread",
                  flush=True)
        _t0 = _time.perf_counter()
        try:
            self._dispatcher.close()            # drain + poison pill + join
        except BaseException as exc:                          # noqa: BLE001
            if _worker_error is None:
                _worker_error = exc
        _t_join = _time.perf_counter() - _t0
        if _worker_error is not None:
            print(f"  [SlabIO.close rank={jax.process_index()}] write worker "
                  f"raised {type(_worker_error).__name__}: {_worker_error} — "
                  f"completing the collective teardown before re-raising.",
                  flush=True)
        if self.fh:
            if _verbose:
                print(f"  [SlabIO.close] writer thread joined in "
                      f"{_t_join:.1f} s; calling H5Fclose collectively",
                      flush=True)
            _t0 = _time.perf_counter()
            # ISSUE-TIME, and this is the one that matters most: S1 is a
            # SIGSEGV that lands in the writer-thread join inside this
            # very call.  A journal whose last line is this one names the
            # ctx handle that died (SLAB_IO_ROOT_CAUSE_AUDIT.md §A/S1).
            with _journal.op_scope("close", self.path, stack=_J_FFI,
                                   mode=self.mode, handle=self.fh):
                self._close_file(self.fh)
            self.fh = 0
            _t_close = _time.perf_counter() - _t0
            if _verbose:
                print(f"  [SlabIO.close] H5Fclose returned in "
                      f"{_t_close:.1f} s", flush=True)
        else:
            _t_close = 0.0
        # RELEASE THE FFI CLAIM HERE, between H5Fclose and the rank-0
        # h5py reopen below — not at the end of the method.  The reopen
        # is the OTHER HDF5 library instance touching this same path, and
        # it is legal precisely because MPI-IO has already let go; a claim
        # still held across it would make this method refuse itself, and
        # correctly so.  Unconditional (not inside ``if self.fh``) so a
        # double close or a handle that never opened still releases.
        from .hdf5_owner import STACK_H5PY, note_close, open_scope
        if getattr(self, "_owner_token", None) is not None:
            note_close(self.path, self._owner_token)
            self._owner_token = None
        # Now that MPI-IO has released the file, rank 0 can safely
        # reopen with h5py to tack on the deferred small-metadata
        # datasets (omega_ev and friends) and the deferred dataset
        # attributes (``k_storage`` and friends).  ONE reopen for both:
        # they are deferred for the same reason and land in the same
        # place, and a second open would be a second thing to keep in
        # step with the barrier below.
        #
        # The rank-0 h5py block is gated on the deferred lists; the
        # BARRIER is not, and must not be.  Those lists are per-rank
        # Python lists, so gating a collective on them makes the number
        # of barriers a rank executes depend on that rank's own control
        # flow — the deadlock shape this audit is looking for.  Today
        # every ``write_attr`` / ``create_dataset`` call site is SPMD so
        # the lists are the same everywhere, but that is a property of
        # the callers, not of this method, and it is not checkable here.
        # An unconditional barrier costs one rendezvous per file close
        # and removes the question.
        if (_worker_error is None
                and (self._deferred_attrs or self._deferred_ds_attrs)
                and jax.process_index() == 0):
            import h5py
            with open_scope(self.path, STACK_H5PY, "a",
                            where="_FfiBackend.close deferred-attr reopen"), \
                    _journal.op_scope(
                        "attr_w", self.path, stack=_J_H5PY, mode="a",
                        cnt=(len(self._deferred_attrs),
                             len(self._deferred_ds_attrs))), \
                    h5py.File(self.path, "a") as h5:
                for name, value in self._deferred_attrs:
                    if name in h5:
                        del h5[name]
                    host = value
                    if not isinstance(host, np.ndarray):
                        host = np.asarray(jax.device_get(host))
                    h5.create_dataset(name, data=host)
                # AFTER the small datasets, because that loop
                # delete-and-recreates by name and a recreated dataset
                # would come back stripped of anything stamped first.
                _apply_dataset_attrs(h5, self._deferred_ds_attrs)
                # Explicit flush before close: this is the ONE serial-h5py
                # write onto a file the FFI also drives, so its bytes must
                # be durable before any rank's next collective open sees
                # the superblock (audit A1 item 3).
                h5.flush()
        # Same reason as the write-ordering barriers above: rank 0 may
        # have just rewritten datasets in this file with serial h5py, and
        # no other rank may reopen it until that is durable.
        _barrier("slab_io_ffi_close_attrs")
        self._deferred_attrs = []
        self._deferred_ds_attrs = []
        if _worker_error is not None:
            raise _worker_error
