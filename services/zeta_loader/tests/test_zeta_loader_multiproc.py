"""Layer L-c: the FIRST-EVER multi-device ``ZetaLoader`` executions.

Survey §7.2 G1, quoted in full because it is the whole reason this file
exists:

    **No ZetaLoader test ever runs on a real multi-device mesh.**  Every one
    uses ``single_device_mesh`` (1x1) or ``mesh=None``.
    ``test_file_io.py:457-460`` says so explicitly: the μ-pad cases exist
    but at P=1 the per-rank clipping is only exercised at caller level.  The
    only 4-device work in the family (``test_zeta_mesh_invariance``'s
    subprocess workers) never opens a ζ file. → **The band-pad bug class is
    OPEN for zeta.**

The charter makes hostile geometry MANDATORY for mesh-touching services, and
this service is mesh-touching in every data read.  The five cells below are
survey §7.3's list, one for one::

    lx run --cpu -N 1 -n 4 python3 \\
        services/zeta_loader/tests/test_zeta_loader_multiproc.py \\
        --mesh 2x2 --tmpdir $SCRATCH/svc_zeta/l_c --report $SCRATCH/.../l_c.json

ONE SET OF CHECK BODIES, TWO CALLERS — the ``_CLI_CELLS`` pattern copied from
``services/distrib_la/tests/test_distrib_la_multiproc.py`` rather than
reinvented.  Every ``check_*(mesh, tmpdir)`` below is called by a pytest cell
(on whatever mesh this process can build: a real emulated 2x2 when the
service suite runs BY PATH, 1x1 in the full-suite run) AND by ``_cli_main``
under ``srun``.  Duplicating the logic across the two would mean the
multi-rank leg tests something slightly different from the thing the suite
pins, which is how a matrix leg drifts out of agreement with its own
reference.

EVERY CELL DEGENERATES GRACEFULLY AT 1x1 and says which geometry it got.  A
cell that silently reported PASS from a 1x1 run would be the exact defect
this file was written to close — so each one returns the mesh it ran on and
the geometry facts it observed, and the anti-tautology self-assertions
(``the fixture must actually pad``) fire at every process count.

THE FILE IS THE SHARED-FS CONTRACT TOO.  Under ``srun`` the ζ files are built
by RANK 0 ONLY and read by everybody, so every builder call is followed by
``sync_global_devices``: a rank that opened the file before rank 0 finished
writing it would read a torn header, and the failure would look like a
loader bug.  ``--tmpdir`` must therefore be a path every rank can see
($SCRATCH, not /tmp).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Path first: this file runs as a bare script under srun, where nothing has
# put services/*/src anywhere (`lx` rewrites the container PYTHONPATH to
# exactly <checkout>/src).
_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)
for _p in (os.path.join(_SERVICES, "lxkit", "src"),
           os.path.join(_SERVICES, "zeta_loader", "src"),
           os.path.join(_REPO, "src"), _TESTS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# CLI multi-rank mode: jax.distributed.initialize must run before ANY
# XLA-backend touch, so it happens at import time when this module is the
# entry point of a multi-task launch.  Same order as distrib_la's CLI mode.
# The ζ path is HOST-side I/O (phdf5 over MPI-IO), so the CPU backend is the
# one that matters and there is no per-platform library warm-up to do.
if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        import jax
        jax.distributed.initialize()

import zeta_synth as Z                                         # noqa: E402


# ---------------------------------------------------------------------------
# Multi-process plumbing
# ---------------------------------------------------------------------------

def _barrier(tag: str) -> None:
    """Every rank waits.  A no-op at one process, by construction.

    ``sync_global_devices`` is what ``gw.isdf_fitting`` itself uses between
    the collective create and the rank-0 header append
    ("zeta_fit_headers_written"), so the fixture builds here use the same
    barrier the production writer does rather than a second mechanism with
    its own failure modes.
    """
    import jax
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)


def _build_on_rank0(path: str, build):
    """Rank 0 writes the ζ file; everybody waits; everybody gets the path.

    NOT "every rank writes its own copy": ``--tmpdir`` is a SHARED path and
    four ranks creating the same HDF5 concurrently with serial h5py is file
    corruption, which is the worst class of defect this codebase has.  The
    payload the other ranks need for comparison is recomputed locally —
    every builder in ``zeta_synth`` is deterministic, so a rank can know
    what is on disk without being told.
    """
    import jax
    if jax.process_index() == 0:
        build(path)
    _barrier(f"zeta_synth_built:{os.path.basename(path)}")
    return path


def _gather(x):
    """A sharded ``jax.Array`` -> full host numpy, multi-process aware."""
    import jax
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(x))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _mesh_shape(mesh) -> tuple[int, int]:
    return int(mesh.shape["x"]), int(mesh.shape["y"])


def _world(mesh) -> int:
    px, py = _mesh_shape(mesh)
    return px * py


def _per_rank_shards_agree_with(x, ref: np.ndarray, what: str) -> int:
    """Compare EVERY addressable shard against the host reference it covers.

    The gathered comparison says the whole window came back right; this says
    each RANK's own tile did, which is the claim a band-pad bug actually
    breaks.  ``shard.index`` is the slice tuple the shard occupies in the
    global array, so indexing the reference with it needs no arithmetic here
    — arithmetic that would be the same arithmetic under test.
    """
    n = 0
    for shard in x.addressable_shards:
        got = np.asarray(shard.data)
        want = ref[shard.index]
        assert got.shape == want.shape, (
            f"{what}: shard {shard.index} shape {got.shape} != {want.shape}")
        assert got.tobytes() == want.tobytes(), (
            f"{what}: shard at {shard.index} differs from the h5py "
            f"reference on device {shard.device}")
        n += 1
    assert n > 0, f"{what}: this process addressed no shards at all"
    return n


class _raises:
    """``pytest.raises`` that also works in the pytest-free CLI mode."""

    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(
                f"expected {self.exc} matching {self.match!r}, nothing raised")
        if not issubclass(et, self.exc if isinstance(self.exc, tuple)
                          else (self.exc,)):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"{et.__name__} raised but {self.match!r} not in {str(ev)!r}")
        return True


# ===========================================================================
# CELL 1 — μ padded past the on-disk extent  (survey §7.3 row 1)
# ===========================================================================

def check_mu_pad_non_divisible(mesh, tmpdir):
    """``mu_count`` past the on-disk μ extent, at three hostile geometries.

    ``v_q_g_flat.py:264`` reads at ``n_rmu_padded`` and SlabIO zero-fills
    past the on-disk extent (decisions.md 2026-08-04).  At P=1 that clipping
    happens once, at the caller; at P=4 it happens PER RANK, and which rank
    straddles the logical boundary depends on the geometry.  The three cases
    are the three distinct geometries, not three sizes:

    ``300 in 304``  the production shape (CrI3 6x6 30Ry bispinor, named in
        ``test_file_io.py:459``).  At 4 ranks the per-rank tile is 76 rows
        and 300 = 3*76 + 72, so the boundary falls STRICTLY INSIDE rank 3's
        tile — the only geometry where a rank must clip mid-tile, and the
        one a caller-level check can never reach.
    ``3 in 4``      the boundary lands exactly on a rank EDGE: ranks 0-2 are
        all payload, rank 3 is all zeros.
    ``1 in 4``      fewer logical μ than ranks: rank 0 has one row, ranks
        1-3 have nothing but zeros.  The degenerate case, and the one where
        an off-by-one in the clip returns garbage instead of an error.

    Returns the observed geometry per case so a 1x1 run cannot be mistaken
    for a 2x2 one in the report.
    """
    from zeta_loader import ZetaLoader

    world = _world(mesh)
    out = {"mesh": "%dx%d" % _mesh_shape(mesh), "cases": []}
    interior = 0
    for n_log, mu_count in ((300, 304), (3, 4), (1, 4)):
        assert mu_count % world == 0, (
            f"mu_count={mu_count} must divide the {world}-device mesh; "
            f"P(None, ('x','y'), None) shards μ over every device")
        assert mu_count > n_log, "the fixture must actually pad"
        n_q, ngkmax = 2, 4
        path = os.path.join(tmpdir, f"zeta_mu_{n_log}_in_{mu_count}.h5")
        _build_on_rank0(path, lambda p, n=n_log: Z.build_gflat(
            p, n_q=n_q, n_rmu=n, ngkmax=ngkmax))
        expect = Z.make_payload(n_q, n_log, ngkmax)

        per_rank = mu_count // world
        where = ("interior-to-a-rank-tile" if n_log % per_rank
                 else "on-a-rank-edge")
        interior += 1 if n_log % per_rank else 0

        with ZetaLoader(path, mesh=mesh) as zl:
            assert int(zl.n_rmu) == n_log and int(zl.n_rmu_disk) == n_log
            z = zl.read_zeta_G_slab(q_offset=0, q_count=n_q,
                                    mu_offset=0, mu_count=mu_count)
            host = _gather(z)
            # PER RANK, not only gathered: build the reference for the whole
            # padded window and hand each shard its own slice of it.
            ref = np.zeros((n_q, mu_count, ngkmax), dtype=np.complex128)
            ref[:, :n_log, :] = expect
            shards = _per_rank_shards_agree_with(
                z, ref, f"mu_pad {n_log} in {mu_count}")

        assert host.shape == (n_q, mu_count, ngkmax), host.shape
        assert np.array_equal(host[:, :n_log, :], expect), (
            f"{n_log} in {mu_count}: the payload rows moved")
        assert not host[:, n_log:, :].any(), (
            f"{n_log} in {mu_count}: pad rows are not EXACT zeros "
            f"(max |.| = {np.abs(host[:, n_log:, :]).max():.3e}) — this is "
            f"the band-pad bug class, per rank")
        out["cases"].append(dict(n_rmu_logical=n_log, mu_count=mu_count,
                                 per_rank=per_rank, boundary=where,
                                 shards=shards))
    # ANTI-TAUTOLOGY: at least one case must put the logical boundary
    # strictly inside a rank's tile, or the per-rank clip was never
    # exercised and this cell measured the caller-level contract again.
    # RECORDED as well as asserted, because at ONE device every boundary is
    # trivially interior to the single tile — the assertion passes and means
    # nothing there, so a 1x1 report must not read as a hostile-geometry
    # result.  ``test_file_io.py:457-460`` is exactly that situation ("at
    # P=1 the per-rank clipping is only exercised at caller level"), and
    # this flag is what keeps the distinction visible in the artifact.
    out["hostile_mu_boundary"] = bool(world > 1 and interior >= 1)
    if world > 1:
        assert interior >= 1, (
            f"no case put the μ boundary inside a rank tile on a "
            f"{world}-device mesh, so the per-rank clipping was not "
            f"exercised: {out['cases']}")
    return out


# ===========================================================================
# CELL 2 — q windows on a non-dividing q axis  (survey §7.3 row 2)
# ===========================================================================

def check_q_window_non_divisible(mesh, tmpdir):
    """``n_q_on_disk = 74`` with ``74 % 4 != 0``, three windows, byte-equal.

    The q axis is ``P(None, ...)`` — REPLICATED — so nothing forces it to
    divide the mesh, and ``_read_g_flat_disk`` takes an arbitrary
    ``q_offset``/``q_count`` window into it.  74 is the live IBZ count
    ``test_zeta_mesh_invariance._worker_cap:291`` already uses, and the
    three windows are the three shapes a caller produces: everything, an
    interior slice that starts off a rank boundary, and the LAST single q
    (where an off-by-one reads past the dataset).

    The reference is a serial h5py hyperslab of the same window, compared
    BYTE FOR BYTE — not to a tolerance.  Neither plan reduces, so anything
    but byte identity is a transport defect and a tolerance would hide it.
    """
    import h5py as h5
    from zeta_loader import ZetaLoader

    n_q, n_rmu, ngkmax = 74, 4, 4
    world = _world(mesh)
    assert n_rmu % world == 0, f"n_rmu={n_rmu} must divide {world} devices"
    path = os.path.join(tmpdir, "zeta_q74.h5")
    _build_on_rank0(path, lambda p: Z.build_gflat(
        p, n_q=n_q, n_rmu=n_rmu, ngkmax=ngkmax, kgrid=(2, 2, 2)))

    out = {"mesh": "%dx%d" % _mesh_shape(mesh), "n_q": n_q, "windows": []}
    # ANTI-TAUTOLOGY, and it DEGENERATES rather than failing at one device.
    # Every extent divides a 1-device mesh, so demanding non-divisibility
    # unconditionally would make this cell red at 1x1 for a reason that has
    # nothing to do with the loader — and a cell that cannot run at 1x1
    # cannot be smoke-tested before the batch queue.  At P>1 the claim is
    # real and is asserted; at P=1 it is recorded as NOT MADE, so a 1x1
    # report cannot be read as a hostile-geometry result.
    out["hostile_q_axis"] = bool(world > 1 and n_q % world != 0)
    if world > 1:
        assert n_q % world != 0, (
            f"n_q={n_q} divides the {world}-device mesh, so this cell is "
            f"not testing a non-dividing q axis")
    with ZetaLoader(path, mesh=mesh) as zl:
        assert zl.n_q_on_disk == n_q
        assert zl.q_layout == "ibz"            # 74 != prod(kgrid)=8
        with h5.File(path, "r") as f:
            ds = f["zeta_q_G"]
            for q_off, q_cnt in ((0, n_q), (3, 47), (n_q - 1, 1)):
                ref = np.asarray(ds[q_off:q_off + q_cnt, 0:n_rmu, :])
                z = zl.read_zeta_G_slab(q_offset=q_off, q_count=q_cnt,
                                        mu_offset=0, mu_count=n_rmu)
                shards = _per_rank_shards_agree_with(
                    z, ref, f"q window [{q_off}, {q_off + q_cnt})")
                host = _gather(z)
                assert host.shape == ref.shape, (host.shape, ref.shape)
                assert host.tobytes() == ref.tobytes(), (
                    f"q window [{q_off}, {q_off + q_cnt}) is not byte-equal "
                    f"to the h5py reference; max |d| = "
                    f"{np.abs(host - ref).max():.3e}")
                out["windows"].append(dict(q_offset=q_off, q_count=q_cnt,
                                           shards=shards))
    return out


# ===========================================================================
# CELL 3 — ragged ngk_per_q  (survey §7.3 rows 3 and 4)
# ===========================================================================

def check_ragged_ngk(mesh, tmpdir):
    """A genuinely ragged per-q sphere, with the anti-tautology pad check.

    The G axis is innermost and ``n_G_sph_disk`` is whatever the sphere
    gave, so ``ngk_per_q`` is ragged in every real file and ``ngkmax`` is a
    max over q.  Three claims:

    1. the fixture ACTUALLY PADS — ``min(ngk) < ngkmax``, asserted, which is
       ``test_gvec_padded_layout.py:208``'s pattern and the thing that makes
       everything below non-vacuous;
    2. the pad slots are EXACT ZEROS ON DISK — the writer's contract, which
       ``_read_g_flat_disk`` relies on ("pad slots at j >= ngk[q] are zero by
       writer construction, so the caller can ignore them") and which
       nothing checked;
    3. ``gvecs()`` returns the sentinel-padded list at whatever P this is —
       the ONLY read-time check that ``isdf_header/gvec_components`` agrees
       with the ``mf_header`` FFT grid, and a collective mesh must not change
       what it answers.
    """
    from common.gvec_fft_box import fft_box_pad_sentinel
    from zeta_loader import ZetaLoader

    grid = (8, 8, 8)
    n_q, n_rmu, ngkmax = 8, 4, 6
    ngk = np.asarray([6, 5, 3, 6, 1, 6, 4, 2], dtype=np.int32)
    world = _world(mesh)
    assert n_rmu % world == 0, f"n_rmu={n_rmu} must divide {world} devices"
    # ANTI-TAUTOLOGY, before anything else touches the file.
    assert int(ngk.min()) < ngkmax, "the fixture must actually pad"
    assert int(ngk.max()) == ngkmax, "ngkmax must be attained by some q"

    path = os.path.join(tmpdir, "zeta_ragged.h5")
    _build_on_rank0(path, lambda p: Z.build_gflat(
        p, n_q=n_q, n_rmu=n_rmu, ngkmax=ngkmax, fft_grid=grid,
        ngk_per_q=ngk))
    sent, _flat = fft_box_pad_sentinel(grid)

    out = {"mesh": "%dx%d" % _mesh_shape(mesh), "ngk": ngk.tolist(),
           "ngkmax": ngkmax}
    with ZetaLoader(path, mesh=mesh) as zl:
        assert zl.ngk_valid(q="ibz").tolist() == ngk.tolist()
        g = zl.gvecs(q="ibz")
        assert g.shape == (n_q, ngkmax, 3) and g.dtype == np.int32
        for j in range(n_q):
            n = int(ngk[j])
            assert np.array_equal(
                g[j, n:], np.broadcast_to(sent, (ngkmax - n, 3))), \
                f"q={j}: pad rows are not the sentinel at P={world}"
        # Pad slots zero ON DISK, read through the collective plan.
        z = zl.read_zeta_G_slab(q_offset=0, q_count=n_q, mu_offset=0,
                                mu_count=n_rmu)
        host = _gather(z)
        padded_q = 0
        for j in range(n_q):
            n = int(ngk[j])
            assert np.count_nonzero(host[j, :, :n]) == host[j, :, :n].size, \
                f"q={j}: a LOGICAL slot came back zero"
            if n < ngkmax:
                padded_q += 1
                assert not host[j, :, n:].any(), (
                    f"q={j}: G pad slots [{n}:{ngkmax}] are not exact zeros")
        assert padded_q >= 1, "no q was padded, so claim 2 proved nothing"
        out["padded_q"] = padded_q
    return out


# ===========================================================================
# CELL 4 — the bispinor μ mismatch, lifted to P=4  (survey §7.3 row 5)
# ===========================================================================

def check_bispinor_mu_mismatch(mesh, tmpdir):
    """FOUR loaders, two centroid counts, one mesh.

    ``gw_init.py:1211-1219`` opens ``ZetaLoader(zeta_h5_path)`` plus three
    ``ZetaLoader(zeta_T_paths[0..2])`` and hands all four straight into
    ``compute_V_q_bispinor_g_flat_to_h5``, so four SlabIO handles are open on
    one mesh at once and the charge and transverse ζ do NOT share a centroid
    count (``test_compute_V_q_bispinor_g_flat.py:129`` uses 4 vs 3 at P=1).
    Read at the COMMON padded extent, the shorter three must zero-fill and
    the longer must not — per rank.

    That four handles coexist is itself part of the claim: the loader holds
    its SlabIO open for its whole lifetime to amortise the phdf5 ctx, so
    four live loaders are four live contexts, and nothing before this ever
    opened more than one on a mesh bigger than 1x1.
    """
    from zeta_loader import ZetaLoader

    n_q, ngkmax = 2, 4
    n_rmu_C, n_rmu_T, mu_count = 4, 3, 4
    world = _world(mesh)
    assert mu_count % world == 0
    assert n_rmu_C != n_rmu_T, "the mismatch is the point"

    paths = {}
    for tag, n_rmu in (("C", n_rmu_C), ("T1", n_rmu_T), ("T2", n_rmu_T),
                       ("T3", n_rmu_T)):
        p = os.path.join(tmpdir, f"zeta_q_{tag}.h5")
        _build_on_rank0(p, lambda q, n=n_rmu: Z.build_gflat(
            q, n_q=n_q, n_rmu=n, ngkmax=ngkmax))
        paths[tag] = p

    loaders = {}
    try:
        for tag, p in paths.items():
            loaders[tag] = ZetaLoader(p, mesh=mesh)
        assert len({id(v.slab_io) for v in loaders.values()}) == 4, (
            "four loaders did not produce four distinct SlabIO handles")
        for tag, zl in loaders.items():
            n_log = n_rmu_C if tag == "C" else n_rmu_T
            assert int(zl.n_rmu) == n_log
            z = zl.read_zeta_G_slab(q_offset=0, q_count=n_q,
                                    mu_offset=0, mu_count=mu_count)
            ref = np.zeros((n_q, mu_count, ngkmax), dtype=np.complex128)
            ref[:, :n_log, :] = Z.make_payload(n_q, n_log, ngkmax)
            _per_rank_shards_agree_with(z, ref, f"bispinor {tag}")
            host = _gather(z)
            assert np.array_equal(host[:, :n_log, :],
                                  Z.make_payload(n_q, n_log, ngkmax))
            assert not host[:, n_log:, :].any(), (
                f"{tag}: rows past n_rmu={n_log} are not exact zeros")
    finally:
        for zl in loaders.values():
            zl.close()
    return {"mesh": "%dx%d" % _mesh_shape(mesh), "handles": len(paths),
            "n_rmu_C": n_rmu_C, "n_rmu_T": n_rmu_T}


# ===========================================================================
# CELL 5 — the two plans, byte-identical  (design D5's whole point)
# ===========================================================================

def check_local_vs_collective_identity(mesh, tmpdir):
    """``read_zeta_G_local`` vs a gathered ``read_zeta_G_slab``, same window.

    THE TWO-PLANS CONTRACT.  ``read_zeta_G_local``'s own docstring makes the
    claim — "the two plans are byte-identical where they overlap: both
    return the same on-disk elements and neither reduces" — and names the
    agreement that makes it true (header ngkmax == dataset G axis, checked
    once at open, which is design D5).  Nothing has ever measured the claim
    at more than one process, and it is at more than one process that it
    could fail: the collective read is a sharded transport with per-rank
    offsets, the local read is a serial h5py hyperslab, and they arrive at
    the same bytes by two entirely different routes.

    Also asserts the LOCAL plan is rank-INDEPENDENT: every rank's serial
    read of the same window must produce identical bytes.  The method is
    "per-rank independent BY CONTRACT" and that contract has never been
    exercised on more than one rank either.
    """
    import jax
    from zeta_loader import ZetaLoader

    n_q, n_rmu, ngkmax = 8, 4, 5
    world = _world(mesh)
    assert n_rmu % world == 0
    path = os.path.join(tmpdir, "zeta_two_plans.h5")
    _build_on_rank0(path, lambda p: Z.build_gflat(
        p, n_q=n_q, n_rmu=n_rmu, ngkmax=ngkmax))

    out = {"mesh": "%dx%d" % _mesh_shape(mesh), "windows": []}
    with ZetaLoader(path, mesh=mesh) as zl:
        # The D5 agreement, restated where it is relied on.
        assert int(zl.ngkmax_zeta) == int(zl.n_G_sph_disk) == ngkmax
        for q0, q1 in ((0, n_q), (2, 6), (n_q - 1, n_q)):
            local = np.asarray(zl.read_zeta_G_local(slice(q0, q1)))
            coll = _gather(zl.read_zeta_G_slab(
                q_offset=q0, q_count=q1 - q0, mu_offset=0, mu_count=n_rmu))
            assert local.shape == coll.shape, (local.shape, coll.shape)
            assert local.dtype == coll.dtype == np.complex128
            assert local.tobytes() == coll.tobytes(), (
                f"the local and collective plans disagree on q window "
                f"[{q0}, {q1}) at {_mesh_shape(mesh)}: max |d| = "
                f"{np.abs(local - coll).max():.3e}.  D5 checks that the two "
                f"size their G axis from the same number; this is the check "
                f"that they then READ the same bytes")
            # The local plan is per-rank independent: allgather a
            # fingerprint and demand every rank produced the same one.
            fp = np.frombuffer(local.tobytes()[:8], dtype=np.int64).copy()
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils
                allfp = np.asarray(multihost_utils.process_allgather(
                    jax.numpy.asarray(fp)))
                assert len(set(allfp.reshape(-1).tolist())) == 1, (
                    f"ranks disagree about the LOCAL read of q window "
                    f"[{q0}, {q1}): {allfp.reshape(-1).tolist()}")
            out["windows"].append([q0, q1])

    # …and the refusal that keeps the local plan honest after close().
    zl2 = ZetaLoader(path, mesh=mesh)
    zl2.close()
    with _raises(RuntimeError, "is closed"):
        zl2.read_zeta_G_local(0)
    return out


# ===========================================================================
# pytest entry points
# ===========================================================================

_ALL_CHECKS = (check_mu_pad_non_divisible, check_q_window_non_divisible,
               check_ragged_ngk, check_bispinor_mu_mismatch,
               check_local_vs_collective_identity)


def _pytest_mesh():
    """The largest SQUARE host mesh this process can build, 2x2 or 1x1.

    Not ``_mesh_1x1``: the emulated 4-device tier is the whole reason the
    service conftest sets ``--xla_force_host_platform_device_count=4``, and
    a pytest cell that always built 1x1 would make that flag decorative.
    Not ``require_devices(4)`` either, because the brief for these bodies is
    that they DEGENERATE GRACEFULLY at one device — every anti-tautology
    self-assertion in them still fires at 1x1, and the geometry each cell
    actually ran on is in its return value and in the report.
    """
    import jax
    from jax.sharding import Mesh
    devs = jax.devices("cpu")
    n = 4 if len(devs) >= 4 else 1
    px, py = (2, 2) if n == 4 else (1, 1)
    return Mesh(np.asarray(devs[:n]).reshape(px, py), ("x", "y"))


def _needs_transport():
    """SKIP unless a collective SlabIO read can run in this process.

    ON WSL THIS ALWAYS SKIPS, and the reason names the probe stage.  The dev
    box has no phdf5 FFI at all — ABSENT, not BROKEN — so these five cells
    are exactly the ones whose answers come from leg L-c.  The allowlist row
    in ``conftest._ALLOWED`` names that leg; the skip-honesty gate is what
    stops this from quietly becoming a suite that never runs anything.
    """
    import pytest
    ok, why = Z.host_tree_state()
    if not ok:
        pytest.skip(why)
    ok, why = Z.slab_io_state()
    if not ok:
        pytest.skip(f"{why}.  Covered by leg L-c: `lx run --cpu -N 1 -n 4 "
                    f"python3 {os.path.relpath(os.path.abspath(__file__), _REPO)} "
                    f"--mesh 2x2 --tmpdir <shared-fs>`")
    return _pytest_mesh()


def test_mu_pad_non_divisible(tmp_path):
    check_mu_pad_non_divisible(_needs_transport(), str(tmp_path))


def test_q_window_non_divisible(tmp_path):
    check_q_window_non_divisible(_needs_transport(), str(tmp_path))


def test_ragged_ngk(tmp_path):
    check_ragged_ngk(_needs_transport(), str(tmp_path))


def test_bispinor_mu_mismatch(tmp_path):
    check_bispinor_mu_mismatch(_needs_transport(), str(tmp_path))


def test_local_vs_collective_identity(tmp_path):
    check_local_vs_collective_identity(_needs_transport(), str(tmp_path))


def test_the_per_rank_comparator_can_fail():
    """RED TWIN for :func:`_per_rank_shards_agree_with`, which needs no
    transport and carries every per-rank claim in this file.

    All five check bodies delegate their "each RANK's own tile is right"
    assertion here, so a comparator that could not fail would make five
    cells report PASS from a walk that compared nothing.  Three ways it must
    fail, because three things can be wrong: a byte differs somewhere in the
    sharded interior, a shape differs, and — the one a naive implementation
    misses — the process addressed NO shards at all, which is what a
    gathered-only check silently tolerates.

    Runs on WSL: ``jax.device_put`` under a NamedSharding needs no phdf5.
    """
    import jax
    import pytest
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = _pytest_mesh()
    world = _world(mesh)
    ref = np.arange(2 * 4 * world * 3, dtype=np.float64).reshape(
        2, 4 * world, 3).astype(np.complex128)
    x = jax.device_put(ref, NamedSharding(mesh, P(None, ("x", "y"), None)))

    # POSITIVE CONTROL first: the comparator accepts the truth, and reports
    # one shard per addressable device.
    assert _per_rank_shards_agree_with(x, ref, "control") == world

    # 1. one byte wrong, in the LAST rank's tile — the tile a per-rank bug
    #    lands in and a first-shard-only check would miss.
    bad = ref.copy()
    bad[1, -1, 2] += 1.0
    with pytest.raises(AssertionError, match="differs from the h5py"):
        _per_rank_shards_agree_with(x, bad, "byte")

    # 2. a shape mismatch is reported as a shape mismatch, not as a byte
    #    difference — different defect, different fix.
    with pytest.raises(AssertionError, match="shape"):
        _per_rank_shards_agree_with(x, ref[:, :, :2], "shape")

    # 3. an object with no addressable shards must FAIL, not pass vacuously.
    class _NoShards:
        addressable_shards = ()

    with pytest.raises(AssertionError, match="addressed no shards"):
        _per_rank_shards_agree_with(_NoShards(), ref, "empty")


def test_the_emulated_tier_really_gets_four_devices():
    """L-b's own gate: when four host devices exist, the mesh IS 2x2.

    ``--xla_force_host_platform_device_count`` is read at the FIRST jax
    import in a process and never again, so the emulated tier is not "on or
    off" — it is "did the service conftest load before anything imported
    jax".  In the full-suite run it did not, the flag is inert, and this
    cell skips.  Under ``pytest services/zeta_loader/tests`` it did, and
    this asserts the flag actually took rather than silently leaving every
    cell above at 1x1.
    """
    from lxkit.testing import require_devices
    require_devices(4, "cpu")
    assert _mesh_shape(_pytest_mesh()) == (2, 2)


def test_the_cli_cells_are_all_reachable():
    """Every ``_CLI_CELLS`` row names a function that exists and every check
    body is in the table.

    Cheap, and it is the failure the CLI mode cannot report: a typo'd or
    dropped row makes the multi-rank leg quietly run a smaller matrix and
    print ``done: 0 failures``.
    """
    names = {name for name, _ in _CLI_CELLS}
    assert len(names) == len(_CLI_CELLS), "duplicate _CLI_CELLS name"
    # Read the GLOBALS each row's lambda actually references, not its label.
    # Matching on the label is the version of this check that silently rots:
    # rename a body and the substring stops matching while both sides still
    # exist.
    called = set()
    for _name, fn in _CLI_CELLS:
        mine = {g for g in fn.__code__.co_names if g.startswith("check_")}
        assert mine, f"{_name}: calls no check body"
        called |= mine
    bodies = {k for k in globals() if k.startswith("check_")}
    missing = bodies - called
    assert not missing, (
        f"check bodies with no _CLI_CELLS row (the 2x2 leg would never run "
        f"them): {sorted(missing)}")
    unknown = called - bodies
    assert not unknown, f"_CLI_CELLS names bodies that do not exist: {unknown}"
    # …and every body also has a pytest cell, so the two callers stay paired.
    cells = {k for k in globals() if k.startswith("test_")}
    for body in bodies:
        assert body.replace("check_", "test_") in cells, (
            f"{body} has no pytest cell; the L-b tier would never run it")


# ===========================================================================
# CLI mode — the real multi-rank matrix
# ===========================================================================

_CLI_CELLS = [
    # (name, fn(mesh, tmpdir))
    ("mu_pad_non_divisible",
     lambda mesh, td: check_mu_pad_non_divisible(mesh, td)),
    ("q_window_non_divisible",
     lambda mesh, td: check_q_window_non_divisible(mesh, td)),
    ("ragged_ngk", lambda mesh, td: check_ragged_ngk(mesh, td)),
    ("bispinor_mu_mismatch",
     lambda mesh, td: check_bispinor_mu_mismatch(mesh, td)),
    ("local_vs_collective_identity",
     lambda mesh, td: check_local_vs_collective_identity(mesh, td)),
]


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PxQ device mesh")
    ap.add_argument("--tmpdir", required=True,
                    help="a SHARED-FILESYSTEM directory every rank can see "
                         "($SCRATCH, not /tmp): rank 0 builds the ζ files "
                         "there and every rank reads them")
    ap.add_argument("--only", default="", help="substring filter")
    ap.add_argument("--report", default="",
                    help="write a JSON artifact here (rank 0 only)")
    args = ap.parse_args()

    mesh = _mesh_from_arg(args.mesh)
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    if jax.process_index() == 0:
        os.makedirs(args.tmpdir, exist_ok=True)
    _barrier("tmpdir_ready")
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()} "
       f"tmpdir={args.tmpdir}", flush=True)

    rows, failures, ran = [], 0, 0
    for name, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        tag = f"{name}[{args.mesh}]"
        t0 = time.perf_counter()
        try:
            out = fn(mesh, args.tmpdir)
            ran += 1
            dt = time.perf_counter() - t0
            rows.append(dict(cell=name, status="PASS", seconds=dt, detail=out))
            p0(f"PASS {tag} {dt:.2f}s {out}", flush=True)
        except AssertionError as exc:
            failures += 1
            dt = time.perf_counter() - t0
            msg = " ".join(str(exc).split())[:600]
            rows.append(dict(cell=name, status="FAIL", seconds=dt, error=msg))
            p0(f"FAIL {tag}: {msg}", flush=True)
        except Exception as exc:                               # noqa: BLE001
            failures += 1
            dt = time.perf_counter() - t0
            msg = f"{type(exc).__name__}: {' '.join(str(exc).split())[:600]}"
            rows.append(dict(cell=name, status="ERROR", seconds=dt, error=msg))
            p0(f"ERROR {tag}: {msg}", flush=True)

    # RAN, not just failures.  "0 failures" out of 0 cells is the shape of
    # every artifact-free green in this tree's history — and the machine-
    # readable line is what the orchestrator greps for.
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    p0(f"SUMMARY zeta_loader_multiproc mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()} "
       f"ran={ran} failures={failures} "
       f"jobid={os.environ.get('SLURM_JOB_ID', '')}", flush=True)

    if args.report and jax.process_index() == 0:
        doc = dict(
            suite="zeta_loader_multiproc", mesh=args.mesh,
            processes=jax.process_count(), devices=jax.device_count(),
            backend=jax.default_backend(),
            jobid=os.environ.get("SLURM_JOB_ID",
                                 os.environ.get("SLURM_JOBID", "")),
            nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
            machine=os.environ.get("NERSC_HOST",
                                   os.environ.get("LX_MACHINE", "")),
            ffi_so=os.environ.get("LORRAX_FFI_SO", ""),
            ffi_host_so=os.environ.get("LORRAX_FFI_HOST_SO", ""),
            jax_version=jax.__version__,
            recorded=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            ran=ran, failures=failures, rows=rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.report)),
                    exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        print(f"wrote {args.report}: {len(rows)} rows "
              f"({os.path.getsize(args.report)} bytes)", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    sys.exit(_cli_main())
