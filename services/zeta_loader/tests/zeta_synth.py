"""Synthetic ``zeta_q.h5`` builders for the zeta_loader suite.

RAW h5py AND numpy, DELIBERATELY.  Every other ζ fixture in this tree
(``tests/test_file_io.py``'s ``_build_zeta_h5_gflat`` /
``_build_zeta_h5_rspace``, ``test_compute_all_V_q_g_flat``'s
``_build_g_flat_zeta_file``) builds its file through
``file_io.mf_header.copy_mf_header`` + ``file_io.isdf_header.IsdfHeader.build``
+ ``write_isdf_header``.  Those are the right tools in the monorepo suite and
the WRONG ones here, for three measured reasons:

1. **The corrupt-file fixtures could not exist.**  ``IsdfHeader.build``
   VALIDATES (``ngk_per_q`` shape against ``gvec_components``,
   ``max(ngk) <= ngkmax``, the G-flat required-fields rule), so a file whose
   header ``ngkmax`` disagrees with its ``zeta_q_G`` G axis — the D5 refusal's
   red twin, the whole point of that check — cannot be produced through the
   writer at all.  Writing the group by hand is what lets a test build the
   state the reader is supposed to refuse.

2. **The import-isolation leg needs a real file with lorrax OFF sys.path.**
   ``test_zeta_loader_import_isolation`` asserts that constructing a
   ``ZetaLoader`` over a REAL minimal HDF5 refuses BY NAMING the missing
   host-tree module.  A builder that imports ``file_io`` cannot run in that
   subprocess; this one is copied into it as source text.

3. **The fixture stops being a restatement of the writer.**  A round-trip
   through ``write_isdf_header`` then ``read_isdf_header`` proves the two
   agree with each other, not that either matches the on-disk contract.  The
   group layout here was transcribed from ``isdf_header.write_isdf_header``
   and ``mf_header._read_group``, and
   ``test_zeta_loader_contract.py::test_the_synthetic_header_matches_the_real_writers``
   pins the transcription against BOTH real readers so a drift in either
   direction turns red here rather than silently making every cell in this
   suite fixture-shaped.

NOTHING IN THIS MODULE IMPORTS jax OR THE HOST TREE AT MODULE SCOPE.  The
multiproc file imports it before ``jax.distributed.initialize``, and the AST
tier imports nothing else at all.  The two capability probes at the bottom
import inside their bodies and RETURN a reason rather than raising, which is
what lets a caller turn them into an honest skip.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pad_sentinel", "payload_value", "make_payload", "make_gvec_components",
    "write_mf_header", "write_isdf_header_raw",
    "build_gflat", "build_rspace", "build_isdf_header_only", "MF_DEFAULTS",
    "host_tree_state", "slab_io_state",
]


# ---------------------------------------------------------------------------
# The pad sentinel, transcribed
# ---------------------------------------------------------------------------

def pad_sentinel(fft_grid) -> tuple[int, int, int]:
    """``common.gvec_fft_box.fft_box_pad_sentinel``'s components, locally.

    Read off the ``fftfreq`` table rather than written as ``-nx//2``, which
    is the distinction that module's own docstring makes: the two agree for
    EVEN extents and differ for odd ones, and BGW FFT grids are even.  A
    local copy of a shared truth is a drift hazard, so
    ``test_the_local_pad_sentinel_agrees_with_the_shared_one`` asserts this
    function against the real one whenever the host tree is importable.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    return tuple(
        int(np.rint(np.fft.fftfreq(n)[n // 2] * n)) for n in (nx, ny, nz))


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def payload_value(q: int, m: int, g: int) -> complex:
    """A distinct NON-ZERO value per ``(q, μ, G)`` cell.

    Non-zero everywhere is load-bearing, not decoration: every μ-pad and
    ragged-``ngk`` cell in this suite asserts "payload here, EXACT zeros
    there", and a payload that could itself be zero would make the zero half
    of that pass for the wrong reason.  Distinct per cell is what turns
    "the right number of elements came back" into "the right ELEMENTS came
    back" — an off-by-one in the μ offset moves a value rather than
    preserving a pattern.
    """
    n = q * 1_000_000 + m * 1_000 + g + 1
    return complex(float(n), -0.5 * float(n))


def make_payload(n_q: int, n_rmu: int, ngkmax: int,
                 ngk_per_q=None) -> np.ndarray:
    """``(n_q, n_rmu, ngkmax)`` complex128; pad slots past ``ngk[q]`` zero.

    Zeroing ``[q, :, ngk[q]:]`` is the WRITER's contract
    (``ZetaLoader._read_g_flat_disk``: "pad slots at ``j >= ngk[q]`` are
    zero by writer construction"), reproduced here so the ragged-sphere
    cells can assert it holds on disk instead of assuming it.
    """
    out = np.zeros((n_q, n_rmu, ngkmax), dtype=np.complex128)
    for q in range(n_q):
        nk = ngkmax if ngk_per_q is None else int(ngk_per_q[q])
        for m in range(n_rmu):
            for g in range(nk):
                out[q, m, g] = payload_value(q, m, g)
    return out


# ---------------------------------------------------------------------------
# The per-q G-list
# ---------------------------------------------------------------------------

def make_gvec_components(n_q: int, ngkmax: int, ngk_per_q,
                         fft_grid) -> np.ndarray:
    """``(n_q, 3, ngkmax)`` int32 — physical rows then sentinel pad.

    The on-disk order is ``(n_q, 3, ngkmax)`` (WFN.h5's ``(3, ng)``
    component order with a leading q axis), which is what ``ZetaLoader.gvecs``
    transposes.  Physical rows are drawn from the small-|G| shell so that
    NO physical G lands on the FFT box's Nyquist corner — ``pad_gvecs_to_
    sentinel`` refuses that collision, and a fixture that tripped it would
    fail the reader for a reason that has nothing to do with the reader.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    sent = pad_sentinel(fft_grid)
    # Enumerate a small box around the origin, in a stable order, skipping
    # anything that wraps onto the sentinel CELL (the refusal is per-cell,
    # not per-Miller-triple).
    lim = 1
    shell = []
    while len(shell) < int(ngkmax):
        lim += 1
        shell = []
        for a in range(-lim, lim + 1):
            for b in range(-lim, lim + 1):
                for c in range(-lim, lim + 1):
                    if (a % nx, b % ny, c % nz) == (sent[0] % nx,
                                                    sent[1] % ny,
                                                    sent[2] % nz):
                        continue
                    shell.append((a * a + b * b + c * c, a, b, c))
        shell.sort()
        shell = [(a, b, c) for _r, a, b, c in shell]
    out = np.empty((int(n_q), 3, int(ngkmax)), dtype=np.int32)
    out[...] = np.asarray(sent, dtype=np.int32)[None, :, None]
    for q in range(int(n_q)):
        n = int(ngk_per_q[q])
        rows = np.asarray(shell[:n], dtype=np.int32)          # (n, 3)
        out[q, :, :n] = rows.T
    return out


# ---------------------------------------------------------------------------
# mf_header — transcribed from file_io.mf_header._read_group
# ---------------------------------------------------------------------------

#: The scalar header values every builder here writes unless overridden.
#: Small and arbitrary EXCEPT ``fft_grid`` and ``kgrid``, which the reader
#: derives ``n_rtot`` and ``q_layout`` from and which cells therefore set.
MF_DEFAULTS = dict(fft_grid=(8, 8, 8), kgrid=(2, 2, 2), nspin=1, nspinor=1,
                   nkpts=3, nbands=4, ntran=2, nat=2)


def write_mf_header(path, *, fft_grid=(8, 8, 8), kgrid=(2, 2, 2), nspin=1,
                    nspinor=1, nkpts=3, nbands=4, ntran=2, nat=2,
                    mode="w") -> None:
    """Write a minimal, self-consistent ``mf_header`` group with raw h5py.

    Every dataset ``file_io.mf_header._read_group`` reads, and no other:
    a fixture that carried extra groups would let a reader regression that
    started depending on one of them pass here and fail in production.
    """
    import h5py as h5

    rng = np.random.default_rng(0xCAFE)
    nkx, nky, nkz = (int(v) for v in kgrid)
    with h5.File(str(path), mode) as f:
        g = f.create_group("mf_header")
        g.create_dataset("versionnumber", data=np.int32(3))
        g.create_dataset("flavor", data=np.int32(2))

        kp = g.create_group("kpoints")
        kp.create_dataset("nspin", data=np.int32(nspin))
        kp.create_dataset("nspinor", data=np.int32(nspinor))
        kp.create_dataset("nrk", data=np.int32(nkpts))
        kp.create_dataset("mnband", data=np.int32(nbands))
        kp.create_dataset("ngkmax", data=np.int32(17))
        kp.create_dataset("ecutwfc", data=np.float64(35.0))
        kp.create_dataset("kgrid", data=np.asarray([nkx, nky, nkz],
                                                   dtype=np.int32))
        kp.create_dataset("shift", data=np.zeros(3, dtype=np.float64))
        kp.create_dataset("ngk", data=np.full((nkpts,), 17, dtype=np.int32))
        kp.create_dataset("ifmin", data=np.ones((nspin, nkpts), dtype=np.int32))
        kp.create_dataset("ifmax", data=np.full((nspin, nkpts), 2,
                                                dtype=np.int32))
        kp.create_dataset("w", data=np.full((nkpts,), 1.0 / nkpts,
                                            dtype=np.float64))
        kp.create_dataset("rk", data=rng.random((nkpts, 3)))
        kp.create_dataset("el", data=rng.random((nspin, nkpts, nbands)))
        kp.create_dataset("occ", data=np.zeros((nspin, nkpts, nbands),
                                               dtype=np.float64))

        gs = g.create_group("gspace")
        gs.create_dataset("ng", data=np.int32(100))
        gs.create_dataset("ecutrho", data=np.float64(140.0))
        gs.create_dataset("FFTgrid",
                          data=np.asarray(fft_grid, dtype=np.int32))

        sym = g.create_group("symmetry")
        sym.create_dataset("ntran", data=np.int32(ntran))
        sym.create_dataset("cell_symmetry", data=np.int32(0))
        sym.create_dataset("mtrx", data=np.broadcast_to(
            np.eye(3, dtype=np.int32), (48, 3, 3)).copy())
        sym.create_dataset("tnp", data=np.zeros((48, 3), dtype=np.float64))

        cr = g.create_group("crystal")
        cr.create_dataset("celvol", data=np.float64(123.4))
        cr.create_dataset("recvol", data=np.float64(0.5))
        cr.create_dataset("alat", data=np.float64(7.6))
        cr.create_dataset("blat", data=np.float64(0.825))
        cr.create_dataset("nat", data=np.int32(nat))
        cr.create_dataset("avec", data=np.eye(3, dtype=np.float64) * 7.6)
        cr.create_dataset("bvec", data=np.eye(3, dtype=np.float64) * 0.825)
        cr.create_dataset("adot", data=np.eye(3, dtype=np.float64) * 57.76)
        cr.create_dataset("bdot", data=np.eye(3, dtype=np.float64) * 0.68)
        cr.create_dataset("atyp", data=np.full((nat,), 14, dtype=np.int32))
        cr.create_dataset("apos", data=rng.random((nat, 3)))


# ---------------------------------------------------------------------------
# isdf_header — transcribed from file_io.isdf_header.write_isdf_header
# ---------------------------------------------------------------------------

def write_isdf_header_raw(path, *, n_rmu, fft_grid, density="scalar",
                          vertex_mu_L=0, zeta_is_done=True,
                          zeta_layout="G_flat", gvec_components=None,
                          ngk_per_q=None, zeta_cutoff_ry=None,
                          fit_provenance=None, r_mu_fft_idx=None,
                          omit_zeta_is_done=False) -> None:
    """Write an ``isdf_header`` group with raw h5py, validating NOTHING.

    ``omit_zeta_is_done`` reproduces a LEGACY file: ``_read_group`` treats a
    missing flag as ``True`` ("they were always written atomically at
    end-of-fit"), and the probe reports ``None`` — which is NOT ``False``,
    a distinction ``ZetaFileProbe``'s own docstring makes and which this
    suite's truth table pins.
    """
    import h5py as h5

    grid = np.asarray(fft_grid, dtype=np.int32)
    if r_mu_fft_idx is None:
        r_mu_fft_idx = (np.arange(3 * int(n_rmu), dtype=np.int32)
                        .reshape(int(n_rmu), 3) % grid[None, :])
    idx = np.asarray(r_mu_fft_idx, dtype=np.int32)
    crystal = idx.astype(np.float64) / grid.astype(np.float64)[None, :]

    with h5.File(str(path), "a") as f:
        g = f.create_group("isdf_header")
        g.create_dataset("density", data=np.bytes_(density))
        g.create_dataset("vertex_mu_L", data=np.int32(vertex_mu_L))
        if not omit_zeta_is_done:
            g.create_dataset("zeta_is_done", data=np.bool_(zeta_is_done))
        g.create_dataset("zeta_layout", data=np.bytes_(zeta_layout))
        c = g.create_group("centroids")
        c.create_dataset("r_mu_fft_idx", data=idx)
        c.create_dataset("r_mu_crystal", data=crystal)
        if gvec_components is not None:
            g.create_dataset("gvec_components",
                             data=np.asarray(gvec_components, dtype=np.int32))
        if ngk_per_q is not None:
            g.create_dataset("ngk", data=np.asarray(ngk_per_q, dtype=np.int32))
        if zeta_cutoff_ry is not None:
            g.create_dataset("zeta_cutoff_ry",
                             data=np.float64(zeta_cutoff_ry))
        if fit_provenance is not None:
            g.create_dataset("fit_provenance",
                             data=np.bytes_(str(fit_provenance)))


# ---------------------------------------------------------------------------
# The builders the cells call
# ---------------------------------------------------------------------------

def build_gflat(path, *, n_q=2, n_rmu=3, ngkmax=4, fft_grid=(8, 8, 8),
                kgrid=(2, 2, 2), ngk_per_q=None, zeta_is_done=True,
                header_n_rmu=None, header_ngkmax=None,
                omit_zeta_is_done=False, omit_zeta_dataset=False,
                fit_provenance=None, dataset_n_rmu=None):
    """A G-flat ζ file.  Returns ``(path, payload)``.

    The three deliberately-corrupting knobs each build a state the reader
    is supposed to REFUSE, and each is the red twin of one ``__init__``
    check:

    ``header_n_rmu``
        the centroid table lists more μ than ``zeta_q_G`` has rows —
        the header-vs-dataset μ agreement.
    ``header_ngkmax``
        the components table's ngkmax disagrees with the dataset's G axis —
        the D5 agreement, the one that keeps the collective plan (which
        sizes from the header) and the local plan (which sizes from the
        dataset) reading the SAME extent.
    ``omit_zeta_dataset``
        header written, first chunk never arrived — a real state, and the
        one ``probe_zeta_file`` reports as ``dataset_name=None``.
    """
    import h5py as h5

    path = str(path)
    n_rmu_disk = int(n_rmu if dataset_n_rmu is None else dataset_n_rmu)
    if ngk_per_q is None:
        ngk_per_q = np.full((int(n_q),), int(ngkmax), dtype=np.int32)
    ngk_per_q = np.asarray(ngk_per_q, dtype=np.int32)
    hdr_ngkmax = int(ngkmax if header_ngkmax is None else header_ngkmax)
    comps = make_gvec_components(int(n_q), hdr_ngkmax,
                                 np.minimum(ngk_per_q, hdr_ngkmax), fft_grid)

    write_mf_header(path, fft_grid=fft_grid, kgrid=kgrid, mode="w")
    write_isdf_header_raw(
        path, n_rmu=int(n_rmu if header_n_rmu is None else header_n_rmu),
        fft_grid=fft_grid, zeta_layout="G_flat", zeta_is_done=zeta_is_done,
        gvec_components=comps, ngk_per_q=ngk_per_q, zeta_cutoff_ry=10.0,
        fit_provenance=fit_provenance, omit_zeta_is_done=omit_zeta_is_done)

    payload = make_payload(int(n_q), n_rmu_disk, int(ngkmax), ngk_per_q)
    if not omit_zeta_dataset:
        with h5.File(path, "a") as f:
            f.create_dataset("zeta_q_G", data=payload)
    return path, payload


def build_rspace(path, *, n_q=2, n_rtot=8, n_rmu=4, fft_grid=(8, 8, 8),
                 kgrid=(2, 2, 2), zeta_is_done=True, header_n_rmu=None):
    """A LEGACY r-space ζ file.  Returns ``(path, payload)``.

    Nothing in the tree has written one since the G-flat migration
    (``fit_zeta_to_h5`` hardcodes ``zeta_layout='G_flat'``), and no data
    method reads one since 2026-08-07 — which is exactly why the suite
    still builds them: ``__init__`` must keep OPENING them for the header
    surface, and every data method must refuse by naming the removal.
    """
    import h5py as h5

    path = str(path)
    write_mf_header(path, fft_grid=fft_grid, kgrid=kgrid, mode="w")
    write_isdf_header_raw(
        path, n_rmu=int(n_rmu if header_n_rmu is None else header_n_rmu),
        fft_grid=fft_grid, zeta_layout="r_space", zeta_is_done=zeta_is_done)
    payload = (np.arange(int(n_q) * int(n_rtot) * int(n_rmu))
               .reshape(int(n_q), int(n_rtot), int(n_rmu))
               .astype(np.complex128) + 1.0)
    with h5.File(path, "a") as f:
        f.create_dataset("zeta_q", data=payload)
    return path, payload


def build_isdf_header_only(path, *, n_rmu=3, fft_grid=(8, 8, 8),
                           kgrid=(2, 2, 2), zeta_is_done=False):
    """Headers written, ζ block absent.  The killed-mid-fit state."""
    write_mf_header(path, fft_grid=fft_grid, kgrid=kgrid, mode="w")
    write_isdf_header_raw(path, n_rmu=n_rmu, fft_grid=fft_grid,
                          zeta_layout="G_flat", zeta_is_done=zeta_is_done,
                          gvec_components=None, ngk_per_q=None)
    return str(path)


# ---------------------------------------------------------------------------
# Capability probes.  RETURN a reason; never raise, never skip.
# ---------------------------------------------------------------------------
# Deciding to skip is the CALLER's, and keeping that decision out of here is
# what lets the same two functions serve the pytest cells (which skip) and
# the multi-rank CLI mode (which has no pytest and must report instead).

def host_tree_state() -> tuple[bool, str]:
    """``(ok, reason)`` — is the LORRAX host tree importable in this process?

    The DATA path of this service reaches ``file_io.mf_header``,
    ``file_io.isdf_header``, ``file_io.slab_io`` and ``common.gvec_fft_box``
    through call-time imports (the wave-1b seam).  A cell that opens a
    ``ZetaLoader`` needs the first two even in header-only mode; the FORMAT
    surface needs none of them, which is why those cells carry no probe.
    """
    try:
        import file_io.isdf_header            # noqa: F401
        import file_io.mf_header              # noqa: F401
    except Exception as exc:                                   # noqa: BLE001
        return False, (
            f"no lorrax host tree on sys.path in this process "
            f"({type(exc).__name__}: {exc}); ZetaLoader's header binders "
            f"live there until wave 1b extracts them")
    return True, "file_io.{mf,isdf}_header importable"


def slab_io_state() -> tuple[bool, str]:
    """``(ok, reason)`` — can a collective SlabIO read run here?

    ON WSL THE ANSWER IS ALWAYS NO, and it is ABSENT rather than BROKEN:
    the dev box has no phdf5 FFI library at all, so ``probe_availability``
    reports the missing handler and every ``read_zeta_G_slab`` / ``load``
    refuses at open.  A cell that turned that into a bare failure would
    make the WSL leg permanently red for a platform fact; a cell that
    turned it into a silent pass would be the 19-skipped-0-failed shape
    lxkit.testing exists to prevent.  The honest report is a skip whose
    reason NAMES THE PROBE STAGE, which is what this returns.
    """
    ok, why = host_tree_state()
    if not ok:
        return False, f"SlabIO transport unavailable: {why}"
    try:
        from file_io.slab_io import probe_availability
    except Exception as exc:                                   # noqa: BLE001
        return False, (f"SlabIO transport unavailable: file_io.slab_io is "
                       f"not importable ({type(exc).__name__}: {exc})")
    try:
        ok, stage, reason = probe_availability()
    except Exception as exc:                                   # noqa: BLE001
        return False, (f"SlabIO transport unavailable: probe_availability "
                       f"raised {type(exc).__name__}: {exc}")
    return bool(ok), (f"SlabIO transport {'available' if ok else 'unavailable'}"
                      f": probe stage {stage!r}: {reason}")
