"""Persistent completion receipt for collective artifact mutations (L3).

SlabIO explicitly writes zero before tensor writes: the native provider uses
H5D_FILL_TIME_NEVER, so creation alone cannot initialize the receipt. Legacy
artifacts without this dataset retain
their format-specific checks. This receipt authenticates completion, not physics.
"""
COMMIT_STATE = 'lorrax_io_committed'


def assert_committed(h5, *, path=None):
    """Refuse a readable HDF5 file left inside a collective write transaction."""
    if COMMIT_STATE in h5 and int(h5[COMMIT_STATE][0]) != 1:
        raise ValueError(
            f"GATE io_global_commit: path={path or h5.filename}; "
            "stage=restart read; artifact is not globally committed. "
            "Do not reuse this incomplete artifact; rebuild in a new run directory.")


#: How long a reopen waits out a transiently short size view (``open_h5_settled``).
SETTLE_TIMEOUT_S = 60.0


def open_h5_settled(path, mode="r", **kwargs):
    """``h5py.File(path, mode)`` for a file another process just committed.

    HDF5 refuses an open when the size this client reports is below the end of
    allocation stored in the superblock (``truncated file: eof < stored_eof``),
    before it touches the file. On Frontier Lustre at P=16 over two nodes,
    many ranks reopening a just-closed file at once see that size transiently
    short, at an exact stripe (1 MiB) boundary, on the writer's own node too,
    and NOT monotonically: one open succeeds and the next, milliseconds later,
    is refused (a ``stat`` threshold cannot fence it; ``fsync`` does not
    either). The data is complete: the same file reports its full size again
    at once. So this refusal, and only this one, is retried (each refusal
    logged) with bounded
    backoff; a file that stays short for ``SETTLE_TIMEOUT_S`` is truly
    truncated and an ``OSError`` naming that cause (HDF5's message, the wait
    and the open count) is raised. Every other error is raised at once.

    The superblock's stored end of file is the writer's own size receipt,
    laid down at its last close; HDF5 checks it on every open, so a short
    view never reaches a reader as a file.  Perlmutter scratch shows the same
    view (2026-11-20/22, lane nips3: five runs lost; every refused eof an exact
    1 MiB multiple, a ``zeta_q.h5`` read by ``zeta_loader`` minutes after its
    collective write, on the writer's node and on others).  So THIS is the one
    route for every serial HDF5 open of an existing file in ``src/`` and
    ``services/*/src`` (ζ, the ISDF tensor file's V, W and bank members,
    model and pole files, QP rotations and QP WFN, dipole, kinetic/ionic,
    parallel transport, Σ and BSE outputs, restart and checkpoint files;
    ``tests/test_settled_opens_cpu.py`` holds the line).  Not covered: a
    file created with ``'w'`` (nothing to read back) and the collective
    MPI-IO opens of ``file_io.slab_io``. Prefer one reader
    (``common.collectives.rank0_read``) where a value is replicated.
    """
    import sys
    import time

    import h5py

    t0, nap, tries = time.monotonic(), 0.01, 0
    while True:
        try:
            handle = h5py.File(path, mode, **kwargs)
        except OSError as exc:
            if "truncated file" not in str(exc):
                raise
            if time.monotonic() - t0 > SETTLE_TIMEOUT_S:
                raise OSError(
                    f"{exc} [open_h5_settled: {path} still reads shorter than the end "
                    f"of file its own superblock records after {tries + 1} opens over "
                    f"{time.monotonic() - t0:.1f} s; the file is truncated (an "
                    f"interrupted writer) or this node's size view of it is stale; "
                    f"do not reuse it: rerun the stage that writes it]") from exc
            tries += 1
            print(f"  [open_h5_settled] {path}: refused open {tries} ({exc}); "
                  f"retrying in {nap:.3f} s", file=sys.stderr, flush=True)
            time.sleep(nap)
            nap = min(2 * nap, 0.2)
            continue
        if tries:
            print(f"  [open_h5_settled] {path}: size view settled after {tries} "
                  f"refused open(s), {time.monotonic() - t0:.3f} s", file=sys.stderr, flush=True)
        return handle


def agree_io_refusal(error, *, path, stage):
    """Agree an artifact refusal across ranks without losing its reason.

    ``common.collectives.agree_io_error`` is THE agreement: every rank raises
    when any rank failed, so a rank-local metadata failure can never leave its
    peers alone in a collective. Its gathered receipt is a fixed record that
    keeps only the first 240 characters of the message, and this tree's paths
    routinely spend more than that, so the reason a reader actually needs
    ("artifact is not globally committed") can fall off the end. A rank that
    failed locally therefore raises the agreed receipt WITH its own untruncated
    diagnostic; ranks that only heard about it raise the receipt alone.
    """
    from common.collectives import agree_io_error

    try:
        agree_io_error(error, path=path, stage=stage)
    except RuntimeError as exc:
        if error is None:
            raise ValueError(str(exc)) from exc
        raise ValueError(f"{exc} Local refusal: {error}") from error


def read_commit_state(path):
    """The receipt of the file at ``path``: 1, 0, or ``None`` if it carries none."""
    with open_h5_settled(str(path), "r") as h5:
        if COMMIT_STATE not in h5:
            return None
        return int(h5[COMMIT_STATE][0])


def set_commit_state(h5, committed):
    """Write the small receipt through an already-open serial HDF5 handle."""
    if COMMIT_STATE not in h5:
        h5.create_dataset(COMMIT_STATE, shape=(1,), dtype='int32')
    h5[COMMIT_STATE][0] = int(committed)
    h5.flush()
