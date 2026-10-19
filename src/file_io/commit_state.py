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
    truncated and the original error is raised. Every other error is raised
    at once. Close-time metadata opens only; no hot path. Prefer one reader
    (``common.collectives.rank0_read``): this is for the opens that remain,
    rank 0's reopen of what other clients wrote and per-rank data reads.
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
                raise
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
    import h5py

    with h5py.File(str(path), "r") as h5:
        if COMMIT_STATE not in h5:
            return None
        return int(h5[COMMIT_STATE][0])


def set_commit_state(h5, committed):
    """Write the small receipt through an already-open serial HDF5 handle."""
    if COMMIT_STATE not in h5:
        h5.create_dataset(COMMIT_STATE, shape=(1,), dtype='int32')
    h5[COMMIT_STATE][0] = int(committed)
    h5.flush()
