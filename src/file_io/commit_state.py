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


def set_commit_state(h5, committed):
    """Write the small receipt through an already-open serial HDF5 handle."""
    if COMMIT_STATE not in h5:
        h5.create_dataset(COMMIT_STATE, shape=(1,), dtype='int32')
    h5[COMMIT_STATE][0] = int(committed)
    h5.flush()
