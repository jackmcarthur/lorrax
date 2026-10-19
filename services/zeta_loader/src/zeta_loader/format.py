"""The ``zeta_q.h5`` FORMAT surface: what a file is, without opening a run.

One function and a record, all pure h5py + numpy.  Nothing here imports
jax, ``file_io`` or anything else from the LORRAX tree, so this half of
the service works with lorrax off ``sys.path`` entirely — which is not a
purity exercise: its callers run BEFORE a
:class:`~zeta_loader.ZetaLoader` could exist (they gate whether the ζ fit
runs at all).

:func:`probe_zeta_file`
    Never raises.  Answers "what is on disk here?" for a path that may
    not exist, may be a directory, may be a zero-byte file left by a
    crashed job, or may be someone else's HDF5.
THE LAYOUT DISPATCH LIVES HERE, ONCE.  ``(('zeta_q_G', 1), ('zeta_q', 2))``
— the dataset name and the axis μ sits on — was written out by hand at
two sites in ``gw/gw_init.py`` (``_check_zeta_h5_matches_basis`` and
``_zeta_reuse_ok``'s extent probe) as well as inside ``ZetaLoader``.
Three copies of one truth, and the cost of that is measured rather than
hypothetical: the gw_init copies probed ``f['zeta_q']`` ONLY for months
after the G-flat migration, so the guard silently passed on exactly the
production files it was written to protect.  Order matters and is not
alphabetical: ``zeta_q_G`` is production and is checked first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import numpy as np

__all__ = ["ZetaFileProbe", "probe_zeta_file"]


#: ``(dataset name, the axis μ sits on)``, production first.  THE copy.
_ZETA_DATASETS: tuple[tuple[str, int], ...] = (("zeta_q_G", 1), ("zeta_q", 2))


@dataclass(frozen=True, eq=False)
class ZetaFileProbe:
    """What :func:`probe_zeta_file` found.  Every field may be absent.

    ``eq=False`` because :attr:`r_mu_fft_idx` is an array and a generated
    ``__eq__`` would compare it elementwise and then raise on the truth
    value — a record whose equality operator raises is worse than one
    that has none.

    Attributes
    ----------
    exists
        ``os.path.exists(path)``.  Everything else is ``None``/``False``
        when this is False.
    readable
        The file opened as HDF5 and every field below was read without
        an exception.  ``exists and not readable`` is the crashed-job /
        zero-byte / not-actually-HDF5 case, and :attr:`error` says which.
    error
        ``"<ExceptionType>: <message>"``, or ``None``.  Present exactly
        when a read was attempted and failed; callers that must
        print-and-continue print this.
    dataset_name
        ``'zeta_q_G'`` (G-flat, production), ``'zeta_q'`` (legacy
        r-space) or ``None`` — a file with an ``isdf_header`` but NO ζ
        dataset at all is a real state (a run killed between the header
        write and the first chunk) and reads as ``None`` here.
    mu_extent
        The centroid count the ζ BLOCK was written at: axis 1 for
        G-flat, axis 2 for r-space.  ``None`` iff ``dataset_name`` is.
        This is the DATASET's opinion; the header's is
        :attr:`r_mu_fft_idx`, and the two disagreeing is a corrupt file.
    zeta_done
        ``isdf_header/zeta_is_done``, or ``None`` when the file predates
        the flag or has no ``isdf_header``.  ``None`` is NOT ``False``:
        the test a caller wants is ``probe.zeta_done is False``, which
        means "a writer stamped this and never came back".
    r_mu_fft_idx
        ``isdf_header/centroids/r_mu_fft_idx`` as ``int64``, or ``None``.
        Its ``shape[0]`` is the HEADER's centroid count and its values
        are flat FFT-box indices, so a caller can check both the count
        and the grid they were built on.
    """

    exists: bool
    readable: bool
    error: str | None
    dataset_name: str | None
    mu_extent: int | None
    zeta_done: bool | None
    r_mu_fft_idx: np.ndarray | None


def probe_zeta_file(path: str | Path) -> ZetaFileProbe:
    """Read what can be read from ``path``.  **NEVER RAISES.**

    Not "raises rarely" — never, for any input, including ``None``, a
    directory, a zero-byte file, an HDF5 file belonging to something
    else, and a ζ truncated mid-write.  That is a CONTRACT, not a
    convenience: both call sites are pre-fit guards that must print and
    continue on a broken file, because the thing they are about to do is
    overwrite it.  A probe that raised would turn "the stale ζ here is
    garbage, refitting" into a traceback at startup.

    One open, one pass, no jax, no lorrax.  Anything unreadable comes
    back as ``readable=False`` plus :attr:`~ZetaFileProbe.error`.
    """
    try:
        p = os.fspath(path)
    except TypeError as exc:                                # noqa: BLE001
        return ZetaFileProbe(False, False, f"{type(exc).__name__}: {exc}",
                             None, None, None, None)

    try:
        exists = os.path.exists(p)
    except Exception as exc:                                # noqa: BLE001
        return ZetaFileProbe(False, False, f"{type(exc).__name__}: {exc}",
                             None, None, None, None)
    if not exists:
        return ZetaFileProbe(False, False, None, None, None, None, None)

    name: str | None = None
    mu_extent: int | None = None
    zeta_done: bool | None = None
    r_mu: np.ndarray | None = None
    try:
        with h5.File(p, "r") as f:
            for _name, _mu_axis in _ZETA_DATASETS:
                dset = f.get(_name)
                if dset is not None and getattr(dset, "ndim", None) == 3:
                    name = _name
                    mu_extent = int(dset.shape[_mu_axis])
                    break
            hdr = f.get("isdf_header")
            if hdr is not None:
                if "zeta_is_done" in hdr:
                    zeta_done = bool(np.asarray(hdr["zeta_is_done"])[()])
                cent = hdr.get("centroids/r_mu_fft_idx")
                if cent is not None:
                    r_mu = np.asarray(cent, dtype=np.int64)
    except Exception as exc:                                # noqa: BLE001
        # Everything: OSError from a non-HDF5 or truncated file, KeyError
        # from a half-written group, IsADirectoryError, a native h5py
        # error with no Python class of its own.  A narrower except here
        # is how a probe that PROMISES not to raise starts raising.
        return ZetaFileProbe(True, False, f"{type(exc).__name__}: {exc}",
                             None, None, None, None)

    return ZetaFileProbe(True, True, None, name, mu_extent, zeta_done, r_mu)
