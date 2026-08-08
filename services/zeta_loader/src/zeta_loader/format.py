"""The ``zeta_q.h5`` FORMAT surface: what a file is, without opening a run.

Two functions and a record, all pure h5py + numpy.  Nothing here imports
jax, ``file_io`` or anything else from the LORRAX tree, so this half of
the service works with lorrax off ``sys.path`` entirely — which is not a
purity exercise: both callers run BEFORE a :class:`~zeta_loader.ZetaLoader`
could exist (they gate whether the ζ fit runs at all), and the second one
runs AFTER the collective handle has closed.

:func:`probe_zeta_file`
    Never raises.  Answers "what is on disk here?" for a path that may
    not exist, may be a directory, may be a zero-byte file left by a
    crashed job, or may be someone else's HDF5.
:func:`write_g0_mu`
    The one sanctioned serial append into a ζ file after the phdf5
    handle is gone.

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

__all__ = ["ZetaFileProbe", "probe_zeta_file", "write_g0_mu"]


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


def write_g0_mu(path: str | Path, g0_logical, *,
                n_rmu_expected: int | None = None) -> tuple[int, ...]:
    """Append ``g0_mu`` to a ζ file.  THE one post-close serial write.

    ``g0_mu`` is ``G0[q, μ] = ζ̃_μ(q, G=0)``, produced by the V_q pass
    and written back into ``zeta_q.h5``.  It is the ONLY thing anything
    writes into a ζ file after the fit, and it is written by rank 0 with
    SERIAL h5py, after the collective phdf5 handle has closed.  This
    function exists so that write has a door instead of being four lines
    of raw ``h5py`` in the GW driver — the service owns a READ contract
    on a file it could not otherwise enforce the writing of.

    THE CALLER OWNS THE RANK-0 GATING AND THE BARRIER, and that is not
    an oversight.  This function is unconditionally serial: it opens the
    file ``mode='a'`` and writes.  Calling it from every rank is a
    concurrent-writer corruption, and calling it before every rank has
    dropped its phdf5 handle is a torn file.  The sequence the caller
    must provide is::

        loader.close()                       # or the SlabIO ctx exits
        barrier()                            # every rank has let go
        if jax.process_index() == 0:
            write_g0_mu(path, g0, n_rmu_expected=meta.n_rmu)
        barrier()                            # nobody reads it early

    Putting the gate inside would make it look safe to call anywhere,
    and the failure it hides is silent file corruption on a shared
    filesystem, which is the worst class of defect this codebase has.

    Parameters
    ----------
    path
        The ζ file.  Opened ``'a'``; an existing ``g0_mu`` is DELETED and
        recreated rather than written into, because a re-run at a
        different centroid count would otherwise hit a shape mismatch
        against a stale dataset.
    g0_logical
        ``(n_q, n_rmu_logical)``.  LOGICAL, not padded: files on disk
        store logical extents so they re-read identically on any process
        count, and the pad entries are exact zeros anyway.  Clipping the
        in-memory padded array is the caller's, because only the caller
        knows which of its axes is μ.
    n_rmu_expected
        Optional guard.  When given it must equal ``g0_logical``'s LAST
        axis, or this refuses.  It is what turns "the caller clipped
        correctly" from a convention into a check.

    Returns
    -------
    tuple[int, ...]
        The shape written, so a caller can log or assert on it.
    """
    arr = np.asarray(g0_logical)
    if n_rmu_expected is not None:
        if arr.ndim == 0:
            raise ValueError(
                f"write_g0_mu({path!s}): n_rmu_expected="
                f"{int(n_rmu_expected)} was given but g0_logical is a "
                f"scalar, which has no μ axis to check it against.")
        got = int(arr.shape[-1])
        if got != int(n_rmu_expected):
            raise ValueError(
                f"write_g0_mu({path!s}): g0_logical's last axis is {got} "
                f"but n_rmu_expected={int(n_rmu_expected)}.  g0_mu must be "
                f"written at the LOGICAL centroid count — a padded array "
                f"would put zero rows on disk that every later reader "
                f"would take for real ζ̃(G=0) values.  Clip the μ axis to "
                f"the logical extent before calling "
                f"(g0[..., :n_rmu]), or pass the count this array really "
                f"has.  Full shape: {tuple(int(s) for s in arr.shape)}.")

    with h5.File(os.fspath(path), "a") as f:
        if "g0_mu" in f:
            del f["g0_mu"]
        f.create_dataset("g0_mu", data=arr)
    return tuple(int(s) for s in arr.shape)
