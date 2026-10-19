"""Map-0 QP identities from projector overlaps (host readout only)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_qp_identity(reference_u, current_u, trusted_mask, *, priority_mask=None):
    """Match QP columns to fixed labels by projector overlap, not by sorting.

    Parameters
    ----------
    reference_u, current_u : (nk, nb, nb) complex host arrays
        Column rotations ``U[k,m,n] = <DFT_m|QP_n>`` on the same loop k set.
        ``reference_u=None`` is the DFT basis itself (``U = I``), whose
        overlaps are ``|current_u|^2`` without a product.
    trusted_mask : (nb,) or (nk, nb) bool
        Reference labels (columns of ``reference_u``) to match. Candidate
        columns span the whole active window because a trusted state can
        cross a scissored sorted level.
    priority_mask : optional bool mask
        Reserves the original trusted labels' optimal assignment before
        newly tracked labels take the remaining columns.

    Returns
    -------
    indices, weights : (nk, nb) host arrays
        Label to current sorted column (-1 outside the trusted set) and the
        assigned projector weight ``|<ref_label|cur_column>|^2``.  Inside an
        exactly degenerate multiplet the pairing of individual labels to
        columns is a gauge choice; the set of columns the multiplet takes is
        not, and every consumer reads energies, which are equal there.
    """
    u = np.asarray(current_u)
    u0 = None if reference_u is None else np.asarray(reference_u)
    nk, nb = u.shape[:2]
    mask = np.asarray(trusted_mask, dtype=bool)
    if (u.shape != (nk, nb, nb) or (u0 is not None and u0.shape != u.shape)
            or mask.shape not in ((nb,), (nk, nb))):
        raise ValueError('SC identity: inconsistent rotation/mask shapes')
    if not mask.any():
        raise ValueError('SC identity: empty trusted subspace')
    if not np.isfinite(u).all() or (u0 is not None and not np.isfinite(u0).all()):
        raise ValueError('SC identity: non-finite rotation')
    mask = np.broadcast_to(mask, (nk, nb))
    priority = (mask if priority_mask is None else
                np.broadcast_to(np.asarray(priority_mask, dtype=bool), (nk, nb)) & mask)
    indices = np.full((nk, nb), -1, dtype=int)
    weights = np.full((nk, nb), np.nan)
    for k in range(u.shape[0]):  # one Hungarian solve per k, as on main
        labels = np.flatnonzero(mask[k])
        overlap = (np.abs(u[k]) ** 2 if u0 is None
                   else np.abs(u0[k].conj().T @ u[k]) ** 2)[labels]
        # Established readout identities keep their own optimum; newly
        # tracked labels take the remaining columns.
        available = np.arange(nb)
        for chosen in (np.flatnonzero(priority[k, labels]),
                       np.flatnonzero(~priority[k, labels])):
            if not chosen.size:
                continue
            rows, columns = linear_sum_assignment(
                overlap[np.ix_(chosen, available)], maximize=True)
            indices[k, labels[chosen[rows]]] = available[columns]
            weights[k, labels[chosen[rows]]] = overlap[chosen[rows], available[columns]]
            available = np.setdiff1d(available, available[columns])
    return indices, weights
