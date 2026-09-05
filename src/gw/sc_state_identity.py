"""Map-0 QP identities from multiplet-projector overlaps (host readout only)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_qp_identity(reference_u, reference_e_ev, current_u, current_e_ev,
                       trusted_mask, *, degeneracy_tol_ev, priority_mask=None):
    """Match QP states to fixed map-0 multiplets without sorting identities.

    Parameters
    ----------
    reference_u, current_u : (nk, nb, nb) complex host arrays
        Column rotations ``U[k,m,n] = <DFT_m|QP_n>`` on the same loop k set.
    reference_e_ev, current_e_ev : (nk, nb) real host arrays
        Sorted spectra in eV, paired with the rotation columns.
    trusted_mask : (nb,) or (nk, nb) bool
        Reference labels (columns of ``reference_u``) to match. Candidate
        columns span the whole active window because a trusted state can
        cross a scissored sorted level. A per-k mask names, at each k, the
        map-0 output columns that carry the trusted DFT bands (they are not
        the sorted positions of those bands once a scissored multiplet has
        crossed them: Na Gamma, arms N1/N2, 2026-09-05).
    degeneracy_tol_ev : float
        The SC exact-degeneracy tolerance, in eV (not the window margin).

    Returns
    -------
    indices, energies_ev, block_labels, weights : (nk, nb) host arrays
        Map-0 label to current sorted column (-1 outside the trusted set),
        block-mean energies (NaN outside), reference block's first label,
        and assigned projector weight per member. Members of a reference
        multiplet are capacity slots with IDENTICAL scores: maximizing
        ``sum_n <u_n|P_block|u_n>`` is invariant to its internal gauge.
        Only the assigned set and its mean have identity; individual
        column pairings within a multiplet have no physical meaning.

    Notes
    -----
    Current exact-degenerate columns receive their block-averaged score,
    so their arbitrary eigenvector gauge cannot affect the assignment.
    At an accidental degeneracy shared by distinct reference multiplets,
    capacity slots can divide that eigenspace; its members have the same
    energy and are reported only through their reference-block means.
    The SC map uses assignments to classify state energies in reference
    coordinates. Output energies remain exact-multiplet means for readout.
    ``priority_mask``, when supplied, reserves the original trusted labels'
    optimal assignment before newly tracked labels take the remaining columns.
    """
    u0, u = np.asarray(reference_u), np.asarray(current_u)
    e0, e = np.asarray(reference_e_ev), np.asarray(current_e_ev)
    mask = np.asarray(trusted_mask, dtype=bool)
    if (e.ndim != 2 or e0.shape != e.shape or
            u.shape != e.shape + (e.shape[1],) or u0.shape != u.shape or
            mask.shape not in ((e.shape[1],), e.shape)):
        raise ValueError('SC identity: inconsistent rotation/spectrum/mask shapes')
    if not mask.any():
        raise ValueError('SC identity: empty trusted subspace')
    if not all(np.isfinite(a).all() for a in (u0, u, e0, e)):
        raise ValueError('SC identity: non-finite rotation or spectrum')
    mask = np.broadcast_to(mask, e.shape)
    priority = (mask if priority_mask is None else
                np.broadcast_to(np.asarray(priority_mask, dtype=bool), e.shape) & mask)
    indices = np.full(e.shape, -1, dtype=int)
    blocks = np.full(e.shape, -1, dtype=int)
    energies = np.full(e.shape, np.nan)
    weights = np.full(e.shape, np.nan)
    for k in range(e.shape[0]):
        labels = np.flatnonzero(mask[k])
        # BGW adjacent-gap grouping, at the SC exact-degeneracy tolerance.
        groups = np.split(np.arange(e.shape[1]),
                          np.flatnonzero(np.diff(e0[k]) > degeneracy_tol_ev) + 1)
        overlap = np.abs(u0[k].conj().T @ u[k]) ** 2
        score = np.empty((len(labels), e.shape[1]))
        selected_groups = []
        for group in groups:
            if not mask[k, group].any():
                continue
            if not mask[k, group].all():
                raise ValueError(f'SC identity: the reference label set cuts a '
                                 f'multiplet at k={k}, columns={group.tolist()}')
            rows = np.searchsorted(labels, group)
            score[rows] = overlap[group].sum(axis=0)
            selected_groups.append((group, rows))
        current_groups = np.split(np.arange(e.shape[1]),
                                  np.flatnonzero(np.diff(e[k]) > degeneracy_tol_ev) + 1)
        for group in current_groups:
            if len(group) > 1:
                score[:, group] = score[:, group].mean(axis=1, keepdims=True)
        # Keep established readout identities' original optimization intact.
        # Newly tracked labels receive the remaining columns; admitting a
        # state must not relabel an already reported multiplet.
        first = np.flatnonzero(priority[k, labels])
        later = np.flatnonzero(~priority[k, labels])
        assigned = np.empty(len(labels), dtype=int)
        available = np.arange(e.shape[1])
        for selected in (first, later):
            if not selected.size:
                continue
            rows, columns = linear_sum_assignment(
                score[np.ix_(selected, available)], maximize=True)
            assigned[selected[rows]] = available[columns]
            available = np.setdiff1d(available, available[columns])
        for group, rows in selected_groups:
            members = np.sort(assigned[rows])
            indices[k, group] = members
            blocks[k, group] = group[0]
            energies[k, group] = e[k, members].mean()
            weights[k, group] = score[rows[0], members].sum() / len(group)
    return indices, energies, blocks, weights
