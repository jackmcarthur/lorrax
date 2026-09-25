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
        ``reference_u=None`` is the DFT basis itself (``U = I``), whose
        overlaps are ``|current_u|^2`` without a product.
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
    u = np.asarray(current_u)
    u0 = None if reference_u is None else np.asarray(reference_u)
    e0, e = np.asarray(reference_e_ev), np.asarray(current_e_ev)
    mask = np.asarray(trusted_mask, dtype=bool)
    if (e.ndim != 2 or e0.shape != e.shape or
            u.shape != e.shape + (e.shape[1],)
            or (u0 is not None and u0.shape != u.shape) or
            mask.shape not in ((e.shape[1],), e.shape)):
        raise ValueError('SC identity: inconsistent rotation/spectrum/mask shapes')
    if not mask.any():
        raise ValueError('SC identity: empty trusted subspace')
    if not all(np.isfinite(a).all() for a in (u, e0, e)) or (
            u0 is not None and not np.isfinite(u0).all()):
        raise ValueError('SC identity: non-finite rotation or spectrum')
    mask = np.broadcast_to(mask, e.shape)
    priority = (mask if priority_mask is None else
                np.broadcast_to(np.asarray(priority_mask, dtype=bool), e.shape) & mask)
    indices = np.full(e.shape, -1, dtype=int)
    blocks = np.full(e.shape, -1, dtype=int)
    energies = np.full(e.shape, np.nan)
    weights = np.full(e.shape, np.nan)
    nb = e.shape[1]
    for k in range(e.shape[0]):
        labels = np.flatnonzero(mask[k])
        # BGW adjacent-gap grouping, at the SC exact-degeneracy tolerance:
        # contiguous blocks [start, start + size) of the sorted spectrum.
        starts = np.r_[0, np.flatnonzero(np.diff(e0[k]) > degeneracy_tol_ev) + 1]
        sizes = np.diff(np.r_[starts, nb])
        n_trusted = np.add.reduceat(mask[k].astype(np.intp), starts)
        cut = np.flatnonzero((n_trusted > 0) & (n_trusted < sizes))
        if cut.size:
            group = np.arange(starts[cut[0]], starts[cut[0]] + sizes[cut[0]])
            raise ValueError(f'SC identity: the reference label set cuts a '
                             f'multiplet at k={k}, columns={group.tolist()}')
        overlap = (np.abs(u[k]) ** 2 if u0 is None
                   else np.abs(u0[k].conj().T @ u[k]) ** 2)
        # Every member of a trusted reference multiplet scores the
        # multiplet's summed overlap; a singleton's sum is its own row.
        score = overlap[labels]
        selected = n_trusted > 0
        multiplets = [np.arange(s0, s0 + g) for s0, g in
                      zip(starts[selected & (sizes > 1)], sizes[selected & (sizes > 1)])]
        for group in multiplets:
            score[np.searchsorted(labels, group)] = overlap[group].sum(axis=0)
        current = np.r_[0, np.flatnonzero(np.diff(e[k]) > degeneracy_tol_ev) + 1]
        current_sizes = np.diff(np.r_[current, nb])
        for c0, g in zip(current[current_sizes > 1], current_sizes[current_sizes > 1]):
            group = np.arange(c0, c0 + g)
            score[:, group] = score[:, group].mean(axis=1, keepdims=True)
        # Keep established readout identities' original optimization intact.
        # Newly tracked labels receive the remaining columns; admitting a
        # state must not relabel an already reported multiplet.
        first = np.flatnonzero(priority[k, labels])
        later = np.flatnonzero(~priority[k, labels])
        assigned = np.empty(len(labels), dtype=int)
        available = np.arange(nb)
        for chosen in (first, later):
            if not chosen.size:
                continue
            rows, columns = linear_sum_assignment(
                score[np.ix_(chosen, available)], maximize=True)
            assigned[chosen[rows]] = available[columns]
            available = np.setdiff1d(available, available[columns])
        singles = starts[selected & (sizes == 1)]
        rows = np.searchsorted(labels, singles)
        members = assigned[rows]
        indices[k, singles] = members
        blocks[k, singles] = singles
        energies[k, singles] = e[k, members]
        weights[k, singles] = score[rows, members]
        for group in multiplets:
            rows = np.searchsorted(labels, group)
            members = np.sort(assigned[rows])
            indices[k, group] = members
            blocks[k, group] = group[0]
            energies[k, group] = e[k, members].mean()
            weights[k, group] = score[rows[0], members].sum() / len(group)
    return indices, energies, blocks, weights
