"""Tiny host-only factor oracle for shared-pole physical realization tests."""
import numpy as np


def full_q_operator(factors, weights, tables, child_operations):
    """Average residue congruences, then unfold; never call the projector."""
    qt, sym = tables['qirr'], tables['sym']
    rows = np.asarray(sym.active_symmetry_rows)
    rotations, _, antiunitary = sym.operation_rows(rows)
    parents = []
    for iq, q in enumerate(qt.q_irr_frac):
        members = []
        for op, rotation, anti in zip(rows, rotations, antiunitary):
            difference = rotation @ q - q
            if not np.allclose(difference, np.rint(difference), atol=1e-12):
                continue
            phase = np.exp(2j*np.pi*(qt.L_table[op] @ q))
            b = factors[iq, qt.sym_perm[op], 0, :] * phase[:, None]
            if anti:
                b = b.conj()
            members.append((b * weights[iq]) @ b.conj().T)
        assert members
        parents.append(sum(members) / len(members))
    children = []
    for iq, op in zip(qt.irr_idx_q, child_operations):
        value = parents[iq]
        phase = np.exp(2j*np.pi*(qt.L_table[op] @ qt.q_irr_frac[iq]))
        if op >= qt.n_sym_spatial:
            value, phase = value.T, phase.conj()
        perm = qt.sym_perm[op]
        children.append(value[np.ix_(perm, perm)] * phase[:, None] * phase.conj()[None, :])
    return np.asarray(children)
