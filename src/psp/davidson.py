"""psp/davidson.py — backwards-compatible shim (moved to solvers.davidson)."""
from solvers.davidson import davidson as _davidson
from solvers.davidson import warmup_davidson_jit


def warmup_jit(nG, nspinor, n_tgt, m_max=None):
    return warmup_davidson_jit(nG, nspinor, n_tgt, m_max)


def davidson_k(apply_H, *, h_diag, nG, nspinor, n_tgt, T_diag=None, **kw):
    return _davidson(apply_H, h_diag=h_diag, dim=nG, n_channels=nspinor,
                     n_eig=n_tgt, diag_for_init=T_diag, **kw)
