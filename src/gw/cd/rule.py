"""Imaginary-axis quadrature of the CD route: the two-panel tan-mapped Gauss-Legendre rule.

Physics-free (would lift into the ``minimax`` service unchanged; it lives here because the
sealed Perlmutter runtime pins the services to the module's own tree, so a worktree's
service edit is not importable under ``lx run``).  Run170 ``build_contracted_actions.py:70-80``
verbatim, as the Fe harness ``fe_cd.py:55-63`` (reproduces ``fe_cd_states.csv``).

    u = scale * tan(theta), theta in [0, pi/4] (round(n/5) nodes) and [pi/4, pi/2] (the rest);
    weight = w_GL * (right - left) * scale / (2 cos^2 theta).
"""
from __future__ import annotations

import numpy as np


def tan_mapped_gauss_rule(n: int, scale: float):
    """Nodes u (same unit as ``scale``) and weights of the n-node rule on (0, inf)."""
    n = int(n)
    if n < 5:
        raise ValueError(f"tan_mapped_gauss_rule: n must be >= 5, got {n}")
    low = round(n / 5)
    parts = []
    for count, left, right in ((low, 0.0, np.pi / 4), (n - low, np.pi / 4, np.pi / 2)):
        q, w = np.polynomial.legendre.leggauss(count)
        theta = (right - left) * q / 2 + (right + left) / 2
        parts.append((scale * np.tan(theta), w * (right - left) * scale / (2 * np.cos(theta) ** 2)))
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
