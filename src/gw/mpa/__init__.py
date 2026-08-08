"""Multipole-W (MPA) fit kernel — staging package.

STAGING LOCATION.  This package is infrastructure landed ahead of the
minimax-service design review; the minimax-service design decides the
final home.  Its placement under ``src/gw/mpa`` is a parking spot, NOT a
ruling.  Everything here is deliberately dependency-light and movable:
jax + numpy + stdlib only, no service imports, no module-level mutable
state, no classes.

Contents
--------
``sampling``
    The double-parallel sample-grid constructor (two horizontal lines in
    the complex-frequency plane, semi-homogeneous powers-of-two real
    partition, nested in ``n_p``).
``pade_fit``
    The fit kernel itself: normalised Pade-in-z^2 linear solve with
    z_max scaling, companion-matrix roots, all-2*n_p-point complex
    least-squares residues, four ordered guards, mandatory residue refit.
``diagnostics``
    Conditioning, backward error, held-out-sample residual, and the
    perturbation-refit propagation harness.
"""

from . import diagnostics, pade_fit, sampling

__all__ = ["sampling", "pade_fit", "diagnostics"]
