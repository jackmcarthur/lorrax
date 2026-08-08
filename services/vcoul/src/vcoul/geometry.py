"""The Coulomb geometry, as ONE object instead of four loose arguments.

Every routine in this service needs the same four numbers about a cell:
the Cartesian reciprocal rows, the cell volume, and — for the 0-D box
kernel only — the reciprocal metric and the FFT grid.  Before the
extraction they travelled as four keyword arguments, or as a ``wfn``
object the kernels reached into, and **five call sites independently
wrote** ``wfn.blat * wfn.bvec`` (``gw/coulomb/base.py:227,325``,
``bulk_3d.py:62``, ``slab_2d.py:72,94``).  A product that five callers
have to remember to take is a footgun with a five-year fuse: one of them
eventually passes ``wfn.bvec`` alone and every number downstream is off
by the lattice constant with no shape error to say so.

:class:`CoulombGeometry` takes the PRODUCT, never the pair.
:meth:`CoulombGeometry.from_wfn` is the one place ``blat * bvec`` is
written IN THIS SERVICE and in ``gw/coulomb/`` — a package-scoped claim,
not a tree-wide one: ``gw/gw_init.py`` and ``gw/isdf_fitting.py`` still
take the product by hand on the live GW path, and nothing gates that.
``from_wfn`` is duck-typed — anything with ``blat``, ``bvec`` and
``cell_volume`` attributes works, so the service never imports a lorrax
loader class to accept a lorrax loader object.

Units: ``bvec`` rows are Cartesian reciprocal lattice vectors in
inverse bohr (blat folded in); ``cell_volume`` is Ω in bohr³.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CoulombGeometry"]


@dataclass(frozen=True, eq=False)
class CoulombGeometry:
    """Frozen cell geometry: Cartesian reciprocal ROWS + Ω (+ box extras).

    Parameters
    ----------
    bvec
        ``(3, 3)`` float64.  ROWS are the Cartesian reciprocal lattice
        vectors b1, b2, b3 in 1/bohr, **with the lattice constant already
        folded in** (``blat * bvec`` in WFN.h5 terms).  The row convention
        is load-bearing: ``U @ bvec`` spans the b1,b2,b3 parallelepiped,
        which is the fundamental domain the mini-BZ draw needs.  ``U @
        bvec.T`` spans the COLUMN parallelepiped, which is not — that is
        the 2026-08-07 head-draw bug, and taking the rows as a named field
        is what stops the question being asked again per call site.
    cell_volume
        Ω, the real-space unit cell volume in bohr³.
    bdot
        ``(3, 3)`` reciprocal metric ``b_i · b_j`` in BGW's 2π/alat units.
        Read ONLY by the 0-D box kernel (the Wigner-Seitz FFT is not a
        closed form in ``bvec``).  ``None`` everywhere else.
    fft_grid
        ``(3,)`` int FFT box dimensions.  Read only by the 0-D box kernel.

    ``eq=False`` on purpose: the fields are numpy arrays, so a generated
    ``__eq__`` would return an array and every ``geom == other`` in a
    caller would raise "truth value of an array is ambiguous" at a
    distance.  Identity equality and identity hashing are the honest
    contract for an array bag.
    """

    bvec: np.ndarray
    cell_volume: float
    bdot: np.ndarray | None = None
    fft_grid: np.ndarray | None = None

    def __post_init__(self):
        bvec = np.asarray(self.bvec, dtype=np.float64)
        if bvec.shape != (3, 3):
            raise ValueError(
                f"CoulombGeometry: bvec must be (3, 3) Cartesian reciprocal "
                f"ROWS (1/bohr, blat folded in); got shape {bvec.shape}.  "
                f"If you have a WFN-style loader, use "
                f"CoulombGeometry.from_wfn(wfn) rather than multiplying "
                f"blat * bvec by hand.")
        object.__setattr__(self, "bvec", bvec)
        object.__setattr__(self, "cell_volume", float(self.cell_volume))
        if self.bdot is not None:
            object.__setattr__(self, "bdot",
                               np.asarray(self.bdot, dtype=np.float64))
        if self.fft_grid is not None:
            object.__setattr__(self, "fft_grid",
                               np.asarray(self.fft_grid, dtype=int))

    @classmethod
    def from_wfn(cls, wfn) -> "CoulombGeometry":
        """Build from any object carrying ``blat``, ``bvec``, ``cell_volume``.

        DUCK-TYPED, deliberately: this service declares no dependency on
        lorrax, so it cannot name ``file_io.WfnLoader`` — and it does not
        need to.  ``bdot`` and ``fft_grid`` are taken when present (the
        0-D box kernel needs them; nothing else does).

        This is the ONE place ``blat * bvec`` is written in this service
        (and in ``gw/coulomb/``).  It is NOT the only place in LORRAX —
        ``gw/gw_init.py`` and ``gw/isdf_fitting.py`` still hand-multiply,
        and no test pins the difference — so treat the exclusivity as the
        intent to route new call sites here, not as a checked invariant.
        """
        bdot = getattr(wfn, "bdot", None)
        fft_grid = getattr(wfn, "fft_grid", None)
        return cls(
            bvec=np.asarray(wfn.blat * wfn.bvec, dtype=np.float64),
            cell_volume=float(wfn.cell_volume),
            bdot=bdot,
            fft_grid=fft_grid,
        )

    @property
    def fact(self) -> float:
        """``1/Ω`` — the BerkeleyGW volume factor applied to every v(q+G)."""
        return 1.0 / float(self.cell_volume)
