"""Double-parallel sample grid for the MPA-W fit stage.

This module owns sampling geometry only.  The executable quadratures, fit,
and Sigma windows are described in
``docs/theory/THEORY_mpa_implementation.md``.

Sources
-------
The protocol implemented here is the published double-parallel sampling
of the multipole-W papers:

* ``docs/theory/THEORY_mpa_implementation.md`` -- the binding LORRAX
  interpretation and its production validity boundary.
* Leon, Ferretti, Varsano, Molinari, Cardoso, *Efficient full frequency
  GW for metals using a multipole approach for the dielectric
  screening*, Phys. Rev. B **107**, 155130 (2023).  Markdown copy in
  this tree's sibling salvage checkout at
  ``~/projects/lorrax_perlmutter_salvage/reports/multipole/Multipole-W-metals.md``
  -- section II C, Eqs. (10) and (11), which are the two parallel lines
  and the semi-homogeneous powers-of-two real partition generalised to
  the exponent ``alpha``.
* Leon, Berland, Cardoso, *Multipole approach for the self-energy*
  (2025), same directory, ``multipole-sigma-2025.md``, section II B --
  the same sampling reused for MPA-Sigma.

What is published and what is a call
------------------------------------
Equation (11) of the metals paper tabulates the partition explicitly for
``n_p = 1..7`` and then prints "...".  There is no published closed form
beyond ``n_p = 7``.  ``partition_fractions`` reproduces the tabulated
sets EXACTLY for ``n_p <= 7`` (this is a test, not a hope) and extends
them with a single stated rule -- see that function's docstring for the
rule and for the boundary at which the extension first becomes
load-bearing.

Units
-----
The published protocol is quoted in Hartree; LORRAX is Ry-native.  The
``energy_unit`` argument selects only the DEFAULTS for the two line
heights and the metal origin shift.  ``omega_m`` is always in the
caller's unit and is never converted.  Passing the line heights
explicitly means the caller owns their units.
"""

from fractions import Fraction

import numpy as np

# The published line heights (metals paper section II C): the near line
# at varpi_1 = 0.1 Ha and the far line at varpi_2 = 1 Ha.  The Ry column
# is the same physical height expressed in LORRAX-native Rydberg.
_VARPI_NEAR = {"Ha": 0.1, "Ry": 0.2}
_VARPI_FAR = {"Ha": 1.0, "Ry": 2.0}

# The metal origin shift: z_1^1 = i * 1e-5 Ha, a stability dodge around
# zero-energy intraband transitions -- NOT a physical broadening.
_METAL_ORIGIN_SHIFT = {"Ha": 1.0e-5, "Ry": 2.0e-5}

_MATERIAL_CLASSES = ("insulator", "metal")
_ENERGY_UNITS = ("Ha", "Ry")

# The published partition sets, in ascending order, as exact fractions of
# omega_m BEFORE the alpha exponent is applied (metals paper Eq. 11).
# Held here verbatim so ``partition_fractions`` can be checked against
# the paper rather than against itself.
PUBLISHED_PARTITION_FRACTIONS = {
    1: (Fraction(0),),
    2: (Fraction(0), Fraction(1)),
    3: (Fraction(0), Fraction(1, 2), Fraction(1)),
    4: (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)),
    5: (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2),
        Fraction(1)),
    6: (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2),
        Fraction(3, 4), Fraction(1)),
    7: (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(3, 8),
        Fraction(1, 2), Fraction(3, 4), Fraction(1)),
}

# Published/reported pole schedules, kept as documentation
# for callers; nothing in this module reads them.
POLE_SCHEDULE = {
    "Si": 8, "hBN": 10, "TiO2": 10, "Al": 8, "Na": 8, "Cu": 12,
}


def partition_fractions(n_p, *, ladder_floor_octaves=3):
    """Return the ``n_p`` real-axis partition fractions of ``omega_m``.

    The returned tuple is ascending, starts at ``Fraction(0)``, ends at
    ``Fraction(1)`` for ``n_p >= 2``, and is computed in exact rational
    arithmetic so that the nesting property below is bit-exact after the
    float conversion.

    THE RULE.  Start from ``{0, 1}``.  Bisect the origin-adjacent
    interval ``[0, x_1]`` repeatedly while its width exceeds
    ``2**-ladder_floor_octaves`` -- this lays down the geometric ladder
    ``0, 1/8, 1/4, 1/2, 1`` that the paper tabulates for ``n_p <= 5``.
    Thereafter bisect the WIDEST remaining interval, ties broken toward
    the origin, never re-entering ``[0, x_1]``.

    WHAT IS PUBLISHED.  Metals-paper Eq. (11) tabulates ``n_p = 1..7``
    and then prints "...".  The rule above reproduces all seven
    tabulated sets exactly.  It also reproduces, for ``n_p = 6..9``, the
    paper's own verbal description -- a "semi-homogeneous partition in
    powers of two", i.e. a fine block of step ``1/8`` abutting a coarse
    block of step ``1/4``, with the boundary sliding right as ``n_p``
    grows, reaching the full eighths grid at ``n_p = 9``.

    WHAT IS A CALL (2026-08-08).  Beyond ``n_p = 9`` the greedy refines
    the widest INTERIOR interval (adding ``3/16`` at ``n_p = 10``) rather
    than lowering the origin floor (which would add ``1/16``).  Both are
    nested and both are consistent with the paper's prose; the interior
    choice is taken because the origin interval is already the densest
    region in the ``alpha``-warped coordinate and because refining it
    further pushes samples toward the ``z = 0`` point that the near line
    already covers exactly.  ``ladder_floor_octaves`` is the knob: raise
    it to 4 to get the origin-refining family instead.  This call is
    inert for every schedule at or below ``n_p = 9`` -- Si 8, Al 8, Na 8
    -- and first bites at hBN/TiO2 (10-11) and Cu (12-15).

    NESTING.  ``partition_fractions(n)`` is a strict subset of
    ``partition_fractions(n + 1)`` for every ``n >= 1``: each step only
    ever inserts, never moves.  This is what the theory plan means by
    "nested partitions are retained so growing ``n_p`` adds samples
    instead of moving them".
    """

    n = int(n_p)
    if n < 1:
        raise ValueError(
            f"GATE n_p_positive: n_p={n_p!r} is not a positive integer. "
            "FALSE case: int(n_p) >= 1.")
    floor_oct = int(ladder_floor_octaves)
    if floor_oct < 0:
        raise ValueError(
            f"GATE ladder_floor_nonnegative: ladder_floor_octaves="
            f"{ladder_floor_octaves!r} is negative. "
            "FALSE case: int(ladder_floor_octaves) >= 0.")

    if n == 1:
        return (Fraction(0),)

    pts = [Fraction(0), Fraction(1)]
    floor_width = Fraction(1, 2 ** floor_oct)

    # Phase 1 -- the geometric ladder into the origin.
    while len(pts) < n and (pts[1] - pts[0]) > floor_width:
        pts.insert(1, (pts[0] + pts[1]) / 2)

    # Phase 2 -- widest interior interval, ties toward the origin.
    while len(pts) < n:
        widths = [pts[i + 1] - pts[i] for i in range(1, len(pts) - 1)]
        if not widths:
            raise ValueError(
                f"GATE partition_exhausted: no interior interval left to "
                f"bisect at n_p={n} with ladder_floor_octaves={floor_oct}. "
                "FALSE case: n_p small enough that the ladder plus the "
                "interior bisections supply n_p distinct points.")
        widest = max(widths)
        # ``index`` returns the FIRST maximum -> ties break toward 0.
        k = widths.index(widest) + 1
        pts.insert(k + 1, (pts[k] + pts[k + 1]) / 2)

    return tuple(pts)


def partition_omegas(n_p, omega_m, *, alpha=1, ladder_floor_octaves=3):
    """Return the ``n_p`` real-axis sample energies, ascending.

    ``omega_n = (fraction)**alpha * omega_m`` -- metals paper Eq. (11),
    where the exponent sits on the FRACTION and not on ``omega_m``.
    ``alpha = 1`` is the linear partition (insulators and Na);
    ``alpha = 2`` is the quadratic partition (Al and Cu), which
    concentrates points at low frequency without changing ``omega_m`` or
    ``n_p``.
    """

    a = int(alpha)
    if a not in (1, 2):
        raise ValueError(
            f"GATE alpha_supported: alpha={alpha!r} is not a scheduled "
            "partition exponent. FALSE case: int(alpha) in (1, 2) -- 1 for "
            "insulators and Na, 2 for Al and Cu (theory plan section B).")
    om = float(omega_m)
    if not (om > 0.0) or not np.isfinite(om):
        raise ValueError(
            f"GATE omega_m_positive: omega_m={omega_m!r} is not a finite "
            "positive energy. FALSE case: float(omega_m) > 0 and finite; "
            "omega_m is the largest included transition energy.")

    fracs = partition_fractions(n_p, ladder_floor_octaves=ladder_floor_octaves)
    return np.asarray([float(f) ** a * om for f in fracs], dtype=np.float64)


def double_parallel_grid(
    n_p,
    omega_m,
    *,
    material_class="insulator",
    alpha=1,
    varpi_near=None,
    varpi_far=None,
    energy_unit="Ha",
    ladder_floor_octaves=3,
):
    """Return the ``2*n_p`` complex sample points of the DP protocol.

    Parameters
    ----------
    n_p
        Number of poles.  The grid carries exactly ``2*n_p`` points --
        the fit's sample support, no more and no less.
    omega_m
        Largest included transition energy, in the caller's unit.
    material_class
        ``"insulator"`` or ``"metal"``.  Selects the near line's first
        sample: ``z = 0`` exactly for insulators, ``z = i*1e-5 Ha`` for
        metals (metals paper section II C -- a stability dodge around
        zero-energy intraband transitions, not a broadening).
    alpha
        Partition exponent, 1 or 2.  See ``partition_omegas``.
    varpi_near, varpi_far
        Line heights.  ``None`` takes the published defaults for
        ``energy_unit``: 0.1 and 1 Ha (equivalently 0.2 and 2 Ry).
        Passing them explicitly means the caller owns their units.
    energy_unit
        ``"Ha"`` or ``"Ry"``; selects defaults only.

    Returns
    -------
    numpy.ndarray, shape ``(2*n_p,)``, dtype complex128
        NEAR line first (``n_p`` points, ascending in ``Re z``), then the
        FAR line (``n_p`` points, ascending in ``Re z``).  This ordering
        is part of the contract: it is what makes the grid nested as a
        PREFIX-per-line rather than only as a set, and the fit's
        diagnostics quote sample indices against it.

    Nesting
    -------
    ``double_parallel_grid(n_p, ...)`` is a strict subset of
    ``double_parallel_grid(n_p + 1, ...)`` with identical remaining
    arguments -- growing ``n_p`` ADDS one point to each line and moves
    none.  ``tests/test_mpa_fit_kernel.py`` asserts this bit-exactly.
    """

    n = int(n_p)
    if n < 1:
        raise ValueError(
            f"GATE n_p_positive: n_p={n_p!r} is not a positive integer. "
            "FALSE case: int(n_p) >= 1.")
    if material_class not in _MATERIAL_CLASSES:
        raise ValueError(
            f"GATE material_class_known: material_class={material_class!r} "
            f"is not one of {_MATERIAL_CLASSES}. FALSE case: "
            "material_class == 'insulator' or 'metal'.")
    if energy_unit not in _ENERGY_UNITS:
        raise ValueError(
            f"GATE energy_unit_known: energy_unit={energy_unit!r} is not "
            f"one of {_ENERGY_UNITS}. FALSE case: energy_unit == 'Ha' or "
            "'Ry'.")

    vn = _VARPI_NEAR[energy_unit] if varpi_near is None else float(varpi_near)
    vf = _VARPI_FAR[energy_unit] if varpi_far is None else float(varpi_far)
    if not (0.0 < vn < vf):
        raise ValueError(
            f"GATE varpi_ordering: (varpi_near, varpi_far)=({vn!r}, {vf!r}) "
            "is not a strictly ordered pair of positive line heights. "
            "FALSE case: 0 < varpi_near < varpi_far -- the near line "
            "resolves structure, the far line carries the envelope.")

    omegas = partition_omegas(
        n, omega_m, alpha=alpha, ladder_floor_octaves=ladder_floor_octaves)

    near = omegas + 1j * vn
    far = omegas + 1j * vf

    # The near line's first sample is the special one.  Insulators put it
    # exactly at the origin; metals shift it up the imaginary axis.
    if material_class == "insulator":
        near[0] = 0.0 + 0.0j
    else:
        near[0] = 1j * _METAL_ORIGIN_SHIFT[energy_unit]

    grid = np.concatenate([near, far]).astype(np.complex128)

    # A duplicated sample makes the Pade Vandermonde exactly singular, so
    # refuse here rather than in the solve.  Note x = z**2 is what the fit
    # actually sees, so the test is on x: +w and -w would collide.
    x = grid * grid
    scale = float(np.max(np.abs(x))) if x.size else 1.0
    gaps = np.abs(x[:, None] - x[None, :]) + np.eye(x.size) * max(scale, 1.0)
    min_gap = float(np.min(gaps))
    if min_gap <= 1.0e-12 * max(scale, 1.0):
        raise ValueError(
            f"GATE distinct_squared_samples: the {x.size} sample points "
            f"collide in x = z**2 (min separation {min_gap:.3e} against "
            f"scale {scale:.3e}) for n_p={n}, omega_m={omega_m!r}, "
            f"alpha={alpha!r}, varpi=({vn!r}, {vf!r}). FALSE case: the "
            "2*n_p values z_j**2 are pairwise separated by more than "
            "1e-12*max|z**2| -- the Pade-in-z^2 solve is singular "
            "otherwise.")

    return grid
