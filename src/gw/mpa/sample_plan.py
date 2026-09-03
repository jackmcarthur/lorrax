"""The complex-frequency sampling object: R4's 2x2 table, as data.

STAGING LOCATION; the minimax-service design decides the final home.
``DESIGN_minimax.md`` R4 point 3 says this object "belongs in the
service" because it is pure algebra on floats -- no jax, no physics, no
bands -- and this module honours that: it imports ``numpy``,
``gw.mpa.sampling`` and the ``minimax`` door (whose declared table the
character dispatch reads) and nothing else.  It sits under ``gw/mpa``
only because that is where the rest of the MPA staging lives; moving it
is a file move.

WHAT THIS IS
------------
``DESIGN_minimax.md`` R4 draws the four analytic characters of a
screening sample ``z = omega + i*varpi`` as a 2x2 table, and observes
that LORRAX already implements three quarters of it as three unrelated
code paths, with ``screening.py``'s *"complex-axis omega=... not
supported"* standing in for the fourth::

    |            |  omega = 0             |  omega != 0            |
    |------------|------------------------|------------------------|
    | varpi = 0  |  exponential_sum       |  sine_sum              |
    |            |  target 1/x            |  target x/(x^2-w^2)    |
    | varpi != 0 |  exponential_sum_imag  |  damped_line           |
    |            |  target x/(x^2+p^2)    |  the MPA kernel        |

The organising fact -- and the reason the four are ONE object rather
than four paths -- is that the four targets are the same function of
``Delta``, evaluated at four positions of ``z``.  Writing the MPA
kernel of theory-plan section B in the form that makes this visible::

    K_z(Delta) = -2 * integral_0^inf dt  e^{i z t} sin(Delta t)
               = -2 * Delta / (Delta**2 - z**2)          (Im z > 0)

and then walking the table:

* ``z = 0``            -> ``K = -2 / Delta``                (1/x)
* ``z = i*varpi``      -> ``K = -2 * Delta/(Delta^2+varpi^2)``
* ``z = omega``        -> ``K = -2 * Delta/(Delta^2-omega^2)``
* ``z = omega+i*varpi`` -> the strip; nothing existing evaluates it.

So the three live families are exactly ``K_z / (-2)`` at their cell's
``z``, and unifying them costs one constant, recorded here as
``KERNEL_FACTOR``.  That is the whole content of "the existing real and
imaginary families are CASES of the sampling object, not parallel
paths": the table below dispatches on the analytic character of ``z``,
and the ROUTE -- which machine evaluates the cell -- is a separate
column from the TARGET, because three cells have a shipped machine and
the fourth needs the damped-tau sweep of ``gw.mpa.evaluator``.

WHAT IS A PLAN
--------------
A plan is a plain dict::

    plan = {"label": str, "points": (point, ...)}
    point = {"index": int, "role": str, "z": complex,
             "omega": float, "varpi": float,
             "character": str, "family": str, "route": str}

The point tuple is ORDERED and that order is the contract: for
``mpa_plan`` it is exactly ``sampling.double_parallel_grid``'s order
(near line ascending, then far line), because the fit kernel indexes
its diagnostics against that order.  Everything else -- which points
share a damped-tau line, which take a shipped kernel -- is DERIVED by
``plan_routes`` from the points themselves, so the grouping cannot
drift out of step with the grid.

The plan carries NO commitment to how a line is evaluated
(``DESIGN_minimax.md`` section 5.1: "the sampling plan must not encode
line-batching ... a per-point evaluator consuming the same plan is the
fallback with no API change").  ``plan_routes`` reports the lines that
EXIST in the geometry; whether the evaluator rides one sweep per line
or one per point is the evaluator's argument, not the plan's.

WHY EXACT ZERO DECIDES THE CHARACTER
------------------------------------
``sample_character`` tests ``omega == 0.0`` and ``varpi == 0.0``
exactly, not against a tolerance.  Two reasons.  The protocol
constructs its zeros exactly -- ``sampling.partition_fractions``
returns ``Fraction(0)`` and ``float(Fraction(0))**alpha * omega_m`` is
``0.0`` bit-for-bit -- so there is nothing to round.  And a sample that
is merely NEAR an axis is analytically ON THE STRIP: the metals
protocol's own origin shift ``z = i*1e-5 Ha`` is a case in point, and
it is a genuine ``exponential_sum_imag`` point, not a static one, for
exactly the reason the papers introduce it.  A tolerance here would
silently reclassify it.
"""

import numpy as np

from minimax import CHARACTERS as _SERVICE_CHARACTERS
from minimax import families_for_character as _families_for_character

from gw.mpa import sampling

#: ``K_z(Delta) = KERNEL_FACTOR * target_of_the_cell(Delta)``.  The
#: three shipped families all approximate the target WITHOUT this
#: factor -- it lives in the chi0 builders today -- so every route
#: adapter applies it once, in one place.
KERNEL_FACTOR = -2.0

#: R4's 2x2, as data.  ``cell`` is ``(omega_is_zero, varpi_is_zero)``.
#: Catalog metadata (builders, domains, shipped-table census) lives on
#: the minimax service's declared table, not here.
FAMILIES = {
    "exponential_sum": {
        "character": "static",
        "cell": (True, True),
        "target": "1/Delta",
        "route": "existing-kernel",
    },
    "exponential_sum_imag": {
        "character": "imag",
        "cell": (True, False),
        "target": "Delta/(Delta**2 + varpi**2)",
        "route": "existing-kernel",
    },
    "sine_sum": {
        "character": "real",
        "cell": (False, True),
        "target": "Delta/(Delta**2 - omega**2)",
        "route": "existing-kernel",
    },
    "damped_line": {
        "character": "strip",
        "cell": (False, False),
        "target": "Delta/(Delta**2 - z**2)",
        "route": "damped-tau",
    },
}

#: character -> family name, read off the minimax service's declared R4
#: table.  Several service families can serve one character (the "imag"
#: cell), but they share one mathematical route name -- that shared route
#: is the one-to-one map this module dispatches on.
_FAMILY_BY_CHARACTER = {
    character: {spec.route for spec in _families_for_character(character)}.pop()
    for character in _SERVICE_CHARACTERS}


def sample_character(z):
    """Return ``'static'``, ``'imag'``, ``'real'`` or ``'strip'``.

    The analytic character of ``z = omega + i*varpi`` -- which cell of
    R4's table the point falls in.  See the module docstring for why
    the zero tests are exact.

    Refuses ``varpi < 0`` and non-finite ``z``.  The lower half plane
    is not an empty cell of the table, it is the WRONG SHEET: the
    time-ordered ``W_c`` is analytic in the upper half plane, the
    damped-tau integral ``integral_0^inf e^{i z t} ...`` diverges for
    ``Im z < 0``, and the fit kernel's own time-ordering guard exists
    to keep fitted poles out of there.  A sample below the real axis is
    a caller bug, and it is cheaper to say so here than to return a
    number nobody can use.
    """

    zc = complex(z)
    if not (np.isfinite(zc.real) and np.isfinite(zc.imag)):
        raise ValueError(
            f"GATE sample_finite: z={z!r} is not finite. FALSE case: "
            "both components of the sample point are finite floats.")
    if zc.imag < 0.0:
        raise ValueError(
            f"GATE sample_upper_half_plane: z={z!r} has varpi="
            f"{zc.imag!r} < 0, which is not a cell of the sampling "
            "table but the wrong analytic sheet -- W_c is analytic in "
            "the upper half plane and the damped-tau integral "
            "diverges below the real axis. FALSE case: Im z >= 0.")
    omega_zero = (zc.real == 0.0)
    varpi_zero = (zc.imag == 0.0)
    for name, spec in FAMILIES.items():
        if spec["cell"] == (omega_zero, varpi_zero):
            return spec["character"]
    raise AssertionError(f"unreachable: {name!r}")   # pragma: no cover


def family_for(z):
    """Return the R4 family name serving ``z``.  The 2x2 dispatch."""

    return _FAMILY_BY_CHARACTER[sample_character(z)]


def sample_point(z, role, *, index=0):
    """One sample point, as a dict: the atom the plan is built from."""

    zc = complex(z)
    fam = family_for(zc)
    return {
        "index": int(index),
        "role": str(role),
        "z": zc,
        "omega": float(zc.real),
        "varpi": float(zc.imag),
        "character": FAMILIES[fam]["character"],
        "family": fam,
        "route": FAMILIES[fam]["route"],
    }


def sampling_plan(points, *, label):
    """Freeze an ordered tuple of points into a plan dict.

    Reindexes the points so ``point['index']`` is its position in the
    plan -- the index the fit's diagnostics quote and the index into
    the evaluated sample-value array.  Refuses duplicate roles, because
    a role is what Sigma looks a W up by and a repeated one silently
    loses a sample.
    """

    pts = tuple(points)
    roles = [p["role"] for p in pts]
    if len(set(roles)) != len(roles):
        dupes = sorted({r for r in roles if roles.count(r) > 1})
        raise ValueError(
            f"GATE distinct_roles: the plan repeats the role(s) "
            f"{dupes} among {len(pts)} points. FALSE case: every "
            "point carries a role no other point carries -- the role "
            "is the key Sigma reads W back by.")
    return {
        "label": str(label),
        "points": tuple(dict(p, index=k) for k, p in enumerate(pts)),
    }


def plan_points(plan):
    """The plan's ordered points."""

    return plan["points"]


def plan_z(plan):
    """The plan's sample grid as ``(n_points,)`` complex128.

    This is the array the fit kernel takes as ``z_samples``; for
    ``mpa_plan`` it is bit-identical to ``double_parallel_grid``'s
    return, which ``tests/test_mpa_evaluator.py`` pins.
    """

    return np.asarray([p["z"] for p in plan["points"]],
                      dtype=np.complex128)


def plan_routes(plan):
    """Group the plan by evaluation route.  DERIVED, never stored.

    Returns ``{"existing": (point, ...), "lines": ((varpi, (point,
    ...)), ...)}`` -- the shipped-kernel points in plan order, and the
    damped-tau points grouped by their line height, lines ascending in
    ``varpi``.  A per-point evaluator ignores the grouping and a
    line-batched one uses it; both consume the same plan, which is the
    property ``DESIGN_minimax.md`` section 5.1 asks the plan to have.
    """

    existing = tuple(p for p in plan["points"]
                     if p["route"] == "existing-kernel")
    by_varpi = {}
    for p in plan["points"]:
        if p["route"] != "damped-tau":
            continue
        by_varpi.setdefault(p["varpi"], []).append(p)
    lines = tuple((v, tuple(by_varpi[v])) for v in sorted(by_varpi))
    return {"existing": existing, "lines": lines}


def refuse_unsupported(plan, *, delta_max=None):
    """Refuse, by name, any point no family can serve.  F6.

    Two things can be wrong, and both are checked BEFORE any physics
    runs, which is the point of having the plan be pure float algebra:

    1. A ``sine_sum`` point (``varpi = 0``, ``omega != 0``) at or below
       the top transition.  The shipped real-axis route decomposes
       ``Delta/(Delta**2-omega**2)`` into two ``1/y`` minimaxes and
       needs ``omega`` above every transition to keep both branches
       positive; inside the band the target has a real pole and no
       quadrature of it exists.  ``delta_max`` is the top transition
       energy in the caller's unit; passing ``None`` skips this check
       and is only right when the plan has no such points.
    2. Anything ``sample_character`` itself refuses -- non-finite, or
       below the real axis.  Those already fired at construction, so
       this pass re-runs them only to make the plan self-checking after
       a hand edit.

    Returns ``None``.  Raises ``ValueError`` naming the offending
    ``z``, its role and its cell.
    """

    for p in plan["points"]:
        sample_character(p["z"])
        if p["family"] != "sine_sum" or delta_max is None:
            continue
        if p["omega"] <= float(delta_max):
            raise ValueError(
                f"GATE sine_sum_above_band: plan {plan['label']!r} "
                f"point {p['role']!r} at z={p['z']!r} is a real-axis "
                f"sample at omega={p['omega']!r}, which is not above "
                f"the top transition Delta_max={float(delta_max)!r}. "
                "The shipped real-axis route "
                "(minimax_screening.build_real_quadrature) needs "
                "omega > Delta_max so both 1/y branches stay "
                "positive; at or below it the target has a real pole. "
                "FALSE case: every varpi == 0, omega != 0 sample sits "
                "above every transition energy.")


def describe_plan(plan):
    """A one-line-per-cell census of the plan.  For cost reports."""

    counts = {}
    for p in plan["points"]:
        counts[p["family"]] = counts.get(p["family"], 0) + 1
    cells = ", ".join(
        f"{name}={counts.get(name, 0)}" for name in FAMILIES)
    return (f"plan {plan['label']!r}: {len(plan['points'])} points "
            f"[{cells}]")


# ---------------------------------------------------------------------------
# The plan the pipeline actually asks for
# ---------------------------------------------------------------------------

def mpa_plan(
    n_p,
    omega_m,
    *,
    material_class="insulator",
    alpha=1,
    schedule="nested",
    varpi_near=None,
    varpi_far=None,
    origin_shift=None,
    energy_unit="Ha",
):
    """The double-parallel MPA protocol, as a plan.

    A thin projection of ``sampling.double_parallel_grid`` -- every
    argument is forwarded verbatim and the point order is the grid's,
    so ``plan_z(mpa_plan(...))`` is bit-identical to
    ``double_parallel_grid(...)``.  The plan adds exactly one thing to
    the grid: the per-point cell and route of R4's table.

    WHAT THE CELLS COME OUT AS, and why that is the theory plan's
    sentence rather than a choice made here.  For an insulator the near
    line's first sample is ``z = 0`` -- the ``static`` cell -- and the
    far line's first sample is ``i*varpi_2`` -- the ``imag`` cell;
    every other sample has a nonzero real part and lands on the
    ``strip``.  For a metal the near line's first sample is
    ``i*origin_shift`` (default ``i*1e-5 Ha`` = ``i*2e-5 Ry``), which is
    ``imag`` and not ``static`` at any legal shift, because the shift is
    strictly positive by ``double_parallel_grid``'s own gate.  That is
    theory-plan section B's "the special pure-imaginary samples ... use
    the existing static and imaginary-axis kernels" read off the table
    instead of hand-listed, and it is why the fourth cell is the only
    thing the MPA fit stage actually needed built.
    """

    grid = sampling.double_parallel_grid(
        n_p, omega_m, material_class=material_class, alpha=alpha,
        schedule=schedule,
        varpi_near=varpi_near, varpi_far=varpi_far,
        origin_shift=origin_shift, energy_unit=energy_unit)
    n = int(n_p)
    pts = tuple(
        sample_point(z, f"{'near' if k < n else 'far'}_{k % n:02d}",
                     index=k)
        for k, z in enumerate(grid))
    label = (f"mpa-double-parallel-n_p={n}-{material_class}"
             f"-alpha={int(alpha)}")
    if str(schedule).lower() != "nested":
        label += f"-schedule={str(schedule).lower()}"
    return sampling_plan(pts, label=label)


def faraday_imaginary_plan(
    n_p,
    omega_max_ry,
    *,
    alpha=1,
    schedule="nested",
):
    """Return the nested imaginary-axis support for the Faraday MPA fit.

    The Hall head is even in ``z`` and its charge-head completion accepts
    imaginary-axis samples only.  Reuse the MPA partition owner rather than
    introducing a second abscissa rule: an ``n_p``-pole Hall fit receives
    ``2*n_p`` points from :func:`sampling.partition_omegas`, placed at
    ``z=i*omega``.  Consequently the one-, two-, and three-pole supports are
    nested sets.  For the standard 2 Ry endpoint their union is
    ``{0, .25i, .5i, 1i, 1.5i, 2i}`` Ry.

    This is sampling geometry only.  It does not change the double-parallel
    body MPA plan or any screening/Sigma quadrature.
    """
    n = int(n_p)
    if n < 1:
        raise ValueError("Faraday MPA n_p must be positive")
    omega = sampling.partition_omegas(
        2 * n, omega_max_ry, alpha=alpha, schedule=schedule)
    points = tuple(
        sample_point(1j * value, f"faraday_imag_{index:02d}", index=index)
        for index, value in enumerate(omega))
    plan = sampling_plan(
        points,
        label=(f"faraday-imaginary-mpa-n_p={n}-alpha={int(alpha)}"
               f"-schedule={str(schedule).strip().lower()}"),
    )
    return {
        **plan,
        "n_poles": n,
        "sampling_alpha": int(alpha),
        "sampling_schedule": str(schedule).strip().lower(),
        "omega_max_ry": float(omega_max_ry),
    }
