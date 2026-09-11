"""The quadrature rules a GW run actually used, for the production report.

Producers record a rule where it is committed to a contraction, never where a
candidate is built: the probe-reuse path builds a dedicated imaginary-axis
rule only to borrow its nodes.  The lifetime is ``common.timing``'s: the
driver resets both at startup and the report reads this log at the end.  Keys
overwrite, so a self-consistent loop reports the rules of its last map.
Energies are in Ry.
"""

from __future__ import annotations

_CHI: dict[tuple, dict] = {}
_SIGMA: dict = {"geometry": None, "plans": 0}


def reset() -> None:
    _CHI.clear()
    _SIGMA.update(geometry=None, plans=0)


def record_minimax(kind, quad, *, target, omega_ry=0.0, nodes=None,
                   error=None, note=""):
    """A Laplace-family screening rule: ``static``, ``imag`` or ``real``.

    ``quad.provenance`` is ``minimax.Provenance.one_line()``: its first word
    is where the rule came from and its last word the certificate.
    """
    provenance = str(quad.provenance or "unrecorded")
    _CHI[(kind, float(omega_ry))] = {
        "kind": kind, "omega_ry": float(omega_ry),
        "x_min": float(quad.x_min), "x_max": float(quad.x_max),
        "nodes": int(quad.node_count if nodes is None else nodes),
        "error": float(quad.max_error if error is None else error),
        "target": float(target),
        "source": provenance.split(" ", 1)[0],
        "certified": provenance.endswith(" CERTIFIED"),
        "note": str(note),
    }


def record_line(rule, *, points, sweeps):
    """A damped Gauss-Legendre line rule; every point on it shares the nodes.

    ``sweeps`` is 2 when the contraction evaluates each node at +t and -t.
    """
    _CHI[("line", float(rule["varpi"]), float(rule["freq_max"]))] = {
        "kind": "line", "varpi_ry": float(rule["varpi"]),
        "freq_max_ry": float(rule["freq_max"]), "a_dim": float(rule["a_dim"]),
        "nodes": int(rule["n_nodes"]), "sweeps": int(sweeps),
        "panels": int(rule["n_panels"]), "kappa0": float(rule["kappa0"]),
        "target": float(rule["rel_tol"]), "points": int(points),
    }


def record_direct(*, z):
    """A sample evaluated by the exact ordered-pair scan (no time nodes)."""
    z = complex(z)
    _CHI[("direct", z.real, z.imag)] = {
        "kind": "direct", "omega_ry": z.real, "varpi_ry": z.imag}


def record_sigma_plan(geometry):
    """The geometry dict ``plan_sigma_windows`` returns, windows included."""
    _SIGMA["geometry"] = geometry
    _SIGMA["plans"] += 1


def chi_rules() -> list[dict]:
    return list(_CHI.values())


def sigma_plan() -> tuple[dict | None, int]:
    return _SIGMA["geometry"], int(_SIGMA["plans"])
