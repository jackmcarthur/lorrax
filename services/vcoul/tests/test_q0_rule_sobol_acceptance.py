"""Sobol-distribution acceptance for the exact Gamma-cell rules.

This is a compute-node certificate, not an ordinary unit test.  It compares
the production exact Wigner--Seitz polygon/polyhedron rules with independent
finite-variance scrambled-Sobol references of the same cell integrals.  Each
reported draw is the mean of ten ``2**18``-point Sobol replicates, matching
the historical production estimator's sample count and seed construction.
In 2-D the full inscribed disk is integrated separately: the bare radial
integral is analytic, the screened radial Jacobian cancels the ``1/|q|``
cusp before a certified product rule, and Sobol samples only the polygon
outside the disk.  In 3-D the exact singular sphere is likewise removed
before sampling.  Every sampled remainder therefore has finite variance.

The physical static-response tensors are frozen from the two P=4 head
captures that motivated this gate:

* MoS2 3x3 scalar slab, ``runs/MoS2/34_*/03_scalar_gnppm``;
* Si 4x4x4, ``runs/DEV/102_*/lane_BULK/evidence/step4/si_native_exact``.

Hexagonal/cubic symmetry is applied to those measured tensors to remove only
the finite-run symmetry noise.  The elongated bulk case uses the physical-
scale response from the production bulk sign oracle on that same geometry.
The literal measured values remain below for audit.

Run this module from the directory that should receive
``q0_rule_sobol_acceptance.{json,txt}``::

    pytest -q -m compute_node /path/to/test_q0_rule_sobol_acceptance.py

The test computes every distribution and writes both artifacts before making
one aggregate assertion, so a failing quantity cannot hide the other draws.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

# Standalone service collection installs the same setting in conftest, but
# keeping it here also makes direct execution/import of this certificate safe.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import vcoul  # noqa: E402


pytestmark = [
    pytest.mark.vcoul,
    pytest.mark.extra,
    pytest.mark.gpu,
    pytest.mark.compute_node,
]

_N_TRIALS = 36
_NSAMPLES = 2**18
_QMC_REPS = 10
_RTOL = 1.0e-8
_ATOL = 1.0e-12
_SLAB_DISK_ORDERS = (32, 48, 64)

# P=4 MoS2 3x3 scalar-slab restart, retained literally for provenance.
_MOS2_SLAB_S_MEASURED = np.asarray([
    [-3.17115033e-1 + 1.78687889e-18j,
     -5.42651333e-6 - 1.58947209e-13j,
     +4.87385386e-6 + 1.05740162e-12j],
    [-5.42651333e-6 + 1.58949415e-13j,
     -3.17127031e-1 - 4.22080290e-18j,
     +9.84904558e-6 + 4.13847141e-13j],
    [+4.87385386e-6 - 1.05740164e-12j,
     +9.84904558e-6 - 4.13847287e-13j,
     -1.56895102e-2 + 4.04886093e-19j],
], dtype=np.complex128)

# P=4 Si 4x4x4 S_cart_head, retained literally for provenance.
_SI_S_MEASURED = np.asarray([
    [-8.34388997e-1 + 5.95442330e-18j,
     -1.53899086e-6 + 6.40244636e-11j,
     -2.17940824e-6 - 8.31365859e-11j],
    [-1.53899086e-6 - 6.40244581e-11j,
     -8.34381855e-1 + 4.48708487e-18j,
     -4.31913405e-7 - 1.25485042e-9j],
    [-2.17940824e-6 + 8.31365849e-11j,
     -4.31913405e-7 + 1.25485041e-9j,
     -8.34393119e-1 + 3.21766914e-18j],
], dtype=np.complex128)

# Physical-scale response used by test_photon_head_sign_oracle.py on the
# elongated bulk geometry.  Unlike numerical symmetry noise, its xy mixing is
# intentional and is retained.
_BULK_ORACLE_S = np.asarray((
    (-1.2e-2, 3.0e-3, 0.0),
    (3.0e-3, -8.0e-3, 0.0),
    (0.0, 0.0, -5.0e-3),
), dtype=np.float64)


def _geometry_from_direct_rows(rows):
    direct = np.asarray(rows, dtype=np.float64)
    bvec = 2.0 * np.pi * np.linalg.inv(direct).T
    return vcoul.CoulombGeometry(
        bvec=bvec, cell_volume=float(abs(np.linalg.det(direct))))


def _cases():
    """Two geometries per dimension and their measured physical S."""
    # Literal reciprocal rows/cell volume from the MoS2 WFN used by both
    # P=4 captures.  This is the requested MoS2 3x3 hexagonal cell.
    mos2_bvec = np.asarray((
        (1.0507622167739268, 0.6066563718417468, 0.0),
        (0.0, 1.2133158112027018, 0.0),
        (0.0, 0.0, 0.27707653970333357),
    ))
    slab_hex = vcoul.CoulombGeometry(
        bvec=mos2_bvec, cell_volume=702.2011633533148)
    mos2_direct = 2.0 * np.pi * np.linalg.inv(mos2_bvec).T
    direct_lengths = np.linalg.norm(mos2_direct, axis=1)
    slab_orthorhombic = _geometry_from_direct_rows((
        (direct_lengths[0], 0.0, 0.0),
        (0.0, 1.21 * direct_lengths[1], 0.0),
        (0.0, 0.0, direct_lengths[2]),
    ))
    bulk_cubic_bvec = np.eye(3)
    bulk_cubic = vcoul.CoulombGeometry(
        bvec=bulk_cubic_bvec,
        cell_volume=float((2.0 * np.pi) ** 3))
    bulk_hex_bvec = np.asarray((
        (1.0, 0.0, 0.0),
        (0.5, np.sqrt(3.0) / 2.0, 0.0),
        (0.0, 0.0, 1.0 / 3.0),
    ))
    bulk_hex = vcoul.CoulombGeometry(
        bvec=bulk_hex_bvec,
        cell_volume=float(
            (2.0 * np.pi) ** 3 / abs(np.linalg.det(bulk_hex_bvec))))

    slab_inplane = float(
        np.real(np.trace(_MOS2_SLAB_S_MEASURED[:2, :2])) / 2.0)
    slab_axial = float(np.real(_MOS2_SLAB_S_MEASURED[2, 2]))
    S_mos2_slab = np.diag((slab_inplane, slab_inplane, slab_axial))
    si_isotropic = float(np.real(np.trace(_SI_S_MEASURED)) / 3.0)
    S_si = np.eye(3) * si_isotropic
    return (
        dict(name="slab_mos2_3x3_hexagonal", dimension=2,
             geometry=slab_hex, kgrid=(3, 3, 1), S=S_mos2_slab),
        dict(name="slab_orthorhombic", dimension=2,
             geometry=slab_orthorhombic, kgrid=(3, 3, 1), S=S_mos2_slab),
        dict(name="bulk_cubic", dimension=3,
             geometry=bulk_cubic, kgrid=(3, 3, 3), S=S_si),
        dict(name="bulk_hexagonal_c_over_a_3", dimension=3,
             geometry=bulk_hex, kgrid=(3, 3, 3), S=_BULK_ORACLE_S),
    )


def _values_from_q(q_cart, *, dimension, geometry, kgrid, S):
    """All sampled quantities before the optional analytic-sphere split."""
    q = np.asarray(q_cart, dtype=np.float64)
    q2 = np.einsum("qi,qi->q", q, q, optimize=True)
    q2_safe = np.where(q2 > 0.0, q2, 1.0)
    v = 8.0 * np.pi / q2_safe
    if dimension == 2:
        zc = np.pi / float(np.asarray(geometry.bvec)[2, 2])
        kxy = np.linalg.norm(q[:, :2], axis=1)
        v *= 1.0 - np.exp(-zc * kxy)
    v = np.where(q2 > 0.0, v, 0.0)
    qSq = np.einsum("qi,ij,qj->q", q, S, q, optimize=True)
    w = v / (1.0 - v * qSq)
    values = {
        "bare_v": v,
        "screened_w": w,
        "tt_trace": -2.0 * v,
    }
    if dimension == 3:
        for axis, label in enumerate("xyz"):
            values[f"tt_{label}{label}"] = (
                -v * (1.0 - q[:, axis] ** 2 / q2_safe))
    return values, q2


def _bulk_sphere_terms(geometry, kgrid, S):
    """Analytic singular contributions inside the inscribed bulk sphere.

    The radial integral is closed form.  Only the smooth angular factor in
    ``W`` remains; a fixed high-order product rule evaluates that factor,
    with its own adjacent-order ratios retained in the artifact.
    """
    q0sph2 = vcoul.minibz_inscribed_sphere_r2(
        geometry.bvec, kgrid, is_2d=False)
    radius = float(np.sqrt(q0sph2))
    bare = (4.0 * radius * float(geometry.cell_volume)
            * float(np.prod(kgrid)) / np.pi)
    angular_ladder = []
    for order in (32, 48, 64):
        z, wz = np.polynomial.legendre.leggauss(order)
        phi = 2.0 * np.pi * (np.arange(2 * order) + 0.5) / (2 * order)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z[:, None] ** 2))
        directions = np.stack((
            np.broadcast_to(radial * np.cos(phi)[None, :],
                            (order, 2 * order)),
            np.broadcast_to(radial * np.sin(phi)[None, :],
                            (order, 2 * order)),
            np.broadcast_to(z[:, None], (order, 2 * order)),
        ), axis=-1).reshape(-1, 3)
        weights = np.repeat(wz / 2.0, 2 * order) / (2 * order)
        nSn = np.einsum(
            "ni,ij,nj->n", directions, S, directions, optimize=True)
        factor = float(np.sum(weights / (1.0 - 8.0 * np.pi * nSn)))
        angular_ladder.append((order, factor))
    ratios = _ladder_ratios([value for _, value in angular_ladder])
    if ratios[-1] > 1.0:
        raise AssertionError(
            "screened analytic-sphere angular rule did not converge: "
            f"ladder={angular_ladder!r}, ratios={ratios!r}")
    terms = {
        "bare_v": bare,
        "screened_w": bare * angular_ladder[-1][1],
        "tt_trace": -2.0 * bare,
        "tt_xx": -(2.0 / 3.0) * bare,
        "tt_yy": -(2.0 / 3.0) * bare,
        "tt_zz": -(2.0 / 3.0) * bare,
    }
    certificate = {
        "radius": radius,
        "screened_angular_ladder": [
            {"order": order, "value": value}
            for order, value in angular_ladder],
        "screened_angular_ladder_ratios": ratios,
    }
    return q0sph2, terms, certificate


def _slab_disk_terms(receipt, S):
    r"""Full inscribed-disk terms for the finite-variance slab reference.

    For in-plane ``q = r n`` the Ismail--Beigi kernel is

    ``v(r) = 8*pi*(1 - exp(-zc*r))/r**2``.

    Its disk measure contributes one radial power, so the bare integral is

    ``16*pi**2 * [gamma + log(zc*R) + E1(zc*R)]``.

    For ``W = v/(1-v*qSq)`` the same Jacobian makes the radial integrand
    finite at Gamma.  A fixed radial Gauss--Legendre / periodic angular rule
    therefore integrates the *whole* disk, not an asymptotic replacement.
    The TT disk is analytic because ``tr(P_T)=2`` pointwise and the angular
    mean of ``P_T`` for an in-plane circle is ``diag(1/2,1/2,1)``.
    Returned terms are already normalized by the mini-BZ polygon area.
    """
    if int(receipt.dimension) != 2 or receipt.slab_zc is None:
        raise ValueError("slab disk reference requires a dimension-2 receipt")
    polygon = np.asarray(receipt.polytope_vertices, dtype=np.float64)
    edge_distances = []
    for face in receipt.polytope_faces:
        if len(face) != 2:
            raise ValueError("slab receipt face must contain one polygon edge")
        left, right = polygon[np.asarray(face, dtype=np.int64)]
        edge = right - left
        edge_distances.append(
            abs(left[0] * right[1] - left[1] * right[0])
            / float(np.linalg.norm(edge)))
    radius = float(min(edge_distances))
    helper_radius = float(np.sqrt(vcoul.minibz_inscribed_sphere_r2(
        receipt.reciprocal_lattice_rows, receipt.kgrid, is_2d=True)))
    if not np.isclose(radius, helper_radius, rtol=2.0e-13, atol=2.0e-15):
        raise AssertionError(
            "receipt polygon and mini-lattice disagree on the inscribed "
            f"disk radius: polygon={radius:.17g}, helper={helper_radius:.17g}")

    area = float(receipt.minibz_measure)
    zc = float(receipt.slab_zc)
    scaled_radius = zc * radius
    if not (area > 0.0 and radius > 0.0 and scaled_radius > 0.0):
        raise AssertionError(
            "slab analytic-disk reference requires positive area, radius, "
            f"and zc; got area={area}, radius={radius}, zc={zc}")
    from scipy.special import exp1
    ein = float(np.euler_gamma + np.log(scaled_radius)
                + exp1(scaled_radius))
    bare = float(16.0 * np.pi**2 * ein / area)

    S2 = np.asarray(S, dtype=np.float64)[:2, :2]
    screened_ladder = []
    bare_radial_ladder = []
    for order in _SLAB_DISK_ORDERS:
        node, weight = np.polynomial.legendre.leggauss(order)
        radial = 0.5 * radius * (node + 1.0)
        radial_weight = 0.5 * radius * weight
        phi = (2.0 * np.pi
               * (np.arange(2 * order, dtype=np.float64) + 0.5)
               / (2 * order))
        direction = np.stack((np.cos(phi), np.sin(phi)), axis=1)
        nSn = np.einsum(
            "pi,ij,pj->p", direction, S2, direction, optimize=True)
        truncation = -np.expm1(-zc * radial[:, None])
        # This is r*v(r).  It tends to 8*pi*zc, so the quadrature never
        # evaluates a singular quantity even as its nodes approach Gamma.
        radial_times_v = 8.0 * np.pi * truncation / radial[:, None]
        denominator = (
            1.0 - 8.0 * np.pi * truncation * nSn[None, :])
        screened = float(
            (2.0 * np.pi / area)
            * np.sum(radial_weight[:, None]
                     * radial_times_v / denominator)
            / (2 * order))
        bare_radial = float(
            (2.0 * np.pi / area)
            * np.sum(radial_weight * radial_times_v[:, 0]))
        screened_ladder.append((int(order), screened))
        bare_radial_ladder.append((int(order), bare_radial))

    screened_ratios = _ladder_ratios(
        [value for _, value in screened_ladder])
    if screened_ratios[-1] > 1.0:
        raise AssertionError(
            "screened analytic-disk rule did not converge: "
            f"ladder={screened_ladder!r}, ratios={screened_ratios!r}")
    bare_errors = [abs(value - bare) for _, value in bare_radial_ladder]
    if bare_errors[-1] > 5.0e-11:
        raise AssertionError(
            "closed-form and radial-rule bare disk integrals disagree: "
            f"analytic={bare:.17g}, ladder={bare_radial_ladder!r}")

    terms = {
        "bare_v": bare,
        "screened_w": screened_ladder[-1][1],
        "tt_trace": -2.0 * bare,
    }
    certificate = {
        "method": (
            "analytic_ismail_beigi_disk_plus_smooth_screened_product_rule"),
        "radius": radius,
        "radius_from_minibz_helper": helper_radius,
        "minibz_area": area,
        "zc": zc,
        "scaled_radius_zc_R": scaled_radius,
        "bare_closed_form": bare,
        "bare_radial_ladder": [
            {"order": order, "value": value,
             "abs_error_vs_closed_form": error}
            for (order, value), error in zip(
                bare_radial_ladder, bare_errors)],
        "screened_ladder": [
            {"order": order, "value": value}
            for order, value in screened_ladder],
        "screened_ladder_ratios": screened_ratios,
        "tt_trace": terms["tt_trace"],
        "tt_tensor_diagonal": [
            -0.5 * bare, -0.5 * bare, -bare],
        "sobol_region": "Wigner-Seitz polygon minus closed inscribed disk",
    }
    return radius * radius, terms, certificate


def _ladder_ratios(values):
    ratios = []
    for previous, current in zip(values, values[1:]):
        scale = max(abs(previous), abs(current))
        ratios.append(abs(current - previous) / (_ATOL + _RTOL * scale))
    return [float(value) for value in ratios]


def _exact_rule(case):
    dimension = int(case["dimension"])
    kernel = vcoul.get_kernel(dimension)
    receipt = vcoul.minibz_photon_cubature(
        kernel, case["geometry"], case["kgrid"])
    ladders = {}
    for chunk in receipt.chunks:
        n = int(chunk.physical_count)
        weight = np.asarray(chunk.sample_weight[:n], dtype=np.float64)
        q = np.asarray(chunk.q_cart[:n], dtype=np.float64)
        D = np.asarray(chunk.D_raw[:n], dtype=np.float64)
        measure = float(np.sum(weight))
        qSq = np.einsum("qi,ij,qj->q", q, case["S"], q, optimize=True)
        v = D[:, 0, 0]
        values = {
            "bare_v": float(np.sum(weight * v) / measure),
            "screened_w": float(
                np.sum(weight * v / (1.0 - v * qSq)) / measure),
            "tt_trace": float(
                np.sum(weight * np.trace(
                    D[:, 1:, 1:], axis1=1, axis2=2)) / measure),
        }
        if dimension == 3:
            for axis, label in enumerate("xyz"):
                values[f"tt_{label}{label}"] = float(
                    np.sum(weight * D[:, axis + 1, axis + 1]) / measure)
        for name, value in values.items():
            ladders.setdefault(name, []).append(value)
    exact = {
        name: {
            "value": values[-1],
            "ladder_values": values,
            "ladder_ratios": _ladder_ratios(values),
        }
        for name, values in ladders.items()
    }
    scalar_certificates = []
    got_v, got_w = kernel.q0_average(
        case["geometry"], case["kgrid"], S_cart=case["S"],
        certificate_fn=scalar_certificates.append)
    np.testing.assert_allclose(
        [float(np.real(got_v)), float(np.real(got_w))],
        [exact["bare_v"]["value"], exact["screened_w"]["value"]],
        rtol=2.0e-15, atol=0.0)
    assert len(scalar_certificates) == 1
    return receipt, exact, scalar_certificates[0]


def _sobol_draw(case, *, trial, nsamples, qmc_reps,
                analytic_region):
    dimension = int(case["dimension"])
    region_r2, analytic_terms, _ = analytic_region

    replicate_values = []
    plain_replicate_values = []
    for rep in range(qmc_reps):
        seed = trial * qmc_reps + rep
        # One replicate per call keeps the production sampler's temporary
        # Voronoi distance array and result residency bounded.
        q = np.asarray(vcoul.minibz_voronoi_batches(
            case["geometry"].bvec,
            case["kgrid"],
            nsamples=nsamples,
            method="sobol",
            qmc_reps=1,
            nmax=3 if dimension == 3 else 1,
            is_2d=dimension == 2,
            seed_offset=seed,
        )[0], dtype=np.float64)
        values, q2 = _values_from_q(
            q, dimension=dimension, geometry=case["geometry"],
            kgrid=case["kgrid"], S=case["S"])
        outside = q2 > region_r2
        reduced = {
            name: float(np.sum(np.where(outside, value, 0.0))
                        / q.shape[0] + analytic_terms[name])
            for name, value in values.items()
        }
        replicate_values.append(reduced)
        if dimension == 2:
            plain_replicate_values.append({
                name: float(np.mean(value))
                for name, value in values.items()})
    corrected = {
        name: float(np.mean([values[name] for values in replicate_values]))
        for name in replicate_values[0]
    }
    plain = None
    if plain_replicate_values:
        plain = {
            name: float(np.mean(
                [values[name] for values in plain_replicate_values]))
            for name in plain_replicate_values[0]
        }
    return corrected, plain


def _summary(rule, draws):
    samples = np.asarray(draws, dtype=np.float64)
    n = int(samples.size)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    se = std / np.sqrt(n)
    z = abs(float(rule) - mean) / se if se > 0.0 else np.inf
    signed_above = int(np.count_nonzero(samples > rule))
    below = int(np.count_nonzero(samples < rule))
    magnitude_above = int(np.count_nonzero(
        np.abs(samples) > abs(float(rule))))
    magnitude_below = int(np.count_nonzero(
        np.abs(samples) < abs(float(rule))))
    mean_pass = bool(abs(float(rule) - mean) <= 2.0 * se)
    sign_pass = bool(magnitude_above / n <= 0.60)
    return {
        "N": n,
        "rule": float(rule),
        "mean": mean,
        "std": std,
        "SE": float(se),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "empirical_cdf_below_rule": below / n,
        "count_draws_above_rule_signed": signed_above,
        "fraction_draws_above_rule_signed": signed_above / n,
        "magnitude_cdf_below_rule_magnitude": magnitude_below / n,
        "count_draws_with_greater_magnitude_than_rule": magnitude_above,
        "fraction_draws_with_greater_magnitude_than_rule": (
            magnitude_above / n),
        "abs_rule_minus_mean_over_SE": float(z),
        "mean_within_2SE": mean_pass,
        "underestimate_magnitude_sign_test": sign_pass,
        "verdict": "PASS" if mean_pass and sign_pass else "FAIL",
        "sorted_draws": [float(value) for value in np.sort(samples)],
    }


def _format_report(payload):
    lines = [
        "Exact Gamma-cell rule vs finite-variance Sobol-reference distributions",
        (f"N={payload['config']['N']}, "
         f"nsamples={payload['config']['nsamples']}, "
         f"qmc_reps={payload['config']['qmc_reps']}"),
        ("Underestimation convention: compare absolute magnitudes; "
         "more_mass means |draw| > |rule|, including negative TT rows."),
        "",
    ]
    for case in payload["cases"]:
        cert = case["rule_certificate"]
        lines.extend((
            f"[{case['dimension']}D {case['geometry']}]",
            (f"method={cert['method']} orders={tuple(cert['orders'])} "
             f"counts={tuple(cert['physical_counts'])}"),
            (f"weight_sum_defects={cert['weight_sum_defects']} "
             f"weighted_q_centroids={cert['weighted_q_centroids']}"),
        ))
        lines.append(
            f"reference_method={case['reference_method']} "
            f"analytic_region={case['analytic_region']}")
        for name, stats in case["quantities"].items():
            ladder = case["exact"][name]
            lines.append(
                f"{name}: N={stats['N']} R={stats['rule']:.17g} "
                f"mean={stats['mean']:.17g} std={stats['std']:.9g} "
                f"SE={stats['SE']:.9g} min={stats['min']:.17g} "
                f"max={stats['max']:.17g} "
                f"CDF_signed(R)={stats['empirical_cdf_below_rule']:.6f} "
                f"CDF_mass(R)="
                f"{stats['magnitude_cdf_below_rule_magnitude']:.6f} "
                f"more_mass="
                f"{stats['count_draws_with_greater_magnitude_than_rule']}"
                f"/{stats['N']} "
                f"|R-mean|/SE={stats['abs_rule_minus_mean_over_SE']:.6f} "
                f"ladder={ladder['ladder_values']} "
                f"ratios={ladder['ladder_ratios']} {stats['verdict']}")
        if "plain_sobol_quantities" in case:
            lines.append("plain_sobol_diagnostic (same draws, no cusp split):")
            for name, stats in case["plain_sobol_quantities"].items():
                lines.append(
                    f"  {name}: mean={stats['mean']:.17g} "
                    f"std={stats['std']:.9g} SE={stats['SE']:.9g} "
                    f"min={stats['min']:.17g} max={stats['max']:.17g} "
                    f"|R-mean|/SE="
                    f"{stats['abs_rule_minus_mean_over_SE']:.6f}")
        lines.append("")
    lines.append(f"overall={payload['verdict']}")
    return "\n".join(lines) + "\n"


def run_acceptance(*, n_trials=_N_TRIALS, nsamples=_NSAMPLES,
                   qmc_reps=_QMC_REPS):
    """Compute all four distributions; return a JSON-serializable record."""
    if n_trials < 2:
        raise ValueError("n_trials must be at least two to define an SE")
    payload = {
        "config": {
            "N": int(n_trials),
            "nsamples": int(nsamples),
            "qmc_reps": int(qmc_reps),
            "sobol_seed_groups": [
                [trial * qmc_reps, (trial + 1) * qmc_reps - 1]
                for trial in range(n_trials)],
            "criteria": {
                "mean": "abs(rule - mean(draws)) <= 2 * std(draws)/sqrt(N)",
                "underestimate": (
                    "count(abs(draws) > abs(rule))/N <= 0.60; magnitude "
                    "makes less cusp mass the same direction for negative TT"),
            },
        },
        "physical_S_sources": {
            "MoS2_slab_measured": _MOS2_SLAB_S_MEASURED.tolist(),
            "Si_measured": _SI_S_MEASURED.tolist(),
            "bulk_oracle": _BULK_ORACLE_S.tolist(),
        },
        "cases": [],
    }
    all_pass = True
    for case in _cases():
        started = time.monotonic()
        receipt, exact, scalar_certificate = _exact_rule(case)
        if int(case["dimension"]) == 2:
            analytic_region = _slab_disk_terms(receipt, case["S"])
            reference_method = (
                "analytic full inscribed disk plus Sobol polygon-minus-disk "
                "remainder (DEBUG reference)")
        else:
            analytic_region = _bulk_sphere_terms(
                case["geometry"], case["kgrid"], case["S"])
            reference_method = (
                "analytic full inscribed sphere plus Sobol "
                "polyhedron-minus-sphere remainder (DEBUG reference)")
        draws = {name: [] for name in exact}
        plain_draws = (
            {name: [] for name in exact}
            if int(case["dimension"]) == 2 else None)
        for trial in range(n_trials):
            print(
                f"Q0_ACCEPTANCE geometry={case['name']} "
                f"trial={trial + 1}/{n_trials}", flush=True)
            draw, plain_draw = _sobol_draw(
                case, trial=trial, nsamples=nsamples, qmc_reps=qmc_reps,
                analytic_region=analytic_region)
            for name, value in draw.items():
                draws[name].append(value)
            if plain_draw is not None:
                for name, value in plain_draw.items():
                    plain_draws[name].append(value)
        quantities = {
            name: _summary(exact[name]["value"], draws[name])
            for name in exact
        }
        all_pass &= all(item["verdict"] == "PASS"
                        for item in quantities.values())
        record = {
            "dimension": int(case["dimension"]),
            "geometry": case["name"],
            "bvec": np.asarray(case["geometry"].bvec).tolist(),
            "cell_volume": float(case["geometry"].cell_volume),
            "kgrid": list(case["kgrid"]),
            "S": np.asarray(case["S"]).tolist(),
            "reference_method": reference_method,
            "analytic_region": analytic_region[2],
            "rule_certificate": {
                "method": receipt.method,
                "orders": list(receipt.orders),
                "physical_counts": list(receipt.physical_counts),
                "weight_sum_defects": list(receipt.weight_sum_defects),
                "weighted_q_centroids": [
                    list(values) for values in receipt.weighted_q_centroids],
                "scalar_final_error_ratio": float(
                    scalar_certificate.final_error_ratio),
            },
            "exact": exact,
            "quantities": quantities,
        }
        if plain_draws is not None:
            record["plain_sobol_quantities"] = {
                name: _summary(exact[name]["value"], plain_draws[name])
                for name in exact
            }
        record["elapsed_seconds"] = float(time.monotonic() - started)
        payload["cases"].append(record)
    payload["verdict"] = "PASS" if all_pass else "FAIL"
    return payload


def test_exact_rules_are_unbiased_against_finite_variance_sobol_references():
    payload = run_acceptance()
    output = Path.cwd()
    json_path = output / "q0_rule_sobol_acceptance.json"
    text_path = output / "q0_rule_sobol_acceptance.txt"
    # Complex fixture literals above are physically real up to roundoff; JSON
    # carries separate real/imag pairs instead of relying on a nonstandard
    # complex-number encoder.
    sources = payload["physical_S_sources"]
    for key, matrix in tuple(sources.items()):
        values = np.asarray(matrix, dtype=np.complex128)
        sources[key] = {
            "real": values.real.tolist(),
            "imag": values.imag.tolist(),
        }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = _format_report(payload)
    text_path.write_text(report)
    print(report, flush=True)
    failed = [
        f"{case['geometry']}:{name}"
        for case in payload["cases"]
        for name, stats in case["quantities"].items()
        if stats["verdict"] != "PASS"
    ]
    assert not failed, (
        f"exact Gamma-cell acceptance failed for {failed}; full distributions "
        f"are in {json_path} and {text_path}")
