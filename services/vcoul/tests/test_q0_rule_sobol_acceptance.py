"""Sobol-distribution acceptance for the exact Gamma-cell rules.

This is a compute-node certificate, not an ordinary unit test.  It compares
the production exact Wigner--Seitz polygon/polyhedron rules with independent
scrambled-Sobol estimates of the same cell integrals.  Each reported draw is
the mean of ten ``2**18``-point Sobol replicates, matching the historical
production estimator.  The 3-D estimator analytically removes the inscribed
sphere before sampling, including the screened singular term, so every
sampled remainder has finite variance.

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
                bulk_sphere=None):
    dimension = int(case["dimension"])
    analytic_terms = {name: 0.0 for name in (
        "bare_v", "screened_w", "tt_trace", "tt_xx", "tt_yy", "tt_zz")}
    q0sph2 = None
    if dimension == 3:
        if bulk_sphere is None:
            bulk_sphere = _bulk_sphere_terms(
                case["geometry"], case["kgrid"], case["S"])
        q0sph2, analytic_terms, _ = bulk_sphere

    replicate_values = []
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
        if dimension == 3:
            outside = q2 > q0sph2
            reduced = {
                name: float(np.sum(np.where(outside, value, 0.0))
                            / q.shape[0] + analytic_terms[name])
                for name, value in values.items()
            }
        else:
            reduced = {
                name: float(np.mean(value)) for name, value in values.items()}
        replicate_values.append(reduced)
    return {
        name: float(np.mean([values[name] for values in replicate_values]))
        for name in replicate_values[0]
    }


def _summary(rule, draws):
    samples = np.asarray(draws, dtype=np.float64)
    n = int(samples.size)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    se = std / np.sqrt(n)
    z = abs(float(rule) - mean) / se if se > 0.0 else np.inf
    above = int(np.count_nonzero(samples > rule))
    below = int(np.count_nonzero(samples < rule))
    mean_pass = bool(abs(float(rule) - mean) <= 2.0 * se)
    sign_pass = bool(above / n <= 0.60)
    return {
        "N": n,
        "rule": float(rule),
        "mean": mean,
        "std": std,
        "SE": float(se),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "empirical_cdf_below_rule": below / n,
        "count_draws_above_rule": above,
        "fraction_draws_above_rule": above / n,
        "abs_rule_minus_mean_over_SE": float(z),
        "mean_within_2SE": mean_pass,
        "underestimate_sign_test": sign_pass,
        "verdict": "PASS" if mean_pass and sign_pass else "FAIL",
        "sorted_draws": [float(value) for value in np.sort(samples)],
    }


def _format_report(payload):
    lines = [
        "Exact Gamma-cell rule vs independent Sobol distributions",
        (f"N={payload['config']['N']}, "
         f"nsamples={payload['config']['nsamples']}, "
         f"qmc_reps={payload['config']['qmc_reps']}"),
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
        if "analytic_sphere" in case:
            lines.append(f"analytic_sphere={case['analytic_sphere']}")
        for name, stats in case["quantities"].items():
            ladder = case["exact"][name]
            lines.append(
                f"{name}: N={stats['N']} R={stats['rule']:.17g} "
                f"mean={stats['mean']:.17g} std={stats['std']:.9g} "
                f"SE={stats['SE']:.9g} min={stats['min']:.17g} "
                f"max={stats['max']:.17g} "
                f"CDF(R)={stats['empirical_cdf_below_rule']:.6f} "
                f"above={stats['count_draws_above_rule']}/{stats['N']} "
                f"|R-mean|/SE={stats['abs_rule_minus_mean_over_SE']:.6f} "
                f"ladder={ladder['ladder_values']} "
                f"ratios={ladder['ladder_ratios']} {stats['verdict']}")
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
                "underestimate": "count(draws > rule)/N <= 0.60",
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
        bulk_sphere = (
            _bulk_sphere_terms(
                case["geometry"], case["kgrid"], case["S"])
            if int(case["dimension"]) == 3 else None)
        draws = {name: [] for name in exact}
        for trial in range(n_trials):
            print(
                f"Q0_ACCEPTANCE geometry={case['name']} "
                f"trial={trial + 1}/{n_trials}", flush=True)
            draw = _sobol_draw(
                case, trial=trial, nsamples=nsamples, qmc_reps=qmc_reps,
                bulk_sphere=bulk_sphere)
            for name, value in draw.items():
                draws[name].append(value)
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
        if int(case["dimension"]) == 3:
            _, _, sphere_certificate = bulk_sphere
            record["analytic_sphere"] = sphere_certificate
        record["elapsed_seconds"] = float(time.monotonic() - started)
        payload["cases"].append(record)
    payload["verdict"] = "PASS" if all_pass else "FAIL"
    return payload


def test_exact_rules_are_unbiased_against_production_sobol_distributions():
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
