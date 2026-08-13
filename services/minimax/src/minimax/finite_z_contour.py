"""One-sided contour quadrature for finite-frequency polarizability.

For a positive transition energy ``Delta`` and an upper-half-plane sample
``z``, split

    -1/(Delta-z) - 1/(Delta+z)

into a resonant ``+i`` real-time contour and a sign-definite rotated
Laplace contour.  A single positive composite Gauss--Legendre rule serves
all requested ``z`` values on each arm.  This module contains frequency
algebra only: no wavefunctions, JAX, storage, or driver configuration.

The returned error is a dense sampled score ``Im(z)*abs(error)``.  It is an
acceptance diagnostic, not a continuum certificate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ContourArm(NamedTuple):
    tau: np.ndarray
    weights: np.ndarray
    frequency_sign: int
    orders: tuple[int, ...]
    contour: complex
    heldout_scaled_error: float


class FiniteZContourRule(NamedTuple):
    resonant: ContourArm
    antiresonant: ContourArm
    heldout_combined_scaled_error: float
    validation_points: int

    @property
    def executed_nodes(self):
        return int(self.resonant.tau.size + self.antiresonant.tau.size)

    def runtime_arguments(self):
        """Return the existing ``compute_chi0_contour`` frequency arrays."""
        tau = np.concatenate((self.resonant.tau, self.antiresonant.tau))
        weights = np.concatenate(
            (self.resonant.weights, self.antiresonant.weights))
        signs = np.concatenate((
            np.full(self.resonant.tau.size, 1, dtype=np.int8),
            np.full(self.antiresonant.tau.size, -1, dtype=np.int8),
        ))
        return tau, weights, signs


def _gauss_panel_bound(order, half_bandwidth, width, envelope):
    n = float(order)
    a = float(half_bandwidth)
    if n <= a:
        return np.inf
    rho = (n + np.sqrt(n * n - a * a)) / max(a, np.finfo(float).tiny)
    if rho <= 1.0 + 1.0e-14:
        return np.inf
    log_bound = a * (rho - 1.0 / rho) - 2.0 * n * np.log(rho)
    return ((64.0 / 15.0) * (0.5 * float(width)) * float(envelope)
            * np.exp(log_bound) / (1.0 - rho ** (-2.0 * n)))


def _damped_line_rule(decay, frequency, target, *, wavelengths=16.0,
                      max_order=256):
    decay = float(decay)
    frequency = float(frequency)
    target = float(target)
    if not (decay > 0.0 and frequency > 0.0 and 0.0 < target < 1.0):
        raise ValueError("decay, frequency, and target must be positive")
    t_max = np.log(2.0 / target) / decay
    panel_target = float(wavelengths) * 2.0 * np.pi / frequency
    n_panels = max(1, int(np.ceil(t_max / panel_target)))
    width = t_max / n_panels
    half_bandwidth = (frequency + decay) * width / 4.0
    panel_error = 0.5 * (target / decay) / n_panels
    nodes, weights, orders = [], [], []
    for panel in range(n_panels):
        left = panel * width
        envelope = np.exp(-decay * left)
        order = int(np.ceil(half_bandwidth)) + 1
        while (order <= int(max_order)
               and _gauss_panel_bound(
                   order, half_bandwidth, width, envelope) > panel_error):
            order += 1
        if order > int(max_order):
            raise ValueError("positive panel rule exceeded max_order")
        x, w = np.polynomial.legendre.leggauss(order)
        nodes.append(left + 0.5 * width * (x + 1.0))
        weights.append(0.5 * width * w)
        orders.append(order)
    return np.concatenate(nodes), np.concatenate(weights), tuple(orders)


def _endpoint_envelope(interval, z, contour, sign):
    endpoints = np.asarray(interval, dtype=np.float64)
    d = (endpoints[None, :] - z[:, None] if int(sign) == 1
         else endpoints[None, :] + z[:, None])
    transformed = complex(contour) * d
    return (float(np.min(transformed.real)),
            float(np.max(np.abs(transformed.imag))))


def _arm_error(delta, z, tau, weights, sign):
    worst = 0.0
    for value in z:
        d = delta - value if int(sign) == 1 else delta + value
        target = -1.0 / d
        for chunk in np.array_split(np.arange(delta.size), 32):
            fit = -(np.exp(-d[chunk, None] * tau[None, :]) @ weights)
            worst = max(worst, float(
                value.imag * np.max(np.abs(fit - target[chunk]))))
    return worst


def _combined_error(delta, z, resonant, antiresonant):
    worst = 0.0
    tr, wr = resonant
    ta, wa = antiresonant
    for value in z:
        for chunk in np.array_split(np.arange(delta.size), 32):
            values = delta[chunk]
            fit = (
                -(np.exp(-(values[:, None] - value) * tr[None, :]) @ wr)
                -(np.exp(-(values[:, None] + value) * ta[None, :]) @ wa))
            exact = -1.0 / (values - value) - 1.0 / (values + value)
            worst = max(worst, float(
                value.imag * np.max(np.abs(fit - exact))))
    return worst


def build_finite_z_contour(interval, z_values, target, *,
                           design_points=2049, validation_points=32769,
                           angle_step=0.025):
    """Build one shared two-arm rule in the caller's energy units."""
    interval = np.asarray(interval, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.complex128)
    if (interval.shape != (2,) or not 0.0 < interval[0] < interval[1]
            or z.ndim != 1 or z.size == 0 or np.any(z.imag <= 0.0)):
        raise ValueError(
            "require a positive transition interval and upper-half-plane z")
    design = np.linspace(interval[0], interval[1], int(design_points))
    validation = np.linspace(
        interval[0], interval[1], int(validation_points))

    eta_min = float(np.min(z.imag))
    resonant_frequency = float(max(
        np.max(np.abs(interval[0] - z.real)),
        np.max(np.abs(interval[1] - z.real))))
    t, h, orders = _damped_line_rule(
        eta_min, resonant_frequency, target)
    resonant_tau = 1j * t
    resonant_weights = 1j * h

    trials = []
    for theta in np.arange(-1.20, 0.5 * float(angle_step),
                           float(angle_step)):
        contour = np.exp(1j * theta)
        decay, frequency = _endpoint_envelope(interval, z, contour, -1)
        if decay <= 0.0 or frequency <= 0.0:
            continue
        at, ah, aorders = _damped_line_rule(decay, frequency, target)
        tau = contour * at
        weights = contour * ah
        error = _arm_error(design, z, tau, weights, -1)
        if error <= float(target):
            trials.append((tau.size, error, abs(theta), contour, tau,
                           weights, aorders))
    if not trials:
        raise ValueError("no rotated antiresonant contour met the target")
    _, _, _, contour, anti_tau, anti_weights, anti_orders = min(trials)

    resonant_error = _arm_error(
        validation, z, resonant_tau, resonant_weights, 1)
    anti_error = _arm_error(
        validation, z, anti_tau, anti_weights, -1)
    combined_error = _combined_error(
        validation, z, (resonant_tau, resonant_weights),
        (anti_tau, anti_weights))
    return FiniteZContourRule(
        ContourArm(resonant_tau, resonant_weights, 1, orders, 1j,
                   resonant_error),
        ContourArm(anti_tau, anti_weights, -1, anti_orders, contour,
                   anti_error),
        combined_error,
        int(validation.size),
    )


__all__ = ["ContourArm", "FiniteZContourRule", "build_finite_z_contour"]
