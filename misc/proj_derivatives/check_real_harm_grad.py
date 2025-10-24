#!/usr/bin/env python3
"""
Self-contained check of qe_real_sph_harmonics_with_grad against finite
differences of qe_real_sph_harmonics. No pytest needed.

Runs for l=1..3 on a set of random directions (avoids poles), compares
analytic angular components to finite differences and prints max errors.

Exit code is nonzero if tolerance is exceeded.
"""

import sys
import math
import argparse
import numpy as np

from isdf.psp.build_projectors_qe import (
    qe_real_sph_harmonics,
    qe_real_sph_harmonics_with_grad,
)


def cart_from_angles(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi);   sp = np.sin(phi)
    x = st * cp; y = st * sp; z = ct
    return np.stack([x, y, z], axis=-1)


def spherical_frame(theta: np.ndarray, phi: np.ndarray):
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi);   sp = np.sin(phi)
    e_r = np.stack([st * cp, st * sp, ct], axis=-1)
    e_t = np.stack([ct * cp, ct * sp, -st], axis=-1)
    e_p = np.stack([-sp, cp, np.zeros_like(st)], axis=-1)
    return e_r, e_t, e_p


def check_once(l: int, n: int, seed: int, h: float, atol: float, rtol: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed + l)
    theta = rng.uniform(low=0.2, high=math.pi - 0.2, size=(n,))
    phi = rng.uniform(low=-math.pi, high=math.pi, size=(n,))
    vecs = cart_from_angles(theta, phi)

    Y, grad_cart = qe_real_sph_harmonics_with_grad(l, vecs)
    _, e_t, e_p = spherical_frame(theta, phi)
    st = np.maximum(np.sin(theta), 1e-9)

    dth_est = (e_t[:, 0][None, :] * grad_cart[0]
               + e_t[:, 1][None, :] * grad_cart[1]
               + e_t[:, 2][None, :] * grad_cart[2])
    dph_est = st[None, :] * (
        e_p[:, 0][None, :] * grad_cart[0]
        + e_p[:, 1][None, :] * grad_cart[1]
        + e_p[:, 2][None, :] * grad_cart[2]
    )

    # FD on theta, phi
    Yp_th = qe_real_sph_harmonics(l, cart_from_angles(theta + h, phi))
    Ym_th = qe_real_sph_harmonics(l, cart_from_angles(theta - h, phi))
    dth_fd = (Yp_th - Ym_th) / (2 * h)

    Yp_ph = qe_real_sph_harmonics(l, cart_from_angles(theta, phi + h))
    Ym_ph = qe_real_sph_harmonics(l, cart_from_angles(theta, phi - h))
    dph_fd = (Yp_ph - Ym_ph) / (2 * h)

    # Errors
    err_th = np.max(np.abs(dth_est - dth_fd) / np.maximum(atol, np.abs(dth_fd) * rtol))
    err_ph = np.max(np.abs(dph_est - dph_fd) / np.maximum(atol, np.abs(dph_fd) * rtol))
    return float(err_th), float(err_ph)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=24, help='number of random directions')
    p.add_argument('--seed', type=int, default=123, help='rng seed')
    p.add_argument('--h', type=float, default=1e-6, help='finite difference step')
    p.add_argument('--atol', type=float, default=5e-6, help='abs tolerance')
    p.add_argument('--rtol', type=float, default=1e-5, help='rel tolerance')
    args = p.parse_args(argv)

    ok = True
    for l in (1, 2, 3):
        eth, eph = check_once(l, args.n, args.seed, args.h, args.atol, args.rtol)
        print(f'l={l}: max normed error dtheta={eth:.3e}, dphi={eph:.3e}')
        if not (eth < 1.0 and eph < 1.0):
            ok = False

    if not ok:
        print('Gradient check FAILED (errors exceed tolerances)')
        return 2
    print('Gradient check PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())

