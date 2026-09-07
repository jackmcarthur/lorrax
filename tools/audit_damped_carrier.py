#!/usr/bin/env python3
"""Audit the frozen carrier's domain using scalar quadrature, without W arrays.

This is an experiment, not a production synthesis adapter. EVAL owns the
spectral/Stieltjes algebra. No analytic continuation or spectral cutoff is used.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
from scipy.integrate import quad

RY_EV = 13.605693122994


def sha(path):
    """Hash a named artifact without traversing its directory."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(record):
    """Authenticate an existing path/hash record."""
    path = Path(record['path'])
    if sha(path) != record['sha256']:
        raise RuntimeError(f'Artifact changed: {path}')
    return path


def fourier(omega, t, density):
    """Integrate the half-line spectral density at real time [inverse Ry].

    Parameters
    ----------
    omega : complex
        One stable pole [Ry].
    t : float
        Real time [inverse Ry]. No complex-time continuation is defined here.
    density : callable
        EVAL's scalar Lorentzian density [inverse Ry].

    Returns
    -------
    value : complex
        Dimensionless integral of L(w) exp(-i*w*t), w > 0.
    error : float
        Sum of QUADPACK absolute-error estimates, not a rigorous bound.
    """
    # Resolve the narrow peak by splitting at its center. Beyond the peak use
    # QUADPACK's infinite oscillatory tail, not a finite cutoff.
    a, g = omega.real, -omega.imag
    edge = a + max(1., 50*g)
    intervals = ((0., a), (a, edge))
    value, error = 0j, 0.
    for func, multiplier in ((np.cos, 1.), (np.sin, -1j)):
        for lo, hi in intervals:
            v, e = quad(lambda w: density(w, omega)*func(w*t), lo, hi,
                        epsabs=2e-12, epsrel=2e-12, limit=300)
            value += multiplier*v
            error += e
        if t == 0:
            v, e = quad(lambda w: density(w, omega)*func(0.), edge, np.inf,
                        epsabs=2e-12, epsrel=2e-12)
        else:
            v, e = quad(lambda w: density(w, omega), edge, np.inf,
                        weight='cos' if multiplier == 1 else 'sin', wvar=t,
                        epsabs=2e-12, limlst=200)
        value += multiplier*v
        error += e
    return value, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--eval-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run this evidence audit on a compute CPU step')
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.eval_dir))
    kernel = importlib.import_module('damped_kernel')
    residue = importlib.import_module('residue_kernel')
    inputs = json.loads(args.inputs.read_text())
    source = checked(inputs['generalized_sigma_owner'])
    source_text = source.read_text()
    for seam in ('t_exec = pole_sign * fit_t',
                 'fit_t = -np.conj(raw_t) if is_valence else raw_t',
                 'jnp.exp(-1j * (omega - e_ref_b) * tau)'):
        if seam not in source_text:
            raise RuntimeError('Frozen synthesis/time convention changed')
    geometry = json.loads(checked(inputs['geometry']).read_text())
    windows = []
    for window in geometry['schedules']['primary/owner3']:
        rule = inputs['run252']['source_rule_sha256']['447:'+window['name']]
        with np.load(checked(rule)) as data:
            raw = data['times']
        t = np.conj(raw) if window['space'] == 'val' else raw
        windows.append(dict(name=window['name'], nodes=len(t),
            invalid_half_line_nodes=int(np.sum(t.imag > 0)),
            max_imag_time_inverse_ry=float(t.imag.max()),
            pole_interval_ev=(RY_EV*np.array(window['pole_interval_ry'])).tolist(),
            frozen_box_ev=(RY_EV*np.array(window['box_ry'])).tolist(),
            rule=rule))
    if sum(w['nodes'] for w in windows) != 447:
        raise RuntimeError('Schedule changed')
    scalars = []
    for gamma_ev in (.1, .5, 2.):
        omega = (5.-1j*gamma_ev)/RY_EV
        mass = 2*np.arctan(5./gamma_ev)/np.pi
        for t in (0., 1., 10.):
            value, error = fourier(omega, t, kernel.spectral_density)
            bare = np.exp(-1j*omega*t)
            if error > 1e-9 or (t == 0 and abs(value-mass) > 1e-10):
                raise RuntimeError('Scalar Fourier calibration failed')
            scalars.append(dict(center_ev=5., gamma_ev=gamma_ev,
                time_inverse_ry=t, exact=[value.real, value.imag],
                bare_exponential=[bare.real, bare.imag],
                absolute_difference=float(abs(value-bare)),
                quadrature_error_estimate=error, exact_mass=mass))
    # Independent rational-W matrix calculation checks the two required
    # Hermitian channels H=(R+R†)/2 and A=(R-R†)/(2i): rho=H L-A D.
    left = np.array([1.+.3j, .2-.5j])
    right = np.array([.4-.2j, .7+.1j])
    r = left[:, None]*right.conj()[None, :]
    h, anti = (r+r.conj().T)/2, (r-r.conj().T)/(2j)
    omega = (5.-.5j)/RY_EV
    w = 4./RY_EV
    rational = r/(w-omega)-r.conj().T/(w+omega.conjugate())
    spectral = -(rational-rational.conj().T)/(2j*np.pi)
    a, g = omega.real, -omega.imag
    dispersive = ((w-a)/((w-a)**2+g*g)+(w+a)/((w+a)**2+g*g))/np.pi
    assembled = h*kernel.spectral_density(w, omega)-anti*dispersive
    matrix_error = float(np.max(np.abs(spectral-assembled)))
    wrong_error = float(np.max(np.abs(spectral-r*kernel.spectral_density(w, omega))))
    if matrix_error > 1e-12 or wrong_error < 1e-3:
        raise RuntimeError('General-residue planted control failed')
    # Negative width flips L, but leaves D unchanged: use the existing stable
    # scalar owner, without replacing signed gamma by an effective eta.
    z = (3.+.25j)/RY_EV
    negative = (5.+.1j)/RY_EV
    stable = negative.conjugate()
    expected = -kernel.stieltjes(z, stable)
    def signed_integrand(w):
        return -kernel.spectral_density(w, stable)/(z-w)
    parts = [quad(lambda w: part(signed_integrand(w)), 0, np.inf,
                  epsabs=1e-11, epsrel=1e-11, limit=400)[0]
             for part in (np.real, np.imag)]
    negative_error = float(abs(parts[0]+1j*parts[1]-expected))
    if negative_error > 1e-9:
        raise RuntimeError('Negative-width quadrature control failed')
    result = dict(status='BLOCKED_FROZEN_HALF_LINE_DOMAIN',
        job_step=os.environ['SLURM_JOB_ID']+'.'+os.getenv('SLURM_STEP_ID', '?'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha(__file__), inputs_sha256=sha(args.inputs),
        kernel_sha256=sha(kernel.__file__), residue_kernel_sha256=sha(residue.__file__),
        scope='Scalar controls and authenticated frozen metadata; no W model or Sigma sweep',
        windows=windows,
        invalid_half_line_nodes=sum(w['invalid_half_line_nodes'] for w in windows),
        scalar_fourier=scalars,
        general_residue=dict(matrix_identity_max_abs=matrix_error,
                            incorrect_single_R_L_max_abs=wrong_error),
        negative_gamma=dict(gamma_ev=-.1, eta_ev=.25,
                            signed_L_stieltjes_quadrature_error=negative_error,
                            production_allow_anticausal_implemented=False),
        schedule_after=None, boxes_after=None, sigma_discrepancy_mev=None,
        reason='Im(t)>0 grows against an algebraic spectral tail; pole centers do not bound spectral support')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'job_step', 'invalid_half_line_nodes')}))


if __name__ == '__main__':
    main()
