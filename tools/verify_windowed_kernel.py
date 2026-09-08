"""Compute-node scalar verification of the physical finite-window kernel."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy.integrate import quad
from windowed_damped_kernel import W_CERT_RY, fourier_components, mass, stieltjes_components


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or args.output.exists():
        raise RuntimeError('Compute-only, new output required')
    sys.path.insert(0, str(args.eval_dir))
    from damped_kernel import spectral_density
    from residue_kernel import width_diagnostic
    inputs = json.loads(args.inputs.read_text())
    times = []
    hashes = {}
    for name, rec in inputs['run252']['source_rule_sha256'].items():
        if not name.startswith('447:'):
            continue
        path = Path(rec['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == rec['sha256']
        hashes[str(path)] = digest
        with np.load(path) as f:
            raw = f['times']
        values = np.conj(raw) if '_val:' in name else raw
        times.extend((name, i, complex(t)) for i, t in enumerate(values))
    assert len(times) == 447
    # Include the four largest positive imaginary nodes, the four most
    # negative nodes, and a deterministic spread across the full schedule.
    imag = np.array([x[2].imag for x in times])
    indices = list(np.argsort(imag)[-4:])+list(np.argsort(imag)[:4])
    for i in np.linspace(0, 446, 40, dtype=int):
        if i not in indices:
            indices.append(int(i))
        if len(indices) == 20:
            break
    records = []
    for center_ev in (.5, 5., 140.):
        for gamma_ev in (.1, .2, .5, 2.):
            omega = (center_ev-1j*gamma_ev)/13.605693122994
            # Tabulate every node; quadrature checks the selected twenty.
            values, dispersive = fourier_components(np.array([x[2] for x in times]), omega)
            for i in indices:
                name, slot, t = times[i]
                split = sorted(set([0., min(W_CERT_RY, 1/max(1., abs(t))),
                                    omega.real, W_CERT_RY]))
                exact = 0j
                error = 0.
                for lo, hi in zip(split[:-1], split[1:]):
                    v,e = quad(lambda w: spectral_density(w, omega)*np.exp(-1j*w*t),
                        lo,hi,complex_func=True,epsabs=1e-18,epsrel=2e-12,limit=500)
                    exact += v
                    error += abs(e)
                relative = abs(values[i]-exact)/max(abs(exact), 1e-300)
                records.append(dict(center_ev=center_ev,gamma_ev=gamma_ev,window=name,
                    node=slot,time=[t.real,t.imag],relative_error=float(relative),
                    exact=[exact.real,exact.imag],e1=[values[i].real,values[i].imag],
                    quad_error_estimate=float(error)))
    worst = max(x['relative_error'] for x in records)
    assert worst < 1e-8, worst
    scalar_rows = []
    for gamma_ev in (-.1,.2,2.):
        omega=(5.-1j*gamma_ev)/13.605693122994
        width_diagnostic(omega,allow_anticausal=gamma_ev<0)
        z=(3.+.25j)/13.605693122994
        fl,fd=stieltjes_components(z,omega)
        a,g=omega.real,abs(omega.imag)
        def density(w):
            l=np.sign(gamma_ev)*spectral_density(w,complex(a,-g))
            d=((w-a)/((w-a)**2+g*g)+(w+a)/((w+a)**2+g*g))/np.pi
            return l-.3*d
        v=quad(lambda w:density(w)/(z-w),0,W_CERT_RY,complex_func=True,
               points=[a,z.real],epsabs=1e-12,epsrel=1e-12,limit=400)[0]
        err=float(abs(fl+.3*fd-v)/abs(v));assert err<1e-10
        scalar_rows.append(dict(gamma_ev=gamma_ev,relative_error=err))
    refusals=[]
    for gamma,allow in ((-.25,False),(-.3,True),(-.1,False)):
        try:
            fourier_components(1.,(5.-1j*gamma)/13.605693122994,allow_anticausal=allow)
        except ValueError as e:
            refusals.append(dict(gamma_ev=gamma,allow_anticausal=allow,message=str(e)))
        else:
            raise AssertionError('Missing width refusal')
    tail=[]
    for a in (.5,5.,50.,140.,149.):
        for g in (.1,.2,.5,2.):
            omega=(a-1j*g)/13.605693122994
            half=2*np.arctan(a/g)/np.pi
            retained=float(mass(omega))
            tail.append(dict(center_ev=a,gamma_ev=g,lost_mass=half-retained,
                retained_mass=retained,near_edge_estimate=g/(np.pi*(149.7645-a)),
                two_mirror_small_width_estimate=g/np.pi*(1/(149.7645-a)-1/(149.7645+a))))
    result=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.getenv('SLURM_STEP_ID','?'),
        normalization='none: unnormalized physical spectral restriction',W_cert_ev=149.7645,
        unique_nodes_checked=20,mode_cases=12,total_quadratures=len(records),
        worst_relative_error=worst,records=records,stieltjes=scalar_rows,width_refusals=refusals,
        tail_mass=tail,rule_sha256=hashes,
        kernel_sha256=hashlib.sha256(Path(__file__).with_name('windowed_damped_kernel.py').read_bytes()).hexdigest(),
        scope='scalar CPU finite-window kernels, not Sigma or GPU execution')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','job_step','worst_relative_error','total_quadratures')}))


if __name__ == '__main__':
    main()
