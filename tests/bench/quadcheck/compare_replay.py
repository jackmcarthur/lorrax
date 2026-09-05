"""Compare identical-input Sigma quadratures; energies and H are in eV.

Fixed DFT-band projectors define the requested principal H blocks. Report
clipped endpoint evaluations separately from actual on-shell samples.
"""
from pathlib import Path
import json
import os
import numpy as np


def stats(x):
    x = np.asarray(x)
    if not x.size:
        return dict(n=0, max_abs_mev=None, rms_mev=None)
    assert np.all(np.isfinite(x))
    a = abs(x)*1000
    return dict(n=int(x.size),max_abs_mev=float(a.max()),
                rms_mev=float(np.sqrt(np.mean(a*a))))


def compare(root, source, *, write=True):
    from gw.eqp_bgw import read_bgw_eqp
    from gw.efermi import solve_mp1_occupations
    from common.units import RYD_TO_EV
    arms = {}
    for name in ('retained','accurate24','dense_contour384'):
        with np.load(root/name/'observables.npz') as f:
            arms[name] = {k:f[k] for k in f.files}
    reference = arms['retained']
    partitions = {'bands5_10':np.arange(4,10), 'bands5_13':np.arange(4,13),
                  'high_tail14_86':np.arange(13,86), 'semicore1_4':np.arange(4),
                  'scissored_all':np.flatnonzero(~reference['in_range'])}
    result = dict(job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                  projector='fixed DFT band principal blocks; 1-based bands',comparisons={})
    eigs = {}
    mus = {}
    for name, arm in arms.items():
        assert np.array_equal(arm['covered'],reference['covered'])
        assert np.array_equal(arm['in_range'],reference['in_range'])
        assert np.array_equal(arm['protected'],reference['protected'])
        eigs[name] = np.linalg.eigvalsh(arm['H_dft_ev'])
        mu,_ = solve_mp1_occupations(eigs[name]/RYD_TO_EV,np.full(512,1/512),9.,0.01,
                                    state_capacity=2.,clamp_tol=1e-8)
        mus[name] = float(mu)*RYD_TO_EV
    result['output_mu_ev'] = mus
    k,_,f4,_ = read_bgw_eqp(str(source/'eqp0_iter0004.dat'))
    idx = np.ravel_multi_index((np.rint(np.mod(k,1)*8).astype(int)%8).T,(8,8,8))
    result['baseline_vs_retained_F4'] = {key:stats((eigs['retained'][idx]-f4)[:,bands]) for key,bands in partitions.items()}
    for first, second in [('accurate24','retained'),('dense_contour384','retained'),('accurate24','dense_contour384')]:
        a,b = arms[first],arms[second]
        ds = a['sigma_on_shell_ev']-b['sigma_on_shell_ev']
        dh = a['H_dft_ev']-b['H_dft_ev']
        pair = {}
        for name,bands in partitions.items():
            covered = reference['covered'][:,bands]
            ha = a['H_dft_ev'][:,bands[:,None],bands]
            hb = b['H_dft_ev'][:,bands[:,None],bands]
            shift = np.linalg.eigvalsh(ha)-np.linalg.eigvalsh(hb)
            pair[name] = dict(sigma_on_shell_covered=stats(ds[:,bands][covered]),
                 sigma_endpoint_clamped_only=stats(ds[:,bands][~covered]),
                 sigma_all_with_endpoint_clamping=stats(ds[:,bands]),
                 sigma_hermitian_on_shell_QP_block=stats((a['sigma_xc_qp_ev']-b['sigma_xc_qp_ev'])[:,bands[:,None],bands]),
                 projected_H=stats(dh[:,bands[:,None],bands]),block_eigenvalue_shift=stats(shift),
                 block_eigenvalue_signed_extrema_mev=[float(shift.min()*1000),float(shift.max()*1000)],
                 full_H_eigenvalue_shift=stats((eigs[first]-eigs[second])[:,bands]),
                 covered_count=int(covered.sum()),total_count=int(covered.size))
        aligned = (eigs[first]-mus[first])-(eigs[second]-mus[second])
        pair['mu_shift_mev'] = (mus[first]-mus[second])*1000
        pair['mu_aligned_bands4_6'] = stats(aligned[:,3:6])
        pair['mu_aligned_per_band4_6'] = {str(i+1):stats(aligned[:,i]) for i in range(3,6)}
        result['comparisons'][first+'_minus_'+second] = pair
    if write:
        (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    import runtime
    runtime.bootstrap(platform='cpu')
    base=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
    compare(base/'runs/Na/14_quadcheck_2026-09-05/replay',
            base/'runs/Na/12_sc_observables_eta05_2026-09-04/arm_E_eta05_rcrop_metric/qsgw')
