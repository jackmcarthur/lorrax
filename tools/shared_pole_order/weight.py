"""Scalar FD denominator weight, authenticated by the Run302 census owner."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import numpy as np

S = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
F = S / 'runs/frequency_integration_sandbox'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Compute receipt required'
    census_path = F / '302_na_campaign_energy_census_20260907/result.json'
    census = json.loads(census_path.read_text())
    reader = Path(census['reader_path'])
    assert sha(reader) == census['reader_sha256']
    spec = importlib.util.spec_from_file_location('canonical268', reader)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    data = owner.load_operand(census['operand_binding'])
    levels = data['levels']
    assert levels.shape == (29, 86)
    ev = census['ry_to_ev']
    # Energies are already relative to mu. The occupation is per-spin FD.
    f = 1 / (1 + np.exp(np.clip(levels / (census['kT_ry'] * ev), -700, 700)))
    energies = np.linspace(-5, 5, 41)
    ew = np.ones(41) / 40
    ew[[0, -1]] *= .5
    omega = np.arange(3601) * .05
    weight = np.zeros_like(omega)
    # Bounded scalar quadratures: no n_mu axes or transition-pair vertices.
    for i, om in enumerate(omega):
        d = energies[:, None, None] - levels[None]
        v = (1-f)[None] / ((d-om)**2+.25**2) + f[None] / ((d+om)**2+.25**2)
        weight[i] = np.sum(ew * np.mean(np.sum(v, axis=2), axis=1))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out/'weight.npz', omega_ev=omega, weight_ev_minus2=weight,
             omega_ry=omega/ev, weight_ry_minus2=weight*ev**2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(omega, weight)
    ax.set(xlabel='Positive boson energy (eV)', ylabel='FD denominator weight (eV⁻²)')
    fig.tight_layout()
    fig.savefig(args.out/'weight.png', dpi=160)
    receipt = dict(jobid=os.environ['SLURM_JOB_ID'], stepid=os.getenv('SLURM_STEP_ID'),
                   census_path=str(census_path), census_sha256=sha(census_path),
                   reader_sha256=sha(reader), script_sha256=sha(__file__),
                   weight_sha256=sha(args.out/'weight.npz'), energy_grid_ev=energies.tolist(),
                   scope='uniform 29 representatives, all 86 FD bands, trapezoid E average',
                   eta_ev=.25, mu_ry=census['mu_ry'], kT_ry=census['kT_ry'],
                   maximum=float(weight.max()), maximum_at_ev=float(omega[weight.argmax()]),
                   finite_even_extension='w(abs(omega)) on [-180,180] eV; zero outside')
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
