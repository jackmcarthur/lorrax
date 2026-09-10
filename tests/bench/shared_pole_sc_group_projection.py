"""Test the canonical magnetic little-group projection in scalar Si SC.

Only the existing local-parent W synthesis seam is wrapped. The symmetry
service owns every operation, phase, transpose and distributed permutation.
Pole factors, spectra, ranks and quadrature requests are unchanged.
"""


def main():
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from gw import gw_jax as driver
    import jax
    import numpy as np
    from gw.mpa import sigma
    from symmetry_maps import project_little_group_operator
    import shared_pole_sc_invariants as observer

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    if jax.process_index() == 0:
        binding = dict(job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                       scope='canonical magnetic little-group average in local-parent Si W synthesis',
                       files={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__),
                                  source/'src/gw/mpa/sigma.py',
                                  source/'services/symmetry_maps/src/symmetry_maps/qgrid_trs.py')})
        (args.output/'group_projection_binding.json').write_text(json.dumps(binding, indent=2)+'\n')

    original = sigma._shared_pole_panel_unfold
    original_routed = sigma._shared_pole_routed_synthesis

    def panel(meta, header, q_span, *, mesh_xy, tables=None):
        if tables is None:
            tables = sigma._shared_pole_panel_tables(meta, header, q_span, mesh_xy=mesh_xy)
        rows, unfold = original(meta, header, q_span, mesh_xy=mesh_xy, tables=tables)
        lo, hi = map(int, q_span)
        metadata = dict(
            q_full_idx=np.asarray(header['q_irr_full_idx'])[lo:hi],
            q_irr_frac=tables['q_frac'],
            sym_mats_k=np.asarray(header['operations']['rotation']),
            sym_perm=tables['packed_perm'], L_table=tables['wraps'],
            active_symmetry_rows=np.asarray(header['operations']['authorized_rows']),
            kgrid=tuple(header['grid']), n_sym_spatial=tables['n_sym_spatial'],
            mesh=mesh_xy, active_mask=meta.mu_basis.active_mask)

        @jax.jit
        def projected(plus, transposed):
            plus, transposed = project_little_group_operator(
                plus, transposed_partner=transposed, **metadata)
            return unfold(plus, transposed)

        return rows, projected

    def refuse_routed(*args, **kwargs):
        raise ValueError('This Si diagnostic requires local-parent synthesis; nonlocal service algebra has its own gate')

    sigma._shared_pole_panel_unfold = panel
    sigma._shared_pole_routed_synthesis = refuse_routed
    try:
        observer.main()
    finally:
        sigma._shared_pole_panel_unfold = original
        sigma._shared_pole_routed_synthesis = original_routed


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
