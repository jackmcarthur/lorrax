"""P4 admission and layout parity on fresh scalar, SOC and bispinor bundles."""
import argparse
import json
from pathlib import Path


def main():
    from runtime import initialize_communicator_stack
    runtime = initialize_communicator_stack(platform='gpu')
    import jax
    import jax.numpy as jnp
    from file_io.restart_bundle import (_find_restart_file, read_metadata,
        load_restart_state_from_h5)

    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    assert runtime.mesh.size == 4
    rows = []
    for name in ('01_scalar_true', '02_soc_true', '03_mos2_true',
                 '04_scalar_false', '05_mos2_false', '06_soc_ns2_fixture'):
        filename = _find_restart_file(str(args.run_root / name / 'cohsex.in'))
        metadata = read_metadata(filename)
        face = load_restart_state_from_h5(filename, runtime.mesh, low_mem_bands=True)
        axis = load_restart_state_from_h5(filename, runtime.mesh, low_mem_bands=False)
        errors = {}
        for field in ('V_qmunu', 'enk_full', 'psi_nmu_parent', 'psi_mun_parent',
                      'psi_nmu_parent_transverse', 'psi_mun_parent_transverse'):
            a, b = getattr(face, field), getattr(axis, field)
            if a is None:
                assert b is None
                continue
            error = float(jnp.max(jnp.abs(a-b)))
            assert error == 0., (name, field, error)
            errors[field] = error
        rows.append(dict(bundle=name, family_shapes=metadata['family_shapes'],
                         absolute_errors=errors))
        if jax.process_index() == 0:
            print('RESTART_LAYOUT_PASS', name, errors, flush=True)
        del face, axis
    if jax.process_index() == 0:
        Path('reader_layout_parity.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
