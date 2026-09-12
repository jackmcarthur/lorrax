"""Use one hash-pinned historical MPA head in an otherwise current SC audit.

The historical module supplies its original builder and kernel globals.
The current driver, support recipe, Sigma realization, and invariant observer
remain their canonical owners. This is an explicitly labeled A/B harness.
"""


def main():
    import argparse
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--legacy-head', type=Path, required=True)
    parser.add_argument('--legacy-head-sha256', required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    # Runtime initializes before either head imports JAX.
    from gw import gw_jax as driver
    from gw import shared_pole_head as current
    import jax
    import shared_pole_sc_invariants as observer

    legacy_path = options.legacy_head.resolve()
    digest = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    if digest != options.legacy_head_sha256:
        raise ValueError('historical MPA head source hash does not match binding')
    module_name = 'gw._shared_pole_old_head_audit'
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = legacy
    spec.loader.exec_module(legacy)

    parsed = argparse.ArgumentParser(add_help=False)
    parsed.add_argument('--output', type=Path, required=True)
    output, _ = parsed.parse_known_args(remaining)
    if jax.process_index() == 0:
        output.output.mkdir(parents=True, exist_ok=True)
        binding = dict(
            job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
            legacy_head=str(legacy_path), legacy_head_sha256=digest,
            current_driver=str(Path(driver.__file__).resolve()),
            scope='Historical MPA head builder only; current production W/Sigma realization and support policy',
        )
        (output.output/'legacy_head_binding.json').write_text(
            json.dumps(binding, indent=2)+'\n')
    saved_builder, saved_plan = current.build_shared_pole_head, current.shared_pole_head_plan
    current.build_shared_pole_head = legacy.build_shared_pole_head
    current.shared_pole_head_plan = legacy.shared_pole_head_plan
    try:
        observer.main()
    finally:
        current.build_shared_pole_head = saved_builder
        current.shared_pole_head_plan = saved_plan
        sys.modules.pop(module_name, None)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
