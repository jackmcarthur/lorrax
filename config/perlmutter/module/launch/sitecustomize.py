"""The lorrax module's interpreter start-up hook (first on PYTHONPATH, module shells only).

Stages the venv's ``nvidia/`` tree to node-local storage inside a Slurm job
(:mod:`lorrax_stage_local`).  Never fails the interpreter: an error leaves the
venv paths in place and prints one line.
"""
try:
    import lorrax_stage_local
    lorrax_stage_local.stage()
except Exception as _exc:                                  # noqa: BLE001
    import sys
    print(f"[lorrax stage] skipped: {type(_exc).__name__}: {_exc}", file=sys.stderr)
