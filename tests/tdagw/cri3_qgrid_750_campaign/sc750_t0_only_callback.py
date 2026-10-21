"""Verification-only native W/Sigma t0 capture; never matched TD admission.

Reuse the ordinary callback unchanged, then stop after its existing collective
snapshot/operator/boundary writes. No integrator, refresh or science gate is
weakened. This intentionally does not execute a time step.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time


class _T0Captured(RuntimeError):
    """Private all-rank stop after the ordinary t0 capture returned."""


def _ordinary_callback():
    path = Path(__file__).with_name('sc750_td_callback.py')
    spec = importlib.util.spec_from_file_location('_sc750_ordinary_t0_owner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_receipt(out, source, started):
    """Root-only small-file authentication after collective owner capture."""
    if (out/'complete.json').exists():
        raise ValueError('GATE sc750_t0_only: full TD completion must not be published')
    names = ('reference.json', 'snapshot_0000.npz', 'snapshot_0000.json',
             'snapshot_0000_boundary.json')
    hashes = {}
    for name in names:
        path = out/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('GATE sc750_t0_only: missing committed t0 artifact '+name)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024**2), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    reference = json.loads((out/'reference.json').read_text())
    if reference.get('actual_active') != [100, 144] or reference.get('virtual_MH_bands') != 750:
        raise ValueError('GATE sc750_t0_only: actual44/full750 source witness required')
    record = dict(status='T0_ONLY_NATIVE_OPERATORS_CAPTURED_NOT_TD_ADMISSION',
        purpose='t0_only', t0_operators_captured=True, time_steps_executed=0,
        completed=False, zero_gate=False, matched_BC_admission=False,
        refresh_admission=False, trajectory_admission=False,
        configured_dt_certificate_exercised=False,
        actual_active=[100, 144], virtual_MH_bands=750,
        source_frame=source['frame']['frame'],
        execution_revision=os.environ.get('LEG_HEAD'),
        source_archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        callback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256=hashes, wall_s=time.perf_counter()-started,
        scope='Ordinary current-W and full-rho one-sided Sigma owner returned and committed t0; '
              'raw number/boundary diagnostics only, no propagation or Pauli-invariance guarantee')
    with (out/'T0_ONLY_DIAGNOSTIC.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run_td_probe(source, *, out, deck_path, budget_s=900., storage_gib=64., diagnostic_role=None):
    """Capture ordinary44/750 t0, preserving every underlying refusal."""
    if diagnostic_role is not None:
        raise ValueError('GATE sc750_t0_only: ordinary current-W only, not substituted source W')
    from common.collectives import rank0_transaction
    from tdagw import stages, snapshot_runtime as sr
    owner = _ordinary_callback()
    started, out = time.perf_counter(), Path(out).resolve()
    original_propagate, original_snapshot = stages.propagate, sr.snapshot_collision

    def propagate(*args, **kwargs):
        observed_snapshot = sr.snapshot_collision
        def stop_after_capture(*call_args, **call_kwargs):
            # This is the ORDINARY callback's wrapper, not just the raw
            # snapshot owner: it returns only after all three t0 writes.
            observed_snapshot(*call_args, **call_kwargs)
            raise _T0Captured('T0_ONLY_CAPTURED_AFTER_COMMITTED_OPERATORS')
        sr.snapshot_collision = stop_after_capture
        try:
            return original_propagate(*args, **kwargs)
        finally:
            sr.snapshot_collision = observed_snapshot

    stages.propagate = propagate
    captured = False
    try:
        try:
            owner.run_td_probe(source, out=out, deck_path=deck_path,
                               budget_s=budget_s, storage_gib=storage_gib)
        except _T0Captured:
            captured = True
    finally:
        stages.propagate, sr.snapshot_collision = original_propagate, original_snapshot
    if not captured:
        raise ValueError('GATE sc750_t0_only: ordinary snapshot capture was not reached')
    rank0_transaction(out/'T0_ONLY_DIAGNOSTIC.json', stage='sc750_t0_only/receipt',
                      write=lambda: _write_receipt(out, source, started))
