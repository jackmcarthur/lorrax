"""Isolate whole-value conjugation at complex shared-pole head samples.

This compares the production Gamma evaluator followed by the two existing
fixed-q actions on a planted real PSD residue. It does not run SC or fit a
physical head, so its errors are algebraic counterexamples, not QP shifts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from gw import gw_jax as driver
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_head import _gamma_body
    from symmetry_maps import build_qgrid_trs_policy

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert int(mesh.shape["x"]) == int(mesh.shape["y"]) == 2
    receipt = json.loads(args.head_receipt.read_text())
    actual_z = [complex(row["real"], row["imag"]) for row in receipt["sample_z"]]
    policy = build_qgrid_trs_policy(
        trs_measured=True, irr_idx_q=np.array([0]), sym_idx_q=np.array([0]),
        q_irr_full_idx=np.array([0]), kgrid=(1, 1, 1), n_sym_spatial=1,
        active_symmetry_rows=np.array([0, 1]))
    def put(value, spec):
        value = np.asarray(value)
        sh = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(value.shape, sh, lambda index: value[index])
    face = P(None, "x", "y")
    b = put(np.eye(4, dtype=complex)[None], face)
    poles = put(np.ones((1, 4)), P())
    counts = put(np.array([4]), P())
    v = put(2*np.eye(4, dtype=complex)[None], face)
    evaluate = _gamma_body(mesh)
    rows = []
    for index, z in enumerate([1+.5j]+actual_z):
        raw = evaluate(put(np.array(z*z), P()), b, poles, counts, v)
        old, _ = policy.project_fixed_q(raw, np.array([0]), measure=False)
        paired, _ = policy.project_fixed_q(
            raw, np.array([0]), transposed_partner=raw.swapaxes(-1, -2), measure=False)
        scalar = 2+1/(z*z-1)
        oracle = put(scalar*np.eye(4, dtype=complex)[None], face)
        raw_error = float(jnp.max(jnp.abs(raw-oracle)))
        old_error = float(jnp.max(jnp.abs(old-oracle)))
        paired_error = float(jnp.max(jnp.abs(paired-oracle)))
        assert raw_error < 2e-12 and paired_error < 2e-12
        if z.real != 0 and z.imag != 0:
            assert old_error > 1e-6
        else:
            assert old_error < 2e-12
        rows.append(dict(index=index-1, z=dict(real=z.real, imag=z.imag),
                         expected=dict(real=scalar.real, imag=scalar.imag),
                         raw_error=raw_error, whole_value_error=old_error,
                         pair_transpose_error=paired_error))
    report = dict(status="PASS", job_step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                  scope=__doc__, input_head_receipt=str(args.head_receipt),
                  input_sha256=hashlib.sha256(args.head_receipt.read_bytes()).hexdigest(),
                  actual_complex_strip_samples=sum(z.real != 0 and z.imag != 0 for z in actual_z),
                  actual_sample_count=len(actual_z), counterexamples=rows)
    if jax.process_index() == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
