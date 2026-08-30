"""P=4 falsifier for the resident polar FFT-field symmetry projection.

The launch must be one rank per GPU.  With cache cold and an XLA dump enabled,
the service module should retain ``P(None,None,None,None)``, contain no host
callback, and require no collective because this small real-space field is
replicated by design.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax                                                        # noqa: E402
import numpy as np                                                 # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P          # noqa: E402

from common.collectives import resolve_mesh                         # noqa: E402
from ffi import _services                                           # noqa: E402

_services.ensure_on_path()
from symmetry_maps import project_polar_fft_field                   # noqa: E402


RTOL = 3.0e-13


def _symmetry(trs_allowed):
    spatial = np.stack([
        np.eye(3, dtype=np.int32),
        np.diag([1, 1, -1]).astype(np.int32),
    ])
    rotations = spatial.astype(np.float64)
    return SimpleNamespace(
        sym_matrices=spatial,
        translations=np.asarray(
            [[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]], dtype=np.float64),
        R_cart_forward=np.concatenate([rotations, -rotations], axis=0),
        trs_allowed=bool(trs_allowed),
    )


def _put_replicated(value, mesh):
    sharding = NamedSharding(mesh, P(None, None, None, None))
    return jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index])


def _relative(got, reference):
    scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    return float(np.max(np.abs(np.asarray(got) - reference))) / scale


def main():
    if jax.process_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "polar projection gate requires P=4 with one rank per GPU; got "
            f"process_count={jax.process_count()}, device_count={jax.device_count()}.")
    mesh = resolve_mesh()
    if tuple(mesh.axis_names) != ("x", "y"):
        raise RuntimeError(f"expected canonical xy mesh, got {mesh.axis_names}")

    rng = np.random.default_rng(20260830)
    real = rng.standard_normal((3, 8, 8, 4))
    complex_value = real + 1j * rng.standard_normal(real.shape)
    failures = []
    for label, value, trs_allowed in (
            ("real-trs-on", real, True),
            ("complex-trs-on", complex_value, True),
            ("complex-trs-off", complex_value, False)):
        sym = _symmetry(trs_allowed)
        reference = project_polar_fft_field(value, sym)
        resident = _put_replicated(value, mesh)
        got = project_polar_fft_field(resident, sym)
        got.field.block_until_ready()
        rel = _relative(got.field, np.asarray(reference.field))
        spec = got.field.sharding.spec
        print(
            f"[polar-projection] rank={jax.process_index()} {label}: "
            f"relative={rel:.3e} residual={got.relative_residual:.3e} "
            f"tolerance={got.relative_residual_tolerance:.3e} spec={spec}")
        if rel > RTOL:
            failures.append(f"{label} host/device relative {rel:.3e}")
        if got.relative_residual > got.relative_residual_tolerance:
            failures.append(f"{label} covariance receipt")
        if spec != P(None, None, None, None):
            failures.append(f"{label} layout {spec}")

        # The second call must hit the same Python jit-cache entry and the
        # same compiled executable; it is the SC-iteration reuse contract.
        warm = project_polar_fft_field(resident, sym)
        warm.field.block_until_ready()
        warm_rel = _relative(warm.field, np.asarray(reference.field))
        if warm_rel > RTOL:
            failures.append(f"{label} warm relative {warm_rel:.3e}")

    if failures:
        raise AssertionError("; ".join(failures))
    print("[polar-projection] VERDICT PASS")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
    finalize_process(rc)
