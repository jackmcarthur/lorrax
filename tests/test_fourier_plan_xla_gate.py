"""LocalFourierPlan's device leg against its XLA leg on the same device (both device legs).

``common.fourier_plan.LocalFourierPlan`` lowers a plan with a GEMM stage to one custom call on a
leg with a route row (``ffi.fft.FOURIER_PLAN_TARGETS``: CUDA ``lorrax_fourier_plan_mathdx``,
ROCM ``lorrax_fourier_plan_rocm``) and to XLA ops elsewhere.  Gate (``ffi.common.families``
rows ``fourier_plan`` / ``fourier_plan_rocm``, tol 1e-12): for every case the SAME plan object
is run twice on the same device -- ``_default_leg`` forced to ``'ffi'`` (the native route) and
to ``'xla'`` (``jnp.fft`` + ``dot_general``) -- and ``max|y_ffi - y_xla| / max|y_xla| <= TOL``.
Each case also checks the XLA leg against ``np.fft`` on the host (the contract itself).

The mesh form: every device of the run's mesh applies the plan to its own block inside
``shard_map`` (the production geometry: plans are constants of a ``shard_map`` body), so P=4
gates the four GCDs at once.  The GEMM/FFT split per axis is chosen by ``device_kind`` only, so
the cases pass synthetic rows (``_ROWS``) to exercise every stage mix on any device: all-GEMM,
the A100 row (full axes FFT, supported axes GEMM), a GEMM axis between two FFT axes (two FFT
runs, both loop geometries), FFT-only with supports (the remap embed/restrict path, which
production sends to the XLA leg), norms (``ortho``/``forward`` scale on alpha or a remap), signs,
``out_perm`` and non-trailing axes.

Device:  ``python -m tests.test_fourier_plan_xla_gate`` on a CUDA or ROCm mesh (one device, or
the 2x2 P=4 mesh).  Under pytest the device gate skips; the CPU checks of the contract run.
"""
import sys

# runtime before numpy (runtime.blas_env; GATE blas_spin).
import runtime  # noqa: F401,E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TOL = 1e-12

#: Synthetic GEMM_CROSSOVER rows (device_kind -> row), patched in for the gate only.
_ROWS = {
    "GATE all-gemm": (range(1, 1025), range(1, 1025), 1.0, range(1, 1025)),
    "GATE mid-gemm": (range(20, 21), range(20, 21), 1.0, range(20, 21)),
    "GATE fft-only": (range(0), range(0)),
}


def _sphere_idx(n, k):
    """``k`` centred indices of an axis of ``n`` (a sphere's projection), as LocalFourierPlan
    reads them (mod n)."""
    h = k // 2
    return np.arange(-h, k - h) % n


def cases():
    """(name, plan kwargs, input shape (local block), device_kind)."""
    s25 = {a: _sphere_idx(25, 13) for a in (-3, -2, -1)}
    s32 = {a: _sphere_idx(32, 17) for a in (-2, -1)}
    out = []
    for kind in ("GATE all-gemm", "NVIDIA A100"):
        out.append((f"Fe 25^3 sphere->box K=13 sign+ [{kind}]",
                    dict(extents=(25, 25, 25), axes=(-3, -2, -1), sign=+1, norm="backward",
                         in_support=s25), (6, 13, 13, 13), kind))
        out.append((f"Fe 25^3 box->sphere K=13 sign- [{kind}]",
                    dict(extents=(25, 25, 25), axes=(-3, -2, -1), sign=-1, norm="backward",
                         out_support=s25), (6, 25, 25, 25), kind))
    out.append(("mixed: axis0 full FFT, axes1-2 supported GEMM, ortho [NVIDIA A100]",
                dict(extents=(24, 25, 25), axes=(-3, -2, -1), sign=-1, norm="ortho",
                     in_support={-2: _sphere_idx(25, 13), -1: _sphere_idx(25, 13)}),
                (3, 24, 13, 13), "NVIDIA A100"))
    out.append(("mixed: in+out supports, forward norm [NVIDIA A100]",
                dict(extents=(32, 25, 20), axes=(-3, -2, -1), sign=+1, norm="forward",
                     in_support={-3: _sphere_idx(32, 17)}, out_support={-2: _sphere_idx(25, 11)}),
                (2, 17, 25, 20), "NVIDIA A100"))
    out.append(("GEMM between two FFT axes (two FFT runs, both loops) [GATE mid-gemm]",
                dict(extents=(24, 20, 18), axes=(-3, -2, -1), sign=-1, norm="backward",
                     in_support={-3: _sphere_idx(24, 15)}, out_support={-1: _sphere_idx(18, 9)}),
                (4, 15, 20, 18), "GATE mid-gemm"))
    out.append(("GEMM between two FFT axes, batch 1 [GATE mid-gemm]",
                dict(extents=(24, 20, 18), axes=(-3, -2, -1), sign=+1, norm="ortho"),
                (1, 24, 20, 18), "GATE mid-gemm"))
    out.append(("FFT-only with supports (remap embed/restrict), forward [GATE fft-only]",
                dict(extents=(25, 25, 25), axes=(-3, -2, -1), sign=-1, norm="forward",
                     in_support=s25, out_support={-1: _sphere_idx(25, 9)}),
                (3, 13, 13, 13), "GATE fft-only"))
    out.append(("2-D in-support + out_perm, sign- [GATE all-gemm]",
                dict(extents=(32, 32), axes=(-2, -1), sign=-1, norm="backward",
                     in_support=s32, out_perm=(0, 2, 1)), (5, 17, 17), "GATE all-gemm"))
    out.append(("2-D out-support N=48 [NVIDIA A100]",
                dict(extents=(48, 48), axes=(-2, -1), sign=+1, norm="backward",
                     out_support={-2: _sphere_idx(48, 25), -1: _sphere_idx(48, 25)}),
                (7, 48, 48), "NVIDIA A100"))
    out.append(("1-D full GEMM axis, 2 leading dims [GATE all-gemm]",
                dict(extents=(27,), axes=(-1,), sign=-1, norm="ortho"), (3, 5, 27), "GATE all-gemm"))
    out.append(("non-trailing axes (1, 2) of a 4-D block [GATE all-gemm]",
                dict(extents=(16, 20), axes=(1, 2), sign=+1, norm="backward",
                     in_support={1: _sphere_idx(16, 9)}), (2, 9, 20, 3), "GATE all-gemm"))
    return out


def np_reference(x, kw):
    """``R_out · F · E_in · x`` in numpy (the module docstring's contract)."""
    nd = x.ndim
    axes = [a % nd for a in kw["axes"]]
    # Embed axis by axis.
    y = x
    for ax, n in zip(kw["axes"], kw["extents"]):
        p = ax % nd
        if ax in kw.get("in_support", {}):
            idx = np.asarray(kw["in_support"][ax]) % n
            shp = list(y.shape); shp[p] = n
            z = np.zeros(shp, np.complex128)
            zi = [slice(None)] * nd; zi[p] = idx
            z[tuple(zi)] = y
            y = z
    f = np.fft.fftn if kw["sign"] < 0 else np.fft.ifftn
    y = f(y, axes=axes, norm=kw.get("norm") or "backward")
    for ax, n in zip(kw["axes"], kw["extents"]):
        if ax in kw.get("out_support", {}):
            y = np.take(y, np.asarray(kw["out_support"][ax]) % n, axis=ax % nd)
    if kw.get("out_perm") is not None:
        y = np.transpose(y, kw["out_perm"])
    return y


def _relative_max(got, expected):
    """Keep nonfinite outputs and shape failures visible to the gate verdict."""
    if got.shape != expected.shape or not (np.isfinite(got).all() and np.isfinite(expected).all()):
        return float("inf")
    scale = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(got - expected)))
    if not (np.isfinite(scale) and np.isfinite(error)):
        return float("inf")
    return error / scale if scale else error


def gate_cases(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import common.fourier_plan as fp
    from lxkit import device_put_process_local
    axes = tuple(mesh.axis_names)
    spec = P(axes)
    out = {}
    saved_rows, saved_leg = dict(fp.GEMM_CROSSOVER), fp._default_leg
    fp.GEMM_CROSSOVER.update(_ROWS)
    try:
        for i, (name, kw, shape, kind) in enumerate(cases()):
            kw = dict(kw)
            plan = fp.LocalFourierPlan(kw.pop("extents"), kw.pop("axes"), mesh=mesh, device_kind=kind,
                                       **{k: v for k, v in kw.items()})
            kw_full = dict(cases()[i][1])
            # One global input, identical on every process; each device owns a block of axis 0.
            rng = np.random.default_rng(1000 + i)
            gshape = (shape[0] * mesh.devices.size,) + tuple(shape[1:])
            xg = rng.standard_normal(gshape) + 1j * rng.standard_normal(gshape)
            x = device_put_process_local(xg, NamedSharding(mesh, spec))
            res = {}
            for leg in ("ffi", "xla"):
                fp._default_leg = (lambda leg=leg: leg)
                f = jax.jit(jax.shard_map(plan, mesh=mesh, in_specs=spec, out_specs=spec))
                res[leg] = f(x)
            fp._default_leg = saved_leg
            shards = lambda t: sorted(t.addressable_shards, key=lambda s: s.index[0].start or 0)
            worst, worst_np = 0.0, 0.0
            for a, b in zip(shards(res["ffi"]), shards(res["xla"])):
                ref = np_reference(xg[b.index[0]], kw_full)
                a, b = np.asarray(a.data), np.asarray(b.data)
                worst = max(worst, _relative_max(a, b))
                worst_np = max(worst_np, _relative_max(b, ref))
            out[name] = dict(rel=worst, rel_np=worst_np, stages=plan.stages)
    finally:
        fp.GEMM_CROSSOVER.clear()
        fp.GEMM_CROSSOVER.update(saved_rows)
        fp._default_leg = saved_leg
    return out


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    jax.config.update("jax_enable_x64", True)
    from ffi.gate import mesh_ffi_platform
    from ffi.fft import FOURIER_PLAN_TARGETS
    mesh = stack.mesh
    leg = mesh_ffi_platform(mesh)
    if leg not in FOURIER_PLAN_TARGETS:
        print(f"SKIP: needs a mesh with a route row {sorted(FOURIER_PLAN_TARGETS)}, got {leg}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = gate_cases(mesh)
    bad = {k for k, m in out.items() if not (m["rel"] <= TOL and m["rel_np"] <= TOL)}
    for k, m in out.items():
        print(f"r{jax.process_index()} {'FAIL' if k in bad else 'ok  '} {k}  ffi-vs-xla {m['rel']:.1e} "
              f"xla-vs-np {m['rel_np']:.1e}  stages {[(s[0], s[1]) for s in m['stages']]}", flush=True)
    print(f"r{jax.process_index()} FOURIER_PLAN_XLA_GATE {leg} {FOURIER_PLAN_TARGETS[leg]} "
          f"{'PASS' if not bad else 'FAIL'} on {tuple(mesh.devices.shape)}: {len(out) - len(bad)}/{len(out)} "
          f"(tol {TOL:g})", flush=True)
    rc = 1 if bad else 0
    runtime.finalize_process(rc)
    return rc


# ---- pytest: the contract on CPU ----------------------------------------------------------

def test_relative_verdict_rejects_nonfinite_outputs_and_preserves_finite_errors():
    expected = np.asarray([1 + 2j, 3 - 4j])
    assert _relative_max(expected, expected) == 0
    assert _relative_max(expected + .25, expected) == .05
    assert _relative_max(np.zeros(2), np.zeros(2)) == 0
    for nonfinite in (complex(np.nan, 0), complex(np.inf, 0)):
        bad = expected.copy()
        bad[0] = nonfinite
        assert np.isinf(_relative_max(bad, expected))
        assert np.isinf(_relative_max(expected, bad))
    huge = np.asarray([complex(1.7e308, 1.7e308)])
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isinf(_relative_max(-huge, huge))
    assert np.isinf(_relative_max(expected[:1], expected))


if __name__ == "__main__":
    sys.exit(main())
