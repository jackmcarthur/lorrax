"""CPU FFTW BSE convolution against the independent XLA route on one/P4 CPUs.

Run as a script on compute nodes, with the CPU MPI prelude for P4.  Checks
ragged chunks, unit/odd/3-D k grids, spinor and batch broadcasts, both output
layouts, FFT normalization, donation, and the compiled memory allocation.
"""
import json
import os
import sys
import time


def main():
    import runtime
    runtime.bootstrap()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import resolve_mesh, warm_mesh_cliques
    from ffi import fft as F

    if jax.default_backend() != "cpu":
        raise RuntimeError("this gate requires the CPU plan backend")
    mesh = resolve_mesh()
    warm_mesh_cliques(mesh)
    F.require_kconv(mesh)
    rank = jax.process_index()
    rng = np.random.default_rng(20261022 + rank)
    xs = P(None, "x", "y", None, None, None)
    ks = P("x", "y", None)
    tls, vls = P(None, None, "x", None, "y"), P(None, "x", "y")
    records = []
    started = time.perf_counter()

    def put(a, spec):
        shape = list(a.shape)
        for i, axis in enumerate(spec):
            for name in (() if axis is None else axis if isinstance(axis, tuple) else (axis,)):
                shape[i] *= int(mesh.shape[name])
        return jax.make_array_from_callback(tuple(shape), NamedSharding(mesh, spec), lambda _: a)

    def random(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    for kg in ((1, 1, 1), (3, 5, 1), (2, 3, 4), (7, 11, 1), (16, 12, 1)):
        nk = int(np.prod(kg))
        for ns in (1, 2):
            for batch in (1, 4):
                local_shape = (ns, 3, 5, ns, batch, nk)
                x = put(random(local_shape), xs)
                k = put(random((3, 5, nk)), ks)
                for layout in (0, 1):
                    for norm in (None, "ortho", "forward"):
                        kwargs = dict(norm=norm, mult=-.37, out_layout=layout)
                        f = jax.jit(F.make_kconv_kminor(mesh, kg, xs, ks, **kwargs))
                        with F.xla_reference():
                            ref = jax.jit(F.make_kconv_kminor(mesh, kg, xs, ks, **kwargs))
                        hlo = f.lower(x, k).as_text()
                        assert F.HOST_KCONV_KMINOR_TARGET in hlo, "CPU call did not take its native target"
                        native = f(x, k).block_until_ready()
                        expected = ref(x, k).block_until_ready()
                        actual_local = np.asarray(native.addressable_shards[0].data)
                        expected_local = np.asarray(expected.addressable_shards[0].data)
                        error = float(np.max(np.abs(actual_local - expected_local)) /
                                      max(float(np.max(np.abs(expected_local))), 1e-300))
                        assert error < 1e-12, (kg, ns, batch, layout, norm, error)
                        records.append(dict(kgrid=kg, ns=ns, batch=batch, layout=layout,
                                            norm=norm, relative_error=error))
                t = put(random((nk, ns, 3, ns * batch, 5)), tls)
                v = put(random((nk, 3, 5)), vls)
                for norm in (None, "ortho", "forward"):
                    local = F.make_local_kconv_klead(mesh, kg, norm=norm, mult=-.37)
                    with F.xla_reference():
                        ref_local = F.make_local_kconv_klead(mesh, kg, norm=norm, mult=-.37)
                    wrap = lambda fn: jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=(tls, vls),
                                                          out_specs=tls, check_vma=False))
                    f, ref = wrap(local), wrap(ref_local)
                    assert F.HOST_KCONV_KLEAD_TARGET in f.lower(t, v).as_text()
                    native = f(t, v).block_until_ready()
                    expected = ref(t, v).block_until_ready()
                    av = np.asarray(native.addressable_shards[0].data)
                    ev = np.asarray(expected.addressable_shards[0].data)
                    error = float(np.max(np.abs(av - ev)) / max(float(np.max(np.abs(ev))), 1e-300))
                    assert error < 1e-12, ("klead", kg, ns, batch, norm, error)
                    records.append(dict(kind="klead", kgrid=kg, ns=ns, batch=batch,
                                        norm=norm, relative_error=error))
                    # GW keeps its existing handler and first transforms
                    # q-space W; it shares the stored-kernel chunk engine.
                    t3 = P(None, None, None, None, "x", None, "y")
                    v3 = P(None, None, None, "x", "y")
                    conv = F.make_kconv_klead(mesh, kg, t3, v3, norm=norm, mult=-.37)
                    with F.xla_reference():
                        ref_conv = F.make_kconv_klead(mesh, kg, t3, v3, norm=norm, mult=-.37)
                    f = jax.jit(lambda g, w: conv.apply(g, conv.prep(w)))
                    ref = jax.jit(lambda g, w: ref_conv.apply(g, ref_conv.prep(w)))
                    assert F.GW_CONV_TARGET in f.lower(t, v).as_text()
                    native = f(t, v).block_until_ready()
                    expected = ref(t, v).block_until_ready()
                    av = np.asarray(native.addressable_shards[0].data)
                    ev = np.asarray(expected.addressable_shards[0].data)
                    error = float(np.max(np.abs(av - ev)) / max(float(np.max(np.abs(ev))), 1e-300))
                    assert error < 1e-12, ("gw", kg, ns, batch, norm, error)
                    records.append(dict(kind="gw", kgrid=kg, ns=ns, batch=batch,
                                        norm=norm, relative_error=error))

    # Empty distributed axes remain empty without decoding zero divisors.
    for zero_axis in (0, 1, 2, 3, 4):
        shape = [2, 3, 5, 2, 1, 6]
        shape[zero_axis] = 0
        x = put(np.zeros(shape, np.complex128), xs)
        k = put(np.zeros((shape[1], shape[2], 6), np.complex128), ks)
        for layout in (0, 1):
            result = jax.jit(F.make_kconv_kminor(mesh, (3, 2, 1), xs, ks, out_layout=layout))(x, k)
            result.block_until_ready()
            assert result.size == 0

    # Compile production-sized shapes without allocating them.  The old
    # route held a whole-X transpose/FFT temporary; the new route must not.
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    analyses = []
    for batch in (1, 4):
        shape = (2, 200 * px, 200 * py, 2, batch, 144)
        kernel_shape = (shape[1], shape[2], 144)
        x = jax.ShapeDtypeStruct(shape, np.complex128, sharding=NamedSharding(mesh, xs))
        k = jax.ShapeDtypeStruct(kernel_shape, np.complex128, sharding=NamedSharding(mesh, ks))
        for layout in (0, 1):
            kwargs = dict(out_layout=layout)
            f = F.make_kconv_kminor(mesh, (12, 12, 1), xs, ks, **kwargs)
            with F.xla_reference():
                ref = F.make_kconv_kminor(mesh, (12, 12, 1), xs, ks, **kwargs)
            native = jax.jit(f).lower(x, k).compile().memory_analysis()
            reference = jax.jit(ref).lower(x, k).compile().memory_analysis()
            donated = jax.jit(f, donate_argnums=(0,)).lower(x, k).compile().memory_analysis()
            fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                      "alias_size_in_bytes")
            mem = {name: {field: getattr(obj, field) for field in fields}
                   for name, obj in (("native", native), ("xla", reference), ("donated", donated))}
            assert native.temp_size_in_bytes <= native.output_size_in_bytes + 1024
            if rank == 0:
                print(json.dumps(dict(kind="kminor", batch=batch, layout=layout, memory=mem)), flush=True)
            # An outer donation may alias a differently laid-out result
            # through a copy; compare live allocation after alias credit.
            assert donated.temp_size_in_bytes - donated.alias_size_in_bytes <= native.temp_size_in_bytes + 1024
            analyses.append(dict(batch=batch, layout=layout, memory=mem))
        kg = (12, 12, 1)
        shape = (144, 2, 200 * px, 2 * batch, 200 * py)
        kernel_shape = (144, shape[2], shape[4])
        x = jax.ShapeDtypeStruct(shape, np.complex128, sharding=NamedSharding(mesh, tls))
        k = jax.ShapeDtypeStruct(kernel_shape, np.complex128, sharding=NamedSharding(mesh, vls))
        fn = F.make_local_kconv_klead(mesh, kg)
        def old_plan(t, vr):
            scale = F.ffi_fft_scale("ifftn", "ortho", 144) * F.ffi_fft_scale("fftn", "ortho", 144)
            tr = F._kfft(t, kg, "ifftn", "plan") * scale
            return F._kfft(tr * vr[:, None, :, None, :], kg, "fftn", "plan")
        wrap = lambda local, **kw: jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(tls, vls),
                                                       out_specs=tls, check_vma=False), **kw)
        native = wrap(fn).lower(x, k).compile().memory_analysis()
        previous = wrap(old_plan).lower(x, k).compile().memory_analysis()
        donated = wrap(fn, donate_argnums=(0,)).lower(x, k).compile().memory_analysis()
        mem = {name: {field: getattr(obj, field) for field in fields}
               for name, obj in (("native", native), ("previous_plan", previous), ("donated", donated))}
        if rank == 0:
            print(json.dumps(dict(kind="klead", batch=batch, memory=mem)), flush=True)
        assert native.temp_size_in_bytes <= previous.temp_size_in_bytes
        assert donated.temp_size_in_bytes - donated.alias_size_in_bytes <= native.temp_size_in_bytes + 1024
        analyses.append(dict(kind="klead", batch=batch, memory=mem))

    record = dict(rank=rank, cases=len(records), max_relative_error=max(r["relative_error"] for r in records),
                  memory=analyses, seconds=time.perf_counter() - started,
                  cases_detail=records)
    out = os.environ.get("KCONV_HOST_GATE_OUT")
    if out:
        import pathlib
        pathlib.Path(out).mkdir(exist_ok=True, parents=True)
        (pathlib.Path(out) / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    if rank == 0:
        print(f"KCONV_HOST_GATE PASS: {len(records)} cases, max relative error "
              f"{record['max_relative_error']:.3e}, {record['seconds']:.2f}s", flush=True)
        print(json.dumps(analyses), flush=True)
    jax.distributed.shutdown()
    return 0


def test_kconv_host_gate_needs_compute_node():
    import pytest
    pytest.skip("run as a script on CPU compute nodes: python3 tests/test_kconv_host_gate.py")


if __name__ == "__main__":
    sys.exit(main())
