"""Mode 11's write call off mathdx (CPU, seconds).

``ffi.fft.make_kconv_chi_unfold``'s ``acc=None`` returns ``alpha[o] * chi_tau``, the update onto exact
zeros: on the XLA and the cpu plan backends it must equal the call on a zero ``acc`` bit for bit.
``rows=`` (the q-row store) runs on the CUDA fused single arm only (``ffi.fft.chi_q_store_ok``): off
mathdx it raises by name, as on a mathdx grid without ``fn.q_store``.  The mathdx route of both is
held to the XLA reference on four GPUs by ``tests/test_kconv_xla_gate.py`` (``arm_cases``).
"""
import contextlib

import numpy as np
import pytest

KG = (3, 2, 2)
NS = 2


def _tables(rng, nk, n_parent, m):
    from symmetry_maps import UnfoldLoadTables
    row = rng.integers(0, n_parent, nk).astype(np.int32)
    trs = (rng.random(nk) < 0.4).astype(np.int32)
    lsrc = rng.integers(-1, m * NS, (nk, m * NS)).astype(np.int32)
    rsrc = rng.integers(-1, m * NS, (nk, m * NS)).astype(np.int32)
    phase = lambda shape: np.exp(2j * np.pi * rng.random(shape))
    q, _ = np.linalg.qr(rng.standard_normal((nk, NS, NS)) + 1j * rng.standard_normal((nk, NS, NS)))
    return UnfoldLoadTables(row=row, trs=trs, lsrc=lsrc, rsrc=rsrc, mph=phase(lsrc.shape),
                            nph=phase(rsrc.shape), spin=q.astype(np.complex128), n_parent=n_parent,
                            mesh_shape=(1, 1))


@pytest.mark.parametrize("backend", ["xla", "plan"])
def test_chi_unfold_write_equals_zero_acc_and_rows_refuse(backend):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from ffi import fft as F
    jax.config.update("jax_enable_x64", True)
    if backend == "plan":
        from ffi.common import ffi_loader
        ok, why = ffi_loader.probe_target(F.FLAT_K_TARGET, "cpu")
        if not ok:
            pytest.skip(f"the cpu plan backend needs {F.FLAT_K_TARGET}: {why}")
    rng = np.random.default_rng(20261017)
    nk, n_parent, m, n_out = int(np.prod(KG)), 5, 3, 2
    tables = _tables(rng, nk, n_parent, m)
    rnd = lambda *s: jnp.asarray(rng.standard_normal(s) + 1j * rng.standard_normal(s))
    Gv, Gvt, Gc, Gct = (rnd(n_parent, m, NS, m, NS) for _ in range(4))
    alpha = rnd(n_out)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    with (F.xla_reference() if backend == "xla" else contextlib.nullcontext()):
        assert F.kconv_backend(mesh, KG) == backend
        fn = F.make_kconv_chi_unfold(mesh, KG, tables, n_out=n_out, complete=True, norm="ortho")
    assert not fn.q_store and not F.chi_q_store_ok(mesh, KG, NS, 1)
    zeros = jnp.zeros((n_out, nk, m, m), jnp.complex128)
    for partners in ((Gvt, Gct), ()):
        got = np.asarray(fn(None, Gv, Gc, alpha, *partners))
        want = np.asarray(fn(zeros, Gv, Gc, alpha, *partners))
        assert got.shape == want.shape and np.array_equal(got.view(np.uint64), want.view(np.uint64)), \
            f"acc=None differs from the call on zeros (partners={bool(partners)})"
    with pytest.raises(ValueError, match="rows= takes acc=None on a grid with fn.q_store"):
        fn(None, Gv, Gc, alpha, Gvt, Gct, rows=[0, 1])
