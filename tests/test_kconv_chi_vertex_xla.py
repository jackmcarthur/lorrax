"""Mode 11 with channel vertices on the XLA composition, against numpy.

``ffi.fft.make_kconv_chi_vertex`` forms, per channel ``(i, j)``,

    acc[ch, k] += sum_ab conj(phase_i[a]) phase_j[b] conj(Gc'[perm_i a, perm_j b]) Gv'_ab,

with ``G' = ifftn_k(sign_k * U_k G[row(k)] U_k^dagger)``.  ``sign_c`` is per
full-k row (the four-current stream's Dirac parity on its mixed quadrants,
``gw.w_isdf._photon_chi_kconvs``), so it multiplies Gc's unfolded k rows
before the transform to R, as the mathdx kernel does on its load.  The XLA
backend and the cpu plan backend share that composition (they differ only in
the k transform), so this test builds the factory on both, the XLA one inside
``ffi.fft.xla_reference()``, and holds each to a numpy composition of the
formula at 1e-12, with a sign that is -1 on some k rows, for partner tiles and
for conj(G) partners.  The plan case needs the host library's flat-k target and
skips, naming why, without it.  The mathdx route is held to the same
composition on four GPUs by ``tests/test_kconv_xla_gate.py``.
"""
import contextlib

import pytest
import numpy as np

KG = (3, 2, 2)
NS = 2
TOL = 1e-12


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


def _reference(Gv, Gvt, Gc, Gct, tables, sign, left, right):
    import jax.numpy as jnp
    from symmetry_maps import apply_unfold_load_tables_local
    t = tables                                          # a 1x1 mesh: the local tables are the tables
    nk = int(np.prod(KG))
    flat = lambda g: jnp.asarray(g.reshape(g.shape[0], g.shape[1] * NS, g.shape[3] * NS))

    def unfolded(g, gt, s):
        O = np.asarray(apply_unfold_load_tables_local(flat(g), flat(gt), t, np.asarray(tables.spin)))
        O = (O * s[:, None, None, None, None]).reshape(*KG, *O.shape[1:])
        return np.fft.ifftn(O, axes=(0, 1, 2), norm="ortho").reshape(nk, *O.shape[3:])
    lower, upper = unfolded(Gv, Gvt, np.ones(nk)), unfolded(Gc, Gct, sign)
    planes = []
    for pl, hl in left:
        for pr, hr in right:
            up = upper[:, :, list(pl)][:, :, :, :, list(pr)]
            planes.append(np.einsum("kxayb,a,b,kxayb->kxy", np.conj(up), np.conj(hl), np.asarray(hr),
                                    lower))
    return np.stack(planes)


@pytest.mark.parametrize("backend", ["xla", "plan"])
def test_chi_vertex_sign_acts_on_k_rows(backend):
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
    rng = np.random.default_rng(20261009)
    nk, n_parent, m = int(np.prod(KG)), 5, 3
    tables = _tables(rng, nk, n_parent, m)
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    Gv, Gvt, Gc, Gct = (rnd(n_parent, m, NS, m, NS) for _ in range(4))
    sign = np.where(rng.random(nk) < 0.5, -1.0, 1.0)
    sign[:2] = (1.0, -1.0)                              # both signs present
    left = (((0, 1), (1, 1j)), ((1, 0), (-1, 1j)))       # (perm, phase) monomial vertices
    right = (((1, 0), (1j, -1)),)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    with (F.xla_reference() if backend == "xla" else contextlib.nullcontext()):
        assert F.kconv_backend(mesh, KG) == backend
        fn = F.make_kconv_chi_vertex(mesh, KG, tables, left_vertices=left, right_vertices=right,
                                     sign_c=sign, norm="ortho")
    acc0 = rnd(len(left) * len(right), nk, m, m)
    for partners in (True, False):
        args = (Gvt, Gct) if partners else ()
        got = np.asarray(fn(jnp.asarray(acc0), jnp.asarray(Gv), jnp.asarray(Gc),
                            *(jnp.asarray(a) for a in args))) - acc0
        want = _reference(Gv, Gvt if partners else np.conj(Gv), Gc, Gct if partners else np.conj(Gc),
                          tables, sign, left, right)
        rel = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
        assert rel <= TOL, f"partners={partners}: rel {rel:.3e} > {TOL:g}"
