"""BSE V(q) coarse prep (bse.vq_interp.prepare_coarse): a q chunk that does not divide the mesh
runs on the face (place()-decided eigh, face GEMMs) with the q-batched route's values, and its
programs hold no whole n_mu x n_mu per rank. Synthetic sphere geometry; P = 4 host devices."""
import re
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh  # noqa: E402

NQ, NGK, NG = 4, 40, 12


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _inputs(NMU):
    rng = np.random.default_rng(11)
    zg = rng.standard_normal((NQ, NMU, NGK)) + 1j * rng.standard_normal((NQ, NMU, NGK))
    a = rng.standard_normal((NQ, NMU, 2 * NMU)) + 1j * rng.standard_normal((NQ, NMU, 2 * NMU))
    c = a @ np.conj(np.swapaxes(a, 1, 2))
    reader = types.SimpleNamespace(read_q_slab=lambda s, nb, sharding: jax.device_put(zg[s:s + nb], sharding))
    qfr = np.c_[rng.uniform(-.5, .5, (NQ, 2)), np.zeros(NQ)]
    zx = dict(nq=NQ, n_mu=NMU, ngkmax=NGK, ZG=reader, qfr=qfr,
              rmu_frac=rng.uniform(0, 1, (NMU, 3)), zeta_cutoff=10.0)
    return zx, c, rng


def _geometry(monkeypatch, vq, rng):
    gs = np.r_[rng.integers(-2, 3, (2, NG)), np.zeros((1, NG), int)]
    slots = rng.integers(-1, NGK, (NQ, NG))
    v = {None: rng.uniform(.5, 2, (NQ, NGK)), "slab_lr": rng.uniform(.1, .4, (NQ, NGK))}
    monkeypatch.setattr(vq, "lr_gset", lambda zx, alpha, degrees=None: gs)
    monkeypatch.setattr(vq, "v_sphere_padded", lambda zx, kind=None, alpha=None: v[kind])
    monkeypatch.setattr(vq, "_sphere_slot", lambda zx, q, GS: slots[q])
    monkeypatch.setattr(vq, "v_slab_on_set", lambda zx, q, GS, kind=None, alpha=None: np.ones(NG))


def test_ragged_chunks_take_the_face_with_the_same_values(mesh, monkeypatch, nmu=32):
    from bse import vq_interp as vq
    from common import gpu_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    zx, c, rng = _inputs(nmu)
    c = jax.device_put(c, NamedSharding(mesh, P(None, "x", "y")))      # C_q arrives on the face
    _geometry(monkeypatch, vq, rng)
    rooms = {"BSE V(q) coarse GEMM": 1 << 10}          # the GEMMs on the face SUMMA (1 KiB room)
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: rooms.get(stage, 1 << 34))
    built = {}
    real_jit = jax.jit

    def jit(fn=None, **kw):                             # record the face programs and their arguments
        if fn is None:
            return lambda f: jit(f, **kw)
        compiled = real_jit(fn, **kw)
        if getattr(fn, "__name__", None) not in ("split", "herm"):
            return compiled

        def run(*args):
            built[fn.__name__] = (compiled, args)
            return compiled(*args)
        return run
    monkeypatch.setattr(vq.jax, "jit", jit)
    kw = dict(degrees={0: 1}, keep_host_mirrors=True)
    batched = vq.prepare_coarse(zx, c, mesh, q_chunk=4, **kw)    # 4 % 4: whole q tiles per owner
    assert not built
    face = vq.prepare_coarse(zx, c, mesh, q_chunk=3, **kw)       # 3 and 1: ragged, the face
    assert set(built) == {"split", "herm"}
    for key in ("S", "V_SRc_np", "Fch"):
        np.testing.assert_allclose(face[key], batched[key], rtol=1e-10, atol=1e-10 * np.abs(batched[key]).max())
    whole = re.compile(rf"c128\[\d+,{nmu},{nmu}\]")
    for name, (compiled, args) in built.items():
        text = compiled.lower(*args).compile().as_text()
        bad = [l for l in text.splitlines() if whole.search(l.replace(" ", ""))]
        names = sorted({m.group(1) for l in bad for m in [re.search(r'op_name="([^"]*)"', l)] if m})
        assert not bad, (name, names[:12])
