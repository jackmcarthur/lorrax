"""Independent Coulomb Gram identities and distributed carrier contracts."""
import os
from pathlib import Path
import subprocess
import sys


_CHILD = r'''
import gc
import json
import re
import sys
import weakref
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host
from gw import v_q_g_flat as G
from lxkit import memory
from runtime.aot_memory import compiled_new_bytes
jax.config.update('jax_enable_x64', True)
side = int(sys.argv[1])
mesh = Mesh(np.asarray(jax.devices()).reshape(side, side), ('x', 'y'))
logical, ng, panel = 13, 11, 4
nmu = 16 if side == 2 else 18
zs = NamedSharding(mesh, P(None, ('x', 'y'), None))
vs = NamedSharding(mesh, P())
rng = np.random.default_rng(482 + side)
physical = rng.normal(size=(logical, ng)) + 1j*rng.normal(size=(logical, ng))
host = np.full((1, nmu, ng), np.nan + 1j*np.nan, np.complex128)
host[0, :logical] = physical
signed = np.linspace(-0.7, 1.3, ng)
signed[0] = 7.25
complex_weights = signed + 0.03j*np.arange(ng)
head_only = np.zeros(ng, np.complex128)
head_only[0] = 2.0 + 3.0j
weights_cases = (signed, complex_weights, head_only, np.zeros(ng))
original_admit = memory.admit
quotes = []
def observe(stage, routes, room, **kw):
    if stage == 'Coulomb G-flat Gram':
        quotes.append(routes[0][1].bytes())
    return original_admit(stage, routes, room, **kw)
memory.admit = observe
worst = 0.0
for nout in ((14, 18) if side == 2 else (15, 21)):
    z = jax.device_put(host, zs)
    before = np.asarray(gather_to_host(z))
    for weights in weights_cases:
        got = G.coulomb_gram_gflat(z, weights, mesh_xy=mesh,
            n_mu_logical=logical, n_mu_out=nout, g_chunk=panel)
        assert got.sharding == NamedSharding(mesh, P('x', 'y'))
        value = np.asarray(gather_to_host(got))
        want = np.zeros((nout, nout), np.complex128)
        # Literal weighted sum: neither v nor the right zeta is conjugated.
        want[:logical, :logical] = np.einsum('mg,g,ng->mn',
            physical.conj(), weights, physical, optimize=False)
        np.testing.assert_allclose(value, want, rtol=2e-13, atol=3e-13)
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(value[logical:], 0)
        np.testing.assert_array_equal(value[:, logical:], 0)
        scale = max(np.linalg.norm(want), 1.0)
        worst = max(worst, float(np.linalg.norm(value-want)/scale))
    # write_g0=False does not erase the G=0 column from the Gram.
    without_head = np.einsum('mg,g,ng->mn', physical.conj(),
        np.r_[0.0, signed[1:]], physical, optimize=False)
    with_head = np.einsum('mg,g,ng->mn', physical.conj(), signed,
        physical, optimize=False)
    assert np.linalg.norm(with_head-without_head) > 1.0
    np.testing.assert_array_equal(np.asarray(gather_to_host(z)), before)
    program = G._gflat_gram_program(mesh, nmu, ng, logical, nout, panel)
    executable = program.lower(
        jax.ShapeDtypeStruct((1, nmu, ng), np.complex128, sharding=zs),
        jax.ShapeDtypeStruct((ng,), np.complex128, sharding=vs)).compile()
    assert quotes[-1] == compiled_new_bytes(executable)
    # Inspect partitioned instructions, not global shape annotations: all
    # panel gathers remain bounded and legal crop/pad stays on mesh faces.
    hlo = executable.as_text()
    forbidden = {(1,nmu,ng), (nmu,ng), (1,nmu,nmu), (nmu,nmu), (nout,nout)}
    for line in hlo.splitlines():
        match = re.match(r'\s*(?:ROOT\s+)?%[^=]+ = c128\[([0-9,]*)\]', line)
        if match:
            shape = tuple(int(x) for x in match[1].split(',') if x)
            assert shape not in forbidden, line
        if ' all-gather(' in line or ' all-gather-start(' in line:
            match = re.search(r'c128\[([0-9,]+)\]', line)
            if match:
                shape = tuple(map(int, match[1].split(',')))
                assert shape == (nmu//side, panel), line
    ref = weakref.ref(z)
    del z, got
    gc.collect()
    assert ref() is None, 'the geometry cache retained a zeta payload'
memory.admit = original_admit
print(json.dumps(dict(side=side, worst_relative=worst, quote_checks=len(quotes))))
'''


def test_weighted_gram_g0_carriers_and_local_memory_contract():
    root = Path(__file__).resolve().parents[1]
    for side in (2, 3):
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=str(root/'src'))
        flags = env.get('XLA_FLAGS', '')
        # Each invocation starts its own backend; parent test ordering cannot
        # turn the P9 control into a one-device skip.
        env['XLA_FLAGS'] = flags + f' --xla_force_host_platform_device_count={side*side}'
        result = subprocess.run([sys.executable, '-c', _CHILD, str(side)],
            cwd=root, env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
