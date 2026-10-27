"""Known skinny handoffs use bounded data transport, with unchanged checks."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=144')
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la import _result_check as R
from lxkit.collectives import slab_bytes


def mesh_for(px, py):
    if len(jax.devices()) < px*py:
        pytest.skip('requires virtual mesh')
    return Mesh(np.array(jax.devices()[:px*py]).reshape(px, py), ('x', 'y'))


def test_probe_handoffs_exact_bits_rectangles_batches_and_multiple_slabs():
    jax.config.update('jax_enable_x64', True)
    for px, py, n in ((2, 2, 14), (3, 3, 21), (8, 10, 80), (12, 12, 84),
                      (2, 2, 49164)):
        mesh = mesh_for(px, py)
        for dtype in (np.float64, np.complex128):
            for batch in ((), (2,)):
                shape = batch+(n, 8)
                values = np.arange(math.prod(shape), dtype=np.float64).reshape(shape)
                host = np.sin(.017*values).astype(dtype)
                if dtype == np.complex128:
                    host += 1j*np.cos(.019*values)
                host.reshape(-1)[0] = -0.
                for source, target in (('x', 'y'), ('y', 'x')):
                    spec = (None,)*len(batch)
                    src = NamedSharding(mesh, P(*spec, source, None))
                    dst = NamedSharding(mesh, P(*spec, target, None))
                    call = jax.jit(lambda v: R._probe_rows(v, mesh=mesh, source=source, target=target),
                                   in_shardings=src, out_shardings=dst)
                    got = call(host)
                    assert got.sharding.is_equivalent_to(dst, len(shape))
                    actual = np.asarray(got)
                    np.testing.assert_array_equal(actual, host)
                    # Exact one-owner addition can canonicalize signed zero.
                    nonzero = host != 0
                    np.testing.assert_array_equal(actual[nonzero].view(np.uint64),
                                                  host[nonzero].view(np.uint64))


def test_full_eigh_and_lu_hlo_have_no_implicit_probe_permutations():
    jax.config.update('jax_enable_x64', True)
    for side in (2, 3, 12):
        mesh = mesh_for(side, side)
        n = 49164 if side == 12 else side*257
        face = NamedSharding(mesh, P(None, 'x', 'y'))
        rep = NamedSharding(mesh, P())
        for dtype in (jnp.float64, jnp.complex128):
            aa = jax.ShapeDtypeStruct((2, n, n), dtype, sharding=face)
            xx = jax.ShapeDtypeStruct((2, n, side*13), dtype, sharding=face)
            w = jax.ShapeDtypeStruct((2, n), jnp.float64, sharding=rep)
            fn = jax.jit(lambda a,w,q,b,x: (
                R.eigh_errors(a,w,q,mesh=mesh,cut=.1),
                R.solve_errors((*R.matrix_sketch(a,mesh), *R.rhs_sketch(b,mesh)),x,mesh)))
            text = fn.lower(aa,w,aa,xx,xx).compile().as_text()
            assert 'collective-permute' not in text
            assert re.search(r' all-gather(?:-start)?\(', text)
            face_bound = 2*(n//side)**2
            for line in text.splitlines():
                m = re.match(r'\s*(?:ROOT )?%\S+ = (?:f64|c128)\[([\d,]*)\]', line)
                if m:
                    assert math.prod(int(d) for d in m[1].split(',') if d) <= face_bound, line
            # Probe-only transport cannot hide a replicated matrix or an
            # unbounded batch*N*8 bank. Compile the actual helper separately.
            rows = NamedSharding(mesh, P(None,'x',None))
            cols = NamedSharding(mesh, P(None,'y',None))
            vv = jax.ShapeDtypeStruct((2,n,8),dtype,sharding=rows)
            move = jax.jit(lambda v:R._probe_rows(v,mesh=mesh,source='x',target='y'),
                           in_shardings=rows,out_shardings=cols).lower(vv).compile()
            mt = move.as_text()
            assert 'collective-permute' not in mt
            gathered = []
            for line in mt.splitlines():
                m = re.match(r'\s*(?:ROOT )?%\S+ = (?:f64|c128)\[([\d,]*)\]',line)
                if m:
                    size = math.prod(int(d) for d in m[1].split(',') if d)*np.dtype(dtype).itemsize
                    assert size <= max(slab_bytes(), 2*(n//side)*8*np.dtype(dtype).itemsize),line
                    if re.search(r' all-gather(?:-start)?\(',line):
                        gathered.append(size)
            if side > 2:
                assert gathered and max(gathered) <= slab_bytes()
