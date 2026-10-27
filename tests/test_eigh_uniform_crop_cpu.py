"""Exact face submatrices with uniform transfers and no replicated row bank."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=144")

import math
import re
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._cyclic import face_crop


def mesh_for(px, py):
    if len(jax.devices()) < px*py:
        pytest.skip("requires virtual process mesh")
    return Mesh(np.array(jax.devices()[:px*py]).reshape(px, py), ("x", "y"))


def test_exact_uniform_crop_both_offsets_batch_rectangle_and_empty():
    jax.config.update("jax_enable_x64", True)
    for px, py in ((1, 1), (2, 2), (3, 3), (8, 10), (12, 12)):
        mesh = mesh_for(px, py)
        n = 2*math.lcm(px, py)
        nr, nc = n+3*px, n+5*py
        for dtype in (np.float64, np.complex128):
            sh = NamedSharding(mesh, P(None, "x", "y"))
            host = np.arange(2*nr*nc).reshape(2, nr, nc).astype(dtype)
            if dtype == np.complex128:
                host = host + .125j*(3+host[:, ::-1, ::-1])
            for offsets in ((0, nc-n), (nr-n, 0), (1, 2)):
                fn = jax.jit(lambda q: face_crop(q, n, offsets=offsets, mesh=mesh),
                             in_shardings=sh, out_shardings=sh)
                got = fn(host)
                expected = host[..., offsets[0]:offsets[0]+n, offsets[1]:offsets[1]+n]
                np.testing.assert_array_equal(np.asarray(got), expected)
                assert got.sharding.is_equivalent_to(sh, 3)
            empty = jax.jit(lambda q: face_crop(q, 0, offsets=(nr, nc), mesh=mesh),
                            in_shardings=sh, out_shardings=sh)(host)
            assert empty.shape == (2, 0, 0) and empty.dtype == dtype
            local = face_crop(jnp.asarray(host), n, offsets=(1, 2), mesh=None)
            np.testing.assert_array_equal(np.asarray(local), host[..., 1:n+1, 2:n+2])
        with pytest.raises(ValueError, match="outside"):
            face_crop(jnp.zeros((nr, nc)), n, offsets=(nr, 0), mesh=mesh)
        if px > 1:
            with pytest.raises(ValueError, match="divide"):
                face_crop(jnp.zeros((nr, nc)), n+1, mesh=mesh)


def test_realistic_crop_hlo_has_four_dependent_uniform_transfers_and_face_bounds():
    jax.config.update("jax_enable_x64", True)
    for px, py, nr, nc, n in ((2, 2, 262, 266, 254),
                              (8, 10, 112, 130, 80),
                              (12, 12, 49920, 49920, 49164)):
        mesh = mesh_for(px, py)
        sh = NamedSharding(mesh, P(None, "x", "y"))
        batch = 2
        bound = batch*max((nr//px)*(-(-(nc//py)//px)*px),
                         (nc//py)*(-(-(n//px)//py)*py))
        for dtype in (jnp.float64, jnp.complex128):
            arg = jax.ShapeDtypeStruct((batch, nr, nc), dtype, sharding=sh)
            exe = jax.jit(lambda q: face_crop(q, n, offsets=(0, nc-n), mesh=mesh),
                          in_shardings=sh, out_shardings=sh).lower(arg).compile()
            text = exe.as_text()
            assert "collective-permute" not in text
            transfers = []
            for line in text.splitlines():
                m = re.match(r"\s*(?:ROOT )?%\S+ = (?:f64|c128)\[([\d,]*)\]", line)
                if m:
                    assert math.prod(int(d) for d in m[1].split(',') if d) <= bound, line
                if re.search(r" all-to-all\(", line):
                    transfers.append(line)
                assert not re.search(r" (?:all-gather|all-reduce)\(", line)
            assert len(transfers) == 4
            # They must be one data chain, rather than independent x/y paths.
            entry = text.rsplit("\nENTRY ", 1)[1]
            definitions = {}
            for line in entry.splitlines():
                m = re.match(r"\s*(?:ROOT )?%(\S+) = (.*)", line)
                if m:
                    definitions[m[1]] = m[2]
            names = [key for key, rhs in definitions.items() if " all-to-all(" in rhs]
            assert len(names) == 4
            def ancestors(name):
                seen, pending = set(), [name]
                while pending:
                    key = pending.pop()
                    if key not in seen and key in definitions:
                        seen.add(key)
                        pending.extend(re.findall(r"%([\w.\-]+)", definitions[key]))
                return seen
            assert all(left in ancestors(right) for left, right in zip(names, names[1:]))
            ma = exe.memory_analysis()
            assert ma.argument_size_in_bytes == batch*(nr//px)*(nc//py)*np.dtype(dtype).itemsize
            assert ma.output_size_in_bytes == batch*(n//px)*(n//py)*np.dtype(dtype).itemsize


def test_exact_elpa_padded_wrapper_discards_sentinel_eigenpairs():
    """Run the production wrapper body, with CPU eigh as the vendor stand-in."""
    from distrib_la import _elpa
    from distrib_la._cyclic import pad_with_sentinels

    tree = ast.parse(Path(_elpa.__file__).read_text())
    body = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_padded")
    for p in (2, 3):
        mesh = mesh_for(p, p)
        n, side = 3*p, 4*p
        face = NamedSharding(mesh, P("x", "y"))
        for dtype in (np.float64, np.complex128):
            grid = np.arange(n*n).reshape(n, n)
            raw = np.sin(.17*grid)
            if dtype == np.complex128:
                raw = raw + 1j*np.cos(.23*grid)
            host = (raw+raw.conj().T+np.diag(np.arange(n))).astype(dtype)
            env = dict(jax=jax, face=face, n=n, side=side, mesh=mesh,
                       pad_with_sentinels=pad_with_sentinels, face_crop=face_crop,
                       _call=jnp.linalg.eigh)
            exec(compile(ast.Module(body=[body], type_ignores=[]), "<actual-ELPA-wrapper>", "exec"), env)
            values, vectors = jax.jit(env['_padded'], in_shardings=face,
                                     out_shardings=(NamedSharding(mesh, P()), face))(host)
            w, q = np.asarray(values), np.asarray(vectors)
            np.testing.assert_allclose(w, np.linalg.eigvalsh(host), rtol=2e-14, atol=2e-14)
            assert np.linalg.norm(host@q-q*w) < 2e-13*np.linalg.norm(host)
            assert np.linalg.norm(q.conj().T@q-np.eye(n)) < 2e-13
            assert vectors.sharding.is_equivalent_to(face, 2)
