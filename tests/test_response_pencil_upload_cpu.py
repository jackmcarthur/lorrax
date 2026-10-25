"""A rank-owned prepared pencil executes without another collective compile."""
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)


def pencil_reference(zps, size, columns, steps):
    """Literal wide Hankel SVD, independent of the production QR/dilation."""
    out = []
    rank = min(size, len(zps) * len(columns))
    for padding, step in steps:
        base = -padding + (np.arange(size)[:, None] + columns[None, :]) * step
        h, shifted = [], []
        for z in zps:
            block = 1 / (base - z)
            scale = 1 / np.linalg.norm(block)
            h.append(scale * block)
            shifted.append(scale / (base + step - z))
        u, s, vh = np.linalg.svd(np.hstack(h), full_matrices=False)
        out.append((s[:rank], u[:, :rank].conj().T @ np.hstack(shifted) @ vh[:rank].conj().T))
    return out


def test_prepared_rank_owned_pencils_have_no_new_compiles(monkeypatch):
    from jax._src import compiler
    from gw.response_pencil import DevicePencils
    zps = np.asarray([.13 + .7j, .91 + 1.3j], np.complex128)
    columns = np.asarray([0, 2, 5], np.int64)
    size = 8
    steps = ((.2, .3), (.4, .17))
    source = DevicePencils()
    source.prepare([(size, len(zps), len(columns))])
    for future in source._ready.values():
        future.result()
    original = compiler.compile_or_get_cached
    compiles = []

    def record(*args, **kwargs):
        compiles.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, 'compile_or_get_cached', record)
    result = list(source(zps, size, columns, steps))
    assert not compiles, 'rank-owned execution requested a new cross-rank compile'
    expected = pencil_reference(zps, size, columns, steps)
    for (s, shift), (s_ref, shift_ref) in zip(result, expected):
        np.testing.assert_allclose(s, s_ref, rtol=3e-12, atol=2e-15)
        # The SVD and dilation bases differ by unitary phases. The projected
        # shift's spectrum is basis independent and fixes the rule nodes.
        actual_roots = np.linalg.eigvals(shift)
        reference_roots = np.linalg.eigvals(shift_ref)
        order = lambda x: x[np.lexsort((x.imag, x.real))]
        np.testing.assert_allclose(order(actual_roots), order(reference_roots), rtol=2e-8, atol=2e-9)
    assert all(np.isfinite(x).all() for pair in result for x in pair)
