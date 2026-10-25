"""Ordinary minimax response: bounded production rows vs full Greens and NumPy.

Run with nine virtual CPU devices. The tiny test tile forces several complete
orbit windows; production keeps the canonical 1 GiB tile unchanged.
"""
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


def fixture(mesh, ns, anti):
    from common.grouped_layout import build_square_grouped_shard_layout
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    rng = np.random.default_rng(379012 + ns)
    grid, nk, parents, nb = (3, 2, 1), 6, 3, 12
    labels = np.arange(19) // 2
    layout = build_square_grouped_shard_layout(labels, tuple(mesh.shape[a] for a in ('x', 'y')))
    mu = layout.axis.n_padded
    canonical = np.tile(np.arange(19), (4, 1))
    for i in range(0, 18, 2):
        canonical[1:, i:i + 2] = [i + 1, i]
    perm = layout.axis.pack_permutations_host(canonical)
    wraps = rng.integers(-1, 2, (4, mu, 3)).astype(float)
    wraps[0] = 0
    wraps[:, ~layout.axis.active_mask] = 0
    irr = np.array([0, 1, 2, 0, 1, 2], np.int32)
    sym = np.array([0, 0, 0, 1, 2, 3], np.int32)
    spin = []
    for k in range(nk):
        a = rng.normal(size=(ns, ns)) + 1j * rng.normal(size=(ns, ns))
        spin.append(np.eye(ns) if k < 3 else np.linalg.qr(a)[0])
    plan = CentroidKUnfoldPlan(mesh, layout, irr, sym, perm, wraps,
        rng.random((parents, 3)), np.asarray(spin), 2 if anti else 4, ns)
    psi = (rng.normal(size=(parents, ns, mu, nb))
           + 1j * rng.normal(size=(parents, ns, mu, nb))) / 3
    psi[:, :, ~layout.axis.active_mask] = 0
    psi[..., 7:] = 0
    e = np.tile(np.linspace(-1.0, 1.0, nb), (parents, 1)) + np.arange(parents)[:, None] * .03
    v = np.zeros((parents, nb), bool); v[:, [0, 2, 3]] = True
    c = np.zeros((parents, nb), bool); c[:, [4, 6]] = True
    # Empty parent support exercises exact-zero active-band ranges.
    c[1] = False
    return grid, plan, psi, e, v, c


def numpy_response(grid, plan, psi, e, v, c, t, alpha, complete, real_times):
    """Independent full parent matrices, literal symmetry action and NumPy FFTs."""
    nk, mu, ns = plan.n_full, psi.shape[2], psi.shape[1]
    out = np.zeros((alpha.shape[0], nk, mu, mu), complex)

    def green(weights):
        raw = np.einsum('pamb,pb,pcnb->pmanc', psi, weights, psi.conj())
        partner = np.einsum('pamb,pb,pcnb->pmanc', psi.conj(), weights, psi)
        full = []
        for k in range(nk):
            p, op = int(plan.irr_idx[k]), int(plan.sym_idx[k])
            idx = plan.sym_perm[op]
            good = idx >= 0
            idx = np.maximum(idx, 0)
            anti = op >= plan.n_sym_spatial
            x = (partner if anti else raw)[p][idx][:, :, idx]
            phase = np.exp(2j * np.pi * (plan.L_table[op] @ plan.k_parent_frac[p]))
            if anti:
                phase = phase.conj()
            x = x * phase[:, None, None, None] * phase.conj()[None, None, :, None]
            x = np.where(good[:, None, None, None] & good[None, None, :, None], x, 0)
            u = plan.spin_action_full[k]
            full.append(np.einsum('ab,mbnc,dc->mand', u, x, u.conj()))
        full = np.asarray(full)
        return np.fft.ifftn(full.reshape(grid + full.shape[1:]), axes=(0, 1, 2), norm='ortho')

    vmax, cmin = e[v].max(), e[c].min()
    for j, time in enumerate(t):
        time = time.real if real_times else time
        gv = green(v * np.exp(time * (e - vmax)))
        gc = green(c * np.exp(-(time if real_times else time.conjugate()) * (e - cmin)))
        pair = (gv * gc.conj()).sum(axis=(4, 6)).reshape(nk, mu, mu)
        if complete:
            pair += pair.conj()
        out += alpha[:, j, None, None, None] * pair[None]
    return np.fft.fftn(out.reshape((len(alpha),) + grid + (mu, mu)), axes=(1, 2, 3),
                       norm='ortho').reshape(len(alpha), nk, mu, mu)


@pytest.mark.parametrize('side,ns,anti,n_out', [(2, 1, False, 2), (2, 2, True, 1),
                                             (2, 2, False, 1), (3, 2, True, 2)])
def test_rows_full_and_independent_reference(monkeypatch, side, ns, anti, n_out):
    from ffi import fft as F
    from gw import w_isdf as W
    from gw.minimax_screening import MinimaxNodes
    from runtime import tiles
    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side), ('x', 'y'))
    grid, plan, psi, e, v, c = fixture(mesh, ns, anti)
    np_, _, mu, nb = psi.shape
    put = lambda a, s: jax.device_put(a, NamedSharding(mesh, s))
    pm = put(psi, P(None, None, 'x', 'y'))
    pn = put(psi.transpose(0, 3, 1, 2), P(None, 'x', None, 'y'))
    tail = (pm, pn, put(v, P()), put(c, P()), put(e, P()), jnp.asarray(e[v].max()), jnp.asarray(e[c].min()))
    with F.xla_reference():
        for contour, real_times in ((False, True), (True, True), (True, False)):
            t = np.array([.1, .4, 1.1], complex)
            if not real_times:
                t += np.array([.07, -.09, .11]) * 1j
            alpha = np.array([[-.3, .12, -.08], [.11, -.07, .03]], complex)
            if contour:
                alpha += np.array([[.2, -.03, .09], [-.06, .04, .1]]) * 1j
            alpha = alpha[:n_out]
            nodes = MinimaxNodes(jnp.asarray(t), jnp.asarray(alpha[0] if n_out == 1 else alpha))
            args = (nodes, *tail)
            full = W._get_chi_minimax_kernel_face(mesh, grid, 6, n_out, contour,
                (np_, nb, mu, ns), k_unfold_plan=plan)
            value = full(*args)
            expected_full = np.stack([np.asarray(a) for a in value]) if n_out > 1 else np.asarray(value)[None]
            expected = numpy_response(grid, plan, psi, e, v, c, t, alpha, not contour, real_times)
            np.testing.assert_allclose(expected_full, expected, rtol=5e-12, atol=3e-11,
                                       err_msg=f'full/reference contour={contour} real_times={real_times}')
            for tile in (1 << 30, 8192):
                monkeypatch.setattr(tiles, 'TILE_BYTES', tile)
                from gw.subtile_stream import green_operand_program, _with_borrowed_green_faces
                for layout in ('face', 'axis'):
                    placed = green_operand_program(mesh, layout)(pm, pn)
                    prepared = _with_borrowed_green_faces(layout, (pm, pn), placed)
                    kernel = W._get_chi_minimax_kernel_fused(mesh, grid, 6, n_out, contour,
                        (np_, nb, mu, ns), k_unfold_plan=plan, real_times=real_times,
                        layout=layout)
                    value = kernel(nodes, *prepared, *tail[2:])
                    got = np.stack([np.asarray(a) for a in value]) if n_out > 1 else np.asarray(value)[None]
                    np.testing.assert_allclose(got, expected, rtol=5e-12, atol=3e-11)
                    if tile == 8192:
                        assert len(kernel.row_plan.windows) > 1
                    else:
                        assert len(kernel.row_plan.windows) == 1


@pytest.mark.parametrize('platform', ['cpu', 'ROCM'])
def test_charge_factory_dials_do_not_reuse_another_backend(monkeypatch, platform):
    """A reference/native context change must rebind the shared FFT factory."""
    from common import fft_helpers
    from ffi import fft as F
    from ffi import gate
    from gw import w_isdf as W
    monkeypatch.setattr(gate, 'mesh_ffi_platform', lambda mesh: platform)
    monkeypatch.setattr(F, '_MATHDX_DOWN', [])
    monkeypatch.setattr(F, '_KDFT_DOWN', [])
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    grid, plan, *_ = fixture(mesh, 1, False)
    def factory(*args, **kwargs):
        return object()

    monkeypatch.setattr(fft_helpers, 'make_kconv_chi_unfold', factory)
    for owner in (lambda: W._charge_stream_kconv(mesh, grid, plan),
                  lambda: W._charge_window_kconv(mesh, grid, plan, plan.n_centroid_packed // 2)):
        W._CHARGE_KCONV.clear()
        native = owner()[0]
        with F.xla_reference():
            reference = owner()[0]
            assert owner()[0] is reference
        assert native is not reference
        assert owner()[0] is native
        # A startup demotion changes the emitted program even when the
        # vendor's broad backend tag stays unchanged. Failure text itself
        # must not cause a new compilation for the same demoted route.
        for flag in (F._MATHDX_DOWN, F._KDFT_DOWN):
            W._CHARGE_KCONV.clear()
            normal = owner()[0]
            flag.append('startup provider probe failed')
            demoted = owner()[0]
            assert demoted is not normal
            flag[0] = 'another diagnostic for the same demotion'
            assert owner()[0] is demoted
            flag.clear()
            assert owner()[0] is normal
    W._CHARGE_KCONV.clear()
