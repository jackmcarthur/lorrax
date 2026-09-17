"""The round mirror exchange: packed realization tables, partner ranks and the four products.

``gw.shared_pole_local.partner_realization`` builds each slot's packed permutation and umklapp phase
of its partner row; the round kernel's ``exchange`` sends Pi^T Phi x to the partner rank (ppermute over
('x','y')), multiplies by that rank's samples and returns Phi^* Pi of the result. Reference, on one
device and in the canonical basis: R_s[W] from ``symmetry_maps.unfold_operator_local`` at the partner
parent's q, then (R_s W)^T x for an original state, conj(R_s W) x for a conjugate state, and the same
with dW scaled by the state's factor. Geometry: the store's triangular C3 x inversion fixture
(orbit-packed basis, varying wraps) on a 2x2 host mesh; slots 0 and 1 are partners on different ranks,
slot 2 is its own partner, slot 3 a synthetic repeat. RED TWIN: the identity realization misses.
"""
import numpy as np

from test_shared_pole_store import _sigma_fixture, _test_mesh


def test_round_exchange_equals_the_canonical_realization():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from file_io import shared_pole_store as store
    from gw.shared_pole_directions import _round_kernels
    from gw.shared_pole_local import partner_realization
    from symmetry_maps import unfold_operator_local

    mesh = _test_mesh()
    meta, tables, recipe, identity = _sigma_fixture(mesh)
    header = store._metadata(meta, tables, recipe, identity)
    basis, qt = meta.mu_basis, header["qirr"]
    n, npk = basis.n_logical, basis.n_packed
    ids, slots = [0, 1, 2, 0], [1, 0, 2, 3]
    partner_parent, partner_row = [1, 0, 2], [3, 4, 5]          # unitary inversion x C3^k rows
    rng = np.random.default_rng(1703)
    cplx = lambda *shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    S, r = 3, 4
    w, dw = cplx(4, S, n, n), cplx(4, S, n, n)
    x = cplx(4, 2, n, r)
    flags, j, scales = (False, True), 1, np.asarray([.7 - .2j, -1.1 + .4j])

    batch = NamedSharding(mesh, P(("x", "y")))
    put = lambda a: jax.make_array_from_callback(a.shape, batch, lambda idx: a[idx])
    pack2 = lambda a: basis.pack_host(basis.pack_host(a, axis=-2), axis=-1)
    alpha, inverse, phase = partner_realization(meta, header, ids, partner_parent, partner_row, mesh_xy=mesh)
    program = _round_kernels(mesh).exchange(flags, tuple((i, s) for i, s in enumerate(slots)))
    rep = NamedSharding(mesh, P())
    got = program(put(pack2(w)), put(pack2(dw)), put(basis.pack_host(x, axis=-2)), alpha, inverse, phase,
                  jax.device_put(np.int32(j), rep), jax.device_put(scales, rep))
    got = [basis.unpack_host(np.asarray(g), axis=-2) for g in got]

    one = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    perm, wraps = np.asarray(qt["sym_perm"], np.int32), np.asarray(qt["L_table"], np.float64)

    def realize(a, row, parent):
        body = shard_map(lambda v: unfold_operator_local(
            v, irr_idx=np.array([0]), sym_idx=np.array([row]), q_irr_frac=np.asarray(qt["q_irr_frac"])[parent][None],
            left_local_perm=perm, left_L_table=wraps, right_local_perm=perm, right_L_table=wraps,
            n_sym_spatial=int(qt["n_sym_spatial"])), mesh=one, in_specs=P(None, "x", "y"),
            out_specs=P(None, "x", "y"), check_vma=False)
        return np.asarray(jax.jit(body)(jnp.asarray(a[None])))[0]

    worst, twin = 0.0, np.inf
    for slot in range(3):
        q = ids[slot]
        row, parent = partner_row[q], partner_parent[q]
        assert ids[slots[slot]] == parent
        for k, conjugate in enumerate(flags):
            for i, (operator, scale) in enumerate(((w, 1.0), (dw, scales[k]))):
                realized = realize(operator[slots[slot], j], row, parent)
                want = ((np.conj(realized) if conjugate else realized.T) @ x[slot, k]) * scale
                value = got[2 * k + i][slot]
                worst = max(worst, float(np.max(np.abs(value - want))) / float(np.max(np.abs(want))))
                raw = operator[slots[slot], j]
                naive = ((np.conj(raw) if conjugate else raw.T) @ x[slot, k]) * scale
                twin = min(twin, float(np.max(np.abs(value - naive))) / float(np.max(np.abs(want))))
    assert worst < 1e-13, worst
    assert twin > 1e-2, twin
