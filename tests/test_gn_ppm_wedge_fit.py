"""The GN tail policy on the q wedge equals the full-zone policy on the unfolded field.

Owner, 2026-09-25: GN-PPM is solved and fitted on the irreducible q wedge.
The only non-lane-wise parts of the fit are the tail policy's order
statistics and its (±q, μ↔ν) orbit closure.  On the wedge every count weights
a row by its star size (``QirrOperator.star_sizes``) and the -q lanes come
from the unfold tables (``QirrOperator.minus_q_partner``); for a covariant
field (the unfold of the wedge field) the policy must then select exactly the
wedge rows of what the full-zone policy selects, with the same counts.
Plans: the order-two glide group with an antiunitary row and A-cubic (48
operations), both genuine groups whose unfold commutes with q -> -q (the
synthetic C3 fixture of the unfold tests does not, so its full-zone closure
is not the image of a wedge closure).  Red twin: unit stars (the unweighted wedge) must miss the
budget on a plan whose stars are not all one.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import zeta_mubatch_fixtures as fixtures


def _mesh():
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _wedge(plan):
    from symmetry_maps import QirrOperator
    return QirrOperator(
        values=None, irr_idx=np.asarray(plan.irr_idx), sym_idx=np.asarray(plan.sym_idx),
        sym_perm=np.asarray(plan.sym_perm), L_table=np.asarray(plan.L_table),
        q_irr_frac=np.asarray(plan.k_parent_frac), n_sym_spatial=int(plan.n_sym_spatial),
        full_rows=np.asarray(plan.parent_full_rows, np.int32))


def _case(mesh, fx, seed, *, unit_stars=False):
    import dataclasses
    from gw.minimax_screening import _coarsen_gn_ppm_extreme_tails
    from symmetry_maps import q_negation_index
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    op = _wedge(plan)
    n_w, mu = op.n_wedge, int(np.asarray(plan.sym_perm).shape[1])
    rng = np.random.default_rng(seed)
    sw = NamedSharding(mesh, P(None, "x", "y"))
    A = rng.uniform(0.2, 3.0, (n_w, mu, mu))
    omega_w = 0.5 * (A + np.swapaxes(A, 1, 2))
    Bw = rng.standard_normal((n_w, mu, mu)) + 1j * rng.standard_normal((n_w, mu, mu))
    Bw = 0.5 * (Bw + np.conj(np.swapaxes(Bw, 1, 2)))
    Wc0w = -2.0 * Bw / omega_w
    # The covariant full-zone field: Omega through the zero-wrap unfold (no
    # phase, as the pole store unfolds it), B and Wc0 through the tables.
    op_zero = dataclasses.replace(op, L_table=np.zeros_like(op.L_table, dtype=np.float64))
    put = lambda a: fixtures._put(np.ascontiguousarray(a), sw)
    omega_f = jnp.real(op_zero.with_values(put(omega_w + 0j)).unfold(mesh))
    B_f = op.with_values(put(Bw)).unfold(mesh)
    Wc0_f = op.with_values(put(Wc0w)).unfold(mesh)
    valid_f = jnp.ones(omega_f.shape, bool)
    q_neg = np.asarray(q_negation_index(kg), np.int32)
    full = _coarsen_gn_ppm_extreme_tails(
        omega_f, B_f, valid_f, Wc0_f, q_neg, 1.0, tail_divisor=5)
    stars = np.ones(n_w, np.int64) if unit_stars else op.star_sizes()
    wedge = _coarsen_gn_ppm_extreme_tails(
        put(omega_w), put(Bw), jnp.ones((n_w, mu, mu), bool), put(Wc0w), None, 1.0,
        tail_divisor=5, q_star=tuple(int(v) for v in stars),
        q_partner=op.minus_q_partner(kg, mesh))
    rows = np.asarray(plan.parent_full_rows)
    om_full = fixtures._host(full[0])[rows]
    om_wedge = fixtures._host(wedge[0])
    return dict(n_full=int(plan.n_full), n_wedge=n_w, stars=op.star_sizes().tolist(),
                counts_full=(int(full[2]), int(full[3])),
                counts_wedge=(int(wedge[2]), int(wedge[3])),
                omega_bitwise=bool(np.array_equal(om_full, om_wedge)),
                anchor=(float(full[6]), float(wedge[6])))


def test_wedge_tail_policy_equals_the_full_zone_policy_rows():
    mesh = _mesh()
    fxs = [fixtures._glide_fixture(mesh, np.random.default_rng(3), 2),
           fixtures._acubic_fixture(mesh, np.random.default_rng(4))]
    for i, fx in enumerate(fxs):
        r = _case(mesh, fx, 11 + i)
        assert r["counts_full"] == r["counts_wedge"], r
        assert r["counts_full"][0] > 0 and r["counts_full"][1] > 0, r
        assert r["omega_bitwise"], r
        assert r["anchor"][0] == r["anchor"][1], r


def test_unit_stars_miss_the_budget_red_twin():
    mesh = _mesh()
    fx = fixtures._acubic_fixture(mesh, np.random.default_rng(4))
    r = _case(mesh, fx, 5, unit_stars=True)
    assert any(s > 1 for s in r["stars"]), r
    assert r["counts_full"] != r["counts_wedge"] or not r["omega_bitwise"], r
