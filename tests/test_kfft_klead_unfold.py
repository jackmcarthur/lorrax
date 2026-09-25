"""The interaction's R-space operand read from the q wedge, against the chain it replaces.

``ffi.fft.make_kfft_klead_unfold(W_wedge)`` must equal, bit for bit,
``make_kconv_klead(...).prep`` of the full-zone interaction that
``symmetry_maps.unfold_isdf_operator`` builds from the same wedge, on both
antiunitary rules: ``conj`` (a Hermitian interaction, no partner tile) and
``pair_transpose`` (the partner read on antiunitary rows: ``unfold_isdf_operator``
takes it with reversed endpoint axes, the door as the tile ``Wt[p, mu, nu]``
it gathers from, so the door's partner is its transpose).  Plans: the
order-two glide group with an antiunitary row, A-cubic (48 operations) and
the C3 group on a 3x3 grid with general umklapp phases (q = n/3).  A Lorentz
block (``nA = nB = 3``, a random orthogonal action per q) is held to the
unfold followed by ``L O R^T`` in XLA within 2 ulp; so is the C3 group's
scalar case, whose general phases the reference forms inside its own fusion.  Red twin: the right
source table rolled by one slot must miss.  ``wedge_case`` is reused on GPU
by ``tests/multi_device/kconv_router_p4.py`` (mathdx mode 9).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import zeta_mubatch_fixtures as fixtures
from test_kconv_klead_unfold import c3_fixture


def _wedge_tables(plan, mesh, rule, *, spin=None, right_spin=None):
    from symmetry_maps import unfold_load_tables
    nk = int(plan.n_full)
    return unfold_load_tables(
        irr_idx=plan.irr_idx, sym_idx=plan.sym_idx, sym_perm=plan.sym_perm,
        L_table=plan.L_table, k_irr_frac=plan.k_parent_frac,
        spin_action_full=np.ones((nk, 1, 1), complex) if spin is None else spin,
        right_spin_action_full=right_spin, n_sym_spatial=plan.n_sym_spatial,
        mesh_xy=mesh, logical_centroid_extent=plan.n_centroid_packed,
        right_logical_centroid_extent=plan.n_centroid_packed, trs_rule=rule)


def _full_zone(plan, mesh, Wd, Wtd, rule):
    from symmetry_maps import unfold_isdf_operator
    mu = int(plan.n_centroid_packed)
    return unfold_isdf_operator(
        Wd, irr_idx=plan.irr_idx, sym_idx=plan.sym_idx, sym_perm=plan.sym_perm,
        L_table=plan.L_table, q_irr_frac=plan.k_parent_frac, mesh_xy=mesh,
        n_sym_spatial=plan.n_sym_spatial, trs_rule=rule,
        trs_pair_q_ibz=Wtd if rule == "pair_transpose" else None,
        left_logical_extent=mu, right_logical_extent=mu)


def wedge_case(mesh, fx, rule, seed=0):
    """(bitwise door == prep(unfold), red twin) for a scalar interaction on the wedge."""
    from ffi import fft as F
    from gw.wavefunction_bundle import SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    n_par, nk = int(plan.n_parent), int(plan.n_full)
    mu = int(np.asarray(plan.sym_perm).shape[1])
    rng = np.random.default_rng(seed + 31 * n_par)
    crand = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    sw = NamedSharding(mesh, P(None, "x", "y"))
    W, Wt = crand(n_par, mu, mu), crand(n_par, mu, mu)
    Wd, Wtd = fixtures._put(W, sw), fixtures._put(Wt, sw)
    Wt_tile = fixtures._put(np.ascontiguousarray(np.swapaxes(Wt, 1, 2)), sw)
    kconv = F.make_kconv_klead(mesh, kg, SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC, norm="ortho", mult=1.0)
    ref = fixtures._host(kconv.prep(_full_zone(plan, mesh, Wd, Wtd, rule)))
    tables = _wedge_tables(plan, mesh, rule)
    anti = bool(np.any(np.asarray(tables.trs)))
    door = F.make_kfft_klead_unfold(mesh, kg, tables, norm="ortho")
    partner = Wt_tile if (anti and rule == "pair_transpose") else None
    got = fixtures._host(door(Wd, partner))
    red = fixtures._host(F.make_kfft_klead_unfold(
        mesh, kg, tables._replace(rsrc=np.roll(tables.rsrc, 1, axis=1)), norm="ortho")(Wd, partner))
    scale = float(np.max(np.abs(ref)))
    return dict(rule=rule, nk=nk, n_parent=n_par, mu=mu, antiunitary=anti,
                door_bitwise=bool(np.array_equal(got, ref)),
                max_abs=float(np.max(np.abs(got - ref))),
                rel=float(np.max(np.abs(got - ref))) / scale,
                red_rel=float(np.max(np.abs(red - ref))) / scale)


def lorentz_wedge_case(mesh, fx, seed=0):
    """(rel of the door vs unfold-then-rotate, red twin) for a 3x3 Lorentz block on the wedge."""
    from ffi import fft as F
    from gw.wavefunction_bundle import SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    n_par, nk = int(plan.n_parent), int(plan.n_full)
    mu = int(np.asarray(plan.sym_perm).shape[1])
    rng = np.random.default_rng(seed + 7 * n_par)
    crand = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    lam = np.stack([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(nk)]).astype(complex)
    W = crand(n_par, mu, 3, mu, 3)
    sw = NamedSharding(mesh, P(None, "x", "y"))
    blocks = [[_full_zone(plan, mesh, fixtures._put(np.ascontiguousarray(W[:, :, a, :, b]), sw),
                          None, "conj") for b in range(3)] for a in range(3)]
    full = np.stack([np.stack([fixtures._host(blocks[a][b]) for b in range(3)], axis=-1)
                     for a in range(3)], axis=2)                 # (nk, mu, 3, mu, 3)
    rot = np.einsum("kac,kxcyd,kbd->kxayb", lam, full, np.conj(lam))
    kconv = F.make_kconv_klead(mesh, kg, SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC, norm="ortho", mult=1.0)
    ref = fixtures._host(kconv.prep(fixtures._put(rot.reshape(nk, 3 * mu, 3 * mu), sw)))
    tables = _wedge_tables(plan, mesh, "conj", spin=lam, right_spin=lam)
    Wd = fixtures._put(W.reshape(n_par, 3 * mu, 3 * mu), sw)
    got = fixtures._host(F.make_kfft_klead_unfold(mesh, kg, tables, norm="ortho")(Wd))
    red = fixtures._host(F.make_kfft_klead_unfold(
        mesh, kg, tables._replace(rsrc=np.roll(tables.rsrc, 1, axis=1)), norm="ortho")(Wd))
    scale = float(np.max(np.abs(ref)))
    return dict(nk=nk, n_parent=n_par, mu=mu,
                rel=float(np.max(np.abs(got - ref))) / scale,
                red_rel=float(np.max(np.abs(red - ref))) / scale)


def _mesh():
    devs = np.asarray(jax.devices()[:4])
    return Mesh(devs.reshape(2, 2), ("x", "y"))


def _cases(mesh):
    rng = np.random.default_rng(3)
    cases = [fixtures._glide_fixture(mesh, rng, 2), fixtures._acubic_fixture(mesh, rng)]
    return cases + [c3_fixture(mesh, 2)]


def test_wedge_door_matches_prep_of_the_unfolded_interaction():
    from ffi import fft as F
    mesh = _mesh()
    assert F.kconv_backend(mesh) == "plan"
    eps = float(np.finfo(float).eps)
    for fx, exact in zip(_cases(mesh), (True, True, False)):
        for rule in ("conj", "pair_transpose"):
            r = wedge_case(mesh, fx, rule)
            # Exact phases (glide, A-cubic): bit for bit.  General phases
            # (q = n/3): the reference forms its phase inside its own fusion,
            # so the two agree to rounding, as the Green tables do.
            assert r["door_bitwise"] if exact else r["rel"] <= 2 * eps, r
            assert r["red_rel"] > 1e-3, r


def test_wedge_door_lorentz_block_matches_unfold_then_rotate():
    mesh = _mesh()
    eps = float(np.finfo(float).eps)
    for fx in _cases(mesh)[:1] + _cases(mesh)[2:]:
        r = lorentz_wedge_case(mesh, fx)
        assert r["rel"] <= 2 * eps, r
        assert r["red_rel"] > 1e-3, r


def test_wedge_door_requires_the_partner_on_the_pair_transpose_rule():
    from ffi import fft as F
    mesh = _mesh()
    fx = fixtures._glide_fixture(mesh, np.random.default_rng(5), 2)
    plan, kg = fx["plan"], tuple(fx["kgrid"])
    tables = _wedge_tables(plan, mesh, "pair_transpose")
    mu = int(np.asarray(plan.sym_perm).shape[1])
    W = fixtures._put(np.zeros((int(plan.n_parent), mu, mu), complex),
                      NamedSharding(mesh, P(None, "x", "y")))
    door = F.make_kfft_klead_unfold(mesh, kg, tables, norm="ortho")
    if np.any(np.asarray(tables.trs)):
        try:
            door(W)
        except ValueError as exc:
            assert "Wt is required" in str(exc)
        else:
            raise AssertionError("a pair-transpose door ran without its partner tile")
    short = fixtures._put(np.zeros((int(plan.n_parent) - 1, mu, mu), complex),
                          NamedSharding(mesh, P(None, "x", "y")))
    try:
        door(short, short)
    except ValueError as exc:
        assert "wedge rows" in str(exc)
    else:
        raise AssertionError("a wedge with a row too few was accepted")
