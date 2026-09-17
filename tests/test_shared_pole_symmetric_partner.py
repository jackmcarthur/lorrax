"""The ordered mirror on a symmetric deck: −q of a parent is another parent's image under a unitary row.

Plant: a time-reversal-broken triangular lattice (3 sites, one C3 orbit on a 6x6 FFT grid, generic
complex hoppings averaged over C3 and the 4x4 Born-von Karman translations), its exact supercell
RPA pencil W_c(z) = C (z s3 - M)^-1 C^H, and W_q = FT_q[W_c] in the orientation of
``tr_broken_lattice.ft_q`` (the physical orientation of the ordered store). The q-IBZ is
``find_irreducible_bz_points`` under C3; the centroid permutation and umklapp wraps are
``centroid_source_map_and_wrap``; realization is the Σ unfold body ``unfold_operator_local``.

Known answer at nonzero complex frequency, for every parent p with partner (p', s) from
``minus_q_parent_partners``:
    R_s[W_q(p')(z)] = W_{-q(p)}(z)             (direct FT at -q)
    conj(R_s[W_q(p')(z)]) = W_q(p)(-conj z)    (the ordered mirror the constructor needs)
RED TWINS: the identity partner (p, E) and the mirror without conjugation both miss by > 1e-2; the
time-reversal-augmented table on the same plant refuses by name.
"""
import numpy as np
import pytest

from tr_broken_lattice import ft_q, w_c

N1 = N2 = 4
FFT = np.array([6, 6, 1])
M2 = np.array([[-1, -1], [1, 0]])                    # C3 in crystal coordinates: a1 -> a2 - a1, a2 -> -a1
SITES = np.array([[1, 2], [3, 1], [2, 3]]) / 6.0     # one C3 orbit on the 6x6 grid (wraps differ)
ZS = (0.7 + 0.3j, -1.2 + 0.1j, 0.4 + 1.1j)


class _SymLattice:
    """Supercell arrays for ``ft_q``/``w_c``; H and v are C3 and translation covariant."""

    def __init__(self, seed=20260917):
        rng = np.random.default_rng(seed)
        self.n1, self.n2, self.ns = N1, N2, len(SITES)
        self.a = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        self.cells = np.array([(i, j) for i in range(N1) for j in range(N2)])
        self.kfrac = np.array([(i / N1, j / N2) for i in range(N1) for j in range(N2)])
        self.nk = N1 * N2
        self.b = 2 * np.pi * np.linalg.inv(self.a).T
        frac = np.array([c + x for c in self.cells for x in SITES])          # supercell fractional coords
        n = len(frac)
        index = {tuple(np.rint(6 * p).astype(int) % (6 * np.array([N1, N2]))): i for i, p in enumerate(frac)}

        def perm_of(transform):
            return np.array([index[tuple(np.rint(6 * transform(p)).astype(int) % (6 * np.array([N1, N2])))]
                            for p in frac])
        group = [perm_of(lambda p, t=t, g=g: np.linalg.matrix_power(M2, g) @ p + t)
                 for t in self.cells for g in range(3)]
        h0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        h0 = 0.35 * (h0 + h0.conj().T)
        h = sum(h0[np.ix_(g, g)] for g in group) / len(group)
        self.h = 0.5 * (h + h.conj().T)
        pos = frac @ self.a
        L = np.array([N1 * self.a[0], N2 * self.a[1]])
        v = np.zeros((n, n))
        for i in range(n):
            d = pos - pos[i]
            dist = np.min([np.linalg.norm(d + m1 * L[0] + m2 * L[1], axis=1)
                           for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)], axis=0)
            v[i] = np.exp(-dist / 0.6) / (1.0 + dist)
        self.v = 0.5 * (v + v.T) + 1e-3 * np.eye(n)
        self.frac, self.group = frac, group

    def kvec(self, kf):
        return np.asarray(kf, float) @ self.b

    def pencil(self, n_occ=1):
        """Per-k Bloch sectors, lowest n_occ bands occupied, supercell RPA pencil."""
        pos = self.frac @ self.a
        site_of = np.tile(np.arange(self.ns), self.nk)
        occ, emp = [], []
        for kf in self.kfrac:
            basis = np.exp(1j * pos @ self.kvec(kf))[:, None] * (site_of[:, None] == np.arange(self.ns)[None, :])
            basis /= np.sqrt(self.nk)
            e, u = np.linalg.eigh(basis.conj().T @ self.h @ basis)
            states = basis @ u
            occ += [(e[i], states[:, i]) for i in range(n_occ)]
            emp += [(e[i], states[:, i]) for i in range(n_occ, self.ns)]
        rho = np.array([np.conj(pi) * pa for _, pi in occ for _, pa in emp]).T
        D = np.array([ea - ei for ei, _ in occ for ea, _ in emp])
        phi = np.hstack([rho, rho.conj()])
        M = np.diag(np.r_[D, D]).astype(complex) + phi.conj().T @ self.v @ phi
        s3 = np.diag(np.r_[np.ones(len(D)), -np.ones(len(D))]).astype(complex)
        return dict(c=self.v @ phi, s3=s3, M=0.5 * (M + M.conj().T))


def _tables(time_reversal_rows=False):
    from symmetry_maps import centroid_source_map_and_wrap, find_irreducible_bz_points
    m3 = np.eye(3, dtype=np.int64)
    m3[:2, :2] = M2
    mtrx = [np.eye(3, dtype=np.int64), np.rint(np.linalg.inv(m3 @ m3)).astype(np.int64),
            np.rint(np.linalg.inv(m3)).astype(np.int64)]                    # r-action mtrx^-1 = C3^g
    mats_k = np.stack([m.T for m in mtrx])
    if time_reversal_rows:
        mats_k = np.concatenate([mats_k, -mats_k])
    grid = (N1, N2, 1)
    coords = np.stack(np.unravel_index(np.arange(N1 * N2), grid), axis=1)
    irr, sym, reps = find_irreducible_bz_points(coords, mats_k)
    full = np.array([int(np.ravel_multi_index(tuple(r), grid)) for r in reps])
    cent = np.column_stack([np.rint(6 * SITES).astype(np.int32), np.zeros(len(SITES), np.int32)])
    perm, wraps = centroid_source_map_and_wrap(cent, np.stack(mtrx), np.zeros((3, 3)), FFT, validate=True,
                                               extend_trs=time_reversal_rows)
    return dict(grid=grid, coords=coords, irr=irr, sym=sym, full=full, mats_k=mats_k, perm=perm, wraps=wraps,
                antiunitary=np.arange(len(mats_k)) >= 3)


def _realize(operator, row, q_frac, t):
    """R_row[operator] through the Σ unfold body on a 1x1 host mesh."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P
    from common.shard_map import shard_map
    from symmetry_maps import unfold_operator_local
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    body = shard_map(lambda a: unfold_operator_local(
        a, irr_idx=np.array([0]), sym_idx=np.array([row]), q_irr_frac=np.r_[np.asarray(q_frac, float), 0.0][None],
        left_local_perm=t["perm"], left_L_table=t["wraps"], right_local_perm=t["perm"], right_L_table=t["wraps"],
        n_sym_spatial=3), mesh=mesh, in_specs=P(None, "x", "y"), out_specs=P(None, "x", "y"), check_vma=False)
    return np.asarray(jax.jit(body)(jnp.asarray(operator[None])))[0]


def test_symmetric_partner_realizes_the_ordered_mirror_at_complex_frequency():
    from symmetry_maps import minus_q_parent_partners, q_negation_index
    t = _tables()
    partner, row = minus_q_parent_partners(t["full"], t["irr"], t["sym"], kgrid=t["grid"],
                                           sym_mats_k=t["mats_k"], antiunitary=t["antiunitary"],
                                           authorized_rows=np.arange(3))
    # The plant must reach the case production symmetric decks create: a non-identity unitary row.
    assert np.any(row != 0) and np.any(partner != np.arange(len(t["full"])))
    lat = _SymLattice()
    pencil = lat.pencil()
    neg = q_negation_index(t["grid"])
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    q_frac = t["coords"][:, :2] / np.array([N1, N2])
    worst, twin_identity, twin_unconjugated, asymmetry = 0.0, 0.0, 0.0, 0.0
    for z in ZS:
        wz, wm = w_c(z, pencil), w_c(-np.conj(z), pencil)
        for p, q in enumerate(t["full"]):
            parent = ft_q(wz, lat, q_frac[t["full"][partner[p]]])
            child = _realize(parent, int(row[p]), q_frac[t["full"][partner[p]]], t)
            minus = ft_q(wz, lat, q_frac[neg[q]])
            mirror = ft_q(wm, lat, q_frac[q])
            asymmetry = max(asymmetry, rel(minus.T, minus))
            worst = max(worst, rel(child, minus), rel(np.conj(child), mirror))
            own = _realize(ft_q(wz, lat, q_frac[q]), 0, q_frac[q], t)
            twin_identity = max(twin_identity, rel(np.conj(own), mirror))
            twin_unconjugated = max(twin_unconjugated, rel(child, mirror))
    assert worst < 1e-12, worst
    assert asymmetry > 1e-2, asymmetry          # the plant carries no hidden transpose reciprocity
    assert twin_identity > 1e-2 and twin_unconjugated > 1e-2, (twin_identity, twin_unconjugated)


def test_time_reversal_augmented_table_on_a_magnet_refuses_the_partner():
    from symmetry_maps import minus_q_parent_partners
    t = _tables(time_reversal_rows=True)
    with pytest.raises(ValueError, match="GATE minus_q_partner.*antiunitary"):
        minus_q_parent_partners(t["full"], t["irr"], t["sym"], kgrid=t["grid"], sym_mats_k=t["mats_k"],
                                antiunitary=t["antiunitary"], authorized_rows=np.arange(6))
