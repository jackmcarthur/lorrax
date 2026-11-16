"""``kgrid_fine`` dipoles: ``densified_dipoles`` against the exact fine-grid answer (CPU, 2x2 host mesh).

``bse.bse_densify.densified_dipoles`` on a toy tight-binding slab whose
orbitals are the htransform nodes (all at the cell origin, unit metric), so the
coarse states ARE ``ctilde`` and the exact fine-grid dipoles are one ``eigh``
away.  Run with
``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.

1. Every fine q that is a coarse k reproduces the coarse velocity in the
   fine states (the gate's identity), also when the fH band order moves a
   band across the window edge (the gate reads the fH's order; handing it the
   DFT order fires), and a permuted fine q list fires
   ``GATE densified_dipole_on_grid``; off the coarse grid the dipoles match
   the exact ones phase-coherently, in the densified states' own gauge.
2. A gauge planted on the coarse states AND their velocity changes nothing;
   planted on the velocity alone (the dipfix failure) it is seen, and the
   velocity image loses its locality.  Under a scissor the fine states come
   from the QP f(H) while ``r = v/ΔE`` keeps the velocity file's energies:
   the dipoles do not move with the QP energies, and dividing by the QP ΔE
   (the red twin) would.
3. ``qp_states=True`` gives ⟨c̃|r̂|ṽ⟩ of the fine states (CrI3 8→12: Σ|d|²
   0.986 with the Rayleigh-quotient ``v/ΔE``) to the htransform's
   interpolation floor, and the Rayleigh-quotient twin is more than 5x
   further off; ``r_window`` is the window position (Hermitian, matches the
   exact one off the diagonal, ``i`` times its c←v block IS ``d_alpha``).
4. ``full_window=True``: the window is Hermitian and matches the exact
   velocity in the densified states' gauge; the bra/ket-swapped window (the
   red twin) does not.
5. Refusals that must fire: ``window_dipoles`` on a densified bundle without
   its own dipoles, an undensified bundle with another k list, a dipole
   stamped for another WFN (or not stamped) or in an older format; the
   rotation's record (small valence↔conduction mixing, window sum kept) and
   the ``GATE densified_pair_rule`` refusal.

The pair rule is also pinned to ``tdagw.td_system.position_from_velocity``'s
(a copy of it lives in ``bse_densify``).
"""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from bandstructure.bse_setup import compute_wfns_fi  # noqa: E402
from bse.absorption_common import window_dipoles  # noqa: E402
from bse.bse_densify import (DENSIFIED_POSITION_FORM,  # noqa: E402
                             _position_from_velocity_rule, densified_dipoles)
from ffi.fft import xla_reference  # noqa: E402
from isdf.galerkin import NodalInterp  # noqa: E402

KC, KF = (6, 6, 1), (9, 9, 1)          # non-nested: 3x3 shared points
ONSITE = np.r_[np.linspace(-1.6, -0.7, 4), np.linspace(0.6, 1.5, 4), 5.0, 5.3]
NB, B_MIN, B_MAX, N_VAL = 8, 2, 6, 2   # fitted window, BSE window, valence rows
NV, LO, HI = N_VAL, B_MIN, B_MAX
HOP, SHIFT = 0.02, 0.1                 # smooth model and QP shifts of test 3


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _model(seed=3, hop=0.05):
    rng = np.random.default_rng(seed)
    m = len(ONSITE)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    h = {(0, 0): np.diag(ONSITE) + 0.02 * (a + a.conj().T)}
    for r in ((1, 0), (0, 1), (1, 1), (1, -1)):
        b = hop * (rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m)))
        h[r], h[(-r[0], -r[1])] = b, b.conj().T
    return h


def _grid(n):
    axes = [np.arange(v) / v for v in n]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return (g + 0.5) % 1.0 - 0.5


def _states(h, kpts, phases=None):
    """C (nk, NB, M) = ⟨orbital|ψ⟩, E (nk, NB), v (nk, 3, NB, NB) = ⟨m|∂H|n⟩."""
    cs, es, vs = [], [], []
    for i, k in enumerate(kpts):
        hk = sum(np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * x
                 for r, x in h.items())
        dk = [sum(1j * r[a] * np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1])) * x
                  for r, x in h.items()) for a in (0, 1)]
        e, u = np.linalg.eigh(hk)
        u = u[:, :NB] * (np.ones(NB) if phases is None else phases[i])[None]
        vk = np.stack([u.conj().T @ d @ u for d in dk] + [np.zeros((NB, NB))])
        cs.append(u.T); es.append(e[:NB]); vs.append(vk)
    return np.array(cs), np.array(es), np.array(vs)


def _sym(kpts):
    nk = len(kpts)
    return SimpleNamespace(
        irr_idx_k=np.arange(nk), sym_idx_k=np.zeros(nk, np.int32),
        sym_mats_k=np.zeros((2, 3, 3)), nk_red=nk, nk_tot=nk,
        unfolded_kpts=np.asarray(kpts),
        cartesian_action=lambda rows, axial, time_odd: np.broadcast_to(
            np.eye(3), (len(np.asarray(rows)), 3, 3)))


def _nodes(c):
    """The toy's orbitals as co-located nodes; their nodal values are ``c``."""
    m = int(c.shape[2])
    return NodalInterp(jnp.asarray(c), np.zeros((m, 3)),
                       np.diag([1.0, 1.0, 10.0]), 2, 1)


def _densify(mesh, c, e, v, *, fh_energies=None, e_file=None, q_perm=None,
             gate_energies=None, band_offset=0, **kw):
    """``compute_wfns_fi`` + ``densified_dipoles`` on the toy, XLA backend."""
    kc = _grid(KC)
    e_fh = e if fh_energies is None else fh_energies
    metadata_e = e if e_file is None else e_file
    if band_offset:
        metadata_e = np.concatenate((np.full((len(e), band_offset), -100.),
                                     metadata_e, np.full((len(e), 1), 100.)), axis=1)
    nodes = _nodes(c)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c), B_at_mu=nodes,
            enk_sigma=jnp.asarray(e_fh.T),
            kgrid_co=KC, kgrid_fine=KF, band_window_fi=(B_MIN, B_MAX),
            mesh_xy=mesh, return_coeffs=True)
        if q_perm is not None:
            fine.q_frac = fine.q_frac[q_perm]
        dip = densified_dipoles(
            (v, metadata_e, "toy dH/dk", "toy.h5"),
            sym=_sym(kc), fh_energies=e_fh if gate_energies is None else gate_energies,
            basis=SimpleNamespace(ctilde=jnp.asarray(c),
                                  band_range=(band_offset, band_offset+NB),
                                  basis_at_nodes=nodes),
            fine=fine, coarse_grid=KC, n_val=N_VAL, window_start=B_MIN,
            a_band_index=None,
            mesh_xy=mesh, log_fn=lambda *a: None, **kw)
    return dip, fine


def _exact_rotated(h, fine):
    """Exact d at the fine q, rotated into the densified states' own gauge."""
    ce, ee, ve = _states(h, fine.q_frac)
    lo, hi = B_MIN, B_MAX
    de = ee[:, lo + N_VAL:hi, None] - ee[:, None, lo:lo + N_VAL]
    dx = ve[:, :, lo + N_VAL:hi, lo:lo + N_VAL] / de[:, None]
    w = np.einsum("qna,qam->qnm", ce[:, lo:hi].conj(), np.asarray(fine.coeffs_fi))
    return np.einsum("qcd,qxce,qef->xqdf", w[:, N_VAL:, N_VAL:].conj(), dx,
                     w[:, :N_VAL, :N_VAL]), dx


def _rel(a, b):
    return float(np.abs(a - b)[:2].max() / np.abs(b)[:2].max())


def test_on_grid_identity_and_off_grid_dipoles():
    mesh, h = _mesh(), _model()
    kc = _grid(KC)
    c, e, v = _states(h, kc, phases=np.exp(2j * np.pi * np.random.default_rng(
        11).random((np.prod(KC), NB))))
    dip, fine = _densify(mesh, c, e, v)
    assert dip["n_on_grid"] == 9 and dip["on_grid_rel"] < 1e-12
    ref, _ = _exact_rotated(h, fine)
    assert dip["d_alpha"].shape == (3, np.prod(KF), B_MAX - B_MIN - N_VAL, N_VAL)
    # The htransform's own interpolation error on this 6x6 toy is ~1%.
    assert _rel(dip["d_alpha"], ref) < 0.02
    assert np.all(dip["de_cv"] > 0)

    # A permuted fine q list fires the on-grid gate.
    perm = np.roll(np.arange(np.prod(KF)), 1)
    with pytest.raises(ValueError, match="GATE densified_dipole_on_grid"):
        _densify(mesh, c, e, v, q_perm=perm)

    # A QP ladder that moves a band across the window edge at a shared k.
    # The fH's own order picks the window there (MoS2 9x9 eqp1 at Gamma swaps
    # DFT bands 18-19 out of [18, 34) for 16-17), so the on-grid state
    # identity reads the window rows from ``fh_energies``; the red twin hands
    # the gate the DFT order and must fire.
    i0 = int(np.flatnonzero(np.all(np.abs(kc) < 1e-12, axis=1))[0])
    e_qp = e.copy()
    e_qp[i0, B_MIN - 1], e_qp[i0, B_MIN] = e[i0, B_MIN] + 0.02, e[i0, B_MIN - 1]
    assert list(np.argsort(e_qp[i0])[B_MIN:B_MAX]) != list(range(B_MIN, B_MAX))
    dip, _ = _densify(mesh, c, e, v, fh_energies=e_qp)
    assert dip["n_on_grid"] == 9
    assert dip["on_grid_state_dev"] < 1e-8 and dip["on_grid_rel"] < 1e-12
    with pytest.raises(ValueError, match="GATE densified_dipole_on_grid"):
        _densify(mesh, c, e, v, fh_energies=e_qp, gate_energies=e)


def _densify_qp(mesh, c, c_qp, e, e_qp, v, *, state_rows=True, rotated_nodes=True):
    """The densifier's QP route on the toy: fH on C_QP / E_QP and the QP states at
    the nodes (the back-map targets), velocity on C / E with the DFT nodes.

    The rank axis is zero-padded to the mesh (the Galerkin carrier).  The toy's
    orbitals are its nodes, so the node states are the compact rows.
    """
    kc = _grid(KC)
    pad = (-c.shape[2]) % int(mesh.size)
    c, c_qp = (np.pad(x, ((0, 0), (0, 0), (0, pad))) for x in (c, c_qp))
    nodes = _nodes(c)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c_qp), B_at_mu=_nodes(c_qp) if rotated_nodes else nodes,
            enk_sigma=jnp.asarray(e_qp.T),
            kgrid_co=KC, kgrid_fine=KF, band_window_fi=(B_MIN, B_MAX),
            mesh_xy=mesh, return_coeffs=True)
        dip = densified_dipoles(
            (v, e, "toy dH/dk", "toy.h5"), sym=_sym(kc), fh_energies=e_qp,
            basis=SimpleNamespace(ctilde=jnp.asarray(c), band_range=(0, NB),
                                  basis_at_nodes=nodes),
            fine=fine, coarse_grid=KC, n_val=N_VAL, window_start=B_MIN,
            a_band_index=None, mesh_xy=mesh, log_fn=lambda *a: None,
            qp_states=True,
            state_ctilde=jnp.asarray(c_qp) if state_rows else None)
    return dip, fine


def test_qp_rotated_compact_states_keep_the_dft_velocity():
    """``qp_rotations_file`` under ``kgrid_fine``: the fH takes C_QP = Uᵀ C (the
    production ``fh_interp`` overlay) and the velocity stays in its DFT rows.

    Cᵀ v C* = C_QPᵀ (Uᴴ v U) C_QP* for a unitary U, so the operator image needs no
    rotation.  At the shared k the fine states are the QP states with E_QP, and
    for a U inside the window's valence and conduction spans the r-form dipoles
    are the DFT ones rotated by U.  A U that mixes a band across the window edge
    passes the on-grid state identity against the QP rows and fires it against
    the DFT rows (the red twin).  A U across the gap gives fine psi at the shared
    k in the QP valence and conduction spans separately; with the unrotated
    states at the back-map targets (the red twin) psi stays the DFT orbitals.
    """
    from bandstructure.fh_interp import _apply_qp_block_to_compact_state
    mesh, h = _mesh(), _model()
    kc = _grid(KC)
    nk = len(kc)
    c, e, v = _states(h, kc)
    e_qp = e + SHIFT * (np.arange(NB) >= B_MIN + N_VAL)[None, :]
    for pairs, crosses in (([(2, 3, 0.4), (4, 5, 0.3)], False),
                           ([(1, 2, 0.35), (4, 5, 0.3)], True),
                           ([(3, 4, 0.3)], "gap")):
        U = np.tile(np.eye(NB, dtype=complex), (nk, 1, 1))
        for a, b, t in pairs:
            U[:, a, a] = U[:, b, b] = np.cos(t)
            U[:, a, b] = -np.exp(-0.7j) * np.sin(t)
            U[:, b, a] = np.exp(0.7j) * np.sin(t)
        np.testing.assert_allclose(np.einsum("kmn,kml->knl", U.conj(), U),
                                   np.tile(np.eye(NB), (nk, 1, 1)), atol=1e-14)
        c_qp, _ = _apply_qp_block_to_compact_state(
            jnp.asarray(c), jnp.asarray(e.T), jnp.asarray(U), jnp.asarray(e_qp),
            band_offset=0)
        c_qp = np.asarray(c_qp)
        v_qp = np.einsum("kmn,kxmp,kpl->kxnl", U.conj(), v, U)
        np.testing.assert_allclose(
            np.einsum("kma,kxmp,kpb->kxab", c_qp, v_qp, c_qp.conj()),
            np.einsum("kma,kxmp,kpb->kxab", c, v, c.conj()), atol=1e-13)
        dip, fine = _densify_qp(mesh, c, c_qp, e, e_qp, v)
        assert dip["n_on_grid"] == 9
        assert dip["on_grid_state_dev"] < 1e-8 and dip["on_grid_rel"] < 1e-12
        hit = [(i, np.flatnonzero(np.all(np.abs((kc - q + 0.5) % 1.0 - 0.5) < 1e-8,
                                         axis=1))) for i, q in enumerate(fine.q_frac)]
        shared = [i for i, j in hit if j.size]
        ko = [int(j[0]) for _, j in hit if j.size]
        np.testing.assert_allclose(np.asarray(fine.enk_full)[shared],
                                   np.sort(e_qp[ko], axis=1)[:, B_MIN:B_MAX], atol=1e-10)
        if crosses == "gap":
            def spans(f):
                psi = np.asarray(f.psi_rmu_Y)[shared][:, :, 0, :c.shape[2]]
                o = np.einsum("qna,qma->qnm", c_qp[ko, B_MIN:B_MAX].conj(), psi)
                return [np.linalg.svd(o[:, sl, sl], compute_uv=False)
                        for sl in (slice(0, N_VAL), slice(N_VAL, None))]
            for sv in spans(fine):
                np.testing.assert_allclose(sv, 1.0, atol=1e-8)
            _, twin = _densify_qp(mesh, c, c_qp, e, e_qp, v, rotated_nodes=False)
            assert min(float(sv.min()) for sv in spans(twin)) < 0.99
            continue
        if crosses:
            with pytest.raises(ValueError, match="GATE densified_dipole_on_grid"):
                _densify_qp(mesh, c, c_qp, e, e_qp, v, state_rows=False)
            continue
        cf = np.asarray(fine.coeffs_fi)[shared][:, :c.shape[2]]
        w = np.einsum("qna,qam->qnm", c[ko, B_MIN:B_MAX].conj(), cf)
        de = e[ko, B_MIN + N_VAL:B_MAX, None] - e[ko, None, B_MIN:B_MIN + N_VAL]
        dx = v[ko][:, :, B_MIN + N_VAL:B_MAX, B_MIN:B_MIN + N_VAL] / de[:, None]
        ref = np.einsum("qcd,qxce,qef->xqdf", w[:, N_VAL:, N_VAL:].conj(), dx,
                        w[:, :N_VAL, :N_VAL])
        assert _rel(np.asarray(dip["d_alpha"])[:, shared], ref) < 1e-8


def test_kgrid_fine_refuses_a_zero_head():
    """GATE kgrid_fine_zero_head fires on a head-off ISDF tensor file and passes a head-on one."""
    from bse.bse_loading import refuse_zero_head_densify
    with pytest.raises(ValueError, match="GATE kgrid_fine_zero_head"):
        refuse_zero_head_densify(0.0, (8, 8, 8))
    refuse_zero_head_densify(363.144629, (8, 8, 8))
