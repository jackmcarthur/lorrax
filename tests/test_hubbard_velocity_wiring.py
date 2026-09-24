"""i[r, V_U] in the ONE velocity path: common.mtxel_sweep.dipole_operator(hubbard=...).

Three gates, all synthetic (a VI3-shaped l=0 + l=2 atomic basis on two sites, random Hermitian W):

* P4 PARITY (``@pytest.mark.mesh(4)``): the band-sharded sweep on a 2x2 mesh, NB = 6 (not a
  multiple of 4, so the padded band carrier runs -- TASTE 11), equals a single-device reference
  that shares no apply-to-ket code with it: p from ``momentum_matrix_k``, V_NL from the matrix-form
  ``vnl_velocity_matrix``, V_U from the matrix-form ``hubbard_matrix_and_velocity``.
* RED TWINS: the same comparison against a reference with (a) W transposed (the ns vs ns^T
  convention QE's new_ns_nc makes easy to get wrong), (b) the Loewdin derivative dropped, or
  (c) the atomic rows left unmasked on the D10 pad columns must FAIL by orders of magnitude --
  so the parity gate can see each defect class it exists for (TASTE 21).
* U = 0 REGRESSION: ``hubbard=None`` keeps the pre-Hubbard operator key and its output is
  bit-identical to the literal p + dV_NL expression (every non-DFT+U deck is unchanged).
Plus the resolver's refusals (DFT+U declared without input; input without DFT+U; card/schema
mismatch) on a synthetic QE schema, and the card parser's QE formulation rule.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.mtxel_sweep import (SweepGeometry, _operator_key, blocks_to_host,
                                dipole_operator, sweep_matrix_elements)
from psp import hubbard_ops as ho
from psp import vnl_ops
from psp.dft_operators import momentum_matrix_k

jax.config.update("jax_enable_x64", True)

NK, NB, NS, NGK, NGKMAX = 2, 6, 2, 120, 124
GRID = (6, 6, 6)
NATOM = 2
LD = 5
TOL = 1e-12


def _atwfc_setup(seed=5):
    """l=0 and l=2 atomic rows on two sites, vnl_ops row order (channel -> atom -> beta -> m)."""
    rng = np.random.default_rng(seed)
    n_q, dq = 400, 0.02
    q = np.arange(n_q) * dq
    tau = rng.uniform(0, 1, size=(NATOM, 3))
    G = np.stack([np.exp(-0.5 * (q / 1.1) ** 2), np.exp(-0.5 * ((q - 0.9) / 0.7) ** 2)])
    Gp = np.stack([-(q / 1.1 ** 2) * G[0], -((q - 0.9) / 0.7 ** 2) * G[1]])
    chans = [vnl_ops.ChannelMeta(l=0, nbeta=1, msize=1, R=1, tau=tau, E=np.zeros((2, 2, 1, 1)),
                                 beta_table_start=0, natoms=NATOM),
             vnl_ops.ChannelMeta(l=2, nbeta=1, msize=5, R=5, tau=tau, E=np.zeros((2, 2, 5, 5)),
                                 beta_table_start=1, natoms=NATOM)]
    rb, rl, rm, rt = [], [], [], []
    for ci, ch in enumerate(chans):
        for a in range(NATOM):
            for m in range(ch.msize):
                rb.append(ci); rl.append(ch.l); rm.append(m); rt.append(tau[a])
    R = len(rb)
    setup = vnl_ops.VNLSetup(
        channels=chans, dq=dq, n_q=n_q, q_max=n_q * dq, G_table=jnp.asarray(G), Gp_table=jnp.asarray(Gp),
        prefactor=1.0, B=np.eye(3) * 1.17, cell_volume=1.0, total_R=R, nspinor=2,
        E_super=jnp.zeros((2, 2, R, R), dtype=jnp.complex128), l_max=2,
        row_beta_idx=jnp.asarray(rb, dtype=jnp.int32), row_l=jnp.asarray(rl, dtype=jnp.int32),
        row_m=jnp.asarray(rm, dtype=jnp.int32), row_tau=jnp.asarray(np.asarray(rt)))
    hub_rows = np.asarray([[2 + 5 * a + m for m in range(5)] for a in range(NATOM)], dtype=np.int32)
    A = rng.normal(size=(NATOM, 2 * LD, 2 * LD)) + 1j * rng.normal(size=(NATOM, 2 * LD, 2 * LD))
    W = 0.5 * (A + np.conj(np.transpose(A, (0, 2, 1))))
    return ho.HubbardSetup(atwfc=setup, row_labels=[], hub_rows=hub_rows,
                           hub_atoms=np.arange(NATOM, dtype=np.int32), W=W, l=2)


def _vnl_setup(seed=9):
    rng = np.random.default_rng(seed)
    n_q, dq = 400, 0.02
    q = np.arange(n_q) * dq
    tau = rng.uniform(0, 1, size=(NATOM, 3))
    G = np.exp(-0.5 * (q / 0.8) ** 2)[None]
    Gp = (-(q / 0.64) * G[0])[None]
    ch = vnl_ops.ChannelMeta(l=1, nbeta=1, msize=3, R=3, tau=tau, E=np.zeros((2, 2, 3, 3)),
                             beta_table_start=0, natoms=NATOM)
    R = 3 * NATOM
    A = rng.normal(size=(2, 2, R, R)) + 1j * rng.normal(size=(2, 2, R, R))
    E = 0.5 * (A + np.conj(np.transpose(A, (1, 0, 3, 2))))
    return vnl_ops.VNLSetup(
        channels=[ch], dq=dq, n_q=n_q, q_max=n_q * dq, G_table=jnp.asarray(G), Gp_table=jnp.asarray(Gp),
        prefactor=1.0, B=np.eye(3) * 1.17, cell_volume=1.0, total_R=R, nspinor=2,
        E_super=jnp.asarray(E), l_max=1,
        row_beta_idx=jnp.zeros(R, dtype=jnp.int32), row_l=jnp.ones(R, dtype=jnp.int32),
        row_m=jnp.asarray(np.tile(np.arange(3), NATOM), dtype=jnp.int32),
        row_tau=jnp.asarray(np.repeat(tau, 3, axis=0)))


def _fixture(seed=3):
    rng = np.random.default_rng(seed)
    nx, ny, nz = GRID
    gv = np.zeros((NK, NGKMAX, 3), dtype=np.int32)
    bidx = np.zeros((NK,) + GRID, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(nx * ny * nz, size=NGK, replace=False)
        for i, c in enumerate(cells):
            g = np.array([c // (ny * nz), (c // nz) % ny, c % nz])
            gv[ik, i] = np.where(g > 2, g - 6, g)
            bidx[ik, g[0], g[1], g[2]] = i
        gv[ik, NGK:] = [0, 0, 0]          # production pad rows are G = 0: Z is LARGE there
    gmask = np.zeros((NK, NGKMAX)); gmask[:, :NGK] = 1.0
    psi = rng.normal(size=(NK, NB, NS, NGKMAX)) + 1j * rng.normal(size=(NK, NB, NS, NGKMAX))
    psi[..., NGK:] = 0.0
    kvecs = rng.normal(size=(NK, 3)) * 0.2
    return psi, gv, gmask, bidx, kvecs, np.eye(3) * 1.17, 1.0


def _reference(psi, gv, gmask, kvecs, B, vset, hub, *, transpose_W=False, drop_lowdin_derivative=False,
               unmasked=False):
    """Single-device matrix-form reference, (nk, 3, nb, nb)."""
    out = []
    W = np.transpose(hub.W, (0, 2, 1)) if transpose_W else hub.W
    for ik in range(NK):
        ket = jnp.asarray(psi[ik] * gmask[ik][None, None, :])
        v = np.asarray(momentum_matrix_k(ket, jnp.asarray(gv[ik]), jnp.asarray(kvecs[ik]), jnp.asarray(B)))
        kd = vnl_ops.build_vnl_kdata_traced(jnp.asarray(kvecs[ik]), jnp.asarray(gv[ik]), vset, compute_dZ=True)
        v = v + np.asarray(vnl_ops.vnl_velocity_matrix(ket, kd.Z, kd.dZ, kd.E_super))
        ka = vnl_ops.build_vnl_kdata_traced(jnp.asarray(kvecs[ik]), jnp.asarray(gv[ik]), hub.atwfc, compute_dZ=True)
        m = jnp.asarray(np.ones(NGKMAX) if unmasked else gmask[ik])
        Z, dZ = ka.Z * m[None, :], ka.dZ * m[None, None, :]
        Zt, dZt, lam = ho.lowdin_rows_with_derivative(Z, dZ, hub.hub_idx)
        assert float(np.min(lam)) > 1e-3, "synthetic atomic basis is ill-conditioned"
        if drop_lowdin_derivative:
            O = np.conj(np.asarray(Z)) @ np.asarray(Z).T
            lo, U = np.linalg.eigh(0.5 * (O + O.conj().T))
            C = (U / np.sqrt(lo)) @ U.conj().T
            dZt = jnp.asarray(np.einsum("rh,arG->ahG", C[:, np.asarray(hub.hub_idx)], np.asarray(dZ)))
        _, vU, _ = ho.hubbard_matrix_and_velocity(ket, Zt, dZt, jnp.asarray(W))
        out.append(v + np.asarray(vU))
    return np.stack(out)


def _sweep(mesh, psi, gv, gmask, bidx, kvecs, bvec, blat, vset, hub):
    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB, ns=NS, nk=NK, cell_volume=1.0)
    nbp = geom.nb
    psi_p = np.zeros((NK, nbp, NS, NGKMAX), dtype=np.complex128); psi_p[:, :NB] = psi
    psi_j = jax.device_put(jnp.asarray(psi_p), NamedSharding(mesh, P(None, ('x', 'y'), None, None)))
    op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vset, vnl_velocity_sign=+1.0, hubbard=hub)
    blk = sweep_matrix_elements(psi_j, geom=geom, operator=op, gvecs=jnp.asarray(gv), gmask=jnp.asarray(gmask),
                                box_index=jnp.asarray(bidx), kvecs=jnp.asarray(kvecs))
    return np.asarray(blocks_to_host(blk, nb=NB)), geom


def _mesh(n):
    devs = jax.devices()
    if len(devs) < n * n:
        pytest.skip(f"needs {n * n} devices, have {len(devs)}")
    return Mesh(np.asarray(devs[:n * n]).reshape(n, n), ("x", "y"))


@pytest.mark.mesh(4)
def test_p4_sharded_hubbard_velocity_matches_single_device_reference_and_red_twins_fail():
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    vset, hub = _vnl_setup(), _atwfc_setup()
    B = bvec * blat
    H4, geom = _sweep(_mesh(2), psi, gv, gmask, bidx, kvecs, bvec, blat, vset, hub)
    assert geom.nb != NB, "the band carrier must be padded (NB not divisible by P)"
    ref = _reference(psi, gv, gmask, kvecs, B, vset, hub)
    scale = float(np.max(np.abs(ref)))
    err = float(np.max(np.abs(H4 - ref)))
    assert err < TOL * scale, f"P4 sweep vs single-device reference: {err:.3e} (scale {scale:.3e})"
    herm = float(np.max(np.abs(H4 - np.conj(np.swapaxes(H4, -1, -2)))))
    assert herm < TOL * scale, herm
    vU_scale = float(np.max(np.abs(ref - _reference(psi, gv, gmask, kvecs, B, vset,
                                                    ho.HubbardSetup(hub.atwfc, [], hub.hub_rows, hub.hub_atoms,
                                                                    0 * hub.W, 2)))))
    assert vU_scale > 1e-2 * scale, "V_U must be a visible part of this fixture's velocity"
    for kw in ({"transpose_W": True}, {"drop_lowdin_derivative": True}, {"unmasked": True}):
        red = float(np.max(np.abs(H4 - _reference(psi, gv, gmask, kvecs, B, vset, hub, **kw))))
        assert red > 1e3 * TOL * scale, f"red twin {kw} passed the parity gate ({red:.3e})"


def test_no_hubbard_is_the_pre_hubbard_operator_bitwise():
    psi, gv, gmask, bidx, kvecs, bvec, blat = _fixture()
    vset = _vnl_setup()
    mesh = _mesh(1)
    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB, ns=NS, nk=NK, cell_volume=1.0)
    op = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vset, vnl_velocity_sign=+1.0)
    op_none = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vset, vnl_velocity_sign=+1.0, hubbard=None)
    assert _operator_key(op_none) == ('dipole', NGKMAX, NS, 1.0, id(vset), 1.0) == _operator_key(op)
    op_u = dipole_operator(geom, bvec=bvec, blat=blat, vnl_setup=vset, vnl_velocity_sign=+1.0,
                           hubbard=_atwfc_setup())
    assert _operator_key(op_u) != _operator_key(op), "a Hubbard operator must not share the plain jit"
    from psp.dft_operators import apply_kinetic_velocity_to_ket
    for ik in range(NK):
        args = (jnp.asarray(psi[ik])[None], jnp.asarray(gv[ik]), jnp.asarray(gmask[ik]),
                jnp.asarray(bidx[ik])[None], jnp.asarray(kvecs[ik]))
        got = np.asarray(op_none.apply(*args, *op_none.consts))
        ket = jnp.asarray(psi[ik]) * jnp.asarray(gmask[ik])[None, None, :]
        B = jnp.asarray(bvec * blat)
        v = apply_kinetic_velocity_to_ket(ket, jnp.asarray(gv[ik]), jnp.asarray(kvecs[ik]), B)
        kd = vnl_ops.build_vnl_kdata_traced(jnp.asarray(kvecs[ik]), jnp.asarray(gv[ik]), vset, compute_dZ=True)
        v = v + vnl_ops.apply_vnl_velocity_to_ket(ket, kd.Z, kd.dZ, kd.E_super)
        want = np.asarray(jnp.moveaxis(v, 0, -1)[None])
        assert np.array_equal(got, want), "hubbard=None must execute the literal p + dV_NL ket"


def test_legacy_vnl_sign_with_hubbard_refuses():
    mesh = _mesh(1)
    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=NGKMAX, nb=NB, ns=NS, nk=NK, cell_volume=1.0)
    with pytest.raises(ValueError, match="GATE dftu_velocity_sign"):
        dipole_operator(geom, bvec=np.eye(3), blat=1.0, vnl_setup=_vnl_setup(), vnl_velocity_sign=-1.0,
                        hubbard=_atwfc_setup())


# ---------------------------------------------------------------------------
# resolver: where the data come from, and when a run refuses
# ---------------------------------------------------------------------------

_CARD = """&CONTROL
  calculation = 'nscf'
/
&SYSTEM
  nat = 3
/
ATOMIC_SPECIES
V  50.9415  V.upf
I 126.90447  I.upf
ATOMIC_POSITIONS bohr
V 0 0 0
I 1 0 0
V 0 1 0
K_POINTS automatic
2 2 1 0 0 0
HUBBARD (ortho-atomic)
U V-3d 6.0
J V-3d 0.8
"""


def _schema(tmp_path, *, dftu=True, kind=1, U_ha=6.0 / ho.HARTREE_EV):
    blk = (f"<dftU><lda_plus_u_kind>{kind}</lda_plus_u_kind><Hubbard_U specie=\"V\" label=\"3d\">{U_ha!r}"
           "</Hubbard_U><U_projection_type>ortho-atomic</U_projection_type></dftU>") if dftu else ""
    p = tmp_path / "data-file-schema.xml"
    p.write_text(f"<espresso><input><dft>{blk}</dft></input><output></output></espresso>")
    return types.SimpleNamespace(schema_path=str(p))


def _wfn(binding):
    return types.SimpleNamespace(qe_symmetry_binding=binding, nspinor=2,
                                 atom_types=np.asarray([23, 53, 23]))


def _occup(tmp_path, seed=1):
    rng = np.random.default_rng(seed)
    ns = np.zeros((5, 5, 4, 3), dtype=complex)
    for a in (0, 2):
        ns[:, :, :, a] = rng.normal(size=(5, 5, 4)) + 1j * rng.normal(size=(5, 5, 4))
    p = tmp_path / f"occup{seed}.txt"
    p.write_text("\n".join(f" ({z.real:.17g},{z.imag:.17g})" for z in ns.flatten(order="F")))
    return p


def test_resolver_rules(tmp_path):
    card = tmp_path / "nscf.in"; card.write_text(_CARD)
    occ = _occup(tmp_path)
    decl = _wfn(_schema(tmp_path))
    with pytest.raises(ValueError, match="GATE dftu_velocity_input"):
        ho.resolve_hubbard_input("", "", wfn=decl, base_dir=str(tmp_path), caller="t")
    with pytest.raises(ValueError, match="both keys, or neither"):
        ho.resolve_hubbard_input(str(card), "", wfn=decl, base_dir=str(tmp_path), caller="t")
    hi = ho.resolve_hubbard_input("nscf.in", occ.name, wfn=decl, base_dir=str(tmp_path), caller="t")
    assert hi.card.formulation == "liechtenstein" and hi.card.projector == "ortho-atomic"
    (sh,) = hi.shells
    assert (sh.element, sh.l, sh.atoms, sh.U_eV, sh.J_eV) == ("V", 2, (0, 2), 6.0, 0.8)
    assert abs(sh.B_eV - 0.114774114774 * 0.8) < 1e-12
    assert ho.hubbard_provenance_for("nscf.in", occ.name, wfn=decl, base_dir=str(tmp_path), caller="t") \
        == hi.provenance
    occ2 = _occup(tmp_path, seed=2)
    assert ho.resolve_hubbard_input("nscf.in", occ2.name, wfn=decl, base_dir=str(tmp_path),
                                    caller="t").provenance != hi.provenance
    plain = _wfn(None)
    assert ho.resolve_hubbard_input("", "", wfn=plain, base_dir=str(tmp_path), caller="t") is None
    nodftu = tmp_path / "plain"; nodftu.mkdir()
    with pytest.raises(ValueError, match="declares no DFT\\+U"):
        ho.resolve_hubbard_input(str(card), str(occ), wfn=_wfn(_schema(nodftu, dftu=False)),
                                 base_dir=str(tmp_path), caller="t")
    assert ho.resolve_hubbard_input("", "", wfn=_wfn(_schema(nodftu, dftu=False)),
                                    base_dir=str(tmp_path), caller="t") is None
    wrong = tmp_path / "wrong"; wrong.mkdir()
    with pytest.raises(ValueError, match="disagrees with the QE schema"):
        ho.resolve_hubbard_input(str(card), str(occ), wfn=_wfn(_schema(wrong, kind=0)),
                                 base_dir=str(tmp_path), caller="t")
    wrongU = tmp_path / "wrongU"; wrongU.mkdir()
    with pytest.raises(ValueError, match="disagrees with the QE schema"):
        ho.resolve_hubbard_input(str(card), str(occ), wfn=_wfn(_schema(wrongU, U_ha=5.0 / ho.HARTREE_EV)),
                                 base_dir=str(tmp_path), caller="t")


def test_card_formulation_and_unsupported_terms(tmp_path):
    p = tmp_path / "a.in"
    p.write_text(_CARD.replace("J V-3d 0.8\n", ""))
    assert ho.parse_qe_hubbard_card(p).formulation == "dudarev"
    p.write_text(_CARD.replace("J V-3d 0.8", "V V-3d I-5p 1 2 0.3"))
    with pytest.raises(ValueError, match="GATE dftu_input"):
        ho.parse_qe_hubbard_card(p)
    p.write_text(_CARD.replace("(ortho-atomic)", "(atomic)"))
    with pytest.raises(ValueError, match="ortho-atomic"):
        ho.parse_qe_hubbard_card(p)
