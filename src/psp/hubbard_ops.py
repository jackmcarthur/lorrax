r"""psp/hubbard_ops.py -- QE's DFT+U operator V_U(k) and its k-derivative.

It closes the ``i[r, V_U]`` gap of the DFT velocity ``v = p + i[r, V_NL]``
for QE ortho-atomic DFT+U, noncollinear, ``lda_plus_u_kind`` 0 (Dudarev) and
1 (Liechtenstein), norm-conserving pseudopotentials (S = 1).  Everything
else refuses by name (``GATE dftu_*``).

WIRING.  ``resolve_hubbard_input`` is the one rule for where the data come
from (deck keys ``hubbard_input`` = the pw.x input with the HUBBARD card,
``hubbard_occupations`` = its ``prefix.save/occup.txt``) and when a run
refuses; ``apply_hubbard_velocity_to_ket`` is the one ket kernel, called by
``common.mtxel_sweep.dipole_operator`` (q = 0, band-sharded sweep) and
``psp.get_dipole_mtxels.compute_finite_q_mtxels``.  The stamp
``hubbard_provenance`` rides in ``dipole.h5`` (``prov_hubbard``) and in the
parallel-transport velocity artifact (``hubbard_provenance_utf8``); the
consumers recompute it from their deck and refuse on a mismatch.

The operator (QE PW/src: orthoatwfc.f90, atomic_wfc_mod.f90 atomic_wfc_so_mag,
new_ns.f90 new_ns_nc, v_of_rho.f90 v_hubbard(_full)_nc, vhpsi.f90 vhpsi_U_nc)::

    V_U(k)   = sum_I  Phi~_I(k) W_I Phi~_I(k)^+
    Phi~(k)  = Phi(k) O(k)^{-1/2},      O = Phi^+ Phi                  (Loewdin, ALL natomwfc)
    W_I[(m1,s1),(m2,s2)] = v%ns_nc(m1, m2, 2*s1+s2, I)                 (Ry; row order m + ldim*s)
    ns(m1,m2,2*s1+s2)    = sum_nk f_nk conj(P_{m1 s1}) P_{m2 s2},  P = Phi~_I^+ psi

``Phi(k)`` is every PP_PSWFC function with occupation >= 0, j-averaged
``(chi_{l+1/2}(l+1) + chi_{l-1/2} l)/(2l+1)`` for a spin-orbit file, times the
QE real Y_lm, in pure spin-up and spin-down spinors.  Because the spinors are
pure, ``O`` is spin-block diagonal with two identical blocks, so the Loewdin
transform acts on the SPATIAL rows only.

The velocity (Ry bohr, same convention as ``2(k+G)``)::

    v_U^a = dV_U/dk_a = sum_I dPhi~_I W_I Phi~_I^+ + Phi~_I W_I dPhi~_I^+
    dPhi~ = dPhi O^{-1/2} + Phi d(O^{-1/2}),       dO = dPhi^+ Phi + Phi^+ dPhi
    d(O^{-1/2}) = U [ (U^+ dO U)_ab * F_ab ] U^+,  F_ab = -1/(sqrt(l_a) sqrt(l_b) (sqrt(l_a)+sqrt(l_b)))

``F_ab`` is the divided difference of ``x^{-1/2}`` (exact and stable, the
diagonal is ``-l^{-3/2}/2``).  The spatial rows ``Phi`` and ``dPhi`` are built by
``psp.vnl_ops`` from the same radial Hankel tables, QE solid harmonics, ``i^l``
and ``exp(-i(k+G).tau)`` as the V_NL projectors ``Z``/``dZ`` -- chi(r)/r takes the
place of beta(r)/r -- so ``i[r, V_U]`` rides the ``i[r, V_NL]`` code path.
``W`` is k-independent at fixed ns (the NSCF / fixed-density Hamiltonian).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp

RYTOEV = 13.605693122994            # QE Modules/constants.f90
_SPIN_PAIR = {0: 0, 1: 2, 2: 1, 3: 3}  # QE is1: (up,dn) <-> (dn,up); 0-based


# ---------------------------------------------------------------------------
# Occupations and the Hubbard potential (QE v_of_rho.f90, plus_u_full.f90)
# ---------------------------------------------------------------------------

def read_occup_nc(path, nat: int, ldmx: int) -> np.ndarray:
    """QE ``prefix.save/occup.txt`` (noncollinear): ``rho%ns_nc``.

    List-directed Fortran write of ``ns_nc(ldmx, ldmx, 4, nat)`` complex, first
    index fastest.  Returns ``ns[m1, m2, is, na]`` (0-based ``is`` = 2*s1+s2).
    """
    text = open(path).read().replace("(", " ").replace(")", " ").replace(",", " ")
    vals = np.asarray(text.split(), dtype=np.float64)
    want = 2 * ldmx * ldmx * 4 * nat
    if vals.size != want:
        # EXACT, not ">=": a background channel (nsb) or DFT+U+V (nsg) block
        # follows ns_nc in the same file, and reading its head as ns would
        # be a silently wrong potential.  Neither is supported here.
        raise ValueError(
            f"GATE dftu_occupations: {path} holds {vals.size // 2} complex "
            f"values, want exactly ns_nc({ldmx},{ldmx},4,{nat}) = {want // 2} "
            "(noncollinear DFT+U without background channels or +V)")
    z = vals[:want:2] + 1j * vals[1:want:2]
    return z.reshape((ldmx, ldmx, 4, nat), order="F")


def _real_ylm_on_sphere(l: int, n_theta: int = 24, n_phi: int = 48):
    """QE-convention real Y_lm (psp.radial.solid_harmonics) on a product rule
    exact for the degree-(4l) polynomials the Coulomb integrals need."""
    from psp.radial.solid_harmonics import solid_harmonics_jax
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    ct = np.repeat(x, n_phi)
    st = np.sqrt(1.0 - ct ** 2)
    ph = np.tile(phi, n_theta)
    xyz = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    w = np.repeat(wx, n_phi) * (2 * np.pi / n_phi)
    Y = np.asarray(solid_harmonics_jax(l, jnp.asarray(xyz)))   # (2l+1, nq)
    return Y, w, xyz


def hubbard_u_matrix(l: int, U: float, J: np.ndarray) -> np.ndarray:
    """QE ``hubbard_matrix``: u(m1,m2,m3,m4) = sum_k a_k F^k (units of U, J).

    ``a_k = 4pi/(2k+1) sum_q <Y_kq Y_m1 Y_m3><Y_kq Y_m2 Y_m4>`` is evaluated
    through the addition theorem ``sum_q Y_kq Y_kq' = (2k+1)/(4pi) P_k(cos g)``,
    so it needs only the l-shell harmonics, in QE's real convention.
    """
    J = np.asarray(J, dtype=float)
    F = {0: U}
    if l == 1:
        F[2] = 5.0 * J[0]
    elif l == 2:
        F[2] = 5.0 * J[0] + 31.5 * J[1]
        F[4] = 9.0 * J[0] - 31.5 * J[1]
    elif l == 3:
        F[2] = 225.0 / 54.0 * J[0] + 32175.0 / 42.0 * J[1] + 2475.0 / 42.0 * J[2]
        F[4] = 11.0 * J[0] - 141570.0 / 77.0 * J[1] + 4356.0 / 77.0 * J[2]
        F[6] = 7361.64 / 594.0 * J[0] + 36808.2 / 66.0 * J[1] - 11154.0e-2 * J[2]
    elif l != 0:
        raise ValueError(f"hubbard_u_matrix: l={l} not implemented (QE: l <= 3)")
    Y, w, xyz = _real_ylm_on_sphere(l)
    cosg = np.clip(xyz @ xyz.T, -1.0, 1.0)
    YY = Y[:, None, :] * Y[None, :, :] * w[None, None, :]          # (m, m', q)
    u = np.zeros((2 * l + 1,) * 4)
    for k, Fk in F.items():
        Pk = np.polynomial.legendre.legval(cosg, [0] * k + [1])
        u += Fk * np.einsum("acq,qp,bdp->abcd", YY, Pk, YY, optimize=True)
    return u


def default_hubbard_J(l: int, J1: float) -> np.ndarray:
    """QE ldaU.f90 defaults: B = 0.114774114774 J for d (E2, E3 for f)."""
    J = np.zeros(3)
    J[0] = J1
    if l == 2:
        J[1] = 0.114774114774 * J1
    elif l == 3:
        J[1] = 0.002268 * J1
        J[2] = 0.0438 * J1
    return J


def v_hubbard_full_nc(ns: np.ndarray, l: int, U: float, J: np.ndarray):
    """QE ``v_hubbard_full_nc`` for ONE atom (Liechtenstein, noncollinear).

    ``ns`` is ``(ld, ld, 4)`` in QE layout; U, J in Ry.  Returns
    ``(v_hub (ld, ld, 4), (eth_dc, eth_noflip, eth_flip, eth))``.
    """
    ld = 2 * l + 1
    ns = np.asarray(ns[:ld, :ld, :], dtype=np.complex128)
    u = hubbard_u_matrix(l, U, J)
    J1 = float(J[0])
    n_tot = np.trace(ns[:, :, 0]) + np.trace(ns[:, :, 3])
    mx = np.real(np.trace(ns[:, :, 1]) + np.trace(ns[:, :, 2]))
    my = 2.0 * np.imag(np.trace(ns[:, :, 1]))
    mz = np.real(np.trace(ns[:, :, 0]) - np.trace(ns[:, :, 3]))
    mag2 = mx ** 2 + my ** 2 + mz ** 2
    N = float(np.real(n_tot))
    eth_dc = 0.5 * (U * N * (N - 1.0) - J1 * N * (0.5 * N - 1.0) - 0.5 * J1 * mag2)
    eth_noflip = 0.0
    eth_flip = 0.0
    v = np.zeros((ld, ld, 4), dtype=np.complex128)
    ns_charge = ns[:, :, 0] + ns[:, :, 3]
    for is_ in range(4):
        is1 = _SPIN_PAIR[is_]
        if is1 == is_:
            eth_noflip += 0.5 * np.real(
                np.einsum("abcd,ac,bd->", u - u.transpose(0, 1, 3, 2),
                          ns[:, :, is_], ns[:, :, is_])
                + np.einsum("abcd,ac,bd->", u, ns[:, :, is_], ns[:, :, 3 - is_]))
            v[:, :, is_] += np.einsum("acbd,cd->ab", u, ns_charge)
        else:
            eth_flip += -0.5 * np.real(
                np.einsum("abdc,ac,bd->", u, ns[:, :, is_], ns[:, :, is1]))
        n_aux = np.trace(ns[:, :, is1])
        v[:, :, is_] += np.eye(ld) * (J1 * n_aux)
        if is1 == is_:
            v[:, :, is_] += np.eye(ld) * (0.5 * (U - J1) - U * n_tot)
        v[:, :, is_] -= np.einsum("acdb,cd->ab", u, ns[:, :, is1])
    eth = eth_noflip + eth_flip - eth_dc
    return v, (eth_dc, eth_noflip, eth_flip, eth)


def v_hubbard_dudarev_nc(ns: np.ndarray, l: int, U: float, alpha: float = 0.0):
    """QE ``v_hubbard_nc`` for ONE atom (Dudarev U_eff, noncollinear; no J0/beta)."""
    ld = 2 * l + 1
    ns = np.asarray(ns[:ld, :ld, :], dtype=np.complex128)
    v = np.zeros((ld, ld, 4), dtype=np.complex128)
    eth = 0.0
    for is_ in range(4):
        is1 = _SPIN_PAIR[is_]
        if is1 == is_:
            v[:, :, is_] += np.eye(ld) * (alpha + 0.5 * U) - U * ns[:, :, is_].T
            eth += np.real((alpha + 0.5 * U) * np.trace(ns[:, :, is_])
                           - 0.5 * U * np.einsum("ab,ba->", ns[:, :, is_], ns[:, :, is_]))
        else:
            v[:, :, is_] += -U * ns[:, :, is1].T
            eth += np.real(-0.5 * U * np.einsum("ab,ba->", ns[:, :, is_], ns[:, :, is1]))
    return v, eth


def hubbard_W(v_hub: np.ndarray) -> np.ndarray:
    """QE vhpsi_U_nc ``vaux``: (2 ld, 2 ld) with row/col = m + ld*s."""
    ld = v_hub.shape[0]
    W = np.zeros((2 * ld, 2 * ld), dtype=np.complex128)
    for s1 in range(2):
        for s2 in range(2):
            W[s1 * ld:(s1 + 1) * ld, s2 * ld:(s2 + 1) * ld] = v_hub[:, :, 2 * s1 + s2]
    return W


# ---------------------------------------------------------------------------
# Atomic-wavefunction rows through the V_NL projector machinery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HubbardShell:
    """One Hubbard channel: element, PP_CHI label (e.g. '3D'), l, U/J in eV,
    and the QE formulation ('liechtenstein' = lda_plus_u_kind 1, 'dudarev' = 0)."""
    element: str
    label: str
    l: int
    U_eV: float
    J_eV: float = 0.0
    kind: str = "liechtenstein"
    #: QE atom indices carrying this shell (``()`` = every atom of ``element``).
    atoms: tuple = ()
    #: d-shell B (Slater F4 knob); ``None`` = QE's default 0.114774114774 J.
    B_eV: float | None = None


@dataclass
class HubbardSetup:
    """k-independent DFT+U data.

    ``atwfc`` is a ``vnl_ops.VNLSetup`` whose "projectors" are the j-averaged
    atomic wavefunctions (its E blocks are zero and never used).  ``hub_rows``
    ``(n_hub, ld)`` index the spatial rows of each Hubbard atom's shell (m in
    QE ylmr2 order), ``hub_atoms`` their QE atom indices, ``W`` ``(n_hub, 2ld,
    2ld)`` the potential matrices in Ry.
    """
    atwfc: object
    row_labels: list
    hub_rows: np.ndarray
    hub_atoms: np.ndarray
    W: np.ndarray
    l: int
    energies_ry: dict = field(default_factory=dict)

    @property
    def hub_idx(self):
        """Flattened Hubbard row indices (atom-major), as a device int32 array."""
        return jnp.asarray(self.hub_rows.reshape(-1), dtype=jnp.int32)

    @property
    def W_dev(self):
        return jnp.asarray(self.W, dtype=jnp.complex128)


def j_averaged_atomic_wfcs(pswfc: dict) -> list:
    """QE atomic_wfc_so_mag / atomic_wfc_nc selection: one radial function per
    (n, l) with occupation >= 0; a j = l-1/2 partner is averaged in."""
    chis = pswfc["chi"]
    has_so = any(c["j"] is not None for c in chis)
    out = []
    for nb, c in enumerate(chis):
        if c["occupation"] < 0.0:
            continue
        l = c["l"]
        if not has_so:
            out.append({"label": c["label"], "l": l, "chi": c["chi"]})
            continue
        j = c["j"]
        if abs(j - l + 0.5) < 1e-4:
            continue                       # j = l-1/2: consumed by its partner
        nc = nb
        if l > 0:
            for ib, d in enumerate(chis):
                if d["l"] == l and abs(d["j"] - l + 0.5) < 1e-4:
                    nc = ib
                    break
        chi = (c["chi"] * (l + 1) + chis[nc]["chi"] * l) / (2 * l + 1)
        out.append({"label": c["label"], "l": l, "chi": chi})
    return out


def _atomic_species(element: str, pswfc: dict, wfcs: list):
    from psp.species import SpeciesData
    from psp.pseudos import symbol_to_Z
    from psp.radial_tables import qe_vloc_radial_scheme
    r, rab = pswfc["r"], pswfc["rab"]
    safe = np.where(r > 0, r, 1.0)
    chi_r = np.stack([np.where(r > 0, w["chi"] / safe, 0.0) for w in wfcs])
    msh, _ = qe_vloc_radial_scheme(r, rab)       # QE init_tab_atwfc: r <= 10, odd
    n = len(wfcs)
    return SpeciesData(
        element=element, z_valence=0.0, z_atomic=int(symbol_to_Z(element)),
        r=r, rab=rab, vloc_r=np.zeros_like(r), rho_core_r=np.zeros_like(r),
        has_nlcc=False, n_proj=n, beta_r=chi_r,
        proj_l=np.asarray([w["l"] for w in wfcs], dtype=np.int32),
        dij=np.zeros((n, n)), kkbeta=int(msh))


def build_atwfc_setup(wfn, upf_paths: dict, *, nspinor: int = 2,
                      q_max: float | None = None, n_q: int | None = None):
    """A ``VNLSetup`` over all atomic wavefunctions (QE natomwfc / npol rows).

    Returns ``(setup, row_labels)``; ``row_labels[r] = (qe_atom, element,
    label, l, m)``.  Row order is vnl_ops': channel (species, l) -> atom ->
    radial function -> m.  Loewdin is permutation covariant, so only the
    Hubbard-row lookup depends on it.
    """
    from psp import vnl_ops
    from psp.radial_tables import build_all_tables
    from psp.upf.load_upf import load_upf_pswfc

    atom_types = np.asarray(wfn.atom_types, dtype=int)
    atom_crys = np.asarray(wfn.atom_crys, dtype=np.float64)
    species, wfc_meta = [], []
    for el, path in upf_paths.items():
        pswfc = load_upf_pswfc(path)
        wfcs = j_averaged_atomic_wfcs(pswfc)
        species.append(_atomic_species(el, pswfc, wfcs))
        wfc_meta.append(wfcs)
    B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
    cell_volume = float(wfn.cell_volume)
    if q_max is None:
        q_max = float(np.sqrt(float(wfn.ecutwfc)))
    q_max *= 1.01
    if n_q is None:
        n_q = max(4000, int(np.ceil(q_max / 5.0e-4)) + 1)
    tables = build_all_tables(species, q_max, n_q)
    q_grid = tables["q"]

    channels, G_rows, Gp_rows, row_labels = [], [], [], []
    row_beta, row_l, row_m, row_tau = [], [], [], []
    beta_idx = 0
    for isp, sp in enumerate(species):
        atoms = [i for i in range(len(atom_types)) if atom_types[i] == sp.z_atomic]
        if not atoms:
            continue
        tau = atom_crys[atoms]
        per_l: dict = {}
        for ip in range(sp.n_proj):
            per_l.setdefault(int(sp.proj_l[ip]), []).append(ip)
        for l, ids in per_l.items():
            start = beta_idx
            for ip in ids:
                F = tables["proj_tables"][isp][ip]
                H = tables["deriv_tables"][isp][ip]
                if l == 0:
                    G, Gp = F.copy(), -H
                else:
                    G = np.empty(n_q)
                    G[1:] = F[1:] / q_grid[1:] ** l
                    G[0] = tables["reduced_origins"][isp][ip]
                    Gp = np.zeros(n_q)
                    Gp[1:] = -H[1:] / q_grid[1:] ** l
                G_rows.append(G)
                Gp_rows.append(Gp)
                beta_idx += 1
            msize = 2 * l + 1
            R = len(ids) * msize
            channels.append(vnl_ops.ChannelMeta(
                l=l, nbeta=len(ids), msize=msize, R=R, tau=tau,
                E=np.zeros((2, 2, R, R)), beta_table_start=start,
                natoms=len(atoms)))
            for ia, a in enumerate(atoms):
                for ib, ip in enumerate(ids):
                    for m in range(msize):
                        row_beta.append(start + ib)
                        row_l.append(l)
                        row_m.append(m)
                        row_tau.append(tau[ia])
                        row_labels.append((a, sp.element, wfc_meta[isp][ip]["label"], l, m))
    total_R = len(row_labels)
    setup = vnl_ops.VNLSetup(
        channels=channels, dq=tables["dq"], n_q=n_q, q_max=q_max,
        G_table=jnp.asarray(np.stack(G_rows)), Gp_table=jnp.asarray(np.stack(Gp_rows)),
        prefactor=4.0 * np.pi / np.sqrt(cell_volume), B=B, cell_volume=cell_volume,
        total_R=total_R, nspinor=nspinor,
        E_super=jnp.zeros((nspinor, nspinor, total_R, total_R), dtype=jnp.complex128),
        l_max=max(ch.l for ch in channels), soc=False,
        soc_provenance="atomic wavefunctions (j-averaged), not a V_NL",
        row_beta_idx=jnp.asarray(row_beta, dtype=jnp.int32),
        row_l=jnp.asarray(row_l, dtype=jnp.int32),
        row_m=jnp.asarray(row_m, dtype=jnp.int32),
        row_tau=jnp.asarray(np.asarray(row_tau).reshape(-1, 3)))
    return setup, row_labels


def build_hubbard_setup(wfn, upf_paths: dict, shells: list, ns: np.ndarray, *,
                        nspinor: int = 2, q_max: float | None = None) -> HubbardSetup:
    """Atomic rows + the Hubbard potential W_I from QE's occupations ``ns``
    (``read_occup_nc`` layout ``ns[m1, m2, is, qe_atom]``)."""
    setup, labels = build_atwfc_setup(wfn, upf_paths, nspinor=nspinor, q_max=q_max)
    ls = {sh.l for sh in shells}
    if len(ls) != 1:
        raise ValueError("prototype: one Hubbard l for every shell")
    l = ls.pop()
    ld = 2 * l + 1
    hub_rows, hub_atoms, Ws, energies = [], [], [], {}
    for sh in shells:
        have = sorted({a for (a, el, lab, ll, m) in labels
                       if el == sh.element and lab.upper() == sh.label.upper() and ll == sh.l})
        atoms = sorted(sh.atoms) if sh.atoms else have
        if not atoms or not set(atoms) <= set(have):
            raise ValueError(
                f"GATE dftu_atomic_wfc: no PP_PSWFC function {sh.element} {sh.label} (l={sh.l}) "
                f"on atoms {atoms} (have {have}); the UPF must carry the Hubbard manifold")
        U = sh.U_eV / RYTOEV
        J = default_hubbard_J(sh.l, sh.J_eV / RYTOEV)
        if sh.B_eV is not None and sh.l == 2:
            J[1] = sh.B_eV / RYTOEV
        for a in atoms:
            rows = [i for i, (aa, el, lab, ll, m) in enumerate(labels)
                    if aa == a and el == sh.element and lab.upper() == sh.label.upper()
                    and ll == sh.l]
            rows = sorted(rows, key=lambda i: labels[i][4])
            if sh.kind == "liechtenstein":
                v, e = v_hubbard_full_nc(ns[:, :, :, a], sh.l, U, J)
            elif sh.kind == "dudarev":
                v, e = v_hubbard_dudarev_nc(ns[:, :, :, a], sh.l, U)
            else:
                raise ValueError(f"unknown Hubbard kind {sh.kind!r}")
            hub_rows.append(rows)
            hub_atoms.append(a)
            Ws.append(hubbard_W(v))
            energies[a] = e
    W = np.stack(Ws)
    herm = float(np.max(np.abs(W - np.conj(np.transpose(W, (0, 2, 1))))))
    if herm > 1e-8:
        raise ValueError(f"Hubbard potential W is not Hermitian (max |W-W^+| = {herm:.2e})")
    return HubbardSetup(atwfc=setup, row_labels=labels,
                        hub_rows=np.asarray(hub_rows, dtype=np.int32),
                        hub_atoms=np.asarray(hub_atoms, dtype=np.int32),
                        W=W, l=l, energies_ry=energies)


# ---------------------------------------------------------------------------
# Per-k operator
# ---------------------------------------------------------------------------

@jax.jit
def lowdin_rows_with_derivative(Z, dZ, hub_idx):
    """Ortho-atomic rows and their k-derivative.

    ``Z`` (R, nG) ket rows (``P = conj(Z) psi``), ``dZ`` (3, R, nG).  Returns
    ``Zt[hub_idx]`` (h, nG), ``dZt[:, hub_idx]`` (3, h, nG) and the Loewdin
    eigenvalues (the conditioning of O).
    """
    O = jnp.conj(Z) @ Z.T                                    # O_rr' = <Z_r|Z_r'>
    O = 0.5 * (O + jnp.conj(O.T))
    lam, U = jnp.linalg.eigh(O)
    sq = jnp.sqrt(lam)
    Oih = (U * (1.0 / sq)[None, :]) @ jnp.conj(U.T)          # O^{-1/2}
    F = -1.0 / (sq[:, None] * sq[None, :] * (sq[:, None] + sq[None, :]))
    C = Oih[:, hub_idx]                                      # (R, h)
    Zt = C.T @ Z

    def one(dZa):
        dO = jnp.conj(dZa) @ Z.T + jnp.conj(Z) @ dZa.T
        dOih = U @ ((jnp.conj(U.T) @ dO @ U) * F) @ jnp.conj(U.T)
        return C.T @ dZa + dOih[:, hub_idx].T @ Z

    dZt = jax.vmap(one)(dZ)
    return Zt, dZt, lam


def _shell_projections(Zt, psi_G, n_hub, ld):
    """P[I, m + ld*s, n] = <phi~_{I m}| psi_{n, s}>  for the Hubbard rows."""
    P = jnp.einsum("rG,nsG->rsn", jnp.conj(Zt), psi_G, optimize=True)    # (h, 2, nb)
    P = P.reshape(n_hub, ld, 2, -1)
    return jnp.transpose(P, (0, 2, 1, 3)).reshape(n_hub, 2 * ld, -1)


@jax.jit
def hubbard_matrix_and_velocity(psi_G, Zt, dZt, W):
    """``<m|V_U|n>`` (nb, nb) and ``<m|dV_U/dk_a|n>`` (3, nb, nb), Ry / Ry bohr.

    ``psi_G`` (nb, 2, nG) on the k's own G-sphere; ``Zt``/``dZt`` from
    :func:`lowdin_rows_with_derivative` for the ``hub_rows`` (flattened
    atom-major); ``W`` (n_hub, 2ld, 2ld).  Also returns the projections
    ``P`` (n_hub, 2ld, nb) for occupations.
    """
    n_hub, two_ld, _ = W.shape
    ld = two_ld // 2
    P = _shell_projections(Zt, psi_G, n_hub, ld)
    dP = jax.vmap(lambda d: _shell_projections(d, psi_G, n_hub, ld))(dZt)
    WP = jnp.einsum("iab,ibn->ian", W, P)
    H = jnp.einsum("iam,ian->mn", jnp.conj(P), WP)
    v = (jnp.einsum("jiam,ian->jmn", jnp.conj(dP), WP)
         + jnp.einsum("iam,iab,jibn->jmn", jnp.conj(P), W, dP))
    return H, v, P


# ---------------------------------------------------------------------------
# The ket form used by the ONE velocity path (common.mtxel_sweep, finite q)
# ---------------------------------------------------------------------------

def _rows_from_shell(X, n_hub, ld):
    """(..., n_hub, 2ld, nb) [index s*ld+m] -> (..., n_hub*ld, 2, nb) rows."""
    lead = X.shape[:-3]
    nb = X.shape[-1]
    X = X.reshape(lead + (n_hub, 2, ld, nb))
    X = jnp.swapaxes(X, -3, -2)                              # (..., n_hub, ld, 2, nb)
    return X.reshape(lead + (n_hub * ld, 2, nb))


def hubbard_velocity_ket(psi_G, Zt, dZt, W):
    """``dV_U/dK_cart |psi>`` from ortho-atomic rows: ``(3, nb, 2, nG)``.

    Band-local (no contraction over bands), so a band-sharded ``psi_G`` stays
    sharded; ``Zt``/``dZt``/``W`` are replicated and small.
    """
    n_hub, two_ld, _ = W.shape
    ld = two_ld // 2
    P = _shell_projections(Zt, psi_G, n_hub, ld)
    dP = jax.vmap(lambda d: _shell_projections(d, psi_G, n_hub, ld))(dZt)
    WP = _rows_from_shell(jnp.einsum("iab,ibn->ian", W, P), n_hub, ld)
    WdP = _rows_from_shell(jnp.einsum("iab,jibn->jian", W, dP), n_hub, ld)
    return (jnp.einsum("jrG,rsn->jnsG", dZt, WP, optimize=True)
            + jnp.einsum("rG,jrsn->jnsG", Zt, WdP, optimize=True))


def apply_hubbard_velocity_to_ket(psi_G, kvec, Gk_int, gmask, hub: "HubbardSetup"):
    """``dV_U/dK_cart`` applied to a ket at one k; traced-safe.

    ``psi_G`` (nb, 2, nG) on this k's (padded) G table, ALREADY masked;
    ``gmask`` (nG,) 1 on physical G, 0 on the D10 pad.  The atomic rows are
    masked before the Loewdin overlap: vnl_ops' Z is finite on a pad column
    (it is evaluated at K = kvec there), and an unmasked O would pick up
    ``ngkmax - ngk`` spurious terms -- an error psi's own mask cannot remove,
    because O is a row-row contraction that never sees psi.
    Returns ``(3, nb, 2, nG)`` in Ry bohr, the layout of
    ``vnl_ops.apply_vnl_velocity_to_ket``.
    """
    from psp import vnl_ops
    kd = vnl_ops.build_vnl_kdata_traced(kvec, Gk_int, hub.atwfc, compute_dZ=True)
    m = jnp.asarray(gmask).astype(kd.Z.dtype)
    Zt, dZt, _ = lowdin_rows_with_derivative(
        kd.Z * m[None, :], kd.dZ * m[None, None, :], hub.hub_idx)
    return hubbard_velocity_ket(psi_G, Zt, dZt, hub.W_dev)


# ---------------------------------------------------------------------------
# Where the Hubbard data come from: the QE input card + prefix.save/occup.txt
# ---------------------------------------------------------------------------

HARTREE_EV = 27.211386245988           # QE Modules/constants.f90 AUTOEV
NO_HUBBARD = "none"
_SUPPORTED_PROJECTORS = ("ortho-atomic",)
_QE_CARDS = ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "CELL_PARAMETERS",
             "OCCUPATIONS", "CONSTRAINTS", "ATOMIC_FORCES", "ADDITIONAL_K_POINTS",
             "SOLVENTS", "HUBBARD", "ATOMIC_VELOCITIES", "TOTAL_CHARGE")
_L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}


def _gate(rule, got, want, why, fix):
    return ValueError(f"GATE {rule}: {why.split(';')[0]}\n  got:  {got}\n  want: {want}\n"
                      f"  why:  {why}\n  fix:  {fix}\n  doc:  src/psp/hubbard_ops.py")


@dataclass(frozen=True)
class QEHubbardCard:
    """The HUBBARD card of a pw.x input, in the card's own units (eV)."""
    path: str
    projector: str
    formulation: str                     # 'dudarev' (kind 0) | 'liechtenstein' (kind 1)
    entries: tuple                       # ((param, species, manifold, value_eV), ...)
    species_of_atom: tuple               # QE atom order -> species label
    element_of_species: dict


def _strip_comment(line):
    for c in ("!", "#"):
        if c in line:
            line = line[:line.index(c)]
    return line.strip()


def parse_qe_hubbard_card(path) -> QEHubbardCard:
    """Read ATOMIC_SPECIES, ATOMIC_POSITIONS and the HUBBARD card of a pw.x input.

    QE's formulation rule (Modules/read_cards.f90 card_hubbard): any J / B /
    E2 / E3 -> lda_plus_u_kind 1 (Liechtenstein); any V -> 2; else U (J0)
    -> 0 (Dudarev).  Only U, J and B are accepted here; J0, V, alpha, beta,
    U2 and Hp refuse, as do non-ortho-atomic projectors.
    """
    lines = [_strip_comment(x) for x in open(path).read().splitlines()]
    card, species, positions, hub, projector = None, {}, [], [], None
    for raw in lines:
        if not raw:
            continue
        head = raw.split()[0].upper().split("(")[0].split("{")[0]
        if head in _QE_CARDS:
            card = head
            if head == "HUBBARD":
                opt = raw[len("HUBBARD"):].strip().strip("(){} ").lower()
                projector = opt
            continue
        if raw.startswith("&") or raw == "/":
            card = None
            continue
        tok = raw.split()
        if card == "ATOMIC_SPECIES" and len(tok) >= 3:
            species[tok[0]] = tok[2]
        elif card == "ATOMIC_POSITIONS":
            positions.append(tok[0])
        elif card == "HUBBARD":
            hub.append(tok)
    if projector is None:
        raise _gate("dftu_input", f"hubbard_input = {path}", "a pw.x input with a HUBBARD card",
                    "no HUBBARD card found", "point hubbard_input at the SCF/NSCF input that made the WFN")
    if projector not in _SUPPORTED_PROJECTORS:
        raise _gate("dftu_input", f"HUBBARD ({projector})", f"HUBBARD ({'|'.join(_SUPPORTED_PROJECTORS)})",
                    "only QE ortho-atomic projectors are implemented (atomic/norm-atomic/wf/pseudo differ in O)",
                    "use ortho-atomic, or implement the projector type in psp.hubbard_ops")
    entries = []
    for tok in hub:
        param = tok[0].upper()
        if param not in ("U", "J", "B"):
            raise _gate("dftu_input", f"HUBBARD line {' '.join(tok)!r}", "U, J or B lines only",
                        "J0, V (DFT+U+V), alpha, beta, U2 and Hp are not implemented",
                        "drop the term or implement it in psp.hubbard_ops")
        if len(tok) != 3 or "-" not in tok[1]:
            raise _gate("dftu_input", f"HUBBARD line {' '.join(tok)!r}", "'U Species-3d value'",
                        "one species-manifold label and one value per line", "fix the card")
        sp, man = tok[1].rsplit("-", 1)
        entries.append((param, sp, man.lower(), float(tok[2])))
    formulation = "liechtenstein" if any(e[0] in ("J", "B") and e[3] != 0.0 for e in entries) else "dudarev"
    elements = {}
    for sp, upf in species.items():
        el = "".join(ch for ch in sp if ch.isalpha())[:2]
        elements[sp] = el[0].upper() + el[1:].lower() if el else sp
    return QEHubbardCard(path=str(path), projector=projector, formulation=formulation,
                         entries=tuple(entries), species_of_atom=tuple(positions),
                         element_of_species=elements)


def qe_dftu_declaration(wfn):
    """What the WFN's authenticated QE schema says about DFT+U.

    ``None`` when no QE schema authenticates this WFN (declaration unknown);
    ``{}`` when it declares no DFT+U; else ``{'kind', 'projector', 'U_eV':
    {(species, manifold): U}}`` read from the ``<input>`` block, whose
    Hubbard_U is Hartree.  (The ``<output>`` block writes Hubbard_U in
    Hartree but Hubbard_J in RYDBERG -- QE 7.6, measured on VI3 X6p0:
    0.0588 = 0.8 eV -- so J is never read from the XML.)
    """
    import xml.etree.ElementTree as ET
    binding = getattr(wfn, "qe_symmetry_binding", None)
    if binding is None and getattr(wfn, "_qe_symmetry_checked", True) is False:
        # The loader binds its schema lazily inside symmetry(); bind here with
        # the SAME resolver (no SymMaps needed) rather than forcing the k maps.
        from symmetry_maps import resolve_qe_symmetry_binding
        binding, _ = resolve_qe_symmetry_binding(
            wfn, wfn_path=wfn.path, schema=getattr(wfn, "_qe_schema_request", None))
    if binding is None:
        return None
    root = ET.parse(binding.schema_path).getroot()
    dftu = [e for e in root.iter() if e.tag.split("}")[-1] == "dftU"]
    inp = root.find("input")
    blocks = [e for e in (inp.iter() if inp is not None else ()) if e.tag.split("}")[-1] == "dftU"]
    if not dftu:
        return {}
    blk = blocks[0] if blocks else dftu[0]
    kind = blk.find("lda_plus_u_kind")
    proj = blk.find("U_projection_type")
    U = {}
    for e in blk.findall("Hubbard_U"):
        val = float(e.text.split()[0]) * HARTREE_EV
        if val != 0.0:
            U[(e.get("specie"), e.get("label").lower())] = val
    return {"kind": int(kind.text) if kind is not None else 0,
            "projector": (proj.text.strip() if proj is not None else "atomic"),
            "U_eV": U, "schema_path": binding.schema_path}


@dataclass(frozen=True)
class HubbardInput:
    """Resolved Hubbard data for one run (``None`` fields -> no DFT+U)."""
    card: QEHubbardCard
    occupations_path: str
    occupations_sha256: str
    shells: tuple                         # HubbardShell (with atoms)
    provenance: str


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hubbard_provenance(card: QEHubbardCard, occ_sha: str, shells) -> str:
    """Canonical JSON stamp: occupations hash, U/J/B per shell, formulation, projector."""
    import json
    return json.dumps({
        "scheme": "lorrax.dftu_velocity/v1",
        "projector": card.projector, "formulation": card.formulation,
        "occupations_sha256": occ_sha,
        "shells": [{"element": sh.element, "manifold": sh.label.lower(), "l": sh.l,
                    "atoms": list(sh.atoms), "U_eV": sh.U_eV, "J_eV": sh.J_eV, "B_eV": sh.B_eV}
                   for sh in shells]}, sort_keys=True)


def resolve_hubbard_input(hubbard_input, hubbard_occupations, *, wfn, base_dir,
                          caller: str) -> HubbardInput | None:
    """The one resolver every producer and consumer calls.

    * The QE schema declares DFT+U and no Hubbard input is given -> REFUSE
      (the velocity would silently lack i[r, V_U]).
    * Hubbard input is given but the schema declares no DFT+U -> REFUSE.
    * Neither -> ``None``: the plain-DFT path, bit-for-bit unchanged.
    * Both -> parsed card + occupations, cross-checked against the schema's
      formulation, projector and U values (Hartree input block).
    """
    import os
    decl = qe_dftu_declaration(wfn)
    given = [bool(str(hubbard_input or "").strip()), bool(str(hubbard_occupations or "").strip())]
    if not any(given):
        if decl:
            raise _gate(
                "dftu_velocity_input", f"{caller}: hubbard_input/hubbard_occupations unset; "
                f"QE schema {decl['schema_path']} declares DFT+U ({decl['projector']}, kind {decl['kind']}, "
                f"U = {decl['U_eV']})", "hubbard_input = <pw.x input with the HUBBARD card> and "
                "hubbard_occupations = <prefix.save/occup.txt>",
                "the DFT velocity is dH/dk and H contains V_U; without it every velocity matrix element "
                "misses i[r,V_U] (about 5% of |v_cv| on VI3 U=6 eV)",
                "add both keys to the deck")
        return None
    if not all(given):
        raise _gate("dftu_velocity_input", f"{caller}: hubbard_input={hubbard_input!r}, "
                    f"hubbard_occupations={hubbard_occupations!r}", "both keys, or neither",
                    "V_U needs the card (U/J/B, projector) AND the occupation matrix", "set the missing key")
    if decl == {}:
        raise _gate("dftu_velocity_input", f"{caller}: Hubbard input given, but the QE schema "
                    "of this WFN declares no DFT+U", "Hubbard keys only for a DFT+U mean field",
                    "adding V_U to a Hamiltonian that never contained it is a wrong velocity",
                    "remove hubbard_input/hubbard_occupations, or point wfn_file at the DFT+U WFN")

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)
    card = parse_qe_hubbard_card(_abs(str(hubbard_input).strip()))
    occ_path = _abs(str(hubbard_occupations).strip())
    nspinor = int(getattr(wfn, "nspinor", 1))
    if nspinor != 2:
        raise _gate("dftu_velocity_input", f"nspinor = {nspinor}", "nspinor = 2 (noncollinear)",
                    "only QE's noncollinear ns_nc / v_hubbard_(full_)nc are implemented",
                    "implement the collinear branch in psp.hubbard_ops")
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    if len(card.species_of_atom) != len(atom_types):
        raise _gate("dftu_velocity_input", f"{len(card.species_of_atom)} atoms in {card.path}",
                    f"{len(atom_types)} atoms (the WFN)", "the card must come from the run that made the WFN",
                    "point hubbard_input at that run's input")
    from psp.pseudos import symbol_to_Z
    for i, (sp, z) in enumerate(zip(card.species_of_atom, atom_types)):
        if symbol_to_Z(card.element_of_species.get(sp, sp)) != int(z):
            raise _gate("dftu_velocity_input", f"atom {i}: species {sp} (Z={symbol_to_Z(card.element_of_species.get(sp, sp))})",
                        f"Z = {int(z)} (the WFN)", "atom order must match the WFN", "use the WFN-producing input")
    vals = {}
    for param, sp, man, v in card.entries:
        vals.setdefault((sp, man), {})[param] = v
    shells = []
    for (sp, man), d in vals.items():
        if "U" not in d:
            raise _gate("dftu_velocity_input", f"{sp}-{man}: {d}", "a U line per Hubbard manifold",
                        "J/B without U is not a supported formulation", "fix the card")
        l = _L_OF[man[-1]]
        J1 = d.get("J", 0.0)
        B = d.get("B", default_hubbard_J(l, J1)[1] if l == 2 else 0.0)
        atoms = tuple(i for i, s in enumerate(card.species_of_atom) if s == sp)
        shells.append(HubbardShell(card.element_of_species.get(sp, sp), man.upper(), l,
                                   float(d["U"]), float(J1), card.formulation, atoms, float(B)))
    if decl:
        want_kind = {"dudarev": 0, "liechtenstein": 1}[card.formulation]
        got_U = {(sp, man): d["U"] for (sp, man), d in vals.items()}
        bad = []
        if decl["kind"] != want_kind:
            bad.append(f"lda_plus_u_kind schema={decl['kind']} card={want_kind}")
        if decl["projector"] != card.projector:
            bad.append(f"projector schema={decl['projector']} card={card.projector}")
        if set(decl["U_eV"]) != set(got_U) or any(abs(decl["U_eV"][k] - got_U[k]) > 1e-6 for k in got_U):
            bad.append(f"U (eV) schema={decl['U_eV']} card={got_U}")
        if bad:
            raise _gate("dftu_velocity_input", "; ".join(bad), "the card that produced this WFN",
                        "the Hubbard input disagrees with the QE schema that authenticates the WFN",
                        "point hubbard_input at the WFN-producing pw.x input")
    ldmx = 2 * max(sh.l for sh in shells) + 1
    ns = read_occup_nc(occ_path, nat=len(atom_types), ldmx=ldmx)
    for sh in shells:
        for a in sh.atoms:
            if not np.any(ns[:, :, :, a]):
                raise _gate("dftu_occupations", f"ns of atom {a} ({sh.element}-{sh.label}) is zero in {occ_path}",
                            "nonzero Hubbard occupations on every Hubbard atom",
                            "the occupations do not belong to this card", "use the matching prefix.save/occup.txt")
    occ_sha = _sha256_file(occ_path)
    return HubbardInput(card=card, occupations_path=occ_path, occupations_sha256=occ_sha,
                        shells=tuple(shells), provenance=hubbard_provenance(card, occ_sha, shells))


def hubbard_provenance_for(hubbard_input, hubbard_occupations, *, wfn, base_dir, caller) -> str:
    """The stamp a consumer expects: ``'none'`` or the resolved JSON."""
    hi = resolve_hubbard_input(hubbard_input, hubbard_occupations, wfn=wfn,
                               base_dir=base_dir, caller=caller)
    return NO_HUBBARD if hi is None else hi.provenance


def build_hubbard_from_input(wfn, hi: HubbardInput, pseudos: dict, *, nspinor: int = 2,
                             q_max: float | None = None) -> "HubbardSetup":
    """HubbardSetup for a resolved input; atomic wavefunctions from the deck's UPFs."""
    upf_paths = {el: getattr(p, "_source_path") for el, p in pseudos.items()}
    for el, p in pseudos.items():
        hdr = p.pp_header
        from psp.upf.upf_model_2_0_1 import UpfLogical
        if hdr.is_ultrasoft in (UpfLogical.TRUE, UpfLogical.T) or hdr.is_paw in (UpfLogical.TRUE, UpfLogical.T):
            raise _gate("dftu_velocity_input", f"{el}: ultrasoft/PAW pseudopotential",
                        "norm-conserving pseudopotentials (S = 1)",
                        "the ortho-atomic overlap and V_U here assume S = 1; US/PAW need S and dS/dk",
                        "implement S in psp.hubbard_ops")
    ns = read_occup_nc(hi.occupations_path, nat=len(np.asarray(wfn.atom_types)),
                       ldmx=2 * max(sh.l for sh in hi.shells) + 1)
    return build_hubbard_setup(wfn, upf_paths, list(hi.shells), ns, nspinor=nspinor, q_max=q_max)
