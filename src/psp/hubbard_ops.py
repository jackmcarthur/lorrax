r"""psp/hubbard_ops.py -- QE's DFT+U operator V_U(k) and its k-derivative.

RESEARCH PROTOTYPE (branch research/dftu-velocity-2026-09-23).  It closes the
``i[r, V_U]`` gap of the DFT velocity ``v = p + i[r, V_NL]`` for QE
ortho-atomic DFT+U, noncollinear, ``lda_plus_u_kind`` 0 (Dudarev) and 1
(Liechtenstein), norm-conserving pseudopotentials (S = 1).

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
    if vals.size < want:
        raise ValueError(f"{path}: {vals.size // 2} complex values, want "
                         f"ns_nc({ldmx},{ldmx},4,{nat}) = {want // 2}")
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
        atoms = sorted({a for (a, el, lab, ll, m) in labels
                        if el == sh.element and lab == sh.label and ll == sh.l})
        if not atoms:
            raise ValueError(f"no atomic wavefunction {sh.element} {sh.label} l={sh.l}")
        U = sh.U_eV / RYTOEV
        J = default_hubbard_J(sh.l, sh.J_eV / RYTOEV)
        for a in atoms:
            rows = [i for i, (aa, el, lab, ll, m) in enumerate(labels)
                    if aa == a and el == sh.element and lab == sh.label and ll == sh.l]
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


def hubbard_k(psi_G, kvec, Gk_int, hub: HubbardSetup):
    """Everything at one k: (H_U, v_U, P, lowdin eigenvalues)."""
    from psp import vnl_ops
    kd = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int, hub.atwfc, compute_dZ=True)
    hub_idx = jnp.asarray(hub.hub_rows.reshape(-1), dtype=jnp.int32)
    Zt, dZt, lam = lowdin_rows_with_derivative(kd.Z, kd.dZ, hub_idx)
    H, v, P = hubbard_matrix_and_velocity(
        psi_G, Zt, dZt, jnp.asarray(hub.W, dtype=jnp.complex128))
    return H, v, P, lam


def occupations_from_projections(P, f) -> np.ndarray:
    """QE new_ns_nc (no symmetrization): ns[m1, m2, 2 s1 + s2, i] summed with
    band weights ``f`` (nb,).  ``P`` (n_hub, 2ld, nb)."""
    P = np.asarray(P)
    n_hub, two_ld, _ = P.shape
    ld = two_ld // 2
    nr = np.einsum("n,ian,ibn->iab", np.asarray(f), np.conj(P), P)       # (i, a, b)
    ns = np.zeros((ld, ld, 4, n_hub), dtype=np.complex128)
    for s1 in range(2):
        for s2 in range(2):
            ns[:, :, 2 * s1 + s2, :] = np.transpose(
                nr[:, s1 * ld:(s1 + 1) * ld, s2 * ld:(s2 + 1) * ld], (1, 2, 0))
    return ns
