#!/usr/bin/env python3
"""Orbital magnetization (modern theory) from a LORRAX spinor WFN.

Computes the per-cell orbital magnetic moment of a spin-orbit-coupled
(2-component spinor) crystal directly from a BerkeleyGW-format ``WFN.h5``,
using the gauge-invariant modern-theory formula evaluated in the
*sum-over-states* (k.p) representation.  The k-space derivative of the
Hamiltonian is taken **analytically** via ``dH/dk = 2(k+G) + dV_NL/dk``
(plus ``i[r, V_U]`` for DFT+U) — no finite differences anywhere in the
velocity operator.

The velocity is the distributed q=0 DFT velocity of the dipole producer:
``velocity_only.h5`` beside ``--wfn`` (``psp.get_dipole_mtxels
--parallel-transport-out velocity_only.h5 --parallel-transport-velocity-only``)
when it exists, otherwise the same band-sharded matrix-element sweep
(``common.mtxel_sweep.dipole_operator``) run here.  Either way it arrives
unfolded to the full BZ by the typed polar symmetry action, so the moment is a
plain full-BZ sum.  The band ceiling is the stored velocity's band count.

Physics (Rydberg atomic units: hbar=1, 2 m_e = 1, energies in Ry, lengths
in Bohr).  Per-cell orbital moment, component gamma, in Bohr magnetons:

    m_gamma / mu_B = (+1/2) * sum_k w_k * Im sum_{n occ} sum_{m != n}
                       eps_{gamma a b} v^a_nm v^b_mn (eps_m + eps_n - 2 mu)
                                                     / (eps_n - eps_m)^2

with v^a_nm = <u_nk| dH_k/dk_a |u_mk> the velocity matrix element (Ry*Bohr),
w_k the full-BZ k-point weights (sum to 1), and the +1/2 master-formula
prefactor yields the electron
orbital moment -mu_B L/hbar (the local term contains H-epsilon).  See ``orbital_magnetization_THEORY.md`` for the full derivation,
sources, and the absolute-sign discussion.

The script also computes the spin moment <sigma_z> from the same WFN as an
internal calibration: it must be ~ +/-6 mu_B for CrI3, which both validates
the wavefunction/occupations and fixes the physical axis so the orbital
moment can be reported *relative to the spin moment* (parallel / antiparallel)
in a convention-robust way.

Orbital magnetization is identically zero without spin-orbit coupling for a
collinear ferromagnet, so the script requires ``nspinor == 2``.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import jax  # noqa: F401  (devices are queried only after runtime startup)

# Allow `python orbital_magnetization.py` as well as `-m psp.orbital_magnetization`
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta
from psp.orbital_response import orbital_pieces_at_k
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops


RY2EV = 13.605693122994
MU_B_PREFACTOR = 0.5  # |m_e/hbar^2| in Ry-a0^2 units (magnitude; sign handled below)


# ----------------------------------------------------------------------
#  The distributed q=0 DFT velocity, full BZ
# ----------------------------------------------------------------------
VELOCITY_ARTIFACT = "velocity_only.h5"


def dft_velocity_full_bz(wfn, sym, *, nbnd, mesh, artifact, pseudos):
    """Return ``(v, source)``: host ``(nk_full, 3, nbnd, nbnd)`` velocity.

    ``v[k, a, m, n] = <u_mk| dH_k/dk_a |u_nk>`` (Ry*Bohr) on the full BZ in
    ``sym`` order.  ``artifact`` (the dipole producer's ``velocity_only.h5``)
    is read through SlabIO when it exists; its WFN fingerprint, k grid,
    reciprocal lattice and band manifold are authenticated by
    :func:`gw.qsgw_head.load_dft_velocity_head`, and its operator (including
    the DFT+U stamp) is the producer's.  Without it the band-sharded q=0
    sweep runs here on the file wedge and is unfolded with the typed polar
    action.  The band matrices are gathered to every rank: the formula below
    is host algebra that every process executes.
    """
    from common.mtxel_sweep import blocks_to_host

    artifact = Path(artifact)
    if artifact.exists():
        from file_io.parallel_transport import HUBBARD_PROVENANCE_ATTR
        from file_io.slab_io import SlabIO
        from gw.qsgw_head import _ascii_stamp, load_dft_velocity_head

        with SlabIO(str(artifact), mode="r", mesh=mesh) as io:
            nb_file = int(io.read_small("band_stop", dtype=np.int64))
            try:
                hubbard = _ascii_stamp(io, str(artifact),
                                       HUBBARD_PROVENANCE_ATTR)
            except Exception:              # written before i[r,V_U] existed
                hubbard = "none"
        if nbnd > nb_file:
            raise ValueError(
                f"{artifact}: stored velocity covers bands [0,{nb_file}); "
                f"--nbnd {nbnd} exceeds it. The band ceiling is the stored "
                "velocity's; regenerate it with more bands or lower --nbnd.")
        meta_file = Meta.from_system(
            wfn, sym, int(wfn.nelec), max(0, nb_file - int(wfn.nelec)),
            nb_file, 0, False)
        head = load_dft_velocity_head(
            str(artifact), mesh=mesh, wfn=wfn, meta=meta_file, config=None)
        v = blocks_to_host(head.velocity_dft_cart, nb=nb_file)   # (3,nk,nb,nb)
        v = np.moveaxis(v, 0, 1)[:, :, :nbnd, :nbnd]
        source = (f"{artifact} (bands [0,{nb_file}) stored, [0,{nbnd}) used;"
                  f" DFT+U stamp {hubbard})")
        return np.ascontiguousarray(v), source

    from common.mtxel_sweep import (VNL_VELOCITY_SIGN_FLIPPED, SweepGeometry,
                                    dipole_operator, sweep_matrix_elements)
    from common.wfn_layout import band_sphere_spec
    from psp.dft_operators import padded_gvectors
    from psp.hubbard_ops import resolve_hubbard_input
    from symmetry_maps import unfold_file_wedge_polar_matrix

    # This route has no deck, so it has no Hubbard input: a DFT+U mean field
    # refuses through the one resolver.  Produce velocity_only.h5 with the
    # dipole driver and the deck's hubbard keys instead.
    resolve_hubbard_input("", "", wfn=wfn, base_dir=str(artifact.parent),
                          caller="psp.orbital_magnetization (sweep route, "
                                 "no V_U term)")
    if not pseudos:
        raise ValueError(
            f"no {VELOCITY_ARTIFACT} beside the WFN and no *.upf found: the "
            "sweep route needs the pseudopotentials for i[r,V_NL]; pass "
            "--pseudo-dir")
    vnl_setup = vnl_ops.build_vnl_setup(
        wfn, sym, None, pseudos, nspinor=int(wfn.nspinor))
    gtab = padded_gvectors(wfn, k="ibz")
    psi_G = wfn.load(bands=(0, nbnd), k="ibz", sharding=band_sphere_spec())
    geom = SweepGeometry(
        mesh=mesh, fft_grid=np.asarray(wfn.fft_grid),
        ngkmax=int(psi_G.shape[3]), nb=nbnd, ns=int(psi_G.shape[2]),
        nk=int(sym.nk_red), cell_volume=float(wfn.cell_volume))
    op = dipole_operator(geom, bvec=wfn.bvec, blat=wfn.blat,
                         vnl_setup=vnl_setup,
                         vnl_velocity_sign=VNL_VELOCITY_SIGN_FLIPPED)
    H = sweep_matrix_elements(
        psi_G, operator=op, geom=geom, gvecs=gtab.gvecs, gmask=gtab.mask,
        box_index=wfn.box_index(k="ibz"), kvecs=np.asarray(gtab.kvecs))
    del psi_G
    v = blocks_to_host(unfold_file_wedge_polar_matrix(sym, H), nb=nbnd)
    return v, (f"q=0 dipole sweep over bands [0,{nbnd}) "
               f"(no {VELOCITY_ARTIFACT} beside the WFN)")


def spin_moment_ibz(wfn, sym, *, nocc):
    """Occupied ``sum_k w_k sum_n <sigma>`` from band-sharded raw IBZ rows.

    The file-weighted IBZ sum is projected with the typed axial, time-odd
    Cartesian action of the active rows, so it is the full-BZ spin moment.
    Pad bands and pad G slots of the loader are exact zeros.
    """
    import jax.numpy as jnp
    from common.collectives import gather_to_host
    from common.wfn_layout import band_sphere_spec

    w_ibz = np.asarray(wfn.kweights, dtype=np.float64)
    w_ibz = w_ibz / w_ibz.sum()
    psi = wfn.load(bands=(0, nocc), k="ibz", sharding=band_sphere_spec())

    @jax.jit
    def _spin(c):
        up, dn = c[:, :, 0], c[:, :, 1]
        overlap = jnp.sum(jnp.conj(up) * dn, axis=(1, 2))
        sz = jnp.sum(jnp.abs(up) ** 2 - jnp.abs(dn) ** 2, axis=(1, 2))
        return jnp.stack((2.0 * overlap.real, 2.0 * overlap.imag, sz), 1)

    per_k = np.asarray(gather_to_host(_spin(psi)), dtype=np.float64)
    del psi
    active_rows = np.asarray(sym.active_symmetry_rows, dtype=np.int32)
    action = sym.cartesian_action(active_rows, axial=True, time_odd=True)
    projector = np.asarray(action, dtype=np.float64).mean(axis=0)
    return projector @ (w_ibz @ per_k)


# ----------------------------------------------------------------------
#  Hellmann-Feynman group-velocity check of the loaded velocity
# ----------------------------------------------------------------------
def hf_group_velocity_check(V, eps_grid, kcrys_grid, B, kgrid):
    """Compare diagonal ``Re<n|dH/dk|n>`` to FD band slopes ``d eps_n / dk``.

    ``V`` is the full-BZ velocity ``(nk, 3, nb, nb)`` actually used below and
    ``eps_grid`` its energies.  Returns the RMS in-plane mismatch (Ry*Bohr)
    over dispersive, non-degenerate bands at k points whose central
    neighbours exist, with the RMS slope for scale and the sample count.  It
    authenticates the kinetic magnitude, units and Cartesian frame of a
    stored velocity against the WFN energies; it cannot see the nonlocal
    sign, whose velocity is almost purely off-diagonal (theory note §7).
    """
    nkx, nky, nkz = (int(x) for x in kgrid)
    Binv = np.linalg.inv(np.asarray(B, dtype=np.float64))
    # map rounded crystal coord -> full-BZ index
    key = lambda kc: (int(round(kc[0] * nkx)) % nkx,
                      int(round(kc[1] * nky)) % nky,
                      int(round(kc[2] * nkz)) % nkz)
    idx_of = {key(kc): i for i, kc in enumerate(kcrys_grid)}

    def grad_cart(ik_full, n):
        kc = kcrys_grid[ik_full]
        g_crys = np.zeros(3)
        steps = [(0, nkx), (1, nky)]                # in-plane only (kz single layer)
        for axis, N in steps:
            if N < 3:
                continue
            kp = kc.copy(); kp[axis] += 1.0 / N
            km = kc.copy(); km[axis] -= 1.0 / N
            ip, im = idx_of.get(key(kp)), idx_of.get(key(km))
            if ip is None or im is None:
                return None
            g_crys[axis] = (eps_grid[ip, n] - eps_grid[im, n]) / (2.0 / N)
        return Binv @ g_crys                         # Cartesian gradient

    mismatch, slopes = [], []
    for ik in range(len(kcrys_grid)):
        eps = eps_grid[ik]
        # pick non-degenerate, dispersive bands (diag velocity = dε/dk needs a
        # non-degenerate band; 2e-3 Ry ≈ 27 meV separation from neighbors)
        for n in range(min(eps.shape[0], 200)):
            if n + 1 < eps.shape[0] and abs(eps[n + 1] - eps[n]) < 2e-3:
                continue
            if n - 1 >= 0 and abs(eps[n] - eps[n - 1]) < 2e-3:
                continue
            gc = grad_cart(ik, n)
            if gc is None or np.linalg.norm(gc[:2]) < 0.02:
                continue
            mismatch.append(np.abs(V[ik][:, n, n].real[:2] - gc[:2]))
            slopes.append(np.abs(gc[:2]))
    if not mismatch:
        return {"rms": float("nan"), "slope_rms": float("nan"), "nsamples": 0}
    return {"rms": float(np.sqrt(np.mean(np.square(mismatch)))),
            "slope_rms": float(np.sqrt(np.mean(np.square(slopes)))),
            "nsamples": len(mismatch)}


# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False,
        description="Per-cell orbital magnetic moment (modern theory, dH/dk).")
    p.add_argument("--wfn", required=True,
                   help="WFN.h5 (BGW format, nspinor=2); the SOS route reads "
                        f"{VELOCITY_ARTIFACT} from the same directory when "
                        "present")
    p.add_argument("--nbnd", type=int, default=None,
                   help="Inner-sum band ceiling (default: all bands in file, "
                        "or all stored velocity bands)")
    p.add_argument("--nocc", type=int, default=None,
                   help="Occupied-band count (default: wfn.nelec)")
    p.add_argument("--mu", type=float, default=None,
                   help="Chemical potential in eV (default: midgap)")
    p.add_argument("--mu-scan", action="store_true",
                   help="Also report m_z at mu = VBM, midgap, CBM (Chern/dM/dmu check)")
    p.add_argument("--deps-tol", type=float, default=1.4e-3,
                   help="Degenerate-denominator skip tolerance in eV (default 1.4e-3)")
    p.add_argument("--pseudo-dir", default=None,
                   help="Directory of *.upf (default: auto-discover near WFN)")
    p.add_argument("--convergence", action="store_true",
                   help="Report m_z vs inner-m band ceiling")
    p.add_argument("--per-band", action="store_true",
                   help="Report m_z contribution per occupied band")
    p.add_argument("--out", default=None, help="Optional .npz dump of E and the sums")
    args = p.parse_args(argv)

    from runtime import initialize_communicator_stack, rank0_print
    runtime = initialize_communicator_stack()

    wfn_dir = Path(args.wfn).absolute().parent      # the run directory
    wfn_path = Path(args.wfn).resolve()
    rank0_print(f"\n[orbmag] WFN: {wfn_path}")
    wfn = WfnLoader(str(wfn_path), mesh=runtime.mesh)
    sym = wfn.symmetry()

    from file_io.dipole import wfn_psi_basis
    if wfn_psi_basis(wfn_path) == "qp":
        sys.exit(f"[orbmag] GATE orbmag_qp_wfn: got the QP WFN {wfn_path}; want "
                 "the DFT WFN of its velocity: this CLI's velocity is the DFT "
                 "p + i[V_NL, r], and with QP energies it is not d(H_QP)/dk "
                 "(claim 3218).  Fix: python -m bandstructure.htransform on the "
                 "WFN_qp.h5 with --velocity dipole_qsgw.h5 (its Sigma velocity).")
    nspinor = int(wfn.nspinor)
    if nspinor != 2:
        sys.exit(f"[orbmag] ERROR: nspinor={nspinor}. Orbital magnetization is "
                 "identically zero without spin-orbit coupling for a collinear "
                 "ferromagnet; this script requires a 2-component spinor WFN.")

    nbnd = int(args.nbnd) if args.nbnd else int(wfn.nbands)
    nbnd = min(nbnd, int(wfn.nbands))
    nocc = int(args.nocc) if args.nocc else int(wfn.nelec)
    deps_tol = args.deps_tol / RY2EV                 # eV -> Ry

    rank0_print(f"[orbmag] nspinor={nspinor}  nbnd={nbnd}  nocc={nocc}  "
                f"nk_file={int(wfn.nkpts)}  nk_full={int(sym.nk_tot)}")

    # Pseudopotentials for the nonlocal velocity (dV_NL/dk).
    pdirs = [args.pseudo_dir] if args.pseudo_dir else []
    pdirs += [str(wfn_path.parent),
              str(wfn_path.parent / ".." / "qe" / "scf"),
              str(wfn_path.parent / ".." / "qe" / "nscf")]
    pseudos = {}
    for d in pdirs:
        if d and Path(d).exists():
            pseudos = load_pseudopotentials(d)
            if pseudos:
                rank0_print(f"[orbmag] pseudopotentials from: {d}  -> "
                            f"{list(pseudos)}")
                break

    V, source = dft_velocity_full_bz(
        wfn, sym, nbnd=nbnd, mesh=runtime.mesh,
        artifact=wfn_dir / VELOCITY_ARTIFACT, pseudos=pseudos)
    rank0_print(f"[orbmag] velocity: {source}")
    nk = int(sym.nk_tot)
    w_k = 1.0 / nk                                   # uniform full-BZ weight
    E = np.asarray(wfn.energies[0], dtype=np.float64)[
        np.asarray(sym.irr_idx_k, dtype=np.int64), :nbnd]
    hf = hf_group_velocity_check(
        V, E, np.asarray(wfn.kvecs(k="full_bz"), dtype=np.float64),
        np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat),
        wfn.kgrid)
    rank0_print(
        "[orbmag] Hellmann-Feynman check of the loaded velocity: RMS "
        f"|Re diag(v) - d eps/dk| = {hf['rms']:.4f} Ry*Bohr against RMS "
        f"slope {hf['slope_rms']:.4f} ({hf['nsamples']} band/k samples). "
        "Diagnostic only: the central difference spans two grid steps "
        "and follows band index, so crossings on a coarse or flat-band "
        "grid inflate it, and it is blind to the nonlocal sign")
    PA = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
    PB = np.zeros((3, nbnd, nbnd), dtype=np.complex128)
    for ik in range(nk):
        pa, pb = orbital_pieces_at_k(V[ik], E[ik], nocc, deps_tol)
        PA += w_k * pa
        PB += w_k * pb
    del V
    cA, cB = PA.sum(axis=(1, 2)), PB.sum(axis=(1, 2))
    # The physical spin moment -mu_B sum_occ <sigma> fixes the reporting axis.
    m_spin = -spin_moment_ibz(wfn, sym, nocc=nocc)
    n_hat = m_spin / np.linalg.norm(m_spin)
    PA_band_n = np.einsum("g,gnm->nm", n_hat, PA)
    PB_band_n = np.einsum("g,gnm->nm", n_hat, PB)

    def C_of_mu(m):
        return cA - 2.0 * m * cB

    # ---- shared: chemical potential + reporting --------------------------
    VBM = float(E[:, nocc - 1].max())
    CBM = float(E[:, nocc].min()) if nocc < nbnd else VBM
    mu = args.mu / RY2EV if args.mu is not None else 0.5 * (VBM + CBM)
    gap_eV = (CBM - VBM) * RY2EV
    rank0_print(f"\n[orbmag] VBM={VBM*RY2EV:.4f} eV  CBM={CBM*RY2EV:.4f} eV  "
          f"indirect gap={gap_eV:.4f} eV   mu={mu*RY2EV:.4f} eV ({mu:.5f} Ry)")
    if gap_eV < 0:
        rank0_print("         NOTE: negative indirect gap at this k-sampling -> the "
              "moment is mu-dependent (run --mu-scan).")
    rank0_print(f"\n[orbmag] spin moment -mu_B sum_occ <sigma> = ({m_spin[0]:+.4f}, "
                f"{m_spin[1]:+.4f}, {m_spin[2]:+.4f}): |m_spin| = "
                f"{np.linalg.norm(m_spin):.3f} mu_B (g = 2)")

    m_orb = MU_B_PREFACTOR * C_of_mu(mu).imag       # (3,) mu_B, file frame
    m_orb_par = float(m_orb @ n_hat)                 # along spin-moment axis

    rank0_print("\n" + "=" * 64)
    rank0_print("ORBITAL MAGNETIC MOMENT  (per unit cell, mu_B)")
    rank0_print("=" * 64)
    rank0_print(f"  m = ({m_orb[0]:+.5f}, {m_orb[1]:+.5f}, {m_orb[2]:+.5f})  "
                "(Cartesian file frame)")
    rank0_print(f"  orbital moment along spin axis: {m_orb_par:+.5f} mu_B  "
          f"({'PARALLEL' if m_orb_par>0 else 'ANTIPARALLEL'} to spin)")
    rank0_print(f"  [full-BZ sum over {nk} k; velocity: {source}]")
    rank0_print("=" * 64)

    if args.mu_scan and nocc < nbnd:
        rank0_print("\n[orbmag] mu-scan (m along the spin axis, mu_B):")
        for label, m in [("VBM", VBM), ("midgap", 0.5 * (VBM + CBM)), ("CBM", CBM)]:
            rank0_print(f"   mu={m*RY2EV:8.4f} eV ({label:6s}):  m = "
                  f"{MU_B_PREFACTOR*float(C_of_mu(m).imag @ n_hat):+.5f}")

    if args.convergence:
        rank0_print("\n[orbmag] convergence vs inner-m band ceiling (m along the "
                    "spin axis, mu_B):")
        col = (PA_band_n - 2.0 * mu * PB_band_n).sum(axis=0)    # sum over occupied n
        cum = np.cumsum(col)                                    # partial sums over m
        for mc in sorted(set([int(0.5*nbnd), int(0.7*nbnd), int(0.85*nbnd), nbnd])):
            rank0_print(f"   mceil={mc:4d}:  m = {MU_B_PREFACTOR*float(cum[mc-1].imag):+.5f}")

    if args.per_band:
        band_n = (PA_band_n - 2.0 * mu * PB_band_n).sum(axis=1)  # per outer-n
        m_par_band = MU_B_PREFACTOR * band_n.imag
        rank0_print("\n[orbmag] per-occupied-band m (along spin axis, mu_B):")
        order = np.argsort(np.abs(m_par_band[:nocc]))[::-1]
        for n in order[:12]:
            rank0_print(f"   band {n:3d}: {m_par_band[n]:+.5f}")

    if args.out and jax.process_index() == 0:
        # colA_n/colB_n: band-resolved columns along the spin axis (summed over
        # occ n, BZ-weighted); cumsum over the inner-m index gives m vs band
        # ceiling (the band-convergence curve) at any mu.
        np.savez_compressed(args.out, E=E, mu=mu, nocc=nocc, m_orb=m_orb,
                            m_spin=m_spin, spin_axis=n_hat,
                            colA_n=PA_band_n.sum(axis=0),
                            colB_n=PB_band_n.sum(axis=0), cA=cA, cB=cB,
                            velocity_source=source)
        rank0_print(f"\n[orbmag] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
