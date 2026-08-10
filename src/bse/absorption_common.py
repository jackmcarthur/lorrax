"""Shared utilities for BSE absorption-spectrum post-processors.

Two routes consume these:
  - ``absorption_eigvecs``   — explicit Ritz vectors A^S from eigenvectors.h5
  - ``absorption_haydock``   — continued-fraction recursion (no eigenvectors)

Common needs:
  - read BGW-format eigenvectors.h5 (our writer matches this spec)
  - read dipole.h5 written by psp.get_dipole_mtxels
  - slice the (nb, nb) dipole window down to the BSE (nc, nv) sub-block
  - Lorentzian broaden
  - Kramers-Kronig
  - write BGW-style 4-column ``absorption_b{1,2,3}.dat``
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

RYD2EV = 13.6056980659


def exciton_dipole_projections(A, d_alpha):
    """⟨0|r̂_α|S⟩ — THE contraction of a BSE eigenvector against the dipole.

    THIS IS THE SINGLE SITE.  Every absorption driver that owns explicit
    eigenvectors calls this and nothing else, because the tree has already
    paid once for two drivers spelling this contraction differently
    (``KNOWN_FAILURES.md``, "THE TWO ABSORPTION DRIVERS DISAGREE ON A
    CONJUGATION").  The conjugate sits on the DIPOLE, not on ``A``.

    DERIVATION, from the spectral representation and this tree's own
    storage conventions — not from BerkeleyGW habit:

      * The exciton state is ``|S⟩ = Σ_t A^S_t â†_ck â_vk |0⟩`` over
        transitions ``t = (c, v, k)``.  That ``A`` is the amplitude in
        THAT basis is fixed by LORRAX's own kernel, not assumed: the
        exchange term is assembled as ``K^x = M V M†`` with
        ``M_t = conj(ψ_c) ψ_v`` (``bse_simple.py``'s V-term, and the
        direct term ``K^d_{tt'} = −Σ conj(ψ_c[k]) ψ_c'[k'] W
        ψ_v[k] conj(ψ_v'[k'])`` documented at ``bse_nontda.py:169``).
        Both are the standard ``⟨t|H|t'⟩`` with the CONDUCTION index on
        the bra, so the operator the solvers diagonalise is H itself and
        not its complex conjugate, and ``A`` is the amplitude and not its
        conjugate.
      * ``slice_dipole_to_bse_window`` returns
        ``d_t = ⟨ck|v̂_α|vk⟩ / ΔE`` — conduction on the BRA (the producer's
        own convention, ``dipole_cart[α,k,m,n] = ⟨mk|v̂_α|nk⟩``).
      * Therefore
        ``⟨0|r̂_α|S⟩ = Σ_t A^S_t ⟨0|r̂_α â†_c â_v|0⟩ = Σ_t A^S_t ⟨vk|r̂_α|ck⟩
                     = Σ_t A^S_t · conj(d^α_t)``.

    Two independent witnesses agree with that line, and they are the
    reason this is a verdict rather than a preference:

      * BerkeleyGW does the identical contraction —
        ``BSE/diag.f90:711`` is ``dipoles_r = Σ u_r · MYCONJG(s1)``, and
        ``Common/mtxel_optical.f90``'s ``mtxel_m``/``mtxel_v`` build
        ``s0(ic,iv) = ⟨ic,k|…|iv,k⟩ / ΔE``, the same index order as ours.
      * The Haydock route needs no eigenvector convention at all: it
        evaluates ``⟨d|(z−H)⁻¹|d⟩``, whose spectral weights are
        ``|⟨S|d⟩|² = |Σ_t conj(A_t) d_t|²`` — the modulus of the
        expression above.  A sum-over-states spectrum built any other way
        does not reproduce the resolvent, which is what
        ``tests/test_absorption_conjugation.py`` measures.

    Note the phase convention: ``Σ_t A_t conj(d_t)`` and
    ``Σ_t conj(A_t) d_t`` have the SAME modulus (they are complex
    conjugates), so ε₂ cannot tell them apart — but the BGW-format
    ``eigenvalues_b*.dat`` writes ``Re`` and ``Im`` in separate columns,
    so the tree emits the BGW-side spelling.  What is NOT a phase choice,
    and is the defect this function closes, is contracting ``A`` with a
    bare ``d``: ``Σ_t A_t d_t`` is a different number in modulus too,
    measured at up to 6.8× per element on the committed fixtures.

    Parameters
    ----------
    A : (N, *T) complex
        Eigenvectors.  Leading axis is the state; the trailing axes are
        the transition block in whatever order the caller holds it.
    d_alpha : (3, *T) complex
        Dipole ``d^α_t``, with the SAME trailing transition axes as ``A``
        (the two drivers hold them in different orders — ``(nk, nc, nv)``
        and ``(nc, nv, nk)`` — and this function is layout-agnostic
        precisely so neither has to reshape at the call site).

    Returns
    -------
    (N, 3) complex128 — ⟨0|r̂_α|S⟩ per state, per polarisation.
    """
    A = np.asarray(A)
    d_alpha = np.asarray(d_alpha)
    if A.shape[1:] != d_alpha.shape[1:]:
        raise ValueError(
            f"eigenvector transition axes {A.shape[1:]} do not match the "
            f"dipole's {d_alpha.shape[1:]}.  Both must be indexed over the "
            f"same (c, v, k) block in the same order; a transpose here is a "
            f"silently wrong oscillator strength, not a broadcast.")
    n_state = A.shape[0]
    n_trans = int(np.prod(A.shape[1:])) if A.ndim > 1 else 1
    return (A.reshape(n_state, n_trans)
            @ np.conj(d_alpha.reshape(d_alpha.shape[0], n_trans)).T
            ).astype(np.complex128)


def load_eigenvectors_h5(path: str | Path):
    """Load BGW-format eigenvectors.h5 (TDA, single Q, single spin assumed).

    Returns
    -------
    eigvals : (N,) float64 — exciton energies in Ry.
    A       : (N, nk, nc, nv) complex128 — eigenvectors in BSE basis.
    params  : dict with keys ``nc, nv, ns, nspinor, nk, kpts, spin_kernel``.

    Note: BGW convention ``ns`` is the spin index (=2 for collinear-spin,
    =1 for spinor or non-spin), distinct from ``nspinor`` (=2 for spinor,
    inferred here from ``spin_kernel == 3``).

    NON-TDA FILES ARE REFUSED, not silently truncated.  ``bse_io``'s
    writer persists the resonant ``X`` to ``exciton_data/eigenvectors``
    and the coupling ``Y`` to a sibling ``eigenvectors_coupling``; this
    reader only ever returned the first.  Reading a non-TDA file and
    contracting ``X`` alone gives the TDA answer for a non-TDA solve,
    which is wrong by the whole anti-resonant channel — see the
    ``NotImplementedError`` below for the contraction it would need.
    """
    with h5py.File(str(path), "r") as f:
        eigvals_eV = np.asarray(f["exciton_data/eigenvalues"][:], dtype=np.float64)
        evecs = np.asarray(f["exciton_data/eigenvectors"][:])  # (nQ,N,nk,nc,nv,ns,2)
        has_coupling = "eigenvectors_coupling" in f["exciton_data"]
        # ``use_tda`` is ours (and BGW's) but not guaranteed on every writer's
        # output, so the presence of the coupling block is the second witness.
        use_tda_ds = f.get("exciton_header/params/use_tda")
        use_tda = True if use_tda_ds is None else bool(int(use_tda_ds[()]))
        if has_coupling or not use_tda:
            raise NotImplementedError(
                f"{path!s} is a NON-TDA eigenvector file "
                f"(use_tda={use_tda}, coupling block "
                f"{'present' if has_coupling else 'absent'}), and the "
                f"sum-over-states absorption route is TDA-only.\n"
                f"The full-BSE oscillator strength is NOT |Σ_t X_t conj(d_t)|²: "
                f"it needs both halves of the paired vector and both the left "
                f"and right solutions,\n"
                f"    ⟨0|r|S⟩_r = Σ_t X^S_t conj(d_t) − Σ_t Y^S_t d_t\n"
                f"    ⟨0|r|S⟩_l = Σ_t X̃^S_t conj(d_t) + Σ_t Ỹ^S_t d_t\n"
                f"    f_S       = Re[ conj(⟨0|r|S⟩_l) · ⟨0|r|S⟩_r ]\n"
                f"(the anti-resonant dipole is −conj(d) because "
                f"s_(c→v) = −conj(s_(v→c)), BerkeleyGW BSE/diag.f90:722-731), "
                f"and no driver in this tree assembles it yet.  Solve with "
                f"--tda, or add the coupling contraction before reading this "
                f"file for absorption.")
        spin_kernel = int(f["exciton_header/params/spin_kernel"][()])
        params = {
            "nc": int(f["exciton_header/params/nc"][()]),
            "nv": int(f["exciton_header/params/nv"][()]),
            "ns": int(f["exciton_header/params/ns"][()]),
            "nspinor": 2 if spin_kernel == 3 else 1,
            "spin_kernel": spin_kernel,
            "nk": int(f["exciton_header/kpoints/nk"][()]),
            "kpts": np.asarray(f["exciton_header/kpoints/kpts"][:]),
        }
    nQ = evecs.shape[0]
    ns = params["ns"]
    if nQ != 1 or ns != 1:
        raise NotImplementedError(
            f"Only nQ=ns=1 supported (got nQ={nQ}, ns={ns})")
    A = evecs[..., 0] + 1j * evecs[..., 1]   # (nQ, N, nk, nc, nv, ns)
    A = A[0, :, :, :, :, 0]                  # (N, nk, nc, nv)
    # BGW convention: v=0 is the highest valence band (closest to gap).
    # Our absorption-side internal convention slices the dipole as
    # ``val_idx = n_occ - n_val .. n_occ`` (lowest-first), so flip the
    # valence axis here on read to match.
    A = A[:, :, :, ::-1]
    # File stores eigenvalues in eV (BGW convention) — convert back to Ry
    # for the rest of the absorption pipeline which works in Ry.
    eigvals_Ry = eigvals_eV / RYD2EV
    return eigvals_Ry, A.astype(np.complex128), params


def load_dipole_h5(path: str | Path):
    """Load dipole.h5 (psp.get_dipole_mtxels output).

    Returns
    -------
    dipole_cart : (3, nk, nb, nb) complex128 — ⟨mk|v̂_α|nk⟩ (Ry), at the arm
                  the file was built with (its ``prov_vnl_velocity_sign``).
    deltaE      : (nk, nb, nb) float64       — E_b - E_b' (Ry).
    attrs       : dict with ``nbands, nk``.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"dipole file {path!s} not found.  This is a PRODUCED input, not a "
            f"deck file that ships with the system: build it from the deck's "
            f"WFN with\n"
            f"    python3 -u -m psp.get_dipole_mtxels -i <deck>.in --skip-vnl "
            f"--out {Path(path).name}\n"
            f"``--skip-vnl`` writes the momentum operator only, which is the "
            f"arm that matches BerkeleyGW's ``use_momentum``; drop it to get "
            f"the full velocity including the nonlocal commutator.")
    with h5py.File(str(path), "r") as f:
        dipole_cart = np.asarray(f["dipole_cart"][:], dtype=np.complex128)
        deltaE = np.asarray(f["deltaE"][:], dtype=np.float64)
        attrs = {"nbands": int(f.attrs["nbands"]), "nk": int(f.attrs["nk"])}
    return dipole_cart, deltaE, attrs


def slice_dipole_to_bse_window(dipole_cart, deltaE, n_occ, n_val, n_cond):
    """Slice (nb, nb) dipole tables → BSE (nc, nv) sub-block.

    Convention: ``dipole_cart[α, k, m, n] = ⟨mk|v̂_α|nk⟩``.
    For the dipole oscillator we need ⟨ck|v̂|vk⟩ ⇒ m=c, n=v.

    Returns
    -------
    d_alpha : (3, nk, nc, nv) complex — d^α_{cvk} = ⟨c|v̂|v⟩ / ΔE   (r-form).
    de_cv   : (nk, nc, nv) float      — E_c - E_v (Ry, > 0).
    """
    val_lo = n_occ - n_val
    cond_hi = n_occ + n_cond
    v_cv = dipole_cart[:, :, n_occ:cond_hi, val_lo:n_occ]   # (3, nk, nc, nv)
    de_cv = deltaE[:, n_occ:cond_hi, val_lo:n_occ]          # (nk, nc, nv)
    d_alpha = v_cv / de_cv[None]
    return d_alpha, de_cv


def build_dipole_vector_bse(d_alpha, n_cond_pad=None, n_val_pad=None):
    """Reshape (3, nk, nc, nv) → (3, nc_pad, nv_pad, nk) BSE block-vector form.

    Pads ``nc → nc_pad`` and ``nv → nv_pad`` with zeros if requested (the
    BSE sharded matvec requires divisibility by the (x, y) mesh shape).
    """
    npol, nk, nc, nv = d_alpha.shape
    nc_pad = nc if n_cond_pad is None else n_cond_pad
    nv_pad = nv if n_val_pad is None else n_val_pad
    out = np.zeros((npol, nc_pad, nv_pad, nk), dtype=d_alpha.dtype)
    out[:, :nc, :nv, :] = np.transpose(d_alpha, (0, 2, 3, 1))
    return out


def lorentzian_broaden(omegas, energies, weights, eta):
    """ε(ω) = Σ_i weights[i] · η/π / ((ω - E_i)² + η²)."""
    omegas = np.asarray(omegas)
    energies = np.asarray(energies)
    weights = np.asarray(weights)
    delta = omegas[:, None] - energies[None, :]
    L = (eta / np.pi) / (delta * delta + eta * eta)
    return L @ weights


def kramers_kronig_eps1(omegas, eps2):
    """Naive principal-value KK: ε₁(ω) = 1 + (2/π) P ∫ ω' ε₂(ω')/(ω'² - ω²) dω'.

    O(N²) over uniform omega grid. For ~1500 points this is sub-second.
    """
    omegas = np.asarray(omegas)
    eps2 = np.asarray(eps2)
    n = omegas.size
    if n < 2:
        return np.ones_like(eps2)
    do = omegas[1] - omegas[0]
    eps1 = np.ones_like(eps2)
    for i in range(n):
        mask = np.arange(n) != i
        kernel = omegas[mask] * eps2[mask] / (omegas[mask] ** 2 - omegas[i] ** 2)
        eps1[i] = 1.0 + (2.0 / np.pi) * np.sum(kernel) * do
    return eps1


def write_absorption_dat(path, omegas_eV, eps2, eps1, jdos, header_lines=None):
    """4-column BGW-style ``absorption_*.dat``.

    Matches BGW's ``BSE/absp.f90:182`` format ``4f16.9``: columns are
    ω(eV), ε₂(ω), ε₁(ω), JDOS(ω).
    """
    with open(str(path), "w") as f:
        if header_lines:
            for line in header_lines:
                f.write(f"# {line}\n")
        f.write(" # Column 1: omega\n")
        f.write(" # Column 2: eps2(omega)\n")
        f.write(" # Column 3: eps1(omega)\n")
        f.write(" # Column 4: JDOS(omega)\n")
        for o, e2, e1, jd in zip(omegas_eV, eps2, eps1, jdos):
            f.write(f" {o:15.9f}  {e2:14.9f}  {e1:14.9f}  {jd:14.9f}\n")


def write_eigenvalues_dat(
    path, eigvals_eV, dipoles_pol, *, n_spin, n_spinor, vol_supercell,
):
    """BGW-format ``eigenvalues_b{1,2,3}.dat`` (matches ``BSE/absp_io.f90:122``).

    Columns (TDA, 4 per row, ``4e16.8``):
      ``E_S (eV)   |⟨0|r̂|S⟩|²   Re⟨0|r̂|S⟩   Im⟨0|r̂|S⟩``

    Header lines:
      ``# neig  =       N``
      ``# vol   =  <V_supercell in bohr³>``
      ``# nspin, nspinor =  ns  nspinor``

    ``vol_supercell = V_unit_cell · N_k`` (BGW convention — what BGW prints
    in the header line).
    """
    n = eigvals_eV.size
    with open(str(path), "w") as f:
        f.write(f"# neig  = {n:9d}\n")
        f.write(f"# vol   = {vol_supercell:15.9E}\n")
        f.write(f"# nspin, nspinor = {n_spin:8d}{n_spinor:8d}\n")
        f.write("#       eig (eV)   abs(dipole)^2      Re(dipole)    Im(dipole)\n")
        for E_S, d_S in zip(eigvals_eV, dipoles_pol):
            mag2 = float((d_S.conjugate() * d_S).real)
            f.write(f"  {E_S:14.8E}  {mag2:14.8E}  {d_S.real:14.8E}  {d_S.imag:14.8E}\n")


def write_absorption_h5(
    path,
    *,
    omegas_eV,
    eps2_3pol,
    eps1_3pol,
    jdos_3pol,
    eigenvalues_Ry: Optional[np.ndarray] = None,
    oscillator_strengths: Optional[np.ndarray] = None,
    haydock_alphas: Optional[np.ndarray] = None,
    haydock_betas: Optional[np.ndarray] = None,
    haydock_norms: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
):
    """Combined H5: ω grid, 3-polarisation ε₂/ε₁/JDOS, plus route artifacts."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("omega_eV", data=omegas_eV)
        f.create_dataset("eps2", data=eps2_3pol)   # (n_omega, 3)
        f.create_dataset("eps1", data=eps1_3pol)
        f.create_dataset("jdos", data=jdos_3pol)
        if eigenvalues_Ry is not None:
            f.create_dataset("eigenvalues_Ry", data=eigenvalues_Ry)
        if oscillator_strengths is not None:
            f.create_dataset("oscillator_strengths", data=oscillator_strengths)  # (N, 3)
        if haydock_alphas is not None:
            f.create_dataset("haydock_alphas", data=haydock_alphas)  # (3, n_iter)
        if haydock_betas is not None:
            f.create_dataset("haydock_betas", data=haydock_betas)
        if haydock_norms is not None:
            f.create_dataset("haydock_norms", data=haydock_norms)   # (3,)
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


__all__ = [
    "RYD2EV",
    "exciton_dipole_projections",
    "load_eigenvectors_h5",
    "load_dipole_h5",
    "slice_dipole_to_bse_window",
    "build_dipole_vector_bse",
    "lorentzian_broaden",
    "kramers_kronig_eps1",
    "write_absorption_dat",
    "write_absorption_h5",
    "write_eigenvalues_dat",
]
