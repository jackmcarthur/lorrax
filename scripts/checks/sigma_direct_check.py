"""Direct q=Gamma verification for W(0), W(i*omega_p), and Sigma branch signs.

Intent
------
This script is a focused debugging driver for the GN-PPM frequency path.  It is
not a production workflow and intentionally prioritizes transparent equations
over performance.  The goal is to verify, with minimal moving parts, that:

1) W(0) and W(i*omega_p) are constructed correctly from a trivial sum-over-
   states chi0 and a direct Dyson solve in the ISDF basis.
2) Head corrections are injected consistently into V and W at omega=0 and
   omega=i*omega_p, either from automatic sources (epshead/s_tensor) or user
   overrides.
3) GN-PPM fit parameters (B, Omega) extracted from W^c(0), W^c(i*omega_p)
   produce sensible Sigma_c^(+) and Sigma_c^(-) diagonal matrix elements when
   evaluated directly at DFT energies.
4) Bare exchange Sigma_x (with V including head) matches gw_jax conventions.
5) Invalid GN modes can be mapped to a static fallback correction using the
   same policy as gw_jax static-limit mode:
      Sigma_invalid = Sigma_occ(Wc0_invalid) - 0.5 * Sigma_RI(Wc0_invalid)
   where Wc0_invalid keeps only entries with invalid fitted Omega.

Input Design
------------
The direct-check input file is intentionally short and explicit.  It defines:

- Band partition for this debug run:
  - total valence count
  - number of conduction bands to report
  - total number of states used in the explicit sums
- Optional band offset (default 0)
- omega_p in Ry
- Optional head overrides:
  - v_head_au
  - w0_head_au
  - wiwp_head_au

`gw_input` points to the usual GW input so path resolution and automatic head
source logic (epshead/s_tensor) can reuse the same conventions as gw_jax.

Notes / Limitations
-------------------
- This driver currently targets the q=Gamma, single-q debug case (as requested
  for the CO molecule test).
- Restart files provide V_qmunu, V0_noG0_munu, G0_mu_nu, psi_full_y, and
  enk_full.  They do not provide precomputed W(i*omega_p) heads; those are
  either computed from auxiliary files (eps0mat/dipole + WFN metadata) or
  provided via overrides.

Status
------
NOTHING IMPORTS THIS AND NOTHING RUNS IT, which is how it came to sit
broken for a month: `1fb160c` (2026-07-08) deleted
`extract_gn_ppm_parameters_from_Wc` under a "grep-verified zero callers"
claim whose grep covered `src/` and not `scripts/`, and the resulting
`ImportError` surfaced only when a survey went looking on 2026-08-08.
The call was repointed at the surviving `fit_gn_ppm_from_wc_pair` (see
the note at the call site) because the deleted function was pure
shape-adaptation over it -- no physics was lost and none had to be
restored.

The durable fix is not in this file.  A debug driver with no importer is
invisible to exactly the kind of deletion sweep that broke it, so either
`scripts/` gets a collection gate that at least imports every module
under it, or a driver that has outlived its question gets deleted rather
than left to rot.  This one has not outlived its question -- the GN-PPM
branch signs and the static-fallback policy it checks are both live, and
it is the only place they are checked against a transparent
sum-over-states reference -- so the recommendation is the gate.
"""

from __future__ import annotations

import argparse
import configparser
import glob
import os
from dataclasses import dataclass
from typing import Callable

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from common import Meta
from common.chi_from_dipole import compute_S_omega, read_dipole_h5
from gw.gw_config import read_cohsex_input
from gw.minimax_screening import fit_gn_ppm_from_wc_pair
from gw.vcoul import compute_q0_averages
from file_io import EPSReader, resolve_input_paths
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

# x64, EXPLICITLY, because this script has no ``runtime.bootstrap()`` and
# ``import jax`` above already resolved the flag — so the
# ``os.environ.setdefault("JAX_ENABLE_X64", "1")`` backstop in
# ``src/gw/__init__.py`` fires too late to matter here.  It has to happen
# before the ``symmetry_maps`` import below, whose module scope evaluates
# ``jnp.int64(10**12)`` and raises OverflowError under x32; the quieter
# consequence is that every ``jnp.complex128`` in this file would
# otherwise have been complex64.  Same call ``gw.gw_jax`` makes after its
# own bootstrap.
jax.config.update("jax_enable_x64", True)

from wfn_loader import WfnLoader                                    # noqa: E402
import symmetry_maps                                            # noqa: E402

RYD2EV = 13.6056980659


@dataclass(frozen=True)
class DirectCheckConfig:
    gw_input: str
    restart_file: str | None
    n_valence: int
    n_cond_solve: int
    n_sum_states: int
    band_offset: int
    omega_p_ry: float
    v_head_au: float | None
    w0_head_au: float | None
    wiwp_head_au: float | None
    wcoul0_source: str
    wcoul0_eta: float


def _parse_optional_float(section: configparser.SectionProxy, key: str) -> float | None:
    raw = section.get(key, fallback="").strip()
    if raw == "":
        return None
    return float(raw)


def read_direct_check_input(path: str) -> DirectCheckConfig:
    cfg = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as fh:
        cfg.read_file(fh)
    if "sigma_direct_check" not in cfg:
        raise ValueError(f"{path} is missing [sigma_direct_check] section")
    s = cfg["sigma_direct_check"]
    gw_input = s.get("gw_input", fallback="").strip()
    if not gw_input:
        raise ValueError("sigma_direct_check input requires gw_input")
    restart_file = s.get("restart_file", fallback="").strip() or None
    n_valence = s.getint("n_valence")
    n_cond_solve = s.getint("n_cond_solve")
    n_sum_states = s.getint("n_sum_states")
    band_offset = s.getint("band_offset", fallback=0)
    omega_p_ry = s.getfloat("omega_p_ry", fallback=2.0)
    wcoul0_source = s.get("wcoul0_source", fallback="s_tensor").strip().lower()
    wcoul0_eta = s.getfloat("wcoul0_eta", fallback=0.0)
    return DirectCheckConfig(
        gw_input=gw_input,
        restart_file=restart_file,
        n_valence=n_valence,
        n_cond_solve=n_cond_solve,
        n_sum_states=n_sum_states,
        band_offset=band_offset,
        omega_p_ry=omega_p_ry,
        v_head_au=_parse_optional_float(s, "v_head_au"),
        w0_head_au=_parse_optional_float(s, "w0_head_au"),
        wiwp_head_au=_parse_optional_float(s, "wiwp_head_au"),
        wcoul0_source=wcoul0_source,
        wcoul0_eta=wcoul0_eta,
    )


def _find_restart_file(gw_input_path: str) -> str:
    input_dir = os.path.dirname(os.path.abspath(gw_input_path))
    cands: list[str] = []
    cands.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    cands.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    if not cands:
        raise FileNotFoundError(f"Could not find isdf_tensors_*.h5 under {input_dir}")
    return cands[-1]


def _determine_wcoul0(
    *,
    source: str,
    omega: complex,
    eta: float,
    params: dict,
    input_dir: str,
    wfn: WfnLoader,
    sym,
    meta: Meta,
    print_fn: Callable[[str], None],
) -> tuple[complex, complex, str]:
    """Resolve (v_head, w_head) using epshead or dipole S(omega), matching gw_jax policy."""
    source = source.strip().lower()
    if source not in ("epshead", "s_tensor"):
        source = "epshead"
    omega_c = complex(omega)

    eps0_path = os.path.join(input_dir, "eps0mat.h5")
    dipole_path = os.path.join(input_dir, "dipole.h5")

    def from_epshead():
        if not os.path.exists(eps0_path):
            return None
        eps0 = EPSReader(eps0_path)
        if abs(omega_c) > 1.0e-14:
            print_fn(
                f"  wcoul0_source=epshead is static-only; using epshead(0) for omega={omega_c} Ry"
            )
        vc0, wc0 = compute_q0_averages(
            wfn,
            jnp.asarray(eps0.epshead, dtype=jnp.complex128),
            meta,
            S_cart=None,
        )
        label = "epshead(0)" if abs(omega_c) > 1.0e-14 else "epshead"
        return complex(vc0), complex(wc0), label

    def from_s_tensor():
        if not os.path.exists(dipole_path):
            return None
        dipole_cart, delta_e = read_dipole_h5(dipole_path)
        nk_tot = int(sym.nk_tot)
        nb = int(dipole_cart.shape[2])
        nelec = int(wfn.nelec)
        occ = np.zeros((nk_tot, nb), dtype=np.float64)
        occ[:, : max(0, min(nelec, nb))] = 1.0
        s_cart = compute_S_omega(
            dipole_cart,
            delta_e,
            jnp.asarray(occ, dtype=jnp.float64),
            float(wfn.cell_volume),
            int(sym.nk_tot),
            int(wfn.nspin),
            int(wfn.nspinor),
            jnp.asarray([omega_c], dtype=jnp.complex128),
            eta=float(eta),
        )[0]
        vc0, wc0 = compute_q0_averages(
            wfn,
            jnp.asarray(0.0, dtype=jnp.float64),
            meta,
            S_cart=s_cart,
        )
        label = "s_tensor" if abs(omega_c) <= 1.0e-14 else f"s_tensor(omega={omega_c} Ry)"
        return complex(vc0), complex(wc0), label

    order = [source] + [s for s in ("epshead", "s_tensor") if s != source]
    for s in order:
        out = from_epshead() if s == "epshead" else from_s_tensor()
        if out is not None:
            return out
    raise RuntimeError(
        "Failed to determine wcoul0 automatically (need eps0mat.h5 or dipole.h5). "
        "Provide v_head_au/w0_head_au/wiwp_head_au overrides."
    )


def _chi0_q0_direct(
    *,
    psi_k: np.ndarray,   # (nb, nspinor, nrmu)
    enk_k: np.ndarray,   # (nb,)
    n_valence: int,
    n_sum_states: int,
    omega: complex,
) -> np.ndarray:
    """Direct q=Gamma chi0 sum over states in the selected band window."""
    b0 = 0
    b1 = int(n_sum_states)
    psi = psi_k[b0:b1]  # (nsum, ns, mu)
    e = enk_k[b0:b1]    # (nsum,)
    nv = int(n_valence)
    if not (0 < nv < psi.shape[0]):
        raise ValueError(f"Invalid n_valence={nv} for n_sum_states={psi.shape[0]}")
    psi_v = psi[:nv]
    psi_c = psi[nv:]
    e_v = e[:nv]
    e_c = e[nv:]

    # rho_{cv,mu} = sum_s psi_c(s,mu) conj(psi_v(s,mu))
    rho_cv = np.einsum("csm,vsm->cvm", psi_c, np.conj(psi_v), optimize=True)
    d_e = e_c[:, None] - e_v[None, :]
    omega2 = complex(omega) ** 2
    denom = d_e * d_e - omega2
    w = np.where(np.abs(denom) > 1.0e-30, d_e / denom, 0.0 + 0.0j)
    chi = np.einsum("cv,cvm,cvn->mn", w, np.conj(rho_cv), rho_cv, optimize=True)
    return (-2.0) * chi.astype(np.complex128)  # Nk=1 for this direct check


def _solve_w_from_chi_direct(v_munu: np.ndarray, chi_munu: np.ndarray, pref: float) -> np.ndarray:
    a = np.eye(v_munu.shape[0], dtype=np.complex128) - v_munu @ (pref * chi_munu)
    return np.linalg.solve(a, v_munu)


def _build_occ_mask(nb: int, n_valence: int) -> np.ndarray:
    occ = np.zeros((nb,), dtype=bool)
    occ[: int(n_valence)] = True
    return occ


def _sigma_x_diag(
    *,
    psi_k: np.ndarray,        # (nb, ns, mu)
    v_head: np.ndarray,       # (mu, mu)
    n_valence: int,
    n_solve: int,
) -> np.ndarray:
    """Direct diagonal bare exchange with V including head."""
    occ_idx = np.arange(int(n_valence), dtype=int)
    n_solve = int(n_solve)
    out = np.zeros((n_solve,), dtype=np.complex128)
    for n in range(n_solve):
        psi_n = psi_k[n]
        m = np.einsum("vsm,sm->vm", np.conj(psi_k[occ_idx]), psi_n, optimize=True)  # (nv,mu)
        mm = np.conj(m[:, :, None]) * m[:, None, :]
        out[n] = -np.sum(mm * v_head[None, :, :])
    return out


def _sigma_c_branches_diag(
    *,
    psi_k: np.ndarray,         # (nb_sum, ns, mu)
    enk_k: np.ndarray,         # (nb_sum,)
    n_valence: int,
    n_solve: int,
    efermi: float,
    b_munu: np.ndarray,        # (mu,mu)
    omega_munu: np.ndarray,    # (mu,mu)
    valid_mask: np.ndarray,    # (mu,mu)
) -> tuple[np.ndarray, np.ndarray]:
    """Direct Sigma_c^(+) and Sigma_c^(-) diagonals at omega_rel = E_n - E_F."""
    nv = int(n_valence)
    nsum = psi_k.shape[0]
    nsolve = int(n_solve)
    tiny = 1.0e-30
    occ_mask = _build_occ_mask(nsum, nv)
    h_v = efermi - enk_k[:nv]                    # (nv,)
    e_c = enk_k[nv:] - efermi                    # (nc,)
    sig_p = np.zeros((nsolve,), dtype=np.complex128)
    sig_m = np.zeros((nsolve,), dtype=np.complex128)
    mask = np.asarray(valid_mask, dtype=bool)

    psi_occ = psi_k[:nv]
    psi_unocc = psi_k[nv:]
    for n in range(nsolve):
        psi_n = psi_k[n]
        omega_rel = float(enk_k[n] - efermi)

        m_v = np.einsum("vsm,sm->vm", np.conj(psi_occ), psi_n, optimize=True)
        mm_v = np.conj(m_v[:, :, None]) * m_v[:, None, :]
        d_p = omega_rel + h_v[:, None, None] + omega_munu[None, :, :]
        safe_p = mask[None, :, :] & (np.abs(d_p) > tiny)
        contrib_p = np.where(safe_p, mm_v * b_munu[None, :, :] / d_p, 0.0 + 0.0j)
        sig_p[n] = -np.sum(contrib_p)

        m_c = np.einsum("csm,sm->cm", np.conj(psi_unocc), psi_n, optimize=True)
        mm_c = np.conj(m_c[:, :, None]) * m_c[:, None, :]
        d_m = omega_rel - e_c[:, None, None] - omega_munu[None, :, :]
        safe_m = mask[None, :, :] & (np.abs(d_m) > tiny)
        contrib_m = np.where(safe_m, mm_c * b_munu[None, :, :] / d_m, 0.0 + 0.0j)
        sig_m[n] = -np.sum(contrib_m)

    return sig_p, sig_m


def _print_sigma_table(
    *,
    enk_solve: np.ndarray,
    efermi: float,
    sigma_x: np.ndarray,
    sigma_p: np.ndarray,
    sigma_m: np.ndarray,
    sigma_invalid_static: np.ndarray | None,
    band_offset: int,
) -> None:
    print("")
    print("Direct Sigma table at q=Gamma (omega = E_n^DFT - E_F)")
    if sigma_invalid_static is None:
        print("n\tE_dft(eV)\tE-Ef(eV)\tSigma_X(eV)\tSigma_Wc+(eV)\tSigma_Wc-(eV)\tSigma_C_tot(eV)")
    else:
        print(
            "n\tE_dft(eV)\tE-Ef(eV)\tSigma_X(eV)\tSigma_Wc+(eV)\tSigma_Wc-(eV)\t"
            "Sigma_C_tot(eV)\tSigma_C_invalid_static(eV)"
        )
    for i in range(enk_solve.shape[0]):
        n = int(band_offset + i)
        e_ev = float(enk_solve[i] * RYD2EV)
        er_ev = float((enk_solve[i] - efermi) * RYD2EV)
        sx = complex(sigma_x[i]) * RYD2EV
        sp = complex(sigma_p[i]) * RYD2EV
        sm = complex(sigma_m[i]) * RYD2EV
        sc = sp + sm
        if sigma_invalid_static is None:
            print(
                f"{n:d}\t{e_ev:.6f}\t{er_ev:.6f}\t"
                f"{sx.real:.6f}\t{sp.real:.6f}\t{sm.real:.6f}\t{sc.real:.6f}"
            )
        else:
            si = complex(sigma_invalid_static[i]) * RYD2EV
            print(
                f"{n:d}\t{e_ev:.6f}\t{er_ev:.6f}\t"
                f"{sx.real:.6f}\t{sp.real:.6f}\t{sm.real:.6f}\t{sc.real:.6f}\t{si.real:.6f}"
            )


def run(cfg_path: str) -> int:
    cfg = read_direct_check_input(cfg_path)
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    gw_input = cfg.gw_input if os.path.isabs(cfg.gw_input) else os.path.join(cfg_dir, cfg.gw_input)
    params = read_cohsex_input(gw_input)
    gw_input_dir = os.path.dirname(os.path.abspath(gw_input))
    resolve_input_paths(params, gw_input_dir)

    restart_file = cfg.restart_file
    if restart_file is None:
        restart_file = _find_restart_file(gw_input)
    elif not os.path.isabs(restart_file):
        restart_file = os.path.join(cfg_dir, restart_file)

    wfn = WfnLoader(params["wfn_file"])
    sym = symmetry_maps.SymMaps(wfn)
    nrmu = None
    with h5py.File(restart_file, "r") as h5:
        v_q = np.asarray(h5["V_qmunu"][0, 0, 0, 0, 0, 0], dtype=np.complex128)
        v_q0_no_g0 = np.asarray(h5["V0_noG0_munu"], dtype=np.complex128)
        g0 = np.asarray(h5["G0_mu_nu"], dtype=np.complex128)
        psi_full = np.asarray(h5["psi_full_y"], dtype=np.complex128)
        enk_full = np.asarray(h5["enk_full"], dtype=np.float64)
        nrmu = int(psi_full.shape[-1])

    if psi_full.shape[0] != 1 or enk_full.shape[0] != 1:
        raise ValueError("Direct checker currently supports only nk=1 (Gamma-only) restart files.")

    b0 = int(cfg.band_offset)
    nsolve = int(cfg.n_valence + cfg.n_cond_solve)
    bsolve1 = b0 + nsolve
    bsum1 = b0 + int(cfg.n_sum_states)
    nb_tot = int(psi_full.shape[1])
    if bsum1 > nb_tot:
        raise ValueError(
            f"Requested n_sum_states with band_offset exceeds available bands: "
            f"offset={b0}, n_sum={cfg.n_sum_states}, nb_tot={nb_tot}"
        )
    if bsolve1 > bsum1:
        raise ValueError("n_valence + n_cond_solve must be <= n_sum_states.")
    if cfg.n_valence <= 0 or cfg.n_valence >= cfg.n_sum_states:
        raise ValueError("n_valence must be in (0, n_sum_states).")

    psi_k_sum = psi_full[0, b0:bsum1]      # (nsum, nspinor, mu)
    enk_k_sum = enk_full[0, b0:bsum1]      # (nsum,)
    psi_k_solve = psi_full[0, b0:bsolve1]
    enk_k_solve = enk_full[0, b0:bsolve1]

    # Midgap reference for the selected valence/conduction partition.
    e_vbm = float(np.max(enk_k_sum[: cfg.n_valence]))
    e_cbm = float(np.min(enk_k_sum[cfg.n_valence :]))
    efermi = 0.5 * (e_vbm + e_cbm)

    # Direct chi0 and Dyson W on q=Gamma (no-head V).
    chi0 = _chi0_q0_direct(
        psi_k=psi_k_sum,
        enk_k=enk_k_sum,
        n_valence=cfg.n_valence,
        n_sum_states=cfg.n_sum_states,
        omega=0.0 + 0.0j,
    )
    chii = _chi0_q0_direct(
        psi_k=psi_k_sum,
        enk_k=enk_k_sum,
        n_valence=cfg.n_valence,
        n_sum_states=cfg.n_sum_states,
        omega=1j * float(cfg.omega_p_ry),
    )
    # Match w_isdf Dyson prefactor convention.
    pref = 2.0 / (np.sqrt(float(sym.nk_tot)) * float(max(wfn.nspin, 1)) * float(max(wfn.nspinor, 1)))
    w0_nohead = _solve_w_from_chi_direct(v_q0_no_g0, chi0, pref)
    wi_nohead = _solve_w_from_chi_direct(v_q0_no_g0, chii, pref)

    meta = Meta.from_system(
        wfn,
        sym,
        int(cfg.n_valence),
        int(cfg.n_cond_solve),
        int(cfg.n_sum_states),
        int(nrmu),
        bispinor=bool(int(wfn.nspinor) == 4),
    )
    meta.sys_dim = int(params.get("sys_dim", 0))

    # Resolve head values (overrides or automatic sources).
    v_head = cfg.v_head_au
    w0_head = cfg.w0_head_au
    wi_head = cfg.wiwp_head_au
    if v_head is None or w0_head is None:
        vc0_auto, wc0_auto, src0 = _determine_wcoul0(
            source=cfg.wcoul0_source,
            omega=0.0 + 0.0j,
            eta=cfg.wcoul0_eta,
            params=params,
            input_dir=gw_input_dir,
            wfn=wfn,
            sym=sym,
            meta=meta,
            print_fn=print,
        )
        if v_head is None:
            v_head = float(np.real(vc0_auto))
        if w0_head is None:
            w0_head = float(np.real(wc0_auto))
        print(f"  Head(omega=0) source: {src0}")
    if wi_head is None:
        _, wci_auto, srci = _determine_wcoul0(
            source=cfg.wcoul0_source,
            omega=1j * float(cfg.omega_p_ry),
            eta=cfg.wcoul0_eta,
            params=params,
            input_dir=gw_input_dir,
            wfn=wfn,
            sym=sym,
            meta=meta,
            print_fn=print,
        )
        wi_head = float(np.real(wci_auto))
        print(f"  Head(omega=i*omega_p) source: {srci}")

    outer_u = np.conj(g0)[:, None] * g0[None, :]
    vol_scale = 1.0 / float(wfn.cell_volume)
    v_headed = v_q0_no_g0 + (complex(v_head) * vol_scale) * outer_u
    w0_headed = w0_nohead + (complex(w0_head) * vol_scale) * outer_u
    wi_headed = wi_nohead + (complex(wi_head) * vol_scale) * outer_u

    print("")
    print("Direct W norms (Frobenius):")
    print(f"  ||W(0)||_F       = {np.linalg.norm(w0_headed, ord='fro'):.10e}")
    print(f"  ||W(i*omega_p)||_F = {np.linalg.norm(wi_headed, ord='fro'):.10e}")

    # GN-PPM fit from head-corrected W^c.
    #
    # THE WRAPPER THIS USED TO CALL IS GONE.  1fb160c (2026-07-08) deleted
    # ``extract_gn_ppm_parameters_from_Wc`` and the ``GodbyNeedsPPM``
    # dataclass it returned, on a zero-callers claim that grepped ``src/``
    # and not ``scripts/``.  Nothing physical went with them: the wrapper
    # only reshaped a 7D (nkx,nky,nkz,1,mu,1,mu) block down to (q,mu,nu),
    # forwarded to ``fit_gn_ppm_from_wc_pair``, and packed the four
    # returns into a dataclass.  So this calls the fit directly on the
    # flat-q form, and the 7D dressing is dropped rather than rebuilt.
    #
    # ``n_mu_logical`` is what 1fb160c made REQUIRED, and it is the point
    # of the whole commit: pad modes are born DEAD (Omega = 0, B = 0,
    # valid = False) so no consumer needs a census mask.  This driver's
    # restart tensors are unpadded and single-q, so the logical centroid
    # count is ``nrmu`` and the mask is all-true -- but passing it
    # explicitly is what keeps that true if the restart file ever carries
    # a padded mu extent.
    if float(cfg.omega_p_ry) <= 0.0:
        raise ValueError("omega_p must be > 0 for GN-PPM extraction.")
    wc0 = w0_headed - v_headed
    wci = wi_headed - v_headed
    omega_munu, b_munu, valid, unfulfilled = fit_gn_ppm_from_wc_pair(
        jnp.asarray(wc0[None], dtype=jnp.complex128),
        jnp.asarray(wci[None], dtype=jnp.complex128),
        1j * float(cfg.omega_p_ry),
        fallback_omega=float(params.get("ppm_fallback_omega", 2.0)),
        n_mu_logical=int(nrmu),
    )
    b_munu = np.asarray(b_munu[0], dtype=np.complex128)
    omega_munu = np.asarray(omega_munu[0], dtype=np.float64)
    valid = np.asarray(valid[0], dtype=bool)
    print(
        f"  GN-PPM fit: omega_p={cfg.omega_p_ry:.6f} Ry, "
        f"unfulfilled={100.0 * float(unfulfilled):.2f}%"
    )

    # Direct diagonal sigma terms for solve bands.
    sigma_x = _sigma_x_diag(
        psi_k=psi_k_solve,
        v_head=v_headed,
        n_valence=cfg.n_valence,
        n_solve=nsolve,
    )
    sigma_p, sigma_m = _sigma_c_branches_diag(
        psi_k=psi_k_sum,
        enk_k=enk_k_sum,
        n_valence=cfg.n_valence,
        n_solve=nsolve,
        efermi=efermi,
        b_munu=b_munu,
        omega_munu=omega_munu,
        valid_mask=valid,
    )
    invalid_mask = (~valid) & (np.abs(wc0) > 1.0e-14)
    sigma_invalid_static = None
    if np.any(invalid_mask):
        wc0_invalid = np.where(invalid_mask, wc0, 0.0 + 0.0j)
        # Occupied-only static term (SX-like piece).
        sigma_occ_invalid = _sigma_x_diag(
            psi_k=psi_k_solve,
            v_head=wc0_invalid,
            n_valence=cfg.n_valence,
            n_solve=nsolve,
        )
        # Resolution-of-identity static term over the chosen sum-state window.
        sigma_ri_invalid = _sigma_x_diag(
            psi_k=psi_k_sum,
            v_head=wc0_invalid,
            n_valence=cfg.n_sum_states,
            n_solve=nsolve,
        )
        sigma_invalid_static = sigma_occ_invalid - 0.5 * sigma_ri_invalid
        print(
            f"  GN invalid static correction active: invalid={100.0 * float(np.mean(invalid_mask)):.2f}% "
            f"(of mu,nu); max|Sigma_invalid|={float(np.max(np.abs(sigma_invalid_static))):.6e} Ry"
        )
    else:
        print("  GN invalid static correction: no invalid entries in fitted mask.")

    print("")
    print("Band partition:")
    print(
        f"  offset={b0}, n_valence={cfg.n_valence}, n_cond_solve={cfg.n_cond_solve}, "
        f"n_sum_states={cfg.n_sum_states}, nsolve={nsolve}"
    )
    print(
        f"  E_F(midgap)={efermi * RYD2EV:.6f} eV  "
        f"(VBM={e_vbm * RYD2EV:.6f} eV, CBM={e_cbm * RYD2EV:.6f} eV)"
    )
    _print_sigma_table(
        enk_solve=enk_k_solve,
        efermi=efermi,
        sigma_x=sigma_x,
        sigma_p=sigma_p,
        sigma_m=sigma_m,
        sigma_invalid_static=sigma_invalid_static,
        band_offset=b0,
    )

    print("")
    print("Head values used (a.u.):")
    print(f"  v_head={float(v_head):.10f}")
    print(f"  w_head(0)={float(w0_head):.10f}")
    print(f"  w_head(i*omega_p)={float(wi_head):.10f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direct q=Gamma check of W(0), W(i*omega_p), and Sigma branches from restart tensors."
    )
    parser.add_argument("-i", "--input", required=True, help="Direct-check input file")
    args = parser.parse_args()
    return run(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
