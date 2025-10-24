from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .build_projectors_qe import (
    build_E_blocks_full,
    precompute_projector_splines,
    qe_real_sph_harmonics,
    qe_real_sph_harmonics_with_grad,
    compute_type_projectors_real,
)
from scipy.special import spherical_jn


@dataclass
class Species:
    pseudo: object
    atom_indices: List[int]
    atom_tau: List[np.ndarray]


@dataclass
class PseudoBundle:
    E_blocks: Dict[int, np.ndarray]
    F_splines: Dict[Tuple[int, int], object]
    lmax: int


@dataclass
class KBundle:
    K_cart: np.ndarray
    K_norm: np.ndarray


def group_assignments_by_pseudo(assignments) -> List[Species]:
    by_key: Dict[int, Dict[str, object]] = {}
    for ap in assignments:
        pseudo = ap.pseudo
        if pseudo is None:
            continue
        key = id(pseudo)
        entry = by_key.setdefault(key, {"pseudo": pseudo, "atom_indices": [], "atom_tau": []})
        entry["atom_indices"].append(int(ap.index))
        entry["atom_tau"].append(np.asarray(ap.position, dtype=float))
    return [Species(pseudo=v["pseudo"], atom_indices=v["atom_indices"], atom_tau=v["atom_tau"]) for v in by_key.values()]


def _infer_lmax_from_pseudo(pseudo) -> int:
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    betas = list(getattr(ppnl, 'pp_beta', [])) if ppnl is not None else []
    if not betas:
        return 0
    return max(int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0))) for beta in betas)


def prepare_pseudo_bundle(pseudo, cell_volume: float, q_max: float) -> PseudoBundle:
    E_blocks = build_E_blocks_full(pseudo)
    F_splines = precompute_projector_splines(pseudo, float(cell_volume), float(q_max))
    lmax = _infer_lmax_from_pseudo(pseudo)
    return PseudoBundle(E_blocks=E_blocks, F_splines=F_splines, lmax=int(lmax))


def prepare_pseudo_cache(species_list: List[Species], cell_volume: float, q_max: float) -> Dict[int, PseudoBundle]:
    cache: Dict[int, PseudoBundle] = {}
    for sp in species_list:
        key = id(sp.pseudo)
        if key not in cache:
            cache[key] = prepare_pseudo_bundle(sp.pseudo, cell_volume, q_max)
    return cache


def prepare_k_bundle(K_cart: np.ndarray) -> KBundle:
    K_cart = np.asarray(K_cart, dtype=float)
    K_norm = np.sqrt(np.sum(K_cart ** 2, axis=1)) if K_cart.size else np.zeros((0,), dtype=float)
    return KBundle(K_cart=K_cart, K_norm=K_norm)


def make_projector_rows(bundle: PseudoBundle, kpack: KBundle, species: Species, K_crys: np.ndarray, cell_volume: float):
    Y_real_by_l = {l: qe_real_sph_harmonics(l, np.asarray(kpack.K_cart)) for l in range(bundle.lmax + 1)}
    rows, meta = compute_type_projectors_real(
        pseudo=species.pseudo,
        atom_indices=species.atom_indices,
        atom_tau=species.atom_tau,
        K_crys_total=np.asarray(K_crys),
        K_cart_total=np.asarray(kpack.K_cart),
        cell_volume=float(cell_volume),
        Y_real_by_l_pre=Y_real_by_l,
        F_splines=bundle.F_splines,
        K_norm_in=np.asarray(kpack.K_norm),
    )
    return rows, meta


def _evaluate_Fprime_bessel(pseudo, l: int, beta_ids: list[int], K_norm: np.ndarray) -> np.ndarray:
    """Compute F'_l(q) via the j'_l kernel: F'_l(q) = ∫ dr r^3 β_l(r) j'_l(q r).

    Returns array of shape (nbeta, nK).
    """
    ppmesh = getattr(pseudo, 'pp_mesh', None)
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppmesh is None or ppnl is None or len(beta_ids) == 0:
        return np.zeros((len(beta_ids), len(K_norm)))
    r = np.asarray(ppmesh.pp_r.value, dtype=float)
    try:
        rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
    except Exception:
        rab = None
    # Collect beta objects in 1-based file order
    betas_all = list(getattr(ppnl, 'pp_beta', []))
    out = []
    nK = int(len(K_norm))
    for bid in beta_ids:
        beta = betas_all[int(bid) - 1]
        raw = np.asarray(beta.value, dtype=float)
        if r[0] == 0.0 and len(r) > 1:
            beta_r = np.empty_like(r)
            beta_r[1:] = raw[1:] / r[1:]
            beta_r[0] = beta_r[1]
        else:
            beta_r = raw / r
        if nK == 0:
            out.append(np.zeros((0,), dtype=float))
            continue
        qr = np.multiply.outer(K_norm, r)  # (nK, nr)
        jn_d = spherical_jn(l, qr, derivative=True)  # (nK, nr)
        integrand = jn_d * (r[None, :] ** 3) * beta_r[None, :]
        if rab is not None:
            vals = np.sum(integrand * rab[None, :], axis=1)
        else:
            # trapezoid
            dr = r[1] - r[0] if len(r) > 1 else 1.0
            w = np.ones_like(r)
            if len(r) > 0:
                w[0] = 0.5; w[-1] = 0.5
            vals = np.sum(integrand * w[None, :], axis=1) * dr
        out.append(vals)
    return np.stack(out, axis=0)


def _evaluate_F_and_Fprime_for_l(
    splines: Dict[Tuple[int, int], object],
    d_spl_cache: Dict[Tuple[int, int], object],
    l: int,
    beta_ids: list[int],
    K_norm: np.ndarray,
    pseudo=None,
    fprime_mode: str = 'spline',
) -> tuple[np.ndarray, np.ndarray]:
    if K_norm.size == 0:
        return np.zeros((len(beta_ids), 0)), np.zeros((len(beta_ids), 0))
    F_list = []
    if fprime_mode == 'bessel' and (pseudo is not None):
        Fp_mat = _evaluate_Fprime_bessel(pseudo, int(l), beta_ids, np.asarray(K_norm, dtype=float))
    else:
        Fp_mat = None
    Fp_list = []
    for bid_idx, bid in enumerate(beta_ids):
        spl = splines.get((int(l), int(bid)))
        if spl is None:
            F_list.append(np.zeros_like(K_norm))
            Fp_list.append(np.zeros_like(K_norm))
            continue
        Fv = spl(K_norm)
        if Fp_mat is not None:
            Fpv = Fp_mat[bid_idx, :]
        else:
            dspl = d_spl_cache.get((int(l), int(bid)))
            if dspl is None:
                dspl = spl.derivative(1)
                d_spl_cache[(int(l), int(bid))] = dspl
            Fpv = dspl(K_norm)
        F_list.append(Fv)
        Fp_list.append(Fpv)
    return np.stack(F_list, axis=0), np.stack(Fp_list, axis=0)


def build_beta_rows_with_grad(
    wfn_k: jax.Array,
    Gk_crys: np.ndarray,
    K_crys: np.ndarray,
    K_cart: np.ndarray,
    plan: Dict[int, Dict[str, object]],
    cell_volume: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (Z_rows, gradZ_rows, meta) for all species/atoms and ℓ.

    Shapes:
      - Z_rows: (total_R, nG)
      - gradZ_rows: (3, total_R, nG)
    """
    K_cart_np = np.asarray(K_cart, dtype=float)
    K_crys_np = np.asarray(K_crys, dtype=float)
    K_norm = np.sqrt(np.sum(K_cart_np ** 2, axis=1)) if K_cart_np.size else np.zeros((0,), dtype=float)

    Z_rows: list[np.ndarray] = []
    dZ_rows_xyz: list[np.ndarray] = []
    dZ_rows_rad: list[np.ndarray] = []
    dZ_rows_ang: list[np.ndarray] = []
    meta_all: list[dict] = []

    d_spl_cache: Dict[Tuple[int, int], object] = {}

    for key, sp in plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        pref = float(sp['prefactor'])
        splines = sp['splines']

        for l, info in sp['l_channels'].items():
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue
            msize = 2 * int(l) + 1

            F_bG, Fp_bG = _evaluate_F_and_Fprime_for_l(splines, d_spl_cache, int(l), beta_ids, K_norm, pseudo=sp['pseudo'], fprime_mode=sp.get('fprime_mode', 'spline'))
            radial = pref * (1j) ** int(l) * F_bG  # (nbeta, nG), A(q) = F(q)
            radial_p = pref * (1j) ** int(l) * Fp_bG  # (nbeta, nG), A'(q) = F'(q)

            Y_real, gradY_cart = qe_real_sph_harmonics_with_grad(int(l), K_cart_np)  # (msize,nG), (3,msize,nG)

            # e_r (3,nG)
            q = K_norm
            q_safe = np.where(q > 0, q, 1e-12)
            er = (K_cart_np / q_safe[:, None]).T

            if tau.ndim == 1:
                tau = tau.reshape(1, 3)
            phase = np.exp(-2j * np.pi * (K_crys_np @ tau.T)).T  # (natoms, nG)

            nbeta = len(beta_ids)
            R = nbeta * msize

            # Build β and ∇β per atom
            for a in range(phase.shape[0]):
                ph = phase[a, :][None, None, :]  # (1,1,nG)
                Z_bmg = radial[:, None, :] * Y_real[None, :, :] * ph  # (nbeta,msize,nG)
                Z_block = Z_bmg.reshape((R, K_cart_np.shape[0]))

                # grad_radial for β(K)=F_l(q)Y: F'_l(q) Y e_r
                base_rad = (radial_p[:, None, :] * Y_real[None, :, :])  # (nbeta,msize,nG)
                grad_rad = er[:, None, None, :] * base_rad[None, :, :, :]  # (3,nbeta,msize,nG)

                # grad_angular: (3,nbeta,msize,nG) = (radial / q) * gradY_cart
                base_ang = (radial[:, None, :] / q_safe[None, None, :])  # (nbeta,1,nG)
                grad_ang = base_ang[None, :, :, :] * gradY_cart[:, None, :, :]  # (3,nbeta,msize,nG)

                # Combine, apply phase already included in Z_bmg (phase derivative cancels in ∇q+∇q′)
                grad_total = (grad_rad + grad_ang) * ph  # (3,nbeta,msize,nG)
                grad_block = grad_total.reshape((3, R, K_cart_np.shape[0]))
                grad_block_rad = (grad_rad * ph).reshape((3, R, K_cart_np.shape[0]))
                grad_block_ang = (grad_ang * ph).reshape((3, R, K_cart_np.shape[0]))

                Z_rows.append(Z_block)
                dZ_rows_xyz.append(grad_block)
                dZ_rows_rad.append(grad_block_rad)
                dZ_rows_ang.append(grad_block_ang)
                meta_all.append({'pseudo_key': key, 'atom_index': a, 'l': int(l), 'beta_ids': list(beta_ids)})

    if Z_rows:
        Z_cat = np.vstack(Z_rows).astype(np.complex128, copy=False)
        dZ_cat = np.concatenate(dZ_rows_xyz, axis=1).astype(np.complex128, copy=False)
        dZ_rad = np.concatenate(dZ_rows_rad, axis=1).astype(np.complex128, copy=False)
        dZ_ang = np.concatenate(dZ_rows_ang, axis=1).astype(np.complex128, copy=False)
    else:
        nG = K_cart_np.shape[0]
        Z_cat = np.zeros((0, nG), dtype=np.complex128)
        dZ_cat = np.zeros((3, 0, nG), dtype=np.complex128)
        dZ_rad = np.zeros_like(dZ_cat)
        dZ_ang = np.zeros_like(dZ_cat)
    # Attach components for optional consumers via attributes
    try:
        Z_cat._dZ_rad = dZ_rad  # type: ignore[attr-defined]
        Z_cat._dZ_ang = dZ_ang  # type: ignore[attr-defined]
    except Exception:
        pass
    return Z_cat, dZ_cat, meta_all


def build_beta_rows_with_grad_components(
    wfn_k: jax.Array,
    Gk_crys: np.ndarray,
    K_crys: np.ndarray,
    K_cart: np.ndarray,
    plan: Dict[int, Dict[str, object]],
    cell_volume: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Like build_beta_rows_with_grad, but also returns radial and angular parts.

    Returns (Z_rows, dZ_total, dZ_radial, dZ_angular, meta).
    Shapes:
      - Z_rows: (R_total, nG)
      - dZ_*: (3, R_total, nG)
    """
    K_cart_np = np.asarray(K_cart, dtype=float)
    K_crys_np = np.asarray(K_crys, dtype=float)
    K_norm = np.sqrt(np.sum(K_cart_np ** 2, axis=1)) if K_cart_np.size else np.zeros((0,), dtype=float)

    Z_rows: list[np.ndarray] = []
    dZ_rows_xyz: list[np.ndarray] = []
    dZ_rows_rad: list[np.ndarray] = []
    dZ_rows_ang: list[np.ndarray] = []
    meta_all: list[dict] = []

    d_spl_cache: Dict[Tuple[int, int], object] = {}

    for key, sp in plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        pref = float(sp['prefactor'])
        splines = sp['splines']

        for l, info in sp['l_channels'].items():
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue
            msize = 2 * int(l) + 1

            F_bG, Fp_bG = _evaluate_F_and_Fprime_for_l(splines, d_spl_cache, int(l), beta_ids, K_norm)
            radial = pref * (1j) ** int(l) * F_bG  # (nbeta, nG)
            radial_p = pref * (1j) ** int(l) * Fp_bG  # (nbeta, nG)

            Y_real, gradY_cart = qe_real_sph_harmonics_with_grad(int(l), K_cart_np)  # (msize,nG), (3,msize,nG)

            # e_r (3,nG)
            q = K_norm
            q_safe = np.where(q > 0, q, 1.0)
            er = (K_cart_np / q_safe[:, None]).T

            if tau.ndim == 1:
                tau = tau.reshape(1, 3)
            phase = np.exp(-2j * np.pi * (K_crys_np @ tau.T)).T  # (natoms, nG)

            nbeta = len(beta_ids)
            R = nbeta * msize

            for a in range(phase.shape[0]):
                ph = phase[a, :][None, None, :]  # (1,1,nG)
                Z_bmg = radial[:, None, :] * Y_real[None, :, :] * ph  # (nbeta,msize,nG)
                Z_block = Z_bmg.reshape((R, K_cart_np.shape[0]))

                base_rad = ((radial_p[:, None, :] + (int(l) * radial[:, None, :] / q_safe[None, None, :])) * Y_real[None, :, :])  # (nbeta,msize,nG)
                grad_rad = er[:, None, None, :] * base_rad[None, :, :, :]  # (3,nbeta,msize,nG)

                base_ang = (radial[:, None, :] / q_safe[None, None, :])  # (nbeta,1,nG)
                grad_ang = base_ang[None, :, :, :] * gradY_cart[:, None, :, :]  # (3,nbeta,msize,nG)

                grad_total = (grad_rad + grad_ang) * ph  # (3,nbeta,msize,nG)
                dZ_block = grad_total.reshape((3, R, K_cart_np.shape[0]))
                dZ_block_rad = (grad_rad * ph).reshape((3, R, K_cart_np.shape[0]))
                dZ_block_ang = (grad_ang * ph).reshape((3, R, K_cart_np.shape[0]))

                Z_rows.append(Z_block)
                dZ_rows_xyz.append(dZ_block)
                dZ_rows_rad.append(dZ_block_rad)
                dZ_rows_ang.append(dZ_block_ang)
                meta_all.append({'pseudo_key': key, 'atom_index': a, 'l': int(l), 'beta_ids': list(beta_ids)})

    if Z_rows:
        Z_cat = np.vstack(Z_rows).astype(np.complex128, copy=False)
        dZ_cat = np.concatenate(dZ_rows_xyz, axis=1).astype(np.complex128, copy=False)
        dZ_rad = np.concatenate(dZ_rows_rad, axis=1).astype(np.complex128, copy=False)
        dZ_ang = np.concatenate(dZ_rows_ang, axis=1).astype(np.complex128, copy=False)
    else:
        nG = K_cart_np.shape[0]
        Z_cat = np.zeros((0, nG), dtype=np.complex128)
        dZ_cat = np.zeros((3, 0, nG), dtype=np.complex128)
        dZ_rad = np.zeros_like(dZ_cat)
        dZ_ang = np.zeros_like(dZ_cat)

    return Z_cat, dZ_cat, dZ_rad, dZ_ang, meta_all


def compute_V_NL_velocity_k(
    wfn_k: jax.Array,
    Gk_crys: np.ndarray,
    K_crys: np.ndarray,
    K_cart: np.ndarray,
    plan: Dict[int, Dict[str, object]],
    cell_volume: float,
    fprime_mode: str = 'bessel',
) -> jax.Array:
    """Compute i [r_i, V_NL] matrix components for i=x,y,z.

    Returns array of shape (3, nb, nb). Uses β and ∇β assembled by
    build_beta_rows_with_grad, contracting against E blocks per ℓ/species.
    """
    nb, nspinor = int(wfn_k.shape[0]), int(wfn_k.shape[1])
    if wfn_k.ndim == 3:
        # Already (nb, nspinor, nG)
        C_bsg = jnp.asarray(wfn_k)
    else:
        Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
        Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
        Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
        C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)

    # Ensure plan entries carry fprime_mode for derivative evaluation
    for key in plan.keys():
        plan[key]['fprime_mode'] = fprime_mode
    Z_rows, dZ_rows, meta = build_beta_rows_with_grad(wfn_k, Gk_crys, K_crys, K_cart, plan, cell_volume)
    nG = C_bsg.shape[-1]
    if Z_rows.shape[-1] != nG:
        raise RuntimeError("Projector rows nG mismatch")

    vNL = jnp.zeros((3, nb, nb), dtype=jnp.complex128)

    offset = 0
    for entry in meta:
        key = entry['pseudo_key']
        l = int(entry['l'])
        beta_ids = list(entry['beta_ids'])
        msize = 2 * l + 1
        R = len(beta_ids) * msize
        Z_block = Z_rows[offset:offset + R, :]  # (R, nG)
        dZ_block = dZ_rows[:, offset:offset + R, :]  # (3, R, nG)
        offset += R

        E_l_np = plan[key]['l_channels'].get(l, {}).get('E')
        if E_l_np is None:
            continue
        E_l = jnp.asarray(E_l_np, dtype=jnp.complex128)  # (2,2,Rl,Rl)

        # P(r,s,j) = Σ_g Z*(r,g) C(j,s,g); dP_i(r,s,j) = Σ_g (dZ_i)*(r,g) C(j,s,g)
        P = jnp.einsum('rg,bsg->rsb', jnp.conj(Z_block), C_bsg, optimize=True)       # (R,2,nb)
        dP = jnp.einsum('irg,bsg->irsb', jnp.conj(dZ_block), C_bsg, optimize=True)   # (3,R,2,nb)

        # In Ry atomic units (ħ=1, m=1/2): v^NL_i = (i/ħ)[V_NL, r_i] = - (∇_{q_i} + ∇_{q'_i}) V_NL
        # term1 + term2 evaluates (∇_q + ∇_{q'}) V_NL contracted with states; apply a minus sign, no extra i.
        term1 = jnp.einsum('irsn, strq, qtm -> inm', jnp.conj(dP), E_l, P, optimize=True)
        term2 = jnp.einsum('rsn, strq, iqtm -> inm', jnp.conj(P), E_l, dP, optimize=True)
        vNL = vNL - (term1 + term2)

    return vNL

__all__ = [
    "Species",
    "PseudoBundle",
    "KBundle",
    "group_assignments_by_pseudo",
    "prepare_pseudo_bundle",
    "prepare_pseudo_cache",
    "prepare_k_bundle",
    "make_projector_rows",
    "build_vnl_plan",
    "compute_V_NL_k_minimal",
    "build_beta_rows_with_grad",
    "compute_V_NL_velocity_k",
    "compute_V_NL_velocity_k_numeric",
]


def build_vnl_plan(pseudos: Dict[str, object], assignments, cell_volume: float, q_max: float):
    """Return a minimal per-species, per-ℓ plan with E blocks and radial splines.

    Plan structure:
      plan = {
        id(pseudo): {
          'pseudo': pseudo,
          'atoms': { 'indices': [..], 'tau': np.ndarray(natoms,3) },
          'prefactor': 4π/√Ω,
          'l_channels': {
              ℓ: {
                  'E': np.ndarray((2,2,Rℓ,Rℓ)),
                  'beta_ids': [idx1, idx2, ...],   # 1-based per UPF order
              }
          },
          'splines': { (ℓ, idx) -> spline },
        }
      }
    """
    # Group atoms by pseudo
    species: Dict[int, Dict[str, object]] = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        ent = species.setdefault(key, {
            'pseudo': ap.pseudo,
            'indices': [],
            'tau': [],
        })
        ent['indices'].append(int(ap.index))
        ent['tau'].append(np.asarray(ap.position, dtype=float))

    plan: Dict[int, Dict[str, object]] = {}
    prefactor = (4.0 * np.pi) / float(np.sqrt(cell_volume))
    for key, ent in species.items():
        pseudo = ent['pseudo']
        E_blocks = build_E_blocks_full(pseudo)  # dict ℓ -> (2,2,Rℓ,Rℓ)
        splines = precompute_projector_splines(pseudo, float(cell_volume), float(q_max))
        # Map betas per ℓ in file order
        l_channels: Dict[int, Dict[str, object]] = {}
        betas = list(getattr(getattr(pseudo, 'pp_nonlocal', None), 'pp_beta', []))
        per_l: Dict[int, List[int]] = {}
        for idx, beta in enumerate(betas, start=1):
            l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
            per_l.setdefault(l, []).append(idx)
        for l, beta_ids in per_l.items():
            l_channels[int(l)] = {
                'E': E_blocks.get(int(l)),
                'beta_ids': list(beta_ids),
            }
        plan[key] = {
            'pseudo': pseudo,
            'atoms': {
                'indices': np.asarray(ent['indices'], dtype=int),
                'tau': np.asarray(ent['tau'], dtype=float) if ent['tau'] else np.zeros((0, 3), dtype=float),
            },
            'prefactor': prefactor,
            'l_channels': l_channels,
            'splines': splines,
        }
    return plan


def compute_V_NL_k_minimal(
    wfn_k: jax.Array,
    Gk_crys: np.ndarray,
    K_crys: np.ndarray,
    K_cart: np.ndarray,
    plan: Dict[int, Dict[str, object]],
    cell_volume: float,
) -> jax.Array:
    """Vectorized minimal V_NL(k) using prebuilt plan.

    Steps per ℓ:
      - Evaluate F_ℓ,β(|K|) via splines and form radial = (4π/√Ω)(i)^ℓ F.
      - Build Y'_{ℓm}(K) once per ℓ; multiply radial and structure phases per atom.
      - Stack rows in beta-major, m-slot order to match E_ℓ layout.
      - Contract: P = Z^†·C, then Δ_ℓ = Σσσ' P^†_σ E^{σσ'}_ℓ P_{σ'}.
    """
    nb, nspinor = int(wfn_k.shape[0]), int(wfn_k.shape[1])
    if getattr(wfn_k, 'ndim', 0) == 3:
        C_bsg = jnp.asarray(wfn_k)
        nG = C_bsg.shape[-1]
    else:
        Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
        Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
        Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
        C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)
        nG = C_bsg.shape[-1]
    nG = C_bsg.shape[-1]

    K_cart_np = np.asarray(K_cart, dtype=float)
    K_crys_np = np.asarray(K_crys, dtype=float)
    K_norm = np.sqrt(np.sum(K_cart_np ** 2, axis=1)) if K_cart_np.size else np.zeros((0,), dtype=float)
    K_norm_j = jnp.asarray(K_norm, dtype=jnp.float64)

    V_NL_k = jnp.zeros((nb, nb), dtype=jnp.complex128)

    for key, sp in plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        indices = np.asarray(sp['atoms']['indices'], dtype=int)
        pref = float(sp['prefactor'])
        splines = sp['splines']

        for l, info in sp['l_channels'].items():
            E_l_np = info['E']
            if E_l_np is None:
                continue
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue
            msize = 2 * int(l) + 1

            # F_ℓ,β(|K|) for all beta in this ℓ
            F_bG = np.stack([splines[(int(l), int(bid))](K_norm) for bid in beta_ids], axis=0) if nG else np.zeros((len(beta_ids), 0))
            radial = pref * (1j) ** int(l) * F_bG  # (nbeta, nG)
            radial_j = jnp.asarray(radial, dtype=jnp.complex128)

            # Y'_{ℓm}(K) and phases per atom
            Y_real = qe_real_sph_harmonics(int(l), K_cart_np)  # (msize, nG) np
            Y_j = jnp.asarray(Y_real, dtype=jnp.complex128)

            # Structure phases per atom: exp(-2πi K_crys·τ)
            if tau.ndim == 1:
                tau = tau.reshape(1, 3)
            phase = np.exp(-2j * np.pi * (K_crys_np @ tau.T))  # (nG, natoms)
            phase_j = jnp.asarray(phase.T, dtype=jnp.complex128)  # (natoms, nG)

            E_l = jnp.asarray(E_l_np, dtype=jnp.complex128)  # (2,2,R,R)
            R = len(beta_ids) * msize

            # For each atom, build Z_block and contract in a fully vectorized way
            for a in range(phase_j.shape[0]):
                Z_bmg = radial_j[:, None, :] * Y_j[None, :, :] * phase_j[a, None, None, :]
                Z_block = Z_bmg.reshape((R, nG))  # (R, nG)

                # ############ VNL MATRIX ELEMENT CONSTRUCTION ############
                # P(r,s,j) = Σ_g Z*(r,g) C(j,s,g)
                P = jnp.einsum('rg,bsg->rsb', jnp.conj(Z_block), C_bsg, optimize=True)
                # Δ_ℓ(i,j) = Σ_{s,t,r,q} conj(P(r,s,i)) E_l(s,t,r,q) P(q,t,j)
                Delta_l = jnp.einsum('rsi, strq, qtj -> ij', jnp.conj(P), E_l, P, optimize=True)
                V_NL_k = V_NL_k + Delta_l

    return V_NL_k


def compute_V_NL_velocity_k_numeric(
    wfn_k: jax.Array,
    Gk_crys: np.ndarray,
    K_crys: np.ndarray,
    K_cart: np.ndarray,
    plan: Dict[int, Dict[str, object]],
    cell_volume: float,
    h: float = 1e-5,
) -> jax.Array:
    """Finite-difference nonlocal velocity v^NL via Z(K±δ) (no state derivatives).

    For each Cartesian axis i, forms K± = K ± δ e_i (in Cartesian), recomputes Z,
    and evaluates v_i = - [ conj(dZ_i) E Z + conj(Z) E dZ_i ] contracted with C.

    Returns: (3, nb, nb)
    """
    import numpy as _np

    nb, nspinor = int(wfn_k.shape[0]), int(wfn_k.shape[1])
    if getattr(wfn_k, 'ndim', 0) == 3:
        C_bsg = jnp.asarray(wfn_k)
        nG = int(C_bsg.shape[-1])
    else:
        Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
        Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
        Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
        C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)
        nG = int(C_bsg.shape[-1])

    # Base Z for reference and E blocks iterator
    Z0, _, meta = build_beta_rows_with_grad(wfn_k, Gk_crys, K_crys, K_cart, plan, cell_volume)
    if int(Z0.shape[-1]) != nG:
        raise RuntimeError("Projector rows nG mismatch (numeric vel)")

    # Recover B and its inverse to map δ_cart -> δ_crys
    K_crys_np = _np.asarray(K_crys, dtype=float)
    K_cart_np = _np.asarray(K_cart, dtype=float)
    B, *_ = _np.linalg.lstsq(K_crys_np, K_cart_np, rcond=None)
    Binv = _np.linalg.inv(B)

    vNL = jnp.zeros((3, nb, nb), dtype=jnp.complex128)

    for ic in range(3):
        d_cart = _np.zeros((3,), dtype=float); d_cart[ic] = float(h)
        d_crys = d_cart @ Binv
        Kp_crys = K_crys_np + d_crys[None, :]
        Km_crys = K_crys_np - d_crys[None, :]
        Kp_cart = Kp_crys @ B
        Km_cart = Km_crys @ B

        Zp, _, _ = build_beta_rows_with_grad(wfn_k, Gk_crys, Kp_crys, Kp_cart, plan, cell_volume)
        Zm, _, _ = build_beta_rows_with_grad(wfn_k, Gk_crys, Km_crys, Km_cart, plan, cell_volume)
        dZ = (Zp - Zm) / (2.0 * float(h))  # (R, nG)

        offset = 0
        acc = jnp.zeros((nb, nb), dtype=jnp.complex128)
        for entry in meta:
            key = entry['pseudo_key']
            l = int(entry['l'])
            beta_ids = list(entry['beta_ids'])
            msize = 2 * l + 1
            R = len(beta_ids) * msize
            Z_block = Z0[offset:offset + R, :]  # (R, nG)
            dZ_block = dZ[offset:offset + R, :]  # (R, nG)
            offset += R
            E_l_np = plan[key]['l_channels'].get(l, {}).get('E')
            if E_l_np is None:
                continue
            E_l = jnp.asarray(E_l_np, dtype=jnp.complex128)  # (2,2,R,R)

            P = jnp.einsum('rg,bsg->rsb', jnp.conj(Z_block), C_bsg, optimize=True)       # (R,2,nb)
            dP = jnp.einsum('rg,bsg->rsb', jnp.conj(dZ_block), C_bsg, optimize=True)     # (R,2,nb)
            term1 = jnp.einsum('rsn, strq, qtm -> nm', jnp.conj(dP), E_l, P, optimize=True)
            term2 = jnp.einsum('rsn, strq, qtm -> nm', jnp.conj(P), E_l, dP, optimize=True)
            acc = acc - (term1 + term2)

        vNL = vNL.at[ic].set(acc)

    return vNL
