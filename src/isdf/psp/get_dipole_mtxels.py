#!/usr/bin/env python3
"""
Dipole/velocity matrix elements calculation: <mk|v|nk> and related pieces.

This module mirrors the setup of get_DFT_mtxels.py and provides a scaffold to
compute:
- Momentum operator matrix elements p_i: sum_G (k+G)_i c*_mk(G) c_nk(G)
- Nonlocal pseudopotential contribution to velocity (initialized projectors)

The nonlocal velocity term is stubbed pending detailed implementation; the code
initializes QE-style projectors so downstream development can fill it in.
"""

import os
import argparse
from pathlib import Path

# Set JAX configs BEFORE importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import jax
# Force CPU backend to avoid GPU plugin errors in test envs
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Support both `python -m isdf.psp.get_dipole_mtxels` and direct script execution
try:
	from ..common.wfnreader import WFNReader
	from ..common import symmetry_maps
	from ..common.load_wfns import read_Gvecs_to_devices
	from ..common import Meta
	from .get_DFT_mtxels import (
		read_cohsex_input,
		load_pseudopotentials,
		build_atom_pp_assignments,
		generate_gvectors_k,
		print_atomic_structure,
	)
except ImportError:
	# Fallback for direct script execution: add project `src` to sys.path and use absolute imports
	import sys as _sys
	from pathlib import Path as _Path
	_sys.path.append(str(_Path(__file__).resolve().parents[2]))  # .../src
	from isdf.common.wfnreader import WFNReader
	from isdf.common import symmetry_maps
	from isdf.common.load_wfns import read_Gvecs_to_devices
	from isdf.common import Meta
	from isdf.psp.get_DFT_mtxels import (
		read_cohsex_input,
		load_pseudopotentials,
		build_atom_pp_assignments,
		generate_gvectors_k,
		print_atomic_structure,
	)

from isdf.psp.projector_pipeline import (
    build_vnl_plan,
    compute_V_NL_k_minimal,
    compute_V_NL_velocity_k,
    compute_V_NL_velocity_k_numeric,
)
from dataclasses import dataclass
import h5py




# --------------------------
# K+G helpers
# --------------------------


def build_K_vectors(Gk_crys, kpoint_crys, wfn):
	k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
	B = jnp.asarray(wfn.bvec, dtype=jnp.float64).T * float(wfn.blat)
	K_cart = jnp.asarray(K_crys) @ B
	return K_crys, K_cart


# --------------------------
# Velocity/dipole pieces and projector pipeline
# --------------------------


from dataclasses import dataclass

def compute_p_operator_k(wfn_k: jax.Array, Gk_crys: np.ndarray, kpoint_crys: np.ndarray, bdot: np.ndarray, bvec: np.ndarray, blat: float) -> jax.Array:
	"""Compute p-operator matrix elements per Cartesian component.

	Returns array of shape (3, nb, nb) for components x,y,z.
	p_i = sum_G (k+G)_cart[i] c*_mk(G) c_nk(G)
	"""
	nb, nspinor = int(wfn_k.shape[0]), int(wfn_k.shape[1])
	Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
	Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
	Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
	C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)
	k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
	B = jnp.asarray(bvec, dtype=jnp.float64).T * float(blat)
	K_cart = jnp.asarray(K_crys) @ B  # (nG, 3)
	p_mn = []
	for i in range(3):
		K_i = K_cart[:, i]
		weighted = C_bsg * K_i[None, None, :]
		p_i = jnp.einsum('msg,nsg->mn', jnp.conj(C_bsg), weighted, optimize=True)
		p_mn.append(p_i)
	return jnp.stack(p_mn, axis=0)


def compute_projected_momentum_bgw_like(
	wfn_k: jax.Array,
	Gk_crys: np.ndarray,
	kpoint_crys: np.ndarray,
	bdot: np.ndarray,
	use_k_term: bool = False,
) -> jax.Array:
	"""Return momentum matrix projected along crystal x-hat with BGW metric.

	Computes 2 * (pol^T bdot (G [+ k])) / |pol| with pol_crys = (1,0,0), and
	sums over G and spinor: sum_G,s conj(C_ic,s,G) C_iv,s,G weight_G.
	"""
	pol = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
	bd = jnp.asarray(bdot, dtype=jnp.float64)
	G = jnp.asarray(Gk_crys, dtype=jnp.float64)
	k = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	vec = G + (k[None, :] if use_k_term else 0.0)
	weights = 2.0 * (pol @ (bd @ vec.T))  # (nG,)
	lpol = jnp.sqrt(pol @ (bd @ pol))
	weights = weights / jnp.maximum(lpol, 1e-16)
	weights = jnp.asarray(weights, dtype=wfn_k.dtype)
	Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
	Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
	Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
	C_bsg = wfn_k[:, :, Gx, Gy, Gz]  # (nb, nspinor, nG)
	weighted = C_bsg * weights[None, None, :]
	return jnp.einsum('msg,nsg->mn', jnp.conj(C_bsg), weighted, optimize=True)


def compute_block_direct_cnk(sym, wfn, k_index: int, c_idx: np.ndarray, v_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4x4 blocks using direct WFN access; returns (p_x/|pol|/ΔE, full/ΔE).

    Projection uses crystal x-hat with BGW metric; sums over spinor and G.
    """
    # G list aligned with coefficients ordering
    G_list = sym.get_gvecs_kfull(wfn, k_index)  # (nG,3) ints
    G_crys = np.asarray(G_list, dtype=float)
    k_crys = np.asarray(sym.unfolded_kpts[k_index], dtype=float)
    bd = np.asarray(wfn.bdot, dtype=float)
    pol = np.array([1.0, 0.0, 0.0], dtype=float)
    lpol = float(np.sqrt(pol @ (bd @ pol)))
    # weights per G for p: 2*(pol^T bd (G+k))/|pol|
    vproj_crys = (G_crys + k_crys[None, :])  # (nG,3)
    weights = 2.0 * (vproj_crys @ (bd @ pol)) / max(lpol, 1e-16)  # (nG,)
    nb_c = len(c_idx)
    nb_v = len(v_idx)
    p_block = np.zeros((nb_c, nb_v), dtype=np.complex128)
    full_block = np.zeros_like(p_block)
    for ic_pos, ic in enumerate(c_idx):
        cm = sym.get_cnk_fullzone(wfn, int(ic), k_index)  # (2,nG)
        for iv_pos, iv in enumerate(v_idx):
            cv = sym.get_cnk_fullzone(wfn, int(iv), k_index)  # (2,nG)
            acc_p = 0.0 + 0.0j
            for s in range(cm.shape[0]):
                acc_p += np.vdot(cm[s], cv[s] * weights)
            p_block[ic_pos, iv_pos] = acc_p
            # For now full=momentum only; vNL term not computed in cnk path
            full_block[ic_pos, iv_pos] = acc_p
    # Divide by ΔE for length-gauge
    try:
        enk = np.asarray(wfn.enk, dtype=float)
        e_c = enk[0, k_index, c_idx]
        e_v = enk[0, k_index, v_idx]
        dE = e_c[:, None] - e_v[None, :]
        mask = np.abs(dE) < 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            p_block = np.where(mask, 0.0, p_block / dE)
            full_block = np.where(mask, 0.0, full_block / dE)
    except Exception:
        pass
    return p_block, full_block


def compute_vnl_velocity_k_stub(*args, **kwargs) -> jax.Array:
    # Scaffold for the nonlocal pseudopotential velocity contribution v^NL_i = i [r_i, V_NL]
    #
    # Target formulation (Cartesian components i=x,y,z):
    #   [r_i, V_NL](q, q') = i Σ_{αβ} [ g_{α,i}(q) D_{αβ} β^*_β(q') - β_α(q) D_{αβ} g^*_{β,i}(q') ]
    # with α ≡ (κ,ℓ,m). Here β_α(q) are the (QE-style) projectors in reciprocal space and
    # g_α(q) = ∇_q β_α(q) is the Cartesian gradient of the projector evaluated at q = |k+G|.
    #
    # Practical recipe to compute g_α(q) stably without (k×G)-FFTs:
    #  1) Work in spherical coordinates of q: (q, θ, φ). Build the local spherical frame
    #     expressed in Cartesian: e_r(θ,φ), e_θ(θ,φ), e_φ(θ,φ).
    #  2) Let R_ℓ(q) denote the radial factor for the projector channel (ℓ,κ), and define
    #        A(q) = q^ℓ R_ℓ(q),    A'(q) = q^{ℓ-1} [ q R'_ℓ(q) + ℓ R_ℓ(q) ].
    #     Using spherical-harmonic Y_{ℓm}(θ,φ) and its surface gradient ∇_Ω Y_{ℓm}, the full
    #     3-vector gradient in Cartesian is
    #        ∇_q β_{κℓm}(q) = const · [ A'(q) Y_{ℓm} e_r + (A(q)/q) ∇_Ω Y_{ℓm} ],
    #     where const includes (4π i^ℓ)/√Ω and any species/j-channel prefactors.
    #  3) Real (tesseral) harmonics per QE:
    #        - Use isdf.psp.build_projectors_qe.qe_real_sph_harmonics(l, vectors)
    #          to generate Y'_{ℓm}(q̂).
    #        - isdf.psp.build_projectors_qe.U_complex_from_real(l) provides the U^ℓ rotation
    #          mapping real↔complex; reuse it to obtain ∇_Ω Y' from complex derivatives if
    #          that is simpler than direct cos/sin forms.
    #     Angular derivatives follow from the complex-harmonic rules with the U^ℓ rotation,
    #     or can be written directly in terms of cos(mφ)/sin(mφ) forms.
    #  4) Batch over all q-vectors (rows of K_cart) and over all projectors to form
    #        β_α(q) with shape (nq, nproj) and g_α(q) with shape (nq, nproj, 3),
    #     then assemble the commutator with two GEMMs against the D_{αβ} blocks per ℓ/species.
    #
    # Implementation notes for this codebase:
    #  - The per-species, per-ℓ plan with E (D) blocks and radial splines is provided by
    #      isdf.psp.projector_pipeline.build_vnl_plan. Extend that plan to expose R_ℓ(q) and
    #      its spline derivative R'_ℓ(q) for A and A'.
    #  - The current V_NL evaluator compute_V_NL_k_minimal already builds Y'_{ℓm}(K̂) and
    #      handles species/atom phase factors. Mirror that structure to produce g_α(q).
    #  - Guard the poles (sinθ→0) when forming (1/sinθ) ∂_φ Y. Prefer small ε or switch
    #      to a local orthonormal frame as needed.
    #  - Near q→0, prefer Bessel recurrences for radial pieces to avoid 0/0 from q^ℓ.
    #
    # Minimal placeholder: return zeros with correct shape (3, nb, nb).
    nb = int(args[0].shape[0])
    return jnp.zeros((3, nb, nb), dtype=jnp.complex128)


# --------------------------
# Main driver
# --------------------------

def main(argv=None):
	parser = argparse.ArgumentParser(description="Dipole/velocity matrix elements <mk|v|nk>")
	parser.add_argument(
		"-i",
		"--input",
		default="cohsex.in",
		help="Input file (INI-like) with [cohsex] block",
	)
	parser.add_argument(
		"--vnl-scale",
		type=str,
		default="1+0j",
		help="Complex scale for i[r,V_NL], e.g. 1, 0.8, 1+0.2j (default: 1+0j)",
	)
	parser.add_argument(
		"--vnl-velocity-mode",
		choices=["analytic", "fd"],
		default="analytic",
		help="Use analytic projector gradient or finite-difference Z for v_NL",
	)
	parser.add_argument(
		"--fprime-mode",
		choices=["spline", "bessel"],
		default="bessel",
		help="How to compute F'_l(q): spline derivative or Bessel-kernel integral",
	)
	parser.add_argument(
		"--vnl-fd-step",
		type=float,
		default=1e-5,
		help="Finite-difference step (|K_cart| units) when --vnl-velocity-mode=fd",
	)
	args = parser.parse_args(argv)

	input_path = Path(args.input).resolve()
	params = read_cohsex_input(str(input_path))
	# Resolve WFN relative to input file directory as preferred
	wfn_path = Path(params.get("wfn_file", "WFN.h5"))
	if not wfn_path.is_absolute():
		wfn_path = (input_path.parent / wfn_path).resolve()

	# Open WFN and symmetry
	wfn = WFNReader(str(wfn_path))
	sym = symmetry_maps.SymMaps(wfn)

	nval = int(params.get("nval", 5))
	ncond = int(params.get("ncond", 5))
	nband = int(params.get("nband", max(int(wfn.nbands), int(wfn.nelec) + ncond)))
	bispinor = bool(params.get("bispinor", False))

	print("\nCreating system metadata...")
	meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)

	# JAX mesh (simple 1D default; we can still use for constraints)
	total_devices = jax.process_count() * jax.local_device_count()
	grid_x = int(np.sqrt(total_devices))
	while total_devices % grid_x != 0:
		grid_x -= 1
	grid_y = total_devices // grid_x
	devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
	mesh_xy = Mesh(devices_2d, ['x', 'y'])

	print("\nLoading wavefunction coefficients to devices...")
	nband_eff = min(nband, int(wfn.nbands))
	brange = (0, nband_eff)
	global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
	print(f"  Loaded {nb_actual} bands in G-space, shape: {global_psi_G.shape}")

	print("\nScanning for pseudopotential files...")
	pseudos = load_pseudopotentials(str(input_path.parent))

	# Structure summary (reuse DFT helper)
	print_atomic_structure(wfn, pseudos)

	# Precompute per-species caches for nonlocal projectors
	assignments = build_atom_pp_assignments(jnp.asarray(wfn.atom_crys, dtype=jnp.float64), jnp.asarray(wfn.atom_types, dtype=jnp.int32), pseudos)
	species_payload: list[tuple[object, np.ndarray]] = []
	for ap in assignments:
		pseudo = ap.pseudo
		if pseudo is None:
			continue
		if not any(id(pseudo) == id(p) for p, _ in species_payload):
			positions = np.asarray([a.position for a in assignments if id(a.pseudo) == id(pseudo)], dtype=float)
			species_payload.append((pseudo, positions))

	# Precompute G and K scaffolding for all k-points
	Gk_crys_all: list[jnp.ndarray] = []
	K_crys_all: list[jnp.ndarray] = []
	K_cart_all: list[jnp.ndarray] = []
	K_norm_all: list[np.ndarray] = []
	bvec_np = np.asarray(wfn.bvec, dtype=float).T
	B = float(wfn.blat) * bvec_np.T
	for i in range(sym.nk_tot):
		Gk_crys_i, _ = generate_gvectors_k(i, sym, wfn, meta)
		Gk = np.asarray(Gk_crys_i, dtype=float)
		kvec = np.asarray(sym.unfolded_kpts[i], dtype=float)
		Kc = Gk + kvec[None, :]
		Kcart = Kc @ B
		Knorm = np.sqrt(np.sum(Kcart**2, axis=1))
		Gk_crys_all.append(jnp.asarray(Gk_crys_i, dtype=jnp.int32))
		K_crys_all.append(jnp.asarray(Kc, dtype=jnp.float64))
		K_cart_all.append(jnp.asarray(Kcart, dtype=jnp.float64))
		K_norm_all.append(Knorm)

	# Build minimal VNL plan
	q_max = 0.0
	for Knorm in K_norm_all:
		if Knorm.size:
			q_max = max(q_max, float(np.max(Knorm)))
	plan = build_vnl_plan(pseudos, assignments, float(wfn.cell_volume), float(q_max))

	# Reshard wavefunctions over mesh
	k_xy_shard = NamedSharding(mesh_xy, P(('x','y'), None, None, None, None, None))
	wfn_k_sharded = jax.lax.with_sharding_constraint(global_psi_G, k_xy_shard)

	nk = sym.nk_tot
	nb = int(wfn_k_sharded.shape[1])
	dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)

	for i in range(1):
		wfn_k = wfn_k_sharded[i]
		kpoint = jnp.asarray(sym.unfolded_kpts[i], dtype=jnp.float64)
		Gk_crys = Gk_crys_all[i]
		# Momentum per component
		p_cart = compute_p_operator_k(
			wfn_k,
			Gk_crys,
			kpoint,
			jnp.asarray(wfn.bdot, dtype=jnp.float64),
			jnp.asarray(wfn.bvec, dtype=jnp.float64),
			float(wfn.blat),
		)  # (3, nb, nb)
		# Nonlocal velocity components via commutator i[r_i, V_NL]
		if args.vnl_velocity_mode == "fd":
			vNL_cart = compute_V_NL_velocity_k_numeric(
				wfn_k,
				Gk_crys,
				K_crys_all[i],
				K_cart_all[i],
				plan,
				float(wfn.cell_volume),
				h=float(args.vnl_fd_step),
			)  # (3, nb, nb)
		else:
			vNL_cart = compute_V_NL_velocity_k(
				wfn_k,
				Gk_crys,
				K_crys_all[i],
				K_cart_all[i],
				plan,
				float(wfn.cell_volume),
				fprime_mode=str(args.fprime_mode),
			)  # (3, nb, nb)

		# Sign convention note (Liu-2024 Eq. 17 / BGW k·p):
		# Our internal assembly returns v^NL = -(∂_q + ∂_{q'}) V_NL, while BGW’s
		# reported ⟨v⟩ uses the opposite sign convention. Flip here so users don’t
		# need --vnl-scale=-1.0 when comparing to BGW outputs.
		vNL_cart = -vNL_cart
		# Optional rescaling (complex supported)
		try:
			_scale_complex = complex(args.vnl_scale)
		except Exception:
			# Fallback: parse as real float
			_scale_complex = complex(float(args.vnl_scale), 0.0)
		if _scale_complex != (1+0j):
			vNL_cart = jnp.asarray(_scale_complex, dtype=jnp.complex128) * vNL_cart
		dipole[:, i] = np.asarray(p_cart + vNL_cart)

		# Emit 4x4 x-direction blocks matching BGW vmtxel convention
		try:
			# Select explicit band-index windows: valence 22:26, conduction 26:30
			# BGW orders valence from top (nearest Fermi) downward along mband
			v_idx = np.arange(25, 21, -1, dtype=int)  # 25,24,23,22
			c_idx = np.arange(26, 30, dtype=int)      # 26,27,28,29
			# BGW "momentum operator" routine divides by m=0.5 (Ry units), i.e. returns p/m = 2*p
			# Project along reciprocal x-hat in crystal coordinates using BGW metric
			# pol_crys = (1,0,0); lpol = sqrt(pol^T bdot pol); projection uses pol_cart = B^T pol
			B = (np.asarray(wfn.bvec, dtype=float).T) * float(wfn.blat)
			pol_crys = np.array([1.0, 0.0, 0.0], dtype=float)
			pol_cart = B @ pol_crys  # Cartesian projection direction corresponding to crystal x-hat
			bdot_np = np.asarray(wfn.bdot, dtype=float)
			lpol = float(np.sqrt(pol_crys @ (bdot_np @ pol_crys)))
			# Momentum projection using BGW weighting with k+G (exact physics):
			# p_m_bgw_k = sum_{G,s} c*_mk(G,s) c_nk(G,s) * 2*(pol^T bdot (k+G))/|pol|
			p_proj = compute_projected_momentum_bgw_like(
				wfn_k,
				np.asarray(Gk_crys),
				np.asarray(kpoint),
				np.asarray(wfn.bdot),
				use_k_term=True,
			)
			# vNL projection: project along unit pol_cart (BGW metric-equivalent)
			v_proj = pol_cart[0] * np.asarray(vNL_cart[0]) + pol_cart[1] * np.asarray(vNL_cart[1]) + pol_cart[2] * np.asarray(vNL_cart[2])
			lpol_cart = float(np.linalg.norm(pol_cart))
			v_proj = v_proj / max(lpol_cart, 1e-16)
			mom_block = p_proj[np.ix_(c_idx, v_idx)]
			pre_full_block = (p_proj + v_proj)[np.ix_(c_idx, v_idx)]
			full_block = pre_full_block.copy()
			# Energies for ΔE normalization from WFN (full-zone index)
			k_red = int(sym.irk_to_k_map[i])
			energies = np.asarray(wfn.energies)
			e_c = energies[0, k_red, c_idx]
			e_v = energies[0, k_red, v_idx]
			dE = e_c[:, None] - e_v[None, :]
			dE_wfn = dE.copy()
			# Divide by E_c(k) - E_v(k) for file output only (BGW mtxel_m divide_energy behavior)
			eps = 1e-12
			mask = np.abs(dE_wfn) < eps
			with np.errstate(divide='ignore', invalid='ignore'):
				mom_block = np.where(mask, 0.0, mom_block / dE_wfn)
				full_block = np.where(mask, 0.0, full_block / dE_wfn)
			out_dir = input_path.parent
			out_dir.mkdir(parents=True, exist_ok=True)
			out_txt = (out_dir / 'vmtxel_4x4_isdf.txt').resolve()
			with out_txt.open('w') as f:
				f.write(f'# ISDF dipole x-component 4x4 blocks at k-index {int(i)} (crystal x̂)\n')
				f.write(f'# rows (c): {c_idx.tolist()}  cols (v): {v_idx.tolist()}\n')
				f.write('# Indices: rows = lowest 4 conduction (from Fermi), cols = highest 4 valence (to Fermi)\n')
				lpol_bdot = float(np.sqrt(pol_crys @ (bdot_np @ pol_crys)))
				f.write(f"# |pol|_bdot={lpol_bdot:.8f}  |B pol|_2={lpol_cart:.8f}\n")
				f.write('\n# Momentum only (p_x) real\n')
				for r in range(4):
					f.write(' '.join(f"{np.real(mom_block[r, c]):.5f}" for c in range(4)) + '\n')
				f.write('\n# Momentum only (p_x) imag\n')
				for r in range(4):
					f.write(' '.join(f"{np.imag(mom_block[r, c]):.5f}" for c in range(4)) + '\n')
				f.write('\n# Momentum + i[r,V_NL] (x) real\n')
				for r in range(4):
					f.write(' '.join(f"{np.real(full_block[r, c]):.5f}" for c in range(4)) + '\n')
				f.write('\n# Momentum + i[r,V_NL] (x) imag\n')
				for r in range(4):
					f.write(' '.join(f"{np.imag(full_block[r, c]):.5f}" for c in range(4)) + '\n')
				# 2x2 block Frobenius norms for Momentum + i[r,V_NL] (gauge-invariant)
				B00 = full_block[0:2, 0:2]
				B01 = full_block[0:2, 2:4]
				B10 = full_block[2:4, 0:2]
				B11 = full_block[2:4, 2:4]
				b00_fn = float(np.sqrt(np.sum(np.abs(B00)**2)))
				b01_fn = float(np.sqrt(np.sum(np.abs(B01)**2)))
				b10_fn = float(np.sqrt(np.sum(np.abs(B10)**2)))
				b11_fn = float(np.sqrt(np.sum(np.abs(B11)**2)))
				f.write('\n# 2x2 block Frobenius norms (Momentum + i[r,V_NL])\n')
				f.write(f"{b00_fn:.6f} {b01_fn:.6f}\n")
				f.write(f"{b10_fn:.6f} {b11_fn:.6f}\n")
			print(f"\nWrote ISDF 4x4 comparison to {out_txt.resolve()}")
		except Exception as e:
			print(f"Failed to write ISDF 4x4 comparison: {e}")

	# Print a small 4x4 grid of lowest dipole mtxels at first k for x,y,z (real parts)
	try:
		m = min(4, dipole.shape[2])
		labels = ['x', 'y', 'z']
		print("\nFirst k-point dipole matrix (real parts), 4x4 lowest bands:")
		for ic in range(3):
			block = np.real(dipole[ic, 0, :m, :m])
			print(f"  component {labels[ic]}:")
			for r in range(m):
				row = " ".join(f"{block[r, c]: .4e}" for c in range(m))
				print(f"    {row}")
	except Exception:
		pass

	# Save
	out_path = Path('dipole.h5').resolve()
	with h5py.File(str(out_path), 'w') as h5:
		h5.create_dataset('dipole_cart', data=dipole)
		h5.attrs['nbands'] = int(wfn.nbands)
		h5.attrs['nk'] = int(sym.nk_tot)
		h5.attrs['note'] = 'dipole_cart[3,x,y] = p_i + i[r_i, V_NL]'

	print(f"\nWrote dipole data to {out_path}")


if __name__ == '__main__':
	main()
