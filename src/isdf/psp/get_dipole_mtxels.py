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

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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

from isdf.psp.projector_pipeline import (
	build_vnl_plan,
	compute_V_NL_velocity_k,
    compute_V_NL_velocity_k_numeric,
    compute_V_NL_k_minimal,
)
from dataclasses import dataclass
import h5py
# --------------------------
# K+G helpers
# --------------------------


def build_K_vectors(Gk_crys, kpoint_crys, wfn):
	k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
	K_crys = jnp.asarray(Gk_crys, dtype=jnp.float64) + k_crys[None, :]
	B = jnp.asarray(wfn.bvec, dtype=jnp.float64) * float(wfn.blat)
	K_cart = jnp.asarray(K_crys) @ B
	return K_crys, K_cart


# --------------------------
# Velocity/dipole pieces and projector pipeline
# --------------------------


 

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
	B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
	K_cart = jnp.asarray(K_crys) @ B  # (nG, 3)
	p_mn = []
	for i in range(3):
		K_i = K_cart[:, i]
		weighted = C_bsg * K_i[None, None, :]
		p_i = jnp.einsum('msg,nsg->mn', jnp.conj(C_bsg), weighted, optimize=True)
		p_mn.append(p_i)
	return 2*jnp.stack(p_mn, axis=0)


def compute_projected_momentum_bgw_like(*args, **kwargs):
    raise NotImplementedError("Debug-only momentum projection removed")


def compute_block_direct_cnk(*args, **kwargs):
    raise NotImplementedError("Debug-only direct cnk path removed")

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
		"--divide-energy",
		action="store_true",
		help="Divide selected debug submatrices by ΔE (E_c - E_v). Default: off",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Print 4x6 x-direction debug table (momentum and momentum+vNL) for first k",
	)
	parser.add_argument(
		"--vnl-mode",
		choices=["analytic", "numeric"],
		default="analytic",
		help="Choose nonlocal velocity evaluation: analytic (dZ) or numeric FD(Z)",
	)
	parser.add_argument(
		"--vnl-h",
		type=float,
		default=1e-5,
		help="Finite-difference step h for --vnl-mode=numeric (in |K_cart| units)",
	)
	parser.add_argument(
		"--vnl-h-rel",
		type=float,
		default=0.0,
		help="Relative step: fraction of median |K|; used if larger than --vnl-h",
	)
	parser.add_argument(
		"--vnl-num-scheme",
		choices=["naive", "richardson"],
		default="naive",
		help="Numeric FD on V_NL: naive central or Richardson-extrapolated",
	)
	parser.add_argument(
		"--debug-kindex",
		type=int,
		default=1,
		help="k-point index to use for --debug table (0-based). Default: 1",
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
	# Choose target band count: at least nelec+ncond, clipped to file bands; honor user nband if larger
	try:
		nband_param = params.get("nband", None)
		if nband_param is None:
			nband = max(int(wfn.nbands), int(wfn.nelec) + int(ncond))
		else:
			nband = int(nband_param)
	except Exception:
		nband = max(int(wfn.nbands), int(wfn.nelec) + int(ncond))
	bispinor = bool(params.get("bispinor", False))

	print("\nCreating system metadata...")
	meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)

	# JAX mesh (simple 1D default; minimal sharding for demo)
	devices = np.array(jax.devices()).reshape(1, -1)
	mesh_xy = Mesh(devices, ['x', 'y'])

	print("\nLoading wavefunction coefficients to devices...")
	# Ensure we load enough conduction bands for debug/output comparisons
	nband_eff = min(int(wfn.nbands), max(int(wfn.nelec) + int(ncond), int(nband)))
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
	bvec_np = np.asarray(wfn.bvec, dtype=float)
	B = float(wfn.blat) * bvec_np
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
	deltaE = np.zeros((nk, nb, nb), dtype=np.float64)

	for i in range(nk):
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
		if args.vnl_mode == "numeric":
			# Numeric derivative on V_NL with optional Richardson and adaptive h
			B = (np.asarray(wfn.bvec, dtype=float)) * float(wfn.blat)
			Binv = np.linalg.inv(B)
			vNL_cart = np.zeros((3, nb, nb), dtype=np.complex128)
			K_cart_this = np.asarray(K_cart_all[i])
			K_med = float(np.median(np.linalg.norm(K_cart_this, axis=1))) if K_cart_this.size else 1.0
			h_base = max(float(args.vnl_h), float(args.vnl_h_rel) * max(K_med, 1.0))
			h1 = h_base
			h2 = 0.5 * h_base
			for ic in range(3):
				# D1 at h1
				d1 = np.zeros((3,), dtype=float); d1[ic] = h1
				d1c = d1 @ Binv
				Kp1_crys = np.asarray(K_crys_all[i]) + d1c[None, :]
				Km1_crys = np.asarray(K_crys_all[i]) - d1c[None, :]
				Kp1_cart = Kp1_crys @ B
				Km1_cart = Km1_crys @ B
				Vp1 = compute_V_NL_k_minimal(wfn_k, Gk_crys, Kp1_crys, Kp1_cart, plan, float(wfn.cell_volume))
				Vm1 = compute_V_NL_k_minimal(wfn_k, Gk_crys, Km1_crys, Km1_cart, plan, float(wfn.cell_volume))
				D1 = - (Vp1 - Vm1) / (2.0 * h1)
				if args.vnl_num_scheme == "richardson":
					# D2 at h2
					d2 = np.zeros((3,), dtype=float); d2[ic] = h2
					d2c = d2 @ Binv
					Kp2_crys = np.asarray(K_crys_all[i]) + d2c[None, :]
					Km2_crys = np.asarray(K_crys_all[i]) - d2c[None, :]
					Kp2_cart = Kp2_crys @ B
					Km2_cart = Km2_crys @ B
					Vp2 = compute_V_NL_k_minimal(wfn_k, Gk_crys, Kp2_crys, Kp2_cart, plan, float(wfn.cell_volume))
					Vm2 = compute_V_NL_k_minimal(wfn_k, Gk_crys, Km2_crys, Km2_cart, plan, float(wfn.cell_volume))
					D2 = - (Vp2 - Vm2) / (2.0 * h2)
					vNL_cart[ic] = (4.0 * D2 - D1) / 3.0
				else:
					vNL_cart[ic] = D1
		else:
			vNL_cart = compute_V_NL_velocity_k(
				wfn_k,
				Gk_crys,
				K_crys_all[i],
				K_cart_all[i],
				plan,
				float(wfn.cell_volume),
				fprime_mode="bessel",
			)  # (3, nb, nb)

		# Sign convention note (Liu-2024 Eq. 17 / BGW k·p):
		# Our internal assembly returns v^NL = -(∂_q + ∂_{q'}) V_NL, while BGW’s
		# reported ⟨v⟩ uses the opposite sign convention. Flip here so users don’t
		# need --vnl-scale=-1.0 when comparing to BGW outputs.
		vNL_cart = -vNL_cart
		dipole[:, i] = np.asarray(p_cart + vNL_cart)

		# ΔE matrix for this k from band energies
		try:
			k_red = int(sym.irk_to_k_map[i])
		except Exception:
			k_red = int(i)
		energies = np.asarray(wfn.energies)
		if energies.ndim >= 3:
			e_b = np.asarray(energies[0, k_red, :nb], dtype=float)
		else:
			e_b = np.asarray(energies[:nb], dtype=float)
		deltaE[i] = e_b[:, None] - e_b[None, :]

		# Optional debug: print 4x6 x-direction blocks for selected k index
		if args.debug and int(i) == int(args.debug_kindex):
			# Choose up to 6 valence (highest) and up to 4 conduction (lowest) bands
			nelec = int(wfn.nelec)
			v_count = min(6, max(0, nelec))
			c_count = min(4, max(0, nb - nelec))
			if v_count == 0 or c_count == 0:
				print("[DEBUG] Skipping 4x6 debug blocks: insufficient v/c bands (v_count=", v_count, ", c_count=", c_count, ")")
				continue
			v_idx = np.arange(nelec - 1, nelec - v_count - 1, -1, dtype=int)  # descending
			c_idx = np.arange(nelec, nelec + c_count, dtype=int)              # ascending
			p_x = np.asarray(p_cart[0])
			full_x = np.asarray(p_cart[0] + vNL_cart[0])
			if args.divide_energy:
				with np.errstate(divide='ignore', invalid='ignore'):
					dE = deltaE[i]
					p_x = np.where(np.abs(dE) < 1e-12, 0.0, p_x / dE)
					full_x = np.where(np.abs(dE) < 1e-12, 0.0, full_x / dE)
			mom_block = p_x[np.ix_(c_idx, v_idx)]
			full_block = full_x[np.ix_(c_idx, v_idx)]
			print("\n[DEBUG] 4x6 x-direction momentum block (real):")
			for r in range(mom_block.shape[0]):
				print(' '.join(f"{np.real(mom_block[r, c]):.5f}" for c in range(mom_block.shape[1])))
			print("[DEBUG] 4x6 x-direction momentum block (imag):")
			for r in range(mom_block.shape[0]):
				print(' '.join(f"{np.imag(mom_block[r, c]):.5f}" for c in range(mom_block.shape[1])))
			print("[DEBUG] 4x6 x-direction (p + vNL) block (real):")
			for r in range(full_block.shape[0]):
				print(' '.join(f"{np.real(full_block[r, c]):.5f}" for c in range(full_block.shape[1])))
			print("[DEBUG] 4x6 x-direction (p + vNL) block (imag):")
			for r in range(full_block.shape[0]):
				print(' '.join(f"{np.imag(full_block[r, c]):.5f}" for c in range(full_block.shape[1])))

			# 2x3 grid of 2x2 Frobenius norms from the 4x6 (p+vNL) block, matching parse_vmtxel.py
			if full_block.shape[0] >= 4 and full_block.shape[1] >= 6:
				B00 = full_block[0:2, 0:2]
				B01 = full_block[0:2, 2:4]
				B02 = full_block[0:2, 4:6]
				B10 = full_block[2:4, 0:2]
				B11 = full_block[2:4, 2:4]
				B12 = full_block[2:4, 4:6]
				fn00 = float(np.linalg.norm(B00, ord='fro'))
				fn01 = float(np.linalg.norm(B01, ord='fro'))
				fn02 = float(np.linalg.norm(B02, ord='fro'))
				fn10 = float(np.linalg.norm(B10, ord='fro'))
				fn11 = float(np.linalg.norm(B11, ord='fro'))
				fn12 = float(np.linalg.norm(B12, ord='fro'))
				print("[DEBUG] 2x3 grid of 2x2 Frobenius norms (|p+vNL|, x-direction):")
				print(f"  {fn00:.6f} {fn01:.6f} {fn02:.6f}")
				print(f"  {fn10:.6f} {fn11:.6f} {fn12:.6f}")


	# Save to dipole.h5 with deltaE
	out_path = Path('dipole.h5').resolve()
	with h5py.File(str(out_path), 'w') as h5:
		h5.create_dataset('dipole_cart', data=dipole)
		h5.create_dataset('deltaE', data=deltaE)
		h5.attrs['nbands'] = int(wfn.nbands)
		h5.attrs['nk'] = int(sym.nk_tot)
		h5.attrs['note'] = 'dipole_cart[3,x,y] = p_i + i[r_i, V_NL]; deltaE[k,:,:] = E_b - E_b\''
	print(f"\nWrote dipole data to {out_path}")


if __name__ == '__main__':
    main()
