"""Test BSE matvec with data from COHSEX restart files.

Run with:
    uv run python -m bse_isdf.test_bse -i cohsex_prod.in

Uses W0_qmunu from isdf_tensors_*.h5 when available, otherwise falls back to V_qmunu.
Tests on a small subset of bands (4 valence, 4 conduction) for fast iteration.

Profiling:
    # Enable JAX profiler tracing (creates tensorboard-compatible files)
    ISDF_JAX_PROFILE_DIR=./jax_traces uv run python -m bse_isdf.test_bse -i cohsex_prod.in
    
    # View with:
    tensorboard --logdir=./jax_traces
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import h5py
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import common.timing as timing
from common import jax_profile

from .bse_jax import (
    apply_bse_hamiltonian_single_device,
    solve_bse,
    compute_pair_amplitude,
)
from .write_eigenvectors import write_eigenvectors_h5, generate_kpts_grid


def load_bse_data_from_restart(
    restart_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    fermi_energy: float = 0.0,
):
    """Load BSE data from COHSEX restart file.
    
    Args:
        restart_file: Path to isdf_tensors_*.h5 (fallback: taggedarrays.h5)
        n_val: Number of valence bands (highest occupied)
        n_cond: Number of conduction bands (lowest unoccupied)
        fermi_energy: Fermi energy in Ry. Default 0.0 (DFT referenced to VBM).
        
    Returns:
        Dictionary with all BSE inputs
    """
    print(f"Loading data from {restart_file}")
    
    with h5py.File(restart_file, 'r') as f:
        # V_qmunu: (1, 1, 1, nkx, nky, nkz, n_rmu, n_rmu)
        V_qmunu = jnp.asarray(f['V_qmunu'][:])
        if "W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            W0_qmunu = jnp.asarray(f["W0_qmunu"][:])
        else:
            W0_qmunu = None
        
        # psi_l: (nk, nb_l, nspinor, n_rmu) - left bands
        # psi_r: (nk, nb_r, nspinor, n_rmu) - right bands
        psi_l = jnp.asarray(f['psi_l'][:])
        psi_r = jnp.asarray(f['psi_r'][:])
        
        # enk_l: (nk, nb_l), enk_r: (nk, nb_r)
        enk_l = jnp.asarray(f['enk_l'][:])
        enk_r = jnp.asarray(f['enk_r'][:])
        
    print(f"  V_qmunu shape: {V_qmunu.shape}")
    if W0_qmunu is not None:
        print(f"  W0_qmunu shape: {W0_qmunu.shape}")
    print(f"  psi_l shape: {psi_l.shape}, psi_r shape: {psi_r.shape}")
    print(f"  enk_l shape: {enk_l.shape}, enk_r shape: {enk_r.shape}")
    
    # Extract k-grid dimensions from V_qmunu
    nkx, nky, nkz = V_qmunu.shape[3:6]
    nk = nkx * nky * nkz
    n_rmu = V_qmunu.shape[-1]
    nspinor = psi_l.shape[2]
    nb_l = psi_l.shape[1]
    nb_r = psi_r.shape[1]
    
    print(f"  k-grid: {nkx}×{nky}×{nkz} = {nk}, n_rmu: {n_rmu}, nspinor: {nspinor}")
    print(f"  nb_l: {nb_l}, nb_r: {nb_r}")
    
    # Mean energy per band across k-points
    mean_enk_l = jnp.mean(enk_l, axis=0)
    mean_enk_r = jnp.mean(enk_r, axis=0)
    
    # Valence: bands with energy < Fermi (occupied)
    val_mask_l = mean_enk_l < fermi_energy
    n_val_available = int(jnp.sum(val_mask_l))
    
    # Conduction: bands with energy > Fermi (unoccupied)
    cond_mask_r = mean_enk_r > fermi_energy
    n_cond_available = int(jnp.sum(cond_mask_r))
    
    ryd2ev = 13.6056980659
    print(f"  Fermi energy: {fermi_energy:.4f} Ry = {fermi_energy * ryd2ev:.3f} eV")
    print(f"  Available: {n_val_available} valence, {n_cond_available} conduction bands")
    
    # Limit to requested number
    n_val = min(n_val, n_val_available)
    n_cond = min(n_cond, n_cond_available)
    
    if n_val == 0 or n_cond == 0:
        raise ValueError(f"No valence ({n_val_available}) or conduction ({n_cond_available}) bands found with Fermi={fermi_energy}")
    
    # Get indices of top valence (highest energy occupied) and bottom conduction (lowest unoccupied)
    val_energies = jnp.where(val_mask_l, mean_enk_l, -jnp.inf)
    val_indices = jnp.argsort(val_energies)[-n_val:]  # Top n_val by energy
    
    cond_energies = jnp.where(cond_mask_r, mean_enk_r, jnp.inf)
    cond_indices = jnp.argsort(cond_energies)[:n_cond]  # Bottom n_cond by energy
    
    print(f"  Valence band indices: {list(map(int, val_indices))}")
    print(f"  Conduction band indices: {list(map(int, cond_indices))}")
    
    # Extract wavefunctions and energies
    psi_v = psi_l[:, val_indices, :, :]  # (nk, n_val, nspinor, n_rmu)
    eps_v = enk_l[:, val_indices]        # (nk, n_val)
    
    psi_c = psi_r[:, cond_indices, :, :]  # (nk, n_cond, nspinor, n_rmu)
    eps_c = enk_r[:, cond_indices]        # (nk, n_cond)
    
    print(f"  Selected: {n_val} valence, {n_cond} conduction bands")
    print(f"  psi_v shape: {psi_v.shape}, psi_c shape: {psi_c.shape}")
    print(f"  Valence energy range: [{float(eps_v.min()):.4f}, {float(eps_v.max()):.4f}] Ry")
    print(f"  Conduction energy range: [{float(eps_c.min()):.4f}, {float(eps_c.max()):.4f}] Ry")
    gap_ry = float(eps_c.min() - eps_v.max())
    print(f"  Gap: {gap_ry:.4f} Ry = {gap_ry * ryd2ev:.3f} eV")
    
    # Extract V at q=0 for direct term: (n_rmu, n_rmu)
    V_q0 = V_qmunu[0, 0, 0, 0, 0, 0, :, :]
    print(f"  V_q0 shape: {V_q0.shape}")
    
    # Get W_q from W0_qmunu if available, else use bare V as placeholder
    # Shape: (n_rmu, n_rmu, nkx, nky, nkz)
    W_src = W0_qmunu if W0_qmunu is not None else V_qmunu
    W_q = W_src[0, 0, 0, :, :, :, :, :].transpose(3, 4, 0, 1, 2)
    print(f"  W_q shape: {W_q.shape}")
    
    return {
        'psi_c': psi_c,
        'psi_v': psi_v,
        'eps_c': eps_c,
        'eps_v': eps_v,
        'W_q': W_q,
        'V_q0': V_q0,
        'nkx': nkx,
        'nky': nky,
        'nkz': nkz,
        'nk': nk,
        'n_rmu': n_rmu,
        'nspinor': nspinor,
        'n_val': n_val,
        'n_cond': n_cond,
        'fermi_energy': fermi_energy,
    }


def test_matvec(data, n_warmup=2, n_bench=10):
    """Test BSE matvec on a single trial vector."""
    print("\n=== Testing BSE matvec ===")
    
    nk = data['nk']
    nc = data['n_cond']
    nv = data['n_val']
    
    # Create random trial vector
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (1, nc, nv, nk), dtype=jnp.float64)
    X = X + 1j * jax.random.normal(jax.random.PRNGKey(43), (1, nc, nv, nk), dtype=jnp.float64)
    X = X / jnp.linalg.norm(X)
    
    print(f"Trial vector shape: {X.shape}")
    
    # Warm-up: compile the JIT function
    print(f"  Warming up JIT ({n_warmup} calls)...")
    with timing.section("bse.matvec.warmup"):
        for _ in range(n_warmup):
            HX = apply_bse_hamiltonian_single_device(
                X, data['psi_c'], data['psi_v'], data['eps_c'], data['eps_v'],
                data['W_q'], data['V_q0'], data['nkx'], data['nky'], data['nkz'],
            )
            HX.block_until_ready()
    
    # Benchmark matvec
    print(f"  Benchmarking ({n_bench} calls)...")
    with timing.section("bse.matvec.bench") as sec:
        for i in range(n_bench):
            with jax_profile.step_annotation("matvec", step_num=i):
                HX = apply_bse_hamiltonian_single_device(
                    X, data['psi_c'], data['psi_v'], data['eps_c'], data['eps_v'],
                    data['W_q'], data['V_q0'], data['nkx'], data['nky'], data['nkz'],
                )
        sec.watch(HX)
    
    print(f"Output shape: {HX.shape}")
    
    # Compute expectation value <X|H|X>
    E = jnp.vdot(X.flatten(), HX.flatten()).real
    ryd2ev = 13.6056980659
    print(f"Expectation value <X|H|X>: {E:.6f} Ry = {E * ryd2ev:.4f} eV")
    
    return HX


def test_pair_amplitude(data):
    """Test spin-traced pair amplitude computation."""
    print("\n=== Testing pair amplitude ===")
    
    psi_c = data['psi_c']
    psi_v = data['psi_v']
    
    with timing.section("bse.pair_amplitude"):
        M = compute_pair_amplitude(psi_c, psi_v)
        M.block_until_ready()
    
    print(f"psi_c shape: {psi_c.shape}")
    print(f"psi_v shape: {psi_v.shape}")
    print(f"Pair amplitude M shape: {M.shape}")  # Should be (nk, nc, nv, n_rmu)
    print(f"M is spin-traced (scalar at each k,c,v,μ point)")
    print(f"|M|_max: {float(jnp.max(jnp.abs(M))):.6f}")
    
    return M


def test_lanczos(data, n_eig=5, max_iter=50, use_jit_lanczos=True):
    """Test Lanczos solver for lowest eigenvalues."""
    lanczos_type = "JIT" if use_jit_lanczos else "Python-loop"
    print(f"\n=== Testing Lanczos solver ({lanczos_type}, n_eig={n_eig}, max_iter={max_iter}) ===")
    
    with timing.section("bse.lanczos") as sec:
        with jax_profile.trace_section("bse_lanczos"):
            eigenvalues, eigenvectors = solve_bse(
                data['psi_c'],
                data['psi_v'],
                data['eps_c'],
                data['eps_v'],
                data['W_q'],
                data['V_q0'],
                data['nkx'],
                data['nky'],
                data['nkz'],
                n_eig=n_eig,
                max_iter=max_iter,
                use_block=False,
                use_jit_lanczos=use_jit_lanczos,
                n_reorth=10,  # Reorthogonalize against last 10 vectors
            )
        sec.watch(eigenvalues, eigenvectors)
    
    print(f"\nLowest {n_eig} exciton energies:")
    for i, E in enumerate(eigenvalues):
        ryd2ev = 13.6056980659
        E_eV = float(E) * ryd2ev  # Ry to eV
        print(f"  {i+1}: {float(E):.6f} Ry = {E_eV:.4f} eV")
    
    print(f"\nEigenvector shapes: {eigenvectors.shape}")
    
    # Verify eigenvalues by computing <ψ|H|ψ> for each
    print("\nVerifying eigenvalues via <ψ|H|ψ>:")
    with timing.section("bse.verify_eigenvalues") as sec:
        for i in range(min(3, n_eig)):
            psi = eigenvectors[i:i+1]  # (1, nc, nv, nk)
            Hpsi = apply_bse_hamiltonian_single_device(
                psi, data['psi_c'], data['psi_v'], data['eps_c'], data['eps_v'],
                data['W_q'], data['V_q0'], data['nkx'], data['nky'], data['nkz'],
            )
            E_check = jnp.vdot(psi.flatten(), Hpsi.flatten()).real
            residual = jnp.linalg.norm(Hpsi - eigenvalues[i] * psi)
            print(f"  E[{i}]: {float(eigenvalues[i]):.6f}, <ψ|H|ψ>: {float(E_check):.6f}, |Hψ - Eψ|: {float(residual):.2e}")
        sec.watch(Hpsi)
    
    return eigenvalues, eigenvectors


def find_restart_file(input_file: str) -> str:
    """Find the restart file from input file directory."""
    input_dir = os.path.dirname(os.path.abspath(input_file))
    
    # Try common patterns
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, 'tmp', 'isdf_tensors_*.h5'))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, 'isdf_tensors_*.h5'))))
    candidates.extend([
        os.path.join(input_dir, 'tmp', 'taggedarrays600.h5'),
        os.path.join(input_dir, 'tmp', 'taggedarrays.h5'),
        os.path.join(input_dir, 'taggedarrays.h5'),
    ])
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find restart file in {input_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test BSE matvec with COHSEX restart data")
    parser.add_argument('-i', '--input', required=True, help="COHSEX input file (used for directory context)")
    parser.add_argument('--n-val', type=int, default=4, help="Number of valence bands")
    parser.add_argument('--n-cond', type=int, default=4, help="Number of conduction bands")
    parser.add_argument('--n-eig', type=int, default=10, help="Number of eigenvalues to compute")
    parser.add_argument('--max-iter', type=int, default=50, help="Maximum Lanczos iterations")
    parser.add_argument('--n-warmup', type=int, default=2, help="JIT warmup iterations")
    parser.add_argument('--n-bench', type=int, default=10, help="Benchmark iterations")
    parser.add_argument('--no-jit-lanczos', action='store_true', help="Use Python-loop Lanczos instead of JIT")
    parser.add_argument('--write-eigenvectors', type=str, default=None,
                       help="Write eigenvectors to HDF5 file (e.g., eigenvectors.h5)")
    args = parser.parse_args(argv)
    
    # Initialize timing
    timing.reset()
    
    print("=" * 60)
    print("BSE Test with COHSEX Restart Data")
    print("=" * 60)
    
    # Print JAX device info
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    if os.environ.get("ISDF_JAX_PROFILE_DIR"):
        print(f"JAX profiler output: {os.environ['ISDF_JAX_PROFILE_DIR']}")
    
    # Find restart file
    restart_file = find_restart_file(args.input)
    
    # Load data
    with timing.section("bse.load_data"):
        data = load_bse_data_from_restart(restart_file, n_val=args.n_val, n_cond=args.n_cond)
    
    # Run tests
    with jax_profile.trace_section("bse_test"):
        test_pair_amplitude(data)
        test_matvec(data, n_warmup=args.n_warmup, n_bench=args.n_bench)
        
        # Run Lanczos
        eigenvalues, eigenvectors = test_lanczos(
            data,
            n_eig=args.n_eig,
            max_iter=args.max_iter,
            use_jit_lanczos=not args.no_jit_lanczos,
        )
    
    # Write eigenvectors if requested
    if args.write_eigenvectors:
        print(f"\n=== Writing eigenvectors to {args.write_eigenvectors} ===")
        with timing.section("bse.write_eigenvectors"):
            kpts = generate_kpts_grid(data['nkx'], data['nky'], data['nkz'])
            write_eigenvectors_h5(
                args.write_eigenvectors,
                np.asarray(eigenvalues),
                np.asarray(eigenvectors),
                kpts,
                n_val=data['n_val'],
                n_cond=data['n_cond'],
                nkx=data['nkx'],
                nky=data['nky'],
                nkz=data['nkz'],
            )
    
    print("\n" + "=" * 60)
    print("BSE Test Complete!")
    print("=" * 60)
    
    # Print timing report
    timing.report(title="\n--- BSE Timing Report ---")


if __name__ == "__main__":
    main()
