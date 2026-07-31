"""Test BSE matvec with data from COHSEX restart files.

Run with:
    uv run python tests/bench/test_bse.py -i cohsex_prod.in

Uses W0_qmunu from isdf_tensors_*.h5 when available, otherwise falls back to V_qmunu.
Tests on a small subset of bands (4 valence, 4 conduction) for fast iteration.

Profiling:
    # Enable JAX profiler tracing (creates tensorboard-compatible files)
    ISDF_JAX_PROFILE_DIR=./jax_traces uv run python tests/bench/test_bse.py -i cohsex_prod.in
    
    # View with:
    tensorboard --logdir=./jax_traces
"""
from __future__ import annotations
import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import common.timing as timing
from common import jax_profile

from bse.bse_jax import (
    apply_bse_hamiltonian_single_device,
    solve_bse,
    compute_pair_amplitude,
)
from bse.bse_io import (
    _find_restart_file,
    _load_ring_subset,
    write_eigenvectors_stream,
)


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


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Test BSE matvec with COHSEX restart data")
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
    restart_file = _find_restart_file(args.input)

    # Load data via the canonical single-device loader (px=py=1).
    with timing.section("bse.load_data"):
        data = _load_ring_subset(
            restart_file, args.n_val, args.n_cond, 1, 1, input_file=args.input)
        data['n_cond'] = data['psi_c'].shape[1]
        data['n_val'] = data['psi_v'].shape[1]
    
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
            n_write = len(eigenvalues)
            write_eigenvectors_stream(
                args.write_eigenvectors,
                eigenvalues,
                eigenvectors,
                data['n_val'],
                data['n_cond'],
                data['nkx'],
                data['nky'],
                data['nkz'],
                n_write,
            )
    
    print("\n" + "=" * 60)
    print("BSE Test Complete!")
    print("=" * 60)
    
    # Print timing report
    timing.report(title="\n--- BSE Timing Report ---")


if __name__ == "__main__":
    main()
