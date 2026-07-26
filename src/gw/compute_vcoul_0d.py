"""
Cell box truncation of the Coulomb potential for 0D (molecular) systems.

Implements the BerkeleyGW algorithm from Common/trunc_cell_box.f90:

  1. Build a real-space grid on a denser mesh (n_in_box × FFT grid).
  2. At each grid point, compute the minimum-image distance to the origin
     using the Wigner-Seitz construction (loop over ±ncell lattice replicas).
     A half-grid shift (trunc_shift = 0.5) avoids the r=0 singularity.
  3. Set V_trunc(r) = scale / r_min, where scale = 2 * sqrt(det(adot))
     is a normalization factor from the real-space metric.
  4. FFT V_trunc(r) -> V_trunc(G), then apply a phase correction
     exp(-2πi G·shift/dNfft) to undo the half-grid shift.
  5. Extract the real part for each G-vector in the Coulomb list.

Unlike 2D slab truncation, no analytic formula exists for the box-truncated
Coulomb in reciprocal space. The G=0 component is computed naturally by the
FFT (it equals the integral of scale/r over the WS cell), so no miniBZ
averaging is needed for the truncated potential.

Reference: BerkeleyGW source files:
  - Common/trunc_cell_box.f90 (serial implementation)
  - Common/vcoul_generator.f90 (driver; confirms no miniBZ avg for box trunc)
  - Common/nrtype.f90 (n_in_box=2, ncell=3, trunc_shift=0.5)

Units: output is in Rydberg, matching BerkeleyGW convention: v(G) = 8π/|G|²
for the untruncated case.
"""

import numpy as np
from numpy.typing import NDArray


# BGW parameters (Common/nrtype.f90)
N_IN_BOX = 2     # extra grid density factor for real-space FFT
NCELL = 3         # ±ncell replicas for Wigner-Seitz minimum image
TRUNC_SHIFT = 0.5 # half-grid shift to avoid 1/r singularity at origin


def _round_up_fft_size(n: int, max_prime: int = 5) -> int:
    """Round n up to a smooth FFT size (only prime factors <= max_prime).

    Mirrors BGW's setup_FFT_sizes / check_FFT_size with Nfac=3,
    which allows only factors of 2, 3, 5.
    """
    primes = [2, 3, 5, 7, 11, 13][:max_prime]
    while True:
        rem = n
        for p in primes:
            while rem % p == 0:
                rem //= p
        if rem == 1:
            return n
        n += 1


def compute_vcoul_box(
    bdot: NDArray,
    fft_grid: NDArray,
    gvecs: NDArray,
) -> NDArray:
    """
    Compute cell-box-truncated Coulomb potential v(G) for q=0.

    Parameters
    ----------
    bdot : (3, 3) float64
        Reciprocal-space metric matrix, bdot[i,j] = b_i · b_j,
        where b_i are reciprocal lattice vectors (2π/a units).
        In Rydberg atomic units this is what BerkeleyGW stores.
    fft_grid : (3,) int
        FFT grid dimensions from the WFN file (e.g., [45, 45, 45]).
    gvecs : (nG, 3) int
        G-vector components in crystal (integer) coordinates,
        ordered as they appear in the WFN/vcoul file.

    Returns
    -------
    vcoul : (nG,) float64
        Truncated Coulomb potential for each G-vector, in Rydberg.
    """
    fft_grid = np.asarray(fft_grid, dtype=int)
    bdot = np.asarray(bdot, dtype=np.float64)
    gvecs = np.asarray(gvecs, dtype=int)

    # --- Step 1: Set up FFT grids ---
    # Nfft: rounded-up FFT-friendly version of the WFN FFT grid
    Nfft = np.array([_round_up_fft_size(int(g)) for g in fft_grid])
    # dNfft: denser grid for real-space potential (n_in_box × Nfft)
    dkmax = fft_grid * N_IN_BOX
    dNfft = np.array([_round_up_fft_size(int(g)) for g in dkmax])

    # --- Step 2: Real-space metric ---
    # adot = (4π²) * bdot^{-1}, then scaled by 1/(dNfft_i * dNfft_j)
    # so that grid-index differences map directly to physical distances:
    #   |r|² = tt^T @ adot @ tt  (tt in grid-index units)
    adot = np.linalg.inv(bdot) * (4.0 * np.pi**2)
    for i in range(3):
        for j in range(3):
            adot[i, j] /= dNfft[i] * dNfft[j]

    # --- Step 3: Scale factor = 2 * sqrt(det(adot)) ---
    # This normalizes V(r) so that V(G) matches the 8π/|G|² convention.
    scale = 2.0 * np.sqrt(np.linalg.det(adot))

    # --- Step 4: Build V_trunc(r) on the dense real-space grid ---
    # At each grid point r, find the minimum-image distance to the origin
    # by looping over ±ncell replicas of the supercell (Wigner-Seitz truncation).
    fftbox = np.zeros(tuple(dNfft), dtype=np.complex128)

    # Precompute replica offsets: l * dNfft for l in [-ncell+1, ncell]
    replica_offsets = []
    for l3 in range(-NCELL + 1, NCELL + 1):
        for l2 in range(-NCELL + 1, NCELL + 1):
            for l1 in range(-NCELL + 1, NCELL + 1):
                replica_offsets.append(np.array([
                    l1 * dNfft[0],
                    l2 * dNfft[1],
                    l3 * dNfft[2],
                ], dtype=np.float64))

    # Vectorized: build coordinate arrays for the full grid
    i1 = np.arange(dNfft[0], dtype=np.float64) + TRUNC_SHIFT
    i2 = np.arange(dNfft[1], dtype=np.float64) + TRUNC_SHIFT
    i3 = np.arange(dNfft[2], dtype=np.float64) + TRUNC_SHIFT
    rr1, rr2, rr3 = np.meshgrid(i1, i2, i3, indexing='ij')

    # Find minimum image distance at each grid point
    r_len_sq = np.full(tuple(dNfft), np.inf, dtype=np.float64)
    for offset in replica_offsets:
        tt1 = rr1 - offset[0]
        tt2 = rr2 - offset[1]
        tt3 = rr3 - offset[2]
        # |t|² = tt^T @ adot @ tt (metric-weighted distance squared)
        d_sq = (
            adot[0, 0] * tt1 * tt1 + adot[1, 1] * tt2 * tt2 + adot[2, 2] * tt3 * tt3
            + 2.0 * adot[0, 1] * tt1 * tt2
            + 2.0 * adot[0, 2] * tt1 * tt3
            + 2.0 * adot[1, 2] * tt2 * tt3
        )
        r_len_sq = np.minimum(r_len_sq, d_sq)

    r_len = np.sqrt(r_len_sq)
    fftbox = (scale / r_len).astype(np.complex128)

    # --- Step 5: FFT to reciprocal space ---
    # V_trunc(r) -> V_trunc(G) using orthonormal FFT convention:
    #   F(G) = (1/sqrt(N)) sum_r f(r) exp(-2πi G·r/N)
    # BGW uses unnormalized FFTs, so we compensate by multiplying scale
    # by sqrt(N) so that the ortho FFT gives the same result.
    N_dense = int(np.prod(dNfft))
    fftbox *= np.sqrt(N_dense)
    fftbox_G = np.fft.fftn(fftbox, norm="ortho")

    # --- Step 6: Extract V(G) for each G-vector with phase correction ---
    # The half-grid shift introduced a phase that must be removed:
    #   phase(G) = 2π * sum_i (G_i * trunc_shift / dNfft_i)
    # and we multiply by exp(-i * phase) to undo it.
    #
    # G-vectors are in crystal (integer) coords; we map them to the dense grid
    # indices using the same convention as BGW: for j in [-Nfft/2, Nfft/2-1],
    # the dense-grid index is j if j>=0, else dNfft+j (since dNfft is a multiple
    # of Nfft for n_in_box integer, j maps directly).
    ncoul = gvecs.shape[0]
    vcoul = np.zeros(ncoul, dtype=np.float64)

    for ig in range(ncoul):
        j1, j2, j3 = gvecs[ig]

        # Map crystal G-index to dense FFT grid index
        di1 = j1 if j1 >= 0 else dNfft[0] + j1
        di2 = j2 if j2 >= 0 else dNfft[1] + j2
        di3 = j3 if j3 >= 0 else dNfft[2] + j3

        # Phase correction for the trunc_shift offset
        phase = 2.0 * np.pi * (
            j1 * TRUNC_SHIFT / dNfft[0]
            + j2 * TRUNC_SHIFT / dNfft[1]
            + j3 * TRUNC_SHIFT / dNfft[2]
        )
        vtemp = fftbox_G[di1, di2, di3] * complex(np.cos(phase), -np.sin(phase))
        vcoul[ig] = vtemp.real

    return vcoul


def main():
    """Load WFN.h5, compute box-truncated vcoul, compare with BGW vcoul file."""
    import argparse
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    from file_io import WfnLoader as WFNReader

    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Compute cell-box-truncated Coulomb potential and compare with BGW vcoul"
    )
    parser.add_argument("--wfn", default="WFN.h5", help="WFN HDF5 file")
    parser.add_argument("--vcoul", default="vcoul", help="BGW vcoul text file for comparison")
    args = parser.parse_args()

    wfn = WFNReader(args.wfn)

    # bdot: reciprocal metric in Bohr^-2, stored directly in the WFN file.
    # This is bdot[i,j] = b_i · b_j where b_i are in 2π/alat units,
    # NOT the same as bvec @ bvec.T (which would be in normalized units).
    bdot = wfn.bdot

    # G-vectors for q=0: all G-vectors in the WFN file for k-point 0
    gvecs_all = wfn.get_gvec_nk(0)  # (ngk, 3) integer crystal coords

    print(f"FFT grid: {wfn.fft_grid}")
    print(f"Number of G-vectors: {gvecs_all.shape[0]}")
    print(f"Cell volume: {wfn.cell_volume:.4f} bohr³")

    vcoul = compute_vcoul_box(bdot, wfn.fft_grid, gvecs_all)

    # Load BGW reference if available
    try:
        data = np.loadtxt(args.vcoul)
        gvecs_ref = data[:, 3:6].astype(int)
        vcoul_ref = data[:, 6]
        print(f"\nBGW vcoul loaded: {len(vcoul_ref)} G-vectors")

        # Match G-vectors between our computation and the reference
        # Build a dict from G-vector tuple -> index for our results
        our_gvec_map = {}
        for i, g in enumerate(gvecs_all):
            our_gvec_map[tuple(g)] = i

        matched = 0
        max_err = 0.0
        max_rel_err = 0.0
        for i in range(len(vcoul_ref)):
            key = tuple(gvecs_ref[i])
            if key in our_gvec_map:
                j = our_gvec_map[key]
                err = abs(vcoul[j] - vcoul_ref[i])
                ref_val = abs(vcoul_ref[i])
                rel = err / ref_val if ref_val > 1e-10 else 0.0
                max_err = max(max_err, err)
                max_rel_err = max(max_rel_err, rel)
                matched += 1
                if i < 10:
                    print(f"  G={key}  ours={vcoul[j]:.8e}  BGW={vcoul_ref[i]:.8e}  err={err:.2e}  rel={rel:.2e}")

        print(f"\nMatched {matched}/{len(vcoul_ref)} G-vectors")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Max relative error: {max_rel_err:.6e}")

    except FileNotFoundError:
        print(f"\nNo BGW vcoul file '{args.vcoul}' found; printing first 20 values:")
        for i in range(min(20, len(vcoul))):
            print(f"  G={tuple(gvecs_all[i])}  v={vcoul[i]:.8e}")


if __name__ == "__main__":
    raise SystemExit(main())
