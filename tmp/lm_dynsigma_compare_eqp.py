import sys
import numpy as np
from gw.eqp_bgw import read_bgw_eqp

ref_dir = sys.argv[1]
new_dir = sys.argv[2]

for name in ("eqp0.dat", "eqp1.dat"):
    k_r, edft_r, eqp_r, off_r = read_bgw_eqp(f"{ref_dir}/{name}")
    k_n, edft_n, eqp_n, off_n = read_bgw_eqp(f"{new_dir}/{name}")
    assert off_r == off_n, f"{name}: band_offset mismatch {off_r} vs {off_n}"
    assert k_r.shape == k_n.shape, f"{name}: k-point shape mismatch"
    assert np.allclose(k_r, k_n, atol=1e-9), f"{name}: k-points differ"
    d_edft = np.abs(edft_r - edft_n)
    d_eqp = np.abs(eqp_r - eqp_n)
    print(f"{name}: shape={eqp_r.shape} "
          f"max|dE_DFT|={np.nanmax(d_edft):.6e} eV "
          f"max|dE_QP|={np.nanmax(d_eqp):.6e} eV "
          f"mean|dE_QP|={np.nanmean(d_eqp):.6e} eV")

print("done")
