# Tests

## test_gw_jax_regression.py

End-to-end static COHSEX calculation on a small MoS2 system (WFNsmall.h5,
40 bands, 4 IBZ k-points, 60 ISDF centroids). Compares sigSX, sigCOH,
sigTOT, and VH against a reference validated to <100 meV MAE vs BerkeleyGW
sigma_hp (nband=80, same WFN). Tolerance: atol=1e-6 (bitwise reproducibility).

Run: `ISDF_COHSEX_TEST_PLATFORM=cpu uv run -- python -m pytest -q tests/test_gw_jax_regression.py -m regression`

## active/test_reshard_all_to_all.py

Validates that ISDF tensor resharding (mu_XY -> r_XY layout) uses JAX
all-to-all collectives, which is required for multi-GPU scaling. Spawns a
subprocess with 4 simulated CPU devices and inspects the XLA compilation log.
