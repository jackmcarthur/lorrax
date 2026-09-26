# Compact ordinary pseudopotential coupling

`psp.vnl_ops.compact_vnl_coupling(setup)` builds a `CompactVNLCoupling` pytree
from the canonical `ChannelMeta.E` blocks and the authenticated coupled row
blocks. Each channel stores its SOC matrix `E[s, t, R, R]` once, plus one
static contiguous row span covering its atoms. The action batches over atoms
at the channel's exact projector width, with no padding inside a coupled block
and no `total_R × total_R` operand; every within-block spin and
radial-projector off-diagonal is kept. Non-contiguous channel rows or an
inconsistent SOC block shape raise `ValueError`.

`setup_H_k_from_kvec(..., compact_vnl=True)` places this pytree in
`HamiltonianK.vnl_E`; pass it
unchanged to the Hamiltonian applications. The default (`compact_vnl=False`)
is the dense array, for consumers that coerce or inspect `vnl_E` as an array,
including Sternheimer orchestration. `apply_projector_coupling` is the one
owner of the compact/dense dispatch, and `projector_coupling_diagonal` gives
the matching spin-summed preconditioner diagonal.

**Cost.** Compact coupling removes the dense structural-zero work from each
coupling application. It does not reduce setup peak memory: `VNLSetup` still
builds the dense matrix, and projector storage and the Z/ψ projections are
unchanged. The number of compiled coupling groups equals the number of
channels, not atoms. For small systems one dense GEMM can be faster than the
batched blocks, so measure the crossover there.

The ionic structure-factor scan (`psp.ionic_gspace`) guards each atom's phase
evaluation with `lax.cond`, so padded atoms evaluate no full-G exponential;
the scan stays reverse-mode differentiable and active atom counts stay runtime
operands.

`tests/test_vnl_compact.py` covers complex `nspinor = 1` and `2` operator and
full-Hamiltonian parity, preconditioner-diagonal parity, inactive-NaN
positions and reverse-mode gradients.

## NSCF use

`psp.run_nscf.run_nscf` sets `compact_vnl=True` for its planned local
[Davidson](davidson.md) stage: one plan (capacity
`DEFAULT_M_MAX_FACTOR · nbnd = 10 · nbnd`) compiled for the common G carrier
and reused with explicit Hamiltonian data, 100 iterations, stall patience 20.
Non-convergence is reported before export.

`apply_H_k_batched(..., vector_batch=32)` applies the sparse-G Hamiltonian in
fixed blocks of 32 vectors plus an exact-width tail, which bounds FFT
production; the initial plane-wave guess uses the same operation. Host setup
selects natural G-sphere shapes, so every rank stages k-point rounds in the
same order (compile agreement), keeps only its own k-point, and then solves
concurrently. Setup is therefore duplicated across ranks, and no all-k device
Hamiltonian or guess cache is kept. Global result assembly and WFN export keep
their own memory costs.
