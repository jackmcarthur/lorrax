# Compact ordinary pseudopotential coupling

`psp.vnl_ops.compact_vnl_coupling(setup)` derives a JAX pytree from the
canonical `ChannelMeta.E` blocks and authenticated `coupled_row_blocks`.
Each channel stores its SOC matrix once, plus static contiguous row spans for its atoms.
The action batches over atoms with that channel's exact projector width;
there is no padding inside the coupled block and no total_R-square matrix
operand. All within-block spin and radial-projector off-diagonals survive.

`setup_H_k(..., compact_vnl=True)` and
`setup_H_k_from_kvec(..., compact_vnl=True)` put this representation in the
existing `HamiltonianK.vnl_E` field. Pass the pytree unchanged to ordinary
Hamiltonian applications. The default remains a dense array for consumers
that explicitly coerce or inspect `vnl_E` as an array (including current
Sternheimer orchestration). Compact/dense dispatch belongs to the shared
`apply_projector_coupling` owner; there is one Hamiltonian physics path.
`projector_coupling_diagonal` provides the matching preconditioner diagonal.

This rollout removes dense structural-zero work from repeated coupling
applications. `VNLSetup` still constructs its dense compatibility matrix,
including for SOC-mode measurement; setup peak memory is therefore not
reduced. The full Z projector storage and Z/psi projections are unchanged.
The number of compiled coupling groups follows the number of channels,
not the atom count. Benchmark the crossover for small systems; batched
block operations are not guaranteed to beat one dense GEMM there.

The ionic structure-factor scan now guards each atom's phase evaluation
with `lax.cond`. Padded atoms execute no full-G exponential. Its fixed scan
preserves reverse-mode differentiation, and active atom counts remain
runtime operands.

Focused contracts are in `tests/test_vnl_compact.py`: complex spin1/spin2
operator and full-Hamiltonian parity, preconditioner diagonal parity,
inactive-NaN positions, and reverse-mode gradients. Numerical/performance
evidence belongs in the sandbox run report, with the exact checkout and
job identified there.

## NSCF rollout

`run_nscf` selects compact coupling for its planned local Davidson stage.
One plan is compiled for the common G carrier and reused with explicit
Hamiltonian data. The legacy default subspace capacity (10 times the band
count), 100-iteration budget, and 20-iteration stall patience are retained; failure to converge is reported
before export. Growing-shape warmup calls are removed.

`apply_H_k_batched` bounds FFT production to 32 vectors, using the existing
sparse-G Hamiltonian owner, with an exact remainder block. The initial
plane-wave guess uses this same bounded operation. Host setup still selects
natural G-sphere shapes, so ranks stage k-point rounds in identical order
under compile agreement, retaining their assigned inputs; owned solves then
run concurrently. This duplicates setup across ranks. It does not retain an
all-k device Hamiltonian/guess cache. Existing global result assembly and
WFN export remain in place and retain their existing memory costs.
