# DESIGN — two-level QSGW self-consistency: an inner loop at frozen screening, an outer loop that refits W

Owner's goal (2026-09-03): a large stability improvement in QSGW self-consistency. This
note is the design for the existing loop (`src/gw/sc_iteration.py`), not for the
fixed-index rewrite (`feat/sc-loop-2026-09-03`, closed negative, claims 647–658).

## 1. The diagnosis the design answers

The fixed-index study measured the self-consistent map directly (report
`reports/sc_loop_2026-09-03`, "Pass 2: pole-store discriminator"): between two
consecutive iterations the Si Γ bottom pair moved 0.027 meV while its raw on-shell
Σ_c moved 46.8 meV under GN-PPM (amplification 1765) and 0.498 meV against 176.6 meV
under MPA (355). Every iteration recomputes χ0 → W → the pole store from rotated
wavefunctions, and the pole fit is not continuous in its inputs: GN-PPM modes cross
the Ω² < 0 threshold and switch to the static-limit branch (`ppm_invalid_mode`),
the MPA Loewner pole set jumps. A map with a local Lipschitz constant of a thousand
has no attractive fixed point; rCROP kills components pointing away from a fixed
point, which is why it converged the earlier loops that did not refit a
threshold-sensitive pole model every iteration, and why nothing converges now.
Lane SCDIAG (`runs/DEV/108_sc_map_sensitivity_2026-09-03`) is measuring the
amplification stage by stage; this design is written to its expected outcome and is
adjusted to its actual one before the fix lane starts.

## 2. Structure

**Inner loop (QSGW0-like): frozen screening.** After the outer step builds W and the
pole store once, the inner loop iterates only G: rotate the wavefunctions with the
current U, rebuild Σ_c(ω) from the FROZEN store (the executor's windows and quadrature
rules are then fixed too, which lane SCFIX provides), form the QSGW Hamiltonian,
diagonalize, rCROP on H in the DFT basis, until `sc_tol_ev`. The map E → Σ inside this
loop is smooth (Σ_c(ω) is a fixed table read at moving energies through a fixed
Hermitian rotation), so the fixed point is attractive where the physics has one.

**Outer loop: refit W.** With the inner loop converged, recompute χ0 → W → store from
the converged wavefunctions and energies, then run the inner loop again from the
previous H; converge when the outer change of the QP energies over the protected
states is below `sc_tol_ev` (or a looser `sc_outer_tol_ev` if the owner wants one;
default equal). Outer iterations are few (QSGW literature: W changes little after
the second refit). rCROP history is reset at each outer refit; the outer variable may
be linearly mixed (`sc_mixing`) if the diagnostic shows the W refit itself is not
contractive.

**Update law.** Owner's proposal (2026-09-03): every off-diagonal Σ̃_mn (m ≠ n) at E_F,
the diagonal Σ̃_mm at E_m (on-shell, eqp0-type, no Z anywhere in the iteration). The
off-diagonal block is then a static Hermitian matrix that changes only with the
basis rotation; the diagonal is the only energy-dependent piece. Window membership
questions disappear for off-diagonals; the diagonal reads the table at clip(E_m)
if E_m leaves the grid, printed as a receipt.

**What does not change.** The existing partition/scissor machinery stays as it is on
main (it is what kept the earlier loops out of the steep regions); the frozen rules
(lane SCFIX) and the BGW-style eqp1 output land independently.

## 3. The threshold-mode question (decided by SCDIAG)

If the census shows GN-PPM modes crossing Ω² = 0 between OUTER iterations carry
enough residue mass to move Σ by tens of meV, the outer loop will still see jumps.
Then the continuous treatment of near-threshold modes becomes part of this design:
a smooth blend between the pole and the static-limit branches across a small window
in Ω² (or a smooth lower clamp on Ω²), which is a physics-numerics choice the owner
rules on with the measured mass in hand. Nothing here relaxes a gate.

## 4. Acceptance

Si 4×4×4 (GN-PPM and MPA) and MoS2 3×3 GN-PPM at ±10 and ±15 eV, P4: inner loops
converge monotonically after two maps; the outer loop converges in ≤ 5 refits; the
converged QP energies agree between windows within the protected states to ≤ 10 meV;
iteration 1 remains bit-exact to one-shot; the frozen-W control from SCDIAG is
reproduced. Per-iteration history plots and eqp0/eqp1 tables as in the SC study.
