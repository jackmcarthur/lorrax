# Self-consistent update on the IBZ — design scaffold

Status: SCAFFOLD, not implemented. Written to be handed to an implementation
team. Read `docs/architecture/decisions.md` first.

Design (owner, 2026-08-04): take Σ_mnk on the symmetry-reduced k, build and
diagonalize H_mnk there, apply the resulting U to ψ_mk(r_μ) on those k, and
unfold the *modified wavefunctions* back to full k through the existing
ψ-unfolding path for the next iteration.

The design is correct as stated and this document scaffolds it. The target is
a community code covering all wavefunction symmetries — a cubic system with
48 operations reduces a 12×12×12 grid by ~24×, and that is the case the
scaffold is built for. The MoS₂ numbers below are a worked example of what
one deck permits, not an argument about whether the work is worth doing.

---

## 1. Nothing is unfolded except ψ

The proposal needs **no matrix unfolding**, and this document previously
over-built that point. Recording the correct version:

* Σ is computed on full k (it comes from an FFT over full k). Getting it onto
  the reduced set is an **index selection** — take the rows at the reduced k.
  Not a symmetry operation.
* H is built and diagonalized only on the reduced k.
* U from that diagonalization is applied to ψ on the reduced k. This is a band
  rotation at fixed r_μ: `einsum('kmn,knu->kmu', U, psi_mu)`. Local GEMM over
  the band axis, no symmetry involved, shards like any other band contraction.
* The rotated ψ is unfolded to full k by `symmetry_maps.unfold_psi`, which
  already exists and already handles the hard parts (non-symmorphic τ phases
  via `tau_phase_row`, spinor rotations, umklapp bookkeeping by the caller).

So the only symmetry operation in the loop is the one that is already
implemented and certified. There is no D(S)ΣD(S)† anywhere and no need for
one.

**One optional diagnostic is worth keeping.** Σ arrives on full k, so its
spread across a star must be zero up to round-off if the symmetry is exact.
Measuring that spread when the rows are selected costs one reduction and
continuously validates the symmetry setup — wrong SU(2) branch, centroid set
not closed under the group, gauge mismatch in the full-BZ ψ. Recommended as a
gate, not required by the algorithm.

## 2. What a deck permits — worked example, `mos2_4x4`

Read from `b600_p64/run_deck/WFN.h5` and `bispinor_zeta_reuse/run_bi4/WFN.h5`
(identical): `symmetry/ntran = 2`, `kpoints/nrk = 10`, `kgrid = 4×4×1`
(16 full-BZ k), `nspinor = 2`, weights `4 × 0.0625 + 6 × 0.125`.

The two stored ops are {E, σ_h}. σ_h maps z → −z, so on a k_z = 0 grid it acts
trivially on every k: the stored **spatial** group reduces 16 → 16. The 16 → 10
is time reversal, and the weights show it — the four weight-1/16 points are
exactly the k ≡ −k (mod G) momenta (Γ and the three M), the six weight-2/16
points are ±k pairs.

Two consequences, and only the second is an action item:

1. On this deck the reduction is 1.6×, and it is time-reversal-based. That is
   fine for the design — the machinery is the same whatever the group — but
   `decisions.md` carries a standing veto *"No time-reversal-symmetry-based
   reductions"*, while `nrk=10` means the WFN only **stores** 10 k, so LORRAX
   already unfolds with TRS (`unfold_psi` has a full TRS branch, `allow_trs`,
   and an independent density-based measurement of whether TRS holds at all).
   **Owner question, not an implementer's**: the veto and the file format
   currently disagree in wording. Most likely the veto is narrower than it
   reads (it forbids halving the q-sum in Σ, not reading a TRS-reduced WFN).
2. `ntran = 2` looks small for a D3h crystal. The likely cause is the symmetry
   list allowed in the original QE input, not any limitation of the file
   format or of BGW — **BGW carries spinor rotations fine**. If the C3
   operations are wanted, the fix is upstream in the QE input; failing that,
   `orbit_syms.recover_symmorphic_density_point_group` recovers a symmorphic
   group from the charge density and `SymMaps.get_spinor_rotations` builds
   SU(2) for **arbitrary** Cartesian rotations passed as an argument, so the
   two compose into a recovery path. Check the deck's QE input first.

## 3. ψ at the ISDF centroids

Applying U at the centroids is clean (§1). **Unfolding across a star at the
centroids is not**, and this is the one place the design needs a decision.

A symmetry op acts on the argument: ψ_{Sk̄}(r) = ψ_k̄(S⁻¹(r − τ)). At a
centroid r_μ the point S⁻¹(r_μ − τ) is in general **not another centroid**, so
no permutation of μ implements the unfold. `orbit_syms.py` states the same
fact from the quadrature side: a raw centroid sum is only point-group
symmetric across a k-star if {r_μ} is closed under the group.

**The existing centroid symmetry code handles fractional translations
correctly** — this document previously implied otherwise and that was wrong.
`orbit_syms` carries τ throughout: the stated convention is
`image_row(r,s) = r @ Rinv[s].T + tau[s] (mod 1)` (`orbit_syms.py:11`), τ is
read as `wfn.translations[:n_sym]/(2π)` (`:178`), `orbit_images` applies it
(`:229`), and `compute_centroid_sym_perm` shifts by τ and raises a named error
when τ lands off-grid (`:439`, `:453`, `:640`). The only τ = 0 routine is
`recover_symmorphic_density_point_group`, which is the optional *group
recovery* helper, not the orbit closure, and it omits non-symmorphic ops
conservatively rather than adding wrong ones.

Two routes:

| route | cost | verdict |
|---|---|---|
| **A. orbit-closed centroids** — unfold becomes a permutation + phase | μ grows to a multiple of the orbit size; ISDF fit changes | **Recommended.** `kmeans_cli --symmetry-adapted` already stores orbit representatives, and V_H already depends on this closure for C3 symmetry across a star — an existing dependency, not new debt |
| **B. unfold in G space, re-evaluate at centroids** | one FFT + gather per unfolded k | Correct, needs no centroid change, but re-pays the FFT the reduction saves. Gated fallback when centroids are not orbit-closed |

## 4. Proposed shape

```
KStar
    star_of(k_reduced)  -> [(k_full, sym_idx, umklapp_G), ...]
    select(A_full)      -> A_reduced      # index selection (Sigma, H); no arithmetic
    spread(A_full)      -> residual       # the §1 diagnostic, optional but advised
```

Iteration:

```
Sigma_red = select(Sigma_full)                      # + spread(), gated
H_red     = T + V_ion + V_H + Sigma_red - Vxc       # reduced k only
eps, U    = eigh(H_red)                             # reduced k only
psi'_red  = einsum('kmn,knu->kmu', U, psi_red)      # band rotation at r_mu
psi'_full = unfold_psi(psi'_red, ...)               # EXISTING path
```

## 5. Order of work

1. Owner ruling on §2.1 (TRS veto wording vs `nrk=10`).
2. `KStar` + `select` + the `spread` diagnostic, on whatever symmetry the deck
   carries. On `mos2_4x4` this exercises a near-trivial group, which is the
   right place to prove the machinery before the group gets interesting.
3. The band rotation and the handoff into `unfold_psi`. This is the whole
   algorithm and it is small.
4. Orbit-closed centroids as a hard precondition for route A (§3), with route
   B as the measured fallback.
5. A deck with a large group — cubic, 48 ops — as the scaling test. That is
   the case the design exists for; a 1.6× deck cannot demonstrate it.

## 6. Scope

This reduces the *update* (build H, diagonalize, rotate ψ), not the Σ
construction, which still runs on full k. Profile the update's share of an
iteration before predicting a speedup.
