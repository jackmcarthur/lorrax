# Self-consistent update on the IBZ — design scaffold

Status: SCAFFOLD, not implemented. Written to be handed to an implementation
team. Read `docs/architecture/decisions.md` first.

Proposal being scaffolded (owner, 2026-08-04): extract Σ_mnk to k_irr, build
and diagonalize H_mnk on k_irr, apply the resulting U_mnk to ψ_mk(r_μ) on
k_irr, and unfold back to full k for the next iteration.

The proposal is sound. **On the current decks it saves almost nothing, and
the little it saves is the reduction the owner has vetoed.** §1 is that
measurement; §2 is the constructive fix that makes the whole thing worth
building; §3–§6 are the actual scaffold.

---

## 1. What the current decks actually permit — measured, not assumed

`mos2_4x4` charge (`b600_p64/run_deck/WFN.h5`) and bispinor
(`bispinor_zeta_reuse/run_bi4/WFN.h5`), read from the files:

| field | value |
|---|---|
| `symmetry/ntran` | **2** |
| `kpoints/nrk` (IBZ k in file) | **10** |
| `kpoints/kgrid` | 4 × 4 × **1** ⇒ 16 full-BZ k |
| `kpoints/nspinor` | 2 |
| `kpoints/w` | 4 × 0.0625 + 6 × 0.125 |

The two stored ops are {E, σ_h}. **σ_h maps z → −z, so on a k_z = 0 grid it
acts trivially on every k.** The stored spatial group therefore reduces
16 → 16: *zero* k-reduction.

The 16 → 10 in the file is time reversal, and the weights prove it: the four
weight-1/16 points are Γ and the three M points — exactly the momenta with
k ≡ −k (mod G) — and the six weight-2/16 points are ±k pairs.
4·(1/16) + 6·(2/16) = 1.

**Consequences.**

1. An IBZ self-consistent update on these decks is a **1.6× saving at best**
   (10/16), not the order-of-magnitude the proposal implies.
2. That 1.6× is **entirely time-reversal-based**, and
   `decisions.md` carries the standing veto *"No time-reversal-symmetry-based
   reductions"*. Building the SC update on the IBZ as the files define it is
   building on the vetoed reduction.
3. **This is an owner question, not an implementer's.** The veto and the file
   format currently disagree: `nrk=10` means the WFN *only stores* 10 k, so
   LORRAX already unfolds with TRS (`symmetry_maps.unfold_psi` has a full
   TRS branch, `allow_trs`, and an independent measurement of whether TRS
   holds via `common.density_symmetry_check`). Either the veto is narrower
   than its wording (e.g. it forbids halving the q-sum in Σ, not reading a
   TRS-reduced WFN), or these decks violate it. **Resolve this before any
   implementation** — it decides whether §2 is a nice-to-have or the only
   legal route.

## 2. The fix that makes this worth building: recover the C3 spinor rotations

`centroid/orbit_syms.py` already documents why the group is too small:

> a non-collinear SOC MoS₂ WFN stores only `ntran=2` = {E, σ_h}: pw2bgw drops
> the C3 rotations (they would need a spinor rotation the BGW `mtrx` format
> can't carry), even though the crystal — and the charge density — are fully
> D3h symmetric.

So the physical group is D3h and the file carries {E, σ_h}. Recovering C3
takes 16 k down to **4** irreducible (Γ, K, M, and one general point), a
genuine **4×** — and it is *spatial*, so it is unaffected by the TRS veto.

Both halves already exist and have never been connected:

| need | exists | note |
|---|---|---|
| find the true point group | `orbit_syms.recover_symmorphic_density_point_group(avec, rho)` | recovers it from the **charge density**, which is the physically correct group (full crystal group for non-magnetic, magnetic point group otherwise) |
| SU(2) for an arbitrary rotation | `SymMaps.get_spinor_rotations(wfn, sym_matrices_cart)` | quaternion (Markley/Shepperd); takes Cartesian matrices as an **argument**, so it is not limited to the stored list |
| unfold ψ with them | `symmetry_maps.unfold_psi` | already takes `sym_mats_k`, `translations`, `U_spinor_spatial` |

**The task is to join them**: recover the group from ρ → convert to Cartesian
(`syms_crystal_to_cartesian`) → `get_spinor_rotations` → extend `sym_mats_k`
and `U_spinor_spatial` → re-derive the IBZ.

Three hazards on that path, all real:

* **SU(2) sign (double group).** ±U represent the same SO(3) rotation. For a
  single unfold k̄ → Sk̄ a global sign is a gauge phase and cancels in
  ⟨m|O|n⟩. It does **not** cancel when two different group elements reach the
  same k, or in the little-group consistency check of §4. Fix the branch once,
  by composition (`U(S₁)U(S₂) = ±U(S₁S₂)`), and gate it.
* **`recover_symmorphic_density_point_group` is τ = 0 only** — it
  "conservatively omits" non-symmorphic ops by construction. Safe for MoS₂
  (symmorphic); a non-symmorphic deck will silently get a smaller group and
  a larger IBZ, which is slow but not wrong.
* **The recovered group must be validated against ψ, not just ρ.** ρ is
  spin-traced. A magnetic or SOC-split system can have a ρ more symmetric
  than ψ. `validate_kgrid_unfolding` exists; extend it to the recovered ops
  and make it a hard gate before the group is used to reduce k.

## 3. Why unfolding the MATRICES is the easy half

For a symmetry op S with ψ at the star point **defined as** the unfold,
ψ_{n,Sk̄} := 𝒰(S) ψ_{n,k̄}, any operator O that commutes with S satisfies

    ⟨m, Sk̄| O |n, Sk̄⟩ = ⟨m, k̄| O |n, k̄⟩

**exactly** — no rotation matrix, no phase. Σ_mn and H_mn are literally
k-independent across a star. Unfolding them is a **broadcast over the star**,
i.e. an index map, not arithmetic.

This is worth stating loudly because it is the cheap part and it is easy to
over-engineer. There is no D(S) Σ D(S)† to apply.

**It holds only under a precondition that must be gated:** the full-BZ ψ must
BE the unfolded IBZ ψ, not independently obtained orbitals at those k. If any
path ever produces full-BZ ψ another way (a QE nscf on the full grid, a
re-diagonalization, a re-phasing), the gauge differs and Σ_mn(Sk̄) ≠
Σ_mn(k̄) off-diagonally — while the diagonal still looks fine, so cheap
checks miss it. **Gate: unfold ψ from k̄, and independently take the full-BZ
ψ the loader gives, and require agreement.**

Degeneracy needs no special handling here *because* of this definition: the
arbitrary basis choice inside a degenerate manifold is made once at k̄ and
carried to the whole star by the unfold.

## 4. The hard half: ψ at the ISDF centroids does NOT unfold by permutation

This is the design's real obstacle and the reason the proposal cannot be
implemented as literally stated.

The proposal's step 3 keeps ψ_mk(r_μ) — ψ sampled at the ISDF centroids — and
rotates it in band space by U from the k̄ diagonalization. **That part is
clean**: a band rotation is a local GEMM over the band axis at fixed r_μ,
touches no symmetry, and shards exactly like the matrix-element sweep.

**Unfolding across the star is not clean.** A symmetry op acts on the
argument: ψ_{Sk̄}(r) = ψ_k̄(S⁻¹(r − τ)). At a centroid r_μ, the point
S⁻¹(r_μ − τ) is in general **not another centroid**, so there is no
permutation of the μ index that implements the unfold. `orbit_syms.py` states
the same fact from the quadrature side:

> A raw centroid sum is only point-group symmetric across a k-star if the
> centroid set {r_μ} is closed under the point group.

Three routes, and the recommendation:

| route | cost | verdict |
|---|---|---|
| **A. orbit-closed centroids** — close {r_μ} under the group; the unfold becomes a permutation + phase | ISDF fit quality changes; μ grows to a multiple of the orbit size | **RECOMMENDED.** `kmeans_cli --symmetry-adapted` already stores orbit representatives, and this is *already required* for V_H to be C3-symmetric across the star — so it is not new debt, it is a dependency the V_H quadrature has anyway |
| **B. unfold in G space, re-evaluate at centroids** | one FFT + gather per unfolded k, exactly the transform the new k-scan owns | Correct and needs no centroid change, but it re-pays the FFT the IBZ reduction was meant to save. Sensible fallback when centroids are not orbit-closed |
| **C. never unfold ψ at centroids** — keep centroid ψ on k_irr and unfold only the objects that need full k | least code | Fails: χ⁰/W need full-BZ ρ_mn(k, r_μ) |

**A with B as the gated fallback.** Route A's precondition — orbit-closed
centroids — is checkable at runtime, so the code can pick.

## 5. Proposed shape

Nothing here is implemented. Names are suggestions.

```
psp/ or common/            KStar                     the index structure
    star_of(k_irr)      -> [(k_full, sym_idx, umklapp_G), ...]
    broadcast(A_irr)    -> A_full           # §3: index map, no arithmetic
    reduce(A_full)      -> A_irr            # weighted average over the star,
                                            #   AND the consistency residual
```

`reduce` returning the **residual** as well as the average is the load-bearing
choice. Σ arrives on full k (it is built from an FFT over full k). Reducing it
to k_irr must be a *no-op up to round-off* if the symmetry is exact. The
spread across a star is therefore a free, continuous measurement of every
assumption in §2–§4 — wrong SU(2) sign, non-orbit-closed centroids, a gauge
mismatch in the full-BZ ψ. **Report it every iteration; gate on it.** It is
the single most informative number this design can produce, and it costs one
reduction.

Iteration:

```
Σ_full  --reduce-->  Σ_irr  (+ residual, gated)
H_irr = T + V_ion + V_H + Σ_irr - Vxc          # on k_irr only
eps_irr, U_irr = eigh(H_irr)                   # k_irr only
psi'_irr(r_mu) = einsum('kmn,knu->kmu', U_irr, psi_irr(r_mu))
psi'_full = unfold(psi'_irr)                   # route A or B of §4
```

## 6. Order of work

1. **Owner ruling on §1.3 (the TRS veto vs `nrk=10`).** Blocking: it decides
   whether the IBZ is 10 or 16 before C3 recovery.
2. `KStar` + `broadcast`/`reduce` with the **residual gate**, using the
   symmetry the files already carry. Measures ~1 (no reduction) on these
   decks and that is the point: it proves the machinery on a trivial group
   before the group gets interesting.
3. Join `recover_symmorphic_density_point_group` → `get_spinor_rotations` →
   `unfold_psi` (§2), with the SU(2) sign gate and the ψ-validation gate.
   **This is where the 4× is.**
4. Orbit-closed centroids as a hard precondition for route A (§4), with route
   B as the measured fallback.
5. Only then move the H build and diagonalization to k_irr. It is the *last*
   step, not the first — it is worthless until 3 lands and unsafe until the
   §5 residual is green.

## 7. What this does NOT change

Σ is still built by FFT over full k. This proposal reduces the *update*
(build H, diagonalize, rotate ψ), not the Σ construction. On these decks the
diagonalization is not the bottleneck, so **quantify the expected saving
against a profile before implementing** — the arithmetic in §1 says it is
1.6× of a step that may be a small fraction of the iteration.
