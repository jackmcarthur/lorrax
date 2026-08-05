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
2. `ntran = 2` is small for a D3h crystal, and the likely cause is the
   symmetry list allowed in the QE input that produced the deck — not the
   file format and not BGW, which carries spinor rotations fine.

   **This is not a reason to regenerate the deck, and the implementation must
   not assume anyone will.** The requirement is the opposite: the
   infrastructure takes the operation list as *input* and is correct for
   whatever it is handed — `ntran = 1` (no reduction, the degenerate case
   that must still be exercised), `ntran = 2` (this deck, 1.6×), or the 48
   operations of a cubic crystal (~24× on a 12×12×12 grid). Nothing in §1,
   §3 or §4 depends on the size or content of the list.

   `mos2_4x4` at `ntran = 2` is therefore the **test case**, not a problem to
   be routed around. A design that only shows a win at 48 operations and is
   untested at 2 is not general; a 1.6× deck exercises every code path
   (star construction, τ phases, spinor rotations, the ψ handoff) at a size
   where the answer can be checked by hand.

   `orbit_syms.recover_symmorphic_density_point_group` +
   `SymMaps.get_spinor_rotations` (which builds SU(2) for **arbitrary**
   Cartesian rotations passed as an argument) compose into an *optional*
   recovery path for a deck whose stored list is shorter than its crystal
   symmetry. Optional: the loop must be correct without it.

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

---

## 7. Status 2026-08-05 — wired, not verified; and the method was wrong

`KStarMap` (`common.symmetry_maps`) is an argument to the SC loop,
`config.sc_on_ibz` turns it on, and it builds correctly:
`KStarMap(nk_full=16, nk_irr=10, reduction=1.60x, n_sym_spatial=2)` on
MoS₂ 4×4. `star_select`/`star_broadcast` are gated at 1.19e-16 with a
negative control (job 7889237), and the TRS rule they encode —
`O(−k) = conj(O(k))`, not equality — was itself found by that gate
(assuming equality is off by 3.6e-01 relative, job 7889235).

**No end-to-end IBZ-vs-full-BZ agreement has been demonstrated.** Four
separate k-set mismatches were hit and fixed ONE TRACEBACK AT A TIME:

| # | site | operand |
|---|---|---|
| 1 | `sigma_dispatch.py:360` | `hartree_basis_rotation` — needed `U_full` |
| 2 | `qsgw_utils.py:449` | `e_qp_ev` — needed `E_full` |
| 3 | `qp_wfn.py:136` | the QP WFN writer wants **IBZ** (`wfn.nkpts`), so the blanket broadcast at the exit was wrong |
| 4 | `sc_iteration.py:1042` | `_rotate_to_dft_basis` AFTER the loop: `last_sigma_basis_U` is IBZ, `last_sigma_result` is full-BZ |

Finding them serially is the wrong method and is why this is not finished.
**Do the audit first**: enumerate every k-indexed operand that crosses
into or out of the loop and label its k-set, then fix them together. The
rule is simple enough to apply by inspection —

* everything `compute_sigma_xc` / `compute_screening` touch is FULL BZ,
  because Σ is an FFT over the k-grid;
* the carried state, `kin_ion`, the scissor operands and the eigh are IBZ;
* each CONSUMER states its own k-set — they do not share one, and #3 is
  the proof: `write_qp_wfn_h5` requires the IBZ because a WFN file stores
  the IBZ by BGW convention.

Band-only quantities (the `BandPartition` masks) need nothing.

---

## 8. Status 2026-08-05 — the writers are separated; the two arms do NOT agree

### 8.1 Each post-SC writer's own k-set, read from its own consumer

`dump_qp_wfn_artifacts` used to hand both its writers whatever k-set the
loop ran on, so it was wrong for one of them in both directions: with
`sc_on_ibz` on, the earlier blanket broadcast broke the WFN writer (§7
row 3); with `sc_on_ibz` OFF — the DEFAULT — the loop is on the full BZ
and the same writer died on any deck whose WFN stores a reduced k-set,

    ValueError: write_qp_wfn_h5: U shape (16, 128, 128) inconsistent with
    (nk=10, nb_active=128).

That killed the `fullbz` arm, which is the CONTROL for the IBZ path, so
no agreement number existed at all.

| writer | k-set | how that was determined |
|---|---|---|
| `write_qp_wfn_h5` | the WFN FILE's, `wfn.nkpts` (10 here) | shape check `qp_wfn.py:136`; the writer copies the source `kpoints`/`mtrx`/`tnp` through and rotates the ψ stored at those k, so `U` must be in the file's own gauge |
| `write_qp_rotations_h5` | FULL BZ (16) | `kpoints_crys` labels `U_mnk`'s rows, and the canonical writer of this same file passes `sym.unfolded_kpts` (`gw_output.py:865-875`); the consumer indexes `U_mnk` by full-BZ index (`postprocess/rotate_wfn_to_qp.py:159`). `write_results` rewrites the file later in the same run from the driver's own full-BZ eigh, so the SC-side copy must match it |
| `eqp0.dat` / `eqp1.dat` | full-BZ INPUT, IBZ OUTPUT | `gw_output.py:770-800` subsets full-BZ arrays with `kirr_to_kfull` and lists the IBZ wedge. Its arrays come from the driver's own eigh of `kin_ion + sigma_total` (`gw_jax.py:630`), both full BZ in both arms — the SC loop's k-set never reaches it, so it needed no change |
| `write_qp_wfn_oneshot` | the WFN file's | already SKIPS with a warning when `nk != wfn.nkpts` (`gw_output.py:368`). Same reduction would make it work; not done, out of scope |

`final_qp_eigenstates` lost its k-set argument: it returns the state's own
k-set and the placement is the caller's, because one argument cannot be
right for two consumers.

`KStarMap.select` is the correct reduction for the WFN writer only because
the row it takes — the first full-BZ member of each star — is the stored k
itself, reached by the IDENTITY operation. MEASURED on mos2_4x4, job
7889366: `kirr_fullids = [0,1,2,4,5,6,7,8,9,10]` strictly increasing,
`sym_idx_k[kirr_fullids] = 0` at all 10, `max|unfolded_kpts[kirr_fullids] −
wfn.kpoints| = 5.6e-17`, `select(broadcast(A)) − A = 0` exactly. Both
properties come from the grid enumeration order, not from a construction
that enforces them; re-measure on a deck with a larger group.

### 8.2 The agreement number: they do not agree, by 0.39 eV

Both arms now run to completion and write `eqp0.dat`, `eqp1.dat`,
`WFN_qp.h5` and `qp_wfn_rotations.h5` (jobs 7889366, 7889373, both arms
rc=0). The comparison is therefore possible for the first time, and it
FAILS.

Deck `mos2_4x4` (`run_800c_valsmoke_fftoff`), `qp_solver = self_consistent`,
`gn_ppm`, identical decks except `sc_on_ibz`:

| run | accelerator | eqp0 E_QP max\|Δ\| | rms | eqp1 max\|Δ\| |
|---|---|---|---|---|
| job 7889366 | rcrop, 2 iters (`converged=False`) | 3.926e-01 eV | 3.739e-02 eV | 3.417e-01 eV |
| job 7889373 | linear α=1, 3 iters | 3.861e-01 eV | 3.676e-02 eV | 3.355e-01 eV |

The rCROP row alone would not prove anything: rCROP mixes a flattened `H`
whose length differs between the arms (16·nb² vs 10·nb²), so its trial
steps are not the same function of the previous iterate and a truncated
rCROP comparison is path-dependent by construction. The `linear` α=1 row
removes that objection — step *n* is then a pure function of step *n−1* in
both arms — and the disagreement survives at the same size. The `E_DFT`
column is identical (max\|Δ\| = 0.000e+00) in both runs, so the k-lists,
band ordering and file layout match and the difference is entirely in the
QP column. `WFN_qp.h5` QP eigenvalues, both arms on the same 10 k, differ
by 1.970e-02 Ry = 2.68e-01 eV.

The disagreement is spread over all 128 active bands and peaks at bands
41-44 (E_DFT ≈ +5.0 to +5.5 eV) at 3.86e-01 eV; the smallest per-band
figures are ~3e-04 eV. Nothing agrees to round-off.

### 8.3 Bisected: Σ is exact, the CARRY is not

Repeat at `sc_max_iter = 1` — a single `gw_iteration_map`, no carry, no
accelerator (job 7889375, both arms rc=0). The same run produces two
numbers from two different points of the same iteration:

| quantity | built from | IBZ vs full BZ |
|---|---|---|
| `eqp0.dat`, `eqp1.dat` | the driver's own eigh of `kin_ion + sigma_total` — i.e. Σ, the k-star select, and the post-loop rotate-back | **0.000000e+00 eV**, bit-identical |
| `WFN_qp.h5` QP energies | `eigh(state_final.H_qp_dft)` — the CARRY, after the scissor refit and the band partition | 1.673e-02 Ry = 2.28e-01 eV |

So in one and the same iteration the Σ path is EXACT and the carry is off
by 0.23 eV. Everything §1 rests on is confirmed: the star-invariance
premise (`KStarMap.spread` on the real Σ+V_H prints 7.559e-12, 1.470e-12,
1.583e-12 relative, job 7889373), the `select`/`broadcast` pair, the ψ
rotation, the Σ build and the rotate-back all reproduce the full-BZ answer
to the last bit. The disagreement is entirely in the post-Σ assembly of
the next carry.

### 8.4 The cause: a REDUCTION over k needs star WEIGHTS

§7 labels each operand with a k-SET. That is not the whole rule. There is
a fourth class it does not name — a REDUCTION OVER k needs star WEIGHTS,
not just a k-set — and the carry contains exactly one such reduction.

Between the (exact) `H_qp_dft_full = kin_ion + delta_h_dft` and the carry
there are only two steps: `_scissor_E_qp_for_outofrange`
(`sc_iteration.py:660`) and `apply_band_partition`. The partition masks
are band-only, so by elimination the 1.673e-02 Ry is the scissor.
`fit_scissor` (`scissor.py:177`) is an unweighted least squares over every
`(k, n)` sample in the fit mask: the full-BZ arm fits 16 k, with 6 of the
10 stars entering twice, while the IBZ arm fits each star once. Different
α/β, hence a different scissor diagonal on the 98 of 128 bands that are
out of the ω-grid window on this deck (`SC partition: protected/in-range =
30/128 bands`, both arms). That diagonal enters the carry, and by
iteration 3 it has reached every band — which is why the 3-iteration
comparison in §8.2 shows a non-zero difference on all 128 bands with the
peak (3.86e-01 eV) at bands 41-44, E_DFT ≈ +5.0 to +5.5 eV.

Fix direction: pass star multiplicities (`np.bincount(sym.irr_idx_k)`) as
sample weights into `fit_scissor` when the loop runs reduced, so the fit
sees the same weighted point cloud on either k-set. Not attempted here.
