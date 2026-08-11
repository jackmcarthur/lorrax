# The sixth wall is the refit's r→G TRANSFORM, not a phase convention — and it is down at all 64 coarse q on both m-legs (2026-08-11)

Base `origin/main` `f09bec97`, branch `lane/sixth-wall-finite-q-2026-08-11`.
Workspace `/pscratch/sd/j/jackm/sixthwall_0811/`, sole writer. The two probed
parents (`zeta52_0811/dp2628n20` and `zsolve_0811/p2628n60z52`), the
`si_bse_debug` fixture, the 2628- and 2988-centroid tables and
`triangle_0810`'s `dipole.h5` were read and not touched.

**NO TOLERANCE MOVED.** The tile bracket is 5.0e-02, the fH ortho cap
1.0e-06, the reference cert grade 0.01 meV, `DEGENERACY_TOL_RY` 1.000 meV —
all untouched. `LORRAX_FH_ORTHO_TOL` and `LORRAX_FI_FSHOULDER_TOL` were never
set on any leg of this lane.

This row answers the object
`2026-08-11-two-window-contract-lands-and-the-sixth-wall-is-finite-q.md` §5
left: "inside `refit_vq`'s own finite-q machinery — the centroid winding
phase, the sphere selection, or `v_on_set`."

## 1. THE DISCRIMINATOR, RUN BEFORE ANY CODE WAS TOUCHED

The dispatch's first discriminator was |V| versus phase at one finite q, on
the reasoning that a total decorrelation with a perfect Γ is the fingerprint
of a phase-structure error. It is sharper to ask the question one level below
the tile: `run_gates` already certifies that the STORED ζ on the STORED sphere
with the producer's own kernel rebuilds the stored `V_qmunu` at 3.3e-14 for
all 64 q, so the stored ζ is a valid reference for the refit's own `zt` — and
the tile is QUADRATIC in `zt` while that comparison is LINEAR.

One leg, `dp2628n20`, one guard, `m_leg="stored"`, four processes, mesh 2×2.
For each q: the refit's own ζ′, transformed four ways, against the stored ζ on
the matched sphere columns, and the resulting tile against the stored tile.

| q | A `e^{−2πi q·s_μ}·FFT[ζ′]` (what the code did) | B `FFT[ζ′]` | **C `FFT[e^{−2πi q·r} ζ′]`** | D wind·C |
|---|---|---|---|---|
| **Γ** | 3.315e-06 | 3.315e-06 | **3.315e-06** | 3.315e-06 |
| (1,2,2) | 1.057e+00 | 1.629e+00 | **1.401e-06** | 1.578e+00 |
| (0,2,2) | 1.017e+00 | 1.756e+00 | **4.574e-06** | 1.659e+00 |
| (2,2,2) | 1.098e+00 | 1.421e+00 | **2.298e-06** | 1.406e+00 |
| (1,0,0) | 4.446e-01 | 7.963e-01 | **1.898e-06** | 8.325e-01 |

and the same four as TILES, against the stored `V_qmunu` slot:

| q | A (the gate's published number) | B | **C** | D |
|---|---|---|---|---|
| **Γ** | 3.695e-06 | 3.695e-06 | **3.695e-06** | 3.695e-06 |
| (1,2,2) | **1.165e+00** | 5.064e-01 | **1.959e-06** | 1.322e+00 |
| (0,2,2) | **1.135e+00** | 4.357e-01 | **7.910e-06** | 1.268e+00 |
| (2,2,2) | **1.156e+00** | 5.729e-01 | **3.983e-06** | 1.369e+00 |
| (1,0,0) | 5.619e-01 | 2.297e-01 | **2.858e-06** | 5.959e-01 |

The A column reproduces the published 1.165 / 1.135 / 1.156 to every digit,
which is what makes the C column a measurement rather than four new numbers.

**AND THE DISCRIMINATOR'S OWN ANSWER IS "NEITHER".** It is not a phase-structure
error: the MAGNITUDES were wrong too. |ζ′| alone read 0.20–0.57 against the
stored |ζ|, the per-row circular concentration of the ratio's phase (1.000 =
a pure per-μ phase) ran 0.04–0.93 rather than 1, and the implied per-μ phase
missed `e^{−2πi q·s_μ}`, its conjugate and its square by a median 0.8–2.8 rad.
A diagonal phase error cannot move |V|; this one moved it by 0.15–0.50 at the
tile. So the winding phase was not merely mis-signed — it was standing in for
something that is not a phase at all.

**The other two suspects are exonerated by the same leg**, and it is worth
saying so because both were live: `_sphere_millers(zx, qw)` returns EXACTLY
the producer's stored G set at every q probed (537/562/588/568/570 Millers,
**0 in the refit sphere and not the stored one, 0 the other way**), and
`rst["v_on_set"]` — which does go through the producer's own door
`gw.compute_vcoul.compute_v_q_per_G` — agrees with that kernel on the stored
sphere at **relF 0.000e+00**.

## 2. WHAT IT ACTUALLY WAS: A DIFFERENT TRANSFORM, NOT A DIFFERENT CONVENTION

The ζ writer is `common.wfn_transforms.accumulate_rchunk_to_gflat`, called
from `gw.isdf_fitting` with `qvec_frac=q`, and its per-q Bloch factor
multiplies ζ ON THE r GRID, before the FFT:

    ZG_μ(G) = Σ_r e^{−2πi q·r} ζ_μ(r) e^{−2πi G·r} = Σ_r e^{−2πi (q+G)·r} ζ_μ(r)

i.e. the transform is taken at **q+G**. `vq_interp.to_sphere` and `recon` are
the same statement in host numpy and have been for as long as they have
existed. `refit_vq` instead took the transform at **G** and multiplied the
result by a per-centroid constant `e^{−2πi q·s_μ}`. That substitution is exact
only for a ζ_μ that is a delta at s_μ, and ζ_μ is a cardinal interpolation
function with support across the whole cell — so the two differ in the
G-CHANNEL STRUCTURE. Both are exact no-ops at q = 0, in the strongest sense:
every slot of both phase factors is exactly `1.0 + 0.0j` there. **That is the
entire reason Γ was clean at 4.688e-06 while nothing else moved.**

The fix is the named `vq_interp.zeta_r_to_sphere_q` — the device twin of the
host `to_sphere` — so the frame now has ONE spelling in the module, and the
winding phase is gone from `refit_vq` entirely. `zx["rmu_frac"]` is still the
right object in the F-scheme, where the same factor is taken OUT of a stored ζ
to leave something smooth enough to interpolate: it is an approximation there
on purpose, and it was never an identity here.

## 3. THE TARGET, MEASURED AT ALL 64 COARSE q ON BOTH LEGS

`refit_ongrid_null`'s default population is Γ plus the three furthest coarse q;
the sixth-wall row measured seven. This claim is stronger, so it is measured
stronger: EVERY coarse q on the bundle's own grid, against the stored
`V_qmunu` slot the gate itself indexes, at the gate's own bracket, with no
tolerance passed in. `dp2628n20`, ONE guard, four processes, mesh 2×2,
Galerkin ζ-window residual 4.471e-15.

| arm | min | median | max (at) | inside 5.0e-02 |
|---|---|---|---|---|
| `m_leg="stored"` — interpolation removed | 1.872e-06 | 2.370e-06 | **8.999e-05** (1,3,2) | **64 of 64** |
| `m_leg="htransform"` — PRODUCTION | 1.822e-06 | 2.356e-06 | **1.504e-04** (2,1,3) | **64 of 64** |

The real interpolation costs a factor of 1.7 at the worst q and nothing at the
median, which is the two-window contract's own statement about itself.

## 4. THE RED TWIN, AND THE INSTRUMENT THAT CAUGHT AN INVALID ARM FIRST

The first base-tree arm printed the SAME 64 numbers as the fix arm to every
digit. It was invalid, and its own log said so in two adjacent lines:
`[inleg] git HEAD: f09bec97…` next to `[lx] source tree:
/pscratch/sd/j/jackm/sixthwall_0811/**tree**/src [LORRAX_CHECKOUT]`. `lx`
resolves the source tree from the AMBIENT `LORRAX_CHECKOUT` at `lx run` time;
a per-leg `env` in the batch manifest is applied inside the container, after
that choice is made. **A base arm therefore needs its own launcher, not its
own manifest row** — measurement-discipline rule 1, caught by the rule's own
instrument. Both probe wrappers now print the imported `bse.vq_interp.__file__`
and whether it carries `zeta_r_to_sphere_q`, so the two instruments cannot
disagree silently again.

## 5. THE LADDER MOVES: AT μ = 2988 THE TWO-WINDOW CONTRACT IS AFFORDABLE ON A DECOUPLED PARENT

`2026-08-11-two-window-contract-lands-and-the-sixth-wall-is-finite-q.md` §3
left a STOP: on the decoupled μ2628 parent the fH orthonormality gate refuses
at the FIRST guard (1.326e-06 against the 1.0e-06 cap), and its diagnosis was
that the lever is centroids, because the Galerkin rank deficit grows the
moment the window widens past nb = 52. That is now tested rather than
believed. A parent built at μ = 2988 with the SAME deck otherwise —
`zeta_nband = 52`, `nband = 60`, `zeta_rcond = 1e-10`, `restart_q_storage =
full`, `LORRAX_FORCE_FULL_BZ=1`, one variable changed and the diff shown in
the setup — 117 s of GW at four processes:

| parent | n_guard | nb_fh | rank / nk·nb_fh | ortho `max\|C Cᴴ−I\|` | cap 1.0e-06 | implied on-grid \|Δε\| |
|---|---|---|---|---|---|---|
| μ2628 (published) | 0 | 52 | 3327 / 3328 | 3.444e-07 | PASS | 3.10e-03 meV |
| μ2628 (published) | 1 | 53 | 3388 / 3392 | 1.326e-06 | **REFUSE** | 1.19e-02 meV |
| **μ2988 (this lane)** | **0** | 52 | **3328 / 3328** | **3.868e-07** | PASS | 3.48e-03 meV |
| **μ2988 (this lane)** | **1** | 53 | **3392 / 3392** | **8.417e-07** | **PASS** | **7.57e-03 meV** |

**The rank deficit is GONE at both guard counts** (3327/3328 → 3328/3328;
3388/3392 → 3392/3392), the ortho at one guard falls below the cap, and the
implied on-grid energy error stays inside the 0.01 meV reference grade. The
diagnosis was right and the lever is centroids. Γ tile null on that parent at
one guard: **6.914e-06** (`m_leg="stored"`) and **5.729e-06**
(`m_leg="htransform"`).

**And the parent is the same physics.** `eqp1.dat` over the 4v8c window
(bands 5–16, 1-indexed) at all eight deck k-points, 96 values, against the
production μ960/nb60 parent on the identical WFN (DFT energies agree at
**0.0000 meV**, which is the check that says the comparison is legal):

| comparison | median \|Δeqp\| | max \|Δeqp\| | direct gap |
|---|---|---|---|
| **μ2988 decoupled vs production μ960/nb60** | **15.46 meV** | **52.44 meV** | 1.14346 → 1.12327 eV |
| μ2628 decoupled vs the same (published) | 14.66 meV | 54.47 meV | 1.14346 → 1.12622 eV |
| **μ2988 decoupled vs μ2628 decoupled** | **0.68 meV** | **4.47 meV** | 1.12622 → 1.12327 eV |

i.e. the new parent sits where the centroid-only delta says it should, and the
extra 360 centroids move the answer by 0.68 meV — the μ ladder is converged
where the ortho gate needed it to move.

## 6. GATES

One combined P=4 pytest leg per tree, `-G 4 -n 1`, over the refit/xbands
suites, `test_f_shoulder_two_window`, `test_refit_vq_shard_p4`,
`test_refit_frame_convention`, `test_layering`, `test_env_grammar` and all
seven `services/*/tests`. Each leg prints its own `[lx] source tree:`, its own
`git HEAD` and its own `frame-fix present = …`, and the three agree:

| tree | source tree the leg imported | frame fix | result |
|---|---|---|---|
| branch `6abe0a78` | `sixthwall_0811/tree/src` | True | **15 failed / 1203 passed / 13 skipped / 1 xfailed** (399 s) |
| base `f09bec97` | `sixthwall_0811/basetree/src` | False | **15 failed / 1199 passed / 13 skipped / 1 xfailed** (590 s) |

**The failure NAME sets are identical — empty set-diff in BOTH directions —
and 1203 − 1199 = 4 is exactly this branch's new cells**
(`tests/test_refit_frame_convention.py`), which is also the collected-count
check. The fifteen are the same fifteen the zsolve and two-window lanes
reported (five `distrib_la` contract cells, two `symmetry_maps` emulated-mesh,
two `vcoul` import-isolation, one `wfn_loader`, four `zeta_loader` skip-honesty
and `test_loo_accuracy_vs_reference_thresholds`); both legs exit 1 because
pytest exits 1 when anything failed, which is the known-red accounting and not
a new refusal.

**`test_refit_vq_shard_p4` passes on the branch.** It drives the four-PROCESS
red twin, whose synthetic `zx` this lane extended with `rfrac` — the tile's
r→G transform now reads it, and the twin's `_Q_TILE` is deliberately non-zero,
so the new frame is exercised across processes on both parities of n_μ.

**Si deck at four processes: rc 0** (46 s).

**Default-deck A/B where the refit is NOT invoked** — the decoupled parent
through `--vq-mode ongrid`, which exercises `initialize_wfns` and
`compute_wfns_fi` and no refit at all, run on both trees with the base arm
provably on `basetree/src`:

| | data bytes | md5 (comments stripped) |
|---|---|---|
| branch | 346 | `0e41fb06ea7a2fbc` |
| base | 346 | `0e41fb06ea7a2fbc` |

**Data-identical**, and identical to the value the two-window lane published
for the same A/B. The only lines that differ are `# Generated by …` and
`# input:`, which name each arm's own timestamp and workdir.

## 7. THE CURVE IS OWED, AND IT IS OWED FOR A RESOURCE REASON, NOT A PHYSICS ONE

The dispatch's stage (c) — the reference-grade certified curve rendered to a
PNG — is NOT delivered by this lane, and the reason is worth stating precisely
because it is not the gate.

**The driver's own gate passes on the new parent.** `exciton_bands --vq-mode
refit --cert-grade reference --q-per-segment 16 --refit-guard-bands 1` on
`xb2988` ran `refit_ongrid_null` over the SAME 7-coarse-q population the
sixth-wall row measured at 1.292 / 1.409 and read **worst 8.944e-06 against
the 5.0e-02 bracket** — so the driver proceeded to the off-grid path, which is
the first time it has ever been allowed to.

Two things then stopped it, in order:

1. **A GPU OOM after 60 off-grid Q**, `RESOURCE_EXHAUSTED … 15.41GiB` inside
   `refit_vq`'s `cq_and_x` — under the `platform` allocator the modulefile
   ships, which the startup banner itself flags as not the campaign default
   (`SMALL_ISSUES` row 48). One bounded relaunch with
   `XLA_PYTHON_CLIENT_ALLOCATOR=default` was made; the override is visible on
   the leg's own `lx run` line.
2. **The relaunch never got a node.** A peer agent's one-GPU legs occupy all
   four nodes of the shared pool (designed co-tenancy), and a `-G 4` leg needs
   a whole one. It has been queued 47 minutes with `--wait 14400` and will run
   if a node drains inside the pool's remaining walltime.

**No number in this row depends on that leg**, and no tolerance was touched to
get around it. What is owed is the `.dat`, the PNG, and the certified per-Q
number — not the tile null, not the ortho ladder, not the Δ-eqp.

## 8. EVIDENCE

`/pscratch/sd/j/jackm/sixthwall_0811/EVIDENCE.md`.
