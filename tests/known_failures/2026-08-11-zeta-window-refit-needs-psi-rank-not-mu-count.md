# The 2x-centroid GW re-run does NOT open the ζ-window refit, and the reason is that the bound everyone has been quoting is the wrong bound (2026-08-11)

**STATUS: MEASUREMENT COMPLETE, STOPPED AT THE THIRD WALL AS INSTRUCTED. NO
`.dat`, NO `.png`, NO TOLERANCE MOVED, NO `LORRAX_FH_ORTHO_TOL` SET.**
Base `origin/main` `72945497`. Branch `lane/mu2x-zeta-window-2026-08-11`
(one additive commit, §5). Workspace `/pscratch/sd/j/jackm/mu2x_0811/`, sole
writer; `xbdense_0810/parentfull` (the production μ960 bundle), `downfold_s1`,
`triangle_0810`, `refitshard_0811`, `xbwin_0811` read-only and untouched.

The owner's preserved alternative to the refuted BSE-window ζ' route
(22.95 meV at (0,¼,¼), `2026-08-11-refit-vq-sharded-fetch-and-cert-grades.md`
§4) was to re-run the Si 4×4×4 60-band production GW deck on ~2x centroids so
that `--vq-mode refit --refit-window zeta` — the DEFAULT window, certified by
the machine-precision tile null rather than by a contracted meV grade — becomes
reachable. It does not become reachable, and this row is why.

## 1. The bound in the docs is necessary and NOT sufficient

`2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md` and the
`exciton_bands` module docstring both state the condition as

> `--vq-mode refit` needs **`n_μ,parent · n_s ≥ nk · nb_ζ`** — the parent ISDF
> basis must be at least the Galerkin rank bound of the ζ-fit window.

On this deck that reads 3840, and the production parent sits at exactly
960 × 2 = 1920, so "double the centroids" looks like it clears it by
construction. **It is a COUNTING bound on the carried extent. The gate is on
the numerical RANK of ψ sampled at those centroids**, and
`streaming_galerkin_solve` prints that rank on every run:

| n_μ (orbit-floored) | n_μ·n_s | vs 3840 | rank(ψ@centroids) | `max\|C Cᴴ − I\|` | on-grid ε error the driver derives | verdict |
|---|---|---|---|---|---|---|
| **960** (production) | 1920 | ±0 | — | **1.568e-03** | 14.1 meV | REFUSED (2026-08-10 row) |
| **1940** | 3880 | +40 (+1.0 %) | 3515 | **1.589e-05** | 0.143 meV | REFUSED |
| **2012** | 4024 | +184 (+4.8 %) | 3548 | **1.751e-05** | 0.158 meV | REFUSED |
| **2604** (`--prune-window vc_x_vc`) | 5208 | +1368 | 3689 | **4.959e-06** | 0.045 meV | REFUSED |
| **2628** | 5256 | +1416 (+36.9 %) | **3706** | **3.467e-06** | 0.031 meV | REFUSED |
| **2632** (`--oversample 3.0`) | 5264 | +1424 | 3594 | **1.495e-05** | 0.135 meV | REFUSED |
| **2988** | 5976 | +2136 (+55.6 %) | **3735** | **3.602e-06** | 0.032 meV | REFUSED |
| **3396** | 6792 | +2952 (+76.9 %) | — | — | — | **PARENT NOT BUILDABLE**, §4 |

Cap is 1.0e-06 and it was not touched. Every row is a real four-process leg —
`jax.device_count()=4 process_count()=4`, `mesh_xy.shape={'x':2,'y':2}`, all six
on the same tree HEAD `f614af35`, verified per leg from the run's own startup
block.

**Read the rank column, not the extent column.** At n_μ = 2988 the basis
carries 5976 columns against a 3840-state window — 56 % more extent than the
counting bound asks for — and still spans only 3735 of the 3840 states. The
capacity rule the code prints says it plainly: `nb < rank(ψ_μ)/nk`, which is
**nb < 58.36 at n_μ = 2988** against a ζ-fit window of **nb = 60**.

## 2. The rank SATURATES, and that is the finding

Directions bought per added basis column, over consecutive rungs:

| segment | Δcolumns | Δrank | rank per column |
|---|---|---|---|
| 1940 → 2012 | +144 | +33 | 0.229 |
| 2012 → 2628 | +1232 | +158 | 0.128 |
| 2628 → 2988 | +720 | +29 | **0.040** |

and the orthonormality residual is FLAT across the last two rungs —
3.467e-06 at 2628, **3.602e-06 (slightly worse) at 2988**. Going from the
production 960 to 1940 bought two decades of ortho (1.57e-03 → 1.59e-05);
everything after that bought half a decade and then nothing. Extrapolating the
last measured segment, the 105 directions still missing at n_μ = 2988 would
need roughly **+2600 more columns, i.e. n_μ ≳ 4300** — 4.5x production — and
the slope is still falling, so that is a lower bound on the answer, not an
estimate of it.

The mechanism is not mysterious and the module already warns about it in the
same breath: the ζ-fit window is `nband = 60` of a 62-band WFN, and
*"640-scale centroids cannot orthonormalize high oscillatory bands (Gram error
→40 % for the top bands)"* (`bandstructure/htransform.py`). Centroids are
placed by a density-weighted k-means; the states they cannot resolve are the
top few, and those are the last directions the rank is missing.

**Two selection variants were tried and both are WORSE than the default**, so
this is not a prune-window or an over-sampling problem:

* `--prune-window vc_x_vc` (adds c×c pair densities — the one that *should*
  target high conduction bands) at n_μ = 2604 raises the pivoted-Cholesky
  point rank a lot (1551 of 2604 independent directions, 59.6 %, against 1154
  of 2628, 43.9 %, for the default) and yet gives a WORSE Galerkin rank (3689
  vs 3706) and a worse ortho (4.959e-06 vs 3.467e-06). **The pair-density rank
  the centroid generator reports and the ψ rank the Galerkin fit needs are
  different objects and they do not move together** — worth knowing before
  anyone tunes one to fix the other.
* `--oversample 3.0` at n_μ = 2632: rank 3594, ortho 1.495e-05 — four times
  worse than the default at the same size.

## 3. The ladder, since it had to be measured anyway

`centroid.kmeans_cli N --orbit` on this deck, requested → **delivered**
(orbit-closed, the floor spends less and never rounds up), all rank-gate PASS:

| requested | orbits | **delivered n_μ** | n_μ·n_s |
|---|---|---|---|
| 1925 | 47 | **1916** | 3832 — **below the bound, illegal** |
| 1950 / 1980 | 48 | **1940** | 3880 |
| 2020 | 49 | **2012** | 4024 |
| 2100 | 50 | **2060** | 4120 |
| 2250 | 54 | **2244** | 4488 |
| 2450 | 58 | **2436** | 4872 |
| 2650 | 61 | **2628** | 5256 |
| 3000 | 68 | **2988** | 5976 |
| 3400 | 79 | **3396** | 6792 |

The production parent is 960 in 23 orbits. **There is no rung at 1920**: the
ladder steps 1916 → 1940 across it, so "exactly the bound" is not even
reachable here and the first legal rung is 1940 (+40, +1.0 %). It refuses.

## 4. The parent stops being buildable before the rank arrives

`gw.gw_jax` at n_μ = 3396 on one node, four processes one GPU each,
**`RESOURCE_EXHAUSTED: Failed to allocate request for 33.00GiB` on device 0**
(A100-40GB, `memory_per_device_gb = 28`). The two obvious re-shapes are both
refused by contracts that are correct:

* `-G 4 -n 1` (one process over four devices, which is what the `lx` banner
  recommends for a memory-bound array) → `ValueError: mesh 2×2=4 !=
  jax.process_count()=1`. `gw_jax` wants one process per mesh device.
* `-N 2 -G 4 -n 8` → `RuntimeError: resolve_mesh: this run has 8 devices over
  8 process(es), and 8 is not a perfect square. Only square 2-D meshes are
  supported` — so the next shape up from 4 is **9 processes (3×3)**, i.e. three
  nodes, which this lane's 2-node allocation could not place.

So the ladder as a *buildable* object ends at n_μ = 2988 on 4 GPUs. Costs for
whoever continues: the parent GW itself is cheap and scales gently —
**54.4 s at n_μ = 2012, 70.0 s at 2604, 73.8 s at 2628, 82.7 s at 2988**
(four A100s, four processes, `LORRAX_FORCE_FULL_BZ=1`, `BFC@0.85`) — and each
bundle is 8–10 GB of `isdf_tensors` plus ~1.2 GB of ζ. The centroid runs are
~1–18 min on one GPU. **Nothing in this lane was expensive; the wall is
numerical, not budgetary.**

## 5. What was NOT the problem, and the one commit this lane carries

The `refit_vq` P>1 sharded-fetch fix and the two-grade certification from
`72945497` are both present and neither was reached — every leg died in
`build_fH_R`, upstream of `refit_prepare`, exactly where the 2026-08-10 row
said the third wall was. The wall did not move; only its stated *cause* did.

The branch carries **one additive commit** and it is not on the path of
anything measured here:
`exciton_bands`'s ζ-window tile null now runs at every distinct coarse tile the
Q PATH lands on, as well as at `refit_ongrid_null`'s own default sample. That
default is Γ plus the three coarse q FURTHEST from Γ, and `v(Q) ~ 1/|Q|²`
amplifies a ζ-fit error most at SMALL |Q| — so the default systematically
excludes the tiles where the gate is most informative, which is the same
four-point-sample defect §4 of the cert-grades row found on the windowed route
(0.858 meV on four corners, 22.952 meV once the segment interiors were
sampled). It only ADDS q to a gate; no bracket, tolerance or grade is touched.
**It has never executed** — no run on this deck has reached it.

## 6. What would change the answer

None of it is a tolerance, and `LORRAX_FH_ORTHO_TOL` was not set at any point.

* **The ζ-fit window is now only FOUR BANDS too wide, and that is new.** At
  n_μ = 2628 the printed capacity is `nb < 57.91`; at 2988, `nb < 58.36`. The
  deck asks for 60. The 2026-08-10 row named "a parent whose ζ-fit window is
  narrow enough that nk·nb_ζ fits the basis it has" as the other way out and
  it then meant a drastic narrowing; at 2.7x centroids it means **nband 60 → 56**
  (nk·nb_ζ = 3584 against a measured rank of 3706). That is an OWNER CALL about
  what the delivered curve is a curve of — the 2026-08-10 row says so
  explicitly — and this lane did not take it, because its dispatch fixed the
  60-band window and told it to stop here with the numbers.
* **More centroids, but the number is ≳ 4300 and not ~1920**, and past ~3400
  the parent needs a 3×3 mesh (three nodes) to build at all.
* **A centroid set selected against ψ rather than against pair densities.**
  Both prune windows tried here score the pivoted Cholesky on pair densities,
  and §2 shows that score is uncorrelated (here, anti-correlated) with the ψ
  rank the Galerkin fit consumes. Nothing in the generator currently selects on
  the quantity `build_fH_R` gates.
* A point-picked (`--no-orbit`) set at the same size carries far more
  independent directions than an orbit-floored one on this deck
  (`2026-08-11-qsign-recut-verdicts.md` §3: 171/185 vs 122/168) and is the
  obvious next probe — but it is the owner's floor-to-orbits interface and is
  not re-litigated here.

**The 0.01 meV reference certification the dispatch was aiming at is further
away than the ortho cap alone suggests.** The driver's own measured conversion
(on-grid `max|Δε| ≈ 9.0e3 × ortho`) puts the best configuration measured here,
n_μ = 2628, at **0.031 meV of htransform representation error before any
exchange refit happens at all** — three times the 0.01 meV gate. Closing the
ortho gate to 1e-6 is what brings that to 0.009 meV, so the cap and the
deliverable are the same requirement, which is the reason it must not be
widened.
