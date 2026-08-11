# AMENDMENT — THE THIRD WALL IS GONE AND THE CURVE IS STILL NOT DRAWN, BECAUSE THE OBJECT BEHIND IT FAILED ITS OWN CERTIFICATION BY 772× (2026-08-11)

Amends `2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md`, whose third wall
read:

> `--vq-mode refit` needs **`n_μ,parent · n_s ≥ nk · nb_ζ`** … On this
> lineage the refit is unreachable, and no flag reaches it.

A flag reaches it now. `--refit-window=bse` fits ζ' on the deck's band window
instead of the producer's ζ-fit window, and on this lineage that turns the
Galerkin rank bound from 64·60 = 3840 (against a parent ISDF basis of
960·2 = 1920) into 64·20 = 1280, which the basis spans with room. The
htransform leg then runs and runs *well*: **ctilde orthonormality 8.882e-15**
against a 1e-6 cap, **Galerkin full-r residual 4.264e-15**, rank 1280 of 1280
with nothing discarded. The refit itself is cheap — **2.0–4.0 s per off-grid
Q** on this deck, so a 129-point path would cost about four minutes of
exchange.

So the wall the previous row named is really gone, and three more were behind
it. Two were plumbing and are fixed on the branch. The third is not plumbing,
and it is the reason there is still no plot.

## THE CERTIFICATION, AND WHY IT IS THE ONE THAT MATTERS

A windowed ζ' is **not** the producer's ζ, so it does not reproduce the stored
`V_qmunu` tiles and the tile-level `refit_ongrid_null` cannot be its gate. It
would be easy — and wrong — to widen that bracket until a windowed refit
"passed" a comparison it was never computing. So the gate moves **up one
level**, to the object the driver actually publishes:

> at every path Q that lands on the coarse exchange-tile grid, solve the BSE
> **twice in the same single-compile scan** — once through the refit exchange,
> once through the producer's own stored tile — with the conduction caches
> ψ_c(k+Q), ε_c(k+Q), W_R, the solver and the Lanczos block *literally the
> same arrays*. The eigenvalue difference is then attributable to the exchange
> and to nothing else, in the units the curve is published in.

Gate: **0.01 meV**, a module constant with no CLI flag and no environment
override, because a knob on that number is a knob for making a failed
certification pass.

## THE NUMBERS (job 56612363 step .62, `xbwin_0811/_logs/probe_p1.log`)

Si 4×4×4 SOC, `downfold_s1/parent936` → `xbdense_0810/child191f`, μ_S = 191,
BSE window 4v8c, refit window = absolute bands [0, 20).

| path Q | tile | max\|ΔE_S\| refit-route vs stored-route, 8 levels |
|---|---|---|
| X  (0, ½, ½) | (0,2,2) | **0.997 meV** |
| W  (¼, ¾, ½) | (3,1,2) | **1.345 meV** |
| L  (½, ½, ½) | (2,2,2) | **3.203 meV** |
| Σ  (¼, ½, ¼) | (3,2,3) | **7.719 meV** |

**Worst 7.719 meV against a 0.01 meV gate — 772× over.** The driver refused,
wrote no `.dat` and no `.png`, and the tolerance was not touched.

## WHAT THE NUMBER IS NOT

It is not the htransform. That is the first thing to suspect and it is
excluded by its own instruments in the same log: a Galerkin representation
good to 4.3e-15 and Galerkin coefficients orthonormal to 8.9e-15 cannot
produce a 7.7 meV error in anything downstream of it. It is not the Coulomb
kernel either — the previous row measured the producer's own door reproducing
the stored tiles from the stored ζ at 3.3e-14, 64 of 64.

It is not the solver: both rows of each comparison are the same scan, the same
compile, the same 200 iterations, and the α-Hermiticity invariant is 2.1e-13
across all ten Q.

And it is not a degeneracy artefact. Every window edge in this run is legal
under `--band-degeneracy strict` on this spectrum, with margins that are not
close: the refit window's top edge at band 20 has a min-over-k gap of
**228.3 meV**, the BSE conduction edge at 16 has **281.8 meV**, the BSE
valence edge at 4 has **48.0 meV**, and the VBM/CBM edge at 8 has
**2501.1 meV**. (Every ODD boundary on this deck is a Kramers pair at exactly
0.000 meV and is refused, which is the guard working.)

What is left is the thing the flag actually changes: **ζ' fitted on a 20-band
window is a different exchange operator on the BSE pair space than ζ fitted on
the producer's 60-band window** — even though every pair density the exchange
contracts lies inside the narrow window. The ISDF fit is a least-squares
problem over *all* pairs in its window, so narrowing the window does not
merely drop pairs the kernel never asks for; it re-weights the fit that
decides ζ' on the pairs it does ask for, at a fixed 191-centroid basis that
was selected against the parent's own retained window.

## THE WINDOW LADDER SAYS THE WINDOW IS NOT THE KNOB

The first thing to ask of a failure blamed on a window is what a wider window
buys. Same bundle, same BSE window (4v8c), same path, only `nband` moved:

| refit window | Galerkin bound nk·nb | worst max\|ΔE_S\| |
|---|---|---|
| nb = 20 (8v + 12c) | 1280 | **7.719 meV** |
| nb = 24 (8v + 16c) | 1536 | **6.812 meV** |
| nb = 28 (8v + 20c) | 1792 | **REFUSED** — `build_fH_R` ortho 1.593e-05 > 1e-6 (rank 1790 of 1792) |

Twenty per cent more window buys twelve per cent of the error, and two steps
up the ladder the parent basis stops spanning: at nb = 28 the Gram-eigh
returns rank 1790 where 1792 is needed and the driver refuses, which is the
THIRD WALL again, four bands earlier than the arithmetic alone would put it
(1792 ≤ 1920 on paper; 1790 in fact). So the reachable window on this bundle
is nb ≤ 24, and across the whole reachable range the certification is flat at
the 7 meV scale. **Widening the refit window is not a route to 0.01 meV
here.**

The error is also not an artefact of the mesh: the same configuration at a
real four-GPU 2×2 (`device_count=4 mesh_xy.shape={'x': 2, 'y': 2}`) gives
**6.819 meV** worst against P=1's 7.719 — same story, same refusal.

## THE SECOND SUSPECT, AND WHY IT IS NOT ENOUGH EITHER

This bundle is a **downfolded child** (μ 960 → 191), and its stored `V_qmunu`
is not a plain ISDF tile at 191 centroids: `downfold_provenance` records a
per-q rank truncation, **retained rank 173–178 of 191** at
**ε_w(V) = 4.2e-3 … 1.4e-2, median 1.0e-2**. So the certification's "reference"
is itself an approximation of a *different kind* from the refit's — a
~1 % rank-truncated tile against a full-rank re-fit — and that is an obvious
candidate for a meV-scale contracted difference, one that would also explain
why the window ladder is flat.

It does not survive contact with the per-q numbers:

| point | tile | q index | ε_w(V) | retained rank | max\|ΔE_S\| |
|---|---|---|---|---|---|
| X | (0,2,2) | 10 | 0.00436 | 177 | 0.997 meV |
| W | (3,1,2) | 54 | 0.01284 | 176 | 1.345 meV |
| L | (2,2,2) | 42 | 0.00470 | 175 | 3.203 meV |
| Σ | (3,2,3) | 59 | 0.01093 | 175 | 7.719 meV |

Pearson r(ε_w, ΔE_S) = **0.30**, and the two orderings disagree: X and L have
nearly the same truncation (0.0044, 0.0047) and differ by 3× in error, while W
carries the LARGEST truncation of the four and the second SMALLEST error. A
1 %-truncated reference is present and is presumably contributing, but it is
not what sets the scale or the q-dependence.

## THE CONTROL THAT SEPARATES THEM

`REPLACE_PARENT`

## WHAT THIS COSTS THE OWNER, AND THE CHOICE THAT IS HIS

The exciton band structure at 16 diagonalisations per segment is still owed,
and the honest statement of why is now one line instead of three walls: **on
this bundle, off-grid exchange is reachable but not yet trustworthy at the
0.01 meV the delivered curve is certified to.** The routes, in the order they
cost:

* **A wider refit window on this same bundle.** Free of any re-run, bounded by
  `nk·nb ≤ n_μ,parent·n_s` — at most nb = 28 here (bound 1792), against the
  producer's 60. The ladder above says what that buys.
* **The owner's preserved alternative: re-run the GW at ~2× centroids.**
  A parent with n_μ ≳ 1920 makes `nk·nb_ζ = 3840` spannable and the refit runs
  at the producer's own ζ-fit window — which is the configuration the TILE
  null certifies, at the 3.3e-14 the kernel already measures. That is the
  curve drawn from a higher-resolution parent rather than from a windowed
  refit, and it is a different physical claim, not a cheaper route to the same
  one. Which of the two the delivered curve should be a curve OF remains an
  owner call, exactly as the previous row said.

The `--q-per-segment` floor and the plotter are ready either way; nothing
downstream of the exchange is waiting on anything.

## THE TWO WALLS THAT WERE PLUMBING (fixed on the branch, both first-ever executions)

* `refit_prepare` streamed the window ψ with `jax.device_get` on an array
  `iter_psi_rchunk_bandwise` yields **mesh-sharded**, so at P=4/four-process it
  died with "Fetching value for a `jax.Array` that spans non-addressable
  devices". Now `_to_host` (this module's three-arm wrapper over
  `common.collectives.gather_to_host`) — and the three arms are why the naive
  repair is wrong: `process_allgather(tiled=True)` on the fully-addressable
  P=1 array would multiply the k axis by the process count.
* `--vq-mode=refit` never asked the loader for the exchange tensor it
  certifies **against**: `load_v_full` was keyed on `ongrid` alone, so a refit
  run walked all the way to its own gate and then refused itself for the
  bundle "carrying none".

Both were invisible until now for the same reason: the run had never reached
them. The `[refit-null]` gate the previous row recorded as never-executed has
still never executed — under `--refit-window=bse` it is not the gate, and it
now refuses a windowed refit state by name rather than being handed a wider
bracket.

## ENVIRONMENT

`LORRAX_WFN_BACKEND=eager` is required for this driver at P=4 on this deck:
the default backend segfaults inside HDF5 1.12.2 ("can't locate ID" /
"Unable to decrement reference count") streaming the WFN across four
processes. The previous lane had this on its fine arm only; it is not a
fine-grid fact, it is a multi-process-WFN-read fact.

## OWED

The two PNGs. The blocker is the number in the table above, and it is a
physics/representation number, not a scheduling one.
