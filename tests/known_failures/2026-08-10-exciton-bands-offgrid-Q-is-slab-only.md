# AMENDMENT — AN EXCITON BANDSTRUCTURE ON A BULK CRYSTAL COULD NOT BE DRAWN AS A BANDSTRUCTURE, AND THE REASON WAS ONE COULOMB KERNEL (2026-08-10)

The owner, looking at the Γ–X–W–L–Γ–Σ exciton band plots this campaign
delivered: *"the exciton band fine grid plot looks like shit. i wanted like 16
diagonalizations on each line segment. i thought that was obvious. it's a
bandstructure."*

He is right about the plot and right that it should have been obvious. This
row is about why nobody had drawn one, because the answer is not that the
previous lane was lazy — **its eight points were the maximum the deck could
express**, and the wall it hit is a real and until now unwritten limit on
`bse.exciton_bands`.

## The arithmetic nobody had done

`--vq-mode ongrid` refuses any path Q that is not on the **exchange-tile
grid** — the grid the GW run wrote `V_qmunu` on, which on this lineage
(`downfold_s1/parent936` → `densify_exp_2026-08-10/child191`) is 4×4×4, i.e.
64 momenta in the whole zone. Densifying with `bse_k_grid = 8 8 8` does not
help and the driver says so: densification moves the k-SUM, it does not create
exchange tiles at new q.

Take the path corners in crystal coordinates — Γ (0,0,0), X (0,½,½),
W (¼,¾,½), L (½,½,½), Σ (¼,½,¼) — and ask how many interior points of each
segment land on multiples of ¼:

| segment | direction | on-grid interior points | intervals |
|---|---|---|---|
| Γ→X | t·(0,½,½) | t = ½ | **2** |
| X→W | Δ = (¼,¼,0) | none | **1** |
| W→L | Δ = (¼,−¼,0) | none | **1** |
| L→Γ | t·(½,½,½) | t = ½ | **2** |
| Γ→Σ | t·(¼,½,¼) | none | **1** |

`2,1,1,2,1` — which is, to the digit, the `K_POINTS` block the campaign decks
carry. Eight diagonalisations is not an oversight; it is the whole of what
this deck's exchange tensor can answer exactly. Sixteen per segment needs
**off-grid** exchange, and a 6×6×6 parent would only take those five numbers
to `3,1,1,3,1` — the on-grid route does not reach a bandstructure on any grid
anyone will run.

## So why was off-grid exchange not available?

`--vq-mode interp` exists for exactly this, and it refused:

```
ValueError: vq_interp is a SLAB model and this deck is not a slab
  (1) the restart's Coulomb-policy stamp says sys_dim=3 ...
  (2) the cell's axes are not slab-separable ... reach 1.000e+00 ...
  (3) the coarse q-grid is not planar: max|q_z| = 5.000e-01 ...
```

All three are "this is a 3-D crystal". `vq_interp`'s b26p long-range model
fits in `|G_z|` channels that only exist on a slab, and every `v(q+G)` in the
module is the 2-D Ismail-Beigi truncation. **On a bulk deck this driver had no
arbitrary-Q exchange at all**, and therefore no exciton bandstructure — only
an exciton line drawing between corners.

Two things made that hard to see, and both are now fixed on the branch:

* the refusal ended by recommending `--vq-mode refit`, **which used the same
  slab kernel** (`refit_vq` → `v_slab_on_set`) and which the driver refused
  anyway with `--vq-mode=refit alone is not wired`. The advice was a dead end
  that read like a workaround;
* a bulk deck reaching `interp` first fails `require_zeta_for_interp`, so on
  an IBZ-ζ lineage you get a ζ refusal, fix the ζ, and only then meet the real
  wall. This lane paid that in full: it re-ran the 960-μ parent GW under
  `LORRAX_FORCE_FULL_BZ=1` (45 s) and re-downfolded (25 s) to get a full-BZ
  transported ζ — necessary, and not sufficient. **The ζ is not the blocker
  and a lane sent to fix ζ will not reach a plot.**

## The blocker is one Coulomb kernel — MEASURED

Rebuilding the bundle's own `V_qmunu` from its own stored ζ at all 64 coarse
q, under three kernels
(`/pscratch/sd/j/jackm/xbdense_0810/probe_v3d2.py`, 2026-08-10):

| kernel | makeVq-vs-disk relative (min / med / max) | under the 5e-6 gate |
|---|---|---|
| `v_slab_on_set` — what the module uses | 5.8e-3 / 3.8e-2 / 5.1e-1 | 0 of 64 |
| bulk 8π/K²/Ω, **no** mini-BZ head | 1.4e-14 / 9.2e-3 / 4.8e-2 | 1 of 64 |
| `gw.compute_vcoul.compute_v_q_per_G` + `build_v_head_miniBZ_fn_3d` | 3.7e-15 / 9.2e-15 / **3.3e-14** | **64 of 64** |

The middle row is the one `vq_interp`'s own docstring predicted and left open
("the remainder attributable to the deck's own `mc_average_vcoul_body = true`
mini-BZ head-slot injection, which this module does not model either"). The
third row closes it by not modelling anything: those two functions **are** the
ones the V_q writer called, so the refit and the producer cannot disagree
about a convention — there is only one implementation of it. Note which row is
load-bearing: the mini-BZ head slot moves this by six orders of magnitude and
the deck's 25 Ry `bare_coulomb_cutoff` moves it by nothing (the ζ sphere is
already inside it), so a lane that ported the cutoff and skipped the head
would have got 9.2e-3 and no idea why.

The refit is the right home for this and the interpolation model is not: the
refit fits ζ at the target Q from the htransform ψ and contracts it with `v`,
with **no fitted model anywhere in the path**, so nothing in it is 2-D except
that one call. The b26p model cannot be rescued the same way — its channels
are the second, unfixable, slab-only fact.

## What the branch does, and what is still owed

`feat/xbands-dense-path-2026-08-10`:

* `make_v_on_set` hands the refit the producer's own kernel on a `sys_dim = 3`
  deck and leaves `v_slab_on_set` untouched everywhere else, so no slab result
  anywhere can move;
* the slab scope assert moves from `load_zeta_coarse` to
  `build_vq_evaluator` — the model build — because both slab-only facts are
  properties of the model and the refit builds none. An `interp`/`both` run on
  a bulk deck still refuses exactly as loudly, one call later, still before
  anything expensive;
* `--vq-mode refit` stops refusing and refits **every** path Q;
* `refit_ongrid_null` refuses the whole run unless the refit reproduces the
  stored `V_qmunu` at coarse q first. Certify where consumed: every off-grid
  tile comes from that same call, so if it cannot reproduce a tile someone
  else wrote down, nothing downstream is worth plotting;
* `--q-per-segment` (default 16) makes a dense path the default sampling.

Two integration seams were found on the way and are handled by name, not by
assert: a **downfolded** bundle needs `B_at_mu` sliced to the kept centroid
rows (the htransform leg fits in the PARENT basis), and the deck's band window
must be the GW's **ζ-fit** window, because the refit re-fits ζ from that
window's pair densities — a narrower BSE window gives a different ζ and the
on-grid null would correctly refuse it.

## AND A THIRD WALL, WHICH IS THE ONE THAT STOPS THIS DECK

The refit ran.  It placed at P=4 across two nodes (`lx run -N 2 -G=2 -n=4` —
the geometry that fits when two long-lived one-GPU co-tenants are pinning a
GPU on each of your own two nodes; worth knowing) and refused inside the
htransform:

```
ValueError: build_fH_R: the Galerkin coefficients are NOT orthonormal —
max|C Cᴴ − I| = 1.568e-03 over all k, above the 1.0e-06 cap
```

That is `docs/drivers.md`'s already-documented rank bound, arriving from a new
direction.  Chain the three requirements and they close on each other:

* the refit re-fits ζ at the target Q **from the pair densities of the ζ-fit
  window**, so the htransform window has to BE that window — here `nb = 60`,
  the parent GW's own `nband`;
* the htransform Galerkin leg needs a basis spanning `nk·nb` = 64 × 60 =
  **3840**;
* the basis it fits in is the parent ISDF one, `n_μ · n_s` = 960 × 2 =
  **1920**.

1920 < 3840, by exactly a factor of two, so the projection cannot be
orthonormal and the driver refuses rather than silently returning energies
wrong by the 14.1 meV it prints.  The campaign's earlier 20-band runs cleared
this because 64 × 20 = 1280 ≤ 1920 — but a 20-band htransform is not the ζ-fit
window, so **the window that makes the refit meaningful is exactly the window
that breaks its htransform leg**.  On this lineage the refit is unreachable,
and no flag reaches it.

The condition to state, since it is written nowhere else:

> `--vq-mode refit` needs **`n_μ,parent · n_s ≥ nk · nb_ζ`** — the parent ISDF
> basis must be at least the Galerkin rank bound of the ζ-fit window.  On a
> 4×4×4 SOC deck with a 60-band GW that is ≳ 1920 centroids; this lineage has
> 960.

Two ways out, both a GW re-run, neither free: a parent fitted with a large
enough centroid set (the fleet already has 1104- and 1128-centroid Si bundles,
e.g. `bandwin666_0810/armF`), or a parent whose ζ-fit window is narrow enough
that `nk·nb_ζ` fits the basis it has.  Either changes the bundle, so neither is
a drop-in for the owner's `child191`, and **which one is right is an owner call
about what the delivered curve is a curve OF** — not something to decide inside
a plot-regeneration lane.

## OWED

**No exciton table has been produced through the refit path**, so there is no
refit exciton number to quote and none appears here.  The kernel table above is
the only physics measurement in this row, and it is an on-grid TILE comparison,
not an energy.  The `[refit-null]` gate has never executed under the driver —
the run refused upstream of it, in the htransform.

The `--q-per-segment` half of the lane IS verified: **17 of 17 cells green** on
a cluster leg (`/pscratch/sd/j/jackm/xbdense_0810/_logs/gates3.log`), of which
10 are that file's pre-existing cells and 7 are this lane's.

**Environment note, cheap, and it cost this lane two legs:** passing
`PYTHONPATH=<worktree>/src` to a containerised `lx run` leg REPLACES the
container's own path, and the leg then has no `h5py` — which surfaces as 13
unrelated test failures with nothing in them about paths.  On the cluster the
worktree is pinned by `LORRAX_CHECKOUT` and `PYTHONPATH` must be left alone.
That is the exact mirror of the WSL trap in `BUILD_NOTES.md`, where
`PYTHONPATH` must be set — so the two boxes want opposite things and neither
page said so.
