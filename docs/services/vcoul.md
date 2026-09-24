# vcoul — the bare and truncated Coulomb interaction

`services/vcoul/` owns `v(q+G)`: the three truncation kernels, the one per-q
evaluation driver, the q→0 cell averages (mini-BZ sampling, the exact slab
Wigner–Seitz cubature, the 3D body head) and the BerkeleyGW `vcoul` table
reader. It is standalone: numpy and jax, optionally scipy, nothing from
LORRAX. Deck keys, loaders and `Meta` stay on the LORRAX side; the service
speaks `CoulombGeometry`, an explicit `kgrid` and an explicit `sys_dim`.
Import top-level names only (`tests/test_layering.py` fails a submodule
import from outside).

Physics owner for the head and cell averages:
[`theory/four-current-head-corrections.md`](../theory/four-current-head-corrections.md);
their wiring into the four-current pipeline:
[`architecture/four_current_wiring.md`](../architecture/four_current_wiring.md).

## Kernels

All kernels return Rydberg values with BerkeleyGW's volume convention, so
the downstream `ζ v ζ†` needs no further factor. With $K = q+G$ in Cartesian
1/bohr:

| `sys_dim` | kernel | $v(K)$ |
|---|---|---|
| 3 (default) | `Bulk3D` | $\dfrac{8\pi}{\Omega\,\lvert K\rvert^2}$ |
| 2 | `Slab2D` | $\dfrac{8\pi}{\Omega\,\lvert K\rvert^2}\left[1 - e^{-z_c \lvert K_\parallel\rvert}\cos(K_z z_c)\right]$, $z_c = \pi / b_{3z}$ (Ismail-Beigi) |
| 0 | `Box0D` | Wigner–Seitz-truncated cell box (`compute_vcoul_box`, BGW `trunc_cell_box.f90`) |

* $K = 0$ is zeroed by `Bulk3D` and `Slab2D`: the q→0 head is a separate
  rank-1 term. In `Box0D` the truncated $v(G=0)$ is finite and is the head.
* `Box0D` is Γ-only and refuses $q \neq 0$: a finite-q request means the
  k-grid is wrong.
* `Slab2D` requires $b_1, b_2$ in the Cartesian xy plane and $b_3$ along
  $+z$; a tilted or reversed slab refuses.
* `get_kernel(sys_dim)` accepts `SysDim` or 0/2/3 (`None` means 3) and
  refuses anything else.

## API

| name | contract |
|---|---|
| `CoulombGeometry(bvec, cell_volume, bdot=None, fft_grid=None)` | Frozen. `bvec` rows are the Cartesian reciprocal vectors in 1/bohr with `blat` folded in; `bdot` and `fft_grid` are read only by `Box0D`. `CoulombGeometry.from_wfn(wfn)` forms `wfn.blat * wfn.bvec`; Coulomb consumers build geometry through it rather than multiplying by hand. |
| `v_qG_table(kernel, q_irr_frac, gvec_components, *, geometry, vcoul_cutoff_ry=None, v_head_fn=None, head_tie_rtol=1e-9)` | The one production `v(q+G)` driver. `gvec_components` is `(n_q, 3, ngkmax)` Miller indices; returns float64 `(n_q, ngkmax)`. Order: bare kernel, then head injection, then the cutoff mask, so a head slot outside the bare-Coulomb cutoff is zeroed like any other G. Pad slots are not zeroed (ζ̃ = 0 there). |
| `head_slot_table(...)` | The same arguments and predicate as `v_qG_table`, reporting the selected head slots without applying them. |
| `v_qG_single(kernel, geometry, qvec_wrapped, comps_qG)` | One q through `v_qG_table`, complex128 `(nG,)`. |
| `build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume, *, nmc=2**18, seed=42)` | The 3D body head: a function `K (m,3) → (m,)` for `v_head_fn`. The defaults are pins the frozen BSE reference depends on. |
| `build_miniBZ_dq_cart(kgrid, bvec, *, nmc=2**18, seed=42)` | `(2·nmc, 3)` mini-BZ offsets: an `nmc` draw unioned with its negation (centrosymmetric by construction). |
| `minibz_frac_to_cart(U, bvec)`, `minibz_cell_affine(bvec, kgrid)` | Fractional → Cartesian as `U @ bvec` (rows; never `U @ bvec.T`), and the full-cell → mini-BZ affine. |
| `minibz_voronoi_batches`, `sample_minibz_qpoints`, `minibz_average`, `minibz_inscribed_sphere_r2`, `minibz_moment_tensor`, `wrap_points_to_voronoi` | The BGW `minibzaverage.f90` port and its tensor-weight (`⟨v q_a q_b⟩`) sibling. `minibz_average` returns **bare** kernel units (no 1/Ω); each caller applies its own volume convention. `wrap_points_to_voronoi` is the one jitted Voronoi fold. |
| `slab_minibz_photon_cubature(kernel, geometry, kgrid) -> SlabMinibzPhotonReceipt` | The exact slab Wigner–Seitz polygon rule (Γ-to-edge Duffy triangulation, fixed 16/24/32 Gauss–Legendre ladder): the one slab q→0 cell-average owner, consumed by `Slab2D.q0_average` and by `gw.head_correction.complete_static_slab_photon_q0`. |
| `Bulk3D.q0_average`, `Slab2D.q0_average`, `*.q0_average_transverse_tensor` | q→0 cell averages `(⟨v⟩, ⟨v/(1 − v qᵀSq)⟩)` in bare units, and the bare transverse (TT) head, § [q→0 cell averages](#q0-cell-averages). |
| `gauss_legendre_interval(order, left, right)` | Immutable float64 Gauss–Legendre rules on finite intervals, shared by the photon cubature and LORRAX's MPA/VNL consumers. |
| `bare_coulomb_sphere_indices`, `bare_coulomb_sphere_mask`, `fft_box_miller` | Which G satisfy `|q+G|² ≤ cutoff` (Ry), and nothing else. The ζ-format sentinel padding is applied on the LORRAX side (`common.coulomb_sphere`). |
| `compute_vcoul_box`, `N_IN_BOX`, `NCELL`, `TRUNC_SHIFT` | The 0-D box FFT and BGW's parameters. |
| `read_bgw_vcoul`, `fill_v_grid_for_q`, `BGWVcoulTable` | Parse and scatter BGW `vcoul` dumps. `find_q_index` has no shifted-q₀ fallback. |

## Head-slot rule

With `v_head_fn` given, at every q except Γ (`min |q+G|² < 1e-12` is skipped)
the head is injected at **every** slot attaining

$$\min_G \lvert q+G\rvert^2 \quad\text{within relative window } \texttt{head\_tie\_rtol} = 10^{-9},$$

and every tied slot receives the **mean** of `v_head_fn` over the tied set,
each member valued at its own Cartesian $K$.

* **Why the argmin.** The G-list pairing between $+q$ and $-q$ is exactly
  $K \mapsto -K$, and $\lvert K\rvert$ is invariant under it, so the argmin
  set at $+q$ maps onto the argmin set at $-q$. A Miller-index label such as
  $G = (0,0,0)$ is not equivariant: the BGW wrap sends a boundary component
  to $+\tfrac12$, never $-\tfrac12$.
* **Why all slots.** On an even k-grid the argmin is degenerate at
  self-paired q ($-q \equiv q$) and the pairing swaps the tied slots, so no
  single-slot choice gives $V_q = \overline{V_{-q}}$.
* **Why the mean.** Per-slot values satisfy $q \to -q$ but not $K \to RK$
  for R in the little group, because the parallelepiped mini-BZ has lower
  symmetry than the crystal; the IBZ-plus-unfold route needs both.
* **Caller obligation:** `v_head_fn` must satisfy $f(K) = f(-K)$. For a
  Monte-Carlo average that requires a centrosymmetric δq set, which
  `build_miniBZ_dq_cart` guarantees.
* A non-callable `v_head_fn` (a per-q `(nkx, nky, nkz)` table) raises
  `TypeError` naming `build_v_head_miniBZ_fn_3d`.

## q→0 cell averages

`Slab2D.q0_average(geometry, kgrid, *, S_cart=None, epshead=None, rule=Q0_RULE_EXACT, ..., certificate_fn=None)`
takes a named rule with no silent alternative:

| `rule` | what it is | status |
|---|---|---|
| `Q0_RULE_EXACT = "wigner_seitz_polygon"` | the exact Wigner–Seitz polygon cubature, the same receipt the packed bispinor Γ completion reduces | default, production |
| `Q0_RULE_SOBOL_DEBUG = "sobol_debug"` | scrambled-Sobol Voronoi draw, carrying a ~0.1–0.2 % sampling error on the `|q|` cusp | debug only; announces itself |

* Under the production rule `nsamples`, `method` and `qmc_reps` select
  nothing. `analytic_sphere=True` (deck key `head_minibz_average`) refuses
  (`GATE slab_q0_analytic_sphere_unavailable`): there is no
  Baldereschi–Tosatti sphere in 2D, where the head is a `|q|` cusp. Unset the
  key on a `sys_dim = 2` deck.
* The 24→32 ladder pair must converge under `atol 1e-12`, `rtol 1e-8`
  (mixed error ratio ≤ 1), else `GATE slab_q0_polygon_not_converged`
  refuses. The certificate (edges, orders, node counts, `⟨v⟩`, error ratio)
  prints once per process and is returned as `SlabQ0Certificate` through
  `certificate_fn`; LORRAX reports it in `gwjax.out` as `Slab WS cert`.
* `static_kappa2` (the 3D Thomas–Fermi model) raises `NotImplementedError`
  on the slab.
* `Slab2D.q0_average_transverse_tensor` (the bare TT head used by
  `gw.v_q_bispinor`'s `bare_transverse` route) still uses the Sobol draw and
  carries its cusp sampling error; the packed route takes its TT head from
  the exact receipt.

`Bulk3D.q0_average` takes no `rule`: the polygon construction is 2-D, so the
3D head keeps the scrambled-Sobol draw plus the Baldereschi–Tosatti analytic
sphere, every dial live. `method="sobol"` without `scipy.stats.qmc` raises
`RuntimeError` naming the fix; `method="auto"` falls back to a uniform draw
with one `warnings.warn` (the results are not bit-comparable with a Sobol
run).

## Backends and cost

Host-side numpy plus one jitted jax helper (`wrap_points_to_voronoi`, CPU or
GPU); no `.so`, no mesh, no process count. The only capability axis is scipy
for the Sobol generator. `v_qG_table` runs once per consumer setup
(milliseconds at Si-fixture shapes) and the 3D head draw once per run
(seconds at `nmc = 2**18`, doubled by centrosymmetrization); neither is on the
GPU steady state.

## Tests

`services/vcoul/tests` (markers `services`, `vcoul`; deselect with
`--no-services`):

* `test_vcoul_import_isolation.py` runs the public surface in a `python -S`
  subprocess with only the service on the path, including the no-scipy
  refusal, with a red twin.
* `test_vcoul_door_smoke.py` covers every kernel, both refusal classes, the
  head-slot rule (argmin vs label, tied-set mean, Γ skip, table refusal) and
  the head-before-cutoff order on cubic and hexagonal cells.
* `test_vcoul_minibz_consolidation.py` pins golden body-head values, a
  composition test with a red twin, and closure of the δq set under negation.
* `test_vcoul_head_slot_reciprocity.py` rebuilds the Si per-q G-lists,
  checks the $K \mapsto -K$ pairing, and requires
  `max |v(+q,i) − v(−q,pair(i))| == 0.0`; red twins reinstate the Miller-label
  rule and a one-sided draw.

`tests/test_vcoul_minibz_head_draw.py` in the main suite guards the draw
convention with hexagonal-cell discrimination; silicon cannot see that bug
class.

## Antipatterns

* **Hand-rolling a fractional-to-Cartesian draw.** `randvals @ bvec.T` has the
  right volume and the wrong shape on non-cubic cells. Use
  `minibz_frac_to_cart` or `minibz_voronoi_batches`.
* **Multiplying `wfn.blat * wfn.bvec` at a Coulomb call site.** Use
  `CoulombGeometry.from_wfn`.
* **Testing a Coulomb formula against the formula itself.** Pin values against
  the closed metric-form expression.
* **Cubic-only tests for the mini-BZ family.** Cubic cells satisfy
  `bvec.T = P·bvec` and are blind to the draw-convention class; add a
  hexagonal or lower-symmetry row.
* **Monte-Carlo q→0 averages where an exact rule exists.** A seeded draw's
  error is deterministic and does not look like noise; a second draw is not a
  convergence study. A new q→0 average names its rule and prints a
  certificate that can refuse.
