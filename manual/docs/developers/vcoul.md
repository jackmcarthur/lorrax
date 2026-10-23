# vcoul

`import vcoul` The bare and truncated Coulomb interaction $v(\bq+\bG)$, its $\bq\to0$ cell averages, and the long-range plane-wave channels.
{: .page-summary }

Every two-particle quantity in LORRAX meets the Coulomb interaction once, in the centroid basis: $V_\bq(\mu,\nu) = \sum_\bG \zeta^*_{\bq\mu}(\bG)\, v(\bq+\bG)\, \zeta_{\bq\nu}(\bG)$ ([ISDF §3.1](../theory/isdf.md#isdf-vq)). `vcoul` supplies $v(\bq+\bG)$ on each per-$\bq$ G sphere for bulk crystals ($8\pi/|\bq+\bG|^2$), slabs (the Ismail-Beigi truncation along $c$) and isolated systems (the Wigner–Seitz cell box), in Rydberg with the $1/\Omega$ factor already applied so that $V_\bq$ comes out in Ry. It also owns what the point kernel cannot give: the $\bq\to0$ head, which replaces the divergent or cusped $\bG$ slot by an average over the mini-Brillouin-zone cell of the Γ sample, $\bar v = \langle v(\bq)\rangle_{\mathrm{mBZ}}$ and $\bar W = \langle v/(1-v\,\bq^{\mathsf T}S\bq)\rangle_{\mathrm{mBZ}}$ ([q → 0 head §2](../theory/q0-head.md#minibz)), and the long-range channels that split $W$ into a smooth short-range part and a non-analytic long-range part ([Short- and long-range W](../theory/w-range-splitting.md#channels)). It sits between the ζ fit and the screening: V_q, the head correction, the BSE exchange head and the W densifiers all call it. A new driver that needs any Coulomb matrix element (a phonon-mediated vertex, a Hartree term on a deformed cell) builds its $v$ here rather than writing $8\pi/K^2$ again.
{: .lead }

<div class="glance" markdown>

Import
:   `import vcoul`; top-level names only. Importing a submodule (`vcoul.minibz`) from LORRAX code fails the layering lint.

Depends on
:   numpy, jax; scipy optional (the scrambled-Sobol generator). Nothing from LORRAX.

Devices
:   Host numpy. `wrap_points_to_voronoi` is one jitted JAX kernel (CPU or GPU). No mesh, no sharding, no process count.

Used by
:   `gw.v_q_g_flat`, `gw.v_q_bispinor`, `gw.head_correction`, `gw.w_isdf`, `gw.lr_channels`, `bse.vq_interp`, `bse.bse_densify` ([below](#used-by))

</div>

## Conventions {: #conventions }

| Quantity | Convention |
|---|---|
| `bvec` | `(3, 3)` float64, **rows** are the Cartesian reciprocal vectors $\mathbf b_1,\mathbf b_2,\mathbf b_3$ in bohr⁻¹ with the lattice constant folded in (`blat * bvec` in WFN.h5 terms). |
| $\bK = \bq+\bG$ | Cartesian bohr⁻¹. From crystal components $n$: `K = n @ bvec` (equivalently `bvec.T @ n` for a column). Never `n @ bvec.T`. |
| `q_irr_frac`, `q_frac` | `(n_q, 3)` float, crystal (fractional) coordinates. |
| `gvec_components` | `(n_q, 3, ngkmax)` int Miller indices, the `isdf_header/gvec_components` layout of the [zeta_q.h5](../drivers/gw/files.md#zeta) file. |
| Cutoffs (`vcoul_cutoff_ry`) | Ry: a G is inside when $|\bq+\bG|^2 \le$ cutoff with $|\bK|^2$ in bohr⁻². |
| Kernel values (`v_qG_table`, `v_qG`) | Ry, BerkeleyGW volume convention: $8\pi/(|\bK|^2\Omega)$. Zero at $\bK = 0$ (bulk, slab). |
| Cell averages, long-range channels | **bare** units, Ry·bohr³ (no $1/\Omega$). The caller divides. `Box0D.q0_average` is the exception: it returns the $1/\Omega$ value. |
| Mini-BZ draws | Cartesian bohr⁻¹ offsets $\delta\bq$ inside the Voronoi cell of the lattice $\mathbf b_i / n_{k,i}$. |

## Geometry and dispatch {: #dispatch }

### CoulombGeometry {: #geometry }

```python
CoulombGeometry(bvec, cell_volume, bdot=None, fft_grid=None)    # frozen dataclass
CoulombGeometry.from_wfn(wfn) -> CoulombGeometry
geometry.fact                                                   # 1 / cell_volume
```

| Field | Shape, units | Read by |
|---|---|---|
| `bvec` | `(3, 3)` float64 rows, bohr⁻¹, `blat` folded in | every kernel |
| `cell_volume` | float, bohr³ | every kernel |
| `bdot` | `(3, 3)` reciprocal metric $\mathbf b_i\cdot\mathbf b_j$ in BerkeleyGW $2\pi/a$ units | `Box0D` only |
| `fft_grid` | `(3,)` int | `Box0D` only |

`from_wfn` is duck-typed: any object with `blat`, `bvec`, `cell_volume` (and optionally `bdot`, `fft_grid`) works, including a [`WfnLoader`](wfn_loader.md) or a [`ZetaLoader`](zeta_loader.md). A `bvec` that is not `(3, 3)` raises `ValueError`. Equality is by identity (`eq=False`), because the fields are arrays.

### get_kernel and the kernels {: #kernels }

```python
get_kernel(sys_dim) -> Bulk3D | Slab2D | Box0D
```

`sys_dim` is a `SysDim` (`BULK_3D = 3`, `SLAB_2D = 2`, `BOX_0D = 0`, an `int` enum) or a plain 0, 2, 3; `None` means 3. Any other value raises `ValueError`. Each kernel class satisfies the `CoulombKernel` protocol: one formula, `v_qG`, and `q0_average`.

| Kernel | $v(\bK)\,\Omega$ (Ry·bohr³) | Requirements |
|---|---|---|
| `Bulk3D` | $\dfrac{8\pi}{\lvert\bK\rvert^2}$ | none |
| `Slab2D` | $\dfrac{8\pi}{\lvert\bK\rvert^2}\Big[1 - e^{-z_c\lvert\bK_\parallel\rvert}\cos(K_z z_c)\Big]$, $z_c = \pi/b_{3z}$ | $\mathbf b_1,\mathbf b_2$ in the Cartesian $xy$ plane, $\mathbf b_3$ along $+z$; otherwise `ValueError` |
| `Box0D` | Wigner–Seitz cell-box FFT (BerkeleyGW `trunc_cell_box.f90`) | Γ only (`q ≠ 0` raises `NotImplementedError`); `geometry.bdot` and `geometry.fft_grid` |

`Slab2D.truncation_half_height(geometry)` returns $z_c$ in bohr. `kernel.v_qG(geometry, qvec_wrapped, comps_qG)` evaluates one $\bq$: `qvec_wrapped` is `(3,)` crystal, `comps_qG` is `(nG, 3)` Miller indices, and the result is complex128 `(nG,)` in Ry, routed through `v_qG_table` so both share one arithmetic path. `v_qG_single(kernel, geometry, qvec_wrapped, comps_qG)` is the same call as a function.

## The v(q+G) table {: #table }

### v_qG_table {: #v-qg-table }

```python
v_qG_table(kernel, q_irr_frac, gvec_components, *, geometry,
           vcoul_cutoff_ry=None, v_head_fn=None, head_tie_rtol=1e-9) -> np.ndarray
```

The production $v(\bq+\bG)$ driver.

| Argument | Meaning |
|---|---|
| `kernel` | from `get_kernel` |
| `q_irr_frac` | `(n_q, 3)` crystal |
| `gvec_components` | `(n_q, 3, ngkmax)` int Miller indices; pad slots may hold anything |
| `geometry` | `CoulombGeometry` |
| `vcoul_cutoff_ry` | Ry; G with $\lvert\bq+\bG\rvert^2$ above it are set to zero. `None`: no mask |
| `v_head_fn` | callable `K_cart (m, 3) -> (m,)` float64 in Ry (with $1/\Omega$); see [head slots](#head-slots). A non-callable raises `TypeError` |
| `head_tie_rtol` | relative window for ties in $\min_\bG\lvert\bq+\bG\rvert^2$ |

Returns host float64 `(n_q, ngkmax)` in Ry. The order is fixed: bare kernel, then head injection, then the cutoff mask, so a head slot outside the cutoff is zeroed like any other G. Pad slots are not zeroed; ζ is zero there.

From `src/gw/compute_vcoul.py` (`compute_v_q_per_G`):

```python
from vcoul import CoulombGeometry, get_kernel, v_qG_table
v_q_per_G = v_qG_table(
    get_kernel(sys_dim), q_irr_frac, gvec_components,
    geometry=CoulombGeometry(bvec=bvec, cell_volume=cell_volume,
                             bdot=bdot, fft_grid=fft_grid),
    vcoul_cutoff_ry=vcoul_cutoff_ry,
    v_head_fn=v_head_fn,
)
```

### Head slots {: #head-slots }

With `v_head_fn` given, at every $\bq$ except Γ ($\min\lvert\bq+\bG\rvert^2 < 10^{-12}$ is skipped) the head value is written into **every** slot attaining $\min_\bG\lvert\bq+\bG\rvert^2$ within `head_tie_rtol`, and every tied slot receives the **mean** of `v_head_fn` over the tied set, each member evaluated at its own Cartesian $\bK$.

- The argmin is used instead of the label $\bG = 0$ because $\lvert\bK\rvert$ is invariant under the $+\bq \leftrightarrow -\bq$ pairing $\bK\mapsto-\bK$, while a Miller label is not (BerkeleyGW's wrap sends a boundary component to $+\tfrac12$).
- All tied slots, because on an even k grid the argmin is degenerate at self-paired $\bq$ and the pairing swaps the tied slots.
- The mean, because the parallelepiped mini-BZ has lower symmetry than the crystal; per-slot values would break $\bK\to R\bK$ covariance.
- The caller must supply $f(\bK) = f(-\bK)$. `build_miniBZ_dq_cart` guarantees it for the Monte Carlo average.

```python
head_slot_table(kernel, q_irr_frac, gvec_components, *, geometry,
                vcoul_cutoff_ry=None, v_head_fn=None, head_tie_rtol=1e-9,
                k_max=None) -> HeadSlotTable
```

Reports the same selection without applying it. `HeadSlotTable` fields, each with a leading `n_q` axis:

| Field | Shape, dtype | Meaning |
|---|---|---|
| `sel` | `(n_q, k_max)` int32 | slot indices attaining the argmin, right-padded by repeating `sel[:, 0]` |
| `mask` | `(n_q, k_max)` float64 | 1 on a genuine tied slot, 0 on padding and on skipped $\bq$ |
| `v_bare` | `(n_q,)` float64 | the point kernel at the head slot (Ry) |
| `v_avg` | `(n_q,)` float64 | the value `v_qG_table` writes there |
| `mult` | `(n_q,)` int32 | tie multiplicity |
| `len2` | `(n_q,)` float64 | $\lvert\bq+\bG\rvert^2$ at the slot, bohr⁻² |

`k_max` defaults to the largest multiplicity over the supplied $\bq$ (1, 2 or 4 on even grids); pass it to pin a shape across calls.

## q → 0 cell averages {: #q0 }

### q0_average on each kernel {: #q0-average }

```python
Bulk3D().q0_average(geometry, kgrid, *, S_cart=None, epshead=None, static_kappa2=None,
                    nsamples=2**18, method="sobol", qmc_reps=10,
                    analytic_sphere=False, extra_chi=None) -> (vc0_mean, wcoul0)
Slab2D().q0_average(geometry, kgrid, *, S_cart=None, epshead=None, static_kappa2=None,
                    rule=Q0_RULE_EXACT, nsamples=2**18, method="sobol", qmc_reps=10,
                    analytic_sphere=False, certificate_fn=None) -> (vc0_mean, wcoul0)
Box0D().q0_average(geometry, kgrid=None, ...) -> (vc0, vc0)
```

Both returns are complex128 JAX scalars:

$$
\texttt{vc0\_mean} = \big\langle v(\bq)\big\rangle_{\mathrm{mBZ}}, \qquad
\texttt{wcoul0} = \Big\langle \frac{v(\bq)}{1 - v(\bq)\,\bq^{\mathsf T}S\,\bq}\Big\rangle_{\mathrm{mBZ}}
$$

in bare units (Ry·bohr³) for `Bulk3D` and `Slab2D`. `kgrid` is the `(3,)` k grid that defines the mini-BZ. The screened average takes exactly one source:

| Argument | Meaning |
|---|---|
| `S_cart` | `(3, 3)` Cartesian long-wavelength polarizability tensor $S$, with $\chi_{00}(\bq)\approx\bq^{\mathsf T}S\bq$ ([q → 0 head §1.2](../theory/q0-head.md#s-tensor)). Bulk adds `extra_chi(q)` to $\bq^{\mathsf T}S\bq$ when given |
| `static_kappa2` | positive scalar $\kappa^2$ (bohr⁻²): Thomas–Fermi $\langle 8\pi/(q^2+\kappa^2)\rangle$. Bulk only; exclusive with `S_cart`. `Slab2D` raises `NotImplementedError` |
| `epshead` | the isotropic legacy model from a BerkeleyGW `epsilon` head |

**Bulk.** Scrambled-Sobol points folded into the Γ Voronoi cell: `nsamples` per replicate, `qmc_reps` replicates. `method="sobol"` without `scipy.stats.qmc` raises `RuntimeError`; `method="auto"` falls back to a uniform draw with one warning, and the result is then not bit-comparable with a Sobol run. `analytic_sphere=True` adds the Baldereschi–Tosatti sphere and widens the Voronoi fold from one to three shells. `Bulk3D.q0_average_screened(geometry, kgrid, *, S_carts, extra_chi_rows=None, ...)` evaluates a stack of `S` rows on one shared draw.

**Slab.** The $1/q$ cusp is integrated on the exact Wigner–Seitz polygon of the mini-lattice (Γ-to-edge triangles, Duffy map, fixed 16/24/32-point Gauss–Legendre ladder). `rule` is named, with no silent alternative:

| `rule` | Meaning |
|---|---|
| `Q0_RULE_EXACT = "wigner_seitz_polygon"` | default; `nsamples`, `method`, `qmc_reps` select nothing |
| `Q0_RULE_SOBOL_DEBUG = "sobol_debug"` | the scrambled-Sobol draw, debug only; announces itself |

Refusals on the slab: the 24→32 pair must agree to `atol 1e-12`, `rtol 1e-8` (`GATE slab_q0_polygon_not_converged`); `analytic_sphere=True` raises `GATE slab_q0_analytic_sphere_unavailable` (there is no sphere for a $|q|$ cusp); an unknown `rule` raises `GATE slab_q0_rule_unknown`. `certificate_fn`, if given, receives one `SlabQ0Certificate(orders, physical_counts, polygon_edges, final_error_ratio)`; the certificate is also printed once per process.

**Box.** `v(G=0)` of the truncated kernel is finite; `q0_average` returns it, with $1/\Omega$ applied, as both members.

`q0_average_transverse_tensor(geometry, kgrid, ...)` on `Bulk3D` and `Slab2D` returns the bare transverse (Coulomb-gauge) head tile used by the four-current route; the slab reads it off the same polygon receipt.

### The 3D body head {: #body-head }

```python
build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume, *, nmc=2**18, seed=42) -> head_fn
build_miniBZ_dq_cart(kgrid, bvec, *, nmc=2**18, seed=42) -> np.ndarray   # (2*nmc, 3)
```

`head_fn(K_cart)` maps `(m, 3)` Cartesian $\bK$ to `(m,)` float64 $\langle 8\pi/\lvert\bK+\delta\bq\rvert^2\rangle_{\mathrm{mBZ}}/\Omega$ in Ry: the `v_head_fn` of `v_qG_table` on bulk input files with `mc_average_vcoul_body = true`. `build_miniBZ_dq_cart` is its draw: an `nmc`-point sample unioned with its negation, so $\{\delta\bq\} = \{-\delta\bq\}$ exactly and the head satisfies $f(\bK) = f(-\bK)$. The defaults are pinned by reference results; changing `nmc` or `seed` changes the head slot of every bulk $V_\bq$.

From `src/gw/v_q_g_flat.py`:

```python
v_head_fn = None
if sys_dim == 3:
    v_head_fn = build_v_head_miniBZ_fn_3d(kgrid, bvec, cell_volume)
```

### Mini-BZ sampling primitives {: #minibz }

| Name | Signature | Returns |
|---|---|---|
| `minibz_frac_to_cart` | `(U, bvec)` | `U @ bvec`: fractional draw to Cartesian, the only allowed spelling; numpy or jax by input |
| `minibz_cell_affine` | `(bvec, kgrid)` | `(3, 3)` float64 map from full-cell Voronoi points to mini-BZ offsets |
| `wrap_points_to_voronoi` | `(randcart, bvec, nmax=1)` | Cartesian points folded onto the Voronoi cell of `bvec`; jitted, `nmax` static (3 = BerkeleyGW `ncell`) |
| `minibz_voronoi_batches` | `(bvec, kgrid, *, nsamples=2**18, method="sobol", qmc_reps=10, nmax=1, is_2d=False, seed_offset=0)` | list of `(N, 3)` Cartesian $\delta\bq$ batches, one per replicate; `is_2d` zeros $q_z$ |
| `sample_minibz_qpoints` | `(geometry, kgrid, *, ..., stream_reps=False)` | the same batches from a `CoulombGeometry`; `stream_reps=True` yields them one at a time |
| `minibz_inscribed_sphere_r2` | `(bvec, kgrid, *, is_2d=False)` | float, squared radius (bohr⁻²) of the largest sphere in the mini-BZ |
| `minibz_average` | `(shift_cart, dq_batches, *, kind, celvol, n_kpts, q0sph2, alpha=None, zc=None, analytic_sphere=False, adaptive=True, n_coarse=250000)` | float, $\langle v(\text{shift}+\delta\bq)\rangle$ in bare units |
| `minibz_moment_tensor` | same arguments | `(3, 3)` $\langle v(\bq)\,q_aq_b\rangle$, bare units, $\bq$ the full momentum |
| `minibz_coulomb_moment` | `(bvec, kgrid, *, is_2d=False, ndirections=8192)` | `(3, 3)` Coulomb-weighted second moment $Q_{ab}$ of the cell (bohr⁻²) |
| `_minibz_kernel_bare` | `(shift_cart, dq_cart, *, kind, alpha=None, zc=None)` | `(v, len2)`, `(N,)` each: the bare kernel on a batch |

`kind` is `"bulk_3d"` ($8\pi/K^2$), `"bulk_3d_lr"` (times $e^{-K^2/4\alpha^2}$), `"slab"` (times the slab factor with half-height `zc`) or `"slab_lr"`. `minibz_average` follows BerkeleyGW `minibzaverage.f90`: inside the inscribed sphere with `analytic_sphere`, the sphere is integrated analytically and the remainder sampled.

### Photon and transverse heads {: #photon }

These serve the four-current (bispinor) route, where the Coulomb-gauge kernel has a charge block and a transverse block. In the fixed basis $(C, T_x, T_y, T_z)$ the raw bare kernel at a sample $\bq$ is $D_{CC} = v(\bq)$, $D_{TT} = -v(\bq)(I-\hat\bq\hat\bq)$, $D_{CT} = 0$, in bare units.

| Name | Meaning |
|---|---|
| `COULOMB_GAUGE_TT_SIGN` | `-1.0`, the sign of $D_{TT}$ |
| `transverse_projector(K_cart, len2, *, eps_K2=1e-30)` | `(..., 3, 3)` $I - \hat\bK\hat\bK$; identity where $\bK = 0$ |
| `apply_transverse_projector(K_cart, vector_cart, len2, *, eps_K2=1e-30, component_axis=-1)` | $J - \bK(\bK\cdot J)/K^2$ without forming the tensor |
| `bulk_photon_D_raw(q_cart)` | `(N, 4, 4)` raw bulk $D$ at nonzero points |
| `iter_minibz_photon_samples(kernel, geometry, kgrid, *, nsamples, method, qmc_reps, analytic_sphere, chunk_size=2**15)` | iterator of `(rep, start, stop, q_cart, D_raw, valid_count, mc_weight, analytic_D_raw)`, fixed-shape host chunks for a caller that solves a coupled head before averaging |
| `slab_minibz_photon_cubature(kernel, geometry, kgrid)` | `SlabMinibzPhotonReceipt`: the exact slab polygon rule with its nodes, weights and $D$ |
| `slab_minibz_graded_photon_cubature(kernel, geometry, kgrid)` | the same receipt on seven geometric radial panels toward Γ, for screened integrands with a near-Γ scale |
| `validate_slab_minibz_photon_receipt(receipt)` | checks that a receipt was issued by the provider and is unmodified; returns it or raises `TypeError` / `ValueError` |
| `SlabMinibzPhotonReceipt` | frozen record: `method`, `orders`, `reciprocal_lattice_rows`, `kgrid`, `mini_lattice_rows`, `polygon_vertices`, `polygon_area`, `slab_zc`, `cell_volume`, `physical_counts`, `padded_counts`, `weight_sum_defects`, `weighted_q_centroids`, `chunks` (one per ladder order: `q_cart`, `D_raw`, `sample_weight`). Only the provider functions can construct one |

From `src/gw/w_isdf.py`:

```python
geometry = CoulombGeometry.from_wfn(wfn)
cubature = slab_minibz_photon_cubature(
    get_kernel(2), geometry, tuple(int(v) for v in meta.kgrid))
```

## Long-range channels {: #lr }

```python
lr_channels(q_frac, *, bvec, sys_dim, n_z=0, head_fn=None) -> LRChannels(gvec, v)
lr_kernel(q_frac, gvec, *, bvec, sys_dim) -> np.ndarray       # (n_q, n_L)
ws_shift(q_frac, bvec, sys_dim) -> np.ndarray                 # (n, 3) int64
inscribed_radius(bvec, sys_dim) -> float                      # bohr⁻¹
```

The long-range channels at $\bq$ are the plane waves whose $v(\bq+\bG)$ is non-analytic as $\bq\to0$: in bulk the Wigner–Seitz slot $\bG_{\mathrm{ws}}(\bq) = \arg\min_\bG\lvert\bq+\bG\rvert$, in a slab that slot plus $G_z = n\,b_3$ for $\lvert n\rvert\le n_z$ ([Short- and long-range W §2](../theory/w-range-splitting.md#channels)).

| Field | Shape, dtype | Meaning |
|---|---|---|
| `LRChannels.gvec` | `(n_q, n_L, 3)` int32 | channel $\bG$ in crystal units, in the frame of its $\bq$; slab channels ordered $n = -n_z\ldots n_z$ |
| `LRChannels.v` | `(n_q, n_L)` float64 | $v(\bq+\bG)$ in bare units (Ry·bohr³), zero at $\bq+\bG = 0$ |

`sys_dim` other than 2 or 3 raises. `head_fn`, if given, replaces the head channel ($G_z = 0$) value at $\bq+\bG\neq0$ by a bare-unit cell average. `ws_shift` ties at the zone boundary go to the first shortest shift in a fixed order. `inscribed_radius` is half the shortest nonzero reciprocal vector of the periodic directions; inside it the WS slot is unique.

From `src/bse/bsemat_export.py`:

```python
bvec = np.asarray(CoulombGeometry.from_wfn(wfn).bvec, dtype=np.float64)
n_z = slab_shells(bvec) if sys_dim == 2 else 0
ch = vc_lr_channels(q_full_L, bvec=bvec, sys_dim=sys_dim, n_z=n_z, head_fn=None)
```

## Sphere predicate, cell box and quadrature {: #sphere }

| Name | Signature | Returns |
|---|---|---|
| `fft_box_miller` | `(fft_grid)` | `(G_frac, G_int)`, `(n_rtot, 3)` float64 and int32, every FFT-box point in fftfreq C order |
| `bare_coulomb_sphere_mask` | `(fft_grid, bvec, q_irr_frac, vcoul_cutoff_ry)` | `(mask, G_int)`; `mask` is `(n_q, n_rtot)` bool, $\lvert\bq+\bG\rvert^2\le$ cutoff |
| `bare_coulomb_sphere_indices` | same | dict: `G_int`, `idx_per_q` (ascending flat indices, index 0 is $\bG=0$), `ngk_per_q` |
| `compute_vcoul_box` | `(bdot, fft_grid, gvecs)` | `(nG,)` cell-box $v(\bG)$ at $\bq = 0$, bare (BerkeleyGW units, no $1/\Omega$) |
| `gauss_legendre_interval` | `(order, left, right)` | read-only float64 `(nodes, weights)` on $[a, b]$; `GAUSS_LEGENDRE_INTERVAL_PROVENANCE` names the rule |

The sphere functions answer only which G are inside. Padding the G list to the FFT-box sentinel layout of WFN.h5 belongs to LORRAX (`common.coulomb_sphere`).

## BerkeleyGW parity {: #bgw }

```python
read_bgw_vcoul(path) -> BGWVcoulTable
fill_v_grid_for_q(table, q_frac, fft_grid, cell_volume, tol=1e-4, sym_mats_k=None) -> np.ndarray
```

`read_bgw_vcoul` parses a BerkeleyGW `vcoul` text dump and keeps the first block of each $\bq$ (later blocks of the same $\bq$ come from fresh mini-BZ draws and are not bit-identical). `BGWVcoulTable` holds `q_fracs` `(n_q, 3)`, `G_miller_per_q` and `vcoul_per_q` (lists of per-$\bq$ arrays) and `find_q_index(q_frac, tol=1e-4, sym_mats_k=None)`, which resolves a $\bq$ through `sym_mats_k` with no shifted-$\bq_0$ fallback. `fill_v_grid_for_q` scatters one $\bq$ onto the `(nx, ny, nz)` FFT grid in LORRAX units (BerkeleyGW value $/\Omega$), with $\bG = 0$ zeroed. This is the `use_bgw_vcoul = true` path, for comparisons against BerkeleyGW.

## Invariants and gotchas {: #gotchas }

- **Rows, not columns.** `bvec` rows are $\mathbf b_i$. A fractional draw maps to Cartesian as `U @ bvec`; `U @ bvec.T` has the right volume and the wrong shape on non-cubic cells. Use `minibz_frac_to_cart` or `CoulombGeometry.from_wfn` instead of multiplying `blat * bvec` at a call site.
- **Two unit conventions.** Kernels and `v_qG_table` include $1/\Omega$; `q0_average` (bulk, slab), `minibz_average`, `minibz_moment_tensor`, `lr_channels` and `lr_kernel` do not. `Box0D.q0_average` and `build_v_head_miniBZ_fn_3d` do. Check which one a value is before adding it to a $V$ tile.
- **$\bK = 0$ is zero, not infinite.** The bulk and slab kernels set $v = 0$ at $\bq+\bG = 0$; the head is a separate rank-1 term supplied through the averages above.
- **`v_head_fn` must be even.** $f(\bK) = f(-\bK)$ is required for $V_{-\bq} = \overline{V_\bq}$. A one-sided Monte Carlo draw breaks it at the noise level; use `build_miniBZ_dq_cart`.
- **Head before cutoff.** In `v_qG_table` the head slot is injected and then masked; a head slot outside `vcoul_cutoff_ry` ends up zero.
- **Slab orientation is fixed.** `Slab2D` refuses a tilted or reversed $\mathbf b_3$; rotate the cell so the vacuum is along $+z$.
- **Seeds are part of the result.** A seeded Monte Carlo average has a deterministic error that does not look like noise; a second draw is not a convergence study. New $\bq\to0$ averages should name their rule and return a certificate.
- **Test on a non-cubic cell.** Cubic cells satisfy `bvec.T = P @ bvec` and cannot distinguish the row and column conventions; include a hexagonal or lower-symmetry case.

## Where LORRAX uses it {: #used-by }

| File | Purpose |
|---|---|
| `src/gw/compute_vcoul.py` | `compute_v_q_per_G`: thin dispatcher over `v_qG_table`; the BerkeleyGW `vcoul` reader path |
| `src/gw/v_q_g_flat.py` | $V_\bq$ from G-flat ζ: `v_qG_table`, `head_slot_table`, the 3D body head |
| `src/gw/v_q_bispinor.py` | the four-current $V_\bq$ |
| `src/gw/vcoul.py` | input-file-facing wrappers `compute_q0_averages`, `compute_q0_averages_screened` over `q0_average` |
| `src/gw/coulomb/` | compatibility adapters taking `(wfn, meta)` |
| `src/gw/head_correction.py`, `src/gw/qsgw_head.py` | the $\bq\to0$ head in Σ and QSGW |
| `src/gw/w_isdf.py`, `src/gw/photon_direct_head.py` | slab photon cubature for the packed Γ completion |
| `src/gw/lr_channels.py`, `src/bse/bse_densify.py`, `src/bse/bsemat_export.py` | long-range channels for W densification and BSE export |
| `src/bse/vq_interp.py` | the per-Q BSE exchange head (`minibz_voronoi_batches`, `_minibz_kernel_bare`) |
| `src/common/coulomb_sphere.py` | the bare-Coulomb G sphere, padded to the WFN.h5 sentinel layout |
| `src/gw/gw_init.py`, `src/gw/isdf_fitting.py` | geometry and per-$\bq$ sphere setup before the ζ fit |
