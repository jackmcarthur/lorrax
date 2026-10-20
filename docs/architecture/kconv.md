# The k-convolution family

Every k-axis convolution and k-axis transform in LORRAX goes through one
router in `src/ffi/fft.py`: the χ₀ time node, the Σ = G⋆W product, the ζ-fit
pair Gram, the BSE W term and the plane transform of the G-space charge ζ fit
(route G, [μ-batch fit](zeta_fit_mubatch.md)). This page explains
the operation they share, why it runs as one fused shared-memory pass on
NVIDIA GPUs, how the router picks a backend, and the twelve kernel modes with
their launch rules, refusals and caches. It is for anyone who calls, changes
or debugs a k-axis operation. Read [the FFI layer](ffi_layout.md) first for
the native boundary this family sits behind, and
[Symmetry §2](../theory/symmetry.md#2-the-parent-k-domain) for the parent
k-points that modes 7, 8, 9 and 11 read.

## 1. The operation {#operation}

| symbol | meaning |
|---|---|
| $(n_{kx}, n_{ky}, n_{kz})$, $N_k$ | the k grid and its size $N_k = n_{kx}n_{ky}n_{kz}$ |
| flat k | the index $(k_x n_{ky} + k_y)\,n_{kz} + k_z$: C order, $k_z$ fastest |
| $R$ | a lattice vector of the $N_k$-cell Born–von Kármán supercell, the Fourier partner of k |
| $\mu$, $\nu$ | left and right ISDF centroid indices; on the 2-D mesh $\mu$ is sharded over X and $\nu$ over Y |
| $a$, $b$, $n_s$ | spin components of the two endpoints; $n_s$ is 1 (scalar), 2 (spinor) or 4 (bispinor) |
| column | the $N_k$ values of one fixed trailing index, for example $(\mu, a, \nu, b)$; every transform runs along columns |
| $F_k$, $F_k^{-1}$ | the unnormalized forward ($e^{-2\pi i k\cdot R}$) and inverse ($e^{+2\pi i k\cdot R}$) DFT of a column |
| $n\vert 1$ | $n$ rounded up to the next odd integer (bitwise OR with 1), the padded length of a shared-memory line |
| $m_x$, $m_y$ | the rank-local left and right centroid extents ($\mu$ on X, $\nu$ on Y) of an operand |
| opt-in shared memory | the most dynamic shared memory one thread block may request. An A100 has 164 KiB per SM, of which at most 163 KiB (166 912 B) is opt-in per block |

Operands are complex128 (16 B per element) unless a mode says otherwise.

The shared operation is the pair correlation of two k-periodic operators,

$$
U_q(\mu,\nu) \;=\; \sum_k \overline{A_k(\mu,\nu)}\,B_{k+q}(\mu,\nu)
\;=\; \frac{1}{N_k}\,F_k\!\left[\,\overline{F_k^{-1}A}\cdot F_k^{-1}B\,\right](q).
$$

A sum over k of a product at k and k + q is a product in R space. The direct
sum costs $O(n_{\rm col}N_k^2)$ for $n_{\rm col}$ columns; the transforms cost
$O(n_{\rm col}N_k\log N_k)$. A convolution $\sum_q G_{k-q}W_q$ is the same
pattern without the conjugate. Each physical use differs only in the R-space
step between the transforms:

| use | R-space step | owner |
|---|---|---|
| χ₀ at one imaginary-time node $\tau$ | $\chi_0(R) \mathrel{+}= \alpha_\tau\sum_{ab}\overline{G^c_{ab}(R)}\,G^v_{ab}(R)$; one forward transform after the τ sum | `gw.w_isdf` |
| Σ at one τ | $\Sigma(R) = G(R)\cdot W(R)$ per centroid pair | `gw.ppm_tau_kernel`, `gw.cohsex_sigma`, `gw.mpa.sector_sigma` |
| ζ-fit pair Gram $C_q$ | $\sum_{ab}\phi_l[a]\phi_r[b]\,\overline{P^L_{ab}(R)}\,P^R_{\pi_l a,\pi_r b}(R)$ of band projectors | `isdf.core`, `isdf.zeta_mubatch` |
| BSE W term | $(F^{-1}T)(R)\cdot W(R)$, $T = \sum_K \Lambda_K\,\Xi_K$ | `bse.bse_stack_matvec`, `bse.bse_ring_comm` |

Here $\alpha_\tau$ is the quadrature weight of node τ; $G^v$ and $G^c$ are the
occupied (valence) and empty (conduction) parts of the Green's function at
that node, in centroid space; $P^L_k(\mu,\nu)$ and $P^R_k(\mu,\nu)$ are the
band projectors $\sum_n \psi_{nk}(\mu)\overline{\psi_{nk}(\nu)}$ over the left
and right band windows of a ζ-fit channel; a vertex is a spin permutation
$\pi$ with a phase $\phi \in \{\pm1, \pm i\}$ per spin component, one per
side; and the BSE trial stack $T$ is a sum over $K = \min(n_c, n_v)$ products
of a left transition leg $\Lambda$ and a right leg $\Xi$ ([BSE](bse.md#the-matvec)).

The physics of each row is on its owner's page. This page owns how the
transform, the R-space step and the transform back are executed.

### Why one fused pass {#why-fused}

Run as separate XLA ops, every step (a gather, the inverse transform per axis,
the product, the forward transform per axis, a transpose) reads and writes the
whole tile through device memory (HBM). The CUDA kernels instead hold a tile
of columns in shared memory from the load through both transforms to the
store, so one convolution reads each operand from HBM once and writes the
result once. The line FFTs are cuFFTDx thread FFTs from the `nvidia-mathdx`
wheel, specialized per k grid when NVRTC compiles the kernel at run time.

At large grids a tile no longer fits a block, but the fused route still wins.
At Ni 20³ ($N_k = 8000$) one $n_s = 2$ pair is four columns, 512 KB, against
the A100's 164 KiB of shared memory per SM, so the tile streams through HBM either
way. Measured at Ni 20³, P64, one Σ τ row pass: a staged route (an XLA gather
of the unfolded Green, a vendor batched 3-D FFT, the product, the forward FFT
and the parent gather) is 1.7–2.1× slower than modes 9 + 7 and holds
2.5–3.2 GB live, against the fused route's ≤ 1 GiB scratch. It wins only on
launch-bound 4³ grids (1.19×).

## 2. The router {#router}

The factory picks the backend from the mesh's device vendor alone
(`ffi.fft.kconv_backend`, the vendor from `lxkit.device_vendor`):

| device vendor | backend |
|---|---|
| CUDA | the nvidia-mathdx kernel family in `liblorrax_ffi.so`: `src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc`, and `kconv_outer_cuda_ffi.cc` for the BSE outer-product load |
| cpu | the plan backend: the reference composition of each mode (the same unfolds, products and spin sums in XLA) with the FFTW3-ABI host handlers of `liblorrax_ffi_host.so` (`lorrax_mklfft_flat_k`, `lorrax_mklfft_gw_conv`, [FFI layer §3c](ffi_layout.md#3c-which-fft-engine-the-host-library-binds)) |
| ROCM | the XLA backend, except the k-leading pair: `make_local_kconv_klead` (mode 2, and so `make_kconv_klead`'s apply: Σ τ, COHSEX, the BSE W term) and `make_local_kfft_klead` (mode 3) take the kdft family of `liblorrax_ffi_rocm.so` (`src/ffi/cpp/kdft/`, hipRTC-built: a column tile's k axes transformed in LDS by mixed-radix Stockham DFT, or for the convolution pass of a group whose axes are ≤ 32 by whole lines in registers with LDS only for the axis exchange, one HBM pass when the k tile fits a block and otherwise one pass per contiguous axis group, 2·(groups−1)+1 for mode 2; plan in `kdft_klead_plan.h`), through `ffi.fft._ROCM_KLEAD_TARGETS`; so does the k-minor pair, `make_local_kconv_kminor` (mode 4: the BSE ring rung, both `out_layout`s) and `make_local_kfft_kminor` (mode 5: the BSE kernel prep), one HBM pass on a block of whole contiguous rows staged transposed in LDS with the same DFT core where one row's k tile fits and is the cheaper plan, else the split arm: row passes over the trailing axis group and k-leading column passes over the others with the K multiply fused in one, 2·(groups−1)+1 HBM passes for mode 4 (out_layout 1 takes an XLA scratch buffer of X's size, and its last row pass stores the permuted layout with the blocks ordered by the store's k stride, `kdft_kminor_plan.h` `l1_order`; plan in `kdft/kdft_kminor_plan.h`; every kdft tile is sized to the leg's block LDS, `common/leg.h` `LRX_LEG_BLOCK_LDS`: 64 KiB on ROCM, 48 KiB on CUDA); nk ≤ `ffi.fft.KDFT_KMINOR_NK_MAX` = 10240 (`GATE kdft-nk`), axes ≤ `KDFT_KMINOR_AXIS_MAX` (`GATE kdft-axis`); `kconv_backend` stays `'xla'`. Where kdft cannot serve: beyond its verified range, an axis above 128 (`GATE kdft-axis`) or nk above 10240 (`GATE kdft-nk`), or a failed startup probe (`GATE kdft-probe`, demoted as the mathdx probe is). Mode 7 (`make_kconv_klead_unfold`, the Σ τ kernel) takes the k-box stage itself, `lorrax_kdft_kconv_klead_unfold` (`src/ffi/cpp/kdft/kbox_unfold_*`, through `ffi.fft._ROCM_KBOX_TARGETS`), on the single arm only: a grid whose spin group does not fit a block's 64 KiB of LDS keeps the XLA unfold + kdft route (`GATE kdft-kbox-residency`, `ffi.fft.kdft_kbox_refusal`), as does an axis above `ffi.fft.KDFT_KBOX_AXIS_MAX` = 40 (its line transform is the stage's unrolled direct DFT; `GATE kdft-axis`). Modes 9 and 11 take the same stage (`src/ffi/cpp/kdft/kbox_rspace_*`, same table) on three arms (`ffi.fft.kdft_kbox_arm`, the mirror of the handler's `plan()`; constants measured on one MI250X GCD): the single arm (one HBM pass), the two-pass split (the stage's (k_y, k_z) plane pass on tiles of whole groups, then the x pencil) and, on 3-D grids whose two-pass tile holds fewer than `KDFT_KBOX_PLANE_MIN_COLS` = 32 columns, the three-pass split (the load with the z lines on tiles of one padded z-line per column, the y lines in place, the x pencil). `make_kfft_klead_unfold` (mode 9, W from its q wedge) → `lorrax_kdft_kfft_klead_unfold`, single while the tile holds at least `KDFT_KBOX_WEDGE_MIN_COLS` = 16 columns (5×5×1, 8×8×1, 4³), else split, the scale applied in the pencil in place on Y. `make_kconv_chi_unfold` / `make_kconv_chi_vertex` (mode 11, the χ₀ τ node, plain and four-current) → `lorrax_kdft_kconv_chi_unfold` / `_chi_vertex`, single while a tile of whole Green pairs leaves two blocks per CU and its axis passes have at least `KDFT_KBOX_CHI_SINGLE_MIN_LINES` = 96 lines ($n_s = 2$ to 6³ and 12×12×1), else split: the plane pass, tile-major, into a $(N_k, \text{chunk}\cdot 2n_s^2)$ intermediate of pair chunks from XLA's scratch allocator (at most `scratch_bytes` = `ffi.fft.kdft_kbox_chi_scratch_bytes`), then the group pencil (a pair's columns on one wavefront, reduced by lane reads); on a 2-D grid it keeps the XLA route past `KDFT_KBOX_CHI_2D_AXIS_MAX` = 24 (`GATE kdft-kbox-chi-2d`, measured slower). The factory builds and loads each image when it traces the call (`ffi.fft._kbox_prebuild` → `lrx_kbox_rspace_prebuild_rocm`), off the running program's execute thread. Each mode has its own axis bound (`KDFT_KBOX_WEDGE_AXIS_MAX`, `KDFT_KBOX_CHI_AXIS_MAX` = 40, `GATE kdft-axis`) and the k-lead family's nk bound (`KDFT_KBOX_RSPACE_NK_MAX` = 10240, `GATE kdft-nk`); within them every grid has an arm (`GATE kdft-kbox-residency` is the handler's defensive rule); beyond any rule the factory keeps its XLA route. The BSE outer load and its fused decode (`make_local_kconv_klead_outer`, `make_local_kconv_klead_outer_decode`) take `lorrax_kdft_kconv_klead_outer` / `_outer_decode` (`src/ffi/cpp/kdft/kconv_outer_*`, through `ffi.fft._ROCM_OUTER_TARGETS`; [§11](#bse-outer)). Mode 6, `make_fused_conv_kplane` (the ISDF route-G pair convolution, the ζ fit's μ-batch loop), takes `lorrax_kdft_kconv_plane` (`src/ffi/cpp/kdft/kbox_plane_*`, through `ffi.fft._ROCM_PAIR_TARGETS`): the CUDA plane mode's load (conj(F·D), the L/R split on the load, no phased copy of D) on the k-box single arm, a tile of NO ≤ 16 output columns with their 2·n_s² operand columns in half a block's LDS (else all of it; `ffi.fft.kdft_plane_no`), the phased pair product into NO compact rows, the forward transform on those rows only; axes ≤ `KDFT_PLANE_AXIS_MAX` = 40 (`GATE kdft-axis`), one output column within the LDS (`GATE kdft-kconv-plane-tile`; `ffi.fft.kdft_plane_refusal`); gate `tests/test_kdft_plane_gate.py`. Mode 1, `make_fused_conv_kparent` (the ζ-fit pair Gram $C_q$ of `isdf.core`, typed parent tables), takes `lorrax_kdft_kconv_parent` (`src/ffi/cpp/kdft/kbox_parent_*`, same table): the CUDA parent mode's typed parent load (both phases, the antiunitary conjugation and the open-spin coefficients on the load, the same `centroid_major` input layout) in mode 6's tile (`ffi.fft.kdft_parent_refusal`, `GATE kdft-kconv-parent-tile`), one thread per (output column, k) reading each source once for all $n_s^2$ entries; measured 1.3–6.2× the XLA route on one MI250X GCD, so routed wherever it fits (record above `ffi.fft._kdft_parent_fn`). Mode 10, `make_plane_fft_gather`, takes `lorrax_kdft_plane_fft_gather` (`src/ffi/cpp/kdft/plane_gather_*`; [§10](#mode-10)) where `ffi.fft.kdft_plane_fft_refusal` is empty, except a plane with an axis in `ffi.fft.KDFT_PLANE_FFT_SLOW_AXES` (32: measured 0.93×; `GATE kdft-plane-slower`). Gate for modes 1 and 10: `tests/test_kdft_parent_gate.py`. Every factory built on a ROCm mesh announces its route once (`[kconv] k-convolution router: ROCM <factory> k-grid <grid> -> <route>`, visible with `LORRAX_DEBUG_PRINT=1`) |
| any other | the XLA backend: the same composition with XLA's k-axis transforms (one dense DFT GEMM on the k axis for $N_k \le$ the leg's `xla_kdft_max_nk`, `jnp.fft` above; [§15](#numerical-contract)); no LORRAX library. The ROCM rows above that are not kdft factories take this backend too |

**Where mathdx cannot serve.** On CUDA a factory takes the XLA backend, with
one `RuntimeWarning` and a router line per grid, where `ffi.fft.mathdx_refusal`
names a reason. It is decided from the k grid and the device's opt-in shared
memory per block, so every rank decides alike:

* an axis above `KCONV_AXIS_MAX` = 40, the fp64 cuFFTDx thread-FFT limit;
* a split-arm plane tile of 16 columns, $16 \cdot 16 \cdot ((n_y (n_z|1))|1)$ B,
  beyond the opt-in memory; this is every k-box mode's floor at $n_s \le 4$ (A100 to
  25³, sm_86/89 to 19³);
* for the k-minor modes 4 and 5, which have no split arm, one k-box column of
  $16 \cdot ((n_x n_y (n_z|1))|1)$ B beyond it;
* a probe compile that failed at startup on any process (`require_kconv`):
  every factory then takes XLA. Rank 0 prints the demotion with its cause and
  its cost ([§1](#why-fused)), and `common.jax_compile_cache.record_demotion`
  puts `demoted kconv` on every compile receipt and the cause in the startup
  facts (`compile_cache.demotions`).

`tests/test_kconv_xla_gate.py` holds the fallback to mathdx on the same grid (CUDA, with
the axis cap forced to 1) and to the XLA reference at a forced axis of 48 (cpu).

The backend follows from the vendor and these limits, so no deck key or environment variable
selects it; one environment dial changes only the engine inside the mathdx
route, the A/B test `LORRAX_BSE_OUTER_KSUM` ([§11](#bse-outer)). The mathdx
family is kept on CUDA because it is decisive on memory ([§1](#why-fused)), and
the host plan route on cpu because it is 3.8–7.4× faster than `jnp.fft` at one
CrI3 6×6 P4 rank tile on 128 Milan cores (the Σ τ convolution 0.16 against
1.20 s, the χ₀ transform pair 0.34 against 1.32 s; peak RSS 5.6 against
4.2 GB; claim 3979) ([XLA reference ruling](decisions.md#xla-reference)). All three
backends return the same callable contract, so a consumer never branches on
the backend.

**The XLA reference.** Factories built inside the context manager
`ffi.fft.xla_reference()` take the XLA backend on any platform. It is a Python
argument of the caller, not a dial: `tests/test_kconv_xla_gate.py` uses it to
hold every mode it covers (0, 2, 3, 4, 5, 7, 8, 9, 10, 11, with mode 11's channel
vertices) to relative 1e-12 of
the XLA backend on the same devices, mathdx on four GPUs and the plan route on
four host devices, and stage-time measurements use it to time the XLA arm.

**Startup.** `runtime.initialize_communicator_stack` calls
`ffi.fft.require_kconv(mesh)`. On CUDA it finds the wheel (`mathdx_root`),
probes every target in `ffi.fft.KCONV_TARGETS` in the loaded library (the q-row store's
included, so a stale library refuses here, not at the first W stream), and
compiles and runs one probe kernel (mode 3 on a 2×1×1 grid). A device that
the installed cuFFTDx cannot compile for is therefore found at startup
(`GATE mathdx-probe` text, naming its compute capability and the wheel version)
instead of at the first convolution; a failed probe sends every factory to the XLA backend,
announced and recorded as above. A CUDA device below compute capability 8.0
refuses first (`GATE cuda-sm80`). On cpu it requires the two host plan
targets and runs each host engine once, a 2-point FFT and (with
`LORRAX_BANDS_GEMM_FFI` on) a 1×1 batched GEMM, so a missing engine library
refuses before setup (`GATE plan-engine`); the XLA backend needs nothing at startup. Each mathdx factory re-probes its own target when it is built, and
every factory checks operand shapes and dtypes at trace time.

**Calling a factory.** Physics code imports the factories from
`common.fft_helpers`, which re-exports them; the flat-k transform enters as
`common.fft_helpers.make_flat_k_fft` and its wrappers, which call
`make_kfft_klead`. Every factory name is unique, so `git grep -n <factory>`
lists its call sites.

- **Pick the factory whose k position matches the tile you hold** (k leading,
  k trailing, parent tables, route-G planes). Never transpose to reach
  another factory: the transpose is a full extra HBM pass.
- **Sharding.** The pair, parent and plane factories and every `make_local_*`
  factory return rank-local callables for use inside the caller's
  `shard_map`; the others wrap their own `shard_map`. The k axes are always
  replicated, so every rank holds whole columns. Specs of the k-leading
  factories are given in the 3-D form, with the three leading axes `None`.
- **Scale.** Every handler takes one total scale $s$, computed in Python from
  `jnp.fft`'s norm conventions (`ffi_fft_scale`, `conv_kpair_scale`). The
  handlers implement no norm of their own. The parent and plane factories fix
  `norm="forward"`, so $s = 1/N_k$, the factor in the equation above.
- **`KConvStored`.** `make_kconv_klead` returns `(prep, apply)`. `prep(W)`
  does the work that depends on W alone, once per W; `apply(T, W_prep)` does
  the rest, once per T. `W_prep` is in the backend's own form (R space, or W
  unchanged on the plan backend, whose `gw_conv` handler transforms W itself);
  pass it only to the `apply` of the same pair.
- **Vertices (modes 0, 1 and 6).** `perm_l`, `perm_r` are permutations of
  `range(n_s)`; `phase_l`, `phase_r` are exact monomials in $\{+1,+i,-1,-i\}$,
  encoded 0–3. Anything else refuses at factory time; $n_s \le 4$.

## 3. Factories and modes {#modes}

One handler file serves modes 0–11, each a value of the compile-time
`LRX_MODE` of one embedded source. A target string is `lorrax_mathdx_<suffix>`;
`ffi_loader._CUDA_TARGET_SYMBOLS` maps it to its C++ handler symbol.

| mode, target suffix | factory (`ffi.fft`) | computes | operands → result | plan and XLA backends (host transforms on cpu, XLA k transforms elsewhere) |
|---|---|---|---|---|
| 0 `kconv_pair` | `make_fused_conv_kpair` | $U = s\,F_k\sum_{ab}\phi_l[a]\phi_r[b]\,\overline{F^{-1}A_{a,b}}\cdot F^{-1}B_{\pi_l a,\pi_r b}$ | `A`, `B` `(n_kx, n_ky, n_kz, n_s, col, μ, n_s)` → `U` `(n_kx, n_ky, n_kz, col, μ)` | two inverse transforms, the spin sum, one forward transform |
| 1 `kconv_parent` | `make_fused_conv_kparent` | mode 0 with both operands unfolded from raw parents on the load ([§9](#mode-1)) | `D_l`, `D_r` `(n_parent, n_s, μ, n_s, ν)` and ten tables → `U` `(N_k, μ, ν)` | the typed parent load in XLA (a full-k `(N_k, n_s, μ, ν, n_s)` copy per side), then mode 0's composition |
| 2 `kconv_klead` | `make_kconv_klead` (`apply`), `make_local_kconv_klead` | $U = s\,F_k(F^{-1}T\cdot V_R)$, $V_R$ already in R space | `T`, `U` `(N_k, a, m_x, b, m_y)`, `V_R` `(N_k, m_x, m_y)` | `lorrax_mklfft_gw_conv` (`make_kconv_klead` on cpu); the inverse transform, the product, the forward transform |
| 3 `kfft_klead` | `make_kfft_klead`, `make_local_kfft_klead`; `prep` of `make_kconv_klead` | $Y = s\,F^{\pm}_k X$, k leading | `(N_k, rows)` | `lorrax_mklfft_flat_k`; the XLA k transform (DFT GEMM, or `jnp.fft` over the three k axes) |
| 4 `kconv_kminor` | `make_kconv_kminor`, `make_local_kconv_kminor` | $U = s\,F_k(F^{-1}X\cdot K_R)$, k trailing, $K_R$ made by mode 5 | `X` `(d0, d1, d2, d3, d4, N_k)`, `K_R` `(d1, d2, N_k)` → X's layout (`out_layout=0`) or `(d0, N_k, d3, d1, d4, d2)` (`out_layout=1`) | k moved to the front, the transforms and the product, k moved back |
| 5 `kfft_kminor` | `make_kfft_kminor`, `make_local_kfft_kminor` | $Y = s\,F^{\pm}_k X$, k trailing | `(rows, N_k)` | the same transpose around one transform |
| 6 `kconv_plane` | `make_fused_conv_kplane` | mode 0 read from route G's plane-transform output (`g` planes of `p` points, `c` centroids per side): the Bloch phase `F[k,g,p]` applied and the `2c` axis split on the load into the left projector (slots `[0, c)`) and the right one (`[c, 2c)`) | `D` `(N_k, g, n_s, 2c, n_s, p)`, `F` `(N_k, g, p)` → `U` `(N_k, c, g·p)` | phase, split and transpose in XLA, then mode 0's composition |
| 7 `kconv_klead_unfold_kf` | `make_kconv_klead_unfold` | mode 2 on the typed unfold of the raw-parent Green ([§5](#unfold-on-load)) | `G`, `Gt` `(n_parent, μ, n_s, ν, n_s)`, `V_R` `(N_k, μ, ν)` → `U` `(n_out, n_s, μ or an x block, n_s, ν)` | `symmetry_maps.apply_unfold_load_tables_local` (a full-k copy), mode 2's composition, the row selection |
| 8 `kconv_klead_lorentz_wparent` | `make_kconv_lorentz_unfold` | the four-current Σ: $U = s\,F_k\sum_{A,B}\gamma_A(F^{-1}\hat G)\gamma_B^\dagger\circ\hat W_R[k,\mu,A,\nu,B]$, $A$ ($B$) over the $n_A$ ($n_B$) Lorentz components of the left (right) endpoint (1 for charge, 3 for current), $\gamma_A$ channel $A$'s signed spin permutation, with $\hat G$ and $\hat W$ both unfolded from parents on the load | `G`, `Gt` as mode 7; `W`, `Wt` `(n_{q,parent}, μ, n_A, ν, n_B)`, $n_A, n_B \le 4$ → `U` `(n_out, n_s, μ, n_s, ν)` | both unfolds, the transforms and the γ block sum |
| 9 `kfft_klead_unfold` | `make_kfft_klead_unfold` | $Y_k = s\,F^{-1}_k(\mathcal L_k\hat O_k \mathcal R_k^\dagger)$: the R-space operand of modes 2 and 7, read from an interaction's q wedge | `W`, `Wt` `(n_wedge, μ·n_l, ν·n_r)` → `Y` `(N_k, μ·n_l, ν·n_r)` | the unfold in XLA, then mode 2's `prep` |
| 10 `plane_fft_gather` | `make_plane_fft_gather` (`LocalFourierPlan(in_gather=…)`) | the route-G plane FFT, gathered on load ([§10](#mode-10)) | `F` `(…, n_col)` → `Y` `(…, n_b, n_c)` | the XLA route: a static-run concatenate, then `jnp.fft.fftn` |
| 11 `kconv_chi_unfold`, `kconv_chi_vertex` | `make_kconv_chi_unfold`, `make_kconv_chi_vertex` | one χ₀ τ node: $acc[o] \mathrel{+}= \alpha_o\sum_{ab}\overline{(F^{-1}\hat G^c)_{ab}}(F^{-1}\hat G^v)_{ab}$ (+ its conjugate when `complete`) | `Gv`, `Gc` and partners `(n_parent, μ, n_s, ν, n_s)`, `α` `(n_out,)` → `acc` `(n_out, N_k, μ, ν)`, in place | the unfold per Green, the inverse transforms, the trace |
| 2 + outer load, `kconv_klead_outer` | `make_local_kconv_klead_outer` | mode 2 with $T = \sum_K \Lambda_K\Xi_K$ formed in shared memory ([§11](#bse-outer)) | `L` = $\Lambda$ `(N_k, a, m_x, K)`, `R` = $\Xi$ `(N_k, K, b, m_y)`, `V_R` `(m_x, m_y, N_k)` → `U` `(N_k, a, m_x, b, m_y)` | the einsum for T, then mode 2's composition |
| 2 + outer load + decode, `kconv_klead_outer_decode` | `make_local_kconv_klead_outer_decode` | the same, with $A = \sum_{a,x}\overline{P_c}\,U$ formed in the store | → `A` `(N_k, n_c, b, m_y)` | the outer composition and the decode einsum |

In modes 0 and 6 each `(col, μ)` entry is one row, an independent column
of the pair operands (`col` is the operand's other non-k index).

Where each mode is used, by owner module:

| modes | used by |
|---|---|
| 0, 1 | the ζ-fit pair Grams: `isdf.core` |
| 6 | route G (`isdf.zeta_mubatch`); the unwired real-space pair convolution `gw.mixed_basis_pair_convolution` |
| 2, 3 | Σ and COHSEX through `make_kconv_klead` (`gw.cohsex_sigma`, `gw.ppm_tau_kernel`); the BSE W term through `make_local_kconv_klead` (`bse.bse_stack_matvec`); the flat-k transform (`gw.w_isdf` for χ₀(R) → χ₀(q), `gw.qsgw_head`, `gw.wavefunction_bundle`, `bandstructure.fh_interp`, `bandstructure.orbital`) |
| 4 | the BSE ring matvec (`bse.bse_ring_comm`) |
| 5 | the kernel prep of the BSE matvecs (`bse.absorption_haydock`, `bse.bse_densify`, `bse.bse_nontda`, `bse.bse_stack_matvec`, `bse.exciton_bands`; `bse.bse_lanczos` through `common.fft_helpers.get_donated_kfft_kminor`, which donates its input) |
| 7 | the Σ τ kernel (`gw.ppm_tau_kernel`) |
| 8 | the four-current sector Σ (`gw.mpa.sector_sigma`) |
| 9 | `gw.ppm_tau_kernel`, `gw.cohsex_sigma`, `gw.screening` |
| 10 | route G, through `common.fourier_plan.LocalFourierPlan` |
| 11 | the χ₀ streams of `gw.w_isdf` ([§5](#unfold-consumers)) |

The library also exports five mathdx targets that no current factory calls
(`kconv_klead_unfold_rows`, `kconv_klead_lorentz_rows`,
`kconv_klead_unfold_block`, `kconv_klead_unfold_xblock`, `kconv_klead_lorentz_conj`). They exist
only for older source trees: the mode-7 ones carry no k-fastest tables, so their single arm is
the chain (the same values, slower; [arms](#arms)). They stay because a
target string never changes its operand contract: a new contract is a new
target, so source trees built against the old ones keep loading
([FFI layer §8](ffi_layout.md#8-hard-invariants)).

## 4. The k-box stage {#kbox}

Every mode except 0, 1, 6 and 10 runs on one stage,
`src/ffi/cpp/cufft/kbox_stage.cuh`. A mode supplies three pieces:

- **Load**: from HBM into the shared-memory bank. A plain copy (modes 2 and
  3), a strided load of each column's k run (modes 4 and 5), the typed unfold
  gather (modes 7, 8, 9, 11), or the outer product (the BSE
  load).
- **Mid**: the R-space step between the transforms: the product with $V_R$,
  the vertex sum, or the spin trace.
- **Store**: the scaled write, through a row map where the mode keeps only
  some k rows.

The stage owns the transforms and the launch rule. A transform runs one line
transform per line along z, then y, then x; the forward transform after the Mid
uses the same axis order. The line transform is the leg's policy, the stage's
one vendor seam (`line_fft`, with the copy and the lane read): a cuFFTDx thread
FFT under NVRTC, the kdft direct DFT against compile-time twiddles under hipRTC
(`LRX_KBOX_DIRECT_DFT`; a wavefront of 64 lanes). `lrx_kbox::fft_direction`
names the leg's direction type, so one mode text serves both legs; the CUDA
images are byte-identical to those of the stage before the split (CUDA 13.4
NVRTC, sm_80, modes 2/3, 7, 8, 9, 11 and the BSE outer load).

**Bank geometry** (`lrx_kbox::Geometry`). A column's z line is padded to
$n_{kz}|1$ and the column stride to $(n_{kx}n_{ky}(n_{kz}|1))|1$ elements;
both are odd. Power-of-two grids otherwise map consecutive lines onto the same
shared-memory banks and serialize the accesses.

**Launch rule** (`lrx_kbox::kbox_plan`), computed once per kernel build from
the k grid and the device's opt-in shared memory per block:

- *group*: the columns a block must hold together: 1; $n_s^2$ for a spin group
  (modes 7 and 8, whose Mid mixes the spin components of one centroid pair);
  $2n_s^2$ for mode 11, which loads two Greens per pair.
- *single pass*: $t_r$ groups resident per block. $t_r$ is the smallest power
  of two that gives 128-byte contiguous runs per k (8 complex128 columns) and
  at least one line per thread in every axis pass, capped so that two blocks
  fit on an SM. A block runs 256 threads, or 512 for a convolution whose axis
  passes have at least 384 lines. One HBM pass, no scratch.
- *split arm*: when fewer than `min_tr` groups fit. Plane passes over
  $(k_y, k_z)$ on tiles of columns, then an x-pencil pass that fuses the
  inverse x transform, the Mid and the forward x transform in registers.
  `min_tr = 2` protects the 128-byte runs of a copy load; a strided or
  gathered load (modes 4, 5, 7, 8, 9, 11) has no run to protect and passes `min_tr = 1`.

Two single-buffered blocks per SM are preferred over one double-buffered
block because they measured faster for the convolution modes. Each line sees
the same thread FFT on the same inputs in the same axis order in either arm,
so the arms agree bit for bit.

| mode | single arm | split arm |
|---|---|---|
| 2, 3 | tiles of whole columns | plane and pencil passes through the output, in place |
| 4, 5 | one column per block at least | none: a column larger than the opt-in memory refuses |
| 7 | tiles of whole $n_s^2$ groups of two or more columns | chunks of pairs through scratch ([§7](#tiles)): a gather plane pass on tiles of the most whole groups the opt-in memory holds; then, if one padded column fits a block (20³: 134 of the A100's 163 KiB), one column-resident pass (x pencil, $V_R$, forward transform, store), else an x pencil, a forward plane pass and a pencil-and-store pass |
| 8 | tiles of whole $n_s^2$ groups plus the pair's $n_A n_B$ W columns | plane and group-pencil passes, chunked over pairs through scratch |
| 9 | tiles of whole $n_l n_r$ groups | plane tiles of whole groups, then the x pencil in place |
| 11 | $t_r$ whole pairs ($2n_s^2$ columns each) | plane passes on 16-column tiles, then a warp-shuffle x pencil per pair (it stages nothing), chunked over pairs through scratch |

### The arms of modes 7, 9 and 11 {#arms}

`build()` in `kconv_mathdx_cuda_ffi.cc` picks one arm per image from the k-grid, $n_s$, the
variant and the device's attributes (opt-in, per-SM and reserved shared memory, threads per SM,
the SM count), and names it in the build log line (`[kconv_mathdx] … arm=<name>`, rank 0). Every
arm gives the same values bit for bit. The CUDA gate (`tests/test_kconv_xla_gate.py`
`ARM_CASES`) runs one grid per arm against the XLA reference and asserts the logged arm on an
A100. The grids per device class come from the host rules evaluated at each class's opt-in
memory (A100 sm_80 163 KiB, H100 sm_90 227 KiB, sm_86/89/120 99 KiB), not from kernel logs;
only the A100 column is measured. The performance thresholds (the tile and block rules, $N_k \ge 128$
for W2) are A100-tuned; elsewhere they are correctness-safe, because every residency bound is
read from the device. "Gain" is the speedup over the plain k-box arm it replaced, from the
measured record above `ffi.fft.make_kconv_klead_unfold`.

| mode | arm (`arm=`) | condition (`build()`) | cubic grids A100 / H100 / 99 KiB | gain (A100) |
|---|---|---|---|---|
| 7 | `m7f`: fused single-arm tile, reading the k-fastest tables | the `_kf` target (the tables present), single arm (a block holds a spin group of ≥ 2 columns), whole group ($d = n_s$), $n_s \le 2$, no tile tables, `pinfo` fits | $n_s$ 1: 2³–13³ / 2³–15³ / 2³–11³; $n_s$ 2: the same | Fe 8³ 7.84 → 4.06 ms |
| 7 | `tt`: tile tables | single arm, $n_s = 4$, `tile_table_plan` finds a tile at ≥ 2 blocks per SM | 2³–8³ / 2³–9³ / 2³–7³ | 1.90× |
| 7 | `chain`: stage, transform3, Mid, transform3, store | single arm otherwise: the legacy targets (no k-fastest tables; an output spin block $d < n_s$ only through `_block`), or $n_s = 4$ without a tile | no current caller | (the reference) |
| 7 | `m7s`: orbit-blocked persistent plane pass, pair-major $W_R$, overlapped column pass | arm 2 (one padded column fits a block), $n_s = 2$, whole group, its static tables fit | 14³–21³ / 16³–23³ / 12³–18³ | 1.10–1.30× |
| 7 | `arm2`: plane pass, column-resident pass | arm 2 otherwise ($n_s$ 1 and 4) | $n_s$ 1: 14³–21³ / 16³–24³ / 12³–18³; $n_s$ 4: 9³–21³ / 10³–24³ / 8³–18³ | (the reference) |
| 7 | `arm1`: plane, x pencil, plane, pencil and store | split, one padded column exceeds a block | ≥ 22³ / ≥ 25³ / ≥ 19³ ($n_s$ 4 refused from 26³ / 30³ / 20³) | (the reference) |
| 9 | `w2`: batched scalar gather, fixed-column store, two blocks a SM | single arm, $n_l = n_r = 1$, sm_80+, $N_k \ge 128$ | 6³–13³ / 6³–15³ / 6³–11³ | Fe 8³ 1.317 → 0.950 ms |
| 9 | `single` | single arm otherwise | 2³–5³ (all classes) | (the reference) |
| 9 | `split` | no block holds a group of two columns | ≥ 14³ / ≥ 16³ / ≥ 12³ | (the reference) |
| 11 | `m11f` (`m11q` with `rows=`): fused single-arm tile | single arm, $n_s = 2$, no vertices, no tile tables, `pinfo` fits | 2³–10³ / 2³–11³ / 2³–9³ | Fe 8³ 9.92 → about 5.0 ms (`acc=None` 2.5×, `rows=` 2.8×) |
| 11 | `tt` | single arm, $n_s = 4$, no vertices | 2³–6³ / 2³–7³ / 2³–5³ | 1.73× |
| 11 | `single` | single arm otherwise ($n_s$ 1, vertices) | $n_s$ 1: 2³–17³ / 2³–19³ / 2³–14³ | (the reference) |
| 11 | `split` | a pair's $2n_s^2$ columns exceed a block | $n_s$ 2: ≥ 11³ / ≥ 12³ / ≥ 10³ | (the reference) |

**k-fastest tables.** m7f reads the endpoint tables in the layout `(m, N_k, n_s)`
(`ffi.fft.klead_unfold_kf`; a warp's gather items read one run along k per centroid) and nothing
else. The factory forms them with `jnp.transpose` from the tables it is handed: constant-folded
once per plan from baked host tables, one XLA transpose of a row pass's cut tables when the
consumer passes `load`. They are `4·N_k·(μ_x + ν_y)·n_s` B a call or pass (Fe 8³ rank tile
80 × 640: 2.95 MB; 13³, 16 × 640: 11.5 MB), counted by the program's `memory_analysis`, no
handler scratch. They replace M7KF, a per-call transpose kernel into the handler's scratch (its
gain is kept: Fe 8³ 4.06 ms, end to end equal to M7KF's within 0.6 % at Fe 8³, 12³ and 13³).

Pruned 2026-10-17 (each below 1.15× where it was measured; a host rule and kernel text apiece;
record above `ffi.fft.make_kconv_klead_unfold`): mode 7's per-call k-fastest index tables (M7KF,
replaced by the XLA-formed tables above), m7s at $n_s = 1$, mode 9's four-block W2 rule below $N_k = 256$ with its fixed-column exception
(W2 there now runs two blocks a SM), and mode 9's persistent split plane pass (m9b). Their cost
against the previous kernels, per call on an A100: mode 7 up to 1.12× at 14³–15³ $n_s = 1$ (the
m7s rule); mode 9 1.20–1.22× at 6³ (0.30 → 0.36 ms) and 1.11–1.23× at ≥ 14³; every other case
of the 45 within 1.02×.

**Why the chain stays.** m7f's gather issues a pair's $n_s^2$ raw source cells by `cp.async`
into that pair's $n_s^2$ bank columns before it forms $U g U^\dagger$ in place. An output spin
block $d < n_s$ has only $d^2$ columns a pair, so the raw cells have no slot; holding the whole
group and storing the block would transform $(n_s/d)^2$ times the columns (4× at $n_s = 2$,
$d = 1$). No current caller asks for $d < n_s$: `gw.greens_function_kernel.sigma_spin_block`'s
$d$ selects x blocks of the whole group (`_xblock`), so the chain serves older trees through the
legacy `_block` target, and $n_s = 4$ where no tile-table tile fits two blocks per SM.

**Early failure.** The startup probe compiles mode 3 only; each arm's image first compiles at
its first call, and an NVRTC failure there is loud and sticky (no XLA demotion). Prebuilding the
images a run will use at factory time, as ROCm's `ffi.fft._kbox_prebuild` does, is a follow-up:
the CUDA leg exports no host prebuild entry, and a demotion through `record_demotion` needs every
process to agree, which only `require_kconv` (a collective point that does not know the grids)
provides.

Each pass of a split arm is its own entry point of one NVRTC program, with its
own thread count, because the heaviest pass's register count would otherwise
limit every pass to fewer warps per SM. The thread counts are chosen in the
handler's `build`. For scale: Fe 8³, $n_s = 2$, mode 7 on an A100 runs the
single arm with 2 pairs per block, 512 threads and 73.9 KB of shared memory,
two blocks per SM.

## 5. Unfold on load {#unfold-on-load}

Symmetry lets the Green and the screened interaction be built only on the
parent k (or q) points, $n_{\rm parent} \ll N_k$ rows
([Symmetry §4](../theory/symmetry.md#4-two-point-operators-g-v-w)). A
transform along k needs every k. Writing a full-k copy first would multiply
the memory by $N_k/n_{\rm parent}$ and add one HBM round trip. Modes 7, 8, 9
and 11 instead form each full-k row inside the load from the parent tile, so
no full-k Green or interaction is ever written.

On one rank, with merged endpoints $i = \mu n_s + a$ (X-local) and
$j = \nu n_s + b$ (Y-local), the load forms

$$
V_k[i,j] = \mathrm{mph}[k,i]\; S_{\mathrm{row}(k)}\big[\mathrm{lsrc}(k,i),\,\mathrm{rsrc}(k,j)\big]\;\mathrm{nph}[k,j],
\qquad
\hat G_k[\mu a,\nu b] = \sum_{c,d} U_k[a,c]\,V_k[\mu c,\nu d]\,\overline{U_k[b,d]},
$$

with $S = G_t$ on an antiunitary row and $S = G$ otherwise, and a source of
−1 read as an exact zero. The tables are
`symmetry_maps.unfold_load_tables` (`UnfoldLoadTables`):

| table | shape, dtype | meaning |
|---|---|---|
| `row` | `(N_k,)` int32 | the parent row of each full k |
| `trs` | `(N_k,)` int32 | 1 on an antiunitary row |
| `lsrc`, `rsrc` | `(N_k, μ·n_s)`, `(N_k, ν·n_s)` int32 | X-local and Y-local merged source of each endpoint, −1 = zero |
| `mph`, `nph` | the same shapes, complex128 | left and right umklapp phases $e^{\pm 2\pi i q\cdot L}$, with the time-reversal rule applied |
| `spin`, `spin_r` | `(N_k, n_s, n_s)` complex128 | the spin action $U_k$; `spin_r` a right action of another width (a Lorentz block) |

`unfold_load_tables` refuses a source map that crosses an X (left) or Y
(right) shard, because the load reads only this rank's parent tile. Centroids
are orbit-packed so that the maps stay local
([symmetry register §4](symmetry_register.md#4-orbit-packed-layout-and-the-axis-local-certificate)).

**Antiunitary rows** come in three forms:

- *pair transpose*: the load reads the transposed partner tile `Gt`
  (`trs_rule="pair_transpose"`), for a Green and for W in mode 8;
- *conj(G)*: a Green built with real weights has `conj(G)` as its partner, so
  the load conjugates `G` itself and no partner tile exists
  (`conj_partner=True` in modes 7 and 8; partners `None` in mode 11);
- *conjugated product*: a Hermitian interaction on its q wedge conjugates the
  phased product (`trs_rule="conj"`), also without a partner tile (mode 9).

**Stored rows.** The Σ consumers need Σ only at the parent k, where the band
projection reads it. Modes 7 and 8 take `store_rows` (the plan's
`parent_full_rows`) as a map `kout`: full-k row k goes to output row
`kout[k]`, or is transformed but not stored when `kout[k] = −1`. Mode 7 can
also store one x block of the left centroids,
`rows = (x0, bx, xs, xn)`: block row $r$ is the local centroid
$\lfloor r/b_x\rfloor x_s + x_0 + r \bmod b_x$ (`ffi.fft.x_block_rows`). A
call reads only its own rows' sources, so a consumer that bounds its output by
x blocks reads the Green once over all of them.

**Placed tables.** `symmetry_maps.device_load_tables` places the tables once
per run (per-k tables replicated, left tables on X, right tables on Y). A
consumer passes them as the `load` operand, so its program holds no table
constants. Without `load` the factory bakes the host tables into the program;
at Fe 20³ with 1792 centroids that is about 0.57 GB of HLO literal per
program, which is why production consumers pass `load`.

### What each unfold mode reads {#unfold-consumers}

**Mode 7** reads the Green of the Σ τ kernel; its $V_R$ operand comes from
mode 9.

**Mode 8** reads the interaction the same way as the Green: `W` is its
irreducible-q parent tile, unfolded through its own `w_tables` (pair-transpose
rule, Lorentz endpoint actions in `spin`/`spin_r`). One transform of $\hat G$
serves every one of the $n_A n_B$ Lorentz blocks, and neither a full-q W nor
a full-grid $W_R$ exists.

**Mode 9** gives the R-space operand that modes 2 and 7 multiply by: W, V or
a pole field read from its q wedge, with the endpoint actions $\mathcal L_k$, $\mathcal R_k$
(1 for a scalar interaction, the Lorentz rotation for a current block).

**Mode 11** serves three χ₀ streams of `gw.w_isdf`:

- the identity-vertex step response;
- the selected-q charge streams of the shared-pole response bank (the χ₀
  samples the W model is fitted to; [shared-pole model](shared_pole_model.md)):
  the direct stream, the retarded stream and the static correlation obtained
  through the Kubo–Martin–Schwinger (KMS) relation. Each node's correlation
  $r$ is transformed once, and its rows at $q$ and $-q$ are combined by the
  stream's pair rule: retarded $-i(r_q - r_{-q})$, KMS static
  $-(r_q + r_{-q})$;
- the four-current direct stream, through `make_kconv_chi_vertex`, with at
  most three monomial vertices per side. A four-spinor Green splits into four
  quadrants by upper and lower Dirac components; `sign_c`, an optional per-k
  sign, gives one quadrant's sign relative to the shared unfold tables.

`make_kconv_chi_unfold`'s `complete` flag adds the complex-conjugate term,
which a real-frequency contour needs. `acc=None` returns $\alpha_o\chi_\tau$ itself, bit for bit
the call on a zero `acc`: the CUDA handler takes a rank-1 one-element placeholder for `acc`
(`ffi.fft._CHI_WRITE_PLACEHOLDER`, not aliased) and writes without reading; other routes add to
zeros. `acc=None, rows=q` (`fn.q_store`, the rule `ffi.fft.chi_q_store_ok`: CUDA, $n_s = 2$,
`n_out` 1, the m11f arm) returns the rows `q` of the flat-k forward transform of that value,
formed in the tile with no full-k accumulator, through `lorrax_mathdx_kconv_chi_unfold_q`
(`KCONV_CHI_UNFOLD_Q_TARGET`, in `KCONV_TARGETS`); a static `rows=` outside $[0, N_k)$ refuses. A grid mode 11 cannot hold takes the
XLA backend ([router](#router)): every grid whose split-arm plane tile fits
holds mode 11 too.

### Distinct electronic and response lattices {#bosonic-response-grid}

The optional scalar response `qgrid` is a full-BZ grid, not a near-Γ patch
or a downsample of electronic input Greens. The Greens and mode-11 inverse
transforms retain the complete electronic `kgrid`. Before the χ forward
transform, `common.fft_helpers.make_response_lattice_fftn` restricts their
real-space spin product to the response lattice. For an even response axis
of length m it keeps |R| < m/2 at unit weight, keeps both ±m/2 images at half
weight and folds those images into the single periodic Nyquist slot; the
axis weights multiply at edges/corners. Odd axes keep |R| ≤ (m−1)/2. Each
response axis divides its electronic axis. Equal grids return the legacy
transform without any changed arithmetic.

The response transform is unitary on its own grid and multiplied by
√(Nq/Nk); the native physical prefactor still uses the electronic Nk. Thus
24×24 electronic Greens and a 6×6 response lattice give a raw factor 1/4,
not a new 1/36 electronic normalization. `selected_q`/`q_ids` then index the
coarse full BZ. The old fused fine-grid q-store is disabled for this case:
the existing R-space mode-11 output is restricted before the smaller FFT.
This explicit rectangular lattice-window approximation is inversion symmetric
but does not assert the original point-group or centroid-home-cell covariance;
the response-domain owner must not reuse an unauthenticated old IBZ reduction.

For Σ, mode 9 inverse-transforms W on `qgrid`. Its real-lattice values are
periodically tiled onto the electronic lattice, using `R mod qgrid` and a
factor √(Nk/Nq), before the unchanged mode-7 electronic convolution. This
implements `Σ_k = (1/Nq) sum_q G_(k−q) W_q` at the coarse full-BZ quadrature
points. Zero-padding W(R) would instead interpolate W onto fine q and is
not this approximation. The host-plan route converts its q-space prepared
operand through the same FFT owner before/after this tiling; equal grids
retain the original preparation. The pole-store header records `grid`
(bosonic) and `electronic_grid` separately; old headers mean equal grids.

The cutoff does not guarantee passivity or lattice-window convergence.
Native validity gates remain in force, and a smaller grid must be checked
against Σ and the intended dynamics. The current implementation retains
the fine electronic Green transforms and R-space product; its saving is
the smaller χ/W storage, Dyson solves and shared-pole construction, not a
reduction of the electronic band sum or ζ fit.

This is fixed-centroid-convention, R-windowed RPA plus ordinary coarse-q
GW, not a matched conserving ring-Φ approximation. Differentiating a ring
functional of the windowed bubble would put the window's weighted adjoint
in Σ (central-image injection with boundary weights), rather than the
periodic tiling specified here. Do not infer energy conservation from a
Φ argument. Reciprocal coarse q pairs still shift fine k by exact grid
permutations, so fresh particle balance can follow the collision event
algebra independently; measure its trace residual and held-rate drift.
The window has signed Fourier weights and can create nonpassivity even
from a passive bubble. Such failures must not be relabelled as physical
gain. Dropping spatial reductions prevents false symmetry reconstruction
but does not restore centroid-home-cell gauge covariance: authenticate the
fixed convention and converge the cutoff.

`qgrid_response_window = fejer` is an explicitly selected experimental
alternative; the default remains `sharp`. On each reduced even response
axis m, set L=m/2 and multiply the real-space product by
`max(1-abs(R)/L,0)`. The old boundary images at ±L are now zero; nonreduced
axes retain weight one, reduced odd axes refuse, and equal grids use the
unchanged legacy FFT. The tensor-product Fourier kernel is nonnegative:
κ(θ)=|Σ(j=0..L−1) exp(ijθ)|²/L per reduced axis. Consequently the coarse
polarization is a normalized positive average of the fine-q polarization
in the fixed centroid convention. No eigenvalue is inspected or floored.
All native Gram, moment, held-response and model gates still apply; neither
this spectral property nor a tiny CPU oracle admits a material model or
guarantees positivity of the nonsecular electronic collision.

This option changes χ (including its moments and full-density response)
consistently, and binds its name into recipe/model identities. It does
**not** change Σ's ordinary coarse-q periodic tiling above. It therefore
remains a non-Φ response-smoothing approximation, not the matched weighted
adjoint of that response restriction. A future ring-functional derivative
would instead use `sqrt(Nk/Nq)*w(R)*W_R[R mod qgrid]` in Σ; it is not silently
enabled here. Compare full complex Σ and the intended dynamics against the
fine-grid reference, and keep fresh number balance separate from held-rate
drift and energy/Pauli claims.

## 6. Row passes and the `live` operand {#live-rows}

The Σ τ kernel and the χ₀ direct stream do not convolve a rank's whole
$(\mu_X, \nu_Y)$ tile at once. They split the rank's local left rows into
passes, each a union of whole centroid orbits
(`gw.subtile_stream.orbit_cuts`), so the unfold of one pass reads only that
pass's rows. The passes are equal-width windows run as one `lax.scan`, so the
program does not grow with the pass count; each pass's tables are cut from
the placed ones on the device (`gw.subtile_stream.window_load`). The window
width comes from `runtime.tiles.TILE_BYTES` and the shapes, never from free
memory.

Because the windows have equal width, the last one usually holds fewer real
rows. Modes 7, 8, 9 and 11 take an optional last operand `live`
(int32 `[2]`, replicated): the live left rows `[lo, hi)`. Every pass gathers
and transforms only the live columns; the pass that stores the output writes
zeros on the other rows (modes 7, 8, 9), and mode 11 adds nothing to them. A
call with `live` compiles its own program, so a call without it carries no
bounds in its registers. `live = [0, rows)` is bitwise equal to the call
without it. The plan and XLA backends zero the dead rows with `ffi.fft.live_row_mask`.

## 7. Scratch and the tile-table load {#tiles}

**Scratch.** Three split arms run their pairs in chunks through an
intermediate that XLA's scratch allocator grants per call:

| mode | intermediate | bound | refusal when not granted |
|---|---|---|---|
| 7 | `(N_k, chunk·d²)`, d the stored spin block; m7s adds the chunk's pair-major $W_R$, `16·N_k` B a pair, and rounds a chunk to whole waves of plane tiles | ≤ 1 GiB (`ffi.fft.klead_unfold_scratch_bytes`) | `GATE mathdx-kconv-unfold-scratch` |
| 8 | `(N_k, chunk·(n_s² + n_A n_B))` | ≤ the call's output size and ≤ 1 GiB | `GATE mathdx-kconv-lorentz-scratch` |
| 11 | `(N_k, chunk·2n_s²)` | `scratch_bytes`; default the smaller of one local parent-Green tile and 1 GiB, at least one pair (`ffi.fft.chi_unfold_scratch_bytes`) | `GATE mathdx-kconv-chi-scratch` |

A chunk only groups pairs into launches and chunks own disjoint outputs, so
the chunk size never changes a value. The compiled program's
`memory_analysis()` does not count this scratch, so callers price it
([memory model](memory-model.md#native-handlers)). Apart from these
intermediates the kernels allocate no device workspace beyond dynamic shared
memory and their static tile tables (`pinfo`, counted in the opt-in check).

**Tile-table load** (modes 7 and 11, single arm, $n_s = 4$, sm_80 and newer).
The plain load keeps one pair's gathered values $g$, $U_k$ and the right
action live per thread: $3n_s^2$ complex values, $12n_s^2$ registers. At
$n_s = 4$ that is 192 registers, which limits a 256-thread block to one per
SM. The tile-table load moves the tables into shared memory instead: each
block stages $U_k$ and the per-k source rows once; each tile of $t_p$ pairs'
`lsrc`/`rsrc` slices, phases and (mode 7) $V_R$ go by `cp.async` one tile
ahead into a second buffer; the gather is one `cp.async` per cell from shared
indices. A persistent grid of the resident blocks walks the tiles.
`tile_table_plan` picks $t_p$ from the device's shared memory per SM, the
per-block reservation and the opt-in maximum, with two to four blocks
resident: mode 11 takes the tile that keeps the most blocks resident, mode 7
the largest tile at two or more. The products and their order are those of
the register load, so the result is bitwise. It is used only when
$12n_s^2 > 64$: at $n_s = 2$ the register load (48 registers) already runs
three blocks per SM, and the tables would shrink the tile (CrI3 8×8,
$n_s = 2$: Σ τ 5.58 → 6.24 s with tables). Mode 11 with vertices keeps the
register load.

## 8. Resident-row modes 0, 1 and 6 {#resident-rows}

The pair modes keep three $N_k$-long banks per row (one `(col, μ)` entry):
$3\cdot16\cdot(N_k|1)$ bytes. A 256-thread block holds
$r_b = \min(16, \lfloor B/\text{row}\rfloor)$ rows with
$B = \min(100\ \text{KiB}, \text{opt-in})$; where that is 0, $r_b$ is what the
opt-in maximum holds. The budget is a fixed 100 KiB rather than the device's
opt-in maximum so that every device with at least that much opt-in memory
gets the same rows per block, and with them the same launch shape; it is
clamped where the opt-in maximum is smaller (99 KiB on sm_86/89/120). On an
A100 one row fits the opt-in memory up to $N_k = 3477$.

Above that, the router streams instead of refusing
(`ffi.fft.pair_resident_refusal` announces the switch once). For each spatial
tile of at most 32 × 64 = 2048 $(\mu,\nu)$ columns and each of the $n_s^2$
spin components, it forms that component's load for that tile only (the
parent tables or the plane gather as in the resident kernel), transforms it
with mode 3, multiplies and sums; one forward transform per tile finishes it.
No full-spin unfolded bank exists; each scalar-spin tile is
$(N_k, \le 32, \le 64)$, 262 MB at 20³. The result is the same contraction,
equal to the resident kernel at round-off.

## 9. Mode 1: the parent-load pair convolution {#mode-1}

Mode 1 is the pair convolution with its operands unfolded from the raw parent
k-points inside the load, so no full-k open-spin array is written. With
$p = \mathrm{irr}[k]$, $o = \mathrm{sym}[k]$, $m = \mathrm{left}[o,\mu]$,
$n = \mathrm{right}[o,\nu]$ and $\mathcal T_k$ complex conjugation when
$\mathrm{trs}[k] \ne 0$, the load builds

$$
P_{k,ab}(\mu,\nu) = \overline{\sum_{c,e}\mathrm{coef}[k,\,a n_s+b,\,c n_s+e]\;
\mathcal T_k\!\left(e^{2\pi i q_p\cdot \mathbf w^{\rm L}_{o,\mu}}\,D_{p,c,e}(m,n)\,e^{-2\pi i q_p\cdot \mathbf w^{\rm R}_{o,\nu}}\right)}
$$

from `D_l` with `coef_l` on the left and from `D_r` with `coef_r` on the
right, then runs mode 0 on $P^L$, $P^R$.

| operand | shape | dtype |
|---|---|---|
| `D_l`, `D_r` | `(n_parent, n_s, μ_local, n_s, ν_local)` logical | complex128 |
| `irr`, `sym` | `(N_k,)` | int32 |
| `left`, `right` (owner-local source maps) | `(n_ops, μ_local)`, `(n_ops, ν_local)` | int32 |
| `L`, `R` (lattice wraps $\mathbf w^{\rm L}$, $\mathbf w^{\rm R}$: the lattice vector each centroid image is shifted by) | `(n_ops, μ_local, 3)`, `(n_ops, ν_local, 3)` | float64 |
| `q` (parent fractional k) | `(n_parent, 3)` | float64 |
| `trs` (antiunitary mask) | `(N_k,)` | int32 |
| `coef_l`, `coef_r` (open-spin coefficients) | `(N_k, n_s², n_s²)` | complex128 |
| result `U` | `(N_k, μ_local, ν_local)` | complex128 |

- **Tables.** `isdf.core._parent_conv_tables_local` builds them from the typed
  unfold plan. The handler checks their shapes and dtypes; it cannot check
  the map values on the device, so they must be the plan's own tables.
- **Layout.** The static attribute `centroid_major` states the physical layout
  of `D`. Both `isdf.core` call sites set it: major-to-minor
  `(parent, ν, spin_r, μ, spin_l)`, requested through the `ffi_call` input
  layout `(0, 4, 3, 2, 1)`, so the GEMM that produces `D` feeds the kernel
  without a transpose. With it unset, `D` is row-major
  `(parent, spin_l, μ, spin_r, ν)`. Only the load's address arithmetic
  differs.
- **Vertex.** Production folds the post-unfold Lorentz vertex into `coef_r`
  (`isdf.core._parent_conv_vertices`, with the phase conjugated because the
  load returns a conjugate). The kernel's `perm`/`phase` attributes therefore
  stay the identity, and every channel of one shape reuses one executable.

## 10. Mode 10: the plane FFT with gather-on-load {#mode-10}

Route G (the G-space charge ζ fit) transforms planes whose occupied cells arrive as a compact cylinder
`F (…, n_col)`; `plane_from_col (n_b·n_c,)` names each flat cell's column
(`n_col` = empty). The call returns

$$
Y[\dots, k_b, k_c] = \sum_{b,c} P[\dots, b, c]\; e^{-2\pi i\,(b k_b/n_b + c k_c/n_c)},
$$

with $P$ the cylinder scattered by `plane_from_col` and zero elsewhere
(`jnp.fft.fftn(P, axes=(-2,-1), norm='backward')`), without writing $P$.
Persistent blocks each hold $P_B$ planes of $(n_b, n_c|1)$ in shared memory
($P_B \le 8$ planes within 64 KiB, else 1; the grid is capped at the resident
block count). A block gathers the occupied rows' cells through
`gidx (rows, n_c)` and `row_of (rows,)`, runs the row FFTs on those rows only,
runs the column FFTs on every column with dead rows read as zero, and stores
each plane once, coalesced. HBM traffic is one read of the cylinder and one
write of the plane.

The kernel is latency-bound rather than bandwidth-bound (A100, Nsight
Compute: one HBM pass at 36–40 % of peak). So when a `(rows, n_c)` staging
block per plane also fits the opt-in memory, the next group's cells are
gathered asynchronously (`src/ffi/cpp/common/lrx_async_gather.h`,
`cp.async`) while the current group's passes run: 1.11–1.37× at 25²–80² on
A100.

Every line FFT is a cuFFTDx thread FFT (at most 40 points). An axis
$n = n_1 n_2$ with $\gcd(n_1, n_2) = 1$ runs as the Good–Thomas
two-dimensional DFT: the input sits at $(n_2 i_1 + n_1 i_2) \bmod n$ and
output $(k_1, k_2)$ is $X[k]$ for $k \equiv k_1 \pmod{n_1}$,
$k \equiv k_2 \pmod{n_2}$, so the split needs index maps and no twiddle
factors. `ffi.fft.plane_fft_split` picks the most balanced coprime split with
both factors at most 40, or $(n, 1)$ for $n \le 40$ without one. Block FFTs
are not used because cuFFTDx's fp64 database lacks lengths such as 45, 54,
75, 90, 150 and 250, which would need Bluestein and a host-built workspace.

**ROCM** (`src/ffi/cpp/kdft/plane_gather_*`): the same kernel design and splits (`plane_fft_split`, factors ≤ 40), with three differences. Each factor line is one register Cooley–Tukey step of direct DFTs against the k-box stage's compile-time twiddles ($M = RQ$, $R$ the largest divisor ≤ $\sqrt M$; $M(R+Q)$ products; a prime factor is one direct DFT), the role cuFFTDx's thread FFT plays on CUDA. The gather is synchronous (gfx90a has no `cp.async` in this subset) and the grid is grid-stride, not persistent. $P_B$ is the most planes (≤ 8) within half a block's 64 KiB of LDS, else one plane within all of it (`ffi.fft.kdft_plane_fft_pb`; `GATE kdft-plane-residency` beyond, so 80² keeps the XLA route on ROCM). Measured per call on one GCD: 25² Fe slab 1.32×, 54² 1.31×, 40²–45² 2.4–3.3×, 32² 0.93× (kept on XLA).

`make_plane_fft_gather` decides once, at build, and announces the route.
Mode 10 serves a plane when both axes split and

$$
16\,n_b\,(n_c|1) + 5\,n_b + 8P_B + 16 \;\le\; \text{opt-in shared memory per block}
$$

(`ffi.fft.plane_resident_bytes`; the second and third terms are the kernel's
static row tables). Every other plane takes the XLA route: an axis with no
split (a prime above 40, or a prime power above 40 such as 41, 49, 64, 81,
121, 125, 128) or a plane too large. The largest square served is 100 on
sm_80/87 (163 KiB), 78 on sm_86/89/120 (99 KiB, so 80² takes the XLA route)
and 119 on sm_90/100 (227 KiB). A block that has its SM alone runs 512
threads (72² and larger on A100), otherwise 256 with two blocks per SM. `F`
must be complex128 (`GATE plane-fft-dtype`, on both routes) and
`plane_from_col` in `[0, n_col]`. `fn(F, start, size)` transforms the slab
`F[:, start:start+size]` of `F (A, S, …, n_col)` in place, so the ζ loop's
group slice is not copied. A direct handler call past either test refuses
(`GATE mathdx-plane-split`, `GATE mathdx-plane-residency`).

## 11. The BSE outer-product load {#bse-outer}

The BSE W term needs $U = s\,F_k(F^{-1}T\cdot W_R)$ with
$T[k,a,\mu,b,\nu] = \sum_K \Lambda[k,a,\mu,K]\,\Xi[k,K,b,\nu]$ ($a$, $b$ spin
components; the factory's operands `L` and `R`). `make_local_kconv_klead_outer`
forms $T$ in the k-box bank on the fp64 tensor cores (`mma.m8n8k4.f64`,
sm_80 and newer) and never stores it; the transforms, the multiply and the
store are mode 2's. K is zero-padded to a multiple of 4 (exact), and
`conj_r` reads $\overline{\Xi}$, which avoids a conjugated copy of the
wavefunction leg. Its K sum reproduces XLA's batched ZGEMM of the same
contraction bit for bit on A100. It needs a 64-column bank,
$64\cdot16\cdot\big((n_{kx}n_{ky}(n_{kz}|1))|1\big)$ bytes, within the
opt-in memory (`ffi.fft.klead_outer_refusal`).

`make_local_kconv_klead_outer_decode` also forms the decode's
$A[k,c,b,\nu] = \sum_{a,\mu}\overline{\psi_c[k,c,a,\mu]}\,U[k,a,\mu,b,\nu]$ ($\psi_c$ the conduction wavefunctions at the centroids) in the store,
so $U$ never reaches HBM: one resident block per SM, two 8-warp groups
alternating on two banks, partial sums added in a fixed phase order
(deterministic). It needs the two banks within the opt-in memory and the
per-lane accumulator $\lceil N_k/16\rceil\cdot\lceil n_c/8\rceil \le 8$ m8n8
blocks (`ffi.fft.klead_outer_decode_refusal`). A caller that gets a reason
from either refusal function keeps the next route down (outer load with an
XLA decode, or the XLA encode with mode 2); the route order and its
announcement are on the [BSE page](bse.md#the-matvec). Both handlers address
each operand with 32-bit offsets from per-tile 64-bit bases, so every
operand of one call holds fewer than $2^{31}$ elements; the BSE W term
splits its $(\mu, \nu)$ tile into blocks that keep each call below that and
its stored U within the fixed tile (`bse_stack_matvec.plan_w_blocks`).
`LORRAX_BSE_OUTER_KSUM=fma` runs both K sums on the fp64 FMA pipe instead of
the tensor cores, for comparing the two on devices where their rates differ.

**ROCM** (`src/ffi/cpp/kdft/kconv_outer_rocm_ffi.cc`, kernels in `kconv_outer_kernel.h` on the
same k-box stage, line transform the direct DFT): the CUDA kernels' work items, V-reuse order,
decode phases (`plan_phases`), slots and item counter, with three differences. The K sum and the
decode run on the FP64 FMA pipe (gfx90a has no spelling of the m8n8k4 fragment contract), so U and
A agree with the XLA chain to round-off (gate `tests/test_kdft_outer_gate.py`, 1e-12; measured
≤ 1.5e-15), K is not padded, and the decode takes $\psi_c$ as the BSE holds it (`prep` is the
identity). The tile is the largest XB × YB ≤ 8 × 8 box of padded columns within half a block's
64 KiB of LDS, else all of it (`ffi.fft.kdft_outer_tile`; an 8 × 8 bank is 66.5 KB at 8×8×1, so
that grid takes a 2 × 8 (x × y) box at half the LDS). One block-wide group (no ping-pong). The rules:
axes ≤ `KDFT_OUTER_AXIS_MAX` = 40 (`GATE kdft-axis`), one padded column within the LDS
(`GATE kdft-kconv-outer-tile`), and for the decode at most `KDFT_OUTER_ACC_MAX` = 32 complex
accumulators per thread at 256 or 512 threads (`GATE kdft-kconv-outer-decode-tile`;
`ffi.fft.kdft_outer_refusal`, `kdft_outer_decode_refusal`). A library without the handlers is
refused by name (`klead_outer_refusal`), and the BSE then announces the XLA encode with kdft mode 2.
Routing is measured (one MI250X GCD, the record beside `ffi.fft.KDFT_OUTER_BOX_COLS`): the outer load
beats the XLA encode + kdft mode 2 only where the CUDA kernel's 8 × 8 box fits half the LDS
(1.37× at 5×5×1); a smaller box re-reads the R leg per x block from L2 (0.74× at 8×8×1 with
μ = ν = 724, 1.07× at 4³), so such grids keep the unfused chain (`GATE kdft-kconv-outer-box`). The
fused decode measured 0.33–0.60× the outer load + XLA decode einsum, so the BSE keeps the XLA decode
(`GATE kdft-kconv-outer-decode-slower`). `LORRAX_KDFT_OUTER=all` routes both wherever the kernels
fit (the gate's and the bench's A/B). Beating the chain at $N_k \ge 64$ needs a denser load: the
kdft register pass (lines in VGPRs, LDS only for the axis exchange) or an MFMA K sum.

## 12. Cost {#cost}

Each transform costs $O(n_{\rm col}N_k\log N_k)$ flops. HBM traffic is one
read of each operand and one write of the result; the split arms of modes 7,
8 and 11 add one write and one read of their chunk intermediate. Modes 6–9
and 11 read the producer's own buffer (the plane FFT output, the parent
Green, the wedge), so the phased, split or unfolded copy that modes 1 and 2
would need is never written. Their gather still reads one source element per
full-k element: parent tiles are $n_{\rm parent}/N_k$ of the full-k memory,
not of the traffic.

## 13. Refusals {#refusals}

| refusal | raised at | condition | fix |
|---|---|---|---|
| `GATE mathdx-headers` | startup (`mathdx_root`); kernel build | no importable `nvidia.mathdx` with `include/cufftdx.hpp` | install the wheel; the `cuda12`/`cuda13` extras of `pyproject.toml` pin `nvidia-mathdx==25.6.0` |
| `GATE kconv-target` | startup, factory | the loaded library lacks the selected target | rebuild the CUDA leg from this tree |
| `GATE cuda-sm80` | startup | a CUDA device below compute capability 8.0: the library's PTX is `compute_80` and the kernels use `cp.async` | an A100 or newer, or CPU |
| `GATE mathdx-probe` | startup, as a recorded demotion | the mode-3 probe kernel fails to compile or run on this device; every factory takes the XLA backend | a wheel whose cuFFTDx supports the device's compute capability |
| `GATE plan-engine` | startup (cpu) | the host FFTW3-ABI FFT or batched GEMM does not load or computes a wrong 2-point/1×1 result | the engine libraries `liblorrax_ffi_host.so` was linked against on `LD_LIBRARY_PATH` |
| `GATE kconv-kgrid` | factory | the k grid is not three positive axes | pass the run's `(n_kx, n_ky, n_kz)` |
| `GATE mathdx-kconv-axis` | handler | a direct mathdx call with a k-grid axis above 40 (`KCONV_AXIS_MAX`); the factories send such grids to the XLA backend | call through the factories |
| `GATE mathdx-kconv-residency` | kernel build | a direct call of mode 0, 1 or 6 whose row exceeds the opt-in memory; the router streams such grids instead ([§8](#resident-rows)) | call through the router |
| `GATE mathdx-kconv-kbox-residency` | kernel build | the tile, plane or pencil the launch rule picked exceeds the opt-in memory (modes 2, 3, 8, 9); one whole column exceeds it (modes 4 and 5, which have no split arm); a split plane tile of one spin group exceeds it (mode 7: $n_s = 4$ at 26³ and larger on A100) | none at the factories, which send these grids to the XLA backend ([router](#router)); reaching it means `mathdx_refusal` and the handler disagree |
| `GATE mathdx-kconv-chi-q` | apply | a q-row store (`rows=`) on a grid whose image is not the m11f arm | none at the factory, which allows `rows=` only where `ffi.fft.chi_q_store_ok` holds; reaching it means that rule and `build()` disagree |
| `GATE mathdx-kconv-chi-residency` | kernel build | mode 11 cannot hold the grid | none here: the router sends such grids to the XLA backend (`mathdx_refusal`), so this gate means that rule and the handler's disagree, which is a bug |
| `GATE mathdx-kconv-unfold-scratch`, `-lorentz-scratch`, `-chi-scratch` | apply | XLA's scratch allocator refuses the split-arm intermediate of mode 7, 8 or 11 ([§7](#tiles)) | for mode 11, a smaller `scratch_bytes` |
| `GATE mathdx-kconv-outer-tile`, `-outer-decode-tile`, `-outer-arch`, `-outer-rank` | kernel build | the BSE outer load's bank or accumulator does not fit, the device is older than sm_80, or K is not a multiple of 4 | call through the factories, which check `klead_outer_refusal` / `klead_outer_decode_refusal` and pad K |
| `klead outer conv` / `klead outer decode`: an operand holds ≥ 2³¹ elements | handler | one call's U, legs, fragment tiles, W block or A reaches $2^{31}$ elements (32-bit offsets) | call through `bse_stack_matvec`, whose `plan_w_blocks` keeps every call's operands below the cap |
| `GATE mathdx-plane-split`, `mathdx-plane-residency` | kernel build | a direct mode-10 call on a plane with no split or too large | call through `make_plane_fft_gather` |
| `GATE plane-fft-dtype` | trace | mode 10's `F` is not complex128 | pass complex128 |
| `GATE kdft-plane-split`, `kdft-plane-residency`, `kdft-plane-slower` | factory build (ROCM) and kernel build | mode 10 on ROCM: an axis with no split into factors ≤ 40, a plane beyond a block's 64 KiB of LDS, or an axis measured slower than the XLA route (`KDFT_PLANE_FFT_SLOW_AXES`) (`ffi.fft.kdft_plane_fft_refusal`, `rocm_plane_fft_target`) | none: the factory keeps the XLA route, one warning per plane |
| `GATE kdft-kconv-parent-tile` | factory build (ROCM) and kernel build | mode 1's output column ($2n_s^2$ padded columns) does not fit a block's 64 KiB of LDS (`ffi.fft.kdft_parent_refusal`) | none: the factory keeps the XLA route, one warning per grid |
| `GATE kdft-kconv-outer-box`, `kdft-kconv-outer-decode-slower` | factory build (ROCM) | the measured routing rule: the outer load's 8 × 8 box exceeds half a block's LDS, or the fused decode (always; measured slower) | none: the BSE keeps the next route down, one warning per grid; `LORRAX_KDFT_OUTER=all` is the A/B |
| `GATE kdft-kconv-outer-tile`, `kdft-kconv-outer-decode-tile` | factory build (ROCM) and kernel build | the ROCM outer load's padded column does not fit a block's 64 KiB of LDS (nk above about 4000), or the fused decode needs more than 32 accumulators per thread (`ffi.fft.kdft_outer_refusal`, `kdft_outer_decode_refusal`) | none: the BSE keeps the next route down (outer load + XLA decode, or the XLA encode + kdft mode 2), one warning per grid |
| `GATE kdft-kbox-residency` | factory build (ROCM) and kernel build | mode 7's spin group (two columns at $n_s = 1$) does not fit a block's 64 KiB of LDS on the k-box single arm (`ffi.fft.kdft_kbox_refusal`, the handler's `kbox_plan`); the split arm is not ported | none: the router keeps the XLA unfold + kdft route, one warning per grid |
| `GATE kdft-kbox-residency` (modes 9, 11) | factory build (ROCM) and kernel build | neither k-box arm holds the grid: the single arm's spin group ($n_l n_r$ columns, mode 9) or Green pair ($2 n_s^2$, mode 11) exceeds a block's 64 KiB of LDS and so does the split arm's plane tile of one group ($n_l n_r$, resp. $n_s^2$, columns of a padded $(k_y, k_z)$ plane; e.g. $n_s = 4$ at 16³) (`ffi.fft.kdft_kbox_wedge_refusal` / `kdft_kbox_chi_refusal` via `kdft_kbox_arm`, the handler's `plan()`) | none: the factory keeps its XLA route, one warning per grid |
| `GATE kdft-nk` (modes 9, 11) | factory build (ROCM) | nk above `ffi.fft.KDFT_KBOX_RSPACE_NK_MAX` = 10240, the k-lead family's verified range (gated at 40×16×16) | none: the factory keeps its XLA route, one warning per grid |
| `GATE kdft-kbox-chi-2d` | factory build (ROCM) | mode 11 on a 2-D k-grid with an axis above `ffi.fft.KDFT_KBOX_CHI_2D_AXIS_MAX` = 24: measured 0.72–0.91× the XLA route (32×32×1, 40×8×1, 40×40×1; one MI250X GCD, 128×128 tile) | none: the factory keeps its XLA route, one warning per grid |
| `GATE kdft-kbox-build` (modes 9, 11) | trace (ROCM) and kernel build | the image of a routed grid did not compile or load (the factory's prebuild, `ffi.fft._kbox_prebuild`, raises the handler's reason) | a defect: rebuild the leg from this tree |
| `GATE kdft-kbox-chi-scratch` | kernel run (ROCM) | XLA's scratch allocator refused mode 11's split-arm intermediate (`scratch_bytes`, at least one pair's $N_k \cdot 2 n_s^2$ c128) | pass a smaller `scratch_bytes` |
| `GATE kdft-kbox-build`, `kdft-kbox-operands`, `kdft-kbox-launch` | kernel build, apply | the hipRTC compile, the operand contract or the launch of `lorrax_kdft_kconv_klead_unfold`, `_kfft_klead_unfold`, `_kconv_chi_unfold` or `_kconv_chi_vertex` failed | a build defect |
| `k-leading unfold conv: …` (and `lorentz conv`, `chi unfold`, `unfold fft`) | factory; apply | tables cut for another mesh, operands whose parent count or endpoint widths differ from the tables, or a missing partner on a plan with antiunitary rows | build the tables from the same plan and mesh as the operands; pass the partner |

A kernel-build failure (an NVRTC compile, missing toolkit headers, a module
load, a residency gate) is sticky: the handler caches it per build key and
returns it on every later call, naming the stage
(`kconv_mathdx (fused cuFFTDx k-convolution): <stage> failed -- …`).

## 14. Build and image key {#build-and-cache}

**In-process build.** NVRTC compiles one image for the device's own
`sm_<cc>` per (CUDA context, mode, $n_{kx}$, $n_{ky}$, $n_{kz}$, $n_s$, right
width, precision, variant) and keeps it in an in-process cache. Modes 0–9 and
11 share one embedded source (`kSrc`); mode 10 has its own (`kPlaneSrc`); the
k-box modes also embed `kbox_stage.cuh` as a named header, turned into text
at configure time (`kbox_stage_src.h.in`). The BSE outer targets
(`kconv_outer_cuda_ffi.cc`) embed the same header in their own programs.

**Headers.** The router passes the wheel's `nvidia/mathdx` directory to every
handler as the string attribute `mathdx_root`; NVRTC includes its headers and
the CUDA toolkit's found beside the loaded `libnvrtc`. No environment variable
names either path. Building `liblorrax_ffi.so` needs no mathdx: the
translation unit links `libnvrtc` and resolves the driver API by `dlsym`. The
CUDA leg always compiles this family; a missing `cufft.h`, `libcufft`,
`nvrtc.h` or `libnvrtc` fails the configure.

**What decides an image.** Modes 0–9 and 11: the mode, the k-grid, $n_s$,
the right width (mode 9), the precision, the variant (mode 7: the output spin
block; mode 8: the vertex widths $8n_a + n_b$; mode 11: completion and
vertices) and the `live` bit of a windowed pass, plus the device's compute
capability and shared-memory limits (the `Key` tuple and `build()`'s `-D` options in
`kconv_mathdx_cuda_ffi.cc`; every arm choice is a function of them). Never $N_\mu$, the band
counts or P. Two kinds depend on the system instead of the k-grid: mode 10 on the FFT plane
$(n_b, n_c)$ and its occupied rows (`build()`'s mode-10 branch), and the BSE outer kernels on the
band rank $K = \min(n_c, n_v)$ padded to 4 and, for decode, $\lceil n_c/8 \rceil$ c-blocks (the
build functions of `kconv_outer_cuda_ffi.cc`). The image key hashes the
kernel text, comments included, so measurement notes live at the Python
owners, not in the kernel text.

The NVRTC build service, its key rule, the cubin cache, the release store and
the build cost are owned by
[compilation §2](compilation.md#2-native-kernels-and-nvrtc-images), and the
build log lines by its §7.

## 15. Numerical contract {#numerical-contract}

- **Dtype.** Modes 2–5 take all-complex128 or all-complex64 operands (the
  complex64 image serves the fp32-GMRES BSE arm) and never cast; every other
  mode is complex128 only. The host plan handlers are complex128 only, so a
  complex64 operand refuses at trace time on a cpu mesh; the XLA backend takes
  either precision.
- **XLA k transforms.** On the XLA backend a k-grid of at most the leg's
  cap (`ffi.fft.XLA_KDFT_MAX_NK`: ROCM and cpu 144; CUDA none until its
  crossover is measured, so CUDA's XLA route is unchanged) is transformed as one GEMM with the dense
  flat-k DFT matrix on the axis where k sits (`ffi.fft._kdft`), not with
  `jnp.fft`, whose XLA FFT op on minor-most dims costs a transpose each way
  around rocFFT/cuFFT. The two agree to round-off (relative 1e-15 at
  complex128 on MI250X up to 12×12×1), not bitwise; above the cap `jnp.fft`
  is unchanged.
- **In place.** Modes 2, 3 and 5, and mode 4 with `out_layout=0`, alias
  operand 0 to the result; mode 11 aliases `acc`. This is safe because each
  block reads all $N_k$ values of its columns before it stores any of them.
- **Products.** Modes 7, 8, 9 and 11 form the phased gather, $U g U^\dagger$
  and the χ trace with fused products: a complex product is two FMAs and two
  multiplies (`lrx_mulf`), a product-sum term four FMAs (`lrx_cmac`). They
  agree with the XLA unfold chain they replace at round-off, not bitwise.
  Mode 6 forms $F\cdot D$ as XLA's unfused complex multiply, so it equals the
  XLA chain it replaces bit for bit. Mode 8 rounds as mode 9 on W followed by
  the $V_R$ call, bit for bit.
- **Layout independence.** The single-pass and split arms run each line's
  thread FFT on the same inputs in the same axis order, so their transforms
  agree bit for bit, and a scratch chunk only groups pairs into launches, so
  its size never changes a value.

## 16. Adding a mode {#adding-a-mode}

1. Add a kernel entry to the embedded source under its `LRX_MODE` value (on
   the k-box stage, a Load, a Mid and a Store).
2. Add a handler and its `XLA_FFI_DEFINE_HANDLER_SYMBOL` in the same
   translation unit. Register the target in `ffi_loader._CUDA_TARGET_SYMBOLS`
   and in `ffi.fft.KCONV_TARGETS`, which `require_kconv` checks at startup. A
   changed operand contract gets a new target; changing an existing target's
   signature is an ABI bump ([FFI layer §8](ffi_layout.md#8-hard-invariants)).
3. Add a router factory in `src/ffi/fft.py` that returns the mathdx call on
   CUDA and the reference composition on the plan and XLA backends, re-export it from
   `common.fft_helpers`, and add a case to `tests/test_kconv_xla_gate.py`.
4. Put the kernel's measured record (its speedup over the plain-XLA route,
   what was tried and did not pay) in the comment block above its Python
   factory, not in the kernel source, whose text is part of the cubin key.
   The [FFI layer's kernel catalog](ffi_layout.md#kernel-catalog) lists those
   blocks.
