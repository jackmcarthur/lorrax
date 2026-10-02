# The k-convolution family

Every k-axis convolution and k-axis transform in LORRAX goes through one
router in `src/ffi/fft.py`: the χ₀ time node, the Σ = G⋆W product, the ζ-fit
pair Gram, the BSE W term and the route-G plane transform. This page explains
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
| BSE W term | $(F^{-1}T)(R)\cdot W(R)$, $T = \sum_K L\,R$ | `bse.bse_stack_matvec`, `bse.bse_ring_comm` |

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
164 KB of shared memory per A100 SM, so the tile streams through HBM either
way. Measured at Ni 20³, P64, one Σ τ row pass: a staged route (an XLA gather
of the unfolded Green, a vendor batched 3-D FFT, the product, the forward FFT
and the parent gather) is 1.7–2.1× slower than modes 9 + 7 and holds
2.5–3.2 GB live, against the fused route's ≤ 1 GiB scratch. It wins only on
launch-bound 4³ grids (1.19×).

## 2. The router {#router}

The factory picks the backend from the mesh platform alone
(`ffi.fft.kconv_backend`):

| mesh platform | backend |
|---|---|
| CUDA | the nvidia-mathdx kernel family in `liblorrax_ffi.so`: `src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc`, and `kconv_outer_cuda_ffi.cc` for the BSE outer-product load |
| cpu | the plan route: the FFTW3-ABI host handlers of `liblorrax_ffi_host.so` (`lorrax_mklfft_flat_k`, `lorrax_mklfft_gw_conv`, [FFI layer §3c](ffi_layout.md#3c-which-fft-engine-the-host-library-binds)) composed with XLA elementwise work |
| any other | refusal, `GATE kconv-platform` |

No environment variable or deck key selects a route, because a routing choice
is policy and policy changes only through the deck
([decisions](decisions.md), 2026-09-24). Both backends return the same
callable contract, so a consumer never branches on the backend. The cpu leg is
the reference composition of each mode: the same unfolds and products in XLA,
with host transforms.

**Startup.** `runtime.initialize_communicator_stack` calls
`ffi.fft.require_kconv(mesh)`. On CUDA it finds the wheel (`mathdx_root`),
probes every target in `ffi.fft.KCONV_TARGETS` in the loaded library, and
compiles and runs one probe kernel (mode 3 on a 2×1×1 grid). A device that
the installed cuFFTDx cannot compile for therefore refuses at startup
(`GATE mathdx-probe`, naming its compute capability and the wheel version)
instead of at the first convolution. On cpu it requires the host flat-k
target. Each factory re-probes its own target when it is built, and checks
operand shapes and dtypes at trace time.

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
  the rest, once per T. `W_prep` is in the backend's own form (R space on
  CUDA, W unchanged on cpu, whose `gw_conv` handler transforms W itself), so
  pass it only to the `apply` of the same pair.
- **Vertices (modes 0, 1 and 6).** `perm_l`, `perm_r` are permutations of
  `range(n_s)`; `phase_l`, `phase_r` are exact monomials in $\{+1,+i,-1,-i\}$,
  encoded 0–3. Anything else refuses at factory time; $n_s \le 4$.

## 3. Factories and modes {#modes}

One handler file serves modes 0–11, each a value of the compile-time
`LRX_MODE` of one embedded source. A target string is `lorrax_mathdx_<suffix>`;
`ffi_loader._CUDA_TARGET_SYMBOLS` maps it to its C++ handler symbol.

| mode, target suffix | factory (`ffi.fft`) | computes | operands → result | cpu leg |
|---|---|---|---|---|
| 0 `kconv_pair` | `make_fused_conv_kpair` | $U = s\,F_k\sum_{ab}\phi_l[a]\phi_r[b]\,\overline{F^{-1}A_{a,b}}\cdot F^{-1}B_{\pi_l a,\pi_r b}$ | `A`, `B` `(n_kx, n_ky, n_kz, n_s, col, μ, n_s)` → `U` `(n_kx, n_ky, n_kz, col, μ)` | two host inverse transforms, the spin sum in XLA, one host forward transform |
| 1 `kconv_parent` | `make_fused_conv_kparent` | mode 0 with both operands unfolded from raw parents on the load ([§9](#mode-1)) | `D_l`, `D_r` `(n_parent, n_s, μ, n_s, ν)` and ten tables → `U` `(N_k, μ, ν)` | the typed parent load in XLA (a full-k `(N_k, n_s, μ, ν, n_s)` copy per side), then mode 0's composition |
| 2 `kconv_klead` | `make_kconv_klead` (`apply`), `make_local_kconv_klead` | $U = s\,F_k(F^{-1}T\cdot V_R)$, $V_R$ already in R space | `T`, `U` `(N_k, a, m_x, b, m_y)`, `V_R` `(N_k, m_x, m_y)` | `lorrax_mklfft_gw_conv` (`make_kconv_klead`), host transforms (`make_local_kconv_klead`) |
| 3 `kfft_klead` | `make_kfft_klead`, `make_local_kfft_klead`; `prep` of `make_kconv_klead` | $Y = s\,F^{\pm}_k X$, k leading | `(N_k, rows)` | `lorrax_mklfft_flat_k` |
| 4 `kconv_kminor` | `make_kconv_kminor`, `make_local_kconv_kminor` | $U = s\,F_k(F^{-1}X\cdot K_R)$, k trailing, $K_R$ made by mode 5 | `X` `(d0, d1, d2, d3, d4, N_k)`, `K_R` `(d1, d2, N_k)` → X's layout (`out_layout=0`) or `(d0, N_k, d3, d1, d4, d2)` (`out_layout=1`) | XLA moves k to the front, host transforms, k moves back |
| 5 `kfft_kminor` | `make_kfft_kminor`, `make_local_kfft_kminor` | $Y = s\,F^{\pm}_k X$, k trailing | `(rows, N_k)` | the same transpose around one host transform |
| 6 `kconv_plane` | `make_fused_conv_kplane` | mode 0 read from the route-G D-plane FFT output: Bloch phase `F[k,g,p]` applied and the `2c` axis split into L = slots `[0, c)` and R = slots `[c, 2c)` on the load | `D` `(N_k, g, n_s, 2c, n_s, p)`, `F` `(N_k, g, p)` → `U` `(N_k, c, g·p)` | phase, split and transpose in XLA, then mode 0's composition |
| 7 `kconv_klead_unfold_xblock` | `make_kconv_klead_unfold` | mode 2 on the typed unfold of the raw-parent Green ([§5](#unfold-on-load)) | `G`, `Gt` `(n_parent, μ, n_s, ν, n_s)`, `V_R` `(N_k, μ, ν)` → `U` `(n_out, n_s, μ or an x block, n_s, ν)` | `symmetry_maps.apply_unfold_load_tables_local` (a full-k copy), mode 2's composition, the row selection |
| 8 `kconv_klead_lorentz_wparent` | `make_kconv_lorentz_unfold` | the four-current Σ: $U = m\,F_k\sum_{AB}\gamma_A(F^{-1}\hat G)\gamma_B^\dagger\circ\hat W_R[k,x,A,y,B]$, with $\hat G$ and $\hat W$ both unfolded from parents on the load | `G`, `Gt` as mode 7; `W`, `Wt` `(n_{q,parent}, μ, n_A, ν, n_B)`, $n_A, n_B \le 4$ → `U` `(n_out, n_s, μ, n_s, ν)` | both unfolds and the γ block sum in XLA, host transforms |
| 9 `kfft_klead_unfold` | `make_kfft_klead_unfold` | $Y_k = s\,F^{-1}_k(L_k\hat O_k R_k^\dagger)$: the R-space operand of modes 2 and 7, read from an interaction's q wedge | `W`, `Wt` `(n_wedge, μ·n_l, ν·n_r)` → `Y` `(N_k, μ·n_l, ν·n_r)` | the unfold in XLA, then mode 2's `prep` |
| 10 `plane_fft_gather` | `make_plane_fft_gather` (`LocalFourierPlan(in_gather=…)`) | the route-G plane FFT, gathered on load ([§10](#mode-10)) | `F` `(…, n_col)` → `Y` `(…, n_b, n_c)` | the XLA route: a static-run concatenate, then `jnp.fft.fftn` |
| 11 `kconv_chi_unfold`, `kconv_chi_vertex` | `make_kconv_chi_unfold`, `make_kconv_chi_vertex` | one χ₀ τ node: $acc[o] \mathrel{+}= \alpha_o\sum_{ab}\overline{(F^{-1}\hat G^c)_{ab}}(F^{-1}\hat G^v)_{ab}$ (+ its conjugate when `complete`) | `Gv`, `Gc` and partners `(n_parent, μ, n_s, ν, n_s)`, `α` `(n_out,)` → `acc` `(n_out, N_k, μ, ν)`, in place | the unfold per Green in XLA, host inverse transforms, the trace in XLA |
| 2 + outer load, `kconv_klead_outer` | `make_local_kconv_klead_outer` | mode 2 with $T = \sum_K L\,R$ formed in shared memory ([§11](#bse-outer)) | `L` `(N_k, a, m_x, K)`, `R` `(N_k, K, b, m_y)`, `V_R` `(m_x, m_y, N_k)` → `U` `(N_k, a, m_x, b, m_y)` | the einsum for T, then mode 2's composition |
| 2 + outer load + decode, `kconv_klead_outer_decode` | `make_local_kconv_klead_outer_decode` | the same, with $A = \sum_{a,x}\overline{P_c}\,U$ formed in the store | → `A` `(N_k, n_c, b, m_y)` | the outer composition and the decode einsum |

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

The library also exports four mathdx targets that no current factory calls
(`kconv_klead_unfold_rows`, `kconv_klead_lorentz_rows`,
`kconv_klead_unfold_block`, `kconv_klead_lorentz_conj`). They stay because a
target string never changes its operand contract: a new contract is a new
target, so source trees built against the old ones keep loading
([FFI layer §8](ffi_layout.md#8-hard-invariants)).

## 4. The k-box stage {#kbox}

Every mode except 0, 1, 6 and 10 runs on one stage,
`src/ffi/cpp/cufft/kbox_stage.cuh`. A mode supplies three pieces:

- **Load**: from HBM into the shared-memory bank. A plain copy (modes 2–5),
  the typed unfold gather (modes 7, 8, 9, 11), or the outer product (the BSE
  load).
- **Mid**: the R-space step between the transforms: the product with $V_R$,
  the vertex sum, or the spin trace.
- **Store**: the scaled write, through a row map where the mode keeps only
  some k rows.

The stage owns the transforms and the launch rule. A transform runs cuFFTDx
thread FFTs along z, then y, then x; the forward transform after the Mid
uses the same axis order.

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
  `min_tr = 2` protects the 128-byte runs of a copy load; a gathered load
  (modes 4, 5, 7, 8, 9, 11) has no run to protect and passes `min_tr = 1`.

Two single-buffered blocks per SM are preferred over one double-buffered
block because they measured faster for the convolution modes. Each line sees
the same thread FFT on the same inputs in the same axis order in either arm,
so the arms agree bit for bit.

| mode | single arm | split arm |
|---|---|---|
| 2, 3 | tiles of whole columns | plane and pencil passes through the output, in place |
| 4, 5 | one column per block at least | none: a column larger than the opt-in memory refuses |
| 7 | tiles of whole $n_s^2$ groups of two or more columns | chunks of pairs through scratch ([§7](#tiles)): a gather plane pass on tiles of the most whole groups the opt-in memory holds; then, if one padded column fits a block (20³: 134 of 163 KB on A100), one column-resident pass (x pencil, $V_R$, forward transform, store), else an x pencil, a forward plane pass and a pencil-and-store pass |
| 8 | tiles of whole $n_s^2$ groups plus the pair's $n_A n_B$ W columns | plane and group-pencil passes, chunked over pairs through scratch |
| 9 | tiles of whole $n_l n_r$ groups | plane tiles of whole groups, then the x pencil in place |
| 11 | $t_r$ whole pairs ($2n_s^2$ columns each) | plane passes on 16-column tiles, then a warp-shuffle x pencil per pair (it stages nothing), chunked over pairs through scratch |

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
a pole field read from its q wedge, with the endpoint actions $L_k$, $R_k$
(1 for a scalar interaction, the Lorentz rotation for a current block).

**Mode 11** serves three χ₀ streams of `gw.w_isdf`: the
identity-vertex step response; the selected-q charge streams of the response
bank (direct, retarded and KMS static), where each node's correlation is
transformed once and its rows at $q$ and $-q$ combine per pair mode (retarded
$-i(r_q - r_{-q})$, KMS static $-(r_q + r_{-q})$); and the four-current
direct stream through `make_kconv_chi_vertex`, with at most three monomial
vertices per side and an optional per-k sign `sign_c` for a Dirac-half
quadrant. Where mode 11 cannot hold the grid (`ffi.fft.chi_unfold_refusal`),
`gw.w_isdf` keeps the full-k Green route (mode 3 on unfolded Greens, the
trace in XLA) and announces that once.

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
without it. The cpu leg zeroes the dead rows with `ffi.fft.live_row_mask`.

## 7. Scratch and the tile-table load {#tiles}

**Scratch.** Three split arms run their pairs in chunks through an
intermediate that XLA's scratch allocator grants per call:

| mode | intermediate | bound | refusal when not granted |
|---|---|---|---|
| 7 | `(N_k, chunk·d²)`, d the stored spin block | ≤ 1 GiB | `GATE mathdx-kconv-unfold-scratch` |
| 8 | `(N_k, chunk·(n_s² + n_A n_B))` | ≤ the call's output size and ≤ 1 GiB | `GATE mathdx-kconv-lorentz-scratch` |
| 11 | `(N_k, chunk·2n_s²)` | `scratch_bytes`; default the smaller of one local parent-Green tile and 1 GiB, at least one pair (`ffi.fft.chi_unfold_scratch_bytes`) | `GATE mathdx-kconv-chi-scratch` |

A chunk only groups pairs into launches and chunks own disjoint outputs, so
the chunk size never changes a value. The compiled program's
`memory_analysis()` does not count this scratch, so callers price it
([memory model](memory-model.md#native-handlers)). Apart from these
intermediates the kernels allocate no device workspace beyond dynamic shared
memory.

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
opt-in maximum holds. On an A100 (166 912 B opt-in) one row fits up to
$N_k = 3477$.

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
\mathcal T_k\!\left(e^{2\pi i q_p\cdot L_{o,\mu}}\,D_{p,c,e}(m,n)\,e^{-2\pi i q_p\cdot R_{o,\nu}}\right)}
$$

from `D_l` with `coef_l` on the left and from `D_r` with `coef_r` on the
right, then runs mode 0 on $P^L$, $P^R$.

| operand | shape | dtype |
|---|---|---|
| `D_l`, `D_r` | `(n_parent, n_s, μ_local, n_s, ν_local)` logical | complex128 |
| `irr`, `sym` | `(N_k,)` | int32 |
| `left`, `right` (owner-local source maps) | `(n_ops, μ_local)`, `(n_ops, ν_local)` | int32 |
| `L`, `R` (lattice wraps) | `(n_ops, μ_local, 3)`, `(n_ops, ν_local, 3)` | float64 |
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

Route G transforms planes whose occupied cells arrive as a compact cylinder
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
$T[k,a,x,b,y] = \sum_K L[k,a,x,K]\,R[k,K,b,y]$. `make_local_kconv_klead_outer`
forms $T$ in the k-box bank on the fp64 tensor cores (`mma.m8n8k4.f64`,
sm_80 and newer) and never stores it; the transforms, the multiply and the
store are mode 2's. K is zero-padded to a multiple of 4 (exact), and
`conj_r` reads $\overline{R}$, which avoids a conjugated copy of the
wavefunction leg. Its K sum reproduces XLA's batched ZGEMM of the same
contraction bit for bit on A100. It needs a 64-column bank,
$64\cdot16\cdot\big((n_{kx}n_{ky}(n_{kz}|1))|1\big)$ bytes, within the
opt-in memory (`ffi.fft.klead_outer_refusal`).

`make_local_kconv_klead_outer_decode` also forms the decode's
$A[k,c,b,y] = \sum_{a,x}\overline{P_c[k,c,a,x]}\,U[k,a,x,b,y]$ in the store,
so $U$ never reaches HBM: one resident block per SM, two 8-warp groups
alternating on two banks, partial sums added in a fixed phase order
(deterministic). It needs the two banks within the opt-in memory and the
per-lane accumulator $\lceil N_k/16\rceil\cdot\lceil n_c/8\rceil \le 8$ m8n8
blocks (`ffi.fft.klead_outer_decode_refusal`). A caller that gets a reason
from either refusal function keeps the next route down (outer load with an
XLA decode, or the XLA encode with mode 2); the route order and its
announcement are on the [BSE page](bse.md#the-matvec).
`LORRAX_BSE_OUTER_KSUM=fma` runs both K sums on the fp64 FMA pipe instead of
the tensor cores, for comparing the two on devices where their rates differ.

## 12. Cost {#cost}

Each transform costs $O(n_{\rm col}N_k\log N_k)$ flops. HBM traffic is one
read of each operand and one write of the result; the split arms of modes 7,
8 and 11 add one write and one read of their chunk intermediate. Modes 6–9
and 11 read the producer's own buffer (the plane FFT output, the parent
Green, the wedge), so the phased, split or unfolded copy that modes 1 and 2
would need is never written. Their gather still reads one source element per
full-k element: parent tiles are $n_{\rm parent}/N_k$ of the full-k memory,
not of the traffic.

On the cpu leg, `gw_conv` stages $V_R = F^{-1}W$ once per call in a reused
host arena of $16\,N_k\,m_x\,m_y$ bytes, invisible to XLA.

## 13. Refusals {#refusals}

| refusal | raised at | condition | fix |
|---|---|---|---|
| `GATE kconv-platform` | startup, factory | the mesh platform is neither CUDA nor cpu | run on a CUDA or cpu mesh |
| `GATE mathdx-headers` | startup (`mathdx_root`); kernel build | no importable `nvidia.mathdx` with `include/cufftdx.hpp` | install the wheel; the `cuda12`/`cuda13` extras of `pyproject.toml` pin `nvidia-mathdx==25.6.0` |
| `GATE kconv-target` | startup, factory | the loaded library lacks the selected target | rebuild the CUDA leg from this tree |
| `GATE mathdx-probe` | startup | the mode-3 probe kernel fails to compile or run on this device | a wheel whose cuFFTDx supports the device's compute capability |
| `GATE kconv-kgrid` | factory | the k grid is not three positive axes | pass the run's `(n_kx, n_ky, n_kz)` |
| `GATE mathdx-kconv-axis` | factory; handler | on CUDA, a k-grid axis above 40 (`KCONV_AXIS_MAX`, the fp64 cuFFTDx thread-FFT limit); the cpu leg has no cap | a smaller k grid |
| `GATE mathdx-kconv-residency` | kernel build | a direct call of mode 0, 1 or 6 whose row exceeds the opt-in memory; the router streams such grids instead ([§8](#resident-rows)) | call through the router |
| `GATE mathdx-kconv-kbox-residency` | kernel build | the tile, plane or pencil the launch rule picked exceeds the opt-in memory (modes 2, 3, 8, 9); one whole column exceeds it (modes 4 and 5, which have no split arm); a split plane tile of one spin group exceeds it (mode 7: $n_s = 4$ at 26³ and larger on A100) | a smaller k grid |
| `GATE mathdx-kconv-chi-residency` | kernel build | mode 11 cannot hold the grid | none here: `gw.w_isdf` asks `chi_unfold_refusal` first and keeps the full-k Green route, so this gate means that predicate and the handler's rule disagree, which is a bug |
| `GATE mathdx-kconv-unfold-scratch`, `-lorentz-scratch`, `-chi-scratch` | apply | XLA's scratch allocator refuses the split-arm intermediate of mode 7, 8 or 11 ([§7](#tiles)) | for mode 11, a smaller `scratch_bytes` |
| `GATE mathdx-kconv-outer-tile`, `-outer-decode-tile`, `-outer-arch`, `-outer-rank` | kernel build | the BSE outer load's bank or accumulator does not fit, the device is older than sm_80, or K is not a multiple of 4 | call through the factories, which check `klead_outer_refusal` / `klead_outer_decode_refusal` and pad K |
| `GATE mathdx-plane-split`, `mathdx-plane-residency` | kernel build | a direct mode-10 call on a plane with no split or too large | call through `make_plane_fft_gather` |
| `GATE plane-fft-dtype` | trace | mode 10's `F` is not complex128 | pass complex128 |
| `k-leading unfold conv: …` (and `lorentz conv`, `chi unfold`, `unfold fft`) | factory; apply | tables cut for another mesh, operands whose parent count or endpoint widths differ from the tables, or a missing partner on a plan with antiunitary rows | build the tables from the same plan and mesh as the operands; pass the partner |
| `LORRAX_FFT_FFI=0` | factory | the cpu leg refuses, and so does `make_flat_k_fft` on both platforms | unset `LORRAX_FFT_FFI` |

A kernel-build failure (an NVRTC compile, missing toolkit headers, a module
load, a residency gate) is sticky: the handler caches it per build key and
returns it on every later call, naming the stage
(`kconv_mathdx (fused cuFFTDx k-convolution): <stage> failed -- …`).

## 14. Build, headers and the cubin cache {#build-and-cache}

**In-process build.** NVRTC compiles one image for the device's own
`sm_<cc>` per (CUDA context, mode, $n_{kx}$, $n_{ky}$, $n_{kz}$, $n_s$, right
width, precision, variant) and keeps it in an in-process cache. Modes 0–9 and
11 share one embedded source (`kSrc`); mode 10 has its own (`kPlaneSrc`); the
k-box modes also embed `kbox_stage.cuh` as a named header, turned into text
at configure time (`kbox_stage_src.h.in`). The BSE outer targets
(`kconv_outer_cuda_ffi.cc`) embed the same header in their own programs.

**Headers.** The router passes the wheel's `nvidia/mathdx` directory to every
handler as the string attribute `mathdx_root`. NVRTC includes its `include/`
and `external/cutlass/include`, plus the CUDA toolkit's `include/` and
`include/cccl`, found beside the loaded `libnvrtc`. No environment variable
names either path. Building `liblorrax_ffi.so` needs no mathdx: the
translation unit links `libnvrtc` and resolves the driver API by `dlsym`. The
CUDA leg always compiles this family; a missing `cufft.h`, `libcufft`,
`nvrtc.h` or `libnvrtc` fails the configure.

**Disk cubin cache.** Images are kept in `ffi.fft.cubin_cache_dir()`:
`$SCRATCH/.cache/lorrax/kconv_mathdx`, or `~/.cache/lorrax/kconv_mathdx`
where the site defines no `SCRATCH`. The cache is always on, has no knob, and
is separate from the XLA compile cache (`ISDF_JAX_CACHE_DIR`). One directory
serves every world size, because an image depends on the device and the
wheel, not on P. A cold build costs about 6 s per image per process, which
the cache pays once.

- **Key.** `src/ffi/cpp/common/nvrtc_build.h` owns the rule for every
  NVRTC-built kernel (this family and the Fourier plan's fused pair): FNV-1a
  over the embedded source, the text of each embedded header, the NVRTC
  options that decide the image (C++ standard, architecture, mode, grid,
  $n_s$, rows per block, precision, SM), and the whole toolchain that can
  change an image (`nvrtc::mathdx_toolchain`: the cuFFTDx or cuBLASDx,
  commonDx, CUTLASS and CCCL version headers, the wheel's dist-info name, and
  the NVRTC version with the loaded `libnvrtc`'s real path). A version header
  that reads empty disables the disk cache for that build rather than
  dropping out of the key. Editing an embedded source or header invalidates
  its images, comments included, so measurement notes live at the Python
  owners, not in the kernel text. File names, include paths and the host code
  are not keyed.
- **File.** `kconv_m<mode>[w<variant>]_<nkx>x<nky>x<nkz>_ns<ns>[x<n_r>][_c64]_sm<XY>_<key>.cubin`
  (and `plan_pair_…` for the Fourier plan), each framed by a `LRXKCONV1`
  header carrying the key and a hash of the payload.
- **Writes and reads.** A write goes to a unique temporary and is renamed
  into place, which is atomic on one filesystem, so concurrent ranks each
  publish a whole file. A read re-hashes the payload and checks for an ELF
  image; a torn, foreign or non-ELF file, or one the driver refuses to load,
  is deleted, recompiled once and replaced.
- **Receipts.** Under `LORRAX_DEBUG_PRINT=1` the startup `[kconv]` line names
  the backend, the wheel root and the cache directory with its image count
  and size. Every kernel build prints `[kconv_mathdx] disk-cache hit` or
  `NVRTC built …` on rank 0, with the grid, rows per block, shared memory and
  whether the cubin was stored.

**Test hook.** Under `LORRAX_KFFT_CPU_TEST_XLA=1` on a cpu mesh the cpu leg
announces itself and uses `jnp.fft` for its k-axis transforms, for
in-process CPU meshes that have no host library. It is never read on CUDA and
is never a production route.

## 15. Numerical contract {#numerical-contract}

- **Dtype.** Modes 2–5 take all-complex128 or all-complex64 operands (the
  complex64 image serves the fp32-GMRES BSE arm) and never cast; every other
  mode is complex128 only. The cpu host handlers are complex128 only, so a
  complex64 operand refuses at trace time on a cpu mesh.
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
   CUDA and the plan-route composition on cpu, and re-export it from
   `common.fft_helpers`.
4. Put the kernel's measured record (its speedup over the plain-XLA route,
   what was tried and did not pay) in the comment block above its Python
   factory, not in the kernel source, whose text is part of the cubin key.
   The [FFI layer's kernel catalog](ffi_layout.md#kernel-catalog) lists those
   blocks.
