# Compute kernels

The kernels every GW and BSE workflow reduces to (k-axis convolutions and transforms over the ISDF centroid basis, GEMMs over bands, local FFTs), with their operands, sharding, precision, memory, backends and the rule that picks each backend.
{: .page-summary }

In the ISDF basis a two-point operator at one Bloch vector is an $N_\mu\times N_\mu$ matrix on the centroids $\br_\mu$. The Green's function at one time node is a contraction over bands, $G_\bk(\mu,\nu;\tau)=\sum_n\psi_{n\bk}(\br_\mu)\,w_n(\tau)\,\psi^*_{n\bk}(\br_\nu)$ with $w_n(\tau)$ the windowed phase $e^{-\tau(E_{n\bk}-E_{\rm ref})}$, so building it is one batched GEMM per node. The equal-time products of [time–frequency (1.7)](../theory/time-frequency.md#conv-time) are then sums over $\bk$ of a product at $\bk$ and $\bk\pm\bq$: $\chi^0_\bq(\mu,\nu;\tau)\propto\sum_\bk G_{\bk+\bq}(\mu,\nu;\tau)\,G_\bk(\nu,\mu;-\tau)$ and $\Sigma_\bk(\mu,\nu;\tau)\propto\sum_\bq G_{\bk-\bq}(\mu,\nu;\tau)\,W_\bq(\mu,\nu;\tau)$. A sum over $\bk$ of such a product is a pointwise product in the conjugate lattice vector $\bR$ of the Born–von Kármán supercell, so each is two inverse FFTs over the $\bk$ grid, one multiply per $(\mu,\nu)$ column and one forward FFT: $\mathcal O(N_\mu^2N_k\log N_k)$ per node instead of $\mathcal O(N_\mu^2N_k^2)$. The minimax rule turns each frequency integral into a sum over a few such nodes. The BSE direct term is the same convolution with a trial-vector tensor in place of $G$. What remains is GEMMs: $G$ from $\psi$ at the centroids, and $\Sigma_\bk(\mu,\nu)$ projected back onto bands. This page documents those kernels.
{: .lead }

## Conventions {: #conventions }

These hold for every kernel on this page unless its section says otherwise.

| symbol | meaning |
|---|---|
| $(n_{kx},n_{ky},n_{kz})$, $N_k$ | the k grid and its size |
| flat k | index $(k_x n_{ky}+k_y)\,n_{kz}+k_z$ (C order, $k_z$ fastest); arrays are `(nk, *trail)` |
| $\bR$ | a lattice vector of the $N_k$-cell supercell, the Fourier partner of $\bk$ |
| $\mu$, $\nu$ | left and right centroid indices; $\mu$ is sharded over mesh axis `'x'`, $\nu$ over `'y'` |
| $n_s$ | spin components per endpoint: 1 (scalar), 2 (spinor), 4 (bispinor) |
| $F_k$, $F_k^{-1}$ | unnormalized forward ($e^{-2\pi i\bk\cdot\bR}$) and inverse ($e^{+2\pi i\bk\cdot\bR}$) DFT along k |
| $n_{\rm parent}$ | the parent k (or q) points the operators are built on ([symmetry §1.4](../theory/symmetry.md#sym-parent-k)) |

- **Mesh.** All kernels run on the 2-D device mesh with axes `('x', 'y')`. The k axes are never sharded: every rank holds whole k columns, and a spec that shards a k axis is refused (`ffi.fft.validate_flat_spec`).
- **Specs.** k-leading specs are written on the 3-D form `(nkx, nky, nkz, *trail)` with the three k entries `None`; the factory converts them to the flat form.
- **Precision.** Operands are complex128 and transforms run in fp64. The exceptions are stated per kernel.
- **Scale.** Every native handler takes one total scale computed in Python from `jnp.fft`'s `norm` convention (`ffi.fft.ffi_fft_scale`, `conv_kpair_scale`); physics prefactors enter through `mult`.

## Kernel summary {: #summary }

| kernel | computes | Python entry | native source (`src/ffi/cpp/`) | backends |
|---|---|---|---|---|
| [χ⁰ time node](#chi0-node) | $acc_o \mathrel{+}= \alpha_o\sum_{ab}\overline{\hat G^c_{ab}(\bR)}\,\hat G^v_{ab}(\bR)$ | `ffi.fft.make_kconv_chi_unfold`, `make_kconv_chi_vertex` | `cufft/kconv_mathdx_cuda_ffi.cc`; `kdft/kbox_rspace_*` | mathdx, kdft, host FFT + XLA, XLA |
| [W wedge to R space](#w-wedge) | $Y_\bk=s\,F_k^{-1}(\mathcal L_\bk\hat O_\bk\mathcal R_\bk^\dagger)$ from the irreducible-q wedge | `ffi.fft.make_kfft_klead_unfold` | as above | mathdx, kdft, host FFT + XLA, XLA |
| [Σ = G⋆W](#sigma-conv) | $U=s\,F_k(F_k^{-1}\hat G\cdot W_\bR)$ at the stored k rows | `ffi.fft.make_kconv_klead_unfold`, `make_kconv_klead`, `make_kconv_lorentz_unfold` | as above; `fftw/fft_flat_k_ffi.cc` | mathdx, kdft, host FFT, XLA |
| [flat-k FFT](#flat-k) | $Y=s\,F^{\pm}_kX$, k leading or trailing | `ffi.fft.make_kfft_klead`, `make_kfft_kminor`; `common.fft_helpers.make_flat_k_fft` | `cufft/kconv_mathdx_cuda_ffi.cc`; `kdft/kdft_klead_*`, `kdft/kdft_kminor_*`; `fftw/fft_flat_k_ffi.cc` | mathdx, kdft, FFTW3 ABI, XLA |
| [BSE W term](#bse-w) | $U=s\,F_k(F_k^{-1}T\cdot W_\bR)$, $T=\sum_K\Lambda_K\Xi_K$ | `ffi.fft.make_local_kconv_klead_outer[_decode]`, `make_local_kconv_klead`, `make_kconv_kminor` | `cufft/kconv_outer_cuda_ffi.cc`; `kdft/kconv_outer_rocm_ffi.cc` | mathdx (fp64 tensor cores), kdft, XLA |
| [ζ-fit pair Gram](#pair-gram) | $U_\bq=s\,F_k\sum_{ab}\phi_l\phi_r\,\overline{F_k^{-1}P^L}\cdot F_k^{-1}P^R$ | `ffi.fft.make_fused_conv_kpair`, `make_fused_conv_kparent`, `make_fused_conv_kplane` | `cufft/kconv_mathdx_cuda_ffi.cc`; `kdft/kbox_parent_*`, `kdft/kbox_plane_*` | mathdx, kdft, host FFT + XLA, XLA |
| [plane FFT with gather](#plane-fft) | 2-D FFT of a plane scattered from its occupied cylinder | `ffi.fft.make_plane_fft_gather`, `LocalFourierPlan(in_gather=…)` | `cufft/kconv_mathdx_cuda_ffi.cc`; `kdft/plane_gather_*` | mathdx, kdft, XLA |
| [Green-build GEMM](#green-gemm) | $G_q=A_q\,\mathrm{diag}(w_q)_{lo:hi}\,B_q$ over a band interval | `distrib_la.gemm_plan`, `panel_matmul` | `active_gemm/`; `cublas/local_active_gemm_ffi.cc` | device-bounds kernel (CUDA, ROCm), cuBLAS/rocBLAS, JAX |
| [band projection](#band-projection) | $O_{\bk,mn}=\sum\overline{\psi_{m\bk}(\mu)}\,O_\bk(\mu,\nu)\,\psi_{n\bk}(\nu)$ | `common.contract_bands.contract_bands_block_reshard` | `cblas/gemm_batch_ffi.cc` (cpu, legacy body) | XLA dots, CBLAS |
| [box FFT](#box-fft) | 3-D FFT of a real-space box per batch row | `common.fft_helpers.local_fftn3`, `make_sharded_fftn_3d` | none | `jnp.fft` (cuFFT, XLA:CPU) |
| [Local Fourier plan](#fourier-plan) | $y=R_{\rm out}\,F\,E_{\rm in}\,x$ on at most three axes | `common.fourier_plan.LocalFourierPlan` | `cufft/fourier_plan_cuda_ffi.cc`; `hipfft/fourier_plan_rocm_ffi.cc` | cuBLAS + cuFFT + cuBLASDx, rocBLAS + hipFFT, XLA |

"mathdx" is the NVIDIA family: cuFFTDx thread FFTs from the `nvidia-mathdx` wheel inside one fused shared-memory pass, compiled by NVRTC per (mode, k grid, $n_s$, device). "kdft" is its ROCm counterpart, compiled by hipRTC. Log lines and refusals name these kernels by mode number (`mode 7`, `mode 11`); the numbers are given below.

## How a backend is chosen {: #routing }

**k-axis kernels.** `ffi.fft.kconv_backend(mesh, kgrid)` reads the vendor of the mesh's first device and returns `mathdx` on CUDA, `plan` on cpu and `xla` elsewhere. No input-file key or environment variable selects it, and all backends return the same callable contract, so callers never branch on the backend.

- **CUDA** takes mathdx unless `ffi.fft.mathdx_refusal(kgrid)` names a reason: a k axis above `KCONV_AXIS_MAX` = 40 (the fp64 thread-FFT limit), a split-arm plane tile of 16 columns that exceeds the device's opt-in shared memory per block (every k-box mode's floor; about 25³ on A100), one k-box column beyond it for the k-trailing modes 4 and 5, or a startup probe compile that failed on any process. Such grids take the XLA backend with one `RuntimeWarning` and a `[kconv] k-convolution router: CUDA -> XLA` line. Devices below compute capability 8.0 are refused at startup.
- **cpu** takes the plan backend: the same unfolds, products and spin sums in XLA with the host FFTW3-ABI handlers `lorrax_mklfft_flat_k` and `lorrax_mklfft_gw_conv` for the k transforms. The FFTW3 symbols are bound at first use (an already loaded MKL or FFTW export, else `$LORRAX_FFTW3_SO` and a fixed list of library names).
- **ROCm** takes kdft for the factories it serves (the k-leading and k-trailing modes 2–5, modes 1, 6, 7, 9, 10, 11 and the BSE outer load) within its verified bounds ($N_k\le 10240$, axis limits per mode, a tile within 64 KiB of LDS); every other factory, and any grid outside the bounds, takes XLA. Each ROCm factory announces its route once.
- **Other platforms** take XLA: the same composition with XLA's k transforms (a dense DFT GEMM on the k axis for $N_k\le144$ on ROCm and cpu, `jnp.fft` otherwise).

`ffi.fft.require_kconv(mesh)` runs once at startup from `runtime.initialize_communicator_stack`: on CUDA it finds the wheel, checks every target in `ffi.fft.KCONV_TARGETS` in the loaded library and compiles and runs one probe kernel; on cpu it runs each host engine once. Inside the context manager `ffi.fft.xla_reference()` every factory takes the XLA backend on any platform; this is how each vendor route is held to relative $10^{-12}$ of the reference (`tests/test_kconv_xla_gate.py`).

On ROCm the variable `LORRAX_KDFT_OUTER` (`measured` by default, or `all`) changes only the engine inside a route, for A/B comparison. The CUDA BSE outer k sum always runs on the FP64 tensor cores (DMMA).

**GEMMs.** The Green build goes through `distrib_la` plans, which resolve their provider per platform ([distrib_la](distrib_la.md)). The band projection's host GEMM has one dial, `LORRAX_BANDS_GEMM_FFI` (`on` default, `off`), which exists only on cpu meshes: off is an announced debug fallback to the XLA einsum. On GPU meshes XLA's `dot` already calls cuBLAS and the dial does not apply. `LORRAX_MKLBLAS_THREADS` (`auto`, `off`, or an integer) sets the BLAS thread team inside that handler.

**Local Fourier plan.** Per axis, a GEMM against a stored Fourier matrix or an FFT, from a measured per-device table (`GEMM_CROSSOVER`); the leg follows the platform the call is lowered for ([below](#fourier-plan)).

## The k-convolution {: #kconv }

All k-axis kernels compute one operation, the pair correlation or convolution of two k-periodic centroid operators:

$$
\begin{aligned}
U_\bq(\mu,\nu)&=\sum_\bk\overline{A_\bk(\mu,\nu)}\,B_{\bk+\bq}(\mu,\nu)=\frac{1}{N_k}F_k\Big[\overline{F_k^{-1}A}\cdot F_k^{-1}B\Big](\bq),\\
\sum_\bq A_{\bk-\bq}B_\bq&=\frac{1}{N_k}F_k\Big[F_k^{-1}A\cdot F_k^{-1}B\Big](\bk).
\end{aligned}
$$

Each use differs only in the $\bR$-space step between the transforms. On CUDA the kernels hold a tile of k columns in shared memory from the load through both transforms to the store, so each operand is read from device memory once and the result written once. They also unfold the parent-k operator to the full grid inside the load, so no full-k copy of $G$ or $W$ is written.

### Shared contract {: #kconv-contract }

- **Rank-local or sharded.** Factories named `make_local_*`, and the pair, parent and plane factories, return rank-local callables for use inside the caller's `shard_map`; the others wrap their own `shard_map`.
- **Unfold-on-load tables.** The parent-k factories take `symmetry_maps.unfold_load_tables` (`UnfoldLoadTables`): per full k, the parent row `row` `(N_k,)` int32, an antiunitary flag `trs`, X-local and Y-local endpoint sources `lsrc`, `rsrc` int32 (−1 reads zero), umklapp phases `mph`, `nph` complex128, and the spin action `spin` `(N_k, n_s, n_s)`. The kernel forms $\hat G_\bk[\mu a,\nu b]=\sum_{cd}U_\bk[a,c]\,V_\bk[\mu c,\nu d]\,\overline{U_\bk[b,d]}$ with $V_\bk$ the phased parent tile ([symmetry_maps](symmetry_maps.md)). The tables refuse a source map that crosses an `'x'` or `'y'` shard, which is why centroids are orbit-packed.
- **Placed tables.** Without the `load` operand the factory bakes the tables into the program as constants (about 0.57 GB of HLO literal at Fe 20³ with 1792 centroids). Production callers pass `load=symmetry_maps.device_load_tables(...)`.
- **Antiunitary partners.** An antiunitary row reads a transposed partner tile `Gt`, or, for a Green with real weights, `conj(G)` formed on the load (pass `None`).
- **Row passes.** Modes 7, 8, 9 and 11 accept an optional last operand `live` (int32 `[2]`, replicated), the live left rows `[lo, hi)` of an equal-width pass inside a `lax.scan`; dead rows are neither gathered nor transformed.
- **In place.** Modes 2, 3, 5, and 4 with `out_layout=0`, alias operand 0 to the output; mode 11 aliases `acc`. Donate those buffers.
- **Scratch.** The split arms of modes 7, 8 and 11 (large grids) take an intermediate from XLA's scratch allocator, at most 1 GiB (`ffi.fft.klead_unfold_scratch_bytes`, `chi_unfold_scratch_bytes`). The compiled program's `memory_analysis()` does not count it; callers price it.
- **Traffic.** One read of each operand and one write of the result per call; the parent tiles are $n_{\rm parent}/N_k$ of the full-k memory, not of the traffic, because the gather reads one source per full-k element. The plan and XLA backends write a full-k unfolded copy first, a factor $N_k/n_{\rm parent}$ more memory.

### χ⁰ time node (mode 11) {: #chi0-node }

One minimax node of the independent-particle response, read from the parent Greens:

$$
\begin{aligned}
acc[o]&\mathrel{+}=\alpha_o\sum_{ab}\overline{\hat G^c_{ab}(\bR)}\,\hat G^v_{ab}(\bR)\quad(+\text{c.c. when } \texttt{complete}),\\
\hat G(\bR)&=F_k^{-1}\big[U_\bk\,G_{{\rm row}(\bk)}\,U_\bk^\dagger\big].
\end{aligned}
$$

For real weights $\overline{G_\bk(\mu,\nu)}=G_\bk(\nu,\mu)$, so the conjugate supplies the reversed endpoint order of the second Green in (1.7). The forward transform is applied once, after the node sum.

| operand | shape | dtype | spec |
|---|---|---|---|
| `Gv`, `Gc` (and partners `Gvt`, `Gct`) | `(n_parent, μ, n_s, ν, n_s)` | c128 | `P(None,'x',None,'y',None)` |
| `alpha` | `(n_out,)` | c128 | replicated |
| `acc` (in place; `None` returns $\alpha\chi_\tau$ itself) | `(n_out, nk, μ, ν)` | c128 | `P(None,None,'x','y')` |

Call: `fn(acc, Gv, Gc, alpha, Gvt=None, Gct=None, load=None)`. `make_kconv_chi_vertex` is the four-current form with at most three monomial vertices per side. `scratch_bytes` bounds the split-arm intermediate. Usage, `src/gw/w_isdf.py` (`_get_chi_minimax_kernel_fused`):

```python
kconv = make_kconv_chi_unfold(mesh_xy, kgrid, k_unfold_plan.unfold_load_tables(),
                              n_out=n_out, complete=not complex_contour, norm="ortho")
chi_fftn = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm="ortho")

def node(acc, xs):                                   # one minimax node, inside lax.scan
    t, alpha = xs
    Gv = build_G_tau(..., -t, mask=mask_v, gemm=g_plan, unfold=False)   # occupied bands
    Gc = build_G_tau(...,  t, mask=mask_c, gemm=g_plan, unfold=False)   # empty bands
    partners = () if Gv.conj_partner else (Gv.transpose, Gc.transpose)
    return kconv(acc, Gv.G, Gc.G, alpha, *partners), None

acc, _ = jax.lax.scan(node, acc0, (nodes.t, alpha_rows))
chi_q = chi_fftn(acc[0])                             # one R -> q transform after the sum
```

### W wedge to R space (mode 9) {: #w-wedge }

`make_kfft_klead_unfold(mesh, kgrid, tables, norm=)` returns `fn(Wp, Wt=None, load=None, live=None) -> Y`: the R-space operand of the Σ convolution, $Y_\bk=s\,F_k^{-1}(\mathcal L_\bk\hat O_\bk\mathcal R_\bk^\dagger)$, read from the interaction on its irreducible-q wedge. `Wp` is `(n_wedge, μ·n_l, ν·n_r)` at `P(None,'x','y')` ($n_l=n_r=1$ for a scalar $W$; a Lorentz block flattened otherwise); `Y` is `(nk, μ·n_l, ν·n_r)` at the same spec. It equals `make_kconv_klead(...).prep` applied to the full-zone interaction, which is never stored. A Hermitian interaction uses `trs_rule="conj"` and needs no `Wt`.

### Σ = G⋆W (modes 7, 2, 8) {: #sigma-conv }

$$
\Sigma_\bk(\mu a,\nu b)=s\,F_k\big[\hat G_{ab}(\bR;\mu,\nu)\cdot W_\bR(\mu,\nu)\big],\qquad \bk\in\texttt{store\_rows}.
$$

- `make_kconv_klead_unfold(mesh, kgrid, tables, store_rows=, norm=, mult=)` (mode 7) returns `fn(G, Gt, W_prep, load=None, rows=None, live=None)`. `G` is the parent Green `(n_parent, μ, n_s, ν, n_s)` at `P(None,'x',None,'y',None)` from `gw.greens_function_kernel.build_G_parents`; `W_prep` comes from mode 9 or from `make_kconv_klead(...).prep`. Every full-k row is transformed, and only the rows in `store_rows` (the Σ consumers pass the parent rows) are written: the output is `(len(store_rows), n_s, μ, n_s, ν)` at `P(None,None,'x',None,'y')`. `rows=(x0, bx, xs, xn)` stores one x block of the left centroids.
- `make_kconv_klead(mesh, kgrid, t_spec, w_spec, norm=, mult=)` (modes 3 then 2) is the full-k form, returning `KConvStored(prep, apply)`: `prep(W)` transforms $W$ `(nk, mx, my)` to R space once per $W$; `apply(T, W_prep)` computes $U=\texttt{mult}\cdot F_k(F_k^{-1}T\cdot W_\bR)$ for `T` `(nk, a, mx, b, my)`, in place on `T`. `W_prep` is in the backend's own form (on cpu the host `gw_conv` handler takes $W$ unchanged and transforms it itself), so pass it only to the `apply` of the same pair.
- `make_kconv_lorentz_unfold` (mode 8) is the four-current Σ, $U=s\,F_k\sum_{AB}\gamma_A(F_k^{-1}\hat G)\gamma_B^\dagger\circ\hat W_\bR[\mu A,\nu B]$, with $W$ also unfolded from its parent-q tile on the load. It has a native kernel on CUDA only.

The physics prefactor rides `mult`. Usage, `src/gw/ppm_tau_kernel.py` (`_sigma_subtile_kernel`):

```python
kconv9 = make_kfft_klead_unfold(mesh_xy, kgrid, w_tables, norm="ortho")
conv = make_kconv_klead_unfold(mesh_xy, kgrid, tables,
                               store_rows=k_unfold_plan.parent_full_rows,
                               norm="ortho", mult=-1.0 / np.sqrt(float(nk)))
project = contract_bands_block_reshard(mesh_xy, channels="none", layout="axis",
                                       face_shape=(n_parent, nb, px * R, ns),
                                       right_face_shape=(n_parent, nb, n_rmu, ns))
# per node and row pass: W_R = kconv9(W_wedge, load=w_load); G = Green GEMM;
# sigma_k = conv(G, Gt, W_R, load=g_load, live=live); then project onto bands.
```

### Flat-k FFT (modes 3, 5) {: #flat-k }

`make_kfft_klead(mesh, kgrid, spec, kind=, norm=)` transforms the leading flat-k axis of `(nk, *trail)`; `make_kfft_kminor` the three trailing k axes of `(..., nkx, nky, nkz)`. `kind` is `'ifftn'` or `'fftn'`, `norm` as `jnp.fft`. `make_local_kfft_klead` and `make_local_kfft_kminor` are the rank-local forms. Physics code calls them through `common.fft_helpers.make_flat_k_fft` (and `make_flat_k_ifftn`, `make_flat_k_fftn`), which refuses an `out_spec` different from `spec`. Usage, `src/tdagw/dynsig.py`:

```python
from common.fft_helpers import make_flat_k_fft
g_ifft = make_flat_k_fft(mesh_xy, kgrid, g_spec, kind="ifftn", norm="ortho")
```

Modes 2–5 also accept complex64 operands on CUDA (both operands the same dtype; no cast), which serves the fp32 GMRES arm of the BSE; the host handlers are complex128 only.

### BSE W term {: #bse-w }

The screened direct term of the BSE matvec, for each trial vector $X$:

$$
\begin{aligned}
(WX)_{cv\bk}&=\frac{1}{N_k}\sum_{\mu\nu ts}\psi^*_{c\bk}(t,\mu)\,\psi_{v\bk}(s,\nu)\sum_{\bk'}W_{\mu\nu}(\bk-\bk')\,T_{\bk'}(t\mu,s\nu),\\
T_{\bk'}(t\mu,s\nu)&=\sum_{c'v'}\psi_{c'\bk'}(t,\mu)\,\psi^*_{v'\bk'}(s,\nu)\,X_{c'v'\bk'}.
\end{aligned}
$$

`bse.bse_stack_matvec.make_w_term(mesh, nkx, nky, nkz)` returns `w_term(X, psi_c_X, psi_v_Y, W_R)` with `X` `(b, n_c, n_v, nk)` at `P(None,'x','y',None)`, `psi_c_X` `(nk, n_c, n_s, μ)` at `P(None,None,None,'x')`, `psi_v_Y` `(nk, n_v, n_s, ν)` at `P(None,None,None,'y')` and `W_R` `(μ, ν, nkx, nky, nkz)` at `P('x','y',None,None,None)`, already in R space. The body is a `lax.scan` over trials with no collective inside, so one $T$-sized buffer is live regardless of the trial count; one reduce-scatter per trial block finishes it. The W term is chosen at trace time from three routes, the first that serves, and announced once (`[bse] W term:`):

| route | kernel | what reaches device memory |
|---|---|---|
| `fused` | `make_local_kconv_klead_outer_decode`: $T=\sum_K\Lambda_K\Xi_K$ formed in shared memory on the fp64 tensor cores (`mma.m8n8k4.f64`, sm_80+), mode 2's transforms, and the decode's $(t,\mu)$ contraction $A=\sum_{a\mu}\overline{\psi_c}\,U$ in the store | neither $T$ nor $U$ |
| `outer` | `make_local_kconv_klead_outer`: the same load, $U$ stored, decode in XLA | $U$ |
| `xla` | XLA batched ZGEMM for $T$, then `make_local_kconv_klead` (mode 2) | $T$ and $U$ |

`ffi.fft.klead_outer_refusal` and `klead_outer_decode_refusal` decide the route (bank and accumulator residency, sm_80+). $K=\min(n_c,n_v)$ is zero-padded to a multiple of 4. Every operand of one handler call must hold fewer than $2^{31}$ elements; `bse_stack_matvec.plan_w_blocks` splits the $(\mu,\nu)$ tile into blocks that respect this. The W term stays complex128 throughout.

The full (non-TDA) screening operator of `bse.bse_ring_comm` holds $k$ trailing and uses `make_kconv_kminor` (mode 4), $U=s\,F_k(F_k^{-1}X\cdot K_\bR)$ for `X` `(d0, d1, d2, d3, d4, nk)` and `K_R` `(d1, d2, nk)`, with `out_layout=1` returning `(d0, nk, d3, d1, d4, d2)`; its kernel prep is `make_kfft_kminor` (mode 5).

### ζ-fit pair Gram (modes 0, 1, 6) {: #pair-gram }

The ISDF fit's Gram matrix $C_\bq$ is the correlation of band projectors $P_\bk(\mu,\nu)=\sum_n\psi_{n\bk}(\mu)\overline{\psi_{n\bk}(\nu)}$ ([ISDF §2.2](../theory/isdf.md#isdf-kconv)):

$$
U_\bq=s\,F_k\sum_{ab}\phi_l[a]\,\phi_r[b]\;\overline{F_k^{-1}P^L_{ab}}\cdot F_k^{-1}P^R_{\pi_la,\pi_rb}.
$$

- `make_fused_conv_kpair` (mode 0): full-k operands `(nkx, nky, nkz, n_s, col, μ, n_s)` to `(nkx, nky, nkz, col, μ)`.
- `make_fused_conv_kparent` (mode 1): the operands `D_l`, `D_r` `(n_parent, n_s, μ, n_s, ν)` unfolded from parent k on the load with ten typed tables, to `(N_k, μ, ν)`; `isdf.core` is the caller.
- `make_fused_conv_kplane` (mode 6): read from the route-G plane-FFT output `D` `(N_k, g, n_s, 2c, n_s, p)` with the Bloch phase `F` `(N_k, g, p)` applied and the left/right split made on the load.

The vertex is a spin permutation `perm_l`, `perm_r` and phases in $\{\pm1,\pm i\}$ (encoded 0–3), $n_s\le4$. These factories fix `norm="forward"`, so $s=1/N_k$. On CUDA a row that does not fit shared memory (above about $N_k=3477$ on A100) is streamed in spatial tiles of at most $32\times64$ columns instead of refused.

### Plane FFT with gather (mode 10) {: #plane-fft }

`make_plane_fft_gather(mesh, plane_from_col, n_col, plane_shape)` returns the 2-D FFT ($\texttt{norm='backward'}$) of the plane $P$ scattered from a compact cylinder `F (…, n_col)`, $Y[\dots,k_b,k_c]=\sum_{bc}P[\dots,b,c]\,e^{-2\pi i(bk_b/n_b+ck_c/n_c)}$, without writing $P$: one read of the cylinder and one write of the plane. `fn(F, start, size)` transforms a slab of `F` in place. It serves a plane when both sides split into coprime factors of at most 40 and the plane plus its row tables fit the opt-in shared memory (`ffi.fft.plane_resident_bytes`; squares to 100 on A100); every other plane takes `jnp.fft.fftn`. `F` must be complex128. It is reached through `LocalFourierPlan(in_gather=…)`.

## GEMM kernels {: #gemm }

### Green-build GEMM {: #green-gemm }

$$
G_\bk(\mu,\nu;\tau)=\sum_{n\in[lo,hi)}\psi_{n\bk}(\mu)\,w_n(\tau)\,\psi^*_{n\bk}(\nu)
$$

at the parent k, built by `gw.greens_function_kernel.build_G_tau` / `build_G_parents` through a `distrib_la` GEMM plan. The band axis keeps its full extent so compiled shapes never change; only the interval $[lo,hi)$ is contracted (the active range). There is no numerical threshold: only exactly zero weight columns outside the support interval are skipped.

- `distrib_la.gemm_plan(mesh, m=, k=, n=, nq=, dtype=, layout=, enable_active_range=True, warmup=)` fixes $D_q=\alpha A_qB_q+\beta C_q$, `A (nq, m, k)`, `B (nq, k, n)`, N,N only, `alpha`/`beta` compile-time. `plan.active_range(A, B, lo, hi, weights=)` contracts the interval; `plan.prepare_active_range(lo, hi)` captures a host-known interval at compile time.
- `layout='face'`: operands at `P(None,'x','y')`, product by `distrib_la.panel_matmul`, a batched 2-D SUMMA. Band panels of at most $N_b/p$ columns are all-gathered (every k in one exchange) and multiplied into the rank's own output tile; no rank holds a band-complete panel. The two live panels are bounded by `panel_bytes`, in production one full-k Green tile, $16\,N_k\,(M/p_x)(N/p_y)$ bytes. `partner=True` returns the antiunitary partner $\bar A\,\mathrm{diag}(w)\,\bar B$ from the same exchange.
- `layout='axis'`: complete local `k`, a local product with no collective.

The local product is chosen per platform by `distrib_la._active_local.ACTIVE_LOCAL_ROUTE`:

| platform | route | kernel |
|---|---|---|
| CUDA, ROCm | `device` | `lorrax_active_gemm_device[_out]` (`src/ffi/cpp/active_gemm/`), one run-time-compiled kernel that reads the bounds on the device: no host read or stream sync, weights applied on the load of `A`, flops over each row's active width only; the inner product on DMMA f64 (CUDA sm_80+) or MFMA f64 (gfx90a/gfx94x), FMA elsewhere |
| cpu | `interval` | a JAX interval product (`distrib_la._active_local.active_local_matmul`) |

The classic-BLAS handler `lorrax_cublas_local_active_range_gemm[_out]` (`src/ffi/cpp/cublas/local_active_gemm_ffi.cc`; cuBLAS on CUDA, rocBLAS on ROCm, the interval as pointer views) remains as the `native` route; it reads the bounds on the host and synchronizes the stream per call, which measured slower inside the Σ τ loop. All routes take float64 or complex128 only, and invalid traced bounds return an all-NaN output rather than clamp. Usage, `src/gw/w_isdf.py`:

```python
g_plan = gemm_plan(mesh_xy, m=n_rmu * ns, k=nb_full, n=n_rmu * ns, nq=nk_in,
                   dtype=jnp.complex128, layout=layout, enable_active_range=True,
                   warmup=False)
```

The full API, providers and refusals are on [distrib_la](distrib_la.md); dense solves ($W_\bq=(1-v_\bq\chi^0_\bq)^{-1}v_\bq$ by LU, eigh) are there too.

### Band projection {: #band-projection }

$$
\mathrm{out}[e?,\bk,m,n]=\sum_{s\mu}\sum_{s'\nu}\overline{\psi_L}[\bk,m,s,\mu]\;O[e?,\bk,s,\mu,s',\nu]\;\psi_R[\bk,s',\nu,n]
$$

`common.contract_bands.contract_bands_block_reshard(mesh, channels=, extra=, layout=, face_shape=, right_face_shape=, ...)` returns `project(psi_left, O, psi_right)`, with `conj` applied to `psi_left` inside. No `(m, n, k)`, `(m, μ)` or `(μ, μ)` object is ever formed on one rank; the result leaves at `P(None,'x','y')`.

| `layout` | mechanism | per-rank memory |
|---|---|---|
| `face` | operator stays on its tiles; ψ_L becomes a resident 1/P slab; ψ_R streams in band chunks; `prepare` / `accumulate` / `finish` for operators in x blocks | $16N_k[n_b n_s(\mu_l+\mu_r)/p+n_b^2+3n_bn_s\mu/P]$ bytes |
| `axis` | every band local; local slab contraction, one reduce-scatter of the $(n_b,n_b)$ partial | $16N_kn_b^2$ per channel |
| `legacy` | right GEMM, `psum_scatter('y')`, left GEMM, `psum_scatter('x')` | two partials |

The mesh's minor axis must be `'y'` (the large reduce-scatter runs on consecutive ranks). `channels="split_reim"` projects $\mathrm{Re}\,O$ and $\mathrm{Im}\,O$ separately. On a cpu mesh the legacy body's right GEMM is `ffi.gemm.gemm_batch`, the host CBLAS handler `lorrax_mklblas_gemm_batch` (`src/ffi/cpp/cblas/gemm_batch_ffi.cc`): $C[i]=A[i]\,B[i\bmod B_B]$, row-major, f64/f32/c128/c64, `cblas_?gemm_batch` where the BLAS exports it, else a loop of plain calls. On GPU meshes all three layouts are XLA dots (cuBLAS).

## Local transforms {: #local-fft }

### Box FFT {: #box-fft }

`common.fft_helpers.local_fftn3` / `local_ifftn3` are `jnp.fft.fftn` / `ifftn` over the three trailing box axes of a device-local shard; `make_sharded_fftn_3d` / `make_sharded_ifftn_3d` wrap them in a `shard_map`. The box axes must be replicated; batch axes may be sharded. They serve $\psi_{n\bk}(\bG)\leftrightarrow\psi_{n\bk}(\br)$, densities and the plane-wave matrix elements (`common.mtxel_sweep`, the wavefunctions at the centroids). There is no native route: XLA lowers them to cuFFT on NVIDIA and to its own FFT on cpu. Call the `local_*` forms only inside a `shard_map`; outside one, JAX gathers the whole operand onto every rank.

### Local Fourier plan {: #fourier-plan }

`LocalFourierPlan(extents, axes, sign=, norm=, in_support=, out_support=, out_perm=, in_gather=)` computes $y=R_{\rm out}\,F_{\rm sign,norm}\,E_{\rm in}\,x$ on one device over at most three axes, equal to `jnp.fft.fftn` (`sign=-1`) or `ifftn` (`sign=+1`) with zero-filled input outside `in_support` and only the `out_support` indices returned. `x` must be complex128.

- **Per axis.** An axis is a GEMM with the stored matrix $A[j',j]=s\,e^{\pm2\pi i(\mathrm{idx}'_{j'}\mathrm{idx}_j\bmod N)/N}$ when $N$ is in the device's measured range and $n_{\rm in}n_{\rm out}\le\kappa N^2$ (`GEMM_CROSSOVER`; on A100, supported axes of 16–128 points with $\kappa=0.54$); otherwise it joins one FFT group. Unknown devices and cpu take the FFT on every axis. `plan.stages` lists the choice.
- **Legs.** A plan with no GEMM axis runs as XLA ops on every platform. With one, CUDA runs one `lorrax_fourier_plan_mathdx` custom call (cuBLAS ZGEMM axes, a fused cuBLASDx kernel for two back-to-back trailing GEMM axes when it fits shared memory, one cuFFT plan for the FFT group); ROCm runs `lorrax_fourier_plan_rocm` (rocBLAS and hipFFT, no fused pair); every other lowering uses `tensordot` and `jnp.fft`.
- **Accuracy.** GEMM axes and the fused pair agree with `np.fft` to relative $10^{-12}$. Sizes past $2^{31}-1$ refuse.

```python
from common.fourier_plan import LocalFourierPlan
y = LocalFourierPlan(N, axes, sign=-1, norm=n)(x)      # == jnp.fft.fftn(x, axes=axes, norm=n)
```

The plane-wave GW path (`gw.plane_wave_pipeline`) builds separable plans for its sphere↔box transforms; the ζ fit uses `in_gather`.

## Building on these kernels {: #extending }

A new term that is an equal-time product of two centroid-space operators summed over k is one of the existing factories:

- a convolution $\sum_\bq A_{\bk-\bq}B_\bq$ is `make_kconv_klead` (full k) or `make_kconv_klead_unfold` (A on parent k);
- a correlation $\sum_\bk\overline{A_\bk}B_{\bk+\bq}$ is `make_kconv_chi_unfold` (one τ node, accumulated) or a pair factory;
- a single transform is `make_flat_k_fft`.

Pick the factory whose k position (leading, trailing, parent tables) matches the tile you hold; a transpose to reach another factory costs a full extra pass through device memory. Build the Green with a `gemm_plan`, project with `contract_bands_block_reshard`, and hold every new route to the XLA backend under `ffi.fft.xla_reference()` at relative $10^{-12}$. A new native mode needs a handler, a target registered in `ffi_loader` and in `ffi.fft.KCONV_TARGETS`, a router factory with its XLA composition, and a case in `tests/test_kconv_xla_gate.py`; a changed operand contract is a new target string, never an edit of an existing one.

## Invariants and gotchas {: #gotchas }

- **k is replicated.** Every k-axis kernel needs whole k columns on every rank. Shard centroids or bands, never k.
- **Same mesh, same tables.** Unfold tables are cut for one mesh shape and one plan; operands from another plan or mesh refuse (`k-leading unfold conv: …`). Pass the device-placed `load` tables in production.
- **Value parity, not bitwise.** Backends agree to relative $10^{-12}$; FFT engines and BLAS libraries reassociate. Within one backend, the single and split arms agree bit for bit, and reruns are bitwise at fixed device and toolkit.
- **Precision.** Only modes 2–5 take complex64, on the mathdx and XLA backends (the host handlers refuse it at trace time); every other kernel is complex128. Do not cast a kernel's inputs down to save memory without checking its dtype contract.
- **Donate aliased operands.** Modes 2, 3, 5, 4 (`out_layout=0`) and 11 write in place.
- **`W_prep` belongs to its pair.** The `prep` output of `KConvStored` is backend-specific; pass it only to the `apply` of the same factory call.
- **Scratch is invisible to `memory_analysis`.** Price the split-arm intermediate (≤ 1 GiB) yourself.
- **A demotion to XLA is announced.** A CUDA grid with an axis above 40, or a failed probe, runs on XLA with a warning and a router log line. Read the log before comparing timings.
- **First call compiles.** Each NVRTC image compiles at its first call and is disk-cached; a kernel-build failure is cached per build key and returned on every later call.
- **The BSE outer load caps a call at $2^{31}$ elements per operand.** Call it through `bse_stack_matvec`, whose block plan keeps each call below the cap.
- **GEMM plans are N,N with compile-time `alpha`, `beta`.** Store a transposed operand once in the complementary layout.
- **Box FFTs only inside `shard_map`.** Eager calls on sharded arrays gather to every rank.

## Where LORRAX uses it {: #where-used }

| module | kernels |
|---|---|
| `gw.w_isdf` | Green-build GEMM, χ⁰ time node (mode 11), flat-k FFT (χ⁰(R) → χ⁰(q)) |
| `gw.screening` | W wedge to R space (mode 9) |
| `gw.ppm_tau_kernel` | Green-build GEMM, mode 9, Σ τ convolution (mode 7), band projection |
| `gw.cohsex_sigma` | mode 9, `make_kconv_klead`, band projection |
| `gw.mpa.sector_sigma` | four-current Σ (mode 8), band projection |
| `gw.photon_sigma`, `gw.wavefunction_bundle` | band projection |
| `gw.qsgw_head`, `bandstructure` | flat-k FFT |
| `isdf.core`, `isdf.zeta_mubatch` | ζ-fit pair Gram (modes 0, 1, 6), plane FFT with gather (mode 10) |
| `bse.bse_stack_matvec` | BSE W term (outer load, fused decode, mode 2), flat-k FFT (mode 5, kernel prep) |
| `bse.bse_ring_comm` | BSE ring rung (mode 4), band projection (legacy) |
| `tdagw.dynsig`, `tdagw.model` | flat-k FFT; BSE W term |
| `common.mtxel_sweep`, `common.wfn_transforms` | box FFT |
| `gw.plane_wave_pipeline` | Local Fourier plan, band projection |

Related pages: [minimax quadrature](minimax.md) for the time nodes and weights, [distrib_la](distrib_la.md) for the GEMM and solver API, [symmetry_maps](symmetry_maps.md) for the unfold tables, [ISDF tensor file loading](zeta_loader.md) and [wavefunction loading](wfn_loader.md) for the operands, and the [developer index](index.md).
