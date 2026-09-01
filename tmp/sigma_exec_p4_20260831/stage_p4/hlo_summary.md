# HLO dump summary

**Dump dir:** `/pscratch/sd/j/jackm/wt_sigma_exec_2026-08-31/tmp/sigma_exec_p4_20260831/stage_p4/xla_dump_rank0`
**Modules dumped:** 189
**Sum of per-module peak live HBM:** 47.52 GiB (upper bound; peaks occur at different times)

_Companion files with richer context:_
- [`memory_details.txt`](memory_details.txt) — top-N modules' memory-usage-report, concatenated
- [`collectives_details.txt`](collectives_details.txt) — HLO context around each collective + source_file:line
- [`remat_details.txt`](remat_details.txt) — every remat warning + nearby HLO lines
- [`retrace_details.txt`](retrace_details.txt) — input signatures that caused each retrace

## Memory — largest modules by peak HBM

| Module | Peak HBM | Top allocation |
|---|---:|---|
| `module_0109.jit_sigma_sx` | 8.25 GiB | 6.60 GiB — preallocated-temp: |
| `module_0374.jit__build` | 6.60 GiB | 2.93 GiB — parameter 1, shape \|c128[4,512,310,310]\| at ShapeIndex {}: |
| `module_0368.jit_concatenate` | 5.87 GiB | 2.93 GiB — output shape is \|c128[4,512,310,310]\|, maybe-live-out: |
| `module_0378.jit__gw_conv` | 3.67 GiB | 2.93 GiB — parameter 0, shape \|c128[512,2,310,2,310]\| at ShapeIndex {}, output shape is \|c128[512,2,310,2,310]\|, maybe-live-out: |
| `module_0380.jit__project` | 3.39 GiB | 2.93 GiB — parameter 1, shape \|c128[512,2,310,2,310]\| at ShapeIndex {}: |
| `module_0376.jit__g_from_selector` | 3.28 GiB | 2.93 GiB — output shape is \|c128[512,2,310,2,310]\|, maybe-live-out: |
| `module_0364.jit__do_unfold` | 2.25 GiB | 1.47 GiB — preallocated-temp: |
| `module_0360.jit__do_unfold` | 2.25 GiB | 1.47 GiB — preallocated-temp: |
| `module_0366.jit_broadcast_in_dim` | 1.47 GiB | 750.78 MiB — output shape is \|c128[1,512,310,310]\|, maybe-live-out: |
| `module_0049.jit_gather` | 752.25 MiB | 750.78 MiB — parameter 0, shape \|c128[512,310,310]\| at ShapeIndex {}: |
| `module_0039.jit_fn` | 750.93 MiB | 750.78 MiB — parameter 0, shape \|c128[512,310,310]\| at ShapeIndex {}: |
| `module_0026.jit__per_rank` | 750.78 MiB | 0.00 B —  |
| `module_0037.jit__identity_fn` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,310,48,2]\|, maybe-live-out: |
| `module_0032.jit_conjugate` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,48,2,310]\|, maybe-live-out: |
| `module_0034.jit_transpose` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,310,48,2]\|, maybe-live-out: |
| `module_0068.jit__identity_fn` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,48,2,310]\|, maybe-live-out: |
| `module_0070.jit_transpose` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,2,310,48]\|, maybe-live-out: |
| `module_0073.jit__identity_fn` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,2,310,48]\|, maybe-live-out: |
| `module_0075.jit_conjugate` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,310,48,2]\|, maybe-live-out: |
| `module_0077.jit_transpose` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,2,310,48]\|, maybe-live-out: |
| `module_0080.jit__identity_fn` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,2,310,48]\|, maybe-live-out: |
| `module_0082.jit_transpose` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,48,2,310]\|, maybe-live-out: |
| `module_0085.jit__identity_fn` | 465.00 MiB | 232.50 MiB — output shape is \|c128[512,48,2,310]\|, maybe-live-out: |
| `module_0342.jit_xn` | 348.75 MiB | 232.50 MiB — parameter 0, shape \|c128[512,2,310,48]\| at ShapeIndex {}: |
| `module_0344.jit_yr` | 348.75 MiB | 232.50 MiB — parameter 0, shape \|c128[512,48,2,310]\| at ShapeIndex {}: |
| `module_0346.jit_xr` | 348.75 MiB | 232.50 MiB — parameter 0, shape \|c128[512,48,2,310]\| at ShapeIndex {}: |
| `module_0348.jit_yn` | 348.75 MiB | 232.50 MiB — parameter 0, shape \|c128[512,2,310,48]\| at ShapeIndex {}: |
| `module_0354.jit_multiply` | 340.20 MiB | 170.10 MiB — output shape is \|c128[4,29,310,310]\|, maybe-live-out: |
| `module_0059.jit_fn` | 232.55 MiB | 232.50 MiB — parameter 0, shape \|c128[512,48,2,310]\| at ShapeIndex {}: |
| `module_0028.jit__per_rank` | 232.50 MiB | 0.00 B —  |
| `module_0356.jit_gather` | 212.62 MiB | 170.10 MiB — parameter 0, shape \|c128[4,29,310,310]\| at ShapeIndex {}: |
| `module_0352.jit__per_rank` | 170.10 MiB | 0.00 B —  |
| `module_0358.jit_broadcast_in_dim` | 85.05 MiB | 42.52 MiB — output shape is \|c128[29,310,310]\|, maybe-live-out: |
| `module_0399.jit__ev_tensors` | 76.50 MiB | 23.62 MiB — maybe-live-out: |
| `module_0390.jit__add_diag` | 74.81 MiB | 23.62 MiB — preallocated-temp: |
| `module_0439.jit__kernel` | 39.84 MiB | 23.62 MiB — parameter 0, shape \|c128[21,512,12,12]\| at ShapeIndex {}: |
| `module_0130.jit__valence_density_kernel` | 32.25 MiB | 16.00 MiB — preallocated-temp: |
| `module_0128.jit_squeeze` | 32.00 MiB | 16.00 MiB — output shape is \|c128[16,2,32,32,32]\|, maybe-live-out: |
| `module_0394.jit__diag_sharded` | 27.56 MiB | 23.62 MiB — parameter 0, shape \|c128[21,512,12,12]\| at ShapeIndex {}: |
| `module_0299.jit__run` | 25.45 MiB | 12.28 MiB — preallocated-temp: |

## Sharding — collectives (largest by output bytes)

| Module | Op | Output bytes | Source | Output type |
|---|---|---:|---|---|
| `module_0360.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,310,310]{1,0,2}` |
| `module_0360.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,620,155]{2,0,1}` |
| `module_0360.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,310,310]{2,0,1}` |
| `module_0360.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,155,620]{1,0,2}` |
| `module_0364.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,310,310]{1,0,2}` |
| `module_0364.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,620,155]{2,0,1}` |
| `module_0364.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,310,310]{2,0,1}` |
| `module_0364.jit__do_unfold` | `all-to-all` | 750.78 MiB | `` | `c128[512,155,620]{1,0,2}` |
| `module_0380.jit__project` | `reduce-scatter` | 58.12 MiB | `` | `c128[512,2,310,12]{2,1,0,3}` |
| `module_0407.jit__identity_fn` | `all-gather-start` | 6.75 MiB | `` | `(c128[512,12,24]{2,0,1}, c128[512,24,24]{2,0,1})` |
| `module_0439.jit__kernel` | `all-gather-start` | 6.75 MiB | `` | `(c128[512,12,24]{2,0,1}, c128[512,24,24]{2,0,1})` |
| `module_0407.jit__identity_fn` | `all-gather-start` | 3.38 MiB | `` | `(c128[512,12,12]{1,0,2}, c128[512,12,24]{1,0,2})` |
| `module_0439.jit__kernel` | `all-gather-start` | 3.38 MiB | `` | `(c128[512,12,12]{1,0,2}, c128[512,12,24]{1,0,2})` |
| `module_0380.jit__project` | `reduce-scatter` | 1.12 MiB | `` | `c128[512,12,12]{2,0,1}` |
| `module_0299.jit__run` | `all-gather-start` | 861.75 KiB | `` | `(c128[6,2,1532]{2,1,0}, c128[12,2,1532]{2,1,0})` |
| `module_0299.jit__run` | `all-gather-start` | 861.75 KiB | `` | `(c128[6,2,1532]{2,1,0}, c128[12,2,1532]{2,1,0})` |
| `module_0005.jit__identity_fn` | `all-gather-start` | 160.00 B | `` | `(f64[2,2]{1,0}, f64[8,2]{1,0})` |
| `module_0014.jit__identity_fn` | `all-gather-start` | 80.00 B | `` | `(f64[2]{0}, f64[8]{0})` |
| `module_0030.jit__identity_fn` | `all-gather-start` | 20.00 B | `` | `(u32[1]{0}, u32[4]{0})` |

## Layout boundaries — transpose/copy/bitcast per module

| Module | Op | Count | Largest instance | Source of largest |
|---|---|---:|---:|---|
| `module_0109.jit_sigma_sx` | `bitcast` | 8 | 2.93 GiB | `` |
| `module_0376.jit__g_from_selector` | `bitcast` | 3 | 2.93 GiB | `` |
| `module_0380.jit__project` | `bitcast` | 7 | 2.93 GiB | `` |
| `module_0109.jit_sigma_sx` | `copy` | 1 | 750.78 MiB | `` |
| `module_0360.jit__do_unfold` | `bitcast` | 14 | 750.78 MiB | `` |
| `module_0360.jit__do_unfold` | `transpose` | 6 | 750.78 MiB | `` |
| `module_0364.jit__do_unfold` | `bitcast` | 13 | 750.78 MiB | `` |
| `module_0364.jit__do_unfold` | `transpose` | 6 | 750.78 MiB | `` |
| `module_0366.jit_broadcast_in_dim` | `bitcast` | 1 | 750.78 MiB | `` |
| `module_0366.jit_broadcast_in_dim` | `copy` | 1 | 750.78 MiB | `` |
| `module_0374.jit__build` | `bitcast` | 11 | 750.78 MiB | `` |
| `module_0039.jit_fn` | `bitcast` | 10 | 375.39 MiB | `` |
| `module_0034.jit_transpose` | `transpose` | 1 | 232.50 MiB | `` |
| `module_0068.jit__identity_fn` | `copy` | 1 | 232.50 MiB | `` |
| `module_0070.jit_transpose` | `transpose` | 1 | 232.50 MiB | `` |
| `module_0073.jit__identity_fn` | `copy` | 1 | 232.50 MiB | `` |
| `module_0077.jit_transpose` | `transpose` | 1 | 232.50 MiB | `` |
| `module_0080.jit__identity_fn` | `copy` | 1 | 232.50 MiB | `` |
| `module_0082.jit_transpose` | `transpose` | 1 | 232.50 MiB | `` |
| `module_0085.jit__identity_fn` | `copy` | 1 | 232.50 MiB | `` |
| `module_0059.jit_fn` | `bitcast` | 10 | 116.25 MiB | `` |
| `module_0109.jit_sigma_sx` | `transpose` | 1 | 116.25 MiB | `` |
| `module_0380.jit__project` | `transpose` | 2 | 116.25 MiB | `` |
| `module_0356.jit_gather` | `bitcast` | 2 | 42.52 MiB | `` |
| `module_0358.jit_broadcast_in_dim` | `copy` | 1 | 42.52 MiB | `` |
| `module_0126.jit_fn` | `bitcast` | 3 | 16.00 MiB | `` |
| `module_0128.jit_squeeze` | `bitcast` | 1 | 16.00 MiB | `` |
| `module_0128.jit_squeeze` | `copy` | 1 | 16.00 MiB | `` |
| `module_0149.jit_gather` | `bitcast` | 2 | 12.00 MiB | `` |
| `module_0151.jit_broadcast_in_dim` | `copy` | 1 | 12.00 MiB | `` |
| `module_0390.jit__add_diag` | `transpose` | 6 | 11.81 MiB | `` |
| `module_0287.jit__per_rank` | `bitcast` | 5 | 8.14 MiB | `` |
| `module_0289.jit__identity_fn` | `copy` | 1 | 8.14 MiB | `` |
| `module_0130.jit__valence_density_kernel` | `bitcast` | 1 | 8.00 MiB | `` |
| `module_0145.jit__broadcast_arrays` | `copy` | 1 | 6.00 MiB | `` |
| `module_0147.jit_broadcast_in_dim` | `bitcast` | 1 | 6.00 MiB | `` |
| `module_0147.jit_broadcast_in_dim` | `copy` | 1 | 6.00 MiB | `` |
| `module_0299.jit__run` | `bitcast` | 17 | 6.00 MiB | `` |
| `module_0115.jit__identity_fn` | `copy` | 1 | 4.50 MiB | `` |
| `module_0407.jit__identity_fn` | `bitcast` | 4 | 4.50 MiB | `` |

## Rematerialization warnings

_None._

## Retrace groups — jit() name → module count

_More than 2 modules for the same jit name means XLA recompiled. Anything above 5 is almost always shape polymorphism — see `retrace_details.txt` for the signatures._

| jit fn | #modules | max peak | Σ peak |
|---|---:|---:|---:|
| `jit_broadcast_in_dim` | 19 | 1.47 GiB | 1.59 GiB |
| `jit_multiply` | 18 | 340.20 MiB | 354.51 MiB |
| `jit__identity_fn` | 13 | 465.00 MiB | 2.31 GiB |
| `jit__per_rank` | 12 | 750.78 MiB | 1.15 GiB |
| `jit_add` | 11 | 13.50 MiB | 35.54 MiB |
| `jit_convert_element_type` | 9 | 768.00 KiB | 960.22 KiB |
| `jit_fn` | 8 | 750.93 MiB | 1010.91 MiB |
| `jit_true_divide` | 6 | 1.25 MiB | 1.25 MiB |
| `jit_gather` | 5 | 752.25 MiB | 1008.15 MiB |
| `jit_transpose` | 4 | 465.00 MiB | 1.82 GiB |
| `jit_subtract` | 4 | 768.00 KiB | 1.13 MiB |
| `jit_conjugate` | 3 | 465.00 MiB | 939.00 MiB |
| `jit__take` | 3 | 24.96 MiB | 30.91 MiB |
| `jit__moveaxis` | 3 | 1.00 MiB | 2.50 MiB |
| `jit_bitwise_and` | 3 | 36.00 KiB | 70.09 KiB |
| `jit_equal` | 3 | 296.00 B | 888.00 B |
| `jit_sum` | 3 | 40.00 B | 120.00 B |
| `jit_sqrt` | 3 | 32.00 B | 64.00 B |
| `jit_concatenate` | 2 | 5.87 GiB | 5.87 GiB |
| `jit__do_unfold` | 2 | 2.25 GiB | 4.50 GiB |
| `jit_squeeze` | 2 | 32.00 MiB | 32.00 MiB |
| `jit__lambda` | 2 | 24.75 MiB | 48.38 MiB |
| `jit__f` | 2 | 4.76 MiB | 5.95 MiB |
| `jit_fft` | 2 | 1.25 MiB | 2.25 MiB |
| `jit_real` | 2 | 768.00 KiB | 768.02 KiB |
| `jit_reshape` | 2 | 512.00 KiB | 1.00 MiB |
| `jit_abs` | 2 | 512.00 KiB | 704.00 KiB |
| `jit_dynamic_slice` | 2 | 288.02 KiB | 288.11 KiB |
| `jit_iota` | 2 | 256.00 B | 512.00 B |
| `jit__multi_slice` | 2 | 40.00 B | 64.00 B |

## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)

| Target | Count |
|---|---:|
| `__cublas$gemm` | 8 |
| `lorrax_phdf5_read` | 6 |
| `lorrax_phdf5_write` | 4 |
| `lorrax_mklfft_flat_k` | 3 |
| `lorrax_phdf5_read_kchunk_union` | 1 |
| `lorrax_mklfft_gw_conv` | 1 |
| `cusolver_syevd_ffi` | 1 |

