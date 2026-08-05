# Multi-node distributed exciton-band pipeline (16 GPU / 4 node)

Branch `agent/bse-multinode` off `agent/bse-integration`, worktree
`lorrax_A_bse_integration`. Allocation JID 56206654 (4 nodes / 16× A100).

Goal: make LORRAX's exciton-bandstructure pipeline (V_Q interpolation + BSE
solve) natively multi-node distributed, run the C_q eigh through the linalg
FFI (cusolverMp) on the 2D-sharded tiles, validate on 16 GPUs, and produce the
flag-on 12×12 MoS2 exciton bandstructure. Head correction stays serial (cheap
per-Q host QMC, deterministic).

## Launch model (forced by cusolverMp: one JAX process per device)

16 processes, 1 GPU each — `srun -N4 -n16 --gres=gpu:4 select_gpu.sh` (sets
`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`), the EXACT proven layout the production
`gw.gw_jax` 16-GPU run uses (`runs/VI3/.../run_vi3_lorrax.sh:25`). NOT
`--gpus-per-task=1` (breaks JAX topology sync — `ffi/common/cpp/run_shifter.sh`
note). `jax.distributed.initialize()` auto-detects the 16-proc topology from the
SLURM env. Square 4×4 mesh (cusolverMp requires p==q). Scripts: `run11.sh`
(launcher), `run_exciton_16gpu.sh off|on` (driver wrapper), both in dir 11.

## The 5 gaps — verified + fixed

1. **Distributed init** — `bse/exciton_bands.py`: added the SINGLE-SOURCED
   `runtime` bootstrap (`set_default_env()` before `import jax`;
   `init_jax_distributed()` + `fallback_to_cpu_if_no_gpu_backend()` after),
   the exact pattern `gw.gw_jax` uses. NO new helper created — `runtime/__init__`
   already owns the SLURM-aware idempotent init (it explicitly replaced 5
   drifting copies); adding a `common/` helper would be a 6th copy. No-op in
   single-process (sentinel + proc_count<=1), so the 1-GPU path is unchanged.

2. **Multi-process launch harness** — `run11.sh` mirrors the proven
   `gw.gw_jax` FFI launch, PYTHONPATH → this worktree. `run_exciton_16gpu.sh`
   parameterises OFF/ON + `EIGH` backend.

3. **Rank-0 I/O guards** — `bse/exciton_bands.py`: added a rank-0 `log()`
   (threaded into the heavy helpers as `log_fn` so progress isn't printed 16×)
   and wrapped the entire outputs block (`.dat` write, matplotlib plot +
   `savefig`, timing summary) in `if jax.process_index()==0:`. Added a `[dist]`
   banner (device_count / process_count / mesh) and an `[dist] evs sharding`
   line as gate-#1 / gate-#2 proof.

4. **Host↔device sharded placement** — probed empirically on 16 GPUs
   (`probe_dist.py`): `jax.device_put(host_numpy, sharded)` IS multiprocess-
   correct in JAX 25.04 (slices the local shard), so `prepare_coarse` /
   `compute_wfns_fi` / `V_stack` device_puts need no change. The REVERSE
   (device→host gather) was the real gap:
     - `gate_htransform_vs_stored` gathered μ-sharded ψ_c via `device_get` →
       fails on process-spanning shards. Added a sharding-aware `_gather_host`
       (device_get when fully-addressable, `process_allgather` when shards are
       remote) — a naive `process_allgather(tiled=True)` on a REPLICATED array
       DUPLICATES its leading axis (144→2304), so the `is_fully_addressable`
       branch is required.
     - `vq_interp.prepare_coarse` gathered the q-sharded ζ-clean tiles
       (`S_b/V_b/F_b`, `out_shardings=P(('x','y'),...)`) via `device_get` → same
       fix (`_to_host`, sharding-aware).
     - valence pad-ε guard: replaced a `device_get→jnp.asarray` round-trip
       (fails on a sharded shard AND drops sharding) with an on-device
       `jnp.where`.
     - `evs` gathers routed through `_gather_host` (defensive; Ritz values are
       replicated).

5. **Head correction serial** — `minibz_head_vlr` is pure host numpy with
   deterministic Sobol QMC (`seed_offset=0`), computed redundantly on every
   process → identical operands. No change needed (verified).

## Extra distributed gap found + fixed (htransform galerkin)

`common/wfn_transforms.py:gflat_to_rmu` band-flat-shards `P(None,('x','y'),...)`
so it required `nb % mesh.size == 0`. The htransform SP/galerkin entry
(`initialize_wfns→streaming_galerkin_solve`) passes an un-rounded band window
(nb=40), which divides 4 (single-node) but not 16 → hard error on 16 devices.
Fix: pad the band axis with ZERO bands (ψ=0 → zero centroid samples, trimmed
from the output), then trim; no-op when nb already divides the mesh (single-node
byte-identical). This is the SP-driver counterpart of the GW path's
`Meta._round_up(world_size)`.

## cusolverMp FFI eigh — VALIDATED on 16 GPU (proof, gate 5a)

`common.cusolvermp_eigh_test --grid 4 4 -n 640` (the exact C_q tile size):
```
[lorrax cusolverMp] library 0.7.2, NCCL 2.26.3, comm path: NCCL, grid: 4x4 (col-major)
complex128 Hermitian 640x640:  max |eval - ref| = 3.865e-12  PASS
float64  symmetric 640x640:    max |eval - ref| = 5.230e-12  PASS
```
The banner proves the FFI context initialized (NCCL comm path, 4×4 grid) — NOT a
silent fallback (the wrapper `raise`s on failure; `_eigh_backend` has no
try/except native fallback). Eigenvalues bit-accurate. The large per-vector
"residuals" the test prints are a harness artifact (raw Q, not `conj().T`).

Run config uses `--eigh-backend cusolvermp` (NOT `off`): the C_q Hermitian eigh
runs through the FFI on the 2D-sharded (`P('x','y')`) 640×640 tiles, per the
owner's "all complex linear algebra on 2D-sharded matrices via the linalg FFI".

## BSE solve linalg audit (coordinator item 6 — report only)

- `solvers.lanczos.block_lanczos_eig_jit`: the projected Rayleigh-Ritz
  `eigh(T)` is small (≤ block·iter = 8·40 = 320²) and replicated — native is
  correct, no FFI needed. Block-QR (`jnp.linalg.qr`) is tall-skinny
  (n_flat×block_size), cheap.
- `bse.bse_stack_matvec`: the W-convolution and V-exchange contractions are
  `jnp.einsum` on 2D-sharded (μ/ν on x/y) operands with `with_sharding_constraint`
  — XLA sharded `dot_general` (distributed via collectives), NOT cublasmp.
- **Recommendation**: keep the matvec as sharded `dot_general`. These are
  BATCHED multi-index pair-basis contractions (k,c,v,μ,ν), not single large
  dense GEMMs; cublasmp targets large 2D dense `C=A@B`. Routing them through
  cublasmp would force 2D reshapes + per-batch FFI calls and lose XLA fusion of
  the ring-comm reduce-scatter pattern. The FFI is the right tool for the C_q
  eigh (a genuine dense 2D Hermitian decomposition, now via cusolverMp) and for
  dense W-solve LU/Cholesky — not for these batched einsums.

## Files changed
- `src/bse/exciton_bands.py` — dist init, rank-0 log/IO guards, `_gather_host`,
  gate + pad-guard + evs gathers, `[dist]` proof lines, `--eigh-backend` flows
  through (already wired to `prepare_coarse`).
- `src/bse/vq_interp.py` — `_to_host` (sharding-aware gather) for the ζ-clean
  tiles in `prepare_coarse`.
- `src/common/wfn_transforms.py` — `gflat_to_rmu` band-axis zero-pad to the mesh.
- `tests/test_runtime_distributed.py` — new unit test for the `runtime` bootstrap.

## Results

### Gate 1 — 16 devices really used — PASS
Both OFF and ON runs logged:
`[dist] jax.device_count()=16 process_count()=16 local_device_count()=1; mesh_xy.shape={'x': 4, 'y': 4} (px=4, py=4)`

### Gate 2 — numerics match dir 10 (backend must not change physics) — PASS (bit-identical)
OFF 16-GPU/cusolverMp vs dir10 4-GPU native `--eigh-backend off`
(`exciton_bands_40interp_8v8c.dat`), 40 Q × 8 eig:
**GLOBAL max|ΔE| = 0.0000e+00 meV** (identical to the .dat 6-decimal / sub-µeV
precision); E_1(Γ)=1.141460 eV both. The cusolverMp FFI does NOT change the
physics. (`logs/numerics_off_vs_dir10.log`, `off_vs_dir10.cmp.npz`)

### Gate 3 — flag-ON (mini-BZ head avg) vs flag-OFF (point value), 16 GPU — PASS
**GLOBAL max|ΔE| = 0.6750 meV** at iQ=38 (|Q|=0.0295, near Γ); mean 0.0051 meV;
near-Γ first-4-Q max: 0.575, 0.066, 0.018, 0.009 meV. The shift is small and
near-Γ localized — the expected mini-BZ Coulomb-head signature (same order as the
≤0.38 meV reference; this is the 12×12 8v8c config). Overlay:
`exciton_bands_16gpu_on_vs_off.png`; deltas `exciton_bands_16gpu_on_vs_off.png.npz`.

### Gate 4 — timing (wall, s)
| stage | dir10 4-GPU native off | 11 OFF 16-GPU cusolverMp | 11 ON |
|-------|----------------------:|--------------------------:|------:|
| load_bse          | 3.4 | 2.35  | 2.83  |
| htransform_psi_cQ | 155 | 15.5  | 15.9  |
| vq_prepare        | 59  | 398.3 | 398.4 |
| vq_eval           | 2.1 | 1.04  | 20.3  |
| solve_scan cold   | 202 | 165.2 | 163.1 |
| solve_scan warm   | 199 | 162.0 | 162.9 |
| TOTAL             | 636 | 758.0 | 778.1 |

Scaling: htransform **10× faster** and the block-Lanczos solve **faster**
(202→165 s cold) on 16 GPU — good scaling. `vq_prepare` **6.7× slower**
(59→398 s): the 144 per-q cusolverMp eighs on small 640×640 tiles are NCCL-
latency-bound across 4 nodes (the expected "cross-node collectives don't help a
small case" regime). Correctness on the distributed FFI is the point, per "ships
to arbitrary device counts"; a batched-across-q FFI eigh would recover this. ON
`vq_eval` +19 s = the per-Q Sobol-QMC mini-BZ head (deterministic, on all procs).

### Gate 5 — golden gates + distributed-init test
`tests/test_runtime_distributed.py` (new) + touched modules (test_wfn_transforms,
test_bse_vq_interp, test_exciton_bands) + named golden gates
(test_gw_jax_regression, test_symmetry_unfold, test_head_correction,
test_minibz_average) on 1 GPU. Result: **59 passed, 0 failed, 0 errors** in 123 s
(5 pre-existing deprecation warnings). `logs/gates_1gpu.log`.

## Matvec is genuinely distributed — interpolation HLO collective audit (owner request)
Compiled HLO of `eval_vq` and `_clean_split` on the 4×4 mesh
(`hlo_interp_audit.py`, artifacts `HLO_AUDIT.md` + `logs/hlo_*_16gpu.txt`):
- **eval_vq** (`V = V_SR + conj(A_x) @ A_y.T`): only 3 `collective-permute` on
  `c128[160,337]` — the two SMALL A reshards (A_x=P('x',·) rows, A_y=P('y',·)
  cols, nG=337 contracted replicated). The outer product is a **LOCAL dot** →
  (640,640) P('x','y'). **No collective carries the n_μ=640 tile dim.**
- **_clean_split** (`R g Rᴴ`, `Sc@V_delta@Sc`, batched over q at qb3): **ZERO
  collectives** — per-q LOCAL matmuls (μ,ν replicated per device, q the only
  sharded axis).
- cusolverMp seam: R gathered from `P('x','y')` (160×160/dev) to qb3 replicated
  (640×640/dev), 6.2 MiB/tile, ~18.8 MiB/dev per 48-q chunk — the expected
  "FFI-eigh then local per-q reconstruction", cheap at n_μ=640.
- **VERDICT: the interpolation tile assembly is a LOCAL outer product; no
  full-n_μ×n_μ 2D reshard / all-gather. Genuinely distributed.**
