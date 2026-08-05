# N_μ²-distribution audit (owner directive)

**Claim proven:** in the 2-D-distributed cuBLASMp reconstruction path
(`bse.vq_interp._recon_body` + `_distributed_prims` + `_recon_distributed_chunk`),
EVERY intermediate of size ∝ n_μ² is sharded across ALL P = Px·Py = 16 procs of
the full 4×4 mesh (per-proc shard = n_μ/Px × n_μ/Py). NONE is replicated `P()`,
half-distributed `P(None,'x')` / `P(None,'y')`, or on a sub-mesh. The only
replicated operands are O(n_μ) or smaller (eigenvalues λ, filter weights g).

Proven from the ACTUAL per-proc `addressable_shards[0].data.shape` and the
`.sharding.spec` on 16 GPUs (`scripts/recon_shard_audit.py`,
`logs/shard_audit.log`), not by assertion. n_μ = 640, mesh 4×4, so a replicated
tile would be 640×640 (6.25 MiB) per proc; the full-mesh shard is 160×160.

## Per-array shard table (n_μ=640, 4×4 mesh)

| tensor | role | logical shape | PartitionSpec | per-proc shard | verdict |
|--------|------|---------------|---------------|----------------|---------|
| Qraw (evec) | eigenvectors (raw cusolverMp buffer) | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| S      | R g_ε Rᴴ (Tikhonov filter)      | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| Sc     | conj(S)                          | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| A_ref  | ζ̃·√v_ref (sphere factor)         | (2, 640, 340) | `P(None,'x','y')` | **(2, 160, 85)**  | FULL-MESH 2-D |
| A_lr   | ζ̃·√v_lr                          | (2, 640, 340) | `P(None,'x','y')` | **(2, 160, 85)**  | FULL-MESH 2-D |
| V_delta| conj(A)Aᵀ, V_ref−V_LR            | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| T1=Sc@Vδ| Sc·V_delta (intermediate)       | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| V_SRc  | conj(S) Vδ conj(S) (SR tile)     | (2, 640, 640) | `P(None,'x','y')` | **(2, 160, 160)** | FULL-MESH 2-D |
| zt     | S·ZG (cleaned ζ̃ on sphere)       | (2, 640, 340) | `P(None,'x','y')` | **(2, 160, 85)**  | FULL-MESH 2-D |

(Batch axis 0 = q-chunk here 2; the n_μ² is on axes 1,2 and both are sharded:
640/Px = 160 on x, 640/Py = 160 on y; the sphere-factor G axis 340/Py = 85.)

**Allowed replicated (O(n_μ), not n_μ²):**

| tensor | logical shape | PartitionSpec | per-proc |
|--------|---------------|---------------|----------|
| lam (eigenvalues) | (2, 640) | `P()` | (2, 640) |
| g (filter weights)| (2, 640) | `P()` | (2, 640) |

**VERDICT: ALL n_μ² tensors FULL-MESH 2-D sharded (9/9).** No tile lands whole
(640×640) or half-distributed (640×160 / 160×640) on any proc.

## Why each stays full-mesh (design, matches `_distributed_prims`)

- **Qraw** comes straight from `cusolvermp.distributed_eigh` at `P('x','y')`
  (160×160/dev) and is stacked + `with_sharding_constraint(P(None,'x','y'))` —
  no gather to replicated (this is the multinode seam that was KILLED).
- **S = R g Rᴴ** via `batched_distributed_gemm(Qraw, g⊙Qraw, transa='C')`:
  cuBLASMp `out_specs=P(None,'x','y')`, and `transa='C'` reads Qraw as-is so R =
  conj(Qraw)ᵀ is never materialised (no transpose reshard, no replicated temp).
- **V_delta / V_SRc / zt** GEMMs: cuBLASMp in/out specs are all
  `P(None,'x','y')`; the pre-transposed operand (Aᵀ for the gram-outer, needed
  because cuBLASMp forbids op(B)≠N on a multi-rank grid) is produced by
  `swapaxes` + `with_sharding_constraint(P(None,'x','y'))` — a full-mesh
  all-to-all that stays 2-D sharded, never replicated.
- **Inside the cuBLASMp `shard_map`**: `check_rep=False`, `in_specs`/`out_specs`
  = `P(None,'x','y')`; the local pre-transpose (`jnp.transpose(local_X)`) acts on
  the per-proc (n_μ/Px × n_μ/Py) shard only, and the cuBLASMp workspace is
  per-proc block-cyclic — no full tile is ever assembled on one rank.
- **λ, g** are the eigenvalues / spectral-filter weights — O(n_μ), correctly
  replicated `P()` (broadcast into the row-scale of Qraw, a local elementwise op).

## HLO-level confirmation (S-gram GEMM, `logs/recon_gram_hlo.txt`)

The compiled S = R g Rᴴ GEMM confirms the distribution in the sharding
annotations, not just the runtime shard shapes:

    HloModule ... num_partitions=16 ...
    %param.3   = c128[2,160,160] parameter(0), sharding={devices=[1,4,4]<=[16]}  # a = Qraw
    %param.1.0 = c128[2,160,160] parameter(1), sharding={devices=[1,4,4]<=[16]}  # b = g⊙Qraw
    %param.2.0 = c128[2,160,160] parameter(2), sharding={devices=[1,4,4]<=[16]}  # c
    %custom-call.10.0 = c128[2,160,160] custom-call(...) custom_call_target="lorrax_cublasmp_batched_gemm"
        backend_config={ transa=2 (C), transb=0 (N), m=640, n=640, k=640,
                         mb_a=nb_a=lld_a=160, mb_c=nb_c=lld_c=160, nq=2 }

`sharding={devices=[1,4,4]<=[16]}` = batch axis unsharded, the two n_μ axes split
4×4 over all 16 devices; the cuBLASMp per-rank block is 160×160 (`mb=nb=lld=160`),
`transa=2` = 'C' (the raw-buffer S-gram with no transpose reshard). No 640×640
tile appears anywhere in the module.
