# Interpolation-linalg HLO collective audit (16-GPU 4x4 mesh)

n_mu=640 (big ISDF tile dim), nG=337, nq=144, mesh 4x4.


### eval_vq (per-Q tile V = V_SR + conj(A_x) @ A_y.T)

- `collective-permute` c128[160,337] c128[160,337]  (small)
- `collective-permute` c128[160,337]  (small)
- `collective-permute` c128[160,337]  (small)

collective counts: collective-permute=3

**CLEAN: no collective operand carries the n_mu=640 tile dim.**

### _clean_split (R g R^H, Sc@V_delta@Sc, batched over q at qb3)


collective counts: none

**CLEAN: no collective operand carries the n_mu=640 tile dim.**

### cusolverMp eigh -> reconstruction seam
For `--eigh-backend cusolvermp`, `distributed_eigh` returns R at `P('x','y')` (2D-distributed, 160x160=25600 elems/dev per q). `_clean_split` consumes R at qb3 = `P(('x','y'),None,None)` (mu,nu REPLICATED, only q sharded) -> R is GATHERED from 2D-sharded to replicated n_mu x n_mu per q. Gather = one full 640x640 c128 tile (6.2 MiB) per q onto its owner device (batched: ~18.8 MiB/device per 48-q chunk). So the reconstruction R g R^H is NOT 2D-distributed — it replicates the mu/nu tile and batches over q. Fine at n_mu=640 (cheap dense per-tile); it is the expected 'FFI eigh then local per-q reconstruction' seam.

## VERDICT: interpolation tile assembly is LOCAL-OUTER-PRODUCT (no 2D x 2D reshard of a full n_mu x n_mu tile)

eval_vq: V=V_SR+conj(A_x)@A_y.T is a local outer product (A_x=P('x',·) rows, A_y=P('y',·) cols, nG contracted replicated); the only comms are the two small A reshards (n_mu x nG ~ 3.3 MiB). _clean_split: R g R^H and Sc@V_delta@Sc are per-q LOCAL matmuls (mu,nu local, q is the only sharded axis).
