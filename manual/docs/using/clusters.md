# Running on clusters

How many GPUs a calculation needs: the job rules, the per-GPU budget, a Green's-function size estimate, and real Perlmutter runs.
{: .page-summary }

## Job rules {: #rules }

- **Process count.** One process per device, and the device count a perfect square: 1, 4, 9, 16, 25, 36, …. Any other count refuses at startup and names the two nearest squares; 32 GPUs (8 Perlmutter nodes) is not a legal job, 36 (9 nodes) is.
- **One GPU model and memory class.** Mixed models refuse (`GATE heterogeneous_gpu_targets`). On Perlmutter a multi-node job requests `-C 'gpu&hbm40g'` or `-C 'gpu&hbm80g'`.
- **Shared run directory.** A multi-node run writes into a directory every node mounts (`$SCRATCH`, CFS). A node-local one (`/tmp`, local NVMe) refuses at the first write (`GATE io_global_commit`, stage `SlabIO.shared_directory`).

Launch lines per machine are on the [installation page](installation.md#routes).

## The per-GPU budget {: #budget }

Every planner prices its stage against one budget per device, set from the card's memory before the deck is read:

| Card | Budget, `linalg = local` | Budget, `linalg = distributed` |
|---|---|---|
| A100-40GB | 31.6 GB | 28.7 GB |
| A100-80GB | 66.4 GB | 63.4 GB |

[`memory_per_device_gb`](../drivers/gw/keywords.md#memory_per_device_gb) can only lower it; a larger value warns and keeps the rule. No stage refuses on a price. A stage priced over the budget prints `memory over budget at <stage>: needs … GB/rank, budget … GB/rank, over by … GB; continuing (an OOM is possible)`, listed under `WARNINGS` in the report, runs at its smallest tile, and fails at allocation if the memory is really missing. Large objects are tiled over every rank, so the fix is more ranks or larger cards. A compiled program larger than the card refuses before it runs (`GATE xla_rematerialization`), with the same fix.

After a GW run, the `MAJOR-STAGE DEVICE AND HOST MEMORY` block of `gwjax.out` lists each stage's device peak (max and min over ranks), its planner price and γ = peak / price, and the host high-water. A stage with γ > 1 was under-priced; the stage that set the run's peak is the one to size against. Some planners (the ζ batch, the response sample group, the tangential panels) size their tiles from the budget, so a run's peak grows with the card and is an upper bound on what it needs.

## How many GPUs? {: #how-many }

**1. Centroids.** Take $N_\mu \approx 10\times$ the number of bands as a starting reference (Ni 20³: 120 bands, 1200 centroids; Fe 20³: 184 bands, 1792). Converge it with the [centroids utility](../utilities/centroids.md).

**2. One Green's function.** In complex128,

$$
G = 16\,n_s^2\,N_\mu^2\,N_k \ \text{bytes},
$$

with $n_s$ the spinor count (1 scalar, 2 spinor, 4 bispinor) and $N_k$ the **full** k grid. The Greens are stored on the irreducible parents, but V, W and the Σ passes are unfolded on the full grid, and the code sizes against it: the ζ-fit receipt prints `G_tile unit = … GB/dev (nk·ns²·μ²·16/P)`, which is $G/P$. On a grid with no symmetry the two counts agree; on a cubic metal the full grid is 12× the irreducible one (Ni 20³: 8000 vs 641).

**3. Total memory.** LORRAX's own multiple is **3**: the tangential screening and Σ ledger holds its live set within $3U$, $U = G/P$ per rank, and warns above it; the ζ fit's feasibility ceiling is $4\,G/P$. Below $3G$ the stages tile and stream instead of failing, so the multiple is a comfort target, not a floor. The 20³ metal runs completed with $G/P$ at about 1.2× the card (47 GB per 40 GB A100).

| Use | GPUs |
|---|---|
| comfortable | smallest square $P$ with $P\cdot B \ge 3G$ |
| measured floor (20³ metals) | $G/P \lesssim 1.2\times$ card memory |

$B$ is the budget per GPU from the table above.

**Worked examples.**

| System | $n_s$ | $N_\mu$ | $N_k$ | $G$ | $3G$ | comfortable on A100-80 |
|---|---|---|---|---|---|---|
| Si SOC, 6×6×6 | 2 | 996 | 216 | 13.7 GB | 41 GB | 1 |
| MoS₂, 9×9×1 | 2 | 1200 | 81 | 7.5 GB | 22 GB | 1 |
| Ni, 20×20×20 | 2 | 1200 | 8000 | 737 GB | 2.2 TB | 36 (9 nodes) |
| Ni bispinor, 20×20×20 | 4 | 1200 | 8000 | 2.95 TB | 8.8 TB | 144 (36 nodes) |

**4. Small k grids.** When $G$ is small the ζ fit sets the peak, not the Green's functions: it holds ψ(G) on the rank's G slots, $16N_kN_bn_sN_{G\psi}/P$, and its μ-batch working set. CrI₃ 6×6×1 has $G = 2.2$ GB, yet its ζ fit asks 70 GB per GPU at P = 4. Read the `ISDF μ-batch plan` receipt in `gwjax.out`; when the fit's price is above `feasibility ceiling 4·G_tile`, the fit sets the node count.

## Local or distributed linear algebra {: #linalg }

The deck key `linalg` (default `local`) sets the layout of every dense solve. Under `local` each GPU solves whole $N_\mu\times N_\mu$ matrices for its share of the irreducible q points, with no distributed factorization; under `distributed` each matrix is factored over the whole mesh (cuSOLVERMp), which divides its memory by $P$ but communicates inside every factorization.

`local` is a request, not a guarantee. At each W Dyson solve the code prices the local route, the face-tiled V, χ and W plus whole copies for $\lceil Q_{\rm irr}/P\rceil$ q points and one LU pair,

$$
\frac{48\,Q_{\rm irr}N_\mu^2}{P} + 16N_\mu^2\left(3\lceil Q_{\rm irr}/P\rceil + 2\right)\ \text{bytes per GPU},
$$

and runs it only if it fits the budget; otherwise it falls back to the distributed LU. Eigensolves choose the same way between whole matrices per GPU and the whole-mesh eigensolver, by fit and modelled speed. The choice prints once as `route W Dyson …: local` (or `distributed`) and appears in the report's Routes table.

More GPUs therefore do two things: they divide every tiled object, and they shrink $\lceil Q_{\rm irr}/P\rceil$ until the whole-matrix route fits, so the solves stop communicating. Past $P = Q_{\rm irr}$ the ζ charge factor has no more q points to spread, and the run says so. Set `linalg = distributed` only when a few whole $N_\mu\times N_\mu$ matrices do not fit one GPU.

## Example runs on Perlmutter {: #examples }

A100 runs from October 2026. Wall time is the driver's `total run` line (start of the process to the report; no queue wait). The minimum is measured where a smaller run exists; otherwise (*) it is the smallest square $P$ whose total budget covers the run's summed per-GPU peaks, which over-states the need because some planners fill the budget.

| System | k grid | Bands | $N_\mu$ | $n_s$ | Driver, method | GPUs used | Minimum | Wall |
|---|---|---|---|---|---|---|---|---|
| Si | 4×4×4 | 34 | 2952 | 1 | gw, MPA one-shot | 4 × 80 GB | 4 × 40 GB* | 338 s |
| Si SOC | 6×6×6 | 60 | 996 | 2 | gw, HL-PPM one-shot | 4 × 40 GB | 1 × 80 GB* | 79 s |
| Si SOC | 10×10×10 | 8 v + 8 c | 996 | 2 | bse, Davidson, 16 roots | 4 × 40 GB | 1 × 80 GB* | 86 s |
| MoS₂ | 9×9×1 | 80 | 1200 | 2 | gw, COHSEX | 4 × 80 GB | 4 × 40 GB* | 26 s |
| CrI₃ | 6×6×1 | 160 | 980 | 2 | gw, MPA one-shot | 4 × 80 GB | 4 × 80 GB (ζ fit already over budget) | 83 s |
| CrI₃ | 8×8×1 | 42 v + 14 c | – | 2 | bse, Davidson, 112 roots | 4 × 80 GB | not logged | 539 s |
| Fe | 4×4×4 | 36 | 4220 | 2 | gw, MPA one-shot | 4 × 80 GB | 4 × 40 GB* | 1188 s |
| Ni | 20×20×20 | 120 | 1200 | 2 | gw, MPA self-consistent | 64 × 40 GB | 16 × 40 GB | 4691 s |
| Ni bispinor | 20×20×20 | 120 | 1200 | 4 | gw, MPA self-consistent | 64 × 80 GB | 64 × 40 GB | 3604 s per SC leg |

Notes on the 20³ metals (1200 centroids, padded to 1216 at P = 64):

- **Ni, spinor.** A 16 × 40 GB run completed one SC iteration in 1215 s with a 36.6 GB peak per GPU, just over its budget. So it fits on 16.
- **Ni, bispinor.** It does not fit on 16: $G/P$ is 184 GB per GPU there. On 64 × 40 GB one SC leg completed in 3537 s (37.6 GB peak, over budget with a warning). On 64 × 80 GB the ledger asked 102 GB per GPU at the Σ sector stage and the peak reached 76 GB. 144 GPUs would be comfortable.

