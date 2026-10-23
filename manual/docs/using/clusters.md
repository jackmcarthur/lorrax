# Running on clusters

Choose a process count, read the memory plan, and compare with measured GW calculations on Perlmutter.
{: .page-summary }

## Job rules {: #rules }

- **Process count.** Use one MPI rank per GPU and a square total: 1, 4, 9, 16, 25, 36, …. The mesh has equal `x` and `y` dimensions. CPU jobs use the same square rank counts; 64 and 144 ranks give 8×8 and 12×12 meshes. Set the cores and threads per rank explicitly using the [machine launch settings](installation.md#routes).
- **One GPU model and memory class.** Mixed models refuse (`GATE heterogeneous_gpu_targets`). On Perlmutter request `-C 'gpu&hbm40g'` or `-C 'gpu&hbm80g'` for a multi-node GPU job.
- **Shared run directory.** Every rank must see the same run directory (`$SCRATCH` or CFS). A multi-node write to a node-local directory refuses (`GATE io_global_commit`, stage `SlabIO.shared_directory`).

Run centroids → dipole → kinetic/ionic matrix elements → GW under the same environment and with matching inputs. Complete band multiplets at every selected window boundary; widening a requested band count is preferable to cutting a degenerate subspace. Preserve the resolved counts in the reference record.

## The memory budget {: #budget }

The planners share one budget per device. Leave [`memory_per_device_gb`](../drivers/gw/keywords.md#memory_per_device_gb) at zero for the automatic budget; a positive value can lower it. The budget leaves room for runtime contexts, native providers and allocation transients.

| GPU | Automatic budget, `linalg = local` | Automatic budget, `linalg = distributed` |
|---|---:|---:|
| A100-40GB | 31.6 GB | 28.7 GB |
| A100-80GB | 66.4 GB | 63.4 GB |

GB means decimal GB throughout this page. On CPU, the budget starts from node memory divided among its ranks. Increasing ranks on one node divides the arrays but also adds a Python/JAX runtime, compiled programs and host staging per rank. Compare the node's combined host usage as well as the largest rank.

Read `TIMING` and the pool/host peaks in `gwjax.out`, together with the `ISDF μ-batch plan`, selected routes and any `memory over budget` warnings. Verbose reports include the detailed stage-memory, price and sizing receipts. A planner can fill the available budget to run a wider batch: its observed peak is not the minimum memory required by that input. When no route fits, admission warns and selects the least-memory available route; it does not guarantee the allocation will succeed. A compiled executable larger than the device refuses before execution.

The reported device peak measures memory in the device pool. NVML also sees reserved pool space and allocations outside that pool, so its number answers a different question. Keep host high-water and per-node totals alongside the device measurement.

## How many processors? {: #how-many }

Start around $N_\mu = 10N_b$ centroids, then converge the basis. The selector retains complete symmetry orbits, and storage may add dummy slots for the mesh. Use the **packed** centroid extent when estimating storage; keep it distinct from the number of physical interpolation points.

One full-zone complex128 Green's function has size

$$
G = 16\,n_s^2\,N_\mu^2\,N_k \quad\text{bytes},
$$

where $N_k$ counts the **full** k grid and $n_s$ is 1 for scalar, 2 for spinor and 4 for bispinor wavefunctions. A face distribution `G_k(mu_X,nu_Y)` on an $X\times Y$ mesh divides this leading storage by $P=XY$. Symmetry can reduce the stored parent set; full-zone consumers still matter for planning.

**$3G/P$ is a useful first planning target, not a universal peak or a fit guarantee.** Small-k-grid calculations can be dominated by wavefunction staging, ζ-fitting FFT/work buffers, Coulomb factors or solver workspace. Streamed routes may use less. Read the actual compiled/native workspace price for the chosen input and mesh, then measure a complete run. Do not infer a smaller working process count by dividing a measured peak by card capacity.

For many-band runs, inspect the wavefunction route too. A two-axis distribution such as `psi_k(n_X,s,mu_Y)` reduces both large axes, giving leading storage proportional to $1/P$. A faster band-complete carrier `psi_k(n,s,mu_Y)` scales as $1/\sqrt P$; bounded panels offer another way to stage a contraction. Record the actual selected carrier and panels: they have different memory and communication costs.

## Local or distributed linear algebra {: #linalg }

`linalg = local` requests whole-matrix solves on individual ranks. The services price those matrices and their workspace and can use the mesh route when they do not fit. `linalg = distributed` requests matrix solves across the mesh. Both operate on distributed inputs and outputs; the distinction is where the solve itself happens.

Whole-matrix solves are often faster when several independent q matrices fit per rank. Distributed solves become useful when one matrix and its workspace are too large. Read the selected route in the report instead of inferring it from the deck alone.

## Reference runs on Perlmutter {: #examples }

These are fresh fixed-input $G_0W_0$ runs from **23 October 2026**, pinned to source [`2d2ca2f104`](https://github.com/jackmcarthur/lorrax/commit/2d2ca2f104a085876a573e8b1278a7874c479b98). They use native full head corrections, default accuracy and automatic batch/layout selection. They establish completed workloads, not k-grid/band/basis convergence or minimum processor counts.

Wall time is the GW driver's `total run`, including setup and compilation but excluding centroid/dipole/kinetic preprocessing, queue wait and allocation startup. All GPU rows use A100-80GB cards and the automatic local-linalg budget (66.35 GB in the saved receipts). Persistent compilation caches were shared; the counts below make their state explicit.

| System | Full k grid / stored parents | Occupied bands | Sum bands | Physical $N_\mu$ | $n_s$ | Method | Resources | GW wall | Pool peak/rank |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| Si SOC, 2 atoms | 4×4×4 / 8 | 8 | 52 | 528 | 2 | GN-PPM | 4 GPUs, 1 node | 59.52 s | 9.71 GB |
| Al₂O₃, 10 atoms | 4×4×4 / 13 | 24 | 154 | 1584 | 1 | GN-PPM | 4 GPUs, 1 node | 39.23 s | 19.03 GB |
| Ferromagnetic bcc Fe | 8×8×8 / 59 | 18 (fractional) | 110 | 1099 | 2 | Shared-pole | 4 GPUs, 1 node | 353.07 s | 33.43 GB |
| Cr:Al₂O₃, 80 atoms | 2×2×2 / 4 | 395 | 2370 | 23700 | 2 | GN-PPM | 16 GPUs, 4 nodes | 915.12 s | 52.89 GB |

| Run | Setup | ISDF / Coulomb | Screening | Self-energy | QP update | Real compiles / cache hits | Summed compile time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Si SOC | 13.27 s | 25.96 s | 9.21 s | 10.41 s | 0.58 s | 134 / 126 | 29.7 s |
| Al₂O₃ | 14.31 s | 13.88 s | 4.48 s | 5.96 s | 0.57 s | 10 / 271 | 2.4 s |
| Fe | 18.59 s | 63.29 s | 176.44 s | 87.07 s | 5.94 s | 497 / 29 | 139.73 s |
| Cr:Al₂O₃ | 16.04 s | 326.67 s | 139.50 s | 426.11 s | 6.49 s | 310 / 11 | 59.7 s |

The Si and Al₂O₃ peaks are the rounded production pool high-water, maximized over ranks; both occur in ζ fitting. They are not exact native-pointer ledgers or total physical GPU memory. Their maximum host `ru_maxrss` is 2.43 and 4.46 GB per rank, respectively. Packed centroid extents were not emitted in these saved reports. The runs have different cache histories, so their wall times are not a controlled performance comparison.

The Si request was widened from 48 to 52 sum bands to complete a multiplet. The Al₂O₃ request was widened from 144 to 154 (156 padded band slots); the requested QP matrix covers bands 1–30. Its dynamic Σ grid begins at −24 eV to cover the deep valence states; the final run reports complete coverage. The earlier Al₂O₃ pilot with static fallback for deep states is excluded from this table.

Fe uses 110 physical sum bands (112 padded slots), 1099 physical centroids (1100 packed slots at P4), and the production shared-pole recipe with a pole budget of $1.0N_\mu$: 1099 poles per q from 35 samples. Band 18 is the occupied edge; the finite-temperature density tails are retained. The requested 4-valence/4-conduction frontier gives a Σ/QP prefix of bands 1–22. Its compile row excludes eight additional runtime-startup compiles; compile seconds are cumulative API time on rank 0, not a sum over MPI ranks or a critical-path decomposition.

Fe's 33.43 GB is the post-startup pool-**used** high-water. The default allocator also maps a **77.895 GB pool reservation** at startup, with context/vendor storage outside it; physical process high-water was not sampled. An exact XLA-request ledger gives 33.434 GB after startup and excludes direct native allocations. These measurements do not establish a 40 GB fit. The largest host `VmHWM` is 5.53 GB/rank.

Fe retains the default `sigma_reads_past_w_support` warning. Its Σ energy grid covers the requested bands; the warning concerns the W model's sampled support range and remains relevant to a separate accuracy study.

The 80-atom case is **Al₃₁CrO₄₈ with SOC**: 395 occupied spinor bands, 2370 physical sum bands (2372 padded slots), and 23700 physical centroids (23712 packed slots at P16). A 4-valence/4-conduction frontier still produces the full 1–399 Σ/QP prefix. The explicit −75…2.75 eV Σ range covers that prefix without out-of-grid static fallback; 831 quadrature nodes take 388.23 s. The default GN prescription treats 3.02% invalid modes with `static_limit`, a separate issue from energy-grid coverage.

Its maximum pool-used peak is 52.89 GB/rank in ζ fitting, versus 41.39 GB in screening and 44.85 GB in Σ. One full-zone Green's function at the packed extent is 17.992 GB/rank, so this run peaks at **2.94G/P**. Host high-water is 15.97 GB/rank. The conservative Cplus quote exceeded the then-current planning room; the completed run stayed below the 66.35 GB budget. This demonstrates the stated 16-GPU workload; a smaller process count was not tested. Original structural/symmetry qualifications and the exact inputs are retained in the [case notes](../assets/cluster-references/2026-10-23/cr_al2o3_80/case_notes.txt).

All ranks exited successfully and the canonical Σ and QP outputs were checked for the stated shapes and finite entries. Preserve the input hashes, exact decks, resolved geometry, selected routes, compilation counts and all-rank exit records when repeating a reference.

The [reproduction notes](../assets/cluster-references/2026-10-23/README.txt) describe the input arrangement and launch sequence. Exact decks and manifests: [Si](../assets/cluster-references/2026-10-23/si_soc/gw.in) / [record](../assets/cluster-references/2026-10-23/si_soc/manifest.json), [Al₂O₃](../assets/cluster-references/2026-10-23/al2o3_10/gw.in) / [record](../assets/cluster-references/2026-10-23/al2o3_10/manifest.json), [Fe](../assets/cluster-references/2026-10-23/fe_8x8x8/gw.in) / [record](../assets/cluster-references/2026-10-23/fe_8x8x8/manifest.json), [Cr:Al₂O₃](../assets/cluster-references/2026-10-23/cr_al2o3_80/gw.in) / [record](../assets/cluster-references/2026-10-23/cr_al2o3_80/manifest.json). The text bundle includes DFT decks and physical centroid tables; large wavefunction and generated matrix files remain in the NERSC evidence archive named in each record.
