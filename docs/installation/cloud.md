# Generic cloud GPU (AWS, RunPod, Vast, Lambda, …)

How to run LORRAX on a rented cloud GPU with no Cray PE and no site module. This covers the
**pure-JAX track** (centroids, wavefunction loading, serial/single-node GW) — the part that
needs only a recent NVIDIA driver and `uv sync`. The distributed FFI stack (multi-node
`eigh`, sharded HDF5, SLATE) is a separate, larger job — see
[FFI native libraries](ffi-native-libs.md).

!!! success "Verified off-NERSC"
    The pure-JAX GPU install was reproduced on a non-NERSC box (consumer Blackwell GPU,
    WSL2, driver 595 / CUDA 13.2): `uv sync` resolves the CUDA-13 wheels, JAX reports
    `backend gpu`, and the core test suite passes (168 passed / 37 skipped). The bundled
    end-to-end GW fixture runs off-NERSC **when it fits in memory** — see the sizing note
    below.

## Two things decide whether it runs: driver and memory

**Driver.** LORRAX pins `jax[cuda13]`; the CUDA-13 wheels bundle the CUDA runtime, so you
do **not** install a CUDA toolkit — but the host needs **NVIDIA driver ≥ 580**. On the
container clouds (RunPod, Vast) the driver belongs to the host and you cannot change it from
inside a container, so *filter for a driver-580+ host* if you want the pinned stack. A
CUDA-12 fallback on the same JAX-0.9 line works on the near-universal driver ≥ 525, but is
not offered as a package extra yet (`pyproject.toml` is authoritative — see its CUDA note).

**Memory.** LORRAX tiles its large arrays across the XY device grid; on a **single** device
they cannot be tiled and must fit whole. The bundled `cohsex` fixture materializes a
~**17 GiB** array (`psi_rmu_band`), so a single-GPU smoke test needs:

| Target | Minimum |
|---|---|
| Single GPU, bundled fixture | **≥ 24 GB VRAM** (RTX 4090 / A100 40 GB / L40S 48 GB) |
| CPU-only (`JAX_PLATFORMS=cpu`) | **≥ ~20 GB RAM** |
| Multi-GPU | fixture tiles across the mesh (~17 GiB ÷ #GPUs per device) |

!!! warning "A tiny GPU will OOM"
    A 16 GB or smaller card (T4, L4 24 GB is borderline, RTX 3060/4060, 8 GB laptop GPUs)
    cannot hold the bundled fixture on a single device: you will see
    `RESOURCE_EXHAUSTED: Out of memory allocating …`. This is single-device tiling, not a
    bug — scale the GPU up, add GPUs, or shrink your own input (fewer bands / k-points /
    centroids).

## Track A — one cloud GPU (smoke test / development)

Cheapest way to answer "does it build and run for me." Rent one GPU with **≥ 24 GB** and a
**driver ≥ 580**, then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # one-time
git clone <lorrax-repo-url> && cd lorrax
uv sync                                            # CUDA-13 GPU wheels, no system CUDA
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"   # expect: gpu [...]
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in   # end-to-end GW
```

Provider notes (on-demand $/hr are **July 2026** ballparks — marketplace/spot prices move
hourly; confirm live rates and host driver version before committing):

| Provider | Cheapest ≥24 GB GPU | ~$/hr | Access | Driver 580+? |
|---|---|---|---|---|
| **RunPod** (Community) | RTX 4090 24 GB | ~0.34 | container, root+SSH | host-managed — pick a CUDA-13 template/host |
| **Vast.ai** | RTX 4090 24 GB | ~0.30 | container, root | filterable per listing — *verify* |
| **Vast / RunPod** | A100 80 GB (Perlmutter-like) | ~1.2–1.4 | container | yes on newer hosts |
| **Lambda Cloud** | A100 40 GB PCIe | ~1.29 | **full VM, root** — update driver yourself | yes |
| **AWS** | g6e.xlarge (L40S 48 GB) | ~1.86 | full EC2 VM, your AMI | yes (choose DLAMI / install) |

**Recommended first iteration:** RunPod Community RTX 4090 (~$0.34/hr) or Vast RTX 4090
(~$0.30/hr) — a build + smoke test costs a couple dollars. Use `JAX_PLATFORMS=cpu` on a
big-RAM CPU instance if you only want to confirm correctness without paying for a GPU.

## Track B — multi-GPU / multi-node (production)

Single-node multi-GPU (up to 8×) works out of the box: JAX builds the XY mesh and tiles the
arrays; request the GPUs and run under `srun`/the provider's launcher. **Multi-node** and
the distributed FFI features (cuSolverMp `eigh`, sharded parallel HDF5, SLATE) additionally
require building [`liblorrax_ffi.so`](ffi-native-libs.md) and a matched MPI + parallel-HDF5
stack — untested off-NERSC (see the [support matrix](index.md#support-matrix)).

Best-fit venues for scale-up (verify current availability):

- **Nebius** — H100 ~$1.5–2.0/hr, real InfiniBand multi-node + managed SLURM; best $/GPU at scale.
- **Lambda 1-Click Clusters** — H100 SXM ~$4/GPU/hr, full-VM root, InfiniBand; closest to a Perlmutter node (watch for stock-outs).
- **AWS ParallelCluster** (p4d A100 / p5 H100 + EFA) — the most mature managed-SLURM + RDMA analog to Perlmutter; on-demand is pricey (~$4–7/GPU/hr) so lean on **spot (50–70% off) + checkpointing**.
- **GCP A3 + Cluster Toolkit** — equivalent to AWS; attractive with credits / aggressive spot discounts.

## Container track (Apptainer / Singularity)

LORRAX runs inside the NVIDIA JAX image (`nvcr.io/nvidia/jax:25.04-py3`); off-NERSC use
Apptainer/Singularity with `--nv` instead of Shifter. A ready-to-adapt recipe for a plain
cloud VM is in [`Dockerfile.cloud`](../../Dockerfile.cloud) at the repo root (pure-JAX; the
FFI stack is layered separately).

!!! note "TODO"
    A worked, copy-paste Apptainer invocation (`apptainer pull`, `--nv`, and the FFI
    `--bind <nvhpc>,<phdf5>,<slate>` contract) is not yet written; the bind-mount contract
    lives in `src/ffi/PORTING.md`. Only the Shifter path is exercised today — see
    [Perlmutter](perlmutter.md).

## See also

- [Installation overview](index.md) — the support matrix and the three tracks
- [Quickstart](../quickstart.md) — the CPU-first, no-FFI smoke test
- [FFI native libraries](ffi-native-libs.md) — the distributed stack, off-Cray recipes
- [`docs/ENVIRONMENT_COMPREHENSIVE.md`](../ENVIRONMENT_COMPREHENSIVE.md) — full JAX-config / memory / troubleshooting reference
