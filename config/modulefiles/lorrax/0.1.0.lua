-- -*- lua -*-
-- LORRAX 0.1.0 — Low-scaling Real-space Real-Axis eXcited state package
-- Lmod modulefile for Shifter-based GPU execution
--
-- Install:  bash config/perlmutter/install.sh
-- Usage:    module load lorrax
--           lxrun python3 -u -m gw.gw_jax -i cohsex.in
--
-- See config/README.md for full documentation.

help([[
LORRAX 0.1.0: Low-scaling Real-space Real-Axis eXcited state package
JAX-based GW with ISDF compression, running in NVIDIA Shifter container.

Commands:
  lxalloc [N] [T]   Interactive GPU allocation with container+mounts
                    pre-staged into setupRoot (N nodes, default 1, T hrs).
  lxrun <cmd>       Run <cmd> on LORRAX_NGPU ranks (default 4) inside
                    the pre-staged container.  Fast: no per-call shifter
                    namespace bring-up.
  lxshell           Single-rank pty shell inside the container — iterate
                    without paying Python+JAX cold start per run.
  lxpre <in> N      Preprocessing (centroids, dipoles, kin_ion).

Typical flow:
  lxalloc                                               # pre-stage + alloc
  lxrun python3 -u -m gw.gw_jax -i cohsex.in           # 4-GPU GW
  LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i ...   # 1-GPU GW
  lxshell     # inside: python3 -m common.slate_batched_test, etc.

Stack: Cray MPICH (--mpi=cray_shasta, --module=gpu,mpich), one GPU per
rank (select_gpu.sh pins CUDA_VISIBLE_DEVICES=$SLURM_LOCALID), JAX
distributed with `local_device_ids=[0]`.  SLATE, cuSOLVERMp, and phdf5
(cray stack) all share this single stack.

Override points (set before `module load`):
    LORRAX_FFI_NVHPC_DIR   default /pscratch/sd/$U0/$U/lorrax_nvhpc
    LORRAX_FFI_PHDF5_DIR   default /pscratch/sd/$U0/$U/lorrax_phdf5_cray/stage
    LORRAX_FFI_SLATE_DIR   default /pscratch/sd/$U0/$U/lorrax_slate_cray/stage
    LORRAX_SLATE_INSTALL_DIR  default $HOME/software/slate/install
    LORRAX_MPI_TYPE        default cray_shasta (|pmi2|pmix|none)
    JAX_COMPILATION_CACHE_DIR  default $SCRATCH/.jax_cache

NOTE: Always use lxrun/lxshell to run LORRAX code on Perlmutter. Do NOT
use bare "python" or "python3" — the module configures a Shifter
container, not the host Python. For local development without Shifter,
use "uv run" instead.
]])

whatis("Name:        LORRAX")
whatis("Version:     0.1.0")
whatis("Description: JAX-based GW with ISDF compression (Shifter container)")

-- Multiple LORRAX checkouts can coexist under different module names
-- (e.g. lorrax_A, lorrax_B, lorrax_C). The family() directive makes Lmod
-- auto-swap them: loading lorrax_B in a shell that already has lorrax_A
-- unloads A first, so LORRAX_ROOT / lxrun never point at a mixed state.
family("lorrax")

-- =========================================================================
--  Paths
-- =========================================================================
-- The modulefile locates LORRAX source relative to its own position:
--   <lorrax>/config/modulefiles/lorrax/0.1.0.lua  →  <lorrax>/src
-- If the file was copied (not symlinked), install.sh patches the fallback.

local this_file = myFileName()
local lorrax_root = this_file:match("(.+)/config/modulefiles/lorrax/.*$")

if lorrax_root == nil then
    lorrax_root = os.getenv("LORRAX_ROOT") or "@LORRAX_ROOT@"
end

local lorrax_src  = pathJoin(lorrax_root, "src")
local lorrax_site = "@LORRAX_SITE@"   -- patched by install.sh
local lorrax_deps = "@LORRAX_DEPS@"   -- patched by install.sh (extra PYTHONPATH entries)
local image       = "@LORRAX_IMAGE@"  -- patched by install.sh

-- =========================================================================
--  Environment — performance-optimal defaults for GPU workloads
-- =========================================================================

-- Lustre filesystem: HDF5 file-locking must be off.
setenv("HDF5_USE_FILE_LOCKING", "FALSE")

-- Memory policy: let XLA grow on demand from the CUDA asynchronous
-- mempool instead of pre-allocating its own BFC pool.  NCCL and
-- cuSOLVERMp draw from the same pool, so XLA and the collective
-- libraries share VRAM cleanly without a fixed split.  At
-- MEM_FRACTION=0.95 with a BFC pool, NCCL gets only ~2 GB scratch on
-- A100 and can surface as `NCCL error 1 unhandled cuda error` inside
-- cuSOLVERMp's syevd (observed 2026-04-20).
setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
setenv("XLA_PYTHON_CLIENT_ALLOCATOR",   "platform")

-- CUDA 12 async memory pool.  Replaces XLA's BFC sub-allocator with
-- cudaMallocAsync — pool grows without blocking the GPU pipeline.
setenv("TF_GPU_ALLOCATOR", "cuda_malloc_async")

-- =========================================================================
--  Shifter setup — Cray MPICH stack
-- =========================================================================
-- On NERSC, the per-srun shifter bring-up is ~5 s regardless of whether
-- `--image`/`--volume` appear on salloc or on the shifter command line
-- (the image is always pre-cached on the node via shifterimg; salloc's
-- --image flag just ensures the cache is warm).  So we keep --image /
-- --module / --volume on the shifter call — simpler + portable.
--
-- The real fast-iter win is `lxshell`: one shifter bring-up, many python
-- invocations inside the pty.  Use it for test iteration; use `lxrun`
-- for multi-rank launches (shifter's MPI needs `srun` on the outside).
--
-- Stack: Cray MPICH (--mpi=cray_shasta, --module=gpu,mpich), one GPU per
-- rank (`select_gpu.sh` pins CUDA_VISIBLE_DEVICES=$SLURM_LOCALID).  This
-- unified stack runs SLATE, cuSOLVERMp, and phdf5 (cray HDF5) side by
-- side.  Callers using `jax.distributed.initialize()` must pass
-- `local_device_ids=[0]` when world > 1 (auto-detected from the length
-- of CUDA_VISIBLE_DEVICES).

-- Build PYTHONPATH: lorrax src + site-packages + optional extra deps
local pypath = lorrax_src .. ":" .. lorrax_site
if lorrax_deps ~= "" then
    pypath = pypath .. ":" .. lorrax_deps
end

-- FFI staged-deps.  Inside the container these appear at the stable
-- mount points /lorrax_nvhpc, /lorrax_phdf5, /lorrax_slate.  Override
-- the host paths via env before `module load` if they live elsewhere.
local user_first_letter = (os.getenv("USER") or "j"):sub(1, 1)
local user = os.getenv("USER") or "jackm"
local nvhpc_host = os.getenv("LORRAX_FFI_NVHPC_DIR")
    or ("/pscratch/sd/" .. user_first_letter .. "/" .. user .. "/lorrax_nvhpc")
local phdf5_host = os.getenv("LORRAX_FFI_PHDF5_DIR")
    or ("/pscratch/sd/" .. user_first_letter .. "/" .. user .. "/lorrax_phdf5_cray/stage")
local slate_host = os.getenv("LORRAX_FFI_SLATE_DIR")
    or ("/pscratch/sd/" .. user_first_letter .. "/" .. user .. "/lorrax_slate_cray/stage")

-- Host-side SLATE install (bundled blaspp/lapackpp live alongside).
local slate_install_host = os.getenv("LORRAX_SLATE_INSTALL_DIR")
    or "/global/homes/j/jackm/software/slate/install"

-- Per-user XLA persistent compilation cache.  Amortises PTX compile
-- across JAX processes (doesn't cut CUDA backend init itself).
local jax_cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR")
    or (os.getenv("SCRATCH") or "/pscratch/sd/" .. user_first_letter ..
        "/" .. user) .. "/.jax_cache"

-- Paths to helper shell wrappers.
local select_gpu_sh = pathJoin(lorrax_src, "ffi/common/cpp/select_gpu.sh")
local in_container_sh = pathJoin(lorrax_src, "ffi/common/cpp/in_container.sh")

-- LD_LIBRARY_PATH inside the container (mpich stack):
--   slate install + cray stage (libsci + libmpi_gtl_cuda + xpmem + lustreapi)
--   phdf5 stage (libhdf5 built against cray-mpich)
--   NVHPC math_libs (libcusolverMp, libcal)
--   shifter's mpich bind-mounts (libmpi.so.12 + deps)
--   darshan (siteFs)
local container_ldlib = table.concat({
    slate_install_host .. "/lib64",
    "/lorrax_slate/lib",
    "/lorrax_phdf5/lib",
    "/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64",
    "/opt/udiImage/modules/mpich",
    "/opt/udiImage/modules/mpich/dep",
    "/global/common/software/nersc9/darshan/default/lib",
}, ":")

-- Full shifter argument string used by lxrun/lxpre/lxshell.  --module
-- and --volume are applied per-srun so each shifter instance sees the
-- Cray MPICH bind-mounts and FFI staged deps; NERSC's image cache makes
-- the --image= lookup near-free (a checksum match, not a pull).
local shifter_args = table.concat({
    "--image=" .. image,
    "--module=gpu,mpich",
    "--volume=" .. nvhpc_host .. ":/lorrax_nvhpc",
    "--volume=" .. phdf5_host .. ":/lorrax_phdf5",
    "--volume=" .. slate_host .. ":/lorrax_slate",
    "--env=PYTHONPATH=" .. pypath,
    "--env=HDF5_USE_FILE_LOCKING=FALSE",
    "--env=XLA_PYTHON_CLIENT_PREALLOCATE=false",
    "--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform",
    "--env=TF_GPU_ALLOCATOR=cuda_malloc_async",
    "--env=LD_LIBRARY_PATH=" .. container_ldlib,
    -- shifter's --module=mpich bind-mounts libmpi_gtl_cuda.so.0 built
    -- against CUDA 11, needs libcudart.so.11.0 we don't ship.  LD_PRELOAD
    -- the staged CUDA-12 copy so the loader binds it first.
    "--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0",
    -- Cray MPICH GPU-Direct RDMA.  shifter's --module=mpich explicitly
    -- unsets this per /etc/shifter/udiRoot.conf; in_container.sh
    -- re-asserts it inside, but passing it also covers one-off uses.
    "--env=MPICH_GPU_SUPPORT_ENABLED=1",
    "--env=JAX_COMPILATION_CACHE_DIR=" .. jax_cache_dir,
    "--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include",
    "--env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich",
}, " ")

setenv("LORRAX_ROOT",            lorrax_root)
setenv("LORRAX_SRC",             lorrax_src)
setenv("LORRAX_SITE",            lorrax_site)
setenv("LORRAX_IMAGE",           image)
setenv("LORRAX_SHIFTER",         "shifter " .. shifter_args)
setenv("LORRAX_FFI_NVHPC_HOST",  nvhpc_host)
setenv("LORRAX_FFI_PHDF5_HOST",  phdf5_host)
setenv("LORRAX_FFI_SLATE_HOST",  slate_host)
setenv("LORRAX_SLATE_INSTALL_DIR", slate_install_host)
setenv("JAX_COMPILATION_CACHE_DIR", jax_cache_dir)

-- =========================================================================
--  Shell functions
-- =========================================================================

-- lxalloc: get an interactive GPU allocation.  salloc exports
-- SLURM_JOBID into the caller's shell so subsequent lxrun/lxshell/lxpre
-- can find it; `bash -c "sleep 100000"` holds the allocation open on
-- the compute node.
--
--   lxalloc        → 1 node, 4 GPUs, 2 hours
--   lxalloc 4      → 4 nodes, 16 GPUs, 2 hours
--   lxalloc 1 4:00 → 1 node, 4 GPUs, 4 hours

set_shell_function("lxalloc", [[
    local nodes="${1:-1}"
    local time="${2:-02:00:00}"
    local gpus=$((nodes * 4))
    echo "Requesting ${nodes} node(s), ${gpus} GPU(s), ${time}..."
    salloc --nodes=${nodes} --qos=interactive --time=${time} \
           --constraint=gpu --gpus=${gpus} --account=m2651 \
           bash -c "sleep 100000"
]], "")

-- lxrun: run a command on `ngpu` ranks inside the Shifter container.
--   lxrun python3 -u -m gw.gw_jax -i cohsex.in       # 4 GPUs / 4 ranks
--   LORRAX_NGPU=1 lxrun python3 -u -m mymod          # single rank
--
-- `select_gpu.sh` pins CUDA_VISIBLE_DEVICES=$SLURM_LOCALID so each rank
-- sees exactly one GPU — required for SLATE/cuSOLVERMp FFI and safer
-- than the default all-GPUs-visible model.  JAX-side callers must use
-- `jax.distributed.initialize(local_device_ids=[0])` when process_count
-- > 1 (auto-detected via CUDA_VISIBLE_DEVICES).
--
-- `in_container.sh` re-asserts MPICH_GPU_SUPPORT_ENABLED=1, which the
-- shifter mpich module explicitly unsets per /etc/shifter/udiRoot.conf.
--
-- Default MPI PMI: `cray_shasta` (what Cray MPICH speaks).  Override with
-- LORRAX_MPI_TYPE=none|pmi2|pmix for non-MPI or legacy workloads.

set_shell_function("lxrun", [[
    # Lustre pre-stripe for large parallel HDF5 writes under tmp/.  The
    # phdf5 FFI paths write via MPI-IO; without an explicit stripe the
    # files inherit pscratch's default 1×1MB layout (~30 MB/s/rank cap).
    # `lfs` isn't in the container so it must run host-side.  Override
    # with LORRAX_NO_PRESTRIPE=1 or LORRAX_LUSTRE_STRIPE_{COUNT,SIZE}.
    if [ -z "${LORRAX_NO_PRESTRIPE:-}" ] && command -v lfs >/dev/null 2>&1; then
        mkdir -p "$PWD/tmp" 2>/dev/null
        lfs setstripe -c "${LORRAX_LUSTRE_STRIPE_COUNT:-16}" \
                      -S "${LORRAX_LUSTRE_STRIPE_SIZE:-4M}" \
                      "$PWD/tmp" >/dev/null 2>&1 || true
    fi
    local ngpu="${LORRAX_NGPU:-4}"
    local mpitype="${LORRAX_MPI_TYPE:-cray_shasta}"
    local mpiflag=""
    if [ "${mpitype}" != "none" ]; then
        mpiflag="--mpi=${mpitype}"
    fi
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    srun $jobflag $mpiflag --gres=gpu:${ngpu} -N 1 -n ${ngpu} \
        ]] .. select_gpu_sh .. [[ \
        shifter ]] .. shifter_args .. [[ \
        ]] .. in_container_sh .. [[ \
        "$@"
]], "")

-- lxshell: drop into an interactive pty shell inside the Shifter
-- container on a compute node.  Single rank, single GPU — useful for
-- iterating on single-process code without paying the ~5 s shifter
-- bring-up per invocation.  For multi-rank MPI runs you still want
-- lxrun (shifter's MPI integration requires `srun` on the outside,
-- per upstream Shifter docs).
--
--   lxshell                     # 1 GPU visible
--   LORRAX_NGPU=4 lxshell       # all 4 GPUs visible to one rank

set_shell_function("lxshell", [[
    local ngpu="${LORRAX_NGPU:-1}"
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    srun $jobflag --pty --gres=gpu:${ngpu} -N 1 -n 1 \
        ]] .. select_gpu_sh .. [[ \
        shifter ]] .. shifter_args .. [[ \
        bash -l
]], "")

-- lxpre: run all 3 LORRAX preprocessing steps (single-GPU each).
--   lxpre cohsex.in 640

set_shell_function("lxpre", [[
    local input="${1:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local ncentroids="${2:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local abs_input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    local run1="srun $jobflag --gres=gpu:1 -N 1 -n 1 ]] .. select_gpu_sh .. [["

    echo "=== [1/3] ISDF centroids (n=$ncentroids) ==="
    $run1 shifter ]] .. shifter_args .. [[ \
        ]] .. in_container_sh .. [[ \
        python3 -u -m centroid.kmeans_isdf "$ncentroids" --no-plot --seed 42 \
        || { echo "FAILED: centroid generation"; return 1; }

    echo "=== [2/3] Dipole matrix elements ==="
    $run1 shifter ]] .. shifter_args .. [[ \
        ]] .. in_container_sh .. [[ \
        python3 -u -m psp.get_dipole_mtxels -i "$abs_input" \
        || { echo "FAILED: dipole matrix elements"; return 1; }

    echo "=== [3/3] Kinetic + ionic Hamiltonian ==="
    $run1 shifter ]] .. shifter_args .. [[ \
        ]] .. in_container_sh .. [[ \
        python3 -u -m gw.kin_ion_io_chunked -i "$abs_input" \
        || { echo "FAILED: kin_ion"; return 1; }

    echo "=== Preprocessing complete ==="
    ls -lh centroids_frac_*.txt dipole.h5 kin_ion.h5 2>/dev/null
]], "")
