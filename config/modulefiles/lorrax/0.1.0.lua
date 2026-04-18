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
  lxalloc [N]   Get an interactive GPU allocation (N nodes, default 1, 2 hrs).
  lxrun <cmd>   Run <cmd> inside the LORRAX Shifter container via srun.
                Defaults to 4 GPUs; override with LORRAX_NGPU=1.
  lxpre <in> N  Run all 3 preprocessing steps (centroids, dipoles, kin_ion).

Examples:
  lxalloc                                               # 1-node / 4-GPU alloc
  lxalloc 4                                             # 4-node / 16-GPU alloc
  lxrun python3 -u -m gw.gw_jax -i cohsex.in           # 4-GPU GW
  LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i ...   # 1-GPU GW
  lxpre cohsex.in 640                                   # all preprocessing

NOTE: Always use lxrun to run LORRAX code on Perlmutter. Do NOT use bare
"python" or "python3" — the module configures a Shifter container, not the
host Python. For local development without Shifter, use "uv run" instead.
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

-- Pre-allocate 95% of GPU memory into XLA's memory pool.
-- LORRAX fills VRAM with donated buffers; a large pool avoids
-- fragmentation and eliminates per-allocation cudaMalloc stalls.
setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

-- CUDA 12 async memory pool.  Replaces XLA's BFC sub-allocator with
-- cudaMallocAsync — pool grows without blocking the GPU pipeline.
setenv("TF_GPU_ALLOCATOR", "cuda_malloc_async")

-- =========================================================================
--  Shifter base command (assembled once, reused by shell functions)
-- =========================================================================

-- Build PYTHONPATH: lorrax src + site-packages + optional extra deps
local pypath = lorrax_src .. ":" .. lorrax_site
if lorrax_deps ~= "" then
    pypath = pypath .. ":" .. lorrax_deps
end

-- FFI staged-deps bind-mounts.  Mirror run_shifter.sh so the FFI path
-- (use_ffi_io=true, cusolvermp eigh, etc.) can be used from lxrun
-- without manually invoking run_shifter.sh.  Default staging locations
-- are the user's scratch dir, matching the stage_*.sh scripts under
-- src/ffi/phdf5/scripts/.  Override via env if they live elsewhere.
--
-- Inside the container the paths are stable and visible to CMake /
-- LD_LIBRARY_PATH at the /lorrax_nvhpc and /lorrax_phdf5 mount points.
local user_first_letter = (os.getenv("USER") or "j"):sub(1, 1)
local user = os.getenv("USER") or "jackm"
local nvhpc_host = os.getenv("LORRAX_FFI_NVHPC_DIR")
    or ("/pscratch/sd/" .. user_first_letter .. "/" .. user .. "/lorrax_nvhpc")
local phdf5_host = os.getenv("LORRAX_FFI_PHDF5_DIR")
    or ("/pscratch/sd/" .. user_first_letter .. "/" .. user .. "/lorrax_phdf5_openmpi/stage")
-- LD_LIBRARY_PATH inside the container: phdf5 stage first (libhdf5 +
-- SONAME shims), then NVHPC math_libs (libcusolverMp, libcal), then
-- container's OpenMPI runtime (libmpi.so.40).
local container_ldlib = table.concat({
    "/lorrax_phdf5/lib",
    "/lorrax_nvhpc/25.5_cuda12.9/math_libs/12.9/lib64",
    "/opt/hpcx/ompi/lib",
}, ":")

local shifter_base = table.concat({
    "shifter",
    "--module=gpu",
    "--image=" .. image,
    "--volume=" .. nvhpc_host .. ":/lorrax_nvhpc",
    "--volume=" .. phdf5_host .. ":/lorrax_phdf5",
    "--env=PYTHONPATH=" .. pypath,
    "--env=HDF5_USE_FILE_LOCKING=FALSE",
    "--env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95",
    "--env=TF_GPU_ALLOCATOR=cuda_malloc_async",
    "--env=LD_LIBRARY_PATH=" .. container_ldlib,
}, " ")

setenv("LORRAX_ROOT",    lorrax_root)
setenv("LORRAX_SRC",     lorrax_src)
setenv("LORRAX_SITE",    lorrax_site)
setenv("LORRAX_IMAGE",   image)
setenv("LORRAX_SHIFTER", shifter_base)
setenv("LORRAX_FFI_NVHPC_HOST", nvhpc_host)
setenv("LORRAX_FFI_PHDF5_HOST", phdf5_host)

-- =========================================================================
--  Shell functions
-- =========================================================================

-- lxalloc: get an interactive GPU allocation.
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

-- lxrun: run any command inside the LORRAX Shifter container.
--   lxrun python3 -u -m gw.gw_jax -i cohsex.in
--   LORRAX_NGPU=1 lxrun python3 -u -m centroid.kmeans_isdf 640

-- ``--mpi=pmix`` is required for the container's OpenMPI (HPC-X) to
-- bootstrap collectives, which the FFI path uses.  It has been observed
-- to cause hangs in non-FFI workloads after V_q on Perlmutter
-- (subsequent collectives wait on PMIx server state that JAX never
-- touches).  Default: OFF.  For FFI runs, set LORRAX_MPI_TYPE=pmix
-- before lxrun:
--     LORRAX_MPI_TYPE=pmix LORRAX_NGPU=4 lxrun python3 -u -m gw.gw_jax ...

set_shell_function("lxrun", [[
    local ngpu="${LORRAX_NGPU:-4}"
    local mpitype="${LORRAX_MPI_TYPE:-}"
    local mpiflag=""
    if [ -n "${mpitype}" ] && [ "${mpitype}" != "none" ]; then
        mpiflag="--mpi=${mpitype}"
    fi
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    srun $jobflag $mpiflag --gres=gpu:${ngpu} -N 1 -n ${ngpu} \
        ]] .. shifter_base .. [[ \
        "$@"
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
    local run1="srun $jobflag --gres=gpu:1 -N 1 -n 1"

    echo "=== [1/3] ISDF centroids (n=$ncentroids) ==="
    $run1 ]] .. shifter_base .. [[ \
        python3 -u -m centroid.kmeans_isdf "$ncentroids" --no-plot --seed 42 \
        || { echo "FAILED: centroid generation"; return 1; }

    echo "=== [2/3] Dipole matrix elements ==="
    $run1 ]] .. shifter_base .. [[ \
        python3 -u -m psp.get_dipole_mtxels -i "$abs_input" \
        || { echo "FAILED: dipole matrix elements"; return 1; }

    echo "=== [3/3] Kinetic + ionic Hamiltonian ==="
    $run1 ]] .. shifter_base .. [[ \
        python3 -u -m gw.kin_ion_io_chunked -i "$abs_input" \
        || { echo "FAILED: kin_ion"; return 1; }

    echo "=== Preprocessing complete ==="
    ls -lh centroids_frac_*.txt dipole.h5 kin_ion.h5 2>/dev/null
]], "")
