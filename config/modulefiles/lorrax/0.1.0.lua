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

local shifter_base = table.concat({
    "shifter",
    "--module=gpu",
    "--image=" .. image,
    "--env=PYTHONPATH=" .. pypath,
    "--env=HDF5_USE_FILE_LOCKING=FALSE",
    "--env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95",
    "--env=TF_GPU_ALLOCATOR=cuda_malloc_async",
}, " ")

setenv("LORRAX_ROOT",    lorrax_root)
setenv("LORRAX_SRC",     lorrax_src)
setenv("LORRAX_SITE",    lorrax_site)
setenv("LORRAX_IMAGE",   image)
setenv("LORRAX_SHIFTER", shifter_base)

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

set_shell_function("lxrun", [[
    local ngpu="${LORRAX_NGPU:-4}"
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    srun $jobflag --gres=gpu:${ngpu} -N 1 -n ${ngpu} \
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
