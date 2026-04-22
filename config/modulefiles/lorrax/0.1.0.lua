-- -*- lua -*-
-- LORRAX 0.1.0 — Low-scaling Real-space Real-Axis eXcited state package
-- Lmod modulefile for Shifter-based GPU execution.
--
-- Install:  bash config/<cluster>/install.sh
-- Usage:    module load lorrax
--           lxrun python3 -u -m gw.gw_jax -i cohsex.in
--
-- Template placeholders `@FOO@` are patched at install time from
-- config/<cluster>/site_config.sh.  See config/README.md.

help([[
LORRAX 0.1.0 — JAX-based GW with ISDF compression, Shifter container.

Commands
  lxalloc [N] [T]   Interactive GPU allocation (N nodes, default 1, T hrs).
  lxrun <cmd>       Run <cmd> on LORRAX_NGPU ranks (default 4) inside the
                    Shifter container.  Cray MPICH + 1 GPU per rank.
  lxshell           Single-rank pty shell inside the container; iterate
                    without paying ~5 s shifter bring-up per python call.
  lxpre <in> N      Preprocessing (centroids, dipoles, kin_ion).

Typical flow
  lxalloc
  lxrun python3 -u -m gw.gw_jax -i cohsex.in
  LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in
  lxshell       # then `python3 -m common.slate_batched_test` etc.

Stack
  --mpi=cray_shasta, --module=gpu,mpich, one GPU per rank via
  select_gpu.sh (CUDA_VISIBLE_DEVICES=$SLURM_LOCALID), LD_PRELOAD the
  CUDA-12 libmpi_gtl_cuda, MPICH_GPU_SUPPORT_ENABLED=1.  SLATE,
  cuSOLVERMp, and Cray-HDF5 phdf5 all share this stack.

Overrides (set before `module load`)
  LORRAX_FFI_NVHPC_DIR      default @LORRAX_FFI_NVHPC_DIR_DEFAULT@
  LORRAX_FFI_PHDF5_DIR      default @LORRAX_FFI_PHDF5_DIR_DEFAULT@
  LORRAX_FFI_SLATE_DIR      default @LORRAX_FFI_SLATE_DIR_DEFAULT@
  LORRAX_SLATE_INSTALL_DIR  default @LORRAX_SLATE_INSTALL_DIR_DEFAULT@
  LORRAX_MPI_TYPE           default @LORRAX_MPI_TYPE_DEFAULT@ (|none|pmi2|pmix)
  LORRAX_NGPU               default 4 for lxrun, 1 for lxshell
  JAX_COMPILATION_CACHE_DIR default $SCRATCH/.jax_cache

Do NOT use bare `python` / `python3` — the module configures a Shifter
container, not the host Python.  For non-Shifter local dev, use `uv run`.
]])

whatis("Name:        LORRAX")
whatis("Version:     0.1.0")
whatis("Description: JAX-based GW with ISDF compression (Shifter container)")

-- Multiple LORRAX checkouts coexist under distinct module names
-- (lorrax_A / _B / _C).  family() makes Lmod auto-swap — loading
-- lorrax_B unloads lorrax_A in the same shell, so LORRAX_ROOT and the
-- shell functions never point at a mixed state.
family("lorrax")

-- =========================================================================
--  Site constants (patched at install time; see site_config.sh)
-- =========================================================================
local image                 = "@LORRAX_IMAGE@"
local lorrax_site           = "@LORRAX_SITE@"
local lorrax_deps           = "@LORRAX_DEPS@"

-- SLURM defaults for lxalloc.
local slurm_account         = "@LORRAX_SLURM_ACCOUNT@"
local slurm_qos             = "@LORRAX_SLURM_QOS@"
local slurm_constraint      = "@LORRAX_SLURM_CONSTRAINT@"
local gpus_per_node         = tonumber("@LORRAX_GPUS_PER_NODE@")

-- Shifter + MPI stack.
local shifter_modules       = "@LORRAX_SHIFTER_MODULES@"      -- "gpu,mpich"
local default_mpi_type      = "@LORRAX_MPI_TYPE_DEFAULT@"     -- "cray_shasta"
local nvhpc_subpath         = "@LORRAX_NVHPC_SUBPATH@"
local mpich_container_dir   = "@LORRAX_MPICH_CONTAINER_DIR@"  -- "/opt/udiImage/modules/mpich"
local darshan_lib_dir       = "@LORRAX_DARSHAN_LIB_DIR@"      -- may be ""

-- Stage-dir defaults (NERSC $SCRATCH-based; overridable per-user).
local default_nvhpc_host    = "@LORRAX_FFI_NVHPC_DIR_DEFAULT@"
local default_phdf5_host    = "@LORRAX_FFI_PHDF5_DIR_DEFAULT@"
local default_slate_host    = "@LORRAX_FFI_SLATE_DIR_DEFAULT@"
local default_slate_install = "@LORRAX_SLATE_INSTALL_DIR_DEFAULT@"

-- =========================================================================
--  LORRAX source location (derived from this modulefile's own path)
-- =========================================================================
-- Works when the modulefile is symlinked into $MODULEFILE_DIR/ (common for
-- A/B/C multi-checkout setups).  If the file was copied instead, install.sh
-- patches @LORRAX_ROOT@ as the fallback.

local this_file   = myFileName()
local lorrax_root = this_file:match("(.+)/config/modulefiles/lorrax/.*$")
if lorrax_root == nil then
    lorrax_root = os.getenv("LORRAX_ROOT") or "@LORRAX_ROOT@"
end
local lorrax_src  = pathJoin(lorrax_root, "src")

-- =========================================================================
--  Per-user paths (runtime-overridable via env vars)
-- =========================================================================

local function env_or(var, fallback)
    local v = os.getenv(var)
    if v and v ~= "" then return v end
    return fallback
end

local nvhpc_host         = env_or("LORRAX_FFI_NVHPC_DIR",     default_nvhpc_host)
local phdf5_host         = env_or("LORRAX_FFI_PHDF5_DIR",     default_phdf5_host)
local slate_host         = env_or("LORRAX_FFI_SLATE_DIR",     default_slate_host)
local slate_install_host = env_or("LORRAX_SLATE_INSTALL_DIR", default_slate_install)

-- XLA persistent compile cache.  Amortises PTX JIT across JAX processes;
-- does NOT cut the CUDA backend init itself.
local jax_cache_dir = env_or("JAX_COMPILATION_CACHE_DIR",
    pathJoin(env_or("SCRATCH", os.getenv("HOME")), ".jax_cache"))

-- =========================================================================
--  Performance env (inside and outside container)
-- =========================================================================

-- Lustre: HDF5 file-locking must be off.
setenv("HDF5_USE_FILE_LOCKING", "FALSE")

-- Allocator choice.
--
-- Default (LORRAX_XLA_PREALLOCATE unset): grow on demand from the CUDA
-- async mempool.  NCCL / cuSOLVERMp / SLATE then share VRAM with XLA
-- cleanly.  At MEM_FRACTION=0.95 with BFC, NCCL starves and surfaces
-- as `NCCL error 1 unhandled cuda error` inside cusolverMpSyevd.
--
-- Opt-in (LORRAX_XLA_PREALLOCATE=1 in shell BEFORE `module load`): BFC
-- preallocation.  Faster for workloads that don't hit cuSOLVERMp (e.g.
-- MoS2 3×3 COHSEX at a 28 GB budget), at the cost of NCCL starvation
-- risk on workloads that do.  Override the fraction with
-- LORRAX_XLA_MEM_FRACTION (default 0.75, JAX default).
local _prealloc_flag = os.getenv("LORRAX_XLA_PREALLOCATE") or ""
local use_preallocate = (_prealloc_flag == "1" or _prealloc_flag == "true")

local allocator_env
if use_preallocate then
    allocator_env = {
        {"XLA_PYTHON_CLIENT_PREALLOCATE",  "true"},
        {"XLA_PYTHON_CLIENT_ALLOCATOR",    "bfc"},
        {"XLA_PYTHON_CLIENT_MEM_FRACTION", env_or("LORRAX_XLA_MEM_FRACTION", "0.75")},
    }
    -- TF_GPU_ALLOCATOR overrides the allocator kind in some jaxlib
    -- builds; force-clear it in preallocate mode.  Shifter/srun reject
    -- empty `--env=KEY=` values, so we unsetenv rather than pass through.
    unsetenv("TF_GPU_ALLOCATOR")
else
    allocator_env = {
        {"XLA_PYTHON_CLIENT_PREALLOCATE",  "false"},
        {"XLA_PYTHON_CLIENT_ALLOCATOR",    "platform"},
        {"TF_GPU_ALLOCATOR",               "cuda_malloc_async"},
    }
end

for _, kv in ipairs(allocator_env) do
    setenv(kv[1], kv[2])
end

-- =========================================================================
--  PYTHONPATH + container LD_LIBRARY_PATH
-- =========================================================================

-- PYTHONPATH: lorrax src + supplemental site-packages + optional deps.
local pypath = lorrax_src .. ":" .. lorrax_site
if lorrax_deps ~= "" then
    pypath = pypath .. ":" .. lorrax_deps
end

-- Container LD_LIBRARY_PATH — order matters:
--   1. SLATE install (slate/blaspp/lapackpp built against cray libs)
--   2. /lorrax_slate : Cray libsci + libmpi_gtl_cuda + xpmem + lustreapi
--   3. /lorrax_phdf5 : Cray HDF5 (libmpi_gnu_*.so.12)
--   4. /lorrax_nvhpc : libcusolverMp + libcal
--   5. Shifter's mpich bind-mount (libmpi.so.12 + PMI + libfabric deps)
--   6. (optional) Darshan I/O profiling
local ldlib_parts = {
    slate_install_host .. "/lib64",
    "/lorrax_slate/lib",
    "/lorrax_phdf5/lib",
    "/lorrax_nvhpc/" .. nvhpc_subpath,
    mpich_container_dir,
    mpich_container_dir .. "/dep",
}
if darshan_lib_dir ~= "" then
    table.insert(ldlib_parts, darshan_lib_dir)
end
local container_ldlib = table.concat(ldlib_parts, ":")

-- =========================================================================
--  Shifter invocation
-- =========================================================================
-- We pass --image/--module/--volume per-shifter-call (NERSC's image cache
-- makes the --image lookup near-free; salloc-time pre-stage doesn't
-- measurably reduce the ~5 s shifter bring-up).  The real fast-iter win
-- is `lxshell` (one shifter bring-up, many python invocations).

local shifter_env_parts = {
    "--env=PYTHONPATH=" .. pypath,
    "--env=HDF5_USE_FILE_LOCKING=FALSE",
    "--env=LD_LIBRARY_PATH=" .. container_ldlib,
    -- Shifter's mpich module ships libmpi_gtl_cuda.so.0 built against
    -- CUDA 11 (needs libcudart.so.11 we don't have).  LD_PRELOAD the
    -- staged CUDA-12 copy so the loader binds it first.
    "--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0",
    -- Cray MPICH GPU-Direct RDMA.  Shifter's --module=mpich explicitly
    -- unsets this per /etc/shifter/udiRoot.conf; in_container.sh
    -- re-asserts it inside, but passing via --env also covers one-off
    -- invocations that don't use in_container.sh (e.g. lxshell).
    "--env=MPICH_GPU_SUPPORT_ENABLED=1",
    "--env=JAX_COMPILATION_CACHE_DIR=" .. jax_cache_dir,
    "--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include",
    "--env=LORRAX_MPICH_LIB_DIR=" .. mpich_container_dir,
}

-- Forward the allocator trio so the container sees the same choice the
-- login side set above.  Empty values clear an inherited inside-container
-- setting (needed when switching TF_GPU_ALLOCATOR off in preallocate mode).
for _, kv in ipairs(allocator_env) do
    table.insert(shifter_env_parts, "--env=" .. kv[1] .. "=" .. kv[2])
end

local shifter_args = table.concat({
    "--image=" .. image,
    "--module=" .. shifter_modules,
    "--volume=" .. nvhpc_host .. ":/lorrax_nvhpc",
    "--volume=" .. phdf5_host .. ":/lorrax_phdf5",
    "--volume=" .. slate_host .. ":/lorrax_slate",
    table.concat(shifter_env_parts, " "),
}, " ")

-- =========================================================================
--  Exports
-- =========================================================================

setenv("LORRAX_ROOT",              lorrax_root)
setenv("LORRAX_SRC",               lorrax_src)
setenv("LORRAX_SITE",              lorrax_site)
setenv("LORRAX_IMAGE",             image)
setenv("LORRAX_SHIFTER",           "shifter " .. shifter_args)
setenv("LORRAX_FFI_NVHPC_HOST",    nvhpc_host)
setenv("LORRAX_FFI_PHDF5_HOST",    phdf5_host)
setenv("LORRAX_FFI_SLATE_HOST",    slate_host)
setenv("LORRAX_SLATE_INSTALL_DIR", slate_install_host)
setenv("JAX_COMPILATION_CACHE_DIR", jax_cache_dir)

-- =========================================================================
--  Shell functions
-- =========================================================================

-- Paths to the helper wrappers (injected into every shell function body).
local select_gpu_sh   = pathJoin(lorrax_src, "ffi/common/cpp/select_gpu.sh")
local in_container_sh = pathJoin(lorrax_src, "ffi/common/cpp/in_container.sh")

-- -------------------------------------------------------------------------
-- lxalloc: interactive GPU allocation.
--   lxalloc          → 1 node, {gpus_per_node} GPUs, 2 hours
--   lxalloc 4        → 4 nodes
--   lxalloc 1 4:00   → 1 node, 4 hours
--
-- `salloc ... bash -c "sleep 100000"` holds the allocation open on the
-- compute node while salloc exports SLURM_JOBID into the caller's shell
-- so lxrun / lxshell / lxpre can pick it up.

set_shell_function("lxalloc", [[
    local nodes="${1:-1}"
    local time="${2:-02:00:00}"
    local gpus=$((nodes * ]] .. gpus_per_node .. [[))
    echo "Requesting ${nodes} node(s), ${gpus} GPU(s), ${time}..."
    salloc --nodes=${nodes} --qos=]] .. slurm_qos .. [[ --time=${time} \
           --constraint=]] .. slurm_constraint .. [[ --gpus=${gpus} \
           --account=]] .. slurm_account .. [[ \
           bash -c "sleep 100000"
]], "")

-- -------------------------------------------------------------------------
-- lxrun: run <cmd> on `LORRAX_NGPU` ranks (default 4) inside the container.
--   lxrun python3 -u -m gw.gw_jax -i cohsex.in
--   LORRAX_NGPU=1 lxrun python3 -u -m common.slate_batched_test ...
--
-- Structure of the generated srun line:
--   srun --jobid=$JID --mpi=cray_shasta --gres=gpu:N -N 1 -n N \
--        select_gpu.sh          # CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
--        shifter ...            # image + mounts + env + LD_PRELOAD
--        in_container.sh        # re-assert MPICH_GPU_SUPPORT_ENABLED=1
--        <user cmd>

set_shell_function("lxrun", [[
    # Lustre pre-stripe for large parallel-HDF5 writes under $PWD/tmp.
    # Without this, files inherit pscratch's default 1×1MB layout (~30
    # MB/s/rank cap).  `lfs` isn't in the container so it must run
    # host-side.  Override with LORRAX_NO_PRESTRIPE=1 or
    # LORRAX_LUSTRE_STRIPE_{COUNT,SIZE}.
    if [ -z "${LORRAX_NO_PRESTRIPE:-}" ] && command -v lfs >/dev/null 2>&1; then
        mkdir -p "$PWD/tmp" 2>/dev/null
        lfs setstripe -c "${LORRAX_LUSTRE_STRIPE_COUNT:-16}" \
                      -S "${LORRAX_LUSTRE_STRIPE_SIZE:-4M}" \
                      "$PWD/tmp" >/dev/null 2>&1 || true
    fi
    local ngpu="${LORRAX_NGPU:-]] .. gpus_per_node .. [[}"
    local mpitype="${LORRAX_MPI_TYPE:-]] .. default_mpi_type .. [[}"
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

-- -------------------------------------------------------------------------
-- lxshell: interactive pty shell inside the container on a compute node.
-- Single rank, single GPU by default — the point is to avoid paying the
-- ~5 s shifter bring-up per python invocation during iteration.  For
-- multi-rank MPI you still want lxrun (upstream Shifter docs: MPI
-- integration requires srun on the outside).
--
--   lxshell                   # 1 GPU visible
--   LORRAX_NGPU=4 lxshell     # all GPUs visible to a single rank

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

-- -------------------------------------------------------------------------
-- lxpre: run the 3 LORRAX preprocessing steps (single-GPU each).
--   lxpre cohsex.in 640

set_shell_function("lxpre", [[
    local input="${1:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local ncentroids="${2:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local abs_input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
    local jobflag=""
    if [ -n "${SLURM_JOBID:-}" ] && [ -z "${SLURM_STEP_ID:-}" ]; then
        jobflag="--jobid=$SLURM_JOBID"
    fi
    local step="srun $jobflag --gres=gpu:1 -N 1 -n 1 ]] .. select_gpu_sh .. [[ \
                shifter ]] .. shifter_args .. [[ ]] .. in_container_sh .. [["

    echo "=== [1/3] ISDF centroids (n=$ncentroids) ==="
    $step python3 -u -m centroid.kmeans_isdf "$ncentroids" --no-plot --seed 42 \
        || { echo "FAILED: centroid generation"; return 1; }

    echo "=== [2/3] Dipole matrix elements ==="
    $step python3 -u -m psp.get_dipole_mtxels -i "$abs_input" \
        || { echo "FAILED: dipole matrix elements"; return 1; }

    echo "=== [3/3] Kinetic + ionic Hamiltonian ==="
    $step python3 -u -m gw.kin_ion_io_chunked -i "$abs_input" \
        || { echo "FAILED: kin_ion"; return 1; }

    echo "=== Preprocessing complete ==="
    ls -lh centroids_frac_*.txt dipole.h5 kin_ion.h5 2>/dev/null
]], "")
