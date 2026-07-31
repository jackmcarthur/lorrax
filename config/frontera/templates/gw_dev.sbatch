#!/bin/bash
# ============================================================================
# gw_dev.sbatch — CANONICAL Frontera CLX multi-node CPU GW launch template.
#
# This is the CERTIFIED launch block, vendored (env audit): it reconciles the
# last certified harness (/scratch2 mos2_4x4_test gw800_merged.sbatch, merged
# tree @ 9f20cb6) with docs/dev/env_vars.md §5 and the impl=mpi recipe in
# docs/dev/mpi_collectives.md.  Values are measured, not invented; do not
# edit the env block without reading those two docs.
#
# What it demonstrates:
#   * srun --mpi=pmi2 + Intel MPI bootstrapped through the staged host PMI2
#     lib ($WORK/host_pmi — config/frontera/stage_host_pmi.sh)
#   * apptainer with the /opt/intel bind and the /hostlibs RDMA staging
#     binds; NEVER --bind anything under /dev (AS.1b: a nosuid,nodev shadow
#     copy breaks every uverbs device open and mlx degrades to tcp)
#   * transport hygiene via config/frontera/mpi_transport_env.sh
#     (fabrics/provider/UCX/PMI; provider banner is mandatory telemetry)
#   * CPU collectives on impl=mpi (gloo reduce-scatter silently corrupts
#     ~5% of executions) with the patched MPIwrapper + the overlay
#     sitecustomize teardown fix
#   * node-local runtime staging via config/frontera/stage_runtime.sh
#
# Edit the #SBATCH block and the "deck" variables; leave the env block alone.
# ============================================================================
#SBATCH -J gw_dev
#SBATCH -p development
#SBATCH -A PHY25006
#SBATCH -N 8
#SBATCH --ntasks-per-node=2
#SBATCH -n 16
#SBATCH -t 01:00:00
#SBATCH -o gw_dev.%j.out
#SBATCH -e gw_dev.%j.out

set -u
set +u; module load tacc-apptainer 2>/dev/null; set -u

# --- deck / paths (user-editable; :? = refuse loudly, never guess) ----------
# All variables the container-side runner reads are EXPORTED — the inner
# heredoc is quoted (no splices) and resolves them from the environment that
# srun + apptainer inherit.
: "${LORRAX_ROOT:?set LORRAX_ROOT to the repo checkout (contains config/frontera)}"
: "${LORRAX_RUN_DIR:?set LORRAX_RUN_DIR to the run directory containing the input deck}"
: "${LORRAX_INPUT:=gw.in}"
: "${LORRAX_SIF:=$SCRATCH/lorrax_setup/py312.sif}"
export LORRAX_ROOT LORRAX_RUN_DIR LORRAX_INPUT
[ -f "$LORRAX_SIF" ] || { echo "REFUSING: no container image at $LORRAX_SIF"; exit 2; }

# Optional node-local runtime bundle (docs/dev/cold_start.md §3.2).  Unset =>
# stage_runtime.sh announces the fallback to the Lustre venv and runs anyway.
#   build once: config/frontera/build_cpu_runtime_bundle.sh
export LORRAX_BUNDLE="${LORRAX_BUNDLE:-$SCRATCH/lorrax_bundle/lorrax_cpu_bundle.tar}"

# --- artifacts outside the repo (machine facts, $WORK-relative — no
# user-literal paths; see env_vars.md §5 for why each is not auto-defaulted)
export LORRAX_MPIWRAPPER_SO="${LORRAX_MPIWRAPPER_SO:-$WORK/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so}"
[ -f "$LORRAX_MPIWRAPPER_SO" ] || {
  echo "REFUSING: no MPIwrapper at $LORRAX_MPIWRAPPER_SO"
  echo "  build it (login node): config/frontera/build_mpiwrapper.sh"
  echo "  An UNPATCHED wrapper grants FUNNELED: ~29% multi-node crash rate (AS.4b)."
  exit 2; }
# staged host PMI2 lib — consumed by mpi_transport_env.sh as
# I_MPI_PMI_LIBRARY=$WORK/host_pmi/libpmi2.so.0 (LORRAX_PMI2_LIB overrides);
# stage it once with config/frontera/stage_host_pmi.sh.

export LORRAX_SLATE_HOST_LIB=${LORRAX_SLATE_HOST_LIB:-$WORK/slate_builds/cpu/install/lib64}
export LORRAX_IMPI_ROOT=${LORRAX_IMPI_ROOT:-/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64}
export LORRAX_HDF5_ROOT=${LORRAX_HDF5_ROOT:-/home1/apps/intel19/impi19_0/phdf5/1.14.6}
export LORRAX_MKL_LIB=${LORRAX_MKL_LIB:-/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin}
export LORRAX_ICC_RUNTIME=${LORRAX_ICC_RUNTIME:-/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin}

# per-LAUNCH coordinator port under shared holders (AP.6; control plane on
# em1 is fine — only the MPI data plane needs the fabric)
_coord_host=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export JAX_COORDINATOR_ADDRESS="$_coord_host:$((13000 + SLURM_JOB_ID % 2000))"

echo "=== gw_dev  $(date)  coord=$JAX_COORDINATOR_ADDRESS  src@$(git -C "$LORRAX_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?') ==="

# HL = RDMA/UCX userspace staging binds (AS.1): without them the mlx provider
# cannot load in-container and LORRAX_MPI_PROVIDER=auto degrades to tcp
# (announced by the I_MPI_DEBUG banner).  NEVER bind anything under /dev
# (AS.1b) — apptainer's default mount already exposes the host /dev, uverbs
# included; a --bind /dev[/...] shadows it nosuid,nodev and breaks it.
HL=/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp,$HL

INNER=$LORRAX_RUN_DIR/gw_dev_inner.$SLURM_JOB_ID.sh
cat > "$INNER" <<'EOS'
# container-side runner (env inherited from the sbatch shell through srun +
# apptainer; everything below reads, never re-hardcodes, the outer variables)
set -u
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=""
export HDF5_USE_FILE_LOCKING=FALSE
# BLAS-side thread teams (MKL pzheevd, SLATE OpenMP, scipy_openblas).  The
# XLA:CPU pool does NOT obey OMP — taskset affinity is its size (ADVICE §10).
export OMP_NUM_THREADS=28 OPENBLAS_NUM_THREADS=28 MKL_NUM_THREADS=28
# compile cache ON (AT/AU verdict; AH fixed the P>1 deadlock — warm
# wall-neutral, outputs byte-identical).  Export ISDF_JAX_CACHE_DIR="" to
# opt out (announced).
export ISDF_JAX_CACHE_DIR=${ISDF_JAX_CACHE_DIR-$SCRATCH/lorrax_jax_cache}

# --- transport hygiene (fabrics, provider case-block, UCX setdefaults,
# I_MPI_PMI_LIBRARY=$WORK/host_pmi/libpmi2.so.0, I_MPI_DEBUG=4 banner) ------
. "$LORRAX_ROOT/config/frontera/mpi_transport_env.sh"

# --- CPU collectives: impl=mpi (docs/dev/mpi_collectives.md) ---------------
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi
export MPITRAMPOLINE_LIB="$LORRAX_MPIWRAPPER_SO"
export LORRAX_MPI_FINALIZE_FIX=skip_atexit   # overlay sitecustomize; without
                                             # it every run exits rc=1 AFTER
                                             # succeeding (double finalize)
# LORRAX_MPI_FORCE_THREAD_MAIN is deliberately NOT set: superseded by the
# in-repo warm_mesh_cliques() (mpi_collectives.md STATUS).  Setting it would
# only mask a missing warm-up call site.

# --- host library search path (host FFI .so deps: SLATE, MKL, phdf5,
# Intel MPI runtime + bundled libfabric, Intel compiler runtime) ------------
export LD_LIBRARY_PATH="$LORRAX_SLATE_HOST_LIB:$LORRAX_MKL_LIB:$LORRAX_HDF5_ROOT/lib:$LORRAX_IMPI_ROOT/lib/release:$LORRAX_IMPI_ROOT/lib:$LORRAX_IMPI_ROOT/libfabric/lib:$LORRAX_ICC_RUNTIME:${LD_LIBRARY_PATH:-}"

# host RDMA/UCX userspace for mlx, staged as symlinks APPENDED to
# LD_LIBRARY_PATH (a bare /hostlibs on the path shadows container glibc):
if [ -d /hostlibs ]; then
  RD=/tmp/rdma_stage.$$; mkdir -p "$RD"
  for l in libibverbs.so.1 librdmacm.so.1 libnl-3.so.200 libnl-route-3.so.200 \
           libucp.so.0 libucs.so.0 libuct.so.0 libucm.so.0 libnuma.so.1; do
    [ -e /hostlibs/$l ] && ln -sf /hostlibs/$l "$RD/$l"
  done
  ln -sfn /hostlibs/ucx "$RD/ucx"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RD:/hostlibs/libibverbs
fi

# --- node-local runtime staging (docs/dev/cold_start.md; loud fallback) ----
. "$LORRAX_ROOT/config/frontera/stage_runtime.sh"
export PYTHONPATH=$LORRAX_OVERLAY_DIR:$LORRAX_SRC_DIR

cd "$LORRAX_RUN_DIR"
S=$SLURM_LOCALID
# taskset = the XLA:CPU THREAD CAP (28 threads/rank), the certified layout.
# NOTE it is NOT socket-local: Frontera CLX enumerates NUMA node0=even cores
# and node1=odd, so the contiguous window 0-27 straddles BOTH sockets.
# Socket-LOCAL pinning (even/odd core lists) is a MEASURED TRADE — 1.164x on
# a production sigma cell but 10-22% SLOWER on bandwidth-bound FFTs
# (HANDOFF_2026-07-29 evidence audit) — and is deliberately NOT applied.
exec taskset -c $((S*28))-$((S*28+27)) \
    "$LORRAX_PY" -u -m gw.gw_jax -i "$LORRAX_INPUT"
EOS
chmod +x "$INNER"

srun --mpi=pmi2 -N "$SLURM_NNODES" --ntasks-per-node=2 -n "$SLURM_NTASKS" \
    apptainer exec --bind "$BIND" "$LORRAX_SIF" bash "$INNER" \
    > "$LORRAX_RUN_DIR/gw_dev.$SLURM_JOB_ID.log" 2>&1
RC=$?
echo "[gw rc=$RC] $(date)"
rm -f "$INNER"

echo "===================== KEY BANNERS ====================="
# "libfabric provider:" (must be mlx unless you asked for the tcp hatch),
# the collectives implementation line, staging, and the physics tail.
grep -aE "libfabric provider|cpu_collectives|collectives|\[stage\]|\[mpi_transport_env\]|SlabIO|restart_write|Backend:|device mesh|TIER resolved|WARN|Error|Traceback|gap|eqp0" \
    "$LORRAX_RUN_DIR/gw_dev.$SLURM_JOB_ID.log" | head -60
echo "===================== STAGE TABLE ====================="
sed -n '/Timing summary/,$p' "$LORRAX_RUN_DIR/gw_dev.$SLURM_JOB_ID.log" | head -80
echo "=== done $(date) ==="
exit $RC
