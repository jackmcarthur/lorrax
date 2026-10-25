#!/usr/bin/env bash
# CPU MPI adapter and toolchain defaults, shared by build, launch and install.
# Only this configuration belongs to the CPU adapter's recipe fingerprint;
# unrelated CUDA/container/site defaults live in site_config.sh.

# Unmodified, pinned upstream MPIwrapper against Cray MPICH. The builder
# publishes immutable, content-addressed releases and atomically advances
# current only after all ABI, dependency and one-MPI gates pass.
LORRAX_MPIWRAPPER_ROOT_DEFAULT="$HOME/software/lorrax_mpiwrapper_cray"
LORRAX_MPIWRAPPER_PREFIX_DEFAULT="$LORRAX_MPIWRAPPER_ROOT_DEFAULT/current"
LORRAX_MPIWRAPPER_COMMIT_DEFAULT="966f4231c96153a08295fc7d0bcbd65e916a73fd"
LORRAX_MPIWRAPPER_ABI_DEFAULT="2.10.0"

# Cray MPICH remains the implementation; MPIwrapper translates its ABI.
LORRAX_PM_PRGENV_DEFAULT="PrgEnv-gnu/8.6.0"
LORRAX_PM_MPICH_DEFAULT="cray-mpich/9.0.1"
LORRAX_PM_CMAKE_DEFAULT="cmake/3.30.2"
