#!/usr/bin/env bash
# Runs INSIDE the Shifter container.  Re-asserts env vars that
# shifter's --module=mpich explicitly unsets (notably
# MPICH_GPU_SUPPORT_ENABLED, see /etc/shifter/udiRoot.conf line
# `module_mpich_siteEnvUnset = MPICH_GPU_SUPPORT_ENABLED`), then exec's
# its arguments.
#
# Used as the final wrapper between the shifter invocation and the
# user command in run_shifter.sh.  Required so SLATE / cuSOLVERMp /
# any other MPI-based GPU library actually uses GPU-Direct RDMA over
# Slingshot (with the libmpi_gtl_cuda LD_PRELOAD set externally),
# instead of host-staging every tile transfer.
export MPICH_GPU_SUPPORT_ENABLED=1
exec "$@"
