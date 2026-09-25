#!/usr/bin/env bash
# ffi_env.sh — back-compat shim: sources mpi_transport_env.sh (Intel-MPI
# transport hygiene).  New jobs source mpi_transport_env.sh directly.
_lorrax_ffi_env_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
. "$_lorrax_ffi_env_dir/mpi_transport_env.sh"
unset _lorrax_ffi_env_dir
