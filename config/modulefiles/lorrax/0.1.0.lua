-- -*- lua -*-
-- LORRAX 0.1.0 Perlmutter environment descriptor.
-- Installed from config/perlmutter/site_config.sh by install.sh.

help([[
LORRAX environment descriptor for the `lx` launcher.

Use `lx run`, `lx test`, and `lx doctor`; this module deliberately defines no
allocation or run shell functions.  It owns only the container image, native
bind mounts, MPI capabilities, and supplemental third-party Python path.
LORRAX runtime owns JAX, allocator, HDF5, compile-cache, and profiling policy.
]])

whatis("Name: LORRAX")
whatis("Version: 0.1.0")
whatis("Description: LORRAX Perlmutter container and native-library capabilities")
family("lorrax")

local image                 = "@LORRAX_IMAGE@"
local lorrax_site           = "@LORRAX_SITE@"
local shifter_modules       = "@LORRAX_SHIFTER_MODULES@"
local nvhpc_subpath         = "@LORRAX_NVHPC_SUBPATH@"
local mpich_container_dir   = "@LORRAX_MPICH_CONTAINER_DIR@"
local darshan_lib_dir       = "@LORRAX_DARSHAN_LIB_DIR@"
local default_nvhpc_host    = "@LORRAX_FFI_NVHPC_DIR_DEFAULT@"
local default_phdf5_host    = "@LORRAX_FFI_PHDF5_DIR_DEFAULT@"
local default_slate_host    = "@LORRAX_FFI_SLATE_DIR_DEFAULT@"
local default_fftw_host     = "@LORRAX_FFI_FFTW_DIR_DEFAULT@"
local default_slate_install = "@LORRAX_SLATE_INSTALL_DIR_DEFAULT@"

local this_file = myFileName()
local lorrax_root = this_file:match("(.+)/config/modulefiles/lorrax/.*$")
if lorrax_root == nil then
    lorrax_root = os.getenv("LORRAX_ROOT") or "@LORRAX_ROOT@"
end
local lorrax_src = pathJoin(lorrax_root, "src")

local function env_or(var, fallback)
    local value = os.getenv(var)
    if value and value ~= "" then return value end
    return fallback
end

local nvhpc_host = env_or("LORRAX_FFI_NVHPC_DIR", default_nvhpc_host)
local phdf5_host = env_or("LORRAX_FFI_PHDF5_DIR", default_phdf5_host)
local slate_host = env_or("LORRAX_FFI_SLATE_DIR", default_slate_host)
local fftw_host = env_or("LORRAX_FFI_FFTW_DIR", default_fftw_host)
local slate_install_host = env_or(
    "LORRAX_SLATE_INSTALL_DIR", default_slate_install)

-- The selected checkout and the supplemental third-party packages are the
-- only Python entries supplied by the site layer. First-party services are
-- selected and attested by LORRAX startup.
local pypath = lorrax_src
if lorrax_site ~= "" then
    pypath = pypath .. ":" .. lorrax_site
end

-- Keep the vendor runtime as one ordered capability description. Never add
-- a second MPI implementation or another copy of a versioned vendor SONAME.
local ldlib_parts = {
    slate_install_host .. "/lib64",
    "/lorrax_slate/lib",
    "/lorrax_phdf5/lib",
    "/lorrax_nvhpc/" .. nvhpc_subpath,
    "/lorrax_fftw/lib",
    mpich_container_dir,
    mpich_container_dir .. "/dep",
}
if darshan_lib_dir ~= "" then
    table.insert(ldlib_parts, darshan_lib_dir)
end
local container_ldlib = table.concat(ldlib_parts, ":")

local shifter_env_parts = {
    "--env=PYTHONPATH=" .. pypath,
    "--env=LD_LIBRARY_PATH=" .. container_ldlib,
    "--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0",
    "--env=MPICH_GPU_SUPPORT_ENABLED=1",
    "--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include",
    "--env=LORRAX_MPICH_LIB_DIR=" .. mpich_container_dir,
}

local shifter_args = table.concat({
    "--image=" .. image,
    "--module=" .. shifter_modules,
    "--volume=" .. nvhpc_host .. ":/lorrax_nvhpc",
    "--volume=" .. phdf5_host .. ":/lorrax_phdf5",
    "--volume=" .. slate_host .. ":/lorrax_slate",
    "--volume=" .. fftw_host .. ":/lorrax_fftw",
    table.concat(shifter_env_parts, " "),
}, " ")

setenv("LORRAX_ROOT", lorrax_root)
setenv("LORRAX_SRC", lorrax_src)
setenv("LORRAX_SITE", lorrax_site)
setenv("LORRAX_IMAGE", image)
setenv("LORRAX_SHIFTER", "shifter " .. shifter_args)
setenv("LORRAX_FFI_NVHPC_HOST", nvhpc_host)
setenv("LORRAX_FFI_PHDF5_HOST", phdf5_host)
setenv("LORRAX_FFI_SLATE_HOST", slate_host)
setenv("LORRAX_FFI_FFTW_HOST", fftw_host)
setenv("LORRAX_SLATE_INSTALL_DIR", slate_install_host)
