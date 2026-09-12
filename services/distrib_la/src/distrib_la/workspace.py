"""Query-only dense workspace accounting; no matrix/workspace allocation."""
from functools import lru_cache
import ctypes
import operator

import numpy as np


def _shapes(shapes):
    try:
        result = tuple(tuple(operator.index(n) for n in shape) for shape in shapes)
    except (TypeError, ValueError) as ex:
        raise ValueError('shapes must be a tuple of operand shape tuples') from ex
    if not result or any(not s or min(s) < 1 for s in result):
        raise ValueError('workspace shapes must have positive extents')
    return result


@lru_cache(maxsize=128)
def _vendor_query(ctx, op, sizes, dtype):
    from distrib_la.loader import get_lib
    lib = get_lib('CUDA')
    name = 'lrx_eigh_workspace_bytes' if op == 'eigh' else 'lrx_gemm_workspace_bytes'
    if not hasattr(lib, name):
        raise RuntimeError(f'workspace query requires a native build exporting {name}')
    fn = getattr(lib, name)
    fn.argtypes = ([ctypes.c_int64]*(1+len(sizes)) + [ctypes.c_int]
                   + [ctypes.POINTER(ctypes.c_uint64)]*2)
    fn.restype = ctypes.c_int
    device, host = ctypes.c_uint64(), ctypes.c_uint64()
    rc = fn(ctx, *sizes, int(np.dtype(dtype).kind == 'c'),
            ctypes.byref(device), ctypes.byref(host))
    if rc:
        raise RuntimeError(f'{name} failed with status={rc}, sizes={sizes}, local={ctx == 0}')
    return int(device.value), int(host.value)


@lru_cache(maxsize=128)
def _local_gemm_temp(shapes, dtype, device):
    """Compile the local batched matmul shape, without executing/allocating it."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import SingleDeviceSharding
    sh = SingleDeviceSharding(device)
    fn = jax.jit(jnp.matmul, in_shardings=(sh, sh), out_shardings=sh)
    args = tuple(jax.ShapeDtypeStruct(s, np.dtype(dtype), sharding=sh) for s in shapes)
    memory = fn.lower(*args).compile().memory_analysis()
    if memory is None:
        raise RuntimeError('local GEMM compiler did not return workspace accounting')
    return int(memory.temp_size_in_bytes)


def _workspace_details(plan, op, shapes, dtype):
    """Internal receipt producer; the public contract returns device bytes."""
    import jax
    from distrib_la.plan import Plan, ROUTE_BATCH_RESHARD
    from distrib_la.matmul_plan import GemmPlan
    from distrib_la._cusolvermp import get_or_init_context
    if not isinstance(plan, (Plan, GemmPlan)):
        raise TypeError('workspace query requires a resolved distrib_la Plan or GemmPlan')
    shapes = _shapes(shapes)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float64'), np.dtype('complex128')):
        raise TypeError('workspace query supports float64 and complex128')
    if any(d.platform != 'gpu' for d in plan.mesh.devices.flat):
        raise ValueError('workspace query currently supports CUDA plans only')
    px, py = int(plan.mesh.shape['x']), int(plan.mesh.shape['y'])
    local = isinstance(plan, Plan) and (plan.is_native or (
        (op != 'eigh' or len(shapes[0]) == 3)
        and plan.batched_route == ROUTE_BATCH_RESHARD))
    if op == 'eigh':
        if not isinstance(plan, Plan) or plan.op != 'eigh':
            raise ValueError('eigh query requires an eigh Plan')
        if (len(shapes) != 1 or len(shapes[0]) not in (2, 3)
                or shapes[0][-2] != shapes[0][-1]):
            raise ValueError('eigh shapes must be ((n,n),) or ((batch,n,n),)')
        n = shapes[0][-1]
        if plan.n is not None and plan.n != n:
            raise ValueError('eigh shape differs from the planned n')
        if n % px or n % py:
            raise ValueError('eigh shape must tile the plan mesh')
        if not local and plan.backend != 'cusolvermp':
            raise ValueError('distributed eigh workspace query supports cusolvermp only')
        ctx = 0 if local else get_or_init_context(plan.mesh)
        device, host = _vendor_query(ctx, op, (n,), dtype.str)
        # Local kernels need an info integer in addition to vendor work.
        # Mp info is persistent context state, not per-operation scratch.
        # A native auto route can execute a whole batch; reserve one
        # vendor workspace/info per member rather than assume serial reuse.
        copies = (shapes[0][0] if local and len(shapes[0]) == 3
                  and plan.batched_route != ROUTE_BATCH_RESHARD else 1)
        scratch = copies*(device + (4 if local else 0))
        return dict(device_bytes=scratch, host_bytes=host,
                    vendor_device_bytes=device if local else None,
                    native_device_bytes=device,
                    dynamic_xla_scratch_bytes=scratch,
                    local=local, formula='copies*(native_device_bytes + local_info(4)); Mp native bytes include aligned vendor workspace and a private operand tile',
                    copies=copies,
                    provider='cusolverDn' if local else 'cusolverMp')
    if op not in ('gemm', 'matmul'):
        raise ValueError('workspace op must be eigh or gemm')
    if (len(shapes) != 2 or len(shapes[0]) not in (2, 3)
            or len(shapes[1]) != len(shapes[0])
            or shapes[0][:-2] != shapes[1][:-2]
            or shapes[0][-1] != shapes[1][-2]):
        raise ValueError('GEMM shapes must be matching N,N matrix or batched operand shapes')
    m, k, n = shapes[0][-2], shapes[0][-1], shapes[1][-1]
    batch = shapes[0][0] if len(shapes[0]) == 3 else 1
    if isinstance(plan, GemmPlan) and ((m,k,n,batch) != (plan.m,plan.k,plan.n,plan.nq)
                                        or dtype != np.dtype(plan.dtype)):
        raise ValueError('GEMM shape/dtype differs from its plan')
    if any(v % axis for v, axis in ((m,px),(k,px),(k,py),(n,py))):
        raise ValueError('GEMM shapes must tile the plan mesh')
    return _gemm_workspace_details(plan.mesh, shapes, dtype, local=local,
        backend=plan.backend,
        ctx_handle=plan.ctx_handle if isinstance(plan, GemmPlan) else None)


def _gemm_workspace_details(mesh, shapes, dtype, *, local, backend, ctx_handle=None):
    """Shared query implementation for planned and eager GEMM routes."""
    import jax
    from distrib_la._cusolvermp import get_or_init_context
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    m, k, n = shapes[0][-2], shapes[0][-1], shapes[1][-1]
    batch = shapes[0][0] if len(shapes[0]) == 3 else 1
    if local:
        nb = (batch+px*py-1)//(px*py)
        local_shapes = ((nb,m,k),(nb,k,n))
        device = _local_gemm_temp(local_shapes, dtype.str, jax.local_devices()[0])
        return dict(device_bytes=device, host_bytes=0, vendor_device_bytes=None,
                    dynamic_xla_scratch_bytes=0, local=True,
                    formula='compiled local batched matmul temp_size_in_bytes (workspace upper bound)',
                    provider='XLA local GEMM')
    if backend not in ('cusolvermp', 'cublasmp'):
        raise ValueError('distributed GEMM workspace query supports cublasmp only')
    ctx = ctx_handle if ctx_handle is not None else get_or_init_context(mesh, col_major=False)
    device, host = _vendor_query(ctx, 'gemm', (m,n,k), dtype.str)
    return dict(device_bytes=device, host_bytes=host, vendor_device_bytes=device,
                dynamic_xla_scratch_bytes=0, local=False,
                formula='one vendor workspace shared by all batch slices; retained by native context',
                provider='cublasMp')


def matmul_workspace_bytes_per_rank(mesh, shapes, dtype, *, backend='auto',
                                    batched_route='batch_reshard'):
    """Query the actual eager matmul route, independently of any eigh plan.

    ``shapes`` are effective N,N rank-2/3 operand shapes after any endpoint
    transpose; ``dtype`` is float64/complex128. Returns device workspace
    bytes per rank. Operand transpose staging and output storage are not
    workspace and must be admitted separately by the caller.
    """
    from distrib_la.matmul import resolve_matmul_backend
    from distrib_la.plan import ROUTE_BATCH_RESHARD
    shapes = _shapes(shapes)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float64'), np.dtype('complex128')):
        raise TypeError('workspace query supports float64 and complex128')
    if any(d.platform != 'gpu' for d in mesh.devices.flat):
        raise ValueError('workspace query currently supports CUDA plans only')
    if (len(shapes) != 2 or len(shapes[0]) not in (2, 3)
            or len(shapes[1]) != len(shapes[0])
            or shapes[0][:-2] != shapes[1][:-2]
            or shapes[0][-1] != shapes[1][-2]):
        raise ValueError('GEMM shapes must be matching N,N matrix or batched operand shapes')
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    m, k, n = shapes[0][-2], shapes[0][-1], shapes[1][-1]
    if any(v % axis for v, axis in ((m,px),(k,px),(k,py),(n,py))):
        raise ValueError('GEMM shapes must tile the plan mesh')
    route = str(batched_route).strip().lower()
    provider = resolve_matmul_backend(backend, mesh, batched_route=route)
    return int(_gemm_workspace_details(mesh, shapes, dtype,
        local=route == ROUTE_BATCH_RESHARD, backend=provider)['device_bytes'])


def workspace_bytes_per_rank(plan, op, shapes, dtype) -> int:
    """Return a CUDA dense-plan device workspace bound without solving.

    Parameters
    ----------
    plan : Plan or GemmPlan
        Resolved dense policy and actual mesh. A Plan also supplies its
        local/distributed policy for N,N GEMM. All ranks query collectively
        when the native context has not yet been initialized.
    op : {'eigh', 'gemm'}
        Dense operation. Eigh includes eigenvectors; GEMM shapes describe
        operands after the service's device transpose/adjoint staging.
    shapes : tuple of tuples
        ((n,n),) or ((batch,n,n),) for eigh; ((m,k),(k,n)) or matching
        batched shapes for GEMM. No operand arrays are accepted or retained.
    dtype : numpy dtype
        float64 or complex128.

    Returns
    -------
    int
        Device bytes per rank for operation workspace. Mp eigh's dynamic
        XLA ScratchAllocator allocation is exactly the native query result:
        aligned vendor workspace plus one private input tile. Host workspace and persistent
        context/communicator resources are separate (see WORKSPACE.md).
        GEMM workspace persists: budget its maximum across planned calls
        plus the largest concurrent eigh scratch. Local GEMM uses a
        compiler temporary-byte upper bound, without executing the kernel.
        Context/handle setup and compiler metadata may allocate resources;
        the query allocates no matrix, result or queried workspace buffer.
    """
    return int(_workspace_details(plan, op, shapes, dtype)['device_bytes'])
