"""Query-only dense workspace accounting; no matrix/workspace allocation."""
from functools import lru_cache
import ctypes
import operator

import numpy as np

from distrib_la.resolve import ONE_DEVICE, mesh_platform


def _shapes(shapes):
    try:
        result = tuple(tuple(operator.index(n) for n in shape) for shape in shapes)
    except (TypeError, ValueError) as ex:
        raise ValueError('shapes must be a tuple of operand shape tuples') from ex
    if not result or any(not s or min(s) < 1 for s in result):
        raise ValueError('workspace shapes must have positive extents')
    return result


@lru_cache(maxsize=None)      # one native call per (context, op, sizes, dtype) for the process
def _vendor_query(ctx, op, sizes, dtype):
    from distrib_la.loader import get_lib
    lib = get_lib('CUDA')
    name = 'lrx_lu_workspace_bytes' if op == 'solve_lu' else 'lrx_eigh_workspace_bytes'
    if op == 'solve_lu':
        sizes = sizes[:-1]          # the posted getrs cap: a cache key only (the native reads it)
    if op == 'eigh' and len(sizes) == 2:
        # (side, block): the descriptor the distributed solve runs at. A
        # library without the block query prices one tile per rank at that side.
        name = 'lrx_eigh_workspace_bytes_block'
        if not hasattr(lib, name):
            name, sizes = 'lrx_eigh_workspace_bytes', sizes[:1]
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


def _off_cuda(plan, op, shapes, dtype):
    """Additional scratch outside the compiled program on CPU/ROCm.

    ELPA's pool and rocBLAS regions are explicit results, already priced by
    the executable. Local ROCm eigensolvers still allocate native scratch.
    Persistent contexts and ELPA's small/fallback allocations are separate.
    """
    from distrib_la.plan import Plan, ROUTE_BATCH_RESHARD
    local = (not isinstance(plan, Plan) or plan.is_native or plan.backend == ONE_DEVICE
             or len(shapes[0]) == 3 and plan.known_route(shapes[0], dtype) == ROUTE_BATCH_RESHARD)
    device, src = 0, 'XLA-managed scratch (compiled program)'
    if mesh_platform(plan.mesh) == 'ROCM' and op == 'eigh' and local:
        n = int(shapes[0][-1])
        share = int(shapes[0][0]) if len(shapes[0]) == 3 else 1
        device = local_eigh_workspace(plan.mesh, share, n, dtype,
                                      batched=len(shapes[0]) == 3 or getattr(plan, 'backend', None) == ONE_DEVICE)
        src = 'local ROCm eigh workspace (local_eigh_workspace)' if device else src
    return dict(device_bytes=device, host_bytes=0, vendor_device_bytes=device or None,
                dynamic_xla_scratch_bytes=device, local=local, formula=src,
                provider=getattr(plan, 'backend', 'XLA'))


@lru_cache(maxsize=None)
def local_eigh_workspace(mesh, share, n, dtype, *, batched=False) -> int:
    """Native scratch outside compiled buffers for a rank's local eigensolve.

    CUDA uses its native size query per local matrix. ROCm's qualified native
    target owns its exact arena in the compiled program; local batched calls
    need no extra charge there. Legacy JAX solves outside its window, and a
    scalar complex128 solve's conditional JAX Jacobi retry, retain the existing
    measured one-matrix allowance. ``batched`` describes the local call's rank,
    not a global batch count or a backend selection.
    """
    platform = mesh_platform(mesh)
    if platform == 'CUDA':
        return int(share) * _vendor_query(0, 'eigh', (int(n),), np.dtype(dtype).str)[0]
    from distrib_la.local_eigh import BATCHED_EIGH_ROUTE, batched_eigh_row
    row = BATCHED_EIGH_ROUTE.get(platform)
    win = None if row is None else row.windows.get(np.dtype(dtype).name)
    if win is None:
        return 0
    native = batched_eigh_row((int(share), int(n), int(n)), dtype, platform)
    if native is not None and (batched or np.dtype(dtype) != np.dtype('complex128')):
        return 0
    # The legacy loop holds one matrix's scratch at a time. The scalar
    # complex128 native route can still enter the existing JAX Jacobi retry.
    return win.workspace_tiles * int(n) ** 2 * np.dtype(dtype).itemsize + int(n) * 8


def _workspace_details(plan, op, shapes, dtype):
    """Internal receipt producer; the public contract returns device bytes."""
    import jax
    from distrib_la.plan import Plan, ROUTE_BATCH_RESHARD
    from distrib_la.matmul_plan import GemmPlan
    if not isinstance(plan, (Plan, GemmPlan)):
        raise TypeError('workspace query requires a resolved distrib_la Plan or GemmPlan')
    shapes = _shapes(shapes)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float64'), np.dtype('complex128')):
        raise TypeError('workspace query supports float64 and complex128')
    # Every platform checks the request against its plan first: the CUDA branch below is
    # the one CPU never runs, so its shape contract must not live only there.
    if op == 'eigh':
        if not isinstance(plan, Plan) or plan.op != 'eigh':
            raise ValueError('eigh query requires an eigh Plan')
        if (len(shapes) != 1 or len(shapes[0]) not in (2, 3)
                or shapes[0][-2] != shapes[0][-1]):
            raise ValueError('eigh shapes must be ((n,n),) or ((batch,n,n),)')
        if plan.n is not None and plan.n != shapes[0][-1]:
            raise ValueError('eigh shape differs from the planned n')
    elif op in ('gemm', 'matmul'):
        if (len(shapes) != 2 or len(shapes[0]) not in (2, 3)
                or len(shapes[1]) != len(shapes[0])
                or shapes[0][:-2] != shapes[1][:-2]
                or shapes[0][-1] != shapes[1][-2]):
            raise ValueError('GEMM shapes must be matching N,N matrix or batched operand shapes')
        batch = shapes[0][0] if len(shapes[0]) == 3 else 1
        if isinstance(plan, GemmPlan) and ((shapes[0][-2], shapes[0][-1], shapes[1][-1], batch)
                                           != (plan.m, plan.k, plan.n, plan.nq)
                                           or dtype != np.dtype(plan.dtype)):
            raise ValueError('GEMM shape/dtype differs from its plan')
    else:
        raise ValueError('workspace op must be eigh or gemm')
    if mesh_platform(plan.mesh) != 'CUDA':
        return _off_cuda(plan, op, shapes, dtype)
    px, py = int(plan.mesh.shape['x']), int(plan.mesh.shape['y'])
    # one_device (the size route) runs each matrix whole on one device with the native
    # kernel: a local workspace, held_per_rank copies per round (the one-device formula).
    one_device = isinstance(plan, Plan) and plan.backend == ONE_DEVICE
    local = isinstance(plan, Plan) and (plan.is_native or one_device or (
        (op != 'eigh' or len(shapes[0]) == 3)
        and plan.known_route(shapes[0], dtype) == ROUTE_BATCH_RESHARD))
    if op == 'eigh':
        n = shapes[0][-1]
        if not local and (n % px or n % py):
            raise ValueError('eigh shape must tile the plan mesh')
        if not local and plan.backend != 'cusolvermp':
            raise ValueError('distributed eigh workspace query supports cusolvermp only')
        if not local:
            from distrib_la._cusolvermp import get_or_init_context
        ctx = 0 if local else get_or_init_context(plan.mesh)
        side = n
        if local:
            # jaxlib's cusolverDn syevd allocates its workspace outside the program's
            # buffers (no workspace output): the native size query, process handle.
            device, host = _vendor_query(0, op, (n,), dtype.str)
        else:
            from distrib_la._cusolvermp import solve_layout
            side, block = solve_layout(n, px)
            device, host = _vendor_query(ctx, op, (side, block), dtype.str)
        # Mp's workspace is allocated outside XLA, one per member a call holds; the checked
        # chain, its padded operand and the gathered retry are XLA buffers of the compiled
        # program (distrib_la.plan.compiled_bytes), not workspace.
        batch = int(shapes[0][0]) if local and len(shapes[0]) == 3 else 1
        copies = (-(-batch // (px * py)) if not one_device and
                  plan.known_route(shapes[0], dtype) == ROUTE_BATCH_RESHARD else batch)
        scratch = copies * device
        return dict(device_bytes=scratch, host_bytes=host,
                    vendor_device_bytes=None if local else device, native_device_bytes=device,
                    dynamic_xla_scratch_bytes=0, local=local,
                    formula='cusolverMp size query (the aligned vendor workspace), one per member; '
                            'local and XLA-held bytes are in the compiled program',
                    checked_device_bytes=0, copies=copies,
                    provider=('cusolverDn (one_device)' if one_device else 'cusolverDn')
                    if local else 'cusolverMp')
    m, k, n = shapes[0][-2], shapes[0][-1], shapes[1][-1]
    # A local GEMM holds whole operands per device: only the distributed one tiles the mesh.
    if not local and any(v % axis for v, axis in ((m,px),(k,px),(k,py),(n,py))):
        raise ValueError('GEMM shapes must tile the plan mesh')
    return _gemm_workspace_details(plan.mesh, shapes, dtype, local=local,
        backend=plan.backend,
        ctx_handle=plan.ctx_handle if isinstance(plan, GemmPlan) else None)


def _gemm_workspace_details(mesh, shapes, dtype, *, local, backend, ctx_handle=None):
    """Shared query implementation for planned and eager GEMM routes."""
    import jax
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
    # Every distributed GEMM is distrib_la.panel_matmul: XLA-managed, no vendor workspace.
    return dict(device_bytes=0, host_bytes=0, vendor_device_bytes=0,
                dynamic_xla_scratch_bytes=0, local=False,
                formula='panel_matmul: XLA-managed panels, no vendor workspace',
                provider='panel_matmul')


def matmul_workspace_bytes_per_rank(mesh, shapes, dtype, *, backend='auto',
                                    batched_route='batch_reshard'):
    """Query the actual eager matmul route, independently of any eigh plan.

    ``shapes`` are effective N,N rank-2/3 operand shapes after any endpoint
    transpose; ``dtype`` is float64/complex128. Returns device workspace
    bytes per rank (0 on a non-CUDA mesh: XLA-managed). Operand transpose staging and output storage are not
    workspace and must be admitted separately by the caller.
    """
    from distrib_la.matmul import resolve_matmul_backend
    from distrib_la.plan import ROUTE_BATCH_RESHARD
    shapes = _shapes(shapes)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float64'), np.dtype('complex128')):
        raise TypeError('workspace query supports float64 and complex128')
    if mesh_platform(mesh) != 'CUDA':
        return 0
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


def workspace_bytes_per_rank(plan, op, shapes, dtype, *, checked=False) -> int:
    """Return dense-plan scratch allocated outside its compiled buffers.

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
    checked : bool
        Include the native scratch of an admitted gathered EIGH retry in
        the bound, for a caller pricing a complete checked program.

    Returns
    -------
    int
        Device bytes per rank for additional operation workspace. ELPA's
        explicit arena is already in the executable's price; its small and
        fallback allocations and persistent contexts remain separate. Mp
        eigh's dynamic XLA ScratchAllocator allocation uses its native query.
        Host workspace and context/communicator resources are separate.
        GEMM workspace persists: budget its maximum across planned calls
        plus the largest concurrent eigh scratch. Local GEMM uses a
        compiler temporary-byte upper bound, without executing the kernel.
        Context/handle setup and compiler metadata may allocate resources;
        the query allocates no matrix, result or queried workspace buffer.
    """
    need = int(_workspace_details(plan, op, shapes, dtype)['device_bytes'])
    if checked and op == 'eigh' and mesh_platform(plan.mesh) in ('CUDA', 'ROCM') and not plan.is_native:
        from distrib_la.plan import _gathered_admitted
        n = int(shapes[0][-1])
        if _gathered_admitted(plan.mesh, (n, n), dtype, plan.budget_bytes):
            need = max(need, local_eigh_workspace(plan.mesh, 1, n, dtype))
    return need


def lu_workspace_bytes(mesh, n, dtype) -> int:
    """Device bytes per rank one cuSOLVERMp solve_lu of an n x n system against n right-hand
    sides allocates beside its operands (cudaMallocAsync, outside XLA's pool), per matrix: the
    CUDA leg's measured row, 11.2 local panels at its blocks (``lxkit.memory.vendor``). Shapes
    only: no vendor query -- cuSOLVERMp's bufferSize builds a cuBLASMp grid (an NCCL communicator
    split) per call, which deadlocked a rank in libfabric's memory hooks (run30k, 2026-10-19)."""
    from distrib_la._cusolvermp import lu_blocks
    from lxkit import leg_for_platform, memory
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    mb, nb = lu_blocks(int(n), px, py)
    item = np.dtype(dtype).itemsize
    return int(memory.vendor(leg_for_platform("CUDA"), "cusolvermp", "solve_lu", int(n), px * py, item,
                             unit=(int(n) // px + mb) * nb * item).fixed)


@lru_cache(maxsize=128)
def local_lu_workspace(mesh, n, dtype) -> int:
    """Dynamic scratch of one CUDA JAX GETRF, beside its compiled buffers.

    JAX's scalar LU uses this same legacy cuSOLVER buffer-size query and
    allocates its result through the FFI scratch allocator. CPU scratch is
    already in its compiled program; no CUDA library is loaded on that leg.
    """
    n = operator.index(n)
    dtype = np.dtype(dtype)
    letters = {'float32': 'S', 'float64': 'D', 'complex64': 'C', 'complex128': 'Z'}
    if not 0 < n <= np.iinfo(np.int32).max or dtype.name not in letters:
        raise ValueError('local LU workspace needs 0 < n <= INT_MAX and a real/complex floating dtype')
    platform = mesh_platform(mesh)
    if platform == 'cpu':
        return 0
    if platform != 'CUDA':
        raise NotImplementedError('local LU workspace query is available on CUDA and CPU')
    from distrib_la.loader import get_lib
    letter = letters[dtype.name]
    lib = get_lib('CUDA')
    create, destroy = lib.cusolverDnCreate, lib.cusolverDnDestroy
    create.argtypes, create.restype = [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int
    destroy.argtypes, destroy.restype = [ctypes.c_void_p], ctypes.c_int
    query = getattr(lib, f'cusolverDn{letter}getrf_bufferSize')
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                      ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    handle, size = ctypes.c_void_p(), ctypes.c_int()
    rc = create(ctypes.byref(handle))
    if rc:
        raise RuntimeError(f'cusolverDnCreate failed with status {rc}')
    try:
        rc = query(handle, int(n), int(n), None, int(n), ctypes.byref(size))
        if rc or size.value < 0:
            raise RuntimeError(f'local LU workspace query failed with status {rc}, n={n}, size={size.value}')
        return int(size.value) * dtype.itemsize
    finally:
        destroy(handle)


def lu_workspace_query(mesh, n, nrhs, dtype, cap) -> int:
    """Device bytes per rank the next cuSOLVERMp ``n x nrhs`` solve_lu allocates beside its
    operands with ``cap`` posted (``Plan.chunk_room``): the native size query
    (``lrx_lu_workspace_bytes``: getrf, the getrs chunk ``cap`` admits, pivots) on the solve's own
    context and descriptors, which sizes and caches the plan the solve then runs; cached per cap.
    Collective: every rank asks at the same point."""
    from distrib_la._cusolvermp import get_or_init_context, lu_blocks, rhs_block
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    mb, nb = lu_blocks(int(n), px, py)
    sizes = (int(n), int(nrhs), mb, nb, mb, rhs_block(int(n), int(nrhs), px, py, mb))
    return _vendor_query(get_or_init_context(mesh, col_major=False), 'solve_lu', sizes + (int(cap),), dtype)[0]


def eigh_stack_proposal(mesh, shape, dtype) -> int:
    """Route (c)'s first attempt on the face stack ``shape`` (nb, n, n) in closed form, where the
    program that will run is not known yet (its cut and room are the call's): five matrices per
    rank's share of the batch (the operand in batch layout, the vectors both ways, the solver's
    work; the CPU compile reads 3.1-4.5 at n 64-512), plus the solver workspace outside the
    program (``local_eigh_workspace``).  A proposal: the run's own program is the price."""
    from distrib_la.local_eigh import batched_workspace_bytes
    share, n = -(-int(shape[0]) // int(mesh.size)), int(shape[-1])
    arena = batched_workspace_bytes((share, n, n), dtype, mesh_platform(mesh))
    return (5 * share * n * n * np.dtype(dtype).itemsize + arena
            + local_eigh_workspace(mesh, share, n, dtype, batched=True))


def eigh_stack_bytes(plan, shape, dtype, phase="all") -> int:
    """Per-rank device bytes a batched eigh of the face stack ``shape`` (nb, n, n) adds beside
    its operand: the program that runs it, compiled at the abstract operand and never run
    (route (c): ``distrib_la.plan._stack_bytes``, plus the solver workspace outside the
    program, ``local_eigh_workspace``; otherwise ``Plan.mesh_bytes``, the plan's own program
    plus its provider's workspace query).  ``phase`` is the one the plan runs (``Plan`` line
    ``phase = ...``): "first" for an eager call, whose retries price their own program, "all"
    for a stack traced in a caller's program.  Every rank computes the same figure."""
    from distrib_la.plan import ROUTE_BATCH_RESHARD, _gathered_admitted, _stack_bytes
    nb, n = int(shape[0]), int(shape[-1])
    ranks = int(plan.mesh.shape['x']) * int(plan.mesh.shape['y'])
    if plan.batched_route == ROUTE_BATCH_RESHARD:
        return (_stack_bytes('eigh', plan.mesh, nb, n, dtype, 1, phase,
                             _gathered_admitted(plan.mesh, (n, n), dtype, plan.budget_bytes))
                + local_eigh_workspace(plan.mesh, -(-nb // ranks), n, dtype, batched=True))
    return plan.mesh_bytes('eigh', nb, n, dtype, phase)
