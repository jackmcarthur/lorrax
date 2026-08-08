"""Distributed BSE kernels (ring collectives and sharded matvecs)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map as _shard_map_fn

from common.collectives import gather_to_host, prepare_mesh, resolve_mesh
from common.contract_bands import contract_bands_block_reshard
from common.fft_helpers import (
    local_fftn3,
    local_ifftn3,
    make_sharded_fftn_3d,
    make_sharded_ifftn_3d,
)
from common.vma import mark_varying
from .bse_io import _find_restart_file, _load_ring_subset, load_bse_data_from_restart_sharded
from .bse_serial import apply_D, apply_bse_hamiltonian_single_device, compute_pair_amplitude


jax.config.update("jax_enable_x64", True)


def create_mesh_xy(px: int, py: int, devices: Optional[list] = None) -> Mesh:
    """THE ('x','y') BSE device mesh, warmed — for a driver that names px/py.

    Every BSE entry point must get its mesh from here or from
    :func:`create_mesh_2d` (which is this function with the square shape
    chosen for it).  There is exactly one mesh factory in ``src/bse/`` and
    this is it.  Only SQUARE shapes are accepted (owner ruling, repo
    ``docs/architecture/decisions.md`` 2026-08-01): a ``px != py`` request
    refuses rather than building a rectangular mesh.

    WHY THIS FUNCTION EXISTS AT ALL.  Four byte-identical
    ``_create_mesh_xy(px, py)`` bodies used to live in ``bse_w_exact``,
    ``bse_feast``, ``bse_pseudopoles`` (and, by import, ``bse_kpm``), none of
    which warmed anything, while only ``create_mesh_2d`` did.  So five BSE
    drivers — ``exciton_bands``, ``bse_w_exact``, ``bse_feast``, ``bse_kpm``,
    ``bse_pseudopoles`` — silently ran WITHOUT the clique warm-up that
    ``bse_jax`` depends on, and each one dies at P>1 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` the moment a collective is
    first issued from inside a jitted program (see
    :func:`common.collectives.warm_mesh_cliques`: 32 refusals at P=16, gate
    7881216).  A duplicated constructor is how a load-bearing side effect goes
    missing without anybody deleting a line.

    THE WARM-UP IS SETUP-TIME AND SIZE-INDEPENDENT.  ``prepare_mesh`` fires
    three 1-element ``psum``s (one per mesh axis + one over all axes) plus,
    on GPU, ``runtime.nccl_warmup``'s three tiny reductions.  ``O(log P)``
    latency, once per process, independent of ``N_mu`` / ``N_k·N_q`` / the
    trial-block width; it emits no HLO into any solver kernel, so the BSE
    communication pattern and every intermediate's sharding are untouched.
    That is what makes it admissible under the standing BSE rule.

    Collective: every rank must reach this together.  Do not make the call
    rank-conditional.
    """
    if devices is None:
        devices = jax.devices()
    px, py = int(px), int(py)
    if px != py:
        raise ValueError(
            f"create_mesh_xy: a {px}x{py} mesh was requested, but only "
            f"square 2-D meshes are supported (repo "
            f"docs/architecture/decisions.md, 2026-08-01) — rectangular "
            f"meshes complicate ScaLAPACK grid geometry and the "
            f"divisibility contracts for no measured benefit.  Request "
            f"px == py (and a square process count).")
    if px * py > len(devices):
        raise ValueError(
            f"Requested px*py={px*py} devices, only {len(devices)} available")
    # Reuse the run's canonical mesh when the request IS that mesh: since
    # the BSE drivers start with ``runtime.initialize_communicator_stack``
    # (2026-08-01), the canonical square ('x','y') mesh already exists — an
    # equal-but-distinct copy here would be a second set of communicators
    # and a second copy of every shape-keyed jit cache (the
    # ``collectives.single_device_mesh`` identity contract).  A different
    # requested shape is a deliberate second mesh and is built as before.
    canonical = resolve_mesh(axis_names=("x", "y"))
    if (tuple(int(n) for n in canonical.devices.shape) == (px, py)
            and list(canonical.devices.flat) == list(devices[: px * py])):
        mesh = canonical
    else:
        mesh = Mesh(np.array(devices[: px * py]).reshape(px, py),
                    axis_names=("x", "y"))
    # resolve_mesh (pass-through here) + warm_mesh_cliques + nccl_warmup.
    mesh = prepare_mesh(mesh)
    _warm_process_allgather(mesh)
    return mesh


def _warm_process_allgather(mesh: Mesh) -> None:
    """Create the ``multihost_utils.process_allgather`` clique on this thread.

    ``warm_mesh_cliques`` warms the cliques of OUR mesh: one per axis plus the
    world set, all created by ``shard_map(psum)`` over a ``('x','y')`` Mesh.
    ``jax.experimental.multihost_utils.process_allgather`` does not use that
    mesh — it builds its OWN one-dimensional ``Mesh(jax.devices(), 'processes')``
    and issues an all-gather on it.  That is a different collective on a
    differently-shaped mesh, and the x/y/world warm-up does not cover it.

    MEASURED, not reasoned: with the mesh warm-up in place and no allgather
    warm-up, every cell of job 7882523 (P=16) died at
    ``exciton_bands.py`` line 592 -> ``_gather_host`` ->
    ``common/collectives.py`` ``process_allgather`` with
    "MPI: Communicator requested from a thread that is not the one MPI was
    initialized from" — 4 of 4 cells, same frame every time.  The gather in
    question is the driver's on-grid diagnostic gate on the mu-sharded
    Q-shifted psi_c, which is genuinely process-spanning, so it MUST take the
    allgather arm; it cannot be avoided by branching differently.

    THE WARM-UP MUST TRAVEL THE REAL PATH.  A first attempt handed
    ``process_allgather`` a HOST numpy array; job 7882531 then failed in
    exactly the same frame as before (4 of 4 cells), because a host operand
    never creates the device clique the real call needs.  So the warm-up now
    builds a genuinely mesh-SHARDED (hence, at P>1, non-fully-addressable)
    array and pushes it through ``gather_to_host`` — the same function, the
    same arm, the same trailing ``np.asarray`` that materialises the result.
    A warm-up that does not reproduce the failing call is not a warm-up; that
    is the same lesson as a negative control that withholds the wrong thing.

    COST.  One all-gather of ``P`` float64 — 512 B at P=64 — once per process
    at driver setup.  ``O(log P)`` latency, independent of ``N_mu``,
    ``N_k*N_q`` and the trial-block width, and it emits no HLO into any solver
    kernel.  Same admissibility argument as ``warm_mesh_cliques``.

    No-op at ``process_count() <= 1``.  Best-effort: a failure here is not
    itself fatal (the real call will raise with a better frame), but it is
    announced.
    """
    if jax.process_count() <= 1:
        return
    try:
        from common.collectives import gather_to_host
        n = int(mesh.devices.size)
        probe = jax.device_put(
            jnp.zeros(n, dtype=jnp.float64),
            NamedSharding(mesh, P(tuple(mesh.axis_names))))
        out = gather_to_host(probe)
        assert out.shape == (n,), out.shape
    except Exception as exc:                      # noqa: BLE001
        if jax.process_index() == 0:
            print(f"  [collectives] process_allgather warm-up failed "
                  f"({type(exc).__name__}: {exc}); a later gather may refuse "
                  f"under JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi", flush=True)


def create_mesh_2d(devices: Optional[list] = None) -> Mesh:
    """Create the square 2-D device mesh for BSE sharding.

    Thin wrapper over :func:`create_mesh_xy` — same warm-up, same contract,
    and the same square-only rule as ``common.collectives.resolve_mesh``:
    a device count that is not a perfect square refuses with the square
    count to request (the refusal fires in ``resolve_mesh``'s vocabulary
    via :func:`create_mesh_xy`'s canonical-mesh reuse, or here).
    """
    import math

    if devices is None:
        devices = jax.devices()

    n_devices = len(devices)
    s = math.isqrt(n_devices)
    if s * s != n_devices:
        raise RuntimeError(
            f"create_mesh_2d: {n_devices} devices is not a perfect square, "
            f"and only square 2-D meshes are supported (repo "
            f"docs/architecture/decisions.md, 2026-08-01).  Request "
            f"{s * s} (a {s}x{s} mesh) or {(s + 1) * (s + 1)} "
            f"(a {s + 1}x{s + 1} mesh) processes.")

    return create_mesh_xy(s, s, devices)


def make_bse_shardings(mesh_xy: Mesh) -> SimpleNamespace:
    return SimpleNamespace(
        X=NamedSharding(mesh_xy, P(None, "x", "y", None)),
        X_full=NamedSharding(mesh_xy, P(None, None, "x", "y", None)),
        psi_x=NamedSharding(mesh_xy, P(None, None, None, "x")),
        psi_y=NamedSharding(mesh_xy, P(None, None, None, "y")),
        V=NamedSharding(mesh_xy, P("x", "y")),
        W=NamedSharding(mesh_xy, P("x", "y", None, None, None)),
        eps=NamedSharding(mesh_xy, P(None, None)),
        R=NamedSharding(mesh_xy, P(None, "x", None, None, "y")),
        T=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        U=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        A=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        D=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        S=NamedSharding(mesh_xy, P(None, "y", None)),
        S_k0=NamedSharding(mesh_xy, P(None, "y")),
        U_mu=NamedSharding(mesh_xy, P(None, "x", None)),
        d_mu=NamedSharding(mesh_xy, P(None, "x")),
    )


def _ring_perm(axis_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % axis_size) for i in range(axis_size))


# ---------------------------------------------------------------------------
# Why the ring accumulators below are ``mark_varying``-ed
# ---------------------------------------------------------------------------
# Every ring sum here is ``acc = jnp.zeros(...)`` followed by a fori_loop whose
# body does ``acc = acc + <something built from a sharded operand>``.  From jax
# 0.7.0 ``shard_map`` tracks varying manual axes, and the loop carry's VMA set
# must be EQUAL at input and output -- so the empty set on the zeros init does
# not match the {x,y} the body produces, and the loop is rejected:
#
#   TypeError: scan body function carry input and carry output must have equal
#   types ... complex128[1,2,399,9] vs complex128[1,2,399,9]{x,y}
#
# (fori_loop with static bounds lowers to scan, which is why the message says
# "scan body".)  These shard_maps go through ``jax.shard_map`` with checking
# ON, so the marking is required; the ``check_vma=False`` shard_maps elsewhere
# in the tree are exempt.  ``mark_varying`` is the identity on jax 0.5.3, so
# the production leg is bit-unchanged.  See ``common/vma.py``.
#
# ('x','y') is what jax itself names in the error, and it is what the bodies
# introduce: the ring buffers come from X (in_spec P(None,'x','y',None)) and
# the psi slices from operands on 'x' or 'y'.  Do NOT widen a mark "to be
# safe" -- a carry marked over an axis it does not vary on fails when it
# leaves through a replicated out_specs entry.
def _ring_sum_valence(
    X: jax.Array,
    psi_v_Y: jax.Array,
    v_chunk: int,
    py: int,
    nu_local: int,
) -> jax.Array:
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_v_Y.shape

    R0 = mark_varying(
        jnp.zeros((X.shape[0], X.shape[1], nk, nspinor, nu_local), dtype=X.dtype),
        ("x", "y"))
    perm = _ring_perm(py)

    def step(i, carry):
        buf, R = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk, v_chunk, nspinor, nu_local)
        )
        R = R + jnp.einsum("kv sN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, R

    _, R_total = lax.fori_loop(0, py, step, (X, R0))
    return R_total


def _ring_sum_conduction(
    R: jax.Array,
    psi_c_X: jax.Array,
    c_chunk: int,
    px: int,
    mu_local: int,
) -> jax.Array:
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_X.shape
    nu_local = R.shape[4]

    T0 = mark_varying(
        jnp.zeros((R.shape[0], mu_local, nu_local, nspinor, nspinor, nk),
                  dtype=R.dtype),
        ("x", "y"))
    perm = _ring_perm(px)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px
        c_start = origin * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_X, (z, c_start, z, z), (nk, c_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kctM,bcksN->bMNtsk", psi_c_slice, buf)
        buf = lax.ppermute(buf, axis_name="x", perm=perm)
        return buf, T

    _, T_total = lax.fori_loop(0, px, step, (R, T0))
    return T_total


def _ring_sum_conduction_first(
    X: jax.Array,
    psi_c_Y: jax.Array,
    c_chunk: int,
    px: int,
    nu_local: int,
) -> jax.Array:
    """Sum over conduction first: R(v,k,s,nu) = sum_c psi_c^*(nu) X(c,v,k)."""
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_Y.shape
    v_chunk = X.shape[2]

    R0 = mark_varying(
        jnp.zeros((X.shape[0], v_chunk, nk, nspinor, nu_local), dtype=X.dtype),
        ("x", "y"))
    perm = _ring_perm(px)

    def step(i, carry):
        buf, R = carry
        origin = (axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px
        c_start = origin * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_Y, (z, c_start, z, z), (nk, c_chunk, nspinor, nu_local)
        )
        R = R + jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_slice), buf)
        buf = lax.ppermute(buf, axis_name="x", perm=perm)
        return buf, R

    _, R_total = lax.fori_loop(0, px, step, (X, R0))
    return R_total


def _ring_sum_valence_second(
    R: jax.Array,
    psi_v_X: jax.Array,
    v_chunk: int,
    py: int,
    mu_local: int,
) -> jax.Array:
    """Finish sum over valence: T(mu,nu,t,s,k) = sum_v psi_v(mu) R(v,k,s,nu)."""
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_v_X.shape
    nu_local = R.shape[4]

    T0 = mark_varying(
        jnp.zeros((R.shape[0], mu_local, nu_local, nspinor, nspinor, nk),
                  dtype=R.dtype),
        ("x", "y"))
    perm = _ring_perm(py)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk, v_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_slice, buf)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, T

    _, T_total = lax.fori_loop(0, py, step, (R, T0))
    return T_total


def apply_V_ring(
    X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_Y: jax.Array,
    M_X: jax.Array,
    V_q0: jax.Array,
    nk: int,
    px: int,
    py: int,
) -> jax.Array:
    # ``M_X`` (nk, c_full, v_full, μ_loc): the hoisted decode-side exchange pair
    # amplitude ``Σ_s conj(ψ_c^X) ψ_v^X`` (μ on x), precomputed ONCE per solve
    # (audit P3) rather than rebuilt from ψ every matvec. The encode side stays
    # fused into the ψ_c^Y/ψ_v^Y ring sum below (it is X-dependent, not hoistable).
    nb_trial = X.shape[0]
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

    nk_local = psi_c_Y.shape[0]
    nspinor = psi_c_Y.shape[2]
    nu_local = psi_c_Y.shape[-1]
    nc_full = M_X.shape[1]
    mu_local = M_X.shape[-1]

    c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
    z = jnp.int32(0)
    psi_c_slice = lax.dynamic_slice(
        psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
    )

    A0 = mark_varying(
        jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=X.dtype),
        ("x", "y"))
    perm_y = _ring_perm(py)

    def step_y(i, carry):
        buf, A = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk_local, v_chunk, nspinor, nu_local)
        )
        # Forward (encode) leg carries the CONJUGATED vertex conj(M) =
        # ψ_c·conj(ψ_v) — see the K^x = M V M† note at the back-contract below.
        R_v = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
        A = A + jnp.einsum("kcsN,bcksN->bcNk", psi_c_slice, R_v)
        buf = lax.ppermute(buf, axis_name="y", perm=perm_y)
        return buf, A

    _, A_local = lax.fori_loop(0, py, step_y, (X, A0))

    S0 = mark_varying(
        jnp.zeros((nb_trial, nu_local, nk_local), dtype=X.dtype),
        ("x", "y"))
    perm_x = _ring_perm(px)

    def step_x(i, carry):
        buf, S = carry
        S = S + jnp.sum(buf, axis=1)
        buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
        return buf, S

    _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

    # Exchange (q=0) is DENSE in (k,k') — SUM over k' into a k-free
    # transition-Hartree density, one V_q0 solve, then broadcast back at every
    # k in the decode (VERDICT.md).  k is a replicated (unsharded) axis here, so
    # the reduction is device-local.  The two 1/sqrt_nk compose to 1/Nk.
    S_total = jnp.sum(S_total, axis=2) / sqrt_nk  # (b, nu_local)

    U_partial = jnp.einsum("MN,bN->bM", V_q0, S_total)  # (b, mu_local)
    U = lax.psum(U_partial, axis_name="y")

    v_start_local = axis_index_y * jnp.asarray(v_chunk, dtype=jnp.int32)
    # Slice this rank's y-block of the valence axis out of the hoisted M_X (was:
    # slice ψ_v^X then contract with ψ_c^X — identical values, one fewer GEMM/iter).
    M_X_slice = lax.dynamic_slice(
        M_X, (z, z, v_start_local, z), (nk_local, nc_full, v_chunk, mu_local)
    )
    # Back-contract carries the BARE vertex: K^x = M V M†, fixed by the
    # transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk.  (The reverse assignment
    # builds conj(K^x), which cannot be covariant alongside the correct W term.)
    # This also sets the non-TDA B block: apply_V_ring_B conjugates ψ^Y on the
    # way in, so its forward leg is the bare M and it assembles B = M V M^T.
    VX_partial = jnp.einsum("kcvM,bM->bcvk", M_X_slice, U)  # broadcast over k
    VX = lax.psum_scatter(VX_partial, axis_name="x", scatter_dimension=1, tiled=True)

    return VX / sqrt_nk


def build_bse_ring_matvec(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    low_mem: bool = True,
    include_W: bool = True,
):
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _encode_T(X, psi_c_X, psi_v_Y):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_c_X.shape[-1]
        n_rmu_local_Y = psi_v_Y.shape[-1]
        R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
        T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)
        return T

    encode_T_ring = _shard_map_fn(
        _encode_T,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_gather(X, psi_c_X, psi_v_Y):
        X_full_v = lax.all_gather(X, "y", axis=2, tiled=True)
        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_Y), X_full_v)
        R_full_c = lax.all_gather(R, "x", axis=1, tiled=True)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_full_c)
        return T

    encode_T_gather = _shard_map_fn(
        _encode_T_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, M_X, V_q0, nk, px, py)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    # Custom-partitioned FFTs on the (kx, ky, kz) axes — those axes are
    # ``None``-sharded in T (sh.T) and W_R (sh.W), so the FFT can run
    # locally on every device.  Plain ``jnp.fft.ifftn`` / ``fftn`` on a
    # sharded tensor forces XLA to all-gather the entire array before
    # the FFT — see ``common.fft_helpers`` for the JAX bug this works
    # around.  In the BSE Lanczos loop those gathers cost ~5 s over
    # 200 matvecs on Si 4×4×4 (profile_sharded_v2/trace_summary.md).
    # T_k 8D spec: (b, μ, ν, ns, ns, kx, ky, kz) — same μ,ν shardings as
    # storage T (6D) but with last nk axis split into 3 replicated dims.
    _T_8d_spec = P(None, "x", "y", None, None, None, None, None)
    _T_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')
    _T_local_fftn = make_sharded_fftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')

    # ψ†Uψ decode = contract_bands_block_reshard, extra="leading" (owner
    # order 2026-07-29; adoption map wk_REL/contract_bands_notes.md §6.2 —
    # the CLEAN drop-in site: the b-stacked U already exists, so the stack
    # axis is free).  Replaces the partitioner-chosen collectives of the
    # historical einsum pair ("kctM,bMNtsk->bcNsk" then "kvsN,bcNsk->bcvk",
    # c-replicated intermediate, LARGE payload on the strided 'x' groups —
    # the exact inversion the primitive's §3.2 policy refuses) with the
    # structural stacked psum_scatter chain: large partial over the
    # node-local 'y' groups, small final over 'x', all b trials on ONE
    # collective per mesh axis (AK.9), impl=mpi warm-up inherited from the
    # factory.  Value-level identical (contraction reassociation — gate at
    # 1e-12, not bit-exact).  The transposes below are rank-local
    # (sharded axes preserved: M stays on 'x', N on 'y', c on 'x', v on
    # 'y'); the U transpose to the primitive's k-leading layout is priced
    # by the parity/perf gate, and composes with the future flat-k conv
    # layout (map §6.1 route (a)) which emits k-leading natively.
    _w_decode = contract_bands_block_reshard(
        mesh_xy, extra="leading",
        divisibility_hint=(
            "BSE callers: n_cond_pad / n_val_pad already pad c to p_x and "
            "v to p_y (bse_io loader); an indivisible window here means an "
            "unpadded/hand-built operand."))

    def _apply_W_from_T(T, psi_c_X, psi_v_Y, W_R):
        nspinor = psi_c_X.shape[2]
        nb_trial = T.shape[0]
        n_rmu_local_X = T.shape[1]
        n_rmu_local_Y = T.shape[2]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))

        T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
        T_R = _T_local_ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = _T_local_fftn(U_R)
        U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

        # (b, M, N, t, s, k) -> (b, k, t, M, s, N): the primitive's
        # canonical O layout (extra="leading"); ψ_v (k, v, s, N) ->
        # (k, s, N, v) = ψ_right.  conj(ψ_c) is applied inside.
        O_b = jnp.transpose(U, (0, 5, 3, 1, 4, 2))
        psi_v_snv = jnp.transpose(psi_v_Y, (0, 2, 3, 1))
        out = _w_decode(psi_c_X, O_b, psi_v_snv)     # (b, nk, c_X, v_Y)
        WX = jnp.transpose(out, (0, 2, 3, 1))        # (b, c_X, v_Y, nk)
        return WX / sqrt_nk

    apply_W_from_T = jax.jit(
        _apply_W_from_T,
        in_shardings=(sh.T, sh.psi_x, sh.psi_y, sh.W),
        out_shardings=sh.X,
        # NB: T (arg 0) is NOT donated — the WX output has a different shape
        # (nt,c,v,k) so the donation is always declined (no aliasable output) and
        # emits no fallback copy. Dropping the cosmetic donate_argnums silences the
        # recurring "donated buffers not usable" warning (audit P5, JOINT_FINDINGS §3).
    )

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _matvec_impl(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
                     M_X, M_Y):
        # M_X: hoisted decode-side exchange pair amplitude (audit P3). M_Y / psi_v_X
        # are unused here — kept for a uniform matvec signature with the stack path.
        D_term = apply_D_term(X, eps_c, eps_v)
        V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        if not include_W:
            return D_term + V_term
        T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return D_term + V_term - W_term

    return jax.jit(
        _matvec_impl,
        in_shardings=(
            sh.X,
            sh.psi_x,
            sh.psi_y,
            sh.psi_x,
            sh.psi_y,
            sh.eps,
            sh.eps,
            sh.W,
            sh.V,
            sh.psi_x,
            sh.psi_y,
        ),
        out_shardings=sh.X,
    )


def build_bse_ring_matvec_full(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    low_mem: bool = True,
    include_W: bool = True,
    screening: bool = False,
):
    """Build full (non-TDA) BSE matvec ``[X;Y] -> H[X;Y]``.

    ``screening`` selects the coupling-block kernel AND the anti-resonant row
    (``_antiresonant_row``), which together fix *which* physical operator this is:

    - ``screening=False`` (default): the OPTICAL BSE, the para-Hermitian
      ``H = [[A, B], [-B*, -A*]]`` (* = complex conjugate) with ``A`` Hermitian
      and ``B`` complex-SYMMETRIC.  The B (coupling) block uses the excitonic
      exchange ``V_B`` (Henneke Eq. 2-20, conjugated pairing
      ``⟨M_t|v|conj(M_t')⟩`` — ``apply_V_ring_B``) plus the c'↔v'-swapped direct
      term.  With ``include_W`` this is the TDHF/BSE exciton Hamiltonian; its
      spectrum is REAL with +-omega pairs (solved by ``bse_nontda``).  NOTE: the
      historical ``-B, -A`` anti-resonant row gave a COMPLEX spectrum for complex
      B and was never value-validated — fixed here (PHASE2_LOG "non-TDA
      eigensolvers").
    - ``screening=True``: the RPA test-charge DENSITY response (the object whose
      resolvent gives the screened Coulomb ``W = v + vχv``).  Here the B block
      uses the SAME RING kernel ``K^A = (1/Nk)⟨M_t|v|M_t'⟩`` as the A block
      (``apply_V_ring``), so ``A+B = D + 2V`` — the standard RPA bubble
      resummation ``χ = χ₀(1 − vχ₀)⁻¹``.  ``include_W`` must be False (RPA
      screening has no screened-W direct term; it IS what builds W).  See
      ``bse_w_exact`` for the identity ``W(0) − v = v(0 − H)⁻¹v`` and its
      per-element convention.
    """
    if screening and include_W:
        raise ValueError("screening=True is the RPA density response; it has no "
                         "screened-W direct term — pass include_W=False.")
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _encode_T(X, psi_c_X, psi_v_Y):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_c_X.shape[-1]
        n_rmu_local_Y = psi_v_Y.shape[-1]
        R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
        T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)
        return T

    encode_T_ring = _shard_map_fn(
        _encode_T,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_gather(X, psi_c_X, psi_v_Y):
        X_full_v = lax.all_gather(X, "y", axis=2, tiled=True)
        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_Y), X_full_v)
        R_full_c = lax.all_gather(R, "x", axis=1, tiled=True)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_full_c)
        return T

    encode_T_gather = _shard_map_fn(
        _encode_T_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B(X, psi_c_Y, psi_v_X):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_v_X.shape[-1]
        n_rmu_local_Y = psi_c_Y.shape[-1]
        R = _ring_sum_conduction_first(X, psi_c_Y, c_chunk, px, n_rmu_local_Y)
        T = _ring_sum_valence_second(R, psi_v_X, v_chunk, py, n_rmu_local_X)
        return T

    encode_T_ring_B = _shard_map_fn(
        _encode_T_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B_gather(X, psi_c_Y, psi_v_X):
        X_full_c = lax.all_gather(X, "x", axis=1, tiled=True)
        R = jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_Y), X_full_c)
        R_full_v = lax.all_gather(R, "y", axis=1, tiled=True)
        T = jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_X, R_full_v)
        return T

    encode_T_gather_B = _shard_map_fn(
        _encode_T_B_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, M_X, V_q0, nk, px, py)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    def _apply_V_ring_B(X, psi_c_Y, psi_v_Y, M_X, V_q0):
        return apply_V_ring(
            X,
            jnp.conj(psi_c_Y),
            jnp.conj(psi_v_Y),
            M_X,
            V_q0,
            nk,
            px,
            py,
        )

    apply_V_ring_B = _shard_map_fn(
        _apply_V_ring_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    # Custom-partitioned FFTs on the (kx, ky, kz) axes — those axes are
    # ``None``-sharded in T (sh.T) and W_R (sh.W), so the FFT can run
    # locally on every device.  Plain ``jnp.fft.ifftn`` / ``fftn`` on a
    # sharded tensor forces XLA to all-gather the entire array before
    # the FFT — see ``common.fft_helpers`` for the JAX bug this works
    # around.  In the BSE Lanczos loop those gathers cost ~5 s over
    # 200 matvecs on Si 4×4×4 (profile_sharded_v2/trace_summary.md).
    # T_k 8D spec: (b, μ, ν, ns, ns, kx, ky, kz) — same μ,ν shardings as
    # storage T (6D) but with last nk axis split into 3 replicated dims.
    _T_8d_spec = P(None, "x", "y", None, None, None, None, None)
    _T_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')
    _T_local_fftn = make_sharded_fftn_3d(
        mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5, 6, 7), norm='ortho')

    # ψ†Uψ decode = contract_bands_block_reshard, extra="leading" (owner
    # order 2026-07-29; adoption map wk_REL/contract_bands_notes.md §6.2 —
    # the CLEAN drop-in site: the b-stacked U already exists, so the stack
    # axis is free).  Replaces the partitioner-chosen collectives of the
    # historical einsum pair ("kctM,bMNtsk->bcNsk" then "kvsN,bcNsk->bcvk",
    # c-replicated intermediate, LARGE payload on the strided 'x' groups —
    # the exact inversion the primitive's §3.2 policy refuses) with the
    # structural stacked psum_scatter chain: large partial over the
    # node-local 'y' groups, small final over 'x', all b trials on ONE
    # collective per mesh axis (AK.9), impl=mpi warm-up inherited from the
    # factory.  Value-level identical (contraction reassociation — gate at
    # 1e-12, not bit-exact).  The transposes below are rank-local
    # (sharded axes preserved: M stays on 'x', N on 'y', c on 'x', v on
    # 'y'); the U transpose to the primitive's k-leading layout is priced
    # by the parity/perf gate, and composes with the future flat-k conv
    # layout (map §6.1 route (a)) which emits k-leading natively.
    _w_decode = contract_bands_block_reshard(
        mesh_xy, extra="leading",
        divisibility_hint=(
            "BSE callers: n_cond_pad / n_val_pad already pad c to p_x and "
            "v to p_y (bse_io loader); an indivisible window here means an "
            "unpadded/hand-built operand."))

    def _apply_W_from_T(T, psi_c_X, psi_v_Y, W_R):
        nspinor = psi_c_X.shape[2]
        nb_trial = T.shape[0]
        n_rmu_local_X = T.shape[1]
        n_rmu_local_Y = T.shape[2]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))

        T_k = T.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nkx, nky, nkz)
        T_R = _T_local_ifftn(T_k)
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = _T_local_fftn(U_R)
        U = U_q.reshape(nb_trial, n_rmu_local_X, n_rmu_local_Y, nspinor, nspinor, nk)

        # (b, M, N, t, s, k) -> (b, k, t, M, s, N): the primitive's
        # canonical O layout (extra="leading"); ψ_v (k, v, s, N) ->
        # (k, s, N, v) = ψ_right.  conj(ψ_c) is applied inside.
        O_b = jnp.transpose(U, (0, 5, 3, 1, 4, 2))
        psi_v_snv = jnp.transpose(psi_v_Y, (0, 2, 3, 1))
        out = _w_decode(psi_c_X, O_b, psi_v_snv)     # (b, nk, c_X, v_Y)
        WX = jnp.transpose(out, (0, 2, 3, 1))        # (b, c_X, v_Y, nk)
        return WX / sqrt_nk

    apply_W_from_T = jax.jit(
        _apply_W_from_T,
        in_shardings=(sh.T, sh.psi_x, sh.psi_y, sh.W),
        out_shardings=sh.X,
        # NB: T (arg 0) is NOT donated — the WX output has a different shape
        # (nt,c,v,k) so the donation is always declined (no aliasable output) and
        # emits no fallback copy. Dropping the cosmetic donate_argnums silences the
        # recurring "donated buffers not usable" warning (audit P5, JOINT_FINDINGS §3).
    )

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X):
        D_term = apply_D_term(X, eps_c, eps_v)
        V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        if not include_W:
            return D_term + V_term
        T = encode_T_ring(X, psi_c_X, psi_v_Y) if low_mem else encode_T_gather(X, psi_c_X, psi_v_Y)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return D_term + V_term - W_term

    def _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X):
        # screening (RPA density response): ring kernel K^A, same as the A block
        # (apply_V_ring_only); optical BSE: excitonic V_B (apply_V_ring_B). Both take
        # the SAME hoisted M_X (audit P3) — apply_V_ring_B conjugates only ψ^Y.
        if screening:
            V_term = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        else:
            V_term = apply_V_ring_B(X, psi_c_Y, psi_v_Y, M_X, V_q0)
        if not include_W:
            return V_term
        T = encode_T_ring_B(X, psi_c_Y, psi_v_X) if low_mem else encode_T_gather_B(X, psi_c_Y, psi_v_X)
        W_term = apply_W_from_T(T, psi_c_X, psi_v_Y, W_R)
        return V_term - W_term

    def _antiresonant_row(X, Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                          W_R, V_q0, M_X):
        """Bottom (anti-resonant) block-row of the non-TDA operator applied to
        ``[X; Y]``: ``Y_out``.  The physics of this row depends on ``screening``:

        * ``screening=True`` (RPA test-charge density response): the coupling
          ``B = K^A`` is Hermitian and ``A`` is Hermitian, so the operator is the
          symplectic ``[[A, B], [-B, -A]]`` and ``Y_out = -B X - A Y``.  Validated
          by the W(0) resolvent closure (PHASE2_LOG "W(0)").

        * ``screening=False`` (OPTICAL BSE): ``A`` is Hermitian (``A = A^H``) but
          the coupling ``B`` is complex-SYMMETRIC (``B = B^T``, NOT Hermitian).
          The physical para-Hermitian Casida operator is ``[[A, B], [-B*, -A*]]``
          (Onida-Reining-Rubio; Rohlfing-Louie), whose spectrum is REAL with
          +-omega pairs, so ``Y_out = -B* X - A* Y``.  ``B* X = conj(B conj(X))``
          and ``A* Y = conj(A conj(Y))`` reuse the SAME appliers on conjugated
          inputs (operator ingredients unchanged), then conjugate the result — no
          new kernel.  The naive ``-B X - A Y`` gives a COMPLEX (unphysical)
          spectrum for complex ``B`` and was the historical, never-value-validated
          bug (PHASE2_LOG "non-TDA eigensolvers", first checked numbers)."""
        if screening:
            AY = _apply_A(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X)
            BX = _apply_B(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X)
            return -BX - AY
        AsY = jnp.conj(_apply_A(jnp.conj(Y), psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                eps_c, eps_v, W_R, V_q0, M_X))
        BsX = jnp.conj(_apply_B(jnp.conj(X), psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                W_R, V_q0, M_X))
        return -BsX - AsY

    def _matvec_impl(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
                     M_X, M_Y):
        # M_X: hoisted decode-side exchange pair amplitude (audit P3), shared by the
        # A and B blocks. M_Y is unused here — kept for a uniform matvec signature.
        X = X_full[0]
        Y = X_full[1]
        AX = _apply_A(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X)
        BY = _apply_B(Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, W_R, V_q0, M_X)
        X_out = AX + BY
        Y_out = _antiresonant_row(X, Y, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                                  eps_c, eps_v, W_R, V_q0, M_X)
        return jnp.stack([X_out, Y_out], axis=0)

    return jax.jit(
        _matvec_impl,
        in_shardings=(
            sh.X_full,
            sh.psi_x,
            sh.psi_y,
            sh.psi_x,
            sh.psi_y,
            sh.eps,
            sh.eps,
            sh.W,
            sh.V,
            sh.psi_x,
            sh.psi_y,
        ),
        out_shardings=sh.X_full,
    )


def build_realspace_random_transition_generator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    n_cond_pad: int,
    n_val_pad: int,
):
    """Build a sharded map: r(nu,k) -> V_q0 @ r -> X(c,v,k) (local c/v chunks).

    Assumes r is column-sharded on the Y axis and V_q0 is tiled (x,y).
    The resulting X is sharded like the BSE matvec (c on x, v on y).
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    if n_cond_pad % px != 0 or n_val_pad % py != 0:
        raise ValueError("n_cond_pad and n_val_pad must be divisible by px/py")

    v_chunk = n_val_pad // py

    def _map(r, psi_c_X, psi_v_X, V_q0):
        # r: (batch, nu_local, nk), column-sharded on y.
        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, r)
        U = lax.psum(U_partial, axis_name="y")  # (batch, mu_local_x, nk), mu on x

        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
        v_start = axis_index_y * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)

        # v is a free output index (tiled on y) -> pre-slice its y-block.  c is
        # NOT pre-sliced: the centroid (mu) contraction below is over the LOCAL
        # x-slice of mu, so it must be completed across x by a psum.  Slicing c to
        # this x-rank's block first (as an earlier version did) silently drops the
        # off-rank mu contributions to every c — correct only at px=1 (~50% wrong
        # at px=2).  Mirror apply_V_ring: full c, psum_scatter over x.
        nk_local, _, nspinor, mu_local = psi_c_X.shape
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk_local, v_chunk, nspinor, mu_local)
        )

        M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_slice)  # (k,c_full,v,mu_x)
        # Decode leg of K^x = M V M† carries the BARE vertex (the conjugate
        # belongs on the encode leg — see build_density_snapshot_operator, the
        # partner half of the v(ω−H)⁻¹v form these two assemble together).
        VX_partial = jnp.einsum("kcvM,bMk->bcvk", M_X, U)                    # local-mu, c full
        # psum over x completes the mu sum across x-slices AND scatters the
        # conduction index onto x -> (b, c_chunk_x, v_chunk_y, k).
        X_local = lax.psum_scatter(
            VX_partial, axis_name="x", scatter_dimension=1, tiled=True)

        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X_local.real.dtype))
        return X_local / sqrt_nk

    _gen = _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "y", None), P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )
    # jit the seed (zeta->pair) reshard boundary.  A bare shard_map re-traces and
    # re-lowers to HLO on EVERY eager call (~2.5 s on the MoS2 fixture) because the
    # trace is not memoized; wrapping it in jax.jit caches the compiled executable
    # so repeated calls dispatch in <1 ms.  The BSE matvec is jitted for exactly
    # this reason — the seed/project boundaries were the only un-jitted ones and
    # dominated the W-column solve (see PHASE2_LOG "W-column resolvent profiling").
    # Bit-faithful: same shard_map body, same in/out shardings.
    return jax.jit(
        _gen,
        in_shardings=(sh.S, sh.psi_x, sh.psi_x, sh.V),
        out_shardings=sh.X,
    )


def build_density_snapshot_operator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    scatter_nu_on_y: bool = False,
):
    """Build a sharded map: s(b,c,v,k) -> d(b,mu) = V_q0 @ sum_k R s.

    The projection back from the (c,v,k) pair basis to the ISDF density
    (centroid) basis.  ``b`` is a batch axis (one probe/trial vector per slot),
    replicated across the mesh; ``mu`` is the centroid output index.

    Two output layouts, selected at build time:

    * ``scatter_nu_on_y=False`` (default): the V_q0 contraction over N is
      completed with a plain ``psum('y')`` and ``d`` returns replicated on the
      batch axis with ``mu`` on ``x`` — ``P(None, 'x')`` = ``(b, mu_X)``.  Used
      by the per-vector density-snapshot callers (``bse_pseudopoles``) that
      device_get one column at a time.

    * ``scatter_nu_on_y=True``: the W-column / screened-W(omega) path, where the
      batch axis IS the probe index ``nu`` and we want the assembled tile
      ``W(mu_X, nu_Y)`` with no replicated ``(mu, nu)`` intermediate.  This
      mirrors the Sigma_PPM reduce-scatter kernel
      (``gw.ppm_tau_kernel._make_project_ri_reduce_scatter``): the same mesh
      axis ``y`` carries BOTH the reduction (finish the V_q0 N-contraction,
      whose partials are scattered across the ``y`` shards) AND the output
      tiling (scatter the ``nu`` batch onto ``y``), fused into ONE
      ``psum_scatter`` — same NCCL volume as the plain ``psum`` but the tile
      lands ``P('x', 'y')`` = ``(mu_X, nu_Y)`` = ``sh.V`` device-local.
      Requires ``b % py == 0`` (pad the probe block; ``padded_mu_extent`` already
      rounds the full-basis centroid count to a multiple of ``px*py``).

    Returns density-space vectors summed over k.
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _map(s, psi_c_Y, psi_v_Y, V_q0):
        nb_trial = s.shape[0]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=s.real.dtype))

        c_chunk = s.shape[1]
        v_chunk = s.shape[2]

        axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

        nk_local = psi_c_Y.shape[0]
        nspinor = psi_c_Y.shape[2]
        nu_local = psi_c_Y.shape[-1]

        c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
        )

        A0 = mark_varying(
            jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=s.dtype),
            ("x", "y"))
        perm_y = _ring_perm(py)

        def step_y(i, carry):
            buf, A = carry
            origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
            v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
            psi_v_slice = lax.dynamic_slice(
                psi_v_Y, (z, v_start, z, z), (nk_local, v_chunk, nspinor, nu_local)
            )
            # Encode leg carries the CONJUGATED vertex conj(M) = ψ_c·conj(ψ_v),
            # matching K^x = M V M† (the decode half lives in
            # build_realspace_random_transition_generator).
            R_v = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
            A = A + jnp.einsum("kcsN,bcksN->bcNk", psi_c_slice, R_v)
            buf = lax.ppermute(buf, axis_name="y", perm=perm_y)
            return buf, A

        _, A_local = lax.fori_loop(0, py, step_y, (s, A0))

        S0 = mark_varying(
            jnp.zeros((nb_trial, nu_local, nk_local), dtype=s.dtype),
            ("x", "y"))
        perm_x = _ring_perm(px)

        def step_x(i, carry):
            buf, S = carry
            S = S + jnp.sum(buf, axis=1)
            buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
            return buf, S

        _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

        S_total = S_total / sqrt_nk

        # V_q0 @ S: local N-slice partials, N tiled on y (mu on x from V_q0).
        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S_total)  # (b, mu_local_x, k)

        if not scatter_nu_on_y:
            U = lax.psum(U_partial, axis_name="y")  # finish N sum, replicate on y
            return jnp.sum(U, axis=-1)              # (b, mu_local_x)

        # Reduce-scatter tail (mirror of ppm_tau_kernel reduce-scatter): sum the
        # local-N partials across y (completes the V_q0 N-contraction) AND scatter
        # the nu batch onto y in one collective, so W lands (mu_X, nu_Y) with no
        # replicated (mu, nu).  Local transpose first (no comms) so nu is the
        # scatter axis; psum_scatter reduces over y and tiles nu across y.
        d_partial = jnp.sum(U_partial, axis=-1)          # (b, mu_local_x)
        d_local = jnp.swapaxes(d_partial, 0, 1)          # (mu_local_x, b)
        return lax.psum_scatter(
            d_local, axis_name="y", scatter_dimension=1, tiled=True
        )                                                # (mu_local_x, b/py)

    _snap = _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"), P("x", "y")),
        out_specs=(P("x", "y") if scatter_nu_on_y else P(None, "x")),
    )
    # jit the projection (pair->zeta) reshard boundary — same rationale as
    # build_realspace_random_transition_generator: cache the compiled executable
    # instead of re-lowering the shard_map on every eager call (~2.8 s -> <3 ms).
    # Bit-faithful.  Output sharding follows the build-time scatter_nu_on_y branch.
    return jax.jit(
        _snap,
        in_shardings=(sh.X, sh.psi_y, sh.psi_y, sh.V),
        out_shardings=(sh.V if scatter_nu_on_y else sh.d_mu),
    )


def build_density_drive_operators(mesh_xy, nkx, nky, nkz, n_cond_pad, n_val_pad):
    """Return callable(s) to build density-driven RHS blocks in transition space.

    Transition vertex: d(t,mu) = sum_s conj(psi_c(t,mu)) * psi_v(t,mu)
    For density-space source g(mu), u = V g. Need f = d^dagger u, fbar = d^T u.
    For complex Bloch spinors, do not assume fbar == conj(f).
    """
    gen = build_realspace_random_transition_generator(mesh_xy, nkx, nky, nkz, n_cond_pad, n_val_pad)

    def f_op(r, psi_c_X, psi_v_X, V_q0):
        return gen(r, psi_c_X, psi_v_X, V_q0)

    def fbar_op(r, psi_c_X, psi_v_X, V_q0):
        return gen(r, jnp.conj(psi_c_X), jnp.conj(psi_v_X), V_q0)

    return f_op, fbar_op


def build_density_readout_operator_full(mesh_xy, nkx, nky, nkz):
    """Return a callable to compute w = V (d X + d* Y) from X_full=[X,Y]."""
    snapshot = build_density_snapshot_operator(mesh_xy, nkx, nky, nkz)

    def _readout(X_full, psi_c_Y, psi_v_Y, V_q0):
        X = X_full[0]
        Y = X_full[1]
        w_x = snapshot(X, psi_c_Y, psi_v_Y, V_q0)
        w_y = snapshot(Y, jnp.conj(psi_c_Y), jnp.conj(psi_v_Y), V_q0)
        return w_x + w_y

    return _readout


def ring_matvec_smoke_test(px: int = 2, py: int = 2) -> None:
    devices = jax.devices()
    if len(devices) < px * py:
        raise RuntimeError(
            f"Need {px*py} devices, found {len(devices)}. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=... before running."
        )
    # ONE mesh factory (create_mesh_xy): these two smoke drivers used to
    # build their own, so they were the last un-warmed constructors left
    # in src/bse/ after the four _create_mesh_xy copies were folded in.
    mesh = create_mesh_xy(px, py, devices)
    sh = make_bse_shardings(mesh)

    nkx, nky, nkz = 2, 2, 1
    nk = nkx * nky * nkz
    nc, nv, nspinor, n_rmu = 4 * px, 4 * py, 2, 8 * px * py

    key = jax.random.PRNGKey(0)
    psi_c = jax.random.normal(key, (nk, nc, nspinor, n_rmu)) + 1j * jax.random.normal(key, (nk, nc, nspinor, n_rmu))
    psi_v = jax.random.normal(key, (nk, nv, nspinor, n_rmu)) + 1j * jax.random.normal(key, (nk, nv, nspinor, n_rmu))
    eps_c = jax.random.uniform(key, (nk, nc), minval=0.1, maxval=0.5)
    eps_v = jax.random.uniform(key, (nk, nv), minval=-0.5, maxval=-0.1)
    W_q = jax.random.normal(key, (n_rmu, n_rmu, nkx, nky, nkz)) * 0.01
    V_q0 = jnp.eye(n_rmu) * 0.05
    X = jax.random.normal(key, (1, nc, nv, nk)) + 1j * jax.random.normal(key, (1, nc, nv, nk))

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v, sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(W_q, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(V_q0, sh.V)
        X = jax.lax.with_sharding_constraint(X, sh.X)

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")
        M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
        M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)
        HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y)
        HX.block_until_ready()

    print(f"HX sharding: {HX.sharding}")
    if hasattr(jax.debug, "inspect_array_sharding"):
        try:
            jax.debug.inspect_array_sharding(HX, name="HX")
        except TypeError:
            try:
                jax.debug.inspect_array_sharding(HX, callback=lambda *_: None)
            except TypeError:
                pass


def ring_matvec_correctness_check(
    input_file: str,
    n_val: int = 4,
    n_cond: int = 4,
    px: int = 2,
    py: int = 2,
    component_check: bool = False,
) -> None:
    restart_file = _find_restart_file(input_file)
    devices = jax.devices()
    if len(devices) < px * py:
        raise RuntimeError(
            f"Need {px*py} devices, found {len(devices)}. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=... before running."
        )
    # ONE mesh factory (create_mesh_xy): these two smoke drivers used to
    # build their own, so they were the last un-warmed constructors left
    # in src/bse/ after the four _create_mesh_xy copies were folded in.
    mesh = create_mesh_xy(px, py, devices)
    sh = make_bse_shardings(mesh)

    payload = _load_ring_subset(restart_file, n_val, n_cond, px, py,
                                input_file=input_file)
    psi_c = payload["psi_c"]
    psi_v = payload["psi_v"]
    eps_c = payload["eps_c"]
    eps_v = payload["eps_v"]
    W_q = payload["W_q"]
    V_q0 = payload["V_q0"]
    X = payload["X"]
    nkx = payload["nkx"]
    nky = payload["nky"]
    nkz = payload["nkz"]
    nk = payload["nk"]
    n_rmu_pad = payload["n_rmu_pad"]

    HX_ref = apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v, sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(W_q, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(V_q0, sh.V)
        X = jax.lax.with_sharding_constraint(X, sh.X)
        W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")

        matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
        M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)
        HX_ring = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y)
        HX_ring.block_until_ready()

    # ``gather_to_host``: HX_ring carries ``out_shardings=sh.X`` =
    # P(None,'x','y',None), tiled on BOTH mesh axes, so a bare ``device_get``
    # raised at P>1 -- i.e. the ring-matvec correctness check could not be run
    # on the geometry whose correctness it exists to check.
    HX_ring_host = gather_to_host(HX_ring)
    diff = jnp.linalg.norm(HX_ring_host - HX_ref) / jnp.maximum(jnp.linalg.norm(HX_ref), 1e-12)
    print(f"Relative error ||HX_ring - HX_ref||/||HX_ref||: {float(diff):.3e}")

    if component_check:
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))
        M = compute_pair_amplitude(psi_c, psi_v)
        D_ref = apply_D(X, eps_c, eps_v)
        S_V = jnp.einsum("kcvN,bcvk->bNk", jnp.conj(M), X) / sqrt_nk
        U_V = jnp.einsum("MN,bNk->bMk", V_q0, S_V)
        V_ref = jnp.einsum("kcvM,bMk->bcvk", M, U_V) / sqrt_nk

        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v), X)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c, R)
        T_k = T.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nkx, nky, nkz)
        T_R = local_ifftn3(T_k, axes=(5, 6, 7), norm="ortho")
        W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")
        U_R = W_R[None, :, :, None, None, :, :, :] * T_R
        U_q = local_fftn3(U_R, axes=(5, 6, 7), norm="ortho")
        U = U_q.reshape(X.shape[0], n_rmu_pad, n_rmu_pad, psi_c.shape[2], psi_c.shape[2], nk)
        A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c), U)
        W_ref = jnp.einsum("kvsN,bcNsk->bcvk", psi_v, A) / sqrt_nk

        comp_matvec = build_bse_ring_matvec(mesh, nkx, nky, nkz)
        with mesh:
            D_ring = apply_D(X, eps_c, eps_v)
            W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")
            V_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R * 0.0, V_q0,
                M_X, M_Y
            )
            W_ring = comp_matvec(
                X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c * 0.0, eps_v * 0.0, W_R, V_q0 * 0.0,
                M_X, M_Y
            )
            D_ring.block_until_ready()
            V_ring.block_until_ready()
            W_ring.block_until_ready()

        def _rel_err(a, b):
            return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-12))

        print(f"Component error D: { _rel_err(D_ring, D_ref):.3e}")
        print(f"Component error V: { _rel_err(V_ring, V_ref):.3e}")
        print(f"Component error W: { _rel_err(-W_ring, W_ref):.3e}")


