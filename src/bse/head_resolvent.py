"""Macroscopic head on the same physical R/A ladder operator as the body.

The velocity dipole d=v_cv/Delta=i*r_cv gives RHS [d;conj(d)] and
readout [d^dag,-d^T]. The common density-vertex sign cancels between
injection and readout. The prefactor is 2/(Omega Nk nspin nspinor), so
the free-pair tensor agrees with compute_S_omega_cv. The engine returns
the micro-reducible tensor only; its consumer restores the macroscopic
Coulomb head once. No TRS or frequency averaging is applied to a result.

The unchanged default folded-RPA control keeps its original [d;-d]
source and branch-sum readout. Operator identity includes the branch
representation; a physical zero-W payload tests the full RPA identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap().
# MUST run before this module's own `import jax` (same rule as bse_w_exact).
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .bse_feast import ladder_matvec_operands, matvec_operands
from .bse_w_exact import _get_block_gmres_solver, _get_block_moment_actions
from .w_ladder import build_ladder_resolvent
from common.collectives import device_put_process_local, gather_to_host

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Operator identity
# ---------------------------------------------------------------------------
#: ``structural key -> matvec``.  TWO jobs, both of them defects this
#: project has already paid for:
#:
#: 1. ``bse_w_exact._get_block_gmres_solver`` keys its compiled engine on
#:    ``id(matvec)`` (``bse_feast.py:49-56`` says the same of the FEAST/GMRES
#:    caches).  A fresh ``build_ladder_resolvent`` per call therefore costs one
#:    full XLA compile and one permanent cache entry per call — the
#:    ``_operator_key`` measurement in ``common.mtxel_sweep`` (2.394 s vs
#:    2.167 s per iteration, one live executable leaked each time).  Caching on
#:    a STRUCTURAL key is the fix, and it is why ``matvec`` is q-independent by
#:    design (``build_ladder_resolvent``'s docstring).
#: 2. The converse hazard, which is the one that produces WRONG NUMBERS rather
#:    than slow ones: two operators that hash the same share a compiled program,
#:    so the second arm of an A/B silently re-runs the first.
#:    ``mtxel_sweep.dipole_operator`` once had to put its (since deleted)
#:    velocity sign into ``_operator_key`` for exactly this reason ("the defect
#:    class this project has already paid for twice").  :func:`assert_distinct_operators` is the
#:    caller-facing assertion; the registry below is the automatic half.
#:
#: ONLY THE MATVEC IS CACHED, and that is the whole point of the split.  The
#: other two things ``build_ladder_resolvent`` returns are PAYLOAD-dependent and
#: caching them would be a correctness bug, not a speed one:
#:
#:   * ``diag_h`` is ``diag(H)`` — a function of ``eps``, ``W_q`` and ``V_q0``.
#:     Reusing one payload's preconditioner on another's operands preconditions
#:     the wrong operator, and the A/B this engine exists to run (Test 1's
#:     zeroed kernel against the production one) is exactly two payloads on one
#:     structure.
#:   * ``W_R`` — ``ensure_W_R`` MUTATES the payload dict, and a cache hit that
#:     skipped it left the second payload without the matvec's 7th operand.
#:
#: So :func:`build_head_operator` calls ``build_ladder_resolvent`` on EVERY call
#: (it prepares the payload and builds that payload's ``diag_h``) and keeps the
#: cached ``matvec`` object in place of the freshly built one.  The value pins
#: the matvec's ``id()`` for the life of the process — the same lifetime trick
#: ``_GMRES_SOLVER_CACHE`` uses, and what makes the id-collision check below a
#: real check rather than a coincidence.
_HEAD_MATVEC_CACHE: dict[tuple, Callable] = {}
#: ``id(matvec) -> structural key``, so a collision is a raise, not a surprise.
_HEAD_OPERATOR_IDS: dict[int, tuple] = {}


@dataclass(frozen=True)
class HeadOperator:
    """The ladder head operator plus everything the solve needs to run it.

    ``key`` is the STRUCTURAL identity (see :func:`build_head_operator`): what
    decides the compiled program.  It deliberately does NOT record the operand
    arm — zeroing ``V_q0`` / ``W_q`` to reach guide Test 1's ``K^d = K^x = 0``
    changes VALUES that ride in as runtime arguments, not the program, so those
    arms SHOULD share one executable.  The arm label belongs on the result
    (:class:`HeadTensorResult.arm`), not here.

    ``diag_h`` is the ONE payload-dependent field, and it is here rather than in
    the structural cache for the reason ``_HEAD_MATVEC_CACHE`` states: it is
    ``diag(H)``, so one instance of this class belongs to ONE payload even
    though several instances may share a ``matvec``.
    """
    matvec: Callable
    diag_h: jax.Array
    sh: object
    operands_fn: Callable
    key: tuple

    @property
    def matvec_id(self) -> int:
        return id(self.matvec)


def head_operator_key(mesh_xy: Mesh, data: dict, *, include_w: bool,
                      fuse_ladder_rung: bool) -> tuple:
    """Structural identity of the head operator — everything that is baked into
    the compiled program, and nothing that is not.

    ``include_w`` selects the direct rung. Without that rung, the historical
    folded RPA builder is used unless ``physical_branches`` explicitly selects
    the physical R/A control. The
    k-grid and the pair-window pads set the shapes, the complete mesh identity
    (axes and device assignment) sets the sharding,
    and ``fuse_ladder_rung`` selects ``bse_ring_comm._impl_core_fused`` — a cost
    switch, but still a different program.  ``V_q0`` / ``W_q`` VALUES are absent
    on purpose: they are runtime operands.
    """
    from distrib_la import mesh_key
    from .bse_preconditioner import exchange_spin_weight

    return (
        "bse.head_resolvent",
        bool(include_w),
        bool(include_w or data.get("physical_branches", False)),
        bool(fuse_ladder_rung),
        ("charge", exchange_spin_weight(int(data["psi_c_X"].shape[2]), "singlet")),
        (int(data["nkx"]), int(data["nky"]), int(data["nkz"])),
        (int(data["n_cond_pad"]), int(data["n_val_pad"])),
        mesh_key(mesh_xy),
    )


def build_head_operator(mesh_xy: Mesh, data: dict, *, include_w: bool = True,
                        fuse_ladder_rung: bool = True) -> HeadOperator:
    """Assemble (or reuse) the head resolvent operator for ``data``.

    Thin over :func:`w_ladder.build_ladder_resolvent` — the SAME operator the
    body ``W^BSE`` rides, which is the whole convention ruling (module
    docstring).  ``gen`` / ``snapshot`` (the zeta<->pair reshard boundaries) are
    dropped: the head seeds from a dipole in the pair basis and reads out a 3x3
    scalar, so neither boundary is crossed.

    ``include_w=False`` is the RPA-ladder arm (``K^d = 0``, ``K^x`` retained) —
    guide Test 2's operator, and structurally a DIFFERENT program, so it gets
    its own cache entry and its own ``matvec`` object.  Reaching guide Test 1
    (``K^d = K^x = 0``) is done instead by zeroing the OPERANDS ``W_q`` and
    ``V_q0`` (:func:`zero_kernel_payload`), which keeps the production operator
    structure under test and shares its executable — see
    :class:`HeadOperator.key`.

    Mutates ``data`` only through ``build_ladder_resolvent``'s ``ensure_W_R``,
    exactly as the body path does — and does so on EVERY call, because that
    mutation is what installs the matvec's 7th operand on THIS payload.  Only
    the ``matvec`` object survives between calls (``_HEAD_MATVEC_CACHE``); the
    preconditioner is rebuilt from this payload every time.
    """
    key = head_operator_key(mesh_xy, data, include_w=include_w,
                            fuse_ladder_rung=fuse_ladder_rung)
    fresh_matvec, diag_h, _gen, _snapshot, sh = build_ladder_resolvent(
        mesh_xy, data, include_w=include_w, fuse_ladder_rung=fuse_ladder_rung)
    # Keep the FIRST matvec seen for this structure and drop the fresh closure.
    # Building it is free (no tracing happens until it is called); what is not
    # free is a second id(), which would mean a second XLA compile and a second
    # permanent _BLOCK_GMRES_CACHE entry for the identical program.
    matvec = _HEAD_MATVEC_CACHE.setdefault(key, fresh_matvec)
    op = HeadOperator(
        matvec=matvec, diag_h=diag_h, sh=sh,
        # The ladder rung consumes four extra PHYSICAL psi operands; the RPA
        # arm has no rung.  Mismatching these is claim 0215's defect
        # (build_ladder_resolvent's docstring), so the choice rides WITH the
        # operator instead of being a call-site argument.
        operands_fn=partial(ladder_matvec_operands if data.get("physical_branches", False)
                            else matvec_operands, include_W=include_w),
        key=key,
    )

    prior = _HEAD_OPERATOR_IDS.get(op.matvec_id)
    if prior is not None and prior != key:
        raise RuntimeError(
            f"head operator identity collision: matvec id={op.matvec_id} is "
            f"already registered under key {prior!r} but is being registered "
            f"under {key!r}.  The block-GMRES engine is cached on id(matvec) "
            f"(bse_w_exact._get_block_gmres_solver), so two structurally "
            f"different operators sharing one id means the second arm of an "
            f"A/B silently re-runs the first.  This should be unreachable — "
            f"the cache above pins every registered matvec — so treat it as a "
            f"lifetime bug, not a tolerance.")
    _HEAD_OPERATOR_IDS[op.matvec_id] = key
    return op


def assert_distinct_operators(*ops: HeadOperator) -> None:
    """Refuse an A/B whose arms are the same compiled program.

    The caller-facing half of the identity rule (see ``_HEAD_MATVEC_CACHE``):
    arms that declare different structural keys must carry different ``matvec``
    objects, and arms that share a ``matvec`` must declare the same key.  Both
    directions are checked because both have been wrong in this tree.

    An A/B that differs only in OPERANDS (Test 1's zeroed ``V_q0`` / ``W_q``, a
    different ``z``, a different q) is a legitimate single-key A/B and must NOT
    be passed here as two ops — it is one operator run twice on purpose.
    """
    by_id: dict[int, tuple] = {}
    by_key: dict[tuple, int] = {}
    for op in ops:
        prior = by_id.get(op.matvec_id)
        if prior is not None and prior != op.key:
            raise AssertionError(
                f"two head operator arms share matvec id={op.matvec_id} but "
                f"declare different structural keys: {prior!r} vs {op.key!r}. "
                f"The GMRES engine is cached on id(matvec), so one of these "
                f"arms would run the other's compiled program.")
        prior_id = by_key.get(op.key)
        if prior_id is not None and prior_id != op.matvec_id:
            raise AssertionError(
                f"key {op.key!r} appears with two matvec objects "
                f"(id={prior_id} and id={op.matvec_id}).  build_head_operator "
                f"caches the matvec on the key, so this means one arm bypassed "
                f"it — and "
                f"each bypass costs a full XLA compile plus a permanent "
                f"_BLOCK_GMRES_CACHE entry for the life of the process.")
        by_id[op.matvec_id] = op.key
        by_key[op.key] = op.matvec_id


def zero_kernel_payload(data: dict) -> dict:
    """A shallow copy of ``data`` with ``V_q0`` and ``W_q`` zeroed — guide Test
    1's ``K^d = K^x = 0``, so ``H_bar = Delta``.

    OPERANDS, not structure: the returned payload runs on the SAME operator and
    the SAME compiled engine as the production one, which is the point.  Zeroing
    a runtime array is the only way to test the production kernel chain at
    ``K = 0``; building a smaller operator instead would test a different
    program.

    ``W_R`` is dropped so ``ensure_W_R`` rebuilds it from the zeroed ``W_q``
    (the real-space rung kernel is an ifft of ``W_q``, and a stale ``W_R`` would
    leave the rung live while ``W_q`` reads zero).
    """
    out = dict(data)
    out["q0_head_ry"] = 0.0
    out["V_q0"] = jnp.zeros_like(data["V_q0"])
    if data.get("W_q") is not None:
        out["W_q"] = jnp.zeros_like(data["W_q"])
        out.pop("W_R", None)
        out.pop("_W_q0_for_precond", None)
    elif data.get("W_R") is not None:
        # An explicitly consumed kernel has no reciprocal snapshot. Zero
        # its real-space operand and saved diagonal slice without a reload.
        out["W_R"] = jnp.zeros_like(data["W_R"])
        if data.get("_W_q0_for_precond") is not None:
            out["_W_q0_for_precond"] = jnp.zeros_like(data["_W_q0_for_precond"])
    return out


# ---------------------------------------------------------------------------
# Dipole seed
# ---------------------------------------------------------------------------
def build_head_dipole(v_cvk, delta_cvk, *, n_cond_pad=None, n_val_pad=None):
    """``d^i_t = v^i_t / Delta_t`` in the BSE pair layout ``(3, c, v, k)``.

    ``v_cvk`` is ``(3, nk, nc, nv)`` — ``<c k| v_hat_i |v k>`` in the SAME band
    windows and orderings the payload's ``eps_c`` / ``eps_v`` use.  ``delta_cvk``
    is ``(nk, nc, nv)``, TAKEN from the caller: guide section 7A wants QP
    ``Delta``, ``dipole.h5`` carries DFT ``Delta``, and this engine is correct
    under either as long as the caller says which (module docstring).

    Reuses ``absorption_common.build_dipole_vector_bse`` for the pad-and-
    transpose so there is one spelling of the ``(3, nk, nc, nv) ->
    (3, nc_pad, nv_pad, nk)`` reshape in the tree, not two.  Returns a numpy
    array (host); :func:`build_head_rhs` puts it on device.
    """
    from .absorption_common import build_dipole_vector_bse

    v = np.asarray(v_cvk, dtype=np.complex128)
    dE = np.asarray(delta_cvk, dtype=np.float64)
    if v.ndim != 4 or v.shape[0] != 3:
        raise ValueError(
            f"build_head_dipole: v_cvk must be (3, nk, nc, nv); got {v.shape}")
    if dE.shape != v.shape[1:]:
        raise ValueError(
            f"build_head_dipole: delta_cvk must be {v.shape[1:]} to match "
            f"v_cvk; got {dE.shape}")
    if not np.all(dE > 0.0):
        raise ValueError(
            "build_head_dipole: delta_cvk has a non-positive entry.  d = v/Delta "
            "diverges there and the head tensor would be meaningless; the "
            "transition window must be a genuine (occupied -> empty) one.  "
            "Metals are refused upstream (gw_config / screening_bse) precisely "
            "so this cannot be reached in production.")
    return build_dipole_vector_bse(v / dE[None], n_cond_pad=n_cond_pad,
                                   n_val_pad=n_val_pad)


def payload_delta(data: dict) -> np.ndarray:
    """The operator's OWN transition energies ``eps_c - eps_v`` in the pair
    layout ``(c, v, k)`` — the diagonal ``Delta`` that ``H_bar`` actually
    carries, on the host.

    Used only to report :attr:`HeadTensorResult.delta_mismatch`; the ``Delta``
    that builds ``d`` stays the caller's (module docstring).
    """
    eps_c = np.asarray(jax.device_get(data["eps_c"]))     # (k, c)
    eps_v = np.asarray(jax.device_get(data["eps_v"]))     # (k, v)
    return np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))


def build_head_rhs(d_pair, sh, *, physical_branches=False):
    """Head seed on ``sh.X_full`` = ``(2, 3, c, v, k)``.

    Physical R/A uses [d;conj(d)]; the folded RPA control uses [d;-d].

    Structurally the same object ``bse_w_exact.build_probe_rhs`` returns at its
    line 783 (``jnp.stack([f, -f], axis=0)``) with the mu-probe density vertex
    ``f`` replaced by the dipole ``d``.  The antiresonant sign is NOT a choice
    made here — it is the seed the body's screening resolvent already uses, and
    substituting the seed is the entire difference between the two.

    ``d_pair`` is ``(3, c, v, k)`` complex (:func:`build_head_dipole`).  Unlike
    the mu-probe path this seed is COMPLEX by nature (``<c|v_hat|v>`` is), so
    ``build_probe_rhs``'s real-only refusal does not apply and cannot be reused:
    that refusal exists because its ``gen`` shard_map is float64 to the vertex,
    and the dipole crosses no such boundary.

    The probe axis is NOT required to be a multiple of ``py`` here (it is in
    ``build_probe_rhs``, whose stage-3 ``snapshot`` reduce-scatters ``nu`` onto
    ``y``).  This engine has no stage 3 — three Cartesian columns stay
    replicated through the solve and are contracted to a 3x3 scalar.
    """
    d = np.asarray(d_pair, dtype=np.complex128)
    if d.ndim != 4 or d.shape[0] != 3:
        raise ValueError(
            f"build_head_rhs: d_pair must be (3, c, v, k); got {d.shape}")
    dev = device_put_process_local(np.ascontiguousarray(d), sh.X)
    return jax.lax.with_sharding_constraint(
        jnp.stack([dev, jnp.conj(dev) if physical_branches else -dev],
                  axis=0).astype(jnp.complex128), sh.X_full)


def project_head_response(d_pair, branches, *, pref, physical_branches=True):
    """Matched 3x3 velocity readout; physical branches or folded-RPA sum."""
    d = jnp.asarray(d_pair)
    if not physical_branches:
        return pref * jnp.einsum("icvk,jcvk->ij", jnp.conj(d), branches, optimize=True)
    return pref * (jnp.einsum("icvk,jcvk->ij", jnp.conj(d), branches[0], optimize=True)
                   - jnp.einsum("icvk,jcvk->ij", d, branches[1], optimize=True))


def head_prefactor(cell_volume: float, nk_tot: int, nspin: int,
                   nspinor: int) -> float:
    """``2 / (Omega N_k n_spin n_spinor)`` — ``compute_S_omega``'s ``pref / 2``.

    THE ONLY FACTOR OF TWO IN THIS MODULE, and it is a halving, not a doubling:
    the resonant+antiresonant reconstruction that ``compute_S_omega`` writes
    algebraically as ``1/[Delta (w^2 - Delta^2)]`` is produced HERE by the
    ``[d; -d]`` seed and the ``X + Y`` readout, which deliver
    ``2/[Delta (z^2 - Delta^2)]``.  Guide section 4 forbids hardcoding a factor
    of two "independently of the body implementation"; this one is derived from
    the body's own seed and pinned by Test 1 against ``compute_S_omega``.

    ``compute_S_omega`` also carries ``(f_v - f_c)``, which is 1 for every
    transition in an insulator's occupied->empty window — the only case
    ``w_bse`` accepts (refused at ``gw_config`` parse and ``screening_bse``
    runtime).
    """
    return 2.0 / (float(cell_volume) * float(nk_tot)
                  * float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadTensorResult:
    """``Xi`` plus every number needed to judge whether it means anything.

    ``xi`` — ``(n_z, 3, 3)`` complex, ``Xi_ij(z) = pref * conj(d_i).(X_j + Y_j)``,
    in the ``chi_from_dipole`` convention: ``chi_00(q->0, z) = q_i Xi_ij q_j``
    with ``q`` Cartesian 1/bohr.

    ``resids`` / ``iters`` — ``(n_z, 3)``.  The TRUE relative residual
    ``||b - (z-H)x|| / ||b||`` recomputed after the solve (not the projected
    Krylov estimate) and the ``while_loop`` exit index.  ``iters == max_iter``
    means TRUNCATED at the cap, not converged: the engine reports it and the
    caller judges, because "it converged" is exactly the claim a frequency near
    an eigenvalue of ``H_bar`` invalidates.

    ``asym`` / ``anti_herm`` — ``(n_z,)``, ``|Xi - Xi^T|/|Xi|`` and
    ``|Xi - Xi^dag|/|Xi|`` (Frobenius).  MEASURED, never assumed; see the
    symmetry ruling in the module docstring for what each one licenses.

    ``delta_mismatch`` — scalar, ``max |Delta_d - Delta_H| / max|Delta_H|``
    between the ``Delta`` that built ``d`` and the operator's own diagonal, or
    ``nan`` when the caller did not say which ``Delta`` built ``d``.  It is
    ``nan`` and not ``0`` in that case ON PURPOSE: ``d = v/Delta`` does not
    determine ``Delta``, so a solve that was never told cannot report agreement,
    and a diagnostic that defaults to "fine" is worse than no diagnostic.
    ~0 means the dipole and the Hamiltonian agree on the transition energies
    (what Test 1 needs); a finite number is the DFT-d-against-QP-operator state
    the guide's section 7A migration exists to close.
    """
    xi: np.ndarray
    z: np.ndarray
    resids: np.ndarray
    iters: np.ndarray
    asym: np.ndarray
    anti_herm: np.ndarray
    delta_mismatch: float
    arm: str
    operator_key: tuple
    pref: float
    #: Optional exact even-s derivative/reflection, from the shared response actions.
    d_xi_ds: np.ndarray | None = None
    xi_minus_z: np.ndarray | None = None
    #: Optional C H**j B / 2 tensors, j=0..3; these are high-frequency moments.
    moments: dict | None = None


def _rel_frob(a, b):
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a)) / den if den > 0.0 else 0.0


def solve_head_tensor(op: HeadOperator, data: dict, d_pair, z_values, *,
                      pref: float, max_iter: int = 200, tol: float = 1e-10,
                      rhs=None, arm: str = "", delta_cvk=None,
                      resid_relative_to: str = "b", with_slope=False,
                      with_moments=False) -> HeadTensorResult:
    """``Xi_ij(z)`` for every ``z`` in ``z_values``, one block-GMRES per ``z``.

    Rides ``bse_w_exact._get_block_gmres_solver`` verbatim — the body's stage-2
    SOLVE — so the head and the body are the same Krylov engine on the same
    operator with the same preconditioner, and a head that converges differently
    from the body is a statement about the seed, not about two solvers.  The
    engine compiles ONCE per ``(id(matvec), max_iter, resid_relative_to,
    dtype)`` and every later ``z`` is dispatch-only; ``z`` is passed as a device
    scalar so it stays a runtime argument.

    ``rhs`` may be hoisted across the ``z`` sweep (it is ``z``-independent),
    which is the same hoist ``build_probe_rhs`` exists to allow on the body
    path.  ``delta_cvk`` is the ``(nk, nc, nv)`` ``Delta`` that BUILT ``d``;
    supplying it turns ``delta_mismatch`` from ``nan`` into a measurement.

    ``with_slope`` uses the body's exact resolvent actions, including R**3
    at zero; residuals and iteration counts then cover every extra solve.
    ``with_moments`` evaluates the common H-power actions once, not per z.
    Both default off and preserve existing value-only callers.

    NO KRYLOV RECYCLING, deliberately: guide section 7D says "correctness does
    not depend on it".  Three right-hand sides and a handful of frequencies do
    not pay for a block-Krylov implementation, and the per-column scan
    (``unroll=1``) is what keeps ONE Krylov workspace alive at a time.
    """
    sh = op.sh
    d = np.asarray(d_pair, dtype=np.complex128)
    physical = bool(data.get("physical_branches", False))
    if rhs is None:
        rhs = build_head_rhs(d, sh, physical_branches=physical)

    solver = _get_block_gmres_solver(op.matvec, sh, max_iter, tol, rhs.dtype,
                                     resid_relative_to, with_slope=bool(with_slope),
                                     retain_branches=physical)
    operands = op.operands_fn(data)

    z_arr = np.atleast_1d(np.asarray(z_values, dtype=np.complex128))
    xi = np.zeros((z_arr.size, 3, 3), dtype=np.complex128)
    resids = np.zeros((z_arr.size, 3), dtype=np.float64)
    iters = np.zeros((z_arr.size, 3), dtype=np.int64)
    slopes = np.zeros_like(xi) if with_slope else None
    reflected = np.zeros_like(xi) if with_slope else None

    for iz, z in enumerate(z_arr):
        solved = solver(rhs, op.diag_h,
                        jnp.asarray(z, dtype=jnp.complex128), operands)
        s_all, res, it = solved[:3]
        # Reuse the uploaded, pair-sharded d in the RHS; do not replicate a
        # host dipole constant or upload it again at every frequency.
        xi_z = project_head_response(rhs[0], s_all, pref=float(pref),
                                     physical_branches=physical)
        xi[iz] = np.asarray(gather_to_host(xi_z))
        resids[iz] = np.asarray(gather_to_host(res))[:3]
        iters[iz] = np.asarray(gather_to_host(it))[:3]
        if with_slope:
            for target, name in ((slopes, "dWc_ds"), (reflected, "Wc_minus_z")):
                target[iz] = np.asarray(gather_to_host(project_head_response(
                    rhs[0], solved[3][name], pref=float(pref), physical_branches=physical)))

    moments = None
    if with_moments:
        actions = _get_block_moment_actions(op.matvec, sh, rhs.dtype,
                                            retain_branches=physical)(rhs, operands)
        moments = {name: np.asarray(gather_to_host(project_head_response(
            rhs[0], value, pref=float(pref), physical_branches=physical)))
                   for name, value in actions.items()}

    asym = np.array([_rel_frob(x - x.T, x) for x in xi])
    anti_herm = np.array([_rel_frob(x - x.conj().T, x) for x in xi])

    mismatch = (float("nan") if delta_cvk is None
                else head_delta_mismatch(delta_cvk, data))

    return HeadTensorResult(
        xi=xi, z=z_arr, resids=resids, iters=iters, asym=asym,
        anti_herm=anti_herm, delta_mismatch=mismatch, arm=arm,
        operator_key=op.key, pref=float(pref),
        d_xi_ds=slopes, xi_minus_z=reflected, moments=moments,
    )


def head_delta_mismatch(delta_d, data: dict) -> float:
    """``max |Delta_d - (eps_c - eps_v)| / max|eps_c - eps_v|``.

    The honest form of :attr:`HeadTensorResult.delta_mismatch`: hand it the
    ``(nk, nc, nv)`` ``Delta`` that actually built ``d`` and it compares against
    the operator's diagonal.  Guide Test 1 is only meaningful at ~0; the
    DFT-dipole-against-QP-operator production state (guide section 7A, owned by
    another stream) shows up here as a finite number instead of as a silent
    O(QP-shift/gap) error in the head.
    """
    dH = payload_delta(data)                                  # (c, v, k)
    dd = np.transpose(np.asarray(delta_d, dtype=np.float64), (1, 2, 0))
    if dd.shape != dH.shape:
        raise ValueError(
            f"head_delta_mismatch: delta_d transposes to {dd.shape}, the "
            f"payload's Delta is {dH.shape}; the dipole and the operator are "
            f"not on the same (c, v, k) window.")
    scale = float(np.max(np.abs(dH))) or 1.0
    return float(np.max(np.abs(dd - dH))) / scale


# ---------------------------------------------------------------------------
# Reference (the Test 1 oracle's own algebra, for reporting only)
# ---------------------------------------------------------------------------
def independent_particle_head(d_pair, delta_pair, z_values, pref: float):
    """``Xi^0_ij(z) = pref * sum_t conj(d_i) d_j [1/(z-D) - 1/(z+D)]`` on host.

    NOT the Test 1 gate — ``common.chi_from_dipole.compute_S_omega`` is, because
    it is the tree's INDEPENDENT implementation and this one is the engine's own
    algebra written out.  Kept for the conditioning report, where the useful
    number is how far the GMRES answer drifts from the exact ``K = 0`` value as
    ``z`` approaches an eigenvalue of ``H_bar`` — a comparison that needs the
    exact value at the same ``z``, in the same convention, with no h5 and no
    occupation bookkeeping.

    ``d_pair`` ``(3, c, v, k)``, ``delta_pair`` ``(c, v, k)``.
    """
    d = np.asarray(d_pair, dtype=np.complex128)
    D = np.asarray(delta_pair, dtype=np.float64)
    z_arr = np.atleast_1d(np.asarray(z_values, dtype=np.complex128))
    out = np.zeros((z_arr.size, 3, 3), dtype=np.complex128)
    for iz, z in enumerate(z_arr):
        w = 1.0 / (z - D) - 1.0 / (z + D)
        out[iz] = pref * np.einsum("icvk,cvk,jcvk->ij", np.conj(d), w, d,
                                   optimize=True)
    return out


__all__ = [
    "HeadOperator",
    "HeadTensorResult",
    "assert_distinct_operators",
    "build_head_dipole",
    "build_head_operator",
    "build_head_rhs",
    "head_delta_mismatch",
    "head_operator_key",
    "head_prefactor",
    "independent_particle_head",
    "payload_delta",
    "solve_head_tensor",
    "zero_kernel_payload",
]
