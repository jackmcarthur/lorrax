"""Gates for the ONE ``diag(H_BSE)``: FEAST's builder is now an adapter.

``bse_feast.build_preconditioner_diagonal_sharded`` used to be a second,
independent implementation of the resonant object
``bse_davidson_helpers.build_bse_exact_diagonal`` assembles — eight un-jitted
einsums that arrived independently at the same ``+1/nk`` exchange / ``−1/nk``
direct normalisation, rebuilt ``M_X``/``M_Y`` from ``psi`` instead of reading the
payload's hoisted pair amplitudes, and paid eight program constructions per call
(PRECOND_BUILD_FREE.md §7.1).  The reference below retains that independent
resonant arithmetic, then applies the separately gated non-TDA row convention
``[diag_h, -conj(diag_h)]``.

Measured on the Si 4x4x4 record deck at P=4 BEFORE they were merged, the two
agreed to ``max|Δ| = 1.11e-16 Ry = 1.5e-15 eV`` on a 0.62 Ry signal — the same
round-off the canonical builder already scores against the dense operator, so
neither resonant construction was wrong (FIX_construction_defects.md §2).
This file is what keeps them from drifting apart again: the old resonant
implementation is transcribed here as the reference, so any future edit to
either side has to explain itself against the arithmetic that was shipped.

Two of the cells are RED TWINS.  ``test_adapter_reads_the_payloads_M_X`` fails
if the adapter goes back to rebuilding the pair amplitudes locally; it passes
trivially — with a zero difference — for an implementation that ignores the
payload, which is exactly the failure it exists to catch.
``test_adapter_honours_the_stashed_W_q0`` does the same for the q=0 slice the
driver stashes before its donated ifft consumes ``W_q``.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import harness  # noqa: F401  (puts src/ on sys.path)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_feast as BF  # noqa: E402
from bse import bse_davidson_helpers as H  # noqa: E402
from bse.bse_preconditioner import energy_diff_cv_k  # noqa: E402


NKX = NKY = NKZ = 2
NK = NKX * NKY * NKZ
NC, NV, NMU = 2, 2, 5

# Round-off only.  The two forms differ in einsum association, so bit-identity
# is not the contract; agreement at the scale of the diagonal's own accuracy
# against the dense operator (1.5e-15 eV = 1.1e-16 Ry) is.
ATOL_RY = 1e-13


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))


def _payload(seed=20260808, w_scale=1.0):
    rng = np.random.default_rng(seed)

    def cx(*shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape),
                           dtype=jnp.complex128)

    psi_c_X = cx(NK, NC, 2, NMU)
    psi_c_Y = cx(NK, NC, 2, NMU)
    psi_v_X = cx(NK, NV, 2, NMU)
    psi_v_Y = cx(NK, NV, 2, NMU)
    return {
        "nkx": NKX, "nky": NKY, "nkz": NKZ,
        "eps_c": jnp.asarray(rng.standard_normal((NK, NC)) + 3.0),
        "eps_v": jnp.asarray(rng.standard_normal((NK, NV))),
        "psi_c_X": psi_c_X, "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X, "psi_v_Y": psi_v_Y,
        # exactly what the loader hoists (audit P3)
        "M_X": jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_X),
        "M_Y": jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_Y), psi_v_Y),
        "V_q0": cx(NMU, NMU),
        "W_q": jnp.asarray(w_scale) * cx(NMU, NMU, NKX, NKY, NKZ),
    }


def _feast_diag_reference(data, mesh_xy, include_W=True, use_tda=True):
    """Independent pre-consolidation resonant arithmetic plus the gated row."""
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    psi_c_X = data["psi_c_X"]
    psi_v_X = data["psi_v_X"]
    psi_c_Y = data["psi_c_Y"]
    psi_v_Y = data["psi_v_Y"]
    M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_X)
    M_Y = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_Y), psi_v_Y)
    V_q0 = data["V_q0"]
    S_v = jnp.einsum("MN,kcvN->kcvM", V_q0, jnp.conj(M_Y))
    V_diag_kcv = jnp.einsum("kcvM,kcvM->kcv", M_X, S_v) / nk
    if include_W:
        W_q0 = data["W_q"][:, :, 0, 0, 0]
        rho_c = jnp.einsum("kcsm,kcsm->kcm", jnp.conj(psi_c_X), psi_c_X)
        rho_v = jnp.einsum("kvsm,kvsm->kvm", jnp.conj(psi_v_Y), psi_v_Y)
        S_w = jnp.einsum("MN,kvN->kvM", W_q0, rho_v)
        W_diag_kcv = jnp.einsum("kcm,kvm->kcv", rho_c, S_w) / nk
    else:
        W_diag_kcv = jnp.zeros_like(V_diag_kcv)
    V_diag = V_diag_kcv.transpose(1, 2, 0)
    W_diag = W_diag_kcv.transpose(1, 2, 0)
    diag_h = energy_diff_cv_k(eps_c, eps_v) + V_diag - W_diag
    if use_tda:
        return jax.lax.with_sharding_constraint(
            diag_h, NamedSharding(mesh_xy, P("x", "y", None)))
    diag_full = jnp.stack([diag_h, -jnp.conj(diag_h)], axis=0)[:, None, ...]
    return jax.lax.with_sharding_constraint(
        diag_full, NamedSharding(mesh_xy, P(None, None, "x", "y", None)))


# ── the consolidation itself ─────────────────────────────────────────────

@pytest.mark.parametrize("include_W,use_tda", [
    (True, True), (False, True), (True, False), (False, False),
])
def test_adapter_matches_the_independent_reference(mesh, include_W, use_tda):
    data = _payload()
    ref = np.asarray(_feast_diag_reference(data, mesh, include_W=include_W,
                                           use_tda=use_tda))
    new = np.asarray(BF.build_preconditioner_diagonal_sharded(
        data, mesh, include_W=include_W, use_tda=use_tda))
    assert new.shape == ref.shape
    assert new.dtype == ref.dtype
    assert np.max(np.abs(new - ref)) < ATOL_RY, (
        f"max|Δ| = {np.max(np.abs(new - ref)):.3e} Ry")


def test_feast_keeps_the_diagonal_complex(mesh):
    """FEAST divides by ``z − diag`` at a complex node, so the operator's
    antihermitian residue is its to keep.  Davidson's ``real()`` is not."""
    data = _payload()
    out = BF.build_preconditioner_diagonal_sharded(data, mesh)
    assert jnp.iscomplexobj(out)
    assert float(jnp.max(jnp.abs(jnp.imag(out)))) > 0.0, (
        "the complex route returned a real-valued diagonal; the residue that "
        "distinguishes it from Davidson's has been dropped")


def test_nontda_antiresonant_diagonal_is_negative_conjugate(mesh):
    """The two preconditioner rows follow the production matvec convention.

    A real-only fixture cannot distinguish ``-diag_h`` from
    ``-conj(diag_h)`` and would let the original defect pass.  ``_payload``
    deliberately produces a non-real exact diagonal, so this cell is the red
    twin for the antiresonant row measured by basis-vector applications of the
    ladder matvec: ``H_AA = -conj(H_RR)``.
    """
    data = _payload(seed=20260816)
    resonant = np.asarray(BF.build_preconditioner_diagonal_sharded(
        data, mesh, include_W=True, use_tda=True))
    full = np.asarray(BF.build_preconditioner_diagonal_sharded(
        data, mesh, include_W=True, use_tda=False))

    assert np.max(np.abs(resonant.imag)) > 1e-6, (
        "the fixture lost the non-real diagonal that distinguishes the two "
        "antiresonant conventions")
    assert np.array_equal(full[0, 0], resonant)
    assert np.array_equal(full[1, 0], -np.conj(resonant))
    assert np.max(np.abs(full[1, 0] + resonant)) > 1e-6, (
        "the red twin did not distinguish -diag_h from -conj(diag_h)")


def test_complex_and_real_routes_share_the_real_part_exactly(mesh):
    """One kernel, two static flags — so this is EQUALITY, not agreement."""
    data = _payload()
    nk = NK
    cplx = np.asarray(H.build_bse_exact_diagonal(
        data["eps_c"], data["eps_v"], data["psi_c_X"], data["psi_v_Y"],
        data["W_q"][:, :, 0, 0, 0], data["M_X"], data["M_Y"], data["V_q0"],
        nk, memo=False, complex_out=True))
    real = np.asarray(H.build_bse_exact_diagonal(
        data["eps_c"], data["eps_v"], data["psi_c_X"], data["psi_v_Y"],
        data["W_q"][:, :, 0, 0, 0], data["M_X"], data["M_Y"], data["V_q0"],
        nk, memo=False, complex_out=False))
    assert np.array_equal(cplx.real, real)


def test_include_W_false_drops_the_term_rather_than_zeroing_it(mesh):
    """``W_q0=None`` must remove the contraction, and the answer must equal
    ``ΔE + V_x/nk``."""
    data = _payload()
    nk = NK
    out = np.asarray(H.build_bse_exact_diagonal(
        data["eps_c"], data["eps_v"], data["psi_c_X"], data["psi_v_Y"],
        None, data["M_X"], data["M_Y"], data["V_q0"], nk, memo=False))
    S = np.einsum('kcvM,MN->kcvN', np.asarray(data["M_X"]),
                  np.asarray(data["V_q0"]))
    V_x = np.real(np.einsum('kcvN,kcvN->cvk', S,
                            np.conj(np.asarray(data["M_Y"]))))
    dE = (np.asarray(data["eps_c"]).T[:, None, :]
          - np.asarray(data["eps_v"]).T[None, :, :])
    assert np.allclose(out, dE + V_x / nk, rtol=0, atol=ATOL_RY)


def test_memo_is_skipped_when_an_operand_is_none():
    """A ``None`` has no weak reference, so the identity memo cannot express
    "the same arrays as last time" — it must stand down, not guess."""
    data = _payload()
    nk = NK
    args = (data["eps_c"], data["eps_v"], data["psi_c_X"], data["psi_v_Y"],
            None, data["M_X"], data["M_Y"], data["V_q0"])
    H.clear_exact_diagonal_memo()
    H.build_bse_exact_diagonal(*args, nk, memo=True).block_until_ready()
    before = H.exact_diagonal_memo_stats()["hits"]
    H.build_bse_exact_diagonal(*args, nk, memo=True).block_until_ready()
    assert H.exact_diagonal_memo_stats()["hits"] == before


# ── red twins ────────────────────────────────────────────────────────────

def test_adapter_reads_the_payloads_M_X(mesh):
    """RED TWIN — audit P3's hoisted pair amplitudes must be the ones used.

    An implementation that rebuilds ``M_X`` from ``psi`` locally returns the
    SAME answer here (difference exactly 0.0), which is the whole point: this
    cell is the only thing that can tell the two apart.  ``build_finite_q_data``
    maintains ``M_X``/``M_Y`` per q, so reading ``psi`` instead is a live
    correctness hazard on the finite-q route, not a style preference.
    """
    data = _payload()
    base = np.asarray(BF.build_preconditioner_diagonal_sharded(data, mesh))
    tampered = dict(data)
    tampered["M_X"] = data["M_X"] * 1.5
    out = np.asarray(BF.build_preconditioner_diagonal_sharded(tampered, mesh))
    assert np.max(np.abs(out - base)) > 1e-6, (
        "perturbing data['M_X'] did not move the diagonal — the adapter is "
        "rebuilding the pair amplitudes from psi instead of reading them")


def test_adapter_honours_the_stashed_W_q0(mesh):
    """RED TWIN — the driver stashes the q=0 slice before its donated ifft
    consumes ``W_q``; the adapter must prefer the stash."""
    data = _payload()
    base = np.asarray(BF.build_preconditioner_diagonal_sharded(data, mesh))
    stashed = dict(data)
    stashed["_W_q0_for_precond"] = data["W_q"][:, :, 0, 0, 0] * 1.5
    out = np.asarray(BF.build_preconditioner_diagonal_sharded(stashed, mesh))
    assert np.max(np.abs(out - base)) > 1e-6, (
        "the stashed '_W_q0_for_precond' was ignored")


def test_adapter_refuses_a_payload_with_no_W_at_all(mesh):
    data = _payload()
    data.pop("W_q")
    with pytest.raises(ValueError, match="q=0 slice"):
        BF.build_preconditioner_diagonal_sharded(data, mesh, include_W=True)


def test_no_second_implementation_in_the_source():
    """The dedupe, gated by inspection.

    Values cannot see this regression: a re-added local implementation returns
    the same numbers to round-off — that is exactly what the pre-consolidation
    measurement showed.  Only the source can.
    """
    src = inspect.getsource(BF.build_preconditioner_diagonal_sharded)
    doc = BF.build_preconditioner_diagonal_sharded.__doc__ or ""
    src = src.replace(doc, "")          # prose about einsums is not an einsum
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "build_bse_exact_diagonal(" in body
    assert "einsum" not in body, (
        "build_preconditioner_diagonal_sharded contracts on its own again; "
        "route it through bse_davidson_helpers.build_bse_exact_diagonal")
