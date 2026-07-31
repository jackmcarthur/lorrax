"""Owner decision D10 — ngkmax-padded G loading must agree with the ragged
path to 1e-12, and the pad mask must be load-bearing.

The three operator kernels ``gw.kin_ion_io`` drives per k
(``compute_kinetic_k``, ``compute_local_V_k``, ``vnl_ops.vnl_matrix``
through ``vnl_matrix_from_kdata``) each take a ``g_mask`` hook.  Feeding
them the loader's fixed-shape ``(ngkmax, 3)`` G table plus that mask is
what collapses "one JIT lowering per DISTINCT ngk" to one, and it is what
makes the k loop uniform enough to pipeline behind a single readback.

WHAT EACH TEST PROVES, and how it can fail
------------------------------------------
Every agreement assertion here is paired with a NEGATIVE CONTROL that
runs the same code with the mask removed and asserts the answer is
WRONG by a wide margin.  That is deliberate: the pad rows of
``WfnLoader.gvecs()`` hold the G-vector ``(0, 0, 0)`` — a *valid* FFT-box
index that aliases Γ — so a masking bug does not raise, it silently
double-counts ψ(G=0).  A green "padded == ragged" on its own would
therefore also be green if both sides were equally wrong; the RED control
is what rules that out.

``kin_ion`` matrix elements sit inside H₀'s ~500 eV cancellation
(⟨T+V_ion+V_NL⟩ ≈ −502 eV, ⟨V_H⟩ ≈ +461 eV, sum ≈ −42 eV), so the
tolerance is stated on the *absolute* matrix element, in the same Ry the
kernels return.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from psp.dft_operators import (
    PaddedGVectors,
    gather_psi_G_from_crys,
    padded_gvectors,
)
from psp.get_DFT_mtxels import compute_kinetic_k, compute_local_V_k


# D10's gate.  Bit-exactness is explicitly NOT required: appending exact
# zeros cannot change a sum in IEEE-754, but a shape change does move
# XLA's choice of reduction BLOCKING, so the two routes may associate the
# same nonzero terms differently.
RTOL_D10 = 1e-12


# ---------------------------------------------------------------------------
# Fixtures: a small but non-degenerate plane-wave problem
# ---------------------------------------------------------------------------

def _fixture(nb=4, nspinor=2, grid=(6, 6, 8), ngk=37, ngkmax=48, seed=7):
    """(ψ box, ragged G, padded G, mask, k, bdot, V_r, cell_volume).

    The G list is drawn WITHOUT replacement from the box's index set and
    deliberately includes ``(0,0,0)`` at a non-zero position, so a missing
    mask collides with a physical component rather than an empty slot —
    the exact failure mode the pad contract has to survive.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid

    flat = rng.choice(nx * ny * nz, size=ngk, replace=False)
    G = np.stack(np.unravel_index(flat, (nx, ny, nz)), axis=-1).astype(np.int32)
    G[3] = np.zeros(3, dtype=np.int32)                    # force a Γ row
    # De-duplicate against the forced Γ row so the ragged list stays a set.
    keep = [0] * ngk
    seen = set()
    for i, row in enumerate(G):
        t = tuple(int(v) for v in row)
        keep[i] = t not in seen
        seen.add(t)
    G = G[np.asarray(keep, dtype=bool)]
    ngk = int(G.shape[0])

    pad = ngkmax - ngk
    assert pad > 0, "fixture must actually pad"
    G_pad = np.concatenate([G, np.zeros((pad, 3), dtype=np.int32)], axis=0)
    mask = np.concatenate([np.ones(ngk), np.zeros(pad)]).astype(np.float64)

    psi = (rng.standard_normal((nb, nspinor, nx, ny, nz))
           + 1j * rng.standard_normal((nb, nspinor, nx, ny, nz)))
    psi_box = jnp.asarray(psi, dtype=jnp.complex128)

    A = rng.standard_normal((3, 3))
    bdot = A @ A.T + 3.0 * np.eye(3)                      # SPD metric
    kvec = np.asarray([0.125, -0.25, 0.0])
    V_r = jnp.asarray(rng.standard_normal((nx, ny, nz)), dtype=jnp.float64)
    return psi_box, G, G_pad, mask, kvec, bdot, V_r, 137.5


def _dev(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _scale(a) -> float:
    return float(np.max(np.abs(np.asarray(a))))


# ---------------------------------------------------------------------------
# 1. The kernels: padded+masked == ragged, and the mask is load-bearing
# ---------------------------------------------------------------------------

def test_kinetic_padded_matches_ragged_and_mask_is_load_bearing():
    psi, G, G_pad, mask, kvec, bdot, _V, _vol = _fixture()

    ragged = compute_kinetic_k(psi, G, kvec, bdot)
    padded = compute_kinetic_k(psi, G_pad, kvec, bdot, g_mask=mask)
    unmasked = compute_kinetic_k(psi, G_pad, kvec, bdot)      # NEGATIVE CONTROL

    scale = _scale(ragged)
    assert scale > 1.0, "fixture produced a degenerate T"
    assert _dev(padded, ragged) <= RTOL_D10 * scale

    # RED: without the mask the pad rows re-add |k|²·|ψ(Γ)|² eleven times
    # over.  If this ever passes, the agreement above proves nothing.
    assert _dev(unmasked, ragged) > 1e-6 * scale


def test_local_V_padded_matches_ragged_and_mask_is_load_bearing():
    psi, G, G_pad, mask, _k, _bdot, V_r, vol = _fixture()

    ragged = compute_local_V_k(psi, G, V_r, vol)
    padded = compute_local_V_k(psi, G_pad, V_r, vol, g_mask=mask)
    unmasked = compute_local_V_k(psi, G_pad, V_r, vol)        # NEGATIVE CONTROL

    scale = _scale(ragged)
    assert scale > 0.0
    assert _dev(padded, ragged) <= RTOL_D10 * scale
    assert _dev(unmasked, ragged) > 1e-6 * scale


def test_gather_psi_G_padded_zeroes_pad_columns():
    """The V_NL route masks ψ_G, not Z — check the ψ side directly.

    ``vnl_ops.vnl_matrix`` contracts ψ_G against Z twice, so zeroing ψ_G
    at the pad columns is sufficient to make Z's (finite, evaluated at
    K = kvec) pad values inert.  This pins that ψ-side contract.
    """
    psi, G, G_pad, mask, *_ = _fixture()

    ragged = np.asarray(gather_psi_G_from_crys(psi, G))
    padded = np.asarray(gather_psi_G_from_crys(psi, G_pad, mask))

    ngk = int(G.shape[0])
    assert np.array_equal(padded[..., :ngk], ragged)
    assert np.all(padded[..., ngk:] == 0)

    # NEGATIVE CONTROL: unmasked pad columns carry the physical Γ
    # coefficient, not zero.
    unmasked = np.asarray(gather_psi_G_from_crys(psi, G_pad))
    assert np.max(np.abs(unmasked[..., ngk:])) > 0.0


# ---------------------------------------------------------------------------
# 2. The sum-of-terms: the whole ⟨m|T+V|n⟩ block, which is what H₀ uses
# ---------------------------------------------------------------------------

def test_summed_operator_block_agrees_at_1e12():
    """T + V_loc together, the combination ``get_kin_ion_k`` returns.

    Tested as a sum because that is the quantity D10 gates: the pieces
    could each drift within tolerance and still cancel differently.
    """
    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture(nb=6, ngk=41)

    ragged = (compute_kinetic_k(psi, G, kvec, bdot)
              + compute_local_V_k(psi, G, V_r, vol))
    padded = (compute_kinetic_k(psi, G_pad, kvec, bdot, g_mask=mask)
              + compute_local_V_k(psi, G_pad, V_r, vol, g_mask=mask))

    scale = _scale(ragged)
    assert _dev(padded, ragged) <= RTOL_D10 * scale


def test_get_kin_ion_k_refuses_a_padded_list_without_its_mask():
    """The mask contract is ENFORCED, not merely documented.

    Pad rows are ``(0,0,0)`` — a valid box index — so an unmasked padded
    list returns a plausible-looking matrix that is wrong by
    ``(ngkmax - ngk)`` extra copies of the Γ component.  A physical
    G-sphere has ``(0,0,0)`` at most once, so >1 all-zero row is a
    sufficient signature.
    """
    import pytest
    from gw.kin_ion_io import get_kin_ion_k

    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture()

    class _W:                                     # only bdot/cell_volume used
        bdot = None
        cell_volume = 137.5

    w = _W()
    w.bdot = bdot

    with pytest.raises(ValueError, match="ngkmax-padded"):
        get_kin_ion_k(psi, G_pad, kvec, V_r, None, w)

    # The RAGGED list (exactly one Γ row) must still be accepted without a
    # mask — otherwise the guard is just breaking the old path.
    out = get_kin_ion_k(psi, G, kvec, V_r, None, w)
    assert np.isfinite(np.asarray(out)).all()
    # ...and the padded list WITH its mask agrees with it at 1e-12.
    out_pad = get_kin_ion_k(psi, G_pad, kvec, V_r, None, w, g_mask=mask)
    assert _dev(out_pad, out) <= RTOL_D10 * _scale(out)


# ---------------------------------------------------------------------------
# 3. PaddedGVectors itself
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Minimal stand-in for ``WfnLoader``'s two G accessors.

    ``padded_gvectors`` reaches the loader through ``_as_loader``, which
    accepts a real ``WfnLoader`` or falls back to reopening ``_filename``.
    Duck-typing the two methods keeps this test free of a WFN fixture;
    the full-deck agreement is measured by the sbatch harness, not here.
    """

    def __init__(self, gvecs, ngk):
        self._g, self._n = gvecs, ngk

    def gvecs(self, *, k="full_bz"):
        return self._g

    def ngk_valid(self, *, k="full_bz"):
        return self._n


def test_padded_gvectors_mask_matches_ngk_valid(monkeypatch):
    import psp.dft_operators as dop

    nk, ngkmax = 4, 9
    ngk = np.asarray([9, 7, 5, 8], dtype=np.int32)
    rng = np.random.default_rng(1)
    g = rng.integers(0, 5, size=(nk, ngkmax, 3)).astype(np.int32)
    for j in range(nk):
        g[j, int(ngk[j]):] = 0                            # loader's pad rows

    fake = _FakeLoader(g, ngk)
    monkeypatch.setattr(dop, "_as_loader", lambda w: fake)

    tab = padded_gvectors(object())
    assert isinstance(tab, PaddedGVectors)
    assert tab.ngkmax == ngkmax and tab.n_k == nk
    assert tab.mask.sum(axis=1).tolist() == ngk.tolist()
    for j in range(nk):
        G_j, m_j = tab.at(j)
        assert np.array_equal(m_j[: int(ngk[j])], np.ones(int(ngk[j])))
        assert np.all(m_j[int(ngk[j]):] == 0.0)
        # Pad rows are (0,0,0): a VALID box index, hence the mask.
        assert np.all(G_j[int(ngk[j]):] == 0)


# ---------------------------------------------------------------------------
# 4. sweep_local_k — the one-readback contract
# ---------------------------------------------------------------------------

def test_sweep_local_k_places_blocks_by_global_index():
    from common.collectives import sweep_local_k

    rng = np.random.default_rng(3)
    ref = {ik: jnp.asarray(rng.standard_normal((2, 2))
                           + 1j * rng.standard_normal((2, 2)))
           for ik in (1, 4, 7)}

    calls: list[int] = []

    def per_k(ik):
        calls.append(ik)
        return ref[ik]

    vals, idx = sweep_local_k([1, 4, 7], 5, (2, 2), per_k,
                              print_fn=lambda *a: None)

    assert calls == [1, 4, 7]
    assert vals.shape == (5, 2, 2)
    assert idx.tolist() == [1, 4, 7, -1, -1]
    for j, ik in enumerate((1, 4, 7)):
        assert np.array_equal(vals[j], np.asarray(ref[ik]))
    # Unused slots stay zero — ``_gather_indexed_blocks`` skips them by
    # index, but a nonzero pad would corrupt any consumer that does not.
    assert not np.any(vals[3:])


def test_sweep_local_k_empty_rank_keeps_the_collective_shape():
    """world > nk leaves ranks with NO k — they still join the gather.

    ``_gather_indexed_blocks`` ends in ``process_allgather``, a
    collective: a rank that returned a differently-shaped placeholder
    would mis-assemble the result on every rank, not just its own.  This
    is the exact failure P=64 on a 16-k deck would hit.
    """
    from common.collectives import sweep_local_k

    def per_k(ik):                      # never called
        raise AssertionError("empty rank must not dispatch")

    vals, idx = sweep_local_k([], 1, (6, 6), per_k, print_fn=lambda *a: None)
    assert vals.shape == (1, 6, 6)
    assert vals.dtype == np.complex128
    assert not np.any(vals)
    assert idx.tolist() == [-1]


def test_sweep_local_k_refuses_a_shape_mismatch():
    """The declared item_shape is enforced, not trusted."""
    from common.collectives import sweep_local_k
    import pytest

    with pytest.raises(ValueError, match="item_shape"):
        sweep_local_k([0], 1, (4, 4), lambda ik: jnp.zeros((3, 3)),
                      print_fn=lambda *a: None)


def test_sweep_local_k_lookahead_does_not_change_values():
    """Pipelining depth is a scheduling knob, never a numerical one."""
    from common.collectives import sweep_local_k

    psi, G, G_pad, mask, kvec, bdot, V_r, vol = _fixture(nb=3)

    def per_k(ik):
        return compute_kinetic_k(psi * (ik + 1.0), G_pad, kvec, bdot,
                                 g_mask=mask)

    a, _ = sweep_local_k(range(4), 4, (3, 3), per_k, lookahead=1,
                         print_fn=lambda *x: None)
    b, _ = sweep_local_k(range(4), 4, (3, 3), per_k, lookahead=8,
                         print_fn=lambda *x: None)
    assert np.array_equal(a, b)
