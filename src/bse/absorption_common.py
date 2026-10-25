"""Shared utilities for BSE absorption-spectrum post-processors.

``absorption_haydock`` (continued-fraction recursion, no eigenvectors)
consumes these.

Common needs:
  - read BGW-format eigenvectors.h5 (our writer matches this spec)
  - read dipole.h5 written by psp.get_dipole_mtxels
  - slice the (nb, nb) dipole window down to the BSE (nc, nv) sub-block
  - Lorentzian broaden
  - Kramers-Kronig
  - write BGW-style 4-column ``absorption_b{1,2,3}.dat``
  - write BerkeleyGW's ``eigenvalues.dat`` / ``eigenvalues_b{1,2,3}.dat``
    (``bse_jax`` writes every root there)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from common.log import get_logger
from common.units import RYD_TO_EV

log = get_logger(__name__)

#: Ry -> eV, from ``common.units`` — the one place the number lives.  Kept
#: under this module's historical name because both absorption drivers import
#: ``absorption_common.RYD2EV``.
RYD2EV = RYD_TO_EV


def exciton_dipole_projections(A, d_alpha):
    """⟨0|r̂_α|S⟩ — THE contraction of a BSE eigenvector against the dipole.

    THIS IS THE SINGLE SITE.  Every absorption driver that owns explicit
    eigenvectors calls this and nothing else, because the tree has already
    paid once for two drivers spelling this contraction differently
    (``KNOWN_FAILURES.md``, "THE TWO ABSORPTION DRIVERS DISAGREE ON A
    CONJUGATION").  The conjugate sits on the DIPOLE, not on ``A``.

    DERIVATION, from the spectral representation and this tree's own
    storage conventions — not from BerkeleyGW habit:

      * The exciton state is ``|S⟩ = Σ_t A^S_t â†_ck â_vk |0⟩`` over
        transitions ``t = (c, v, k)``.  That ``A`` is the amplitude in
        THAT basis is fixed by LORRAX's own kernel, not assumed: the
        exchange term is assembled as ``K^x = M V M†`` with
        ``M_t = conj(ψ_c) ψ_v`` (the stack matvec's V-term, and the
        direct term ``K^d_{tt'} = −Σ conj(ψ_c[k]) ψ_c'[k'] W
        ψ_v[k] conj(ψ_v'[k'])`` documented at ``bse_nontda.py:169``).
        Both are the standard ``⟨t|H|t'⟩`` with the CONDUCTION index on
        the bra, so the operator the solvers diagonalise is H itself and
        not its complex conjugate, and ``A`` is the amplitude and not its
        conjugate.
      * ``slice_dipole_to_bse_window`` returns
        ``d_t = ⟨ck|v̂_α|vk⟩ / ΔE`` — conduction on the BRA (the producer's
        own convention, ``dipole_cart[α,k,m,n] = ⟨mk|v̂_α|nk⟩``).
      * Therefore
        ``⟨0|r̂_α|S⟩ = Σ_t A^S_t ⟨0|r̂_α â†_c â_v|0⟩ = Σ_t A^S_t ⟨vk|r̂_α|ck⟩
                     = Σ_t A^S_t · conj(d^α_t)``.

    Two independent witnesses agree with that line, and they are the
    reason this is a verdict rather than a preference:

      * BerkeleyGW does the identical contraction —
        ``BSE/diag.f90:711`` is ``dipoles_r = Σ u_r · MYCONJG(s1)``, and
        ``Common/mtxel_optical.f90``'s ``mtxel_m``/``mtxel_v`` build
        ``s0(ic,iv) = ⟨ic,k|…|iv,k⟩ / ΔE``, the same index order as ours.
      * The Haydock route needs no eigenvector convention at all: it
        evaluates ``⟨d|(z−H)⁻¹|d⟩``, whose spectral weights are
        ``|⟨S|d⟩|² = |Σ_t conj(A_t) d_t|²`` — the modulus of the
        expression above.  A sum-over-states spectrum built any other way
        does not reproduce the resolvent, which is what
        ``tests/test_absorption_conjugation.py`` measures.

    Note the phase convention: ``Σ_t A_t conj(d_t)`` and
    ``Σ_t conj(A_t) d_t`` have the SAME modulus (they are complex
    conjugates), so ε₂ cannot tell them apart — but the BGW-format
    ``eigenvalues_b*.dat`` writes ``Re`` and ``Im`` in separate columns,
    so the tree emits the BGW-side spelling.  What is NOT a phase choice,
    and is the defect this function closes, is contracting ``A`` with a
    bare ``d``: ``Σ_t A_t d_t`` is a different number in modulus too,
    measured at up to 6.8× per element on the committed fixtures.

    Parameters
    ----------
    A : (N, *T) complex
        Eigenvectors.  Leading axis is the state; the trailing axes are
        the transition block in whatever order the caller holds it.
    d_alpha : (3, *T) complex
        Dipole ``d^α_t``, with the SAME trailing transition axes as ``A``
        (the two drivers hold them in different orders — ``(nk, nc, nv)``
        and ``(nc, nv, nk)`` — and this function is layout-agnostic
        precisely so neither has to reshape at the call site).

    Returns
    -------
    (N, 3) complex128 — ⟨0|r̂_α|S⟩ per state, per polarisation.

    DISTRIBUTED OPERANDS.  When ``A`` or ``d_alpha`` is a ``jax.Array`` (the
    solver's eigenvectors, sharded over the transition axes) the same
    contraction runs as one ``tensordot`` over the trailing axes: each rank
    contracts its own (c, v, k) tile and one all-reduce of the (N, 3) result
    follows.  Nothing is gathered, and the result stays a ``jax.Array``.
    """
    if not (_is_jax_array(A) or _is_jax_array(d_alpha)):
        A = np.asarray(A)
        d_alpha = np.asarray(d_alpha)
    if A.shape[1:] != d_alpha.shape[1:]:
        raise ValueError(
            f"eigenvector transition axes {A.shape[1:]} do not match the "
            f"dipole's {d_alpha.shape[1:]}.  Both must be indexed over the "
            f"same (c, v, k) block in the same order; a transpose here is a "
            f"silently wrong oscillator strength, not a broadcast.")
    if _is_jax_array(A) or _is_jax_array(d_alpha):
        import jax.numpy as jnp
        axes = tuple(range(1, A.ndim))
        return jnp.tensordot(A, jnp.conj(d_alpha), axes=(axes, axes)
                             ).astype(jnp.complex128)
    n_state = A.shape[0]
    n_trans = int(np.prod(A.shape[1:])) if A.ndim > 1 else 1
    return (A.reshape(n_state, n_trans)
            @ np.conj(d_alpha.reshape(d_alpha.shape[0], n_trans)).T
            ).astype(np.complex128)


def _is_jax_array(x) -> bool:
    """A ``jax.Array`` (or tracer), without importing jax for NumPy callers."""
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


#: Share of TOTAL per-device memory one block of eigenvector rows may occupy
#: while :func:`exciton_dipoles_distributed` contracts it.  Total, not free
#: memory: a block width read off ambient free memory changes run to run.
_DIPOLE_BLOCK_BUDGET_FRAC = 0.05


def exciton_dipoles_distributed(eigenvectors, d_block, n_states: int):
    """⟨0|r̂_α|S⟩ for the first ``n_states`` rows of the solver's eigenvectors.

    ``eigenvectors`` : ``(N, 1, nc_pad, nv_pad, nk)`` (TDA, as the BSE solves
    return them) or ``(N, nc_pad, nv_pad, nk)``, a ``jax.Array`` sharded over
    the transition axes.  ``d_block`` : ``(3, nc_pad, nv_pad, nk)``, the
    ``build_dipole_vector_bse`` layout on the same transition axes (pad rows
    are zero in both, so they add nothing).

    The contraction is :func:`exciton_dipole_projections` applied to blocks of
    states, a batched GEMM on each rank's (c, v, k) tile plus one (block, 3)
    all-reduce per block, so no rank ever holds another rank's eigenvector
    tile.  The block width comes from the per-device memory budget
    (``_DIPOLE_BLOCK_BUDGET_FRAC`` of total device memory over the local bytes
    of one state), the existing BSE path's sizing rule.  Every process must
    call this (the slices and the reduction are global computations); every
    process returns the same host ``(n_states, 3)`` complex128 array.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from common.collectives import gather_to_host
    from common.gpu_utils import device_budget_bytes

    tda_axis = eigenvectors.ndim == 5
    if tda_axis and int(eigenvectors.shape[1]) != 1:
        raise NotImplementedError(
            "exciton dipoles are TDA-only: the full-BSE oscillator strength "
            "needs both halves of the paired vector (see load_eigenvectors_h5)")
    n_states = int(n_states)
    sharding = eigenvectors.sharding
    local = sharding.shard_shape(eigenvectors.shape)
    per_state = 16 * int(np.prod(local[1:]))
    budget = device_budget_bytes() * _DIPOLE_BLOCK_BUDGET_FRAC
    block = int(min(max(n_states, 1), max(1, budget // per_state)))
    if isinstance(sharding, NamedSharding):
        # The dipole takes the eigenvectors' transition layout, so each rank
        # contracts the (c, v, k) tile it already holds.
        spec = tuple(sharding.spec) + (None,) * (
            eigenvectors.ndim - len(sharding.spec))
        d_sharding = NamedSharding(sharding.mesh, P(None, *spec[-3:]))
        out_sharding = NamedSharding(sharding.mesh, P())
    else:
        d_sharding = out_sharding = sharding

    def _block(i0, size):
        # The block is sliced INSIDE the program, so no copy of the whole
        # stack is ever made; one compile per distinct (i0, size).
        def contract(A, d):
            A = jax.lax.slice_in_dim(A, i0, i0 + size, axis=0)
            return exciton_dipole_projections(A[:, 0] if tda_axis else A, d)
        return jax.jit(contract, in_shardings=(sharding, d_sharding),
                       out_shardings=out_sharding)

    d_block = jnp.asarray(d_block)
    out = np.zeros((n_states, int(d_block.shape[0])), np.complex128)
    for i0 in range(0, n_states, block):
        size = min(block, n_states - i0)
        out[i0:i0 + size] = gather_to_host(
            _block(i0, size)(eigenvectors, d_block))
    return out


def eps2_from_exciton_dipoles(omegas_Ry, energies_Ry, dipoles, eta_Ry,
                              V_cell, n_k, n_spin, n_spinor):
    """Sum-over-states ε₂(ω), ``(n_omega, 3)``, from saved per-state dipoles.

    ``ε₂^α(ω) = 16π²/(V·N_k·n_spin·n_spinor) · Σ_S |⟨0|r̂_α|S⟩|² L_η(ω − E_S)``
    with the Lorentzian of :func:`lorentzian_broaden`: the prefactor and the
    broadening ``absorption_haydock`` uses, so over the states it includes this
    is the Haydock resolvent's spectrum state by state.  States above the last
    one saved are absent: below ``E_N`` their tails add at most
    ``pref · W_missing · L_η(E_N − ω)``.
    """
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    weights = np.abs(np.asarray(dipoles)) ** 2
    return pref * lorentzian_broaden(omegas_Ry, np.asarray(energies_Ry),
                                     weights, eta_Ry)


def load_exciton_dipoles_h5(path: str | Path):
    """Eigenvalues (Ry) and per-state dipoles ``(N, 3)`` from eigenvectors.h5.

    Reads ``exciton_data/dipoles`` (a LORRAX extension of the BGW file,
    ``bse/eigenvectors.h5.spec``) without touching the eigenvectors.
    """
    with h5py.File(str(path), "r") as f:
        if "dipoles" not in f["exciton_data"]:
            raise KeyError(
                f"{path!s} has no exciton_data/dipoles: it was written without "
                f"dipole_file (bse.bse_jax), or by BerkeleyGW")
        eig_eV = np.asarray(f["exciton_data/eigenvalues"][()], dtype=np.float64)
        raw = np.asarray(f["exciton_data/dipoles"][0])        # (N, 3, 2)
    return eig_eV / RYD2EV, raw[..., 0] + 1j * raw[..., 1]


def load_eigenvectors_h5(path: str | Path):
    """Load BGW-format eigenvectors.h5 (TDA, single Q, single spin assumed).

    Returns
    -------
    eigvals : (N,) float64 — exciton energies in Ry.
    A       : (N, nk, nc, nv) complex128 — eigenvectors in BSE basis.
    params  : dict with keys ``nc, nv, ns, nspinor, nk, kpts, spin_kernel``.

    Note: BGW convention ``ns`` is the spin index (=2 for collinear-spin,
    =1 for spinor or non-spin), distinct from ``nspinor`` (=2 for spinor,
    inferred here from ``spin_kernel == 3``).

    NON-TDA FILES ARE REFUSED, not silently truncated.  A non-TDA file
    (``use_tda = 0``) holds the resonant ``X`` in ``exciton_data/eigenvectors``
    and the coupling ``Y`` in ``eigenvectors_deexcitation`` (BerkeleyGW's
    layout, written by ``bse_window.write_eigenvectors_stream`` too); this
    reader only ever returned the first.  Reading a non-TDA file and
    contracting ``X`` alone gives the TDA answer for a non-TDA solve,
    which is wrong by the whole anti-resonant channel — see the
    ``NotImplementedError`` below for the contraction it would need.
    ``kpts`` is BerkeleyGW's ``(nk, 3)``.
    """
    with h5py.File(str(path), "r") as f:
        eigvals_eV = np.asarray(f["exciton_data/eigenvalues"][:], dtype=np.float64)
        evecs = np.asarray(f["exciton_data/eigenvectors"][:])  # (nQ,N,nk,nc,nv,ns,2)
        use_tda = bool(int(f["exciton_header/params/use_tda"][()]))
        if not use_tda:
            raise NotImplementedError(
                f"{path!s} is a NON-TDA eigenvector file (use_tda = 0), and the "
                f"sum-over-states absorption route is TDA-only.\n"
                f"The full-BSE oscillator strength is NOT |Σ_t X_t conj(d_t)|²: "
                f"it needs both halves of the paired vector and both the left "
                f"and right solutions,\n"
                f"    ⟨0|r|S⟩_r = Σ_t X^S_t conj(d_t) − Σ_t Y^S_t d_t\n"
                f"    ⟨0|r|S⟩_l = Σ_t X̃^S_t conj(d_t) + Σ_t Ỹ^S_t d_t\n"
                f"    f_S       = Re[ conj(⟨0|r|S⟩_l) · ⟨0|r|S⟩_r ]\n"
                f"(the anti-resonant dipole is −conj(d) because "
                f"s_(c→v) = −conj(s_(v→c)), BerkeleyGW BSE/diag.f90:722-731), "
                f"and no driver in this tree assembles it yet.  Solve with "
                f"full_bse = false, or add the coupling contraction before reading this "
                f"file for absorption.")
        spin_kernel = int(f["exciton_header/params/spin_kernel"][()])
        params = {
            "nc": int(f["exciton_header/params/nc"][()]),
            "nv": int(f["exciton_header/params/nv"][()]),
            "ns": int(f["exciton_header/params/ns"][()]),
            "nspinor": 2 if spin_kernel == 3 else 1,
            "spin_kernel": spin_kernel,
            "nk": int(f["exciton_header/kpoints/nk"][()]),
            "kpts": np.asarray(f["exciton_header/kpoints/kpts"][:]),
        }
    nQ = evecs.shape[0]
    ns = params["ns"]
    if nQ != 1 or ns != 1:
        raise NotImplementedError(
            f"Only nQ=ns=1 supported (got nQ={nQ}, ns={ns})")
    A = evecs[..., 0] + 1j * evecs[..., 1]   # (nQ, N, nk, nc, nv, ns)
    A = A[0, :, :, :, :, 0]                  # (N, nk, nc, nv)
    # BGW convention: v=0 is the highest valence band (closest to gap).
    # Our absorption-side internal convention slices the dipole as
    # ``val_idx = n_occ - n_val .. n_occ`` (lowest-first), so flip the
    # valence axis here on read to match.
    A = A[:, :, :, ::-1]
    # File stores eigenvalues in eV (BGW convention) — convert back to Ry
    # for the rest of the absorption pipeline which works in Ry.
    eigvals_Ry = eigvals_eV / RYD2EV
    return eigvals_Ry, A.astype(np.complex128), params





#: The circular convention every helicity table states in its header: the
#: field defines the label (campaign-wide, 2026-10-16), not the exciton.
HELICITY_CONVENTION = (
    "σ+ is the field E(t) = E₀ (cos ωt, sin ωt, 0): x → y, counter-clockwise "
    "seen from +z",
    "D = ⟨0|r|S⟩ (exciton_data/dipoles); I± = |D_x ∓ i D_y|² / (2 N_k) in "
    "bohr² per unit cell",
    "η = (I+ − I−) / (I+ + I−): +1 is pure σ+, −1 pure σ−",
)


def inplane_helicity(dipoles, n_k):
    r"""Per-state circular absorption strengths in the plane, per unit cell.

    σ± is the field :math:`E(t) = E_0(\cos\omega t, \pm\sin\omega t, 0) =
    \mathrm{Re}[E_0\,\varepsilon_\pm e^{-i\omega t}]` with
    :math:`\varepsilon_\pm = (\hat x \pm i\hat y)/\sqrt 2` (σ+ rotates
    x → y, counter-clockwise seen from +z).  Absorption into |S⟩ goes as
    :math:`|\langle S|\varepsilon\cdot\hat r|0\rangle|^2 =
    |\varepsilon\cdot D^*|^2` with :math:`D = \langle 0|\hat r|S\rangle`:

    .. math::
        I_\pm = \frac{|D_x \mp i D_y|^2}{2 N_k},\qquad
        \eta = \frac{I_+ - I_-}{I_+ + I_-}.

    ``dipoles`` is :func:`exciton_dipole_projections`' ``(n_states, 3)``,
    which is ⟨0|r̂|S⟩ times the global −i of the velocity form
    (``d = ⟨c|v̂|v⟩/ΔE = i⟨c|r̂|v⟩``) on the native route and on the
    ``kgrid_fine`` route alike; no modulus sees that factor.  The
    ⟨S|r̂|0⟩ spelling (a conjugated D) swaps I+ and I−, so the convention
    lives in this function and nowhere else.  An exciton normalised over the
    N_k-point zone has |D|² ∝ N_k, so the division gives strengths per unit
    cell (bohr²), and I+ + I− is the in-plane |D|²/N_k.

    Returns ``(i_plus, i_minus, eta)``, each ``(n_states,)``; ``eta`` is NaN
    for a state with no in-plane strength.
    """
    d = np.asarray(dipoles, dtype=np.complex128)
    n_k = int(n_k)
    if d.ndim != 2 or d.shape[1] < 2 or n_k < 1:
        raise ValueError(
            f"inplane_helicity: want (n_states, 3) dipoles and n_k >= 1; got "
            f"shape {d.shape} and n_k = {n_k}")
    i_plus = np.abs(d[:, 0] - 1j * d[:, 1]) ** 2 / (2.0 * n_k)
    i_minus = np.abs(d[:, 0] + 1j * d[:, 1]) ** 2 / (2.0 * n_k)
    total = i_plus + i_minus
    eta = np.full(total.shape, np.nan)
    np.divide(i_plus - i_minus, total, out=eta, where=total > 0.0)
    return i_plus, i_minus, eta


def helicity_table_lines(energies_ev, dipoles, n_k):
    """The ``bse_jax helicity_table`` report block, one row per state.

    The convention (:data:`HELICITY_CONVENTION`) and ``N_k`` head the
    table; the rows are :func:`inplane_helicity` of ``dipoles`` beside each
    state's energy (eV), named ``S001…`` as in the exciton-spectrum table.
    """
    i_plus, i_minus, eta = inplane_helicity(dipoles, n_k)
    energies_ev = np.asarray(energies_ev, dtype=np.float64)
    if energies_ev.shape != i_plus.shape:
        raise ValueError(
            f"helicity_table_lines: {energies_ev.shape[0]} energies for "
            f"{i_plus.shape[0]} dipoles")
    lines = [f"  {line}" for line in HELICITY_CONVENTION]
    lines += [f"  N_k = {int(n_k)}", "",
              "  state   energy (eV)    I+ (bohr²)     I− (bohr²)       η"]
    for index, (e, ip, im, h) in enumerate(
            zip(energies_ev, i_plus, i_minus, eta), start=1):
        lines.append(f"  S{index:03d}  {e:12.5f}  {ip:13.6e}  {im:13.6e}  "
                     + ("      n/a" if not np.isfinite(h) else f"{h:+9.4f}"))
    return lines


def require_dipole_not_short(attrs, nb_stored: int, band_hi: int, *,
                             dipole_path) -> None:
    """Refuse a ``dipole.h5`` that stops below the BSE window's top band.

    ``band_hi`` is the window's top (absolute, 0-based, exclusive); the file
    holds bands up to its ``band_max`` stamp, else ``nb_stored``.  A short
    file would leave the transitions above its top with no dipole: zero
    oscillator strength, and no message.
    """
    from file_io.dipole import DIPOLE_BAND_MAX_ATTR, DIPOLE_BAND_MIN_ATTR
    lo = int(np.asarray(attrs.get(DIPOLE_BAND_MIN_ATTR, 0)).reshape(-1)[0])
    hi = int(np.asarray(attrs.get(DIPOLE_BAND_MAX_ATTR,
                                  lo + int(nb_stored))).reshape(-1)[0])
    if int(band_hi) > hi:
        raise ValueError(
            f"GATE dipole_bands_short: {dipole_path!s} holds bands up to "
            f"{hi}; want at least {int(band_hi)}, the top of the BSE window; "
            f"why: the transitions to bands {hi + 1}-{int(band_hi)} would get "
            f"no dipole and zero oscillator strength; fix: regenerate it with "
            f"python -m psp.get_dipole_mtxels and nval/ncond/number_bands "
            f"reaching band {int(band_hi)}.")


def slice_dipole_to_bse_window(dipole_cart, deltaE, n_occ, n_val, n_cond):
    """Slice (nb, nb) dipole tables → BSE (nc, nv) sub-block.

    Convention: ``dipole_cart[α, k, m, n] = ⟨mk|v̂_α|nk⟩``.
    For the dipole oscillator we need ⟨ck|v̂|vk⟩ ⇒ m=c, n=v.

    Returns
    -------
    d_alpha : (3, nk, nc, nv) complex — d^α_{cvk} = ⟨c|v̂|v⟩ / ΔE   (r-form).
    de_cv   : (nk, nc, nv) float      — E_c - E_v (Ry, > 0).
    """
    from file_io.dipole import require_dipole_bands
    val_lo = n_occ - n_val
    cond_hi = n_occ + n_cond
    require_dipole_not_short({}, int(dipole_cart.shape[-1]), cond_hi,
                             dipole_path="dipole_cart")
    require_dipole_bands({}, int(dipole_cart.shape[-1]), val_lo, cond_hi,
                         dipole_path="dipole_cart", where="BSE window")
    v_cv = dipole_cart[:, :, n_occ:cond_hi, val_lo:n_occ]   # (3, nk, nc, nv)
    de_cv = deltaE[:, n_occ:cond_hi, val_lo:n_occ]          # (nk, nc, nv)
    d_alpha = v_cv / de_cv[None]
    return d_alpha, de_cv


def _deck_wfn_fingerprint(wfn_path):
    """``wfn_fingerprint`` of the deck WFN (header and bounded samples)."""
    from common.parallel_transport import wfn_fingerprint
    from wfn_loader import WfnLoader
    return wfn_fingerprint(WfnLoader(str(wfn_path)))


def window_dipoles(data, dipole_file, *, n_occ, wfn_path):
    """``(d_alpha, de_cv)`` of the BSE window ``data`` solves: THE dipole source.

    A ``kgrid_fine`` bundle carries its own (``data["dipoles"]``, built by
    ``bse_densify.densified_dipoles`` from ``dipole_file``, the coarse WFN's
    velocity, in the fine ψ gauge) and never slices a file: the coarse file
    has another k list, and a fine grid's ``dipole.h5`` has the right shape
    in the wrong gauge.  Any other bundle reads ``dipole_file``
    after ``file_io.dipole.require_dipole_k_grid`` and the file's WFN stamp
    against ``wfn_path`` (``file_io.dipole.require_dipole_wfn_fingerprint``),
    and reads only the loader's resolved c/v window. On a
    ``qp_rotations_file`` bundle
    (``data["qp_rotation"]``), takes the QP-basis dipoles of
    ``bse.bse_qp_rotation.qp_window_dipoles`` from the same checked file,
    which must then hold the velocity p − i[r, V_NL] of H_DFT
    (``GATE bse_qp_dipole_velocity``).

    Returns
    -------
    d_alpha : (3, nk, nc_pad, nv_pad) complex, matrix-face sharded
        ⟨ck|v̂_α|vk⟩ / ΔE (bohr); carrier slots are zero.
    de_cv : (nk, nc_pad, nv_pad) float, matrix-face sharded
        The velocity's own E_c − E_v (Ry), QP energies on the rotated route;
        carrier slots are infinite and do not count as physical transitions.
    """
    import os
    if data.get("densified"):
        dip = data.get("dipoles")
        if dip is None or dip["source"] != os.path.abspath(str(dipole_file)):
            raise ValueError(
                f"GATE densified_dipoles: got a kgrid_fine bundle whose "
                f"dipoles are {None if dip is None else dip['source']!r}; "
                f"want those built from {os.path.abspath(str(dipole_file))!r}; "
                f"why: a densified bundle's dipoles exist only in its own ψ "
                f"gauge, and slicing a dipole file onto it is wrong (another "
                f"k list, or the right shape in another gauge); fix: pass "
                f"velocity_file=dipole_file to "
                f"load_bse_data_from_restart_sharded.")
        from common.collectives import device_put_process_local
        from jax.sharding import NamedSharding, PartitionSpec as P
        mesh = data["psi_c_X"].sharding.mesh
        nc, nv = map(int, dip["de_cv"].shape[-2:])
        extra = ((0, 0), (0, int(data["n_cond_pad"]) - nc),
                 (0, int(data["n_val_pad"]) - nv))
        # The densifier already owns these host-window fields. Place their
        # existing values once; downstream optical consumers share the same
        # legal pair carrier as file-backed windows.
        d = device_put_process_local(np.pad(dip["d_alpha"], ((0, 0), *extra)),
            NamedSharding(mesh, P(None, None, "x", "y")))
        de = device_put_process_local(np.pad(dip["de_cv"], extra, constant_values=np.inf),
            NamedSharding(mesh, P(None, "x", "y")))
        return d, de
    from file_io.dipole import (require_dipole_k_grid,
                                require_dipole_wfn_fingerprint)
    from file_io.restart_bundle import load_dipole_h5, read_dipole_metadata
    attrs = read_dipole_metadata(dipole_file)
    require_dipole_k_grid(
        attrs, int(data["nkx"]) * int(data["nky"]) * int(data["nkz"]),
        dipole_path=dipole_file)
    require_dipole_wfn_fingerprint(attrs, dipole_path=dipole_file,
                                   wfn_path=wfn_path,
                                   want=_deck_wfn_fingerprint(wfn_path))
    mesh = data["psi_c_X"].sharding.mesh
    nc, nv = int(data["n_cond"]), int(data["n_val"])
    output_shape = int(data["n_cond_pad"]), int(data["n_val_pad"])
    qp = data.get("qp_rotation")
    rows = (n_occ, n_occ + nc) if qp is None else qp["band_range"]
    cols = (n_occ - nv, n_occ) if qp is None else qp["band_range"]
    velocity, delta, attrs = load_dipole_h5(
        dipole_file, wfn_path=wfn_path, mesh_xy=mesh,
        band_rows=rows, band_cols=cols,
        band_shape=output_shape if qp is None else None)
    if qp is None:
        return velocity / delta[None], delta
    if bool(attrs.get("skip_vnl", False)):
        raise ValueError(
            f"GATE bse_qp_dipole_velocity: got {dipole_file} with skip_vnl = "
            "true (momentum only); want the velocity p - i[r, V_NL] of "
            "H_DFT; rebuild it with python -m psp.get_dipole_mtxels -i <deck>.")
    from .bse_qp_rotation import qp_window_dipoles
    return qp_window_dipoles(velocity, delta, qp, mesh_xy=mesh,
                             output_shape=output_shape)


def window_transition_energies(data):
    """The BSE operator's own window transitions ``E_c − E_v``, ``(nk, nc, nv)`` Ry.

    Where each transition sits in H (QP under ``eqp_corrections`` /
    ``eqp_co_corrections``, the QP states' energies under ``qp_rotations_file``):
    the δ positions of the independent-particle spectra (``absorption_noeh.dat``,
    Haydock's noeh column), as BerkeleyGW places them.  :func:`window_dipoles`'
    ``de_cv`` is the velocity's own ΔE, the dipoles' denominator, and stays that
    (DFT energies on the ``eqp`` routes, 0.7 eV low on Si).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = data["psi_c_X"].sharding.mesh
    nc, nv = int(data["n_cond"]), int(data["n_val"])
    def transitions(ec, ev):
        d = ec[:, :, None] - ev[:, None, :]
        live = ((jnp.arange(d.shape[1])[None, :, None] < nc)
                & (jnp.arange(d.shape[2])[None, None, :] < nv))
        return jnp.where(live, d.real, jnp.inf)
    return jax.jit(transitions,
        out_shardings=NamedSharding(mesh, P(None, "x", "y")))(data["eps_c"], data["eps_v"])


def window_band_energies(data):
    """The operator's skinny window-energy tables on the host, pads dropped."""
    def host(a):
        if getattr(a, "is_fully_addressable", True):
            return np.asarray(a)
        return np.asarray(a.addressable_data(0))          # replicated table
    nc, nv = int(data["n_cond"]), int(data["n_val"])
    ec, ev = host(data["eps_c"])[:, :nc], host(data["eps_v"])[:, :nv]
    return ec.real.astype(np.float64), ev.real.astype(np.float64)


def build_dipole_vector_bse(d_alpha, n_cond_pad=None, n_val_pad=None):
    """Reshape (3, nk, nc, nv) → (3, nc_pad, nv_pad, nk) BSE block-vector form.

    Pads ``nc → nc_pad`` and ``nv → nv_pad`` with zeros if requested (the
    BSE sharded matvec requires divisibility by the (x, y) mesh shape).
    """
    npol, nk, nc, nv = d_alpha.shape
    nc_pad = nc if n_cond_pad is None else int(n_cond_pad)
    nv_pad = nv if n_val_pad is None else int(n_val_pad)
    if nc_pad < nc or nv_pad < nv:
        raise ValueError("dipole pair carrier cannot be smaller than its input")
    if _is_jax_array(d_alpha):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        sh = d_alpha.sharding
        out = (NamedSharding(sh.mesh, P(None, "x", "y", None))
               if isinstance(sh, NamedSharding) else sh)
        return jax.jit(lambda d: jnp.pad(d.transpose(0, 2, 3, 1),
            ((0, 0), (0, nc_pad - nc), (0, nv_pad - nv), (0, 0))),
            out_shardings=out)(d_alpha)
    out = np.zeros((npol, nc_pad, nv_pad, nk), dtype=d_alpha.dtype)
    out[:, :nc, :nv, :] = np.transpose(d_alpha, (0, 2, 3, 1))
    return out


#: Elements of one (n_omega, states) Lorentzian block: 2^24 float64 = 128 MiB.
LORENTZ_BLOCK_ELEMENTS = 1 << 24


def lorentzian_broaden(omegas, energies, weights, eta):
    """ε(ω) = Σ_i weights[i] · η/π / ((ω - E_i)² + η²).

    ``weights`` is ``(n_states,)`` or ``(n_states, m)`` (one column per
    polarization), so one Lorentzian serves every column.  The
    ``(n_omega, n_states)`` kernel is formed in state blocks of at most
    :data:`LORENTZ_BLOCK_ELEMENTS`: a whole kernel is ``n_omega`` x the number of
    transitions (4001 x 32768 = 1.0 GB on Si 8^3; 4000 x 1e7 = 320 GB at a
    production window), which is host memory nothing priced.
    """
    omegas = np.asarray(omegas)
    energies = np.asarray(energies).reshape(-1)
    weights = np.asarray(weights)
    out = np.zeros((omegas.size,) + weights.shape[1:],
                   dtype=np.result_type(weights.dtype, np.float64))
    step = max(1, LORENTZ_BLOCK_ELEMENTS // max(1, omegas.size))
    for i in range(0, energies.size, step):
        delta = omegas[:, None] - energies[None, i:i + step]
        out += ((eta / np.pi) / (delta * delta + eta * eta)) @ weights[i:i + step]
    return out


def polarization_directions(polarization, wfn_path):
    """``(directions, labels)``: the light polarizations a spectrum is computed for.

    ``polarization`` unset: the three Cartesian axes x, y, z (rows of the
    identity), labelled ``b1 b2 b3`` as the per-direction files have always been.
    ``polarization = p1 p2 p3`` (BerkeleyGW's keyword): components along the
    reciprocal lattice vectors b1 b2 b3.  The Cartesian vector Σ_i p_i b_i is
    built first and normalized; the crystal coefficients themselves are never
    normalized (on a non-orthogonal lattice that is a different direction and
    length).  ``directions`` is ``(n_pol, 3)``, real, unit rows, Cartesian.
    """
    if polarization is None:
        return np.eye(3), ("b1", "b2", "b3")
    p = np.asarray(polarization, dtype=np.float64).reshape(-1)
    if p.size != 3 or not np.all(np.isfinite(p)) or not np.any(p):
        raise ValueError(f"polarization takes three components p1 p2 p3 along b1 b2 b3, "
                         f"not all zero; got {polarization!r}")
    import h5py
    with h5py.File(wfn_path, "r") as f:
        bvec = np.asarray(f["mf_header/crystal/bvec"][()], dtype=np.float64)  # rows b_i
    e = p @ bvec
    return (e / np.linalg.norm(e))[None, :], ("e",)


def project_dipoles(d_alpha, directions):
    """``ê · d`` for each direction: ``d_alpha`` (3, ...) Cartesian -> (n_pol, ...)."""
    if _is_jax_array(d_alpha):
        import jax.numpy as jnp
        return jnp.tensordot(jnp.asarray(directions, dtype=jnp.float64), d_alpha, axes=(1, 0))
    return np.tensordot(np.asarray(directions, dtype=np.float64), np.asarray(d_alpha), axes=(1, 0))


def broaden(omegas, centers, weights, width, kind="lorentzian"):
    """Σ_i w_i δ(ω - c_i) with a Lorentzian (half width ``width``) or a Gaussian
    (standard deviation ``width``) of unit area.  ``weights`` (n_centers, n_cols)
    -> (n_omega, n_cols); all energies in one unit."""
    if kind == "lorentzian":
        return lorentzian_broaden(omegas, centers, weights, width)
    if kind != "gaussian":
        raise ValueError(f"broadening {kind!r}: want lorentzian or gaussian")
    om = np.asarray(omegas, dtype=np.float64)[:, None]
    c = np.asarray(centers, dtype=np.float64).reshape(-1)[None, :]
    g = np.exp(-0.5 * ((om - c) / width) ** 2) / (np.sqrt(2.0 * np.pi) * width)
    return g @ np.asarray(weights, dtype=np.float64).reshape(c.shape[1], -1)


@lru_cache(maxsize=16)
def _transition_spectrum_program(mesh, kind):
    """Sum broadened transitions locally; only the spectrum is reduced."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    def local(d, de, omega, width):
        live = jnp.isfinite(de)
        centers = jnp.where(live, de, 0.)
        weights = jnp.abs(d) ** 2
        def one(_, om):
            distance = om - centers
            if kind == "lorentzian":
                kernel = width / (jnp.pi * (distance ** 2 + width ** 2))
            else:
                kernel = jnp.exp(-.5 * (distance / width) ** 2) / (jnp.sqrt(2 * jnp.pi) * width)
            kernel = jnp.where(live, kernel, 0.)
            values = jnp.concatenate([jnp.sum(weights * kernel[None], axis=(1, 2, 3)),
                                      jnp.sum(kernel)[None]])
            return None, values
        return jax.lax.psum(jax.lax.scan(one, None, omega)[1], ("x", "y"))
    program = jax.shard_map(local, mesh=mesh,
        in_specs=(P(None, None, "x", "y"), P(None, "x", "y"), P(), P()),
        out_specs=P(), check_vma=False)
    return jax.jit(program, out_shardings=NamedSharding(mesh, P()))


def _transition_spectrum(d_alpha, de_cv, omegas, width, kind):
    """Broaden the same physical transitions without gathering a CV bank."""
    from common.collectives import gather_to_host
    if kind not in ("lorentzian", "gaussian"):
        raise ValueError(f"broadening {kind!r}: want lorentzian or gaussian")
    if _is_jax_array(d_alpha) or _is_jax_array(de_cv):
        if (not (_is_jax_array(d_alpha) and _is_jax_array(de_cv))
                or not hasattr(d_alpha.sharding, "mesh")
                or not hasattr(de_cv.sharding, "mesh")):
            raise ValueError("transition spectra require both host arrays or both distributed device fields")
        import jax.numpy as jnp
        mesh = d_alpha.sharding.mesh
        return np.asarray(gather_to_host(_transition_spectrum_program(mesh, kind)(
            d_alpha, de_cv, jnp.asarray(omegas), jnp.asarray(width))))
    de = np.asarray(de_cv).reshape(-1)
    live = np.isfinite(de)
    weights = np.abs(np.asarray(d_alpha)).reshape(d_alpha.shape[0], -1)[:, live].T ** 2
    return broaden(omegas, de[live], np.column_stack([weights, np.ones(live.sum())]), width, kind)


def write_state_spectra(*, energies_Ry, exciton_dipoles, d_alpha, de_cv,
                        directions, labels, V_cell, nk, n_spinor, width_eV, kind,
                        omegas_eV, no_eps1, prefix="absorption"):
    """BerkeleyGW's ``absorption_eh.dat`` and ``absorption_noeh.dat`` from solved states.

    ε₂(ω) = 16π²/(V N_k n_spinor) Σ_S |ê·⟨0|r|S⟩|² δ(ω − Ω_S) over the computed
    states (eh) and Σ_cvk |ê·d_cvk|² δ(ω − ΔE_cvk) over the window's
    transitions (noeh), the prefactor ``jdos_from_transitions`` and the Haydock
    route use; δ is a Gaussian of standard deviation ``width_eV`` or a
    Lorentzian of half width ``width_eV`` (``kind``).  ε₁ by Kramers-Kronig on
    the grid.  Column 4 is the density of the computed exciton states (eh) or of
    the transitions (noeh), per eV.  One pair of files per direction:
    ``<prefix>_eh.dat`` for a deck ``polarization``, ``<prefix>_b1_eh.dat`` ..
    ``_b3`` for the three axes.  The eh spectrum holds only the computed
    states: it is complete below the highest of them.  Rank 0 writes; the
    paths are returned on every rank.
    """
    import jax
    omegas_eV = np.asarray(omegas_eV, dtype=np.float64)
    pref = 16.0 * np.pi ** 2 / (V_cell * nk * n_spinor)
    om_Ry, w_Ry = omegas_eV / RYD2EV, float(width_eV) / RYD2EV
    e_S = np.asarray(energies_Ry, dtype=np.float64).reshape(-1)
    D = project_dipoles(np.asarray(exciton_dipoles).T, directions)          # (npol, nS)
    eps2_eh = pref * broaden(om_Ry, e_S, np.abs(D.T) ** 2, w_Ry, kind)       # (nw, npol)
    dos_eh = broaden(omegas_eV, e_S * RYD2EV, np.ones((e_S.size, 1)), width_eV, kind)[:, 0]
    transitions = _transition_spectrum(project_dipoles(d_alpha, directions),
                                        de_cv, om_Ry, w_Ry, kind)
    eps2_noeh = pref * transitions[:, :-1]
    jdos = transitions[:, -1] / RYD2EV
    complete = float(e_S.max() * RYD2EV) if e_S.size else float("nan")
    paths = []
    for col, label in enumerate(labels):
        for tag, eps2, col4 in (("eh", eps2_eh, dos_eh), ("noeh", eps2_noeh, jdos)):
            name = f"{prefix}_{tag}.dat" if label == "e" else f"{prefix}_{label}_{tag}.dat"
            paths.append(name)
            if jax.process_index() != 0:
                continue
            e2 = eps2[:, col]
            e1 = np.ones_like(e2) if no_eps1 else kramers_kronig_eps1(om_Ry, e2)
            # BerkeleyGW's header exactly (a reader may skip four lines); what
            # LORRAX used to add there is logged instead.
            e = np.asarray(directions[col], dtype=np.float64)
            write_absorption_dat(name, omegas_eV, e2, e1, col4)
            log.info(
                f"{name}: polarization {label} = Cartesian ({e[0]:.6f}, {e[1]:.6f}, "
                f"{e[2]:.6f}); {kind} broadening {float(width_eV):.4f} eV; V_cell = "
                f"{V_cell:.6f} bohr^3, N_k = {nk}, n_spinor = {n_spinor}; "
                + (f"{e_S.size} computed states, complete below {complete:.4f} eV"
                   if tag == "eh" else "window transitions"))
    return paths


def jdos_from_transitions(d_alpha, de_cv, omegas_Ry, V_cell, n_k, eta_Ry,
                          n_spin, n_spinor):
    """Independent-particle ε₂⁰(ω) — the 4th column of BGW's absorption_*.dat.

    THIS IS THE SINGLE SITE, for the same reason
    :func:`exciton_dipole_projections` is: ``absorption_haydock`` and a second
    absorption driver (the eigenvector route, since deleted) each carried
    their own copy of this formula with the arguments in a different order,
    and one of them inlined the Lorentzian instead of calling
    :func:`lorentzian_broaden`.

    ``f^α_{cvk} = |d^α_{cvk}|²`` at transition energies ``ΔE_cv``, broadened
    with the same prefactor ``16π²/(V·N_k·n_spin·n_spinor)`` the interacting
    ε₂ uses.  Returns ``(n_omega, 3)``.
    """
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    return pref * _transition_spectrum(d_alpha, de_cv, omegas_Ry,
                                       eta_Ry, "lorentzian")[:, :-1]


def kramers_kronig_eps1(omegas, eps2):
    """Naive principal-value KK: ε₁(ω) = 1 + (2/π) P ∫ ω' ε₂(ω')/(ω'² - ω²) dω'.

    O(N²) over uniform omega grid. For ~1500 points this is sub-second.
    """
    omegas = np.asarray(omegas)
    eps2 = np.asarray(eps2)
    n = omegas.size
    if n < 2:
        return np.ones_like(eps2)
    do = omegas[1] - omegas[0]
    eps1 = np.ones_like(eps2)
    for i in range(n):
        mask = np.arange(n) != i
        kernel = omegas[mask] * eps2[mask] / (omegas[mask] ** 2 - omegas[i] ** 2)
        eps1[i] = 1.0 + (2.0 / np.pi) * np.sum(kernel) * do
    return eps1


def write_absorption_dat(path, omegas_eV, eps2, eps1, jdos):
    """BerkeleyGW's ``absorption_*.dat``, byte for byte in layout.

    ``BSE/absp.f90``: four list-directed header lines (``write(10,*)``, hence
    the leading blank) and rows ``100 format(4f16.9)``: ω (eV), ε₂(ω), ε₁(ω),
    JDOS(ω).  Nothing else, so a reader that skips four lines reads it.
    """
    with open(str(path), "w") as f:
        f.write(" # Column 1: omega\n")
        f.write(" # Column 2: eps2(omega)\n")
        f.write(" # Column 3: eps1(omega)\n")
        f.write(" # Column 4: JDOS(omega)\n")
        for o, e2, e1, jd in zip(omegas_eV, eps2, eps1, jdos):
            f.write(f"{o:16.9f}{e2:16.9f}{e1:16.9f}{jd:16.9f}\n")


def fortran_e(value, width: int, digits: int) -> str:
    """``value`` as Fortran ``Ew.d`` writes it (gfortran): ``-`` only when
    negative, ``0.``, ``digits`` digits, ``E`` and a signed two-digit exponent
    (``+ddd`` without the ``E`` past 99), right-justified in ``width``."""
    x = float(value)
    if not np.isfinite(x):
        return ("NaN" if np.isnan(x) else "-Infinity" if x < 0 else
                "Infinity").rjust(width)
    if x == 0.0:
        mantissa, exponent = "0" * int(digits), 0
    else:
        head, tail = f"{abs(x):.{int(digits) - 1}e}".split("e")
        mantissa, exponent = head.replace(".", ""), int(tail) + 1
    sign = "-" if np.signbit(x) else ""
    power = f"E{exponent:+03d}" if abs(exponent) <= 99 else f"{exponent:+04d}"
    return f"{sign}0.{mantissa}{power}".rjust(int(width))


#: BerkeleyGW's file per polarization (``BSE/absp_io.f90``, ``suffix``): one
#: polarization writes ``eigenvalues.dat``, three write ``eigenvalues_b1.dat``
#: to ``_b3.dat``, the Cartesian x, y and z (``BSE/vmtxel.f90``).
BGW_POLARIZATION_SUFFIXES = ("b1", "b2", "b3")


def bgw_eigenvalue_lines(eigvals_eV, dipoles_pol=None, *, n_spin: int,
                         n_spinor: int, vol_supercell: float,
                         triplet: bool = False) -> list[str]:
    """The lines of one BerkeleyGW ``eigenvalues*.dat`` (TDA, complex flavor).

    BerkeleyGW ``BSE/absp_io.f90`` ``write_eigenvalues``::

        write(14,'(a,i8)') '# neig  = ', neig
        write(14,'(a,e16.9)') '# vol   = ', vol
        write(14,'(a,2i8)') '# nspin, nspinor = ', xct%nspin, xct%nspinor
        '#       eig (eV)   abs(dipole)^2' // '    ' // '  Re(dipole)    Im(dipole)'
        write(14,'(4e16.8)') evals(ii), cs(ii,ipol), dipoles_r(ii,ipol)

    The triplet kernel writes `` mg `` in place of the four blanks (``triplet``)
    and the same dipoles: BerkeleyGW computes them as for the singlet.

    ``dipoles_pol`` (``(N,)`` complex) is ``D = <0|r|S>`` along this file's
    polarization; ``vol_supercell`` is ``N_k V_cell`` (bohr^3, BerkeleyGW's
    ``xct%nktotal * crys%celvol``).  Without dipoles the rows carry the
    energy column alone, under ``#       eig (eV)``.
    """
    energies = np.asarray(eigvals_eV, dtype=np.float64).reshape(-1)
    lines = [f"# neig  = {energies.size:8d}",
             f"# vol   = {fortran_e(vol_supercell, 16, 9)}",
             f"# nspin, nspinor = {int(n_spin):8d}{int(n_spinor):8d}"]
    if dipoles_pol is None:
        lines.append("#       eig (eV)")
        lines += [fortran_e(e, 16, 8) for e in energies]
        return lines
    d = np.asarray(dipoles_pol, dtype=np.complex128).reshape(-1)
    if d.size != energies.size:
        raise ValueError(f"bgw_eigenvalue_lines: {energies.size} energies for "
                         f"{d.size} dipoles")
    lines.append("#       eig (eV)   abs(dipole)^2" + (" mg " if triplet else "    ")
                 + "  Re(dipole)    Im(dipole)")
    lines += ["".join(fortran_e(v, 16, 8) for v in
                      (e, float(np.abs(di) ** 2), di.real, di.imag))
              for e, di in zip(energies, d)]
    return lines


def write_eigenvalues_dat(
    path, eigvals_eV, dipoles_pol=None, *, n_spin, n_spinor, vol_supercell,
    triplet=False,
):
    """One BerkeleyGW ``eigenvalues*.dat`` at ``path``
    (:func:`bgw_eigenvalue_lines`)."""
    with open(str(path), "w") as f:
        f.write("\n".join(bgw_eigenvalue_lines(
            eigvals_eV, dipoles_pol, n_spin=n_spin, n_spinor=n_spinor,
            vol_supercell=vol_supercell, triplet=triplet)) + "\n")


def eigenvalue_file_names(labels) -> list[str]:
    """BerkeleyGW's names per polarization (``BSE/absp_io.f90``): one
    polarization (a deck ``polarization``, label ``e``) is ``eigenvalues.dat``,
    the three axes ``eigenvalues_b1.dat`` .. ``_b3.dat``."""
    return (["eigenvalues.dat"] if tuple(labels) == ("e",) else
            [f"eigenvalues_{label}.dat" for label in labels])


def write_bgw_eigenvalues(directory, eigvals_eV, dipoles=None, *, n_spin,
                          n_spinor, vol_supercell, directions=None, labels=None,
                          triplet=False) -> list[str]:
    """Every root in BerkeleyGW's own files; returns the paths written.

    ``dipoles`` ``(N, 3)`` (``<0|r_a|S>``, Cartesian) are projected on each of
    ``directions`` (``polarization_directions``; default the three axes), one
    file per direction named by :func:`eigenvalue_file_names`, the column
    BerkeleyGW's ``dipoles_r = Σ A conj(ê·s)``.  Without dipoles one
    ``eigenvalues.dat`` lists the energies.
    """
    directory = Path(directory)
    if dipoles is None:
        jobs = [("eigenvalues.dat", None)]
    else:
        if directions is None:
            directions, labels = np.eye(3), BGW_POLARIZATION_SUFFIXES
        d = project_dipoles(np.asarray(dipoles, dtype=np.complex128).T, directions)
        jobs = list(zip(eigenvalue_file_names(labels), d))
    paths = []
    for name, d_pol in jobs:
        write_eigenvalues_dat(directory / name, eigvals_eV, d_pol,
                              n_spin=n_spin, n_spinor=n_spinor,
                              vol_supercell=vol_supercell, triplet=triplet)
        paths.append(str(directory / name))
    return paths


def _noeh_header(nk, nc, nv, n_spin, n_spinor, vol_supercell):
    return [f"# neig  = {nk * nc * nv:8d}",
            f"# vol   = {fortran_e(vol_supercell, 16, 9)}",
            "# nspin, nspinor, nkpt, ncb, nvb = "
            + "".join(f"{int(x):8d}" for x in (n_spin, n_spinor, nk, nc, nv)),
            "#   ik    ic    iv    is         ec (eV)         ev (eV)        eig (eV)"
            "   abs(dipole)^2      Re(dipole)      Im(dipole)"]


def _noeh_record(ik, ic, iv, c, v, d):
    return f"{ik + 1:6d}{ic + 1:6d}{iv + 1:6d}{1:6d}" + "".join(
        fortran_e(x, 16, 8) for x in (c, v, c - v, float(abs(d) ** 2), d.real, d.imag))


def bgw_noeh_lines(ec_eV, ev_eV, d_pol, *, n_spin: int, n_spinor: int,
                   vol_supercell: float) -> list[str]:
    """The lines of one BerkeleyGW ``eigenvalues*_noeh.dat`` (complex flavor).

    BerkeleyGW ``BSE/absp_io.f90`` ``write_eigenvalues_noeh``::

        write(10,'(a,i8)') '# neig  = ', neig
        write(10,'(a,e16.9)') '# vol   = ', vol
        write(10,'(a,5i8)') '# nspin, nspinor, nkpt, ncb, nvb = ', ...
        '#   ik    ic    iv    is         ec (eV)         ev (eV)        eig (eV)'
        // '   abs(dipole)^2      Re(dipole)      Im(dipole)'
        do ik; do ic; do iv; do is
          write(10,'(4i6,6e16.8)') ik, ic, iv, is, ec, ev, ec - ev, |s0|^2, s0

    ``ec_eV`` ``(nk, nc)`` and ``ev_eV`` ``(nk, nv)`` are the operator's window
    energies (QP where corrected) with the valence axis in LORRAX's ascending
    order; BerkeleyGW counts ``iv`` down from the VBM, so ``iv = 1`` is the
    last LORRAX valence band.  ``d_pol`` ``(nk, nc, nv)`` is ``ê·d`` of the
    window dipoles ``⟨ck|v̂|vk⟩/ΔE``, BerkeleyGW's ``s0``.  One spin channel
    (``is = 1``); ``vol_supercell`` is ``N_k V_cell`` as in ``eigenvalues*.dat``.
    """
    ec = np.asarray(ec_eV, dtype=np.float64)
    ev = np.asarray(ev_eV, dtype=np.float64)
    d = np.asarray(d_pol, dtype=np.complex128)
    nk, nc = ec.shape
    nv = int(ev.shape[1])
    if d.shape != (nk, nc, nv):
        raise ValueError(f"bgw_noeh_lines: dipoles {d.shape} for window ({nk}, {nc}, {nv})")
    lines = _noeh_header(nk, nc, nv, n_spin, n_spinor, vol_supercell)
    for ik in range(nk):
        for ic in range(nc):
            for iv in range(nv):
                c, v, s = ec[ik, ic], ev[ik, nv - 1 - iv], d[ik, ic, nv - 1 - iv]
                lines.append(_noeh_record(ik, ic, iv, c, v, s))
    return lines


def write_bgw_eigenvalues_noeh(directory, ec_eV, ev_eV, d_alpha, *, n_spin,
                               n_spinor, vol_supercell, directions=None,
                               labels=None) -> list[str]:
    """BerkeleyGW's non-interacting transitions, one file per polarization:
    ``eigenvalues_noeh.dat`` for a deck ``polarization`` (label ``e``),
    ``eigenvalues_b1_noeh.dat`` .. ``_b3`` for the three axes
    (:func:`bgw_noeh_lines`).  ``d_alpha`` ``(3, nk, nc, nv)`` Cartesian.
    Returns the paths written."""
    from contextlib import ExitStack
    from common.collectives import gather_to_host, rank0_transaction
    directory = Path(directory)
    if directions is None:
        directions, labels = np.eye(3), BGW_POLARIZATION_SUFFIXES
    d = project_dipoles(d_alpha, directions)
    ec, ev = np.asarray(ec_eV), np.asarray(ev_eV)
    nk, nc, nv = int(ec.shape[0]), int(ec.shape[1]), int(ev.shape[1])
    paths = [str(directory / name.replace(".dat", "_noeh.dat"))
             for name in eigenvalue_file_names(labels)]
    stack, files = ExitStack(), []
    def open_files():
        for path in paths:
            f = stack.enter_context(open(path, "w"))
            files.append(f)
            f.write("\n".join(_noeh_header(nk, nc, nv, n_spin, n_spinor, vol_supercell)) + "\n")
    try:
        rank0_transaction(directory, stage="bse.noeh.open", write=open_files)
        # At most 1024 transitions (48 KiB for three polarizations) leave
        # the device at once. Every rank participates; only rank 0 writes.
        for start in range(0, nk * nc * nv, 1024):
            stop = min(start + 1024, nk * nc * nv)
            if _is_jax_array(d):
                values = gather_to_host(_noeh_block_program(d.sharding.mesh, nc, nv)(
                    d, np.arange(start, stop, dtype=np.int32)))
            else:
                idx = np.arange(start, stop)
                values = d[:, idx // (nc * nv), idx // nv % nc, nv - 1 - idx % nv]
            def write_block():
                for k, flat in enumerate(range(start, stop)):
                    ik, ic, iv = flat // (nc * nv), flat // nv % nc, flat % nv
                    for f, value in zip(files, values[:, k]):
                        f.write(_noeh_record(ik, ic, iv, ec[ik, ic], ev[ik, nv - 1 - iv], value) + "\n")
            rank0_transaction(directory, stage="bse.noeh.block", write=write_block)
    finally:
        rank0_transaction(directory, stage="bse.noeh.close", write=stack.close)
    return paths


@lru_cache(maxsize=16)
def _noeh_block_program(mesh, nc, nv):
    """Reduce a requested small text block from its owned CV tiles."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    def local(d, idx):
        ik, ic, iv = idx // (nc * nv), idx // nv % nc, nv - 1 - idx % nv
        c0 = jax.lax.axis_index("x") * d.shape[2]
        v0 = jax.lax.axis_index("y") * d.shape[3]
        owned = (ic >= c0) & (ic < c0 + d.shape[2]) & (iv >= v0) & (iv < v0 + d.shape[3])
        values = d[:, ik, jnp.clip(ic - c0, 0, d.shape[2] - 1),
                   jnp.clip(iv - v0, 0, d.shape[3] - 1)]
        return jax.lax.psum(jnp.where(owned[None], values, 0.), ("x", "y"))
    program = jax.shard_map(local, mesh=mesh,
        in_specs=(P(None, None, "x", "y"), P()), out_specs=P(), check_vma=False)
    return jax.jit(program, out_shardings=NamedSharding(mesh, P()))


def write_absorption_h5(
    path,
    *,
    omegas_eV,
    eps2_3pol,
    eps1_3pol,
    jdos_3pol,
    eigenvalues_Ry: Optional[np.ndarray] = None,
    oscillator_strengths: Optional[np.ndarray] = None,
    haydock_alphas: Optional[np.ndarray] = None,
    haydock_betas: Optional[np.ndarray] = None,
    haydock_norms: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
):
    """Combined H5: ω grid, 3-polarisation ε₂/ε₁/JDOS, plus route artifacts."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset("omega_eV", data=omegas_eV)
        f.create_dataset("eps2", data=eps2_3pol)   # (n_omega, 3)
        f.create_dataset("eps1", data=eps1_3pol)
        f.create_dataset("jdos", data=jdos_3pol)
        if eigenvalues_Ry is not None:
            f.create_dataset("eigenvalues_Ry", data=eigenvalues_Ry)
        if oscillator_strengths is not None:
            f.create_dataset("oscillator_strengths", data=oscillator_strengths)  # (N, 3)
        if haydock_alphas is not None:
            f.create_dataset("haydock_alphas", data=haydock_alphas)  # (3, n_iter)
        if haydock_betas is not None:
            f.create_dataset("haydock_betas", data=haydock_betas)
        if haydock_norms is not None:
            f.create_dataset("haydock_norms", data=haydock_norms)   # (3,)
        if metadata:
            for k, v in metadata.items():
                f.attrs[k] = v


__all__ = [
    "RYD2EV",
    "exciton_dipole_projections",
    "exciton_dipoles_distributed",
    "eps2_from_exciton_dipoles",
    "load_eigenvectors_h5",
    "load_exciton_dipoles_h5",
    "slice_dipole_to_bse_window",
    "window_dipoles",
    "build_dipole_vector_bse",
    "lorentzian_broaden",
    "jdos_from_transitions",
    "kramers_kronig_eps1",
    "write_absorption_dat",
    "write_absorption_h5",
    "BGW_POLARIZATION_SUFFIXES",
    "bgw_eigenvalue_lines",
    "fortran_e",
    "require_dipole_not_short",
    "write_bgw_eigenvalues",
    "write_eigenvalues_dat",
]
