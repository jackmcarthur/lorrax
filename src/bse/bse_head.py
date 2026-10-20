"""The q=0 Coulomb head: one resolution of its scalars, one injection of it.

AUTHORITY RULE — ``vhead`` goes on the exchange tile unconditionally, and
``whead`` enters the direct term (as the exact diagonal below) only when a
real screened ``W0`` was loaded.
``compute_vcoul`` zeroes ``v(G=G'=0)`` at q=0 before the Dyson solve, so the
q=0 exchange tile is bare Coulomb either way and reinstating the mini-BZ
average is always right.  ``whead`` is the head of the SCREENED interaction.
The restart loaders no longer fall back to bare ``V`` for ``W``: a restart
without a ready ``W0_qmunu`` requests the static response owner through
``bse_loading``; ``read_bse_payload`` still refuses a missing screened body.
``_inject_q0_head`` still owns the ``w0_ready`` gate in ONE spelling, so a
screened head can never reach an unscreened tile by any other path.

DEFERRAL IS THE SECOND, SEPARATE QUESTION, and it is deliberately not spelled
as the first.  When a coarse→fine densification is pending, W's head belongs on
the tile but not yet and not as a Kronecker delta, so ``defer_whead`` skips it
here and ``bse_densify`` re-attaches an analytic per-fine-q head instead.  That
is a different statement from ``w0_ready = False`` ("this W is bare V, a
screened head does not belong on it at all"), and collapsing the two would put
the wrong reason in the log on a perfectly screened tile.

THE SCREENED q=0 HEAD IS EXACT, NOT FITTED.  At q = 0, G = 0 the direct
kernel's pair densities are band overlaps, M_nn'(k,k;G=0) = <nk|n'k> =
delta_nn', so the head term is
``K^d_head(kcv, k'c'v') = (whead / (Omega N_k)) delta_kk' delta_cc' delta_vv'``:
a constant on the transition diagonal of the A block, and nothing in the
coupling block (there the overlaps are <ck|v'k> = 0).  The ISDF fit
``sum_mu zeta_mu(G=0) psi*_n(r_mu) psi_n'(r_mu)`` is not exactly orthonormal
(MoS2 12x12, N_mu 1198: content 1.008, a 4.3 meV rigid over-binding), so
``whead`` no longer goes on the W tile as the rank-1 ``conj(g0) g0``.
``_inject_q0_head`` returns the scalar ``q0_head_ry`` instead, the bundle
carries it under that key, and every operator subtracts it from its
conduction diagonal through :func:`head_shifted_eps_c`.  ``vhead`` on the
exchange tile is unchanged.

The provenance of the scalars is the module's other job.  ``vhead`` and
``whead_0freq`` are cross-code validation knobs — they pin the head to an
externally supplied value, typically BerkeleyGW's — so the deck's value beats
the restart's, a malformed value refuses rather than falling back, and every
injection logs which channel each number came from.
"""
from __future__ import annotations

import os
from typing import Optional

import jax.numpy as jnp

from common.units import RYD_TO_EV
from .bse_window import _parse_wfn_path


def _parse_head_overrides(input_file: Optional[str]):
    """Extract ``vhead`` and ``whead_0freq`` overrides from cohsex.in.

    Returns ``(vhead, whead_0freq)`` where each is ``complex`` or ``None``
    if the key is absent / blank. These take precedence over any
    restart-file head values when assembling the q=0 rank-1 update.

    These are the SAME two deck keys the GW side reads
    (``gw_config.HeadConfig.vhead`` / ``.whead_0freq``), so this reader
    holds itself to the GW reader's parsing contract: the two sides agree
    on any deck or neither does, which
    ``tests/test_bse_head_override.py`` asserts against ``gw_config``'s
    own reader deck by deck.  Two rules follow:

    * **Inline comments are stripped**, because ``gw_config`` builds its
      ``ConfigParser`` with ``inline_comment_prefixes=('#',)``.  A deck
      line written ``vhead = 3303.748102  # BGW`` would otherwise pin Σ's
      head and not the BSE's — one run screening with two q=0 heads.
    * **A malformed value refuses**, naming the file, line and key,
      rather than falling back to the restart.  A validation knob that
      no-ops without a word is worse than no knob at all.  That refusal
      is also why this reader is not simply ``read_lorrax_input``, whose
      ValueError names neither the key nor the line
      (``tests/known_failures/2026-08-11-bse-head-deck-reader-duplication.md``).
    """
    if input_file is None or not os.path.isfile(input_file):
        return None, None
    vhead = None
    whead0 = None
    with open(input_file) as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key = key.strip().lower()
            if key not in ("vhead", "whead_0freq"):
                continue
            val = val.partition("#")[0].strip()   # GW-parity: inline comment
            if not val:
                continue
            try:
                parsed = complex(float(val))
            except ValueError:
                raise ValueError(
                    f"{input_file}:{lineno}: q=0 head override {key!r} has a "
                    f"non-numeric value {val!r}.  This key pins the q=0 "
                    f"Coulomb head to an externally supplied scalar (e.g. "
                    f"BerkeleyGW's) and is read by BOTH the GW and the BSE "
                    f"side; refusing rather than silently falling back to the "
                    f"restart's own head, which would leave the two sides "
                    f"screening with different q=0 heads."
                ) from None
            if key == "vhead":
                vhead = parsed
            else:
                whead0 = parsed
    return vhead, whead0


def _resolve_head_params(
    input_file: Optional[str],
    vhead_restart,
    whead_restart,
    cell_volume: Optional[float] = None,
):
    """Resolve ``(vhead, whead, cell_volume, source)`` for q=0 head injection.

    cohsex.in ``vhead``/``whead_0freq`` overrides take precedence over the
    restart-file head values; ``cell_volume`` (Bohr³) is pulled from the WFN
    when not supplied.  Any of the first three may return ``None`` (head
    skipped).  Single-sourced by both the sharded and single-device restart
    loaders.

    ``source`` is a short human-readable provenance string ("deck" /
    "restart" / "none" per channel).  It exists because the override is a
    cross-code validation knob: the one question a reader of the log has
    is whether the head that was actually injected is the deck's pinned
    external value or the run's own computed one, and printing the value
    alone does not answer it (the two agree to 8 digits on the anchor
    deck by construction, which is exactly when confusing them is easy).
    """
    vhead_in, whead0_in = _parse_head_overrides(input_file)
    vhead = vhead_in if vhead_in is not None else vhead_restart
    if whead0_in is not None:
        whead = jnp.asarray([whead0_in], dtype=jnp.complex128)
    else:
        whead = whead_restart
    source = "v:{}, w:{}".format(
        "deck" if vhead_in is not None else
        ("restart" if vhead_restart is not None else "none"),
        "deck" if whead0_in is not None else
        ("restart" if whead_restart is not None else "none"),
    )
    if cell_volume is None and input_file is not None:
        try:
            from ffi import _services
            _services.ensure_on_path()
            from wfn_loader import WfnLoader
            cell_volume = float(WfnLoader(_parse_wfn_path(input_file)).cell_volume)
        except Exception as exc:
            print(f"BSE head: cell_volume unresolved ({exc}); skipping head")
            cell_volume = None
    return vhead, whead, cell_volume, source


def q0_head_diagonal(whead, cell_volume: float, nk: int) -> float:
    """The exact q=0 screened head as a transition-diagonal constant, in Ry.

    ``Delta_0 = Re(whead) / (Omega N_k)``: with the exact overlaps
    ``<nk|n'k> = delta_nn'`` the q=0, G=G'=0 term of the direct kernel is
    ``Delta_0 delta_kk' delta_cc' delta_vv'`` (module docstring), and
    ``H = D + V - W`` takes it as ``-Delta_0`` on the A-block diagonal.

    Parameters
    ----------
    whead : complex or float
        Screened q=0 head, Ry bohr^3 (the scalar the rank-1 ISDF head used,
        ``head_correction._head_rank1_scalars``).
    cell_volume : float
        Omega, bohr^3.
    nk : int
        Number of k points of the BSE grid the operator runs on.
    """
    return float(complex(whead).real) / (float(cell_volume) * int(nk))


def head_shifted_eps_c(eps_c, data):
    """Conduction diagonal of the BSE operator, ``eps_c - Delta_0``.

    The one place the exact q=0 head (:func:`q0_head_diagonal`) enters an
    operator: every matvec forms its D term as ``eps_c - eps_v``, so taking
    ``Delta_0`` off the conduction leg puts ``-Delta_0`` on every transition
    of the A block (and of the finite-Q pair ``eps_c(k+Q) - eps_v(k)``, whose
    q=0 direct term pairs ``<c k+Q|c' k+Q>`` with ``<v k|v' k>``).  The
    bundle's ``eps_c`` stays the QP/DFT energy; only operator operands are
    shifted.  A bundle without ``q0_head_ry`` (head-less loads, synthetic
    fixtures) returns ``eps_c`` itself.

    ``eps_c`` : (nk, n_c_pad) Ry, any sharding.  Returns the same shape.
    """
    shift = float(data.get("q0_head_ry", 0.0) or 0.0)
    return eps_c - shift if shift else eps_c


def _inject_q0_head(
    V_q0,
    W_q,
    g0_mu,
    g0_nu,
    vhead,
    whead,
    cell_volume: float,
    *,
    w0_ready: bool,
    nk: int,
    defer_whead: bool = False,
):
    """Inject the rank-1 q=0 ``vhead``; resolve ``whead`` to its exact diagonal.

    One authoritative spelling of the head injection for both restart
    loaders: the sharded one (:func:`load_bse_data_from_restart_sharded`)
    and the single-device full-file one (:func:`_load_ring_subset`).  The
    module docstring states why the gate is here and why ``defer_whead`` is
    kept distinct from it.

    Passing ``W_q=None`` into the injector (rather than dropping the returned
    array) is what makes the skip total: the helper then computes no ``w``
    scalar and touches no W element, so the returned tile is the input object
    and not a re-derived copy of it.

    Parameters
    ----------
    V_q0 : jax.Array
        ``(n_μ, n_ν)`` q=0 exchange tile.  Sharded ``P("x", "y")`` on the
        sharded path, unsharded on the single-device one.
    W_q : jax.Array
        ``(n_μ, n_ν, nkx, nky, nkz)`` screened tensor; returned unchanged.
        The screened head no longer goes on it (module docstring).
    g0_mu, g0_nu : jax.Array
        ``ζ(0, μ, G=0)`` on the μ- and ν-axis respectively (the same vector
        twice; two copies so the rank-1 update is process-local when the
        tensors are dual-sharded).
    vhead, whead, cell_volume :
        As resolved by :func:`_resolve_head_params` — Ry and Bohr³.
    w0_ready : bool
        Did a real screened ``W0`` load, or is ``W_q`` the bare-V fallback?
        False gives ``q0_head_ry = 0`` and logs ``whead=skipped``.
    nk : int
        Number of k points of the loaded grid (:func:`q0_head_diagonal`).
    defer_whead : bool
        A coarse→fine densification is pending and will re-attach W's head
        itself, per fine q (C1, :mod:`gw.head_densify`).  The W tile must then
        reach the densifier as the PRE-injection body, because what the
        densifier does to a Kronecker-delta head is exactly the defect C1
        exists to remove — so the head is not added here and
        :func:`_interpolate_bse_data_to_grid` adds the fine-grid one instead.
        Do NOT spell this as ``w0_ready=False``: that would put the wrong
        reason in the log on a perfectly screened tile.  ``vhead`` is
        unaffected either way — the q=0 exchange tile is k-grid invariant and
        is carried through the densifier unchanged, so its head is injected
        here as always.

    Returns
    -------
    tuple
        ``(V_q0, W_q, q0_head_ry, log_fragment)``: ``q0_head_ry`` is
        :func:`q0_head_diagonal` when ``w0_ready`` and not deferred, else 0;
        the fragment is the ``"vhead=…, whead=…"`` half of each loader's own
        log line.
    """
    from gw.head_correction import apply_q0_head_rank1_sharded

    apply_w = w0_ready and not defer_whead and whead is not None
    V_q0, _ = apply_q0_head_rank1_sharded(
        V_q0, None, g0_mu, g0_nu, vhead, None, cell_volume, omega_index=0)
    q0_head_ry = (q0_head_diagonal(jnp.asarray(whead).reshape(-1)[0],
                                   cell_volume, nk) if apply_w else 0.0)
    v_str = (f"vhead={complex(vhead).real:.6f}"
             if vhead is not None else "vhead=skipped")
    if whead is None or not w0_ready:
        w_str = "whead=skipped"
    elif defer_whead:
        w_str = (f"whead[0]={complex(whead[0]).real:.6f} DEFERRED to the "
                 f"coarse→fine densifier (C1)")
    else:
        w_str = (f"whead[0]={complex(whead[0]).real:.6f} as the exact-overlap "
                 f"diagonal {q0_head_ry * RYD_TO_EV * 1e3:.4f} meV")
    return V_q0, W_q, q0_head_ry, f"{v_str}, {w_str}"
