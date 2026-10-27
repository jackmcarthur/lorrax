"""The GN-PPM GW stage inventory: every stage's per-rank device bytes from the deck's sizes.

No device and no compile: a stage is the residents the driver holds when it opens
(``operands``) plus its own transient (``xla``), as :class:`lxkit.memory.Price` rows.
Transients come from the runtime's own price functions where one exists:

- ζ μ-batch loop: :func:`gw.gflat_memory_model.plan_zeta_route_g` (its smallest
  configuration; the loop fills the budget above it),
- V(q): :func:`gw.v_q_g_flat.vq_tile_bytes`,
- W Dyson: :func:`gw.w_isdf.dyson_route_prices` (the admission's two routes),
- Σ_x and Σ τ: :func:`gw.ppm_tau_kernel.sigma_row_plan` / :func:`sigma_pass_bytes`
  (the row passes the kernel books).

The rest are the arrays the stage allocates, read off ``LORRAX_MEMORY_TRACE`` on
MoS₂/hBN (148 atoms, spinor, N_μ 16320, P = 4) and cited per term.  The
calibration (predicted against measured stage peaks, MoS₂/hBN N_μ 3264-16320 at
P = 4 and Cr:Al₂O₃ N_μ 23700 at P = 16) is ``tests/test_stage_inventory_calibration.py``.

Units, per rank (complex128): ``M = 16 μ²/P`` one face tile, ``QM`` the stored q
wedge of them, ``F = 2·16·n_par·n_b·n_s·μ/P`` the two centroid faces ψ(μ),
``ΨG = 16·n_par·n_b·n_s·N_G/P`` the conj ψ(G) slice the ζ fit holds.
"""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

from lxkit import memory

C16 = 16.0
GB = 1e9
#: A100-80GB (docs/architecture/memory-model.md "The rule"): local and distributed budgets.
BUDGET = {"local": 66.35 * GB, "distributed": 63.4 * GB}
#: lxkit.legs CUDA distributed context (cuSOLVERMp/NCCL), outside the pool.
CONTEXT = 3.6 * GB
#: A100 opt-in shared memory per block: the mathdx mode-7 single arm (no run-time scratch).
OPTIN_SMEM = 166912


@dataclasses.dataclass(frozen=True)
class Deck:
    """The sizes a GN-PPM charge run allocates by (all counts as the run pads them).

    ``nk`` full-zone k; ``n_parent`` stored ψ parents (IBZ k); ``nq`` stored q wedge;
    ``nb`` sum bands (ζ fit, χ₀, Σ); ``nb_qp`` QP matrix bands (Σ's projection
    extent and the bands Σ Hartree boxes); ``ng`` ψ sphere
    (ngkmax); ``fft`` the density grid; ``nmu`` centroids; ``trs`` time reversal
    used (antiunitary Green rows in Σ).  ``hartree_unchunked``: the density scan
    before 2410aa812 boxed every local band at once (reports older than it).  ``sys_dim``
    2: a slab (a second V(q) stack, the long-range channel).  ``validation_width``: q per
    program of the all-q Coulomb validation (1, its floor; a run admits the widest that fits)."""
    nk: int
    n_parent: int
    nq: int
    ns: int
    nb: int
    nb_qp: int
    ng: int
    fft: tuple
    nmu: int
    kgrid: tuple = (1, 1, 1)
    trs: bool = True
    sys_dim: int = 3
    validation_width: int = 1
    hartree_unchunked: bool = False


def _pad(n: int, p: int) -> int:
    return -(-int(n) // int(p)) * int(p)


def _mesh(P: int) -> tuple[int, int]:
    px = int(math.isqrt(int(P)))
    return px, int(P) // px


def _units(d: Deck, P: int) -> dict:
    mu, nb = _pad(d.nmu, P), _pad(d.nb, P)
    M = C16 * mu * mu / P
    return dict(mu=mu, nb=nb, M=M, QM=d.nq * M, n2=C16 * mu * mu,
                F=2 * C16 * d.n_parent * nb * d.ns * mu / P,
                PsiG=C16 * d.n_parent * nb * d.ns * d.ng / P)


def _p(stage, operands, xla, src_ops, src_xla, ranks):
    return memory.Price(operands=memory.Term(float(operands), source=src_ops),
                        xla=memory.Term(float(xla), source=src_xla), ranks=int(ranks), label=stage)


def _sigma_new(d: Deck, P: int, *, static: bool) -> float:
    """The Σ node's new bytes (``ppm_tau_kernel.sigma_pass_bytes``), every local row a cut."""
    from gw.ppm_tau_kernel import sigma_pass_bytes, sigma_row_plan
    from ffi.fft import klead_unfold_scratch_bytes
    px, py = _mesh(P)
    u = _units(d, P)
    sizes = dict(nk=d.nk, n_parent=d.n_parent, n_w=d.nq, nb=u["nb"], nb_sig=_pad(d.nb_qp, P),
                 n_rmu=u["mu"], ns=d.ns, px=px, py=py,
                 partner=int(d.trs and not static), static=static)
    # Orbit cuts are a property of the centroid table; here every ``g``-th local row, at most
    # 256 cuts a rank (the window search is quadratic in the cut count).
    L = u["mu"] // px
    g = max(1, -(-L // 256))
    R, windows = sigma_row_plan(**sizes, cuts=lambda: range(g, L, g))
    price = sigma_pass_bytes(
        **sizes, R=R, passes=len(windows), sum_layout="axis", stores_rows_in_kernel=True,
        scratch=lambda rows: klead_unfold_scratch_bytes(d.kgrid, d.ns, rows * (u["mu"] // py),
                                                        optin=OPTIN_SMEM))
    return price["new"]


@lru_cache(maxsize=4096)
def zeta_plan(d: Deck, P: int):
    """The charge μ-batch fit's :class:`gw.gflat_memory_model.MuBatchPlan`, C on the face."""
    from types import SimpleNamespace
    from gw.gflat_memory_model import plan_zeta_route_g
    u = _units(d, P)
    meta = SimpleNamespace(nk_tot=d.nk, nspinor=d.ns, n_rmu_padded=u["mu"], n_rmu=u["mu"],
                           fft_grid=tuple(d.fft))
    r = (3.0 * d.ng / (4.0 * math.pi)) ** (1.0 / 3.0)
    n_a = max(d.fft)
    n_col = min(math.prod(d.fft) // n_a, math.ceil(1.2 * math.pi * r * r))
    with _quiet():
        z = plan_zeta_route_g(meta=meta, mesh_xy=int(P), n_q_selected=d.nq, ngkmax=d.ng,
                              psi_ngkmax=d.ng, fit_nb=u["nb"], n_col=n_col,
                              n_s=min(n_a, math.ceil(2.4 * r) + 1),
                              budget_gb=BUDGET["local"] / GB, psi_face_bytes=u["F"],
                              n_parent=d.n_parent, factor_on_face=True)
    return z


class _quiet:
    """The planners' banners and warnings off: an inventory prices many meshes."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")
        memory.set_sink(lambda *a, **k: None)

    def __exit__(self, *exc):
        memory.set_sink(None)
        memory.reset()
        self._w.__exit__(*exc)


def stage_prices(d: Deck, P: int) -> dict:
    """``{stage: ((route, Price), ...)}`` per rank at ``P`` ranks; cheapest route first where
    a stage has two.  Stage names are the report's TIMING rows."""
    return dict(_stage_prices(d, int(P)))


@lru_cache(maxsize=4096)
def _stage_prices(d: Deck, P: int) -> tuple:
    from gw.w_isdf import dyson_route_prices
    u = _units(d, P)
    px, py = _mesh(P)
    M, QM, F, PsiG, n2, mu = u["M"], u["QM"], u["F"], u["PsiG"], u["n2"], u["mu"]
    st = {}
    trace = "memtrace MoS2/hBN 16320 P4, Al2O3 4^3 6000 P4"
    # V(q) stays resident from V(q) to Sigma: one [Q, mu, mu] face stack, two on a slab
    # (sys_dim 2: the long-range channel's own V, W.lr_channels).
    nV = 2 if d.sys_dim == 2 else 1
    # ISDF setup + I/O: gw_jax.load_centroid_wfns.  psi_G_store.gslot.read holds the
    # band-sharded read copy beside the G-sharded slice; a2a_faces' exchange adds its
    # temp and output (one slice each) while forming the two faces.
    st["ISDF setup + I/O"] = (("only", _p("ISDF setup + I/O", PsiG + F, 3 * PsiG,
                                          "psi(G) G slice + faces",
                                          f"read copy + face exchange temp/out, 3 psi(G) slices [{trace}]",
                                          P)),)
    # gw_init's canonical Coulomb all-q validation: compiled programs of 64, 32, ... 1 full-zone
    # q, the widest that fits the room admitted (4 M a q: Al2O3 4^3 took 64 q, 36.9 GB, the run's
    # peak).  It fills the room, so its floor is one q.
    st["Coulomb all-q validation"] = (("only", _p(
        "Coulomb all-q validation", QM + F, 4 * d.validation_width * M, "V(q) + faces",
        f"4 M per q, {d.validation_width} q a program (floor 1) [{trace}]", P)),)
    # zeta_fit.CCT: c_q_from_psi_sm's face GEMM (m = n = mu*ns per parent) forms the pair
    # products on the parents and their full-zone unfold, (nk + n_par) ns^2 M, beside one
    # pair of faces, and returns C on the face (QM).
    st["zeta C = (psi psi)(psi psi)^H"] = (("only", _p(
        "zeta CCT", PsiG + F, QM + (d.nk + d.n_parent) * d.ns ** 2 * M + F,
        "psi(G) slice + faces",
        f"C_q + pair products on nk + n_parent k + face pair [{trace}]", P)),)
    # zeta_fit.cholesky (charge B, rank-truncated pseudo-inverse): whole per q owner under
    # linalg=local (5.5 whole tiles: C, its eigenvectors, the syevd workspace, B), or the
    # mesh eigh's face-tiled pair with the solver's contexts.
    whole_q = -(-d.nq // P)
    st["zeta factor B"] = (
        ("local", _p("zeta factor", PsiG + F + QM, 5.5 * whole_q * n2 + QM,
                     "psi(G) slice + faces + C", f"5.5 whole mu^2 per q owner + B [{trace}]", P)),
        ("distributed", _p("zeta factor", PsiG + F + QM, (3 * d.nq + 2) * M + CONTEXT + QM,
                           "psi(G) slice + faces + C",
                           "face-tiled eigh pair (3Q+2) M + contexts + B (model, as W Dyson)", P)))
    z = zeta_plan(d, P)
    st["zeta mu-batch"] = (("only", _p("zeta mu-batch", F, z.min_config_bytes - F, "faces",
                                       "plan_zeta_route_g smallest configuration (C on the face)",
                                       P)),)
    # V(q) on route G (isdf.mubatch_v): the Z store's G tiles of the plan, 6 copies of a
    # tile's q-padded rows (the planner's per-G price), beside B, the V accumulator and faces.
    q_pad = _pad(d.nq, P)
    st["V(q)"] = (("only", _p("V(q)", 2 * QM + F, 6 * C16 * q_pad * mu * z.g_tile / P,
                              "B + V accumulator + faces",
                              "plan_zeta_route_g G tile, 6 x 16 Q_pad mu g_tile / P", P)),)
    # chi.exec: chi0 [nk, mu, mu] on the face (+15 % program temp), the faces' placement
    # (out 2 F, temp F), beside V(q).
    st["chi0"] = (("only", _p("chi0", nV * QM + F, 1.15 * d.nk * M + 3 * F, "V(q) + faces",
                              f"chi0 [nk, mu, mu] + placement [{trace}]", P)),)
    # W.slice_to_ibz: chi0 still whole-zone beside its q-wedge slice.
    st["W slice to IBZ"] = (("only", _p("W slice to IBZ", nV * QM + F + d.nk * M, QM,
                                        "V(q) + faces + chi0", f"wedge slice [{trace}]", P)),)
    loc, dist = dyson_route_prices(mu, d.nq, P, memory.Term(CONTEXT, source="lxkit.legs CUDA contexts"))
    st["W"] = (("local", loc + _p("W", F, 0.0, "faces", "", P)),
               ("distributed", dist + _p("W", F, 0.0, "faces", "", P)))
    # Sigma residents (sigma.exchange / tau_sweep entry): V(q), the W pair (the GN-PPM pole
    # pair in the sweep), faces.
    sig_res = nV * QM + 2 * QM + F
    st["Sigma exchange"] = (("only", _p("Sigma exchange", sig_res, _sigma_new(d, P, static=True),
                                        "V(q) + W pair + faces",
                                        "ppm_tau_kernel.sigma_pass_bytes (static)", P)),)
    # gn_ppm.fit: the fitted pole fields and their masks beside the W pair, and the tail
    # coarsening's temp (0.75-2.5 QM measured).
    st["GN-PPM fit"] = (("only", _p("GN-PPM fit", sig_res, 3 * QM, "Sigma residents",
                                    f"pole fields + masks + coarsening temp [{trace}]", P)),)
    # The window executable runs past the kernel's own analytic price by ~0.04 nk Q M
    # (Al2O3 4^3: booked 7.24, measured 11.66 GB; MoS2/hBN 2x2x1: 17.92 / 17.80).
    st["Sigma tau sweep"] = (("only", _p("Sigma tau sweep", sig_res,
                                         _sigma_new(d, P, static=False) + 0.04 * d.nk * d.nq * M,
                                         "V(q) + GN-PPM pole pair + faces",
                                         "ppm_tau_kernel.sigma_pass_bytes + window executable "
                                         f"excess 0.04 nk Q M [{trace}]", P)),)
    # sigma.exec beyond the sweep (the report's "Sigma other"): the pole synthesis kernel's
    # temp (<= 2.5 QM) and the fit's output beside the pole pair, and the sweep's band-complete
    # psi rows and columns.
    psi_axis = C16 * d.n_parent * d.ns * u["nb"] * (mu // px + mu // py)
    st["Sigma other"] = (("only", _p("Sigma other", sig_res + QM, 2.5 * QM + psi_axis,
                                     "Sigma residents + fitted poles",
                                     f"pole kernel temp 2.5 QM + axis psi rows/columns [{trace}]",
                                     P)),)
    # Sigma Hartree (gw.hartree): the psi(G) wedge sphere over P, then the larger of the
    # density phase (gw.qsgw_density.density_band_chunk boxes) and the V_H matrix sweep
    # (common.mtxel_sweep.sweep_transient_bytes at one k a step: four ket copies of the rank's
    # bands on the G carrier, two nb^2 blocks, and its band-chunk boxes), as
    # hartree._book_density_price / _book_sweep_price book them.
    from gw.qsgw_density import DENSITY_BOX_COPIES, density_band_chunk
    from runtime.tiles import tile_units
    n_r = math.prod(d.fft)
    nb_h = _pad(d.nb_qp, P)
    nb_loc = nb_h // P
    sphere = C16 * d.n_parent * nb_h * d.ns * d.ng / P
    chunk = nb_loc if d.hartree_unchunked else density_band_chunk(
        n_loc=nb_loc, k_tile=1, ns=d.ns, n_grid=n_r)
    density = (2 if d.hartree_unchunked else DENSITY_BOX_COPIES) * C16 * chunk * d.ns * n_r
    per_band = 4 * C16 * d.ns * n_r
    sweep_chunk = -(-nb_loc // -(-nb_loc // tile_units(per_band, nb_loc)))
    sweep = (4 * C16 * nb_loc * d.ns * _pad(d.ng, P) + 2 * C16 * nb_h * nb_h
             + per_band * sweep_chunk)
    st["Sigma Hartree"] = (("only", _p("Sigma Hartree", sig_res + sphere, max(density, sweep),
                                       "Sigma residents + psi(G) wedge sphere",
                                       "max(density step boxes "
                                       + ("(all local bands)" if d.hartree_unchunked else "(band chunk)")
                                       + ", V_H sweep k step + boxes)", P)),)
    return tuple(st.items())


def stage_bytes(d: Deck, P: int) -> dict:
    """``{stage: (route, bytes per rank, budget)}``: each stage's cheapest route that fits its
    budget, else its cheapest route."""
    out = {}
    for name, routes in _stage_prices(d, int(P)):
        rows = [(r, p.bytes(), BUDGET["distributed" if r == "distributed" else "local"])
                for r, p in routes]
        fit = [r for r in rows if r[1] <= r[2]]
        out[name] = (fit or sorted(rows, key=lambda r: r[1] - r[2]))[0]
    return out


#: Host bytes per rank beside the Z store: MoS2/hBN N_mu 3264-16320 at P = 4 measured
#: 5.7-8.8 GB/rank above it (the process, the psi(G) read staging); the planner's host share
#: is 0.8 of a Perlmutter GPU node's 256 GB over its 4 ranks.
HOST_BASE, HOST_SHARE = 9e9, 0.8 * 256e9 / 4


def host_bytes(d: Deck, P: int) -> float:
    """Host bytes per rank: the ζ Z store when the plan keeps it on the host, plus the base."""
    z = zeta_plan(d, int(P))
    return HOST_BASE + (z.store_bytes if z.placement == "host" else 0.0)


def minimum_p(d: Deck, *, max_side: int = 64) -> dict:
    """The smallest square P whose every stage fits its budget, and the binding stage there
    (the Z store moves to disk past the host share, so the host is reported, not a wall)."""
    for side in range(1, max_side + 1):
        P = side * side
        rows = stage_bytes(d, P)
        if all(b <= lim for _, b, lim in rows.values()):
            name = max(rows, key=lambda k: rows[k][1] / rows[k][2])
            return dict(p=P, binding=name, route=rows[name][0], gb=rows[name][1] / GB, rows=rows,
                        host_gb=host_bytes(d, P) / GB)
    return dict(p=None, binding=None, route=None, gb=None, rows=None, host_gb=None)
