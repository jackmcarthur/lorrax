"""BerkeleyGW 4.0 ``epsmat.h5`` / ``eps0mat.h5``: schema, G order, writer.

One owner for the format facts the LORRAX ε⁻¹ export and the BerkeleyGW
projection both depend on; the reader (:class:`file_io.epsreader.EPSReader`)
takes its layout from here.  Sources: ``Common/epswrite_hdf5.f90`` (version
3), ``Common/write_matrix.f90``, ``Epsilon/epsilon_main.f90``,
``Common/sort_inc.f90``, ``Common/input_utils.f90``.

Layout (h5py order, the reverse of the Fortran declaration)::

    mats/matrix           f8 (nq, nmatrix, nfreq, nmtx_max, nmtx_max, 2)
                          [iq, s, iw, a, b, :] = (Re, Im) eps^-1_q(G_b, G_a)
    mats/matrix-diagonal  f8 (nq, nmtx_max, 2)
    eps_header/gspace     nmtx (nq,), nmtx_max, ekin (nq, ng) Ry in rho order,
                          gind_eps2rho / gind_rho2eps (nq, ng) 1-based,
                          vcoul (nq, nmtx_max) = v(q+G) in Ry bohr^3 (no 1/Omega)
    eps_header/qpoints    nq, qpts (nq, 3) crystal, qgrid (3,), qpt_done (nq,)
    eps_header/freqs      freq_dep, nfreq, nfreq_imag, freqs (nfreq, 2) eV
    eps_header/params     matrix_type 0 (eps^-1), matrix_flavor 2, nmatrix,
                          icutv (0 bulk, 6 slab), ecuts Ry, nband, efermi Ry
    mf_header             the WFN header, verbatim (its gspace/components is
                          the rho sphere every gind indexes)

The element orientation is the one non-obvious fact: BerkeleyGW writes the
column ``igp`` of ε⁻¹ as the Fortran slab ``(:, ig, igp)``, so the h5py block
``[a, b]`` is ε⁻¹(G_b, G_a), the TRANSPOSE (not the adjoint) of the matrix.
Logical datasets are int32 0/1.
"""
from __future__ import annotations

from typing import NamedTuple

import h5py
import numpy as np

__all__ = [
    "MATRIX", "DIAGONAL", "TOL_ZERO", "VERSION", "ICUTV_OF_SYS_DIM",
    "QGspace", "kinetic_energies", "gvector_order", "gcutoff", "q_gspace",
    "matrix_block_window", "create_epsmat_file", "write_q_gspace",
    "write_q_diagonal", "read_q_gspace", "read_epsmat_header", "midgap_ry",
]

MATRIX = "mats/matrix"
DIAGONAL = "mats/matrix-diagonal"
#: ``Common/nrtype.f90`` ``TOL_Zero``: the shell tolerance of the G sort.
TOL_ZERO = 1.0e-12
#: ``VER_EPS_HDF5``.
VERSION = 3
#: ``TRUNC_NONE`` / ``TRUNC_SLAB`` by LORRAX ``sys_dim``.
ICUTV_OF_SYS_DIM = {3: 0, 2: 6}


class QGspace(NamedTuple):
    """BerkeleyGW's G list at one q (all arrays host NumPy).

    ``ekin`` ``(ng,)`` Ry, rho order; ``order`` ``(ng,)`` int, eps index ->
    rho index (0-based); ``nmtx`` the leading count with ekin <= ecuts.
    """
    ekin: np.ndarray
    order: np.ndarray
    nmtx: int


def kinetic_energies(components, bdot, q):
    """|q+G|² in Ry by BerkeleyGW's route (``input_utils.f90`` ``kinetic_energies``).

    Parameters
    ----------
    components : (ng, 3) int
        Miller indices of the WFN rho sphere (``mf_header/gspace/components``).
    bdot : (3, 3) float
        Reciprocal metric in Ry (``mf_header/crystal/bdot``).
    q : (3,) float
        Crystal coordinates; BerkeleyGW passes 0 for the q→0 point.

    Returns
    -------
    (ng,) float64
        ``|U(q+G)|²`` with ``bdot = UᵀU`` (``dpotrf('U')``), the value Sigma
        recomputes for the first G of every q and refuses beyond 1e-12.
    """
    U = np.linalg.cholesky(np.asarray(bdot, dtype=np.float64)).T
    qg = np.asarray(components, dtype=np.float64) + np.asarray(q, dtype=np.float64)[None, :]
    return np.sum((qg @ U.T) ** 2, axis=1)


def gvector_order(components, ekin):
    """eps -> rho permutation of ``sortrx_gvec`` (``Common/sort_inc.f90``).

    The comparator is ``e_i - e_j > TOL or (e_i - e_j > -TOL and GK_i > GK_j)``
    with ``GK = gz + n_z (gy + n_y gx)`` over the sphere's Miller box, so the
    order is ascending energy shell by shell and ascending GK inside a shell
    (shells separated by more than ``TOL_ZERO``).

    Returns
    -------
    (ng,) int64, 0-based.
    """
    g = np.asarray(components, dtype=np.int64)
    eff = g.max(axis=0) - g.min(axis=0) + 1
    gk = g[:, 2] + eff[2] * (g[:, 1] + eff[1] * g[:, 0])
    o = np.argsort(ekin, kind="stable")
    shell = np.concatenate([[0], np.cumsum(np.diff(ekin[o]) > TOL_ZERO)])
    return o[np.lexsort((gk[o], shell))]


def gcutoff(ekin_sorted, ecuts):
    """``sort.F90`` ``gcutoff`` verbatim: BerkeleyGW's nmtx, which a written file must match.

    Equal to ``#{ekin <= ecuts}`` on a sorted list whenever
    ``ekin[0] <= ecuts < ekin[-1]`` (every q of a real rho sphere: ekin[0] is
    |q+G|² of the smallest vector and ecutrho = 4 ecutwfc > ecuts).  Two edges
    are BerkeleyGW's, kept on purpose: the whole list counts only when
    ``ekin[-1] < ecuts`` strictly (``ecuts == ekin[-1]`` returns ``ng - 1``),
    and the result is never below 1 (the bisection starts at ``gdn = 1``).
    """
    e = np.asarray(ekin_sorted)
    if e[-1] < ecuts:
        return int(e.size)
    up, dn = e.size, 1
    for _ in range(e.size):
        mid = (up + dn) // 2
        if mid == dn:
            break
        if e[mid - 1] > ecuts:
            up = mid
        else:
            dn = mid
    return int(dn)


def q_gspace(components, bdot, q, ecuts, *, is_q0):
    """BerkeleyGW's G list at ``q``; the q→0 point sorts by |G|² (``epsilon_main.f90``)."""
    ekin = kinetic_energies(components, bdot, np.zeros(3) if is_q0 else q)
    order = gvector_order(components, ekin)
    return QGspace(ekin=ekin, order=order, nmtx=gcutoff(ekin[order], float(ecuts)))


def matrix_block_window(iq, ifreq, n_pad):
    """``(offset, shape)`` of one ``mats/matrix`` (q, ω) block, mesh-padded.

    The block is ``(1, 1, 1, n_pad, n_pad, 2)`` at ``(iq, 0, ifreq, 0, 0, 0)``;
    its sharding is ``P(None, None, None, 'x', 'y', None)``.  A SlabIO write
    clips to ``nmtx_max`` and a read zero-fills past it.
    """
    return (int(iq), 0, int(ifreq), 0, 0, 0), (1, 1, 1, int(n_pad), int(n_pad), 2)


def midgap_ry(wfn_file, nband):
    """BerkeleyGW's semiconductor ``efermi``: (max VBM + min CBM) / 2, in Ry.

    ``Common/input_utils.f90:find_efermi`` with ``fermi_level`` unset: the
    VBM over ``el[s, k, :ifmax]`` and the CBM over ``el[s, k, ifmax:nband]``,
    both over every spin and k of the WFN (``el`` in Ry, ``ifmax`` the
    1-based count of occupied bands, ``mf_header/kpoints``).
    """
    with h5py.File(wfn_file, "r") as w:
        el = np.asarray(w["mf_header/kpoints/el"][()], dtype=np.float64)
        ifmax = np.asarray(w["mf_header/kpoints/ifmax"][()], dtype=np.int64)
    nb = min(int(nband), el.shape[-1])
    cells = list(np.ndindex(ifmax.shape))
    vbm = max(float(el[s, k, : ifmax[s, k]].max()) for s, k in cells)
    cbm = min(float(el[s, k, ifmax[s, k]: nb].min()) for s, k in cells)
    return 0.5 * (vbm + cbm)


def create_epsmat_file(path, *, wfn_file, qpts, qgrid, nmtx, ecuts, nband,
                       efermi_ry, icutv, freq_dep=0, freqs_ev=((0.0, 0.0),),
                       nfreq_imag=0):
    """Write a version-3 skeleton: headers, empty contiguous matrix datasets.

    Serial h5py, ONE process (the caller's rank 0), before any SlabIO handle
    opens the path.  ``mf_header`` is copied from ``wfn_file`` byte for
    byte, so every ``gind`` indexes the same rho sphere Sigma and Kernel map
    with ``findvector``.  ``qpt_done`` starts 0; :func:`write_q_diagonal`
    sets it per q after the matrix block landed.

    Parameters
    ----------
    qpts : (nq, 3) float, crystal coordinates.
    nmtx : (nq,) int.
    freqs_ev : (nfreq, 2) float, (Re, Im) in eV, BerkeleyGW's
        ``dFreqGrid + Re dFreqBrd, Im dFreqBrd``.
    """
    qpts = np.asarray(qpts, dtype=np.float64).reshape(-1, 3)
    nmtx = np.asarray(nmtx, dtype=np.int32).reshape(-1)
    freqs = np.asarray(freqs_ev, dtype=np.float64).reshape(-1, 2)
    nq, nfreq, nmax = qpts.shape[0], freqs.shape[0], int(nmtx.max())
    with h5py.File(wfn_file, "r") as w, h5py.File(path, "w") as f:
        w.copy("mf_header", f)
        ng = int(np.asarray(f["mf_header/gspace/ng"][()]))
        i4 = np.int32
        f["eps_header/versionnumber"] = i4(VERSION)
        f["eps_header/flavor"] = i4(2)
        p = f.create_group("eps_header/params")
        for k, v in (("matrix_type", 0), ("has_advanced", 0), ("nmatrix", 1),
                     ("matrix_flavor", 2), ("icutv", int(icutv)),
                     ("nband", int(nband)), ("intraband_flag", 0),
                     ("subsample", 0), ("subspace", 0)):
            p[k] = i4(v)
        p["ecuts"] = np.float64(ecuts)
        p["efermi"] = np.float64(efermi_ry)
        p["intraband_overlap_min"] = np.float64(0.5)
        g = f.create_group("eps_header/qpoints")
        g["nq"] = i4(nq)
        g["qpts"] = qpts
        g["qgrid"] = np.asarray(qgrid, dtype=i4).reshape(3)
        g["qpt_done"] = np.zeros(nq, dtype=i4)
        g = f.create_group("eps_header/freqs")
        g["freq_dep"] = i4(freq_dep)
        g["nfreq"] = i4(nfreq)
        g["nfreq_imag"] = i4(nfreq_imag)
        g["freqs"] = freqs
        g = f.create_group("eps_header/gspace")
        g["nmtx"] = nmtx
        g["nmtx_max"] = i4(nmax)
        g.create_dataset("ekin", shape=(nq, ng), dtype="f8")
        g.create_dataset("vcoul", shape=(nq, nmax), dtype="f8")
        g.create_dataset("gind_eps2rho", shape=(nq, ng), dtype="i4")
        g.create_dataset("gind_rho2eps", shape=(nq, ng), dtype="i4")
        # Contiguous (no chunks): SlabIO reuses an identical existing dataset.
        f.create_dataset(MATRIX, shape=(nq, 1, nfreq, nmax, nmax, 2), dtype="f8")
        f.create_dataset(DIAGONAL, shape=(nq, nmax, 2), dtype="f8")


def write_q_gspace(path, iq, qg, vcoul_ry_bohr3):
    """Write one q's ``ekin``, ``gind_*`` and ``vcoul`` (serial h5py, one rank)."""
    inv = np.empty_like(qg.order)
    inv[qg.order] = np.arange(qg.order.size)
    with h5py.File(path, "a") as f:
        g = f["eps_header/gspace"]
        g["ekin"][iq] = qg.ekin
        g["gind_eps2rho"][iq] = (qg.order + 1).astype(np.int32)
        g["gind_rho2eps"][iq] = (inv + 1).astype(np.int32)
        g["vcoul"][iq, :qg.nmtx] = np.asarray(vcoul_ry_bohr3, dtype=np.float64)[:qg.nmtx]


def write_q_diagonal(path, iq, diag):
    """Write ``matrix-diagonal`` of one q and set its ``qpt_done`` (serial h5py, one rank)."""
    d = np.asarray(diag, dtype=np.complex128)
    with h5py.File(path, "a") as f:
        f[DIAGONAL][iq, :d.size, 0] = d.real
        f[DIAGONAL][iq, :d.size, 1] = d.imag
        done = f["eps_header/qpoints/qpt_done"]
        done[iq] = np.int32(1)


def read_epsmat_header(path):
    """Small, replicated facts of an eps file as a dict (serial h5py, any rank)."""
    with h5py.File(path, "r") as f:
        h = {k: np.asarray(f[f"eps_header/{k}"][()]) for k in (
            "qpoints/qpts", "qpoints/qgrid", "qpoints/qpt_done", "gspace/nmtx",
            "gspace/nmtx_max", "freqs/freq_dep", "freqs/nfreq", "freqs/freqs",
            "params/ecuts", "params/icutv", "params/matrix_type",
            "params/nmatrix", "params/matrix_flavor", "flavor", "versionnumber")}
        h["components"] = np.asarray(f["mf_header/gspace/components"][()], dtype=np.int64)
        h["bdot"] = np.asarray(f["mf_header/crystal/bdot"][()], dtype=np.float64)
        h["celvol"] = float(np.asarray(f["mf_header/crystal/celvol"][()]))
    h["qpts"] = h.pop("qpoints/qpts").reshape(-1, 3)
    h["nmtx"] = h.pop("gspace/nmtx").reshape(-1).astype(np.int64)
    return h


def read_q_gspace(path, iq):
    """``(QGspace, vcoul)`` of one stored q, the file's own G order (serial h5py)."""
    with h5py.File(path, "r") as f:
        g = f["eps_header/gspace"]
        nmtx = int(np.asarray(g["nmtx"][iq]))
        order = np.asarray(g["gind_eps2rho"][iq], dtype=np.int64) - 1
        ekin = np.asarray(g["ekin"][iq], dtype=np.float64)
        vcoul = np.asarray(g["vcoul"][iq, :nmtx], dtype=np.float64)
    return QGspace(ekin=ekin, order=order, nmtx=nmtx), vcoul
