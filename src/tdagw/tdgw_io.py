"""tdagw/tdgw_io.py — TDGW's file formats: ``tdgw.h5``, ``tdgw.toml``, ``kernel.h5``.

The owner of every convention LORRAX shares with TDGW (Xiaoxun Gong's
TD-aGW code): Hartree atomic units inside files, attoseconds, eV and V/m in
decks, the CODATA constants TDGW takes from ``scipy.constants``
(``TDGW/constants.py``), the ``tdgw.h5`` group layout (TDGW README "File
structure of tdgw.h5"), and the BerkeleyGW ``kernel.h5`` reading of
``TDGW/td/io.py::read_kernel``.  A ``tdgw.h5`` written here is read by
``tdgw-pp`` unchanged (it reads the header, the checkpoint and the
``time_domain`` datasets, plus ``tdgw.toml`` beside it).

The header (structure, WFN header, dipole) is read from a TDGW output, as
TDGW's own ``load_header_from`` does, including the dipole reconstruction of
``DipoleFromWFN.load``: S~ = U V^H of the energy-masked overlap, the
k-derivative r for periodic directions, v0 = i (E_n - E_m) r and
v0_intra = sum_b w_b b (E_{k+b} - E_k).
"""
from __future__ import annotations

import os

import numpy as np
import scipy.constants as _cst

__all__ = ["HA_IN_EV", "T_IN_ATTOSEC", "EFIELD_IN_VPERM", "HBAR_IN_EV_AS",
           "TDGW_LAYOUT_VERSION", "efield_function", "read_header",
           "header_operators", "read_bgw_kernel", "init_output",
           "write_samples", "write_checkpoint", "dump_toml"]

# TDGW/constants.py, verbatim definitions
HA_IN_EV = _cst.physical_constants["Hartree energy in eV"][0]
T_IN_ATTOSEC = _cst.physical_constants["atomic unit of time"][0] / _cst.atto
EFIELD_IN_VPERM = _cst.physical_constants["atomic unit of electric field"][0]
HBAR_IN_EV_AS = _cst.physical_constants["reduced Planck constant in eV s"][0] / _cst.atto
#: The tdgw.h5 layout version this writer emits (TDGW __version__ 0.3.0).
TDGW_LAYOUT_VERSION = (0, 3, 0)

_KPTS_TOL = 1e-8


# ---------------------------------------------------------------------------
# The drive: TDGW's [system.Efunc] expressions, evaluated in jnp
# ---------------------------------------------------------------------------

def efield_function(trials):
    """E(t) for ``ntr`` trials from TDGW Efunc tables, traceable.

    ``trials`` is a list of dicts {E_x, E_y, E_z}, each value a float (V/m)
    or a TDGW expression in ``t`` (attoseconds) with ``np``,
    ``HBAR_IN_EV_AS`` and ``EFIELD_IN_VPERM`` in scope.  ``np`` is bound to
    ``jax.numpy``, so the expressions TDGW evaluates with NumPy trace into
    the jitted step.  Returns ``f(t_au) -> (ntr, 3)`` float64 in a.u.,
    E_au = E(t_au * T_IN_ATTOSEC) / EFIELD_IN_VPERM (TDGW ``EFunc``).
    """
    import jax.numpy as jnp

    compiled = []
    for tr in trials:
        row = []
        for key in ("E_x", "E_y", "E_z"):
            v = tr.get(key, 0.0)
            row.append(compile(v, f"<Efunc {key}>", "eval")
                       if isinstance(v, str) else float(v))
        compiled.append(row)

    def f(t_au):
        t_as = t_au * T_IN_ATTOSEC
        scope = {"np": jnp, "HBAR_IN_EV_AS": HBAR_IN_EV_AS,
                 "EFIELD_IN_VPERM": EFIELD_IN_VPERM}
        rows = []
        for row in compiled:
            vals = [jnp.asarray(eval(c, scope, {"t": t_as}), jnp.float64)
                    if not isinstance(c, float)
                    else jnp.asarray(c, jnp.float64) for c in row]
            rows.append(jnp.stack(vals))
        return jnp.stack(rows) / EFIELD_IN_VPERM
    return f


# ---------------------------------------------------------------------------
# Header: read a TDGW output's tdgw_header
# ---------------------------------------------------------------------------

def _process_geometry(rprim, geometry):
    """TDGW ``process_geometry``: which Cartesian directions are periodic."""
    if geometry == "bulk":
        return np.array([True, True, True])
    if geometry == "slab":
        return np.array([True, True, False])
    if geometry == "wire":
        return np.array([False, False, True])
    if geometry == "box":
        return np.array([False, False, False])
    raise ValueError(f"GATE td_geometry: got: geometry={geometry!r}; want: "
                     f"bulk|slab|wire|box; fix: set [system] geometry")


def _k_lookup(kpts, kgrid, kshift):
    """code(k) -> index; codes as TDGW ``KPtsLookUp.encode``."""
    kgrid = np.asarray(kgrid, np.int64)
    kshift = np.asarray(kshift, float)

    def encode(k):
        tmp = k * kgrid - kshift
        idx = np.round(tmp)
        if np.max(np.abs((idx - tmp) / kgrid)) > _KPTS_TOL:
            raise ValueError("GATE td_kgrid: got: k points off the declared "
                             "k grid; want: kgrid/kshift-consistent k; fix: "
                             "regenerate the header")
        idx = np.remainder(idx.astype(np.int64), kgrid)
        return np.ravel_multi_index((idx[..., 0], idx[..., 1], idx[..., 2]), kgrid)

    table = -np.ones(int(np.prod(kgrid)), np.int64)
    table[encode(kpts)] = np.arange(len(kpts))
    return encode, table


def read_header(tdgw_dir: str):
    """The raw ``tdgw_header`` groups of ``tdgw_dir/tdgw.h5`` (read-only)."""
    import h5py
    path = os.path.join(tdgw_dir, "tdgw.h5")
    with h5py.File(path, "r") as f:
        g = f["tdgw_header"]
        st, wf, dp = g["structure"], g["wfn_header"], g["dipole"]
        hdr = dict(
            path=path,
            rprim=st["rprim"][()], atomic_numbers=st["atomic_numbers"][()],
            atomic_positions_red=st["atomic_positions_red"][()],
            kpts=wf["kpts"][()],
            kgrid=wf["kgrid"][()] if "kgrid" in wf else None,
            kshift=wf["kshift"][()] if "kshift" in wf else None,
            bands=tuple(int(wf["bands"][k][()]) for k in ("start", "stop", "step")),
            nspinor=int(wf["nspinor"][()]), fftgrid=wf["FFTgrid"][()],
            e_kn=wf["e_kn"][()], f_kn=wf["f_kn"][()], w_k=wf["w_k"][()],
            itype=int(dp["itype"][()]), rinter=dp["rinter_iknn"][()],
            rintra=dp["rintra_ikn"][()] if "rintra_ikn" in dp else None,
            bvecs=dp["bvecs"][()] if "bvecs" in dp else None,
            bvecs_weight=dp["bvecs_weight"][()] if "bvecs_weight" in dp else None,
            is_patched=bool(dp["is_patched"][()]) if "is_patched" in dp else False,
            olp=dp["olp_bknn"][()] if "olp_bknn" in dp else None)
    return hdr


def header_operators(hdr: dict, *, geometry: str, esmear_ha: float,
                     intraband: bool) -> dict:
    """Host operators of :func:`tdagw.propagate.place_operators` from a header.

    Reproduces TDGW's ``load_dipole`` (itype 1: r from vmtxel, no intraband
    commutator; itype 2: ``DipoleFromWFN.load``) and ``calc_v0_inter``.
    ``intraband = False`` drops the commutator of Eq. (3) (TDGW's
    ``TDGW_RINTRA_IN_EOM=false``).
    """
    e_kn = np.asarray(hdr["e_kn"], float)
    nk, nb = e_kn.shape
    rinter = np.array(hdr["rinter"], complex)
    rprim = np.asarray(hdr["rprim"], float)
    gprim = np.linalg.inv(rprim.T)
    intra = None
    v0_intra = None
    if hdr["itype"] == 2 and geometry != "box":
        if hdr["is_patched"]:
            raise ValueError("GATE td_patched_sampling: got: a patched-sampling "
                             "header; want: a regular k grid; fix: use an "
                             "unpatched TDGW header")
        b_red = np.asarray(hdr["bvecs"], float)
        w_b = np.asarray(hdr["bvecs_weight"], float)
        b_cart = 2 * np.pi * b_red @ gprim
        wb = (w_b[None, :] * b_cart.T)                       # (3, nbvec)
        encode, table = _k_lookup(hdr["kpts"], hdr["kgrid"], hdr["kshift"])
        kmap = np.stack([table[encode(hdr["kpts"] + b)] for b in b_red])
        if np.any(kmap < 0):
            raise ValueError("GATE td_kplusb: got: a k+b outside the k list; "
                             "want: a closed k grid; fix: use the full-BZ WFN "
                             "the header was made from")
        olp = np.asarray(hdr["olp"], complex)                # (nbvec, nk, nb, nb)
        e_kb = e_kn[kmap]                                    # (nbvec, nk, nb)
        mask = np.exp(-((e_kn[None, :, :, None] - e_kb[:, :, None, :])
                        / esmear_ha) ** 2 / 2.0)
        u, _, vh = np.linalg.svd(olp * mask)
        stilde = u @ vh
        olpcov = olp @ np.conj(np.swapaxes(stilde, -1, -2))
        rk = 1j * np.tensordot(wb, olpcov - np.eye(nb)[None, None], axes=1)
        rk = 0.5 * (rk + np.conj(np.swapaxes(rk, -1, -2)))
        periodic = _process_geometry(rprim, geometry)
        rinter[periodic] = rk[periodic]
        v0_intra = np.tensordot(wb, e_kb - e_kn[None], axes=1)
        rintra = (None if geometry == "bulk" or hdr["rintra"] is None
                  else np.asarray(hdr["rintra"], float))
        if intraband:
            intra = dict(stilde=stilde, kmap=kmap, wb=wb, rintra=rintra)
    idx = np.arange(nb)
    rinter[:, :, idx, idx] = 0.0
    v0_inter = 1j * rinter * (e_kn[None, :, :, None] - e_kn[None, :, None, :])
    return dict(e_qp=e_kn, f_kn=np.asarray(hdr["f_kn"], float), r_inter=rinter,
                w_k=np.asarray(hdr["w_k"], float), v0_inter=v0_inter,
                v0_intra=v0_intra, intra=intra,
                cell_volume=float(abs(np.linalg.det(rprim))),
                nspinor=int(hdr["nspinor"]))


# ---------------------------------------------------------------------------
# BerkeleyGW kernel.h5, TDGW's reading
# ---------------------------------------------------------------------------

def read_bgw_kernel(path: str, *, ivbm: int, bands: tuple):
    """TDGW ``read_kernel``: the dense kernel K acting on rho, Ha.

    Returns (K (nk, n1, n2, nk, n1, n2) complex, tda, cslice, vslice, kpts):
    TDA (nblocks = 1) gives (k, c, v, k', c', v') with K = -kernel/2 and the
    valence order reversed to ascending; the full kernel (nblocks = 4) is
    mapped onto the propagated window ``bands`` = (start, stop, step), 1-based
    absolute indices.  cslice/vslice are window positions (TDA).
    """
    import h5py
    with h5py.File(path, "r") as f:
        hg = f["bse_header"]
        flavor = int(hg["flavor"][()])
        nblocks = int(hg["params/nblocks"][()])
        kpts = hg["kpoints/kpts"][()]
        nvb, ncb = int(hg["bands/nvb"][()]), int(hg["bands/ncb"][()])
        if flavor != 2:
            raise ValueError("GATE td_kernel_flavor: got: a real-flavor "
                             "kernel.h5; want: complex (flavor 2); fix: rerun "
                             "BerkeleyGW kernel in complex flavor")
        if nblocks == 1:
            raw = -f["kernel"][:, :, nvb:nvb + ncb, 0:nvb, nvb:nvb + ncb, 0:nvb, :]
        elif nblocks == 4:
            raw = -f["kernel"][()]
        else:
            raise ValueError(f"GATE td_kernel_blocks: got: nblocks={nblocks}; "
                             f"want: 1 (TDA) or 4 (full)")
    raw = raw / 2.0                                          # Ry -> Ha
    k = raw[..., 0] + 1j * raw[..., 1]
    start, stop, step = bands
    win = np.arange(start, stop, step)
    if nblocks == 1:
        k = k[:, :, :, ::-1, :, ::-1]
        k = np.transpose(k, (1, 2, 3, 0, 4, 5))
        c_b = np.arange(ivbm + 1, ivbm + 1 + ncb)
        v_b = np.arange(ivbm - nvb + 1, ivbm + 1)
        cs = int(np.searchsorted(win, c_b[0])); vs = int(np.searchsorted(win, v_b[0]))
        if not (np.array_equal(win[cs:cs + ncb], c_b)
                and np.array_equal(win[vs:vs + nvb], v_b)):
            raise ValueError("GATE td_kernel_bands: got: kernel bands outside "
                             "the propagated window; want: ibndmin <= ivbm-nvb+1 "
                             "and ivbm+ncb <= ibndmax; fix: widen [system] bands")
        return k, True, (cs, cs + ncb), (vs, vs + nvb), kpts
    file_bands = np.concatenate([np.arange(ivbm, ivbm - nvb, -1),
                                 np.arange(ivbm + 1, ivbm + 1 + ncb)])
    pos = {b: i for i, b in enumerate(file_bands)}
    try:
        bmap = np.array([pos[b] for b in win])
    except KeyError as exc:
        raise ValueError("GATE td_kernel_bands: got: a window band absent from "
                         "the full kernel; fix: match ibndmin/ibndmax to the "
                         "kernel's nvb/ncb") from exc
    k = k[:, :, bmap[:, None, None, None], bmap[None, :, None, None],
          bmap[None, None, :, None], bmap[None, None, None, :]]
    return np.transpose(k, (1, 2, 3, 0, 4, 5)), False, None, None, kpts


# ---------------------------------------------------------------------------
# tdgw.h5 writer (rank 0 only; the caller gathers)
# ---------------------------------------------------------------------------

def _copy_or_write_header(f, hdr):
    import h5py
    g = f.create_group("tdgw_header")
    g.create_dataset("version", data=np.array(TDGW_LAYOUT_VERSION), dtype=np.int32)
    if hdr.get("path"):
        with h5py.File(hdr["path"], "r") as src:
            for name in ("structure", "wfn_header", "dipole"):
                src.copy(src["tdgw_header"][name], g, name=name)
        return g
    st = g.create_group("structure")
    st["rprim"] = hdr["rprim"]; st["atomic_numbers"] = hdr["atomic_numbers"]
    st["atomic_positions_red"] = hdr["atomic_positions_red"]
    wf = g.create_group("wfn_header")
    wf["kpts"] = hdr["kpts"]
    if hdr.get("kgrid") is not None:
        wf["kgrid"] = hdr["kgrid"]; wf["kshift"] = hdr["kshift"]
    bg = wf.create_group("bands")
    for k, v in zip(("start", "stop", "step"), hdr["bands"]):
        bg.create_dataset(k, data=v, dtype=np.int64)
    wf.create_dataset("nspinor", data=hdr["nspinor"], dtype=np.int64)
    wf["FFTgrid"] = hdr["fftgrid"]
    wf["e_kn"] = hdr["e_kn"]; wf["f_kn"] = hdr["f_kn"]; wf["w_k"] = hdr["w_k"]
    dp = g.create_group("dipole")
    dp.create_dataset("itype", data=hdr["itype"], dtype=np.int32)
    dp["rinter_iknn"] = hdr["rinter"]
    e = np.asarray(hdr["e_kn"])
    dp["v0inter_iknn"] = 1j * np.asarray(hdr["rinter"]) * (e[None, :, :, None] - e[None, :, None, :])
    if hdr.get("td_system") is not None:
        import json
        # LORRAX provenance of a bundle-built header (tdagw.td_system); TDGW
        # and tdgw-pp ignore attributes.
        dp.attrs["lorrax_td_system"] = json.dumps(hdr["td_system"], default=_json_default)
    return g


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def init_output(path: str, hdr: dict, *, t0, tf, mdt, mtsteps, measurements):
    """Create ``tdgw.h5``: header, checkpoint stubs, empty time-domain sets.

    ``measurements``: list of (dset_name, times (n,), data shape, dtype).
    Times and data are in a.u., the layout of TDGW ``BaseDriver.init_datafile``.
    """
    import h5py
    with h5py.File(path, "w") as f:
        g = _copy_or_write_header(f, hdr)
        g.create_dataset("tgrid", data=np.array([t0, tf, mdt]), dtype=np.float64)
        g.create_dataset("tsteps", data=np.asarray(mtsteps), dtype=np.float64)
        d = f.create_group("tdgw_data")
        c = d.create_group("checkpoint")
        c.create_dataset("it_next", data=-1, dtype=np.int64)
        c.create_dataset("tchkp", data=0.0, dtype=np.float64)
        nk, nb = np.asarray(hdr["e_kn"]).shape
        c.create_dataset("lessG_knn", shape=(nk, nb, nb), dtype=np.complex128)
        td = d.create_group("time_domain")
        for name, times, shape, dtype in measurements:
            m = td.create_group(name)
            m.create_dataset("times", data=np.asarray(times))
            m.create_dataset("data", shape=(len(times),) + tuple(shape), dtype=dtype)


def write_samples(path: str, rows: dict):
    """rows: {dset_name: (start, data)}; writes data[...] at rows start:."""
    import h5py
    with h5py.File(path, "a") as f:
        td = f["tdgw_data/time_domain"]
        for name, (start, data) in rows.items():
            if len(data):
                td[name]["data"][start:start + len(data)] = data


def write_checkpoint(path: str, *, it_next: int, tchkp: float, lessG):
    """TDGW checkpoint: next measurement index, state time, G^< = i rho."""
    import h5py
    with h5py.File(path, "a") as f:
        c = f["tdgw_data/checkpoint"]
        c["it_next"][...] = it_next
        c["tchkp"][...] = tchkp
        c["lessG_knn"][...] = lessG


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):                                 # inline table, e.g. lorrax.trials rows
        return "{" + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"cannot write {type(v).__name__} to toml")


def dump_toml(path: str, table: dict):
    """A minimal TOML writer for the nested dicts of a TDGW deck."""
    lines = []

    def emit(prefix, d):
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            lines.append(f"[{prefix}]")
        for k, v in scalars.items():
            if v is None:
                continue
            lines.append(f"{k} = {_toml_value(v)}")
        if scalars:
            lines.append("")
        for k, v in tables.items():
            emit(f"{prefix}.{k}" if prefix else str(k), v)

    emit("", table)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
