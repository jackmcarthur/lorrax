"""The on-site Breit debug file ``onsite_breit.h5`` (design note ``ONSITE_INCODE.md`` section 6).

Written by ``gw.onsite_breit_cli`` (``onsite_breit.h5`` beside the deck) and
by the SC loop (``onsite_breit_iterNNNN.h5`` beside ``sigma_lorentz_iterNNNN.h5``).
One rank (0) writes through ``common.collectives.rank0_atomic_file_transaction``:
a staging sibling, validated by :func:`read_onsite_breit_h5`, then renamed, as
``file_io.sigma_output.write_sc_sigma_lorentz_h5`` does.  Every array handed in
is a HOST array (replicated values every rank holds); a device array refuses.

LAYOUT
------
``dH_kij_ev``            (nk, nb, nb) c128, eV: the matrix that entered H (DFT band
                         basis of the window; full BZ (CLI) or the loop's k-set (SC))
``kpoints_crys``         (nk, 3) f64, crystal coordinates of those rows
``density_matrix_ab``    (R_os, R_os) c128: D in the shell basis
``h_block_ab_ev``        (R_os, R_os) c128, eV: the block potential h that entered H_OS
``density_matrix_Iab``   (I, n_max, n_max) c128: D's shell-atom blocks, zero-padded
                         to the largest 2(2l+1); ``shell_dim_I`` gives each block's size
``h_Iab_ev/<piece>``     (I, n_max, n_max) c128, eV, piece in ``dir_ae dir_ps x_ae
                         x_ps total``; NaN for a shell-atom without that piece
``moments/N_I``, ``moments/S_Ic``, ``moments/L_Ic``, ``moments/herm_I``
                         Tr D, <S>, <L> (Cartesian, hbar = 1), max|D - D^+|
``shell_atom_{atom,element,label,l,start,stop}_I``  per shell-atom labels
``<name>``               each ``extra`` array (the CLI: per-piece matrices, the
                         projection table, the smooth field, spectra, maps)

Root attributes: ``format`` (:data:`ONSITE_BREIT_FORMAT`), ``kset``,
``band_start_1based``, ``nb``, ``projector``, ``double_counting``,
``exchange``, ``setup_provenance`` (the setup's canonical JSON),
``sidecar_sha256`` (JSON ``{element: sha256}``), ``lorrax_version``, the
commit state, and every caller attribute (WFN fingerprint, code commit, ...).
"""
import json

import numpy as np
import h5py

from common.provenance import lorrax_version
from common.units import RYD_TO_EV as RY_TO_EV

ONSITE_BREIT_FORMAT = "lorrax.onsite_breit_debug/v1"
PIECES = ("dir_ae", "dir_ps", "x_ae", "x_ps", "total")
_RESERVED = ("dH_kij_ev", "kpoints_crys", "density_matrix_ab", "h_block_ab_ev",
             "density_matrix_Iab", "h_Iab_ev", "moments", "shell_dim_I")


def _host(name, a, dtype=None):
    """A host numpy copy; a device array (``addressable_shards``) refuses."""
    if hasattr(a, "addressable_shards"):
        raise TypeError(f"write_onsite_breit_h5: {name} is a device array; pass a host array "
                        "(np.asarray of a replicated value) so no rank gathers in the writer")
    return np.asarray(a) if dtype is None else np.asarray(a, dtype=dtype)


def read_onsite_breit_h5(path) -> dict:
    """Read and check a committed file: ``{"dH_kij_ev", "kpoints_crys", "attrs"}``."""
    from .commit_state import assert_committed
    with h5py.File(path, "r") as h5:
        assert_committed(h5, path=path)
        dH = np.asarray(h5["dH_kij_ev"])
        kp = np.asarray(h5["kpoints_crys"])
        attrs = dict(h5.attrs)
        if (h5.attrs.get("format") != ONSITE_BREIT_FORMAT or dH.ndim != 3
                or dH.shape[1] != dH.shape[2] or kp.shape != (dH.shape[0], 3)
                or "density_matrix_Iab" not in h5):
            raise ValueError(f"invalid on-site Breit file {path}: dH {dH.shape}, k {kp.shape}")
    return {"dH_kij_ev": dH, "kpoints_crys": kp, "attrs": attrs}


def write_onsite_breit_h5(path, *, field, dH_kij_ry, kpoints, kset: str, band_start_1based: int,
                          attrs: dict, extra: dict | None = None) -> None:
    """Atomically publish one on-site Breit debug file (rank 0 writes; call on every rank).

    Parameters
    ----------
    path : str
        Destination.
    field : gw.hartree.OnsiteBreitField
        Its host ``density_matrix``, ``h_block``, ``pieces`` and ``setup`` are
        written; ``matrix`` and ``projections`` (device) are not read.
    dH_kij_ry : (nk, nb, nb) complex, host, Ry
        The matrix that entered H (H_OS, or smooth + H_OS); written in eV.
    kpoints : (nk, 3) float, host
        Crystal coordinates of its rows.
    kset : str
        ``"full_bz"`` or the SC loop's k-set name.
    band_start_1based : int
        First band of the window, 1-based (BerkeleyGW convention).
    attrs : dict
        Caller attributes (scalars or strings), stamped at the root.
    extra : dict, optional
        ``{name: host array}`` written as root datasets; a name that collides
        with the layout refuses.
    """
    from common.collectives import rank0_atomic_file_transaction
    from psp.onsite_breit import shell_moments
    from .commit_state import set_commit_state

    dH = _host("dH_kij_ry", dH_kij_ry, np.complex128) * RY_TO_EV
    kp = _host("kpoints", kpoints, np.float64)
    if dH.ndim != 3 or dH.shape[1] != dH.shape[2] or kp.shape != (dH.shape[0], 3):
        raise ValueError(f"write_onsite_breit_h5: dH {dH.shape} and k points {kp.shape} disagree")
    setup = field.setup
    D = _host("field.density_matrix", field.density_matrix, np.complex128)
    h = _host("field.h_block", field.h_block, np.complex128) * RY_TO_EV
    sas = list(setup.shell_atoms)
    dims = np.array([sa["stop"] - sa["start"] for sa in sas], dtype=np.int32)
    n_max = int(dims.max()) if dims.size else 0
    D_I = np.zeros((len(sas), n_max, n_max), np.complex128)
    for i, sa in enumerate(sas):
        sl = slice(sa["start"], sa["stop"])
        D_I[i, :dims[i], :dims[i]] = D[sl, sl]
    h_I = {}
    for name in PIECES:
        if all(pc.get(name) is None for pc in field.pieces):
            continue
        arr = np.full((len(sas), n_max, n_max), np.nan + 0j, np.complex128)
        for i, pc in enumerate(field.pieces):
            if pc.get(name) is not None:
                arr[i, :dims[i], :dims[i]] = _host(f"piece {name}", pc[name], np.complex128) * RY_TO_EV
        h_I[name] = arr
    mo = [shell_moments(D[sa["start"]:sa["stop"], sa["start"]:sa["stop"]], sa["l"]) for sa in sas]
    extra = {k: _host(f"extra[{k}]", v) for k, v in (extra or {}).items()}
    clash = sorted(set(extra) & set(_RESERVED))
    if clash:
        raise ValueError(f"write_onsite_breit_h5: extra names {clash} collide with the layout")

    def _write(staging):
        with h5py.File(staging, "w") as f:
            f.create_dataset("dH_kij_ev", data=dH)
            f.create_dataset("kpoints_crys", data=kp)
            f.create_dataset("density_matrix_ab", data=D)
            f.create_dataset("h_block_ab_ev", data=h)
            f.create_dataset("density_matrix_Iab", data=D_I)
            f.create_dataset("shell_dim_I", data=dims)
            g = f.create_group("h_Iab_ev")
            for name, arr in h_I.items():
                g.create_dataset(name, data=arr)
            g = f.create_group("moments")
            g.create_dataset("N_I", data=np.array([m["N"] for m in mo], np.float64))
            g.create_dataset("S_Ic", data=np.array([m["S"] for m in mo], np.float64).reshape(-1, 3))
            g.create_dataset("L_Ic", data=np.array([m["L"] for m in mo], np.float64).reshape(-1, 3))
            g.create_dataset("herm_I", data=np.array([m["herm"] for m in mo], np.float64))
            for key in ("atom", "l", "start", "stop"):
                f.create_dataset(f"shell_atom_{key}_I", data=np.array([sa[key] for sa in sas], np.int32))
            for key in ("element", "label"):
                f.create_dataset(f"shell_atom_{key}_I",
                                 data=np.array([str(sa[key]) for sa in sas], dtype=h5py.string_dtype()))
            for name, arr in extra.items():
                f.create_dataset(name, data=arr)
            f.attrs["format"] = ONSITE_BREIT_FORMAT
            f.attrs["kset"] = str(kset)
            f.attrs["band_start_1based"] = int(band_start_1based)
            f.attrs["nb"] = int(dH.shape[1])
            f.attrs["units"] = "eV for *_ev; D and the projection table dimensionless"
            f.attrs["projector"] = str(setup.projector)
            f.attrs["double_counting"] = str(setup.double_counting)
            f.attrs["exchange"] = str(setup.exchange)
            f.attrs["setup_provenance"] = str(setup.provenance)
            f.attrs["sidecar_sha256"] = json.dumps(dict(setup.sidecar_sha256), sort_keys=True)
            f.attrs["lorrax_version"] = lorrax_version()
            for key, value in dict(attrs).items():
                f.attrs[str(key)] = value
            set_commit_state(f, True)

    rank0_atomic_file_transaction(path, stage="onsite_breit.write", write=_write,
                                  validate_file=read_onsite_breit_h5)
