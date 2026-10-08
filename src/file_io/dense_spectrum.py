"""Read-only, native-band tiles for small finite-basis reference calculations.

The archive is not a production WFN: its parent k spheres have different
dimensions. The caller keeps the canonical source ``WfnLoader`` open, owning
typed symmetry transport and paired k/G representatives. This reader owns
archive finalization, source authentication, and zero carrier bands. Bulk
coefficients pass through SlabIO with bands distributed over all XY ranks;
centroid sampling uses ``common.wfn_transforms.gflat_to_rmu`` unchanged.
No Green function, response, occupation model, or head is implemented here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .slab_io import SlabIO
from .commit_state import COMMIT_STATE, assert_committed, agree_io_refusal


SCHEMA = "lorrax.dense-native-spectrum.v1"


def _refuse(message):
    raise ValueError(f"GATE dense_spectrum_reference: {message}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def verify_source_bindings(bindings, *, source_aliases=None):
    """Authenticate original reconstruction bytes at explicit relocated paths.

    The original mapping stays the archive's identity. An alias mapping is
    complete and exact: no original binding may be omitted or added, and
    each actual file must hash to that original binding's digest. Aliases
    change addresses only, never the identity or an authentication gate.
    """
    if not isinstance(bindings, dict) or not bindings:
        _refuse("source reconstruction bindings must be a nonempty mapping")
    if source_aliases is None:
        actual = {path: path for path in bindings}
    else:
        if (not isinstance(source_aliases, dict)
                or set(source_aliases) != set(bindings)):
            _refuse("source aliases must cover exactly the original reconstruction bindings")
        actual = dict(source_aliases)
        if any(not isinstance(path, str) or not Path(path).is_absolute()
               for path in actual.values()):
            _refuse("source aliases must name explicit absolute paths")
    for original, digest in bindings.items():
        if (not isinstance(digest, str) or len(digest) != 64
                or _sha256(actual[original]) != digest):
            _refuse(f"changed source reconstruction binding {original}")
    return actual


def _read_metadata(path):
    """Only metadata/eigenvalue vectors; never coefficient payloads."""
    with h5py.File(path, "r") as handle:
        if COMMIT_STATE not in handle or handle[COMMIT_STATE].shape != (1,):
            _refuse("new native archive misses the canonical length-one commit dataset")
        assert_committed(handle, path=str(path))
        attrs = dict(handle.attrs)
        names = ("ngk", "basis_dimensions", "kpoints_crystal", "kweights",
                 "kgrid", "source_occupations", "num_electrons")
        data = {name: np.asarray(handle[name]) for name in names}
        data["io_committed"] = int(handle[COMMIT_STATE][0])
        parents = []
        for ik in range(len(data["ngk"])):
            group = handle[f"k{ik:05d}"]
            parents.append({
                "energies_ry": np.asarray(group["energies_ry"]),
                "gvecs": np.asarray(group["gvecs"]),
                "checks": np.asarray(group["checks"]),
                "coefficient_shape": tuple(group["coefficients"].shape),
                "coefficient_dtype": np.dtype(group["coefficients"].dtype),
            })
    return attrs, data, parents


def native_band_geometry(dimension, ngk, nspinor, ngkmax, mesh_size,
                         b0, b1, *, pad_to=None, max_local_read_bytes):
    """Validate one bounded read and identify physical columns before I/O.

    The returned real/imaginary read buffer is band-sharded. Its live input
    and complex output together cost at most twice ``local_payload_bytes``;
    FFT workspace is separately controlled by the canonical transform owner.
    """
    values = (dimension, ngk, nspinor, ngkmax, mesh_size, b0, b1)
    if any(int(v) != v for v in values):
        _refuse("band geometry contains a noninteger count")
    dimension, ngk, ns, ngkmax, world, b0, b1 = map(int, values)
    if (dimension != ns * ngk or min(dimension, ngk, ns, ngkmax, world) <= 0
            or ngk > ngkmax or not 0 <= b0 < b1):
        _refuse("invalid native dimension, G extent, mesh, or band interval")
    requested = b1 - b0
    if pad_to is not None and (int(pad_to) != pad_to or int(pad_to) < requested):
        _refuse("pad_to must be an integer at least the requested band width")
    extent = max(requested, requested if pad_to is None else int(pad_to))
    extent = ((extent + world - 1) // world) * world
    local_bytes = extent // world * ns * ngkmax * 16
    if int(max_local_read_bytes) <= 0 or 2 * local_bytes > int(max_local_read_bytes):
        _refuse(f"band tile live read/complex payload {2 * local_bytes} bytes per rank "
                f"exceeds explicit reference bound {int(max_local_read_bytes)}")
    live = max(0, min(b1, dimension) - b0)
    valid = np.arange(extent) < live
    return (extent, ns, ngkmax, 2), live, valid[None, :], local_bytes


class DenseSpectrumReader:
    """Collective resource guard for a finalized, source-bound native archive.

    ``source_wfn`` is the still-open canonical WfnLoader for the seed WFN.
    ``expected_source_sha256`` comes from the frozen experiment manifest;
    ``expected_archive_sha256`` comes from the independent native-spectrum
    parser's checksum and closure receipt. Both are mandatory.
    Reading coefficients is collective on ``mesh``. This class never closes
    the caller's source loader and never admits this archive to production
    WFN or unmasked shared-pole census routes.
    """
    def __init__(self, path, source_wfn, *, mesh, expected_source_sha256,
                 expected_archive_sha256,
                 max_local_read_bytes=256 * 1024**2, source_aliases=None):
        self.path = str(Path(path).resolve())
        self.source_wfn = source_wfn
        self.mesh = mesh
        self.max_local_read_bytes = int(max_local_read_bytes)
        self._io = None
        error = None
        try:
            self._authenticate(expected_source_sha256, expected_archive_sha256,
                               source_aliases=source_aliases)
        except Exception as exc:
            error = exc
        agree_io_refusal(error, path=self.path, stage="native spectrum reference open")
        if error is not None:
            raise error
        self._io = SlabIO(self.path, mode="r", mesh=mesh)

    def _authenticate(self, expected_source_sha256, expected_archive_sha256,
                      *, source_aliases=None):
        source_wfn = self.source_wfn
        archive_digest = str(expected_archive_sha256)
        if len(archive_digest) != 64 or _sha256(self.path) != archive_digest:
            _refuse("archive SHA256 differs from the independently checked receipt")
        try:
            attrs, data, parents = _read_metadata(self.path)
        except (KeyError, TypeError, ValueError, OSError) as error:
            _refuse(f"archive metadata/groups could not be authenticated: {error}")
        if (_text(attrs.get("schema", "")) != SCHEMA
                or attrs.get("finalized") != 1 or attrs.get("complete_native_basis") != 1
                or data.get("io_committed") != 1):
            _refuse("archive is not finalized, committed, and complete")
        if (_text(attrs.get("energy_units", "")) != "Ry"
                or _text(attrs.get("coefficient_convention", ""))
                != "band,spinor,source-QE-G,real-imag"):
            _refuse("archive units or coefficient convention differ")
        try:
            bindings = json.loads(_text(attrs["source_sha256_bindings"]))
            seed = str(Path(_text(attrs["dense_h_source_wfn"])).resolve())
        except (KeyError, TypeError, ValueError) as error:
            _refuse(f"missing or malformed source binding: {error}")
        expected = str(expected_source_sha256)
        if (not isinstance(bindings, dict) or len(expected) != 64 or bindings.get(seed) != expected
                or _sha256(source_wfn.path) != expected):
            _refuse("seed WFN SHA256 differs from the manifest/archive")
        bound_names = {Path(p).name for p in bindings}
        if (not {"data-file-schema.xml", "charge-density.hdf5", "run_dense_h.py", "qp_wfn.py"}
                <= bound_names or not any(Path(p).suffix.lower() == ".upf" for p in bindings)):
            _refuse("archive misses the density/XML/UPF/producer bindings")
        # Bind the full DFT reconstruction (density, XML, UPFs and producer
        # files), not only a mutable path. These are exact, bounded paths.
        self.reconstruction_source_paths = verify_source_bindings(
            bindings, source_aliases=source_aliases)
        ns = int(source_wfn.nspinor)
        ngk = np.asarray(source_wfn.ngk_valid(k="ibz"), dtype=np.int64)
        if int(attrs.get("nspinor", 0)) != ns:
            _refuse("source spinor count differs")
        expected_metadata = {
            "ngk": ngk, "basis_dimensions": ns * ngk,
            "kpoints_crystal": np.asarray(source_wfn.kpoints),
            "kweights": np.asarray(source_wfn.kweights),
            "kgrid": np.asarray(source_wfn.kgrid),
            "source_occupations": np.asarray(source_wfn.occs),
            "num_electrons": np.asarray(source_wfn.num_electrons),
        }
        for name, values in expected_metadata.items():
            if (data[name].shape != values.shape or not np.array_equal(data[name], values)
                    or not np.isfinite(data[name]).all()):
                _refuse(f"archive/source metadata differ at {name}")
        if ngk.ndim != 1 or len(parents) != len(ngk) or not len(ngk) or np.any(ngk <= 0):
            _refuse("invalid native parent census")
        import distrib_la
        tolerance = distrib_la.roundoff_tol(ns * int(ngk.max()), dtype=np.complex128)
        for ik, parent in enumerate(parents):
            d, g = ns * int(ngk[ik]), int(ngk[ik])
            e, gv, checks = parent["energies_ry"], parent["gvecs"], parent["checks"]
            if (e.shape != (d,) or not np.isfinite(e).all() or np.any(np.diff(e) < 0)
                    or gv.shape != (g, 3) or not np.issubdtype(gv.dtype, np.integer)
                    or not np.array_equal(gv, source_wfn.get_gvec_nk(ik))
                    or len(np.unique(gv, axis=0)) != g
                    or checks.shape != (3,) or not np.isfinite(checks).all()
                    or np.any(checks < 0) or np.any(checks > tolerance)
                    or parent["coefficient_shape"] != (d, ns, g, 2)
                    or parent["coefficient_dtype"] != np.dtype(np.float64)):
                _refuse(f"malformed/incomplete native parent {ik}")
        self.nspinor = ns
        self.ngk = ngk.copy()
        self.basis_dimensions = ns * self.ngk
        self.ngkmax = int(source_wfn.ngkmax)
        self.energies_ry = tuple(p["energies_ry"].copy() for p in parents)
        for array in (self.ngk, self.basis_dimensions, *self.energies_ry):
            array.flags.writeable = False
        self.signature = {"schema": SCHEMA, "seed_wfn_sha256": expected,
                          "archive_sha256": archive_digest,
                          "source_sha256_bindings": bindings,
                          "complete_native_basis": True}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._io is not None:
            io, self._io = self._io, None
            io.close()

    def load_parent_band_tile(self, parent, b0, b1, *, pad_to=None):
        """Return ψ_G ``(1,B,ns,ngkmax)`` and boolean native mask ``(1,B)``.

        ψ_G has ``P(None,('x','y'),None,None)``. Every carrier column and G
        slot outside the exact native spectrum is zero. The returned mask is
        mandatory for all occupations, energy bounds, and external targets;
        an unmasked ``1-f`` would count nonexistent empty states.
        """
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        if self._io is None:
            _refuse("reader is closed")
        if int(parent) != parent or not 0 <= int(parent) < len(self.ngk):
            _refuse("parent lies outside the authenticated source wedge")
        parent = int(parent)
        if int(b1) > int(self.basis_dimensions.max()):
            _refuse("band request exceeds every native spectrum")
        shape, live, valid, _ = native_band_geometry(
            self.basis_dimensions[parent], self.ngk[parent], self.nspinor,
            self.ngkmax, self.mesh.size, b0, b1, pad_to=pad_to,
            max_local_read_bytes=self.max_local_read_bytes)
        spec = P(("x", "y"), None, None, None)
        if live:
            raw = self._io.read_slab(f"k{parent:05d}/coefficients", shape=shape,
                dtype=np.float64, offset=(int(b0), 0, 0, 0),
                valid_shape=(live, self.nspinor, int(self.ngk[parent]), 2),
                partition_spec=spec)
            psi = raw[..., 0] + 1j * raw[..., 1]
        else:
            psi = jax.jit(lambda: jnp.zeros(shape[:-1], jnp.complex128),
                out_shardings=NamedSharding(self.mesh, P(("x", "y"), None, None)))()
        psi = jnp.where(jnp.asarray(valid[0])[:, None, None], psi, 0)[None]
        return psi, valid

    def sample_parent_band_tile(self, parent, b0, b1, centroids, *,
                                pad_to=None, chunk_size=None):
        """Canonical full-Bloch centroid samples and explicit native mask.

        FFT work is O(B*N_FFT*log N_FFT)/P, with B bounded by the tile;
        no G-vector sum is nested with a band-pair contraction. The existing
        transform owns its FFT workspace and output placement. Callers form
        the persistent two μ/band faces through the existing bundle owners.
        """
        from common.wfn_transforms import gflat_to_rmu
        psi, valid = self.load_parent_band_tile(parent, b0, b1, pad_to=pad_to)
        loader = self.source_wfn
        samples = gflat_to_rmu(psi, loader.ibz_box_index_one_dev(int(parent)),
            centroids, mesh=self.mesh, fft_grid=loader.fft_grid,
            kvecs_frac=loader.kvecs(k="ibz")[int(parent):int(parent) + 1],
            norm="ortho", chunk_size=chunk_size)
        return samples, valid
