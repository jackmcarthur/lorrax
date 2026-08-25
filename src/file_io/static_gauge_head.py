"""Immutable SlabIO artifact for the gauge-complete static photon head.

The artifact is a physics format above the SlabIO transport.  It stores the
bounded direct head tensor and Hall pseudovector replicated, while the two
body wings stay on their incumbent packed-body axes.  No wavefunction,
centroid body, or band carrier is gathered here.

This module deliberately does not make ``FULL_SCREENED`` reachable.  Its
sealed result proves that bytes passed through this exact loader; the GW
driver must still keep its independent refusal until the gauged VNL producer
is connected to this writer and that connection is reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import barrier
from file_io.slab_io import SlabIO
from gw.head_correction import (
    StaticGaugeHeadResponse,
    require_static_gauge_head_response,
)
from gw.photon_layout import (
    PHOTON_BARE_PROPAGATOR,
    PHOTON_BASIS_ORDERING,
    PhotonBasisLayout,
)


STATIC_GAUGE_HEAD_CONVENTION_ID = (
    "lorrax.static_gauge_head/v1"
    "|fourier=zeta_G(q):FFT_backward[exp(-i*q_cart.r)*zeta(r)]"
    "|current=j:c*Psi^dagger*alpha*Psi"
    "|lorentz=(C,Jx,Jy,Jz)"
    "|q_cart=bohr^-1"
    "|units=lorrax_Rydberg_atomic_units;Omega=bohr^3"
    "|head=Pi_reg(q):q_a*q_b*S_direct[a,b]"
    "|hall_CT=+i*epsilon[b,a,i]*sigma_H[b]*q[a]"
    "|hall_TC=CT^dagger"
    "|normalization=S_direct_includes_1/Omega;Y,Z_unscaled;"
    "Schur_adds_YWZ/Omega"
    f"|packing={PHOTON_BASIS_ORDERING}"
    f"|bare={PHOTON_BARE_PROPAGATOR}"
    "|ibz_unfold=symmetry_maps_service"
    "|dtype=S,Y,Z:complex128;sigma_H:float64"
)

_S_DATASET = "S_direct_cart"
_Y_DATASET = "Y_cart_x"
_Z_DATASET = "Z_cart_y"
_LOADER_TOKEN = object()


@dataclass(frozen=True)
class LoadedStaticGaugeHeadResponse(StaticGaugeHeadResponse):
    """A validated response that can only be issued by this loader.

    The private token distinguishes this result from the public construction
    record.  It is a provenance type, not permission to enable production:
    the lower ``FULL_SCREENED`` refusal remains unconditional until the real
    producer calls the writer below.
    """

    convention_id: str
    artifact_path: str
    source_write_ibz_only: bool
    source_low_mem_bands: bool
    _loader_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._loader_token is not _LOADER_TOKEN:
            raise TypeError(
                "LoadedStaticGaugeHeadResponse is issued only by "
                "load_static_gauge_head_artifact")


def write_static_gauge_head_artifact(
    path: str | Path,
    response: StaticGaugeHeadResponse,
    *,
    mesh_xy: Mesh,
    source_write_ibz_only: bool,
    source_low_mem_bands: bool,
) -> None:
    """Collectively write and atomically publish one immutable artifact.

    ``response`` must already be the output of the one gauged-operator
    transaction.  Its single fingerprint binds the Hamiltonian/configuration,
    sigma.p vertex, VNL/downfolded contact, and Hall response.  The two source
    flags are storage-policy disclosures only: neither changes the convention
    or is permitted to split that fingerprint.

    The final path is create-once.  SlabIO writes ``<path>.partial``; after
    collective close every process races the same atomic hard-link publish,
    validates the winning inode, and removes the temporary name.  No existing
    final or partial path is replaced.
    """
    response = require_static_gauge_head_response(response, mesh_xy)
    layout = response.layout
    if (layout.ordering != PHOTON_BASIS_ORDERING
            or layout.bare_propagator != PHOTON_BARE_PROPAGATOR):
        raise ValueError(
            "StaticGaugeHead artifact accepts only the canonical photon "
            "ordering and bare propagator")
    fixed_dtypes = (
        (response.S_direct, "S_direct", np.dtype(np.complex128)),
        (response.Y_x, "Y_x", np.dtype(np.complex128)),
        (response.Z_y, "Z_y", np.dtype(np.complex128)),
        (response.sigma_H, "sigma_H", np.dtype(np.float64)),
    )
    for array, name, wanted in fixed_dtypes:
        got = np.dtype(array.dtype)
        if got != wanted:
            raise TypeError(
                f"StaticGaugeHead {name} dtype {got} != fixed {wanted}")

    final_path = Path(path)
    partial_path = Path(str(final_path) + ".partial")
    if final_path.name.endswith(".partial"):
        raise ValueError("StaticGaugeHead final path may not end in '.partial'")
    if not final_path.parent.is_dir():
        raise FileNotFoundError(
            f"StaticGaugeHead parent directory does not exist: "
            f"{final_path.parent}")
    if os.path.lexists(final_path):
        raise FileExistsError(
            f"immutable StaticGaugeHead artifact already exists: {final_path}")
    if os.path.lexists(partial_path):
        raise FileExistsError(
            f"stale/in-flight StaticGaugeHead partial exists: {partial_path}")

    n_body = int(layout.packed_extent)
    hall = np.asarray(response.sigma_H)
    convention_i32 = np.frombuffer(
        STATIC_GAUGE_HEAD_CONVENTION_ID.encode("utf-8"),
        dtype=np.uint8).astype(np.int32)
    fingerprint_i32 = np.frombuffer(
        response.hamiltonian_config_operator_fingerprint.encode("ascii"),
        dtype=np.uint8).astype(np.int32)
    with SlabIO(str(partial_path), mode="w", mesh=mesh_xy) as io:
        io.create_dataset(
            _S_DATASET, shape=(2, 2, 4, 4), dtype=np.complex128)
        io.create_dataset(
            _Y_DATASET, shape=(2, 4, n_body), dtype=np.complex128)
        io.create_dataset(
            _Z_DATASET, shape=(2, n_body, 4), dtype=np.complex128)
        io.write_slab(_S_DATASET, response.S_direct)
        io.write_slab(_Y_DATASET, response.Y_x)
        io.write_slab(_Z_DATASET, response.Z_y)
        io.write_attr("sigma_H_cart", hall)
        io.write_attr("convention_id_i32", convention_i32)
        io.write_attr("hamiltonian_config_operator_fingerprint_i32",
                      fingerprint_i32)
        io.write_attr(
            "logical_extents", np.asarray(layout.logical_extents,
                                          dtype=np.int32))
        io.write_attr(
            "padded_extents", np.asarray(layout.padded_extents,
                                         dtype=np.int32))
        io.write_attr("mesh_side", np.int32(layout.mesh_side))
        io.write_attr("ward_residual", np.float64(response.ward_residual))
        io.write_attr(
            "hermiticity_residual",
            np.float64(response.hermiticity_residual))
        io.write_attr(
            "source_write_ibz_only", np.int32(source_write_ibz_only))
        io.write_attr(
            "source_low_mem_bands", np.int32(source_low_mem_bands))
        # Deferred metadata lands only after all three SlabIO writes drain.
        # Publication below occurs only after close returns on every process.
        io.write_attr("complete", np.int32(1))

    try:
        os.link(partial_path, final_path)
    except FileExistsError:
        if not os.path.samefile(partial_path, final_path):
            raise FileExistsError(
                "immutable StaticGaugeHead publish collided with a different "
                f"artifact at {final_path}") from None
    barrier("static_gauge_head_artifact_published")
    try:
        os.unlink(partial_path)
    except FileNotFoundError:
        pass


def load_static_gauge_head_artifact(
    path: str | Path,
    *,
    mesh_xy: Mesh,
    expected_hamiltonian_config_operator_fingerprint: str,
) -> LoadedStaticGaugeHeadResponse:
    """Collectively load one completed artifact onto its native shardings.

    Convention, completion, fingerprint, and layout metadata are checked
    before either body wing is read.  ``Y`` returns on
    ``P(None,None,'x')`` and ``Z`` on ``P(None,'y',None)`` under both band
    storage policies; the loader has no alternate low-memory implementation.
    Only O(1) metadata and the bounded Hall vector cross through
    :meth:`SlabIO.read_small`; neither packed wing is gathered to host.
    """
    artifact_path = Path(path)
    if artifact_path.name.endswith(".partial"):
        raise ValueError("refusing to load a StaticGaugeHead partial path")
    expected = str(expected_hamiltonian_config_operator_fingerprint).strip()
    if (not expected.startswith("sha256:")
            or len(expected) != len("sha256:") + 64
            or any(c not in "0123456789abcdef" for c in expected[7:])):
        raise ValueError(
            "expected StaticGaugeHead fingerprint must be "
            "sha256:<64 lowercase hex>")

    with SlabIO(str(artifact_path), mode="r", mesh=mesh_xy) as io:
        complete = int(np.asarray(io.read_small("complete")))
        convention = np.asarray(
            io.read_small("convention_id_i32"), dtype=np.int32
        ).astype(np.uint8).tobytes().decode("utf-8")
        fingerprint = np.asarray(
            io.read_small(
                "hamiltonian_config_operator_fingerprint_i32"),
            dtype=np.int32,
        ).astype(np.uint8).tobytes().decode("ascii")
        if complete != 1:
            raise ValueError(
                f"StaticGaugeHead artifact is incomplete: complete={complete}")
        if convention != STATIC_GAUGE_HEAD_CONVENTION_ID:
            raise ValueError(
                "StaticGaugeHead convention mismatch; refusing any implicit "
                "Fourier/sign/unit/normalization conversion")
        if fingerprint != expected:
            raise ValueError(
                "StaticGaugeHead Hamiltonian/config/operator fingerprint "
                f"mismatch: artifact={fingerprint!r}, expected={expected!r}")

        logical = tuple(int(v) for v in np.asarray(
            io.read_small("logical_extents"), dtype=np.int32).reshape(-1))
        padded = tuple(int(v) for v in np.asarray(
            io.read_small("padded_extents"), dtype=np.int32).reshape(-1))
        mesh_side = int(np.asarray(io.read_small("mesh_side")))
        write_ibz_only = int(np.asarray(
            io.read_small("source_write_ibz_only")))
        low_mem_bands = int(np.asarray(
            io.read_small("source_low_mem_bands")))
        if write_ibz_only not in (0, 1) or low_mem_bands not in (0, 1):
            raise ValueError(
                "StaticGaugeHead storage-policy stamps must be zero or one")
        ward_residual = float(np.asarray(io.read_small("ward_residual")))
        hermiticity_residual = float(np.asarray(
            io.read_small("hermiticity_residual")))
        sigma_H = io.read_small("sigma_H_cart")

        layout = PhotonBasisLayout(
            logical_extents=logical,
            padded_extents=padded,
            mesh_side=mesh_side,
        )
        layout.assert_mesh(mesh_xy)
        n_body = int(layout.packed_extent)
        S_direct = io.read_slab(
            _S_DATASET, shape=(2, 2, 4, 4), partition_spec=P())
        Y_x = io.read_slab(
            _Y_DATASET, shape=(2, 4, n_body),
            partition_spec=P(None, None, "x"))
        Z_y = io.read_slab(
            _Z_DATASET, shape=(2, n_body, 4),
            partition_spec=P(None, "y", None))

    fixed_dtypes = (
        (S_direct, "S_direct", np.dtype(np.complex128)),
        (Y_x, "Y_x", np.dtype(np.complex128)),
        (Z_y, "Z_y", np.dtype(np.complex128)),
        (sigma_H, "sigma_H", np.dtype(np.float64)),
    )
    for array, name, wanted in fixed_dtypes:
        got = np.dtype(array.dtype)
        if got != wanted:
            raise TypeError(
                f"StaticGaugeHead {name} dtype {got} != fixed {wanted}")

    loaded = LoadedStaticGaugeHeadResponse(
        layout=layout,
        S_direct=S_direct,
        sigma_H=sigma_H,
        Y_x=Y_x,
        Z_y=Z_y,
        hamiltonian_config_operator_fingerprint=fingerprint,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=ward_residual,
        hermiticity_residual=hermiticity_residual,
        convention_id=convention,
        artifact_path=str(artifact_path),
        source_write_ibz_only=bool(write_ibz_only),
        source_low_mem_bands=bool(low_mem_bands),
        _loader_token=_LOADER_TOKEN,
    )
    require_static_gauge_head_response(loaded, mesh_xy)
    return loaded


__all__ = [
    "LoadedStaticGaugeHeadResponse",
    "STATIC_GAUGE_HEAD_CONVENTION_ID",
    "load_static_gauge_head_artifact",
    "write_static_gauge_head_artifact",
]
