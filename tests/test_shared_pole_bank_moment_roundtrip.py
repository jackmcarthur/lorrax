"""Bank -> constructor moment round trip for an ordered (time-reversal-broken) store.

A planted particle-hole plant with z-moments m_k is written to bank.h5 as M_k = m_k/2 (all four orders,
odd M0/M2 included) through the production writer, read back through the reader the constructor uses,
and turned into the ordered infinity block. The stored convention must survive bitwise, and the ordered
model built from the read moments must reproduce every order m0..m3 and the plant itself.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from types import SimpleNamespace

from file_io.slab_io import SlabIO
from file_io.shared_pole_store import (initialize_shared_pole_bank, read_shared_pole_bank,
                                       validate_shared_pole_bank, write_shared_pole_bank)
from test_shared_pole_bank import _bank_fixture, _matrix
from test_shared_pole_ordered import ZS, _ops, _put, _states, _trim, rel


def _packed_operator(meta, mesh, logical):
    basis = meta.mu_basis
    host = np.zeros((1, basis.n_canonical, basis.n_canonical), np.complex128)
    host[0, :basis.n_logical, :basis.n_logical] = logical
    spec = P(None, "x", "y")
    array = jax.make_array_from_callback(host.shape, NamedSharding(mesh, spec), lambda index: host[index])
    return basis.pack_operator(array, spec=spec)


def test_ordered_bank_moments_round_trip_into_the_constructor(tmp_path):
    from gw.shared_pole_constructor import (
        assemble_ordered_shared_pole_pencil, ordered_moment_identity, reduce_ordered_shared_pole_pencil)
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates

    mesh, meta, tables, recipe, identity = _bank_fixture()
    basis = meta.mu_basis
    plant = _trim(np.random.default_rng(31), 6, basis.n_logical, eps=.4)
    planted = {f"M{k}": plant.moment(k) / 2 for k in range(4)}
    assert np.linalg.norm(planted["M0"]) > 1e-3 and np.linalg.norm(planted["M2"]) > 1e-3

    sym = SimpleNamespace(**vars(tables["sym"]))
    sym.trs_allowed = False
    path = tmp_path / "bank.h5"
    header = initialize_shared_pole_bank(path, meta=meta, tables=dict(tables, sym=sym), recipe=recipe,
                                         identity=identity, mesh_xy=mesh)
    assert header["odd_moments"]
    nq = header["bank_shape"]["nq"]
    device = {name: _packed_operator(meta, mesh, value) for name, value in planted.items()}
    sample = _matrix(meta, mesh, samples=True, value=3)
    for q in range(nq):
        for i in range(2):
            write_shared_pole_bank(path, q_span=(q, q + 1), sample_span=(i, i + 1), Wc=sample,
                                   dWc_ds=-sample, meta=meta, expected_identity=identity, mesh_xy=mesh)
        write_shared_pole_bank(path, q_span=(q, q + 1), M1=device["M1"], M3=device["M3"],
                               meta=meta, expected_identity=identity, mesh_xy=mesh)
        write_shared_pole_bank(path, q_span=(q, q + 1), M0=device["M0"], M2=device["M2"],
                               meta=meta, expected_identity=identity, mesh_xy=mesh)
    header = validate_shared_pole_bank(path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        exact = read_shared_pole_bank(io, (nq - 1, nq), meta=meta, header=header,
                                      fields=("M0", "M1", "M2", "M3"))

    # The stored convention M_k = m_k/2 survives write and read bitwise.
    read = {}
    for name, value in planted.items():
        canonical = basis.unpack_host(basis.unpack_host(np.asarray(exact[name]), axis=-1), axis=-2)
        read[name] = canonical[0, :basis.n_logical, :basis.n_logical]
        assert read[name].tobytes() == value.tobytes(), name

    # The constructor's ordered infinity block from the read moments reproduces every order and the plant.
    mm, eigh = _ops()
    # Three infinity directions: on one direction q^H M0 q and q^H M2 q vanish (M0, M2 are i Im(.)),
    # which would leave the odd orders without a reference.
    qi = np.linalg.eigh(2 * read["M1"])[1][:, -3:]
    for k in (0, 2):
        assert np.linalg.norm(qi.conj().T @ read[f"M{k}"] @ qi) > 1e-3 * np.linalg.norm(read[f"M{k}"])
    infinity = tuple(_put(a) for a in (qi, *(read[f"M{k}"] @ qi for k in range(4))))
    pencil = assemble_ordered_shared_pole_pencil(_states(plant, .9 + .35j), infinity, matmul=mm)
    active = jnp.ones((1, pencil[0].shape[-1]), bool)
    model, signed, diagnostics = reduce_ordered_shared_pole_pencil(pencil, active, eigh=eigh, matmul=mm, gates=gates)
    rows = ordered_moment_identity(signed, infinity, matmul=mm)
    assert all(float(rows[f"m{k}"][0]) < 1e-12 for k in range(4)), {k: float(v[0]) for k, v in rows.items()}
    c, mu, kept = (np.asarray(a[0]) for a in signed)
    for z in ZS:
        value = (c[:, kept] / (z * mu[kept] - 1)) @ c[:, kept].conj().T
        assert rel(value, plant.F(z)) < 1e-12
