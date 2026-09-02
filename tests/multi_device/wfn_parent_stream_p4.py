"""P=4 gate for one-parent WFN streaming and device symmetry actions.

Exercises the real multi-child schedule with two band tiles and the three
new rank-local kernels directly.  The synthetic nonsymmorphic arm proves the
phase exponential is a separate one-G-row executable; the real gnppm arm
proves the four-component route stages one G row per parent and does not
restore host child-G construction or a retained full-k FFT index.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def _fail(message: str) -> None:
    print(f"[wfn-parent-stream-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def _assert_local_equal(got, expected, *, label: str, atol: float = 0.0):
    expected = np.asarray(expected)
    for shard in got.addressable_shards:
        actual = np.asarray(shard.data)
        reference = expected[shard.index]
        if not np.allclose(actual, reference, rtol=0.0, atol=atol):
            delta = np.max(np.abs(actual - reference))
            _fail(f"{label}: local max_abs={delta:.3e}")


def _assert_local_pair_equal(got, expected, *, label: str):
    got_shards = got.addressable_shards
    expected_shards = expected.addressable_shards
    if [s.index for s in got_shards] != [s.index for s in expected_shards]:
        _fail(f"{label}: addressable shard indices differ")
    for actual, reference in zip(got_shards, expected_shards):
        a = np.asarray(actual.data)
        b = np.asarray(reference.data)
        if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
            _fail(
                f"{label}: local max_abs={np.max(np.abs(a - b)):.3e}")


def _forbidden_collectives(hlo: str) -> list[str]:
    return [
        op for op in (
            "all-gather", "all-to-all", "all-reduce", "reduce-scatter",
            "collective-permute",
        )
        if re.search(rf"\b{re.escape(op)}(?:-start|-done)?\(", hlo)
    ]


def _direct_kernel_gate(mesh: Mesh) -> tuple[int, int, int, int, int]:
    from common.collectives import device_put_process_local
    from wfn_loader.loader import (
        _parent_bispinor_lift_kernel,
        _parent_box_index_kernel,
        _parent_phase_kernel,
        _parent_to_full_k_unfold_kernel,
    )

    band_sh = NamedSharding(mesh, P(None, ("x", "y"), None, None))
    rep0 = NamedSharding(mesh, P())
    rep1 = NamedSharding(mesh, P(None))
    rep2 = NamedSharding(mesh, P(None, None))

    rng = np.random.default_rng(20260901)
    nb, ng = 8, 19
    parent_host = (
        rng.standard_normal((1, nb, 2, ng))
        + 1j * rng.standard_normal((1, nb, 2, ng))
    ).astype(np.complex128)
    g_host = rng.integers(-5, 6, size=(ng, 3), dtype=np.int32)
    S_host = np.asarray(
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.int32)
    tau_host = np.asarray([0.37, -0.23, 0.11], dtype=np.float64)
    U_host = np.eye(2, dtype=np.complex128)

    parent = device_put_process_local(parent_host, band_sh)
    g_parent = device_put_process_local(g_host, rep2)
    S = device_put_process_local(S_host, rep2)
    tau = device_put_process_local(tau_host, rep1)
    U = device_put_process_local(U_host, rep2)
    tr_true = device_put_process_local(np.asarray(True), rep0)
    tr_false = device_put_process_local(np.asarray(False), rep0)

    phase_fn = _parent_phase_kernel(mesh)
    phase_compiled = phase_fn.lower(S, tau, g_parent).compile()
    phase_hlo = phase_compiled.as_text().lower()
    phase_exp = len(re.findall(r"\bexponential\(", phase_hlo))
    if phase_exp != 1:
        _fail(
            "nonzero phase executable must contain exactly one vector "
            f"exponential, found {phase_exp}")
    forbidden = _forbidden_collectives(phase_hlo)
    if forbidden:
        _fail(f"phase executable has unexpected collective(s) {forbidden}")
    phase = phase_fn(S, tau, g_parent)
    phase.block_until_ready()
    expected_phase = np.exp(
        -1j * (g_host @ (S_host.T @ tau_host)))
    _assert_local_equal(
        phase, expected_phase, label="nonsymmorphic phase", atol=2e-14)

    with_phase_fn = _parent_to_full_k_unfold_kernel(mesh, True)
    with_phase_compiled = with_phase_fn.lower(
        parent, U, phase, tr_true).compile()
    with_phase_hlo = with_phase_compiled.as_text().lower()
    if re.search(r"\bexponential\(", with_phase_hlo):
        _fail("band/spin unfold executable recomputes the phase exponential")
    forbidden = _forbidden_collectives(with_phase_hlo)
    if forbidden:
        _fail(f"phase unfold has unexpected collective(s) {forbidden}")
    child = with_phase_fn(parent, U, phase, tr_true)
    child.block_until_ready()
    expected_child = np.conj(parent_host) * expected_phase[None, None, None, :]
    _assert_local_equal(
        child, expected_child, label="typed nonsymmorphic TR child",
        atol=2e-14)

    no_phase_fn = _parent_to_full_k_unfold_kernel(mesh, False)
    no_phase_compiled = no_phase_fn.lower(parent, U, tr_false).compile()
    no_phase_hlo = no_phase_compiled.as_text().lower()
    if re.search(r"\bexponential\(", no_phase_hlo):
        _fail("zero-translation executable contains an exponential")
    forbidden = _forbidden_collectives(no_phase_hlo)
    if forbidden:
        _fail(f"zero-phase unfold has unexpected collective(s) {forbidden}")
    identity_child = no_phase_fn(parent, U, tr_false)
    identity_child.block_until_ready()
    _assert_local_equal(
        identity_child, parent_host, label="zero-phase identity child")

    grid = (17, 17, 17)
    zero_shift = device_put_process_local(
        np.zeros(3, dtype=np.int32), rep1)
    ngk = device_put_process_local(np.asarray(ng, dtype=np.int32), rep0)
    ng_valid = ng - 3
    g_box_host = g_host.copy()
    g_box_host[ng_valid:] = np.asarray([12345, -7777, 9001])
    g_box = device_put_process_local(g_box_host, rep2)
    ngk = device_put_process_local(
        np.asarray(ng_valid, dtype=np.int32), rep0)
    box_fn = _parent_box_index_kernel(mesh, grid, ng)
    box_compiled = box_fn.lower(g_box, S, zero_shift, ngk).compile()
    box_hlo = box_compiled.as_text().lower()
    box_memory = box_compiled.memory_analysis()
    forbidden = _forbidden_collectives(box_hlo)
    if forbidden:
        _fail(f"one-child FFT index has unexpected collective(s) {forbidden}")
    if re.search(r"\bwhile\(", box_hlo):
        _fail("one-child FFT index scalarized into a while loop")
    box = box_fn(g_box, S, zero_shift, ngk)
    box.block_until_ready()
    g_child_box = np.einsum(
        "ij,gj->gi", S_host, g_box_host[:ng_valid])
    cells = np.ravel_multi_index(
        tuple((g_child_box[:, axis] % grid[axis]) for axis in range(3)),
        grid)
    expected_box = np.full(np.prod(grid), ng, dtype=np.int32)
    expected_box[cells] = np.arange(ng_valid, dtype=np.int32)
    _assert_local_equal(
        box, expected_box.reshape((1, *grid)),
        label="one-child FFT index")

    umklapp_host = np.asarray([1, -2, 0], dtype=np.int32)
    kvec_host = np.asarray([0.1, -0.2, 0.0], dtype=np.float64)
    bvec_host = np.asarray([
        [1.1, 0.2, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.4],
    ], dtype=np.float64)
    umklapp = device_put_process_local(umklapp_host, rep1)
    kvec = device_put_process_local(kvec_host, rep1)
    bvec = device_put_process_local(bvec_host, rep2)
    lift_fn = _parent_bispinor_lift_kernel(mesh, "raw")
    lift_compiled = lift_fn.lower(
        identity_child, g_parent, S, umklapp, kvec, bvec).compile()
    lift_hlo = lift_compiled.as_text().lower()
    forbidden = _forbidden_collectives(lift_hlo)
    if forbidden:
        _fail(f"parent bispinor lift has unexpected collective(s) {forbidden}")
    lifted = lift_fn(identity_child, g_parent, S, umklapp, kvec, bvec)
    lifted.block_until_ready()
    from common.bispinor_init import lift_to_4spinor
    from symmetry_maps import unfold_reciprocal_carriers
    child_g = unfold_reciprocal_carriers(S_host, g_host, umklapp_host)
    expected_lift = np.asarray(lift_to_4spinor(
        jnp.asarray(parent_host), jnp.asarray(child_g[None], dtype=jnp.float64),
        jnp.asarray(kvec_host[None]), jnp.asarray(bvec_host)))
    _assert_local_equal(
        lifted, expected_lift, label="device parent-G bispinor lift",
        atol=2e-14)
    if tuple(lifted.sharding.spec) != (None, ("x", "y"), None, None):
        _fail(f"bispinor lift output sharding is {lifted.sharding.spec}")
    return (
        phase_exp, ng, nb,
        int(box_memory.temp_size_in_bytes),
        int(box_memory.output_size_in_bytes),
    )


def _real_multiband_schedule_gate(mesh: Mesh) -> tuple[int, int, int, int]:
    import common.wfn_transforms as transforms
    from common.wfn_transforms import load_centroids_band_chunked
    from wfn_loader import IBZRows, WfnLoader

    path = ROOT / "tests" / "regression" / "gnppm_debug" / "WFN.h5"
    if not path.exists():
        _fail(f"missing checked-in WFN fixture {path}")

    with WfnLoader(str(path), mesh=mesh, backend="eager") as loader:
        sym = loader.symmetry()
        groups = loader.full_k_parent_groups()
        nk_full = int(sym.nk_tot)
        if not any(len(children) > 1 for _, children in groups):
            _fail("gnppm fixture has no reusable parent")
        if np.any(np.abs(loader.translations) > 1e-12):
            _fail("gnppm fixture no longer pins the zero-phase schedule")

        nb, band_tile = min(8, int(loader.nbands)), 4
        n_band_tiles = (nb + band_tile - 1) // band_tile
        r_mu = jnp.asarray([[0, 0, 0], [1, 1, 1]], dtype=jnp.int32)
        meta = SimpleNamespace(
            nk_tot=nk_full, nspinor=4,
            fft_grid=tuple(int(v) for v in loader.fft_grid),
            memory_per_device_gb=1000.0, b_id_4_user=nb,
        )
        band_spec = P(None, ("x", "y"), None, None)
        preloaded = loader.load(
            bands=(0, nb), k="full_bz", sharding=band_spec,
            bispinor=True)
        ref_y, ref_x = load_centroids_band_chunked(
            loader, sym, meta, r_mu, True, mesh, (0, nb),
            band_chunk_size=nb, psi_G_flat=preloaded)
        # The reference path populated the full-k caches by design. Clear
        # them before the production schedule so cache absence is measured.
        loader._gvecs_cache.clear()
        loader._gvecs_dev_cache.clear()
        loader._parent_unfold_g_cache = None

        parent_g_requests: list[int] = []
        legacy_lift_requests: list[tuple[int, ...]] = []
        host_gvec_requests: list[tuple[int, ...] | str] = []
        one_box_requests: list[int] = []
        fft_k_extents: list[int] = []
        original_parent_g = loader._host_parent_g_row
        original_lift = loader._apply_bispinor_lift
        original_gvecs = loader.gvecs
        original_one_box = loader.full_k_box_index_one_dev
        original_fft = transforms.gflat_to_rmu

        def _parent_g(parent):
            parent_g_requests.append(int(parent))
            return original_parent_g(parent)

        def _legacy_lift(*args, **kwargs):
            legacy_lift_requests.append(tuple(
                int(v) for v in np.asarray(kwargs["k"]).reshape(-1)))
            return original_lift(*args, **kwargs)

        def _gvecs(*args, **kwargs):
            raw = kwargs.get("k", "full_bz")
            host_gvec_requests.append(
                raw if isinstance(raw, str) else tuple(
                    int(v) for v in np.asarray(raw).reshape(-1)))
            return original_gvecs(*args, **kwargs)

        def _fft(*args, **kwargs):
            fft_k_extents.append(int(args[0].shape[0]))
            return original_fft(*args, **kwargs)

        def _one_box(full_k):
            one_box_requests.append(int(full_k))
            return original_one_box(full_k)

        def _full_box(*_args, **_kwargs):
            _fail("one-k schedule built a retained full-k FFT index")

        loader._host_parent_g_row = _parent_g
        loader._apply_bispinor_lift = _legacy_lift
        loader.gvecs = _gvecs
        loader.full_k_box_index_one_dev = _one_box
        loader.box_index_dev = _full_box
        transforms.gflat_to_rmu = _fft
        try:
            got_y, got_x = load_centroids_band_chunked(
                loader, sym, meta, r_mu, True, mesh, (0, nb),
                band_chunk_size=band_tile, k_chunk_size=1)
        finally:
            transforms.gflat_to_rmu = original_fft

        all_parents = [int(parent) for parent, _children in groups]
        reusable = [
            int(parent) for parent, children in groups if len(children) > 1]
        singleton = [
            (int(children[0]),)
            for _, children in groups if len(children) == 1
            for _ in range(n_band_tiles)
        ]
        expected_children = [
            int(child)
            for _parent, children in groups
            for child in children
        ]
        if parent_g_requests != all_parents:
            _fail(
                f"parent-G requests {parent_g_requests}, expected {all_parents}")
        if one_box_requests != expected_children:
            _fail(
                f"one-box requests {one_box_requests}, expected "
                f"{expected_children}")
        if legacy_lift_requests:
            _fail(
                f"legacy lift reached the parent stream: {legacy_lift_requests}")
        if host_gvec_requests:
            _fail(f"host child-G cache reached the parent stream: "
                  f"{host_gvec_requests}")
        if loader._gvecs_cache or loader._gvecs_dev_cache:
            _fail("one-k schedule retained a full-k G/index cache")
        if fft_k_extents != [1] * (nk_full * n_band_tiles):
            _fail(
                f"FFT k extents are {fft_k_extents}; expected strict one-k")
        _assert_local_pair_equal(got_y, ref_y, label="real 4c Y face")
        _assert_local_pair_equal(got_x, ref_x, label="real 4c X face")
        return len(groups), nk_full, len(reusable), len(singleton)


def main() -> None:
    if (jax.process_count() != 4 or jax.device_count() != 4
            or jax.local_device_count() != 1):
        _fail(
            "requires four ranks with one GPU each; got "
            f"processes={jax.process_count()}, devices={jax.device_count()}, "
            f"local_devices={jax.local_device_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    phase_receipt = _direct_kernel_gate(mesh)
    schedule_receipt = _real_multiband_schedule_gate(mesh)
    if jax.process_index() == 0:
        print(
            "[wfn-parent-stream-p4] PASS: one exponential/G executable; "
            "zero-phase skip; zero kernel collectives; P4 4c parity; "
            f"phase={phase_receipt}; schedule={schedule_receipt}; "
            f"source={os.path.realpath(__file__)}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
