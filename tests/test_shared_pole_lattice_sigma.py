"""Ordered Sigma reproduces real-space Sigma = iGW on a time-reversal-broken lattice.

This is the physics gate of the ordered route. An ordered store in the physical orientation
W_q = FT_q[W] (each parent keeps its positive modes; -q is its own parent) feeds the production
shared-pole tau kernel. Conduction windows take W_+(q); valence windows take
shared_pole_hole_kernel's W_+(-q)^T. Both equal -psi^H [G(t) o W_branch(t)] psi on the
Born-von Karman supercell to 1e-10 relative. Feeding the valence window W_+(q)^T instead (the
swapped routing) misses by at least 1e-3.

Scope: CPU 1x1 mesh, the jnp flat-k FFT emulation, identity-group parents through the production
typed unfold plan. The tau quadrature and the omega accumulation carry no q labels and are not
exercised. Oracle: the TRINT Sigma orientation plant, gates G1/G2 (sandbox run
425_trint_20260915, harness/sigma_orientation_plant.py).
"""
import numpy as np

from tr_broken_lattice import Lattice, cpu_flat_k_fft, ft_q, minus_index, mode_momenta, rpa_modes, w_c


def test_ordered_sigma_matches_real_space_igw_and_the_swapped_routing_does_not(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import gw.ppm_tau_kernel as tau_kernel
    from gw.mpa.sigma import shared_pole_hole_kernel, synthesize_shared_pole_parents
    from symmetry_maps import q_negation_index
    from gw.wavefunction_bundle import BandSlices, parent_sigma_operands, sigma_face_kernel_kwargs
    from multi_device.full_photon_head_sigma_gate import _bundle

    import distrib_la

    cpu_flat_k_fft(monkeypatch)
    # The decomposed IFFT, G W, FFT chain through the emulated flat-k transforms (no host FFI library).
    monkeypatch.setattr(tau_kernel, "_fft_ffi_fused_enabled", lambda: False)

    # Every planned GEMM (G build with active ranges, band projection) through the service's
    # local plan: on one CPU device auto would resolve a provider without a warmed kernel.
    local = distrib_la.local_gemm_plan
    monkeypatch.setattr(distrib_la, "gemm_plan", lambda mesh_, layout=None, **options: local(mesh_, **options))
    lat = Lattice(3, 3, 3, n_occ=1)
    e, u = lat.bands()
    psi = lat.bloch_states(e, u)                                  # [nk, nb, N], supercell-normalized
    nk, nb, N = psi.shape
    ns = lat.ns
    cell = np.sqrt(nk) * psi[:, :, :ns]                           # cell-normalized values at the home sites
    occ = np.zeros((nk, nb))
    occ[:, :lat.n_occ] = 1.0
    mu_f = 0.5 * (e[:, 0].max() + e[:, 1].min())
    omega, a, pencil = rpa_modes(psi, e, lat.n_occ, lat.coulomb())
    minus = minus_index(lat)
    assert q_negation_index((lat.n1, lat.n2, 1)).tolist() == minus

    # Instrument: the ordered store in the physical orientation reproduces FT_q[W_c(z)].
    momentum, residual = mode_momenta(lat, a)
    assert residual.max() < 1e-8
    parents = []
    for q in range(nk):
        chosen = np.flatnonzero(momentum == q)
        parents.append((np.sqrt(2 * omega[chosen])[None, :] * np.sqrt(nk) * a[:ns, chosen], omega[chosen] ** 2))
    for z in (0.7 + 0.3j, -1.2 + 0.1j, 2.5j):
        supercell = w_c(z, pencil)
        for q, qf in enumerate(lat.kfrac):
            (b, p2), (bm, p2m) = parents[q], parents[minus[q]]
            om, omm = np.sqrt(p2), np.sqrt(p2m)
            stored = (b / (2 * om * (z - om))) @ b.conj().T - (np.conj(bm) / (2 * omm * (z + omm))) @ bm.T
            want = ft_q(supercell, lat, qf)
            assert np.linalg.norm(stored - want) <= 1e-12 * np.linalg.norm(want)
    odd_even = np.linalg.norm(w_c(0.7j, pencil) - w_c(0.7j, pencil).T) / np.linalg.norm(w_c(0.7j, pencil))
    assert odd_even > 1e-2, "the lattice must break time reversal"

    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda x, spec=P(): jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))
    wfns = _bundle(mesh, cell[:, :, None, :], e, occ, BandSlices.from_band_edges(0, 0, 0, nb, nb),
                   kgrid=(lat.n1, lat.n2, 1))
    packed = int(wfns.green_parent.plan.n_centroid_packed)
    width = max(p[0].shape[1] for p in parents)
    factors = np.stack([np.pad(b, ((0, packed - ns), (0, width - b.shape[1]))) for b, _ in parents])[:, :, None, :]
    poles2 = np.stack([np.pad(p2, (0, width - p2.shape[0]), constant_values=1.0) for _, p2 in parents])
    intervals = np.asarray([[0, b.shape[1]] for b, _ in parents], np.int32)
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P(None, "x", "y")))
    synthesize = jax.jit(lambda x, y, p, r, E, t: synthesize_shared_pole_parents(
        x, y, p, r, E, t, mesh_xy=mesh, gemm=gemm)[0])
    faces = (put(factors, P(None, "x", None, "y")), put(factors, P(None, "y", None, "x")),
             put(poles2), put(intervals))
    hole, minus_q = shared_pole_hole_kernel(mesh), put(np.asarray(minus, np.int32))
    swapped = {"on": False}

    def build(space, _omega, _indices, _bounds, _phase_real, E_ref_B, t_node, _active_count=None):
        plus = synthesize(*faces, E_ref_B, t_node)
        if space != "val":
            return plus
        return jnp.swapaxes(plus, -1, -2) if swapped["on"] else hole(plus, minus_q)

    kernel = tau_kernel.get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=(lat.n1, lat.n2, 1), brackets=None, w_synthesis=build,
        **sigma_face_kernel_kwargs(wfns))
    xn, yr, xr, yn, _, _ = parent_sigma_operands(wfns)

    def production(space, t, ref_a, ref_b):
        energy, mask = (e - mu_f, occ == 0) if space == "cond" else (mu_f - e, occ > 0)
        out = kernel(xn, yr, xr, yn, put(energy, P(None, None)), put(mask, P(None, None)), space, None,
                     put(np.zeros(1, np.int32)), put(np.zeros((1, 6))), put(np.zeros(1, bool)),
                     put(np.float64(ref_a)), put(np.float64(ref_b)), put(np.complex128(t)))
        return np.diagonal(np.asarray(out)[..., :nb, :nb], axis1=-2, axis2=-1)

    def reference(space, t, ref_a, ref_b):
        selected, energy = ((occ == 0), e - mu_f) if space == "cond" else ((occ > 0), mu_f - e)
        G = sum(np.outer(psi[k, n], psi[k, n].conj()) * np.exp(-1j * t * (energy[k, n] - ref_a))
                for k in range(nk) for n in range(nb) if selected[k, n])
        weights = np.exp(-1j * (omega - ref_b) * t)
        W = (a * weights) @ a.conj().T if space == "cond" else (np.conj(a) * weights) @ a.T
        return np.diagonal(-np.einsum("kmr,rs,kns->kmn", psi.conj(), G * W, psi), axis1=-2, axis2=-1)

    worst, control = 0.0, np.inf
    for t in (0.35 - 0.8j, 1.1 - 0.25j, -0.6j):
        for ref_a, ref_b in ((0.0, 0.0), (0.3, -0.2)):
            for space in ("cond", "val"):
                want = reference(space, t, ref_a, ref_b)
                scale = np.max(np.abs(want))
                worst = max(worst, np.max(np.abs(production(space, t, ref_a, ref_b) - want)) / scale)
                if space == "val":
                    swapped["on"] = True
                    control = min(control, np.max(np.abs(production(space, t, ref_a, ref_b) - want)) / scale)
                    swapped["on"] = False
    print(f"GATE lattice Sigma = iGW (3 t x 2 refs x cond/val): max diagonal rel {worst:.2e} <= 1e-10; "
          f"swapped valence routing min rel {control:.2e} >= 1e-3; plant |W - W^T|/|W| {odd_even:.3f}")
    assert worst <= 1e-10, worst
    assert control >= 1e-3, control
