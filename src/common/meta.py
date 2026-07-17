from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

from runtime.padding import round_up as _round_up


@dataclass
class Meta:
    rank: int
    n_proc: int
    b_id_0: int
    b_id_1: int
    b_id_2: int
    b_id_3: int
    b_id_4: int               # padded upper bound: divisible by world; used everywhere shapes matter
    fft_grid: tuple
    cell_volume: float
    n_rtot: int
    n_rmu: int                # logical centroid count from the centroid file (== n_rmu_user)
    npol: int
    nfreq: int
    nspin: int
    nspinor: int
    nspinor_wfnfile: int
    nkx: int
    nky: int
    nkz: int
    nk_tot: int
    n_rmu_padded: int = 0     # n_rmu rounded up to ``world_size`` (= jax.device_count() = ∏ p_a
                              # over the device mesh).  Worst-case sharding divisor — any
                              # single- or product-axis PartitionSpec on the μ dim divides this.
                              # Mirrors the band-axis pattern (b_id_4 padded vs b_id_4_user
                              # logical).  Output writers + SlabIO valid_shape= use n_rmu;
                              # in-memory shardings use n_rmu_padded.
    b_id_4_user: int = 0      # original user-supplied nband; b_id_4_user == b_id_4 when no pad. Output writers slice to this.

    def __post_init__(self):
        self.nelec = self.b_id_2
        self.nb_sigma = self.b_id_3 - self.b_id_0
        self.kgrid = (self.nkx, self.nky, self.nkz)
        self.kgrid_np = np.asarray(self.kgrid, dtype=np.int32)
        self.kgrid_jax = jnp.asarray(self.kgrid_np)
        self.fft_grid_np = np.asarray(self.fft_grid, dtype=np.int32)
        self.fft_grid_jax = jnp.asarray(self.fft_grid_np)
        b0, b1, b2, b3, b4 = (
            self.b_id_0,
            self.b_id_1,
            self.b_id_2,
            self.b_id_3,
            self.b_id_4,
        )
        self.band_edges = (b0, b1, b2, b3, b4)
        self.band_ranges = SimpleNamespace(
            valence=(b1, b2),
            conduction=(b2, b3),
            sigma=(b1, b3),
            full=(b0, b4),
            occupied=(b0, b2),
            val_plus_sigma=(b0, b3),
            cond_plus_sigma=(b1, b4),
        )

    def band_range(self, name: str) -> tuple[int, int]:
        if not hasattr(self, "band_ranges"):
            raise AttributeError("Meta.band_ranges not initialised")
        try:
            return getattr(self.band_ranges, name)
        except AttributeError as exc:
            raise KeyError(f"Unknown band window '{name}'") from exc

    @classmethod
    def from_system(
        cls,
        wfn,
        sym,
        nval: int,
        ncond: int,
        nband: int,
        n_rmu: int,
        bispinor: bool = False,
    ):
        rank = jax.process_index()
        rank_topo = np.where(np.asarray(jax.devices()) == rank)
        n_proc = jax.process_count()
        b_id_0 = 0
        b_id_1 = int(wfn.nelec - nval)
        b_id_2 = int(wfn.nelec)
        b_id_3 = int(wfn.nelec + ncond)
        # b_id_4 is the padded upper bound: rounded up so the band axis
        # divides the device mesh.  Sharded readers + Cholesky tiles +
        # FFT helpers all assume b_id_4 - b_id_0 % world_size == 0.
        # Output writers slice back to b_id_4_user (the user's nband).
        # ψ(G) for bands in [b_id_4_user, b_id_4) is forced to zero in
        # load_centroids_band_chunked; energies use a finite sentinel; pair
        # densities / Σ_X / Σ_C therefore see zero contribution from pads.
        b_id_4_user = int(nband)
        # --- Degeneracy-close the SCREENING ISDF fit window's top boundary ---
        # b_id_4 is the ζ/χ0 fit window top (→ gw_init.fit_zeta
        # band_range_right = (b1, b4)).  If nband splits a degenerate multiplet
        # at any k, the fitted ζ̃ — and every V_q/W_q tile built from it — is
        # non-covariant under the crystal symmetry at that k (Round-2 root
        # cause; reports/bse_refactor_map_2026-07-15 PHASE2_LOG, FINDINGS2
        # Task 2/3).  Round the PHYSICAL window top DOWN to the largest
        # degeneracy-closed boundary (BGW TOL_Degeneracy).  This changes
        # SCREENING physics (drops partial-multiplet conduction bands from χ0)
        # but NOT the σ OUTPUT band SET (b_id_3 = nelec + ncond).  b_id_3
        # carries the IDENTICAL exposure — the σ conduction top can split a
        # multiplet too — but closing it would drop reported QP bands, so it is
        # deliberately left untouched here (owner decision 2026-07-16).
        from gw.degen_average import (
            round_band_window_to_closed_shell, TOL_DEGENERACY_RY)
        _energies = getattr(wfn, "energies", None)
        # wfn.energies is (nspin, nk, nband_file); flatten (spin, k) onto the
        # leading axis so the min-k gap respects both spin channels.  A normal
        # WFN stores nband_file ≥ nband, so the b_id_4_user boundary is testable
        # (band b_id_4_user exists); if the file is exactly nband wide the top
        # boundary sits at the file edge and is reported closed.  Skip when
        # energies are absent (a mock wfn) or narrower than the window.
        _e2 = None if _energies is None else np.asarray(_energies)
        if _e2 is not None:
            _e2 = _e2.reshape(-1, _e2.shape[-1])
        if _e2 is not None and b_id_4_user <= _e2.shape[1]:
            b4_closed = round_band_window_to_closed_shell(
                _e2, b_id_4_user, TOL_DEGENERACY_RY, direction="down")
            # b_id_3 ≤ b_id_4 is a hard invariant (BandSlices.from_band_edges,
            # and physically: a reported QP band must lie inside the GF /
            # screening band sum).  If the closed shell falls below b_id_3, the
            # σ top splits the SAME multiplet; we do NOT reduce the σ SET, so
            # clamp the screening window at b_id_3 and warn about the exposure.
            if b4_closed < b_id_3:
                if rank == 0:
                    print(
                        f"!!! WARNING [screening degeneracy]: nband="
                        f"{b_id_4_user} splits a degenerate multiplet and the "
                        f"nearest closed shell ({b4_closed}) is below the "
                        f"σ-output top b_id_3={b_id_3}, which splits the SAME "
                        f"multiplet. Clamping the screening window at b_id_3 "
                        f"(no bands dropped); reduce ncond to a "
                        f"degeneracy-closed value to close the screening "
                        f"window.", flush=True)
                b4_closed = b_id_3
            if b4_closed < b_id_4_user:
                if rank == 0:
                    print(
                        f"!!! WARNING [screening degeneracy]: rounding the "
                        f"ISDF fit window top DOWN {b_id_4_user} → {b4_closed} "
                        f"to keep whole degenerate shells (BGW TOL_Degeneracy "
                        f"{TOL_DEGENERACY_RY:g} Ry); dropping "
                        f"{b_id_4_user - b4_closed} conduction band(s) from "
                        f"the χ0/W screening sum.", flush=True)
                b_id_4_user = b4_closed
        world_size = int(jax.device_count())
        # Compose with mesh-divisibility: round the CLOSED physical top UP to
        # world_size with ZERO pad bands (ψ=0, sentinel energies — see the
        # block above), NEVER with real bands, so the padded window can never
        # re-cross into a multiplet.  Order matters: close first, pad second.
        b_id_4 = _round_up(b_id_4_user, world_size)
        # Both readers handle pad-past-file:
        #   - ``read_Gvecs_to_devices`` (legacy path): pre-zeroed buffer
        #     + capped iteration drop bands past wfn.nbands.
        #   - ``WfnLoader.coeffs_gspace`` (phdf5 path): bulk FFI
        #     read up to the largest world-aligned slice within the
        #     file, then a small replicated tail via h5py + a pure-zero
        #     pad for slots past wfn.nbands.
        # No cap needed here.
        fft_grid = tuple(int(x) for x in wfn.fft_grid)
        cell_volume = float(wfn.cell_volume)
        n_rtot = int(np.prod(fft_grid))
        nspin = int(wfn.nspin)
        nspinor_wfnfile = int(wfn.nspinor)
        nspinor = 4 if bispinor else nspinor_wfnfile
        npol = 4 if nspinor == 4 else 1
        nkx, nky, nkz = (int(x) for x in wfn.kgrid)
        nk_tot = int(sym.nk_tot)
        # n_rmu_padded uses world_size (== ∏ p_a over the device mesh), the
        # worst-case divisor for any single- or product-axis PartitionSpec on
        # the μ dim.  Parallel to b_id_4's use of world_size (line 100).
        # ``padded_mu_extent`` = round_up(n_rmu, world_size) plus the
        # test-only LORRAX_EXTRA_MU_PAD rows (pad-extent-invariance gate).
        from runtime.padding import padded_mu_extent
        n_rmu_padded = padded_mu_extent(n_rmu, world_size)
        return cls(
            rank,
            n_proc,
            b_id_0,
            b_id_1,
            b_id_2,
            b_id_3,
            b_id_4,
            fft_grid,
            cell_volume,
            n_rtot,
            n_rmu,
            npol,
            1,
            nspin,
            nspinor,
            nspinor_wfnfile,
            nkx,
            nky,
            nkz,
            nk_tot,
            n_rmu_padded=n_rmu_padded,
            b_id_4_user=b_id_4_user,
        )
