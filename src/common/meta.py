from dataclasses import dataclass
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
                              # logical).  Output writers state n_rmu as the dataset shape;
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
        # ── There is deliberately NO ``band_ranges`` here ────────────────
        # A ``Meta.band_ranges`` SimpleNamespace (+ a ``band_range(name)``
        # accessor) used to live at this point.  Nothing in src/ or tests/
        # ever read it, and its ``sigma=(b1, b3)`` entry CONTRADICTED the
        # real Σ window — reading it as "the bands ρ is built from" cost
        # workstream N a wrong root-cause attribution.  Deleted (AD).
        # The single source of truth for band windows is
        # ``gw.wavefunction_bundle.BandSlices`` (built from
        # ``meta.band_edges`` in ``gw_jax.py``).  Do not reintroduce a
        # second one here.

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
        world_size = int(jax.device_count())
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
