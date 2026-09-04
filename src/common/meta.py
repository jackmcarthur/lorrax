from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from runtime.padding import padded_axis


@dataclass
class Meta:
    rank: int
    n_proc: int
    b_id_0: int
    b_id_1: int
    b_id_2: int
    b_id_3: int
    b_id_4: int               # padded upper bound for the persistent WFN face layouts
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
    # ── THE χ / Σ BAND-COUNT SPLIT (2026-08-16) ──────────────────────────
    # Global top of the χ0/W band sum and of the Σ band sum respectively.
    # 0 means "not supplied" and resolves to ``b_id_4`` in __post_init__,
    # which is what every Meta built by hand (psp tools, tests, the
    # bandstructure path) wants and is bit-identical to the pre-split tree.
    #
    # THE RULE, and it is the whole of the bit-identity argument: the LARGER
    # of the two counts takes the PADDED edge ``b_id_4`` verbatim, and only
    # the smaller one takes a literal user value.  When the two are equal —
    # every deck that names only the umbrella — both are ``b_id_4`` and no
    # slice anywhere changes by a single band, pad bands included.
    #
    # NOTE the pad interaction, which is the one genuinely new thing here.
    # ψ(G) is zeroed on ``[b_id_4_user, b_id_4)`` (the mesh pad), so a
    # consumer could historically over-slice into the pad harmlessly.  Bands
    # in ``[min(chi,sigma), b_id_4_user)`` are NOT pad: they are real states
    # with nonzero ψ that the smaller consumer must not sum.  The narrow
    # slice is load-bearing, not decorative.
    b_id_4_chi: int = 0
    b_id_4_sigma: int = 0

    @property
    def b_id_4_chi_user(self) -> int:
        """LOGICAL (unpadded) top of the χ0/W band sum.

        ``b_id_4_chi`` carries the mesh pad when χ is the larger consumer and
        is literal when it is not; ``b_id_4_user`` is the unpadded max.  The
        min of the two is the count with no pad in it either way, which is
        what a band COUNT means (pad ψ is exactly zero and contributes
        nothing, so it must never appear in an N the physics divides by).
        """
        return min(int(self.b_id_4_chi), int(self.b_id_4_user))

    @property
    def b_id_4_sigma_user(self) -> int:
        """LOGICAL (unpadded) top of the Σ band sum.

        **This is the N the band extrapolation's 0.80/0.90/1.00 fractions are
        of**, and it is deliberately a named property rather than a ``min``
        spelled at each call site: at the call site the difference between
        this, ``b_id_4_user`` (= max(chi, sigma)) and ``b_id_4_chi_user`` is
        invisible on an unsplit deck and 148 bands wide on a split one.
        """
        return min(int(self.b_id_4_sigma), int(self.b_id_4_user))

    def __post_init__(self):
        if not self.b_id_4_chi:
            self.b_id_4_chi = self.b_id_4
        if not self.b_id_4_sigma:
            self.b_id_4_sigma = self.b_id_4
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
        nband_chi: int | None = None,
        nband_sigma: int | None = None,
        mesh_xy=None,
    ):
        rank = jax.process_index()
        rank_topo = np.where(np.asarray(jax.devices()) == rank)
        n_proc = jax.process_count()
        b_id_0 = 0
        # Deck guard: nval counts occupied BANDS, and a spin-restricted
        # scalar WFN (nspinor=1) holds HALF as many of those as the
        # nspinor=2 deck it was ported from (2 electrons per band) — the
        # unhalved nval lands here as a negative b_id_1 and silent nonsense
        # downstream.
        if int(nval) > int(wfn.nelec):
            raise ValueError(
                f"Meta.from_system: nval={int(nval)} valence bands requested "
                f"but the WFN holds only nelec={int(wfn.nelec)} occupied "
                f"bands.  If this deck was ported from an nspinor=2 run to a "
                f"scalar (nspinor=1) WFN, halve its band counts: each scalar "
                f"band holds two electrons.  Fix: nval <= {int(wfn.nelec)}.")
        b_id_1 = int(wfn.nelec - nval)
        # TODO(metal-band-windows): ``wfn.nelec=max(ifmax)`` is only the
        # highest partially occupied band boundary.  Replace this single b2
        # valence/conduction split when finite-q metallic screening and Sigma
        # consume occupation weights rather than integer band manifolds.
        b_id_2 = int(wfn.nelec)
        b_id_3 = int(wfn.nelec + ncond)
        # b_id_4 is the padded upper bound for the two persistent face-WFN
        # layouts.  Their band axes are sharded over x and y separately, so
        # the canonical divisor is lcm(px, py), NOT the device product.  The
        # transient G-sphere loader uses independently product-padded chunks.
        # Output writers slice back to b_id_4_user (the user's nband).
        # ψ(G) for bands in [b_id_4_user, b_id_4) is forced to zero in
        # load_centroids_band_chunked; energies use a finite sentinel; pair
        # densities / Σ_X / Σ_C therefore see zero contribution from pads.
        # ``nband`` is the LOADED extent = max(chi, sigma); the two optional
        # counts are the per-consumer tops inside it.  Passing neither is the
        # unsplit case and reproduces the pre-2026-08-16 arithmetic exactly.
        nband_chi = int(nband if nband_chi is None else nband_chi)
        nband_sigma = int(nband if nband_sigma is None else nband_sigma)
        if max(nband_chi, nband_sigma) != int(nband):
            raise ValueError(
                f"Meta.from_system: nband={int(nband)} must be "
                f"max(nband_chi={nband_chi}, nband_sigma={nband_sigma}) — it "
                f"is the extent the centroid ψ is LOADED over, so it cannot "
                f"be smaller than either consumer (nothing would be there to "
                f"read) and must not be larger than both (that would load "
                f"bands no consumer sums).  Resolution belongs to "
                f"gw_config.resolve_band_counts; do not re-derive it here.")
        b_id_4_user = int(nband)
        world_size = int(jax.device_count())
        if mesh_xy is None:
            # Compatibility for standalone P=1 constructors and old library
            # callers. Production drivers pass their live mesh so the owner
            # derives the divisor from the actual face specs below.
            band_divisor_or_mesh = world_size
            band_specs = None
        else:
            from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
            band_divisor_or_mesh = mesh_xy
            band_specs = ((PSI_NMU_SPEC, 1), (PSI_MUN_SPEC, 3))
        b_id_4 = padded_axis(
            b_id_4_user, band_divisor_or_mesh,
            name="full loaded-band carrier", specs=band_specs).carrier
        # The LARGER count takes the padded edge (so the default path is
        # byte-for-byte what it was); the smaller takes its literal value.
        b_id_4_chi = b_id_4 if nband_chi >= nband_sigma else nband_chi
        b_id_4_sigma = b_id_4 if nband_sigma >= nband_chi else nband_sigma
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
            b_id_4_chi=b_id_4_chi,
            b_id_4_sigma=b_id_4_sigma,
        )
