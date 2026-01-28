from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp


def _round_up(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


@dataclass
class Meta:
    rank: int
    n_proc: int
    b_id_0: int
    b_id_1: int
    b_id_2: int
    b_id_3: int
    b_id_4: int
    fft_grid: tuple
    n_rtot: int
    n_rmu: int
    npol: int
    nfreq: int
    nspin: int
    nspinor: int
    nspinor_wfnfile: int
    nkx: int
    nky: int
    nkz: int
    nk_tot: int
    nbnd_jax: int
    n_rtot_jax: int
    n_rmu_jax: int
    # ============================================================================
    # [NUFFT BACKEND] Q-grid for non-uniform FFT k→R transforms (experimental)
    # ============================================================================
    # If nqx/nqy/nqz differ from nkx/nky/nkz, NUFFT will be used for k→R
    # transforms in ISDF fitting. All q-dependent arrays (C_q, Z_q, V_q, etc.)
    # will be sized to the q-grid instead of the k-grid.
    # ============================================================================
    nqx: int = None  # Q-grid dimensions (defaults to k-grid if not specified)
    nqy: int = None
    nqz: int = None
    nq_tot: int = None

    def __post_init__(self):
        # Cache commonly reused grid/band descriptors to avoid ad-hoc rebuilds elsewhere.
        self.kgrid = (self.nkx, self.nky, self.nkz)
        self.kgrid_np = np.asarray(self.kgrid, dtype=np.int32)
        self.kgrid_jax = jnp.asarray(self.kgrid_np)
        self.fft_grid_np = np.asarray(self.fft_grid, dtype=np.int32)
        self.fft_grid_jax = jnp.asarray(self.fft_grid_np)
        
        # [NUFFT BACKEND] Set q-grid to k-grid if not explicitly specified
        if self.nqx is None:
            self.nqx = self.nkx
        if self.nqy is None:
            self.nqy = self.nky
        if self.nqz is None:
            self.nqz = self.nkz
        if self.nq_tot is None:
            self.nq_tot = self.nqx * self.nqy * self.nqz
        
        # [NUFFT BACKEND] Cache q-grid arrays (for NUFFT or standard FFT)
        self.qgrid = (self.nqx, self.nqy, self.nqz)
        self.qgrid_np = np.asarray(self.qgrid, dtype=np.int32)
        self.qgrid_jax = jnp.asarray(self.qgrid_np)
        
        # [NUFFT BACKEND] Flag to indicate whether NUFFT is needed
        self.use_nufft = (self.nqx != self.nkx or self.nqy != self.nky or self.nqz != self.nkz)
        
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
        q_grid: tuple[int, int, int] | None = None,  # [NUFFT BACKEND] Optional q-grid
    ):
        rank = jax.process_index()
        rank_topo = np.where(np.asarray(jax.devices()) == rank)
        n_proc = jax.process_count()
        b_id_0 = 0
        b_id_1 = int(wfn.nelec - nval)
        b_id_2 = int(wfn.nelec)
        b_id_3 = int(wfn.nelec + ncond)
        b_id_4 = int(nband)
        fft_grid = tuple(int(x) for x in wfn.fft_grid)
        n_rtot = int(np.prod(fft_grid))
        nspin = int(wfn.nspin)
        nspinor_wfnfile = int(wfn.nspinor)
        nspinor = 4 if bispinor else nspinor_wfnfile
        npol = 4 if nspinor == 4 else 1
        nkx, nky, nkz = (int(x) for x in wfn.kgrid)
        nk_tot = int(sym.nk_tot)
        nbnd_jax = _round_up(b_id_4, n_proc)
        n_rtot_jax = _round_up(n_rtot, n_proc)
        n_rmu_jax = _round_up(n_rmu, n_proc)
        
        # [NUFFT BACKEND] Parse optional q-grid
        if q_grid is not None:
            nqx, nqy, nqz = (int(x) for x in q_grid)
            nq_tot = nqx * nqy * nqz
        else:
            nqx, nqy, nqz = nkx, nky, nkz  # Default to k-grid
            nq_tot = nk_tot
        
        return cls(
            rank,
            n_proc,
            b_id_0,
            b_id_1,
            b_id_2,
            b_id_3,
            b_id_4,
            fft_grid,
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
            nbnd_jax,
            n_rtot_jax,
            n_rmu_jax,
            nqx,  # [NUFFT BACKEND] Q-grid dimensions
            nqy,
            nqz,
            nq_tot,
        )
