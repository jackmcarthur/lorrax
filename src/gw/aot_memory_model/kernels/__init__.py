"""Kernel registry — importing this package side-effects registers all kernels."""
from . import chi0_tau_step    # noqa: F401
from . import pair_density     # noqa: F401
from . import cct_lr           # noqa: F401
from . import zct_lr           # noqa: F401
from . import solve_q          # noqa: F401
from . import vq_mu_chunk      # noqa: F401
from . import sigma_kij        # noqa: F401
from . import load_psi_rchunk  # noqa: F401  -- two kernels registered inside
from . import slab_write       # noqa: F401
