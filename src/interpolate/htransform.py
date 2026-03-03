import os
import re
import argparse
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Allow GPU if available (previously forced CPU)

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erf
from jax.sharding import Mesh
from jax import lax

from isdf_io import WFNReader
from common import symmetry_maps
from common import Meta
from common.load_wfns import read_Gvecs_to_devices, get_sharded_wfns, get_enk_bandrange


def solve_q0_galerkin(
    psi_mu: jax.Array,
    psi_r: jax.Array,
    *,
    rtol: float = 1e-8,
    log_fn=None,
):
    """Solve psi_mu @ Q = psi_r using a Galerkin projection (spin folded into μ)."""
    psi_mu = jnp.asarray(psi_mu, dtype=jnp.complex128)
    psi_r = jnp.asarray(psi_r, dtype=psi_mu.dtype)
    if psi_mu.shape[:-1] != psi_r.shape[:-1]:
        raise ValueError("psi_mu and psi_r must share all leading dimensions")

    nk, nb, nspin, n_mu = psi_mu.shape
    psi_mu_state = psi_mu.reshape(nk * nb, nspin * n_mu)
    psi_r_state = psi_r.reshape(nk * nb, -1)

    U, s, Vh = jnp.linalg.svd(psi_mu_state, full_matrices=False)
    mask = s > (s.max() * rtol)
    if not bool(mask.any()):
        raise ValueError("SVD dropped all singular values; increase rtol")
    U = U[:, mask]
    s = s[mask]
    #basis = Vh[mask].conj().T
    coeffs = U * s
    Q_reduced, *_ = jnp.linalg.lstsq(coeffs, psi_r_state, rcond=rtol)
    G = Q_reduced @ Q_reduced.conj().T
    chol = jnp.linalg.cholesky(G)
    Q_reduced = jsp_linalg.solve_triangular(chol, Q_reduced, lower=True)
    coeffs = coeffs @ chol

    # if log_fn is not None:
    #     residual = jnp.linalg.norm((coeffs @ Q_reduced) - psi_r_state, axis=1)
    #     log_fn(
    #         "Galerkin residual min/median/max: "
    #         f"{float(residual.min()):.3e} / {float(jnp.median(residual)):.3e} / {float(residual.max()):.3e}"
    #     )

    #Q_full = basis @ Q_reduced # do not uncomment.
    S = Q_reduced @ Q_reduced.conj().T
    ctilde = coeffs.reshape(nk, nb, coeffs.shape[1])
    return  S, ctilde


def _f_params_from_energies(enk_nb_nk: jax.Array, top_band_index: int) -> tuple[float, float, float]:
    E_top_k = enk_nb_nk[top_band_index]
    span_top = jnp.max(E_top_k) - jnp.min(E_top_k)
    span_total = jnp.max(E_top_k) - jnp.min(enk_nb_nk)
    a = jnp.maximum(4.0 * (span_top + 1e-14), span_total + 1e-14)
    n = 3.0
    epsilon0 = float(jnp.max(E_top_k))
    return float(a), float(n), epsilon0


def _f_eval_piece(x: jax.Array, a: float, n: float) -> jax.Array:
    a = float(a)
    n = float(n)
    erf_half = erf(n * 0.5)
    y = x
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    f_left = y + 0.5 * a
    arg = n * (0.5 + y / a)
    term1 = a * (jnp.exp(-(n * 0.5) ** 2) - jnp.exp(-((n * (a + 2 * y)) / (2 * a)) ** 2)) / (2 * n * jnp.sqrt(jnp.pi) * erf_half)
    term2 = (a + 2 * y) * (erf_half - erf(arg)) / (4 * erf_half)
    f_mid = term1 + term2
    f = jnp.where(cond_left, f_left, 0.0)
    f = jnp.where(cond_mid, f_mid, f)
    return jnp.where(y >= 0, 0.0, f)


def _f_eval_piece_derivative(x: jax.Array, a: float, n: float) -> jax.Array:
    a = float(a)
    n = float(n)
    erf_half = erf(n * 0.5)
    y = x
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    df = jnp.zeros_like(y)
    df = jnp.where(cond_left, 1.0, df)
    arg = n * (0.5 + y / a)
    df = jnp.where(cond_mid, 0.5 - erf(arg) / (2 * erf_half), df)
    return jnp.where(y >= 0, 0.0, df)


def f_transform_eigs(enk_nb_nk: jax.Array) -> tuple[jax.Array, float, float, float]:
    nb, _ = enk_nb_nk.shape
    a, n, epsilon0 = _f_params_from_energies(enk_nb_nk, top_band_index=nb - 1)
    x = enk_nb_nk - epsilon0
    f_eps = _f_eval_piece(x, a=a, n=n)
    f_eps = jnp.where(f_eps > 0, 0.0, f_eps)
    return f_eps, a, n, epsilon0


def f_inv_newton(y: jax.Array, a: float, n: float, max_iter: int = 64) -> jax.Array:
    f_left = _f_eval_piece(jnp.asarray(-a, dtype=jnp.float64), a, n)
    y_target = jnp.clip(y, f_left, 0.0)

    def body_fun(_, x_curr):
        res = _f_eval_piece(x_curr, a, n) - y_target
        df = _f_eval_piece_derivative(x_curr, a, n)
        step = res / jnp.where(jnp.abs(df) > 1e-12, df, 1.0)
        x_next = jnp.clip(x_curr - step, -a, 0.0)
        return x_next

    return lax.fori_loop(0, max_iter, body_fun, y_target)


def load_wfns_and_enk_for_sigma(wfn, sym, nval: int, ncond: int, nband: int):
    nelec = int(wfn.nelec)
    nsigmarange = (int(nelec - nval), int(nelec + ncond))
    enk_sigma, _ = get_enk_bandrange(wfn, sym, nsigmarange, nsigmarange)
    return nsigmarange, jnp.asarray(enk_sigma).transpose(1, 0)


def setup_wfn_and_sym(wfn_file: str):
    wfn = WFNReader(wfn_file)
    sym = symmetry_maps.SymMaps(wfn)
    return wfn, sym


def _clean_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip()
    if not label:
        return None
    # Map common aliases to actual Unicode Greek letters so Matplotlib displays them
    if label.lower() in {'gg', 'gamma', '\\u0393'}:
        label = 'Γ'
    elif label.lower() in {'gl', 'lambda', '\\u039b'}:
        label = 'Λ'
    elif label.lower() in {'gs', 'sigma', '\\u03a3'}:
        label = 'Σ'
    return label


def generate_kpath_from_qe_segments(params: dict, wfn) -> tuple[jnp.ndarray, np.ndarray, list[str | None]] | None:
    seginfo = params.get("kpoints_crystal_b")
    if not seginfo:
        return None
    segments = seginfo.get("segments", [])
    if len(segments) < 2:
        return None
    nodes_crys = [np.asarray(seg["k"], dtype=float) for seg in segments]
    labels = [_clean_label(seg.get("label")) for seg in segments]
    pts_crys = [nodes_crys[0]]
    node_indices = [0]
    for i in range(len(nodes_crys) - 1):
        k0 = nodes_crys[i]
        k1 = nodes_crys[i + 1]
        n = max(1, int(segments[i + 1].get("n", 1)))
        for t in range(1, n + 1):
            alpha = t / float(n)
            pts_crys.append((1.0 - alpha) * k0 + alpha * k1)
        node_indices.append(len(pts_crys) - 1)
    kpoints = np.stack(pts_crys, axis=0)
    return jnp.asarray(kpoints, dtype=jnp.float64), np.asarray(node_indices, dtype=int), labels


def _build_trivial_mesh() -> Mesh:
    devices = np.asarray(jax.devices())
    if devices.size == 0:
        raise RuntimeError("No JAX devices available")
    return Mesh(devices.reshape(1, devices.size), ['x', 'y'])


def _load_centroids(centroids_path: str, fft_grid: tuple[int, int, int]) -> np.ndarray:
    centroids_frac = np.loadtxt(centroids_path, ndmin=2)
    if centroids_frac.size == 0:
        raise ValueError(f"Centroids file {centroids_path} is empty")
    fft_grid = np.asarray(fft_grid, dtype=int)
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    centroid_indices = np.mod(centroid_indices, fft_grid)
    return centroid_indices


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


def _make_logger(verbose: bool):
    return print if verbose else (lambda *_, **__: None)


def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Parse EQP (or sigX) energies from a BerkeleyGW-style file.

    Supports lines like either:
      - "n=10  EDFT=  ...  EQP= -12.3456 + 0.0000i"
      - "n=10  sigX=  -12.3456 + 0.000000i  VH= ..." (x-only files)

    Returns energies shaped as (nb, nk_full) matching the requested window.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    # Precompile regexes
    k_header = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*$")
    n_line = re.compile(r"n\s*=\s*(\d+)")
    eqp_val = re.compile(r"EQP\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    sigx_val = re.compile(r"sigX\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

    # Accumulate per-k maps from absolute band index -> energy
    energies_by_k: list[dict[int, float]] = []
    current_k: int | None = None

    with open(eqp_file, "r", encoding="utf8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m_k = k_header.match(line)
            if m_k is not None:
                # Start new k-point block
                current_k = int(m_k.group(1))
                # Ensure list is long enough
                while len(energies_by_k) <= current_k:
                    energies_by_k.append({})
                continue

            if current_k is None:
                continue

            # Parse band index and energy value
            m_n = n_line.search(line)
            if m_n is None:
                continue
            band_idx = int(m_n.group(1))

            m_eqp = eqp_val.search(line)
            val = None
            if m_eqp is not None:
                val = float(m_eqp.group(1))
            else:
                m_sigx = sigx_val.search(line)
                if m_sigx is not None:
                    val = float(m_sigx.group(1))

            if val is not None:
                energies_by_k[current_k][band_idx] = val

    if not energies_by_k:
        raise ValueError(f"No k-point blocks found in {os.path.basename(eqp_file)}")

    nk = len(energies_by_k)
    if hasattr(sym, "nk_tot"):
        expected_nk = int(sym.nk_tot)
        if nk != expected_nk:
            raise ValueError(
                f"EQP file has nk={nk}, but symmetry maps expect nk={expected_nk}"
            )
    # Build (nk, nb) by slicing the absolute band indices [start:end]
    energies_window = np.zeros((nk, nb), dtype=np.float64)
    for ik in range(nk):
        kmap = energies_by_k[ik]
        for j, b_abs in enumerate(range(start, end)):
            if b_abs not in kmap:
                raise ValueError(
                    f"Missing EQP for band {b_abs} at k-point {ik} in {os.path.basename(eqp_file)}"
                )
            energies_window[ik, j] = kmap[b_abs]

    # Return (nb, nk) to match internal conventions
    return jnp.asarray(energies_window.T, dtype=jnp.float64)


def initialize_wfns(input_path: str, params: dict, log_fn, eqp_file: str | None = None):
    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    wfn_file = _resolve(params["wfn_file"])
    wfn, sym = setup_wfn_and_sym(wfn_file)
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    centroid_indices = _load_centroids(centroid_path, tuple(int(x) for x in wfn.fft_grid))

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, int(centroid_indices.shape[0]), params.get("bispinor", False))
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn, sym, nval, ncond, nband)

    # Optionally override energies with EQP values from a file only if explicitly requested via CLI
    if eqp_file:
        eqp_path = _resolve(eqp_file)
        if not os.path.isfile(eqp_path):
            log_fn(f"EQP file not found: {eqp_path}")
        else:
            try:
                enk_sigma = read_eqp_energies(eqp_path, sym, nsigmarange)
                log_fn(f"Using EQP energies from {os.path.basename(eqp_path)} for band window {nsigmarange}")
            except Exception as exc:
                log_fn(f"EQP override skipped for {os.path.basename(eqp_path)}: {exc}")

    mesh = _build_trivial_mesh()
    bandrange = (int(nsigmarange[0]), int(nsigmarange[1]))
    global_psiG, nb_actual = read_Gvecs_to_devices(wfn, sym, bandrange, meta, params.get("bispinor", False), mesh)
    psi_rtot, psi_rmu, psi_rmuT = get_sharded_wfns(global_psiG, sym, meta, centroid_indices, nb_actual, False, mesh)
    jax.block_until_ready((psi_rtot, psi_rmu))
    del global_psiG, psi_rmuT

    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, nb={psi_rmu.shape[1]}, mu={psi_rmu.shape[-1]}")
    return wfn, sym, meta, psi_rtot, psi_rmu, enk_sigma


def initialize_kpath(wfn, params):
    info = generate_kpath_from_qe_segments(params, wfn)
    if info is None:
        return None, None, None, None, []
    kpath_frac, node_indices, node_labels = info
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    k_cart = np.asarray(kpath_frac) @ bvec * blat * (2.0 * np.pi)
    seg_len = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    gamma_positions = [int(idx) for idx, lbl in zip(node_indices, node_labels) if (lbl or '').strip() == '\\u0393']
    return kpath_frac, x_path, node_indices, node_labels, gamma_positions


def h_transform(meta, psi_rtot, psi_rmu, enk_sigma, wfn, kpath_data, log_fn):
    S, ctilde = solve_q0_galerkin(psi_rmu, psi_rtot, rtol=1e-8, log_fn=log_fn)
    nk = int(meta.nkx * meta.nky * meta.nkz)
    states = ctilde.shape[1]
    rank = ctilde.shape[2]

    f_eps, a_f, n_f, epsilon0 = f_transform_eigs(enk_sigma)
    log_fn(f"Transform parameters: a={a_f:.6f}, n={n_f:.2f}, eps0={epsilon0:.6f}")

    coeffs = ctilde.reshape(nk, states, rank)
    f_eps_ki = jnp.where(f_eps.T > 0, 0.0, f_eps.T)
    band_weights = jnp.sqrt(jnp.clip(-f_eps_ki, 0.0, None))
    weighted = coeffs * band_weights[..., None]
    fH_k = -jnp.einsum('kim,kin->kmn', weighted, jnp.conj(weighted), optimize=True)
    fH_k = 0.5 * (fH_k + jnp.swapaxes(fH_k, -1, -2).conj())
    log_fn(
        "fH_k real range: [{:.3e}, {:.3e}], |imag|max={:.3e}".format(
            float(jnp.min(jnp.real(fH_k))),
            float(jnp.max(jnp.real(fH_k))),
            float(jnp.max(jnp.abs(jnp.imag(fH_k)))),
        )
    )

    S_sym = (S + S.conj().T) * 0.5
    S_sym += 1e-10 * jnp.mean(jnp.real(jnp.diag(S_sym))) * jnp.eye(rank, dtype=S_sym.dtype)
    S_chol = jnp.linalg.cholesky(S_sym)

    def _solve_generalized(mat):
        y = jsp_linalg.solve_triangular(S_chol, mat, lower=True)
        z = jsp_linalg.solve_triangular(S_chol, y, lower=True, trans=2)
        return jnp.linalg.eigvalsh((z + z.conj().T) * 0.5)

    fH_grid = fH_k.reshape(meta.nkx, meta.nky, meta.nkz, rank, rank).transpose(3, 4, 0, 1, 2)
    # Use the default inverse-FFT normalization (1/N). This pairs with the
    # explicit forward projection below so that the round-trip reproduces fH_k.
    fH_R = jnp.fft.ifftn(fH_grid, axes=(-3, -2, -1), norm=None)
    fH_R = fH_R.transpose(2, 3, 4, 0, 1).reshape(nk, rank, rank)
    fH_R_flat = fH_R.reshape(nk, rank * rank)

    R_grid = jnp.stack(jnp.meshgrid(_shift_indices(meta.nkx), _shift_indices(meta.nky), _shift_indices(meta.nkz), indexing='ij'), axis=-1).reshape(nk, 3)

    def project_to_q(q_frac: jax.Array) -> jax.Array:
        phase = jnp.exp(-2j * jnp.pi * (q_frac @ R_grid.T))
        mat = 0.5 * (phase @ fH_R_flat).reshape(q_frac.shape[0], rank, rank)
        return mat + jnp.swapaxes(mat, 1, 2).conj()

    # Round-trip diagnostic at Γ only. Identify the Γ index in fH_k by
    # matching the projection at q=0 against all k-slices.
    q0 = jnp.zeros((1, 3), dtype=jnp.float64)
    fH_gamma_rt = project_to_q(q0)[0]
    diffs = jnp.max(jnp.abs(fH_k - fH_gamma_rt), axis=(1, 2))
    rt_err = float(jnp.min(diffs))
    log_fn(f"FFT Γ round-trip max error: {rt_err:.3e}")

    fermi_energy = float(wfn.efermi)
    nb_keep = int(f_eps.shape[0])

    kpath_frac, x_path, node_indices, node_labels, gamma_positions = kpath_data
    energies_on_path = None
    energies_sorted = None
    path_range = None
    gamma_exact = None

    if kpath_frac is not None:
        wrapped_k = (kpath_frac + 0.5) % 1.0 - 0.5
        # Process q-points in batches to avoid OOM on large paths
        batch_size = 32
        nq = wrapped_k.shape[0]
        lambda_q_list = []
        for i in range(0, nq, batch_size):
            batch_k = wrapped_k[i:i+batch_size]
            batch_H = project_to_q(batch_k)
            batch_eigs = jax.vmap(_solve_generalized)(batch_H)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)  # Free memory before next batch
        lambda_q = jnp.concatenate(lambda_q_list, axis=0)
        energies_on_path = jax.vmap(lambda row: f_inv_newton(row.real, a=a_f, n=n_f) + epsilon0)(lambda_q)
        energies_sorted = np.asarray(jnp.sort(energies_on_path, axis=1)[:, :nb_keep])
        # Determine Fermi energy as the maximum along path of the wfn.nelec-th band (1-based -> 0-based)
        fermi_band_idx = int(wfn.nelec) - 1
        if 0 <= fermi_band_idx < energies_sorted.shape[1]:
            fermi_energy = float(np.max(energies_sorted[:, fermi_band_idx]))
        if not gamma_positions:
            gamma_positions = [int(jnp.argmin(jnp.linalg.norm(wrapped_k, axis=1)))]
        gamma_exact = np.sort(np.asarray(enk_sigma[:, 0]))[:nb_keep]
        # Shift all reported energies so VBM is at 0
        if energies_on_path is not None:
            energies_on_path = energies_on_path - fermi_energy
        if energies_sorted is not None:
            energies_sorted = energies_sorted - fermi_energy
        if gamma_exact is not None:
            gamma_exact = gamma_exact - fermi_energy
        # Recompute path range after shift and report
        path_range = (float(energies_sorted.min()), float(energies_sorted.max()))
        log_fn(f"Path energy range: {path_range[0]:.6f} to {path_range[1]:.6f} Ry (VBM@0)")
        # Γ deltas (in mRy) remain unchanged by uniform shift
        delta = (energies_sorted[gamma_positions[0]] - gamma_exact) * 1000.0
        log_fn("Γ Δε (mRy): " + ", ".join(f"{d:+.2f}" for d in delta[:6]))
        # After shifting, the Fermi level indicator is at 0
        fermi_energy = 0.0

    return {
        "nk_total": nk,
        "nb_keep": nb_keep,
        "fermi_energy": fermi_energy,
        "energies_on_path": energies_on_path,
        "energies_sorted": energies_sorted,
        "path_range": path_range,
        "gamma_exact": gamma_exact,
        "kpath_data": (kpath_frac, x_path, node_indices, node_labels, gamma_positions),
    }


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    for band in range(nb_keep):
        ax.plot(x_path, energies_sorted[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_sorted[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    energies = np.asarray(energies_on_path)
    kpoints = np.asarray(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy\n')
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hamiltonian interpolation driver")
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    parser.add_argument("--eqp-file", default=None, help="Path to EQP/sigX file to override DFT band energies")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic details")
    args = parser.parse_args(argv)
    log = _make_logger(args.verbose)

    from gw_isdf.gw_jax import read_cohsex_input
    params = read_cohsex_input(args.input)
    
    # Override WFN file if provided via CLI
    if args.wfn_file is not None:
        params["wfn_file"] = args.wfn_file
        log(f"Using WFN file from CLI: {args.wfn_file}")

    wfn, sym, meta, psi_rtot, psi_rmu, enk_sigma = initialize_wfns(args.input, params, log, args.eqp_file)
    kpath_data = initialize_kpath(wfn, params)
    result = h_transform(meta, psi_rtot, psi_rmu, enk_sigma, wfn, kpath_data, log)

    if args.plot:
        plot_bands(result)

    output_dir = os.path.dirname(os.path.abspath(args.input))
    write_bands_to_file(
        os.path.join(output_dir, 'bandstructure.dat'),
        result['energies_sorted'],  # Use sorted & truncated to nb_keep, not raw eigenvalues
        kpath_data[0],
        kpath_data[1],
    )

    summary = f"HT complete: {result['nb_keep']} bands, nk={result['nk_total']}, fermi={result['fermi_energy']:.6f} Ry"
    if result['path_range'] is not None:
        summary += f", path range [{result['path_range'][0]:.6f}, {result['path_range'][1]:.6f}] Ry"
    print(summary)
    return 0


if __name__ == "__main__":
    main()
