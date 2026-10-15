"""distrib_la.eigh_local: the one local Hermitian eigensolver, with a per-leg implementation row."""
import io
import pathlib
import tokenize

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from distrib_la import COMPLEX_EIGH_ROUTE, eigh_local, eigh_route
from distrib_la.local_eigh import _reflected_eigh

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _herm(b, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((b, n, n))
    if np.issubdtype(dtype, np.complexfloating):
        x = x + 1j * rng.standard_normal((b, n, n))
    return ((x + np.conj(np.swapaxes(x, -1, -2))) / 2).astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("uplo", [None, "U"])
def test_cpu_is_jnp_eigh_bit_for_bit(dtype, uplo):
    a = jnp.asarray(_herm(3, 12, dtype))
    w0, v0 = jnp.linalg.eigh(a, UPLO=uplo)
    w1, v1 = eigh_local(a, UPLO=uplo)
    assert np.array_equal(np.asarray(w0), np.asarray(w1))
    assert np.array_equal(np.asarray(v0), np.asarray(v1))
    r = eigh_local(a, UPLO=uplo)
    assert r.eigenvalues is r[0] and r.eigenvectors is r[1]


def test_jit_and_vmap_trace_one_eigh():
    a = jnp.asarray(_herm(4, 8, np.complex128))
    text = jax.jit(jax.vmap(eigh_local)).lower(a).as_text()
    assert text.count("eigh") >= 1


def test_the_row_routes_complex_on_rocm_only():
    assert COMPLEX_EIGH_ROUTE == {"ROCM": "reflector"}
    assert eigh_route(jnp.complex128, "ROCM") == "reflector"
    assert eigh_route(jnp.complex64, "ROCM") == "reflector"
    assert eigh_route(jnp.float64, "ROCM") is None
    assert eigh_route(jnp.complex128, "CUDA") is None
    assert eigh_route(jnp.complex128, "cpu") is None
    assert eigh_route(jnp.complex128) is None                      # this process: cpu


@pytest.mark.parametrize("kind", ["random", "real_valued", "diagonal", "zero_row", "block", "zero", "scalar"])
@pytest.mark.parametrize("uplo,sym", [(None, True), ("U", False), ("L", False)])
def test_reflector_route_is_exact(kind, uplo, sym):
    """The ROCm route's math (run here on cpu): same spectrum, orthonormal vectors of A."""
    a = _herm(2, 40, np.complex128, seed=3)
    if kind == "real_valued": a = a.real.astype(np.complex128)
    if kind == "diagonal":    a = np.stack([np.diag(np.diag(m)) for m in a])
    if kind == "zero_row":    a[:, -3:, :] = 0; a[:, :, -3:] = 0
    if kind == "block":       a[:, :20, 20:] = 0; a[:, 20:, :20] = 0
    if kind == "zero":        a[0] = 0                     # one empty slot beside a live one
    if kind == "scalar":      a[1] = 2.5 * np.eye(40)
    w, v = (np.asarray(t) for t in _reflected_eigh(jnp.asarray(a), uplo, sym))
    ref = np.linalg.eigvalsh(a)
    assert np.abs(w - ref).max() <= 1e-12 * max(1.0, np.abs(a).max())
    eye = np.eye(40)[None]
    assert np.abs(np.conj(np.swapaxes(v, -1, -2)) @ v - eye).max() < 1e-12
    assert np.abs(v @ (w[..., None] * np.conj(np.swapaxes(v, -1, -2))) - a).max() < 1e-11 * max(1.0, np.abs(a).max())


def test_src_calls_no_jnp_eigh_directly():
    """One owner for the local eigensolver: src code calls distrib_la.eigh_local."""
    offenders = []
    for path in SRC.rglob("*.py"):
        toks = list(tokenize.generate_tokens(io.StringIO(path.read_text()).readline))
        for i in range(len(toks) - 4):
            if [t.string for t in toks[i:i + 5]] == ["jnp", ".", "linalg", ".", "eigh"]:
                offenders.append(f"{path.relative_to(SRC)}:{toks[i].start[0]}")
    assert not offenders, "call distrib_la.eigh_local instead of jnp.linalg.eigh: " + ", ".join(offenders)
