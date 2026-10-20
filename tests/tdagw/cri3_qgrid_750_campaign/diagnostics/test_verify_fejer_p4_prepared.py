"""Cheap preparation checks only; never imports JAX or claims CUDA execution."""
import ast
from pathlib import Path
import runpy

import numpy as np


HARNESS = Path(__file__).with_name("verify_fejer_p4_38dbea1.py")


def test_prepared_native_harness_syntax_origin_pin_and_no_reference_override():
    source = HARNESS.read_text()
    tree = ast.parse(source)
    namespace = runpy.run_path(str(HARNESS))
    assert namespace["REVISION"] == "38dbea1665518cb13018b972bf6dd2a059910ed5"
    assert {"response_stage", "sigma_stage", "factory_stage", "main"} <= namespace.keys()
    assert "xla_reference" not in source and "monkeypatch" not in source
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    imports = [node for node in ast.walk(main) if isinstance(node, ast.Import)]
    driver = next(node.lineno for node in imports if any(a.name == "gw.gw_jax" for a in node.names))
    arrays = [node.lineno for node in imports if any(a.name in ("numpy", "jax") for a in node.names)]
    assert all(driver < line for line in arrays)
    assert "<= 120" in source and 'kconv_backend(mesh, (8, 8, 1)) != "mathdx"' in source


def test_literal_window_preserves_constant_and_positive_discrete_average():
    literal = runpy.run_path(str(HARNESS))["literal_crop"]
    rng = np.random.default_rng(43)
    fine = rng.normal(size=(64, 2, 4, 4))+1j*rng.normal(size=(64, 2, 4, 4))
    fine = fine@fine.conj().swapaxes(-1, -2)
    coords = np.indices((8, 8, 1)).reshape(3, -1).T
    q = np.indices((4, 4, 1)).reshape(3, -1).T
    angle = 2*np.pi*(q[:, None]/np.array([4, 4, 1])-coords[None]/np.array([8, 8, 1]))
    kernel = ((abs(1+np.exp(1j*angle[:, :, 0]))**2/2)
              *(abs(1+np.exp(1j*angle[:, :, 1]))**2/2))/64
    # sqrt(Nq/Nk) cancels the unitary transform-size change, yielding
    # normalized averaging with the SAME electronic physical prefactor.
    expected = np.einsum("qk,koij->qoij", kernel, fine)
    actual = literal(fine, (8, 8, 1), (4, 4, 1))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert np.linalg.eigvalsh(actual).min() > 0
