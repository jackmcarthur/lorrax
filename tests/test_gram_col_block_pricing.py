"""Pure capacity gates for pivoted-Cholesky Gram column blocking."""

_GRAM_PRICING_WORKER = r'''
import centroid.pivoted_cholesky as pc

print("GRAM_PRICING_MODULE=" + pc.__file__)

budget = 8 * 2**30
expected = {8: 2896, 81: 908}
widths = {}
for nk, expected_width in expected.items():
    width = pc.auto_gram_col_block_width(
        nk, nspinor=2, budget_bytes=budget, divisor=4,
    )
    widths[nk] = width
    assert width == expected_width, (nk, width, expected_width)
    assert width % 4 == 0
    assert pc.gram_col_block_bytes(nk, 2, width) <= budget
    assert pc.gram_col_block_bytes(nk, 2, width + 4) > budget

assert widths[81] < widths[8], widths

# The registered square law is not the whole physical allocation at P=1:
# pair_density keeps all M candidate rows and blocks only its columns.  The
# runtime cap must therefore be tighter when M exceeds the model width.
p1_model = pc.auto_gram_col_block_width(
    36, nspinor=2, budget_bytes=budget,
)
p1_width = pc._cap_gram_width_to_device_footprint(
    p1_model, nk=36, nspinor=2, n_rows=3600, budget_bytes=budget,
    x_shards=1, y_shards=1,
)
assert p1_width < p1_model, (p1_width, p1_model)
assert pc.gram_col_block_device_bytes(
    36, 2, 3600, p1_width,
) <= budget
assert pc.gram_col_block_device_bytes(
    36, 2, 3600, p1_width + 1,
) > budget

required = pc.gram_col_block_bytes(81, 2, 256)
try:
    pc.auto_gram_col_block_width(81, 2, required - 1, divisor=4)
except MemoryError as exc:
    assert "refuses before pair-density" in str(exc), exc
else:
    raise AssertionError("undersized Gram budget did not refuse")

print(f"GRAM_COL_BLOCK_PRICING_OK nk8={widths[8]} nk81={widths[81]} "
      f"p1_model={p1_model} p1_capped={p1_width}")
'''


def test_auto_gram_col_block_pricing_at_nk8_and_nk81():
    """The registered square-law model fits its budget and shrinks at nk81."""
    import os
    import subprocess
    import sys

    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _GRAM_PRICING_WORKER],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, (
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}")
    print(out.stdout, end="")
    assert "GRAM_COL_BLOCK_PRICING_OK" in out.stdout, out.stdout
