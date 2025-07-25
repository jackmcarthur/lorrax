import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isdf.gw_isdf import cohsex_jax


def test_cohsex_jax_example(tmp_path):
    example_dir = os.path.join(ROOT, "examples", "cohsex_test")
    cwd = os.getcwd()
    try:
        os.chdir(example_dir)
        out_file = "eqp0_noqsym.dat"
        if os.path.exists(out_file):
            os.remove(out_file)
        cohsex_jax.main(["--input", "cohsex_test.in"])
        assert os.path.exists(out_file)
    finally:
        os.chdir(cwd)
