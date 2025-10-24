import os
import sys
from isdf.gw_isdf.cohsex_jax import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.chdir(os.path.join(os.path.dirname(__file__), "tests/cohsex_debug"))
        sys.argv += ["--input", "cohsex_test.in"]
    raise SystemExit(main())
