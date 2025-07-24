from isdf.gw_isdf.cohsex_isdf import main
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.chdir(os.path.join(os.path.dirname(__file__), "examples/cohsex_test"))
        sys.argv += ["--input", "cohsex_test.in"]
    raise SystemExit(main())
