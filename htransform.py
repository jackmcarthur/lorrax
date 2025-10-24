import os
import sys

# Ensure local src/ is importable when running as a script
PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Force CPU for this entrypoint as well
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from isdf.interpolate.htransform import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.chdir(os.path.join(os.path.dirname(__file__), "tests/cohsex_prod"))
        sys.argv += ["--input", "cohsex_prod.in"]
    raise SystemExit(main())


