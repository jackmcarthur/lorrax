import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
#os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
#import jax
from isdf.gw_isdf.cohsex_jax import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        os.chdir(os.path.join(os.path.dirname(__file__), "examples/cohsex_prod"))
        sys.argv += ["--input", "cohsex_prod.in"]
    raise SystemExit(main())
