import importlib.util,sys
from pathlib import Path
import gw
spec=importlib.util.spec_from_file_location('gw.ppm_tau_kernel',Path(__file__).with_name('tau_candidate.py'))
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
gw.ppm_tau_kernel=module
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
