import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"00_tools"))
import compile_receipts
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
