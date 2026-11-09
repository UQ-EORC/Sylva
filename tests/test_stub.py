"""The type stub of the compiled core matches the build that is installed."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_core_stub.py"


def test_core_stub_is_current():
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or "run scripts/gen_core_stub.py"
