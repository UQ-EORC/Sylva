"""Python outputs for r/sylva/tests/testthat/test-limits-python.R.

    python tests/parity/export_r_limits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, write_expected  # noqa: E402

from parity import cases_limits as cl  # noqa: E402


def main():
    d = OUT / "limits"
    d.mkdir(parents=True, exist_ok=True)
    e = {}
    e.update(cl.human())
    e.update(cl.budget())
    e["message"] = str(e["message"])
    write_expected(d, e)


if __name__ == "__main__":
    main()
