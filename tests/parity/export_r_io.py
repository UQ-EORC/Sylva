"""Inputs and Python outputs for r/sylva/tests/testthat/test-io-python.R.

    python tests/parity/export_r_io.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, save_cloud, write_expected  # noqa: E402

from parity import cases_io as ci  # noqa: E402


def main():
    d = OUT / "io"
    d.mkdir(parents=True, exist_ok=True)
    save_cloud(d, "cloud", ci._cloud())
    rng = np.random.default_rng(32)
    a = rng.uniform(0, 10, (40, 5))
    np.savetxt(d / "plain.xyz", a, fmt="%.6f")
    np.savetxt(d / "semi.txt", a, fmt="%.6f", delimiter=";", header="x;y;z;a;b", comments="")
    (d / "pts.pts").write_text("40\n" + "\n".join(" ".join(f"{v:.6f}" for v in r) for r in a) + "\n")
    m = rng.normal(size=(4, 4))
    (d / "sop.dat").write_text("\n".join(" ".join(repr(float(v)) for v in r) for r in m) + "\n")
    e = {}
    e.update(ci.round_trips())
    e.update(ci.ascii_columns())
    write_expected(d, e)


if __name__ == "__main__":
    main()
