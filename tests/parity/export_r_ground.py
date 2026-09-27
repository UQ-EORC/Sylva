"""Inputs and Python outputs for r/sylva/tests/testthat/test-ground-python.R.

    python tests/parity/export_r_ground.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, save_cloud, write_expected  # noqa: E402

from parity import cases_ground as cg  # noqa: E402


def main():
    d = OUT / "ground"
    d.mkdir(parents=True, exist_ok=True)
    save_cloud(d, "plot", cg.plot())
    rng = np.random.default_rng(6)
    grid = rng.normal(0.0, 1.0, (7, 9))
    pts = rng.uniform(-1.0, 7.0, (400, 3))
    np.savetxt(d / "grid.csv", grid, delimiter=",", fmt="%.17g")
    np.savetxt(d / "points.csv", pts, delimiter=",", fmt="%.17g")
    e = {}
    e.update({k: v for k, v in cg.classify().items()})
    e.update({k: v for k, v in cg.terrain().items()})
    e.update({f"given_{k}": v for k, v in cg.sample_given().items()})
    write_expected(d, e)


if __name__ == "__main__":
    main()
