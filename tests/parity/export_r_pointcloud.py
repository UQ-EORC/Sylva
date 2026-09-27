"""Inputs and Python outputs for r/sylva/tests/testthat/test-pointcloud-python.R.

    python tests/parity/export_r_pointcloud.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, save_cloud, write_expected  # noqa: E402

from parity import cases_pointcloud as cp  # noqa: E402
from sylva import registration  # noqa: E402


def main():
    d = OUT / "pointcloud"
    d.mkdir(parents=True, exist_ok=True)
    for seed, n in ((21, 200), (22, 40), (23, 25)):
        save_cloud(d, f"cloud{seed}", cp._cloud(seed, n))
    rng = np.random.default_rng(24)
    np.savetxt(d / "array.csv", rng.uniform(0, 1, (30, 6)), delimiter=",", fmt="%.17g")
    m = registration.translation(1.0, 2.0, -3.0) @ registration.rotation_z(71.0)
    m[2, 0] = 0.01
    e = {"matrix": m}
    e.update(cp.methods())
    e.update(cp.concat())
    e.update(cp.from_array())
    write_expected(d, e)


if __name__ == "__main__":
    main()
