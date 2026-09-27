"""Inputs and Python outputs for r/sylva/tests/testthat/test-registration-python.R.

    python tests/parity/export_r_registration.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, save_cloud, write_expected  # noqa: E402

from parity import cases_registration as cr  # noqa: E402
from sylva import PointCloud, registration  # noqa: E402


def main():
    d = OUT / "registration"
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "scene.csv", cr.scene(), delimiter=",", fmt="%.17g")
    rng = np.random.default_rng(12)
    src = rng.uniform(-5, 5, (40, 3))
    m = registration.translation(0.4, -1.2, 0.3) @ registration.rotation_z(12.0)
    dst = src @ m[:3, :3].T + m[:3, 3] + rng.normal(0.0, 0.002, src.shape)
    np.savetxt(d / "kabsch.csv", np.column_stack([src, dst]), delimiter=",", fmt="%.17g")
    rng = np.random.default_rng(13)
    a = PointCloud(rng.uniform(0, 1, (50, 3)), {"i": np.arange(50, dtype=np.uint16), "only_a": np.ones(50)})
    b = PointCloud(rng.uniform(0, 1, (30, 3)), {"i": np.arange(30, dtype=np.uint16)})
    save_cloud(d, "merge_a", a)
    save_cloud(d, "merge_b", b)
    e = {}
    e.update(cr.transforms())
    e.update(cr.kabsch())
    e.update(cr.icp())
    e.update(cr.merge())
    e["icp_source_transform"] = np.linalg.inv(registration.translation(0.15, -0.1, 0.05) @ registration.rotation_z(2.0))
    write_expected(d, e)


if __name__ == "__main__":
    main()
