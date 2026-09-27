"""Inputs and Python outputs for r/sylva/tests/testthat/test-filters-python.R.

    python tests/parity/export_r_filters.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, save_cloud, write_expected  # noqa: E402

from parity import cases_filters as cf  # noqa: E402
from sylva import filters  # noqa: E402


def main():
    d = OUT / "filters"
    d.mkdir(parents=True, exist_ok=True)
    for seed in (1, 2, 3):
        save_cloud(d, f"cloud{seed}", cf.cloud(seed))
    save_cloud(d, "cloud4", cf.cloud(4, 1500))
    e = {}
    s = cf.subsample()
    e["voxel_first_id"] = s["voxel_first_id"]
    e["voxel_first_xyz"] = s["voxel_first_xyz"]
    e["voxel_centroid"] = s["voxel_centroid_xyz"]
    for k in ("random_n", "random_fraction", "min_distance"):
        e[f"{k}_id"] = s[f"{k}_id"]
    c = cf.crops()
    for k in ("box", "box_open", "cylinder", "cylinder_z", "range", "range_default"):
        e[f"{k}_id"] = c[f"{k}_id"]
    o = cf.outliers()
    e["sor_mask"] = o["sor_mask"]
    e["sor_id"] = o["sor_id"]
    e["ror_mask"] = o["ror_mask"]
    e["ror_id"] = o["ror_id"]
    g = cf.geometry()
    e["normals"] = g["normals"]
    e["planarity"] = g["planarity"]
    e["linearity"] = g["linearity"]
    e["clusters"] = g["clusters"]
    e["knn_d"] = g["knn_d"]
    e["knn_i"] = g["knn_i"] + 1
    c4 = cf.cloud(4, 1500)
    e["normals_registration"] = np.abs(filters.estimate_normals(c4))
    write_expected(d, e)


if __name__ == "__main__":
    main()
