"""Parity cases for sylva.filters."""

import numpy as np

from sylva import PointCloud, filters


def cloud(seed=1, n=3000):
    """Uniform points in a 10 m cube with an integer, a small-integer and a
    float attribute, plus a few isolated points far from the rest."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 10.0, (n, 3))
    xyz[:5] += 40.0 + rng.uniform(0.0, 10.0, (5, 3))
    attrs = {"intensity": rng.integers(0, 60000, n).astype(np.uint16),
             "id": np.arange(n, dtype=np.int32), "w": rng.normal(0.0, 1.0, n)}
    return PointCloud(xyz, attrs)


def _flat(prefix, c):
    out = {f"{prefix}_xyz": c.xyz}
    for k, v in c.attrs.items():
        out[f"{prefix}_{k}"] = v
    return out


def subsample():
    c = cloud()
    out = {}
    out.update(_flat("voxel_first", filters.voxel_downsample(c, 0.7)))
    out.update(_flat("voxel_centroid", filters.voxel_downsample(c, 0.7, method="centroid")))
    out.update(_flat("random_n", filters.random_subsample(c, n=500, seed=3)))
    out.update(_flat("random_fraction", filters.random_subsample(c, fraction=0.125, seed=4)))
    out.update(_flat("min_distance", filters.min_distance_subsample(c, 0.6)))
    return out


def crops():
    c = cloud(2)
    out = {}
    out.update(_flat("box", filters.crop_box(c, (1.0, 2.0, 3.0), (6.0, 7.5, 9.0))))
    out.update(_flat("box_open", filters.crop_box(c, (None, 2.0, float("nan")), (5.0, None, 4.0))))
    out.update(_flat("cylinder", filters.crop_cylinder(c, (5.0, 4.0), 3.2)))
    out.update(_flat("cylinder_z", filters.crop_cylinder(c, (5.0, 4.0), 3.2, zmin=1.0, zmax=6.5)))
    out.update(_flat("range", filters.range_filter(c, (2.0, 3.0, 1.0), min_range=2.0, max_range=7.0)))
    out.update(_flat("range_default", filters.range_filter(c, max_range=9.0)))
    return out


def outliers():
    c = cloud(3)
    return {
        "sor_mask": filters.statistical_outlier_removal(c, k=8, std_ratio=1.5, return_mask=True),
        "sor_id": filters.statistical_outlier_removal(c).attrs["id"],
        "ror_mask": filters.radius_outlier_removal(c, 0.8, min_neighbors=3, return_mask=True),
        "ror_id": filters.radius_outlier_removal(c, 0.8).attrs["id"],
    }


def geometry():
    c = cloud(4, 1500)
    pl, li = filters.planarity_linearity(c, k=15)
    d, i = filters.knn(c.xyz, c.xyz[::50] + 0.01, 6)
    return {
        "normals": np.abs(filters.estimate_normals(c, k=10)),
        "planarity": pl, "linearity": li,
        "clusters": filters.euclidean_clusters(c.xyz, 0.9, min_points=4),
        "knn_d": d, "knn_i": i,
    }


CASES = {"subsample": subsample, "crops": crops, "outliers": outliers, "geometry": geometry}
