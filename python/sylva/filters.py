"""Subsampling, cropping, noise filtering and local geometry."""

from __future__ import annotations

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = [
    "voxel_downsample", "random_subsample", "min_distance_subsample", "crop_box",
    "crop_cylinder", "range_filter", "statistical_outlier_removal", "radius_outlier_removal",
    "estimate_normals", "planarity_linearity", "euclidean_clusters", "knn",
]


def voxel_downsample(cloud: PointCloud, voxel_size: float, method: str = "first") -> PointCloud:
    """Keep one point per voxel.

    ``"first"`` keeps an original point per voxel (attributes preserved);
    ``"centroid"`` returns voxel centroids (attributes dropped).
    """
    if method == "first":
        return cloud[_core.voxel_downsample_indices(cloud.xyz, voxel_size)]
    if method == "centroid":
        return PointCloud(_core.voxel_centroids(cloud.xyz, voxel_size))
    raise ValueError(f"unknown method {method!r}")


def random_subsample(cloud: PointCloud, n: int | None = None, fraction: float | None = None,
                     seed: int = 0) -> PointCloud:
    """Randomly select ``n`` points or a ``fraction`` of points (original order kept)."""
    if (n is None) == (fraction is None):
        raise ValueError("give exactly one of n or fraction")
    if n is None:
        n = int(round(len(cloud) * fraction))
    return cloud[_core.random_indices(len(cloud), n, seed)]


def min_distance_subsample(cloud: PointCloud, distance: float) -> PointCloud:
    """Approximate Poisson-disk subsampling: no two kept points closer than ``distance``."""
    return cloud[_core.min_distance_indices(cloud.xyz, distance)]


def crop_box(cloud: PointCloud, min_xyz, max_xyz) -> PointCloud:
    """Keep points inside an axis-aligned box. Use ``None``/``nan`` for open bounds."""
    lo = np.array([-np.inf if v is None else v for v in min_xyz], dtype=float)
    hi = np.array([np.inf if v is None else v for v in max_xyz], dtype=float)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    return cloud[np.all((cloud.xyz >= lo) & (cloud.xyz <= hi), axis=1)]


def crop_cylinder(cloud: PointCloud, center_xy, radius: float,
                  zmin: float = -np.inf, zmax: float = np.inf) -> PointCloud:
    """Keep points within a vertical cylinder (e.g. a circular plot)."""
    d2 = np.sum((cloud.xyz[:, :2] - np.asarray(center_xy, dtype=float)) ** 2, axis=1)
    return cloud[(d2 <= radius**2) & (cloud.z >= zmin) & (cloud.z <= zmax)]


def range_filter(cloud: PointCloud, origin=(0.0, 0.0, 0.0), min_range: float = 0.0,
                 max_range: float = np.inf) -> PointCloud:
    """Keep points within ``[min_range, max_range]`` of ``origin``."""
    r = np.linalg.norm(cloud.xyz - np.asarray(origin, dtype=float), axis=1)
    return cloud[(r >= min_range) & (r <= max_range)]


def statistical_outlier_removal(cloud: PointCloud, k: int = 8, std_ratio: float = 2.0,
                                return_mask: bool = False):
    """Drop points whose mean distance to ``k`` neighbours exceeds
    ``mean + std_ratio * std`` of that statistic."""
    mask = _core.statistical_outlier_mask(cloud.xyz, k, std_ratio)
    return mask if return_mask else cloud[mask]


def radius_outlier_removal(cloud: PointCloud, radius: float, min_neighbors: int = 4,
                           return_mask: bool = False):
    """Drop points with fewer than ``min_neighbors`` others within ``radius``."""
    mask = _core.radius_outlier_mask(cloud.xyz, radius, min_neighbors)
    return mask if return_mask else cloud[mask]


def estimate_normals(cloud: PointCloud, k: int = 12) -> np.ndarray:
    """Unoriented ``(N, 3)`` normals from PCA of ``k`` nearest neighbours."""
    return _core.estimate_normals(cloud.xyz, k)


def planarity_linearity(cloud: PointCloud, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per-point planarity and linearity from local eigenvalues."""
    return _core.planarity_linearity(cloud.xyz, k)


def euclidean_clusters(xyz: np.ndarray, radius: float, min_points: int = 1) -> np.ndarray:
    """Connected components of the radius graph; small clusters get ``-1``.
    Labels are ordered by decreasing cluster size."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    return _core.euclidean_clusters(xyz, radius, min_points)


def knn(xyz: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(distances, indices)`` of the ``k`` nearest points in ``xyz`` to each query."""
    return _core.knn(np.ascontiguousarray(xyz, dtype=np.float64),
                     np.ascontiguousarray(queries, dtype=np.float64), k)
