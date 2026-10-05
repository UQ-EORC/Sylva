# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
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


def voxel_downsample(cloud: PointCloud, voxel_size: float, method: str = "first",
                     origin=None) -> PointCloud:
    """Thin a cloud to at most one point per cubic voxel.

    Evens out the density falloff with range from each scanner, which
    otherwise weights everything near the scanner. The grid starts at the
    cloud's minimum corner, or at ``origin``.

    Parameters
    ----------
    cloud
        Input points.
    voxel_size
        Voxel edge length (m).
    method : {"first", "centroid"}
        ``"first"`` keeps one original point per voxel (the first in file
        order), with its attributes; ``"centroid"`` returns the mean of the
        points in each voxel and drops attributes.
    origin
        ``(x, y, z)`` of a corner of the voxel grid, so that different
        clouds are thinned on one grid (``(0, 0, 0)`` for voxels at
        multiples of ``voxel_size``, as :func:`sylva.als.tiles.from_scans` and
        :func:`sylva.als.tiles.voxel_downsample` use); ``"first"`` only.

    Returns
    -------
    PointCloud
        With ``"first"``, points stay in their original order.

    Raises
    ------
    ValueError
        For an unknown ``method``, or an ``origin`` with ``"centroid"``.
    """
    if origin is not None:
        if method != "first":
            raise ValueError("origin is only supported with method='first'")
        o = tuple(float(v) for v in origin)
        if len(o) != 3 or not np.all(np.isfinite(o)):
            raise ValueError(f"origin must be three finite numbers, got {origin!r}")
        if not (np.isfinite(voxel_size) and voxel_size > 0):
            raise ValueError(f"voxel_size must be a positive number, got {voxel_size}")
        return cloud[_core.voxel_downsample_indices_at(cloud.xyz, float(voxel_size), o)]
    if method == "first":
        return cloud[_core.voxel_downsample_indices(cloud.xyz, voxel_size)]
    if method == "centroid":
        return PointCloud(_core.voxel_centroids(cloud.xyz, voxel_size), crs=cloud.crs)
    raise ValueError(f"unknown method {method!r}")


def random_subsample(cloud: PointCloud, n: int | None = None, fraction: float | None = None,
                     seed: int = 0) -> PointCloud:
    """Random subsample without replacement.

    Parameters
    ----------
    cloud
        Input points.
    n
        Number of points to keep.
    fraction
        Fraction of points to keep (0-1). Give exactly one of ``n`` and
        ``fraction``.
    seed
        Random seed; the same seed gives the same subsample.

    Returns
    -------
    PointCloud
        The selected points in their original order.

    Raises
    ------
    ValueError
        If both or neither of ``n`` and ``fraction`` are given.
    """
    if (n is None) == (fraction is None):
        raise ValueError("give exactly one of n or fraction")
    if n is None:
        n = int(round(len(cloud) * fraction))
    return cloud[_core.random_indices(len(cloud), n, seed)]


def min_distance_subsample(cloud: PointCloud, distance: float) -> PointCloud:
    """Thin to an even spacing: no two kept points closer than ``distance``.

    An approximate Poisson-disk sample. Unlike :func:`voxel_downsample` the
    result has no grid pattern, which suits normals, curvature and
    segmentation.

    Parameters
    ----------
    cloud
        Input points.
    distance
        Minimum spacing (m).

    Returns
    -------
    PointCloud
        Original points (with attributes) in their original order.
    """
    return cloud[_core.min_distance_indices(cloud.xyz, distance)]


def crop_box(cloud: PointCloud, min_xyz, max_xyz) -> PointCloud:
    """Keep points inside an axis-aligned box (bounds inclusive).

    Parameters
    ----------
    cloud
        Input points.
    min_xyz, max_xyz
        Three lower and three upper bounds. ``None`` or NaN leaves that side
        open, e.g. ``crop_box(c, (None, None, 0.5), (None, None, None))``.

    Returns
    -------
    PointCloud
    """
    lo = [np.nan if v is None else float(v) for v in min_xyz]
    hi = [np.nan if v is None else float(v) for v in max_xyz]
    return cloud[_core.crop_box_mask(cloud.xyz, lo, hi)]


def crop_cylinder(cloud: PointCloud, center_xy, radius: float,
                  zmin: float = -np.inf, zmax: float = np.inf) -> PointCloud:
    """Keep points inside a vertical cylinder, e.g. a circular plot.

    Parameters
    ----------
    cloud
        Input points.
    center_xy
        Plot centre ``(x, y)``.
    radius
        Horizontal radius (m).
    zmin, zmax
        Vertical limits on z (not on height above ground).

    Returns
    -------
    PointCloud
    """
    cx, cy = np.asarray(center_xy, dtype=float)
    mask = _core.crop_cylinder_mask(cloud.xyz, float(cx), float(cy), float(radius), float(zmin),
                                    float(zmax))
    return cloud[mask]


def range_filter(cloud: PointCloud, origin=(0.0, 0.0, 0.0), min_range: float = 0.0,
                 max_range: float = np.inf) -> PointCloud:
    """Keep points by 3D distance from a scanner.

    Parameters
    ----------
    cloud
        Input points, in the same frame as ``origin``.
    origin
        Scanner position.
    min_range, max_range
        Distance limits (m), inclusive.

    Returns
    -------
    PointCloud
    """
    origin = [float(v) for v in np.asarray(origin, dtype=float)]
    return cloud[_core.range_mask(cloud.xyz, origin, float(min_range), float(max_range))]


def statistical_outlier_removal(cloud: PointCloud, k: int = 8, std_ratio: float = 2.0,
                                return_mask: bool = False):
    """Remove isolated points by their distance to neighbours (Rusu et al. 2008).

    For each point the mean distance to its ``k`` nearest neighbours is
    computed; points where this exceeds ``mean + std_ratio * std`` over the
    whole cloud are dropped. Removes flying points and edge effects. Because
    the threshold is global, run it per scan or on a thinned cloud when
    density varies strongly with range.

    This is the SOR filter of CloudCompare (``CloudSamplingTools::sorFilter``):
    the same mean distance (the point itself excluded), population mean and
    standard deviation, and keep condition. ``k=6, std_ratio=1.0`` gives
    CloudCompare's defaults.

    Parameters
    ----------
    cloud
        Input points.
    k
        Neighbours per point.
    std_ratio
        Threshold in standard deviations; lower removes more.
    return_mask
        Return the boolean keep-mask instead of the filtered cloud.

    Returns
    -------
    PointCloud or numpy.ndarray
        The kept points, or the mask (True = keep) with ``return_mask``.
    """
    mask = _core.statistical_outlier_mask(cloud.xyz, k, std_ratio)
    return mask if return_mask else cloud[mask]


def radius_outlier_removal(cloud: PointCloud, radius: float, min_neighbors: int = 4,
                           return_mask: bool = False):
    """Remove points with too few neighbours within a fixed radius.

    Parameters
    ----------
    cloud
        Input points.
    radius
        Search radius (m).
    min_neighbors
        Points with fewer other points within ``radius`` are dropped.
    return_mask
        Return the boolean keep-mask instead of the filtered cloud.

    Returns
    -------
    PointCloud or numpy.ndarray
        The kept points, or the mask (True = keep) with ``return_mask``.
    """
    mask = _core.radius_outlier_mask(cloud.xyz, radius, min_neighbors)
    return mask if return_mask else cloud[mask]


def estimate_normals(cloud: PointCloud, k: int = 12) -> np.ndarray:
    """Surface normals from local principal components.

    Parameters
    ----------
    cloud
        Input points.
    k
        Neighbours used for each point's covariance.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` unit normals (eigenvector of the smallest eigenvalue).
        The sign is arbitrary; orient them yourself if needed.
    """
    return _core.estimate_normals(cloud.xyz, k)


def planarity_linearity(cloud: PointCloud, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Local shape descriptors from neighbourhood eigenvalues.

    With the covariance eigenvalues of the ``k`` neighbours sorted
    ``l1 <= l2 <= l3``, planarity is ``(l2 - l1) / l3`` and linearity is
    ``(l3 - l2) / l3`` (Weinmann et al. 2015). Both lie in 0-1; stems and
    branches are linear, leaves and ground planar.

    Parameters
    ----------
    cloud
        Input points.
    k
        Neighbours per point.

    Returns
    -------
    planarity, linearity : numpy.ndarray
        One value per point.
    """
    return _core.planarity_linearity(cloud.xyz, k)


def euclidean_clusters(xyz: np.ndarray, radius: float, min_points: int = 1) -> np.ndarray:
    """Group points into clusters joined by chains of close points.

    Two points are connected if they are within ``radius``; clusters are the
    connected components.

    Parameters
    ----------
    xyz
        ``(N, 3)`` coordinates (e.g. ``cloud.xyz``).
    radius
        Linking distance (m).
    min_points
        Clusters smaller than this are labelled -1.

    Returns
    -------
    numpy.ndarray
        int64 label per point: 0 is the largest cluster, 1 the next, ...;
        -1 for points in clusters that are too small.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    return _core.euclidean_clusters(xyz, radius, min_points)


def knn(xyz: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours of query points (kd-tree).

    Parameters
    ----------
    xyz
        ``(N, 3)`` points to search.
    queries
        ``(M, 3)`` query points. A query that is also in ``xyz`` finds
        itself at distance 0.
    k
        Neighbours per query.

    Returns
    -------
    distances, indices : numpy.ndarray
        ``(M, k)`` arrays, nearest first.
    """
    return _core.knn(np.ascontiguousarray(xyz, dtype=np.float64),
                     np.ascontiguousarray(queries, dtype=np.float64), k)
