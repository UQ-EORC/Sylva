# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Resampling, local geometry and nearest-neighbour search for coregistration.

Scans arrive with tens of millions of points at wildly varying density (dense
near the scanner, sparse at range), which slows ICP down and biases it towards
whichever region is densest; everything here works on voxel centroids.
"""

from __future__ import annotations

import numpy as np

from .. import _core

__all__ = ["KdTree", "estimate_normals", "planar_filter", "voxel_downsample"]


def _xyz(points) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def voxel_downsample(
    points: np.ndarray, voxel: float, *, centroid: bool = True, return_counts: bool = False
):
    """One point per voxel.

    Parameters
    ----------
    points
        ``(n, 3)`` points.
    voxel
        Voxel size (m), on a grid anchored at ``floor(p / voxel)``.
    centroid
        Average the points of each voxel; otherwise keep the first.
    return_counts
        Also return how many points fed each output point.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        ``(m, 3)`` points, ordered by voxel, and optionally ``(m,)`` counts.
    """
    if voxel <= 0:
        raise ValueError("voxel size must be positive")
    pts, counts = _core.coreg_voxel_centroids(_xyz(points), float(voxel), bool(centroid))
    return (pts, counts) if return_counts else pts


def estimate_normals(
    points: np.ndarray, k: int = 20, radius: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and planarity by local PCA.

    Parameters
    ----------
    points
        ``(n, 3)`` points.
    k
        Neighbours, the point itself included.
    radius
        Neighbours further than this (m) are left out; a point with fewer
        than 3 within it gets a zero normal and planarity 0.

    Returns
    -------
    normals : numpy.ndarray
        ``(n, 3)``.
    planarity : numpy.ndarray
        ``(l1 - l0) / l2`` in [0, 1]: how plane-like each neighbourhood is.
    """
    pts = _xyz(points)
    if len(pts) < 3:
        return np.zeros((len(pts), 3)), np.zeros(len(pts))
    normals, planarity, _, _ = _core.coreg_local_pca(pts, int(k), radius)
    return normals, planarity


def planar_filter(
    points: np.ndarray,
    min_planarity: float = 0.35,
    voxel: float | None = 0.05,
    k: int = 20,
    radius: float | None = 0.15,
) -> np.ndarray:
    """Keep the locally planar points: stems, ground and logs.

    The most useful step before ICP in a forest. Crown returns have no stable
    geometry between viewpoints (occlusion, leaves, wind) and make thousands
    of plausible but wrong correspondences; filtering both clouds leaves the
    stable skeleton, vertical stem surfaces plus the terrain that fixes the
    height.

    Parameters
    ----------
    points
        ``(n, 3)`` points.
    min_planarity
        Planarity a point needs.
    voxel
        Voxel centroids at this size first (m); None to skip.
    k, radius
        Neighbourhood of the planarity.

    Returns
    -------
    numpy.ndarray
        ``(m, 3)`` planar points.
    """
    return _core.coreg_planar_filter(_xyz(points), float(min_planarity), voxel, int(k), radius)


class KdTree:
    """Nearest-neighbour search over a fixed set of 2-D or 3-D points.

    Parameters
    ----------
    points
        ``(n, 2)`` or ``(n, 3)`` points.
    """

    def __init__(self, points: np.ndarray) -> None:
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("points must have shape (n, 2) or (n, 3)")
        self.n = len(pts)
        self._tree = _core.CoregKdTree(pts)

    def query(
        self, queries: np.ndarray, distance_upper_bound: float = np.inf
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest point to each query.

        Parameters
        ----------
        queries
            ``(q, d)`` points of the tree's dimension.
        distance_upper_bound
            Only neighbours within this distance count.

        Returns
        -------
        distance : numpy.ndarray
            ``(q,)``, ``inf`` where there is none.
        index : numpy.ndarray
            ``(q,)``, ``n`` where there is none (as scipy's cKDTree).
        """
        q = np.ascontiguousarray(np.asarray(queries, dtype=np.float64))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return self._tree.query(q, float(distance_upper_bound))
