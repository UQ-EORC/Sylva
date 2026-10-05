# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Rigid transforms (SO(3) and SE(3)) used throughout coregistration.

Transforms are 4x4 homogeneous matrices in column-vector convention,
``q = T @ [p, 1]``. Twists are ordered ``xi = [wx, wy, wz, tx, ty, tz]``,
rotation first, the convention of the pose-graph solver. The maps and fits
run in the Rust core.
"""

from __future__ import annotations

import numpy as np

from .. import _core

__all__ = [
    "identity",
    "invert",
    "kabsch",
    "kabsch_2d_yaw",
    "rotation_angle",
    "se3_exp",
    "se3_log",
    "skew",
    "so3_exp",
    "so3_log",
    "transform_difference",
    "transform_points",
    "transform_vectors",
    "yaw_transform",
]


def _mat4(T) -> np.ndarray:
    """A float64 4x4 from a 4x4 (or 3x4) transform."""
    T = np.asarray(T, dtype=float)
    if T.shape == (3, 4):
        T = np.vstack([T, [0.0, 0.0, 0.0, 1.0]])
    return np.ascontiguousarray(T)


def _xyz(points) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    return pts


def identity() -> np.ndarray:
    """The 4x4 identity transform."""
    return np.eye(4)


def skew(v: np.ndarray) -> np.ndarray:
    """The 3x3 skew-symmetric matrix of a 3-vector."""
    return _core.coreg_skew(np.asarray(v, dtype=float).reshape(3))


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    return _core.coreg_so3_exp(np.asarray(w, dtype=float).reshape(3))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    return _core.coreg_so3_log(np.ascontiguousarray(np.asarray(R, dtype=float)[:3, :3]))


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Transform of a twist ``[w, t]``.

    Parameters
    ----------
    xi
        ``(6,)`` twist, rotation first.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` transform.
    """
    return _core.coreg_se3_exp(np.asarray(xi, dtype=float).reshape(6))


def se3_log(T: np.ndarray) -> np.ndarray:
    """Twist ``[w, t]`` of a transform; the inverse of :func:`se3_exp`."""
    return _core.coreg_se3_log(_mat4(T))


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform."""
    return _core.coreg_invert(_mat4(T))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a transform to ``(n, 3)`` points."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3).copy()
    return _core.coreg_transform_points(_mat4(T), _xyz(pts))


def transform_vectors(T: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate ``(n, 3)`` directions (the translation is ignored)."""
    vec = np.asarray(vectors, dtype=float)
    if vec.size == 0:
        return vec.reshape(0, 3).copy()
    return _core.coreg_transform_vectors(_mat4(T), _xyz(vec))


def yaw_transform(yaw: float, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """A rotation of ``yaw`` radians about +z followed by a translation."""
    return _core.coreg_yaw_transform(float(yaw), float(tx), float(ty), float(tz))


def kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares rigid transform mapping ``source`` onto ``target``.

    Parameters
    ----------
    source, target
        ``(n, 3)`` corresponding points, ``n >= 3``.
    weights
        Optional ``(n,)`` weights.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` proper rigid transform (reflections suppressed).

    Raises
    ------
    ValueError
        On mismatched shapes, fewer than 3 points or zero total weight.
    """
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source and target must both be (N, 3) arrays of equal length")
    if src.shape[0] < 3:
        raise ValueError("at least 3 correspondences are required")
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    return _core.coreg_kabsch(src, dst, w)


def kabsch_2d_yaw(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Least-squares yaw and translation (4 degrees of freedom) between point sets.

    The right estimator for levelled scans, whose roll and pitch are known.
    """
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 2:
        raise ValueError("source and target must be (N, 3) arrays of equal length, N >= 2")
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    return _core.coreg_kabsch_2d_yaw(src, dst, w)


def rotation_angle(T: np.ndarray) -> float:
    """Rotation magnitude of a transform, in radians (exact near zero)."""
    return float(
        _core.coreg_rotation_angle(np.ascontiguousarray(np.asarray(T, dtype=float)[:3, :3]))
    )


def transform_difference(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """``(rotation_rad, translation_m)`` of the discrepancy ``A^-1 B``."""
    return _core.coreg_transform_difference(_mat4(A), _mat4(B))
