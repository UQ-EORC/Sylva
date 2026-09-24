# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Rigid transforms (SO(3) and SE(3)) used throughout coregistration.

Transforms are 4x4 homogeneous matrices in column-vector convention,
``q = T @ [p, 1]``. Twists are ordered ``xi = [wx, wy, wz, tx, ty, tz]``,
rotation first, the convention of the pose-graph solver.
"""

from __future__ import annotations

import numpy as np

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

_EPS = 1e-12


def identity() -> np.ndarray:
    """The 4x4 identity transform."""
    return np.eye(4)


def skew(v: np.ndarray) -> np.ndarray:
    """The 3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * K
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    R = np.asarray(R, dtype=float)[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    if np.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > _EPS:
            axis = A[:, k] / axis[k]
        axis = axis / max(np.linalg.norm(axis), _EPS)
        if w @ axis < 0:
            axis = -axis
        return axis * theta
    return w * (theta / (2.0 * np.sin(theta)))


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / t2) * K
        + ((theta - np.sin(theta)) / (t2 * theta)) * (K @ K)
    )


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
    xi = np.asarray(xi, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[:3])
    T[:3, 3] = _left_jacobian(xi[:3]) @ xi[3:]
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Twist ``[w, t]`` of a transform; the inverse of :func:`se3_exp`."""
    T = np.asarray(T, dtype=float)
    w = so3_log(T[:3, :3])
    return np.concatenate([w, np.linalg.solve(_left_jacobian(w), T[:3, 3])])


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform."""
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a transform to ``(n, 3)`` points."""
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3).copy()
    # einsum rather than a matmul: numpy's bundled OpenBLAS has corrupted rows
    # when called from several threads at once, and pairs run in threads.
    return np.einsum("ij,kj->ik", pts, T[:3, :3]) + T[:3, 3]


def transform_vectors(T: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate ``(n, 3)`` directions (the translation is ignored)."""
    vec = np.asarray(vectors, dtype=float)
    if vec.size == 0:
        return vec.reshape(0, 3).copy()
    return np.einsum("ij,kj->ik", vec, np.asarray(T, dtype=float)[:3, :3])


def yaw_transform(yaw: float, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    """A rotation of ``yaw`` radians about +z followed by a translation."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = (tx, ty, tz)
    return T


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
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.sum() <= _EPS:
        raise ValueError("weights sum to zero")
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(axis=0)
    mu_d = (w[:, None] * dst).sum(axis=0)
    U, _, Vt = np.linalg.svd((src - mu_s).T @ (w[:, None] * (dst - mu_d)))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


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
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.sum() <= _EPS:
        raise ValueError("weights sum to zero")
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(axis=0)
    mu_d = (w[:, None] * dst).sum(axis=0)
    a, b = src - mu_s, dst - mu_d
    yaw = np.arctan2(
        (w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(),
        (w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])).sum(),
    )
    R = so3_exp(np.array([0.0, 0.0, yaw]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def rotation_angle(T: np.ndarray) -> float:
    """Rotation magnitude of a transform, in radians (exact near zero)."""
    return float(np.linalg.norm(so3_log(np.asarray(T, dtype=float)[:3, :3])))


def transform_difference(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """``(rotation_rad, translation_m)`` of the discrepancy ``A^-1 B``."""
    D = invert(A) @ np.asarray(B, dtype=float)
    return rotation_angle(D), float(np.linalg.norm(D[:3, 3]))
