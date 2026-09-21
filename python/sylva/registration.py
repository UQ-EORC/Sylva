"""Rigid registration of scan positions."""

from __future__ import annotations

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["kabsch", "icp", "merge_scans", "rotation_z", "translation", "estimate_normals"]


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid 4x4 transform mapping paired ``source`` onto ``target``."""
    return _core.kabsch(np.ascontiguousarray(source, dtype=float),
                        np.ascontiguousarray(target, dtype=float))


def icp(source: PointCloud, target: PointCloud, init: np.ndarray | None = None,
        max_correspondence_distance: float = 0.5, max_iterations: int = 50,
        tolerance: float = 1e-6, method: str = "point", trim: float = 1.0,
        normal_k: int = 12) -> tuple[np.ndarray, dict]:
    """Iterative closest point.

    ``method`` is ``"point"`` (point-to-point) or ``"plane"`` (point-to-plane,
    using ``target.attrs['nx','ny','nz']`` if present else PCA normals).
    ``trim`` keeps that fraction of closest correspondences per iteration
    (trimmed ICP for partial overlap).

    Returns ``(4x4 transform, {"rmse", "iterations", "n_correspondences"})``.
    """
    init = None if init is None else np.ascontiguousarray(init, dtype=float)
    return _core.icp(source.xyz, target.xyz, init, max_correspondence_distance, max_iterations,
                     tolerance, method, trim, normal_k)


def estimate_normals(cloud: PointCloud, k: int = 12) -> np.ndarray:
    return _core.estimate_normals(cloud.xyz, k)


def merge_scans(clouds: list[PointCloud], transforms: list[np.ndarray] | None = None,
                scan_ids: bool = True) -> PointCloud:
    """Transform each cloud and concatenate; adds a ``scan_id`` attribute."""
    out = []
    for i, c in enumerate(clouds):
        if transforms is not None:
            c = c.transform(transforms[i])
        if scan_ids:
            c = c.with_attrs(scan_id=np.full(len(c), i, dtype=np.int32))
        out.append(c)
    return PointCloud.concatenate(out)


def rotation_z(angle_deg: float) -> np.ndarray:
    """4x4 rotation about z."""
    a = np.radians(angle_deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def translation(dx: float, dy: float, dz: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m
