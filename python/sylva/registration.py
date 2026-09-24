# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Rigid registration of scan positions."""

from __future__ import annotations

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["kabsch", "icp", "merge_scans", "rotation_z", "translation", "estimate_normals"]


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best rigid transform between paired points (Kabsch 1976; Umeyama 1991, no scale).

    Use with matched targets (reflectors, tie points) to register scans.

    Parameters
    ----------
    source, target
        ``(N, 3)`` corresponding points, N >= 3 and not collinear.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix mapping ``source`` onto ``target`` in the least
        squares sense.
    """
    return _core.kabsch(np.ascontiguousarray(source, dtype=float),
                        np.ascontiguousarray(target, dtype=float))


def icp(source: PointCloud, target: PointCloud, init: np.ndarray | None = None,
        max_correspondence_distance: float = 0.5, max_iterations: int = 50,
        tolerance: float = 1e-6, method: str = "point", trim: float = 1.0,
        normal_k: int = 12) -> tuple[np.ndarray, dict]:
    """Iterative closest point.

    ``method`` is ``"point"`` (point-to-point, Besl & McKay 1992) or
    ``"plane"`` (point-to-plane, Chen & Medioni 1992, linearised as in Low
    2004; normals from ``target.attrs['nx','ny','nz']`` if present, else
    PCA). ``trim`` keeps that fraction of closest correspondences per
    iteration (trimmed ICP for partial overlap, Chetverikov et al. 2002).

    Parameters
    ----------
    source
        Cloud to move.
    target
        Fixed reference cloud.
    init
        Starting ``(4, 4)`` transform (e.g. from the SOP or :func:`kabsch`);
        ICP only converges from a start within about
        ``max_correspondence_distance``.
    max_correspondence_distance
        Pairs further apart (m) are ignored.
    max_iterations
        Iteration limit.
    tolerance
        Stop when the RMSE changes less than this.
    method : {"point", "plane"}
        Error metric; point-to-plane converges faster on stems and ground.
    trim
        Fraction of pairs kept (0-1).
    normal_k
        Neighbours for PCA normals with ``"plane"``.

    Returns
    -------
    transform : numpy.ndarray
        ``(4, 4)`` matrix mapping ``source`` onto ``target``.
    info : dict
        ``rmse`` (m), ``iterations`` and ``n_correspondences``.

    Notes
    -----
    Thin both clouds to 2-5 cm first; ICP on full-density TLS is slow and
    no more accurate. Check the result with :func:`sylva.quality.stem_noise`.
    """
    init = None if init is None else np.ascontiguousarray(init, dtype=float)
    return _core.icp(source.xyz, target.xyz, init, max_correspondence_distance, max_iterations,
                     tolerance, method, trim, normal_k)


def estimate_normals(cloud: PointCloud, k: int = 12) -> np.ndarray:
    """PCA normals; the same as :func:`sylva.filters.estimate_normals`.

    Parameters
    ----------
    cloud
        Input points.
    k
        Neighbours per point.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` unit normals with arbitrary sign.
    """
    return _core.estimate_normals(cloud.xyz, k)


def merge_scans(clouds: list[PointCloud], transforms: list[np.ndarray] | None = None,
                scan_ids: bool = True) -> PointCloud:
    """Put several scans in one frame and merge them.

    Parameters
    ----------
    clouds
        One cloud per scan position.
    transforms
        One ``(4, 4)`` matrix per cloud (SOPs or ICP results); None if the
        clouds are already registered.
    scan_ids
        Add a ``scan_id`` attribute (int32, the index in ``clouds``), which
        :func:`sylva.quality.stem_noise` can read.

    Returns
    -------
    PointCloud
        Only attributes present in every cloud are kept.
    """
    out = []
    for i, c in enumerate(clouds):
        if transforms is not None:
            c = c.transform(transforms[i])
        if scan_ids:
            c = c.with_attrs(scan_id=np.full(len(c), i, dtype=np.int32))
        out.append(c)
    return PointCloud.concatenate(out)


def rotation_z(angle_deg: float) -> np.ndarray:
    """Rotation about the z axis.

    Parameters
    ----------
    angle_deg
        Angle in degrees, counter-clockwise seen from above.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix.
    """
    a = np.radians(angle_deg)
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def translation(dx: float, dy: float, dz: float) -> np.ndarray:
    """Translation matrix.

    Parameters
    ----------
    dx, dy, dz
        Shift (m).

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix.
    """
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m
