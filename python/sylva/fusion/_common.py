# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Helpers shared by the fusion functions."""

from __future__ import annotations

import numpy as np

from ..pointcloud import PointCloud

GROUND = 2


def _matrix(transform) -> np.ndarray | None:
    from .registration import Registration  # register imports this module

    if transform is None:
        return None
    if isinstance(transform, Registration):
        return transform.transform
    m = np.asarray(transform, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"transform must be a finite (4, 4) matrix or a Registration, got shape {m.shape}")
    return m


def _ground(cloud: PointCloud, mask, what: str) -> np.ndarray:
    if mask is not None:
        g = np.ascontiguousarray(mask, dtype=bool).ravel()
        if len(g) != len(cloud):
            raise ValueError(f"{what}_ground has {len(g)} values for {len(cloud)} points")
        return g
    if "classification" not in cloud.attrs:
        raise ValueError(f"the {what} cloud has no 'classification' attribute; classify its ground first "
                         f"(sylva.ground.classify_ground_csf) or pass {what}_ground")
    return np.ascontiguousarray(np.asarray(cloud.attrs["classification"]) == GROUND)


def _transform_xyz(m: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ m[:3, :3].T + m[:3, 3]
