"""Ground classification, DTM generation, height normalisation and CHM."""

from __future__ import annotations

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .raster import Raster

__all__ = [
    "classify_ground_csf", "classify_ground_pmf", "ground_mask", "make_dtm",
    "normalize_height", "flatten", "make_chm",
]

GROUND = 2


def classify_ground_csf(cloud: PointCloud, cloth_resolution: float = 0.5, rigidness: int = 2,
                        class_threshold: float = 0.3, iterations: int = 500,
                        time_step: float = 0.65, return_mask: bool = False):
    """Cloth Simulation Filter (Zhang et al. 2016).

    A cloth is dropped onto the inverted surface; points within
    ``class_threshold`` of the settled cloth are ground. ``rigidness`` runs 1
    (steep) to 3 (flat). Returns the cloud with a ``classification``
    attribute (2 = ground, 1 = other), or the boolean mask.
    """
    mask = _core.csf_ground_mask(cloud.xyz, cloth_resolution, rigidness, class_threshold,
                                 iterations, time_step)
    return mask if return_mask else _with_class(cloud, mask)


def classify_ground_pmf(cloud: PointCloud, cell_size: float = 0.5, max_window: float = 10.0,
                        slope: float = 0.3, initial_distance: float = 0.15,
                        max_distance: float = 2.0, return_mask: bool = False):
    """Progressive Morphological Filter (Zhang et al. 2003). Faster than CSF
    but less robust under dense understorey."""
    mask = _core.pmf_ground_mask(cloud.xyz, cell_size, max_window, slope, initial_distance,
                                 max_distance)
    return mask if return_mask else _with_class(cloud, mask)


def _with_class(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    return cloud.with_attrs(classification=np.where(mask, GROUND, 1).astype(np.uint8))


def ground_mask(cloud: PointCloud) -> np.ndarray:
    """Boolean ground mask from the ``classification`` attribute (ASPRS class 2)."""
    if "classification" not in cloud.attrs:
        raise ValueError("cloud has no 'classification' attribute; run classify_ground_* first")
    return np.asarray(cloud.attrs["classification"]) == GROUND


def make_dtm(cloud: PointCloud, resolution: float = 0.5, bounds=None) -> Raster:
    """DTM from ground points: lowest point per cell, gaps filled from the
    nearest cells. ``bounds`` is ``(xmin, ymin, xmax, ymax)``."""
    ground = cloud[ground_mask(cloud)]
    return Raster._from_core(_core.make_dtm(ground.xyz, resolution, bounds))


def normalize_height(cloud: PointCloud, dtm: Raster, attr: str = "height") -> PointCloud:
    """Add height above the DTM as an attribute (coordinates unchanged)."""
    return cloud.with_attrs(**{attr: cloud.z - dtm.sample(cloud.x, cloud.y)})


def flatten(cloud: PointCloud, dtm: Raster) -> PointCloud:
    """Return a cloud whose z is height above the DTM."""
    h = cloud.z - dtm.sample(cloud.x, cloud.y)
    return PointCloud(np.column_stack([cloud.x, cloud.y, h]), dict(cloud.attrs))


def make_chm(cloud: PointCloud, resolution: float = 0.5, height_attr: str = "height",
             bounds=None, min_height: float = 0.0) -> Raster:
    """Canopy height model: maximum height per cell (0 where empty)."""
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return Raster._from_core(_core.make_chm(cloud.xyz, h, resolution, bounds, min_height))
