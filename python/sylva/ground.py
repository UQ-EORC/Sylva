# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ground classification, DTM generation, height normalisation and CHM."""

from __future__ import annotations

import numpy as np

from . import _core, interpolate
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
    """Classify ground with the Cloth Simulation Filter (Zhang et al. 2016).

    The cloud is turned upside down and a cloth is dropped onto it; points
    within ``class_threshold`` of the settled cloth are ground. The default
    for TLS plots; robust to understorey and moderate slopes.

    Parameters
    ----------
    cloud
        Points in a frame with z up (project or projected coordinates, not
        a tilted scanner frame). Thin to ~2-5 cm first for large plots.
    cloth_resolution
        Cloth grid spacing (m). Smaller follows the terrain more closely
        but can drape over logs and low shrubs.
    rigidness : {1, 2, 3}
        1 for steep terrain, 2 for moderate slopes, 3 for flat ground.
    class_threshold
        Distance from the cloth (m) within which a point is ground.
    iterations
        Maximum simulation steps.
    time_step
        Simulation step; the default rarely needs changing.
    return_mask
        Return the boolean ground mask instead of the cloud.

    Returns
    -------
    PointCloud or numpy.ndarray
        The cloud with ``classification`` set to 2 (ground) or 1 (other),
        overwriting any existing classification; or the mask (True = ground).

    See Also
    --------
    classify_ground_pmf : faster alternative.
    make_dtm : the next step.
    """
    mask = _core.csf_ground_mask(cloud.xyz, cloth_resolution, rigidness, class_threshold,
                                 iterations, time_step)
    return mask if return_mask else _with_class(cloud, mask)


def classify_ground_pmf(cloud: PointCloud, cell_size: float = 0.5, max_window: float = 10.0,
                        slope: float = 0.3, initial_distance: float = 0.15,
                        max_distance: float = 2.0, return_mask: bool = False):
    """Classify ground with the Progressive Morphological Filter (Zhang et al. 2003).

    Opens a minimum-height grid with growing windows; points close to the
    opened surface are ground. Faster than CSF but less robust under dense
    understorey.

    Parameters
    ----------
    cloud
        Points in a frame with z up.
    cell_size
        Grid cell size (m).
    max_window
        Largest window (m); should exceed the largest non-ground object
        (e.g. crown or building) footprint.
    slope
        Terrain slope (rise over run) used to grow the height threshold
        with window size.
    initial_distance
        Height threshold (m) for the smallest window.
    max_distance
        Upper limit on the height threshold (m).
    return_mask
        Return the boolean ground mask instead of the cloud.

    Returns
    -------
    PointCloud or numpy.ndarray
        As for :func:`classify_ground_csf`.

    Raises
    ------
    ValueError
        For non-positive sizes.
    """
    mask = _core.pmf_ground_mask(cloud.xyz, cell_size, max_window, slope, initial_distance,
                                 max_distance)
    return mask if return_mask else _with_class(cloud, mask)


def _with_class(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    return cloud.with_attrs(classification=np.where(mask, GROUND, 1).astype(np.uint8))


def ground_mask(cloud: PointCloud) -> np.ndarray:
    """Which points are ground.

    Parameters
    ----------
    cloud
        A cloud with a ``classification`` attribute (from a classifier here
        or from a LAS file).

    Returns
    -------
    numpy.ndarray
        True where ``classification == 2`` (ASPRS ground).

    Raises
    ------
    ValueError
        If the cloud has no ``classification``.
    """
    if "classification" not in cloud.attrs:
        raise ValueError("cloud has no 'classification' attribute; run classify_ground_* first")
    return np.asarray(cloud.attrs["classification"]) == GROUND


def make_dtm(cloud: PointCloud, resolution: float = 0.5, bounds=None,
             method: str = "lowest") -> Raster:
    """Digital terrain model from the ground points.

    By default each cell takes its lowest ground point. Empty cells (under
    stems, behind occlusion) are filled from the nearest measured cells and
    smoothed; measured cells keep their value.

    Parameters
    ----------
    cloud
        A classified cloud (see :func:`ground_mask`).
    resolution
        Cell size (m).
    bounds
        ``(xmin, ymin, xmax, ymax)`` of the grid; the extent of the ground
        points if None. Pass the plot extent to get identical grids across
        dates.
    method : {"lowest", "tin", "natural", "idw"}
        ``"lowest"`` (the default) is the per-cell minimum described above.
        The others interpolate the ground points at cell centres with
        :func:`sylva.interpolate.grid` (a triangulation, natural neighbours
        or inverse distance weighting with its defaults); cells outside the
        convex hull of the ground points are then filled from the nearest
        interpolated cell. Interpolation passes through every ground point,
        so thin the ground to one point per cell or so first
        (e.g. ``filters.voxel_downsample``), or noise in the ground returns
        shows up as roughness in the surface.

    Returns
    -------
    Raster
        Ground elevation, no NaN cells.

    Raises
    ------
    ValueError
        With fewer than 3 ground points, no ``classification`` or an unknown
        method.
    """
    if method not in ("lowest", "tin", "natural", "idw"):
        raise ValueError(f"unknown method {method!r}; expected 'lowest', 'tin', 'natural' or 'idw'")
    ground = cloud[ground_mask(cloud)]
    if method == "lowest":
        return Raster._from_core(_core.make_dtm(ground.xyz, resolution, bounds))
    if len(ground) < 3:
        raise ValueError("need at least 3 ground points")
    dtm = interpolate.grid(ground, resolution, method=method, bounds=bounds)
    return dtm.fill_nearest() if np.isnan(dtm.data).any() else dtm


def normalize_height(cloud: PointCloud, dtm: Raster, attr: str = "height") -> PointCloud:
    """Add height above ground as an attribute.

    Parameters
    ----------
    cloud
        Points in the DTM's frame.
    dtm
        Terrain from :func:`make_dtm`.
    attr
        Name of the new attribute.

    Returns
    -------
    PointCloud
        The cloud with ``attr`` = z minus the interpolated DTM; coordinates
        are unchanged. Points outside the DTM take the edge value.
    """
    return cloud.with_attrs(**{attr: _heights_above(cloud, dtm)})


def flatten(cloud: PointCloud, dtm: Raster) -> PointCloud:
    """Replace z with height above ground.

    Parameters
    ----------
    cloud
        Points in the DTM's frame.
    dtm
        Terrain from :func:`make_dtm`.

    Returns
    -------
    PointCloud
        x, y unchanged and z = height above the DTM. Use for methods that
        read z as height; keep :func:`normalize_height` when true
        elevations must be preserved.
    """
    h = _heights_above(cloud, dtm)
    return PointCloud(np.column_stack([cloud.x, cloud.y, h]), dict(cloud.attrs))


def _heights_above(cloud: PointCloud, dtm: Raster) -> np.ndarray:
    return _core.raster_heights_above(cloud.xyz, dtm.data, float(dtm.xmin), float(dtm.ymin),
                                      float(dtm.resolution))


def make_chm(cloud: PointCloud, resolution: float = 0.5, height_attr: str = "height",
             bounds=None, min_height: float = 0.0) -> Raster:
    """Canopy height model: the highest point in each cell.

    Parameters
    ----------
    cloud
        Points with heights (see :meth:`PointCloud.heights`).
    resolution
        Cell size (m).
    height_attr
        Attribute holding height above ground; z is used if absent.
    bounds
        ``(xmin, ymin, xmax, ymax)``; the cloud extent if None.
    min_height
        Cells with nothing above this height are 0.

    Returns
    -------
    Raster
        Maximum height per cell (m). No pit filling is applied, and TLS
        under-samples the upper canopy far from the scanners.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return Raster._from_core(_core.make_chm(cloud.xyz, h, resolution, bounds, min_height))
