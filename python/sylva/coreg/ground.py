# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The terrain model coregistration measures heights from.

Stem detection works on height above ground, not raw z: on sloping terrain a
fixed elevation band cuts through the canopy on one side of a plot and the
litter on the other. This is tlsalign's raster DTM: a low percentile of z per
cell, deep pits rejected, gaps filled from the nearest observed cell, a grey
opening and smoothing, then a slope limit so a shrub or a log cannot lift the
surface. It is deliberately not :mod:`sylva.ground`'s CSF, so that the stems
and heights registration sees are tlsalign's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _core

__all__ = ["GroundModel", "fit_ground"]


@dataclass
class GroundModel:
    """A raster terrain model with bilinear interpolation.

    Attributes
    ----------
    elevation
        ``(ny, nx)`` terrain height, gap-free.
    origin
        ``(x0, y0)``, taken as the centre of cell ``[0, 0]`` when sampling
        (tlsalign's convention, a quarter-cell off its binning; kept so that
        heights match).
    cell_size
        Grid spacing (m).
    observed
        Cells backed by ground returns rather than filled.
    """

    elevation: np.ndarray
    origin: np.ndarray
    cell_size: float
    observed: np.ndarray

    def __post_init__(self) -> None:
        self.elevation = np.ascontiguousarray(self.elevation, dtype=np.float64)
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(2)
        self.observed = np.ascontiguousarray(self.observed, dtype=bool)

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Terrain height at ``(n, 2)`` locations (clamped at the edges)."""
        xy = np.ascontiguousarray(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return _core.coreg_ground_height(
            self.elevation, float(self.origin[0]), float(self.origin[1]), float(self.cell_size), xy
        )

    def support(self, xy: np.ndarray) -> np.ndarray:
        """Were ``(n, 2)`` locations backed by ground returns?"""
        xy = np.ascontiguousarray(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return _core.coreg_ground_support(
            self.observed, float(self.origin[0]), float(self.origin[1]), float(self.cell_size), xy
        )

    def normalise(self, points: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Heights above ground of ``(n, 3)`` points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, dtype=dtype)
        return (points[:, 2] - self.height_at(points[:, :2])).astype(dtype, copy=False)

    @property
    def slope_deg(self) -> float:
        """Mean terrain slope (degrees), a sanity check on the fit."""
        gy, gx = np.gradient(self.elevation, self.cell_size)
        return float(np.degrees(np.arctan(np.hypot(gx, gy))).mean())


def fit_ground(
    points: np.ndarray,
    cell_size: float = 0.5,
    *,
    percentile: float = 5.0,
    max_slope: float = 1.0,
    smooth_cells: int = 3,
    opening_cells: int = 5,
    min_points_per_cell: int = 1,
    pit_depth: float = 3.0,
    pit_window: int = 9,
    max_points: int | None = 8_000_000,
    seed: int = 0,
) -> GroundModel:
    """Fit a raster terrain model to a point cloud.

    Parameters
    ----------
    points
        ``(n, 3)`` points, z up.
    cell_size
        Resolution (m).
    percentile
        Per-cell z percentile taken as ground; robust to the few below-ground
        multipath returns.
    max_slope
        Largest rise over run between neighbouring cells.
    smooth_cells, opening_cells
        Windows (cells) of the smoothing and of the grey opening that removes
        low objects such as logs.
    min_points_per_cell
        Points a cell needs to count as observed.
    pit_depth, pit_window
        A cell more than ``pit_depth`` m below the median of the
        ``pit_window`` cells around it is a pit, not ground. Tilted VZ-400i
        scans carry echoes 20-30 m under the terrain, and one such cell
        otherwise drags the surface down through the slope limit.
    max_points
        Fit from at most this many points, drawn at random (None: all).
    seed
        Seed of that draw.

    Returns
    -------
    GroundModel

    Raises
    ------
    ValueError
        On an empty cloud, a non-positive cell size or no observed cell.
    """
    xyz = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    elevation, origin, observed = _core.coreg_fit_ground(
        xyz,
        float(cell_size),
        float(percentile),
        float(max_slope),
        int(smooth_cells),
        int(opening_cells),
        int(min_points_per_cell),
        float(pit_depth),
        int(pit_window),
        int(max_points or 0),
        int(seed),
    )
    return GroundModel(elevation, np.asarray(origin), float(cell_size), observed)
