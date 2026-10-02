# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Georeferenced 2D grid for DTMs and CHMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core


@dataclass
class Raster:
    """A regular 2D grid in map coordinates, used for DTMs and CHMs.

    ``data[row, col]`` covers x from ``xmin + col * resolution`` and y from
    ``ymin + row * resolution``, so row 0 is the *southern* edge (the array is
    upside down compared with an image). :meth:`to_geotiff` and
    :meth:`to_ascii_grid` flip it to the usual north-up order on export.

    Parameters
    ----------
    data
        ``(rows, cols)`` values; NaN marks cells with no data.
    xmin, ymin
        Coordinates of the south-west corner of the grid (not of the first
        cell centre).
    resolution
        Cell size in the units of x and y (m).
    crs
        Optional CRS string (``"EPSG:28355"``) written by :meth:`to_geotiff`
        and, as a ``.prj``, by :meth:`to_ascii_grid`. Sylva never reprojects.
    """

    data: np.ndarray
    xmin: float
    ymin: float
    resolution: float
    crs: str | None = None

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(np.asarray(self.data, dtype=np.float64))

    @classmethod
    def _from_core(cls, d: dict) -> Raster:
        return cls(d["data"], d["xmin"], d["ymin"], d["resolution"], d.get("crs"))

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of ``data``."""
        return self.data.shape

    @property
    def xmax(self) -> float:
        """Eastern edge of the grid."""
        return self.xmin + self.data.shape[1] * self.resolution

    @property
    def ymax(self) -> float:
        """Northern edge of the grid."""
        return self.ymin + self.data.shape[0] * self.resolution

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates.

        Returns
        -------
        X, Y : numpy.ndarray
            Two ``(rows, cols)`` arrays, aligned with ``data``.
        """
        rows, cols = self.data.shape
        return _core.raster_cell_centers(rows, cols, float(self.xmin), float(self.ymin),
                                         float(self.resolution))

    def cell_index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Cell containing each coordinate.

        Parameters
        ----------
        x, y
            Coordinates (scalars or arrays of the same shape).

        Returns
        -------
        row, col : numpy.ndarray
            int64 indices. They are *not* clipped: points outside the grid get
            negative or too-large indices, so check against :attr:`shape`.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        row, col = _core.raster_cell_index(float(self.xmin), float(self.ymin),
                                           float(self.resolution), np.ascontiguousarray(x.ravel()),
                                           np.ascontiguousarray(y.ravel()))
        return row.reshape(y.shape), col.reshape(x.shape)

    def sample(self, x, y) -> np.ndarray:
        """Interpolate the grid at arbitrary coordinates.

        Parameters
        ----------
        x, y
            Coordinates; flattened to 1D.

        Returns
        -------
        numpy.ndarray
            Bilinear interpolation between cell centres, one value per
            coordinate. Coordinates beyond the outer cell centres take the
            edge value; a NaN neighbour makes the result NaN, so call
            :meth:`fill_nearest` first if the grid has holes.
        """
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        return _core.raster_sample(self.data, self.xmin, self.ymin, self.resolution, x, y)

    def fill_nearest(self) -> Raster:
        """Fill NaN cells from the nearest valid cell.

        Returns
        -------
        Raster
            A copy with every NaN replaced; the input is unchanged.
        """
        return Raster(_core.raster_fill_nearest(self.data), self.xmin, self.ymin,
                      self.resolution, self.crs)

    def to_geotiff(self, path: str | Path, crs: str | None = None) -> None:
        """Write a single-band float32 GeoTIFF, north-up, nodata = NaN.

        Written by Sylva itself (uncompressed, in strips; BigTIFF from
        4 GB), so no GDAL or rasterio is needed.

        Parameters
        ----------
        path
            Output file.
        crs
            CRS to write (an EPSG code such as ``"EPSG:28356"``, or WKT);
            defaults to :attr:`crs`. With neither, the file has a
            geotransform but no CRS. A CRS with an EPSG code goes in the
            file's GeoKeys; one without (custom WKT) goes in a GDAL
            ``<path>.aux.xml`` beside it, which GDAL-based software reads.

        Raises
        ------
        ValueError
            For an empty raster or a CRS that cannot be read.
        """
        c = crs or self.crs
        _core.write_geotiff(str(path), self.data, self.xmin, self.ymin, self.resolution,
                            None if c is None or c.startswith("user-defined") else c)

    def to_ascii_grid(self, path: str | Path, nodata: float = -9999.0,
                      crs: str | None = None) -> None:
        """Write an ESRI ASCII grid (``.asc``), north-up, with its CRS in a
        ``.prj`` beside it.

        Parameters
        ----------
        path
            Output file.
        nodata
            Value written for NaN cells (``NODATA_value`` in the header).
        crs
            CRS to write, as WKT, to the ``.prj`` beside it (``dtm.asc`` ->
            ``dtm.prj``); defaults to :attr:`crs`. With neither, no
            ``.prj`` is written.
        """
        c = crs or self.crs
        _core.write_ascii_grid(str(path), self.data, self.xmin, self.ymin, self.resolution, nodata,
                               None if c is None or c.startswith("user-defined") else c)

    @classmethod
    def from_ascii_grid(cls, path: str | Path) -> Raster:
        """Read an ESRI ASCII grid written by :meth:`to_ascii_grid` or GIS software.

        Parameters
        ----------
        path
            ``.asc`` file. ``NODATA_value`` cells become NaN.

        Returns
        -------
        Raster
            Grid with row 0 at the south; ``crs`` from the ``.prj`` beside
            it (``EPSG:<code>`` when its WKT names one), else None.
        """
        return cls._from_core(_core.read_ascii_grid(str(path)))
