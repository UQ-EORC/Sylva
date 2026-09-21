"""Georeferenced 2D grid for DTMs and CHMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core


@dataclass
class Raster:
    """A north-up grid. ``data[row, col]``; row 0 is the *southern* edge.

    Use :meth:`to_geotiff` (rasterio) or :meth:`to_ascii_grid` to export in
    conventional north-up orientation.
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
        return cls(d["data"], d["xmin"], d["ymin"], d["resolution"])

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def xmax(self) -> float:
        return self.xmin + self.data.shape[1] * self.resolution

    @property
    def ymax(self) -> float:
        return self.ymin + self.data.shape[0] * self.resolution

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Meshgrid ``(X, Y)`` of cell-centre coordinates."""
        xs = self.xmin + (np.arange(self.data.shape[1]) + 0.5) * self.resolution
        ys = self.ymin + (np.arange(self.data.shape[0]) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)

    def cell_index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Integer ``(row, col)`` of the cell containing each coordinate."""
        col = np.floor((np.asarray(x) - self.xmin) / self.resolution).astype(np.int64)
        row = np.floor((np.asarray(y) - self.ymin) / self.resolution).astype(np.int64)
        return row, col

    def sample(self, x, y) -> np.ndarray:
        """Bilinear interpolation at coordinates (edges clamped, NaN propagates)."""
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        return _core.raster_sample(self.data, self.xmin, self.ymin, self.resolution, x, y)

    def fill_nearest(self) -> Raster:
        """Return a copy with NaN cells filled from the nearest valid cell."""
        return Raster(_core.raster_fill_nearest(self.data), self.xmin, self.ymin,
                      self.resolution, self.crs)

    def to_geotiff(self, path: str | Path, crs: str | None = None) -> None:
        """Write a GeoTIFF (requires ``rasterio``; install ``sylva[geotiff]``)."""
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError as exc:
            raise ImportError("to_geotiff requires rasterio: pip install sylva[geotiff]") from exc
        transform = from_origin(self.xmin, self.ymax, self.resolution, self.resolution)
        with rasterio.open(
            path, "w", driver="GTiff", height=self.data.shape[0], width=self.data.shape[1],
            count=1, dtype="float32", crs=crs or self.crs, transform=transform, nodata=np.nan,
        ) as dst:
            dst.write(np.flipud(self.data).astype(np.float32), 1)

    def to_ascii_grid(self, path: str | Path, nodata: float = -9999.0) -> None:
        """Write an ESRI ASCII grid."""
        _core.write_ascii_grid(str(path), self.data, self.xmin, self.ymin, self.resolution, nodata)

    @classmethod
    def from_ascii_grid(cls, path: str | Path) -> Raster:
        return cls._from_core(_core.read_ascii_grid(str(path)))
