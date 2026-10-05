# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Catalogues of airborne lidar tiles: the tile, chunk and catalogue types."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster


@dataclass(frozen=True)
class Tile:
    """One LAS/LAZ file of a :class:`Catalog`, as its header describes it.

    Parameters
    ----------
    path
        The file.
    bounds
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` from the header.
    n_points
        Point count from the header.
    point_format
        LAS point data record format (0-10).
    version
        LAS version as ``(major, minor)``.
    crs
        ``"EPSG:<code>"`` when the CRS records name an EPSG code, the WKT
        when they do not, None without CRS records.
    scale, offset
        Coordinate quantisation and offset per axis.
    spatial_index
        True with a LAStools ``.lax`` index beside the file, or for a COPC
        file. Recorded for information; Sylva reads every file in full.
    file_size
        Bytes on disk.
    """

    path: str
    bounds: tuple[float, float, float, float, float, float]
    n_points: int
    point_format: int
    version: tuple[int, int]
    crs: str | None
    scale: tuple[float, float, float]
    offset: tuple[float, float, float]
    spatial_index: bool
    file_size: int

    @property
    def xy_bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the tile."""
        b = self.bounds
        return (b[0], b[1], b[3], b[4])

    @property
    def area(self) -> float:
        """Area of the tile's extent (m²)."""
        b = self.bounds
        return max(b[3] - b[0], 0.0) * max(b[4] - b[1], 0.0)


@dataclass(frozen=True)
class Chunk:
    """A piece of a catalogue processed at once.

    Parameters
    ----------
    index
        Position in the list of chunks; results come back in this order.
    core
        ``(xmin, ymin, xmax, ymax)`` of the area the chunk is responsible
        for. On a grid a point belongs to the chunk whose half-open square
        ``[xmin, xmax) x [ymin, ymax)`` holds it.
    outer
        The core grown by the buffer: every point inside is read.
    own
        For one chunk per tile, the index of that tile in the catalogue:
        its points are the core, every other file's are buffer. None on a
        grid.
    files
        Indices of the tiles overlapping ``outer``.
    est_points
        Points expected in ``outer`` (from the header counts, assuming each
        tile is evenly covered), used to size the work to memory.
    name
        A file name for the chunk's outputs: the tile's own name, or
        ``<xmin>_<ymin>`` of a grid chunk.
    """

    index: int
    core: tuple[float, float, float, float]
    outer: tuple[float, float, float, float]
    own: int | None
    files: tuple[int, ...]
    est_points: int
    name: str

    @property
    def buffer(self) -> float:
        """Width of the buffer (m)."""
        return float(self.core[0] - self.outer[0])

    def _core(self) -> dict:
        return {"index": self.index, "core": self.core, "outer": self.outer, "own": self.own,
                "files": list(self.files), "est_points": self.est_points, "name": self.name}

    @classmethod
    def _from_core(cls, d: dict) -> Chunk:
        return cls(int(d["index"]), tuple(d["core"]), tuple(d["outer"]), d["own"],
                   tuple(d["files"]), int(d["est_points"]), d["name"])


@dataclass
class Catalog:
    """Tiles of an airborne survey, known from their headers alone.

    Build one with :func:`catalog`; the processing functions of this module
    take it and those that write tiles return the catalogue of what they
    wrote, so that steps chain. It is plain data and pickles.

    Parameters
    ----------
    tiles
        The readable files, in the order given.
    missing
        Paths asked for that do not exist.
    unreadable
        ``(path, reason)`` for files whose header could not be read.
    tolerance
        How far (m) tile extents may overlap, or fall short of each other,
        before :meth:`issues` reports an overlap or a hole.

    Notes
    -----
    A catalogue with missing or unreadable files can be inspected but not
    processed: every processing function raises ``ValueError`` for it.
    """

    tiles: list[Tile]
    missing: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    tolerance: float = 1.0

    def __len__(self) -> int:
        return len(self.tiles)

    def __repr__(self) -> str:
        extra = ""
        if self.missing or self.unreadable:
            extra = f", {len(self.missing)} missing, {len(self.unreadable)} unreadable"
        return f"Catalog({len(self.tiles)} tiles, {self.n_points:,} points{extra})"

    @property
    def paths(self) -> list[str]:
        """Paths of the tiles."""
        return [t.path for t in self.tiles]

    @property
    def n_points(self) -> int:
        """Points in all tiles, from the headers."""
        return int(sum(t.n_points for t in self.tiles))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """``(xmin, ymin, zmin, xmax, ymax, zmax)`` over all tiles; None if empty."""
        if not self.tiles:
            return None
        b = np.array([t.bounds for t in self.tiles])
        return (*b[:, :3].min(axis=0).tolist(), *b[:, 3:].max(axis=0).tolist())

    @property
    def crs(self) -> str | None:
        """The CRS every tile declares, or None if they differ or declare none."""
        crs = {t.crs for t in self.tiles}
        return crs.pop() if len(crs) == 1 else None

    def _core(self) -> dict:
        t = self.tiles
        return {
            "paths": [x.path for x in t], "bounds": [x.bounds for x in t],
            "n_points": [x.n_points for x in t], "point_format": [x.point_format for x in t],
            "version": [x.version for x in t], "crs": [x.crs for x in t],
            "scale": [x.scale for x in t], "offset": [x.offset for x in t],
            "spatial_index": [x.spatial_index for x in t], "file_size": [x.file_size for x in t],
            "missing": list(self.missing), "unreadable": list(self.unreadable),
        }

    @classmethod
    def _from_core(cls, d: dict, tolerance: float) -> Catalog:
        tiles = [Tile(p, tuple(b), int(n), int(f), tuple(v), c, tuple(s), tuple(o), bool(i), int(z))
                 for p, b, n, f, v, c, s, o, i, z in zip(
                     d["paths"], d["bounds"], d["n_points"], d["point_format"], d["version"],
                     d["crs"], d["scale"], d["offset"], d["spatial_index"], d["file_size"],
                     strict=True)]
        return cls(tiles, list(d["missing"]), [tuple(u) for u in d["unreadable"]], tolerance)

    def issues(self) -> list[tuple[str, str]]:
        """Everything that could make processing fail or mislead.

        Returns
        -------
        list of (str, str)
            ``(kind, message)`` pairs; empty for a clean tiling. Kinds:
            ``missing`` and ``unreadable`` files, ``empty`` tiles,
            ``mixed_crs`` (tiles in different coordinate systems, or some
            with none), ``no_crs``, ``mixed_point_format`` (attributes not
            in every format are dropped where tiles meet), ``mixed_scale``,
            ``overlap`` (tile extents overlapping by more than
            :attr:`tolerance`; their points are counted twice) and ``gap``
            (holes enclosed by tiles).
        """
        return [tuple(i) for i in _core.als_issues(self._core(), float(self.tolerance))]

    def validate(self) -> None:
        """Raise if :meth:`issues` finds anything.

        Raises
        ------
        ValueError
            Listing every issue.
        """
        issues = self.issues()
        if issues:
            raise ValueError("the catalogue has problems:\n" + "\n".join(f"  [{k}] {m}" for k, m in issues))

    def report(self) -> str:
        """A plain-text summary: extent, points, density, formats, CRS and issues.

        Returns
        -------
        str
            Several lines, ready to print.
        """
        return _core.als_report(self._core(), float(self.tolerance))

    def summary(self) -> dict:
        """The numbers behind :meth:`report`.

        Returns
        -------
        dict
            ``n_tiles``, ``n_points``, ``bounds``, ``area`` (m² of tile
            extents), ``density`` (points/m² over the tile extents, an
            underestimate for flight lines), ``crs``, ``point_formats``
            (format to tile count), ``indexed`` (tiles with a spatial index)
            and ``issues``.
        """
        area = float(sum(t.area for t in self.tiles))
        formats: dict[int, int] = {}
        for t in self.tiles:
            formats[t.point_format] = formats.get(t.point_format, 0) + 1
        return {
            "n_tiles": len(self.tiles), "n_points": self.n_points, "bounds": self.bounds,
            "area": area, "density": self.n_points / area if area > 0 else float("nan"),
            "crs": self.crs, "point_formats": formats,
            "indexed": sum(t.spatial_index for t in self.tiles), "issues": self.issues(),
        }

    def overlaps(self) -> list[tuple[int, int, float]]:
        """Pairs of overlapping tiles.

        Returns
        -------
        list of (int, int, float)
            Tile indices ``i < j`` and the area of the overlap of their
            extents (m²), for extents overlapping by more than
            :attr:`tolerance` in both x and y.
        """
        return [tuple(o) for o in _core.als_overlaps(self._core(), float(self.tolerance))]

    def gaps(self) -> list[tuple[tuple[float, float, float, float], float]]:
        """Holes in the coverage.

        Returns
        -------
        list of (tuple, float)
            The bounding box ``(xmin, ymin, xmax, ymax)`` and area (m²) of
            each area enclosed by tiles but inside none (sampled on a grid of
            at most 1000 x 1000 cells). A concave outline is not a hole.
        """
        return [(tuple(b), a) for b, a in _core.als_gaps(self._core(), float(self.tolerance))]

    def chunks(self, chunk_size: float | None = None, buffer: float = 20.0,
               origin=None) -> list[Chunk]:
        """Divide the catalogue into buffered chunks.

        Parameters
        ----------
        chunk_size
            Side of square chunks (m) on a regular grid; None for one chunk
            per tile.
        buffer
            Width (m) of the band of neighbouring points read around each
            chunk.
        origin
            ``(x, y)`` of a grid corner; by default the catalogue's minimum
            snapped down to a multiple of ``chunk_size``.

        Returns
        -------
        list of Chunk
            West to east within south-to-north rows on a grid; tile order
            otherwise. Grid squares no tile reaches are left out.

        Raises
        ------
        ValueError
            For a non-positive ``chunk_size``, a negative ``buffer``, or a
            catalogue with missing or unreadable files.
        """
        o = None if origin is None else (float(origin[0]), float(origin[1]))
        cs = None if chunk_size is None else float(chunk_size)
        return [Chunk._from_core(d) for d in _core.als_plan(self._core(), cs, float(buffer), o)]

    def read(self, bounds=None) -> PointCloud:
        """Read the points inside a box, from whichever tiles hold them.

        Parameters
        ----------
        bounds
            ``(xmin, ymin, xmax, ymax)``, inclusive; the whole catalogue if
            None (only sensible when it fits in memory).

        Returns
        -------
        PointCloud
            Points in tile order with the LAS attributes that every tile
            involved has.
        """
        if bounds is None:
            b = self.bounds
            if b is None:
                raise ValueError("the catalogue has no tiles")
            bounds = (b[0], b[1], b[3], b[4])
        bounds = tuple(float(v) for v in bounds)
        if len(bounds) != 4 or bounds[2] < bounds[0] or bounds[3] < bounds[1]:
            raise ValueError(f"bounds must be (xmin, ymin, xmax, ymax), got {bounds}")
        xyz, attrs = _core.als_read_region(self._core(), bounds)
        return PointCloud(xyz, attrs)

    def _raster(self, d: dict) -> Raster:
        r = Raster._from_core(d)
        crs = self.crs
        if crs is not None and not crs.startswith("user-defined"):
            r.crs = crs
        return r


def _files(paths, pattern: str, recursive: bool) -> list[str]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            walk = p.rglob("*") if recursive else p.iterdir()
            out.extend(str(f) for f in sorted(walk)
                       if f.is_file() and fnmatch.fnmatch(f.name.lower(), pattern.lower()))
        else:
            out.append(str(p))
    return out


def catalog(paths, pattern: str = "*.la[sz]", recursive: bool = False,
            tolerance: float = 1.0) -> Catalog:
    """Build a catalogue from LAS/LAZ files, reading only their headers.

    Parameters
    ----------
    paths
        A directory, a file, or a list of either. Directories are searched
        for ``pattern``.
    pattern
        File name pattern (shell style, case-insensitive) within directories.
    recursive
        Search directories' subdirectories too.
    tolerance
        See :attr:`Catalog.tolerance`.

    Returns
    -------
    Catalog
        Tiles in sorted path order within each directory. Missing and
        unreadable files are recorded rather than raised; see
        :meth:`Catalog.issues`.

    Raises
    ------
    ValueError
        If nothing matches.
    """
    files = _files(paths, pattern, recursive)
    if not files:
        raise ValueError(f"no files matching {pattern!r} in {paths!r}")
    if not (np.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(f"tolerance must be zero or more, got {tolerance}")
    return Catalog._from_core(_core.als_catalog(files), float(tolerance))


def _as_catalog(cat) -> Catalog:
    if isinstance(cat, Catalog):
        return cat
    return catalog(cat)


def _workers(workers: int | None) -> int:
    if workers is None:
        return 0
    if int(workers) < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    return int(workers)


def _format(format: str | None) -> str | None:
    if format is None:
        return None
    f = str(format).lower().lstrip(".")
    if f not in ("las", "laz"):
        raise ValueError(f"format must be 'las' or 'laz', got {format!r}")
    return f


def _written(paths: list[str], cat: Catalog) -> Catalog:
    if not paths:
        return Catalog([], tolerance=cat.tolerance)
    return Catalog._from_core(_core.als_catalog(paths), cat.tolerance)
