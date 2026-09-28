# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Large-area airborne lidar (ALS and UAV lidar): tiles, chunks and buffers.

An airborne survey comes as hundreds of LAS/LAZ tiles, together far larger
than memory, and no tile can be processed on its own: a ground filter or a
DTM cell at a tile edge needs the points across it. This module follows
lidR's ``LAScatalog`` (Roussel et al. 2020):

1. :func:`catalog` reads only the file headers (extent, point count, point
   format, CRS) and checks the tiling for overlaps, holes, mixed CRS and
   mixed formats, without reading a point.
2. The area is divided into chunks, one per tile or a regular grid
   (``chunk_size``), each read with a ``buffer`` of points from its
   neighbours. Only the files that overlap a chunk are opened, and only the
   points inside its buffered box are kept.
3. Chunks run in parallel, as many at a time as the memory budget of
   :mod:`sylva.limits` allows, and results are assembled in chunk order, so
   they do not depend on the number of workers. Point outputs keep only the
   chunk's own (core) points; raster outputs are joined into one seamless
   raster on a grid shared by the whole catalogue.

:func:`apply` runs any function this way; :func:`classify_ground`,
:func:`dtm`, :func:`chm`, :func:`normalize`, :func:`filter`,
:func:`retile` and :func:`decimate` are built on the same engine, in the
Rust core. :func:`sylva.synthetic.als_flight` simulates an airborne survey
to try them on.

Examples
--------
>>> from sylva import als
>>> cat = als.catalog("tiles/")                            # doctest: +SKIP
>>> print(cat.report())                                    # doctest: +SKIP
>>> ground = als.classify_ground(cat, "ground/")           # doctest: +SKIP
>>> dtm = als.dtm(ground, resolution=1.0)                  # doctest: +SKIP
>>> chm = als.chm(ground, resolution=0.5)                  # doctest: +SKIP
"""

from __future__ import annotations

import fnmatch
import itertools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core, progress
from .pointcloud import PointCloud
from .raster import Raster

__all__ = [
    "Tile", "Catalog", "Chunk", "catalog", "apply", "classify_ground", "dtm", "chm", "normalize",
    "filter", "retile", "decimate", "write_tiles",
]

_BUFFER_ATTR = "buffer"


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
            extents), ``density`` (points/m²), ``crs``, ``point_formats``
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


def _strip_buffer(cloud: PointCloud, chunk: Chunk) -> PointCloud:
    """The core points of a function's output, without the buffer flag."""
    if _BUFFER_ATTR in cloud.attrs:
        keep = ~np.asarray(cloud.attrs[_BUFFER_ATTR], dtype=bool)
        return cloud[keep].without(_BUFFER_ATTR)
    x0, y0, x1, y1 = chunk.core
    inside = (cloud.x >= x0) & (cloud.y >= y0)
    inside &= (cloud.x <= x1) & (cloud.y <= y1) if chunk.own is not None else (cloud.x < x1) & (cloud.y < y1)
    return cloud[inside]


def apply(catalog: Catalog, fn: Callable, chunk_size: float | None = None, buffer: float = 20.0,
          workers: int | None = None, out: str | Path | None = None, format: str | None = None,
          origin=None):
    """Run a function over a catalogue, chunk by chunk, with buffers.

    Each chunk is read (only the files overlapping it, only the points in
    its buffered box) and handed to ``fn`` as ``fn(cloud, chunk)``.
    ``cloud`` carries a boolean ``buffer`` attribute, True for the points
    around the chunk that belong to its neighbours; ``chunk`` is the
    :class:`Chunk`. Chunks with no points of their own are skipped.

    Parameters
    ----------
    catalog
        From :func:`catalog` (or anything :func:`catalog` accepts).
    fn
        ``fn(cloud, chunk)`` returning a :class:`~sylva.PointCloud`, a
        :class:`~sylva.Raster`, None or anything else. A point cloud should
        keep the ``buffer`` attribute (any subset of ``cloud`` does), so
        that the buffer can be removed; without it the points outside the
        chunk's core are dropped by position.
    chunk_size
        Side of square chunks (m); None for one chunk per tile.
    buffer
        Width (m) of the band of neighbouring points around each chunk.
        Make it at least as wide as whatever ``fn`` looks across (a
        filter's window, a crown's radius).
    workers
        Chunks processed at once; the number of CPUs if None. Fewer are
        run when that many chunks would not fit in the memory budget
        (:mod:`sylva.limits`, assuming about 256 bytes per point). The
        result does not depend on it.
    out
        Directory for point outputs: each chunk's core points are written
        to ``out/<chunk name>.<ext>`` in the point format, scale and CRS of
        its tile, instead of being returned.
    format
        ``"las"`` or ``"laz"`` for the files in ``out``; that of each
        chunk's tile if None.
    origin
        Grid corner for ``chunk_size``; see :meth:`Catalog.chunks`.

    Returns
    -------
    PointCloud, Catalog, Raster or list
        What ``fn`` returned, assembled: point clouds are stacked in chunk
        order without their buffers (or, with ``out``, the :class:`Catalog`
        of the files written); rasters, which must share a resolution and
        the catalogue grid's alignment (corner a whole number of cells from
        the catalogue's minimum snapped down to the resolution), are joined
        into one, each cell from the chunk whose core holds its centre; any
        other results come back as a list in chunk order, None for skipped
        chunks.

    Raises
    ------
    ValueError
        For bad arguments, a catalogue with missing files, a chunk too large
        for the memory budget, or rasters that do not line up. Whatever
        ``fn`` raises is raised as it is, and no new chunks are started.

    Examples
    --------
    Any single-cloud function can run this way. A canopy height model from
    tiles already normalised, on the catalogue grid (``bounds`` from the
    chunk's buffered box, snapped to the cells, keeps each chunk's raster on
    that grid):

    >>> def canopy(cloud, chunk):                          # doctest: +SKIP
    ...     x0, y0, x1, y1 = chunk.outer
    ...     b = (np.floor(x0), np.floor(y0), x1, y1)
    ...     return ground.make_chm(cloud, 1.0, bounds=b)
    >>> chm = als.apply(cat, canopy, buffer=0)             # doctest: +SKIP

    Numbers per chunk come back as a list:

    >>> counts = als.apply(cat, lambda c, ch: int((~c.attrs["buffer"]).sum()))  # doctest: +SKIP
    """
    cat = _as_catalog(catalog)
    if not callable(fn):
        raise ValueError("fn must be callable as fn(cloud, chunk)")
    fmt = _format(format)
    chunks = cat.chunks(chunk_size, buffer, origin)
    if not chunks:
        return []
    core = cat._core()
    n_workers = _core.als_workers([c.est_points for c in chunks], _workers(workers),
                                  int(_core.ALS_BYTES_PER_POINT))
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)

    def job(i: int):
        ch = chunks[i]
        xyz, attrs, buf = _core.als_read_chunk(core, ch._core())
        if buf.all():
            return None
        attrs[_BUFFER_ATTR] = buf
        res = fn(PointCloud(xyz, attrs), ch)
        if isinstance(res, PointCloud):
            res = _strip_buffer(res, ch)
            if out is not None:
                if len(res) == 0:
                    return None
                like = ch.own if ch.own is not None else ch.files[0]
                ext = fmt or (Path(cat.tiles[like].path).suffix.lower().lstrip(".") or "laz")
                ext = ext if ext in ("las", "laz") else "laz"
                path = _core.als_output_path(core, str(out), ch.name, ext)
                _core.als_write_like(core, like, path, res.xyz, res.attrs)
                return _Written(path)
        return res

    results: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=n_workers) as ex, \
            progress.task("processing ALS chunks", len(chunks)) as bar:
        order = iter(range(len(chunks)))
        running = {i: ex.submit(job, i) for i in itertools.islice(order, n_workers)}
        for i in range(len(chunks)):
            try:
                results[i] = running.pop(i).result()
            except BaseException:
                for f in running.values():
                    f.cancel()
                raise
            bar.update()
            nxt = next(order, None)
            if nxt is not None:
                running[nxt] = ex.submit(job, nxt)
    return _assemble(cat, chunks, results, out)


@dataclass(frozen=True)
class _Written:
    path: str


def _assemble(cat: Catalog, chunks: list[Chunk], results: list, out):
    got = [r for r in results if r is not None]
    if got and all(isinstance(r, _Written) for r in got):
        return _written([r.path for r in got], cat)
    if out is not None and not got:
        return Catalog([], tolerance=cat.tolerance)
    if got and all(isinstance(r, PointCloud) for r in got):
        return PointCloud.concatenate(got)
    if got and all(isinstance(r, Raster) for r in got):
        res = float(got[0].resolution)
        if any(abs(float(r.resolution) - res) > 1e-9 * res for r in got):
            raise ValueError("the rasters returned differ in resolution; they cannot be joined")
        parts = [(np.ascontiguousarray(r.data, dtype=np.float64), float(r.xmin), float(r.ymin),
                  float(r.resolution), c.core)
                 for r, c in zip(results, chunks, strict=True) if r is not None]
        return cat._raster(_core.als_mosaic(cat._core(), res, parts))
    return [r.path if isinstance(r, _Written) else r for r in results]


def _run_kw(chunk_size, buffer, workers) -> dict:
    return {"chunk_size": None if chunk_size is None else float(chunk_size),
            "buffer": float(buffer), "workers": _workers(workers)}


def classify_ground(catalog: Catalog, out: str | Path, method: str = "csf",
                    last_returns: bool = False, cloth_resolution: float = 0.5,
                    rigidness: int = 2, class_threshold: float = 0.3, iterations: int = 500,
                    time_step: float = 0.65, cell_size: float = 0.5, max_window: float = 10.0,
                    slope: float = 0.3, initial_distance: float = 0.15,
                    max_distance: float = 2.0, chunk_size: float | None = None,
                    buffer: float = 20.0, workers: int | None = None,
                    format: str | None = None) -> Catalog:
    """Classify ground over a whole catalogue and write the classified tiles.

    The filter of :func:`sylva.ground.classify_ground_csf` or
    :func:`sylva.ground.classify_ground_pmf` runs on each tile with its
    buffer, and each tile's own points are written with ``classification``
    2 (ground) or 1 (other). Points already classified as noise (7 or 18)
    take no part and keep their class. On a single tile, with the same
    parameters, the classes are those of the single-cloud function.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the classified tiles (created if needed), one per
        input tile with the same name.
    method : {"csf", "pmf"}
        Cloth simulation or progressive morphological filter.
    last_returns
        Only last returns (``return_number == number_of_returns``) can be
        ground; the others are classified 1. Helps under dense canopy.
    cloth_resolution, rigidness, class_threshold, iterations, time_step
        CSF parameters, as in :func:`sylva.ground.classify_ground_csf`.
    cell_size, max_window, slope, initial_distance, max_distance
        PMF parameters, as in :func:`sylva.ground.classify_ground_pmf`.
        ``max_window`` should exceed the largest building or crown.
    chunk_size, buffer, workers
        As for :func:`apply`. The buffer should be at least about half the
        largest non-ground object, so that the filter sees ground around it.
    format
        ``"las"`` or ``"laz"``; the input's if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For an unknown method, bad parameters, or an output that would
        overwrite an input tile.
    """
    if method not in ("csf", "pmf"):
        raise ValueError(f"unknown method {method!r}; expected 'csf' or 'pmf'")
    cat = _as_catalog(catalog)
    paths = _core.als_classify_ground(
        cat._core(), str(out), method, bool(last_returns), _format(format),
        cloth_resolution=float(cloth_resolution), rigidness=int(rigidness),
        class_threshold=float(class_threshold), iterations=int(iterations),
        time_step=float(time_step), cell_size=float(cell_size), max_window=float(max_window),
        slope=float(slope), initial_distance=float(initial_distance),
        max_distance=float(max_distance), **_run_kw(chunk_size, buffer, workers))
    return _written(paths, cat)


def dtm(catalog: Catalog, resolution: float = 1.0, method: str = "lowest", power: float = 2.0,
        k: int = 12, max_distance: float | None = None, chunk_size: float | None = None,
        buffer: float = 20.0, workers: int | None = None) -> Raster:
    """Digital terrain model of a whole catalogue, seamless across tiles.

    Each chunk's DTM is made from its ground points (``classification ==
    2``) with its buffer, as :func:`sylva.ground.make_dtm` makes it, on one
    grid for the whole catalogue; each cell is then taken from the chunk
    whose core holds it. Away from the outer edge of the survey the result
    equals the DTM of all the tiles merged into one cloud.

    Parameters
    ----------
    catalog
        Tiles with ground classified (e.g. by :func:`classify_ground`).
    resolution
        Cell size (m).
    method : {"lowest", "tin", "natural", "idw"}
        As in :func:`sylva.ground.make_dtm`: the lowest ground point per
        cell with gaps filled, or an interpolation at cell centres.
    power, k, max_distance
        IDW settings (see :func:`sylva.interpolate.grid`).
    chunk_size, buffer, workers
        As for :func:`apply`. The buffer must exceed the widest gap in the
        ground (under a building or a dense crown) for the fill to match
        across tiles.

    Returns
    -------
    Raster
        On the catalogue grid (south-west corner at its minimum snapped down
        to a multiple of ``resolution``), with the catalogue's CRS. Cells in
        chunks with fewer than 3 ground points are NaN.

    Raises
    ------
    ValueError
        For an unknown method or a non-positive resolution; if the tiles
        have no ``classification``.
    """
    if method not in ("lowest", "tin", "natural", "idw"):
        raise ValueError(f"unknown method {method!r}; expected 'lowest', 'tin', 'natural' or 'idw'")
    cat = _as_catalog(catalog)
    md = None if max_distance is None else float(max_distance)
    d = _core.als_dtm(cat._core(), float(resolution), method, float(power), int(k), md,
                      **_run_kw(chunk_size, buffer, workers))
    return cat._raster(d)


def _heights(dtm) -> tuple[str, tuple | None]:
    if isinstance(dtm, str):
        if dtm != "auto":
            raise ValueError(f"dtm must be 'auto', None or a Raster, got {dtm!r}")
        return "auto", None
    if dtm is None:
        return "z", None
    if isinstance(dtm, Raster):
        return "dtm", (np.ascontiguousarray(dtm.data, dtype=np.float64), float(dtm.xmin),
                       float(dtm.ymin), float(dtm.resolution))
    raise ValueError(f"dtm must be 'auto', None or a Raster, got {type(dtm).__name__}")


def chm(catalog: Catalog, resolution: float = 0.5, dtm="auto", dtm_resolution: float = 1.0,
        min_height: float = 0.0, chunk_size: float | None = None, buffer: float = 20.0,
        workers: int | None = None) -> Raster:
    """Canopy height model of a whole catalogue: the highest point per cell.

    Heights above ground are computed per chunk, then gridded as
    :func:`sylva.ground.make_chm` grids them (the "point to raster" method,
    ``p2r`` in lidR).

    Parameters
    ----------
    catalog
        The tiles.
    resolution
        Cell size (m).
    dtm : "auto", None or Raster
        Where heights come from. ``"auto"``: a DTM made per chunk from its
        ground points at ``dtm_resolution`` (the tiles must have ground
        classified). A :class:`~sylva.Raster`: z minus this DTM. None: z as
        it is, for tiles already normalised (``normalize(...,
        replace_z=True)``), or for a surface model.
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    min_height
        Cells with nothing at or above this height are 0.
    chunk_size, buffer, workers
        As for :func:`apply`.

    Returns
    -------
    Raster
        Maximum height per cell (m), on the catalogue grid, with the
        catalogue's CRS. No pit filling. Cells in chunks without enough
        ground for ``"auto"`` are NaN.

    Raises
    ------
    ValueError
        For a bad ``dtm`` or resolution.
    """
    cat = _as_catalog(catalog)
    mode, raster = _heights(dtm)
    d = _core.als_chm(cat._core(), float(resolution), mode, raster, float(dtm_resolution),
                      float(min_height), **_run_kw(chunk_size, buffer, workers))
    return cat._raster(d)


def normalize(catalog: Catalog, out: str | Path, dtm="auto", dtm_resolution: float = 1.0,
              replace_z: bool = False, chunk_size: float | None = None, buffer: float = 20.0,
              workers: int | None = None, format: str | None = None) -> Catalog:
    """Height above ground for every point of a catalogue, written as tiles.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the normalised tiles, one per input tile.
    dtm : "auto" or Raster
        ``"auto"`` makes a DTM per chunk from its ground points (and its
        buffer's) at ``dtm_resolution``, as :func:`dtm` would; or give one.
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    replace_z
        Replace z by the height and keep the elevation in an ``elevation``
        attribute (as lidR's ``normalize_height`` does), rather than adding
        a ``height`` attribute. Replaced z is what :func:`chm` with
        ``dtm=None`` and the TLS functions that read z as height expect.
    chunk_size, buffer, workers
        As for :func:`apply`.
    format
        ``"las"`` or ``"laz"``; the input's if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For a bad ``dtm``, or a tile with fewer than 3 ground points within
        its buffer when ``dtm="auto"``.
    """
    cat = _as_catalog(catalog)
    mode, raster = _heights(dtm)
    if mode == "z":
        raise ValueError("normalize needs dtm='auto' or a DTM Raster")
    paths = _core.als_normalize(cat._core(), str(out), mode, raster, float(dtm_resolution),
                                bool(replace_z), _format(format),
                                **_run_kw(chunk_size, buffer, workers))
    return _written(paths, cat)


def filter(catalog: Catalog, out: str | Path, method: str = "ror", radius: float = 2.0,
           min_neighbors: int = 3, k: int = 8, std_ratio: float = 3.0, classify: bool = False,
           chunk_size: float | None = None, buffer: float = 10.0, workers: int | None = None,
           format: str | None = None) -> Catalog:
    """Remove noise (isolated points: birds, multipath, atmospheric returns).

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the filtered tiles, one per input tile.
    method : {"ror", "sor"}
        ``"ror"``: fewer than ``min_neighbors`` others within ``radius``
        (:func:`sylva.filters.radius_outlier_removal`); purely local, so
        tile edges cannot show. ``"sor"``: mean distance to ``k``
        neighbours beyond ``std_ratio`` standard deviations above the mean
        (:func:`sylva.filters.statistical_outlier_removal`); the mean and
        deviation are those of each tile with its buffer.
    radius, min_neighbors
        ``"ror"`` settings (m, count).
    k, std_ratio
        ``"sor"`` settings.
    classify
        Keep every point and set ``classification`` 7 (ASPRS low noise) on
        the noise, rather than removing it. :func:`classify_ground` leaves
        such points out.
    chunk_size, buffer, workers
        As for :func:`apply`; the buffer should exceed ``radius``.
    format
        ``"las"`` or ``"laz"``; the input's if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For an unknown method or non-positive settings.
    """
    if method not in ("ror", "sor"):
        raise ValueError(f"unknown method {method!r}; expected 'ror' or 'sor'")
    cat = _as_catalog(catalog)
    paths = _core.als_filter(cat._core(), str(out), method, int(k), float(std_ratio),
                             float(radius), int(min_neighbors), bool(classify), _format(format),
                             **_run_kw(chunk_size, buffer, workers))
    return _written(paths, cat)


def retile(catalog: Catalog, out: str | Path, size: float, buffer: float = 0.0, origin=None,
           workers: int | None = None, format: str | None = None) -> Catalog:
    """Cut a catalogue into new square tiles.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the new tiles, named ``<xmin>_<ymin>.<ext>``.
    size
        Side of the new tiles (m).
    buffer
        With a buffer (m), each tile also holds the points within that
        distance of it, flagged by a ``buffer`` attribute (1 for buffer
        points), for software that processes tiles one at a time.
    origin
        ``(x, y)`` of a tile corner; by default the catalogue's minimum
        snapped down to a multiple of ``size``.
    workers
        As for :func:`apply`.
    format
        ``"las"`` or ``"laz"``; that of the first input tile each new tile
        draws on if None.

    Returns
    -------
    Catalog
        The new tiles. Every input point is in exactly one of them (not
        counting buffers).

    Raises
    ------
    ValueError
        For a non-positive size or negative buffer.
    """
    cat = _as_catalog(catalog)
    o = None if origin is None else (float(origin[0]), float(origin[1]))
    paths = _core.als_retile(cat._core(), str(out), float(size), float(buffer), o,
                             _format(format), _workers(workers))
    return _written(paths, cat)


def decimate(catalog: Catalog, out: str | Path, method: str = "random", fraction: float = 0.5,
             size: float = 1.0, seed: int = 0, chunk_size: float | None = None,
             workers: int | None = None, format: str | None = None) -> Catalog:
    """Thin a catalogue.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the thinned tiles.
    method : {"random", "voxel", "highest"}
        ``"random"``: keep ``round(fraction * n)`` points of each tile (as
        :func:`sylva.filters.random_subsample`, seeded with ``seed`` plus
        the tile's position in the catalogue, or the chunk's index on a
        grid). ``"voxel"``: the first point of each ``size`` m voxel.
        ``"highest"``: the highest point of each ``size`` m cell in x, y, a
        light surface for canopy height models. Voxels and cells are
        anchored at the coordinate origin, so they line up across tiles,
        but one that straddles two tiles keeps a point in each.
    fraction
        Share of points kept by ``"random"`` (0-1).
    size
        Voxel or cell size (m).
    seed
        Seed of ``"random"``.
    chunk_size, workers
        As for :func:`apply` (no buffer is needed).
    format
        ``"las"`` or ``"laz"``; the input's if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For an unknown method, a fraction outside 0-1 or a non-positive size.
    """
    if method not in ("random", "voxel", "highest"):
        raise ValueError(f"unknown method {method!r}; expected 'random', 'voxel' or 'highest'")
    cat = _as_catalog(catalog)
    cs = None if chunk_size is None else float(chunk_size)
    paths = _core.als_decimate(cat._core(), str(out), method, float(fraction), int(seed),
                               float(size), _format(format), cs, _workers(workers))
    return _written(paths, cat)


def write_tiles(cloud: PointCloud, out: str | Path, size: float, origin=None, format: str = "laz",
                point_format: int = 6, scale: float = 0.001, epsg: int | None = None,
                tolerance: float = 1.0) -> Catalog:
    """Write a cloud as square LAS/LAZ tiles.

    Parameters
    ----------
    cloud
        Points; attributes are written as in :func:`sylva.io.write` (the
        LAS dimensions by name, anything else as extra bytes).
    out
        Directory for the tiles, named ``<xmin>_<ymin>.<ext>``.
    size
        Tile side (m).
    origin
        ``(x, y)`` of a tile corner; by default the cloud's minimum snapped
        down to a multiple of ``size``.
    format
        ``"las"`` or ``"laz"``.
    point_format
        LAS point data record format.
    scale
        Coordinate quantisation (m).
    epsg
        EPSG code recorded in each file (as GeoTIFF keys), so that the
        catalogue knows its CRS.
    tolerance
        For the returned catalogue.

    Returns
    -------
    Catalog
        The tiles written (empty tiles are not written).

    Raises
    ------
    ValueError
        For a non-positive size, an unknown format or non-finite coordinates.
    """
    fmt = _format(format)
    if fmt is None:
        raise ValueError("format must be 'las' or 'laz'")
    o = None if origin is None else (float(origin[0]), float(origin[1]))
    written = _core.als_write_tiles(cloud.xyz, cloud.attrs, str(out), float(size), o, fmt,
                                    int(point_format), float(scale),
                                    None if epsg is None else int(epsg))
    if not written:
        return Catalog([], tolerance=tolerance)
    return Catalog._from_core(_core.als_catalog([p for p, _ in written]), float(tolerance))



# Individual trees (tree tops, crowns, labelled tiles) live in their own module.
from .als_trees import (  # noqa: E402
    LinearWindow,
    Trees,
    TreeTops,
    crown_hull,
    find_trees,
    li2012,
    locate_trees,
    segment_crowns,
    segment_trees,
)

__all__ += ["LinearWindow", "TreeTops", "Trees", "locate_trees", "segment_crowns", "li2012",
            "crown_hull", "segment_trees", "find_trees"]
