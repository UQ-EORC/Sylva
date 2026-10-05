# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Whole-catalogue processing: ground, DTM, CHM, normalisation, filtering, retiling and thinning."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster
from .catalogue import Catalog, _as_catalog, _format, _workers, _written
from .engine import _run_kw


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
        IDW settings (see :func:`sylva.geo.interpolate.grid`).
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
        min_height: float = 0.0, drop_noise: bool = True, chunk_size: float | None = None,
        buffer: float = 20.0, workers: int | None = None) -> Raster:
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
    drop_noise
        Leave out points classified as noise (7 or 18), as
        :func:`filter` with ``classify=True`` marks them.
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
                      float(min_height), bool(drop_noise), **_run_kw(chunk_size, buffer, workers))
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

    Each input file is read once, whatever its extent: its points are sent to
    the tiles that hold them, held in temporary part files under ``out``,
    and each tile is then assembled from its parts. Flight lines and other
    large files without a spatial index, which every chunk of the other
    functions would decompress in full, are best retiled first.

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
