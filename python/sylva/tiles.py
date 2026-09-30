# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Point clouds of any size as tiles, with results identical to the whole cloud.

A plot scanned from a hundred positions holds hundreds of millions of points,
more than fit in memory with room to process them. This module keeps such a
plot as square LAS/LAZ tiles and processes it one tile at a time, each with a
buffer of points from its neighbours, on the engine of :mod:`sylva.als`
(:class:`Catalog`, :class:`Chunk`, :func:`apply`, :func:`retile` and
:func:`write_tiles` are the same objects here as there):

1. :func:`from_scans` builds the tiles from the scans, one scan at a time,
   thinning them on one global voxel grid and recording each point's
   ``scan_id``. The tiles hold exactly what
   :func:`sylva.filters.voxel_downsample` keeps of all the scans
   concatenated, on the same grid.
2. :func:`voxel_downsample`, :func:`statistical_outlier_removal`,
   :func:`radius_outlier_removal`, :func:`estimate_normals`,
   :func:`planarity_linearity` and :func:`detect_stems` give, tile by tile,
   the result of their single-cloud counterparts on the whole plot. The
   neighbourhood operations are exact for any buffer: a point whose
   neighbours might lie beyond it is evaluated again from a wider read.
3. :func:`classify_ground`, :func:`dtm` and :func:`normalize` are the
   :mod:`sylva.als` operations with settings for plots.

Tiles keep every attribute they carry (``tree_id``, ``wood``, ``scan_id``,
``height``, ...): LAS dimensions by name, others as extra bytes. Each
function that writes tiles returns the :class:`Catalog` of what it wrote;
:func:`last_run` tells how many points were held at once.

Examples
--------
>>> from sylva import tiles
>>> cat = tiles.from_scans(sorted(Path("scans").glob("*.laz")), "tiles/",   # doctest: +SKIP
...                        tile_size=10.0, voxel_size=0.02, transforms=sops)
>>> clean = tiles.statistical_outlier_removal(cat, "sor/", k=6, std_ratio=1.0)  # doctest: +SKIP
>>> tiles.last_run().max_points                                            # doctest: +SKIP
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core, als, progress
from .als import Catalog, Chunk, _as_catalog, _format, _workers, _written, apply, catalog, retile, write_tiles
from .pointcloud import PointCloud
from .raster import Raster
from .trees import Tree

__all__ = [
    "Catalog", "Chunk", "catalog", "apply", "retile", "write_tiles", "RunInfo", "last_run",
    "from_scans", "voxel_downsample", "statistical_outlier_removal", "radius_outlier_removal",
    "estimate_normals", "planarity_linearity", "classify_ground", "dtm", "normalize",
    "detect_stems",
]


@dataclass(frozen=True)
class RunInfo:
    """What a tiled run did, to check that memory stayed bounded.

    Parameters
    ----------
    chunks
        Tiles processed (written, for :func:`from_scans`).
    max_points
        Most points held at once by one tile: the tile with its buffer, a
        widened read, or for :func:`from_scans` the largest scan or tile
        being assembled. Memory is about 256 bytes times this, per worker.
    points_read
        Points read over the run, buffers and second reads included.
    rereads
        Tiles read a second time, wider, because some of their points had
        neighbours that could lie beyond the buffer.
    widened_points
        Points evaluated from such a wider read.
    """

    chunks: int
    max_points: int
    points_read: int
    rereads: int
    widened_points: int


_LAST: list[RunInfo | None] = [None]


def last_run() -> RunInfo | None:
    """The :class:`RunInfo` of the last tiled operation of this module.

    Returns
    -------
    RunInfo or None
        None before any has run, and after :func:`classify_ground`,
        :func:`dtm` and :func:`normalize`, which report nothing.
    """
    return _LAST[0]


def _record(info: dict | None) -> None:
    _LAST[0] = None if info is None else RunInfo(**{k: int(v) for k, v in info.items()})


def _finish(paths, info, cat: Catalog) -> Catalog:
    _record(info)
    return _written(list(paths), cat)


def _buffer(buffer: float) -> float:
    b = float(buffer)
    if not (np.isfinite(b) and b >= 0):
        raise ValueError(f"buffer must be a non-negative number of metres, got {buffer}")
    return b


def _matrix(t) -> np.ndarray | None:
    if t is None:
        return None
    m = np.ascontiguousarray(t, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"each transform must be a finite 4x4 matrix, got shape {m.shape}")
    return m


def from_scans(scans, out: str | Path, tile_size: float = 10.0, voxel_size: float | None = None,
               transforms=None, origin=(0.0, 0.0, 0.0), bounds=None, format: str = "laz",
               point_format: int = 6, scale: float = 0.001, epsg: int | None = None,
               workers: int | None = None, tolerance: float = 1.0) -> Catalog:
    """Build square tiles from scans, thinned on one global voxel grid.

    The scans are taken one at a time, in order. Each is moved by its
    transform, cropped to ``bounds``, and keeps the first point of each
    voxel it has; those points go, tile by tile, to scratch files in
    ``out`` at full precision. Each tile is then assembled keeping the first
    point per voxel in scan order, and written with a ``scan_id`` attribute
    (the scan's position in ``scans``). Tiles are cut along voxel
    boundaries, so no voxel is split between two tiles.

    The tiles hold exactly the points that
    ``filters.voxel_downsample(concatenated, voxel_size, origin=origin)``
    keeps of all the (transformed, cropped) scans concatenated in order,
    with their attributes; only their coordinates are rounded to ``scale``.
    Only one scan, and one tile per worker, is in memory at a time.

    Parameters
    ----------
    scans
        Paths of point cloud files (any format :func:`sylva.read` reads),
        :class:`~sylva.PointCloud` objects, or a mix, in scan order.
    out
        Directory for the tiles, named ``<xmin>_<ymin>.<format>``.
    tile_size
        Side of the tiles (m); a whole number of voxels.
    voxel_size
        Voxel size (m) of the thinning; None keeps every point.
    transforms
        One 4x4 matrix (or None) per scan, applied before anything else
        (a scan's SOP, say), as :meth:`sylva.PointCloud.transform` applies it.
    origin
        ``(x, y, z)`` of a corner of the voxel grid and of the tile grid;
        ``(0, 0, 0)`` puts voxels and tiles at multiples of their size.
    bounds
        ``(xmin, ymin, xmax, ymax)``: keep only the points inside (closed),
        a plot and its buffer, say.
    format
        ``"las"`` or ``"laz"``.
    point_format
        LAS point data record format of the tiles.
    scale
        Coordinate quantisation of the tiles (m).
    epsg
        EPSG code recorded in each tile.
    workers
        Tiles assembled at once; the number of CPUs if None, fewer if memory
        is short.
    tolerance
        For the returned catalogue (see :class:`Catalog`).

    Returns
    -------
    Catalog
        The tiles, south to north and west to east.

    Raises
    ------
    ValueError
        For a tile size that is not a whole number of voxels, bad bounds, a
        transform that is not 4x4, or non-finite coordinates.
    OSError
        For a scan that cannot be read.

    Notes
    -----
    The scratch files hold every scan's thinned points (about 40 bytes each
    plus the attributes) until the tiles are written; they are removed
    afterwards, and after an error.
    """
    scans = list(scans)
    if not scans:
        raise ValueError("no scans given")
    if transforms is None:
        transforms = [None] * len(scans)
    transforms = list(transforms)
    if len(transforms) != len(scans):
        raise ValueError(f"{len(transforms)} transforms for {len(scans)} scans")
    mats = [_matrix(t) for t in transforms]
    fmt = _format(format)
    if fmt is None:
        raise ValueError("format must be 'las' or 'laz'")
    o = tuple(float(v) for v in origin)
    if len(o) != 3:
        raise ValueError(f"origin must be (x, y, z), got {origin!r}")
    b = None if bounds is None else tuple(float(v) for v in bounds)
    if b is not None and len(b) != 4:
        raise ValueError(f"bounds must be (xmin, ymin, xmax, ymax), got {bounds!r}")
    vs = None if voxel_size is None else float(voxel_size)
    tiler = _core.ScanTiler(str(out), float(tile_size), vs, o, b, fmt, int(point_format),
                            float(scale), None if epsg is None else int(epsg))
    with progress.task("tiling scans", len(scans)) as bar:
        for scan, m in zip(scans, mats, strict=True):
            if isinstance(scan, PointCloud):
                tiler.add_cloud(scan.xyz, scan.attrs, m)
            elif isinstance(scan, (str, os.PathLike)):
                tiler.add_file(str(scan), m)
            else:
                raise ValueError(f"a scan must be a path or a PointCloud, got {type(scan).__name__}")
            bar.update()
    written, info = tiler.finish(_workers(workers))
    _record(info)
    if not written:
        return Catalog([], tolerance=tolerance)
    return Catalog._from_core(_core.als_catalog([p for p, _ in written]), float(tolerance))


def voxel_downsample(catalog: Catalog, out: str | Path, voxel_size: float,
                     origin=(0.0, 0.0, 0.0), workers: int | None = None,
                     format: str | None = None) -> Catalog:
    """Thin tiles to the first point of each voxel of one global grid.

    Points are taken in catalogue order (tile by tile, each in file order),
    so the result is ``filters.voxel_downsample(cat.read(), voxel_size,
    origin=origin)``: a voxel that straddles two tiles keeps one point, from
    the first tile that has one. Each tile is read with a buffer of one
    voxel, which is all this needs.

    Parameters
    ----------
    catalog
        The tiles (a :class:`Catalog`, or anything :func:`catalog` takes).
    out
        Directory for the thinned tiles, one per input tile with its name.
    voxel_size
        Voxel size (m).
    origin
        ``(x, y, z)`` of a corner of the voxel grid.
    workers
        Tiles processed at once; the number of CPUs if None, fewer if memory
        is short. The result does not depend on it.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For a non-positive voxel size or an output that would overwrite an
        input tile.
    """
    cat = _as_catalog(catalog)
    o = tuple(float(v) for v in origin)
    paths, info = _core.tiles_thin(cat._core(), str(out), float(voxel_size), o, _format(format),
                                   _workers(workers))
    return _finish(paths, info, cat)


def statistical_outlier_removal(catalog: Catalog, out: str | Path, k: int = 8,
                                std_ratio: float = 2.0, classify: bool = False,
                                buffer: float = 1.0, workers: int | None = None,
                                format: str | None = None) -> Catalog:
    """Statistical outlier removal over all the tiles, as on the whole cloud.

    CloudCompare's SOR (:func:`sylva.filters.statistical_outlier_removal`)
    compares each point's mean distance to its ``k`` nearest neighbours with
    ``mean + std_ratio * std`` of that distance over the whole cloud. Tiled,
    it takes two passes: the per-point distances tile by tile (each with its
    buffer, and a wider second read for any point whose neighbours could lie
    beyond it), kept in scratch files beside the output; then the mean and
    standard deviation over every point, summed in catalogue order as the
    whole-cloud filter sums them; then the tiles are written with the
    threshold applied. The points kept are exactly those
    ``filters.statistical_outlier_removal(cat.read(), k, std_ratio)`` keeps.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the filtered tiles.
    k
        Neighbours per point.
    std_ratio
        Threshold in standard deviations; lower removes more.
    classify
        Keep every point and set ``classification`` 7 (low noise) on the
        outliers instead of removing them.
    buffer
        Width (m) of the band of neighbouring points read around each tile.
        The result does not depend on it; wider than the ``k``-th neighbour
        distance of almost every point, it saves second reads.
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For ``k`` below 1, a non-finite ``std_ratio`` or a negative buffer.
    """
    cat = _as_catalog(catalog)
    if int(k) < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    paths, info = _core.tiles_sor(cat._core(), str(out), int(k), float(std_ratio), bool(classify),
                                  _format(format), _buffer(buffer), _workers(workers))
    return _finish(paths, info, cat)


def radius_outlier_removal(catalog: Catalog, out: str | Path, radius: float,
                           min_neighbors: int = 4, classify: bool = False,
                           buffer: float | None = None, workers: int | None = None,
                           format: str | None = None) -> Catalog:
    """Radius outlier removal over all the tiles, as on the whole cloud.

    Points with fewer than ``min_neighbors`` others within ``radius`` are
    removed, exactly as :func:`sylva.filters.radius_outlier_removal` removes
    them from the whole cloud.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the filtered tiles.
    radius
        Search radius (m).
    min_neighbors
        Points with fewer other points within ``radius`` are outliers.
    classify
        Keep every point and set ``classification`` 7 on the outliers.
    buffer
        Band read around each tile (m); 1 cm more than ``radius`` if None,
        which is all this needs. A narrower one gives the same result, with
        second reads.
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For a non-positive radius or a negative buffer.
    """
    cat = _as_catalog(catalog)
    if int(min_neighbors) < 0:
        raise ValueError(f"min_neighbors must be zero or more, got {min_neighbors}")
    # A little over the radius: tile extents are rounded to the coordinate scale.
    b = float(radius) + 0.01 if buffer is None else _buffer(buffer)
    paths, info = _core.tiles_ror(cat._core(), str(out), float(radius), int(min_neighbors),
                                  bool(classify), _format(format), b, _workers(workers))
    return _finish(paths, info, cat)


def _features(catalog, out, k, feature, buffer, workers, format) -> Catalog:
    cat = _as_catalog(catalog)
    if int(k) < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    paths, info = _core.tiles_features(cat._core(), str(out), int(k), feature, _format(format),
                                       _buffer(buffer), _workers(workers))
    return _finish(paths, info, cat)


def estimate_normals(catalog: Catalog, out: str | Path, k: int = 12, buffer: float = 1.0,
                     workers: int | None = None, format: str | None = None) -> Catalog:
    """Add surface normals to every point of the tiles.

    Each point gets ``normal_x``, ``normal_y`` and ``normal_z``: the values
    :func:`sylva.filters.estimate_normals` gives it in the whole cloud (the
    eigenvector of the smallest eigenvalue of its ``k`` neighbours'
    covariance; the sign is arbitrary).

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the tiles with normals.
    k
        Neighbours per point (at least 3 are used).
    buffer
        Band read around each tile (m); the result does not depend on it.
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For ``k`` below 1 or a negative buffer.

    Notes
    -----
    Neighbours at equal distances are ordered by the search tree, whose
    layout differs between a tile and the whole cloud; where such a tie
    falls at the ``k``-th neighbour (common with quantised coordinates), a
    point's value can come from another neighbour. On the Tumbarumba plot
    this affected 0.02 per cent of the points.
    """
    return _features(catalog, out, k, "normals", buffer, workers, format)


def planarity_linearity(catalog: Catalog, out: str | Path, k: int = 20, buffer: float = 1.0,
                        workers: int | None = None, format: str | None = None) -> Catalog:
    """Add planarity and linearity to every point of the tiles.

    Each point gets ``planarity`` and ``linearity``, the values
    :func:`sylva.filters.planarity_linearity` gives it in the whole cloud.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the tiles with the features.
    k
        Neighbours per point (at least 3 are used).
    buffer
        Band read around each tile (m); the result does not depend on it.
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For ``k`` below 1 or a negative buffer.

    Notes
    -----
    Neighbours at equal distances are ordered by the search tree, whose
    layout differs between a tile and the whole cloud; where such a tie
    falls at the ``k``-th neighbour (common with quantised coordinates), a
    point's value can come from another neighbour. On the Tumbarumba plot
    this affected 0.02 per cent of the points.
    """
    return _features(catalog, out, k, "shape", buffer, workers, format)


def classify_ground(catalog: Catalog, out: str | Path, method: str = "csf",
                    cloth_resolution: float = 0.5, rigidness: int = 2,
                    class_threshold: float = 0.3, iterations: int = 500,
                    time_step: float = 0.65, cell_size: float = 0.5, max_window: float = 10.0,
                    slope: float = 0.3, initial_distance: float = 0.15,
                    max_distance: float = 2.0, buffer: float = 10.0,
                    workers: int | None = None, format: str | None = None) -> Catalog:
    """Classify ground in every tile (:func:`sylva.als.classify_ground`, one chunk per tile).

    Each tile's own points are written with ``classification`` 2 (ground) or
    1 (other); points classified 7 or 18 (noise, e.g. by
    :func:`statistical_outlier_removal` with ``classify``) take no part and
    keep their class.

    With ``method="pmf"`` the result is that of
    :func:`sylva.ground.classify_ground_pmf` on the whole cloud once the
    buffer covers how far the filter looks: its grid is anchored at
    multiples of ``cell_size`` in every tile, and each opening of half-width
    ``w`` cells reaches ``2 w`` cells, so ``buffer >= 2 * cell_size * (1 + 2
    + 4 + ... + W)`` (``W`` the largest half-width, ``max_window /
    cell_size`` or less) plus the widest gap in the ground is sufficient;
    narrower buffers agree everywhere but near the tile edges in practice.
    ``method="csf"`` is not exact: the cloth is one sheet whose settling
    depends on the whole of it, so points near a tile edge can differ from
    the whole-cloud classification (rarely, with a buffer of a few metres).

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for the classified tiles.
    method : {"csf", "pmf"}
        Cloth simulation or progressive morphological filter.
    cloth_resolution, rigidness, class_threshold, iterations, time_step
        CSF parameters (:func:`sylva.ground.classify_ground_csf`).
    cell_size, max_window, slope, initial_distance, max_distance
        PMF parameters (:func:`sylva.ground.classify_ground_pmf`).
    buffer
        Band read around each tile (m).
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For an unknown method or bad parameters.
    """
    _record(None)
    return als.classify_ground(
        catalog, out, method=method, cloth_resolution=cloth_resolution, rigidness=rigidness,
        class_threshold=class_threshold, iterations=iterations, time_step=time_step,
        cell_size=cell_size, max_window=max_window, slope=slope,
        initial_distance=initial_distance, max_distance=max_distance, chunk_size=None,
        buffer=_buffer(buffer), workers=workers, format=format)


def dtm(catalog: Catalog, resolution: float = 0.5, method: str = "lowest", buffer: float = 10.0,
        workers: int | None = None) -> Raster:
    """Digital terrain model of all the tiles (:func:`sylva.als.dtm`, one chunk per tile).

    Away from the outer edge of the plot, and with a buffer wider than the
    widest gap in the ground, each cell is that of
    :func:`sylva.ground.make_dtm` on the ground points of the whole cloud.

    Parameters
    ----------
    catalog
        Tiles with ground classified (:func:`classify_ground`).
    resolution
        Cell size (m).
    method : {"lowest", "tin", "natural", "idw"}
        As in :func:`sylva.als.dtm`.
    buffer
        Band read around each tile (m).
    workers
        Tiles processed at once; see :func:`voxel_downsample`.

    Returns
    -------
    Raster
        On a grid anchored at multiples of ``resolution``.

    Raises
    ------
    ValueError
        For an unknown method, or tiles without ground points.
    """
    _record(None)
    return als.dtm(catalog, resolution=resolution, method=method, chunk_size=None,
                   buffer=_buffer(buffer), workers=workers)


def normalize(catalog: Catalog, out: str | Path, dtm="auto", dtm_resolution: float = 0.5,
              replace_z: bool = False, buffer: float = 10.0, workers: int | None = None,
              format: str | None = None) -> Catalog:
    """Height above ground for every point of the tiles (:func:`sylva.als.normalize`).

    With ``dtm="auto"`` each tile makes its DTM from its own and its
    buffer's ground points at ``dtm_resolution``, as :func:`dtm` would, so
    the heights are those of :func:`sylva.ground.normalize_height` with the
    whole-cloud :func:`sylva.ground.make_dtm` (to rounding of the cell
    positions, below a nanometre).

    Parameters
    ----------
    catalog
        Tiles with ground classified.
    out
        Directory for the normalised tiles.
    dtm : "auto" or Raster
        Make the DTM per tile, or use this one (from :func:`dtm`, say).
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    replace_z
        Replace z by the height (keeping the elevation in ``elevation``)
        rather than adding a ``height`` attribute.
    buffer
        Band read around each tile (m); wider than the widest gap in the
        ground for ``"auto"``.
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.

    Raises
    ------
    ValueError
        For a bad ``dtm``, or a tile with fewer than 3 ground points within
        its buffer with ``"auto"``.
    """
    _record(None)
    return als.normalize(catalog, out, dtm=dtm, dtm_resolution=dtm_resolution,
                         replace_z=replace_z, chunk_size=None, buffer=_buffer(buffer),
                         workers=workers, format=format)


def detect_stems(catalog: Catalog, height_attr: str = "height", buffer: float = 2.0,
                 workers: int | None = None, **params) -> list[Tree]:
    """Detect stems over all the tiles, as in the whole cloud.

    :func:`sylva.trees.detect_stems` runs on each tile with its buffer, and
    each tile keeps the stems whose position (at the reference height) lies
    in it, or nearest to it. RANSAC draws its samples from a random stream
    per cluster (``cluster_seeds=True``), so a stem depends only on the
    points around it: with a buffer wider than the clusters and circles of
    the stems at a tile edge (``max_cluster_extent``, 2 m, is enough), the
    stems are those ``trees.detect_stems(cat.read(), cluster_seeds=True,
    **params)`` finds, with the same ids.

    Parameters
    ----------
    catalog
        Height-normalised tiles (:func:`normalize`).
    height_attr
        Attribute holding heights; z is used if the tiles do not have it.
    buffer
        Band read around each tile (m).
    workers
        Tiles processed at once; see :func:`voxel_downsample`.
    **params
        Keyword parameters of :func:`sylva.trees.detect_stems`.

    Returns
    -------
    list of Tree
        Sorted by ``quality`` with ids 1..n.

    Raises
    ------
    ValueError
        For an unknown parameter or a negative buffer.
    """
    cat = _as_catalog(catalog)
    found, info = _core.tiles_detect_stems(cat._core(), str(height_attr), _buffer(buffer),
                                           _workers(workers), params)
    _record(info)
    return [Tree._from_core(t) for t in found]


def _add_commands(sub, fmt: dict) -> None:
    """The ``tiles-*`` commands (see :mod:`sylva.cli`)."""
    def common(s, buffer: float | None = 1.0):
        s.add_argument("--pattern", default="*.la[sz]", help="file name pattern within the directory")
        if buffer is not None:
            s.add_argument("--buffer", type=float, default=buffer,
                           help="band of neighbouring points read around each tile (m)")
        s.add_argument("--workers", type=int, default=None,
                       help="tiles at once (default: one per CPU, fewer if memory is short)")

    def report(out: Catalog, what: str, dest: str) -> None:
        info = last_run()
        extra = f"; at most {info.max_points:,} points held at once" if info else ""
        print(f"{what} {out.n_points:,} points in {len(out)} tiles -> {dest}{extra}")

    def from_scans_cmd(a):
        transforms = None
        if a.transforms:
            m = np.load(a.transforms)
            if m.ndim != 3 or m.shape[1:] != (4, 4) or len(m) != len(a.scans):
                raise ValueError(f"{a.transforms} must hold one 4x4 matrix per scan, "
                                 f"got shape {m.shape} for {len(a.scans)} scans")
            transforms = list(m)
        out = from_scans(a.scans, a.output, tile_size=a.tile_size, voxel_size=a.voxel,
                         transforms=transforms, bounds=a.bounds, format=a.format, scale=a.scale,
                         epsg=a.epsg, workers=a.workers)
        report(out, "tiled", a.output)

    s = sub.add_parser("tiles-from-scans", help="square tiles from scans, thinned on one global "
                       "voxel grid", **fmt)
    s.add_argument("scans", nargs="+", help="scan files, in scan order")
    s.add_argument("output", help="directory for the tiles")
    s.add_argument("--tile-size", type=float, default=10.0, help="tile side (m)")
    s.add_argument("--voxel", type=float, default=None,
                   help="keep the first point per voxel of this size (m) over all scans")
    s.add_argument("--transforms", help=".npy of shape (n_scans, 4, 4) applied to the scans")
    s.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="keep only the points inside this box")
    s.add_argument("--format", choices=["las", "laz"], default="laz", help="tile format")
    s.add_argument("--scale", type=float, default=0.001, help="coordinate quantisation (m)")
    s.add_argument("--epsg", type=int, default=None, help="EPSG code recorded in the tiles")
    s.add_argument("--workers", type=int, default=None, help="tiles assembled at once")
    s.set_defaults(func=from_scans_cmd)

    def thin_cmd(a):
        out = voxel_downsample(catalog(a.input, pattern=a.pattern), a.output, a.voxel,
                               workers=a.workers)
        report(out, "thinned to", a.output)

    s = sub.add_parser("tiles-thin", help="thin tiles to one point per voxel of a global grid",
                       **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("output", help="directory for the thinned tiles")
    s.add_argument("--voxel", type=float, required=True, help="voxel size (m)")
    common(s, buffer=None)
    s.set_defaults(func=thin_cmd)

    def filter_cmd(a):
        cat = catalog(a.input, pattern=a.pattern)
        if a.method == "sor":
            out = statistical_outlier_removal(cat, a.output, k=a.k, std_ratio=a.std_ratio,
                                              classify=a.classify, buffer=a.buffer,
                                              workers=a.workers)
        else:
            out = radius_outlier_removal(cat, a.output, a.radius, min_neighbors=a.min_neighbors,
                                         classify=a.classify, buffer=a.buffer, workers=a.workers)
        report(out, "kept" if not a.classify else "classified", a.output)

    s = sub.add_parser("tiles-filter", help="noise filter over tiles, as on the whole cloud",
                       **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("output", help="directory for the filtered tiles")
    s.add_argument("--method", choices=["sor", "ror"], default="sor",
                   help="statistical (CloudCompare SOR) or radius outlier removal")
    s.add_argument("--k", type=int, default=8, help="SOR neighbours")
    s.add_argument("--std-ratio", type=float, default=2.0, help="SOR threshold (standard deviations)")
    s.add_argument("--radius", type=float, default=0.1, help="ROR search radius (m)")
    s.add_argument("--min-neighbors", type=int, default=4, help="ROR neighbours needed")
    s.add_argument("--classify", action="store_true",
                   help="classify noise as 7 rather than removing it")
    common(s)
    s.set_defaults(func=filter_cmd)

    def features_cmd(a):
        cat = catalog(a.input, pattern=a.pattern)
        fn = estimate_normals if a.feature == "normals" else planarity_linearity
        k = a.k if a.k is not None else (12 if a.feature == "normals" else 20)
        out = fn(cat, a.output, k=k, buffer=a.buffer, workers=a.workers)
        report(out, f"{a.feature} for", a.output)

    s = sub.add_parser("tiles-features", help="normals or planarity and linearity for every point "
                       "of the tiles", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("output", help="directory for the tiles with the features")
    s.add_argument("--feature", choices=["normals", "shape"], default="normals",
                   help="normal_x/y/z, or planarity and linearity")
    s.add_argument("--k", type=int, default=None, help="neighbours per point (12 or 20)")
    common(s)
    s.set_defaults(func=features_cmd)

    def stems_cmd(a):
        cat = catalog(a.input, pattern=a.pattern)
        found = detect_stems(cat, height_attr=a.height_attribute, buffer=a.buffer,
                             workers=a.workers, min_arc_deg=a.min_arc)
        import pandas as pd
        pd.DataFrame([t.as_dict() for t in found]).to_csv(a.output, index=False)
        print(f"{len(found):,} stems from {len(cat)} tiles -> {a.output}")

    s = sub.add_parser("tiles-stems", help="stems and DBH over height-normalised tiles", **fmt)
    s.add_argument("input", help="directory of height-normalised LAS/LAZ tiles")
    s.add_argument("output", help="CSV of the stems")
    s.add_argument("--height-attribute", default="height",
                   help="attribute holding heights (z if the tiles lack it)")
    s.add_argument("--min-arc", type=float, default=0.0,
                   help="longest contiguous arc a circle must cover (degrees; 130 suits merged plots)")
    common(s, buffer=2.0)
    s.set_defaults(func=stems_cmd)
