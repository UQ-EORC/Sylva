# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Boolean masks over the points of a cloud: polygons, rasters, attribute
expressions and distance to another cloud.

Every mask is a boolean array with one entry per point, ``True`` where the
point is kept. Masks combine with ``&``, ``|`` and ``~`` and index a cloud
directly (``cloud[mask]``). Each mask function has a ``crop_*`` companion that
returns the kept points in one call, with ``invert=True`` to keep the others.

Polygons and rasters are taken in the cloud's coordinate frame. Nothing here
reprojects: if a layer is stored in another CRS, reproject the layer or the
cloud first.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster

__all__ = [
    "Polygon", "MultiPolygon", "Polygons", "read_polygons", "polygon_index", "inside_polygons",
    "crop_polygons", "raster_mask", "crop_raster", "expression", "crop_expression", "near",
    "crop_near", "difference",
]


def _ring(values, what: str) -> np.ndarray:
    ring = np.asarray(values, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1] < 2:
        raise ValueError(f"{what} must be an (K, 2) array of x, y, got shape {ring.shape}")
    return np.ascontiguousarray(ring[:, :2])


@dataclass
class Polygon:
    """A polygon: an exterior ring and optional holes, in x, y.

    Rings are ``(K, 2)`` arrays of vertices in either orientation, open or
    closed (a repeated first vertex is ignored). Extra columns, such as z,
    are dropped.

    Parameters
    ----------
    exterior
        ``(K, 2)`` vertices of the outer boundary.
    holes
        ``(K, 2)`` vertices of each hole.

    Raises
    ------
    ValueError
        If a ring is not a 2-D array with at least two columns.
    """

    exterior: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.exterior = _ring(self.exterior, "exterior")
        self.holes = [_ring(h, f"hole {i}") for i, h in enumerate(self.holes)]


@dataclass
class MultiPolygon:
    """One feature of a polygon layer: polygons treated as one shape.

    A point is inside a multipolygon if it is inside any of its parts. A
    feature with no parts (a null geometry in the file) contains nothing.

    Parameters
    ----------
    parts
        The polygons.
    properties
        The feature's attributes (``.dbf`` fields or GeoJSON properties).
    """

    parts: list[Polygon] = field(default_factory=list)
    properties: dict = field(default_factory=dict)


@dataclass
class Polygons:
    """A polygon layer, as returned by :func:`read_polygons`.

    Behaves as a sequence of :class:`MultiPolygon`: ``len(layer)``,
    iteration, ``layer[i]`` for one feature, and ``layer[mask]`` or
    ``layer[[0, 3]]`` for a sub-layer, e.g.
    ``layer[[f.properties["plot"] == 7 for f in layer]]``.

    Parameters
    ----------
    features
        The features, in file order.
    crs
        The layer's CRS as stored in the file: the WKT of a shapefile's
        ``.prj``, or the name in a GeoJSON ``crs`` member (for example
        ``"urn:ogc:def:crs:EPSG::28355"``). It is informative only;
        coordinates are never reprojected.
    """

    features: list[MultiPolygon] = field(default_factory=list)
    crs: str | None = None

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[MultiPolygon]:
        return iter(self.features)

    def __getitem__(self, index) -> MultiPolygon | Polygons:
        if isinstance(index, (int, np.integer)):
            return self.features[index]
        if isinstance(index, slice):
            return Polygons(self.features[index], self.crs)
        idx = np.asarray(index)
        if idx.dtype == bool:
            if idx.shape != (len(self),):
                raise ValueError(f"boolean index has shape {idx.shape}, expected ({len(self)},)")
            idx = np.flatnonzero(idx)
        return Polygons([self.features[int(i)] for i in idx], self.crs)

    @property
    def properties(self) -> list[dict]:
        """The attributes of every feature, in order."""
        return [f.properties for f in self.features]


def read_polygons(path: str | Path, layer: str | None = None) -> Polygons:
    """Read the polygons of a shapefile or a GeoJSON file.

    Parameters
    ----------
    path
        A ``.shp`` file (its ``.dbf`` attributes and ``.prj`` CRS are read
        when present), a ``.geojson`` or ``.json`` file, or a directory
        holding such files.
    layer
        For a directory, the name of the file to read, without extension
        (``"plots"`` for ``plots.shp``). It may be omitted when the directory
        holds a single layer. For a file it must be None or the file's own
        name.

    Returns
    -------
    Polygons
        One :class:`MultiPolygon` per record, with holes and multiple parts
        as stored. A shapefile's hole is attached to the exterior ring that
        contains it. Null geometries give features with no parts.
        Coordinates are as stored in the file, in its CRS.

    Raises
    ------
    OSError
        If the file is missing or cannot be parsed.
    ValueError
        If the file holds points or lines rather than polygons, or its
        format is not supported.

    Notes
    -----
    No reprojection is done: the polygons must be in the same frame as the
    clouds they will mask. :attr:`Polygons.crs` reports the file's CRS so
    that it can be checked.
    """
    d = _core.read_polygons(str(path), layer)
    coords, rings, parts = d["coords"], d["ring_start"], d["part_start"]
    features = []
    for f, props in enumerate(d["properties"]):
        polys = []
        for p in range(d["feature_start"][f], d["feature_start"][f + 1]):
            r = [coords[rings[i]:rings[i + 1]] for i in range(parts[p], parts[p + 1])]
            polys.append(Polygon(r[0], r[1:]))
        features.append(MultiPolygon(polys, props))
    return Polygons(features, d["crs"])


def _as_features(polygons) -> list[list[Polygon]]:
    """Normalise every accepted polygon input to a list of features."""
    if isinstance(polygons, Polygons):
        return [f.parts for f in polygons.features]
    if isinstance(polygons, MultiPolygon):
        return [polygons.parts]
    if isinstance(polygons, Polygon):
        return [[polygons]]
    if isinstance(polygons, np.ndarray) or not isinstance(polygons, Iterable):
        arr = np.asarray(polygons, dtype=np.float64)
        if arr.ndim == 2:
            return [[Polygon(arr)]]
        if arr.ndim == 3:
            return [[Polygon(a)] for a in arr]
        raise ValueError(f"polygon coordinates must be (K, 2) arrays, got shape {arr.shape}")
    items = list(polygons)
    # A list of vertex pairs is one ring; a list of rings or polygons is many features.
    if items and not isinstance(items[0], (Polygon, MultiPolygon, Polygons)):
        try:
            arr = np.asarray(items, dtype=np.float64)
        except (ValueError, TypeError):
            arr = None
        if arr is not None and arr.ndim == 2:
            return [[Polygon(arr)]]
    out = []
    for i, item in enumerate(items):
        if isinstance(item, Polygons):
            out.extend(f.parts for f in item.features)
        elif isinstance(item, MultiPolygon):
            out.append(item.parts)
        elif isinstance(item, Polygon):
            out.append([item])
        else:
            out.append([Polygon(_ring(item, f"polygon {i}"))])
    return out


def _flatten(features: list[list[Polygon]]):
    coords, ring_start, part_start, feature_start = [], [0], [0], [0]
    n = 0
    for parts in features:
        for p in parts:
            for r in [p.exterior, *p.holes]:
                coords.append(r)
                n += len(r)
                ring_start.append(n)
            part_start.append(len(ring_start) - 1)
        feature_start.append(len(part_start) - 1)
    xy = np.ascontiguousarray(np.concatenate(coords)) if coords else np.zeros((0, 2))
    as_i64 = lambda v: np.asarray(v, dtype=np.int64)  # noqa: E731
    return xy, as_i64(ring_start), as_i64(part_start), as_i64(feature_start)


def polygon_index(cloud: PointCloud, polygons) -> np.ndarray:
    """Which polygon each point falls in.

    Parameters
    ----------
    cloud
        Input points; only x and y are used.
    polygons
        A :class:`Polygons` layer, a :class:`MultiPolygon`, a
        :class:`Polygon`, a ``(K, 2)`` array of vertices (one polygon
        without holes), or a list of any of these (one feature each).

    Returns
    -------
    numpy.ndarray
        int64, one per point: the index of the first feature containing the
        point, or -1. Useful to label points with a plot or stand number:
        ``layer[i].properties``.

    Raises
    ------
    ValueError
        If a ring has fewer than three distinct vertices or a non-finite
        coordinate.

    Notes
    -----
    Polygons are closed sets: a point on an edge or vertex is inside, a
    point on the boundary of a hole is inside, and only the open interior
    of a hole is excluded. The test uses exact orientation predicates, so
    points on edges are classified consistently whatever the edge's slope,
    and a point on an edge shared by two polygons belongs to both (it is
    reported for the first). Points with a non-finite x or y are outside.
    """
    xy, rings, parts, feats = _flatten(_as_features(polygons))
    return _core.mask_polygon_index(cloud.xyz, xy, rings, parts, feats)


def inside_polygons(cloud: PointCloud, polygons) -> np.ndarray:
    """Mask of the points whose x, y fall inside any of the polygons.

    Parameters
    ----------
    cloud
        Input points; only x and y are used.
    polygons
        As for :func:`polygon_index`: a layer from :func:`read_polygons`, a
        :class:`Polygon` or :class:`MultiPolygon`, a ``(K, 2)`` vertex array,
        or a list of these.

    Returns
    -------
    numpy.ndarray
        Boolean, one per point. Edges and vertices count as inside (see
        :func:`polygon_index`).

    Raises
    ------
    ValueError
        If a ring has fewer than three distinct vertices or a non-finite
        coordinate.

    Notes
    -----
    Polygons are taken in the cloud's frame. There is no reprojection, even
    when the layer's ``crs`` differs from the cloud's; reproject one of them
    first.

    Examples
    --------
    >>> from sylva.geo import masks
    >>> plots = masks.read_polygons("plots.shp")           # doctest: +SKIP
    >>> keep = masks.inside_polygons(cloud, plots) & (cloud.z > 0)   # doctest: +SKIP
    """
    return polygon_index(cloud, polygons) >= 0


def crop_polygons(cloud: PointCloud, polygons, invert: bool = False) -> PointCloud:
    """Keep the points inside (or outside) polygons.

    Parameters
    ----------
    cloud
        Input points.
    polygons
        As for :func:`inside_polygons`.
    invert
        Keep the points outside every polygon instead.

    Returns
    -------
    PointCloud
        The kept points with all attributes, in their original order.
    """
    mask = inside_polygons(cloud, polygons)
    return cloud[~mask if invert else mask]


def raster_mask(cloud: PointCloud, raster: Raster, min: float | None = None,
                max: float | None = None, values: Iterable[float] | None = None) -> np.ndarray:
    """Mask of the points whose raster cell satisfies a condition.

    Parameters
    ----------
    cloud
        Input points; only x and y are used.
    raster
        Grid in the cloud's frame, e.g. a CHM, a DTM or a classification.
    min, max
        Keep points whose cell value lies in ``[min, max]`` (bounds
        inclusive; either may be omitted).
    values
        Keep points whose cell value equals one of these (compared as
        float64), e.g. land-cover classes. Give either a range or values.
        With neither, every point over a valid (non-NaN) cell is kept.

    Returns
    -------
    numpy.ndarray
        Boolean, one per point. Points outside the raster and points over a
        NaN cell are False.

    Raises
    ------
    ValueError
        If both a range and ``values`` are given, ``min > max``, a bound is
        NaN, or the raster's resolution is not positive.

    Notes
    -----
    A point belongs to the cell ``floor((x - xmin) / resolution)``,
    ``floor((y - ymin) / resolution)``: cells include their southern and
    western edges, so a point exactly on the raster's northern or eastern
    edge is outside.
    """
    if values is not None and (min is not None or max is not None):
        raise ValueError("give min and/or max, or values, not both")
    vals = None
    if values is not None:
        vals = [float(v) for v in np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()]
    lo = None if min is None else float(min)
    hi = None if max is None else float(max)
    return _core.mask_raster(cloud.xyz, raster.data, float(raster.xmin), float(raster.ymin),
                             float(raster.resolution), lo, hi, vals)


def crop_raster(cloud: PointCloud, raster: Raster, min: float | None = None,
                max: float | None = None, values: Iterable[float] | None = None,
                invert: bool = False) -> PointCloud:
    """Keep the points whose raster cell satisfies a condition.

    Parameters
    ----------
    cloud
        Input points.
    raster, min, max, values
        As for :func:`raster_mask`.
    invert
        Keep the other points instead, including those outside the raster
        and over NaN cells.

    Returns
    -------
    PointCloud
        The kept points with all attributes, in their original order.
    """
    mask = raster_mask(cloud, raster, min=min, max=max, values=values)
    return cloud[~mask if invert else mask]


def expression(cloud: PointCloud, expr: str) -> np.ndarray:
    """Mask of the points satisfying an expression over their attributes.

    Parameters
    ----------
    cloud
        Input points.
    expr
        A condition such as ``"height > 2 & classification != 2"``. It may
        use:

        - the coordinates ``x``, ``y``, ``z`` and any attribute of the cloud
          by name (the coordinates hide attributes named x, y or z);
        - numbers (``2``, ``-0.5``, ``1e3``) and ``true``, ``false``;
        - arithmetic ``+ - * /`` and unary minus;
        - comparisons ``< <= > >= == !=``, which may be chained
          (``1.3 <= z < 40``);
        - membership: ``classification in (3, 4, 5)`` and ``not in``;
        - logic: ``&`` (or ``and``), ``|`` (or ``or``), ``!`` (or ``not``),
          and parentheses.

        ``&`` and ``|`` bind more loosely than comparisons, so no
        parentheses are needed around them (unlike NumPy). The expression is
        parsed and evaluated by the Rust core; it is never run as Python.

    Returns
    -------
    numpy.ndarray
        Boolean, one per point.

    Raises
    ------
    ValueError
        On a syntax error (the message gives the position and marks it), an
        unknown attribute (the message lists the cloud's attributes), or a
        number used where a condition is needed (``height & z > 1``).
    TypeError
        If ``expr`` is not a string.

    Notes
    -----
    Values are compared as float64, so integers beyond 2**53 lose
    precision. Comparisons follow IEEE rules: any comparison with NaN is
    False except ``!=``. Boolean attributes are conditions in their own
    right (``withheld & z > 1``) and count as 0 or 1 in arithmetic.

    Examples
    --------
    >>> from sylva.geo import masks
    >>> keep = masks.expression(cloud, "height > 2 & classification != 2")  # doctest: +SKIP
    >>> canopy = cloud.where("height > 2 & classification != 2")            # doctest: +SKIP
    """
    if not isinstance(expr, str):
        raise TypeError(f"expr must be a string, got {type(expr).__name__}")
    return _core.mask_expression(expr, cloud.xyz, cloud.attrs)


def crop_expression(cloud: PointCloud, expr: str, invert: bool = False) -> PointCloud:
    """Keep the points satisfying an attribute expression.

    Parameters
    ----------
    cloud
        Input points.
    expr
        Condition, as for :func:`expression`.
    invert
        Keep the points that do not satisfy it instead.

    Returns
    -------
    PointCloud
        The kept points with all attributes, in their original order.
        :meth:`sylva.PointCloud.where` is the same without ``invert``.
    """
    mask = expression(cloud, expr)
    return cloud[~mask if invert else mask]


def _xyz(other) -> np.ndarray:
    if isinstance(other, PointCloud):
        return other.xyz
    xyz = np.ascontiguousarray(np.asarray(other, dtype=np.float64))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"other must be a PointCloud or an (N, 3) array, got shape {xyz.shape}")
    return xyz


def near(cloud: PointCloud, other, distance: float, horizontal: bool = False) -> np.ndarray:
    """Mask of the points within a distance of another cloud.

    Parameters
    ----------
    cloud
        Input points.
    other
        Reference points: a :class:`~sylva.PointCloud` or an ``(M, 3)``
        array.
    distance
        Maximum distance to the nearest point of ``other`` (m, inclusive).
    horizontal
        Measure the distance in x, y only, ignoring z (for example to keep
        everything above or below a set of stem points).

    Returns
    -------
    numpy.ndarray
        Boolean, one per point of ``cloud``. With an empty ``other``, all
        False.

    Raises
    ------
    ValueError
        If ``distance`` is negative or not finite.

    Notes
    -----
    A k-d tree is built on ``other`` and queried for every point in
    parallel; the answer for each point is its exact nearest distance, so
    the mask does not depend on the number of threads. Points of ``other``
    with a non-finite coordinate are ignored, and points of ``cloud`` with
    one are never near. Building the tree takes memory of the order of
    ``other.xyz``; tens of millions of points on both sides are fine.
    """
    return _core.mask_near(cloud.xyz, _xyz(other), float(distance), bool(horizontal))


def crop_near(cloud: PointCloud, other, distance: float, horizontal: bool = False,
              invert: bool = False) -> PointCloud:
    """Keep the points within a distance of another cloud.

    Parameters
    ----------
    cloud
        Input points.
    other, distance, horizontal
        As for :func:`near`.
    invert
        Keep the points farther than ``distance`` instead (as
        :func:`difference`).

    Returns
    -------
    PointCloud
        The kept points with all attributes, in their original order.
    """
    mask = near(cloud, other, distance, horizontal)
    return cloud[~mask if invert else mask]


def difference(cloud: PointCloud, other, distance: float, horizontal: bool = False) -> PointCloud:
    """Points of ``cloud`` farther than ``distance`` from every point of ``other``.

    The points that ``other`` does not explain: for change detection between
    two epochs (``difference(after, before, 0.05)`` is what appeared,
    ``difference(before, after, 0.05)`` what was lost), or to remove an
    already-segmented object from a scene.

    Parameters
    ----------
    cloud
        Input points.
    other
        Reference points: a :class:`~sylva.PointCloud` or an ``(M, 3)``
        array.
    distance
        Points at more than this distance from their nearest neighbour in
        ``other`` are kept (m).
    horizontal
        Measure the distance in x, y only.

    Returns
    -------
    PointCloud
        The kept points with all attributes; the same as
        ``cloud[~near(cloud, other, distance)]``, so points with a
        non-finite coordinate are kept.

    Raises
    ------
    ValueError
        If ``distance`` is negative or not finite.
    """
    return crop_near(cloud, other, distance, horizontal, invert=True)
