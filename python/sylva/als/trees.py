# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Individual trees from airborne lidar: tree tops, crowns and labelled points.

The functions are re-exported from :mod:`sylva.als` and follow lidR's
algorithms (Roussel et al. 2020), whose implementations are reproduced
where the papers leave a detail open:

- :func:`locate_trees`: tree tops by the local maximum filter (``lmf``;
  Popescu and Wynne 2004) on a canopy height model or on points, with a
  fixed window or one that grows with height.
- :func:`segment_crowns`: crowns on a CHM from tree tops, by
  marker-controlled watershed (Meyer and Beucher 1990) or by the region
  growing of Dalponte and Coomes (2016).
- :func:`li2012`: the point-based segmentation of Li et al. (2012).
- :func:`segment_trees`: a whole segmentation of one cloud, giving each
  point its tree and each tree its top, crown outline and area.
- :func:`find_trees`: the same over a catalogue of tiles, in buffered
  chunks, with labelled tiles written back.

References
----------
Dalponte, M. and Coomes, D. A. (2016). Tree-centric mapping of forest carbon
density from airborne laser scanning and hyperspectral data. Methods in
Ecology and Evolution 7, 1236-1245.

Duckham, M., Kulik, L., Worboys, M. and Galton, A. (2008). Efficient
generation of simple polygons for characterizing the shape of a set of
points in the plane. Pattern Recognition 41(10), 3224-3236.

Li, W., Guo, Q., Jakubowski, M. K. and Kelly, M. (2012). A new method for
segmenting individual trees from the lidar point cloud. Photogrammetric
Engineering & Remote Sensing 78(1), 75-84.

Meyer, F. and Beucher, S. (1990). Morphological segmentation. Journal of
Visual Communication and Image Representation 1(1), 21-46.

Popescu, S. C. and Wynne, R. H. (2004). Seeing the trees in the forest:
using lidar and multispectral data fusion with local filtering and variable
window size for estimating tree height. Photogrammetric Engineering &
Remote Sensing 70(5), 589-604.

Roussel, J.-R. et al. (2020). lidR: an R package for analysis of airborne
laser scanning (ALS) data. Remote Sensing of Environment 251, 112061.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from . import Catalog, _as_catalog, _format, _heights, _run_kw, _written
from ..pointcloud import PointCloud
from ..raster import Raster

__all__ = ["LinearWindow", "TreeTops", "Trees", "locate_trees", "segment_crowns", "li2012",
           "crown_hull", "segment_trees", "find_trees"]

_METHODS = ("dalponte2016", "watershed", "li2012", "tops")
_TABLE_STEP = 0.01


@dataclass(frozen=True)
class LinearWindow:
    """A window size that grows linearly with height:
    ``clip(intercept + slope * h, min, max)`` metres for a site ``h`` m high.

    Calling it on heights gives the sizes, so it can stand wherever a
    window function is taken; the Rust core evaluates it directly, which
    keeps catalogue runs exact.

    Parameters
    ----------
    intercept, slope
        Size (m) at height 0 and its increase per metre of height.
    min, max
        Bounds on the size (m); ``0 < min <= max``.

    Examples
    --------
    lidR's documentation uses ``function(x) {x * 0.1 + 3}``; bounded:

    >>> w = LinearWindow(3.0, 0.1, min=3.0, max=8.0)
    >>> float(w(20.0))
    5.0
    """

    intercept: float
    slope: float
    min: float = 1.0
    max: float = 30.0

    def __call__(self, h):
        h = np.asarray(h, dtype=np.float64)
        return np.clip(self.intercept + self.slope * h, self.min, self.max)


def _window(window, hmax: float) -> tuple[str, list[float]]:
    """``(kind, values)`` for the core: a fixed size, a linear window, or a
    callable sampled every centimetre of height up to ``hmax``."""
    if isinstance(window, LinearWindow):
        w = window
        vals = [float(w.intercept), float(w.slope), float(w.min), float(w.max)]
        if not (np.all(np.isfinite(vals)) and 0 < w.min <= w.max):
            raise ValueError(f"a LinearWindow needs finite values and 0 < min <= max, got {w}")
        return "linear", vals
    if callable(window):
        top = max(float(hmax), 1.0) + 1.0
        step = max(_TABLE_STEP, top / 1_000_000)
        h = np.arange(0.0, top + step, step)
        ws = np.asarray(window(h), dtype=np.float64)
        if ws.shape == ():
            ws = np.full(h.shape, float(ws))
        if ws.shape != h.shape:
            raise ValueError("the window function must return one size per height, got shape "
                             f"{ws.shape}")
        if not (np.all(np.isfinite(ws)) and np.all(ws > 0)):
            raise ValueError("the window function must give positive, finite sizes (m) for heights "
                             f"from 0 to {top:.1f} m")
        return "table", [step, *ws.tolist()]
    try:
        ws = float(window)
    except (TypeError, ValueError):
        raise ValueError("window must be a number, a LinearWindow or a function of height, "
                         f"got {window!r}") from None
    if not (np.isfinite(ws) and ws > 0):
        raise ValueError(f"the window size must be a positive number of metres, got {window}")
    return "fixed", [ws]


def _shape(shape: str) -> str:
    if shape not in ("circular", "square"):
        raise ValueError(f"unknown window shape {shape!r}; expected 'circular' or 'square'")
    return shape


def _point_heights(cloud: PointCloud, heights) -> np.ndarray:
    if heights is None:
        h = cloud.z
    elif isinstance(heights, str):
        if heights not in cloud.attrs:
            raise ValueError(f"the cloud has no attribute {heights!r}")
        h = cloud.attrs[heights]
    else:
        h = heights
    h = np.ascontiguousarray(np.asarray(h, dtype=np.float64))
    if h.shape != (len(cloud),):
        raise ValueError(f"heights must have one value per point ({len(cloud)}), "
                         f"got shape {h.shape}")
    return h


@dataclass
class TreeTops:
    """Tree tops from :func:`locate_trees`.

    Parameters
    ----------
    x, y
        Position of each top (m): a CHM cell centre or a point.
    height
        The CHM value or the point's height (m).
    index
        Row-major cell index in the CHM, or the point's index in the cloud.
    """

    x: np.ndarray
    y: np.ndarray
    height: np.ndarray
    index: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    @property
    def xyz(self) -> np.ndarray:
        """``(k, 3)`` array of ``x, y, height``."""
        return np.column_stack([self.x, self.y, self.height]).reshape(-1, 3)


def locate_trees(source, window=5.0, hmin: float = 2.0, shape: str = "circular",
                 heights=None) -> TreeTops:
    """Tree tops by the local maximum filter (lidR's ``locate_trees(...,
    lmf(ws, hmin, shape))``; Popescu and Wynne 2004).

    A CHM cell (its centre, with the cell's value) or a point is a tree top
    when it is at least ``hmin`` high and nothing within its window is
    higher. The window is a disc of diameter ``ws`` (or a square of side
    ``ws``) centred on the site, and ``ws`` may depend on the site's height.
    Of equal-height maxima within each other's windows only the first, in
    order of x then y, is kept (lidR keeps whichever it tags first).

    Parameters
    ----------
    source : Raster or PointCloud
        A canopy height model (NaN cells take no part), or a cloud of
        heights above ground.
    window : float, LinearWindow or callable
        Window size ``ws`` (m): a number; a :class:`LinearWindow`; or a
        function taking an array of heights and returning the sizes, e.g.
        ``lambda h: 0.1 * h + 3`` (evaluated at every site).
    hmin
        Lowest tree top (m).
    shape : {"circular", "square"}
        Window shape.
    heights
        For a cloud: the height of each point, as an array or the name of
        an attribute; z if None (a normalised cloud).

    Returns
    -------
    TreeTops
        In order of x, then y.

    Raises
    ------
    ValueError
        For a non-positive or non-finite window size, an unknown shape, or
        heights that do not match the cloud.
    """
    _shape(shape)
    if not np.isfinite(hmin):
        raise ValueError(f"hmin must be a number, got {hmin}")
    if isinstance(source, Raster):
        data = np.ascontiguousarray(source.data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("the CHM must be a 2-D raster")
        if callable(window) and not isinstance(window, LinearWindow):
            kind, vals = "site", _site_window(window, data.ravel())
        else:
            kind, vals = _window(window, 0.0)
        idx = _core.als_trees_lmf_raster(data, float(source.xmin), float(source.ymin),
                                         float(source.resolution), kind, vals, float(hmin), shape)
        rows, cols = np.divmod(idx, data.shape[1])
        x = source.xmin + (cols + 0.5) * source.resolution
        y = source.ymin + (rows + 0.5) * source.resolution
        return TreeTops(x.astype(np.float64), y.astype(np.float64), data.ravel()[idx], idx)
    if isinstance(source, PointCloud):
        h = _point_heights(source, heights)
        if callable(window) and not isinstance(window, LinearWindow):
            kind, vals = "site", _site_window(window, h)
        else:
            kind, vals = _window(window, 0.0)
        idx = _core.als_trees_lmf_points(np.ascontiguousarray(source.xyz), h, kind, vals,
                                         float(hmin), shape)
        return TreeTops(source.x[idx].copy(), source.y[idx].copy(), h[idx], idx)
    raise ValueError(f"source must be a Raster (CHM) or a PointCloud, got {type(source).__name__}")


def _site_window(fn: Callable, h: np.ndarray) -> list[float]:
    ws = np.asarray(fn(h), dtype=np.float64)
    if ws.shape == ():
        ws = np.full(h.shape, float(ws))
    if ws.shape != h.shape:
        raise ValueError("the window function must return one size per height, got shape "
                         f"{ws.shape}")
    ok = np.isfinite(h)
    if not (np.all(np.isfinite(ws[ok])) and np.all(ws[ok] > 0)):
        raise ValueError("the window function must give positive, finite sizes (m)")
    return ws.tolist()


def _tops_array(tops) -> np.ndarray:
    if isinstance(tops, TreeTops):
        return np.ascontiguousarray(tops.xyz, dtype=np.float64)
    t = np.asarray(tops, dtype=np.float64)
    if t.size == 0:
        return np.zeros((0, 3))
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("tops must be TreeTops or an array of (x, y, height), "
                         f"got shape {t.shape}")
    return np.ascontiguousarray(t)


def segment_crowns(chm: Raster, tops, method: str = "dalponte2016", th_tree: float = 2.0,
                   th_seed: float = 0.45, th_cr: float = 0.55, max_cr: float = 10.0) -> Raster:
    """Crowns on a canopy height model, grown from tree tops.

    ``"dalponte2016"`` is the region growing of Dalponte and Coomes (2016)
    as lidR implements it (``dalponte2016(chm, ttops, th_tree, th_seed,
    th_cr, max_cr)``): each crown adds, sweep after sweep, the 4-neighbours
    of its cells that are higher than ``th_tree``, higher than ``th_seed``
    times the top's CHM value and than ``th_cr`` times the crown's mean
    height, at most 5 % above the top, and fewer than ``max_cr`` cells from
    it in x and in y. ``"watershed"`` is a marker-controlled watershed
    (Meyer and Beucher 1990): the CHM is flooded from the tops, highest
    cells first, over 8-connected cells higher than ``th_tree``, each cell
    taking the crown that reaches it first; tops on cells not higher than
    ``th_tree`` grow nothing.

    Parameters
    ----------
    chm
        Canopy height model.
    tops : TreeTops or array (k, 3)
        Tree tops ``(x, y, height)``, e.g. from :func:`locate_trees`. A top
        seeds the cell it falls in; of several in one cell the highest is
        kept.
    method : {"dalponte2016", "watershed"}
        How crowns grow.
    th_tree
        Cells not higher than this (m) are in no crown.
    th_seed, th_cr
        Growing thresholds of Dalponte and Coomes (0-1).
    max_cr
        Largest crown extent, in cells from the top in x and in y
        (``"dalponte2016"``; lidR's ``max_cr``).

    Returns
    -------
    Raster
        The crown of each cell: ``k + 1`` for ``tops[k]``, NaN for none.

    Raises
    ------
    ValueError
        For an unknown method, thresholds outside 0-1, or a non-positive
        ``max_cr``.
    """
    if method not in ("dalponte2016", "watershed"):
        raise ValueError(f"unknown method {method!r}; expected 'dalponte2016' or 'watershed'")
    data = np.ascontiguousarray(chm.data, dtype=np.float64)
    t = _tops_array(tops)
    lab = _core.als_trees_crowns(data, float(chm.xmin), float(chm.ymin), float(chm.resolution), t,
                                 method, float(th_tree), float(th_seed), float(th_cr),
                                 float(max_cr)).astype(np.float64)
    lab[lab == 0] = np.nan
    return Raster(lab, chm.xmin, chm.ymin, chm.resolution, chm.crs)


def li2012(cloud: PointCloud, dt1: float = 1.5, dt2: float = 2.0, R: float = 2.0,
           Zu: float = 15.0, hmin: float = 2.0, speed_up: float = 10.0,
           heights=None) -> np.ndarray:
    """Point-based tree segmentation of Li et al. (2012), as lidR's
    ``li2012(dt1, dt2, R, Zu, hmin, speed_up)`` implements it.

    Points are taken from the highest down. The highest point left starts
    a tree (its set P, with an empty set N); every point left within
    ``speed_up`` of that top, from the highest down, joins P or N by its
    smallest horizontal distances ``d1`` to P and ``d2`` to N. A local
    maximum (highest within a disc of diameter ``R``) joins N if ``d1 >
    dt`` or ``d2 < d1 < dt``, and P otherwise, where ``dt`` is ``dt2`` for
    points higher than ``Zu`` and ``dt1`` below; any other point joins P
    if ``d1 <= d2``. P is the tree; N stays for the next ones. It stops
    when the highest point left is lower than ``hmin``. Equal heights are
    taken in order of x, then y.

    Parameters
    ----------
    cloud
        Points; see ``heights``.
    dt1, dt2
        Spacing thresholds (m) below and above ``Zu``.
    R
        Diameter (m) of the local maximum window (lidR passes ``R`` as its
        window size); 0 makes every point a local maximum.
    Zu
        Height (m) above which ``dt2`` applies.
    hmin
        Lowest tree top (m).
    speed_up
        Largest crown radius (m) considered; it only saves time when
        larger than any crown.
    heights
        Height of each point (array or attribute name); z if None.

    Returns
    -------
    numpy.ndarray
        Tree of each point, 1, 2, ... in the order found (tallest first),
        0 for none.

    Raises
    ------
    ValueError
        For non-positive thresholds or a negative ``R``.
    """
    h = _point_heights(cloud, heights)
    return _core.als_trees_li2012(np.ascontiguousarray(cloud.xyz), h, float(dt1), float(dt2),
                                  float(R), float(Zu), float(hmin), float(speed_up))


def crown_hull(xy, hull: str = "convex", concavity: float = 2.0) -> np.ndarray:
    """Outline of a crown's points seen from above.

    Parameters
    ----------
    xy
        ``(N, 2)`` or ``(N, 3)`` coordinates.
    hull : {"convex", "concave"}
        The convex hull, or the characteristic shape of Duckham et al.
        (2008): from the Delaunay triangulation, outline edges longer than
        ``concavity`` are removed, longest first, while the outline stays a
        simple polygon.
    concavity
        Edge length (m) of the concave outline; smaller follows the points
        more closely.

    Returns
    -------
    numpy.ndarray
        ``(k, 2)`` vertices, counter-clockwise, the first not repeated.
        Fewer than 3 distinct points (or collinear ones) give their convex
        hull.
    """
    a = np.asarray(xy, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] not in (2, 3):
        raise ValueError(f"xy must have shape (N, 2) or (N, 3), got {a.shape}")
    return _core.als_trees_hull(np.ascontiguousarray(a), str(hull), float(concavity))


@dataclass
class Trees:
    """Trees found by :func:`segment_trees` or :func:`find_trees`.

    Parameters
    ----------
    id
        1, 2, ... in order of the tops' x, then y.
    x, y, height
        The top: a CHM cell centre and its value, or the highest point of
        the tree and its height (m).
    crown_area
        Area (m²) of the crown outline; NaN without a crown.
    n_points
        Points labelled with the tree.
    crowns
        Crown outline of each tree, ``(k, 2)`` counter-clockwise, from its
        points.
    tree_id
        For :func:`segment_trees`, the tree of each point of the cloud (0
        for none).
    catalog
        For :func:`find_trees` with ``out``, the labelled tiles written.
    at_edge
        Crowns that reached the inner edge of their chunk's buffer; more
        than 0 means the buffer may be too narrow for them to be complete.
    unmatched_points
        Points written with id 0 because the tree their chunk gave them was
        found by no chunk as its own; 0 with an adequate buffer.
    crs
        Coordinate system of the positions.
    """

    id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    height: np.ndarray
    crown_area: np.ndarray
    n_points: np.ndarray
    crowns: list = field(default_factory=list)
    tree_id: np.ndarray | None = None
    catalog: Catalog | None = None
    at_edge: int = 0
    unmatched_points: int = 0
    crs: str | None = None

    def __len__(self) -> int:
        return len(self.id)

    def __repr__(self) -> str:
        return f"Trees({len(self)} trees)"

    @classmethod
    def _from_core(cls, d: dict, **kw) -> Trees:
        return cls(np.asarray(d["id"]), np.asarray(d["x"]), np.asarray(d["y"]),
                   np.asarray(d["height"]), np.asarray(d["crown_area"]),
                   np.asarray(d["n_points"]), list(d["crowns"]), **kw)

    def table(self) -> dict[str, np.ndarray]:
        """The per-tree columns: ``id``, ``x``, ``y``, ``height``,
        ``crown_area`` and ``n_points``."""
        return {"id": self.id, "x": self.x, "y": self.y, "height": self.height,
                "crown_area": self.crown_area, "n_points": self.n_points}

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as CSV, one row per tree.

        Parameters
        ----------
        path
            Output file.
        """
        t = self.table()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(t))
            for row in zip(*t.values(), strict=True):
                w.writerow([int(v) if isinstance(v, np.integer) else float(v) for v in row])

    def to_geojson(self, path: str | Path, geometry: str = "crowns") -> None:
        """Write the trees as a GeoJSON FeatureCollection.

        Parameters
        ----------
        path
            Output file.
        geometry : {"crowns", "tops"}
            Crown polygons (trees without a crown of 3 vertices are left
            out) or top points; the :meth:`table` columns are the
            properties.
        """
        if geometry not in ("crowns", "tops"):
            raise ValueError(f"geometry must be 'crowns' or 'tops', got {geometry!r}")
        feats = []
        t = self.table()
        for k in range(len(self)):
            props = {c: (int(v[k]) if c in ("id", "n_points") else
                         (None if not np.isfinite(v[k]) else float(v[k]))) for c, v in t.items()}
            if geometry == "tops":
                geom = {"type": "Point", "coordinates": [float(self.x[k]), float(self.y[k])]}
            else:
                ring = np.asarray(self.crowns[k])
                if len(ring) < 3:
                    continue
                coords = [[float(a), float(b)] for a, b in ring]
                geom = {"type": "Polygon", "coordinates": [coords + [coords[0]]]}
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
        fc = {"type": "FeatureCollection", "features": feats}
        if self.crs is not None and self.crs.startswith("EPSG:"):
            name = f"urn:ogc:def:crs:EPSG::{self.crs[5:]}"
            fc["crs"] = {"type": "name", "properties": {"name": name}}
        Path(path).write_text(json.dumps(fc))


def _settings(method, resolution, window, hmin, shape, tops_from, th_tree, th_seed, th_cr,
              max_cr, dt1, dt2, R, Zu, speed_up, hull, concavity, min_point_height, smooth,
              hmax) -> dict:
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(_METHODS)}")
    if tops_from not in ("chm", "points"):
        raise ValueError(f"tops_from must be 'chm' or 'points', got {tops_from!r}")
    if hull not in ("convex", "concave"):
        raise ValueError(f"unknown hull {hull!r}; expected 'convex' or 'concave'")
    if int(smooth) != smooth or smooth < 0:
        raise ValueError(f"smooth must be a whole number of cells, 0 or more, got {smooth}")
    kind, vals = _window(window, hmax)
    return {"resolution": float(resolution), "window_kind": kind, "window_values": vals,
            "hmin": float(hmin), "shape": _shape(shape), "tops_from": tops_from,
            "th_tree": float(th_tree), "th_seed": float(th_seed), "th_cr": float(th_cr),
            "max_cr": float(max_cr), "dt1": float(dt1), "dt2": float(dt2), "R": float(R),
            "Zu": float(Zu), "speed_up": float(speed_up), "hull": hull,
            "concavity": float(concavity), "min_point_height": float(min_point_height),
            "smooth": float(smooth)}


def segment_trees(cloud: PointCloud, method: str = "dalponte2016", resolution: float = 0.5,
                  window=5.0, hmin: float = 2.0, shape: str = "circular",
                  tops_from: str = "chm", th_tree: float = 2.0, th_seed: float = 0.45,
                  th_cr: float = 0.55, max_cr: float = 10.0, dt1: float = 1.5,
                  dt2: float = 2.0, R: float = 2.0, Zu: float = 15.0, speed_up: float = 10.0,
                  hull: str = "convex", concavity: float = 2.0,
                  min_point_height: float = 0.5, smooth: int = 0, heights=None) -> Trees:
    """Find the trees of one cloud: tops, crowns and each point's tree.

    For ``"dalponte2016"`` and ``"watershed"`` a CHM is made at
    ``resolution`` (the highest point per cell, as
    :func:`sylva.ground.make_chm`, on a grid aligned with multiples of the
    resolution and two empty cells wider than the cloud), tops are found by
    :func:`locate_trees` on it (or on the points with ``tops_from=
    "points"``), crowns are grown by :func:`segment_crowns`, and each point
    takes the crown of its cell. ``"li2012"`` labels the points directly
    (:func:`li2012`) and a tree's top is its highest point. ``"tops"``
    finds tops only. :func:`find_trees` runs the same on a catalogue.

    Parameters
    ----------
    cloud
        Points; see ``heights``.
    method : {"dalponte2016", "watershed", "li2012", "tops"}
        Segmentation.
    resolution
        CHM cell size (m).
    window, hmin, shape
        Tree-top detection, as for :func:`locate_trees`; a function of
        height is sampled every centimetre and interpolated linearly.
    tops_from : {"chm", "points"}
        Find tops on the CHM or on the points.
    th_tree, th_seed, th_cr, max_cr
        Crown growing, as for :func:`segment_crowns` (``max_cr`` in cells).
    dt1, dt2, R, Zu, speed_up
        :func:`li2012` settings; its ``hmin`` is ``hmin``.
    hull : {"convex", "concave"}
        Crown outline from each tree's points, as for :func:`crown_hull`.
    concavity
        Edge length (m) of a concave outline.
    min_point_height
        Points lower than this (m) are in no tree and outline no crown.
        (For ``"li2012"`` this cannot change the labels of the higher
        points, which are taken first.)
    smooth
        Smooth the CHM with a mean filter over ``(2 * smooth + 1)`` squared
        cells before finding tops and growing crowns (lidR's examples use a
        3 x 3 focal mean, ``smooth=1``); 0 for none. A CHM top's height is
        still the unsmoothed cell value.
    heights
        Height of each point (array or attribute name); z if None (a
        normalised cloud).

    Returns
    -------
    Trees
        With ``tree_id`` giving each point's tree (0 for none).

    Raises
    ------
    ValueError
        For an unknown method, hull or shape, or a bad setting.

    Examples
    --------
    >>> from sylva import als
    >>> w = als.LinearWindow(2, 0.1, 3, 8)
    >>> trees = als.segment_trees(normalised, window=w)       # doctest: +SKIP
    >>> trees.to_geojson("crowns.geojson")                    # doctest: +SKIP
    """
    h = _point_heights(cloud, heights)
    fin = h[np.isfinite(h)]
    s = _settings(method, resolution, window, hmin, shape, tops_from, th_tree, th_seed, th_cr,
                  max_cr, dt1, dt2, R, Zu, speed_up, hull, concavity, min_point_height, smooth,
                  float(fin.max()) if fin.size else 0.0)
    d, labels = _core.als_trees_segment(np.ascontiguousarray(cloud.xyz), h, method, s)
    return Trees._from_core(d, tree_id=labels, crs=cloud.crs)


def find_trees(catalog, out: str | Path | None = None, method: str = "dalponte2016",
               resolution: float = 0.5, dtm="auto", dtm_resolution: float = 1.0,
               window=5.0, hmin: float = 2.0, shape: str = "circular", tops_from: str = "chm",
               th_tree: float = 2.0, th_seed: float = 0.45, th_cr: float = 0.55,
               max_cr: float = 10.0, dt1: float = 1.5, dt2: float = 2.0, R: float = 2.0,
               Zu: float = 15.0, speed_up: float = 10.0, hull: str = "convex",
               concavity: float = 2.0, min_point_height: float = 0.5, smooth: int = 0,
               attribute: str = "tree_id", chunk_size: float | None = None,
               buffer: float = 30.0, workers: int | None = None,
               format: str | None = None) -> Trees:
    """Find the trees of a whole catalogue, and optionally write every tile
    with each point's tree id.

    Each chunk is segmented with its buffer as :func:`segment_trees` does
    it, on the CHM grid shared by the whole catalogue. A tree belongs to
    the chunk that holds its top (a CHM-cell top to the chunk whose core
    is nearest the cell centre, as raster outputs are joined; a point top
    to the chunk whose core holds the point) and is reported once, with
    the crown that chunk finds for it. Ids are 1, 2, ... in order of the
    tops' x, then y, over the whole catalogue, so they do not depend on the
    chunks or the number of workers; with a wide enough buffer the trees
    are those :func:`segment_trees` finds in all the tiles merged.

    Parameters
    ----------
    catalog
        The tiles.
    out
        Directory for labelled tiles (one per input tile, or per chunk on a
        grid): every point with its tree id in ``attribute`` (0 for none).
        A point is written by the chunk whose core holds it and takes the
        id of the tree whose crown holds it, wherever that tree's top is.
        Needs crowns (not ``method="tops"``).
    method, resolution, window, hmin, shape, tops_from
        As for :func:`segment_trees`. A window function is sampled every
        centimetre from 0 to the catalogue's height range.
    th_tree, th_seed, th_cr, max_cr, dt1, dt2, R, Zu, speed_up
        As for :func:`segment_trees`.
    hull, concavity, min_point_height, smooth
        As for :func:`segment_trees`.
    dtm : "auto", None or Raster
        Heights above ground: ``"auto"`` makes one DTM of the whole
        catalogue from its ground points first (as :func:`sylva.als.dtm`
        with ``method="lowest"`` at ``dtm_resolution``), so that a point
        has the same height in every chunk that reads it; a
        :class:`~sylva.Raster` is subtracted; None takes z as height
        (normalised tiles).
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    attribute
        Name of the tree id attribute in the written tiles.
    chunk_size, workers, format
        As for :func:`sylva.als.apply`.
    buffer
        Width (m) of the band read around each chunk. For a crown near a
        chunk edge to be complete and the same as in one run, the buffer
        must hold the crown and the crowns and tops that compete with it:
        at least half the largest window plus two crown diameters for the
        CHM methods, twice ``speed_up`` for ``"li2012"``.

    Returns
    -------
    Trees
        With ``catalog`` (the labelled tiles) when ``out`` is given. A
        warning is issued when crowns reach the edge of a buffer or points
        could not be matched to a tree, both signs of a narrow buffer.

    Raises
    ------
    ValueError
        As :func:`segment_trees`, for ``out`` with ``method="tops"``, or for
        a catalogue without ground points with ``dtm="auto"``.
    """
    cat = _as_catalog(catalog)
    mode, raster = _heights(dtm)
    b = cat.bounds
    hmax = 0.0 if b is None else max(b[5] - b[2], b[5], 1.0)
    s = _settings(method, resolution, window, hmin, shape, tops_from, th_tree, th_seed, th_cr,
                  max_cr, dt1, dt2, R, Zu, speed_up, hull, concavity, min_point_height, smooth,
                  hmax)
    d = _core.als_trees_catalog(cat._core(), method, s, mode, raster, float(dtm_resolution),
                                None if out is None else str(out), str(attribute), _format(format),
                                **_run_kw(chunk_size, buffer, workers))
    written = _written(list(d["written"]), cat) if out is not None else None
    t = Trees._from_core(d, catalog=written, at_edge=int(d["at_edge"]),
                         unmatched_points=int(d["unmatched_points"]), crs=cat.crs)
    if t.at_edge or t.unmatched_points:
        warnings.warn(f"{t.at_edge} crowns reached the edge of their chunk's buffer and "
                      f"{t.unmatched_points} points could not be matched to a tree; a wider buffer "
                      f"(now {buffer} m) may be needed for complete crowns", stacklevel=2)
    return t
