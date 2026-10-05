# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Canopy gaps and their change."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import numpy as np

from ... import _core
from ...raster import Raster
from ._common import _num, _pos, _raster_arg
from .surface import SurfaceChange


def _polygons(parts) -> list:
    return [(np.asarray(ext), [np.asarray(h) for h in holes]) for ext, holes in parts]


def _geojson_polygon(parts) -> dict:
    def ring(r):
        c = [[float(x), float(y)] for x, y in r]
        return c + [c[0]]
    polys = [[ring(ext)] + [ring(h) for h in holes] for ext, holes in parts]
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def _write_geojson(path, geoms: list, props: list, crs: str | None) -> None:
    feats = [{"type": "Feature", "geometry": g, "properties": p}
             for g, p in zip(geoms, props, strict=True)]
    fc = {"type": "FeatureCollection", "features": feats}
    if crs is not None and crs.startswith("EPSG:"):
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs[5:]}"}}
    Path(path).write_text(json.dumps(fc))


def _json_value(v):
    if isinstance(v, (np.integer, Integral)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return v
    v = float(v)
    return v if np.isfinite(v) else None


@dataclass
class Gaps:
    """Canopy gaps of one CHM, from :func:`canopy_gaps`.

    Attributes
    ----------
    labels
        ``(rows, cols)`` gap id of each cell (0 for none).
    id, area, n_cells, x, y, mean_height, max_height
        Per gap: id (1, 2, ... in order of the first cell from the
        south-west), area (m²), cells, centre, and mean and largest CHM
        height of its cells (m).
    polygons
        Per gap, its outline along cell edges: a list of parts, each
        ``(exterior, holes)`` with ``(k, 2)`` rings (exteriors
        counter-clockwise). Parts touch only at corners.
    area_with_data
        Area (m²) of the cells with a CHM value.
    xmin, ymin, resolution
        The CHM's grid.
    crs
        Its coordinate system.
    settings
        ``height``, ``min_area``, ``max_area``, ``connectivity``.
    """

    labels: np.ndarray
    id: np.ndarray
    area: np.ndarray
    n_cells: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mean_height: np.ndarray
    max_height: np.ndarray
    polygons: list
    area_with_data: float
    xmin: float
    ymin: float
    resolution: float
    crs: str | None = None
    settings: dict = field(default_factory=dict)

    @classmethod
    def _from_core(cls, d: dict, r: Raster, settings: dict) -> Gaps:
        return cls(np.asarray(d["labels"]), np.asarray(d["id"]), np.asarray(d["area"]),
                   np.asarray(d["n_cells"]), np.asarray(d["x"]), np.asarray(d["y"]),
                   np.asarray(d["mean_height"]), np.asarray(d["max_height"]),
                   [_polygons(p) for p in d["polygons"]], float(d["area_with_data"]),
                   float(r.xmin), float(r.ymin), float(r.resolution), r.crs, dict(settings))

    def __len__(self) -> int:
        return len(self.id)

    def __repr__(self) -> str:
        return f"Gaps({len(self)} gaps, {self.gap_fraction * 100:.1f} % of the area)"

    @property
    def gap_fraction(self) -> float:
        """Share of the area with data that is gap."""
        if self.area_with_data <= 0:
            return float("nan")
        return float(self.area.sum() / self.area_with_data)

    def table(self) -> dict[str, np.ndarray]:
        """``id``, ``area``, ``n_cells``, ``x``, ``y``, ``mean_height`` and
        ``max_height`` per gap."""
        return {"id": self.id, "area": self.area, "n_cells": self.n_cells, "x": self.x,
                "y": self.y, "mean_height": self.mean_height, "max_height": self.max_height}

    def size_distribution(self, bins=None) -> tuple[np.ndarray, np.ndarray]:
        """Number of gaps per size class.

        Parameters
        ----------
        bins
            Bin edges (m²); by default powers of two from the smallest
            gap size allowed.

        Returns
        -------
        (edges, counts)
        """
        lo = max(self.settings.get("min_area", 0.0), self.resolution ** 2)
        if bins is None:
            top = max(float(self.area.max()) if len(self) else lo, lo)
            k = int(np.ceil(np.log2(top / lo))) + 1
            bins = lo * 2.0 ** np.arange(k + 1)
        edges = np.asarray(bins, dtype=float)
        counts, _ = np.histogram(self.area, bins=edges)
        return edges, counts

    def size_exponent(self, xmin: float | None = None) -> tuple[float, float, int]:
        """Exponent of a power law ``p(a) ~ a^-alpha`` fitted to the gap
        sizes by maximum likelihood (Clauset et al. 2009).

        Parameters
        ----------
        xmin
            Smallest size included (m²); the smallest allowed by default.

        Returns
        -------
        (alpha, standard error, number of gaps)
            NaN with fewer than two gaps.
        """
        if xmin is None:
            xmin = max(self.settings.get("min_area", 0.0), self.resolution ** 2)
        a, se, n = _core.change_als_size_exponent(np.ascontiguousarray(self.area, dtype=np.float64),
                                                  _pos("xmin", xmin))
        return float(a), float(se), int(n)

    def to_geojson(self, path, properties: dict | None = None) -> None:
        """Write the gap outlines as GeoJSON polygons with :meth:`table` as
        properties.

        Parameters
        ----------
        path
            Output file.
        properties
            Extra per-gap columns (name to sequence).
        """
        t = self.table()
        if properties:
            t = {**t, **properties}
        props = [{k: _json_value(v[i]) for k, v in t.items()} for i in range(len(self))]
        _write_geojson(path, [_geojson_polygon(p) for p in self.polygons], props, self.crs)


def canopy_gaps(chm: Raster, height: float = 2.0, min_area: float = 10.0,
                max_area: float | None = None, connectivity: int = 8) -> Gaps:
    """Canopy gaps of a CHM: connected cells no higher than ``height``.

    As ForestGapR's ``getForestGaps`` (Silva et al. 2019): cells at or below
    the height threshold are joined across edges and corners
    (``connectivity=8``) or edges only (4), and regions whose area lies
    between ``min_area`` and ``max_area`` are gaps. A threshold of 2 m
    follows Brokaw's (1982) definition of a gap as an opening reaching down
    to within 2 m of the ground. NaN cells are never gap.

    Parameters
    ----------
    chm
        Canopy height model.
    height
        Gap height threshold (m).
    min_area, max_area
        Size limits (m²); no upper limit by default.
    connectivity : {8, 4}
        Cell neighbourhood.

    Returns
    -------
    Gaps
    """
    r = _raster_arg(chm, "chm")
    settings = _gap_settings(height, min_area, max_area, connectivity)
    d = _core.change_als_gaps(r, settings["height"], settings["min_area"],
                              _inf(settings["max_area"]), settings["connectivity"])
    return Gaps._from_core(d, chm, settings)


def _inf(v):
    return float("inf") if v is None else v


def _gap_settings(height, min_area, max_area, connectivity) -> dict:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    lo = _num("min_area", min_area, 0.0)
    hi = None if max_area is None else _num("max_area", max_area, lo)
    return dict(height=_num("height", height), min_area=lo, max_area=hi,
                connectivity=int(connectivity))


#: Names of the cell codes of a gap change (``GapChange.cells``).
GAP_CELLS = ("canopy", "stable_gap", "formed", "closed", "uncertain", "no_data")


@dataclass
class GapChange:
    """Canopy gaps of two surveys and their dynamics, from :func:`gap_change`.

    Attributes
    ----------
    a, b
        The gaps of each survey.
    cells
        ``(rows, cols)`` codes of :data:`GAP_CELLS`: ``canopy`` (no gap in
        either), ``stable_gap``, ``formed`` (gap only in the second survey,
        with a significant loss), ``closed`` (gap only in the first, with a
        significant gain), ``uncertain`` (crossed the threshold without a
        significant change) and ``no_data``.
    status_a, closed_area
        Per gap of the first survey: ``closed`` (no part is gap any more),
        ``shrunk``, ``stable`` or ``uncertain``; and its area closed (m²).
    status_b, formed_area
        Per gap of the second survey: ``new``, ``expanded``, ``stable`` or
        ``uncertain``; and its area formed (m²).
    areas
        Area (m²) per cell code.
    years
        Time between the surveys, if given.
    """

    a: Gaps
    b: Gaps
    cells: np.ndarray
    status_a: list
    closed_area: np.ndarray
    status_b: list
    formed_area: np.ndarray
    areas: dict
    years: float | None = None

    def summary(self) -> dict:
        """Gap fractions, areas formed and closed (with annual rates as a
        share of the area compared when ``years`` is known), gap counts by
        status and the size-distribution exponents of both surveys."""
        compared = sum(v for k, v in self.areas.items() if k != "no_data")
        out = {"gap_fraction_a": self.a.gap_fraction, "gap_fraction_b": self.b.gap_fraction,
               "area_compared": compared, "area_formed": self.areas["formed"],
               "area_closed": self.areas["closed"], "area_uncertain": self.areas["uncertain"]}
        for k in ("new", "expanded", "stable", "uncertain"):
            out[f"gaps_b_{k}"] = int(sum(1 for s in self.status_b if s == k))
        for k in ("closed", "shrunk", "stable", "uncertain"):
            out[f"gaps_a_{k}"] = int(sum(1 for s in self.status_a if s == k))
        if self.years and compared > 0:
            out["formation_rate"] = self.areas["formed"] / compared / self.years
            out["closure_rate"] = self.areas["closed"] / compared / self.years
        for k, g in (("a", self.a), ("b", self.b)):
            alpha, se, n = g.size_exponent()
            out[f"size_exponent_{k}"], out[f"size_exponent_se_{k}"] = alpha, se
        return out

    def report(self) -> str:
        """A summary in words."""
        s = self.summary()
        lines = [f"Gaps (at most {self.a.settings['height']:g} m high, at least "
                 f"{self.a.settings['min_area']:g} m²): {len(self.a)} in survey a "
                 f"({s['gap_fraction_a'] * 100:.1f} %), {len(self.b)} in survey b "
                 f"({s['gap_fraction_b'] * 100:.1f} %)",
                 f"  formed   {s['area_formed']:,.0f} m² (gaps new: {s['gaps_b_new']}, "
                 f"expanded: {s['gaps_b_expanded']})",
                 f"  closed   {s['area_closed']:,.0f} m² (gaps closed: {s['gaps_a_closed']}, "
                 f"shrunk: {s['gaps_a_shrunk']})",
                 f"  uncertain {s['area_uncertain']:,.0f} m² crossed the threshold without "
                 "a significant change"]
        if "formation_rate" in s:
            lines.append(f"  rates    formation {s['formation_rate'] * 100:.2f} %/yr, closure "
                         f"{s['closure_rate'] * 100:.2f} %/yr of the area compared")
        return "\n".join(lines)

    def to_geojson(self, path, survey: str = "b") -> None:
        """Write one survey's gaps with their status and area formed or
        closed.

        Parameters
        ----------
        path
            Output file.
        survey : {"a", "b"}
            Which gaps.
        """
        if survey == "a":
            self.a.to_geojson(path, {"status": self.status_a, "closed_area": self.closed_area})
        elif survey == "b":
            self.b.to_geojson(path, {"status": self.status_b, "formed_area": self.formed_area})
        else:
            raise ValueError(f"survey must be 'a' or 'b', got {survey!r}")


def gap_change(change, chm_b: Raster | None = None, height: float = 2.0, min_area: float = 10.0,
               max_area: float | None = None, connectivity: int = 8,
               years: float | None = None) -> GapChange:
    """Gap formation and closure between two surveys.

    Gaps are found in each CHM as :func:`canopy_gaps` finds them. A cell
    *forms* a gap when it is in a gap of the second survey only and its CHM
    fell significantly, and *closes* one when it is in a gap of the first
    only and its CHM rose significantly (as ForestGapR's ``GapChangeDec``,
    Silva et al. 2019, but with the level of detection of
    :func:`surface_change`); a cell that crossed the threshold without a
    significant change is ``uncertain``. Each gap of the second survey is
    ``new``, ``expanded``, ``stable`` or ``uncertain``, each of the first
    ``closed``, ``shrunk``, ``stable`` or ``uncertain``.

    Parameters
    ----------
    change
        A CHM :class:`SurfaceChange`, or the first survey's CHM (a
        :class:`~sylva.Raster`) with ``chm_b``; without a surface change
        every transition counts.
    chm_b
        The second survey's CHM, on the grid of the first.
    height, min_area, max_area, connectivity
        As for :func:`canopy_gaps`.
    years
        Time between the surveys, for annual rates.

    Returns
    -------
    GapChange
    """
    if isinstance(change, SurfaceChange):
        if change.surface != "chm":
            raise ValueError(f"gap change needs a CHM change, got a {change.surface!r} change")
        ra, rb, cls = change.a, change.b, np.ascontiguousarray(change.classes, dtype=np.uint8)
    elif isinstance(change, Raster) and isinstance(chm_b, Raster):
        ra, rb, cls = change, chm_b, None
    else:
        raise ValueError("give a SurfaceChange, or two CHMs (Raster) on one grid")
    if years is not None:
        years = _pos("years", years)
    s = _gap_settings(height, min_area, max_area, connectivity)
    d = _core.change_als_gap_change(_raster_arg(ra, "chm_a"), _raster_arg(rb, "chm_b"), cls,
                                    s["height"], s["min_area"], _inf(s["max_area"]),
                                    s["connectivity"])
    return GapChange(Gaps._from_core(d["a"], ra, s), Gaps._from_core(d["b"], rb, s),
                     np.asarray(d["cells"]), list(d["status_a"]), np.asarray(d["closed_area"]),
                     list(d["status_b"]), np.asarray(d["formed_area"]),
                     dict(zip(d["cell_names"], d["areas"], strict=True)), years)
