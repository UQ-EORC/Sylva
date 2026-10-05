# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Tree-level change between two airborne surveys."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from ... import _core
from ...raster import Raster
from ._common import _confidence, _num, _pos, _raster_arg, _share
from .align import ALSAlignment, _alignment, _cell
from .surface import SurfaceChange

_TREE_COLUMNS = ("id_a", "id_b", "status", "x", "y", "distance", "height_a", "height_b", "dh",
                 "sigma", "lod", "dh_change", "crown_area_a", "crown_area_b", "crown_loss",
                 "crown_gain", "observed")


@dataclass
class ALSTreeChange:
    """Trees of two airborne surveys compared, from :func:`tree_change`.

    ``table`` has one row per tree of the first survey (with its partner in
    the second, if any) and one per tree found only in the second:

    ``id_a``, ``id_b``
        Ids in each survey (0 for none).
    ``status``
        ``survivor``, ``damaged``, ``dead``, ``undetected`` or
        ``unobserved`` for a tree of the first survey; ``recruit``,
        ``released``, ``undetected`` or ``unobserved`` for one found only
        in the second.
    ``x``, ``y``
        Top in the first survey's frame.
    ``distance``
        Between the partners' tops (m).
    ``height_a``, ``height_b``, ``dh``, ``sigma``, ``lod``
        Heights (m), height change, its standard deviation and level of
        detection.
    ``dh_change``
        ``growth``, ``decrease``, ``below_detection`` or ``unmeasured``
        (empty without a partner).
    ``crown_area_a``, ``crown_area_b``
        Crown areas (m²).
    ``crown_loss``, ``crown_gain``
        Shares of the crown with a significant canopy loss or gain.
    ``observed``
        Share of the crown with data in both surveys.

    Attributes
    ----------
    table
        The columns above, as NumPy arrays.
    crs
        Coordinate system of the positions.
    settings
        The settings of the call.
    """

    table: dict
    crs: str | None = None
    settings: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table["id_a"])

    def __repr__(self) -> str:
        c = self.counts()
        return "ALSTreeChange(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")"

    def _core(self) -> dict:
        t = self.table
        out = {k: np.ascontiguousarray(t[k], dtype=np.float64) for k in _TREE_COLUMNS
               if k not in ("id_a", "id_b", "status", "dh_change")}
        out["id_a"] = np.ascontiguousarray(t["id_a"], dtype=np.int64)
        out["id_b"] = np.ascontiguousarray(t["id_b"], dtype=np.int64)
        out["status"] = [str(s) for s in t["status"]]
        out["dh_change"] = [str(s) for s in t["dh_change"]]
        return out

    def counts(self) -> dict[str, int]:
        """Rows per status (``undetected`` and ``unobserved`` split by the
        survey the tree was found in)."""
        out: dict[str, int] = {}
        for s, a in zip(self.table["status"], self.table["id_a"], strict=True):
            key = s if s not in ("undetected", "unobserved") else f"{s}_{'a' if a else 'b'}"
            out[key] = out.get(key, 0) + 1
        return out

    def summary(self, area: float | None = None, years: float | None = None, mask=None) -> dict:
        """Totals: trees by fate, mean height growth of the survivors with
        its standard error (from their scatter, and from their measurement
        uncertainties alone), crown area lost and, with ``years``, annual
        mortality and recruitment rates (Sheil et al. 1995). With ``area``
        (m²), counts and crown areas are also given per hectare.

        Parameters
        ----------
        area
            Area covered (m²).
        years
            Time between the surveys.
        mask
            Rows to include (bool array), e.g. the trees in a plot.

        Returns
        -------
        dict
        """
        m = None if mask is None else np.ascontiguousarray(mask, dtype=bool)
        if years is not None:
            years = _pos("years", years)
        s = dict(_core.change_als_tree_summary(self._core(), m, years))
        if area is not None:
            ha = _pos("area", area) / 1e4
            for k in ("survivors", "damaged", "dead", "recruits", "released"):
                s[k + "_per_ha"] = s[k] / ha
            s["crown_area_lost_per_ha"] = (s["crown_area_dead"] + s["crown_area_damaged"]) / ha
        return s

    def grid(self, resolution: float, bounds=None) -> dict[str, Raster]:
        """Totals per grid cell: ``survivors``, ``damaged``, ``dead``,
        ``recruits``, ``mean_growth``, ``growth_se`` and
        ``crown_area_lost`` (m²).

        Parameters
        ----------
        resolution
            Cell size (m).
        bounds
            ``(xmin, ymin, xmax, ymax)``; the trees' extent snapped to
            multiples of ``resolution`` by default.

        Returns
        -------
        dict of Raster
        """
        res = _pos("resolution", resolution)
        x, y = np.asarray(self.table["x"]), np.asarray(self.table["y"])
        if bounds is None:
            if len(x) == 0:
                raise ValueError("no trees to grid")
            bounds = (np.floor(x.min() / res) * res, np.floor(y.min() / res) * res,
                      x.max(), y.max())
        x0, y0, x1, y1 = (float(v) for v in bounds)
        nc = int(np.floor((x1 - x0) / res)) + 1
        nr = int(np.floor((y1 - y0) / res)) + 1
        d = _core.change_als_tree_grid(self._core(), x0, y0, res, nr, nc)
        out = {}
        for k, v in d.items():
            r = Raster._from_core(v)
            r.crs = self.crs
            out[k] = r
        return out

    def to_csv(self, path) -> None:
        """Write the table as CSV.

        Parameters
        ----------
        path
            Output file.
        """
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(_TREE_COLUMNS))
            for k in range(len(self)):
                w.writerow([_cell(self.table[c][k]) for c in _TREE_COLUMNS])

    def to_pandas(self):
        """The table as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({c: self.table[c] for c in _TREE_COLUMNS})

    def report(self, years: float | None = None) -> str:
        """A summary in words.

        Parameters
        ----------
        years
            Time between the surveys, for annual rates.
        """
        s = self.summary(years=years)
        lines = [f"Trees: {s['survivors']} survivors, {s['damaged']} damaged, {s['dead']} dead, "
                 f"{s['recruits']} recruits, {s['released']} released",
                 f"  not assessed  {s['undetected_a']} + {s['undetected_b']} undetected, "
                 f"{s['unobserved_a']} + {s['unobserved_b']} unobserved (survey a + b)"]
        if s["n_growth"]:
            lines.append(f"  height growth {s['mean_growth']:+.2f} ± {s['growth_se']:.2f} m "
                         f"(mean ± SE over {s['n_growth']} survivors; measurement alone "
                         f"± {s['growth_measurement_se']:.2f}); {s['n_growth_detected']} "
                         "individually above their level of detection")
        if years:
            lines.append(f"  rates         mortality {s['mortality_rate'] * 100:.2f} %/yr, "
                         f"recruitment {s['recruitment_rate'] * 100:.2f} %/yr")
        return "\n".join(lines)


def _tree_dict(trees, name: str) -> dict:
    if hasattr(trees, "table") and hasattr(trees, "crowns"):
        t = {"id": trees.id, "x": trees.x, "y": trees.y, "height": trees.height,
             "crown_area": trees.crown_area, "crowns": trees.crowns}
    elif isinstance(trees, dict):
        missing = [k for k in ("x", "y", "height") if k not in trees]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        t = dict(trees)
    else:
        raise ValueError(f"{name} must be sylva.als.Trees or a dict of columns, got "
                         f"{type(trees).__name__}")
    n = len(np.asarray(t["x"]))
    out = {"id": np.ascontiguousarray(t.get("id", np.arange(1, n + 1)), dtype=np.int64),
           "x": np.ascontiguousarray(t["x"], dtype=np.float64),
           "y": np.ascontiguousarray(t["y"], dtype=np.float64),
           "height": np.ascontiguousarray(t["height"], dtype=np.float64),
           "crown_area": np.ascontiguousarray(t.get("crown_area", np.full(n, np.nan)),
                                              dtype=np.float64)}
    crowns = t.get("crowns") or [np.zeros((0, 2))] * n
    out["crowns"] = [np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1, 2))
                     for c in crowns]
    if any(len(v) != n for k, v in out.items()):
        raise ValueError(f"the columns of {name} differ in length")
    if not (np.all(np.isfinite(out["x"])) and np.all(np.isfinite(out["y"]))
            and np.all(np.isfinite(out["height"]))):
        raise ValueError(f"{name} has non-finite positions or heights")
    return out


def tree_change(trees_a, trees_b, change: SurfaceChange, alignment: ALSAlignment | None = None,
                max_distance: float = 1.5, max_growth: float = 0.3, max_drop: float = 0.2,
                height_weight: float = 1.0, dead_fraction: float = 0.5,
                damage_fraction: float = 0.3, min_observed: float = 0.5,
                confidence: float = 0.95) -> ALSTreeChange:
    """Airborne trees of two surveys matched, with height growth, mortality,
    damage and recruitment.

    The trees are matched by the optimal assignment of
    :func:`sylva.change.match_trees` (Kuhn 1955; Munkres 1957) with the tree
    height in place of the diameter: a pair must lie within
    ``max_distance`` and its height may grow by at most ``max_growth`` or
    fall by at most ``max_drop`` of the larger height. The CHM change says
    what became of the others:

    - a matched tree is a ``survivor``, or ``damaged`` when a significant
      loss covers ``damage_fraction`` of its crown or its height fell by
      more than its level of detection;
    - an unmatched tree of the first survey is ``damaged`` when a tree of
      the second stands within ``2 * max_distance`` of its top on
      significantly lowered canopy (a broken or collapsed crown), ``dead``
      when a significant loss covers ``dead_fraction`` of its crown,
      ``unobserved`` when less than ``min_observed`` of its crown has data,
      and otherwise ``undetected`` (its canopy is still there);
    - an unmatched tree of the second survey is a ``recruit`` when the
      canopy at its top rose significantly from below ``1 - max_growth`` of
      its height (more than an existing crown can grow), ``released`` when
      it fell (an understorey tree exposed by the loss of a neighbour),
      ``unobserved`` without data, and otherwise ``undetected`` (no
      significant change, or a rise a growing crown explains: a top split
      off a neighbour's crown).

    The standard deviation of a height is that of its top's CHM cell in
    ``change`` (noise, sampling of the apex, DTM); the level of detection of
    a height change is the normal quantile of ``confidence`` times the
    standard deviation of the difference.

    Parameters
    ----------
    trees_a, trees_b
        The trees of each survey, from :func:`sylva.als.find_trees` (or a
        dict with ``x``, ``y``, ``height`` and optionally ``id``,
        ``crown_area`` and ``crowns``), each in its own survey's frame.
    change
        The CHM change of the same surveys (:func:`chm_change`), best at
        the resolution the trees were found at.
    alignment
        Moves the second survey's trees into the first's frame (the one
        ``change`` was made with).
    max_distance, max_growth, max_drop, height_weight
        Matching: largest distance (m), largest relative height gain and
        loss, and the weight of the squared relative height difference
        next to the squared distance over ``max_distance``.
    dead_fraction, damage_fraction, min_observed
        Shares of the crown, as above.
    confidence
        Of the levels of detection.

    Returns
    -------
    ALSTreeChange
    """
    if not isinstance(change, SurfaceChange) or change.surface != "chm":
        raise ValueError("change must be a CHM SurfaceChange (from chm_change)")
    ta, tb = _tree_dict(trees_a, "trees_a"), _tree_dict(trees_b, "trees_b")
    settings = dict(max_distance=_pos("max_distance", max_distance),
                    max_growth=_pos("max_growth", max_growth),
                    max_drop=_share("max_drop", max_drop),
                    height_weight=_num("height_weight", height_weight, 0.0),
                    dead_fraction=_share("dead_fraction", dead_fraction),
                    damage_fraction=_share("damage_fraction", damage_fraction),
                    min_observed=_share("min_observed", min_observed),
                    confidence=_confidence(confidence))
    if settings["max_drop"] >= 1:
        raise ValueError("max_drop must be less than 1")
    s = settings
    d = _core.change_als_trees(ta, tb, _raster_arg(change.a, "chm_a"),
                               _raster_arg(change.b, "chm_b"),
                               _raster_arg(change.sigma_a, "sigma_a"),
                               _raster_arg(change.sigma_b, "sigma_b"),
                               np.ascontiguousarray(change.classes, dtype=np.uint8),
                               _alignment(alignment), s["max_distance"], s["max_growth"],
                               s["max_drop"], s["height_weight"], s["confidence"],
                               s["dead_fraction"], s["damage_fraction"], s["min_observed"])
    table = {k: np.asarray(d[k]) for k in _TREE_COLUMNS if k not in ("status", "dh_change")}
    table["status"] = np.asarray(d["status"], dtype="U10")
    table["dh_change"] = np.asarray(d["dh_change"], dtype="U15")
    return ALSTreeChange(table, change.a.crs, settings)
