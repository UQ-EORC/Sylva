# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Area-based metrics of airborne lidar: grids, plots and single clouds.

The standard set follows the area-based metrics of Roussel et al. (2020):
height statistics, quantiles, cumulative deciles and the normalised
Shannon entropy of the heights, with intensity and return metrics, plus
canopy cover and gap fraction from first returns. :func:`grid_metrics`
computes them on a raster over a whole catalogue, chunk by chunk, so that
a cell on a tile edge has the value it would have in a single-tile run;
:func:`plot_metrics` computes them for circular or polygon plots, reading
only the tiles each plot overlaps; :func:`cloud_metrics` for one cloud in
memory. A Python function of the points can replace the standard set, at
the cost of speed.

These functions are also available from :mod:`sylva.als`.

Examples
--------
>>> from sylva import als
>>> cat = als.catalog("normalised/")                          # doctest: +SKIP
>>> m = als.grid_metrics(cat, 20.0, ["zmean", "zq95", "cover"], dtm=None)  # doctest: +SKIP
>>> m["zq95"].to_geotiff("zq95.tif")                          # doctest: +SKIP
>>> plots = als.plot_metrics(cat, [(500, 200), (650, 340)], radius=11.28,
...                          dtm=None)                           # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..geo import masks
from ..util import progress
from ..pointcloud import PointCloud
from ..raster import Raster

__all__ = ["grid_metrics", "pixel_metrics", "plot_metrics", "cloud_metrics", "metric_names",
           "PlotMetrics"]

_PLOT_BATCH = 256


def metric_names(threshold: float = 2.0) -> list[str]:
    """Names of the standard metrics, in the order they are computed.

    Parameters
    ----------
    threshold
        The height of ``pzabove<threshold>`` (its name depends on it).

    Returns
    -------
    list of str
        ``n``, the height metrics ``zmax``, ``zmean``, ``zsd``, ``zskew``,
        ``zkurt``, ``zentropy``, ``pzabovezmean``, ``pzabove2``, ``zq5`` ..
        ``zq95`` and ``zpcum1`` .. ``zpcum9``; ``cover`` and
        ``gap_fraction``; the intensity metrics ``itot``, ``imax``,
        ``imean``, ``isd``, ``iskew``, ``ikurt``, ``ipground`` and
        ``ipcumzq10`` .. ``ipcumzq90``; the return metrics ``p1th`` ..
        ``p5th`` and ``pground``. See the guide for their definitions.
    """
    return list(_core.als_metric_names(float(threshold)))


def _params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative) -> dict:
    for name, v in (("threshold", threshold), ("cover_break", cover_break)):
        if not np.isfinite(float(v)):
            raise ValueError(f"{name} must be a finite height, got {v}")
    if not (np.isfinite(float(entropy_bin)) and float(entropy_bin) > 0):
        raise ValueError(f"entropy_bin must be a positive number, got {entropy_bin}")
    if min_height is not None and not np.isfinite(float(min_height)):
        raise ValueError(f"min_height must be a finite height or None, got {min_height}")
    return {"threshold": float(threshold), "entropy_bin": float(entropy_bin),
            "cover_break": float(cover_break),
            "min_height": None if min_height is None else float(min_height),
            "drop_noise": bool(drop_noise), "clamp_negative": bool(clamp_negative)}


def _source(dtm, dtm_resolution) -> dict:
    """The core's height arguments for ``dtm``."""
    if isinstance(dtm, Raster):
        r = (np.ascontiguousarray(dtm.data, dtype=np.float64), float(dtm.xmin), float(dtm.ymin),
             float(dtm.resolution))
        return {"mode": "dtm", "dtm": r, "dtm_resolution": float(dtm_resolution), "attribute": None}
    if dtm is None:
        return {"mode": "z", "dtm": None, "dtm_resolution": 1.0, "attribute": None}
    if isinstance(dtm, str):
        if dtm == "auto":
            if not (np.isfinite(float(dtm_resolution)) and float(dtm_resolution) > 0):
                raise ValueError(f"dtm_resolution must be a positive number, got {dtm_resolution}")
            return {"mode": "auto", "dtm": None, "dtm_resolution": float(dtm_resolution),
                    "attribute": None}
        if not dtm:
            raise ValueError("dtm must be 'auto', None, a Raster or an attribute name")
        return {"mode": "attribute", "dtm": None, "dtm_resolution": 1.0, "attribute": dtm}
    raise ValueError(f"dtm must be 'auto', None, a Raster or an attribute name, "
                     f"got {type(dtm).__name__}")


def _names(metrics) -> tuple[list[str] | None, str | None]:
    """``(names, single)``: the names asked for, and the one name if a str was given."""
    if metrics is None:
        return None, None
    if isinstance(metrics, str):
        return [metrics], metrics
    names = [str(m) for m in metrics]
    if not names:
        raise ValueError("metrics must name at least one metric, or be None for all")
    return names, None


def _workers(workers) -> int:
    from .catalogue import _workers as w

    return w(workers)


def _as_dict(value) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return {"value": float(value)}


def cloud_metrics(cloud: PointCloud, height=None, threshold: float = 2.0,
                  entropy_bin: float = 1.0, cover_break: float = 2.0,
                  min_height: float | None = None, drop_noise: bool = True,
                  clamp_negative: bool = False) -> dict[str, float]:
    """The standard metrics of one cloud.

    Parameters
    ----------
    cloud
        Returns, normalised (z is height above ground) unless ``height``
        says otherwise.
    height
        None to take z as the height, the name of an attribute holding it
        (``"height"``, as :func:`sylva.als.normalize` and
        :func:`sylva.ground.normalize_height` write), or an array of one
        height per point.
    threshold
        Height (m) of ``pzabove<threshold>``.
    entropy_bin
        Bin width (m) of ``zentropy``.
    cover_break
        Height (m) above which a first return counts as canopy for
        ``cover`` and ``gap_fraction``.
    min_height
        Returns lower than this are left out of every metric (heights below 0 are
        often dropped first, since ``zentropy`` is NaN with a negative height).
    drop_noise
        Leave out returns classified as noise (7 or 18).
    clamp_negative
        Set heights below 0 to 0 before anything else (the other usual
        habit). Ground returns a few centimetres below
        the DTM are in nearly every cell of a survey, and make ``zentropy``
        NaN there; clamping keeps them, as ground, where ``min_height=0``
        would leave out the ones below it.

    Returns
    -------
    dict
        Metric name to value. The intensity and return metrics are present
        when the cloud has ``intensity``, ``return_number`` and
        ``classification``. With no returns left, ``n`` is 0 and every
        other metric NaN.

    Raises
    ------
    ValueError
        For an unknown attribute, heights of the wrong length or bad
        settings.
    """
    p = _params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative)
    if height is None:
        h = None
    elif isinstance(height, str):
        if height not in cloud.attrs:
            raise ValueError(f"the cloud has no {height!r} attribute")
        h = np.ascontiguousarray(cloud.attrs[height], dtype=np.float64)
    else:
        h = np.ascontiguousarray(height, dtype=np.float64).ravel()
        if len(h) != len(cloud):
            raise ValueError(f"{len(h)} heights for {len(cloud)} points")
    names, values = _core.als_cloud_metrics(cloud.xyz, cloud.attrs, h, **p)
    return dict(zip(names, values.tolist(), strict=True))


def grid_metrics(catalog, resolution: float = 20.0, metrics=None, func: Callable | None = None,
                 dtm="auto", dtm_resolution: float = 1.0, threshold: float = 2.0,
                 entropy_bin: float = 1.0, cover_break: float = 2.0,
                 min_height: float | None = None, drop_noise: bool = True,
                 clamp_negative: bool = False, chunk_size: float | None = None,
                 buffer: float = 20.0, workers: int | None = None):
    """Rasters of area-based metrics over a whole catalogue.

    Every return is assigned to the cell of the catalogue grid that holds
    it (cells are half-open, ``[x0, x0 + resolution)``), and the metrics of
    each cell are computed from all of its returns, whichever tiles they
    come from. Each chunk computes the cells within about one cell of its
    core from its points and its buffer, so the result is the same, to the
    bit, for any ``chunk_size`` and any number of ``workers``, and equals
    the metrics of the cells of all the tiles merged into one cloud.

    Parameters
    ----------
    catalog
        The tiles (a :class:`~sylva.als.Catalog`, or anything
        :func:`sylva.als.catalog` accepts).
    resolution
        Cell size (m); 10 to 30 m is usual for area-based models.
    metrics
        None for every standard metric (:func:`metric_names`), a list of
        names, or one name. With ``func``, the names of its results to keep.
    func
        A function of one cell's returns, ``func(cloud)``, returning a dict
        of name to number (or one number, named ``"value"``), in place of
        the standard metrics. ``cloud`` is a :class:`~sylva.PointCloud`
        whose z is the height above ground, with the tiles' attributes, in
        a fixed order (height, x, y, then attributes). It runs once per
        cell in Python, so it is far slower than the built-in set.
    dtm : "auto", None, Raster or str
        Where heights come from, as for :func:`sylva.als.chm`: ``"auto"``
        (a DTM per chunk from its ground points at ``dtm_resolution``), a
        DTM :class:`~sylva.Raster`, None (z is already the height, for
        tiles normalised with ``replace_z=True``), or the name of an
        attribute holding the height (``"height"`` for tiles written by
        :func:`sylva.als.normalize` without ``replace_z``).
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative
        As for :func:`cloud_metrics`.
    chunk_size, buffer
        As for :func:`sylva.als.apply`. The buffer is raised to at least 1.5
        cells; with ``dtm="auto"`` it should also exceed the widest gap in
        the ground.
    workers
        As for :func:`sylva.als.apply`; not used with ``func``, which runs
        one chunk at a time.

    Returns
    -------
    dict of str to Raster, or Raster
        One :class:`~sylva.Raster` per metric on the catalogue grid (as
        :func:`sylva.als.chm` returns), in the order asked, with the
        catalogue's CRS; a single Raster when ``metrics`` is one name.
        Cells without returns are NaN (``n`` included).

    Raises
    ------
    ValueError
        For an unknown metric, bad settings, a ``func`` result that is not
        a number or a dict of numbers, or tiles without ground for
        ``dtm="auto"``.

    Examples
    --------
    >>> m = als.grid_metrics(cat, 20.0, ["zmean", "zq95", "cover"])   # doctest: +SKIP
    >>> top = als.grid_metrics(cat, 20.0, "zmax", dtm=None)           # doctest: +SKIP
    >>> def canopy(c):                                                # doctest: +SKIP
    ...     z = c.z[c.z > 2]
    ...     return {"hmean": z.mean() if len(z) else np.nan, "npts": len(c)}
    >>> mine = als.grid_metrics(cat, 20.0, func=canopy)               # doctest: +SKIP
    """
    from .catalogue import _as_catalog

    cat = _as_catalog(catalog)
    if not (np.isfinite(float(resolution)) and float(resolution) > 0):
        raise ValueError(f"resolution must be a positive number, got {resolution}")
    names, single = _names(metrics)
    p = _params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative)
    src = _source(dtm, dtm_resolution)
    cs = None if chunk_size is None else float(chunk_size)
    if func is None:
        got, rasters = _core.als_grid_metrics(cat._core(), float(resolution), names, **src, **p,
                                              chunk_size=cs, buffer=float(buffer),
                                              workers=_workers(workers))
        out = {n: cat._raster(d) for n, d in zip(got, rasters, strict=True)}
        if not clamp_negative and (min_height is None or min_height < 0):
            _warn_entropy(out)
    else:
        if not callable(func):
            raise ValueError("func must be callable as func(cloud)")
        out = _grid_func(cat, float(resolution), func, names, src, p, cs, float(buffer))
    return out[single] if single is not None else out


pixel_metrics = grid_metrics


def _warn_entropy(out: dict) -> None:
    """Warn when ``zentropy`` is NaN in most cells that have returns."""
    z = out.get("zentropy")
    others = [r.data for k, r in out.items() if k != "zentropy"]
    if z is None or not others:
        return
    has = np.isfinite(out["n"].data if "n" in out else others[0])
    nan = has & ~np.isfinite(z.data)
    if has.sum() and nan.sum() > 0.5 * has.sum():
        import warnings

        warnings.warn(f"zentropy is NaN in {100 * nan.sum() / has.sum():.0f} % of the cells with "
                      "returns: it is NaN for any cell with a height below 0, and "
                      "ground returns just under the DTM are in most cells; pass "
                      "clamp_negative=True (or min_height=0)", stacklevel=3)


def _grid_func(cat, resolution, func, names, src, p, chunk_size, buffer) -> dict[str, Raster]:
    core = cat._core()
    grid = Raster._from_core(_core.als_grid(core, resolution))
    n_cells = grid.data.size
    rank = np.full(n_cells, 2, dtype=np.uint8)
    values: dict[str, np.ndarray] = {}
    chunks = _core.als_metrics_plan(core, resolution, chunk_size, buffer)
    with progress.task("ALS metrics", len(chunks)) as bar:
        for ch in chunks:
            got = _core.als_metric_cells(core, ch, resolution, **src, **p)
            bar.update()
            if got is None:
                continue
            xyz, attrs, heights, cells, starts, central = got
            xyz[:, 2] = heights
            for g, cell in enumerate(cells.tolist()):
                r = 0 if central[g] else 1
                if r >= rank[cell]:
                    continue
                s, e = int(starts[g]), int(starts[g + 1])
                res = _as_dict(func(PointCloud(xyz[s:e], {k: v[s:e] for k, v in attrs.items()})))
                for k, v in res.items():
                    if k not in values:
                        values[k] = np.full(n_cells, np.nan)
                    values[k][cell] = v
                rank[cell] = r
    if not values and src["mode"] == "auto":
        raise ValueError("no chunk has 3 ground points (classification 2); classify ground first "
                         "(als.classify_ground), or give a DTM")
    keep = list(values) if names is None else names
    missing = [n for n in keep if n not in values]
    if missing:
        raise ValueError(f"func returned no {', '.join(map(repr, missing))}; "
                         f"it returned {', '.join(values) or 'nothing'}")
    shape = grid.data.shape
    return {n: cat._raster({"data": values[n].reshape(shape), "xmin": grid.xmin,
                            "ymin": grid.ymin, "resolution": grid.resolution}) for n in keep}


@dataclass
class PlotMetrics:
    """Metrics of plots, one row per plot, from :func:`plot_metrics`.

    Parameters
    ----------
    columns
        The table, a dict of equal-length arrays: ``plot`` (the plot's
        position in the input, from 0), ``id`` when ids were given, then one
        column per metric. A plot without returns has ``n`` 0 and NaN
        elsewhere.
    names
        The metric columns, in order.
    """

    columns: dict
    names: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.columns["plot"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def __repr__(self) -> str:
        return f"PlotMetrics({len(self)} plots, {len(self.names)} metrics)"

    def row(self, i: int) -> dict:
        """The metrics of plot ``i`` as a dict.

        Parameters
        ----------
        i : int
            Plot index.
        """
        return {k: v[i] for k, v in self.columns.items()}

    def as_dict(self) -> dict:
        """The table as a dict of arrays (a copy)."""
        return {k: np.array(v, copy=True) for k, v in self.columns.items()}

    def to_pandas(self):
        """The table as a :class:`pandas.DataFrame` (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(self.columns)

    def to_csv(self, path: str | Path) -> None:
        """Write the table as CSV (empty cells for NaN).

        Parameters
        ----------
        path
            Output file.
        """
        values = np.column_stack([np.asarray(self.columns[n], dtype=np.float64)
                                  for n in self.names]) if self.names else np.zeros((len(self), 0))
        ids = [str(i) for i in self.columns["id"]] if "id" in self.columns else None
        _core.als_metrics_csv(str(path), list(self.names), ids, np.ascontiguousarray(values))


def _plots(plots, radius) -> tuple[int, Callable]:
    """The number of plots and a function giving the core's plot arguments
    for a slice of them."""
    if radius is not None:
        centres = np.asarray(plots, dtype=np.float64)
        if centres.ndim == 1 and centres.shape == (2,):
            centres = centres[None, :]
        if centres.ndim != 2 or centres.shape[1] != 2:
            raise ValueError(f"with a radius, plots must be (N, 2) centres, "
                             f"got shape {centres.shape}")
        r = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(centres),))
        if not np.all(np.isfinite(r) & (r > 0)):
            raise ValueError("radius must be positive and finite")
        circles = np.ascontiguousarray(np.column_stack([centres, r]))
        return len(circles), lambda s: {"circles": circles[s], "polygons": None}
    features = masks._as_features(plots)
    return len(features), lambda s: {"circles": None,
                                     "polygons": masks._flatten(features[s])}


def plot_metrics(catalog, plots, radius=None, metrics=None, func: Callable | None = None,
                 dtm="auto", dtm_resolution: float = 1.0, threshold: float = 2.0,
                 entropy_bin: float = 1.0, cover_break: float = 2.0,
                 min_height: float | None = None, drop_noise: bool = True,
                 clamp_negative: bool = False, ids=None, buffer: float = 20.0,
                 workers: int | None = None) -> PlotMetrics:
    """Area-based metrics of field plots from a catalogue.

    Plots are grouped by the tiles they overlap; each group reads only
    those tiles, and only the points in the group's box (grown by
    ``buffer`` for ``dtm="auto"``). A plot on a tile edge gets its points
    from every tile it overlaps.

    Parameters
    ----------
    catalog
        The tiles.
    plots
        With ``radius``: plot centres, an ``(N, 2)`` array of x, y (or one
        ``(x, y)``). Without: polygons, in any form :mod:`sylva.geo.masks`
        accepts (a :class:`~sylva.geo.masks.Polygons` layer from
        :func:`sylva.geo.masks.read_polygons`, a list of
        :class:`~sylva.geo.masks.Polygon` or ``(K, 2)`` vertex arrays), one plot
        per feature. Points on a plot's boundary are inside it.
    radius
        Radius (m) of circular plots, one for all or one per plot.
    metrics
        None for every standard metric, or a list of names.
    func
        A function of one plot's returns, as for :func:`grid_metrics`. A
        plot without returns is passed an empty cloud (with no attributes
        when it overlaps no tile).
    dtm, dtm_resolution
        Where heights come from, as for :func:`grid_metrics`.
    threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative
        As for :func:`cloud_metrics`.
    ids
        Labels of the plots (one per plot), kept as an ``id`` column.
    buffer
        With ``dtm="auto"``, the band (m) of points read around the plots
        to make their DTM.
    workers
        Groups of plots read at once, as for :func:`sylva.als.apply`.

    Returns
    -------
    PlotMetrics
        One row per plot, in the order given. Plots outside every tile have
        ``n`` 0 and NaN elsewhere.

    Raises
    ------
    ValueError
        For bad plots (a non-positive radius, a ring with fewer than three
        vertices), an unknown metric or bad settings.

    Examples
    --------
    >>> t = als.plot_metrics(cat, [(520.0, 310.0), (580.5, 402.0)], radius=11.28)  # doctest: +SKIP
    >>> t["zq95"], t.to_pandas()                                                  # doctest: +SKIP
    >>> layer = masks.read_polygons("plots.shp")                                  # doctest: +SKIP
    >>> names = [f.properties["name"] for f in layer]                            # doctest: +SKIP
    >>> t = als.plot_metrics(cat, layer, ids=names)                              # doctest: +SKIP
    """
    from .catalogue import _as_catalog

    cat = _as_catalog(catalog)
    n, args = _plots(plots, radius)
    if ids is not None:
        ids = list(ids)
        if len(ids) != n:
            raise ValueError(f"{len(ids)} ids for {n} plots")
    if not (np.isfinite(float(buffer)) and float(buffer) >= 0):
        raise ValueError(f"buffer must be zero or more, got {buffer}")
    names, _ = _names(metrics)
    p = _params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative)
    src = _source(dtm, dtm_resolution)
    kw = {**src, **p, "buffer": float(buffer), "workers": _workers(workers)}
    if func is None:
        got, values = _core.als_plot_metrics(cat._core(), **args(slice(None)), names=names, **kw)
        columns = {n_: values[:, j] for j, n_ in enumerate(got)}
    else:
        if not callable(func):
            raise ValueError("func must be callable as func(cloud)")
        rows: list[dict] = []
        with progress.task("ALS plots", n) as bar:
            for s in range(0, n, _PLOT_BATCH):
                batch = _core.als_plot_points(cat._core(), **args(slice(s, s + _PLOT_BATCH)), **kw)
                for item in batch:
                    if item is None:
                        xyz, attrs = np.zeros((0, 3)), {}
                    else:
                        xyz, attrs, heights = item
                        xyz[:, 2] = heights
                    rows.append(_as_dict(func(PointCloud(xyz, attrs))))
                    bar.update()
        seen: list[str] = []
        for r in rows:
            seen.extend(k for k in r if k not in seen)
        got = seen if names is None else names
        missing = [k for k in got if k not in seen]
        if missing:
            raise ValueError(f"func returned no {', '.join(map(repr, missing))}; "
                             f"it returned {', '.join(seen) or 'nothing'}")
        columns = {k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64) for k in got}
    table = {"plot": np.arange(n)}
    if ids is not None:
        table["id"] = np.asarray(ids)
    table.update(columns)
    return PlotMetrics(table, list(columns))


def _add_commands(sub, fmt: dict, common: Callable, write_raster: Callable) -> None:
    """The ``als-metrics`` and ``als-plot-metrics`` commands (see :mod:`sylva.cli`)."""
    def heights(s):
        s.add_argument("--dtm-resolution", type=float, default=1.0,
                       help="cell size of the DTM made on the fly from the ground points (m)")
        s.add_argument("--dtm", help="use this DTM (.asc) instead of making one")
        s.add_argument("--normalized", action="store_true",
                       help="the tiles are already normalised (z is height)")
        s.add_argument("--height-attribute",
                       help="take heights from this attribute (e.g. 'height' from als-normalize)")
        s.add_argument("--threshold", type=float, default=2.0,
                       help="height of the pzabove metric (m)")
        s.add_argument("--cover-break", type=float, default=2.0,
                       help="height above which first returns are canopy for cover (m)")
        s.add_argument("--entropy-bin", type=float, default=1.0, help="bin width of zentropy (m)")
        s.add_argument("--min-height", type=float, default=None,
                       help="leave out returns below this height (m)")
        s.add_argument("--keep-noise", action="store_true",
                       help="keep returns classified as noise (7, 18)")
        s.add_argument("--clamp-negative", action="store_true",
                       help="set heights below 0 to 0 (else zentropy is NaN where any is)")
        s.add_argument("--metrics", help="comma-separated metric names (default: all)")

    s = sub.add_parser("als-metrics", help="area-based metrics of a directory of ALS tiles, as "
                       "one raster per metric", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("output", help="directory for the rasters, one <metric>.<format> each")
    s.add_argument("--resolution", type=float, default=20.0, help="cell size (m)")
    s.add_argument("--format", choices=["asc", "tif"], default="asc",
                   help="raster format")
    heights(s)
    common(s)
    s.set_defaults(func=lambda a: _cmd_metrics(a, write_raster))

    s = sub.add_parser("als-plot-metrics", help="area-based metrics of plots from a directory of "
                       "ALS tiles, as a CSV table", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("plots", help="plot polygons (.shp, .geojson) or a CSV of centres with "
                   "columns x, y and optionally radius and id")
    s.add_argument("output", help="CSV file for the table")
    s.add_argument("--radius", type=float, default=None,
                   help="radius of circular plots (m), for a CSV without a radius column")
    s.add_argument("--id-field", default=None,
                   help="polygon attribute (or CSV column) to write as the plot id")
    heights(s)
    s.add_argument("--pattern", default="*.la[sz]", help="file name pattern within the directory")
    s.add_argument("--buffer", type=float, default=20.0,
                   help="band of points read around the plots for the on-the-fly DTM (m)")
    s.add_argument("--workers", type=int, default=None,
                   help="groups of plots at once (default: one per CPU, fewer if memory is short)")
    s.set_defaults(func=_cmd_plot_metrics)


def _cli_kw(args) -> dict:
    if args.normalized:
        dtm = None
    elif args.height_attribute:
        dtm = args.height_attribute
    elif args.dtm:
        dtm = Raster.from_ascii_grid(args.dtm)
    else:
        dtm = "auto"
    return {"dtm": dtm, "dtm_resolution": args.dtm_resolution, "threshold": args.threshold,
            "cover_break": args.cover_break, "entropy_bin": args.entropy_bin,
            "min_height": args.min_height, "drop_noise": not args.keep_noise,
            "clamp_negative": args.clamp_negative,
            "metrics": [m.strip() for m in args.metrics.split(",")] if args.metrics else None}


def _cmd_metrics(args, write_raster) -> None:
    from .. import als

    cat = als.catalog(args.input, pattern=args.pattern)
    out = grid_metrics(cat, args.resolution, chunk_size=args.chunk_size, buffer=args.buffer,
                       workers=args.workers, **_cli_kw(args))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for name, r in out.items():
        write_raster(r, str(Path(args.output) / f"{name}.{args.format}"))
    shape = next(iter(out.values())).shape
    print(f"{len(out)} metrics on a {shape[0]} x {shape[1]} grid at {args.resolution} m from "
          f"{len(cat)} tiles -> {args.output}")


def _read_plot_csv(path: str, radius, id_field):
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no plots")
    cols = {k.strip().lower(): k for k in rows[0]}
    for c in ("x", "y"):
        if c not in cols:
            raise ValueError(f"{path} needs columns x and y, found {', '.join(rows[0])}")
    xy = np.array([[float(r[cols["x"]]), float(r[cols["y"]])] for r in rows])
    if "radius" in cols:
        rad = np.array([float(r[cols["radius"]]) for r in rows])
    elif radius is not None:
        rad = radius
    else:
        raise ValueError(f"{path} has no radius column; give --radius")
    key = id_field or ("id" if "id" in cols else None)
    ids = None
    if key is not None:
        if key.lower() not in cols:
            raise ValueError(f"{path} has no {key!r} column")
        ids = [r[cols[key.lower()]] for r in rows]
    return xy, rad, ids


def _cmd_plot_metrics(args) -> None:
    from .. import als

    cat = als.catalog(args.input, pattern=args.pattern)
    if args.plots.lower().endswith(".csv"):
        plots, radius, ids = _read_plot_csv(args.plots, args.radius, args.id_field)
    else:
        plots, radius = masks.read_polygons(args.plots), None
        ids = None
        if args.id_field:
            missing = [i for i, f in enumerate(plots) if args.id_field not in f.properties]
            if missing:
                raise ValueError(f"{len(missing)} plot(s) have no {args.id_field!r} attribute")
            ids = [f.properties[args.id_field] for f in plots]
    table = plot_metrics(cat, plots, radius=radius, ids=ids, buffer=args.buffer,
                         workers=args.workers, **_cli_kw(args))
    table.to_csv(args.output)
    print(f"{len(table.names)} metrics of {len(table)} plots from {len(cat)} tiles -> "
          f"{args.output}")
