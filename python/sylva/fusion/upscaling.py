# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Upscaling plot measurements with airborne metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..raster import Raster
from .link import TreeLinks, _table, _volumes


def plot_values(trees, area: float, wood_density: float, *, volumes=None, min_dbh: float = 0.0) -> dict:
    """Plot totals per hectare from a plot's TLS trees.

    Parameters
    ----------
    trees
        :class:`sylva.trees.Tree` objects, a table (dict of arrays) with
        ``dbh`` and optionally ``volume`` and ``tree_id``, or a
        :class:`TreeLinks` (its TLS trees).
    area
        Plot area (m²). Give only the trees inside the plot.
    wood_density
        Basic wood density (kg/m³), e.g. 500 to 700 for many hardwoods;
        there is no default, since it depends on the species.
    volumes
        Wood volume (m³) per tree, as for :func:`link_trees`, if the trees
        do not carry a ``volume``.
    min_dbh
        Inventory threshold (m): smaller trees are left out.

    Returns
    -------
    dict
        ``agb`` (Mg/ha: volume times wood density), ``volume`` (m³/ha),
        ``basal_area`` (m²/ha), ``stems`` (per ha), ``n_trees`` and
        ``missing_volume`` (trees counted without a volume).

    Raises
    ------
    ValueError
        For a non-positive area or wood density.
    """
    if isinstance(trees, TreeLinks):
        t = trees.tls
    elif isinstance(trees, Mapping):
        t = {k: np.asarray(v) for k, v in trees.items()}
    else:
        t = _table(trees, "trees", ("tree_id", "x", "y", "dbh", "volume"))
    if "dbh" not in t:
        raise ValueError("the trees need a 'dbh' (m)")
    dbh = np.asarray(t["dbh"], dtype=float).ravel()
    n = len(dbh)
    ids = np.asarray(t["tree_id"]) if "tree_id" in t else np.arange(1, n + 1)
    vol = _volumes(volumes, ids, n) if volumes is not None else \
        (np.asarray(t["volume"], dtype=float) if "volume" in t else np.full(n, np.nan))
    return dict(_core.fusion_plot_summary(np.ascontiguousarray(dbh), np.ascontiguousarray(vol), float(area),
                                          float(wood_density), float(min_dbh)))


def _predictor_matrix(predictors, names) -> tuple[np.ndarray, list[str], tuple | None]:
    """Stack predictors (arrays or Rasters on one grid) as columns."""
    if isinstance(predictors, Mapping) or hasattr(predictors, "columns"):
        src = predictors.columns if hasattr(predictors, "columns") and not isinstance(predictors, Mapping) \
            else predictors
        names = list(src) if names is None else list(names)
        missing = [n for n in names if n not in src]
        if missing:
            raise ValueError(f"no predictor {missing[0]!r}; have {sorted(src)}")
        cols = [src[n] for n in names]
    else:
        a = np.asarray(predictors, dtype=float)
        a = a.reshape(-1, 1) if a.ndim == 1 else a
        names = [f"x{k + 1}" for k in range(a.shape[1])] if names is None else list(names)
        if len(names) != a.shape[1]:
            raise ValueError(f"{len(names)} names for {a.shape[1]} predictors")
        cols = [a[:, k] for k in range(a.shape[1])]
    grid = None
    if cols and all(isinstance(c, Raster) for c in cols):
        r0 = cols[0]
        for c in cols[1:]:
            if c.shape != r0.shape or c.xmin != r0.xmin or c.ymin != r0.ymin or c.resolution != r0.resolution:
                raise ValueError("predictor rasters must share one grid")
        grid = (r0.shape, r0.xmin, r0.ymin, r0.resolution, r0.crs)
        cols = [c.data.ravel() for c in cols]
    elif any(isinstance(c, Raster) for c in cols):
        raise ValueError("give every predictor as a Raster or none")
    x = np.column_stack([np.asarray(c, dtype=float).ravel() for c in cols]) if cols else np.zeros((0, 0))
    return np.ascontiguousarray(x), names, grid


@dataclass
class Model:
    """A regression of plot values on ALS metrics, from :func:`fit_model`.

    Attributes
    ----------
    form
        ``"linear"`` (``y = b0 + Σ bk xk``) or ``"loglog"``
        (``ln y = b0 + Σ bk ln xk``).
    names
        Predictor names, in the order of ``coef[1:]``.
    coef, se
        Coefficients (intercept first) and their standard errors, on the
        model's scale.
    sigma, df
        Residual standard error (model scale) and its degrees of freedom.
    n
        Plots.
    r2, adj_r2
        Coefficient of determination on the model's scale, and adjusted.
    xtx_inv
        ``(XᵀX)⁻¹`` for prediction variances.
    correction
        Back-transformation factor ``exp(σ²/2)`` of a loglog model
        (Baskerville 1972); 1 for linear.
    y, fitted, loo
        Plot values, fitted values and leave-one-out predictions (scale of y).
    loo_rmse, loo_bias, loo_r2, loo_rrmse
        Leave-one-out RMSE, mean error (prediction minus observation),
        ``1 - PRESS / SS`` and RMSE as a percentage of the mean of y.
    x_min, x_max
        Range of each predictor over the plots.
    """

    form: str
    names: list
    coef: np.ndarray
    se: np.ndarray
    sigma: float
    df: int
    n: int
    r2: float
    adj_r2: float
    xtx_inv: np.ndarray
    correction: float
    y: np.ndarray
    fitted: np.ndarray
    loo: np.ndarray
    loo_rmse: float
    loo_bias: float
    loo_r2: float
    loo_rrmse: float
    x_min: np.ndarray
    x_max: np.ndarray

    def equation(self) -> str:
        """The fitted model as text."""
        if self.form == "loglog":
            terms = " + ".join(f"{b:.4g} ln({n})" for b, n in zip(self.coef[1:], self.names, strict=True))
            return f"ln y = {self.coef[0]:.4g} + {terms}"
        terms = " + ".join(f"{b:.4g} {n}" for b, n in zip(self.coef[1:], self.names, strict=True))
        return f"y = {self.coef[0]:.4g} + {terms}"

    def summary(self) -> str:
        """Coefficients, fit and leave-one-out statistics as text."""
        t = _core.fusion_t_quantile(0.975, float(self.df))
        lines = [f"{self.form} model, {self.n} plots: {self.equation()}",
                 "  term            coef        se    95 % interval"]
        for name, b, s in zip(["intercept"] + list(self.names), self.coef, self.se, strict=True):
            lines.append(f"  {name:<12} {b:>9.4g} {s:>9.3g}    [{b - t * s:.4g}, {b + t * s:.4g}]")
        lines.append(f"  R² {self.r2:.3f} (adjusted {self.adj_r2:.3f}), residual SE {self.sigma:.4g} "
                     f"on {self.df} df")
        lines.append(f"  leave-one-out: RMSE {self.loo_rmse:.4g} ({self.loo_rrmse:.1f} %), bias "
                     f"{self.loo_bias:+.4g}, R² {self.loo_r2:.3f}")
        return "\n".join(lines)

    def predict(self, predictors, level: float = 0.95) -> dict:
        """Predict at new predictor values.

        Parameters
        ----------
        predictors
            A dict of arrays or of :class:`~sylva.Raster` (one grid) named
            as the model's predictors, or a ``(n, k)`` array in their order.
        level
            Coverage of the prediction interval.

        Returns
        -------
        dict
            ``mean`` (for loglog, ``exp(ŷ + σ²/2)``), ``se`` (standard error
            of a new observation; for loglog that of the log-normal
            distribution), ``lower``, ``upper`` (the t interval, for loglog
            taken back from the log scale) and ``extrapolated`` (a predictor
            outside the plots' range). Rasters when the predictors are
            rasters (``extrapolated`` as 0 and 1, NaN where there is no
            prediction), arrays otherwise. Missing or, for loglog,
            non-positive predictors give NaN.
        """
        if isinstance(predictors, Mapping) or hasattr(predictors, "columns"):
            x, _, grid = _predictor_matrix(predictors, self.names)
        else:
            x, _, grid = _predictor_matrix(predictors, list(self.names))
        d = _core.fusion_predict(self.form, list(map(float, self.coef)), float(self.sigma), int(self.df),
                                 np.ascontiguousarray(self.xtx_inv), list(map(float, self.x_min)),
                                 list(map(float, self.x_max)), x, float(level))
        if grid is None:
            return dict(d)
        shape, xmin, ymin, res, crs = grid
        out = {k: Raster(np.asarray(d[k]).reshape(shape), xmin, ymin, res, crs)
               for k in ("mean", "se", "lower", "upper")}
        ext = np.where(np.isfinite(out["mean"].data), np.asarray(d["extrapolated"], float).reshape(shape), np.nan)
        out["extrapolated"] = Raster(ext, xmin, ymin, res, crs)
        return out


def fit_model(y, predictors, model: str = "loglog", names=None) -> Model:
    """Regress plot values on ALS metrics.

    Ordinary least squares, in one of two forms:

    - ``"loglog"`` (the default): ``ln y = b0 + Σ bk ln xk``, the usual form
      for biomass against canopy height (a power law), with predictions
      taken back as ``exp(ŷ + σ²/2)`` (Baskerville 1972);
    - ``"linear"``: ``y = b0 + Σ bk xk``.

    Leave-one-out predictions come from the hat matrix in closed form (they
    equal refitting without each plot), and with them the RMSE, bias and R²
    a new plot can expect. Keep the predictors few (one or two for ten
    plots): each costs a degree of freedom, and correlated metrics (``zq95``
    and ``zmax``) add little.

    Parameters
    ----------
    y
        Plot values, e.g. the ``agb`` of :func:`plot_values` per plot.
    predictors
        Per plot, the metrics: a dict of arrays, a
        :class:`sylva.als.PlotMetrics`, or an ``(n, k)`` array.
    model : {"loglog", "linear"}
        Model form.
    names
        The predictors to use (keys of ``predictors``), or names for the
        columns of an array; every column if None.

    Returns
    -------
    Model

    Raises
    ------
    ValueError
        With fewer than ``k + 2`` plots for ``k`` coefficients, non-finite
        values, values that are not positive for ``loglog``, or collinear
        predictors.

    Examples
    --------
    >>> m = fusion.fit_model(agb, plot_metrics, "loglog", names=["zq95", "cover"])  # doctest: +SKIP
    >>> print(m.summary())                                                          # doctest: +SKIP
    """
    yv = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    x, nm, grid = _predictor_matrix(predictors, names)
    if grid is not None:
        raise ValueError("plot predictors must be arrays, not rasters")
    if x.shape[0] != len(yv):
        raise ValueError(f"{x.shape[0]} rows of predictors for {len(yv)} plot values")
    d = _core.fusion_fit(yv, x, str(model))
    return Model(d["form"], nm, d["coef"], d["se"], float(d["sigma"]), int(d["df"]), int(d["n"]),
                 float(d["r2"]), float(d["adj_r2"]), d["xtx_inv"], float(d["correction"]), yv, d["fitted"],
                 d["loo"], float(d["loo_rmse"]), float(d["loo_bias"]), float(d["loo_r2"]),
                 float(d["loo_rrmse"]), d["x_min"], d["x_max"])


@dataclass
class Upscaling:
    """A model and its wall-to-wall prediction, from :func:`upscale`.

    Attributes
    ----------
    model
        The fitted :class:`Model`.
    mean, se, lower, upper
        Rasters of the prediction, the standard error of a new observation
        and the interval at ``level``.
    extrapolated
        Raster: 1 where a predictor lies outside the range of the plots, 0
        elsewhere, NaN without a prediction.
    level
        Coverage of the interval.
    """

    model: Model
    mean: Raster
    se: Raster
    lower: Raster
    upper: Raster
    extrapolated: Raster
    level: float = 0.95

    def to_ascii_grids(self, directory: str | Path, prefix: str = "") -> list[str]:
        """Write the rasters as ESRI ASCII grids.

        Parameters
        ----------
        directory
            Output directory (created if needed).
        prefix
            Prepended to ``mean.asc``, ``se.asc``, ``lower.asc``,
            ``upper.asc`` and ``extrapolated.asc``.

        Returns
        -------
        list of str
            The files written.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in ("mean", "se", "lower", "upper", "extrapolated"):
            p = out / f"{prefix}{name}.asc"
            getattr(self, name).to_ascii_grid(p)
            paths.append(str(p))
        return paths


def upscale(y, plot_metrics, grid: Mapping, predictors, model: str = "loglog",
            level: float = 0.95) -> Upscaling:
    """Fit plot values on ALS metrics and predict them over the ALS grid.

    :func:`fit_model` on the plots, then :meth:`Model.predict` on every cell
    of ``grid``. The grid's cells should have the plots' area (e.g. 25 m
    cells for 0.06 ha plots), since metrics such as the height percentiles
    depend on the area they are computed over.

    Parameters
    ----------
    y
        Plot values (e.g. ``agb`` from :func:`plot_values`), one per plot.
    plot_metrics
        The plots' ALS metrics: :class:`sylva.als.PlotMetrics` (from
        :func:`sylva.als.plot_metrics`) or a dict of arrays.
    grid
        The same metrics as rasters on one grid (from
        :func:`sylva.als.grid_metrics`), by name.
    predictors
        Names of the metrics to use.
    model : {"loglog", "linear"}
        Model form.
    level
        Coverage of the prediction interval.

    Returns
    -------
    Upscaling

    Raises
    ------
    ValueError
        As :func:`fit_model`, or for a grid without a predictor.

    Examples
    --------
    >>> plots = als.plot_metrics(cat, centres, radius=15.0, metrics=["zq95", "cover"])  # doctest: +SKIP
    >>> grid = als.grid_metrics(cat, 25.0, ["zq95", "cover"])                           # doctest: +SKIP
    >>> up = fusion.upscale(agb, plots, grid, ["zq95"], "loglog")                       # doctest: +SKIP
    >>> up.mean.to_geotiff("agb.tif")                                                   # doctest: +SKIP
    """
    names = [predictors] if isinstance(predictors, str) else list(predictors)
    missing = [n for n in names if n not in grid]
    if missing:
        raise ValueError(f"the grid has no {missing[0]!r} raster")
    m = fit_model(y, plot_metrics, model, names)
    p = m.predict({n: grid[n] for n in names}, level)
    if not isinstance(p["mean"], Raster):
        raise ValueError("grid must hold Rasters")
    return Upscaling(m, p["mean"], p["se"], p["lower"], p["upper"], p["extrapolated"], float(level))
