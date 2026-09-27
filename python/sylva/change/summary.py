# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Plot-level growth, mortality and recruitment between two epochs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .. import _core
from .trees import TreeIncrements

__all__ = ["Estimate", "PlotSummary", "plot_summary"]


class Estimate(NamedTuple):
    """One plot-level quantity with its interval."""

    estimate: float
    low: float
    high: float
    unit: str


@dataclass
class PlotSummary:
    """Result of :func:`plot_summary`.

    Quantities (``summary[name]`` gives an :class:`Estimate`):

    ``n_survivors``, ``n_deaths``, ``n_recruits``, ``n_ambiguous``
        Tree counts; ambiguous trees (merges and splits) are left out of
        everything else.
    ``mortality_rate``, ``recruitment_rate``
        Annual rates, ``1 - (S / N0) ** (1 / years)`` and ``1 - (S / N1) **
        (1 / years)`` (Sheil et al. 1995), with ``S`` survivors and ``N0``,
        ``N1`` the trees of each epoch.
    ``dbh_increment``
        Mean DBH increment of the survivors (m/yr).
    ``basal_area_growth``, ``basal_area_mortality``, ``basal_area_recruitment``, ``basal_area_net``
        m²/ha/yr: the survivors' increase, the basal area of the dead trees
        (first epoch), of the recruits (second epoch), and growth plus
        recruitment minus mortality.
    ``volume_growth``, ``volume_mortality``, ``volume_recruitment``, ``volume_net``
        The same for stem volume (``form_factor`` x basal area x height),
        m³/ha/yr.
    ``biomass_*``
        Volume times ``wood_density`` (Mg/ha/yr), when a density is given.

    Attributes
    ----------
    quantities
        ``{name: Estimate}`` in the order above.
    settings
        The settings of the call, for :func:`provenance`.
    """

    quantities: dict
    settings: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> Estimate:
        return self.quantities[name]

    def __contains__(self, name: str) -> bool:
        return name in self.quantities

    def as_dict(self) -> dict:
        """``{name: (estimate, low, high, unit)}`` as plain tuples."""
        return {k: tuple(v) for k, v in self.quantities.items()}

    def report(self) -> str:
        """Plain-text table of the quantities and their intervals."""
        level = int(round(100 * self.settings.get("confidence", 0.95)))
        lines = [f"{'quantity':<24} {'estimate':>12} {f'{level} % interval':>26}  unit"]
        for k, q in self.quantities.items():
            lines.append(f"{k:<24} {q.estimate:>12.5g} {f'[{q.low:.5g}, {q.high:.5g}]':>26}  {q.unit}")
        return "\n".join(lines)


def _pick(measured, fallback, se_default=np.nan):
    """Measured values where finite, else the fallback with no error."""
    value = np.where(np.isfinite(measured[0]), measured[0], fallback)
    se = np.where(np.isfinite(measured[0]), measured[1], se_default)
    return value, se


def plot_summary(increments: TreeIncrements, area: float, years: float = 1.0, n_draws: int = 2000,
                 confidence: float = 0.95, form_factor: float = 0.5, wood_density: float | None = None,
                 seed: int = 0) -> PlotSummary:
    """Growth, mortality and recruitment of a plot, with intervals.

    Every quantity is computed from the measured trees, and again for
    ``n_draws`` Monte Carlo draws in which each tree's DBH, height and
    increments are perturbed by their standard errors (the registration
    uncertainty is part of the increments' errors). Survivors whose
    increment could not be measured take the mean measured increment in the
    estimate and a randomly drawn measured survivor's increment in each
    draw. The interval is the central ``confidence`` range of the draws.

    Parameters
    ----------
    increments
        From :func:`tree_increments`; its match gives the deaths and
        recruits, its measurements their DBH and height. A tree without a
        stem profile falls back to the DBH and height of its tree object,
        with no measurement error.
    area
        Plot area (m²). Restrict the trees to the plot beforehand so that
        trees and area cover the same ground.
    years
        Time between the epochs.
    n_draws
        Monte Carlo draws.
    confidence
        Central probability of the intervals.
    form_factor
        Stem volume over basal area times height.
    wood_density
        Oven-dry wood density (t/m³) for biomass; biomass is left out if None.
    seed
        Random seed of the draws.

    Returns
    -------
    PlotSummary

    Raises
    ------
    ValueError
        If a setting is out of range or a tree has no DBH at all.

    Notes
    -----
    The intervals cover measurement noise and registration only. The counts
    of deaths and recruits are taken as exact (a wrong match is not
    modelled), and the sampling error of a plot as a sample of a stand is
    not included. On repeated :func:`sylva.synthetic.forest_epochs` plots
    the 95 % intervals of the basal-area terms cover the true values in
    about 95 % of the runs; see *Change detection* in the guide.

    Examples
    --------
    >>> s = change.plot_summary(inc, area=30 * 30, years=5)
    >>> s["basal_area_net"]
    """
    if not isinstance(increments, TreeIncrements):
        raise ValueError("increments must be a TreeIncrements from tree_increments")
    m = increments.match
    ma, mb = increments.measures
    amb = m.ambiguous_pairs()
    keep = ~amb
    c = increments.columns

    def attr(trees, idx, name):
        return np.array([float(getattr(trees[k], name, np.nan)) for k in idx], dtype=float)

    i, j = m.pairs[:, 0][keep], m.pairs[:, 1][keep]
    dbh, dbh_se = _pick((ma["dbh"][i], ma["dbh_se"][i]), attr(m.trees_a, i, "dbh"))
    h, h_se = _pick((ma["height"][i], ma["height_se"][i]), attr(m.trees_a, i, "height"))
    surv = np.column_stack([dbh, dbh_se, h, h_se, c["d_dbh"][keep], c["d_dbh_se"][keep],
                            c["d_height"][keep], c["d_height_se"][keep]]).reshape(-1, 8)

    def singles(meas, trees, idx):
        idx = np.asarray(idx, dtype=np.int64)
        d, dse = _pick((meas["dbh"][idx], meas["dbh_se"][idx]), attr(trees, idx, "dbh"))
        hh, hse = _pick((meas["height"][idx], meas["height_se"][idx]), attr(trees, idx, "height"))
        return np.column_stack([d, dse, hh, hse]).reshape(-1, 4)

    dead = singles(ma, m.trees_a, np.flatnonzero(m.status_a == "dead"))
    rec = singles(mb, m.trees_b, np.flatnonzero(m.status_b == "recruit"))
    n_amb = int(amb.sum()) + int(np.sum(np.isin(m.status_a, ["merged", "split"]))) \
        + int(np.sum(np.isin(m.status_b, ["merged", "split"])))
    rows = _core.change_plot_summary(np.ascontiguousarray(surv), np.ascontiguousarray(dead), np.ascontiguousarray(rec),
                                     n_amb, float(area), float(years), int(n_draws), float(confidence),
                                     float(form_factor), float("nan") if wood_density is None else float(wood_density),
                                     int(seed))
    settings = {"area": area, "years": years, "n_draws": n_draws, "confidence": confidence,
                "form_factor": form_factor, "wood_density": wood_density, "seed": seed}
    return PlotSummary({name: Estimate(est, lo, hi, unit) for name, unit, est, lo, hi in rows}, settings)
