# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change between the quantitative structure models of two epochs.

:func:`compare_qsms` compares two cylinder models of one tree, the epochs
aligned beforehand: the stem radius change by height (the taper increment),
branch matching (matched branches with their growth, lost and new branches),
volume change by branch order, and height, DBH and crown change.

Every quantity is labelled trusted only where both models fitted that part
to points. A cylinder counts as measured when its ``n_points`` is positive,
as in :meth:`sylva.qsm.QSM.metrics`; the rest of a model comes from the taper
and pipe-model priors, and a change there is reported, but as untrusted.
With a ray-traced grid of the later epoch (:func:`sylva.voxels.ray_voxelize`),
a branch missing from the later model is lost only where the later scans saw
its space empty; where they did not see it, it is unobserved.

:func:`compare_plot_qsms` does the same for every tree of a plot, given a tree
match (survivor pairs, deaths and recruits), and totals the changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..qsm import QSM, PlotQSMs

__all__ = ["QSMChange", "PlotQSMChange", "compare_qsms", "compare_plot_qsms"]


def _settings(height_step, max_base_distance, max_angle, parent_penalty, direction_reach, min_measured, min_fits,
              radius_sigma, clip, min_observed, max_filled, top_band, crown_branch_length, crown_slice) -> dict:
    """The settings as the core reads them; the core checks their ranges."""
    if isinstance(min_fits, bool) or int(min_fits) != min_fits:
        raise ValueError(f"min_fits must be a whole number, got {min_fits!r}")
    d = {"height_step": height_step, "max_base_distance": max_base_distance, "max_angle": max_angle,
         "parent_penalty": parent_penalty, "direction_reach": direction_reach, "min_measured": min_measured,
         "radius_sigma": radius_sigma, "clip": clip, "min_observed": min_observed, "max_filled": max_filled,
         "top_band": top_band, "crown_branch_length": crown_branch_length, "crown_slice": crown_slice}
    out = {k: float(v) for k, v in d.items()}
    out["min_fits"] = int(min_fits)
    return out


def _cylinders(model, name: str) -> np.ndarray:
    """A model's ``(n, 12)`` float array; an error for anything else."""
    if isinstance(model, QSM):
        c = model.cylinders
    else:
        c = np.asarray(model, dtype=np.float64)
        if c.ndim != 2 or (c.size and c.shape[1] != 12):
            raise ValueError(f"{name} must be a QSM or an (n, 12) cylinder array, got shape {c.shape}")
        c = c.reshape(-1, 12)
    if not np.isfinite(c[:, :8]).all():
        raise ValueError(f"{name} has non-finite cylinder coordinates or radii")
    return np.ascontiguousarray(c, dtype=np.float64)


def _grid(grid, name: str):
    """A ray-traced grid as ``(origin, voxel_size, state)`` for the core."""
    if grid is None:
        return None
    try:
        origin, size, state = grid.origin, grid.voxel_size, grid["state"]
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a RayVoxelGrid from sylva.voxels.ray_voxelize") from e
    state = np.asarray(state)
    if state.ndim != 3:
        raise ValueError(f"{name} state must be (nz, ny, nx), got shape {state.shape}")
    return ([float(v) for v in origin], float(size), np.ascontiguousarray(state, dtype=np.uint8))


@dataclass
class QSMChange:
    """Change between two models of one tree; from :func:`compare_qsms`.

    Model ``a`` is the earlier epoch and ``b`` the later; every change is
    ``b - a``. Lengths m, volumes m³, angles degrees.

    Attributes
    ----------
    volume_a, volume_b
        Whole-tree wood volume of each model.
    trusted_change, untrusted_change
        ``volume_b - volume_a`` split in two: the part both models measured
        (stem bins and branches fitted to points in both, lost branches
        confirmed empty, new branches measured) and the rest. They sum to
        :attr:`change`.
    trusted_sigma
        Standard uncertainty of ``trusted_change``, from the scatter of the
        measured radii (stem bins and branches) and ``radius_sigma``.
    measured_volume_a, measured_volume_b
        Share of each model's volume in measured cylinders.
    orders
        Volume change by branch order (0 the stem): arrays ``order``,
        ``volume_a``, ``volume_b``, ``trusted_change`` and
        ``untrusted_change``. The stem's trusted part is the sum over the
        trusted :attr:`taper` bins; a branch counts under its order in the
        earlier model (the later one for new branches).
    taper
        The stem profile, one entry per height bin: ``z0`` and ``z1`` (bin
        above the base of model ``a``), ``radius_a`` and ``radius_b`` (a line
        through the measured radii, at the bin centre), ``increment``
        (``radius_b - radius_a``), ``sigma`` (its standard uncertainty: the
        residual scatter about both lines and ``radius_sigma``, not divided
        by the number of cylinders, since neighbouring fits are smoothed
        together), ``n_fits_a`` / ``n_fits_b`` (measured cylinders),
        ``measured_a`` / ``measured_b`` (measured share of the stem length),
        ``volume_a`` / ``volume_b``, ``fitted`` (both models fitted the bin
        to points: ``min_fits`` and ``min_measured``) and ``trusted``
        (fitted, and within ``clip`` of its uncertainty from the mean
        increment; a fitted bin far out of line is almost always a radius the
        QSM regularised, such as a stem tip tapered to its apex).
    taper_increment, taper_sigma, n_taper_bins
        The mean radius increment of the stem over the trusted bins, weighted
        by ``1 / sigma**2``, its standard uncertainty (the weighted-mean
        error, scaled up by the Birge ratio when the bins scatter more than
        their uncertainties allow) and the number of bins used; NaN without
        trusted bins.
    matched
        One entry per matched branch pair (the stem pair included):
        ``id_a``, ``id_b``, ``order_a``, ``order_b``, ``base_distance``,
        ``angle``, ``parent_consistent``, ``length_a``, ``length_b``,
        ``volume_a``, ``volume_b``, ``mean_radius_a``, ``mean_radius_b``,
        ``tip_shift`` (distance between the tips), ``measured_a``,
        ``measured_b``, ``volume_sigma`` and ``trusted``.
    lost, new
        Branches of ``a`` without a match in ``b``, and of ``b`` without one
        in ``a``: ``id``, ``order``, ``parent``, ``base_x/y/z``, ``length``,
        ``volume``, ``measured``, ``status``, ``observed_share`` and
        ``filled_share`` (NaN without a grid), ``volume_sigma`` and
        ``trusted``. ``status`` is ``"lost"`` (or ``"new"``);
        ``"unobserved"`` when the other epoch's grid did not observe the
        branch's space; ``"present"`` when that space holds returns (the
        branch is there but the other model lacks it).
    height_a, height_b, height_trusted
        Tree heights, and whether a measured cylinder reaches within
        ``top_band`` of the top in both models.
    dbh_a, dbh_b, dbh_trusted
        Stem diameter at 1.3 m, and whether its bin is trusted.
    crown_area_a, crown_area_b, crown_volume_a, crown_volume_b, crown_trusted
        Projected area and stacked-hull volume of the crown outlined by the
        branches (:meth:`sylva.qsm.QSM.metrics`); trusted when both models
        measured at least ``min_measured`` of their branch length and no
        unmatched branch is unobserved.
    base_z
        Height of the base both stem profiles are measured from.
    """

    volume_a: float
    volume_b: float
    trusted_change: float
    untrusted_change: float
    trusted_sigma: float
    measured_volume_a: float
    measured_volume_b: float
    orders: dict[str, np.ndarray]
    taper: dict[str, np.ndarray]
    taper_increment: float
    taper_sigma: float
    n_taper_bins: int
    matched: dict[str, np.ndarray]
    lost: dict[str, np.ndarray]
    new: dict[str, np.ndarray]
    height_a: float
    height_b: float
    height_trusted: bool
    dbh_a: float
    dbh_b: float
    dbh_trusted: bool
    crown_area_a: float
    crown_area_b: float
    crown_volume_a: float
    crown_volume_b: float
    crown_trusted: bool
    base_z: float

    @classmethod
    def _from_core(cls, d: dict) -> QSMChange:
        for part in ("lost", "new"):
            d[part]["status"] = np.asarray(d[part]["status"], dtype=str)
        return cls(**{f: d[f] for f in cls.__dataclass_fields__})

    @property
    def change(self) -> float:
        """Total volume change, ``volume_b - volume_a`` (m³)."""
        return self.volume_b - self.volume_a

    def branches(self, status: str) -> dict[str, np.ndarray]:
        """Unmatched branches of one status.

        Parameters
        ----------
        status
            ``"lost"``, ``"unobserved"``, ``"present"`` (branches of model
            ``a``) or ``"new"``.

        Returns
        -------
        dict
            The columns of :attr:`lost` (or :attr:`new`) for those branches.

        Raises
        ------
        ValueError
            For an unknown status.
        """
        if status not in ("lost", "unobserved", "present", "new"):
            raise ValueError(f"unknown status {status!r}; use lost, unobserved, present or new")
        src = self.new if status == "new" else self.lost
        keep = src["status"] == status
        return {k: np.asarray(v)[keep] for k, v in src.items()}

    def summary(self) -> dict:
        """Headline numbers with units in the keys.

        Returns
        -------
        dict
            ``volume_a_m3``, ``volume_b_m3``, ``change_m3``,
            ``trusted_change_m3``, ``untrusted_change_m3``,
            ``trusted_sigma_m3``, ``taper_increment_m``, ``taper_sigma_m``,
            ``height_change_m``, ``dbh_change_m``, ``n_matched``, and the
            unmatched branches by status (``n_lost``, ``n_unobserved``,
            ``n_present``, ``n_new``).
        """
        lost, new = self.lost["status"], self.new["status"]
        return {
            "volume_a_m3": self.volume_a,
            "volume_b_m3": self.volume_b,
            "change_m3": self.change,
            "trusted_change_m3": self.trusted_change,
            "untrusted_change_m3": self.untrusted_change,
            "trusted_sigma_m3": self.trusted_sigma,
            "taper_increment_m": self.taper_increment,
            "taper_sigma_m": self.taper_sigma,
            "height_change_m": self.height_b - self.height_a,
            "dbh_change_m": self.dbh_b - self.dbh_a,
            "n_matched": len(self.matched["id_a"]),
            "n_lost": int((lost == "lost").sum()),
            "n_unobserved": int((lost == "unobserved").sum() + (new == "unobserved").sum()),
            "n_present": int((lost == "present").sum() + (new == "present").sum()),
            "n_new": int((new == "new").sum()),
        }


def compare_qsms(qsm_a: QSM, qsm_b: QSM, grid_a=None, grid_b=None, *, height_step: float = 1.0,
                 max_base_distance: float = 0.5, max_angle: float = 35.0, parent_penalty: float = 1.0,
                 direction_reach: float = 1.0, min_measured: float = 0.5, min_fits: int = 3,
                 radius_sigma: float = 0.001, clip: float = 3.0, min_observed: float = 0.5, max_filled: float = 0.5,
                 top_band: float = 1.0, crown_branch_length: float = 1.0, crown_slice: float = 0.5) -> QSMChange:
    """Compare two models of one tree from two epochs.

    Both models must be in one frame (align the epochs first). The stem is
    compared by height bins above the base of ``qsm_a``: in each bin a line
    through each model's measured stem radii gives the radius at the bin
    centre, and their difference is the radius increment. Bins both models
    fitted and in line with the others (within ``clip`` uncertainties of the
    mean) are trusted, and their weighted mean is the taper increment.
    Branches (one ``branch_id`` chain each, as
    :meth:`sylva.qsm.QSM.branches`) are matched one order at a time, parents
    first: a pair is admissible when the bases are within
    ``max_base_distance`` and the directions (over ``direction_reach`` past
    the first cylinder) within ``max_angle``, costs ``distance /
    max_base_distance + angle / max_angle``, plus ``parent_penalty`` when
    the parents are not matched to each other, and the cheapest pairs are
    taken first. Unmatched branches of ``qsm_a`` are
    lost and of ``qsm_b`` new; with a grid of the other epoch, one whose
    space the grid did not observe is ``"unobserved"`` instead.

    Parameters
    ----------
    qsm_a, qsm_b
        The earlier and the later model (:class:`sylva.qsm.QSM`, or its
        ``(n, 12)`` cylinder array).
    grid_a, grid_b
        Optional :class:`sylva.voxels.RayVoxelGrid` of each epoch, from
        :func:`sylva.voxels.ray_voxelize` over that epoch's pulses (misses
        included; ``occlusion=True`` separates occluded space). ``grid_b``
        checks lost branches and ``grid_a`` new ones: each is sampled every
        half voxel along the branch's cylinder axes, leaving out the part
        inside its parent.
    height_step
        Height of the stem bins (m).
    max_base_distance, max_angle, parent_penalty, direction_reach
        Branch matching, as above (m, degrees, cost, m).
    min_measured
        Share of a stem bin's length, or of a branch's length, that must be
        measured (fitted to points) in a model for that part to be trusted.
    min_fits
        Measured stem cylinders a bin needs in each model to be trusted.
    radius_sigma
        Radius uncertainty of one fit (m), added in quadrature to the
        scatter-based uncertainties. The scatter about a line understates
        it, since the QSM smooths radii along the stem; 1 mm suits circle
        fits on TLS stems.
    clip
        Bins further than this many of their own uncertainties from the
        mean increment are not trusted (see :attr:`QSMChange.taper`).
    min_observed
        Share of a branch's samples the grid must have observed (crossed by
        a pulse or holding a return); below it the branch is unobserved.
    max_filled
        Share of the observed samples holding returns above which the branch
        is ``"present"`` rather than lost or new.
    top_band
        A model's height is trusted when a measured cylinder reaches within
        this distance of its top (m).
    crown_branch_length, crown_slice
        Crown base and slice height, as in :meth:`sylva.qsm.QSM.metrics`.

    Returns
    -------
    QSMChange

    Raises
    ------
    ValueError
        For a model that is not a QSM or ``(n, 12)`` array, non-finite
        cylinders, a grid that is not a ray-traced grid, or a setting out of
        range.

    Notes
    -----
    On a synthetic tree scanned from three positions in each epoch (the
    change detection guide), a stem thickened by 10 mm was recovered as a
    taper increment of 10.1 to 10.3 mm with a stated uncertainty of 0.6 mm
    over five noise draws; a cut limb was lost, a limb extended by 1 m was
    matched 0.95 to 1.00 m longer, and a limb hidden by foliage in the later
    epoch was unobserved in every draw.

    A cylinder is measured when its ``n_points`` is positive. The QSM keeps
    that count when it replaces a weak fit by its allometric prior, so a
    measured branch can still carry a prior radius; such a branch shows as
    trusted radius change that did not happen. Check large trusted growth
    of thin branches against ``volume_sigma``.

    Examples
    --------
    >>> grid_b = voxels.ray_voxelize(shots_b, 0.1, occlusion=True)
    >>> c = change.compare_qsms(model_2020, model_2025, grid_b=grid_b)
    >>> c.taper_increment, c.taper_sigma
    >>> c.branches("lost")["volume"].sum()
    """
    a = _cylinders(qsm_a, "qsm_a")
    b = _cylinders(qsm_b, "qsm_b")
    s = _settings(height_step, max_base_distance, max_angle, parent_penalty, direction_reach, min_measured,
                  min_fits, radius_sigma, clip, min_observed, max_filled, top_band, crown_branch_length, crown_slice)
    d = _core.change_compare_qsms(a, b, s, _grid(grid_a, "grid_a"), _grid(grid_b, "grid_b"))
    return QSMChange._from_core(d)


@dataclass
class PlotQSMChange:
    """QSM change over a plot; from :func:`compare_plot_qsms`.

    Attributes
    ----------
    rows
        One dict per tree, in the order survivors, deaths, recruits (each as
        given): ``fate`` (``"survivor"``, ``"death"``, ``"recruit"``),
        ``tree_id_a``, ``tree_id_b`` (None where the tree has none),
        ``volume_a_m3``, ``volume_b_m3``, ``change_m3``,
        ``trusted_change_m3``, ``untrusted_change_m3``, ``trusted_sigma_m3``,
        ``stem_change_m3``, ``branch_change_m3``, ``taper_increment_m``,
        ``taper_sigma_m``, ``height_a_m``, ``height_b_m``, ``n_matched``,
        ``n_lost``, ``n_unobserved``, ``n_present``, ``n_new`` and ``note``
        (why a row has no comparison). A dead tree counts its whole volume
        as lost, a recruit its whole volume as new; the measured part is the
        trusted one.
    changes
        ``{(tree_id_a, tree_id_b): QSMChange}`` for the compared survivors.
    totals
        ``n_survivors``, ``n_deaths``, ``n_recruits``, ``n_unmodelled``
        (rows without a model, left out of the sums), ``growth_m3``
        (survivors' change), ``growth_trusted_m3``, ``growth_sigma_m3``,
        ``mortality_m3`` and ``recruitment_m3`` (positive volumes),
        ``net_m3`` (growth - mortality + recruitment) and
        ``net_trusted_m3``.
    """

    rows: list[dict]
    changes: dict[tuple[int, int], QSMChange] = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def table(self) -> list[dict]:
        """The rows, ready for ``pandas.DataFrame`` or a CSV.

        Returns
        -------
        list of dict
            :attr:`rows`, copied.
        """
        return [dict(r) for r in self.rows]

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as a CSV (empty cells for NaN and missing ids).

        Parameters
        ----------
        path
            Output file.
        """
        _core.change_plot_csv(str(path), self.rows)


def _models(qsms, name: str) -> Mapping:
    if isinstance(qsms, PlotQSMs):
        return qsms.models
    if isinstance(qsms, Mapping):
        return qsms
    raise ValueError(f"{name} must be a PlotQSMs or a mapping of tree id to QSM")


def _ids(values, name: str, width: int) -> list:
    out = []
    for v in values:
        t = tuple(v) if width == 2 else (v,)
        if len(t) != width:
            raise ValueError(f"each entry of {name} must be a pair of tree ids, got {v!r}")
        out.append(tuple(int(x) for x in t) if width == 2 else int(t[0]))
    return out


def compare_plot_qsms(qsms_a, qsms_b, pairs: Iterable, deaths: Iterable = (), recruits: Iterable = (),
                      grid_a=None, grid_b=None, **settings) -> PlotQSMChange:
    """Compare the QSMs of every tree of a plot between two epochs.

    Survivors are compared with :func:`compare_qsms`; a dead tree counts
    its whole earlier volume as mortality and a recruit its whole later
    volume as recruitment. Trees run in parallel; the result does not
    depend on the number of threads.

    Parameters
    ----------
    qsms_a, qsms_b
        Models of each epoch: :class:`sylva.qsm.PlotQSMs` or
        ``{tree_id: QSM}``. A tree without a model gives a row with a note
        and no change.
    pairs
        Survivors as ``(tree_id_a, tree_id_b)`` pairs, e.g. from a tree
        match between the epochs.
    deaths
        Tree ids of epoch ``a`` that died.
    recruits
        Tree ids of epoch ``b`` that are new.
    grid_a, grid_b
        Optional ray-traced grids of the whole plot in each epoch, as for
        :func:`compare_qsms`.
    **settings
        Keyword settings of :func:`compare_qsms`.

    Returns
    -------
    PlotQSMChange

    Raises
    ------
    ValueError
        For a tree id used twice in one epoch, a malformed pair, an unknown
        setting, or anything :func:`compare_qsms` rejects.

    Examples
    --------
    >>> plot = change.compare_plot_qsms(qsms_2020, qsms_2025, pairs=[(1, 4), (2, 5)],
    ...                                 deaths=[3], recruits=[9], grid_b=grid_2025)
    >>> plot.totals["net_trusted_m3"]; plot.to_csv("qsm_change.csv")
    """
    a_models = _models(qsms_a, "qsms_a")
    b_models = _models(qsms_b, "qsms_b")
    pairs = _ids(pairs, "pairs", 2)
    deaths = _ids(deaths, "deaths", 1)
    recruits = _ids(recruits, "recruits", 1)
    used_a = [p[0] for p in pairs] + deaths
    used_b = [p[1] for p in pairs] + recruits
    for used, which in ((used_a, "a"), (used_b, "b")):
        seen = set()
        for t in used:
            if t in seen:
                raise ValueError(f"tree {t} of epoch {which} appears more than once")
            seen.add(t)
    names = set(compare_qsms.__kwdefaults__)
    unknown = set(settings) - names
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    kw = {**compare_qsms.__kwdefaults__, **settings}
    s = _settings(**kw)

    def model(models, t, name):
        m = models.get(t)
        return None if m is None else _cylinders(m, f"{name} tree {t}")

    trees = [("survivor", ia, ib, model(a_models, ia, "qsms_a"), model(b_models, ib, "qsms_b")) for ia, ib in pairs]
    trees += [("death", ia, None, model(a_models, ia, "qsms_a"), None) for ia in deaths]
    trees += [("recruit", None, ib, None, model(b_models, ib, "qsms_b")) for ib in recruits]
    rows, changes, totals = _core.change_compare_plot(trees, s, _grid(grid_a, "grid_a"), _grid(grid_b, "grid_b"))
    out = {(r["tree_id_a"], r["tree_id_b"]): QSMChange._from_core(c)
           for r, c in zip(rows, changes, strict=True) if c is not None}
    return PlotQSMChange(rows=list(rows), changes=out, totals=dict(totals))
