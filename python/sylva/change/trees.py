# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Trees matched between epochs, and their increments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from .epochs import EpochAlignment, _transform_xy

__all__ = ["TreeMatch", "match_trees", "TreeIncrements", "tree_increments"]


class _Row:
    """A tree of a table given as a dict of arrays."""

    def __init__(self, table: dict, k: int):
        for name in ("tree_id", "x", "y", "dbh", "height"):
            if name in table:
                setattr(self, name, table[name][k])
        if not hasattr(self, "tree_id"):
            self.tree_id = k + 1

    def __repr__(self) -> str:
        return f"Tree(tree_id={self.tree_id}, x={self.x:.2f}, y={self.y:.2f})"


def _as_trees(trees, what: str) -> list:
    """A list of objects with ``tree_id``, ``x``, ``y`` and optionally
    ``dbh`` and ``height``: :class:`sylva.trees.Tree` objects, or a table
    (dict of arrays) such as :attr:`sylva.synthetic.ForestEpochs.trees`."""
    if isinstance(trees, dict):
        if "x" not in trees or "y" not in trees:
            raise ValueError(f"{what} needs 'x' and 'y' columns")
        return [_Row(trees, k) for k in range(len(np.asarray(trees["x"])))]
    out = list(trees)
    for t in out:
        if not (hasattr(t, "x") and hasattr(t, "y")):
            raise ValueError(f"{what} must be trees with x and y, got {type(t).__name__}")
    return out


def _table(trees: list) -> np.ndarray:
    return np.array([[float(t.x), float(t.y), float(getattr(t, "dbh", np.nan)),
                      float(getattr(t, "height", np.nan))] for t in trees], dtype=float).reshape(-1, 4)


def _matrix(transform) -> np.ndarray | None:
    if transform is None:
        return None
    if isinstance(transform, EpochAlignment):
        return transform.transform
    m = np.asarray(transform, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be a (4, 4) matrix or an EpochAlignment, got shape {m.shape}")
    return m


@dataclass
class TreeMatch:
    """Result of :func:`match_trees`.

    Attributes
    ----------
    trees_a, trees_b
        The trees of each epoch, as given (each in its own frame).
    pairs
        ``(n, 2)`` indices into ``trees_a`` and ``trees_b`` of every
        survivor, sorted by the first.
    distance
        Horizontal distance (m) of each pair in the reference frame.
    cost
        Assignment cost of each pair.
    status_a
        Per tree of the first epoch: ``survivor``, ``dead``, ``merged``
        (found in the second epoch only as part of a neighbour's stem) or
        ``split`` (found as several stems, none of them matched to it).
    status_b
        Per tree of the second epoch: ``survivor``, ``recruit``, ``split``
        (part of a stem the first epoch saw as one with a neighbour) or
        ``merged`` (one stem standing for several of the first epoch, none
        of them matched to it).
    related_a, related_b
        The partner (for a survivor) or the stem merged into / split from,
        as an index into the other epoch's trees; -1 for deaths and recruits.
    transform
        The transform applied to the second epoch's positions (identity if
        none was given).
    alignment
        The :class:`EpochAlignment` given as ``transform``, if any.
    settings
        The settings of the call, for :func:`provenance`.
    """

    trees_a: list
    trees_b: list
    pairs: np.ndarray
    distance: np.ndarray
    cost: np.ndarray
    status_a: np.ndarray
    status_b: np.ndarray
    related_a: np.ndarray
    related_b: np.ndarray
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    alignment: EpochAlignment | None = None
    settings: dict = field(default_factory=dict)

    @property
    def survivors(self) -> list[tuple]:
        """``(tree in epoch a, tree in epoch b)`` of every survivor."""
        return [(self.trees_a[i], self.trees_b[j]) for i, j in self.pairs]

    @property
    def deaths(self) -> list:
        """Trees of the first epoch not found in the second."""
        return [t for t, s in zip(self.trees_a, self.status_a, strict=True) if s == "dead"]

    @property
    def recruits(self) -> list:
        """Trees of the second epoch not found in the first."""
        return [t for t, s in zip(self.trees_b, self.status_b, strict=True) if s == "recruit"]

    @property
    def merged(self) -> list[tuple]:
        """``(tree in epoch a, stem in epoch b it went into)`` of the trees
        of the first epoch that merged into a neighbour's stem."""
        return [(self.trees_a[i], self.trees_b[self.related_a[i]]) for i, s in enumerate(self.status_a) if s == "merged"]

    @property
    def split(self) -> list[tuple]:
        """``(stem in epoch a it came from, tree in epoch b)`` of the trees
        of the second epoch that split off a stem of the first."""
        return [(self.trees_a[self.related_b[j]], self.trees_b[j]) for j, s in enumerate(self.status_b) if s == "split"]

    def ambiguous_pairs(self) -> np.ndarray:
        """Survivor pairs involved in a merge or a split, whose increments
        compare a stem with more (or less) than itself.

        Returns
        -------
        numpy.ndarray
            One bool per row of :attr:`pairs`.
        """
        merged_into = {int(self.related_a[i]) for i, s in enumerate(self.status_a) if s == "merged"}
        split_from = {int(self.related_b[j]) for j, s in enumerate(self.status_b) if s == "split"}
        return np.array([int(j) in merged_into or int(i) in split_from for i, j in self.pairs], dtype=bool)

    def counts(self) -> dict:
        """Numbers of survivors, deaths and recruits, and of the trees of
        either epoch in a merge or a split."""
        return {"survivors": len(self.pairs), "deaths": int(np.sum(self.status_a == "dead")),
                "recruits": int(np.sum(self.status_b == "recruit")),
                "merged": int(np.sum(self.status_a == "merged") + np.sum(self.status_b == "merged")),
                "split": int(np.sum(self.status_a == "split") + np.sum(self.status_b == "split"))}


def match_trees(trees_a, trees_b, max_distance: float = 1.0, transform=None, dbh_tolerance: float = 0.35,
                max_shrink: float = 0.15, dbh_weight: float = 1.0, height_weight: float = 0.25, merge_factor: float = 0.8) -> TreeMatch:
    """Match the trees of two epochs: survivors, deaths and recruits.

    The second epoch's stem positions are moved by ``transform`` into the
    first epoch's frame, and the pairs are chosen by an optimal assignment
    (Kuhn 1955; Munkres 1957) that first matches as many trees as it can
    and then minimises the total cost ``(d / max_distance)² + dbh_weight *
    ddbh² + height_weight * dh²``, with ``ddbh`` and ``dh`` the relative
    differences (``(b - a) / max(a, b)``). A pair is allowed only within
    ``max_distance``, and only if the DBH grew by at most ``dbh_tolerance``
    or shrank by at most ``max_shrink``, so a felled tree and a young one
    grown beside it are not taken for one tree. Unlike nearest-neighbour
    matching, a neighbour's closer stem cannot take a tree's partner when the
    pair would leave the neighbour unmatched. The assignment is solved in each
    group of trees that could be matched to one another, so large plots cost
    little.

    Trees left over are checked for a merge before being called dead: when a
    stem of the second epoch is wide enough (``merge_factor`` times the
    quadratic sum of the DBHs) to stand for its partner and an unmatched
    neighbour of the first epoch, the neighbour is ``merged`` into it, as
    happens when the detector sees two touching stems as one. Splits are
    the mirror case.

    Parameters
    ----------
    trees_a, trees_b
        Trees of the two epochs (:class:`sylva.trees.Tree`, e.g. from
        :func:`sylva.trees.detect_stems` and :func:`sylva.trees.tree_heights`),
        or tables (dicts of arrays with ``x``, ``y`` and optionally
        ``tree_id``, ``dbh``, ``height``), each in its own frame.
    max_distance
        Farthest apart (m) the two stems of one tree can be.
    transform
        :class:`EpochAlignment` or ``(4, 4)`` matrix taking the second
        epoch onto the first; the epochs are taken to share a frame if None.
    dbh_tolerance
        Largest relative DBH increase of one tree; NaN DBHs are not tested.
    max_shrink
        Largest relative DBH decrease of one tree. Stems do not shrink, so
        this only absorbs the DBH error of the two detections.
    dbh_weight, height_weight
        Weights of the relative DBH and height differences in the cost.
    merge_factor
        Test for merges and splits, as above.

    Returns
    -------
    TreeMatch

    Raises
    ------
    ValueError
        If a position is not finite or a setting is out of range.

    Examples
    --------
    >>> m = change.match_trees(stems_a, stems_b, transform=alignment)
    >>> m.counts()
    {'survivors': 14, 'deaths': 2, 'recruits': 2, 'merged': 0, 'split': 0}
    """
    ta, tb = _as_trees(trees_a, "trees_a"), _as_trees(trees_b, "trees_b")
    m = _matrix(transform)
    a, b = _table(ta), _table(tb)
    if m is not None:
        b[:, :2] = _transform_xy(m, b[:, :2])
    d = _core.change_match_trees(a, b, float(max_distance), float(dbh_tolerance), float(max_shrink), float(dbh_weight),
                                 float(height_weight), float(merge_factor))
    return TreeMatch(
        trees_a=ta, trees_b=tb, pairs=d["pairs"], distance=d["distance"], cost=d["cost"],
        status_a=np.array(d["status_a"], dtype=object).astype(str), status_b=np.array(d["status_b"], dtype=object).astype(str),
        related_a=d["related_a"], related_b=d["related_b"],
        transform=np.eye(4) if m is None else m,
        alignment=transform if isinstance(transform, EpochAlignment) else None,
        settings={"max_distance": max_distance, "dbh_tolerance": dbh_tolerance, "max_shrink": max_shrink,
                  "dbh_weight": dbh_weight,
                  "height_weight": height_weight, "merge_factor": merge_factor})


@dataclass
class TreeIncrements:
    """Result of :func:`tree_increments`: one row per survivor.

    Columns (``columns[name]``, or ``increments[name]``), lengths in m:

    ``tree_id_a``, ``tree_id_b``, ``x``, ``y``
        The tree in each epoch and its position in the first.
    ``dbh_a``, ``dbh_a_se``, ``dbh_b``, ``dbh_b_se``
        DBH of each epoch at 1.3 m from a weighted line through the stem
        slices, and its standard error.
    ``d_dbh``, ``d_dbh_se``, ``d_dbh_mdi``, ``n_slices``
        DBH increment from the paired slices, its standard error, the
        minimum detectable increment at the chosen confidence, and the
        number of slices measured in both epochs.
    ``dbh_change``
        ``growth`` (at least the MDI), ``below_detection``, ``decrease``
        (a significant decrease) or ``unmeasured``.
    ``height_a``, ``height_b``, ``d_height``, ``d_height_se``, ``d_height_mdi``, ``height_change``
        Top height of each epoch, and the same for its increment.
    ``crown_area_a``, ``crown_area_b``, ``d_crown_area``, ``crown_volume_a``, ``crown_volume_b``, ``d_crown_volume``
        Crown projection area (m²) and stacked-hull volume (m³); descriptive,
        without a detection level: occlusion changes them as much as growth.
    ``implausible``
        A significant DBH decrease, or an increase above ``max_dbh_increment``.
    ``ambiguous``
        The pair is part of a merge or split (:meth:`TreeMatch.ambiguous_pairs`).
    ``flags``
        The row's warnings as text: ``unmeasured``, ``implausible``,
        ``ambiguous``, ``height_decrease``.

    Attributes
    ----------
    columns
        The table, a dict of equal-length arrays.
    match
        The :class:`TreeMatch` the rows come from.
    measures
        Per epoch, the measurement of every tree (not only survivors): a dict
        of arrays with ``dbh``, ``dbh_se``, ``taper``, ``height``,
        ``height_se``, ``crown_area``, ``crown_volume``, ``n_points`` and the
        ``(n, k)`` slice ``diameter`` and ``diameter_se``.
    slice_heights
        Heights (m) of the stem slices.
    settings
        The settings of the call, for :func:`provenance`.
    """

    columns: dict
    match: TreeMatch
    measures: list
    slice_heights: np.ndarray
    settings: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.columns["d_dbh"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def as_dict(self) -> dict:
        """The table as a dict of arrays (a copy)."""
        return {k: np.array(v, copy=True) for k, v in self.columns.items()}

    def to_pandas(self):
        """The table as a :class:`pandas.DataFrame` (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(self.columns)


def _noise(v, what: str) -> float:
    if v is None:
        return 0.0
    if hasattr(v, "summary"):
        s = v.summary()
        if "sigma_local" not in s:
            raise ValueError(f"{what}: the stem-noise result measured no slice")
        return float(s["sigma_local"])
    v = float(v)
    if not v >= 0:
        raise ValueError(f"{what} must be >= 0, got {v}")
    return v


def _labels(labels, cloud: PointCloud, what: str) -> np.ndarray:
    lab = np.ascontiguousarray(labels, dtype=np.int64).ravel()
    if len(lab) != len(cloud):
        raise ValueError(f"{what} has {len(lab)} labels for {len(cloud)} points")
    return lab


def tree_increments(match: TreeMatch, cloud_a: PointCloud, cloud_b: PointCloud, labels_a, labels_b,
                    noise_a=None, noise_b=None, registration_sigma=None, height_attr: str = "height",
                    slice_heights=None, slice_thickness: float = 0.1, search_radius: float = 0.75,
                    min_slices: int = 3, confidence: float = 0.95, max_dbh_increment: float | None = None,
                    top_points: int = 5, height_error: float = 0.02, top_radius: float = 0.3,
                    top_gap: float = 0.5, slice_correlation: float = 0.3,
                    dbh_accuracy: float = 0.002) -> TreeIncrements:
    """DBH, height and crown increments of the survivors, each with the
    smallest increment the data can detect.

    Each epoch's stems are cut into thin slices (1 to 3 m every 0.25 m by
    default) and a circle fitted to each: RANSAC, then least squares on the
    inliers, with one inlier distance for both epochs (three times the
    larger range noise, at least 1 cm). Slices far off the taper line of the
    others (a branch junction) are dropped. The DBH increment is the
    weighted mean of the paired differences at the same heights, which
    cancels the stem's own irregularity (a flattened or fluted stem has the
    same shape in both epochs). Its standard error combines

    - each slice's circle-fit precision, ``2 s / sqrt(n) / sqrt(c)`` with
      ``s`` the fit residual (at least the epoch's range noise), ``n`` the
      inlier points and ``c`` the share of the circumference seen;
    - the scatter of the differences between slices, where it exceeds that
      (Birge ratio);
    - a share ``slice_correlation`` of the slice errors that does not
      average out, because the same scanners see every slice of a stem from
      the same directions;
    - the vertical registration uncertainty times the stem taper.

    The minimum detectable increment (MDI) is that error times the normal
    quantile of ``confidence``: an increment smaller than it is reported as
    ``below_detection``, not as growth.

    Height is the highest point of the tree above its epoch's own terrain,
    continued upwards through points the segmentation left unassigned within
    ``top_radius`` of the stem (segmentation often loses a thin leader);
    vertical misregistration cancels because each epoch is normalised by its
    own terrain. Its uncertainty combines how far the next highest points
    (``top_points``) lie below the top with an assumed error of
    ``height_error`` times the height for a top that was missed: a thin
    leader that no pulse hit, or a top hidden behind the crown, leaves no
    trace in the epoch's own data. Crown area and volume are reported as
    measured.

    Parameters
    ----------
    match
        From :func:`match_trees`; its trees' ``x``, ``y`` place the slices.
    cloud_a, cloud_b
        The segmented clouds of the two epochs, each in its own frame, with
        height above ground in ``height_attr``.
    labels_a, labels_b
        Tree label of each point (:func:`sylva.trees.segment_trees`), equal
        to the trees' ``tree_id``; -1 for none.
    noise_a, noise_b
        Range noise of each epoch (m), or a :class:`sylva.quality.StemNoise`
        result whose ``sigma_local`` is used; a floor on the circle-fit
        residual. None uses the fit residual alone.
    registration_sigma
        Vertical one-sigma alignment uncertainty (m), or an
        :class:`EpochAlignment` (its ``sigma_vertical``); by default the
        alignment ``match`` was made with, else 0.
    height_attr
        Attribute holding height above ground.
    slice_heights
        Heights (m) of the stem slices; 1 to 3 m every 0.25 m if None.
    slice_thickness
        Slice thickness (m).
    search_radius
        Only points this close (m) to the stem centre enter a slice.
    min_slices
        Fewest slices measured in both epochs for an increment.
    confidence
        Two-sided confidence level of the MDI.
    max_dbh_increment
        Increments above this (m) are flagged implausible; None for no limit.
    top_points
        Highest points used for the height uncertainty.
    height_error
        One-sigma error of a top height as a fraction of the height. The
        default of 2 % covers the missed tops of the synthetic validation
        (multi-scan, 0.25 degree); calibrate it against repeated or
        destructive measurements for other scanners and stands.
    top_radius, top_gap
        Unassigned points (label -1) within ``top_radius`` (m) of the stem
        extend the tree upwards while each lies within ``top_gap`` (m) of the
        top reached so far; 0 disables.
    slice_correlation
        Correlation of the slice errors of one stem in one epoch, 0 to 1.
    dbh_accuracy
        Absolute one-sigma accuracy (m) of a single-epoch DBH beyond its
        precision (bark roughness and edge points widen every circle of a
        stem alike). It cancels in increments and is added only to
        ``dbh_a_se`` and ``dbh_b_se``, which :func:`plot_summary` uses for
        dead trees and recruits.

    Returns
    -------
    TreeIncrements

    Raises
    ------
    ValueError
        If the labels do not fit the clouds or a setting is out of range.

    Notes
    -----
    On twenty :func:`sylva.synthetic.forest_epochs` plots (range noise 3 and
    5 mm, five scans per epoch) the MDI of the DBH increment was 1.4 mm
    (median) and every one of the 280 true increments lay within the MDI of
    its measurement; with two echoes per pulse, whose displaced edge points
    widen the circles unevenly, 93 % did. See *Change detection* in the
    guide.

    Examples
    --------
    >>> inc = change.tree_increments(m, cloud_a, cloud_b, labels_a, labels_b,
    ...                              noise_a=0.003, noise_b=0.005)
    >>> inc["dbh_change"]
    """
    if not isinstance(match, TreeMatch):
        raise ValueError("match must be a TreeMatch from match_trees")
    la, lb = _labels(labels_a, cloud_a, "labels_a"), _labels(labels_b, cloud_b, "labels_b")
    sa, sb = _noise(noise_a, "noise_a"), _noise(noise_b, "noise_b")
    if registration_sigma is None:
        registration_sigma = match.alignment
    if isinstance(registration_sigma, EpochAlignment):
        reg = registration_sigma.sigma_vertical
        reg = 0.0 if not np.isfinite(reg) else float(reg)
    else:
        reg = 0.0 if registration_sigma is None else float(registration_sigma)
        if not reg >= 0:
            raise ValueError(f"registration_sigma must be >= 0, got {registration_sigma}")
    hs = np.arange(1.0, 3.01, 0.25) if slice_heights is None else np.asarray(slice_heights, dtype=float).ravel()
    if len(hs) == 0:
        raise ValueError("slice_heights is empty")
    if int(min_slices) < 1:
        raise ValueError(f"min_slices must be >= 1, got {min_slices}")

    def trees(ts):
        return np.array([[float(getattr(t, "tree_id", k + 1)), float(t.x), float(t.y)] for k, t in enumerate(ts)],
                        dtype=float).reshape(-1, 3)

    pairs = np.ascontiguousarray(match.pairs, dtype=np.int64).reshape(-1, 2)
    d = _core.change_tree_increments(
        np.ascontiguousarray(cloud_a.xyz, dtype=float), np.ascontiguousarray(cloud_a.heights(height_attr)), la,
        trees(match.trees_a), sa,
        np.ascontiguousarray(cloud_b.xyz, dtype=float), np.ascontiguousarray(cloud_b.heights(height_attr)), lb,
        trees(match.trees_b), sb,
        pairs, reg, [float(h) for h in hs], float(slice_thickness), float(search_radius), int(min_slices),
        float(confidence), float("nan") if max_dbh_increment is None else float(max_dbh_increment), int(top_points),
        float(height_error), float(top_radius), float(top_gap), float(slice_correlation), float(dbh_accuracy))
    ma, mb, inc = d["a"], d["b"], d["increments"]
    i, j = pairs[:, 0], pairs[:, 1]
    ta, tb = match.trees_a, match.trees_b
    ambiguous = match.ambiguous_pairs()
    cols = {
        "tree_id_a": np.array([getattr(ta[k], "tree_id", k + 1) for k in i], dtype=np.int64),
        "tree_id_b": np.array([getattr(tb[k], "tree_id", k + 1) for k in j], dtype=np.int64),
        "x": np.array([float(ta[k].x) for k in i]), "y": np.array([float(ta[k].y) for k in i]),
        "dbh_a": ma["dbh"][i], "dbh_a_se": ma["dbh_se"][i], "dbh_b": mb["dbh"][j], "dbh_b_se": mb["dbh_se"][j],
        "d_dbh": inc["d_dbh"], "d_dbh_se": inc["d_dbh_se"], "d_dbh_mdi": inc["d_dbh_mdi"],
        "n_slices": inc["n_slices"], "dbh_change": np.array(inc["dbh_change"], dtype=object).astype(str),
        "height_a": ma["height"][i], "height_b": mb["height"][j], "d_height": inc["d_height"],
        "d_height_se": inc["d_height_se"], "d_height_mdi": inc["d_height_mdi"],
        "height_change": np.array(inc["height_change"], dtype=object).astype(str),
        "crown_area_a": ma["crown_area"][i], "crown_area_b": mb["crown_area"][j], "d_crown_area": inc["d_crown_area"],
        "crown_volume_a": ma["crown_volume"][i], "crown_volume_b": mb["crown_volume"][j],
        "d_crown_volume": inc["d_crown_volume"], "implausible": inc["implausible"], "ambiguous": ambiguous,
    }
    flags = []
    for k in range(len(pairs)):
        f = [name for name, on in (("unmeasured", cols["dbh_change"][k] == "unmeasured"),
                                   ("implausible", cols["implausible"][k]), ("ambiguous", ambiguous[k]),
                                   ("height_decrease", cols["height_change"][k] == "decrease")) if on]
        flags.append(",".join(f))
    cols["flags"] = np.array(flags, dtype=object).astype(str) if flags else np.zeros(0, dtype=str)
    settings = {"noise_a": sa, "noise_b": sb, "registration_sigma": reg, "height_attr": height_attr,
                "slice_heights": hs.tolist(), "slice_thickness": slice_thickness, "search_radius": search_radius,
                "min_slices": min_slices, "confidence": confidence, "max_dbh_increment": max_dbh_increment,
                "top_points": top_points, "height_error": height_error,
                "top_radius": top_radius, "top_gap": top_gap, "slice_correlation": slice_correlation,
                "dbh_accuracy": dbh_accuracy}
    return TreeIncrements(columns=cols, match=match, measures=[ma, mb], slice_heights=hs, settings=settings)
