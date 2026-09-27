# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Tree-level processing: stem detection, DBH, tree height, segmentation, crowns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = [
    "Tree", "fit_circle", "fit_circle_ransac", "detect_stems", "prune_trees", "basal_area",
    "dbh_profile",
    "merge_branches", "segment_trees", "tree_heights", "crown_metrics", "crown_metrics_all",
    "convex_hull_area", "crown_shape", "detect_buttress",
]


@dataclass
class Tree:
    """A detected tree and its stem fit.

    Lengths are metres and positions are in the cloud's frame.

    Attributes
    ----------
    tree_id
        1..n; matches the labels of :func:`segment_trees`.
    x, y
        Stem centre at breast height.
    dbh
        Stem diameter at breast height (m) from the taper fit.
    height
        Filled by :func:`tree_heights`; NaN until then.
    n_points
        Stem points after detection; all points after :func:`tree_heights`.
    n_slices
        Layers the stem was found in.
    rmse
        Circle-fit residual (m).
    lean_deg
        Stem lean from vertical (degrees).
    extra
        Free-form per-tree values, written out by :meth:`as_dict`.
    """

    tree_id: int
    x: float
    y: float
    dbh: float = float("nan")
    height: float = float("nan")
    n_points: int = 0
    inlier_fraction: float = float("nan")
    """Fraction of the stem circumference observed (angular coverage)."""
    n_slices: int = 0
    rmse: float = float("nan")
    lean_deg: float = float("nan")
    quality: float = float("nan")
    """0-1 confidence combining fit residual, coverage and slice support."""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Flat dict of every field, with ``extra`` merged in; one row of a tree table.

        Examples
        --------
        >>> import pandas as pd
        >>> table = pd.DataFrame([t.as_dict() for t in stems])
        """
        return {
            "tree_id": self.tree_id, "x": self.x, "y": self.y, "dbh": self.dbh,
            "height": self.height, "n_points": self.n_points,
            "inlier_fraction": self.inlier_fraction, "n_slices": self.n_slices,
            "rmse": self.rmse, "lean_deg": self.lean_deg, "quality": self.quality, **self.extra,
        }

    @classmethod
    def _from_core(cls, d: dict) -> Tree:
        return cls(d["tree_id"], d["x"], d["y"], d["dbh"], d["height"], d["n_points"],
                   d["inlier_fraction"], d["n_slices"], d["rmse"], d["lean_deg"], d["quality"])

    def _to_core(self) -> dict:
        return {k: v for k, v in self.as_dict().items() if k != "extra"}


def fit_circle(xy: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares circle through 2D points.

    An algebraic fit (Kåsa 1976) refined by Levenberg-Marquardt on the
    geometric distance. Not robust to outliers; use :func:`fit_circle_ransac` on raw
    slices.

    Parameters
    ----------
    xy
        ``(N, 2)`` points, N >= 3.

    Returns
    -------
    cx, cy, r, rmse : float
        Centre, radius and RMS distance of the points from the circle.
    """
    return _core.fit_circle(np.ascontiguousarray(xy, dtype=float))


def fit_circle_ransac(xy: np.ndarray, threshold: float = 0.01, iterations: int = 200,
                      min_radius: float = 0.02, max_radius: float = 1.5,
                      seed: int = 0) -> tuple[float, float, float, np.ndarray]:
    """Circle fit that tolerates outliers (twigs, noise, a second stem).

    RANSAC (Fischler & Bolles 1981) on circles through three points, then a
    least-squares refit on the inliers.

    Parameters
    ----------
    xy
        ``(N, 2)`` points of a stem slice.
    threshold
        Inlier distance from the circle (m).
    iterations
        RANSAC trials.
    min_radius, max_radius
        Radius limits (m); candidate circles outside are rejected.
    seed
        Random seed.

    Returns
    -------
    cx, cy, r : float
        Circle refit on the inliers.
    inlier_mask : numpy.ndarray
        Boolean, one per input point.

    Raises
    ------
    ValueError
        If no circle within the radius limits is found.
    """
    return _core.fit_circle_ransac(np.ascontiguousarray(xy, dtype=float), threshold, iterations,
                                   min_radius, max_radius, seed)


def detect_stems(cloud: PointCloud, height_attr: str = "height", **params) -> list[Tree]:
    """Detect stems in a height-normalised cloud by linking circles across slices.

    The 1-5 m band is cut into 0.3 m layers every 0.25 m; each layer is
    clustered in 2-D and circles fitted by RANSAC (Fischler & Bolles 1981)
    with an angular-coverage check; circles are linked across layers into
    chains that must span ``min_slices`` (3) layers and lean under
    ``max_lean_deg`` (25). DBH is
    read from a linear taper at ``reference_height`` (1.3 m).

    Keyword parameters (defaults in brackets): ``slice_min`` [1.0],
    ``slice_max`` [5.0], ``slice_thickness`` [0.3], ``slice_step`` [0.25],
    ``reference_height`` [1.3], ``min_radius`` [0.015], ``max_radius`` [0.75],
    ``cluster_cell`` [0.06], ``min_cluster_points`` [12],
    ``max_cluster_extent`` [2.0], ``ransac_iterations`` [120],
    ``ransac_tolerance`` [0.02], ``max_circles_per_cluster`` [3],
    ``min_circle_inliers`` [10], ``min_coverage`` [0.12] (fraction of
    circumference seen; ~0.5 for single scans), ``min_arc_deg`` [0]
    (longest contiguous arc; 130+ suits merged multi-scan plots),
    ``max_circle_rmse`` [0.02], ``link_radius`` [0.2],
    ``link_radius_ratio`` [0.45], ``link_radius_abs`` [0.02], ``min_slices`` [3],
    ``max_lean_deg`` [25], ``prefilter`` [True] (keep only locally planar points
    with a near-horizontal normal, i.e. bark, before slicing; ``prefilter_k``
    [16], ``prefilter_max_nz`` [0.6], ``prefilter_max_variation`` [0.15]),
    ``seed`` [0].

    Parameters
    ----------
    cloud
        A cloud with height above ground (:func:`sylva.ground.normalize_height`).
        Merged multi-scan clouds work best; thin to ~1 cm for speed.
    height_attr
        Attribute holding heights; z is used if absent.
    **params
        The keyword parameters above.

    Returns
    -------
    list of Tree
        Sorted by ``quality`` with ids 1..n. ``height`` is NaN until
        :func:`tree_heights`.

    Notes
    -----
    Detection favours recall: some candidates are low branches, shrubs or
    duplicates. The operational sequence is :func:`merge_branches`,
    :func:`segment_trees`, :func:`tree_heights` and :func:`prune_trees`,
    which removes candidates without a canopy above them. Benchmarked on
    manually segmented plots in *Benchmarks > Tree detection*.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    found = _core.detect_stems(cloud.xyz, h, **params)
    return [Tree._from_core(t) for t in found]


def prune_trees(trees: list[Tree], labels: np.ndarray, min_height: float = 3.0,
                merge_radius: float = 0.2, max_dbh: float | None = None,
                min_quality_short: float = 0.0,
                short_slices: int = 4) -> tuple[list[Tree], np.ndarray]:
    """Drop short candidates and merge duplicates after segmentation.

    Trees lower than ``min_height`` (from :func:`tree_heights`; NaN, i.e. no
    points, counts as low) are removed; of any two stems closer than
    ``merge_radius`` the one with more points survives and absorbs the
    other's points. ``min_quality_short`` additionally drops candidates
    supported by fewer than ``short_slices`` layers whose ``quality`` is
    below it (0.15 helps in conifer stands with many low branches, at a
    small recall cost in dense rainforest).

    Parameters
    ----------
    trees
        Trees with ``height`` and ``n_points`` from :func:`tree_heights`.
    labels
        Point labels from :func:`segment_trees`.
    min_height
        Minimum tree height (m).
    merge_radius
        Candidates closer together than this are treated as one stem. A
        coppice stool or a low fork puts real stems 0.3 m apart, so this is
        deliberately small; raise it where a single stem is scanned twice.
        Stems closer than this (m) are merged.
    max_dbh
        Drop stems wider than this (m), e.g. to remove walls or rocks.
    min_quality_short
        Minimum ``quality`` for short-chain candidates.
    short_slices
        Candidates with fewer layers than this count as short.

    Returns
    -------
    trees : list of Tree
        Survivors sorted by DBH (largest first), ids renumbered 1..n,
        ``n_points`` recounted.
    labels : numpy.ndarray
        Labels remapped to the new ids; points of dropped trees are -1.
    """
    kept, out_labels = _core.prune_trees([t._to_core() for t in trees], np.ascontiguousarray(labels, dtype=np.int64),
                                         float(min_height), float(merge_radius),
                                         None if max_dbh is None else float(max_dbh), float(min_quality_short),
                                         int(short_slices))
    out = []
    for i, d in kept:
        t = Tree._from_core(d)
        t.extra = dict(trees[i].extra)
        out.append(t)
    return out, out_labels


def basal_area(trees, area: float, min_dbh: float = 0.0) -> float:
    """Basal area of a plot (m²/ha): the stems' cross-sections at breast height.

    ``sum(pi * (dbh / 2) ** 2) / area * 1e4`` over stems with ``dbh >=
    min_dbh``; stems without a DBH (NaN) are left out. Restrict ``trees`` to
    the plot first (e.g. stems within the plot radius) so that stems and
    ``area`` cover the same ground.

    Parameters
    ----------
    trees
        Trees (DBH from :func:`detect_stems`, usually after
        :func:`prune_trees`), or their DBHs (m) as an array.
    area
        Plot area (m²), e.g. ``np.pi * radius**2`` for a circular plot.
    min_dbh
        Smallest DBH counted (m); 0.1 is a common inventory threshold.

    Returns
    -------
    float
        Basal area in m²/ha.

    Examples
    --------
    >>> radius = 50.0
    >>> in_plot = [t for t in stems if np.hypot(t.x, t.y) <= radius]
    >>> trees.basal_area(in_plot, np.pi * radius**2, min_dbh=0.1)
    """
    if not area > 0:
        raise ValueError(f"area must be positive, got {area}")
    dbh = np.array([getattr(t, "dbh", t) for t in np.ravel(np.asarray(trees, dtype=object))],
                   dtype=float)
    return _core.basal_area(dbh, float(area), float(min_dbh))


def dbh_profile(cloud: PointCloud, center_xy, height_attr: str = "height",
                heights: np.ndarray | None = None, slice_thickness: float = 0.1,
                search_radius: float = 0.75) -> np.ndarray:
    """Stem diameter at a series of heights (a taper curve).

    A RANSAC circle is fitted to each thin slice around the stem.

    Parameters
    ----------
    cloud
        A height-normalised cloud.
    center_xy
        Stem position, e.g. ``(tree.x, tree.y)``.
    height_attr
        Attribute holding heights.
    heights
        Heights to measure at (m); 0.5 to 10 m every 0.5 m if None.
    slice_thickness
        Slice thickness (m).
    search_radius
        Only points within this horizontal distance (m) of ``center_xy``.
        Keep it below the spacing to the next stem.

    Returns
    -------
    numpy.ndarray
        ``(n, 2)`` rows of ``(height, diameter)`` in m; diameter is NaN where
        a slice has fewer than 10 points or no circle fits.
    """
    if heights is None:
        heights = np.arange(0.5, 10.5, 0.5)
    heights = np.ascontiguousarray(heights, dtype=float)
    h = np.ascontiguousarray(cloud.heights(height_attr))
    d = _core.dbh_profile(cloud.xyz, h, float(center_xy[0]), float(center_xy[1]), heights,
                          slice_thickness, search_radius)
    return np.column_stack([heights, d])


def segment_trees(cloud: PointCloud, trees: list[Tree], height_attr: str = "height",
                  k: int = 6, max_edge: float = 1.0, voxel_size: float = 0.03,
                  seed_height: float = 1.5, seed_radius: float = 0.25, seed_ring: bool = True,
                  power: float = 6.0,
                  angle_penalty: bool = True, gravity: float = 0.0,
                  cut_above_ground: float = 0.25, height_prior: bool = True,
                  height_prior_radius: float = 1.5, height_prior_power: float = 1.0,
                  low_height: float = 0.5,
                  low_radius: float = 1.0, wood_costs: bool = False,
                  wood_k: int = 20, wood_threshold: float = 0.9, understorey_height: float = 10.0,
                  understorey_band: float = 0.5) -> np.ndarray:
    """Assign each point to a stem by least-cost path through a directed kNN
    graph (multi-source Dijkstra from stem seeds), after raycloudtools'
    ``rayextract trees`` (Devereux et al. 2026).

    The edge cost is ``d ** power`` times an angle penalty
    ``min(exp(0.046 * deg_from_vertical), 100)`` -- climbing is free,
    horizontal edges cost x63 and downward ones x100 -- so paths run up
    through a tree instead of leaking across the ground or understorey.
    ``gravity`` adds raycloudtools' term ``1 + g * lateral**2`` from the seed
    (0.3 favours vertical trees). ``height_prior`` scales each tree's costs
    by ``1 / height`` (raycloudtools' per-root scaling) so tall trees win
    contested crown points over understorey stems. ``wood_costs`` classifies
    graph nodes as wood by local anisotropy and applies class factors
    (wood->wood x0.1, leaf->leaf x20, wood->leaf x1000) so foliage cannot
    bridge neighbouring trees. Points below ``cut_above_ground`` and
    unreachable points get ``-1``, and so do points below ``low_height``
    further than ``low_radius`` (or 1.5 DBH) from their tree's base: ground
    remnants, litter and understorey reached along the surface would
    otherwise be modelled as part of the tree. Returns ``tree_id`` per
    point.

    The graph is built on a ``voxel_size`` downsample; labels propagate to the
    full cloud by nearest graph node (set 0 to use every point).

    Parameters
    ----------
    cloud
        A height-normalised cloud of the plot.
    trees
        Stems from :func:`detect_stems` (after :func:`merge_branches`).
    height_attr
        Attribute holding heights.
    k
        Neighbours per graph node.
    max_edge
        Longest graph edge (m); gaps wider than this are not crossed.
    voxel_size
        Graph resolution (m); 0 uses every point (slow on plots).
    seed_height, seed_radius
        Graph nodes within ``seed_radius`` of a stem and below
        ``seed_height`` start that tree's search.
    power
        Exponent on edge length; higher prefers many short steps, so paths
        stay inside a crown rather than jumping the gaps between neighbouring
        crowns. 6 was chosen on the validation areas of the Cherlet et al.
        (2026) benchmark, where it raised F1 over the earlier 4 on every plot
        with interlocking crowns (see *Benchmarks > Tree detection*).
    angle_penalty
        Apply the angle factor above.
    gravity
        Lateral-distance term; 0 disables it.
    cut_above_ground
        Points below this height (m) are left unassigned.
    height_prior, height_prior_radius
        Scale costs by 1 / tree height; tree height is estimated from points
        within ``height_prior_radius`` of the stem.
    low_height, low_radius
        Near-ground points farther than ``low_radius`` from their stem are
        unassigned.
    wood_costs, wood_k, wood_threshold
        Enable wood/leaf edge factors; ``wood_k`` neighbours and
        ``wood_threshold`` anisotropy classify nodes.
    understorey_height, understorey_band
        Let the understorey compete, after raycloudtools, where every point's
        path runs to the ground and small plants keep their own. Graph nodes
        up to ``understorey_band`` above ``cut_above_ground`` and more than
        ``max(low_radius, 1.5 DBH)`` from every stem become extra sources,
        with path costs scaled as for a tree ``understorey_height`` tall
        (with ``height_prior``). Grass, shrubs and saplings around a stem go
        to them and are left unassigned instead of joining the tree. 0
        disables. On the CHERLET test blocks the default raises F1 at
        Litchfield from 0.73 to 0.83 and cuts the share of tree points that
        are really understorey from 17 % to 6 %; scales above about 20 start
        to take points from real trees.

    Returns
    -------
    numpy.ndarray
        int64 per point: the ``tree_id`` of its tree, or -1.

    Notes
    -----
    Accuracy against manually segmented plots is in *Benchmarks > Tree
    detection*; the losses are crown leakage between interlocking
    neighbours. Labels that have to be right can be corrected by hand in
    Segfix (https://github.com/tim-devereux/segfix), which reads a
    ``tree_id`` column from LAS/LAZ and writes it back in place; map -1 to 0
    and use int32 first, because Segfix reads a negative id as noise.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return _core.segment_trees(cloud.xyz, h, [t._to_core() for t in trees], k, max_edge,
                               voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty,
                               gravity, cut_above_ground, height_prior, height_prior_radius,
                               height_prior_power,
                               low_height, low_radius, wood_costs, wood_k, wood_threshold,
                               understorey_height, understorey_band)


def merge_branches(cloud: PointCloud, trees: list[Tree], height_attr: str = "height",
                   ground_height: float = 0.5, trunk_scale: float = 1.5, trunk_min: float = 0.15,
                   search_radius: float = 6.0, **graph_params) -> tuple[list[Tree], np.ndarray]:
    """Drop candidates that are branches or secondary stems of another candidate.

    Every graph node's least-cost path to the ground is traced (as in
    raycloudtools; Devereux et al. 2026); a candidate whose seed routes to the ground through
    another candidate's trunk (within ``max(trunk_scale * radius,
    trunk_min)`` of its axis, below its seed height) is merged into it.
    ``graph_params`` are the :func:`segment_trees` graph settings. Returns
    ``(surviving trees, merged_into)`` where ``merged_into[i]`` is the id the
    i-th input tree now belongs to. Call before :func:`segment_trees`.

    Parameters
    ----------
    cloud
        A height-normalised cloud of the plot.
    trees
        Candidates from :func:`detect_stems`.
    height_attr
        Attribute holding heights.
    ground_height
        Graph nodes below this height (m) count as ground.
    trunk_scale, trunk_min
        Width of the trunk zone around a candidate's axis, in stem radii and
        its minimum (m).
    search_radius
        Only candidates within this distance (m) are compared.
    **graph_params
        :func:`segment_trees` graph settings (``k``, ``voxel_size``, ...).

    Returns
    -------
    trees : list of Tree
        The surviving candidates.
    merged_into : numpy.ndarray
        For each input tree, the id it now belongs to (its own if kept).
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    kept, merged = _core.merge_branches(cloud.xyz, h, [t._to_core() for t in trees],
                                        ground_height=ground_height, trunk_scale=trunk_scale,
                                        trunk_min=trunk_min, search_radius=search_radius,
                                        **graph_params)
    return [Tree._from_core(t) for t in kept], merged


def tree_heights(cloud: PointCloud, labels: np.ndarray, trees: list[Tree],
                 height_attr: str = "height", percentile: float = 100.0) -> list[Tree]:
    """Set each tree's height and point count from the segmentation.

    Parameters
    ----------
    cloud
        The cloud that was segmented.
    labels
        Point labels from :func:`segment_trees`.
    trees
        Trees to update; modified in place and returned.
    height_attr
        Attribute holding heights.
    percentile
        Height percentile of the tree's points; 100 is the top point, 99
        is more robust to stray points above the crown.

    Returns
    -------
    list of Tree
        The same objects, with ``height`` (NaN if a tree has no points) and
        ``n_points`` set.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    out = _core.tree_heights(h, np.ascontiguousarray(labels, dtype=np.int64),
                             [t._to_core() for t in trees], percentile)
    for t, d in zip(trees, out, strict=True):
        t.height = d["height"]
        t.n_points = d["n_points"]
    return trees


def crown_metrics(cloud: PointCloud, labels: np.ndarray, tree_id: int,
                  height_attr: str = "height", crown_base_fraction: float = 0.1) -> dict:
    """Crown size of one segmented tree.

    Crown base is found from a height histogram of the tree's points in
    0.5 m bins: the first bin, from a quarter of the tree height upwards,
    holding at least ``crown_base_fraction`` of the fullest bin.

    Parameters
    ----------
    cloud
        The cloud that was segmented.
    labels
        Point labels from :func:`segment_trees`.
    tree_id
        Tree to measure.
    height_attr
        Attribute holding heights.
    crown_base_fraction
        Density threshold for the crown base (0-1).

    Returns
    -------
    dict
        ``crown_area`` (convex hull of the points above crown base, m²),
        ``crown_base_height`` (m), ``crown_depth`` (top minus base, m) and
        ``crown_diameter`` (diameter of a circle of the same area, m). Empty
        if the tree has fewer than 4 points.

    See Also
    --------
    crown_shape : volume, surface and asymmetry.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    m = _core.crown_metrics(cloud.xyz, h, np.ascontiguousarray(labels, dtype=np.int64), tree_id,
                            crown_base_fraction)
    return m or {}


def crown_metrics_all(cloud: PointCloud, labels: np.ndarray, height_attr: str = "height",
                      crown_base_fraction: float = 0.1) -> dict[int, dict]:
    """:func:`crown_metrics` for every tree in one pass (parallel).

    Parameters
    ----------
    cloud
        The cloud that was segmented.
    labels
        Point labels from :func:`segment_trees`; -1 is ignored.
    height_attr
        Attribute holding heights.
    crown_base_fraction
        Density threshold for the crown base.

    Returns
    -------
    dict
        ``{tree_id: metrics}``; trees with fewer than 4 points are missing.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return _core.crown_metrics_all(cloud.xyz, h, np.ascontiguousarray(labels, dtype=np.int64),
                                   crown_base_fraction)


def _count_ridges(persistence: np.ndarray, level: float = 0.6, dip: float = 0.2) -> int:
    """Flanges around the stem: runs of angle where protrusions persist, split
    where persistence dips by ``dip`` between two peaks (neighbouring flanges)."""
    return _core.count_ridges(np.ascontiguousarray(persistence, dtype=float), float(level), float(dip))


def detect_buttress(cloud: PointCloud, base_xy=None, height_attr: str = "height",
                    max_radius: float = 4.0, slice_height: float = 0.1, max_height: float = 6.0,
                    low: float = 1.0, bins: int = 36, max_circle_fit: float = 0.55,
                    min_ridges: int = 2, bark_only: bool = True, voxel: float = 0.02) -> dict:
    """Does a stem have buttresses, and how high do they reach?

    Flanges are thin walls radiating from the stem, continuous from the
    ground up to where they merge into the trunk. Two signals capture that:

    - **A circle stops explaining the base.** A RANSAC circle is fitted to
      every ``slice_height`` slice; on a round stem it explains most of the
      bark points at every height, on a buttressed one only a small share
      near the ground.
    - **Protrusions persist at the same angle.** Around the stem centre,
      angular bins holding points well outside the stem radius
      (``1.4 r + 0.1 m``) in at least 60 % of the slices below ``low`` are
      ridges; neighbouring ridges are split where persistence dips. Clutter
      (grass, shrubs, litter) does not line up from slice to slice.

    With ``bark_only`` (the default), only bark-like points take part: locally
    planar (planarity >= 0.4 over 20 neighbours) with a normal within 60
    degrees of horizontal. Flanges and round bark are both vertical surfaces;
    grass, shrubs and resprouts clumped around a stem in plot data are not,
    and without this filter they read as flanges.

    A tree is buttressed when the median circle fit below ``low`` is under
    ``max_circle_fit`` and there are at least ``min_ridges`` ridges. On 97
    visually labelled harvest trees (Cameroon, Peru, Guyana, Indonesia,
    Wytham) this rule gets 94.8 % right in leave-one-out (26 of 29
    buttressed trees found, 2 false alarms in 68).

    Parameters
    ----------
    cloud
        One tree's points (or a plot's; only points within ``max_radius`` of
        the stem are used), with height above ground.
    base_xy
        Stem centre; the median of the points 2.5-3.5 m up (1.2-1.8 m if
        those are too few) if None.
    height_attr
        Attribute holding height above ground.
    max_radius
        Horizontal reach from the stem centre (m).
    slice_height
        Slice thickness (m).
    max_height
        Highest slice (m); also the highest possible top.
    low
        Top of the base zone the decision looks at (m).
    bins
        Angular bins for the ridges.
    max_circle_fit, min_ridges
        Decision thresholds.
    bark_only
        Use only locally planar, near-vertical surface points.
    voxel
        The points are thinned to this spacing first (m), so the result does
        not depend on the density they come at; 0 keeps every point.

    Returns
    -------
    dict
        ``buttressed`` (bool); ``base_circle_fit`` and ``stem_circle_fit``
        (median share of bark points a circle explains below ``low`` and
        above 2 m); ``stem_radius`` (m, from the round slices above 2 m);
        ``ridges`` and ``ridge_share`` (share of the angle with persistent
        protrusions); ``spread`` (95th percentile distance of the base points
        from the centre, in stem radii); ``top`` (m above ground, where a
        circle explains the stem again, NaN if not buttressed); ``centre``.

    See Also
    --------
    sylva.qsm.buttress_mesh : rebuild the base once it is known to be buttressed.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr), dtype=float)
    base = None if base_xy is None else tuple(float(v) for v in np.asarray(base_xy, dtype=float)[:2])
    return _core.detect_buttress(np.ascontiguousarray(cloud.xyz, dtype=float), h, base, float(max_radius),
                                 float(slice_height), float(max_height), float(low), int(bins),
                                 float(max_circle_fit), int(min_ridges), bool(bark_only), float(voxel))


def convex_hull_area(xy: np.ndarray) -> float:
    """Area of the convex hull of 2D points.

    Parameters
    ----------
    xy
        ``(N, 2)`` points; non-finite rows are ignored.

    Returns
    -------
    float
        Area in squared input units (NaN for fewer than 3 distinct points).
    """
    return _core.convex_hull_area(np.ascontiguousarray(xy, dtype=float))


def crown_shape(points, base_xy=None, crown_base: float | None = None, slice_height: float = 0.5) -> dict:
    """Crown shape from a tree's points (or any 3-D outline of the crown).

    Points below ``crown_base`` (an absolute z) are ignored. Returns the
    vertical projection (convex hull ``projected_area``, equivalent
    ``diameter``, ``max_width``), ``volume`` and ``surface`` of convex hulls
    stacked every ``slice_height`` metres (they follow the crown's taper,
    unlike one 3-D hull), ``base_height`` and ``top_height`` (z of the lowest
    and highest crown point), and the horizontal ``offset`` of the crown's
    centroid from ``base_xy`` (the stem; the lowest point when ``None``),
    its ``offset_direction`` (deg, counter-clockwise from +x) and
    ``asymmetry`` (offset over the equivalent crown radius).

    Parameters
    ----------
    points
        A :class:`~sylva.PointCloud` or ``(N, 3)`` array of one tree, e.g.
        ``cloud[labels == tree_id]`` or points sampled from a QSM's leaves.
    base_xy
        Stem position the offset is measured from.
    crown_base
        Absolute z of the crown base; everything is crown if None.
    slice_height
        Thickness (m) of the stacked hull slices.

    Returns
    -------
    dict
        The keys above, lengths in m, area m², volume m³.
    """
    xyz = np.ascontiguousarray(points.xyz if isinstance(points, PointCloud) else points, dtype=float)
    base = None if base_xy is None else (float(base_xy[0]), float(base_xy[1]))
    return _core.crown_shape(xyz, base, float("-inf") if crown_base is None else float(crown_base), float(slice_height))
