"""Tree-level processing: stem detection, DBH, tree height, segmentation, crowns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = [
    "Tree", "fit_circle", "fit_circle_ransac", "detect_stems", "prune_trees", "dbh_profile",
    "merge_branches", "segment_trees", "tree_heights", "crown_metrics", "crown_metrics_all",
    "convex_hull_area", "crown_shape",
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

    An algebraic fit refined by Levenberg-Marquardt on the geometric
    distance. Not robust to outliers; use :func:`fit_circle_ransac` on raw
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
    clustered in 2-D and circles fitted by RANSAC with an angular-coverage
    check; circles are linked across layers into chains that must span
    ``min_slices`` (3) layers and lean under ``max_lean_deg`` (25). DBH is
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
                merge_radius: float = 0.5, max_dbh: float | None = None,
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
    keep = [t for t in trees if t.height >= min_height  # NaN height (no points) is dropped
            and (max_dbh is None or t.dbh <= max_dbh)
            and (t.n_slices >= short_slices or t.quality >= min_quality_short)]
    keep.sort(key=lambda t: -t.n_points)
    survivors: list[Tree] = []
    absorbed: dict[int, int] = {}
    for t in keep:
        for s in survivors:
            if np.hypot(t.x - s.x, t.y - s.y) <= merge_radius:
                absorbed[t.tree_id] = s.tree_id
                break
        else:
            survivors.append(t)
    survivors.sort(key=lambda t: -t.dbh)
    new_id = {t.tree_id: i + 1 for i, t in enumerate(survivors)}
    lut = np.full(int(max(labels.max(), max((t.tree_id for t in trees), default=0))) + 2, -1,
                  dtype=np.int64)
    for old, new in new_id.items():
        lut[old] = new
    for old, target in absorbed.items():
        lut[old] = new_id[target]
    labels = np.asarray(labels, dtype=np.int64)
    out_labels = np.where(labels >= 0, lut[np.clip(labels, 0, len(lut) - 1)], -1)
    out = []
    for t in survivors:
        n = int((out_labels == new_id[t.tree_id]).sum())
        out.append(Tree(new_id[t.tree_id], t.x, t.y, t.dbh, t.height, n, t.inlier_fraction,
                        t.n_slices, t.rmse, t.lean_deg, t.quality, dict(t.extra)))
    return out, out_labels


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
                  k: int = 10, max_edge: float = 1.0, voxel_size: float = 0.05,
                  seed_height: float = 1.5, seed_radius: float = 0.5, power: float = 3.0,
                  angle_penalty: bool = True, gravity: float = 0.0,
                  cut_above_ground: float = 0.25, height_prior: bool = True,
                  height_prior_radius: float = 1.5, low_height: float = 0.5,
                  low_radius: float = 1.0, wood_costs: bool = False,
                  wood_k: int = 20, wood_threshold: float = 0.9) -> np.ndarray:
    """Assign each point to a stem by least-cost path through a directed kNN
    graph (multi-source Dijkstra from stem seeds).

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
        Exponent on edge length; higher prefers many short steps.
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
                               voxel_size, seed_height, seed_radius, power, angle_penalty,
                               gravity, cut_above_ground, height_prior, height_prior_radius,
                               low_height, low_radius, wood_costs, wood_k, wood_threshold)


def merge_branches(cloud: PointCloud, trees: list[Tree], height_attr: str = "height",
                   ground_height: float = 0.5, trunk_scale: float = 1.5, trunk_min: float = 0.15,
                   search_radius: float = 6.0, **graph_params) -> tuple[list[Tree], np.ndarray]:
    """Drop candidates that are branches or secondary stems of another candidate.

    Every graph node's least-cost path to the ground is traced (as in
    raycloudtools); a candidate whose seed routes to the ground through
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
