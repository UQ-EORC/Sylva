"""Tree-level processing: stem detection, DBH, tree height, segmentation, crowns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = [
    "Tree", "fit_circle", "fit_circle_ransac", "detect_stems", "prune_trees", "dbh_profile",
    "merge_branches", "segment_trees", "tree_heights", "crown_metrics", "crown_metrics_all",
    "convex_hull_area",
]


@dataclass
class Tree:
    """Summary of a detected tree."""

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
    """Geometric (Levenberg-Marquardt) circle fit: ``(cx, cy, r, rmse)``."""
    return _core.fit_circle(np.ascontiguousarray(xy, dtype=float))


def fit_circle_ransac(xy: np.ndarray, threshold: float = 0.01, iterations: int = 200,
                      min_radius: float = 0.02, max_radius: float = 1.5,
                      seed: int = 0) -> tuple[float, float, float, np.ndarray]:
    """RANSAC circle fit robust to occlusion and noise: ``(cx, cy, r, inlier_mask)``."""
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

    Returns trees sorted by ``quality`` with ids 1..n. Run
    :func:`segment_trees` and :func:`prune_trees` afterwards to drop
    candidates without a canopy above them.
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
    small recall cost in dense rainforest). Returns ``(trees, labels)`` with
    ids renumbered 1..n and dropped points labelled ``-1``.
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
    """Stem diameter at a series of heights (taper): ``(n, 2)`` of ``(h, diameter)``."""
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
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    kept, merged = _core.merge_branches(cloud.xyz, h, [t._to_core() for t in trees],
                                        ground_height=ground_height, trunk_scale=trunk_scale,
                                        trunk_min=trunk_min, search_radius=search_radius,
                                        **graph_params)
    return [Tree._from_core(t) for t in kept], merged


def tree_heights(cloud: PointCloud, labels: np.ndarray, trees: list[Tree],
                 height_attr: str = "height", percentile: float = 100.0) -> list[Tree]:
    """Fill in ``height`` and ``n_points`` for each tree from segmentation labels."""
    h = np.ascontiguousarray(cloud.heights(height_attr))
    out = _core.tree_heights(h, np.ascontiguousarray(labels, dtype=np.int64),
                             [t._to_core() for t in trees], percentile)
    for t, d in zip(trees, out, strict=True):
        t.height = d["height"]
        t.n_points = d["n_points"]
    return trees


def crown_metrics(cloud: PointCloud, labels: np.ndarray, tree_id: int,
                  height_attr: str = "height", crown_base_fraction: float = 0.1) -> dict:
    """Crown projection area (convex hull), crown base height, depth and diameter."""
    h = np.ascontiguousarray(cloud.heights(height_attr))
    m = _core.crown_metrics(cloud.xyz, h, np.ascontiguousarray(labels, dtype=np.int64), tree_id,
                            crown_base_fraction)
    return m or {}


def crown_metrics_all(cloud: PointCloud, labels: np.ndarray, height_attr: str = "height",
                      crown_base_fraction: float = 0.1) -> dict[int, dict]:
    """:func:`crown_metrics` for every tree in one pass; ``{tree_id: metrics}``."""
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return _core.crown_metrics_all(cloud.xyz, h, np.ascontiguousarray(labels, dtype=np.int64),
                                   crown_base_fraction)


def convex_hull_area(xy: np.ndarray) -> float:
    return _core.convex_hull_area(np.ascontiguousarray(xy, dtype=float))
