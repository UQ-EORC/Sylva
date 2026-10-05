# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Wood-point filtering ahead of a QSM."""

from __future__ import annotations

from .. import _core
from ..pointcloud import PointCloud


def wood_points(cloud: PointCloud, k: int = 20, threshold: float = 0.85,
                voxel_size: float | None = 0.02, medium_threshold: float = 0.75,
                scale_radius: float = 0.0, passage: bool = True, min_passage: int = 3,
                target_res: float = 0.2,
                graph_k: int = 10, max_edge: float = 1.0, base_height: float = 0.25,
                assign_dist: float = 0.05, assign_scale: float = 0.0,
                component_res: float = 0.05,
                component_min: int = 200, sor_k: int = 50, sor_std: float = 1.0,
                dilate_dist: float = 0.03, method: str = "passage") -> PointCloud:
    """Leaf / wood separation for one tree's points, returning the wood
    thinned to ``voxel_size``.

    Two cues are combined. Local anisotropy -- planarity + linearity over
    ``k`` neighbours (and, with ``scale_radius`` > 0, again over that wider
    neighbourhood, the lower score counting: better leaf labels, but it
    removes wood a QSM needs, see :func:`sylva.leaves.classify_leaf_wood`)
    -- marks bark and branch surfaces
    above ``threshold`` (high likelihood) and ``medium_threshold`` (kept
    only after statistical outlier removal, ``sor_k`` / ``sor_std``, and
    next to wood already found; set it at or above ``threshold`` to skip
    the step, which on small leafy crowns pulls foliage in around the
    stem). Topology recovers what anisotropy
    misses (the path-frequency cue of Vicari et al. 2019): shortest paths
    from the base over a kNN graph are traced to one target per
    ``target_res`` cell, and any point that at least
    ``min_passage`` of those paths run through is wood, with its neighbours
    within ``assign_dist`` -- a roughly or thinly scanned trunk is neither
    planar nor linear locally but every path to the crown crosses it. A
    path follows one side of a stem, so this keeps the points on and beside
    the paths rather than the whole section; ``assign_scale`` (try 0.03)
    widens the reach with the share of the tree a point carries, up to that
    fraction of the tree height on the trunk. High-likelihood points survive in connected
    components (``component_res``) that touch passage wood or hold
    ``component_min`` points; the result is dilated by ``dilate_dist`` for
    thin branches. ``passage=False`` gives the anisotropy-only filter.

    This filter is tuned to give a QSM what it needs (every stem and
    branch surface), not the most accurate leaf labels; for leaf area and
    leaf angles use :func:`sylva.leaves.classify_leaf_wood`.

    Parameters
    ----------
    cloud
        One segmented tree.
    k
        Neighbours for the anisotropy features.
    threshold, medium_threshold
        High- and medium-likelihood anisotropy cut-offs (0-1).
    voxel_size
        Thin the input to this spacing (m) first; None or 0 keeps every
        point.
    scale_radius
        Second, wider feature scale (m); 0 disables.
    passage, min_passage, target_res
        Path-passage wood: enable, paths a point must carry, and target
        cell size (m).
    graph_k, max_edge, base_height
        kNN graph neighbours, longest edge (m), and height (m) of the base
        region the paths start from.
    assign_dist, assign_scale
        Reach (m) around passage points, and its growth with the share of
        the tree a point carries.
    component_res, component_min
        Connected-component cell size (m) and minimum size.
    sor_k, sor_std
        Statistical outlier removal for the medium-likelihood points.
    dilate_dist
        Final dilation (m).
    method : {"passage", "gbs"}
        ``"gbs"`` uses the graph-based labeller of
        :func:`sylva.leaves.classify_leaf_wood` instead.

    Returns
    -------
    PointCloud
        The wood points, thinned to ``voxel_size``, with attributes.
    """
    from ..filters import voxel_downsample

    thin = voxel_downsample(cloud, voxel_size) if voxel_size else cloud
    if method == "gbs":
        from ..leaves import classify_leaf_wood

        return thin[classify_leaf_wood(thin, voxel_size=0.0, method="gbs")]
    mask = _core.wood_mask(thin.xyz, k, threshold, medium_threshold, scale_radius, graph_k, max_edge,
                           base_height, target_res, min_passage, assign_dist, assign_scale,
                           component_res,
                           component_min, sor_k, sor_std, dilate_dist, passage)
    return thin[mask]
