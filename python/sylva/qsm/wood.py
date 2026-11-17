# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Wood-point filtering ahead of a QSM."""

from __future__ import annotations

from .. import _core
from ..pointcloud import PointCloud


def wood_points(cloud: PointCloud, k: int = 20, threshold: float = 0.85,
                voxel_size: float | None = 0.02,
                scale_radius: float = 0.0, paths: bool = True, min_support: float = 0.3,
                graph_k: int = 8, max_edge: float = 1.0, base_height: float = 0.25,
                reach_scale: float = 0.0,
                grow_reach: float = 0.05, grow_hops: int = 1,
                method: str = "paths") -> PointCloud:
    """Leaf / wood separation for one tree's points, returning the wood
    thinned to ``voxel_size``.

    Two cues are combined. Topology finds the wood (the path-frequency cue
    of Vicari et al. 2019, measured as a length): over the tree of shortest
    paths from the base on a kNN graph, the *support* of a point is the
    length of path from it to the farthest point whose path runs through
    it, and a point that supports at least ``min_support`` is wood. A leaf,
    or a bark point where a path ends, supports nothing; a point on a
    branch supports the branch beyond it -- so a roughly or thinly scanned
    trunk, neither planar nor linear locally, is kept because the paths to
    the crown run up it. The graph joins the pieces of the cloud wherever
    two come within ``max_edge`` of each other, so a trunk cut off from its
    crown by an occluded stretch still carries the crown's paths.
    ``reach_scale`` (try 0.03) also takes the points within that fraction
    of a wood point's support, about a metre on a trunk that carries 30 m
    and nothing on a twig. Local anisotropy
    then completes it: from the wood the paths found, the filter grows over
    the same graph, a step of at most ``grow_reach`` at a time, through
    points whose planarity + linearity over ``k`` neighbours is above
    ``threshold`` (bark and branch surfaces; with ``scale_radius`` > 0 the
    lower of that and the score over the wider neighbourhood counts: better
    leaf labels, but it removes wood a QSM needs, see
    :func:`sylva.leaves.classify_leaf_wood`), and takes ``grow_hops`` steps
    more whatever the anisotropy, for twigs too thin to have any.
    ``paths=False`` leaves the paths out and keeps the points above
    ``threshold`` alone.

    This filter is tuned to give a QSM what it needs (every stem and
    branch surface), not the most accurate leaf labels; for leaf area and
    leaf angles use :func:`sylva.leaves.classify_leaf_wood`.

    Parameters
    ----------
    cloud
        One segmented tree.
    k
        Neighbours for the anisotropy features.
    threshold
        Anisotropy above which a point is wood-like (0-1).
    voxel_size
        Thin the input to this spacing (m) first; None or 0 keeps every
        point.
    scale_radius
        Second, wider feature scale (m); 0 disables.
    paths, min_support
        Wood from the shortest paths: enable, and the length of path (m) a
        point must support.
    graph_k, max_edge, base_height
        kNN graph neighbours, longest edge (m), and height (m) of the base
        region the paths start from.
    reach_scale
        Reach around a wood point as a fraction of its support; 0 for none.
    grow_reach, grow_hops
        Longest graph step (m) the wood grows across, and the steps taken
        beyond it whatever the anisotropy.
    method : {"paths", "gbs"}
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
    mask = _core.wood_mask(thin.xyz, k, threshold, scale_radius, graph_k, max_edge,
                           base_height, min_support, reach_scale,
                           grow_reach, grow_hops, paths)
    return thin[mask]
