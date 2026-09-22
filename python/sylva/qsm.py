"""Quantitative structure models: skeletonisation and cylinder fitting.

A QSM is a set of connected cylinders. :func:`build_qsm` bins geodesic
distance from the base over a kNN graph, splits bins into connected segments,
fits a RANSAC cylinder to each and links parents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["QSM", "fit_cylinder", "fit_cylinder_ransac", "skeletonize", "build_qsm", "wood_points",
           "write_obj", "write_ply_mesh"]

COLUMNS = ("sx", "sy", "sz", "ax", "ay", "az", "length", "radius", "parent", "branch_order",
           "branch_id", "n_points")


@dataclass
class QSM:
    """A tree as connected cylinders; build with :func:`build_qsm`.

    Attributes
    ----------
    cylinders
        ``(n, 12)`` float array, one row per cylinder, with columns
        :data:`COLUMNS`:

        | Column | Meaning |
        |---|---|
        | ``sx, sy, sz`` | start (base) of the cylinder |
        | ``ax, ay, az`` | unit axis, pointing away from the tree base |
        | ``length``, ``radius`` | m |
        | ``parent`` | row of the parent cylinder, -1 for the first |
        | ``branch_order`` | 0 stem, 1 first-order branch, ... |
        | ``branch_id`` | branch the cylinder belongs to |
        | ``n_points`` | points the fit used; 0 where the radius came from priors |

    Examples
    --------
    >>> q = build_qsm(wood_points(tree_cloud))
    >>> q.total_volume, q.dbh
    >>> q.to_csv("tree_001.csv"); q.to_ply("tree_001.ply")
    """

    cylinders: np.ndarray

    def __post_init__(self) -> None:
        self.cylinders = np.ascontiguousarray(self.cylinders, dtype=np.float64).reshape(-1, 12)

    def __len__(self) -> int:
        return len(self.cylinders)

    def column(self, name: str) -> np.ndarray:
        """One column of :attr:`cylinders`.

        Parameters
        ----------
        name
            A name from :data:`COLUMNS`, e.g. ``"radius"``.

        Returns
        -------
        numpy.ndarray
            A view, length ``len(self)``.

        Raises
        ------
        ValueError
            For an unknown name.
        """
        return self.cylinders[:, COLUMNS.index(name)]

    @property
    def start(self) -> np.ndarray:
        """Cylinder start points, ``(n, 3)``."""
        return self.cylinders[:, 0:3]

    @property
    def axis(self) -> np.ndarray:
        """Unit axes, ``(n, 3)``."""
        return self.cylinders[:, 3:6]

    @property
    def end(self) -> np.ndarray:
        """Cylinder end points, ``start + axis * length``."""
        return self.start + self.axis * self.column("length")[:, None]

    @property
    def volumes(self) -> np.ndarray:
        """Volume of each cylinder (m³)."""
        return np.pi * self.column("radius") ** 2 * self.column("length")

    @property
    def total_volume(self) -> float:
        """Woody volume of the tree (m³); multiply by basic density for biomass."""
        return float(self.volumes.sum())

    @property
    def stem_volume(self) -> float:
        """Volume of the order-0 cylinders (m³)."""
        return float(self.volumes[self.column("branch_order") == 0].sum())

    @property
    def branch_volume(self) -> float:
        """Volume of all branches, order >= 1 (m³)."""
        return self.total_volume - self.stem_volume

    @property
    def total_length(self) -> float:
        """Summed cylinder length (m)."""
        return float(self.column("length").sum())

    @property
    def max_branch_order(self) -> int:
        """Highest branch order (0 for a bare stem or empty model)."""
        return int(self.column("branch_order").max()) if len(self) else 0

    @property
    def dbh(self) -> float:
        """Stem diameter at 1.3 m above the base (m), from the stem cylinders."""
        return _core.qsm_summary(self.cylinders)["dbh"]

    def metrics(self, crown_branch_length: float = 1.0, crown_slice: float = 0.5) -> dict:
        """Tree architecture from the cylinders.

        Height, DBH, volumes and lengths (total, stem, branches, and per
        branch order), branch and tip counts, path fraction (mean base-to-tip
        path over the longest), crown base height (lowest first-order branch
        at least ``crown_branch_length`` long), stem lean and its direction,
        sweep (greatest offset of the stem from the chord between the base and
        the crown base, over its length), the stem taper profile, the crown
        outlined by the branches (see :func:`sylva.trees.crown_shape`, slices
        ``crown_slice`` high), length-weighted median insertion and zenith
        angles of first-order branches (deg), and the share of volume and
        length whose radius was fitted to points rather than filled in by
        the taper and pipe-model priors -- a quality flag for the model.
        Heights are above the stem base.

        Parameters
        ----------
        crown_branch_length
            Shortest first-order branch (m) that marks the crown base.
        crown_slice
            Slice height (m) of the stacked crown hulls.

        Returns
        -------
        dict
            ``height``, ``dbh``, ``total_volume``, ``stem_volume``,
            ``branch_volume``, ``total_length``, ``stem_length``,
            ``max_order``, ``n_branches_by_order``, ``length_by_order``,
            ``volume_by_order`` (lists indexed by order), ``n_tips``,
            ``path_fraction``, ``crown_base_height``, ``lean`` and
            ``lean_direction`` (deg), ``sweep``, ``taper_heights`` and
            ``taper_radii`` (arrays), ``crown`` (dict as
            :func:`sylva.trees.crown_shape`), ``measured_volume_fraction``,
            ``measured_length_fraction``, ``median_insertion_angle`` and
            ``median_branch_zenith`` (deg). Lengths m, volumes m³.

        Notes
        -----
        Validated against the source meshes of simulated trees: height -2 %,
        DBH -1 %, crown area -3 %, branch zenith error 3 degrees
        (*Benchmarks > QSMs*). A low ``measured_volume_fraction`` means most
        of the volume came from priors; treat such trees with care.
        """
        return _core.qsm_metrics(np.ascontiguousarray(self.cylinders, dtype=float),
                                 float(crown_branch_length), float(crown_slice))

    def branches(self) -> dict[str, np.ndarray]:
        """One row per branch (``branch_id`` chain; the stem is order 0), as
        columns: ``id``, ``order``, ``parent`` (branch it grows from, -1 for
        the stem), ``n_cylinders``, ``length`` (m), ``volume`` (m3),
        ``base_radius`` and length-weighted ``mean_radius`` (m),
        ``base_height`` and ``tip_height`` (m above the stem base),
        ``insertion_angle`` (deg between its direction over 0.5 m past its
        first cylinder, which only joins it to the parent's axis, and the
        parent's direction over 0.5 m either side of the junction), ``zenith`` and ``azimuth`` of the base-to-tip chord (deg),
        ``tortuosity`` (length over chord), ``n_children`` and
        ``measured_fraction`` (share of the length fitted to points).

        Returns
        -------
        dict
            ``{column: array}``, one entry per branch; ready for
            ``pandas.DataFrame``.
        """
        return _core.qsm_branches(np.ascontiguousarray(self.cylinders, dtype=float))

    def summary(self) -> dict:
        """Headline numbers with units in the keys.

        Returns
        -------
        dict
            ``n_cylinders``, ``total_volume_m3``, ``stem_volume_m3``,
            ``branch_volume_m3``, ``total_length_m``, ``max_branch_order``
            and ``dbh_m``. Use :meth:`metrics` for the full set.
        """
        s = _core.qsm_summary(self.cylinders)
        return {
            "n_cylinders": len(self),
            "total_volume_m3": s["total_volume"],
            "stem_volume_m3": s["stem_volume"],
            "branch_volume_m3": s["branch_volume"],
            "total_length_m": s["total_length"],
            "max_branch_order": s["max_branch_order"],
            "dbh_m": s["dbh"],
        }

    def to_csv(self, path: str | Path) -> None:
        """Write the cylinders as CSV with a header of :data:`COLUMNS`.

        Parameters
        ----------
        path
            Output file; read back with :meth:`from_csv`.
        """
        _core.qsm_write_csv(self.cylinders, str(path))

    def to_treefile(self, path: str | Path) -> None:
        """Write in the raycloudtools/treetools ``_trees.txt`` format.

        Parameters
        ----------
        path
            Output file, for ``treeinfo``, ``treemesh`` and other treetools.
        """
        _core.qsm_write_treefile(self.cylinders, str(path))

    @classmethod
    def from_csv(cls, path: str | Path) -> QSM:
        """Read cylinders written by :meth:`to_csv`.

        Parameters
        ----------
        path
            CSV with a header line and the 12 :data:`COLUMNS`.

        Returns
        -------
        QSM
        """
        return cls(np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2))

    def mesh(self, sides: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triangle mesh of the cylinders, with end caps.

        Parameters
        ----------
        sides
            Facets around each cylinder.

        Returns
        -------
        vertices : numpy.ndarray
            ``(n, 3)`` float.
        faces : numpy.ndarray
            ``(m, 3)`` vertex indices, 0-based.
        owner : numpy.ndarray
            Cylinder row of each face.
        """
        return _core.qsm_mesh(self.cylinders, sides)

    def to_obj(self, path: str | Path, sides: int = 12) -> None:
        """Write the cylinder mesh as a Wavefront OBJ (Blender, MeshLab, CloudCompare).

        Parameters
        ----------
        path
            Output file.
        sides
            Facets around each cylinder.
        """
        v, f, _ = self.mesh(sides)
        write_obj(path, [(v, f)])

    def to_ply(self, path: str | Path, sides: int = 12, color=None) -> None:
        """Write the cylinder mesh as a binary PLY with face colours.

        Parameters
        ----------
        path
            Output file.
        sides
            Facets around each cylinder.
        color
            One RGB triple (0-255) for every face; by default faces are
            coloured by branch order, brown stem to green twigs.
        """
        v, f, owner = self.mesh(sides)
        if color is None:
            order = self.column("branch_order").astype(int)
            face_rgb = _ORDER_COLORS[np.minimum(order[owner], len(_ORDER_COLORS) - 1)]
        else:
            face_rgb = np.tile(np.asarray(color, dtype=np.uint8), (len(f), 1))
        write_ply_mesh(path, v, f, face_rgb)


_ORDER_COLORS = np.array([[139, 90, 43], [205, 133, 63], [222, 184, 135], [60, 179, 113],
                          [46, 139, 87], [34, 139, 34]], dtype=np.uint8)


def write_obj(path: str | Path, meshes: list[tuple[np.ndarray, np.ndarray]],
              names: list[str] | None = None) -> None:
    """Write several meshes to one OBJ file, one named object each.

    Parameters
    ----------
    path
        Output file.
    meshes
        ``(vertices, faces)`` pairs with 0-based face indices, e.g. from
        :meth:`QSM.mesh` for every tree of a plot.
    names
        Object names; ``tree_1``, ``tree_2``, ... if None.
    """
    with open(path, "w") as fh:
        fh.write("# sylva QSM mesh\n")
        offset = 1
        for i, (v, f) in enumerate(meshes):
            fh.write(f"o {names[i] if names else f'tree_{i + 1}'}\n")
            np.savetxt(fh, v, fmt="v %.4f %.4f %.4f")
            np.savetxt(fh, np.asarray(f, dtype=np.int64) + offset, fmt="f %d %d %d")
            offset += len(v)


def write_ply_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray,
                   face_rgb: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY triangle mesh.

    Parameters
    ----------
    path
        Output file.
    vertices
        ``(n, 3)`` coordinates, stored as float32.
    faces
        ``(m, 3)`` 0-based vertex indices.
    face_rgb
        Optional ``(m, 3)`` uint8 colour per face.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    header = ["ply", "format binary_little_endian 1.0", "comment sylva QSM mesh",
              f"element vertex {len(vertices)}", "property float x", "property float y",
              "property float z", f"element face {len(faces)}",
              "property list uchar int vertex_indices"]
    if face_rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(vertices.tobytes())
        if face_rgb is None:
            rec = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
            arr = np.empty(len(faces), dtype=rec)
        else:
            rec = np.dtype([("n", "u1"), ("i", "<i4", (3,)), ("rgb", "u1", (3,))])
            arr = np.empty(len(faces), dtype=rec)
            arr["rgb"] = np.asarray(face_rgb, dtype=np.uint8)
        arr["n"] = 3
        arr["i"] = faces
        fh.write(arr.tobytes())


def fit_cylinder(xyz: np.ndarray, axis_init=None) -> dict:
    """Least-squares cylinder through 3D points.

    Parameters
    ----------
    xyz
        ``(N, 3)`` points on a roughly cylindrical surface (a stem section).
    axis_init
        Starting axis direction; the principal direction of the points if
        None.

    Returns
    -------
    dict
        ``point`` (a point on the axis), ``axis`` (unit vector), ``radius``
        and ``rmse`` (m).
    """
    return _core.fit_cylinder(np.ascontiguousarray(xyz, dtype=float),
                              None if axis_init is None else tuple(float(v) for v in axis_init))


def fit_cylinder_ransac(xyz: np.ndarray, threshold: float = 0.02, iterations: int = 100,
                        sample_size: int = 12, seed: int = 0) -> dict:
    """Cylinder fit that tolerates outliers (leaves, twigs, noise).

    Parameters
    ----------
    xyz
        ``(N, 3)`` points.
    threshold
        Inlier distance from the surface (m).
    iterations
        RANSAC trials.
    sample_size
        Points per trial fit.
    seed
        Random seed.

    Returns
    -------
    dict
        As :func:`fit_cylinder`, plus ``inliers`` (boolean per point).
    """
    return _core.fit_cylinder_ransac(np.ascontiguousarray(xyz, dtype=float), threshold,
                                     iterations, sample_size, seed)


def skeletonize(cloud: PointCloud, base_xy=None, k: int = 15, max_edge: float = 1.0,
                bin_length: float = 0.1) -> dict:
    """Graph skeleton of one tree, the first stage of :func:`build_qsm`.

    Parameters
    ----------
    cloud
        One tree's (wood) points.
    base_xy
        Stem position; the lowest points if None.
    k
        Neighbours in the kNN graph.
    max_edge
        Longest graph edge (m).
    bin_length
        Width of the geodesic distance bins (m).

    Returns
    -------
    dict
        ``segment_id`` per point (-1 if disconnected from the base),
        ``geodesic`` distance from the base per point (m), segment
        ``centres`` and ``(child, parent)`` ``edges``.
    """
    return _core.skeletonize(cloud.xyz, None if base_xy is None else tuple(base_xy), k,
                             max_edge, bin_length)


def build_qsm(cloud: PointCloud, base_xy=None, k: int = 15, max_edge: float = 1.0,
              bin_length: float = 0.1, min_points: int = 1, ransac_threshold: float = 0.02,
              max_radius: float = 1.0, taper_limit: float = 1.1, max_rmse: float = 0.03,
              smooth_steps: int = 10, apex_radius: float = 0.0025, min_arc_deg: float = 90.0,
              min_inlier_fraction: float = 0.05, prune_points: int = 5, fit_min_points: int = 50,
              crop_length: float = 0.0, butt_height: float = 0.6,
              relative_tolerance: float = 0.08, base_radius: float = 0.0,
              allometry_tolerance: float = 0.3, buttress_equivalent_area: bool = True,
              buttress_max_inlier_fraction: float = 0.3, pipe_slack: float = 1.2,
              branch_min_inlier_fraction: float = 0.3, cluster_eps: float = 0.1,
              centre_fit_points: int = 100, radius_smooth_steps: int = 15,
              butt_swell: float = 1.1, butt_vertical_run: int = 4,
              butt_max_lean_deg: float = 50.0, chain_max_d: float = 0.1,
              fourier_min_radius: float = 0.15) -> QSM:
    """Skeletonise then fit cylinders for a single segmented tree.

    Skeleton nodes (geodesic bins of ``bin_length``) are Taubin-smoothed and
    a RANSAC circle is fitted to each node's points in the plane
    perpendicular to the skeleton; accepted fits need a contiguous arc of
    ``min_arc_deg`` and ``min_inlier_fraction`` inliers. Radii are then
    regularised along every root-to-tip path: an allometric prior anchored
    on ``base_radius`` (pass the measured DBH / 2) replaces weak fits more
    than ``allometry_tolerance`` away, chains are made non-increasing, gaps
    are interpolated and unmeasured branches take the prior. Leafy tips
    shorter than ``crop_length`` are not reconstructed. Run
    :func:`wood_points` first on leafy trees.

    Parameters
    ----------
    cloud
        One segmented tree, ideally wood only (:func:`wood_points`), in
        metres with z up. 1-2 cm spacing is enough; denser input is slower
        without being more accurate.
    base_xy
        Stem position (e.g. ``(tree.x, tree.y)``); the lowest points if None.
    k, max_edge
        kNN graph: neighbours per point and longest edge (m). Raise
        ``max_edge`` if occlusion leaves gaps along branches.
    bin_length
        Geodesic shell width (m); sets the cylinder length.
    min_points
        Smallest segment kept.
    ransac_threshold
        Circle inlier distance (m).
    max_radius
        Upper limit on any radius (m); set from the measured DBH.
    taper_limit
        A child may be at most this times its parent's radius; wider fits
        are clamped.
    max_rmse
        Circle fits with a larger RMSE (m) are rejected.
    smooth_steps
        Smoothing passes over skeleton node positions.
    apex_radius
        Smallest tip radius (m).
    min_arc_deg
        Contiguous arc a circle fit must cover (degrees).
    min_inlier_fraction, branch_min_inlier_fraction
        Inlier share a stem / branch circle needs.
    prune_points
        Unmeasured leaf segments with fewer points are pruned as foliage.
    fit_min_points
        Segments with fewer points take their radius from the taper model.
    crop_length
        Leafy tips with less subtree length (m) are not reconstructed.
    butt_height
        Below this height (m) only the largest component per shell is kept.
    relative_tolerance
        Refit band around the circle, as a fraction of its radius.
    base_radius
        Breast-height radius (m) anchoring the taper prior; 0 estimates it.
        Pass a field DBH / 2 when you have one.
    allometry_tolerance
        Weak fits further than this fraction from the prior are replaced.
    buttress_equivalent_area, buttress_max_inlier_fraction
        Use an equivalent-area radius where a circle explains fewer than
        that share of a section's points (buttresses, fluting).
    pipe_slack
        Children's summed cross-section may exceed the parent's by this
        factor (pipe model); 0 disables.
    cluster_eps
        Clustering radius within a shell (m); 0 uses graph components.
    centre_fit_points
        Clusters with at least this many points get a circle-fitted centre.
    radius_smooth_steps
        Smoothing passes over radii along each axis.
    butt_swell
        Unmeasured stem nodes below the lowest fit may exceed it by this
        factor.
    butt_vertical_run, butt_max_lean_deg
        Replace the stem below the first run of this many cylinders within
        ``butt_max_lean_deg`` of vertical by a vertical stump; 0 disables.
    chain_max_d
        Skeleton chaining radius (m).
    fourier_min_radius
        Sections at least this wide (m) and well covered use an
        equivalent-area Fourier contour instead of a circle; 0 disables.

    Returns
    -------
    QSM

    Notes
    -----
    On 72 destructively harvested trees volume has -3.8 % bias and
    19.9 % rRMSE (*Benchmarks > QSMs*). On simulated trees the deficit is
    in twigs: about a third of the twig length is recovered, so
    ``measured_length_fraction`` is low even when volume is good.
    """
    d = _core.build_qsm(cloud.xyz, None if base_xy is None else tuple(base_xy), k, max_edge,
                        bin_length, min_points, ransac_threshold, max_radius, taper_limit,
                        max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction,
                        prune_points, fit_min_points, crop_length, butt_height,
                        relative_tolerance, base_radius, allometry_tolerance,
                        buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack,
                        branch_min_inlier_fraction, cluster_eps, centre_fit_points,
                        radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius)
    return QSM(d["cylinders"])


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
    misses: shortest paths from the base over a kNN graph are traced to one
    target per ``target_res`` cell, and any point that at least
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
    from .filters import voxel_downsample

    thin = voxel_downsample(cloud, voxel_size) if voxel_size else cloud
    if method == "gbs":
        from .leaves import classify_leaf_wood

        return thin[classify_leaf_wood(thin, voxel_size=0.0, method="gbs")]
    mask = _core.wood_mask(thin.xyz, k, threshold, medium_threshold, scale_radius, graph_k, max_edge,
                           base_height, target_res, min_passage, assign_dist, assign_scale,
                           component_res,
                           component_min, sor_k, sor_std, dilate_dist, passage)
    return thin[mask]
