# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The quantitative structure model (QSM) of a tree: cylinders fitted to its wood."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ._common import _check_weights
from .mesh import _rgb

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
        return _core.qsm_ends(self.cylinders)

    @property
    def volumes(self) -> np.ndarray:
        """Volume of each cylinder (m³)."""
        return _core.qsm_volumes(self.cylinders)

    @property
    def total_volume(self) -> float:
        """Woody volume of the tree (m³); multiply by basic density for biomass."""
        return _core.qsm_totals(self.cylinders)["total_volume"]

    @property
    def stem_volume(self) -> float:
        """Volume of the order-0 cylinders (m³)."""
        return _core.qsm_totals(self.cylinders)["stem_volume"]

    @property
    def branch_volume(self) -> float:
        """Volume of all branches, order >= 1 (m³)."""
        return _core.qsm_totals(self.cylinders)["branch_volume"]

    @property
    def total_length(self) -> float:
        """Summed cylinder length (m)."""
        return _core.qsm_totals(self.cylinders)["total_length"]

    @property
    def max_branch_order(self) -> int:
        """Highest branch order (0 for a bare stem or empty model)."""
        return int(_core.qsm_totals(self.cylinders)["max_branch_order"])

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
        return cls(_core.qsm_read_csv(str(path)))

    def mesh(self, sides: int = 12, contiguous: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triangle mesh of the cylinders.

        Parameters
        ----------
        sides
            Facets around each cylinder.
        contiguous
            One continuous tube per branch instead of one closed tube per
            cylinder: consecutive cylinders share a ring, so a branch has no
            caps inside it and its surface runs unbroken from base to tip.
            The ring frame is carried along the branch so the facets do not
            twist, and a shared ring takes the mean of the two radii. Each
            branch is still its own closed surface, pushed into its parent
            rather than welded to it, and the mesh is about half the size.

        Returns
        -------
        vertices : numpy.ndarray
            ``(n, 3)`` float.
        faces : numpy.ndarray
            ``(m, 3)`` vertex indices, 0-based.
        owner : numpy.ndarray
            Cylinder row of each face.
        """
        return _core.qsm_mesh(self.cylinders, sides, contiguous)

    def to_obj(self, path: str | Path, sides: int = 12, contiguous: bool = False) -> None:
        """Write the cylinder mesh as a Wavefront OBJ (Blender, MeshLab, CloudCompare).

        Parameters
        ----------
        path
            Output file.
        sides
            Facets around each cylinder.
        contiguous
            One continuous tube per branch (see :meth:`mesh`).
        """
        _core.qsm_write_obj(str(path), self.cylinders, int(sides), bool(contiguous))

    def volume_above(self, z: float) -> float:
        """Cylinder volume above a horizontal plane, cutting cylinders that cross it.

        Use it to join a :func:`buttress_mesh` to the model: the buttress
        volume below its top plus the cylinders above it.

        Parameters
        ----------
        z
            Absolute height of the plane (e.g. ``Buttress.top_z``).

        Returns
        -------
        float
            Volume (m³); a cylinder crossing the plane counts by the share of
            its axis above it.
        """
        return _core.qsm_volume_above(self.cylinders, float(z))

    def above(self, z: float) -> QSM:
        """The model above a horizontal plane, cutting cylinders that cross it.

        The companion of :meth:`volume_above`: a cylinder crossing the plane
        keeps the part above it, one below it is dropped, and a child whose
        parent went is made a branch of its own. Use it to leave room for a
        :class:`Buttress` under the wood.

        Parameters
        ----------
        z
            Absolute height of the plane (e.g. ``Buttress.top_z``).

        Returns
        -------
        QSM
            A new model; the original is unchanged.

        Notes
        -----
        A cylinder is cut where its axis crosses the plane, so a leaning one
        keeps a slanted stub rather than being squared off.
        """
        return QSM(_core.qsm_above(self.cylinders, float(z)))

    def to_ply(self, path: str | Path, sides: int = 12, color=None, contiguous: bool = False) -> None:
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
        contiguous
            One continuous tube per branch (see :meth:`mesh`).
        """
        _core.qsm_write_ply(str(path), self.cylinders, int(sides), _rgb(color), bool(contiguous))


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
              allometry_tolerance: float = 0.3, stem_radius_cap: float = 0.0,
              buttress_equivalent_area: bool = True,
              buttress_max_inlier_fraction: float = 0.3, pipe_slack: float = 1.2,
              branch_min_inlier_fraction: float = 0.3, spacing_scale: float = 1.5,
              radius_power: float = 0.0, power_above_spacing: float = 0.025,
              sensor_noise: float = 0.02, cluster_eps: float = 0.1,
              centre_fit_points: int = 100, radius_smooth_steps: int = 15,
              butt_swell: float = 1.1, butt_vertical_run: int = 4,
              butt_max_lean_deg: float = 50.0, chain_max_d: float = 0.1,
              fourier_min_radius: float = 0.15, min_weight: float = 0.0,
              min_mean_weight: float = 0.5, weights=None) -> QSM:
    """Skeletonise then fit cylinders for a single segmented tree.

    Skeleton nodes (geodesic bins of ``bin_length``) are Taubin-smoothed and
    a RANSAC circle is fitted to each node's points in the plane
    perpendicular to the skeleton; accepted fits need a contiguous arc of
    ``min_arc_deg`` and ``min_inlier_fraction`` inliers. Radii are then
    regularised along every root-to-tip path: an allometric prior anchored
    on ``base_radius`` (pass the measured DBH / 2) replaces weak fits more
    than ``allometry_tolerance`` away, chains are made non-increasing, gaps
    are interpolated and unmeasured branches take the prior, a pipe model
    (Shinozaki et al. 1964) that shares a parent's cross-section among its
    children. Past 2.5 cm point spacing the radius is a power mean of the
    distances to the axis, after raycloudtools (Devereux et al. 2026).
    Leafy tips shorter than ``crop_length`` are not reconstructed. Run
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
    stem_radius_cap
        With a ``base_radius``, anchor the taper prior on it and set aside
        every main-stem circle wider than this many times it, strong or not;
        the stem follows the prior there and those cylinders count as
        unmeasured. For boles sheathed in foliage (epicormic regrowth after
        fire), whose circles fit the foliage rather than the bark; a stem is
        nowhere above breast height much wider than at it. 1.5 brought 12
        sheathed Tumbarumba eucalypts from a median QSM / field DBH of 2.2
        to 0.98 (their volume from 105 to 36 m³) and changed 8 clean ones by
        2 % in volume. 0 (default)
        disables. :func:`build_plot` passes each tree's DBH / 2 as
        ``base_radius``.
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
    min_weight
        With ``weights``: points with a lower weight are dropped before the
        graph is built. 0 (default) keeps every point.
    min_mean_weight
        With ``weights``: a circle whose inliers have a lower mean weight is
        not a measurement, and the section takes its radius from the priors.
    weights
        Wood weight per point in [0, 1], for example a wood confidence
        (:func:`sylva.leaves.classify_leaf_wood` with ``return_scores=True``). Every point with at least
        ``min_weight`` builds the graph and the skeleton, so connectivity is
        that of the whole tree, but the weights decide each section's radius:
        RANSAC scores a candidate circle by the summed weight of its inliers,
        the circle is refitted by weighted least squares, a section needs a
        summed weight of ``fit_min_points`` to be fitted, its inliers a summed
        weight of ``min_points`` and a mean weight of ``min_mean_weight``,
        and the inlier share is a share of weight. The arc and the contour of
        a section use its confident points (weight at least 0.5), and a
        cylinder's ``n_points`` counts its confident inliers. Unmeasured leaf
        segments are pruned by summed weight (``prune_points``). Weights of
        one everywhere give exactly the unweighted model. None (default) fits
        every point alike.

    Returns
    -------
    QSM

    Raises
    ------
    ValueError
        If ``weights`` does not have one value per point, or holds a value
        that is NaN or outside [0, 1].

    Notes
    -----
    On 72 destructively harvested trees volume has -3.8 % bias and
    19.9 % rRMSE (*Benchmarks > QSMs*). On simulated trees the deficit is
    in twigs: about a third of the twig length is recovered, so
    ``measured_length_fraction`` is low even when volume is good.
    """
    if weights is not None:
        weights = _check_weights(weights, len(cloud))
    d = _core.build_qsm(cloud.xyz, None if base_xy is None else tuple(base_xy), k, max_edge,
                        bin_length, min_points, ransac_threshold, max_radius, taper_limit,
                        max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction,
                        prune_points, fit_min_points, crop_length, butt_height,
                        relative_tolerance, base_radius, allometry_tolerance, stem_radius_cap,
                        buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack,
                        branch_min_inlier_fraction, spacing_scale, radius_power, power_above_spacing, sensor_noise,
                        cluster_eps, centre_fit_points,
                        radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius,
                        min_weight, min_mean_weight, weights=weights)
    return QSM(d["cylinders"])
