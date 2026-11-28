# Sylva: LiDAR processing for forest ecology and remote sensing research.
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
            twist, and a shared ring has the radius of the cylinder that
            starts at it. Each
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
                bin_length: float = 0.125) -> dict:
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
              bin_length: float = 0.125, min_points: int = 1, ransac_threshold: float = 0.02,
              max_radius: float = 1.0, taper_limit: float = 1.1, max_rmse: float = 0.03,
              centre_passes: int = 4, twig_radius: float = 0.002, min_arc_deg: float = 0.0,
              min_inlier_fraction: float = 0.05, prune_points: int = 5, fit_min_points: int = 30,
              crop_length: float = 0.0, relative_tolerance: float = 0.08, base_radius: float = 0.0,
              outlier_sigma: float = 3.0, stem_radius_cap: float = 0.0, pipe_slack: float = 1.2,
              spacing_scale: float = 1.5, radius_power: float = 0.0, power_above_spacing: float = 0.025,
              sensor_noise: float = 0.02, butt_swell: float = 1.1, min_weight: float = 0.0,
              min_mean_weight: float = 0.5, min_snr: float = 5.0, noise_ratio: float = 1.4,
              place_axes: bool = True, width_slack: float = 1.5, pipe_excess: float = 3.0,
              weights=None) -> QSM:
    """Skeletonise then fit cylinders for a single segmented tree.

    The points are binned into shells of path length from the base over a
    kNN graph that joins the pieces of the cloud lying within ``max_edge``
    of each other, and each shell is split into its connected pieces, the
    nodes of the skeleton. A RANSAC circle is fitted to each node's points
    in the plane perpendicular to the skeleton, and is a measurement when it
    stands clear of the noise. Wood is a surface, so its points lie on their
    circle to within the scan's noise, while the points of foliage, or of
    several twigs taken as one section, are spread across theirs. The
    scatter of a section is the larger of two figures: the root mean square
    distance from the circle of the points within its band, and 1.4826
    times the median distance of *all* the section's points from it, which
    is the same for a clean ring and far larger when the ring holds under
    half the points. The radius has to be at least ``min_snr`` times that
    scatter, and on a branch the scatter of the band no more than
    ``noise_ratio`` times the scatter on the tree's own stem.

    How much of the circle the points have to cover follows from the same
    noise. A circle through an arc is fixed by the arc's chord and its rise
    above the chord, and the noise acts on the rise: an error ``e`` in it
    moves the radius by ``e (1 + cos(a/2)) / (2 (1 - cos(a/2)))`` for an arc
    of angle ``a``, half the error for a half circle, 1.1 times it for 135
    degrees, 2.9 times for 90. So below a half circle ``min_snr`` is
    multiplied by that factor relative to the half circle (2.2 at 135
    degrees, 5.8 at 90, 14 at 60): a thick limb seen from below is measured
    from the part of its ring that the scanner saw, and a twig is not.

    Each axis then gets a taper of its own, ``r = k x^b``, fitted robustly
    (Theil-Sen) to its accepted sections, with ``x`` the longest way from
    the node to a tip times the share of the cross-section its length
    claimed at every fork on the way up (raycloudtools, Lowe and Stepanas
    2021; Devereux et al. 2026). Sections more than ``outlier_sigma`` robust
    deviations off the taper of their own axis are set aside, and the axis
    is made non-increasing between the rest. Below its lowest section the
    stem follows its taper (up to ``butt_swell`` times that section). A
    measured DBH (``base_radius``) ties the stem's taper down when the stem
    has few sections.

    What is not measured takes its radius from what it carries. The *load*
    of a node is the length of branch beyond it, its own cylinder included,
    and beyond an axis's last section, or on an axis with none, the
    cross-section *area* is in proportion to the load, from the last
    measured node or from the node the axis hangs from:
    ``r = r0 sqrt(load / load0)``. This is the pipe model (Shinozaki et al.
    1964) with the foliage a branch feeds taken as spread evenly along the
    wood beyond it. Areas then add up at every fork, as Leonardo's rule has
    them, and a branch thins towards its tip as the square root of what is
    left of it. A radius in proportion to length, which versions up to 2.1
    used, gives a side branch a fifth to a half of its true radius on trees
    whose radii are known (*Benchmarks > QSMs*).

    Two checks use the same load and the points themselves. A section on an
    axis with fewer than four sections has no taper to be judged by, and is
    set aside when its radius is more than ``pipe_excess`` times what its
    load gives it from the nearest section kept on the way to the root:
    foliage that sleeves a twig makes a ring like a branch's, and only what
    the twig carries tells them apart. And no unmeasured radius is more than
    ``width_slack`` times the width of its section's points,
    ``sqrt(2 (v - noise^2))`` with ``v`` their largest variance across the
    axis, which is the radius of a branch seen from one side or from all
    round, and more than it where leaves or twigs share the section. It
    takes no fit and holds down to a radius of twice the noise.

    Past 2.5 cm point spacing the radius is a power mean of the distances
    to the centre of the least-squares circle through the section (after
    raycloudtools, Devereux et al. 2026), where that circle passes the same
    tests as a fitted one; a section without such a circle is unmeasured.

    Positions come last (``place_axes``). The mean of a node's points is not
    the centre of its section: of a ring seen from one side it lies up to a
    radius towards the scanner. A node with an accepted circle sits on the
    circle's centre. Every other node wide enough to be resolved (a radius
    of ``min_snr`` times the stem's scatter) has a radius by then, from its
    axis, and a ring of known radius is located by far less of an arc than
    one whose radius is unknown too: its centre is the point that puts the
    most of its points on a circle of that radius, taken when the arc they
    cover rises clear of the band they are counted in. A resolved node with
    neither lies between the nearest placed nodes of its axis, no further
    from the mean of its points than its radius. Each resolved node then
    moves onto a line fitted through the nodes within two radii of it along
    the axis, because an axis does not bend within its own width while the
    centres of neighbouring sections err independently. Below its lowest
    accepted section the stem is the straight line through the sections
    above it, each node at its own height: the shells there follow root
    flare and what is left of the ground, not the stem.
    Leafy tips shorter than ``crop_length`` are not reconstructed. Run
    :func:`wood_points` first on leafy trees.

    Parameters
    ----------
    cloud
        One segmented tree, ideally wood only (:func:`wood_points`), in
        metres with z up. 1-2 cm spacing is enough; denser input is slower
        without being more accurate. Leave out the ground and the litter
        around the base (the wood filter does): the skeleton follows them
        outwards and they are modelled as branches.
    base_xy
        Stem position (e.g. ``(tree.x, tree.y)``); the lowest points if None.
    k, max_edge
        kNN graph: neighbours per point and longest edge (m). Raise
        ``max_edge`` if occlusion leaves gaps along branches.
    bin_length
        Geodesic shell width (m); sets the cylinder length.
    min_points
        Fewest circle inliers for a section's fit to count as a measurement;
        segments are not dropped by it.
    ransac_threshold
        Circle inlier distance (m).
    max_radius
        Upper limit on any radius (m); set from the measured DBH.
    taper_limit
        A node wider than this times the node it hangs from takes that
        node's radius.
    max_rmse
        Circle fits with a larger RMSE (m) are rejected.
    centre_passes
        Smoothing passes over the skeleton's node positions: each moves a
        node half way to the mean of its neighbours.
    twig_radius
        Radius of a twig (m): what a tip is given, and the least any
        cylinder has.
    min_arc_deg
        A floor on the contiguous arc a section's points must cover
        (degrees), on top of the test against the noise. 0 (default) leaves
        the arc to that test; 180 is the rule of versions up to 2.1.
    min_inlier_fraction
        Inlier share a circle needs.
    prune_points
        Unmeasured leaf segments with fewer points are pruned as foliage.
    fit_min_points
        Segments with fewer points take their radius from the taper model.
        The count holds at every point spacing: the shells widen with the
        spacing, so a ring wide enough to be resolved still holds as many.
    crop_length
        Leafy tips with less subtree length (m) are not reconstructed.
    relative_tolerance
        Refit band around the circle, as a fraction of its radius.
    base_radius
        Breast-height radius (m): the stem's taper is tied to it when the
        stem has fewer than four sections of its own (always with
        ``stem_radius_cap``). Pass a field DBH / 2 when you have one; 0 for
        none.
    outlier_sigma
        Sections further than this many robust standard deviations (in
        log radius) from the taper of their own axis are set aside.
    stem_radius_cap
        With a ``base_radius``, tie the stem's taper to it and set aside
        every main-stem circle wider than this many times it, however well
        fitted; the stem follows the taper there and those cylinders count as
        unmeasured. For boles sheathed in foliage (epicormic regrowth after
        fire), whose circles fit the foliage rather than the bark; a stem is
        nowhere above breast height much wider than at it. 1.5 brought 12
        sheathed Tumbarumba eucalypts from a median QSM / field DBH of 2.2
        to 0.98 (their volume from 105 to 36 m³) and changed 8 clean ones by
        2 % in volume. 0 (default)
        disables. :func:`build_plot` passes each tree's DBH / 2 as
        ``base_radius``.
    pipe_slack
        Children's summed cross-section may exceed the parent's by this
        factor (pipe model); the excess is taken from the unmeasured
        children, and a measured child keeps its radius. 0 disables.
    spacing_scale
        Scale the circle band and the shell length to the cloud's median
        point spacing: the band is at least this many spacings and the
        shells at least four. Neither is made tighter than given. 0 leaves
        the given values alone.
    radius_power
        Take each section's radius from a power mean of its points'
        distances to the axis, ``(mean d^q)^(1/q)``, instead of a circle
        fit; 0 (default) fits circles.
    power_above_spacing
        Point spacing (m) at or above which the power mean (``q = 0.25``)
        takes over from the circle fit when ``radius_power`` is 0 and
        ``spacing_scale`` is on; 0 never switches.
    sensor_noise
        Not used: a power-mean radius is judged by ``min_snr`` and
        ``noise_ratio``, like a circle. Kept so that existing calls work.
    butt_swell
        Unmeasured stem nodes below the lowest fit may exceed it by this
        factor.
    min_weight
        With ``weights``: points with a lower weight are dropped before the
        graph is built. 0 (default) keeps every point.
    min_mean_weight
        With ``weights``: a circle whose inliers have a lower mean weight is
        not a measurement, and the section takes its radius from the priors.
    min_snr
        A section is a measurement when its radius is at least this many
        times the scatter of its points about their circle, and more for an
        arc under half the circle (see above): a thinner ring is not
        resolved from the noise. With 5 mm of scatter the default, 5, leaves
        radii under 2.5 cm to the taper, under 5.6 cm for a ring seen over
        135 degrees and under 15 cm over 90. 0 disables, and then
        ``min_arc_deg`` is the only test of the arc.
    noise_ratio
        A branch section is a measurement when the scatter of its points is
        at most this many times the median scatter on the tree's own stem,
        the scan's noise as this tree shows it. Applies when the stem has
        at least four sections. 0 disables.
    place_axes
        Place the nodes whose own circle was not accepted from their points
        with the radius held, straighten each axis over its own width and
        run the stem straight below its lowest accepted section (see
        above). False leaves those nodes at the smoothed mean of their
        points, as versions up to 2.1 did.
    width_slack
        An unmeasured radius is at most this many times the width of its
        section's points (see above), where the section holds
        ``fit_min_points``. At 1.5 a branch seen over 100 degrees or more
        of its ring is never cut. 0 disables.
    pipe_excess
        A section on an axis with fewer than four sections is set aside when
        its radius is more than this many times what the length of branch
        it carries gives it from the nearest section kept on the way to the
        root. 0 disables.
    weights
        Wood weight per point in [0, 1], for example a wood confidence
        (:func:`sylva.leaves.classify_leaf_wood` with ``return_scores=True``). Every point with at least
        ``min_weight`` builds the graph and the skeleton, so connectivity is
        that of the whole tree, but the weights decide each section's radius:
        RANSAC scores a candidate circle by the summed weight of its inliers,
        the circle is refitted by weighted least squares, a section needs a
        summed weight of ``fit_min_points`` to be fitted, its inliers a summed
        weight of ``min_points`` and a mean weight of ``min_mean_weight``,
        and the inlier share is a share of weight. The arc of
        a section uses its confident points (weight at least 0.5), and a
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
    On 72 destructively harvested tropical trees volume has +0.5 % bias and
    16.8 % rRMSE, and on 91 more from Belgium and Guyana that played no
    part in any choice, -5 % and 11 %, and +10 % and 32 % (*Benchmarks >
    QSMs*). About a third of the twig length of a simulated tree is
    recovered, so ``measured_length_fraction`` is low even when volume is
    good.
    """
    if weights is not None:
        weights = _check_weights(weights, len(cloud))
    d = _core.build_qsm(cloud.xyz, None if base_xy is None else tuple(base_xy), k, max_edge,
                        bin_length, min_points, ransac_threshold, max_radius, taper_limit,
                        max_rmse, centre_passes, twig_radius, min_arc_deg, min_inlier_fraction,
                        prune_points, fit_min_points, crop_length, relative_tolerance, base_radius,
                        outlier_sigma, stem_radius_cap, pipe_slack, spacing_scale, radius_power,
                        power_above_spacing, sensor_noise, butt_swell, min_weight, min_mean_weight,
                        min_snr, noise_ratio, place_axes, width_slack, pipe_excess,
                        weights=weights)
    return QSM(d["cylinders"])
