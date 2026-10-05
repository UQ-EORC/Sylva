# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Quantitative structure models: skeletonisation and cylinder fitting.

A QSM is a set of connected cylinders. :func:`build_qsm` bins geodesic
distance from the base over a kNN graph, splits bins into connected segments
(Verroust and Lazarus 2000; Xu et al. 2007), fits a RANSAC cylinder to each
and links parents.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["QSM", "PlotQSMs", "build_plot", "fit_cylinder", "fit_cylinder_ransac", "skeletonize", "build_qsm", "wood_points",
           "write_obj", "write_ply_mesh", "Buttress", "buttress_mesh", "TreeMesh"]

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


def _rgb(color):
    """One RGB triple as three ints (0-255), or None."""
    return None if color is None else [int(c) for c in np.broadcast_to(np.asarray(color, dtype=np.uint8), (3,))]


def _faces(faces) -> np.ndarray:
    """``(m, 3)`` vertex indices as uint32; an error for negative ones."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if (f < 0).any():
        raise ValueError("face indices must not be negative")
    return np.ascontiguousarray(f, dtype=np.uint32)


@dataclass
class Buttress:
    """An irregular stem base as a closed mesh; build with :func:`buttress_mesh`.

    Attributes
    ----------
    vertices, faces
        Closed triangle mesh (``(n, 3)`` coordinates, ``(m, 3)`` 0-based
        vertex indices) from the ground to ``top``.
    volume
        Volume below ``top`` (m³), the sum of the slice areas.
    top, top_z
        Where the buttress ends: height above ground (m) and absolute z.
    heights, areas, solidities
        Per slice: bottom height (m), cross-section area (m²), and solidity
        (area over convex-hull area; about 1 for a round stem, low for flanges).
    open
        Per slice, True where the outline did not close (part of the bark
        unseen): the seen bark was kept, thickened to ``close_radius``, plus a
        circle where the points form a good arc.
    """

    vertices: np.ndarray
    faces: np.ndarray
    volume: float
    top: float
    top_z: float
    heights: np.ndarray
    areas: np.ndarray
    solidities: np.ndarray
    open: np.ndarray

    def total_volume(self, model: QSM) -> float:
        """Buttress volume plus the QSM's volume above the buttress.

        Parameters
        ----------
        model
            The tree's cylinder model.

        Returns
        -------
        float
            Volume (m³).
        """
        return self.volume + model.volume_above(self.top_z)

    def fuse(self, model: QSM, sides: int = 12, contiguous: bool = True,
             overlap: float = 0.1) -> "TreeMesh":
        """Join this buttress to a QSM as one mesh of the whole stem.

        The buttress replaces the cylinders below its top: the model is cut
        at ``top_z`` (:meth:`QSM.above`) and both surfaces go into one mesh,
        so nothing is counted twice and the volume is the one
        :meth:`total_volume` reports. The parts stay watertight and
        separately labelled rather than being welded into a single shell — a
        boolean union of a flanged base and a thousand tubes is not something
        a triangle mesh survives cleanly.

        A cut tube ends square to its own axis, so a leaning stem would hang
        over the buttress top by up to its radius times the lean. ``overlap``
        cuts the wood that much lower, letting it reach down inside the base
        where the seam cannot be seen. It changes no volume: those are read
        from ``top_z`` either way.

        Parameters
        ----------
        model
            The tree's cylinder model, in the same frame as the buttress.
        sides
            Facets around each cylinder.
        contiguous
            One continuous tube per branch (see :meth:`QSM.mesh`).
        overlap
            How far below ``top_z`` (m) the wood is cut, so that it meets the
            buttress inside it. 0 abuts the two exactly at the plane.

        Returns
        -------
        TreeMesh
            Vertices, faces, a part label per face, and the volumes.

        Examples
        --------
        >>> b = buttress_mesh(cloud, base_xy=(x, y))          # doctest: +SKIP
        >>> b.fuse(model).to_ply("tree.ply")                  # doctest: +SKIP
        """
        d = _core.qsm_fuse(np.ascontiguousarray(self.vertices, dtype=float), _faces(self.faces),
                           float(self.volume), float(self.top_z), model.cylinders, int(sides), bool(contiguous),
                           float(overlap))
        return TreeMesh(d["vertices"], d["faces"], d["part"], d["buttress_volume"], d["wood_volume"],
                        d["top_z"], d["offset"], d["overhang"])

    def to_obj(self, path: str | Path) -> None:
        """Write the mesh as a Wavefront OBJ object named ``buttress``.

        Parameters
        ----------
        path
            Output file.
        """
        write_obj(path, [(self.vertices, self.faces)], names=["buttress"])

    def to_ply(self, path: str | Path) -> None:
        """Write the mesh as a binary PLY.

        Parameters
        ----------
        path
            Output file.
        """
        write_ply_mesh(path, self.vertices, self.faces)


def _section(vertices: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """Segments where a mesh crosses the plane ``z``, as ``(n, 2, 2)`` in xy."""
    return _core.qsm_section(np.ascontiguousarray(vertices, dtype=float), _faces(faces), float(z))


def _segments(section) -> np.ndarray:
    return np.ascontiguousarray(section, dtype=float).reshape(-1, 2, 2)


def _inside(section: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Even-odd test of ``points`` against a soup of segments (:func:`_section`)."""
    return _core.qsm_inside(_segments(section), np.ascontiguousarray(points, dtype=float))


def _join_fit(base: np.ndarray, wood: np.ndarray, cell: float = 0.02) -> tuple[float, float]:
    """How well the wood sits inside the base: centre offset (m) and the share
    of the wood's cross-section outside it."""
    return _core.qsm_join_fit(_segments(base), _segments(wood), float(cell))


@dataclass
class TreeMesh:
    """A buttress and a QSM as one mesh; build with :meth:`Buttress.fuse`.

    Attributes
    ----------
    vertices, faces
        The two surfaces in one mesh (``(n, 3)`` coordinates, ``(m, 3)``
        0-based vertex indices).
    part
        Per face: 0 for the buttress, 1 for the wood above it.
    buttress_volume, wood_volume
        Volume below and above ``top_z`` (m³).
    top_z
        Absolute height where the buttress ends and the cylinders start.
    offset
        Distance (m) between the middle of the base and the middle of the wood
        at the join. A stem that sits over its base is a few centimetres out.
    overhang
        Share of the wood's cross-section at the join that lies outside the
        base. Anything much above zero means the two do not agree about where
        the stem is: usually a buttress top found too low, a stem axis pulled
        off by the flanges, or a neighbour's wood in the cloud.
    """

    vertices: np.ndarray
    faces: np.ndarray
    part: np.ndarray
    buttress_volume: float
    wood_volume: float
    top_z: float
    offset: float = float("nan")
    overhang: float = float("nan")

    @property
    def volume(self) -> float:
        """Whole-stem volume (m³): buttress plus the wood above it."""
        return self.buttress_volume + self.wood_volume

    def to_obj(self, path: str | Path) -> None:
        """Write the mesh as objects ``buttress`` and ``wood``.

        Parameters
        ----------
        path
            Output file.
        """
        _core.tree_mesh_write_obj(str(path), np.ascontiguousarray(self.vertices, dtype=float), _faces(self.faces),
                                  np.ascontiguousarray(self.part, dtype=np.uint8))

    def to_ply(self, path: str | Path, color=None) -> None:
        """Write the mesh as a binary PLY, the buttress darker than the wood.

        Parameters
        ----------
        path
            Output file.
        color
            One RGB triple (0-255) for every face; by default the buttress is
            bark brown and the wood keeps the stem colour.
        """
        _core.tree_mesh_write_ply(str(path), np.ascontiguousarray(self.vertices, dtype=float), _faces(self.faces),
                                  np.ascontiguousarray(self.part, dtype=np.uint8), _rgb(color))


def buttress_mesh(cloud: PointCloud, base_xy, ground_z: float | None = None,
                  height_attr: str = "height", resolution: float = 0.02,
                  slice_height: float = 0.05, close_radius: float = 0.08, max_radius: float = 4.0,
                  max_height: float = 6.0, top: float | None = None,
                  solidity: float = 0.9, max_flare: float = 1.0, smooth: int = 10) -> Buttress:
    """Rebuild a buttressed or otherwise irregular stem base as a closed mesh.

    A cylinder cannot follow a flanged base: a circle fitted to a star-shaped
    section misses the flanges or spans the gaps between them (at 1.3 m a big
    tropical tree can be a 3 m wide star that a circle explains 15 % of).
    Here the base is rebuilt volumetrically, so any shape works:

    1. The tree's points are cut into ``slice_height`` slices and rasterised
       at ``resolution``.
    2. A morphological closing of ``close_radius`` bridges gaps that occlusion
       leaves in the bark, and flood-filling from outside gives the solid
       cross-section. Where the outline does not close (part of the bark
       unseen), the seen bark is kept, thickened to ``close_radius`` (a flange
       seen from outside becomes a flange of that thickness), plus a circle
       where the points form a good arc of a round stem.
    3. Slices are built from the top down. A buttress only widens towards
       the ground, so each section contains the one above, and the part of a
       slice kept is the part connected to the section above, and no wider
       than ``max_flare`` allows: neighbouring stems, shrubs and logs stay
       out, litter and ground around the base are not closed into the solid,
       and the core carries down where near the ground only the outsides of
       the flanges were seen.
    4. The buttress ends where the section turns convex (solidity at or above
       ``solidity`` for four slices), unless ``top`` is given.
    5. The stacked sections become a watertight surface (surface nets,
       Gibson 1998, smoothed as in Taubin 1995); the volume is the sum of slice areas, with the
       boundary cells counted as half.

    Parameters
    ----------
    cloud
        One tree's points (or the plot's; only points within ``max_radius``
        of ``base_xy`` are used), with height above ground.
    base_xy
        Stem centre ``(x, y)``, e.g. ``(tree.x, tree.y)``.
    ground_z
        Terrain elevation at the stem, to place the mesh; ``z - height`` of the
        lowest points near the stem if None.
    height_attr
        Attribute holding height above ground.
    resolution
        Raster cell (m).
    slice_height
        Slice thickness (m).
    close_radius
        Gaps in the outline up to twice this wide are bridged (m). Larger
        closes more occlusion but also fills narrow gaps between flanges.
    max_radius
        Horizontal reach from the stem centre (m).
    max_height
        Highest possible top (m above ground).
    top
        Buttress top (m above ground); found from the solidity if None.
    solidity
        Solidity at which a section counts as a round stem.
    max_flare
        How fast a section may widen going down (m out per m down); the
        default 1.0 is 45°. Flanges flare well within this, while litter and
        the ground around the base do not, so a scan line that rings the stem
        is no longer closed into the solid. 0 lifts the limit.
    smooth
        Taubin smoothing passes over the mesh.

    Returns
    -------
    Buttress
        Empty (no faces, zero volume) if too few points are near the stem.

    Examples
    --------
    >>> b = qsm.buttress_mesh(tree, (t.x, t.y))
    >>> b.volume, b.top, b.total_volume(model)
    >>> b.to_obj("buttress.obj")
    """
    h = np.ascontiguousarray(cloud.heights(height_attr), dtype=float)
    d = _core.qsm_buttress_mesh(cloud.xyz, h, float(base_xy[0]), float(base_xy[1]),
                                None if ground_z is None else float(ground_z), resolution, slice_height,
                                close_radius, max_radius, max_height, top, solidity, 4, 0.5, 30,
                                float(max_flare), int(smooth))
    return _buttress(d)


def _buttress(d: dict) -> Buttress:
    return Buttress(d["vertices"], d["faces"], d["volume"], d["top"], d["top_z"], d["heights"],
                    d["areas"], d["solidities"], d["open"])


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
    meshes = list(meshes)
    names = [str(names[i]) if names else f"tree_{i + 1}" for i in range(len(meshes))]
    _core.write_obj(str(path), [(np.ascontiguousarray(v, dtype=float).reshape(-1, 3), _faces(f)) for v, f in meshes],
                    names)


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
    faces = np.ascontiguousarray(np.asarray(faces).astype(np.int32, copy=False).reshape(-1, 3))
    rgb = None
    if face_rgb is not None:
        rgb = np.ascontiguousarray(np.broadcast_to(np.asarray(face_rgb, dtype=np.uint8), (len(faces), 3)))
    _core.write_ply_mesh(str(path), np.ascontiguousarray(vertices, dtype=float).reshape(-1, 3), faces, rgb)


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


def _check_weights(weights, n: int, what: str = "weights") -> np.ndarray:
    """Per-point weights as a float array, checked: one per point, in [0, 1]."""
    w = np.asarray(weights)
    if w.dtype.kind not in "biuf":
        raise ValueError(f"{what} must be numbers in [0, 1]")
    w = np.ascontiguousarray(w, dtype=float)
    if w.ndim != 1 or len(w) != n:
        raise ValueError(f"{what} must have one value per point ({w.size} for {n} points)")
    bad = ~((w >= 0.0) & (w <= 1.0))
    if bad.any():
        raise ValueError(f"{what} must be finite and in [0, 1] (point {int(np.argmax(bad))} has {w[bad][0]})")
    return w


@dataclass
class PlotQSMs:
    """Every tree of a plot modelled; build with :func:`build_plot`.

    Attributes
    ----------
    models
        ``{tree_id: QSM}`` for the trees that were fitted.
    buttresses
        ``{tree_id: Buttress}`` where a buttress was found and meshed.
    skipped
        ``{tree_id: reason}`` for the trees that were not modelled: too few
        points, or the message of the fit that failed.
    """

    models: dict[int, QSM]
    buttresses: dict[int, "Buttress"]
    skipped: dict[int, str]

    def __len__(self) -> int:
        return len(self.models)

    def volume(self, tree_id: int) -> float:
        """Wood volume of one tree (m³), buttress included where there is one.

        Parameters
        ----------
        tree_id
            Which tree.

        Returns
        -------
        float
        """
        return _core.plot_total_volume([self._entry(tree_id, self.models[tree_id])])

    @property
    def total_volume(self) -> float:
        """Wood volume of the whole plot (m³)."""
        return _core.plot_total_volume(self._entries())

    def _entry(self, t: int, m: QSM) -> tuple:
        """One tree as the core reads a plot."""
        b = self.buttresses.get(t)
        if b is not None:
            b = (np.ascontiguousarray(b.vertices, dtype=float), _faces(b.faces), float(b.volume), float(b.top),
                 float(b.top_z))
        points = self._points.get(t)
        height = self._heights.get(t)
        return (int(t), np.ascontiguousarray(m.cylinders, dtype=float), None if points is None else int(points),
                None if height is None else float(height), b)

    def _entries(self) -> list:
        return [self._entry(t, m) for t, m in self.models.items()]

    def table(self) -> list[dict]:
        """One row per tree, ready for a CSV.

        Returns
        -------
        list of dict
            ``tree_id``, ``points``, ``volume_m3``, ``dbh_m``, ``height_m``,
            ``n_cylinders``, ``measured_volume`` and ``measured_length`` (the
            share of the model that was fitted to points rather than taken
            from the taper and pipe-model priors), and ``buttress_m3`` /
            ``buttress_top_m`` (blank without a buttress).
        """
        keys = ("tree_id", "points", "volume_m3", "dbh_m", "height_m", "n_cylinders", "measured_volume",
                "measured_length", "buttress_m3", "buttress_top_m")
        return [{k: "" if v is None else v for k, v in zip(keys, row, strict=True)}
                for row in _core.plot_table(self._entries())]

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as a CSV.

        Parameters
        ----------
        path
            Output file.
        """
        _core.plot_write_csv(str(path), self._entries())

    def write_meshes(self, directory: str | Path, fmt: str = "ply", sides: int = 12,
                     contiguous: bool = True, prefix: str = "tree") -> list[Path]:
        """Write a surface mesh per tree into ``directory``.

        A tree with a buttress is written fused (:meth:`Buttress.fuse`), so
        the flanged base and the cylinders above it come out as one file;
        every other tree is its cylinder mesh.

        Parameters
        ----------
        directory
            Created if it does not exist.
        fmt : {"ply", "obj"}
            PLY is binary and carries face colours; OBJ is text and keeps the
            buttress and the wood as named objects.
        sides
            Facets around each cylinder.
        contiguous
            One continuous tube per branch (see :meth:`QSM.mesh`).
        prefix
            File name stem; files are ``<prefix><tree_id>.<fmt>``.

        Returns
        -------
        list of pathlib.Path
            The files written, in tree order.

        Raises
        ------
        ValueError
            For an unknown format.
        """
        d = Path(directory)
        written = _core.plot_write_meshes(str(d), self._entries(), str(fmt), int(sides), bool(contiguous), str(prefix))
        return [d / Path(p).name for p in written]

    def write_cylinders(self, directory: str | Path, prefix: str = "tree") -> None:
        """Write one cylinder CSV per tree into ``directory``.

        Parameters
        ----------
        directory
            Created if it does not exist.
        prefix
            File name stem; files are ``<prefix><tree_id>.csv``.
        """
        _core.plot_write_cylinders(str(directory), self._entries(), str(prefix))

    #: filled in by build_plot
    _points: dict = None
    _heights: dict = None

    def __post_init__(self) -> None:
        if self._points is None:
            object.__setattr__(self, "_points", {})
        if self._heights is None:
            object.__setattr__(self, "_heights", {})


def build_plot(cloud: PointCloud, labels, stems=None, voxel_size: float = 0.01,
               wood=True, buttress: bool = False, min_points: int = 2000,
               height_attr: str = "height", **params) -> PlotQSMs:
    """A QSM for every tree of a segmented plot.

    The loop that :func:`build_qsm` needs around it: each tree's points are
    taken from ``labels``, thinned, put through the wood filter and fitted,
    and a tree that cannot be fitted is recorded rather than raising. Progress
    is reported (:mod:`sylva.util.progress`), so a plot of a few hundred trees is
    not silent.

    Parameters
    ----------
    cloud
        The whole plot, height-normalised (needed for ``buttress``).
    labels
        Tree id per point, as :func:`sylva.trees.segment_trees` returns;
        anything below 0 is not part of a tree.
    stems
        The detected trees, used for the stem centre each model is built
        around, and their DBH, which anchors each model's base radius
        (``base_radius = dbh / 2``) unless ``base_radius`` is given. Without
        that anchor, a small tree with a leafy crown can take its trunk
        radius from a foliage clump. Without stems the centre is the middle
        of the tree's own points between 0.5 and 1.5 m.
    voxel_size
        Thin each tree to this spacing first (m); 0 keeps every point.
    wood : bool, array or str
        Where each tree's wood comes from. True (default) runs
        :func:`wood_points` on each tree; False fits every point (clouds that
        are wood already). A boolean array with one value per point gives the
        wood directly (True is wood; for instance from
        :func:`sylva.leaves.classify_leaf_wood`): each tree is fitted on its
        wood points. A float array of weights in [0, 1] (a wood confidence,
        ``classify_leaf_wood(..., return_scores=True)``) fits each tree on all its points
        with those weights (``build_qsm(weights=)``). A string names a cloud
        attribute: integer or boolean values are labels (1 or True is wood, 0
        and -1 are not), floating-point values are weights. A tree's wood
        points are selected before they are thinned; with weights each point
        kept by the thinning keeps its own weight.
    buttress
        Look for a buttress on each tree (:func:`sylva.trees.detect_buttress`)
        and mesh it, so the volume of a flanged base is not left to the
        cylinders. Needs ``height_attr`` on the cloud.
    min_points
        Trees with fewer points than this are skipped.
    height_attr
        Attribute holding height above ground.
    **params
        Passed to :func:`build_qsm`.

    Returns
    -------
    PlotQSMs
        The models, any buttresses, and why a tree was skipped.

    Examples
    --------
    >>> labels = trees.segment_trees(cloud, stems)          # doctest: +SKIP
    >>> plot = qsm.build_plot(cloud, labels, stems)         # doctest: +SKIP
    >>> plot.total_volume, len(plot), plot.skipped          # doctest: +SKIP
    >>> plot.to_csv("trees.csv"); plot.write_cylinders("qsms/")   # doctest: +SKIP

    Notes
    -----
    Volumes are the cylinders' own unless a buttress was meshed, in which
    case :meth:`PlotQSMs.volume` is the mesh below its top plus the cylinders
    above it.

    A QSM needs points on the stem surface: with roughly 1 cm spacing the
    default 0.1 m shells hold plenty, but a cloud thinned to 3-5 cm leaves
    most shells with too few, and those cylinders take their radius from the
    taper and pipe-model priors instead. That runs large - on one 20 m
    savanna tree, 0.34 m DBH at full resolution against 1.10 m at 5 cm - so
    ``build_plot`` warns when the median model was hardly fitted at all, and
    ``measured_length`` in :meth:`PlotQSMs.table` says so per tree.
    """
    labels = np.asarray(labels)
    if len(labels) != len(cloud):
        raise ValueError("labels must have one value per point")
    wood_flag, wood_labels, wood_weights = _plot_wood(cloud, wood)
    if labels.dtype.kind == "f":
        whole = np.isnan(labels) | (labels == np.trunc(labels))
        if not whole.all():
            raise ValueError("labels must be whole numbers")
        labels = np.where(np.isnan(labels), -1, labels)
    qsm_params = _qsm_settings(params)
    stems = list(stems) if stems is not None else []
    heights = cloud.attrs.get(height_attr)
    d = _core.qsm_build_plot(cloud.xyz, np.ascontiguousarray(labels, dtype=np.int64),
                             None if heights is None else np.ascontiguousarray(heights, dtype=float),
                             [int(s.tree_id) for s in stems], [(float(s.x), float(s.y)) for s in stems],
                             [float(getattr(s, "dbh", np.nan)) for s in stems],
                             float(voxel_size), wood_flag, bool(buttress), float(min_points), qsm_params,
                             wood_labels=wood_labels, wood_weights=wood_weights)
    out = PlotQSMs({t: QSM(c) for t, c in d["models"]}, {t: _buttress(b) for t, b in d["buttresses"]},
                   dict(d["skipped"]))
    object.__setattr__(out, "_points", dict(d["points"]))
    object.__setattr__(out, "_heights", dict(d["heights"]))
    share = d["median_measured_length"]
    if share is not None:
        # A model whose cylinders were never fitted to points is the taper and
        # pipe-model priors talking, and those inflate. The usual cause is a
        # cloud too sparse for the shell width.
        if share < 0.1:
            warnings.warn(
                f"only {share:.0%} of the median model's length was fitted to points: "
                f"the cloud may be too sparse for bin_length={params.get('bin_length', 0.1)} m. "
                "Radii then come from the priors and run large; check measured_length in the table.",
                stacklevel=2,
            )
    return out


def _plot_wood(cloud: PointCloud, wood):
    """``build_plot``'s ``wood``: the filter switch, or per-point labels or weights."""
    if isinstance(wood, (bool, np.bool_)):
        return bool(wood), None, None
    if isinstance(wood, str):
        if wood not in cloud.attrs:
            raise ValueError(f"no attribute {wood!r} on the cloud (it has {sorted(cloud.attrs)})")
        values = np.asarray(cloud.attrs[wood])
        what = f"attribute {wood!r}"
    else:
        values = np.asarray(wood)
        what = "wood"
    if values.ndim != 1 or len(values) != len(cloud):
        raise ValueError(f"{what} must have one value per point ({values.size} for {len(cloud)} points)")
    if values.dtype.kind == "b":
        return True, np.ascontiguousarray(values), None
    if values.dtype.kind in "iu":
        if not np.isin(values, (-1, 0, 1)).all():
            raise ValueError(f"{what} must hold wood labels 1 (wood), 0 or -1 (not wood)")
        return True, np.ascontiguousarray(values == 1), None
    if values.dtype.kind == "f":
        return True, None, _check_weights(values, len(cloud), what)
    raise ValueError(f"{what} must be True/False, per-point labels or per-point weights")


def _qsm_settings(params: dict) -> dict:
    """Every :func:`build_qsm` setting, its default unless ``params`` sets it."""
    sig = inspect.signature(build_qsm)
    for k in ("cloud", "base_xy"):
        if k in params:
            raise TypeError(f"build_qsm() got multiple values for argument '{k}'")
    if "weights" in params:
        raise TypeError("build_plot() takes per-point weights as wood=, not weights=")
    sig.bind(None, **params)  # a TypeError for a setting build_qsm does not take
    out = {k: p.default for k, p in sig.parameters.items() if k not in ("cloud", "base_xy", "weights")}
    out.update(params)
    return out


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
