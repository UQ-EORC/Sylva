# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Quantitative structure models: skeletonisation and cylinder fitting.

A QSM is a set of connected cylinders. :func:`build_qsm` bins geodesic
distance from the base over a kNN graph, splits bins into connected segments,
fits a RANSAC cylinder to each and links parents.
"""

from __future__ import annotations

import csv
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
        v, f, _ = self.mesh(sides, contiguous)
        write_obj(path, [(v, f)])

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
        z0, z1 = self.start[:, 2], self.end[:, 2]
        lo, hi = np.minimum(z0, z1), np.maximum(z0, z1)
        span = np.maximum(hi - lo, 1e-12)
        share = np.where(hi <= z, 0.0, np.where(lo >= z, 1.0, (hi - z) / span))
        return float((self.volumes * share).sum())

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
        rows = self.cylinders.copy()
        s, a, L = self.start, self.axis, self.column("length")
        e = self.end
        keep = np.maximum(s[:, 2], e[:, 2]) > z
        dz = a[:, 2] * L
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(dz) > 1e-12, (z - s[:, 2]) / np.where(np.abs(dz) > 1e-12, dz, 1.0), 0.0)
        cut = keep & (s[:, 2] < z) & (np.abs(dz) > 1e-12)  # starts below, so trim the base
        t = np.clip(t, 0.0, 1.0)
        rows[cut, 0:3] = s[cut] + a[cut] * (t[cut] * L[cut])[:, None]
        rows[cut, 6] = L[cut] * (1.0 - t[cut])
        # Renumber: a dropped parent leaves its children as branch bases.
        idx = np.full(len(self), -1)
        idx[keep] = np.arange(int(keep.sum()))
        rows = rows[keep]
        par = rows[:, 8].astype(int)
        rows[:, 8] = np.where(par >= 0, idx[np.clip(par, 0, len(idx) - 1)], -1)
        return QSM(rows)

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
        v, f, owner = self.mesh(sides, contiguous)
        if color is None:
            order = self.column("branch_order").astype(int)
            face_rgb = _ORDER_COLORS[np.minimum(order[owner], len(_ORDER_COLORS) - 1)]
        else:
            face_rgb = np.tile(np.asarray(color, dtype=np.uint8), (len(f), 1))
        write_ply_mesh(path, v, f, face_rgb)


_ORDER_COLORS = np.array([[139, 90, 43], [205, 133, 63], [222, 184, 135], [60, 179, 113],
                          [46, 139, 87], [34, 139, 34]], dtype=np.uint8)


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
        wv, wf, _ = model.above(self.top_z - float(overlap)).mesh(sides, contiguous)
        v = np.vstack([self.vertices, wv]) if len(wv) else np.asarray(self.vertices, float)
        f = np.vstack([self.faces, wf + len(self.vertices)]) if len(wf) else np.asarray(self.faces)
        part = np.concatenate([np.zeros(len(self.faces), np.uint8), np.ones(len(wf), np.uint8)])
        # How the two meet: the stem should sit inside the base at the join.
        base = _section(np.asarray(self.vertices, float), np.asarray(self.faces), self.top_z - 0.05)
        stem = _section(wv, wf, self.top_z + 0.05) if len(wf) else np.zeros((0, 2, 2))
        offset, overhang = _join_fit(base, stem)
        return TreeMesh(v, f.astype(np.uint32), part, self.volume,
                        model.volume_above(self.top_z), self.top_z, offset, overhang)

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
    t = vertices[faces]
    d = t[:, :, 2] - z
    hit = ~((d > 0).all(1) | (d < 0).all(1))
    t, d = t[hit], d[hit]
    if not len(t):
        return np.zeros((0, 2, 2))
    ends = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        cross = (da > 0) != (db > 0)
        w = np.where(cross, da / np.where(da == db, 1e-12, da - db), np.nan)[:, None]
        ends.append((t[:, a] + w * (t[:, b] - t[:, a]))[:, :2])
    e = np.stack(ends, 1)
    ok = ~np.isnan(e).any(2)
    keep = ok.sum(1) >= 2
    e, ok = e[keep], ok[keep]
    first = np.argmax(ok, 1)
    second = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], 1)
    i = np.arange(len(e))
    return np.stack([e[i, first], e[i, second]], 1)


def _inside(section: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Even-odd test of ``points`` against a soup of segments (:func:`_section`)."""
    hits = np.zeros(len(points), int)
    for (x0, y0), (x1, y1) in section:
        if y0 == y1:
            continue
        lo, hi = min(y0, y1), max(y0, y1)
        m = (points[:, 1] >= lo) & (points[:, 1] < hi)
        if not m.any():
            continue
        f = (points[m, 1] - y0) / (y1 - y0)
        hits[m] += (x0 + f * (x1 - x0)) > points[m, 0]
    return hits % 2 == 1


def _join_fit(base: np.ndarray, wood: np.ndarray, cell: float = 0.02) -> tuple[float, float]:
    """How well the wood sits inside the base: centre offset (m) and the share
    of the wood's cross-section outside it."""
    if not len(base) or not len(wood):
        return float("nan"), float("nan")
    offset = float(np.linalg.norm(wood.reshape(-1, 2).mean(0) - base.reshape(-1, 2).mean(0)))
    lo = wood.reshape(-1, 2).min(0) - cell
    hi = wood.reshape(-1, 2).max(0) + cell
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], cell), np.arange(lo[1], hi[1], cell))
    p = np.column_stack([gx.ravel(), gy.ravel()])
    inw = _inside(wood, p)
    if not inw.any():
        return offset, float("nan")
    return offset, float((inw & ~_inside(base, p)).sum() / inw.sum())


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
        parts, names = [], []
        for k, name in ((0, "buttress"), (1, "wood")):
            f = self.faces[self.part == k]
            if len(f):
                used, inv = np.unique(f, return_inverse=True)
                parts.append((self.vertices[used], inv.reshape(f.shape)))
                names.append(name)
        write_obj(path, parts, names=names)

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
        if color is None:
            face_rgb = np.where(self.part[:, None] == 0, _BUTTRESS_COLOR, _ORDER_COLORS[0]).astype(np.uint8)
        else:
            face_rgb = np.tile(np.asarray(color, dtype=np.uint8), (len(self.faces), 1))
        write_ply_mesh(path, self.vertices, self.faces, face_rgb)


_BUTTRESS_COLOR = np.array([101, 67, 33], dtype=np.uint8)


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
       Taubin-smoothed); the volume is the sum of slice areas, with the
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
    cx, cy = float(base_xy[0]), float(base_xy[1])
    if ground_z is None:
        near = np.hypot(cloud.x - cx, cloud.y - cy) <= 1.0
        base = (cloud.z - h)[near] if near.any() else cloud.z - h
        ground_z = float(np.median(base))
    d = _core.buttress_mesh(cloud.xyz, h, cx, cy, float(ground_z), resolution, slice_height,
                            close_radius, max_radius, max_height, top, solidity, 4, 0.5, 30,
                            float(max_flare), int(smooth))
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
              branch_min_inlier_fraction: float = 0.3, spacing_scale: float = 1.5,
              radius_power: float = 0.0, power_above_spacing: float = 0.025,
              sensor_noise: float = 0.02, cluster_eps: float = 0.1,
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
                        branch_min_inlier_fraction, spacing_scale, radius_power, power_above_spacing, sensor_noise,
                        cluster_eps, centre_fit_points,
                        radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius)
    return QSM(d["cylinders"])


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
        m = self.models[tree_id]
        b = self.buttresses.get(tree_id)
        return b.total_volume(m) if b is not None else m.total_volume

    @property
    def total_volume(self) -> float:
        """Wood volume of the whole plot (m³)."""
        return float(sum(self.volume(t) for t in self.models))

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
        rows = []
        for t, m in sorted(self.models.items()):
            b = self.buttresses.get(t)
            s = m.summary()
            fit = m.metrics()
            rows.append({
                "tree_id": t,
                "points": self._points.get(t, ""),
                "volume_m3": round(self.volume(t), 5),
                "dbh_m": round(s["dbh_m"], 4),
                "height_m": round(self._heights.get(t, float("nan")), 2),
                "n_cylinders": s["n_cylinders"],
                "measured_volume": round(fit["measured_volume_fraction"], 3),
                "measured_length": round(fit["measured_length_fraction"], 3),
                "buttress_m3": round(b.volume, 5) if b is not None else "",
                "buttress_top_m": round(b.top, 2) if b is not None else "",
            })
        return rows

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as a CSV.

        Parameters
        ----------
        path
            Output file.
        """
        rows = self.table()
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]) if rows else ["tree_id"])
            w.writeheader()
            w.writerows(rows)

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
        if fmt not in ("ply", "obj"):
            raise ValueError("fmt must be 'ply' or 'obj'")
        from . import progress

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        out = []
        with progress.task("writing meshes", len(self.models)) as prog:
            for t, m in sorted(self.models.items()):
                path = d / f"{prefix}{t}.{fmt}"
                b = self.buttresses.get(t)
                mesh = b.fuse(m, sides=sides, contiguous=contiguous) if b is not None else None
                if mesh is not None:
                    mesh.to_ply(path) if fmt == "ply" else mesh.to_obj(path)
                elif fmt == "ply":
                    m.to_ply(path, sides=sides, contiguous=contiguous)
                else:
                    m.to_obj(path, sides=sides, contiguous=contiguous)
                out.append(path)
                prog.update()
        return out

    def write_cylinders(self, directory: str | Path, prefix: str = "tree") -> None:
        """Write one cylinder CSV per tree into ``directory``.

        Parameters
        ----------
        directory
            Created if it does not exist.
        prefix
            File name stem; files are ``<prefix><tree_id>.csv``.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for t, m in self.models.items():
            m.to_csv(d / f"{prefix}{t}.csv")

    #: filled in by build_plot
    _points: dict = None
    _heights: dict = None

    def __post_init__(self) -> None:
        if self._points is None:
            object.__setattr__(self, "_points", {})
        if self._heights is None:
            object.__setattr__(self, "_heights", {})


def build_plot(cloud: PointCloud, labels, stems=None, voxel_size: float = 0.01,
               wood: bool = True, buttress: bool = False, min_points: int = 2000,
               height_attr: str = "height", **params) -> PlotQSMs:
    """A QSM for every tree of a segmented plot.

    The loop that :func:`build_qsm` needs around it: each tree's points are
    taken from ``labels``, thinned, put through the wood filter and fitted,
    and a tree that cannot be fitted is recorded rather than raising. Progress
    is reported (:mod:`sylva.progress`), so a plot of a few hundred trees is
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
        around. Without them the centre is the middle of the tree's own
        points between 0.5 and 1.5 m.
    voxel_size
        Thin each tree to this spacing first (m); 0 keeps every point.
    wood
        Run :func:`wood_points` on each tree first. Turn it off for clouds
        that are wood already.
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
    from . import filters, progress, trees as _trees

    labels = np.asarray(labels)
    if len(labels) != len(cloud):
        raise ValueError("labels must have one value per point")
    ids = [int(t) for t in np.unique(labels) if t >= 0]
    centres = {int(s.tree_id): (s.x, s.y) for s in stems} if stems is not None else {}
    heights = cloud.attrs.get(height_attr)
    models, bases, skipped, points, tops = {}, {}, {}, {}, {}
    with progress.task("fitting QSMs", len(ids)) as prog:
        for tid in ids:
            prog.update()
            m = labels == tid
            n = int(m.sum())
            if n < min_points:
                skipped[tid] = f"{n} points"
                continue
            tree = cloud[m]
            points[tid] = n
            if heights is not None:
                tops[tid] = float(heights[m].max())
            thin = filters.voxel_downsample(tree, voxel_size) if voxel_size > 0 else tree
            base = centres.get(tid)
            if base is None:
                h = thin.attrs.get(height_attr)
                low = thin.xyz[(h > 0.5) & (h < 1.5)] if h is not None else thin.xyz
                base = tuple(np.median(low[:, :2] if len(low) > 20 else thin.xyz[:, :2], axis=0))
            try:
                models[tid] = build_qsm(wood_points(thin) if wood else thin, base_xy=base, **params)
            except (ValueError, RuntimeError) as e:
                skipped[tid] = str(e)
                continue
            if buttress and heights is not None:
                found = _trees.detect_buttress(tree, base_xy=base, height_attr=height_attr)
                if found["buttressed"]:
                    b = buttress_mesh(tree, found["centre"], height_attr=height_attr, top=found["top"])
                    if len(b.faces):
                        bases[tid] = b
    out = PlotQSMs(models, bases, skipped)
    object.__setattr__(out, "_points", points)
    object.__setattr__(out, "_heights", tops)
    if models:
        # A model whose cylinders were never fitted to points is the taper and
        # pipe-model priors talking, and those inflate. The usual cause is a
        # cloud too sparse for the shell width.
        share = float(np.median([m.metrics()["measured_length_fraction"] for m in models.values()]))
        if share < 0.1:
            warnings.warn(
                f"only {share:.0%} of the median model's length was fitted to points: "
                f"the cloud may be too sparse for bin_length={params.get('bin_length', 0.1)} m. "
                "Radii then come from the priors and run large; check measured_length in the table.",
                stacklevel=2,
            )
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
