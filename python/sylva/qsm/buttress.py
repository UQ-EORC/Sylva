# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Buttressed stem bases as closed meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from .mesh import _faces, _rgb, write_obj, write_ply_mesh

if TYPE_CHECKING:
    from .model import QSM


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
