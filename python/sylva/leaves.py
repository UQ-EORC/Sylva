"""Foliage for a structure model: leaf/wood labels, leaf angle distribution,
leaf area density, and leaf meshes placed on a QSM.

A QSM describes the wood. For radiative transfer or visualisation the leaves
are added as flat polygons whose total area, spatial distribution and
orientation follow what was measured::

    from sylva import leaves, qsm

    wood = leaves.classify_leaf_wood(tree)              # bool per point
    model = qsm.build_qsm(qsm.wood_points(tree))
    foliage = tree[~wood]
    angles = leaves.leaf_angle_distribution(foliage)     # from point normals
    area = leaves.leaf_area_density(foliage, voxel_size=0.25)
    mesh = leaves.add_leaves(model, area, angles, leaf_points=foliage)
    leaves.write_tree_obj("tree.obj", model, mesh)

Leaf area from points counts what the scanner saw, so occluded foliage is
missing and the figure is a lower bound. With pulse data, take the leaf area
density of a ray-traced voxel grid instead (:meth:`LeafAreaGrid.from_voxels`),
or scale to a known total with :meth:`LeafAreaGrid.scaled_to`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .qsm import QSM, write_obj

__all__ = ["classify_leaf_wood", "LeafAngleDistribution", "leaf_angle_distribution",
           "LeafAreaGrid", "leaf_area_density", "LeafMesh", "add_leaves", "write_tree_obj",
           "single_leaf_area"]


def _xyz(points) -> np.ndarray:
    return np.ascontiguousarray(points.xyz if isinstance(points, PointCloud) else points, dtype=float)


def classify_leaf_wood(cloud: PointCloud, voxel_size: float = 0.02, method: str = "gbs", **wood_params) -> np.ndarray:
    """Wood (``True``) / leaf (``False``) for every point of one tree.

    ``method="gbs"`` (default) is the graph-based separation of Tian et al.
    (2022): edges joining points that grow in different directions are cut,
    the cut graph is split into shells of path length at several
    ``intervals`` and a piece is wood when it spans its shell and is
    cylindrical or linear, with a taper check down its path. By default the
    shells are 0.1-1 m with a 45 deg direction limit for trees under 15 m and
    0.5-3 m with 27 deg for taller ones (the authors' two settings). Options:
    ``intervals``, ``max_angle``, ``linearity``, ``circle_error``,
    ``graph_k``, ``max_edge``, ``min_points``. On 30 manually labelled
    tropical trees (Van den Broeck et al. 2025) it scores accuracy 0.90 and
    mIoU 0.79; the passage filter below scores 0.68 and 0.51.

    ``method="passage"``:

    The leaf/wood filter of :func:`sylva.qsm.wood_points` (path passage from
    the base plus local anisotropy) runs on the cloud thinned to
    ``voxel_size``, here with a second anisotropy scale of 10 cm
    (``scale_radius``): a single leaf is as planar as bark over a few
    centimetres, but wider than a leaf foliage is a jumble of orientations.
    Each point takes the label of its nearest thinned neighbour.
    ``wood_params`` are that filter's options (``threshold`` is passed as
    ``high_threshold``).

    """
    if method == "gbs":
        xyz = _xyz(cloud)
        if wood_params.get("intervals") is None and len(xyz):
            # Shell sizes follow the tree: the authors' two settings, switched on height.
            tall = float(np.ptp(xyz[:, 2])) >= 15.0
            wood_params["intervals"] = [0.5, 1.0, 1.5, 2.0, 3.0] if tall else [0.1, 0.2, 0.3, 0.5, 1.0]
            wood_params.setdefault("max_angle", 0.15 * np.pi if tall else 0.25 * np.pi)
        return _core.classify_leaf_wood_gbs(xyz, float(voxel_size), **wood_params)
    if method != "passage":
        raise ValueError("method must be 'passage' or 'gbs'")
    if "threshold" in wood_params:
        wood_params["high_threshold"] = wood_params.pop("threshold")
    return _core.classify_leaf_wood(_xyz(cloud), float(voxel_size), **wood_params)


@dataclass
class LeafAngleDistribution:
    """Leaf inclination angle distribution (inclination = angle between the
    leaf normal and the vertical; 0 deg is a horizontal leaf)."""

    bin_centres: np.ndarray  #: rad, over [0, pi/2]
    density: np.ndarray  #: probability per bin, sums to 1
    mean: float  #: rad
    std: float  #: rad
    beta_a: float  #: beta distribution on t = 2 theta / pi: f(t) ~ t^(a-1) (1-t)^(b-1)
    beta_b: float
    chi: float  #: Campbell's ellipsoidal parameter (1 spherical, > 1 planophile)
    de_wit: str | None  #: nearest de Wit type

    @property
    def mean_deg(self) -> float:
        return float(np.degrees(self.mean))

    def g(self, beam_zenith) -> np.ndarray:
        """Projection function G for beams at the given zenith angles (rad)."""
        z = np.atleast_1d(np.asarray(beam_zenith, dtype=float))
        return np.array(_core.leaf_projection_histogram(self.bin_centres, self.density, z))

    @classmethod
    def from_type(cls, name: str = "spherical", n_bins: int = 18) -> "LeafAngleDistribution":
        """An analytic de Wit distribution: ``spherical``, ``uniform``,
        ``planophile``, ``erectophile``, ``plagiophile`` or ``extremophile``."""
        t = (np.arange(n_bins) + 0.5) * (np.pi / 2) / n_bins
        f = {"spherical": np.sin(t), "uniform": np.full(n_bins, 1.0),
             "planophile": 1 + np.cos(2 * t), "erectophile": 1 - np.cos(2 * t),
             "plagiophile": 1 - np.cos(4 * t), "extremophile": 1 + np.cos(4 * t)}[name]
        f = f / f.sum()
        return leaf_angle_distribution(t, weights=f, n_bins=n_bins, inclinations=True)


def leaf_angle_distribution(leaf_points, k: int = 12, n_bins: int = 18, weights=None,
                            inclinations: bool = False, res: float | None = 0.0) -> LeafAngleDistribution:
    """Leaf angle distribution from leaf points.

    Normals come from a PCA over ``k`` neighbours. By default the points are
    first thinned to ``res`` (0: chosen from the point spacing) and each is weighted by the leaf area it stands
    for, so densely scanned leaves do not dominate; pass ``res=None`` to use
    every point equally. With ``inclinations=True`` the first argument is an
    array of inclinations (rad) instead of points.
    """
    if inclinations:
        incl = np.ascontiguousarray(leaf_points, dtype=float)
        w = None if weights is None else np.ascontiguousarray(weights, dtype=float)
    elif res is not None:
        _, w, incl = _core.point_leaf_area(_xyz(leaf_points), float(res), int(k))
    else:
        incl, _ = _core.leaf_inclinations(_xyz(leaf_points), int(k))
        w = None if weights is None else np.ascontiguousarray(weights, dtype=float)
    d = _core.leaf_angle_distribution(incl, w, int(n_bins))
    return LeafAngleDistribution(**d)


@dataclass
class LeafAreaGrid:
    """One-sided leaf area density on a regular grid."""

    origin: np.ndarray  #: min corner (3,)
    voxel_size: float
    density: np.ndarray  #: m2 m-3, shape (nz, ny, nx)

    @property
    def area(self) -> np.ndarray:
        """Leaf area per voxel (m2)."""
        return self.density * self.voxel_size**3

    @property
    def total_area(self) -> float:
        return float(np.nansum(self.area))

    def scaled_to(self, total_area: float) -> "LeafAreaGrid":
        """Same spatial pattern, total leaf area set to ``total_area`` (m2)."""
        t = self.total_area
        return LeafAreaGrid(self.origin, self.voxel_size, self.density * (total_area / t if t > 0 else 0.0))

    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Height of each layer's centre and its leaf area (m2)."""
        z = self.origin[2] + (np.arange(self.density.shape[0]) + 0.5) * self.voxel_size
        return z, np.nansum(self.area, axis=(1, 2))

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Centres and leaf area of the voxels that hold any."""
        a = np.nan_to_num(self.area)
        k, j, i = np.nonzero(a > 0)
        centres = self.origin + (np.column_stack([i, j, k]) + 0.5) * self.voxel_size
        return centres, a[k, j, i]

    @classmethod
    def from_voxels(cls, grid, field: str = "pad_fpl") -> "LeafAreaGrid":
        """Leaf (or plant) area density of a :class:`sylva.voxels.RayVoxelGrid`
        field, e.g. a leaf area density computed with ``leaf_classes``."""
        return cls(np.asarray(grid.origin, float), float(grid.voxel_size), np.nan_to_num(np.asarray(grid[field], float)))


def leaf_area_density(leaf_points, voxel_size: float = 0.25, res: float | None = None, k: int = 12) -> LeafAreaGrid:
    """Leaf area density from leaf points alone.

    The points are thinned to one per cube of side ``res``; a surface with
    unit normal ``n`` crosses ``(|nx| + |ny| + |nz|) / res**2`` such cubes per
    unit area, so each survivor stands for ``res**2 / (|nx| + |ny| + |nz|)``.
    ``res`` must be a few times the point spacing (smaller cubes fall between
    the points and undercount) and well under the leaf size (larger ones
    overcount at the edges); by default it is 3.5 times the median
    nearest-neighbour spacing, within about 15 % on unoccluded synthetic
    leaves. Only foliage the scanner saw is counted.
    """
    pts, area, _ = _core.point_leaf_area(_xyz(leaf_points), float(res or 0.0), int(k))
    if len(pts) == 0:
        return LeafAreaGrid(np.zeros(3), float(voxel_size), np.zeros((1, 1, 1)))
    # Align the grid to multiples of the voxel size so cells match add_leaves' hashing.
    origin = np.floor(pts.min(0) / voxel_size) * voxel_size
    idx = np.floor((pts - origin) / voxel_size).astype(int)
    shape = idx.max(0) + 1
    dens = np.zeros((shape[2], shape[1], shape[0]))
    np.add.at(dens, (idx[:, 2], idx[:, 1], idx[:, 0]), area)
    return LeafAreaGrid(origin, float(voxel_size), dens / voxel_size**3)


def single_leaf_area(length: float, width: float) -> float:
    """One-sided area (m2) of the leaf shape used by :func:`add_leaves`."""
    x, y = np.array(_OUTLINE).T
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * length * width)


#: Unit leaf outline (along, across), as in the core.
_OUTLINE = [(0.0, 0.0), (0.2, 0.36), (0.45, 0.5), (1.0, 0.0), (0.45, -0.5), (0.2, -0.36)]


@dataclass
class LeafMesh:
    vertices: np.ndarray  #: (V, 3)
    faces: np.ndarray  #: (F, 3) into ``vertices``
    centres: np.ndarray  #: (L, 3) one per leaf
    normals: np.ndarray  #: (L, 3)
    inclination: np.ndarray  #: (L,) rad
    cylinder: np.ndarray  #: (L,) index of the nearest QSM cylinder, -1 if none in reach
    leaf_area: float  #: one-sided area of one leaf (m2)

    def __len__(self) -> int:
        return len(self.centres)

    @property
    def total_area(self) -> float:
        return self.leaf_area * len(self)

    def to_obj(self, path: str | Path) -> None:
        write_obj(path, [(self.vertices, self.faces)], names=["leaves"])


def add_leaves(model: QSM | None, leaf_area: LeafAreaGrid | float, angles: LeafAngleDistribution | str = "spherical",
               leaf_points=None, leaf_length: float = 0.08, leaf_width: float = 0.04,
               max_branch_distance: float = 0.5, jitter: float = 0.01, seed: int = 1) -> LeafMesh:
    """Leaf polygons for a QSM.

    ``leaf_area`` is a :class:`LeafAreaGrid`, or a total (m2) to spread over
    ``leaf_points`` in proportion to the area they show. Each voxel receives
    leaves until its area is met, centred on leaf points of that voxel where
    there are any (uniformly inside it otherwise), with normals drawn from
    ``angles`` (a distribution or a de Wit type name) and a uniform azimuth;
    blades within ``max_branch_distance`` of a cylinder point away from it.
    Leaves may intersect: no collision test is made.
    """
    if isinstance(angles, str):
        angles = LeafAngleDistribution.from_type(angles)
    seeds = np.zeros((0, 3)) if leaf_points is None else _xyz(leaf_points)
    if not isinstance(leaf_area, LeafAreaGrid):
        if len(seeds) == 0:
            raise ValueError("a total leaf area needs leaf_points to distribute it over")
        leaf_area = leaf_area_density(seeds).scaled_to(float(leaf_area))
    centres, area = leaf_area.cells()
    cyl = np.zeros((0, 12)) if model is None else np.ascontiguousarray(model.cylinders, dtype=float)
    # The core hashes points into cells from the coordinate origin: work in the grid's frame.
    o = np.asarray(leaf_area.origin, float)
    cyl = cyl.copy()
    cyl[:, 0:3] -= o
    d = _core.insert_leaves(np.ascontiguousarray(centres - o), np.ascontiguousarray(area), float(leaf_area.voxel_size),
                            np.ascontiguousarray(seeds - o), angles.bin_centres, angles.density, cyl, float(leaf_length),
                            float(leaf_width), float(max_branch_distance), float(jitter), int(seed))
    d["vertices"] = d["vertices"] + o
    d["centres"] = d["centres"] + o
    return LeafMesh(**d)


def write_tree_obj(path: str | Path, model: QSM, leaf_mesh: LeafMesh, sides: int = 12) -> None:
    """Wood cylinders and leaves in one OBJ, as two named objects."""
    v, f, _ = model.mesh(sides)
    write_obj(path, [(v, f), (leaf_mesh.vertices, leaf_mesh.faces)], names=["wood", "leaves"])
