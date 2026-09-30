# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
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
from .qsm import QSM

__all__ = ["classify_leaf_wood", "LeafAngleDistribution", "leaf_angle_distribution",
           "LeafAreaGrid", "leaf_area_density", "LeafMesh", "LeafShape", "add_leaves", "write_tree_obj",
           "single_leaf_area", "default_leaf", "set_default_leaf"]


def _xyz(points) -> np.ndarray:
    return np.ascontiguousarray(points.xyz if isinstance(points, PointCloud) else points, dtype=float)


def classify_leaf_wood(cloud: PointCloud, voxel_size: float = 0.02, method: str = "gbs",
                       return_scores: bool = False, **wood_params):
    """Wood (``True``) / leaf (``False``) for every point of one tree.

    ``method="gbs"`` (default) is the graph-based separation of Tian and Li
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

    ``return_scores=True`` adds a wood confidence in [0, 1] per point, a
    weight for :func:`sylva.qsm.build_qsm` (``weights=``) and
    :func:`sylva.qsm.build_plot` (``wood=``). For ``"passage"`` it is 1
    where the filter keeps the point and 0.4 times the point's anisotropy
    (planarity + linearity) where it drops it: dropped points still count a
    little towards a circle but are never confident wood. For ``"gbs"`` it is
    half the label plus half the share of the shell scales (``intervals``) at
    which the point's piece was classified wood: 1 for a piece that is
    cylindrical or linear at every scale, 0.5 for a point that is wood only
    by lying on a path or next to wood, 0 for a leaf. The labels are
    unchanged.

    Parameters
    ----------
    cloud
        One segmented tree.
    voxel_size
        Work on the cloud thinned to this spacing (m); every input point gets
        the label of its nearest thinned point. 0 uses every point.
    method : {"gbs", "passage"}
        Classifier; see above.
    return_scores
        Also return the wood confidence per point (see above).
    **wood_params
        For ``"gbs"``: ``intervals`` (shell thicknesses, m), ``max_angle``
        (rad), ``linearity`` [0.9], ``circle_error`` [0.2, relative to the
        radius], ``graph_k`` [8], ``max_edge`` [1.0], ``base_height``
        [0.25], ``min_points`` [10]. For ``"passage"``: the options of
        :func:`sylva.qsm.wood_points`.

    Returns
    -------
    numpy.ndarray or tuple
        Boolean per input point, True for wood; with ``return_scores``, the
        tuple ``(labels, confidence)``, the confidence a float per point.

    Raises
    ------
    ValueError
        For an unknown method.

    See Also
    --------
    sylva.qsm.wood_points : the filter that prepares QSM input.
    """
    if method == "gbs":
        # Shell sizes follow the tree (the authors' two settings, switched on height).
        if return_scores:
            mask, confidence, _ = _core.classify_leaf_wood_gbs_scores(_xyz(cloud), float(voxel_size), **wood_params)
            return mask, confidence
        return _core.classify_leaf_wood_gbs(_xyz(cloud), float(voxel_size), **wood_params)
    if method != "passage":
        raise ValueError("method must be 'passage' or 'gbs'")
    if "threshold" in wood_params:
        wood_params["high_threshold"] = wood_params.pop("threshold")
    if return_scores:
        mask, confidence, _, _ = _core.classify_leaf_wood_scores(_xyz(cloud), float(voxel_size), **wood_params)
        return mask, confidence
    return _core.classify_leaf_wood(_xyz(cloud), float(voxel_size), **wood_params)


@dataclass
class LeafAngleDistribution:
    """Leaf inclination angle distribution (inclination = angle between the
    leaf normal and the vertical; 0 deg is a horizontal leaf).

    Build with :func:`leaf_angle_distribution` from points, or
    :meth:`from_type` for a textbook distribution.

    Attributes
    ----------
    bin_centres
        Inclination bin centres over 0-π/2 (rad).
    density
        Probability per bin, sums to 1.
    mean, std
        Mean and standard deviation of the inclination (rad).
    beta_a, beta_b
        Beta distribution fitted on t = 2θ/π (Goel and Strebel 1984).
    chi
        Campbell's (1990) ellipsoidal parameter (1 spherical, > 1 planophile).
    de_wit
        Nearest de Wit (1965) type by name.
    """

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
        """Mean leaf inclination in degrees (57.3 for spherical)."""
        return float(np.degrees(self.mean))

    def g(self, beam_zenith) -> np.ndarray:
        """Leaf projection function G for this distribution (Wilson 1960).

        Parameters
        ----------
        beam_zenith
            Beam zenith angle(s) in radians.

        Returns
        -------
        numpy.ndarray
            G per angle (0.5 everywhere for a spherical distribution).
        """
        z = np.atleast_1d(np.asarray(beam_zenith, dtype=float))
        return np.array(_core.leaf_projection_histogram(self.bin_centres, self.density, z))

    @classmethod
    def from_type(cls, name: str = "spherical", n_bins: int = 18) -> "LeafAngleDistribution":
        """A textbook de Wit (1965) distribution.

        Parameters
        ----------
        name
            ``spherical``, ``uniform``, ``planophile``, ``erectophile``,
            ``plagiophile`` or ``extremophile``.
        n_bins
            Bins over 0-90 degrees.

        Returns
        -------
        LeafAngleDistribution

        Raises
        ------
        KeyError
            For an unknown name.
        """
        return cls(**_core.leaf_de_wit(name, int(n_bins)))


def leaf_angle_distribution(leaf_points, k: int = 12, n_bins: int = 18, weights=None,
                            inclinations: bool = False, res: float | None = 0.0) -> LeafAngleDistribution:
    """Leaf angle distribution from leaf points.

    Normals come from a PCA over ``k`` neighbours, as in Vicari et al.
    (2019). By default the points are
    first thinned to ``res`` (0: chosen from the point spacing) and each is weighted by the leaf area it stands
    for, so densely scanned leaves do not dominate; pass ``res=None`` to use
    every point equally. With ``inclinations=True`` the first argument is an
    array of inclinations (rad) instead of points.

    Parameters
    ----------
    leaf_points
        Leaf points of a tree or plot (:class:`~sylva.PointCloud` or
        ``(N, 3)``), e.g. ``tree[~classify_leaf_wood(tree)]``; or
        inclinations with ``inclinations=True``.
    k
        Neighbours for the PCA normals. Too many blur across leaves; too
        few follow scanner noise.
    n_bins
        Histogram bins over 0-90 degrees.
    weights
        Weights for inclinations or for unthinned points (``res=None``).
    inclinations
        Treat the first argument as inclinations.
    res
        Thinning cube size (m); 0 picks 3.5 × the median spacing; None
        disables thinning and area weighting.

    Returns
    -------
    LeafAngleDistribution

    Notes
    -----
    Range noise tilts point normals towards random, which biases planophile
    canopies towards spherical. Use well-registered, close-range data.
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
    """One-sided leaf area density on a regular grid.

    Build with :func:`leaf_area_density` (from points) or
    :meth:`from_voxels` (from a ray-traced grid).

    Attributes
    ----------
    origin
        Minimum corner, ``(x, y, z)``.
    voxel_size
        Voxel edge (m).
    density
        Leaf area density (m² m⁻³), ``(nz, ny, nx)``.
    """

    origin: np.ndarray  #: min corner (3,)
    voxel_size: float
    density: np.ndarray  #: m2 m-3, shape (nz, ny, nx)

    def _args(self):
        return (np.ascontiguousarray(self.origin, dtype=float), float(self.voxel_size),
                np.ascontiguousarray(self.density, dtype=float))

    @property
    def area(self) -> np.ndarray:
        """Leaf area per voxel (m²), shaped like ``density``."""
        return _core.leaf_grid_area(*self._args())

    @property
    def total_area(self) -> float:
        """Total one-sided leaf area (m²)."""
        return _core.leaf_grid_total_area(*self._args())

    def scaled_to(self, total_area: float) -> "LeafAreaGrid":
        """Rescale to a known total leaf area, keeping the spatial pattern.

        Use with an independent total, e.g. from allometry, litter traps or
        a ray-traced grid, since point-based totals miss occluded foliage.

        Parameters
        ----------
        total_area
            Target one-sided leaf area (m²).

        Returns
        -------
        LeafAreaGrid
        """
        return LeafAreaGrid(self.origin, self.voxel_size, _core.leaf_grid_scaled(*self._args(), float(total_area)))

    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Vertical leaf area profile.

        Returns
        -------
        z, area : numpy.ndarray
            z of each layer's centre (grid frame, not height above ground)
            and its leaf area (m²).
        """
        return _core.leaf_grid_profile(*self._args())

    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """Voxels that hold leaf area.

        Returns
        -------
        centres : numpy.ndarray
            ``(n, 3)`` voxel centres.
        area : numpy.ndarray
            Leaf area of each (m²).
        """
        return _core.leaf_grid_cells(*self._args())

    @classmethod
    def from_voxels(cls, grid, field: str = "pad_fpl") -> "LeafAreaGrid":
        """Leaf area density from a ray-traced grid.

        Parameters
        ----------
        grid
            A :class:`sylva.voxels.RayVoxelGrid`.
        field
            Density field; ``"lad_fpl"`` for leaf area when the grid was
            built with ``leaf_classes``, ``"pad_fpl"`` for plant area.

        Returns
        -------
        LeafAreaGrid
            NaN (unobserved) voxels become 0.
        """
        return cls(*_core.leaf_grid_from_voxels(grid._core, field))


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

    Parameters
    ----------
    leaf_points
        Leaf points (:class:`~sylva.PointCloud` or ``(N, 3)``).
    voxel_size
        Output grid resolution (m).
    res
        Thinning cube size (m); None picks 3.5 × the median spacing.
    k
        Neighbours for the normals.

    Returns
    -------
    LeafAreaGrid
        Aligned to multiples of ``voxel_size``. A lower bound on the true
        leaf area wherever foliage was occluded.
    """
    return LeafAreaGrid(*_core.leaf_area_density(_xyz(leaf_points), float(voxel_size), float(res or 0.0), int(k)))


def single_leaf_area(length: float, width: float, shape: "LeafShape | None" = None) -> float:
    """One-sided area of one leaf of the given size.

    Parameters
    ----------
    length, width
        Leaf length and width (m).
    shape
        Blade to measure; the built-in outline by default.

    Returns
    -------
    float
        Area (m²), 0.562 × length × width for the built-in outline.
    """
    if shape is None:
        return _core.single_leaf_area(float(length), float(width))
    return _core.single_leaf_area(float(length), float(width), shape.vertices, shape.faces)


def _unit_blade() -> tuple[np.ndarray, np.ndarray]:
    """The built-in blade as (vertices (V, 3), faces (F, 3)) in unit leaf space."""
    return _core.leaf_shape_check(None, None, 0.08, 0.04)


@dataclass(frozen=True)
class LeafShape:
    """The blade one leaf is cut from, and how big it is.

    The blade lives in unit leaf space — ``(along, across, up)`` with the base
    at the origin, the tip at ``along = 1`` and the greatest width 1 across —
    and is scaled by ``length`` along and by ``width`` across and up when
    placed. It may therefore be curled or made of several leaflets, not only
    a flat outline. Default is the built-in six-sided blade at 8 × 4 cm.

    Examples
    --------
    A bigger leaf of the built-in shape, and one scanned from life::

        shape = leaves.LeafShape(length=0.15, width=0.06)
        shape = leaves.LeafShape.from_obj("eucalypt_leaf.obj")  # keeps its own size
    """

    vertices: np.ndarray = None  #: (V, 3) unit leaf space (along, across, up)
    faces: np.ndarray = None  #: (F, 3) into ``vertices``
    length: float = 0.08  #: blade length (m)
    width: float = 0.04  #: greatest blade width (m)

    def __post_init__(self) -> None:
        v = None if self.vertices is None else np.ascontiguousarray(self.vertices, float)
        f = None if self.faces is None else np.ascontiguousarray(self.faces, np.uint32)
        if (v is not None and (v.ndim != 2 or v.shape[1] != 3)) or (f is not None and (f.ndim != 2 or f.shape[1] != 3)):
            raise ValueError("vertices must be (V, 3) and faces (F, 3)")
        v, f = _core.leaf_shape_check(v, f, float(self.length), float(self.width))
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)

    @property
    def area(self) -> float:
        """One-sided area of one leaf (m²): the sum of its triangles."""
        return _core.leaf_shape_area(self.vertices, self.faces, float(self.length), float(self.width))

    def resized(self, length: float | None = None, width: float | None = None) -> "LeafShape":
        """The same blade at a new size.

        Parameters
        ----------
        length, width
            New length and width (m); unchanged where None.

        Returns
        -------
        LeafShape
        """
        return LeafShape(self.vertices, self.faces, self.length if length is None else length,
                         self.width if width is None else width)

    def scaled_to(self, area: float) -> "LeafShape":
        """The same blade and aspect ratio, scaled to a one-sided area (m²).

        Parameters
        ----------
        area
            Wanted area of one leaf (m²).

        Returns
        -------
        LeafShape
        """
        return self.resized(*_core.leaf_shape_scaled(self.vertices, self.faces, float(self.length), float(self.width),
                                                     float(area)))

    @classmethod
    def from_mesh(cls, vertices, faces, length: float | None = None, width: float | None = None,
                  normalise: bool = True) -> "LeafShape":
        """A custom blade from a mesh of one leaf.

        The mesh is read in leaf space: the base at the smallest x, the tip
        along +x, the blade across ±y and any curl in z. With ``normalise``
        it is mapped into unit space and its own extent becomes the default
        size, so a leaf modelled in metres keeps the size it was drawn at.

        Parameters
        ----------
        vertices
            (V, 3) mesh vertices, or (V, 2) for a flat blade.
        faces
            (F, 3) triangles into ``vertices``.
        length, width
            Size to place the blade at (m); the mesh's own extent by default.
        normalise
            Map the mesh into unit leaf space. Pass False for vertices that
            are already in it.

        Returns
        -------
        LeafShape

        Raises
        ------
        ValueError
            If the mesh is empty or flat along the blade.
        """
        v = np.ascontiguousarray(vertices, float)
        if v.ndim == 2 and v.shape[1] == 2:
            v = np.column_stack([v, np.zeros(len(v))])
        if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
            raise ValueError("vertices must be (V, 3) or (V, 2)")
        f = np.ascontiguousarray(faces, np.uint32)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("vertices must be (V, 3) and faces (F, 3)")
        return cls(*_core.leaf_shape_from_mesh(v, f, None if length is None else float(length),
                                               None if width is None else float(width), bool(normalise)))

    @classmethod
    def from_obj(cls, path: str | Path, length: float | None = None, width: float | None = None,
                 normalise: bool = True) -> "LeafShape":
        """A custom blade from an OBJ file holding one leaf.

        Only ``v`` and ``f`` lines are read; polygons are fanned into
        triangles and texture and normal indices are ignored.

        Parameters
        ----------
        path
            OBJ file of a single leaf, oriented as in :meth:`from_mesh`.
        length, width
            Size to place the blade at (m); the mesh's own extent by default.
        normalise
            Map the mesh into unit leaf space.

        Returns
        -------
        LeafShape

        Raises
        ------
        ValueError
            If the file holds no faces.
        """
        return cls(*_core.leaf_shape_from_obj(str(path), None if length is None else float(length),
                                              None if width is None else float(width), bool(normalise)))


_DEFAULT = LeafShape()


def default_leaf() -> LeafShape:
    """The leaf :func:`add_leaves` uses when none is given.

    Returns
    -------
    LeafShape
    """
    return _DEFAULT


def set_default_leaf(shape: LeafShape | None = None, length: float | None = None,
                     width: float | None = None) -> LeafShape:
    """Set the leaf :func:`add_leaves` uses when none is given.

    Sets it for the process, so a site's leaf size or a scanned blade is
    chosen once rather than at every call::

        leaves.set_default_leaf(length=0.15, width=0.06)

    Parameters
    ----------
    shape
        Blade to make the default; the current default if None.
    length, width
        Size to set on it (m); unchanged where None.

    Returns
    -------
    LeafShape
        The previous default, to restore it later.
    """
    global _DEFAULT
    was = _DEFAULT
    _DEFAULT = (shape or _DEFAULT).resized(length, width)
    return was


@dataclass
class LeafMesh:
    """Leaf polygons from :func:`add_leaves`; ``len()`` is the leaf count."""

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
        """One-sided leaf area of all leaves (m²)."""
        return self.leaf_area * len(self)

    def to_obj(self, path: str | Path) -> None:
        """Write the leaves as one OBJ object named ``leaves``.

        Parameters
        ----------
        path
            Output file.
        """
        _core.write_obj(str(path), [(np.ascontiguousarray(self.vertices, float),
                                     np.ascontiguousarray(self.faces, np.uint32))], ["leaves"])


def add_leaves(model: QSM | None, leaf_area: LeafAreaGrid | float, angles: LeafAngleDistribution | str = "spherical",
               leaf_points=None, leaf_length: float | None = None, leaf_width: float | None = None,
               shape: LeafShape | None = None, max_branch_distance: float = 0.5, jitter: float = 0.01,
               seed: int = 1) -> LeafMesh:
    """Leaf polygons for a QSM.

    ``leaf_area`` is a :class:`LeafAreaGrid`, or a total (m2) to spread over
    ``leaf_points`` in proportion to the area they show. Each voxel receives
    leaves until its area is met, centred on leaf points of that voxel where
    there are any (uniformly inside it otherwise), with normals drawn from
    ``angles`` (a distribution or a de Wit type name) and a uniform azimuth;
    blades within ``max_branch_distance`` of a cylinder point away from it.
    Leaves may intersect: no collision test is made, unlike the insertion
    of Åkerblom et al. (2018).

    Parameters
    ----------
    model
        The tree's QSM, or None for leaves without wood.
    leaf_area
        Where and how much leaf area: a grid, or a total (m²).
    angles
        Leaf angle distribution, or a de Wit type name.
    leaf_points
        Leaf points to centre leaves on; required with a total area.
    leaf_length, leaf_width
        Leaf size (m); the shape's own size by default. Area per leaf is
        :attr:`LeafShape.area`.
    shape
        Blade to cut the leaves from, such as one scanned from life
        (:meth:`LeafShape.from_obj`); :func:`default_leaf` by default.
    max_branch_distance
        Leaves within this distance (m) of a cylinder are attached to it.
    jitter
        Random offset (m) of leaf centres.
    seed
        Random seed; the same inputs and seed give the same leaves.

    Returns
    -------
    LeafMesh

    Raises
    ------
    ValueError
        With a total area but no ``leaf_points``.
    """
    if isinstance(angles, str):
        angles = LeafAngleDistribution.from_type(angles)
    seeds = np.zeros((0, 3)) if leaf_points is None else _xyz(leaf_points)
    if isinstance(leaf_area, LeafAreaGrid):
        grid, total = leaf_area._args(), 0.0
    else:
        grid, total = (np.zeros(3), 0.0, None), float(leaf_area)
    cyl = np.zeros((0, 12)) if model is None else np.ascontiguousarray(model.cylinders, dtype=float)
    shape = (shape or _DEFAULT).resized(leaf_length, leaf_width)
    d = _core.add_leaves(*grid, total, seeds, np.ascontiguousarray(angles.bin_centres, dtype=float),
                         np.ascontiguousarray(angles.density, dtype=float), cyl, shape.vertices, shape.faces,
                         float(shape.length), float(shape.width), float(max_branch_distance), float(jitter), int(seed))
    return LeafMesh(**d)


def write_tree_obj(path: str | Path, model: QSM, leaf_mesh: LeafMesh, sides: int = 12,
                   contiguous: bool = False) -> None:
    """Write wood cylinders and leaves to one OBJ, as objects ``wood`` and ``leaves``.

    Parameters
    ----------
    path
        Output file.
    model
        The tree's QSM.
    leaf_mesh
        Leaves from :func:`add_leaves`.
    sides
        Facets around each cylinder.
    contiguous
        One continuous tube per branch (see :meth:`sylva.qsm.QSM.mesh`).
    """
    _core.write_tree_obj(str(path), np.ascontiguousarray(model.cylinders, dtype=float),
                         np.ascontiguousarray(leaf_mesh.vertices, float), np.ascontiguousarray(leaf_mesh.faces, np.uint32),
                         int(sides), bool(contiguous))
