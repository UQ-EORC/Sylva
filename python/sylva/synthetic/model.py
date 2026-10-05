# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Realistic synthetic trees and plots, with their truth.

These are available from :mod:`sylva.synthetic` as :func:`tree_model` and
:func:`plot`. A tree is grown from an archetype with recursive branching,
allometric lengths and angles, pipe-model radii, a leaning and curving
stem, and leaves of known size and orientation; a plot mixes such trees
at a given stem density and diameter distribution on rough, sloping
terrain with understorey, grass, fallen logs and stumps. Every result
carries its truth: the wood as a cylinder table (a :class:`~sylva.qsm.QSM`),
the leaves as blades, DBH, height, crown and per-point labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _core
from ..leaves import LeafAngleDistribution, leaf_angle_distribution
from ..pointcloud import PointCloud
from ..qsm import QSM

__all__ = ["tree_model", "SyntheticTree", "plot", "Plot", "archetype", "scanner_preset",
           "ARCHETYPES", "LABELS", "SCANNERS"]

#: Archetypes of :func:`tree_model`.
ARCHETYPES = ("broadleaf", "conifer", "eucalypt", "savanna", "shrub")

#: Values of the ``label`` attribute of :func:`tree_model` and :func:`plot`.
LABELS = {1: "ground", 2: "stem", 3: "branch", 4: "leaf", 5: "understorey", 6: "dead wood"}

#: Scanner models of :func:`scanner_preset` and ``synthetic.scan(scanner=...)``.
SCANNERS = ("vz400", "vz400i", "vz2000i")

_DE_WIT = ("spherical", "uniform", "planophile", "erectophile", "plagiophile", "extremophile")


def _lad(lad) -> tuple[str | None, list[float]]:
    """``(name, params)`` for the core from a LAD given as a name or tuple."""
    if lad is None:
        return None, []
    if isinstance(lad, str):
        if lad not in _DE_WIT:
            raise ValueError(f"unknown leaf angle distribution {lad!r}; expected one of {_DE_WIT}, "
                             "('beta', mu, nu) or ('ellipsoidal', chi)")
        return lad, []
    lad = tuple(lad)
    if len(lad) == 3 and lad[0] == "beta":
        mu, nu = float(lad[1]), float(lad[2])
        if not (mu > 0 and nu > 0):
            raise ValueError(f"beta leaf angle parameters must be positive, got {mu} and {nu}")
        return "beta", [mu, nu]
    if len(lad) == 2 and lad[0] == "ellipsoidal":
        chi = float(lad[1])
        if not chi > 0:
            raise ValueError(f"the ellipsoidal parameter must be positive, got {chi}")
        return "ellipsoidal", [chi]
    raise ValueError(f"cannot read the leaf angle distribution {lad!r}")


def _positive(name: str, v) -> float:
    v = float(v)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a positive number, got {v}")
    return v


def _leaves(d: dict, prefix: str = "leaf_") -> dict:
    keys = ("centre", "normal", "axis", "length", "width", "area", "cylinder", "epicormic")
    return {k: d[prefix + k] for k in keys}


def archetype(name: str) -> dict:
    """The architectural constants of an archetype.

    Parameters
    ----------
    name
        One of :data:`ARCHETYPES`.

    Returns
    -------
    dict
        ``crown_base`` and ``leader`` (fractions of the height at which the
        crown starts and the stem ends), ``crown_radius`` (fraction of the
        height, an upper bound), ``crown_k`` (crown radius
        ``crown_k (0.6 + 12 dbh)`` m), ``lai`` (leaf area per crown area,
        an upper bound) and ``leaf_k`` (leaf area ``leaf_k dbh²``),
        ``allometry`` (``(a, b, c)`` of the height curve
        ``1.3 + a (1 - exp(-b D))^c``, D in cm), ``leaf_size`` (m),
        ``lad``, ``butt_swell``, ``lean`` (degrees) and ``sweep``.
    """
    return dict(_core.synthetic_archetype(str(name)))


@dataclass
class SyntheticTree:
    """A tree from :func:`tree_model` with its truth.

    Attributes
    ----------
    points
        The tree's points, with attributes ``classification`` (4 leaf, 5
        wood), ``label`` (2 stem, 3 branch, 4 leaf; see :data:`LABELS`),
        ``branch_order`` (0 stem, 1, 2, ...; for a leaf, the order of its
        twig), ``cylinder`` (row of :attr:`qsm`, -1 for leaves), ``leaf``
        (row of :attr:`leaves`, -1 for wood) and ``epicormic`` (1 on
        epicormic shoots and their leaves).
    qsm
        The wood as cylinders, the truth for a QSM: the points were drawn on
        exactly these cylinders (on the shaped section for the stem, whose
        radius is the area-equivalent one), so volumes and lengths are exact.
        ``n_points`` counts the points drawn on each.
    leaves
        One entry per leaf blade (an ellipse): ``centre``, ``normal`` (unit,
        pointing up), ``axis`` (unit, along the blade), ``length``,
        ``width``, ``area`` (``π L W / 4``), ``cylinder`` (the twig it hangs
        from) and ``epicormic``.
    archetype
        Name of the archetype.
    base
        Stem base.
    stem_bh
        The stem axis at breast height (1.3 m along the stem): where a
        leaning stem's DBH is measured.
    dbh
        Area-equivalent diameter of the stem at breast height (m).
    height
        Highest point of wood or leaf above the base (m).
    crown_base
        Height of the lowest first-order branch that is not epicormic (m).
    crown_area
        Area of the convex hull of the crown seen from above (m²).
    crown_extent
        ``(xmin, ymin, xmax, ymax)`` of the crown.
    leaf_area
        One-sided leaf area (m²), epicormic leaves included.
    epicormic_leaf_area
        Leaf area on epicormic shoots (m²).
    lad
        Name of the leaf angle distribution the normals were drawn from.
    cylinder_epicormic
        Per cylinder, whether it belongs to an epicormic shoot.
    """

    points: PointCloud
    qsm: QSM
    leaves: dict
    archetype: str
    base: np.ndarray
    stem_bh: np.ndarray
    dbh: float
    height: float
    crown_base: float
    crown_area: float
    crown_extent: tuple
    leaf_area: float
    epicormic_leaf_area: float
    lad: str
    cylinder_epicormic: np.ndarray

    @property
    def wood_volume(self) -> float:
        """Volume of the wood (m³), the sum of the cylinders."""
        return float(self.qsm.total_volume)

    def volume_by_order(self) -> dict:
        """Wood volume (m³) per branch order, from the cylinders.

        Returns
        -------
        dict
            ``{order: volume}``, order 0 being the stem.
        """
        order = self.qsm.column("branch_order").astype(int)
        v = self.qsm.volumes
        return {int(o): float(v[order == o].sum()) for o in np.unique(order)}

    def length_by_order(self) -> dict:
        """Axis length (m) per branch order, from the cylinders.

        Returns
        -------
        dict
            ``{order: length}``.
        """
        order = self.qsm.column("branch_order").astype(int)
        length = self.qsm.column("length")
        return {int(o): float(length[order == o].sum()) for o in np.unique(order)}

    def leaf_angles(self, n_bins: int = 18) -> LeafAngleDistribution:
        """The leaf angle distribution of the generated leaves, weighted by
        their area.

        Parameters
        ----------
        n_bins
            Bins over 0-90 degrees.

        Returns
        -------
        LeafAngleDistribution
            Whose :meth:`~sylva.leaves.LeafAngleDistribution.g` gives the
            leaf projection function G of the tree's leaves.
        """
        nz = np.clip(np.abs(self.leaves["normal"][:, 2]), 0.0, 1.0)
        return leaf_angle_distribution(np.arccos(nz), inclinations=True, n_bins=n_bins,
                                       weights=self.leaves["area"])


def tree_model(archetype: str = "broadleaf", x: float = 0.0, y: float = 0.0, z0: float = 0.0,
               dbh: float = 0.3, height: float = 15.0, *, crown_radius: float | None = None,
               crown_base: float | None = None, max_order: int = 3, lean: float | None = None,
               sweep: float | None = None, butt_swell: float | None = None, buttresses: int = 0,
               buttress_height: float = 1.0, buttress_extent: float = 0.5, ellipticity: float = 0.0,
               bark_depth: float = 0.0, leaf_area: float | None = None, leaf_size=None, lad=None,
               epicormic: float = 0.0, point_density: float = 2000.0, pipe_exponent: float = 2.0,
               min_radius: float = 0.0015, breast_height: float = 1.3,
               seed: int = 0) -> SyntheticTree:
    """A tree with a realistic architecture and its truth.

    The stem leans and curves smoothly, tapers, and swells at the butt; it
    can be elliptical, fissured and buttressed. First-order branches leave
    it from the crown base up (in whorls for the conifer) with a length set
    by the archetype's crown envelope; forking archetypes end the stem in
    codominant limbs. Each branch carries children up to ``max_order``,
    each a fixed fraction of its parent's length (shorter towards the tip),
    at the archetype's branching angle, bending up or drooping. Radii follow
    the pipe model: above the lowest branch every segment's cross-sectional
    area is proportional to the leaf area it carries (Shinozaki et al.
    1964), continuous with the stem taper below. Leaves are elliptical
    blades on the terminal twigs, with normals drawn from the leaf angle
    distribution. Points are drawn at ``point_density`` per m² on the wood
    surface and on one side of each leaf.

    Archetypes (:func:`archetype` lists their constants):

    ``"broadleaf"``
        crown from 35 % of the height, ellipsoidal, the stem forking into
        two limbs at 80 %; planophile leaves 10 × 5 cm.
    ``"conifer"``
        a central leader to the top with whorls of five branches from 25 %
        of the height, a conical crown, drooping branches; spherical
        distribution of 5 × 1.2 cm blades standing in for shoots.
    ``"eucalypt"``
        a clear bole to 55 % of the height, three ascending limbs from 65 %,
        a sparse crown (low leaf area) of pendulous (erectophile) 15 × 3 cm
        leaves, a noticeable lean.
    ``"savanna"``
        a short bole forking at 30 % of the height into three spreading
        limbs, a wide flat crown, small planophile leaves.
    ``"shrub"``
        several stems from near the ground; ``dbh`` is the stem diameter.

    Parameters
    ----------
    archetype
        One of :data:`ARCHETYPES`.
    x, y, z0
        Stem base.
    dbh
        Diameter at ``breast_height`` along the stem (m), honoured exactly
        (as the area-equivalent diameter) unless the crown starts below it.
    height
        Target height (m); the realised height is :attr:`SyntheticTree.height`.
    crown_radius
        Crown radius (m); by default ``min(crown_radius * height, crown_k
        (0.6 + 12 dbh))`` from :func:`archetype`.
    crown_base
        Height of the lowest branch (m); by default the archetype's.
    max_order
        Highest branch order, 1 to 4.
    lean
        Lean of the stem (degrees, towards a random azimuth); by default
        the archetype's.
    sweep
        Slope amplitude of the stem's smooth curvature (0 for straight).
    butt_swell
        Extra radius at the ground relative to the taper (e.g. 0.3).
    buttresses
        Number of buttress flanges (0 for none).
    buttress_height
        Height the flanges reach (m).
    buttress_extent
        Relative extra radius at the crest of a flange at the ground.
    ellipticity
        ``a / r - 1`` of the stem section; the area stays that of the
        round section.
    bark_depth
        Depth of bark fissures (m); the area stays that of the round section.
    leaf_area
        One-sided leaf area of the crown (m²); by default
        ``min(lai π R², leaf_k dbh²)`` from :func:`archetype`.
    leaf_size
        ``(length, width)`` of a leaf blade (m).
    lad
        Leaf angle distribution: a de Wit name (``"spherical"``,
        ``"planophile"``, ``"erectophile"``, ``"plagiophile"``,
        ``"extremophile"``, ``"uniform"``), ``("beta", mu, nu)`` with Goel
        and Strebel's (1984) parameters, or ``("ellipsoidal", chi)``
        (Campbell 1990); by default the archetype's.
    epicormic
        Epicormic shoots per metre of bole below the crown (0 for none), as
        on eucalypts resprouting after fire.
    point_density
        Points per m² of wood surface and of leaf.
    pipe_exponent
        Exponent ``e`` of the radius rule ``r ∝ W^(1/e)`` in the leaf area
        ``W`` carried (2 for the pipe model).
    min_radius
        Smallest twig radius (m).
    breast_height
        Distance along the stem at which ``dbh`` applies (m).
    seed
        Random seed; the same seed gives the same tree on every machine.

    Returns
    -------
    SyntheticTree

    Raises
    ------
    ValueError
        For an unknown archetype or leaf angle distribution, or a size that
        is not positive.

    Examples
    --------
    >>> t = synthetic.tree_model("eucalypt", dbh=0.5, height=30, epicormic=4, seed=1)
    >>> t.dbh, t.height, t.wood_volume, t.leaf_area
    >>> t.volume_by_order(), t.leaf_angles().mean_deg
    """
    name, params = _lad(lad)
    if leaf_size is not None:
        leaf_size = (float(leaf_size[0]), float(leaf_size[1]))
    opt = lambda v: None if v is None else float(v)  # noqa: E731
    d = _core.synthetic_tree_model(
        str(archetype), float(x), float(y), float(z0), float(dbh), float(height), opt(crown_radius),
        opt(crown_base), int(max_order), opt(lean), opt(sweep), opt(butt_swell), int(buttresses),
        float(buttress_height), float(buttress_extent), float(ellipticity), float(bark_depth),
        opt(leaf_area), leaf_size, name, params, float(epicormic), float(point_density),
        float(pipe_exponent), float(min_radius), float(breast_height), int(seed))
    return SyntheticTree(
        points=PointCloud(d["xyz"], d["attrs"]), qsm=QSM(d["cylinders"]), leaves=_leaves(d),
        archetype=d["archetype"], base=np.asarray(d["base"]), stem_bh=np.asarray(d["stem_bh"]),
        dbh=d["dbh"], height=d["height"], crown_base=d["crown_base"], crown_area=d["crown_area"],
        crown_extent=tuple(d["crown_extent"]), leaf_area=d["total_leaf_area"],
        epicormic_leaf_area=d["epicormic_leaf_area"], lad=d["lad"],
        cylinder_epicormic=d["cylinder_epicormic"])


@dataclass
class Plot:
    """A plot from :func:`plot` with its truth.

    Attributes
    ----------
    points
        All points, with ``classification`` (2 ground, 3 understorey, 4
        leaf, 5 wood and dead wood), ``label`` (1 ground, 2 stem, 3 branch,
        4 leaf, 5 understorey, 6 dead wood; :data:`LABELS`), ``tree_id`` (0
        off the trees, then 1.. by decreasing DBH), ``branch_order`` (-1 off
        the trees), ``cylinder`` and ``leaf`` (rows of :attr:`cylinders`
        and :attr:`leaves`, -1 elsewhere) and ``epicormic``.
    trees
        Truth per tree, a table of equal-length arrays: ``tree_id``,
        ``archetype``, ``x``, ``y``, ``z`` (stem base on the terrain),
        ``x_bh``, ``y_bh`` (the stem axis at 1.3 m, where a leaning stem is
        found), ``dbh``, ``height`` (above the terrain at the stem),
        ``crown_base``, ``crown_area``, ``wood_volume``, ``stem_volume``,
        ``branch_volume`` (m³) and ``leaf_area`` (m²).
    cylinders
        ``(n, 12)`` cylinder rows of all trees (:data:`sylva.qsm.COLUMNS`);
        ``parent`` refers to rows of the same tree. Use :meth:`qsm`.
    cylinder_tree
        Tree of each cylinder.
    leaves
        Leaf blades of all trees, as in :class:`SyntheticTree`, plus ``tree``.
    dead_wood
        Fallen logs and stumps: ``kind`` (``"log"`` or ``"stump"``),
        ``start``, ``axis``, ``length``, ``radius``; a stump starts 0.1 m
        below the ground.
    shrubs
        Understorey shrubs: ``x``, ``y``, ``height``, ``leaf_area``.
    grass
        Grass tufts: ``x``, ``y``, ``radius``.
    terrain
        ``slope``, ``aspect`` and ``modes``, for :meth:`ground_height`.
    size
        Side of the plot (m); trees stand in ``[0, size]²``.
    """

    points: PointCloud
    trees: dict
    cylinders: np.ndarray
    cylinder_tree: np.ndarray
    leaves: dict
    dead_wood: dict
    shrubs: dict
    grass: dict
    terrain: dict
    size: float

    def qsm(self, tree_id: int) -> QSM:
        """The cylinders of one tree.

        Parameters
        ----------
        tree_id
            From :attr:`trees`.

        Returns
        -------
        QSM
            With ``parent`` rows relative to that tree.

        Raises
        ------
        KeyError
            For a tree the plot does not have.
        """
        rows = self.cylinder_tree == int(tree_id)
        if not rows.any():
            raise KeyError(f"no tree {tree_id}")
        return QSM(self.cylinders[rows])

    def ground_height(self, x, y) -> np.ndarray:
        """The terrain's elevation.

        Parameters
        ----------
        x, y
            Coordinates (m).

        Returns
        -------
        numpy.ndarray
            Elevation of the plane and its micro-relief.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        t = self.terrain
        z = _core.synthetic_plot_terrain(np.ascontiguousarray(t["modes"]).reshape(-1, 4), float(t["slope"]),
                                         float(t["aspect"]), np.ascontiguousarray(x).ravel(),
                                         np.ascontiguousarray(y).ravel()).reshape(x.shape)
        return z[()] if z.ndim == 0 else z

    @property
    def stem_density(self) -> float:
        """Stems per hectare."""
        return len(self.trees["dbh"]) / (self.size * self.size / 1e4)

    @property
    def basal_area(self) -> float:
        """Basal area (m² ha⁻¹)."""
        return float(np.pi / 4 * np.sum(self.trees["dbh"] ** 2) / (self.size * self.size / 1e4))


def plot(size: float = 30.0, density: float = 600.0, dbh=("weibull", 1.8, 0.22),
         min_dbh: float = 0.07, archetypes="broadleaf", *, height_noise: float = 0.1,
         slope: float = 0.1, aspect: float = 90.0, roughness: float = 0.05,
         roughness_length: float = 2.0, shrubs: float = 400.0, grass_cover: float = 0.2,
         grass_height: float = 0.5, logs: float = 60.0, stumps: float = 40.0,
         ground_density: float = 400.0, margin: float = 2.0, point_density: float = 1000.0,
         max_order: int = 3, lad=None, epicormic: float = 0.0, seed: int = 0) -> Plot:
    """A forest plot of :func:`tree_model` trees, with its truth.

    ``round(density * size² / 10⁴)`` diameters are drawn from the
    distribution and truncated below at ``min_dbh``; each tree takes an
    archetype by weight, a height from the archetype's Chapman-Richards
    curve ``1.3 + a (1 - exp(-b D))^c`` (D in cm) times a lognormal scatter
    ``exp(N(0, height_noise))``, and the archetype's crown and leaf area
    allometries. Stems are placed uniformly at random, largest first, never
    closer than ``0.75 (D_i + D_j) + 0.2`` m. The terrain is a plane rising
    by ``slope`` per metre towards ``aspect`` plus a Gaussian random field
    (64 Fourier modes) of standard deviation ``roughness`` and correlation
    length ``roughness_length``. Shrubs (``shrub`` archetype, 0.5 to 2.5 m),
    grass tufts (15 cm radius, blades up to ``grass_height``), fallen logs
    (2 to 8 m long, 5 to 25 cm radius, lying on the ground) and stumps (0.2
    to 1 m tall, 10 to 30 cm radius) are scattered over the plot. Nothing is
    drawn below the terrain.

    Parameters
    ----------
    size
        Side of the square plot (m).
    density
        Stems per hectare.
    dbh
        Diameter distribution: ``("weibull", shape, scale)`` (scale in m),
        ``("reverse_j", mean)`` (a negative exponential above ``min_dbh``
        with mean excess ``mean`` m, de Liocourt's reverse-J), or a sequence
        of diameters (m), one tree each (``density`` is then ignored).
    min_dbh
        Smallest diameter (m).
    archetypes
        An archetype name, or ``{name: weight}``.
    height_noise
        Standard deviation of ``log(height)`` about the allometry.
    slope
        Rise per metre (e.g. 0.1 is 5.7 degrees).
    aspect
        Direction of steepest ascent (degrees clockwise from north, +y).
    roughness
        Standard deviation of the micro-relief (m).
    roughness_length
        Correlation length of the micro-relief (m).
    shrubs
        Shrubs per hectare.
    grass_cover
        Fraction of the plot covered by grass tufts.
    grass_height
        Tallest grass (m).
    logs, stumps
        Fallen logs and stumps per hectare.
    ground_density
        Terrain points per m², over the plot and ``margin`` beyond it.
    margin
        Terrain beyond the plot edge (m).
    point_density
        Points per m² of wood, leaf, understorey and dead wood.
    max_order
        Highest branch order of the trees (1 to 4).
    lad
        Leaf angle distribution of every tree, as in :func:`tree_model`;
        by default each archetype's.
    epicormic
        Epicormic shoots per metre of bole on every tree.
    seed
        Random seed. Trees are grown in parallel, each with its own seed, so
        the plot does not depend on the number of threads.

    Returns
    -------
    Plot

    Raises
    ------
    ValueError
        For unknown names, values out of range, or more stems than fit in
        the plot without overlap.

    Examples
    --------
    >>> p = synthetic.plot(size=20, density=800, archetypes={"eucalypt": 2, "broadleaf": 1})
    >>> p.trees["dbh"], p.stem_density, p.basal_area
    >>> p.points.attrs["label"]      # 1 ground ... 6 dead wood
    """
    if isinstance(dbh, (tuple, list)) and dbh and isinstance(dbh[0], str):
        kind, params = str(dbh[0]), [float(v) for v in dbh[1:]]
        if kind not in ("weibull", "reverse_j"):
            raise ValueError(f"unknown diameter distribution {kind!r}; expected 'weibull' or 'reverse_j'")
    else:
        params = [float(v) for v in np.asarray(dbh, dtype=float).ravel()]
        if not params:
            raise ValueError("dbh must be a distribution or a non-empty sequence of diameters")
        kind = "given"
    if isinstance(archetypes, str):
        archetypes = {archetypes: 1.0}
    arch = [(str(k), float(v)) for k, v in dict(archetypes).items()]
    name, lad_params = _lad(lad)
    d = _core.synthetic_plot(
        _positive("size", size), float(density), kind, params, float(min_dbh), arch, float(height_noise),
        float(slope), float(aspect), float(roughness), float(roughness_length), float(shrubs),
        float(grass_cover), float(grass_height), float(logs), float(stumps), float(ground_density),
        float(margin), float(point_density), int(max_order), name, lad_params, float(epicormic),
        int(seed))
    leaves = _leaves(d)
    leaves["tree"] = d["leaf_tree"]
    trees = dict(d["trees"])
    trees["archetype"] = np.asarray(trees["archetype"])
    dw = dict(d["dead_wood"])
    dw["kind"] = np.asarray(dw["kind"], dtype="<U5")
    sh = d["shrubs"]
    gr = d["grass"]
    return Plot(points=PointCloud(d["xyz"], d["attrs"]), trees=trees, cylinders=d["cylinders"],
                cylinder_tree=d["cylinder_tree"], leaves=leaves, dead_wood=dw,
                shrubs={"x": sh[:, 0], "y": sh[:, 1], "height": sh[:, 2], "leaf_area": sh[:, 3]},
                grass={"x": gr[:, 0], "y": gr[:, 1], "radius": gr[:, 2]},
                terrain={"slope": d["terrain_slope"], "aspect": d["terrain_aspect"],
                         "modes": d["terrain_modes"]},
                size=float(size))


def scanner_preset(name: str) -> dict:
    """Field of view and beam of a scanner model, as ``synthetic.scan``
    takes them.

    Parameters
    ----------
    name
        One of :data:`SCANNERS` (``"VZ-400"`` and similar spellings work).

    Returns
    -------
    dict
        ``min_zenith_deg`` and ``max_zenith_deg`` (the vertical field of
        view, +60 to -40 degrees for these), ``beam_divergence`` (mrad),
        ``exit_diameter`` (m, a nominal value) and ``range_noise`` (m, the
        ranging precision), from the manufacturer's data sheets. Check them
        against your instrument; the angular step is the scan's
        ``resolution_deg``.

    Raises
    ------
    ValueError
        For an unknown model.
    """
    return dict(_core.synthetic_scanner_preset(str(name)))
