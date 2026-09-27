# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
# see THIRD_PARTY_NOTICES.md.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ray-traced voxel grids with AMAPVox-style Beer-Lambert statistics.

A port of ``rayvoxel`` (J. Rivory, unpublished), a reimplementation of AMAPVox
on raycloudtools.
Every pulse is traced through the grid, and each voxel accumulates beam
counts, potential and free path lengths, beam sections and mean beam angles.
From these the attenuation coefficient is estimated by free path length
(``fpl``, bias corrected after Pimont et al. 2018), exact potential path length
(``ppl``), ``transmittance`` or Bailey & Mahaffee's (2017) eq. 10
(``bailey``), and divided by a projection function ``G`` to give plant, leaf
and wood area density. ``G`` comes from an analytic leaf angle distribution or,
with ``inclination=True``, from per-tree inclination angle distributions
estimated on the echoes (Vicari et al. 2019).

:func:`sylva.canopy.density_grid` is the lightweight alternative when only
hits and path lengths are needed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from . import _core
from .qsm import QSM
from .raster import Raster
from .shots import Shots

__all__ = ["RayVoxelGrid", "ray_voxelize", "laser_spec", "leaf_projection", "STATES", "tree_sampling",
           "EXCLUDED", "PLANT", "LEAF", "WOOD"]

#: Values of ``RayVoxelGrid.state``.
STATES = {"unobserved": 0, "occluded": 1, "empty": 2, "filled": 3}

#: Per-echo foliage codes.
EXCLUDED, PLANT, LEAF, WOOD = 0, 1, 2, 3


def laser_spec(name: str) -> tuple[float, float]:
    """Beam geometry of a scanner known to AMAPVox.

    Parameters
    ----------
    name
        Scanner name, e.g. ``"VZ-400"``, ``"LMS-Q780"``,
        ``"FARO-FOCUS-X330"``.

    Returns
    -------
    diameter, divergence : float
        Beam diameter at exit (m) and full divergence (rad).

    Raises
    ------
    ValueError
        For an unknown scanner; pass ``beam=(diameter, divergence)`` to
        :func:`ray_voxelize` instead.
    """
    spec = _core.laser_spec(name)
    if spec is None:
        raise ValueError(f"unknown laser {name!r}")
    return spec


def leaf_projection(theta, lad: str = "spherical", lad_params: Sequence[float] = ()) -> np.ndarray:
    """Leaf projection function G(θ) of an analytic leaf angle distribution.

    G is the mean projected leaf area per unit leaf area in the beam
    direction; area density is attenuation over G.

    Parameters
    ----------
    theta
        Beam zenith angle(s) in radians.
    lad
        ``spherical`` (G = 0.5 everywhere), ``uniform``, ``planophile``,
        ``erectophile``, ``plagiophile``, ``extremophile`` (de Wit 1965),
        ``ellipsoidal`` (Campbell 1990; ``lad_params=[chi]``) or
        ``twoParamBeta`` (Goel & Strebel 1984; ``[mu, nu]``).
    lad_params
        Parameters of the ellipsoidal or beta distribution.

    Returns
    -------
    numpy.ndarray
        G for each angle.

    Raises
    ------
    ValueError
        For an unknown distribution or wrong parameters.

    See Also
    --------
    sylva.leaves.LeafAngleDistribution.g : G from a measured distribution.
    """
    t = np.ascontiguousarray(np.atleast_1d(theta), dtype=np.float64)
    return _core.leaf_projection(t, lad, list(lad_params))


class RayVoxelGrid:
    """Voxel statistics from :func:`ray_voxelize`.

    The grid lives in the Rust core; arrays are copied out on first use and
    cached, shaped ``(nz, ny, nx)`` like :class:`sylva.canopy.DensityGrid`.
    Raw accumulators (:attr:`fields`) and derived quantities (:attr:`metrics`,
    plus ``attenuation_<m>``, ``pad_<m>``, ``lad_<m>``, ``wad_<m>`` for any
    attenuation method ``m``) are available by name or as attributes::

        grid["num_hits"]; grid.path_length; grid.pad_fpl; grid.transmittance

    Notes
    -----
    Main raw fields (sums over the pulses crossing each voxel):

    | Field | Meaning |
    |---|---|
    | ``num_beams`` | pulses entering the voxel |
    | ``num_hits`` | echoes in the voxel (``num_hit_leaf``, ``_wood``, ``_plant`` by class) |
    | ``num_hits_weighted`` | echoes weighted by their share of the pulse (``weighting``) |
    | ``num_beams_occluded`` | pulses reaching it only after their last echo (``occlusion=True``) |
    | ``path_length`` | potential path length: full chords of the entering pulses (m) |
    | ``free_path_length`` | path actually travelled inside the voxel (m) |
    | ``bs_entering``, ``bs_intercepted`` | beam section entering and stopped (m², needs ``beam``) |
    | ``wood_volume`` | QSM wood (m³) after :meth:`add_wood_volume` |

    Derived metrics:

    | Metric | Meaning |
    |---|---|
    | ``state`` | 0 unobserved, 1 occluded, 2 empty, 3 filled (:data:`STATES`) |
    | ``attenuation_<m>`` | attenuation λ (m⁻¹) by method ``m`` (see ``attenuation``) |
    | ``pad_<m>``, ``lad_<m>``, ``wad_<m>`` | plant, leaf, wood area density λ / G (m² m⁻³) |
    | ``pad_g0_5`` | plant area density with G = 0.5 |
    | ``pad_g_corrected`` | PAD with G at the voxel's mean beam zenith, first attenuation method |
    | ``surface_area`` | ``pad_g0_5`` × voxel volume (m²) |
    | ``transmittance`` | beam-section transmittance (0-1) |
    | ``mean_zenith_angle``, ``mean_azimuth_angle`` | mean beam direction (degrees) |
    | ``azimuth_concentration`` | 0 (all azimuths) to 1 (one azimuth) |
    | ``mean_laser_dist`` | mean distance from the scanner (m) |
    | ``sd_path_length`` | spread of the chord lengths (m) |
    | ``distance_from_ground`` | voxel centre above the DTM (m, needs ``dtm``) |
    | ``exploration_rate`` | share of sub-voxels crossed (``subvoxel_split``) |
    | ``g_plant``, ``g_leaf``, ``g_wood`` | G used per voxel |
    | ``wood_volume_density`` | QSM wood volume per voxel volume (m³ m⁻³) |
    """

    def __init__(self, core) -> None:
        self._core = core
        self._cache: dict[str, np.ndarray] = {}

    @property
    def origin(self) -> np.ndarray:
        """Minimum corner of the grid, ``(x, y, z)``."""
        return self._core.origin

    @property
    def voxel_size(self) -> float:
        """Voxel edge (m)."""
        return self._core.voxel_size

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid size as ``(nx, ny, nz)``; arrays are ``(nz, ny, nx)``."""
        return self._core.shape

    @property
    def fields(self) -> list[str]:
        """Names of the raw accumulators this grid holds (depends on the options)."""
        return self._core.field_names()

    @property
    def metrics(self) -> list[str]:
        """Names of the derived metrics; see the class notes."""
        return self._core.metric_names()

    def __repr__(self) -> str:
        nx, ny, nz = self.shape
        return f"RayVoxelGrid({nx}x{ny}x{nz} @ {self.voxel_size:g} m)"

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            get = self._core.field if name in self.fields else self._core.metric
            self._cache[name] = get(name)
        return self._cache[name]

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except ValueError as e:
            raise AttributeError(name) from e

    @property
    def observed(self) -> np.ndarray:
        """Boolean ``(nz, ny, nx)``: voxels crossed by at least one pulse before its last echo."""
        return self["state"] >= STATES["empty"]

    def occlusion_profile(self, min_height: float = 0.0, max_height: float | None = None) -> dict:
        """What the scan saw of the canopy space, layer by layer.

        The canopy space is every voxel from ``min_height`` above the ground
        (``distance_from_ground``, needs a DTM; else above the grid floor) up
        to ``max_height`` -- by default the highest layer holding a filled
        voxel. Returns per layer the ``height`` of its centre, the voxel
        count and the shares ``observed`` (a pulse went through or ended in
        it), ``occluded`` (only pulses already stopped reached it; needs
        ``occlusion=True``) and ``unobserved``, the mean pulses entering a
        voxel (``mean_beams``), and plot totals under ``"total"``.

        Parameters
        ----------
        min_height
            Bottom of the canopy space (m above ground).
        max_height
            Top of the canopy space; the highest filled voxel if None.

        Returns
        -------
        dict
            Arrays ``height``, ``n_voxels``, ``observed``, ``occluded``,
            ``unobserved``, ``mean_beams`` (one value per layer) and
            ``total``: ``{"observed", "occluded", "unobserved", "top"}``.
        """
        return self._core.occlusion_profile(float(min_height), None if max_height is None else float(max_height))

    def observed_map(self, min_height: float = 0.0, max_height: float | None = None) -> np.ndarray:
        """Map of how much of each column's canopy space was observed.

        Parameters
        ----------
        min_height, max_height
            Canopy space, as for :meth:`occlusion_profile`.

        Returns
        -------
        numpy.ndarray
            ``(ny, nx)`` share observed; NaN for columns with no canopy
            space. Useful to find the parts of a plot to rescan.
        """
        return self._core.observed_map(float(min_height), None if max_height is None else float(max_height))

    def z_levels(self) -> np.ndarray:
        """Bottom z of each voxel layer.

        Returns
        -------
        numpy.ndarray
            Length ``nz``, bottom first (m).
        """
        return self._core.z_levels()

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxel-centre coordinates.

        Returns
        -------
        X, Y, Z : numpy.ndarray
            Each ``(nz, ny, nx)``, aligned with the voxel arrays.
        """
        return self._core.centers()

    def profile(self, name: str = "pad_fpl", min_beams: int = 1) -> np.ndarray:
        """Mean of a metric per vertical layer over voxels crossed by at least
        ``min_beams`` pulses (unobserved voxels do not count as empty).
        Multiply by the voxel size and sum for a plant area index.

        Parameters
        ----------
        name
            Field or metric, e.g. ``"pad_fpl"`` or ``"pad_ppl"``.
        min_beams
            Minimum pulses for a voxel to count. Raise it (5-10) to keep
            poorly sampled voxels out of the mean.

        Returns
        -------
        numpy.ndarray
            One value per layer, bottom first; NaN where no voxel qualifies.
            Layers are in grid z, not height above ground.
        """
        return self._core.profile(name, float(min_beams))

    @property
    def tree_iad(self) -> dict[int, dict]:
        """Per-tree inclination angle distributions: normalised ``liad`` /
        ``wiad`` / ``piad`` histograms over ``bin_centres`` (rad), the
        angle-integrated ``g_leaf`` / ``g_wood`` / ``g_plant`` and the closest
        de Wit type of each. Empty unless ``inclination=True``.

        Returns
        -------
        dict
            ``{tree_id: {...}}``.
        """
        return self._core.tree_iad()

    def add_wood_volume(self, qsms: QSM | Iterable[QSM]) -> None:
        """Add QSM wood volume to the grid.

        Rasterises QSM cylinders into the ``wood_volume`` (m³) field and the
        ``wood_volume_density`` (m³ m⁻³) metric. Repeated calls add up.

        Parameters
        ----------
        qsms
            One QSM or several, in the grid's frame.
        """
        for q in [qsms] if isinstance(qsms, QSM) else qsms:
            self._core.add_wood_volume(q.cylinders)
        self._cache.pop("wood_volume", None)
        self._cache.pop("wood_volume_density", None)

    def to_dict(self, names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
        """Copy arrays out by name.

        Parameters
        ----------
        names
            Fields and metrics to include; every raw field if None.

        Returns
        -------
        dict
            ``{name: (nz, ny, nx) array}``, e.g. for ``np.savez``.
        """
        return {n: self[n] for n in (self.fields if names is None else names)}

    def write(self, path: str | Path, format: str | None = None, include_unobserved: bool = False,
              filled_only: bool = False) -> int:
        """Write an AMAPVox ``.vox`` file, or a space-delimited ``.txt`` table
        with voxel centres (``format`` ``"vox"`` / ``"text"``, by default from
        the extension). Observed and occluded voxels are written unless
        ``filled_only`` or ``include_unobserved``.

        Parameters
        ----------
        path
            Output file; overwritten.
        format : {"vox", "text"}, optional
            From the extension if None (``.vox`` is AMAPVox, anything else text).
        include_unobserved
            Also write voxels no pulse reached.
        filled_only
            Only write voxels holding echoes.

        Returns
        -------
        int
            Number of voxels written.
        """
        if format is None:
            format = "vox" if Path(path).suffix.lower() == ".vox" else "text"
        return self._core.write(str(path), format, include_unobserved, filled_only)

    def write_iad_csv(self, path: str | Path) -> None:
        """Write per-tree inclination distributions as CSV.

        Parameters
        ----------
        path
            Output file: one row per tree with de Wit types and inclination
            histograms. Needs ``inclination=True``.
        """
        self._core.write_iad_csv(str(path))


def _echo_codes(shots: Shots, attr: str) -> np.ndarray:
    if attr not in shots.echo_attrs:
        raise KeyError(f"shots have no {attr!r} echo attribute")
    return shots.echo_attrs[attr]


def ray_voxelize(
    shots: Shots | str | Path,
    voxel_size: float = 0.1,
    bounds=None,
    *,
    dtm: Raster | None = None,
    ground: np.ndarray | None = None,
    ground_class: int | None = None,
    ground_distance: float = 0.2,
    foliage: np.ndarray | None = None,
    leaf_classes: Sequence[int] = (),
    wood_classes: Sequence[int] = (),
    class_attr: str = "classification",
    tree_attr: str | None = "tree_id",
    intensity_attr: str = "intensity",
    weighting: str = "equal",
    attenuation: str | Sequence[str] = "fpl",
    laser: str | None = None,
    beam: tuple[float, float] | None = None,
    lad: str = "spherical",
    lad_params: Sequence[float] = (),
    inclination: bool = False,
    n_iad_bins: int = 18,
    knn_normal: int = 10,
    triangle_lmax: float = 0.05,
    occlusion: bool = False,
    flat_top: bool = False,
    neighbour_prior_min_rays: int = 0,
    subvoxel_split: int = 0,
    subvoxel_min_beams: int = 10,
    average_leaf_area: float = 0.005,
    unbounded_range: float = np.inf,
) -> RayVoxelGrid:
    """Trace ``shots`` through a voxel grid (see the module docstring).

    ``shots`` may be the path of a shots file (:meth:`sylva.Shots.save`). It
    is then streamed a few row groups at a time, so memory stays at the size
    of the grid plus a few million pulses; the grid defaults to the echo
    bounds recorded in the file, and echo labels must come from attributes
    (``ground_class`` / ``dtm``, ``leaf_classes`` / ``wood_classes``,
    ``tree_attr``) rather than from ``ground`` / ``foliage`` arrays.

    Parameters
    ----------
    shots
        Pulses in one frame (every scan of a plot together), or the path of
        a shots file. Must include the misses: from
        :meth:`sylva.Shots.fill_missing`, a ray cloud or a shots file made
        from either.
    voxel_size
        Voxel edge (m). 0.1-0.5 m is usual; memory is about 0.25 kB per
        voxel while tracing, so a 50 x 50 x 40 m plot at 0.1 m needs ~25 GB.
    bounds
        ``(min_xyz, max_xyz)``; by default the extent of the echoes. The max
        corner is snapped up to a whole number of voxels.
    dtm, ground, ground_class, ground_distance
        Ground echoes are traced up to but never count as hits. Give them as a
        boolean ``ground`` mask per echo, as a ``ground_class`` code of the
        ``class_attr`` echo attribute, or as echoes no more than
        ``ground_distance`` above the ``dtm`` (or below it). The DTM also stops
        occlusion rays below ground and gives ``distance_from_ground``.
    foliage, leaf_classes, wood_classes, class_attr
        Per-echo codes (:data:`EXCLUDED`, :data:`PLANT`, :data:`LEAF`,
        :data:`WOOD`), or the ``class_attr`` codes that mean leaf and wood
        (other codes below 3 are excluded, the rest are plant, as in
        rayvoxel). With neither, every non-ground echo is plant.
    tree_attr
        Echo attribute grouping echoes into trees for the inclination
        distributions; all echoes are pooled as tree 0 when it is missing.
    weighting
        Share of a pulse carried by each echo: ``equal`` (``1 / n``, as
        AMAPVox's ``EqualEchoWeight``), ``full`` (last),
        ``first``, ``relative`` or ``strongest`` (by ``intensity_attr``).
    attenuation
        One or more of ``fpl``, ``ppl``, ``transmittance``, ``bailey``; the
        first is used for ``pad_g_corrected``. ``ppl`` enables the exact solve;
        ``bailey`` needs leaf and wood echoes and turns on ``inclination``.
    laser, beam
        A scanner name for :func:`laser_spec`, or ``(exit diameter [m],
        divergence [rad])``. Enables the beam-section metrics (``bs_*``,
        ``transmittance``) and weights FPL / PPL by beam section.
    lad, lad_params
        Analytic leaf angle distribution, see :func:`leaf_projection`.
    inclination, n_iad_bins, knn_normal, triangle_lmax
        Estimate inclination angle distributions from echo normals (PCA over
        ``knn_normal`` neighbours); ``triangle_lmax`` caps Bailey facet edges.
    occlusion
        Also trace beyond each pulse's last echo (``num_beams_occluded``).
    flat_top
        Start the path in each column's top voxel at the highest echo.
    neighbour_prior_min_rays
        Top up voxels crossed by fewer weighted beams from their 26 neighbours.
    subvoxel_split, subvoxel_min_beams
        ``N`` (2-4) for an ``N³`` sub-voxel grid giving ``exploration_rate``.
    average_leaf_area
        Mean leaf area (m²) of the effective free path correction for leaves
        of finite size (Pimont et al. 2018, 2019); 0 disables.
    unbounded_range
        How far pulses without an echo are traced (default: to the grid edge).

    Returns
    -------
    RayVoxelGrid

    Raises
    ------
    ValueError
        For inconsistent options (``laser`` and ``beam``; arrays with a shots
        file; arrays of the wrong length; unknown methods).
    KeyError
        If a named echo attribute is missing.
    """
    methods = [attenuation] if isinstance(attenuation, str) else list(attenuation)
    if laser is not None:
        if beam is not None:
            raise ValueError("give laser or beam, not both")
        beam = laser_spec(laser)
    if isinstance(shots, (str, Path)):
        if ground is not None or foliage is not None:
            raise ValueError("ground / foliage arrays need in-memory shots; "
                             "label a shots file through its echo attributes")
        corners = None
        if bounds is not None:
            corners = (tuple(map(float, bounds[0])), tuple(map(float, bounds[1])))
        core = _core.ray_voxelize_file(
            str(shots), float(voxel_size), corners,
            None if dtm is None else (dtm.data, dtm.xmin, dtm.ymin, dtm.resolution),
            class_attr, ground_class, float(ground_distance), [int(c) for c in leaf_classes],
            [int(c) for c in wood_classes], tree_attr or "", intensity_attr,
            weighting, occlusion, flat_top, int(neighbour_prior_min_rays),
            None if beam is None else (float(beam[0]), float(beam[1])),
            int(subvoxel_split), int(subvoxel_min_beams), float(average_leaf_area), lad,
            [float(p) for p in lad_params], methods, inclination, int(n_iad_bins),
            int(knn_normal), float(triangle_lmax), float(unbounded_range),
        )
        return RayVoxelGrid(core)

    n = shots.n_echoes
    if ground is not None:
        ground = np.ascontiguousarray(ground, dtype=bool)
    elif ground_class is not None:
        _echo_codes(shots, class_attr)
    if foliage is not None:
        foliage = np.ascontiguousarray(foliage, dtype=np.uint8)
    elif len(leaf_classes) or len(wood_classes):
        _echo_codes(shots, class_attr)
    if weighting in ("relative", "strongest"):
        _echo_codes(shots, intensity_attr)
    for name, a in (("ground", ground), ("foliage", foliage)):
        if a is not None and len(a) != n:
            raise ValueError(f"{name} has {len(a)} values for {n} echoes")

    # Echo labels not given as arrays come from the attributes, as for a file.
    core = _core.ray_voxelize(
        shots._to_core(), float(voxel_size),
        None if bounds is None else (tuple(map(float, bounds[0])), tuple(map(float, bounds[1]))),
        ground, foliage, None if dtm is None else (dtm.data, dtm.xmin, dtm.ymin, dtm.resolution),
        class_attr, ground_class, float(ground_distance), [int(c) for c in leaf_classes],
        [int(c) for c in wood_classes], tree_attr or "", intensity_attr,
        weighting, occlusion, flat_top, int(neighbour_prior_min_rays),
        None if beam is None else (float(beam[0]), float(beam[1])),
        int(subvoxel_split), int(subvoxel_min_beams), float(average_leaf_area), lad,
        [float(p) for p in lad_params], methods, inclination, int(n_iad_bins), int(knn_normal),
        float(triangle_lmax), float(unbounded_range),
    )
    return RayVoxelGrid(core)


def tree_sampling(grid: RayVoxelGrid, cloud, labels, min_beams: float = 10.0,
                  above: float = 2.0) -> dict[str, np.ndarray]:
    """How well each tree was seen, from a ray-traced ``grid`` built with
    ``occlusion=True`` (so occluded voxels are told from unreached ones).

    A tree's crown envelope is, voxel layer by voxel layer, the convex hull
    of its points (``labels`` per point, negative ignored). That envelope is
    observed almost by construction -- the parts of a crown nobody saw left
    no points -- so the informative columns are the pulses that reached it
    and whether its top is real:

    ``above_observed_fraction``
        share of the voxels up to ``above`` metres over the tree's highest
        point, within the footprint of its top metre, that were observed
        (empty, or filled by a neighbour). Low means the top may be hidden
        and the tree taller than its points.
    ``median_beams``, ``p10_beams``, ``beams_by_quarter``
        pulses entering an envelope voxel (unseen voxels count 0): overall,
        the 10th percentile, and the median per quarter of the envelope's
        height, bottom first.
    ``well_sampled_fraction``
        observed envelope voxels with at least ``min_beams`` pulses.

    Also ``tree_id``, ``n_voxels`` and ``volume`` of the envelope and its
    ``observed_fraction``, ``occluded_fraction``, ``unobserved_fraction``.

    Parameters
    ----------
    grid
        Grid from :func:`ray_voxelize` with ``occlusion=True``, covering the
        trees.
    cloud
        A :class:`~sylva.PointCloud` or ``(N, 3)`` array in the grid's frame.
    labels
        Tree id per point (e.g. from :func:`sylva.trees.segment_trees`).
    min_beams
        Pulses for a voxel to count as well sampled.
    above
        Height (m) above each tree's top that is checked.

    Returns
    -------
    dict
        The columns above as arrays, one row per tree; ready for
        ``pandas.DataFrame``.
    """
    xyz = np.ascontiguousarray(cloud.xyz if hasattr(cloud, "xyz") else cloud, dtype=float)
    lab = np.ascontiguousarray(labels, dtype=np.int64)
    return grid._core.tree_sampling(xyz, lab, float(min_beams), float(above))
