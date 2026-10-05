# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ray-based canopy structure from airborne and UAV lidar.

The functions here are re-exported by :mod:`sylva.als`, and take either one
:class:`~sylva.PointCloud` or a whole :class:`~sylva.als.Catalog`, which
they process chunk by chunk with the engine of :mod:`sylva.als`:

1. **Trajectory.** :func:`read_trajectory` reads an SBET or a text table of
   time, position and attitude; :meth:`Trajectory.positions` interpolates
   the sensor position at any GPS time. Without one,
   :func:`estimate_trajectory` recovers an approximate trajectory from the
   pulses with several returns.
2. **Pulses.** :func:`pulses` groups returns into pulses by GPS time and
   gives :class:`~sylva.Shots` whose origin is the sensor, so that
   :func:`sylva.voxels.ray_voxelize` traces airborne pulses as it traces
   terrestrial ones.
3. **Canopy.** :func:`gap_profile` gives gap-fraction profiles, plant area
   density and gridded plant area index by the method of MacArthur & Horn
   (1969), corrected for each return's beam angle; :func:`ray_voxelize`
   traces the pulses of a whole catalogue through one voxel lattice.

Examples
--------
>>> from sylva import als
>>> traj = als.read_trajectory("flight.out", crs="EPSG:32755")        # doctest: +SKIP
>>> prof = als.gap_profile("tiles/", traj, resolution=20.0)            # doctest: +SKIP
>>> height, pad = prof.profile()                                       # doctest: +SKIP
>>> vox = als.ray_voxelize("tiles/", traj, voxel_size=2.0)             # doctest: +SKIP
>>> vox.pai(dtm=dtm).to_geotiff("pai.tif")                             # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster
from .trajectory import _as_trajectory, _catalog, _gap, _is_cloud


@dataclass
class ALSProfile:
    """Gap-fraction counts of an area, per grid cell and height layer.

    From :func:`gap_profile`. Layer 0 holds the returns below
    ``min_height`` (the pulses that got through the canopy), layers ``1``
    to ``nz`` the bins of ``bin_size`` above it, and the last layer the
    returns above the top bin. ``weight`` sums each return's share of its
    pulse, ``weight_k`` that share times its extinction per unit vertical
    plant area, ``G(θ) / cos θ`` at its beam zenith θ. Because these are
    counts, cells and chunks add up exactly; the densities are derived from
    them. It is plain data and pickles.

    Parameters
    ----------
    xmin, ymin
        South-west corner of the grid.
    resolution
        Cell size (m).
    min_height, bin_size
        Bottom of the first layer and layer thickness (m).
    weight, weight_k
        ``(nz + 2, ny, nx)``, row 0 at the south.
    crs
        CRS of the grid, if known.
    n_skipped
        Returns left out (no trajectory position, or beyond ``max_zenith``).
    """

    xmin: float
    ymin: float
    resolution: float
    min_height: float
    bin_size: float
    weight: np.ndarray
    weight_k: np.ndarray
    crs: str | None = None
    n_skipped: int = 0

    def __post_init__(self) -> None:
        self.weight = np.ascontiguousarray(self.weight, dtype=np.float64)
        self.weight_k = np.ascontiguousarray(self.weight_k, dtype=np.float64)
        w, wk = self.weight, self.weight_k
        if w.ndim != 3 or w.shape != wk.shape or w.shape[0] < 3:
            raise ValueError("weight and weight_k must be (nz + 2, ny, nx) arrays of one shape")

    @classmethod
    def _from_core(cls, d: dict, crs: str | None = None) -> ALSProfile:
        return cls(d["xmin"], d["ymin"], d["resolution"], d["min_height"], d["bin_size"],
                   d["weight"], d["weight_k"], crs, int(d["n_skipped"]))

    def __repr__(self) -> str:
        nz, ny, nx = self.shape
        return (f"ALSProfile({ny} x {nx} cells of {self.resolution:g} m, "
                f"{nz} layers of {self.bin_size:g} m)")

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(nz, ny, nx)``: layers and cells."""
        return (self.weight.shape[0] - 2, self.weight.shape[1], self.weight.shape[2])

    @property
    def heights(self) -> np.ndarray:
        """Bottom of each layer (m above ground)."""
        return self.min_height + self.bin_size * np.arange(self.shape[0])

    def _raster(self, data: np.ndarray) -> Raster:
        return Raster(data, self.xmin, self.ymin, self.resolution, self.crs)

    def _products(self):
        return _core.als_profile_products(self.weight, self.weight_k, float(self.bin_size))

    def pad(self) -> np.ndarray:
        """Plant area density of every layer and cell (m²/m³).

        Layer ``i`` lets through the weight below it out of the weight at
        or below it, ``T``; its density is ``-ln T / (k̄ dz)``, with ``k̄``
        the mean extinction of the returns at or below it (MacArthur & Horn
        1969, with the beam angle of each return). A layer nothing passed
        gets ``T = 0.5 / max(E, 1)`` (half a pulse of the ``E`` that
        entered), a lower bound; lidR's ``LAD()`` gives NA there.

        Returns
        -------
        numpy.ndarray
            ``(nz, ny, nx)``; NaN where no pulse reached a layer.
        """
        return self._products()[0]

    def pai(self) -> Raster:
        """Plant area index of each cell above ``min_height``.

        ``-ln P / k̄``, with ``P`` the share of the weight below
        ``min_height`` (the gap probability) and ``k̄`` the mean extinction
        of all returns; saturated cells (no return below) as in :meth:`pad`.
        With one beam angle this is the sum of the layer densities times
        ``bin_size``.

        Returns
        -------
        Raster
            NaN for cells without returns.
        """
        return self._raster(self._products()[1])

    def cover(self) -> Raster:
        """Canopy cover of each cell: ``1 - P`` at ``min_height``.

        Returns
        -------
        Raster
        """
        return self._raster(self._products()[2])

    def _mask(self, mask):
        if mask is None:
            return None
        m = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        if m.shape != self.shape[1:]:
            raise ValueError(f"mask must be {self.shape[1:]}, got {m.shape}")
        return m

    def profile(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        """Plant area density profile of an area, from its pooled counts.

        The counts of the cells are added before the densities are derived,
        so this is the profile of the area seen as one column: an effective
        profile, lower than the mean of the cells' profiles where the canopy
        is clumped between cells.

        Parameters
        ----------
        mask
            ``(ny, nx)`` booleans selecting the cells; all if None.

        Returns
        -------
        height : numpy.ndarray
            Bottom of each layer (m).
        pad : numpy.ndarray
            Plant area density (m²/m³); NaN above the highest return.
        """
        pad, _, _ = self._pooled(mask)
        return self.heights, pad

    def pgap(self, mask=None) -> tuple[np.ndarray, np.ndarray]:
        """Gap probability profile of an area: the share of the pooled
        weight below each layer boundary.

        Parameters
        ----------
        mask
            ``(ny, nx)`` booleans selecting the cells; all if None.

        Returns
        -------
        height : numpy.ndarray
            ``min_height`` and the top of every layer (m).
        pgap : numpy.ndarray
            Gap probability at each height (not corrected for beam angle).
        """
        _, pg, _ = self._pooled(mask)
        return self.min_height + self.bin_size * np.arange(len(pg)), pg

    def pooled_pai(self, mask=None) -> float:
        """Plant area index of an area from its pooled counts (see :meth:`profile`).

        Parameters
        ----------
        mask
            ``(ny, nx)`` booleans selecting the cells; all if None.

        Returns
        -------
        float
        """
        return float(self._pooled(mask)[2])

    def _pooled(self, mask):
        return _core.als_profile_pooled(self.weight, self.weight_k, float(self.bin_size),
                                        self._mask(mask))

    def metrics(self, strata: float = 5.0) -> dict[str, Raster]:
        """Summary metrics of each cell's profile, as rasters, for use
        beside the height metrics of :func:`sylva.als.grid_metrics` in a
        model of height, cover or biomass.

        - ``pulses``: the cell's weight (its number of pulses with the
          ``"equal"`` weighting); a check on how well the cell is sampled.
        - ``pai`` and ``cover``: as :meth:`pai` and :meth:`cover`.
        - ``fhd``: foliage height diversity, the Shannon index
          ``-sum(p ln p)`` of the shares ``p`` of the plant area in each
          layer (MacArthur & MacArthur 1961; GEDI L2B ``fhd_normal``). 0
          where there is no plant area.
        - ``pad_max`` and ``height_pad_max``: the density of the densest
          layer and the height of its middle.
        - ``height_pad_mean`` and ``height_pad_sd``: the mean and standard
          deviation of height weighted by plant area: where the canopy is
          and how deep it is.
        - For each stratum ``[a, b)`` of ``strata`` m from the ground
          (the strata of GEDI L2B with ``strata=5``):
          ``pavd_<a>_<b>``, its mean plant area density;
          ``pai_above_<h>``, the plant area index above ``h``, the bottom
          of the stratum's lowest layer; ``cover_above_<h>``, the canopy
          cover at ``h``.

        A layer belongs to the stratum its middle falls in, and strata
        below ``min_height`` are left out (so with ``min_height=1`` the
        lowest stratum is ``pavd_0_5`` over 1 to 5 m). Layers no pulse
        reached are left out of every sum. ``pai`` corrects each return for
        its own beam angle; the layer sums behind ``pai_above_<h>``
        correct each layer by the mean angle of the returns that reached
        it, so the two can differ slightly under a wide swath.

        Parameters
        ----------
        strata
            Thickness of the strata (m); at least ``bin_size``.

        Returns
        -------
        dict[str, Raster]
            One raster per metric, in a fixed order. Cells without returns
            are NaN, but for ``pulses`` (0).
        """
        names, values = _core.als_profile_metrics(self.weight, self.weight_k,
                                                  float(self.min_height), float(self.bin_size),
                                                  float(strata))
        return {n: self._raster(values[k]) for k, n in enumerate(names)}

    def area_metrics(self, mask=None, strata: float = 5.0) -> dict[str, float]:
        """The metrics of :meth:`metrics` for an area, from its pooled counts
        (the area seen as one column, as :meth:`profile`).

        Parameters
        ----------
        mask
            ``(ny, nx)`` booleans selecting the cells; all if None.
        strata
            Thickness of the strata (m).

        Returns
        -------
        dict[str, float]
        """
        m = self._mask(mask)
        cells = np.arange(self.shape[1] * self.shape[2]) if m is None else np.flatnonzero(m)
        names, values = _core.als_profile_area_metrics(
            self.weight, self.weight_k, float(self.min_height), float(self.bin_size), float(strata),
            [cells.tolist()])
        return {n: float(values[0, k]) for k, n in enumerate(names)}

    def plot_cells(self, plots, radius=None) -> list[np.ndarray]:
        """The cells of each plot: those whose centre lies inside it.

        Parameters
        ----------
        plots, radius
            As for :func:`sylva.als.plot_metrics`: ``(N, 2)`` centres with
            a radius, or polygons in any form :mod:`sylva.geo.masks` accepts.

        Returns
        -------
        list of numpy.ndarray
            Row-major cell indices (``row * nx + column``) of each plot.
        """
        from ..geo import masks

        _, ny, nx = self.shape
        res = self.resolution
        cx = self.xmin + (np.arange(nx) + 0.5) * res
        cy = self.ymin + (np.arange(ny) + 0.5) * res

        def cells_in(x0, y0, x1, y1, inside):
            c = np.flatnonzero((cx >= x0) & (cx <= x1))
            r = np.flatnonzero((cy >= y0) & (cy <= y1))
            if not (c.size and r.size):
                return np.zeros(0, np.int64)
            gx, gy = np.meshgrid(cx[c], cy[r])
            keep = inside(gx.ravel(), gy.ravel())
            return (r[:, None] * nx + c[None, :]).ravel()[keep].astype(np.int64)

        if radius is not None:
            centres = np.asarray(plots, dtype=np.float64)
            if centres.shape == (2,):
                centres = centres[None, :]
            if centres.ndim != 2 or centres.shape[1] != 2:
                raise ValueError("with a radius, plots must be (N, 2) centres, "
                                 f"got shape {centres.shape}")
            radii = np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(centres),))
            if not np.all(np.isfinite(radii) & (radii > 0)):
                raise ValueError("radius must be positive and finite")
            return [cells_in(x - r, y - r, x + r, y + r,
                             lambda px, py, x=x, y=y, r=r: (px - x) ** 2 + (py - y) ** 2 <= r * r)
                    for (x, y), r in zip(centres, radii, strict=True)]
        out = []
        for parts in masks._as_features(plots):
            ring = np.concatenate([p.exterior for p in parts])
            (x0, y0), (x1, y1) = ring.min(axis=0), ring.max(axis=0)
            feature = masks.MultiPolygon(parts)

            def inside(px, py, feature=feature):
                pts = PointCloud(np.column_stack([px, py, np.zeros_like(px)]))
                return masks.inside_polygons(pts, feature)

            out.append(cells_in(x0, y0, x1, y1, inside))
        return out

    def plot_metrics(self, plots, radius=None, ids=None, strata: float = 5.0):
        """The metrics of :meth:`metrics` for field plots, each from the
        pooled counts of the cells whose centre lies inside it, as a table
        to join with :func:`sylva.als.plot_metrics` and the plot data.

        The plot is represented by whole cells, so use a resolution well
        below the plot size (a 2 m grid for a 0.1 ha plot). Because the
        counts add up exactly, pooling small cells gives the profile of the
        plot itself. ``area`` gives the area of the cells used.

        Parameters
        ----------
        plots, radius
            As for :func:`sylva.als.plot_metrics`.
        ids
            Optional identifiers, one per plot.
        strata
            Thickness of the strata (m).

        Returns
        -------
        PlotMetrics
            Columns ``plot``, ``id`` (with ``ids``), ``area`` (m²), then
            the metrics. A plot covering no cell centre has ``area`` 0 and
            ``pulses`` 0, NaN elsewhere.
        """
        from .metrics import PlotMetrics

        cells = self.plot_cells(plots, radius)
        if ids is not None and len(ids) != len(cells):
            raise ValueError(f"{len(ids)} ids for {len(cells)} plots")
        names, values = _core.als_profile_area_metrics(
            self.weight, self.weight_k, float(self.min_height), float(self.bin_size), float(strata),
            [c.tolist() for c in cells])
        values = np.asarray(values).reshape(len(cells), len(names))
        columns = {"plot": np.arange(len(cells))}
        if ids is not None:
            columns["id"] = np.asarray(ids)
        columns["area"] = np.array([len(c) for c in cells], dtype=np.float64) * self.resolution ** 2
        columns.update({n: values[:, k] for k, n in enumerate(names)})
        return PlotMetrics(columns, ["area", *names])


def _angles(angles: str, trajectory) -> str:
    if angles == "auto":
        return "trajectory" if trajectory is not None else "scan_angle"
    if angles not in ("trajectory", "scan_angle", "none"):
        raise ValueError("angles must be 'auto', 'trajectory', 'scan_angle' or 'none', "
                         f"got {angles!r}")
    if angles == "trajectory" and trajectory is None:
        raise ValueError("angles='trajectory' needs a trajectory")
    return angles


def _cloud_heights(cloud: PointCloud, dtm, dtm_resolution: float) -> np.ndarray:
    from .. import ground
    if isinstance(dtm, Raster):
        return cloud.z - dtm.sample(cloud.x, cloud.y)
    if dtm is None:
        return np.asarray(cloud.z, dtype=np.float64)
    if dtm != "auto":
        raise ValueError(f"dtm must be 'auto', None or a Raster, got {dtm!r}")
    if "classification" not in cloud.attrs or int((cloud.attrs["classification"] == 2).sum()) < 3:
        raise ValueError("dtm='auto' needs ground points (classification 2); classify the "
                         "ground first, give a DTM, or dtm=None for normalised heights")
    d = ground.make_dtm(cloud[cloud.attrs["classification"] == 2], float(dtm_resolution))
    return cloud.z - d.fill_nearest().sample(cloud.x, cloud.y)


def gap_profile(source, trajectory=None, resolution: float = 10.0, bin_size: float = 1.0,
                min_height: float = 1.0, max_height: float | None = None,
                top_quantile: float = 0.99999, drop_noise: bool = True, weighting: str = "equal",
                angles: str = "auto", lad: str = "spherical", lad_params: Sequence[float] = (),
                g: float | None = None, max_zenith: float = 90.0, anchor: str = "ground",
                dtm="auto", dtm_resolution: float = 1.0,
                bounds=None, max_gap: float | None = None, time_offset: float = 0.0,
                chunk_size: float | None = None, buffer: float = 20.0,
                workers: int | None = None) -> ALSProfile:
    """Gap-fraction profiles, plant area density and plant area index.

    The returns of each grid cell are binned by height above ground. A
    pulse passes a height if it has weight below it, so the gap
    probability at height ``z`` is the share of the cell's weighted returns
    below ``z`` (Armston et al. 2013), and the transmittance of a layer the
    weight below it over the weight at or below it. Inverting Beer-Lambert
    layer by layer gives the plant area density profile of MacArthur & Horn
    (1969), which is lidR's ``LAD()`` (Bouvier et al. 2015) with
    ``weighting="all", angles="none", g=0.5``.

    **Scan angle.** A beam at zenith θ crosses ``dz / cos θ`` of a layer of
    thickness ``dz`` and meets foliage in proportion to ``G(θ)``, the
    projection of the leaf angle distribution, so each return carries the
    extinction ``k = G(θ) / cos θ`` per unit of vertical plant area, and a
    layer's density divides ``-ln T`` by the mean ``k`` of the returns that
    reached it. Without this, returns from the edge of a 30° swath read 15 %
    denser than at nadir. θ comes from the trajectory (the angle of the
    line from each return to the sensor), else from the LAS ``scan_angle``
    (which includes roll but not pitch).

    **Which cell.** An oblique pulse stopped in the canopy leaves its return
    to one side of where it would have reached the ground. Counted where
    they are, the returns of a cell come from pulses picked partly by their
    outcome, which biases the gap probability wherever the pulse density
    varies across the swath (by 5 % on the synthetic layer of the guide).
    With a trajectory each return is therefore counted, by default, in the
    cell where its beam meets the ground (``anchor="ground"``): the pulses
    of a cell are then chosen by their geometry alone. Without a trajectory
    the returns stay where they are.

    Every return is counted once, so a catalogue gives the same counts,
    cell for cell, whatever the chunks, as long as the buffer is wider than
    the drift of a beam between its returns and the ground (canopy height
    times the tangent of the scan angle); the buffer also serves the
    ``"auto"`` DTM.

    Parameters
    ----------
    source
        A :class:`~sylva.PointCloud`, or a catalogue (anything
        :func:`sylva.als.catalog` accepts).
    trajectory
        :class:`Trajectory` (or mapping, or file) for the beam angles.
    resolution
        Cell size (m). Cells should hold a few hundred pulses; 10-30 m is
        usual.
    bin_size
        Layer thickness (m).
    min_height
        Bottom of the lowest layer (m): returns below it, ground included,
        are pulses that went through.
    max_height
        Top of the highest layer; returns above it are intercepted above
        every layer. By default the layers stop where ``top_quantile`` of
        the weight above ``min_height`` lies below.
    top_quantile
        With ``max_height`` None, the top layers that together hold no more
        than ``1 - top_quantile`` of the weight above ``min_height`` are
        dropped (their returns count as above every layer), so that a few
        stray returns far above the canopy (birds, haze, a mast) do not
        stretch the profile with empty layers. The layers kept are as they
        would be without it, since a layer's transmittance depends only on
        the returns at and below it. 1 reaches the highest return.
    drop_noise
        Leave out returns classified as noise (7 or 18).
    weighting : {"equal", "first", "all"}
        Share of a pulse each return stands for: ``1 / number_of_returns``,
        first returns only (the original MacArthur-Horn), or one per return
        (as lidR's ``LAD()``).
    angles : {"auto", "trajectory", "scan_angle", "none"}
        Source of the beam zenith; ``"auto"`` uses the trajectory if given,
        else ``scan_angle``. ``"none"`` treats every beam as vertical.
    lad, lad_params
        Leaf angle distribution for ``G(θ)`` (see
        :func:`sylva.voxels.leaf_projection`).
    g
        A constant ``G`` instead of ``lad`` (lidR's ``k``, 0.5 by default
        there).
    max_zenith
        Returns whose beam is further than this from nadir (degrees) are
        left out.
    anchor : {"ground", "return"}
        Count each return in the cell where its beam meets the ground
        (needs the trajectory; see above), or where the return is.
    dtm : "auto", None or Raster
        Heights: ``"auto"`` makes a DTM from the ground points
        (classification 2; per chunk with its buffer for a catalogue), a
        :class:`~sylva.Raster` is subtracted, None uses z as it is
        (normalised tiles).
    dtm_resolution
        Cell size (m) of the ``"auto"`` DTM.
    bounds
        ``(xmin, ymin, xmax, ymax)`` of the grid for a point cloud; its
        extent by default. A catalogue uses the catalogue grid.
    max_gap, time_offset
        As for :func:`pulses`.
    chunk_size, buffer, workers
        As for :func:`sylva.als.apply`.

    Returns
    -------
    ALSProfile

    Raises
    ------
    ValueError
        For bad settings, a weighting whose attributes are missing, or
        angles that cannot be had.
    """
    ang = _angles(angles, trajectory)
    traj = None if trajectory is None else _as_trajectory(trajectory)
    if ang != "trajectory":
        traj = None
    mh = None if max_height is None else float(max_height)
    common = (float(resolution), float(min_height), float(bin_size), mh, float(top_quantile),
              bool(drop_noise), str(weighting), str(lad),
              [float(v) for v in lad_params], None if g is None else float(g), ang,
              None if traj is None else traj._core(), _gap(max_gap), float(time_offset),
              float(max_zenith), str(anchor))
    if _is_cloud(source):
        h = np.ascontiguousarray(_cloud_heights(source, dtm, dtm_resolution), dtype=np.float64)
        b = None if bounds is None else tuple(float(v) for v in bounds)
        d = _core.als_profile_cloud(source.xyz, source.attrs, h, b, *common)
        return ALSProfile._from_core(d)
    from .engine import _run_kw
    from .ops import _heights
    cat = _catalog(source)
    mode, raster = _heights(dtm)
    kw = _run_kw(chunk_size, buffer, workers)
    d = _core.als_profile_catalog(cat._core(), mode, raster, float(dtm_resolution), *common,
                                  kw["chunk_size"], kw["buffer"], kw["workers"])
    crs = cat.crs
    return ALSProfile._from_core(d, None if crs is None or crs.startswith("user-defined") else crs)


@dataclass
class ALSVoxels:
    """Ray-traced voxel fields of an airborne survey, from :func:`ray_voxelize`.

    The lattice is anchored at multiples of ``voxel_size``; every array is
    ``(nz, ny, nx)`` with row 0 at the south and layer 0 at the bottom.
    Columns no chunk traced are NaN. It is plain data and pickles.

    Parameters
    ----------
    origin
        ``(x, y, z)`` of the lattice's lower corner.
    voxel_size
        Voxel edge (m).
    fields
        The fields and metrics asked for, by name (see
        :class:`sylva.voxels.RayVoxelGrid`); ``num_beams`` is always there.
    reach
        The largest horizontal distance (m) a pulse covered between the top
        of the grid and its last return: the buffer that brings every pulse
        crossing a chunk into it.
    n_pulses
        Pulses traced (a pulse within reach of several chunks counts in each).
    crs
        CRS, if known.
    """

    origin: np.ndarray
    voxel_size: float
    fields: dict[str, np.ndarray]
    reach: float = 0.0
    n_pulses: int = 0
    crs: str | None = None

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)

    def __repr__(self) -> str:
        nz, ny, nx = self.shape
        return f"ALSVoxels({nx} x {ny} x {nz} at {self.voxel_size:g} m, fields={list(self.fields)})"

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(nz, ny, nx)``."""
        return next(iter(self.fields.values())).shape

    @property
    def z_levels(self) -> np.ndarray:
        """Bottom z of each layer."""
        return self.origin[2] + self.voxel_size * np.arange(self.shape[0])

    def _pad_name(self, name):
        if name is not None:
            if name not in self.fields:
                raise KeyError(f"no field {name!r}; the grid has {list(self.fields)}")
            return name
        for k in self.fields:
            if k.startswith("pad_"):
                return k
        raise KeyError("the grid has no pad_ field; name one")

    def _dtm(self, dtm):
        if dtm is None:
            return None
        return (np.ascontiguousarray(dtm.data, dtype=np.float64), float(dtm.xmin), float(dtm.ymin),
                float(dtm.resolution))

    def _beams(self, mask) -> np.ndarray:
        beams = np.ascontiguousarray(self.fields["num_beams"], dtype=np.float64)
        if mask is None:
            return beams
        m = np.asarray(mask, dtype=bool)
        if m.shape != self.shape[1:]:
            raise ValueError(f"mask must be {self.shape[1:]}, got {m.shape}")
        return np.where(m[None, :, :], beams, 0.0)

    def profile(self, name: str | None = None, dtm: Raster | None = None,
                bin_size: float | None = None, min_beams: float = 1.0,
                mask=None) -> tuple[np.ndarray, np.ndarray]:
        """Mean of a field per height above ground.

        Parameters
        ----------
        name
            Field; the first ``pad_`` field if None.
        dtm
            Terrain: heights are those of voxel centres above it. Without
            one, above the grid floor.
        bin_size
            Height bin (m); the voxel size if None.
        min_beams
            Voxels crossed by fewer pulses are left out.
        mask
            ``(ny, nx)`` booleans selecting the columns; all if None.

        Returns
        -------
        height : numpy.ndarray
            Bottom of each bin, from 0.
        mean : numpy.ndarray
            NaN for bins without voxels.
        """
        n = self._pad_name(name)
        bs = self.voxel_size if bin_size is None else float(bin_size)
        values = np.ascontiguousarray(self.fields[n], dtype=np.float64)
        return _core.als_voxel_height_profile(tuple(self.origin), float(self.voxel_size), values,
                                              self._beams(mask), self._dtm(dtm), bs,
                                              float(min_beams))

    def pai(self, name: str | None = None, dtm: Raster | None = None, min_height: float = 0.0,
            min_beams: float = 1.0) -> Raster:
        """Plant area index of each column: the sum of a density field times
        the voxel size.

        Parameters
        ----------
        name
            Density field; the first ``pad_`` field if None.
        dtm
            Terrain, for ``min_height``.
        min_height
            Voxels whose centre is lower above the DTM are left out (the
            ground itself, understorey).
        min_beams
            Voxels crossed by fewer pulses count as 0.

        Returns
        -------
        Raster
            ``(ny, nx)`` on the voxel columns; NaN where no voxel was seen.
        """
        n = self._pad_name(name)
        values = np.ascontiguousarray(self.fields[n], dtype=np.float64)
        s = _core.als_voxel_column_sums(tuple(self.origin), float(self.voxel_size), values,
                                        self._beams(None), self._dtm(dtm), float(min_height),
                                        float(min_beams))
        return Raster(s, float(self.origin[0]), float(self.origin[1]), float(self.voxel_size),
                      self.crs)


def ray_voxelize(source, trajectory, voxel_size: float = 1.0,
                 fields: Sequence[str] | None = None, *, dtm: Raster | None = None,
                 ground_class: int | None = 2, ground_distance: float = 0.2,
                 class_attr: str = "classification", leaf_classes: Sequence[int] = (),
                 wood_classes: Sequence[int] = (), tree_attr: str = "tree_id",
                 intensity_attr: str = "intensity", weighting: str = "equal",
                 attenuation: str | Sequence[str] = "fpl", laser: str | None = None,
                 beam: tuple[float, float] | None = None, lad: str = "spherical",
                 lad_params: Sequence[float] = (), average_leaf_area: float = 0.005,
                 occlusion: bool = False, unbounded_range: float = np.inf, bounds=None,
                 z_range=None,
                 max_gap: float | None = None, time_offset: float = 0.0, fill_missing: bool = False,
                 max_fill: int = 8, drop_incomplete: bool = False, chunk_size: float | None = None,
                 buffer: float = 20.0, workers: int | None = None) -> ALSVoxels:
    """Trace the pulses of a survey through one voxel lattice.

    The returns are grouped into pulses from the sensor as :func:`pulses`
    does, and traced by the ray tracer of :func:`sylva.voxels.ray_voxelize`
    along their actual beam directions, so every voxel's path lengths,
    hits and mean beam angle, and so its plant area density, account for the
    scan angle.

    On a catalogue, each chunk reconstructs the pulses of its core and
    buffer points and traces them through the columns whose centres lie in
    its core; the columns are put together into one lattice anchored at the
    catalogue's minimum corner snapped down to a multiple of
    ``voxel_size``. A pulse that crosses a chunk's core from outside it is
    traced there as long as one of its returns lies in the buffer. That
    needs a buffer at least the height of the grid above the pulse's last
    return times the tangent of its zenith (for a 30 m canopy and a 30°
    swath, 17 m); :attr:`ALSVoxels.reach` reports the largest such distance
    among the pulses traced, and a warning is given when it exceeds the
    buffer. With a sufficient buffer the result does not depend on the
    chunks, up to floating-point rounding.

    Use z as measured, not heights above ground: normalising moves each
    return by a different amount and bends the rays.

    Parameters
    ----------
    source
        A :class:`~sylva.PointCloud`, or a catalogue.
    trajectory
        :class:`Trajectory` (or mapping, or file).
    voxel_size
        Voxel edge (m). Memory is about 0.4 kB per voxel of a chunk while
        tracing, and 8 bytes per voxel and field of the result.
    fields
        Fields and metrics to keep (see :class:`sylva.voxels.RayVoxelGrid`);
        by default ``pad_<attenuation>``, ``transmittance``,
        ``num_hits_weighted`` and ``free_path_length``. ``num_beams`` is
        always added.
    dtm, ground_class, ground_distance, class_attr
        Echo labels, as for :func:`sylva.voxels.ray_voxelize`; by default
        returns of class 2 are ground (traced up to, never hits).
    leaf_classes, wood_classes, tree_attr, intensity_attr
        As for :func:`sylva.voxels.ray_voxelize`.
    weighting, attenuation, laser, beam, lad, lad_params
        As for :func:`sylva.voxels.ray_voxelize`.
    average_leaf_area, occlusion, unbounded_range
        As for :func:`sylva.voxels.ray_voxelize`.
    bounds
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` for a point cloud; its
        extent by default.
    z_range
        ``(zmin, zmax)`` of a catalogue's lattice; the header z range by
        default.
    max_gap, time_offset, fill_missing, max_fill, drop_incomplete
        As for :func:`pulses`.
    chunk_size, buffer, workers
        As for :func:`sylva.als.apply`. Chunks are sized to memory with
        their voxels counted.

    Returns
    -------
    ALSVoxels

    Raises
    ------
    ValueError
        For bad settings, a lattice too large for the memory budget, or
        returns outside the trajectory.

    Warns
    -----
    UserWarning
        When a catalogue's buffer is narrower than the pulses' reach.
    """
    from ..voxels import laser_spec
    methods = [attenuation] if isinstance(attenuation, str) else list(attenuation)
    if not methods:
        raise ValueError("give at least one attenuation method")
    if laser is not None:
        if beam is not None:
            raise ValueError("give laser or beam, not both")
        beam = laser_spec(laser)
    names = [f"pad_{methods[0]}", "transmittance", "num_hits_weighted", "free_path_length"] \
        if fields is None else [str(f) for f in fields]
    if "num_beams" not in names:
        names.append("num_beams")
    if int(max_fill) < 0:
        raise ValueError(f"max_fill must be non-negative, got {max_fill}")
    traj = _as_trajectory(trajectory)
    d = None if dtm is None else (np.ascontiguousarray(dtm.data, dtype=np.float64), float(dtm.xmin),
                                  float(dtm.ymin), float(dtm.resolution))
    kw = {"chunk_size": None, "buffer": 0.0, "workers": 0}
    crs = None
    if _is_cloud(source):
        src = (source.xyz, source.attrs)
    else:
        from .engine import _run_kw
        cat = _catalog(source)
        src = cat._core()
        kw = _run_kw(chunk_size, buffer, workers)
        crs = cat.crs
    b = None if bounds is None else tuple(float(v) for v in bounds)
    zr = None if z_range is None else (float(z_range[0]), float(z_range[1]))
    out = _core.als_ray_voxelize(
        src, traj._core(), names, b, zr, _gap(max_gap), float(time_offset), bool(fill_missing),
        int(max_fill), bool(drop_incomplete), float(voxel_size), d, class_attr,
        None if ground_class is None else int(ground_class), float(ground_distance),
        [int(c) for c in leaf_classes], [int(c) for c in wood_classes], tree_attr or "",
        intensity_attr, weighting, bool(occlusion),
        None if beam is None else (float(beam[0]), float(beam[1])), float(average_leaf_area), lad,
        [float(p) for p in lad_params], methods, float(unbounded_range), kw["chunk_size"],
        kw["buffer"], kw["workers"])
    if not _is_cloud(source) and out["reach"] > kw["buffer"]:
        warnings.warn(f"pulses reach {out['reach']:.1f} m sideways between the top of the grid and "
                      f"their last return, more than the {kw['buffer']:g} m buffer: pulses from "
                      "beyond a chunk's buffer may cross its voxels without being traced there (a "
                      "lower z_range top or a wider buffer avoids it)", stacklevel=2)
    if crs is not None and crs.startswith("user-defined"):
        crs = None
    return ALSVoxels(out["origin"], float(out["voxel_size"]), dict(out["fields"]),
                     float(out["reach"]), int(out["n_pulses"]), crs)
