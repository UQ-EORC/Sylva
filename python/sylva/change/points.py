# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Surface change between two epochs: point distances and rasters of difference.

* :func:`distances` compares two point clouds, either by the distance to the
  nearest point (cloud-to-cloud) or by M3C2 (Lague et al. 2013), which also
  gives each distance its 95 % level of detection.
* :func:`dod` subtracts two rasters (a DTM or CHM of difference) and marks
  the cells whose change exceeds the level of detection.

Both epochs must be in one frame (``align_epochs`` in this package registers them); the
registration error left after alignment enters the level of detection
through ``registration_sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster

__all__ = ["PointDistances", "distances", "DoD", "dod"]

_METHODS = ("c2c", "m3c2")


def _xyz(cloud, name: str) -> np.ndarray:
    xyz = cloud.xyz if isinstance(cloud, PointCloud) else np.asarray(cloud, dtype=np.float64)
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{name} must be a PointCloud or an (N, 3) array, got shape {xyz.shape}")
    return xyz


def _positive(name: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive number, got {v!r}")
    return float(v)


@dataclass
class PointDistances:
    """Distances between two epochs at a set of core points.

    Attributes
    ----------
    method : {"c2c", "m3c2"}
        How the distances were measured.
    core_points
        ``(N, 3)`` positions the distances refer to.
    distance
        ``(N,)`` distance (m). For ``"c2c"`` the unsigned distance to the
        nearest point of the reference epoch; for ``"m3c2"`` the signed
        distance from the reference to the compared surface along
        ``normal`` (positive where the surface moved the way the normal
        points). NaN where it could not be measured.
    lod
        ``(N,)`` 95 % level of detection (m), M3C2 only; NaN where either
        cylinder held fewer than ``min_points`` points.
    significant
        ``(N,)`` bool, ``abs(distance) > lod``; M3C2 only.
    normal
        ``(N, 3)`` unit normals used by M3C2.
    n_a, n_b
        Points of each epoch in the projection cylinder (M3C2).
    spread_a, spread_b
        Standard deviation (m, ``n - 1`` denominator) of each epoch's
        positions along the normal inside the cylinder: the local roughness
        plus the range noise (M3C2).
    """

    method: str
    core_points: np.ndarray
    distance: np.ndarray
    lod: np.ndarray | None = None
    significant: np.ndarray | None = None
    normal: np.ndarray | None = None
    n_a: np.ndarray | None = None
    n_b: np.ndarray | None = None
    spread_a: np.ndarray | None = None
    spread_b: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.distance)

    def __repr__(self) -> str:
        extra = ""
        if self.significant is not None:
            extra = f", significant={int(self.significant.sum()):,}"
        return f"PointDistances({self.method}, n={len(self):,}{extra})"

    def to_cloud(self) -> PointCloud:
        """The core points with the results as attributes.

        Returns
        -------
        PointCloud
            ``distance`` and, for M3C2, ``lod``, ``significant``, ``nx``,
            ``ny``, ``nz`` (the normal), ``n_a``, ``n_b``, ``spread_a`` and
            ``spread_b``, ready for :func:`sylva.write`.
        """
        attrs = {"distance": self.distance}
        if self.method == "m3c2":
            attrs.update(lod=self.lod, significant=self.significant, nx=self.normal[:, 0], ny=self.normal[:, 1],
                         nz=self.normal[:, 2], n_a=self.n_a, n_b=self.n_b, spread_a=self.spread_a,
                         spread_b=self.spread_b)
        return PointCloud(self.core_points, attrs)


def distances(a, b, method: str = "m3c2", core_points=None, *, normal_scale: float | None = None,
              projection_scale: float | None = None, max_depth: float | None = None,
              registration_sigma: float = 0.0, normals=None, orientation="up", min_points: int = 4,
              max_distance: float | None = None) -> PointDistances:
    """Distances between two epochs of a point cloud.

    ``"c2c"`` gives, for each point of ``b``, the distance to the nearest
    point of ``a``: quick, unsigned, and biased upwards by point spacing and
    noise, so it suits a first look rather than a test of change.

    ``"m3c2"`` is the Multiscale Model to Model Cloud Comparison of Lague,
    Brodu and Leroux (2013). At each core point a plane is fitted to the
    points of ``a`` within ``normal_scale / 2``; both clouds are then
    projected onto its normal inside a cylinder of diameter
    ``projection_scale`` reaching ``max_depth`` either way, and the distance
    is the difference of the two mean positions. Its 95 % level of
    detection is (their eq. 1)::

        LoD95 = 1.96 * (sqrt(s_a**2 / n_a + s_b**2 / n_b) + reg)

    where ``s`` is the spread of each cloud's positions along the normal, ``n`` their count and ``reg`` the registration error
    ``registration_sigma``. A distance is significant when it exceeds this
    level. Every core point is computed on its own in parallel (k-d trees on
    both clouds), so tens of millions of points are practical and the result
    does not depend on the thread count; use a spatially thinned copy of
    ``a`` as core points to save time.

    Parameters
    ----------
    a
        Reference (earlier) epoch, a :class:`~sylva.PointCloud` or an
        ``(N, 3)`` array. Points with a non-finite coordinate are ignored.
    b
        Compared (later) epoch, in the same frame.
    method : {"m3c2", "c2c"}
        Comparison.
    core_points
        Where distances are reported (cloud or ``(M, 3)`` array). By default
        the points of ``a`` for M3C2 and of ``b`` for C2C; C2C accepts no
        other core points.
    normal_scale
        M3C2: diameter ``D`` (m) of the neighbourhood the normal is fitted
        to. Lague et al. suggest about 20 to 25 times the roughness of the
        surface; for stems a scale below the stem diameter keeps the normal
        radial.
    projection_scale
        M3C2: diameter ``d`` (m) of the projection cylinder; large enough to
        hold at least about 30 points of each cloud.
    max_depth
        M3C2: how far along the normal, either way, points are searched (m);
        at least the largest change expected. Defaults to ``normal_scale``.
    registration_sigma
        M3C2: registration error (m, one standard deviation), for example
        ``EpochAlignment.registration_sigma``.
    normals
        M3C2: ``(M, 3)`` normals at the core points to use instead of
        fitting them (they are normalised, not re-oriented).
    orientation
        M3C2: sign of the fitted normals. ``"up"`` (+z; ground and other
        near-horizontal surfaces), a direction ``(dx, dy, dz)`` given as
        ``("direction", (dx, dy, dz))``, or a location such as the scanner
        position given as ``("towards", (x, y, z))``, which makes stem
        normals point outwards to it so that growth is positive.
    min_points
        M3C2: fewest points of each epoch in the cylinder for a level of
        detection (Lague et al. regard the estimate as reliable from about
        4); below it ``lod`` is NaN and the point is never significant,
        though a distance is still given.
    max_distance
        C2C: distances above this are NaN (m).

    Returns
    -------
    PointDistances

    Raises
    ------
    ValueError
        For an unknown method, missing or non-positive M3C2 scales, a
        negative ``registration_sigma``, an invalid ``orientation``,
        ``normals`` not matching the core points, or core points given to
        C2C.

    Examples
    --------
    >>> d = sylva.change.distances(epoch1, epoch2, "m3c2", core_points=core,
    ...                            normal_scale=0.5, projection_scale=0.3,
    ...                            max_depth=1.0, registration_sigma=0.005)  # doctest: +SKIP
    >>> d.distance[d.significant]                                           # doctest: +SKIP
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(map(repr, _METHODS))}")
    xa, xb = _xyz(a, "a"), _xyz(b, "b")
    if method == "c2c":
        if core_points is not None:
            raise ValueError("c2c measures from the points of b; core_points apply to m3c2 only")
        if max_distance is not None and (isinstance(max_distance, bool) or not isinstance(max_distance, Real)
                                         or np.isnan(max_distance) or max_distance < 0):
            raise ValueError(f"max_distance must be a non-negative number, got {max_distance!r}")
        d = _core.change_c2c(xa, xb, None if max_distance is None else float(max_distance))
        return PointDistances("c2c", xb, d)

    if normal_scale is None or projection_scale is None:
        raise ValueError("m3c2 needs normal_scale and projection_scale (diameters in m)")
    normal_scale = _positive("normal_scale", normal_scale)
    projection_scale = _positive("projection_scale", projection_scale)
    max_depth = normal_scale if max_depth is None else _positive("max_depth", max_depth)
    if (isinstance(registration_sigma, bool) or not isinstance(registration_sigma, Real)
            or not np.isfinite(registration_sigma) or registration_sigma < 0):
        raise ValueError(f"registration_sigma must be a finite non-negative number, got {registration_sigma!r}")
    if isinstance(min_points, bool) or not isinstance(min_points, Integral) or min_points < 1:
        raise ValueError(f"min_points must be a positive integer, got {min_points!r}")
    xc = xa if core_points is None else _xyz(core_points, "core_points")
    nv = None
    if normals is not None:
        nv = _xyz(normals, "normals")
        if len(nv) != len(xc):
            raise ValueError(f"normals has {len(nv)} rows for {len(xc)} core points")
    direction = towards = None
    if isinstance(orientation, str) and orientation == "up":
        direction = (0.0, 0.0, 1.0)
    elif (isinstance(orientation, tuple) and len(orientation) == 2 and orientation[0] in ("direction", "towards")):
        v = np.asarray(orientation[1], dtype=np.float64)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise ValueError(f"orientation {orientation[0]!r} needs three finite numbers, got {orientation[1]!r}")
        if orientation[0] == "direction":
            if not np.any(v):
                raise ValueError("orientation direction must not be zero")
            direction = tuple(v.tolist())
        else:
            towards = tuple(v.tolist())
    else:
        raise ValueError(f"orientation must be 'up', ('direction', (dx, dy, dz)) or ('towards', (x, y, z)), got {orientation!r}")
    r = _core.change_m3c2(xa, xb, xc, nv, normal_scale, projection_scale, max_depth, float(registration_sigma),
                          int(min_points), direction, towards)
    return PointDistances("m3c2", xc, r["distance"], r["lod"], r["significant"], r["normal"], r["n_a"], r["n_b"],
                          r["spread_a"], r["spread_b"])


@dataclass
class DoD:
    """A raster of difference with its level of detection.

    Attributes
    ----------
    difference
        ``raster_b - raster_a`` on the overlap of the two rasters (m); NaN
        where either is NaN.
    lod
        Level of detection per cell (m); NaN where it is unknown.
    significant
        ``(rows, cols)`` bool, ``abs(difference) > lod``.
    volume_gained, volume_lost
        Volume (m³) raised and lowered over the significant cells (both
        positive).
    net_volume
        ``volume_gained - volume_lost`` (m³).
    area_changed
        Area of the significant cells (m²).
    area_compared
        Area of the cells with a finite difference and level of detection
        (m²).
    """

    difference: Raster
    lod: Raster
    significant: np.ndarray
    volume_gained: float
    volume_lost: float
    net_volume: float
    area_changed: float
    area_compared: float

    def thresholded(self) -> Raster:
        """The difference where it is significant, 0 elsewhere and NaN where
        it could not be assessed.

        Returns
        -------
        Raster
        """
        d = self.difference.data
        out = np.where(self.significant, d, 0.0)
        out[~(np.isfinite(d) & np.isfinite(self.lod.data))] = np.nan
        return Raster(out, self.difference.xmin, self.difference.ymin, self.difference.resolution,
                      self.difference.crs)


def _cell_value(name: str, v, res: float):
    if v is None:
        return None
    if isinstance(v, Raster):
        return (v.data, float(v.xmin), float(v.ymin), float(v.resolution))
    if isinstance(v, bool) or not isinstance(v, Real) or np.isnan(v) or v < 0:
        raise ValueError(f"{name} must be a non-negative number or a Raster, got {v!r}")
    return float(v)


def dod(raster_a: Raster, raster_b: Raster, min_detectable=None, sigma_a=None, sigma_b=None) -> DoD:
    """Raster of difference between two epochs (a DTM or CHM of difference).

    The difference is ``raster_b - raster_a``. A cell's change is significant
    when it exceeds the level of detection, which is either given
    (``min_detectable``) or propagated from the standard deviations of the
    two surfaces as::

        LoD95 = 1.96 * sqrt(sigma_a**2 + sigma_b**2)

    (Brasington et al. 2003; Wheaton et al. 2010), which assumes the errors
    of the two surfaces are independent and normal. Include the vertical
    registration error in the sigmas, for example ``sigma_a =
    np.hypot(dtm_sigma, registration_sigma)``.

    Parameters
    ----------
    raster_a, raster_b
        Earlier and later surface, with the same resolution on the same
        lattice (corners a whole number of cells apart); the result covers
        their overlap.
    min_detectable
        Level of detection (m): a number, or a :class:`~sylva.Raster` on the
        same lattice.
    sigma_a, sigma_b
        Standard deviation of each surface (m): numbers or rasters on the
        same lattice. Give one and the other is taken as equal to it.

    Returns
    -------
    DoD

    Raises
    ------
    ValueError
        If the rasters differ in resolution or lattice or do not overlap, if
        neither or both of ``min_detectable`` and the sigmas are given, or if
        a threshold is negative.

    Examples
    --------
    >>> change = sylva.change.dod(chm_2020, chm_2024, sigma_a=0.15)   # doctest: +SKIP
    >>> change.net_volume, change.area_changed                         # doctest: +SKIP
    """
    for name, r in (("raster_a", raster_a), ("raster_b", raster_b)):
        if not isinstance(r, Raster):
            raise ValueError(f"{name} must be a Raster, got {type(r).__name__}")
    res = float(raster_a.resolution)
    m = _cell_value("min_detectable", min_detectable, res)
    sa = _cell_value("sigma_a", sigma_a, res)
    sb = _cell_value("sigma_b", sigma_b, res)
    d = _core.change_dod((raster_a.data, float(raster_a.xmin), float(raster_a.ymin), res),
                         (raster_b.data, float(raster_b.xmin), float(raster_b.ymin), float(raster_b.resolution)),
                         m, sa, sb)
    crs = raster_a.crs or raster_b.crs
    diff = Raster._from_core(d["difference"])
    lod = Raster._from_core(d["lod"])
    diff.crs = lod.crs = crs
    return DoD(diff, lod, d["significant"], d["volume_gained"], d["volume_lost"], d["net_volume"],
               d["area_changed"], d["area_compared"])
