# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Interpolation between point clouds and rasters.

Three directions are covered:

* :func:`transfer_attributes` carries attributes from one cloud to another
  (for example labels computed on a thinned cloud back to the full cloud);
* :func:`grid` interpolates points onto a :class:`~sylva.raster.Raster` by
  inverse distance weighting, a triangulated irregular network or
  natural-neighbour interpolation;
* :func:`sample_raster` and :func:`sample_rasters` read rasters at the
  points of a cloud.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral, Real

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster

__all__ = ["transfer_attributes", "grid", "sample_raster", "sample_rasters"]

_TRANSFER_METHODS = ("nearest", "idw", "majority")
_GRID_METHODS = ("idw", "tin", "natural")
_SAMPLE_METHODS = ("bilinear", "nearest")


def transfer_attributes(source: PointCloud, target: PointCloud, names: str | Iterable[str] | None = None,
                        method: str = "nearest", k: int = 8, power: float = 2.0,
                        max_distance: float | None = None, fill=None) -> PointCloud:
    """Interpolate attributes of one cloud onto the points of another.

    A k-d tree is built on ``source`` and every ``target`` point is
    processed independently and in parallel, so the target can hold hundreds
    of millions of points; the source is typically a thinned copy of it.
    Distances are 3D.

    Parameters
    ----------
    source
        Cloud carrying the attributes. Points with a non-finite coordinate
        are ignored.
    target
        Cloud that receives them.
    names
        Attribute name or names to transfer; every attribute of ``source``
        if None.
    method : {"nearest", "idw", "majority"}
        ``"nearest"`` copies the value of the nearest source point and works
        for any dtype. ``"idw"`` takes the inverse-distance weighted mean
        over the ``k`` nearest source points (weights ``1 / d**power``; a
        target that coincides with source points takes the mean of their
        values; non-finite source values are skipped); numeric attributes
        only. ``"majority"`` takes the most common value among the ``k``
        nearest source points, ties going to the value of the nearest one;
        use it for labels such as ``tree_id``, ``classification`` or a wood
        mask, which must not be averaged.
    k
        Neighbours considered by ``"idw"`` and ``"majority"``.
    power
        Inverse-distance exponent for ``"idw"`` (0 gives the plain mean).
    max_distance
        Only source points within this distance (m) of a target point are
        used. Target points with none keep no value and get ``fill``. None
        means no limit.
    fill
        Value for target points left without one (beyond ``max_distance``,
        non-finite coordinates, or an empty source). By default NaN for
        floating-point results (including every ``"idw"`` result) and -1
        for integer and boolean attributes.

    Returns
    -------
    PointCloud
        ``target`` with the transferred attributes added (replacing
        attributes of the same name); coordinates and other attributes are
        shared with ``target``. ``"nearest"`` and ``"majority"`` keep the
        source dtype, except that an unsigned or boolean attribute whose
        ``fill`` it cannot represent (for example -1 in ``uint8``) is widened
        to the smallest signed type holding both, whenever ``max_distance``
        is given or some point is left unfilled; pass a representable
        ``fill`` (``fill=0``) to keep the dtype. ``"idw"`` returns float32
        for float32 attributes and float64 otherwise.

    Raises
    ------
    ValueError
        For an unknown method or attribute name, ``"idw"`` on a
        non-numeric attribute, ``k < 1``, a negative ``power`` or a
        negative ``max_distance``.

    Examples
    --------
    Segment a thinned copy of a cloud, then label every original point:

    >>> thin = sylva.filters.voxel_downsample(cloud, 0.05)          # doctest: +SKIP
    >>> thin = thin.with_attrs(tree_id=labels)                       # doctest: +SKIP
    >>> full = interpolate.transfer_attributes(thin, cloud, "tree_id",
    ...                                        method="majority", k=5,
    ...                                        max_distance=0.1)     # doctest: +SKIP
    """
    if method not in _TRANSFER_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(map(repr, _TRANSFER_METHODS))}")
    if isinstance(names, str):
        names = [names]
    names = list(source.attrs) if names is None else list(names)
    missing = [n for n in names if n not in source.attrs]
    if missing:
        raise ValueError(f"source has no attribute(s) {missing}; it has {sorted(source.attrs)}")
    _check_k(k)
    _check_max_distance(max_distance)
    if not isinstance(power, Real) or not np.isfinite(power) or power < 0:
        raise ValueError(f"power must be a finite non-negative number, got {power!r}")
    if not names:
        return target.with_attrs()

    if method == "idw":
        cols = []
        for n in names:
            a = source.attrs[n]
            if a.dtype.kind not in "iuf":
                raise ValueError(f"attribute {n!r} has dtype {a.dtype}; 'idw' needs numbers, use 'nearest' or 'majority'")
            cols.append(np.ascontiguousarray(a, dtype=np.float64))
        outs = _core.interp_idw(source.xyz, target.xyz, cols, int(k), float(power), max_distance)
        new = {}
        for n, out in zip(names, outs):
            if fill is not None:
                out[np.isnan(out)] = fill
            new[n] = out.astype(np.float32) if source.attrs[n].dtype == np.float32 else out
        return target.with_attrs(**new)

    if method == "nearest":
        idx = _core.interp_nearest_indices(source.xyz, target.xyz, max_distance)
        indices = [idx] * len(names)
    else:
        indices = _core.interp_majority_indices(source.xyz, target.xyz,
                                                [_labels(source.attrs[n]) for n in names], int(k),
                                                max_distance)
    new = {}
    for n, idx in zip(names, indices):
        new[n] = _gather(source.attrs[n], idx, fill, max_distance is not None)
    return target.with_attrs(**new)


def _labels(values: np.ndarray) -> np.ndarray:
    """Numeric labels as they are; anything else (strings) as integer codes."""
    if values.dtype.kind in "iufb":
        return values
    return np.unique(values, return_inverse=True)[1].astype(np.int64)


def _check_k(k) -> None:
    if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")


def _check_max_distance(max_distance) -> None:
    if max_distance is not None and (not isinstance(max_distance, Real) or not max_distance >= 0):
        raise ValueError(f"max_distance must be None or a non-negative number, got {max_distance!r}")


def _gather(values: np.ndarray, idx: np.ndarray, fill, may_miss: bool) -> np.ndarray:
    """``values[idx]`` with ``fill`` where ``idx < 0``, widening the dtype if needed."""
    miss = idx < 0
    any_miss = bool(miss.any())
    if fill is None:
        fill = np.nan if values.dtype.kind in "fc" else -1
    dtype = values.dtype
    if values.dtype.kind in "iub" and (may_miss or any_miss) and not _holds(values.dtype, fill):
        dtype = _widen(values.dtype, fill)
    if len(values) == 0:
        return np.full(len(idx), fill, dtype=dtype)
    out = values[np.where(miss, 0, idx)].astype(dtype, copy=False)
    if any_miss:
        out[miss] = fill
    return out


def _holds(dtype: np.dtype, fill) -> bool:
    """Whether an integer or boolean ``dtype`` can store ``fill`` exactly."""
    try:
        f = float(fill)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(f) or f != int(f):
        return False
    if dtype.kind == "b":
        return f in (0.0, 1.0)
    info = np.iinfo(dtype)
    return info.min <= int(f) <= info.max


def _widen(dtype: np.dtype, fill) -> np.dtype:
    f = float(fill)
    if not np.isfinite(f) or f != int(f):
        return np.result_type(dtype, np.float64)
    return np.result_type(np.int8 if dtype.kind == "b" else dtype, np.min_scalar_type(int(f)))


def grid(cloud: PointCloud, resolution: float, value: str = "z", method: str = "idw", bounds=None,
         power: float = 2.0, k: int = 12, max_distance: float | None = None) -> Raster:
    """Interpolate point values onto a regular grid.

    Each cell takes the value interpolated at its centre from the points'
    x, y positions. Grid geometry follows :func:`sylva.ground.make_dtm`:
    row 0 is at ``ymin``; without ``bounds`` the south-west corner is the
    points' minimum snapped down to a multiple of ``resolution`` and the
    grid extends past their maximum.

    Parameters
    ----------
    cloud
        Points to interpolate, e.g. the ground points of a classified cloud.
        Points with a non-finite x, y or value are ignored, and points that
        share an x, y are merged into one carrying their mean value.
    resolution
        Cell size (m).
    value
        ``"z"`` or the name of a numeric attribute to interpolate.
    method : {"idw", "tin", "natural"}
        ``"idw"``: inverse distance weighting over the ``k`` nearest points
        in x, y (weights ``1 / d**power``); every cell gets a value unless
        ``max_distance`` is set, and values stay within the range of the
        data. ``"tin"``: linear interpolation on the Delaunay triangulation
        of the points, exact for planar data. ``"natural"``: natural-
        neighbour (Sibson) interpolation, smooth away from the data points
        and exact for linear functions. Both ``"tin"`` and ``"natural"``
        leave cells outside the convex hull of the points NaN.
    bounds
        ``(xmin, ymin, xmax, ymax)`` of the grid; the extent of the points
        if None.
    power
        Inverse-distance exponent (``"idw"`` only).
    k
        Neighbours per cell (``"idw"`` only).
    max_distance
        Cells whose centre is farther than this (m, in x, y) from every
        point are NaN, for every method; None means no limit.

    Returns
    -------
    Raster
        Interpolated values, NaN where there is none. ``crs`` is taken from
        the cloud when it has one.

    Raises
    ------
    ValueError
        For an unknown method or attribute, a non-positive ``resolution``,
        invalid ``bounds``, ``k < 1``, a negative ``power`` or
        ``max_distance``, or an empty cloud without ``bounds``.

    See Also
    --------
    sylva.ground.make_dtm : a DTM with these methods (``method="tin"``).

    Examples
    --------
    >>> ground = cloud[sylva.ground.ground_mask(cloud)]                   # doctest: +SKIP
    >>> dtm = interpolate.grid(ground, 0.25, method="tin")                # doctest: +SKIP
    >>> intensity = interpolate.grid(cloud, 0.5, value="intensity")      # doctest: +SKIP
    """
    if method not in _GRID_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(map(repr, _GRID_METHODS))}")
    if not isinstance(resolution, Real) or not resolution > 0 or not np.isfinite(resolution):
        raise ValueError(f"resolution must be a positive number, got {resolution!r}")
    if value == "z":
        values = cloud.z
    elif value in cloud.attrs:
        values = cloud.attrs[value]
        if values.dtype.kind not in "iufb":
            raise ValueError(f"attribute {value!r} has dtype {values.dtype}; grid needs numbers")
    else:
        raise ValueError(f"value must be 'z' or an attribute of the cloud; it has {sorted(cloud.attrs)}")
    _check_k(k)
    _check_max_distance(max_distance)
    if not isinstance(power, Real) or not np.isfinite(power) or power < 0:
        raise ValueError(f"power must be a finite non-negative number, got {power!r}")
    if bounds is not None:
        bounds = tuple(float(b) for b in bounds)
        if len(bounds) != 4:
            raise ValueError("bounds must be (xmin, ymin, xmax, ymax)")
    d = _core.interp_grid(cloud.xyz, np.ascontiguousarray(values, dtype=np.float64), float(resolution),
                          bounds, method, float(power), int(k), max_distance)
    return Raster(d["data"], d["xmin"], d["ymin"], d["resolution"], getattr(cloud, "crs", None))


def sample_raster(cloud: PointCloud, raster: Raster, name: str, method: str = "bilinear") -> PointCloud:
    """Add the value of a raster at each point as an attribute.

    Parameters
    ----------
    cloud
        Points in the raster's frame.
    raster
        Grid to read, e.g. a DTM, CHM or canopy cover map.
    name
        Name of the new attribute.
    method : {"bilinear", "nearest"}
        ``"bilinear"`` interpolates between the four surrounding cell centres
        (as :meth:`sylva.raster.Raster.sample`), holding the edge value in
        the outer half of the border cells; a NaN among the four cells gives
        NaN. ``"nearest"`` takes the value of the cell containing the point.

    Returns
    -------
    PointCloud
        The cloud with ``name`` added as float64. Points outside the
        raster's extent (``xmin <= x < xmax``, ``ymin <= y < ymax``) get NaN.

    Raises
    ------
    ValueError
        For an unknown method.

    See Also
    --------
    sylva.ground.normalize_height : height above a DTM, which extends the
        DTM's edge values beyond it instead of returning NaN.
    """
    if method not in _SAMPLE_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(map(repr, _SAMPLE_METHODS))}")
    return cloud.with_attrs(**{name: _sample(cloud, raster, method)})


def sample_rasters(cloud: PointCloud, rasters: Mapping[str, Raster], method: str = "bilinear") -> PointCloud:
    """Add the values of several rasters at each point as attributes.

    Parameters
    ----------
    cloud
        Points in the rasters' frame.
    rasters
        Attribute name to raster, e.g. ``{"ground": dtm, "canopy": chm}``.
        The rasters need not share a grid.
    method : {"bilinear", "nearest"}
        As for :func:`sample_raster`, applied to every raster.

    Returns
    -------
    PointCloud
        The cloud with one float64 attribute per raster, NaN outside it.

    Raises
    ------
    ValueError
        For an unknown method.
    """
    if method not in _SAMPLE_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(map(repr, _SAMPLE_METHODS))}")
    return cloud.with_attrs(**{n: _sample(cloud, r, method) for n, r in rasters.items()})


def _sample(cloud: PointCloud, raster: Raster, method: str) -> np.ndarray:
    if not isinstance(raster, Raster):
        raise ValueError(f"expected a sylva Raster, got {type(raster).__name__}")
    return _core.interp_sample_raster(raster.data, float(raster.xmin), float(raster.ymin),
                                      float(raster.resolution), cloud.xyz, method)
