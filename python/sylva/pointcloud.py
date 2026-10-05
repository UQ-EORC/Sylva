# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Numpy-backed point cloud container."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from . import _core


@dataclass
class PointCloud:
    """A set of 3D points with optional per-point attributes.

    Every function in Sylva that takes points takes a ``PointCloud`` and
    returns a new one; nothing is modified in place. Coordinates are metres
    in whatever frame the data came in (scanner, project or projected CRS).

    Parameters
    ----------
    xyz
        ``(N, 3)`` array of coordinates, stored as C-contiguous float64.
    attrs
        Mapping of attribute name to a length-``N`` array (``intensity``,
        ``classification``, ``height``, ...). Names follow laspy conventions
        so LAS files round-trip.
    crs
        Coordinate reference system of ``xyz``, or None when unknown: an
        EPSG code (``"EPSG:7855"``; an integer is turned into that form), a
        PROJ string or WKT. :func:`sylva.read` sets it from LAS/LAZ headers
        and :func:`sylva.write` stores it in LAS/LAZ files. It is carried
        through subsetting, :meth:`transform`, :meth:`translate`,
        :meth:`rotate`, :meth:`recentre` and :meth:`concatenate` unchanged,
        so after a local shift it describes the frame the offset returns
        to; :func:`sylva.geo.coords.reproject` changes it.

    Raises
    ------
    ValueError
        If ``xyz`` is not ``(N, 3)``, an attribute is not length ``N`` or
        ``crs`` is not a string, integer or None.

    Notes
    -----
    Attributes Sylva itself writes and reads:

    | Name | Written by | Meaning |
    |---|---|---|
    | ``classification`` | :func:`sylva.ground.classify_ground_csf` | ASPRS codes, 2 = ground |
    | ``height`` | :func:`sylva.ground.normalize_height` | height above the DTM (m) |
    | ``tree_id`` | you, from :func:`sylva.trees.segment_trees` | tree number, -1 unassigned |
    | ``scan_id`` | :func:`sylva.registration.merge_scans` | index of the source scan |
    | ``nx``, ``ny``, ``nz`` | ray-cloud PLY | vector from point to sensor |

    Indexing with a boolean mask, integer array or slice returns a subset
    with all attributes: ``cloud[cloud.z > 1.3]``.

    Examples
    --------
    >>> import numpy as np, sylva
    >>> cloud = sylva.PointCloud(np.random.rand(100, 3), {"intensity": np.ones(100)})
    >>> tall = cloud[cloud.z > 0.5]
    """

    xyz: np.ndarray
    attrs: dict[str, np.ndarray] = field(default_factory=dict)
    crs: str | None = None

    def __post_init__(self) -> None:
        if self.crs is not None:
            if isinstance(self.crs, (int, np.integer)) and not isinstance(self.crs, bool):
                self.crs = f"EPSG:{int(self.crs)}"
            elif not isinstance(self.crs, str):
                raise ValueError(f"crs must be a string, an EPSG code or None, got {type(self.crs).__name__}")
        self.xyz = np.ascontiguousarray(np.asarray(self.xyz, dtype=np.float64))
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")
        self.attrs = {k: np.ascontiguousarray(np.asarray(v)) for k, v in dict(self.attrs).items()}
        for name, values in self.attrs.items():
            if values.ndim != 1 or len(values) != len(self.xyz):
                raise ValueError(
                    f"attribute {name!r} has shape {values.shape}, expected ({len(self.xyz)},)"
                )

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        if self.crs is None:
            return f"PointCloud(n={len(self):,}, attrs={sorted(self.attrs)})"
        crs = self.crs if len(self.crs) <= 40 else self.crs[:37] + "..."
        return f"PointCloud(n={len(self):,}, attrs={sorted(self.attrs)}, crs={crs!r})"

    def __getitem__(self, index) -> PointCloud:
        """Subset by boolean mask, integer indices or slice."""
        return PointCloud(self.xyz[index], {k: v[index] for k, v in self.attrs.items()}, self.crs)

    @property
    def x(self) -> np.ndarray:
        """x coordinates, a view into ``xyz`` (length ``N``)."""
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        """y coordinates, a view into ``xyz`` (length ``N``)."""
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        """z coordinates, a view into ``xyz`` (length ``N``)."""
        return self.xyz[:, 2]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(min_xyz, max_xyz)``, two length-3 arrays."""
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def heights(self, attr: str = "height") -> np.ndarray:
        """Height above ground for each point.

        Parameters
        ----------
        attr
            Attribute holding normalised heights.

        Returns
        -------
        numpy.ndarray
            ``attr`` as float64 if the cloud has it, otherwise z. Falling back
            to z is only right for clouds that are already height-normalised
            (e.g. from :func:`sylva.ground.flatten`).
        """
        return np.asarray(self.attrs[attr], dtype=np.float64) if attr in self.attrs else self.z

    def copy(self) -> PointCloud:
        """Deep copy.

        Returns
        -------
        PointCloud
            A cloud whose coordinates and attribute arrays are independent
            of this one's.
        """
        return PointCloud(self.xyz.copy(), {k: v.copy() for k, v in self.attrs.items()}, self.crs)

    def with_attrs(self, **attrs: np.ndarray) -> PointCloud:
        """Add or replace attributes.

        Parameters
        ----------
        **attrs
            ``name=array`` pairs, each length ``N``.

        Returns
        -------
        PointCloud
            A new cloud sharing the coordinate array; existing attributes are
            kept unless replaced.
        """
        return PointCloud(self.xyz, {**self.attrs, **attrs}, self.crs)

    def without(self, *names: str) -> PointCloud:
        """Drop attributes.

        Parameters
        ----------
        *names
            Attribute names to remove; names that are absent are ignored.

        Returns
        -------
        PointCloud
            A new cloud sharing the coordinate array.
        """
        return PointCloud(self.xyz, {k: v for k, v in self.attrs.items() if k not in names}, self.crs)

    def where(self, expr: str) -> PointCloud:
        """Points satisfying an attribute expression.

        Parameters
        ----------
        expr
            A condition over the coordinates ``x``, ``y``, ``z`` and the
            attributes, e.g. ``"height > 2 & classification != 2"``; see
            :func:`sylva.geo.masks.expression` for the syntax.

        Returns
        -------
        PointCloud
            The points for which ``expr`` is true, with all attributes, in
            their original order.

        Raises
        ------
        ValueError
            On a syntax error or an unknown attribute; the message gives the
            position.
        """
        from .geo.masks import expression

        return self[expression(self, expr)]

    def transform(self, matrix: np.ndarray) -> PointCloud:
        """Apply a rigid or affine transform.

        Parameters
        ----------
        matrix
            ``(4, 4)`` homogeneous matrix acting on column vectors, e.g. a
            RIEGL SOP from :func:`sylva.io.read_matrix_file` or the result of
            :func:`sylva.registration.icp`.

        Returns
        -------
        PointCloud
            Transformed copy. Attributes are carried over unchanged, so
            direction-like attributes (``nx``, ``ny``, ``nz``) are *not*
            rotated.

        Raises
        ------
        ValueError
            If ``matrix`` is not 4x4.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        return PointCloud(_core.transform_xyz(self.xyz, matrix), dict(self.attrs), self.crs)

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> PointCloud:
        """Shift every point by a constant offset.

        Parameters
        ----------
        dx, dy, dz
            Offset in the units of the coordinates (m).

        Returns
        -------
        PointCloud
            Shifted copy with the same attributes and ``crs``. Each
            coordinate gets one addition, so shifting back by the negated
            offset returns the original to within one rounding, and exactly
            for whole-metre offsets of millimetre data.

        Raises
        ------
        ValueError
            If an offset is not finite.

        Examples
        --------
        >>> local = cloud.translate(-500_000, -6_900_000)
        """
        from .geo.coords import translation_matrix

        return PointCloud(_core.coords_apply(self.xyz, translation_matrix(dx, dy, dz)),
                          dict(self.attrs), self.crs)

    def rotate(self, angle_deg: float, axis: str | Iterable[float] = "z",
               about: Iterable[float] | None = None) -> PointCloud:
        """Rotate every point about an axis.

        Parameters
        ----------
        angle_deg
            Angle in degrees, positive counter-clockwise when looking down
            the axis towards the origin (right-handed): 90 about ``"z"``
            turns +x into +y.
        axis
            ``"x"``, ``"y"``, ``"z"`` or any 3-vector (normalised here).
        about
            A point ``(x, y, z)`` on the axis; the origin if None. Rotate
            about the plot centre, or :meth:`recentre` first, to keep
            projected coordinates from swinging far away.

        Returns
        -------
        PointCloud
            Rotated copy with the same attributes and ``crs``. Direction-like
            attributes (``nx``, ``ny``, ``nz``) are not rotated. Multiples
            of 90 degrees are exact.

        Raises
        ------
        ValueError
            If the angle, axis or centre is not finite, the axis is zero or
            an unknown name, or ``about`` does not have three values.

        Examples
        --------
        >>> turned = cloud.rotate(30, about=cloud.xyz.mean(axis=0))
        """
        from .geo.coords import rotation_matrix

        return PointCloud(_core.coords_apply(self.xyz, rotation_matrix(angle_deg, axis, about)),
                          dict(self.attrs), self.crs)

    def recentre(self, origin: Iterable[float] | None = None) -> tuple[PointCloud, np.ndarray]:
        """Move the coordinates close to zero.

        Projected coordinates in the millions of metres leave float32 (and
        many viewers and meshing tools) with centimetre precision; local
        coordinates keep it.

        Parameters
        ----------
        origin
            Point that becomes ``(0, 0, 0)``. By default the minimum corner
            of the (finite) points rounded down to whole metres, so the
            offset is exact and short.

        Returns
        -------
        cloud : PointCloud
            Shifted copy with the same attributes and ``crs``.
        offset : numpy.ndarray
            Length-3 offset that undoes the shift:
            ``cloud.translate(*offset)`` gives the original coordinates.

        Raises
        ------
        ValueError
            If ``origin`` is not three finite numbers.

        Examples
        --------
        >>> local, offset = cloud.recentre()
        >>> restored = local.translate(*offset)
        """
        if origin is None:
            o = np.asarray(_core.coords_recentre_origin(self.xyz), dtype=np.float64)
        else:
            o = np.asarray(origin, dtype=np.float64).reshape(-1)
            if o.shape != (3,) or not np.all(np.isfinite(o)):
                raise ValueError(f"origin must be three finite numbers, got {origin!r}")
        return self.translate(*(-o)), o

    @classmethod
    def concatenate(cls, clouds: Iterable[PointCloud]) -> PointCloud:
        """Stack several clouds into one.

        Parameters
        ----------
        clouds
            Clouds to merge, in order.

        Returns
        -------
        PointCloud
            All points; only attributes present in *every* input are kept.
            Use :func:`sylva.registration.merge_scans` to also record which
            scan each point came from. ``crs`` is the inputs' common CRS
            (clouds without one are taken to share it), or None.

        Raises
        ------
        ValueError
            If ``clouds`` is empty, or two clouds have different CRSs.
        """
        clouds = list(clouds)
        if not clouds:
            raise ValueError("no clouds to concatenate")
        common = set.intersection(*(set(c.attrs) for c in clouds))
        crss = list(dict.fromkeys(c.crs for c in clouds if c.crs is not None))
        if len(crss) > 1:
            from .geo.coords import same_crs

            crss = [c for i, c in enumerate(crss) if not any(same_crs(c, d) for d in crss[:i])]
        if len(crss) > 1:
            raise ValueError(f"cannot concatenate clouds in different CRSs: {crss}; "
                             "reproject them first (sylva.geo.coords.reproject)")
        return cls(
            np.vstack([c.xyz for c in clouds]),
            {k: np.concatenate([c.attrs[k] for c in clouds]) for k in sorted(common)},
            crss[0] if crss else None,
        )

    @classmethod
    def from_array(cls, array: np.ndarray, names: Mapping[int, str] | None = None) -> PointCloud:
        """Build a cloud from a 2D array.

        Parameters
        ----------
        array
            ``(N, 3 + k)`` array; the first three columns are x, y, z.
        names
            Column index to attribute name, e.g. ``{3: "intensity"}``.
            Unnamed extra columns are called ``col3``, ``col4``, ...

        Returns
        -------
        PointCloud
        """
        array = np.asarray(array)
        names = names or {}
        attrs = {names.get(i, f"col{i}"): array[:, i] for i in range(3, array.shape[1])}
        return cls(array[:, :3], attrs)
