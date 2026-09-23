# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Numpy-backed point cloud container."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np


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

    Raises
    ------
    ValueError
        If ``xyz`` is not ``(N, 3)`` or an attribute is not length ``N``.

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

    def __post_init__(self) -> None:
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
        return f"PointCloud(n={len(self):,}, attrs={sorted(self.attrs)})"

    def __getitem__(self, index) -> PointCloud:
        """Subset by boolean mask, integer indices or slice."""
        return PointCloud(self.xyz[index], {k: v[index] for k, v in self.attrs.items()})

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
        return PointCloud(self.xyz.copy(), {k: v.copy() for k, v in self.attrs.items()})

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
        return PointCloud(self.xyz, {**self.attrs, **attrs})

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
        return PointCloud(self.xyz, {k: v for k, v in self.attrs.items() if k not in names})

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
        xyz = self.xyz @ matrix[:3, :3].T + matrix[:3, 3]
        return PointCloud(xyz, dict(self.attrs))

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
            scan each point came from.

        Raises
        ------
        ValueError
            If ``clouds`` is empty.
        """
        clouds = list(clouds)
        if not clouds:
            raise ValueError("no clouds to concatenate")
        common = set.intersection(*(set(c.attrs) for c in clouds))
        return cls(
            np.vstack([c.xyz for c in clouds]),
            {k: np.concatenate([c.attrs[k] for c in clouds]) for k in sorted(common)},
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
